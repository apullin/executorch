# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse the decomposed RoPE chain into a single `espressif::rope` custom op.

Rotary position embedding lowers to eight edge ops per application:

    x_rot, x_pass = x[..., :rot], x[..., rot:]      2x slice_copy
    x1, x2        = x_rot.chunk(2, -1)              split_with_sizes_copy
    rotated       = cat([-x2, x1], -1)              neg, cat
    out           = x_rot * cos + rotated * sin     mul, mul, add
    result        = cat([out, x_pass], -1)          cat

Each writes a full-width temporary. Measured on a speech encoder: 96 ops
across 12 applications, **34% of all portable output traffic** -- the largest
single fusable target measured, and general, because every RoPE model pays
it, the story model's prefill included.

WHY A CUSTOM OP AND NOT A DELEGATE EPILOGUE. The delegate already applies clamp,
gelu and residual adds in registers between accumulate and store, and rotation
mixes lanes only WITHIN one output row, so a q/k projection epilogue is possible
and would cost zero extra traffic instead of one read and one write. But:

  * it needs the row index to equal the position, which holds for prefill and
    NOT for a decode step, where the position is a runtime input -- so it would
    be a prefill-only optimisation with a second path still needed,
  * it needs a blob version bump, a cos/sin table in the descriptor, and new
    delegate arithmetic, which means device parity rather than blob identity,
  * and it needs THIS pass anyway. The matcher below is the hard part of both
    designs; the epilogue adds a numerics change on top of it.

Matching is structural and every link is checked -- op identity, operand
identity, the split being an exact half, the two halves belonging to the same
slice, and no intermediate escaping to a consumer outside the pattern. A site
failing any check is left completely alone and recorded in `rejected` with the
reason, so a partial rewrite is not representable. `expect=` turns a miscount
into an exception.

Handles both rotary widths: partial (`rot < head_dim`, which has the trailing
cat) and full (`rot == head_dim`, which does not).
"""

from typing import List, Optional

import torch
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult
from torch.library import Library

# ---- op declaration ---------------------------------------------------------
# FRAGMENT, not DEF, so this module can coexist with the others that open the
# "espressif" namespace (fuse_rmsnorm_pass.py, for one).
lib = Library("espressif", "FRAGMENT")
lib.define("rope(Tensor x, Tensor cos, Tensor sin, int rot_dim) -> Tensor")
lib.define("rope.out(Tensor x, Tensor cos, Tensor sin, int rot_dim, *, "
           "Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
          rot_dim: int) -> torch.Tensor:
    """The canonical form, which BOTH matched spellings reduce to exactly.

    The op serves both table layouts without assuming duplicated values.
    HF-style tables are [T, rot], so each output half reads its corresponding
    table half. This repo's pairwise spelling builds [T, rot/2], in which case
    the one table half is deliberately shared by both output halves.

    Bit-exactness against both source spellings is not an approximation:
    `x1*c - x2*s` and `x1*cos + (-x2)*sin` are the same IEEE operations, and the
    second half likewise.

    The kernel is not this arithmetic: ops/espressif_ops.cpp shim_rope rounds
    each output once, in the order GCC's contraction chose on the chip
    (fma(-x2, s, x1*c), fma(x2, c, x1*s)), and spells it with std::fma so the
    host runs it too. test/reference.py rope_f32_device mirrors the kernel;
    this Composite (the source's two roundings) is 1 ulp off it in about a
    quarter of outputs.
    """
    half = rot_dim // 2
    x_rot = x[..., :rot_dim]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    c0, s0 = cos[..., :half], sin[..., :half]
    if cos.shape[-1] == rot_dim:
        c1, s1 = cos[..., half:rot_dim], sin[..., half:rot_dim]
    else:
        c1, s1 = c0, s0
    out = torch.cat([x1 * c0 - x2 * s0, x2 * c1 + x1 * s1], dim=-1)
    if rot_dim == x.shape[-1]:
        return out
    return torch.cat([out, x[..., rot_dim:]], dim=-1)


def _rope_out(x, cos, sin, rot_dim, *, out):
    out.copy_(_rope(x, cos, sin, rot_dim))
    return out


impl.impl("rope", _rope, "CompositeExplicitAutograd")
impl.impl("rope.out", _rope_out, "CompositeExplicitAutograd")

ROPE = torch.ops.espressif.rope.default


# ---- structural matching ----------------------------------------------------
def _opname(target) -> str:
    name = getattr(target, "_name", None)
    if name:
        return name
    name = getattr(target, "name", None)
    if callable(name):
        return name()
    return str(target)


def _is(node, *names) -> bool:
    """Name match, tolerant of the overload suffix.

    The edge dialect reports `aten::split_with_sizes_copy` where aten reports
    `aten::split_with_sizes_copy.default`, and matching only the fully qualified
    form would silently match NOTHING -- the pass would fuse 0 sites and every
    rejection would blame the wrong link. Comparing the base name too is what
    makes one matcher work against both dialects.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return False
    got = _opname(node.target)
    if got in names:
        return True
    base = got.split(".")[0]
    return any(base == n.split(".")[0] for n in names)


MUL = ("aten::mul.Tensor",)
ADD = ("aten::add.Tensor",)
NEG = ("aten::neg.default",)
CAT = ("aten::cat.default",)
SLICE = ("aten::slice_copy.Tensor",)
SPLIT = ("aten::split_with_sizes_copy.default",)


class _Reject(Exception):
    """A site that looked like RoPE but is not. Carries the failed check."""


def _getitem_src(node):
    """Resolve `getitem(split, i)` to `(split_node, i)`, else None."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and "getitem" in str(node.target)):
        return None
    src, idx = node.args[0], node.args[1]
    return (src, idx) if isinstance(idx, int) else None


def _static_int(value, what):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _Reject(f"{what} is not a static integer: {value}") from None


def _unit_alpha(node) -> bool:
    """add/sub with alpha 1 (keyword or third positional), i.e. a plain sum."""
    alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
    return alpha == 1


def _joins_last_axis(cat) -> bool:
    """Whether `cat` concatenates along the last axis of its output, which is
    the only axis espressif::rope joins anything on."""
    out = cat.meta.get("val")
    if out is None:
        return False
    dim = _static_int(cat.args[1] if len(cat.args) > 1
                      else cat.kwargs.get("dim", 0), "cat dimension")
    return (dim + out.dim() if dim < 0 else dim) == out.dim() - 1


def _last_axis_slice(node):
    """(source, start, end, width) for a unit-step slice of the last axis.

    `end` is clamped to the width, as slicing does. Any other slice -- another
    axis, a stride -- raises _Reject.
    """
    source = node.args[0]
    xv = source.meta.get("val")
    if xv is None:
        raise _Reject("no shape metadata on the slice source")
    width = _static_int(xv.shape[-1], "slice width")
    dim = _static_int(node.args[1] if len(node.args) > 1 else 0,
                      "slice dimension")
    if dim < 0:
        dim += xv.dim()
    if dim != xv.dim() - 1:
        raise _Reject(f"slice dimension {dim} is not the last axis")
    start = _static_int(node.args[2] if len(node.args) > 2 and node.args[2] is not None
                        else 0, "slice start")
    end_arg = node.args[3] if len(node.args) > 3 else None
    end = width if end_arg is None else min(_static_int(end_arg, "slice end"), width)
    step = _static_int(node.args[4] if len(node.args) > 4 else 1, "slice step")
    if step != 1:
        raise _Reject(f"slice step is {step}, not 1")
    return source, start, end, width


def _halves(node):
    """Return (source, half_index, split_carrier) for one exact half.

    Export uses two equivalent representations in real models: getitem from a
    split-with-sizes, or explicit last-axis slices. The latter is the current
    Transformers/Muse-Glimmer `rotate_half` graph.
    """
    got = _getitem_src(node)
    if got is not None:
        split, idx = got
        if not _is(split, *SPLIT):
            raise _Reject(f"{split} is not a split")
        source = split.args[0]
        xv = source.meta.get("val")
        if xv is None:
            raise _Reject("no shape metadata on the split source")
        width = _static_int(xv.shape[-1], "split width")
        if width % 2:
            raise _Reject(f"split width {width} is odd")
        sizes = split.args[1]
        dim = _static_int(split.args[2] if len(split.args) > 2 else 0,
                          "split dimension")
        if dim < 0:
            dim += xv.dim()
        if dim != xv.dim() - 1:
            raise _Reject(f"split dimension {dim} is not the last axis")
        if list(sizes) != [width // 2, width // 2]:
            raise _Reject(f"split sizes are {list(sizes)}, not exact halves")
        return source, idx, split

    if not _is(node, *SLICE):
        raise _Reject(f"{node} is neither a chunk output nor a slice")
    source, start, end, width = _last_axis_slice(node)
    if width % 2:
        raise _Reject(f"slice width {width} is odd")
    bounds = (start, end)
    if bounds == (0, width // 2):
        idx = 0
    elif bounds == (width // 2, width):
        idx = 1
    else:
        raise _Reject(f"slice bounds are {bounds}, not exact halves of {width}")
    return source, idx, None


def _validate_tables(x, cos, sin, rot_dim):
    """Check the fp32 broadcast layout implemented by the device shim."""
    xv = x.meta.get("val")
    cv = cos.meta.get("val") if isinstance(cos, torch.fx.Node) else None
    sv = sin.meta.get("val") if isinstance(sin, torch.fx.Node) else None
    if xv is None or cv is None or sv is None:
        raise _Reject("x/cos/sin have no shape metadata")
    if xv.dtype != torch.float32 or cv.dtype != torch.float32 or sv.dtype != torch.float32:
        raise _Reject(f"rope is fp32-only, got {xv.dtype}/{cv.dtype}/{sv.dtype}")
    if tuple(cv.shape) != tuple(sv.shape):
        raise _Reject(f"cos shape {tuple(cv.shape)} != sin shape {tuple(sv.shape)}")
    if xv.dim() < 2 or cv.dim() < 1:
        raise _Reject("rope needs a position axis and a table width")
    width = _static_int(xv.shape[-1], "input width")
    if not 0 < rot_dim <= width or rot_dim % 2:
        raise _Reject(f"invalid rotary width {rot_dim} for input width {width}")
    cs_w = _static_int(cv.shape[-1], "table width")
    if cs_w not in (rot_dim // 2, rot_dim):
        raise _Reject(f"table width {cs_w} is not {rot_dim // 2} or {rot_dim}")
    if any(_static_int(d, "table leading dimension") != 1 for d in cv.shape[:-2]):
        raise _Reject(f"table leading dimensions are not singleton: {tuple(cv.shape)}")
    table_t = _static_int(cv.shape[-2], "table positions") if cv.dim() >= 2 else 1
    input_t = _static_int(xv.shape[-2], "input positions")
    if table_t not in (1, input_t):
        raise _Reject(f"table positions {table_t} do not broadcast to input {input_t}")


class FuseRoPEPass(ExportPass):
    """Replace matched RoPE chains with espressif::rope."""

    def __init__(self, expect: Optional[int] = None):
        super().__init__()
        self.fused = 0
        self.rejected: List[tuple] = []
        self.expect = expect

    def _match(self, add):
        """Return (x, cos, sin, rot_dim, final_node) or raise _Reject.

        Anchored on the ADD, because it is the one node every spelling of this
        pattern has exactly once -- the trailing cat exists only for partial
        rotary, and the leading slices only when rot < width.
        """
        a, b = add.args[0], add.args[1]
        if not (_is(a, *MUL) and _is(b, *MUL)):
            raise _Reject("add's operands are not both muls")
        if not _unit_alpha(add):
            raise _Reject("the add scales its second operand (alpha != 1)")

        # One mul is x_rot*cos, the other rotated*sin. Which is which is decided
        # by finding the cat([-x2, x1]) -- not by argument order, which the
        # author is free to write either way.
        def split_mul(m):
            return m.args[0], m.args[1]

        cand = None
        for rot_mul, plain_mul in ((a, b), (b, a)):
            for rot_side, sin_side in (split_mul(rot_mul),
                                       tuple(reversed(split_mul(rot_mul)))):
                if _is(rot_side, *CAT):
                    cand = (rot_mul, plain_mul, rot_side, sin_side)
                    break
            if cand:
                break
        if cand is None:
            raise _Reject("neither mul takes a cat([-x2, x1])")
        _, plain_mul, rot_cat, sin = cand

        parts = rot_cat.args[0]
        if not (isinstance(parts, (list, tuple)) and len(parts) == 2):
            raise _Reject("the rotate-half cat does not take exactly two parts")
        neg_part, plain_part = parts
        if not _is(neg_part, *NEG):
            raise _Reject("the rotate-half cat's first part is not a neg")
        if not _joins_last_axis(rot_cat):
            raise _Reject("the rotate-half cat is not on the last axis")

        half_a = neg_part.args[0]
        half_b = plain_part
        x_a, idx_a, carrier_a = _halves(half_a)   # neg(x2) -> x2 is half 1
        x_b, idx_b, carrier_b = _halves(half_b)   # x1 -> half 0
        if x_a is not x_b or carrier_a is not carrier_b:
            raise _Reject("the two halves come from different sources")
        if (idx_a, idx_b) != (1, 0):
            raise _Reject(f"halves are ({idx_a}, {idx_b}), expected (1, 0)")

        x_rot = x_a
        # The other mul must be x_rot * cos, on the SAME tensor the halves came
        # from. Without this a chain that rotates one tensor and scales another
        # would match.
        p0, p1 = split_mul(plain_mul)
        if p0 is x_rot:
            cos = p1
        elif p1 is x_rot:
            cos = p0
        else:
            raise _Reject("the non-rotated mul does not multiply the split's source")

        xv = x_rot.meta.get("val")
        if xv is None:
            raise _Reject("no shape metadata on the rotary tensor")
        rot_dim = int(xv.shape[-1])
        if rot_dim % 2:
            raise _Reject(f"rotary width {rot_dim} is odd")
        _validate_tables(x_rot, cos, sin, rot_dim)

        # Partial rotary: x_rot is a slice of a wider x, and the chain ends in a
        # cat that puts the pass-through lanes back. Full rotary: x_rot IS x.
        final, x = add, x_rot
        if _is(x_rot, *SLICE):
            base = x_rot.args[0]
            users = [u for u in add.users]
            if len(users) != 1 or not _is(users[0], *CAT):
                raise _Reject("a sliced rotary part does not end in a cat")
            tail = users[0]
            tparts = tail.args[0]
            if not (isinstance(tparts, (list, tuple)) and len(tparts) == 2
                    and tparts[0] is add and _is(tparts[1], *SLICE)
                    and tparts[1].args[0] is base):
                raise _Reject("the trailing cat is not [rotated, pass-through] "
                              "of the same source")
            # The op rotates base[..., :rot] and appends base[..., rot:]. A
            # common source is not enough: rotating the suffix, or slicing and
            # rejoining the position axis, has the same shape and computes
            # something else.
            _, r0, r1, width = _last_axis_slice(x_rot)
            _, p0, p1, _ = _last_axis_slice(tparts[1])
            if ((r0, r1, p0, p1) != (0, rot_dim, rot_dim, width)
                    or not _joins_last_axis(tail)):
                raise _Reject("the partial rotary is not x[..., :rot] rotated "
                              "then x[..., rot:] appended on the last axis")
            final, x = tail, base

        # Nothing intermediate may escape, or fusing would delete a value someone
        # else reads. The SPLIT is the exception and legitimately has two users:
        # it returns a tuple, and both halves are consumed inside the pattern.
        # Requiring one user there would reject every real site.
        for mid in (rot_cat, neg_part, plain_mul, cand[0], half_a, half_b):
            if isinstance(mid, torch.fx.Node) and len(mid.users) != 1:
                raise _Reject(f"{mid} is consumed outside the pattern")
        if carrier_a is not None:
            split_users = list(carrier_a.users)
            if set(split_users) != {half_a, half_b}:
                raise _Reject("the split does not feed exactly the matched halves")
        return x, cos, sin, rot_dim, final

    def _match_pairwise(self, cat):
        """The other spelling: `cat([x1*cos - x2*sin, x2*cos + x1*sin], -1)`.

        The story model writes RoPE this way (story/glimmer_model.py), and
        HuggingFace writes the rotate-half form matched above. They compute the
        same thing. Which one a model uses is not something the backend should
        get to care about, as with bmm-as-linear and 4-D q@kT (BmmToMm,
        QkTo3D).
        """
        parts = cat.args[0]
        if not (isinstance(parts, (list, tuple)) and len(parts) == 2):
            raise _Reject("cat does not take exactly two parts")
        lo, hi = parts
        if not (_is(lo, "aten::sub.Tensor") and _is(hi, *ADD)):
            raise _Reject("cat's parts are not (sub, add)")
        if not _joins_last_axis(cat):
            raise _Reject("the two rotated halves are not joined on the last axis")
        if not (_unit_alpha(lo) and _unit_alpha(hi)):
            raise _Reject("the sub or add scales its second operand (alpha != 1)")
        for n in (lo, hi):
            if not all(_is(a, *MUL) for a in n.args[:2]):
                raise _Reject(f"{n} is not built from two muls")

        # sub = x1*cos - x2*sin ; add = x2*cos + x1*sin
        (m_x1c, m_x2s), (m_x2c, m_x1s) = lo.args[:2], hi.args[:2]

        def operands(m):
            return m.args[0], m.args[1]

        def shared(m, n):
            """The tensor two muls have in common, or None."""
            a = set(o for o in operands(m) if isinstance(o, torch.fx.Node))
            b = set(o for o in operands(n) if isinstance(o, torch.fx.Node))
            common = a & b
            return next(iter(common)) if len(common) == 1 else None

        cos = shared(m_x1c, m_x2c)
        sin = shared(m_x2s, m_x1s)
        if cos is None or sin is None:
            raise _Reject("could not identify shared cos/sin operands")
        x1 = next((o for o in operands(m_x1c) if o is not cos), None)
        x2 = next((o for o in operands(m_x2c) if o is not cos), None)
        if x1 is None or x2 is None or x1 is x2:
            raise _Reject("could not identify the two halves")
        if {o for o in operands(m_x1s) if o is not sin} != {x1}:
            raise _Reject("the sin terms do not use the same halves")
        if {o for o in operands(m_x2s) if o is not sin} != {x2}:
            raise _Reject("the sin terms do not use the same halves")

        xa, ia, carrier_a = _halves(x1)
        xb, ib, carrier_b = _halves(x2)
        if xa is not xb or carrier_a is not carrier_b:
            raise _Reject("the halves come from different sources")
        if (ia, ib) != (0, 1):
            raise _Reject(f"halves are ({ia}, {ib}), expected (0, 1)")
        x = xa
        xv = x.meta.get("val")
        if xv is None:
            raise _Reject("no shape metadata on the rotary tensor")
        rot_dim = int(xv.shape[-1])
        if rot_dim % 2:
            raise _Reject(f"rotary width {rot_dim} is odd")
        _validate_tables(x, cos, sin, rot_dim)
        for mid in (lo, hi, m_x1c, m_x2s, m_x2c, m_x1s):
            if len(mid.users) != 1:
                raise _Reject(f"{mid} is consumed outside the pattern")
        if set(x1.users) != {m_x1c, m_x1s}:
            raise _Reject(f"{x1} is consumed outside its two rotation terms")
        if set(x2.users) != {m_x2c, m_x2s}:
            raise _Reject(f"{x2} is consumed outside its two rotation terms")
        if carrier_a is not None and set(carrier_a.users) != {x1, x2}:
            raise _Reject("the split does not feed exactly the matched halves")
        return x, cos, sin, rot_dim, cat

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        changed = False
        for node in list(graph.nodes):
            if _is(node, *ADD):
                matcher = self._match
            elif _is(node, *CAT):
                matcher = self._match_pairwise
            else:
                continue
            if not node.users:
                continue        # already replaced this round
            try:
                x, cos, sin, rot_dim, final = matcher(node)
            except _Reject as why:
                self.rejected.append((node.name, str(why)))
                continue
            except (AttributeError, IndexError, TypeError, KeyError) as why:
                self.rejected.append((node.name, f"malformed: {why}"))
                continue

            with graph.inserting_before(final):
                fused = graph.call_function(ROPE, (x, cos, sin, rot_dim))
                fused.meta = dict(final.meta)
            final.replace_all_uses_with(fused)
            self.fused += 1
            changed = True

        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()
        if self.expect is not None and self.fused != self.expect:
            raise RuntimeError(
                f"FuseRoPEPass fused {self.fused}, expected {self.expect}. "
                f"Rejected: {self.rejected[:6]}")
        return PassResult(graph_module, changed)


def fuse_rope(edge_manager, method_name: str = "forward",
              expect: Optional[int] = None) -> FuseRoPEPass:
    """Run the pass over an EdgeProgramManager in place; return it for its counts."""
    p = FuseRoPEPass(expect=expect)
    ep = edge_manager.exported_program(method_name)
    p.call(ep.graph_module)
    return p
