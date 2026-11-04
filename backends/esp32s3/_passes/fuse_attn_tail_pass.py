# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse the decode attention tail into one `espressif::attn_tail` custom op.

After the delegated q@k^T (OP_BMM_DYN_I8), every Glimmer decode layer runs four
portable ops on its scores [KH, H/KH, L] (KV head is the bmm batch, its query heads
are the rows -- the compact GQA layout of glimmer_decode.py):

    s     = scores * Dh**-0.5             mul.Tensor  (scale: a lifted fp32 constant)
    s     = where(mask[1,1,L], s, -inf)   where.self  (-inf: a shared full() scalar)
    probs = _softmax(s, -1, False)
    out   = bmm(probs, V[KH, L, Dh])

On the story model that chain measured 19.6 ms/token on the chip (mul 1.2, where 2.3,
softmax 8.3, bmm 7.8 -- with L = 64 local, 256 global), and the arithmetic is the small
part of it: each portable call costs ~60-80 us of dispatch, newlib's expf goes through an
errno wrapper, and every softmax lane pays the ROM's software float divide. One fused
call makes one pass per score row for scale, mask and max, maps it with fast_math.h's
fast_exp_neg (a 64-entry table exp), takes ONE reciprocal per row, and accumulates
P@V in the portable bmm's index order.

Semantics: out = softmax(where(mask, scores * scale, -inf), -1) @ values, exactly the
graph it replaces. A FULLY MASKED ROW IS NaN, as it is in that graph: softmax over a row
of -inf computes exp(-inf - -inf) = exp(NaN), in eager PyTorch and in ExecuTorch's
portable kernel alike, and the device kernel's fast_exp_neg(NaN) is NaN too. No decode step
produces one (the current position is always visible), so NaN is the honest answer
rather than a zero that would hide a broken mask.

Numerics. On the host the op is bit-identical to the unfused graph: the
CompositeExplicitAutograd implementation below IS that graph, in aten ops, which is also
what gives meta/FakeTensor behaviour for free. On the device it is not, by design:
fast_exp_neg (<= 1.004 ulp) replaces newlib expf and a reciprocal-multiply replaces a
divide per lane. models.attn_tail_f32_reference transcribes the device arithmetic, fused
multiply-adds included, and the decode goldens use it (glimmer_decode.py).

Matching is structural, anchored on each `_softmax`: every link is checked (op identity,
operand identity, the softmax axis, fp32 throughout, the -inf fill, a mask that is one
row shared by every head, and that no intermediate is consumed outside the chain). A site
failing any check is left alone and recorded in `rejected` with the reason -- prefill's
attention, whose softmax feeds a view rather than the bmm, is the expected example.
`expect=` turns a miscount into an exception.

The scale reaches the edge graph as a lifted tensor constant, so the entry point is
`fuse_attn_tail(edge_manager)`, which reads the constant table, as fuse_rmsnorm_pass.py
does. The inserted node is the functional overload; ToOutVarPass rewrites it to
`espressif::attn_tail.out` inside to_executorch(), the name the device kernel registers.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult
from torch.library import Library

from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import constant_map

# ---- op declaration ---------------------------------------------------------
# FRAGMENT, not DEF: fuse_rmsnorm_pass.py and fuse_rope_pass.py open the same
# "espressif" namespace too.
lib = Library("espressif", "FRAGMENT")
lib.define("attn_tail(Tensor scores, Tensor mask, float scale, Tensor values) -> Tensor")
lib.define("attn_tail.out(Tensor scores, Tensor mask, float scale, Tensor values, *, "
           "Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")


def _attn_tail(scores: torch.Tensor, mask: torch.Tensor, scale: float,
               values: torch.Tensor) -> torch.Tensor:
    # The replaced graph op for op. `scores * scale` multiplies by the fp32 value
    # of the scale, as mul.Tensor by the 0-dim constant does.
    masked = torch.where(mask, scores * scale,
                         torch.full((), float("-inf"), dtype=scores.dtype,
                                    device=scores.device))
    return torch.bmm(torch.softmax(masked, dim=-1), values)


def _attn_tail_out(scores, mask, scale, values, *, out):
    out.copy_(_attn_tail(scores, mask, scale, values))
    return out


impl.impl("attn_tail", _attn_tail, "CompositeExplicitAutograd")
impl.impl("attn_tail.out", _attn_tail_out, "CompositeExplicitAutograd")

ATTN_TAIL = torch.ops.espressif.attn_tail.default


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
    """Exact overload match. The edge dialect drops `.default` from a name that
    aten spells with it, so default overloads are listed in both spellings --
    explicitly, because comparing base names alone would also admit mul.Scalar
    or where.ScalarOther, which this pass has not been written against."""
    return (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and _opname(node.target) in names)


SOFTMAX = ("aten::_softmax", "aten::_softmax.default")
WHERE = ("aten::where.self",)
MUL = ("aten::mul.Tensor",)
BMM = ("aten::bmm", "aten::bmm.default")
FULL = ("aten::full", "aten::full.default")


def _fake(node):
    return node.meta.get("val", None) if isinstance(node, torch.fx.Node) else None


def _shape(value) -> Tuple[int, ...]:
    try:
        return tuple(int(d) for d in value.shape)
    except (TypeError, ValueError):
        raise _Reject(f"dynamic shape {tuple(value.shape)}; the kernel's "
                      "validation and scratch size are per call, not symbolic") from None


class _Reject(Exception):
    """A site that looked like the attention tail but is not one."""


class FuseAttnTailPass(ExportPass):
    """Replace matched mul -> where -> _softmax -> bmm chains with espressif::attn_tail.

    `constants` maps placeholder node names to their tensors so the lifted scale can be
    read; without it only an inline scalar scale can match.
    """

    def __init__(self, constants: Optional[Dict[str, torch.Tensor]] = None,
                 expect: Optional[int] = None) -> None:
        super().__init__()
        self.constants: Dict[str, torch.Tensor] = dict(constants or {})
        self.expect = expect
        self.fused = 0
        self.candidates = 0
        self.rejected: List[Tuple[str, str]] = []

    # -- helpers ------------------------------------------------------------
    def _scale_of(self, arg) -> float:
        """The fp32 value the mul multiplies by, as a Python float (exact)."""
        if isinstance(arg, (float, int)) and not isinstance(arg, bool):
            # An inline scalar multiplies an fp32 tensor at fp32, so pass the
            # fp32 value; every consumer then sees the same number.
            return float(torch.tensor(float(arg), dtype=torch.float32).item())
        if isinstance(arg, torch.fx.Node) and arg.op == "placeholder":
            t = self.constants.get(arg.name)
            if t is None:
                raise _Reject(f"scale placeholder {arg.name!r} is not a known constant")
            if t.numel() != 1 or t.dtype != torch.float32:
                raise _Reject(f"scale constant {arg.name!r} is "
                              f"{tuple(t.shape)}/{t.dtype}, not one fp32 value")
            return float(t.reshape(()).item())
        raise _Reject(f"scale operand {getattr(arg, 'name', arg)!r} is not a "
                      "constant scalar")

    def _require_neg_inf(self, arg) -> None:
        """The where's false operand must be the constant -inf, as one fp32 value."""
        if isinstance(arg, (float, int)) and not isinstance(arg, bool):
            if not (math.isinf(arg) and arg < 0):
                raise _Reject(f"masked fill is {arg!r}, not -inf")
            return
        if _is(arg, *FULL):
            size = arg.args[0] if arg.args else arg.kwargs.get("size")
            fill = arg.args[1] if len(arg.args) > 1 else arg.kwargs.get("fill_value")
            if not isinstance(fill, (float, int)) or not (math.isinf(fill) and fill < 0):
                raise _Reject(f"masked fill is full({fill!r}), not -inf")
            if any(int(d) != 1 for d in size):
                raise _Reject(f"masked fill full() has size {list(size)}, not one value")
            v = _fake(arg)
            if v is None or v.dtype != torch.float32:
                raise _Reject("masked fill full() is not fp32")
            return
        if isinstance(arg, torch.fx.Node) and arg.op == "placeholder":
            t = self.constants.get(arg.name)
            if t is None or t.numel() != 1 or t.dtype != torch.float32:
                raise _Reject(f"masked fill {arg.name!r} is not a known fp32 scalar")
            value = float(t.reshape(()).item())
            if not (math.isinf(value) and value < 0):
                raise _Reject(f"masked fill constant is {value!r}, not -inf")
            return
        raise _Reject(f"masked fill {getattr(arg, 'name', arg)!r} is not a constant")

    @staticmethod
    def _only_user(node, user) -> None:
        if len(node.users) != 1 or user not in node.users:
            raise _Reject(
                f"{node.name} has {len(node.users)} users; the intermediate escapes "
                "the pattern so fusing it would delete a live value")

    def _match(self, softmax: torch.fx.Node):
        """Return (scores, mask, scale, values, bmm, dead) or raise _Reject."""
        if len(softmax.args) != 3 or softmax.kwargs:
            raise _Reject("_softmax does not take (x, dim, half_to_float)")
        where, dim, half_to_float = softmax.args
        if half_to_float is not False:
            raise _Reject("_softmax requests half_to_float")

        users = list(softmax.users)
        if len(users) != 1 or not _is(users[0], *BMM):
            raise _Reject("softmax output does not feed exactly one bmm "
                          f"({[u.name for u in users]})")
        bmm = users[0]
        if len(bmm.args) != 2 or bmm.kwargs or bmm.args[0] is not softmax:
            raise _Reject("probabilities are not the left operand of a plain bmm")
        values = bmm.args[1]
        if values is softmax:
            raise _Reject("bmm multiplies the probabilities by themselves")

        if not _is(where, *WHERE):
            raise _Reject("softmax input is not where.self")
        self._only_user(where, softmax)
        if len(where.args) != 3 or where.kwargs:
            raise _Reject("where.self does not take (mask, value, fill)")
        mask, mul, fill = where.args
        self._require_neg_inf(fill)

        if not _is(mul, *MUL):
            raise _Reject("the selected value is not a mul.Tensor")
        self._only_user(mul, where)
        if len(mul.args) != 2 or mul.kwargs:
            raise _Reject("mul is not a plain 2-operand mul")
        # Either operand order: exactly one side is the constant scale.
        a, b = mul.args
        try:
            scale, scores = self._scale_of(b), a
        except _Reject as first:
            try:
                scale, scores = self._scale_of(a), b
            except _Reject:
                raise first from None
        if not isinstance(scores, torch.fx.Node):
            raise _Reject("both mul operands are constants")

        sv, mv, wv, pv = _fake(scores), _fake(mask), _fake(where), _fake(softmax)
        vv, ov, kv = _fake(values), _fake(bmm), _fake(mul)
        if any(v is None for v in (sv, mv, wv, pv, vv, ov, kv)):
            raise _Reject("an operand has no fake-tensor meta to check")
        for name, v in (("scores", sv), ("scaled scores", kv), ("masked scores", wv),
                        ("probabilities", pv), ("values", vv), ("output", ov)):
            if v.dtype != torch.float32:
                raise _Reject(f"{name} is {v.dtype}, and the kernel is fp32-only")
        if mv.dtype != torch.bool:
            raise _Reject(f"mask is {mv.dtype}, not bool")

        s_shape = _shape(sv)
        if len(s_shape) != 3:
            raise _Reject(f"scores are {s_shape}, not the bmm layout [B, M, L]")
        if not (dim in (-1, 2)):
            raise _Reject(f"softmax runs over dim {dim}, not the last axis")
        for name, v in (("scaled scores", kv), ("masked scores", wv),
                        ("probabilities", pv)):
            if _shape(v) != s_shape:
                raise _Reject(f"{name} are {_shape(v)}, not {s_shape}; the chain "
                              "broadcasts where the kernel does not")
        batches, rows, width = s_shape
        v_shape = _shape(vv)
        if len(v_shape) != 3 or v_shape[:2] != (batches, width):
            raise _Reject(f"values are {v_shape}, not [{batches}, {width}, N]")
        if _shape(ov) != (batches, rows, v_shape[2]):
            raise _Reject(f"output {_shape(ov)} != [{batches}, {rows}, {v_shape[2]}]")
        if min(s_shape + v_shape) < 1:
            raise _Reject("an empty dimension")
        m_shape = _shape(mv)
        if (not 1 <= len(m_shape) <= 3 or m_shape[-1] != width
                or any(d != 1 for d in m_shape[:-1])):
            raise _Reject(f"mask is {m_shape}, not one [..., 1, {width}] row; the "
                          "kernel applies a single mask row to every head")

        return scores, mask, scale, values, bmm, [bmm, softmax, where, mul]

    # -- pass ---------------------------------------------------------------
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        sites = [n for n in graph.nodes if _is(n, *SOFTMAX)]
        self.candidates += len(sites)
        for softmax in sites:
            try:
                scores, mask, scale, values, bmm, dead = self._match(softmax)
            except _Reject as why:
                self.rejected.append((softmax.name, str(why)))
                continue
            except (AttributeError, IndexError, TypeError, KeyError) as why:
                self.rejected.append((softmax.name, f"malformed: {why}"))
                continue

            # At the bmm: every operand (the scores and mask before the mul and
            # where, the values after the cache write) is defined by then.
            with graph.inserting_before(bmm):
                fused = graph.call_function(ATTN_TAIL, (scores, mask, scale, values))
            fused.meta = dict(bmm.meta)
            bmm.replace_all_uses_with(fused)
            # Reverse-topological, each link proven single-use: every erase sees
            # zero users. The shared -inf full() goes with eliminate_dead_code
            # once no where reads it.
            for n in dead:
                graph.erase_node(n)
            self.fused += 1

        if self.fused:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        if self.expect is not None and self.fused != self.expect:
            raise RuntimeError(
                f"attention-tail fusion expected {self.expect} sites, fused "
                f"{self.fused} (candidates {self.candidates}); rejected: "
                f"{self.rejected[:6]}")
        return PassResult(graph_module, self.fused > 0)


# ---- entry point ------------------------------------------------------------
def fuse_attn_tail(edge_manager, method_name: str = "forward",
                   expect: Optional[int] = None) -> FuseAttnTailPass:
    """Run the fusion on an EdgeProgramManager in place; returns the pass for its counts.

    Like fuse_rmsnorm, the scale constants it reads stay behind as unused placeholders
    (4 bytes each in the constant segment) rather than having the signature rewritten.
    """
    ep = edge_manager.exported_program(method_name)
    p = FuseAttnTailPass(constants=constant_map(ep), expect=expect)
    p.call(ep.graph_module)
    return p
