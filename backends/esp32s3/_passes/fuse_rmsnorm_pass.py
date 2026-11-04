# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse decomposed scaled or weightless RMSNorm into one Espressif custom op.

The story model's RMSNorm (examples/espressif/esp32s3/story/glimmer_model.py) is

    weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

which torch.export lowers to six edge ops. Muse Glimmer additionally uses the weightless
form `x * rsqrt(mean(x**2) + eps)` for embedding and Q/K normalization. The two forms
emit `espressif::rmsnorm` and `espressif::rmsnorm_noscale`, respectively, and share the
same device reduction kernel.

Other spellings of the same operation fuse too: the inverse root as `pow(v, -0.5)`
(Gemma 3n and Muse Glimmer's reference norm), `reciprocal(sqrt(v))`, `1 / sqrt(v)` or
the division `x / sqrt(v)`; the square as `x * x`; the weight before, after, or folded
into the inverse root; and a weightless norm times a constant 1.0 (the reference norm's
`with_scale=False` buffer). They differ from the kernel's `(w * x) * rsqrt(v)` only in
rounding. A different function -- eps outside the root, a sum instead of a mean (L2
norm), a centred variance (LayerNorm), another exponent -- is not matched.

Why fuse. NOT for flops: one norm over D=96 is ~300 flops against ~393k MACs per token,
and the model is bandwidth-bound at ~1 MAC/byte, so the arithmetic is noise. The payoff is
(a) the memory plan -- `x**2` and `w*x` are full-size activation temporaries that the
planner must find room for on a 512 KB part -- and (b) five kernel dispatches per norm
collapsing to one. See test_rmsnorm.py, which measures both rather than asserting them.

Matching is structural: the chain is walked from the final multiply backwards and every
link is checked (op identity, operand identity, dim/keepdim, dtype, weight rank, and that
no intermediate is consumed by anything outside the pattern). A site that fails any check
is left completely alone and recorded in `self.rejected` with a reason, so a partial
rewrite is not representable. `expect=` turns a miscount into an exception.

`eps` reaches the edge graph as a *lifted tensor constant placeholder*, not an inline
scalar -- so resolving it needs the ExportedProgram's constant table, which is why the
entry point is `fuse_rmsnorm(edge_manager)` rather than a bare pass instance.

The op declaration is a functional overload plus an out
variant (ExecuTorch's memory planner requires the latter), both implemented for
CompositeExplicitAutograd. The impl is written in terms of aten ops, which is what gives
it meta/FakeTensor behaviour for free -- there is no separate abstract impl to keep in
sync. The library is opened as FRAGMENT, not DEF, so this module can coexist in one
process with the other modules that open the same namespace.

The inserted node is the *functional* overload; ExecuTorch's ToOutVarPass rewrites it to
`espressif::rmsnorm.out` inside to_executorch(), the same path dot_i8 takes. That is the
name the device kernel registers.
"""

from typing import Dict, List, Optional, Tuple

import torch
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult
from torch.library import Library

# ---- op declaration ---------------------------------------------------------
lib = Library("espressif", "FRAGMENT")
lib.define("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor")
lib.define("rmsnorm.out(Tensor x, Tensor weight, float eps, *, Tensor(a!) out) -> Tensor(a!)")
lib.define("rmsnorm_noscale(Tensor x, float eps) -> Tensor")
lib.define("rmsnorm_noscale.out(Tensor x, float eps, *, Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Deliberately the same operation order as the graph it replaces -- (weight * x)
    # scaled by rsqrt -- so the fused and unfused programs agree bit-for-bit on the
    # host, and the device kernel has an unambiguous reference. The other accepted
    # spellings (pow -0.5, x / sqrt, weight after) agree with it to rounding only.
    return (weight * x) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rmsnorm_out(x: torch.Tensor, weight: torch.Tensor, eps: float, *, out: torch.Tensor):
    out.copy_(_rmsnorm(x, weight, eps))
    return out


def _rmsnorm_noscale(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rmsnorm_noscale_out(x: torch.Tensor, eps: float, *, out: torch.Tensor):
    out.copy_(_rmsnorm_noscale(x, eps))
    return out


impl.impl("rmsnorm", _rmsnorm, "CompositeExplicitAutograd")
impl.impl("rmsnorm.out", _rmsnorm_out, "CompositeExplicitAutograd")
impl.impl("rmsnorm_noscale", _rmsnorm_noscale, "CompositeExplicitAutograd")
impl.impl("rmsnorm_noscale.out", _rmsnorm_noscale_out, "CompositeExplicitAutograd")

RMSNORM = torch.ops.espressif.rmsnorm.default
RMSNORM_NOSCALE = torch.ops.espressif.rmsnorm_noscale.default


# ---- structural matching ----------------------------------------------------
def _opname(target) -> str:
    """Canonical 'aten::mul.Tensor' spelling for edge, aten and custom overloads."""
    name = getattr(target, "_name", None)
    if name:
        return name
    name = getattr(target, "name", None)
    if callable(name):
        return name()
    return str(target)


def _is(node, *names) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and _opname(node.target) in names
    )


MUL = ("aten::mul.Tensor",)
DIV = ("aten::div.Tensor",)
POW = ("aten::pow.Tensor_Scalar",)
MEAN = ("aten::mean.dim",)
ADD = ("aten::add.Tensor", "aten::add.Scalar")
RSQRT = ("aten::rsqrt.default", "aten::rsqrt")
SQRT = ("aten::sqrt.default", "aten::sqrt")
RECIPROCAL = ("aten::reciprocal.default", "aten::reciprocal")
TO_COPY = ("dim_order_ops::_to_dim_order_copy.default",
           "dim_order_ops::_to_dim_order_copy")


def _fake(node):
    return node.meta.get("val", None) if isinstance(node, torch.fx.Node) else None


def _inverse_rms(node):
    """``(v, nodes)`` when `node` computes ``v ** -0.5``, else None.

    Three spellings of one operation: ``rsqrt(v)``; ``pow(v, -0.5)``, which
    Gemma 3n and Muse Glimmer's reference norm write; and ``reciprocal(sqrt(v))``
    (``1 / sqrt(v)`` exports as that times 1). `nodes` is the chain from `node`
    down to, but excluding, `v`.
    """
    if _is(node, *RSQRT) and len(node.args) == 1:
        return node.args[0], [node]
    if _is(node, *POW) and len(node.args) == 2 and node.args[1] == -0.5:
        return node.args[0], [node]
    if (_is(node, *RECIPROCAL) and len(node.args) == 1
            and _is(node.args[0], *SQRT) and len(node.args[0].args) == 1):
        return node.args[0].args[0], [node, node.args[0]]
    return None


def _scale_parts(node):
    """``(a, v, nodes)`` when `node` is ``a * v**-0.5``, else None.

    Either ``mul(a, r)`` with `r` an `_inverse_rms` spelling, or the division
    ``a / sqrt(v)``. Both are the kernel's ``a * rsqrt(v)``; `nodes` is the
    inverse-RMS chain, excluding `node` and `v`.
    """
    if _is(node, *MUL) and len(node.args) == 2 and not node.kwargs:
        for a, r in (node.args, node.args[::-1]):
            got = _inverse_rms(r)
            if got is not None and a is not r:
                return (a, *got)
    if (_is(node, *DIV) and len(node.args) == 2 and not node.kwargs
            and _is(node.args[1], *SQRT) and len(node.args[1].args) == 1):
        return node.args[0], node.args[1].args[0], [node.args[1]]
    return None


class _Reject(Exception):
    """A site that looked like RMSNorm but is not one. Carries the failed check."""



class FuseRMSNormPass(ExportPass):
    """Replace matched RMSNorm chains with espressif::rmsnorm.

    `constants` maps placeholder *node names* to their materialised tensors, so a lifted
    `eps` can be read. Without it, only graphs where eps survived as an inline scalar can
    match, and every other site is rejected with a reason rather than guessed at.
    """

    def __init__(
        self,
        constants: Optional[Dict[str, torch.Tensor]] = None,
        expect: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.constants: Dict[str, torch.Tensor] = dict(constants or {})
        self.expect = expect
        self.fused = 0
        self.candidates = 0
        self.rejected: List[Tuple[str, str]] = []

    # -- helpers ------------------------------------------------------------
    def _eps_of(self, arg) -> float:
        if isinstance(arg, (float, int)) and not isinstance(arg, bool):
            return float(arg)
        if isinstance(arg, torch.fx.Node) and arg.op == "placeholder":
            t = self.constants.get(arg.name)
            if t is None:
                raise _Reject(f"eps placeholder {arg.name!r} is not a known constant")
            if t.numel() != 1 or not t.dtype.is_floating_point:
                raise _Reject(f"eps constant {arg.name!r} is {tuple(t.shape)}/{t.dtype}")
            return float(t.reshape(()).item())
        if isinstance(arg, torch.fx.Node) and _is(arg, "aten::full.default"):
            raise _Reject("eps is a materialised full() tensor, not a scalar constant")
        raise _Reject(f"eps operand is {type(arg).__name__}, not a constant scalar")

    @staticmethod
    def _only_user(node, user) -> None:
        if len(node.users) != 1 or user not in node.users:
            raise _Reject(
                f"{node.name} has {len(node.users)} users; the intermediate escapes "
                "the pattern so fusing it would delete a live value"
            )

    def _unit_constant(self, node) -> bool:
        """Whether `node` is a build-time constant of one element equal to 1.0.

        Multiplying by it is the identity (x * 1.0 == x for every float), so a
        norm whose "weight" is such a constant is the weightless norm. Two real
        spellings produce one: Gemma 3n / Muse Glimmer's reference norm keeps a
        0-d ``weight = torch.tensor(1.0)`` buffer when ``with_scale=False``, and
        ``1 / sqrt(v)`` exports as ``reciprocal(sqrt(v)) * 1`` (the integer 1
        reaching the mul through a dtype-converting dim-order copy).
        """
        if _is(node, *TO_COPY) and node.args:
            node = node.args[0]
        if not (isinstance(node, torch.fx.Node) and node.op == "placeholder"):
            return False
        t = self.constants.get(node.name)
        return (t is not None and t.numel() == 1 and t.dtype != torch.bool
                and float(t.detach().reshape(())) == 1.0)

    def _match(self, out_mul: torch.fx.Node):
        """Return (x, weight, eps, dead_nodes) or raise _Reject."""
        args = out_mul.args
        if len(args) != 2 or out_mul.kwargs:
            raise _Reject("final op is not a plain 2-operand mul or div")

        # Where the weight sits. Every association here is RMSNorm, and the
        # kernel computes all of them as (w * x) * rsqrt(...):
        #   weightless / A (Glimmer's order):    out = s(x)  or  s(w * x)
        #   B (nn.RMSNorm, HF Llama/Gemma/T5):   out = s(x) * w
        #   C (the weight folded into r):        out = x * (w * r)
        # where s(a) is a * r with r = v ** -0.5 (rsqrt, pow -0.5, reciprocal of
        # sqrt) or a / sqrt(v). nn.RMSNorm spells eps as add.Scalar where a
        # hand-written model lifts a tensor constant, hence ADD covering both
        # spellings.
        #
        # Same op, not a substitute: every spelling is x * (mean(x^2) + eps)
        # ** -0.5, and they differ from the kernel only in rounding -- the size
        # of difference the kernel already has against the rsqrt graph (its
        # summation order and two Newton steps; see test_rmsnorm.py). A
        # different function stays portable: eps outside the root, a sum
        # (L2 norm), a centred variance (LayerNorm), any other exponent.
        inner, outer = None, None
        scale = _scale_parts(out_mul)
        if scale is None:
            if not _is(out_mul, *MUL):
                raise _Reject("no inverse-RMS operand")
            for o, q in (args, args[::-1]):
                scale = _scale_parts(q) if o is not q else None
                if scale is not None:
                    inner, outer = q, o
                    break
            if inner is None:
                raise _Reject("no inverse-RMS operand")
            self._only_user(inner, out_mul)
        a, add, inv = scale
        user = inner if inner is not None else out_mul
        for n in inv:
            self._only_user(n, user)
            user = n

        # (add(mean(x ** 2, -1, keepdim), eps)) ** -0.5
        if not _is(add, *ADD):
            raise _Reject("the inverse-RMS input is not add.Tensor")
        self._only_user(add, user)
        if len(add.args) != 2 or add.kwargs.get("alpha", 1) != 1:
            raise _Reject("add is not a plain 2-operand add with alpha=1")
        mean = next((m for m in add.args if _is(m, *MEAN)), None)
        if mean is None:
            raise _Reject("add has no mean.dim operand")
        eps = self._eps_of(add.args[0] if add.args[1] is mean else add.args[1])
        self._only_user(mean, add)

        if len(mean.args) < 3:
            raise _Reject("mean.dim is missing dim/keepdim")
        square, dim, keepdim = mean.args[0], mean.args[1], mean.args[2]
        if len(mean.args) > 3 and mean.args[3] is not None:
            raise _Reject("mean.dim requests a dtype cast")
        if mean.kwargs.get("dtype") is not None:
            raise _Reject("mean.dim requests a dtype cast")
        if keepdim is not True:
            raise _Reject("mean.dim is not keepdim=True")
        if _is(square, *POW):
            if len(square.args) != 2 or square.args[1] != 2:
                raise _Reject(f"pow exponent is {square.args[1:]}, not 2")
        elif not (_is(square, *MUL) and len(square.args) == 2
                  and square.args[0] is square.args[1]):
            raise _Reject("mean input is not x ** 2 (pow.Tensor_Scalar or x * x)")
        self._only_user(square, mean)

        x = square.args[0]
        xv = _fake(x)
        if xv is None:
            raise _Reject("x has no fake-tensor meta to check")
        # dim must name exactly the last axis, however it is spelled.
        want = {-1, xv.dim() - 1}
        got = dim if isinstance(dim, (list, tuple)) else [dim]
        if len(got) != 1 or got[0] not in want:
            raise _Reject(f"mean reduces {got}, not the last dim of a {xv.dim()}D tensor")

        # The weight, by identity with the squared tensor. `x` legitimately has
        # two users inside the pattern, so this is decided by identity and
        # never by use counts.
        w_mul = None
        if inner is None:
            if a is x:
                weight = None
            elif _is(a, *MUL) and len(a.args) == 2 and not a.kwargs:
                if a.args[0] is a.args[1]:
                    raise _Reject("w_mul squares its input; there is no weight")
                if a.args[0] is x:
                    weight = a.args[1]
                elif a.args[1] is x:
                    weight = a.args[0]
                else:
                    raise _Reject(
                        "the multiplied tensor is not the tensor that was squared")
                self._only_user(a, out_mul)
                w_mul = a
            else:
                raise _Reject("the normalized tensor is not the tensor that was squared")
        elif a is x:
            # B (and C): the weight multiplies AFTER the norm here, (x * r) * w,
            # while the kernel computes (w * x) * r. Same operation, rounding
            # apart, for real activations; only a product w * x beyond FLT_MAX
            # (|w x| > 3.4e38, e.g. w = 1e30 on x = 1e10) overflows where the
            # source order would not.
            weight = outer                        # B
        elif outer is x and _is(inner, *MUL):
            weight = a                            # C
        else:
            raise _Reject("the gated tensor is not the tensor that was squared")

        ov = _fake(out_mul)
        if ov is None:
            raise _Reject("output has no fake-tensor meta to check")
        checked = [("x", xv), ("out", ov)]
        if weight is not None:
            wv = _fake(weight)
            if wv is None:
                raise _Reject("weight has no fake-tensor meta to check")
            if wv.numel() == 1:
                if not self._unit_constant(weight):
                    raise _Reject(
                        f"weight is a {tuple(wv.shape)} scalar, not the constant "
                        "1.0; the kernel scales per column, so a scalar gain "
                        "stays a portable multiply")
                weight = None
            elif wv.dim() != 1 or wv.shape[0] != xv.shape[-1]:
                raise _Reject(
                    f"weight is {tuple(wv.shape)}, not 1D of length {int(xv.shape[-1])}; "
                    "the device kernel indexes it per-column"
                )
            else:
                checked.append(("weight", wv))
        for name, v in checked:
            if v.dtype != torch.float32:
                raise _Reject(f"{name} is {v.dtype}, and the kernel is fp32-only")
        if tuple(ov.shape) != tuple(xv.shape):
            raise _Reject(f"output {tuple(ov.shape)} != x {tuple(xv.shape)}")

        # Reverse topological order; every link was proven single-use, so each
        # erase sees zero users.
        dead = [out_mul]
        if inner is not None:
            dead.append(inner)
        if w_mul is not None:
            dead.append(w_mul)
        dead.extend([*inv, add, mean, square])
        return x, weight, eps, dead

    # -- pass ---------------------------------------------------------------
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        # Snapshot: the node list is mutated as sites are rewritten.
        sites = []
        for n in graph.nodes:
            if _scale_parts(n) is None:
                continue
            # Variant B's REWRITE point has no direct rsqrt operand -- the outer mul
            # sees only (inner_mul, weight), with the rsqrt one level down -- so the
            # node to rewrite is this mul's single user. Try it FIRST. Otherwise a
            # centered Glimmer norm written `(x * rsqrt(...)) * (1 + weight)` would
            # fuse only the inner weightless norm and leave the scale multiply
            # portable. If the outer mul is a q-scale scalar rather than a norm weight,
            # its shape check refuses it and the inner candidate still fuses.
            # Variant C's `x * (w * r)` is found the same way, from the inner
            # (w * r), which is itself a scale node.
            users = list(n.users)
            if len(users) == 1 and _is(users[0], *MUL):
                sites.append(users[0])
            sites.append(n)
        seen_ids = set()
        sites = [n for n in sites
                 if not (id(n) in seen_ids or seen_ids.add(id(n)))]
        self.candidates += len(sites)

        for out_mul in sites:
            # A successful outer (Variant B) match erases its inner candidate.
            if out_mul._erased:
                continue
            try:
                x, weight, eps, dead = self._match(out_mul)
            except _Reject as why:
                self.rejected.append((out_mul.name, str(why)))
                continue

            with graph.inserting_before(out_mul):
                if weight is None:
                    fused = graph.call_function(RMSNORM_NOSCALE, (x, eps))
                else:
                    fused = graph.call_function(RMSNORM, (x, weight, eps))
            fused.meta = dict(out_mul.meta)
            fused.name = f"espressif_rmsnorm_{self.fused}"
            out_mul.replace_all_uses_with(fused)
            # `dead` is already in reverse-topological order and every link was proven
            # single-use, so each erase sees zero users.
            for n in dead:
                graph.erase_node(n)
            self.fused += 1

        if self.fused:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        if self.expect is not None and self.fused != self.expect:
            raise RuntimeError(
                f"RMSNorm fusion expected {self.expect} sites, fused {self.fused} "
                f"(candidates {self.candidates}); rejected: {self.rejected}"
            )
        return PassResult(graph_module, self.fused > 0)


# ---- entry point ------------------------------------------------------------
def constant_map(exported_program) -> Dict[str, torch.Tensor]:
    """Placeholder node name -> materialised tensor, for every lifted constant.

    `eps` is lifted to a `_lifted_tensor_constantN` input, so the pass cannot read it out
    of the graph alone. Parameters and buffers are included too; a model that stored eps
    in a buffer would then still match.

    Except a buffer the program MUTATES, which is state rather than a constant: a fusion
    that read it would bake its export-time value into the op and ignore every later
    write. Same rule as ExecuTorch's constant_prop_pass (`is_constant_buffer`).
    """
    sig = exported_program.graph_signature
    mutated = set(getattr(sig, "buffers_to_mutate", {}).values())
    buffers = {name: target
               for name, target in getattr(sig, "inputs_to_buffers", {}).items()
               if target not in mutated}
    tables = (
        (getattr(sig, "inputs_to_lifted_tensor_constants", {}), exported_program.constants),
        (getattr(sig, "inputs_to_parameters", {}), exported_program.state_dict),
        (buffers, dict(exported_program.named_buffers())),
    )
    out: Dict[str, torch.Tensor] = {}
    for names_to_targets, store in tables:
        for node_name, target in names_to_targets.items():
            t = store.get(target)
            if isinstance(t, torch.Tensor):
                out[node_name] = t
    return out


def fuse_rmsnorm(edge_manager, method_name: str = "forward", expect: Optional[int] = None):
    """Run the fusion on an EdgeProgramManager in place; returns the pass for its counts.

    In place because EdgeProgramManager.transform() reconstructs the manager and would
    lose the pass object holding the counts, and because the caller wants the *same*
    manager to hand to to_executorch().
    """
    ep = edge_manager.exported_program(method_name)
    p = FuseRMSNormPass(constants=constant_map(ep), expect=expect)
    result = p.call(ep.graph_module)
    if result.modified:
        # Signature-affecting cleanup is deliberately NOT done here: the eps constants
        # become unused placeholders, which cost 4 bytes each in the constant segment and
        # nothing at runtime. Rewriting graph_signature/input_specs is the kind of surgery
        # that breaks silently at emit time, so test_rmsnorm.py measures the leftover
        # instead of hiding it.
        ep.graph_module.recompile()
    return p
