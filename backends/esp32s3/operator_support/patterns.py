# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph patterns the partitioner claims or fuses into a matmul."""

import math

import numpy as np
import torch
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.backends.esp32s3.operator_support.const_weights import (
    FOLDED,
    _const_tensor,
    _is_build_time_const,
    _is_pass_through_const_path,
    _is_torchao_dequant,
    _pass_through_source,
)
from executorch.backends.esp32s3.serialization.espressif_blob import (
    OP_ADDMM,
    OP_EMBED_I8,
)


# `mm` matters as much as `addmm`: a bias-free Linear -- which is what a transformer
# block is built from -- lowers to `mm`, so claiming only `addmm` delegates nothing at
# all from a real model. Both map to one runtime path; a missing bias becomes a zero
# vector at build time, which costs N floats and saves a second kernel.
SUPPORTED = {
    exir_ops.edge.aten.addmm.default: OP_ADDMM,
    exir_ops.edge.aten.mm.default: OP_ADDMM,
    # The gating op for a per-layer-embedding model. A `ple_table` of 32768 x 6 x 128 =
    # 25.17M params left fp32 is 100.7 MB against a 16 MB flash budget -- so such a model
    # cannot be lowered at deployed scale until the embedding is quantised. At int4 plus
    # scales it is 13.0 MB and fits. Unlike a matmul there is no arithmetic here at all:
    # a gather reads exactly one row per token, so this is pure storage format, and the
    # only thing that matters is bytes on the flash side of the mmap.
    exir_ops.edge.aten.embedding.default: OP_EMBED_I8,
}


def _from_promoted_checkpoint(node):
    """The dtype this op's weight ARRIVED as, or None if it was ordinary fp32.

    Returns the dtype string rather than a bool because fp8 and fp16 are not the
    same case and must not share a policy. See REQUANTIZE_POLICIES.

    True when the weight chain bottoms out in a constant `promote_fp8_constants`
    upcast -- an fp8, fp16 or bf16 tensor the model shipped with. Walks the same
    build-time-foldable ops `_resolve_const` does, because that is the chain the
    weight actually travels.

    The reason this is knowable at all is that promotion tags the placeholder.
    After promotion the graph is pure fp32 and carries no other trace of where
    the numbers came from.

    Only the WEIGHT is followed: the operand this op quantises, through the
    chain `_resolve_const` replays and the selected-row head's index_select.
    Walking every argument would let a conv's promoted fp16 BIAS -- stored as
    fp32 whatever it arrived as -- answer for its fp8 weight under "half".
    """
    n = _weight_operand(node)
    while (isinstance(n, torch.fx.Node) and n.op == "call_function"
           and (n.target in FOLDED
                or n.target is exir_ops.edge.aten.index_select.default
                or _is_torchao_dequant(n))):
        n = n.args[0]
    if isinstance(n, torch.fx.Node) and n.op == "placeholder":
        return n.meta.get("esp32_promoted_from")
    return None


def _weight_operand(node):
    """The operand a core op quantises, or None for anything else."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if node.target is exir_ops.edge.aten.addmm.default:
        return node.args[2]
    if node.target is exir_ops.edge.aten.embedding.default:
        return node.args[0]
    if node.target in (exir_ops.edge.aten.mm.default,
                       exir_ops.edge.aten.convolution.default):
        return node.args[1]
    return None


def _conv_producer(node, grouped_ok: bool = False) -> bool:
    """A convolution an epilogue can ride on.

    `grouped_ok` splits the two kinds of epilogue, and the split is arithmetic
    rather than bookkeeping.

    A grouped conv emits `g` ops, each writing a disjoint slice of the output
    channels. A POINTWISE epilogue -- relu, gelu, clamp, silu -- is a function of
    one element, so applying it to every group is applying it to the whole
    output, exactly. Those fuse, by setting the flag on all g ops.

    A RESIDUAL does not follow, and not because it is harder: the runtime adds it
    during the NCHW scatter, indexing the residual as the destination is indexed,
    and for a group that destination is offset into a tensor the residual pointer
    does not share an origin with. It is refusable at no cost and wrong at some,
    so it stays refused: an index disagreement like this one passes every
    structural check, and only a measured answer shows it.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target is exir_ops.edge.aten.convolution.default):
        return False
    from executorch.backends.esp32s3.operator_support.conv import (
        ConvRefused,
        admit,
        split_groups,
    )

    try:
        return grouped_ok or len(split_groups(admit(node))) == 1
    except ConvRefused:
        return False


def _silu_pattern(node, allow_conv: bool = False):
    # Defaults FALSE, and that default is load-bearing. This function is called
    # from the gate, from preprocess, and from `_tail_root_mm`; with a True
    # default the gate (passing False) and the walks (passing nothing) would
    # disagree about which patterns exist, which surfaces as
    # "gate and walk disagree" from a completely unrelated op. Every caller that
    # wants conv-rooted SiLU has to say so.
    """If `node` is any corner of  mul(P, sigmoid(P))  where P is a matmul's output
    or a single view_copy of it, return (mm, sigmoid, mul, P); else None.

    The diamond is the point: P feeds BOTH the sigmoid and the mul, so the
    single-user gates that carry every other fusion here are exactly wrong for this
    one. P-not-mm matters just as much: a 3-D F.linear traces as view -> mm -> view,
    so the diamond forms one numel-preserving reshape ABOVE the matmul it belongs
    to; without seeing through that view, silu/gelu fuse at no site of such a
    model. (nn.Linear's trace lands the diamond on the naked addmm.)
    """
    SIG = exir_ops.edge.aten.sigmoid.default
    MUL = exir_ops.edge.aten.mul.Tensor
    VIEW = exir_ops.edge.aten.view_copy.default
    # Matmuls, not everything in SUPPORTED: an embedding is a gather with no
    # epilogue, and preprocess records no fusion tail for it.
    MM = (exir_ops.edge.aten.mm.default, exir_ops.edge.aten.addmm.default)

    def claimable(P):
        # Conv is claimed by its own branch, not by membership in SUPPORTED, so
        # it is named here, as in the clamp and gelu gates: otherwise a SiLU
        # after a convolution finds no producer and stays portable (40
        # sigmoid+mul pairs on a YOLO, 46% of its portable output traffic).
        return (isinstance(P, torch.fx.Node) and P.op == "call_function"
                and (P.target in MM
                     or (allow_conv and _conv_producer(P, grouped_ok=True))))

    def underlying_mm(P):
        if not (isinstance(P, torch.fx.Node) and P.op == "call_function"):
            return None
        if claimable(P):
            return P
        if (P.target is VIEW and isinstance(P.args[0], torch.fx.Node)
                and claimable(P.args[0])
                and set(P.args[0].users) == {P}):
            return P.args[0]
        return None

    def from_mul(mul):
        if not (mul.op == "call_function" and mul.target is MUL
                and len(mul.args) == 2):
            return None
        a, b = mul.args
        for sig, P in ((a, b), (b, a)):
            if (isinstance(sig, torch.fx.Node) and sig.op == "call_function"
                    and sig.target is SIG and sig.args[0] is P
                    and isinstance(P, torch.fx.Node)
                    and set(sig.users) == {mul}):
                mm = underlying_mm(P)
                if mm is not None and set(P.users) == {sig, mul}:
                    return (mm, sig, mul, P)
        return None

    if node.op != "call_function":
        return None
    if node.target is MUL:
        return from_mul(node)
    if node.target is SIG and len(node.users) == 1:
        return from_mul(next(iter(node.users)))
    if node.target is VIEW and len(node.users) in (1, 2):
        for u in node.users:
            if isinstance(u, torch.fx.Node) and u.op == "call_function":
                got = from_mul(u) if u.target is MUL else None
                if got is None and u.target is SIG and len(u.users) == 1:
                    got = from_mul(next(iter(u.users)))
                if got and got[3] is node:
                    return got
    # Rooting the walk AT the producer. `allow_conv` is honoured here for the
    # same reason `from_mul` honours it above: conv is claimed by its own branch
    # rather than by membership, so `in SUPPORTED` alone makes a convolution
    # invisible as a root. Reachable in practice only when the caller hands this
    # the producer directly -- entering from the mul finds the same pattern --
    # but the two spellings must agree: a disagreement between them crashes
    # preprocess, as `conv -> reshape -> relu` would.
    if node.target in MM or (allow_conv
                             and _conv_producer(node, grouped_ok=True)):
        for u in node.users:
            if not isinstance(u, torch.fx.Node):
                continue
            got = from_mul(u) if u.op == "call_function" else None
            if got is None and u.op == "call_function" and u.target is VIEW:
                got = _silu_pattern(u, allow_conv=allow_conv)
            if got and got[0] is node:
                return got
    return None


def _through_view(n):
    """One numel-preserving view hop toward the producer, for gelu/clamp gates whose
    matmul sits under the same reshape the diamond does.

    The VIEW must be single-use as well as the matmul. Checking only the matmul
    would let a clamp on `h = linear(x)` with [B,T,D] x be claimed while h also
    feeds four other ops: the view gate refuses h, the clamp is claimed anyway,
    and preprocess fails with "gate and walk disagree" (xIELU's clamp spelling)."""
    VIEW = exir_ops.edge.aten.view_copy.default
    if (isinstance(n, torch.fx.Node) and n.op == "call_function"
            and n.target is VIEW and isinstance(n.args[0], torch.fx.Node)
            and len(n.args[0].users) == 1 and len(n.users) == 1):
        return n.args[0]
    return n


def _unary_act(node):
    """``(act_fn, slope)`` when `node` is a one-op activation the delegate can
    apply between accumulate and store, else None. The slope is leaky_relu's
    and None for the rest. tanh-GELU is ExecuTorch's own gelu(approximate=
    'tanh') formula, run in the epilogue -- the same op, not a substitute for
    the exact one, which is why it is claimed."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if node.target is exir_ops.edge.aten.gelu.default:
        approx = node.kwargs.get(
            "approximate", node.args[1] if len(node.args) > 1 else "none")
        return {"none": ("gelu", None),
                "tanh": ("gelu_tanh", None)}.get(approx)
    if node.target is exir_ops.edge.aten.sigmoid.default:
        return ("sigmoid", None)
    if node.target is exir_ops.edge.aten.leaky_relu.default:
        slope = node.kwargs.get(
            "negative_slope", node.args[1] if len(node.args) > 1 else 0.01)
        if isinstance(slope, (int, float)) and math.isfinite(slope):
            return ("leaky", float(slope))
    return None


def _act_admits_clamp(node) -> bool:
    """Whether the runtime can clamp after this one-op activation. It applies
    the activation and then the clamp, but leaky ReLU carries its slope in the
    clamp's own field, so a clamp over one has nowhere to go."""
    act = _unary_act(node)
    return act is not None and act[0] != "leaky"


def _claimable_mm(P, allow_conv=False):
    """The matmul (or conv) behind P -- P itself, or the single-use matmul under
    one numel-preserving view of it, the 3-D F.linear trace -- else None. The
    producer test shared by the diamond-shaped activations."""
    VIEW = exir_ops.edge.aten.view_copy.default

    def claimable(n):
        # A matmul or conv: an embedding is a gather, with no accumulator for
        # an epilogue to run on.
        return (isinstance(n, torch.fx.Node) and n.op == "call_function"
                and (n.target in _MM_TARGETS
                     or (allow_conv and _conv_producer(n, grouped_ok=True))))

    if claimable(P):
        return P
    if (isinstance(P, torch.fx.Node) and P.op == "call_function"
            and P.target is VIEW and claimable(P.args[0])
            and set(P.args[0].users) == {P}):
        return P.args[0]
    return None


_TO_DIM_ORDER_COPY = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
_MM_TARGETS = (exir_ops.edge.aten.mm.default, exir_ops.edge.aten.addmm.default)


def _scalar_operand(ep, value):
    """A build-time scalar operand of a decomposed activation.

    Returns ``(ok, number, feeder)``. Export lifts the 3 and the 6 -- of
    torch's hardswish and of a hand-written ``x * relu6(x + 3) / 6`` alike --
    to 0-dim tensor constants behind a ``_to_dim_order_copy`` to fp32; a
    Python number is accepted too. ``feeder`` is that copy node, which the
    gate must claim so the constant lands in the partition. ``number`` is None
    while the partitioner runs: it sees FakeTensors, so a tensor constant's
    value is only readable in preprocess, which writes it to the descriptor.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float,
                                                         torch.fx.Node)):
        return False, None, None
    if not isinstance(value, torch.fx.Node):
        return True, float(value), None
    feeder = None
    node = value
    out = node.meta.get("val")
    if out is None or tuple(out.shape) != () or out.dtype != torch.float32:
        return False, None, None
    if node.op == "call_function" and node.target is _TO_DIM_ORDER_COPY:
        # The copy is what makes an integer literal's int64 constant fp32.
        if len(node.users) != 1:
            return False, None, None
        feeder, node = node, node.args[0]
    if not (isinstance(node, torch.fx.Node) and node.op == "placeholder"
            and ep is not None and _const_tensor(ep, node) is not None):
        return False, None, None
    val = node.meta.get("val")
    if (val is None or tuple(val.shape) != () or val.dtype == torch.bool
            or val.is_complex()):
        return False, None, None
    tensor = _const_tensor(ep, node)
    from torch._subclasses.fake_tensor import FakeTensor
    if isinstance(tensor, FakeTensor):
        return True, None, feeder
    return True, float(tensor.detach().reshape(())), feeder


def _hard_pattern(node, ep=None):
    """torch's hardswish / hardsigmoid decomposition over a matmul output.

        hardsigmoid  div(clamp(add(P, a), 0, 6), d)
        hardswish    div(mul(P, clamp(add(P, a), 0, 6)), d)

    a = 3 and d = 6 in torch's own; whatever they are travels in the
    descriptor. P is the matmul or its single view, every inner edge is
    single-use and the clamp CHAIN composes to exactly [0, 6] (torch emits
    clamp(min=0) then clamp(max=6); relu6 spells it as one hardtanh). From any
    corner -- the scalar feeders included -- returns a dict with kind, mm, P,
    the nodes the gate claims and a/d (None while partitioning, which cannot
    read a tensor constant); else None. Like the SiLU diamond, hardswish's P
    has two users by construction.
    """
    from executorch.backends.esp32s3.operator_support.patterns import (
        _clamp_bounds,
        compose_clamp,
    )

    ADD = exir_ops.edge.aten.add.Tensor
    MUL = exir_ops.edge.aten.mul.Tensor
    DIV = exir_ops.edge.aten.div.Tensor

    def from_div(div):
        if not (isinstance(div, torch.fx.Node) and div.op == "call_function"
                and div.target is DIV and len(div.args) == 2
                and not div.kwargs.get("rounding_mode")):
            return None
        num, divisor = div.args
        ok_d, v_d, f_d = _scalar_operand(ep, divisor)
        if not ok_d:
            return None
        if not (isinstance(num, torch.fx.Node) and len(num.users) == 1):
            return None
        mul = None
        if num.op == "call_function" and num.target is MUL:
            mul = num
        top = num
        if mul is not None:
            if len(mul.args) != 2:
                return None
            a, b = mul.args
            # The clamped factor is whichever side is a clamp; the other is P.
            top, P_mul = (b, a) if _clamp_bounds(b) is not None else (a, b)
            if not (isinstance(top, torch.fx.Node)
                    and len(top.users) == 1):
                return None
        else:
            P_mul = None
        clamps, n, interval = [], top, (-math.inf, math.inf)
        while (isinstance(n, torch.fx.Node) and n.op == "call_function"
               and _clamp_bounds(n) is not None):
            if clamps and len(n.users) != 1:
                return None
            interval = compose_clamp(interval, _clamp_bounds(n))
            clamps.append(n)
            n = n.args[0]
        if not clamps or interval != (0.0, 6.0):
            return None
        add = n
        if not (isinstance(add, torch.fx.Node) and add.op == "call_function"
                and add.target is ADD and len(add.args) == 2
                and add.kwargs.get("alpha", 1) == 1
                and len(add.users) == 1):
            return None
        P, offset = add.args
        ok_a, v_a, f_a = _scalar_operand(ep, offset)
        if not ok_a:
            return None
        mm = _claimable_mm(P)
        if mm is None:
            return None
        if mul is not None:
            if P_mul is not P or set(P.users) != {add, mul}:
                return None
        elif set(P.users) != {add}:
            return None
        kind = "hardswish" if mul is not None else "hardsigmoid"
        inner = [add, *clamps] + ([mul] if mul is not None else []) + [div]
        feeders = [f for f in (f_a, f_d) if f is not None]
        return dict(kind=kind, mm=mm, P=P, div=div, nodes=inner,
                    feeders=feeders, offset=v_a, divisor=v_d)

    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    # Walk down to the div. Every inner node has one user and P has two, so
    # from any corner the div is at most six hops away; wider fan-out is not
    # this pattern and is not explored.
    frontier, seen = [node], set()
    for _ in range(7):
        nxt = []
        for cand in frontier:
            if cand in seen:
                continue
            seen.add(cand)
            got = from_div(cand)
            if got is not None and (node in got["nodes"]
                                    or node in got["feeders"]
                                    or node is got["P"]):
                return got
            if len(cand.users) <= 2:
                nxt.extend(u for u in cand.users
                           if isinstance(u, torch.fx.Node))
        frontier = nxt
    return None


def _relu2_pattern(node):
    """``square(relu(P))`` -- ``pow(r, 2)``, ``torch.square`` or ``r * r`` --
    over a claimable matmul, the relu any clamp chain composing to exactly
    [0, inf). Returns ``(mm, relu_top, square)`` from the square, else None.
    The relu itself is claimed by the clamp gate as usual; preprocess turns
    that interval plus this square into one ReLU^2 activation."""
    from executorch.backends.esp32s3.operator_support.patterns import (
        _clamp_bounds,
        compose_clamp,
    )

    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if node.target is exir_ops.edge.aten.pow.Tensor_Scalar:
        if len(node.args) != 2 or isinstance(node.args[1], bool) \
                or node.args[1] != 2:
            return None
        r = node.args[0]
    elif node.target is exir_ops.edge.aten.mul.Tensor:
        if len(node.args) != 2 or node.args[0] is not node.args[1]:
            return None
        r = node.args[0]
    else:
        return None
    if not (isinstance(r, torch.fx.Node) and set(r.users) == {node}):
        return None
    n, interval = r, (-math.inf, math.inf)
    while (isinstance(n, torch.fx.Node) and n.op == "call_function"
           and _clamp_bounds(n) is not None):
        if n is not r and len(n.users) != 1:
            return None
        interval = compose_clamp(interval, _clamp_bounds(n))
        n = _through_view(n.args[0])
    if n is r or interval != (0.0, math.inf):
        return None
    if not (isinstance(n, torch.fx.Node) and n.op == "call_function"
            and n.target in _MM_TARGETS and len(n.users) == 1):
        return None
    return (n, r, node)


# A build-time scalar whose value cannot be read yet. ExecuTorch partitions a
# FAKE program (backend_api.to_backend -> get_fake_program): parameters and
# persistent buffers are FakeTensors there with no value behind them, and
# float() on one raises GuardOnDataDependentSymNode. The softcap scan visits
# nearly every node, so any `mul` by a 1-element parameter (Gemma 4's
# layer_scalar, a learned lambda) would crash lowering. Lifted literals are
# fake too but keep their value in FakeTensor.constant, which is why literal
# caps can be read.
PENDING_SCALAR = object()


def _scalar_constant(ep, value):
    """Return one build-time scalar as float, None for runtime/non-scalars, or
    PENDING_SCALAR for a 1-element constant whose value is fake right now."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, torch.fx.Node) or ep is None:
        return None
    tensor = _const_tensor(ep, value)
    if tensor is None or tensor.numel() != 1:
        return None
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(tensor, FakeTensor):
        if tensor.constant is None:
            return PENDING_SCALAR
        tensor = tensor.constant
    return float(tensor.detach().reshape(()))


def _softcap_pattern(node, ep):
    """Glimmer's ``cap*tanh((scale*MM)/cap)`` terminal head epilogue.

    Return ``(root, value, scaled, divided, tanh, final, scale, cap)`` from any
    node in the pattern. ``value`` is either the matmul itself or its one
    numel-preserving Linear view. The pre-scale is optional: without it
    ``scaled`` is None and ``scale`` is 1.0. Every edge is single-use, both
    cap constants must be identical and positive, and all scalars must be
    build-time values.
    A scalar held in a parameter is fake while partitioning, so a structural
    match on one comes back with ``scale`` and ``cap`` both None: the gate
    refuses it by name instead of guessing a value it cannot check.
    """
    MUL = exir_ops.edge.aten.mul.Tensor
    DIV = exir_ops.edge.aten.div.Tensor
    TANH = exir_ops.edge.aten.tanh.default
    MM = (exir_ops.edge.aten.mm.default, exir_ops.edge.aten.addmm.default)

    def split_scalar(n, target, *, commutative=True):
        if not (isinstance(n, torch.fx.Node) and n.op == "call_function"
                and n.target is target and len(n.args) >= 2):
            return None
        pairs = [(n.args[0], n.args[1])]
        if commutative:
            pairs.append((n.args[1], n.args[0]))
        for value, scalar in pairs:
            got = _scalar_constant(ep, scalar)
            if isinstance(value, torch.fx.Node) and got is not None:
                return value, got
        return None

    def from_final(final):
        tail = split_scalar(final, MUL)
        if tail is None:
            return None
        tanh, cap_out = tail
        if not (tanh.op == "call_function" and tanh.target is TANH
                and len(tanh.users) == 1):
            return None
        divided = tanh.args[0]
        div = split_scalar(divided, DIV, commutative=False)
        if div is None or len(divided.users) != 1:
            return None
        scaled, cap_in = div
        pre = split_scalar(scaled, MUL)
        if pre is None:
            # Gemma 2/3n's spelling, tanh(z / cap) * cap, has no output
            # multiplier. A descriptor scale of 1.0 is the same operation:
            # the runtime's y *= 1.0f changes no bit.
            value, scale, scaled = scaled, 1.0, None
        elif len(scaled.users) != 1:
            return None
        else:
            value, scale = pre
        root = _through_view(value)
        if not (isinstance(root, torch.fx.Node) and root.op == "call_function"
                and root.target in MM and len(value.users) == 1):
            return None
        if any(s is PENDING_SCALAR for s in (scale, cap_in, cap_out)):
            return (root, value, scaled, divided, tanh, final, None, None)
        if not (np.isfinite(scale) and np.isfinite(cap_in)
                and cap_in > 0.0 and cap_in == cap_out):
            return None
        return (root, value, scaled, divided, tanh, final, scale, cap_in)

    if not isinstance(node, torch.fx.Node):
        return None
    frontier, seen = [node], set()
    for _ in range(6):
        nxt = []
        for candidate in frontier:
            if candidate in seen:
                continue
            seen.add(candidate)
            got = from_final(candidate)
            if got is not None and node in got[:6]:
                return got
            nxt.extend(u for u in candidate.users if isinstance(u, torch.fx.Node))
        frontier = nxt
    return None


def _tail_root_mm(a, allow_conv: bool = False):
    """The matmul (or conv) under a single-use fusion chain ending at `a`.

    Mirrors what the epilogue gates claim, in the runtime's epilogue order read
    from the tail down: clamp-family links, then at most one activation -- a
    one-op activation (`_unary_act`), the silu diamond's mul corner or ReLU^2's
    square -- then at most one view directly over the producer, every link
    single-use. An activation is claimed only DIRECTLY over its producer, so
    nothing but the producer may sit beneath one: gelu(clamp(x@W)) is an
    act-over-clamp the runtime cannot represent. Beneath a clamp only an
    activation the clamp gate also admits there (`_act_admits_clamp`) extends
    the tail; anything else ends the walk, because the gates would leave that
    chain portable and a residual or multiply claimed on top of it would reach
    preprocess with no tail to ride. Embeddings are deliberately not roots --
    the runtime's residual add lives in the matmul epilogue, and a gather has
    no epilogue."""
    from executorch.backends.esp32s3.operator_support.patterns import _clamp_bounds

    MM = (exir_ops.edge.aten.mm.default, exir_ops.edge.aten.addmm.default)
    VIEW = exir_ops.edge.aten.view_copy.default
    n = a
    past_clamp = False
    for _ in range(16):  # chains are short; this is a cycle guard, not a budget
        if not (isinstance(n, torch.fx.Node) and n.op == "call_function"):
            return None
        if n.target in MM:
            return n
        # A conv is a root too when the caller allows it. The delegate applies a
        # residual in the same epilogue for both, so the only question is whether
        # the OTHER half of the pairing is in place -- see the add gate.
        # No grouped_ok here, deliberately: this is the root test the RESIDUAL
        # walk uses, and a residual cannot ride on a grouped conv (the scatter
        # indexes into a group-offset destination the residual does not share
        # an origin with).
        if allow_conv and _conv_producer(n):
            return n
        # The silu diamond's mul corner is the one activation reached here:
        # below it the diamond polices its own structure. A clamp over it is
        # act-then-clamp, which the clamp gate admits too.
        pat = _silu_pattern(n, allow_conv=allow_conv)
        if pat is not None and n is pat[2]:
            return pat[0]
        # ReLU^2 turns its relu's interval into the activation itself, so
        # preprocess has no clamp field left for one above the square.
        square = None if past_clamp else _relu2_pattern(n)
        if square is not None:
            return square[0]
        if n.target is VIEW:
            # Only the linear-trace position -- a reshape directly over the matmul.
            # That is the one place the view GATE will also claim it; accepting a
            # view anywhere else would make the gate and this walk disagree about
            # what lands in the partition.
            src = n.args[0]
            # A CONV is a valid source here too: conv is claimed by its own
            # branch rather than by membership in MM, so without it
            # `conv(x).reshape(s) + r.reshape(s)` would find no root and the
            # whole residual would stay portable -- four nodes, on a spelling
            # anyone writing a flattened residual produces.
            if (isinstance(src, torch.fx.Node) and src.op == "call_function"
                    and len(src.users) == 1
                    and (src.target in MM
                         or (allow_conv and _conv_producer(src)))):
                return src
            return None
        # The runtime applies the activation and THEN the clamp, which is
        # exactly what the clamp and activation gates admit: walking down from
        # the tail, any clamp links, then at most one activation directly over
        # the producer. Rooting gelu(clamp(x@W)) here would leave its gelu
        # portable, and preprocess would fail the export on the claimed add.
        if _clamp_bounds(n) is not None:
            prev = n.args[0]
            if not (isinstance(prev, torch.fx.Node) and len(prev.users) == 1):
                return None
            past_clamp = True
            n = prev
            continue
        if _unary_act(n) is not None and (not past_clamp
                                          or _act_admits_clamp(n)):
            prev = n.args[0]
            if not (isinstance(prev, torch.fx.Node) and len(prev.users) == 1):
                return None
            src = _through_view(prev)
            if (isinstance(src, torch.fx.Node) and src.op == "call_function"
                    and len(src.users) == 1
                    and (src.target in MM
                         or (allow_conv and _conv_producer(src)))):
                return src
            return None
        return None
    return None


def _is_symbolic_shaped(node) -> bool:
    """True if any tensor this node produces carries a non-int (SymInt) dim.

    The delegate is a STATIC-SHAPE backend and says so here rather than
    discovering it on device: `OpDesc` stores m/k/n as uint32 resolved during
    preprocess, weights are packed and padded to exactly that k_pad, and every
    scratch buffer (xq, wu, chain slots) is sized once at init from those
    extents. A symbolic dim has no value to write into any of them, so
    claiming such a node would silently bake whatever the tracing example
    happened to be -- correct on the example input and wrong on the next one.
    Refusing leaves the node on ExecuTorch's portable kernels, which DO handle
    dynamic shapes, so a model that is dynamic in part still lowers and runs;
    only the dynamic subgraphs stay undelegated.
    """
    val = node.meta.get("val")
    if val is None:
        return False
    for v in (val if isinstance(val, (list, tuple)) else [val]):
        shape = getattr(v, "shape", None)
        if shape is None:
            continue
        for dim in shape:
            if not isinstance(dim, int):
                return True
    return False


def _leads_to_softmax(node, hops: int = 6) -> bool:
    """Whether this value's consumer chain reaches _softmax within a few
    single-user hops through scale-and-mask plumbing (views, the 1/sqrt(d)
    scalar, additive masks). This is the test that tells q@kT apart from p@v:
    scores feed a softmax, attention outputs do not."""
    PASSTHRU = (
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.mul.Scalar,
        exir_ops.edge.aten.div.Scalar,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.add.Tensor,
        # Blocked prefill flattens token and GQA-head axes for the Q@K BMM,
        # then restores [token, head, key] before masking. This permutation is
        # score plumbing just like the adjacent views; P@V still cannot pass
        # the defining reaches-softmax test.
        exir_ops.edge.aten.permute_copy.default,
        # Causal/local masks are also commonly written as
        # where(valid, qk_scores, -inf). The q@k producer is still the value
        # being quantised; where only selects or masks it before softmax.
        exir_ops.edge.aten.where.self,
        exir_ops.edge.aten.clone.default,
        # How a clone or .contiguous() appears in a dim-order edge graph:
        # blocked prefill with more than one KV head restores [token, head,
        # key] through a reshape that needs a copy, and the scores must
        # still count as reaching the softmax.
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    )
    SOFTMAX = exir_ops.edge.aten._softmax.default
    # Pure data movement between the score ops does not spend a hop: the
    # reshapes, transposes and copies that restore a layout carry no score
    # arithmetic (KH > 1 prefill takes view, permute, copy, view in a row).
    MOVES = {
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }
    n = node
    steps = 0
    while hops > 0 and steps < 4 * hops + 16:
        steps += 1
        users = [u for u in n.users if isinstance(u, torch.fx.Node)
                 and u.op == "call_function"]
        # F.scaled_dot_product_attention's decomposition (ATen _safe_softmax)
        # reads the softmax INPUT a second time, eq(t, -inf), to zero fully
        # masked rows. sdpa_to_qk_pass.py keeps that guard everywhere (no row
        # is provably finite), so it counts as a probe of the scores, not a
        # second consumer that makes them p@v: q@kT stays identifiable.
        softmax = next((u for u in users if u.target is SOFTMAX), None)
        if softmax is not None and len(users) == 2:
            from executorch.backends.esp32s3._passes.sdpa_to_qk_pass import safe_softmax_guard

            guard = safe_softmax_guard(softmax)
            if guard is not None and guard["eq"] in users:
                return True
        if len(users) != 1:
            return False
        u = users[0]
        if u.target is SOFTMAX:
            return True
        if u.target not in PASSTHRU:
            return False
        if u.target not in MOVES:
            hops -= 1
        n = u
    return False


def _qk_bmm_structure(node):
    """If `node` is  bmm(A, permute(X, (0,2,1)))  with 3-D operands agreeing on
    batch and inner dim, return (A, P, X); else None. Structure only -- the
    softmax-consumer test lives in _qk_bmm_pattern, because preprocess re-checks
    structure INSIDE the partition subprogram, where the softmax (a portable op)
    no longer exists to be found. The permute must be single-use because it is
    FOLDED: the delegate reads X in its natural [B,N,K] layout."""
    BMM = exir_ops.edge.aten.bmm.default
    PERM = exir_ops.edge.aten.permute_copy.default
    if not (node.op == "call_function" and node.target is BMM):
        return None
    A, P = node.args[:2]
    if not (isinstance(A, torch.fx.Node) and isinstance(P, torch.fx.Node)
            and P.op == "call_function" and P.target is PERM
            and list(P.args[1]) == [0, 2, 1] and len(P.users) == 1):
        return None
    X = P.args[0]
    if not isinstance(X, torch.fx.Node):
        return None
    av, xv = A.meta.get("val"), X.meta.get("val")
    if (av is None or xv is None
            or len(av.shape) != 3 or len(xv.shape) != 3
            or int(av.shape[0]) != int(xv.shape[0])
            or int(av.shape[2]) != int(xv.shape[2])):
        return None
    return (A, P, X)


_KCACHE_CASTS = (_TO_DIM_ORDER_COPY, exir_ops.edge.aten._to_copy.default)
_KCACHE_SCALE_VIEWS = (exir_ops.edge.aten.view_copy.default,
                       exir_ops.edge.aten.unsqueeze_copy.default)


def _kcache_keys(X):
    """If the q@kT key operand `X` dequantises an int8 K cache, its parts.

    The spelling is kcache_i8.dequantized_keys after export,

        mul(cast(codes[B,N,K] int8 -> fp32), view(scales[B,N] fp32 -> [B,N,1]))

    in either operand order, with every link single-use: the delegate reads the
    codes and scales in place and never materialises the fp32 keys, so nothing
    else may consume them. Returns {codes, scales, cast, view, mul} or None.
    """
    MUL = exir_ops.edge.aten.mul.Tensor
    if not (isinstance(X, torch.fx.Node) and X.op == "call_function"
            and X.target is MUL and len(X.args) == 2 and not X.kwargs
            and len(X.users) == 1):
        return None
    for cast, view in (X.args, X.args[::-1]):
        if not (isinstance(cast, torch.fx.Node) and isinstance(view, torch.fx.Node)
                and cast.op == "call_function" and cast.target in _KCACHE_CASTS
                and view.op == "call_function"
                and view.target in _KCACHE_SCALE_VIEWS
                and len(cast.users) == 1 and len(view.users) == 1):
            continue
        codes, scales = cast.args[0], view.args[0]
        if not (isinstance(codes, torch.fx.Node)
                and isinstance(scales, torch.fx.Node)):
            continue
        cv, sv = codes.meta.get("val"), scales.meta.get("val")
        castv, viewv = cast.meta.get("val"), view.meta.get("val")
        if any(v is None for v in (cv, sv, castv, viewv)):
            continue
        dim_order = cast.kwargs.get("dim_order")
        if (cv.dtype != torch.int8 or castv.dtype != torch.float32
                or sv.dtype != torch.float32 or viewv.dtype != torch.float32
                or len(cv.shape) != 3 or len(sv.shape) != 2
                or (dim_order is not None
                    and list(dim_order) != list(range(len(cv.shape))))):
            continue
        b, n, _ = (int(d) for d in cv.shape)
        if ([int(d) for d in sv.shape] != [b, n]
                or [int(d) for d in viewv.shape] != [b, n, 1]):
            continue
        return dict(codes=codes, scales=scales, cast=cast, view=view, mul=X)
    return None


def _kcache_member_bmm(node):
    """The q@kT bmm whose int8-cache key dequantisation `node` belongs to.

    For the cast, the scale view and the multiply of a `_kcache_keys` operand
    that reaches `bmm(A, permute(X, (0,2,1)))`; None for anything else. Both
    the gate and the preprocess walk ask this, so they cannot disagree about
    which nodes fold into OP_BMM_KCACHE_I8.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    mul = node
    if node.target in _KCACHE_CASTS or node.target in _KCACHE_SCALE_VIEWS:
        if len(node.users) != 1:
            return None
        mul = next(iter(node.users))
    if not (isinstance(mul, torch.fx.Node) and len(mul.users) == 1):
        return None
    permute = next(iter(mul.users))
    if len(permute.users) != 1:
        return None
    bmm = next(iter(permute.users))
    got = _qk_bmm_structure(bmm)
    if got is None or got[1] is not permute or got[2] is not mul:
        return None
    parts = _kcache_keys(mul)
    if parts is None or node not in (parts["cast"], parts["view"], mul):
        return None
    return bmm


def _kcache_affine_keys(X):
    """If the q@kT key operand `X` reads ExecuTorch's stock affine K cache,
    its parts: the spelling stock_llm.rewrite_kcache leaves,

        espressif.dequantize_per_token_i8(codes [B,N,K] int8,
                                          scales [B,N,1] fp32,
                                          zero_points [B,N,1] int8)

    single-use, feeding the bmm's folded permute. The delegate reads the
    three in place (OP_BMM_KCACHE_AFFINE_I8) and never materialises the
    fp32 keys. Returns {codes, scales, zps, dq} or None."""
    op = getattr(X, "target", None)
    schema = getattr(getattr(op, "_op", op), "_schema", None)
    if not (isinstance(X, torch.fx.Node) and X.op == "call_function"
            and schema is not None
            and schema.name == "espressif::dequantize_per_token_i8"
            and len(X.args) == 3 and not X.kwargs and len(X.users) == 1):
        return None
    codes, scales, zps = X.args
    if not all(isinstance(n, torch.fx.Node) for n in (codes, scales, zps)):
        return None
    cv, sv, zv = (n.meta.get("val") for n in (codes, scales, zps))
    if (any(v is None for v in (cv, sv, zv)) or cv.dtype != torch.int8
            or sv.dtype != torch.float32 or zv.dtype != torch.int8
            or len(cv.shape) != 3):
        return None
    b, n, _ = cv.shape
    if list(sv.shape) != [b, n, 1] or list(zv.shape) != [b, n, 1]:
        return None
    return dict(codes=codes, scales=scales, zps=zps, dq=X)


def _kcache_affine_bmm(node):
    """(the q@kT structure, the affine K-cache parts) when `node` is the bmm
    of an OP_BMM_KCACHE_AFFINE_I8, else None. Structure only: these keys are
    keys by provenance -- the transposed read of a K cache -- so neither the
    softmax-consumer test nor quantize_qk is asked for."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target is exir_ops.edge.aten.bmm.default):
        return None
    got = _qk_bmm_structure(node)
    if got is None:
        return None
    parts = _kcache_affine_keys(got[2])
    return None if parts is None else (got, parts)


def _kcache_affine_member_bmm(node):
    """The affine K-cache bmm whose folded permute or dequantisation `node`
    is, else None (and None for the bmm itself)."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and len(node.users) == 1):
        return None
    user = next(iter(node.users))
    if node.target is exir_ops.edge.aten.permute_copy.default:
        got = _kcache_affine_bmm(user)
        return user if got is not None and got[0][1] is node else None
    if _kcache_affine_keys(node) is not None and len(user.users) == 1:
        bmm = next(iter(user.users))
        got = _kcache_affine_bmm(bmm)
        return bmm if got is not None and got[0][2] is node else None
    return None


def _weight_side_const(node, hops=4):
    """True if `node` traces back to a placeholder through reshapes only.

    The cheap proxy for "this operand is a weight, not an activation", usable
    from the partitioner where the ExportedProgram (and so `_resolve_const`) is
    not in hand. Single-use links only: a reshape shared with something outside
    the partition is not exclusively ours to fold.
    """
    # expand_copy belongs here: a 2-D weight feeding a BATCHED matmul is
    # broadcast to [1,K,N] first, so the chain from a bmm back to its weight is
    # view -> expand -> permute -> placeholder; without it nothing batched
    # would match.
    RESHAPE = (exir_ops.edge.aten.view_copy.default,
               exir_ops.edge.aten.permute_copy.default,
               exir_ops.edge.aten.expand_copy.default)
    for _ in range(hops):
        if not isinstance(node, torch.fx.Node):
            return False
        if node.op == "placeholder":
            return True
        if not (node.op == "call_function" and node.target in RESHAPE
                and len(node.users) == 1):
            return False
        node = node.args[0]
    return False


def _bmm_is_linear(node):
    """A batch-1 `bmm` against a constant operand: an `mm` wearing a batch dim.

    `bmm(x[1,M,K], W[1,K,N])` computes exactly what `mm(x[M,K], W[K,N])`
    computes -- same arithmetic, same weights, same bytes -- but this backend
    claims `mm` and not `bmm`, so the identical model loses every one of these.

    Which spelling torch emits is NOT under the model author's control: the same
    `nn.Linear` lowers to `mm` in one graph and to a batch-1 `bmm` in another
    depending on what feeds it. Measured on a speech encoder, where the
    q/k/v projections and fc1 came out as `bmm` and o_proj and fc2 as `mm` --
    24 of 36 projections lost to a spelling difference.

    Reported by name rather than left in the generic bucket because those two
    read identically ("op not in the supported set") and mean completely
    different things: one is an op with no kernel, the other is an op this
    backend already implements under another name.
    """
    if node.target is not exir_ops.edge.aten.bmm.default:
        return False
    a, b = node.args[:2]
    if not (isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)):
        return False
    av, bv = a.meta.get("val"), b.meta.get("val")
    if av is None or bv is None:
        return False
    if len(av.shape) != 3 or len(bv.shape) != 3:
        return False
    if int(av.shape[0]) != 1 or int(bv.shape[0]) != 1:
        return False
    if int(av.shape[2]) != int(bv.shape[1]):
        return False
    return _weight_side_const(b)


def _qk_bmm_pattern(node):
    """The partitioner's gate: q@kT structure AND its scores reach a softmax."""
    got = _qk_bmm_structure(node)
    if got is None or not _leads_to_softmax(node):
        return None
    return got


def _selected_mm_pattern(node, exported_program=None):
    """Match ``x @ index_select(W[V,K], 0, rows).T``.

    The full table must be a build-time constant, while ``rows`` is a static-
    shape int64 runtime tensor.  Both folded nodes are single-use because the
    delegate does not materialize the gathered table.  The returned dimensions
    distinguish the stored vocabulary ``V`` from the result width ``R``.
    """
    MM = exir_ops.edge.aten.mm.default
    PERM = exir_ops.edge.aten.permute_copy.default
    SELECT = exir_ops.edge.aten.index_select.default
    if not (isinstance(node, torch.fx.Node)
            and node.op == "call_function" and node.target is MM):
        return None
    act, permute = node.args[:2]
    if not (isinstance(act, torch.fx.Node)
            and isinstance(permute, torch.fx.Node)
            and permute.op == "call_function" and permute.target is PERM
            and list(permute.args[1]) == [1, 0]
            and len(permute.users) == 1):
        return None
    select = permute.args[0]
    if not (isinstance(select, torch.fx.Node)
            and select.op == "call_function" and select.target is SELECT
            and len(select.users) == 1 and select.args[1] == 0):
        return None
    table, rows = select.args[0], select.args[2]
    if not (isinstance(table, torch.fx.Node)
            and isinstance(rows, torch.fx.Node)):
        return None
    # A plain constant table, or quantised codes whose dequant (or
    # float-offset chain) IS the table node: preprocess packs those verbatim,
    # zero points and all. A pass-through behind a folded op has no defined
    # meaning for the gathered rows and stays portable.
    if (exported_program is None
            or not _is_build_time_const(exported_program, table)
            or (_is_pass_through_const_path(table)
                and _pass_through_source(table) is not table)):
        return None
    # Static shapes only. Every extent below is baked into the descriptor and
    # the SelectedRows record, and `int()` on a backed SymInt specialises to the
    # tracing example without complaint -- the gate's dynamic-shape refusal
    # does not see this pattern, because the selected branch returns first.
    if any(_is_symbolic_shaped(n)
           for n in (node, act, table, rows, select, permute)):
        return None
    av = act.meta.get("val")
    tv = table.meta.get("val")
    rv = rows.meta.get("val")
    ov = node.meta.get("val")
    if any(value is None for value in (av, tv, rv, ov)):
        return None
    if (len(av.shape) != 2 or len(tv.shape) != 2 or len(rv.shape) != 1
            or len(ov.shape) != 2):
        return None
    m, k = (int(value) for value in av.shape)
    vocab, table_k = (int(value) for value in tv.shape)
    selected = int(rv.shape[0])
    if (k != table_k or tuple(int(value) for value in ov.shape)
            != (m, selected)):
        return None
    return {
        "mm": node, "permute": permute, "select": select,
        "act": act, "table": table, "rows": rows,
        "m": m, "k": k, "vocab": vocab, "selected": selected,
    }


def _selected_mm_from_member(node, exported_program=None):
    """Return a selected matmul pattern when ``node`` is either folded member."""
    if not isinstance(node, torch.fx.Node):
        return None
    candidates = []
    if node.target is exir_ops.edge.aten.mm.default:
        candidates = [node]
    elif len(node.users) == 1:
        first = next(iter(node.users))
        if first.target is exir_ops.edge.aten.mm.default:
            candidates = [first]
        elif len(first.users) == 1:
            second = next(iter(first.users))
            if second.target is exir_ops.edge.aten.mm.default:
                candidates = [second]
    for candidate in candidates:
        pattern = _selected_mm_pattern(candidate, exported_program)
        if pattern is not None and node in (
                pattern["select"], pattern["permute"], pattern["mm"]):
            return pattern
    return None


# ---- clamp canonicalisation -------------------------------------------------
def _clamp_bounds(node):
    """(lo, hi) as floats with +/-inf for open sides, or None if not clamp-family
    with scalar bounds. relu6 never appears by that name -- it exports as
    hardtanh(0, 6) -- which is exactly why the family is matched by bounds rather
    than by op identity."""
    import math

    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    t = node.target
    if t == exir_ops.edge.aten.relu.default:
        return (0.0, math.inf)
    if t == exir_ops.edge.aten.hardtanh.default:
        args = list(node.args) + [-1.0, 1.0][len(node.args) - 1:]
        return (float(args[1]), float(args[2]))
    if t == exir_ops.edge.aten.clamp.default:
        lo = node.args[1] if len(node.args) > 1 else None
        hi = node.args[2] if len(node.args) > 2 else None
        if isinstance(lo, torch.fx.Node) or isinstance(hi, torch.fx.Node):
            return None  # tensor bounds: not a scalar interval
        return (float(lo) if lo is not None else -math.inf,
                float(hi) if hi is not None else math.inf)
    return None


def compose_clamp(outer, inner):
    """Interval of clamp_outer(clamp_inner(x)), exact.

    Derived piecewise, and NOT the naive (max(lo), min(hi)): for x below l1 the
    inner pins to l1 and the outer then clamps THAT, so each resulting bound is the
    inner bound passed through the outer clamp --

        lo' = clamp(l1, l2, h2)      hi' = clamp(h1, l2, h2)

    which also gets the crossed cases right (l2 > h1 collapses to a constant, with
    lo' == hi'). Pure min/max: composing then applying is BIT-IDENTICAL to applying
    twice, which is what lets the canonicaliser claim exactness.
    """
    l1, h1 = inner
    l2, h2 = outer
    c = lambda v: min(max(v, l2), h2)  # noqa: E731
    return (c(l1), c(h1))
