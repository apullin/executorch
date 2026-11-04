# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolving a matmul's weight: constants, folded views and pre-quantised dequant chains."""

import contextlib
from typing import List

import numpy as np
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export import ExportedProgram

from executorch.backends.esp32s3.serialization.espressif_blob import BACKEND_NAME


# nn.Linear lowers to permute_copy(weight) + addmm, so the weight arrives as a computed
# value rather than a constant. Claiming the permute lets us apply it at build time,
# which deletes the transpose from the runtime entirely. Folded, never emitted.
FOLDED_PERMUTE = exir_ops.edge.aten.permute_copy.default
FOLDED_SLICE = exir_ops.edge.aten.slice_copy.Tensor
FOLDED = {FOLDED_PERMUTE, FOLDED_SLICE}


def _const_tensor(ep: ExportedProgram, node: torch.fx.Node):
    """The tensor behind a placeholder, or None if it is a runtime activation."""
    # Only placeholders are graph inputs, so nothing else can name one. Said
    # first because each signature lookup below rebuilds a dict of every input
    # spec, and the softcap gate asks this of ordinary nodes ~11,000 times per
    # story Glimmer partition: most of its partitioning time.
    if node.op != "placeholder":
        return None
    sources = _placeholder_sources(ep)
    if sources is not None:
        source = sources.get(node.name)
        if source is None:
            return None
        in_constants, target = source
        return (ep.constants if in_constants else ep.state_dict)[target]
    if is_param(ep, node):
        return get_param(ep, node)
    if is_buffer(ep, node):
        return get_buffer(ep, node)
    if is_lifted_tensor_constant(ep, node):
        return get_lifted_tensor_constant(ep, node)
    return None


# While partition() runs: [program, its signature, the signature's input_specs
# list, that list's length, {placeholder name: (in_constants, target)}].
_PLACEHOLDERS = None


@contextlib.contextmanager
def _placeholder_index(ep: ExportedProgram):
    """Answer _const_tensor's placeholder lookups on `ep` from one map.

    For partition(), whose gates ask about the same weights over and over:
    torch's is_param/get_param/is_buffer/... rebuild a dict of every input
    spec on each call, so each lookup costs O(inputs), and partitioning a
    graph is thousands of lookups. The map is built on first use and again
    whenever the signature object or its spec list is replaced (or the list
    changes length); partition() itself never edits either. The idea of
    ExecuTorch #22164, which caches these names in an Arm pass.
    """
    global _PLACEHOLDERS
    saved, _PLACEHOLDERS = _PLACEHOLDERS, [ep, None, None, -1, None]
    try:
        yield
    finally:
        _PLACEHOLDERS = saved


def _placeholder_sources(ep: ExportedProgram):
    """The active map for `ep`, or None when partition() is not indexing it."""
    state = _PLACEHOLDERS
    if state is None or state[0] is not ep:
        return None
    sig = ep.graph_signature
    specs = sig.input_specs
    if state[1] is not sig or state[2] is not specs or state[3] != len(specs):
        state[1:] = [sig, specs, len(specs), _signature_sources(sig)]
    return state[4]


def _signature_sources(sig):
    """{placeholder name: (in_constants, target)}: exactly what get_param,
    get_buffer and get_lifted_tensor_constant read, in their precedence."""
    from torch.export.graph_signature import InputKind, TensorArgument

    non_persistent = set(sig.non_persistent_buffers)
    sources = {}
    for kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
        for spec in sig.input_specs:
            if not (spec.kind == kind and isinstance(spec.arg, TensorArgument)
                    and isinstance(spec.target, str)):
                continue
            in_constants = (kind == InputKind.CONSTANT_TENSOR
                            or (kind == InputKind.BUFFER and spec.target in non_persistent))
            sources.setdefault(spec.arg.name, (in_constants, spec.target))
    return sources


class PassThroughWeight:
    """A weight that arrives ALREADY QUANTISED in the graph, to be consumed verbatim.

    Produced when `_resolve_const` meets `torchao::dequantize_affine` over constant
    codes -- the shape `Int8DynActInt4WeightQuantizer` leaves after export. Carrying
    this through instead of the dequantised fp32 is the whole point: re-quantising a
    quantised tensor is strictly destructive, while the codes and scales as-given are
    lossless to keep.

    `permuted` tracks the folded transpose the same way the fp32 path folds it: codes
    are stored [N, K] with per-row groups, the graph permutes to [K, N] for mm, and
    the packer wants [N, K] back -- so exactly one folded permute means the codes are
    already in packing orientation.

    Asymmetric weights (EXT_WEIGHT_AFFINE) carry more, normalised by
    asym_records.normalize_weight: `codes` are then the SIGNED stored codes
    c = q - storage_bias, `zero_points` their integer zero points in the same
    domain, `float_zero_points` a fractional zero point on q (embeddings only),
    `offsets` a float offset added after each group's scale (W-off), each shaped
    like `scales`. A symmetric weight leaves all four at None/0 and packs as
    plain codes and scales.
    """

    def __init__(self, codes, scales, gsize, qmin, qmax, *, zero_points=None,
                 float_zero_points=None, offsets=None, storage_bias=0):
        self.codes = codes          # int8 container, values in [qmin, qmax]
        self.scales = scales        # [N] or [N, n_groups] fp32
        self.gsize = gsize          # elements per scale group (K = per-row)
        self.qmin, self.qmax = qmin, qmax
        self.zero_points = zero_points              # int64 tensor or None
        self.float_zero_points = float_zero_points  # fp32 tensor or None
        self.offsets = offsets                      # fp32 tensor or None
        self.storage_bias = storage_bias            # 0, 8 or 128
        # Described from the graph's structure alone, values unread: what the
        # partition gate sees of a weight whose constants are FakeTensors.
        self.structural = False
        self.permuted = False
        # A tied output head may be a PREFIX of a larger embedding allocation.
        # Keep the full storage so content-addressed PTD data deduplicates with
        # the embedding, while the descriptor exposes only these logical rows.
        self.logical_rows = None
        # A frontend can pad a ragged final scale group for torchao dequant and
        # slice it back to the true K. Padded columns are not shared storage, so
        # preprocess packs only this logical prefix.
        self.logical_cols = None
        # The shape the codes arrived in. A convolution weight is quantised per
        # output channel as [C_out, C_in/g, (kd,) kh, kw] and is reshaped to
        # [N, K] on the way in; preprocess checks this against the conv's
        # geometry so a mis-oriented table cannot pack quietly.
        self.source_shape = None
        # Codes of a TRANSPOSED convolution's [C_in, C_out, ...] weight,
        # turned on the way in into its dense equivalent's [C_out, C_in, ...]
        # with the taps flipped (the transform _conv_weight applies to a
        # float weight), so rows are output channels here too.
        self.transposed = False

    @property
    def affine(self) -> bool:
        """Whether this weight needs EXT_WEIGHT_AFFINE."""
        return (self.zero_points is not None
                or self.float_zero_points is not None
                or self.offsets is not None or self.storage_bias != 0)


def _slice_pass_through(pt: "PassThroughWeight", lo: int, hi: int):
    """Rows [lo, hi) of a pass-through weight, for one group of a grouped conv."""
    def rows(table):
        return None if table is None else table[lo:hi]

    out = PassThroughWeight(pt.codes[lo:hi], pt.scales[lo:hi], pt.gsize,
                            pt.qmin, pt.qmax, zero_points=rows(pt.zero_points),
                            float_zero_points=rows(pt.float_zero_points),
                            offsets=rows(pt.offsets),
                            storage_bias=pt.storage_bias)
    out.permuted = pt.permuted
    out.source_shape = pt.source_shape
    out.transposed = pt.transposed
    return out


def _pass_through_source(node):
    """The dequant node (or float-offset chain root) at the bottom of a folded
    weight path, or None."""
    while (isinstance(node, torch.fx.Node) and node.op == "call_function"
           and node.target in FOLDED):
        node = node.args[0]
    if _is_torchao_dequant(node) or _w_off_pattern(node) is not None:
        return node
    return None


def _is_torchao_affine(node) -> bool:
    """torchao's `dequantize_affine`, matched by its exact schema name, which
    both the aten OpOverload and the edge dialect's wrapper report. A substring
    match would also take `dequantize_affine_float8_non_decomposed(tensor, scale,
    dtype)` and read its three arguments as the integer op's seven."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return False
    schema = getattr(node.target, "_schema", None)
    return getattr(schema, "name", None) == "torchao::dequantize_affine"


def _pt2e_dequant_kind(node):
    """"per_channel", "per_channel_group" or "per_tensor" for a PT2E
    (quantized_decomposed) dequant, else None.

    `dequantize_per_channel` is a strict prefix of
    `dequantize_per_channel_group`, whose signature is (codes, scales, zp, qmin,
    qmax, dtype, group_size) -- so the group spelling is tested first, or its
    args[3] would be read as `axis` when it is really `quant_min`. The
    per-tensor op has two overloads the delegate reads (.default with a float
    scale and int zero point, .tensor with 0-d tensors); .tensor2's range
    tensors are not taken.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    target = str(node.target)
    if "quantized_decomposed" not in target:
        return None
    if "dequantize_per_channel_group" in target:
        return "per_channel_group"
    if "dequantize_per_channel" in target:
        return "per_channel"
    if "dequantize_per_tensor" in target and "tensor2" not in target:
        return "per_tensor"
    return None


def _is_torchao_dequant(node) -> bool:
    """A weight-side dequant the delegate can absorb, in any spelling.

    Every producer here means "these codes ARE the weights":

      torchao  `dequantize_affine`               -- the quantize_ API path
                                                    (Int8DynActInt4WeightQuantizer,
                                                    IntxWeightOnlyConfig)
      PT2E     `dequantize_per_channel`,         -- what convert_pt2e emits for a
               `dequantize_per_channel_group`,      weight spec: per channel
               `dequantize_per_tensor`              (Esp32Quantizer, XNNPACK, Arm),
                                                    per group, per tensor
                                                    (XNNPACK is_per_channel=False,
                                                    Cortex-M Linear)

    Zero points ride along (EXT_WEIGHT_AFFINE). An activation-side
    dequant reads a quantize NODE, not constant codes, so it is never claimed:
    the gate asks _is_build_time_const of the whole dequant first.
    """
    return _is_torchao_affine(node) or _pt2e_dequant_kind(node) is not None


class LoweringRefused(RuntimeError):
    """A weight preprocess cannot lower, with the refusal kind the gate reports
    for it. The gate runs the same checks first (Esp32OperatorSupport.
    _lowering_problem), so a model meeting one of these keeps that op portable
    instead of failing the whole lowering: a gate that claims what preprocess
    then refuses (an Arm int32 bias, say) crashes the export."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{BACKEND_NAME}: {kind}: {detail}")
        self.kind = kind
        self.detail = detail


def _resolved_zero_points(zeros, arg):
    """The zero point a dequant names: a tensor, a Python number, or None.

    A zero-point NODE that does not resolve to a constant is not proof of
    zero: `_resolve_const` returns None both for "no zero-point" and for "a
    zero-point this pass cannot fold", and treating the second as the first
    would silently turn the runtime dot from sum(x*w) - zp*sum(x) into
    sum(x*w). Unresolvable-but-present is refused by name.
    """
    if zeros is None and isinstance(arg, torch.fx.Node):
        raise LoweringRefused(
            "weight zero point not constant",
            f"{arg.name} cannot be evaluated at build time; it cannot be "
            f"assumed zero")
    if isinstance(zeros, PassThroughWeight):
        raise LoweringRefused("weight zero point not constant",
                              "a zero point that is itself quantised")
    return zeros


def _as_numpy(value):
    """A constant tensor (or number) as numpy, for the weight checks."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concrete(value):
    """A real tensor for `value`: itself, or -- while partitioning, where
    ExecuTorch hands the gate FakeTensors for parameters and buffers
    (get_fake_program) -- the real tensor that FakeTensor was made from,
    which the real program still holds. None when it cannot be found, and the
    caller then judges by structure alone. Non-tensors pass through."""
    if not isinstance(value, torch.Tensor):
        return value
    from torch._subclasses.fake_tensor import FakeTensor

    if not isinstance(value, FakeTensor):
        return value
    try:
        converter = value.fake_mode.fake_tensor_converter
        memo = converter.tensor_memo
        for real, tid in list(
                converter.meta_converter.describer.lookup_tensor.items()):
            if memo.get(tid) is value:
                return real
    except Exception:          # torch internals moved: structure alone
        return None
    return None


def _real_or(value):
    """`value`'s real tensor when the partition gate can find it (_concrete),
    else `value` itself. Applied to a placeholder's tensor BEFORE any reshape:
    a reshaped FakeTensor is a new one, with no real tensor behind it."""
    real = _concrete(value)
    return value if real is None else real


def _pass_through(codes, scales, gsize, qmin, qmax, zeros=None,
                  source_shape=None):
    """Codes, scales and zero points off the graph -> PassThroughWeight.

    The codes must lie in their declared range (_check_code_range); unsigned
    codes are stored signed and the zero points moved with them, both exactly
    (asym_records.normalize_weight). A zero point of 0 everywhere leaves the
    weight symmetric, packed as plain codes and scales. While partitioning,
    a weight whose values the gate cannot reach is described by structure
    alone (`structural`): what the declared range and dtypes imply.
    """
    from executorch.backends.esp32s3.serialization.asym_records import (
        SchemeRefused,
        normalize_weight,
        storage_bias_for,
    )

    real_codes = _concrete(codes)
    real_zeros = _concrete(zeros) if zeros is not None else None
    if real_codes is None or (zeros is not None and real_zeros is None):
        try:
            bias = storage_bias_for(int(qmin), int(qmax))
        except SchemeRefused as why:
            raise LoweringRefused("weight code or zero-point range unsupported",
                                  str(why)) from None
        floating = (isinstance(zeros, torch.Tensor)
                    and zeros.dtype.is_floating_point)
        pt = PassThroughWeight(
            codes, scales, gsize, int(qmin) - bias, int(qmax) - bias,
            zero_points=None if floating else zeros,
            float_zero_points=zeros if floating else None, storage_bias=bias)
        pt.source_shape = source_shape
        pt.structural = True
        return pt
    codes, zeros = real_codes, real_zeros
    _check_code_range(codes, qmin, qmax)
    zp = None if zeros is None else _as_numpy(zeros)
    try:
        scheme = normalize_weight(_as_numpy(codes), qmin, qmax, zp)
    except SchemeRefused as why:
        raise LoweringRefused("weight code or zero-point range unsupported",
                              str(why)) from None

    def tensor(a):
        return None if a is None else torch.from_numpy(np.ascontiguousarray(a))

    pt = PassThroughWeight(
        torch.from_numpy(np.ascontiguousarray(scheme.codes)),
        scales.float() if isinstance(scales, torch.Tensor) else scales,
        gsize, scheme.qmin, scheme.qmax,
        zero_points=tensor(scheme.zero_points),
        float_zero_points=tensor(scheme.float_zero_points),
        storage_bias=scheme.storage_bias)
    pt.source_shape = source_shape
    return pt


def _turned(pt, transposed: bool):
    """`pt`, marked as a transposed convolution's turned codes or not."""
    pt.transposed = transposed
    return pt


def _check_code_range(codes, qmin: int, qmax: int) -> None:
    """Pass-through codes must actually lie in the range they declare.

    The storage rung is chosen from the DECLARED range alone (see
    _pass_through_format), and pack_rows then writes nibbles as
    (v + 8) & 0x0F -- so int8 codes carrying a "-7, 7" declaration are
    packed modulo 16 and the model becomes noise, silently.
    Eager `dequantize_affine` ignores qmin/qmax on the dequant direction, so
    such a graph is perfectly valid in PyTorch and exports clean: nothing else
    in the pipeline would ever notice. This is one comparison, and it stands
    between a correct artifact and an unrelated one.

    It reads values, so it refuses at partition time only where the gate
    reaches the real codes (_concrete: every parameter, buffer or constant
    placeholder); a weight it holds only as a FakeTensor (codes computed by
    graph ops, or torch internals moved) is claimed by structure and refused
    here in preprocess, which fails the whole export, by name.
    """
    lo, hi = int(codes.min()), int(codes.max())
    if lo < qmin or hi > qmax:
        raise LoweringRefused(
            "weight codes outside their declared range",
            f"pass-through weight declares [{qmin}, {qmax}] but its codes span "
            f"[{lo}, {hi}]; packing would wrap them modulo the declared width. "
            f"Fix the dequant node's quant_min/quant_max to match the data.")


def _torchao_code_range(node):
    """(quant_min, quant_max) of a torchao dequantize_affine.

    Both default to None -- positionally omitted or passed explicitly -- and
    torchao then takes the declared input dtype's own range. Reading them as
    mandatory ints would raise TypeError (int(None)) or IndexError on a graph
    torchao itself runs fine.
    """
    # dequantize_affine(input, block_size, scale, zero_point, input_dtype,
    #                   quant_min=None, quant_max=None)
    args = list(node.args) + [None] * (7 - len(node.args))
    qmin = args[5] if args[5] is not None else node.kwargs.get("quant_min")
    qmax = args[6] if args[6] is not None else node.kwargs.get("quant_max")
    if qmin is None or qmax is None:
        dtype = args[4] if args[4] is not None else node.kwargs.get("input_dtype")
        try:
            info = torch.iinfo(dtype)
        except TypeError:
            raise RuntimeError(
                f"{BACKEND_NAME}: {node.name} leaves its code range to its "
                f"input dtype {dtype}, which has no integer range to take"
            ) from None
        qmin = info.min if qmin is None else qmin
        qmax = info.max if qmax is None else qmax
    for bound in (qmin, qmax):
        if isinstance(bound, float) and not bound.is_integer():
            # int() would truncate it.
            raise LoweringRefused(
                "weight code range not integral",
                f"{node.name} declares quant_min {qmin!r}, quant_max {qmax!r}")
    return int(qmin), int(qmax)


# --- float-offset weights (W-off), as torchao's reference dequant traces -------
# tinygemm/HQQ, fl(fl((q - mid) * s) + m), and GGUF, fl(fl(q * d) + mn), have no
# dequantize op of their own: torchao's _dequantize_affine_tinygemm and
# _dequantize_gguf are plain torch, and export records them as elementwise
# aten ops over constant buffers --
#
#     [view_copy]( add( mul( [sub(] to_f32( [view_copy](Q) ) [, MID)], S ), M ) )
#
# with integer codes Q, an integer mid point MID (tinygemm), and S, M built
# from constants by views, multiplies and float casts (GGUF's d = ss * qs and
# mn = sm * qm). That chain is matched here and absorbed like a dequant op: its
# codes, scales and offsets pass through verbatim (the blob's float offsets).
_WOFF_VIEW = exir_ops.edge.aten.view_copy.default
_WOFF_ADD = exir_ops.edge.aten.add.Tensor
_WOFF_MUL = exir_ops.edge.aten.mul.Tensor
_WOFF_SUB = exir_ops.edge.aten.sub.Tensor
_WOFF_CASTS = (exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
               exir_ops.edge.aten._to_copy.default)
_WOFF_PARAM_OPS = (_WOFF_VIEW, _WOFF_MUL) + _WOFF_CASTS
_WOFF_MEMBER_OPS = (_WOFF_VIEW, _WOFF_ADD, _WOFF_MUL, _WOFF_SUB) + _WOFF_CASTS


def _woff_param(node, members, depth=0) -> bool:
    """A constant parameter expression (S or M): placeholders and numbers
    under views, multiplies and float casts. Its call nodes join `members`."""
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return True
    if not isinstance(node, torch.fx.Node) or depth > 8:
        return False
    if node.op == "placeholder":
        # Any numeric constant: GGUF keeps its 6-bit block scales as
        # integers, which the multiply promotes.
        val = node.meta.get("val")
        return val is not None and val.dtype != torch.bool
    if node.op != "call_function" or node.target not in _WOFF_PARAM_OPS:
        return False
    if node.target in _WOFF_CASTS and node.kwargs.get("dtype") not in (
            None, torch.float32):
        return False
    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and not _woff_param(
                arg, members, depth + 1):
            return False
    members.append(node)
    return True


def _woff_codes(node, members):
    """(codes placeholder, mid) for [sub(]to_f32([view_copy](Q))[, MID)], the
    side of the multiply that carries integer codes; None otherwise. MID is a
    number or a 0-d placeholder (export lifts tinygemm's mid point)."""
    chain, mid = [], None
    if (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target is _WOFF_SUB):
        if node.kwargs.get("alpha", 1) != 1:
            return None
        mid = node.args[1]
        if isinstance(mid, torch.fx.Node):
            val = mid.meta.get("val")
            if mid.op != "placeholder" or val is None or val.numel() != 1:
                return None
        elif not isinstance(mid, (int, float)) or isinstance(mid, bool):
            return None
        chain.append(node)
        node = node.args[0]
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target in _WOFF_CASTS
            and node.kwargs.get("dtype") == torch.float32):
        return None
    chain.append(node)
    node = node.args[0]
    if (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target is _WOFF_VIEW):
        chain.append(node)
        node = node.args[0]
    if not (isinstance(node, torch.fx.Node) and node.op == "placeholder"):
        return None
    val = node.meta.get("val")
    if val is None or val.dtype.is_floating_point or val.dtype == torch.bool:
        return None
    members.extend(chain)
    return node, mid


def _w_off_pattern(node):
    """The float-offset chain rooted at `node` (its last op), or None:
    {"root", "codes", "mid", "scale", "offset", "rows", "cols", "groups",
    "members"} -- rows x cols the dequantised weight, `groups` scale groups per
    row, `members` every call node the delegate absorbs with it."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    root, members = node, []
    if node.target is _WOFF_VIEW:
        members.append(node)
        node = node.args[0]
        if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
            return None
    if node.target is not _WOFF_ADD or node.kwargs.get("alpha", 1) != 1:
        return None
    add = node
    out = root.meta.get("val")
    if out is None or out.dim() != 2 or out.dtype != torch.float32:
        return None
    rows, cols = (int(d) for d in out.shape)
    for mul, offset in ((add.args[0], add.args[1]), (add.args[1], add.args[0])):
        if not (isinstance(mul, torch.fx.Node) and mul.op == "call_function"
                and mul.target is _WOFF_MUL):
            continue
        for x, scale in ((mul.args[0], mul.args[1]), (mul.args[1], mul.args[0])):
            found = list(members)
            codes = _woff_codes(x, found)
            if codes is None:
                continue
            if not (_woff_param(scale, found) and _woff_param(offset, found)):
                continue
            if any(isinstance(t, torch.fx.Node)
                   and not getattr(t.meta.get("val"), "dtype",
                                   torch.float32).is_floating_point
                   for t in (scale, offset)):
                continue
            # The codes as the multiply reads them: [rows, groups, g] (or the
            # [rows, cols] codes themselves, one group per row).
            xval = x.meta.get("val")
            if xval is None:
                continue
            shape = tuple(int(d) for d in xval.shape)
            if len(shape) == 3 and shape[0] == rows and shape[1] * shape[2] == cols:
                groups = shape[1]
            elif shape == (rows, cols):
                groups = 1
            else:
                continue
            found.extend((mul, add))
            return {"root": root, "codes": codes[0], "mid": codes[1],
                    "scale": scale, "offset": offset, "rows": rows,
                    "cols": cols, "groups": groups,
                    "members": set(found)}
    return None


def _w_off_member(node):
    """The float-offset chain `node` belongs to, or None. A member's users
    lead (singly) to the chain's root within a few hops."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target in _WOFF_MEMBER_OPS):
        return None
    cur = node
    for _ in range(16):
        pattern = _w_off_pattern(cur)
        if pattern is not None and node in pattern["members"]:
            return pattern
        if len(cur.users) != 1:
            return None
        cur = next(iter(cur.users))
        if cur.op != "call_function" or cur.target not in _WOFF_MEMBER_OPS:
            return None
    return None


class _NoValues(Exception):
    """A constant the partition gate holds only as a FakeTensor."""


def _eval_const_expr(ep: ExportedProgram, node):
    """A constant parameter expression evaluated with torch's own ops, so it
    is exactly the value eager computes (GGUF's d = fl(ss * qs))."""
    if not isinstance(node, torch.fx.Node):
        return torch.tensor(node, dtype=torch.float32)
    if node.op == "placeholder":
        value = _const_tensor(ep, node)
        if value is None:
            raise LoweringRefused("float-offset weight not constant",
                                  f"{node.name} is a runtime input")
        real = _concrete(value)
        if real is None:
            raise _NoValues(node.name)
        return real
    args = [_eval_const_expr(ep, a) if isinstance(a, torch.fx.Node) else a
            for a in node.args]
    return _callable_op(node.target)(*args, **node.kwargs)


def _resolve_w_off(ep: ExportedProgram, pattern):
    """A float-offset chain as a PassThroughWeight: codes, scales and offsets
    as given. tinygemm's fl(fl((q - mid) * s) + m) stores c = q - mid with zero
    point 0; GGUF's fl(fl(q * d) + mn) stores c = q - storage_bias with zero
    point -storage_bias. Either way c - zw is q - mid (or q), exactly."""
    q = _const_tensor(ep, pattern["codes"])
    if q is None:
        return None
    rows, cols, groups = pattern["rows"], pattern["cols"], pattern["groups"]
    gsize = cols // groups
    try:
        real = _concrete(q)
        if real is None:
            raise _NoValues(pattern["codes"].name)
        q = real.detach().reshape(rows, cols).to(torch.int64)
        scale = _eval_const_expr(ep, pattern["scale"])
        offset = _eval_const_expr(ep, pattern["offset"])
    except _NoValues:
        # The partition gate without values: int8 storage, which every code
        # width here fits, and offsets of the right shape.
        pt = PassThroughWeight(q, torch.ones(rows, groups), gsize, -128, 127,
                               offsets=torch.zeros(rows, groups),
                               storage_bias=0)
        pt.structural = True
        return pt
    try:
        scale = torch.broadcast_to(scale.float(), (rows, groups, 1)).reshape(
            rows, groups) if scale.dim() == 3 else torch.broadcast_to(
            scale.float(), (rows, 1)).reshape(rows, 1)
        offset = torch.broadcast_to(offset.float(), (rows, groups, 1)).reshape(
            rows, groups) if offset.dim() == 3 else torch.broadcast_to(
            offset.float(), (rows, 1)).reshape(rows, 1)
    except RuntimeError as why:
        raise LoweringRefused("float-offset weight layout unsupported",
                              f"scale/offset do not broadcast per group: {why}")
    if scale.shape[1] != groups or offset.shape[1] != groups:
        raise LoweringRefused(
            "float-offset weight layout unsupported",
            f"{groups} groups per row, scales {tuple(scale.shape)}, offsets "
            f"{tuple(offset.shape)}")
    mid = pattern["mid"]
    if mid is not None:
        mid = _eval_const_expr(ep, mid) if isinstance(mid, torch.fx.Node) else mid
        mid = float(mid)
        if mid != round(mid):
            raise LoweringRefused("float-offset weight layout unsupported",
                                  f"mid point {mid} is not an integer")
        codes = q - int(mid)
        lo, hi = int(codes.min()), int(codes.max())
        # Signed codes with zero point 0: i4 storage when they fit it and its
        # groups tile PIE's 32-code steps.
        i4 = lo >= -8 and hi <= 7 and (groups == 1 or gsize % 32 == 0)
        qmin, qmax = (-8, 7) if i4 else (-128, 127)
        pt = _pass_through(codes, scale, gsize, qmin, qmax)
    else:
        lo, hi = int(q.min()), int(q.max())
        i4 = lo >= 0 and hi <= 15 and (groups == 1 or gsize % 32 == 0)
        qmin, qmax = (0, 15) if i4 else ((0, 255) if lo >= 0 else (-128, 127))
        pt = _pass_through(q, scale, gsize, qmin, qmax)
    pt.offsets = offset.contiguous()
    return pt


def _require_float32_dequant(node, scales=None):
    """A weight dequant is consumed as codes and one float32 scale per group:
    refuse by name one whose output, or whose scale, is not float32. The op
    multiplies in the scale's dtype and rounds to its output
    dtype, so eager rounds every weight through them -- fl32(fl64((c - z) * s))
    for float64 scales, a float16 product for float16 ones -- which a float32
    scale cannot reproduce, and `.float()` would round silently. A per-tensor
    PT2E scale is exempt (pass scales=None): that op reads it as a Python
    float and multiplies in float32, by fl32(s), whatever its tensor's dtype."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and val.dtype != torch.float32:
        raise LoweringRefused(
            "weight dequant output not float32",
            f"{node.name} dequantises to {val.dtype}; the delegate's weights "
            f"are codes times a float32 scale, consumed by float32 matmuls")
    if isinstance(scales, torch.Tensor) and scales.dtype != torch.float32:
        raise LoweringRefused(
            "weight scale not float32",
            f"{node.name}: {scales.dtype} scales; the dequant multiplies in "
            f"that dtype, which one float32 scale per group cannot reproduce")


def _slice_bounds(start, end, extent):
    """A slice's start and end over a dim of `extent`, as aten.slice reads
    them: an absent start is 0 and an absent end the extent, a negative bound
    counts from the end, and each is clamped into [0, extent], the end no
    lower than the start. torch.export writes an open-ended slice
    (`W.t()[:, 8:]`) with end INT64_MAX, and keeps a negative start
    (`[:, -8:]`)."""
    start = 0 if start is None else int(start)
    end = extent if end is None else int(end)
    if start < 0:
        start += extent
    if end < 0:
        end += extent
    start = min(max(start, 0), extent)
    end = min(max(end, start), extent)
    return start, end


def _resolve_const(ep: ExportedProgram, node: torch.fx.Node):
    """Fold a chain of build-time-computable ops down to a tensor, else None.

    Returns a torch.Tensor for the fp32 path, a PassThroughWeight when the chain
    bottoms out in a torchao weight-dequant, or None when it is not constant."""
    if node.op == "placeholder":
        return _const_tensor(ep, node)
    if node.op == "call_function" and node.target in FOLDED:
        src = _resolve_const(ep, node.args[0])
        if src is None:
            return None
        if node.target is FOLDED_SLICE:
            dim = int(node.args[1]) if len(node.args) > 1 else 0
            start = node.args[2] if len(node.args) > 2 else None
            end = node.args[3] if len(node.args) > 3 else None
            step = node.args[4] if len(node.args) > 4 else 1
            step = 1 if step is None else int(step)
            if isinstance(src, PassThroughWeight):
                rows = int(src.codes.shape[0])
                cols = int(src.codes.shape[1])
                dim = dim + 2 if dim < 0 else dim
                extent = rows if dim == 0 else cols if dim == 1 else 0
                start, end = _slice_bounds(start, end, extent)
                prior = (src.logical_rows if dim == 0 else
                         src.logical_cols if dim == 1 else None)
                if (src.permuted or dim not in (0, 1) or prior is not None
                        or start != 0 or step != 1
                        or not 0 < end <= extent):
                    raise RuntimeError(
                        f"{BACKEND_NAME}: pass-through weight supports one row "
                        f"and one column prefix before transpose; got dim={dim} "
                        f"start={start} end={end} step={step}, "
                        f"shape=({rows}, {cols}), permuted={src.permuted}, "
                        f"prior={prior}")
                # Python's `[:, :K]` spelling can retain a full row slice next
                # to the meaningful column slice. A no-op slice carries no
                # logical contract and must not consume that dimension's slot.
                if end < extent:
                    if dim == 0:
                        src.logical_rows = end
                    else:
                        src.logical_cols = end
                return src
            # Ordinary fp32 constants take any forward slice. This is folded
            # to a compact payload; storage aliasing is required only for
            # PassThroughWeight above.
            dim = dim + src.dim() if dim < 0 else dim
            if dim < 0 or dim >= src.dim():
                raise RuntimeError(
                    f"{BACKEND_NAME}: constant slice dim={dim} is outside "
                    f"shape {tuple(src.shape)}")
            start, end = _slice_bounds(start, end, int(src.shape[dim]))
            if step <= 0:
                raise RuntimeError(
                    f"{BACKEND_NAME}: unsupported constant slice dim={dim} "
                    f"start={start} end={end} step={step} shape={tuple(src.shape)}")
            indices = [slice(None)] * src.dim()
            indices[dim] = slice(start, end, step)
            return src[tuple(indices)].contiguous()
        if isinstance(src, PassThroughWeight):
            # Only the [N,K] -> [K,N] transpose is meaningful here; anything fancier
            # has no defined meaning for the group axis and must fail, not guess.
            if tuple(node.args[1]) != (1, 0) or src.permuted:
                raise RuntimeError(
                    f"{BACKEND_NAME}: unsupported permute {node.args[1]} of a "
                    f"quantised weight (already permuted: {src.permuted})"
                )
            src.permuted = True
            return src
        return src.permute(*node.args[1]).contiguous()
    kind = _pt2e_dequant_kind(node)
    if kind is not None:
        return _resolve_pt2e_dequant(ep, node, kind)
    if _is_torchao_affine(node):
        # dequantize_affine(codes, block_size, scale, zero_point, dtype, qmin, qmax)
        codes = _resolve_const(ep, node.args[0])
        if codes is None or isinstance(codes, PassThroughWeight):
            return None
        scales = _resolve_const(ep, node.args[2])
        zero_arg = node.args[3] if len(node.args) > 3 else node.kwargs.get(
            "zero_point")
        zeros = (_resolve_const(ep, zero_arg)
                 if isinstance(zero_arg, torch.fx.Node) else zero_arg)
        block = node.args[1]
        qmin, qmax = _torchao_code_range(node)
        if scales is None:
            return None
        _require_float32_dequant(node, scales)
        zeros = _real_or(_resolved_zero_points(zeros, zero_arg))
        codes = _real_or(codes)
        if codes.dim() != 2 or len(block) != 2:
            raise LoweringRefused(
                "quantised-weight layout unsupported",
                f"codes {tuple(codes.shape)} with block {list(block)}")
        rows, cols = (int(d) for d in codes.shape)
        rblock, gsize = int(block[0]), int(block[1])
        # One row per block (per channel or per group), or one block of rows:
        # torchao's per-tensor block, whose scales broadcast down the rows.
        if rblock not in (1, rows) or gsize < 1 or cols % gsize:
            raise LoweringRefused(
                "quantised-weight layout unsupported",
                f"codes {tuple(codes.shape)} with block {list(block)}")
        groups = cols // gsize
        scales = _per_row_groups(scales, rows, groups, rblock != 1, node)
        if isinstance(zeros, torch.Tensor):
            zeros = _per_row_groups(zeros, rows, groups, rblock != 1, node)
        return _pass_through(codes, scales, gsize, qmin, qmax, zeros)
    pattern = _w_off_pattern(node)
    if pattern is not None:
        return _resolve_w_off(ep, pattern)
    return None


def _per_row_groups(table, rows, groups, broadcast, node):
    """A per-group parameter (scales or zero points) as [rows, groups]: torchao
    keeps the block structure ([rows, groups], or [1, groups] for its
    per-tensor block), PT2E a flat [rows * groups] vector."""
    t = table
    if broadcast and t.numel() == groups:
        t = t.reshape(1, groups).expand(rows, groups)
    if t.numel() != rows * groups:
        raise LoweringRefused(
            "quantised-weight layout unsupported",
            f"{node.name}: a parameter of {t.numel()} values for "
            f"{rows} rows x {groups} groups")
    return t.reshape(rows, groups).contiguous()


def _callable_op(target):
    """The OpOverload to call for a graph node's target: an edge-dialect
    EdgeOpOverload wraps it as `._op`, while an OpOverload's own `._op` is
    its raw C++ callable, which must not be called directly."""
    if isinstance(target, torch._ops.OpOverload):
        return target
    return getattr(target, "_op", target)


def _evaluate_dequant(node, args):
    """Run a dequant op itself on resolved constants: the exact fp32 value
    eager computes. Used for 1-D codes -- an int32 bias with its derived scale
    (Arm, Cortex-M) -- which is consumed as fp32 anyway."""
    op = _callable_op(node.target)
    try:
        return op(*args, **node.kwargs)
    except Exception as why:   # an op rejecting its inputs: not lowerable
        raise LoweringRefused("quantised bias not evaluable",
                              f"{node.name}: {why}") from None


def _feeds_transposed_conv(dq) -> bool:
    """Whether every reader of the dequant `dq` is a transposed convolution
    taking it as its weight."""
    users = list(dq.users)
    return bool(users) and all(
        u.op == "call_function"
        and u.target is exir_ops.edge.aten.convolution.default
        and len(u.args) > 6 and u.args[1] is dq and bool(u.args[6])
        for u in users)


def _resolve_pt2e_dequant(ep: ExportedProgram, node, kind):
    """A PT2E dequant over constant codes: a PassThroughWeight (2-D codes, or
    a convolution's per-output-channel [C_out, ...] codes reshaped to [N, K]),
    or for 1-D codes the fp32 value the op itself computes (an int32 bias).

      per_channel        (codes, scales, zero_points, axis, qmin, qmax, dtype)
      per_channel_group  (codes, scales, zero_points, qmin, qmax, dtype, group)
      per_tensor         (codes, scale, zero_point, qmin, qmax, dtype), the
                         scale a float or 0-d tensor; the op multiplies in
                         float32, so the stored scale is fl32(scale)
    """
    codes = _resolve_const(ep, node.args[0])
    if codes is None or isinstance(codes, PassThroughWeight):
        return None
    codes = _real_or(codes)
    _require_float32_dequant(node)
    args = list(node.args)
    resolved = [codes]
    for arg in args[1:3]:
        value = _resolve_const(ep, arg) if isinstance(arg, torch.fx.Node) else arg
        if isinstance(arg, torch.fx.Node) and (
                value is None or isinstance(value, PassThroughWeight)):
            if arg is args[2]:
                _resolved_zero_points(None, arg)     # refused by name
            return None
        resolved.append(value)
    scales, zeros = resolved[1], _real_or(resolved[2])
    resolved[2] = zeros
    if codes.dim() == 1:
        real = [_concrete(value) for value in resolved]
        if any(r is None and isinstance(v, torch.Tensor)
               for r, v in zip(real, resolved)):
            # The partition gate without values: a bias of this length.
            return torch.zeros(int(codes.shape[0]))
        return _evaluate_dequant(node, real + args[3:])
    if kind == "per_channel":
        axis, qmin, qmax = int(args[3]), int(args[4]), int(args[5])
    else:
        qmin, qmax = int(args[3]), int(args[4])
    if kind != "per_tensor":
        _require_float32_dequant(node, scales)
    source_shape = tuple(int(d) for d in codes.shape)
    transposed = codes.dim() > 2 and _feeds_transposed_conv(node)
    if codes.dim() > 2:
        if kind == "per_channel_group":
            raise LoweringRefused("quantised-weight layout unsupported",
                                  f"group-quantised codes {source_shape}")
        if transposed:
            # A transposed conv's weight is [C_in, C_out, (kd,) kh, kw], and
            # its output channels are axis 1 -- where XNNPACK's quantizer puts
            # them (ch_axis=1). Turned into the dense equivalent's [C_out,
            # C_in, ...] with the taps flipped, exactly as _conv_weight turns
            # a float weight, the codes pack like any conv's.
            if kind == "per_channel" and axis != 1:
                raise LoweringRefused(
                    "quantised-weight layout unsupported",
                    f"a transposed convolution's codes quantised on axis "
                    f"{axis}; its output channels are axis 1")
            codes = codes.flip(list(range(2, codes.dim()))).transpose(
                0, 1).contiguous()
            axis = 0
        # A convolution weight quantised per OUTPUT channel (axis 0), or per
        # tensor: [C_out, C_in/g, (kd,) kh, kw] reshapes to exactly the
        # [N, K] the packer takes and the im2col gather walks -- the identity
        # the whole conv delegate rests on, applied to codes instead of
        # floats. Refusing it here would leave every conv of a PT2E CNN
        # portable: the gate would see "not a plain constant" and fall back.
        codes = codes.reshape(codes.shape[0], -1)
    rows, cols = (int(d) for d in codes.shape)
    if kind == "per_tensor":
        for what, value in (("scale", scales), ("zero point", zeros)):
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                # Eager's dequantize_per_tensor.tensor raises; taking the
                # first element would run another model.
                raise LoweringRefused(
                    "per-tensor weight parameter not a scalar",
                    f"{node.name}: a per-tensor {what} of {value.numel()} "
                    f"values")
        if isinstance(scales, torch.Tensor):
            real = _concrete(scales)
            scales = 1.0 if real is None else real    # 1.0: the gate, unread
        if isinstance(zeros, torch.Tensor):
            real = _concrete(zeros)
            zeros = zeros if real is None else real
        scale = float(scales) if not isinstance(scales, torch.Tensor) else float(
            scales.reshape(-1)[0])
        scales = torch.full((rows,), float(np.float32(scale)),
                            dtype=torch.float32)
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(zeros, FakeTensor):
            # The gate, unread: an unknown zero point, judged as present.
            zeros = zeros.reshape(-1)[:1].expand(rows)
            return _turned(_pass_through(codes, scales, cols, qmin, qmax,
                                         zeros, source_shape), transposed)
        zero = (zeros if not isinstance(zeros, torch.Tensor)
                else zeros.reshape(-1)[0].item())
        zeros = None if zero in (None, 0) else torch.full(
            (rows,), zero, dtype=torch.float32 if isinstance(zero, float)
            else torch.int64)
        return _turned(_pass_through(codes, scales, cols, qmin, qmax, zeros,
                                     source_shape), transposed)
    if kind == "per_channel":
        if axis != 0:
            raise LoweringRefused(
                "quantised-weight layout unsupported",
                f"per-channel weight quantised on axis {axis}; only axis 0 "
                f"(per output channel) matches the [N,K] row layout")
        return _turned(_pass_through(codes, scales, cols, qmin, qmax, zeros,
                                     source_shape), transposed)
    group = int(args[6]) if len(args) > 6 else int(node.kwargs.get(
        "group_size", 128))
    if group > cols and scales.numel() == rows:
        group = cols      # torch's own rule: a single group per row
    if group < 1 or cols % group:
        raise LoweringRefused("quantised-weight layout unsupported",
                              f"group size {group} does not divide {cols}")
    groups = cols // group
    scales = _per_row_groups(scales, rows, groups, False, node)
    if isinstance(zeros, torch.Tensor):
        zeros = _per_row_groups(zeros, rows, groups, False, node)
    return _pass_through(codes, scales, group, qmin, qmax, zeros, source_shape)


def _const_fqn(ep: ExportedProgram, node) -> str | None:
    """The parameter/buffer/constant name behind a (folded) weight node: its
    codes, for a dequant or a float-offset chain."""
    while (isinstance(node, torch.fx.Node) and node.op == "call_function"
           and (node.target in FOLDED or _is_torchao_dequant(node))):
        node = node.args[0]
    pattern = _w_off_pattern(node)
    if pattern is not None:
        node = pattern["codes"]
    if not (isinstance(node, torch.fx.Node) and node.op == "placeholder"):
        return None
    sig = ep.graph_signature
    return (sig.inputs_to_parameters.get(node.name)
            or sig.inputs_to_buffers.get(node.name)
            or sig.inputs_to_lifted_tensor_constants.get(node.name))


def _is_build_time_const(ep: ExportedProgram, node: torch.fx.Node) -> bool:
    """Whether `_resolve_const` has a constant source, without evaluating it.

    The partitioner runs while FakeTensor guards are active.  Calling the full
    resolver there also validates code ranges with data-dependent reductions,
    which is correct in preprocess but illegal during the structural gate.
    This predicate answers only the question the gate needs: parameter/buffer
    versus runtime input.
    """
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "placeholder":
        return _const_tensor(ep, node) is not None
    if node.op == "call_function" and node.target in FOLDED:
        return _is_build_time_const(ep, node.args[0])
    if _pt2e_dequant_kind(node) is not None:
        # codes, scale(s), zero point(s): every one that is a node is constant.
        return (_is_build_time_const(ep, node.args[0])
                and all(not isinstance(arg, torch.fx.Node)
                        or _is_build_time_const(ep, arg)
                        for arg in node.args[1:3]))
    if _is_torchao_affine(node):
        required = (node.args[0], node.args[2])
        zero = node.args[3] if len(node.args) > 3 else None
        return (all(_is_build_time_const(ep, arg) for arg in required)
                and (not isinstance(zero, torch.fx.Node)
                     or _is_build_time_const(ep, zero)))
    pattern = _w_off_pattern(node)
    if pattern is not None:
        leaves = [n for n in (pattern["codes"], pattern["mid"])
                  if isinstance(n, torch.fx.Node)]
        for member in pattern["members"]:
            leaves.extend(a for a in member.all_input_nodes
                          if a.op == "placeholder")
        return all(_const_tensor(ep, leaf) is not None for leaf in leaves)
    return False


def _unsplittable_constant(ep: ExportedProgram, node: torch.fx.Node):
    """``(constant, user)`` when a constant `node` would pull into its partition
    is also held by a user that cannot be given its own copy, else None.

    `partition()` tags every constant operand into the claiming partition.
    Another user of the same constant is fine as long as ExecuTorch's
    `_maybe_duplicate_constant_nodes` can hand it a private copy -- but
    `duplicate_constant_node` rewrites only TOP-LEVEL args and kwargs. A user
    holding the constant inside a container (torch.cond's operand tuple, a cat
    list) keeps the tagged original, and backend_api then raises "constant
    data node ... has user (cond) which has tag (None)". The partitioner may
    not edit the graph, so the op is refused instead. Walks the folded and
    dequant weight path, which is claimed along with the op.
    """
    stack, seen = list(node.all_input_nodes), set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n.op == "placeholder":
            if _const_tensor(ep, n) is None:
                continue
            for user in n.users:
                if not (any(a is n for a in user.args)
                        or any(v is n for v in user.kwargs.values())):
                    return n, user
        elif n.op == "call_function" and (
                n.target in FOLDED or _is_torchao_dequant(n)
                or _w_off_member(n) is not None):
            stack.extend(n.all_input_nodes)
    return None


def _is_pass_through_const_path(node: torch.fx.Node) -> bool:
    """Whether a folded weight path bottoms out in quantized codes."""
    return _pass_through_source(node) is not None


def _shape_of(node: torch.fx.Node) -> List[int]:
    val = node.meta.get("val")
    if val is None:
        raise RuntimeError(f"node {node.name} has no 'val' meta; cannot infer shape")
    return list(val.shape)


def _check_pass_through_matmul(weight, k, n):
    """Already-quantised codes are taken verbatim. The graph permuted the
    stored [N,K] to [K,N] for mm; exactly one folded permute means the codes
    are already in the packer's orientation."""
    if not weight.permuted:
        raise LoweringRefused(
            "pass-through weight layout ambiguous",
            "quantised weight reached mm without the expected transpose")
    storage_shape = tuple(weight.codes.shape)
    logical_rows = (storage_shape[0] if weight.logical_rows is None
                    else weight.logical_rows)
    logical_cols = (storage_shape[1] if weight.logical_cols is None
                    else weight.logical_cols)
    if logical_cols != k or logical_rows != n:
        raise LoweringRefused(
            "pass-through weight shape mismatch",
            f"quantised codes are {storage_shape} with logical rows/cols "
            f"{logical_rows}/{logical_cols}, expected logical shape ({n}, {k})")
