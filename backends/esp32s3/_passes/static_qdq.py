# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static int8 quantize/dequantize pairs: Stage A.

A stock static recipe -- XNNPACK static and QAT, Arm a8w8, Cortex-M --
leaves every quantised tensor as a calibrated pair,

    q = quantize_per_tensor.default(x, s, z, qmin, qmax, dtype)
    d = dequantize_per_tensor.default(q, s, z, qmin, qmax, dtype)

fake quantisation with the model's own constants. Stage A keeps fp32 between
delegates and reproduces each pair's rounding wherever a delegate takes it:

  input    a claimed matmul or conv whose activation is d (through pure data
           movement) quantises it with (s, z, qmin, qmax), rounding to even
           before the zero point: the static EXT_ACT_QUANT. The gate records
           the scheme (Esp32OperatorSupport._static_act_problem). Requantising
           d's value with the same constants returns q's codes exactly
           (|c - z| <= 255 keeps every error far below half a code), so the
           record is exact whether the op reads x or d.
  output   a claimed matmul or conv whose fused epilogue ends in x, read by q
           alone, applies the pair after that epilogue (EXT_OUT_QUANT), and
           the partition's output is d's value: the gate claims q and d with
           the op (Esp32OperatorSupport._requant_problem).

absorb_static_qdq does the rewriting the partitioner may not (to_backend
checks that a partitioner leaves the graph alone). lower_esp32 runs it right
before partitioning:

  - it merges the dequantises convert_pt2e's DuplicateDQPass made of one
    quantize, and identical quantizes of one tensor, so that an output pair
    has one tail for the op to write;
  - it removes a pair that reads another pair's value on the same grid (Arm's
    uint8 model IO requantises to int8 with the zero point moved by 128): an
    identity, bit for bit;
  - where x has no claimable producer, it points every claimable consumer at x
    itself, its scheme recorded, and the pair leaves the graph once nothing
    else reads it -- "the partition input becomes the fp32 tensor before q".

No duplication: the shared constants are absorbed on every claimed
consumer, and a dequantise, then its quantize, goes only once nothing
reads it (dead-code elimination). A pair that any portable op still reads
stays whole and runs on ExecuTorch's own quantize/dequantize kernels (the
safety net, gen_kernels.py): never a quantize whose dequantise was removed.

Refused by name, never substituted:
  - a static scale that is not a float32 value: torch quantises with
    fl32(fl64(1/s)), ExecuTorch's kernel with 1.0f/(float)s and the record
    carries one f32 (226 of the 999 scales 0.001..0.999 differ). On an
    absorbed pair the consumer stays portable; on a pair left portable the
    export stops (refuse_portable_scales, after the post-lowering passes).
    Observer-made scales are always float32 values. An observer that saw an
    inf makes an infinite scale: no record carries it, but a portable pair
    runs as torch does (both reciprocals are 0).
  - int16 activations with a zero point other than 0, uint16 activations,
    per-channel activations, and a static pair on an integer bmm of two
    activations: the consumer stays portable, named in
    the partitioner's report.
inf and NaN activations are out of contract, as for every quantiser here.

int16 pairs (Arm a16w8) are absorbed like int8 ones: a claimed
op's input pair becomes its static EXT_ACT_QUANT with code_bits 16, its
output pair an EXT_OUT_QUANT over the int16 range, and the runtime dots the
int16 codes against int8 weights in int64 (esp32_asym.cpp a16_column).
Requantising a dequantised int16 value returns its code too: three float32
roundings of |c| <= 32768 stay below 0.006 of a code. The 16-bit record goes on an
ungrouped int8 matmul or conv only (the gate refuses the rest by name,
espressif_operator_support._a16_problem).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Q = "quantized_decomposed::quantize_per_tensor"
DQ = "quantized_decomposed::dequantize_per_tensor"
QC = "quantized_decomposed::quantize_per_channel"
DQC = "quantized_decomposed::dequantize_per_channel"

# Pure data movement: a quantisation commutes with each (elementwise, one
# value in, the same value out), so an op reading d through them reads d.
MOVES = ("aten::view_copy", "aten::clone", "aten::permute_copy",
         "aten::alias_copy", "aten::squeeze_copy", "aten::unsqueeze_copy",
         "dim_order_ops::_to_dim_order_copy")

# What a fake quantisation after the epilogue may follow: activations the
# device computes as torch does, operation for operation (a clamp, leaky
# ReLU, hardswish/hardsigmoid's decomposition, ReLU^2). The rest are the
# device's approximations (fast_expf, fast_tanhf), so their requantised codes
# could differ from eager's by more than a tie.
APPROXIMATE_ACTS = frozenset({"sigmoid", "silu", "gelu", "gelu_tanh", "softcap"})


class StaticPairRefused(RuntimeError):
    """A static quantize the export refuses by rule name."""

    def __init__(self, rule: str, where: str, detail: str):
        self.rule = rule
        super().__init__(f"esp32: static quantize {where} refused [{rule}]: "
                         f"{detail}")


def _schema(node):
    from executorch.backends.esp32s3.partition.espressif_partitioner import _overload

    op = _overload(node)
    return None if op is None else (op._schema.name, op._overloadname)


def _is(node, name, overload="default"):
    return _schema(node) == (name, overload)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_static_q(node) -> bool:
    """A quantize_per_tensor.default with constant scale and zero point."""
    return (_is(node, Q) and len(node.args) >= 6 and _number(node.args[1])
            and isinstance(node.args[2], int) and not node.kwargs)


def static_pair(node):
    """The static quantize whose dequantise `node` is, with the very same
    constants, or None."""
    if not (_is(node, DQ) and len(node.args) >= 6):
        return None
    q = node.args[0]
    if not is_static_q(q) or tuple(node.args[1:6]) != tuple(q.args[1:6]):
        return None
    out = node.kwargs.get("out_dtype", node.args[6] if len(node.args) > 6
                          else None)
    import torch

    return q if out in (None, torch.float32) else None


def is_move(node) -> bool:
    """Pure data movement that keeps the dtype (see MOVES)."""
    schema = _schema(node)
    if schema is None or schema[0] not in MOVES or not node.args:
        return False
    src = node.args[0]
    a = getattr(src, "meta", {}).get("val")
    b = node.meta.get("val")
    return (a is not None and b is not None
            and getattr(a, "dtype", None) == getattr(b, "dtype", None))


def static_source(value, depth: int = 8):
    """The static quantize whose pair produces `value`, through pure data
    movement, or None."""
    for _ in range(depth):
        q = static_pair(value)
        if q is not None:
            return q
        if not is_move(value):
            return None
        value = value.args[0]
    return None


def constant_codes_source(value, ep, depth: int = 8):
    """The dequantize_per_tensor behind `value`, through pure data movement,
    of integer codes that are a program constant, or None: the pair of an
    activation computed from constants alone (learned query tokens through
    a Linear), whose quantize convert_pt2e folded into the codes. Its
    constants are the pair's, and requantising its value with them returns
    the codes, as for a pair whose quantize is a node (static_pair), so a
    consumer's static record is exact. Only a declared range that is the
    whole code type, which every stored code is inside."""
    import torch

    from executorch.backends.esp32s3.operator_support.const_weights import (
        _is_build_time_const,
    )

    whole = {torch.int8: (-128, 127), torch.uint8: (0, 255),
             torch.int16: (-32768, 32767)}
    for _ in range(depth):
        if _is(value, DQ) and len(value.args) >= 6:
            codes, dtype = value.args[0], value.args[5]
            out = value.kwargs.get("out_dtype", value.args[6]
                                   if len(value.args) > 6 else None)
            val = getattr(codes, "meta", {}).get("val")
            if (getattr(val, "dtype", None) == dtype
                    and whole.get(dtype) == tuple(value.args[3:5])
                    and out in (None, torch.float32)
                    and _is_build_time_const(ep, codes)):
                return value
            return None
        if not is_move(value):
            return None
        value = value.args[0]
    return None


def per_channel_source(value, depth: int = 8):
    """A dequantize_per_channel over a quantize_per_channel NODE (an
    activation quantised per channel) behind `value`, or None. Weight codes
    are constants, not quantize nodes, so a weight never matches."""
    for _ in range(depth):
        if _is(value, DQC) and _is(value.args[0], QC):
            return value
        if not is_move(value):
            return None
        value = value.args[0]
    return None


@dataclass(frozen=True)
class QParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    dtype: object


def qparams(q) -> QParams:
    import torch

    s, z, qmin, qmax, dtype = q.args[1:6]
    return QParams(float(s), int(z), int(qmin), int(qmax),
                   dtype if isinstance(dtype, torch.dtype) else dtype)


def float32_scale(scale) -> bool:
    import numpy as np

    as_f32 = float(np.float32(scale))
    return math.isfinite(as_f32) and as_f32 == float(scale)


def scheme_problem(q):
    """(kind, detail) for what the static records cannot carry about
    q's pair, or None: int8/uint8 codes per tensor, or int16 codes with zero
    point 0 (Arm a16w8, code_bits 16), a positive float32
    scale with a finite reciprocal, the zero point inside the code range."""
    import numpy as np
    import torch

    p = qparams(q)
    if p.dtype == torch.uint16:
        return ("static uint16 activation",
                "uint16 codes do not fit the int16 codes a 16-bit record "
                "carries, and their zero point could not be 0; no stock "
                "recipe makes them")
    if p.dtype not in (torch.int8, torch.uint8, torch.int16):
        return ("static activation dtype",
                f"{p.dtype} codes; static pairs are int8, uint8 or int16")
    lo, hi = {torch.int8: (-128, 127), torch.uint8: (0, 255),
              torch.int16: (-32768, 32767)}[p.dtype]
    if not lo <= p.qmin < p.qmax <= hi or not p.qmin <= p.zero_point <= p.qmax:
        return ("static activation range",
                f"[{p.qmin}, {p.qmax}] zero point {p.zero_point} for "
                f"{p.dtype} codes")
    if p.dtype == torch.int16 and p.zero_point != 0:
        return ("static int16 zero point",
                f"zero point {p.zero_point}: 16-bit codes are symmetric -- "
                f"the int16 loop subtracts no zero point and its 131,040 "
                f"bound assumes |xq| <= 32768; no stock recipe makes one "
                f"(Arm a16w8 and Cortex-M's int16 config are symmetric)")
    inv = np.float32(1.0) / np.float32(p.scale) if p.scale > 0 else np.inf
    if not (math.isfinite(p.scale) and p.scale > 0 and np.isfinite(inv)):
        return ("static activation scale out of range",
                f"scale {p.scale!r}: positive, finite, with a finite "
                f"float32 reciprocal")
    if not float32_scale(p.scale):
        return ("static scale not a float32 value",
                f"scale {p.scale!r}: torch quantises with fl32(fl64(1/s)) and "
                f"the record carries one f32")
    return None


def act_scheme(q):
    """The mirrors.ActScheme a consumer of q's pair carries (A5; a uint8
    range is shifted onto int8 codes when the record is packed)."""
    from executorch.backends.esp32s3.quantizer.mirrors import ActScheme

    p = qparams(q)
    return ActScheme("pt2e/static", "static", "static", "rne_before_zp",
                     p.qmin, p.qmax, scale=p.scale, zero_point=p.zero_point)


def out_record(q, codes=False):
    """The EXT_OUT_QUANT record of q's pair (asym_records.OutQuant); with
    `codes`, Stage B's: the op writes q's int8 codes (stage_b.py)."""
    from executorch.backends.esp32s3.serialization.asym_records import out_quant

    p = qparams(q)
    return out_quant(p.scale, p.zero_point, p.qmin, p.qmax, codes=codes)


def requant_dq(q):
    """q's one dequantise, when q has exactly one reader and it is q's pair."""
    users = list(q.users)
    if len(users) == 1 and static_pair(users[0]) is q:
        return users[0]
    return None


# --- the pre-partition rewrite -------------------------------------------------------

def _replace(node, value, signature=None):
    """node's readers read `value` instead, node goes; a graph output it was
    is renamed in the program's signature, which names every user output."""
    if signature is not None:
        from torch.export.graph_signature import TensorArgument

        for spec in signature.output_specs:
            if getattr(spec.arg, "name", None) == node.name:
                spec.arg = TensorArgument(name=value.name)
    node.replace_all_uses_with(value)
    node.graph.erase_node(node)


def merge_duplicate_pairs(graph, signature=None) -> int:
    """Merge identical quantizes of one tensor, then identical dequantises of
    one quantize (convert_pt2e's DuplicateDQPass copies), into the first of
    each in graph order: the same values, one node. Returns nodes removed.
    `signature`: the program's, when a merged node may be a graph output."""
    removed = 0
    seen = {}
    for q in [n for n in graph.nodes if is_static_q(n)]:
        key = (q.args[0], tuple(q.args[1:6]))
        first = seen.setdefault(key, q)
        if first is not q:
            _replace(q, first, signature)
            removed += 1
    for q in [n for n in graph.nodes if is_static_q(n)]:
        keep = {}
        for d in [u for u in list(q.users) if static_pair(u) is q]:
            key = (tuple(d.args), tuple(sorted(d.kwargs.items(), key=str)))
            first = keep.setdefault(key, d)
            if first is not d:
                _replace(d, first, signature)
                removed += 1
    return removed


def elide_identity_pairs(graph, signature=None) -> int:
    """Remove each static pair that reads another pair's dequantised value on
    the same grid: q2(d1(..)) with s2 == s1 and every code of d1's range
    landing inside q2's after the zero-point shift (Arm's uint8 model IO:
    uint8 at the boundary, int8 inside, the zero point moved by 128). d1's
    values are fl((c - z1) * s); requantising them with s returns c - z1
    exactly (|c - z1| <= 255), so d2 = fl((c - z1) * s) = d1, bit for bit:
    d2's readers read d1. Returns the pairs removed."""
    removed = 0
    for q2 in [n for n in graph.nodes if is_static_q(n)]:
        d1 = q2.args[0]
        q1 = static_pair(d1)
        if q1 is None:
            continue
        a, b = qparams(q1), qparams(q2)
        if (a.scale != b.scale
                or a.qmin - a.zero_point + b.zero_point < b.qmin
                or a.qmax - a.zero_point + b.zero_point > b.qmax):
            continue
        for d2 in [u for u in list(q2.users) if static_pair(u) is q2]:
            _replace(d2, d1, signature)
        if not q2.users:
            graph.erase_node(q2)
            removed += 1
    return removed


def _absorbable(support, node, src):
    """The claimable matmul/conv consumers reached from `node`, a reader of
    `src`, when every path from it through pure data movement ends in one's
    activation; else None (some reader would need the dequantised value --
    a portable op, or the fp32 loop of a weight a recipe left float)."""
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        _activation_arg_index,
    )

    if is_move(node):
        if not node.users:
            return None
        found = []
        for user in node.users:
            got = _absorbable(support, user, node)
            if got is None:
                return None
            found.extend(got)
        return found
    idx = _activation_arg_index(node)
    if (idx is None or idx >= len(node.args) or node.args[idx] is not src
            or sum(a is src for a in node.args) != 1):
        return None
    return [node] if support._quantised_problem(node) is None else None


@dataclass
class StaticReport:
    """What absorb_static_qdq did: duplicates merged, consumers pointed at the
    tensor before their pair, pairs removed, pairs left for the gate to take
    as an op's output quantisation, and the pairs it left portable."""

    merged: int = 0
    identities: int = 0
    absorbed: int = 0
    removed: int = 0
    requant: int = 0
    portable: list = field(default_factory=list)
    stage_b: object = None      # stage_b.StageBReport under the spec

    def summary(self) -> str:
        return (f"static q/dq: {self.absorbed} consumer(s) absorbed their input "
                f"pair, {self.requant} pair(s) to requantize an op's output, "
                f"{self.removed} pair(s) removed, {self.merged} duplicate(s) "
                f"merged, {self.identities} identity pair(s) elided, "
                f"{len(self.portable)} pair(s) left portable"
                + (f"; {self.stage_b.summary()}" if self.stage_b is not None
                   else ""))


def absorb_static_qdq(edge_manager, partitioner,
                      method_name: str = "forward",
                      lut_report=None) -> StaticReport:
    """Rewrite static pairs for the partitioner that comes next (module
    docstring). `partitioner` is the Esp32Partitioner about to run: its gate,
    with its policies, decides which consumers are claimable, and the same
    gate claims the output pairs at partition time.

    A consumer pointed at the tensor before its pair carries
    node.meta["esp32_act_absorbed"]: the partitioner refuses to leave one
    portable ([act-consumer-unclaimed]), since it would then compute on the
    unquantised tensor.

    Under Esp32Partitioner(stage_b=True), stage_b.rewrite_stage_b first turns
    every region whose codes can travel as codes into int8 tensors and marks
    its pairs (`lut_report`, fold_quant_luts' report of the same export, gives
    each table's input scale); the rest is Stage A as above."""
    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    report = StaticReport()
    if not any(is_static_q(n) for n in graph.nodes):
        return report
    report.merged = merge_duplicate_pairs(graph, ep.graph_signature)
    report.identities = elide_identity_pairs(graph, ep.graph_signature)
    support = partitioner.gate(ep)
    if partitioner.compile_spec.stage_b:
        from executorch.backends.esp32s3._passes.stage_b import rewrite_stage_b

        report.stage_b = rewrite_stage_b(ep, support, lut_report)
        support = partitioner.gate(ep)
    from executorch.backends.esp32s3._passes.stage_b import STAGE_B

    for q in [n for n in graph.nodes if is_static_q(n)]:
        if scheme_problem(q) is not None or STAGE_B in q.meta:
            # The gate names why a scheme's consumers stay portable; a Stage B
            # producer's codes are its op's to write (stage_b.py).
            continue
        x = q.args[0]
        for d in [u for u in list(q.users) if static_pair(u) is q]:
            plan = [(hop, _absorbable(support, hop, d)) for hop in list(d.users)]
            if (not all(consumers for _, consumers in plan)
                    and requant_dq(q) is d
                    and support._requant_problem(q) is None):
                # Some reader needs d's value, and q reads a claimed op's
                # epilogue: the gate takes the pair whole as that op's output
                # quantisation, and every reader, claimed or not, reads d.
                report.requant += 1
                continue
            for hop, consumers in plan:
                if not consumers:
                    continue
                hop.replace_input_with(d, x)
                for c in consumers:
                    c.meta["esp32_act_absorbed"] = True
                report.absorbed += len(consumers)
    before = sum(1 for n in graph.nodes if is_static_q(n))
    graph.eliminate_dead_code()
    graph.lint()
    ep.graph_module.recompile()
    left = [n for n in graph.nodes if is_static_q(n)]
    report.removed = before - len(left)
    report.portable = [(q.name, "read by a portable op or a graph output")
                       for q in left if STAGE_B not in q.meta
                       and (requant_dq(q) is None
                            or support._requant_problem(q) is not None)]
    if left:
        # The ops' .out variants, which to_executorch needs to emit a pair
        # left portable (it names the kernel the runtime looks up).
        import executorch.kernels.quantized  # noqa: F401
    return report


def widen_portable_zero_points(ep) -> int:
    """Give a dequantize left portable the qparam dtypes ExecuTorch's kernel
    takes, which aborts the process on any other (op_dequantize.cpp's
    ET_CHECK_MSGs). A dequantize_per_channel's zero points must be int64,
    and some recipes store int32 ones (Cortex-M's convolution weights). A
    dequantize_per_tensor.tensor reads its scale and zero point from
    one-element tensors that must be float64 and int64: PyTorch's choosers
    make them so, but a program can hold a float32 scale or an int32 zero
    point, as a constant or as an input, which torch's op reads with .item()
    all the same. An exact cast goes in front of each -- an integer zero
    point to int64, a float32, float16 or bfloat16 scale to the float64 of
    the same value -- which constant propagation then folds into a new
    constant where the qparam is one. Returns how many casts went in; before
    constant_prop_pass, in apply_passes.

    A float zero point is widened to int64 where it is a program constant
    whose integer the kernel subtracts exactly as torch's op subtracts the
    float (_widens_exactly). A float zero point it cannot read so (a program
    input, a fractional value, one computed in the graph) or that the kernel
    would subtract differently (past float32's integers or int32's range) is
    left for refuse_portable_scales to refuse by name. So is an integer
    constant the kernel would subtract differently (2^24 + 1, which torch
    takes as 2^24 per tensor): it keeps its dtype, and refuse_portable_scales
    names it (dequant-zero-point-inexact; per channel, where such a zero
    point also lies outside the op's range, per-channel-zero-point-range).
    An integer zero point that is a program input is widened as before: its
    value is the run's, and the kernel's contract."""
    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = ep.graph_module.graph
    cast = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    ints = (torch.int8, torch.uint8, torch.int16, torch.int32)
    floats = (torch.float32, torch.float16, torch.bfloat16)
    constants = None
    widened = 0
    for n in list(graph.nodes):
        if _is(n, DQC):
            wanted, codes_at = [(2, ints, torch.int64)], 6
        elif _is(n, DQ, "tensor"):
            wanted = [(1, floats, torch.float64), (2, ints, torch.int64)]
            codes_at = 5
        else:
            continue
        zero = n.args[2] if len(n.args) > 2 else None
        zero_val = getattr(zero, "meta", {}).get("val")
        if isinstance(zero_val, torch.Tensor) and zero_val.dtype != torch.int64:
            if constants is None:
                from executorch.exir.passes.constant_prop_pass import (
                    get_constant_placeholder_dict,
                )

                constants = get_constant_placeholder_dict(ep)
            value = constants.get(zero)
            codes = n.args[codes_at] if len(n.args) > codes_at else None
            if zero_val.is_floating_point():
                if _is(n, DQ, "tensor") and _widens_exactly(value, codes):
                    wanted.append((2, floats + (torch.float64,), torch.int64))
            elif value is not None and not _widens_exactly(value, codes):
                wanted = [w for w in wanted if w[0] != 2]
        for index, narrow, dtype in wanted:
            arg = n.args[index] if index < len(n.args) else None
            val = getattr(arg, "meta", {}).get("val")
            if val is None or val.dtype not in narrow:
                continue
            with graph.inserting_before(n):
                wide = graph.call_function(
                    cast, (arg,), {"dtype": dtype,
                                   "dim_order": list(range(val.dim()))})
            wide.meta["val"] = val.to(dtype)
            n.update_arg(index, wide)
            widened += 1
    if widened:
        graph.lint()
        ep.graph_module.recompile()
    return widened


def _widens_exactly(value, codes) -> bool:
    """Whether a zero point held in a constant (float or integer) can be the
    int64 one ExecuTorch's dequantize kernels take without changing any
    output, for codes of dtype `codes` (None, an unknown value, cannot).

    Per tensor, torch's op subtracts the zero point from each code in
    float32, fl32(c) - fl32(z), an integer one too (it reads it as a Python
    int); the kernel narrows the int64 to int32, subtracts in int32 and
    rounds the difference to float32 (op_dequantize.cpp, DEQUANTIZE_IMPL),
    and both then multiply by fl32(scale). The two agree for every code when
    z is an integer float32 holds exactly, z itself is an int32 (the kernel
    narrows it before it subtracts), every c - z is an int32 and the codes
    are exact in float32 (every 8- and 16-bit code): each side is then the one
    rounding of the exact difference. Zero is exact for any codes. Past those
    bounds they part: 2^64 saturates the int64 cast, the kernel narrows 2^32
    to 0 and 2^31 to INT32_MIN (so c - z overflows int32 for uint8 or uint16
    codes, though c - 2^31 itself would fit), and torch subtracts 2^24 + 1 as
    2^24.

    Per channel, torch subtracts the zero points as integers, and
    refuse_portable_scales first requires each within the op's [quant_min,
    quant_max] (per-channel-zero-point-range), where this test always holds
    for 8- and 16-bit codes; what it still refuses there is a nonzero zero
    point with 32-bit codes, whose difference can overflow the kernel's int32.
    The output dtype and the scales have rules of their own
    (refuse_portable_scales: dequant-out-dtype, per-channel-scale-inexact)."""
    import torch

    if not isinstance(value, torch.Tensor) or not isinstance(codes, torch.dtype):
        return False
    v = value.detach().to(torch.float64)
    if not bool(torch.isfinite(v).all() and (v == v.round()).all()):
        return False
    if not bool((v.to(torch.float32).to(torch.float64) == v).all()):
        return False
    if not bool(((v >= -(2 ** 31)) & (v <= 2 ** 31 - 1)).all()):
        return False
    if bool((v == 0).all()):
        return True
    # int32 codes are not here: torch rounds a code itself past 2^24.
    bounds = {torch.int8: (-128, 127), torch.uint8: (0, 255),
              torch.int16: (-32768, 32767), torch.uint16: (0, 65535)}.get(codes)
    if bounds is None:
        return False
    lo, hi = bounds
    return bool(((lo - v) >= -(2 ** 31)).all() and ((hi - v) <= 2 ** 31 - 1).all())


def refuse_portable_scales(ep) -> int:
    """Refuse a static pair left to ExecuTorch's kernels that they would not
    compute as torch does, by rule name; returns how many static quantizes
    run portable. After every pass that can remove one
    (fold_quant_lut_pass.elide_lut_pairs).

      static-scale-not-float32   a quantize_per_tensor whose finite scale is
                                 not a float32 value: the kernel multiplies
                                 by 1.0f/(float)s, torch by fl32(fl64(1/s)).
                                 An infinite or NaN scale (its observer saw
                                 an inf) is left alone: both reciprocals are
                                 0 (or NaN), so both kernels quantise a
                                 finite input to the zero point and
                                 dequantise it to NaN, as torch does
      per-channel-activation    a quantize_per_channel of an activation (out
                                 of scope; ExecuTorch's kernel also takes
                                 float64 scales only, and aborts on float32)
      per-channel-dequant-dtype  a dequantize_per_channel whose zero points
                                 are not int64 or whose scales are not
                                 float32/float64: ExecuTorch's kernel aborts
      per-channel-scale-inexact  a dequantize_per_channel with float64 scales
                                 float32 does not hold exactly, or that are
                                 not known at export: torch multiplies the
                                 integer c - z by the scales in their dtype
                                 and rounds to the output once, ExecuTorch's
                                 kernels by fl32(s) (0.0123: 89 of 256 int8
                                 codes differ). Held, they still need codes
                                 of 8 or 16 bits, or constant codes whose
                                 every c - z float32 holds: the kernel
                                 rounds c - z to float32 before it
                                 multiplies, torch never does, and a 32-bit
                                 difference past 2^24 rounds
      per-channel-zero-point-range  a dequantize_per_channel whose zero
                                 point, known at export, lies outside the
                                 op's [quant_min, quant_max]: ExecuTorch's
                                 kernel aborts on one where it takes its int8
                                 -> float32 path (dequantize_optimized checks
                                 the range on every target)
      per-tensor-dequant-dtype   a dequantize_per_tensor.tensor whose scale
                                 is not float64 or whose zero point is not
                                 int64 after widen_portable_zero_points (a
                                 float zero point it cannot widen exactly,
                                 _widens_exactly): ExecuTorch's kernel aborts
      dequant-zero-point-inexact a dequantize whose integer zero point, known
                                 at export, the kernel subtracts from the
                                 codes differently from torch
                                 (_widens_exactly): per tensor, past float32's
                                 integers or int32's range (torch subtracts
                                 fl32(z) in float32, the kernel z in int32);
                                 per channel, a nonzero one with 32-bit codes
                                 (torch subtracts integers exactly, the
                                 kernel in int32)
      dequant-out-dtype          a dequantize whose output is neither float32
                                 nor a float16 it computes exactly. Per
                                 tensor, torch computes in float32 and rounds
                                 to the output after each step; per channel
                                 it multiplies in the scales' dtype and
                                 rounds once; ExecuTorch's kernels compute in
                                 float32 and cast. float16 agrees with 8-bit
                                 codes whose every c - z float16 holds
                                 (|c - z| <= 2048) and, per channel, float32
                                 scales; float64 differs at once (and per
                                 channel the kernel's 1-D path casts c - z
                                 before it multiplies); bfloat16 makes the
                                 kernel abort

    A zero point that is a program input is not checked: its value belongs to
    the run. The kernels narrow it to int32 and subtract in int32, which gives
    torch's bits for every zero point a quantize chooser makes (within its
    codes' range, so also every c - z a float16 output holds); one outside
    that range is outside the kernels' contract, and per channel the int8
    path aborts on it. Float64 per-channel scales known only at run time are
    refused (per-channel-scale-inexact): nothing proves float32 holds them.
    """
    import torch

    def dtype(value):
        val = getattr(value, "meta", {}).get("val")
        return getattr(val, "dtype", None)

    constants = None

    def constant(arg):
        """The tensor a constant placeholder holds; None for anything else
        (an input, or a value computed in the graph)."""
        nonlocal constants
        if not isinstance(arg, torch.fx.Node):
            return None
        if constants is None:
            from executorch.exir.passes.constant_prop_pass import (
                get_constant_placeholder_dict,
            )

            constants = get_constant_placeholder_dict(ep)
        return constants.get(arg)

    def zero_values(n):
        """A dequantize's integer zero point(s) as float64, when the export
        knows them: a number, or a constant tensor; None otherwise (none, an
        input, a computed one, or a float one, which the dtype rules
        handle)."""
        zero = n.args[2] if len(n.args) > 2 else None
        if _number(zero):
            return torch.tensor([float(zero)], dtype=torch.float64)
        val = getattr(zero, "meta", {}).get("val")
        if not isinstance(val, torch.Tensor) or val.is_floating_point():
            return None
        value = constant(zero)
        return None if value is None else value.detach().to(torch.float64)

    def shown(values):
        return [int(v) for v in values.flatten()[:4].tolist()]

    def refuse_inexact_zero(n, codes_at, per_channel=False):
        values = zero_values(n)
        codes = n.args[codes_at] if len(n.args) > codes_at else None
        if values is None or _widens_exactly(values, codes):
            return
        if per_channel:
            how = ("torch subtracts per-channel zero points from the codes as "
                   "integers, and ExecuTorch's dequantize kernel narrows each "
                   "to int32 and subtracts in int32; they agree only when z "
                   "and every c - z are int32s, which no nonzero zero point "
                   "proves for 32-bit codes")
        else:
            how = ("torch subtracts a zero point from each code in float32, "
                   "as fl32(z), and ExecuTorch's dequantize kernel narrows it "
                   "to int32 and subtracts in int32; they agree only for an "
                   "integer float32 holds exactly whose difference from every "
                   "code is an int32")
        raise StaticPairRefused(
            "dequant-zero-point-inexact", n.name,
            f"zero point(s) {shown(values)} with {codes} codes: {how}")

    def refuse_zero_point_range(n):
        values = zero_values(n)
        low, high = n.args[4], n.args[5]
        if values is None or bool(((values >= low) & (values <= high)).all()):
            return
        outside = values[(values < low) | (values > high)]
        raise StaticPairRefused(
            "per-channel-zero-point-range", n.name,
            f"zero point(s) {shown(outside)} outside [{low}, {high}]: "
            f"ExecuTorch's dequantize_per_channel aborts on such a zero point "
            f"where it takes its int8 -> float32 path (dequantize_optimized "
            f"checks quant_min <= z <= quant_max on every target)")

    def refuse_inexact_scales(n):
        scales = n.args[1]
        if dtype(scales) != torch.float64:
            return  # float32: both multiply by them; others are refused below
        value = constant(scales)
        if value is None:
            raise StaticPairRefused(
                "per-channel-scale-inexact", n.name,
                "float64 scales known only at run time: torch multiplies the "
                "integer c - z by them in float64, ExecuTorch's kernels by "
                "fl32(s), and nothing proves float32 holds them")
        v = value.detach().to(torch.float64)
        held = v.to(torch.float32).to(torch.float64) == v
        if not bool(held.all()):
            raise StaticPairRefused(
                "per-channel-scale-inexact", n.name,
                f"float64 scale(s) {v[~held].flatten()[:4].tolist()} float32 "
                f"does not hold: torch multiplies the integer c - z by them "
                f"in float64 and rounds once, ExecuTorch's kernels multiply "
                f"by fl32(s), and the two differ (0.0123: 89 of 256 int8 "
                f"codes)")
        # Held, a float64 scale gives torch's product exactly -- a 24-bit
        # scale times a c - z float32 holds is exact in float64 -- so both
        # sides round the same product to float32 once. The kernel, though,
        # rounds c - z to float32 before it multiplies, and torch never
        # does: past 2^24 a 32- or 64-bit code's difference is rounded twice
        # on one side only (2^24 + 1 at fl32(0.1): torch 0x49ccccce, the
        # kernel 0x49cccccd). Every 8- and 16-bit code's difference is
        # exact in float32; wider codes pass only where the export knows
        # every difference float32 holds (constant codes and zero points).
        codes = n.args[6] if len(n.args) > 6 else None
        if codes in (torch.int8, torch.uint8, torch.int16, torch.uint16):
            return
        if differences_held(constant(n.args[0]), n):
            return
        raise StaticPairRefused(
            "per-channel-scale-inexact", n.name,
            f"float64 scales with {codes} codes: torch multiplies the exact "
            f"integer c - z by the scale in float64 and rounds the product "
            f"once, ExecuTorch's kernels round c - z to float32 first, and a "
            f"difference past 2^24 rounds there (2^24 + 1 at fl32(0.1): "
            f"torch 1677721.75, the kernel 1677721.625); nothing at export "
            f"proves every difference float32 holds")

    def differences_held(codes, n):
        """Whether constant `codes` minus the op's constant zero points (or
        none) are every one a value float32 holds: the kernel's float32
        c - z is then exact, as torch's integer one."""
        if codes is None:
            return False
        zero = n.args[2] if len(n.args) > 2 else None
        if zero is None:
            z = torch.zeros(1, dtype=torch.int64)
        else:
            z = constant(zero)
            if z is None:
                return False
            z = z.detach().to(torch.int64)
        axis = n.args[3] if len(n.args) > 3 else 0
        c = codes.detach().to(torch.int64)
        shape = [1] * c.dim()
        if z.numel() > 1:
            shape[axis % c.dim()] = z.numel()
        d = c - z.reshape(shape)
        if bool(((d >= 2 ** 62) | (d <= -(2 ** 62))).any()):
            return False      # no round trip through float32 is safe there
        return bool((d.to(torch.float32).to(torch.int64) == d).all())

    def refuse_out_dtype(n, codes_at, per_channel=False):
        val = n.meta.get("val")
        out = getattr(val, "dtype", None)
        if out is None:
            out = n.kwargs.get("out_dtype")
        if out in (None, torch.float32):
            return
        why = (f"output {out}: torch's dequantize computes per tensor in "
               f"float32 and rounds to the output after each step, per "
               f"channel in the scales' dtype and rounds once, ExecuTorch's "
               f"kernels in float32 and then cast (float64 differs at the "
               f"first rounding; bfloat16 makes the kernel abort)")
        if out == torch.float16:
            codes = n.args[codes_at] if len(n.args) > codes_at else None
            bounds = {torch.int8: (-128, 127), torch.uint8: (0, 255)}.get(codes)
            values = zero_values(n)
            if bounds is None:
                why = (f"float16 output of {codes} codes: torch rounds codes "
                       f"or their differences past 2048 to float16, the "
                       f"kernel's float32 arithmetic does not")
            elif per_channel and dtype(n.args[1]) != torch.float32:
                why = (f"float16 output with {dtype(n.args[1])} scales per "
                       f"channel: torch multiplies in the scales' dtype and "
                       f"rounds the product once to float16, the kernel "
                       f"rounds it to float32 first (a float64 scale "
                       f"float32 holds included)")
            elif values is not None and bool(
                    ((bounds[0] - values).abs() > 2048).any()
                    or ((bounds[1] - values).abs() > 2048).any()):
                why = (f"float16 output with zero point(s) {shown(values)}: "
                       f"some c - z passes 2048, where float16 stops holding "
                       f"every integer, so torch's float16 subtraction rounds "
                       f"it and the kernel's int32 one does not (2^24: torch "
                       f"-inf, the kernel -1.0)")
            else:
                return  # every step exact: the same float32 product, rounded
        raise StaticPairRefused("dequant-out-dtype", n.name, why)

    left = 0
    for n in ep.graph_module.graph.nodes:
        if _is(n, Q):
            left += 1
            scale = n.args[1] if len(n.args) > 1 else None
            if (_number(scale) and math.isfinite(float(scale))
                    and not float32_scale(scale)):
                raise StaticPairRefused(
                    "static-scale-not-float32", n.name,
                    f"scale {float(scale)!r} is not a float32 value: "
                    f"ExecuTorch's quantize kernel multiplies by "
                    f"1.0f/(float)s, torch by fl32(fl64(1/s)), and the two "
                    f"differ for such scales (observer-made scales are always "
                    f"float32 values)")
        elif _is(n, QC):
            raise StaticPairRefused(
                "per-channel-activation", n.name,
                "an activation quantised per channel is out of scope (no "
                "stock default config emits one), and ExecuTorch's "
                "quantize_per_channel kernel takes float64 scales only")
        elif _is(n, DQ):
            refuse_out_dtype(n, 5)
            refuse_inexact_zero(n, 5)
        elif _is(n, DQC):
            refuse_out_dtype(n, 6, per_channel=True)
            refuse_zero_point_range(n)
            refuse_inexact_zero(n, 6, per_channel=True)
            zeros = n.args[2] if len(n.args) > 2 else None
            if (dtype(n.args[1]) not in (torch.float32, torch.float64)
                    or (zeros is not None
                        and dtype(zeros) != torch.int64)):
                raise StaticPairRefused(
                    "per-channel-dequant-dtype", n.name,
                    f"scales {dtype(n.args[1])}, zero points "
                    f"{dtype(zeros) if zeros is not None else None}: "
                    f"ExecuTorch's dequantize_per_channel kernel takes "
                    f"float32/float64 scales and int64 zero points")
            refuse_inexact_scales(n)
        elif _is(n, DQ, "tensor"):
            refuse_out_dtype(n, 5)
            refuse_inexact_zero(n, 5)
            if (dtype(n.args[1]) != torch.float64
                    or dtype(n.args[2]) != torch.int64):
                raise StaticPairRefused(
                    "per-tensor-dequant-dtype", n.name,
                    f"scale {dtype(n.args[1])}, zero point "
                    f"{dtype(n.args[2])}: ExecuTorch's "
                    f"dequantize_per_tensor.Tensor_out kernel takes a float64 "
                    f"scale and an int64 zero point, and a float zero point "
                    f"widens to one exactly only as a constant holding an "
                    f"integer that float32 holds exactly and that the "
                    f"kernel's int32 subtraction from every code takes as "
                    f"torch's float32 one does")
    return left
