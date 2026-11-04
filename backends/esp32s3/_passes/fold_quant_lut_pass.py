# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::quant_lut: an activation between two static quantisers, as a table.

A static int8 graph (XNNPACK, Arm or Cortex-M PT2E) leaves every activation
between two quantise/dequantise pairs:

    q1(X) -> dq1 -> A -> q2 -> dq2

and ExecuTorch runs A on its portable kernel, which is not ATen's: portable
sigmoid, tanh, exp, log, erf and friends call newlib or libm, ATen calls Sleef
or a polynomial, and portable gelu even computes in double. So the device's
codes after q2 drift from the eager model the user validated. The fix is the
one Arm and Cortex-M ship: A reads only the 256 codes dq1 can produce, so
evaluate it on all of them at export and look the answer up at run time.

    espressif::quant_lut(Tensor x, Tensor table, float inv_scale,
                         int zero_point, int quant_min, int quant_max)

per element, exactly:

    v = fl(x * inv_scale)          inv_scale = fl32(1/s1), the multiplier
                                   quantize_per_tensor itself uses
    v = clamp(v, -512, 512)        NaN -> 0 first: its code is z1
    c = clamp(rne(v) + z1, quant_min, quant_max)
    y = table[c - quant_min]       a copy: denormals and NaN payloads survive

which is q1's code for every x -- for NaN, the zero point, as ExecuTorch 1.6's
portable quantize kernel defines it (runtime/quant_rules.h); torch's own code
for it is platform-defined. The table is built BEFORE to_edge, from
the ATen node the eager model runs (to_edge turns silu into sigmoid*mul and
hardswish into four ops, a different function), evaluated by torch itself:

  build_lut       all codes through the graph's own dq1, then A on 32 copies of
                  each (one thread, every element in a full vector block, every
                  4-lane group uniform), then the graph's own q2 and dq2 when
                  every user of A is one identical quantize (V_q, values on q2's
                  grid), else the raw values (V_raw). Codes whose value differs
                  in torch's scalar path or in a lane-coupled vector are "split
                  codes": recorded, never refused. sqrt alone is evaluated by
                  its definition instead (EXACT): IEEE 754's correctly rounded
                  root, which the portable kernel computes and ATen's kernel
                  does not on every machine (equal on finite inputs, a NaN as
                  0x7FC00000).
  fold_quant_luts the rewrite, in lower_esp32(..., quant_luts=True) before
                  to_edge: A becomes quant_lut(q1's input, V, ...); dq1 and q1
                  go when nothing else reads them. Named refusals leave the
                  graph exactly as it was.
  elide_lut_pairs in apply_passes, after partitioning: a q2 -> dq2 pair after a
                  table whose entries it maps to themselves is an identity on
                  everything the table can output, so it goes.

The kernel is ops/quant_lut.cpp (host and device, the same
source, -ffp-contract=off); mirrors.quant_lut is its numpy mirror; this
module's CompositeExplicitAutograd body is the torch reference.
"""

import contextlib
import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  (quantized_decomposed ops)
from torch.library import Library

# FRAGMENT: the "espressif" namespace is shared with the other custom ops.
lib = Library("espressif", "FRAGMENT")
lib.define("quant_lut(Tensor x, Tensor table, float inv_scale, int zero_point, "
           "int quant_min, int quant_max) -> Tensor")
lib.define("quant_lut.out(Tensor x, Tensor table, float inv_scale, "
           "int zero_point, int quant_min, int quant_max, *, "
           "Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")

# The quantize step's rails. With |zero point| <= 255 any |v| past 512 lands on
# a code rail anyway, and inside them 1.5 * 2^23 rounds half to even exactly.
RAIL = 512.0


def quant_lut(x, table, inv_scale, zero_point, quant_min, quant_max):
    """The reference: q1's code for every element (NaN's the zero point),
    then the table entry."""
    v = x * inv_scale
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v).clamp(-RAIL, RAIL)
    codes = (torch.round(v) + zero_point).clamp(quant_min, quant_max)
    return table[codes.to(torch.int64) - quant_min]


def _quant_lut_out(x, table, inv_scale, zero_point, quant_min, quant_max, *,
                   out):
    out.copy_(quant_lut(x, table, inv_scale, zero_point, quant_min, quant_max))
    return out


impl.impl("quant_lut", quant_lut, "CompositeExplicitAutograd")
impl.impl("quant_lut.out", _quant_lut_out, "CompositeExplicitAutograd")


@torch.library.register_fake("espressif::quant_lut")
def _quant_lut_fake(x, table, inv_scale, zero_point, quant_min, quant_max):
    return torch.empty_like(x)


QUANT_LUT = torch.ops.espressif.quant_lut.default

# The code-to-code variant (Stage B, stage_b.py): int8 codes in, int8 codes
# out, the table the codes Stage A's next quantize gives each entry. Per
# element: table[clamp(x, quant_min, quant_max) - quant_min] -- the clamp
# changes no code a producer on this grid writes.
lib.define("quant_lut_codes(Tensor x, Tensor table, int quant_min, "
           "int quant_max) -> Tensor")
lib.define("quant_lut_codes.out(Tensor x, Tensor table, int quant_min, "
           "int quant_max, *, Tensor(a!) out) -> Tensor(a!)")


def quant_lut_codes(x, table, quant_min, quant_max):
    """The reference: each code's table entry."""
    codes = x.to(torch.int64).clamp(quant_min, quant_max)
    return table[codes - quant_min]


def _quant_lut_codes_out(x, table, quant_min, quant_max, *, out):
    out.copy_(quant_lut_codes(x, table, quant_min, quant_max))
    return out


impl.impl("quant_lut_codes", quant_lut_codes, "CompositeExplicitAutograd")
impl.impl("quant_lut_codes.out", _quant_lut_codes_out,
          "CompositeExplicitAutograd")


@torch.library.register_fake("espressif::quant_lut_codes")
def _quant_lut_codes_fake(x, table, quant_min, quant_max):
    return torch.empty_like(x, dtype=table.dtype)


QUANT_LUT_CODES = torch.ops.espressif.quant_lut_codes.default

# --- What the rewrite takes ---------------------------------------------------

_ATEN = torch.ops.aten

# Functional overloads only. relu, hardtanh, clamp, abs, neg and round are
# not here: exact and cheap as portable ops, or fused by the delegate.
ALLOWLIST = {
    _ATEN.sigmoid.default: "sigmoid",
    _ATEN.tanh.default: "tanh",
    _ATEN.silu.default: "silu",
    _ATEN.gelu.default: "gelu",
    _ATEN.elu.default: "elu",
    _ATEN.selu.default: "selu",
    _ATEN.celu.default: "celu",
    _ATEN.leaky_relu.default: "leaky_relu",
    _ATEN.hardswish.default: "hardswish",
    _ATEN.hardsigmoid.default: "hardsigmoid",
    _ATEN.mish.default: "mish",
    _ATEN.softplus.default: "softplus",
    _ATEN.exp.default: "exp",
    _ATEN.expm1.default: "expm1",
    _ATEN.log.default: "log",
    _ATEN.log1p.default: "log1p",
    _ATEN.log2.default: "log2",
    _ATEN.log10.default: "log10",
    _ATEN.erf.default: "erf",
    _ATEN.sqrt.default: "sqrt",
    _ATEN.rsqrt.default: "rsqrt",
    _ATEN.reciprocal.default: "reciprocal",
    _ATEN.pow.Tensor_Scalar: "pow",
}

# The in-place spellings, refused by name (lut-inplace).
INPLACE = {
    getattr(getattr(_ATEN, f"{name}_"), overload): name
    for name, overload in (
        ("sigmoid", "default"), ("tanh", "default"), ("silu", "default"),
        ("gelu", "default"), ("elu", "default"), ("selu", "default"),
        ("celu", "default"), ("leaky_relu", "default"),
        ("hardswish", "default"), ("hardsigmoid", "default"),
        ("mish", "default"), ("exp", "default"), ("expm1", "default"),
        ("log", "default"), ("log1p", "default"), ("log2", "default"),
        ("log10", "default"), ("erf", "default"), ("sqrt", "default"),
        ("rsqrt", "default"), ("reciprocal", "default"), ("pow", "Scalar"))
    if hasattr(_ATEN, f"{name}_")
    and hasattr(getattr(_ATEN, f"{name}_"), overload)
}

# ATen kernels that are pure functions of the element on the build machines
# measured (Apple silicon; x86-64 with AVX-512): the same bits in the vector
# path, the scalar tail, any thread split and any stride (test_quant_lut.py
# asserts it, so a torch upgrade that breaks one fails loudly). Bits of a NaN
# that came in are exempt: no code dequantises to NaN, and ATen's AVX-512
# maximum hands hardswish and hardsigmoid an all-ones NaN in the vector path
# where the scalar path keeps the input's. pow is pure for integer exponents
# only.
PURE = frozenset({"tanh", "exp", "expm1", "log", "log1p", "log2", "log10",
                  "sqrt", "rsqrt", "reciprocal", "hardswish", "hardsigmoid",
                  "leaky_relu"})
# Vector paths that fall back for the WHOLE vector when one lane is large
# (erf past |x| ~ 9.34, gelu's erf form past ~13.2): a lane's result depends on
# its neighbours. Probed with 20, which triggers both.
COUPLED_PARTNER = 20.0


def lane_coupled(name, args, kwargs):
    if name == "erf":
        return True
    if name == "gelu":
        return kwargs.get("approximate", "none") == "none"
    return False


def is_pure(name, args, kwargs):
    if name == "pow":
        exponent = args[0] if args else kwargs.get("exponent")
        return float(exponent).is_integer()
    return name in PURE


def sqrt_exact(xs):
    """sqrt of a float32 tensor as IEEE 754 defines it, the correctly rounded
    root: what the portable kernel the table replaces computes (std::sqrt;
    libgcc's __ieee754_sqrtf on the S3, held to the hardest cases by
    divsqrt_check) and ATen's NEON kernel too. The x86 wheel's kernel is not
    (its float sqrt goes through MKL: on an AVX-512 Threadripper 2,898,419 of
    17,223,013 float32 inputs come out one ulp off), so a table evaluated by it
    would hold neither the portable chain's bits nor the table another machine
    builds. numpy's float32 sqrt is the hardware's on both.

    The contract: equal to the portable chain on finite inputs, -0 and the
    denormals included; a NaN out (a negative code's root, or a NaN in) is
    written as the quiet NaN 0x7FC00000, ATen's word on both machines,
    whatever sign and payload the chain's C library gives its own NaN (the
    conformance checks hold a NaN by NaN-ness; test_sqrt_table_nan_word pins
    the word)."""
    import numpy as np

    with np.errstate(invalid="ignore"):
        y = np.sqrt(xs.detach().numpy().astype(np.float32))
    y = np.where(np.isnan(y), np.float32(np.nan), y).astype(np.float32)
    return torch.from_numpy(y)


# Activations whose table holds their definition's values, not torch's kernel's
# (the target -> the definition on a float32 tensor).
EXACT = {_ATEN.sqrt.default: sqrt_exact}


# --- Matching quantize / dequantize nodes ----------------------------------------

def _schema(target):
    """('namespace::name', overload) of an ATen or edge-dialect op target."""
    # An edge-dialect EdgeOpOverload wraps the OpOverload as _op; an
    # OpOverload's own _op is its C++ callable.
    op = target if isinstance(target, torch._ops.OpOverload) else getattr(
        target, "_op", None)
    if not isinstance(op, torch._ops.OpOverload):
        return None, None
    return op._schema.name, op._overloadname


Q = "quantized_decomposed::quantize_per_tensor"
DQ = "quantized_decomposed::dequantize_per_tensor"


def _is(node, name, overload="default"):
    return (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and _schema(node.target) == (name, overload))


def _is_dequant(node):
    """Any dequantize: per tensor, channel, token or group, any overload, or
    torchao's. An activation reading one of these is a candidate (and is
    refused unless it is the static per-tensor kind); one reading a float
    tensor is not a candidate at all."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    name, _ = _schema(node.target)
    return (name is not None and name.split("::")[0] in
            ("quantized_decomposed", "torchao") and "dequantize" in name)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_qparams(args):
    """(scale, zero point, qmin, qmax, dtype) as Python constants, or None."""
    if len(args) < 6:
        return None
    scale, zero_point, qmin, qmax, dtype = args[1:6]
    if not (_number(scale) and all(isinstance(v, int) and not isinstance(v, bool)
                                   for v in (zero_point, qmin, qmax))
            and isinstance(dtype, torch.dtype)):
        return None
    return float(scale), zero_point, qmin, qmax, dtype


def _has_node(value):
    if isinstance(value, torch.fx.Node):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_node(v) for v in value)
    if isinstance(value, dict):
        return any(_has_node(v) for v in value.values())
    return False


def _contiguous(fake):
    """espressif_ops.cpp contiguous_tensor's predicate: strides of a dense
    row-major tensor, size-1 axes exempt."""
    expected = 1
    for size, stride in reversed(list(zip(fake.shape, fake.stride()))):
        if size > 1 and stride != expected:
            return False
        expected *= size
    return True


def reciprocal(scale):
    """fl32(1/s) as a Python float: mirrors.reciprocal, the multiplier
    quantize_per_tensor applies (a float64 quotient rounded to float32)."""
    from executorch.backends.esp32s3.quantizer import mirrors

    return float(mirrors.reciprocal(scale))


CODE_RANGE = {torch.int8: (-128, 127), torch.uint8: (0, 255)}


def refusal(A):
    """The named rule that keeps activation node A portable, or None.

    A is an allowlisted (or in-place) activation with users whose first
    argument is a dequantize."""
    if A.target in INPLACE:
        return "lut-inplace"
    if _has_node(list(A.args[1:])) or _has_node(dict(A.kwargs)):
        return "lut-extra-input"
    dq1 = A.args[0]
    if not _is(dq1, DQ) or not isinstance(dq1.args[0], torch.fx.Node):
        return "lut-input-not-static"
    q1 = dq1.args[0]
    qp = _scalar_qparams(q1.args) if _is(q1, Q) else None
    if qp is None or _scalar_qparams(dq1.args) is None:
        return "lut-input-not-static"
    if tuple(dq1.args[1:6]) != tuple(q1.args[1:6]):
        return "lut-dq-mismatch"
    scale, zero_point, qmin, qmax, dtype = qp
    if (dtype not in CODE_RANGE or qmin < CODE_RANGE[dtype][0]
            or qmax > CODE_RANGE[dtype][1] or qmax - qmin + 1 > 256):
        return "lut-wide-codes"
    if not (math.isfinite(scale) and scale > 0 and qmin < qmax
            and qmin <= zero_point <= qmax):
        return "lut-bad-qparams"
    inv = reciprocal(scale)
    if not (math.isfinite(inv) and inv > 0):
        return "lut-bad-qparams"
    X = q1.args[0]
    xv = X.meta.get("val") if isinstance(X, torch.fx.Node) else None
    av = A.meta.get("val")
    if (not isinstance(xv, torch.Tensor) or xv.dtype != torch.float32
            or not isinstance(av, torch.Tensor) or av.dtype != torch.float32
            or dq1.kwargs.get("out_dtype", None) not in (None, torch.float32)):
        return "lut-dtype"
    if not _contiguous(xv):
        return "lut-layout"
    return None


# --- Building a table ----------------------------------------------------------

@contextlib.contextmanager
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _dq_op():
    return torch.ops.quantized_decomposed.dequantize_per_tensor.default


def _q_op():
    return torch.ops.quantized_decomposed.quantize_per_tensor.default


def bits(t):
    return t.contiguous().view(torch.int32)


@dataclass
class LutBuild:
    """One table, as fold_quant_luts inserts it and its report records it."""

    op: str
    kind: str                    # "q" (values on q2's grid) or "raw"
    table: torch.Tensor          # float32 [qmax1 - qmin1 + 1]
    scale: float                 # s1, q1's own
    inv_scale: float             # fl32(1/s1)
    zero_point: int
    quant_min: int
    quant_max: int
    out_qparams: Optional[tuple] = None   # q2's (s2, z2, qmin2, qmax2, dtype)
    split: dict = field(default_factory=dict)   # context -> [codes]

    @property
    def name(self):
        digest = hashlib.sha256(
            self.table.numpy().tobytes()
            + struct.pack("<I", self.table.numel())).hexdigest()
        return f"esp32_lut_{digest[:16]}"


def evaluate(target, xs, args=(), kwargs=None):
    """target over the values xs in the canonical vector context: 32 copies
    of each (a full 2 * Vec::size() block for NEON, AVX2 and AVX-512, every
    4-lane group uniform, at most 8,192 elements: below GRAIN_SIZE and GELU's
    threading threshold), one thread, no grad. Returns one value per input."""
    kwargs = kwargs or {}
    with one_thread(), torch.no_grad():
        out = target(xs.repeat_interleave(32), *args, **kwargs)
    copies = bits(out).reshape(-1, 32)
    if not bool((copies == copies[:, :1]).all()):
        raise RuntimeError(f"{target}: the 32 copies of one value disagree in "
                           f"the canonical vector context")
    return out.reshape(-1, 32)[:, 0].contiguous()


def table_and_split(target, xs, args=(), kwargs=None, out_qparams=None,
                    coupled=False):
    """(table, {context: [indices into xs]}) for activation `target` on the
    dequantized codes xs: the canonical vector evaluation, then q2 and dq2
    when out_qparams is given (V_q) or nothing (V_raw); and the positions whose
    value differs -- after the same q2/dq2 -- in torch's scalar lambda (each
    code alone, a 1-element tensor: what cpu_kernel_vec hands a block's tail)
    and, for a lane-coupled kernel, beside a fallback-triggering lane (each
    code in [x, 20, 20, 20], one vector per code, none in the scalar loop).
    An EXACT target's table is its definition's values instead of the vector
    evaluation's, and its split codes are where torch's scalar lambda differs
    from those."""
    kwargs = kwargs or {}

    def finish(values):
        if out_qparams is None:
            return values
        return _dq_op()(_q_op()(values, *out_qparams), *out_qparams)

    exact = EXACT.get(target)
    table = finish(exact(xs) if exact else evaluate(target, xs, args, kwargs))
    want = bits(table)
    with one_thread(), torch.no_grad():
        found = {"scalar": torch.stack([target(xs[i:i + 1], *args, **kwargs)[0]
                                        for i in range(xs.numel())])}
        if coupled:
            partner = torch.full((xs.numel(), 4), COUPLED_PARTNER)
            partner[:, 0] = xs
            found["coupled"] = target(partner.reshape(-1), *args,
                                      **kwargs)[0::4].contiguous()
    split = {context: [int(i) for i in
                       (bits(finish(values)) != want).nonzero().flatten()]
             for context, values in found.items()}
    return table, split


def build_lut(A, out_qparams=None):
    """The table for activation node A (fold_quant_luts has checked it).

    out_qparams: q2's args (scale, zp, qmin, qmax, dtype) for a V_q table,
    None for V_raw."""
    name = ALLOWLIST[A.target]
    dq1 = A.args[0]
    q1 = dq1.args[0]
    scale, zero_point, qmin, qmax, dtype = _scalar_qparams(q1.args)
    codes = torch.arange(qmin, qmax + 1, dtype=torch.int64).to(dtype)
    # The graph's own dq1 with its own args: fl(fl(c - z) * fl32(s)).
    xs = _dq_op()(codes, *dq1.args[1:], **dq1.kwargs)
    args, kwargs = tuple(A.args[1:]), dict(A.kwargs)
    table, split = table_and_split(A.target, xs, args, kwargs, out_qparams,
                                   lane_coupled(name, args, kwargs))
    build = LutBuild(name, "raw" if out_qparams is None else "q", table,
                     scale, reciprocal(scale), zero_point, qmin, qmax,
                     out_qparams)
    build.split = {context: [qmin + i for i in where]
                   for context, where in split.items()}
    return build


# --- The rewrite -----------------------------------------------------------------

@dataclass
class QuantLutReport:
    tables: list = field(default_factory=list)     # (node name, LutBuild)
    refused: list = field(default_factory=list)    # (node name, op, rule)
    constants: dict = field(default_factory=dict)  # name -> users

    def summary(self):
        shared = len(self.tables) - len(self.constants)
        parts = [f"quant_lut: {len(self.tables)} table(s)"
                 f"{f' ({shared} shared)' if shared else ''}"]
        if self.refused:
            parts.append(f"{len(self.refused)} refused: " + ", ".join(
                f"{node} ({op}) {rule}" for node, op, rule in self.refused))
        split = [f"{node} {sum(map(len, b.split.values()))}"
                 for node, b in self.tables if any(b.split.values())]
        if split:
            parts.append("split codes " + ", ".join(split))
        return "; ".join(parts)


def _candidates(graph):
    for node in graph.nodes:
        if (node.op == "call_function"
                and (node.target in ALLOWLIST or node.target in INPLACE)
                and node.users and node.args and _is_dequant(node.args[0])):
            yield node


def _out_qparams(A):
    """q2's args when every user of A is one identical quantize_per_tensor
    (dtype included) with constant, valid qparams: the PT2E norm."""
    users = list(A.users)
    if not users or not all(_is(u, Q) and u.args[0] is A and not u.kwargs
                            for u in users):
        return None
    args = {tuple(u.args[1:6]) for u in users}
    if len(args) != 1 or any(len(u.args) != 6 for u in users):
        return None
    qp = _scalar_qparams(users[0].args)
    if qp is None or not (math.isfinite(qp[0]) and qp[0] > 0):
        return None
    return tuple(users[0].args[1:6])


def fold_quant_luts(exported_program):
    """Rewrite every allowlisted activation between static quantisers into
    espressif::quant_lut, in place, on an ATen-dialect ExportedProgram (or a
    dict of them, one per method). Returns a QuantLutReport. A graph with no
    candidate is not touched at all (no recompile, no dead-code pass), so a
    symmetric export is byte-identical with the pass on or off."""
    if isinstance(exported_program, dict):
        report = QuantLutReport()
        for ep in exported_program.values():
            part = fold_quant_luts(ep)
            report.tables += part.tables
            report.refused += part.refused
            report.constants.update(part.constants)
        return report

    from executorch.backends.transforms.utils import create_constant_placeholder
    from torch.export.graph_signature import InputKind

    ep = exported_program
    graph = ep.graph_module.graph
    report = QuantLutReport()
    for A in list(_candidates(graph)):
        op = ALLOWLIST.get(A.target) or INPLACE[A.target]
        rule = refusal(A)
        if rule:
            report.refused.append((A.name, op, rule))
            continue
        dq1 = A.args[0]
        q1 = dq1.args[0]
        build = build_lut(A, _out_qparams(A))
        name = build.name
        first = next(n for n in graph.nodes if n.op == "placeholder")
        with graph.inserting_before(first):
            table = create_constant_placeholder(
                ep, graph, name, InputKind.CONSTANT_TENSOR, build.table)
        with graph.inserting_before(A):
            lut = graph.call_function(QUANT_LUT, (
                q1.args[0], table, build.inv_scale, build.zero_point,
                build.quant_min, build.quant_max))
        lut.meta.update({key: A.meta[key] for key in
                         ("val", "tensor_meta", "stack_trace", "nn_module_stack")
                         if key in A.meta})
        A.replace_all_uses_with(lut)
        graph.erase_node(A)
        # dq1, then q1, only once nothing else reads them: either may feed a
        # graph output or another portable op.
        for dead in (dq1, q1):
            if not dead.users:
                graph.erase_node(dead)
        report.tables.append((lut.name, build))
        report.constants.setdefault(table.name, []).append(lut.name)
    if report.tables:
        graph.lint()
        ep.graph_module.recompile()
    return report


# --- After partitioning ------------------------------------------------------------

def _constant(ep, node):
    from torch._export.utils import (get_buffer, get_lifted_tensor_constant,
                                     get_param)

    if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
        return None
    for getter in (get_lifted_tensor_constant, get_param, get_buffer):
        value = getter(ep, node)
        if value is not None:
            return value
    return None


def _same_bits(a, b):
    """Bitwise, through int32 views: NaN entries compare equal to themselves."""
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def elide_lut_pairs(ep):
    """Drop q -> dq pairs after a quant_lut whose table they map to itself.

    T's output is always one of its table's entries, and on those entries the
    pair is an identity, so dq can read T directly; q goes once nothing reads
    it. Edge-dialect ExportedProgram, in place; returns the pairs removed."""
    graph = ep.graph_module.graph
    removed = 0
    for T in [n for n in graph.nodes if n.op == "call_function"
              and _schema(n.target) == ("espressif::quant_lut", "default")]:
        V = _constant(ep, T.args[1])
        if V is None:
            continue
        for q in list(T.users):
            if not _is(q, Q) or q.args[0] is not T or q.kwargs:
                continue
            for dq in list(q.users):
                if not _is(dq, DQ) or dq.args[0] is not q:
                    continue
                with torch.no_grad():
                    back = _dq_op()(_q_op()(V, *q.args[1:]), *dq.args[1:],
                                    **dq.kwargs)
                if _same_bits(back, V):
                    dq.replace_all_uses_with(T)
                    graph.erase_node(dq)
                    removed += 1
            if not q.users:
                graph.erase_node(q)
    if removed:
        graph.lint()
        ep.graph_module.recompile()
    return removed


# --- The conformance golden (test/runtime/quant_lut_golden.h) -----------------------

# The quantize step's five scales: a power of two, three float32 values, and a
# double that is not one (0.007: its multiplier is quantize_per_tensor's
# float64 quotient rounded once).
GOLDEN_SCALES = (2.0 ** -7, 0.05000000074505806, 0.007, 0.30000001192092896,
                 3.700000047683716)
GOLDEN_K = (-520, 520)
# +0, -0, +-denorm_min, +-FLT_MAX, +-inf, NaN (quant_lut.cpp kSpecials).
GOLDEN_SPECIALS = (0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7F7FFFFF,
                   0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000)
GOLDEN_RANGES = ((-128, 127, torch.int8), (-127, 127, torch.int8),
                 (0, 255, torch.uint8))


def golden_inputs(inv):
    """[(k, [x - 2 ulp, x - 1 ulp, x, x + 1 ulp, x + 2 ulp])] for every k whose
    tie k + 1/2 some float32 x reaches exactly, fl(x * inv) == k + 1/2. The
    search is quant_lut.cpp's, step for step: x0 = fl32((k + 1/2) / inv) in
    float64, then 1, 2 and 3 ulps up and down, alternating."""
    import numpy as np

    f32, inf = np.float32, np.float32(np.inf)
    inv = f32(inv)
    groups = []
    for k in range(GOLDEN_K[0], GOLDEN_K[1] + 1):
        target = f32(k + 0.5)
        up = down = f32(np.float64(k + 0.5) / np.float64(inv))
        tie = None
        for step in range(4):
            if step:
                up, down = np.nextafter(up, inf), np.nextafter(down, -inf)
            for cand in ((up,) if step == 0 else (up, down)):
                if f32(cand * inv) == target:
                    tie = cand
                    break
            if tie is not None:
                break
        if tie is None:
            continue
        d1 = np.nextafter(tie, -inf)
        u1 = np.nextafter(tie, inf)
        groups.append((k, [np.nextafter(d1, -inf), d1, tie, u1,
                           np.nextafter(u1, inf)]))
    return groups


def golden_header():
    """quant_lut_golden.h's text: for each scale, its multiplier, how many
    ties exist, the FNV-1a of every input the check generates (so the device
    proves it generated the same ones), torch's rounding of each tie input as
    one bit (round(fl(x * inv)) - k, always 0 or 1) and of each special value.

    Torch's own quantize_per_tensor computes them, into int32 codes clamped at
    +-1024 (wider than every tie the check makes, k + 1/2 for |k| <= 520): past
    512 any zero point in [-255, 255] lands on a rail, so
    clamp(r + zp, qmin, qmax) is quantize_per_tensor's code for EVERY zero
    point and range -- which this function asserts over the three ranges and
    the check's zero points before writing anything. NaN is left out of that
    assertion (torch's code for it is platform-defined) and written as 0, so
    the check expects its code to be the zero point, ExecuTorch 1.6's rule."""
    import numpy as np

    from executorch.backends.esp32s3.quantizer import mirrors

    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    rows, bitstream = [], []
    for scale in GOLDEN_SCALES:
        inv = np.float32(mirrors.reciprocal(scale))
        groups = golden_inputs(inv)
        xs = [x for _, five in groups for x in five]
        specials = np.array(GOLDEN_SPECIALS, dtype=np.uint32).view(np.float32)
        inputs = np.concatenate([np.array(xs, dtype=np.float32), specials])
        t = torch.from_numpy(inputs.copy())
        r = quantize(t, scale, 0, -1024, 1024, torch.int32).numpy().astype(np.int64)
        nan = np.isnan(inputs)
        for qmin, qmax, dtype in GOLDEN_RANGES:
            for zp in sorted({qmin, -1, 0, 1, qmax}):
                if not qmin <= zp <= qmax:
                    continue
                want = quantize(t, scale, zp, qmin, qmax, dtype).numpy()
                derived = np.clip(r + zp, qmin, qmax)
                if not np.array_equal(derived[~nan], want[~nan].astype(np.int64)):
                    raise RuntimeError(f"scale {scale} zp {zp} [{qmin}, {qmax}]: "
                                       "the clamped rounding is not torch's code")
        ks = np.repeat([k for k, _ in groups], 5)
        offsets = r[:len(xs)] - ks
        if not np.isin(offsets, (0, 1)).all():
            raise RuntimeError(f"scale {scale}: a tie input rounds outside k, k+1")
        h = 2166136261
        for byte in inputs.tobytes():
            h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
        rows.append((int(inv.view(np.uint32)), len(groups), h,
                     len(bitstream) // 8,
                     [0 if n else int(v) for v, n in zip(r[len(xs):], nan[len(xs):])]))
        bitstream += [int(b) for b in offsets]
        bitstream += [0] * (-len(bitstream) % 8)
    packed = [sum(bit << i for i, bit in enumerate(bitstream[j:j + 8]))
              for j in range(0, len(bitstream), 8)]
    lines = [
        "// Generated by quant_lut.py golden_header() -- do not edit;",
        "// test_quant_lut.py regenerates it with torch and compares.",
        "//",
        "// Torch's quantize_per_tensor on the inputs quant_lut.cpp's conformance",
        "// check generates: per scale, its multiplier fl32(1/s), the number of",
        "// k in [kKMin, kKMax] whose tie k + 1/2 some float32 reaches exactly,",
        "// the FNV-1a of every generated input, one bit per tie input (torch's",
        "// round(fl(x * inv)) - k) and the clamped rounding of each special.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace espressif {",
        "namespace quant_lut {",
        "namespace golden {",
        "",
        f"constexpr int kKMin = {GOLDEN_K[0]};",
        f"constexpr int kKMax = {GOLDEN_K[1]};",
        f"constexpr int kSpecials = {len(GOLDEN_SPECIALS)};",
        "",
        "struct Scale {",
        "  uint32_t inv_bits;",
        "  uint32_t groups;",
        "  uint32_t input_fnv;",
        "  uint32_t bits_offset;  // bytes into kBits",
        "  int16_t specials[kSpecials];",
        "};",
        "",
        "constexpr Scale kScales[] = {",
    ]
    for scale, (inv_bits, groups, h, offset, specials) in zip(GOLDEN_SCALES, rows):
        lines.append(f"    {{0x{inv_bits:08X}u, {groups}u, 0x{h:08X}u, {offset}u,"
                     f"  // s = {scale!r}")
        lines.append("     {" + ", ".join(str(v) for v in specials) + "}},")
    lines += ["};", "", f"constexpr uint8_t kBits[{len(packed)}] = {{"]
    for j in range(0, len(packed), 12):
        lines.append("    " + ", ".join(f"0x{b:02X}" for b in packed[j:j + 12]) + ",")
    lines += ["};", "", "}  // namespace golden", "}  // namespace quant_lut",
              "}  // namespace espressif", ""]
    return "\n".join(lines)
