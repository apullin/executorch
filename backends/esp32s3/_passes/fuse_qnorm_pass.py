# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::qnorm: a decomposed quantised RMSNorm or LayerNorm as one op.

The stock static recipes leave a norm in pieces. Arm decomposes RMSNorm into
x^2, a sum, a multiply by 1/N, an eps add, rsqrt and two multiplies, and
LayerNorm into two means, two subtractions, a square, a third sum, the same
tail and a bias add, with a quantise/dequantise pair after every op (its mean
is a sum and a multiply, each viewed to 4-D and back). Lowered, one Arm
RMSNorm is about 27 plan instructions and one LayerNorm about 54, each a
portable kernel call costing ~60-80 us on the chip.

    espressif::qnorm(Tensor x, Tensor tables, Tensor quant, Tensor? weight,
                     Tensor? bias, int kind, int flags, float eps)

reproduces every intermediate quantisation the pattern has (the kernel's
contract is ops/qnorm.h, where the arithmetic is written out):

  per element   the input's quantiser, the subtractions, the multiplies by
                the rsqrt and the weight, the bias add and the pairs between
                them: one IEEE float32 operation or one quantise each, as
                eager computes them;
  tables        every function of ONE code, built here by torch from the
                graph's own nodes on all 256 codes (quant_lut's construction):
                x^2 through its pair, and everything from a reduction's
                quantiser to the next value an element reads -- the mean
                (x 1/N and its pair), or eps, rsqrt and their pairs;
  reductions    summed in index order from 0.0f, the portable sum's order.
                ATen sums in another, so a reduction's code may differ from
                eager's by one step, as the portable sum, mean and layer_norm
                may; given the reduction codes, every other value is eager's
                bit for bit.

The rewrite (fold_qnorms, in lower_esp32 before fold_quant_luts, on by
default: Esp32Partitioner's fused_norms) matches the patterns
exactly: a pattern
that differs in any op, argument or grid stays decomposed, reported by its
rule.

int16 codes (Arm's a16w8 recipe; kinds 2 and 3, the same patterns) have
65,536 codes, too many for a 256-entry table, so nothing is tabulated per
code but the one function no float32 operation gives:

  per element   x^2 is a float32 multiply and its pair, like every other
                per-element step (aten's pow(x, 2) is x * x; the build checks
                that on every code of the square's input grid);
  programs      each per-row chain (a reduction's quantiser to the value an
                element reads) is a short program run once per row: its
                quantise and dequantise steps and constant operations as
                eager computes them, and rsqrt as an espressif::lut16 table
                from the code before it to the code after it, fitted from
                the graph by lut_fit.py over the codes the program can reach
                (direct when those are few), its flipped codes listed as
                exceptions: exact on every code;
  reductions    as above.

The build checks every program against the chain torch evaluates, on every
code of its reduction's quantiser the sum can reach, bit for bit.

The kernel is ops/qnorm.cpp (host and device, the same source,
-ffp-contract=off); mirrors.qnorm is its numpy mirror; this module's
CompositeExplicitAutograd body is the torch reference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  (quantized_decomposed ops)
from torch.library import Library

from executorch.backends.esp32s3._passes import fold_quant_lut_pass as ql

lib = Library("espressif", "FRAGMENT")
lib.define("qnorm(Tensor x, Tensor tables, Tensor quant, Tensor? weight, "
           "Tensor? bias, int kind, int flags, float eps) -> Tensor")
lib.define("qnorm.out(Tensor x, Tensor tables, Tensor quant, Tensor? weight, "
           "Tensor? bias, int kind, int flags, float eps, *, "
           "Tensor(a!) out) -> Tensor(a!)")
impl = Library("espressif", "IMPL")

# qnorm.h's kinds, flags, slots and tables.
RMS, LAYER = 0, 1
MEAN, EPS = 1, 2
RMS_SLOTS = ("x", "sq", "red", "m")
LAYER_SLOTS = ("x", "red1", "red2", "c1", "c2", "red3", "n", "w")
RMS_TABLES = ("sq", "r")
LAYER_TABLES = ("m1", "m2", "sq", "r")
TABLE_LEN = 256
# An absent slot: quant_min > quant_max.
ABSENT = (1.0, 1.0, 0.0, 1.0, 0.0)
# int16 codes: kinds, slots, program steps and the tables tensor's layout
# (qnorm.h has the same).
RMS16, LAYER16 = 2, 3
RMS16_SLOTS = ("x", "sq", "m")
LAYER16_SLOTS = ("x", "c1", "c2", "sq", "n", "w")
CODE, TABLE, DEQ, MUL, ADD, DIV, RSUB, RDIV = range(1, 9)
STEP_WORDS = 6
TABLE_WORDS = 12
MAX_EXCEPTIONS = 256
CODE_RANGE16 = {torch.int8: (-128, 127), torch.uint8: (0, 255),
                torch.int16: (-32768, 32767)}
RAIL16 = 65536.0


def _slot(quant, s):
    """(inv, scale, zero point, qmin, qmax) as Python numbers, or None."""
    inv, scale, zp, qmin, qmax = (float(v) for v in quant[s])
    if qmin > qmax:
        return None
    return inv, scale, int(zp), int(qmin), int(qmax)


def _codes(v, q):
    """quant_lut's quantise step (its reference, verbatim)."""
    inv, _, zp, qmin, qmax = q
    v = v * inv
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v).clamp(-ql.RAIL,
                                                                  ql.RAIL)
    return (torch.round(v) + zp).clamp(qmin, qmax).to(torch.int64)


def _values(codes, q):
    """dequantize_per_tensor: fl(fl(c - zp) * scale)."""
    _, scale, zp, _, _ = q
    return (codes.to(torch.float32) - zp) * scale


def _fq(v, q):
    return v if q is None else _values(_codes(v, q), q)


def _lookup(tables, t, codes, q):
    return tables[t][codes - q[3]]


def _sum(s):
    """Each row's sum in index order from 0.0f (the kernel's, and the
    portable sum's): one float32 add per element."""
    acc = torch.zeros(s.shape[0], dtype=torch.float32)
    for j in range(s.shape[1]):
        acc = acc + s[:, j]
    return acc.unsqueeze(1)


def _codes16(v, q):
    """lut16's quantise step: quant_lut's with int16 rails (+-2^16)."""
    inv, _, zp, qmin, qmax = q
    v = v * inv
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v).clamp(-RAIL16,
                                                                  RAIL16)
    return (torch.round(v) + zp).clamp(qmin, qmax).to(torch.int64)


def _fq16(v, q):
    return v if q is None else _values(_codes16(v, q), q)


def _f32(word):
    """The float32 whose bits are the int32 `word`, as a 0-dim tensor."""
    return torch.tensor([word], dtype=torch.int32).view(torch.float32)[0]


def _data_words(scheme, count):
    """int32 words of a lut16 table's data: int16 entries two to a word,
    int32 nodes, four-float rows."""
    return (count + 1) // 2 if scheme & 7 == 0 else count if scheme & 7 in (1, 3) \
        else 4 * count


def parse16(tables):
    """A kind 2 or 3 tables tensor (int32, qnorm.h's layout) as (programs,
    luts): each program a list of (op, five int32 arguments), each lut16
    table a dict of its arguments, data and exceptions."""
    t = tables.reshape(-1).contiguous()
    w = t.tolist()
    programs = []
    for p in range(w[0]):
        first, steps = w[2 + 2 * p], w[3 + 2 * p]
        programs.append([(w[first + STEP_WORDS * s],
                          w[first + STEP_WORDS * s + 1:
                            first + STEP_WORDS * (s + 1)])
                         for s in range(steps)])
    luts = []
    base = 2 + 2 * w[0]
    for i in range(w[1]):
        (scheme, lo, hi, shift, G, out_min, out_max, count, at, exceptions,
         exc_at, _) = w[base + TABLE_WORDS * i:base + TABLE_WORDS * (i + 1)]
        data = t[at:at + _data_words(scheme, count)]
        if scheme & 7 == 0:
            data = data.view(torch.int16)[:count]
        elif scheme & 7 in (2, 4):
            data = data.view(torch.float32).reshape(count, 4)
        luts.append(dict(scheme=scheme, lo=lo, hi=hi, shift=shift, G=G,
                         out_min=out_min, out_max=out_max, data=data,
                         exceptions=[(w[exc_at + 2 * e], w[exc_at + 2 * e + 1])
                                     for e in range(exceptions)]))
    return programs, luts


def _table16(lut, c):
    """A program's TABLE step: lut16's table step (fold_lut16_pass.codes16),
    then the exceptions (input code -> output code)."""
    from executorch.backends.esp32s3._passes import fold_lut16_pass

    y = fold_lut16_pass.codes16(c, lut["data"], lut["scheme"], lut["lo"],
                            lut["hi"], lut["shift"], lut["G"], lut["out_min"],
                            lut["out_max"])
    for code, out in lut["exceptions"]:
        y = torch.where(c == code, torch.full_like(y, out), y)
    return y


def run_program(steps, luts, v):
    """One per-row program on v (float32 [rows, 1]): its value, and the codes
    of its first quantise step (the reduction's)."""
    c = first = None
    for op, a in steps:
        if op == CODE:
            c = _codes16(v, (_f32(a[0]), None, a[2], a[3], a[4]))
            first = c if first is None else first
        elif op == TABLE:
            c = _table16(luts[a[0]], c)
        elif op == DEQ:
            v = (c.to(torch.float32) - a[2]) * _f32(a[1])
        else:
            k = _f32(a[0])
            v = (v * k if op == MUL else v + k if op == ADD else
                 v / k if op == DIV else k - v if op == RSUB else k / v)
    return v, first


def _reference16(x, tables, quant, w, b, kind):
    """Kinds 2 and 3: qnorm.h's int16 arithmetic, one torch op per
    rounding."""
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    programs, luts = parse16(tables)
    if kind == RMS16:
        qx, qsq, qm = (_slot(quant, s) for s in range(len(RMS16_SLOTS)))
        a = rows if qx is None else _values(_codes16(rows, qx), qx)
        r, k = run_program(programs[0], luts, _sum(_fq16(a * a, qsq)))
        y = a * r
        if w is not None:
            y = _fq16(y, qm) * w
        codes = (k,)
    else:
        qx, qc1, qc2, qsq, qn, qw = (_slot(quant, s)
                                     for s in range(len(LAYER16_SLOTS)))
        a = _values(_codes16(rows, qx), qx)
        acc1 = _sum(a)
        mu1, k1 = run_program(programs[0], luts, acc1)
        mu2, k2 = run_program(programs[1], luts, acc1)
        c2 = _fq16(a - mu2, qc2)
        r, k3 = run_program(programs[2], luts, _sum(_fq16(c2 * c2, qsq)))
        y = _fq16(a - mu1, qc1) * r
        if w is not None:
            y = _fq16(y, qn) * w
            if b is not None:
                y = _fq16(y, qw) + b
        elif b is not None:
            y = _fq16(y, qn) + b
        codes = (k1, k2, k3)
    return y.reshape(x.shape).contiguous(), codes


def reduction_slots(kind):
    """The slots whose codes the reductions give, in reference_with_codes'
    order."""
    return ("red",) if kind in (RMS, RMS16) else ("red1", "red2", "red3")


def reference_with_codes(x, tables, quant, weight, bias, kind, flags, eps):
    """The reference, and each reduction's codes per row (reduction_slots'
    order): qnorm.h's arithmetic, one torch op per rounding."""
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    w = None if weight is None else weight.reshape(1, n)
    b = None if bias is None else bias.reshape(1, n)
    if kind in (RMS16, LAYER16):
        return _reference16(x, tables, quant, w, b, kind)
    if kind == RMS:
        qx, qsq, qred, qm = (_slot(quant, s) for s in range(len(RMS_SLOTS)))
        if qx is not None:
            cx = _codes(rows, qx)
            a = _values(cx, qx)
            s = _lookup(tables, 0, cx, qx)
        else:
            a = rows
            s = _fq(rows * rows, qsq)
        v = _sum(s)
        if flags & MEAN:
            v = v / float(n)
        if flags & EPS:
            v = v + eps
        k = _codes(v, qred)
        r = _lookup(tables, 1, k, qred)
        y = a * r
        if w is not None:
            y = _fq(y, qm) * w
        codes = (k,)
    else:
        qx, qr1, qr2, qc1, qc2, qr3, qn, qw = (
            _slot(quant, s) for s in range(len(LAYER_SLOTS)))
        a = _values(_codes(rows, qx), qx)
        acc1 = _sum(a)
        k1, k2 = _codes(acc1, qr1), _codes(acc1, qr2)
        mu1 = _lookup(tables, 0, k1, qr1)
        mu2 = _lookup(tables, 1, k2, qr2)
        acc2 = _sum(_lookup(tables, 2, _codes(a - mu2, qc2), qc2))
        k3 = _codes(acc2, qr3)
        r = _lookup(tables, 3, k3, qr3)
        y = _fq(a - mu1, qc1) * r
        if w is not None:
            y = _fq(y, qn) * w
            if b is not None:
                y = _fq(y, qw) + b
        elif b is not None:
            y = _fq(y, qn) + b
        codes = (k1, k2, k3)
    return y.reshape(x.shape).contiguous(), codes


def qnorm(x, tables, quant, weight, bias, kind, flags, eps):
    """The reference: qnorm.h's arithmetic, one torch op per rounding."""
    return reference_with_codes(x, tables, quant, weight, bias, kind, flags,
                                eps)[0]


def _qnorm_out(x, tables, quant, weight, bias, kind, flags, eps, *, out):
    out.copy_(qnorm(x, tables, quant, weight, bias, kind, flags, eps))
    return out


impl.impl("qnorm", qnorm, "CompositeExplicitAutograd")
impl.impl("qnorm.out", _qnorm_out, "CompositeExplicitAutograd")


@torch.library.register_fake("espressif::qnorm")
def _qnorm_fake(x, tables, quant, weight, bias, kind, flags, eps):
    return torch.empty_like(x)


QNORM = torch.ops.espressif.qnorm.default




# --- Matching -------------------------------------------------------------------

Q, DQ = ql.Q, ql.DQ
# Data movement a norm's values pass through unchanged: pure reshapes (Arm
# views a mean's input to 4-D and its output back) and float32-to-float32
# casts (HF's .to(input_dtype)). Each may be followed by a pair on its
# input's grid: an identity, evaluated as written.
MOVES = {"aten::view", "aten::reshape", "aten::_unsafe_view",
         "aten::unsqueeze", "aten::squeeze", "aten::flatten", "aten::clone",
         "aten::alias", "aten::contiguous", "aten::to", "aten::_to_copy"}
BINARY = {"aten::mul": "mul", "aten::add": "add", "aten::sub": "sub",
          "aten::div": "div"}
REDUCTIONS = {("aten::sum", "dim_IntList"): "sum", ("aten::mean", "dim"): "mean"}
ASSERTS = {"aten::_assert_tensor_metadata"}


class Refused(Exception):
    """A pattern kept decomposed, by its rule (the report lists it)."""

    def __init__(self, rule, detail=""):
        super().__init__(f"{rule}: {detail}" if detail else rule)
        self.rule = rule
        self.detail = detail


def _call(node):
    return isinstance(node, torch.fx.Node) and node.op == "call_function"


def _name(node):
    return ql._schema(node.target) if _call(node) else (None, None)


def _is_q(node):
    return ql._is(node, Q)


def _is_dq(node):
    return ql._is(node, DQ)


def _is_move(node):
    name, _ = _name(node)
    if name not in MOVES:
        return False
    if name in ("aten::to", "aten::_to_copy"):
        # a float32 -> float32 cast and nothing else (device, layout,
        # memory format: refused as an op the pattern does not have)
        src, val = node.args[0], node.meta.get("val")
        extra = {k: v for k, v in node.kwargs.items() if k != "dtype"}
        return (not extra and isinstance(val, torch.Tensor)
                and val.dtype == torch.float32
                and _dtype(src) == torch.float32)
    return True


def _dtype(node):
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return val.dtype if isinstance(val, torch.Tensor) else None


def _shape(node):
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return tuple(val.shape) if isinstance(val, torch.Tensor) else None


def _binary(node):
    name, overload = _name(node)
    return BINARY.get(name) if overload in ("Tensor", "Scalar") else None


def _reduction(node):
    return REDUCTIONS.get(_name(node))


def _users(node):
    """Readers of node's value (dtype/shape asserts read none)."""
    return [u for u in node.users if _name(u)[0] not in ASSERTS]


def _qargs(node):
    return tuple(node.args[1:6])


def _check_pair(q, dq):
    """A static int8 or int16 pair: constant arguments, the same on both
    nodes, a positive finite scale with a finite reciprocal. Raises
    Refused."""
    qp, dp = ql._scalar_qparams(q.args), ql._scalar_qparams(dq.args)
    if qp is None or dp is None or q.kwargs:
        raise Refused("qnorm-input-not-static", q.name)
    if _qargs(q) != _qargs(dq) or dq.kwargs.get("out_dtype") not in (
            None, torch.float32):
        raise Refused("qnorm-dq-mismatch", dq.name)
    scale, zp, qmin, qmax, dtype = qp
    if dtype not in CODE_RANGE16:
        raise Refused("qnorm-wide-codes", f"{q.name}: {dtype}")
    lo, hi = CODE_RANGE16[dtype]
    if qmin < lo or qmax > hi:
        raise Refused("qnorm-wide-codes", q.name)
    inv = ql.reciprocal(scale)
    if not (math.isfinite(scale) and scale > 0 and qmin < qmax
            and qmin <= zp <= qmax and math.isfinite(inv) and inv > 0):
        raise Refused("qnorm-bad-qparams", q.name)
    return qp


def _is_constant(ep, node, seen=None):
    """A Python number, a lifted parameter, buffer or constant, or an op on
    nothing else (a dequantised frozen weight, its reshape)."""
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return True
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "placeholder":
        return ql._constant(ep, node) is not None
    if node.op != "call_function":
        return False
    inputs = node.all_input_nodes
    return bool(inputs) and all(_is_constant(ep, a) for a in inputs)


def _constant_value(ep, node):
    """The value of a constant (_is_constant), as torch computes it."""
    if not isinstance(node, torch.fx.Node):
        return node
    if node.op == "placeholder":
        value = ql._constant(ep, node)
        return value.detach() if isinstance(value, torch.Tensor) else value

    def value(a):
        if isinstance(a, torch.fx.Node):
            return _constant_value(ep, a)
        if isinstance(a, (list, tuple)):
            return type(a)(value(v) for v in a)
        return a

    with torch.no_grad():
        return node.target(*value(node.args), **{k: value(v) for k, v in
                                                 node.kwargs.items()})


def _binary_parts(ep, node):
    """(kind, tensor operand, constant operand, constant first) of a binary op
    with one constant operand, alpha 1; None otherwise."""
    kind = _binary(node)
    if kind is None or len(node.args) != 2:
        return None
    if any(k != "alpha" or v != 1 for k, v in node.kwargs.items()):
        return None
    a, b = node.args
    if _is_constant(ep, b) and not _is_constant(ep, a):
        return kind, a, b, False
    if _is_constant(ep, a) and not _is_constant(ep, b):
        return kind, b, a, True
    return None


@dataclass
class Chain:
    """A function of one code: the quantize whose code keys it, then every
    node after it in order (dequantise, quantize, a constant operation,
    rsqrt, a square), ending at the value an element or row reads. Moves are
    left out: they change no value."""

    key: torch.fx.Node
    steps: list


def _back(ep, node, nodes, stop):
    """Walk back from `node` (a value something reads) through moves, pairs
    and constant operations until stop(n) holds. Returns (n, steps in
    forward order). Every node passed joins `nodes`."""
    steps = []
    while True:
        if stop(node):
            return node, steps
        if _is_move(node):
            nodes.add(node)
            node = node.args[0]
            continue
        if _is_dq(node):
            q = node.args[0]
            if not _is_q(q):
                raise Refused("qnorm-input-not-static", node.name)
            _check_pair(q, node)
            nodes.update((node, q))
            steps[:0] = [q, node]
            node = q.args[0]
            continue
        parts = _binary_parts(ep, node) if _call(node) else None
        if parts is not None:
            nodes.add(node)
            steps.insert(0, node)
            node = parts[1]
            continue
        raise Refused("qnorm-op", f"{_name(node)[0] or node} inside the pattern")


def _forward(node, nodes):
    """Follow node's one reader through its pair and moves (each move with or
    without an identity pair after it) to the next operation. Returns
    (operation, steps in forward order, the value node it reads)."""
    steps = []
    while True:
        users = _users(node)
        if len(users) != 1:
            raise Refused("qnorm-extra-consumer",
                          f"{node.name} has {len(users)} readers")
        user = users[0]
        if _is_move(user):
            nodes.add(user)
            node = user
            continue
        if _is_q(user):
            dqs = _users(user)
            if len(dqs) != 1 or not _is_dq(dqs[0]):
                raise Refused("qnorm-extra-consumer",
                              f"{user.name} has {len(dqs)} readers")
            _check_pair(user, dqs[0])
            nodes.update((user, dqs[0]))
            steps += [user, dqs[0]]
            node = dqs[0]
            continue
        return user, steps, node


def _source(node):
    """Through moves and pairs back to what is quantised: (the value, the
    first quantize met or None)."""
    first = None
    while True:
        if _is_move(node):
            node = node.args[0]
        elif _is_dq(node) and _is_q(node.args[0]):
            first = first or node.args[0]
            node = node.args[0].args[0]
        else:
            return node, first


def _input_quantiser(node, shape):
    """An element read (`node`, what a pattern op reads) back to the norm's
    input: through moves and pairs on one grid (identities: Arm's views to
    4-D and back, HF's float32 casts) to the outermost pair whose quantize
    reads a tensor of the per-element `shape`. Returns (x, its quantize), or
    (node, None) for an fp32 read with no pair. Nothing on the way joins the
    pattern: the op reads x and applies the pair itself, and what the
    rewrite leaves unread goes with it (_erase_unread)."""
    found, grid, cur = None, None, node
    while True:
        if _is_move(cur):
            cur = cur.args[0]
        elif _is_dq(cur) and _is_q(cur.args[0]) and (
                grid is None or _qargs(cur.args[0]) == grid):
            q = cur.args[0]
            _check_pair(q, cur)
            grid = _qargs(q)
            if _shape(q.args[0]) == shape:
                found = q
            cur = q.args[0]
        else:
            break
    if found is not None:
        return found.args[0], found
    if grid is None and _shape(node) == shape:
        return node, None
    raise Refused("qnorm-shape", f"{node.name}: no input of shape {shape}")


def _is_layer(ep, other):
    """M1's other operand is (x - mean) through a pair: a sub whose second
    operand is a reduction's chain."""
    src, _ = _source(other)
    if _binary(src) != "sub" or len(src.args) != 2:
        return None
    try:
        red, _ = _back(ep, src.args[1], set(), stop=_reduction)
    except Refused:
        return None
    return src


def _is_square(node):
    name, overload = _name(node)
    if name == "aten::pow" and overload == "Tensor_Scalar":
        exponent = node.args[1] if len(node.args) > 1 else node.kwargs.get(
            "exponent")
        return (isinstance(exponent, (int, float)) and not isinstance(
            exponent, bool) and float(exponent) == 2.0 and len(node.args) == 2)
    if _binary(node) == "mul" and len(node.args) == 2 and not node.kwargs:
        a, b = node.args
        return isinstance(a, torch.fx.Node) and _source(a) == _source(b)
    return False


def _pair_of(steps, what):
    """The one quantiser a run of steps applies (identity pairs after a move
    allowed, on the same grid), or None for no pair."""
    qs = [n for n in steps if _is_q(n)]
    if any(not (_is_q(n) or _is_dq(n)) for n in steps):
        raise Refused("qnorm-op", f"an operation after {what}")
    if any(_qargs(q) != _qargs(qs[0]) for q in qs[1:]):
        raise Refused("qnorm-grid", f"{what} is requantised")
    return qs[0] if qs else None


def _check_reduction(red, n):
    _, dims, *rest = red.args
    keepdim = rest[0] if rest else red.kwargs.get("keepdim", False)
    shape = _shape(red.args[0])
    if (red.kwargs.get("dtype") is not None or not isinstance(dims, (list, tuple))
            or len(dims) != 1 or not shape or keepdim is not True
            or int(dims[0]) % len(shape) != len(shape) - 1 or shape[-1] != n):
        raise Refused("qnorm-reduction", f"{red.name} is not a reduction over "
                      "the last axis with keepdim")


def _operand(node, value):
    """The other operand of a binary node one of whose operands is value."""
    a, b = node.args
    if a is value and b is not value:
        return b
    if b is value and a is not value:
        return a
    raise Refused("qnorm-op", f"{node.name} does not read {value.name} once")


@dataclass
class Match:
    """One norm, as fold_qnorms rewrites it."""

    kind: int
    x: torch.fx.Node             # the fp32 tensor the op reads
    output: torch.fx.Node        # the pattern's last op
    nodes: set                   # every node the op replaces, output included
    slots: dict                  # slot name -> its quantize node, or None
    tables: dict                 # table name -> Chain
    flags: int = 0
    eps: float = 0.0
    weight: Optional[torch.fx.Node] = None
    bias: Optional[torch.fx.Node] = None
    reductions: list = field(default_factory=list)
    elem_reads: list = field(default_factory=list)   # per-element operands
    row_reads: list = field(default_factory=list)    # per-row operands


def _mean_chain(ep, mu, nodes, x, q_x, n):
    """A LayerNorm mean: sum over x's values, its pair, then constant
    operations and pairs up to the value `mu` a subtraction reads. Returns
    (sum node, Chain keyed by the sum's quantiser)."""
    red, steps = _back(ep, mu, nodes, stop=_reduction)
    if _reduction(red) != "sum" or not steps or not _is_q(steps[0]):
        raise Refused("qnorm-missing-pair", f"{red.name}: no pair right after "
                      "the mean's sum")
    _check_reduction(red, n)
    xs, qs = _input_quantiser(red.args[0], _shape(x))
    if xs is not x or qs is None or _qargs(qs) != _qargs(q_x):
        raise Refused("qnorm-grid", f"{red.name} sums another input or grid")
    nodes.add(red)
    return red, Chain(steps[0], steps[1:])


def match(ep, rsqrt):
    """The norm around an aten.rsqrt node, or Refused."""
    R = rsqrt
    nodes = {R}
    red, tail = _back(ep, R.args[0], nodes, stop=_reduction)
    nodes.add(red)
    first_q = next((i for i, s in enumerate(tail) if _is_q(s)), None)
    if first_q is None:
        raise Refused("qnorm-missing-pair", "no pair between the reduction "
                      "and rsqrt")
    q_red, pre = tail[first_q], tail[:first_q]
    M1, r_steps, r_value = _forward(R, nodes)
    if _binary(M1) != "mul" or M1.kwargs:
        raise Refused("qnorm-op", f"rsqrt is read by {_name(M1)[0]}")
    nodes.add(M1)
    other = _operand(M1, r_value)
    table_r = Chain(q_red, tail[first_q + 1:] + [R] + r_steps)
    src = _is_layer(ep, other)
    layer = src is not None

    # The reduction's own fp32 tail before its quantiser: a mean's division
    # and an eps add (Cortex-M's RMSNorm), nothing else.
    flags, eps = 0, 0.0
    if _reduction(red) == "mean":
        flags |= MEAN
    for step in pre:
        parts = _binary_parts(ep, step)
        if layer or parts is None or parts[0] != "add" or flags & EPS:
            raise Refused("qnorm-op", f"{step.name} between the reduction "
                          "and its quantiser")
        value = _constant_value(ep, parts[2])
        value = float(value.reshape(())) if isinstance(value, torch.Tensor) \
            else float(value)
        eps = float(torch.tensor(value, dtype=torch.float32))
        flags |= EPS
    if layer and flags:
        raise Refused("qnorm-op", f"{red.name}: a LayerNorm variance as a mean")

    sq, sq_steps = _back(ep, red.args[0], nodes, stop=_is_square)
    nodes.add(sq)
    if layer:
        return _match_layer(ep, R, red, q_red, table_r, M1, other, src, sq,
                            sq_steps, nodes)
    return _match_rms(ep, R, red, q_red, table_r, M1, other, sq, sq_steps,
                      flags, eps, nodes)


def _square_input(sq):
    return sq.args[0]


def _match_rms(ep, R, red, q_red, table_r, M1, other, sq, sq_steps, flags,
               eps, nodes):
    shape = _shape(M1)
    x, q_x = _input_quantiser(other, shape)
    x_sq, q_sq_in = _input_quantiser(_square_input(sq), shape)
    if _binary(sq) == "mul":
        x_b, q_b = _input_quantiser(sq.args[1], shape)
        if x_b is not x_sq or (q_b is None) != (q_sq_in is None) or (
                q_b is not None and _qargs(q_b) != _qargs(q_sq_in)):
            raise Refused("qnorm-grid", f"{sq.name}: two different operands")
    if x_sq is not x or (q_x is None) != (q_sq_in is None) or (
            q_x is not None and _qargs(q_x) != _qargs(q_sq_in)):
        raise Refused("qnorm-grid", "the square and the multiply read the "
                      "input on different grids")
    n = _shape(x)[-1] if _shape(x) else None
    if not n:
        raise Refused("qnorm-dtype", f"{x}: no shape")
    _check_reduction(red, n)
    q_sq = _pair_of(sq_steps, sq.name)
    slots = {"x": q_x, "sq": q_sq, "red": q_red, "m": None}
    tables = {"r": table_r}
    if q_x is not None:
        tables["sq"] = Chain(q_x, [("dq", q_x), sq] + sq_steps)
    m = Match(RMS, x, M1, nodes, slots, tables, flags, eps, reductions=[red],
              elem_reads=[other], row_reads=[_operand(M1, other)])
    _extend_weight(ep, m, ("m",), nodes)
    return m


def _extend_weight(ep, m, slot_names, nodes):
    """Grow the match over a multiply by a constant weight (and, for a
    LayerNorm, a bias add after it): each joins only if it is the one reader
    of the pattern's output so far, through at most one pair."""
    for slot, kind in zip(slot_names, ("mul", "add")):
        trial = set()
        try:
            nxt, steps, value = _forward(m.output, trial)
        except Refused:
            return
        parts = _binary_parts(ep, nxt) if _call(nxt) else None
        if parts is None or parts[0] != kind or parts[1] is not value:
            return
        q = _pair_of(steps, m.output.name)
        const = parts[2]
        if not isinstance(const, torch.fx.Node):
            return
        nodes.update(trial)
        nodes.add(nxt)
        m.elem_reads.append(value)
        m.slots[slot] = q
        if kind == "mul":
            m.weight = const
        else:
            m.bias = const
        m.output = nxt
        if m.kind == RMS:
            return


def _match_layer(ep, R, red, q_red, table_r, M1, other, sub1, sq, sq_steps,
                 nodes):
    # x - mean1, through its pair, into the multiply by rsqrt.
    _, c1_steps = _back(ep, other, nodes, stop=lambda n: n is sub1)
    q_c1 = _pair_of(c1_steps, sub1.name)
    # x - mean2, through its pair, squared, through its pair, summed.
    sub2, c2_steps = _back(ep, _square_input(sq), nodes,
                           stop=lambda n: _binary(n) == "sub")
    if _binary(sq) == "mul" and _source(sq.args[1])[0] is not sub2:
        raise Refused("qnorm-grid", f"{sq.name}: two different operands")
    if not c2_steps or not _is_q(c2_steps[0]):
        raise Refused("qnorm-missing-pair", f"{sub2.name}: no pair before "
                      "the square")
    q_c2 = c2_steps[0]
    _pair_of(c2_steps, sub2.name)
    for sub in (sub1, sub2):
        if sub.kwargs or len(sub.args) != 2:
            raise Refused("qnorm-op", f"{sub.name}: alpha")
        nodes.add(sub)
    shape = _shape(M1)
    x, q_x = _input_quantiser(sub1.args[0], shape)
    x2, q_x2 = _input_quantiser(sub2.args[0], shape)
    if q_x is None or x2 is not x or q_x2 is None or _qargs(q_x2) != _qargs(q_x):
        raise Refused("qnorm-grid", "the two subtractions read different "
                      "inputs or grids")
    n = _shape(x)[-1] if _shape(x) else None
    if not n:
        raise Refused("qnorm-dtype", f"{x}: no shape")
    _check_reduction(red, n)
    red1, table_m1 = _mean_chain(ep, sub1.args[1], nodes, x, q_x, n)
    red2, table_m2 = _mean_chain(ep, sub2.args[1], nodes, x, q_x, n)
    slots = {"x": q_x, "red1": table_m1.key, "red2": table_m2.key, "c1": q_c1,
             "c2": q_c2, "red3": q_red, "n": None, "w": None}
    tables = {"m1": table_m1, "m2": table_m2,
              "sq": Chain(q_c2, c2_steps[1:] + [sq] + sq_steps), "r": table_r}
    m = Match(LAYER, x, M1, nodes, slots, tables, reductions=[red1, red2, red],
              elem_reads=[other, sub1.args[0], sub2.args[0]],
              row_reads=[_operand(M1, other), sub1.args[1], sub2.args[1]])
    _extend_weight(ep, m, ("n", "w"), nodes)
    return m


def _rowshape(x):
    return tuple(_shape(x)[:-1]) + (1,)


def _check(ep, m):
    """What the kernel assumes of a match: an fp32 contiguous input, the
    output its shape, per-element operands of that shape and per-row ones
    of [..., 1], weight and bias over the last axis, and no reader outside
    the pattern of anything inside it but the output. Raises Refused."""
    x = m.x
    xv = x.meta.get("val") if isinstance(x, torch.fx.Node) else None
    ov = m.output.meta.get("val")
    if (not isinstance(xv, torch.Tensor) or xv.dtype != torch.float32
            or not isinstance(ov, torch.Tensor) or ov.dtype != torch.float32):
        raise Refused("qnorm-dtype", f"{x}: not float32")
    if not ql._contiguous(xv):
        raise Refused("qnorm-layout", f"{x.name}: not contiguous")
    if tuple(ov.shape) != tuple(xv.shape):
        raise Refused("qnorm-shape", f"{m.output.name}: {tuple(ov.shape)} "
                      f"from {tuple(xv.shape)}")
    n = xv.shape[-1]
    rows = _rowshape(x)
    for node in m.elem_reads:
        if _shape(node) != tuple(xv.shape):
            raise Refused("qnorm-shape", f"{node.name}: {_shape(node)} read "
                          f"per element of {tuple(xv.shape)}")
    for node in m.row_reads:
        if _shape(node) != rows:
            raise Refused("qnorm-shape", f"{node.name}: {_shape(node)} read "
                          f"per row of {tuple(xv.shape)}")
    for const in (m.weight, m.bias):
        if const is None:
            continue
        value = _constant_value(ep, const)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.numel() != n or value.shape[-1] != n):
            raise Refused("qnorm-weight", f"{const}: not float32 over the "
                          "last axis")
    for node in m.nodes - {m.output}:
        outside = [u.name for u in _users(node) if u not in m.nodes]
        if outside:
            raise Refused("qnorm-extra-consumer",
                          f"{node.name} is read by {', '.join(outside)}")


# --- Building -------------------------------------------------------------------

def _q_op():
    return torch.ops.quantized_decomposed.quantize_per_tensor.default


def _dq_op():
    return torch.ops.quantized_decomposed.dequantize_per_tensor.default


def evaluate_chain(ep, chain):
    """chain over every code of its key, by torch, each step as the graph
    computes it (its own quantize and dequantise arguments, its constants,
    rsqrt in quant_lut's canonical vector context): float32 [codes]."""
    _, _, qmin, qmax, dtype = ql._scalar_qparams(chain.key.args)
    cur = torch.arange(qmin, qmax + 1, dtype=torch.int64).to(dtype)
    with ql.one_thread(), torch.no_grad():
        for step in chain.steps:
            if isinstance(step, tuple):                  # ("dq", q)
                cur = _dq_op()(cur, *step[1].args[1:6])
            elif _is_q(step):
                cur = _q_op()(cur, *step.args[1:6])
            elif _is_dq(step):
                cur = _dq_op()(cur, *step.args[1:6], **step.kwargs)
            elif _name(step)[0] == "aten::rsqrt":
                cur = ql.evaluate(step.target, cur)
            elif _is_square(step):
                cur = (step.target(cur, *step.args[1:]) if _name(step)[0] ==
                       "aten::pow" else step.target(cur, cur))
            else:
                kind, _, const, first = _binary_parts(ep, step)
                value = _constant_value(ep, const)
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        raise Refused("qnorm-op", f"{step.name}: a constant of "
                                      f"{value.numel()} elements in a row's chain")
                    value = value.reshape(())
                cur = step.target(value, cur) if first else step.target(cur, value)
                cur = cur.reshape(-1)
    if cur.dtype != torch.float32 or cur.numel() != qmax - qmin + 1:
        raise Refused("qnorm-dtype", f"chain from {chain.key.name} gives "
                      f"{cur.dtype} [{cur.numel()}]")
    return cur.contiguous()


def quant_row(q):
    """A slot's row: (fl32(1/s), fl32(s), zp, qmin, qmax); ABSENT for None."""
    if q is None:
        return ABSENT
    scale, zp, qmin, qmax, _ = ql._scalar_qparams(q.args)
    return (ql.reciprocal(scale), float(torch.tensor(scale, dtype=torch.float32)),
            float(zp), float(qmin), float(qmax))


@dataclass
class QnormBuild:
    """One fused norm's constants."""

    kind: int
    flags: int
    eps: float
    tables: torch.Tensor          # float32 [T, 256]
    quant: torch.Tensor           # float32 [S, 5]
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]
    n: int
    slots: dict                   # slot -> (scale, zp, qmin, qmax) or None
    luts: list = field(default_factory=list)   # int16: (lut_fit.Table,
                                               # exceptions) per TABLE step

    def summary(self):
        kind = "rmsnorm" if self.kind in (RMS, RMS16) else "layernorm"
        pairs = sum(v is not None for v in self.slots.values())
        wide = ""
        if self.kind in (RMS16, LAYER16):
            wide = ", int16, " + ", ".join(
                f"rsqrt {t.scheme} {t.nbytes} B + {len(e)} exception(s)"
                for t, e in self.luts)
        return (f"{kind} over {self.n}, {pairs} pair(s) inside"
                f"{', mean' if self.flags & MEAN else ''}"
                f"{', eps' if self.flags & EPS else ''}"
                f"{', weight' if self.weight is not None else ''}"
                f"{', bias' if self.bias is not None else ''}{wide}")


def build(ep, m):
    if _wide(m):
        return build16(ep, m)
    names = RMS_TABLES if m.kind == RMS else LAYER_TABLES
    slot_names = RMS_SLOTS if m.kind == RMS else LAYER_SLOTS
    tables = torch.zeros(len(names), TABLE_LEN, dtype=torch.float32)
    for i, name in enumerate(names):
        if name in m.tables:
            values = evaluate_chain(ep, m.tables[name])
            tables[i, :values.numel()] = values
    quant = torch.tensor([quant_row(m.slots[s]) for s in slot_names],
                         dtype=torch.float32)
    n = _shape(m.x)[-1]

    def vector(const):
        if const is None:
            return None
        return _constant_value(ep, const).reshape(n).to(torch.float32).contiguous()

    slots = {s: (None if m.slots[s] is None else
                 tuple(ql._scalar_qparams(m.slots[s].args)[:4]))
             for s in slot_names}
    return QnormBuild(m.kind, m.flags, m.eps, tables, quant, vector(m.weight),
                      vector(m.bias), n, slots)


def _digest(tag, t):
    data = t.detach().numpy().tobytes() + str(tuple(t.shape)).encode()
    return f"esp32_{tag}_{hashlib.sha256(data).hexdigest()[:16]}"


# --- Building, int16 codes --------------------------------------------------------

def _quantizers(m):
    qs = [q for q in m.slots.values() if q is not None]
    for chain in m.tables.values():
        qs.append(chain.key)
        qs += [s for s in chain.steps if not isinstance(s, tuple) and _is_q(s)]
    return qs


def _wide(m):
    """Whether a quantiser of the pattern has more codes than a 256-entry
    table holds (Arm a16w8's int16): kinds 2 and 3."""
    for q in _quantizers(m):
        _, _, qmin, qmax, dtype = ql._scalar_qparams(q.args)
        if dtype not in ql.CODE_RANGE or qmax - qmin + 1 > TABLE_LEN:
            return True
    return False


def _qrow(q):
    """A quantiser as the kernel reads it (_slot of its quant row), or None."""
    if q is None:
        return None
    return _slot(torch.tensor([quant_row(q)], dtype=torch.float32), 0)


def _bits(value):
    """The int32 bits of float32(value)."""
    return int(torch.tensor([float(value)], dtype=torch.float32)
               .view(torch.int32)[0])


def _same(a, b):
    return a.shape == b.shape and bool(
        (a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32))
        .all())


def _apply(op, k, v):
    return (v * k if op == MUL else v + k if op == ADD else v / k
            if op == DIV else k - v if op == RSUB else k / v)


def _scalar(ep, const, node):
    """A row chain's constant as the float32 eager computes with."""
    value = _constant_value(ep, const)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise Refused("qnorm-op", f"{node.name}: a constant of "
                          f"{value.numel()} elements in a row's chain")
        value = value.reshape(()).item()
    return float(torch.tensor(float(value), dtype=torch.float32))


def _square_pair(chain):
    """The quantiser after a square chain's x^2 (None: the sum reads it
    unquantised)."""
    at = next(i for i, s in enumerate(chain.steps)
              if not isinstance(s, tuple) and _is_square(s))
    return next((s for s in chain.steps[at + 1:] if _is_q(s)), None)


def _check_square(ep, chain, q_sq):
    """Per element, x^2 is fq_SQ(fl(a * a)) on the value a of the code keying
    `chain`: refused unless that equals the chain torch evaluates (pow(x, 2)
    or x * x, then the pair and any identity pairs) bit for bit on every
    code of the key's grid."""
    want = evaluate_chain(ep, chain)
    _, _, qmin, qmax, _ = ql._scalar_qparams(chain.key.args)
    a = _values(torch.arange(qmin, qmax + 1, dtype=torch.int64),
                _qrow(chain.key))
    if not _same(_fq16(a * a, _qrow(q_sq)), want):
        raise Refused("qnorm-program", f"{chain.key.name}: x^2 is not one "
                      "float32 multiply and its pair")


_FITS16 = {}


def _fit_table(q_in, dq_in, f, q_out, lo, hi):
    """f's lut16 table from q_in's codes [lo, hi] to q_out's codes: lut_fit's
    choice (its fewest-flips candidate when nothing meets the bar within the
    budget), then every code of [lo, hi] whose output differs from eager's
    as an exception. Returns (lut_fit.Table, exceptions). Cached per process
    on everything it reads."""
    import numpy as np

    from executorch.backends.esp32s3.quantizer import lut_fit

    key = (repr(q_in.args[1:6]), repr(dq_in.args[1:6]),
           repr(sorted(dq_in.kwargs.items())), str(f.target),
           repr(q_out.args[1:6]), lo, hi)
    if key in _FITS16:
        return _FITS16[key]
    s1, z1, _, _, dtype = ql._scalar_qparams(q_in.args)
    s2, z2, qmin2, qmax2, _ = ql._scalar_qparams(q_out.args)
    codes = torch.arange(lo, hi + 1, dtype=torch.int64)
    with ql.one_thread(), torch.no_grad():
        xs = _dq_op()(codes.to(dtype), *dq_in.args[1:6], **dq_in.kwargs)
        y = ql.evaluate(f.target, xs)
        E = _q_op()(y, *q_out.args[1:6]).to(torch.int64).numpy()

    def values64(xs64):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(np.asarray(xs64, np.float64))

    site = lut_fit.Site("rsqrt", lut_fit.Grid(float(s1), int(z1), lo, hi),
                        lut_fit.Grid(float(s2), int(z2), qmin2, qmax2),
                        values=lambda _xs: y.numpy(), values64=values64,
                        target=E)
    try:
        table = lut_fit.fit(site)
    except lut_fit.BudgetExceeded as error:
        if error.best is None:
            raise Refused("qnorm-table", f"{f.name}: {error}") from error
        table = error.best
    bad = np.nonzero(table.evaluate(site.codes) != E)[0]
    if len(bad) > MAX_EXCEPTIONS:
        raise Refused("qnorm-table", f"{f.name}: {len(bad)} codes flip on "
                      f"{table.describe()}")
    _FITS16[key] = table, [(int(site.codes[i]), int(E[i])) for i in bad]
    return _FITS16[key]


def _lut(table, exceptions):
    """A fitted table as parse16 gives it."""
    from executorch.backends.esp32s3._passes import fold_lut16_pass

    return dict(scheme=table.scheme_id, lo=table.lo, hi=table.hi,
                shift=table.shift, G=table.G, out_min=table.out_qmin,
                out_max=table.out_qmax,
                data=torch.from_numpy(table.data.copy()).to(
                    fold_lut16_pass.TABLE_DTYPE[table.scheme_id]),
                exceptions=list(exceptions))


def _program(ep, chain, pre, luts, nonneg):
    """A per-row chain as a program: `pre` (the reduction's float32 steps
    before its quantiser: a mean's division, Cortex-M's eps), then the
    chain from its key -- each pair a CODE and a DEQ, each constant
    operation one step, and rsqrt between two pairs a CODE, a TABLE and a
    DEQ, its table (appended to `luts`) fitted over the codes the program
    reaches. The program runs on every code of the key the sum can reach
    (a sum of squares is not negative when `nonneg`) and must give the
    chain's values (evaluate_chain) bit for bit. Returns [(op, five int32
    arguments)]. Raises Refused."""
    steps = [chain.key] + list(chain.steps)
    plan, i = [], 0
    while i < len(steps):
        s = steps[i]
        if isinstance(s, tuple):
            raise Refused("qnorm-op", f"{chain.key.name}: a dequantise of "
                          "another pair in a row's chain")
        if _is_q(s):
            dq = steps[i + 1] if i + 1 < len(steps) else None
            if not (_is_dq(dq) and dq.args[0] is s):
                raise Refused("qnorm-missing-pair", f"{s.name}: no dequantise")
            f = steps[i + 2] if i + 2 < len(steps) else None
            if _name(f) == ("aten::rsqrt", "default"):
                q2, dq2 = (steps[i + 3:i + 5] + [None, None])[:2]
                if not (_is_q(q2) and _is_dq(dq2) and dq2.args[0] is q2):
                    raise Refused("qnorm-missing-pair", f"{f.name}: no pair "
                                  "after it")
                plan += [("code", s), ("table", (s, dq, f, q2)), ("deq", q2)]
                i += 5
            else:
                plan += [("code", s), ("deq", s)]
                i += 2
            continue
        parts = _binary_parts(ep, s) if _call(s) else None
        if parts is None:
            raise Refused("qnorm-op", f"{_name(s)[0] or s} in a row's chain "
                          "(int16 codes)")
        kind, _, const, first = parts
        k = _scalar(ep, const, s)
        if kind == "sub":
            op, k = (RSUB, k) if first else (ADD, -k)
        elif kind == "div":
            op = RDIV if first else DIV
        else:
            op = MUL if kind == "mul" else ADD
        plan.append(("op", (op, k)))
        i += 1

    prog = [(op, [_bits(k), 0, 0, 0, 0]) for op, k in pre]
    qkey = _qrow(chain.key)
    lowest = qkey[3]
    if nonneg:
        v0 = torch.zeros(1)
        for op, k in pre:
            v0 = _apply(op, _f32(_bits(k)), v0)
        lowest = max(lowest, int(_codes16(v0, qkey)[0]))
    codes = torch.arange(lowest, qkey[4] + 1, dtype=torch.int64)
    v, c = _values(codes, qkey), None
    for what, arg in plan:
        if what == "code":
            q = _qrow(arg)
            prog.append((CODE, [_bits(q[0]), 0, q[2], q[3], q[4]]))
            c = _codes16(v, q)
        elif what == "deq":
            q = _qrow(arg)
            prog.append((DEQ, [0, _bits(q[1]), q[2], 0, 0]))
            v = _values(c, q)
        elif what == "table":
            table, exceptions = _fit_table(*arg, int(c.min()), int(c.max()))
            prog.append((TABLE, [len(luts), 0, 0, 0, 0]))
            luts.append((table, exceptions))
            c = _table16(_lut(table, exceptions), c)
        else:
            op, k = arg
            prog.append((op, [_bits(k), 0, 0, 0, 0]))
            v = _apply(op, _f32(_bits(k)), v)
    want = evaluate_chain(ep, chain)[codes - qkey[3]]
    if not _same(v, want):
        raise Refused("qnorm-program", f"{chain.key.name}: the program "
                      "differs from the chain it replaces")
    return prog


def _lut_words(table):
    import numpy as np

    data = np.ascontiguousarray(table.data)
    if table.scheme_id & 7 == 0:
        data = data.astype("<i2")
        if len(data) % 2:
            data = np.append(data, np.int16(0))
        return data.view("<i4").tolist()
    if table.scheme_id & 7 in (1, 3):
        return data.astype("<i4").tolist()
    return data.astype("<f4").reshape(-1).view("<i4").tolist()


def encode16(programs, luts):
    """qnorm.h's int32 layout: the counts, a (first word, steps) header per
    program, a header per table, the steps (six words each), then each
    table's data and its exceptions as (input code, output code) pairs."""
    head = 2 + 2 * len(programs) + TABLE_WORDS * len(luts)
    words, heads, body = [len(programs), len(luts)], [], []
    at = head
    for prog in programs:
        heads += [at, len(prog)]
        for op, args in prog:
            body += [op] + list(args)
        at += STEP_WORDS * len(prog)
    for table, exceptions in luts:
        data = _lut_words(table)
        pairs = [v for pair in exceptions for v in pair]
        heads += [table.scheme_id, table.lo, table.hi, table.shift, table.G,
                  table.out_qmin, table.out_qmax, int(table.data.shape[0]),
                  at, len(exceptions), at + len(data), 0]
        body += data + pairs
        at += len(data) + len(pairs)
    return torch.tensor(words + heads + body, dtype=torch.int32)


def build16(ep, m):
    """Kinds 2 and 3 (the module docstring): the per-element slots and the
    per-row programs with their lut16 tables, each checked against what
    torch evaluates for the chain it replaces."""
    n = _shape(m.x)[-1]
    luts = []
    if m.kind == RMS:
        kind, names = RMS16, RMS16_SLOTS
        pre = ([(DIV, float(n))] if m.flags & MEAN else []) + (
            [(ADD, m.eps)] if m.flags & EPS else [])
        programs = [_program(ep, m.tables["r"], pre, luts, nonneg=True)]
        slots = dict(m.slots)
        if "sq" in m.tables:          # the square reads x's codes
            _check_square(ep, m.tables["sq"], slots["sq"])
    else:
        kind, names = LAYER16, LAYER16_SLOTS
        programs = [_program(ep, m.tables[t], [], luts, nonneg=t == "r")
                    for t in ("m1", "m2", "r")]
        slots = dict(m.slots, sq=_square_pair(m.tables["sq"]))
        _check_square(ep, m.tables["sq"], slots["sq"])
    quant = torch.tensor([quant_row(slots[s]) for s in names],
                         dtype=torch.float32)

    def vector(const):
        if const is None:
            return None
        return _constant_value(ep, const).reshape(n).to(torch.float32).contiguous()

    # the summary counts the pattern's pairs as kinds 0 and 1 do
    described = {s: (None if m.slots[s] is None else
                     tuple(ql._scalar_qparams(m.slots[s].args)[:4]))
                 for s in (RMS_SLOTS if m.kind == RMS else LAYER_SLOTS)}
    return QnormBuild(kind, 0, 0.0, encode16(programs, luts), quant,
                      vector(m.weight), vector(m.bias), n, described, luts)


# --- The rewrite ----------------------------------------------------------------

@dataclass
class QnormReport:
    fused: list = field(default_factory=list)     # (node name, QnormBuild)
    refused: list = field(default_factory=list)   # (anchor, rule, detail)
    removed: int = 0                              # nodes the ops replaced

    def summary(self):
        parts = [f"qnorm: {len(self.fused)} norm(s) fused"
                 + (f" ({self.removed} nodes)" if self.fused else "")]
        if self.fused:
            kinds = {}
            for _, b in self.fused:
                kinds[b.summary()] = kinds.get(b.summary(), 0) + 1
            parts.append(", ".join(f"{k} x{v}" for k, v in kinds.items()))
        if self.refused:
            parts.append(f"{len(self.refused)} kept decomposed: " + ", ".join(
                f"{a} {rule}" for a, rule, _ in self.refused))
        return "; ".join(parts)


def _candidates(graph):
    return [n for n in graph.nodes if _name(n) == ("aten::rsqrt", "default")]


def _erase(graph, nodes):
    """Erase `nodes` (no reader left outside them), their asserts first."""
    order = {n: i for i, n in enumerate(graph.nodes)}
    for node in sorted(nodes, key=order.get, reverse=True):
        for user in list(node.users):
            if _name(user)[0] in ASSERTS:
                graph.erase_node(user)
        graph.erase_node(node)


def _erase_unread(ep, graph, candidates):
    """Erase what the rewrite left unread among `candidates` and their
    inputs, repeatedly: the input's dequantise and quantize, a weight's
    dequantise and reshape, and a constant placeholder nothing reads any more
    (its frozen codes; delete_constant_placeholder keeps the signature)."""
    from executorch.backends.transforms.utils import delete_constant_placeholder

    todo = list(candidates)
    removed = 0
    while todo:
        node = todo.pop()
        if getattr(node, "_erased", False) or _users(node):
            continue
        if node.op == "placeholder":
            if (not node.users and ql._constant(ep, node) is not None
                    and node.name not in ep.graph_signature.buffers_to_mutate):
                delete_constant_placeholder(ep, node)
                removed += 1
            continue
        if not _call(node):
            continue
        inputs = node.all_input_nodes
        for user in list(node.users):
            graph.erase_node(user)        # asserts
        graph.erase_node(node)
        removed += 1
        todo.extend(inputs)
    return removed


def fold_qnorms(exported_program):
    """Rewrite every matching decomposed norm into espressif::qnorm, in
    place, on an ATen-dialect ExportedProgram (or a dict of them, one per
    method). Returns a QnormReport. A graph with no rsqrt is not touched,
    nor one with no quantize at all: a float model's norms are not
    quantised norms kept decomposed, and fused_norms is on by default."""
    if isinstance(exported_program, dict):
        report = QnormReport()
        for ep in exported_program.values():
            part = fold_qnorms(ep)
            report.fused += part.fused
            report.refused += part.refused
            report.removed += part.removed
        return report

    from executorch.backends.transforms.utils import create_constant_placeholder
    from torch.export.graph_signature import InputKind

    ep = exported_program
    graph = ep.graph_module.graph
    report = QnormReport()
    if not any(_is_q(n) for n in graph.nodes):
        return report
    for R in _candidates(graph):
        if getattr(R, "_erased", False):
            continue                      # erased with an earlier match
        try:
            m = match(ep, R)
            _check(ep, m)
            b = build(ep, m)
        except Refused as refusal:
            report.refused.append((R.name, refusal.rule, refusal.detail))
            continue
        first = next(n for n in graph.nodes if n.op == "placeholder")
        consts = {}
        with graph.inserting_before(first):
            for tag, value in (("qnorm_t", b.tables), ("qnorm_q", b.quant),
                               ("qnorm_w", b.weight), ("qnorm_b", b.bias)):
                consts[tag] = None if value is None else \
                    create_constant_placeholder(
                        ep, graph, _digest(tag, value),
                        InputKind.CONSTANT_TENSOR, value)
        with graph.inserting_before(m.output):
            node = graph.call_function(QNORM, (
                m.x, consts["qnorm_t"], consts["qnorm_q"], consts["qnorm_w"],
                consts["qnorm_b"], b.kind, b.flags, b.eps))
        node.meta.update({key: m.output.meta[key] for key in
                          ("val", "tensor_meta", "stack_trace", "nn_module_stack")
                          if key in m.output.meta})
        m.output.replace_all_uses_with(node)
        edge = {a for n in m.nodes for a in n.all_input_nodes} - m.nodes
        _erase(graph, m.nodes)
        report.removed += len(m.nodes)
        report.removed += _erase_unread(ep, graph, edge)
        report.fused.append((node.name, b))
    if report.fused:
        graph.lint()
        ep.graph_module.recompile()
    return report
