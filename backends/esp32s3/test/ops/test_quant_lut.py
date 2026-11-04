# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::quant_lut: the tests that need neither a delegate nor static
quantize/dequantize kernels.

  construction  every allowlisted op x 40 seeded qparam pairs (int8 [-128, 127],
                [-127, 127], uint8 [0, 255]; zero points at both ends; scales
                log-uniform 1e-3..1, every fifth a decimal that is not a float32
                value), V_q and V_raw: each table equals torch's evaluation of the
                graph's own nodes on an all-codes, block-uniform input (one
                thread), bitwise (sqrt's: its definition's, and torch's wherever
                its kernel gives those bits); the folded graph gives the same
                bits; identical tables share one constant; the allowlist is
                exactly what the ops export to.
  purity        the measurement: each op in C_vec, C_scalar, C_tail, C_lane,
                C_threads and C_strided on all codes of 50 qparam sets, 2^20
                random float32 in [-20, 20] and special values. The pure set is
                asserted identical across contexts (a NaN out of a NaN in by
                NaN-ness alone); the rest is reported, with the split codes of
                200 random tables per op.
                QUANT_LUT_PURITY_REPORT=<path> writes it as JSON.
  rewrite       one synthetic graph per refusal rule: each refusal names its
                own rule and leaves the graph
                exactly as it was; tables counted after to_edge and after
                partitioning; the stock recipes' census models (XNNPACK static
                and QAT, Arm a8w8, Cortex-M) folded, counted and run against
                the recipe's own model; symmetric exports byte-identical with
                the pass on or off; the pair elision only where the pair is an
                identity.
  mirror        mirrors.quant_lut against the op's torch reference and
                quantize_per_tensor on ties, their neighbours and special values.
  conformance   quant_lut_golden.h is what torch generates; the host's
                --selftest line (the CTest espressif_quant_lut runs it too).
  host rows     luts.pte's four methods on the host runner: the mirror's bits.
  object code   quant_lut.cpp.obj of the parity build has no madd.s / msub.s.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner. It imports
torch (about 2 GB).
"""

import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from executorch.backends.esp32s3._passes import fold_quant_lut_pass as ql
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.host import host_runner

ROOT = Path(__file__).resolve().parent
Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
F32 = np.float32

# Every allowlisted ATen op, with non-default arguments where it takes any;
# gelu in both forms and pow with an integer and a non-integer exponent.
OPS = (
    ("sigmoid", torch.sigmoid),
    ("tanh", torch.tanh),
    ("silu", F.silu),
    ("gelu", F.gelu),
    ("gelu_tanh", lambda x: F.gelu(x, approximate="tanh")),
    ("elu", lambda x: F.elu(x, 0.7)),
    ("selu", F.selu),
    ("celu", lambda x: F.celu(x, 1.3)),
    ("leaky_relu", lambda x: F.leaky_relu(x, 0.05)),
    ("hardswish", F.hardswish),
    ("hardsigmoid", F.hardsigmoid),
    ("mish", F.mish),
    ("softplus", lambda x: F.softplus(x, 1.5, 12.0)),
    ("exp", torch.exp),
    ("expm1", torch.expm1),
    ("log", torch.log),
    ("log1p", torch.log1p),
    ("log2", torch.log2),
    ("log10", torch.log10),
    ("erf", torch.erf),
    ("sqrt", torch.sqrt),
    ("rsqrt", torch.rsqrt),
    ("reciprocal", torch.reciprocal),
    ("pow3", lambda x: x ** 3),
    ("pow1_5", lambda x: x ** 1.5),
)
PURE = {"tanh", "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt",
        "rsqrt", "reciprocal", "hardswish", "hardsigmoid", "leaky_relu", "pow3"}
COUPLED = {"erf", "gelu"}
RANGES = ((-128, 127, torch.int8), (-127, 127, torch.int8), (0, 255, torch.uint8))


def float64_root(x):
    """float64's correctly rounded root rounded once more to float32: for a
    square root, the correctly rounded float32 root (53 >= 2 * 24 + 2 bits)."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.asarray(x, np.float64)).astype(F32)


# Each definition fold_quant_lut_pass.EXACT evaluates, by another computation.
ORACLES = {"sqrt": float64_root}


def bits(t):
    return np.ascontiguousarray(np.asarray(t, dtype=F32)).view(np.int32)


def all_codes(qp):
    """The dequantized value of every code: fl(fl(c - z) * fl32(s))."""
    scale, zp, qmin, qmax, dtype = qp
    codes = torch.arange(qmin, qmax + 1, dtype=torch.int64).to(dtype)
    return DQ(codes, scale, zp, qmin, qmax, dtype)


def qparam_pairs(seed=0, count=40):
    """[(input qparams, output qparams)]: ranges cycled, the zero point at qmin,
    at qmax or inside, scales log-uniform 1e-3..1 (every fifth input scale a
    three-decimal number, not a float32 value), int8 outputs."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(count):
        qmin, qmax, dtype = RANGES[i % 3]
        zp = (qmin, qmax)[i % 4] if i % 4 < 2 else int(rng.integers(qmin, qmax + 1))
        scale = float(10 ** rng.uniform(-3, 0))
        if i % 5 == 0:
            scale = round(scale, 3) or 0.001
            if float(F32(scale)) == scale:
                scale += 0.001
            assert float(F32(scale)) != scale
        out = (float(10 ** rng.uniform(-3, 0)), int(rng.integers(-128, 128)),
               -128, 127, torch.int8)
        pairs.append(((scale, zp, qmin, qmax, dtype), out))
    return pairs


class Sites(torch.nn.Module):
    """One static site per input: q1 -> dq1 -> fn (-> q2 -> dq2), each with its
    own qparams; `repeat` sites copy site 0 on input 0 (a shared table)."""

    def __init__(self, fn, pairs, quantize_output, repeat=0):
        super().__init__()
        self.fn, self.pairs, self.out, self.repeat = fn, pairs, quantize_output, repeat

    def site(self, x, qp, oq):
        a = self.fn(DQ(Q(x, *qp), *qp))
        return DQ(Q(a, *oq), *oq) if self.out else a

    def forward(self, xs):
        outs = [self.site(x, qp, oq) for x, (qp, oq) in zip(xs, self.pairs)]
        outs += [self.site(xs[0], *self.pairs[0]) for _ in range(self.repeat)]
        return tuple(outs)


def lut_nodes(graph):
    return [n for n in graph.nodes if n.op == "call_function"
            and ql._schema(n.target) == ("espressif::quant_lut", "default")]


# --- construction ---------------------------------------------------------------

class _Single(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def test_allowlist_is_what_the_ops_export_to():
    x = torch.rand(1, 64) + 0.5
    exported = set()
    for name, fn in OPS:
        targets = [n.target for n in torch.export.export(_Single(fn), (x,)).graph.nodes
                   if n.op == "call_function"]
        assert len(targets) == 1 and targets[0] in ql.ALLOWLIST, (name, targets)
        exported.add(targets[0])
    assert exported == set(ql.ALLOWLIST)


@pytest.mark.parametrize("quantize_output", [True, False], ids=["V_q", "V_raw"])
@pytest.mark.parametrize("name,fn", OPS, ids=[name for name, _ in OPS])
def test_construction(name, fn, quantize_output):
    pairs = qparam_pairs(seed=sum(map(ord, name)))
    inputs = []
    for qp, _ in pairs:
        xs = all_codes(qp)
        codes = Q(xs, *qp).to(torch.int64)
        assert torch.equal(codes, torch.arange(qp[2], qp[3] + 1)), "all-codes input"
        inputs.append(xs.repeat_interleave(32))
    model = Sites(fn, pairs, quantize_output, repeat=1).eval()
    ep = torch.export.export(model, (tuple(inputs),))
    with ql.one_thread(), torch.no_grad():
        eager = ep.module()(tuple(inputs))
    [target] = {n.target for n in ep.graph.nodes if n.target in ql.ALLOWLIST}
    exact = ql.EXACT.get(target)
    if exact is not None:
        # sqrt's table is its definition, the correctly rounded root, which
        # torch's kernel is on Apple silicon and not in the x86 wheel (MKL):
        # the definition is held to its oracle, the tables to the model with
        # the definition in its place, and torch's own evaluation to them
        # wherever its kernel gives the definition's bits.
        for x in inputs:
            got, want = exact(x).numpy(), ORACLES[name](x.numpy())
            nan = np.isnan(want)
            assert np.array_equal(np.isnan(got), nan), name
            assert np.array_equal(bits(got)[~nan], bits(want)[~nan]), name
        with torch.no_grad():
            by_definition = Sites(exact, pairs, quantize_output, repeat=1)(
                tuple(inputs))
        for x, mine, want in zip(inputs + inputs[:1], eager, by_definition):
            with ql.one_thread(), torch.no_grad():
                kept = bits(fn(x)) == bits(exact(x))
            assert np.array_equal(bits(mine)[kept], bits(want)[kept]), name
        eager = by_definition
    report = ql.fold_quant_luts(ep)
    assert not report.refused
    assert len(report.tables) == len(pairs) + 1
    # One constant per distinct table (two random sites can build the same
    # one: log over an all-negative domain, a saturated output grid), and the
    # repeat of site 0 reads site 0's.
    distinct = {bits(b.table).tobytes() for _, b in report.tables}
    assert len(report.constants) == len(distinct)
    luts = lut_nodes(ep.graph)
    assert luts[0].args[1] is luts[-1].args[1]
    for (_, build), y in zip(report.tables, eager):
        assert build.kind == ("q" if quantize_output else "raw")
        copies = bits(y).reshape(-1, 32)
        assert (copies == copies[:, :1]).all()
        assert np.array_equal(copies[:, 0], bits(build.table)), (name, build.scale)
    with ql.one_thread(), torch.no_grad():
        folded = ep.module()(tuple(inputs))
    for want, got in zip(eager, folded):
        assert np.array_equal(bits(want), bits(got))


# The quiet NaN sqrt_exact writes for a NaN out: ATen's word on every machine.
SQRT_NAN = np.int32(0x7FC00000)


def test_sqrt_table_nan_word():
    """sqrt_exact's contract: the portable
    chain's bits on finite inputs, and a NaN out -- a negative code's root,
    a NaN in -- written as the quiet NaN 0x7FC00000, whatever payload the
    chain's own C library gives it. The construction and chain checks hold a
    NaN by NaN-ness only, with expectations built by the same EXACT helper;
    here the word is pinned as a number: every negative code's entry of a
    raw sqrt table fold_quant_luts builds, and sqrt_exact on -1, -inf, NaN
    and the negative denormal. -0's root stays -0, and the finite entries
    are the float64 oracle's."""
    qp = (0.05, 10, -128, 127, torch.int8)          # codes below 10 negative
    xs = all_codes(qp)
    ep = torch.export.export(Sites(torch.sqrt, [(qp, None)], False).eval(),
                             ((xs.repeat_interleave(32),),))
    report = ql.fold_quant_luts(ep)
    [(_, build)] = report.tables
    assert build.kind == "raw"
    words = bits(build.table)
    negative = (xs < 0).numpy()
    assert negative.sum() == 138 and (~negative).sum() == 118
    assert (words[negative] == SQRT_NAN).all(), sorted(
        {hex(int(w) & 0xFFFFFFFF) for w in words[negative]})
    assert np.array_equal(words[~negative],
                          bits(float64_root(xs.numpy()[~negative])))
    special = torch.tensor([-1.0, float("-inf"), float("nan"), -1.4e-45,
                            -0.0, 0.0])
    got = bits(ql.sqrt_exact(special))
    assert list(got[:4]) == [SQRT_NAN] * 4, [hex(int(w) & 0xFFFFFFFF)
                                             for w in got]
    assert got[4] == np.int32(-2 ** 31) and got[5] == 0, got


# --- purity: the measurement ------------------------------------------------------

SPECIALS = (0.0, -0.0, 1.4e-45, -1.4e-45, 1.1754944e-38, -1.1754944e-38,
            3.4028235e38, -3.4028235e38, math.inf, -math.inf, math.nan,
            1e-30, -1e-30, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 20.0,
            -20.0, 9.3454, -9.3454, 13.2163, -13.2163, 87.33, -87.33, 88.72,
            -88.72, 103.97, -103.97)


def purity_inputs():
    rng = np.random.default_rng(5)
    codes = []
    for i in range(50):
        qmin, qmax, dtype = RANGES[i % 3]
        zp = int(rng.integers(qmin, qmax + 1))
        codes.append(all_codes((float(10 ** rng.uniform(-3, 0)), zp, qmin, qmax,
                                dtype)).numpy())
    random = rng.uniform(-20, 20, 1 << 20).astype(F32)
    return {"codes": np.concatenate(codes), "random": random,
            "specials": np.array(SPECIALS, dtype=F32)}


def run_vec(fn, values):
    with ql.one_thread(), torch.no_grad():
        out = fn(torch.from_numpy(values).repeat_interleave(32))
    return out.numpy()[::32].copy()


def run_scalar(fn, values):
    with ql.one_thread(), torch.no_grad():
        return np.array([fn(torch.from_numpy(values[i:i + 1]))[0].item()
                         for i in range(values.size)], dtype=F32)


def run_tail(fn, values, k):
    """Each value in the last k positions of its own contiguous run of 32 + k
    (rows of a wider buffer, so the runs cannot merge)."""
    rows = -(-values.size // k)
    buf = torch.full((rows, 33 + k), 0.25)
    buf[:, 32:32 + k] = torch.from_numpy(np.resize(values, rows * k).reshape(rows, k))
    with ql.one_thread(), torch.no_grad():
        out = fn(buf[:, :32 + k])
    return out[:, 32:].reshape(-1).numpy()[:values.size].copy()


def run_lane(fn, values):
    """Each value in lane 0 of a [x, 20, 20, 20] vector."""
    lanes = np.full((values.size + values.size % 2, 4), F32(ql.COUPLED_PARTNER))
    lanes[:values.size, 0] = values
    with ql.one_thread(), torch.no_grad():
        out = fn(torch.from_numpy(lanes.reshape(-1)))
    return out.numpy()[0::4][:values.size].copy()


def run_threads(fn, values, threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            return fn(torch.from_numpy(values)).numpy().copy()
    finally:
        torch.set_num_threads(previous)


def run_strided(fn, values):
    buf = torch.zeros(values.size, 2)
    buf[:, 0] = torch.from_numpy(values)
    with ql.one_thread(), torch.no_grad():
        return fn(buf[:, 0]).numpy().copy()


def differ(a, b, x):
    """Elements of a and b (an op's outputs on the inputs x in two contexts)
    whose bits differ, but for a NaN out of a NaN in: its bits are what the
    context's min/max make of them (ATen's AVX-512 maximum returns all ones,
    the scalar std::max its operand; hardswish and hardsigmoid on x86), and no
    table ever sees one: every code dequantises to a finite value. A NaN out
    of anything else is held to its bits."""
    nan_in = np.isnan(x) & np.isnan(a) & np.isnan(b)
    return int(((bits(a) != bits(b)) & ~nan_in).sum())


def test_purity_measurement():
    """The pure set is identical in every context; the rest is measured."""
    sets = purity_inputs()
    every = np.concatenate([sets["codes"], sets["random"], sets["specials"]])
    sample = np.concatenate([sets["codes"], sets["random"][:1 << 14],
                             sets["specials"]])
    threads = np.resize(every, 65537)
    table, failures = {}, []
    for name, fn in OPS:
        ref = run_vec(fn, every)
        ref_sample = run_vec(fn, sample)
        ref_threads = run_vec(fn, threads)
        row = {"C_scalar": (differ(run_scalar(fn, sample), ref_sample, sample),
                            sample.size),
               "C_lane": (differ(run_lane(fn, every), ref, every), every.size),
               "C_strided": (differ(run_strided(fn, every), ref, every), every.size)}
        for k in (1, 3, 7, 15, 31):
            row[f"C_tail{k}"] = (differ(run_tail(fn, every, k), ref, every),
                                 every.size)
        for t in (1, 2, 3, 8):
            row[f"C_threads{t}"] = (differ(run_threads(fn, threads, t), ref_threads,
                                           threads), threads.size)
        table[name] = row
        if name in PURE and any(count for count, _ in row.values()):
            failures.append((name, {c: n for c, (n, _) in row.items() if n}))
    splits = split_codes_per_random_tables()
    path = os.environ.get("QUANT_LUT_PURITY_REPORT")
    if path:
        Path(path).write_text(json.dumps({"contexts": table, "split_codes": splits},
                                         indent=1))
    for name, row in table.items():
        cells = ", ".join(f"{c} {n / total:.2e}" for c, (n, total) in row.items() if n)
        print(f"  {name:12s} {'pure' if name in PURE else '    '} "
              f"{cells or 'identical in every context'}")
    for name, stats in splits.items():
        print(f"  split codes {name}: {stats}")
    assert not failures, f"pure kernels differ across contexts: {failures}"


def split_codes_per_random_tables(count=200):
    """{op: {"V_q": [mean, max], "V_raw": [mean, max]}} of split codes (scalar
    and coupled together) over `count` random tables per non-pure op."""
    rng = np.random.default_rng(11)
    out = {}
    for name, fn in OPS:
        if name in PURE:
            continue
        counts = {"V_q": [], "V_raw": []}
        for i in range(count):
            qmin, qmax, dtype = RANGES[i % 3]
            qp = (float(10 ** rng.uniform(-3, 0.3)), int(rng.integers(qmin, qmax + 1)),
                  qmin, qmax, dtype)
            oq = (float(10 ** rng.uniform(-3, 0)), int(rng.integers(-128, 128)),
                  -128, 127, torch.int8)
            xs = all_codes(qp)
            for kind, out_qparams in (("V_q", oq), ("V_raw", None)):
                _, split = ql.table_and_split(fn, xs, out_qparams=out_qparams,
                                              coupled=name in COUPLED)
                counts[kind].append(len(set().union(*map(set, split.values()))))
        out[name] = {kind: [float(np.mean(c)), int(max(c))]
                     for kind, c in counts.items()}
    return out


# --- the rewrite ---------------------------------------------------------------------

def static_program(fn, qp=(1 / 16, 0, -128, 127, torch.int8),
                   oq=(0.02, -3, -128, 127, torch.int8), x=None, prelude=None,
                   dq_qp=None):
    x = torch.randn(1, 64) if x is None else x

    class M(torch.nn.Module):
        def forward(self, x):
            if prelude is not None:
                x = prelude(x)
            a = fn(DQ(Q(x, *qp), *(dq_qp or qp)))
            return DQ(Q(a, *oq), *oq) if oq else a

    return torch.export.export(M().eval(), (x,))


def snapshot(ep):
    return (ep.graph_module.code, [(s.kind, s.arg.name) for s in
                                   ep.graph_signature.input_specs],
            sorted(ep.constants))


def assert_refused(ep, rule, op="sigmoid"):
    before = snapshot(ep)
    report = ql.fold_quant_luts(ep)
    assert [r[1:] for r in report.refused] == [(op, rule)], report.refused
    assert not report.tables
    assert snapshot(ep) == before


def act_node(ep, name="sigmoid"):
    return next(n for n in ep.graph.nodes if n.op == "call_function"
                and ql.ALLOWLIST.get(n.target, ql.INPLACE.get(n.target)) == name)


def test_refusal_extra_input():
    ep = static_program(lambda d: F.leaky_relu(d, 0.1))
    node = act_node(ep, "leaky_relu")
    node.args = (node.args[0], next(n for n in ep.graph.nodes if n.op == "placeholder"))
    ep.graph_module.recompile()
    assert_refused(ep, "lut-extra-input", "leaky_relu")


def test_refusal_extra_input_in_kwargs():
    ep = static_program(F.gelu)
    node = act_node(ep, "gelu")
    node.kwargs = {"approximate": next(n for n in ep.graph.nodes if n.op == "placeholder")}
    ep.graph_module.recompile()
    assert_refused(ep, "lut-extra-input", "gelu")


def test_refusal_input_not_static_tensor_overload():
    tq = torch.ops.quantized_decomposed.quantize_per_tensor.tensor
    tdq = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor

    class M(torch.nn.Module):
        def forward(self, x):
            s, z = torch.tensor(1 / 16), torch.tensor(0)
            return torch.sigmoid(tdq(tq(x, s, z, -128, 127, torch.int8), s, z,
                                     -128, 127, torch.int8))

    assert_refused(torch.export.export(M(), (torch.randn(1, 64),)),
                   "lut-input-not-static")


def test_refusal_input_not_static_dynamic_chain():
    choose = torch.ops.quantized_decomposed.choose_qparams.tensor
    tq = torch.ops.quantized_decomposed.quantize_per_tensor.tensor
    tdq = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor

    class M(torch.nn.Module):
        def forward(self, x):
            s, z = choose(x, -128, 127, 2 ** -23, torch.int8)
            return torch.sigmoid(tdq(tq(x, s, z, -128, 127, torch.int8), s, z,
                                     -128, 127, torch.int8))

    assert_refused(torch.export.export(M(), (torch.randn(1, 64),)),
                   "lut-input-not-static")


def test_refusal_dq_mismatch():
    assert_refused(static_program(torch.sigmoid,
                                  dq_qp=(1 / 8, 0, -128, 127, torch.int8)),
                   "lut-dq-mismatch")


def test_refusal_wide_codes():
    assert_refused(static_program(torch.sigmoid,
                                  qp=(1 / 256, 0, -32768, 32767, torch.int16)),
                   "lut-wide-codes")


def test_refusal_bad_zero_point():
    assert_refused(static_program(torch.sigmoid, qp=(1 / 16, 200, -128, 127,
                                                     torch.int8)),
                   "lut-bad-qparams")


def test_refusal_reciprocal_not_finite():
    # A scale whose float32 reciprocal overflows, as the shim refuses.
    assert_refused(static_program(torch.sigmoid, qp=(1e-300, 0, -128, 127,
                                                     torch.int8)),
                   "lut-bad-qparams")


def test_refusal_dtype_half_input():
    assert_refused(static_program(torch.sigmoid, prelude=lambda x: x.half()),
                   "lut-dtype")


def test_refusal_layout_channels_last():
    # X's strides, not dq1's (always a fresh contiguous tensor).
    assert_refused(static_program(torch.sigmoid, x=torch.randn(1, 4, 3, 3),
                                  prelude=lambda x: x.contiguous(
                                      memory_format=torch.channels_last)),
                   "lut-layout")


def test_refusal_inplace():
    ep = static_program(torch.sigmoid)
    act_node(ep).target = torch.ops.aten.sigmoid_.default
    ep.graph_module.recompile()
    assert_refused(ep, "lut-inplace")


def test_activation_without_users_is_not_a_candidate():
    ep = static_program(torch.sigmoid)
    dq1 = act_node(ep).args[0]
    with ep.graph.inserting_after(dq1):
        ep.graph.call_function(torch.ops.aten.tanh.default, (dq1,))
    ep.graph_module.recompile()
    report = ql.fold_quant_luts(ep)
    assert len(report.tables) == 1 and not report.refused
    assert any(n.target is torch.ops.aten.tanh.default for n in ep.graph.nodes)


def test_dq_and_q_kept_while_read_elsewhere():
    # dq1 (here a second output) and q1 stay; the table reads X.
    class M(torch.nn.Module):
        def forward(self, x):
            d = DQ(Q(x, 1 / 16, 0, -128, 127, torch.int8), 1 / 16, 0, -128, 127,
                   torch.int8)
            return torch.sigmoid(d), d

    ep = torch.export.export(M(), (torch.randn(1, 64),))
    report = ql.fold_quant_luts(ep)
    assert len(report.tables) == 1
    lut = lut_nodes(ep.graph)[0]
    assert lut.args[0].op == "placeholder"
    names = [str(n.target) for n in ep.graph.nodes]
    assert sum("dequantize_per_tensor" in n for n in names) == 1
    assert sum("quantize_per_tensor" in n and "dequantize" not in n for n in names) == 1


def test_shared_q1_goes_once_both_readers_are_tables():
    class M(torch.nn.Module):
        def forward(self, x):
            q = Q(x, 1 / 16, 0, -128, 127, torch.int8)
            a = DQ(q, 1 / 16, 0, -128, 127, torch.int8)
            b = DQ(q, 1 / 16, 0, -128, 127, torch.int8)
            return torch.sigmoid(a), torch.tanh(b)

    ep = torch.export.export(M(), (torch.randn(1, 64),))
    report = ql.fold_quant_luts(ep)
    assert len(report.tables) == 2
    assert not any("quantize" in str(n.target) for n in ep.graph.nodes)


def test_v_q_needs_identical_args_dtype_included():
    class M(torch.nn.Module):
        def forward(self, x):
            a = torch.sigmoid(DQ(Q(x, 1 / 16, 0, -128, 127, torch.int8), 1 / 16, 0,
                                 -128, 127, torch.int8))
            return (DQ(Q(a, 0.01, 0, -128, 127, torch.int8), 0.01, 0, -128, 127,
                       torch.int8),
                    DQ(Q(a, 0.01, 0, -128, 127, torch.int16), 0.01, 0, -128, 127,
                       torch.int16))

    ep = torch.export.export(M(), (torch.randn(1, 64),))
    report = ql.fold_quant_luts(ep)
    assert [b.kind for _, b in report.tables] == ["raw"]


def test_tables_counted_after_to_edge_and_after_partitioning():
    """The tables are lifted as CONSTANT_TENSORs before to_edge: count them
    through to_edge, partitioning and the pair elision."""
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import (
        _to_edge_validated_once,
        apply_passes,
        lower_esp32,
    )

    def program():
        class M(torch.nn.Module):
            def forward(self, x):
                outs = []
                for fn in (torch.sigmoid, torch.erf, F.silu):
                    a = fn(DQ(Q(x, 1 / 16, 0, -128, 127, torch.int8), 1 / 16, 0,
                              -128, 127, torch.int8))
                    outs.append(DQ(Q(a, 0.01, 1, -128, 127, torch.int8), 0.01, 1,
                                   -128, 127, torch.int8))
                return tuple(outs)

        return torch.export.export(M(), (torch.randn(1, 64),))

    ep = program()
    ql.fold_quant_luts(ep)
    edge = _to_edge_validated_once(ep).exported_program()
    assert len(lut_nodes(edge.graph)) == 3 and len(edge.constants) == 3
    lowered = lower_esp32(program(), [Esp32Partitioner()], quant_luts=True)
    assert len(lowered.quant_lut_report.tables) == 3
    assert len(lut_nodes(lowered.exported_program().graph)) == 3
    rep = apply_passes(lowered, quiet=True)
    assert rep["lut_pairs"] == 3
    left = [str(n.target) for n in lowered.exported_program().graph.nodes
            if n.op == "call_function"]
    assert not any("quantize" in t for t in left), left


def test_elision_only_where_the_pair_is_an_identity():
    """A raw table holding NaN, or a dq that moves its entries, keeps its pair."""
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    # A float32 scale: the pair stays portable, and a portable quantize's
    # scale must be one (static_qdq.refuse_portable_scales).
    s2 = float(np.float32(0.05))

    class M(torch.nn.Module):
        def forward(self, x):
            a = torch.log(DQ(Q(x, 1 / 16, 0, -128, 127, torch.int8), 1 / 16, 0,
                             -128, 127, torch.int8))
            first = DQ(Q(a, s2, 0, -128, 127, torch.int8), s2, 0, -128, 127,
                       torch.int8)
            return first, a

    lowered = lower_esp32(torch.export.export(M(), (torch.randn(1, 64),)),
                          [Esp32Partitioner()], quant_luts=True)
    assert [b.kind for _, b in lowered.quant_lut_report.tables] == ["raw"]
    assert apply_passes(lowered, quiet=True)["lut_pairs"] == 0


# The stock recipes' census models (quant_census): tables per model, then what
# the rewrite refused. Every activation sits between two static pairs; the
# Cortex-M flow feeds se_hardswish channels-last (refused by X's layout), and
# fuses its SE gate's hardsigmoid into the conv before it, with no pair in
# between (not a candidate).
CENSUS = {
    "xnnpack-static": {"se_hardswish": (2, [])},
    "xnnpack-qat": {"se_hardswish": (2, [])},
    "arm-a8w8": {"se_hardswish": (2, [])},
    "cortex-m": {"se_hardswish": (0, [("hardswish", "lut-layout")])},
}
CENSUS_MODELS = ("sigmoid", "tanh", "gelu_erf", "gelu_tanh", "silu", "se_hardswish")


@pytest.mark.parametrize("recipe", sorted(CENSUS))
def test_census_models(recipe):
    """Tables per model, counted again after to_edge, and the folded
    program bit-identical to the recipe's own quantised model on its inputs."""
    from executorch.backends.esp32s3._passes import _to_edge_validated_once
    from executorch.backends.esp32s3.test.quant_census import (
        models as census_models,
        recipes,
    )

    for name in CENSUS_MODELS:
        net, inputs, calibration = census_models.model(name)
        try:
            q = recipes.quantize(recipe, net, inputs, calibration)
        except Exception as error:  # noqa: BLE001
            if recipes.is_stub_error(error):
                pytest.skip(f"{recipe} needs real TOSA work here: {error}")
            raise
        ep = torch.export.export(q.program, q.inputs, strict=False)
        with torch.no_grad():
            eager = ep.module()(*q.inputs)
        report = ql.fold_quant_luts(ep)
        want = CENSUS[recipe].get(name, (1, []))
        assert (len(report.tables), [r[1:] for r in report.refused]) == want, name
        assert all(b.kind == "q" for _, b in report.tables)
        with torch.no_grad():
            folded = ep.module()(*q.inputs)
        assert np.array_equal(bits(eager), bits(folded)), name
        edge = _to_edge_validated_once(ep).exported_program()
        assert len(lut_nodes(edge.graph)) == want[0], name


@pytest.mark.parametrize("name", ["mlp", "block", "convnet", "acts"])
def test_symmetric_export_identical_with_the_pass_on(name):
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.test.models import SPECS, ramp_input

    buffers = []
    for on in (False, True):
        torch.manual_seed(0)
        factory, shape = SPECS[name]
        ep = torch.export.export(factory().eval(), (ramp_input(shape),))
        edge = lower_esp32(ep, [Esp32Partitioner()], quant_luts=on)
        assert edge.quant_lut_report is None or not (
            edge.quant_lut_report.tables or edge.quant_lut_report.refused)
        apply_passes(edge, quiet=True)
        buffers.append(edge.to_executorch(ExecutorchBackendConfig()).buffer)
    assert buffers[0] == buffers[1]


# --- the speed tables: each the portable chain it replaces, bit for bit ------------------

# Exact as portable ops, so a table is a speed choice only (hardswish and
# hardsigmoid: to_edge's add, clamp, mul and div, four or three portable ops,
# become one).
SPEED = ("hardswish", "hardsigmoid", "leaky_relu", "sqrt", "rsqrt", "reciprocal")


def test_speed_tables_are_allowlisted_and_pure():
    """Each speed op is an allowlisted table and in the pure set the purity
    measurement asserts (test_purity_measurement): a torch upgrade that made
    one impure would fail there before its table could drift."""
    names = set(ql.ALLOWLIST.values())
    assert set(SPEED) <= names and set(SPEED) <= ql.PURE <= names


def _lower_program(ep, luts):
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    edge = lower_esp32(ep, [Esp32Partitioner()], quant_luts=luts)
    apply_passes(edge, quiet=True)
    graph = edge.exported_program().graph_module.graph
    assert bool(lut_nodes(graph)) == luts
    return edge.to_executorch(ExecutorchBackendConfig()).buffer


def _run_host(runner, tmp_path, tag, program, inputs):
    pte = tmp_path / f"{tag}.pte"
    pte.write_bytes(program)
    command = [str(runner), "--pte", str(pte)]
    outs = []
    for i, x in enumerate(inputs):
        path = tmp_path / f"{tag}.in{i}.bin"
        x.numpy().tofile(path)
        out = tmp_path / f"{tag}.out{i}.bin"
        command += ["--input", str(path)]
        outs.append(out)
    for out in outs:
        command += ["--output", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return [np.fromfile(out, dtype=F32) for out in outs]


@pytest.mark.parametrize("quantize_output", [True, False], ids=["V_q", "V_raw"])
@pytest.mark.parametrize("name", SPEED)
def test_speed_table_is_the_portable_chain(runner, tmp_path, name,
                                           quantize_output):
    """The speed tables are allowed only because each gives, on every code,
    the bits of the portable chain it replaces: both programs on the host
    runner -- ExecuTorch's portable kernels and the Espressif overrides, the
    device's own sources -- over every code of 40 seeded grids, each once
    (QEMU's st_spd_chain and st_spd_lut rows are the device half).

    The one exception is a code whose value is not finite (sqrt and rsqrt
    below zero, rsqrt and reciprocal at zero). After an output pair the chain
    quantises that value with ExecuTorch's kernel, which since 1.6 gives NaN
    the zero point and an infinity its rail (runtime/quant_rules.h), while
    the table holds eager torch's quantize of it, whose code for NaN is
    platform-defined. Without a pair (V_raw) the two agree on infinities bit
    for bit and on NaN as a NaN."""
    fn = dict(OPS)[name]
    # A portable quantize takes float32 scales only (static_qdq's
    # refuse_portable_scales): the construction test covers the others.
    pairs = [((float(F32(qp[0])),) + qp[1:], (float(F32(oq[0])),) + oq[1:])
             for qp, oq in qparam_pairs(seed=sum(map(ord, name)))]
    inputs = [all_codes(qp) for qp, _ in pairs]
    model = Sites(fn, pairs, quantize_output).eval()
    outs = {}
    for luts in (False, True):
        ep = torch.export.export(model, (tuple(inputs),))
        outs[luts] = _run_host(runner, tmp_path, f"{name}{int(luts)}",
                               _lower_program(ep, luts), inputs)
    excused = 0
    for (qp, _), x, chain, table in zip(pairs, inputs, outs[False], outs[True]):
        with torch.no_grad():
            finite = torch.isfinite(fn(x)).numpy()
        nan = np.isnan(chain)
        if quantize_output:
            excused += int((~finite).sum())
            assert np.array_equal(bits(chain)[finite], bits(table)[finite]), qp
        else:
            assert np.array_equal(nan, np.isnan(table)), qp
            assert np.array_equal(bits(chain)[~nan], bits(table)[~nan]), qp
    if name in ("hardswish", "hardsigmoid", "leaky_relu"):
        assert excused == 0


# --- the mirror -------------------------------------------------------------------------

def test_mirror_matches_reference_and_quantize_per_tensor():
    table = np.arange(256, dtype=np.int32)
    table = np.where(table % 3 == 0, 0x7FA00000 | table, 0x00000100 | table)
    table = table.view(F32)
    for scale in ql.GOLDEN_SCALES:
        inv = float(qm.reciprocal(scale))
        xs = np.concatenate([np.array(five, dtype=F32) for _, five in
                             ql.golden_inputs(inv)]
                            + [np.array(ql.GOLDEN_SPECIALS, dtype=np.uint32).view(F32)])
        nan = np.isnan(xs)
        for qmin, qmax, dtype in RANGES:
            for zp in sorted({qmin, -1, 0, 1, qmax}):
                if not qmin <= zp <= qmax:
                    continue
                t = torch.from_numpy(table[:qmax - qmin + 1].copy())
                ref = ql.quant_lut(torch.from_numpy(xs), t, inv, zp, qmin, qmax)
                mine = qm.quant_lut(xs, t.numpy(), inv, zp, qmin, qmax)
                assert np.array_equal(bits(ref), bits(mine))
                codes = qm.quant_lut_codes(xs, inv, zp, qmin, qmax)
                want = Q(torch.from_numpy(xs), scale, zp, qmin, qmax, dtype)
                assert np.array_equal(codes[~nan], want.numpy()[~nan].astype(np.int64))
                # NaN: the zero point, ExecuTorch 1.6's rule (torch's own
                # code for it is platform-defined)
                assert (codes[nan] == zp).all()


# --- the C++ conformance check --------------------------------------------------------

def test_golden_header_is_torchs():
    path = ROOT.parent / "runtime" / "quant_lut_golden.h"
    assert path.read_text() == ql.golden_header(), (
        "quant_lut_golden.h is stale: regenerate it with quant_lut.golden_header()")


@pytest.fixture(scope="session")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def test_host_conformance(runner):
    out = subprocess.run([str(runner), "--selftest"], capture_output=True,
                         text=True, timeout=600).stdout
    line = re.search(r"^quant_lut check: (\d+) case\(s\), (\d+) mismatch\(es\).*$",
                     out, re.M)
    assert line, out[-2000:]
    assert int(line.group(1)) > 300000 and line.group(2) == "0", line.group(0)
    assert "malformed 25/25 refused, out untouched, control accepted" in line.group(0)


def test_host_rows_are_the_mirror(runner, tmp_path):
    """luts.pte on the host runner: every method's output bits are the mirror's
    (run_parity holds the device to the same bits)."""
    from executorch.backends.esp32s3.test.fixtures import quant_lut

    et, refs, _ = quant_lut.build_program()
    pte = tmp_path / "luts.pte"
    pte.write_bytes(et.buffer)
    lines, outs = [], {}
    for method, (layout, n, specs) in quant_lut.METHODS.items():
        ramp = (np.arange(n, dtype=F32) * F32(0.0078125) - F32(0.5))
        inp = tmp_path / f"{method}.in.bin"
        ramp.tofile(inp)
        outs[method] = [tmp_path / f"{method}.out{k}.bin" for k in range(len(specs))]
        lines.append(f"{method} {inp} -> {' '.join(map(str, outs[method]))}")
    script = tmp_path / "steps.txt"
    script.write_text("\n".join(lines) + "\n")
    done = subprocess.run([str(runner), "--pte", str(pte), "--script", str(script)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    for method, paths in outs.items():
        got = np.concatenate([np.fromfile(p, dtype=F32) for p in paths])
        assert np.array_equal(bits(got), bits(refs[method])), method
    calls = re.search(r"quant_lut dispatch: calls (\d+), elements (\d+), refusals (\d+)",
                      done.stdout)
    assert calls and int(calls.group(1)) > 0 and calls.group(3) == "0"


def test_object_code_has_no_fused_multiply_add():
    from executorch.backends.esp32s3.test.fixtures import quant_lut

    from executorch.backends.esp32s3.test import firmware

    build = firmware.PROJECT / "build"
    if not list(build.glob("esp-idf/espressif_backend/**/quant_lut.cpp.obj")):
        pytest.skip("no parity build (run_parity.py builds one and checks this too)")
    fused, read = quant_lut.fused_multiply_adds(build)
    assert read > 100 and not fused, (read, fused[:5])   # the kernel and shim: ~590
