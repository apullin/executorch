# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::qnorm: the fused quantised norm's host tests.

  rewrite     the stock recipes' decomposed norms (Arm a8w8 RMSNorm and
              LayerNorm, Cortex-M RMSNorm) fold into one op each, and the
              folded op against the recipe's eager model holds the bar: at
              most one code at each reduction's quantiser, every row whose
              reduction codes agree bit for bit; Cortex-M's LayerNorm stays
              whole (its recipe keeps layer_norm, no pair inside).
  int16       the same patterns on int16 codes (Arm a16w8's grids, all or
              some pairs): kinds 2 and 3 hold the same bar; their mirror ==
              the reference, and a synthetic program using every step and a
              table with exceptions too.
  refusals    hand-written patterns, one per rule: a missing pair, a
              different grid (a requantised input, a dequantise whose
              arguments differ), an extra reader of an intermediate, int32
              codes. Each leaves the graph exactly as it was.
  default     on: Esp32Partitioner() lowers to the same bytes as
              fused_norms=True, the norm fused; fused_norms=False keeps it
              decomposed.
  mirror      mirrors.qnorm == the torch reference on random inputs,
              tables and quantisers, every slot present and absent.
  kernel      the C++ kernel (host runner) == the mirror on random inputs,
              the folded hand-written patterns' programs (int8 and int16),
              and the synthetic programs called directly.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner. It imports
torch (about 2 GB).
"""

import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn

from executorch.backends.esp32s3._passes import fuse_qnorm_pass as qn
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
I8 = (-128, 127, torch.int8)


def bits(a):
    return np.asarray(a, F32).view(np.int32)


def qd():
    return torch.ops.quantized_decomposed


def grid(t):
    """Asymmetric int8 (scale, zero point) over t's range, a float32 scale."""
    lo, hi = min(float(t.min()), 0.0), max(float(t.max()), 0.0)
    s = float(F32(max(hi - lo, 1e-6) / 255.0))
    return s, int(np.clip(np.rint(-128 - lo / s), -128, 127))


class Hand(nn.Module):
    """A norm as convert_pt2e leaves Arm's, every pair written out with
    grids measured on the float pattern, so each test can drop, move or
    widen one."""

    def __init__(self, n, kind="rms", drop=(), seed=5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.kind, self.drop = n, kind, set(drop)
        self.weight = nn.Parameter(1.0 + 0.25 * torch.randn(n, generator=g))
        self.bias = nn.Parameter(0.1 * torch.randn(n, generator=g))
        self.eps = 1e-5
        self.grids = {}
        self.dtype = {}
        self.extra = None           # a name whose value is also returned
        self.mismatch = None        # a name whose dequantise gets another scale
        self.regrid = None          # a name read again on another grid
        self.calibrate = True
        with torch.no_grad():
            self.forward(torch.randn(16, n, generator=g) * 0.8 + 0.3)
        self.calibrate = False

    def pair(self, t, name):
        if self.calibrate:
            self.grids[name] = grid(t)
        if name in self.drop:
            return t
        s, z = self.grids[name]
        lo, hi, dtype = self.dtype.get(name, I8)
        if dtype == torch.int16:
            s, z = s / 256, 0
        q = qd().quantize_per_tensor(t, s, z, lo, hi, dtype)
        ds = s * 1.5 if name == self.mismatch else s
        return qd().dequantize_per_tensor(q, ds, z, lo, hi, dtype)

    def forward(self, x):
        p, n = self.pair, self.n
        a = p(x, "x")
        if self.kind == "rms":
            sq = p(torch.pow(a, 2), "sq")
            s = p(torch.sum(sq, dim=-1, keepdim=True), "red")
            m = p(s * (1.0 / n), "mean")
            e = p(m + self.eps, "eps")
            r = p(torch.rsqrt(e), "r")
            if self.regrid and not self.calibrate:
                s2, z2 = self.grids["x"]
                a = qd().dequantize_per_tensor(qd().quantize_per_tensor(
                    x, s2 * 2, z2, *I8), s2 * 2, z2, *I8)
            y = p(a * r, "m")
            out = self.weight * y
            named = {"sq": sq, "red": s, "mean": m, "eps": e, "r": r}
        else:
            s1 = p(a.sum(-1, keepdim=True), "red1")
            mu1 = p(s1 * (1.0 / n), "m1")
            c1 = p(a - mu1, "c1")
            s2 = p(a.sum(-1, keepdim=True), "red2")
            mu2 = p(s2 * (1.0 / n), "m2")
            c2 = p(a - mu2, "c2")
            sq = p(c2 * c2, "sq")
            s3 = p(sq.sum(-1, keepdim=True), "red3")
            v = p(s3 * (1.0 / n), "v")
            e = p(v + self.eps, "e")
            r = p(torch.rsqrt(e), "r")
            nrm = p(c1 * r, "n")
            w = p(nrm * self.weight, "w")
            out = w + self.bias
            named = {"m1": mu1, "c2": c2, "sq": sq, "red3": s3, "r": r}
        if self.extra and not self.calibrate:
            return out, named[self.extra]
        return out


def export(model, n=24, rows=6, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, generator=g) * 0.8 + 0.3
    return torch.export.export(model, (x,), strict=False), (x,)


def snapshot(ep):
    return ep.graph_module.code


def assert_refused(model, rule):
    ep, _ = export(model)
    before = snapshot(ep)
    report = qn.fold_qnorms(ep)
    assert not report.fused, report.summary()
    assert [r for _, r, _ in report.refused] == [rule], report.summary()
    assert snapshot(ep) == before


# --- rewrite on hand-written patterns ------------------------------------------------

@pytest.mark.parametrize("kind", ["rms", "ln"])
def test_hand_pattern_folds_and_holds_the_bar(kind):
    model = Hand(24, kind).eval()
    ep, (x,) = export(model)
    report = qn.fold_qnorms(ep)
    assert len(report.fused) == 1 and not report.refused, report.summary()
    nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
    assert [str(n.target) for n in nodes] == ["espressif.qnorm.default"], nodes
    from executorch.backends.esp32s3.test.fixtures import qnorm

    batches = [(torch.randn(6, 24, generator=torch.Generator().manual_seed(s))
                * 0.8 + 0.3,) for s in range(24)]
    gap = qnorm.norm_eager_gap(model, batches)
    assert gap.norms == 1 and gap.ok(), gap.summary()


# --- refusals ----------------------------------------------------------------------------

def test_refusal_missing_pair():
    # nothing quantised between the sum and rsqrt: rsqrt reads no code
    assert_refused(Hand(24, drop=("red", "mean", "eps")).eval(),
                   "qnorm-missing-pair")


def test_refusal_missing_pair_layer_mean():
    assert_refused(Hand(24, "ln", drop=("red1",)).eval(), "qnorm-missing-pair")


def test_refusal_dq_mismatch():
    model = Hand(24).eval()
    model.mismatch = "mean"
    assert_refused(model, "qnorm-dq-mismatch")


def test_refusal_different_grid():
    model = Hand(24).eval()
    model.regrid = True
    assert_refused(model, "qnorm-grid")


@pytest.mark.parametrize("kind,name", [("rms", "mean"), ("rms", "r"),
                                       ("ln", "c2"), ("ln", "red3")])
def test_refusal_extra_consumer(kind, name):
    model = Hand(24, kind).eval()
    model.extra = name
    assert_refused(model, "qnorm-extra-consumer")


def test_refusal_int32_codes():
    model = Hand(24).eval()
    model.dtype["red"] = (-2 ** 31, 2 ** 31 - 1, torch.int32)
    assert_refused(model, "qnorm-wide-codes")


# --- int16 codes (kinds 2 and 3) -----------------------------------------------------------

def wide(model, names=None):
    """A Hand model with its pairs (or `names`) on symmetric int16 codes,
    Arm a16w8's grid shape."""
    for name in names or list(model.grids):
        model.dtype[name] = (-32767, 32767, torch.int16)
    return model


@pytest.mark.parametrize("kind,names", [("rms", None), ("ln", None),
                                        ("rms", ("red", "mean", "eps", "r")),
                                        ("ln", ("x", "c2", "sq", "red3"))])
def test_int16_pattern_folds_and_holds_the_bar(kind, names):
    """All pairs int16, or only some (the rest int8): kinds 2 and 3, each
    program checked against its chain at the build, and the bar against
    eager."""
    model = wide(Hand(24, kind), names).eval()
    ep, _ = export(model)
    report = qn.fold_qnorms(ep)
    assert len(report.fused) == 1 and not report.refused, report.summary()
    (_, b), = report.fused
    assert b.kind == (qn.RMS16 if kind == "rms" else qn.LAYER16)
    assert b.tables.dtype == torch.int32 and len(b.luts) == 1, b.summary()
    from executorch.backends.esp32s3.test.fixtures import qnorm

    batches = [(torch.randn(6, 24, generator=torch.Generator().manual_seed(s))
                * 0.8 + 0.3,) for s in range(24)]
    gap = qnorm.norm_eager_gap(model, batches)
    assert gap.norms == 1 and gap.ok(), gap.summary()


def _int16_build(kind, n=24, names=None):
    model = wide(Hand(n, kind), names).eval()
    ep, _ = export(model, n=n)
    (_, b), = qn.fold_qnorms(ep).fused
    return b


def _check_mirror(b, x):
    t = lambda a: None if a is None else torch.from_numpy(  # noqa: E731
        np.ascontiguousarray(a))
    ref = qn.qnorm(t(x), b.tables, b.quant, b.weight, b.bias, b.kind, b.flags,
                   b.eps)
    mine = qm.qnorm(x, b.tables.numpy(), b.quant.numpy(),
                    None if b.weight is None else b.weight.numpy(),
                    None if b.bias is None else b.bias.numpy(), b.kind,
                    b.flags, b.eps)
    assert np.array_equal(bits(ref.numpy()), bits(mine))


@pytest.mark.parametrize("kind", ["rms", "ln"])
def test_int16_mirror_matches_reference(kind):
    b = _int16_build(kind)
    rng = np.random.default_rng(11)
    for trial in range(8):
        _check_mirror(b, (rng.standard_normal((6, 24)) * (0.2 + trial))
                      .astype(F32))


def synthetic_build(kind):
    """A kind 2 or 3 build whose last program uses every step (constants on
    either side of a subtraction and a division) and an NSAU cubic rsqrt
    table with exceptions, the others plain pairs; random quantisers."""
    from executorch.backends.esp32s3.quantizer import lut_fit

    site = lut_fit.site_for("rsqrt", 2.0 ** -10, 0, 2.0 ** -9, 0, qmin=0,
                            qmax=32767)
    table = lut_fit.fit_scheme(site, "nsau_cubic", 2)
    got, want = table.evaluate(site.codes), site.E()
    # the table's flips, plus two made-up exceptions
    exceptions = {int(site.codes[i]): int(want[i])
                  for i in np.nonzero(got != want)[0][:12]}
    exceptions.update({5: 7, 20000: 3})
    exceptions = sorted(exceptions.items())
    q16 = lambda s, z: [qn._bits(qm.reciprocal(s)), 0, z, -32767, 32767]  # noqa: E731
    deq = lambda s, z: [0, qn._bits(s), z, 0, 0]  # noqa: E731
    k = lambda v: [qn._bits(v), 0, 0, 0, 0]  # noqa: E731
    pair = [(qn.CODE, q16(0.01, 3)), (qn.DEQ, deq(0.01, 3))]
    last = [(qn.CODE, q16(0.02, 0)), (qn.DEQ, deq(0.02, 0)),
            (qn.MUL, k(0.125)), (qn.ADD, k(-0.75)), (qn.DIV, k(3.0)),
            (qn.RSUB, k(1.5)), (qn.RDIV, k(-2.0)), (qn.ADD, k(1.25)),
            (qn.CODE, q16(2.0 ** -10, 0)), (qn.TABLE, [0, 0, 0, 0, 0]),
            (qn.DEQ, deq(2.0 ** -9, 0)), (qn.MUL, k(0.5))]
    programs = [last] if kind == qn.RMS16 else [pair, pair, last]
    tables = qn.encode16(programs, [(table, exceptions)])
    names = qn.RMS16_SLOTS if kind == qn.RMS16 else qn.LAYER16_SLOTS
    rng = np.random.default_rng(kind)
    rows = []
    for name in names:
        s = float(F32(10.0 ** rng.uniform(-4, -2)))
        rows.append((float(qm.reciprocal(s)), s, int(rng.integers(-40, 40)),
                     -32767, 32767))
    quant = torch.tensor(rows, dtype=torch.float32)
    n = 24
    weight = torch.randn(n, generator=torch.Generator().manual_seed(kind))
    bias = (torch.randn(n, generator=torch.Generator().manual_seed(9))
            if kind == qn.LAYER16 else None)
    return qn.QnormBuild(kind, 0, 0.0, tables, quant, weight, bias, n, {},
                         [(table, exceptions)])


@pytest.mark.parametrize("kind", [qn.RMS16, qn.LAYER16])
def test_int16_program_steps_mirror(kind):
    b = synthetic_build(kind)
    rng = np.random.default_rng(5)
    for trial in range(6):
        _check_mirror(b, (rng.standard_normal((5, b.n)) * (0.5 + 2 * trial))
                      .astype(F32))


# --- the mirror against the reference ------------------------------------------------------

def random_build(kind, n, rng, absent=()):
    slots = qn.RMS_SLOTS if kind == qn.RMS else qn.LAYER_SLOTS
    tables = rng.standard_normal((2 if kind == qn.RMS else 4, 256)).astype(F32)
    rows = []
    for name in slots:
        if name in absent:
            rows.append(qn.ABSENT)
            continue
        s = float(F32(10.0 ** rng.uniform(-3, 0)))
        qmin, qmax = (-128, 127) if rng.random() < 0.8 else (0, 255)
        z = int(rng.integers(qmin, qmax + 1))
        rows.append((float(qm.reciprocal(s)), s, z, qmin, qmax))
    quant = np.array(rows, F32)
    weight = rng.standard_normal(n).astype(F32)
    bias = rng.standard_normal(n).astype(F32) if kind == qn.LAYER else None
    return tables, quant, weight, bias


@pytest.mark.parametrize("case", range(24))
def test_mirror_matches_reference(case):
    rng = np.random.default_rng(case)
    kind = qn.RMS if case % 2 == 0 else qn.LAYER
    n = int(rng.integers(1, 70))
    absent = ()
    flags, eps = 0, 0.0
    if kind == qn.RMS:
        absent = [("x",), ("m",), ("sq",), ("x", "sq")][case // 2 % 4]
        flags = int(rng.integers(0, 4))
        eps = float(F32(1e-6))
    else:
        absent = [(), ("c1",), ("n", "w"), ("w",)][case // 2 % 4]
    tables, quant, weight, bias = random_build(kind, n, rng, absent)
    if case % 3 == 0:
        weight, bias = None, None
    x = (rng.standard_normal((5, n)) * rng.uniform(0.1, 4)).astype(F32)
    t = lambda a: None if a is None else torch.from_numpy(a.copy())  # noqa: E731
    ref = qn.qnorm(t(x), t(tables), t(quant), t(weight), t(bias), kind, flags,
                   eps)
    mine = qm.qnorm(x, tables, quant, weight, bias, kind, flags, eps)
    assert np.array_equal(bits(ref.numpy()), bits(mine))


# --- the default -----------------------------------------------------------------------------

def _lower(ep, **options):
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    edge = lower_esp32(ep, [Esp32Partitioner(**options)])
    apply_passes(edge, quiet=True)
    graph = edge.exported_program().graph_module.graph
    norms = sum(n.op == "call_function" and "qnorm" in str(n.target)
                for n in graph.nodes)
    return edge.to_executorch(ExecutorchBackendConfig()).buffer, norms


def test_default_on_is_byte_identical():
    import copy

    model = Hand(24, "ln").eval()
    ep, _ = export(model)
    default, n0 = _lower(copy.deepcopy(ep))
    off, n1 = _lower(copy.deepcopy(ep), fused_norms=False)
    on, n2 = _lower(copy.deepcopy(ep), fused_norms=True)
    assert default == on and n0 == n2 == 1 and n1 == 0 and off != on


def test_float_graph_is_neither_touched_nor_reported():
    """fused_norms is on by default, so every float model goes through the
    fold: its rsqrt is no quantised norm kept decomposed, nothing to report."""
    class Rms(nn.Module):
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    ep = torch.export.export(Rms(), (torch.randn(2, 8),), strict=False)
    before = str(ep.graph)
    report = qn.fold_qnorms(ep)
    assert not report.fused and not report.refused and str(ep.graph) == before


# --- the kernel against the mirror -----------------------------------------------------------

@pytest.fixture(scope="session")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def _run_against_mirror(runner, tmp_path, program, b, rows, n, seed):
    pte = tmp_path / "qn.pte"
    pte.write_bytes(program)
    rng = np.random.default_rng(seed)
    for trial in range(8):
        xs = (rng.standard_normal((rows, n)) * (0.3 + trial)).astype(F32)
        inp, out = tmp_path / f"in{trial}.bin", tmp_path / f"out{trial}.bin"
        xs.tofile(inp)
        done = subprocess.run([str(runner), "--pte", str(pte), "--input",
                               str(inp), "--output", str(out)],
                              capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        got = np.fromfile(out, dtype=F32)
        want = qm.qnorm(xs, b.tables.numpy(), b.quant.numpy(),
                        None if b.weight is None else b.weight.numpy(),
                        None if b.bias is None else b.bias.numpy(),
                        b.kind, b.flags, b.eps).reshape(-1)
        assert np.array_equal(bits(got), bits(want)), trial
        assert "qnorm dispatch: calls 1," in done.stdout
        if b.kind in (qn.RMS16, qn.LAYER16):
            # its table checked once, read where the program holds it
            assert "lut16 tables: validated 1, reused 0" in done.stdout, \
                done.stdout[-2000:]


@pytest.mark.parametrize("kind,n,rows,codes", [
    ("rms", 24, 6, 8), ("rms", 61, 3, 8), ("ln", 24, 6, 8), ("ln", 7, 9, 8),
    ("rms", 24, 6, 16), ("rms", 61, 3, 16), ("ln", 24, 6, 16),
    ("ln", 21, 5, 16)])
def test_kernel_is_the_mirror(runner, tmp_path, kind, n, rows, codes):
    """The folded hand-written pattern's program on the host runner (the
    device's kernel source) against mirrors.qnorm with the fold's own
    constants, on random inputs past the calibrated range; int8 and int16
    codes."""
    model = Hand(n, kind).eval()
    if codes == 16:
        wide(model)
    ep, (x,) = export(model, n=n, rows=rows)
    probe = torch.export.export(model, (x,), strict=False)
    report = qn.fold_qnorms(probe)
    (_, b), = report.fused
    program, norms = _lower(ep, fused_norms=True)
    assert norms == 1
    _run_against_mirror(runner, tmp_path, program, b, rows, n, n * 7 + rows)


@pytest.mark.parametrize("kind,codes", [("rms", 8), ("ln", 8), ("rms", 16),
                                        ("ln", 16)])
def test_kernel_is_the_mirror_on_nonfinite_inputs(runner, tmp_path, kind, codes):
    """NaN of either sign, the infinities and +-FLT_MAX in the rows the
    folded pattern's program reads, int8 and int16 codes: the kernel's bits
    are mirrors.qnorm's. Each quantise step takes a NaN to the zero point's
    code and an infinity to a rail (quant_rules.h's rule; qnorm.cpp's
    code_of and code16 through quant_scaled), as ExecuTorch's quantize does.
    Without quant_scaled a NaN fails the rail compares and takes the bottom
    rail instead."""
    n, rows = 24, 5
    model = Hand(n, kind).eval()
    if codes == 16:
        wide(model)
    ep, (x,) = export(model, n=n, rows=rows)
    probe = torch.export.export(model, (x,), strict=False)
    report = qn.fold_qnorms(probe)
    (_, b), = report.fused
    program, norms = _lower(ep, fused_norms=True)
    assert norms == 1
    pte = tmp_path / "qn.pte"
    pte.write_bytes(program)
    xs = x.numpy().astype(F32).copy()
    nan, inf, big = np.float32(np.nan), np.float32(np.inf), np.finfo(F32).max
    xs[0, 3] = nan
    xs[1, 5], xs[1, 7] = -nan, inf
    xs[2, 0] = -inf
    xs[3, 1], xs[3, 2] = big, -big
    xs[4, :] = nan
    inp, out = tmp_path / "in.bin", tmp_path / "out.bin"
    xs.tofile(inp)
    done = subprocess.run([str(runner), "--pte", str(pte), "--input", str(inp),
                           "--output", str(out)], capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    got = np.fromfile(out, dtype=F32)
    want = qm.qnorm(xs, b.tables.numpy(), b.quant.numpy(),
                    None if b.weight is None else b.weight.numpy(),
                    None if b.bias is None else b.bias.numpy(),
                    b.kind, b.flags, b.eps).reshape(-1)
    assert np.array_equal(np.isnan(got), np.isnan(want)), (got, want)
    ok = ~np.isnan(want)
    assert np.array_equal(bits(got[ok]), bits(want[ok])), (got, want)
    assert "qnorm dispatch: calls 1," in done.stdout


class Direct(nn.Module):
    """espressif::qnorm called directly on a build's constants."""

    def __init__(self, b):
        super().__init__()
        self.register_buffer("tables", b.tables.clone())
        self.register_buffer("quant", b.quant.clone())
        self.register_buffer("weight", b.weight.clone())
        if b.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", b.bias.clone())
        self.kind = b.kind

    def forward(self, x):
        return torch.ops.espressif.qnorm.default(
            x, self.tables, self.quant, self.weight, self.bias, self.kind, 0,
            0.0)


@pytest.mark.parametrize("kind", [qn.RMS16, qn.LAYER16])
def test_kernel_program_steps(runner, tmp_path, kind):
    """synthetic_build's programs (every step, exceptions) on the host
    runner against the mirror."""
    b = synthetic_build(kind)
    rows = 5
    x = torch.randn(rows, b.n)
    ep = torch.export.export(Direct(b).eval(), (x,), strict=False)
    program, norms = _lower(ep)
    assert norms == 1
    _run_against_mirror(runner, tmp_path, program, b, rows, b.n, 17 + kind)


def _malformed(b, edit):
    """synthetic_build's RMS16 constants with one word edited."""
    import dataclasses

    tables, quant = b.tables.clone(), b.quant.clone()
    w = tables.tolist()
    steps = 4 + qn.TABLE_WORDS                    # one program, one table
    data_at, exc_at = w[4 + 8], w[4 + 10]
    last = steps + qn.STEP_WORDS * (w[3] - 1)
    if edit == "count":
        tables[0] = 2
    elif edit == "op":
        tables[steps] = 99
    elif edit == "table first":
        tables[steps] = qn.TABLE
    elif edit == "data offset":
        tables[4 + 8] = tables.numel()
    elif edit == "exception order":
        tables[exc_at + 2] = tables[exc_at]
    elif edit == "coefficient":
        tables[data_at] = 0x7FC00000              # NaN
    elif edit == "ends on a code":
        tables[last:last + 6] = torch.tensor(
            [qn.CODE, qn._bits(4096.0), 0, 0, -32767, 32767])
    elif edit == "quantiser":
        quant[0, 0] = 0.0
    return dataclasses.replace(b, tables=tables, quant=quant)


@pytest.mark.parametrize("edit,why", [
    ("count", "program or table count"), ("op", "program step"),
    ("table first", "program step"), ("data offset", "table data"),
    ("exception order", "exceptions"), ("coefficient", "coefficient"),
    ("ends on a code", "program ends on a code"), ("quantiser", "quantiser")])
def test_kernel_refuses_malformed_tables(runner, tmp_path, edit, why):
    """One edit per check of the int16 shim: the instruction fails, naming
    what is wrong (the control is test_kernel_program_steps)."""
    b = _malformed(synthetic_build(qn.RMS16), edit)
    x = torch.randn(5, b.n)
    ep = torch.export.export(Direct(b).eval(), (x,), strict=False)
    program, norms = _lower(ep)
    assert norms == 1
    pte, inp = tmp_path / "bad.pte", tmp_path / "in.bin"
    pte.write_bytes(program)
    x.numpy().tofile(inp)
    done = subprocess.run([str(runner), "--pte", str(pte), "--input", str(inp),
                           "--output", str(tmp_path / "out.bin")],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert f"[qnorm] refused: {why}" in done.stdout + done.stderr, \
        (done.stdout + done.stderr)[-2000:]
