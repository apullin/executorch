# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::lut16 (_passes/fold_lut16_pass.py): the op, the fold, the rows.

  reference   the op's torch body == mirrors.lut16, bit for bit, every
              scheme, on ties, rails, specials and random inputs
  rows        export_lut16's seven a16w8 sites fold to one table each, the
              folded program's torch reference == the mirror, flips against
              eager within the bar (and reported with codes)
  refusals    quant_lut's kept (lut-dq-mismatch), lut16-output, lut16-budget;
              int8 sites are quant_lut's; a refused site's graph is untouched
  default     on; luts16=False on the partitioner or on lower_esp32
              never runs the fold; on, a graph with no int16 site is
              byte-identical to off
  recipe      a stock Arm a16w8 graph folds its activations
  host        each row on the host runner (ESP32_HOST_RUNNER, built from this
              tree) == the mirror, and its lut16 dispatch line
"""

import os
import re
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import lut16 as E
from executorch.backends.esp32s3.quantizer import lut_fit
from executorch.backends.esp32s3._passes import fold_lut16_pass as Q16
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.host import host_runner

S12 = 2.0 ** -12
qd = torch.ops.quantized_decomposed


def _site_table(fn, scheme, size, s1=S12, s2=S12, qmin=-32767, qmax=32767):
    site = lut_fit.site_for(fn, s1, 0, s2, 0, qmin=qmin, qmax=qmax)
    return lut_fit.fit_scheme(site, scheme, size)


def _inputs(s1, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    ties = (np.arange(-40, 40) + 0.5) * s1
    rails = np.array([32767.5, -32767.5, 32766.5, 40000, -40000, 65536, -65536, 70000,
                      -70000, 1e30, -1e30]) * s1
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45],
                        dtype=np.float64)
    x = np.concatenate([rng.normal(0, 3, n), rng.uniform(-9, 9, n), ties, rails, specials])
    return x.astype(np.float32)


@pytest.mark.parametrize("fn,scheme,size", [
    ("sigmoid", "direct", None), ("silu", "uniform_linear", 512),
    ("tanh", "uniform_cubic", 128), ("rsqrt", "nsau_linear", 5),
    ("reciprocal", "nsau_cubic", 3), ("exp", "nsau_cubic", 2)])
def test_reference_is_the_mirror(fn, scheme, size):
    qmin, qmax = (-512, 511) if scheme == "direct" else (-32767, 32767)
    s1 = S12 * 8 if scheme == "direct" else S12
    t = _site_table(fn, scheme, size, s1=s1, qmin=qmin, qmax=qmax)
    x = _inputs(s1)
    inv = float(qm.reciprocal(s1))
    args = (inv, 0, qmin, qmax, t.scheme_id, t.lo, t.hi, t.shift, t.G, S12, 0,
            t.out_qmin, t.out_qmax)
    table = torch.from_numpy(np.ascontiguousarray(t.data)).to(Q16.TABLE_DTYPE[t.scheme_id])
    got = Q16.LUT16(torch.from_numpy(x), table, *args).numpy()
    want = qm.lut16(x, t.data, *args)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    # NaN takes the zero point's code, ExecuTorch 1.6's rule (as quant_lut's
    # kernel does), in the mirror and the torch reference
    nan = np.isnan(x)
    for zp in (0, 5):
        assert np.array_equal(qm.lut16_quantize(x[nan], inv, zp, qmin, qmax),
                              np.full(nan.sum(), zp))
        assert np.array_equal(
            Q16.quantize16(torch.from_numpy(x[nan]), inv, zp, qmin, qmax).numpy(),
            np.full(nan.sum(), zp))


def test_quantize_step_is_quantize_per_tensor():
    x = _inputs(S12, seed=3)
    x = x[np.isfinite(x)]
    inv = float(qm.reciprocal(S12))
    want = qd.quantize_per_tensor.default(torch.from_numpy(x), S12, 5, -32767, 32767,
                                          torch.int16).numpy().astype(np.int64)
    assert np.array_equal(qm.lut16_quantize(x, inv, 5, -32767, 32767), want)
    got = Q16.quantize16(torch.from_numpy(x), inv, 5, -32767, 32767).numpy()
    assert np.array_equal(got, want)


# --- the rows ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    built = {}
    for fixture in E.fixtures():
        et = E.build(fixture)
        built[fixture.name] = (fixture, bytes(et.buffer))
    return built


def test_rows_fold_and_match_the_mirror(rows):
    for name, (fixture, program) in rows.items():
        gap = fixture.gap
        print(f"{name}: {gap['table']}; {gap['flips']} of {gap['inputs']} inputs flip "
              f"against eager {gap['flipped_inputs']}")
        assert fixture.table.flips <= fixture.table.bar
        assert gap["max_steps"] <= 1 and gap["flips"] <= 2
        assert len(program) < 16 * 1024


# --- refusals --------------------------------------------------------------------------

class Pair16(torch.nn.Module):
    def __init__(self, fn, s1=S12, s2=S12, dq_scale=None, out_q=True, dtype=torch.int16,
                 qmin=-32767, qmax=32767):
        super().__init__()
        self.fn, self.s1, self.s2 = fn, s1, s2
        self.dq_scale = s1 if dq_scale is None else dq_scale
        self.out_q, self.dtype, self.qmin, self.qmax = out_q, dtype, qmin, qmax

    def forward(self, x):
        c = qd.quantize_per_tensor.default(x, self.s1, 0, self.qmin, self.qmax, self.dtype)
        d = qd.dequantize_per_tensor.default(c, self.dq_scale, 0, self.qmin, self.qmax,
                                             self.dtype)
        y = self.fn(d)
        if not self.out_q:
            return y
        c2 = qd.quantize_per_tensor.default(y, self.s2, 0, self.qmin, self.qmax, self.dtype)
        return qd.dequantize_per_tensor.default(c2, self.s2, 0, self.qmin, self.qmax,
                                                self.dtype)


def _export(model, n=64):
    return torch.export.export(model.eval(), (torch.randn(n),), strict=True)


def _graph_text(ep):
    return str(ep.graph_module.graph)


@pytest.mark.parametrize("model,rule", [
    (Pair16(torch.sigmoid, dq_scale=S12 * 2), "lut-dq-mismatch"),
    (Pair16(torch.sigmoid, out_q=False), "lut16-output"),
])
def test_refusals_leave_the_graph(model, rule):
    ep = _export(model)
    before = _graph_text(ep)
    report = Q16.fold_fitted_luts(ep)
    assert [r[2] for r in report.refused] == [rule] and not report.tables
    assert _graph_text(ep) == before


def test_budget_refusal():
    ep = _export(Pair16(F.silu))
    before = _graph_text(ep)
    report = Q16.fold_fitted_luts(ep, budget=300)
    assert [r[2] for r in report.refused] == ["lut16-budget"]
    assert "best uniform_cubic" in report.refused[0][3]
    assert _graph_text(ep) == before


def test_int8_sites_are_quant_luts():
    ep = _export(Pair16(torch.sigmoid, s1=1 / 32, s2=1 / 128, dtype=torch.int8,
                        qmin=-128, qmax=127))
    before = _graph_text(ep)
    report = Q16.fold_fitted_luts(ep)
    assert not report.tables and not report.refused
    assert _graph_text(ep) == before


# --- the default ---------------------------------------------------------------------------

def _pte(model, **kwargs):
    """The program, lowered with Esp32Partitioner(luts16=partitioner_luts16)
    (its default when not given) and lower_esp32(**kwargs)."""
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.exir import ExecutorchBackendConfig

    ep = _export(model)
    options = ({"luts16": kwargs.pop("partitioner_luts16")}
               if "partitioner_luts16" in kwargs else {})
    partitioner = Esp32Partitioner(**options)
    lowered = _passes.lower_esp32(ep, [partitioner], **kwargs)
    _passes.apply_passes(lowered, quiet=True)
    return bytes(lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16)).buffer), lowered


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(64, 64)
        self.b = torch.nn.Linear(64, 64)

    def forward(self, x):
        return self.b(torch.sigmoid(self.a(x)))


def test_on_by_default_and_off_changes_nothing():
    """On by default (Esp32Partitioner's luts16, which lower_esp32
    follows); luts16=False on either never runs the fold (no report, no
    table) and both spellings give the same bytes; True on lower_esp32
    overrides a partitioner's False."""
    default, lowered = _pte(Pair16(torch.tanh))
    assert len(lowered.lut16_report.tables) == 1
    off, lowered = _pte(Pair16(torch.tanh), partitioner_luts16=False)
    assert lowered.lut16_report is None and b"lut16" not in off
    off_here, lowered = _pte(Pair16(torch.tanh), luts16=False)
    assert lowered.lut16_report is None and off_here == off
    assert default != off
    on, lowered = _pte(Pair16(torch.tanh), luts16=True, partitioner_luts16=False)
    assert on == default and len(lowered.lut16_report.tables) == 1


def test_on_without_int16_sites_is_byte_identical():
    class Lin(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x) * 2

    for model in (Lin(), Pair16(torch.sigmoid, s1=1 / 32, s2=1 / 128, dtype=torch.int8,
                                qmin=-128, qmax=127)):
        off, _ = _pte(model, partitioner_luts16=False)
        on, lowered = _pte(model)
        assert on == off
        assert not lowered.lut16_report.tables


# --- a stock a16w8 graph -----------------------------------------------------------------

def test_arm_a16w8_recipe_folds_its_activations():
    from executorch.backends.esp32s3.test.quant_census import recipes

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1)
            self.a = torch.nn.Linear(32, 32)
            self.b = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.b(F.silu(self.a(x))) + torch.tanh(x)

    torch.manual_seed(2)
    inputs = (torch.randn(4, 32),)
    batches = [(torch.randn(4, 32),) for _ in range(4)]
    model = recipes.quantize("arm-a16w8", Block().eval(), inputs, batches).program
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(model, inputs, strict=True)
    lowered = _passes.lower_esp32(ep, [Esp32Partitioner(luts16=True)])
    report = lowered.lut16_report
    print(report.summary())
    for line in report.lines():
        print("  " + line)
    assert len(report.tables) >= 2, report.summary()
    assert not [r for r in report.refused if r[2] == "lut16-budget"]


# --- the host runner ---------------------------------------------------------------------

def _runner():
    path = str(host_runner())
    if not os.path.exists(path):
        pytest.skip(f"no host runner at {path}")
    probe = subprocess.run([path, "--selftest"], capture_output=True, text=True)
    if "lut16 check:" not in probe.stdout:
        pytest.skip(f"{path} has no lut16 kernel")
    return path


def test_host_rows_are_the_mirror(rows, tmp_path):
    runner = _runner()
    for name, (fixture, program) in rows.items():
        pte = tmp_path / f"{name}.pte"
        pte.write_bytes(program)
        x = fixture.input()
        (tmp_path / "in.bin").write_bytes(x.tobytes())
        out = tmp_path / "out.bin"
        run = subprocess.run([runner, "--pte", str(pte), "--input", str(tmp_path / "in.bin"),
                              "--output", str(out)], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        got = np.frombuffer(out.read_bytes(), dtype=np.float32)
        assert np.array_equal(got.view(np.uint32), fixture.reference.view(np.uint32)), name
        m = re.search(r"lut16 dispatch: calls (\d+), elements (\d+), refusals (\d+)",
                      run.stdout)
        assert m and m.group(1) == "1" and m.group(3) == "0", run.stdout[-2000:]


# --- the observer's range (quantizer/lut16_range.py) -----------------------------------------

def test_window_rule():
    from executorch.backends.esp32s3.quantizer import lut16_range as R

    s = S12
    assert R.window((-3000 * s, 3000 * s), s, 0, -32767, 32767) == (-3750, 3750)
    # the codes a range covers: floor at the bottom, ceil at the top; 2 at least
    assert R.window((0.3 * s, 8.6 * s), s, 0, -32767, 32767) == (-2, 11)
    assert R.window((5 * s, 5 * s), s, 3, -32767, 32767) == (6, 10)
    assert R.window((-9.0, 9.0), s, 0, -32767, 32767) == (-32767, 32767)
    assert R.window((20.0, 30.0), s, 0, -32767, 32767) == (32767, 32767)
    for rng in (None, (1.0, -1.0), (float("-inf"), 1.0)):
        assert R.window(rng, s, 0, -32767, 32767) is None


class SiluBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.a = torch.nn.Linear(32, 48)
        self.b = torch.nn.Linear(48, 32)

    def forward(self, x):
        return self.b(F.silu(self.a(x)))


def _a16_converted(carry):
    """SiluBlock under Esp32Quantizer(preset="arm_a16w8"), 8 seeded
    calibration batches, with or without carry_observer_ranges."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
        Esp32Quantizer,
        carry_observer_ranges,
    )

    torch.manual_seed(5)
    x = torch.randn(4, 32)
    gm = torch.export.export(SiluBlock().eval(), (x,), strict=False).module(
        check_guards=False)
    prepared = prepare_pt2e(gm, Esp32Quantizer(preset="arm_a16w8"))
    gen = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for _ in range(8):
            prepared(torch.randn(4, 32, generator=gen) * 0.5)
    if carry:
        assert carry_observer_ranges(prepared) > 0
    return convert_pt2e(prepared), x


def test_strip_keeps_each_range_on_its_quantize():
    from executorch.backends.esp32s3.quantizer import lut16_range as R
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    carried, x = _a16_converted(carry=True)
    plain, _ = _a16_converted(carry=False)
    ranges = [n for n in carried.graph.nodes if R._is_range(n)]
    assert ranges
    for n in ranges:
        assert fold_quant_lut_pass._is(n.args[0], fold_quant_lut_pass.Q)
        assert all(fold_quant_lut_pass._is(u, fold_quant_lut_pass.DQ) for u in n.users)
        lo, hi = n.args[1], n.args[2]
        assert lo <= hi
    with torch.no_grad():
        assert torch.equal(carried(x), plain(x))
    ep = torch.export.export(carried, (x,), strict=False)
    ep_plain = torch.export.export(plain, (x,), strict=False)
    assert R.strip_observed_ranges(ep) == len(ranges)
    assert len([n for n in ep.graph.nodes if R.META in n.meta]) == len(ranges)
    assert _graph_text(ep) == _graph_text(ep_plain)
    assert R.strip_observed_ranges(ep_plain) == 0


def _lowered_pte(ep, luts16):
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.exir import ExecutorchBackendConfig

    lowered = _passes.lower_esp32(ep, [Esp32Partitioner(luts16=luts16)])
    _passes.apply_passes(lowered, quiet=True)
    return bytes(lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16)).buffer), lowered


def test_carried_ranges_fit_smaller_tables():
    carried, x = _a16_converted(carry=True)
    plain, _ = _a16_converted(carry=False)
    _, with_range = _lowered_pte(torch.export.export(carried, (x,), strict=False), True)
    _, graph_only = _lowered_pte(torch.export.export(plain, (x,), strict=False), True)
    r, g = with_range.lut16_report, graph_only.lut16_report
    print(r.summary())
    for line in r.lines() + g.lines():
        print("  " + line)
    assert len(r.tables) == len(g.tables) == 1 and not r.refused and not g.refused
    (node, build), (_, whole) = r.tables[0], g.tables[0]
    lo, hi = r.windows[node]
    t = build.table
    assert lo <= t.lo <= t.hi <= hi and t.codes == hi - lo + 1
    assert t.flips <= t.bar
    assert t.nbytes < whole.table.nbytes
    assert not g.windows


def test_carried_ranges_without_tables_change_no_byte():
    carried, x = _a16_converted(carry=True)
    plain, _ = _a16_converted(carry=False)
    a, _ = _lowered_pte(torch.export.export(carried, (x,), strict=False), False)
    b, _ = _lowered_pte(torch.export.export(plain, (x,), strict=False), False)
    assert a == b


# --- the flagged scheme words: OFFSET and TAIL ---------------------------------------------

def _flagged_tables():
    """(name, Table, s1) for every flagged scheme: NSAU OFFSET tables over a
    window away from 0 and a lopsided one, uniform TAIL tables where f is a
    line toward the rails (and a linear one, forced)."""
    out = []
    recip = lut_fit.site_for("reciprocal", S12, 0, S12, 0)
    out.append(("nsau cubic offset", lut_fit.fit_scheme(recip, "nsau_cubic", 4, lo=534,
                                                        hi=32767), S12))
    out.append(("nsau linear offset", lut_fit.fit_scheme(recip, "nsau_linear", 4, lo=534,
                                                         hi=32767), S12))
    exp = lut_fit.site_for("exp", 2.0 ** -7, 0, S12, 0)
    out.append(("nsau cubic offset lopsided", lut_fit.fit_scheme(exp, "nsau_cubic", 2),
                2.0 ** -7))
    for fn in ("hardswish", "gelu"):
        out.append((f"{fn} tail", lut_fit.fit(lut_fit.site_for(fn, S12, 0, S12, 0)), S12))
    hs = lut_fit.site_for("hardswish", S12, 0, S12, 0)
    out.append(("uniform linear tail", lut_fit.fit_scheme(
        hs, "uniform_linear", 64, lo=-16382, hi=16385, flags=lut_fit.TAIL), S12))
    return out


def test_flagged_reference_is_the_mirror():
    for name, t, s1 in _flagged_tables():
        print(f"{name}: {t.describe()}")
        assert t.flags, name
        x = _inputs(s1)
        inv = float(qm.reciprocal(s1))
        args = (inv, 0, -32767, 32767, t.scheme_id, t.lo, t.hi, t.shift, t.G, S12, 0,
                t.out_qmin, t.out_qmax)
        table = torch.from_numpy(np.ascontiguousarray(t.data)).to(
            Q16.TABLE_DTYPE[t.scheme_id])
        got = Q16.LUT16(torch.from_numpy(x), table, *args).numpy()
        want = qm.lut16(x, t.data, *args)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), name
        codes = torch.arange(-32768, 32768)
        assert np.array_equal(
            Q16.codes16(codes, table, t.scheme_id, t.lo, t.hi, t.shift, t.G, t.out_qmin,
                        t.out_qmax).numpy(),
            t.evaluate(codes.numpy())), name


def test_rows_carry_the_flags(rows):
    flags = {name: fixture.table.flags for name, (fixture, _) in rows.items()}
    assert flags["q16_recip_window"] == lut_fit.OFFSET
    assert flags["q16_gelu"] == lut_fit.TAIL
    assert flags["q16_reciprocal"] == lut_fit.OFFSET and flags["q16_rsqrt"] == lut_fit.OFFSET
    window = rows["q16_recip_window"][0]
    assert window.table.lo == 534 and window.table.hi == 32767
    assert window.gap["flips"] == 0
