# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""lut_fit: the int16 table fitter and mirrors.lut16.

  chooser     NSAU for rsqrt, reciprocal and log, direct for ranges up to
              1,024 codes, uniform cubic otherwise, within the bar and the
              budget, against torch's own E; BudgetExceeded when nothing fits
  flagged     an OFFSET table is the plain table sliced, a TAIL table the plain
              table inside its range; the scheme words and int32 safety bounds
  mirror      an NSAU table's two signs, the linear schemes' overflow check

mirrors.lut16 itself is held to the kernel's bits by test_lut16.py (the
host runner) and by the lut16 parity rows. Imports torch (the chooser's
targets).
"""

import numpy as np
import pytest

from executorch.backends.esp32s3.quantizer import lut_fit
from executorch.backends.esp32s3.quantizer import mirrors as qm

S12 = 2.0 ** -12
ALL = np.arange(-32768, 32768, dtype=np.int64)
SYM = np.arange(-32767, 32768, dtype=np.int64)


def _random_linear(rng, n, F, G, mode="one"):
    """Random continuous nodes inside lut16_linear_safe's bounds."""
    span = 2 ** 29 // (1 << F) if mode == "one" else 2 ** 28
    base = rng.integers(-32768, 32768) << G
    steps = rng.integers(-(1 << (G + 6)), 1 << (G + 6), n)
    V = np.clip(base + np.cumsum(steps), -span // 2, span // 2)
    return V.astype(np.int64)


# --- the mirror -------------------------------------------------------------------------

def _random_cubic(rng, K):
    C = np.zeros((K, 4), dtype=np.float32)
    C[:, 0] = rng.uniform(-40000, 40000, K)
    C[:, 1:] = rng.uniform(-3000, 3000, (K, 3))
    C[: K // 8, 0] = np.round(C[: K // 8, 0]) + 0.5      # rne ties at t = 0
    C[: K // 8, 1:] = 0
    return C.astype(np.float32)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_mirror_nsau_cubic_signs(m):
    """Both signs: a negative code reads the table's second half at |c|, a
    non-negative one the first half, each as its one-sided table would."""
    rng = np.random.default_rng(10 + m)
    half = qm.lut16_nsau_half(0, 32767, m)
    C = _random_cubic(rng, half)
    C2 = np.concatenate([C, _random_cubic(rng, half)])
    both = qm.lut16_codes(SYM, 4, C2, -32767, 32767, m, 0, -32768, 32767)
    neg = qm.lut16_codes(-SYM[SYM < 0], 4, C2[half:], 0, 32767, m, 0, -32768, 32767)
    assert np.array_equal(both[SYM < 0], neg)
    pos = qm.lut16_codes(SYM[SYM >= 0], 4, C, 0, 32767, m, 0, -32768, 32767)
    assert np.array_equal(both[SYM >= 0], pos)


def test_linear_safe_rejects_overflow():
    V = np.array([2 ** 22, -(2 ** 22)], dtype=np.int64)
    assert not qm.lut16_linear_safe(1, V, 0, 511, 9, 0)
    assert qm.lut16_linear_safe(1, V // 4, 0, 511, 9, 0)


# --- the chooser (torch's E) --------------------------------------------------------

@pytest.mark.parametrize("func,scheme", [
    ("sigmoid", "uniform_cubic"), ("tanh", "uniform_cubic"), ("silu", "uniform_cubic"),
    ("gelu", "uniform_cubic"), ("exp", "uniform_cubic"), ("rsqrt", "nsau_cubic"),
    ("reciprocal", "nsau_cubic"), ("log", "nsau_cubic")])
def test_chooser_picks(func, scheme):
    site = lut_fit.site_for(func, S12, 0, S12, 0)
    table = lut_fit.fit(site)
    print(f"{func}: {table.describe()}")
    assert table.scheme == scheme
    assert table.flips <= table.bar
    assert table.nbytes <= lut_fit.DEFAULT_BUDGET
    assert np.abs(table.data).max() <= lut_fit.COEF_MAX


def test_chooser_direct_for_narrow_ranges():
    site = lut_fit.site_for("sigmoid", S12 * 8, 3, S12, 0, qmin=-512, qmax=511,
                            out_qmin=-32767, out_qmax=32767)
    table = lut_fit.fit(site)
    assert table.scheme == "direct" and table.flips == 0
    assert table.nbytes == 2 * 1024


def test_chooser_budget():
    site = lut_fit.site_for("silu", S12, 0, S12, 0)
    with pytest.raises(lut_fit.BudgetExceeded) as error:
        lut_fit.fit(site, lut_fit.Spec(budget=300))
    assert error.value.best is not None and error.value.best.nbytes <= 300


def test_linear_schemes_fit_safely():
    site = lut_fit.site_for("sigmoid", S12, 0, S12, 0)
    for scheme, size in (("uniform_linear", 512), ("nsau_linear", 5)):
        table = lut_fit.fit_scheme(site, scheme, size)
        print(table.describe())
        assert qm.lut16_linear_safe(table.scheme_id, table.data, table.lo, table.hi,
                                    table.shift, table.G)
        assert table.flips < 0.01 * table.codes


def test_chooser_coarse_input_exp():
    """llama3_2's softmax exp under Arm a16w8: a coarse input (s1 = 2^-7) and
    2^-12 out. Past ln(8) exp sits on the output rail: active_range cuts the
    table there, so no segment has to follow the kink."""
    site = lut_fit.site_for("exp", 2.0 ** -7, 0, S12, 0)
    lo, hi = lut_fit.active_range(site)
    assert lo == -32767 and 266 <= hi <= 300
    table = lut_fit.fit(site)
    print(table.describe())
    assert table.flips <= table.bar and table.hi == hi


# --- the flagged schemes: OFFSET and TAIL ------------------------------------------

@pytest.mark.parametrize("lo,hi,m", [(534, 32767, 5), (-32767, 267, 2), (-5000, -37, 3),
                                     (-20, 1000, 2), (1, 3000, 4), (-32767, 32767, 0),
                                     (0, 32767, 3), (-1, 0, 1), (7, 7, 2)])
@pytest.mark.parametrize("base", [3, 4])
def test_mirror_offset_is_the_plain_table_sliced(lo, hi, m, base):
    """An OFFSET table is the plain one with the segments no code of [lo, hi]
    reaches dropped: the same output on every code."""
    rng = np.random.default_rng(abs(lo * 7 + hi + m))
    sid = base | lut_fit.OFFSET
    n_plain = qm.lut16_entries(base, lo, hi, m)
    if base == 4:
        plain = _random_cubic(rng, n_plain)
    else:
        plain = _random_linear(rng, n_plain, 0, 2, "two")
    jp, _, _ = qm.lut16_index(ALL, base, lo, hi, m)
    jo, _, _ = qm.lut16_index(ALL, sid, lo, hi, m)
    n = qm.lut16_entries(sid, lo, hi, m)
    table = np.zeros((n, 4), np.float32) if base == 4 else np.zeros(n, np.int64)
    table[jo] = plain[jp]
    if base == 3:
        table[jo + 1] = plain[jp + 1]
    assert n <= n_plain and jo.min() == 0 and jo.max() + (base == 3) == n - 1
    got = qm.lut16_codes(ALL, sid, table, lo, hi, m, 2, -32768, 32767)
    want = qm.lut16_codes(ALL, base, plain, lo, hi, m, 2, -32768, 32767)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("lo,hi,F", [(-3000, 3000, 7), (100, 9000, 9), (-16382, 16385, 12),
                                     (-100, 155, 0), (-32768, 32767, 9)])
def test_mirror_tail_is_the_plain_table_inside(lo, hi, F):
    """A TAIL table changes only the codes past the table: inside it, linear
    or cubic, the output is the plain table's."""
    rng = np.random.default_rng(F + 3)
    K = ((hi - lo) >> F) + 1
    G = 2
    V = _random_linear(rng, K + 1, F, G)
    V[0], V[-1] = V[1] - 3, V[-2] + 5          # gentle edges: the reach stays in int32
    inside = (ALL >= lo) & (ALL <= hi)
    if qm.lut16_linear_safe(1 | lut_fit.TAIL, V, lo, hi, F, G):
        got = qm.lut16_codes(ALL, 1 | lut_fit.TAIL, V, lo, hi, F, G, -32768, 32767)
        plain = qm.lut16_codes(ALL, 1, V, lo, hi, F, G, -32768, 32767)
        assert np.array_equal(got[inside], plain[inside])
    C = _random_cubic(rng, K)
    C[0, 2:] = C[-1, 2:] = 0.0
    C[0, 1] = C[-1, 1] = np.float32(2.0 ** F / 4)
    got = qm.lut16_codes(ALL, 2 | lut_fit.TAIL, C, lo, hi, F, 0, -32768, 32767)
    plain = qm.lut16_codes(ALL, 2, C, lo, hi, F, 0, -32768, 32767)
    assert np.array_equal(got[inside], plain[inside])


def test_scheme_words():
    for sid in (0, 1, 2, 3, 4, 11, 12, 17, 18):
        assert qm.lut16_scheme_error(sid) is None
    assert qm.lut16_scheme_error(5) == "scheme"
    assert qm.lut16_scheme_error(16) == "tail on a direct table"
    assert qm.lut16_scheme_error(20) == "tail on an NSAU table"
    assert qm.lut16_scheme_error(10) == "offset on a uniform table"
    assert qm.lut16_scheme_error(8) == "offset on a direct table"
    assert qm.lut16_scheme_error(34) == "scheme"


def test_tail_safety_bounds():
    lo, hi, F = -16382, 16385, 12
    C = np.zeros((8, 4), np.float32)
    C[:, 0] = 100.0
    assert qm.lut16_cubic_safe(18, C, lo, hi, F)
    C[0, 3] = 1.0e5          # 4^3 * 1e5 past 2^22 at code -32768
    assert qm.lut16_cubic_safe(2, C, lo, hi, F)
    assert not qm.lut16_cubic_safe(18, C, lo, hi, F)
    V = np.zeros(48, np.int64)
    V[0] = 1_000_000         # inside int32 over its segment, not out to -32768
    assert qm.lut16_linear_safe(1, V, -3000, 3000, 7, 4)
    assert not qm.lut16_linear_safe(17, V, -3000, 3000, 7, 4)


def _window_site(fn, lo, hi, s1=S12, s2=S12):
    """A site over the window [lo, hi] of q1's codes (quant_lut16.site_from's
    shape: the grid holds the window, the bar is its float32 floor)."""
    full = lut_fit.site_for(fn, s1, 0, s2, 0)
    sel = slice(lo - full.inq.qmin, hi - full.inq.qmin + 1)
    y = full.y()[sel]
    return lut_fit.Site(fn, lut_fit.Grid(s1, 0, lo, hi), full.outq, values=lambda _x: y,
                        values64=full.values64, target=full.E()[sel], floor_bar=True)


def test_window_reciprocal_takes_the_offset():
    """llama3_2's reciprocal window: NSAU segments from the
    window's first code; the same outputs as the plain layout, half the bytes.
    At m = 4 the c0 flip fit leaves 9 codes a hair from a rounding tie past
    the window's float32 floor (3); tilting those segments (refine_joint)
    leaves 2."""
    site = _window_site("reciprocal", 534, 32767)
    table = lut_fit.fit_smallest(site)
    print(table.describe())
    assert table.scheme == "nsau_cubic" and table.flags == lut_fit.OFFSET
    assert lut_fit.within_bar(table) and table.shift == 4 and table.nbytes == 1536
    plain = lut_fit.fit_scheme(site, "nsau_cubic", table.shift, flags=0)
    assert np.array_equal(plain.evaluate(site.codes), table.evaluate(site.codes))
    assert table.nbytes < plain.nbytes // 2 + 512
    c0_only = lut_fit.fit_scheme(lut_fit.Site(**{**site.__dict__, "_cache": {},
                                                 "floor_bar": False}), "nsau_cubic", 4)
    assert c0_only.flips > table.bar >= table.flips


@pytest.mark.parametrize("fn,kwargs", [("hardswish", {}), ("gelu", {}), ("pow", {"args": (2,)})])
def test_tail_tables_where_f_is_a_line(fn, kwargs):
    """Graph-only: where f is a line toward the rails (hardswish x and 0, GELU
    x and 0, x^2 on the output rail) the TAIL table stops there, smaller,
    with no flip past it."""
    site = lut_fit.site_for(fn, S12, 0, S12, 0, **kwargs)
    if fn == "pow":
        site.tail_rule = True
    table = lut_fit.fit(site)
    print(f"{fn}: {table.describe()}")
    assert table.flags == lut_fit.TAIL and lut_fit.within_bar(table)
    assert table.tail_flips == 0 and table.nbytes < 2048
    assert qm.lut16_cubic_safe(table.scheme_id, table.data, table.lo, table.hi, table.shift)
    whole = lut_fit.fit(lut_fit.Site(fn, site.inq, site.outq, values=site.values,
                                     values64=site.values64, target=site.E(),
                                     tail_rule=False))
    assert table.nbytes < whole.nbytes


def test_no_tail_where_f_is_no_line():
    site = lut_fit.site_for("silu", S12, 0, S12, 0)
    table = lut_fit.fit(site)
    assert table.flags == 0
    site = lut_fit.site_for("sigmoid", S12, 0, S12, 0)
    assert not site.tails and lut_fit.fit_tail(site, 64) is None
