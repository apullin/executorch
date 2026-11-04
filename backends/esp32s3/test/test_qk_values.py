# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The q@kT value check (operator_support/qk_values.py): a grid the sample
runs show keeps the bmm portable, whatever the rules made of its spelling.

  judge     each test finds its own grid (and is the one that does, so a
            test turned off fails here), plain values stay plain, a value
            that does not vary or has no finite values gets no verdict, rows
            a run never wrote and lookups at token ids make no grid;
  gate      grids the graph rules alone admit (one per kind of hole
            in the rules, the residual case, and grids behind ops
            that compute) are kept portable by their values
            and named in the partition log, while plain spellings stay
            delegated;
  pitfalls  degenerate example inputs, token ids drawn within their table
            and positions kept at the example's values, written buffers run
            on copies with state carried, runs that raise, determinism and
            the global RNG, and no runs without a q@kT candidate.
"""

import contextlib
import io

import numpy as np
import pytest
import torch
import torch.nn as nn

from executorch.backends.esp32s3.operator_support import qk_operands, qk_values
from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
    KIND_QK_VALUES,
)

# --- judge ----------------------------------------------------------------------


def _runs(f, shape, n=12, seed=0):
    g = np.random.default_rng(seed)
    return [np.asarray(f(g.standard_normal(shape).astype(np.float32), i, g),
                       dtype=np.float32) for i in range(n)]


def _rowscale(x, levels=127):
    return np.abs(x).max(-1, keepdims=True) / levels


_C16 = np.random.default_rng(7).uniform(0.05, 0.25, 16).astype(np.float32)
_C716 = np.random.default_rng(8).uniform(0.05, 0.25, (7, 16)).astype(np.float32)
_BOOK = np.random.default_rng(9).standard_normal((8, 16)).astype(np.float32)

# Each spelling and the one test that finds it first (in judge's order).
JUDGE_CASES = {
    "sign": (lambda x, i, g: np.sign(x), (7, 16), "levels"),
    "mask_times_row_scale": (
        lambda x, i, g: (x > 0) * _rowscale(x, 1), (7, 16), "row levels"),
    "codebook_rows": (
        lambda x, i, g: _BOOK[g.integers(0, 8, 7)], (7, 16), "vector levels"),
    "int4_rows": (
        lambda x, i, g: np.round(x / _rowscale(x, 7)) * _rowscale(x, 7),
        (7, 16), "scaled levels"),
    "int8_rows": (
        lambda x, i, g: np.round(x / _rowscale(x)) * _rowscale(x),
        (7, 16), "row lattice"),
    "int8_groups_of_4": (
        lambda x, i, g: (np.round(x.reshape(7, 4, 4) / _rowscale(
            x.reshape(7, 4, 4))) * _rowscale(x.reshape(7, 4, 4))).reshape(7, 16),
        (7, 16), "group lattice"),
    "int8_channels_per_run": (   # KIVI's keys: a step per channel per call
        lambda x, i, g: np.round(x / (np.abs(x).max(0) / 127)) * (
            np.abs(x).max(0) / 127), (7, 16), "channel lattice"),
    "wide_codes_per_run": (      # rows too narrow for the row tests
        lambda x, i, g: np.round(x * 300) * (0.001 * (i + 1)), (2, 4),
        "tensor lattice"),
    "codes_one_row_channel_scale": (
        lambda x, i, g: np.trunc(x) * _C16, (1, 16),
        "channel lattice across inputs"),
    "two_codes_a_run_one_step": (   # six runs: channels of six values
        lambda x, i, g: np.repeat(np.round(x[:, :2] * 400), 2, -1) * 0.01,
        (1, 4), "tensor lattice across inputs", 6),
    "element_step": (
        lambda x, i, g: np.floor(x / _C716) * _C716, (7, 16),
        "element lattice across inputs"),
}


@pytest.mark.parametrize("case", sorted(JUDGE_CASES))
def test_each_test_finds_its_grid(case):
    f, shape, test, *n = JUDGE_CASES[case]
    v = qk_values.judge(_runs(f, shape, *n))
    assert v.grid is True and v.test == test, v


PLAIN_CASES = {
    "normal": (lambda x, i, g: x, (7, 16)),
    "affine": (lambda x, i, g: 2 * x + 1, (7, 16)),
    "relu": (lambda x, i, g: np.maximum(x, 0), (7, 16)),
    "softmax_rows": (lambda x, i, g: np.exp(x) / np.exp(x).sum(-1, keepdims=True),
                     (7, 16)),
    "one_row": (lambda x, i, g: x * 3, (1, 16)),
    "one_row_relu": (lambda x, i, g: np.maximum(x, 0), (1, 16)),
    "keys": (lambda x, i, g: x @ _BOOK.T @ _BOOK / 8, (31, 16)),
}


@pytest.mark.parametrize("case", sorted(PLAIN_CASES))
def test_plain_values_stay_plain(case):
    f, shape = PLAIN_CASES[case]
    assert qk_values.judge(_runs(f, shape)).grid is False


def test_no_verdict_without_variation_or_finite_values():
    same = [np.arange(16, dtype=np.float32).reshape(1, 16) / 7] * 6
    assert qk_values.judge(same) == (None, "constant",
                                     "the same values in every run")
    near = [np.full((7, 16), 3.0, np.float32) + 1e-6 * k for k in range(6)]
    assert qk_values.judge(near).test == "degenerate"
    nan = [np.full((7, 16), np.nan, np.float32)] * 6
    assert qk_values.judge(nan).grid is None


def test_non_finite_rows_left_out():
    def holes(f):
        def g_(x, i, g):
            v = f(x, i, g).astype(np.float32)
            v[i % 7] = np.nan if i % 2 else np.inf
            return v
        return g_
    grid = qk_values.judge(_runs(holes(JUDGE_CASES["int8_rows"][0]), (7, 16)))
    assert grid.grid is True and grid.test == "row lattice"
    plain = qk_values.judge(_runs(holes(lambda x, i, g: x), (7, 16)))
    assert plain.grid is False


def _cache_runs(fill, rows, grid, n=12, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = np.full((32, 16), fill, np.float32)
        for r in rows:
            x = g.standard_normal(16).astype(np.float32)
            c[r] = np.round(x / _rowscale(x)) * _rowscale(x) if grid else x
        out.append(c)
    return out


def _judge_classed(runs, redrawn):
    """judge on rows classed as the check classes them (_row_classes): the
    unwritten left out, the rest judged with every test; None when every
    row is unwritten."""
    unwritten, carried, same = qk_values._row_classes(runs, redrawn)
    assert not carried.any()            # no replay here: nothing carried
    keep = qk_values._keep_rows(unwritten, same)
    if not len(keep):
        return None
    return qk_values.judge_runs([r[keep] for r in runs])[0]


def _redrawn(runs, written_rows, seed=5):
    """Run 0 with every row but `written_rows` drawn afresh (the buffer's
    initial content redrawn; the rows run 0 wrote kept)."""
    out = np.random.default_rng(seed).standard_normal(runs[0].shape).astype(
        np.float32)
    out[written_rows] = runs[0][written_rows]
    return out


def test_unwritten_rows_carry_no_information():
    """A cache's rows no run wrote -- the same in every run, and moved when
    the buffer's initial content is drawn afresh -- are left out: they
    neither make a grid of a plain cache nor hide the grid a cache
    stores."""
    for fill in (0.0, 1.0):
        runs = _cache_runs(fill, [3, 4], False)
        unwritten, carried, _ = qk_values._row_classes(runs, _redrawn(runs, [3, 4]))
        assert unwritten.sum() == 30 and not carried.any()
        assert _judge_classed(runs, _redrawn(runs, [3, 4])).grid is False
        runs = _cache_runs(fill, [3, 4], True)
        v = _judge_classed(runs, _redrawn(runs, [3, 4]))
        assert v.grid is True and v.test == "row lattice", (fill, v)


def test_lookups_at_token_ids_are_not_levels():
    """Runs that vary the token ids alone see a table's own values again:
    a lookup at ids is plain, though its values are few."""
    table = np.random.default_rng(3).standard_normal(12).astype(np.float32)
    runs = _runs(lambda x, i, g: table[g.integers(0, 12, (7, 16))], (7, 16))
    assert qk_values.judge(runs).grid is True
    v = qk_values.judge(runs, lookups=True)
    assert v.grid is False, v


def test_lattice_at_a_full_int8_range():
    """Codes spanning exactly the 255 steps of int8 (one run at -128, one at
    127), each element over its own constant, as a static pair over a table
    row makes them: the span measures a hair over 255 steps after the
    division, and a step search bounded by the measured span truncates its
    candidates to none."""
    g = np.random.default_rng(4)
    runs = []
    for i in range(12):
        codes = (np.full((7, 16), -128.0) if i == 0 else np.full((7, 16), 127.0)
                 if i == 1 else np.clip(np.round(g.standard_normal((7, 16)) / 0.0123)
                                        + 3, -128, 127))
        runs.append(((codes - 3) * np.float32(0.0123) / _C716).astype(np.float32))
    v = qk_values.judge(runs)
    assert v.grid is True and v.test == "element lattice across inputs", v


def test_a_value_a_rounding_apart_is_one_lattice_point():
    """Rows of int8 codes times the row's step, where two elements that hold
    one value in eager torch come out one ulp apart (the partitioner's
    program can compute two equal products so): one point of the lattice,
    not a step of one ulp that breaks it."""
    g = np.random.default_rng(7)
    runs = []
    for _ in range(12):
        x = g.standard_normal((7, 16)).astype(np.float32)
        s = np.abs(x).max(-1, keepdims=True) / 127
        r = (np.round(x / s) * s).astype(np.float32)
        r[:, 15] = np.nextafter(r[:, 3], np.float32(np.inf))   # r[3], one ulp up
        runs.append(r)
    v = qk_values.judge(runs)
    assert v.grid is True and v.test == "row lattice", v


def _spread(x):
    # a spread group scale: rows of four values, each written
    # four times, near 7 (cat([amax_group / 3 + 7] * 4))
    a = np.abs(x.reshape(7, 4, 4)).max(-1) / 3.0 + 7.0
    return np.concatenate([a] * 4, -1)


def test_few_distinct_values_are_no_wide_lattice():
    """Four distinct values in a row of sixteen lie on some 4,096-step
    lattice by chance (held to 4,096 steps, this spread passes as a row
    lattice on six runs, though not on the gate's twelve): a group of fewer
    than eight distinct values is held to 255 steps, whatever its size."""
    g = np.random.default_rng(5)
    runs = [_spread(g.standard_normal((7, 16))).astype(np.float32) for _ in range(6)]
    v = qk_values.judge(runs)
    assert v.grid is False, v


def test_half_windows_find_a_ceiling_grid():
    """Sixteen levels, and one rare level more in every later run: the
    sixteen-level ceiling holds in the first six runs and not in all twelve,
    so the runs are judged whole and in halves (judge_runs)."""
    g = np.random.default_rng(6)
    levels = g.standard_normal(16).astype(np.float32)
    runs = []
    for i in range(12):
        pick = levels[g.integers(0, 16, (7, 2))]
        rows = np.repeat(pick, 2, -1)                  # [a, a, b, b]: d = 4
        if i >= 6:
            rows[0, 0] = 10.0 + i                      # a level of its own
        runs.append(rows.astype(np.float32))
    assert qk_values.judge(runs).grid is False
    v, n = qk_values.judge_runs(runs)
    assert v.grid is True and v.test == "levels" and n == 6, (v, n)


def test_quantised_amplitude_is_no_scaled_constant():
    """A scaled constant -- every row one vector up to its
    own scale -- is judged by its row amplitudes too: called plain whatever
    its amplitude, exp(round(x[..., :1])) expanded to the row (sixteen
    copies of a rounded level) would be admitted, 40.063 against the
    portable 40.0."""
    g = np.random.default_rng(3)
    grid = [np.exp(np.round(g.standard_normal((7, 1)))).repeat(16, 1)
            .astype(np.float32) for _ in range(12)]
    v = qk_values.judge(grid)
    assert v.grid is True and v.test == "scaled constant", v
    one_row = [np.exp(np.round(g.standard_normal((1, 1)))).repeat(16, 1)
               .astype(np.float32) for _ in range(12)]
    assert qk_values.judge(one_row).grid is True
    # a constant direction times a plain run-time norm stays plain
    plain = [(_C16 * np.linalg.norm(g.standard_normal((7, 16)), axis=-1,
                                    keepdims=True)).astype(np.float32)
             for _ in range(12)]
    v = qk_values.judge(plain)
    assert v.grid is False and v.test == "scaled constant", v
    # ... and so does one run-time scale per run, every row of a run the
    # same amplitude (a table times amax(|x|) at constant codes): one level
    # a run is no grid
    per_run = [(_C16 * np.abs(g.standard_normal((7, 16))).max()).astype(
        np.float32) * np.ones((7, 1), np.float32) for _ in range(12)]
    v = qk_values.judge(per_run)
    assert v.grid is False and v.test == "scaled constant", v
    assert qk_values.judge(per_run[:6]).grid is False
    # ... and three amplitudes, each its own run's (an expanded one-row state
    # buffer: test_qk_matrix's plain@alias-state@expand mqa m1 n1): three
    # values lie on some 255-step lattice by chance, so one step needs
    # SMALL_GROUP distinct amplitudes
    assert qk_values._amplitude_grid(np.array([0.1, 0.1, 0.3, 0.3, 0.7, 0.7]),
                                     np.array([0, 0, 1, 1, 2, 2])) is None
    # ... while a state the forward halves each call has run amplitudes
    # a * 2**-k, on one step whatever the inputs: a grid by values, and the
    # value check, which cannot tell such state from a fresh value, keeps a
    # q@kT that reads it portable (ValueCheck.state_read: speed only)
    decay = [np.repeat((row_ := _C16.reshape(1, 16) * 1.23) * 0.5 ** k, 4, 0)
             .astype(np.float32) for k in range(12)]
    assert qk_values.judge(decay).grid is True


def test_zero_amplitudes_are_a_level():
    """A zero row has no direction, so the scaled
    constant's pattern leaves it out; its amplitude is zero, one of the
    levels. Were the amplitudes to leave it out too, relu(sign(t)) expanded
    to the row (rows of zeros and ones) would give _amplitude_grid only
    ones and pass as plain, deterministically: admitted, 40.063 against the
    portable 40.0. Zeros beside a plain amplitude (relu(t)) stay plain."""
    g = np.random.default_rng(1729)
    grid, plain = [], []
    for _ in range(12):
        t = g.standard_normal((7, 1)).astype(np.float32)
        grid.append(np.maximum(t / np.maximum(t, -t), 0).repeat(16, 1))
        plain.append(np.maximum(t, 0).repeat(16, 1))
    v = qk_values.judge(grid)
    assert v.grid is True and v.test == "scaled constant", v
    assert "amplitudes on 2 levels" in v.detail, v
    assert qk_values.judge(grid[:6]).grid is True
    # zeros and a constant direction, not all ones
    assert qk_values.judge([r * _C16 for r in grid]).grid is True
    v = qk_values.judge(plain)
    assert v.grid is False and v.test == "scaled constant", v


@pytest.mark.parametrize("codes", [(0, 1, 2, 255), (0, 2, 9, 255), (0, 3, 99, 255)])
def test_lattice_candidates_at_the_boundary(codes):
    """A grid whose codes span exactly 255 steps measures a
    hair over 255 after a division, and a candidate bound taken from that
    span drops the true step -- dmin itself when the smallest gap is one
    code, dmin / 2 or dmin / 3 when it is several (forcing dmin alone does
    not cover those)."""
    u = (np.array(codes, np.float32) * np.float32(0.07)).astype(np.float64)
    assert qk_values._lattice_group(u, 255)
    g = np.random.default_rng(3)
    steps = g.uniform(0.05, 0.25, (7, 16)).astype(np.float32)
    runs = []
    for block in ([*codes, 63, 150], [*codes, 84, 200]):
        b = np.empty((6, 7, 16), np.float32)
        for r, c in np.ndindex(7, 16):
            b[:, r, c] = g.permutation(np.array(block, np.float32)) * steps[r, c]
        runs.extend(b)
    assert [qk_values.judge(w).grid for w in (runs, runs[:6], runs[6:])] == \
        [True, True, True]


def test_populated_cache_rows_are_no_codebook():
    """A cache holding ordinary rows from earlier calls, two
    rows overwritten each call: the unchanged rows come back in every run,
    which vector levels took for a codebook. They are the same in every run
    and the redraw moves them: unwritten, left out."""
    g = np.random.default_rng(0)
    base = g.standard_normal((32, 16)).astype(np.float32)
    runs = []
    for _ in range(12):
        a = base.copy()
        a[3:5] = g.standard_normal((2, 16)).astype(np.float32)
        runs.append(a)
    v = _judge_classed(runs, _redrawn(runs, [3, 4]))
    assert v.grid is False, v


class _FakeProgram:
    """Enough of an ExportedProgram for ValueCheck.verdict on given runs."""

    class graph_signature:            # noqa: N801
        inputs_to_buffers = {}
        buffers_to_mutate = {}


class _FakeOperand:
    name, op, all_input_nodes = "q", "call_function", []


def test_half_windows_through_the_value_check():
    """The halves pinned through ValueCheck itself -- two
    halves of six runs with sixteen levels each, thirty-two in all: levels in
    each half and not in the whole, so the verdict is levels on six runs."""
    g = np.random.default_rng(9)
    runs = []
    for _ in range(2):
        levels = g.standard_normal(16).astype(np.float32)
        runs.extend(levels[g.integers(0, 16, (6, 7, 16))])
    assert qk_values.judge(runs).grid is False
    ep = _FakeProgram()
    check = qk_values.ValueCheck(ep)
    check.samples = {"q": [(qk_values.FLOATS_VARY,
                            [(np.arange(7), r, 7) for r in runs])]}
    got = check.grid(_FakeOperand())
    assert got is not None and got[0] == "levels" and got[3] == 6, got


def _qdq_np(x, s=np.float32(0.0123), zp=3):
    c = np.clip(np.round(x / s) + zp, -128, 127)
    return ((c - zp) * s).astype(np.float32)


def test_full_range_int8_codes_per_element_are_a_grid():
    """A static int8 pair in numpy whose codes reach both
    clamps, over a per-element divisor that holds still across the
    runs."""
    g = np.random.default_rng(0)
    den = np.abs(g.standard_normal((7, 16))).astype(np.float32) + 0.05
    runs = [(_qdq_np(g.standard_normal((7, 16)).astype(np.float32)) / den)
            .astype(np.float32) for _ in range(12)]
    assert qk_values.judge(runs).grid is True


def test_asymmetric_int8_groups_of_four_are_a_grid():
    """Asymmetric int8 in groups of four channels (a zero
    point, codes over the full range) is a group lattice."""
    g = np.random.default_rng(0)

    def asym(x, gs=4):
        r = x.reshape(*x.shape[:-1], -1, gs)
        lo, hi = r.min(-1, keepdims=True), r.max(-1, keepdims=True)
        s_ = ((hi - lo) / 255).astype(np.float32)
        zp = np.round(-128 - lo / s_)
        c = np.clip(np.round(r / s_) + zp, -128, 127)
        return ((c - zp) * s_).reshape(x.shape).astype(np.float32)

    runs = [asym(g.standard_normal((7, 16)).astype(np.float32)) for _ in range(12)]
    assert qk_values.judge(runs).grid is True


def test_single_full_range_groups_are_lattices():
    """2,000 single groups spanning the full int8 range, each
    a lattice (a step search bounded by the measured span misses 830)."""
    g = np.random.default_rng(1)
    misses = 0
    for _ in range(2000):
        s_ = np.float32(g.uniform(0.001, 0.1))
        codes = np.concatenate([[-128, 127], g.integers(-128, 128, 10)])
        u = np.unique((codes.astype(np.float32) * s_).astype(np.float64))
        misses += not qk_values._lattice_group(u, 255)
    assert misses == 0, misses


def test_disabled_test_is_not_tried():
    f, shape, test, *_ = JUDGE_CASES["int8_rows"]
    qk_values._DISABLED.add(test)
    try:
        assert qk_values.judge(_runs(f, shape)).test != test
    finally:
        qk_values._DISABLED.discard(test)


# --- the gate -------------------------------------------------------------------

D = 16


def _qdq(x):
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    qd = torch.ops.quantized_decomposed
    return qd.dequantize_per_tensor.default(
        qd.quantize_per_tensor.default(x, 0.0123, 3, -128, 127, torch.int8),
        0.0123, 3, -128, 127, torch.int8)


class Head(nn.Module):
    """softmax(f(q) @ k^T / 4): the query from `fq(self, xq, ids)`."""

    def __init__(self, fq, rows=7, vocab=16):
        super().__init__()
        self.fq = fq
        g = torch.Generator().manual_seed(5)
        self.c = nn.Parameter(torch.rand(D, generator=g) * 0.2 + 0.05)
        self.c3 = nn.Parameter(torch.rand(1, rows, D, generator=g) * 0.2 + 0.05)
        self.register_buffer("levels", torch.linspace(-1.0, 1.0, 16))
        self.register_buffer("table", torch.randn(vocab, D, generator=g))
        self.register_buffer("state", torch.randn(1, rows, D, generator=g))
        self.w = nn.Parameter(torch.randn(D, D, generator=g) / 4)

    def forward(self, xq, xk, ids):
        q = self.fq(self, xq, ids)
        return torch.softmax(torch.bmm(q, xk.transpose(1, 2)) * 0.25, -1)


class StateHead(nn.Module):
    """The query is a buffer the forward writes after the q@kT (a write
    before the next call): each call's operand is the previous call's
    grid, a nearest-level lookup."""

    def __init__(self):
        super().__init__()
        self.register_buffer("levels", torch.tensor([-1.0, -0.3333, 0.3333, 1.0]))
        self.register_buffer("state", torch.randn(1, 7, D))

    def forward(self, xq, xk, ids):
        q = self.state
        out = torch.softmax(torch.bmm(q, xk.transpose(1, 2)) * 0.25, -1)
        self.state.copy_(self.levels[(xq.unsqueeze(-1) - self.levels).abs().argmin(-1)])
        return out


# Grids the graph rules alone admit: one per kind of hole in the rules,
# the residual case, and grids behind an op that computes.
RULES_MISSED = {
    "one_row_channel_scale": (lambda m, x, ids: x.to(torch.int8).float() * m.c, 1),
    "sign_by_maximum": (lambda m, x, ids: x / torch.maximum(x, -x), 7),
    "table_scaled_then_take": (lambda m, x, ids: torch.take(
        m.levels * x.abs().amax(),
        (x.unsqueeze(-1) - m.levels * x.abs().amax()).abs().argmin(-1)), 7),
    "sign_rolled_both_sides": (
        lambda m, x, ids: x.roll(3, -1) / torch.abs(x).roll(3, -1), 7),
    "ratio_one_plus_sign": (
        lambda m, x, ids: torch.addcdiv(x / x, x, torch.abs(x)), 7),
    "sign_big_clamp_guard": (
        lambda m, x, ids: x / torch.clamp(torch.abs(x), 1e-8, 1e30), 7),
    "row_at_token_ids": (lambda m, x, ids: torch.ceil(x) + torch.nn.functional.embedding(
        ids[..., :1], m.table), 1),
    "rsqrt_big_clamp_guard": (lambda m, x, ids: x * torch.rsqrt(
        torch.clamp(x * x, 1e-7, 1e30)), 7),
    "sign_split_cat": (lambda m, x, ids: torch.cat(torch.split(x, 8, -1), -1)
                       / torch.sqrt(x * x), 7),
    "pow_big_clamp_guard": (lambda m, x, ids: x / torch.clamp(
        torch.pow(x * x, 0.5), 1e-12, 1e30), 7),
    "exp_of_round": (lambda m, x, ids: torch.exp(torch.round(x)), 7),
    "saturating_clamp": (lambda m, x, ids: torch.clamp(64 * x, -1.0, 1.0), 7),
    "magic_round_unequal": (lambda m, x, ids: (4 * x + 1.5 * 2 ** 23)
                            - (1.5 * 2 ** 23 - 2), 7),
    "one_hot_matmul_lookup": (lambda m, x, ids: torch.nn.functional.one_hot(
        (x.unsqueeze(-1) - m.levels).abs().argmin(-1), 16).float() @ m.levels, 7),
    # A grid over a factor of the
    # operand's own shape that holds still from run to run (a table row at
    # the token ids, a constant), full int8 codes (the rules take the
    # quotient for new values; each element is one step across the runs).
    "static_pair_over_table_row": (lambda m, x, ids: _qdq(x) / torch.maximum(
        m.c, torch.abs(torch.nn.functional.embedding(ids, m.table))), 7),
    "static_pair_over_constant": (lambda m, x, ids: _qdq(x) / m.c3, 7),
    "int8_codes_times_constant": (lambda m, x, ids: torch.clamp(
        torch.round(x / 0.0123), -128, 127) * m.c3, 7),
    # A rounded amplitude expanded to the row (a scaled
    # constant whose scale is a grid); 40.063 against 40.0 when admitted
    "rounded_amplitude_expanded": (lambda m, x, ids: torch.exp(
        torch.round(x[..., :1])).expand_as(x), 7),
    # Zero and one amplitudes (a zero row has no
    # direction, so no amplitude): relu(sign) expanded to the
    # row, 40.063 against 40.0 when admitted; and zeros or a constant row
    "zero_one_amplitude_expanded": (lambda m, x, ids: torch.relu(
        x[..., :1] / torch.maximum(x[..., :1], -x[..., :1])).expand_as(x), 7),
    "zero_or_constant_row": (lambda m, x, ids: torch.relu(
        x[..., :1] / torch.maximum(x[..., :1], -x[..., :1])) * m.c, 7),
}

class KeyHead(nn.Module):
    """softmax(q @ f(k)^T / 4): the key from `fk(self, xk)`, read by
    OP_BMM_DYN_I8 (quantised by the delegate, so judged by values)."""

    def __init__(self, fk):
        super().__init__()
        self.fk = fk
        g = torch.Generator().manual_seed(6)
        self.c = nn.Parameter(torch.rand(D, generator=g) * 0.2 + 0.05)
        self.c3 = nn.Parameter(torch.rand(1, 31, D, generator=g) * 0.2 + 0.05)

    def forward(self, xq, xk, ids):
        k = self.fk(self, xk)
        return torch.softmax(torch.bmm(xq, k.transpose(1, 2)) * 0.25, -1)


# Grids on the key side, which the rules admit; int8 codes
# per element (the per-channel step is a scale the rules read), a sign the
# rules miss, and a rounded amplitude.
KEY_GRIDS = {
    "key_int8_codes_times_constant": lambda m, k: torch.clamp(
        torch.round(k / 0.0123), -128, 127) * m.c3,
    "key_sign_by_maximum": lambda m, k: k / torch.maximum(k, -k),
    "key_rounded_amplitude_expanded": lambda m, k: torch.exp(
        torch.round(k[..., :1])).expand_as(k),
    "key_zero_one_amplitude_expanded": lambda m, k: torch.relu(
        k[..., :1] / torch.maximum(k[..., :1], -k[..., :1])).expand_as(k),
}

PLAIN_SPELLINGS = {
    "plain": (lambda m, x, ids: x * 2.0 + 1.0, 7),
    "projection": (lambda m, x, ids: x @ m.w, 7),
    "rmsnorm": (lambda m, x, ids: x * torch.rsqrt((x * x).mean(-1, keepdim=True)
                                                  + 1e-6) * m.c, 7),
    "rope_like": (lambda m, x, ids: x * torch.cos(m.c) + x.roll(8, -1) * torch.sin(m.c), 7),
    "one_row_projection": (lambda m, x, ids: torch.relu(x @ m.w) + m.c, 1),
    "with_token_rows": (lambda m, x, ids: x + torch.nn.functional.embedding(
        ids, m.table), 7),
}


def _inputs(rows, zeros=False):
    g = torch.Generator().manual_seed(11)
    xq = torch.zeros(1, rows, D) if zeros else torch.randn(1, rows, D, generator=g)
    xk = torch.zeros(1, 31, D) if zeros else torch.randn(1, 31, D, generator=g)
    ids = torch.zeros(1, rows, dtype=torch.long) if zeros else torch.randint(
        0, 16, (1, rows), generator=g)
    return xq, xk, ids


def _lower(model, args, example=True):
    """(portable bmms left, partition log, partitioner) for `model` lowered
    under quantize_qk; `example=False` hides the example inputs."""
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(model.eval(), args)
    if not example:
        ep.example_inputs = None
    part = Esp32Partitioner(quantize_qk=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        edge = lower_esp32(ep, [part])
    gm = edge.exported_program().graph_module
    bmms = sum(1 for n in gm.graph.nodes if n.op == "call_function"
               and "aten.bmm" in str(n.target))
    return bmms, buf.getvalue(), part


def _rules(model, args, side=0):
    """The rules' origin kind of the query (side 0) or key (2), at
    partition time."""
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        _qk_bmm_pattern,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    got = []
    orig = Esp32Partitioner.partition

    def spy(self, program):
        for n in program.graph_module.graph.nodes:
            pat = _qk_bmm_pattern(n) if n.op == "call_function" else None
            if pat is not None:
                got.append(qk_operands.origin(pat[side], program).kind)
        return orig(self, program)

    Esp32Partitioner.partition = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lower_esp32(torch.export.export(model.eval(), args),
                        [Esp32Partitioner(quantize_qk=True)])
    finally:
        Esp32Partitioner.partition = orig
    return got


@pytest.mark.parametrize("case", sorted(RULES_MISSED) + ["grid_in_written_state"])
def test_rules_missed_grids_stay_portable_by_values(case):
    if case == "grid_in_written_state":
        make, rows = StateHead, 7
    else:
        fq, rows = RULES_MISSED[case]
        make = lambda: Head(fq, rows)  # noqa: E731
    args = _inputs(rows)
    assert _rules(make(), args) == [qk_operands.PLAIN]
    bmms, log, part = _lower(make(), args)
    assert bmms == 1, log
    assert part.support.kinds[KIND_QK_VALUES] >= 1, dict(part.support.kinds)
    assert "left portable by its operand values" in log, log


@pytest.mark.parametrize("case", sorted(KEY_GRIDS))
def test_key_side_grids_stay_portable_by_values(case):
    """The key OP_BMM_DYN_I8 quantises itself is judged too
    (the value check records got[2]): a grid on it keeps the q@kT portable
    by its values, and the log names the key."""
    make = lambda: KeyHead(KEY_GRIDS[case])  # noqa: E731
    args = _inputs(7)
    assert _rules(make(), args, side=2) == [qk_operands.PLAIN]
    bmms, log, part = _lower(make(), args)
    assert bmms == 1, log
    assert part.support.kinds[KIND_QK_VALUES] >= 1
    assert "the key, is on a grid" in log, log


class IdsLm(nn.Module):
    """softmax(f(h Wq) (h Wk)^T / 4) with h = emb(ids) --
    an LLM's shape, token ids and no float input."""

    def __init__(self, fq, vocab=1000):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.emb = nn.Embedding(vocab, D)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, D, generator=g))
        self.wq = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.wk = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.register_buffer("levels", torch.linspace(-1.0, 1.0, 16))
        self.register_buffer("nf4", torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
            0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]))
        self.fq = fq

    def forward(self, ids):
        h = self.emb(ids)
        q = self.fq(self, h @ self.wq)
        return torch.softmax(torch.bmm(q, (h @ self.wk).transpose(1, 2)) * 0.25, -1)


IDS_GRIDS = {
    "exp_of_round": lambda m, q: torch.exp(torch.round(q)),
    "one_hot_nf4": lambda m, q: torch.nn.functional.one_hot(
        (q.unsqueeze(-1) - m.nf4).abs().argmin(-1), 16).float() @ m.nf4,
    "nf4_take": lambda m, q: torch.take(
        m.nf4 * q.abs().amax(),
        (q.unsqueeze(-1) / q.abs().amax() - m.nf4).abs().argmin(-1)),
    "saturating_clamp": lambda m, q: torch.clamp(64 * q, -1.0, 1.0),
}


@pytest.mark.parametrize("case", sorted(IDS_GRIDS))
def test_token_id_only_grids_stay_portable(case):
    """A program with no float input has no float regime,
    and the token-id regime leaves out the counting tests, so these would
    reach the delegate. The rows looked up at the token ids are drawn unit
    normal in a regime of their own (EMBEDDINGS_VARY), where every test
    runs."""
    ids = (torch.randint(0, 1000, (1, 7), generator=torch.Generator().manual_seed(1)),)
    bmms, log, part = _lower(IdsLm(IDS_GRIDS[case]), ids)
    assert bmms == 1, log
    assert part.support.kinds[KIND_QK_VALUES] >= 1


def test_token_id_only_plain_stays_delegated():
    ids = (torch.randint(0, 1000, (1, 7), generator=torch.Generator().manual_seed(1)),)
    bmms, log, part = _lower(IdsLm(lambda m, q: q), ids)
    assert bmms == 0, log


class TwoLayer(nn.Module):
    """Layer 1 attends over a K/V cache written at
    pos (cached) or over its own k, v (uncached); layer 2 is the same in
    both -- its key a K cache written at pos, its query fq(h @ wq2),
    computed fresh from this call's inputs (positions are held, so every run
    reads the rows it just wrote). `seed` draws the weights; `causal` masks
    the cache rows past pos, as ExecuTorch's llama and the story do."""

    def __init__(self, fq, cached, T=16, seed=12, causal=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq1, self.wk1, self.wv1, self.wo1 = w(), w(), w(), w()
        self.wq2, self.wk2 = w(), w()
        self.register_buffer("kc1", torch.zeros(1, T, D))
        self.register_buffer("vc1", torch.zeros(1, T, D))
        self.register_buffer("kc2", torch.zeros(1, T, D))
        self.register_buffer("book", torch.randn(8, D, generator=g))
        self.c = nn.Parameter(torch.rand(D, generator=g) * 0.2 + 0.05)
        self.cached, self.fq, self.causal = cached, fq, causal

    def forward(self, x, pos):
        q1, k1, v1 = x @ self.wq1, x @ self.wk1, x @ self.wv1
        if self.cached:
            self.kc1.index_copy_(1, pos, k1)
            self.vc1.index_copy_(1, pos, v1)
            K, V = self.kc1, self.vc1
        else:
            K, V = k1, v1
        s = torch.bmm(q1, K.transpose(1, 2)) * 0.25
        if self.causal:
            valid = (torch.arange(K.shape[1]) <= pos.reshape(-1, 1)).unsqueeze(0)
            s = torch.where(valid, s, torch.full((), float("-inf")))
        p = torch.softmax(s, -1)
        h = x + (p @ V) @ self.wo1
        self.kc2.index_copy_(1, pos, h @ self.wk2)
        q2 = self.fq(self, h @ self.wq2)
        s2 = torch.bmm(q2, self.kc2.transpose(1, 2)) * 0.25
        if self.causal:
            s2 = torch.where(valid, s2, torch.full((), float("-inf")))
        return torch.softmax(s2, -1)


LAYER_TWO_GRIDS = {
    # the README's own exp(round(x[..., :1])) expanded to the row
    "rounded_amplitude_expanded": lambda m, y: torch.exp(
        torch.round(y[..., :1])).expand_as(y),
    "rounded_amplitude_times_c": lambda m, y: torch.exp(
        torch.round(y[..., :1])) * m.c,
    # a codebook of eight rows (vector levels)
    "codebook_rows": lambda m, y: torch.exp(torch.nn.functional.one_hot(
        ((y.unsqueeze(-2) - m.book) ** 2).sum(-1).argmin(-1), 8).float()
        @ m.book),
}

_TWO_ARGS = (torch.randn(1, 7, D, generator=torch.Generator().manual_seed(1)),
             torch.arange(7))
_TWO_ARGS_DECODE = (torch.randn(1, 1, D, generator=torch.Generator().manual_seed(1)),
                    torch.arange(1))


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("case", sorted(LAYER_TWO_GRIDS))
def test_layer_two_query_grid_stays_portable(case, cached):
    """Layer 2's query is a grid computed fresh every call, with or without
    a cache in layer 1. A check that walks every input back to a buffer
    calls the query written and drops the tests that find it (constant
    rows, vector levels, amplitudes); rows are classed from values: the query is
    computed this call, with cache rows no run wrote among what layer 1
    reads, so the redraw moves it -- but the replay (run 0's inputs on the
    state that enters run 1) leaves it in place: no row is carried, and
    every row is judged with every test. Two bmms
    left: layer 1's p @ V and layer 2's q@kT."""
    seen = []
    orig = qk_values.ValueCheck._read

    def spy(self):
        seen.append(self)
        return orig(self)

    qk_values.ValueCheck._read = spy
    try:
        bmms, log, _ = _lower(TwoLayer(LAYER_TWO_GRIDS[case], cached),
                              _TWO_ARGS)
    finally:
        qk_values.ValueCheck._read = orig
    assert bmms == 2, log
    assert "left portable by its operand values" in log, log
    carried = [c for v in seen[0].row_counts.values() for _, _, c, _ in v]
    assert not any(carried), seen[0].row_counts


def _layer_two_delegated(cases):
    """The (grid, seed) cases whose layer-2 q@kT a cached TwoLayer delegates."""
    out = []
    for grid, seed, args, causal in cases:
        bmms, log, _ = _lower(TwoLayer(LAYER_TWO_GRIDS[grid], True, seed=seed,
                                       causal=causal), args)
        if bmms != 2:
            out.append((grid, seed))
    return out


def test_layer_two_query_grid_over_weight_seeds_at_a_decode_step():
    """The redraw moves a query computed from cache
    rows no run wrote; a carried class that takes it delegates the layer-2
    q@kT at 7 of 16 weight seeds where the uncached twin keeps all 16
    portable. Carried is what the replay changes: none of the 16."""
    cases = [(g, s, _TWO_ARGS_DECODE, False) for s in range(16)
             for g in ("rounded_amplitude_expanded", "rounded_amplitude_times_c")]
    assert _layer_two_delegated(cases) == []


def test_layer_two_query_grid_over_weight_seeds_at_a_masked_position():
    """The same with a causal mask, decoding at position 6: layer 1 reads
    the six cache rows before it, which no run wrote."""
    args = (_TWO_ARGS_DECODE[0], torch.tensor([6]))
    cases = [(g, s, args, True) for s in range(16)
             for g in ("rounded_amplitude_expanded", "rounded_amplitude_times_c")]
    assert _layer_two_delegated(cases) == []


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("case", ["rounded_amplitude_expanded",
                                  "rounded_amplitude_times_c"])
def test_layer_two_decode_query_grid_stays_portable(case, cached):
    """The same at a decode step: one query row, a one-row scaled constant
    whose amplitude is rounded."""
    bmms, log, _ = _lower(TwoLayer(LAYER_TWO_GRIDS[case], cached),
                          _TWO_ARGS_DECODE)
    assert bmms == 2, log


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
def test_layer_two_plain_query_stays_delegated(cached):
    bmms, log, _ = _lower(TwoLayer(lambda m, y: torch.exp(0.1 * y), cached),
                          _TWO_ARGS)
    assert bmms == 1, log


def test_layer_two_query_is_not_carried():
    """The cached model's layer-2 query on the same inputs, with the caches
    carried from a previous call and with them reset: the same values. No
    buffer carries it from run to run."""
    m = TwoLayer(LAYER_TWO_GRIDS["codebook_rows"], True).eval()
    seen = []
    m.fq = lambda mod, y: seen.append(LAYER_TWO_GRIDS["codebook_rows"](mod, y)) \
        or seen[-1]
    x2 = torch.randn(1, 7, D, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        m(*_TWO_ARGS)
        m(x2, _TWO_ARGS[1])
        for b in (m.kc1, m.vc1, m.kc2):
            b.zero_()
        m(x2, _TWO_ARGS[1])
    assert torch.equal(seen[1], seen[2])


@pytest.mark.parametrize("tokens,start", [(1, 0), (4, 0), (1, 5), (4, 3)])
def test_stock_llama_queries_are_no_written_buffer(tokens, start):
    """On ExecuTorch's own llama (construct_transformer,
    a KV cache, two layers): layer 2's query depends on layer 1's caches.
    Classed from values, no row of either query is unwritten or carried --
    past position 0 layer 1 attends over cache rows no run wrote, so the
    redraw moves the layer-2 query, but the replay does not change it, so
    none is carried. The keys'
    unwritten rows, the cache rows no run wrote, are left out and their
    written rows judged; every query and key is plain."""
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        _qk_bmm_pattern,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.fixtures import stock_llm as esl
    from executorch.examples.models.llama.llama_transformer import (
        construct_transformer,
    )
    from executorch.examples.models.llama.model_args import ModelArgs

    esl._stock_ops()
    margs = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      vocab_size=256, hidden_dim=128, max_seq_len=16,
                      max_context_len=16, use_kv_cache=True,
                      enable_dynamic_shape=False, max_batch_size=1)
    torch.manual_seed(0)
    model = construct_transformer(margs).eval().to(dtype=torch.float32)
    assert sum(p.numel() for p in model.parameters()) < 200_000
    inputs = (torch.randint(0, 256, (1, tokens),
                            generator=torch.Generator().manual_seed(1)),
              {"input_pos": torch.arange(start, start + tokens,
                                         dtype=torch.int64)})
    ep = esl.llm_export(model, inputs)
    pairs, verdicts, checks = [], {}, []
    orig_part = Esp32Partitioner.partition
    orig_verdict = qk_values.ValueCheck.verdict
    orig_read = qk_values.ValueCheck._read

    def spy_part(self, program):
        for n in program.graph_module.graph.nodes:
            pat = _qk_bmm_pattern(n) if n.op == "call_function" else None
            if pat is not None:
                pairs.append((pat[0].name, pat[2].name))
        return orig_part(self, program)

    def spy_verdict(self, operand):
        got = orig_verdict(self, operand)
        verdicts[operand.name] = got
        return got

    def spy_read(self):
        checks.append(self)
        return orig_read(self)

    Esp32Partitioner.partition = spy_part
    qk_values.ValueCheck.verdict = spy_verdict
    qk_values.ValueCheck._read = spy_read
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            lower_esp32(ep, [Esp32Partitioner(quantize_qk=True)])
    finally:
        Esp32Partitioner.partition = orig_part
        qk_values.ValueCheck.verdict = orig_verdict
        qk_values.ValueCheck._read = orig_read
    log = buf.getvalue()
    # every q@kT delegated: no value refusal of any kind
    for why in ("left portable by its operand values", "was not judged",
                "reads state an earlier call wrote"):
        assert why not in log, log
    assert len(pairs) == 2, pairs
    counts = checks[0].row_counts
    for q, k in pairs:
        assert all(c == 0 and u == 0 for _, _, c, u in counts.get(q, [])), \
            counts.get(q)
        assert counts.get(k) and all(u > 0 and f > 0
                                     for _, f, _, u in counts[k]), counts.get(k)
        for name in (q, k):
            got = verdicts.get(name)
            assert got is not None and got[0].grid is False, (name, got)


def _zero_one(x):
    """relu(sign(t)) expanded to the row, t = x[..., :1]: rows of zeros and
    ones, a grid the rules miss."""
    t = x[..., :1]
    return torch.relu(t / torch.maximum(t, -t)).expand_as(x)


class CatSliceState(nn.Module):
    """cat([state, fresh], 1)[:, 7:] keeps no row of
    the state -- a buffer the forward writes after the q@kT -- only the
    fresh zero/one rows, on the query or on the key."""

    def __init__(self, side):
        super().__init__()
        self.side = side
        self.register_buffer("state", torch.zeros(1, 7, D))

    def forward(self, xq, xk, ids):
        x = xq if self.side == "q" else xk
        y = torch.cat([self.state, _zero_one(x)], 1)[:, 7:]
        q, k = (y, xk) if self.side == "q" else (xq, y)
        out = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)
        self.state.copy_(x[:, :7])
        return out


class CopyIntoState(nn.Module):
    """copy(state, fresh) holds fresh's values; its
    destination, a buffer the forward writes, supplies none of them."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 7, D))

    def forward(self, xq, xk, ids):
        q = torch.ops.aten.copy.default(self.state, _zero_one(xq))
        out = torch.softmax(torch.bmm(q, xk.transpose(1, 2)) * 0.25, -1)
        self.state.copy_(xq)
        return out


class GridIntoCache(nn.Module):
    """A fresh grid written into a K cache by a functional write (index_copy
    or index_put), the key the whole cache: the rows written are fresh,
    judged with every test; the rows never written are left out."""

    def __init__(self, write):
        super().__init__()
        self.write = write
        self.register_buffer("kc", torch.zeros(1, 31, D))

    def forward(self, xq, xk, pos):
        g = _zero_one(xk)
        if self.write == "index_copy":
            self.kc.index_copy_(1, pos, g)
        else:
            self.kc[:, pos] = g
        return torch.softmax(torch.bmm(xq, self.kc.transpose(1, 2)) * 0.25, -1)


class CacheAndOneHot(nn.Module):
    """K = cat([cache, one_hot(pos)]) with pos held: the one-hot rows are the
    same in every run, but no buffer's initial content reaches them -- fresh,
    judged, not left out; only the cache rows never written are."""

    def __init__(self):
        super().__init__()
        self.register_buffer("kc", torch.zeros(1, 24, D))

    def forward(self, xq, xk, pos):
        self.kc.index_copy_(1, pos, _zero_one(xk))
        k = torch.cat([self.kc, torch.nn.functional.one_hot(pos, D).float()
                       .unsqueeze(0)], 1)
        return torch.softmax(torch.bmm(xq, k.transpose(1, 2)) * 0.25, -1)


def _qk_args(rows_k=7, seed=4):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 7, D, generator=g), torch.randn(1, rows_k, D, generator=g),
            torch.arange(rows_k))


@pytest.mark.parametrize("side", ["q", "k"])
def test_cat_then_slice_of_a_state_is_judged_fresh(side):
    """A state reached through cat, then a slice: a walk
    that calls the operand written leaves its fresh zero/one rows -- each
    all one value -- out as unwritten and delegates the bmm (40.063 for
    40.0). Classed from values, every row is fresh."""
    make = lambda: CatSliceState(side)  # noqa: E731
    args = _inputs(7)
    assert _rules(make(), args, side=0 if side == "q" else 2) == [qk_operands.PLAIN]
    bmms, log, _ = _lower(make(), args)
    assert bmms == 1, log
    assert "left portable by its operand values" in log, log


def test_copy_into_a_written_buffer_is_judged_fresh():
    """A copy into a written buffer: its values are its
    source's, not its destination's, so its rows are fresh."""
    args = _inputs(7)
    assert _rules(CopyIntoState(), args) == [qk_operands.PLAIN]
    bmms, log, _ = _lower(CopyIntoState(), args)
    assert bmms == 1, log
    assert "left portable by its operand values" in log, log


@pytest.mark.parametrize("write", ["index_copy", "index_put"])
def test_fresh_grid_written_into_a_cache_stays_portable(write):
    """A fresh grid a functional write puts in a K cache: its rows, each all
    one value, are judged, not left out as unwritten with the rows no run
    wrote."""
    bmms, log, _ = _lower(GridIntoCache(write), _qk_args())
    assert bmms == 1, log
    assert "the key, is on a grid" in log, log


def test_program_constant_rows_beside_a_cache_are_judged():
    """cat([cache, one_hot(pos)]), pos held: the one-hot rows are the same
    in every run, yet the redraw does not move them -- fresh, judged; the
    seventeen cache rows no run wrote are the only rows left out. (The rules
    refuse this key for its one-hot rows; the value check is asked
    directly.)"""
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        _qk_bmm_pattern,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    got = []
    orig = Esp32Partitioner.partition

    def spy(self, program):
        check = qk_values.ValueCheck(program, self.example_inputs)
        for n in program.graph_module.graph.nodes:
            pat = _qk_bmm_pattern(n) if n.op == "call_function" else None
            if pat is not None:
                got.append((check.verdict(pat[2]), check.row_counts.get(
                    pat[2].name)))
        return orig(self, program)

    Esp32Partitioner.partition = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ep = torch.export.export(CacheAndOneHot().eval(), _qk_args())
            lower_esp32(ep, [Esp32Partitioner(quantize_qk=True)])
    finally:
        Esp32Partitioner.partition = orig
    assert len(got) == 1, got
    verdict, counts = got[0]
    assert counts and all(u == 17 and f == 14 for _, f, _, u in counts), counts
    assert verdict is not None and verdict[0].grid is True, verdict


class ReadSideCache(nn.Module):
    """A K cache of T rows written at pos,
    the key the cache read through fk -- a sign, a rounding, RoPE or a
    projection applied on read. A decode step's one written row must be
    judged however long the cache: were such a key no written operand,
    past MAX_ROWS its rows would be thinned evenly and the written row
    skipped."""

    def __init__(self, fk, T):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.register_buffer("kc", torch.zeros(1, T, D))
        self.w = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.cos = nn.Parameter(torch.rand(T, D, generator=g))
        self.sin = nn.Parameter(torch.rand(T, D, generator=g))
        self.fk = fk

    def forward(self, xq, xk, pos):
        self.kc.index_copy_(1, pos, xk)
        k = self.fk(self, self.kc)
        return torch.softmax(torch.bmm(xq, k.transpose(1, 2)) * 0.25, -1)


def _rope_on_read(m, k):
    return k * m.cos + torch.roll(k, 1, -1) * m.sin


READ_SIDE_GRIDS = {
    # test_qk_values's LongCache with the sign moved from the write to the read
    "sign_by_maximum": (lambda m, k: k / torch.maximum(k, -k), (32, 2048)),
    "exp_round_2k": (lambda m, k: torch.exp(torch.round(2 * k)), (1024,)),
    "exp_round_kw": (lambda m, k: torch.exp(torch.round(k @ m.w)), (1024,)),
    "exp_round_rope": (lambda m, k: torch.exp(torch.round(
        _rope_on_read(m, k))), (1024,)),
}

READ_SIDE_PLAIN = {
    "cache": lambda m, k: k,
    "rope": _rope_on_read,
    "projection": lambda m, k: k @ m.w,
}


def _decode_args():
    g = torch.Generator().manual_seed(11)
    return (torch.randn(1, 1, D, generator=g), torch.randn(1, 1, D, generator=g),
            torch.tensor([5]))


@pytest.mark.parametrize("case,T", [(c, T) for c, (_, Ts) in
                                    sorted(READ_SIDE_GRIDS.items()) for T in Ts])
def test_read_side_grid_of_a_long_cache_stays_portable(case, T):
    bmms, log, _ = _lower(ReadSideCache(READ_SIDE_GRIDS[case][0], T),
                          _decode_args())
    assert bmms == 1, log
    assert "left portable by its operand values" in log, log


@pytest.mark.parametrize("case", sorted(READ_SIDE_PLAIN))
def test_read_side_plain_key_of_a_long_cache_stays_delegated(case):
    bmms, log, _ = _lower(ReadSideCache(READ_SIDE_PLAIN[case], 1024),
                          _decode_args())
    assert bmms == 0, log


def test_no_verdict_logs_judges_own_reason():
    """When judge gives no verdict on runs
    that were recorded, the log gives judge's reason, not 'no sample
    recorded' -- a key within 0.1% of one number is degenerate."""
    bmms, log, _ = _lower(KeyHead(lambda m, k: 1.0 + 1e-6 * k), _inputs(7))
    assert bmms == 0, log
    assert "the key: degenerate: all within 0.1% of" in log, log


class SelectorCache(nn.Module):
    """A populated K cache, rows 0-6 overwritten each
    call with fresh zero/one rows (a grid), and -- with `selector` -- a
    written float buffer of zeros, halved each call, that indexes a one-row
    table: valid in every run, invalid once the redraw draws it afresh, so
    the redrawn run raises."""

    def __init__(self, selector):
        super().__init__()
        g = torch.Generator().manual_seed(8)
        self.register_buffer("kc", torch.randn(1, 128, D, generator=g))
        self.register_buffer("sel", torch.zeros(32))
        self.register_buffer("table", torch.randn(1, D, generator=g))
        self.selector = selector

    def forward(self, xq, xk, pos):
        self.kc.index_copy_(1, pos, _zero_one(xk))
        out = torch.softmax(torch.bmm(xq, self.kc.transpose(1, 2)) * 0.25, -1)
        if not self.selector:
            return out
        extra = self.table.index_select(0, self.sel.long())
        self.sel.mul_(0.5)
        return out, extra


def test_failed_redraw_keeps_the_qk_portable():
    """When the redrawn run raises, the rows cannot be
    classed. Judged all as fresh -- the cache's 121 untouched rows with the 7
    fresh zero/one rows -- the untouched rows would hide the grid the
    classes find and the bmm would be delegated (40.063 for 40.0). Rows that
    cannot be classed are not judged: the q@kT stays portable, and the log
    says why."""
    bmms, log, _ = _lower(SelectorCache(True), _qk_args())
    assert bmms == 1, log
    assert "was not judged" in log and "drawn afresh raised" in log, log


def test_redraw_that_runs_finds_the_grid_among_untouched_rows():
    """The control: the same cache without the selector -- the redraw runs,
    the untouched rows are left out, and the fresh rows are a grid."""
    bmms, log, _ = _lower(SelectorCache(False), _qk_args())
    assert bmms == 1, log
    assert "left portable by its operand values" in log, log


class _StateTwoLayer(nn.Module):
    """TwoLayer's weights (drawn in its order from
    `seed`), layer 1 attending over a cache whose earlier writes a later
    call reads, layer 2 computed this call: its query
    exp(round((h @ wq2)[..., :1])) * c, a grid. `kind`:
      shiftpop  both layers over shifted caches that start populated;
      shift0    layer 1 over a shifted cache that starts at zeros;
      counter   layer 1 over a cache written at an int counter the forward
                advances (torchtune's KVCache), masked past it;
      cross     counter self-attention, then cross-attention over an
                encoder input's projection;
      mixed     layer 1 shifted; layer 2's key cat([pk2, exp(round(h @ wk2))])
                -- a cat a carrying walk passes, its new row computed
                from layer 1."""

    def __init__(self, kind, seed, T=15):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq1, self.wk1, self.wv1, self.wo1 = w(), w(), w(), w()
        self.wq2, self.wk2 = w(), w()
        self.register_buffer("book", torch.randn(8, D, generator=g))
        self.c = nn.Parameter(torch.rand(D, generator=g) * 0.2 + 0.05)
        p = torch.Generator().manual_seed(7)
        start = (lambda: torch.randn(1, T, D, generator=p)) if kind in (
            "shiftpop", "mixed") else (lambda: torch.zeros(1, T, D))
        self.register_buffer("pk1", start())
        self.register_buffer("pv1", start())
        self.register_buffer("pk2", start())
        self.register_buffer("kc1", torch.zeros(1, 96, D))
        self.register_buffer("vc1", torch.zeros(1, 96, D))
        self.register_buffer("cnt", torch.zeros(1, dtype=torch.long))
        self.kind = kind

    def forward(self, x, enc):
        q1, k1, v1 = x @ self.wq1, x @ self.wk1, x @ self.wv1
        n = x.shape[1]
        if self.kind in ("counter", "cross"):
            pos = self.cnt + torch.arange(n)
            self.kc1.index_copy_(1, pos, k1)
            self.vc1.index_copy_(1, pos, v1)
            valid = (torch.arange(96) <= pos.reshape(-1, 1)).unsqueeze(0)
            s = torch.bmm(q1, self.kc1.transpose(1, 2)) * 0.25
            s = torch.where(valid, s, torch.full((), float("-inf")))
            h = x + (torch.softmax(s, -1) @ self.vc1) @ self.wo1
            self.cnt.add_(n)
        else:
            K = torch.cat([self.pk1, k1], 1)
            V = torch.cat([self.pv1, v1], 1)
            p = torch.softmax(torch.bmm(q1, K.transpose(1, 2)) * 0.25, -1)
            h = x + (p @ V) @ self.wo1
            self.pk1.copy_(K[:, n:])
            self.pv1.copy_(V[:, n:])
        q2 = torch.exp(torch.round((h @ self.wq2)[..., :1])) * self.c
        if self.kind == "cross":
            k2 = enc @ self.wk2
        elif self.kind == "mixed":
            k2 = torch.cat([self.pk2, torch.exp(torch.round(h @ self.wk2))], 1)
            self.pk2.copy_(k2[:, n:])
        elif self.kind == "shiftpop":
            k2 = torch.cat([self.pk2, h @ self.wk2], 1)
            self.pk2.copy_(k2[:, n:])
        else:
            k2 = h @ self.wk2
        return torch.softmax(torch.bmm(q2, k2.transpose(1, 2)) * 0.25, -1)


@pytest.mark.parametrize("kind", ["shiftpop", "shift0", "counter", "cross",
                                  "mixed"])
def test_state_an_earlier_call_wrote_keeps_the_qk_portable(kind):
    """The replay changes every row that depends on
    state an earlier call wrote, not only rows that are that state -- layer
    2's query behind such a cache, computed this call, is such a row, and
    its values can lose the vector levels and amplitudes the tests look
    for: judged as a fresh value, it is delegated at some weight seeds
    (shiftpop 14; counter and cross 1, 4, 5, 11; shift0 6) where every
    uncached twin is portable. From values such state cannot be told from a
    fresh value,
    so a q@kT whose operands have a row the replay changes that is no row a
    written buffer held entering run 0 (a value computed from that state)
    stays portable -- at every seed, and for a mixed key whose new row a cat
    brings in. A buffer's own rows read through moves are judged instead,
    with every test (RingBesideFresh, CounterBesideFresh)."""
    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        _qk_bmm_pattern,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    g = torch.Generator().manual_seed(1)
    args = (torch.randn(1, 1, D, generator=g), torch.randn(1, 5, D, generator=g))
    delegated = []
    orig = Esp32Partitioner.partition
    for seed in range(16):
        names = []

        def spy(self, program, names=names):
            names[:] = [n.name for n in program.graph_module.graph.nodes
                        if n.op == "call_function" and _qk_bmm_pattern(n)]
            return orig(self, program)

        Esp32Partitioner.partition = spy
        try:
            bmms, log, _ = _lower(_StateTwoLayer(kind, seed), args)
        finally:
            Esp32Partitioner.partition = orig
        # layer 2's q@kT, the last in the graph: kept portable by values
        if not names or (f"q@kT {names[-1]} left portable by its operand "
                         f"values") not in log:
            delegated.append(seed)
    assert delegated == [], delegated


class RingBesideFresh(nn.Module):
    """K = cat([ring, k_new]), the ring a buffer
    the forward overwrites after its q@kT, so each call reads the rows the
    call before wrote -- rows the replay changes that are the buffer's own
    -- beside fresh rows. `grid`:
      new   k_new = exp(round(x @ wk)), a fresh grid; ring <- x @ wr, plain;
      ring  k_new = x @ wk, plain; ring <- exp(round(x @ wr)), a grid the
            call that wrote it never attended to;
      none  both plain (the control)."""

    def __init__(self, grid, seed, T=7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq, self.wk, self.wr = w(), w(), w()
        self.grid = grid
        self.register_buffer("ring", torch.zeros(1, T, D))

    def forward(self, x):
        k_new, r = x @ self.wk, x @ self.wr
        if self.grid == "new":
            k_new = torch.exp(torch.round(k_new))
        elif self.grid == "ring":
            r = torch.exp(torch.round(r))
        k = torch.cat([self.ring, k_new], 1)
        out = torch.softmax(torch.bmm(x @ self.wq, k.transpose(1, 2)) * 0.25,
                            -1)
        self.ring.copy_(r)
        return out


_RING_ARGS = (torch.randn(1, 7, D, generator=torch.Generator().manual_seed(2)),)


def _ring_logs(grid, seeds=range(8)):
    """{weight seed: (portable bmms, partition log)} for RingBesideFresh."""
    return {s: _lower(RingBesideFresh(grid, s), _RING_ARGS)[:2] for s in seeds}


@pytest.mark.parametrize("grid", ["new", "ring"])
def test_rows_of_one_kind_hide_no_grid_in_the_other(grid):
    """Judged only together, a key's fresh rows and the ring rows an earlier
    call wrote let rows of one kind hide a grid in the other -- a fresh grid
    beside a plain ring, or a ring written with a grid beside plain fresh
    keys, which judging the fresh rows alone also misses. So each kind is
    judged alone too, with every test: portable at every weight seed, by the key's
    grid (its ring rows are the buffer's own: no state refusal)."""
    assert _rules(RingBesideFresh(grid, 0), _RING_ARGS, side=2) == [
        qk_operands.PLAIN]
    missed = [s for s, (bmms, log) in _ring_logs(grid).items()
              if bmms != 1 or "the key, is on a grid" not in log
              or "reads state an earlier call wrote" in log]
    assert missed == [], missed


def test_plain_ring_beside_plain_keys_stays_delegated():
    delegated = [s for s, (bmms, _) in _ring_logs("none").items() if bmms == 0]
    assert delegated == list(range(8)), delegated


def test_a_group_too_small_to_judge_leaves_the_others_to_decide():
    """judge_regime: all rows, then each group alone (the value check's:
    fresh rows, a written buffer's rows, the rows the replay leaves in
    place); a grid in any is a grid, and a group with no verdict of its own
    (one row the same in every run) leaves the others to decide."""
    rng = np.random.default_rng(3)
    plain = [rng.standard_normal((7, D)).astype(np.float32) for _ in range(12)]
    grid = [np.exp(np.round(rng.standard_normal((7, D)))).astype(np.float32)
            for _ in range(12)]
    runs = [np.concatenate([p, g]) for p, g in zip(plain, grid)]
    first = np.arange(14) < 7
    assert qk_values.judge_runs(runs)[0].grid is False
    for fresh in (first, ~first):
        v, _ = qk_values.judge_regime(runs, [("its fresh rows alone", fresh),
                                             ("its buffer rows alone", ~fresh)])
        assert v.grid and v.detail.endswith(" alone"), v
    one = np.arange(8) == 7               # a constant row: no verdict alone
    runs = [np.concatenate([p, np.ones((1, D), np.float32)]) for p in plain]
    assert qk_values.judge_runs([r[one] for r in runs])[0].grid is None
    v, _ = qk_values.judge_regime(runs, [("a", ~one), ("b", one)])
    assert v.grid is False, v


class CounterBesideFresh(nn.Module):
    """K = cat([kc, k_new]) over B heads (the
    bmm's batch), kc a cache written at an int counter the forward advances
    after its q@kT (torchtune's KVCache). The one-call replay changes only
    the rows run 0 wrote; filed as fresh, the rows runs 1-11 write would
    dilute this call's rows and fill the 512 fresh rows kept.
    `grid`: new -- k_new = exp(round(x @ wk)); cache -- the cache written
    with exp(round(x @ wr)); none -- both plain (the control)."""

    def __init__(self, grid, seed, S=256, B=8):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq, self.wk, self.wr = w(), w(), w()
        self.grid = grid
        self.register_buffer("kc", torch.zeros(B, S, D))
        self.register_buffer("cnt", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        n = x.shape[1]
        k_new, r = x @ self.wk, x @ self.wr
        if self.grid == "new":
            k_new = torch.exp(torch.round(k_new))
        elif self.grid == "cache":
            r = torch.exp(torch.round(r))
        k = torch.cat([self.kc, k_new], 1)
        out = torch.softmax(torch.bmm(x @ self.wq, k.transpose(1, 2)) * 0.25,
                            -1)
        self.kc.index_copy_(1, self.cnt + torch.arange(n), r)
        self.cnt.add_(n)
        return out


def _counter_logs(grid, seeds=(0, 1)):
    """{(tokens a head, weight seed): (portable bmms, partition log)} for
    CounterBesideFresh over 8 heads at 7, 8 and 16 tokens a call."""
    out = {}
    for n in (7, 8, 16):
        x = torch.randn(8, n, D, generator=torch.Generator().manual_seed(2))
        for s in seeds:
            out[n, s] = _lower(CounterBesideFresh(grid, s), (x,))[:2]
    return out


@pytest.mark.parametrize("grid", ["new", "cache"])
def test_a_counter_caches_later_rows_are_the_buffers_kind(grid):
    """A row's kind is decided by value -- a row
    whose run-0 value is a row a written buffer held entering run 0 is the
    buffer's own, however long ago a run wrote it -- so this call's grid is
    judged alone without the cache's later rows (new), and a grid the cache
    holds is judged alone without this call's plain keys (cache)."""
    x = torch.randn(8, 8, D, generator=torch.Generator().manual_seed(2))
    assert _rules(CounterBesideFresh(grid, 0), (x,), side=2) == [
        qk_operands.PLAIN]
    missed = [k for k, (bmms, log) in _counter_logs(grid).items()
              if bmms != 1 or "the key, is on a grid" not in log
              or "reads state an earlier call wrote" in log]
    assert missed == [], missed


def test_plain_counter_cache_beside_plain_keys_stays_delegated():
    refused = [k for k, (bmms, _) in _counter_logs("none").items() if bmms]
    assert refused == [], refused


class RingBig(nn.Module):
    """RingBesideFresh over B heads (the bmm's
    batch), N new rows a head a call; the ring [B, T, D] is overwritten
    after the q@kT with the last T rows of r. Past 8 N = 512 fresh rows
    that change from run to run, an order that fills the row budget with
    them keeps no ring row, so a grid only the ring holds goes unjudged."""

    def __init__(self, grid, seed, B=8, T=7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq, self.wk, self.wr = w(), w(), w()
        self.grid, self.T = grid, T
        self.register_buffer("ring", torch.zeros(B, T, D))

    def forward(self, x):
        k_new, r = x @ self.wk, x @ self.wr
        if self.grid == "ring":
            r = torch.exp(torch.round(r))
        k = torch.cat([self.ring, k_new], 1)
        out = torch.softmax(torch.bmm(x @ self.wq, k.transpose(1, 2)) * 0.25,
                            -1)
        self.ring.copy_(r[:, -self.T:])
        return out


def _ring_big_logs(grid, seeds=(0, 1)):
    """{(new rows a head, weight seed): (portable bmms, partition log)} for
    RingBig over 8 heads at 64, 65 and 80 new rows a head."""
    out = {}
    for n in (64, 65, 80):
        x = torch.randn(8, n, D, generator=torch.Generator().manual_seed(2))
        for s in seeds:
            out[n, s] = _lower(RingBig(grid, s), (x,))[:2]
    return out


def test_a_ring_past_512_fresh_rows_is_still_judged_alone():
    """Each kind keeps its own budget of rows that
    change from run to run, so a grid the ring holds is judged alone however
    many fresh rows vary."""
    missed = [k for k, (bmms, log) in _ring_big_logs("ring").items()
              if bmms != 1 or "the key, is on a grid" not in log
              or "reads state an earlier call wrote" in log]
    assert missed == [], missed


def test_plain_ring_past_512_fresh_rows_stays_delegated():
    refused = [k for k, (bmms, _) in _ring_big_logs("none").items() if bmms]
    assert refused == [], refused


class EmptyBesideCache(nn.Module):
    """A written K cache read whole as the key, the
    query exp(round(xq)) -- a grid the rules call plain -- and an unrelated
    written buffer `aux` of `width` columns. With no elements, the held-row
    snapshot's reshape(-1, 0) raised in every regime, which counted as a
    program that cannot run: the bare rules decided and delegated the
    grid."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer("kc", torch.zeros(1, 31, D))
        self.register_buffer("aux", torch.zeros(1, width))

    def forward(self, xq, xk, pos):
        self.kc.index_copy_(1, pos, xk)
        out = torch.softmax(torch.bmm(torch.exp(torch.round(xq)),
                                      self.kc.transpose(1, 2)) * 0.25, -1)
        self.aux.add_(1)
        return out


@pytest.mark.parametrize("width", [0, 1])
def test_an_empty_written_buffer_leaves_the_value_check_on(width):
    bmms, log, _ = _lower(EmptyBesideCache(width), _qk_args())
    assert bmms == 1, log
    assert "the query, is on a grid" in log, log


@pytest.mark.parametrize("step", ["snapshot", "row classes"])
def test_rows_that_cannot_be_classed_keep_the_qk_portable(step, monkeypatch):
    """Failing to read the rows the written buffers
    held entering run 0, or to class a candidate's rows, is no program that
    cannot run -- the candidates stay unjudged and their q@kT portable,
    never the bare rules (which delegate this plain cache)."""
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(qk_values, "_held_rows" if step == "snapshot"
                        else "_row_classes", boom, raising=False)
    g = torch.Generator().manual_seed(6)
    args = (torch.randn(1, 7, D, generator=g), torch.randn(1, 2, D, generator=g),
            torch.tensor([3, 4]))
    bmms, log, _ = _lower(Cache(0.0), args)
    assert bmms == 1, log
    assert "the key, was not judged" in log, log
    assert "the rules alone decided" not in log, log


class RingZeroFresh(nn.Module):
    """A ring of 14 rows: rows 7-13 the call before's x @ wr (written after
    the q@kT at held positions), rows 0-6 zeros no run writes; the key
    cat([ring, k_new]) with k_new = _zero_one(x @ wk), a 0/1 grid the rules
    miss. With example rows -wk[:, 0], k_new is all zeros in run 0, a row
    the ring holds: by value the fresh rows count as the buffer's own (the
    named limit) and the ring's plain rows hide their grid among the buffer
    rows. The group of the rows the replay leaves in place still holds them
    alone."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq, self.wk, self.wr = w(), w(), w()
        self.register_buffer("ring", torch.zeros(1, 14, D))

    def forward(self, x):
        k = torch.cat([self.ring, _zero_one(x @ self.wk)], 1)
        out = torch.softmax(torch.bmm(x @ self.wq, k.transpose(1, 2)) * 0.25,
                            -1)
        self.ring.index_copy_(1, torch.arange(7, 14), x @ self.wr)
        return out


def _zero_rows_input(model):
    """Example rows -wk[:, 0]: (x @ wk)[..., 0] < 0, so _zero_one gives 0."""
    return (-model.wk[:, 0].detach()).reshape(1, 1, D).expand(1, 7, D).clone()


def test_the_rows_the_replay_leaves_in_place_are_judged_alone():
    """The rest of this file passes without the extra rows and groups the
    replay adds. Here only the group of the rows the replay leaves in place
    sees this call's grid alone: its rows are a
    written buffer's by value in run 0, beside the ring's plain rows."""
    for seed in (0, 1):
        m = RingZeroFresh(seed)
        args = (_zero_rows_input(m),)
        assert _rules(RingZeroFresh(seed), args, side=2) == [qk_operands.PLAIN]
        bmms, log, _ = _lower(m, args)
        assert bmms == 1, (seed, log)
        assert "judged on the rows the replay leaves in place alone" in log, (
            seed, log)


class StateRead(nn.Module):
    """K = tanh(ring), the ring the call before's x @ wr (written after the
    q@kT): a plain value computed from state an earlier call wrote, no row a
    written buffer holds."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def w():
            return nn.Parameter(torch.randn(D, D, generator=g) / 4)

        self.wq, self.wr = w(), w()
        self.register_buffer("ring", torch.zeros(1, 7, D))

    def forward(self, x):
        out = torch.softmax(torch.bmm(x @ self.wq, torch.tanh(self.ring)
                                      .transpose(1, 2)) * 0.25, -1)
        self.ring.copy_(x @ self.wr)
        return out


def test_a_plain_value_computed_from_earlier_state_keeps_the_qk_portable():
    """The state refusal itself (no other test here fails without it).
    Values cannot tell state an earlier call wrote
    from a fresh value, so a q@kT whose operand is computed from it stays
    portable even when its values are plain -- speed only (README)."""
    for seed in (0, 1):
        x = torch.randn(1, 7, D, generator=torch.Generator().manual_seed(2))
        bmms, log, _ = _lower(StateRead(seed), (x,))
        assert bmms == 1, (seed, log)
        assert "the key, reads state an earlier call wrote" in log, (seed, log)


class ReplayRaises(nn.Module):
    """A plain key read from a written float cache (so its rows are classed)
    beside an int counter, one more each call, that indexes a 14-row table
    at cnt + 12 when the query's first element is past 100 -- the example
    input's is, unit-normal draws never are. Every run indexes validly, and
    so does the redraw (int buffers are kept); only the replay, run 0's
    inputs on the state that enters run 1, indexes row 14."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(9)
        self.register_buffer("kc", torch.zeros(1, 31, D))
        self.register_buffer("cnt", torch.zeros(1, dtype=torch.long))
        self.register_buffer("table", torch.randn(14, D, generator=g))

    def forward(self, xq, xk, pos):
        self.kc.index_copy_(1, pos, xk)
        out = torch.softmax(torch.bmm(xq, self.kc.transpose(1, 2)) * 0.25, -1)
        extra = self.table.index_select(
            0, self.cnt + 12 * (xq[0, 0, :1] > 100).long())
        self.cnt.add_(1)
        return out, extra


def test_a_replay_that_raises_keeps_the_qk_portable():
    """When the replay raises, the rows cannot be classed: the q@kT stays
    portable, logged, as when the redraw raises -- even this plain key."""
    xq, xk, pos = _qk_args()
    xq = xq.clone()
    xq[0, 0, 0] = 1000.0
    bmms, log, _ = _lower(ReplayRaises(), (xq, xk, pos))
    assert bmms == 1, log
    assert "the replay of run 0 on the state that enters run 1 raised" in log, log


def test_a_replay_that_runs_leaves_the_plain_key_delegated():
    bmms, log, _ = _lower(ReplayRaises(), _qk_args())
    assert bmms == 0, log


class UnwrittenRowsOnly(nn.Module):
    """The key reads only rows of a populated cache the forward never writes."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(6)
        self.register_buffer("kc", torch.randn(1, 31, D, generator=g))

    def forward(self, xq, xk, pos):
        self.kc.index_copy_(1, pos, xk)
        return torch.softmax(torch.bmm(xq, self.kc[:, 16:].transpose(1, 2))
                             * 0.25, -1)


def test_rows_all_left_out_as_unwritten_are_logged_as_such():
    """When every row an operand reads is left out as
    one no run wrote, the no-verdict log says why -- here the rows of a
    populated cache past the rows written, the same in every run and moved
    when the cache's initial content is drawn afresh."""
    bmms, log, _ = _lower(UnwrittenRowsOnly(), _qk_args())
    assert bmms == 0, log
    assert "only a written buffer's initial content reaches" in log, log


class PosLm(nn.Module):
    """Learned positions (pos indexes a 1,024-row table)
    that also write an 8-row cache. Drawn over the table's rows, they ran
    off the cache, and the fallback judged nothing."""

    def __init__(self, fq):
        super().__init__()
        g = torch.Generator().manual_seed(4)
        self.wte = nn.Embedding(100, D)
        self.wpe = nn.Embedding(1024, D)
        self.wq = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.wk = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.register_buffer("kc", torch.zeros(1, 8, D))
        self.fq = fq

    def forward(self, ids, pos):
        h = self.wte(ids) + self.wpe(pos)
        self.kc.index_copy_(1, pos[0], h @ self.wk)
        q = self.fq(self, h @ self.wq, ids)
        return torch.softmax(torch.bmm(q, self.kc.transpose(1, 2)) * 0.25, -1)


def test_positions_that_write_a_cache_are_held():
    """An integer input that also writes (or indexes) a tensor that is not a
    parameter is a position, held at the example's values: no run raises,
    and the grid is found."""
    from executorch.backends.esp32s3.operator_support.qk_values import _token_rows

    model = PosLm(RULES_MISSED["sign_by_maximum"][0])
    args = (torch.tensor([[7, 9]]), torch.tensor([[2, 3]]))
    seen = {}
    orig = qk_values.ValueCheck._read

    def spy(self):
        nodes = {n.name: n for n in self.ep.graph_module.graph.nodes}
        seen["pos"] = _token_rows(self.ep, nodes["pos"])
        seen["ids"] = _token_rows(self.ep, nodes["ids"])
        orig(self)
        seen["notes"], seen["failure"] = list(self.notes), self.failure

    qk_values.ValueCheck._read = spy
    try:
        bmms, log, part = _lower(model, args)
    finally:
        qk_values.ValueCheck._read = orig
    assert seen["pos"] is None and seen["ids"] == 100, seen
    assert seen["notes"] == [] and seen["failure"] is None, seen
    assert bmms == 1, log


class LongCache(nn.Module):
    """A key cache written at pos by a sign the rules call
    plain; with 2,048 rows the written row fell between the rows sampled."""

    def __init__(self, S):
        super().__init__()
        g = torch.Generator().manual_seed(2)
        self.wk = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.register_buffer("k", torch.zeros(1, S, D))

    def forward(self, xq, xk, pos):
        kk = xk @ self.wk
        self.k.index_copy_(1, pos, kk / torch.maximum(kk, -kk))
        return torch.softmax(torch.bmm(xq, self.k.transpose(1, 2)) * 0.25, -1)


@pytest.mark.parametrize("S", [32, 2048])
def test_written_key_grid_in_a_long_cache_stays_portable(S):
    """The rows a run wrote are kept before the rows are thinned to
    MAX_ROWS: the written row of a 2,048-row cache is judged."""
    args = (torch.randn(1, 7, D), torch.randn(1, 1, D), torch.tensor([5]))
    bmms, log, part = _lower(LongCache(S), args)
    assert bmms == 1, log


def test_no_verdict_is_logged():
    """A q@kT the values cannot judge is not silent: the
    log names it and why (a query that holds still: the same values in every
    run)."""

    class Still(nn.Module):
        """The query: rows of a buffer at positions (held): the same values
        in every run, whatever the floats do."""

        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.randn(8, D))

        def forward(self, xq, xk, pos):
            q = self.table.index_select(0, pos).unsqueeze(0) + 0.0 * xq
            return torch.softmax(torch.bmm(q, xk.transpose(1, 2)) * 0.25, -1)

    bmms, log, part = _lower(Still(), (torch.randn(1, 7, D), torch.randn(1, 31, D),
                                       torch.arange(7)))
    assert "no verdict from values" in log, log


def test_value_check_reseeds_no_device_generator(monkeypatch):
    """Nothing the check runs calls a device generator's
    seed (torch.manual_seed reseeds CUDA, MPS and XPU)."""
    calls, inside = [], []
    for mod, name in ((torch.cuda, "manual_seed_all"), (torch.mps, "manual_seed"),
                      (torch.xpu, "manual_seed_all")):
        monkeypatch.setattr(mod, name,
                            lambda *a, _n=name: calls.append(_n) if inside else None)
    orig = qk_values.ValueCheck._read

    def spy(self):
        inside.append(1)
        try:
            return orig(self)
        finally:
            inside.pop()

    monkeypatch.setattr(qk_values.ValueCheck, "_read", spy)
    _lower(Head(RULES_MISSED["sign_by_maximum"][0]), _inputs(7))
    assert calls == [], calls


@pytest.mark.parametrize("case", sorted(PLAIN_SPELLINGS))
def test_plain_spellings_stay_delegated(case):
    fq, rows = PLAIN_SPELLINGS[case]
    bmms, log, part = _lower(Head(fq, rows), _inputs(rows))
    assert bmms == 0, log
    assert "q@kT value check" not in log, log


def test_rules_refusal_stands():
    """Values never decide an operand the rules refused: the reason stays
    the rules'."""
    bmms, log, part = _lower(Head(lambda m, x, ids: torch.round(x * 4), 7),
                             _inputs(7))
    assert bmms == 1
    assert part.support.kinds[KIND_QK_VALUES] == 0
    assert part.support.kinds["q@kT operand already on a grid"] >= 1


# --- pitfalls -------------------------------------------------------------------


def test_degenerate_example_inputs_get_draws():
    """Zeros as example inputs: the draws still find a grid, and a plain
    query stays delegated."""
    grid = Head(RULES_MISSED["sign_by_maximum"][0])
    assert _lower(grid, _inputs(7, zeros=True))[0] == 1
    plain = Head(PLAIN_SPELLINGS["projection"][0])
    assert _lower(plain, _inputs(7, zeros=True))[0] == 0


def test_token_ids_drawn_within_their_table():
    from executorch.backends.esp32s3.operator_support.qk_values import _token_rows

    class Lm(nn.Module):
        def __init__(self, fq):
            super().__init__()
            self.emb = nn.Embedding(50, D)
            self.wq = nn.Linear(D, D)
            self.register_buffer("cache", torch.zeros(1, 8, D))
            self.fq = fq

        def forward(self, ids, pos):
            h = self.emb(ids)
            self.cache.index_copy_(1, pos, h)
            return torch.softmax(torch.bmm(self.fq(self, h), self.cache.transpose(
                1, 2)), -1)

    args = (torch.zeros(1, 2, dtype=torch.long), torch.tensor([5, 6]))
    seen = {}
    orig = qk_values.ValueCheck._read

    def spy(self):
        nodes = {n.name: n for n in self.ep.graph_module.graph.nodes}
        seen["ids"] = _token_rows(self.ep, nodes["ids"])
        seen["pos"] = _token_rows(self.ep, nodes["pos"])
        orig(self)
        seen["notes"], seen["failure"] = list(self.notes), self.failure
        seen["regimes"] = sorted({e[0] for v in self.samples.values() for e in v})

    qk_values.ValueCheck._read = spy
    try:
        plain = _lower(Lm(lambda m, h: m.wq(h)), args)
        assert plain[0] == 0, plain[1]
        # ids index a 50-row table: drawn within it; pos indexes the cache
        # write: kept at the example's values, so no run raises
        assert seen["ids"] == 50 and seen["pos"] is None
        assert seen["notes"] == [] and seen["failure"] is None
        assert seen["regimes"] == sorted([qk_values.EMBEDDINGS_VARY,
                                          qk_values.IDS_VARY])
        grid = _lower(Lm(lambda m, h: m.wq(h) / torch.maximum(m.wq(h), -m.wq(h))),
                      args)
        assert grid[0] == 1, grid[1]
    finally:
        qk_values.ValueCheck._read = orig


class Cache(nn.Module):
    """A float K cache written at `pos` (two rows a call) and read whole by
    the q@kT (OP_BMM_DYN_I8's key)."""

    def __init__(self, fill):
        super().__init__()
        g = torch.Generator().manual_seed(2)
        self.wk = nn.Parameter(torch.randn(D, D, generator=g) / 4)
        self.register_buffer("k", torch.full((1, 32, D), fill))

    def forward(self, xq, xk, pos):
        self.k.index_copy_(1, pos, xk @ self.wk)
        return torch.softmax(torch.bmm(xq, self.k.transpose(1, 2)) * 0.25, -1)


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_written_cache_keeps_its_delegated_qk(fill):
    """Runs on copies (the program's buffer keeps its value), and the rows
    no run wrote make no grid of the key."""
    m = Cache(fill)
    args = (torch.randn(1, 7, D), torch.randn(1, 2, D), torch.tensor([3, 4]))
    bmms, log, part = _lower(m, args)
    assert bmms == 0, log
    assert bool((m.k == fill).all())


class ShiftCache(nn.Module):
    """A decode step's keys: cat of a cache and the new row, the cache then
    shifted by one row (test_qk_matrix's CatCache): each call's rows are the
    last call's, a place lower."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("past", torch.randn(1, 15, D, generator=g))

    def forward(self, q, k_new):
        k = torch.cat([self.past, k_new], dim=1)
        out = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)
        self.past.copy_(k[:, 1:])
        return out


def test_shifted_cache_rows_are_no_codebook():
    """The rows a shifted cache carries come back in every later run, which
    vector levels read as a codebook (test_qk_matrix's
    test_cat_cache_keys[plain] is refused so). No exemption judges such
    rows without vector levels: they are the cache's own
    rows, judged with every test, and vector levels keeps the q@kT
    portable. Speed only: a precise rule for state an earlier call wrote is
    future work."""
    args = (torch.randn(1, 3, D), torch.randn(1, 1, D))
    bmms, log, part = _lower(ShiftCache(), args)
    assert bmms == 1, log
    assert "left portable by its operand values" in log, log


def test_failed_runs_leave_the_rules_to_decide():
    """A float input used as an index raises on drawn inputs. The example
    inputs alone would be one run, and one run cannot judge (judge needs
    runs that differ), so the rules decide, the log says so
    and why, and the export never fails."""

    class Indexed(nn.Module):
        """A second output indexes a table at a float input's integer
        part: valid for the example, out of range for most draws."""

        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.randn(8, D))

        def forward(self, xq, xk, at):
            out = torch.softmax(torch.bmm(xq * 2.0, xk.transpose(1, 2)), -1)
            return out, self.table.index_select(0, at.long())

    args = (torch.randn(1, 7, D), torch.randn(1, 31, D),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    bmms, log, part = _lower(Indexed(), args)
    assert bmms == 0, log
    assert "the example inputs alone, one run, cannot judge" in log, log
    assert "the rules alone decided every q@kT" in log, log
    bmms, log, part = _lower(Indexed(), args, example=False)
    assert bmms == 0, log
    assert "the rules alone decided every q@kT" in log, log


def test_deterministic_and_global_rng_untouched():
    def Noisy():
        return Head(RULES_MISSED["sign_by_maximum"][0])

    states = []
    orig = qk_values.ValueCheck._read

    def spy(self):
        before = torch.get_rng_state()
        orig(self)
        states.append(torch.equal(torch.get_rng_state(), before))

    torch.manual_seed(1234)
    qk_values.ValueCheck._read = spy
    try:
        first = _lower(Noisy(), _inputs(7))
        torch.manual_seed(99)
        second = _lower(Noisy(), _inputs(7))
    finally:
        qk_values.ValueCheck._read = orig
    assert states == [True, True]
    assert first[0] == second[0] == 1
    assert [l for l in first[1].splitlines() if "q@kT" in l] == [
        l for l in second[1].splitlines() if "q@kT" in l]


def test_no_device_generator_is_seeded():
    """torch.manual_seed seeds the CUDA and MPS generators
    too, and fork_rng(devices=[]) restores the CPU's alone. The value check
    seeds the CPU generator only (restored after), never torch.manual_seed,
    and every device generator present is left as it was."""
    calls = []
    real = torch.manual_seed

    def spy(seed):
        calls.append(seed)
        return real(seed)

    devices = []
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(99)
        devices.append(("mps", torch.mps.get_rng_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(99)
        devices.append(("cuda", lambda: torch.cuda.get_rng_state_all()[0]))
    before = {name: get().clone() for name, get in devices}
    cpu = torch.get_rng_state()
    torch.manual_seed = spy
    try:
        bmms, log, part = _lower(Head(RULES_MISSED["sign_by_maximum"][0]),
                                 _inputs(7))
    finally:
        torch.manual_seed = real
    assert bmms == 1, log
    assert calls == [], calls
    assert torch.equal(torch.get_rng_state(), cpu)
    for name, get in devices:
        assert torch.equal(get(), before[name]), name


def test_budget_keeps_the_qk_portable(monkeypatch):
    """Samples and runs are bounded. Past
    MAX_BYTES recorded, or MAX_SECONDS of runs, no operand is judged, every
    q@kT the check would judge stays portable (exact is safe; a grid not
    judged must not be admitted), and the log says so."""
    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        KIND_QK_VALUES_BUDGET,
    )

    for knob, value in (("MAX_BYTES", 1), ("MAX_SECONDS", 0.0)):
        monkeypatch.setattr(qk_values, knob, value)
        bmms, log, part = _lower(Head(PLAIN_SPELLINGS["projection"][0]),
                                 _inputs(7))
        assert bmms == 1, (knob, log)
        assert part.support.kinds[KIND_QK_VALUES_BUDGET] >= 1, knob
        assert "value check passed its budget" in log, (knob, log)
        monkeypatch.undo()


def test_samples_subsampled_as_recorded(monkeypatch):
    """Rows past MAX_ROWS are dropped as each run records (evenly spaced,
    the same rows in every run), not after twelve whole copies: the bytes
    recorded are MAX_ROWS rows a run."""
    monkeypatch.setattr(qk_values, "MAX_ROWS", 3)
    seen = {}
    real = qk_values.ValueCheck._read

    def spy(self):
        real(self)
        seen["shapes"] = {n: [rows.shape for e in v for runs in [e[1]]
                              for _idx, rows, _n in runs]
                          for n, v in self.samples.items()}
        seen["bytes"] = self.recorded

    monkeypatch.setattr(qk_values.ValueCheck, "_read", spy)
    _lower(Head(PLAIN_SPELLINGS["projection"][0]), _inputs(7))
    shapes = [s_ for v in seen["shapes"].values() for s_ in v]
    assert shapes and all(s_ == (3, D) for s_ in shapes), seen["shapes"]
    assert seen["bytes"] == len(shapes) * (3 * D * 4 + 3 * 8)   # rows, indices


def test_no_runs_without_a_candidate():
    calls = []
    orig = qk_values.ValueCheck._read

    def spy(self):
        calls.append(1)
        return orig(self)

    qk_values.ValueCheck._read = spy
    try:
        _lower(nn.Sequential(nn.Linear(D, D)), (torch.randn(1, 7, D),))
        assert calls == []
        from executorch.backends.esp32s3._passes import lower_esp32
        from executorch.backends.esp32s3.partition.espressif_partitioner import (
            Esp32Partitioner,
        )

        model = Head(RULES_MISSED["sign_by_maximum"][0])
        with contextlib.redirect_stdout(io.StringIO()):
            lower_esp32(torch.export.export(model.eval(), _inputs(7)),
                        [Esp32Partitioner(quantize_qk=False)])
        assert calls == []
        _lower(model, _inputs(7))
        assert calls == [1]           # once per program, for every gate
    finally:
        qk_values.ValueCheck._read = orig


def test_samples_are_released_after_the_partition():
    """The module-level cache is keyed by the partition's
    graph module, which upstream caches keep alive in a user's process, so
    the samples must go when the partition returns -- only the verdicts
    stay. Judging frees a candidate's samples, so a plain Head (query and
    key both judged) leaves release() nothing to free, and would pass with
    release() disabled. Here the query is a grid:
    the gate stops before it judges the key, whose samples only release()
    frees; the check is held, as a user holding the program holds it."""
    seen = []
    orig = qk_values.ValueCheck._read

    def spy(self):
        seen.append(self)
        return orig(self)

    qk_values.ValueCheck._read = spy
    try:
        bmms, log, _ = _lower(Head(RULES_MISSED["sign_by_maximum"][0]),
                              _inputs(7))
    finally:
        qk_values.ValueCheck._read = orig
    assert bmms == 1 and "the query, is on a grid" in log, log
    (check,) = seen
    assert check.samples is not None
    held = sum(rows.nbytes + idx.nbytes for runs in check.samples.values()
               for e in runs for idx, rows, _n in e[1])
    assert held == 0, f"{held} bytes of sample runs still held"


def test_check_holds_its_program_weakly():
    """The check holds its program by a weak reference and its tables by
    node name, so it is never itself the strong path that keeps a program
    alive."""
    import weakref

    seen = []
    orig = qk_values.ValueCheck._read

    def spy(self):
        seen.append(self)
        return orig(self)

    qk_values.ValueCheck._read = spy
    try:
        _lower(Head(RULES_MISSED["sign_by_maximum"][0]), _inputs(7))
    finally:
        qk_values.ValueCheck._read = orig
    check = seen[0]
    assert isinstance(check._ep, weakref.ref)
    assert all(isinstance(k, str) for k in check._verdicts)


def _fx_held(obj, seen=None):
    """The fx Nodes, Graphs and GraphModules `obj` holds through dicts,
    lists, tuples and sets (a check's attributes)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, (torch.fx.Node, torch.fx.Graph, torch.fx.GraphModule)):
        return [type(obj).__name__]
    if isinstance(obj, dict):
        items = list(obj.keys()) + list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)
    else:
        return []
    return [h for i in items for h in _fx_held(i, seen)]


def test_token_id_program_is_not_kept_alive_by_its_check():
    """The regime that draws the rows looked up at
    token ids kept the aten.embedding nodes it replaces, and a node holds its
    graph, the graph its module -- the key of the weak cache whose value is
    the check -- so every token-id program stayed alive after its partition.
    The check holds them by name: once the upstream caches let go
    (release_upstream_caches), the partition's graph module is collected
    and its cache entry is gone."""
    import gc
    import weakref

    from executorch.backends.esp32s3.test.suite.plugin import (
        release_upstream_caches,
    )

    seen = []
    orig = qk_values.ValueCheck._read

    def spy(self):
        seen.append((weakref.ref(self.ep.graph_module), self))
        return orig(self)

    qk_values.ValueCheck._read = spy
    try:
        ids = (torch.randint(0, 1000, (1, 7),
                             generator=torch.Generator().manual_seed(1)),)
        bmms, log = _lower(IdsLm(IDS_GRIDS["exp_of_round"]), ids)[:2]
    finally:
        qk_values.ValueCheck._read = orig
    assert bmms == 1, log
    gm_ref, check = seen[0]
    assert check._embeddings, "no aten.embedding drawn"
    assert _fx_held(vars(check)) == []
    check_ref = weakref.ref(check)
    del seen, check
    release_upstream_caches()
    gc.collect()
    assert gm_ref() is None, "the partition's graph module is still alive"
    assert check_ref() is None, "the weak cache still holds the check"
