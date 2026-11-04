# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fitted int16 lookup tables for espressif::lut16.

An activation f between two static int16 quantisers,

    q1(X) -> dq1 -> f -> q2 -> dq2

reads one of at most 65,536 input codes and writes one output code. For input
code c the eager graph computes

    E(c) = q2(f(dq1(c)))

with f evaluated by torch itself in float32 -- the graph's own dq1, the ATen
kernel in quant_lut's canonical vector context, the graph's own q2 -- as
quant_lut builds its int8 tables (float32, not float64). A direct table over
every code is exact but 128 KB, so this module fits small ones instead, in five
schemes:

  direct          one int16 output code per input code of [lo, hi]
  uniform_linear  2^F-code segments from lo, continuous int32 nodes in units
                  of 2^-G codes, integer interpolation
  uniform_cubic   2^F-code segments, a float32 cubic per segment (Horner, one
                  rounding per operation), round half to even
  nsau_linear     float-like segments: |c| < 2^(m+1) one code each, every
                  octave above split into 2^m (NSAU finds the octave); both
                  signs when lo < 0
  nsau_cubic      the same segments, cubic

and two flags on the scheme word (mirrors.LUT16_OFFSET, LUT16_TAIL):

  offset          NSAU: each sign's half holds only the segments its codes
                  in [lo, hi] reach (a window away from 0 drops the
                  segments below it, and the sign it does not reach) --
                  used wherever it is smaller, which is every NSAU table
                  with negative codes or lo > 0
  tail            uniform, for the functions the tail rule names
                  (TAIL_FUNCS: SiLU, GELU, hardswish, x^2): a code past [lo,
                  hi] reads the edge segment at its own offset instead of
                  clamping, so where f is a line past some code (GELU and
                  hardswish become x or 0) the table stops there and the
                  edge segment, fitted as that line, continues to the rail.
                  Only where that is exact: the codes past [lo, hi] meet the
                  float32-floor bar of a window, on top of the table's own

Every candidate is scored by its output-code flips: the input codes whose
table output differs from E. mirrors.lut16_codes is the arithmetic of
every score, the device's step for step, so a table's flips are the device's.

  Grid, Site      a quantiser's qparams; a site (f between two grids), its
                  target E and the float32 floor, computed once
  fit_scheme      one scheme at one size: least squares on u = f / s2 + z2,
                  then the flip fit -- for linear tables the exact Viterbi
                  over integer nodes at the largest int32-safe node precision
                  G, computed from the fitted table; for cubic tables, whose
                  segments are independent, each segment's exact minimum
                  over c0 offsets (and, for an NSAU table over a window
                  that still misses its float32-floor bar, over c0 and c1
                  together: refine_joint)
  active_range    the codes a table must hold: an end saturated at an
                  output rail is cut back, its codes clamp to the cut
  fit             the scheme chooser under a byte budget (Spec): direct for
                  input ranges up to 1,024 codes, NSAU cubic for rsqrt,
                  reciprocal and log, uniform cubic otherwise -- the smallest
                  size whose flips are within the bar, the other cubic scheme
                  as the fallback -- or BudgetExceeded, which is
                  fold_fitted_luts's lut16-budget refusal
  fit_smallest    the chooser for a site fitted over an observed window:
                  every candidate in order of bytes, the first within the bar
                  (no tail candidates: a window's site holds only the
                  window's codes; the graph-only table, tail and all,
                  competes in fold_lut16_pass.fit_parts)
  linear_ends     where E becomes a line toward each rail (the tail's cut)

numpy and scipy at import time; torch only to evaluate a site's f.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from executorch.backends.esp32s3.quantizer import mirrors as qm

SCHEMES = qm.LUT16_SCHEMES
SCHEME_ID = {name: i for i, name in enumerate(SCHEMES)}
OFFSET, TAIL = qm.LUT16_OFFSET, qm.LUT16_TAIL
I16 = (-32768, 32767)
# The tail rule: the functions whose ends a line continues past the table.
# x^2 is "pow" with exponent 2 (Site.tail_rule says so; site_from sets it).
TAIL_FUNCS = frozenset({"silu", "gelu", "hardswish"})

# The chooser: a direct table up to this many
# input codes; NSAU segments for the functions singular at 0; a byte budget
# per site; and the flips bar, the float32 floor's scale (0.05% of codes).
# 8 KB, not 4: reciprocal on Arm a16w8's site (s1 = s2 = 2^-12,
# every code) flips 72 codes at NSAU m = 3 (3.3 KB) and meets the bar only at
# m = 4 (6 KB). The chooser takes the smallest table within the bar, so the
# smooth functions still get 1-2 KB.
DIRECT_MAX = 1024
SINGULAR = frozenset({"rsqrt", "reciprocal", "log", "log2", "log10"})
DEFAULT_BUDGET = 8192
BAR = 0.0005
UNIFORM_SIZES = (16, 32, 64, 128, 256, 512)     # segments
NSAU_SIZES = (1, 2, 3, 4, 5)                     # m: 2^m segments per octave
# The runtime's bound on cubic coefficients (quant_lut16.cpp validates it):
# |y| stays below 2^22, where the 1.5 * 2^23 rounding is exact.
COEF_MAX = float(2 ** 19)


class BudgetExceeded(ValueError):
    """No scheme meets the bar within the byte budget (lut16-budget). .best
    is the candidate with the fewest flips, for the report."""

    def __init__(self, message, best=None):
        super().__init__(message)
        self.best = best


# --- quantisers and sites ------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """A static per-tensor quantiser: q(y) = clamp(rne(fl(y * fl(1/s))) + z),
    dq(c) = fl(fl(c - z) * fl(s))."""

    scale: float
    zero_point: int
    qmin: int
    qmax: int

    @property
    def codes(self):
        return np.arange(self.qmin, self.qmax + 1, dtype=np.int64)

    @property
    def size(self):
        return self.qmax - self.qmin + 1


def _torch_function(fn, kwargs):
    """xs (float32 numpy) -> torch's f(xs) as float32 numpy, in quant_lut's
    canonical vector context, for an allowlisted activation name."""
    import torch

    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    target = {name: op for op, name in fold_quant_lut_pass.ALLOWLIST.items()}[fn]
    args = tuple(kwargs.pop("args", ())) if "args" in kwargs else ()

    def values(xs):
        t = torch.from_numpy(np.ascontiguousarray(xs, dtype=np.float32))
        return fold_quant_lut_pass.evaluate(target, t, args, kwargs).numpy()

    def values64(xs):
        t = torch.from_numpy(np.asarray(xs, dtype=np.float64))
        with torch.no_grad():
            return target(t, *args, **kwargs).numpy()

    return values, values64


@dataclass
class Site:
    """One table site: f between the input grid `inq` and the output grid
    `outq`. `fn` names f (fold_quant_lut_pass.ALLOWLIST's names: the chooser
    reads it); `values` evaluates f on float32 inputs as the eager graph does
    (default: torch's ATen kernel for `fn`); `values64` in float64, for the
    float32 floor (optional). `target` overrides E (the report's float64
    targets in test_lut_fit.py)."""

    fn: str
    inq: Grid
    outq: Grid
    values: Optional[Callable] = None
    values64: Optional[Callable] = None
    kwargs: dict = field(default_factory=dict)
    target: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    floor_bar: bool = False     # the bar is the float32 floor alone (bar_flips)
    tail_rule: Optional[bool] = None   # the tail rule; None: fn in TAIL_FUNCS
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def tails(self):
        """Whether the tail rule extrapolates this function's ends."""
        return self.fn in TAIL_FUNCS if self.tail_rule is None else bool(self.tail_rule)

    def __post_init__(self):
        if self.values is None:
            self.values, default64 = _torch_function(self.fn, dict(self.kwargs))
            if self.values64 is None:
                self.values64 = default64

    @property
    def codes(self):
        return self.inq.codes

    def x(self):
        """dq1 of every input code (the decomposed op's arithmetic)."""
        if "x" not in self._cache:
            self._cache["x"] = qm.dequantize_per_tensor(
                self.codes, self.inq.scale, self.inq.zero_point).astype(np.float32)
        return self._cache["x"]

    def y(self):
        """f(dq1(c)): the eager graph's float32 values (a float64 `values`,
        the report's reference f, is kept as it is)."""
        if "y" not in self._cache:
            y = np.asarray(self.values(self.x()))
            self._cache["y"] = y if y.dtype == np.float64 else y.astype(np.float32)
        return self._cache["y"]

    def E(self):
        """The eager output code of every input code."""
        if self.target is not None:
            return np.asarray(self.target, dtype=np.int64)
        if "E" not in self._cache:
            self._cache["E"] = quantize_codes(self.y(), self.outq)
        return self._cache["E"]

    def u(self, margin=0.25):
        """The real value a table interpolates: f / s2 + z2 per input code.
        It continues past the output rails up to `margin` of the code range,
        so the final clamp saturates with no kink inside a segment, and is
        capped there; where f is not finite (a pole, NaN) it is E itself."""
        key = ("u", margin)
        if key not in self._cache:
            y = self.y().astype(np.float64)
            E = self.E().astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                u = y / self.outq.scale + self.outq.zero_point
            span = (self.outq.qmax - self.outq.qmin) * margin
            u = np.clip(u, self.outq.qmin - span, self.outq.qmax + span)
            self._cache[key] = np.where(np.isfinite(y), u, E)
        return self._cache[key]

    def w(self):
        """Fit weights per input code: uniform (the graph alone) unless given;
        given weights are normalised to mean 1 and floored at 1e-4, so nodes
        over empty regions follow the uniform fit."""
        if self.weights is None:
            return np.ones(self.inq.size)
        w = np.asarray(self.weights, dtype=np.float64)
        return np.maximum(w / max(w.mean(), 1e-300), 1e-4)

    def floor_mask(self):
        """Per input code, whether it is on the float32 floor: its code from
        float32-rounded f differs from the code of float64 f quantised
        directly. None without values64."""
        if self.values64 is None:
            return None
        if "floor_mask" not in self._cache:
            y = np.asarray(self.values64(self.x().astype(np.float64)), dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                direct = np.rint(y / self.outq.scale) + self.outq.zero_point
                rounded = quantize_codes(np.clip(y, -3e38, 3e38).astype(np.float32),
                                         self.outq)
            ok = np.isfinite(direct)
            direct = np.clip(np.where(ok, direct, 0), self.outq.qmin, self.outq.qmax)
            self._cache["floor_mask"] = ok & (direct.astype(np.int64) != rounded)
        return self._cache["floor_mask"]

    def floor(self):
        """The float32 floor: how many input codes floor_mask() marks. None
        without values64."""
        mask = self.floor_mask()
        return None if mask is None else int(mask.sum())


def quantize_codes(y, grid):
    """quantize_per_tensor's codes for float32 values (NaN: torch gives the
    zero point on macOS arm64's NEON path; ET's kernel is undefined there)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default(
        t, grid.scale, grid.zero_point, grid.qmin, grid.qmax, torch.int16)
    return q.numpy().astype(np.int64)


# --- tables -----------------------------------------------------------------------

@dataclass
class Table:
    """A fitted table: the op's arguments (scheme, lo, hi, shift, G and the
    data tensor) plus its score against E."""

    scheme: str
    lo: int
    hi: int
    shift: int          # F (uniform: log2 segment width) or m (NSAU)
    G: int              # linear: node precision; 0 otherwise
    data: np.ndarray    # int16 [n] | int32 [nodes] | float32 [segments, 4]
    out_qmin: int
    out_qmax: int
    flips: int = 0
    codes: int = 0
    max_err: int = 0
    flipped: list = field(default_factory=list)   # the first 32 flipped codes
    floor: Optional[int] = None
    bar: int = 0
    flags: int = 0      # OFFSET (NSAU) or TAIL (uniform), on the scheme word
    tail_flips: int = 0       # with TAIL: codes past [lo, hi] that flip
    tail_bar: int = 0         # ... and their float32 floor

    @property
    def scheme_id(self):
        return SCHEME_ID[self.scheme] | self.flags

    @property
    def label(self):
        """The scheme's name with its flag (census records, reports)."""
        return self.scheme + ("+offset" if self.flags & OFFSET else "") + (
            "+tail" if self.flags & TAIL else "")

    @property
    def nbytes(self):
        return int(self.data.nbytes)

    @property
    def size(self):
        """Segments (entries for direct; with both halves of a linear NSAU
        table, its node chains less one node each)."""
        chains = 2 if self.scheme == "nsau_linear" and self.lo < 0 and (
            self.hi >= 0 or not self.flags & OFFSET) else 1
        return int(self.data.shape[0]) - (chains if self.scheme.endswith("linear") else 0)

    def evaluate(self, c):
        """Output codes for input codes c (the device arithmetic)."""
        return qm.lut16_codes(c, self.scheme_id, self.data, self.lo, self.hi,
                              self.shift, self.G, self.out_qmin, self.out_qmax)

    def describe(self):
        size = (f"{self.size} entries" if self.scheme == "direct" else
                f"m={self.shift}, {self.size} segments" if self.scheme.startswith("nsau")
                else f"{self.size} x 2^{self.shift}")
        g = f", G={self.G}" if self.scheme.endswith("linear") else ""
        tail = (f"; {self.tail_flips} of the codes past it flip (floor {self.tail_bar})"
                if self.flags & TAIL else "")
        return (f"{self.label} [{self.lo}, {self.hi}] {size}{g}, {self.nbytes} B: "
                f"{self.flips} of {self.codes} codes flip (bar {self.bar}, float32 "
                f"floor {self.floor}), max error {self.max_err}{tail}")


def _score(table, site):
    E = site.E()
    T = table.evaluate(site.codes)
    d = np.abs(T - E)
    bad = np.nonzero(d)[0]
    table.flips = int(len(bad))
    table.codes = int(len(E))
    table.max_err = int(d.max()) if len(d) else 0
    table.flipped = [int(c) for c in site.codes[bad[:32]]]
    table.floor = site.floor()
    table.bar = bar_flips(site)
    if table.flags & TAIL:
        past = (site.codes < table.lo) | (site.codes > table.hi)
        mask = site.floor_mask()
        table.tail_flips = int((d[past] > 0).sum())
        table.tail_bar = int(mask[past].sum()) if mask is not None else 0
    return table


def within_bar(table):
    """The table meets its bar: its flips, and with TAIL the flips past [lo,
    hi] -- the codes the tail is used for -- within their float32 floor
    (a window's bar, lut16_range)."""
    return table.flips <= table.bar and (
        not table.flags & TAIL or table.tail_flips <= table.tail_bar)


def bar_flips(site):
    """Flips a fitted table may have: 0.05% of the input codes, or the float32
    floor where that is larger. A site with floor_bar (one fitted over an
    observed window: a few hundred or thousand codes, the ones the data sits
    on) gets the float32 floor alone, without the 0.05% that stands in for it
    over all 65,535 codes."""
    if site.floor_bar:
        return site.floor() or 0
    return max(int(math.ceil(BAR * site.inq.size)), site.floor() or 0)


def active_range(site, margin=2.0):
    """[lo, hi] of the input codes a table must hold: an end where f has
    saturated at an output rail -- every code from there on has E at the rail
    and u at least `margin` codes past it -- is cut back to its first code,
    and the codes beyond clamp to it (exp past ln(rail), a pole's rail
    region). A cubic cannot follow the kink where f meets the rail; past the
    cut no segment has to."""
    c, u, E, y = site.codes, site.u(), site.E(), site.y()
    qmin2, qmax2 = site.outq.qmin, site.outq.qmax
    lo, hi = int(c[0]), int(c[-1])
    top = (E == qmax2) & ((u >= qmax2 + margin) | (y == np.inf))
    if top[-1]:
        run = np.flatnonzero(~top)
        hi = int(c[run[-1] + 1]) if len(run) else int(c[0])
    bottom = (E == qmin2) & ((u <= qmin2 - margin) | (y == -np.inf))
    if bottom[0]:
        run = np.flatnonzero(~bottom)
        lo = int(c[run[0] - 1]) if len(run) else int(c[-1])
    return lo, max(lo, hi)


# --- segments ---------------------------------------------------------------------

def uniform_shift(lo, hi, K):
    """The F of K (or fewer) 2^F-code segments covering [lo, hi]."""
    n = hi - lo + 1
    return max(0, int(math.ceil(math.log2(max(n, 1) / K)))) if n > K else 0


def segment_count(scheme, lo, hi, shift, flags=0):
    """Entries the device indexes as segment starts: uniform ceil(n / 2^F);
    NSAU the rows of its halves (mirrors.lut16_nsau_layout; a linear
    table's node chains count one node more each)."""
    if scheme in ("uniform_linear", "uniform_cubic"):
        return ((hi - lo) >> shift) + 1
    entries = qm.lut16_entries(SCHEME_ID[scheme] | flags, lo, hi, shift)
    if scheme == "nsau_linear":
        return entries - 1
    return entries


def table_bytes(scheme, lo, hi, shift, flags=0):
    entries = qm.lut16_entries(SCHEME_ID[scheme] | flags, lo, hi, shift)
    return entries * (2 if scheme == "direct" else 16 if scheme.endswith("cubic") else 4)


def nsau_flags(lo, hi, shift, scheme="nsau_cubic"):
    """OFFSET when it makes the NSAU table smaller, else 0."""
    return OFFSET if (table_bytes(scheme, lo, hi, shift, OFFSET)
                      < table_bytes(scheme, lo, hi, shift)) else 0


def _locate(site, scheme, lo, hi, shift, flags=0):
    """(entry j, t, log2 width) of every input code: the device's index (a
    cubic segment's row; linear tables: segment j lies between nodes j and j
    + 1)."""
    return qm.lut16_index(site.codes, SCHEME_ID[scheme] | flags, lo, hi, shift)


# --- fitters ----------------------------------------------------------------------

def _u_clamped(site, lo, hi, flags=0):
    """u at each code's table position: a code past [lo, hi] is read at the
    end it clamps to, so it weighs on that point (active_range) -- with
    TAIL, at its own code, where the edge segment reaches it."""
    codes = site.codes if flags & TAIL else np.clip(site.codes, lo, hi)
    return site.u()[codes - site.inq.qmin]


def fit_cubic(site, scheme, lo, hi, shift, flags=0, lines=(), floor=1e-4):
    """Per-segment least squares of a cubic in tf = t / width on u, rounded to
    float32: c1..c3 first, then c0 refitted on the rounded higher terms so the
    constant absorbs their rounding. The rows in `lines`
    (a TAIL table's edge segments where f is a line) are fitted as lines,
    c2 = c3 = 0, over their codes and every code past them."""
    u = _u_clamped(site, lo, hi, flags)
    j, t, F = _locate(site, scheme, lo, hi, shift, flags)
    K = segment_count(scheme, lo, hi, shift, flags)
    C = np.zeros((K, 4), dtype=np.float32)
    order = np.argsort(j, kind="stable")
    bounds = np.searchsorted(j[order], np.arange(K + 1))
    w = site.w()
    for s in range(K):
        idx = order[bounds[s]:bounds[s + 1]]
        if len(idx) == 0:
            continue
        tf = t[idx] / np.exp2(F[idx].astype(np.float64))
        A = np.stack([np.ones_like(tf), tf, tf ** 2, tf ** 3], axis=1)
        if s in lines:
            A[:, 2:] = 0.0
        sw = np.sqrt(w[idx])
        coef, *_ = np.linalg.lstsq(A * sw[:, None], u[idx] * sw, rcond=None)
        high = coef[1:].astype(np.float32).astype(np.float64)
        if s in lines:
            high[1:] = 0.0
        resid = u[idx] - A[:, 1:] @ high
        c0 = float((w[idx] * resid).sum() / w[idx].sum())
        C[s] = [c0, high[0], high[1], high[2]]
    return np.clip(C, -COEF_MAX, COEF_MAX).astype(np.float32)


def refine_cubic(site, scheme, lo, hi, shift, C, flags=0, radius=48, step=1.0 / 64):
    """The cubic tables' flip fit: segments are independent, so for each the
    exact minimum of its weighted flips over c0 + k * step, |k| <= radius
    (codes), the device's arithmetic on every code of the segment (ties:
    the smaller sum of |error|, then the smaller shift)."""
    E = site.E()
    w = site.w()
    j, t, F = _locate(site, scheme, lo, hi, shift, flags)
    C = C.copy()
    F32 = np.float32
    tf = (t.astype(F32) * np.exp2(-F).astype(F32)).astype(F32)
    c1, c2, c3 = C[j, 1], C[j, 2], C[j, 3]
    h = (c2 + (tf * c3).astype(F32)).astype(F32)
    h = (c1 + (tf * h).astype(F32)).astype(F32)
    h = (tf * h).astype(F32)              # y = fl(c0 + h)
    order = np.argsort(j, kind="stable")
    bounds = np.searchsorted(j[order], np.arange(len(C) + 1))
    ks = np.arange(-radius, radius + 1)
    magic = qm.LUT16_MAGIC
    lo2, hi2 = site.outq.qmin, site.outq.qmax
    for s in range(len(C)):
        idx = order[bounds[s]:bounds[s + 1]]
        if len(idx) == 0:
            continue
        c0 = (C[s, 0] + (ks * step)).astype(F32)
        y = (c0[:, None] + h[idx][None, :]).astype(F32)
        r = ((y + magic).astype(F32) - magic).astype(F32)
        r = np.clip(r.astype(np.int64), lo2, hi2)
        miss = ((r != E[idx]) * w[idx]).sum(1)
        err = (np.abs(r - E[idx]) * w[idx]).sum(1)
        best = np.lexsort((np.abs(ks), err, miss))[0]
        C[s, 0] = c0[best]
    return np.clip(C, -COEF_MAX, COEF_MAX).astype(np.float32)


# refine_joint runs on a window's candidate only within this many times its
# bar (+ 8): a table further off does not get there (the reciprocal window
# at m = 3: 35 flips -> 15 against a bar of 3), and each search costs seconds.
JOINT_REACH = 4


def refine_joint(site, scheme, lo, hi, shift, C, flags=0, lines=(), radius=48,
                 step=1.0 / 128, c1_span=0.02, c1_steps=81):
    """A window's NSAU table's second chance (fit_scheme: floor_bar sites,
    NSAU cubic only -- the steep singular sites, where a uniform window's
    smaller tables would sit at the edge of the bar and flip more held-out
    codes): each segment that still flips after refine_cubic gets the exact
    minimum of its flips
    over c0 + k * step and c1 + d, |k| <= radius, |d| <= c1_span, the
    device's arithmetic on every code of the segment (ties: the smaller sum
    of |error|, then the smaller move). A window's bar is its float32 floor,
    and what misses it is often a code a hair from a rounding tie (u = k +
    0.5001): tilting the segment moves it, which c0 alone cannot without
    moving another. Rows in `lines` stay lines."""
    E = site.E()
    j, t, F = _locate(site, scheme, lo, hi, shift, flags)
    C = C.copy()
    F32 = np.float32
    magic = qm.LUT16_MAGIC
    lo2, hi2 = site.outq.qmin, site.outq.qmax
    tf_all = (t.astype(F32) * np.exp2(-F).astype(F32)).astype(F32)

    def evaluate(row, idx):
        tf = tf_all[idx]
        y = (row[2] + (tf * row[3]).astype(F32)).astype(F32)
        y = (row[1] + (tf * y).astype(F32)).astype(F32)
        y = (row[0] + (tf * y).astype(F32)).astype(F32)
        r = ((y + magic).astype(F32) - magic).astype(F32)
        return np.clip(r.astype(np.int64), lo2, hi2)

    ks = np.arange(-radius, radius + 1) * step
    ds = np.linspace(-c1_span, c1_span, c1_steps)
    order = np.argsort(j, kind="stable")
    bounds = np.searchsorted(j[order], np.arange(len(C) + 1))
    for s in range(len(C)):
        if s in lines:
            continue
        idx = order[bounds[s]:bounds[s + 1]]
        if len(idx) == 0:
            continue
        miss0 = int((evaluate(C[s], idx) != E[idx]).sum())
        if miss0 == 0:
            continue
        tf = tf_all[idx]
        h2 = (C[s, 2] + (tf * C[s, 3]).astype(F32)).astype(F32)
        best = (miss0, 0, 0.0, C[s, 0], C[s, 1])
        for d in ds:
            c1 = F32(C[s, 1] + d)
            h = (tf * (c1 + (tf * h2).astype(F32)).astype(F32)).astype(F32)
            c0 = (C[s, 0] + ks).astype(F32)
            y = (c0[:, None] + h[None, :]).astype(F32)
            r = ((y + magic).astype(F32) - magic).astype(F32)
            r = np.clip(r.astype(np.int64), lo2, hi2)
            miss = (r != E[idx][None, :]).sum(1)
            err = np.abs(r - E[idx][None, :]).sum(1)
            moves = np.abs(ks) / step + abs(d) / (c1_span / (c1_steps // 2))
            k = np.lexsort((moves, err, miss))[0]
            cand = (int(miss[k]), int(err[k]), float(moves[k]), c0[k], c1)
            if cand[:3] < best[:3]:
                best = cand
        C[s, 0], C[s, 1] = best[3], best[4]
    return np.clip(C, -COEF_MAX, COEF_MAX).astype(np.float32)


def _hat_system(site, scheme, lo, hi, shift, flags=0):
    j, t, F = _locate(site, scheme, lo, hi, shift, flags)
    a = t / np.exp2(F.astype(np.float64))
    return j, t, F, 1.0 - a, a


def fit_linear_lsq(site, scheme, lo, hi, shift, G, flags=0):
    """Least squares of the continuous piecewise-linear table on u, nodes in
    units of 2^-G codes."""
    from scipy.linalg import solveh_banded

    u = _u_clamped(site, lo, hi, flags)
    j, _, _, p0, p1 = _hat_system(site, scheme, lo, hi, shift, flags)
    N = node_count(scheme, lo, hi, shift, flags)
    w = site.w()
    diag = np.bincount(j, w * p0 * p0, N) + np.bincount(j + 1, w * p1 * p1, N)
    off = np.bincount(j, w * p0 * p1, N)[:N - 1]
    b = np.bincount(j, w * p0 * u, N) + np.bincount(j + 1, w * p1 * u, N)
    # nodes no code reaches (the negative half's node 0) follow a weak tie to
    # their neighbour instead of floating
    empty = diag == 0
    diag = diag + 1e-9 * diag.max() + empty * 1e-6
    ab = np.zeros((2, N))
    ab[0, 1:] = off
    ab[1, :] = diag
    V = solveh_banded(ab, b)
    span = 0.25 * (site.outq.qmax - site.outq.qmin)
    V = np.clip(V, site.outq.qmin - span, site.outq.qmax + span)
    return np.rint(V * (1 << G)).astype(np.int64)


def node_count(scheme, lo, hi, shift, flags=0):
    """Nodes of a linear table (both NSAU node chains, with OFFSET each as
    long as its half's segments)."""
    return qm.lut16_entries(SCHEME_ID[scheme] | flags, lo, hi, shift)


def safe_G(site, scheme, lo, hi, shift, flags=0, most=8):
    """The largest node precision G <= `most` for which the device's int32
    interpolation cannot overflow, judged on a G = 0 least-squares fit (flat
    segments allow wide ones). linear_safe re-checks the final
    integer nodes exactly."""
    V = fit_linear_lsq(site, scheme, lo, hi, shift, 0, flags).astype(np.float64)
    j, _, F = _locate(site, scheme, lo, hi, shift, flags)
    S = len(V) - 1          # segment slots; the link between NSAU halves is empty
    W = np.zeros(S)
    W[j] = np.exp2(F)
    if flags & TAIL:
        below, above = qm.lut16_tail_reach(lo, hi, shift)
        W[0], W[-1] = max(W[0], below + 1.0), max(W[-1], above + 1.0)
    seg = np.arange(S)
    d = np.abs(V[seg + 1] - V[seg]) + 2.0
    for G in range(most, -1, -1):
        ok = (d * 2.0 ** G * W).max() < 2 ** 30
        if scheme == "uniform_linear":
            ok = ok and ((np.abs(V[seg]) + 2) * 2.0 ** G * np.exp2(shift)).max() < 2 ** 30
        if ok:
            return G
    return 0


def linear_safe(scheme, V, lo, hi, shift, G, flags=0):
    """Whether every intermediate of the interpolation fits int32 for every
    segment (the runtime's own check, quant_lut16.cpp)."""
    return qm.lut16_linear_safe(SCHEME_ID[scheme] | flags, V, lo, hi, shift, G)


def fit_viterbi(site, scheme, lo, hi, shift, G, V0, flags=0, radius=3, passes=6,
                tie=1e-6):
    """Minimise the flips over integer nodes V0 + step * d, |d| <= radius, by
    dynamic programming along the node chain (a segment's cost depends on its
    two nodes only), coarse to fine."""
    steps = sorted({max(1, 1 << max(G - 4, 0)), max(1, 1 << max(G - 6, 0)), 1},
                   reverse=True)
    E = site.E()
    w = np.ones(len(E)) if site.weights is None else np.asarray(site.weights, np.float64)
    j, t, F = _locate(site, scheme, lo, hi, shift, flags)
    order = np.argsort(j, kind="stable")
    js, ts, Es, Fs, ws = j[order], t[order], E[order], F[order], w[order]
    N = len(V0)
    K = N - 1
    bounds = np.searchsorted(js, np.arange(K + 1))
    uniform = scheme == "uniform_linear"
    V = np.asarray(V0, dtype=np.int64).copy()
    for step in steps:
        deltas = step * np.arange(-radius, radius + 1, dtype=np.int64)
        R = len(deltas)
        for _ in range(passes):
            cand = V[:, None] + deltas[None, :]
            cost = np.zeros((K, R, R))
            for s in range(K):
                a, b = bounds[s], bounds[s + 1]
                if a == b:
                    continue
                tt, ee, ww, Fseg = ts[a:b], Es[a:b], ws[a:b], int(Fs[a])
                v0 = cand[s][:, None, None]
                v1 = cand[s + 1][None, :, None]
                y = _interp(v0, v1, tt[None, None, :], Fseg, G, uniform)
                y = np.clip(y, site.outq.qmin, site.outq.qmax)
                miss = y != ee[None, None, :]
                cost[s] = (miss * ww).sum(-1) + tie * (np.abs(y - ee) * (ww + 1e-3)).sum(-1)
            acc = np.zeros(R)
            back = np.zeros((K, R), dtype=np.int64)
            for s in range(K):
                tot = acc[:, None] + cost[s]
                back[s] = np.argmin(tot, axis=0)
                acc = tot[back[s], np.arange(R)]
            sel = np.zeros(N, dtype=np.int64)
            sel[-1] = int(np.argmin(acc))
            for s in range(K - 1, -1, -1):
                sel[s] = back[s][sel[s + 1]]
            Vn = cand[np.arange(N), sel]
            if np.array_equal(Vn, V):
                break
            V = Vn
    return V


def _interp(v0, v1, t, F, G, uniform):
    if uniform:
        sh = F + G
        rnd = (1 << (sh - 1)) if sh > 0 else 0
        return ((v0 << F) + (v1 - v0) * t + rnd) >> sh
    rnd = (1 << (G - 1)) if G > 0 else 0
    return (v0 + (((v1 - v0) * t) >> F) + rnd) >> G


def fit_scheme(site, scheme, size=None, lo=None, hi=None, G=None, viterbi=True,
               refine=True, flags=None, lines=()):
    """One scheme at one size over [lo, hi] (default: the input grid's whole
    range), scored against E. size: K segments for uniform tables (the
    smallest 2^F covering the range in K), m for NSAU; ignored for direct.
    flags: OFFSET or TAIL (mirrors.lut16_index), default OFFSET for an
    NSAU table it makes smaller; `lines`, a TAIL table's edge rows fitted as
    lines (fit_cubic)."""
    if lo is None or hi is None:
        alo, ahi = active_range(site)
        lo = alo if lo is None else lo
        hi = ahi if hi is None else hi
    oq = site.outq
    if scheme == "direct":
        E = site.E()
        data = E[lo - site.inq.qmin:hi - site.inq.qmin + 1].astype(np.int16)
        return _score(Table(scheme, lo, hi, 0, 0, data, oq.qmin, oq.qmax), site)
    if scheme.startswith("nsau"):
        lo, hi = max(lo, -32767), min(hi, 32767)
        shift = 3 if size is None else int(size)
        if flags is None:
            flags = nsau_flags(lo, hi, shift, scheme)
    else:
        shift = uniform_shift(lo, hi, 128 if size is None else int(size))
    flags = flags or 0
    if scheme.endswith("cubic"):
        data = fit_cubic(site, scheme, lo, hi, shift, flags, lines)
        if refine:
            data = refine_cubic(site, scheme, lo, hi, shift, data, flags)
        table = _score(Table(scheme, lo, hi, shift, 0, data, oq.qmin, oq.qmax,
                             flags=flags), site)
        if (refine and site.floor_bar and scheme == "nsau_cubic" and not within_bar(table)
                and table.flips <= JOINT_REACH * table.bar + 8):
            data = refine_joint(site, scheme, lo, hi, shift, data, flags, lines)
            table = _score(Table(scheme, lo, hi, shift, 0, data, oq.qmin, oq.qmax,
                                 flags=flags), site)
        return table
    Gs = [safe_G(site, scheme, lo, hi, shift, flags)] if G is None else [G]
    while True:
        g = Gs[-1]
        V = fit_linear_lsq(site, scheme, lo, hi, shift, g, flags)
        if viterbi:
            V = fit_viterbi(site, scheme, lo, hi, shift, g, V, flags)
        if linear_safe(scheme, V, lo, hi, shift, g, flags) or g == 0:
            break
        Gs.append(g - 1)
    if not linear_safe(scheme, V, lo, hi, shift, Gs[-1], flags):
        raise ValueError(f"{scheme} at {size}: no int32-safe node precision")
    table = Table(scheme, lo, hi, shift, Gs[-1], V.astype(np.int32), oq.qmin, oq.qmax,
                  flags=flags)
    return _score(table, site)


# --- the tail -----------------------------------------------------------------------

# A tail run shorter than this is not worth a table edge.
MIN_TAIL = 64


def _line_exact(site, a, b, anchor):
    """Whether one line, evaluated as the device evaluates a TAIL table's edge
    line (y = fl(c0 + fl(t * c1)), t = c - anchor, rounded half to even and
    clamped to the output rails; a 2^F-wide segment's tf and c1 scale
    exactly), gives E on every code of [a, b] but at most those on the
    float32 floor there: least squares on u, float32, then c0's flip fit."""
    F32 = np.float32
    sel = slice(a - site.inq.qmin, b - site.inq.qmin + 1)
    E = site.E()[sel]
    u = site.u()[sel]
    t = (site.codes[sel] - anchor).astype(np.float64)
    A = np.stack([np.ones_like(t), t], axis=1)
    (c0, c1), *_ = np.linalg.lstsq(A, u, rcond=None)
    c1 = F32(c1)
    h = (t.astype(F32) * c1).astype(F32)
    base = F32(np.mean(u - t * np.float64(c1)))
    ks = np.arange(-16, 17)
    magic = qm.LUT16_MAGIC
    mask = site.floor_mask()
    allowed = int(mask[sel].sum()) if mask is not None else 0
    for k in ks[np.argsort(np.abs(ks), kind="stable")]:
        y = (F32(base + F32(k / 64.0)) + h).astype(F32)
        r = ((y + magic).astype(F32) - magic).astype(F32)
        r = np.clip(r.astype(np.int64), site.outq.qmin, site.outq.qmax)
        if int((r != E).sum()) <= allowed:
            return True
    return False


def linear_ends(site):
    """(L, R): E is a line (_line_exact) on [qmin, L] and on [R, qmax] of the
    site's input grid, each the longest such run found by bisection on its
    inner end (a run inside an exact run is exact), or None for a side with
    no run of MIN_TAIL codes. Where f becomes x, 0 or an output rail toward
    a rail of q1 -- GELU, hardswish, x^2 -- a TAIL table can stop at the run
    and let its edge segment, that line, continue."""
    key = "linear_ends"
    if key in site._cache:
        return site._cache[key]
    qmin, qmax = site.inq.qmin, site.inq.qmax
    ends = [None, None]
    if qmax - qmin + 1 >= 4 * MIN_TAIL:
        # the right run [R, qmax]: the smallest R
        lo_b, hi_b = qmin + MIN_TAIL, qmax - MIN_TAIL + 1
        if _line_exact(site, hi_b, qmax, hi_b):
            while hi_b - lo_b > 1:
                mid = (lo_b + hi_b) // 2
                if _line_exact(site, mid, qmax, mid):
                    hi_b = mid
                else:
                    lo_b = mid
            ends[1] = hi_b
        # the left run [qmin, L]: the largest L
        lo_b, hi_b = qmin + MIN_TAIL - 1, qmax - MIN_TAIL
        if _line_exact(site, qmin, lo_b, lo_b):
            while hi_b - lo_b > 1:
                mid = (lo_b + hi_b) // 2
                if _line_exact(site, qmin, mid, mid):
                    lo_b = mid
                else:
                    hi_b = mid
            ends[0] = lo_b
    site._cache[key] = tuple(ends)
    return site._cache[key]


def fit_tail(site, size):
    """The uniform cubic TAIL table at `size` (the plain table's segment width
    at that size, so the two compare at one resolution), or None when it is
    not smaller than the plain one or cannot be built. Codes past a TAIL
    table are not clamped, so each end of q1's codes needs one of: a linear
    run (linear_ends), where the table's edge segment is the segment that
    ends (left) or starts (right) on the grid inside the run, fitted as the
    run's line, which continues to the rail; or the table reaching q1's end
    itself (no code past it). An end active_range cut at an output rail
    without a run is the second kind's failure: None."""
    if not site.tails:
        return None
    qmin, qmax = site.inq.qmin, site.inq.qmax
    alo, ahi = active_range(site)
    L, R = linear_ends(site)
    F = uniform_shift(alo, ahi, size)
    W = 1 << F
    lines = set()
    if L is not None and L - W + 1 >= qmin and L - W + 1 > alo - W:
        lo = L - W + 1
        lines.add(0)
    elif alo == qmin:
        lo = alo
    else:
        return None
    if R is not None:
        last = lo + ((max(R - lo, 0) + W - 1) >> F << F)
        lines.add(-1)
    elif ahi == qmax:
        last = lo + (((ahi - lo) >> F) << F)
    else:
        return None
    if not lines or last < lo:
        return None
    hi = min(last + W - 1, 32767)
    K = ((hi - lo) >> F) + 1
    if 16 * K >= table_bytes("uniform_cubic", alo, ahi, F):
        return None
    rows = {0 if r == 0 else K - 1 for r in lines}
    # K segments over [lo, hi] are 2^F wide (uniform_shift; the last may
    # reach past 32767), the plain table's width at `size`
    table = fit_scheme(site, "uniform_cubic", K, lo=lo, hi=hi, flags=TAIL, lines=rows)
    return table if table.shift == F else None


# --- the chooser ------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    """What fit() may spend: `budget` bytes of table, flips within the bar.
    `scheme` (and `size`) force one candidate, still scored."""

    budget: int = DEFAULT_BUDGET
    scheme: Optional[str] = None
    size: Optional[int] = None


def candidates(site, spec=Spec()):
    """(scheme, size) in the chooser's order, each within the budget: direct
    for ranges up to DIRECT_MAX codes; then the primary cubic scheme (NSAU
    for SINGULAR functions, uniform otherwise) by increasing size; then the
    other cubic scheme. NSAU sizes count OFFSET where it is smaller."""
    lo, hi = active_range(site)
    if spec.scheme is not None:
        yield spec.scheme, spec.size
        return
    if hi - lo + 1 <= DIRECT_MAX and table_bytes("direct", lo, hi, 0) <= spec.budget:
        yield "direct", None
        return
    order = (["nsau_cubic", "uniform_cubic"] if site.fn in SINGULAR
             else ["uniform_cubic", "nsau_cubic"])
    for scheme in order:
        if scheme == "uniform_cubic":
            sizes = [(K, uniform_shift(lo, hi, K)) for K in UNIFORM_SIZES]
        else:
            sizes = [(m, m) for m in NSAU_SIZES]
        seen = set()
        for size, shift in sizes:
            nlo, nhi = (max(lo, -32767), min(hi, 32767)) if scheme == "nsau_cubic" else (lo, hi)
            flags = nsau_flags(nlo, nhi, shift) if scheme == "nsau_cubic" else 0
            nbytes = table_bytes(scheme, nlo, nhi, shift, flags)
            if nbytes > spec.budget or (shift, nbytes) in seen:
                continue
            seen.add((shift, nbytes))
            yield scheme, size


def fit(site, spec=Spec()):
    """The chooser: the first candidate whose flips are within the bar. Raises
    BudgetExceeded when none is (or none fits the budget at all). A uniform
    cubic candidate of a site the tail rule names is tried first with
    TAIL (fit_tail), which is smaller where it exists."""
    best = None
    for scheme, size in candidates(site, spec):
        tries = []
        if scheme == "uniform_cubic" and spec.scheme is None:
            tries.append(lambda: fit_tail(site, size))
        tries.append(lambda: fit_scheme(site, scheme, size))
        for attempt in tries:
            table = attempt()
            if table is None or table.nbytes > spec.budget:
                continue
            if within_bar(table):
                return table
            if best is None or (table.flips, table.nbytes) < (best.flips, best.nbytes):
                best = table
    raise BudgetExceeded(
        f"lut16-budget: no scheme within {spec.budget} B meets the bar"
        + (f" (best {best.describe()})" if best is not None else ""), best)


def fit_smallest(site, spec=Spec()):
    """The smallest table within the bar among every candidate the budget
    allows -- direct over any range, uniform cubic and NSAU cubic at every
    size -- tried in order of bytes (ties: direct, which is exact, then
    uniform, then NSAU). For a site fitted over an observed window
    (fold_lut16_pass.fit_parts), where fit()'s graph-only order
    is no guide: a window of a few hundred codes makes direct exact but
    often larger than a cubic (x^2), and a window away from 0 makes uniform
    segments cheaper than NSAU for a singular function (reciprocal) -- or,
    with OFFSET, NSAU segments from the window's first one. A forced
    spec.scheme is fit()'s. Raises BudgetExceeded as fit() does."""
    if spec.scheme is not None:
        return fit(site, spec)
    lo, hi = active_range(site)
    nlo, nhi = max(lo, -32767), min(hi, 32767)
    cands = [("direct", None, table_bytes("direct", lo, hi, 0), 0)]
    cands += [("uniform_cubic", K, table_bytes("uniform_cubic", lo, hi,
                                                uniform_shift(lo, hi, K)), 1)
              for K in UNIFORM_SIZES]
    cands += [("nsau_cubic", m, table_bytes("nsau_cubic", nlo, nhi, m,
                                            nsau_flags(nlo, nhi, m)), 2)
              for m in NSAU_SIZES]
    seen, best = set(), None
    for scheme, size, nbytes, _ in sorted(cands, key=lambda c: (c[2], c[3])):
        shift = (0 if scheme == "direct" else uniform_shift(lo, hi, size)
                 if scheme == "uniform_cubic" else size)
        if nbytes > spec.budget or (scheme, shift, nbytes) in seen:
            continue
        seen.add((scheme, shift, nbytes))
        table = fit_scheme(site, scheme, size)
        if table.nbytes > spec.budget:
            continue
        if within_bar(table):
            return table
        if best is None or (table.flips, table.nbytes) < (best.flips, best.nbytes):
            best = table
    raise BudgetExceeded(
        f"lut16-budget: no scheme within {spec.budget} B meets the bar"
        + (f" (best {best.describe()})" if best is not None else ""), best)


def site_for(fn, s1, z1, s2, z2, qmin=-32767, qmax=32767, out_qmin=None, out_qmax=None,
             **kwargs):
    """A Site for an allowlisted activation between two int16 grids, f from
    torch (Arm a16w8's site is site_for(fn, 2**-12, 0, 2**-12, 0))."""
    return Site(fn, Grid(float(s1), int(z1), qmin, qmax),
                Grid(float(s2), int(z2), qmin if out_qmin is None else out_qmin,
                     qmax if out_qmax is None else out_qmax), kwargs=kwargs)
