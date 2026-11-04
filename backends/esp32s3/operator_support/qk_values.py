# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The q@kT value check: an operand the sample runs show on a grid stays
portable, whatever the spelling that made it.

qk_operands.origin() admits a q@kT operand by a proof over the graph: every
path from it ends in plain float computation. Its lists name the spellings
of a grid it knows. A grid spelled some other way -- a magnitude written
maximum(x, -x), a table looked up through take, codes times a per-channel
scale on a one-row operand, a grid behind exp -- passes the rules as plain,
and OP_BMM_DYN_I8 would requantise a value the model already rounded. A
list closes such spellings one at a time and leaves a long tail (generated
grids that pass the rules alone are easy to find).

So the rules are not the only judge. Behind them, on every q@kT the
rules would delegate, this module runs the program on a few input sets and
looks at the operands the delegate would quantise itself (every query, and
the key under OP_BMM_DYN_I8). If the values show a grid, the bmm stays
portable on the converted model's bits. Values never admit: an operand the
rules refuse stays refused, and an operand the values cannot judge is left
to the rules.

The input sets, drawn with a local numpy generator from a fixed seed (SEED;
numpy, since torch's randn gives other values on x86 and ARM), never the
global RNG (the program's own random ops run on the CPU
generator, seeded inside fork_rng(devices=[]) and restored afterwards; no
device's generator is touched -- torch.manual_seed would seed CUDA's and
MPS's as well):

  example  the program's example inputs (lower_esp32 hands the exported
           program's to the partitioner; to_edge drops them), when there
           are any;
  draws    DRAWS more sets, of each input's shape and dtype:
             floats     unit normal, fresh in every set;
             token ids  the integer inputs that index a table (through
                        moves, the indices of aten.embedding, or of a
                        gather, index_select or index of a parameter) and
                        index or write no other tensor: uniform over the
                        table's rows, fresh in every set;
             others     positions (an integer input that also writes a
                        cache or indexes a buffer, whatever learned table
                        it picks rows of: _token_uses), cache indices,
                        masks, bools: the example's values (zeros without
                        an example).
           Up to three regimes, each judged on its own: the floats varying
           with the rest held (what a run varies is the value a grid is
           made from); the token ids varying with
           the floats held -- a table looked up at them shows its own values
           from run to run, so the tests that count values or rows over all
           runs (levels, vector levels, scaled levels) are not tried there;
           and the rows aten.embedding looks up at the token ids drawn unit
           normal in every run with the ids held (EMBEDDINGS_VARY), where
           every test runs: a program with no float input -- an LLM's
           token ids, positions and cache -- is judged on floats there.
           A program with no kind to vary is left to the rules.

Each set is one run of the program under no_grad (an fx interpreter over
the partitioner's graph, parameters and buffers read through
const_weights._concrete), and only when a q@kT the rules would delegate
exists; the runs are made once per program, for every gate that asks
(for_program). Written buffers (K and V caches, any state the forward
writes) run on copies, carried from one run to the next within a regime
after one unrecorded warm-up run, so a value the forward stores reaches an
operand that reads it the next call. Which rows hold only a written
buffer's initial content is decided per row, from values: where a written
floating-point buffer reaches a candidate, the regime runs twice more. The
redraw -- run 0's inputs, after a warm-up on them too, with that initial
content drawn afresh -- finds the unwritten rows: a row the same in every
run that the redraw moves holds content no run wrote (zeros, a constant
fill, rows from before), and is left out; every other row is judged with
every test, a value computed this call from content no run wrote included
(a query past a layer that attends over a cache's unwritten rows). The
replay -- run 0's inputs on the state that enters run 1 -- finds the rows
that depend on state an earlier call wrote (a cache shifted a row a call,
a cache written at a counter the forward advances, a previous call's
write, a state the forward scales each call): from values such state
cannot be told from a fresh value -- a state the forward halves each call
has amplitudes on one step, a shifted cache's rows come back a place lower
-- so a candidate with any such row that is no row of a written buffer (a
value computed from that state) is not judged: the gate keeps its q@kT
portable (state_read), speed only. A cache's own rows read through moves
are judged with every test (_row_classes). Which rows are a written
buffer's own is decided by value, for every row: a row whose run-0 value
is bit for bit a row a written buffer held entering run 0 (the replay
reaches one call, and a counter cache's rows that later runs write are
past it); the others are fresh. Each kind keeps up to MAX_ROWS rows that
change from run to run (_keep_rows), and the rows the replay's order keeps
(the others first, then the rows the replay changes, MAX_ROWS in all) are
kept too. An operand with rows of more than one kind is
judged on each group alone as well as on all its rows together, each with
every test: its fresh rows, its buffer rows, the rows the replay leaves in
place and the rows kept the replay's way (the last two both drawn from
the replay-ordered selection); a grid in any is a grid
(judge_regime), since rows of one kind can hide a grid in the other (a
plain ring an earlier call wrote beside a fresh grid, a ring holding a grid
beside plain fresh keys). Integer and bool
buffers are not redrawn (the replay restores them as the runs left them):
to the redraw, a row they alone reach stays fresh, judged. When the
redrawn or the replayed run raises, the rows cannot be classed, so every
candidate it tracked is left unjudged (unjudged) and the gate keeps its
q@kT portable, logged -- judging them all as fresh could hide a grid among
rows the classes would have set apart. So does a
failure to read the rows the written buffers held entering run 0 (a buffer
with no elements holds none: _held_rows) or to class a candidate's rows:
it leaves those candidates unjudged, never the rules alone.
Non-finite values are left out (a row holding one); an operand with no
finite row gets
no verdict, and the partition log names each q@kT the values gave no
verdict on, and why. A regime whose runs raise on drawn inputs is that
kind held -- not run, and logged -- while the others judge; when every
regime raises, or the program cannot be run at all, the rules alone decide
and the log says why (one run of the example inputs alone cannot judge:
judge needs runs that differ). The twelve runs of a regime are judged
whole and in halves of WINDOW (judge_runs): a test with a ceiling (LEVELS,
SCALED_LEVELS) sees fewer rare values in six runs, one that needs evidence
more in twelve.

Bounded, so that a large program costs a refusal, never the process: each
run records at most MAX_ROWS rows of each candidate as it goes, evenly
spaced, the same in every run -- of one a written buffer reaches, all its
rows, and once its rows are classed up to three times MAX_ROWS of them:
MAX_ROWS that change from run to run of each kind, others to MAX_ROWS in
all, and the rows the replay's order keeps (_keep_rows), so a decode
step's one row of a long cache is kept -- the input sets
are drawn as the runs need them, a candidate's samples are freed once
judged, and every sample when the partition returns (release): only the
verdicts stay. Past MAX_BYTES recorded or
MAX_SECONDS of runs, no candidate is judged and every q@kT the check would
judge stays portable, logged (exact is safe; a grid not judged must not be
admitted). On the story Glimmer's shape, two regimes of thirteen runs a
method take 2.9 s of a 17.0 s lowering of both methods (an x86 Linux host)
and 16 MB of a 640 MB peak (8.6 MB of samples); the redrawn and replayed
runs a written buffer adds to each regime take about 0.4 s and 30 MB more
(on a Mac).

judge() decides grid from values alone, with these tests:

  no verdict  every run the same values (the operand does not depend on what
              varies), or every value within 0.1% of one number;
  plain       every row one vector up to its own scale, or each row
              position one vector in every run (a constant pattern times a
              run-time scale is not a grid made from the inputs) -- unless
              the rows' amplitudes are themselves on few levels or one
              step (_amplitude_grid: exp(round(x[..., :1])) expanded to the
              row is a grid);
  grid when
    levels         every value, over all runs, is one of at most LEVELS
                   levels (values within 4 ulps one level), the runs'
                   distinct counts summing to twice that, and two of the
                   levels seen in two runs or more;
    row levels     each row holds at most ROW_LEVELS distinct values, half
                   the rows two or more, and at most a quarter of the rows
                   more than one level of a single value;
    vector levels  the rows are at most VECTOR_LEVELS fixed vectors over
                   all runs, each run showing them again (a codebook);
    scaled levels  each row divided by its largest-magnitude value takes
                   few values over all rows and runs, leaving out zeros and
                   the element divided by (at most SCALED_LEVELS, a third of
                   the elements of the distinct rows and of the rows'
                   distinct values summed): a table or codes times a row's
                   run-time scale;
    lattice        in one grouping, every group on an affine lattice
                   v0 + k * s, integers 0 <= k <= K_MAX (K_MAX_SMALL for
                   any group of fewer than SMALL_GROUP distinct values, so
                   every group of GS channels: a handful of values lies on
                   some lattice of k <= K_MAX almost always), each value
                   within ULPS ulps of max(|v|, |v0|, 1) -- values that
                   close are one point (_merged: one value two kernels
                   compute a rounding apart) -- the step at least MIN_STEP
                   times that tolerance (float32's own granularity is no
                   step); a quarter of the groups holding three or more
                   distinct values. Groupings: each row (row lattice),
                   each group of GS channels of a row (group lattice), each
                   channel of a run (channel lattice), each run (tensor
                   lattice), each channel pooled over all runs (channel
                   lattice across inputs: a constant per-channel step, the
                   one-row operand's scale), everything pooled (tensor
                   lattice across inputs), and each element across the runs
                   (element lattice across inputs: a constant per-element
                   step).

The tests' sparse-value limits are three: the levels a level count needs
seen again (zeros and a few unrelated values are not levels), the row levels
a sparse row needs repeated, and the small-group bound. A very sparse
operand (a relu past two standard deviations) can still pass a lattice by
chance and is refused: speed only.

The contract this gives (README, q@kT): what keeps the converted model's
bits is any operand that is a grid by the rules or by values on the sample
runs, a grid behind an op that computes included (exp(round(x)), a
saturation landing on few levels). The residual: a grid the rules do not
name that the sample runs do not show -- a grid only on inputs unlike the
samples (a range the draws never reach, an id the draws never pick); an
operand at one of these tests' thresholds; and the tests' measured limits.
At a threshold -- a saturation whose rows are about a quarter on a grid
(the 25% richness bar), a level count at the LEVELS ceiling -- an operand
can fall either way between the gate's sample runs and any other set of
samples of the same inputs. Of 29,330 generated grids in scope, 18 were
admitted, none of them a grid judge() finds on all of eight other seed sets
(each on zero to six); with inputs ten times larger and smaller, 104 of
29,389, none a grid on all of eight unit-normal seed sets -- inputs unlike
the samples. The
limits: a per-element step that varies with the input
(round(x / s) * s with s = |x| / 2 + 0.1 has no lattice in any grouping),
code ranges past K_MAX (past K_MAX_SMALL in a group of fewer than eight
distinct values), codebook rows on a one-row operand, rows that only a
written buffer's initial content reaches (which no run wrote: left out),
and two the buffer kind by value brings: example inputs that make a fresh
row equal to a row a written buffer holds (all zeros beside a cache's zero
rows) make it the buffer's own, where the cache's plain rows can hide its
grid (a counter cache beside fresh keys on a 0/1 grid, with example rows
that zero them: export with representative example inputs; a 0/1 grid also
makes all-zero rows by itself, whatever the example inputs, so
representative inputs do not remove that case); and a cache
whose starting content is non-zero and plain dilutes the rows later runs
write (real caches start at zero; ExecuTorch keeps no
initial content for a mutated buffer unless InitializedMutableBufferPass
marks it). Such an operand is
a value the model computes,
requantised to int8 rows within the mode's error.

That measurement is partly circular: it judged the generated grids with
the same tests. test_qk_matrix's hand-written origins and the
nano zoo's real operands are the independent evidence.
"""

import time
import weakref
from typing import NamedTuple, Optional

import numpy as np
import torch

# --- judge()'s thresholds ---
LEVELS = 16
ROW_LEVELS = 3
VECTOR_LEVELS = 64
SCALED_LEVELS = 128
K_MAX = 4096
K_MAX_SMALL = 255          # groups of a few values (GS channels, one element)
SMALL_GROUP = 8            # fewer distinct values: held to K_MAX_SMALL
ULPS = 16.0
MIN_STEP = 16.0
GS = 4
CLUSTER_ULPS = 4.0         # values this close are one level
DEGENERATE = 1e-3          # all values within this fraction of one number
NORM_BIN = 1e5             # a row divided by its maximum, binned at 1e-5

# --- the sample runs --------------------------------------------------------
SEED = 1729
DRAWS = 11                 # with the example inputs, twelve runs a regime
WINDOW = 6                 # judged whole and in halves of six
MAX_ROWS = 512             # rows of a run judged (evenly spaced past this)
MAX_BYTES = 256 << 20      # recorded sample bytes per program, at most
MAX_SECONDS = 120.0        # the runs of one program, at most

FLOATS_VARY = "floats vary"
IDS_VARY = "token ids vary"
EMBEDDINGS_VARY = "rows at token ids drawn"
LATTICES = ("row lattice", "group lattice", "channel lattice",
            "tensor lattice", "channel lattice across inputs",
            "tensor lattice across inputs", "element lattice across inputs")
# Off switches, one per test, for the mutant checks: a name here is never
# tried. Never set in a lowering.
_DISABLED = set()


def _spacing(v):
    return np.spacing(np.asarray(v, dtype=np.float32)).astype(np.float64)


def _lattice_steps(d, tol, kmax, dmin, dmax, cands=None):
    """Steps s (dmin / j, refined by least squares) on which every
    difference d is k * s, k <= kmax, within tol."""
    if cands is None:
        # A conservative bound: codes spanning exactly kmax steps measure a
        # hair over kmax after a division (a static pair over a table row,
        # one code per element over a constant), so the floor of
        # kmax * dmin / dmax can drop the true divisor of dmin -- one, or more
        # when the smallest observed gap is several codes. One candidate past
        # the bound, with slack for the endpoints' rounding; the residual and
        # k <= kmax checks below reject the extra ones.
        jmax = min(kmax, int(np.floor(kmax * dmin / dmax * (1 + 1e-4))) + 1)
        cands = dmin / np.arange(1, jmax + 1, dtype=np.float64)
    K = np.rint(d[None, :] / cands[:, None])        # (J, n)
    kk = np.sum(K * K, axis=1)
    s = np.sum(K * d[None, :], axis=1) / np.maximum(kk, 1.0)
    res = np.abs(d[None, :] - K * s[:, None])
    big = s >= MIN_STEP * float(np.max(tol))
    ok = np.all(res <= tol[None, :], axis=1) & (np.max(K, axis=1) <= kmax) & big
    return cands[ok]


def _lattice_group(u, kmax):
    """u: sorted distinct float64 values (from float32) of one group: True
    when u - u[0] = k * s for integers k <= kmax, within tolerance. A large
    group is screened on 64 of its differences first (a lattice holds on
    every subset), bounding the work."""
    if len(u) <= 2:
        return True
    d = u[1:] - u[0]
    tol = ULPS * _spacing(np.maximum(np.maximum(np.abs(u[1:]), abs(u[0])), 1.0))
    dmin = float(d[0])
    if dmin <= tol[0]:
        return False
    dmax = float(d[-1])
    if len(d) > 64:
        pick = np.linspace(0, len(d) - 1, 64).astype(int)
        cands = _lattice_steps(d[pick], tol[pick], kmax, dmin, dmax)
        if len(cands) == 0:
            return False
        for c0 in cands[:64]:
            if len(_lattice_steps(d, tol, kmax, dmin, dmax,
                                  cands=np.array([c0]))):
                return True
        return False
    return bool(len(_lattice_steps(d, tol, kmax, dmin, dmax)))


def _merged(u):
    """Sorted distinct values with neighbours within ULPS ulps of a run's
    first value merged into it (no chaining): one value computed by two
    kernels a rounding apart -- the partitioner's program and eager torch
    can compute two equal products so -- is one point of a lattice, not a
    step of one ulp."""
    if len(u) <= 1:
        return u
    tol = ULPS * _spacing(np.maximum(np.abs(u), 1.0))
    keep = [0]
    for i in range(1, len(u)):
        if u[i] - u[keep[-1]] > tol[keep[-1]]:
            keep.append(i)
    return u[keep]


def _grouping(groups, count, kmax=K_MAX):
    """Every group on a lattice, and a quarter of them holding three or
    more distinct values (a group of one value is on every lattice). A group
    of fewer than SMALL_GROUP distinct values is held to K_MAX_SMALL: a
    handful of values lies on some lattice of k <= K_MAX almost always
    (three nearly surely; four in a row of sixteen, repeated, near 7 in
    spread group scales), so it cannot tell a step that large from
    chance, whatever the group's size."""
    if count == 0:
        return False
    rich = 0
    for g in groups:
        u = _merged(np.unique(np.asarray(g, dtype=np.float32).astype(np.float64)))
        if len(u) >= 3:
            rich += 1
        k = min(kmax, K_MAX_SMALL) if len(u) < SMALL_GROUP else kmax
        if not _lattice_group(u, k):
            return False
    return rich * 4 >= count


def _levels(values, cap):
    """The clusters' first values (sorted), as _clusters merges them, or
    None past `cap` clusters."""
    u = np.unique(np.asarray(values, dtype=np.float32)).astype(np.float64)
    if len(u) > cap * 64:
        return None
    starts = [u[0]] if len(u) else []
    for v in u[1:]:
        if v - starts[-1] > CLUSTER_ULPS * float(
                _spacing(max(abs(v), abs(starts[-1])))):
            starts.append(v)
            if len(starts) > cap:
                return None
    return np.asarray(starts)


def _clusters(values, cap, counts=False):
    """Distinct values, neighbours within CLUSTER_ULPS ulps of a cluster's
    first value merged (the rounding of x / sqrt(x * x + eps) around 1 is
    one level; no chaining); counting stops past `cap`. With `counts`,
    (clusters, how many of them hold a single value)."""
    u, c = np.unique(np.asarray(values, dtype=np.float32), return_counts=True)
    u = u.astype(np.float64)
    if len(u) <= 1:
        return (len(u), int(np.sum(c == 1))) if counts else len(u)
    if len(u) > cap * 64:      # more distinct values than cap clusters hold
        return (cap + 1, 0) if counts else cap + 1
    n, start, size, single = 1, u[0], int(c[0]), 0
    for v, k in zip(u[1:], c[1:]):
        if v - start > CLUSTER_ULPS * float(_spacing(max(abs(v), abs(start)))):
            single += size == 1
            n += 1
            if n > cap:
                return (n, single) if counts else n
            start, size = v, 0
        size += int(k)
    single += size == 1
    return (n, single) if counts else n


def _amplitude_grid(amps, runs):
    """Whether the row amplitudes of a scaled constant (each row's element of
    largest magnitude, sign included; `runs` the run of each) are themselves
    on a grid: few levels, two of them seen again in other runs -- a run-time
    scale per run gives each run its own amplitude, one level a run, which is
    no grid -- or one lattice of SMALL_GROUP distinct values or more. The
    words, or None."""
    amps = np.asarray(amps, dtype=np.float32)
    runs = np.asarray(runs)
    if len(amps) < 3:
        return None
    starts = _levels(amps, LEVELS)
    if starts is not None and len(starts) >= 2:
        hit = np.searchsorted(starts, amps.astype(np.float64), "right") - 1
        seen = np.array([len(np.unique(runs[hit == i])) for i in
                         range(len(starts))])
        if len(amps) >= 2 * len(starts) and int(np.sum(seen >= 2)) >= 2:
            return f"on {len(starts)} levels"
    # One step only with SMALL_GROUP distinct amplitudes or more: three to
    # seven values lie on some 255-step lattice by chance often enough (an
    # expanded one-row state buffer's few run amplitudes do, in
    # test_qk_matrix), and few levels are the branch above's.
    u = _merged(np.unique(amps.astype(np.float64)))
    if len(u) >= SMALL_GROUP and _lattice_group(u, K_MAX):
        return "on one step"
    return None


def _distinct_per_row(rows):
    s = np.sort(rows, axis=1)
    return 1 + np.count_nonzero(np.diff(s, axis=1), axis=1)


def _normalised(rows, at_max=False):
    """Each row divided by its largest-magnitude element (sign included),
    binned at 1/NORM_BIN; rows of zeros left out: (int array, kept mask),
    and with `at_max` each kept row's index of that element."""
    r64 = rows.astype(np.float64)
    i = np.argmax(np.abs(r64), axis=1)
    m = r64[np.arange(len(r64)), i]
    keep = m != 0
    norm = np.rint(r64[keep] / m[keep, None] * NORM_BIN).astype(np.int64)
    return (norm, keep, i[keep]) if at_max else (norm, keep)


class Verdict(NamedTuple):
    grid: Optional[bool]       # True grid, False plain, None no verdict
    test: str                  # the test that decided
    detail: str


def judge(samples, lookups=False):
    """A Verdict on one operand's values over the sample runs (arrays of one
    shape, ..., D): every test, on the rows its caller hands it (the value
    check leaves out the rows only a written buffer's initial content
    reaches, and judges no operand with a row computed from state an
    earlier call wrote -- a row the replay changes that is no row a written
    buffer held; such a buffer's own rows are judged).
    `lookups`: the runs vary token ids alone, so a table looked up at them
    shows its own values and rows again from run to run -- the levels,
    vector levels and scaled levels tests, which count values and rows over
    all runs, are not tried (a lookup at token ids is plain; one at codes the
    model computes from a float shows in the runs that vary the floats)."""
    arrs = []
    for s in samples:
        a = np.asarray(s, dtype=np.float32)
        a = a.reshape(1, 1) if a.ndim == 0 else a.reshape(-1, a.shape[-1])
        arrs.append(a)
    if not arrs or any(a.shape != arrs[0].shape for a in arrs):
        return Verdict(None, "runs", "no sample runs of one shape")
    V = np.stack(arrs)                                  # (N, R, D)
    if V.shape[1] > MAX_ROWS:
        V = V[:, np.linspace(0, V.shape[1] - 1, MAX_ROWS).astype(int)]
    N, R, d = V.shape
    ok = np.all(np.isfinite(V), axis=2)                 # (N, R) rows judged
    if not ok.any():
        return Verdict(None, "finite", "no finite values")
    if all(np.array_equal(ok[0], ok[i]) and np.array_equal(
            V[0][ok[0]].view(np.uint32), V[i][ok[i]].view(np.uint32))
           for i in range(1, N)):
        return Verdict(None, "constant", "the same values in every run")
    per_run = [V[i][ok[i]] for i in range(N)]           # valid rows per run
    rows = np.concatenate(per_run, 0)                   # (M, D)
    allv = rows.reshape(-1)
    lo, hi = float(allv.min()), float(allv.max())
    if hi - lo <= DEGENERATE * max(abs(lo), abs(hi)):
        return Verdict(None, "degenerate", f"all within 0.1% of {hi!r}")
    norm, _, at = _normalised(rows, at_max=True)
    scaled = None
    if len(norm) and (norm == norm[:1]).all():
        scaled = "every row one vector up to its own scale"
    elif ok.all():
        pos = [_normalised(V[i]) for i in range(N)]
        if all(k.all() for _, k in pos) and all(
                np.array_equal(pos[0][0], p) for p, _ in pos[1:]):
            scaled = "each row position one vector up to its scale"
    if scaled is not None:
        # A constant pattern times a run-time amplitude is no grid made from
        # the inputs -- unless the amplitude is one: exp(round(x[..., :1]))
        # expanded to the row is one vector up to a rounded scale. Every
        # finite row's amplitude counts, a zero row's too: it has no
        # direction, so it is left out of the pattern alone (relu(sign(
        # x[..., :1])) expanded is rows of zeros and ones, two levels, which
        # the delegate would requantise to 40.063 for 40.0).
        r64 = rows.astype(np.float64)
        amps = r64[np.arange(len(r64)), np.argmax(np.abs(r64), axis=1)]
        run_of = np.concatenate([np.full(len(p), i) for i, p in
                                 enumerate(per_run)])
        got = None if "amplitudes" in _DISABLED else \
            _amplitude_grid(amps, run_of)
        if got is not None:
            return Verdict(True, "scaled constant", f"{scaled}, its row "
                           f"amplitudes {got}")
        return Verdict(False, "scaled constant", scaled)
    distinct = _distinct_per_row(rows)
    rich = int(np.count_nonzero(distinct >= 3))
    M = len(rows)
    if "levels" not in _DISABLED and not lookups:
        starts = _levels(allv, LEVELS)
        if starts is not None and len(starts) >= 2:
            nlev = len(starts)
            # Each level a run shows, and in how many runs: a level is
            # shared, seen again in another run (zeros and a few unrelated
            # values are not levels).
            seen = np.zeros(nlev, dtype=np.int64)
            per = []
            for r in per_run:
                if not len(r):
                    continue
                hit = np.unique(np.searchsorted(
                    starts, r.reshape(-1).astype(np.float64), "right") - 1)
                seen[hit] += 1
                per.append(len(hit))
            shared = int(np.count_nonzero(seen >= 2))
            if sum(per) >= 2 * nlev and max(per) >= 2 and shared >= 2:
                return Verdict(True, "levels",
                               f"{nlev} distinct values over all runs")
    if "row levels" not in _DISABLED and d > 2 * ROW_LEVELS:
        # A level holds several values: at most a quarter of the rows have
        # more than one level of a single value (a sparse row, zeros and a
        # few unrelated values, is not binarised).
        rl, sparse = [], 0
        for r in rows:
            n, single = _clusters(r, ROW_LEVELS, counts=True)
            if n > ROW_LEVELS:
                break
            rl.append(n)
            sparse += single > 1
        else:
            if (sum(1 for n in rl if n >= 2) * 2 >= len(rl)
                    and sparse * 4 <= len(rl)):
                return Verdict(True, "row levels",
                               f"at most {max(rl)} distinct values in each row")
    if ("vector levels" not in _DISABLED and not lookups
            and rich * 4 >= M):
        keys = {r.tobytes() for r in rows}
        if len(keys) <= VECTOR_LEVELS:
            per_rows = sum(len({r.tobytes() for r in p}) for p in per_run)
            if per_rows >= 2 * len(keys):
                return Verdict(True, "vector levels",
                               f"{len(keys)} distinct rows over all runs")
    if ("scaled levels" not in _DISABLED and not lookups and len(norm)
            and rich * 4 >= M):
        # Zero and the element each row is divided by (1 after the
        # division) are in every row whatever its values: only the other
        # elements count.
        trivial = norm == 0
        trivial[np.arange(len(norm)), at] = True
        nd = len(np.unique(norm[~trivial]))
        patterns = len({n.tobytes() for n in norm})
        per_row = sum(len(np.unique(n[~t])) for n, t in zip(norm, trivial))
        if nd <= min(SCALED_LEVELS, patterns * d / 3, per_row / 3):
            return Verdict(True, "scaled levels",
                           f"{nd} distinct values over rows divided by "
                           f"their maxima")
    groupings = [("row lattice", lambda: iter(rows), M, K_MAX)]
    if d % GS == 0:
        groupings.append(("group lattice",
                          lambda: iter(rows.reshape(-1, GS)), M * d // GS,
                          K_MAX))
    if R >= 3:
        groupings.append(("channel lattice",
                          lambda: (c for p in per_run if len(p) for c in p.T),
                          sum(d for p in per_run if len(p)), K_MAX))
    groupings.append(("tensor lattice",
                      lambda: (p.reshape(-1) for p in per_run if len(p)),
                      sum(1 for p in per_run if len(p)), K_MAX))
    groupings.append(("channel lattice across inputs", lambda: iter(rows.T),
                      d, K_MAX))
    groupings.append(("tensor lattice across inputs", lambda: iter([allv]),
                      1, K_MAX))
    if N >= 3:
        live = [r for r in range(R) if ok[:, r].any()]
        groupings.append(("element lattice across inputs",
                          lambda: (V[ok[:, r], r, c] for r in live
                                   for c in range(d)),
                          len(live) * d, K_MAX))
    for name, groups, count, kmax in groupings:
        if name in _DISABLED:
            continue
        if _grouping(groups(), count, kmax):
            return Verdict(True, name, "every group on its own step")
    return Verdict(False, "plain", f"{len(np.unique(allv))} distinct values, "
                   f"no levels and no lattice")


def _aligned(recorded):
    """The runs' recorded (row indices, rows, row count) as arrays over the
    union of the rows any run recorded, NaN where a run did not record a
    row (a row a run did not write: left out as non-finite)."""
    if not recorded:
        return []
    union = np.unique(np.concatenate([idx for idx, _, _ in recorded]))
    d = recorded[0][1].shape[-1] if recorded[0][1].ndim == 2 else 1
    out = []
    for idx, rows, _ in recorded:
        a = np.full((len(union), d), np.nan, np.float32)
        a[np.searchsorted(union, idx)] = rows.reshape(len(idx), d)
        out.append(a)
    return out


def judge_runs(values, lookups=False):
    """(Verdict, runs judged) on one regime's runs: all of them, then each
    half of WINDOW when there are two. A test with a ceiling (levels'
    LEVELS, scaled levels' SCALED_LEVELS) sees fewer rare values in fewer
    runs -- a grid with one rare level more in every six runs is levels in
    six and not in twelve -- while a lattice gains evidence from more."""
    halves = [values[:WINDOW], values[WINDOW:]] \
        if len(values) >= 2 * WINDOW else []
    first = None
    for window in [values] + halves:
        v = judge(window, lookups=lookups)
        if v.grid:
            return v, len(window)
        if first is None or (first[0].grid is None and v.grid is not None):
            first = (v, len(window))
    return first


def _row_classes(runs, redrawn, replayed=None):
    """(unwritten, carried, same): masks over the rows of one regime's runs
    (arrays (R, D) of one shape, run 0 first), given `redrawn` -- run 0
    again with every floating-point written buffer's initial content drawn
    afresh, after a warm-up on run 0's inputs -- and `replayed` -- run 0's
    inputs again on the state that enters run 1 (either None when not run).
    Same: the row is bit for bit the same in every run. Unwritten: the same
    in every run, and moved by the redraw -- it reads only content no run of
    the regime wrote. Carried: changed by the replay -- it depends on state
    an earlier call wrote (a cache shifted a row a call, a cache written at
    a counter, a previous call's write, a decaying state); a candidate with
    one that is no row of a written buffer is not judged
    (ValueCheck.state_read). The replay reaches one call, so it does not
    decide the kinds judged alone: a row whose run-0 value is bit for bit a
    row a written buffer held entering run 0 is that buffer's own, the
    others fresh (ValueCheck._run_regime, judge_regime)."""
    base = np.ascontiguousarray(runs[0], dtype=np.float32).view(np.uint32)
    same = np.ones(base.shape[0], bool)
    for r in runs[1:]:
        same &= (np.ascontiguousarray(r, dtype=np.float32).view(np.uint32)
                 == base).all(1)
    if redrawn is None or redrawn.shape != runs[0].shape:
        moved = np.zeros(base.shape[0], bool)
    else:
        moved = (np.ascontiguousarray(redrawn, dtype=np.float32)
                 .view(np.uint32) != base).any(1)
    if replayed is None or replayed.shape != runs[0].shape:
        changed = np.zeros(base.shape[0], bool)
    else:
        changed = (np.ascontiguousarray(replayed, dtype=np.float32)
                   .view(np.uint32) != base).any(1)
    unwritten = moved & same
    return unwritten, changed & ~unwritten, same


def _keep_rows(unwritten, same, kind=None, each=True):
    """The rows to judge, all but the unwritten, each set evenly spaced (a
    decode step's one written row of a long cache is kept): the rows that
    change from run to run, at most MAX_ROWS of each kind (`kind`, a mask,
    or None: a written buffer's rows; the others fresh) -- or, with
    each=False, the others first and then `kind`'s, MAX_ROWS in all (the
    replay's order, `kind` the rows the replay changes) -- then the rows the
    same in every run, to MAX_ROWS in all. Each kind its own budget: past
    MAX_ROWS fresh rows that change, the replay's order alone keeps no
    buffer row."""
    def spaced(a, k):
        return a if len(a) <= k else a[np.linspace(0, len(a) - 1, k)
                                       .astype(np.int64)]

    car = np.zeros(len(unwritten), bool) if kind is None else kind
    vary = ~unwritten & ~same
    fresh = spaced(np.flatnonzero(vary & ~car), MAX_ROWS)
    carr = spaced(np.flatnonzero(vary & car),
                  MAX_ROWS if each else MAX_ROWS - len(fresh))
    rest = spaced(np.flatnonzero(~unwritten & same),
                  max(0, MAX_ROWS - len(fresh) - len(carr)))
    return np.sort(np.concatenate([fresh, carr, rest])).astype(np.int64)


def judge_regime(values, groups=(), lookups=False):
    """(Verdict, runs judged) on one regime's runs (arrays (R, D) of one
    shape): all the rows together, then each of `groups` -- (what it is,
    mask over R) -- alone, each with every test; a group that is empty,
    every row, or a mask already judged is skipped. The value check's
    groups: its fresh rows, the rows of a written buffer it reads (each
    decided by value), the rows the replay leaves in place, and the rows
    kept the replay's way together (the last two both drawn from the
    replay-ordered selection). A grid in any is
    a grid -- rows of one kind can hide a grid in the other: a plain ring
    an earlier call wrote beside a fresh grid, a ring holding a grid beside
    plain fresh keys, a counter cache's later rows beside this call's
    -- then a plain verdict in any
    is plain. A group with too few rows to judge gives no verdict; the
    others decide."""
    got = [judge_runs(values, lookups=lookups)]
    seen = []
    for what, mask in groups:
        if (mask is None or not mask.any() or mask.all()
                or any(np.array_equal(mask, m) for m in seen)):
            continue
        seen.append(mask)
        v, n = judge_runs([x[mask] for x in values], lookups=lookups)
        got.append((v._replace(detail=f"{v.detail}, judged on {what}"), n))
    for v, n in got:
        if v.grid:
            return v, n
    for v, n in got:
        if v.grid is not None:
            return v, n
    return got[0]


# --- running the program ------------------------------------------------------

_FLOATS = (torch.float32, torch.float64, torch.float16, torch.bfloat16)


_ID_MOVES = frozenset({
    "aten::view_copy", "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::unsqueeze_copy", "aten::unsqueeze", "aten::squeeze_copy",
    "aten::squeeze", "aten::expand_copy", "aten::expand", "aten::slice_copy",
    "aten::slice", "aten::select_copy", "aten::select", "aten::permute_copy",
    "aten::permute", "aten::clone", "aten::alias_copy", "aten::alias",
    "aten::_to_copy", "dim_order_ops::_to_dim_order_copy",
    "dim_order_ops::_clone_dim_order", "aten::flatten",
})


_INDEX_WRITES = frozenset({
    "aten::index_copy", "aten::index_put", "aten::index_put_",
    "aten::scatter", "aten::scatter_add", "aten::scatter_reduce",
    "aten::index_fill", "aten::index_add", "aten::slice_scatter",
    "aten::select_scatter",
})


def _token_uses(ep, node):
    """(rows, embeddings) for integer input `node`: the rows of the smallest
    table it indexes through moves -- aten.embedding, or a gather,
    index_select or index of a parameter -- and the aten.embedding nodes it
    feeds; (None, ()) when it indexes no such table, or when it also indexes
    or writes any other tensor (a cache written at it, a buffer gathered at
    it): a position, held at the example's values whatever table it also
    picks rows of (GPT-2's learned positions index a 1,024-row table and an
    8-row cache)."""
    from executorch.backends.esp32s3.operator_support.qk_operands import (
        _schema_name,
    )

    params = set(ep.graph_signature.inputs_to_parameters)
    rows, embeddings = [], []
    frontier, seen = [node], {node}
    while frontier:
        n = frontier.pop()
        for u in n.users:
            if u in seen or u.op != "call_function":
                continue
            seen.add(u)
            name = _schema_name(u)
            args = u.args
            if name in _INDEX_WRITES:
                return None, ()
            table, dim = None, 0
            if name == "aten::embedding" and len(args) > 1 and args[1] is n:
                table = args[0]
            elif name == "aten::index_select" and len(args) > 2 and \
                    args[2] is n:
                table, dim = args[0], int(args[1])
            elif name == "aten::index" and len(args) > 1 and isinstance(
                    args[1], (list, tuple)) and n in args[1]:
                table, dim = args[0], list(args[1]).index(n)
            elif name == "aten::gather" and len(args) > 2 and args[2] is n:
                table, dim = args[0], int(args[1])
            if table is not None:
                if not (isinstance(table, torch.fx.Node) and (
                        name == "aten::embedding" or table.name in params)):
                    return None, ()          # indexes some other tensor
                val = table.meta.get("val")
                try:
                    rows.append(int(val.shape[dim]))
                except (AttributeError, TypeError, ValueError, IndexError):
                    pass
                if name == "aten::embedding":
                    embeddings.append(u)
                continue
            if name in _ID_MOVES:
                frontier.append(u)
    return (min(rows), tuple(embeddings)) if rows else (None, ())


def _token_rows(ep, node):
    """The rows of the smallest table token-id input `node` indexes, or None
    (a position: _token_uses)."""
    return _token_uses(ep, node)[0]


class _Failed(Exception):
    pass


class _Budget(Exception):
    """The runs passed MAX_BYTES or MAX_SECONDS: every candidate stays
    portable (exact is safe; a grid not judged must not be admitted)."""


class _Unclassed(Exception):
    """A tracked candidate's rows could not be classed: it is left unjudged
    (ValueCheck.unjudged), its q@kT portable -- never the rules alone."""


def _held_rows(live, names):
    """{row width: bytes of every row the floating-point buffers `names`
    hold in `live`}, each buffer viewed (-1, its last dimension) in float32.
    A buffer with no elements holds no row (a [1, 0] buffer's
    reshape(-1, 0) would raise, and leave the bare rules to decide)."""
    held = {}
    for name in names:
        v = live[name]
        if v.is_floating_point() and v.dim() and v.numel():
            rows = v.detach().to(torch.float32).reshape(
                -1, v.shape[-1]).cpu().numpy()
            held.setdefault(rows.shape[1], set()).update(
                r.tobytes() for r in rows)
    return held


class ValueCheck:
    """The sample runs of one program and the verdicts on its q@kT operands.
    Built once per program (for_program); verdict() runs the program on
    first use."""

    def __init__(self, ep, example_inputs=None):
        # Weakly, and every table by node name: the module-level cache
        # (_CHECKS) is keyed by the program's graph module, which a strong
        # path from its value would keep alive -- every program partitioned
        # in a process, kept.
        self._ep = weakref.ref(ep)
        self.example_inputs = example_inputs
        self.samples = None        # {node name: [(regime, [arrays])]}
        self.failure = None        # why values decide nothing: the rules alone
        self.over_budget = None    # why every candidate stays portable
        self.notes = []            # fallbacks taken
        self.recorded = 0          # sample bytes recorded
        self._embeddings = []      # aten.embedding node names at token ids
        self._rng = None
        self.runs = 0
        self.seconds = 0.0
        self._verdicts = {}
        self._silent = {}          # why values gave an operand no verdict
        self.row_counts = {}       # {name: [(regime, rows the replay leaves,
        #                            rows it changes, unwritten rows)]}
        self.unjudged = {}         # {name: why its rows could not be classed}
        self.state_read = {}       # {name: why: it reads state a call wrote}

    @property
    def ep(self):
        return self._ep()

    # -- inputs -----------------------------------------------------------
    def _user_inputs(self):
        """[(placeholder, example tensor or None)] in graph order."""
        sig = self.ep.graph_signature
        user = list(sig.user_inputs)
        nodes = {n.name: n for n in self.ep.graph_module.graph.nodes
                 if n.op == "placeholder"}
        examples = [None] * len(user)
        if self.example_inputs is not None:
            leaves = torch.utils._pytree.tree_leaves(self.example_inputs)
            if len(leaves) == len(user):
                examples = list(leaves)
            else:
                self.notes.append(
                    f"{len(leaves)} example inputs for {len(user)} graph "
                    f"inputs: drawn without them")
        return [(nodes[name], ex if isinstance(ex, torch.Tensor) else None)
                for name, ex in zip(user, examples)]

    def _input_sets(self, inputs):
        """[(regime, sets)]: sets() yields each set, a {placeholder: tensor},
        lazily (the first the example inputs where there are any), drawn in
        one order from the local generator whatever runs between."""
        # numpy, not torch: torch 2.15's randn gives other values on x86 and
        # ARM from sixteen elements up, and a verdict at a threshold must not
        # depend on the machine that lowers.
        rng = np.random.default_rng(SEED)
        kinds = {}
        self._embeddings = []
        for node, ex in inputs:
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                raise _Failed(f"graph input {node.name} is not a tensor")
            shape = tuple(ex.shape) if ex is not None else tuple(val.shape)
            if not all(isinstance(s, int) for s in shape):
                raise _Failed(f"graph input {node.name} has a symbolic shape "
                              f"and no example")
            dtype = val.dtype
            if dtype in _FLOATS:
                kinds[node] = ("float", shape, dtype, None)
            elif dtype == torch.bool or not (dtype.is_floating_point
                                              or dtype.is_complex):
                rows, embeddings = ((None, ()) if dtype == torch.bool
                                    else _token_uses(self.ep, node))
                kinds[node] = ("ids" if rows else "fixed", shape, dtype, rows)
                # by name: a node holds its graph, and the graph its module
                # -- the key of the weak cache this check is the value of
                self._embeddings.extend(u.name for u in embeddings)
            else:
                raise _Failed(f"graph input {node.name} of dtype {dtype}")

        def draw(node):
            kind, shape, dtype, rows = kinds[node]
            if kind == "float":
                return torch.from_numpy(rng.standard_normal(shape).astype(
                    np.float32)).to(dtype)
            if kind == "ids":
                return torch.from_numpy(rng.integers(0, rows, shape)).to(dtype)
            return torch.zeros(shape, dtype=dtype)

        base = {node: (ex.detach().clone() if ex is not None else draw(node))
                for node, ex in inputs}

        def sets_of(vary):
            yield dict(base)
            for _ in range(DRAWS):
                s = dict(base)
                for n in vary:
                    s[n] = draw(n)
                yield s

        def held():
            for _ in range(DRAWS + 1):
                yield dict(base)

        regimes = []
        for regime in ("float", "ids"):
            vary = [n for n, k in kinds.items() if k[0] == regime]
            if vary:
                regimes.append((FLOATS_VARY if regime == "float"
                                else IDS_VARY,
                                lambda vary=vary: sets_of(vary)))
        # The rows looked up at token ids, drawn unit normal in every run with
        # the ids held: under varying ids a lookup shows its table's own values
        # again, so the counting tests are left out there, and a program with
        # no float input (an LLM: ids, positions, a cache) would have no run
        # that varies a float -- exp(round(q)) unseen.
        if self._embeddings:
            regimes.append((EMBEDDINGS_VARY, held))
        self._rng = rng
        return regimes

    # -- runs -------------------------------------------------------------
    def _state(self):
        """{placeholder name: real tensor} for parameters, buffers and
        constants; the written buffers' names."""
        from executorch.backends.esp32s3.operator_support.const_weights import (
            _concrete,
        )

        ep = self.ep
        sig = ep.graph_signature
        state = {}
        for n in ep.graph_module.graph.nodes:
            if n.op != "placeholder" or n.name in sig.user_inputs:
                continue
            fqn = (sig.inputs_to_parameters.get(n.name)
                   or sig.inputs_to_buffers.get(n.name)
                   or sig.inputs_to_lifted_tensor_constants.get(n.name))
            if fqn is None:
                raise _Failed(f"placeholder {n.name} is neither an input, a "
                              f"parameter, a buffer nor a constant")
            v = ep.state_dict[fqn] if fqn in ep.state_dict else \
                ep.constants.get(fqn)
            real = _concrete(v) if isinstance(v, torch.Tensor) else None
            if real is None:
                raise _Failed(f"no real tensor behind {n.name}")
            state[n.name] = real
        written = {name for name, fqn in sig.inputs_to_buffers.items()
                   if fqn in sig.buffers_to_mutate.values()}
        return state, written

    def _run_regime(self, regime, sets, state, written, wanted, tracked,
                    warm, t0):
        """({candidate: per run (row indices, rows, row count) of its (-1, D)
        view}, {tracked candidate: [(group, indices of its rows among the
        rows kept)] -- its fresh rows, the rows of a written buffer it reads
        (by value), the rows the replay leaves in place, and the rows kept
        the replay's way, each judged alone (judge_regime)}),
        state carried between runs. A candidate no
        written floating-point buffer reaches records MAX_ROWS rows evenly
        spaced, the same in every run. One that such a buffer reaches
        (`tracked`) records all its rows; once the runs are done the regime
        runs twice more -- the replay, run 0's inputs on the state that
        enters run 1, and the redraw, run 0's inputs after a warm-up on them
        too with every floating-point written buffer's initial content drawn
        afresh -- and its rows are classed (_row_classes): a candidate with
        a row the replay changes that is no row a written buffer held
        entering run 0 reads state an earlier call wrote and is not judged
        (self.state_read); the unwritten are left out, and of the others up
        to MAX_ROWS that change from run to run of each kind are kept, with
        the rows the replay's order keeps (_keep_rows). When the replayed or
        the redrawn run raises, when the rows the written buffers held cannot
        be read (_held_rows), or when classing a candidate's rows raises, the
        candidates it affects are left unjudged (self.unjudged): the gate
        keeps their q@kT portable. Past MAX_BYTES recorded or MAX_SECONDS
        since t0 the runs stop (_Budget). In EMBEDDINGS_VARY every
        aten.embedding at token ids returns unit-normal rows of its shape,
        the same draws in the redrawn run as in the runs it repeats."""
        from torch.export.graph_signature import OutputKind

        sig = self.ep.graph_signature
        gm = self.ep.graph_module
        mutated = [(i, spec.target) for i, spec in enumerate(sig.output_specs)
                   if spec.kind == OutputKind.BUFFER_MUTATION]
        by_fqn = {fqn: name for name, fqn in sig.inputs_to_buffers.items()}
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        got = {n: [] for n in wanted}
        full = {n: [] for n in tracked}      # every row of every run
        redrawn = {}                         # the redrawn run's rows
        drawn_rows = set(self._embeddings) if regime == EMBEDDINGS_VARY \
            else set()            # node names, resolved on the live graph
        draws = {}                # (run number, node name): rows drawn
        rng = self._rng
        check = self

        def account(nbytes):
            check.recorded += nbytes
            if check.recorded > MAX_BYTES:
                raise _Budget(f"the samples passed {MAX_BYTES >> 20} MiB")

        def record(node, r, into):
            # Every leading dimension is a row axis; the last is the dot's width.
            flat = r.detach().reshape(-1, r.shape[-1]) if r.dim() \
                else r.detach().reshape(1, 1)
            n_rows = flat.shape[0]
            if node in tracked:
                rows = flat.to(torch.float32).cpu().numpy().copy()
                account(rows.nbytes)
                if into is None:
                    full[node].append(rows)
                else:
                    into[node] = rows
                return
            if into is not None:
                return
            idx = torch.arange(n_rows)
            if len(idx) > MAX_ROWS:
                idx = idx[torch.linspace(0, len(idx) - 1, MAX_ROWS).long()]
            rows = flat[idx].to(torch.float32).cpu().numpy().copy()
            idx = idx.cpu().numpy().astype(np.int64)
            account(rows.nbytes + idx.nbytes)
            got[node].append((idx, rows, n_rows))

        class _Record(torch.fx.Interpreter):
            def __init__(self, module, keep, run_no, into=None):
                super().__init__(module)
                self.keep, self.run_no, self.into = keep, run_no, into

            def run_node(self, node):
                if time.perf_counter() - t0 > MAX_SECONDS:
                    raise _Budget(f"the runs took past {MAX_SECONDS:.0f} s")
                if node.name in drawn_rows:
                    key = (self.run_no, node.name)
                    val = draws.get(key)
                    if val is None:
                        shape = tuple(node.meta["val"].shape)
                        val = torch.from_numpy(rng.standard_normal(
                            shape).astype(np.float32)).to(
                                node.meta["val"].dtype)
                        if self.run_no < 2:      # what the redrawn run repeats
                            draws[key] = val
                    return val
                r = super().run_node(node)
                if node in wanted and self.keep and isinstance(
                        r, torch.Tensor) and r.is_floating_point():
                    record(node, r, self.into)
                return r

        def forward(live, inputs, keep, run_no, into=None):
            args = [inputs[p] if p in inputs else live[p.name]
                    for p in placeholders]
            out = _Record(gm, keep, run_no, into).run(*args)
            self.runs += 1
            out = out if isinstance(out, (tuple, list)) else (out,)
            for i, fqn in mutated:
                name = by_fqn.get(fqn)
                if name is not None and isinstance(out[i], torch.Tensor):
                    live[name] = out[i]

        drawn = sets()
        first = next(drawn)
        second = next(drawn, None)
        warm_inputs = second if second is not None else first
        runs = [(warm_inputs, False)] if warm else []

        def ordered():
            yield from runs
            yield first, True
            if second is not None:
                yield second, True
            for s_ in drawn:
                yield s_, True

        live = {name: (state[name].detach().clone() if name in written
                       else state[name]) for name in state}
        rng0 = entering1 = None
        cached = {}           # {row width: bytes of every row a written
        #                       floating-point buffer held entering run 0}
        unclassed = None      # why the rows cannot be classed, if so
        for run_no, (inputs, keep) in enumerate(ordered()):
            if keep and rng0 is None:
                rng0 = torch.default_generator.get_state()
                if tracked:
                    try:
                        cached = _held_rows(live, written)
                    except Exception as e:  # noqa: BLE001 - unjudged
                        # Not a program that cannot run: the rows cannot
                        # be classed, so the tracked candidates stay
                        # portable (never the rules alone).
                        unclassed = (
                            f"{regime}: the rows the written buffers held "
                            f"entering run 0 could not be read "
                            f"({type(e).__name__}: {str(e)[:120]}), so its "
                            f"rows could not be classed")
            forward(live, inputs, keep, run_no)
            if keep and entering1 is None and tracked:
                entering1 = {name: live[name].detach().clone()
                             for name in written}
        classes = {}
        if not tracked:
            return got, classes
        # The replay: run 0's inputs again on the state that enters run 1,
        # the CPU generator as run 0 had it -- a row it changes depends on
        # state an earlier call wrote: carried.
        replayed = {}
        torch.default_generator.set_state(rng0)
        try:
            forward({name: (entering1[name] if name in written else v)
                     for name, v in state.items()},
                    first, True, 1 if warm else 0, into=replayed)
        except _Budget:
            raise
        except Exception as e:  # noqa: BLE001 - not judged: kept portable
            replayed.clear()
            why = (f"{regime}: the replay of run 0 on the state that enters "
                   f"run 1 raised ({type(e).__name__}: {str(e)[:120]}), so "
                   f"its rows could not be classed")
            for node in tracked:
                self.unjudged.setdefault(node.name, why)
            self.notes.append(f"{why}; every candidate it tracked stays "
                              f"portable")
        entering1 = None
        # The redrawn run: run 0 again, from the initial state with every
        # floating-point written buffer's content drawn afresh (integer and
        # bool buffers kept: a row they alone reach stays fresh), after a
        # warm-up on run 0's inputs rather than the warm-up's own, so that a
        # row an earlier call wrote and the state carried into run 0 moves
        # too: what the redraw leaves in place, run 0 wrote itself.
        redraw = np.random.default_rng(SEED + 1)
        live = {}
        for name, v in state.items():
            if name in written and v.is_floating_point():
                live[name] = torch.from_numpy(redraw.standard_normal(
                    tuple(v.shape)).astype(np.float32)).to(v.dtype)
            else:
                live[name] = v.detach().clone() if name in written else v
        torch.default_generator.manual_seed(SEED)    # as the runs began
        try:
            if warm:
                forward(live, first, False, 1)
            forward(live, first, True, 1 if warm else 0, into=redrawn)
        except _Budget:
            raise
        except Exception as e:  # noqa: BLE001 - not judged: kept portable
            redrawn.clear()
            why = (f"{regime}: the run with the written buffers' initial "
                   f"content drawn afresh raised ({type(e).__name__}: "
                   f"{str(e)[:120]}), so its rows could not be classed")
            for node in tracked:
                self.unjudged.setdefault(node.name, why)
            self.notes.append(f"{why}; every candidate it tracked stays "
                              f"portable")
        failed = None
        for node in tracked:
            runs_of = full.pop(node)
            if not runs_of:
                continue
            nbytes = sum(r.nbytes for r in runs_of)
            if any(r.shape != runs_of[0].shape for r in runs_of):
                keep_idx = np.arange(min(len(r) for r in runs_of))[:MAX_ROWS]
                counts = (len(keep_idx), 0, 0)
            else:
                try:
                    if unclassed is not None:
                        raise _Unclassed(unclassed)
                    unwritten, carried_all, same = _row_classes(
                        runs_of, redrawn.get(node), replayed.get(node))
                    counts = (int((~unwritten & ~carried_all).sum()),
                              int(carried_all.sum()), int(unwritten.sum()))
                    # The kinds by value, for every row: a row whose run-0
                    # value is bit for bit a row a written buffer held
                    # entering run 0 is the buffer's own (a cache row read
                    # through moves, however long ago a run wrote it -- the
                    # replay reaches one call), the others fresh. A row the
                    # replay changes that is no such row is computed from
                    # state an earlier call wrote: not judged, the q@kT
                    # portable (state_read).
                    rows0 = np.ascontiguousarray(runs_of[0], dtype=np.float32)
                    held = cached.get(rows0.shape[1], set())
                    buffer = np.fromiter((r.tobytes() in held for r in rows0),
                                         bool, len(rows0)) & ~unwritten
                    computed = np.flatnonzero(carried_all & ~buffer)
                    # Both per-kind and replay-ordered row selections: either
                    # can reveal a grid the other hides. The replay-ordered
                    # rows give two judgements of their own: the rows the
                    # replay leaves in place alone, all of them together.
                    kinds = _keep_rows(unwritten, same, buffer)
                    old = _keep_rows(unwritten, same, carried_all, each=False)
                    keep_idx = np.union1d(kinds, old)
                    classes[node] = [
                        ("its fresh rows alone", kinds[~buffer[kinds]]),
                        ("the rows of a written buffer it reads alone",
                         kinds[buffer[kinds]]),
                        ("the rows the replay leaves in place alone",
                         old[~carried_all[old]]),
                        ("its rows kept the replay's way, together", old)]
                except _Budget:
                    raise
                except Exception as e:  # noqa: BLE001 - unjudged: portable
                    # Rows that cannot be classed are not judged, and the
                    # q@kT stays portable -- never the rules alone, which
                    # would treat it as a program that cannot run.
                    why = unclassed if isinstance(e, _Unclassed) else (
                        f"{regime}: classing its rows raised "
                        f"({type(e).__name__}: {str(e)[:120]})")
                    self.unjudged.setdefault(node.name, why)
                    failed = failed or why
                    classes.pop(node, None)
                    keep_idx = np.arange(min(len(r) for r in runs_of))[
                        :MAX_ROWS]
                    counts = (len(keep_idx), 0, 0)
                    computed = ()
                if len(computed):
                    self.state_read.setdefault(node.name, (
                        f"{regime}: {len(computed)} of its {len(unwritten)} "
                        f"rows change when run 0's inputs replay on the "
                        f"state that enters run 1 and are no row a written "
                        f"buffer holds -- computed from state an earlier "
                        f"call wrote, which values cannot tell from a fresh "
                        f"value"))
            for extra in (redrawn.pop(node, None), replayed.pop(node, None)):
                if extra is not None:
                    nbytes += extra.nbytes
            got[node] = [(keep_idx, r[keep_idx].copy(), r.shape[0])
                         for r in runs_of]
            kept = sum(rows.nbytes + idx.nbytes for idx, rows, _ in got[node])
            kept += sum(idx.nbytes for _, idx in classes.get(node, ()))
            check.recorded += kept - nbytes      # the full rows freed
            self.row_counts.setdefault(node.name, []).append((regime,) + counts)
        if failed is not None:
            self.notes.append(f"{failed}; every candidate it could not class "
                              f"stays portable")
        return got, classes

    def _read(self):
        from executorch.backends.esp32s3.operator_support.patterns import (
            _kcache_affine_bmm,
            _qk_bmm_pattern,
        )

        gm = self.ep.graph_module
        wanted = set()
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            affine = _kcache_affine_bmm(n)
            got = affine[0] if affine is not None else _qk_bmm_pattern(n)
            if got is not None:
                wanted.add(got[0])
                if affine is None:
                    wanted.add(got[2])
        self.samples = {n.name: [] for n in wanted}
        if not wanted:
            return
        inputs = self._user_inputs()
        regimes = self._input_sets(inputs)
        if not regimes:
            self.failure = "no graph input a run can vary"
            return
        state, written = self._state()
        warm = bool(written)
        floats = {name for name in written if state[name].is_floating_point()}
        tracked = {n for n in wanted if self._reaches_written(n, floats)}
        t0 = time.perf_counter()
        dropped = []
        # The program's own random ops run on the CPU generator, seeded for
        # determinism and restored after: fork_rng(devices=[]) saves the
        # CPU's state alone, so nothing here may seed another device's
        # (torch.manual_seed would seed CUDA's and MPS's as well).
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for regime, sets in regimes:
                torch.default_generator.manual_seed(SEED)
                try:
                    got, classes = self._run_regime(
                        regime, sets, state, written, wanted, tracked, warm,
                        t0)
                except _Budget:
                    raise
                except Exception as e:  # noqa: BLE001 - this kind held
                    # Holding the kind that raised is this regime not run:
                    # the others still judge (one run of the example inputs
                    # alone could not -- judge needs runs that differ).
                    why = f"{type(e).__name__}: {str(e)[:160]}"
                    dropped.append(regime)
                    self.notes.append(f"the runs where {regime} raised "
                                      f"({why}); that kind is held, its "
                                      f"regime not judged")
                    continue
                for n, values in got.items():
                    self.samples[n.name].append((regime, values,
                                                 classes.get(n)))
        if len(dropped) == len(regimes):
            raise _Failed(f"every regime's runs raised ({', '.join(dropped)}"
                          f"), and the example inputs alone, one run, cannot "
                          f"judge")

    def _ensure(self):
        if self.samples is not None:
            return
        t0 = time.perf_counter()
        try:
            self._read()
        except _Budget as e:
            self.over_budget = str(e)
            self.samples = {}
        except _Failed as e:
            self.failure = str(e)
        except Exception as e:  # noqa: BLE001 - never fail the export
            self.failure = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            if self.samples is None:
                self.samples = {}
            self.seconds += time.perf_counter() - t0

    def _reaches_written(self, node, floats):
        """Whether a floating-point buffer the forward writes (`floats`) is
        among `node`'s ancestors: the prefilter for the redrawn run. Which of
        its rows hold only that buffer's initial content is decided from
        values, per row (_row_classes); this walk decides no verdict."""
        if not floats:
            return False
        frontier, seen = [node], {node}
        while frontier:
            n = frontier.pop()
            if n.op == "placeholder" and n.name in floats:
                return True
            for a in n.all_input_nodes:
                if a not in seen:
                    seen.add(a)
                    frontier.append(a)
        return False

    def verdict(self, operand):
        """The Verdict on `operand` with the regime and run count it was
        decided on: (Verdict, regime, runs), or None when values decide
        nothing (no grid in any regime is a plain verdict)."""
        if operand.name in self._verdicts:
            return self._verdicts[operand.name]
        self._ensure()
        result = None
        why = "no sample recorded"
        if self.failure is None and self.over_budget is None:
            for entry in self.samples.get(operand.name, ()):
                regime, recorded = entry[0], entry[1]
                kinds = entry[2] if len(entry) > 2 else None
                if recorded and all(len(idx) == 0 for idx, _, _ in recorded):
                    why = ("every row it read only a written buffer's initial "
                           "content reaches -- the same in every run, moved "
                           "when that content is drawn afresh -- which no run "
                           "wrote: left out")
                    continue
                values = _aligned(recorded)
                groups = ()
                if kinds is not None:
                    # _aligned's rows: the union of the indices recorded
                    union = np.unique(np.concatenate(
                        [idx for idx, _, _ in recorded]))
                    groups = [(what, np.isin(union, idx))
                              for what, idx in kinds]
                v, runs = judge_regime(values, groups,
                                       lookups=regime == IDS_VARY)
                if v.grid is None:              # judge's own reason
                    why = f"{v.test}: {v.detail} ({regime})"
                if v.grid or (result is None and v.grid is not None):
                    result = (v, regime, runs)
                if v.grid:
                    break
        if result is None:
            self._silent[operand.name] = why
        self.samples.pop(operand.name, None)       # judged: freed
        self._verdicts[operand.name] = result
        return result

    def why_silent(self, operand):
        """Why verdict() gave `operand` no verdict (it returned None)."""
        return self._silent.get(operand.name, "no sample recorded")

    def release(self):
        """Keep the verdicts, drop every sample (Esp32Partitioner.partition
        calls this when it returns: the program it judged is the lowered
        program's root, which upstream caches keep alive)."""
        self.samples = {} if self.samples is not None else None

    def grid(self, operand):
        """(test, detail, regime, runs) when the sample runs show `operand`
        on a grid, else None."""
        got = self.verdict(operand)
        if got is None or not got[0].grid:
            return None
        v, regime, runs = got
        return v.test, v.detail, regime, runs


_CHECKS = weakref.WeakKeyDictionary()


def _fingerprint(gm):
    return hash(tuple((n.name, str(n.target)) for n in gm.graph.nodes))


def for_program(ep, example_inputs=None):
    """The ValueCheck of `ep`, shared by every gate that asks about the same
    graph (the partitioner's regrouping rounds, its probe twins): the
    program runs once per input set."""
    gm = ep.graph_module
    key = (_fingerprint(gm), id(example_inputs))
    got = _CHECKS.get(gm)
    if got is not None and got[0] == key and got[1].ep is ep:
        return got[1]
    check = ValueCheck(ep, example_inputs)
    _CHECKS[gm] = (key, check)
    return check
