# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::lut16: an activation between two static int16 quantisers, as a
fitted table.

quant_lut (fold_quant_lut_pass.py) turns an activation between two int8
quantisers into a 256-entry table and refuses int16 sites as lut-wide-codes:
Arm a16w8's activations have 65,535 codes. This module takes those sites:

    q1(X) -> dq1 -> A -> q2 -> dq2     becomes     lut16(X, T, ...) -> q2 -> dq2

with a table T that lut_fit.py fits at lowering time from the graph alone --
the eager answer E(c) = q2(A(dq1(c))) of every input code, A evaluated by
torch -- in one of five schemes (direct, uniform linear, uniform cubic, NSAU
linear, NSAU cubic), chosen under a byte budget.

    espressif::lut16(Tensor x, Tensor table, float inv_scale, int zero_point,
                     int quant_min, int quant_max, int scheme, int lo, int hi,
                     int shift, int G, float out_scale, int out_zero_point,
                     int out_quant_min, int out_quant_max) -> Tensor

per element, in this order (the contract; mirrors.lut16 is its numpy
mirror, ops/quant_lut16.cpp the kernel, built
-ffp-contract=off, and this module's CompositeExplicitAutograd body the
torch reference):

  1. c = clamp(rne(fl(x * inv_scale)) + zero_point, quant_min, quant_max),
     fl(x * inv_scale) clamped to [-2^16, 2^16] first (NaN -> 0): quantize_
     per_tensor's code for every x, the zero point for NaN (ExecuTorch 1.6's
     rule, runtime/quant_rules.h)
  2. the table step on c clamped to [lo, hi] (mirrors.lut16_codes):
     direct     y = T[c - lo]
     uniform    u = c - lo, j = u >> F, t = u & (2^F - 1)        (F = shift)
     NSAU       a = |c|, sh = max(floor(log2(a | 1)) - m, 0), q = a >> sh,
                j = (sh << m) + q + the bias of c's sign, t = a - (q << sh)
     linear     uniform y = ((V[j] << F) + (V[j+1] - V[j]) t + 2^(F+G-1)) >> (F+G)
                NSAU    y = (V[j] + (((V[j+1] - V[j]) t) >> sh) + 2^(G-1)) >> G
     cubic      tf = fl(t) * 2^-F (exact), y = fl(c0 + fl(tf fl(c1 + fl(tf
                fl(c2 + fl(tf c3)))))), then rne by (y + 1.5 2^23) - 1.5 2^23
     The scheme word carries two flags (mirrors.LUT16_OFFSET/_TAIL):
     8 on an NSAU scheme (11, 12) lays each sign's half out from the
     segment of its first code (mirrors.lut16_nsau_layout: the
     biases); 16 on a uniform one (17, 18) leaves c unclamped (int16 only)
     and clamps the segment instead, j = clamp(u >> F, 0, K - 1), t = u -
     (j << F), so past [lo, hi] the edge segment continues
  3. y = clamp(y, out_quant_min, out_quant_max)
  4. out = fl(fl(y - out_zero_point) * out_scale), dequantize's arithmetic

So the output is always on q2's grid: elide_lut16_pairs drops the q2 -> dq2
pair after the table once partitioning has kept it portable.

  fold_fitted_luts   the rewrite, before to_edge, after fold_quant_luts: every
                     candidate quant_lut refuses as lut-wide-codes, with
                     quant_lut's other refusals kept and two of its own:
                     lut16-output (no single int16 q2 after A) and
                     lut16-budget (no scheme within the byte budget meets the
                     flips bar; the op stays portable). On by default
                     (Esp32Partitioner's luts16, which lower_esp32 follows
                     unless its own luts16 is given); with luts16=False nothing
                     here runs and every byte is unchanged. A site whose q1
                     carries an observed range (lut16_range.py:
                     espressif_quantizer.carry_observer_ranges, stripped onto q1
                     first thing in the lowering) is fitted and scored over
                     that range's window (site_window); the rest over every
                     code. A lut16 node already in the graph (a table trained
                     through, lut16_train.py) is no activation: it
                     stays as it is, bytes and all.
  elide_lut16_pairs  in apply_passes, after partitioning.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.library import Library

from executorch.backends.esp32s3.quantizer import lut_fit
from executorch.backends.esp32s3._passes import fold_quant_lut_pass
from executorch.backends.esp32s3._passes.fold_quant_lut_pass import (
    _contiguous,
    _is,
    _scalar_qparams,
    DQ,
    Q,
)

lib = Library("espressif", "FRAGMENT")
_ARGS = ("Tensor x, Tensor table, float inv_scale, int zero_point, int quant_min, "
         "int quant_max, int scheme, int lo, int hi, int shift, int G, "
         "float out_scale, int out_zero_point, int out_quant_min, int out_quant_max")
lib.define(f"lut16({_ARGS}) -> Tensor")
lib.define(f"lut16.out({_ARGS}, *, Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")

RAIL = 65536.0
MAGIC = 12582912.0


def quantize16(x, inv_scale, zero_point, quant_min, quant_max):
    """Step 1: q1's code of every element, int64 (NaN -> zero_point)."""
    v = x * inv_scale
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v).clamp(-RAIL, RAIL)
    return (torch.round(v) + zero_point).clamp(quant_min, quant_max).to(torch.int64)


def nsau_half(lo, hi, m):
    return int(lut_fit.qm.lut16_nsau_half(lo, hi, m))


def index16(c, scheme, lo, hi, shift):
    """Step 2's index: (j, t, F) per code, int64 (mirrors.lut16_index)."""
    base = scheme & lut_fit.qm.LUT16_BASE
    if base in (1, 2):
        if scheme & lut_fit.TAIL:
            u = c.clamp(-32768, 32767) - lo
            j = (u >> shift).clamp(0, ((hi - lo) >> shift))
            return j, u - (j << shift), torch.full_like(u, shift)
        u = c.clamp(lo, hi) - lo
        return u >> shift, u & ((1 << shift) - 1), torch.full_like(u, shift)
    c = c.clamp(lo, hi)
    a = c.abs()
    e = torch.frexp((a | 1).to(torch.float64))[1].to(torch.int64) - 1
    sh = (e - shift).clamp(min=0)
    q = a >> sh
    t = a - (q << sh)
    pos, neg, _ = lut_fit.qm.lut16_nsau_layout(scheme, lo, hi, shift)
    return (sh << shift) + q + torch.where(c < 0, neg, pos), t, sh


def codes16(c, table, scheme, lo, hi, shift, G, out_quant_min, out_quant_max):
    """Steps 2 and 3: the output code of every input code c (int64)."""
    base = scheme & lut_fit.qm.LUT16_BASE
    if base == 0:
        y = table.to(torch.int64)[c.clamp(lo, hi) - lo]
    elif base in (1, 3):
        V = table.to(torch.int64)
        j, t, F = index16(c, scheme, lo, hi, shift)
        v0, v1 = V[j], V[j + 1]
        if base == 1:
            sh = shift + G
            y = ((v0 << shift) + (v1 - v0) * t + ((1 << (sh - 1)) if sh else 0)) >> sh
        else:
            y = (v0 + (((v1 - v0) * t) >> F) + ((1 << (G - 1)) if G else 0)) >> G
    else:
        C = table.reshape(-1, 4)
        j, t, F = index16(c, scheme, lo, hi, shift)
        # 2^-F built from its bits: exact, as the device's float.s scale is
        step = ((127 - F).to(torch.int32) << 23).view(torch.float32)
        tf = t.to(torch.float32) * step
        y = C[j, 2] + tf * C[j, 3]
        y = C[j, 1] + tf * y
        y = C[j, 0] + tf * y
        y = ((y + MAGIC) - MAGIC).to(torch.int64)
    return y.clamp(out_quant_min, out_quant_max)


def lut16(x, table, inv_scale, zero_point, quant_min, quant_max, scheme, lo, hi,
          shift, G, out_scale, out_zero_point, out_quant_min, out_quant_max):
    """The reference: torch integer ops and float32 ops, one rounding each."""
    c = quantize16(x, inv_scale, zero_point, quant_min, quant_max)
    y = codes16(c, table, scheme, lo, hi, shift, G, out_quant_min, out_quant_max)
    return (y - out_zero_point).to(torch.float32) * out_scale


def _lut16_out(x, table, inv_scale, zero_point, quant_min, quant_max, scheme, lo,
               hi, shift, G, out_scale, out_zero_point, out_quant_min,
               out_quant_max, *, out):
    out.copy_(lut16(x, table, inv_scale, zero_point, quant_min, quant_max, scheme,
                    lo, hi, shift, G, out_scale, out_zero_point, out_quant_min,
                    out_quant_max))
    return out


impl.impl("lut16", lut16, "CompositeExplicitAutograd")
impl.impl("lut16.out", _lut16_out, "CompositeExplicitAutograd")


@torch.library.register_fake("espressif::lut16")
def _lut16_fake(x, table, inv_scale, zero_point, quant_min, quant_max, scheme, lo,
                hi, shift, G, out_scale, out_zero_point, out_quant_min,
                out_quant_max):
    return torch.empty_like(x)


LUT16 = torch.ops.espressif.lut16.default
TABLE_DTYPE = {0: torch.int16, 1: torch.int32, 2: torch.float32, 3: torch.int32,
               4: torch.float32, 11: torch.int32, 12: torch.float32, 17: torch.int32,
               18: torch.float32}

# --- What the rewrite takes ---------------------------------------------------------

I16 = (-32768, 32767)


def _int16_grid(qp):
    scale, zero_point, qmin, qmax, dtype = qp
    return (dtype == torch.int16 and I16[0] <= qmin < qmax <= I16[1]
            and qmin <= zero_point <= qmax and math.isfinite(scale) and scale > 0)


def refusal(A):
    """The named rule keeping int16 site A portable, or None to fold it.
    "skip" for a site that is not lut16's (quant_lut takes it: int8)."""
    rule = fold_quant_lut_pass.refusal(A)
    if rule is None:
        return "skip"
    if rule != "lut-wide-codes":
        return rule
    dq1 = A.args[0]
    q1 = dq1.args[0]
    qp = _scalar_qparams(q1.args)
    scale, zero_point, qmin, qmax, dtype = qp
    if dtype != torch.int16 or qmin < I16[0] or qmax > I16[1]:
        return "lut-wide-codes"
    if not _int16_grid(qp):
        return "lut-bad-qparams"
    inv = fold_quant_lut_pass.reciprocal(scale)
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
    out = fold_quant_lut_pass._out_qparams(A)
    if out is None or not _int16_grid(_scalar_qparams((None,) + tuple(out))):
        return "lut16-output"
    return None


# --- Fitting a site --------------------------------------------------------------------

_FITS = {}


def site_from(target, args, kwargs, in_qparams, out_qparams, window=None,
              dq_kwargs=None):
    """lut_fit.Site for activation `target` (its constant args and kwargs)
    between the input quantiser in_qparams = (s1, z1, qmin1, qmax1, dtype1)
    -- dq1's arguments -- and the output quantiser out_qparams, q2's: dq1 on
    every input code of `window` ([lo, hi]; default every code of q1), the
    activation by torch in quant_lut's canonical vector context, q2."""
    op = fold_quant_lut_pass.ALLOWLIST[target]
    s1, z1, qmin1, qmax1, dtype1 = in_qparams
    s2, z2, qmin2, qmax2, dtype2 = out_qparams
    lo, hi = (qmin1, qmax1) if window is None else window
    codes = torch.arange(lo, hi + 1, dtype=torch.int64).to(dtype1)
    with torch.no_grad():
        xs = fold_quant_lut_pass._dq_op()(codes, *in_qparams, **(dq_kwargs or {}))
    y = fold_quant_lut_pass.evaluate(target, xs, args, kwargs)
    with torch.no_grad():
        E = fold_quant_lut_pass._q_op()(y, *out_qparams).to(torch.int64).numpy()
    y = y.numpy()

    def values(_xs):
        return y

    def values64(xs64):
        with torch.no_grad():
            return target(torch.from_numpy(np.asarray(xs64, np.float64)), *args,
                          **kwargs).numpy()

    return lut_fit.Site(op, lut_fit.Grid(float(s1), z1, lo, hi),
                        lut_fit.Grid(float(s2), z2, qmin2, qmax2), values=values,
                        values64=values64, target=E, floor_bar=window is not None,
                        tail_rule=op in lut_fit.TAIL_FUNCS or (
                            op == "pow" and len(args) > 0 and args[0] == 2))


def site_of(A, window=None):
    """lut_fit.Site for node A: its graph's own dq1 on every input code (or
    those of `window`), A by torch in quant_lut's canonical vector context,
    its graph's own q2."""
    dq1 = A.args[0]
    return site_from(A.target, tuple(A.args[1:]), dict(A.kwargs), tuple(dq1.args[1:]),
                     fold_quant_lut_pass._out_qparams(A), window, dict(dq1.kwargs))


def site_window(A):
    """The input codes a table for node A holds when its q1 carries an
    observed range (lut16_range: espressif::observed_range, stripped onto
    q1's meta at lowering), else None: every code."""
    q1 = A.args[0].args[0]
    rng = q1.meta.get("observed_range")
    if rng is None:
        return None
    from executorch.backends.esp32s3.quantizer import lut16_range

    s1, z1, qmin1, qmax1, _ = _scalar_qparams(q1.args)
    return lut16_range.window(rng, s1, z1, qmin1, qmax1)


def fit_parts(target, args, kwargs, in_qparams, out_qparams, spec, window=None,
              dq_kwargs=None):
    """(Table, None) or (None, BudgetExceeded) for site_from's site, cached
    per process on everything the fit reads: lut_fit.fit's chooser over
    every code; over a window, lut_fit.fit_smallest's against the float32
    floor of the window's codes, or the graph-only table where that is no
    larger."""
    key = (str(target), repr(tuple(args)), repr(sorted(dict(kwargs).items())),
           repr(tuple(in_qparams)), repr(sorted((dq_kwargs or {}).items())),
           repr(tuple(out_qparams)), spec, window)
    if key not in _FITS:
        site = site_from(target, args, kwargs, in_qparams, out_qparams, window,
                         dq_kwargs)
        chooser = lut_fit.fit if window is None else lut_fit.fit_smallest
        try:
            _FITS[key] = (chooser(site, spec), None)
        except lut_fit.BudgetExceeded as error:
            _FITS[key] = (None, error)
        if window is not None and spec.scheme is None:
            # the graph-only table competes, and wins where it is no larger:
            # a carried range never costs bytes, and that table holds every
            # code, so nothing past the window clamps
            whole, _ = fit_parts(target, args, kwargs, in_qparams, out_qparams, spec,
                                 None, dq_kwargs)
            ranged = _FITS[key][0]
            if whole is not None and (ranged is None or whole.nbytes <= ranged.nbytes):
                _FITS[key] = (whole, None)
    return _FITS[key]


def fit_site(A, spec, window=None):
    """fit_parts for node A (over `window`'s codes, or every code)."""
    dq1 = A.args[0]
    return fit_parts(A.target, tuple(A.args[1:]), dict(A.kwargs), tuple(dq1.args[1:]),
                     fold_quant_lut_pass._out_qparams(A), spec, window, dict(dq1.kwargs))


@dataclass
class Lut16Build:
    op: str
    table: lut_fit.Table
    inv_scale: float
    zero_point: int
    quant_min: int
    quant_max: int
    out_scale: float
    out_zero_point: int

    @property
    def tensor(self):
        return torch.from_numpy(np.ascontiguousarray(self.table.data)).to(
            TABLE_DTYPE[self.table.scheme_id])

    @property
    def name(self):
        t = self.table
        digest = hashlib.sha256(
            np.ascontiguousarray(t.data).tobytes()
            + struct.pack("<7i", t.scheme_id, t.lo, t.hi, t.shift, t.G,
                          *t.data.shape[:1], t.data.ndim)).hexdigest()
        return f"esp32_lut16_{digest[:16]}"


@dataclass
class Lut16Report:
    tables: list = field(default_factory=list)     # (node name, Lut16Build)
    refused: list = field(default_factory=list)    # (node name, op, rule, detail)
    constants: dict = field(default_factory=dict)  # name -> users
    windows: dict = field(default_factory=dict)    # node -> codes of a carried range

    def summary(self):
        shared = len(self.tables) - len(self.constants)
        parts = [f"lut16: {len(self.tables)} table(s)"
                 f"{f' ({shared} shared)' if shared else ''}"
                 f", {sum(b.table.nbytes for _, b in self.tables)} B"
                 f"{f', {len(self.windows)} over carried ranges' if self.windows else ''}"]
        flips = [f"{node} {b.op} {b.table.scheme} {b.table.flips}"
                 for node, b in self.tables if b.table.flips]
        if flips:
            parts.append("flips " + ", ".join(flips))
        if self.refused:
            parts.append(f"{len(self.refused)} refused: " + ", ".join(
                f"{node} ({op}) {rule}" for node, op, rule, _ in self.refused))
        return "; ".join(parts)

    def lines(self):
        """One line per table and refusal, with the flipped codes."""
        out = []
        for node, b in self.tables:
            t = b.table
            out.append(f"{node} ({b.op}): {t.describe()}"
                       + (f"; over the carried range's codes {list(self.windows[node])}"
                          if node in self.windows else "")
                       + (f"; flipped codes {t.flipped}" if t.flips else ""))
        for node, op, rule, detail in self.refused:
            out.append(f"{node} ({op}): {rule}" + (f" -- {detail}" if detail else ""))
        return out


def fold_fitted_luts(exported_program, budget=lut_fit.DEFAULT_BUDGET):
    """Rewrite every int16 activation between static quantisers into
    espressif::lut16, in place, on an ATen-dialect ExportedProgram (or a dict
    of them). Run after fold_quant_luts (which takes the int8 sites). Returns
    a Lut16Report; a graph with no lut16 site is not touched at all."""
    if isinstance(exported_program, dict):
        report = Lut16Report()
        for ep in exported_program.values():
            part = fold_fitted_luts(ep, budget)
            report.tables += part.tables
            report.refused += part.refused
            report.constants.update(part.constants)
            report.windows.update(part.windows)
        return report

    from executorch.backends.transforms.utils import create_constant_placeholder
    from torch.export.graph_signature import InputKind

    ep = exported_program
    graph = ep.graph_module.graph
    report = Lut16Report()
    spec = lut_fit.Spec(budget=budget)
    for A in list(fold_quant_lut_pass._candidates(graph)):
        op = fold_quant_lut_pass.ALLOWLIST.get(A.target) or fold_quant_lut_pass.INPLACE[A.target]
        rule = refusal(A)
        if rule == "skip":
            continue
        if rule:
            report.refused.append((A.name, op, rule, ""))
            continue
        window = site_window(A)
        table, error = fit_site(A, spec, window)
        if table is None:
            report.refused.append((A.name, op, "lut16-budget", str(error)))
            continue
        dq1 = A.args[0]
        q1 = dq1.args[0]
        s1, z1, qmin1, qmax1, _ = _scalar_qparams(q1.args)
        s2, z2, _, _, _ = fold_quant_lut_pass._out_qparams(A)
        build = Lut16Build(op, table, fold_quant_lut_pass.reciprocal(s1), z1, qmin1, qmax1,
                           float(np.float32(s2)), z2)
        name = build.name
        first = next(n for n in graph.nodes if n.op == "placeholder")
        with graph.inserting_before(first):
            const = create_constant_placeholder(
                ep, graph, name, InputKind.CONSTANT_TENSOR, build.tensor)
        with graph.inserting_before(A):
            node = graph.call_function(LUT16, (
                q1.args[0], const, build.inv_scale, z1, qmin1, qmax1,
                table.scheme_id, table.lo, table.hi, table.shift, table.G,
                build.out_scale, z2, table.out_qmin, table.out_qmax))
        node.meta.update({key: A.meta[key] for key in
                          ("val", "tensor_meta", "stack_trace", "nn_module_stack")
                          if key in A.meta})
        A.replace_all_uses_with(node)
        graph.erase_node(A)
        for dead in (dq1, q1):
            if not dead.users:
                graph.erase_node(dead)
        report.tables.append((node.name, build))
        report.constants.setdefault(const.name, []).append(node.name)
        if window is not None and window[0] <= table.lo and table.hi <= window[1]:
            report.windows[node.name] = window
    if report.tables:
        graph.lint()
        ep.graph_module.recompile()
    return report


# --- After partitioning -------------------------------------------------------------------

def _grid_values(args):
    """Every value lut16 can output for out args (scale, zp, qmin, qmax)."""
    scale, zero_point, qmin, qmax = args
    y = torch.arange(qmin, qmax + 1, dtype=torch.int64)
    return (y - zero_point).to(torch.float32) * scale


def elide_lut16_pairs(ep):
    """Drop q -> dq pairs after a lut16 whose output grid they map to itself
    (fold_quant_lut_pass.elide_lut_pairs for the int16 op): lut16's output is
    always one of fl(fl(y - z2) * s2), y in [out_quant_min, out_quant_max], so
    where the pair returns each of those values bit for bit, dq can read lut16
    directly. Edge-dialect ExportedProgram, in place; returns pairs removed."""
    graph = ep.graph_module.graph
    removed = 0
    for T in [n for n in graph.nodes if n.op == "call_function"
              and fold_quant_lut_pass._schema(n.target) == ("espressif::lut16", "default")]:
        grid = _grid_values((T.args[11], T.args[12], T.args[13], T.args[14]))
        for q in list(T.users):
            if not _is(q, Q) or q.args[0] is not T or q.kwargs:
                continue
            for dq in list(q.users):
                if not _is(dq, DQ) or dq.args[0] is not q:
                    continue
                with torch.no_grad():
                    back = fold_quant_lut_pass._dq_op()(fold_quant_lut_pass._q_op()(grid, *q.args[1:]),
                                              *dq.args[1:], **dq.kwargs)
                if fold_quant_lut_pass._same_bits(back, grid):
                    dq.replace_all_uses_with(T)
                    graph.erase_node(dq)
                    removed += 1
            if not q.users:
                graph.erase_node(q)
    if removed:
        graph.lint()
        ep.graph_module.recompile()
    return removed
