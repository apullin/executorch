# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training through a fitted int16 table (opt-in).

fit_tables(prepared, spec) -- espressif_quantizer's helper -- takes a PT2E model
after calibration (prepare_pt2e or prepare_qat_pt2e), freezes its static
activation observers, fits a table for every int16 activation site (an
allowlisted activation between two static int16 per-tensor observers, the
sites fold_lut16_pass.fold_fitted_luts takes at lowering), and swaps the
activation for a Lut16 module:

    X -> obs1 -> silu -> obs2      becomes      X -> lut16_0 -> obs2

Lut16's forward is espressif::lut16 itself -- quantise with q1's own
qparams, the table, q2's code, dequantise -- so the network trains against
the function the device runs, bit for bit. Its backward is one of:

  grad="true"   (default) the straight-through estimator: the activation's
                own derivative f'(dq1(c)), by autograd on the ATen op, and 0
                where x quantises outside q1's range (PT2E fake-quant's rule)
  grad="table"  the deployed function's slope at c: the segment's
                derivative (cubic, linear) or the entries' difference
                (direct), 0 past the table's [lo, hi]
  learn=True    the table's own entries learn too: the forward runs them
                rounded to the device's types (int16 codes, int32 nodes, or
                float32 coefficients within the runtime's 2^19), and each
                entry gets the gradient of the output by its weight in the
                interpolation; the buffer `table` always holds what the last
                forward ran

After training, convert_pt2e leaves each lut16_N module where it is (it is
no observer, no quantised module), torch.export records it as
espressif::lut16 on its `table` buffer, and the lowering takes that node as
it stands: no refit, the same bytes (fold_fitted_luts folds activations, not
lut16 nodes). q1 and its dequantise go when nothing else reads X's observer;
q2 -> dq2 after the table goes at lowering (elide_lut16_pairs) because the
frozen obs2 is the grid the table writes.

Imports torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from executorch.backends.esp32s3.quantizer import lut_fit
from executorch.backends.esp32s3._passes import fold_quant_lut_pass
from executorch.backends.esp32s3._passes import fold_lut16_pass as Q16
from executorch.backends.esp32s3.quantizer import lut16_range

GRADS = ("true", "table")
_BIG = 1 << 40


# --- the module -----------------------------------------------------------------------

def _round_table(nodes, scheme, out_qmin, out_qmax):
    """Float entries -> what the device holds, as the op's table tensor."""
    if scheme & 7 == 0:
        return torch.round(nodes).clamp(out_qmin, out_qmax).to(torch.int16)
    if scheme & 7 in (1, 3):
        return torch.round(nodes).clamp(-(2 ** 31), 2 ** 31 - 1).to(torch.int32)
    return nodes.clamp(-lut_fit.COEF_MAX, lut_fit.COEF_MAX).to(torch.float32)


class Lut16(nn.Module):
    """One fitted table in a model: espressif::lut16 on x with q1's and q2's
    frozen qparams, and a training backward (module docstring)."""

    def __init__(self, build, in_scale, target, args=(), kwargs=None, grad="true",
                 learn=False):
        super().__init__()
        if grad not in GRADS:
            raise ValueError(f"grad must be one of {GRADS}, got {grad!r}")
        t = build.table
        self.register_buffer("table", build.tensor.clone())
        self.scalars = (build.inv_scale, build.zero_point, build.quant_min,
                        build.quant_max, t.scheme_id, t.lo, t.hi, t.shift, t.G,
                        build.out_scale, build.out_zero_point, t.out_qmin, t.out_qmax)
        self.in_scale = float(np.float32(in_scale))     # dq1's multiplier
        self.op = build.op
        self.target = target
        self.fn_args = tuple(args)
        self.fn_kwargs = dict(kwargs or {})
        self.grad = grad
        self.describe = t.describe()
        self.nodes = nn.Parameter(self.table.to(torch.float32)) if learn else None

    def extra_repr(self):
        return f"{self.op}: {self.describe}; grad={self.grad}, learn={self.nodes is not None}"

    def forward(self, x):
        if torch.compiler.is_exporting() or not torch.is_grad_enabled():
            return Q16.LUT16(x, self.table, *self.scalars)
        return _Lut16Fn.apply(x, self.table if self.nodes is None else self.nodes, self)


def _input_codes(x, mod):
    """(c clamped into q1's range, inside q1's range before the clamp)."""
    inv, zp, qmin, qmax = mod.scalars[:4]
    c = Q16.quantize16(x, inv, zp, qmin, qmax)
    v = x * inv
    raw = torch.round(torch.where(v == v, v, torch.full_like(v, -Q16.RAIL)))
    raw = raw.clamp(-Q16.RAIL, Q16.RAIL) + zp
    return c, (raw >= qmin) & (raw <= qmax)


def _true_slope(c, mod):
    """f'(dq1(c)) by autograd on the ATen activation, float32."""
    zp = mod.scalars[1]
    with torch.enable_grad():
        xs = ((c - zp).to(torch.float32) * mod.in_scale).detach().requires_grad_(True)
        y = mod.target(xs, *mod.fn_args, **mod.fn_kwargs)
        (d,) = torch.autograd.grad(y, xs, torch.ones_like(y))
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _table_slope(c, table, mod):
    """d(output code)/d(input code) of the deployed function at c, float64;
    0 where c clamps to the table's ends (with a tail nothing clamps)."""
    (_, _, _, _, scheme, lo, hi, shift, G, _, _, _, _) = mod.scalars
    base = scheme & 7
    inside = (c >= lo) & (c <= hi)
    if scheme & lut_fit.TAIL:
        inside = torch.ones_like(inside)
    if base == 0:
        T = table.to(torch.float64)
        up = (c + 1).clamp(lo, hi)
        down = (c - 1).clamp(lo, hi)
        width = (up - down).clamp(min=1).to(torch.float64)
        d = (T[up - lo] - T[down - lo]) / width
        return torch.where(inside, d, torch.zeros_like(d))
    j, t, F = Q16.index16(c, scheme, lo, hi, shift)
    W = torch.exp2(F.to(torch.float64))
    if base in (1, 3):
        V = table.to(torch.float64)
        d = (V[j + 1] - V[j]) / (2.0 ** G) / W
    else:
        C = table.reshape(-1, 4).to(torch.float64)
        tf = t.to(torch.float64) / W
        d = (C[j, 1] + 2 * C[j, 2] * tf + 3 * C[j, 3] * tf * tf) / W
    if base in (3, 4):
        d = torch.where(c.clamp(lo, hi) < 0, -d, d)    # NSAU indexes |c|
    return torch.where(inside, d, torch.zeros_like(d))


def _node_grad(g, c, table, mod):
    """d(loss)/d(table entry): each output's gradient spread over the
    entries it read, by their weight in the device's interpolation (straight
    through the final rounding; 0 where the output sits on a rail)."""
    (_, _, _, _, scheme, lo, hi, shift, G, s2, _, oqmin, oqmax) = mod.scalars
    pre = Q16.codes16(c, table, scheme, lo, hi, shift, G, -_BIG, _BIG)
    live = ((pre >= oqmin) & (pre <= oqmax)).to(torch.float64)
    g = g.to(torch.float64) * s2 * live
    out = torch.zeros(table.numel(), dtype=torch.float64)
    if scheme & 7 == 0:
        out.index_add_(0, (c.clamp(lo, hi) - lo).reshape(-1), g.reshape(-1))
    elif scheme & 7 in (1, 3):
        j, t, F = Q16.index16(c, scheme, lo, hi, shift)
        a = t.to(torch.float64) / torch.exp2(F.to(torch.float64))
        out.index_add_(0, j.reshape(-1), (g * (1 - a) / 2.0 ** G).reshape(-1))
        out.index_add_(0, (j + 1).reshape(-1), (g * a / 2.0 ** G).reshape(-1))
    else:
        j, t, F = Q16.index16(c, scheme, lo, hi, shift)
        tf = t.to(torch.float64) / torch.exp2(F.to(torch.float64))
        for k in range(4):
            out.index_add_(0, (4 * j + k).reshape(-1), (g * tf ** k).reshape(-1))
    return out.reshape(table.shape).to(torch.float32)


class _Lut16Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, entries, mod):
        if mod.nodes is not None:
            (_, _, _, _, scheme, _, _, _, _, _, _, oqmin, oqmax) = mod.scalars
            table = _round_table(entries.detach(), scheme, oqmin, oqmax)
            with torch.no_grad():
                mod.table.copy_(table)
        else:
            table = entries
        ctx.mod = mod
        ctx.save_for_backward(x, table)
        return Q16.LUT16(x, table, *mod.scalars)

    @staticmethod
    def backward(ctx, g):
        x, table = ctx.saved_tensors
        mod = ctx.mod
        c, inside = _input_codes(x.detach(), mod)
        gx = gnodes = None
        if ctx.needs_input_grad[0]:
            if mod.grad == "true":
                d = _true_slope(c, mod).to(torch.float64)
            else:
                d = _table_slope(c, table, mod) * (mod.scalars[9] / mod.in_scale)
            gx = (g.to(torch.float64) * d * inside.to(torch.float64)).to(g.dtype)
        if mod.nodes is not None and ctx.needs_input_grad[1]:
            gnodes = _node_grad(g, c, table, mod)
        return gx, gnodes, None


# --- fit_tables -------------------------------------------------------------------------

def _activation_observer(module):
    from torchao.quantization.pt2e.observer import _is_activation_post_process

    return _is_activation_post_process(module)


def _static_int16(module):
    return (lut16_range._static_per_tensor(module) and module.dtype == torch.int16)


def _qparams(module):
    scale, zero_point = module.calculate_qparams()
    return (float(scale), int(zero_point), int(module.quant_min), int(module.quant_max),
            module.dtype)


def _frozen_forward(self, x):
    """A PTQ observer that no longer updates: forward passes x through, and
    calculate_qparams answers from the statistics it had."""
    return x


_FROZEN = {}


def freeze_activation_observers(prepared):
    """Stop every static activation observer of a prepared model where it
    is: a fake-quantiser's observer off (it keeps fake-quantising on its
    frozen qparams), a PTQ observer's forward a pass-through (its class
    becomes a Frozen subclass in place). Weight observers and dynamic
    activations keep observing. Returns the number frozen."""
    readers = {}
    for node in prepared.graph.nodes:
        if node.op == "call_module" and node.args:
            readers.setdefault(node.target, []).append(node.args[0])
    frozen = 0
    for name, module in prepared.named_modules(remove_duplicate=False):
        if not _activation_observer(module) or not lut16_range._static_per_tensor(module):
            continue
        inputs = readers.get(name, [])
        if not any(isinstance(x, torch.fx.Node) and x.op != "get_attr" for x in inputs):
            continue
        if hasattr(module, "disable_observer"):
            module.disable_observer()
        elif type(module) not in _FROZEN.values():
            cls = type(module)
            if cls not in _FROZEN:
                _FROZEN[cls] = type(f"Frozen{cls.__name__}", (cls,),
                                    {"__module__": __name__, "forward": _frozen_forward})
            module.__class__ = _FROZEN[cls]
        frozen += 1
    return frozen


def _site(S, modules):
    """(obs1 node, obs2 nodes) when S is an int16 table site of the prepared
    graph, else None."""
    if S.op != "call_function" or S.target not in fold_quant_lut_pass.ALLOWLIST or not S.args:
        return None
    if fold_quant_lut_pass._has_node(list(S.args[1:])) or fold_quant_lut_pass._has_node(dict(S.kwargs)):
        return None
    o1 = S.args[0]
    if not (isinstance(o1, torch.fx.Node) and o1.op == "call_module"):
        return None
    m1 = modules.get(o1.target)
    if m1 is None or not _activation_observer(m1) or not _static_int16(m1):
        return None
    users = list(S.users)
    if not users or not all(u.op == "call_module" and u.args and u.args[0] is S
                            for u in users):
        return None
    m2 = [modules.get(u.target) for u in users]
    if any(m is None or not _activation_observer(m) or not _static_int16(m) for m in m2):
        return None
    if len({_qparams(m) for m in m2}) != 1:
        return None
    return o1, users


def fit_tables(prepared, spec=None, *, grad="true", learn=False, ranges=True,
               margin=lut16_range.MARGIN, strict=True):
    """Fit a lut16 table for every int16 activation site of a calibrated
    PT2E model and train through it from here on (module docstring).

    spec      lut_fit.Spec: the byte budget (default 8 KB) and an optional
              forced scheme/size; the chooser takes the smallest table
              within the flips bar
    grad      "true" (default) or "table": Lut16's backward for x
    learn     the table entries learn too
    ranges    fit over the input observer's [min_val, max_val] widened by
              `margin` (lut16_range.window), as the lowering
              does with carried ranges; False: every code of q1 (the
              graph-only fit)
    strict    a table must meet the flips bar, else its site keeps the
              activation (lut16-budget); False takes the best candidate
              within the budget anyway (studies of small tables)

    Freezes the static activation observers first, so the tables and the
    grids around them stay as fitted. Returns a fold_lut16_pass.Lut16Report
    (tables by module name; sites the budget refuses keep their
    activation)."""
    spec = spec or lut_fit.Spec()
    freeze_activation_observers(prepared)
    graph = prepared.graph
    modules = dict(prepared.named_modules(remove_duplicate=False))
    report = Q16.Lut16Report()
    taken = set(modules)
    for S in list(graph.nodes):
        site = _site(S, modules)
        if site is None:
            continue
        o1, o2 = site
        m1 = modules[o1.target]
        in_qp = _qparams(m1)
        out_qp = _qparams(modules[o2[0].target])
        if not (Q16._int16_grid(in_qp) and Q16._int16_grid(out_qp)):
            continue
        op = fold_quant_lut_pass.ALLOWLIST[S.target]
        win = (lut16_range.window(lut16_range.observer_range(m1), *in_qp[:4],
                                  margin=margin) if ranges else None)
        args, kwargs = tuple(S.args[1:]), dict(S.kwargs)
        table, error = Q16.fit_parts(S.target, args, kwargs, in_qp, out_qp, spec, win)
        if table is None and not strict and getattr(error, "best", None) is not None:
            table = error.best
        if table is None:
            report.refused.append((S.name, op, "lut16-budget", str(error)))
            continue
        s1, z1, qmin1, qmax1, _ = in_qp
        s2, z2 = out_qp[:2]
        build = Q16.Lut16Build(op, table, fold_quant_lut_pass.reciprocal(s1), z1, qmin1, qmax1,
                               float(np.float32(s2)), z2)
        module = Lut16(build, s1, S.target, args, kwargs, grad=grad, learn=learn)
        i = 0
        while f"lut16_{i}" in taken:
            i += 1
        name = f"lut16_{i}"
        taken.add(name)
        prepared.add_submodule(name, module)
        with graph.inserting_before(S):
            node = graph.call_module(name, (o1.args[0],))
        node.meta.update({k: S.meta[k] for k in ("val", "tensor_meta", "stack_trace",
                                                 "nn_module_stack") if k in S.meta})
        S.replace_all_uses_with(node)
        graph.erase_node(S)
        if not o1.users:
            graph.erase_node(o1)
        report.tables.append((name, build))
    if report.tables:
        graph.lint()
        prepared.recompile()
    return report


def tables(model):
    """{module name: Lut16} of a model (prepared, converted or not)."""
    return {name: m for name, m in model.named_modules() if isinstance(m, Lut16)}


def check_tables(model):
    """Raise ValueError if a trained table left what the runtime accepts:
    linear nodes whose interpolation could leave int32, cubic coefficients
    past 2^19 or, with a tail, edge cubics past 2^22 (quant_lut16.cpp's
    checks; mirrors.py's own predicates)."""
    from executorch.backends.esp32s3.quantizer import mirrors as qm

    for name, m in tables(model).items():
        (_, _, _, _, scheme, lo, hi, shift, G, _, _, _, _) = m.scalars
        data = m.table.detach().numpy()
        if scheme & 7 in (1, 3) and not qm.lut16_linear_safe(
                scheme, data.astype(np.int64), lo, hi, shift, G):
            raise ValueError(f"{name}: trained linear nodes are not int32-safe")
        if scheme & 7 in (2, 4) and not np.all(np.abs(data) <= lut_fit.COEF_MAX):
            raise ValueError(f"{name}: trained cubic coefficients past 2^19")
        if scheme & 7 in (2, 4) and not qm.lut16_cubic_safe(scheme, data, lo, hi, shift):
            raise ValueError(f"{name}: a trained tail's edge cubic passes 2^22")
        if not np.all(np.isfinite(data.astype(np.float64))):
            raise ValueError(f"{name}: non-finite table entries")
