# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""espressif::qnorm parity rows: the stock recipes' decomposed norms as one
op, on QEMU, held to the mirror's bits.

Each row is a Linear, a norm and (for some) a second Linear, quantised by a
stock static flow (quant_census.recipes, asym_static's calibration),
lowered with Esp32Partitioner(fused_norms=True) so that
fuse_qnorm_pass.fold_qnorms rewrites the norm before to_edge:

  qn_arm_rms      Arm a8w8, HF's RMSNorm (pow, mean, eps, rsqrt, two
                  multiplies) over 24: Arm's pieces, a pair after each
  qn_arm_rms64    the same over 64, read by a second Linear
  qn_arm_ln       Arm a8w8, nn.LayerNorm(24): two means, two subtractions,
                  the square, the variance, eps, rsqrt, weight and bias
  qn_arm_ln64     the same over 64, read by a second Linear
  qn_cm_rms       Cortex-M, the same RMSNorm: fp32 x^2, mean and eps, a pair
                  around the rsqrt
  qn_arm_ln_tail  qn_arm_ln on 5 rows of 21 (a width that is no multiple of
                  NEON's 8, so ATen's sum runs a scalar tail)
  qn_a16_rms      Arm a16w8, the RMSNorm over 24 on int16 codes (qnorm kind
                  2: x^2 per element, the mean, eps and rsqrt as a program
                  with an espressif::lut16 rsqrt table)
  qn_a16_rms64    the same over 64, read by a second Linear
  qn_a16_ln       Arm a16w8, nn.LayerNorm(24) on int16 codes (kind 3)
  qn_a16_ln64     the same over 64, read by a second Linear

The device's output bits are the mirror's: asym_static.StaticMirror on
the folded program, whose qnorm node runs this op's torch reference (the
kernel's arithmetic, mirrors.qnorm's too). Against the recipe's eager
model each row reports NormGap: the reductions' codes (at most one step
from eager's, the flip rate reported) and, for every row whose reduction
codes all equal eager's, the output bit for bit (the bar's "exact
elsewhere"). Cortex-M's LayerNorm is not a row: its recipe keeps
layer_norm whole between two pairs, no pair inside (st_cm_layernorm).

export_all(out_dir) writes every program and the image main.cpp reads from
the asymnorm flash partition (asym_norm_fixtures.bin, asym.py's layout);
it returns {row: reference}. Imports torch.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.test.fixtures import asym_static as st
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
IMAGE = "asym_norm_fixtures.bin"


class RMSNorm(nn.Module):
    """HF's LlamaRMSNorm, a random weight."""

    def __init__(self, n, eps=1e-6, seed=211):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(1.0 + 0.25 * torch.randn(n, generator=g))
        self.eps = eps

    def forward(self, x):
        h = x.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        return self.weight * h.to(x.dtype)


def layer_norm(n, seed=223):
    norm = nn.LayerNorm(n)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.25 * torch.randn(n, generator=g))
        norm.bias.copy_(0.1 * torch.randn(n, generator=g))
    return norm


@dataclass
class Row:
    name: str
    recipe: str
    build: object          # () -> float nn.Module
    shape: tuple
    norms: int = 1         # fused norms the program must hold
    # the norm, and the pair after it that the graph output reads
    portable: tuple = ("espressif.qnorm", "quantize_per_tensor",
                       "dequantize_per_tensor")


SPECS = (
    Row("qn_arm_rms", "arm-a8w8", lambda: st.Reduce(RMSNorm(24)), (4, 32)),
    Row("qn_arm_rms64", "arm-a8w8",
        lambda: st.Reduce(RMSNorm(64), n=64, consumer=True), (8, 32)),
    Row("qn_arm_ln", "arm-a8w8", lambda: st.Reduce(layer_norm(24)), (4, 32)),
    Row("qn_arm_ln64", "arm-a8w8",
        lambda: st.Reduce(layer_norm(64), n=64, consumer=True), (8, 32)),
    Row("qn_cm_rms", "cortex-m", lambda: st.Reduce(RMSNorm(24)), (4, 32)),
    Row("qn_arm_ln_tail", "arm-a8w8",
        lambda: st.Reduce(layer_norm(21), n=21), (5, 32)),
    Row("qn_a16_rms", "arm-a16w8", lambda: st.Reduce(RMSNorm(24)), (4, 32)),
    Row("qn_a16_rms64", "arm-a16w8",
        lambda: st.Reduce(RMSNorm(64), n=64, consumer=True), (8, 32)),
    Row("qn_a16_ln", "arm-a16w8", lambda: st.Reduce(layer_norm(24)), (4, 32)),
    Row("qn_a16_ln64", "arm-a16w8",
        lambda: st.Reduce(layer_norm(64), n=64, consumer=True), (8, 32)),
)
ROWS = tuple(r.name for r in SPECS)


def quantized(row):
    from executorch.backends.esp32s3.test.quant_census import recipes

    inputs = (ea.ramp(row.shape),)
    q = recipes.quantize(row.recipe, row.build().eval(), inputs,
                         st.calibration(row.shape))
    return q.program, q.inputs


def folded(model, inputs):
    """The export lower_esp32 sees, its norms folded (fold_qnorms, then
    fold_quant_luts, lower_esp32's order): the program StaticMirror runs."""
    from executorch.backends.esp32s3._passes import fuse_qnorm_pass
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    ep = torch.export.export(model, inputs, strict=False)
    report = fuse_qnorm_pass.fold_qnorms(ep)
    fold_quant_lut_pass.fold_quant_luts(ep)
    return ep, report


def mirror(model, inputs):
    ep, report = folded(model, inputs)
    with torch.no_grad():
        out = st.StaticMirror(ep.module(), garbage_collect_values=False).run(
            *inputs)
    if isinstance(out, (tuple, list)):
        return st.flat(out), report
    return np.asarray(out.detach().numpy(), F32), report


# --- the bar against eager ------------------------------------------------------------

@dataclass
class NormGap:
    """A row's fused norms against the recipe's eager model on the same
    input: each reduction's code per row (at most one step apart), and the
    norm's output bit for bit wherever every reduction code agrees."""

    norms: int = 0
    reductions: int = 0       # reduction codes compared (where their inputs
                              # are eager's: a variance after an agreeing mean)
    flips: int = 0            # ... that differ from eager's
    steps: int = 0            # the largest difference
    rows: int = 0             # norm rows compared
    rows_exact: int = 0       # rows whose codes all agree and whose output
    rows_agree: int = 0       # bits are eager's / rows whose codes all agree

    def ok(self):
        return self.steps <= 1 and self.rows_exact == self.rows_agree

    def summary(self):
        rate = self.flips / max(self.reductions, 1)
        return (f"{self.norms} norm(s): {self.flips} of {self.reductions} "
                f"reduction code(s) differ from eager (rate {rate:.2e}, at most "
                f"{self.steps} step); {self.rows_exact} of {self.rows_agree} "
                f"rows with eager's codes exact ({self.rows} rows)")

    def add(self, other):
        for name in ("norms", "reductions", "flips", "rows", "rows_exact",
                     "rows_agree"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.steps = max(self.steps, other.steps)


def norm_eager_gap(model, batches):
    """NormGap over input batches: the eager program (an fx Interpreter over
    the unfolded export, run on each batch) against the fused op's reference
    on eager's own input, at the reductions' quantisers (their nodes, matched
    by name in the export) and at the output."""
    from executorch.backends.esp32s3._passes import fuse_qnorm_pass

    gap = NormGap()
    probe = torch.export.export(model, batches[0], strict=False)
    graph = probe.graph_module.graph
    matches = []
    for R in fuse_qnorm_pass._candidates(graph):
        try:
            m = fuse_qnorm_pass.match(probe, R)
            fuse_qnorm_pass._check(probe, m)
        except fuse_qnorm_pass.Refused:
            continue
        matches.append((m, fuse_qnorm_pass.build(probe, m)))
    gap.norms = len(matches)
    module = probe.module() if matches else None
    for batch in batches if matches else ():
        interp = torch.fx.Interpreter(module, garbage_collect_values=False)
        with torch.no_grad():
            interp.run(*batch)
        values = {n.name: v for n, v in interp.env.items()}
        for m, b in matches:
            x = values[m.x.name]
            want = values[m.output.name]
            with torch.no_grad():
                got, codes = fuse_qnorm_pass.reference_with_codes(
                    x, b.tables, b.quant, b.weight, b.bias, b.kind, b.flags,
                    b.eps)
            # A reduction is compared where its inputs are eager's: always for
            # the sums over x (RMSNorm's, a LayerNorm's two means), and for a
            # LayerNorm's variance where the mean it subtracts (RED2's code)
            # agreed; past a mean's flip the variance sums other values.
            agree = torch.ones(codes[0].shape[0], dtype=torch.bool)
            diffs = {}
            for slot, mine in zip(fuse_qnorm_pass.reduction_slots(b.kind), codes):
                q = m.slots[slot]
                eager = values[q.name].to(torch.int64).reshape(-1)
                diffs[slot] = (mine.reshape(-1) - eager).abs()
            for slot, diff in diffs.items():
                compared = (diffs["red2"] == 0 if slot == "red3"
                            else torch.ones_like(diff, dtype=torch.bool))
                gap.reductions += int(compared.sum())
                gap.flips += int(((diff > 0) & compared).sum())
                gap.steps = max(gap.steps, int((diff * compared).max()))
                agree &= diff == 0
            n = x.shape[-1]
            same = (got.reshape(-1, n).contiguous().view(torch.int32)
                    == want.reshape(-1, n).contiguous().view(torch.int32)).all(1)
            gap.rows += int(agree.numel())
            gap.rows_agree += int(agree.sum())
            gap.rows_exact += int((same & agree).sum())
    return gap


def gap_batches(shape, count=48, seed=233):
    """The device's ramp, the calibration batches and `count` more random
    ones (not seen by the observers): the flip rate's sample."""
    g = torch.Generator().manual_seed(seed)
    batches = [(b[0],) for b in st.calibration(shape)]
    batches += [(torch.randn(shape, generator=g) * 0.9 + 0.5,)
                for _ in range(count)]
    return batches


# --- lowering and the image -------------------------------------------------------------

@dataclass
class Fixture(ea.Fixture):
    gap: NormGap | None = None
    report: str = ""
    norms: int = 1
    keeps: tuple = ()
    outputs: int = 1        # asym_static.write_image's flags read
    bar: str = "exact"      # these two


def fixtures():
    out = []
    for row in SPECS:
        model, inputs = quantized(row)
        ref, report = mirror(model, inputs)
        if len(report.fused) != row.norms:
            raise RuntimeError(f"{row.name}: {len(report.fused)} norm(s) "
                               f"fused, expected {row.norms}: "
                               f"{report.summary()}")
        fx = Fixture(row.name, model, inputs, ref, {}, norms=row.norms,
                     keeps=row.portable)
        ep = torch.export.export(model, inputs, strict=False)
        fx.outputs = len(st.output_nodes(ep.module()))
        fx.report = report.summary()
        fx.gap = norm_eager_gap(model, gap_batches(row.shape))
        out.append(fx)
    assert tuple(f.name for f in out) == ROWS
    return out


def lower(fixture, fused_norms=True):
    """The fixture through lower_esp32 with Esp32Partitioner(fused_norms=...)
    and apply_passes: nothing left outside the delegates but the norm (and,
    unfused, what its pieces need). Returns (program, the backend's log)."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(fixture.model, fixture.inputs, strict=False)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lowered = _passes.lower_esp32(
            ep, [Esp32Partitioner(fused_norms=fused_norms)])
        _passes.apply_passes(lowered, quiet=True)
    graph = lowered.exported_program().graph_module.graph
    if fused_norms:
        norms = sum(n.op == "call_function" and "qnorm" in str(n.target)
                    for n in graph.nodes)
        left = [str(n.target) for n in graph.nodes if n.op == "call_function"
                and "call_delegate" not in str(n.target)
                and "getitem" not in str(n.target)
                and not any(k in str(n.target) for k in fixture.keeps)]
        if norms != fixture.norms or left:
            raise RuntimeError(f"{fixture.name}: {norms} qnorm node(s), ops "
                               f"left portable: {left}\n{log.getvalue()}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue()


def emitted_programs():
    """Every row's emitted program (dump_required_ops)."""
    return [lower(f)[0] for f in fixtures()]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte) and the QEMU image (asym_norm_fixtures.bin).
    Returns {row: reference}; each row's gap to eager is on export_all.gaps."""
    table = fixtures()
    programs, refs = {}, {}
    export_all.gaps = {}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        export_all.gaps[fixture.name] = fixture.gap
        et, log = lower(fixture)
        programs[fixture.name] = bytes(et.buffer)
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(et.buffer)
        if verbose:
            print(f"  {fixture.name}.pte: {len(et.buffer):,} B; "
                  f"{fixture.report}; vs eager: {fixture.gap.summary()}")
    st.write_image(os.path.join(out_dir, IMAGE), table, programs)
    return refs


if __name__ == "__main__":
    for name, ref in export_all().items():
        print(f"  {name}: reference {np.shape(ref)}; "
              f"{export_all.gaps[name].summary()}")
