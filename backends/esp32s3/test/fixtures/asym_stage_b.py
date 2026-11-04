# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage B: int8 codes between claimed ops and across delegate boundaries,
end to end, with Stage A's bits as the bar.

Under Esp32Partitioner(stage_b=True) a static int8 graph's codes travel as
codes (stage_b.py): a claimed op writes its output quantize's int8 codes
(EXT_OUT_QUANT out_dtype 1), the next reads them (EXT_ACT_QUANT in_dtype 1),
through an int8 chain slot inside one delegate and as an int8 tensor between
two, where the portable ops between them run on codes. Each row here is a
static block quantised by a stock flow (quant_census.recipes), lowered twice:
its Stage A twin (sa_*) and its Stage B variant (sb_*), both held to the same
output bits -- Stage A's, which asym_static.StaticMirror reproduces
(the device's arithmetic for every matmul and convolution, torch's for the
rest). The QEMU image holds both, so the device proves Stage B == Stage A in
one run.

  mlp      XNNPACK static Linear + ReLU -> Linear (st_xnn_mlp's block): the
           two chain through an int8 slot (row blocks of 4)
  mlp1     the same block on one row (the decode walk, run_rows_affine)
  mlp3d    the same block on a 3-D input: each Linear is view -> addmm ->
           view, so the two land in two delegates with the views between
           them, run on codes
  conv     XNNPACK static 3x3 convs, padded + ReLU, stride 2, dilation 2
           (st_xnn_conv's): two int8 chain slots, each conv's padding the
           codes' zero point
  dwconv   Arm a8w8 depthwise 3x3 + ReLU, grouped 3x3, 1x1 (st_arm_dwconv's):
           the depthwise and grouped walks on codes
  res      XNNPACK static residual block: Linear + ReLU chained into a Linear
           on codes inside one delegate, a residual add of dequantised values
           (kept fp32 by name: an add needs fp32), then a Linear
  poolcat  XNNPACK static conv + ReLU, max_pool, two convs, cat, conv: int8
           tensors across four delegates, max_pool and cat on codes
  siglut   XNNPACK static Linear, sigmoid, Linear (st_xnn_sig_lut's block):
           espressif::quant_lut_codes between two delegates on codes
  mbnet    XNNPACK static MobileNet block: conv, depthwise, pointwise,
           depthwise stride 2, pointwise, each ReLU-fused, chained on codes
           (depthwise convolutions reading codes)
  sig3d    siglut's block on a 3-D input (each Linear traced as view ->
           addmm -> view): the views and the code table between the two
           delegates, all on codes
  pad      XNNPACK static conv + ReLU, an explicit F.pad, conv: the pad runs
           between the two delegates on codes, padding with 0.0's code
  fanout   XNNPACK static conv + ReLU whose codes feed a conv in its own
           delegate (a chain) AND a max_pool outside it (an int8 output), the
           two branches concatenated on codes, then a conv

and split-pass entries (sb_mlp_split, sb_mlp1_split): the Stage B programs
of mlp and mlp1 run again under the min_latency override, their columns
split across the two cores, codes written by both. The other static int8 rows
Stage B changes ride as Stage B variants alone (mlp_pt, mlp_qat, mlp_arm,
mlp_cm, mlp_u8io: st_xnn_mlp_pt's, st_xnnqat_mlp's, st_arm_mlp's,
st_cm_mlp's and st_arm_u8io's blocks); their Stage A twins are those rows in
asymstat, held to the same bits. The rest of the static int8 rows and all the
int16 ones keep Stage A's blobs byte for byte under the spec (test_stage_b.py).

export_all(out_dir) writes every program and the packed image main.cpp reads
from the asymstgb flash partition (asym_stageb_fixtures.bin; asym.py's
layout); it returns {row: reference}. Imports torch.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.test.fixtures import asym_static as st
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
IMAGE = "asym_stageb_fixtures.bin"


class ResChain(nn.Module):
    """Linear + ReLU -> Linear, plus the input (a residual add), -> Linear."""

    def __init__(self, seed=71):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc3(self.fc2(F.relu(self.fc1(x))) + x)


class MbBlock(nn.Module):
    """conv s2, depthwise, pointwise, depthwise s2, pointwise; ReLU after all
    but the last."""

    def __init__(self, seed=83):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.dw1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw1 = nn.Conv2d(8, 16, 1)
        self.dw2 = nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.dw1(x))
        x = F.relu(self.pw1(x))
        x = F.relu(self.dw2(x))
        return self.pw2(x)


class Fanout(nn.Module):
    """conv + ReLU read by a strided conv and, pooled, by a 1x1 conv; the two
    concatenated, then a conv."""

    def __init__(self, seed=79):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(8, 8, 1)
        self.c4 = nn.Conv2d(16, 4, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.c1(x))
        return self.c4(torch.cat([self.c2(h), self.c3(F.max_pool2d(h, 2))],
                                 dim=1))


class PadConv(nn.Module):
    """conv + ReLU, F.pad by 1 (constant 0.0), an unpadded conv."""

    def __init__(self, seed=89):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 4, 3)

    def forward(self, x):
        return self.c2(F.pad(F.relu(self.c1(x)), (1, 1, 1, 1)))


class ConvPoolCat(nn.Module):
    """conv + ReLU, max_pool, two convs side by side, cat, conv."""

    def __init__(self, seed=73):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.ca = nn.Conv2d(8, 8, 3, padding=1)
        self.cb = nn.Conv2d(8, 8, 1)
        self.c2 = nn.Conv2d(16, 4, 3, padding=1)

    def forward(self, x):
        p = F.max_pool2d(F.relu(self.c1(x)), 2)
        return self.c2(torch.cat([self.ca(p), self.cb(p)], dim=1))


@dataclass
class Row:
    name: str
    recipe: str
    build: object
    shape: tuple
    luts: bool = True
    # What Stage B must have done: producers that write codes, ops between
    # delegates on codes, and the boundaries it keeps fp32, by rule.
    producers: int = 0
    code_ops: tuple = ()
    kept: tuple = ()
    split: bool = False   # its Stage B program again, split across the cores
    # Its Stage A twin rides here too; False for a static int8 row's block,
    # whose Stage A program is that row's (asymstat), held to the same bits.
    twin: bool = True


SPECS = (
    Row("mlp", "xnnpack-static", st.MLP, (4, 64), producers=1, split=True),
    Row("mlp1", "xnnpack-static", st.MLP, (1, 64), producers=1, split=True),
    Row("mlp3d", "xnnpack-static", st.MLP, (1, 8, 64), producers=1),
    Row("conv", "xnnpack-static", st.ConvNet, (1, 4, 9, 9), producers=2),
    Row("dwconv", "arm-a8w8", st.DwGrouped, (1, 8, 7, 7), producers=2),
    Row("res", "xnnpack-static", ResChain, (4, 64), producers=1,
        kept=("stage-b-reader",)),
    Row("poolcat", "xnnpack-static", ConvPoolCat, (1, 4, 10, 10), producers=3,
        code_ops=("pool", "cat")),
    Row("siglut", "xnnpack-static", lambda: st.ActMLP(torch.sigmoid), (4, 64),
        producers=1, code_ops=("table",)),
    Row("sig3d", "xnnpack-static", lambda: st.ActMLP(torch.sigmoid),
        (1, 8, 64), producers=1, code_ops=("table",)),
    Row("mbnet", "xnnpack-static", MbBlock, (1, 4, 12, 12), producers=4),
    Row("pad", "xnnpack-static", PadConv, (1, 4, 8, 8), producers=1,
        code_ops=("pad",)),
    Row("fanout", "xnnpack-static", Fanout, (1, 4, 8, 8), producers=3,
        code_ops=("pool", "cat")),
    # The other static int8 rows Stage B changes, as Stage B variants only:
    # st_xnn_mlp_pt, st_xnnqat_mlp, st_arm_mlp, st_cm_mlp, st_arm_u8io.
    Row("mlp_pt", "xnnpack-static-pt", st.MLP, (4, 64), producers=1,
        twin=False),
    Row("mlp_qat", "xnnpack-qat", st.MLP, (4, 64), producers=1, twin=False),
    Row("mlp_arm", "arm-a8w8", st.MLP, (4, 64), producers=1, twin=False),
    Row("mlp_cm", "cortex-m", st.MLP, (4, 64), producers=1, twin=False),
    Row("mlp_u8io", "arm-a8w8-u8io", st.MLP, (4, 64), producers=1,
        twin=False),
)
ROWS = (tuple(f"{prefix}_{r.name}" for r in SPECS
              for prefix in (("sa", "sb") if r.twin else ("sb",)))
        + tuple(f"sb_{r.name}_split" for r in SPECS if r.split))
QEMU_ROWS = ROWS


@dataclass
class Fixture(st.Fixture):
    stage_b: bool = False
    row: Row = None
    stage_report: object = None


def fixtures():
    """Every row's Stage A twin and Stage B variant, one quantised model and
    one reference (the device mirror's: Stage A's bits) per row."""
    out = []
    for row in SPECS:
        spec = st.Row(row.name, row.recipe, row.build, row.shape, luts=row.luts)
        model, inputs, _info = st.quantized(spec)
        ref = st.mirror(model, inputs, row.luts)
        gap = st.eager_gap(model, inputs, ref)
        for stage_b in ((False, True) if row.twin else (True,)):
            fx = Fixture(f"{'sb' if stage_b else 'sa'}_{row.name}", model,
                         inputs, ref, {}, luts=row.luts, stage_b=stage_b,
                         row=row)
            fx.eager = gap
            out.append(fx)
    for row in SPECS:
        if row.split:
            twin = next(f for f in out if f.name == f"sb_{row.name}")
            out.append(replace(twin, name=f"sb_{row.name}_split",
                               split_pass=True, program_of=twin.name))
    assert tuple(f.name for f in out) == ROWS
    return out


def lower_program(model, inputs, luts=True, stage_b=False, partitioner=None):
    """lower_esp32 (+ apply_passes, to_executorch) with or without Stage B,
    the partitioner's other options from `partitioner` (a fixture's).
    Returns (program, the backend's log, the static q/dq report)."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(model, inputs, strict=False)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lowered = _passes.lower_esp32(
            ep, [Esp32Partitioner(stage_b=stage_b, **(partitioner or {}))],
            quant_luts=luts)
        _passes.apply_passes(lowered, quiet=True)
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue(), lowered.static_qdq_report


def lower(fixture):
    """One fixture's program; a Stage B variant must have done what its row
    says (StageBReport against Row), so a regression that quietly keeps a
    region fp32 fails the export."""
    et, log, static = lower_program(fixture.model, fixture.inputs, fixture.luts,
                                    fixture.stage_b)
    report = static.stage_b if static is not None else None
    fixture.stage_report = report
    if fixture.stage_b:
        row = fixture.row
        kinds = tuple(sorted({k for _, k in report.code_ops} - {"move"}))
        rules = tuple(sorted({rule for _, rule, _ in report.kept}))
        if (len(report.converted) != row.producers
                or kinds != tuple(sorted(row.code_ops))
                or rules != tuple(sorted(row.kept)) or report.identity):
            raise RuntimeError(
                f"{fixture.name}: Stage B did not do what the row says "
                f"(producers {len(report.converted)} != {row.producers}, ops "
                f"{kinds} != {row.code_ops}, kept {rules} != {row.kept}, "
                f"identity failures {report.identity}):\n{report.summary()}")
    return et, log


def emitted_programs():
    """Every row's emitted program (dump_required_ops: the portable kernels
    the parity firmware must register for them -- the ops on codes)."""
    return [lower(f)[0] for f in fixtures() if not f.split_pass]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte) and the QEMU image (asym_stageb_fixtures.
    bin). Returns {row: reference}; each Stage B variant's report is on
    export_all.reports."""
    table = fixtures()
    programs, refs = {}, {}
    export_all.reports = {}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        if fixture.split_pass:
            continue
        et, log = lower(fixture)
        programs[fixture.name] = bytes(et.buffer)
        if fixture.stage_b:
            export_all.reports[fixture.name] = fixture.stage_report
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(et.buffer)
        if verbose:
            ops = [line.split(": ", 1)[-1] for line in log.splitlines()
                   if "[Esp32Backend]" in line]
            print(f"  {fixture.name}.pte: {len(et.buffer):,} B; "
                  + "; ".join(ops))
            if fixture.stage_b:
                print(f"    {fixture.stage_report.summary()}")
    ea.write_image(os.path.join(out_dir, IMAGE), table, programs)
    return refs


if __name__ == "__main__":
    for name, ref in export_all().items():
        print(f"  {name}: reference {np.shape(ref)}")
