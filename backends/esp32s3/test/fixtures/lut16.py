# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The int16 table op's parity rows (espressif::lut16, fold_lut16_pass.py):
asym_lut16_fixtures.bin, partition subtype 0x5E.

One program per Arm a16w8 activation site -- sigmoid, SiLU, GELU, tanh, exp,
rsqrt and reciprocal -- each a hand-written int16 pair around the activation
on a16w8's grid (int16 [-32767, 32767], zero point 0; s = 2^-12, where the
HistogramObserver's eps floor puts every census scale; rsqrt's output scale
2^-9 holds its 64), lowered with luts16=True: the activation becomes a fitted
lut16 table and its trailing pair goes.

The input is x * 0 + V, V a constant holding the row's values, so the
device's fixed ramp feeds exactly what is chosen here: every 64th code of
[-32767, 32767] (1,024), then edge codes (the rails, +-1, 0, the small codes a
pole's table handles one by one, past the rails and past the quantiser's
+-2^16 clamp) and quantiser ties (k + 1/2 beside 0 and both rails).

Expected: device bits == the numpy mirror (mirrors.lut16) == the folded
program's torch reference == the host runner, exactly; flips against the eager
graph (the activation by ATen, then q2) within the bar, reported with their
input codes.

The flagged scheme words ride in these rows: q16_gelu's table is a
TAIL table (GELU is x past ~5.4 and 0 below ~-4.1: the table stops there and
its edge segments, those lines, continue to the rails), q16_rsqrt's and
q16_reciprocal's NSAU tables take OFFSET (the negative half from segment 1),
and one more row carries an observed range:

  q16_recip_window  reciprocal on a16w8's grid with espressif::observed_range
                    between its q1 and dq1 (lut16_range: the range a
                    calibration observed on llama3_2's first reciprocal, codes
                    4,096..32,597), so the table is fitted over its window,
                    [534, 32767]: NSAU cubic with OFFSET, from the window's
                    first segment. Its inputs stay inside the window: a code
                    past it clamps, by design, and would flip against eager.

One more row trains through a table (lut16_train.py):

  qat_lut16   a nano MLP, Linear(64, 64) -> SiLU -> Linear(64, 32), under
              Esp32Quantizer(preset="arm_a16w8", is_qat=True): calibrated,
              its SiLU swapped for a lut16 table by espressif_quantizer.
              fit_tables (over the observer's window, learnable entries),
              fine-tuned 40 steps on a teacher's labels, converted and
              lowered with no refit. Expected: device bits == the mirror of
              the trained program (asym_static.StaticMirror: the
              delegate's integer Linears, lut16 by its torch reference), and
              == the trained graph run eagerly in torch (bar 0 flips); the
              table's bytes in the program == the trained module's.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.quantizer import lut16_range  # noqa: F401  (registers espressif::observed_range)
from executorch.backends.esp32s3._passes import fold_lut16_pass  # noqa: F401  (registers espressif::lut16)
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test import firmware

ROOT = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
IMAGE = "asym_lut16_fixtures.bin"
SUBTYPE = 0x5E
PARTITION_BYTES = 0x30000
FIXTURE_ALL_OUTPUTS = 4
S12 = 2.0 ** -12
QMIN, QMAX = -32767, 32767

# (row, activation, s1, s2)
SITES = (
    ("q16_sigmoid", torch.sigmoid, S12, S12),
    ("q16_silu", F.silu, S12, S12),
    ("q16_gelu", F.gelu, S12, S12),
    ("q16_tanh", torch.tanh, S12, S12),
    ("q16_exp", torch.exp, S12, S12),
    ("q16_rsqrt", torch.rsqrt, S12, 2.0 ** -9),
    ("q16_reciprocal", torch.reciprocal, S12, S12),
    ("q16_recip_window", torch.reciprocal, S12, S12),
)
# rows whose q1 carries an observed range: (min, max) in units of s1
OBSERVED = {"q16_recip_window": (4096, 32597)}


def input_codes(row=None):
    """The row's input, in units of s1: a strided sweep, edges, ties (+ 1/2);
    for a row with an observed range, the same kinds of codes inside its
    window (lut16_range.window)."""
    if row in OBSERVED:
        lo, hi = lut16_range.window(tuple(v * S12 for v in OBSERVED[row]), S12, 0,
                                    QMIN, QMAX)
        sweep = np.arange(lo, hi + 1, 32, dtype=np.float64)
        edges = [lo, lo + 1, lo + 2, 1023, 1024, 1025, 2047, 2048, 4096, hi - 1, hi]
        ties = [k + 0.5 for k in (lo, lo + 1, 1000, 1023, 4095, 20000, hi - 1)]
    else:
        sweep = np.arange(QMIN, QMAX + 1, 64, dtype=np.float64)
        edges = [QMIN, QMIN + 1, -2, -1, 0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 63, 64,
                 65, QMAX - 1, QMAX, 32768, -32768, 40000, -40000, 65536, -65536,
                 70000, -70000, 2 ** 20, -(2 ** 20)]
        ties = [k + 0.5 for k in (-3, -2, -1, 0, 1, 2, QMAX - 2, QMAX - 1, QMAX,
                                  QMIN - 1, QMIN, QMIN + 1, 1000, -1001)]
    values = np.concatenate([sweep, edges, ties])
    pad = -len(values) % 32
    fill = values[:1] if row in OBSERVED else np.zeros(1)
    return np.concatenate([values, np.repeat(fill, pad)])


class Site16(torch.nn.Module):
    """x * 0 + V -> int16 pair -> activation -> int16 pair; with `observed`,
    espressif::observed_range between the first pair's q and dq (what
    lut16_range's carriers emit after calibration)."""

    def __init__(self, fn, s1, s2, row=None):
        super().__init__()
        self.fn, self.s1, self.s2 = fn, s1, s2
        self.observed = OBSERVED.get(row)
        self.register_buffer("V", torch.from_numpy(
            (input_codes(row) * s1).astype(np.float32)))

    def forward(self, x):
        qd = torch.ops.quantized_decomposed
        v = x * 0.0 + self.V
        c = qd.quantize_per_tensor.default(v, self.s1, 0, QMIN, QMAX, torch.int16)
        if self.observed is not None:
            lo, hi = self.observed
            c = torch.ops.espressif.observed_range(c, lo * self.s1, hi * self.s1)
        d = qd.dequantize_per_tensor.default(c, self.s1, 0, QMIN, QMAX, torch.int16)
        y = self.fn(d)
        c2 = qd.quantize_per_tensor.default(y, self.s2, 0, QMIN, QMAX, torch.int16)
        return qd.dequantize_per_tensor.default(c2, self.s2, 0, QMIN, QMAX, torch.int16)


def ramp(n):
    """main.cpp fill_inputs' float ramp."""
    return (np.arange(n, dtype=np.float32) * np.float32(0.0078125)
            - np.float32(0.5)).astype(np.float32)


@dataclass
class Fixture:
    name: str
    fn: object
    s1: float
    s2: float
    qemu: bool = True
    split_pass: bool = False
    program_of: str = ""
    long_modulus: int = 0
    partitioner: dict = field(default_factory=dict)
    reference: np.ndarray = None
    gap: dict = field(default_factory=dict)
    table: object = None

    def codes(self):
        return input_codes(self.name)

    def input(self):
        """The device's input for this row: main.cpp's ramp."""
        return ramp(len(self.codes()))


# --- the QAT row -----------------------------------------------------------------------

QAT_ROW = "qat_lut16"
QAT_SHAPE = (4, 64)
QAT_STEPS = 40


class SiluMlp(torch.nn.Module):
    """Linear -> SiLU -> Linear: one int16 table site under a16w8."""

    def __init__(self, k=64, hidden=64, n=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(k, hidden)
        self.fc2 = torch.nn.Linear(hidden, n)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


@dataclass
class QatFixture:
    name: str
    qemu: bool = True
    split_pass: bool = False
    program_of: str = ""
    long_modulus: int = 0
    partitioner: dict = field(default_factory=dict)
    reference: np.ndarray = None
    gap: dict = field(default_factory=dict)
    table: object = None
    trained: object = None      # the trained table tensor

    def input(self):
        return ramp(int(np.prod(QAT_SHAPE))).reshape(QAT_SHAPE)


def qat_model():
    """(converted model, its fit report): the row's MLP fine-tuned through
    its lut16 table. Seeded, so the program is the same every run."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

    from executorch.backends.esp32s3.quantizer import lut16_train
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
        Esp32Quantizer,
        fit_tables,
    )

    torch.manual_seed(0)
    teacher = SiluMlp().eval()
    with torch.no_grad():
        for p in teacher.parameters():
            p.mul_(2.0)
    torch.manual_seed(1)
    net = SiluMlp().eval()
    x = torch.from_numpy(QatFixture(QAT_ROW).input())
    gm = torch.export.export(net, (x,), strict=False).module(check_guards=False)
    prepared = prepare_qat_pt2e(gm, Esp32Quantizer(preset="arm_a16w8", is_qat=True))
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for _ in range(8):
            prepared(torch.randn(8, 64, generator=gen))
    report = fit_tables(prepared, learn=True)
    if len(report.tables) != 1 or report.refused:
        raise RuntimeError(f"{QAT_ROW}: expected one table, got {report.summary()}")
    opt = torch.optim.Adam([p for p in prepared.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(QAT_STEPS):
        xb = torch.randn(16, 64, generator=gen)
        with torch.no_grad():
            yb = teacher(xb)
        loss = (prepared(xb) - yb).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    lut16_train.check_tables(prepared)
    return convert_pt2e(prepared), report


def lower_qat(fixture):
    """(ExecutorchProgramManager, the exported trained program, the eager
    trained graph's output) for the QAT row."""
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.quantizer import lut16_train
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.exir import ExecutorchBackendConfig

    converted, report = qat_model()
    (module,) = lut16_train.tables(converted).values()
    fixture.trained = module.table.detach().clone()
    fixture.table = report.tables[0][1].table
    x = torch.from_numpy(fixture.input())
    with torch.no_grad():
        eager = converted(x)
    ep = torch.export.export(converted, (x,), strict=False)
    lowered = _passes.lower_esp32(torch.export.export(converted, (x,), strict=False),
                                    [Esp32Partitioner()])
    _passes.apply_passes(lowered, quiet=True)
    graph = lowered.exported_program().graph_module.graph
    left = sorted({str(n.target) for n in graph.nodes if n.op == "call_function"
                   and "executorch_call_delegate" not in str(n.target)
                   and "getitem" not in str(n.target)})
    if not any("lut16" in t for t in left):
        raise RuntimeError(f"{fixture.name}: no lut16 left: {left}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    if fixture.trained.numpy().tobytes() not in bytes(et.buffer):
        raise RuntimeError(f"{fixture.name}: the trained table's bytes are not in the "
                           f"program")
    return et, ep, eager


def build_qat(fixture):
    """Lower the QAT row; its reference is the trained program's mirror,
    its gap the eager trained graph against it."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as st

    et, ep, eager = lower_qat(fixture)
    x = torch.from_numpy(fixture.input())
    with torch.no_grad():
        out = st.StaticMirror(ep.module(), garbage_collect_values=False).run(x)
    shaped = np.asarray(out.detach().numpy(), np.float32)
    gap = st.eager_gap(ep.module(), (x,), shaped)
    mirror = shaped.reshape(-1)
    eager = eager.numpy().astype(np.float32).reshape(-1)
    fixture.reference = mirror
    fixture.gap = {
        "flips": int(gap.flips), "inputs": int(mirror.size), "max_steps": int(gap.steps),
        "flipped_inputs": [], "bar": 0,
        "bits_equal_eager": bool(np.array_equal(eager.view(np.uint32),
                                                mirror.view(np.uint32))),
        "table": fixture.table.describe() + f" (trained, {fixture.trained.numel()} entries)",
        "table_flips_all_codes": fixture.table.flips,
    }
    return et


def fixtures():
    return ([Fixture(name, fn, s1, s2) for name, fn, s1, s2 in SITES]
            + [QatFixture(QAT_ROW)])


def lower(fixture):
    """(ExecutorchProgramManager, the folded ExportedProgram before lowering,
    the lut16 report) for one row."""
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.exir import ExecutorchBackendConfig

    model = Site16(fixture.fn, fixture.s1, fixture.s2, fixture.name).eval()
    x = torch.from_numpy(ramp(len(fixture.codes())))
    ep = torch.export.export(model, (x,), strict=True)
    eager = model(x)
    lowered = _passes.lower_esp32(ep, [Esp32Partitioner(luts16=True)])
    report = lowered.lut16_report
    if report is None or len(report.tables) != 1 or report.refused:
        raise RuntimeError(f"{fixture.name}: expected one lut16 table, got "
                           f"{report.summary() if report else None}")
    # the fold rewrote the ATen program in place, before to_edge: its module
    # runs lut16's torch reference
    folded = ep
    _passes.apply_passes(lowered, quiet=True)
    graph = lowered.exported_program().graph_module.graph
    left = sorted({str(n.target) for n in graph.nodes if n.op == "call_function"
                   and "executorch_call_delegate" not in str(n.target)
                   and "getitem" not in str(n.target)})
    if any("quantize" in t for t in left) or not any("lut16" in t for t in left):
        raise RuntimeError(f"{fixture.name}: ops left portable: {left}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, folded, report, eager


def build(fixture):
    """Lower one row and compute its reference and eager gap."""
    if isinstance(fixture, QatFixture):
        return build_qat(fixture)
    et, folded, report, eager = lower(fixture)
    _, build16 = report.tables[0]
    t = build16.table
    values = (fixture.codes() * fixture.s1).astype(np.float32)
    mirror = qm.lut16(values, t.data, build16.inv_scale, build16.zero_point,
                      build16.quant_min, build16.quant_max, t.scheme_id, t.lo, t.hi,
                      t.shift, t.G, build16.out_scale, build16.out_zero_point,
                      t.out_qmin, t.out_qmax)
    x = torch.from_numpy(ramp(len(values)))
    with torch.no_grad():
        torch_ref = folded.module()(x).numpy()
    if not np.array_equal(torch_ref.view(np.uint32), mirror.view(np.uint32)):
        bad = np.nonzero(torch_ref.view(np.uint32) != mirror.view(np.uint32))[0]
        raise RuntimeError(f"{fixture.name}: torch reference != mirror at {bad[:8]}")
    eager = eager.numpy()
    codes_dev = np.rint(mirror / np.float32(fixture.s2)).astype(np.int64)
    codes_eager = np.rint(eager / np.float32(fixture.s2)).astype(np.int64)
    flipped = np.nonzero(codes_dev != codes_eager)[0]
    fixture.reference = mirror
    fixture.table = t
    fixture.gap = {
        "flips": int(len(flipped)), "inputs": int(len(values)),
        "max_steps": int(np.abs(codes_dev - codes_eager).max()),
        "flipped_inputs": [float(fixture.codes()[i]) for i in flipped[:16]],
        "table": t.describe(), "bar": t.bar,
        "table_flips_all_codes": t.flips,
    }
    return et


def write_image(path, table, programs):
    data = bytearray(ea.write_image(path, table, programs))
    for i in range(len(table)):
        at = 12 + i * ea.FIXTURE_ENTRY.size + 36
        flags = int.from_bytes(data[at:at + 4], "little") | FIXTURE_ALL_OUTPUTS
        data[at:at + 4] = flags.to_bytes(4, "little")
    if len(data) > PARTITION_BYTES:
        raise RuntimeError(f"{IMAGE}: {len(data):,} B past its {PARTITION_BYTES:,} B "
                           f"partition")
    with open(path, "wb") as f:
        f.write(data)
    return bytes(data)


def emitted_programs():
    """Every row's program (dump_required_ops: the portable kernels the
    parity firmware must register for them)."""
    return [(lower_qat(f) if isinstance(f, QatFixture) else lower(f))[0]
            for f in fixtures()]


def export_all(out_dir=MAIN, verbose=True):
    """Every row's <name>.pte and the QEMU image. Returns {row: reference};
    export_all.gaps holds each row's flips against eager."""
    table = fixtures()
    programs, refs, gaps = {}, {}, {}
    for fixture in table:
        et = build(fixture)
        programs[fixture.name] = bytes(et.buffer)
        refs[fixture.name] = fixture.reference
        gaps[fixture.name] = fixture.gap
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(et.buffer)
        if verbose:
            print(f"  {fixture.name}.pte: {len(et.buffer):,} B; {fixture.gap['table']}; "
                  f"{fixture.gap['flips']} of {fixture.gap['inputs']} inputs flip "
                  f"against eager")
    data = write_image(os.path.join(out_dir, IMAGE), table, programs)
    if verbose:
        print(f"  {IMAGE}: {len(data):,} B of {PARTITION_BYTES:,}")
    export_all.gaps = gaps
    return refs


if __name__ == "__main__":
    export_all()
