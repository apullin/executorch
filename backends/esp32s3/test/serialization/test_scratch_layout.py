# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""serialization/scratch_layout.py against the runtime, on the host runner.

Every scratch path the delegate has, one fixture family at a time: the row
block's quantised rows, conv staging, depthwise taps and grouped slices in
1-D, 2-D and 3-D; the i4 (both cores' unpack rows), i8 and i16 rungs; fp32
convolutions and 2-byte float weights; chains of floats and of Stage B
codes; dynamic q@kT and the int8 K cache (keys read in place or staged);
asymmetric weights (zero points, float offsets), activations and int16
activations; the selected-row head; the tied head split across the cores or
staged to int8 at load; min-latency over the static int8 conv path, which
reserves nothing for a split worker since no convolution splits; and a gather
that needs no scratch.

For each, scratch_layout.scratch_bytes of every delegate blob in the program
is the runtime's own sum: its init report's "B temp". And under plan_scratch
each call declares exactly that plus the alignment slack, no blob changes, the
output bits are those of the same program without it, and every call that
needs scratch runs on its planned buffer (the backend report's counts), also
when the planned buffers start off a 16-byte boundary (--planned-offset 4, as
a heap_caps_malloc'd arena does); an under-declared buffer falls back to the
temp allocator, logged once and counted, never overrun.

    python -m pytest test_scratch_layout.py -q

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import functools
import importlib
import io
import re
import subprocess
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from executorch.exir import ExecutorchBackendConfig

from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.serialization import scratch_layout
from executorch.backends.esp32s3.test import models
from executorch.backends.esp32s3.test.host import host_runner

INIT = re.compile(r"init: (\d+) op\(s\), (\d+) B blob, (\d+) B temp")


@pytest.fixture(scope="module")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@dataclass
class Case:
    """One program: build() lowers it (ExecutorchProgramManager, inputs);
    flags go to the host runner."""

    name: str
    build: object
    flags: tuple = ()


def symmetric(factory, shape, **options):
    """A models.py fixture through Esp32Partitioner(**options)."""

    def build():
        torch.manual_seed(0)
        model = factory().eval()
        assert sum(p.numel() for p in model.parameters()) <= 200_000
        x = models.ramp_input(shape)
        ep = torch.export.export(model, (x,))
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(ep, [Esp32Partitioner(**options)])
            apply_passes(edge, quiet=True)
        return edge.to_executorch(ExecutorchBackendConfig(delegate_alignment=16)), (x,)

    return build


@functools.lru_cache(maxsize=None)
def _fixture_table(module):
    mod = importlib.import_module(f"executorch.backends.esp32s3.test.fixtures.{module}")
    with contextlib.redirect_stdout(io.StringIO()):
        return mod, {f.name: f for f in mod.fixtures()}


def fixture(module, name):
    """A row of a fixtures module, through the module's own lower()."""

    def build():
        mod, table = _fixture_table(module)
        fx = table[name]
        with contextlib.redirect_stdout(io.StringIO()):
            program, _ = mod.lower(fx)
        return program, fx.inputs

    return build


@functools.lru_cache(maxsize=None)
def _quantized(name, stage_b):
    from executorch.backends.esp32s3.test.fixtures import asym_stage_b as sb
    from executorch.backends.esp32s3.test.fixtures import asym_static as st

    if stage_b:
        row = next(r for r in sb.SPECS if r.name == name)
        row = st.Row(row.name, row.recipe, row.build, row.shape, luts=row.luts)
    else:
        row = next(r for r in st.SPECS if r.name == name)
    with contextlib.redirect_stdout(io.StringIO()):
        model, inputs, _ = st.quantized(row)
    return row, model, inputs


def static(name):
    """An asym_static row (a stock recipe's static int8 graph), lowered as
    the module lowers it, without its mirror."""

    def build():
        from executorch.backends.esp32s3.test.fixtures import asym_static as st

        row, model, inputs = _quantized(name, False)
        fx = st.Fixture(row.name, model, inputs, None, {}, luts=row.luts,
                        keeps=row.portable, dynamic="dynamic" in row.recipe)
        with contextlib.redirect_stdout(io.StringIO()):
            program, _ = st.lower(fx)
        return program, inputs

    return build


@functools.lru_cache(maxsize=None)
def _a16_quantized(name):
    from executorch.backends.esp32s3.test.fixtures import asym_a16

    row = next(r for r in asym_a16.SPECS if r.name == name)
    with contextlib.redirect_stdout(io.StringIO()):
        model, inputs = asym_a16.quantized(row)
    return row, model, inputs


def a16(name):
    """An asym_a16 row (int16 activations), lowered as the module lowers it,
    without its mirror."""

    def build():
        from executorch.backends.esp32s3.test.fixtures import asym_a16
        from executorch.backends.esp32s3.test.fixtures import asym_static as st

        row, model, inputs = _a16_quantized(name)
        here = {"format_policy": "i8"} if name in asym_a16.WEIGHTS_HERE else {}
        fx = st.Fixture(row.name, model, inputs, None, here, luts=row.luts,
                        keeps=row.portable)
        with contextlib.redirect_stdout(io.StringIO()):
            program, _ = st.lower(fx)
        return program, inputs

    return build


def stage_b(name):
    """An asym_stage_b row's Stage B variant: chains of int8 codes."""

    def build():
        from executorch.backends.esp32s3.test.fixtures import asym_stage_b as sb

        row, model, inputs = _quantized(name, True)
        with contextlib.redirect_stdout(io.StringIO()):
            program, _, _ = sb.lower_program(model, inputs, row.luts, stage_b=True)
        return program, inputs

    return build


def conv_split(name, stage_b, min_latency=False):
    """test_conv_fast's model at a size where the split, were it on
    (esp32_backend.cpp kConvSplit), would split every conv on the static int8
    path under min-latency. No convolution splits, so a delegate that can split
    from init (min_latency in its compile spec, or --min-latency, the option at
    init) reserves no scratch for a worker's part: the single-core layout."""

    def build():
        from executorch.backends.esp32s3.test import test_conv_fast as cf

        model, x = cf.quantised(name, "arm-a8w8", cf.SPLIT_SHAPES[name])
        ep = torch.export.export(model, (x,), strict=False)
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(ep, [Esp32Partitioner(stage_b=stage_b,
                                                     min_latency=min_latency)])
            apply_passes(edge, quiet=True)
        return edge.to_executorch(ExecutorchBackendConfig(delegate_alignment=16)), (x,)

    return build


CASES = [
    # The row block, chains of floats, the three rungs.
    Case("mlp i8", symmetric(models.MLP, (4, 96))),
    Case("mlp i4", symmetric(models.MLP, (4, 96), format_policy="i4")),
    Case("mlp i16", symmetric(models.MLP, (4, 96), format_policy="i16")),
    Case("mlp i8 one row", symmetric(models.MLP, (1, 96))),
    Case("block", symmetric(models.Block, (1, 8, 96))),
    Case("block i4 min-latency", symmetric(models.Block, (1, 8, 96), format_policy="i4",
                                           min_latency=True), ("--min-latency",)),
    # Convolutions: staging, row scales, depthwise taps, grouped slices.
    Case("conv", symmetric(models.ConvD, (1, 4, 9, 9))),
    Case("conv i4", symmetric(models.ConvD, (1, 4, 9, 9), format_policy="i4")),
    Case("conv residual", symmetric(models.ConvResD, (1, 8, 9, 9))),
    Case("conv1d", symmetric(models.Conv1dD, (1, 4, 20))),
    Case("conv3d", symmetric(models.Conv3dD, (1, 3, 4, 8, 8))),
    Case("depthwise", symmetric(models.DwConvD, (1, 8, 7, 7))),
    Case("depthwise 1d", symmetric(models.DwConv1dD, (1, 8, 24))),
    Case("grouped", symmetric(models.GroupConvD, (1, 8, 7, 7))),
    Case("grouped 8", symmetric(models.GroupConv8D, (1, 16, 9, 9))),
    Case("convnet", symmetric(models.ConvNet, (1, 1, 16, 16))),
    # Float weights: the fp32 convolution's patch, 2-byte rows widened.
    Case("conv fp32", symmetric(models.ConvD, (1, 4, 9, 9), format_policy="fp32")),
    Case("mlp fp16", symmetric(models.MLP, (4, 96), format_policy="fp16")),
    Case("conv bf16", symmetric(models.ConvD, (1, 4, 9, 9), format_policy="bf16")),
    # q@kT: dynamic keys; the int8 K cache read in place (d=32) or staged (d=8).
    Case("q@kT", symmetric(models.AttnQK, (2, 8, 32), quantize_qk=True)),
    Case("K cache", symmetric(models.AttnKC8, (2, 8, 32), quantize_qk=True)),
    Case("K cache staged", symmetric(lambda: models.AttnKC8(d=8), (2, 8, 8),
                                     quantize_qk=True)),
    # Asymmetric weights: zero points in row blocks and rows, fused int4,
    # float offsets, conv and depthwise, a gather alone (no scratch), the tied
    # head split across the cores and staged to int8, the selected-row head.
    Case("asym i8 m4", fixture("asym", "asym_i8_m4")),
    Case("asym i4 m1", fixture("asym", "asym_i4_m1")),
    Case("asym i4 m1 min-latency", fixture("asym", "asym_i4_m1"), ("--min-latency",)),
    Case("asym i4 m8", fixture("asym", "asym_i4_m8")),
    Case("asym float offsets", fixture("asym", "asym_woff")),
    Case("asym conv", fixture("asym", "asym_conv")),
    Case("asym embedding", fixture("asym", "asym_embed")),
    Case("asym head split", fixture("asym", "asym_head")),
    Case("asym head staged", fixture("asym", "asym_head_staged")),
    Case("asym selected head", fixture("asym", "asym_select")),
    # Asymmetric activations.
    Case("act a2x", fixture("asym_act", "asym_a2x_i8")),
    Case("act a3 i4 m8", fixture("asym_act", "asym_a3_i4_m8")),
    Case("act shared", fixture("asym_act", "asym_actshare")),
    Case("act conv", fixture("asym_act", "asym_a2_conv")),
    Case("act head", fixture("asym_act", "asym_a3_head")),
    # 16-bit activations: the int16 loop's row of codes.
    Case("a16 mlp", a16("a16_arm_mlp")),
    Case("a16 conv", a16("a16_arm_conv")),
    Case("a16 depthwise", a16("a16_arm_dwconv")),
    Case("a16 cortex-m linear", a16("a16_cm_linear")),
    # Stock recipes' static graphs, channels-last included.
    Case("static mlp", static("st_xnn_mlp")),
    Case("static conv", static("st_xnn_conv")),
    Case("static cl conv", static("st_cm_cl_conv")),
    Case("static float conv", static("st_xnn_conv_f32")),
    Case("static half", static("st_xnn_half")),
    # Stage B: chains of int8 codes.
    Case("stage b mbnet", stage_b("mbnet")),
    Case("stage b fanout", stage_b("fanout")),
    # The static int8 conv path under min-latency, where no conv splits: no
    # region for a worker's part, depthwise (fp32 between the ops) or dense
    # (codes between them, an fp32 input plane).
    Case("conv split depthwise", conv_split("depthwise_chain", False, True),
         ("--min-latency",)),
    Case("conv split dense", conv_split("dense", True, True), ("--min-latency",)),
    Case("conv split pie depthwise", conv_split("pie_depthwise", True, True),
         ("--min-latency",)),
    # Its row-dilated and wide PIE depthwise convs: _pie_dw_need's strip raised
    # to the 5 and 3 rows an output row reads, 202 x 16 bytes each.
    Case("conv split pie dilated wide", conv_split("pie_dilated_wide", True, True),
         ("--min-latency",)),
    # The same programs on single-core handles: the same layouts.
    Case("conv single-core depthwise", conv_split("depthwise_chain", False)),
    Case("conv single-core dense", conv_split("dense", True)),
    Case("conv single-core pie depthwise", conv_split("pie_depthwise", True)),
    Case("conv single-core pie dilated wide", conv_split("pie_dilated_wide", True)),
]


def delegate_blobs(program):
    """Each delegate's blob, in the method's delegate order (init's)."""
    plan = program.executorch_program.execution_plan[0]
    data = program.executorch_program.backend_delegate_data
    return [bytes(data[d.processed.index].data) for d in plan.delegates]


def delegate_min_latency(program):
    """Each delegate's compile-spec min_latency, in delegate order."""
    from executorch.backends.esp32s3.compile_spec import SPEC_MIN_LATENCY

    plan = program.executorch_program.execution_plan[0]
    return [any(spec.key == SPEC_MIN_LATENCY and bytes(spec.value) != b"\0"
                for spec in d.compile_specs) for d in plan.delegates]


def can_split(program, flags):
    """Per delegate, whether init can split it across the cores: min_latency
    in its compile spec, or the option set before init (--min-latency)."""
    return [m or "--min-latency" in flags for m in delegate_min_latency(program)]


def run(runner, program, inputs, work, flags=()):
    """(every output's bytes, the runner's log) of one forward."""
    work.mkdir(parents=True, exist_ok=True)
    pte = work / "model.pte"
    pte.write_bytes(program.buffer)
    program.write_tensor_data_to_file(str(work))
    plan = program.executorch_program.execution_plan[0]
    command = [str(runner), "--pte", str(pte), "--method", "forward", *flags]
    ptds = sorted(work.glob("*.ptd"))
    assert len(ptds) <= 1, ptds
    for ptd in ptds:
        command += ["--ptd", str(ptd)]
    flat = torch.utils._pytree.tree_flatten(inputs)[0]
    for i, (x, index) in enumerate(zip(flat, plan.inputs)):
        x = x.detach()
        order = list(plan.values[index].val.dim_order)
        if order and order != sorted(order):
            x = x.permute(*order)
        path = work / f"in{i}.bin"
        np.ascontiguousarray(x.contiguous().numpy()).tofile(path)
        command += ["--input", str(path)]
    outs = [work / f"out{k}.bin" for k in range(len(plan.outputs))]
    for out in outs:
        out.unlink(missing_ok=True)
        command += ["--output", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    log = done.stdout + done.stderr
    assert done.returncode == 0, log[-3000:]
    return [out.read_bytes() for out in outs], log


def init_reports(log):
    return [(int(n), int(size), int(temp)) for n, size, temp in INIT.findall(log)]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_layout_matches_the_runtime(runner, tmp_path, case):
    """Each delegate's scratch_layout.scratch_bytes is its init report's temp
    (the reports name each blob's size, which pins one to the other)."""
    program, inputs = case.build()
    blobs = delegate_blobs(program)
    assert blobs, f"{case.name}: nothing delegated"
    _, log = run(runner, program, inputs, tmp_path, case.flags)
    reports = init_reports(log)
    assert [size for _, size, _ in reports] == [len(b) for b in blobs], log[-2000:]
    split = can_split(program, case.flags)
    got = [scratch_layout.scratch_bytes(b, m) for b, m in zip(blobs, split)]
    assert got == [temp for _, _, temp in reports], (
        case.name, [scratch_layout.layout(b, m).pieces for b, m in zip(blobs, split)])


def test_min_latency_reserves_no_conv_scratch(runner, tmp_path):
    """No convolution splits across the cores (esp32_backend.cpp kConvSplit,
    mirrored as scratch_layout.CONV_SPLIT), so a delegate that can split from
    init reserves what a single-core one does. The runtime's own init
    reports, for the conv path's programs at the sizes where the split would
    have split every conv: compiled for min_latency and run with
    --min-latency, each delegate's temp is the same program's compiled and
    run for one core, and that is the mirror's single-core layout. (A
    runtime with kConvSplit on reserves a worker's part and fails here; the
    mirror compared with itself, as before, could not.)"""
    assert scratch_layout.CONV_SPLIT is False
    for name, stage_b in (("depthwise_chain", False), ("dense", True),
                          ("pie_depthwise", True), ("pie_dilated_wide", True)):
        split, inputs = conv_split(name, stage_b, True)()
        single, _ = conv_split(name, stage_b)()
        _, log_split = run(runner, split, inputs, tmp_path / name / "split",
                           ("--min-latency",))
        _, log_single = run(runner, single, inputs, tmp_path / name / "single")
        temps_split = [t for _, _, t in init_reports(log_split)]
        temps_single = [t for _, _, t in init_reports(log_single)]
        blobs = delegate_blobs(single)
        assert blobs and len(temps_split) == len(blobs), (name, log_split[-800:])
        assert temps_split == temps_single, (name, temps_split, temps_single)
        assert temps_single == [scratch_layout.scratch_bytes(b, False)
                                for b in blobs], (name, temps_single)


def test_layout_pieces():
    """The bump allocator's rule (ScratchLayout::reserve): nothing for zero
    bytes, every piece at the next 16-byte boundary, the total its end."""
    layout = scratch_layout.ScratchLayout()
    assert layout.reserve("a", 0) is None
    assert layout.reserve("a", 5) == 0
    assert layout.reserve("b", 16) == 16
    assert layout.reserve("c", 1) == 32
    assert layout.bytes == 33
    assert [p[0] for p in layout.pieces] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="32-bit"):
        layout.reserve("d", 0xFFFFFFFF)


# ------------------------------------------------------------ plan_scratch

COUNTS = re.compile(r"esp32 scratch: (\d+) call\(s\) on planned scratch, (\d+) on the temp "
                    r"allocator \(none planned\), (\d+) fallback")


def scratch_counts(log):
    """(planned, temp, fallback) calls from the runner's backend report."""
    found = COUNTS.search(log)
    assert found, log[-2000:]
    return tuple(int(g) for g in found.groups())


def delegate_calls(program):
    plan = program.executorch_program.execution_plan[0]
    return [i.instr_args for chain in plan.chains for i in chain.instructions
            if type(i.instr_args).__name__ == "DelegateCall"]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_planned_scratch(runner, tmp_path, monkeypatch, case):
    """plan_scratch declares each call's scratch_bytes plus SCRATCH_SLACK
    (nothing where it is 0), changes no blob, gives the same output bits as
    the program without it, and every call that needs scratch runs on its
    planned buffer, whether the arena's base is aligned or 4 bytes past a
    boundary: none takes the temp allocator, none falls back. Without it
    every such call takes the temp allocator, as before."""
    from executorch.backends.esp32s3.partition import espressif_partitioner
    from executorch.backends.esp32s3.serialization.blob_writer import SCRATCH_SLACK

    monkeypatch.setattr(espressif_partitioner, "PLAN_SCRATCH_DEFAULT", False)
    off, inputs = case.build()
    blobs = delegate_blobs(off)
    # What preprocess declares: the compile spec's min_latency (the cases
    # that pass --min-latency for convs say so in it too).
    need = [scratch_layout.scratch_bytes(b, m)
            for b, m in zip(blobs, delegate_min_latency(off))]
    assert need == [scratch_layout.scratch_bytes(b, m)
                    for b, m in zip(blobs, can_split(off, case.flags))]
    needing = sum(1 for n in need if n)
    assert not any(call.scratch for call in delegate_calls(off))
    out_off, log_off = run(runner, off, inputs, tmp_path / "off", case.flags)
    assert scratch_counts(log_off) == (0, needing, 0)

    monkeypatch.setattr(espressif_partitioner, "PLAN_SCRATCH_DEFAULT", True)
    on, _ = case.build()
    assert delegate_blobs(on) == blobs
    calls = delegate_calls(on)
    declared = [[s.size for s in call.scratch or []] for call in calls]
    assert declared == [[need[c.delegate_index] + SCRATCH_SLACK] if need[c.delegate_index]
                        else [] for c in calls]
    for offset in (0, 4):
        out_on, log_on = run(runner, on, inputs, tmp_path / f"on{offset}",
                             case.flags + ("--planned-offset", str(offset)))
        assert out_on == out_off, (case.name, offset)
        assert scratch_counts(log_on) == (needing, 0, 0), (offset, log_on[-2000:])


def test_under_declared_scratch_falls_back(runner, tmp_path, monkeypatch):
    """A planned buffer smaller than the runtime's layout (a drift between
    scratch_layout.py and esp32_backend.cpp) is never overrun: the call takes
    the temp allocator, the report counts it and the log says so once, and
    the outputs are unchanged."""
    from executorch.backends.esp32s3.partition import espressif_partitioner

    case = next(c for c in CASES if c.name == "block")
    monkeypatch.setattr(espressif_partitioner, "PLAN_SCRATCH_DEFAULT", False)
    off, inputs = case.build()
    need = [scratch_layout.scratch_bytes(b) for b in delegate_blobs(off)]
    needing = sum(1 for n in need if n)
    assert needing >= 2
    out_off, _ = run(runner, off, inputs, tmp_path / "off")

    exact = scratch_layout.scratch_bytes
    monkeypatch.setattr(scratch_layout, "scratch_bytes",
                        lambda blob, *args: max(exact(blob, *args) - 16, 0))
    monkeypatch.setattr(espressif_partitioner, "PLAN_SCRATCH_DEFAULT", True)
    short, _ = case.build()
    out_short, log = run(runner, short, inputs, tmp_path / "short")
    assert out_short == out_off
    planned = sum(1 for n in need if n > 16)
    assert scratch_counts(log) == (0, needing - planned, planned), log[-2000:]
    assert log.count("esp32: planned scratch unusable") == 1, log[-2000:]


def test_plan_scratch_switch(monkeypatch):
    """plan_scratch is on by default; off, it is absent from the compile
    specs (the program is as it was without it); on, it is one spec that
    round-trips. ESP32_PLAN_SCRATCH=0 or 1 sets the default for a whole run
    (read once, at import, by every process); any other value is refused."""
    from executorch.backends.esp32s3.compile_spec import (
        Esp32CompileSpec,
        SPEC_PLAN_SCRATCH,
    )
    from executorch.backends.esp32s3.partition import espressif_partitioner

    def keys(specs):
        return [spec.key for spec in specs]

    off = Esp32Partitioner(plan_scratch=False)
    assert SPEC_PLAN_SCRATCH not in keys(off.delegation_spec.compile_specs)
    on = Esp32Partitioner(plan_scratch=True).compile_spec
    assert SPEC_PLAN_SCRATCH in keys(on.to_list())
    assert Esp32CompileSpec.from_list(on.to_list()) == on and on.plan_scratch
    for value, want in (("", True), ("0", False), ("1", True)):
        monkeypatch.setenv("ESP32_PLAN_SCRATCH", value)
        assert espressif_partitioner._plan_scratch_default() is want
    monkeypatch.setenv("ESP32_PLAN_SCRATCH", "yes")
    with pytest.raises(ValueError, match="ESP32_PLAN_SCRATCH"):
        espressif_partitioner._plan_scratch_default()
    monkeypatch.setattr(espressif_partitioner, "PLAN_SCRATCH_DEFAULT", True)
    assert Esp32Partitioner().compile_spec.plan_scratch
    assert not Esp32Partitioner(plan_scratch=False).compile_spec.plan_scratch
