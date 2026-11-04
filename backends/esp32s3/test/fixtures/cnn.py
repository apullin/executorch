# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CNN and RNN parity rows: whole models through a stock static recipe and
the backend's defaults, held to the host runner's output bits.

Each row is a census model (quant_census.models, its own seeded weights and
BatchNorm statistics), quantised by Arm's a8w8 recipe as the census quantises
it (quant_census.recipes) and lowered with Esp32Partitioner at its defaults
(Stage B, the code tables, the fused norms), its weights tagged external as
the census tags them, and memory_plan="esp32":

  cnn_arm_ds_cnn    MLPerf Tiny keyword spotting DS-CNN, [1, 1, 49, 10]
  cnn_arm_resnet8   MLPerf Tiny CIFAR-10 ResNet-8, [1, 3, 32, 32]: on the
                    board only (below)
  rnn_arm_lstm      the census's nn.LSTM(32, 64) over 8 steps, its last step
                    through Linear(64, 12)
  rnn_arm_gru       the same with nn.GRU
  cnn_arm_dw5_plane 5x5 depthwise convs on the plane path (dw5_plane1_s3,
                    dw5_plane_s3: 16 channels at stride 1 on 37-pixel rows,
                    two kernel calls; 24 at stride 2), an 8-channel 3x3
                    (dw3_plane1_s3) and an unpadded 5x5 (the scalar loop),
                    [1, 3, 15, 37]
  cnn_arm_dw5_pairs 5x5 depthwise convs in the plane path's pairs form
                    (dw5_pairs_s3: 7 x 8 padded, row dilation 2, unpadded),
                    [1, 3, 7, 8]

The two dw5 rows are this file's own small models (LOCAL_MODELS, not the
census's): every depthwise conv in the census's models is a 3x3, so without
them no parity row runs the 5x5 kernels' assembly end to end.

Every convolution and matmul runs in a delegate, and the export asserts what
each row leaves between its delegates (Row.portable): the pads, the global
average pool and the flatten, ResNet-8's residual adds and ReLUs, the
recurrent gates' tables, products and sums.

cnn_arm_ds_cnn_split runs DS-CNN's program again in main.cpp's min_latency
pass (asym.FIXTURE_SPLIT_PASS), where none of its nine convolutions splits
across the cores (no convolution does: on the board a split conv gave wrong
outputs and cache-error panics, the backend README's Known limits), and holds
it to the plain row's bits. The export runs it on the host runner under
--min-latency too (the same bits, convsplit 0) and keeps its temp, the plain
row's (no split worker's scratch), which must fit the parity firmware's temp
arena (PARITY_TEMP) as every QEMU row's must.

The device feeds main.cpp's ramp (asym.ramp). The recipe calibrates on the
ramp and on the census's own eight batches, so the ramp sits inside every
calibrated grid.

The reference is the host runner (test/host_runner), which runs each
emitted program with the delegate's and the portable kernels' own C++ on
the build machine, so no op between the delegates needs a Python model (a
float reduction is where one fails: asym_static's code rows). For these
programs that is the device's arithmetic bit for bit: the delegates' static
int8 loops and the table op build with -ffp-contract=off on both, and what
the recipe leaves between the delegates -- quantise, dequantise, pads, a
global average pool, data moves, fills, adds, subtractions, ReLUs and
products -- holds no multiply-add for the chip's compiler to fuse. Against
the recipe's eager model each row reports its cosine and its largest
difference in steps of the output's grid, and must pass the census's bar
(quant_census.MIN_COSINE).

Memory. Every row's arena is planned with memory_plan="esp32" whatever
run_parity's --memory-plan says (the census measures them so). A row runs
under QEMU only if its arena and largest delegate temp together fit
QEMU_BUDGET; the export asserts that each row's board_only says so.
ResNet-8 does not: since ExecuTorch 1.6 (#22707) its stride-2 stages pad
explicitly, ZeroPad2d((0, 1, 0, 1)), which no convolution can absorb, and
the second stage's delegate then reads the pad's fp32 output and the ReLU's
(its skip convolution) beside its two outputs: a 200,768 B live bound,
where 1.5.1's graph needed 143,360 B. Its image entry carries FIXTURE_PSRAM:
main.cpp skips it where the firmware has no PSRAM, and run_parity holds it
only on silicon. Every entry asks main.cpp to print the largest free blocks
before the planned buffers are allocated (FIXTURE_HEAP).

Weights. An unrolled recurrent layer reads one weight at every step, and
each step is its own delegate (the gates run between them): inline, each
blob would carry its own copy of the recurrent weight (the LSTM's program
would be 327 KB). Tagged external, their content-addressed keys store it
once. One partition holds the named data of every row (merge_named_data),
and every entry loads it (asym.FIXTURE_PTD).

export_all(out_dir) writes every program, the packed image main.cpp reads
from the cnnrnn flash partition (cnn_fixtures.bin, asym.py's layout) and the
named data it reads from cnnrnnw (cnn_weights.ptd); it returns {row:
reference}. Imports torch.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from executorch.backends.esp32s3.test import firmware
from executorch.backends.esp32s3.test.fixtures import asym as ea

MAIN = str(firmware.MAIN)
F32 = np.float32
IMAGE = "cnn_fixtures.bin"
PTD_TAG = "cnn_weights"
WEIGHTS = f"{PTD_TAG}.ptd"
RECIPE = "arm-a8w8"
# Image entry flags beyond asym.py's: main.cpp prints the largest free blocks
# before a row's planned buffers are allocated (32), and skips a row where
# the firmware has no PSRAM (64).
FIXTURE_HEAP = 32
FIXTURE_PSRAM = 64
# What a row's arena and largest delegate temp may total and still run in the
# parity firmware under QEMU, which has no PSRAM: its largest internal block
# at boot is 172,032 B, before the rows before it fragment the heap.
QEMU_BUDGET = 165_000
# main.cpp's temp_arena: all the temporary memory run_model gives a program
# (no PSRAM tier), so the largest delegate temp of a row it runs, in either
# pass.
PARITY_TEMP = 32 * 1024

QDQ = ("quantized_decomposed.quantize_per_tensor",
       "quantized_decomposed.dequantize_per_tensor")
CNN = ("constant_pad_nd", "avg_pool2d", "view_copy")
GATES = ("espressif.quant_lut", "slice_copy", "select_copy", "unsqueeze_copy",
         "mul.Tensor", "add.Tensor", "full", "cat")


@dataclass(frozen=True)
class Row:
    name: str
    model: str           # the quant_census model
    portable: tuple      # every op the row leaves outside its delegates
    board_only: bool = False
    split: bool = False  # its program again in the min_latency pass (<name>_split)


SPECS = (
    Row("cnn_arm_ds_cnn", "ds_cnn", CNN, split=True),
    Row("cnn_arm_resnet8", "resnet8", CNN + ("add.Tensor", "relu"),
        board_only=True),
    Row("rnn_arm_lstm", "lstm", GATES),
    Row("rnn_arm_gru", "gru", GATES + ("sub.Tensor",)),
    Row("cnn_arm_dw5_plane", "dw5_plane", ()),
    Row("cnn_arm_dw5_pairs", "dw5_pairs", ()),
)


def _conv(cin, cout, k, stride=1, padding=0, dilation=1, groups=1, act=None,
          bias=True):
    layers = [torch.nn.Conv2d(cin, cout, k, stride, padding, dilation, groups,
                              bias=bias)]
    return layers + ([act()] if act is not None else [])


def _dw5_plane():
    nn = torch.nn
    return nn.Sequential(
        *_conv(3, 16, 1, act=nn.ReLU),
        *_conv(16, 16, 5, 1, 2, groups=16, act=nn.ReLU),
        *_conv(16, 24, 1, act=nn.ReLU),
        *_conv(24, 24, 5, 2, 2, groups=24, act=nn.ReLU6),
        *_conv(24, 8, 1, act=nn.ReLU),
        *_conv(8, 8, 3, 1, 1, groups=8, act=nn.ReLU),
        *_conv(8, 8, 5, 1, 0, groups=8, bias=False),
        *_conv(8, 8, 1),
    )


def _dw5_pairs():
    nn = torch.nn
    return nn.Sequential(
        *_conv(3, 16, 1, act=nn.ReLU),
        *_conv(16, 16, 5, 1, 2, groups=16, act=nn.ReLU),
        *_conv(16, 8, 1, act=nn.ReLU),
        *_conv(8, 8, 5, 1, (4, 2), (2, 1), groups=8, act=nn.ReLU6),
        *_conv(8, 8, 5, 1, 0, groups=8, bias=False),
        *_conv(8, 8, 1),
    )


# This file's own models (test_conv_fast's PlaneDepthwise and PairsDepthwise
# shapes): {name: (builder, input shape)}. Seeded weights; calibrated on the
# ramp and four seeded batches.
LOCAL_MODELS = {
    "dw5_plane": (_dw5_plane, (1, 3, 15, 37)),
    "dw5_pairs": (_dw5_pairs, (1, 3, 7, 8)),
}
ROWS = (tuple(r.name for r in SPECS)
        + tuple(f"{r.name}_split" for r in SPECS if r.split))
QEMU_ROWS = (tuple(r.name for r in SPECS if not r.board_only)
             + tuple(f"{r.name}_split" for r in SPECS
                     if r.split and not r.board_only))


@contextlib.contextmanager
def one_thread():
    """Calibrate on one thread: torch's threaded reductions vary from run to
    run, and the exported grids (so the programs) must not."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def quantized(row):
    """(the recipe's converted module, the device's inputs, the recipe's
    info): the census model at seed 0, calibrated on the ramp and the
    census's batches."""
    from executorch.backends.esp32s3.test.quant_census import models, recipes

    if row.model in LOCAL_MODELS:
        build, shape = LOCAL_MODELS[row.model]
        torch.manual_seed(0)
        net = build().eval()
        calibration = [(torch.rand(shape) * 2 - 1,) for _ in range(4)]
        inputs = (torch.zeros(shape),)
    else:
        net, inputs, calibration = models.model(row.model, 0)
    x = ea.ramp(tuple(inputs[0].shape))
    with one_thread():
        q = recipes.quantize(RECIPE, net, (x,), [(x,)] + list(calibration))
    return q.program, q.inputs, q.info


def lower(model, inputs, min_latency=False):
    """lower_esp32 with Esp32Partitioner(external_weights=PTD_TAG), and
    min_latency when asked (each delegate then splits its projections across
    the cores from init; no convolution splits), apply_passes,
    and to_executorch planned with memory_plan="esp32".
    Returns (the ExecutorchProgramManager, its named data,
    quant_census.lower.graph_report of the lowered graph, the arena)."""
    import executorch.kernels.quantized  # noqa: F401  (the q/dq .out variants)
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.memory_planning import planning_pass
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.quant_census.lower import graph_report

    ep = torch.export.export(model, inputs, strict=False)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        edge = _passes.lower_esp32(ep, [Esp32Partitioner(
            external_weights=PTD_TAG, min_latency=min_latency)])
        _passes.apply_passes(edge, quiet=True)
    report = graph_report(edge.exported_program().graph_module)
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    # The tagged named data, as write_tensor_data_to_file would write it.
    named = next((bytes(cord) for name, cord in (et._tensor_data or {}).items()
                  if name in (PTD_TAG, WEIGHTS)), b"")
    plan = et.executorch_program.execution_plan[0]
    return et, named, report, sum(plan.non_const_buffer_sizes[1:])


def merge_named_data(blobs):
    """One FlatTensor file holding the named data of every blob. The
    backend's keys hash their payload, so a key two programs share is the
    same bytes (checked); every entry stays 16-byte aligned, as the delegate's
    vector loads need."""
    from executorch.exir._serialize._cord import Cord
    from executorch.exir._serialize.data_serializer import DataEntry, DataPayload
    from executorch.extension.flat_tensor.serialize.serialize import (
        FlatTensorSerializer,
    )

    serializer = FlatTensorSerializer()
    buffers, entries = [], {}
    for blob in blobs:
        payload = serializer.deserialize(Cord(blob))
        for key, entry in payload.named_data.items():
            data = bytes(payload.buffers[entry.buffer_index])
            if key in entries:
                if bytes(buffers[entries[key].buffer_index]) != data:
                    raise RuntimeError(f"named data {key!r} differs between programs")
                continue
            buffers.append(data)
            entries[key] = DataEntry(buffer_index=len(buffers) - 1, alignment=16,
                                     tensor_layout=entry.tensor_layout)
    return bytes(serializer.serialize(DataPayload(buffers=buffers, named_data=entries)))


# --- the host runner: the reference ----------------------------------------------------

def host_runner():
    """This tree's host runner, brought up to date first.

    test.host.host_runner() names it ($ESP32_HOST_RUNNER, else the tree's
    cmake-out-espressif). Its CMake build directory must build this tree's
    sources (CMAKE_HOME_DIRECTORY and ET_ROOT); `cmake --build` then
    rebuilds whatever changed since it was last built, so every reference
    comes from this tree's runtime. With no build directory there, one is
    configured as test/host_runner/README.md does."""
    from executorch.backends.esp32s3.test.host import ET_ROOT
    from executorch.backends.esp32s3.test.host import host_runner as runner_path

    runner = runner_path()
    build = runner.parent
    project = ET_ROOT / "backends" / "esp32s3" / "test" / "host_runner"
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        text = cache.read_text(errors="replace")
        for key, want in (("CMAKE_HOME_DIRECTORY", project), ("ET_ROOT", ET_ROOT)):
            found = re.search(rf"^{key}:\w+=(.*)$", text, re.M)
            if found and Path(found.group(1)).resolve() != want.resolve():
                raise SystemExit(f"{runner}: its build directory builds "
                                 f"{key}={found.group(1)}, not this tree's {want}; "
                                 "point ESP32_HOST_RUNNER at this tree's build")
    elif runner.is_file():
        raise SystemExit(f"{runner} has no CMake build directory beside it, so "
                         "it cannot be brought up to date; build this tree's "
                         "runner (test/host_runner/README.md)")
    else:
        flatbuffers = os.environ.get("ESPRESSIF_FLATBUFFERS_INCLUDE")
        command = ["cmake", "-S", str(project), "-B", str(build),
                   "-DCMAKE_BUILD_TYPE=Release", f"-DPYTHON_EXECUTABLE={sys.executable}"]
        if flatbuffers:
            command.append(f"-DESPRESSIF_FLATBUFFERS_INCLUDE={flatbuffers}")
        env = {k: v for k, v in os.environ.items() if k != "ET_ROOT"}
        print(f"  host runner: configuring {build}", flush=True)
        done = subprocess.run(command, capture_output=True, text=True, env=env)
        if done.returncode != 0:
            raise SystemExit(f"host runner configure FAILED (in a worktree without "
                             f"submodules, set ESPRESSIF_FLATBUFFERS_INCLUDE to a "
                             f"flatbuffers include directory):\n{done.stdout[-2000:]}"
                             f"{done.stderr[-2000:]}")
    done = subprocess.run(["cmake", "--build", str(build), "-j4", "--target",
                           "esp32_host_runner"], capture_output=True, text=True)
    if done.returncode != 0 or not runner.is_file():
        raise SystemExit(f"host runner build FAILED:\n{done.stdout[-2000:]}"
                         f"{done.stderr[-2000:]}")
    return runner


def run_host(runner, program, named, x, flags=(), splits=False,
             scratch=False):
    """(the program's float32 output on input x, flat; the largest delegate
    temp its init lines report), its weights from the named data `named`,
    the runner given `flags`; with `splits`, also the convs it split across
    the cores (its "esp32 convsplit" count); with `scratch`, also its calls
    on planned scratch, on the temp allocator with none planned, and those
    whose planned scratch was unusable (its "esp32 scratch:" line)."""
    with tempfile.TemporaryDirectory() as tmp:
        pte, ptd, inp, out = (Path(tmp) / name for name in
                              ("row.pte", WEIGHTS, "in.bin", "out.bin"))
        pte.write_bytes(program)
        ptd.write_bytes(named)
        np.ascontiguousarray(x.detach().numpy(), dtype="<f4").tofile(inp)
        done = subprocess.run([str(runner), "--pte", str(pte), "--ptd", str(ptd),
                               "--input", str(inp), "--output", str(out),
                               *flags],
                              capture_output=True, text=True, timeout=600)
        log = done.stdout + done.stderr
        # One output, float32 (ScalarType 6).
        if done.returncode != 0 or not re.search(r"^output 0: dtype=6 ", log, re.M):
            raise RuntimeError(f"host runner exit {done.returncode}: {log[-2000:]}")
        temps = [int(t) for t in re.findall(
            r"init: \d+ op\(s\), \d+ B blob, (\d+) B temp", log)]
        result = (np.fromfile(out, dtype="<f4"), max(temps, default=0))
        if splits:
            found = re.search(r"^esp32 convsplit: (\d+) conv op", log, re.M)
            result += (int(found.group(1)) if found else 0,)
        if scratch:
            found = re.search(r"^esp32 scratch: (\d+) call\(s\) on planned "
                              r"scratch, (\d+) on the temp allocator .*?, "
                              r"(\d+) fallback\(s\)", log, re.M)
            if found is None:
                raise RuntimeError(f"no esp32 scratch line: {log[-2000:]}")
            result += (tuple(int(g) for g in found.groups()),)
        return result


# --- the rows and the image -------------------------------------------------------------

@dataclass
class Fixture(ea.Fixture):
    row: Row = None
    info: dict = field(default_factory=dict)
    program: bytes = b""
    named: bytes = b""      # the program's own named data (merged for the image)
    report: dict = field(default_factory=dict)   # graph_report, arena, temp
    gap: dict = field(default_factory=dict)      # against the recipe's eager


def check(fixture):
    """Every MAC in a delegate, nothing outside them but the row's own
    portable ops, and board_only exactly where QEMU_BUDGET does not fit."""
    row, report = fixture.row, fixture.report
    outside = set(report["outside"]) | set(report["qdq"])
    left = sorted(o for o in outside
                  if not any(o.startswith(p) for p in row.portable + QDQ))
    missing = sorted(p for p in row.portable
                     if not any(o.startswith(p) for o in outside))
    if report["macs_delegated"] != report["macs"] or left or missing:
        raise RuntimeError(
            f"{row.name}: {report['macs_delegated']:,} of {report['macs']:,} MACs "
            f"delegated, ops outside {report['outside']} {report['qdq']}: "
            f"unexpected {left}, absent {missing}")
    need = report["arena"] + report["temp"]
    if row.board_only != (need > QEMU_BUDGET):
        raise RuntimeError(
            f"{row.name}: arena {report['arena']:,} B + delegate temp "
            f"{report['temp']:,} B = {need:,} B against QEMU's {QEMU_BUDGET:,} B, "
            f"but board_only is {row.board_only}: move the row")
    if not row.board_only and report["temp"] > PARITY_TEMP:
        raise RuntimeError(
            f"{fixture.name}: its largest delegate temp, {report['temp']:,} B"
            f"{' under min_latency' if fixture.split_pass else ''}, is past "
            f"main.cpp's temp_arena ({PARITY_TEMP:,} B): run_model would refuse it")


def fixtures(runner=None):
    """Every row: quantised, lowered, run on the host runner with the merged
    named data, checked, and compared with the recipe's eager model."""
    from executorch.backends.esp32s3.test.quant_census import MIN_COSINE
    from executorch.backends.esp32s3.test.quant_census.lower import compare

    runner = runner or host_runner()
    built = []
    for row in SPECS:
        model, inputs, info = quantized(row)
        et, named, report, arena = lower(model, inputs)
        built.append((row, model, inputs, info, bytes(et.buffer), named, report,
                      arena))
    weights = merge_named_data([b[5] for b in built])
    out = []
    for row, model, inputs, info, program, named, report, arena in built:
        reference, temp = run_host(runner, program, weights, inputs[0])
        with torch.no_grad():
            eager = model(*inputs)
        gap = compare(reference, eager, info.get("output_qparams"))
        fx = Fixture(row.name, model, inputs, reference, {"external_weights": PTD_TAG},
                     row=row, info=info, program=program, named=named, gap=gap,
                     report={**report, "arena": arena, "temp": temp})
        check(fx)
        if not gap["finite"] or gap["cosine"] < MIN_COSINE:
            raise RuntimeError(f"{row.name}: the host runner's output is past the "
                               f"census bar against the recipe's eager model: {gap}")
        out.append(fx)
    for row in SPECS:
        if not row.split:
            continue
        # The same program in the min_latency pass: no conv splits, and the bits
        # are the plain row's.
        plain = next(f for f in out if f.name == row.name)
        split, temp, splits = run_host(runner, plain.program, weights,
                                       plain.inputs[0], ("--min-latency",),
                                       splits=True)
        same = split.tobytes() == plain.reference.tobytes()
        if not same or splits != 0:
            raise RuntimeError(f"{row.name}: under --min-latency the host runner "
                               f"split {splits} conv(s), and its output "
                               f"{'is' if same else 'is NOT'} the plain run's")
        twin = replace(plain, name=f"{row.name}_split", split_pass=True,
                       program_of=row.name,
                       report={**plain.report, "temp": temp, "convsplit": splits})
        check(twin)
        out.append(twin)
    assert tuple(f.name for f in out) == ROWS
    fixtures.weights = weights
    return out


def summary(fixture):
    r, g = fixture.report, fixture.gap
    steps = f", {g['max_steps']:g} step(s)" if "max_steps" in g else ""
    where = "board only" if fixture.row.board_only else "QEMU and board"
    if fixture.split_pass:
        where += (f"; min_latency pass, {r['convsplit']} conv(s) split, "
                  f"{fixture.program_of}'s program")
    return (f"{len(fixture.program):,} B + {len(fixture.named):,} B named data, "
            f"{r['delegates']} delegate(s), {r['ops_outside']} op(s) + "
            f"{r['qdq_outside']} q/dq outside; arena {r['arena']:,} B + delegate "
            f"temp {r['temp']:,} B ({where}); vs eager cosine {g['cosine']}, "
            f"max abs {g['max_abs']:.3g}{steps}")


def write_image(path, table):
    """asym.py's image layout: every entry loads the image's named data
    (asym.FIXTURE_PTD) and asks for the heap report (FIXTURE_HEAP); a
    board-only row's also carries FIXTURE_PSRAM."""
    programs = {f.name: f.program for f in table}
    data = bytearray(ea.write_image(path, table, programs))
    for i, fixture in enumerate(table):
        at = 12 + i * ea.FIXTURE_ENTRY.size + 36            # the flags word
        word = (int.from_bytes(data[at:at + 4], "little") | FIXTURE_HEAP
                | (FIXTURE_PSRAM if fixture.row.board_only else 0))
        data[at:at + 4] = word.to_bytes(4, "little")
    with open(path, "wb") as f:
        f.write(data)
    return bytes(data)


def emitted_programs():
    """Every row's emitted program (dump_required_ops): no host runner."""
    return [lower(*quantized(row)[:2])[0] for row in SPECS]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte), the QEMU image (cnn_fixtures.bin) and its
    named data (cnn_weights.ptd). Returns {row: reference}; the rows are on
    export_all.fixtures."""
    table = fixtures()
    export_all.fixtures = {f.name: f for f in table}
    refs = {}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        if not fixture.split_pass:
            with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
                f.write(fixture.program)
        if verbose:
            print(f"  {fixture.name}.pte: {summary(fixture)}")
    with open(os.path.join(out_dir, WEIGHTS), "wb") as f:
        f.write(fixtures.weights)
    image = write_image(os.path.join(out_dir, IMAGE), table)
    if verbose:
        print(f"  {IMAGE}: {len(image):,} B; {WEIGHTS}: {len(fixtures.weights):,} B")
    return refs


if __name__ == "__main__":
    for name, ref in export_all().items():
        print(f"  {name}: reference {np.shape(ref)}")
