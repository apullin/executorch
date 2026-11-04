# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The static int8 convolution path (runtime/esp32_conv.h) against the general
loops it replaces, on the host runner.

Every geometry the path takes -- depthwise 3x3 and 5x5, dilated, strided,
padded and not; pointwise at stride 1 and 2; dense 3x3 with padding; grouped;
conv1d; with and without a clamp and a bias -- quantised by two stock static
recipes (Arm a8w8 and XNNPACK static, both affine int8 activations) and lowered
both ways: Stage B (codes between claimed ops, which the path reads in place)
and Stage A (fp32 between them, which the path quantises once per element);
and by Esp32Quantizer's XNNPACK dynamic preset, whose per-tensor affine
activation is chosen once per call over the whole input.
Each program runs with the path on and with --no-conv-fast: the outputs must
be the same bits, every int8 conv with an activation record must have taken
the path (the backend's "esp32 convfast" count equals the program's), and
--force-scalar (the scalar dot instead of the PIE gate's) changes nothing
either. Nor does --min-latency, under which no convolution splits across the
cores (on the board a split conv gave wrong outputs and cache-error panics:
the backend README's Known limits): every op on the path runs whole on one
core, the "esp32 convsplit" count stays 0, and init reserves no scratch for a
worker's part. test_min_latency_splits_no_conv runs the models at sizes where
the split, were it on, would split every conv, and
test_projections_still_split checks that min-latency still splits a
projection's columns beside them (the worker on a thread of its own on the
host, esp32_platform.h). pie_depthwise holds the 3x3
depthwise convs the PIE path runs 16 channels a vector (16, 24, 32 and 48
channels, so a remainder one by one), and test_pie_depthwise_taken checks
that they took it, so that
--force-scalar compares the vector path with the scalar one.
pie_dilated_wide holds the path's other gates on a 200-wide plane (the width
an audio model's time axis gives): a 3x3 depthwise dilated (2, 1), which PIE
takes with its taps' rows dil_h strip rows apart; one dilated 2 both ways,
which PIE refuses (its taps must be adjacent along w) and the scalar loop
takes; and a plain one. At that width the 8 KB strip holds fewer rows than
one output row reads, so pie_rows_in raises it to them: a run that never
advances (the clamp gone) times out instead of hanging the suite.
test_pie_dilation_and_wide_strips_taken pins which took PIE. On the host the
vector kernel is its C reference (esp32_reference_kernels.c); the assembly
runs in QEMU (the parity CNN rows) and on the board (the bench's self-checks).

A Conv3d over one depth plane, one tap deep, is the 2-D conv of that plane,
and the path takes it -- unless its depth axis is padded: then every depth tap
is padding (the input zero point's code), and the path, which reads plane 0,
must leave it to the general loops (taken by the path, it is 0.208 off the
converted model). Last, every convolution geometry the general loops
take -- 1-D, 2-D and 3-D, strided, padded and dilated on each axis, grouped,
depthwise with and without a channel multiplier, transposed with output
padding -- runs both ways, as parallel branches of one model per input
shape: the same bits, and the path takes exactly the ops the AOT mirror of
its admission (scratch_layout._conv_fast_op) says it does.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner; the
fixture brings that build up to date first.
"""

import contextlib
import functools
import io
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from executorch.backends.esp32s3.serialization import espressif_blob as eb
from executorch.backends.esp32s3.serialization import scratch_layout
from executorch.backends.esp32s3.test.fixtures import cnn

RECIPES = ("arm-a8w8", "xnnpack-static", "esp32-xnnpack-dynamic")
DYNAMIC = {"esp32-xnnpack-dynamic"}


def conv_bn(cin, cout, k, stride=1, padding=0, dilation=1, groups=1, act=None,
            bias=True):
    layers = [nn.Conv2d(cin, cout, k, stride, padding, dilation, groups, bias=bias)]
    if act is not None:
        layers.append(act())
    return layers


class DepthwiseChain(nn.Module):
    """MobileNet's block: 3x3 depthwise at stride 1 and 2, pad 1, then 1x1."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(3, 8, 3, 1, 1, act=nn.ReLU),
            *conv_bn(8, 8, 3, 1, 1, groups=8, act=nn.ReLU6),
            *conv_bn(8, 16, 1, act=nn.ReLU),
            *conv_bn(16, 16, 3, 2, 1, groups=16, act=nn.ReLU),
            *conv_bn(16, 8, 1),
        )

    def forward(self, x):
        return self.body(x)


class DepthwiseVariants(nn.Module):
    """5x5 depthwise, a dilated 3x3, an unpadded strided 3x3, no bias."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(4, 8, 1, act=nn.ReLU),
            *conv_bn(8, 8, 5, 1, 2, groups=8, act=nn.ReLU),
            *conv_bn(8, 8, 3, 1, 2, dilation=2, groups=8, act=nn.ReLU6),
            *conv_bn(8, 8, 3, 2, 0, groups=8, bias=False),
            *conv_bn(8, 6, 1, act=nn.ReLU),
        )

    def forward(self, x):
        return self.body(x)


class PieDepthwise(nn.Module):
    """3x3 depthwise convs the PIE path takes (dilation 1, >= 16 channels):
    16, 24 (a vector and 8 channels one by one), 32 and 48 channels, at
    stride 1 and 2, padded and not, at output widths that are not a multiple
    of the path's 32-pixel runs; pointwise convs between them."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(3, 16, 3, 1, 1, act=nn.ReLU),
            *conv_bn(16, 16, 3, 1, 1, groups=16, act=nn.ReLU6),
            *conv_bn(16, 24, 1, act=nn.ReLU),
            *conv_bn(24, 24, 3, 2, 1, groups=24, act=nn.ReLU),
            *conv_bn(24, 32, 1, act=nn.ReLU),
            *conv_bn(32, 32, 3, 1, 0, groups=32, act=nn.ReLU),
            *conv_bn(32, 48, 1, act=nn.ReLU),
            *conv_bn(48, 48, 3, 1, 1, groups=48, bias=False),
            *conv_bn(48, 8, 1),
        )

    def forward(self, x):
        return self.body(x)


class PieDilatedWide(nn.Module):
    """32 channels (16 a core when split) on a 200-wide plane, 3x3 depthwise
    each: dilated (2, 1), padded (2, 1) -- PIE; dilated 2 both ways --
    scalar, by PIE's dil_w == 1 gate; plain -- PIE. An output row of the two
    PIE ones reads 5 and 3 strip rows of 202 x 16 bytes, more than the
    strip's 8 KB holds (2), so pie_rows_in raises the strip to them."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(4, 32, 1, act=nn.ReLU),
            *conv_bn(32, 32, 3, 1, (2, 1), (2, 1), groups=32, act=nn.ReLU),
            *conv_bn(32, 32, 3, 1, 2, 2, groups=32, act=nn.ReLU),
            *conv_bn(32, 32, 3, 1, 1, groups=32, act=nn.ReLU),
            *conv_bn(32, 8, 1),
        )

    def forward(self, x):
        return self.body(x)


class PlaneDepthwise(nn.Module):
    """The depthwise convs the plane path takes (5x5, and 3x3 of fewer than
    16 channels): MobileNet-v3 Small's 5x5 at stride 1 and 2, padded 2, at 16
    and 24 channels, output rows of 37 pixels (two kernel calls, the second
    of 5) and of 19; a 3x3 of 8 channels; and an unpadded 5x5, which leaves
    the plane no room for the kernel's reads past its last row (and Stage B
    reads it in place), so the scalar loop; pointwise convs between them."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(3, 16, 1, act=nn.ReLU),
            *conv_bn(16, 16, 5, 1, 2, groups=16, act=nn.ReLU),
            *conv_bn(16, 24, 1, act=nn.ReLU),
            *conv_bn(24, 24, 5, 2, 2, groups=24, act=nn.ReLU6),
            *conv_bn(24, 8, 1, act=nn.ReLU),
            *conv_bn(8, 8, 3, 1, 1, groups=8, act=nn.ReLU),
            *conv_bn(8, 8, 5, 1, 0, groups=8, bias=False),
            *conv_bn(8, 8, 1),
        )

    def forward(self, x):
        return self.body(x)


class PairsDepthwise(nn.Module):
    """The 5x5 depthwise convs the plane path runs two output rows a vector
    (stride 1, output rows of at most 8 pixels, MobileNet-v3 Small's 7 x 7
    layers): 7 x 8 outputs (an odd row count) padded 2; row dilation 2; and
    unpadded (3 x 4: Stage B codes read in place, no scratch); pointwise
    convs between them."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(3, 16, 1, act=nn.ReLU),
            *conv_bn(16, 16, 5, 1, 2, groups=16, act=nn.ReLU),
            *conv_bn(16, 8, 1, act=nn.ReLU),
            *conv_bn(8, 8, 5, 1, (4, 2), (2, 1), groups=8, act=nn.ReLU6),
            *conv_bn(8, 8, 5, 1, 0, groups=8, bias=False),
            *conv_bn(8, 8, 1),
        )

    def forward(self, x):
        return self.body(x)


class Dense(nn.Module):
    """ResNet's convs: 3x3 with padding at stride 1 and 2, a strided 1x1, a
    grouped 3x3 and a bias-free 1x1."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(3, 8, 3, 1, 1, act=nn.ReLU),
            *conv_bn(8, 16, 3, 2, 1, act=nn.ReLU),
            *conv_bn(16, 16, 1, 2),
            *conv_bn(16, 16, 3, 1, 1, groups=2, act=nn.ReLU),
            *conv_bn(16, 8, 1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class Conv1d(nn.Module):
    """A conv1d, a depthwise conv1d and a pointwise one."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(4, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 8, 3, padding=1, groups=8), nn.ReLU(),
            nn.Conv1d(8, 6, 1),
        )

    def forward(self, x):
        return self.body(x)


class Conv3dDepth(nn.Module):
    """A Conv3d over one depth plane with a one-tap depth kernel: the 2-D
    conv of that plane when the depth axis is not padded. With pad_d=1
    (stride_d=3 keeps d_out at 1) every depth tap reads padding instead."""

    def __init__(self, pad_d):
        super().__init__()
        self.conv = nn.Conv3d(2, 4, kernel_size=(1, 3, 3), stride=(3, 1, 1),
                              padding=(pad_d, 1, 1))

    def forward(self, x):
        return torch.relu(self.conv(x))


MODELS = {
    "depthwise_chain": (DepthwiseChain, (1, 3, 13, 11)),
    "depthwise_variants": (DepthwiseVariants, (1, 4, 15, 12)),
    "dense": (Dense, (1, 3, 12, 12)),
    "conv1d": (Conv1d, (1, 4, 20)),
    "conv3d_depth_pad1": (functools.partial(Conv3dDepth, 1), (1, 2, 1, 6, 6)),
    "conv3d_depth_pad0": (functools.partial(Conv3dDepth, 0), (1, 2, 1, 6, 6)),
    "pie_depthwise": (PieDepthwise, (1, 3, 20, 37)),
    "pie_dilated_wide": (PieDilatedWide, (1, 4, 6, 200)),
    "plane_depthwise": (PlaneDepthwise, (1, 3, 23, 37)),
    "pairs_depthwise": (PairsDepthwise, (1, 3, 7, 8)),
}
# The models with fewer than three convolutions, and how many each has.
SMALL = {"conv3d_depth_pad1": 1, "conv3d_depth_pad0": 1}
# The models whose convolutions the path must leave to the general loops: a
# padded depth axis.
REFUSED = {"conv3d_depth_pad1"}
# The same models at inputs where every conv writes at least 1,024 elements
# (esp32_backend.cpp kConvSplitMinOutputs; a grouped conv's per group): the
# split, were it on (kConvSplit), would split every one under min-latency, and
# the dense convs run several row blocks. Min-latency must split none.
SPLIT_SHAPES = {
    "depthwise_chain": (1, 3, 26, 22),
    "depthwise_variants": (1, 4, 32, 28),
    "dense": (1, 3, 48, 48),
    "conv1d": (1, 4, 200),
    # Already: every conv writes at least 1,088 elements.
    "pie_depthwise": (1, 3, 20, 37),
    # Already: every conv writes at least 9,600 elements.
    "pie_dilated_wide": (1, 4, 6, 200),
}
# pie_depthwise's conv outputs at that shape, and those the scalar dot
# computes with PIE on: none. Every output's dot is PIE's: the depthwise groups
# of 16, the 24-channel depthwise conv's last 8 channels on the plane path
# (8 x 190; before it, the scalar loop's), and the dense convs' rows; on one
# core, which is where every conv runs, min-latency or not.
PIE_DW_TOTAL = 70_576
PIE_DW_SCALAR = 0
# plane_depthwise's conv outputs at its shape, and those the scalar loop
# computes with PIE on: the unpadded 5x5's (8 x 8 x 15). The other depthwise
# convs run on the plane path (16 x 23 x 37, 24 x 12 x 19 and 8 x 12 x 19
# outputs), the dense convs' rows on PIE.
PLANE_TOTAL = 58_696
PLANE_SCALAR = 8 * 120
# pairs_depthwise's conv outputs at its shape: 16 x 56 (pointwise, then the
# padded 5x5), 8 x 56 (pointwise, then the dilated 5x5), 8 x 12 (the unpadded
# 5x5, 3 x 4, then pointwise). Every depthwise conv takes the pairs form, so
# the scalar loop computes none.
PAIRS_TOTAL = 2 * 16 * 56 + 2 * 8 * 56 + 2 * 8 * 12
# pie_dilated_wide's conv outputs at that shape (32 x 1,200 for each of its
# three depthwise convs and its first pointwise one, 8 x 1,200 for its last),
# and those the scalar loop computes with PIE on: the symmetrically dilated
# depthwise conv's. The others' dots are PIE's.
PIE_WIDE_TOTAL = 163_200
PIE_WIDE_SCALAR = 32 * 1200


def quantised(name, recipe, shape=None):
    """The model at seed 0, calibrated by `recipe` on its input and four
    random batches: (the converted module, its input)."""
    cls, default = MODELS[name]
    return quantise(cls, shape or default, recipe)


def quantise(cls, shape, recipe):
    """quantised() for a model class not in MODELS."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    torch.manual_seed(0)
    net = cls().eval()
    x = torch.rand(shape) * 2 - 1
    calibration = [(torch.rand(shape) * 2 - 1,) for _ in range(4)]
    with cnn.one_thread():
        q = recipes.quantize(recipe, net, (x,), [(x,)] + calibration)
    return q.program, q.inputs[0]


def lower(model, x, stage_b):
    """cnn.lower's lowering with Stage B on or off: (program, named data)."""
    import executorch.kernels.quantized  # noqa: F401  (the q/dq .out variants)
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.memory_planning import planning_pass
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(model, (x,), strict=False)
    with contextlib.redirect_stdout(io.StringIO()):
        edge = _passes.lower_esp32(
            ep, [Esp32Partitioner(external_weights=cnn.PTD_TAG, stage_b=stage_b)])
        _passes.apply_passes(edge, quiet=True)
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    named = next((bytes(cord) for key, cord in (et._tensor_data or {}).items()
                  if key in (cnn.PTD_TAG, cnn.WEIGHTS)), b"")
    return bytes(et.buffer), named


def conv_descs(program):
    """The program's delegate conv ops with int8 weights and an activation
    record (the ones the path takes in these models)."""
    return [d for blob in eb.scan_blobs(program) for d in blob.descs
            if d["flags"] & eb.FLAG_CONV and d["flags"] & eb.FLAG_ACT_QUANT
            and d["opcode"] == eb.OP_ADDMM_I8]


def conv_ops(program):
    return len(conv_descs(program))


def run(runner, program, named, x, *flags, outputs=1, with_log=False):
    """(the output bytes, every output's in order, the "esp32 convfast" op
    count and the "esp32 convsplit" op count) of one host run, and with
    `with_log` its output."""
    with tempfile.TemporaryDirectory() as tmp:
        pte, ptd, inp = (Path(tmp) / n for n in ("m.pte", cnn.WEIGHTS, "in.bin"))
        outs = [Path(tmp) / f"out{i}.bin" for i in range(outputs)]
        pte.write_bytes(program)
        ptd.write_bytes(named)
        np.ascontiguousarray(x.detach().numpy(), dtype="<f4").tofile(inp)
        command = [str(runner), "--pte", str(pte), "--ptd", str(ptd),
                   "--input", str(inp)]
        for out in outs:
            command += ["--output", str(out)]
        # Every run here takes about a second; a conv loop that never
        # advances fails in half a minute rather than stalling the suite.
        done = subprocess.run([*command, *flags], capture_output=True,
                              text=True, timeout=30)
        log = done.stdout + done.stderr
        assert done.returncode == 0, log[-2000:]
        count = re.search(r"^esp32 convfast: (\d+) conv op", log, re.M)
        assert count is not None, "the host runner was built without the path"
        split = re.search(r"^esp32 convsplit: (\d+) conv op", log, re.M)
        assert split is not None, "the host runner cannot split the path"
        result = (b"".join(out.read_bytes() for out in outs), int(count.group(1)),
                  int(split.group(1)))
        return result + (log,) if with_log else result


def host_jobs(log):
    """The split jobs the host ran on its worker thread (the backend
    report's host line)."""
    found = re.search(r"^esp32 split: (\d+) job\(s\) on the host's worker "
                      r"thread", log, re.M)
    assert found is not None, "the host runner runs the split worker inline"
    return int(found.group(1))


def columns(log):
    """The backend report's (PIE, scalar) output columns: the conv path
    counts each output once, on the side that computed its dot."""
    found = re.search(r"columns PIE (\d+) / scalar (\d+)", log)
    assert found is not None, "no columns line in the backend report"
    return int(found.group(1)), int(found.group(2))


@pytest.fixture(scope="module")
def runner():
    return cnn.host_runner()


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("name", list(MODELS))
def test_same_bits_as_the_general_loops(runner, name, recipe, stage_b):
    model, x = quantised(name, recipe)
    program, named = lower(model, x, stage_b)
    convs = conv_ops(program)
    if recipe in DYNAMIC or (name in SMALL and recipe != "arm-a8w8"):
        # The dynamic preset quantises only some convolutions (the ones a
        # ReLU does not follow, as XNNPACK's dynamic config does), and only
        # Arm's static recipe is required to quantise a Conv3d.
        if convs == 0:
            pytest.skip(f"{recipe} leaves every conv of {name} in float")
    else:
        need = SMALL.get(name, 3)
        assert convs >= need, f"{name}: only {convs} int8 conv op(s) delegated"
    fast, taken, unsplit = run(runner, program, named, x)
    general, none, _ = run(runner, program, named, x, "--no-conv-fast")
    scalar, _, _ = run(runner, program, named, x, "--force-scalar")
    split, _, splits = run(runner, program, named, x, "--min-latency")
    assert unsplit == 0, "a conv split without min-latency"
    assert fast == general, "the path's output differs from the general loops'"
    want = 0 if name in REFUSED else convs
    assert taken == want, f"{taken} of {convs} conv ops took the path, not {want}"
    assert none == 0
    assert scalar == fast, "the scalar dot changed the path's output"
    assert splits == 0, f"min-latency split {splits} conv(s)"
    assert split == fast, "min-latency changed the output"
    out = np.frombuffer(fast, dtype="<f4")
    assert np.isfinite(out).all() and np.abs(out).max() > 0


# ---- every geometry the general loops take, both ways

C3, C2, C1 = nn.Conv3d, nn.Conv2d, nn.Conv1d
T3, T2, T1 = nn.ConvTranspose3d, nn.ConvTranspose2d, nn.ConvTranspose1d
# {input shape: [conv, ...]}, each conv a branch reading the input. On the
# depth axis of a one-plane input: one tap at stride 1, 2 and 3 with and
# without padding, dilated, a three-tap kernel padded back to one plane, and
# each of those grouped and depthwise; a three-plane input; then 2-D and 1-D
# convs strided, padded and dilated, grouped, depthwise with a channel
# multiplier, and transposed with output padding.
GEOMETRIES = {
    (1, 4, 1, 6, 6): [
        C3(4, 4, (1, 3, 3), padding=(0, 1, 1)),
        C3(4, 4, (1, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
        C3(4, 4, (1, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1)),
        C3(4, 4, (1, 3, 3), padding=(0, 1, 1), dilation=(2, 1, 1)),
        C3(4, 4, (1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1)),
        C3(4, 4, (1, 3, 3), stride=(3, 2, 2), padding=(1, 0, 0), dilation=(3, 1, 1)),
        C3(4, 4, (1, 1, 1), stride=(5, 1, 1), padding=(2, 0, 0)),
        C3(4, 4, (3, 3, 3), padding=(1, 1, 1)),
        C3(4, 4, (2, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
        C3(4, 4, (1, 3, 3), padding=(0, 1, 1), groups=4),
        C3(4, 4, (1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), groups=4),
        C3(4, 8, (1, 3, 3), padding=(0, 1, 1), groups=2),
        C3(4, 8, (1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), groups=2),
        C3(4, 8, (1, 3, 3), padding=(0, 1, 1), groups=4),
        T3(4, 4, (1, 3, 3), padding=(0, 1, 1)),
    ],
    (1, 4, 3, 5, 5): [
        C3(4, 4, (1, 3, 3), padding=(0, 1, 1)),
        C3(4, 4, (3, 3, 3), padding=(0, 1, 1)),
        C3(4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        C3(4, 4, (3, 3, 3), padding=(1, 1, 1), groups=4),
        T3(4, 4, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1),
           output_padding=(1, 0, 0)),
    ],
    (1, 8, 7, 7): [
        C2(8, 8, 3, padding=1),
        C2(8, 8, 3, stride=2),
        C2(8, 8, 3, padding=2, dilation=2),
        C2(8, 8, (1, 3), stride=(2, 1), padding=(0, 1)),
        C2(8, 8, 1),
        C2(8, 8, 1, stride=2),
        C2(8, 8, 3, padding=1, groups=2),
        C2(8, 8, 3, padding=1, groups=8),
        C2(8, 16, 3, padding=1, groups=8),
        C2(8, 8, 5, stride=2, padding=2, groups=8),
        T2(8, 4, 3, stride=2, padding=1, output_padding=1),
        T2(8, 8, 3, stride=2, padding=1, output_padding=1, groups=2),
        T2(8, 4, 3, stride=3, padding=0, dilation=2, output_padding=2),
    ],
    (1, 4, 12): [
        C1(4, 8, 3, padding=1),
        C1(4, 8, 3, stride=2),
        C1(4, 4, 3, padding=2, dilation=2, groups=4),
        C1(4, 8, 1),
        T1(4, 4, 3, stride=2, output_padding=1),
    ],
}


class Branches(nn.Module):
    """Every conv of one input shape, each on the input, each its own
    output."""

    def __init__(self, convs):
        super().__init__()
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        return tuple(conv(x) for conv in self.convs)


def path_ops(program):
    """(the ops the AOT mirror says the path takes, the delegate's int8
    convolutions with an activation record whose depth is one plane, one
    tap and padded)."""
    taken = padded = 0
    for blob in eb.scan_blobs(program):
        for op in scratch_layout.read_ops(blob.data[blob.offset:]):
            taken += scratch_layout._conv_fast_op(op)
            g = op.conv
            padded += (g is not None and op.act is not None
                       and op.desc.opcode == eb.OP_ADDMM_I8
                       and g.d_in == g.d_out == g.kd == 1 and g.pad_d > 0)
    return taken, padded


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("shape", list(GEOMETRIES),
                         ids=["x".join(map(str, s)) for s in GEOMETRIES])
def test_every_geometry_same_bits(runner, shape, stage_b):
    """Each convolution geometry the general loops take, quantised by Arm's
    static recipe: the path's output bits are the general loops', and it
    takes exactly the ops scratch_layout._conv_fast_op admits (the init
    scratch declared for them): every 2-D one, none transposed, and no 3-D
    one but a one-plane, one-tap, unpadded depth."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    torch.manual_seed(0)
    net = Branches(GEOMETRIES[shape]).eval()
    x = torch.rand(shape) * 2 - 1
    calibration = [(torch.rand(shape) * 2 - 1,) for _ in range(4)]
    with cnn.one_thread():
        q = recipes.quantize("arm-a8w8", net, (x,), [(x,)] + calibration)
    x = q.inputs[0]
    program, named = lower(q.program, x, stage_b)
    outputs = len(GEOMETRIES[shape])
    fast, taken, _ = run(runner, program, named, x, outputs=outputs)
    general, none, _ = run(runner, program, named, x, "--no-conv-fast",
                           outputs=outputs)
    admitted, padded = path_ops(program)
    assert fast == general, "the path's output differs from the general loops'"
    assert none == 0
    assert taken == admitted, (
        f"{taken} op(s) took the path; the AOT mirror admits {admitted}")
    assert conv_ops(program) > 0, "no int8 convolution was delegated"
    if shape[2:3] == (3,) and len(shape) == 5:
        # Three depth planes: no op is the 2-D conv of one.
        assert admitted == 0, f"{admitted} op(s) of a three-plane input admitted"
    else:
        assert admitted > 0, f"no op of {shape} admitted"
    if shape == (1, 4, 1, 6, 6):
        assert padded > 0, "no padded one-plane depth reached the delegate"


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
@pytest.mark.parametrize("name", list(SPLIT_SHAPES))
def test_min_latency_splits_no_conv(runner, name, recipe, stage_b):
    """At SPLIT_SHAPES, where the split, were it on, would split every conv on
    the path, min-latency splits none and starts no worker (these models have
    no projection to split), and the output is the single-core path's and the
    general loops' bit for bit, with the PIE dot and the scalar one."""
    model, x = quantised(name, recipe, SPLIT_SHAPES[name])
    program, named = lower(model, x, stage_b)
    convs = conv_ops(program)
    assert convs >= 3, f"{name}: only {convs} int8 conv op(s) delegated"
    fast, taken, _ = run(runner, program, named, x)
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    dual, _, splits, log = run(runner, program, named, x, "--min-latency",
                               with_log=True)
    scalar, _, scalar_splits = run(runner, program, named, x, "--min-latency",
                                   "--force-scalar")
    assert taken == convs and splits == 0 and scalar_splits == 0
    assert host_jobs(log) == 0, "min-latency started a worker for a conv"
    assert fast == general, "the path's output differs from the general loops'"
    assert dual == fast, "min-latency changed the output"
    assert scalar == fast, "the scalar dot changed the min-latency output"


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_pie_depthwise_taken(runner, recipe, stage_b):
    """pie_depthwise's depthwise convs run on the PIE path, every whole group
    of 16 channels, after its start-up self-test passed, on one core, the same
    under min-latency (no conv splits); --force-scalar runs no output there.
    The other tests compare these runs' bits with each other and with the
    general loops."""
    model, x = quantised("pie_depthwise", recipe)
    program, named = lower(model, x, stage_b)
    total = sum(d["m"] * d["n"] for d in conv_descs(program))
    assert total == PIE_DW_TOTAL, f"{total} conv outputs on the path"
    fast, _, _, log = run(runner, program, named, x, with_log=True)
    assert "PIE depthwise 16-lane selftest: PASS" in log, log[-2000:]
    assert KERNELS_PASS in log, log[-2000:]
    assert columns(log) == (total - PIE_DW_SCALAR, PIE_DW_SCALAR)
    split, _, splits, log = run(runner, program, named, x, "--min-latency",
                                with_log=True)
    assert splits == 0
    assert columns(log) == (total - PIE_DW_SCALAR, PIE_DW_SCALAR)
    scalar, _, _, log = run(runner, program, named, x, "--force-scalar",
                            with_log=True)
    assert columns(log) == (0, total)
    assert scalar == fast and split == fast


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_pie_dilation_and_wide_strips_taken(runner, recipe, stage_b):
    """pie_dilated_wide's (2, 1)-dilated and plain depthwise convs run on
    the PIE path, their strips raised to the rows an output row reads, and the
    one dilated both ways runs scalar, refused by the path's dil_w == 1 gate:
    on one core, and the same under min-latency (no conv splits);
    --force-scalar runs none of them on PIE; and every run's bits are the
    general loops'."""
    model, x = quantised("pie_dilated_wide", recipe)
    program, named = lower(model, x, stage_b)
    total = sum(d["m"] * d["n"] for d in conv_descs(program))
    assert total == PIE_WIDE_TOTAL, f"{total} conv outputs on the path"
    fast, _, _, log = run(runner, program, named, x, with_log=True)
    assert "PIE depthwise 16-lane selftest: PASS" in log, log[-2000:]
    assert KERNELS_PASS in log, log[-2000:]
    assert columns(log) == (total - PIE_WIDE_SCALAR, PIE_WIDE_SCALAR)
    split, _, splits, log = run(runner, program, named, x, "--min-latency",
                                with_log=True)
    assert splits == 0
    assert columns(log) == (total - PIE_WIDE_SCALAR, PIE_WIDE_SCALAR)
    scalar, _, _, log = run(runner, program, named, x, "--force-scalar",
                            with_log=True)
    assert columns(log) == (0, total)
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    assert fast == split == scalar == general


# The conv path's other kernels' boot check (esp32_conv.cpp pie_dw_init): the
# fused dot and epilogue, and the transpose; on a FAIL each step takes the code
# it replaces, with the same bits.
KERNELS_PASS = ("PIE conv kernels selftest: fused dot and epilogue PASS, "
                "transpose PASS")


def plane_kernel_calls(log):
    """The host runner's calls into the depthwise plane kernels since it
    started (their boot self-test's included)."""
    found = re.search(r"host depthwise plane kernels: (\d+) call", log)
    assert found is not None, "no host depthwise plane kernels line"
    return int(found.group(1))


PLANES_PASS = ", depthwise planes PASS"


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_plane_depthwise_taken(runner, recipe, stage_b):
    """plane_depthwise's padded 5x5 convs and its 8-channel 3x3 run on the
    plane path (PIE) after its boot self-test passed, the unpadded 5x5 on the
    scalar loop; the same under min-latency (no conv splits); --force-scalar
    runs no output on PIE. Their outputs are the scalar loop's bits, so only
    the counts can show a gate that closed: the plane kernels must run more
    often than in the --force-scalar run, which calls them only in the boot
    self-test. Every run's bits are the general loops'."""
    model, x = quantised("plane_depthwise", recipe)
    program, named = lower(model, x, stage_b)
    total = sum(d["m"] * d["n"] for d in conv_descs(program))
    assert total == PLANE_TOTAL, f"{total} conv outputs on the path"
    fast, _, _, log = run(runner, program, named, x, with_log=True)
    assert KERNELS_PASS + PLANES_PASS in log, log[-2000:]
    assert columns(log) == (total - PLANE_SCALAR, PLANE_SCALAR)
    on = plane_kernel_calls(log)
    split, _, splits, log = run(runner, program, named, x, "--min-latency",
                                with_log=True)
    assert splits == 0
    assert columns(log) == (total - PLANE_SCALAR, PLANE_SCALAR)
    scalar, _, _, log = run(runner, program, named, x, "--force-scalar",
                            with_log=True)
    assert columns(log) == (0, total)
    off = plane_kernel_calls(log)
    assert on > off, f"the plane kernels never ran: {on} vs {off}"
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    assert fast == split == scalar == general


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_pairs_depthwise_taken(runner, recipe, stage_b):
    """pairs_depthwise's 5x5 convs all run on PIE in the plane path's
    two-rows-a-vector form (the unpadded one too: it needs no scratch), as
    test_plane_depthwise_taken checks the plane path: columns, kernel calls
    above a --force-scalar run's, and every run's bits the general loops'."""
    model, x = quantised("pairs_depthwise", recipe)
    program, named = lower(model, x, stage_b)
    total = sum(d["m"] * d["n"] for d in conv_descs(program))
    assert total == PAIRS_TOTAL, f"{total} conv outputs on the path"
    fast, _, _, log = run(runner, program, named, x, with_log=True)
    assert KERNELS_PASS + PLANES_PASS in log, log[-2000:]
    assert columns(log) == (total, 0)
    on = plane_kernel_calls(log)
    split, _, splits, log = run(runner, program, named, x, "--min-latency",
                                with_log=True)
    assert splits == 0
    assert columns(log) == (total, 0)
    scalar, _, _, log = run(runner, program, named, x, "--force-scalar",
                            with_log=True)
    assert columns(log) == (0, total)
    assert on > plane_kernel_calls(log), "the pairs kernel never ran"
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    assert fast == split == scalar == general


class DsCnnBlocks(nn.Module):
    """DS-CNN's body at its shapes (25 x 5, 64 channels): a pointwise conv
    making 64 channels of codes, then two blocks of a 3x3 depthwise conv and
    a 64 -> 64 pointwise conv, each conv with a ReLU -- the convs the board's
    DS-CNN time is made of, its first conv aside."""

    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(8, 64, 1), nn.ReLU()]
        for _ in range(2):
            layers += [nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.ReLU(),
                       nn.Conv2d(64, 64, 1), nn.ReLU()]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def conv_kernel_calls(log):
    """The host runner's calls into the conv path's three kernels since it
    started (their boot self-tests' included)."""
    found = re.search(r"host conv kernels: qacc_lane_codes (\d+), "
                      r"dot_rows_codes (\d+), transpose16 (\d+) call", log)
    assert found is not None, "no host conv kernels line in the runner's output"
    return dict(zip(("qacc_lane_codes", "dot_rows_codes", "transpose16"),
                    map(int, found.groups())))


@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_conv_kernels_serve_ds_cnn_blocks(runner, recipe):
    """On DS-CNN's own shapes, lowered with Stage B as the bench runs them,
    each of the path's three kernels serves its convs: qacc_lane_codes_s3 the
    depthwise convs' codes, dot_rows_codes_s3 the pointwise convs (codes out,
    and the last one's fp32), transpose16_s3 the depthwise strips and the
    pointwise rows. Their outputs are the bits of the loops they replace --
    the general loops' here -- so only their calls can show a gate that
    closed (a `fusable` turned false, codes out sent to lane_float, the
    transpose's flag off): each count must be above a
    --force-scalar run's, which calls them only in the boot self-tests (those
    run at init either way). Both self-test lines must say PASS."""
    model, x = quantise(DsCnnBlocks, (1, 8, 25, 5), recipe)
    program, named = lower(model, x, True)
    assert conv_ops(program) == 5
    fast, taken, _, log = run(runner, program, named, x, with_log=True)
    assert taken == 5
    assert "PIE depthwise 16-lane selftest: PASS" in log, log[-2000:]
    assert KERNELS_PASS in log, log[-2000:]
    on = conv_kernel_calls(log)
    scalar, _, _, log = run(runner, program, named, x, "--force-scalar",
                            with_log=True)
    off = conv_kernel_calls(log)
    for kernel, calls in on.items():
        assert calls > off[kernel], f"{kernel} never ran: {on} vs {off}"
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    assert fast == scalar == general


class LongRows(nn.Module):
    """Pointwise convs whose rows are past the fused kernel's registers
    (k_pad 144 and 160, over 112), as MobileNetV2's projections are, between
    convs from and to short rows; the first reads the model's fp32 input, a
    stride-1 pointwise conv quantising it as it gathers (QuantFast)."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(8, 144, 1), nn.ReLU(),
            nn.Conv2d(144, 24, 1), nn.ReLU(),
            nn.Conv2d(24, 160, 1), nn.ReLU(),
            nn.Conv2d(160, 32, 1),
        )

    def forward(self, x):
        return self.body(x)


LONG_ROWS_PASS = "PIE long-row dot selftest: PASS"


def long_row_calls(log):
    """The host runner's calls into dot_rows_long_s3 since it started (its
    boot self-test's included)."""
    found = re.search(r"host conv long rows: dot_rows_long (\d+) call", log)
    assert found is not None, "no host conv long rows line in the runner's output"
    return int(found.group(1))


@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_long_row_kernel_serves_wide_pointwise(runner, recipe):
    """Pointwise convs with k_pad past 112 take dot_rows_long_s3 (the dot and
    the integer epilogue in one loop, the weights past their sixth chunk
    streamed): it runs more often than in a --force-scalar run, where only
    its boot self-test calls it, and the bits are the scalar loops' and the
    general loops'. A gate that closed (the kernel's flag, or `fusable`
    for long rows) keeps the bits, so only the count shows it."""
    model, x = quantise(LongRows, (1, 8, 12, 12), recipe)
    program, named = lower(model, x, True)
    assert conv_ops(program) == 4
    fast, taken, _, log = run(runner, program, named, x, with_log=True)
    assert taken == 4
    assert KERNELS_PASS in log and LONG_ROWS_PASS in log, log[-2000:]
    on = long_row_calls(log)
    # Its first conv reads the model's fp32 input at stride 1: QuantFast's
    # pointwise gather (the boot self-tests never call it).
    assert fp32_fast_calls(log)["gather_pointwise_fast"] > 0, log[-2000:]
    scalar, _, _, log = run(runner, program, named, x, "--force-scalar",
                            with_log=True)
    off = long_row_calls(log)
    assert on > off, f"dot_rows_long_s3 never ran: {on} vs {off}"
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    assert fast == scalar == general


def fp32_fast_calls(log):
    """The host runner's calls into the dense convs' fp32 fast paths
    (QuantFast): prepare's plane, stride-1 pointwise rows, other dense rows."""
    found = re.search(r"host conv fp32 codes: quantize_span_fast (\d+), "
                      r"gather_pointwise_fast (\d+), gather_dense_fast (\d+) call",
                      log)
    assert found is not None, "no host conv fp32 codes line in the runner's output"
    return dict(zip(("quantize_span_fast", "gather_pointwise_fast",
                     "gather_dense_fast"), map(int, found.groups())))


class Fp32Stem(nn.Module):
    """A first conv on the model's fp32 input, 3x3 stride 2 from 3 channels
    as the MobileNets' stems are, then a pointwise conv on its codes."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 1),
        )

    def forward(self, x):
        return self.body(x)


# 3 x 106 x 106 is past kPlaneBudget (32 KB of codes): the first conv
# quantises each tap as it gathers its rows (gather_dense_fast, which copies
# the overlapping columns); 3 x 64 x 64 fits, so its input is quantised whole
# first (quantize_span_fast).
@pytest.mark.parametrize("size,route", ((106, "gather_dense_fast"),
                                        (64, "quantize_span_fast")),
                         ids=("past_plane_budget", "plane"))
@pytest.mark.parametrize("recipe", ("arm-a8w8", "xnnpack-static"))
def test_fp32_stem_takes_the_branch_free_quantiser(runner, recipe, size, route):
    """A dense conv on fp32 input takes the branch-free quantiser
    (esp32_conv.cpp QuantFast) on its route -- gather_dense_fast end to end
    past kPlaneBudget, quantize_span_fast within it -- and gives the bits of
    the general loops (and of the scalar dots). Its codes are code_of's, so a
    route not taken (dense() or quantize_span falling back to code_of:
    MobileNetV3-Small at ~900 ms) shows only in its call count."""
    model, x = quantise(Fp32Stem, (1, 3, size, size), recipe)
    program, named = lower(model, x, True)
    assert conv_ops(program) == 2
    fast, taken, _, log = run(runner, program, named, x, with_log=True)
    assert taken == 2
    calls = fp32_fast_calls(log)
    assert calls[route] > 0, f"{route} never ran: {calls}"
    scalar, _, _ = run(runner, program, named, x, "--force-scalar")
    general, _, _ = run(runner, program, named, x, "--no-conv-fast")
    assert fast == scalar == general


class ConvHead(nn.Module):
    """Two convs on the path, then a projection wide enough for min-latency
    to split its columns (64, at least the runtime's kSplitMinCols): a CNN
    with a classifier, as the bench's MobileNetV2s are."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            *conv_bn(3, 16, 3, 1, 1, act=nn.ReLU),
            *conv_bn(16, 16, 3, 2, 1, act=nn.ReLU),
        )
        self.head = nn.Linear(16 * 8 * 8, 64)

    def forward(self, x):
        return self.head(torch.flatten(self.body(x), 1))


def split_rows(log):
    """The projection rows the backend split across the cores (its report's
    column-split count)."""
    found = re.search(r"^esp32 split: (\d+) column-split addmm row\(s\) forked",
                      log, re.M)
    assert found is not None, "no column-split line in the backend report"
    return int(found.group(1))


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
def test_projections_still_split(runner, stage_b):
    """Min-latency still splits a projection's columns across the cores (the
    story's matmul split, exact on silicon) while every conv beside it runs
    on one core: the convs take the path and split none, the projection's row
    forks once, on the host's worker thread, and the bits are the single-core
    run's."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    torch.manual_seed(0)
    net = ConvHead().eval()
    x = torch.rand(1, 3, 16, 16) * 2 - 1
    calibration = [(torch.rand(1, 3, 16, 16) * 2 - 1,) for _ in range(4)]
    with cnn.one_thread():
        q = recipes.quantize("arm-a8w8", net, (x,), [(x,)] + calibration)
    program, named = lower(q.program, q.inputs[0], stage_b)
    assert conv_ops(program) == 2
    single, taken, splits, single_log = run(runner, program, named, q.inputs[0],
                                            with_log=True)
    dual, _, dual_splits, dual_log = run(runner, program, named, q.inputs[0],
                                         "--min-latency", with_log=True)
    assert taken == 2 and splits == 0 and dual_splits == 0
    assert split_rows(single_log) == 0 and host_jobs(single_log) == 0
    assert split_rows(dual_log) > 0, "min-latency no longer splits the projection"
    assert host_jobs(dual_log) > 0
    assert dual == single, "splitting the projection changed the output"


def kept_on_one_core(log):
    """The calls the backend kept on one core because they write PSRAM."""
    found = re.search(r"^esp32 split: (\d+) call\(s\) kept on one core: they write PSRAM",
                      log, re.M)
    assert found, "the backend report has no count of calls kept on one core"
    return int(found.group(1))


@pytest.mark.parametrize("stage_b", [True, False], ids=["stage_b", "stage_a"])
def test_projection_writing_psram_runs_on_one_core(runner, stage_b):
    """A projection whose output is in PSRAM does not split under min-latency:
    on the board, a line of PSRAM written while both cores run can lose bytes
    (the backend README's Known limits). The runner's --psram-planned marks
    every planned buffer (the delegate's outputs and its scratch) as PSRAM:
    then nothing forks, every call is counted as kept on one core, and the
    bits are the same. The same program with its planned buffers internal
    still splits its projection."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    torch.manual_seed(0)
    net = ConvHead().eval()
    x = torch.rand(1, 3, 16, 16) * 2 - 1
    calibration = [(torch.rand(1, 3, 16, 16) * 2 - 1,) for _ in range(4)]
    with cnn.one_thread():
        q = recipes.quantize("arm-a8w8", net, (x,), [(x,)] + calibration)
    program, named = lower(q.program, q.inputs[0], stage_b)
    single, _, _, single_log = run(runner, program, named, q.inputs[0], with_log=True)
    internal, _, _, internal_log = run(runner, program, named, q.inputs[0],
                                       "--min-latency", with_log=True)
    psram, _, _, psram_log = run(runner, program, named, q.inputs[0],
                                 "--min-latency", "--psram-planned", with_log=True)
    assert split_rows(internal_log) > 0 and host_jobs(internal_log) > 0, (
        "with internal planned buffers the projection no longer splits")
    assert kept_on_one_core(internal_log) == 0
    assert split_rows(psram_log) == 0, "a projection writing PSRAM split"
    assert host_jobs(psram_log) == 0
    assert kept_on_one_core(psram_log) > 0
    assert kept_on_one_core(single_log) == 0, "a single-core run counted a split refused"
    assert psram == single and internal == single, "the output changed"


def init_temps(log):
    """Each delegate's temp, in init order (its init report's "B temp")."""
    return [int(t) for t in re.findall(r"init: \d+ op\(s\), \d+ B blob, (\d+) B temp",
                                       log)]


def test_min_latency_reserves_no_conv_scratch(runner):
    """With no conv split, init reserves no scratch for a split worker's part
    of a conv: under min-latency (the option set before init, as a dual-core
    firmware does) every delegate's temp is the single-core one's, the bits
    are the same, and nothing splits."""
    for name in SPLIT_SHAPES:
        for stage_b in (True, False):
            model, x = quantised(name, "arm-a8w8", SPLIT_SHAPES[name])
            program, named = lower(model, x, stage_b)
            single, _, unsplit, single_log = run(runner, program, named, x,
                                                 with_log=True)
            dual, _, splits, dual_log = run(runner, program, named, x,
                                            "--min-latency", with_log=True)
            s, d = init_temps(single_log), init_temps(dual_log)
            assert single == dual and unsplit == 0 and splits == 0
            assert len(s) == len(d) > 0 and s == d, (name, stage_b, s, d)
