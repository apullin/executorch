# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Channels-last graph inputs (make_inputs_row_major_pass.py): the delegate takes the
row-major program, and the bits are the row-major program's.

lower_esp32 copies a graph input in another dim order (the Cortex-M flow's
NHWC images) to row-major right after to_edge, re-derives what follows, and
copies an output whose dim order that changed back to the declared one. Here:

  - a row-major input leaves the program untouched, byte for byte;
  - channels-last convolutions (the backend's own weights and activations,
    and static pairs as convert_pt2e spells them) are claimed, each delegate
    blob is byte-identical to the row-major program's, and the host runner's
    output equals the row-major program's on the same tensor exactly, the
    program's input and output keeping their channels-last dim order;
  - a squeeze-excite block, whose portable hardswish adds a 0-dim constant to
    a [1, C, 1, 1] tensor (the census's se_hardswish: without the copy,
    ExecuTorch's add refuses the two dim orders), runs;
  - a reflect or replicate pad, whose decomposition copies its result back
    to the input's channels-last order, follows the input instead: the conv
    after it is claimed as for the row-major input, same blob, same bits;
  - a dim order the model sets itself stays portable, refused by name (also
    a channels-last clone of a channels-last input), and so does everything
    when the metadata cannot be re-derived (the graph is left as it was).

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

from executorch.backends.esp32s3 import _passes
from executorch.backends.esp32s3._passes import make_inputs_row_major_pass
import executorch.backends.esp32s3._passes.make_inputs_row_major_pass as row_major
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
CL = (0, 2, 3, 1)
qd = torch.ops.quantized_decomposed


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


class ConvChain(nn.Module):
    """Float weights (the backend quantises them, activations per row):
    3x3 padded + ReLU, depthwise 3x3 strided, 1x1."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.a = nn.Conv2d(4, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8)
        self.pw = nn.Conv2d(8, 6, 1)

    def forward(self, x):
        return self.pw(self.dw(F.relu(self.a(x))))


def _pair(t, s, z):
    q = qd.quantize_per_tensor(t, s, z, -128, 127, torch.int8)
    return qd.dequantize_per_tensor(q, s, z, -128, 127, torch.int8)


class StaticConvs(nn.Module):
    """Static pairs with float32 constants around two int8 per-channel convs,
    as convert_pt2e leaves them (the 3x3's padding is the input zero point)."""

    S = (float(F32(1 / 64)), float(F32(1 / 32)), float(F32(1 / 16)))

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(9)
        for name, shape in (("1", (8, 4, 3, 3)), ("2", (4, 8, 1, 1))):
            self.register_buffer(f"c{name}", torch.randint(
                -127, 128, shape, generator=g, dtype=torch.int8))
            self.register_buffer(f"s{name}",
                                 torch.rand(shape[0], generator=g) * 4e-3 + 1e-3)
            self.register_buffer(f"z{name}", torch.zeros(shape[0],
                                                         dtype=torch.int64))
            self.register_buffer(f"b{name}", torch.randn(shape[0], generator=g)
                                 * 0.05)

    def _w(self, i):
        return qd.dequantize_per_channel(
            getattr(self, f"c{i}"), getattr(self, f"s{i}"),
            getattr(self, f"z{i}"), 0, -127, 127, torch.int8)

    def forward(self, x):
        h = _pair(F.conv2d(_pair(x, self.S[0], 3), self._w(1), self.b1,
                           padding=1), self.S[1], -5)
        return _pair(F.conv2d(h, self._w(2), self.b2), self.S[2], 2)


class SqueezeExcite(nn.Module):
    """MobileNetV3's block: conv, hardswish, then a hardsigmoid gate on
    [1, C, 1, 1] tensors; both activations stay portable and add a 0-dim
    constant."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.fc1 = nn.Conv2d(8, 4, 1)
        self.fc2 = nn.Conv2d(4, 8, 1)
        self.project = nn.Conv2d(8, 8, 1)

    def forward(self, x):
        h = F.hardswish(self.conv(x))
        s = F.hardsigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(h, 1)))))
        return self.project(h * s)


class InnerChannelsLast(nn.Module):
    """A row-major input the model itself turns channels-last."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(13)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x.contiguous(memory_format=torch.channels_last))


class PadConv(nn.Module):
    """A pad torch decomposes into index ops and a copy back to the input's
    memory format, then a conv and its ReLU."""

    def __init__(self, mode):
        super().__init__()
        torch.manual_seed(19)
        self.mode = mode
        self.conv = nn.Conv2d(4, 8, 3, stride=2)

    def forward(self, x):
        return torch.relu(self.conv(F.pad(x, (1, 1, 1, 1), mode=self.mode)))


class CloneChannelsLast(nn.Module):
    """A channels-last input the model copies to channels-last itself."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(13)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x.clone(memory_format=torch.channels_last))


class TokensPlusLinear(nn.Module):
    """x + Linear(4, 4)(y), x [1, C, H, 1] and y [1, C, H, 4]. With x fed
    channels-last, eager's add takes its layout from both operands, x's
    channels-last and the Linear's row-major output: strides in dim order
    [0, 2, 1, 3], which the program declares for its output (also a
    reflect pad's output in place of the Linear's)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(17)
        self.fc = nn.Linear(4, 4)

    def forward(self, x, y):
        return x + self.fc(y)


def image(shape=(1, 4, 9, 9), seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def lower(model, x):
    partitioner = Esp32Partitioner()
    ep = torch.export.export(model.eval(), x if isinstance(x, tuple) else (x,))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lowered = _passes.lower_esp32(ep, [partitioner])
    return lowered, partitioner, log.getvalue()


def portable(lowered):
    return [str(n.target) for n in lowered.exported_program().graph.nodes
            if n.op == "call_function"
            and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)]


def blobs(lowered):
    gm = lowered.exported_program().graph_module
    return [getattr(gm, n.args[0].target).processed_bytes
            for n in gm.graph.nodes if n.op == "call_function"
            and "call_delegate" in str(n.target)]


def emit(lowered):
    from executorch.exir import ExecutorchBackendConfig

    _passes.apply_passes(lowered, quiet=True)
    return lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16))


def io_orders(et):
    plan = et.executorch_program.execution_plan[0]
    return ([tuple(plan.values[i].val.dim_order) for i in plan.inputs],
            [tuple(plan.values[i].val.dim_order) for i in plan.outputs])


def run_host(runner, tmp_path, et, x, like_shape):
    """The program on the host runner: each input (x, or a tuple of them)
    written, and the output read, in the memory order the program declares."""
    ins, outs = io_orders(et)
    pte = tmp_path / "model.pte"
    pte.write_bytes(et.buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, (t, order) in enumerate(zip(x if isinstance(x, tuple) else (x,), ins)):
        t = t.detach()
        if order != tuple(range(t.dim())):
            t = t.permute(*order)
        np.ascontiguousarray(t.contiguous().numpy(), F32).tofile(
            tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    command += ["--output", str(tmp_path / "out.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    data = np.fromfile(tmp_path / "out.bin", dtype=F32)
    order = outs[0]
    if order != tuple(range(len(like_shape))):
        stored = [like_shape[d] for d in order]
        return np.transpose(data.reshape(stored), np.argsort(order))
    return data.reshape(like_shape)


def bits(a):
    return np.ascontiguousarray(a, F32).view(np.uint32)


def test_row_major_inputs_are_untouched(monkeypatch):
    """No copy, no report, and the same program bytes as with the pass off."""
    x = image()
    lowered, _, log = lower(ConvChain(), x)
    assert not lowered.row_major_report.inputs
    assert "row-major inputs" not in log
    assert not any("_to_dim_order_copy" in t for t in portable(lowered))
    with_pass = bytes(emit(lowered).buffer)
    monkeypatch.setattr(row_major, "make_inputs_row_major",
                        lambda edge, method_name="forward":
                        make_inputs_row_major_pass.RowMajorReport())
    without, _, _ = lower(ConvChain(), x)
    assert bytes(emit(without).buffer) == with_pass


@pytest.mark.parametrize("make", [ConvChain, StaticConvs],
                         ids=["backend-weights", "static-pairs"])
def test_channels_last_is_the_row_major_program(runner, tmp_path, make):
    x = image()
    cl, part, log = lower(make(), x.contiguous(memory_format=torch.channels_last))
    rm, _, _ = lower(make(), x)
    report = cl.row_major_report
    assert [order for _, order in report.inputs] == [CL], log
    assert "runtime tensor is not row-major" not in part.support.kinds
    left = portable(cl)
    assert not any("convolution" in t for t in left), left
    assert sum("_to_dim_order_copy" in t for t in left) == 2, left  # in, out
    assert blobs(cl) == blobs(rm) and blobs(cl)
    et_cl, et_rm = emit(cl), emit(rm)
    assert io_orders(et_cl) == ([CL], [CL])
    with torch.no_grad():
        shape = tuple(make().eval()(x).shape)
    (tmp_path / "cl").mkdir()
    (tmp_path / "rm").mkdir()
    got_cl = run_host(runner, tmp_path / "cl", et_cl, x, shape)
    got_rm = run_host(runner, tmp_path / "rm", et_rm, x, shape)
    np.testing.assert_array_equal(bits(got_cl), bits(got_rm))


def test_squeeze_excite_runs(runner, tmp_path):
    """The census's se_hardswish case, without the quantizer: channels-last
    activations meeting a 0-dim constant in a portable add."""
    x = image(seed=4)
    cl, _, _ = lower(SqueezeExcite(), x.contiguous(memory_format=torch.channels_last))
    rm, _, _ = lower(SqueezeExcite(), x)
    assert blobs(cl) == blobs(rm)
    with torch.no_grad():
        shape = tuple(SqueezeExcite().eval()(x).shape)
    (tmp_path / "cl").mkdir()
    (tmp_path / "rm").mkdir()
    got_cl = run_host(runner, tmp_path / "cl", emit(cl), x, shape)
    got_rm = run_host(runner, tmp_path / "rm", emit(rm), x, shape)
    np.testing.assert_array_equal(bits(got_cl), bits(got_rm))


def test_an_output_order_no_copy_writes_stays_row_major(runner, tmp_path):
    """An output declared in dim order [0, 2, 1, 3], which neither
    _to_dim_order_copy nor ExecuTorch's elementwise kernels write, is left
    row-major instead of crashing the lowering ("Failed to map a given
    dim_order"); the program declares the order it computes, runs, and gives
    the row-major program's values."""
    x, y = image((1, 4, 6, 1), seed=6), image((1, 4, 6, 4), seed=7)
    x_cl = x.contiguous(memory_format=torch.channels_last)
    model = TokensPlusLinear().eval()
    with torch.no_grad():
        want = model(x_cl, y)
    assert row_major.dim_order(want) == (0, 2, 1, 3)
    cl, _, log = lower(TokensPlusLinear(), (x_cl, y))
    rm, _, _ = lower(TokensPlusLinear(), (x, y))
    report = cl.row_major_report
    assert [order for _, order in report.inputs] == [CL], log
    assert report.row_major_outputs == [(0, (0, 2, 1, 3))] and not report.outputs
    assert "left row-major" in log
    assert blobs(cl) == blobs(rm) and blobs(cl)
    et_cl, et_rm = emit(cl), emit(rm)
    assert io_orders(et_cl) == ([CL, (0, 1, 2, 3)], [(0, 1, 2, 3)])
    (tmp_path / "cl").mkdir()
    (tmp_path / "rm").mkdir()
    got_cl = run_host(runner, tmp_path / "cl", et_cl, (x, y), tuple(want.shape))
    got_rm = run_host(runner, tmp_path / "rm", et_rm, (x, y), tuple(want.shape))
    np.testing.assert_array_equal(bits(got_cl), bits(got_rm))


def test_a_dim_order_the_model_sets_stays_portable():
    lowered, part, _ = lower(InnerChannelsLast(), image())
    assert not lowered.row_major_report.inputs
    assert part.support.kinds.get("runtime tensor is not row-major")
    assert any("convolution" in t for t in portable(lowered))


@pytest.mark.parametrize("mode,source", [("reflect", "aten::reflection_pad2d"),
                                         ("replicate", "aten::replication_pad2d")])
def test_a_pad_decomposition_follows_the_input(runner, tmp_path, mode, source):
    """The pad's decomposition ends in a copy to the input's memory format,
    frozen at export as a _clone_dim_order to channels-last. After the
    input's copy it writes the default dim order, so the conv and its
    ReLU are claimed as for the row-major input, with the same blob, and the
    host runner gives the row-major program's bits (rather than both
    staying portable, refused as not row-major)."""
    x = image()
    cl, part, log = lower(PadConv(mode), x.contiguous(memory_format=torch.channels_last))
    rm, _, _ = lower(PadConv(mode), x)
    report = cl.row_major_report
    assert [order for _, order in report.inputs] == [CL], log
    assert "runtime tensor is not row-major" not in part.support.kinds
    left = portable(cl)
    assert not any("convolution" in t or "relu" in t for t in left), left
    assert blobs(cl) == blobs(rm) and blobs(cl)
    et_cl, et_rm = emit(cl), emit(rm)
    assert io_orders(et_cl) == ([CL], [CL])
    with torch.no_grad():
        shape = tuple(PadConv(mode).eval()(x).shape)
    (tmp_path / "cl").mkdir()
    (tmp_path / "rm").mkdir()
    got_cl = run_host(runner, tmp_path / "cl", et_cl, x, shape)
    got_rm = run_host(runner, tmp_path / "rm", et_rm, x, shape)
    np.testing.assert_array_equal(bits(got_cl), bits(got_rm))
    assert [op for _, op in getattr(report, "released", [])] == [source], log
    assert "writes the default dim order" in log


def test_a_copy_the_model_writes_stays():
    """A channels-last clone the model writes of its channels-last input is
    the model's own, kept: the conv after it stays portable, by name."""
    lowered, part, log = lower(CloneChannelsLast(),
                               image().contiguous(memory_format=torch.channels_last))
    report = lowered.row_major_report
    assert [order for _, order in report.inputs] == [CL], log
    assert not getattr(report, "released", []), log
    assert part.support.kinds.get("runtime tensor is not row-major")
    assert any("convolution" in t for t in portable(lowered))


def test_left_as_it_was_when_metadata_cannot_be_rederived(monkeypatch):
    def fail(graph, changed, mode):
        raise RuntimeError("no fake kernel")

    monkeypatch.setattr(row_major, "_repropagate", fail)
    lowered, part, log = lower(ConvChain(),
                               image().contiguous(memory_format=torch.channels_last))
    report = lowered.row_major_report
    assert not report.inputs and "no fake kernel" in report.skipped
    assert "left as they are" in log
    assert not any("_to_dim_order_copy" in t for t in portable(lowered))
    assert part.support.kinds.get("runtime tensor is not row-major")
    # a pad's released copy goes back to the channels-last order it had
    lowered, _, _ = lower(PadConv("reflect"),
                          image().contiguous(memory_format=torch.channels_last))
    assert not getattr(lowered.row_major_report, "released", [])
    clones = [n.kwargs.get("dim_order") for n in lowered.exported_program().graph.nodes
              if "_clone_dim_order" in str(n.target)]
    assert clones == [list(CL)], clones
