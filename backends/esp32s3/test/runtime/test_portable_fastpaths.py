# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The standard-op overrides' fast paths against ExecuTorch's portable kernels.

espressif_ops.cpp gives a few portable aten ops a fast path for the story
decode's exact cases and leaves every other input to the portable kernel. A
fast path must give that kernel's bytes for every input it takes.
`esp32_host_runner --selftest` runs the op library's conformance cases: the
override and the portable kernel on the same hand-built tensors (the decode's
shapes, IEEE special values, near misses that must fall back), comparing every
written byte and the failure state, and checking the path each case took.
QEMU runs the same cases on the chip's build (run_parity.py). The program
tests here run small exported graphs through the registered overrides on the
host and compare them with eager PyTorch, which these exact ops match.

Build the runner first (test/host_runner/README.md).
ESP32_HOST_RUNNER can override the default cmake-out-espressif/esp32_host_runner path.
"""

import copy
import re
import subprocess

import numpy as np
import pytest
import torch
from torch import nn

from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.backends.esp32s3.test.host import host_runner

# Every operator the conformance cases cover; each must have fast and
# fallback cases, so a refused story shape or an accepted near miss shows.
CHECKED_OPS = ("mul.out", "le.Tensor_out", "index_put_", "split_with_sizes_copy.out",
               "remainder.Scalar_out", "quantize_per_tensor.out",
               "dequantize_per_tensor.out", "avg_pool2d.out",
               # The element-wise loops (ops/elementwise.h), which the host
               # registers as a firmware built --elementwise-fast does.
               "mul.out-channels", "constant_pad_nd.out")


@pytest.fixture(scope="session")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def test_conformance_cases(runner):
    result = subprocess.run([str(runner), "--selftest"], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    log = result.stdout
    assert "MISMATCH" not in log, log
    per_op = {name: tuple(map(int, counts)) for name, *counts in re.findall(
        r"^fastpath check: (\S+) (\d+) case\(s\), (\d+) fast, (\d+) fallback, "
        r"(\d+) mismatch\(es\)$", log, re.M)}
    assert set(per_op) == set(CHECKED_OPS), log
    for name, (cases, fast, fallback, mismatches) in per_op.items():
        assert mismatches == 0 and fast > 0 and fallback > 0, (name, log)
        assert cases == fast + fallback, (name, log)
    total = re.search(r"^fastpath check: (\d+) case\(s\), (\d+) mismatch\(es\)$", log, re.M)
    assert total, log
    assert int(total[1]) == sum(counts[0] for counts in per_op.values())
    assert total[2] == "0", log


def test_fast_layer_norm_rows(runner):
    # The opt-in fast LayerNorm (ESP32_FAST_LAYERNORM) is registered only in
    # its own firmware variant, but its row function is in every build, and
    # --selftest holds it to the portable kernel's: the mean and rstd bits
    # exactly, each output within the reciprocal multiply's rounding.
    result = subprocess.run([str(runner), "--selftest"], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    line = re.search(r"^fast_layer_norm check: (\d+) case\(s\), (\d+) mismatch\(es\); "
                     r"(\d+) rows, widest output gap ([0-9.]+) of its bound$",
                     result.stdout, re.M)
    assert line, result.stdout
    assert int(line[1]) > 0 and int(line[3]) > 0, line[0]
    assert line[2] == "0", result.stdout
    assert float(line[4]) <= 1.0, line[0]


def dispatch(log, name):
    """The counters of one "<name> dispatch:" line, by label."""
    match = re.search(rf"^{re.escape(name)} dispatch: (.*)$", log, re.M)
    assert match, log
    return {label: int(count) for label, count in
            (field.rsplit(" ", 1) for field in match[1].split(", "))}


def run_program(runner, module, inputs, directory, reinplace=False):
    """Export `module` without a delegate, run it once on the host runner and
    return (outputs, eager outputs, log). Inputs and outputs are numpy arrays;
    `reinplace` lowers as the story does, buffer writes in place."""
    module.eval()
    assert sum(b.numel() for b in module.buffers()) <= 4096
    reference = copy.deepcopy(module)  # eager runs on untouched buffers
    exported = torch.export.export(module, tuple(torch.from_numpy(x) for x in inputs))
    program = to_edge(exported).to_executorch(
        ExecutorchBackendConfig(run_reinplace_pass=reinplace))
    pte = directory / "program.pte"
    pte.write_bytes(program.buffer)
    command = [str(runner), "--pte", str(pte)]
    for index, value in enumerate(inputs):
        path = directory / f"input{index}.bin"
        np.ascontiguousarray(value).tofile(path)
        command += ["--input", str(path)]
    with torch.no_grad():
        expected = reference(*(torch.from_numpy(x.copy()) for x in inputs))
    expected = [e.numpy() for e in (expected if isinstance(expected, tuple) else (expected,))]
    for index in range(len(expected)):
        command += ["--output", str(directory / f"output{index}.bin")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = [np.fromfile(directory / f"output{index}.bin", dtype=e.dtype).reshape(e.shape)
               for index, e in enumerate(expected)]
    return outputs, expected, result.stdout


def special_floats(shape, seed):
    """fp32 values with every IEEE special among random ones."""
    generator = np.random.default_rng(seed)
    bits = generator.integers(0, 2**32, size=int(np.prod(shape)), dtype=np.uint64).astype(np.uint32)
    values = bits.view(np.float32).copy()
    normal = generator.standard_normal(values.size).astype(np.float32)
    values[values.size // 2:] = normal[values.size // 2:]
    specials = np.array([0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000,
                         0x00000001, 0x807fffff, 0x7f7fffff], dtype=np.uint32).view(np.float32)
    values[:specials.size] = specials
    return values.reshape(shape)


class QueryScale(nn.Module):
    """The decode's per-layer query scale: fp32 times a 0-d fp32 constant."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.17677669529663687, dtype=torch.float32))

    def forward(self, query):
        return query * self.scale


def test_mul_by_zero_dim_constant(runner, tmp_path):
    query = special_floats((8, 32), seed=11)
    outputs, expected, log = run_program(runner, QueryScale(), [query], tmp_path)
    # NaN bits are not compared: eager may give another payload. Everything
    # else is one IEEE multiply, the same bits everywhere.
    np.testing.assert_array_equal(outputs[0], expected[0])
    finite = ~np.isnan(expected[0])
    assert np.array_equal(outputs[0][finite].view(np.uint32),
                          expected[0][finite].view(np.uint32))
    counts = dispatch(log, "mul.out")
    assert counts["scalar"] == 1 and counts["fallback"] == 0, log


class WindowMask(nn.Module):
    """The decode's attention mask: a position table against the 0-d position."""

    def __init__(self):
        super().__init__()
        self.register_buffer("positions", torch.arange(64))

    def forward(self, position):
        return self.positions <= position


@pytest.mark.parametrize("position", [0, 37, 63, -5, 2**62])
def test_le_against_zero_dim_position(runner, tmp_path, position):
    outputs, expected, log = run_program(
        runner, WindowMask(), [np.asarray(position, dtype=np.int64)], tmp_path)
    assert outputs[0].dtype == np.bool_
    np.testing.assert_array_equal(outputs[0], expected[0])
    counts = dispatch(log, "le.Tensor_out")
    assert counts["scalar"] == 1 and counts["fallback"] == 0, log


class CacheWrite(nn.Module):
    """The decode's KV write: one row into a persistent cache at a position,
    which the reinplace pass turns into aten::index_put_ on the buffer. The
    cache starts at zero, as a program's planned mutable buffers do."""

    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 16, 4, dtype=dtype))

    def forward(self, row, position):
        self.cache.index_copy_(1, position.reshape(1), row)
        return self.cache * 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
@pytest.mark.parametrize("position", [0, 9, 15])
def test_index_put_cache_row(runner, tmp_path, dtype, position):
    row = (np.arange(4).reshape(1, 1, 4) * 7 - 11).astype(torch.empty(0, dtype=dtype).numpy().dtype)
    outputs, expected, log = run_program(
        runner, CacheWrite(dtype), [row, np.asarray(position, dtype=np.int64)], tmp_path,
        reinplace=True)
    np.testing.assert_array_equal(outputs[0], expected[0])
    assert outputs[0].dtype == expected[0].dtype
    counts = dispatch(log, "index_put_")
    assert counts["row-copy"] == 1 and counts["fallback"] == 0, log


class WindowWrite(nn.Module):
    """A Mistral-style sliding-window cache write: one position of a
    [1, heads, window, dim] cache, which may count from the end, written with
    a [heads, 1, dim] row that leaves out the leading 1. Transformers' decode
    exports this; ExecuTorch's portable kernel takes it since #23529."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 6, 8))

    def forward(self, row, position):
        self.cache[:, :, position.reshape(1)] = row
        return self.cache * 1


@pytest.mark.parametrize("position", [-1, -6, 0, 5])
def test_index_put_window_row_from_the_end(runner, tmp_path, position):
    row = (np.arange(16, dtype=np.float32).reshape(2, 1, 8) * 0.75 - 3.0)
    outputs, expected, log = run_program(
        runner, WindowWrite(), [row, np.asarray(position, dtype=np.int64)], tmp_path,
        reinplace=True)
    np.testing.assert_array_equal(outputs[0], expected[0])
    counts = dispatch(log, "index_put_")
    assert counts["row-copy"] == 1 and counts["fallback"] == 0, log


class EmbeddingSplit(nn.Module):
    """The decode's per-layer embeddings: one fp32 [1,8,D] split into its rows."""

    def forward(self, embeddings):
        return torch.split(embeddings, 1, dim=1)


def test_split_into_rows(runner, tmp_path):
    embeddings = special_floats((1, 8, 32), seed=12)
    outputs, expected, log = run_program(runner, EmbeddingSplit(), [embeddings], tmp_path)
    assert len(outputs) == 8
    for output, reference in zip(outputs, expected):  # copies: every bit, NaNs too
        assert np.array_equal(output.view(np.uint32), reference.view(np.uint32))
    counts = dispatch(log, "split_with_sizes_copy.out")
    assert counts["chunk-copy"] == 1 and counts["fallback"] == 0, log


class WindowSlot(nn.Module):
    """The decode's sliding-window slot: the 0-d position modulo the window."""

    def forward(self, position):
        return position % 64


@pytest.mark.parametrize("position", [0, 63, 70, 255, 2**62 + 5])
def test_remainder_of_position(runner, tmp_path, position):
    outputs, expected, log = run_program(
        runner, WindowSlot(), [np.asarray(position, dtype=np.int64)], tmp_path)
    np.testing.assert_array_equal(outputs[0], expected[0])
    counts = dispatch(log, "remainder.Scalar_out")
    assert counts["non-negative"] == 1 and counts["fallback"] == 0, log


def test_remainder_of_negative_position_stays_portable(runner, tmp_path):
    # ExecuTorch's portable integer remainder is C++'s truncating %, not
    # torch.remainder's floor-mod: -3 % 64 gives -3 there, 61 in eager. The
    # fast path only takes inputs where the two agree; this one stays portable.
    outputs, expected, log = run_program(
        runner, WindowSlot(), [np.asarray(-3, dtype=np.int64)], tmp_path)
    assert int(expected[0]) == 61 and int(outputs[0]) == -3
    counts = dispatch(log, "remainder.Scalar_out")
    assert counts["non-negative"] == 0 and counts["fallback"] == 1, log


class QuantizedTail(nn.Module):
    """What a static int8 CNN leaves between its last two delegates: the
    activation's quantize/dequantize pair, then the global average pool."""

    def forward(self, x):
        codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, 0.0235, -3, -128, 127, torch.int8)
        values = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            codes, 0.0235, -3, -128, 127, torch.int8)
        return codes, values, nn.functional.avg_pool2d(values, kernel_size=(5, 5))


def test_quantized_tail(runner, tmp_path):
    import executorch.kernels.quantized  # noqa: F401  (the q/dq .out variants)

    # Every value finite (eager's NaN code is platform-defined); some far past
    # the range, so codes land on both rails. Codes and dequantized values
    # match eager torch; the pool sums in ExecuTorch's portable order, which
    # eager does not promise, so it is checked against that order in float32.
    x = np.random.default_rng(5).standard_normal((1, 8, 5, 5)).astype(np.float32) * 2
    x.flat[:4] = [1e6, -1e6, 0.0235 * 2.5, -0.0235 * 0.5]
    outputs, expected, log = run_program(runner, QuantizedTail(), [x], tmp_path)
    np.testing.assert_array_equal(outputs[0], expected[0])
    assert outputs[0].max() == 127 and outputs[0].min() == -128
    np.testing.assert_array_equal(outputs[1].view(np.uint32), expected[1].view(np.uint32))
    planes = outputs[1].reshape(8, 25)
    reference = np.empty(8, dtype=np.float32)
    for plane in range(8):
        accum = planes[plane, 0]
        for value in planes[plane, 1:]:
            accum = np.float32(value + accum)
        reference[plane] = np.float32(accum / np.float32(25))
    assert np.array_equal(outputs[2].reshape(8).view(np.uint32), reference.view(np.uint32))
    assert dispatch(log, "quantize_per_tensor.out") == {"fp32": 1, "fallback": 0}, log
    assert dispatch(log, "dequantize_per_tensor.out") == {"to-fp32": 1, "fallback": 0}, log
    assert dispatch(log, "avg_pool2d.out") == {"global": 1, "windowed": 0, "fallback": 0}, log
