# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A0 on finite rows too small to have a reciprocal.

A0, the backend's own per-row scheme, takes s = fl(absmax / 127) and codes
fl(x * fl(1 / s)) rounded half away. A finite row like [FLT_MIN, 0] has
s = 9.3e-41, fl(1 / s) = inf and fl(x * inv) = [inf, NaN], so the integer
conversion would be undefined (and differs by host). The rule, in every A0
quantiser and mirror (runtime/quant_rules.h a0_inverse_scale): a zero or
subnormal scale has inverse 0, so that row's codes are all 0. These tests
hold the delegate's q@kT and Linear rows, the int8 K-cache writer and the
mirrors to it, bit for bit, on the host runner.
"""

import struct
import subprocess

import numpy as np
import pytest
import torch
from torch import nn

FLT_MIN = np.float32(2.0 ** -126)
TINY = np.float32(np.finfo(np.float32).tiny)  # == FLT_MIN
SUBNORMAL = np.float32(1e-40)


@pytest.fixture(scope="module")
def runner():
    from executorch.backends.esp32s3.test.host import host_runner

    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def tiny_rows(width):
    """Rows at and around the cut, each followed by a normal one."""
    rows = []
    # The first row: amax FLT_MIN, so s is subnormal and fl(1 / s) = inf.
    row = np.zeros(width, np.float32)
    row[0] = FLT_MIN
    rows.append(row)
    # amax subnormal (s underflows further).
    row = np.zeros(width, np.float32)
    row[1] = -SUBNORMAL
    row[2] = SUBNORMAL / np.float32(2)
    rows.append(row)
    # The smallest amax whose scale is normal: s = FLT_MIN exactly.
    row = np.zeros(width, np.float32)
    row[0] = np.float32(127) * FLT_MIN
    row[3] = -FLT_MIN
    rows.append(row)
    # Just below it: s rounds to the largest subnormal.
    row = np.zeros(width, np.float32)
    row[0] = np.nextafter(np.float32(127) * FLT_MIN, np.float32(0))
    rows.append(row)
    return np.stack(rows)


def assert_bits_equal(got, want):
    """Every output's IEEE bits equal (the sign of a zero included)."""
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    assert got.shape == want.shape, (got.shape, want.shape)
    differ = got.view(np.uint32) != want.view(np.uint32)
    assert not differ.any(), (
        f"{int(differ.sum())} of {differ.size} outputs differ from the mirror; "
        f"first at {np.argwhere(differ)[0].tolist()}: got "
        f"{got[tuple(np.argwhere(differ)[0])]!r}, want {want[tuple(np.argwhere(differ)[0])]!r}")


def test_mirror_rule():
    from executorch.backends.esp32s3.quantizer import mirrors
    from executorch.backends.esp32s3.ops.kcache_i8 import quantize_rows_i8

    scales = np.array([0.0, SUBNORMAL, np.nextafter(FLT_MIN, np.float32(0)),
                       FLT_MIN, np.float32(1.0), np.float32(3e38)], np.float32)
    inverse = mirrors.a0_inverse(scales)
    expected = np.array([0.0, 0.0, 0.0, np.float32(2.0 ** 126), 1.0,
                         np.float32(1) / np.float32(3e38)], np.float32)
    np.testing.assert_array_equal(inverse, expected)
    assert np.isfinite(np.float32(1) / FLT_MIN)

    rows = tiny_rows(8)
    scale = (np.abs(rows).max(axis=1) / np.float32(127)).astype(np.float32)
    codes = mirrors.quantize_half_away(rows, scale[:, None])
    # Below the cut every code is 0; at it (s = FLT_MIN) the row quantises.
    np.testing.assert_array_equal(codes[[0, 1, 3]], 0)
    np.testing.assert_array_equal(codes[2], [127, 0, 0, -1, 0, 0, 0, 0])

    # The K-cache writer's Composite and the story's mirror take the same rule.
    torch_codes, torch_scales = quantize_rows_i8(torch.from_numpy(rows))
    np.testing.assert_array_equal(torch_codes.numpy(), codes)
    np.testing.assert_array_equal(torch_scales.numpy(), scale)

    from executorch.examples.espressif.esp32s3.story.glimmer_model import dynamic_i8_qk

    keys = torch.randn(3, 8)
    scores = dynamic_i8_qk(torch.from_numpy(rows), keys)
    assert torch.isfinite(scores).all(), scores
    assert (scores[[0, 1, 3]] == 0).all(), scores


class QK(nn.Module):
    def forward(self, query, key):
        scores = torch.bmm(query, key.transpose(1, 2))
        return scores, torch.softmax(scores, dim=-1)


class Rows(nn.Module):
    def forward(self, x):
        from executorch.backends.esp32s3.ops.kcache_i8 import QUANTIZE_ROWS_I8

        codes, scales = QUANTIZE_ROWS_I8(x)
        return codes, scales


def export(model, inputs, directory, partition=True, **options):
    from executorch.exir import ExecutorchBackendConfig, to_edge

    model.eval().requires_grad_(False)
    exported = torch.export.export(model, inputs)
    if partition:
        from executorch.backends.esp32s3._passes import lower_esp32
        from executorch.backends.esp32s3.partition.espressif_partitioner import (
            Esp32Partitioner,
        )

        edge = lower_esp32(exported, [Esp32Partitioner(**options)])
    else:
        edge = to_edge(exported)
    program = edge.to_executorch(ExecutorchBackendConfig(delegate_alignment=16))
    pte = directory / "model.pte"
    pte.write_bytes(program.buffer)
    return pte, edge


def run(runner, pte, inputs, like):
    command = [str(runner), "--pte", str(pte)]
    for index, value in enumerate(inputs):
        path = pte.parent / f"input{index}.bin"
        np.ascontiguousarray(value).tofile(path)
        command += ["--input", str(path)]
    paths = [pte.parent / f"output{index}.bin" for index in range(len(like))]
    for path in paths:
        path.unlink(missing_ok=True)
        command += ["--output", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return [np.fromfile(path, dtype=ref.dtype).reshape(ref.shape)
            for path, ref in zip(paths, like)], result.stdout


def test_bmm_dyn_tiny_rows(runner, tmp_path):
    # OP_BMM_DYN_I8: tiny query rows score exactly 0 and a tiny key row gives
    # exact zeros in its column, bit for bit as bmm_qk_i8.
    from executorch.backends.esp32s3.test.reference import bmm_qk_i8

    torch.manual_seed(7)
    width = 17
    query = np.concatenate([tiny_rows(width), np.random.default_rng(1).standard_normal(
        (2, width)).astype(np.float32)]).reshape(1, -1, width).copy()
    key = np.random.default_rng(2).standard_normal((1, 5, width)).astype(np.float32)
    key[0, 1] = tiny_rows(width)[0]
    key[0, 3] = tiny_rows(width)[2]
    pte, _ = export(QK(), (torch.from_numpy(query), torch.from_numpy(key)),
                    tmp_path, quantize_qk=True)
    expected = bmm_qk_i8(query[0], key[0])[None]
    (scores, _), log = run(runner, pte, (query, key),
                           (expected, expected))
    assert "esp32 bmmqk: 1 dynamic int8 bmm" in log, log
    np.testing.assert_array_equal(scores, expected)
    assert (scores[0, [0, 1, 3]] == 0).all(), scores[0, :4]
    assert (scores[0, :, 1] == 0).all(), scores[0, :, 1]
    assert np.isfinite(scores).all()


def test_linear_tiny_rows(runner, tmp_path):
    # An A0 Linear without bias: a tiny row's output is exactly 0, as the
    # mirror gives it (not ~1e-40, from a code of 127).
    from executorch.backends.esp32s3.test.reference import addmm_i8_absmax

    torch.manual_seed(11)
    model = nn.Linear(16, 8, bias=False)
    x = np.concatenate([tiny_rows(16), np.random.default_rng(3).standard_normal(
        (2, 16)).astype(np.float32)])
    pte, _ = export(model, (torch.from_numpy(x),), tmp_path, format_policy="i8")
    expected = addmm_i8_absmax(x, model.weight.detach().numpy(),
                               np.zeros(8, np.float32))
    expected = np.asarray(expected, dtype=np.float32)
    (y,), _ = run(runner, pte, (x,), (expected,))
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(y[[0, 1, 3]], 0)


def test_kcache_writer_tiny_rows(runner, tmp_path):
    # espressif::quantize_rows_i8.out, the C++ K-cache writer: zero codes for
    # the rows below the cut, the Composite's codes and scales everywhere.
    from executorch.backends.esp32s3.ops.kcache_i8 import quantize_rows_i8

    x = np.concatenate([tiny_rows(16), np.random.default_rng(4).standard_normal(
        (2, 16)).astype(np.float32)])
    pte, _ = export(Rows(), (torch.from_numpy(x),), tmp_path, partition=False)
    codes, scales = quantize_rows_i8(torch.from_numpy(x))
    (got_codes, got_scales), _ = run(
        runner, pte, (x,), (codes.numpy(), scales.numpy()))
    np.testing.assert_array_equal(got_codes, codes.numpy())
    np.testing.assert_array_equal(got_scales, scales.numpy())
    np.testing.assert_array_equal(got_codes[[0, 1, 3]], 0)


# The cut is written at eight sites:
# quantize_row_t (above), the K-cache writer (above), and these, each fed a
# row whose absmax is FLT_MIN, so A0's scale is subnormal.

def _blob_opcodes(edge):
    from executorch.exir.backend.utils import get_delegated_payload

    from executorch.backends.esp32s3.serialization.espressif_blob import (
        BLOB_HEADER_BYTES,
        BLOB_HEADER_FORMAT,
        OP_DESC_BYTES,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
    )

    opcodes = []
    for payload in get_delegated_payload(edge.exported_program().graph_module).values():
        blob = payload[2]
        _, _, count, _ = struct.unpack_from(BLOB_HEADER_FORMAT, blob)
        for i in range(count):
            desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
                OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES + i * OP_DESC_BYTES)))
            opcodes.append(desc["opcode"])
    return opcodes


def _tiny_patch_image():
    # Output (0, 0)'s 3x3 patch (padding 1) reads input rows and columns 0-1
    # of both channels: [FLT_MIN, 0, 0, 0] in channel 0, zeros in channel 1,
    # so its absmax is FLT_MIN.
    torch.manual_seed(3)
    x = torch.randn(1, 2, 5, 5)
    x[0, :, 0:2, 0:2] = 0.0
    x[0, 0, 0, 0] = float(FLT_MIN)
    return x


@pytest.mark.parametrize("kind", ["dense", "depthwise"])
def test_conv_tiny_patch(runner, tmp_path, kind):
    # quantize_conv_row_t (a dense conv's patch rows) and quantize_taps_i8 (a
    # depthwise conv's taps): output (0, 0) reads zero codes and is exactly 0,
    # bit for bit as conv_iq_absmax / depthwise_iq_absmax.
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import (
        conv_iq_absmax,
        depthwise_iq_absmax,
    )

    x = _tiny_patch_image()
    if kind == "dense":
        model, mirror = nn.Conv2d(2, 3, 3, 1, 1, bias=False), conv_iq_absmax
    else:
        model, mirror = (nn.Conv2d(2, 2, 3, 1, 1, groups=2, bias=False),
                         depthwise_iq_absmax)
    model.eval().requires_grad_(False)
    gm = to_edge(torch.export.export(model, (x,))).exported_program().graph_module
    conv = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(conv)
    pte, edge = export(model, (x,), tmp_path, format_policy="i8")
    targets = [str(n.target) for n in edge.exported_program().graph_module.graph.nodes
               if n.op == "call_function"]
    assert not any("convolution" in t for t in targets), targets
    want = mirror(x, model.weight, None, geom, fmt="i8").numpy()
    (got,), _ = run(runner, pte, (x.numpy(),), (want,))
    assert_bits_equal(got, want)
    first = got.reshape(1, geom.c_out, geom.h_out, geom.w_out)[0, :, 0, 0]
    np.testing.assert_array_equal(first, 0)


class AffineKCacheQK(nn.Module):
    """q @ k^T against the stock affine int8 key cache (OP_BMM_KCACHE_AFFINE_I8:
    esp32_asym.cpp kcache_query_row quantises each query row with A0)."""

    def forward(self, query, codes, scales, zero_points):
        from executorch.backends.esp32s3._passes import stock_llm  # noqa: F401

        keys = torch.ops.espressif.dequantize_per_token_i8(codes, scales, zero_points)
        scores = torch.bmm(query, keys.transpose(1, 2))
        return scores, torch.softmax(scores, dim=-1)


def test_affine_kcache_tiny_query_rows(runner, tmp_path):
    # kcache_query_row: a query row below the cut scores exactly 0 against every
    # key, bit for bit as mirrors.kcache_scores.
    from executorch.backends.esp32s3.quantizer import mirrors
    from executorch.backends.esp32s3.serialization.espressif_blob import (
        OP_BMM_KCACHE_AFFINE_I8,
    )

    width = 16
    generator = np.random.default_rng(6)
    query = np.concatenate([tiny_rows(width), generator.standard_normal(
        (2, width)).astype(np.float32)]).reshape(1, -1, width).copy()
    codes = generator.integers(-128, 128, (1, 5, width)).astype(np.int8)
    scales = (generator.random((1, 5, 1)) * 0.02 + 0.001).astype(np.float32)
    zero_points = generator.integers(-30, 30, (1, 5, 1)).astype(np.int8)
    inputs = (query, codes, scales, zero_points)
    # The affine K cache's int8 query is opt-in, like every int8 q@kT
    # (quantize_qk): without it the bmm stays portable.
    pte, edge = export(AffineKCacheQK(), tuple(torch.from_numpy(t) for t in inputs),
                       tmp_path, quantize_qk=True)
    assert OP_BMM_KCACHE_AFFINE_I8 in _blob_opcodes(edge)
    want = mirrors.kcache_scores(query[0], codes[0], scales[0, :, 0],
                                 zero_points[0, :, 0])[None]
    (got, _), _ = run(runner, pte, inputs, (want, want))
    assert_bits_equal(got, want)
    np.testing.assert_array_equal(got[0, [0, 1, 3]], 0)


class AffineLinear(nn.Module):
    """A PT2E per-channel int8 weight with zero points and no bias, fp32 in:
    an affine op with no activation record, so esp32_asym.cpp quantize_row_a0
    quantises each input row with A0."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.register_buffer("codes", torch.randint(-128, 128, (8, 16), generator=g,
                                                    dtype=torch.int8))
        self.register_buffer("scales", torch.rand(8, generator=g) * 2e-2 + 1e-3)
        self.register_buffer("zero_points", torch.randint(-30, 30, (8,),
                                                          generator=g))

    def forward(self, x):
        from executorch.backends.esp32s3.test.fixtures.asym import _decomposed

        weight = _decomposed().dequantize_per_channel(
            self.codes, self.scales, self.zero_points, 0, -128, 127, torch.int8)
        return torch.nn.functional.linear(x, weight)


def test_affine_a0_tiny_rows(runner, tmp_path):
    # quantize_row_a0: an input row below the cut gives zero codes and outputs
    # exactly 0, bit for bit as mirrors.affine_matmul with A0 activations
    # (quantize_half_away).
    from executorch.backends.esp32s3.quantizer import mirrors
    from executorch.backends.esp32s3.test.fixtures import asym

    model = AffineLinear().eval()
    x = np.concatenate([tiny_rows(16), np.random.default_rng(7).standard_normal(
        (2, 16)).astype(np.float32)])
    program, log = asym.lower(asym.Fixture("a0_tiny_rows", model,
                                           (torch.from_numpy(x),), None, {}))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    want = mirrors.affine_matmul(
        x, model.codes.numpy(), model.scales.numpy().astype(np.float32), None,
        act=mirrors.A0, bits=8,
        zero_points=model.zero_points.numpy().astype(np.int64)).y
    (got,), _ = run(runner, pte, (x,), (want,))
    assert_bits_equal(got, want)
    np.testing.assert_array_equal(got[[0, 1, 3]], 0)
