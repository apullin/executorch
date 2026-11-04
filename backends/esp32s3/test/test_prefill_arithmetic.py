# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prefill's portable attention tail, held to the chip's arithmetic.

The story's prefill leaves P@V and softmax on the fp32 overrides
(ops/espressif_ops.cpp shim_bmm_out, shim_softmax_out). The chip fuses the
bmm's multiply-adds (madd.s) and takes exp from newlib's libm. The host
runner, built with -ffp-contract=off, would round twice and call its own libm,
so the shim spells the fma out and the host's softmax runs a transcription of
newlib's expf; the prefill oracle (test/reference.py, used by
story/glimmer_decode.py) mirrors both instead of using np.exp, so the host
runner, the oracle and the chip compute the same bits. These tests run the overrides themselves on the
host runner against the oracle, bit for bit.
"""

import subprocess

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def runner():
    from executorch.backends.esp32s3.test.host import host_runner

    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


class Bmm(nn.Module):
    def forward(self, left, right):
        return torch.bmm(left, right)


class Softmax(nn.Module):
    def forward(self, scores):
        return torch.softmax(scores, dim=-1)


def run_portable(runner, model, inputs, directory):
    """Export without a partitioner (every op on the portable kernels and
    their Espressif overrides) and run it on the host runner."""
    from executorch.exir import to_edge

    program = to_edge(torch.export.export(model.eval(), inputs)).to_executorch()
    operators = {op.name for plan in program.executorch_program.execution_plan
                 for op in plan.operators}
    pte = directory / "model.pte"
    pte.write_bytes(program.buffer)
    command = [str(runner), "--pte", str(pte)]
    for index, value in enumerate(inputs):
        path = directory / f"input{index}.bin"
        np.ascontiguousarray(value.numpy()).tofile(path)
        command += ["--input", str(path)]
    output = directory / "output0.bin"
    output.unlink(missing_ok=True)
    command += ["--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    with torch.no_grad():
        like = model(*inputs)
    return np.fromfile(output, dtype=np.float32).reshape(like.shape), operators, result.stdout


def test_bmm_is_fused_like_the_chip(runner, tmp_path):
    from executorch.backends.esp32s3.test.reference import bmm_f32_portable

    # Three probabilities of fl(1/3) over V = [0.1, 0.2, 0.3].
    third = np.float32(1) / np.float32(3)
    left = torch.full((1, 1, 3), float(third))
    right = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.float32)
    got, operators, _ = run_portable(runner, Bmm(), (left, right), tmp_path)
    assert "aten::bmm" in operators, operators
    unfused = np.float32(0)
    for value in right.numpy().reshape(-1):
        unfused = np.float32(unfused + np.float32(third * value))
    assert hex(int(np.float32(unfused).view(np.uint32))) == "0x3e4cccce"
    assert hex(int(got.reshape(-1)[0].view(np.uint32))) == "0x3e4ccccd", got
    np.testing.assert_array_equal(got, bmm_f32_portable(left.numpy(), right.numpy()))

    # Both loops of the shim: four-column blocks and the column tail, a long k.
    generator = torch.Generator().manual_seed(5)
    left = torch.randn(2, 5, 97, generator=generator)
    right = torch.randn(2, 97, 7, generator=generator)
    (tmp_path / "random").mkdir()
    got, _, _ = run_portable(runner, Bmm(), (left, right), tmp_path / "random")
    np.testing.assert_array_equal(got, bmm_f32_portable(left.numpy(), right.numpy()))


def softmax_rows():
    """Rows whose exp arguments x - max cover newlib expf's branches: k = 0
    (|x| <= ln2/2), the near reduction (ln2/2 < |x| < 1.5 ln2), the general
    one, |x| < 2^-23, the subnormal results below -87.3, the underflow past
    -103.97, and masked lanes (-inf); plus scaled-score-like random rows."""
    generator = np.random.default_rng(9)
    width = 64
    sweep = -np.concatenate([
        np.linspace(0, 0.35, 200), np.linspace(0.34, 1.05, 200),
        np.linspace(1.0, 30, 400), np.linspace(30, 87.3, 200),
        np.linspace(87.3, 104.5, 200), np.array([1e-8, 6e-8, 1.2e-7]),
    ]).astype(np.float32)
    sweep = np.concatenate([sweep, np.zeros(-len(sweep) % (width - 1), np.float32)])
    rows = sweep.reshape(-1, width - 1)
    rows = np.concatenate([np.zeros((len(rows), 1), np.float32), rows], axis=1)
    masked = generator.standard_normal((4, width)).astype(np.float32)
    masked[:, 40:] = -np.inf
    scores = (generator.standard_normal((24, width)) * 3).astype(np.float32)
    return np.concatenate([rows, masked, scores]).astype(np.float32)


def test_softmax_takes_the_chips_expf(runner, tmp_path):
    from executorch.backends.esp32s3.test.reference import softmax_f32_portable

    scores = torch.from_numpy(softmax_rows())
    got, operators, _ = run_portable(runner, Softmax(), (scores,), tmp_path)
    assert "aten::_softmax" in operators, operators
    expected = softmax_f32_portable(scores.numpy())
    np.testing.assert_array_equal(got, expected)


def test_expf_mirror_is_newlibs_accuracy():
    # The transcription's sanity, independent of the host runner: fdlibm's
    # float exp is within 1 ulp of the correctly rounded value everywhere, and
    # its specials are newlib's. (Its exactness against the chip is the board's
    # prefill values bar: the chip runs newlib's own expf.)
    from executorch.backends.esp32s3.quantizer.mirrors import expf_newlib

    x = np.linspace(-110, 90, 400_001, dtype=np.float32)
    got = expf_newlib(x)
    with np.errstate(over="ignore"):
        exact = np.exp(x.astype(np.float64)).astype(np.float32)
    finite = np.isfinite(exact) & (exact > 0)
    distance = np.abs(got[finite].view(np.int32).astype(np.int64)
                      - exact[finite].view(np.int32).astype(np.int64))
    assert distance.max() <= 1, distance.max()
    assert (got[x > np.float32(88.72284)] == np.inf).all()
    assert (got[x < np.float32(-103.9721)] == 0).all()
    specials = expf_newlib(np.array([0.0, -0.0, np.inf, -np.inf, np.nan], np.float32))
    assert specials[0] == 1 and specials[1] == 1 and specials[2] == np.inf
    assert specials[3] == 0 and np.isnan(specials[4])


class Rope(nn.Module):
    """espressif::rope itself (what fuse_rope_pass leaves of the story's RoPE),
    with its tables as buffers."""

    def __init__(self, cos, sin, rot_dim):
        super().__init__()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.rot_dim = rot_dim

    def forward(self, x):
        return torch.ops.espressif.rope(x, self.cos, self.sin, self.rot_dim)


def run_custom(runner, model, inputs, directory):
    """Export a model that calls an espressif custom op directly (no
    partitioner, the op kept as is) and run it on the host runner."""
    from executorch.exir import EdgeCompileConfig, to_edge

    program = to_edge(torch.export.export(model.eval(), inputs),
                      compile_config=EdgeCompileConfig(_check_ir_validity=False)
                      ).to_executorch()
    operators = {op.name for plan in program.executorch_program.execution_plan
                 for op in plan.operators}
    pte = directory / "model.pte"
    pte.write_bytes(program.buffer)
    command = [str(runner), "--pte", str(pte)]
    for index, value in enumerate(inputs):
        path = directory / f"input{index}.bin"
        np.ascontiguousarray(value.numpy()).tofile(path)
        command += ["--input", str(path)]
    output = directory / "output0.bin"
    output.unlink(missing_ok=True)
    command += ["--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    with torch.no_grad():
        like = model(*inputs)
    return np.fromfile(output, dtype=np.float32).reshape(like.shape), operators


def test_rope_is_fused_like_the_chip(runner, tmp_path):
    # shim_rope on the chip is mul.s + msub.s and mul.s + madd.s, one rounding
    # per output; the host spells
    # that order with std::fma, and rope_f32_device mirrors it. Three table
    # layouts: the story's pairwise [T, Dh/2] tables (prefill rows), a decode
    # step's 1-D [Dh/2] tables, and HF-style full-width [T, rot] tables with a
    # partial rotary width (a pass-through tail).
    from executorch.backends.esp32s3._passes import fuse_rope_pass  # noqa: F401
    from executorch.backends.esp32s3.test.reference import rope_f32_device

    generator = torch.Generator().manual_seed(29)

    def tables(shape):
        angle = torch.rand(shape, generator=generator) * 6.3
        return torch.cos(angle), torch.sin(angle)

    cases = [
        ("pairwise prefill", torch.randn(2, 5, 32, generator=generator),
         tables((5, 16)), 32),
        ("decode step", torch.randn(8, 32, generator=generator), tables((16,)), 32),
        ("full-width partial", torch.randn(3, 4, 10, generator=generator),
         tables((4, 6)), 6),
    ]
    unfused_differs = 0
    for index, (name, x, (cos, sin), rot) in enumerate(cases):
        directory = tmp_path / f"case{index}"
        directory.mkdir()
        got, operators = run_custom(runner, Rope(cos, sin, rot), (x,), directory)
        assert "espressif::rope" in operators, (name, operators)
        want = rope_f32_device(x.numpy(), cos.numpy(), sin.numpy(), rot)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32),
                                      err_msg=name)
        with torch.no_grad():
            eager = torch.ops.espressif.rope(x, cos, sin, rot).numpy()
        unfused_differs += int((eager.view(np.uint32) != want.view(np.uint32)).sum())
    # The torch spelling (the op's Composite) rounds each product: the cases
    # must be able to tell the two apart.
    assert unfused_differs > 0


def test_softmax_mirror_takes_empty_tensors():
    # An empty reduction dimension, or no rows at all: nothing to reduce, an
    # empty result of the same shape, as the portable kernel the shim falls
    # back to returns (a reshape(-1, 0) would raise).
    from executorch.backends.esp32s3.test.reference import softmax_f32_portable

    for shape, dim in (((2, 0), -1), ((0, 3), -1), ((3, 0, 4), 1), ((0,), 0)):
        got = softmax_f32_portable(np.empty(shape, np.float32), dim=dim)
        assert got.shape == shape and got.dtype == np.float32, (shape, got.shape)
    # A non-empty tensor still reduces along `dim`, whichever axis it is.
    rows = np.array([[0.0, 1.0, -3.0], [2.0, 2.0, 0.5]], np.float32)
    np.testing.assert_array_equal(softmax_f32_portable(rows, dim=0),
                                  softmax_f32_portable(rows.T, dim=-1).T)
