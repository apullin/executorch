# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host gates for the mutable cached-attention custom op.

The eager checks run the CompositeExplicitAutograd oracle. The host-runner
checks run the C++ kernel (espressif_ops.cpp shim_cached_attention) itself,
through esp32_host_runner's --script mode, which keeps one loaded method and
so its two cache buffers across steps, as the firmware's decode loop does.
"""

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from executorch.backends.esp32s3.ops.cached_attention import CACHED_ATTENTION


def reference(q, k, v, kcache, vcache, pos):
    index = pos.reshape(1)
    kcache.index_copy_(1, index, k.unsqueeze(1))
    vcache.index_copy_(1, index, v.unsqueeze(1))
    scores = torch.matmul(
        q.unsqueeze(1), kcache.transpose(1, 2)).squeeze(1)
    scores = scores / math.sqrt(q.shape[-1])
    valid = torch.arange(kcache.shape[1]) <= pos
    scores = torch.where(valid.unsqueeze(0), scores, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(
        probabilities.unsqueeze(1), vcache).squeeze(1)


class StatefulAttention(torch.nn.Module):
    def __init__(self, heads=4, context=16, width=24):
        super().__init__()
        self.register_buffer("kcache", torch.zeros(heads, context, width))
        self.register_buffer("vcache", torch.zeros(heads, context, width))

    def forward(self, q, k, v, pos):
        return CACHED_ATTENTION(
            q, k, v, self.kcache, self.vcache, pos,
            float(math.sqrt(q.shape[-1])))


def check_eager_numerics():
    torch.manual_seed(17)
    model = StatefulAttention()
    reference_k = torch.zeros_like(model.kcache)
    reference_v = torch.zeros_like(model.vcache)
    with torch.no_grad():
        for position in range(8):
            q, k, v = (torch.randn(4, 24) for _ in range(3))
            pos = torch.tensor(position, dtype=torch.long)
            expected = reference(
                q, k, v, reference_k, reference_v, pos)
            actual = model(q, k, v, pos)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(model.kcache, reference_k)
    assert torch.equal(model.vcache, reference_v)
    print("PASS eager numerics and recurrent cache mutation")


def export_program(heads=4, context=16, width=24):
    fresh = StatefulAttention(heads, context, width)
    args = (torch.randn(heads, width), torch.randn(heads, width),
            torch.randn(heads, width), torch.tensor(0, dtype=torch.long))
    exported = torch.export.export(fresh, args)
    from executorch.exir import to_edge

    edge = to_edge(exported)
    mutated = set(edge.exported_program().graph_signature.buffers_to_mutate.values())
    assert mutated == {"kcache", "vcache"}, mutated
    return edge.to_executorch()


def check_one_emitted_op():
    program = export_program()
    operators = {
        operator.name + (f".{operator.overload}" if operator.overload else "")
        for plan in program.executorch_program.execution_plan
        for operator in plan.operators
    }
    assert operators == {"espressif::cached_attention.out"}, operators
    print("PASS one emitted op, two buffer mutations, no materialized cache graph")


def run_kernel(runner, directory, steps, heads, context, width):
    """Run the C++ kernel over `steps` ((q, k, v, pos) arrays) in ONE runner
    process, so the cache rows persist between steps; return each step's
    attention output [H, Dh]."""
    program = export_program(heads, context, width)
    pte = directory / "cached_attention.pte"
    pte.write_bytes(program.buffer)
    # The two buffer mutations come first, the user output last.
    count = len(program.executorch_program.execution_plan[0].outputs)
    lines = []
    for index, (q, k, v, pos) in enumerate(steps):
        inputs = []
        for name, value in (("q", q), ("k", k), ("v", v)):
            path = directory / f"{name}{index}.bin"
            np.ascontiguousarray(value, dtype=np.float32).tofile(path)
            inputs.append(str(path))
        path = directory / f"pos{index}.bin"
        np.asarray(pos, dtype=np.int64).tofile(path)
        inputs.append(str(path))
        outputs = [str(directory / f"out{index}_{j}.bin") for j in range(count)]
        for name in outputs:
            Path(name).unlink(missing_ok=True)
        lines.append(f"forward {' '.join(inputs)} -> {' '.join(outputs)}")
    script = directory / "steps.txt"
    script.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        [str(runner), "--pte", str(pte), "--script", str(script)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    results = []
    for index in range(len(steps)):
        out = np.fromfile(directory / f"out{index}_{count - 1}.bin", dtype=np.float32)
        assert out.size == heads * width, out.size
        results.append(out.reshape(heads, width))
    return results


def kernel_mirror(q, kcache, vcache, pos, scale_denominator):
    """shim_cached_attention's arithmetic, but numpy's float32 exp for libm's
    expf: scores by fma in lane order, one reciprocal of the denominator, the
    weights normalised, then P@V by fma in token order."""
    from executorch.backends.esp32s3.quantizer.mirrors import fma_f32

    heads, width = q.shape
    out = np.zeros((heads, width), np.float32)
    for head in range(heads):
        scores = np.zeros(pos + 1, np.float32)
        for token in range(pos + 1):
            dot = np.float32(0)
            for lane in range(width):
                dot = fma_f32(q[head, lane], kcache[head, token, lane], dot)
            scores[token] = np.float32(dot) / np.float32(scale_denominator)
        weights = np.exp(scores - scores.max()).astype(np.float32)
        denominator = np.float32(0)
        for weight in weights:
            denominator = np.float32(denominator + weight)
        weights = (weights * (np.float32(1) / denominator)).astype(np.float32)
        for lane in range(width):
            acc = np.float32(0)
            for token in range(pos + 1):
                acc = fma_f32(weights[token], vcache[head, token, lane], acc)
            out[head, lane] = acc
    return out


def check_kernel_normalises_before_pv(runner, directory):
    # One head, width 1, context 3. Q and every key are zero, so
    # each step's weights are exactly 1 and the denominator is the step count;
    # every V row is FLT_MAX/2. Softmax then P@V stays finite (FLT_MAX/2 at
    # positions 0 and 1, fl(1/3)-weighted at 2). Dividing an unnormalised sum
    # instead overflowed at position 2: FLT_MAX + FLT_MAX/2 = inf.
    half_max = np.float32(np.finfo(np.float32).max / np.float32(2))
    zero = np.zeros((1, 1), np.float32)
    value = np.full((1, 1), half_max, np.float32)
    steps = [(zero, zero, value, pos) for pos in range(3)]
    got = run_kernel(runner, directory, steps, heads=1, context=3, width=1)
    kcache = np.zeros((1, 3, 1), np.float32)
    vcache = np.full((1, 3, 1), half_max, np.float32)
    for pos, actual in enumerate(got):
        expected = kernel_mirror(zero, kcache, vcache, pos, 1.0)
        assert np.isfinite(actual).all(), (pos, actual)
        np.testing.assert_array_equal(actual, expected)
    # And the Composite (softmax, then matmul) agrees.
    model = StatefulAttention(1, 3, 1)
    with torch.no_grad():
        for pos, actual in enumerate(got):
            eager = CACHED_ATTENTION(
                torch.from_numpy(zero), torch.from_numpy(zero),
                torch.from_numpy(value), model.kcache, model.vcache,
                torch.tensor(pos), 1.0).numpy()
            np.testing.assert_allclose(actual, eager, rtol=1e-6, atol=0)
    print("PASS kernel normalises before P@V (V rows of FLT_MAX/2 stay finite)")


def check_kernel_matches_composite(runner, directory):
    # Random recurrent decode, 4 heads, 8 positions of a 16-row cache. The
    # kernel and the Composite differ only in libm's expf and torch's
    # reduction order; the mirror differs only in expf.
    torch.manual_seed(23)
    heads, context, width = 4, 16, 24
    steps = []
    model = StatefulAttention(heads, context, width)
    expected = []
    with torch.no_grad():
        for pos in range(8):
            q, k, v = (torch.randn(heads, width) for _ in range(3))
            expected.append(model(q, k, v, torch.tensor(pos)).numpy())
            steps.append((q.numpy(), k.numpy(), v.numpy(), pos))
    got = run_kernel(runner, directory, steps, heads, context, width)
    kcache = model.kcache.numpy()
    vcache = model.vcache.numpy()
    for pos, (actual, eager) in enumerate(zip(got, expected)):
        np.testing.assert_allclose(actual, eager, rtol=1e-5, atol=1e-6)
        mirror = kernel_mirror(steps[pos][0], kcache, vcache, pos,
                               math.sqrt(width))
        np.testing.assert_allclose(actual, mirror, rtol=2e-6, atol=2e-7)
    print("PASS kernel matches the Composite over 8 recurrent steps")


def test_eager_numerics():
    check_eager_numerics()


def test_one_emitted_op():
    check_one_emitted_op()


@pytest.fixture(scope="module")
def runner():
    from executorch.backends.esp32s3.test.host import host_runner

    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def test_kernel_normalises_before_pv(runner, tmp_path):
    check_kernel_normalises_before_pv(runner, tmp_path)


def test_kernel_matches_composite(runner, tmp_path):
    check_kernel_matches_composite(runner, tmp_path)


def main():
    check_eager_numerics()
    check_one_emitted_op()
    from executorch.backends.esp32s3.test.host import host_runner

    runner = host_runner()
    if not runner.is_file():
        print(f"SKIP host-runner checks: {runner} not built")
        return
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        check_kernel_normalises_before_pv(runner.resolve(), Path(first))
        check_kernel_matches_composite(runner.resolve(), Path(second))


if __name__ == "__main__":
    main()
