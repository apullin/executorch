# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static int8 graphs (Stage A): what the export does with a stock static
recipe's quantize/dequantize pairs, and the host runner's bits.

static_qdq.py describes the scheme. Here, on small graphs spelled the way
convert_pt2e leaves them (quantize_per_tensor.default / dequantize_per_tensor
.default with float32 constants, weights as dequantize_per_channel over int8
codes):

  - an input pair in front of a claimed Linear is absorbed: the pair leaves
    the graph, the delegate reads the tensor before it and carries the static
    EXT_ACT_QUANT, and its output is mirrors.affine_matmul's bit for bit;
  - a pair a portable op also reads stays whole for it, while the
    Linear reads the tensor before the pair -- never a quantize whose
    dequantise was removed;
  - what the records cannot carry is refused by name and the op stays
    portable on the dequantised value: a scale that is not a float32 value, a
    per-channel activation quantiser. (int16 codes: test_asym_a16.py.)

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
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3._passes import static_qdq
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
qd = torch.ops.quantized_decomposed


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def f32(value):
    return float(F32(value))


class Linear8(nn.Module):
    """An int8 per-channel Linear as convert_pt2e leaves it: codes, scales and
    zero points as buffers, dequantised in forward."""

    def __init__(self, k, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("c", torch.randint(-127, 128, (n, k), generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(n, generator=g) * 2e-3 + 1e-4)
        self.register_buffer("z", torch.zeros(n, dtype=torch.int64))
        self.register_buffer("b", torch.randn(n, generator=g) * 0.1)

    def forward(self, x):
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -127, 127,
                                      torch.int8)
        return F.linear(x, w, self.b)

    def mirror(self, x, act):
        return qm.affine_matmul(
            np.asarray(x, F32), self.c.numpy().astype(np.int64),
            self.s.numpy().astype(F32), self.b.numpy().astype(F32), act=act,
            bits=8).y


def pair(x, s, z, qmin=-128, qmax=127, dtype=torch.int8):
    q = qd.quantize_per_tensor(x, s, z, qmin, qmax, dtype)
    return qd.dequantize_per_tensor(q, s, z, qmin, qmax, dtype)


class InputPair(nn.Module):
    S, Z = f32(0.0213), -7

    def __init__(self):
        super().__init__()
        self.fc = Linear8(48, 24, 3)

    def forward(self, x):
        return self.fc(pair(x, self.S, self.Z))


class SharedPair(nn.Module):
    """One pair read by a Linear and by a portable cat."""

    S, Z = f32(0.031), 4

    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 32, 5)

    def forward(self, x):
        d = pair(x, self.S, self.Z)
        return torch.cat([self.fc(d), d], dim=-1)


class DoubleScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 16, 9)

    def forward(self, x):
        return self.fc(pair(x, 0.1, 3))       # 0.1 is not a float32 value


class PerChannelAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 16, 11)
        self.register_buffer("cs", torch.full((4,), 0.03))
        self.register_buffer("cz", torch.zeros(4, dtype=torch.int64))

    def forward(self, x):
        q = qd.quantize_per_channel(x, self.cs, self.cz, 0, -128, 127,
                                    torch.int8)
        d = qd.dequantize_per_channel(q, self.cs, self.cz, 0, -128, 127,
                                      torch.int8)
        return self.fc(d)


def lower(model, inputs, **options):
    partitioner = Esp32Partitioner(**options)
    ep = torch.export.export(model.eval(), inputs)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lowered = _passes.lower_esp32(ep, [partitioner])
    return lowered, partitioner, log.getvalue()


def calls(lowered):
    return [str(n.target) for n in lowered.exported_program().graph.nodes
            if n.op == "call_function" and "getitem" not in str(n.target)]


def run_host(runner, tmp_path, lowered, inputs):
    from executorch.exir import ExecutorchBackendConfig

    _passes.apply_passes(lowered, quiet=True)
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16))
    pte = tmp_path / "model.pte"
    pte.write_bytes(et.buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, x in enumerate(inputs):
        np.asarray(x, F32).tofile(tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    command += ["--output", str(tmp_path / "out.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    return np.fromfile(tmp_path / "out.bin", dtype=F32), done.stderr


def bits(a):
    return np.asarray(a, F32).view(np.uint32)


def test_input_pair_absorbed(runner, tmp_path):
    model = InputPair()
    x = torch.linspace(-3, 3, 4 * 48).reshape(4, 48)
    lowered, _, log = lower(model, (x,))
    report = lowered.static_qdq_report
    assert (report.absorbed, report.removed, report.portable) == (1, 1, [])
    assert calls(lowered) == ["executorch_call_delegate"], calls(lowered)
    assert "+aq" in log and "+sum16" in log, log
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    want = model.fc.mirror(x, qm.A5.static(InputPair.S, InputPair.Z))
    assert np.array_equal(bits(out), bits(want.reshape(-1)))


class ConstantCodes(nn.Module):
    """A Linear over int8 codes that are a program constant -- a pair whose
    quantize convert_pt2e folded, as for learned query tokens -- held 3-D,
    so that the matmul reads them through a view, or 2-D (`direct`), read
    as they are; times the input."""

    S, Z = f32(0.0117), 0

    def __init__(self, direct=False):
        super().__init__()
        g = torch.Generator().manual_seed(17)
        codes = torch.randint(-128, 128, (1, 16, 32), generator=g,
                              dtype=torch.int8)
        self.register_buffer("codes", codes[0].clone() if direct else codes)
        self.fc = Linear8(32, 24, 19)

    def forward(self, x):
        d = qd.dequantize_per_tensor(self.codes, self.S, self.Z, -128, 127,
                                     torch.int8)
        return self.fc(d) * x

    def mirror(self, x):
        d = qd.dequantize_per_tensor(self.codes, self.S, self.Z, -128, 127,
                                     torch.int8)
        y = self.fc.mirror(d.reshape(16, 32), qm.A5.static(self.S, self.Z))
        return (y.reshape(1, 16, 24) * np.asarray(x, F32)).astype(F32)


@pytest.mark.parametrize("direct", [False, True], ids=["viewed", "direct"])
def test_constant_codes_carry_their_pair(runner, tmp_path, direct):
    """A dequantize of constant codes is a pair whose quantize
    convert_pt2e folded. The Linear reading it carries that pair's static
    record, as for an absorbed pair, and the host runner's output is
    mirrors.affine_matmul's bit for bit with it: the dequantised codes
    requantise to themselves. Without the record the delegate quantised
    them again per row (A0) and moved the dot. Read directly (2-D codes)
    as behind a view: a constant precheck that refuses the direct spelling
    leaves the Linear and both dequantises portable on every call."""
    model = ConstantCodes(direct)
    x = torch.linspace(-2, 2, 16 * 24).reshape((16, 24) if direct else (1, 16, 24))
    lowered, _, log = lower(model, (x,))
    assert "executorch_call_delegate" in calls(lowered), calls(lowered)
    assert "+aq" in log, log
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    assert np.array_equal(bits(out), bits(model.mirror(x).reshape(-1)))


class ConstantCodesQK(nn.Module):
    """softmax(q @ k^T) with one operand int8 codes that are a program
    constant -- learned query tokens or a learned key memory whose pair
    convert_pt2e folded -- read as they are or through a view (`viewed`);
    the other operand the runtime input, with no pair."""

    S, Z = f32(0.0117), 0

    def __init__(self, side, viewed):
        super().__init__()
        g = torch.Generator().manual_seed(23)
        self.register_buffer("codes", torch.randint(
            -128, 128, (8, 32) if viewed else (1, 8, 32), generator=g,
            dtype=torch.int8))
        self.side, self.viewed = side, viewed

    def forward(self, x):                      # [1, 12, 32]
        c = qd.dequantize_per_tensor(self.codes, self.S, self.Z, -128, 127,
                                     torch.int8)
        if self.viewed:
            c = c.unsqueeze(0)
        q, k = (c, x) if self.side == "query" else (x, c)
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)


@pytest.mark.parametrize("side", ["query", "key"])
@pytest.mark.parametrize("viewed", [False, True])
def test_constant_codes_keep_qk_portable(runner, tmp_path, side, viewed):
    """Constant codes in the q@kT gate: an operand that dequantises constant
    codes is a static pair whose quantize convert_pt2e folded, and the
    delegate has no integer bmm that keeps a recipe's grid on an operand,
    so the bmm is refused by the static-pair rule, as for a pair with a
    quantize node, and runs portable on the recipe's values. Claimed, it
    quantised them again per row its own way."""
    model = ConstantCodesQK(side, viewed)
    x = torch.linspace(-2, 2, 12 * 32).reshape(1, 12, 32)
    lowered, partitioner, _ = lower(model, (x,), quantize_qk=True)
    assert "executorch_call_delegate" not in calls(lowered), calls(lowered)
    kinds = partitioner.support.kinds
    assert kinds["static pair on a bmm of two activations"] >= 1, kinds
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    with torch.no_grad():
        want = model(x).numpy().reshape(-1)
    assert np.abs(out - want).max() <= 1e-6, np.abs(out - want).max()


def test_shared_pair_stays_whole_for_its_portable_reader():
    model = SharedPair()
    x = torch.linspace(-2, 2, 2 * 32).reshape(2, 32)
    lowered, _, _ = lower(model, (x,))
    report = lowered.static_qdq_report
    assert (report.absorbed, report.removed) == (1, 0)
    graph = lowered.exported_program().graph
    qs = [n for n in graph.nodes if static_qdq.is_static_q(n)]
    assert len(qs) == 1
    (d,) = qs[0].users                         # its dequantise, still there
    assert static_qdq.static_pair(d) is qs[0]
    assert len(d.users) == 1 and "aten.cat" in str(next(iter(d.users)).target)
    # The delegate reads the tensor before the pair: the graph input.
    delegate = next(n for n in graph.nodes
                    if "call_delegate" in str(n.target))
    assert any(a.op == "placeholder" for a in delegate.args[1:])


@pytest.mark.parametrize("model, kind", [
    (DoubleScale, "static scale not a float32 value"),
    (PerChannelAct, "per-channel activation quantiser"),
])
def test_refused_by_name_and_left_portable(model, kind):
    lowered, partitioner, _ = lower(model(), (torch.randn(4, 32),))
    assert "executorch_call_delegate" not in calls(lowered)
    assert partitioner.support.kinds[kind] >= 1, partitioner.support.kinds


def test_merge_keeps_one_dequantise_per_quantize():
    """convert_pt2e's DuplicateDQPass copies, merged back: the same values."""
    class Dup(nn.Module):
        def forward(self, x):
            q = qd.quantize_per_tensor(x, 0.0625, 1, -128, 127, torch.int8)
            a = qd.dequantize_per_tensor(q, 0.0625, 1, -128, 127, torch.int8)
            b = qd.dequantize_per_tensor(q, 0.0625, 1, -128, 127, torch.int8)
            return a + b, a * b

    ep = torch.export.export(Dup(), (torch.randn(8),))
    graph = ep.graph_module.graph
    assert static_qdq.merge_duplicate_pairs(graph) == 1
    graph.lint()
    q = next(n for n in graph.nodes if static_qdq.is_static_q(n))
    assert len(q.users) == 1


# --- the output side -------------------------------------------------------------------

class OutputPair(nn.Module):
    """Linear -> ReLU -> its output pair -> the graph output: the pair rides
    on the Linear as its output quantisation, after the ReLU."""

    S, Z, SO, ZO = f32(0.0213), -7, f32(0.0117), -128

    def __init__(self):
        super().__init__()
        self.fc = Linear8(48, 24, 13)

    def forward(self, x):
        return pair(F.relu(self.fc(pair(x, self.S, self.Z))), self.SO, self.ZO)

    def mirror(self, x):
        y = self.fc.mirror(x, qm.A5.static(self.S, self.Z))
        return qm.fake_quant_output(np.maximum(y, F32(0)), self.SO, self.ZO,
                                    -128, 127)


def test_output_pair_requantizes_the_op(runner, tmp_path):
    model = OutputPair()
    x = torch.linspace(-3, 3, 4 * 48).reshape(4, 48)
    lowered, _, log = lower(model, (x,))
    report = lowered.static_qdq_report
    assert (report.absorbed, report.requant, report.portable) == (1, 1, [])
    assert calls(lowered) == ["executorch_call_delegate"], calls(lowered)
    assert "+relu+oq[0.0117,-128]" in log, log
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    assert np.array_equal(bits(out), bits(model.mirror(x).reshape(-1)))


class AfterThePair(nn.Module):
    """An activation that reads the fake-quantised value (after the pair)
    must not fuse into the op, nor must a residual add of dequantised values."""

    def __init__(self, tail):
        super().__init__()
        self.fc = Linear8(32, 32, 17)
        self.tail = tail

    def forward(self, x):
        d = pair(self.fc(pair(x, f32(0.03), 1)), f32(0.02), 3)
        if self.tail == "relu":
            return F.relu(d)
        return d + pair(x, f32(0.03), 1)


@pytest.mark.parametrize("tail, portable", [("relu", "relu"), ("add", "add")])
def test_nothing_fuses_after_the_pair(tail, portable):
    lowered, _, log = lower(AfterThePair(tail), (torch.randn(2, 32),))
    left = calls(lowered)
    assert any(f"aten.{portable}" in t for t in left), left
    assert "+oq[0.02,3]" in log and "+res" not in log and "+relu" not in log, log


class Approximated(nn.Module):
    """A sigmoid fused onto the Linear, then a pair: the device's sigmoid is
    an approximation, so the pair is not taken as the op's output
    quantisation (it stays portable on the safety-net kernels)."""

    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 32, 19)

    def forward(self, x):
        return pair(torch.sigmoid(self.fc(pair(x, f32(0.03), 1))),
                    f32(1 / 256), -128)


def test_requantize_refused_after_an_approximated_activation():
    lowered, partitioner, log = lower(Approximated(), (torch.randn(2, 32),))
    kinds = partitioner.support.kinds
    assert kinds["requantize after an approximated activation"] == 2, kinds
    assert "+sigmoid" in log and "+oq" not in log, log


# --- the safety net: pairs between portable ops ----------------------------------------

class Between(nn.Module):
    """Static pairs with nothing claimable around them, int8 and uint8,
    running on ExecuTorch's own quantize and dequantize kernels, registered
    on the host and the device."""

    def forward(self, x, y):
        a = pair(x, f32(0.05), 3)
        b = pair(F.relu(y), f32(0.02), 7, 0, 255, torch.uint8)
        return torch.cat([a, b], dim=-1)


def test_pairs_between_portable_ops_are_torchs(runner, tmp_path):
    model = Between().eval()
    # Ties, rails and ordinary values for both pairs' scales.
    x = (torch.arange(-40, 40, dtype=torch.float32) * 0.025).reshape(20, 4)
    y = (torch.arange(-40, 40, dtype=torch.float32) * 0.01).reshape(20, 4)
    x[0, 0], y[0, 0] = 1e6, -1e6
    lowered, _, _ = lower(model, (x, y))
    left = calls(lowered)
    assert sum("quantize_per_tensor" in t for t in left) == 4, left
    out, _ = run_host(runner, tmp_path, lowered, (x, y))
    with torch.no_grad():
        want = model(x, y).numpy()
    assert np.array_equal(bits(out), bits(want.reshape(-1)))


class PortableDoubleScale(nn.Module):
    def forward(self, x):
        return torch.cat([pair(x, 0.1, 2), x], dim=-1)


def test_portable_quantize_with_a_double_scale_is_refused():
    """ExecuTorch's kernel multiplies by 1.0f/(float)s, torch by
    fl32(fl64(1/s)): a portable quantize must have a float32 scale."""
    lowered, _, _ = lower(PortableDoubleScale(), (torch.randn(2, 8),))
    with pytest.raises(static_qdq.StaticPairRefused,
                       match=r"\[static-scale-not-float32\]"):
        _passes.apply_passes(lowered, quiet=True)


class PortableInfiniteScale(nn.Module):
    """A pair whose observer saw an inf, beside its input: XNNPACK
    QAT over x / (m + 1e-6), whose divisor the fake quantisation rounded to
    exactly 0."""

    def forward(self, x):
        return torch.cat([pair(x, float("inf"), -128), x], dim=-1)


def test_portable_quantize_with_an_infinite_scale_runs(runner, tmp_path):
    """inf is a float32 value whose reciprocal both kernels take as 0, torch
    as fl32(fl64(1/inf)) and ExecuTorch as 1.0f/(float)inf: every finite input
    quantises to the zero point and dequantises to 0 * inf = NaN in both. So
    the export is not refused, and the program runs as torch does."""
    model = PortableInfiniteScale().eval()
    x = torch.linspace(-2, 2, 2 * 8).reshape(2, 8)
    lowered, _, _ = lower(model, (x,))
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    with torch.no_grad():
        want = model(x).numpy()
    out = out.reshape(want.shape)
    assert np.isnan(want[:, :8]).all() and np.isnan(out[:, :8]).all(), out
    assert np.array_equal(bits(out[:, 8:]), bits(want[:, 8:]))


class PortablePerChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cs", torch.full((4,), 0.05))
        self.register_buffer("cz", torch.zeros(4, dtype=torch.int64))

    def forward(self, x):
        q = qd.quantize_per_channel(x, self.cs, self.cz, 1, -128, 127,
                                    torch.int8)
        return torch.cat([qd.dequantize_per_channel(
            q, self.cs, self.cz, 1, -128, 127, torch.int8), x], dim=-1)


def test_portable_per_channel_activation_is_refused():
    """Out of scope, and ExecuTorch's quantize_per_channel kernel aborts on
    the float32 scales PT2E stores: refused at export, by name."""
    lowered, _, _ = lower(PortablePerChannel(), (torch.randn(2, 4),))
    with pytest.raises(static_qdq.StaticPairRefused,
                       match=r"\[per-channel-activation\]"):
        _passes.apply_passes(lowered, quiet=True)


class Requantized(nn.Module):
    """q2(d1(x)): one pair's value requantised by another's constants (the
    Cortex-M softmax output's fixed grid, then the next layer's). The Linear
    pointed past q2's pair reads d1's value and quantises it with q2's
    constants, as the model does."""

    S1, Z1, S2, Z2 = f32(0.03), 5, f32(0.0117), -9

    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 16, 23)

    def forward(self, x):
        d1 = pair(x, self.S1, self.Z1)
        return torch.cat([self.fc(pair(d1, self.S2, self.Z2)), d1[:, :16]], -1)


def test_requantized_pair_quantises_with_the_nearer_constants(runner, tmp_path):
    model = Requantized()
    x = torch.linspace(-2, 2, 2 * 32).reshape(2, 32)
    lowered, _, _ = lower(model, (x,))
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    d1 = qm.fake_quant_output(x.numpy(), model.S1, model.Z1, -128, 127)
    y = model.fc.mirror(d1, qm.A5.static(model.S2, model.Z2))
    want = np.concatenate([y, d1[:, :16]], axis=-1)
    assert np.array_equal(bits(out), bits(want.reshape(-1)))


class Int32ZeroPoints(nn.Module):
    """A per-channel weight whose op stays portable (a Linear behind a
    per-channel activation pair it cannot absorb), with int32 zero points as
    Cortex-M stores them: ExecuTorch's dequantize_per_channel takes int64
    only, so the export widens them (exactly)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(29)
        self.register_buffer("c", torch.randint(-127, 128, (8, 16), generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(8, generator=g) * 1e-2 + 1e-3)
        self.register_buffer("z", torch.zeros(8, dtype=torch.int32))

    def forward(self, x):
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -127, 127,
                                      torch.int8)
        return x + w.sum(dim=1)


def test_int32_zero_points_are_widened_for_the_kernel():
    from executorch.exir import to_edge

    model = Int32ZeroPoints().eval()
    x = torch.randn(2, 8)
    ep = torch.export.export(model, (x,))
    edge = to_edge(ep)
    program = edge.exported_program()
    assert static_qdq.widen_portable_zero_points(program) == 1
    dq = next(n for n in program.graph.nodes
              if static_qdq._is(n, static_qdq.DQC))
    assert dq.args[2].meta["val"].dtype == torch.int64


# --- the parity rows (fixtures/asym_static.py) on the host ----------------------------

from executorch.backends.esp32s3.test.fixtures import asym_static  # noqa: E402


@pytest.fixture(scope="module")
def static_rows():
    return {f.name: f for f in asym_static.fixtures()}


@pytest.mark.parametrize("name", asym_static.ROWS)
def test_static_row_host_bits_are_the_mirror(runner, tmp_path, static_rows,
                                             name):
    """Each stock-recipe block lowers with nothing portable but what its row
    keeps, runs on the host runner bit for bit as StaticMirror computes it --
    a code row (float reductions) within one code of it, element
    by element -- and stays within Stage A's bar of the recipe's eager model;
    a table row's tables match eager's activations but at recorded split
    codes in scalar-tail positions."""
    fixture = static_rows[name]
    et, _ = asym_static.lower(fixture)
    out = run_row(runner, tmp_path, et, fixture.inputs, fixture.outputs)
    want = fixture.reference.reshape(-1)
    if fixture.bar == "code":
        gap = asym_static.code_gap(want[:fixture.measured],
                                          out[:fixture.measured], fixture.grid)
        assert gap.ok(), gap.summary()
    else:
        assert np.array_equal(bits(out), bits(want))
    assert fixture.eager.ok(), fixture.eager
    if fixture.lut_gap is not None:
        assert fixture.lut_gap.ok(), fixture.lut_gap.summary()


def run_row(runner, tmp_path, et, inputs, outputs=1):
    """A row's program on the host runner, each input written in the memory
    order its strides give (a channels-last row's NHWC, as the device fills
    it; float32, or the int64 rule); its outputs as the program writes them,
    one after another."""
    pte = tmp_path / "row.pte"
    pte.write_bytes(et.buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, x in enumerate(inputs):
        asym_static.memory_order(x).numpy().tofile(tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    for k in range(outputs):
        command += ["--output", str(tmp_path / f"out{k}.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    return np.concatenate([np.fromfile(tmp_path / f"out{k}.bin", dtype=F32)
                           for k in range(outputs)])


@pytest.mark.parametrize("name", [r.name for r in asym_static.SPECS
                                  if r.channels_last])
def test_channels_last_rows_are_the_row_major_program(runner, tmp_path,
                                                      static_rows, name):
    """The channels-last row and the same converted model exported with a
    row-major input: the same delegate blobs, and the same output bits on the
    same tensor (make_inputs_row_major_pass.py moves values, it never rounds them)."""
    fixture = static_rows[name]
    et_cl, _ = asym_static.lower(fixture)
    rm = asym_static.Fixture(
        f"{name}_rm", fixture.model, tuple(x.contiguous() for x in fixture.inputs),
        fixture.reference, {}, luts=fixture.luts,
        keeps=tuple(k for k in fixture.keeps if "dim_order" not in k))
    et_rm, _ = asym_static.lower(rm)

    def blobs(et):
        return [bytes(d.data) for d in et.executorch_program.backend_delegate_data]

    assert blobs(et_cl) == blobs(et_rm) and all(blobs(et_cl))
    (tmp_path / "cl").mkdir()
    (tmp_path / "rm").mkdir()
    out_cl = run_row(runner, tmp_path / "cl", et_cl, fixture.inputs)
    out_rm = run_row(runner, tmp_path / "rm", et_rm, rm.inputs)
    shape = fixture.reference.shape
    if len(shape) == 4:     # channels-last bytes back to the logical order
        out_cl = out_cl.reshape(shape).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(bits(out_cl.reshape(-1)),
                                  bits(out_rm.reshape(-1)))


def test_identity_pairs_on_one_grid_are_elided():
    """Arm's uint8 model IO: a uint8 pair, then an int8 one on the same scale
    with the zero point moved by 128 -- an identity on the first's values,
    removed; a pair on another scale stays."""
    class IO(nn.Module):
        def forward(self, x):
            s = f32(0.0222)
            u8 = pair(x, s, 101, 0, 255, torch.uint8)
            same = pair(u8, s, -27)                  # 101 - 128
            other = pair(u8, f32(0.03), -27)         # another grid
            return same + other

    ep = torch.export.export(IO(), (torch.randn(8),))
    graph = ep.graph_module.graph
    assert static_qdq.elide_identity_pairs(graph, ep.graph_signature) == 1
    graph.lint()
    qs = [static_qdq.qparams(n) for n in graph.nodes
          if static_qdq.is_static_q(n)]
    assert sorted((q.scale, q.zero_point) for q in qs) == sorted(
        [(f32(0.0222), 101), (f32(0.03), -27)])
