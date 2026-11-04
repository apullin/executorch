# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""16-bit activations (Arm a16w8): what the export does with a static
int16 pair, the host runner's bits, and what is refused by name.

On small graphs spelled the way convert_pt2e leaves them (quantize_per_tensor
/ dequantize_per_tensor .default with int16 and float32 constants, weights as
dequantize_per_channel over int8 codes):

  - an int16 input pair in front of a claimed Linear is absorbed: the
    delegate carries a static EXT_ACT_QUANT with code_bits 16 and computes
    mirrors.affine_matmul's 16-bit branch bit for bit;
  - an int16 output pair after a fused ReLU becomes the op's EXT_OUT_QUANT
    over the int16 range;
  - refused by name, the op left portable on the model's own arithmetic: an
    int16 zero point other than 0, uint16 codes, int16 weights under int16
    activations (a16w16), per-channel int16 activations, int4 or grouped
    weights under 16-bit codes, and a weight the backend would quantise at a
    policy other than i8;
  - every asym_a16 row runs on the host runner bit for bit as its
    mirror computes it.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3 import _passes
from executorch.backends.esp32s3.test.fixtures import asym_a16
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.asym.test_asym_static import (
    Linear8,
    bits,
    calls,
    f32,
    lower,
    pair,
    qd,
    run_host,
    runner,
)  # noqa: F401 -- fixture

F32 = np.float32
I16 = (-32767, 32767, torch.int16)


def a6(s):
    return qm.A6.static(s)


class Int16In(nn.Module):
    """An int16 input pair in front of an int8 per-channel Linear."""

    S = f32(2.0 ** -12)          # Arm a16w8's eps floor

    def __init__(self):
        super().__init__()
        self.fc = Linear8(48, 24, 3)

    def forward(self, x):
        return self.fc(pair(x, self.S, 0, *I16))


class Int16InOut(nn.Module):
    """Linear -> ReLU -> an int16 output pair: the pair rides on the Linear
    as its output quantisation, after the ReLU, over the int16 range."""

    S, SO = f32(0.000123), f32(2.0 ** -11)

    def __init__(self):
        super().__init__()
        self.fc = Linear8(48, 24, 13)

    def forward(self, x):
        return pair(F.relu(self.fc(pair(x, self.S, 0, *I16))), self.SO, 0,
                    *I16)

    def mirror(self, x):
        y = self.fc.mirror(x, a6(self.S))
        return qm.fake_quant_output(np.maximum(y, F32(0)), self.SO, 0,
                                    -32767, 32767)


def test_int16_input_pair_absorbed(runner, tmp_path):
    model = Int16In()
    x = torch.linspace(-9, 9, 4 * 48).reshape(4, 48)   # past both rails
    lowered, _, log = lower(model, (x,))
    report = lowered.static_qdq_report
    assert (report.absorbed, report.removed, report.portable) == (1, 1, [])
    assert calls(lowered) == ["executorch_call_delegate"], calls(lowered)
    assert "+aq" in log and "+sum" not in log, log
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    want = model.fc.mirror(x, a6(Int16In.S))
    assert np.array_equal(bits(out), bits(want.reshape(-1)))


def test_int16_output_pair_requantizes_the_op(runner, tmp_path):
    model = Int16InOut()
    x = torch.linspace(-3, 3, 4 * 48).reshape(4, 48)
    lowered, _, log = lower(model, (x,))
    report = lowered.static_qdq_report
    assert (report.absorbed, report.requant, report.portable) == (1, 1, [])
    assert calls(lowered) == ["executorch_call_delegate"], calls(lowered)
    assert "+relu+oq[0.000488281,0]" in log, log
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    assert np.array_equal(bits(out), bits(model.mirror(x).reshape(-1)))


# --- refused by name -------------------------------------------------------------------

class Int16ZeroPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 16, 7)

    def forward(self, x):
        return self.fc(pair(x, f32(2.0 ** -12), 5, *I16))


class Uint16(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 16, 9)

    def forward(self, x):
        return self.fc(pair(x, f32(2.0 ** -12), 0, 0, 65535, torch.uint16))


class A16W16(nn.Module):
    """int16 activations against an int16 per-channel weight (a16w16)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(11)
        self.register_buffer("c", torch.randint(-32767, 32768, (16, 32),
                                                generator=g,
                                                dtype=torch.int16))
        self.register_buffer("s", torch.rand(16, generator=g) * 1e-5 + 1e-6)
        self.register_buffer("z", torch.zeros(16, dtype=torch.int64))

    def forward(self, x):
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -32767, 32767,
                                      torch.int16)
        return F.linear(pair(x, f32(2.0 ** -12), 0, *I16), w)


class A16W4(nn.Module):
    """int16 activations against int4 per-channel codes ([-8, 7])."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(13)
        self.register_buffer("c", torch.randint(-8, 8, (16, 32), generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(16, generator=g) * 1e-2 + 1e-3)
        self.register_buffer("z", torch.zeros(16, dtype=torch.int64))

    def forward(self, x):
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -8, 7,
                                      torch.int8)
        return F.linear(pair(x, f32(2.0 ** -12), 0, *I16), w)


class A16Grouped(nn.Module):
    """int16 activations against int8 codes with per-group scales (32)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(17)
        self.register_buffer("c", torch.randint(-127, 128, (16, 64),
                                                generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(16, 2, generator=g) * 1e-2 + 1e-3)
        self.register_buffer("z", torch.zeros(16, 2, dtype=torch.int64))

    def forward(self, x):
        w = qd.dequantize_per_channel_group(self.c, self.s, self.z, -127, 127,
                                            torch.int8, 32, torch.float32)
        return F.linear(pair(x, f32(2.0 ** -12), 0, *I16), w)


class PerChannel16(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear8(32, 16, 19)
        self.register_buffer("cs", torch.full((4,), 2.0 ** -12))
        self.register_buffer("cz", torch.zeros(4, dtype=torch.int64))

    def forward(self, x):
        q = qd.quantize_per_channel(x, self.cs, self.cz, 0, -32767, 32767,
                                    torch.int16)
        d = qd.dequantize_per_channel(q, self.cs, self.cz, 0, -32767, 32767,
                                      torch.int16)
        return self.fc(d)


class FloatWeight(nn.Module):
    """An int16 pair in front of a float Linear (Cortex-M's int16 config
    leaves its Linears float): at an explicit policy the backend quantises
    the weight itself; the default keeps it float."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(23)
        self.fc = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc(pair(x, f32(2.0 ** -12), 0, *I16))


@pytest.mark.parametrize("model, kind", [
    (Int16ZeroPoint, "static int16 zero point"),
    (Uint16, "static uint16 activation"),
    (A16W16, "int16 weights with static int16 activations"),
    (A16W4, "16-bit activation codes on int4 weights"),
    (A16Grouped, "16-bit activation codes on grouped weights"),
    (PerChannel16, "per-channel activation quantiser"),
], ids=lambda v: getattr(v, "__name__", v))
def test_refused_by_name_and_left_portable(model, kind):
    lowered, partitioner, _ = lower(model(), (torch.randn(4, 32)
                                              if model is not A16Grouped
                                              else torch.randn(4, 64),))
    assert "executorch_call_delegate" not in calls(lowered)
    assert partitioner.support.kinds[kind] >= 1, partitioner.support.kinds


def test_weight_quantised_here_needs_the_i8_policy(runner, tmp_path):
    """At an explicit i8 policy the backend's own int8 rows take the 16-bit
    record (and compute the mirror with those codes); at another explicit
    policy the op is refused by name and stays portable. (The default keeps
    the weight float: test_float_weights.py.)"""
    model = FloatWeight().eval()
    x = torch.linspace(-4, 4, 4 * 32).reshape(4, 32)
    ep = torch.export.export(model, (x,))
    auto = Esp32Partitioner(format_policy="auto")
    lowered = _passes.lower_esp32(ep, [auto])
    assert "executorch_call_delegate" not in calls(lowered)
    assert auto.support.kinds[
        "16-bit activation codes on a weight quantised here"] >= 1

    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        quantize_absmax_rows,
    )

    lowered = _passes.lower_esp32(torch.export.export(model, (x,)),
                                    [Esp32Partitioner(format_policy="i8")])
    assert calls(lowered) == ["executorch_call_delegate"], calls(lowered)
    out, _ = run_host(runner, tmp_path, lowered, (x,))
    w = model.fc.weight.detach().numpy().astype(F32)
    codes, scales = quantize_absmax_rows(w, FORMATS["i8"])
    want = qm.affine_matmul(x.numpy(), np.asarray(codes, np.int64),
                            np.asarray(scales, F32),
                            model.fc.bias.detach().numpy().astype(F32),
                            act=a6(f32(2.0 ** -12)), bits=8).y
    assert np.array_equal(bits(out), bits(want.reshape(-1)))


# --- the parity rows (fixtures/asym_a16.py) on the host -------------------------------

@pytest.fixture(scope="module")
def a16_rows():
    return {f.name: f for f in asym_a16.fixtures()}


@pytest.mark.parametrize("name", asym_a16.ROWS)
def test_a16_row_host_bits_are_the_mirror(runner, tmp_path, a16_rows, name):
    """Each block lowers with nothing portable but what its row keeps, every
    claimed op on the int16 loop, runs on the host runner bit for bit as
    StaticMirror computes it, and stays within one int16 step of eager."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as st

    fixture = a16_rows[name]
    et, log = st.lower(fixture)
    ops = [line for line in log.splitlines() if "[Esp32Backend]" in line]
    assert ops and all("+aq" in line for line in ops
                       if "addmm" in line), "\n".join(ops)
    pte = tmp_path / f"{name}.pte"
    pte.write_bytes(et.buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, x in enumerate(fixture.inputs):
        np.asarray(x, F32).tofile(tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    command += ["--output", str(tmp_path / "out.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    out = np.fromfile(tmp_path / "out.bin", dtype=F32)
    assert np.array_equal(bits(out), bits(fixture.reference.reshape(-1)))
    if name not in asym_a16.WEIGHTS_HERE:
        assert fixture.eager.steps <= 1, fixture.eager
