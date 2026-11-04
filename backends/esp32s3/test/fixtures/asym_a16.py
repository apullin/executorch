# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""int16 activations x int8 weights (Arm a16w8), end to end: fixtures
from the stock quantizer and the device's arithmetic for each.

Arm's get_symmetric_a16w8_quantization_config() quantises activations
static per tensor to int16 over [-32767, 32767] with zero point 0 (the
mirrors' A6; HistogramObserver, eps 2^-12) and weights to int8 per channel
over [-127, 127] (eps 2^-16, the base config's), with int32 biases. The backend
runs such a model with its own rounding: each claimed matmul or convolution
takes its int16 input pair as a static EXT_ACT_QUANT with code_bits 16, and
its output pair as EXT_OUT_QUANT over the int16 range (static_qdq.py); the
runtime quantises to int16 codes, dots them against the int8 codes in int64
and applies the two-rounding epilogue (esp32_asym.cpp, a16_column).

StaticMirror (asym_static.py) is the reference: the converted graph
with every matmul and convolution the device's way -- mirrors.py.
affine_matmul / affine_conv, whose 16-bit branch accumulates in int64 -- and
everything else as torch computes it. A row holds the device's output bits
to the mirror's, exactly, and reports its gap to the recipe's eager model:
the output codes that flip, in the int16 output pair's steps, and the
largest difference.

  a16_arm_mlp        Arm a16w8, per-channel weights: Linear (its output
                     pair as the op's output quantisation), ReLU, Linear.
                     Unlike a8w8, a16w8 puts a pair between the Linear and
                     its ReLU, which then reads the fake-quantised value and
                     runs portable (exact) between the two delegates
  a16_arm_mlp_pt     the same with per-tensor weights (is_per_channel=False)
  a16_arm_conv       convolutions: 3x3 padded, ReLU (portable, as above), 3x3
                     stride 2, 3x3 dilation 2 -- K = 36 and 72, ragged against
                     the 16-element row alignment
  a16_arm_dwconv     depthwise 3x3 padded, ReLU, grouped 3x3 (groups 2), 1x1
  a16_arm_residual   Linear, a residual add of dequantised int16 values
                     between two delegates (portable, on the safety-net
                     kernels), Linear
  a16_rails          hand-written int16 pairs (float32 constants) around an
                     int8 per-channel Linear with K = 1024: the input pair's
                     scale puts the ramp on both rails, and the weights at
                     +-127 follow the input's sign in half the columns, so
                     |acc| passes 2^31 (an int32 accumulator would wrap)
  a16_conv_relu      hand-written int16 pairs (float32 constants) around an
                     int8 per-channel 3x3 padded conv with its ReLU fused
                     before the output pair: the int16 loop's epilogue, the
                     activation then the output quantisation (stock a16w8
                     never fuses one: it puts a pair before the ReLU)
  a16_cm_linear      Cortex-M's INT16_PER_TENSOR_CONFIG on a Linear + ReLU
                     divided by a positive constant: the config quantises
                     the div alone (its Linear check needs int8
                     activations), and the div's input pair reaches the
                     Linear's output through the ReLU, so the Linear stays
                     float between int16 pairs. The row lowers at an
                     explicit format_policy="i8", so the backend quantises
                     the weight (per-row absmax int8 codes) under the model's
                     16-bit record; the default policy would keep it float
                     (the fp32 loop, both pairs portable). The pair after the
                     Linear is its output quantisation; the ReLU, the div and
                     their int16 pairs run portable (the safety-net kernels).
                     Against eager the gap is that weight quantisation, not
                     a rounding: the one-step bar does not apply
                     (WEIGHTS_HERE)

export_all(out_dir) writes every program and the packed image main.cpp reads
from the asyma16 flash partition (asym_a16_fixtures.bin; asym.py's
layout); it returns {row: reference output}. Imports torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.test.fixtures import asym_static as st
from executorch.backends.esp32s3.test.fixtures.asym import ramp
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
IMAGE = "asym_a16_fixtures.bin"
ROWS = ("a16_arm_mlp", "a16_arm_mlp_pt", "a16_arm_conv", "a16_arm_dwconv",
        "a16_arm_residual", "a16_rails", "a16_conv_relu", "a16_cm_linear")
QEMU_ROWS = ROWS
# Rows whose float weights the backend quantises (the recipe left them float,
# and the row lowers at an explicit format_policy="i8"): their gap to eager is
# the backend's int8 rows, not Stage A's one-code bar.
WEIGHTS_HERE = ("a16_cm_linear",)


class Rails(nn.Module):
    """int16 pairs with float32 constants, as convert_pt2e spells them, around
    an int8 per-channel Linear [16, 1024]. The input pair's scale 2^-16
    (range +-0.5) puts every ramp value past 0.5 on the positive rail and -0.5
    on the negative one; columns 0..7 have weights of +-127 with the input's
    sign, so their |acc| = 127 * sum |xq| passes 2^31; columns 8..15 are
    random. The output pair is chosen from the float model on the ramp."""

    K, N = 1024, 16
    SI = float(F32(2.0 ** -16))

    def __init__(self, seed=71):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        x = ramp((1, self.K))[0]
        sign = torch.where(x >= 0, 1, -1).to(torch.int8)
        aligned = (sign * 127).expand(8, self.K)
        rand = torch.randint(-127, 128, (8, self.K), generator=g,
                             dtype=torch.int8)
        self.register_buffer("c", torch.cat([aligned, rand]).contiguous())
        self.register_buffer("s", torch.rand(self.N, generator=g) * 1e-6
                             + 1e-6)
        self.register_buffer("z", torch.zeros(self.N, dtype=torch.int64))
        self.register_buffer("b", torch.randn(self.N, generator=g) * 0.1)
        self.so = None
        with torch.no_grad():
            y = self._linear(self._pair(ramp((2, self.K)), self.SI))
        self.so = float(F32(float(y.abs().max()) / 32767))

    @staticmethod
    def _pair(t, s):
        qd = ea._decomposed()
        i16 = (-32767, 32767, torch.int16)
        return qd.dequantize_per_tensor(qd.quantize_per_tensor(t, s, 0, *i16),
                                        s, 0, *i16)

    def _linear(self, x):
        qd = ea._decomposed()
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -127, 127,
                                      torch.int8)
        return F.linear(x, w, self.b)

    def forward(self, x):
        return self._pair(self._linear(self._pair(x, self.SI)), self.so)


class ConvRelu(nn.Module):
    """int16 pairs with float32 constants around an int8 per-channel 3x3
    conv (padding 1) whose ReLU comes before the output pair, so it fuses:
    the delegate applies the ReLU, then the output quantisation. The output
    scale is chosen from the float model on the ramp."""

    SI = float(F32(2.0 ** -12))

    def __init__(self, seed=79):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("c", torch.randint(-127, 128, (8, 4, 3, 3),
                                                generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(8, generator=g) * 2e-3 + 1e-3)
        self.register_buffer("z", torch.zeros(8, dtype=torch.int64))
        self.register_buffer("b", torch.randn(8, generator=g) * 0.1)
        self.so = None
        with torch.no_grad():
            y = F.relu(self._conv(Rails._pair(ramp((1, 4, 9, 9)), self.SI)))
        self.so = float(F32(float(y.max()) / 32767))

    def _conv(self, x):
        qd = ea._decomposed()
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -127, 127,
                                      torch.int8)
        return F.conv2d(x, w, self.b, padding=1)

    def forward(self, x):
        return Rails._pair(F.relu(self._conv(Rails._pair(x, self.SI))),
                           self.so)


class LinReluDiv(nn.Module):
    """A Linear and its ReLU, divided by a positive constant: the div is the
    one op Cortex-M's int16 config quantises (CortexMDivCheck)."""

    def __init__(self, seed=73):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(64, 32)
        self.register_buffer("d", torch.rand(4, 32) + 0.5)

    def forward(self, x):
        return F.relu(self.fc(x)) / self.d


@dataclass
class Row(st.Row):
    per_channel: bool = True


RELU = ("aten.relu",)
SPECS = (
    Row("a16_arm_mlp", "arm-a16w8", st.MLP, (4, 64),
        portable=st.VIEWS + RELU),
    Row("a16_arm_mlp_pt", "arm-a16w8", st.MLP, (4, 64),
        portable=st.VIEWS + RELU, per_channel=False),
    Row("a16_arm_conv", "arm-a16w8", st.ConvNet, (1, 4, 9, 9), portable=RELU),
    Row("a16_arm_dwconv", "arm-a16w8", st.DwGrouped, (1, 8, 7, 7),
        portable=RELU),
    Row("a16_arm_residual", "arm-a16w8", st.Residual, (4, 64),
        portable=st.VIEWS + ("aten.add.Tensor", "quantize_per_tensor",
                             "dequantize_per_tensor")),
    Row("a16_rails", "", Rails, (2, 1024)),
    Row("a16_conv_relu", "", ConvRelu, (1, 4, 9, 9)),
    Row("a16_cm_linear", "cortex-m-int16", LinReluDiv, (4, 64),
        portable=RELU + ("aten.div.Tensor", "quantize_per_tensor",
                         "dequantize_per_tensor")),
)
assert tuple(r.name for r in SPECS) == ROWS


def _arm_a16w8(net, inputs, batches, per_channel):
    """Arm's a16w8 config as Arm ships it (quant_census.recipes' arm-a16w8,
    or its per-tensor-weight form)."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    recipes.tosa_serializer_source()
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer, get_symmetric_a16w8_quantization_config)

    quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u55-128")).set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel))
    return recipes._pt2e(net, inputs, batches, quantizer, False)


def _cortex_m_int16(net, inputs, batches):
    """CortexMQuantizer with its INT16_PER_TENSOR_CONFIG, as Cortex-M ships
    it for int16 activations (2-D inputs: no channels-last)."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    recipes.tosa_serializer_source()
    from executorch.backends.cortex_m.quantizer.quantization_configs import (
        INT16_PER_TENSOR_CONFIG)
    from executorch.backends.cortex_m.quantizer.quantizer import (
        CortexMQuantizer)

    quantizer = CortexMQuantizer(per_tensor_config=INT16_PER_TENSOR_CONFIG)
    return recipes._pt2e(net, inputs, batches, quantizer, False)


def quantized(row):
    """(converted module, its example inputs)."""
    x = ramp(row.shape)
    if not row.recipe:
        return row.build().eval(), (x,)
    if row.recipe == "cortex-m-int16":
        return _cortex_m_int16(row.build().eval(), (x,),
                               st.calibration(row.shape)), (x,)
    return _arm_a16w8(row.build().eval(), (x,), st.calibration(row.shape),
                      row.per_channel), (x,)


class HereMirror(st.StaticMirror):
    """StaticMirror for a matmul whose weight the recipe left float, lowered
    at an explicit i8 policy: the backend quantises it, quantize_absmax_rows'
    per-row int8 codes and fp32 scales, exactly as preprocess does."""

    KEEP_FLOAT = False

    def _weight(self, value, n_out):
        d = st._behind_moves(value)
        if st._is(d, "dequantize"):
            return super()._weight(value, n_out)
        from executorch.backends.esp32s3.serialization.weight_formats import (
            FORMATS,
            quantize_absmax_rows,
        )

        w = self.env[d].detach().numpy().astype(F32)
        codes, scales = quantize_absmax_rows(w.reshape(n_out, -1),
                                             FORMATS["i8"])
        return (np.asarray(codes, np.int64), np.asarray(scales, F32), None)


def mirror(model, inputs, luts, weights_here=False):
    """st.mirror, with HereMirror for a row whose weights are quantised
    here."""
    ep = st.mirrored_program(model, inputs, luts)
    cls = HereMirror if weights_here else st.StaticMirror
    with torch.no_grad():
        out = cls(ep.module(), garbage_collect_values=False).run(*inputs)
    return np.asarray(out.detach().numpy(), F32)


def fixtures():
    out = []
    for row in SPECS:
        model, inputs = quantized(row)
        ref = mirror(model, inputs, row.luts, row.name in WEIGHTS_HERE)
        here = {"format_policy": "i8"} if row.name in WEIGHTS_HERE else {}
        fx = st.Fixture(row.name, model, inputs, ref, here, luts=row.luts,
                        keeps=row.portable)
        fx.eager = st.eager_gap(model, inputs, ref)
        out.append(fx)
    assert tuple(f.name for f in out) == ROWS
    return out


def emitted_programs():
    """Every row's emitted program (dump_required_ops: the portable kernels
    the parity firmware must register for them)."""
    return [st.lower(f)[0] for f in fixtures()]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte) and the QEMU image (asym_a16_fixtures.bin).
    Returns {row: reference}; each row's gap to eager is on export_all.gaps."""
    table = fixtures()
    programs, refs = {}, {}
    export_all.gaps = {}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        export_all.gaps[fixture.name] = fixture.eager
        et, log = st.lower(fixture)
        programs[fixture.name] = bytes(et.buffer)
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(et.buffer)
        if verbose:
            ops = [line.split(": ", 1)[-1] for line in log.splitlines()
                   if "[Esp32Backend]" in line]
            g = fixture.eager
            print(f"  {fixture.name}.pte: {len(et.buffer):,} B; vs eager "
                  f"{g.flips} code flip(s), max abs {g.max_abs:.3g}; "
                  + "; ".join(ops))
    ea.write_image(os.path.join(out_dir, IMAGE), table, programs)
    return refs


if __name__ == "__main__":
    for name, ref in export_all().items():
        gap = export_all.gaps[name]
        print(f"  {name}: reference {np.shape(ref)}, {gap.flips} flip(s), "
              f"max abs {gap.max_abs:.3g}")
