# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static int8 graphs (Stage A), end to end: fixtures from the stock static
quantizers and the device's arithmetic for each.

The backend runs a stock static recipe's model with its own rounding: each
static quantize/dequantize pair a delegate takes becomes its op's static input
record or its output quantisation, and a pair left between portable ops runs
on ExecuTorch's own kernels (static_qdq.py). Each fixture here is a small
block quantised by a stock flow exactly as its owner ships it
(quant_census.recipes), and its reference is what the device computes for it:
StaticMirror runs the converted graph with every matmul and convolution the
device's way -- the activation's codes from the pair in front of it, the
exact integer dot, the two-rounding epilogue (mirrors.affine_matmul /
affine_conv) -- and every other op as torch does (the pairs, ReLU, the add,
leaky ReLU, the table op: the ones the device computes as torch does), but
sqrt, which the device and its table compute correctly rounded and torch's
kernel does not on every machine (fold_quant_lut_pass.EXACT). A row
holds the device's output bits to the mirror's, exactly; against the
recipe's eager model it reports the output codes that flip (Stage A's bar: at
most one, near a tie) and the largest difference.

  st_xnn_mlp        XNNPACK static, per-channel weights: Linear + ReLU (the
                    pair after the ReLU absorbed), Linear, the output pair as
                    the op's output quantisation
  st_xnn_mlp_pt     the same with per-tensor weights (is_per_channel=False)
  st_xnnqat_mlp     the same through XNNPACK's QAT flow (prepare_qat_pt2e)
  st_arm_mlp        Arm a8w8: per-channel weights, int32 biases
  st_cm_mlp         Cortex-M: per-tensor Linear weights, int32 biases
  st_xnn_conv       XNNPACK static convolutions: 3x3 padded + ReLU (fused),
                    3x3 stride 2, 3x3 dilation 2 -- the padding written as
                    each static zero point
  st_arm_dwconv     Arm a8w8: depthwise 3x3 padded + ReLU, grouped 3x3
                    (groups 2), 1x1
  st_xnn_residual   XNNPACK static: Linear, a residual add of dequantised
                    values between two delegates (portable, on the safety-net
                    kernels for the input's pair), Linear
  st_xnn_sig_lut    XNNPACK static: Linear, sigmoid as espressif::quant_lut
                    (the default), Linear
  st_arm_leaky      Arm a8w8: Linear, leaky ReLU left portable (lowered with
                    quant_luts=False), Linear
  st_u8_io          uint8 pairs (hand-written, float32 constants) around a
                    Linear: the input record's codes and zero point ride
                    shifted onto int8, the output's range stays uint8
  st_arm_u8io       Arm a8w8 with uint8 model IO (set_io(get_uint8_io_
                    quantization_config())): uint8 pairs at the boundary,
                    requantised to int8 inside with the zero point moved by
                    128 -- identities on the same grid, elided
  st_cm_hardsig     Cortex-M: Linear + hardsigmoid fused (its pattern puts no
                    pair between them), its quantised value read by the next
                    Linear and, sliced, by the output (a portable cat)
  st_fused_acts     hand-written pairs: Linear + hardswish, then Linear +
                    leaky ReLU, each fused and requantised after its
                    activation -- what an output quantisation may follow
  st_cm_cl_conv     Cortex-M with its channels-last image: 1x1, ReLU, padded
                    3x3, 1x1 (the census's conv_pad); lower_esp32 copies the
                    NHWC input to row-major and the 4-D output back
                    (make_inputs_row_major_pass.py), so the device feeds and
                    hashes NHWC
  st_cm_cl_dw       the same with a depthwise 3x3 and a flattened output
                    (no copy back; Cortex-M requantises the flatten, a pair
                    left on the safety-net kernels)

A channels-last row's input is the device's ramp in its memory order, NHWC
(cl_ramp), and its reference is held in the memory order the program writes:
the device hashes the bytes it writes.

The ops the stock recipes leave between delegates, each between quantised
tensors exactly as its recipe leaves it. Exact rows hold the device's bits to
the mirror's, which computes every op between the delegates as torch does --
so the portable kernel is eager's arithmetic, given identical input codes:

  st_{xnn,arm,a16}_cnn_ops  convolutions with relu6 (hardtanh), max_pool2d,
                    a cat, avg_pool2d and a constant pad between them; a ReLU
                    in one branch (fused under a8w8, portable under a16w8)
  st_cm_ops         Cortex-M on row-major images (its conv patterns need
                    channels-last ones): relu6, both pools, cat, pad, permute
  st_{xnn,arm,cm,a16}_binary  add, sub, mul and div of two Linears' outputs
                    (Arm divides by a reciprocal table and a multiply)
  st_{xnn,arm,cm,a16}_chains  hardswish and hardsigmoid (to_edge's add,
                    clamp, mul and div), leaky ReLU, and for XNNPACK and
                    Cortex-M sqrt, rsqrt and reciprocal (Arm spells sqrt as
                    pow(x, 0.5)), all portable: quant_luts=False, or a16w8
  st_arm_llm_ops    the pieces Arm's LLM graphs leave outside (LlmPieces),
                    with an int64 input

Code rows are float reductions in another order than ATen's: the
first output is held to at most one code of its grid, element by element (the
device prints every value, FIXTURE_DUMP; run_parity counts the codes that
differ). XNNPACK quantises no softmax, norm or bmm output, nor Cortex-M a
mean or layer_norm (its graph output takes only a quantised producer's pair),
so those rows give the op a second reader, a Linear or conv, whose input
quantiser is the grid:

  st_{xnn,cm}_mean, st_xnn_adapool, st_arm_sum     mean.dim, adaptive pool, sum
  st_{xnn,arm,cm,a16}_softmax                      _softmax, or Arm's pieces
  st_{xnn,arm,cm}_layernorm, st_xnn_groupnorm      the norms
  st_{xnn,arm,cm}_bmm                              q @ k^T of two Linears

Tables between delegates: st_{xnn,arm,cm}_act_luts chain
gelu, tanh-gelu, tanh, SiLU and hardswish tables, exact against the mirror;
against eager, each table's codes are compared on eager's own inputs in the
graph's own layout (lut_eager_gap: a difference is allowed only at a recorded
split code in torch's scalar tail). st_spd_chain and st_spd_lut run the six
speed-only tables and the portable chains they replace on every finite code
of their grids, against one reference.

export_all(out_dir) writes every program and the packed images main.cpp reads
from the asymstat flash partition (asym_static_fixtures.bin; asym.py's
layout) and, for the code, table and speed rows, from asymops
(asym_ops_fixtures.bin); it returns {row: reference output}. Imports torch.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.fixtures.asym import ramp
from executorch.backends.esp32s3.test.fixtures.quant_lut import ByDefinition
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
IMAGE = "asym_static_fixtures.bin"
# The rows held to one code, the tables and the speed proof ride in a
# second image: they do not fit in the asymstat partition beside the rest.
OPS_IMAGE = "asym_ops_fixtures.bin"
ROWS = ("st_xnn_mlp", "st_xnn_mlp_pt", "st_xnnqat_mlp", "st_arm_mlp",
        "st_cm_mlp", "st_xnn_conv", "st_arm_dwconv", "st_xnn_residual",
        "st_xnn_sig_lut", "st_arm_leaky", "st_u8_io", "st_arm_u8io",
        "st_cm_hardsig", "st_fused_acts", "st_cm_cl_conv", "st_cm_cl_dw",
        # The ops between delegates, exact...
        "st_xnn_cnn_ops", "st_arm_cnn_ops", "st_a16_cnn_ops", "st_cm_ops",
        "st_xnn_binary", "st_arm_binary", "st_cm_binary", "st_a16_binary",
        "st_xnn_chains", "st_arm_chains", "st_cm_chains", "st_a16_chains",
        "st_arm_llm_ops",
        # ... at most one code ...
        "st_xnn_mean", "st_xnn_adapool", "st_cm_mean", "st_arm_sum",
        "st_xnn_softmax", "st_arm_softmax", "st_cm_softmax", "st_a16_softmax",
        "st_xnn_layernorm", "st_arm_layernorm", "st_cm_layernorm",
        "st_xnn_groupnorm", "st_xnn_bmm", "st_arm_bmm", "st_cm_bmm",
        # ... tables between delegates, and the speed tables' proof.
        "st_xnn_act_luts", "st_arm_act_luts", "st_cm_act_luts",
        "st_spd_chain", "st_spd_lut",
        # Weights a recipe leaves float stay float: a Linear's and a table's,
        # a convolution's (the fp32 convolution), a half checkpoint's (the
        # fp16/bf16 rungs).
        "st_xnn_float_w", "st_xnn_conv_f32", "st_xnn_half")
QEMU_ROWS = ROWS
# The rows in OPS_IMAGE (the asymops partition, 0x5C): from the first code row
# on.
OPS_ROWS = ROWS[ROWS.index("st_xnn_mean"):]


# --- the blocks (float), quantised by the stock flows ---------------------------------

class MLP(nn.Module):
    def __init__(self, seed=31):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, 32)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ConvNet(nn.Module):
    def __init__(self, seed=37):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(8, 4, 3, padding=2, dilation=2)

    def forward(self, x):
        return self.c3(self.c2(F.relu(self.c1(x))))


class DwGrouped(nn.Module):
    def __init__(self, seed=41):
        super().__init__()
        torch.manual_seed(seed)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.gc = nn.Conv2d(8, 8, 3, padding=1, groups=2)
        self.pw = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        return self.pw(self.gc(F.relu(self.dw(x))))


class Residual(nn.Module):
    def __init__(self, seed=43):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc2(self.fc1(x) + x)


class ActMLP(nn.Module):
    def __init__(self, act, seed=47):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, 32)
        self.act = act

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class U8IO(nn.Module):
    """uint8 model IO, spelled as convert_pt2e would: pairs with float32
    constants around an int8 per-channel Linear."""

    SI, ZI, SO, ZO = float(F32(1 / 128)), 64, float(F32(0.0234)), 131

    def __init__(self, seed=53):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("c", torch.randint(-127, 128, (24, 64), generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(24, generator=g) * 2e-3 + 2e-3)
        self.register_buffer("z", torch.zeros(24, dtype=torch.int64))
        self.register_buffer("b", torch.randn(24, generator=g) * 0.1)

    def forward(self, x):
        qd = ea._decomposed()
        u8 = (0, 255, torch.uint8)
        d = qd.dequantize_per_tensor(
            qd.quantize_per_tensor(x, self.SI, self.ZI, *u8), self.SI, self.ZI,
            *u8)
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -127, 127,
                                      torch.int8)
        y = F.linear(d, w, self.b)
        return qd.dequantize_per_tensor(
            qd.quantize_per_tensor(y, self.SO, self.ZO, *u8), self.SO, self.ZO,
            *u8)


class HardsigCat(nn.Module):
    """Linear + hardsigmoid whose quantised value feeds the next Linear and,
    sliced, the output: the Linear writes it through its output
    quantisation, after the hardsigmoid."""

    def __init__(self, seed=67):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, 32)

    def forward(self, x):
        h = F.hardsigmoid(self.fc1(x))
        return torch.cat([self.fc2(h), h[:, :16]], dim=-1)


class FusedActs(nn.Module):
    """Static pairs with float32 constants (chosen once from the float
    model on the calibration batches) around two int8 per-channel Linears,
    each with its activation fused before the pair that follows it."""

    def __init__(self, seed=61):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        for name, (n, k) in (("1", (48, 64)), ("2", (32, 48))):
            self.register_buffer(f"c{name}", torch.randint(
                -127, 128, (n, k), generator=g, dtype=torch.int8))
            self.register_buffer(f"s{name}", torch.rand(n, generator=g) * 4e-3
                                 + 2e-3)
            self.register_buffer(f"z{name}", torch.zeros(n, dtype=torch.int64))
            self.register_buffer(f"b{name}", torch.randn(n, generator=g) * 0.1)
        self.qp = None
        float_out = []
        with torch.no_grad():
            for (x,) in calibration((4, 64)):
                float_out.append(self._float(x))
        self.qp = [self._qparams(torch.cat([o[i].reshape(-1) for o in
                                            float_out])) for i in range(3)]

    @staticmethod
    def _qparams(t):
        lo, hi = min(float(t.min()), 0.0), max(float(t.max()), 0.0)
        s = float(F32((hi - lo) / 255.0))
        return s, int(np.clip(np.rint(-128 - lo / s), -128, 127))

    def _w(self, i):
        qd = ea._decomposed()
        return qd.dequantize_per_channel(getattr(self, f"c{i}"),
                                         getattr(self, f"s{i}"),
                                         getattr(self, f"z{i}"), 0, -127, 127,
                                         torch.int8)

    def _float(self, x):
        h = F.hardswish(F.linear(x, self._w(1), self.b1))
        y = F.leaky_relu(F.linear(h, self._w(2), self.b2), 0.0625)
        return x, h, y

    def forward(self, x):
        qd = ea._decomposed()

        def pair(t, i):
            s, z = self.qp[i]
            q = qd.quantize_per_tensor(t, s, z, -128, 127, torch.int8)
            return qd.dequantize_per_tensor(q, s, z, -128, 127, torch.int8)

        h = pair(F.hardswish(F.linear(pair(x, 0), self._w(1), self.b1)), 1)
        y = pair(F.leaky_relu(F.linear(h, self._w(2), self.b2), 0.0625), 2)
        # h's dequantised value is also an output: the first Linear writes it,
        # through its output quantisation, after the hardswish.
        return torch.cat([y, h[:, :16]], dim=-1)


class PadConvs(nn.Module):
    """1x1, ReLU, a padded 3x3 (dense or depthwise), 1x1: what the padding
    as the input zero point reads, here behind a channels-last input."""

    def __init__(self, groups=1, flat=False, seed=71):
        super().__init__()
        torch.manual_seed(seed)
        self.expand = nn.Conv2d(4, 8, 1)
        self.padded = nn.Conv2d(8, 8, 3, padding=1, groups=groups)
        self.project = nn.Conv2d(8, 4, 1)
        self.flat = flat

    def forward(self, x):
        y = self.project(self.padded(F.relu(self.expand(x))))
        return y.flatten(1) if self.flat else y


# --- the ops a stock recipe leaves between delegates ------------------------------------

class CnnOps(nn.Module):
    """Convolutions with the ops the stock recipes leave between them: relu6
    (a hardtanh), max_pool2d, a cat of two branches -- one through a ReLU,
    which a8w8 fuses and a16w8 leaves portable -- avg_pool2d and a constant
    pad (its zero quantises to the zero point)."""

    def __init__(self, seed=83):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.c3 = nn.Conv2d(8, 4, 3, padding=1)
        self.c4 = nn.Conv2d(8, 4, 1)
        self.c5 = nn.Conv2d(8, 4, 3)

    def forward(self, x):                                   # [1, 4, 12, 12]
        h = F.max_pool2d(self.c2(F.relu6(self.c1(x))), 2)   # [1, 8, 6, 6]
        h = torch.cat([F.relu(self.c3(h)), self.c4(h)], dim=1)
        h = F.pad(F.avg_pool2d(h, 2), (1, 1, 1, 1))         # [1, 8, 5, 5]
        return self.c5(h)                                   # [1, 4, 3, 3]


class RowMajorOps(nn.Module):
    """The same ops on row-major images (Cortex-M's conv patterns need
    channels-last ones): a Linear's output viewed as an image,
    relu6, max_pool2d and avg_pool2d side by side, a cat, a constant pad and
    a permute, then a Linear."""

    def __init__(self, seed=89):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(72, 16)

    def forward(self, x):                                   # [2, 32]
        h = F.relu6(self.fc1(x)).reshape(2, 4, 4, 4)
        h = torch.cat([F.max_pool2d(h, 2), F.avg_pool2d(h, 2)], dim=1)
        h = F.pad(h, (1, 0, 0, 1))                          # [2, 8, 3, 3]
        return self.fc2(h.permute(0, 2, 3, 1).reshape(2, 72))


class BinaryOps(nn.Module):
    """Two Linears' outputs combined by add, sub, mul and div, each between
    quantised tensors as the recipe leaves them (Arm spells the division as a
    reciprocal and a multiply), then a Linear. The divisor is a ReLU plus
    one half: positive."""

    def __init__(self, seed=97):
        super().__init__()
        torch.manual_seed(seed)
        self.a = nn.Linear(32, 24)
        self.b = nn.Linear(32, 24)
        self.fc = nn.Linear(24, 16)

    def forward(self, x):                                   # [4, 32]
        ha, hb = self.a(x), self.b(x)
        m = (ha + hb) * (ha - hb)
        return self.fc(m / (F.relu(hb) + 0.5))


class Chains(nn.Module):
    """The activations whose portable kernels are ATen's arithmetic exactly:
    hardswish and hardsigmoid (to_edge's
    add, clamp, mul and div), leaky ReLU, then -- with `roots` -- sqrt,
    rsqrt and reciprocal of a ReLU plus one quarter (positive), each between
    Linears. A row lowers it with quant_luts=False, or under a16w8, where no
    table applies: the ops stay portable."""

    def __init__(self, roots=True, seed=101):
        super().__init__()
        torch.manual_seed(seed)
        self.roots = roots
        self.f = nn.ModuleList([nn.Linear(32, 24)] + [
            nn.Linear(24, 24) for _ in range(5 if roots else 2)]
            + [nn.Linear(24, 16)])

    def forward(self, x):                                   # [4, 32]
        f = iter(self.f)
        h = F.hardswish(next(f)(x))
        h = F.hardsigmoid(next(f)(h))
        h = F.leaky_relu(next(f)(h), 0.125)
        if self.roots:
            h = torch.sqrt(F.relu(next(f)(h)) + 0.25)
            h = torch.rsqrt(F.relu(next(f)(h)) + 0.25)
            h = torch.reciprocal(F.relu(next(f)(h)) + 0.25)
        return next(f)(h)


class LlmPieces(nn.Module):
    """The pieces an LLM graph leaves outside its delegates under Arm
    (the census's arm-a8w8 llama3_2), between two
    Linears: rotate-half (slice, neg, cat), a table gathered at positions
    from a cumulative sum (cumsum, index), a mask from token ids (remainder,
    eq, logical_not, bitwise_and) selecting between the activations and a
    full tensor (where, full_like, unsqueeze), a max subtracted (amax, sub)
    and a broadcast (unsqueeze, expand). ids is the device's int64 input
    rule, (7j + 3) mod 11."""

    MODULUS = 11

    def __init__(self, seed=107):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.register_buffer("table", torch.randn(16, 16))

    def forward(self, x, ids):                  # x [1, 8, 64], ids [1, 8] int64
        h = self.fc1(x)                                       # [1, 8, 32]
        rot = torch.cat([-h[..., 16:], h[..., :16]], dim=-1)
        mask = (ids % 2) == 0
        pos = torch.cumsum(mask.to(torch.int64), dim=-1)
        cs = self.table[pos.clamp(max=15)]                    # [1, 8, 16]
        keep = torch.logical_not(mask) & ((ids % 3) == 0)
        g = torch.where(keep.unsqueeze(-1), h * torch.cat([cs, cs], dim=-1)
                        + rot, torch.full_like(h, -1.0))
        g = g - g.amax(dim=-1, keepdim=True)
        e = g.unsqueeze(1).expand(1, 2, 8, 32).reshape(1, 16, 32)[:, ::2]
        return self.fc2(e)


class Reduce(nn.Module):
    """A Linear, then one reduction or normalisation whose float arithmetic
    differs from ATen's in order: its
    output is the row's output, at most one code from eager's. With
    `consumer`, a second Linear reads it too and is a second output: where
    the recipe quantises the op's output nowhere, the code is the one that
    Linear's input quantiser gives (grid_of finds it). `width` is the op's
    output's last dimension, n unless the op reduces it (a mean over it: 1)."""

    def __init__(self, op, k=32, n=24, consumer=False, width=None, seed=109):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(k, n)
        self.op = op
        self.after = nn.Linear(width or n, 16) if consumer else None

    def forward(self, x):
        y = self.op(self.fc(x))
        return y if self.after is None else (y, self.after(y))


class ConvReduce(nn.Module):
    """Reduce over a convolution's output: adaptive_avg_pool2d, group_norm."""

    def __init__(self, op, consumer=False, seed=113):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.op = op
        self.after = nn.Conv2d(8, 4, 1) if consumer else None

    def forward(self, x):                                   # [1, 4, 6, 6]
        y = self.op(self.conv(x))
        return y if self.after is None else (y, self.after(y))


class Bmm(nn.Module):
    """A batched matmul of two activations (queries and keys from two
    Linears): portable fp32 in Stage A."""

    def __init__(self, consumer=False, seed=127):
        super().__init__()
        torch.manual_seed(seed)
        self.q = nn.Linear(32, 16)
        self.k = nn.Linear(32, 16)
        self.after = nn.Linear(8, 16) if consumer else None

    def forward(self, x):                                   # [2, 8, 32]
        y = torch.bmm(self.q(x), self.k(x).transpose(1, 2))   # [2, 8, 8]
        return y if self.after is None else (y, self.after(y))


class ActChain(nn.Module):
    """gelu (erf), gelu (tanh), tanh, SiLU and hardswish between Linears,
    each a table. Three rows of widths 26,
    22, 18, 14 and 10: every activation's length is 2 or 6 past a multiple
    of eight, so eager runs its last elements in torch's scalar tail."""

    def __init__(self, seed=131):
        super().__init__()
        torch.manual_seed(seed)
        widths = (32, 26, 22, 18, 14, 10, 16)
        self.f = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths,
                                                                 widths[1:]))

    def forward(self, x):                                   # [3, 32]
        h = F.gelu(self.f[0](x))
        h = F.gelu(self.f[1](h), approximate="tanh")
        h = torch.tanh(self.f[2](h))
        h = F.silu(self.f[3](h))
        h = F.hardswish(self.f[4](h))
        return self.f[5](h)


class FloatLeft(nn.Module):
    """The weights XNNPACK's static per-channel recipe leaves float: it
    quantises no embedding and no addmm, so a table and HF's Conv1D (GPT-2's
    Linear: x @ W + b as one addmm) reach the backend float, as in the
    census's xnnpack-static gpt2 and llama3_2. The backend keeps them float:
    the gather portable at fp32, the Conv1D in the delegate's fp32 loop. The
    recipe quantises the add of the two and the Linear reading it. The table
    and the Conv1D's weights are dyadic, like the ramp, so every fp32 sum is
    exact in any order, fused or not: device, host, mirror and eager agree
    bit for bit. ids is the device's int64 input rule, (7j + 3) mod 16."""

    MODULUS = 16

    def __init__(self, seed=113):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def dyadic(*shape, den):
            return torch.randint(-8, 9, shape, generator=g).float() / den

        torch.manual_seed(seed)
        self.table = nn.Embedding(self.MODULUS, 32)
        with torch.no_grad():
            self.table.weight.copy_(dyadic(self.MODULUS, 32, den=8))
        self.conv1d_w = nn.Parameter(dyadic(32, 32, den=16))  # [in, out], as HF
        self.conv1d_b = nn.Parameter(dyadic(32, den=16))
        self.fc = nn.Linear(32, 16)

    def forward(self, x, ids):                  # x [1, 8, 32], ids [1, 8] int64
        h = torch.addmm(self.conv1d_b, x.view(-1, 32), self.conv1d_w)
        return self.fc(h.view(1, 8, 32) + self.table(ids))


def _lossless_fc(fc, g):
    """A Linear(256, n) XNNPACK's dynamic recipe quantises without loss on
    the ramp: one token of 256 ramp values spans exactly 255 steps of 2^-7
    (scale 2^-7, zero point -64), and each weight row is k/128 with a +-127
    in it (per-channel scale 2^-7), so the recipe's model, the device and
    every mirror compute the same exact sum."""
    with torch.no_grad():
        w = torch.randint(-127, 128, fc.weight.shape, generator=g)
        w[:, 0] = 127
        fc.weight.copy_(w.float() / 128)
        fc.bias.copy_(torch.randint(-8, 9, fc.bias.shape, generator=g).float()
                      / 16)


class ConvLeft(nn.Module):
    """The convolutions XNNPACK's dynamic recipe leaves float, as in the
    census's CNN pairs (its dynamic config quantises a conv with no relu
    after it, depthwise aside): a dense conv and its relu into a depthwise
    conv, each kept float in the delegate's fp32 convolution. The recipe quantises the Linear (the
    second output), which makes the program recipe-quantised. Dyadic weights
    on the ramp: every fp32 sum is exact in any order, fused or not, so
    device, host, mirror and eager agree bit for bit."""

    def __init__(self, seed=127):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def dyadic(*shape):
            return torch.randint(-8, 9, shape, generator=g).float() / 16

        torch.manual_seed(seed)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        with torch.no_grad():
            for conv in (self.conv, self.dw):
                conv.weight.copy_(dyadic(*conv.weight.shape))
                conv.bias.copy_(dyadic(*conv.bias.shape))
        self.fc = nn.Linear(256, 4)
        _lossless_fc(self.fc, g)

    def forward(self, x):                       # x [1, 256]
        return self.dw(torch.relu(self.conv(x.view(1, 4, 8, 8)))), self.fc(x)


class HalfLeft(nn.Module):
    """A half checkpoint's weights reaching the backend float: Linears
    written as matmuls (the recipe quantises nn.Linear, not mm), bf16 and
    fp16, a bf16 convolution into a relu (the recipe quantises a conv without
    one) and two tables, fp16 and bf16, each upcast as a
    loaded checkpoint is (w.to(float32), which promote_fp8_constants folds).
    The Linear the recipe does quantise makes the program recipe-quantised,
    so each keeps its type: every 2-byte rung, matmul, conv and gather.
    Dyadic values the half types hold exactly, on the ramp: exact
    everywhere. ids is the device's int64 input rule, (7j + 3) mod 16."""

    MODULUS = 16

    def __init__(self, seed=131):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def dyadic(*shape, den):
            return torch.randint(-8, 9, shape, generator=g).float() / den

        torch.manual_seed(seed)
        self.lin_w = nn.Parameter(dyadic(8, 32, den=16).bfloat16())
        self.lin2_w = nn.Parameter(dyadic(8, 32, den=16).half())
        self.conv_w = nn.Parameter(dyadic(4, 4, 3, 3, den=16).bfloat16())
        self.conv_b = nn.Parameter(dyadic(4, den=16))
        self.table = nn.Parameter(dyadic(self.MODULUS, 8, den=8).half())
        self.table2 = nn.Parameter(dyadic(self.MODULUS, 8, den=8).bfloat16())
        self.fc = nn.Linear(256, 4)
        _lossless_fc(self.fc, g)

    def forward(self, x, ids):                  # x [1, 256], ids [1, 8] int64
        rows = x.view(8, 32)
        lin = rows @ self.lin_w.to(torch.float32).t()
        lin2 = rows @ self.lin2_w.to(torch.float32).t()
        conv = F.relu(F.conv2d(x.view(1, 4, 8, 8),
                               self.conv_w.to(torch.float32), self.conv_b,
                               padding=1))
        emb = F.embedding(ids, self.table.to(torch.float32))
        emb2 = F.embedding(ids, self.table2.to(torch.float32))
        return lin, lin2, conv, emb, emb2, self.fc(x)


class SpeedOps(nn.Module):
    """The six speed-only tables on every code
    of an input grid whose value is finite, once with an output pair after
    the op (V_q) and once without (V_raw), between hand-written pairs with
    power-of-two float32 scales. The ramp, j/128 - 1/2, becomes code
    lo + j by exact steps (a slice, then (x + 1/2) * 128 s1 + (lo - z1) s1).
    Lowered twice -- as tables (st_spd_lut) and as the portable chains they
    replace (st_spd_chain) -- both are held to one reference, and the export
    asserts that the tables' mirror equals the chains' (torch's own ops) bit
    for bit."""

    # (op, s1, z1, qmin, qmax, dtype, first code, codes, (s2, z2) for V_q)
    SITES = (
        ("hardswish", 1 / 16, 0, -128, 127, torch.int8, -128, 256, (1 / 16, -122)),
        ("hardsigmoid", 1 / 16, 0, -128, 127, torch.int8, -128, 256, (1 / 256, -128)),
        ("leaky_relu", 1 / 16, 5, -128, 127, torch.int8, -128, 256, (1 / 16, 0)),
        ("sqrt", 1 / 16, 0, 0, 255, torch.uint8, 0, 256, (1 / 64, -128)),
        # rsqrt and reciprocal skip the code whose value is 0 (infinity, which
        # ExecuTorch's quantize kernel leaves undefined).
        ("rsqrt", 1 / 16, 0, 0, 255, torch.uint8, 1, 255, (1 / 32, -128)),
        ("reciprocal", 1 / 16, 0, -128, 127, torch.int8, 1, 127, (1 / 8, 0)),
        ("reciprocal", 1 / 16, 0, -128, 127, torch.int8, -128, 128, (1 / 8, 0)),
    )
    FNS = {"hardswish": F.hardswish, "hardsigmoid": F.hardsigmoid,
           "leaky_relu": lambda x: F.leaky_relu(x, 0.1), "sqrt": torch.sqrt,
           "rsqrt": torch.rsqrt, "reciprocal": torch.reciprocal}

    RAW = (0, 3, 4, 5)   # the sites run again without the output pair

    def forward(self, x):                                   # [256]
        qd = ea._decomposed()
        outs = []
        for raw in (False, True):
            for i, (op, s1, z1, qmin, qmax, dtype, lo, n, (s2, z2)) in \
                    enumerate(self.SITES):
                if raw and i not in self.RAW:
                    continue
                v = (x[:n] + 0.5) * (128 * s1) + (lo - z1) * s1
                d = qd.dequantize_per_tensor(
                    qd.quantize_per_tensor(v, s1, z1, qmin, qmax, dtype), s1, z1,
                    qmin, qmax, dtype)
                y = self.FNS[op](d)
                if not raw:
                    i8 = (-128, 127, torch.int8)
                    y = qd.dequantize_per_tensor(
                        qd.quantize_per_tensor(y, s2, z2, *i8), s2, z2, *i8)
                outs.append(y)
        return tuple(outs)


@dataclass
class Row:
    name: str
    recipe: str          # a quant_census recipe, or "" for a hand-written graph
    build: object        # () -> float nn.Module
    shape: tuple
    luts: bool = True    # lower_esp32(quant_luts=...)
    portable: tuple = () # ops the row keeps outside its delegates on purpose
    channels_last: bool = False   # the input is NHWC in memory (cl_ramp)
    # "exact": the device's output bits are the mirror's. "code": the first
    # output is held to at most one code of its grid (grid_of) against the
    # mirror, element by element.
    bar: str = "exact"
    long_shape: tuple = ()   # an int64 second input: (7j + 3) mod long_modulus
    long_modulus: int = 0


VIEWS = ("aten.view_copy",)
# make_inputs_row_major_pass.py's copies of a channels-last input to row-major and back.
DIM_ORDER = "dim_order_ops._to_dim_order_copy"
QDQ = ("quantize_per_tensor", "dequantize_per_tensor")
CNN = ("aten.max_pool2d_with_indices", "aten.cat", "aten.avg_pool2d",
       "aten.constant_pad_nd", "aten.hardtanh")
ROW_MAJOR = VIEWS + CNN + ("aten.permute_copy",)
BINARY = ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor",
          "aten.div.Tensor")
CHAINS = BINARY + ("aten.clamp", "aten.leaky_relu", "aten.sqrt", "aten.rsqrt",
                   "aten.reciprocal")
LLM = VIEWS + BINARY + (
    "aten.remainder", "aten.eq", "aten.logical_not",
    "aten.bitwise_and", "aten.cumsum", "aten.clamp", "aten.index.Tensor",
    "aten.slice_copy", "aten.neg", "aten.cat", "aten.full", "aten.where",
    "aten.amax", "aten.unsqueeze_copy", "aten.expand_copy",
    "dim_order_ops._to_dim_order_copy")
LUTS = ("espressif.quant_lut",)
# What the defaults put between delegates wherever a row's graph allows:
# Stage B's code tables, the fused norms and the int16 tables. Kept in every
# row; their own rows are asymstgb's, asymnorm's and asymlut16's.
DEFAULT_OPS = ("espressif.quant_lut_codes", "espressif.qnorm", "espressif.lut16")
SPEED_OPS = ("aten.slice_copy", "aten.add.Tensor", "aten.mul.Tensor")
SPECS = (
    Row("st_xnn_mlp", "xnnpack-static", MLP, (4, 64), portable=VIEWS),
    Row("st_xnn_mlp_pt", "xnnpack-static-pt", MLP, (4, 64), portable=VIEWS),
    Row("st_xnnqat_mlp", "xnnpack-qat", MLP, (4, 64), portable=VIEWS),
    Row("st_arm_mlp", "arm-a8w8", MLP, (4, 64), portable=VIEWS),
    Row("st_cm_mlp", "cortex-m", MLP, (4, 64), portable=VIEWS),
    Row("st_xnn_conv", "xnnpack-static", ConvNet, (1, 4, 9, 9)),
    Row("st_arm_dwconv", "arm-a8w8", DwGrouped, (1, 8, 7, 7)),
    Row("st_xnn_residual", "xnnpack-static", Residual, (4, 64),
        portable=VIEWS + ("aten.add.Tensor", "quantize_per_tensor",
                          "dequantize_per_tensor")),
    Row("st_xnn_sig_lut", "xnnpack-static", lambda: ActMLP(torch.sigmoid),
        (4, 64), portable=VIEWS + ("espressif.quant_lut",)),
    Row("st_arm_leaky", "arm-a8w8",
        lambda: ActMLP(lambda h: F.leaky_relu(h, 0.125)), (4, 64), luts=False,
        portable=VIEWS + ("aten.leaky_relu",)),
    Row("st_u8_io", "", U8IO, (4, 64)),
    Row("st_arm_u8io", "arm-a8w8-u8io", MLP, (4, 64)),
    Row("st_cm_hardsig", "cortex-m", HardsigCat, (4, 64),
        portable=("aten.cat", "aten.slice_copy", "quantize_per_tensor",
                  "dequantize_per_tensor")),
    Row("st_fused_acts", "", FusedActs, (4, 64),
        portable=("aten.cat", "aten.slice_copy")),
    Row("st_cm_cl_conv", "cortex-m", PadConvs, (1, 4, 6, 6),
        portable=(DIM_ORDER,), channels_last=True),
    Row("st_cm_cl_dw", "cortex-m", lambda: PadConvs(groups=8, flat=True),
        (1, 4, 6, 6), portable=(DIM_ORDER, "aten.view_copy",
                                "quantize_per_tensor", "dequantize_per_tensor"),
        channels_last=True),
    # Exact: portable kernels that perform ATen's arithmetic, given identical
    # input codes.
    Row("st_xnn_cnn_ops", "xnnpack-static", CnnOps, (1, 4, 12, 12),
        portable=CNN + QDQ),
    Row("st_arm_cnn_ops", "arm-a8w8", CnnOps, (1, 4, 12, 12),
        portable=CNN + QDQ),
    Row("st_a16_cnn_ops", "arm-a16w8", CnnOps, (1, 4, 12, 12),
        portable=CNN + QDQ + ("aten.relu",)),
    Row("st_cm_ops", "cortex-m", RowMajorOps, (2, 32),
        portable=ROW_MAJOR + QDQ),
    Row("st_xnn_binary", "xnnpack-static", BinaryOps, (4, 32),
        portable=BINARY + QDQ + ("aten.relu",)),
    Row("st_arm_binary", "arm-a8w8", BinaryOps, (4, 32),
        portable=BINARY + QDQ + LUTS + ("aten.relu",)),
    Row("st_cm_binary", "cortex-m", BinaryOps, (4, 32),
        portable=BINARY + QDQ + ("aten.relu",)),
    Row("st_a16_binary", "arm-a16w8", BinaryOps, (4, 32),
        portable=BINARY + QDQ + ("aten.relu", "aten.reciprocal")),
    Row("st_xnn_chains", "xnnpack-static", Chains, (4, 32), luts=False,
        portable=CHAINS + QDQ),
    Row("st_arm_chains", "arm-a8w8", lambda: Chains(roots=False), (4, 32),
        luts=False, portable=CHAINS + QDQ),
    Row("st_cm_chains", "cortex-m", Chains, (4, 32), luts=False,
        portable=CHAINS + QDQ),
    Row("st_a16_chains", "arm-a16w8", lambda: Chains(roots=False), (4, 32),
        portable=CHAINS + QDQ + ("aten.relu",)),
    Row("st_arm_llm_ops", "arm-a8w8", LlmPieces, (1, 8, 64), portable=LLM + QDQ,
        long_shape=(1, 8), long_modulus=LlmPieces.MODULUS),
    # At most one code: float reductions in another order than ATen's.
    Row("st_xnn_mean", "xnnpack-static",
        lambda: Reduce(lambda h: h.mean(dim=-1, keepdim=True)), (4, 8, 32),
        portable=VIEWS + ("aten.mean",) + QDQ, bar="code"),
    Row("st_xnn_adapool", "xnnpack-static",
        lambda: ConvReduce(lambda h: F.adaptive_avg_pool2d(h, 2)),
        (1, 4, 6, 6), portable=("aten._adaptive_avg_pool2d",) + QDQ,
        bar="code"),
    Row("st_cm_mean", "cortex-m",
        lambda: Reduce(lambda h: h.mean(dim=-1, keepdim=True), consumer=True,
                       width=1), (4, 8, 32),
        portable=VIEWS + ("aten.mean",) + QDQ, bar="code"),
    Row("st_arm_sum", "arm-a8w8",
        lambda: Reduce(lambda h: h.sum(dim=-1, keepdim=True)), (4, 8, 32),
        portable=VIEWS + ("aten.sum",) + QDQ, bar="code"),
    Row("st_xnn_softmax", "xnnpack-static",
        lambda: Reduce(lambda h: F.softmax(h, dim=-1), consumer=True), (4, 32),
        portable=("aten._softmax",) + QDQ, bar="code"),
    Row("st_arm_softmax", "arm-a8w8",
        lambda: Reduce(lambda h: F.softmax(h, dim=-1)), (4, 32),
        portable=("aten.amax", "aten.sub.Tensor", "aten.sum", "aten.mul.Tensor")
        + LUTS + QDQ, bar="code"),
    Row("st_cm_softmax", "cortex-m",
        lambda: Reduce(lambda h: F.softmax(h, dim=-1)), (4, 32),
        portable=("aten._softmax",) + QDQ, bar="code"),
    Row("st_a16_softmax", "arm-a16w8",
        lambda: Reduce(lambda h: F.softmax(h, dim=-1)), (4, 32),
        portable=("aten.amax", "aten.sub.Tensor", "aten.sum", "aten.mul.Tensor",
                  "aten.exp", "aten.reciprocal") + QDQ, bar="code"),
    Row("st_xnn_layernorm", "xnnpack-static",
        lambda: Reduce(nn.LayerNorm(24), consumer=True), (4, 32),
        portable=("aten.native_layer_norm",) + QDQ, bar="code"),
    Row("st_arm_layernorm", "arm-a8w8", lambda: Reduce(nn.LayerNorm(24)),
        (4, 32), portable=VIEWS + BINARY + ("aten.sum",) + LUTS + QDQ,
        bar="code"),
    Row("st_cm_layernorm", "cortex-m",
        lambda: Reduce(nn.LayerNorm(24), consumer=True), (4, 32),
        portable=("aten.native_layer_norm",) + QDQ, bar="code"),
    Row("st_xnn_groupnorm", "xnnpack-static",
        lambda: ConvReduce(nn.GroupNorm(2, 8), consumer=True), (1, 4, 6, 6),
        portable=("aten.native_group_norm",) + QDQ, bar="code"),
    Row("st_xnn_bmm", "xnnpack-static", lambda: Bmm(consumer=True), (2, 8, 32),
        portable=VIEWS + ("aten.permute_copy", "aten.bmm") + QDQ, bar="code"),
    Row("st_arm_bmm", "arm-a8w8", Bmm, (2, 8, 32),
        portable=VIEWS + ("aten.permute_copy", "aten.bmm") + QDQ, bar="code"),
    Row("st_cm_bmm", "cortex-m", Bmm, (2, 8, 32),
        portable=VIEWS + ("aten.permute_copy", "aten.bmm") + QDQ, bar="code"),
    # Tables between delegates.
    Row("st_xnn_act_luts", "xnnpack-static", ActChain, (3, 32),
        portable=LUTS),
    Row("st_arm_act_luts", "arm-a8w8", ActChain, (3, 32), portable=LUTS),
    Row("st_cm_act_luts", "cortex-m", ActChain, (3, 32), portable=LUTS),
    # The speed-only tables against the portable chains they replace.
    Row("st_spd_chain", "", SpeedOps, (256,), luts=False,
        portable=SPEED_OPS + QDQ + CHAINS),
    Row("st_spd_lut", "", SpeedOps, (256,), portable=SPEED_OPS + LUTS),
    # The table and the Conv1D the recipe left float, kept float.
    Row("st_xnn_float_w", "xnnpack-static", FloatLeft, (1, 8, 32),
        portable=VIEWS + ("aten.embedding", "aten.add.Tensor") + QDQ,
        long_shape=(1, 8), long_modulus=FloatLeft.MODULUS),
    Row("st_xnn_conv_f32", "xnnpack-dynamic", ConvLeft, (1, 256),
        portable=VIEWS),
    Row("st_xnn_half", "xnnpack-dynamic", HalfLeft, (1, 256),
        portable=VIEWS + ("aten.permute_copy",), long_shape=(1, 8),
        long_modulus=HalfLeft.MODULUS),
)
assert tuple(r.name for r in SPECS) == ROWS
# The speed rows' two programs answer to one reference (fixtures() asserts it).
SPEED_ROWS = ("st_spd_chain", "st_spd_lut")


def calibration(shape, seed=59, extra=()):
    """Eight batches over the device's ramp's range and past it, so that the
    ramp the device feeds sits inside the calibrated grid. `extra` follows the
    float input in every batch (a row's int64 input)."""
    g = torch.Generator().manual_seed(seed)
    batches = [(ramp(shape), *extra)]
    for _ in range(7):
        batches.append((torch.randn(shape, generator=g) * 0.9 + 0.5, *extra))
    return batches


def row_inputs(row):
    """The device's inputs: the float ramp, and the int64 rule's tensor for a
    row with one."""
    x = cl_ramp(row.shape) if row.channels_last else ramp(row.shape)
    if not row.long_modulus:
        return (x,)
    n = int(np.prod(row.long_shape))
    return (x, ea.long_input(n, row.long_modulus).reshape(row.long_shape))


def _arm_uint8_io(net, inputs, batches):
    """Arm a8w8 with uint8 model inputs and outputs, as Arm ships it."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    recipes.tosa_serializer_source()
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
        get_uint8_io_quantization_config,
    )

    quantizer = (EthosUQuantizer(EthosUCompileSpec("ethos-u55-128"))
                 .set_global(get_symmetric_quantization_config())
                 .set_io(get_uint8_io_quantization_config()))
    return recipes._pt2e(net, inputs, batches, quantizer, False)


def cl_ramp(shape):
    """main.cpp's ramp written into a channels-last [N, C, H, W] input: it
    fills memory, NHWC, so element (n, c, h, w) is the ramp's
    ((n * H + h) * W + w) * C + c, and the tensor keeps those strides."""
    n, c, h, w = shape
    return ramp((n, h, w, c)).permute(0, 3, 1, 2)


def memory_order(t):
    """`t`'s values in the order its strides lay them out (NHWC for a
    channels-last image): what the device reads and writes."""
    order = sorted(range(t.dim()), key=lambda d: -t.stride(d))
    return t.permute(*order).contiguous()


def quantized(row):
    """(converted module, its example inputs, the recipe's info)."""
    inputs = row_inputs(row)
    if not row.recipe:
        return row.build().eval(), inputs, {}
    batches = calibration(row.shape, extra=inputs[1:])
    if row.recipe == "arm-a8w8-u8io":
        return _arm_uint8_io(row.build().eval(), inputs, batches), inputs, {}
    from executorch.backends.esp32s3.test.quant_census import recipes

    q = recipes.quantize(row.recipe, row.build().eval(), inputs, batches)
    return q.program, q.inputs, q.info


# --- the device's arithmetic ------------------------------------------------------------

_MOVES = ("view", "reshape", "flatten", "squeeze", "unsqueeze", "permute",
          "clone", "contiguous", "t.default", "transpose")


def _is(node, name):
    return (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and name in str(node.target))


def _behind_moves(node):
    while isinstance(node, torch.fx.Node) and node.op == "call_function" and any(
            m in str(node.target) for m in _MOVES):
        node = node.args[0]
    return node


class StaticMirror(ByDefinition):
    """The converted graph as the device computes it: matmuls and convolutions
    from the codes of the static pair in front of each (the exact integer dot
    and the device's epilogue), a Linear or addmm whose weight the recipe left
    float as the delegate's fp32 loop (models.addmm_k_outer), everything else
    as torch computes it but sqrt, which the device computes correctly rounded
    and torch does not on every machine (ByDefinition: fold_quant_lut_pass.EXACT,
    a table's definition too). A quantised op with no static pair in front of
    it is refused: every fixture's is quantised."""

    MATMULS = ("aten.linear.default", "aten.conv2d.default")
    # The default policy keeps a weight the recipe left float; a mirror of a
    # row lowered at an explicit policy quantises it (asym_a16).
    KEEP_FLOAT = True
    # A dynamic recipe's row (st_xnn_conv_f32, st_xnn_half): its quantised
    # Linear is computed as torch computes it, which is the device's sum only
    # because those rows quantise without loss (_lossless_fc); eager_gap's
    # max abs 0 holds the row to it.
    DYNAMIC_EAGER = False

    def run_node(self, n):
        self._node = n
        return super().run_node(n)

    def _act_scheme(self, value):
        d = _behind_moves(value)
        if not _is(d, "dequantize_per_tensor"):
            raise RuntimeError(f"{self._node.name}: no static pair in front")
        s, z, qmin, qmax, dtype = d.args[1:6]
        return qm.ActScheme("A5", "static", "static", "rne_before_zp",
                            int(qmin), int(qmax), scale=float(s),
                            zero_point=int(z))

    def _weight(self, value, n_out):
        """(codes [N, ...] int64, scales [N] float32, zero points or None)."""
        d = _behind_moves(value)
        codes = self.env[d.args[0]] if isinstance(d.args[0], torch.fx.Node) \
            else d.args[0]
        codes = codes.to(torch.int64).numpy()
        if _is(d, "dequantize_per_channel"):
            scales = self.env[d.args[1]].to(torch.float32).numpy()
            zeros = self.env[d.args[2]] if d.args[2] is not None else None
            zeros = None if zeros is None else zeros.to(torch.int64).numpy()
        elif _is(d, "dequantize_per_tensor"):
            scales = np.full(n_out, F32(d.args[1]), F32)
            zeros = np.full(n_out, int(d.args[2]), np.int64)
        else:
            raise RuntimeError(f"{self._node.name}: weight is not quantised")
        if zeros is not None and not np.any(zeros):
            zeros = None
        return codes, scales, zeros

    def call_function(self, target, args, kwargs):
        name = str(target)
        if self.KEEP_FLOAT and name in ("aten.addmm.default",
                                        "aten.linear.default"):
            weight = 2 if name == "aten.addmm.default" else 1
            if not _is(_behind_moves(self._node.args[weight]), "dequantize"):
                return self._fp32_loop(name, args, kwargs)
        if not any(m in name for m in self.MATMULS):
            return super().call_function(target, args, kwargs)
        node = self._node
        weight = _behind_moves(node.args[1])
        if (self.KEEP_FLOAT and "conv2d" in name
                and not _is(weight, "dequantize")):
            return torch.from_numpy(self._float_conv(node, args, kwargs))
        if self.DYNAMIC_EAGER:
            # A dynamic chain's dequantise reads the scale its chooser
            # computed (a node), a static pair's a constant.
            d = _behind_moves(node.args[0])
            if (not _is(d, "dequantize_per_tensor")
                    or isinstance(d.args[1], torch.fx.Node)):
                return super().call_function(target, args, kwargs)
        act = self._act_scheme(node.args[0])
        x = args[0].detach().numpy().astype(F32)
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        bias = None if bias is None else bias.detach().numpy().astype(F32)
        if "linear" in name:
            n_out = int(args[1].shape[0])
            codes, scales, zeros = self._weight(node.args[1], n_out)
            x2 = x.reshape(-1, x.shape[-1])
            y = qm.affine_matmul(x2, codes, scales, bias, act=act, bits=8,
                                 zero_points=zeros).y
            return torch.from_numpy(y.reshape(*x.shape[:-1], n_out))
        return torch.from_numpy(self._conv(node, x, args, bias, act))

    @staticmethod
    def _fp32_loop(name, args, kwargs):
        """A weight the recipe left float, kept float: the delegate's fp32
        loop, the bias then k-outer madd.s steps (models.addmm_k_outer)."""
        from executorch.backends.esp32s3.test.reference import addmm_k_outer

        if name == "aten.addmm.default":
            bias, x, w = (a.detach().numpy() for a in args[:3])
            return torch.from_numpy(addmm_k_outer(x, w, bias))
        x, w = args[0].detach().numpy(), args[1].detach().numpy()
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        bias = (np.zeros(w.shape[0], F32) if bias is None
                else bias.detach().numpy())
        y = addmm_k_outer(x.reshape(-1, x.shape[-1]), w.T, bias)
        return torch.from_numpy(y.reshape(*x.shape[:-1], w.shape[0]))

    def _float_conv(self, node, args, kwargs):
        """A convolution whose weight the recipe left float (or a half
        checkpoint's, widened by its upcast): the delegate's fp32
        convolution, per group (mirrors.float_conv, fused as the device
        builds it)."""
        from executorch.backends.esp32s3.operator_support.conv import split_groups

        x = args[0].detach().numpy().astype(F32)
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        geom = self._geometry(node, x, args[1])
        w = args[1].detach().numpy().astype(F32).reshape(geom.c_out, -1)
        b = (np.zeros(geom.c_out, F32) if bias is None
             else bias.detach().numpy().astype(F32))
        subs = split_groups(geom)
        per = geom.c_out // len(subs) if len(subs) > 1 else geom.c_out
        parts = []
        for sub in subs:
            rows = slice(sub.group_idx * per, sub.group_idx * per + per) \
                if len(subs) > 1 else slice(None)
            parts.append(qm.float_conv(x, w[rows], b[rows], sub))
        y = np.concatenate(parts, axis=1)
        return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)

    def _geometry(self, node, x, w):
        """The ConvGeom of an aten.conv2d node, as _conv reads it."""
        from executorch.backends.esp32s3.operator_support.conv import ConvGeom

        given = list(node.args[3:]) + [None] * 4
        named = dict(zip(("stride", "padding", "dilation", "groups"), given))
        named.update(node.kwargs)

        def pair_of(v, default):
            v = default if v is None else v
            return [int(v)] * 2 if isinstance(v, int) else [int(t) for t in v]

        stride = pair_of(named["stride"], 1)
        padding = pair_of(named["padding"], 0)
        dilation = pair_of(named["dilation"], 1)
        groups = int(named["groups"] or 1)
        c_out, c_in_g, kh, kw = (int(d) for d in w.shape)
        _, c_in, h_in, w_in = x.shape
        h_out = (h_in + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        return ConvGeom(h_in=h_in, w_in=w_in, c_in=c_in_g, h_out=h_out,
                        w_out=w_out, c_out=c_out, kh=kh, kw=kw,
                        stride_h=stride[0], stride_w=stride[1],
                        pad_h=padding[0], pad_w=padding[1], dil_h=dilation[0],
                        dil_w=dilation[1], groups=groups, c_out_total=c_out)

    def _conv(self, node, x, args, bias, act):
        from executorch.backends.esp32s3.operator_support.conv import ConvGeom, split_groups

        w = args[1]
        # aten.conv2d(input, weight, bias, stride=1, padding=0, dilation=1,
        # groups=1): export drops trailing defaults.
        given = list(args[3:]) + [None] * 4
        named = dict(zip(("stride", "padding", "dilation", "groups"), given))
        named.update(kwargs := {k: v for k, v in node.kwargs.items()})

        def pair_of(v, default):
            v = default if v is None else v
            return [int(v)] * 2 if isinstance(v, int) else [int(t) for t in v]

        stride = pair_of(named["stride"], 1)
        padding = pair_of(named["padding"], 0)
        dilation = pair_of(named["dilation"], 1)
        groups = int(named["groups"] or 1)
        c_out, c_in_g, kh, kw = (int(d) for d in w.shape)
        _, c_in, h_in, w_in = x.shape
        h_out = (h_in + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        geom = ConvGeom(h_in=h_in, w_in=w_in, c_in=c_in_g, h_out=h_out,
                        w_out=w_out, c_out=c_out, kh=kh, kw=kw,
                        stride_h=stride[0], stride_w=stride[1],
                        pad_h=padding[0], pad_w=padding[1], dil_h=dilation[0],
                        dil_w=dilation[1], groups=groups, c_out_total=c_out)
        codes, scales, zeros = self._weight(node.args[1], c_out)
        codes = codes.reshape(c_out, -1)
        b = np.zeros(c_out, F32) if bias is None else bias
        subs = split_groups(geom)
        per_out = c_out // len(subs) if len(subs) > 1 else c_out
        parts = []
        for sub in subs:
            lo = sub.group_idx * per_out if len(subs) > 1 else 0
            rows = slice(lo, lo + per_out)
            parts.append(qm.affine_conv(
                x, codes[rows], scales[rows], b[rows], sub, act,
                zero_points=None if zeros is None else zeros[rows]))
        y = np.concatenate(parts, axis=1)
        return y.reshape(1, c_out, h_out, w_out)


def mirrored_program(model, inputs, luts):
    """The program StaticMirror runs: the export lower_esp32 sees, with its
    tables folded in when the row lowers with them (quant_lut.fold_quant_luts:
    the very tables the device looks up)."""
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    ep = torch.export.export(model, inputs, strict=False)
    if luts:
        fold_quant_lut_pass.fold_quant_luts(ep)
    return ep


def flat(out):
    """A program's outputs as main.cpp's all_outputs reads them: one float32
    vector, in output order."""
    return np.concatenate([np.asarray(o.detach().numpy(), F32).reshape(-1)
                           for o in out])


def mirror(model, inputs, luts, dynamic=False):
    ep = mirrored_program(model, inputs, luts)
    with torch.no_grad():
        # Values kept: a matmul reads its weight's codes behind the dequant.
        interp = StaticMirror(ep.module(), garbage_collect_values=False)
        interp.DYNAMIC_EAGER = dynamic
        out = interp.run(*inputs)
    if isinstance(out, (tuple, list)):
        return flat(out)
    return np.asarray(out.detach().numpy(), F32)


class _LastQuantize(torch.fx.Interpreter):
    """Eager, recording the value the graph output's quantize reads (the
    quantize, not its dequantise: "quantize_per_tensor" is a substring of
    "dequantize_per_tensor", whose input is the codes)."""

    def run_node(self, n):
        out = super().run_node(n)
        if _is(n, "quantize_per_tensor") and not _is(n, "dequantize"):
            self.last = (self.env[n.args[0]], n.args[1:6])
        return out


@dataclass
class EagerGap:
    """The row against the recipe's eager model: output codes that flip (in
    the output pair's steps), how close to a tie each sat in eager, and the
    largest difference."""

    flips: int
    steps: int
    max_abs: float
    tie_distances: list = field(default_factory=list)

    def ok(self, bound=1, near=1e-3):
        return self.flips <= bound and all(d <= near for d in self.tie_distances)


def eager_gap(model, inputs, device_out):
    ep = torch.export.export(model, inputs, strict=False)
    rec = _LastQuantize(ep.module(), garbage_collect_values=False)
    with torch.no_grad():
        out = rec.run(*inputs)
    if isinstance(out, (tuple, list)):
        return _outputs_gap(ep, rec, out, device_out)
    eager = out.detach().numpy().astype(F32)
    diff = np.abs(np.asarray(device_out, F32) - eager)
    v, (s, z, qmin, qmax, _) = rec.last
    s = float(s)
    steps = np.rint(diff / F32(s)).astype(np.int64)
    flips = np.flatnonzero(steps.reshape(-1))
    pre = (v.detach().numpy().astype(F32).reshape(-1) * qm.reciprocal(s)).astype(F32)
    ties = [float(abs(abs(pre[i] - np.floor(pre[i])) - 0.5)) for i in flips]
    return EagerGap(int(flips.size), int(steps.max(initial=0)),
                    float(diff.max(initial=0.0)), ties)


def _outputs_gap(ep, rec, out, device_out):
    """eager_gap over several outputs: each in its own grid's steps (grid_of),
    an output with no grid by its largest difference alone."""
    device = np.asarray(device_out, F32).reshape(-1)
    nodes = output_nodes(rec.module)
    flips, steps, max_abs, ties, at = 0, 0, 0.0, [], 0
    for node, value in zip(nodes, out):
        eager = value.detach().numpy().astype(F32).reshape(-1)
        mine = device[at:at + eager.size]
        at += eager.size
        diff = np.abs(mine - eager)
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        grid = grid_of(node)
        if grid is None:
            continue
        s = grid[0]
        step = np.rint(diff / F32(s)).astype(np.int64)
        flipped = np.flatnonzero(step)
        flips += int(flipped.size)
        steps = max(steps, int(step.max(initial=0)))
        pre_node = node.args[0].args[0] if _is(node, "dequantize_per_tensor") \
            else node
        pre = (rec.env[pre_node].detach().numpy().astype(F32).reshape(-1)
               * qm.reciprocal(s)).astype(F32)
        ties += [float(abs(abs(pre[i] - np.floor(pre[i])) - 0.5))
                 for i in flipped]
    return EagerGap(flips, steps, max_abs, ties)


def output_nodes(gm):
    """The graph's output values, in order."""
    out = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    return list(out) if isinstance(out, (tuple, list)) else [out]


def grid_of(node):
    """(scale, zero point, qmin, qmax) of the code an output holds: its own
    pair's when the output is a static dequantise; else the static quantize
    that reads it (XNNPACK, and Cortex-M for a mean or a layer_norm, quantise
    a reduction's output only where the next quantised op reads it); else
    None."""
    def args(q):
        s, z, qmin, qmax = q.args[1:5]
        return float(s), int(z), int(qmin), int(qmax)

    if _is(node, "dequantize_per_tensor") and \
            _is(node.args[0], "quantize_per_tensor"):
        return args(node)
    for user in node.users:
        if _is(user, "quantize_per_tensor") and not _is(user, "dequantize"):
            return args(user)
    return None


@dataclass
class CodeGap:
    """A row held to at most one code: its first output against the
    reference, element by element, in the codes of its grid."""

    elements: int
    flips: int        # elements whose code differs
    steps: int        # the largest code difference
    max_abs: float    # the largest value difference

    def ok(self):
        return self.steps <= 1

    def summary(self):
        return (f"{self.flips} of {self.elements} code(s) differ "
                f"(rate {self.flips / max(self.elements, 1):.2e}), at most "
                f"{self.steps} step(s), max abs {self.max_abs:.3g}")


def code_gap(want, got, grid):
    """CodeGap of `got` against `want` (float32 vectors) in `grid`'s codes:
    quantize_per_tensor's rounding, so a value already on the grid gives its
    own code back."""
    s, z, qmin, qmax = grid
    want = np.asarray(want, F32).reshape(-1)
    got = np.asarray(got, F32).reshape(-1)
    inv = qm.reciprocal(s)

    def codes(v):
        return np.clip(np.rint((v * inv).astype(F32)) + z, qmin, qmax)

    d = np.abs(codes(want) - codes(got))
    return CodeGap(int(want.size), int((d > 0).sum()), int(d.max(initial=0)),
                   float(np.abs(want - got).max(initial=0.0)))


@dataclass
class LutGap:
    """A row's tables against eager's activations on identical input codes:
    each activation evaluated by torch in the graph's own layout (its scalar
    tail included) on eager's own input -- sqrt by its definition, as its table
    holds it (fold_quant_lut_pass.EXACT) -- and the table looked up at the same
    codes. Elements whose output differs must sit in torch's scalar tail (the
    last n mod 8 of n: NEON's 2 * Vec<float>::size()) or beside a coupled lane,
    at a split code the export recorded."""

    tables: int = 0
    elements: int = 0
    differ: int = 0
    explained: int = 0
    split_codes: int = 0

    def ok(self):
        return self.differ == self.explained

    def summary(self):
        return (f"{self.tables} table(s) over {self.elements} activation(s): "
                f"{self.differ} differ from eager, {self.explained} at a "
                f"recorded split code in a scalar-tail or coupled position; "
                f"{self.split_codes} split code(s) recorded")


def lut_eager_gap(model, inputs):
    """LutGap of the row's tables (fold_quant_luts on one export) against the
    activations eager runs (another export, unfolded): candidates pair up in
    graph order, refused ones skipped."""
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    folded = torch.export.export(model, inputs, strict=False)
    report = fold_quant_lut_pass.fold_quant_luts(folded)
    ep = torch.export.export(model, inputs, strict=False)
    gm = ep.module()
    refused = {name for name, _, _ in report.refused}
    acts = [n for n in gm.graph.nodes if n.op == "call_function"
            and n.target in fold_quant_lut_pass.ALLOWLIST and n.users
            and fold_quant_lut_pass._is_dequant(n.args[0]) and n.name not in refused]
    interp = ByDefinition(gm, garbage_collect_values=False)
    with torch.no_grad():
        interp.run(*inputs)
    gap = LutGap()
    qop = torch.ops.quantized_decomposed.quantize_per_tensor.default
    for A, (_, build) in zip(acts, report.tables):
        gap.tables += 1
        gap.split_codes += sum(map(len, build.split.values()))
        q1 = A.args[0].args[0]
        x = interp.env[q1.args[0]]
        with torch.no_grad():
            table = fold_quant_lut_pass.quant_lut(x, build.table, build.inv_scale,
                                        build.zero_point, build.quant_min,
                                        build.quant_max)
            if build.kind == "q":
                q2 = next(u for u in A.users if _is(u, "quantize_per_tensor"))
                want = interp.env[q2].to(torch.int64).reshape(-1)
                got = qop(table, *q2.args[1:6]).to(torch.int64).reshape(-1)
            else:
                want = interp.env[A].contiguous().view(torch.int32).reshape(-1)
                got = table.contiguous().view(torch.int32).reshape(-1)
            codes = qop(x, *q1.args[1:6]).to(torch.int64).reshape(-1)
        n = int(want.numel())
        gap.elements += n
        tail = n - n % 8
        split = set(build.split.get("scalar", []))
        coupled = set(build.split.get("coupled", []))
        lanes = x.reshape(-1).abs()
        for i in (want != got).nonzero().flatten().tolist():
            gap.differ += 1
            code = int(codes[i])
            # A whole-vector fallback (erf past |x| ~ 9.34, gelu's past 13.2)
            # couples the four lanes of one NEON vector.
            beside = bool((lanes[i - i % 4:i - i % 4 + 4] > COUPLED_FROM).any())
            gap.explained += ((i >= tail and code in split)
                              or (beside and code in coupled))
    return gap


# The smallest |x| at which a lane-coupled vector path (erf's) falls back.
COUPLED_FROM = 9.3


# --- lowering and the image -------------------------------------------------------------

@dataclass
class Fixture(ea.Fixture):
    luts: bool = True
    keeps: tuple = ()
    eager: EagerGap | None = None
    report: str = ""
    bar: str = "exact"
    grid: tuple | None = None    # a "code" row's first output's grid
    measured: int = 0            # ... and its element count
    outputs: int = 1
    lut_gap: LutGap | None = None
    dynamic: bool = False        # a dynamic recipe's: chains stripped at lowering


def fixtures():
    out = []
    for row in SPECS:
        model, inputs, _info = quantized(row)
        dynamic = "dynamic" in row.recipe
        ref = mirror(model, inputs, row.luts, dynamic)
        eager = eager_gap(model, inputs, ref)
        if row.channels_last and ref.ndim == 4:
            # The program writes its output back channels-last.
            ref = np.ascontiguousarray(ref.transpose(0, 2, 3, 1))
        fx = Fixture(row.name, model, inputs, ref, {}, long_modulus=row.long_modulus,
                     luts=row.luts, keeps=row.portable, bar=row.bar,
                     dynamic=dynamic)
        fx.eager = eager
        ep = torch.export.export(model, inputs, strict=False)
        nodes = output_nodes(ep.module())
        fx.outputs = len(nodes)
        if row.bar == "code":
            fx.grid = grid_of(nodes[0])
            fx.measured = int(np.prod(nodes[0].meta["val"].shape))
            if fx.grid is None:
                raise RuntimeError(f"{row.name}: no grid for its first output")
        if row.luts and any("quant_lut" in k for k in row.portable):
            fx.lut_gap = lut_eager_gap(model, inputs)
        out.append(fx)
    assert tuple(f.name for f in out) == ROWS
    # The speed rows: the tables' mirror is torch's own ops, bit for bit, on
    # every code they hold (the chains' mirror).
    chain, lut = (next(f for f in out if f.name == n) for n in SPEED_ROWS)
    if not np.array_equal(chain.reference.view(np.int32),
                          lut.reference.view(np.int32)):
        raise RuntimeError("st_spd: the tables' mirror differs from the chains'")
    return out


def lower(fixture):
    """The fixture through lower_esp32 (its static pairs absorbed, tables per
    the row) and apply_passes; nothing left outside the delegates but what the
    row keeps on purpose. Returns (program, the backend's log)."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(fixture.model, fixture.inputs, strict=False)
    log = io.StringIO()
    strip = contextlib.nullcontext()
    if fixture.dynamic:
        # A dynamic recipe's activation chains, absorbed by lower_esp32's own
        # strip right before partitioning, as the census lowers them
        # (quant_census.lower.strip_before_partitioning checks that it ran).
        from executorch.backends.esp32s3.test.quant_census.lower import (
            strip_before_partitioning,
        )

        strip = strip_before_partitioning({})
    with contextlib.redirect_stdout(log), strip:
        lowered = _passes.lower_esp32(
            ep, [Esp32Partitioner(**fixture.partitioner)],
            quant_luts=fixture.luts)
    with contextlib.redirect_stdout(log):
        _passes.apply_passes(lowered, quiet=True)
    graph = lowered.exported_program().graph_module.graph
    left = [str(n.target) for n in graph.nodes if n.op == "call_function"
            and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)
            and not any(k in str(n.target) for k in fixture.keeps + DEFAULT_OPS)]
    if left:
        raise RuntimeError(f"{fixture.name}: ops left portable: {left}\n"
                           f"{log.getvalue()}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue()


def emitted_programs():
    """Every row's emitted program (dump_required_ops: the portable kernels
    the parity firmware must register for them, the safety-net pairs')."""
    return [lower(f)[0] for f in fixtures()]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte) and the QEMU image (asym_static_fixtures.bin).
    Returns {row: reference}; each row's gap to eager is on export_all.gaps,
    and every fixture (its bar, a code row's grid) on export_all.fixtures."""
    table = fixtures()
    programs, refs = {}, {}
    export_all.gaps = {}
    export_all.fixtures = {f.name: f for f in table}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        export_all.gaps[fixture.name] = fixture.eager
        et, log = lower(fixture)
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
    write_image(os.path.join(out_dir, IMAGE),
                [f for f in table if f.name not in OPS_ROWS], programs)
    write_image(os.path.join(out_dir, OPS_IMAGE),
                [f for f in table if f.name in OPS_ROWS], programs)
    return refs


FIXTURE_ALL_OUTPUTS = 4   # hash every output (run_model's all_outputs)
FIXTURE_DUMP = 8          # and print every value (a row held to one code)


def write_image(path, table, programs):
    """asym.py's image layout, with FIXTURE_ALL_OUTPUTS on the rows whose
    program has several outputs and FIXTURE_DUMP on the code rows."""
    data = bytearray(ea.write_image(path, table, programs))
    for i, f in enumerate([f for f in table if f.qemu]):
        flags = ((FIXTURE_ALL_OUTPUTS if f.outputs > 1 or f.bar == "code"
                  else 0) | (FIXTURE_DUMP if f.bar == "code" else 0))
        at = 12 + i * ea.FIXTURE_ENTRY.size + 36            # the flags word
        word = int.from_bytes(data[at:at + 4], "little") | flags
        data[at:at + 4] = word.to_bytes(4, "little")
    with open(path, "wb") as f:
        f.write(data)
    return bytes(data)


if __name__ == "__main__":
    for name, ref in export_all().items():
        gap = export_all.gaps[name]
        print(f"  {name}: reference {np.shape(ref)}, {gap.flips} flip(s), "
              f"max abs {gap.max_abs:.3g}")
