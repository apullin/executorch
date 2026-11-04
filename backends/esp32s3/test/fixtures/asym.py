# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight zero points and float offsets, end to end: fixtures and the device's
arithmetic for each.

The backend passes weights through with their zero points and float offsets
(EXT_WEIGHT_AFFINE).
Each fixture here is a small model whose weights arrive already quantised the
way a stock flow spells them, and its reference is what the device computes
for it -- mirrors.affine_matmul / embedding_* under the backend's own
activations (A0), every rounding in the device's order -- so a row holds the
device's output bits to the mirror's, exactly:

  asym_i8_m1, asym_i8_m4
      PT2E int8 codes: per-channel zero points spanning [-128, 127] with an
      Arm-style int32 bias (dequantize_per_channel over 1-D codes), groups of
      32 (dequantize_per_channel_group), and a per-tensor zero point
      (dequantize_per_tensor, a double scale); one row (the rows path) and four
      (row blocks)
  asym_i4_m1, asym_i4_m8
      torchao IntxWeightOnlyConfig(int4, ASYMMETRIC) at group 128, group 32 and
      per channel: decode (the fused int4 kernel) and prefill (row blocks)
  asym_i4_split
      asym_i4_m1 run under min_latency: the affine split worker
  asym_woff
      a GGUF Q4_K Linear, fl(fl(q * d) + mn) with fp16-exact mins, then a
      tinygemm one, fl(fl((q - 8) * s) + m): torchao's own reference dequants,
      as export traces them
  asym_conv
      PT2E int8 per-channel zero points on a padded 3x3 conv, a fused relu,
      then a 3x3 depthwise conv
  asym_embed
      a tinygemm (float-offset) embedding table
  asym_embed_fqp
      torch.ao's per_channel_affine_float_qparams embedding: uint8 codes with
      a fractional float zero point per row (PT2E dequantize_per_channel),
      fl(fl(q - z) * s)
  asym_head
      an int4 zero-point table tied between an embedding and a head over its
      first 80 rows, weights in a .ptd, the head split across the cores
      (split_tied_head)
  asym_head_staged
      the same head staged to int8 at load (stage_tied_head): host only -- the
      parity firmware's QEMU has no PSRAM to stage into
  asym_select
      a selected-row head over an int4 zero-point table

export_all(out_dir) writes every program, the head's .ptd and the packed image
of the QEMU programs main.cpp reads from a flash partition (asym_fixtures.bin);
it returns {row: reference output}. Imports torch.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
PTD_TAG = "asym_weights"

# The packed image main.cpp's run_asym_fixtures reads (little-endian):
#   "ASYF", u32 version, u32 count, then per program 48 B:
#   char name[24] (NUL-padded), u32 offset, u32 size, u32 long_modulus,
#   u32 flags (FIXTURE_PTD, FIXTURE_SPLIT_PASS), u32 reserved[2];
#   then the programs, each 16-byte aligned.
FIXTURES_MAGIC = b"ASYF"
FIXTURES_VERSION = 1
FIXTURE_PTD = 1           # loads with the partition's .ptd
FIXTURE_SPLIT_PASS = 2    # runs in the min_latency pass, not the plain one
FIXTURE_ENTRY = struct.Struct("<24sIIIIII")
# Every row fixtures() builds, in order (test parametrisation needs the names
# without building the models).
ROWS = ("asym_i8_m1", "asym_i8_m4", "asym_i4_m1", "asym_i4_m8",
        "asym_i4_split", "asym_woff", "asym_conv", "asym_embed",
        "asym_embed_fqp", "asym_head", "asym_head_staged", "asym_select")
# Rows the parity firmware cannot run: staging needs PSRAM, which its QEMU
# lacks. The host runner runs every row.
HOST_ONLY = ("asym_head_staged",)
QEMU_ROWS = tuple(name for name in ROWS if name not in HOST_ONLY)


def _decomposed():
    """torch.ops.quantized_decomposed, which needs its registering import."""
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    return torch.ops.quantized_decomposed


def ramp(shape):
    """main.cpp's float input rule: element j is j * 2^-7 - 0.5."""
    from executorch.backends.esp32s3.test.models import ramp_input

    return ramp_input(shape)


def long_input(n, modulus):
    """main.cpp's int64 input rule: element j is (7j + 3) mod `modulus`."""
    return torch.tensor([(j * 7 + 3) % modulus for j in range(n)],
                        dtype=torch.int64)


def _relu(h):
    """The device's fused relu, a clamp to [0, inf): y < 0 ? 0 : y."""
    h = np.asarray(h, dtype=F32)
    return np.where(h < 0, F32(0), h).astype(F32)


def _np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _affine(x, codes, scales, bias, bits, gsize=None, zero_points=None,
            offsets=None):
    """One affine matmul the device's way (mirrors.affine_matmul with
    the backend's per-row absmax activations)."""
    return qm.affine_matmul(
        np.asarray(x, dtype=F32), _np(codes), _np(scales).astype(F32),
        None if bias is None else _np(bias).astype(F32), act=qm.A0, bits=bits,
        gsize=gsize, zero_points=None if zero_points is None else _np(
            zero_points).astype(np.int64),
        offsets=None if offsets is None else _np(offsets).astype(F32)).y


# --- int8: PT2E per-channel, per-group and per-tensor zero points ---------------

class I8Net(nn.Module):
    """x[M,128] -> per-channel int8 (zero points, int32 bias) -> relu ->
    groups of 32 (zero points) -> relu -> per-tensor (zero point) -> [M,32]."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(11)

        def codes(n, k):
            return torch.randint(-128, 128, (n, k), generator=g,
                                 dtype=torch.int8)

        z1 = torch.randint(-128, 128, (64,), generator=g, dtype=torch.int64)
        z1[0], z1[1] = -128, 127          # both ends of the code range
        self.register_buffer("c1", codes(64, 128))
        self.register_buffer("z1", z1)
        self.register_buffer("s1", torch.rand(64, generator=g) * 2e-3 + 1e-4)
        # Arm's bias: int32 codes at the derived scale, past 2^24 so their
        # conversion to float rounds.
        self.register_buffer("b1", torch.randint(-2 ** 26, 2 ** 26, (64,),
                                                 generator=g, dtype=torch.int32))
        self.register_buffer("bs1", torch.rand(64, generator=g) * 1e-6 + 1e-8)
        self.register_buffer("bz1", torch.zeros(64, dtype=torch.int64))
        self.register_buffer("c2", codes(64, 64))
        self.register_buffer("z2", torch.randint(-128, 128, (128,), generator=g,
                                                 dtype=torch.int64))
        self.register_buffer("s2", torch.rand(128, generator=g) * 2e-3 + 1e-4)
        self.register_buffer("c3", codes(32, 64))
        self.s3 = 0.0021     # a double: the op multiplies by its float32 value
        self.z3 = 37

    def forward(self, x):
        qd = _decomposed()
        w1 = qd.dequantize_per_channel(self.c1, self.s1, self.z1, 0, -128, 127,
                                       torch.int8)
        b1 = qd.dequantize_per_channel(self.b1, self.bs1, self.bz1, 0,
                                       -2 ** 31, 2 ** 31 - 1, torch.int32)
        h = F.relu(F.linear(x, w1, b1))
        w2 = qd.dequantize_per_channel_group(self.c2, self.s2, self.z2, -128,
                                             127, torch.int8, 32, torch.float32)
        h = F.relu(F.linear(h, w2))
        w3 = qd.dequantize_per_tensor(self.c3, self.s3, self.z3, -128, 127,
                                      torch.int8)
        return F.linear(h, w3)

    def bias(self):
        """The int32 bias as PT2E's own dequant evaluates it: fl(fl(b) * s)."""
        qd = _decomposed()
        return _np(qd.dequantize_per_channel(self.b1, self.bs1, self.bz1, 0,
                                             -2 ** 31, 2 ** 31 - 1,
                                             torch.int32))

    def reference(self, x):
        b1 = self.bias()
        assert np.array_equal(b1, qm.dequantize_int32_bias(_np(self.b1),
                                                           _np(self.bs1)))
        h = _relu(_affine(x, self.c1, self.s1, b1, 8, zero_points=self.z1))
        h = _relu(_affine(h, self.c2, _np(self.s2).reshape(64, 2), None, 8,
                          gsize=32, zero_points=_np(self.z2).reshape(64, 2)))
        return _affine(h, self.c3, np.full(32, F32(self.s3)), None, 8,
                       zero_points=np.full(32, self.z3))


# --- int4: torchao IntxWeightOnlyConfig(ASYMMETRIC) -------------------------------

class I4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def i4_net():
    """fc1 at group 128, fc2 at group 32 (torchao 0.18's 8da default), fc3 per
    output channel, all with torchao's own affine zero points."""
    from torchao.quantization import (IntxWeightOnlyConfig, MappingType,
                                      quantize_)
    from torchao.quantization.granularity import PerAxis, PerGroup

    torch.manual_seed(12)
    net = I4Net().eval()
    for name, granularity in (("fc1", PerGroup(128)), ("fc2", PerGroup(32)),
                              ("fc3", PerAxis(0))):
        quantize_(net, IntxWeightOnlyConfig(
            weight_dtype=torch.int4, granularity=granularity,
            mapping_type=MappingType.ASYMMETRIC),
            filter_fn=lambda module, fqn, name=name: fqn == name)
    return net


def i4_layer(layer):
    """(codes, scales [N, G], zero points [N, G], group or None, bias) of one
    torchao-quantised Linear, as the backend receives them."""
    t = layer.weight
    scales = _np(t.scale).astype(F32)
    group = None if scales.shape[1] == 1 else int(t.block_size[1])
    return (_np(t.qdata), scales, _np(t.zero_point).astype(np.int64), group,
            _np(layer.bias).astype(F32))


def i4_reference(net, x):
    h = np.asarray(x, dtype=F32)
    layers = (net.fc1, net.fc2, net.fc3)
    for i, layer in enumerate(layers):
        codes, scales, zeros, group, bias = i4_layer(layer)
        h = _affine(h, codes, scales, bias, 4, gsize=group, zero_points=zeros)
        if i < len(layers) - 1:
            h = _relu(h)
    return h


# --- float offsets: GGUF and tinygemm, as torchao's reference dequants trace ----

class WoffNet(nn.Module):
    """x[2,256] -> GGUF Q4_K Linear 256->64 -> relu -> tinygemm Linear 64->32."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(13)
        self.register_buffer("gq", torch.randint(0, 16, (64, 256), generator=g,
                                                 dtype=torch.uint8))
        self.register_buffer("gss", torch.rand(64, 1, generator=g) * 1e-3 + 1e-4)
        # The mins' super-block scale is a power of two and their 6-bit codes
        # small integers, so every min fl(sm * qm) is exact in fp16: the
        # backend stores those offsets as halves (tinygemm's below are fp32).
        self.register_buffer("gsm", torch.full((64, 1), 2.0 ** -6))
        self.register_buffer("gqs", torch.randint(1, 64, (64, 8), generator=g)
                             .to(torch.float32))
        self.register_buffer("gqm", torch.randint(0, 64, (64, 8), generator=g)
                             .to(torch.float32))
        self.register_buffer("gb", torch.randn(64, generator=g) * 0.1)
        self.register_buffer("tq", torch.randint(0, 16, (32, 64), generator=g,
                                                 dtype=torch.uint8))
        self.register_buffer("ts", torch.rand(32, 2, generator=g) * 0.02 + 1e-3)
        self.register_buffer("tm", torch.randn(32, 2, generator=g) * 0.05)
        self.register_buffer("tb", torch.randn(32, generator=g) * 0.1)

    def forward(self, x):
        from torchao.quantization.quant_primitives import (
            _dequantize_affine_tinygemm, _dequantize_gguf)

        w1 = _dequantize_gguf(self.gq, [1, 32], torch.uint4, self.gss,
                              self.gsm, self.gqs, self.gqm, torch.float32)
        h = F.relu(F.linear(x, w1, self.gb))
        w2 = _dequantize_affine_tinygemm(self.tq, [1, 32], self.ts, self.tm,
                                         torch.uint8, 0, 15)
        return F.linear(h, w2, self.tb)

    def reference(self, x):
        d = _np(self.gss * self.gqs)      # fl(ss * qs), as the graph computes
        mn = _np(self.gsm * self.gqm)
        assert np.array_equal(mn.astype(np.float16).astype(F32), mn)
        codes = _np(self.gq).astype(np.int64) - 8
        h = _relu(_affine(x, codes, d, self.gb, 4, gsize=32,
                          zero_points=np.full((64, 8), -8), offsets=mn))
        codes = _np(self.tq).astype(np.int64) - 8
        return _affine(h, codes, self.ts, self.tb, 4, gsize=32,
                       offsets=self.tm)


# --- convolutions: PT2E per-channel zero points ------------------------------------

class ConvNet(nn.Module):
    """x[1,4,9,9] -> 3x3 conv (pad 1) -> relu -> 3x3 depthwise (pad 1)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(14)
        for name, shape in (("1", (8, 4, 3, 3)), ("2", (8, 1, 3, 3))):
            self.register_buffer("c" + name, torch.randint(
                -128, 128, shape, generator=g, dtype=torch.int8))
            self.register_buffer("z" + name, torch.randint(
                -128, 128, (8,), generator=g, dtype=torch.int64))
            self.register_buffer("s" + name,
                                 torch.rand(8, generator=g) * 2e-3 + 1e-4)
            self.register_buffer("b" + name, torch.randn(8, generator=g) * 0.1)

    def forward(self, x):
        qd = _decomposed()
        w1 = qd.dequantize_per_channel(self.c1, self.s1, self.z1, 0, -128, 127,
                                       torch.int8)
        h = F.relu(F.conv2d(x, w1, self.b1, padding=1))
        w2 = qd.dequantize_per_channel(self.c2, self.s2, self.z2, 0, -128, 127,
                                       torch.int8)
        return F.conv2d(h, w2, self.b2, padding=1, groups=8)

    def reference(self, x):
        from executorch.backends.esp32s3.operator_support.conv import ConvGeom
        from executorch.backends.esp32s3.test.reference import im2col_reference

        common = dict(h_in=9, w_in=9, h_out=9, w_out=9, c_out=8, kh=3, kw=3,
                      stride_h=1, stride_w=1, pad_h=1, pad_w=1, dil_h=1,
                      dil_w=1, c_out_total=8)
        dense = ConvGeom(c_in=4, groups=1, **common)
        cols = im2col_reference(torch.as_tensor(x), dense).numpy().astype(F32)
        y = _affine(cols, _np(self.c1).reshape(8, -1), self.s1, self.b1, 8,
                    zero_points=self.z1)                      # [81, 8]
        h = _relu(y).T.reshape(1, 8, 9, 9)
        # Depthwise: every channel gathers its own input, one row at a time.
        one = ConvGeom(c_in=1, groups=1, **{**common, "c_out": 1,
                                             "c_out_total": 1})
        out = np.zeros((8, 81), dtype=F32)
        for ch in range(8):
            cols = im2col_reference(torch.from_numpy(h[:, ch:ch + 1]),
                                    one).numpy().astype(F32)
            out[ch] = _affine(cols, _np(self.c2)[ch].reshape(1, -1),
                              _np(self.s2)[ch:ch + 1], _np(self.b2)[ch:ch + 1],
                              8, zero_points=_np(self.z2)[ch:ch + 1])[:, 0]
        return out.reshape(1, 8, 9, 9)


# --- embeddings and heads -----------------------------------------------------------

class EmbedWoff(nn.Module):
    """A tinygemm table [23, 64]: fl(fl((q - 8) * s) + m) per group of 32."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(15)
        self.register_buffer("q", torch.randint(0, 16, (23, 64), generator=g,
                                                dtype=torch.uint8))
        self.register_buffer("s", torch.rand(23, 2, generator=g) * 0.02 + 1e-3)
        self.register_buffer("m", torch.randn(23, 2, generator=g) * 0.05)

    def forward(self, idx):
        from torchao.quantization.quant_primitives import (
            _dequantize_affine_tinygemm)

        table = _dequantize_affine_tinygemm(self.q, [1, 32], self.s, self.m,
                                            torch.uint8, 0, 15)
        return F.embedding(idx, table)

    def reference(self, idx):
        return qm.embedding_w_off(_np(self.q), _np(self.s), _np(self.m),
                                  _np(idx), gsize=32)


class EmbedFloatQParams(nn.Module):
    """torch.ao's per_channel_affine_float_qparams table [17, 48]: uint8
    codes, a scale and a FRACTIONAL float zero point per row, dequantised by
    PT2E's dequantize_per_channel as fl(fl(q - z) * s)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(18)
        self.register_buffer("q", torch.randint(0, 256, (17, 48), generator=g,
                                                dtype=torch.uint8))
        self.register_buffer("s", torch.rand(17, generator=g) * 0.01 + 1e-3)
        self.register_buffer("z", torch.rand(17, generator=g) * 255.0)

    def forward(self, idx):
        qd = _decomposed()
        table = qd.dequantize_per_channel(self.q, self.s, self.z, 0, 0, 255,
                                          torch.uint8)
        return F.embedding(idx, table)

    def reference(self, idx):
        rows = _np(idx)
        q = _np(self.q).astype(F32)[rows]
        z = _np(self.z).astype(F32)[rows, None]
        s = _np(self.s).astype(F32)[rows, None]
        return ((q - z).astype(F32) * s).astype(F32)


def _zero_point_table(g, rows):
    zeros = torch.randint(-8, 8, (rows, 2), generator=g, dtype=torch.int8)
    zeros[0, 0], zeros[1, 1] = -8, 7      # both ends of the int4 range
    return zeros


class TiedHead(nn.Module):
    """An int4 zero-point table [96, 64] (groups of 32) tied between an
    embedding gather and a head over its first 80 rows."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(16)
        self.register_buffer("codes", torch.randint(-8, 8, (96, 64), generator=g,
                                                    dtype=torch.int8))
        self.register_buffer("scales", torch.rand(96, 2, generator=g) * 0.05
                             + 1e-3)
        self.register_buffer("zeros", _zero_point_table(g, 96))

    def forward(self, idx):
        import torchao  # noqa: F401 -- registers torch.ops.torchao

        table = torch.ops.torchao.dequantize_affine(
            self.codes, [1, 32], self.scales, self.zeros, torch.int8, -8, 7)
        return F.linear(F.embedding(idx, table), table[:80])

    def reference(self, idx):
        e = qm.embedding_dequant(_np(self.codes), _np(self.scales),
                                 _np(self.zeros), _np(idx))
        return _affine(e, _np(self.codes)[:80], _np(self.scales)[:80], None, 4,
                       gsize=32, zero_points=_np(self.zeros)[:80])


class SelectHead(nn.Module):
    """The legal-token head over an int4 zero-point table [257, 64]."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(17)
        self.register_buffer("codes", torch.randint(-8, 8, (257, 64),
                                                    generator=g,
                                                    dtype=torch.int8))
        self.register_buffer("scales", torch.rand(257, 2, generator=g) * 0.05
                             + 1e-3)
        self.register_buffer("zeros", _zero_point_table(g, 257))

    def forward(self, value, rows):
        import torchao  # noqa: F401

        table = torch.ops.torchao.dequantize_affine(
            self.codes, [1, 32], self.scales, self.zeros, torch.int8, -8, 7)
        return F.linear(value, torch.index_select(table, 0, rows))

    def reference(self, value, rows):
        r = _np(rows)
        return _affine(value, _np(self.codes)[r], _np(self.scales)[r], None, 4,
                       gsize=32, zero_points=_np(self.zeros)[r])


# --- the fixture table ----------------------------------------------------------------

@dataclass
class Fixture:
    name: str
    model: nn.Module
    inputs: tuple
    reference: np.ndarray
    partitioner: dict
    long_modulus: int = 0
    qemu: bool = True          # in the parity firmware's image
    split_pass: bool = False   # a row of the min_latency pass (no program)
    program_of: str = ""       # a split-pass row runs this fixture's program


def fixtures():
    """Every fixture with its inputs (main.cpp's rules) and reference."""
    out = []
    i8 = I8Net().eval()
    for m in (1, 4):
        x = ramp((m, 128))
        out.append(Fixture(f"asym_i8_m{m}", i8, (x,), i8.reference(x), {}))
    i4 = i4_net()
    for m in (1, 8):
        x = ramp((m, 256))
        out.append(Fixture(f"asym_i4_m{m}", i4, (x,), i4_reference(i4, x), {}))
    out.append(Fixture("asym_i4_split", None, (), out[-2].reference, {},
                       split_pass=True, program_of="asym_i4_m1"))
    woff = WoffNet().eval()
    x = ramp((2, 256))
    out.append(Fixture("asym_woff", woff, (x,), woff.reference(x), {}))
    conv = ConvNet().eval()
    x = ramp((1, 4, 9, 9))
    out.append(Fixture("asym_conv", conv, (x,), conv.reference(x), {}))
    emb = EmbedWoff().eval()
    idx = long_input(5, 23)
    out.append(Fixture("asym_embed", emb, (idx,), emb.reference(idx), {},
                       long_modulus=23))
    fqp = EmbedFloatQParams().eval()
    idx = long_input(4, 17)
    out.append(Fixture("asym_embed_fqp", fqp, (idx,), fqp.reference(idx), {},
                       long_modulus=17))
    head = TiedHead().eval()
    idx = long_input(3, 96)
    out.append(Fixture("asym_head", head, (idx,), head.reference(idx),
                       dict(external_weights=PTD_TAG, split_tied_head=True),
                       long_modulus=96))
    out.append(Fixture("asym_head_staged", head, (idx,), head.reference(idx),
                       dict(external_weights=PTD_TAG, split_tied_head=True,
                            stage_tied_head=True),
                       long_modulus=96, qemu=False))
    sel = SelectHead().eval()
    value, rows = ramp((2, 64)), long_input(7, 257)
    out.append(Fixture("asym_select", sel, (value, rows),
                       sel.reference(value, rows), {}, long_modulus=257))
    assert tuple(f.name for f in out) == ROWS
    assert tuple(f.name for f in out if f.qemu) == QEMU_ROWS
    return out


def lower(fixture):
    """The fixture through the backend: (ExecutorchProgramManager, the
    preprocess log). Everything must delegate: no dequant op, float-offset
    chain or matmul may be left portable."""
    import contextlib
    import io

    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    ep = torch.export.export(fixture.model, fixture.inputs)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lowered = lower_esp32(ep, [Esp32Partitioner(**fixture.partitioner)])
        apply_passes(lowered, quiet=True)
    graph = lowered.exported_program().graph_module.graph
    left = [str(n.target) for n in graph.nodes if n.op == "call_function"
            and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)]
    if left:
        raise RuntimeError(f"{fixture.name}: ops left portable: {left}\n"
                           f"{log.getvalue()}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue()


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte), the head's .ptd (asym_weights.ptd), and the
    QEMU image (asym_fixtures.bin). Returns {row: reference output}."""
    table = fixtures()
    programs, refs = {}, {}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        if fixture.split_pass:
            continue
        et, log = lower(fixture)
        programs[fixture.name] = bytes(et.buffer)
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(et.buffer)
        if fixture.partitioner.get("external_weights"):
            et.write_tensor_data_to_file(out_dir)
        if verbose:
            ops = [line for line in log.splitlines() if "[Esp32Backend]" in line]
            print(f"  {fixture.name}.pte: {len(et.buffer):,} B; "
                  + "; ".join(o.split(": ", 1)[-1] for o in ops))
    write_image(os.path.join(out_dir, "asym_fixtures.bin"), table, programs)
    return refs


def write_image(path, table, programs):
    """The packed QEMU image (layout at FIXTURES_MAGIC)."""
    rows = [f for f in table if f.qemu]
    header = len(FIXTURES_MAGIC) + 8 + FIXTURE_ENTRY.size * len(rows)
    start = -(-header // 16) * 16        # programs start 16-byte aligned
    body, offsets = bytearray(), {}
    for f in rows:
        if f.split_pass:
            continue
        body += bytes(-(start + len(body)) % 16)
        offsets[f.name] = (start + len(body), len(programs[f.name]))
        body += programs[f.name]
    entries = []
    for f in rows:
        source = f.program_of or f.name
        offset, size = offsets[source]
        flags = (table_flags(table, source)
                 | (FIXTURE_SPLIT_PASS if f.split_pass else 0))
        entries.append(FIXTURE_ENTRY.pack(f.name.encode(), offset, size,
                                          f.long_modulus, flags, 0, 0))
    data = (FIXTURES_MAGIC + struct.pack("<II", FIXTURES_VERSION, len(rows))
            + b"".join(entries))
    data += bytes(start - len(data)) + bytes(body)
    with open(path, "wb") as f:
        f.write(data)
    return data


def table_flags(table, name):
    fixture = next(f for f in table if f.name == name)
    return FIXTURE_PTD if fixture.partitioner.get("external_weights") else 0


def read_image(data):
    """[(name, program bytes, long_modulus, flags)] from a packed image."""
    if data[:4] != FIXTURES_MAGIC:
        raise ValueError("not an asym fixtures image")
    version, count = struct.unpack_from("<II", data, 4)
    if version != FIXTURES_VERSION:
        raise ValueError(f"asym fixtures image version {version}")
    out = []
    for i in range(count):
        name, offset, size, modulus, flags, _, _ = FIXTURE_ENTRY.unpack_from(
            data, 12 + i * FIXTURE_ENTRY.size)
        out.append((name.rstrip(b"\0").decode(), data[offset:offset + size],
                    modulus, flags))
    return out


if __name__ == "__main__":
    for name, ref in export_all().items():
        print(f"  {name}: reference {ref.shape}")
