# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic activation quantisers, end to end: fixtures and the device's
arithmetic for each.

The backend runs a model's OWN dynamic activation quantiser on the device:
the exporter absorbs
the chooser -> quantize -> dequantize chain in front of each matmul and conv
(strip_dynamic_act_qdq_pass) and records its scheme (EXT_ACT_QUANT), and
the runtime reproduces it operation for operation, with the weight sums S'
its zero point needs. Each fixture here is a small model whose activations
arrive quantised the way a stock flow spells them, and its reference is what
the device computes for it -- mirrors.affine_matmul / affine_conv with
the chain's scheme, every rounding in the device's order -- so a row holds the
device's output bits to the mirror's, exactly:

  asym_a2x_i8      PT2E's affine chooser at XNNPACK's eps 2^-12, per tensor,
                   into int8 weights per channel (symmetric, then zero
                   points) and per tensor (symmetric, then a zero point); four
                   rows (row blocks)
  asym_a2a_i8      the same at Arm's eps 2^-16, one row (the rows path)
  asym_a3_i8       torchao's per-token affine chooser (A3), three rows, the
                   same weights
  asym_a7_i8       the symmetric choosers on the model's grid: torchao
                   SYMMETRIC per token (127.5), PT2E's at [-128, 127] (127.5,
                   eps 2^-12) and at [-127, 127] with eps 2^-16, then a uint8
                   torchao chooser (codes shifted onto int8)
  asym_a3_i4_m1    torchao 8da4w heads over one input: group 128 symmetric,
                   group 128 with zero points (these two share one quantised
                   row), group 32 with zero points (the 8da default), per
                   channel with zero points; decode, the fused int4 kernel
  asym_a3_i4_m8    the same heads at eight rows (row blocks)
  asym_a3_i4_split asym_a3_i4_m1 under min_latency: the affine split worker,
                   shared and standalone
  asym_a3_8da8w    torchao 8da8w with weight zero points per channel
  asym_a3_s32      torchao 8da8w heads over one 512-wide input whose weight
                   sums S' need int32: groups of 256 with zero
                   points (256 x 255 > 32767) and a 512-long symmetric row
                   (512 x 128); every other device row's sums fit int16
  asym_a3_woff     GGUF and tinygemm float offsets under A3: X'_g = X_g - n_g zx
  asym_actshare    one input, five heads: two identical A3 records (shared),
                   an XNNPACK A2, a torchao A7 and the backend's own per-row
                   head -- mixed schemes must not share
  asym_ties        every recipe's eps floor and exact rounding ties with odd
                   and even zero points, each head held to torch eager too
  asym_a2_conv     XNNPACK/Arm A2 convolutions: padded 3x3, strided, dilated,
                   depthwise (zero points) and transposed (weights quantised
                   here), the padding written as zx
  asym_a2_conv13   a conv1d and a conv3d under Arm's A2
  asym_a3_select   the selected-row head over int4 zero points, A3
  asym_a3_head     an int4 zero-point table tied between an embedding and a
                   head (weights in a .ptd), A3 on the head, split
  asym_a3_head_staged  the same head staged to int8 at load: host only (the
                   parity QEMU has no PSRAM)
  mlp_ao_asym      a torchao int4 MLP (the legacy
                   Int8DynActInt4WeightQuantizer, groups of 128) honouring its
                   own scheme: per-token affine, eps FLT_EPSILON, float32 zero
                   points

export_all(out_dir) writes every program, the head's .ptd and the packed image
main.cpp reads from the asymact partition (asym_act_fixtures.bin; the format is
asym.py's); it returns {row: reference output}. Imports torch.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.fixtures.asym import Fixture, long_input, ramp
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
PTD_TAG = "asym_act_weights"
IMAGE = "asym_act_fixtures.bin"
ROWS = ("asym_a2x_i8", "asym_a2a_i8", "asym_a3_i8", "asym_a7_i8",
        "asym_a3_i4_m1", "asym_a3_i4_m8", "asym_a3_i4_split", "asym_a3_8da8w",
        "asym_a3_s32", "asym_a3_woff", "asym_actshare", "asym_ties", "asym_a2_conv",
        "asym_a2_conv13", "asym_a3_select", "asym_a3_head",
        "asym_a3_head_staged", "mlp_ao_asym")
HOST_ONLY = ("asym_a3_head_staged",)
QEMU_ROWS = tuple(name for name in ROWS if name not in HOST_ONLY)
# Rows whose program returns several outputs: the device hashes them all, in
# output order, as one flat tensor (run_model's all_outputs).
MULTI_OUTPUT = ("asym_a3_i4_m1", "asym_a3_i4_m8", "asym_a3_i4_split",
                "asym_a3_s32", "asym_actshare", "asym_ties", "asym_a2_conv13")

# The schemes the chains below record (strip_dynamic_act_qdq_pass),
# as mirrors.ActScheme.
A2X, A2A, A3, A7 = qm.A2_XNNPACK, qm.A2_ARM, qm.A3_TORCHAO, qm.A7_TORCHAO
A3_LEGACY = qm.A3_LEGACY
A1_128 = qm.ActScheme("pt2e/symmetric", "symmetric", "tensor",
                      "rne_before_zp", -128, 127, eps=qm.EPS_XNNPACK)
A1_ARM = qm.ActScheme("pt2e/symmetric", "symmetric", "tensor",
                      "rne_before_zp", -127, 127, eps=qm.EPS_ARM)
A3_U8 = qm.ActScheme("torchao/asymmetric", "affine", "row", "rne_before_zp",
                     0, 255, eps=qm.FLT_MIN)
A2_FLT = qm.ActScheme("pt2e/affine", "affine", "tensor", "rne_before_zp",
                      -128, 127, eps=qm.FLT_EPSILON)


def _np(t):
    return ea._np(t)


# --- the chains, as the stock flows spell them -------------------------------------

def pt2e_chain(x, eps, symmetric=False, qmin=-128, qmax=127):
    """PT2E's dynamic per-tensor chain (XNNPACK's and Arm's is_dynamic
    configs): choose_qparams(_symmetric).tensor -> quantize_per_tensor.tensor
    -> dequantize_per_tensor.tensor."""
    qd = ea._decomposed()
    choose = (qd.choose_qparams_symmetric.tensor if symmetric
              else qd.choose_qparams.tensor)
    s, z = choose(x, qmin, qmax, eps, torch.int8)
    q = qd.quantize_per_tensor.tensor(x, s, z, qmin, qmax, torch.int8)
    return qd.dequantize_per_tensor.tensor(q, s, z, qmin, qmax, torch.int8)


def torchao_chain(x, mapping="ASYMMETRIC", eps=None, dtype=torch.int8,
                  qmin=-128, qmax=127, zp_dtype=torch.int8):
    """torchao's per-token chain (Int8DynamicActivationIntxWeightConfig):
    choose_qparams_affine -> quantize_affine -> dequantize_affine."""
    import torchao  # noqa: F401 -- registers torch.ops.torchao

    block = [1] * (x.dim() - 1) + [int(x.shape[-1])]
    ops = torch.ops.torchao
    s, z = ops.choose_qparams_affine(x, mapping, block, dtype, qmin, qmax, eps,
                                     None, zp_dtype)
    q = ops.quantize_affine(x, block, s, z, dtype, qmin, qmax)
    return ops.dequantize_affine(q, block, s, z, dtype, qmin, qmax)


def mm(x, codes, scales, bias, act, bits, gsize=None, zero_points=None,
       offsets=None):
    """One affine matmul the device's way: mirrors.affine_matmul with
    the chain's scheme."""
    return qm.affine_matmul(
        np.asarray(x, dtype=F32), _np(codes).astype(np.int64),
        _np(scales).astype(F32),
        None if bias is None else _np(bias).astype(F32), act=act, bits=bits,
        gsize=gsize,
        zero_points=None if zero_points is None
        else _np(zero_points).astype(np.int64),
        offsets=None if offsets is None else _np(offsets).astype(F32)).y


relu = ea._relu


def _codes(g, n, k, lo=-128, hi=127):
    return torch.randint(lo, hi + 1, (n, k), generator=g, dtype=torch.int8)


def _scales(g, n, low=1e-4, span=2e-3):
    return torch.rand(n, generator=g) * span + low


# --- int8: per channel and per tensor, symmetric and zero points ---------------------

class I8Act(nn.Module):
    """x[M,64] -> per-channel symmetric (bias) -> relu -> per-channel zero
    points -> per-tensor symmetric -> per-tensor zero point -> [M,32], each
    matmul behind one dynamic chain (`chain`, a per-layer callable)."""

    def __init__(self, chains, seed=21):
        super().__init__()
        self.chains = chains
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("c1", _codes(g, 48, 64, -127, 127))
        self.register_buffer("s1", _scales(g, 48))
        self.register_buffer("z1", torch.zeros(48, dtype=torch.int64))
        self.register_buffer("b1", torch.randn(48, generator=g) * 0.1)
        self.register_buffer("c2", _codes(g, 48, 48))
        z2 = torch.randint(-128, 128, (48,), generator=g, dtype=torch.int64)
        z2[0], z2[1] = -128, 127
        self.register_buffer("z2", z2)
        self.register_buffer("s2", _scales(g, 48))
        self.register_buffer("c3", _codes(g, 32, 48, -127, 127))
        self.s3 = float(np.float32(0.0017))      # float32 values: static
        self.register_buffer("c4", _codes(g, 32, 32))
        self.s4 = float(np.float32(0.0023))      # scales must be
        self.z4 = -37

    def forward(self, x):
        qd = ea._decomposed()
        w1 = qd.dequantize_per_channel(self.c1, self.s1, self.z1, 0, -127, 127,
                                       torch.int8)
        h = F.relu(F.linear(self.chains[0](x), w1, self.b1))
        w2 = qd.dequantize_per_channel(self.c2, self.s2, self.z2, 0, -128, 127,
                                       torch.int8)
        h = F.linear(self.chains[1](h), w2)
        w3 = qd.dequantize_per_tensor(self.c3, self.s3, 0, -127, 127,
                                      torch.int8)
        h = F.linear(self.chains[2](h), w3)
        w4 = qd.dequantize_per_tensor(self.c4, self.s4, self.z4, -128, 127,
                                      torch.int8)
        return F.linear(self.chains[3](h), w4)

    def reference(self, x, acts):
        h = relu(mm(x, self.c1, self.s1, self.b1, acts[0], 8))
        h = mm(h, self.c2, self.s2, None, acts[1], 8, zero_points=self.z2)
        h = mm(h, self.c3, np.full(32, F32(self.s3)), None, acts[2], 8)
        return mm(h, self.c4, np.full(32, F32(self.s4)), None, acts[3], 8,
                  zero_points=np.full(32, self.z4))


def _same(chain, n=4):
    return [chain] * n


def i8_nets():
    """(row, model, schemes, rows of input) for the int8 fixtures."""
    a2x = lambda x: pt2e_chain(x, qm.EPS_XNNPACK)  # noqa: E731
    a2a = lambda x: pt2e_chain(x, qm.EPS_ARM)  # noqa: E731
    a3 = lambda x: torchao_chain(x)  # noqa: E731
    sym = [lambda x: torchao_chain(x, "SYMMETRIC"),
           lambda x: pt2e_chain(x, qm.EPS_XNNPACK, symmetric=True),
           lambda x: pt2e_chain(x, qm.EPS_ARM, symmetric=True, qmin=-127),
           lambda x: torchao_chain(x, dtype=torch.uint8, qmin=0, qmax=255,
                                   zp_dtype=torch.int32)]
    return (("asym_a2x_i8", I8Act(_same(a2x), 21), _same(A2X), 4),
            ("asym_a2a_i8", I8Act(_same(a2a), 22), _same(A2A), 1),
            ("asym_a3_i8", I8Act(_same(a3), 23), _same(A3), 3),
            ("asym_a7_i8", I8Act(sym, 24), [A7, A1_128, A1_ARM, A3_U8], 2))


# --- int4: torchao 8da4w heads -------------------------------------------------------

class Heads(nn.Module):
    """Four Linear(256, 32) heads over one input, returned as four outputs
    (32 columns: wide enough to split across the cores)."""

    def __init__(self):
        super().__init__()
        self.fa, self.fb = nn.Linear(256, 32), nn.Linear(256, 32)
        self.fc, self.fd = nn.Linear(256, 32), nn.Linear(256, 32)

    def forward(self, x):
        return self.fa(x), self.fb(x), self.fc(x), self.fd(x)


def i4_heads():
    """torchao Int8DynamicActivationIntxWeightConfig per head: group 128
    symmetric, group 128 asymmetric, group 32 asymmetric (0.18's default
    granularity), per channel asymmetric."""
    from torchao.quantization import (Int8DynamicActivationIntxWeightConfig,
                                      MappingType, quantize_)
    from torchao.quantization.granularity import PerAxis, PerGroup

    torch.manual_seed(31)
    net = Heads().eval()
    for name, gran, mapping in (
            ("fa", PerGroup(128), MappingType.SYMMETRIC),
            ("fb", PerGroup(128), MappingType.ASYMMETRIC),
            ("fc", PerGroup(32), MappingType.ASYMMETRIC),
            ("fd", PerAxis(0), MappingType.ASYMMETRIC)):
        quantize_(net, Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4, weight_granularity=gran,
            weight_mapping_type=mapping),
            filter_fn=lambda module, fqn, name=name: fqn == name)
    return net


def torchao_layer(layer):
    """(codes, scales [N, G], zero points [N, G] or None, group, bias) of a
    torchao-quantised Linear, as the backend receives them."""
    codes, scales, zeros, group, bias = ea.i4_layer(layer)
    return codes, scales, (zeros if np.any(zeros) else None), group, bias


def heads_reference(net, x, act=A3, bits=4):
    out = []
    for layer in (net.fa, net.fb, net.fc, net.fd):
        codes, scales, zeros, group, bias = torchao_layer(layer)
        out.append(mm(x, codes, scales, bias, act, bits, gsize=group,
                      zero_points=zeros))
    return np.concatenate([o.reshape(-1) for o in out])


class Mlp8(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(64, 48), nn.Linear(48, 32)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def da8w():
    """torchao 8da8w: int8 weights per channel with zero points."""
    from torchao.quantization import (Int8DynamicActivationIntxWeightConfig,
                                      MappingType, quantize_)
    from torchao.quantization.granularity import PerAxis

    torch.manual_seed(32)
    net = Mlp8().eval()
    quantize_(net, Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int8, weight_granularity=PerAxis(0),
        weight_mapping_type=MappingType.ASYMMETRIC))
    return net


class WideHeads(nn.Module):
    """Two int8 heads over one 512-wide input."""

    K = 512

    def __init__(self):
        super().__init__()
        self.fa, self.fb = nn.Linear(self.K, 16), nn.Linear(self.K, 16)

    def forward(self, x):
        return self.fa(x), self.fb(x)


def wide_heads():
    """torchao 8da8w with int32 weight sums: fa in groups of 256 with zero
    points, fb one symmetric 512-long row per channel."""
    from torchao.quantization import (Int8DynamicActivationIntxWeightConfig,
                                      MappingType, quantize_)
    from torchao.quantization.granularity import PerAxis, PerGroup

    torch.manual_seed(33)
    net = WideHeads().eval()
    for name, gran, mapping in (("fa", PerGroup(256), MappingType.ASYMMETRIC),
                                ("fb", PerAxis(0), MappingType.SYMMETRIC)):
        quantize_(net, Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8, weight_granularity=gran,
            weight_mapping_type=mapping),
            filter_fn=lambda module, fqn, name=name: fqn == name)
    return net


def wide_heads_reference(net, x):
    out = []
    for layer in (net.fa, net.fb):
        codes, scales, zeros, group, bias = torchao_layer(layer)
        out.append(mm(x, codes, scales, bias, A3, 8, gsize=group,
                      zero_points=zeros))
    return np.concatenate([o.reshape(-1) for o in out])


def da8w_reference(net, x):
    c1, s1, z1, g1, b1 = torchao_layer(net.fc1)
    c2, s2, z2, g2, b2 = torchao_layer(net.fc2)
    h = relu(mm(x, c1, s1, b1, A3, 8, gsize=g1, zero_points=z1))
    return mm(h, c2, s2, b2, A3, 8, gsize=g2, zero_points=z2)


# --- float offsets under an activation zero point -------------------------------------

class WoffAct(ea.WoffNet):
    """asym.py's GGUF -> relu -> tinygemm net with A3 in front of each."""

    def forward(self, x):
        from torchao.quantization.quant_primitives import (
            _dequantize_affine_tinygemm, _dequantize_gguf)

        w1 = _dequantize_gguf(self.gq, [1, 32], torch.uint4, self.gss,
                              self.gsm, self.gqs, self.gqm, torch.float32)
        h = F.relu(F.linear(torchao_chain(x), w1, self.gb))
        w2 = _dequantize_affine_tinygemm(self.tq, [1, 32], self.ts, self.tm,
                                         torch.uint8, 0, 15)
        return F.linear(torchao_chain(h), w2, self.tb)

    def reference(self, x):
        d = _np(self.gss * self.gqs)
        mn = _np(self.gsm * self.gqm)
        codes = _np(self.gq).astype(np.int64) - 8
        h = relu(mm(x, codes, d, self.gb, A3, 4, gsize=32,
                    zero_points=np.full((64, 8), -8), offsets=mn))
        codes = _np(self.tq).astype(np.int64) - 8
        return mm(h, codes, self.ts, self.tb, A3, 4, gsize=32,
                  offsets=self.tm)


# --- activation sharing across schemes --------------------------------------------

class ActShare(nn.Module):
    """x[4,64] -> five Linear(64,16) heads: two behind identical torchao A3
    chains (symmetric codes, then zero points: they share one quantised row
    and its X_g), an XNNPACK A2, a torchao A7, and a zero-point weight with
    no chain (the backend's own per-row absmax, an affine op without an
    activation record)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(41)
        for i in range(5):
            self.register_buffer(f"c{i}", _codes(g, 16, 64, -127, 127))
            self.register_buffer(f"s{i}", _scales(g, 16))
        self.register_buffer("z1", torch.randint(-128, 128, (16,), generator=g,
                                                 dtype=torch.int64))
        self.register_buffer("z0", torch.zeros(16, dtype=torch.int64))

    def _w(self, i, zero_points=None):
        qd = ea._decomposed()
        z = self.z0 if zero_points is None else zero_points
        lo = -127 if zero_points is None else -128
        return qd.dequantize_per_channel(getattr(self, f"c{i}"),
                                         getattr(self, f"s{i}"), z, 0, lo, 127,
                                         torch.int8)

    def forward(self, x):
        return (F.linear(torchao_chain(x), self._w(0)),
                F.linear(torchao_chain(x), self._w(1, self.z1)),
                F.linear(pt2e_chain(x, qm.EPS_XNNPACK), self._w(2)),
                F.linear(torchao_chain(x, "SYMMETRIC"), self._w(3)),
                F.linear(x, self._w(4, self.z1)))

    def reference(self, x):
        out = [mm(x, self.c0, self.s0, None, A3, 8),
               mm(x, self.c1, self.s1, None, A3, 8, zero_points=self.z1),
               mm(x, self.c2, self.s2, None, A2X, 8),
               mm(x, self.c3, self.s3, None, A7, 8),
               mm(x, self.c4, self.s4, None, qm.A0, 8, zero_points=self.z1)]
        return np.concatenate([np.asarray(o, F32).reshape(-1) for o in out])


# --- rounding ties and eps floors, each recipe's ---------------------------------------

def _tie_row(k, zx_odd, s=2.0 ** -4):
    """A K-wide activation whose affine scale is exactly s (a power of two, so
    fl(1/s) is exact) and whose zero point is odd or even, holding exact
    halves (j + 0.5) * s that round-before-zp and round-after-zp send to
    different codes when zx is odd."""
    lo_codes = 3 if zx_odd else 4         # zx = -128 + lo_codes
    row = np.zeros(k, dtype=F32)
    row[0] = -lo_codes * s
    row[1] = (255 - lo_codes) * s        # hi - lo = 255 s exactly
    halves = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 7.5, -2.5, 100.5])
    row[2:2 + halves.size] = (halves * s).astype(F32)
    return row


def _floor_gain(k, eps):
    """Gains that scale the ramp (-0.5 + j/128, all negative) into a range
    narrower than 255 * eps on both sides of 0, so the scale floors at eps."""
    pattern = np.array([1.0, -1.0, 37.0, -64.5, 100.0, 3.0, -0.25, 0.0])
    return (np.resize(pattern, k) * eps * 0.5).astype(F32)


SHIFT, GAIN = "shift", "gain"
TIES = (  # (head, chain, scheme, (how, builder)): h = x + shift or x * gain
    ("xnn_tie_odd", lambda x: pt2e_chain(x, qm.EPS_XNNPACK), A2X,
     (SHIFT, lambda k: _tie_row(k, True))),
    ("xnn_tie_even", lambda x: pt2e_chain(x, qm.EPS_XNNPACK), A2X,
     (SHIFT, lambda k: _tie_row(k, False))),
    ("xnn_floor", lambda x: pt2e_chain(x, qm.EPS_XNNPACK), A2X,
     (GAIN, lambda k: _floor_gain(k, qm.EPS_XNNPACK))),
    ("arm_tie_odd", lambda x: pt2e_chain(x, qm.EPS_ARM), A2A,
     (SHIFT, lambda k: _tie_row(k, True))),
    ("arm_floor", lambda x: pt2e_chain(x, qm.EPS_ARM), A2A,
     (GAIN, lambda k: _floor_gain(k, qm.EPS_ARM))),
    ("flt_floor", lambda x: pt2e_chain(x, qm.FLT_EPSILON), A2_FLT,
     (GAIN, lambda k: _floor_gain(k, qm.FLT_EPSILON))),
    ("ao_tie_odd", lambda x: torchao_chain(x), A3,
     (SHIFT, lambda k: _tie_row(k, True))),
    ("ao_floor", lambda x: torchao_chain(x), A3,
     (GAIN, lambda k: _floor_gain(k, qm.FLT_MIN))),
    ("ao_sym_tie", lambda x: torchao_chain(x, "SYMMETRIC"), A7,
     (SHIFT, lambda k: _tie_row(k, False))),
)


class Ties(nn.Module):
    """One input (the device's ramp) moved per head onto that head's designed
    activations -- shifted exactly onto rounding ties (every term a multiple
    of 2^-7), or scaled into an eps floor -- then an identity-like weight
    passes the codes through. The add and the multiply stay portable."""

    K = 32

    def __init__(self):
        super().__init__()
        base = torch.from_numpy(np.asarray(ramp((1, self.K)), F32))
        g = torch.Generator().manual_seed(42)
        for i, (_, _, _, (how, build)) in enumerate(TIES):
            row = torch.from_numpy(build(self.K)).reshape(1, self.K)
            self.register_buffer(f"move{i}", row - base if how == SHIFT
                                 else row)
            codes = torch.zeros(8, self.K, dtype=torch.int8)
            for j in range(8):
                codes[j, j + 2 if j < 6 else j - 6] = 127
            codes[7] = torch.randint(-127, 128, (self.K,), generator=g,
                                     dtype=torch.int8)
            self.register_buffer(f"c{i}", codes)
            self.register_buffer(f"s{i}", torch.full((8,), 2.0 ** -7))
        self.register_buffer("z", torch.zeros(8, dtype=torch.int64))

    def _moved(self, i, x):
        move = getattr(self, f"move{i}")
        return x + move if TIES[i][3][0] == SHIFT else x * move

    def forward(self, x):
        qd = ea._decomposed()
        out = []
        for i, (_, chain, _, _) in enumerate(TIES):
            w = qd.dequantize_per_channel(getattr(self, f"c{i}"),
                                          getattr(self, f"s{i}"), self.z, 0,
                                          -127, 127, torch.int8)
            out.append(F.linear(chain(self._moved(i, x)), w))
        return tuple(out)

    def reference(self, x):
        out = []
        for i, (_, _, act, _) in enumerate(TIES):
            h = _np(self._moved(i, torch.as_tensor(np.asarray(x, F32))))
            out.append(mm(h.astype(F32), getattr(self, f"c{i}"),
                          getattr(self, f"s{i}"), None, act, 8))
        return np.concatenate([o.reshape(-1) for o in out])

    def check_eager(self, x):
        """Each head's dequantised activation in torch eager equals the
        mirror's, bit for bit: the chain's codes ARE mirrors.py's."""
        for i, (head, chain, act, _) in enumerate(TIES):
            h = self._moved(i, x)
            want = _np(chain(h)).astype(F32)
            xq, sx, zx = qm.quantize_activation(_np(h).astype(F32), act)
            got = ((xq - zx[:, None]).astype(F32) * sx[:, None]).astype(F32)
            if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                raise RuntimeError(f"asym_ties head {head}: the mirror's codes "
                                   f"are not torch's")


# --- convolutions ---------------------------------------------------------------------

class ConvAct(nn.Module):
    """x[1,4,9,9] -> padded 3x3 (symmetric) -> relu -> strided 3x3 (zero
    points) -> dilated 3x3 (symmetric) -> depthwise 3x3 (zero points, Arm's
    eps) -> 2x2 stride-2 transposed (float weights, quantised here), each
    behind an A2 chain."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(51)

        def conv_codes(shape, symmetric):
            lo = -127 if symmetric else -128
            return torch.randint(lo, 128, shape, generator=g, dtype=torch.int8)

        self.specs = (("1", (8, 4, 3, 3), True), ("2", (8, 8, 3, 3), False),
                      ("3", (8, 8, 3, 3), True), ("d", (8, 1, 3, 3), False))
        for name, shape, symmetric in self.specs:
            self.register_buffer("c" + name, conv_codes(shape, symmetric))
            self.register_buffer("s" + name, _scales(g, 8))
            z = (torch.zeros(8, dtype=torch.int64) if symmetric else
                 torch.randint(-128, 128, (8,), generator=g, dtype=torch.int64))
            self.register_buffer("z" + name, z)
            self.register_buffer("b" + name, torch.randn(8, generator=g) * 0.1)
        self.up = nn.ConvTranspose2d(8, 4, 2, stride=2)
        with torch.no_grad():
            self.up.weight.copy_(torch.randn(8, 4, 2, 2, generator=g) * 0.2)
            self.up.bias.copy_(torch.randn(4, generator=g) * 0.1)

    def _w(self, name, symmetric):
        qd = ea._decomposed()
        return qd.dequantize_per_channel(
            getattr(self, "c" + name), getattr(self, "s" + name),
            getattr(self, "z" + name), 0, -127 if symmetric else -128, 127,
            torch.int8)

    def forward(self, x):
        xnn = lambda t: pt2e_chain(t, qm.EPS_XNNPACK)  # noqa: E731
        h = F.relu(F.conv2d(xnn(x), self._w("1", True), self.b1, padding=1))
        h = F.conv2d(xnn(h), self._w("2", False), self.b2, stride=2, padding=1)
        h = F.conv2d(xnn(h), self._w("3", True), self.b3, padding=2,
                     dilation=2)
        h = F.conv2d(pt2e_chain(h, qm.EPS_ARM), self._w("d", False), self.bd,
                     padding=1, groups=8)
        return self.up(xnn(h))


class Conv13(nn.Module):
    """A conv1d over x1[1,4,20] and a conv3d over x3[1,2,4,6,6], Arm's A2."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(52)
        self.register_buffer("c1", torch.randint(-128, 128, (8, 4, 3),
                                                 generator=g, dtype=torch.int8))
        self.register_buffer("s1", _scales(g, 8))
        self.register_buffer("z1", torch.randint(-128, 128, (8,), generator=g,
                                                 dtype=torch.int64))
        self.register_buffer("c3", torch.randint(-127, 128, (4, 2, 3, 3, 3),
                                                 generator=g, dtype=torch.int8))
        self.register_buffer("s3", _scales(g, 4))
        self.register_buffer("z3", torch.zeros(4, dtype=torch.int64))
        self.register_buffer("b3", torch.randn(4, generator=g) * 0.1)

    def forward(self, x1, x3):
        qd = ea._decomposed()
        w1 = qd.dequantize_per_channel(self.c1, self.s1, self.z1, 0, -128, 127,
                                       torch.int8)
        w3 = qd.dequantize_per_channel(self.c3, self.s3, self.z3, 0, -127, 127,
                                       torch.int8)
        return (F.conv1d(pt2e_chain(x1, qm.EPS_ARM), w1, padding=1, stride=2),
                F.conv3d(pt2e_chain(x3, qm.EPS_ARM), w3, self.b3, padding=1))


def conv_geoms(model, inputs):
    """The ConvGeom of every convolution, in graph order, as the backend
    reads it (conv_support.admit on the edge graph)."""
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.operator_support.conv import admit

    edge = to_edge(torch.export.export(model, inputs))
    return [admit(n) for n in edge.exported_program().graph.nodes
            if n.op == "call_function" and "convolution" in str(n.target)]


def _conv(x, codes_nk, scales, bias, geom, act, zero_points=None):
    return qm.affine_conv(np.asarray(x, F32), _np(codes_nk).astype(np.int64),
                          _np(scales).astype(F32),
                          None if bias is None else _np(bias).astype(F32),
                          geom, act, zero_points=None if zero_points is None
                          else _np(zero_points).astype(np.int64))


def conv_reference(model, x):
    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        quantize_absmax_rows,
    )

    g1, g2, g3, gd, gt = conv_geoms(model, (torch.as_tensor(x),))
    h = relu(_conv(x, _np(model.c1).reshape(8, -1), model.s1, model.b1, g1,
                   A2X))
    h = _conv(h, _np(model.c2).reshape(8, -1), model.s2, model.b2, g2, A2X,
              zero_points=model.z2)
    h = _conv(h, _np(model.c3).reshape(8, -1), model.s3, model.b3, g3, A2X)
    h = _conv(h, _np(model.cd).reshape(8, -1), model.sd, model.bd, gd, A2A,
              zero_points=model.zd)
    # The transposed weight as the backend packs it: taps flipped, [C_out,
    # C_in * taps], quantised here per row (the backend's own codes).
    w = model.up.weight.detach().flip([2, 3]).transpose(0, 1).reshape(4, -1)
    codes, scales = quantize_absmax_rows(_np(w).astype(F32), FORMATS["i8"])
    return _conv(h, codes, scales, model.up.bias, gt, A2X)


def conv13_reference(model, x1, x3):
    g1, g3 = conv_geoms(model, (torch.as_tensor(x1), torch.as_tensor(x3)))
    y1 = _conv(x1, _np(model.c1).reshape(8, -1), model.s1, None, g1, A2A,
               zero_points=model.z1)
    y3 = _conv(x3, _np(model.c3).reshape(4, -1), model.s3, model.b3, g3, A2A)
    return np.concatenate([y1.reshape(-1), y3.reshape(-1)])


# --- heads: selected rows, a tied head with external weights --------------------------

class SelectAct(ea.SelectHead):
    def forward(self, value, rows):
        import torchao  # noqa: F401

        table = torch.ops.torchao.dequantize_affine(
            self.codes, [1, 32], self.scales, self.zeros, torch.int8, -8, 7)
        return F.linear(torchao_chain(value), torch.index_select(table, 0, rows))

    def reference(self, value, rows):
        r = _np(rows)
        return mm(value, _np(self.codes)[r], _np(self.scales)[r], None, A3, 4,
                  gsize=32, zero_points=_np(self.zeros)[r])


class HeadAct(ea.TiedHead):
    def forward(self, idx):
        import torchao  # noqa: F401

        table = torch.ops.torchao.dequantize_affine(
            self.codes, [1, 32], self.scales, self.zeros, torch.int8, -8, 7)
        return F.linear(torchao_chain(F.embedding(idx, table)), table[:80])

    def reference(self, idx):
        e = qm.embedding_dequant(_np(self.codes), _np(self.scales),
                                 _np(self.zeros), _np(idx))
        return mm(e, _np(self.codes)[:80], _np(self.scales)[:80], None, A3, 4,
                  gsize=32, zero_points=_np(self.zeros)[:80])


# --- mlp_ao, honouring its scheme ----------------------------------------------------

def ao_mlp():
    """(model, input): an MLP quantised by torchao's
    Int8DynActInt4WeightQuantizer(groupsize=128) -- int4 codes in an int8
    container, [N, n_groups] scales -- whose codes and scales survive to the
    edge graph as plain constants the delegate consumes verbatim. fc1 has
    K=256 (two groups of 128), fc2 K=128 (one)."""
    from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

    class AoMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 128)   # K=256 -> 2 groups of 128
            self.fc2 = nn.Linear(128, 64)    # K=128 -> 1 group

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.manual_seed(0)
    m = AoMLP().eval()
    mq = Int8DynActInt4WeightQuantizer(groupsize=128).quantize(m)
    from executorch.backends.esp32s3.test.models import ramp_input

    x = ramp_input((1, 8, 256))
    return mq, x


def mlp_ao_reference(model, x):
    """ao_mlp's model with its own activations: the legacy quantiser's
    per-token affine chooser (eps FLT_EPSILON) on both Linears."""
    def layer(lin):
        return (_np(lin.weight), _np(lin.scales).astype(F32),
                _np(lin.zeros).astype(np.float32), _np(lin.bias).astype(F32))

    x2 = np.asarray(x, F32).reshape(-1, 256)
    c1, s1, z1, b1 = layer(model.fc1)
    c2, s2, z2, b2 = layer(model.fc2)
    for z in (z1, z2):
        if np.any(z):
            raise RuntimeError("mlp_ao_asym: the legacy quantiser's weight "
                               "zero points are no longer all zero")
    h = relu(mm(x2, c1, s1, b1, A3_LEGACY, 4, gsize=128))
    return mm(h, c2, s2, b2, A3_LEGACY, 4, gsize=128)


# --- the fixture table ----------------------------------------------------------------

def fixtures():
    """Every fixture with its inputs (main.cpp's rules) and reference."""
    out = []
    for name, net, acts, m in i8_nets():
        net = net.eval()
        x = ramp((m, 64))
        out.append(Fixture(name, net, (x,), net.reference(x, acts), {}))
    heads = i4_heads()
    for m in (1, 8):
        x = ramp((m, 256))
        out.append(Fixture(f"asym_a3_i4_m{m}", heads, (x,),
                           heads_reference(heads, x), {}))
    out.append(Fixture("asym_a3_i4_split", None, (), out[-2].reference, {},
                       split_pass=True, program_of="asym_a3_i4_m1"))
    net = da8w()
    x = ramp((2, 64))
    out.append(Fixture("asym_a3_8da8w", net, (x,), da8w_reference(net, x), {}))
    wide = wide_heads()
    x = ramp((2, WideHeads.K))
    out.append(Fixture("asym_a3_s32", wide, (x,), wide_heads_reference(wide, x),
                       {}))
    woff = WoffAct().eval()
    x = ramp((2, 256))
    out.append(Fixture("asym_a3_woff", woff, (x,), woff.reference(x), {}))
    share = ActShare().eval()
    x = ramp((4, 64))
    out.append(Fixture("asym_actshare", share, (x,), share.reference(x), {}))
    ties = Ties().eval()
    x = ramp((1, Ties.K))
    ties.check_eager(x)
    out.append(Fixture("asym_ties", ties, (x,), ties.reference(x), {}))
    conv = ConvAct().eval()
    x = ramp((1, 4, 9, 9))
    out.append(Fixture("asym_a2_conv", conv, (x,), conv_reference(conv, x),
                       {}))
    c13 = Conv13().eval()
    x1, x3 = ramp((1, 4, 20)), ramp((1, 2, 4, 6, 6))
    out.append(Fixture("asym_a2_conv13", c13, (x1, x3),
                       conv13_reference(c13, x1, x3), {}))
    sel = SelectAct().eval()
    value, rows = ramp((2, 64)), long_input(7, 257)
    out.append(Fixture("asym_a3_select", sel, (value, rows),
                       sel.reference(value, rows), {}, long_modulus=257))
    head = HeadAct().eval()
    idx = long_input(3, 96)
    out.append(Fixture("asym_a3_head", head, (idx,), head.reference(idx),
                       dict(external_weights=PTD_TAG, split_tied_head=True),
                       long_modulus=96))
    out.append(Fixture("asym_a3_head_staged", head, (idx,),
                       head.reference(idx),
                       dict(external_weights=PTD_TAG, split_tied_head=True,
                            stage_tied_head=True), long_modulus=96,
                       qemu=False))
    mq, x = ao_mlp()
    out.append(Fixture("mlp_ao_asym", mq.eval(), (x,),
                       mlp_ao_reference(mq, x), {}))
    assert tuple(f.name for f in out) == ROWS
    assert tuple(f.name for f in out if f.qemu) == QEMU_ROWS
    return out


def lower(fixture):
    """The fixture through the backend, its dynamic chains absorbed first
    (strip_dynamic_act_qdq_pass), as every stock flow needs: then
    asym.lower's rule, nothing left portable."""
    import contextlib
    import io

    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import fold_conv_bn_pass
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3._passes import strip_dynamic_act_qdq_pass
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(fixture.model, fixture.inputs)
    log = io.StringIO()
    fold = fold_conv_bn_pass.fold_conv_bn
    absorbed = []

    def absorb_then_fold(edge, method_name="forward"):
        absorbed.append(strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq(edge, method_name))
        return fold(edge, method_name)

    _passes.fold_conv_bn = absorb_then_fold
    try:
        with contextlib.redirect_stdout(log):
            lowered = _passes.lower_esp32(
                ep, [Esp32Partitioner(**fixture.partitioner)])
            _passes.apply_passes(lowered, quiet=True)
    finally:
        _passes.fold_conv_bn = fold
    if not absorbed or not absorbed[0]:
        raise RuntimeError(f"{fixture.name}: no dynamic chain was absorbed")
    graph = lowered.exported_program().graph_module.graph
    allowed = PORTABLE.get(fixture.name, ())
    left = [str(n.target) for n in graph.nodes if n.op == "call_function"
            and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)
            and not any(a in str(n.target) for a in allowed)]
    if left:
        raise RuntimeError(f"{fixture.name}: ops left portable: {left}\n"
                           f"{log.getvalue()}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue()


FIXTURE_ALL_OUTPUTS = 4   # hash every output (run_model's all_outputs)
# Portable ops a row keeps on purpose: asym_ties moves the device's ramp onto
# its designed activations before the chains.
# mlp_ao_asym keeps the three reshapes around its Linears.
PORTABLE = {"asym_ties": ("aten.add.Tensor", "aten.mul.Tensor"),
            "mlp_ao_asym": ("aten.view_copy",)}


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte), the head's .ptd (asym_act_weights.ptd) and
    the QEMU image (asym_act_fixtures.bin). Returns {row: reference}."""
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
    write_image(os.path.join(out_dir, IMAGE), table, programs)
    return refs


def write_image(path, table, programs):
    """asym.py's image layout, with FIXTURE_ALL_OUTPUTS on the rows whose
    program has several outputs."""
    data = bytearray(ea.write_image(path, table, programs))
    rows = [f for f in table if f.qemu]
    for i, f in enumerate(rows):
        if f.name in MULTI_OUTPUT:
            at = 12 + i * ea.FIXTURE_ENTRY.size + 36       # the flags word
            flags = int.from_bytes(data[at:at + 4], "little")
            data[at:at + 4] = (flags | FIXTURE_ALL_OUTPUTS).to_bytes(4, "little")
    with open(path, "wb") as f:
        f.write(data)
    return bytes(data)


if __name__ == "__main__":
    for name, ref in export_all().items():
        print(f"  {name}: reference {np.shape(ref)}")
