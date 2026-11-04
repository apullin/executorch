# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""16-bit activations (Arm a16w8): the int16 loop's arithmetic at its
limits, bit for bit against mirrors.py, torch's own quantize_per_tensor for
int16, and exact integer arithmetic.

esp32_host_runner --asym-conformance runs esp32_asym.cpp's own functions on a
static 16-bit EXT_ACT_QUANT record -- quantize_row16, quantize_conv_row16 and
a16_column (the int64 dot and its two-rounding epilogue) -- and every result
must equal the reference in every bit:

  - codes against mirrors.quantize_activation AND torch's
    quantize_per_tensor.default with torch.int16, for scales that are powers
    of two (Arm a16w8's eps floor, 2^-12, among them) and that are not, over
    Arm's [-32767, 32767] and the full int16 range;
  - exact rounding ties (k + 1/2) / inv next to small codes and next to both
    rails, with odd and even floors;
  - both rails: inputs past the range up to +-FLT_MAX, whose fl(x * inv) the
    runtime clamps to +-2^22 before rounding;
  - the conv gather (padding is the zero point, 0) at every edge: padding,
    stride 2, dilation 2, transposed off-lattice taps (with output padding
    and dilation), conv1d, conv3d and a depthwise channel;
  - matmuls against mirrors.affine_matmul's 16-bit branch: int16
    extremes against int8 -127/127 weights, a K whose dot passes int32 (an
    int32 accumulator would wrap; int64 must be exact), K at the 131,040
    bound, ragged K (the k_pad tail), per-tensor weights, and -32768 against
    -128 at the bound.

The mirror's own conversion of the int64 dot to float32 is held to an
integer-only round-half-even, and its dot to Python's integers.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import struct

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.serialization.asym_records import (
    SchemeRefused,
    act_quant,
    check_act_bounds,
)
from executorch.backends.esp32s3.test.asym.test_asym_act_conformance import (
    GATHER_FIELDS,
    _arr,
)
from executorch.backends.esp32s3.test.asym.test_asym_static_conformance import run
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
FLT_MAX = float(np.finfo(F32).max)
BIAS = 16


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def a6(scale, qmin=-32767, qmax=32767):
    """A6 with the graph's constant (a float32 scale, as observers make)."""
    return qm.ActScheme("A6", "static", "static", "rne_before_zp", qmin, qmax,
                        scale=float(F32(scale)), zero_point=0)


SCHEMES = {
    "Arm s 2^-12 (the eps floor)": a6(2.0 ** -12),
    "Arm s 2^-16": a6(2.0 ** -16),
    "Arm s 0.000123": a6(0.000123),
    "Arm s 3.1e-5 (amax 1.0158 / 32767)": a6(1.0158 / 32767),
    "full int16 s 2^-10": a6(2.0 ** -10, -32768, 32767),
    "full int16 s 0.0071": a6(0.0071, -32768, 32767),
}

# Codes whose ties the inputs hit: small ones, the middle, both rails and
# just past them (clamped).
TIE_CODES = (list(range(-40, 41)) + [-16385, -16384, 16383, 16384]
             + list(range(-32770, -32760)) + list(range(32760, 32770)))


def tie_of(k, inv):
    """[x - 2 ulp .. x + 2 ulp] around the float32 x with fl(x * inv) ==
    k + 1/2 exactly, or [] (quant_lut.golden_inputs' search, any k)."""
    inf, target = F32(np.inf), F32(k + 0.5)
    up = down = F32(np.float64(k + 0.5) / np.float64(inv))
    for step in range(4):
        if step:
            up, down = np.nextafter(up, inf), np.nextafter(down, -inf)
        for cand in ((up,) if step == 0 else (up, down)):
            if F32(cand * inv) == target:
                d1, u1 = np.nextafter(cand, -inf), np.nextafter(cand, inf)
                return [np.nextafter(d1, -inf), d1, cand, u1,
                        np.nextafter(u1, inf)]
    return []


def tie_inputs(scheme):
    """Exact ties next to TIE_CODES, both rails, and the specials a static
    scale meets: zeros, denormals, values far past the range."""
    inv = qm.reciprocal(scheme.scale)
    xs = [x for k in TIE_CODES for x in tie_of(k, inv)]
    s = scheme.scale
    xs += [0.0, -0.0, 1e-45, -1e-45, 32767 * s, -32767 * s, 32768 * s,
           -32768 * s, 40000 * s, -40000 * s, 1e6 * s, -1e6 * s, FLT_MAX,
           -FLT_MAX, 3e30, -3e30]
    return np.asarray(xs, F32)


class Case:
    """One kind-11/12 case: its wire record and the expected arrays."""

    def __init__(self, name, kind, k_pad, n, rows, k, *, act, x, codes=b"",
                 scales=b"", bias=None, gather=b"", expected=()):
        self.name, self.expected = name, list(expected)
        flags = BIAS if bias is not None else 0
        header = struct.pack("<IIIIIIIIIi", kind, flags, k_pad, 0, n, rows, k,
                             0, 0, 0)
        self.record = header + _arr(
            b"", b"", codes, scales, b"", b"",
            b"" if bias is None else np.asarray(bias, F32).tobytes(), b"",
            act.pack(), np.asarray(x, F32).tobytes(), gather)


def quantize_case(name, scheme, x, k_pad):
    """Kind 11 without columns: rows x k quantised to int16 codes."""
    x = np.asarray(x, F32)
    rows, k = x.shape
    xq, _, _ = qm.quantize_activation(x, scheme)
    codes = np.zeros((rows, k_pad), np.int64)
    codes[:, :k] = xq
    return Case(name, 11, k_pad, 0, rows, k, act=act_quant(scheme), x=x,
                expected=[codes.astype(np.int16).tobytes()])


def matmul_case(name, scheme, x, codes, scales, bias, k_pad=None):
    """Kind 11: codes, then every column's int64 dot and epilogue; expected
    from mirrors.affine_matmul (its 16-bit branch)."""
    x = np.asarray(x, F32)
    rows, k = x.shape
    k_pad = k_pad or -(-k // 16) * 16
    codes = np.asarray(codes, np.int64)
    n = codes.shape[0]
    trace = qm.affine_matmul(x, codes, scales, bias, act=scheme, bits=8)
    xq = np.zeros((rows, k_pad), np.int64)
    xq[:, :k] = trace.xq
    full = np.zeros((n, k_pad), np.int64)
    full[:, :k] = codes
    return Case(name, 11, k_pad, n, rows, k, act=act_quant(scheme), x=x,
                codes=full.astype(np.int8).tobytes(),
                scales=np.asarray(scales, F32).tobytes(), bias=bias,
                expected=[xq.astype(np.int16).tobytes(),
                          trace.y.astype(F32).tobytes()])


def conv_case(name, scheme, x, geom, k_pad):
    """Kind 12: the int16 gather of every output position; expected from
    mirrors.im2col_codes (padding written as the zero point, 0)."""
    x4 = qm.as_cdhw(np.asarray(x, F32))
    xq, _, zx = qm.quantize_activation(x4.reshape(1, -1), scheme)
    cols = qm.im2col_codes(xq.reshape(x4.shape), geom, int(zx[0]),
                           k_pad=k_pad)
    gather = struct.pack("<20I", *(int(getattr(geom, f))
                                   for f in GATHER_FIELDS))
    return Case(name, 12, k_pad, 0, geom.m, geom.k, act=act_quant(scheme),
                x=x4.reshape(-1), gather=gather,
                expected=[cols.astype(np.int16).tobytes()])


def quantize_cases():
    out = []
    for name, scheme in SCHEMES.items():
        xs = tie_inputs(scheme)
        k = 56
        rows = np.resize(xs, -(-xs.size // k) * k).reshape(-1, k)
        out.append(quantize_case(f"{name}: ties and rails (k 56 of 64)",
                                 scheme, rows, 64))
    return out


def conv_cases(rng):
    from executorch.backends.esp32s3.operator_support.conv import ConvGeom

    geoms = {
        "3x3 pad 1": ConvGeom(h_in=5, w_in=6, c_in=2, h_out=5, w_out=6,
                              c_out=4, kh=3, kw=3, stride_h=1, stride_w=1,
                              pad_h=1, pad_w=1, dil_h=1, dil_w=1),
        "3x3 stride 2 pad 1": ConvGeom(h_in=7, w_in=7, c_in=2, h_out=4,
                                       w_out=4, c_out=4, kh=3, kw=3,
                                       stride_h=2, stride_w=2, pad_h=1,
                                       pad_w=1, dil_h=1, dil_w=1),
        "3x3 dilation 2 pad 2": ConvGeom(h_in=6, w_in=6, c_in=1, h_out=6,
                                         w_out=6, c_out=2, kh=3, kw=3,
                                         stride_h=1, stride_w=1, pad_h=2,
                                         pad_w=2, dil_h=2, dil_w=2),
        "transposed 3x3 stride 2": ConvGeom(h_in=3, w_in=3, c_in=2, h_out=7,
                                            w_out=7, c_out=2, kh=3, kw=3,
                                            stride_h=2, stride_w=2, pad_h=0,
                                            pad_w=0, dil_h=1, dil_w=1,
                                            transposed=1),
        "transposed 3x3 stride 2 pad 1 output_padding 1": ConvGeom(
            h_in=3, w_in=4, c_in=2, h_out=6, w_out=8, c_out=2, kh=3, kw=3,
            stride_h=2, stride_w=2, pad_h=1, pad_w=1, dil_h=1, dil_w=1,
            transposed=1),
        # Output padding 2 at stride 2, which only the dilation of 3 allows.
        "transposed 3x3 dil (3, 2) output_padding (2, 1)": ConvGeom(
            h_in=3, w_in=3, c_in=2, h_out=9, w_out=6, c_out=2, kh=3, kw=3,
            stride_h=2, stride_w=1, pad_h=2, pad_w=1, dil_h=3, dil_w=2,
            transposed=1),
        "conv1d k3 stride 2 pad 1": ConvGeom(h_in=1, w_in=11, c_in=3,
                                             h_out=1, w_out=6, c_out=2, kh=1,
                                             kw=3, stride_h=1, stride_w=2,
                                             pad_h=0, pad_w=1, dil_h=1,
                                             dil_w=1),
        "conv3d 3x3x3 pad 1": ConvGeom(d_in=3, h_in=4, w_in=4, c_in=2,
                                       d_out=3, h_out=4, w_out=4, c_out=2,
                                       kd=3, kh=3, kw=3, stride_d=1,
                                       stride_h=1, stride_w=1, pad_d=1,
                                       pad_h=1, pad_w=1, dil_d=1, dil_h=1,
                                       dil_w=1),
        "depthwise channel 3x3 pad 1": ConvGeom(h_in=5, w_in=5, c_in=1,
                                                h_out=5, w_out=5, c_out=1,
                                                kh=3, kw=3, stride_h=1,
                                                stride_w=1, pad_h=1, pad_w=1,
                                                dil_h=1, dil_w=1),
    }
    schemes = list(SCHEMES.values())
    out = []
    for i, (gname, geom) in enumerate(geoms.items()):
        scheme = schemes[i % len(schemes)]
        shape = (1, geom.c_in, geom.d_in, geom.h_in, geom.w_in)
        # Past the range too: rails inside the gather.
        x = (rng.standard_normal(shape) * scheme.scale * 20000).astype(F32)
        out.append(conv_case(f"{scheme.name} s {scheme.scale:.3g} {gname}",
                             scheme, x, geom, -(-geom.k // 16) * 16))
    return out


def matmul_cases(rng):
    out = []
    arm, full = SCHEMES["Arm s 2^-12 (the eps floor)"], SCHEMES[
        "full int16 s 2^-10"]
    odd = SCHEMES["Arm s 0.000123"]
    k = 96
    x = np.stack([np.resize(tie_inputs(odd), k)] + [
        (rng.standard_normal(k) * 2.0).astype(F32) for _ in range(3)])
    codes = rng.integers(-127, 128, (8, k))
    scales = (rng.random(8) * 0.01 + 1e-4).astype(F32)
    bias = (rng.standard_normal(8) * 0.1).astype(F32)
    out.append(matmul_case("ties and rails x int8 per channel", odd, x, codes,
                           scales, bias))
    out.append(matmul_case("ragged K 100 of 112, no bias", arm,
                           (rng.standard_normal((3, 100)) * 3).astype(F32),
                           rng.integers(-127, 128, (5, 100)),
                           (rng.random(5) * 0.01 + 1e-4).astype(F32), None))
    out.append(matmul_case("per-tensor weights (one scale)", arm,
                           (rng.standard_normal((2, 64)) * 3).astype(F32),
                           rng.integers(-127, 128, (6, 64)),
                           np.full(6, F32(0.0037)), bias[:6]))
    # int16 extremes against -127/127: both rails, the weights following the
    # input's sign (|acc| = K * 32767 * 127) or against it, or random. At K =
    # 600, |acc| = 2,496,845,400 > 2^31 - 1: an int32 accumulator wraps.
    k = 600
    rails = np.stack([np.full(k, 9.0, F32), np.full(k, -9.0, F32),
                      np.where(np.arange(k) % 2 == 0, 9.0, -9.0).astype(F32)])
    signs = np.where(np.arange(k) % 2 == 0, 127, -127)
    w = np.stack([np.full(k, 127), np.full(k, -127), signs, -signs,
                  rng.integers(-127, 128, k)])
    out.append(matmul_case(f"rails x +-127 at K {k}: |acc| past int32", arm,
                           rails, w, np.full(5, F32(2.0 ** -20)), None))
    # The bound itself: K = 131,040, every code at a rail against 127, and
    # int16's own -32768 against -128 (|acc| = 131,040 * 2^22, below 2^39).
    bound = qm.dot_length_bound(qm.WIDE_SPAN, 128, qm.ACCX_MAX)
    out.append(matmul_case(f"rails x 127 at the bound {bound:,}", arm,
                           np.full((1, bound), 9.0, F32),
                           np.full((1, bound), 127),
                           np.full(1, F32(2.0 ** -24)), None))
    out.append(matmul_case(f"-32768 x -128 at the bound {bound:,}", full,
                           np.full((1, bound), -1e9, F32),
                           np.full((1, bound), -128),
                           np.full(1, F32(2.0 ** -24)), None))
    return out


def cases():
    rng = np.random.default_rng(2061)
    return quantize_cases() + conv_cases(rng) + matmul_cases(rng)


def test_a16_arithmetic_is_the_mirror(runner, tmp_path):
    bad = run(runner, tmp_path, cases())
    assert not bad, f"{len(bad)} arrays differ: {bad[:12]}"


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_mirror_codes_are_quantize_per_tensor(name):
    """quantize_activation -- the reference above -- is torch's own
    quantize_per_tensor.default with torch.int16, ties and rails included."""
    scheme = SCHEMES[name]
    xs = tie_inputs(scheme)
    q = torch.ops.quantized_decomposed.quantize_per_tensor(
        torch.from_numpy(xs), scheme.scale, 0, scheme.qmin, scheme.qmax,
        torch.int16)
    mine, _, _ = qm.quantize_activation(xs[None], scheme)
    assert np.array_equal(mine[0], q.to(torch.int64).numpy()), name


def test_ties_reach_both_parities_and_both_rails():
    """The inputs hold exact ties next to odd and even codes, small and at
    the rails, for every scheme (otherwise the tie cases prove nothing)."""
    for name, scheme in SCHEMES.items():
        xs = tie_inputs(scheme)
        v = (xs * qm.reciprocal(scheme.scale)).astype(F32)
        ties = v[np.isfinite(v) & (np.abs(v) < 40000)
                 & (v - np.floor(v) == F32(0.5))]
        floors = np.floor(ties).astype(np.int64)
        assert {int(f) % 2 for f in floors} == {0, 1}, name
        assert floors.max() >= 32766 and floors.min() <= -32767, name
        codes, _, _ = qm.quantize_activation(xs[None], scheme)
        assert codes.max() == scheme.qmax and codes.min() == scheme.qmin, name


def test_wide_dots_are_exact_integers():
    """The 16-bit branch's dot is the exact integer -- Python's, at K = 600
    where int32 would wrap -- and the int32 path's wrap would differ."""
    scheme = SCHEMES["Arm s 2^-12 (the eps floor)"]
    x = np.full((1, 600), 9.0, F32)
    codes = np.full((1, 600), 127)
    trace = qm.affine_matmul(x, codes, np.ones(1, F32), act=scheme, bits=8)
    exact = 600 * 32767 * 127
    assert exact > qm.INT32_MAX
    assert int(trace.acc[0, 0, 0]) == exact
    assert int(qm.wrap_i32(exact)) != exact


def nearest_float(v):
    """Round-half-even of an integer to float32, by integers alone."""
    v = int(v)
    if v == 0:
        return 0.0
    m, shift = abs(v), 0
    while (m >> shift) >= 1 << 24:
        shift += 1
    q = m >> shift
    if shift:
        rest, half = m & ((1 << shift) - 1), 1 << (shift - 1)
        q += rest > half or (rest == half and q & 1)
    return float(np.copysign(np.float32(q) * np.float32(2.0 ** shift), v))


def test_mirror_converts_the_dot_with_one_rounding_to_even():
    """device_epilogue's fl(acc) on 40-bit dots -- exact ties at 2^25, 2^32
    and 2^38, one either side, random values -- is one round-half-even."""
    rng = np.random.default_rng(7)
    vals = []
    for e in (25, 32, 38):
        tie = (1 << e) + (1 << (e - 24))
        vals += [tie, tie + (1 << (e - 23)), tie - 1, tie + 1, -tie]
    vals += [int(v) for v in rng.integers(-(1 << 39), 1 << 39, 2000)]
    acc = np.asarray(vals, np.int64).reshape(1, -1, 1)
    got = qm.device_epilogue(acc, np.ones(1, F32), np.ones(acc.shape[1], F32))
    want = np.asarray([nearest_float(v) for v in vals], F32)
    assert np.array_equal(got.reshape(-1).view(np.uint32), want.view(np.uint32))


def test_bound_is_131040_and_refused_past_it():
    """131,040 (|xq| <= 32768 against |c| <= 128 in the 40-bit
    ACCX), refused by name at export one group past it -- not the int32
    bound the 8-bit records use (480 here)."""
    bound = qm.dot_length_bound(qm.WIDE_SPAN, 128, qm.ACCX_MAX)
    assert bound == 131_040
    assert qm.act_dot_bound(SCHEMES["Arm s 2^-16"], 8, False) == bound
    record = act_quant(SCHEMES["Arm s 2^-16"])
    assert record.code_bits == 16
    check_act_bounds(record, 8, bound, zero_point=False, offsets=False)
    with pytest.raises(SchemeRefused, match=r"\[act-dot-bound\].*40-bit"):
        check_act_bounds(record, 8, bound + 32, zero_point=False,
                         offsets=False)
