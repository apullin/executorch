# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static int8 graphs (Stage A): the runtime's static quantiser at its
limits, bit for bit against mirrors.py and torch's own quantize_per_tensor.

esp32_host_runner --asym-conformance runs esp32_asym.cpp's own functions on
a static EXT_ACT_QUANT record -- choose (the record's constants),
quantize_row, quantize_conv_row, then the corrected dot with the zero point's
zx * S' term -- and every result must equal the reference in every bit:

  - codes against mirrors.quantize_activation AND torch's
    quantize_per_tensor.default, for float32 scales that are powers of two
    and that are not;
  - exact rounding ties, (j + 1/2) / inv, with odd and even zero points (a
    rounding after the zero point would differ there), found for any scale
    by quant_lut.golden_inputs;
  - both rails: inputs far past the calibrated range, up to +-FLT_MAX, whose
    fl(x * inv) the runtime clamps to +-2^22 before rounding;
  - NaN of either sign, the infinities and +-FLT_MAX: ExecuTorch 1.6's rule
    (runtime/quant_rules.h), NaN the zero point and an infinity its rail,
    for the activations and the output record, against ExecuTorch's own
    quantize_per_tensor kernel (torch's code for NaN is platform-defined);
  - uint8 ranges, shifted onto int8 codes with their zero point;
  - the conv gather's padding written as the static zero point at every
    edge: padding, stride 2, dilation 2, transposed off-lattice taps (with
    output padding and dilation), conv1d, conv3d and a depthwise channel;
  - the corrected dot end to end against mirrors.affine_matmul: per
    row and grouped weights, int8 weight zero points, the int4 kernels, and
    a static zero point at the 65,792 bound.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import struct
import subprocess

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

from executorch.backends.esp32s3._passes import fold_quant_lut_pass
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.serialization.asym_records import (
    act_quant,
    out_quant,
)
from executorch.backends.esp32s3.test.asym.test_asym_act_conformance import (
    GATHER_FIELDS,
    TERMS,
    Case,
    _arr,
    _terms,
    matmul_case,
    quantize_case,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
FLT_MAX = float(np.finfo(F32).max)


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def static(scale, zero_point, qmin=-128, qmax=127):
    """A5 with the graph's constants (a float32 scale, as observers make)."""
    return qm.ActScheme("A5", "static", "static", "rne_before_zp", qmin, qmax,
                        scale=float(F32(scale)), zero_point=int(zero_point))


# Scales: powers of two (every tie is an exact half of the input grid) and
# float32 values that are not (their ties come from quant_lut.golden_inputs).
SCHEMES = {
    "int8 s 2^-4 zp -7 (odd)": static(2.0 ** -4, -7),
    "int8 s 2^-4 zp -8 (even)": static(2.0 ** -4, -8),
    "int8 s 0.0213 zp 3 (odd)": static(0.0213, 3),
    "int8 s 0.0213 zp 0": static(0.0213, 0),
    "int8 s 0.37 zp -128 (rail)": static(0.37, -128),
    "int8 s 0.0071 zp 127 (rail)": static(0.0071, 127),
    "Arm [-127, 127] s 0.05 zp 2": static(0.05, 2, -127, 127),
    "uint8 s 2^-5 zp 131 (odd)": static(2.0 ** -5, 131, 0, 255),
    "uint8 s 0.0123 zp 128": static(0.0123, 128, 0, 255),
}


# NaN of either sign, the infinities, +-FLT_MAX and values past every rail.
NONFINITE = np.asarray([np.nan, -np.nan, np.inf, -np.inf, FLT_MAX, -FLT_MAX,
                        0.0, 1e30, -1e30, np.nan, np.inf, -np.inf, 1.0, -1.0,
                        np.nan, -np.nan], F32)


def portable_quantize(x, scale, zero_point, qmin, qmax, dtype):
    """ExecuTorch's own quantize_per_tensor kernel (kernels/quantized/cpu),
    which its quantized ops library registers with torch as the .out
    variant: on this aarch64 host its NEON block and its scalar tail."""
    import executorch.kernels.quantized  # noqa: F401 -- registers the kernels

    out = torch.empty(x.shape, dtype=dtype)
    return torch.ops.quantized_decomposed.quantize_per_tensor.out(
        torch.from_numpy(np.ascontiguousarray(x)), scale, zero_point, qmin,
        qmax, dtype, out=out)


def tie_inputs(scheme):
    """Exact ties near every code and both rails, plus the specials a static
    scale meets: zeros, denormals, values far past the range."""
    inv = float(qm.reciprocal(scheme.scale))
    xs = [x for k, five in fold_quant_lut_pass.golden_inputs(inv)
          if -300 <= k <= 300 for x in five]
    s = scheme.scale
    xs += [0.0, -0.0, 1e-45, -1e-45, 300 * s, -300 * s, 1e6 * s, -1e6 * s,
           FLT_MAX, -FLT_MAX, 3e30, -3e30]
    return np.asarray(xs, F32)


def static_quantize_cases():
    out = []
    for name, scheme in SCHEMES.items():
        xs = tie_inputs(scheme)
        k = 64
        n = -(-xs.size // k) * k
        rows = np.resize(xs, n).reshape(-1, k)
        out.append(quantize_case(f"{name}: ties and rails, groups of 16 "
                                 f"(k 56 of 64)", scheme, rows[:, :56],
                                 k_pad=64, gsize=16))
        out.append(quantize_case(f"{name}: non-finite", scheme,
                                 NONFINITE[None], k_pad=16))
    return out


def conv_case(name, scheme, x, geom, k_pad=None, gsize=0):
    """Kind 5: the conv gather of every output position at the record's
    constants; expected from mirrors.im2col_codes (zx padding), a uint8
    range's codes and zero point shifted onto int8 as the record carries
    them."""
    x4 = qm.as_cdhw(np.asarray(x, F32))
    xq, sx, zx = qm.quantize_activation(x4.reshape(1, -1), scheme)
    if scheme.qmin >= 0:
        xq, zx = qm.uint8_to_int8(xq, zx)
    xq = xq.reshape(x4.shape)
    k = geom.k
    k_pad = k_pad or k
    cols = qm.im2col_codes(xq, geom, int(zx[0]), k_pad=k_pad)
    m = geom.m
    sxv = np.full(m, sx[0], F32)
    zxv = np.full(m, zx[0], np.int64)
    X, T = _terms(cols, zxv, sxv, k, k_pad, gsize)
    gather = struct.pack("<20I", *(int(getattr(geom, f)) for f in GATHER_FIELDS))
    return Case(name, 5, TERMS, k_pad, gsize, 0, m, k, act=act_quant(scheme),
                x=x4.reshape(-1), gather=gather,
                expected=[cols.astype(np.int8).tobytes(),
                          sxv.tobytes(), zxv.astype(np.int32).tobytes(),
                          X.tobytes(), T.tobytes()])


def static_conv_cases(rng):
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
    # The zero point far from 0 (the rail and the odd ones), so a padded tap
    # written as 0 instead of zx would show in every code it touches.
    schemes = [SCHEMES["int8 s 0.37 zp -128 (rail)"],
               SCHEMES["int8 s 2^-4 zp -7 (odd)"],
               SCHEMES["uint8 s 2^-5 zp 131 (odd)"],
               SCHEMES["int8 s 0.0071 zp 127 (rail)"]]
    out = []
    for i, (gname, geom) in enumerate(geoms.items()):
        scheme = schemes[i % len(schemes)]
        shape = (1, geom.c_in, geom.d_in, geom.h_in, geom.w_in)
        x = rng.standard_normal(shape).astype(F32)
        k_pad = -(-geom.k // 16) * 16
        out.append(conv_case(f"static zp {scheme.zero_point} {gname}", scheme,
                             x, geom, k_pad=k_pad,
                             gsize=16 if geom.k > 16 else 0))
    return out


def static_matmul_cases(rng):
    out = []
    odd, rail = SCHEMES["int8 s 2^-4 zp -7 (odd)"], SCHEMES[
        "int8 s 0.37 zp -128 (rail)"]
    x = np.stack([np.resize(tie_inputs(odd), 96)] + [
        (rng.standard_normal(96) * 4).astype(F32) for _ in range(3)])
    for pie in (True, False):
        out.append(matmul_case(
            f"static zp -7 x int8 symmetric per row, ties and rails "
            f"({'PIE' if pie else 'scalar'})", rng, odd, x, bits=8, n=8,
            pie=pie))
    out.append(matmul_case(
        "static zp -7 x int8 zero points per group of 32, ragged (k 80 of 96)",
        rng, odd, x[:, :80], bits=8, n=8, k_pad=96, gsize=32,
        zero_points=rng.integers(-128, 128, (8, 3))))
    out.append(matmul_case(
        "uint8 static x int8 per tensor zero point", rng,
        SCHEMES["uint8 s 2^-5 zp 131 (odd)"], x, bits=8, n=4,
        zero_points=np.full((4, 1), 5)))
    for fused in (False, True):
        out.append(matmul_case(
            f"static x int4 group 32 symmetric "
            f"({'fused int4 kernel' if fused else 'unpacked int4 codes'})",
            rng, odd, rng.standard_normal((1, 64)).astype(F32), bits=4, n=8,
            gsize=32, fused=fused))
    # The bound for affine activations x symmetric int8 weights: every code
    # at the positive rail, 255 from the zero point, against -128.
    bound = 65_792
    out.append(matmul_case(
        f"static zp -128 x int8 symmetric at the bound {bound:,}: |acc| = "
        f"{bound * 255 * 128:,}", rng, rail, np.full((1, bound), 1e3, F32),
        bits=8, n=1, codes=np.full((1, bound), -128), bias=False))
    return out


class OutCase:
    """Kind 7: values requantised by an EXT_OUT_QUANT record; expected from
    mirrors.fake_quant_output."""

    def __init__(self, name, scheme, ys):
        self.name = name
        record = out_quant(scheme.scale, scheme.zero_point, scheme.qmin,
                           scheme.qmax)
        want = qm.fake_quant_output(ys, scheme.scale, scheme.zero_point,
                                    scheme.qmin, scheme.qmax)
        self.expected = [np.asarray(want, F32).tobytes()]
        header = struct.pack("<IIIIIIIIIi", 7, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.record = header + _arr(*([b""] * 8)) + _arr(
            record.pack(), np.asarray(ys, F32).tobytes())


def out_values(scheme):
    """tie_inputs plus the infinities, which land on the rails as torch's
    clamp puts them (NaN is out of contract: torch's code for it is
    platform-defined)."""
    return np.concatenate([tie_inputs(scheme),
                           np.asarray([np.inf, -np.inf], F32)])


def out_cases():
    # The output range is kept as the model's quantize names it (uint8 is
    # not shifted: the codes never leave the op).
    return [OutCase(f"requantize {name}", scheme, out_values(scheme))
            for name, scheme in SCHEMES.items()] + [
        OutCase(f"requantize {name}: non-finite", scheme, NONFINITE)
        for name, scheme in SCHEMES.items()]


def cases():
    rng = np.random.default_rng(2028)
    return static_quantize_cases() + static_conv_cases(rng) + \
        static_matmul_cases(rng) + out_cases()


def run(runner, tmp_path, table):
    data = b"ASCF" + struct.pack("<II", 1, len(table)) + b"".join(
        c.record for c in table)
    (tmp_path / "cases.bin").write_bytes(data)
    done = subprocess.run(
        [str(runner), "--asym-conformance", str(tmp_path / "cases.bin"),
         str(tmp_path / "out.bin")], capture_output=True, text=True,
        timeout=600)
    assert done.returncode == 0, done.stderr[-1500:]
    out = (tmp_path / "out.bin").read_bytes()
    at, bad = 0, []
    for case in table:
        for i, want in enumerate(case.expected):
            n, = struct.unpack_from("<I", out, at)
            got = out[at + 4:at + 4 + n]
            at += 4 + n
            if got != want:
                bad.append(f"{case.name} [array {i}]")
    assert at == len(out)
    return bad


def test_static_activations_are_the_mirror(runner, tmp_path):
    bad = run(runner, tmp_path, cases())
    assert not bad, f"{len(bad)} arrays differ: {bad[:12]}"


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_mirror_codes_are_quantize_per_tensor(name):
    """quantize_activation -- the reference the cases above hold the runtime
    to -- is torch's own quantize_per_tensor.default on the static constants,
    ties and rails included."""
    scheme = SCHEMES[name]
    xs = tie_inputs(scheme)
    dtype = torch.uint8 if scheme.qmin >= 0 else torch.int8
    q = torch.ops.quantized_decomposed.quantize_per_tensor(
        torch.from_numpy(xs), scheme.scale, scheme.zero_point, scheme.qmin,
        scheme.qmax, dtype)
    mine, _, _ = qm.quantize_activation(xs[None], scheme)
    assert np.array_equal(mine[0], q.to(torch.int64).numpy()), name


def test_ties_reach_both_parities():
    """The inputs hold exact ties next to odd and even codes for every scheme
    (otherwise the tie cases would prove nothing)."""
    for name, scheme in SCHEMES.items():
        xs = tie_inputs(scheme)
        v = (xs * qm.reciprocal(scheme.scale)).astype(F32)
        ties = v[np.isfinite(v) & (np.abs(v) < 400)
                 & (v - np.floor(v) == F32(0.5))]
        assert {int(np.floor(t)) % 2 for t in ties} == {0, 1}, name


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_non_finite_mirror_is_the_portable_kernel(name):
    """On NaN, the infinities and +-FLT_MAX, quantize_activation and
    fake_quant_output -- the references above -- give ExecuTorch's own
    quantize_per_tensor kernel's codes (since 1.6: NaN the zero point, an
    infinity its rail) and their dequantised values. Not torch's own ops:
    their code for NaN is platform-defined."""
    scheme = SCHEMES[name]
    dtype = torch.uint8 if scheme.qmin >= 0 else torch.int8
    q = portable_quantize(NONFINITE, scheme.scale, scheme.zero_point,
                          scheme.qmin, scheme.qmax, dtype)
    q = q.to(torch.int64).numpy()
    mine, _, _ = qm.quantize_activation(NONFINITE[None], scheme)
    assert np.array_equal(mine[0], q), name
    assert (q[np.isnan(NONFINITE)] == scheme.zero_point).all(), name
    want = qm.dequantize_per_tensor(q, scheme.scale, scheme.zero_point)
    got = qm.fake_quant_output(NONFINITE, scheme.scale, scheme.zero_point,
                               scheme.qmin, scheme.qmax)
    assert np.array_equal(np.asarray(got, F32).view(np.uint32),
                          np.asarray(want, F32).view(np.uint32)), name


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_output_mirror_is_torchs_quantize_then_dequantize(name):
    """fake_quant_output -- what kind 7 holds requantize to -- is torch's own
    dequantize_per_tensor(quantize_per_tensor(y)) on the output's constants,
    ties, rails and infinities included."""
    scheme = SCHEMES[name]
    ys = out_values(scheme)
    dtype = torch.uint8 if scheme.qmin >= 0 else torch.int8
    qd = torch.ops.quantized_decomposed
    args = (scheme.scale, scheme.zero_point, scheme.qmin, scheme.qmax, dtype)
    torch_y = qd.dequantize_per_tensor(
        qd.quantize_per_tensor(torch.from_numpy(ys), *args), *args)
    mine = qm.fake_quant_output(ys, *args[:4])
    assert np.array_equal(np.asarray(mine, F32).view(np.uint32),
                          torch_y.numpy().view(np.uint32)), name
