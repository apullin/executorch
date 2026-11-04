# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic activation quantisers: the runtime's arithmetic at its limits,
bit for bit against mirrors.py and torch's own ops.

esp32_host_runner --asym-conformance (kinds 4-6) runs esp32_asym.cpp's own
functions -- choose, quantize_row, quantize_conv_row, then accumulate /
accumulate_fused with the activation zero point's zx * S' term -- on inputs
chosen to be adversarial, and every result must equal the reference in every
bit:

  - codes, scales and zero points against mirrors.quantize_activation
    AND against torch: torchao's choose_qparams_affine + quantize_affine for
    the per-token schemes, quantized_decomposed's choose_qparams(_symmetric)
    .tensor + quantize_per_tensor.tensor for the per-tensor ones;
  - exact rounding ties with odd and even zero points (round-before-zp
    against round-after-zp would differ there);
  - rows that are all positive, all negative, constant, all zero, or a
    single element;
  - non-finite rows (device against the mirror only: torch's choosers
    propagate a NaN, and its code for one is platform-defined): a NaN is
    skipped by the chooser and takes the zero point, an infinity makes the
    scale infinite, ExecuTorch 1.6's rule (runtime/quant_rules.h);
  - every eps floor: 2^-12 (XNNPACK), 2^-16 (Arm), FLT_EPSILON (PT2E's
    default, the legacy quantiser), FLT_MIN (torchao's default);
  - uint8 ranges shifted onto int8, and the symmetric grids (127, 127.5);
  - ragged groups (k below k_pad, a short last group), X_g and T_g with
    their real counts n_g;
  - the conv gather's padding written as zx at every edge: padding, stride,
    dilation, transposed off-lattice taps (with output padding and
    dilation), conv1d, conv3d and a depthwise channel, against
    mirrors.im2col_codes;
  - the corrected dot end to end against mirrors.affine_matmul:
    -128 x -128 products, int8 x int8 zero points at their bound (33,024)
    and affine x symmetric int8 at 65,792, |acc| near 2^31, the fused int4
    kernel and the unpacked path, int16 and int32 weight sums, float offsets
    with X'_g = X_g - n_g * zx.

One group past each bound is refused by name at export (test_asym_act.py) and
by the runtime at load (asym_blob_cases' dot-bound cases).

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import struct
import subprocess

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

pass
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.serialization.asym_records import (
    act_quant,
    weight_sums_table,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32

FUSED, PIE, BIAS, SUMS16, SUMS32, TERMS = 1, 2, 16, 32, 64, 128
ZP_NONE, ZP_INT8 = 0, 1
OFF_NONE, OFF_FP32 = 0, 1


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def _arr(*parts):
    return b"".join(struct.pack("<I", len(p)) + p for p in parts)


class Case:
    """One dynamic-activation case: its wire record and the expected arrays."""

    def __init__(self, name, kind, flags, k_pad, gsize, n, rows, k, *,
                 act, x, codes=b"", scales=b"", zps=b"", offsets=b"",
                 bias=b"", sums=b"", gather=b"", zp_kind=ZP_NONE,
                 offset_kind=OFF_NONE, storage_bias=0, expected=()):
        self.name, self.expected = name, list(expected)
        header = struct.pack("<IIIIIIIIIi", kind, flags, k_pad, gsize, n, rows,
                             k, zp_kind, offset_kind, storage_bias)
        self.record = header + _arr(b"", b"", codes, scales, zps, offsets,
                                    bias, b"", act.pack(),
                                    np.asarray(x, F32).tobytes(), sums, gather)


def _groups(k_pad, gsize):
    g = gsize or k_pad
    return -(-k_pad // g)


def _terms(xq, zx, sx, k, k_pad, gsize):
    """X_g over k_pad and T_g with n_g real elements, as the mirrors do."""
    g = gsize or k_pad
    spans = [(a, min(a + g, k_pad)) for a in range(0, k_pad, g)]
    X = np.stack([xq[:, a:b].sum(axis=1) for a, b in spans], axis=1)
    real = [(a, min(b, k)) for a, b in spans]
    T = qm.offset_terms(X, zx, sx, real)
    return X.astype(np.int32), T.astype(F32)


def quantize_case(name, scheme, x, k_pad=None, gsize=0):
    """Kind 4: quantise rows x k with the scheme's record; expected codes,
    scales, zero points, X_g and T_g from mirrors.py."""
    x = np.asarray(x, F32)
    rows, k = x.shape
    k_pad = k_pad or k
    xq, sx, zx = qm.quantize_activation(x, scheme)
    if scheme.qmin >= 0:                      # uint8 rides shifted onto int8
        xq, zx = qm.uint8_to_int8(xq, zx)
    codes = np.zeros((rows, k_pad), np.int64)
    codes[:, :k] = xq
    X, T = _terms(codes, zx, sx, k, k_pad, gsize)
    return Case(name, 4, TERMS, k_pad, gsize, 0, rows, k, act=act_quant(scheme),
                x=x, expected=[codes.astype(np.int8).tobytes(),
                               sx.astype(F32).tobytes(),
                               zx.astype(np.int32).tobytes(), X.tobytes(),
                               T.tobytes()])


GATHER_FIELDS = ("h_in", "w_in", "c_in", "h_out", "w_out", "kh", "kw",
                 "stride_h", "stride_w", "pad_h", "pad_w", "dil_h", "dil_w",
                 "d_in", "d_out", "kd", "stride_d", "pad_d", "dil_d",
                 "transposed")


def conv_case(name, scheme, x, geom, k_pad=None, gsize=0):
    """Kind 5: the conv gather of every output position at the tensor's
    parameters; expected from mirrors.im2col_codes (zx padding)."""
    x4 = qm.as_cdhw(np.asarray(x, F32))
    xq, sx, zx = qm.quantize_activation(x4.reshape(1, -1), scheme)
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


def _pack_i4(codes):
    u = (np.asarray(codes, dtype=np.int16) + 8).astype(np.uint8)
    return ((u[:, 1::2] << 4) | (u[:, 0::2] & 0x0F)).astype(np.uint8)


def matmul_case(name, rng, scheme, x, *, bits, n, k_pad=None, gsize=0,
                codes=None, zero_points=None, offsets=None, fused=False,
                pie=True, bias=True):
    """Kind 6: quantise each row, then every column's corrected dot and the
    two-rounding epilogue; expected from mirrors.affine_matmul."""
    x = np.asarray(x, F32)
    rows, k = x.shape
    k_pad = k_pad or k
    groups = _groups(k_pad, gsize)
    lo, hi = (-8, 7) if bits == 4 else (-128, 127)
    if codes is None:
        codes = rng.integers(lo, hi + 1, size=(n, k))
    codes = np.asarray(codes, np.int64)
    full = np.zeros((n, k_pad), np.int64)
    full[:, :k] = codes
    scales = (rng.random((n, groups)) * 0.01 + 1e-4).astype(F32)
    b = (rng.standard_normal(n) * 0.1).astype(F32) if bias else None
    zw = None
    if zero_points is not None:
        zw = np.broadcast_to(np.asarray(zero_points, np.int64),
                             (n, groups)).copy()
    m = None if offsets is None else np.asarray(offsets, F32).reshape(n, groups)
    trace = qm.affine_matmul(x, codes, scales, b, act=scheme, bits=bits,
                             gsize=gsize or None, zero_points=zw, offsets=m,
                             i4_fused=fused)
    act = act_quant(scheme)
    flags = (FUSED if fused else 0) | (PIE if pie else 0) | (BIAS if bias else 0)
    sums = b""
    if act.affine:
        table = weight_sums_table(codes, n, groups, k, gsize, zw)
        longest = gsize if 0 < gsize < k else k
        if longest * qm.weight_span(bits, zw is not None) <= 32767:
            flags |= SUMS16
            sums = table.astype(np.int16).tobytes()
        else:
            flags |= SUMS32
            sums = table.astype(np.int32).tobytes()
    stored = _pack_i4(full) if fused else full.astype(np.int8)
    return Case(name, 6, flags, k_pad, gsize, n, rows, k, act=act, x=x,
                codes=stored.tobytes(), scales=scales.tobytes(),
                zps=b"" if zw is None else zw.astype(np.int8).tobytes(),
                offsets=b"" if m is None else m.tobytes(),
                bias=b"" if b is None else b.tobytes(), sums=sums,
                zp_kind=ZP_INT8 if zw is not None else ZP_NONE,
                offset_kind=OFF_FP32 if m is not None else OFF_NONE,
                expected=[trace.y.astype(F32).tobytes()])


# --- adversarial activations -------------------------------------------------------------

def tie_row(k, zx_odd, s=2.0 ** -4):
    """Scale exactly s, zero point odd or even, exact halves (j + 0.5) s."""
    lo = 3 if zx_odd else 4
    row = np.zeros(k, F32)
    row[0], row[1] = -lo * s, (255 - lo) * s
    halves = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 7.5, -2.5, 100.5, -3.5])
    row[2:2 + halves.size] = halves * s
    return row


def floor_row(k, eps, rng):
    """A range narrower than 255 * eps: the scale floors at eps."""
    return (rng.uniform(-60.0, 100.0, k) * eps).astype(F32)


def adversarial_rows(k, rng, eps):
    return np.stack([
        tie_row(k, True), tie_row(k, False),
        np.abs(rng.standard_normal(k)).astype(F32) + 0.25,      # all positive
        -np.abs(rng.standard_normal(k)).astype(F32) - 0.25,     # all negative
        np.full(k, 0.75, F32),                                  # constant
        np.full(k, -1.5, F32),
        np.zeros(k, F32),                                       # all zero
        floor_row(k, eps, rng),
        rng.standard_normal(k).astype(F32) * 3.0,
    ])


def nonfinite_rows(k):
    """ExecuTorch 1.6's rule (runtime/quant_rules.h) under a dynamic chooser:
    a NaN among finite values is skipped by the chooser and takes the zero
    point; an infinity makes the scale infinite, so every code is the zero
    point (affine: qmin); a row of NaNs is an all-zero row's."""
    nan, inf = F32(np.nan), F32(np.inf)
    rows = np.zeros((4, k), F32)
    rows[0, :6] = [0.5, nan, -1.75, 3.25, -nan, 2.0]
    rows[1, :6] = [0.5, inf, -1.75, 3.0, nan, 1.0]
    rows[2, :6] = [0.5, -inf, 1.75, 3.0, 0.0, 1.0]
    rows[3] = nan
    return rows


SCHEMES = {
    "A2/xnnpack": qm.A2_XNNPACK, "A2/arm": qm.A2_ARM,
    "A3/torchao": qm.A3_TORCHAO, "A3/legacy": qm.A3_LEGACY,
    "A7/torchao": qm.A7_TORCHAO, "A1": qm.A1,
    "A1/[-128,127] eps 2^-16": qm.ActScheme("s", "symmetric", "tensor",
                                            "rne_before_zp", -128, 127,
                                            eps=qm.EPS_ARM),
    "uint8 affine per token": qm.ActScheme("u8", "affine", "row",
                                           "rne_before_zp", 0, 255,
                                           eps=qm.FLT_MIN),
    "PT2E affine at FLT_EPSILON": qm.ActScheme("f", "affine", "tensor",
                                               "rne_before_zp", -128, 127,
                                               eps=qm.FLT_EPSILON),
}


def quantize_cases(rng):
    out = []
    for name, scheme in SCHEMES.items():
        eps = max(scheme.eps, 2.0 ** -126)
        rows = adversarial_rows(48, rng, eps)
        if scheme.granularity == "row":
            out.append(quantize_case(f"{name}: adversarial rows, groups of 16 "
                                     f"(k 40 of 48)", scheme, rows[:, :40],
                                     k_pad=48, gsize=16))
            out.append(quantize_case(f"{name}: one element", scheme,
                                     rows[2:4, :1], k_pad=16))
            out.append(quantize_case(f"{name}: non-finite rows", scheme,
                                     nonfinite_rows(12), k_pad=16))
        else:
            for i, row in enumerate(rows):
                out.append(quantize_case(f"{name}: tensor {i}", scheme,
                                         np.stack([row, row[::-1]])))
            for i, row in enumerate(nonfinite_rows(12)):
                out.append(quantize_case(f"{name}: non-finite tensor {i}",
                                         scheme, np.stack([row, row[::-1]]),
                                         k_pad=16))
    return out


def conv_cases(rng):
    from executorch.backends.esp32s3.operator_support.conv import ConvGeom

    out = []
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
        "5x5 pad 2 (asymmetric reach)": ConvGeom(h_in=4, w_in=3, c_in=1,
                                                 h_out=4, w_out=3, c_out=2,
                                                 kh=5, kw=5, stride_h=1,
                                                 stride_w=1, pad_h=2, pad_w=2,
                                                 dil_h=1, dil_w=1),
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
        "transposed 2x2 stride 2 pad 0": ConvGeom(h_in=3, w_in=4, c_in=2,
                                                  h_out=6, w_out=8, c_out=2,
                                                  kh=2, kw=2, stride_h=2,
                                                  stride_w=2, pad_h=0,
                                                  pad_w=0, dil_h=1, dil_w=1,
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
    for (gname, geom), (sname, scheme) in zip(
            geoms.items(), [("A2/xnnpack", qm.A2_XNNPACK),
                            ("A2/arm", qm.A2_ARM)] * 6):
        shape = (1, geom.c_in, geom.d_in, geom.h_in, geom.w_in)
        x = rng.standard_normal(shape).astype(F32)
        if gname.startswith("transposed 3x3"):
            # All positive: zx = qmin, the farthest a padded tap's code sits
            # from a real one.
            x = np.abs(x) + 0.1
        k_pad = -(-geom.k // 16) * 16
        out.append(conv_case(f"{sname} {gname}", scheme, x, geom, k_pad=k_pad,
                             gsize=16 if geom.k > 16 else 0))
    return out


def matmul_cases(rng):
    a3, a2, a7 = qm.A3_TORCHAO, qm.A2_XNNPACK, qm.A7_TORCHAO
    out = []
    x = adversarial_rows(96, rng, qm.FLT_MIN)
    for pie in (True, False):
        out.append(matmul_case(
            f"A3 x int8 symmetric per row, adversarial rows "
            f"({'PIE' if pie else 'scalar'})", rng, a3, x, bits=8, n=8,
            pie=pie))
    out.append(matmul_case(
        "A3 x int8 zero points per group of 32, ragged (k 80 of 96)", rng, a3,
        x[:, :80], bits=8, n=8, k_pad=96, gsize=32,
        zero_points=rng.integers(-128, 128, (8, 3))))
    out.append(matmul_case(
        "A2 per tensor x int8 zero points per row", rng, a2, x, bits=8, n=8,
        zero_points=np.array([[-128], [127], [0], [1], [-1], [64], [-64], [5]])))
    out.append(matmul_case(
        "A7 x int8 symmetric (-128 x -128 reachable)", rng, a7, x, bits=8,
        n=4, codes=np.full((4, 96), -128)))
    for fused in (False, True):
        tag = "fused int4 kernel" if fused else "unpacked int4 codes"
        out.append(matmul_case(
            f"A3 x int4 zero points per group of 32 ({tag})", rng, a3,
            rng.standard_normal((2, 64)).astype(F32), bits=4, n=8, gsize=32,
            zero_points=rng.integers(-8, 8, (8, 2)), fused=fused))
        out.append(matmul_case(
            f"A3 x int4 group 128 symmetric ({tag})", rng, a3,
            rng.standard_normal((1, 256)).astype(F32), bits=4, n=4,
            gsize=128, fused=fused))
        out.append(matmul_case(
            f"A3 x int4 float offsets, X' = X - n zx ({tag})", rng, a3,
            np.abs(rng.standard_normal((1, 64))).astype(F32), bits=4, n=4,
            gsize=32, offsets=rng.standard_normal((4, 2)) * 0.05,
            zero_points=np.full((4, 2), -8), fused=fused))
    # The bounds: one row whose codes sit 255 from its zero point everywhere
    # (an all-positive row at its maximum: xq = 127, zx = -128).
    for zero, bound, c in ((False, 65_792, -128), (True, 33_024, -128)):
        xb = np.full((1, bound), 1.0, F32)
        zp = np.array([[127]]) if zero else None
        out.append(matmul_case(
            f"A3 x int8 {'zero points' if zero else 'symmetric'} at the bound "
            f"{bound:,}: |acc| = {bound * 255 * (255 if zero else 128):,}", rng,
            a3, xb, bits=8, n=1, codes=np.full((1, bound), c),
            zero_points=zp, bias=False))
    out.append(matmul_case(
        "A3 x int8 per row, k 300: int32 weight sums", rng, a3,
        rng.standard_normal((2, 300)).astype(F32), bits=8, n=4, k_pad=304,
        zero_points=rng.integers(-128, 128, (4, 1))))
    return out


def cases():
    rng = np.random.default_rng(2027)
    return quantize_cases(rng) + conv_cases(rng) + matmul_cases(rng)


def test_dynamic_activations_are_the_mirror(runner, tmp_path):
    table = cases()
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
    assert not bad, f"{len(bad)} arrays differ: {bad[:12]}"


# --- the mirrors' codes are torch's -------------------------------------------------------

def _torch_codes(scheme, x):
    """Codes, scale and zero point from torch's own ops for the scheme."""
    xt = torch.from_numpy(np.asarray(x, F32))
    if scheme.granularity == "row":
        import torchao  # noqa: F401

        ops = torch.ops.torchao
        mapping = "SYMMETRIC" if scheme.chooser == "symmetric" else "ASYMMETRIC"
        dtype = torch.uint8 if scheme.qmin >= 0 else torch.int8
        block = [1, xt.shape[-1]]
        eps = None if scheme.eps == qm.FLT_MIN else scheme.eps
        s, z = ops.choose_qparams_affine(xt, mapping, block, dtype, scheme.qmin,
                                         scheme.qmax, eps, None, torch.int32)
        q = ops.quantize_affine(xt, block, s, z, dtype, scheme.qmin,
                                scheme.qmax)
        return (q.to(torch.int64).numpy(), s.reshape(-1).numpy(),
                z.reshape(-1).to(torch.int64).numpy())
    qd = torch.ops.quantized_decomposed
    choose = (qd.choose_qparams_symmetric.tensor
              if scheme.chooser == "symmetric" else qd.choose_qparams.tensor)
    s, z = choose(xt, scheme.qmin, scheme.qmax, scheme.eps, torch.int8)
    q = qd.quantize_per_tensor.tensor(xt, s, z, scheme.qmin, scheme.qmax,
                                      torch.int8)
    rows = xt.shape[0]
    return (q.to(torch.int64).numpy(), np.full(rows, float(s), F32),
            np.full(rows, int(z), np.int64))


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_mirror_codes_are_torchs(name):
    """quantize_activation -- the reference every case above is held to --
    gives torch's own codes, scales and zero points on the adversarial
    rows (the bar: codes identical to quantize_per_tensor and
    quantize_affine)."""
    scheme = SCHEMES[name]
    rng = np.random.default_rng(7)
    rows = adversarial_rows(48, rng, max(scheme.eps, 2.0 ** -126))
    blocks = [rows] if scheme.granularity == "row" else [r[None] for r in rows]
    for block in blocks:
        xq, sx, zx = qm.quantize_activation(block, scheme)
        tq, ts, tz = _torch_codes(scheme, block)
        assert np.array_equal(xq, tq), name
        assert np.array_equal(sx.astype(F32).view(np.uint32),
                              ts.astype(F32).view(np.uint32)), name
        assert np.array_equal(zx, tz), name
