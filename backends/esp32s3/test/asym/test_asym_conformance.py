# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight zero points and float offsets: the runtime's arithmetic at its
limits, bit for bit against mirrors.py.

esp32_host_runner --asym-conformance runs esp32_asym.cpp's own column math
(affine_weights, codes_row, accumulate, accumulate_fused with the C references
of the PIE kernels, fold_zero_points, dequant_row) on cases chosen to be
adversarial, and every result must equal mirrors.py's in every bit:

  - zero points at both ends of the code range (int4 -8/7, int8 -128/127)
    against codes at the other end and activations at +-127;
  - unsigned sources stored signed (storage bias 8 or 128, with or without a
    zero-point table);
  - ragged last groups (k < k_pad, the tail outside every count);
  - an int8 zero-point weight at K = 66,304, the exactness bound for the
    backend's activations, with the worst-case operands (|acc| = 2^31 - 228,353);
  - the fused int4 kernel and the unfused path, the PIE and the scalar dot;
  - fp16 and fp32 scales and offsets (W-off: tinygemm, GGUF);
  - the staged fold (c - zw), and every embedding dequantisation: integer
    zero points, fractional float zero points, offsets.

One group past the bound is refused, by the exporter by name
(asym_records.check_weight_bounds) and by the runtime at load (asym_blob_cases'
"dot-bound" case for backend activations, test_asym_refusals.py).

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import struct
import subprocess

import numpy as np
import pytest

from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.serialization.asym_records import (
    SchemeRefused,
    check_weight_bounds,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32

FUSED, PIE, FP16_SCALES, I4_TABLE, BIAS = 1, 2, 4, 8, 16
ZP_NONE, ZP_INT8, ZP_FP32 = 0, 1, 2
OFF_NONE, OFF_FP32, OFF_FP16 = 0, 1, 2
BOUND = 66_304            # int8 zero-point weights x the backend's +-127 rows


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def _pack_i4(codes):
    """pack_rows' nibbles: element 2i low, 2i+1 high, each code + 8."""
    u = (np.asarray(codes, dtype=np.int16) + 8).astype(np.uint8)
    return ((u[:, 1::2] << 4) | (u[:, 0::2] & 0x0F)).astype(np.uint8)


def _table(values, kind_fp16):
    v = np.asarray(values, dtype=F32)
    return v.astype(np.float16).tobytes() if kind_fp16 else v.tobytes()


class Case:
    """One case: its wire record and its expected result bytes."""

    def __init__(self, name, kind, flags, k_pad, gsize, n, rows, k, zp_kind,
                 offset_kind, storage_bias, arrays, expected):
        self.name, self.expected = name, expected
        header = struct.pack("<IIIIIIIIIi", kind, flags, k_pad, gsize, n, rows,
                             k, zp_kind, offset_kind, storage_bias)
        body = b"".join(struct.pack("<I", len(a)) + a for a in arrays)
        self.record = header + body


def _groups(k_pad, gsize):
    g = gsize or k_pad
    return -(-k_pad // g)


def matmul_case(name, rng, *, bits, k, k_pad, gsize, n, rows, zero_points=None,
                storage_bias=0, offsets=None, fused=False, pie=True,
                fp16_scales=False, fp16_offsets=False, codes=None, xq=None,
                bias=True):
    """A matmul: random (or given) codes and activation codes, every group's
    scale, and the mirror's result -- corrected dots, offset terms and the
    two-rounding epilogue (mirrors.py's own functions)."""
    groups = _groups(k_pad, gsize)
    assert groups == -(-k // (gsize or k_pad)) or gsize == 0
    lo, hi = (-8, 7) if bits == 4 else (-128, 127)
    if codes is None:
        codes = rng.integers(lo, hi + 1, size=(n, k_pad))
    codes = np.asarray(codes, dtype=np.int64)
    codes[:, k:] = 0                      # pack_rows' zero tail
    if xq is None:
        xq = rng.integers(-127, 128, size=(rows, k_pad))
    xq = np.asarray(xq, dtype=np.int64)
    xq[:, k:] = 0                         # the quantisers' zero tail
    sx = (rng.random(rows) * 0.02 + 1e-3).astype(F32)
    scales = (rng.random((n, groups)) * 0.01 + 1e-4).astype(F32)
    if fp16_scales:
        scales = scales.astype(np.float16).astype(F32)
    b = (rng.standard_normal(n) * 0.1).astype(F32) if bias else None
    zw = None
    if zero_points is not None:
        zw = np.broadcast_to(np.asarray(zero_points, dtype=np.int64),
                             (n, groups)).copy()
    elif storage_bias:
        zw = np.full((n, groups), -storage_bias, dtype=np.int64)
    m = None
    if offsets is not None:
        m = np.asarray(offsets, dtype=F32).reshape(n, groups)
        if fp16_offsets:
            m = m.astype(np.float16).astype(F32)
    g = gsize if gsize else None
    acc, X, _ = qm.corrected_dots(xq[:, :k], np.zeros(rows, np.int64),
                                  codes[:, :k], zw, g, i4_fused=fused)
    T = None
    if m is not None:
        T = qm.offset_terms(X, np.zeros(rows, np.int64), sx,
                            qm.group_spans(k, g))
    y = qm.device_epilogue(acc, sx, scales, b, m, T)
    stored = _pack_i4(codes) if fused else codes.astype(np.int8)
    flags = ((FUSED if fused else 0) | (PIE if pie else 0)
             | (FP16_SCALES if fp16_scales else 0) | (BIAS if bias else 0))
    arrays = [xq.astype(np.int8).tobytes(), sx.tobytes(), stored.tobytes(),
              _table(scales, fp16_scales),
              b"" if zero_points is None else
              np.asarray(zw, np.int8).tobytes(),
              b"" if m is None else _table(m, fp16_offsets),
              b"" if b is None else b.tobytes(), b""]
    return Case(name, 1, flags, k_pad, gsize, n, rows, k,
                ZP_INT8 if zero_points is not None else ZP_NONE,
                (OFF_FP16 if fp16_offsets else OFF_FP32) if m is not None
                else OFF_NONE, storage_bias, arrays,
                np.asarray(y, F32).tobytes())


def gather_case(name, rng, *, bits, d, gsize, vocab, indices, zero_points=None,
                float_zero_points=None, storage_bias=0, offsets=None,
                fp16_offsets=False):
    """An embedding gather: every row dequantised as the reference does."""
    groups = -(-d // (gsize or d))
    lo, hi = (-8, 7) if bits == 4 else (-128, 127)
    codes = rng.integers(lo, hi + 1, size=(vocab, d)).astype(np.int64)
    scales = (rng.random((vocab, groups)) * 0.05 + 1e-3).astype(F32)
    g = gsize or d
    expand = lambda t: np.asarray(t)[:, np.arange(d) // g]  # noqa: E731
    idx = np.asarray(indices)
    s = expand(scales)
    if float_zero_points is not None:
        z = np.asarray(float_zero_points, dtype=F32).reshape(vocab, groups)
        q = (codes + storage_bias).astype(F32)
        out = ((q - expand(z)).astype(F32) * s).astype(F32)
        zp_kind, zp_bytes = ZP_FP32, z.tobytes()
    else:
        zw = (np.zeros((vocab, groups), np.int64) if zero_points is None
              else np.broadcast_to(np.asarray(zero_points, np.int64),
                                   (vocab, groups)).copy())
        if zero_points is None and storage_bias:
            zw[:] = -storage_bias
        out = ((codes - expand(zw)).astype(F32) * s).astype(F32)
        zp_kind = ZP_INT8 if zero_points is not None else ZP_NONE
        zp_bytes = zw.astype(np.int8).tobytes() if zero_points is not None \
            else b""
    m_bytes, offset_kind = b"", OFF_NONE
    if offsets is not None:
        m = np.asarray(offsets, dtype=F32).reshape(vocab, groups)
        if fp16_offsets:
            m = m.astype(np.float16).astype(F32)
        out = (out + expand(m)).astype(F32)
        m_bytes = _table(m, fp16_offsets)
        offset_kind = OFF_FP16 if fp16_offsets else OFF_FP32
    k_pad = d + (d % 2 if bits == 4 else 0)
    padded = np.zeros((vocab, k_pad), np.int64)
    padded[:, :d] = codes
    stored = _pack_i4(padded) if bits == 4 else padded.astype(np.int8)
    arrays = [b"", b"", stored.tobytes(), scales.tobytes(), zp_bytes, m_bytes,
              b"", idx.astype(np.uint32).tobytes()]
    return Case(name, 2, I4_TABLE if bits == 4 else 0, k_pad, gsize, vocab,
                len(idx), d, zp_kind, offset_kind, storage_bias, arrays,
                out[idx].astype(F32).tobytes())


def fold_case(name, rng, *, k_pad, gsize, n, zero_points=None, storage_bias=0):
    """FLAG_STAGE_I8's fold: expanded int4 codes minus each group's zero
    point, the k_pad tail included."""
    groups = _groups(k_pad, gsize)
    codes = rng.integers(-8, 8, size=(n, k_pad)).astype(np.int64)
    zw = (np.broadcast_to(np.asarray(zero_points, np.int64), (n, groups))
          if zero_points is not None else np.full((n, groups), -storage_bias))
    want = codes - np.asarray(zw)[:, np.arange(k_pad) // (gsize or k_pad)]
    assert want.min() >= -15 and want.max() <= 15
    arrays = [b"", b"", codes.astype(np.int8).tobytes(),
              np.ones((n, groups), F32).tobytes(),
              b"" if zero_points is None else
              np.asarray(zw, np.int8).tobytes(), b"", b"", b""]
    return Case(name, 3, 0, k_pad, gsize, n, 0, k_pad,
                ZP_INT8 if zero_points is not None else ZP_NONE, OFF_NONE,
                storage_bias, arrays, want.astype(np.int8).tobytes())


def cases():
    rng = np.random.default_rng(2026)
    ends8 = np.array([[-128], [127], [-128], [127], [0], [1], [-1], [63]])
    ends4 = np.array([[-8], [7], [-8], [7], [0], [3], [-3], [5]])
    out = []
    for pie in (True, False):
        out.append(matmul_case(
            f"i8 per-row zero points at the range ends ({'PIE' if pie else 'scalar'})",
            rng, bits=8, k=96, k_pad=96, gsize=0, n=8, rows=3,
            zero_points=ends8, pie=pie))
    out.append(matmul_case(
        "i8 groups of 16, ragged (k 40 of 48), zero points per group", rng,
        bits=8, k=40, k_pad=48, gsize=16, n=8, rows=2,
        zero_points=rng.integers(-128, 128, size=(8, 3))))
    out.append(matmul_case(
        "i8 uint8 source: storage bias 128, no table", rng, bits=8, k=64,
        k_pad=64, gsize=0, n=4, rows=2, storage_bias=128))
    out.append(matmul_case(
        "i8 uint8 source with a table (zw = z - 128)", rng, bits=8, k=64,
        k_pad=64, gsize=32, n=4, rows=2, storage_bias=128,
        zero_points=rng.integers(-128, 128, size=(4, 2))))
    out.append(matmul_case(
        "i8 groups of 32, fp16 scales and fp32 offsets", rng, bits=8, k=64,
        k_pad=64, gsize=32, n=4, rows=2, fp16_scales=True,
        zero_points=rng.integers(-128, 128, size=(4, 2)),
        offsets=rng.standard_normal((4, 2)) * 0.05))
    # The exactness bound: x = +127 everywhere against c - zw = -255 and +255.
    k = BOUND
    worst = np.empty((2, k), np.int64)
    worst[0], worst[1] = -128, 127
    out.append(matmul_case(
        f"i8 zero points at the bound: one group of {k}, |acc| = "
        f"{k * 127 * 255:,}", rng, bits=8, k=k, k_pad=k, gsize=0, n=2, rows=1,
        zero_points=np.array([[127], [-128]]), codes=worst,
        xq=np.full((1, k), 127), bias=False))
    for fused in (False, True):
        tag = "fused int4 kernel" if fused else "unpacked int4 codes"
        out.append(matmul_case(
            f"i4 groups of 32, zero points at the range ends ({tag})", rng,
            bits=4, k=256, k_pad=256, gsize=32, n=8, rows=1,
            zero_points=np.hstack([ends4] * 8), fused=fused))
        out.append(matmul_case(
            f"i4 per-row zero points, ragged k 200 of 224 ({tag})", rng,
            bits=4, k=200, k_pad=224, gsize=0, n=8, rows=1,
            zero_points=ends4, fused=fused))
        out.append(matmul_case(
            f"i4 GGUF: storage bias 8 (zero point -8), fp16 offsets ({tag})",
            rng, bits=4, k=128, k_pad=128, gsize=32, n=4, rows=1,
            storage_bias=8, offsets=rng.integers(0, 64, (4, 4)) / 64.0,
            fp16_offsets=True, fused=fused))
        out.append(matmul_case(
            f"i4 tinygemm: zero point 0, fp32 offsets ({tag})", rng, bits=4,
            k=64, k_pad=64, gsize=32, n=4, rows=1,
            offsets=rng.standard_normal((4, 2)) * 0.05, fused=fused))
    out.append(matmul_case(
        "i4 unpacked, several rows (a row block)", rng, bits=4, k=64,
        k_pad=64, gsize=32, n=8, rows=5,
        zero_points=rng.integers(-8, 8, size=(8, 2))))
    out.append(gather_case(
        "int4 table, zero points at the range ends", rng, bits=4, d=64,
        gsize=32, vocab=8, indices=[0, 1, 7, 3, 3],
        zero_points=np.hstack([ends4, ends4[::-1]])))
    out.append(gather_case(
        "int8 table, fractional float zero points on uint8 codes", rng,
        bits=8, d=40, gsize=32, vocab=6, indices=[5, 0, 2],
        float_zero_points=rng.random((6, 2)) * 255.0, storage_bias=128))
    out.append(gather_case(
        "int4 table, tinygemm fp32 offsets, ragged d 40", rng, bits=4, d=40,
        gsize=32, vocab=6, indices=[1, 4],
        offsets=rng.standard_normal((6, 2)) * 0.05))
    out.append(gather_case(
        "int4 table, GGUF storage bias 8 and fp16 offsets", rng, bits=4,
        d=64, gsize=32, vocab=5, indices=[4, 0],
        storage_bias=8, offsets=rng.integers(0, 64, (5, 2)) / 64.0,
        fp16_offsets=True))
    out.append(gather_case(
        "int8 table, uint8 source without a table (storage bias 128)", rng,
        bits=8, d=32, gsize=0, vocab=4, indices=[3, 1], storage_bias=128))
    out.append(fold_case(
        "staged fold, zero points at the range ends", rng, k_pad=64,
        gsize=32, n=8, zero_points=np.hstack([ends4, ends4[::-1]])))
    out.append(fold_case(
        "staged fold, storage bias 8 without a table", rng, k_pad=64,
        gsize=0, n=4, storage_bias=8))
    return out


def test_column_math_is_the_mirror(runner, tmp_path):
    table = cases()
    data = b"ASCF" + struct.pack("<II", 1, len(table)) + b"".join(
        c.record for c in table)
    (tmp_path / "cases.bin").write_bytes(data)
    done = subprocess.run(
        [str(runner), "--asym-conformance", str(tmp_path / "cases.bin"),
         str(tmp_path / "out.bin")], capture_output=True, text=True,
        timeout=300)
    assert done.returncode == 0, done.stderr[-1500:]
    out = (tmp_path / "out.bin").read_bytes()
    at, bad = 0, []
    for case in table:
        n, = struct.unpack_from("<I", out, at)
        got = out[at + 4:at + 4 + n]
        at += 4 + n
        if got != case.expected:
            bad.append(case.name)
    assert at == len(out)
    assert not bad, f"{len(bad)} of {len(table)} cases differ: {bad}"


def test_exactness_bounds_by_name():
    """At the bound an int8 zero-point weight exports; one group past it is
    refused by name, as the runtime's dot-bound rule refuses it at load."""
    check_weight_bounds(8, BOUND, zero_point=True, offsets=False)
    with pytest.raises(SchemeRefused, match="exact int32 accumulator"):
        check_weight_bounds(8, BOUND + 32, zero_point=True, offsets=False)
    # Symmetric int8 binds at 132,096; int4 with a zero point binds at the
    # runtime's kMaxDotLen first; float offsets are exact to 132,104.
    check_weight_bounds(8, 131_040, zero_point=False, offsets=False)
    check_weight_bounds(4, 132_096, zero_point=True, offsets=True)
    with pytest.raises(SchemeRefused, match="2\\^24"):
        check_weight_bounds(4, 132_128, zero_point=True, offsets=True)
    assert qm.dot_length_bound(127, 255) == BOUND
