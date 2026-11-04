# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch's stock quantised K/V cache and embeddings: the runtime's
arithmetic at its limits, bit for bit against mirrors.py and torch's
own ops.

esp32_host_runner --asym-conformance runs the device's own functions on cases
chosen to be adversarial (kinds 8-10 of host_runner/asym_conformance.cpp, and
kind 2's gather for the tables):

  - the A4 writer (kv_affine::quantize_row): exact rounding ties under odd
    and under even zero points (power-of-two scales make x * (1 / s) exact,
    so x / s + z lands on n + 1/2), both zero-point branches of XNNPACK's
    chooser, constant positive and negative rows, an all-zero row, rows
    floored at FLT_EPSILON, one-element rows, random rows -- codes, scales
    and zero points equal to mirrors.kcache_write AND to torch's own
    choose_qparams_per_token_asymmetric + quantize_per_token;
  - OP_BMM_KCACHE_AFFINE_I8's arithmetic (esp32_asym::kcache_query_row and
    kcache_score): random rows, a ragged head (k < k_pad), and k = 66,304,
    the opcode's exactness bound, with the worst operands (|acc| =
    2,147,255,040), on the PIE and the scalar dot -- equal to
    mirrors.kcache_scores; one group past the bound is refused by the
    exporter and the gate by name, and at load (asym_blob_cases'
    "kcache-dot-bound", test_asym_refusals.py);
  - non-finite rows: a NaN is skipped by the chooser and takes the zero
    point, an infinity makes the scale infinite (+inf's NaN zero point is
    0), and every non-finite element's code is ExecuTorch's own
    quantize_per_token kernel's at the row's scale and zero point (its 1.6
    rule, runtime/quant_rules.h);
  - the cache read (kv_affine::dequantize_row) over the whole code and
    zero-point range: mirrors.vcache_dequant and torch's
    dequantize_per_token;
  - the 4-bit tables: ExecuTorch's high-nibble-first bytes, unpacked as
    embedding_4bit unpacks them and repacked by the backend's pack_rows, are
    exactly the nibble swap (mirrors.swap_nibbles), and the device's
    gather of them equals torch's embedding_4bit; 2-bit codes stored as int4
    gather to torch's embedding_2bit.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import struct
import subprocess

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.asym.test_asym_conformance import (
    I4_TABLE,
    PIE,
    ZP_NONE,
    Case,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
BOUND = 66_304            # A0 queries x zero-pointed key codes (kcache-dot-bound)


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def run_cases(runner, tmp_path, table):
    data = b"ASCF" + struct.pack("<II", 1, len(table)) + b"".join(
        c.record for c in table)
    (tmp_path / "cases.bin").write_bytes(data)
    done = subprocess.run(
        [str(runner), "--asym-conformance", str(tmp_path / "cases.bin"),
         str(tmp_path / "out.bin")], capture_output=True, text=True,
        timeout=600)
    assert done.returncode == 0, done.stderr[-1500:]
    out = (tmp_path / "out.bin").read_bytes()
    at, got = 0, []
    for case in table:
        parts = []
        for _ in range(getattr(case, "arrays_out", 1)):
            n, = struct.unpack_from("<I", out, at)
            parts.append(out[at + 4:at + 4 + n])
            at += 4 + n
        got.append(b"".join(parts))
    assert at == len(out)
    return got


# ---- the writer ------------------------------------------------------------------

def tie_rows():
    """Rows whose every quantised value is an exact tie: with s a power of two
    (span = 255 s), x = (m + 1/2) s gives x / s + z = m + z + 1/2 exactly, and
    round-half-to-even goes up or down with the parity of m + z. Each (lo, hi)
    pair, in units of s, picks the zero-point branch and the parity:
      (-128, 127) from max, z = 0     (-127, 128) from max, z = -1
      (-100, 155) from min, z = -28   (-101, 154) from min, z = -27"""
    rows = []
    for k in (3, 7, 10):
        s = F32(2.0 ** -k)
        for lo, hi in ((-128, 127), (-127, 128), (-100, 155), (-101, 154)):
            m = np.arange(lo, hi - 1, 9, dtype=np.int64)[:30]
            ties = ((m.astype(F32) + F32(0.5)) * s).astype(F32)
            row = np.concatenate([[F32(lo) * s, F32(hi) * s], ties]).astype(F32)
            rows.append(np.pad(row, (0, 32 - row.size), mode="edge"))
    return rows


def writer_rows():
    rng = np.random.default_rng(80)
    rows = tie_rows()
    rows += [np.full(32, F32(0.75)), np.full(32, F32(-1.5)),
             np.zeros(32, F32),
             (rng.standard_normal(32) * F32(1e-7)).astype(F32),   # eps floor
             np.full(32, F32(3e-8)),                              # eps, one-sided
             (rng.standard_normal(32) * F32(3.0)).astype(F32),
             (rng.standard_normal(32) * F32(200.0)).astype(F32),
             (rng.random(32) * F32(5.0)).astype(F32),             # all positive
             -(rng.random(32) * F32(5.0)).astype(F32)]            # all negative
    return np.stack(rows).astype(F32)


def torch_writer(rows):
    """torch's own ops, as QuantizedKVCache._quantize and its casts run them."""
    qd = torch.ops.quantized_decomposed
    x = torch.from_numpy(rows)
    scales, zps = qd.choose_qparams_per_token_asymmetric.default(x, torch.int8)
    codes = qd.quantize_per_token.default(x, scales, zps, -128, 127, torch.int8)
    return (codes.numpy(), scales.to(torch.float32).numpy().reshape(-1),
            zps.to(torch.int8).numpy().reshape(-1))


def writer_case(name, rows):
    codes, scales, zps = qm.kcache_write(rows)
    case = Case(name, 8, 0, 0, 0, 0, rows.shape[0], rows.shape[1], ZP_NONE, 0,
                0, [b"", rows.astype(F32).tobytes(), b"", b"", b"", b"", b"",
                    b""],
                codes.tobytes() + scales.astype(F32).tobytes() + zps.tobytes())
    case.arrays_out = 3
    return case


def test_the_mirror_is_torch():
    """mirrors.kcache_write, the reference the device is held to, is
    torch's own writer on every adversarial row (the ties included)."""
    rows = writer_rows()
    mine = qm.kcache_write(rows)
    theirs = torch_writer(rows)
    for a, b in zip(mine, theirs):
        assert np.array_equal(np.asarray(a).view(np.uint8),
                              np.asarray(b).view(np.uint8))
    # The tie rows really are ties, under both parities and both branches.
    zps = set(int(z) for z in mine[2][:12])
    assert zps == {0, -1, -28, -27}


def test_writer_is_the_mirror(runner, tmp_path):
    rows = writer_rows()
    single = np.asarray([[F32(-0.3)], [F32(0.0)], [F32(7.0)]], F32)
    table = [writer_case("adversarial rows", rows),
             writer_case("one-element rows", single)]
    got = run_cases(runner, tmp_path, table)
    for case, data in zip(table, got):
        assert data == case.expected, case.name


# ---- non-finite values -------------------------------------------------------------

def nonfinite_rows():
    """NaN of either sign, the infinities and +-FLT_MAX in K/V rows: a NaN
    beside finite values, +inf with a NaN (scale inf; its zero point's
    arithmetic is NaN, taken as 0), -inf as the only infinity (zero point
    127), both infinities, a row of NaNs (an all-zero row's: zero point 127),
    +-FLT_MAX (the span overflows, so the scale is inf with every value
    finite), and +-1e30, a large finite span, with a NaN."""
    nan, inf, big = F32(np.nan), F32(np.inf), F32(np.finfo(F32).max)
    return np.asarray([[0.5, nan, -1.75, 3.25, -nan, 2.0, -0.0, 1.0],
                       [0.5, inf, -1.75, 3.0, 0.0, 1.0, nan, 2.0],
                       [0.5, -inf, 1.75, 3.0, 0.0, 1.0, 2.0, -2.0],
                       [inf, -inf, nan, 1.0, -1.0, 0.0, 2.0, 3.0],
                       [nan, -nan, nan, nan, nan, nan, nan, nan],
                       [big, -big, 1.0, 0.0, 2.0, 3.0, -1.0, 0.5],
                       [1e30, -1e30, 1e20, -3e9, 0.0, 1.0, 2.0, nan]], F32)


def test_writer_non_finite_rows_are_the_mirror(runner, tmp_path):
    table = [writer_case("non-finite rows", nonfinite_rows())]
    got = run_cases(runner, tmp_path, table)
    assert got[0] == table[0].expected


def test_non_finite_writer_follows_the_portable_kernel():
    """The rule mirrors.kcache_write -- and so the device -- follows on
    non-finite rows, and ExecuTorch's own quantize_per_token kernel on every
    element whose value is not finite (on every element of a row whose scale
    is infinite: inv 0 makes every product 0 or NaN)."""
    import executorch.kernels.quantized  # noqa: F401 -- registers the kernels

    rows = nonfinite_rows()
    codes, scales, zps = qm.kcache_write(rows)
    finite = rows[0][np.isfinite(rows[0])][None]
    _, s0, z0 = qm.kcache_write(finite)
    assert s0[0] == scales[0] and z0[0] == zps[0]      # the NaN is skipped
    assert (codes[0][np.isnan(rows[0])] == zps[0]).all()
    assert np.isinf(scales[[1, 2, 3, 5]]).all()
    assert zps[[1, 2, 3, 4, 5]].tolist() == [0, 127, 0, 127, 127]
    assert np.isfinite(scales[[0, 4, 6]]).all()
    for r in (1, 2, 3, 4, 5):
        assert (codes[r] == zps[r]).all(), r
    out = torch.empty(rows.shape, dtype=torch.int8)
    portable = torch.ops.quantized_decomposed.quantize_per_token.out(
        torch.from_numpy(rows), torch.from_numpy(scales.astype(np.float64))[:, None],
        torch.from_numpy(zps.astype(np.int64))[:, None], -128, 127, torch.int8,
        out=out).numpy()
    compared = ~np.isfinite(rows) | np.isinf(scales)[:, None]
    assert np.array_equal(codes[compared], portable[compared])


# ---- the score kernel --------------------------------------------------------------

def score_case(name, q, codes, scales, zps, k_pad, pie):
    want = qm.kcache_scores(q, codes, scales, zps)
    rows, k = q.shape
    n = codes.shape[0]
    return Case(name, 9, PIE if pie else 0, k_pad, 0, n, rows, k, ZP_NONE, 0,
                0, [b"", q.astype(F32).tobytes(), codes.astype(np.int8).tobytes(),
                    scales.astype(F32).tobytes(), zps.astype(np.int8).tobytes(),
                    b"", b"", b""],
                want.astype(F32).tobytes())


def score_cases():
    rng = np.random.default_rng(81)
    out = []
    for k, k_pad in ((16, 16), (24, 32), (64, 64)):
        q = (rng.standard_normal((3, k)) * 2).astype(F32)
        keys = (rng.standard_normal((9, k)) * 3).astype(F32)
        codes, scales, zps = qm.kcache_write(keys)
        q[2] = 0.0                                  # an all-zero query row
        for pie in (True, False):
            out.append(score_case(f"random k={k} pie={pie}", q, codes, scales,
                                  zps, k_pad, pie))
    # At the bound: every query code 127 (a row of equal values), key codes at
    # the far end of a zero point at the other: |acc| = 127 * 255 * 66,304.
    q = np.ones((1, BOUND), F32)
    codes = np.vstack([np.full(BOUND, -128), np.full(BOUND, 127)]).astype(np.int8)
    zps = np.asarray([127, -128], np.int8)
    scales = np.asarray([F32(1e-3), F32(2e-3)], F32)
    for pie in (True, False):
        out.append(score_case(f"k={BOUND}, |acc| = 2,147,255,040, pie={pie}",
                              q, codes, scales, zps, BOUND, pie))
    return out


def test_scores_are_the_mirror(runner, tmp_path):
    table = score_cases()
    got = run_cases(runner, tmp_path, table)
    bad = [c.name for c, data in zip(table, got) if data != c.expected]
    assert not bad, bad
    # The bound case's integer side is what the mirror says it is.
    assert 127 * 255 * BOUND == 2_147_255_040 < 2 ** 31


def test_one_group_past_the_bound_is_refused():
    """The exporter's bound and the runtime's agree: 66,304, and 66,336 is
    refused by the gate by name (and at load, asym_blob_cases)."""
    from executorch.backends.esp32s3.serialization import espressif_blob
    from executorch.backends.esp32s3.serialization import weight_formats
    from executorch.backends.esp32s3.operator_support import espressif_operator_support
    from executorch.backends.esp32s3._passes import stock_llm

    assert qm.dot_length_bound(127, 255) == BOUND
    assert espressif_blob.KCACHE_AFFINE_DOT_BOUND == BOUND == stock_llm.KCACHE_DOT_BOUND
    assert espressif_operator_support._pad_to(BOUND + 1, espressif_blob.VEC_BYTES) > espressif_blob.KCACHE_AFFINE_DOT_BOUND


# ---- the read ------------------------------------------------------------------------

def test_read_is_the_mirror_and_torch(runner, tmp_path):
    rng = np.random.default_rng(82)
    rows, k = 6, 40
    codes = rng.integers(-128, 128, size=(rows, k)).astype(np.int8)
    codes[0, :2] = (-128, 127)
    zps = np.asarray([-128, 127, 0, 5, -7, 1], np.int8)
    scales = np.asarray([2.0 ** -23, 0.5, 3e-3, 1.25, 7e-5, 0.0], F32)
    want = qm.vcache_dequant(codes, scales, zps)
    qd = torch.ops.quantized_decomposed
    theirs = qd.dequantize_per_token.default(
        torch.from_numpy(codes), torch.from_numpy(scales).reshape(rows, 1).double(),
        torch.from_numpy(zps.astype(np.int64)).reshape(rows, 1), -128, 127,
        torch.int8, torch.float32).numpy()
    assert np.array_equal(want.view(np.uint32), theirs.view(np.uint32))
    case = Case("read", 10, 0, 0, 0, 0, rows, k, ZP_NONE, 0, 0,
                [b"", b"", codes.tobytes(), scales.tobytes(), zps.tobytes(),
                 b"", b"", b""], want.astype(F32).tobytes())
    assert run_cases(runner, tmp_path, [case])[0] == case.expected


# ---- the tables ----------------------------------------------------------------------

def table_case(name, bits, packed, scales, indices, torch_out):
    """The device's gather (kind 2) of an ExecuTorch table the way the
    exporter stores it: unpacked as the op unpacks it, packed by pack_rows."""
    from executorch.backends.esp32s3.serialization import espressif_blob
    from executorch.backends.esp32s3.serialization import weight_formats
    from executorch.backends.esp32s3.operator_support import espressif_operator_support
    from executorch.backends.esp32s3._passes import stock_llm

    codes = stock_llm.unpack_codes(torch.from_numpy(packed), bits).numpy()
    vocab, d = codes.shape
    fmt = weight_formats.FORMATS["i4"]
    stored = np.frombuffer(weight_formats.pack_rows(codes.astype(np.int8), d, fmt),
                           np.uint8)
    gsize = d // scales.shape[1]
    return Case(name, 2, I4_TABLE, d, gsize, vocab, len(indices), d, ZP_NONE,
                0, 0, [b"", b"", stored.tobytes(), scales.astype(F32).tobytes(),
                       b"", b"", b"", np.asarray(indices, np.uint32).tobytes()],
                torch_out.astype(F32).tobytes()), stored


def test_nibble_swap_and_two_bit_tables(runner, tmp_path):
    rng = np.random.default_rng(83)
    qd = torch.ops.quantized_decomposed
    vocab, d, group = 12, 64, 32
    idx = np.asarray([3, 0, 11, 7, 3])
    scales = (rng.random((vocab, d // group)) * 0.05 + 1e-3).astype(F32)
    p4 = rng.integers(0, 256, size=(vocab, d // 2)).astype(np.uint8)
    p2 = rng.integers(0, 256, size=(vocab, d // 4)).astype(np.uint8)
    t4 = qd.embedding_4bit.dtype(torch.from_numpy(p4), torch.from_numpy(scales),
                                 None, -8, 7, torch.from_numpy(idx),
                                 dtype=torch.float32).numpy()
    t2 = qd.embedding_2bit.dtype(torch.from_numpy(p2), torch.from_numpy(scales),
                                 None, -2, 1, torch.from_numpy(idx),
                                 dtype=torch.float32).numpy()
    c4, stored4 = table_case("4-bit table", 4, p4, scales, idx, t4)
    c2, _ = table_case("2-bit table as int4", 2, p2, scales, idx, t2)
    # Repacking ExecuTorch's 4-bit bytes is exactly the nibble swap.
    assert np.array_equal(stored4.reshape(vocab, -1), qm.swap_nibbles(p4))
    # The mirror's view of both tables agrees with the ops themselves.
    assert np.array_equal(qm.unpack_et_4bit(p4),
                          qm.unpack_backend_i4(qm.swap_nibbles(p4)))
    assert np.array_equal(
        qm.embedding_dequant(qm.unpack_et_2bit(p2), scales, None, idx)
        .view(np.uint32), t2.view(np.uint32))
    got = run_cases(runner, tmp_path, [c4, c2])
    assert got[0] == c4.expected and got[1] == c2.expected
