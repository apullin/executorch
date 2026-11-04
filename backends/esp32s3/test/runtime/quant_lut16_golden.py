# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""quant_lut16_golden.h: the expected bits of espressif::lut16's conformance
check (test/runtime/quant_lut16_check.cpp), from quantizer/mirrors.py.

    python quant_lut16_golden.py      # rewrites the header

numpy only (no torch), so it runs anywhere. The check builds its tables at
run time from a seeded xorshift32 stream, step for step as make_table() does
here (the header carries each table's FNV-1a, so a generator that drifts is
told apart from an arithmetic mismatch), then compares by FNV-1a:

  codes   every table on all 65,536 int16 codes, code to code: the device's
          loops (eval_codes) and the C reference (eval_code) separately
  stage A the op itself (quantize, table, dequantize) for six cases over
          rounding ties k + 1/2 and their +-1, +-2 ulp neighbours for k near
          0, +-2^15 and the +-2^16 rails, the special values, and a strided
          sweep x = fl(c * s)

The tables cover every scheme and the edges of what lut16 accepts: linear
nodes near the int32 bound, alternating +-2^30 nodes at F = 0 (the node
difference wraps but t is 0), F + G = 31, a one-segment F = 16 table, F = 0
cubic segments, NSAU with m = 0 and with both, one or a narrow sign range,
cubic segments whose output is an exact tie k + 1/2 (constant, or at every
odd t), and direct tables whose entries the output range clamps. The
flagged schemes: NSAU with OFFSET (11, 12) over a window away
from 0, one wholly negative, two lopsided ones and the full range; uniform
with TAIL (17, 18), whose codes past [lo, hi] read the edge segments -- a
cubic TAIL table's edge rows are lines here (c1 = 2^(F-1), half a code per
code), as the fitter makes them, where the codes past it reach a segment or more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from executorch.backends.esp32s3.quantizer import mirrors as qm

ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "quant_lut16_golden.h"

F32 = np.float32
# Exponent ranges (unbiased) of random cubic coefficients c0..c3: |c| < 2^16.
COEF_EXP = ((-3, 15), (-6, 12), (-10, 10), (-14, 8))

# name, scheme, lo, hi, shift, G, out_min, out_max, seed, amp (linear: node
# amplitude, -1 = alternating +-2^30)
TABLES = (
    ("direct A", 0, -300, 700, 0, 0, -30000, 30000, 0x1234567, 0),
    ("direct B", 0, -32768, -32768, 0, 0, -32768, 32767, 0x2345678, 0),
    ("uniform linear A", 1, -32767, 32767, 7, 8, -32767, 32767, 0x3456789, 5600000),
    ("uniform linear B", 1, -1000, 3000, 4, 12, -5000, 7000, 0x456789A, 46600000),
    ("uniform linear C", 1, 100, 200, 0, 3, -32768, 32767, 0x56789AB, -1),
    ("uniform linear D", 1, -32768, 32767, 15, 16, -32768, 32767, 0x6789ABC, 10900),
    ("uniform cubic A", 2, -32767, 32767, 9, 0, -32767, 32767, 0x789ABCD, 0),
    ("uniform cubic B", 2, 5, 900, 3, 0, -20000, 20000, 0x89ABCDE, 0),
    ("uniform cubic C", 2, -32768, 32767, 16, 0, -32768, 32767, 0x9ABCDEF, 0),
    ("uniform cubic D", 2, -50, 50, 0, 0, -32768, 32767, 0xABCDEF1, 0),
    ("nsau linear A", 3, -32767, 32767, 4, 6, -32767, 32767, 0xBCDEF12, 1000000),
    ("nsau linear B", 3, 0, 32767, 3, 8, -32768, 32767, 0xCDEF123, 520000),
    ("nsau linear C", 3, -20, 1000, 2, 0, -32768, 32767, 0xDEF1234, 40000),
    ("nsau linear D", 3, -32767, 32767, 0, 2, -32768, 32767, 0xEF12345, 65000),
    ("nsau cubic A", 4, -32767, 32767, 3, 0, -32767, 32767, 0xF123456, 0),
    ("nsau cubic B", 4, 0, 32767, 2, 0, -32768, 32767, 0x1357924, 0),
    ("nsau cubic C", 4, -5, 70, 5, 0, -1000, 1000, 0x2468ACE, 0),
    ("nsau cubic D", 4, -32767, 32767, 0, 0, -32768, 32767, 0x369CF25, 0),
    ("nsau cubic E", 4, 1, 3000, 4, 0, -32768, 32767, 0x48D159E, 0),
    ("nsau linear offset A", 11, 534, 32767, 4, 6, -32767, 32767, 0x5A1B2C3, 900000),
    ("nsau linear offset B", 11, -5000, -37, 3, 4, -32768, 32767, 0x6B2C3D4, 400000),
    ("nsau linear offset C", 11, -32767, 267, 2, 8, -32768, 32767, 0x7C3D4E5, 250000),
    ("nsau cubic offset A", 12, 534, 32767, 5, 0, -32767, 32767, 0x8D4E5F6, 0),
    ("nsau cubic offset B", 12, -5000, -37, 3, 0, -32768, 32767, 0x9E5F607, 0),
    ("nsau cubic offset C", 12, -20, 1000, 2, 0, -1000, 1000, 0xAF60718, 0),
    ("nsau cubic offset D", 12, -32767, 32767, 0, 0, -32768, 32767, 0xB071829, 0),
    ("uniform linear tail A", 17, -3000, 3000, 7, 4, -32768, 32767, 0xC18293A, 6000),
    ("uniform linear tail B", 17, 100, 200, 0, 0, -32768, 32767, 0xD293A4B, 16000),
    ("uniform cubic tail A", 18, -16382, 16385, 12, 0, -32767, 32767, 0xE3A4B5C, 0),
    ("uniform cubic tail B", 18, -32768, 32767, 9, 0, -32768, 32767, 0xF4B5C6D, 0),
    ("uniform cubic tail C", 18, -100, 155, 0, 0, -32768, 32767, 0x05C6D7E, 0),
)

# The op's stage A cases: table, scale s, zero point, [qmin, qmax], out
# scale, out zero point (inv = fl32(1/s), quantize_per_tensor's multiplier).
CASES = (
    ("uniform cubic A", 2.0 ** -12, 0, -32767, 32767, 2.0 ** -12, 0),
    ("nsau cubic A", 0.0123, -1000, -32768, 32767, 0.00731, 123),
    ("uniform linear A", 3.7, 5, -100, 30000, 0.5, -7),
    ("direct A", 1.0, 0, -32768, 32767, 1.0, 0),
    ("nsau linear A", 2.0 ** -12, 0, -32767, 32767, 0.01, 0),
    ("uniform cubic B", 0.05, 17, -32768, 32767, 0.3, -20),
    ("nsau cubic offset A", 2.0 ** -12, 0, -32767, 32767, 2.0 ** -12, 0),
    ("uniform cubic tail A", 2.0 ** -12, 0, -32767, 32767, 2.0 ** -12, 0),
    ("uniform linear tail A", 0.37, 3, -30000, 30000, 0.5, 7),
    ("nsau linear offset B", 0.0123, -1000, -32768, 32767, 0.00731, 123),
)
WINDOWS = ((-40, 40), (32700, 32800), (-32800, -32700), (65480, 65600),
           (-65600, -65480))
SWEEP = (-70000, 70000, 61)
# +0, -0, +-denorm_min, +-FLT_MAX, +-inf, NaN (quant_lut_check's).
SPECIALS = (0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7F7FFFFF,
            0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000)


class Rng:
    """xorshift32, as quant_lut16_check.cpp's."""

    def __init__(self, seed):
        self.x = seed & 0xFFFFFFFF

    def next(self):
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.x = x
        return x


def count(scheme, lo, hi, shift):
    return qm.lut16_entries(scheme, lo, hi, shift)


def bits_f32(b):
    return np.array([b], dtype=np.uint32).view(F32)[0]


def segment_shift(scheme, lo, hi, shift, s):
    """The log2 width of cubic row s (NSAU: of its segment k = s - the bias
    of its half; the negative half's rows come after the positive's)."""
    if scheme & 7 == 2:
        return shift
    pos, neg, _ = qm.lut16_nsau_layout(scheme, lo, hi, shift)
    n0 = qm.lut16_nsau_k(max(1, -hi), shift) if scheme & 8 else 0
    k = s - neg if (lo < 0 and s >= neg + n0) else s - pos
    return max((k >> shift) - 1, 0)


def make_table(spec):
    """The table the check generates for a TABLES row."""
    _, scheme, lo, hi, shift, G, _, _, seed, amp = spec
    rng = Rng(seed)
    n = count(scheme, lo, hi, shift)
    if scheme == 0:
        raw = [rng.next() >> 16 for _ in range(n)]
        return np.array(raw, dtype=np.uint16).view(np.int16)
    if scheme & 7 in (1, 3):
        if amp == -1:
            V = [(1 << 30) if i % 2 == 0 else -(1 << 30) for i in range(n)]
        else:
            V = [int(rng.next() % (2 * amp + 1)) - amp for _ in range(n)]
        return np.array(V, dtype=np.int32)
    C = np.zeros((n, 4), dtype=F32)
    for s in range(n):
        kind = s % 5
        if kind == 0:
            C[s, 0] = F32(int(rng.next() % 60001) - 30000) + F32(0.5)
        elif kind == 1:
            C[s, 0] = F32(int(rng.next() % 2001) - 1000)
            F = segment_shift(scheme, lo, hi, shift, s)
            C[s, 1] = F32(2.0 ** (F - 1)) if F >= 1 else F32(0.0)
        else:
            for j, (emin, emax) in enumerate(COEF_EXP):
                r = rng.next()
                sign = r & 1
                e = emin + int((r >> 1) % (emax - emin + 1))
                mant = rng.next() & 0x7FFFFF
                C[s, j] = bits_f32((sign << 31) | ((e + 127) << 23) | mant)
    if scheme & 16:
        # TAIL: an edge row with codes past it is a line, half a code per code
        below, above = qm.lut16_tail_reach(lo, hi, shift)
        for s, reach in ((0, below), (n - 1, above)):
            if reach >= (1 << shift):
                C[s, 1:] = (F32(2.0 ** (shift - 1)), F32(0.0), F32(0.0))
    return C


def fnv1a(data, h=2166136261):
    for byte in np.ascontiguousarray(data).tobytes():
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def stage_a_inputs(s, inv):
    """The check's inputs for one case, in its order."""
    inv64 = np.float64(F32(inv))
    xs = []
    for k0, k1 in WINDOWS:
        for k in range(k0, k1 + 1):
            x0 = F32(np.float64(k + 0.5) / inv64)
            d1 = np.nextafter(x0, F32(-np.inf))
            u1 = np.nextafter(x0, F32(np.inf))
            xs += [np.nextafter(d1, F32(-np.inf)), d1, x0, u1,
                   np.nextafter(u1, F32(np.inf))]
    specials = np.array(SPECIALS, dtype=np.uint32).view(F32)
    sweep = (np.arange(*SWEEP, dtype=np.int64).astype(F32) * F32(s)).astype(F32)
    return np.concatenate([np.array(xs, dtype=F32), specials, sweep])


def golden_header():
    names = [t[0] for t in TABLES]
    tables = {t[0]: make_table(t) for t in TABLES}
    codes = np.arange(-32768, 32768, dtype=np.int64)
    rows = []
    for spec in TABLES:
        name, scheme, lo, hi, shift, G, out_min, out_max, seed, amp = spec
        data = tables[name]
        assert qm.lut16_scheme_error(scheme) is None, name
        if scheme & 7 in (1, 3):
            assert qm.lut16_linear_safe(scheme, data, lo, hi, shift, G), name
        if scheme & 7 in (2, 4):
            assert qm.lut16_cubic_safe(scheme, data, lo, hi, shift), name
        y = qm.lut16_codes(codes, scheme, data, lo, hi, shift, G, out_min, out_max)
        rows.append((spec, fnv1a(data), fnv1a(y.astype(np.int16)), len(data)))
    cases = []
    for table, s, zp, qmin, qmax, out_s, out_zp in CASES:
        spec = TABLES[names.index(table)]
        _, scheme, lo, hi, shift, G, out_min, out_max, _, _ = spec
        s32 = F32(s)
        inv = qm.reciprocal(s32)
        out32 = F32(out_s)
        x = stage_a_inputs(s32, inv)
        y = qm.lut16(x, tables[table], inv, zp, qmin, qmax, scheme, lo, hi, shift, G,
                     out32, out_zp, out_min, out_max)
        cases.append((names.index(table), int(np.array([s32]).view(np.uint32)[0]),
                      int(np.array([inv]).view(np.uint32)[0]), zp, qmin, qmax,
                      int(np.array([out32]).view(np.uint32)[0]), out_zp, len(x),
                      fnv1a(x), fnv1a(y.astype(F32))))
    lines = [
        "// Generated by quant_lut16_golden.py -- do not edit; test_quant_lut16.py",
        "// regenerates it and compares.",
        "//",
        "// quant_mirrors' bits for espressif::lut16's conformance check",
        "// (quant_lut16_check.cpp): per table, how the check generates it (a",
        "// seeded xorshift32 stream, quant_lut16_golden.make_table) and the FNV-1a",
        "// of its bytes and of its output on all 65,536 int16 codes; per stage A",
        "// case, the op's arguments and the FNV-1a of its inputs and outputs.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace espressif {",
        "namespace lut16 {",
        "namespace golden {",
        "",
        "struct TableSpec {",
        "  const char* name;",
        "  int32_t scheme, lo, hi, shift, G, out_min, out_max;",
        "  uint32_t seed;",
        "  int32_t amp;  // linear: node amplitude; -1: alternating +-2^30",
        "  int32_t count;",
        "  uint32_t table_fnv, codes_fnv;",
        "};",
        "",
        "constexpr TableSpec kTables[] = {",
    ]
    for (name, scheme, lo, hi, shift, G, out_min, out_max, seed, amp), tf, cf, n in rows:
        lines.append(f'    {{"{name}", {scheme}, {lo}, {hi}, {shift}, {G}, {out_min}, '
                     f'{out_max}, 0x{seed:08X}u, {amp}, {n}, 0x{tf:08X}u, 0x{cf:08X}u}},')
    lines += [
        "};",
        "",
        "// Random cubic coefficients c0..c3: exponent ranges (unbiased).",
        "constexpr int32_t kCoefExp[4][2] = {"
        + ", ".join(f"{{{a}, {b}}}" for a, b in COEF_EXP) + "};",
        "",
        "struct Case {",
        "  int32_t table;",
        "  uint32_t scale_bits, inv_bits;",
        "  int32_t zero_point, quant_min, quant_max;",
        "  uint32_t out_scale_bits;",
        "  int32_t out_zero_point;",
        "  uint32_t inputs, input_fnv, output_fnv;",
        "};",
        "",
        "constexpr Case kCases[] = {",
    ]
    for t, sb, ib, zp, qmin, qmax, ob, ozp, n, xf, yf in cases:
        lines.append(f"    {{{t}, 0x{sb:08X}u, 0x{ib:08X}u, {zp}, {qmin}, {qmax}, "
                     f"0x{ob:08X}u, {ozp}, {n}u, 0x{xf:08X}u, 0x{yf:08X}u}},")
    lines += [
        "};",
        "",
        "// Rounding ties k + 1/2 for k in each window (and +-1, +-2 ulps).",
        "constexpr int32_t kWindows[][2] = {"
        + ", ".join(f"{{{a}, {b}}}" for a, b in WINDOWS) + "};",
        f"constexpr int32_t kSweepFirst = {SWEEP[0]};",
        f"constexpr int32_t kSweepEnd = {SWEEP[1]};",
        f"constexpr int32_t kSweepStep = {SWEEP[2]};",
        "constexpr uint32_t kSpecials[] = {"
        + ", ".join(f"0x{v:08X}u" for v in SPECIALS) + "};",
        "",
        "}  // namespace golden",
        "}  // namespace lut16",
        "}  // namespace espressif",
        "",
    ]
    return "\n".join(lines)


def main():
    HEADER.write_text(golden_header())
    print(f"wrote {HEADER}")


if __name__ == "__main__":
    main()
