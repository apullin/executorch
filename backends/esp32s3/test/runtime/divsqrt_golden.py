# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The device's division and square root, held to exact rounding.

The exact ops between delegates assume the LX7 rounds float division and sqrtf
correctly to nearest even: div kept as a true division, sqrt, rsqrt and
reciprocal held exact, and every table built from torch's own float32
arithmetic. On the ESP32-S3 neither is a single instruction. Float division is
libgcc's __divsf3 in ROM -- DIV0.S, a Newton refinement in MADDN.S, then
DIVN.S -- and sqrtf is libgcc's __ieee754_sqrtf, SQRT0.S through DIVN.S.
ExecuTorch's portable reciprocal and rsqrt compute 1.0 / x in double (ROM
__divdf3, soft-float) and narrow to float. This module makes the inputs on
which a wrong rounding would show, and their correctly rounded results,
computed exactly with rationals:

  hardest  quotients, reciprocals and square roots closest to a midpoint
           between two floats (where round-to-nearest decides), and closest to
           a float (where only the sticky bit does), from below and above:
           built with modular arithmetic for quotients, found exhaustively
           over every significand for reciprocals and square roots
  ties     quotients exactly halfway between two subnormals: division's only
           exact ties (a normal quotient of two floats is never a midpoint, and
           a square root never is)
  edges    zeros of both signs, infinities, NaN, denormal operands and
           results, overflow to infinity and the rounding into FLT_MAX,
           underflow to zero and to denorm_min
  random   finite operands from xorshift32 (divsqrt_check.cpp draws the same
           ones), held by FNV-1a of the results

golden_header() is test/runtime/divsqrt_golden.h, which
divsqrt_check.cpp checks every case against with the device's own operations,
printing one "divsqrt check:" line that run_parity gates on. Under QEMU that
proves QEMU's model of the FPU instructions, not the silicon: it is a board
row too (run_hardware runs the same parity firmware on the chip).
A NaN result is checked as a NaN of any payload. Imports numpy, not torch.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import numpy as np

ANY_NAN32 = 0x7FC00000          # an expected NaN: any NaN passes
ANY_NAN64 = 0x7FF8000000000000
F32_MAX_EXP, F32_MIN_EXP, F32_P = 127, -126, 24
F64_MAX_EXP, F64_MIN_EXP, F64_P = 1023, -1022, 53

# divsqrt_check.cpp's random streams: xorshift32 seeds and lengths.
RANDOM_DIV = (0x2545F491, 4096)
RANDOM_SQRT = (0x9E3779B9, 4096)


# --- exact float arithmetic ---------------------------------------------------------

def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits64(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _fmt(bits, wide):
    return (bits >> (63 if wide else 31)) & 1, \
        (bits >> (52 if wide else 23)) & (0x7FF if wide else 0xFF), \
        bits & ((1 << (52 if wide else 23)) - 1)


def exact(bits, wide=False):
    """The finite float's value as a Fraction (float32 bits, or float64)."""
    sign, exp, frac = _fmt(bits, wide)
    p, bias = (F64_P, 1023) if wide else (F32_P, 127)
    if exp == (0x7FF if wide else 0xFF):
        raise ValueError("not finite")
    if exp == 0:
        value = Fraction(frac) * Fraction(2) ** (1 - bias - (p - 1))
    else:
        value = Fraction(frac + (1 << (p - 1))) * Fraction(2) ** (exp - bias - (p - 1))
    return -value if sign else value


def is_nan(bits, wide=False):
    _, exp, frac = _fmt(bits, wide)
    return exp == (0x7FF if wide else 0xFF) and frac != 0


def is_inf(bits, wide=False):
    _, exp, frac = _fmt(bits, wide)
    return exp == (0x7FF if wide else 0xFF) and frac == 0


def round_float(q, wide=False, negative_zero=False):
    """The bits of q rounded to nearest even in float32 (or float64), with
    overflow to infinity and gradual underflow. q is an exact Fraction; a zero
    result takes negative_zero's sign when q is 0 itself."""
    p, emin, emax = (F64_P, F64_MIN_EXP, F64_MAX_EXP) if wide else \
        (F32_P, F32_MIN_EXP, F32_MAX_EXP)
    width = 64 if wide else 32
    sign = q < 0 or (q == 0 and negative_zero)
    q = abs(q)
    sign_bit = (1 << (width - 1)) if sign else 0
    if q == 0:
        return sign_bit
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    step = max(e, emin) - (p - 1)
    scaled = q / Fraction(2) ** step
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    value = Fraction(n) * Fraction(2) ** step
    if value >= Fraction(2) ** (emax + 1):
        return sign_bit | ((0x7FF << 52) if wide else (0xFF << 23))
    if n == 0:
        return sign_bit
    as_double = float(n) * 2.0 ** step      # exact: n < 2^p, step in range
    out = bits64(as_double) if wide else bits32(np.float32(as_double))
    assert (f64(out) if wide else f32(out)) == as_double
    return out | sign_bit


def div(a, b, wide=False):
    """IEEE a / b on float bits: the correctly rounded quotient, or the
    specials' results (an expected NaN is ANY_NAN)."""
    nan = ANY_NAN64 if wide else ANY_NAN32
    inf = (0x7FF << 52) if wide else (0xFF << 23)
    top = 63 if wide else 31
    sign = ((a >> top) ^ (b >> top)) & 1
    sbit = sign << top
    if is_nan(a, wide) or is_nan(b, wide):
        return nan
    if is_inf(a, wide):
        return nan if is_inf(b, wide) else inf | sbit
    if is_inf(b, wide):
        return sbit
    xa, xb = exact(a, wide), exact(b, wide)
    if xb == 0:
        return nan if xa == 0 else inf | sbit
    if xa == 0:
        return sbit
    return round_float(xa / xb, wide)


def _isqrt_round(q):
    """sqrt(q) rounded to float32, q a positive Fraction: integer square root
    of q scaled by 4^t so the root has 40+ bits, then the remainder decides."""
    e = q.numerator.bit_length() - q.denominator.bit_length()
    t = max(0, (80 - e) // 2 + 1)
    scaled = q * Fraction(4) ** t
    s = scaled.numerator // scaled.denominator    # floor(q * 4^t)
    r = math.isqrt(s)                              # floor(sqrt(floor(q 4^t)))
    # sqrt(q) in (r, r + 1) / 2^t unless exact; r has 40+ bits, so rounding
    # r + 1/2 (inside the open interval, never a float32 midpoint at this
    # width unless the root is exact) to 24 bits gives the right float.
    exact_root = r * r == s and scaled.denominator == 1
    value = Fraction(r) if exact_root else Fraction(2 * r + 1, 2)
    return round_float(value / Fraction(2) ** t)


def sqrt(x):
    """IEEE sqrtf on float32 bits."""
    if is_nan(x):
        return ANY_NAN32
    if x == 0x80000000 or x == 0:
        return x
    if x >> 31:
        return ANY_NAN32
    if is_inf(x):
        return x
    return _isqrt_round(exact(x))


def recip_portable(x):
    """ExecuTorch's portable reciprocal on float32: (float)(1.0 / (double)x)."""
    if is_nan(x):
        return ANY_NAN32
    d = div(bits64(1.0), bits64(f32(x)), wide=True)
    return narrow(d)


def rsqrt_portable(x):
    """ExecuTorch's portable rsqrt on float32: (float)(1.0 / std::sqrt(x)),
    sqrtf then a double division, narrowed."""
    s = sqrt(x)
    if is_nan(s):
        return ANY_NAN32
    return narrow(div(bits64(1.0), bits64(f32(s)), wide=True))


def narrow(d):
    """double -> float32, to nearest even."""
    if is_nan(d, True):
        return ANY_NAN32
    if is_inf(d, True):
        return (0xFF << 23) | ((d >> 63) << 31)
    return round_float(exact(d, True), negative_zero=bool(d >> 63))


# --- the hardest cases -----------------------------------------------------------

def _distance_classes(Q, R, B):
    """For quotients Q + R/B (Q carrying one bit past the float's 24): the
    distance to the nearest midpoint (Q odd: R/B above; Q even: (B - R)/B
    below the next) and to the nearest float (Q even: R/B above; Q odd:
    (B - R)/B below), in units of the bit after the last."""
    odd = (Q & 1) == 1
    mid = np.where(odd, R, B - R).astype(np.float64) / B
    near = np.where(odd, B - R, R).astype(np.float64) / B
    return mid, near


def hardest_quotients(count=48, seed=11):
    """[(A, B)] 24-bit significands whose quotient lies within ~2^-24 of a
    bit past the last of a midpoint or of a float, from both sides: for odd B
    and a residue r, A = r * (2^s)^-1 mod B makes A * 2^s = Q * B + r."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        B = int(rng.integers(1 << 23, 1 << 24)) | 1
        shift = 24
        inv = pow(1 << shift, -1, B)
        for r in (1, 2, 3, B - 1, B - 2, B - 3):
            A = (r * inv) % B
            while A < (1 << 23):
                A += B
            if A >= (1 << 24) or A < B:
                continue
            out.append((A, B))
            if len(out) >= count:
                break
    return out


def hardest_reciprocals(count=24):
    """[B] 24-bit significands whose reciprocal 2^48 / B is nearest a midpoint
    or a float (exhaustive over [2^23 + 1, 2^24))."""
    B = np.arange((1 << 23) + 1, 1 << 24, dtype=np.int64)
    num = np.int64(1) << np.int64(48)
    Q = num // B
    R = num - Q * B
    mid, near = _distance_classes(Q, R, B)
    # A stable sort: numpy's default sort breaks ties by its platform's algorithm
    # (x86-64 with AVX-512 and arm64 differ), and the picks below are full of ties.
    picks = (list(B[np.argsort(mid, kind="stable")[:count]])
             + list(B[np.argsort(near, kind="stable")[:count // 2]]))
    return [int(b) for b in picks]


def hardest_square_roots(count=32):
    """[(X, p)]: sqrt(X * 2^(p + 29)) for 24-bit X and parity p nearest a
    midpoint or a float of its 24-bit rounding (exhaustive over X)."""
    out = []
    for p in (0, 1):
        X = np.arange(1 << 23, 1 << 24, dtype=np.uint64)
        S = X << np.uint64(29 + p)
        r = np.floor(np.sqrt(S.astype(np.float64))).astype(np.uint64)
        r -= (r * r > S).astype(np.uint64)
        r += ((r + np.uint64(1)) * (r + np.uint64(1)) <= S).astype(np.uint64)
        # r has 26 or 27 bits; the float keeps 24, the next bit decides.
        bits = np.where(r >= (np.uint64(1) << np.uint64(26)), 3, 2).astype(np.uint64)
        frac = (S - r * r).astype(np.float64) / (2.0 * r.astype(np.float64) + 1.0)
        low = (r & ((np.uint64(1) << bits) - np.uint64(1))).astype(np.float64) + frac
        half = (np.uint64(1) << (bits - np.uint64(1))).astype(np.float64)
        unit = half * 2.0
        mid = np.abs(low - half) / unit
        near = np.minimum(low, unit - low) / unit
        exact_root = frac == 0
        near = np.where(exact_root & (low == 0), np.inf, near)
        for order in (np.argsort(mid, kind="stable")[:count],
                      np.argsort(near, kind="stable")[:count // 2]):
            out += [(int(X[i]), p) for i in order]
        del X, S, r, frac, low, mid, near
    return out


def hardest_rsqrt(count=16):
    """[x bits]: x in [1, 4) whose sqrtf result s has the hardest 1 / s."""
    xs = []
    for p in (0, 1):
        X = np.arange(1 << 23, 1 << 24, dtype=np.float64)
        x = X * 2.0 ** (p - 23)
        s = np.sqrt(x.astype(np.float32))            # IEEE sqrtf on this Mac
        S = (s.astype(np.float64) * 2.0 ** 23).astype(np.int64)   # [2^23, 2^24)
        one = S == (1 << 23)                                      # 1 / 1 is exact
        S = np.where(one, (1 << 23) + 1, S)
        num = np.int64(1) << np.int64(48)
        Q = num // S
        R = num - Q * S
        mid, _ = _distance_classes(Q, R, S)
        mid = np.where(one, np.inf, mid)
        for i in np.argsort(mid, kind="stable")[:count]:
            xs.append(bits32(np.float32(x[i])))
    return xs


def _make(sig, exp2):
    """float32 bits of sig * 2^exp2 (exactly representable by construction)."""
    out = round_float(Fraction(sig) * Fraction(2) ** exp2)
    assert exact(out) == Fraction(sig) * Fraction(2) ** exp2
    return out


# --- the cases ---------------------------------------------------------------------

SPECIAL = {
    "+0": 0x00000000, "-0": 0x80000000, "+inf": 0x7F800000, "-inf": 0xFF800000,
    "nan": 0x7FC00000, "denorm_min": 0x00000001, "-denorm_min": 0x80000001,
    "denorm_max": 0x007FFFFF, "flt_min": 0x00800000, "flt_max": 0x7F7FFFFF,
    "-flt_max": 0xFF7FFFFF, "one": 0x3F800000, "-one": 0xBF800000,
    "two": 0x40000000, "three": 0x40400000, "half": 0x3F000000,
    "one_minus": 0x3F7FFFFF, "one_plus": 0x3F800001, "seven": 0x40E00000,
}


def division_cases():
    """[(a, b, expected)] float32 bits: hardest quotients at several
    exponents, hardest reciprocals (1 / b), subnormal ties and the edges."""
    S = SPECIAL
    cases = []
    for i, (A, B) in enumerate(hardest_quotients()):
        ea, eb = (-23 + (i % 7) * 9 - 27, -23 + (i % 5) * 7 - 14)
        a, b = _make(A, ea), _make(B, eb)
        if i % 3 == 1:
            a |= 0x80000000
        cases.append((a, b))
    for i, B in enumerate(hardest_reciprocals()):
        cases.append((S["one"], _make(B, -23 + (i % 9) - 4)))
    # Subnormal quotients exactly halfway between two neighbours: odd * 2^-150.
    for odd, (num_exp, divisor) in zip((1, 3, 5, 7, 255, 2 ** 22 + 1, 2 ** 23 - 1),
                                       ((-149, 2), (-149, 2), (-148, 4),
                                        (-147, 8), (-140, 1024), (-149, 2),
                                        (-129, 2 ** 21))):
        a = _make(odd, num_exp)
        b = _make(divisor, 0)
        cases.append((a, b))
    for odd in (3, 5, 9, 11):                     # 3 * odd * 2^-150 / 3
        cases.append((_make(3 * odd, -150 + 2), _make(3, 2)))
    edges = [
        ("+0", "one"), ("-0", "one"), ("+0", "-one"), ("one", "+0"),
        ("one", "-0"), ("-one", "+0"), ("+0", "+0"), ("-0", "+0"),
        ("+inf", "two"), ("-inf", "two"), ("two", "+inf"), ("two", "-inf"),
        ("+inf", "+inf"), ("nan", "one"), ("one", "nan"), ("flt_max", "half"),
        ("flt_max", "one_minus"), ("flt_max", "one_plus"), ("-flt_max", "half"),
        ("flt_min", "two"), ("flt_min", "flt_max"), ("denorm_min", "two"),
        ("denorm_min", "three"), ("denorm_min", "one_plus"),
        ("denorm_max", "one_minus"), ("denorm_max", "denorm_min"),
        ("denorm_min", "denorm_max"), ("flt_max", "denorm_min"),
        ("one", "three"), ("two", "three"), ("one", "seven"), ("seven", "three"),
        ("one_minus", "one_plus"), ("one_plus", "one_minus"), ("one", "one"),
    ]
    cases += [(S[x], S[y]) for x, y in edges]
    # FLT_MAX exactly, and quotients past FLT_MAX + half an ulp: infinity.
    cases += [(S["flt_max"], S["one"]),
              (S["flt_max"], _make((1 << 24) - 2, -24))]
    return [(a, b, div(a, b)) for a, b in cases]


def sqrt_cases():
    """[(x, expected)]: the hardest roots at several exponents, perfect
    squares, denormals and the edges."""
    S = SPECIAL
    xs = []
    for i, (X, p) in enumerate(hardest_square_roots()):
        j = (i % 11) * 12 - 60           # x = X * 2^(2j + p - 23), normal
        xs.append(_make(X, 2 * j + p - 23))
    xs += [S[k] for k in ("+0", "-0", "+inf", "-inf", "nan", "denorm_min",
                          "-denorm_min", "denorm_max", "flt_min", "flt_max",
                          "-one", "one", "two", "three", "one_minus",
                          "one_plus", "half")]
    xs += [_make(v, 0) for v in (4, 9, 16, 25, 4095 ** 2)]
    xs += [_make(1, -148), _make(1, -149), _make(3, -149), _make(2 ** 23 - 1, -149)]
    return [(x, sqrt(x)) for x in xs]


def recip_cases():
    """[(x, expected)] for the portable reciprocal (double division,
    narrowed): the hardest reciprocals and the edges, including subnormal
    results."""
    S = SPECIAL
    xs = [_make(B, -23 + (i % 7) - 3) | (0x80000000 if i % 4 == 3 else 0)
          for i, B in enumerate(hardest_reciprocals())]
    xs += [S[k] for k in ("+0", "-0", "+inf", "-inf", "nan", "denorm_min",
                          "denorm_max", "flt_min", "flt_max", "-flt_max",
                          "three", "seven", "one_minus", "one_plus")]
    xs += [_make(3, 125), _make(5, 125), _make((1 << 24) - 1, 104)]
    return [(x, recip_portable(x)) for x in xs]


def rsqrt_cases():
    """[(x, expected)] for the portable rsqrt (sqrtf, then a double division,
    narrowed)."""
    S = SPECIAL
    xs = list(hardest_rsqrt())
    xs += [x for x, _ in sqrt_cases()[:24]]
    xs += [S[k] for k in ("+0", "-0", "+inf", "nan", "-one", "denorm_min",
                          "flt_max", "flt_min", "three")]
    return [(x, rsqrt_portable(x)) for x in xs]


def double_division_cases():
    """[(a, b, expected)] float64 bits for ROM __divdf3 (the soft-float double
    division the portable reciprocal and rsqrt narrow): the hardest 53-bit
    quotients by the same construction, and the edges."""
    rng = np.random.default_rng(29)
    cases = []
    while len(cases) < 12:
        B = (int(rng.integers(1 << 52 - 31, 1 << 53 - 31)) << 31 |
             int(rng.integers(0, 1 << 31))) | 1
        if not (1 << 52) <= B < (1 << 53):
            continue
        inv = pow(1 << 53, -1, B)
        for r in (1, B - 1):
            A = (r * inv) % B
            while A < (1 << 52):
                A += B
            if A < (1 << 53) and A >= B:
                a = round_float(Fraction(A) * Fraction(2) ** (-52 + len(cases)), True)
                b = round_float(Fraction(B) * Fraction(2) ** -52, True)
                cases.append((a, b))
    one = bits64(1.0)
    cases += [(one, bits64(3.0)), (one, bits64(1.5 * 2.0 ** 1023)),
              (bits64(2.0 ** -1074), bits64(2.0)), (bits64(3 * 2.0 ** -1074), bits64(2.0)),
              (one, 0), (0, 0), (one, 0x7FF0000000000000)]
    return [(a, b, div(a, b, True)) for a, b in cases]


def xorshift32(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state


def random_finite(state, subnormal_every=16):
    """(next state, float32 bits): a finite float from one xorshift32 draw --
    every 16th a subnormal, the rest normal with any exponent."""
    state = xorshift32(state)
    sign = state >> 31
    mant = state & 0x7FFFFF
    if (state >> 23) % subnormal_every == 0:
        exp = 0
    else:
        exp = 1 + ((state >> 23) & 0xFF) % 254
    return state, (sign << 31) | (exp << 23) | mant


def random_streams():
    """(division FNV, sqrt FNV): the results of divsqrt_check.cpp's random
    cases, each computed exactly (and checked against numpy's IEEE float32
    on this machine)."""
    seed, count = RANDOM_DIV
    h = 2166136261
    state = seed
    for _ in range(count):
        state, a = random_finite(state)
        state, b = random_finite(state)
        q = div(a, b)
        assert not is_nan(q)
        with np.errstate(all="ignore"):
            assert bits32(np.float32(f32(a)) / np.float32(f32(b))) == q
        h = fnv1a(h, q)
    seed, count = RANDOM_SQRT
    g = 2166136261
    state = seed
    for _ in range(count):
        state, x = random_finite(state)
        x &= 0x7FFFFFFF
        r = sqrt(x)
        assert bits32(np.sqrt(np.float32(f32(x)))) == r
        g = fnv1a(g, r)
    return h, g


def fnv1a(h, word):
    for b in range(4):
        h = ((h ^ ((word >> (8 * b)) & 0xFF)) * 16777619) & 0xFFFFFFFF
    return h


def golden_header():
    """divsqrt_golden.h's text. Every expected result is checked against
    this machine's own IEEE arithmetic (numpy float32 / float64) where it has
    one, so the generator itself is cross-checked."""
    divs, roots = division_cases(), sqrt_cases()
    recips, rsqrts = recip_cases(), rsqrt_cases()
    ddivs = double_division_cases()
    with np.errstate(all="ignore"):
        for a, b, q in divs:
            got = bits32(np.float32(f32(a)) / np.float32(f32(b)))
            assert got == q or (q == ANY_NAN32 and is_nan(got)), (hex(a), hex(b))
        for x, r in roots:
            got = bits32(np.sqrt(np.float32(f32(x))))
            assert got == r or (r == ANY_NAN32 and is_nan(got)), hex(x)
        for x, r in recips:
            got = bits32(np.float32(1.0 / np.float64(np.float32(f32(x)))))
            assert got == r or (r == ANY_NAN32 and is_nan(got)), hex(x)
        for x, r in rsqrts:
            s = np.sqrt(np.float32(f32(x)))
            got = bits32(np.float32(1.0 / np.float64(s)))
            assert got == r or (r == ANY_NAN32 and is_nan(got)), hex(x)
        for a, b, q in ddivs:
            got = bits64(np.float64(f64(a)) / np.float64(f64(b)))
            assert got == q or (q == ANY_NAN64 and is_nan(got, True)), (hex(a), hex(b))
    div_fnv, sqrt_fnv = random_streams()

    def table(name, ctype, rows, width):
        lines = [f"constexpr {ctype} {name}[] = {{"]
        for row in rows:
            lines.append("    {" + ", ".join(f"0x{v:0{width}X}u" if width <= 8
                                             else f"0x{v:0{width}X}ull"
                                             for v in row) + "},")
        lines.append("};")
        return lines

    lines = [
        "// Generated by divsqrt_golden.py golden_header() -- do not edit;",
        "// test_divsqrt.py regenerates it and compares.",
        "//",
        "// The device's float division, sqrtf and ExecuTorch's portable",
        "// reciprocal / rsqrt (a double division, narrowed) against exact",
        "// rounding to nearest even: the hardest quotients, reciprocals and",
        "// square roots, subnormal ties and the edges, as float bits.",
        "// kAnyNan32 / kAnyNan64 as an expected result: any NaN passes.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace espressif {",
        "namespace divsqrt {",
        "namespace golden {",
        "",
        f"constexpr uint32_t kAnyNan32 = 0x{ANY_NAN32:08X}u;",
        f"constexpr uint64_t kAnyNan64 = 0x{ANY_NAN64:016X}ull;",
        "",
        "struct Div32 { uint32_t a, b, q; };",
        "struct Unary32 { uint32_t x, r; };",
        "struct Div64 { uint64_t a, b, q; };",
        "",
    ]
    lines += table("kDiv", "Div32", divs, 8) + [""]
    lines += table("kSqrt", "Unary32", roots, 8) + [""]
    lines += table("kRecip", "Unary32", recips, 8) + [""]
    lines += table("kRsqrt", "Unary32", rsqrts, 8) + [""]
    lines += table("kDiv64", "Div64", ddivs, 16) + [""]
    lines += [
        "// The random streams: xorshift32 from the seed, two draws per",
        "// division (a, then b), one per root (its sign cleared); every 16th",
        "// draw a subnormal. Expected: FNV-1a of every result's bits.",
        f"constexpr uint32_t kRandomDivSeed = 0x{RANDOM_DIV[0]:08X}u;",
        f"constexpr uint32_t kRandomDivCount = {RANDOM_DIV[1]}u;",
        f"constexpr uint32_t kRandomDivFnv = 0x{div_fnv:08X}u;",
        f"constexpr uint32_t kRandomSqrtSeed = 0x{RANDOM_SQRT[0]:08X}u;",
        f"constexpr uint32_t kRandomSqrtCount = {RANDOM_SQRT[1]}u;",
        f"constexpr uint32_t kRandomSqrtFnv = 0x{sqrt_fnv:08X}u;",
        "",
        "}  // namespace golden",
        "}  // namespace divsqrt",
        "}  // namespace espressif",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import sys

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "divsqrt_golden.h")
    text = golden_header()
    if "--check" in sys.argv:
        with open(path) as f:
            sys.exit(0 if f.read() == text else 1)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")
