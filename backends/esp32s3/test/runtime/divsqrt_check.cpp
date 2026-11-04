/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The device's float division and square root against exact rounding
// (divsqrt.h; the cases and their correctly rounded results are
// divsqrt_golden.h, from divsqrt_golden.py). Compiled -ffp-contract=off so
// every expression below is the operations it spells, nothing fused.

#include "divsqrt.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "divsqrt_golden.h"

namespace espressif {
namespace divsqrt {

namespace {

float as_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

uint32_t as_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}

double as_double(uint64_t u) {
  double d;
  memcpy(&d, &u, sizeof d);
  return d;
}

uint64_t as_bits64(double d) {
  uint64_t u;
  memcpy(&u, &d, sizeof u);
  return u;
}

// The operations, each behind volatile operands so that no case is folded
// at build time by the compiler's own (correctly rounded) arithmetic.
float divide(float a, float b) {
  volatile float va = a, vb = b;
  return va / vb;
}

float root(float x) {
  volatile float vx = x;
  return std::sqrt(static_cast<float>(vx));
}

// ExecuTorch's portable kernels for float input (op_reciprocal.cpp,
// op_rsqrt.cpp): `T f(T x) { return 1.0 / x; }` and `return 1.0 /
// std::sqrt(x);` with T = float -- a double division, narrowed on return.
float reciprocal(float x) {
  volatile float vx = x;
  const float y = vx;
  return 1.0 / y;
}

float rsqrt(float x) {
  volatile float vx = x;
  const float y = vx;
  return 1.0 / std::sqrt(y);
}

double divide64(double a, double b) {
  volatile double va = a, vb = b;
  return va / vb;
}

uint32_t fnv1a(uint32_t h, uint32_t word) {
  for (int b = 0; b < 4; ++b) {
    h = (h ^ ((word >> (8 * b)) & 0xFFu)) * 16777619u;
  }
  return h;
}

// divsqrt_golden.random_finite: one xorshift32 draw, every 16th a subnormal,
// the rest normal with any exponent.
uint32_t random_finite(uint32_t* state) {
  uint32_t s = *state;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  *state = s;
  const uint32_t mant = s & 0x7FFFFFu;
  const uint32_t exp = ((s >> 23) % 16u == 0) ? 0u : 1u + ((s >> 23) & 0xFFu) % 254u;
  return (s & 0x80000000u) | (exp << 23) | mant;
}

struct Tally {
  const char* what;
  uint32_t cases = 0, bad = 0;
};

uint32_t g_printed = 0;

void note(Tally& t, bool good, uint64_t x, uint64_t y, uint64_t got,
          uint64_t want) {
  ++t.cases;
  if (good) {
    return;
  }
  ++t.bad;
  if (g_printed++ < 8) {
    printf("  MISMATCH divsqrt %s: %llx, %llx -> %llx, want %llx\n", t.what,
           (unsigned long long)x, (unsigned long long)y,
           (unsigned long long)got, (unsigned long long)want);
  }
}

bool same32(uint32_t got, uint32_t want) {
  return want == golden::kAnyNan32 ? std::isnan(as_float(got)) : got == want;
}

}  // namespace

void run_check() {
  g_printed = 0;
  Tally div{"division"}, sq{"sqrtf"}, rc{"reciprocal"}, rs{"rsqrt"},
      d64{"double division"};
  for (const golden::Div32& c : golden::kDiv) {
    const uint32_t got = as_bits(divide(as_float(c.a), as_float(c.b)));
    note(div, same32(got, c.q), c.a, c.b, got, c.q);
  }
  for (const golden::Unary32& c : golden::kSqrt) {
    const uint32_t got = as_bits(root(as_float(c.x)));
    note(sq, same32(got, c.r), c.x, 0, got, c.r);
  }
  for (const golden::Unary32& c : golden::kRecip) {
    const uint32_t got = as_bits(reciprocal(as_float(c.x)));
    note(rc, same32(got, c.r), c.x, 0, got, c.r);
  }
  for (const golden::Unary32& c : golden::kRsqrt) {
    const uint32_t got = as_bits(rsqrt(as_float(c.x)));
    note(rs, same32(got, c.r), c.x, 0, got, c.r);
  }
  for (const golden::Div64& c : golden::kDiv64) {
    const uint64_t got = as_bits64(divide64(as_double(c.a), as_double(c.b)));
    const bool good = c.q == golden::kAnyNan64 ? std::isnan(as_double(got))
                                               : got == c.q;
    note(d64, good, c.a, c.b, got, c.q);
  }

  uint32_t state = golden::kRandomDivSeed, h = 2166136261u;
  for (uint32_t i = 0; i < golden::kRandomDivCount; ++i) {
    const uint32_t a = random_finite(&state);
    const uint32_t b = random_finite(&state);
    h = fnv1a(h, as_bits(divide(as_float(a), as_float(b))));
  }
  const bool random_div = h == golden::kRandomDivFnv;
  state = golden::kRandomSqrtSeed;
  h = 2166136261u;
  for (uint32_t i = 0; i < golden::kRandomSqrtCount; ++i) {
    const uint32_t x = random_finite(&state) & 0x7FFFFFFFu;
    h = fnv1a(h, as_bits(root(as_float(x))));
  }
  const bool random_sqrt = h == golden::kRandomSqrtFnv;

  const Tally* all[] = {&div, &sq, &rc, &rs, &d64};
  uint32_t cases = 2, bad = !random_div + !random_sqrt;
  for (const Tally* t : all) {
    cases += t->cases;
    bad += t->bad;
  }
  printf("divsqrt check: %u case(s), %u mismatch(es) (division %u/%u, "
         "sqrtf %u/%u, reciprocal %u/%u, rsqrt %u/%u, double division "
         "%u/%u; random division of %u %s, random sqrtf of %u %s)\n",
         (unsigned)cases, (unsigned)bad, (unsigned)(div.cases - div.bad),
         (unsigned)div.cases, (unsigned)(sq.cases - sq.bad),
         (unsigned)sq.cases, (unsigned)(rc.cases - rc.bad),
         (unsigned)rc.cases, (unsigned)(rs.cases - rs.bad),
         (unsigned)rs.cases, (unsigned)(d64.cases - d64.bad),
         (unsigned)d64.cases, (unsigned)golden::kRandomDivCount,
         random_div ? "ok" : "HASH MISMATCH",
         (unsigned)golden::kRandomSqrtCount,
         random_sqrt ? "ok" : "HASH MISMATCH");
}

}  // namespace divsqrt
}  // namespace espressif
