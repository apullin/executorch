/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Single-precision elementary functions for the delegate's activation
// epilogues. newlib's expf/tanhf/erff go through errno wrappers and the ROM's
// software float division; these are straight-line float code. Accuracy,
// measured on the host against double references over the epilogue ranges:
// expf <= 1.01 ulp, tanhf <= 1.30 ulp, GELU <= 3.9e-7 absolute (the erf form
// would give 4.5e-7: 1 + erf(x) cancels for negative x).
#pragma once
#include <math.h>
#include <stdint.h>

#if defined(ESP_PLATFORM)
#include "esp_attr.h"
// Internal SRAM: const data would land in flash .rodata, where a cache line
// evicted by the weight stream costs more to refill than the exp itself.
#define FAST_MATH_DRAM DRAM_ATTR
#else
#define FAST_MATH_DRAM
#endif

// e^x with Cephes' range reduction and minimax polynomial (<= 1.01 ulp, above).
// NaN propagates; below FLT_MIN's exponent (x < -87.34, including -inf) the
// result flushes to +0, above 88.72 it is +inf.
static inline float fast_expf(float x) {
  if (x != x) return x;
  if (x > 88.7228317f) return INFINITY;
  if (x < -87.3365447f) return 0.0f;
  const float fx = x * 1.44269504088896341f;
  int n = (int)(fx + (fx >= 0.0f ? 0.5f : -0.5f));
  const float fn = (float)n;
  float r = x - fn * 0.693359375f;
  r = r - fn * -2.12194440e-4f;
  const float z = r * r;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * z + r + 1.0f;
  if (n > 127) {
    p *= 2.0f;
    n = 127;
  }
  // A union, not memcpy: at -Os GCC leaves a 4-byte memcpy as a ROM call,
  // one per exp. GCC and Clang define type punning through a union.
  union {
    uint32_t bits;
    float value;
  } scale = {(uint32_t)(n + 127) << 23};
  return p * scale.value;
}

// 2^(j/64) correctly rounded to float32, as bits, minus j << 17. fast_exp_neg
// adds k << 17 for k = 64n + j, which leaves 2^(j/64) with n added to its
// exponent field: one integer add scales by 2^n, and no mask is needed to
// split k. test_attn_tail.py re-derives every entry exactly.
static const uint32_t kFastExpNegTable[64] FAST_MATH_DRAM = {
    0x3f800000, 0x3f7f64d2, 0x3f7ecd87, 0x3f7e3a29, 0x3f7daac3, 0x3f7d1f62,
    0x3f7c980f, 0x3f7c14d5, 0x3f7b95c2, 0x3f7b1adf, 0x3f7aa43a, 0x3f7a31dc,
    0x3f79c3d3, 0x3f795a2b, 0x3f78f4f0, 0x3f78942d, 0x3f7837f0, 0x3f77e046,
    0x3f778d3a, 0x3f773eda, 0x3f76f532, 0x3f76b051, 0x3f767043, 0x3f763516,
    0x3f75fed7, 0x3f75cd94, 0x3f75a15b, 0x3f757a3a, 0x3f75583f, 0x3f753b79,
    0x3f7523f6, 0x3f7511c4, 0x3f7504f3, 0x3f74fd92, 0x3f74fbaf, 0x3f74ff5b,
    0x3f7508a4, 0x3f75179a, 0x3f752c4d, 0x3f7546cd, 0x3f75672a, 0x3f758d75,
    0x3f75b9be, 0x3f75ec15, 0x3f76248c, 0x3f766334, 0x3f76a81e, 0x3f76f35b,
    0x3f7744fd, 0x3f779d16, 0x3f77fbb8, 0x3f7860f5, 0x3f78ccdf, 0x3f793f89,
    0x3f79b907, 0x3f7a396a, 0x3f7ac0c7, 0x3f7b4f30, 0x3f7be4ba, 0x3f7c8177,
    0x3f7d257d, 0x3f7dd0df, 0x3f7e83b3, 0x3f7f3e0c,
};

#define FAST_MATH_INLINE inline __attribute__((always_inline))

// A float's bits and back. The S3's GCC moves these through the stack (a store
// and a load) at -Os and -O2 alike; rfr and wfr cross register files directly.
static FAST_MATH_INLINE uint32_t fast_math_bits(float value) {
#if defined(__XTENSA__)
  uint32_t bits;
  __asm__("rfr %0, %1" : "=a"(bits) : "f"(value));
  return bits;
#else
  const union {
    float value;
    uint32_t bits;
  } pun = {value};
  return pun.bits;
#endif
}

static FAST_MATH_INLINE float fast_math_float(uint32_t bits) {
#if defined(__XTENSA__)
  float value;
  __asm__("wfr %0, %1" : "=f"(value) : "a"(bits));
  return value;
#else
  const union {
    uint32_t bits;
    float value;
  } pun = {bits};
  return pun.value;
#endif
}

// fast_exp_neg's constants, for a loop to keep in registers.
struct FastExpNeg {
  float shifter;     // 1.5 * 2^23
  float inv_ln2_64;  // 64/ln2
  float ln2_64_hi;   // fast_expf's 0.693359375, / 64: 9 significant bits
  float ln2_64_lo;   // fast_expf's -2.12194440e-4, / 64
  float c3, c2;      // 1/6, 1/2
  float flush;       // fast_expf's flush threshold, ~ln(FLT_MIN)
  float zero;
  const uint32_t* table;
};

static FAST_MATH_INLINE FastExpNeg fast_exp_neg_constants() {
  return {0x1.8p23f,       0x1.715476p+6f, 0x1.63p-7f,   -0x1.bd0106p-19f,
          0x1.555556p-3f,  0.5f,           -87.3365447f, 0.0f,
          kFastExpNegTable};
}

// fast_exp_neg_constants() for a loop. The empty asm hides the values, so GCC
// holds each in a register rather than rebuilding it from a literal (l32r,
// wfr) at every use, which -Os otherwise does inside the loop.
static FAST_MATH_INLINE FastExpNeg fast_exp_neg_hoisted() {
  FastExpNeg c = fast_exp_neg_constants();
#if defined(__XTENSA__)
  __asm__("" : "+f"(c.shifter), "+f"(c.inv_ln2_64), "+f"(c.ln2_64_hi),
          "+f"(c.ln2_64_lo), "+f"(c.c3), "+f"(c.c2), "+f"(c.flush),
          "+f"(c.zero));
#endif
  return c;
}

// Each lane loop is unrolled even at -Os, which otherwise keeps the loop and
// the lanes in stack arrays.
#define FAST_MATH_EACH_LANE(i) \
  _Pragma("GCC unroll 8") for (int i = 0; i < N; ++i)

// e^x for x <= 0, the softmax's exp, over N independent lanes:
// 2^n * 2^(j/64) * e^r with k = 64n + j = round(x * 64/ln2), |r| <= ln2/128.
//  - Rounding: adding 1.5 * 2^23 leaves k in the low mantissa bits of z, and
//    z - 1.5 * 2^23 is k exactly; no float-to-int conversion.
//  - Reduction: k * hi and x - k * hi are exact for |k| < 2^13, which covers
//    every input above the flush threshold.
//  - e^r - 1 as r + r^2 (1/2 + r/6): its truncation error, below 4e-11, is
//    far under float32's ulp.
//  - Scaling: the table entry plus k << 17 is 2^(j/64) * 2^n (see the table).
// Every fused step is an explicit fmaf, so GCC's -ffp-contract=fast cannot
// move a fusion site; models.py's fast_exp_neg_f32_reference is this
// arithmetic, bit for bit. Measured on the host against float64 over every
// float32 in [-87.3365447, 0]: <= 1.004 ulp. NaN propagates and x < -87.3365447,
// including -inf, flushes to +0, as in fast_expf. Positive x is outside the
// contract: the exponent add wraps and the result is meaningless (attn_tail's
// s - max is <= 0 unless a NaN score has already poisoned the row).
//
// Each step is written for every lane before the next. GCC keeps that source
// order on the LX7, so one lane's FPU latency is spent on the other lanes'
// work; the empty asm on each result stops GCC sinking a lane's arithmetic
// under the flush test, which would serialise the lanes again. Every lane
// computes exactly what fast_exp_neg does.
template <int N>
static FAST_MATH_INLINE void fast_exp_neg_lanes(const FastExpNeg& c,
                                                const float (&x)[N],
                                                float (&y)[N]) {
  float z[N], kf[N], r[N], p[N];
  uint32_t k[N];
  FAST_MATH_EACH_LANE(i) z[i] = fmaf(x[i], c.inv_ln2_64, c.shifter);
  FAST_MATH_EACH_LANE(i) k[i] = fast_math_bits(z[i]);
  FAST_MATH_EACH_LANE(i) kf[i] = z[i] - c.shifter;
  FAST_MATH_EACH_LANE(i) r[i] = fmaf(-kf[i], c.ln2_64_hi, x[i]);
  FAST_MATH_EACH_LANE(i) r[i] = fmaf(-kf[i], c.ln2_64_lo, r[i]);
  FAST_MATH_EACH_LANE(i) {
    const float r2 = r[i] * r[i];
    p[i] = fmaf(r2, fmaf(r[i], c.c3, c.c2), r[i]);
  }
  FAST_MATH_EACH_LANE(i) {
    const float t = fast_math_float(c.table[k[i] & 63] + (k[i] << 17));
    y[i] = fmaf(t, p[i], t);
#if defined(__XTENSA__)
    __asm__ volatile("" : "+f"(y[i]));
#endif
  }
  FAST_MATH_EACH_LANE(i) y[i] = x[i] < c.flush ? c.zero : y[i];
}

static FAST_MATH_INLINE float fast_exp_neg(float x) {
  const float in[1] = {x};
  float out[1];
  fast_exp_neg_lanes<1>(fast_exp_neg_constants(), in, out);
  return out[0];
}

// tanh(x): Cephes' odd polynomial below |x| = 0.625, 1 - 2/(e^2|x| + 1) above.
static inline float fast_tanhf(float x) {
  if (x == 0.0f) return x;  // keeps the sign of zero, as newlib does
  const float a = fabsf(x);
  if (a < 0.625f) {
    const float z = x * x;
    float p = -5.70498872745e-3f;
    p = p * z + 2.06390887954e-2f;
    p = p * z - 5.37397155531e-2f;
    p = p * z + 1.33314422036e-1f;
    p = p * z - 3.33332819422e-1f;
    return p * z * x + x;
  }
  // tanh(x) rounds to 1.0f only above ~9.0115; below that the formula's
  // 0.99999994 is the correctly rounded value.
  if (a > 9.1f) return x > 0.0f ? 1.0f : -1.0f;
  const float t = 1.0f - 2.0f / (fast_expf(2.0f * a) + 1.0f);
  return x < 0.0f ? -t : t;
}

// erfc(x) with Numerical Recipes' Chebyshev fit (relative error < 1.2e-7
// before float rounding), valid for every real x.
static inline float fast_erfcf(float x) {
  const float z = fabsf(x);
  const float t = 1.0f / (1.0f + 0.5f * z);
  float p = 0.17087277f;
  p = p * t - 0.82215223f;
  p = p * t + 1.48851587f;
  p = p * t - 1.13520398f;
  p = p * t + 0.27886807f;
  p = p * t - 0.18628806f;
  p = p * t + 0.09678418f;
  p = p * t + 0.37409196f;
  p = p * t + 1.00002368f;
  const float ans = t * fast_expf(-z * z - 1.26551223f + t * p);
  return x >= 0.0f ? ans : 2.0f - ans;
}

// Exact GELU through erfc, which keeps relative accuracy for negative inputs:
// 0.5 x (1 + erf(x/sqrt2)) == 0.5 x erfc(-x/sqrt2).
static inline float fast_gelu(float x) {
  return 0.5f * x * fast_erfcf(-x * 0.70710678118654752f);
}
