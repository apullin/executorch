/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The chip's expf, on the chip and on the host: what the softmax override
// (espressif_ops.cpp shim_softmax_out) takes exp from.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

// newlib's expf (libm/math/wf_exp.c over ef_exp.c, fdlibm's float exp) as
// the ESP32-S3 toolchain's libm runs it on the chip, transcribed from its
// disassembly in a story image (esp-14.2.0_20260121): the reduction's
// k = invln2 * x + halF[sign] is a madd.s and hi = x - t * ln2HI an msub.s,
// the polynomial is four madd.s and one msub.s, each rounded once (std::fma
// here), the two divisions are __divsf3 (correctly rounded, as `/` is), and
// everything else is a separately rounded float op. The wrapper's errno
// handling changes no value. The host's softmax uses this, so the host runs
// the chip's exp bit for bit whatever its own libm and -ffp-contract, and
// test/reference.py's softmax mirror (quantizer/mirrors.py expf_newlib)
// transcribes the same code. The chip keeps calling newlib's expf itself.
#if !defined(__XTENSA__)
static inline float newlib_expf(float x) {
  static constexpr float kHalf[2] = {0.5f, -0.5f};
  static constexpr float kLn2Hi[2] = {6.9313812256e-01f, -6.9313812256e-01f};
  static constexpr float kLn2Lo[2] = {9.0580006145e-06f, -9.0580006145e-06f};
  constexpr float kInvLn2 = 1.4426950216e+00f;
  constexpr float kP1 = 1.6666667163e-01f;
  constexpr float kP2 = -2.7777778450e-03f;
  constexpr float kP3 = 6.6137559770e-05f;
  constexpr float kP4 = -1.6533901999e-06f;
  constexpr float kP5 = 4.1381369442e-08f;
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  const int32_t signed_bits = static_cast<int32_t>(bits);
  const uint32_t sign = bits >> 31;
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return x + x;  // NaN
  }
  if (magnitude == 0x7f800000u) {
    return sign == 0 ? x : 0.0f;  // exp(+-inf) = {inf, 0}
  }
  if (signed_bits > 0x42b17217) {
    return INFINITY;  // __math_oflowf(0)
  }
  if (sign != 0 && magnitude > 0x42cff1b5u) {
    return 0.0f;  // __math_uflowf(0)
  }
  float hi = 0.0f;
  float lo = 0.0f;
  int32_t k = 0;
  if (magnitude > 0x3eb17218u) {  // |x| > ln2 / 2
    if (magnitude < 0x3f851592u) {  // and |x| < 1.5 ln2
      hi = x - kLn2Hi[sign];
      lo = kLn2Lo[sign];
      k = 1 - static_cast<int32_t>(sign) - static_cast<int32_t>(sign);
    } else {
      k = static_cast<int32_t>(std::fma(kInvLn2, x, kHalf[sign]));
      const float t = static_cast<float>(k);
      hi = std::fma(-t, kLn2Hi[0], x);  // t * ln2HI is exact
      lo = t * kLn2Lo[0];
    }
    x = hi - lo;
  } else if (magnitude < 0x34000000u) {  // |x| < 2^-23
    return 1.0f + x;
  }
  const float t = x * x;
  float p = std::fma(t, kP5, kP4);
  p = std::fma(p, t, kP3);
  p = std::fma(p, t, kP2);
  p = std::fma(p, t, kP1);
  const float c = std::fma(-p, t, x);
  if (k == 0) {
    return 1.0f - ((x * c) / (c - 2.0f) - x);
  }
  float y = 1.0f - ((lo - (x * c) / (2.0f - c)) - hi);
  std::memcpy(&bits, &y, sizeof(bits));
  if (k >= -125) {
    bits += static_cast<uint32_t>(k) << 23;
    std::memcpy(&y, &bits, sizeof(y));
    return y;
  }
  bits += static_cast<uint32_t>(k + 100) << 23;
  std::memcpy(&y, &bits, sizeof(y));
  return y * 0x1p-100f;
}
#endif

// The chip's expf, on the chip and on the host.
static inline float espressif_softmax_expf(float x) {
#if defined(__XTENSA__)
  return std::exp(x);
#else
  return newlib_expf(x);
#endif
}

namespace espressif {
namespace expf_check {

/// The chip's expf against test/runtime/expf_golden.h (mirrors.expf_newlib's
/// bits): every branch boundary of newlib's expf and two xorshift32 streams
/// (a dense range sweep and random finite bit patterns), through
/// espressif_softmax_expf, so newlib's own expf on the chip (the parity
/// firmware, QEMU or silicon) and the host's transcription of it
/// (--selftest). Prints one "expf check:" line and each mismatch; returns
/// the mismatches. Defined in test/runtime/expf_check.cpp, which only test
/// builds link.
uint32_t run_check();

}  // namespace expf_check
}  // namespace espressif
