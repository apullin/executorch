/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The code arithmetic of the asymmetric schemes, shared by the two units that
// compute it: esp32_asym.cpp (the affine loops) and esp32_conv.cpp (the static
// int8 convolution path). Every definition here is internal to each unit that
// includes it, and both units build -ffp-contract=off (both CMakeLists), so
// each float operation below is one IEEE operation in either: the two paths
// cannot round differently, which is what lets esp32_conv.cpp's output equal
// the affine loops' bit for bit.
//
// Twins in esp32_backend.cpp (f16_to_f32, round_half_*_i): change both together.
#pragma once

#include <cstddef>
#include <cstdint>

#include "esp32_platform.h"
#include "quant_rules.h"

namespace esp32_codes {
namespace {

/// IEEE half -> float, exactly: esp32_backend.cpp's f16_to_f32 (integer bit
/// manipulation, so the flags of either file cannot change it).
inline float half_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      uint32_t m = mant;
      int e = -1;
      do {
        m <<= 1;
        ++e;
      } while ((m & 0x400u) == 0);
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  // A union, not memcpy: at -Os GCC leaves a 4-byte memcpy as a ROM call.
  const union {
    uint32_t bits;
    float value;
  } pun = {bits};
  return pun.value;
}

/// Element `index` of a table of fp32 or IEEE-half values.
inline float table_float(const void* table, bool half, size_t index) {
  if (half) {
    return half_to_float(static_cast<const uint16_t*>(table)[index]);
  }
  return static_cast<const float*>(table)[index];
}

/// Round to nearest, ties to even: torch.round. esp32_backend.cpp's
/// round_half_even_i, here under this file's -ffp-contract=off: v - floor(v)
/// is exact wherever it can be near 0.5 (Sterbenz), and nothing fuses it
/// with the multiply that made v. |v| <= kRail (code_of's clamp) or 127, or
/// NaN, which rounds to 0: the zero point's code (quant_rules.h).
inline int32_t rne_i(float v) {
  v = quant_scaled(v);
  const int32_t f = esp32_platform_floor_i(v);
  const float frac = v - (float)f;
  return (frac > 0.5f || (frac == 0.5f && (f & 1))) ? f + 1 : f;
}

/// Half away from zero: esp32_backend.cpp's round_half_away_i, the +0.5f a
/// float add of its own. Always inline, as there.
inline __attribute__((always_inline)) int32_t half_away_i(float v) {
  return v < 0.0f ? -esp32_platform_floor_i(-v + 0.5f)
                  : esp32_platform_floor_i(v + 0.5f);
}

/// fl(1 / s), or 0 for a zero or subnormal scale (that row's codes are all
/// 0): quant_rules.h's a0_inverse_scale.
inline float a0_inverse(float s) {
  return a0_inverse_scale(s);
}

/// One A0 code: fl(x * inv) rounded half away (or to even), clamped to +-127.
/// Always inline, as code_of: it is each A0 loop's body.
inline __attribute__((always_inline)) int32_t a0_code(float x, float inv,
                                                      bool ties_to_even) {
  const float v = x * inv;
  const int32_t q = ties_to_even ? rne_i(v) : half_away_i(v);
  return q > 127 ? 127 : (q < -127 ? -127 : q);
}

inline int32_t clamp_code(int32_t q, int32_t qmin, int32_t qmax) {
  return q < qmin ? qmin : (q > qmax ? qmax : q);
}

/// Where code_of clamps fl(x * inv) before rounding: 2^22, below which floor
/// and the fraction stay exact and int32 cannot overflow, and far past every
/// code a 16-bit range and zero point can reach, so no code changes. A
/// dynamic scale keeps |x * inv| within qmax - qmin anyway; a static one
/// (the model's calibrated constant) does not, and an outlier far past the
/// calibrated range must still land on its rail, as torch's clamp puts it.
constexpr float kRail = 4194304.0f;

/// One activation's code: clamp(rne(fl(x * inv)) + zx). The add is exact (an
/// integral float plus a small integer), so integers stand in for torch's
/// float32 sum. NaN takes zx and an infinity its rail, ExecuTorch 1.6's rule
/// (quant_rules.h): the rails let NaN through, and rne_i rounds it to 0.
/// Always inline: it is each quantise loop's body, and at -Os GCC outlines it
/// once the int16 quantisers make it six call sites.
inline __attribute__((always_inline)) int32_t code_of(float x, float inv,
                                                      int32_t zx, int32_t qmin,
                                                      int32_t qmax) {
  float v = x * inv;
  v = v < -kRail ? -kRail : v;
  v = v > kRail ? kRail : v;
  return clamp_code(rne_i(v) + zx, qmin, qmax);
}

}  // namespace
}  // namespace esp32_codes
