/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Float-weight loops: the fp32 convolution, and the fp16/bf16 rungs for
// matmul, convolution and embedding. See esp32_float.h.

#include "esp32_float.h"

namespace esp32_float {
namespace {

// A float from its bits. Through a union rather than wfr: GCC's Xtensa
// backend gives no loop holding an inline asm a zero-overhead LOOP, and the
// union's store and reload cost less than the branch and counter it saves.
inline __attribute__((always_inline)) float float_of(uint32_t bits) {
  const union {
    uint32_t bits;
    float value;
  } pun = {bits};
  return pun.value;
}

/// The halves a weight rarely holds -- zero, subnormals, infinities, NaNs --
/// from their magnitude bits `em` (the sign is the caller's).
inline __attribute__((always_inline)) uint32_t rare_half_bits(uint32_t em) {
  if (em >= 0x7C00u) {
    // Infinity, or a NaN with its payload.
    return 0x7F800000u | ((em & 0x3FFu) << 13);
  }
  if (em == 0) {
    return 0;
  }
  // Subnormal: em * 2^-24 with em < 2^10. Its top bit p (0..9) becomes the
  // implicit bit of an fp32 normal with exponent p - 24.
  const uint32_t p = 31u - (uint32_t)__builtin_clz(em);
  return ((p + 127u - 24u) << 23) | ((em << (23u - p)) & 0x7FFFFFu);
}

/// f16_bits, inlined into the loops that widen as they read. A normal half
/// is a shift and an add -- the exponent rebiased from 15 to 127, the
/// mantissa widened -- and one unsigned compare sends every other pattern to
/// rare_half_bits, off the path a weight takes (written as one if-chain, GCC
/// computes all three cases for every element).
inline __attribute__((always_inline)) uint32_t half_bits(uint16_t h) {
  const uint32_t em = h & 0x7FFFu;
  uint32_t bits = (em << 13) + ((127u - 15u) << 23);
  // em - 0x400 wraps past 0x77FF exactly when em < 0x400 or em >= 0x7C00.
  if (__builtin_expect(em - 0x0400u > 0x77FFu, 0)) {
    bits = rare_half_bits(em);
  }
  return bits | ((uint32_t)(h >> 15) << 31);
}

// The three element types, each read as the float it stores.
struct F16 {
  uint16_t bits;
};
struct Bf16 {
  uint16_t bits;
};
inline __attribute__((always_inline)) float widen(float w) { return w; }
inline __attribute__((always_inline)) float widen(F16 w) {
  return float_of(half_bits(w.bits));
}
inline __attribute__((always_inline)) float widen(Bf16 w) {
  return float_of(bf16_bits(w.bits));
}

/// acc + x[0] * w[0] + ... + x[k-1] * w[k-1], one step at a time in k order:
/// opcode 1's sum for one output element. A counted loop over walking
/// pointers, which GCC turns into a zero-overhead LOOP.
template <typename W>
inline __attribute__((always_inline)) float dot(float acc, const float* x,
                                                const W* w, uint32_t k) {
  for (; k != 0; --k) {
    acc += *x++ * widen(*w++);
  }
  return acc;
}

/// dot over a patch read in place: tap t is x[offs[t]], or a real 0.0 where
/// offs[t] is -1 -- the gathered patch's values, so the same sum.
template <typename W>
inline __attribute__((always_inline)) float tap_dot(float acc, const float* x,
                                                    const int32_t* offs,
                                                    const W* w, uint32_t k) {
  for (; k != 0; --k) {
    const int32_t o = *offs++;
    const float v = o >= 0 ? x[o] : 0.0f;
    acc += v * widen(*w++);
  }
  return acc;
}

/// `rows` rows of k stored elements widened into dst as the floats they
/// store (the values widen() gives), row by row.
template <typename W>
void widen_rows(const W* w, uint32_t rows, uint32_t k, float* dst) {
  for (size_t t = (size_t)rows * k; t != 0; --t) {
    *dst++ = widen(*w++);
  }
}

/// Output rows [j0, j1) of a matmul over weight rows w (row j at
/// w + (j - j0) * k): y[r * n + j] for every r of m.
template <typename W>
void matmul_rows(const W* w, const float* x, const float* bias, float* y,
                 uint32_t m, uint32_t k, uint32_t n, uint32_t j0,
                 uint32_t j1) {
  for (uint32_t r = 0; r < m; ++r) {
    const float* xr = x + (size_t)r * k;
    float* yr = y + (size_t)r * n;
    for (uint32_t j = j0; j < j1; ++j) {
      yr[j] = dot(bias != nullptr ? bias[j] : 0.0f, xr,
                  w + (size_t)(j - j0) * k, k);
    }
  }
}

/// Output channels [j0, j1) of a convolution over weight rows w (row j at
/// w + (j - j0) * k): every output position r of m, NCHW y[j * m + r].
template <typename W>
void conv_rows(const W* w, const float* x, const esp32_asym::ConvGather& g,
               bool depthwise, const float* bias, float* y, uint32_t m,
               uint32_t k, uint32_t j0, uint32_t j1, void* scratch) {
  if (depthwise) {
    // Output channel j reads input channel j alone: a one-channel conv over
    // its own plane, whose taps sit at the same offsets in every plane. So
    // the walk runs once per position, and each channel reads its taps in
    // place (k is a few taps; a gather per channel would cost more than the
    // dot, as it does in run_depthwise_i8).
    int32_t* offs = static_cast<int32_t*>(scratch);
    const size_t plane = (size_t)g.d_in * g.h_in * g.w_in;
    for (uint32_t r = 0; r < m; ++r) {
      esp32_asym::conv_tap_offsets(g, r, offs);
      for (uint32_t j = j0; j < j1; ++j) {
        y[(size_t)j * m + r] =
            tap_dot(bias != nullptr ? bias[j] : 0.0f, x + (size_t)j * plane,
                    offs, w + (size_t)(j - j0) * k, k);
      }
    }
    return;
  }
  float* patch = static_cast<float*>(scratch);
  for (uint32_t r = 0; r < m; ++r) {
    esp32_asym::gather_conv_row(x, g, r, patch);
    for (uint32_t j = j0; j < j1; ++j) {
      y[(size_t)j * m + r] = dot(bias != nullptr ? bias[j] : 0.0f, patch,
                                 w + (size_t)(j - j0) * k, k);
    }
  }
}

/// A 2-byte weight's matmul: its rows widened once, a block at a time, into
/// `wide` and every row of x dotted with the fp32 values there, or (one row
/// of x, or a row wider than the block) widened as they are read.
template <typename W>
void matmul_t(const W* w, const float* x, const float* bias, float* y,
              uint32_t m, uint32_t k, uint32_t n, float* wide) {
  const uint32_t jb = widen_block(m, k, n);
  if (jb == 0) {
    matmul_rows(w, x, bias, y, m, k, n, 0, n);
    return;
  }
  for (uint32_t j0 = 0; j0 < n; j0 += jb) {
    const uint32_t j1 = j0 + jb < n ? j0 + jb : n;
    widen_rows(w + (size_t)j0 * k, j1 - j0, k, wide);
    matmul_rows(wide, x, bias, y, m, k, n, j0, j1);
  }
}

template <typename W>
void conv_t(const W* w, const float* x, const esp32_asym::ConvGather& g,
            bool depthwise, const float* bias, float* y, uint32_t m,
            uint32_t k, uint32_t n, void* scratch) {
  const uint32_t jb = widen_block(m, k, n);
  if (jb == 0) {
    conv_rows(w, x, g, depthwise, bias, y, m, k, 0, n, scratch);
    return;
  }
  // The block after the patch (or the taps): k words in, as scratch_bytes
  // lays it out.
  float* wide = static_cast<float*>(scratch) + k;
  for (uint32_t j0 = 0; j0 < n; j0 += jb) {
    const uint32_t j1 = j0 + jb < n ? j0 + jb : n;
    widen_rows(w + (size_t)j0 * k, j1 - j0, k, wide);
    conv_rows(static_cast<const float*>(wide), x, g, depthwise, bias, y, m, k,
              j0, j1, scratch);
  }
}

}  // namespace

uint32_t f16_bits(uint16_t h) { return half_bits(h); }

void matmul(Storage s, const void* w, const float* x, const float* bias,
            float* y, uint32_t m, uint32_t k, uint32_t n, void* scratch) {
  float* wide = static_cast<float*>(scratch);
  switch (s) {
    case Storage::kF16:
      matmul_t(static_cast<const F16*>(w), x, bias, y, m, k, n, wide);
      break;
    case Storage::kBf16:
      matmul_t(static_cast<const Bf16*>(w), x, bias, y, m, k, n, wide);
      break;
    case Storage::kF32:
      // Unreachable: execute() sends opcode 1's matmul to run_addmm_fp32.
      break;
  }
}

void conv(Storage s, const void* w, const float* x,
          const esp32_asym::ConvGather& g, bool depthwise, const float* bias,
          float* y, uint32_t m, uint32_t k, uint32_t n, void* scratch) {
  switch (s) {
    case Storage::kF16:
      conv_t(static_cast<const F16*>(w), x, g, depthwise, bias, y, m, k, n,
             scratch);
      break;
    case Storage::kBf16:
      conv_t(static_cast<const Bf16*>(w), x, g, depthwise, bias, y, m, k, n,
             scratch);
      break;
    case Storage::kF32:
      conv_rows(static_cast<const float*>(w), x, g, depthwise, bias, y, m, k,
                0, n, scratch);
      break;
  }
}

void embed_row(Storage s, const void* table, uint32_t row, uint32_t d,
               float* dst) {
  uint32_t* out = reinterpret_cast<uint32_t*>(dst);
  const uint16_t* src = static_cast<const uint16_t*>(table) + (size_t)row * d;
  if (s == Storage::kBf16) {
    for (uint32_t t = 0; t < d; ++t) {
      out[t] = bf16_bits(src[t]);
    }
  } else {
    for (uint32_t t = 0; t < d; ++t) {
      out[t] = half_bits(src[t]);
    }
  }
}

}  // namespace esp32_float
