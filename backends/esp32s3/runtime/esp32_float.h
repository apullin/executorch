/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Float weights in the delegate: the fp32 convolution, and fp16/bf16 as stored
// weight formats for matmul, convolution and embedding
// (serialization/espressif_blob.py has the wire format).
//
// A 2-byte weight is stored as shipped and widened to fp32 as it is read: bf16
// by a 16-bit shift, fp16 exactly (normals, subnormals, infinities, NaN
// payloads). The arithmetic after widening is the fp32 loop's -- y = bias, then
// y += x * w in k order -- the sum esp32_backend.cpp's opcode 1 forms. This
// file is built with that file's contraction (the device fuses each step into
// madd.s, one rounding; the host builds both with -ffp-contract=off, two), so
// a 2-byte weight computes exactly what the fp32 loop computes on its widened
// value, on either side.
//
// Loops only: esp32_backend.cpp checks the shapes, applies the fused epilogue
// over what these write, and counts the work.

#pragma once

#include <cstddef>
#include <cstdint>

#include "esp32_asym.h"

namespace esp32_float {

// The opcode selects the weight's storage type, as across the integer ladder.
// Weights [n, k] row-major (the torch Linear/conv layout), inline or one
// external key, k_pad == k, no scales. A runtime without these opcodes
// refuses them as unknown.
constexpr uint32_t kOpAddmmF16 = 11;
constexpr uint32_t kOpAddmmBf16 = 12;
constexpr uint32_t kOpEmbedF16 = 13;
constexpr uint32_t kOpEmbedBf16 = 14;

inline bool is_matmul(uint32_t opcode) {
  return opcode == kOpAddmmF16 || opcode == kOpAddmmBf16;
}
inline bool is_embed(uint32_t opcode) {
  return opcode == kOpEmbedF16 || opcode == kOpEmbedBf16;
}
inline bool is_op(uint32_t opcode) {
  return is_matmul(opcode) || is_embed(opcode);
}

/// How a weight element is stored.
enum class Storage : uint32_t { kF32, kF16, kBf16 };

/// The storage of a float-weight opcode (1, 11-14).
inline Storage storage_of(uint32_t opcode) {
  if (opcode == kOpAddmmF16 || opcode == kOpEmbedF16) {
    return Storage::kF16;
  }
  if (opcode == kOpAddmmBf16 || opcode == kOpEmbedBf16) {
    return Storage::kBf16;
  }
  return Storage::kF32;
}

/// The fp32 bit pattern of an IEEE binary16 value, exactly: every finite
/// half is an fp32 value, an infinity stays one, and a NaN keeps its sign and
/// payload (shifted up 13 bits, so a quiet NaN stays quiet).
uint32_t f16_bits(uint16_t h);

/// The fp32 bit pattern of a bfloat16 value: its 16 bits are fp32's top half.
inline uint32_t bf16_bits(uint16_t h) { return (uint32_t)h << 16; }

/// What one block of widened 2-byte weight rows may cost in scratch: the
/// per-call budget an int8 row block has (esp32_backend.cpp's
/// kRowBlockScratchBytes).
constexpr uint32_t kWidenBytes = 16 * 1024;

/// How many rows of a 2-byte weight are widened to fp32 at a time, once for
/// all m rows (positions) of x -- or 0 where they are widened as they are
/// read: a single row (or position) reuses no widened value, and a row wider
/// than the budget does not fit. The loops and the scratch planner both ask.
inline uint32_t widen_block(uint32_t m, uint32_t k, uint32_t n) {
  const uint64_t row = (uint64_t)k * sizeof(float);
  if (m < 2 || row == 0 || row > kWidenBytes) {
    return 0;
  }
  const uint64_t rows = kWidenBytes / row;
  return rows < n ? (uint32_t)rows : n;
}

/// The scratch a float-weight op's loop takes: a convolution's patch (or a
/// depthwise op's tap offsets), k words, then for a 2-byte weight its
/// widened block (widen_block rows of k floats).
inline uint64_t scratch_bytes(Storage s, bool conv, uint32_t m, uint32_t k,
                              uint32_t n) {
  const uint64_t words = conv ? (uint64_t)k : 0;
  const uint64_t block =
      s == Storage::kF32 ? 0 : (uint64_t)widen_block(m, k, n) * k;
  return (words + block) * sizeof(float);
}

/// y[r * n + j] = bias[j] (0 without one) + sum over t of x[r * k + t] *
/// w[j * k + t], for m rows. No epilogue. `scratch` holds scratch_bytes(s,
/// false, m, k, n).
void matmul(Storage s, const void* w, const float* x, const float* bias,
            float* y, uint32_t m, uint32_t k, uint32_t n, void* scratch);

/// A convolution: every output position r of m gathers its patch (k values,
/// quantize_conv_row's column order and padding) and dots it with each
/// output channel's weight row, writing NCHW y[j * m + r]. `x` and `y` are
/// this op's channel slice (a grouped conv's group already applied). A
/// depthwise op (`depthwise`) reads input channel j for output channel j
/// through a one-channel `g`, its taps in place. `scratch` holds
/// scratch_bytes(s, true, m, k, n): k floats (the patch) or k int32 offsets
/// (a depthwise op's taps), then any widened block. No epilogue.
void conv(Storage s, const void* w, const float* x,
          const esp32_asym::ConvGather& g, bool depthwise, const float* bias,
          float* y, uint32_t m, uint32_t k, uint32_t n, void* scratch);

/// Table row `row` (d elements) widened into dst bit for bit: integer moves
/// only, so a signalling NaN reaches the output as stored.
void embed_row(Storage s, const void* table, uint32_t row, uint32_t d,
               float* dst);

}  // namespace esp32_float
