/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ExecuTorch 1.6's rule for quantising a value that is not finite
// (kernels/quantized/cpu/op_quantize.cpp, quantize_val_with_inv_scale;
// pytorch/executorch#22811):
//
//   NaN   the zero point, clamped to [quant_min, quant_max]
//   +inf  quant_max
//   -inf  quant_min
//
// with every value clamped before it is converted to an integer, so no
// conversion is undefined. Every quantiser here whose codes have a portable
// reference follows it:
//
//   esp32_asym.cpp    the affine records' static and dynamic activations,
//                     int16 activations and output requantisation (code_of),
//                     and its copy of the delegate's ties-to-even rounding
//   esp32_backend.cpp round_half_even_i: PT2E's folded per-tensor
//                     activations (kFlagActPerTensor)
//   kv_affine.cpp     the stock quantised K/V cache's writer
//   quant_lut.cpp,    the table ops' input quantisers, and the fused norm's
//   quant_lut16.cpp,  quantize/dequantize pairs, 8- and 16-bit
//   qnorm.cpp
//
// Each scales first, v = fl(x * inv), and rounds quant_scaled(v): a NaN
// becomes 0, which rounds to 0 and adds nothing to the zero point. The
// quantiser's own clamps (a float rail before the rounding, the code range
// after it) put the infinities and every finite value past the range on
// quant_min and quant_max. The choosers (a dynamic scale and zero
// point) have no portable kernel to follow: their min/max comparisons skip a
// NaN, and an infinity gives an infinite scale, as in torch. The backend's
// own rows (A0: an absmax scale, rounded half away from zero) have no
// portable reference either; they keep their arithmetic, NaN out of contract.
//
// test/runtime/nonfinite_check.cpp holds every one of those quantisers to the
// rule (run_check below); the parity run calls it on the device, --selftest
// on the host.
#pragma once

#include <cstdint>

/// v = fl(x * inv) as a quantiser rounds it: 0 for NaN, so the code is the
/// zero point; every other value, the infinities included, unchanged for the
/// caller's clamps. One compare and one branch or select (v == v fails only
/// for NaN), before any conversion to an integer.
static inline __attribute__((always_inline)) float quant_scaled(float v) {
  return v == v ? v : 0.0f;
}

/// The same, branch-free, for a hand-tuned loop that keeps a 0.0f in a
/// register (an empty asm pins it, as the loop's other constants). GCC turns
/// the select above into a branch around a move on the LX7, taken by every
/// finite element (it if-converts only a min or max): on the device the
/// select is written out instead, oeq.s then movf.s.
static inline __attribute__((always_inline)) float quant_scaled(float v,
                                                                float zero) {
#if defined(__XTENSA__)
  __asm__("oeq.s b0, %0, %0\n\tmovf.s %0, %1, b0" : "+f"(v) : "f"(zero) : "b0");
  return v;
#else
  return v == v ? v : zero;
#endif
}

/// The same for a float64 quantise (the K/V cache writer's).
static inline __attribute__((always_inline)) double quant_scaled(double v) {
  return v == v ? v : 0.0;
}

/// FLT_MIN, the smallest normal float: A0's smallest scale with a reciprocal.
constexpr float kA0MinScale = 0x1p-126f;

/// A0's inverse scale (the backend's own rows: s = fl(absmax / qmax), codes
/// fl(x * fl(1 / s)) rounded half away, clamped to +-qmax): fl(1 / s) for a
/// normal scale; 0 for a scale that underflowed to a subnormal or zero, so
/// every code of that row is 0 and the row reads as zeros (an absmax below
/// 127 * 2^-126, about 1.5e-36, at qmax 127). The scale itself is returned
/// unchanged. Without the cut a finite row would reach an unsafe conversion:
/// [FLT_MIN, 0] has s = 9.3e-41, fl(1 / s) = inf and fl(x * inv) = [inf,
/// NaN]. With it, 1 / s <= 2^126 and every
/// finite x converts within +-(qmax + 1/2), so no conversion is out of
/// range. A normal threshold rather than the overflow's own edge (2^-128)
/// keeps the rule the same on an FPU that flushes subnormals. The compare
/// costs one per row, none per
/// element. Every A0 quantiser (esp32_backend.cpp's rows, conv rows and
/// taps, esp32_codes.h's a0_inverse, the int8 K-cache writer) and every
/// mirror (quantizer/mirrors.py quantize_half_away, test/reference.py,
/// ops/kcache_i8.py, the story's dynamic_i8_qk) uses this rule.
static inline __attribute__((always_inline)) float a0_inverse_scale(float s) {
  return s >= kA0MinScale ? 1.0f / s : 0.0f;
}

namespace espressif {
namespace nonfinite {

/// A per-tensor quantize: x's code at a float32 scale, zero point and range.
using Reference = int32_t (*)(float x, float scale, int32_t zero_point,
                              int32_t quant_min, int32_t quant_max);

/// ExecuTorch 1.6's quantize_val, transcribed: the device's reference.
int32_t transcribed_code(float x, float scale, int32_t zero_point,
                         int32_t quant_min, int32_t quant_max);

/// Drives the affine records' quantisers, the backend's PT2E rows and the K/V
/// cache's writer through their own entry points on NaN, the infinities,
/// +-FLT_MAX and values far past the range (the table ops have their own
/// sweeps, specials included), and prints one "nonfinite check:" line.
/// `reference` is the portable kernel on the host (asym_conformance.cpp),
/// which the transcription is then held to as well; null on the device,
/// which uses transcribed_code. Returns the mismatches.
uint32_t run_check(Reference reference);

}  // namespace nonfinite
}  // namespace espressif
