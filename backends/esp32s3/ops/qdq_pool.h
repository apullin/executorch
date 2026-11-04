/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Exact fast paths for the portable operators a quantised CNN leaves between
// its delegates: ExecuTorch's per-tensor quantize and dequantize
// (quantized_decomposed::quantize_per_tensor.out and dequantize_per_tensor.out)
// and aten::avg_pool2d.out. Each override gives the portable kernel's bytes
// for every input it takes and hands every other input to that kernel.
//
// On the S3 the portable quantize costs about 350 cycles an element at -O2:
// per element it converts the zero point to double, calls nearbyintf and
// clamps in soft float64. The fast loop does the same arithmetic in about a
// dozen instructions (qdq_pool.cpp says why it is exact). The pool keeps the
// portable kernel's summation order and division and drops its per-element
// index arithmetic.
//
// A firmware registers an override only when its program's operator list
// names the operator (gen_kernels.py writes ESP32_GENERATED_FAST_QUANTIZE,
// _DEQUANTIZE and _AVG_POOL2D into Esp32KernelVariant.h), because the
// portable kernel must then be linked as the fallback. A program without
// these operators (the story's) compiles none of this.
#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace qdq_pool {

// The override shims, in the schema's argument order (out last).
void shim_quantize_per_tensor_out(executorch::runtime::KernelRuntimeContext& ctx,
                                  executorch::runtime::Span<executorch::runtime::EValue*> stack);
void shim_dequantize_per_tensor_out(executorch::runtime::KernelRuntimeContext& ctx,
                                    executorch::runtime::Span<executorch::runtime::EValue*> stack);
void shim_avg_pool2d_out(executorch::runtime::KernelRuntimeContext& ctx,
                         executorch::runtime::Span<executorch::runtime::EValue*> stack);

// The element loops, for the checks. quantize: n float32 values x to codes of
// `bytes` (1 or 2) bytes, signed or not, at inv = 1 / float(scale) as the
// portable kernel rounds it, the range within the code type and
// quant_min <= zero_point <= quant_max (the shim's conditions). dequantize:
// n codes to float32, at float(scale). Input and output must not overlap
// (the shims send an overlapping call to the portable kernel).
void quantize_f32(const float* x, void* out, size_t n, size_t bytes, bool is_signed,
                  float inv, int32_t zero_point, int32_t quant_min, int32_t quant_max);
void dequantize_f32(const void* codes, size_t bytes, bool is_signed, float* out, size_t n,
                    int32_t zero_point, float scale);

// What each call took: the fast loop, or the portable kernel. Plain
// increments, as espressif_ops.cpp's overrides count (all but the RMSNorm
// pair): the executor runs these operators on its own task and never splits
// one across cores, so the counts are exact while one Method runs them at a
// time. Two Methods running them at once on two tasks race on these words (a
// C++ data race; in practice a lost count). Only report_dispatch() and the
// fast-path check read them, and no output depends on them. A relaxed atomic
// add is not free here: with -mdisable-hardware-atomics it is a call to
// __atomic_fetch_add_4 (two windowed calls, a memw and an s32c1i, 20
// instructions more than the add), roughly 0.2 us on calls of 5-11 us.
struct Counters {
  uint64_t quantize_fast;
  uint64_t quantize_fallback;
  uint64_t dequantize_fast;
  uint64_t dequantize_fallback;
  uint64_t pool_global;    // every window a whole plane: planes summed four at a time
  uint64_t pool_windowed;  // any other window
  uint64_t pool_fallback;
};
// The live counts: the fast-path check reads them around each case and puts
// them back when it is done.
Counters& counters();

// The A/B switch: off sends every call to the portable kernel (counted as a
// fallback). On by default.
void set_enabled(bool enabled);
bool enabled();

// One "<op> dispatch:" line per operator that ran; nothing for one that did
// not, so a program without these operators prints what it printed before.
void report_dispatch();

}  // namespace qdq_pool
}  // namespace espressif
