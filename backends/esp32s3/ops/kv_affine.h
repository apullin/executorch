/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The kernels ExecuTorch's stock quantised K/V cache needs outside the
// delegate (_passes/stock_llm.py):
//
//   espressif::quantize_per_token_xnn.out(Tensor x, *, Tensor(a!) codes,
//       Tensor(b!) scales, Tensor(c!) zero_points) -> (Tensor(a!), ...)
//     QuantizedKVCache's writer: per row of x[..., D], XNNPACK's affine
//     chooser (A4) and quantize_per_token, as torch computes them --
//       lo = min(amin, 0), hi = max(amax, 0)
//       s  = max(fl(fl(hi - lo) / 255), FLT_EPSILON)
//       dmin = fl(lo / s), dmax = fl(hi / s)
//       z  = fl(-128 - dmin) if fl(fl(-128 + dmin) + fl(127 + dmax)) > 0
//            else fl(127 - dmax), then clamped to [-128, 127] and rounded
//            half to even (all float32)
//       q  = clamp(rne(fl64(fl64(x * fl64(1 / s)) + z)), -128, 127)
//     -- the quantise in float64, which the S3 runs in software: once per
//     element of each token, head and K or V row written.
//   espressif::dequantize_per_token_i8.out(Tensor codes, Tensor scales,
//       Tensor zero_points, *, Tensor(a!) out) -> Tensor(a!)
//     the cache read: fl((float)(q - z) * s), one rounding of the exact
//     product, which is dequantize_per_token's float64 product rounded once.
//   llama::update_cache.out(Tensor value, Tensor(a!) cache, SymInt start_pos,
//       *, Tensor(b!) out) -> Tensor(b!)
//     ExecuTorch's cache write (extension/llm/custom_ops/op_update_cache.cpp,
//     start_pos form): value[B, S, ...] copied into cache[b, start_pos:]. A
//     malformed call fails the instruction where ExecuTorch's kernel aborts.
//
// Built -ffp-contract=off on the host and the device (kv_affine.cpp), so the
// two give the same bits, and those of mirrors.kcache_write and
// vcache_dequant.

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace kv_affine {

/// One row of n floats: its codes, scale and zero point (the writer's
/// arithmetic, which the shim runs per row and the conformance harness
/// drives directly).
void quantize_row(const float* x, size_t n, int8_t* codes, float* scale,
                  int8_t* zero_point);

/// One row of n codes back to floats.
void dequantize_row(const int8_t* codes, size_t n, float scale,
                    int8_t zero_point, float* out);

// The kCustomKernels entries.
void shim_quantize(executorch::runtime::KernelRuntimeContext& ctx,
                   executorch::runtime::Span<executorch::runtime::EValue*> st);
void shim_dequantize(executorch::runtime::KernelRuntimeContext& ctx,
                     executorch::runtime::Span<executorch::runtime::EValue*> st);
void shim_update_cache(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> st);
// aten::_local_scalar_dense(Tensor self) -> Scalar: the stock cache's
// input_pos[0].item(), as ExecuTorch's prim op computes it (a tensor's first
// value as a Scalar). Registered on the device only: the firmware compiles no
// prim ops, while the host links ExecuTorch's register_prim_ops.cpp, whose
// kernel this would duplicate.
void shim_local_scalar_dense(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> st);

struct Stats {
  uint64_t quantized_rows;
  uint64_t dequantized_rows;
  uint64_t cache_writes;
  uint64_t refusals;
};
Stats stats();

/// One "kv_affine dispatch:" line, printed only once one of the ops has run
/// or refused: the story firmware runs none, so its log prints no such
/// line.
void report_dispatch();

}  // namespace kv_affine
}  // namespace espressif
