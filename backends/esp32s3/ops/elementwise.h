/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Exact fast loops for the element-wise operators a quantised CNN runs
// between its delegates (MobileNetV3's squeeze-excite and hard activations,
// the stride-2 pads of MobileNet-v1, ResNet-8 and DS-CNN):
//
//   aten::mul.out    a contiguous fp32 tensor times one value per leading
//                    index, broadcast over the trailing axes: the
//                    squeeze-excite scale, [1,C,H,W] x [1,C,1,1], in either
//                    operand order
//   aten::constant_pad_nd.out  int8 or fp32, padded on its last two axes
//                    only, no crop: MobileNet-v1's and the CNNs' stride-2
//                    pads (the portable kernel recurses per row and costs
//                    ~300 cycles a row on the S3; this one copies rows)
//   espressif::quant_lut.out, quant_lut_codes.out (quant_lut.h): their loops
//
// The portable mul.out takes a broadcast through a multidimensional index per
// element, and espressif_ops.cpp's own broadcast path through an odometer
// that calls Tensor::size() per element: 64-69 cycles an element on the S3
// for one multiply. The loops here keep one value in a register for a whole
// plane and step four elements at a time, the lanes written out step by step
// so independent operations cover the FPU's and the loads' latency. After
// that they run at about PSRAM's bandwidth (~2.4 cycles a byte, a written
// byte counted twice). Each gives the bytes its operator's existing kernel
// gives for every input it takes, and leaves every other input to that
// kernel:
//
//   mul    one IEEE multiply per element, as the portable kernel; every scale
//          must be a number, because only a NaN times a NaN can tell the
//          operand order (which payload survives), and compilers order a
//          commutative multiply as they like
//   quant_lut, quant_lut_codes: the same per-element steps as quant_lut.cpp's
//          loops (quant_lut.cpp says why the reordered clamps are exact)
//   pad    the same bytes: every input element copied, every other output
//          element the value the portable kernel casts (its own cast), as
//          bits
//
// The mul path is registered only in firmware built with it (gen_kernels.py
// --elementwise-fast writes ESP32_GENERATED_EW_MUL into Esp32KernelVariant.h
// for programs that name aten::mul.out): the story firmware names it too, and
// its object code must not change. (The residual add needs no loop of its
// own: espressif_ops.cpp's same-shape add already runs at PSRAM's bandwidth,
// 38 cycles an element for its 16 bytes of traffic.) The pad path replaces
// the portable kernel's registration as qdq_pool's overrides do
// (ESP32_GENERATED_EW_CONSTANT_PAD, again only with --elementwise-fast). The
// table loops serve quant_lut.cpp wherever a program names a table op.

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace elementwise {

// out[o * inner + j] = scale[o] * x[o * inner + j], for o < outer, j < inner.
// Every scale a number (the caller checks). out may be x itself; it must not
// otherwise overlap x or scale.
void mul_channels_f32(const float* scale, const float* x, float* out,
                      size_t outer, size_t inner);

// quant_lut.cpp's loop (quant_lut.h): out[i] = table[code(x[i]) - quant_min]
// copied as 32 bits, with code(x) = clamp(rne(quant_scaled(fl(x * inv))) +
// zero_point, quant_min, quant_max). The shim's conditions hold: inv finite
// and positive, quant_min <= zero_point <= quant_max, the range at most 256
// codes. out is x itself or apart from it.
void quant_lut_f32(const float* x, float* out, size_t n, const float* table,
                   float inv, int32_t zero_point, int32_t quant_min,
                   int32_t quant_max);

// quant_lut_codes' loop over the whole int8 range (quant_min -128, quant_max
// 127, so no clamp): out[i] = table[x[i] + 128]. out is x itself or apart
// from it.
void quant_lut_codes_i8(const int8_t* x, int8_t* out, size_t n,
                        const int8_t* table);

// The registered mul shim's fast path, in the schema's argument order (out
// last). Returns false, having written nothing, for a call outside it;
// espressif_ops.cpp's shim_mul_out then takes it.
bool try_mul_out(executorch::runtime::KernelRuntimeContext& ctx,
                 executorch::runtime::Span<executorch::runtime::EValue*> stack);

// The registered aten::constant_pad_nd.out: the row copy where it applies,
// else the portable kernel (which then is linked as its fallback).
void shim_constant_pad_nd_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> stack);

// The calls each loop took. Plain increments, as qdq_pool's (qdq_pool.h says
// why): exact while one Method runs these operators at a time. The table
// loops count their own calls, so a check that drives quant_lut::apply puts
// the counts back afterwards (quant_lut_check.cpp).
struct Counters {
  uint64_t mul_channels;  // calls taken by the channel-scale loop
  uint64_t lut_f32;       // quant_lut_f32 calls
  uint64_t lut_codes;     // quant_lut_codes_i8 calls
  uint64_t pad_rows;      // constant_pad_nd calls the row copy took
  uint64_t pad_fallback;  // constant_pad_nd calls left to the portable kernel
};
Counters& counters();

// The A/B switch (the host runners' --no-ew-fast): off sends every call to
// the kernel each loop replaces, which computes the same bits. On by default.
void set_enabled(bool enabled);
bool enabled();

// One "<op> elementwise dispatch:" line per operator whose loop ran; nothing
// otherwise, so a firmware whose programs never reach these loops prints
// none of these lines.
void report_dispatch();

}  // namespace elementwise
}  // namespace espressif
