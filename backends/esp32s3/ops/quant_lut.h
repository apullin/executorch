/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// espressif::quant_lut: an activation between two static quantisers, as a
// table lookup (_passes/fold_quant_lut_pass.py builds the table at export).
//
//   espressif::quant_lut.out(Tensor x, Tensor table, float inv_scale,
//                            int zero_point, int quant_min, int quant_max, *,
//                            Tensor(a!) out) -> Tensor(a!)
//
// Per element: v = fl(x * inv_scale), NaN -> 0, clamped to [-512, 512];
// c = clamp(rne(v) + zero_point, quant_min, quant_max); out = table[c -
// quant_min], copied as 32 bits, so denormals and NaN payloads survive. That
// is quantize_per_tensor's code for every x -- for NaN, the zero point, as
// ExecuTorch 1.6's portable kernel defines it (quant_rules.h) -- and the same
// bits on the host and the device: quant_lut.cpp builds with
// -ffp-contract=off.

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace quant_lut {

// The op's arithmetic over n elements (the shim's loop, and what the
// conformance check drives directly).
void apply(const float* x, float* out, size_t n, const float* table,
           float inv_scale, int32_t zero_point, int32_t quant_min,
           int32_t quant_max);

// The kCustomKernels entry for espressif::quant_lut.out. Every malformed
// argument fails the instruction (InvalidArgument; a wrong argument count is
// InvalidProgram) before out is written.
void shim(executorch::runtime::KernelRuntimeContext& ctx,
          executorch::runtime::Span<executorch::runtime::EValue*> stack);

struct Stats {
  uint64_t calls;
  uint64_t elements;
  uint64_t refusals;
};
Stats stats();
// For checks that drive the shim themselves and must leave a run's counts as
// its programs made them.
void restore_stats(const Stats& saved);

// One "quant_lut dispatch:" line, printed only once the op has run or refused:
// the story firmware never runs a table, so its log prints no such line.
// The same for the code-to-code variant
// below ("quant_lut_codes dispatch:").
void report_dispatch();

// espressif::quant_lut_codes: the code-to-code variant, Stage B's
// (stage_b.py): int8 codes in, int8 codes out, a table of at most 256 int8
// codes (Stage A's next code for each entry).
//
//   espressif::quant_lut_codes.out(Tensor x, Tensor table, int quant_min,
//                                  int quant_max, *, Tensor(a!) out)
//
// Per element: out = table[clamp(x, quant_min, quant_max) - quant_min].
// Integers only.
void apply_codes(const int8_t* x, int8_t* out, size_t n, const int8_t* table,
                 int32_t quant_min, int32_t quant_max);

// The kCustomKernels entry for espressif::quant_lut_codes.out; malformed
// arguments fail the instruction as shim's do, before out is written.
void codes_shim(executorch::runtime::KernelRuntimeContext& ctx,
                executorch::runtime::Span<executorch::runtime::EValue*> stack);

Stats codes_stats();

// The quantize step against two independent references -- a nearbyintf
// implementation and torch's own quantize_per_tensor (quant_lut_golden.h) --
// on every rounding tie of k + 1/2 for k in [-520, 520] and its +-1, +-2 ulp
// neighbours, the special values, five scales and the three code ranges; the
// output bits against the table; and one case per malformed argument. Prints
// one "quant_lut check:" line; the counters are restored afterwards. Defined
// in quant_lut_check.cpp, which only test builds link.
void run_conformance_check();

}  // namespace quant_lut
}  // namespace espressif
