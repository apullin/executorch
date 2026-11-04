/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/error.h>

namespace espressif {

// Explicit registration keeps the kernels linked from an ESP-IDF archive.
// The portable registry must omit the overridden aten entries (gen_kernels.py).
executorch::runtime::Error register_ops();
size_t kernel_count();

// Read-only snapshot for runner diagnostics; kernels keep their own counters.
struct DispatchStats {
  uint64_t add_same;
  uint64_t add_leading_broadcast;
  uint64_t add_fallback;
  uint64_t permute_adjacent_block;
  uint64_t permute_adjacent_element;
  uint64_t permute_fallback;
  uint64_t bmm_p_le_n;
  uint64_t bmm_p_gt_n;
  uint64_t bmm_fallback;
  uint64_t mul_same;
  uint64_t mul_broadcast;
  uint64_t mul_fallback;
  uint64_t mul_scalar;
  uint64_t where_mask_scalar;
  uint64_t where_fallback;
  uint64_t div_scalar_denominator;
  uint64_t div_fallback;
  uint64_t le_scalar;
  uint64_t le_fallback;
  uint64_t index_put_rows;
  uint64_t index_put_fallback;
  uint64_t split_chunks;
  uint64_t split_fallback;
  uint64_t remainder_non_negative;
  uint64_t remainder_fallback;
  uint64_t softmax_last_dim;
  uint64_t softmax_fallback;
  int rms_asm;
  int rms_scalar;
  int rope_rows;
  uint64_t ln2d_cols;
  uint64_t cached_attention_calls;
  uint64_t cached_attention_positions;
  uint64_t ple_prepare_calls;
  uint64_t ple_prepare_values;
  uint64_t attn_tail_calls;
  uint64_t attn_tail_rows;
  uint64_t quantize_rows_calls;
  uint64_t quantize_rows_rows;
  uint64_t fast_layer_norm_calls;
  uint64_t fast_layer_norm_rows;
  uint64_t fast_layer_norm_fallback;
};

DispatchStats dispatch_stats();

// Prints the standard-operator overrides' counters, one "<op> dispatch:" line
// per operator (the format run_parity.py parses): calls per fast path, and the
// calls that fell through to the portable kernel. Every runner prints the same
// lines, so a board, QEMU or host log can show which path a program took.
void report_override_dispatch();

// Device self-checks the test runner drives: the RMSNorm assembly's domain
// refusals with their scalar fallback, and layer_norm2d's refusals.
void run_rmsnorm_domain_check();
void run_ln2d_refusal_check();
// The standard-op overrides against ExecuTorch's portable kernels, byte for
// byte, on story shapes, special values and near misses that must fall back;
// prints one "fastpath check:" line per operator and a total.
void run_portable_fastpath_check();
// The opt-in fast LayerNorm's row arithmetic (ESP32_FAST_LAYERNORM), built in
// every variant, against the portable kernel's: the same mean and rstd bits,
// outputs within the reciprocal multiply's rounding. Prints one
// "fast_layer_norm check:" line.
void run_fast_layer_norm_check();

}  // namespace espressif

// Select the existing scalar RMSNorm fallback for QEMU or hardware A/B checks.
extern "C" void rmsnorm_set_force_scalar(int on);
