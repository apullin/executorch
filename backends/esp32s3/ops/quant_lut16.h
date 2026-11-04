/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// espressif::lut16: an activation between two static int16 quantisers, as a
// fitted table (quantizer/lut_fit.py fits it at lowering time;
// _passes/fold_lut16_pass.py folds it into the graph).
//
//   espressif::lut16.out(Tensor x, Tensor table, float inv_scale,
//                        int zero_point, int quant_min, int quant_max,
//                        int scheme, int lo, int hi, int shift, int G,
//                        float out_scale, int out_zero_point,
//                        int out_quant_min, int out_quant_max, *,
//                        Tensor(a!) out) -> Tensor(a!)
//
// Per element (mirrors.lut16 is the numpy mirror, fold_lut16_pass.py's
// CompositeExplicitAutograd body the torch reference; the order is the
// contract):
//   1. c = clamp(rne(fl(x * inv_scale)) + zero_point, quant_min, quant_max),
//      with fl(x * inv_scale) clamped to [-2^16, 2^16] first (NaN -> 0):
//      quantize_per_tensor's code for every x, the zero point for NaN as
//      ExecuTorch 1.6's portable kernel defines it (quant_rules.h).
//   2. c = clamp(c, lo, hi), then the scheme's index:
//      uniform  u = c - lo; k = u >> F; t = u & (2^F - 1)        (F = shift)
//      NSAU     a = |c|; sh = max(floor(log2(a | 1)) - m, 0);    (m = shift)
//               q = a >> sh; k = (sh << m) + q; t = a - (q << sh);
//               k += the bias of c's sign (nsau_layout)
//      Two flags on the scheme word change this:
//      kOffset  NSAU (11, 12): each sign's half holds only the segments its
//               codes in [lo, hi] reach, from the segment of its first code
//               (nsau_layout), not from code 0 at the larger half's size
//      kTail    uniform (17, 18): c is not clamped to [lo, hi] (int16 only);
//               k = clamp(u >> F, 0, K - 1), t = u - (k << F), so past
//               [lo, hi] the edge segment goes on -- its line (linear), its
//               cubic (cubic: a line where the fitter uses the tail)
//   3. y = the table step, in int32 or float32 with one rounding per step:
//      direct          y = T[c - lo]
//      uniform linear  y = (V[k] * 2^F + (V[k+1] - V[k]) * t + 2^(F+G-1))
//                          >> (F + G)
//      NSAU linear     y = (V[k] + (((V[k+1] - V[k]) * t) >> sh) + 2^(G-1))
//                          >> G
//      cubic           tf = fl(t) * 2^-F (exact), Horner
//                      y = fl(c0 + fl(tf * fl(c1 + fl(tf * fl(c2 +
//                          fl(tf * c3)))))), then (y + 1.5 * 2^23) - 1.5 *
//                          2^23, round half to even
//      shifts are floors on two's complement int32.
//   4. y = clamp(y, out_quant_min, out_quant_max);
//      out = fl(fl(y - out_zero_point) * out_scale)
//
// Tables (the `table` tensor): direct int16 [hi - lo + 1]; uniform linear
// int32 [((hi - lo) >> F) + 2]; uniform cubic float32 [((hi - lo) >> F) + 1,
// 4] as (c0, c1, c2, c3); NSAU linear int32 [half + 1] per sign; NSAU cubic
// float32 [half, 4] per sign; both signs when lo < 0 (negative half after the
// positive one), half = nsau_half(lo, hi, m). With kTail the same as without;
// with kOffset each half holds segments k(first code) .. k(last code) of its
// sign, + 1 node when linear (nsau_layout().entries in all). Linear nodes are
// in units of 2^-G output codes. validate() refuses every table whose
// arithmetic could leave int32 or the exact range of the rounding (mirrors.
// lut16_linear_safe, cubic |c| <= 2^19, a tail's edge cubic below
// 2^22 out to the int16 codes: lut16_cubic_safe), so the kernel has no undefined
// behaviour and the host, the device and the mirrors agree bit for bit:
// quant_lut16.cpp builds with -ffp-contract=off, and on the device its
// uniform linear, uniform cubic and NSAU cubic loops are hand-scheduled
// (quant_lut16_s3.S).

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace lut16 {

enum Scheme : int32_t {
  kDirect = 0,
  kUniformLinear = 1,
  kUniformCubic = 2,
  kNsauLinear = 3,
  kNsauCubic = 4,
};

// The scheme word is a Scheme in its low three bits plus these flags
// (mirrors.LUT16_OFFSET/_TAIL): 11, 12, 17 and 18 are the valid
// flagged words. A runtime without a flag refuses its words (validate's
// "scheme", or the shim's "arguments": its dtype rule wants float32 for
// every scheme but 0, 1 and 3).
constexpr int32_t kBaseMask = 7;
constexpr int32_t kOffset = 8;
constexpr int32_t kTail = 16;

// One table, as the shim builds it from the op's arguments (or a fused
// kernel from its own record). `count` is the table's first dimension:
// entries (direct), nodes (linear) or segments of four floats (cubic).
struct Table {
  int32_t scheme;
  int32_t lo, hi;    // input codes are clamped to [lo, hi] first
  int32_t shift;     // uniform: F, log2 of the segment width; NSAU: m
  int32_t G;         // linear: node precision; ignored otherwise
  int32_t out_min, out_max;
  const void* data;  // int16 | int32 | float[4] rows
  int32_t count;
};

// Segments in one sign's half of an NSAU table over [lo, hi].
int32_t nsau_half(int32_t lo, int32_t hi, int32_t m);

// Where an NSAU table's codes read (mirrors.lut16_nsau_layout): a code
// of segment k reads entry k + pos_bias (c >= 0) or k + neg_bias (c < 0) --
// its row (cubic), its first node (linear) -- of `entries`. Plain: 0 and
// half (+ 1), both halves when lo < 0; kOffset: -k(max(lo, 0)) and P -
// k(max(1, -hi)), P the positive half's entries (0 when hi < 0).
struct NsauLayout {
  int32_t pos_bias, neg_bias, entries;
};
NsauLayout nsau_layout(int32_t scheme, int32_t lo, int32_t hi, int32_t m);

// nullptr when the table is well formed (scheme and flags, domain, shift, G,
// output range, count, the int32 bounds of linear nodes -- a tail's edge
// segments out to the int16 codes -- finite cubic coefficients within 2^19,
// a tail's edge cubics below 2^22), else what is wrong. Every other function
// here requires a table validate() accepted.
const char* validate(const Table& table);

// validate(), once per table. A program's tables are constants, so a table
// whose data pointer and arguments were accepted before is not read again
// (validate() reads every node or coefficient: on a short vector that costs
// more than the table saves). Returns nullptr with *use set to the table to
// evaluate, or what is wrong. On the device the first use also copies a
// table that lives outside internal RAM (in the program's flash or PSRAM)
// into the internal heap, within a budget; *use then reads the copy, which
// is what gets validated. A table that does not fit is read in place. The
// shim calls this, and so does qnorm for its tables.
const char* checked(const Table& table, Table* use);

// Drops every remembered table and frees the copies. Whoever releases the
// memory a program's tables live in calls this (main.cpp's run_model, after
// each program; the conformance check, around each case that edits a table
// in place): a different table could later sit at the same address.
void forget_tables();

// The output code of one input code (any int32: it is clamped to [lo, hi],
// or with kTail to int16). The plain C reference, on every target.
int32_t eval_code(const Table& table, int32_t code);

// Code to code over n elements: out[i] = eval_code(table, in[i]). in and out
// may alias. On the device the uniform linear, uniform cubic (with and
// without kTail) and NSAU cubic (with and without kOffset) schemes run
// hand-scheduled loops.
void eval_codes(const Table& table, const int16_t* in, int16_t* out, size_t n);

// The codes q1 can produce, clamped to the table's domain: [clamp(quant_min,
// lo, hi), clamp(quant_max, lo, hi)], with kTail [quant_min, quant_max] in
// int16. A producer that clamps its codes to it can call
// eval_codes_in_domain, which skips the clamp.
void domain(const Table& table, int32_t quant_min, int32_t quant_max,
            int32_t* low, int32_t* high);
void eval_codes_in_domain(const Table& table, const int16_t* in, int16_t* out,
                          size_t n);

// The op's arithmetic over n floats (steps 1-4 above).
void apply(const Table& table, const float* x, float* out, size_t n,
           float inv_scale, int32_t zero_point, int32_t quant_min,
           int32_t quant_max, float out_scale, int32_t out_zero_point);

// The kCustomKernels entry for espressif::lut16.out. Every malformed argument
// fails the instruction (InvalidArgument; a wrong argument count is
// InvalidProgram) before out is written.
void shim(executorch::runtime::KernelRuntimeContext& ctx,
          executorch::runtime::Span<executorch::runtime::EValue*> stack);

struct Stats {
  uint64_t calls;
  uint64_t elements;
  uint64_t refusals;
  // checked(): validate() runs, calls answered by a remembered table, and
  // the tables copied into internal RAM (with their bytes) or read in place
  uint64_t validations;
  uint64_t reused;
  uint64_t copied;
  uint64_t copied_bytes;
  uint64_t in_place;
};
Stats stats();
void restore_stats(const Stats& saved);

// One "lut16 dispatch:" line, printed only once the op has run or refused:
// the story firmware never runs a table, so its log prints no such line.
// Then one "lut16 tables:" line once any table has gone through checked()
// (the op's or qnorm's).
void report_dispatch();

// Every scheme on all 65,536 int16 codes (the device loops and the C
// reference), the op's quantize and dequantize steps on rounding ties near 0,
// +-2^15 and the +-2^16 rails, the special values and a strided sweep, all
// against the mirrors' hashes (quant_lut16_golden.h, quant_lut16_golden.py),
// and one case per malformed argument through the shim. Prints one "lut16
// check:" line; the counters are restored afterwards. Defined in
// quant_lut16_check.cpp, which only test builds link.
void run_conformance_check();

}  // namespace lut16
}  // namespace espressif
