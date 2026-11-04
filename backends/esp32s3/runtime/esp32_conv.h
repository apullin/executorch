/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The static int8 convolution path: an affine int8 conv (esp32_asym.h) whose
// activation codes are a pure function of each input element -- Stage B's
// codes, or fp32 under a static or per-tensor scheme -- computed without the
// per-patch gather of the general loops.
//
// The general loops (esp32_backend.cpp's run_blocked_affine and
// run_depthwise_affine) build one output position's patch at a time: every
// tap's index through the full transposed/3-D/dilated mapping, its code
// through the record's quantiser, then one call per output element for the
// dot, the zero-point terms, the epilogue and the requantise. On a 3x3
// depthwise conv that is ~67 host instructions a MAC (DS-CNN: 61% of the
// model). This path instead:
//
//   - makes each input element's code ONCE per op: Stage B's codes are read
//     where they are; fp32 under a static (or per-tensor) record is quantised
//     plane by plane with the record's own quantiser (code_of), the code the
//     gather would have made for that element at every tap that reads it;
//   - runs a depthwise conv directly on the channel's NCHW code plane, nine
//     multiply-adds an output for a 3x3, the interior without bounds checks;
//   - gathers a dense conv's rows (a pointwise conv's: a strided transpose)
//     from the code plane into the row block, and runs the columns over the
//     block with the int8 dot (dot_i8_s3 under PIE), writing each output
//     channel's positions contiguously, so nothing is staged or scattered.
//
// Bit for bit the general loops' result. The integer part is exact: the raw
// dot and acc = raw - zx * S'[n] wrap in uint32 exactly as dot_codes and
// less_zx_sums do, and modular sums do not depend on their order, so a tap off
// the input (code zx) may be summed directly. The float part is the affine
// column math with one scale group, operation for operation:
//
//   y = fl(b + fl(fl(fl(acc) * sx) * s[n]))      (b = 0 without a bias)
//   y = clamp(y, lo, hi)                         (FLAG_CLAMP, torch.clamp's)
//   codes out: clamp(rne(fl(y * fl(1/so))) + zo)  (requantize_code)
//   fp32 out:  fl((yq - zo) * so)                 (requantize), or y as is
//
// in a unit built -ffp-contract=off (as esp32_asym.cpp), with the quantisers
// of esp32_codes.h. What it does not take (esp32_backend.cpp's
// conv_fast_op decides, and the general loops keep): dynamic per-row
// activations (their scale depends on the patch), ops without an activation
// record (the backend's own rows), transposed convs, 3-D convs (but one
// whose depth is one plane, one tap and no padding: the 2-D conv of that
// plane), int4/int16 weights, weight zero points and float offsets, scale
// groups, fused activations and residual operands.
#pragma once

#include <cstddef>
#include <cstdint>

namespace esp32_conv {

/// One 2-D convolution (depth axes 1) as this path reads it: the gather's
/// geometry for ONE group (c_in its input channels; a depthwise op's is 1)
/// and the descriptor's extents.
struct Op {
  uint32_t h_in, w_in, c_in;
  uint32_t h_out, w_out;
  uint32_t kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
  // Every output channel reads its own input channel (groups == channels).
  bool depthwise;
  uint32_t m, k, n, k_pad;   // m = h_out * w_out, k = c_in * kh * kw
  bool clamp;
  float clamp_lo, clamp_hi;
};

/// How the op's activation codes are made.
struct Act {
  // Stage B: the model's codes (NCHW, this group's channels), or null.
  const int8_t* x8;
  // fp32 input (NCHW, this group's channels) when x8 is null.
  const float* x;
  // The record's quantiser for fp32: code_of (rne before zx, [qmin, qmax])
  // at fl(1 / sx), as quantize_conv_row computes it.
  int32_t qmin, qmax;
  // The record's scale and zero point (static, or chosen once over the
  // tensor); zx is also the code a tap off the input reads.
  float sx;
  int32_t zx;
};

/// The weight side: one scale group per row, no zero points or offsets.
struct Columns {
  const int8_t* wq;        // [n, k_pad] int8 codes, k_pad apart
  bool pie;                // rows and the row block 16-byte aligned: dot_i8_s3
  const void* scales;      // [n], fp32 or IEEE half
  bool scales_fp16;
  const void* sums;        // S'[n], int16 or int32, or null (zx needs none)
  bool sums_i16;
  const float* bias;       // [n], or null
};

/// Where the outputs go: NCHW, this group's channels (element n * m + r).
struct Out {
  float* y;                // fp32 out, or null
  int8_t* y8;              // Stage B codes out, or null
  bool requant;            // an output record: requantise (codes or fake-quant)
  float out_inv;           // its fl(1 / so)
  float out_scale;
  int32_t zo, qmin, qmax;
};

/// The per-call scratch this path may use (the handle's xq region, 16-byte
/// aligned): `rows` activation rows of k_pad for a dense conv's row block
/// (the backend's conv_fast_rows: one pass over the weights per block), then
/// a code plane when the op needs one (scratch_need).
struct Scratch {
  int8_t* base;
  uint32_t rows;
  // The op's codes made once before its parts (prepare): a dense conv's fp32
  // input quantised whole, which every part reads; null otherwise. A second
  // core's part has a scratch of its own (part_scratch_need) and this plane.
  const int8_t* plane;
};

/// The part of an op one core computes: output channels [lo, hi) of a
/// depthwise conv, output positions [lo, hi) of a dense one. Every output
/// element is computed whole, by the operations the whole op uses, on the
/// core whose part holds it, so the parts of a split op write exactly the
/// unsplit op's bytes. The backend runs every op whole, Part{0, units}, on
/// one core: its split across the cores is off (esp32_backend.cpp
/// kConvSplit; the board: README, Known limits).
struct Part {
  uint32_t lo, hi;
};

/// What one call did, for the backend's counters: dots under PIE and not,
/// output elements written, those the integer epilogue wrote (codes out,
/// esp32_conv.cpp), and of those, the ones it left to the float sequence.
struct Counts {
  uint32_t pie, scalar;
  uint64_t outputs;
  uint64_t fixed, fallback;
};

/// The largest fp32 plane (bytes of codes) a dense conv quantises whole; a
/// larger one quantises each tap as its row is gathered (still once per
/// element for a stride-1 pointwise conv, the common large case).
constexpr uint32_t kPlaneBudget = 32 * 1024;

/// Scratch bytes `op` needs at `rows` rows a block, `fp32_in` its input
/// being fp32 (a Stage B op reads its codes in place): the row block (dense
/// convs), then the code plane (a depthwise conv's one channel, a dense
/// conv's whole input within kPlaneBudget), 16-byte aligned. init reserves
/// at least this; execute assumes it.
uint64_t scratch_need(const Op& op, bool fp32_in, uint32_t rows);

/// Scratch bytes one part of `op` needs on a core of its own: its row block
/// (dense convs) or its padded channel plane (depthwise convs, none when
/// Stage B codes are read in place), not prepare's shared code plane.
uint64_t part_scratch_need(const Op& op, bool fp32_in, uint32_t rows);

/// The op's units, which a Part ranges over: its output channels (depthwise)
/// or its output positions (dense).
inline uint32_t units(const Op& op) {
  return op.depthwise ? op.n : op.m;
}

/// Where the op's units split between the cores: half, rounded to whole
/// groups of 16 channels for a depthwise op the PIE path takes.
uint32_t split_at(const Op& op);

/// Measures, once, which QACC lane holds each of the 16 channels for the PIE
/// depthwise path, and holds dw33_qacc16_s3 to the scalar sums and
/// qacc_lane_codes_s3 to the epilogue it implements; the path runs only after
/// it passed (it prints one "PIE depthwise 16-lane selftest" line). The
/// backend calls it when a blob has ops on this path.
bool pie_dw_init(void);

/// What an op does once, before its parts: a dense conv whose fp32 input is
/// quantised whole (kPlaneBudget) quantises it into the plane after the row
/// block. Returns the codes the parts read (Scratch::plane), or null.
const int8_t* prepare(const Op& op, const Act& act, const Scratch& scratch);

/// prepare() in parts: where the op's plane goes (null when it has none),
/// and elements [part.lo, part.hi) of its fp32 input quantised into it, each
/// element's code whatever the part (so two cores may share the work).
int8_t* plane_of(const Op& op, const Act& act, const Scratch& scratch);
inline uint32_t plane_elements(const Op& op) {
  return op.c_in * op.h_in * op.w_in;
}
void prepare_part(const Act& act, int8_t* plane, Part part);

/// Computes part `part` of the op, after prepare (`scratch.plane` its
/// result). `op`, `act`, `cols` and `out` describe a conv the backend has
/// decided this path takes (esp32_backend.cpp conv_fast_op).
void run_part(const Op& op, const Act& act, const Columns& cols,
              const Out& out, const Scratch& scratch, Part part,
              Counts* counts);

/// Computes the whole op on one core: prepare, then run_part over all of it.
void run(const Op& op, const Act& act, const Columns& cols, const Out& out,
         const Scratch& scratch, Counts* counts);

#if defined(ESP32_HOST_BUILD)
/// Host only: calls since start-up into the dense convs' fp32 fast paths --
/// quantize_span_fast (prepare's plane), gather_pointwise_fast (stride-1
/// pointwise rows) and gather_dense_fast (other dense rows) -- so a test can
/// tell their routes were taken: their codes are code_of's, so the bits alone
/// cannot.
struct QuantFastCalls {
  uint64_t span, pointwise, dense;
};
QuantFastCalls quant_fast_calls();
#endif

/// The bench image's epilogue micro-benchmark (esp32_backend_microbench):
/// the cycles an output of the float epilogue and of the integer one, each
/// over `reps` runs of kRun synthetic accumulators of one DS-CNN-like
/// channel, and the outputs on which the two disagree (0).
struct EpilogueBench {
  uint32_t cycles_float, cycles_fixed;  // per 1,000 outputs
  uint32_t fallbacks, mismatches;
};
EpilogueBench bench_epilogue(uint32_t (*cycles)(void), uint32_t reps);

}  // namespace esp32_conv
