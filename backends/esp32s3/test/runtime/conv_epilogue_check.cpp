/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The static int8 conv path's integer epilogue (runtime/esp32_conv.cpp)
// against the float sequence it replaces, on the host: for random and
// adversarial channels, the two must give the same code on every accumulator
// in [-amax, amax] (check_channel's bisection over the float sequence's
// intervals of one code), and on a share of them at every accumulator one by
// one (the brute force, which also tests check_channel's premise, that the
// float sequence is monotone). The loops themselves (finish_fixed against
// finish_float, codes and requantised fp32) are compared on runs of random
// accumulators with a wrapping zx * S'.
//
// The unit is included whole, so this sees its internal functions; it builds
// -ffp-contract=off, as the unit does everywhere.
//
//   esp32_conv_epilogue_check [--channels N] [--brute N] [--seed S]
//
// Prints one "conv epilogue check:" line; exits 1 on any mismatch.

#ifndef ESP32_CONV_EPI_CHECK
#define ESP32_CONV_EPI_CHECK 1
#endif
#define ESP32_CONV_EPI_CHECK_QUIET 1
#include "esp32_conv.cpp"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using esp32_conv::Act;
using esp32_conv::Op;
using esp32_conv::Out;

struct Case {
  float sx, s, b, out_inv, lo, hi;
  int32_t zo, qmin, qmax, zx;
  uint32_t k;
};

std::mt19937_64 g_rng;

double uniform(double a, double b) {
  return std::uniform_real_distribution<double>(a, b)(g_rng);
}
int64_t integer(int64_t a, int64_t b) {
  return std::uniform_int_distribution<int64_t>(a, b)(g_rng);
}
float log_uniform(double a, double b) {
  return (float)std::exp(uniform(std::log(a), std::log(b)));
}

/// IEEE half -> float of a random positive half (fp16-stored scales).
float half_scale() {
  for (;;) {
    const uint16_t h = (uint16_t)integer(0x0400, 0x5BFF);  // normal, < 256
    const float v = esp32_codes::half_to_float(h);
    if (v > 0.0f) {
      return v;
    }
  }
}

const uint32_t kLengths[] = {1,   2,   3,   9,    16,   25,   27,   40,
                             64,  144, 256, 288,  576,  960,  1152, 2048,
                             4096, 8192};

/// One random channel. `family` picks the kind: realistic scales, exact
/// binary scales (exact ties), a bias that cancels a large accumulator,
/// gains near the integer form's limits, narrow code ranges.
Case random_case(int family) {
  Case c;
  c.k = kLengths[integer(0, sizeof(kLengths) / sizeof(kLengths[0]) - 1)];
  c.qmin = -128;
  c.qmax = 127;
  if (integer(0, 7) == 0) {
    c.qmin = (int32_t)integer(-128, 0);
    c.qmax = (int32_t)integer(c.qmin, 127);
  } else if (integer(0, 7) == 0) {
    c.qmin = -127;
  }
  c.zo = (int32_t)integer(c.qmin, c.qmax);
  c.zx = (int32_t)integer(-128, 127);
  // Scales: activation, weight (sometimes an fp16 table's) and 1 / output.
  c.sx = log_uniform(1e-4, 1.0);
  c.s = integer(0, 3) == 0 ? half_scale() * 1e-3f : log_uniform(1e-5, 0.1);
  c.out_inv = 1.0f / log_uniform(1e-3, 10.0);
  float so = 1.0f / c.out_inv;
  c.b = 0.0f;
  switch (integer(0, 3)) {
    case 0:
      break;
    case 1:
      c.b = (float)(uniform(-1.0, 1.0) * so * 200.0);
      break;
    case 2:
      c.b = (float)(uniform(-1.0, 1.0) * so * 20.0);
      break;
    default:
      c.b = (float)(uniform(-1.0, 1.0) * so * 1e4);
      break;
  }
  switch (family) {
    case 1: {
      // Exact binary scales and a bias on a code or half-code: ties exactly
      // at rounding boundaries for many accumulators.
      c.sx = std::ldexp(1.0f, (int)integer(-12, 0));
      c.s = std::ldexp(1.0f, (int)integer(-12, -2));
      c.out_inv = std::ldexp(1.0f, (int)integer(0, 10));
      so = 1.0f / c.out_inv;
      c.b = (float)(integer(-200, 200) * 0.5) * so;
      break;
    }
    case 2: {
      // A large bias the accumulator cancels: y in range only for |acc|
      // near |c| / g.
      const float g = c.sx * c.s * c.out_inv;
      const double reach = (double)c.k * 128.0 * 128.0 * g;
      c.b = (float)(uniform(-1.0, 1.0) * reach * so);
      break;
    }
    case 3: {
      // The gain near its limits: g close to 1/4 (short reductions), or tiny.
      if (integer(0, 1) == 0) {
        c.k = (uint32_t)integer(1, 9);
        c.out_inv = (float)(uniform(0.05, 0.2499) / ((double)c.sx * c.s));
      } else {
        c.out_inv = (float)(uniform(1e-9, 1e-7) / ((double)c.sx * c.s));
      }
      break;
    }
    case 4: {
      // Bias and gain that put c + zo + 1/2 within a hair of an integer.
      const float g = c.sx * c.s * c.out_inv;
      (void)g;
      const double want = (double)integer(-40, 40) + (integer(0, 1) ? 1e-6 : -1e-6);
      c.b = (float)(want * so);
      break;
    }
    default:
      break;
  }
  // The clamp: none, ReLU, ReLU6, or random bounds.
  switch (integer(0, 4)) {
    case 0:
    case 1:
      c.lo = -INFINITY;
      c.hi = INFINITY;
      break;
    case 2:
      c.lo = 0.0f;
      c.hi = INFINITY;
      break;
    case 3:
      c.lo = 0.0f;
      c.hi = 6.0f;
      break;
    default: {
      const float a = (float)(uniform(-150.0, 150.0) * so);
      const float b = (float)(uniform(-150.0, 150.0) * so);
      c.lo = a < b ? a : b;
      c.hi = a < b ? b : a;
      break;
    }
  }
  return c;
}

/// The op terms run_part makes, for codes out.
esp32_conv::Epi epi_of(const Case& c) {
  esp32_conv::Epi e;
  e.sx = c.sx;
  e.lo = c.lo;
  e.hi = c.hi;
  e.out_inv = c.out_inv;
  e.out_scale = 1.0f / c.out_inv;
  e.zo = c.zo;
  e.qmin = c.qmin;
  e.qmax = c.qmax;
  Op op;
  std::memset(&op, 0, sizeof(op));
  op.k = c.k;
  Act act;
  std::memset(&act, 0, sizeof(act));
  act.zx = c.zx;
  Out out;
  std::memset(&out, 0, sizeof(out));
  int8_t sink = 0;
  out.y8 = &sink;
  esp32_conv::fixed_epi(op, act, out, &e);
  return e;
}

/// The channel as the path makes it, with a random S' (|S'| <= k * 128,
/// as the weights' sum can be).
esp32_conv::Channel channel_of(const Case& c, const esp32_conv::Epi& e,
                               int32_t* sum) {
  *sum = (int32_t)integer(-(int64_t)c.k * 128, (int64_t)c.k * 128);
  return esp32_conv::channel_of(c.s, c.b, c.zx, (uint32_t)*sum, e);
}

uint64_t g_brute_accs = 0, g_brute_failures = 0, g_loop_failures = 0;
uint64_t g_loop_elements = 0, g_slow = 0, g_slow_of = 0;

/// Every acc in [-amax, amax], one by one.
void brute(const esp32_conv::Channel& ch, const esp32_conv::Epi& e) {
  const int64_t amax = (int64_t)e.amax;
  for (int64_t a = -amax; a <= amax; ++a) {
    bool slow = false;
    const int32_t q =
        esp32_conv::fixed_code((int32_t)a, ch.fx, e, ch.s, ch.b, &slow);
    const int32_t want = esp32_conv::float_code((int32_t)a, ch.s, ch.b, e);
    ++g_brute_accs;
    g_slow += slow ? 1 : 0;
    ++g_slow_of;
    if (q != want) {
      if (g_brute_failures++ < 8) {
        std::printf("  brute mismatch: acc %" PRId64 " fixed %d float %d\n",
                    a, (int)q, (int)want);
      }
    }
  }
}

/// finish_fixed against finish_float on runs of accumulators, codes and
/// requantised fp32: raw = acc + zx * S', the dot the path hands them.
void loops(const Case& c, const esp32_conv::Epi& e) {
  int32_t sum = 0;
  const esp32_conv::Channel ch = channel_of(c, e, &sum);
  const int64_t zxs = (int64_t)c.zx * sum;
  const int64_t amax = (int64_t)e.amax;
  int32_t raw[esp32_conv::kRun];
  int8_t a8[esp32_conv::kRun], b8[esp32_conv::kRun];
  float af[esp32_conv::kRun], bf[esp32_conv::kRun];
  for (int rep = 0; rep < 4; ++rep) {
    const uint32_t n = (uint32_t)integer(1, esp32_conv::kRun);
    for (uint32_t t = 0; t < n; ++t) {
      // Mostly small accumulators (where real ones live), some anywhere.
      const int64_t a = integer(0, 3) == 0 ? integer(-amax, amax)
                                           : integer(-amax / 64 - 1, amax / 64 + 1);
      raw[t] = (int32_t)(a + zxs);
    }
    esp32_conv::finish_fixed<esp32_conv::kOutCodes>(raw, n, ch, e, nullptr, a8,
                                                    0);
    esp32_conv::finish_float<esp32_conv::kOutCodes>(raw, n, ch.zxs, ch.s, ch.b,
                                                    e, nullptr, b8, 0);
    esp32_conv::finish_fixed<esp32_conv::kOutRequant>(raw, n, ch, e, af,
                                                      nullptr, 0);
    esp32_conv::finish_float<esp32_conv::kOutRequant>(raw, n, ch.zxs, ch.s,
                                                      ch.b, e, bf, nullptr, 0);
    g_loop_elements += n;
    if (std::memcmp(a8, b8, n) != 0 || std::memcmp(af, bf, n * sizeof(float)) != 0) {
      ++g_loop_failures;
    }
  }
}

// ---- the whole path on exact ties --------------------------------------------
// The kernels note the elements whose integer code may differ from the float
// sequence's, near a rounding boundary, and the path redoes them after the
// loop (lane_codes, rows_codes, lane_fixed, finish_fixed). With a model's
// scales the integer code is almost always right there anyway, so a path that
// skipped the redo would still pass every model row. Power-of-two scales make
// exact ties: g = sx * s * out_inv = 1/8 puts y = acc / 8 on k + 1/2 for every
// acc = 4 (mod 8), where the integer form rounds up and the float sequence to
// even, so half of those take the redo's code. Codes in [-4, 4] keep y inside
// the codes' range. run() on a 3x3 depthwise layer (two 16-channel PIE groups,
// pad 1) and a 64 -> 32 pointwise one (the fused dot, k_pad 64), codes out and
// fp32 requantised, against the float sequence element by element; and
// each layer must have called the kernels that serve it, as many times as
// its shape asks (the C loops they replace give the same outputs, so only the calls
// show a closed gate).
uint64_t g_tie_outputs = 0, g_tie_ties = 0, g_tie_failures = 0;
// A layer that did not call the kernels serving it, or a boot check that
// refused the host's C kernel: the bits stay right, the kernels do not run.
uint64_t g_kernel_failures = 0;

void tie_layer(bool depthwise, bool requant) {
  constexpr uint32_t kC = 32;
  Op op;
  std::memset(&op, 0, sizeof(op));
  op.h_in = op.h_out = 25;
  op.w_in = op.w_out = 5;
  op.stride_h = op.stride_w = op.dil_h = op.dil_w = 1;
  op.depthwise = depthwise;
  op.c_in = depthwise ? 1 : 64;
  op.kh = op.kw = depthwise ? 3 : 1;
  op.pad_h = op.pad_w = depthwise ? 1 : 0;
  op.k = depthwise ? 9 : 64;
  op.k_pad = depthwise ? 16 : 64;
  op.n = kC;
  op.m = op.h_out * op.w_out;
  const uint32_t in_ch = depthwise ? kC : 64;
  const size_t plane = (size_t)op.h_in * op.w_in;
  std::vector<int8_t> x(in_ch * plane + 32);
  for (int8_t& v : x) {
    v = (int8_t)integer(-4, 4);
  }
  alignas(16) static int8_t wq[kC * 64];
  std::vector<float> scales(kC, 0.125f);
  std::vector<int32_t> sums(kC);
  for (uint32_t n = 0; n < kC; ++n) {
    int32_t sum = 0;
    for (uint32_t j = 0; j < op.k_pad; ++j) {
      const int8_t w = j < op.k ? (int8_t)integer(-4, 4) : 0;
      wq[n * op.k_pad + j] = w;
      sum += w;
    }
    sums[n] = sum;
  }
  Act act;
  std::memset(&act, 0, sizeof(act));
  act.x8 = x.data();
  act.qmin = -128;
  act.qmax = 127;
  act.sx = 1.0f;
  act.zx = 0;
  esp32_conv::Columns cols;
  std::memset(&cols, 0, sizeof(cols));
  cols.wq = wq;
  cols.pie = true;
  cols.scales = scales.data();
  cols.sums = sums.data();
  Out out;
  std::memset(&out, 0, sizeof(out));
  out.requant = true;
  out.out_inv = 1.0f;
  out.out_scale = 1.0f;
  out.zo = 0;
  out.qmin = -128;
  out.qmax = 127;
  std::vector<int8_t> y8(kC * op.m, 0x5A);
  std::vector<float> y(kC * op.m, -7.0f);
  if (requant) {
    out.y = y.data();
  } else {
    out.y8 = y8.data();
  }
  const uint32_t rows = depthwise ? 0 : 64;
  alignas(16) static int8_t scratch[32 * 1024];
  if (esp32_conv::scratch_need(op, false, rows) > sizeof(scratch)) {
    ++g_tie_failures;
    return;
  }
  esp32_conv::Scratch sc{scratch, rows, nullptr};
  esp32_conv::Counts counts{0, 0, 0, 0, 0};
  const Esp32ReferenceKernelCalls before = esp32_reference_kernel_calls();
  esp32_conv::run(op, act, cols, out, sc, &counts);
  const Esp32ReferenceKernelCalls after = esp32_reference_kernel_calls();
  // Which kernels served the layer. Their outputs equal the C loops', so
  // the comparison below cannot tell; the calls can, and must be exactly
  // these (a closed gate -- `fusable` false, codes out sent to lane_float,
  // the transpose's flag off -- makes one of them 0):
  //   depthwise, codes out: qacc_lane_codes_s3 once per channel per run of
  //     kPieRun (32) pixels: 32 channels x 4 runs (125 pixels) = 128. fp32
  //     out takes lane_fixed, in C: 0.
  //   depthwise: transpose16_s3 for each interior row of each group's strip
  //     whose tiles stay inside the input -- all 25 rows of group 0, the
  //     first 19 of group 1 (the last plane's end: 16*125 + 15*125 + 5*ih +
  //     32 <= 32*125 for ih <= 18) -- and group 0's weight pack (the last
  //     group's packs element by element): 25 + 19 + 1 = 45.
  //   pointwise: dot_rows_codes_s3 once per channel per row block (64 and
  //     61 rows, each within kRun): 32 x 2 = 64, codes out or fp32.
  //   pointwise: transpose16_s3 per 16 input channels per row block while
  //     the tiles stay inside the input -- 4 groups of the first block, 3 of
  //     the second (48*125 + 64 + 15*125 + 80 > 64*125) -- 7.
  const uint64_t lane = after.qacc_lane_codes - before.qacc_lane_codes;
  const uint64_t fused = after.dot_rows_codes - before.dot_rows_codes;
  const uint64_t moves = after.transpose16 - before.transpose16;
  const uint64_t want_lane = depthwise && !requant ? 128 : 0;
  const uint64_t want_rows = depthwise ? 0 : 64;
  const uint64_t want_moves = depthwise ? 45 : 7;
  if (lane != want_lane || fused != want_rows || moves != want_moves) {
    std::printf("  kernels: %s %s called qacc_lane_codes %llu (want %llu), "
                "dot_rows_codes %llu (want %llu), transpose16 %llu (want %llu)\n",
                depthwise ? "depthwise" : "pointwise",
                requant ? "fp32" : "codes", (unsigned long long)lane,
                (unsigned long long)want_lane, (unsigned long long)fused,
                (unsigned long long)want_rows, (unsigned long long)moves,
                (unsigned long long)want_moves);
    ++g_kernel_failures;
  }
  esp32_conv::Epi e;
  e.sx = act.sx;
  e.lo = -INFINITY;
  e.hi = INFINITY;
  e.out_inv = out.out_inv;
  e.out_scale = out.out_scale;
  e.zo = out.zo;
  e.qmin = out.qmin;
  e.qmax = out.qmax;
  esp32_conv::fixed_epi(op, act, out, &e);
  for (uint32_t n = 0; n < kC; ++n) {
    for (uint32_t r = 0; r < op.m; ++r) {
      const int32_t oh = (int32_t)(r / op.w_out), ow = (int32_t)(r % op.w_out);
      int32_t acc = 0;
      if (depthwise) {
        for (int32_t i = 0; i < 3; ++i) {
          for (int32_t j = 0; j < 3; ++j) {
            const int32_t ih = oh + i - 1, iw = ow + j - 1;
            if (ih >= 0 && ih < (int32_t)op.h_in && iw >= 0 && iw < (int32_t)op.w_in) {
              acc += x[n * plane + ih * op.w_in + iw] * wq[n * op.k_pad + i * 3 + j];
            }
          }
        }
      } else {
        for (uint32_t ci = 0; ci < 64; ++ci) {
          acc += x[ci * plane + r] * wq[n * op.k_pad + ci];
        }
      }
      const int32_t q = esp32_conv::float_code(acc, 0.125f, 0.0f, e);
      const size_t at = (size_t)n * op.m + r;
      const bool ok = requant ? y[at] == (float)(q - e.zo) * e.out_scale
                              : y8[at] == (int8_t)q;
      ++g_tie_outputs;
      g_tie_ties += (acc % 8 == 4 || acc % 8 == -4) ? 1 : 0;
      if (!ok && g_tie_failures++ < 8) {
        std::printf("  tie mismatch: %s %s channel %u position %u acc %d want %d\n",
                    depthwise ? "depthwise" : "pointwise",
                    requant ? "fp32" : "codes", (unsigned)n, (unsigned)r,
                    (int)acc, (int)q);
      }
    }
  }
}

// The plane path (dw5_plane_s3, dw3_plane_s3) on exact ties, as tie_layer:
// a depthwise layer of `channels` channels (a 3x3 of 16 or more takes the
// 16-channel path, so these have 8), codes out or fp32 requantised, against
// the float sequence element by element; and exactly one kernel call per
// channel per output row per kPlaneCall (32) pixels of it.
void tie_plane_layer(uint32_t k, uint32_t stride, uint32_t h, uint32_t w,
                     bool requant, uint32_t dil_h = 1) {
  constexpr uint32_t kC = 8;
  Op op;
  std::memset(&op, 0, sizeof(op));
  op.h_in = h;
  op.w_in = w;
  op.kh = op.kw = k;
  op.stride_h = op.stride_w = stride;
  op.dil_h = dil_h;
  op.dil_w = 1;
  op.pad_h = dil_h * (k / 2);
  op.pad_w = k / 2;
  op.h_out = (h + 2 * op.pad_h - dil_h * (k - 1) - 1) / stride + 1;
  op.w_out = (w + 2 * op.pad_w - k) / stride + 1;
  op.depthwise = true;
  op.c_in = 1;
  op.k = k * k;
  op.k_pad = (op.k + 15) / 16 * 16;
  op.n = kC;
  op.m = op.h_out * op.w_out;
  const size_t plane = (size_t)op.h_in * op.w_in;
  std::vector<int8_t> x(kC * plane + 32);
  for (int8_t& v : x) {
    v = (int8_t)integer(-4, 4);
  }
  alignas(16) static int8_t wq[kC * 32];
  std::vector<float> scales(kC, 0.125f);
  std::vector<int32_t> sums(kC);
  for (uint32_t n = 0; n < kC; ++n) {
    int32_t sum = 0;
    for (uint32_t j = 0; j < op.k_pad; ++j) {
      const int8_t wv = j < op.k ? (int8_t)integer(-4, 4) : 0;
      wq[n * op.k_pad + j] = wv;
      sum += wv;
    }
    sums[n] = sum;
  }
  Act act;
  std::memset(&act, 0, sizeof(act));
  act.x8 = x.data();
  act.qmin = -128;
  act.qmax = 127;
  act.sx = 1.0f;
  act.zx = 0;
  esp32_conv::Columns cols;
  std::memset(&cols, 0, sizeof(cols));
  cols.wq = wq;
  cols.pie = true;
  cols.scales = scales.data();
  cols.sums = sums.data();
  Out out;
  std::memset(&out, 0, sizeof(out));
  out.requant = true;
  out.out_inv = 1.0f;
  out.out_scale = 1.0f;
  out.qmin = -128;
  out.qmax = 127;
  std::vector<int8_t> y8(kC * op.m, 0x5A);
  std::vector<float> y(kC * op.m, -7.0f);
  if (requant) {
    out.y = y.data();
  } else {
    out.y8 = y8.data();
  }
  alignas(16) static int8_t scratch[32 * 1024];
  if (esp32_conv::scratch_need(op, false, 0) > sizeof(scratch)) {
    ++g_tie_failures;
    return;
  }
  esp32_conv::Scratch sc{scratch, 0, nullptr};
  esp32_conv::Counts counts{0, 0, 0, 0, 0};
  const Esp32ReferenceKernelCalls before = esp32_reference_kernel_calls();
  esp32_conv::run(op, act, cols, out, sc, &counts);
  const Esp32ReferenceKernelCalls after = esp32_reference_kernel_calls();
  const uint64_t calls = after.dw_plane - before.dw_plane;
  // The pairs form (5x5, stride 1, rows of at most 8): a call per four runs
  // of two output rows; else a call per output row per kPlaneCall pixels.
  const bool pairs = k == 5 && stride == 1 && op.w_out <= 8;
  const uint64_t want =
      pairs ? (uint64_t)kC * (((op.h_out + 1) / 2 + 3) / 4)
            : (uint64_t)kC * op.h_out *
                  ((op.w_out + esp32_conv::kPlaneCall - 1) /
                   esp32_conv::kPlaneCall);
  if (calls != want || counts.pie != kC * op.m || counts.scalar != 0) {
    std::printf("  kernels: depthwise %ux%u stride %u %s called dw_plane %llu "
                "(want %llu), pie %u scalar %u\n",
                (unsigned)k, (unsigned)k, (unsigned)stride,
                requant ? "fp32" : "codes", (unsigned long long)calls,
                (unsigned long long)want, (unsigned)counts.pie,
                (unsigned)counts.scalar);
    ++g_kernel_failures;
  }
  esp32_conv::Epi e;
  e.sx = act.sx;
  e.lo = -INFINITY;
  e.hi = INFINITY;
  e.out_inv = out.out_inv;
  e.out_scale = out.out_scale;
  e.zo = out.zo;
  e.qmin = out.qmin;
  e.qmax = out.qmax;
  esp32_conv::fixed_epi(op, act, out, &e);
  for (uint32_t n = 0; n < kC; ++n) {
    for (uint32_t r = 0; r < op.m; ++r) {
      const int32_t oh = (int32_t)(r / op.w_out), ow = (int32_t)(r % op.w_out);
      int32_t acc = 0;
      for (int32_t i = 0; i < (int32_t)k; ++i) {
        for (int32_t j = 0; j < (int32_t)k; ++j) {
          const int32_t ih = oh * (int32_t)stride + i * (int32_t)dil_h -
                             (int32_t)op.pad_h;
          const int32_t iw = ow * (int32_t)stride + j - (int32_t)op.pad_w;
          if (ih >= 0 && ih < (int32_t)op.h_in && iw >= 0 &&
              iw < (int32_t)op.w_in) {
            acc += x[n * plane + ih * op.w_in + iw] *
                   wq[n * op.k_pad + i * k + j];
          }
        }
      }
      const int32_t q = esp32_conv::float_code(acc, 0.125f, 0.0f, e);
      const size_t at = (size_t)n * op.m + r;
      const bool ok = requant ? y[at] == (float)(q - e.zo) * e.out_scale
                              : y8[at] == (int8_t)q;
      ++g_tie_outputs;
      g_tie_ties += (acc % 8 == 4 || acc % 8 == -4) ? 1 : 0;
      if (!ok && g_tie_failures++ < 8) {
        std::printf("  tie mismatch: depthwise %ux%u stride %u %s channel %u "
                    "position %u acc %d want %d\n",
                    (unsigned)k, (unsigned)k, (unsigned)stride,
                    requant ? "fp32" : "codes", (unsigned)n, (unsigned)r,
                    (int)acc, (int)q);
      }
    }
  }
}

// The plane path's reads (plane_geo): for every geometry it accepts -- 3x3
// and 5x5, column strides 1 and 2, row strides 1 to 3, row dilations 1 and
// 2, pads 0 to 2, inputs up to 24 rows of 1 to 40 columns and some as wide
// as 600 (past the pad row's reach) -- each tap's two aligned 16-byte
// blocks, over every output row and kernel call, computed here from the
// kernels' contract and depthwise_plane's row pointers, must stay inside the
// plane's Hp * Wp bytes of scratch (scratch_need) or the pad row: plane_geo's
// own bound is the code under test, not used here. Also counts what it
// refuses.
uint64_t g_reads_checked = 0, g_reads_refused = 0, g_reads_failures = 0;

void plane_reads() {
  for (uint32_t k = 3; k <= 5; k += 2) {
    for (uint32_t sw = 1; sw <= 2; ++sw) {
      for (uint32_t sh = 1; sh <= 3; ++sh) {
        for (uint32_t dil = 1; dil <= 2; ++dil) {
          for (uint32_t pad = 0; pad <= 2; ++pad) {
            static const uint32_t kWide[] = {100, 200, 230, 240, 250, 260, 300, 600};
            for (uint32_t h = 1; h <= 24; ++h) {
              for (uint32_t wi = 0; wi < 40 + sizeof(kWide) / sizeof(kWide[0]); ++wi) {
                const uint32_t w = wi < 40 ? wi + 1 : kWide[wi - 40];
                const uint32_t span_h = dil * (k - 1) + 1;
                if (h + 2 * pad < span_h || w + 2 * pad < k) {
                  continue;
                }
                Op op;
                std::memset(&op, 0, sizeof(op));
                op.h_in = h;
                op.w_in = w;
                op.c_in = 1;
                op.kh = op.kw = k;
                op.stride_h = sh;
                op.stride_w = sw;
                op.dil_h = dil;
                op.dil_w = 1;
                op.pad_h = op.pad_w = pad;
                op.h_out = (h + 2 * pad - span_h) / sh + 1;
                op.w_out = (w + 2 * pad - k) / sw + 1;
                op.depthwise = true;
                op.n = 1;
                op.m = op.h_out * op.w_out;
                op.k = k * k;
                op.k_pad = 32;
                esp32_conv::PlaneGeo g;
                if (!esp32_conv::plane_geo(op, &g)) {
                  ++g_reads_refused;
                  continue;
                }
                ++g_reads_checked;
                const uint64_t Hp = (uint64_t)(op.h_out - 1) * sh + span_h;
                const uint64_t Wp = (uint64_t)(op.w_out - 1) * sw + k;
                const uint32_t ka = (k + 1) / 2, kb = k / 2;
                // The rows stored: the input rows the window reads.
                const uint64_t below = Hp > pad ? Hp - pad : 0;
                const uint64_t rows_in = h < below ? h : below;
                uint64_t plane_end = 0, pad_end = 0;
                for (uint32_t oh = 0; oh < op.h_out; ++oh) {
                  for (uint32_t i = 0; i < k; ++i) {
                    const int64_t ih = (int64_t)oh * sh + (int64_t)i * dil - pad;
                    const bool stored = ih >= 0 && ih < (int64_t)rows_in;
                    const uint64_t base = stored ? (uint64_t)ih * Wp : 0;
                    for (uint32_t ow0 = 0; ow0 < op.w_out; ow0 += 16) {
                      // A taps at base + ow0 + j, B taps at base + boff +
                      // ow0 + j; each load reads [a & ~15, (a & ~15) + 32).
                      uint64_t far = base + ow0 + (ka - 1);
                      const uint64_t b = base + g.boff + ow0 + (kb - 1);
                      far = b > far ? b : far;
                      const uint64_t end = ((far + 16) | 15) + 1;
                      uint64_t& e = stored ? plane_end : pad_end;
                      e = end > e ? end : e;
                    }
                  }
                }
                if (plane_end > Hp * Wp || pad_end > esp32_conv::kPlanePadBytes ||
                    rows_in * Wp > Hp * Wp) {
                  if (g_reads_failures++ < 8) {
                    std::printf("  plane reads: %ux%u s%u/%u dil %u pad %u in %ux%u: "
                                "plane to %llu of %llu, pad row to %llu\n",
                                (unsigned)k, (unsigned)k, (unsigned)sh,
                                (unsigned)sw, (unsigned)dil, (unsigned)pad,
                                (unsigned)h, (unsigned)w,
                                (unsigned long long)plane_end,
                                (unsigned long long)(Hp * Wp),
                                (unsigned long long)pad_end);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t channels = 20000, brute_n = 40, seed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--channels")) {
      channels = std::strtoull(argv[i + 1], nullptr, 10);
    } else if (!std::strcmp(argv[i], "--brute")) {
      brute_n = std::strtoull(argv[i + 1], nullptr, 10);
    } else if (!std::strcmp(argv[i], "--seed")) {
      seed = std::strtoull(argv[i + 1], nullptr, 10);
    } else {
      std::fprintf(stderr, "usage: %s [--channels N] [--brute N] [--seed S]\n",
                   argv[0]);
      return 2;
    }
  }
  g_rng.seed(seed);
  uint64_t eligible = 0, brute_done = 0, family_counts[5] = {0, 0, 0, 0, 0};
  for (uint64_t i = 0; i < channels; ++i) {
    const int family = (int)(i % 5);
    const Case c = random_case(family);
    const esp32_conv::Epi e = epi_of(c);
    int32_t sum = 0;
    const esp32_conv::Channel ch = channel_of(c, e, &sum);
    if (ch.fx.guard == 0) {
      continue;
    }
    ++eligible;
    ++family_counts[family];
    if (!esp32_conv::check_channel(ch, e) && esp32_conv::g_epi_check.failures <= 8) {
      std::printf("  channel mismatch: k %u sx %a s %a b %a out_inv %a lo %a hi %a "
                  "zo %d q [%d, %d] zx %d\n",
                  (unsigned)c.k, c.sx, c.s, c.b, c.out_inv, c.lo, c.hi, (int)c.zo,
                  (int)c.qmin, (int)c.qmax, (int)c.zx);
    }
    loops(c, e);
    // The brute force on a share of the channels, small and large reaches.
    if (brute_done < brute_n && (int64_t)e.amax <= (int64_t{1} << 25)) {
      brute(ch, e);
      ++brute_done;
    }
  }
  // The kernels' boot checks pass on the host's C kernels (a check that
  // refused a correct kernel would leave the C loops running unnoticed),
  // so the layers below run the new paths.
  esp32_conv::pie_dw_init();
  const bool kernels_on =
      esp32_conv::g_pie_dw_state.load() == 2 &&
      esp32_conv::g_rows_codes_ok.load() != 0 &&
      esp32_conv::g_transpose_ok.load() != 0 &&
      esp32_conv::g_plane_ok.load() != 0;
  if (!kernels_on) {
    std::printf("  a kernel's boot check refused the host's C kernel\n");
    ++g_kernel_failures;
  }
  // The path's redo of the elements near a boundary, on exact ties.
  for (int depthwise = 0; depthwise < 2; ++depthwise) {
    for (int requant = 0; requant < 2; ++requant) {
      tie_layer(depthwise != 0, requant != 0);
    }
  }
  // The plane path: 5x5 at strides 1 and 2 (v3-small's), 3x3 rows of two
  // kernel calls (40 pixels), each codes out and fp32.
  for (int requant = 0; requant < 2; ++requant) {
    tie_plane_layer(5, 1, 14, 14, requant != 0);
    tie_plane_layer(5, 2, 15, 15, requant != 0);
    tie_plane_layer(3, 1, 6, 40, requant != 0);
    // The pairs form: 7-wide rows, an odd row count and an even one, and
    // row dilation 2.
    tie_plane_layer(5, 1, 9, 7, requant != 0);
    tie_plane_layer(5, 1, 8, 6, requant != 0);
    tie_plane_layer(5, 1, 9, 7, requant != 0, 2);
  }
  plane_reads();
  std::printf("conv epilogue check: plane reads: %" PRIu64 " geometries the plane "
              "path takes, %" PRIu64 " it refuses, %" PRIu64 " reading outside\n",
              g_reads_checked, g_reads_refused, g_reads_failures);
  std::printf("conv epilogue check: ties: %" PRIu64 " outputs of the path, %" PRIu64
              " on an exact tie, %" PRIu64 " mismatch(es); %" PRIu64
              " kernel failure(s)\n",
              g_tie_outputs, g_tie_ties, g_tie_failures, g_kernel_failures);
  const auto& k = esp32_conv::g_epi_check;
  const uint64_t failures = k.failures + g_brute_failures + g_loop_failures +
                            g_tie_failures + g_kernel_failures +
                            g_reads_failures;
  std::printf("conv epilogue check: %" PRIu64 " channels, %" PRIu64
              " on the integer form (by family %" PRIu64 "/%" PRIu64 "/%" PRIu64
              "/%" PRIu64 "/%" PRIu64 "), %" PRIu64 " code intervals, %" PRIu64
              " evaluations; brute force %" PRIu64 " channels, %" PRIu64
              " accumulators, %.4f%% on the float fallback; loops %" PRIu64
              " elements; %" PRIu64 " mismatch(es)\n",
              channels, eligible, family_counts[0], family_counts[1],
              family_counts[2], family_counts[3], family_counts[4], k.intervals,
              k.evaluations, brute_done, g_brute_accs,
              g_slow_of ? 100.0 * (double)g_slow / (double)g_slow_of : 0.0,
              g_loop_elements, failures);
  return failures == 0 ? 0 : 1;
}
