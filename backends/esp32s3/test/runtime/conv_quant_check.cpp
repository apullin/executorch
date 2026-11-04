/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The static int8 conv path's branch-free fp32 quantiser (QuantFast,
// runtime/esp32_conv.cpp) against code_of, the quantiser it replaces in the
// dense gathers, on the host:
//
//   - the code of a value: with inv = 1 the product is the value itself, so
//     sweeping x sweeps every v = fl(x * inv) the quantiser can see. The
//     default sweep takes every float of magnitude 2^-3 to 2^10 at a
//     mantissa stride of 8, every half-integer in [-1024, 1024] and its
//     neighbours four ulps either side, the specials (both zeros, the
//     infinities, NaNs of both signs and several payloads, the extremes, the
//     subnormals) and 2^24 random bit patterns, for zero points inside and
//     outside the codes and narrow code ranges; --full sweeps all 2^32
//     floats for each of them. Other scales are compared on random inputs
//     (the product is the same IEEE operation in both);
//   - the loops that use it, on random tensors with specials among them:
//     quantize_span (prepare's plane), gather_pointwise_fast against
//     gather_pointwise with code_of, and gather_dense_fast (its runs and the
//     copies of the overlapping columns) against gather_dense, over random
//     geometries and blocks; each row block must equal code_of's byte for
//     byte, the bytes past column k included (untouched by both).
//
// The unit is included whole, so this sees its internal functions; it builds
// -ffp-contract=off, as the unit does everywhere.
//
//   esp32_conv_quant_check [--full] [--seed S]
//
// Prints one "conv quant check:" line; exits 1 on any mismatch.

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
using namespace esp32_conv;

std::mt19937_64 g_rng;
uint64_t g_values = 0, g_value_failures = 0;
uint64_t g_blocks = 0, g_block_failures = 0;

float from_bits(uint32_t b) {
  float f;
  memcpy(&f, &b, sizeof(f));
  return f;
}

struct Record {
  int32_t zx, qmin, qmax;
};

// Zero points inside the codes (the bench models' -128 .. 127 and 0), at
// the edges, outside them (0 then clamps to a rail), and narrow ranges.
const Record kRecords[] = {
    {0, -128, 127},  {-128, -128, 127}, {127, -128, 127}, {-31, -128, 127},
    {27, -128, 127}, {200, -128, 127},  {-300, -128, 127}, {3, -5, 9},
    {-7, -5, 9},     {0, 0, 0},
};

Act act_of(const Record& r, float sx) {
  Act a;
  memset(&a, 0, sizeof(a));
  a.qmin = r.qmin;
  a.qmax = r.qmax;
  a.zx = r.zx;
  a.sx = sx;
  return a;
}

/// One value: QuantFast's code against code_of's, at inv = fl(1 / sx).
inline void check_value(const QuantFastRegs& q, const Quant& ref, float x) {
  ++g_values;
  const int8_t got = q(x);
  const int8_t want = ref(x);
  if (got != want) {
    if (g_value_failures < 10) {
      uint32_t b;
      memcpy(&b, &x, sizeof(b));
      printf("  value mismatch: x bits 0x%08x zx %d [%d, %d]: fast %d, code_of %d\n",
             b, ref.zx, ref.qmin, ref.qmax, got, want);
    }
    ++g_value_failures;
  }
}

void sweep_values(const Record& r, bool full) {
  // sx = 1: inv = 1, so v = x for every x.
  const Act a = act_of(r, 1.0f);
  if (!quant_fast_ok(a)) {
    printf("  record zx %d [%d, %d] refused\n", r.zx, r.qmin, r.qmax);
    ++g_value_failures;
    return;
  }
  const QuantFastRegs q(quant_fast_of(a));
  const Quant ref = quant_of(a);
  if (full) {
    uint32_t b = 0;
    do {
      check_value(q, ref, from_bits(b));
    } while (++b != 0);
    return;
  }
  // Magnitudes 2^-3 .. 2^10, both signs, every eighth mantissa.
  for (uint32_t sign = 0; sign < 2; ++sign) {
    for (uint32_t e = 127 - 3; e < 127 + 10; ++e) {
      for (uint32_t m = 0; m < (1u << 23); m += 8) {
        check_value(q, ref, from_bits((sign << 31) | (e << 23) | m));
      }
    }
  }
  // Every half-integer in [-1024, 1024] and four ulps either side.
  for (int32_t k = -1024; k <= 1024; ++k) {
    const float h = (float)k + 0.5f;
    uint32_t hb;
    memcpy(&hb, &h, sizeof(hb));
    for (int32_t d = -4; d <= 4; ++d) {
      check_value(q, ref, from_bits(hb + (uint32_t)d));
    }
    check_value(q, ref, (float)k);
  }
  // The specials.
  const uint32_t specials[] = {
      0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
      0xFFC00000u, 0x7F800001u, 0xFF800001u, 0x7FFFFFFFu, 0xFFFFFFFFu,
      0x7FA00000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x00000001u, 0x80000001u,
      0x007FFFFFu, 0x807FFFFFu, 0x00800000u, 0x80800000u, 0x4A800000u,
      0xCA800000u, 0x4B000000u, 0xCB000000u, 0x4AFFFFFFu, 0xCAFFFFFFu,
      0x4B400000u, 0xCB400000u};
  for (const uint32_t s : specials) {
    check_value(q, ref, from_bits(s));
  }
  for (uint32_t i = 0; i < (1u << 24); ++i) {
    check_value(q, ref, from_bits((uint32_t)g_rng()));
  }
}

/// Other scales on random inputs, specials among them.
void random_scales(const Record& r) {
  std::uniform_real_distribution<float> lg(-12.0f, 6.0f);
  for (int s = 0; s < 64; ++s) {
    const float sx = std::exp2(lg(g_rng));
    const Act a = act_of(r, sx);
    const QuantFastRegs q(quant_fast_of(a));
    const Quant ref = quant_of(a);
    std::uniform_real_distribution<float> u(-300.0f * sx, 300.0f * sx);
    for (int i = 0; i < 20000; ++i) {
      float x = u(g_rng);
      if (i % 101 == 0) {
        x = from_bits((uint32_t)g_rng());
      } else if (i % 7 == 0) {
        // A tie when sx is a power of two, close to one otherwise.
        x = sx * ((float)(int32_t)(g_rng() % 600) - 300.0f + 0.5f);
      }
      check_value(q, ref, x);
    }
  }
}

float random_input(std::mt19937_64& rng, float sx) {
  const uint32_t pick = (uint32_t)(rng() % 64);
  if (pick == 0) {
    const float s[] = {NAN, -NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 3e30f,
                       -3e30f};
    return s[rng() % 8];
  }
  if (pick == 1) {
    return sx * ((float)(int32_t)(rng() % 400) - 200.0f + 0.5f);
  }
  std::uniform_real_distribution<float> u(-160.0f * sx, 160.0f * sx);
  return u(rng);
}

Op random_op(bool pointwise) {
  Op op;
  memset(&op, 0, sizeof(op));
  auto pick = [](uint32_t lo, uint32_t hi) {
    return lo + (uint32_t)(g_rng() % (hi - lo + 1));
  };
  op.c_in = pick(1, pointwise ? 40 : 5);
  op.h_in = pick(1, 17);
  op.w_in = pick(1, 23);
  if (pointwise) {
    op.kh = op.kw = 1;
    op.stride_h = pick(1, 2);
    op.stride_w = op.stride_h;
  } else {
    const uint32_t ks[] = {1, 2, 3, 3, 3, 5, 7};
    op.kh = ks[g_rng() % 7];
    op.kw = ks[g_rng() % 7];
    op.stride_h = pick(1, 3);
    op.stride_w = pick(1, 3);
    op.pad_h = pick(0, op.kh);
    op.pad_w = pick(0, op.kw);
  }
  op.dil_h = (!pointwise && g_rng() % 4 == 0) ? 2 : 1;
  op.dil_w = (!pointwise && g_rng() % 4 == 0) ? 2 : 1;
  const int64_t eh = (int64_t)op.h_in + 2 * op.pad_h - (int64_t)op.dil_h * (op.kh - 1) - 1;
  const int64_t ew = (int64_t)op.w_in + 2 * op.pad_w - (int64_t)op.dil_w * (op.kw - 1) - 1;
  if (eh < 0 || ew < 0) {
    op.m = 0;
    return op;
  }
  op.h_out = (uint32_t)(eh / op.stride_h) + 1;
  op.w_out = (uint32_t)(ew / op.stride_w) + 1;
  op.m = op.h_out * op.w_out;
  op.k = op.c_in * op.kh * op.kw;
  op.k_pad = (op.k + 15) / 16 * 16;
  op.n = 1;
  return op;
}

/// Row blocks of one random op both ways: the general gathers with code_of, the
/// fast ones with QuantFast, each block from a random first position and
/// size, every byte of the rows (the tail past k pre-filled alike).
void check_gathers(bool pointwise) {
  const Op op = random_op(pointwise);
  if (op.m == 0) {
    return;
  }
  const Record& r = kRecords[g_rng() % (sizeof(kRecords) / sizeof(kRecords[0]))];
  const float sx = std::exp2((float)(int32_t)(g_rng() % 9) - 6.0f) *
                   (g_rng() % 2 ? 1.0f : 1.37f);
  const Act a = act_of(r, sx);
  const size_t in = (size_t)op.c_in * op.h_in * op.w_in;
  std::vector<float> x(in);
  for (float& v : x) {
    v = random_input(g_rng, sx);
  }
  const uint32_t blk = 1 + (uint32_t)(g_rng() % 70);
  std::vector<int8_t> want((size_t)blk * op.k_pad), got(want.size());
  uint32_t oh_lo = 0, oh_hi = 0, ow_lo = 0, ow_hi = 0;
  if (!pointwise) {
    interior(op.h_out, op.h_in, op.kh, op.stride_h, op.pad_h, op.dil_h, &oh_lo,
             &oh_hi);
    interior(op.w_out, op.w_in, op.kw, op.stride_w, op.pad_w, op.dil_w, &ow_lo,
             &ow_hi);
  }
  const QuantFast qf = quant_fast_of(a);
  for (int b = 0; b < 4; ++b) {
    const uint32_t r0 = (uint32_t)(g_rng() % op.m);
    const uint32_t left = op.m - r0;
    const uint32_t nrows = 1 + (uint32_t)(g_rng() % (left < blk ? left : blk));
    memset(want.data(), 0x6B, want.size());
    memset(got.data(), 0x6B, got.size());
    if (pointwise) {
      gather_pointwise(op, Quantised{x.data(), quant_of(a)}, r0, nrows,
                       want.data());
      if (op.stride_h == 1 && op.stride_w == 1) {
        gather_pointwise_fast(op, x.data(), qf, r0, nrows, got.data());
      } else {
        gather_pointwise(op, QuantisedFast{x.data(), QuantFastRegs(qf)}, r0,
                         nrows, got.data());
      }
    } else {
      gather_dense(op, Quantised{x.data(), quant_of(a)}, a.zx, r0, nrows,
                   want.data(), oh_lo, oh_hi, ow_lo, ow_hi);
      gather_dense_fast(op, x.data(), qf, a.zx, r0, nrows, got.data(), oh_lo,
                        oh_hi, ow_lo, ow_hi);
    }
    ++g_blocks;
    if (want != got) {
      if (g_block_failures < 10) {
        printf("  %s block mismatch: c_in %u %ux%u k %ux%u s %u,%u p %u,%u d %u,%u "
               "r0 %u rows %u zx %d\n", pointwise ? "pointwise" : "dense",
               op.c_in, op.h_in, op.w_in, op.kh, op.kw, op.stride_h,
               op.stride_w, op.pad_h, op.pad_w, op.dil_h, op.dil_w, r0, nrows,
               r.zx);
      }
      ++g_block_failures;
    }
  }
}

/// quantize_span (prepare's plane) against code_of element by element.
void check_spans() {
  for (int s = 0; s < 200; ++s) {
    const Record& r = kRecords[g_rng() % (sizeof(kRecords) / sizeof(kRecords[0]))];
    const float sx = std::exp2((float)(int32_t)(g_rng() % 9) - 6.0f) * 1.11f;
    const Act a = act_of(r, sx);
    const size_t n = (size_t)(g_rng() % 300);
    std::vector<float> x(n);
    for (float& v : x) {
      v = random_input(g_rng, sx);
    }
    std::vector<int8_t> got(n + 1, 0x33);
    quantize_span(a, x.data(), n, got.data());
    const Quant ref = quant_of(a);
    bool ok = got[n] == 0x33;
    for (size_t i = 0; i < n; ++i) {
      ok = ok && got[i] == ref(x[i]);
    }
    ++g_blocks;
    if (!ok) {
      if (g_block_failures < 10) {
        printf("  quantize_span mismatch: n %zu zx %d\n", n, r.zx);
      }
      ++g_block_failures;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool full = false;
  uint64_t seed = 0x5EEDu;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--full") == 0) {
      full = true;
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = strtoull(argv[++i], nullptr, 0);
    } else {
      printf("usage: esp32_conv_quant_check [--full] [--seed S]\n");
      return 2;
    }
  }
  g_rng.seed(seed);
  // A zero point too far from its codes for the adder's range is refused
  // (the gathers then keep code_of).
  {
    const Record far = {(1 << 22) + 200, -128, 127};
    if (quant_fast_ok(act_of(far, 1.0f))) {
      printf("  a zero point 2^22 past its codes was not refused\n");
      ++g_value_failures;
    }
  }
  for (const Record& r : kRecords) {
    sweep_values(r, full);
    random_scales(r);
  }
  check_spans();
  for (int i = 0; i < 3000; ++i) {
    check_gathers(i % 3 == 0);
  }
  printf("conv quant check: %" PRIu64 " value(s)%s, %" PRIu64
         " mismatch(es); %" PRIu64 " row block(s) and span(s), %" PRIu64
         " mismatch(es)\n",
         g_values, full ? " (every float)" : "", g_value_failures, g_blocks,
         g_block_failures);
  return (g_value_failures == 0 && g_block_failures == 0) ? 0 : 1;
}
