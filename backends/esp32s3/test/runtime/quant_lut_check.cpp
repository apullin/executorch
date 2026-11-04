/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// espressif::quant_lut's conformance check (quant_lut.h): the kernel's
// quantize step against ExecuTorch 1.6's portable quantize_val (a
// transcription, nearbyintf's rounding) and torch's own quantize_per_tensor
// (quant_lut_golden.h), NaN and the infinities included, its output bits, and
// one case per malformed argument through the shim. Test code: main.cpp's parity run and
// the host's --selftest call it; the story firmware never does, so it links
// none of this file. -ffp-contract=off like the kernel, so the references
// compute each step as written.

#include <cmath>
#include <cstdio>
#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include "elementwise.h"
#include "quant_lut.h"
#include "quant_lut_golden.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace espressif {
namespace quant_lut {

namespace {

uint32_t float_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}

float bits_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

uint32_t fnv1a(uint32_t h, uint32_t word) {
  for (int b = 0; b < 4; ++b) {
    h = (h ^ ((word >> (8 * b)) & 0xFFu)) * 16777619u;
  }
  return h;
}

// Entry i of the check's table: its low byte is i, so an output decodes to
// its code; the rest makes signalling and negative quiet NaN payloads,
// denormals and a normal value, which a float copy could disturb.
uint32_t entry(int i) {
  static const uint32_t kind[4] = {0x7F800100u, 0x00000100u, 0xFFC00100u,
                                   0x3F800100u};
  return kind[i & 3] | (uint32_t)i;
}

// The first independent reference: ExecuTorch 1.6's quantize_val_with_inv_scale
// (kernels/quantized/cpu/op_quantize.cpp), transcribed: NaN takes the zero
// point, any other value zp + nearbyint(fl(x * inv)), clamped in float64
// before the conversion, so the infinities land on the rails.
int32_t nearbyint_code(float x, float inv, int32_t zero_point,
                       int32_t quant_min, int32_t quant_max) {
  const float v = x * inv;
  const double q = std::isnan(v) ? (double)zero_point
                                 : (double)zero_point + (double)nearbyintf(v);
  const double c = q < (double)quant_min
                       ? (double)quant_min
                       : (q > (double)quant_max ? (double)quant_max : q);
  return (int32_t)c;
}

struct Range {
  int32_t quant_min, quant_max;
};

struct Tally {
  uint64_t cases = 0, nearbyint = 0, torch = 0, printed = 0;
};

// Every zero point of {qmin, -1, 0, 1, qmax} inside each range.
void check_input(Tally& tally, const float table[256], float x, float inv,
                 bool has_torch, int32_t torch_round) {
  static const Range kRanges[3] = {{-128, 127}, {-127, 127}, {0, 255}};
  for (const Range& range : kRanges) {
    const int32_t zps[5] = {range.quant_min, -1, 0, 1, range.quant_max};
    for (int z = 0; z < 5; ++z) {
      const int32_t zp = zps[z];
      bool repeat = zp < range.quant_min || zp > range.quant_max;
      for (int earlier = 0; earlier < z; ++earlier) {
        repeat |= zps[earlier] == zp;
      }
      if (repeat) {
        continue;
      }
      float y;
      apply(&x, &y, 1, table, inv, zp, range.quant_min, range.quant_max);
      const uint32_t got = float_bits(y);
      const int32_t code = range.quant_min + (int32_t)(got & 0xFFu);
      const bool decodes = got == entry((int)(got & 0xFFu));
      const int32_t ref =
          nearbyint_code(x, inv, zp, range.quant_min, range.quant_max);
      // The golden's rounding of NaN is 0, so its code is the zero point:
      // ExecuTorch 1.6's (torch's own is platform-defined).
      const bool nan = std::isnan(x);
      int32_t want_torch = torch_round + zp;
      want_torch = want_torch < range.quant_min ? range.quant_min : want_torch;
      want_torch = want_torch > range.quant_max ? range.quant_max : want_torch;
      const bool bad_ref = !decodes || code != ref || (nan && code != zp);
      const bool bad_torch = has_torch && code != want_torch;
      ++tally.cases;
      tally.nearbyint += bad_ref;
      tally.torch += bad_torch;
      if ((bad_ref || bad_torch) && tally.printed++ < 8) {
        printf("  MISMATCH quant_lut x %08lx inv %08lx zp %ld [%ld, %ld]: "
               "output %08lx (code %ld), nearbyintf %ld, torch %ld\n",
               (unsigned long)float_bits(x), (unsigned long)float_bits(inv),
               (long)zp, (long)range.quant_min, (long)range.quant_max,
               (unsigned long)got, (long)code, (long)ref,
               has_torch ? (long)want_torch : -9999L);
      }
    }
  }
}

// The malformed-argument cases, through the shim on hand-built tensors. Each
// must fail with its error and leave out's bytes as they were.
struct Malformed {
  int refused = 0, total = 0;
  bool untouched = true;
};

struct Shape {
  int8_t dim;
  executorch::aten::TensorImpl::SizesType sizes[4];
  executorch::aten::TensorImpl::DimOrderType order[4];
  executorch::aten::TensorImpl::StridesType strides[4];
};

enum ShapeId : uint8_t {
  kRow,       // [2, 3]
  kColumn,    // [2, 3], not contiguous
  kSwapped,   // [3, 2]: static, so out cannot become x's [2, 3]
  kNhwc,      // [1, 2, 2, 2] channels-last
  kTable256,
  kTable255,
  kTable384,
  kStrided,   // [256], stride 2
  kSquare,    // [16, 16]
  kTable1,
};

const Shape kShapes[] = {
    {2, {2, 3}, {0, 1}, {3, 1}},
    {2, {2, 3}, {0, 1}, {1, 2}},
    {2, {3, 2}, {0, 1}, {2, 1}},
    {4, {1, 2, 2, 2}, {0, 2, 3, 1}, {8, 1, 4, 2}},
    {1, {256}, {0}, {1}},
    {1, {255}, {0}, {1}},
    {1, {384}, {0}, {1}},
    {1, {256}, {0}, {2}},
    {2, {16, 16}, {0, 1}, {16, 1}},
    {1, {1}, {0}, {1}},
};

struct Case {
  const char* what;
  ShapeId xs, ts, os;
  ScalarType xt, tt, ot;
  double inv;
  int16_t zp, qmin, qmax;
  bool inv_is_int, zp_is_double;
  uint8_t slots;
  Error want;
};

constexpr ScalarType F = ScalarType::Float;
constexpr Error kBad = Error::InvalidArgument;
const Case kCases[] = {
    {"control", kRow, kTable256, kRow, F, F, F, 0.5, 0, -128, 127, false,
     false, 8, Error::Ok},
    {"x fp16", kRow, kTable256, kRow, ScalarType::Half, F, F, 0.5, 0, -128, 127,
     false, false, 8, kBad},
    {"x int32", kRow, kTable256, kRow, ScalarType::Int, F, F, 0.5, 0, -128, 127,
     false, false, 8, kBad},
    {"table int32", kRow, kTable256, kRow, F, ScalarType::Int, F, 0.5, 0, -128,
     127, false, false, 8, kBad},
    {"out fp64", kRow, kTable256, kRow, F, F, ScalarType::Double, 0.5, 0, -128,
     127, false, false, 8, kBad},
    {"table 2-D", kRow, kSquare, kRow, F, F, F, 0.5, 0, -128, 127, false, false,
     8, kBad},
    {"table strided", kRow, kStrided, kRow, F, F, F, 0.5, 0, -128, 127, false,
     false, 8, kBad},
    {"table short", kRow, kTable255, kRow, F, F, F, 0.5, 0, -128, 127, false,
     false, 8, kBad},
    {"range of 384", kRow, kTable384, kRow, F, F, F, 0.5, 0, -128, 255, false,
     false, 8, kBad},
    {"qmin -129", kRow, kTable256, kRow, F, F, F, 0.5, 0, -129, 126, false,
     false, 8, kBad},
    {"qmax 256", kRow, kTable256, kRow, F, F, F, 0.5, 1, 1, 256, false, false, 8,
     kBad},
    {"qmin == qmax", kRow, kTable1, kRow, F, F, F, 0.5, 5, 5, 5, false, false,
     8, kBad},
    {"zp below", kRow, kTable256, kRow, F, F, F, 0.5, -129, -128, 127, false,
     false, 8, kBad},
    {"zp above", kRow, kTable256, kRow, F, F, F, 0.5, 128, -128, 127, false,
     false, 8, kBad},
    {"inv 0", kRow, kTable256, kRow, F, F, F, 0.0, 0, -128, 127, false, false,
     8, kBad},
    {"inv inf", kRow, kTable256, kRow, F, F, F, INFINITY, 0, -128, 127, false,
     false, 8, kBad},
    {"inv NaN", kRow, kTable256, kRow, F, F, F, NAN, 0, -128, 127, false, false,
     8, kBad},
    {"inv negative", kRow, kTable256, kRow, F, F, F, -0.5, 0, -128, 127, false,
     false, 8, kBad},
    {"inv not float32", kRow, kTable256, kRow, F, F, F, 0.1, 0, -128, 127,
     false, false, 8, kBad},
    {"inv an int", kRow, kTable256, kRow, F, F, F, 2.0, 0, -128, 127, true,
     false, 8, kBad},
    {"zp a double", kRow, kTable256, kRow, F, F, F, 0.5, 0, -128, 127, false,
     true, 8, kBad},
    {"out other shape", kRow, kTable256, kSwapped, F, F, F, 0.5, 0, -128, 127,
     false, false, 8, kBad},
    {"x strided", kColumn, kTable256, kRow, F, F, F, 0.5, 0, -128, 127, false,
     false, 8, kBad},
    {"x channels-last", kNhwc, kTable256, kNhwc, F, F, F, 0.5, 0, -128, 127,
     false, false, 8, kBad},
    {"out strided", kRow, kTable256, kColumn, F, F, F, 0.5, 0, -128, 127, false,
     false, 8, kBad},
    {"seven slots", kRow, kTable256, kRow, F, F, F, 0.5, 0, -128, 127, false,
     false, 7, Error::InvalidProgram},
};

void check_malformed(Malformed& result, bool& control_ok) {
  using executorch::aten::TensorImpl;
  using executorch::runtime::TensorShapeDynamism;
  alignas(16) static float x[16], out[16], table[512];
  for (int i = 0; i < 16; ++i) {
    x[i] = 0.25f * (float)(i - 8);
  }
  for (int i = 0; i < 512; ++i) {
    table[i] = (float)i;
  }
  for (const Case& c : kCases) {
    Shape xs = kShapes[c.xs], ts = kShapes[c.ts], os = kShapes[c.os];
    TensorImpl xi(c.xt, xs.dim, xs.sizes, x, xs.order, xs.strides,
                  TensorShapeDynamism::STATIC);
    TensorImpl ti(c.tt, ts.dim, ts.sizes, table, ts.order, ts.strides,
                  TensorShapeDynamism::STATIC);
    TensorImpl oi(c.ot, os.dim, os.sizes, out, os.order, os.strides,
                  TensorShapeDynamism::STATIC);
    for (int i = 0; i < 16; ++i) {
      out[i] = bits_float(0xA5A5A5A5u);
    }
    EValue ex{Tensor(&xi)}, et{Tensor(&ti)}, eo{Tensor(&oi)};
    EValue einv = c.inv_is_int ? EValue((int64_t)c.inv) : EValue(c.inv);
    EValue ezp = c.zp_is_double ? EValue((double)c.zp) : EValue((int64_t)c.zp);
    EValue eqmin{(int64_t)c.qmin}, eqmax{(int64_t)c.qmax};
    EValue* stack[8] = {&ex, &et, &einv, &ezp, &eqmin, &eqmax, &eo, &eo};
    KernelRuntimeContext ctx;
    shim(ctx, Span<EValue*>(stack, c.slots));
    const Error got = ctx.failure_state();
    if (c.want == Error::Ok) {
      float want[6];
      apply(x, want, 6, table, 0.5f, 0, -128, 127);
      control_ok = got == Error::Ok && memcmp(out, want, sizeof want) == 0;
      continue;
    }
    bool same = true;
    for (int i = 0; i < 16; ++i) {
      same &= float_bits(out[i]) == 0xA5A5A5A5u;
    }
    ++result.total;
    result.refused += got == c.want;
    result.untouched &= same;
    if (got != c.want || !same) {
      printf("  MISMATCH quant_lut malformed '%s': error %d (want %d), out "
             "%s\n", c.what, (int)got, (int)c.want,
             same ? "untouched" : "WRITTEN");
    }
  }
}

// The four-lane loops apply and apply_codes take (elementwise.h) against the
// element loops they replace, transcribed here from quant_lut.cpp: every
// length to 37 at every start to 3, apart and in place, and a partly
// overlapping pair, which apply leaves on its element loop. Inputs: each
// configuration's rounding ties k + 1/2 and their neighbours, the special
// values and arbitrary bit patterns; outputs compared as bits.
struct LoopTally {
  uint64_t cases = 0, mismatches = 0, printed = 0;
  uint64_t fast_f32 = 0, fast_codes = 0;  // cases the loops must have served
};

typedef uint32_t __attribute__((__may_alias__)) word_t;

void reference_apply(const float* x, float* out, size_t n, const float* table,
                     float inv, int32_t zero_point, int32_t quant_min,
                     int32_t quant_max) {
  const int32_t low = quant_min - zero_point;
  const int32_t high = quant_max - zero_point;
  const word_t* entries =
      reinterpret_cast<const word_t*>(table) + (zero_point - quant_min);
  word_t* dst = reinterpret_cast<word_t*>(out);
  for (size_t i = 0; i < n; ++i) {
    float b = x[i] * inv;
    b = b == b ? b : 0.0f;
    b = b > -512.0f ? b : -512.0f;
    b = b < 512.0f ? b : 512.0f;
    const float r = (b + 12582912.0f) - 12582912.0f;
    int32_t c = (int32_t)r;
    c = c < low ? low : c;
    c = c > high ? high : c;
    dst[i] = entries[c];
  }
}

void reference_codes(const int8_t* x, int8_t* out, size_t n,
                     const int8_t* table, int32_t quant_min,
                     int32_t quant_max) {
  for (size_t i = 0; i < n; ++i) {
    int32_t c = x[i];
    c = c < quant_min ? quant_min : c;
    c = c > quant_max ? quant_max : c;
    out[i] = table[c - quant_min];
  }
}

uint32_t g_loop_random = 0x9e3779b9u;

uint32_t loop_random() {
  uint32_t x = g_loop_random;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return g_loop_random = x;
}

void loop_mismatch(LoopTally& tally, const char* what, size_t n, size_t at) {
  ++tally.mismatches;
  if (tally.printed++ < 8) {
    printf("  MISMATCH quant_lut loop %s: length %u, start %u\n", what,
           (unsigned)n, (unsigned)at);
  }
}

void check_f32_loops(LoopTally& tally, const float table[256]) {
  struct Config {
    uint32_t inv_bits;
    int16_t zp, qmin, qmax;
  };
  // MobileNetV3's hardswish and hardsigmoid inputs among them; zero points at
  // both ends of the range; the uint8 range.
  static const Config kConfigs[] = {
      {0x43913498u, -12, -128, 127},  // 290.41159
      {0x44a7ece6u, -23, -128, 127},  // 1343.4031
      {0x3f000000u, 0, -128, 127},    {0x40f00000u, 127, -128, 127},
      {0x3c23d70au, -128, -128, 127}, {0x40400000u, 100, 0, 255},
      {0x42c80000u, -1, -127, 127},
  };
  constexpr size_t kPool = 64;
  alignas(16) static float pool[kPool], fast[kPool + 4], ref[kPool + 4],
      buf[kPool + 8];
  for (const Config& config : kConfigs) {
    float inv;
    memcpy(&inv, &config.inv_bits, sizeof inv);
    for (size_t i = 0; i < kPool; ++i) {
      const uint32_t r = loop_random();
      switch (i % 4) {
        case 0: {  // a tie k + 1/2 or a neighbour, within and past the range
          const int k = (int)(r % 600u) - 300;
          float x = ((float)k + 0.5f) / inv;
          for (uint32_t step = (r >> 12) % 3u; step > 0; --step) {
            x = nextafterf(x, (r & 0x100u) ? INFINITY : -INFINITY);
          }
          pool[i] = x;
          break;
        }
        case 1: {
          static const uint32_t kSpecial[10] = {
              0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u,
              0xffc12345u, 0x7f800001u, 0x00000001u, 0x7f7fffffu, 0xff7fffffu};
          memcpy(&pool[i], &kSpecial[(r >> 8) % 10u], sizeof(float));
          break;
        }
        default:
          memcpy(&pool[i], &r, sizeof(float));
          break;
      }
    }
    for (size_t n = 0; n <= 37; ++n) {
      for (size_t at = 0; at < 4; ++at) {
        const float* x = pool + at;
        memset(fast, 0xA5, sizeof fast);
        memset(ref, 0xA5, sizeof ref);
        apply(x, fast, n, table, inv, config.zp, config.qmin, config.qmax);
        reference_apply(x, ref, n, table, inv, config.zp, config.qmin,
                        config.qmax);
        ++tally.cases;
        ++tally.fast_f32;
        if (memcmp(fast, ref, sizeof fast) != 0) {
          loop_mismatch(tally, "fp32 apart", n, at);
        }
        // In place: out is x.
        memcpy(buf, x, n * sizeof(float));
        apply(buf, buf, n, table, inv, config.zp, config.qmin, config.qmax);
        ++tally.cases;
        ++tally.fast_f32;
        if (memcmp(buf, ref, n * sizeof(float)) != 0) {
          loop_mismatch(tally, "fp32 in place", n, at);
        }
        // Partly overlapping: out one element past x, the element loop's.
        if (n > 1) {
          memcpy(buf, x, n * sizeof(float));
          memcpy(fast, x, n * sizeof(float));
          apply(buf, buf + 1, n, table, inv, config.zp, config.qmin,
                config.qmax);
          reference_apply(fast, fast + 1, n, table, inv, config.zp,
                          config.qmin, config.qmax);
          ++tally.cases;
          if (memcmp(buf, fast, (n + 1) * sizeof(float)) != 0) {
            loop_mismatch(tally, "fp32 overlapping", n, at);
          }
        }
      }
    }
  }
}

void check_codes_loops(LoopTally& tally) {
  constexpr size_t kPool = 300;
  alignas(16) static int8_t table[256], pool[kPool], fast[kPool + 8],
      ref[kPool + 8], buf[kPool + 8];
  for (int i = 0; i < 256; ++i) {
    table[i] = (int8_t)loop_random();
  }
  for (size_t i = 0; i < kPool; ++i) {  // every code, then arbitrary ones
    pool[i] = (int8_t)(i < 256 ? i : loop_random());
  }
  struct Range {
    int16_t qmin, qmax;
  };
  // The whole range (the loop's), and two clamped ones (the element loop's).
  static const Range kRanges[3] = {{-128, 127}, {-100, 100}, {-128, 126}};
  for (const Range& range : kRanges) {
    const bool loop = range.qmin == -128 && range.qmax == 127;
    for (size_t n = 0; n <= 37; ++n) {
      for (size_t at = 0; at < 4; ++at) {
        const int8_t* x = pool + at * 61;  // every alignment of x
        memset(fast, 0x5A, sizeof fast);
        memset(ref, 0x5A, sizeof ref);
        apply_codes(x, fast, n, table, range.qmin, range.qmax);
        reference_codes(x, ref, n, table, range.qmin, range.qmax);
        ++tally.cases;
        tally.fast_codes += loop;
        if (memcmp(fast, ref, sizeof fast) != 0) {
          loop_mismatch(tally, "codes apart", n, at);
        }
        memcpy(buf + at, x, n);  // in place, at every alignment
        apply_codes(buf + at, buf + at, n, table, range.qmin, range.qmax);
        ++tally.cases;
        tally.fast_codes += loop;
        if (memcmp(buf + at, ref, n) != 0) {
          loop_mismatch(tally, "codes in place", n, at);
        }
        if (n > 1) {  // out one code past x: the element loop's
          memcpy(buf, x, n);
          memcpy(fast, x, n);
          apply_codes(buf, buf + 1, n, table, range.qmin, range.qmax);
          reference_codes(fast, fast + 1, n, table, range.qmin, range.qmax);
          ++tally.cases;
          if (memcmp(buf, fast, n + 1) != 0) {
            loop_mismatch(tally, "codes overlapping", n, at);
          }
        }
      }
    }
  }
}

// The output over the table: the element loops read each entry
// just before writing the element that may overwrite it, so the four-lane
// loops, which read four entries before storing, must leave these calls to
// them. The table is the output exactly, then the output starts one entry
// further. fp32: x [1,0,1,0] at inverse scale 1, zero point 0, range [0,3],
// the table [10,20,0,0] (the element loops give [20,20,20,20]); int8: the whole
// range's table, codes [-127,-128,-127,-128].
void check_table_aliases(LoopTally& tally) {
  const float x[4] = {1.0f, 0.0f, 1.0f, 0.0f};
  for (int shift = 0; shift < 2; ++shift) {
    alignas(16) float fast[8] = {10, 20, 0, 0, 30, 40, 50, 60};
    alignas(16) float ref[8] = {10, 20, 0, 0, 30, 40, 50, 60};
    apply(x, fast + shift, 4, fast, 1.0f, 0, 0, 3);
    reference_apply(x, ref + shift, 4, ref, 1.0f, 0, 0, 3);
    ++tally.cases;
    if (memcmp(fast, ref, sizeof fast) != 0) {
      loop_mismatch(tally, shift ? "fp32 out one entry into the table"
                                 : "fp32 out over the table", 4, shift);
    }
  }
  const int8_t codes[4] = {-127, -128, -127, -128};
  alignas(16) static int8_t fast[260], ref[260];
  for (int shift = 0; shift < 2; ++shift) {
    for (int i = 0; i < 260; ++i) {
      fast[i] = ref[i] = (int8_t)(i * 7 + 3);
    }
    apply_codes(codes, fast + shift, 4, fast, -128, 127);
    reference_codes(codes, ref + shift, 4, ref, -128, 127);
    ++tally.cases;
    if (memcmp(fast, ref, sizeof fast) != 0) {
      loop_mismatch(tally, shift ? "codes out one entry into the table"
                                 : "codes out over the table", 4, shift);
    }
  }
}

// A zero point outside [quant_min, quant_max]: the shim
// refuses one, but apply is public, and the element loop indexes the table with the
// signed difference; the four-lane loop's base needs quant_min <= zero point.
void check_zero_point_outside(LoopTally& tally, const float table[256]) {
  const float x[8] = {0.0f, 1.0f, 2.0f, 50.0f, 300.0f, -5.0f, 7.0f, 9.0f};
  const int32_t zps[2] = {-10, 265};
  for (int32_t zp : zps) {
    float fast[8], ref[8];
    apply(x, fast, 8, table, 1.0f, zp, 0, 255);
    reference_apply(x, ref, 8, table, 1.0f, zp, 0, 255);
    ++tally.cases;
    if (memcmp(fast, ref, sizeof fast) != 0) {
      loop_mismatch(tally, zp < 0 ? "fp32 zero point below the range"
                                  : "fp32 zero point above the range", 8, 0);
    }
  }
}

}  // namespace

void run_conformance_check() {
  const Stats saved = stats();
  // apply runs the four-lane loop for every case below: their counts go back
  // as they were at the end, as quant_lut's own.
  const elementwise::Counters ew_start = elementwise::counters();
  static float table[256];
  for (int i = 0; i < 256; ++i) {
    table[i] = bits_float(entry(i));
  }
  // The special values after the ties, in quant_lut_golden.h's order.
  static const uint32_t kSpecials[golden::kSpecials] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x7F7FFFFFu,
      0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u};
  Tally tally;
  bool inputs_identical = true;
  for (const golden::Scale& scale : golden::kScales) {
    const float inv = bits_float(scale.inv_bits);
    uint32_t fnv = 2166136261u;
    uint32_t groups = 0;
    uint32_t bit = scale.bits_offset * 8;
    for (int k = golden::kKMin; k <= golden::kKMax; ++k) {
      const float target = (float)k + 0.5f;
      const float x0 = (float)(((double)k + 0.5) / (double)inv);
      // x0, then 1, 2 and 3 ulps up and down, alternating: the first whose
      // product with inv is exactly the tie (fold_quant_lut_pass.py's search).
      float tie = 0.0f;
      bool found = false;
      float up = x0, down = x0;
      for (int step = 0; step <= 3 && !found; ++step) {
        if (step > 0) {
          up = nextafterf(up, INFINITY);
          down = nextafterf(down, -INFINITY);
        }
        const float cands[2] = {up, down};
        for (int which = 0; which < (step == 0 ? 1 : 2) && !found; ++which) {
          if (cands[which] * inv == target) {
            tie = cands[which];
            found = true;
          }
        }
      }
      if (!found) {
        continue;
      }
      ++groups;
      float xs[5];
      xs[2] = tie;
      xs[1] = nextafterf(tie, -INFINITY);
      xs[0] = nextafterf(xs[1], -INFINITY);
      xs[3] = nextafterf(tie, INFINITY);
      xs[4] = nextafterf(xs[3], INFINITY);
      for (int j = 0; j < 5; ++j, ++bit) {
        fnv = fnv1a(fnv, float_bits(xs[j]));
        const int32_t offset = (golden::kBits[bit >> 3] >> (bit & 7)) & 1;
        check_input(tally, table, xs[j], inv, true, k + offset);
      }
    }
    for (int s = 0; s < golden::kSpecials; ++s) {
      const float x = bits_float(kSpecials[s]);
      fnv = fnv1a(fnv, kSpecials[s]);
      check_input(tally, table, x, inv, true, scale.specials[s]);
    }
    if (fnv != scale.input_fnv || groups != scale.groups) {
      inputs_identical = false;
      printf("  MISMATCH quant_lut inputs for inv %08lx: %lu groups, fnv "
             "%08lx (golden %lu, %08lx)\n", (unsigned long)scale.inv_bits,
             (unsigned long)groups, (unsigned long)fnv,
             (unsigned long)scale.groups, (unsigned long)scale.input_fnv);
    }
  }
  Malformed malformed;
  bool control_ok = false;
  check_malformed(malformed, control_ok);
  // The four-lane loops, counted by their own calls: a case they should have
  // served but did not is a mismatch too (their outputs alone cannot show it).
  const elementwise::Counters ew_loops = elementwise::counters();
  LoopTally loops;
  check_f32_loops(loops, table);
  check_codes_loops(loops);
  check_table_aliases(loops);
  check_zero_point_outside(loops, table);
  const uint64_t served_f32 = elementwise::counters().lut_f32 - ew_loops.lut_f32;
  const uint64_t served_codes =
      elementwise::counters().lut_codes - ew_loops.lut_codes;
  const bool served =
      served_f32 == loops.fast_f32 && served_codes == loops.fast_codes;
  if (!served) {
    printf("  MISMATCH quant_lut loops served fp32 %llu of %llu, codes %llu of "
           "%llu\n", (unsigned long long)served_f32,
           (unsigned long long)loops.fast_f32, (unsigned long long)served_codes,
           (unsigned long long)loops.fast_codes);
  }
  elementwise::counters() = ew_start;
  const uint64_t mismatches =
      tally.nearbyint + tally.torch + (inputs_identical ? 0 : 1) +
      (uint64_t)(malformed.total - malformed.refused) +
      (malformed.untouched ? 0 : 1) + (control_ok ? 0 : 1) +
      loops.mismatches + (served ? 0 : 1);
  printf("quant_lut check: %llu case(s), %llu mismatch(es) (nearbyintf %llu, "
         "torch golden %llu, inputs %s), malformed %d/%d refused, out %s, "
         "control %s, loops %llu/%llu fp32 %llu codes %llu\n",
         (unsigned long long)(tally.cases + loops.cases),
         (unsigned long long)mismatches,
         (unsigned long long)tally.nearbyint, (unsigned long long)tally.torch,
         inputs_identical ? "identical" : "DIFFER", malformed.refused,
         malformed.total, malformed.untouched ? "untouched" : "WRITTEN",
         control_ok ? "accepted" : "REFUSED",
         (unsigned long long)(loops.cases - loops.mismatches),
         (unsigned long long)loops.cases, (unsigned long long)served_f32,
         (unsigned long long)served_codes);
  restore_stats(saved);
}

}  // namespace quant_lut
}  // namespace espressif
