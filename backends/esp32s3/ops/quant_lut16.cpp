/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// espressif::lut16 (quant_lut16.h): the kernel, its shim and counters (the
// conformance check is quant_lut16_check.cpp, so a build that never runs it
// -- the story's -- links none of it). Built -ffp-contract=off on the host
// and the device: every multiply and add of the cubic step and of the
// quantize and dequantize steps rounds on its own, as torch's eager ops and
// the mirrors do, and run_parity checks the object code for madd.s/msub.s.
// On the device quant_lut16.lf links this file after every other flash
// object, so the story firmware's code keeps its addresses, and the
// uniform linear, uniform cubic (with and without kTail) and NSAU cubic (with
// and without kOffset) table steps run the hand-scheduled loops of
// quant_lut16_s3.S (the C loops below are the host's and the reference; the
// device check runs both). A table is validated on its first use only, and
// on the device copied into internal RAM then (checked(); forget_tables()
// ends a program's tables).

#include "quant_lut16.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include "quant_rules.h"

#if defined(__XTENSA__)
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#endif

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

#if defined(__XTENSA__)
// What the hand-scheduled loops read (quant_lut16_s3.S; the offsets are
// fixed there). Each loop takes codes already in the table's domain: the
// uniform loops u = c - lo as uint16, the NSAU and uniform tail loops c
// itself as int16.
extern "C" {
struct Lut16LoopArgs {
  const void* data;      //  0: NSAU: the positive half's row of segment 0
  int32_t out_min;       //  4
  int32_t out_max;       //  8
  int32_t sar;           // 12: uniform linear F + G, uniform cubic F
  int32_t mask;          // 16: 2^F - 1
  int32_t pow2g;         // 20: uniform linear 2^G
  int32_t pow2f;         // 24: uniform linear and cubic tail 2^F
  int32_t rnd;           // 28: uniform linear 2^(F+G-1), or 0
  float scale;           // 32: uniform cubic 2^-F
  int32_t m31;           // 36: NSAU 31 - m
  int32_t pow2m;         // 40: NSAU 2^m
  int32_t negoff;        // 44: NSAU bytes from data to the negative half's
                         //     row of segment 0, 0 one-sided
  const float* pow2neg;  // 48: NSAU 2^-sh for sh = 0..15
  int32_t lo;            // 52: uniform cubic tail lo
  int32_t kmax;          // 56: uniform cubic tail K - 1
};
void lut16_uniform_linear_s3(const uint16_t* u, int16_t* y, int n,
                             const Lut16LoopArgs* args);
void lut16_uniform_cubic_s3(const uint16_t* u, int16_t* y, int n,
                            const Lut16LoopArgs* args);
void lut16_uniform_cubic_tail_s3(const int16_t* c, int16_t* y, int n,
                                 const Lut16LoopArgs* args);
void lut16_nsau_cubic_s3(const int16_t* c, int16_t* y, int n,
                         const Lut16LoopArgs* args);
}
#endif

namespace espressif {
namespace lut16 {
namespace {

// GCC at -Os re-materialises float constants on every loop iteration (l32r +
// wfr each); an empty asm keeps each in its register. No-op off Xtensa.
#if defined(__XTENSA__)
#define LUT16_PIN(f) __asm__("" : "+f"(f))
#else
#define LUT16_PIN(f) ((void)0)
#endif

// Elements per pass of apply(): quantize into a code buffer, the table step
// in place, dequantize.
constexpr size_t kChunk = 256;
constexpr float kMagic = 12582912.0f;  // 1.5 * 2^23
constexpr float kCoefMax = 524288.0f;  // 2^19

#if defined(__XTENSA__)
DRAM_ATTR const float kPow2Neg[16] = {
    1.0f,        0.5f,         0.25f,        0.125f,
    0.0625f,     0.03125f,     0.015625f,    0.0078125f,
    0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f,
    0.000244140625f, 0.0001220703125f, 6.103515625e-05f, 3.0517578125e-05f};
#endif

uint64_t g_calls = 0;
uint64_t g_elements = 0;
uint64_t g_refusals = 0;

// checked()'s memory: the tables accepted since the last forget_tables(), as
// the program gave them, and each one's internal-RAM copy (or nullptr: read
// in place). Past kRemembered tables, a table is validated on every call.
struct Remembered {
  Table table;
  void* copy;
};
constexpr int kRemembered = 16;
#if defined(__XTENSA__)
// Internal RAM the copies may hold at once (two tables at the 8 KB default
// budget of lut_fit.py).
constexpr size_t kCopyBudget = 16384;
#endif
Remembered g_remembered[kRemembered];
int g_count = 0;
size_t g_held = 0;
uint64_t g_validations = 0;
uint64_t g_reused = 0;
uint64_t g_copied = 0;
uint64_t g_copied_bytes = 0;
uint64_t g_in_place = 0;

inline int32_t clampi(int32_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// 2^-s for 0 <= s <= 126, from its bits.
inline float pow2neg(int32_t s) {
  const uint32_t bits = (uint32_t)(127 - s) << 23;
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

// The scheme word's base scheme and flags.
inline int32_t base_of(int32_t scheme) {
  return scheme & kBaseMask;
}

bool is_uniform(int32_t scheme) {
  return base_of(scheme) == kUniformLinear || base_of(scheme) == kUniformCubic;
}

bool is_nsau(int32_t scheme) {
  return base_of(scheme) == kNsauLinear || base_of(scheme) == kNsauCubic;
}

bool is_linear(int32_t scheme) {
  return base_of(scheme) == kUniformLinear || base_of(scheme) == kNsauLinear;
}

bool has_tail(int32_t scheme) {
  return (scheme & kTail) != 0;
}

// The NSAU segment of |code| a: (sh << m) + (a >> sh).
inline int32_t nsau_k(uint32_t a, int32_t m) {
  int32_t sh = 31 - __builtin_clz(a | 1u) - m;
  sh = sh < 0 ? 0 : sh;
  return (int32_t)(((uint32_t)sh << m) + (a >> sh));
}

// The NSAU index of a code in the domain (mirrors.lut16_index): k is
// the entry it reads, its segment plus the bias of its sign.
struct NsauIndex {
  int32_t k, t, sh;
};

inline NsauIndex nsau_index(int32_t c, int32_t m, const NsauLayout& l) {
  const uint32_t a = (uint32_t)(c < 0 ? -c : c);
  int32_t sh = 31 - __builtin_clz(a | 1u) - m;
  sh = sh < 0 ? 0 : sh;
  const uint32_t q = a >> sh;
  NsauIndex index;
  index.k = (int32_t)(((uint32_t)sh << m) + q) + (c < 0 ? l.neg_bias : l.pos_bias);
  index.t = (int32_t)(a - (q << sh));
  index.sh = sh;
  return index;
}

// The layout run() and step() read: the NSAU biases, or zeros.
NsauLayout layout_of(const Table& t) {
  if (!is_nsau(t.scheme)) {
    return NsauLayout{0, 0, 0};
  }
  return nsau_layout(t.scheme, t.lo, t.hi, t.shift);
}

// Segments of a uniform table.
inline int32_t uniform_segments(const Table& t) {
  return ((t.hi - t.lo) >> t.shift) + 1;
}

// A kTail table's segment and offset of a code (any int16): k = clamp(u >>
// F, 0, K - 1), t = u - (k << F), u = c - lo (an arithmetic shift: a floor).
struct TailIndex {
  int32_t k, t;
};

inline TailIndex tail_index(const Table& tab, int32_t c) {
  const int32_t u = c - tab.lo;
  int32_t k = u >> tab.shift;
  const int32_t last = uniform_segments(tab) - 1;
  k = k < 0 ? 0 : (k > last ? last : k);
  TailIndex index;
  index.k = k;
  index.t = u - k * (int32_t)(1u << tab.shift);
  return index;
}

// The largest reach of a kTail table's edge segments, in codes from their
// first code (mirrors.lut16_tail_reach): below, |t| down to code
// -32768; above, t up to 32767 -- each at least the segment's 2^F - 1.
void tail_reach(const Table& t, int32_t* below, int32_t* above) {
  const int32_t width = (int32_t)(1u << t.shift) - 1;
  const int32_t last = t.lo + (uniform_segments(t) - 1) * (int32_t)(1u << t.shift);
  *below = t.lo + 32768 > 0 ? t.lo + 32768 : 0;
  *above = 32767 - last > width ? 32767 - last : width;
}

// One cubic row at offset t of a 2^F-code segment.
inline int32_t cubic(const float* row, int32_t t, int32_t F) {
  const float tf = (float)t * pow2neg(F);
  float y = row[2] + tf * row[3];
  y = row[1] + tf * y;
  y = row[0] + tf * y;
  const float r = (y + kMagic) - kMagic;
  return (int32_t)r;  // |r| < 2^22: validate() bounds the coefficients
}

// V[k] * 2^F + (V[k+1] - V[k]) * t + 2^(F+G-1), then >> (F + G). validate()
// keeps every term and sum inside int32; the difference is formed modulo
// 2^32, which only matters when F = 0 and t is 0.
inline int32_t uniform_linear_at(const int32_t* V, int32_t k, int32_t t,
                                 int32_t F, int32_t G) {
  const int32_t v0 = V[k];
  const int32_t dv = (int32_t)((uint32_t)V[k + 1] - (uint32_t)v0);
  const int32_t sh = F + G;
  const int32_t rnd = sh > 0 ? (int32_t)(1u << (sh - 1)) : 0;
  return (v0 * (int32_t)(1u << F) + dv * t + rnd) >> sh;
}

inline int32_t uniform_linear(const int32_t* V, uint32_t u, int32_t F,
                              int32_t G) {
  return uniform_linear_at(V, (int32_t)(u >> F),
                           (int32_t)(u & ((1u << F) - 1u)), F, G);
}

inline int32_t nsau_linear(const int32_t* V, const NsauIndex& i, int32_t G) {
  const int32_t v0 = V[i.k];
  const int32_t dv = V[i.k + 1] - v0;
  const int32_t rnd = G > 0 ? (int32_t)(1u << (G - 1)) : 0;
  return (v0 + ((dv * i.t) >> i.sh) + rnd) >> G;
}

// The table step of one code in the domain: [lo, hi], or int16 with kTail.
inline int32_t step(const Table& t, int32_t c, const NsauLayout& l) {
  int32_t y = 0;
  if (has_tail(t.scheme)) {
    const TailIndex i = tail_index(t, c);
    y = base_of(t.scheme) == kUniformLinear
            ? uniform_linear_at(static_cast<const int32_t*>(t.data), i.k, i.t,
                                t.shift, t.G)
            : cubic(static_cast<const float*>(t.data) + 4 * i.k, i.t, t.shift);
    return clampi(y, t.out_min, t.out_max);
  }
  switch (base_of(t.scheme)) {
    case kDirect:
      y = static_cast<const int16_t*>(t.data)[c - t.lo];
      break;
    case kUniformLinear:
      y = uniform_linear(static_cast<const int32_t*>(t.data),
                         (uint32_t)(c - t.lo), t.shift, t.G);
      break;
    case kUniformCubic: {
      const uint32_t u = (uint32_t)(c - t.lo);
      const float* rows = static_cast<const float*>(t.data);
      y = cubic(rows + 4 * (u >> t.shift),
                (int32_t)(u & ((1u << t.shift) - 1u)), t.shift);
      break;
    }
    case kNsauLinear:
      y = nsau_linear(static_cast<const int32_t*>(t.data),
                      nsau_index(c, t.shift, l), t.G);
      break;
    case kNsauCubic: {
      const NsauIndex i = nsau_index(c, t.shift, l);
      y = cubic(static_cast<const float*>(t.data) + 4 * i.k, i.t, i.sh);
      break;
    }
  }
  return clampi(y, t.out_min, t.out_max);
}

// Whether run() takes u = c - lo (uint16) rather than the code itself.
bool takes_u(int32_t scheme) {
  return is_uniform(scheme) && !has_tail(scheme);
}

// The table step over a buffer of codes in the domain: u = c - lo (uint16)
// for the uniform schemes without kTail, c (int16) otherwise. in and out may
// alias. The device runs the uniform (cubic with and without kTail) and NSAU
// cubic schemes in quant_lut16_s3.S, so it compiles only the direct, NSAU
// linear and uniform linear tail loops here.
void run_c(const Table& t, const int16_t* in, int16_t* out, size_t n) {
  if (t.scheme == kDirect) {
    const int16_t* T = static_cast<const int16_t*>(t.data);
    for (size_t i = 0; i < n; ++i) {
      out[i] = (int16_t)clampi(T[in[i] - t.lo], t.out_min, t.out_max);
    }
    return;
  }
  const NsauLayout l = layout_of(t);
  if (base_of(t.scheme) == kNsauLinear) {
    const int32_t* V = static_cast<const int32_t*>(t.data);
    for (size_t i = 0; i < n; ++i) {
      const int32_t y = nsau_linear(V, nsau_index(in[i], t.shift, l), t.G);
      out[i] = (int16_t)clampi(y, t.out_min, t.out_max);
    }
    return;
  }
  if (t.scheme == (kUniformLinear | kTail)) {
    for (size_t i = 0; i < n; ++i) {
      out[i] = (int16_t)step(t, in[i], l);
    }
    return;
  }
#if !defined(__XTENSA__)
  if (takes_u(t.scheme)) {
    const uint16_t* u = reinterpret_cast<const uint16_t*>(in);
    for (size_t i = 0; i < n; ++i) {
      out[i] = (int16_t)step(t, (int32_t)u[i] + t.lo, l);
    }
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    out[i] = (int16_t)step(t, in[i], l);
  }
#endif
}

void run(const Table& t, const int16_t* in, int16_t* out, size_t n) {
#if defined(__XTENSA__)
  Lut16LoopArgs a;
  memset(&a, 0, sizeof a);
  a.data = t.data;
  a.out_min = t.out_min;
  a.out_max = t.out_max;
  const uint16_t* u = reinterpret_cast<const uint16_t*>(in);
  switch (t.scheme) {
    case kUniformLinear:
      a.sar = t.shift + t.G;
      a.mask = (int32_t)((1u << t.shift) - 1u);
      a.pow2g = (int32_t)(1u << t.G);
      a.pow2f = (int32_t)(1u << t.shift);
      a.rnd = a.sar > 0 ? (int32_t)(1u << (a.sar - 1)) : 0;
      lut16_uniform_linear_s3(u, out, (int)n, &a);
      return;
    case kUniformCubic:
      a.sar = t.shift;
      a.mask = (int32_t)((1u << t.shift) - 1u);
      a.scale = pow2neg(t.shift);
      lut16_uniform_cubic_s3(u, out, (int)n, &a);
      return;
    case kUniformCubic | kTail:
      a.sar = t.shift;
      a.pow2f = (int32_t)(1u << t.shift);
      a.scale = pow2neg(t.shift);
      a.lo = t.lo;
      a.kmax = uniform_segments(t) - 1;
      lut16_uniform_cubic_tail_s3(in, out, (int)n, &a);
      return;
    case kNsauCubic:
    case kNsauCubic | kOffset: {
      // Row k + bias: the loop adds negoff for c < 0 to a base biased for
      // c >= 0 (an address outside the table when the bias is negative;
      // only the sum is ever read).
      const NsauLayout l = layout_of(t);
      a.data = static_cast<const float*>(t.data) + 4 * l.pos_bias;
      a.m31 = 31 - t.shift;
      a.pow2m = (int32_t)(1u << t.shift);
      a.negoff = (l.neg_bias - l.pos_bias) * 16;
      a.pow2neg = kPow2Neg;
      lut16_nsau_cubic_s3(in, out, (int)n, &a);
      return;
    }
    default:
      break;
  }
#endif
  run_c(t, in, out, n);
}

// Codes in [low, high] -> the buffer run() takes (kTail: any int16 code).
void to_domain_codes(const Table& t, const int16_t* in, int16_t* buf,
                     size_t n, bool clamp) {
  const bool tail = has_tail(t.scheme);
  const int32_t low = clamp && !tail ? t.lo : -32768;
  const int32_t high = clamp && !tail ? t.hi : 32767;
  if (takes_u(t.scheme)) {
    uint16_t* u = reinterpret_cast<uint16_t*>(buf);
    for (size_t i = 0; i < n; ++i) {
      u[i] = (uint16_t)(clampi(in[i], low, high) - t.lo);
    }
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    buf[i] = (int16_t)clampi(in[i], low, high);
  }
}

// The quantize step (quant_lut::apply's, with int16 rails), writing r + add
// for r clamped to [low, high]: the code itself (add = zero point) or the
// uniform schemes' u = code - lo (add = zero point - lo, stored as uint16).
void quantise(const float* x, int16_t* buf, size_t n, float inv_scale,
              int32_t low, int32_t high, int32_t add) {
  float inv = inv_scale;
  float lo = -65536.0f;
  float hi = 65536.0f;
  float magic = kMagic;
  float zero = 0.0f;
  LUT16_PIN(inv);
  LUT16_PIN(lo);
  LUT16_PIN(hi);
  LUT16_PIN(magic);
  LUT16_PIN(zero);
  uint16_t* dst = reinterpret_cast<uint16_t*>(buf);
  for (size_t i = 0; i < n; ++i) {
    // NaN: 0, the zero point's code (quant_rules.h); infinities: the rails.
    float b = quant_scaled(x[i] * inv, zero);
    b = b > lo ? b : lo;
    b = b < hi ? b : hi;
    const float r = (b + magic) - magic;
    int32_t c = (int32_t)r;
    c = c < low ? low : c;
    c = c > high ? high : c;
    dst[i] = (uint16_t)(c + add);
  }
}

void dequantise(const int16_t* y, float* out, size_t n, int32_t zero_point,
                float scale) {
  float s = scale;
  LUT16_PIN(s);
  for (size_t i = 0; i < n; ++i) {
    out[i] = (float)((int32_t)y[i] - zero_point) * s;
  }
}

// One linear segment's int32 bound (mirrors.lut16_linear_safe):
// uniform |V0| * 2^F + |dV| * reach + 2^(F+G-1) < 2^31 (reach 2^F - 1, or a
// tail's edge reach), NSAU |dV| * (2^sh - 1) < 2^31 and |V0| + |dV| +
// 2^(G-1) < 2^31.
bool segment_safe(const int32_t* V, int32_t at, bool uniform, int32_t sh,
                  int64_t reach, int64_t rnd) {
  const int64_t limit = (int64_t)1 << 31;
  int64_t v0 = V[at];
  int64_t dv = (int64_t)V[at + 1] - v0;
  v0 = v0 < 0 ? -v0 : v0;
  dv = dv < 0 ? -dv : dv;
  if (uniform) {
    return (v0 << sh) + dv * reach + rnd < limit;
  }
  return dv * (((int64_t)1 << sh) - 1) < limit && v0 + dv + rnd < limit;
}

// mirrors.lut16_linear_safe, in int64: |V| <= 2^30, and every segment
// the index reaches within segment_safe's bound.
bool linear_safe(const Table& t) {
  const int32_t* V = static_cast<const int32_t*>(t.data);
  for (int32_t i = 0; i < t.count; ++i) {
    if (V[i] > (1 << 30) || V[i] < -(1 << 30)) {
      return false;
    }
  }
  if (base_of(t.scheme) == kUniformLinear) {
    const int32_t shift = t.shift + t.G;
    const int64_t rnd = shift > 0 ? (int64_t)1 << (shift - 1) : 0;
    const int32_t segments = t.count - 1;
    int32_t below = 0, above = 0;
    if (has_tail(t.scheme)) {
      tail_reach(t, &below, &above);
    }
    for (int32_t k = 0; k < segments; ++k) {
      int64_t reach = ((int64_t)1 << t.shift) - 1;
      reach = k == 0 && below > reach ? below : reach;
      reach = k == segments - 1 && above > reach ? above : reach;
      if (!segment_safe(V, k, true, t.shift, reach, rnd)) {
        return false;
      }
    }
    return true;
  }
  // NSAU: each half's segments k0..k1 at entries k + bias
  const int64_t rnd = t.G > 0 ? (int64_t)1 << (t.G - 1) : 0;
  const NsauLayout l = layout_of(t);
  for (int half = 0; half < 2; ++half) {
    int32_t k0 = 0, k1 = -1, bias = 0;
    if (t.scheme & kOffset) {
      if (half == 0 && t.hi >= 0) {
        k0 = nsau_k((uint32_t)(t.lo > 0 ? t.lo : 0), t.shift);
        k1 = nsau_k((uint32_t)t.hi, t.shift);
        bias = l.pos_bias;
      } else if (half == 1 && t.lo < 0) {
        k0 = nsau_k((uint32_t)(-t.hi > 1 ? -t.hi : 1), t.shift);
        k1 = nsau_k((uint32_t)(-t.lo), t.shift);
        bias = l.neg_bias;
      }
    } else if (half == 0 || t.lo < 0) {
      k1 = nsau_half(t.lo, t.hi, t.shift) - 1;
      bias = half == 0 ? l.pos_bias : l.neg_bias;
    }
    for (int32_t k = k0; k <= k1; ++k) {
      int32_t sh = (k >> t.shift) - 1;
      sh = sh < 0 ? 0 : sh;
      if (!segment_safe(V, k + bias, false, sh, 0, rnd)) {
        return false;
      }
    }
  }
  return true;
}

// A kTail cubic table's edge rows below 2^22 everywhere past [lo, hi]:
// |c0| + |c1| T + |c2| T^2 + |c3| T^3 < 2^22, T = reach / 2^F
// (mirrors.lut16_cubic_safe, in double, the same products).
bool tail_cubic_safe(const Table& t) {
  const float* C = static_cast<const float*>(t.data);
  int32_t below = 0, above = 0;
  tail_reach(t, &below, &above);
  const float* rows[2] = {C, C + 4 * (t.count - 1)};
  const int32_t reach[2] = {below, above};
  for (int e = 0; e < 2; ++e) {
    const double T = (double)reach[e] / (double)(1u << t.shift);
    const double T2 = T * T;
    const double bound = std::fabs((double)rows[e][0]) +
                         std::fabs((double)rows[e][1]) * T +
                         std::fabs((double)rows[e][2]) * T2 +
                         std::fabs((double)rows[e][3]) * (T2 * T);
    if (!(bound < 4194304.0)) {
      return false;
    }
  }
  return true;
}

// Contiguous strides in the default dim order (quant_lut.cpp's dense).
bool dense(const Tensor& tensor) {
  const auto order = tensor.dim_order();
  for (size_t dim = 0; dim < order.size(); ++dim) {
    if (order[dim] != dim) {
      return false;
    }
  }
  ssize_t expected = 1;
  for (ssize_t dim = tensor.dim(); dim-- > 0;) {
    if (tensor.size(dim) > 1 && tensor.strides()[dim] != expected) {
      return false;
    }
    expected *= tensor.size(dim);
  }
  return true;
}

void refuse(KernelRuntimeContext& ctx, Error error) {
  ++g_refusals;
  ctx.fail(error);
}

bool in_int16(int64_t v) {
  return v >= -32768 && v <= 32767;
}

bool exact_float_scale(double v) {
  return std::isfinite(v) && v > 0.0 && (double)(float)v == v;
}

}  // namespace

int32_t nsau_half(int32_t lo, int32_t hi, int32_t m) {
  uint32_t a = (uint32_t)(lo < 0 ? -lo : lo);
  const uint32_t b = (uint32_t)(hi < 0 ? -hi : hi);
  a = a > b ? a : b;
  a = a > 1u ? a : 1u;
  int32_t sh = 31 - __builtin_clz(a) - m;
  sh = sh < 0 ? 0 : sh;
  return (int32_t)(((uint32_t)sh << m) + (a >> sh) + 1u);
}

NsauLayout nsau_layout(int32_t scheme, int32_t lo, int32_t hi, int32_t m) {
  const int32_t extra = base_of(scheme) == kNsauLinear ? 1 : 0;
  NsauLayout l;
  if ((scheme & kOffset) == 0) {
    const int32_t half = nsau_half(lo, hi, m) + extra;
    l.pos_bias = 0;
    l.neg_bias = half;
    l.entries = half * (lo < 0 ? 2 : 1);
    return l;
  }
  int32_t pos = 0, neg = 0, p0 = 0, n0 = 0;
  if (hi >= 0) {
    p0 = nsau_k((uint32_t)(lo > 0 ? lo : 0), m);
    pos = nsau_k((uint32_t)hi, m) - p0 + 1 + extra;
  }
  if (lo < 0) {
    n0 = nsau_k((uint32_t)(-hi > 1 ? -hi : 1), m);
    neg = nsau_k((uint32_t)(-lo), m) - n0 + 1 + extra;
  }
  l.pos_bias = -p0;
  l.neg_bias = pos - n0;
  l.entries = pos + neg;
  return l;
}

// What a scheme word's flags may not be, by name (mirrors.
// lut16_scheme_error), or nullptr.
static const char* scheme_error(int32_t scheme) {
  const int32_t base = base_of(scheme);
  if (scheme < 0 || base > kNsauCubic || (scheme & ~(kBaseMask | kOffset | kTail)) != 0) {
    return "scheme";
  }
  if (scheme & kTail) {
    return base == kDirect ? "tail on a direct table"
           : is_nsau(base) ? "tail on an NSAU table"
                           : nullptr;
  }
  if ((scheme & kOffset) != 0 && !is_nsau(base)) {
    return base == kDirect ? "offset on a direct table" : "offset on a uniform table";
  }
  return nullptr;
}

const char* validate(const Table& t) {
  if (const char* why = scheme_error(t.scheme)) {
    return why;
  }
  if (t.data == nullptr) {
    return "no table";
  }
  if (!in_int16(t.lo) || !in_int16(t.hi) || t.lo > t.hi) {
    return "domain";
  }
  if (!in_int16(t.out_min) || !in_int16(t.out_max) || t.out_min >= t.out_max) {
    return "output range";
  }
  if (t.G < 0 || t.G > 16) {
    return "G";
  }
  int32_t want = 0;
  if (is_nsau(t.scheme)) {
    if (t.lo < -32767 || t.shift < 0 || t.shift > 8) {
      return t.lo < -32767 ? "domain" : "shift";
    }
    want = nsau_layout(t.scheme, t.lo, t.hi, t.shift).entries;
  } else {
    if (t.shift < 0 || t.shift > 16) {
      return "shift";
    }
    const int32_t segments = ((t.hi - t.lo) >> t.shift) + 1;
    want = t.scheme == kDirect ? t.hi - t.lo + 1
           : base_of(t.scheme) == kUniformLinear ? segments + 1 : segments;
  }
  if (t.count != want) {
    return "table size";
  }
  if (is_linear(t.scheme)) {
    if (!linear_safe(t)) {
      return has_tail(t.scheme) ? "tail outside int32" : "nodes outside int32";
    }
  } else if (t.scheme != kDirect) {
    const float* c = static_cast<const float*>(t.data);
    for (int32_t i = 0; i < 4 * t.count; ++i) {
      if (!(std::fabs(c[i]) <= kCoefMax)) {  // NaN fails too
        return "coefficient";
      }
    }
    if (has_tail(t.scheme) && !tail_cubic_safe(t)) {
      return "tail past 2^22";
    }
  }
  return nullptr;
}

namespace {

bool same_table(const Table& a, const Table& b) {
  return a.data == b.data && a.scheme == b.scheme && a.lo == b.lo &&
         a.hi == b.hi && a.shift == b.shift && a.G == b.G &&
         a.out_min == b.out_min && a.out_max == b.out_max &&
         a.count == b.count;
}

// The bytes of a table's data: count entries, nodes or four-float rows.
size_t table_bytes(const Table& t) {
  const size_t each = t.scheme == kDirect ? 2 : is_linear(t.scheme) ? 4 : 16;
  return each * (size_t)(t.count > 0 ? t.count : 0);
}

}  // namespace

const char* checked(const Table& t, Table* use) {
  for (int i = 0; i < g_count; ++i) {
    if (same_table(g_remembered[i].table, t)) {
      *use = t;
      if (g_remembered[i].copy != nullptr) {
        use->data = g_remembered[i].copy;
      }
      ++g_reused;
      return nullptr;
    }
  }
  ++g_validations;
  Table candidate = t;
  void* copy = nullptr;
#if defined(__XTENSA__)
  // A table in flash or PSRAM goes to the internal heap at first use, when
  // the copies fit their budget; validate() then reads the copy, so what the
  // kernel evaluates is exactly what was checked. Otherwise: in place.
  const size_t bytes = table_bytes(t);
  if (g_count < kRemembered && t.data != nullptr && !esp_ptr_internal(t.data) &&
      bytes > 0 && g_held + bytes <= kCopyBudget) {
    copy = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (copy != nullptr) {
      memcpy(copy, t.data, bytes);
      candidate.data = copy;
    }
  }
#endif
  const char* why = validate(candidate);
  if (why != nullptr) {
#if defined(__XTENSA__)
    heap_caps_free(copy);
#endif
    return why;
  }
  *use = candidate;
  if (g_count < kRemembered) {
    g_remembered[g_count].table = t;
    g_remembered[g_count].copy = copy;
    ++g_count;
    if (copy != nullptr) {
      g_held += table_bytes(t);
      ++g_copied;
      g_copied_bytes += table_bytes(t);
    } else {
      ++g_in_place;
    }
  }
  return nullptr;
}

void forget_tables() {
  for (int i = 0; i < g_count; ++i) {
#if defined(__XTENSA__)
    heap_caps_free(g_remembered[i].copy);
#endif
    g_remembered[i].copy = nullptr;
  }
  g_count = 0;
  g_held = 0;
}

int32_t eval_code(const Table& t, int32_t code) {
  return has_tail(t.scheme) ? step(t, clampi(code, -32768, 32767), layout_of(t))
                            : step(t, clampi(code, t.lo, t.hi), layout_of(t));
}

void domain(const Table& t, int32_t quant_min, int32_t quant_max,
            int32_t* low, int32_t* high) {
  const int32_t lo = has_tail(t.scheme) ? -32768 : t.lo;
  const int32_t hi = has_tail(t.scheme) ? 32767 : t.hi;
  *low = clampi(quant_min, lo, hi);
  *high = clampi(quant_max, lo, hi);
}

void eval_codes_in_domain(const Table& t, const int16_t* in, int16_t* out,
                          size_t n) {
  if (!takes_u(t.scheme)) {
    run(t, in, out, n);
    return;
  }
  int16_t buf[kChunk];
  for (size_t i = 0; i < n; i += kChunk) {
    const size_t m = n - i < kChunk ? n - i : kChunk;
    to_domain_codes(t, in + i, buf, m, false);
    run(t, buf, out + i, m);
  }
}

void eval_codes(const Table& t, const int16_t* in, int16_t* out, size_t n) {
  int16_t buf[kChunk];
  for (size_t i = 0; i < n; i += kChunk) {
    const size_t m = n - i < kChunk ? n - i : kChunk;
    to_domain_codes(t, in + i, buf, m, true);
    run(t, buf, out + i, m);
  }
}

void apply(const Table& t, const float* x, float* out, size_t n,
           float inv_scale, int32_t zero_point, int32_t quant_min,
           int32_t quant_max, float out_scale, int32_t out_zero_point) {
  // clamp(clamp(c, qmin, qmax), lo, hi) == clamp(c, low, high).
  int32_t low, high;
  domain(t, quant_min, quant_max, &low, &high);
  const int32_t add = takes_u(t.scheme) ? zero_point - t.lo : zero_point;
  int16_t buf[kChunk];
  for (size_t i = 0; i < n; i += kChunk) {
    const size_t m = n - i < kChunk ? n - i : kChunk;
    quantise(x + i, buf, m, inv_scale, low - zero_point, high - zero_point,
             add);
    run(t, buf, buf, m);
    dequantise(buf, out + i, m, out_zero_point, out_scale);
  }
}

void shim(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Fifteen arguments, out, and the returned alias of out.
  if (st.size() != 17) {
    refuse(ctx, Error::InvalidProgram);
    return;
  }
  // Types first: an EValue accessor aborts on the wrong tag. Every int
  // argument (zero points, ranges, scheme, lo, hi, shift, G) must fit int16.
  static const uint8_t kInts[] = {3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14};
  bool ok = st[0]->isTensor() && st[1]->isTensor() && st[2]->isDouble() &&
            st[11]->isDouble() && st[15]->isTensor();
  int64_t v[15] = {};
  for (uint8_t i : kInts) {
    ok = ok && st[i]->isInt();
    if (ok) {
      v[i] = st[i]->toInt();
      ok = in_int16(v[i]);
    }
  }
  if (!ok) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const Tensor& x = st[0]->toTensor();
  const Tensor& table = st[1]->toTensor();
  const double inv_scale = st[2]->toDouble();
  const double out_scale = st[11]->toDouble();
  Tensor& out = st[15]->toTensor();
  // v: 3 zero point, 4/5 quant range, 6 scheme, 7/8 lo/hi, 9 shift, 10 G,
  // 12 out zero point, 13/14 out range.
  // the table's dtype by the base scheme, so a bad flag is validate()'s
  const ScalarType want = base_of((int32_t)v[6]) == kDirect ? ScalarType::Short
                          : is_linear((int32_t)v[6])        ? ScalarType::Int
                                                            : ScalarType::Float;
  const int want_dim = want == ScalarType::Float ? 2 : 1;
  ok = x.scalar_type() == ScalarType::Float &&
       out.scalar_type() == ScalarType::Float && dense(x) && dense(out) &&
       v[4] < v[5] && v[4] <= v[3] && v[3] <= v[5] && v[13] <= v[12] &&
       v[12] <= v[14] && exact_float_scale(inv_scale) &&
       exact_float_scale(out_scale) && table.scalar_type() == want &&
       table.dim() == want_dim && (want_dim == 1 || table.size(1) == 4) &&
       dense(table);
  const char* reason = ok ? nullptr : "arguments";
  Table t, use;
  if (ok) {
    t.scheme = (int32_t)v[6];
    t.lo = (int32_t)v[7];
    t.hi = (int32_t)v[8];
    t.shift = (int32_t)v[9];
    t.G = (int32_t)v[10];
    t.out_min = (int32_t)v[13];
    t.out_max = (int32_t)v[14];
    t.data = table.const_data_ptr();
    t.count = (int32_t)table.size(0);
    reason = checked(t, &use);
  }
  if (reason != nullptr) {
    printf("  [lut16] refused: %s (scheme %d, table %d-D of dtype %d)\n",
           reason, (int)v[6], (int)table.dim(), (int)table.scalar_type());
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  if (executorch::runtime::resize_tensor(out, x.sizes()) != Error::Ok ||
      !dense(out)) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  apply(use, x.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
        (size_t)x.numel(), (float)inv_scale, (int32_t)v[3], (int32_t)v[4],
        (int32_t)v[5], (float)out_scale, (int32_t)v[12]);
  ++g_calls;
  g_elements += (uint64_t)x.numel();
}

Stats stats() {
  return {g_calls,  g_elements, g_refusals,     g_validations,
          g_reused, g_copied,   g_copied_bytes, g_in_place};
}

void restore_stats(const Stats& saved) {
  g_calls = saved.calls;
  g_elements = saved.elements;
  g_refusals = saved.refusals;
  g_validations = saved.validations;
  g_reused = saved.reused;
  g_copied = saved.copied;
  g_copied_bytes = saved.copied_bytes;
  g_in_place = saved.in_place;
}

void report_dispatch() {
  if (g_calls != 0 || g_refusals != 0) {
    printf("lut16 dispatch: calls %llu, elements %llu, refusals %llu\n",
           (unsigned long long)g_calls, (unsigned long long)g_elements,
           (unsigned long long)g_refusals);
  }
  if (g_validations != 0 || g_reused != 0) {
    printf("lut16 tables: validated %llu, reused %llu, copied to internal RAM "
           "%llu (%llu B), read in place %llu\n",
           (unsigned long long)g_validations, (unsigned long long)g_reused,
           (unsigned long long)g_copied, (unsigned long long)g_copied_bytes,
           (unsigned long long)g_in_place);
  }
}

}  // namespace lut16
}  // namespace espressif
