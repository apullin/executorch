/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The static int8 convolution path (esp32_conv.h says what and why).
//
// Built -ffp-contract=off (the affine column math is a fixed sequence of IEEE
// operations, as in esp32_asym.cpp) and -O2 whatever the build: the loops are
// short and branchy at -Os (esp32_float.cpp's reasoning). esp32_conv.lf links
// the object after every other flash object, so it moves no hot function.

#include "esp32_conv.h"

#include "esp32_codes.h"
#include "esp32_kernels.h"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#if ESP32_CONV_EPI_CHECK  // host proof builds: conv_epilogue_check, or a host runner built with it
#include <mutex>
#include <string>
#include <unordered_set>
#endif

namespace esp32_conv {
using namespace esp32_codes;
namespace {

inline uint64_t align16(uint64_t bytes) {
  return (bytes + 15u) & ~uint64_t{15u};
}

/// Outputs computed per run before their epilogue (finish).
constexpr uint32_t kRun = 64;

// ---- the epilogue ---------------------------------------------------------
// One output element from its corrected accumulator: esp32_asym.cpp's
// add_group with one group (from the bias, or +0.0f), esp32_backend.cpp's
// clamp_output, then requantize_code / requantize / nothing. Where the output
// is a code (codes out, or fp32 requantised from its code), the integer
// epilogue below computes that code without a float operation, and the float
// sequence for any element it cannot prove.

enum OutMode { kOutCodes = 0, kOutRequant = 1, kOutFloat = 2 };

struct Epi {
  float sx;
  // The clamp's bounds: the op's, or -inf/+inf without FLAG_CLAMP, which
  // compare false against every value (NaN and -0.0 included): the identity.
  float lo, hi;
  float out_inv, out_scale;
  int32_t zo, qmin, qmax;
  // The integer epilogue's terms for the op (fixed_epi): whether it may run,
  // the codes the clamp allows -- code(lo) and code(hi), by the float
  // sequence -- the bound on |acc|, and fixed_of's per-op factors:
  // fl(sx * out_inv), and the guard's g~, |c~| and constant coefficients.
  bool fixed;
  int32_t code_lo, code_hi;
  float amax;
  float sx_out, guard_g, guard_c, guard_0;
};

// ---- the integer epilogue ---------------------------------------------------
// For codes out (and fp32 requantised from its code), the float sequence is
//
//   v = clamp(fl(b + fl(fl(fl(acc) * sx) * s)), lo, hi)
//   q = clamp(rne(clamp(fl(v * out_inv), +-kRail)) + zo, qmin, qmax)
//
// a function of the integer acc alone. With sx, s and out_inv positive and
// finite, every step is monotone non-decreasing (IEEE rounding is monotone,
// and so are the clamps and the rounding to an integer), so q is a monotone
// step function of acc. Two consequences:
//
//   1. The clamp to [lo, hi] (lo <= hi) commutes into the codes: with
//      phi(v) = q's sequence from v, phi(clamp(v, lo, hi)) =
//      clamp(phi(v), phi(lo), phi(hi)) for any monotone phi. phi(lo) and
//      phi(hi) are computed once per op by the same float sequence (code_lo,
//      code_hi; -inf and +inf give qmin and qmax), and the clamp to
//      [qmin, qmax] is inside them.
//   2. Away from the rounding boundaries, any value close enough to the
//      float sequence's rounds the same way. The integer form computes
//
//        Y = acc * G + C,  G = floor(g~ * 2^32),
//        C = floor(c~ * 2^32) + zo * 2^32 + 2^31
//
//      in int64, with g~ = fl(s * fl(sx * out_inv)) and c~ = fl(b * out_inv):
//      Y / 2^32 approximates y + zo + 1/2 for the real y = (acc*sx*s + b) *
//      out_inv, so q = floor(Y / 2^32) (Y's high word), clamped to
//      [code_lo, code_hi] -- unless Y's low word f is within `guard` of a
//      multiple of 2^32 (a rounding boundary), where the element takes the
//      float sequence instead.
//
// The guard is a bound on |Y / 2^32 - (w + zo + 1/2)|, w = fl(v * out_inv)
// before the clamps, over every |acc| <= amax (the reduction length times the
// largest |x - zx| * |w|: (128 + |zx|) * 128 for int8 codes). With u = 2^-24
// and g = sx*s*out_inv, c = b*out_inv (reals), the float sequence's five
// roundings give |w - y| <= 3.0002u |acc| g + 2.0001u |y| (fl(acc) exact or
// not; no step underflows while sx, s and out_inv are in [2^-60, 2^60], and
// a subnormal sum is exact), |y| <= |acc| g + |c|; the integer terms give
// |G/2^32 - g| <= 2.0001u g + 2^-32 and |C/2^32 - (c + zo + 1/2)| <=
// u|c| + 2^-32. In units of 2^-32 (u = 256 units) the total is below
//
//   guard = 1.001 * (amax * (1795 g~ + 1) + 771 |c~| + 4)
//
// (the factor covers this float evaluation of it, its coefficients made once
// per op). Each channel whose terms stay in range (s in [2^-60, 2^60],
// g~ < 1/4, |c~| < 2^20, guard < 2^30) runs the integer form; any other
// channel, and fp32 out, run the float sequence. The guard's range gives
// amax g~ < 2^20, so |acc * G + C| / 2^32 < 2^21 and Y's high word is the
// code before the clamps. test/runtime/conv_epilogue_check.cpp holds the two equal
// on every acc in [-amax, amax] of random and adversarial channels, and of
// every channel the bench models run (ESP32_CONV_EPI_CHECK).

/// The integer form's terms for one channel (fixed_of); guard 0 runs the
/// float sequence.
struct Fixed {
  int64_t c;       // C: b * out_inv + zo + 1/2, in units of 2^-32
  // C less zx * S' * G, for the raw dot: raw * G + c_raw = acc * G + c
  // exactly (|acc| <= amax < 2^30 does not wrap, and the dot never does).
  int64_t c_raw;
  int32_t g;       // G: sx * s * out_inv, in units of 2^-32
  uint32_t guard;  // the bound on the integer form's error, in those units
};

/// sx, s and out_inv in [2^-60, 2^60]: no step of the float sequence
/// underflows or overflows.
constexpr float kScaleMin = 0x1p-60f;
constexpr float kScaleMax = 0x1p60f;

inline bool scale_in_range(float s) {
  return s >= kScaleMin && s <= kScaleMax;
}

/// floor(x * 2^32), exactly, for a finite |x| < 2^30: from x's bits.
inline int64_t floor_fixed32(float x) {
  const union {
    float value;
    uint32_t bits;
  } pun = {x};
  const uint32_t bits = pun.bits;
  const int32_t e = (int32_t)((bits >> 23) & 0xFFu);
  if (e == 0) {
    // Zero or subnormal: |x| * 2^32 < 1.
    return ((bits >> 31) != 0 && (bits & 0x7FFFFFu) != 0) ? -1 : 0;
  }
  // x = m * 2^(e - 150), so x * 2^32 = m * 2^(e - 118).
  const uint64_t m = (bits & 0x7FFFFFu) | 0x800000u;
  const int32_t sh = e - 118;
  uint64_t mag = 0;
  bool exact = true;
  if (sh >= 0) {
    mag = m << sh;  // sh <= 39 for |x| < 2^30
  } else if (sh > -32) {
    mag = m >> -sh;
    exact = (m & ((uint64_t{1} << -sh) - 1)) == 0;
  } else {
    exact = false;  // 0 < x * 2^32 < 1 in magnitude
  }
  if ((bits >> 31) != 0) {
    return -(int64_t)mag - (exact ? 0 : 1);
  }
  return (int64_t)mag;
}

/// One channel's integer form into *f, or guard 0 for the float sequence
/// (c_raw is the caller's). Per op terms come from Epi (fixed_epi), so a
/// channel costs two products, the guard's multiply-adds and C's bits. The
/// fields are stored one by one: a returned struct is copied by a memcpy call.
inline void fixed_of(float s, float b, const Epi& e, Fixed* f) {
  f->guard = 0;
  if (!e.fixed || !scale_in_range(s)) {
    return;
  }
  const float g = s * e.sx_out;
  const float c = b * e.out_inv;
  const float ca = c < 0.0f ? -c : c;
  const float guard = g * e.guard_g + ca * e.guard_c + e.guard_0;
  // NaN fails every compare. g~ < 1/4 keeps G inside int32: the guard's
  // range alone does not (a short reduction allows g~ up to ~36).
  if (!(g < 0.25f) || !(ca < 1048576.0f) || !(guard < 1073741824.0f)) {
    return;
  }
  f->g = esp32_platform_floor_i(g * 4294967296.0f);
  f->c = floor_fixed32(c) + (int64_t)e.zo * 4294967296LL +
         (int64_t{1} << 31);
  f->guard = (uint32_t)guard + 1u;
}

/// The column math of one element: fl(b + fl(fl(fl(acc) * sx) * s)), then
/// the clamp.
inline __attribute__((always_inline)) float column_value(int32_t acc,
                                                         float sx, float s,
                                                         float b, float lo,
                                                         float hi) {
  const float term = (float)acc * sx;
  float v = b + term * s;
  v = v < lo ? lo : v;
  v = v > hi ? hi : v;
  return v;
}

/// The float sequence's code for one element: the integer form's fallback.
/// Out of line: rare, and a float call keeps the integer loop's registers.
[[gnu::noinline]] int32_t float_code(int32_t acc, float s, float b,
                                     const Epi& e) {
  return code_of(column_value(acc, e.sx, s, b, e.lo, e.hi), e.out_inv, e.zo,
                 e.qmin, e.qmax);
}

/// Whether the integer form falls back at Y: its low word within `guard`
/// (< 2^31) of a multiple of 2^32, a rounding boundary.
inline __attribute__((always_inline)) bool near_boundary(int64_t Y,
                                                         uint32_t guard) {
  const int32_t f = (int32_t)(uint32_t)Y;
  const uint32_t d = f < 0 ? 0u - (uint32_t)f : (uint32_t)f;
  return d < guard;
}

/// The integer form's code for one element, or the float sequence's where
/// it cannot prove its own (*slow set).
inline int32_t fixed_code(int32_t acc, const Fixed& fx, const Epi& e,
                          float s, float b, bool* slow) {
  const int64_t Y = (int64_t)acc * fx.g + fx.c;
  if (near_boundary(Y, fx.guard)) {
    *slow = true;
    return float_code(acc, s, b, e);
  }
  *slow = false;
  const int32_t q = (int32_t)(Y >> 32);
  return q < e.code_lo ? e.code_lo : (q > e.code_hi ? e.code_hi : q);
}

// The loops below are out of line (noinline) on purpose: inlined into run(),
// they share one register allocation, and the LX7's 16 float registers then
// spill the epilogue's constants to the stack, reloaded every element.

/// The epilogue over `count` raw accumulators of one output channel, written
/// to elements at, at + 1, ...: acc = raw - zx * S' (wrapping), then the
/// column math, the clamp and the store. Separate from the dots on purpose:
/// no call inside the loop (the dot kernel is one), so its float constants
/// stay in registers, and its parameters are copied to locals because an
/// int8 store may alias anything a reference points at.
template <int kMode>
[[gnu::noinline]] void finish_float(const int32_t* raw, uint32_t count,
                                    uint32_t zxs, float s, float b,
                                    const Epi& epi, float* y, int8_t* y8,
                                    size_t at) {
  const float sx = epi.sx, lo = epi.lo, hi = epi.hi;
  const float out_inv = epi.out_inv, out_scale = epi.out_scale;
  const int32_t zo = epi.zo, qmin = epi.qmin, qmax = epi.qmax;
  float* __restrict yo = y != nullptr ? y + at : nullptr;
  int8_t* __restrict yo8 = y8 != nullptr ? y8 + at : nullptr;
  for (uint32_t t = 0; t < count; ++t) {
    const int32_t acc = (int32_t)((uint32_t)raw[t] - zxs);
    const float term = (float)acc * sx;
    float v = b + term * s;
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    if (kMode == kOutFloat) {
      yo[t] = v;
      continue;
    }
    const int32_t q = code_of(v, out_inv, zo, qmin, qmax);
    if (kMode == kOutCodes) {
      yo8[t] = (int8_t)q;
    } else {
      yo[t] = (float)(q - zo) * out_scale;
    }
  }
}

/// Column n's per-channel constants: its scale, bias (or +0.0f), zx * S' and
/// its integer epilogue's terms.
struct Channel {
  float s;
  float b;
  uint32_t zxs;
  Fixed fx;
};

#if ESP32_CONV_EPI_CHECK
// A host build's proof check (test/runtime/conv_epilogue_check.cpp, and the
// host runner built with -DESP32_CONV_EPI_CHECK=1 for every channel a model
// runs): the integer epilogue against the float sequence on every acc in
// [-amax, amax]. The float sequence's code E(acc) is a monotone step
// function, so its intervals of one code are found by bisection (at most
// qmax - qmin + 1 of them). On one interval the integer form's code is E's
// wherever it falls back, and elsewhere F(acc) = clamp(floor(Y / 2^32)),
// monotone in acc; so it equals E on the whole interval if F does at the
// first and at the last acc of the interval where it does not fall back.
struct EpiCheck {
  uint64_t channels, failures, intervals, evaluations;
};
EpiCheck g_epi_check = {0, 0, 0, 0};

/// A model run on a host runner built with the check prints its tally at
/// exit (the standalone check prints its own).
struct EpiCheckReport {
  ~EpiCheckReport() {
#if !defined(ESP32_CONV_EPI_CHECK_QUIET)
    printf("esp32 convepi check: %llu channel(s) held to the float sequence "
           "over every accumulator, %llu code interval(s), %llu "
           "evaluation(s), %llu mismatch(es)\n",
           (unsigned long long)g_epi_check.channels,
           (unsigned long long)g_epi_check.intervals,
           (unsigned long long)g_epi_check.evaluations,
           (unsigned long long)g_epi_check.failures);
#endif
  }
} g_epi_check_report;

/// F(acc) and whether the integer form falls back there.
inline int32_t check_fixed(int32_t acc, const Fixed& fx, const Epi& e,
                           bool* slow) {
  const int64_t Y = (int64_t)acc * fx.g + fx.c;
  *slow = near_boundary(Y, fx.guard);
  const int32_t q = (int32_t)(Y >> 32);
  return q < e.code_lo ? e.code_lo : (q > e.code_hi ? e.code_hi : q);
}

/// Holds one channel's integer form to the float sequence over [-amax, amax];
/// true when they agree everywhere.
bool check_channel(const Channel& ch, const Epi& e) {
  const int64_t amax = (int64_t)e.amax;
  ++g_epi_check.channels;
  int64_t lo = -amax;
  bool ok = true;
  while (lo <= amax) {
    const int32_t k = float_code((int32_t)lo, ch.s, ch.b, e);
    // The last acc of E's interval of code k: gallop, then bisect.
    int64_t step = 1, good = lo, bad = amax + 1;
    while (good + step <= amax) {
      ++g_epi_check.evaluations;
      if (float_code((int32_t)(good + step), ch.s, ch.b, e) != k) {
        bad = good + step;
        break;
      }
      good += step;
      step *= 2;
    }
    while (bad - good > 1) {
      const int64_t mid = good + (bad - good) / 2;
      ++g_epi_check.evaluations;
      if (float_code((int32_t)mid, ch.s, ch.b, e) == k) {
        good = mid;
      } else {
        bad = mid;
      }
    }
    const int64_t hi = good;
    ++g_epi_check.intervals;
    // The first and the last acc of [lo, hi] where the integer form does
    // not fall back (where it does, it computes E itself).
    bool slow = true;
    for (int64_t a = lo; a <= hi; ++a) {
      ++g_epi_check.evaluations;
      const int32_t f = check_fixed((int32_t)a, ch.fx, e, &slow);
      if (!slow) {
        ok = ok && f == k;
        break;
      }
    }
    for (int64_t a = hi; a >= lo; --a) {
      ++g_epi_check.evaluations;
      const int32_t f = check_fixed((int32_t)a, ch.fx, e, &slow);
      if (!slow) {
        ok = ok && f == k;
        break;
      }
    }
    lo = hi + 1;
  }
  if (!ok) {
    ++g_epi_check.failures;
  }
  return ok;
}
#endif

/// The integer epilogue over `count` (<= kRun) raw accumulators of one
/// output channel (codes, or fp32 requantised from them), the elements
/// finish_float writes: floor((raw * G + c_raw) / 2^32) clamped to
/// [code_lo, code_hi], then the float sequence (acc = raw - zx * S',
/// wrapping, as there) for the elements near a rounding boundary. Those are
/// noted in the loop and redone after it, so the loop makes no call and keeps
/// its terms in registers. Returns how many there were.
template <int kMode>
[[gnu::noinline]] uint32_t finish_fixed(const int32_t* raw, uint32_t count,
                                        const Channel& ch, const Epi& epi,
                                        float* y, int8_t* y8, size_t at) {
  const int64_t C = ch.fx.c_raw;
  const int32_t G = ch.fx.g;
  const uint32_t guard = ch.fx.guard;
  const int32_t qlo = epi.code_lo, qhi = epi.code_hi, zo = epi.zo;
  const float out_scale = epi.out_scale;
  float* __restrict yo = y != nullptr ? y + at : nullptr;
  int8_t* __restrict yo8 = y8 != nullptr ? y8 + at : nullptr;
  // The raw sums near a boundary, noted by address: the loop keeps no index.
  const int32_t* slow[kRun];
  const int32_t** next = slow;
  const int32_t* const end = raw + count;
  for (const int32_t* r = raw; r != end; ++r) {
    const int64_t Y = (int64_t)*r * G + C;
    if (__builtin_expect(near_boundary(Y, guard), 0)) {
      *next++ = r;
    }
    int32_t q = (int32_t)(Y >> 32);
    q = q < qlo ? qlo : q;
    q = q > qhi ? qhi : q;
    if (kMode == kOutCodes) {
      *yo8++ = (int8_t)q;
    } else {
      *yo++ = (float)(q - zo) * out_scale;
    }
  }
  yo = y != nullptr ? y + at : nullptr;
  yo8 = y8 != nullptr ? y8 + at : nullptr;
  for (const int32_t** p = slow; p != next; ++p) {
    const size_t t = (size_t)(*p - raw);
    const int32_t acc = (int32_t)((uint32_t)raw[t] - ch.zxs);
    const int32_t q = float_code(acc, ch.s, ch.b, epi);
    if (kMode == kOutCodes) {
      yo8[t] = (int8_t)q;
    } else {
      yo[t] = (float)(q - zo) * out_scale;
    }
  }
  return (uint32_t)(next - slow);
}

/// One channel's constants from its scale, bias, zero point and S'.
inline Channel channel_of(float s, float b, int32_t zx, uint32_t sum,
                          const Epi& e) {
  Channel ch;
  ch.s = s;
  ch.b = b;
  ch.zxs = (uint32_t)zx * sum;
  fixed_of(s, b, e, &ch.fx);
  if (ch.fx.guard != 0) {
    ch.fx.c_raw = ch.fx.c - (int64_t)zx * (int64_t)(int32_t)sum * ch.fx.g;
  }
#if ESP32_CONV_EPI_CHECK
  if (ch.fx.guard != 0) {
    // Each distinct channel once (a dense conv makes its channels again for
    // every row block): every term the two forms read. Under one lock: the
    // host's split worker makes its part's channels on a thread of its own,
    // and the set and the check's counters are shared.
    static std::mutex lock;
    std::lock_guard<std::mutex> hold(lock);
    static std::unordered_set<std::string> seen;
    std::string key(reinterpret_cast<const char*>(&ch.s), sizeof(float) * 2);
    key.append(reinterpret_cast<const char*>(&e.sx), sizeof(float) * 3);
    key.append(reinterpret_cast<const char*>(&e.out_inv), sizeof(float));
    const int32_t ints[5] = {e.zo, e.qmin, e.qmax, e.code_lo, e.code_hi};
    key.append(reinterpret_cast<const char*>(ints), sizeof(ints));
    key.append(reinterpret_cast<const char*>(&e.amax), sizeof(float));
    if (seen.insert(key).second) {
      check_channel(ch, e);
    }
  }
#endif
  return ch;
}

inline Channel channel(const Columns& c, int32_t zx, uint32_t n,
                       const Epi& e) {
  uint32_t sum = 0;
  if (c.sums != nullptr) {
    sum = c.sums_i16
        ? (uint32_t)(int32_t) static_cast<const int16_t*>(c.sums)[n]
        : (uint32_t) static_cast<const int32_t*>(c.sums)[n];
  }
  return channel_of(table_float(c.scales, c.scales_fp16, n),
                    c.bias != nullptr ? c.bias[n] : 0.0f, zx, sum, e);
}

/// One channel's epilogue over a run of raw accumulators: the integer form
/// when the channel has one, else the float sequence. Returns the elements
/// that took the float sequence as the integer form's fallback.
template <int kMode>
inline uint32_t finish(const int32_t* raw, uint32_t count, const Channel& ch,
                       const Epi& e, float* y, int8_t* y8, size_t at) {
  if (kMode != kOutFloat && ch.fx.guard != 0) {
    return finish_fixed<kMode>(raw, count, ch, e, y, y8, at);
  }
  finish_float<kMode>(raw, count, ch.zxs, ch.s, ch.b, e, y, y8, at);
  return 0;
}

// ---- the activation codes ---------------------------------------------------

/// The record's fp32 quantiser, one element at a time: the code the general
/// loops' gather makes for that element at every tap that reads it
/// (quantize_conv_row: code_of at fl(1 / s)).
struct Quant {
  float inv;
  int32_t zx, qmin, qmax;
  inline __attribute__((always_inline)) int8_t operator()(float v) const {
    return (int8_t)code_of(v, inv, zx, qmin, qmax);
  }
};

inline Quant quant_of(const Act& a) {
  return Quant{1.0f / a.sx, a.zx, a.qmin, a.qmax};
}

// ---- the fp32 codes without branches (dense convs) ---------------------------
// The dense gathers' code for an fp32 element (each tap's, or each element of
// prepare's plane), v = fl(x * inv) at Quant's fl(1 / sx), made as
//
//   q = rne(clamp(v, lo, hi)) + zx,   lo = qmin - zx, hi = qmax - zx,
//
// a NaN taken as 0 before the clamp, and the rounding made by the float
// adder: for |c| <= 2^22, fl(c + 1.5 * 2^23) lies in [2^23, 2^24], where the
// floats are the integers, so the add rounds c to the nearest one, ties to
// even, and its bits less 0x4B400000 are that integer. This is code_of's
// code for every x: rne is monotone and leaves an integer as it is, so
// rounding v clamped to [lo, hi] gives clamp(rne(v), lo, hi), which is
// code_of's clamp of rne(v) + zx to [qmin, qmax], less zx; code_of's rail
// clamp to +-kRail changes no value inside [lo, hi] (|lo|, |hi| <= 2^22 =
// kRail) and keeps the others on their side of it; and a NaN rounds as 0 in
// both, then clamped (a zero point outside the codes takes its nearest
// code). The unit builds -ffp-contract=off, so the product and the add are
// rounded each on its own, as in code_of. The clamps are selects (olt.s and
// movt.s), the NaN test too (quant_rules.h's quant_scaled with a zero kept in
// a register), and the bits come out by rfr: no branch an element, where
// code_of takes four or five (its rounding is floor, a float, a subtract and
// two compares). test/runtime/conv_quant_check.cpp holds the two equal on
// every float.

#if defined(ESP32_HOST_BUILD)
// Host only: calls into the three fast paths below (quant_fast_calls), so a
// test can tell their routes were taken -- their codes are code_of's, so the
// bits alone cannot. Relaxed atomics: a split worker would call them from a
// thread of its own. The firmware counts nothing.
std::atomic<uint64_t> g_fast_spans{0}, g_fast_pointwise{0}, g_fast_dense{0};
#define ESP32_QUANT_FAST_CALL(c) (c).fetch_add(1, std::memory_order_relaxed)
#else
#define ESP32_QUANT_FAST_CALL(c) ((void)0)
#endif

constexpr float kRoundMagic = 12582912.0f;  // 1.5 * 2^23
constexpr uint32_t kRoundMagicBits = 0x4B400000u;
constexpr int32_t kRoundSpan = 1 << 22;

/// The record's quantiser in that form (quant_fast_ok: lo and hi within
/// 2^22, every int8 record whose zero point is within 2^21 of its codes).
struct QuantFast {
  float inv, lo, hi;
  uint32_t bias;  // zx - 0x4B400000, modulo 2^32
};

inline bool quant_fast_ok(const Act& a) {
  const int64_t lo = (int64_t)a.qmin - a.zx, hi = (int64_t)a.qmax - a.zx;
  return lo <= hi && lo >= -kRoundSpan && hi <= kRoundSpan;
}

inline QuantFast quant_fast_of(const Act& a) {
  QuantFast q;
  q.inv = quant_of(a).inv;
  q.lo = (float)(int32_t)((int64_t)a.qmin - a.zx);
  q.hi = (float)(int32_t)((int64_t)a.qmax - a.zx);
  q.bias = (uint32_t)a.zx - kRoundMagicBits;
  return q;
}

/// A float's bits: rfr on the chip (a union or memcpy goes through the
/// stack there, at -O2 too).
inline __attribute__((always_inline)) uint32_t float_bits(float v) {
#if defined(__XTENSA__)
  uint32_t r;
  __asm__("rfr %0, %1" : "=a"(r) : "f"(v));
  return r;
#else
  uint32_t r;
  memcpy(&r, &v, sizeof(r));
  return r;
#endif
}

/// Keeps a loop's float constants in registers: GCC re-materialises them
/// per use otherwise (l32r and wfr each), and cannot see through an empty
/// asm.
inline __attribute__((always_inline)) void pin_floats(float& a, float& b,
                                                      float& c, float& d,
                                                      float& e) {
#if defined(__XTENSA__)
  __asm__("" : "+f"(a), "+f"(b), "+f"(c), "+f"(d), "+f"(e));
#else
  (void)a, (void)b, (void)c, (void)d, (void)e;
#endif
}

/// The loop constants of QuantFast, pinned (pin_floats), and one element.
struct QuantFastRegs {
  float inv, lo, hi, magic, zero;
  uint32_t bias;
  explicit QuantFastRegs(const QuantFast& q)
      : inv(q.inv), lo(q.lo), hi(q.hi), magic(kRoundMagic), zero(0.0f),
        bias(q.bias) {
    pin_floats(inv, lo, hi, magic, zero);
  }
  /// fl(x * inv), a NaN 0, clamped to [lo, hi]: the value the adder rounds.
  inline __attribute__((always_inline)) float clamped(float x) const {
    float v = quant_scaled(x * inv, zero);
    v = v < lo ? lo : v;
    return v > hi ? hi : v;
  }
  /// The code of a clamped value, rounded by the adder.
  inline __attribute__((always_inline)) int8_t code(float c) const {
    return (int8_t)(float_bits(c + magic) + bias);
  }
  inline __attribute__((always_inline)) int8_t operator()(float x) const {
    return code(clamped(x));
  }
};

/// n fp32 values to codes, four a step written out lane by lane: GCC keeps
/// the source order on the LX7, so the four chains interleave as written.
[[gnu::noinline]] void quantize_span_fast(const QuantFast& qf, const float* x,
                                          size_t n, int8_t* dst) {
  ESP32_QUANT_FAST_CALL(g_fast_spans);
  const QuantFastRegs q(qf);
  size_t t = 0;
  for (; t + 4 <= n; t += 4) {
    float a = x[t] * q.inv, b = x[t + 1] * q.inv;
    float c = x[t + 2] * q.inv, d = x[t + 3] * q.inv;
    a = quant_scaled(a, q.zero);
    b = quant_scaled(b, q.zero);
    c = quant_scaled(c, q.zero);
    d = quant_scaled(d, q.zero);
    a = a < q.lo ? q.lo : a;
    b = b < q.lo ? q.lo : b;
    c = c < q.lo ? q.lo : c;
    d = d < q.lo ? q.lo : d;
    a = a > q.hi ? q.hi : a;
    b = b > q.hi ? q.hi : b;
    c = c > q.hi ? q.hi : c;
    d = d > q.hi ? q.hi : d;
    dst[t] = q.code(a);
    dst[t + 1] = q.code(b);
    dst[t + 2] = q.code(c);
    dst[t + 3] = q.code(d);
  }
  for (; t < n; ++t) {
    dst[t] = q(x[t]);
  }
}

/// n fp32 values to codes.
void quantize_span(const Act& a, const float* x, size_t n, int8_t* dst) {
  if (quant_fast_ok(a)) {
    quantize_span_fast(quant_fast_of(a), x, n, dst);
    return;
  }
  const Quant q = quant_of(a);
  const float inv = q.inv;
  const int32_t zx = q.zx, qmin = q.qmin, qmax = q.qmax;
  for (size_t t = 0; t < n; ++t) {
    dst[t] = (int8_t)code_of(x[t], inv, zx, qmin, qmax);
  }
}

/// A code source for the gathers: codes already made (a plane), or fp32
/// quantised as each tap is read.
struct Codes {
  const int8_t* p;
  inline __attribute__((always_inline)) int8_t operator()(size_t i) const {
    return p[i];
  }
};
struct Quantised {
  const float* x;
  Quant q;
  inline __attribute__((always_inline)) int8_t operator()(size_t i) const {
    return q(x[i]);
  }
};
/// fp32 quantised as each tap is read, by QuantFast (the dense convs whose
/// record allows it): the same code as Quantised's, without its branches.
struct QuantisedFast {
  const float* x;
  QuantFastRegs q;
  inline __attribute__((always_inline)) int8_t operator()(size_t i) const {
    return q(x[i]);
  }
};

/// The output range [lo, hi) of one axis whose every tap lands inside the
/// input: o * stride - pad >= 0 and o * stride - pad + (ksize - 1) * dil <=
/// extent - 1.
inline void interior(uint32_t out, uint32_t extent, uint32_t ksize,
                     uint32_t stride, uint32_t pad, uint32_t dil, uint32_t* lo,
                     uint32_t* hi) {
  const int64_t span = (int64_t)(ksize - 1) * dil;
  const int64_t first = ((int64_t)pad + stride - 1) / stride;
  // o * stride may reach last_ok.
  const int64_t last_ok = (int64_t)extent - 1 - span + pad;
  int64_t end = last_ok < 0 ? 0 : last_ok / stride + 1;
  if (end > (int64_t)out) {
    end = out;
  }
  int64_t begin = first > end ? end : first;
  *lo = (uint32_t)begin;
  *hi = (uint32_t)end;
}

// ---- depthwise --------------------------------------------------------------

/// The extent of one axis an op's outputs read, from the first output's first
/// tap (o = 0, kk = 0, at -pad) to the last output's last tap.
inline uint32_t read_extent(uint32_t out, uint32_t ksize, uint32_t stride,
                            uint32_t dil) {
  return (out - 1) * stride + (ksize - 1) * dil + 1;
}

/// Whether every tap of every output lands inside the input (no padding read,
/// nothing past the end).
inline bool all_interior(const Op& op) {
  return op.pad_h == 0 && op.pad_w == 0 &&
         read_extent(op.h_out, op.kh, op.stride_h, op.dil_h) <= op.h_in &&
         read_extent(op.w_out, op.kw, op.stride_w, op.dil_w) <= op.w_in;
}

/// A run of 3x3 outputs with no dilation: rows p0, p1, p2 of the first
/// output's patch, `kStride` (0: `stride`) apart from one output to the
/// next. The taps are immediate offsets 0, 1, 2 from three row pointers, so
/// the nine weights can stay in registers.
template <uint32_t kStride>
inline void dot33_run(const int8_t* p0, const int8_t* p1, const int8_t* p2,
                      uint32_t stride, const int8_t* w, int32_t* raw,
                      uint32_t cnt) {
  const uint32_t st = kStride != 0 ? kStride : stride;
  const int32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4],
                w5 = w[5], w6 = w[6], w7 = w[7], w8 = w[8];
  for (uint32_t u = 0; u < cnt; ++u) {
    raw[u] = p0[0] * w0 + p0[1] * w1 + p0[2] * w2 + p1[0] * w3 + p1[1] * w4 +
        p1[2] * w5 + p2[0] * w6 + p2[1] * w7 + p2[2] * w8;
    p0 += st;
    p1 += st;
    p2 += st;
  }
}

/// One channel whose every tap is inside `corner` (pitch W): output (oh, ow)'s
/// patch starts at corner + oh * stride_h * W + ow * stride_w. Its raw dots
/// are made a run at a time, then finished together: raw - zx * S', then the
/// epilogue, into y/y8 (h_out x w_out). A run spans output rows (a channel's
/// outputs are contiguous), so a narrow output -- DS-CNN's 5 wide, the
/// MobileNets' last 6 and 3 -- pays one epilogue call per kRun outputs, not
/// one per row. Returns the elements the integer epilogue left to the float
/// sequence.
template <int kMode>
[[gnu::noinline]] uint32_t depthwise_channel(const Op& op,
                                             const int8_t* corner, size_t W,
                                             const int8_t* w,
                                             const Channel& ch, const Epi& e,
                                             float* y, int8_t* y8) {
  const uint32_t kh = op.kh, kw = op.kw;
  const uint32_t sw = op.stride_w;
  const size_t dh = op.dil_h, dw = op.dil_w;
  const bool k33 = kh == 3 && kw == 3 && dw == 1;
  int32_t raw[kRun];
  uint32_t slow = 0;
  uint32_t fill = 0;   // raw entries made and not yet finished
  size_t first = 0;    // the output element raw[0] holds
  for (uint32_t oh = 0; oh < op.h_out; ++oh) {
    const int8_t* row = corner + (size_t)oh * op.stride_h * W;
    for (uint32_t ow0 = 0; ow0 < op.w_out;) {
      const uint32_t left = op.w_out - ow0;
      const uint32_t cnt = left < kRun - fill ? left : kRun - fill;
      const int8_t* p = row + (size_t)ow0 * sw;
      int32_t* r = raw + fill;
      if (k33) {
        const int8_t* p1 = p + dh * W;
        const int8_t* p2 = p + 2 * dh * W;
        if (sw == 1) {
          dot33_run<1>(p, p1, p2, 1, w, r, cnt);
        } else if (sw == 2) {
          dot33_run<2>(p, p1, p2, 2, w, r, cnt);
        } else {
          dot33_run<0>(p, p1, p2, sw, w, r, cnt);
        }
      } else {
        for (uint32_t u = 0; u < cnt; ++u, p += sw) {
          uint32_t a = 0;
          uint32_t t = 0;
          for (uint32_t i = 0; i < kh; ++i) {
            const int8_t* pr = p + (size_t)i * dh * W;
            for (uint32_t j = 0; j < kw; ++j, ++t) {
              a += (uint32_t)(pr[(size_t)j * dw] * (int32_t)w[t]);
            }
          }
          r[u] = (int32_t)a;
        }
      }
      ow0 += cnt;
      fill += cnt;
      if (fill == kRun) {
        slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
        first += fill;
        fill = 0;
      }
    }
  }
  if (fill > 0) {
    slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
  }
  return slow;
}

/// A channel's codes into a padded plane (Hp x Wp, every tap of every output
/// inside it): the window the outputs read, from (-pad_h, -pad_w), with a tap
/// off the input holding `pad` -- the code the gather writes there -- and an
/// element inside it its code (fp32 quantised here, once).
template <typename Src>
[[gnu::noinline]] void pad_channel(const Op& op, Src src, int32_t pad,
                                   int8_t* dst, uint32_t Hp, uint32_t Wp) {
  memset(dst, (int8_t)pad, (size_t)Hp * Wp);
  // The input rows and columns inside the window (none when the padding
  // alone spans it).
  const uint32_t in_rows = Hp > op.pad_h ? Hp - op.pad_h : 0;
  const uint32_t in_cols = Wp > op.pad_w ? Wp - op.pad_w : 0;
  const uint32_t rows = op.h_in < in_rows ? op.h_in : in_rows;
  const uint32_t cols = op.w_in < in_cols ? op.w_in : in_cols;
  for (uint32_t ih = 0; ih < rows; ++ih) {
    int8_t* d = dst + (size_t)(ih + op.pad_h) * Wp + op.pad_w;
    const size_t at = (size_t)ih * op.w_in;
    for (uint32_t iw = 0; iw < cols; ++iw) {
      d[iw] = src(at + iw);
    }
  }
}

// ---- depthwise on PIE: 16 channels a vector ----------------------------------
// A 3x3 depthwise conv (no column dilation) whose part holds 16 channels or
// more runs them 16 at a time: the 16 channels' codes are made into a
// channels-last strip of the padded plane (16 bytes a pixel; the strip a band
// of output rows' input rows, within kPieStripBytes), dw33_qacc16_s3 makes
// each output pixel's 16 raw sums as one QACC image (nine ee.vmulas.s8.qacc),
// and each channel's run takes the epilogue straight from its 20-bit lane in
// the run's images (pie_finish16: qacc_lane_codes_s3 for codes out), with no
// unpacking or transposing between them. A raw sum is the same exact integer
// sum the scalar loop makes (sums do not depend on their order, and |sum| <=
// 9 * 128 * 128 fits QACC's lanes), so the outputs are the scalar path's
// bytes. Channels past the last full 16, a part with
// fewer than 16, and every call with PIE off (Columns::pie: --force-scalar,
// misaligned weights) take the scalar path; so does everything when the
// start-up check fails (pie_dw_init).

/// Output pixels a run of QACC images holds before their epilogue.
constexpr uint32_t kPieRun = 32;
/// The channels-last strip's budget (at least the 3 rows one output row reads).
constexpr uint32_t kPieStripBytes = 8 * 1024;

/// Which QACC lane holds each of the 16 channels, measured by pie_dw_init;
/// the PIE path runs only in state 2.
std::atomic<int> g_pie_dw_state{0};  // 0 unmeasured, 1 measuring, 2 on, 3 off
uint8_t g_pie_lane[16];

inline bool pie_dw_geometry(const Op& op) {
  return op.depthwise && op.kh == 3 && op.kw == 3 && op.dil_w == 1 &&
         op.n >= 16;
}

/// The strip's padded rows: as many as the budget holds, at least the
/// 2 * dil_h + 1 one output row reads, at most the padded plane's.
inline uint32_t pie_rows_in(const Op& op, uint32_t Hp, uint32_t Wp) {
  const uint32_t need = 2 * op.dil_h + 1;
  uint32_t rows = kPieStripBytes / (Wp * 16u);
  rows = rows < need ? need : rows;
  return rows > Hp ? Hp : rows;
}

/// The scratch one core's PIE depthwise uses: the strip, the 9 x 16 packed
/// weights, kPieRun QACC images, 16 runs of raw sums and 16 channels' terms.
inline uint64_t pie_dw_scratch(const Op& op) {
  const uint32_t Hp = read_extent(op.h_out, op.kh, op.stride_h, op.dil_h);
  const uint32_t Wp = read_extent(op.w_out, op.kw, op.stride_w, op.dil_w);
  return align16((uint64_t)pie_rows_in(op, Hp, Wp) * Wp * 16u) + 144u +
         (uint64_t)kPieRun * 64u + 16u * kPieRun * sizeof(int32_t) +
         align16(16u * sizeof(Channel));
}

/// One QACC image's 16 lanes, sign-extended from 20 bits, in lane order.
inline void qacc_lanes(const uint8_t* img, int32_t* lanes) {
  for (uint32_t h = 0; h < 2; ++h) {
    uint32_t w[5];
    memcpy(w, img + 32u * h, sizeof(w));
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t bit = 20u * i, q = bit >> 5, sh = bit & 31u;
      uint32_t v = w[q] >> sh;
      if (sh > 12u) {
        v |= w[q + 1] << (32u - sh);
      }
      lanes[8 * h + i] = (int32_t)(v << 12) >> 12;
    }
  }
}

/// The 16 channels' codes into a channels-last strip: padded rows
/// [r_first, r_first + r_count) of the window the outputs read (Wp wide),
/// a tap off the input holding `pad`.
template <typename Src>
[[gnu::noinline]] void pad_strip16(const Op& op, Src src, int32_t pad,
                                   uint32_t c0, uint32_t r_first,
                                   uint32_t r_count, uint32_t Wp,
                                   int8_t* strip) {
  const size_t plane = (size_t)op.h_in * op.w_in;
  const size_t pitch = (size_t)Wp * 16u;
  memset(strip, (int8_t)pad, (size_t)r_count * pitch);
  // The columns inside the input: q - pad_w in [0, w_in).
  const uint32_t q_lo = op.pad_w < Wp ? op.pad_w : Wp;
  const uint32_t q_end = op.pad_w + op.w_in < Wp ? op.pad_w + op.w_in : Wp;
  for (uint32_t r = 0; r < r_count; ++r) {
    const int64_t ih = (int64_t)r_first + r - op.pad_h;
    if (ih < 0 || ih >= (int64_t)op.h_in) {
      continue;
    }
    int8_t* row = strip + (size_t)r * pitch;
    for (uint32_t j = 0; j < 16; ++j) {
      const size_t base = (size_t)(c0 + j) * plane + (size_t)ih * op.w_in -
                          op.pad_w;
      int8_t* d = row + j;
      for (uint32_t q = q_lo; q < q_end; ++q) {
        d[(size_t)q * 16u] = src(base + q);
      }
    }
  }
}

/// Whether transpose16_s3 passed its start-up check (pie_dw_init); until
/// then, and if it failed, every strip takes pad_strip16.
std::atomic<int> g_transpose_ok{0};

/// `bytes` (a multiple of 16) of the strip at p (16-byte aligned) set to the
/// pad code, four bytes a store: a pad region is a few vectors a row, which a
/// memset call (this build: -fno-builtin-memset) would cost more than.
inline void fill_pad(int8_t* p, uint32_t pad4, size_t bytes) {
  typedef uint32_t __attribute__((may_alias)) Word;
  Word* w = reinterpret_cast<Word*>(p);
  for (size_t i = 0; i < bytes / 16u; ++i, w += 4) {
    w[0] = pad4;
    w[1] = pad4;
    w[2] = pad4;
    w[3] = pad4;
  }
}

/// pad_strip16 for Stage B codes (16 channels from c0 of x8's n_ch planes):
/// the pad code only where a tap is off the input (rows outside it, the
/// columns either side), and each input row's 16 channels transposed into
/// its strip row by transpose16_s3. A row whose transposes would read past
/// the last plane (the kernel reads whole 16-byte tiles, and the next
/// aligned block) is copied element by element instead. Its reads also
/// start at the aligned 16-byte block holding each tile's first code, up to
/// 15 bytes before x8 for the first group: mapped memory whenever that code
/// is (transpose16_s3.S), so that side needs no check. The strip is
/// pad_strip16's, byte for byte.
[[gnu::noinline]] void strip16_codes(const Op& op, const int8_t* x8,
                                     uint32_t n_ch, int32_t pad, uint32_t c0,
                                     uint32_t r_first, uint32_t r_count,
                                     uint32_t Wp, int8_t* strip) {
  const size_t plane = (size_t)op.h_in * op.w_in;
  const size_t pitch = (size_t)Wp * 16u;
  const uint32_t pad4 = (uint32_t)(uint8_t)pad * 0x01010101u;
  // The columns inside the input: q - pad_w in [0, w_in).
  const uint32_t q_lo = op.pad_w < Wp ? op.pad_w : Wp;
  const uint32_t q_end = op.pad_w + op.w_in < Wp ? op.pad_w + op.w_in : Wp;
  const uint32_t cols = q_end > q_lo ? q_end - q_lo : 0;
  const int8_t* const end = x8 + (size_t)n_ch * plane;
  for (uint32_t r = 0; r < r_count; ++r) {
    int8_t* row = strip + (size_t)r * pitch;
    const int64_t ih = (int64_t)r_first + r - op.pad_h;
    if (ih < 0 || ih >= (int64_t)op.h_in || cols == 0) {
      fill_pad(row, pad4, pitch);
      continue;
    }
    fill_pad(row, pad4, (size_t)q_lo * 16u);
    fill_pad(row + (size_t)q_end * 16u, pad4, (size_t)(Wp - q_end) * 16u);
    const int8_t* src = x8 + (size_t)c0 * plane + (size_t)ih * op.w_in +
                        (q_lo - op.pad_w);
    int8_t* dst = row + (size_t)q_lo * 16u;
    if (src + 15 * plane + (size_t)(cols - 1) / 16u * 16u + 32u <= end) {
      transpose16_s3(src, (int)plane, dst, 16, (int)cols);
    } else {
      for (uint32_t j = 0; j < 16; ++j) {
        for (uint32_t q = 0; q < cols; ++q) {
          dst[(size_t)q * 16u + j] = src[(size_t)j * plane + q];
        }
      }
    }
  }
}

/// A strip of 16 channels: strip16_codes for Stage B codes once
/// transpose16_s3 passed its check, else (and for fp32, quantised element by
/// element as it is copied) pad_strip16.
inline void make_strip16(const Op& op, Codes src, int32_t pad, uint32_t c0,
                         uint32_t r_first, uint32_t r_count, uint32_t Wp,
                         int8_t* strip) {
  if (g_transpose_ok.load(std::memory_order_relaxed) != 0) {
    strip16_codes(op, src.p, op.n, pad, c0, r_first, r_count, Wp, strip);
  } else {
    pad_strip16(op, src, pad, c0, r_first, r_count, Wp, strip);
  }
}
inline void make_strip16(const Op& op, Quantised src, int32_t pad, uint32_t c0,
                         uint32_t r_first, uint32_t r_count, uint32_t Wp,
                         int8_t* strip) {
  pad_strip16(op, src, pad, c0, r_first, r_count, Wp, strip);
}

/// One 32-bit word of a QACC image (4-byte aligned): a plain load, not a
/// memcpy, which this build (-fno-builtin-memcpy) leaves a call.
inline __attribute__((always_inline)) uint32_t image_word(const uint8_t* p) {
  typedef uint32_t __attribute__((may_alias)) Word;
  return *reinterpret_cast<const Word*>(p);
}

/// Where QACC lane `lane` (0-15) sits in a 64-byte image, as qacc_lanes
/// reads it: the byte offset of the 32-bit word holding its bit 0 (QACC_L's
/// lanes 0-7 from +0, QACC_H's 8-15 from +32, 20 bits each), and that bit's
/// position in the word; a lane at a position past 12 runs into the next word.
struct LaneAt {
  uint32_t off, shift;
};
inline LaneAt lane_at(uint32_t lane) {
  const uint32_t bit = 20u * (lane & 7u);
  return LaneAt{32u * (lane >> 3) + 4u * (bit >> 5), bit & 31u};
}

/// One raw sum from its lane at p (lane_at's word): bits [shift, shift + 20)
/// of the little-endian word pair, sign-extended; kSpan when they run into
/// the second word (shift > 12).
template <bool kSpan>
inline __attribute__((always_inline)) int32_t lane_raw(const uint8_t* p,
                                                       uint32_t shift) {
  uint32_t v;
  if (kSpan) {
#if defined(__XTENSA__)
    // The word pair shifted right as one 64-bit value: SRC, where GCC
    // writes the two shifts and the or out (three more instructions).
    __asm__("ssr %1\n\tsrc %0, %2, %3"
            : "=a"(v)
            : "a"(shift), "a"(image_word(p + 4)), "a"(image_word(p)));
#else
    v = (image_word(p) >> shift) | (image_word(p + 4) << (32u - shift));
#endif
  } else {
    v = image_word(p) >> shift;
  }
  return (int32_t)(v << 12) >> 12;
}

/// The integer epilogue of one channel's run straight from its QACC lane,
/// fp32 out (requantised from its code; codes out take lane_codes): the raw
/// sums of `fill` pixels at p, p + 64, ... (lane_raw), each taking
/// finish_fixed's arithmetic -- floor((raw * G + c_raw) / 2^32) clamped to
/// [code_lo, code_hi] -- into elements at, at + 1, ...; those near a rounding
/// boundary are noted by image address and redone by the float sequence
/// after the loop, so the loop makes no call. No lanes are unpacked to a
/// staging run first: the run is read where the kernel left it. Returns how
/// many took the float sequence.
template <int kMode, bool kSpan>
[[gnu::noinline]] uint32_t lane_fixed(const uint8_t* img, uint32_t fill,
                                      uint32_t shift, const Channel& ch,
                                      const Epi& epi, float* y, int8_t* y8,
                                      size_t at) {
  const int64_t C = ch.fx.c_raw;
  const int32_t G = ch.fx.g;
  const uint32_t guard = ch.fx.guard;
  const int32_t qlo = epi.code_lo, qhi = epi.code_hi, zo = epi.zo;
  const float out_scale = epi.out_scale;
  float* __restrict yo = y != nullptr ? y + at : nullptr;
  int8_t* __restrict yo8 = y8 != nullptr ? y8 + at : nullptr;
  const uint8_t* slow[kPieRun];
  const uint8_t** next = slow;
  const uint8_t* const end = img + (size_t)fill * 64u;
  for (const uint8_t* p = img; p != end; p += 64) {
    const int64_t Y = (int64_t)lane_raw<kSpan>(p, shift) * G + C;
    if (__builtin_expect(near_boundary(Y, guard), 0)) {
      *next++ = p;
    }
    int32_t q = (int32_t)(Y >> 32);
    q = q < qlo ? qlo : q;
    q = q > qhi ? qhi : q;
    if (kMode == kOutCodes) {
      *yo8++ = (int8_t)q;
    } else {
      *yo++ = (float)(q - zo) * out_scale;
    }
  }
  yo = y != nullptr ? y + at : nullptr;
  yo8 = y8 != nullptr ? y8 + at : nullptr;
  for (const uint8_t** s = slow; s != next; ++s) {
    const size_t t = (size_t)(*s - img) / 64u;
    const int32_t acc =
        (int32_t)((uint32_t)lane_raw<kSpan>(*s, shift) - ch.zxs);
    const int32_t q = float_code(acc, ch.s, ch.b, epi);
    if (kMode == kOutCodes) {
      yo8[t] = (int8_t)q;
    } else {
      yo[t] = (float)(q - zo) * out_scale;
    }
  }
  return (uint32_t)(next - slow);
}

/// One channel's run by the float sequence (fp32 out, or a channel without
/// the integer form): its raw sums from its lane, then finish_float.
template <int kMode>
[[gnu::noinline]] void lane_float(const uint8_t* img, uint32_t fill,
                                  uint32_t shift, const Channel& ch,
                                  const Epi& epi, float* y, int8_t* y8,
                                  size_t at) {
  int32_t raw[kPieRun];
  for (uint32_t u = 0; u < fill; ++u) {
    raw[u] = shift > 12u ? lane_raw<true>(img + (size_t)u * 64u, shift)
                         : lane_raw<false>(img + (size_t)u * 64u, shift);
  }
  finish_float<kMode>(raw, fill, ch.zxs, ch.s, ch.b, epi, y, y8, at);
}

/// A channel's terms for qacc_lane_codes_s3: its integer form (fixed_of,
/// channel_of's c_raw), the op's code clamp, and its lane's bit position.
inline Esp32LaneEpi lane_epi(const Channel& ch, const Epi& e, uint32_t shift) {
  Esp32LaneEpi t;
  t.g = ch.fx.g;
  t.c_lo = (uint32_t)(uint64_t)ch.fx.c_raw;
  t.c_hi = (int32_t)(ch.fx.c_raw >> 32);
  t.guard = ch.fx.guard;
  t.qlo = e.code_lo;
  t.qhi = e.code_hi;
  t.shift = shift;
  return t;
}

/// lane_fixed for codes out: the run's integer epilogue by
/// qacc_lane_codes_s3 (lane_fixed's loop in assembly: GCC's own keeps no
/// zero-overhead loop around the funnel shift, three instructions more an
/// element), then the float sequence for the elements it noted.
[[gnu::noinline]] uint32_t lane_codes(const uint8_t* img, uint32_t fill,
                                      const Esp32LaneEpi& t,
                                      const Channel& ch, const Epi& epi,
                                      int8_t* yo8) {
  const uint8_t* slow[kPieRun];
  const int n = qacc_lane_codes_s3(img, (int)fill, yo8, &t, slow);
  for (int i = 0; i < n; ++i) {
    const size_t u = (size_t)(slow[i] - img) / 64u;
    const int32_t raw = t.shift > 12u ? lane_raw<true>(slow[i], t.shift)
                                      : lane_raw<false>(slow[i], t.shift);
    const int32_t acc = (int32_t)((uint32_t)raw - ch.zxs);
    yo8[u] = (int8_t)float_code(acc, ch.s, ch.b, epi);
  }
  return (uint32_t)n;
}

/// A run of `fill` pixels' QACC images: each channel's epilogue, from its
/// lane (g_pie_lane; terms[j] its lane_epi when it has the integer form),
/// into its outputs from element `first` on.
template <int kMode>
[[gnu::noinline]] uint32_t pie_finish16(const uint8_t* img, uint32_t fill,
                                        const Channel* chans,
                                        const Esp32LaneEpi* terms,
                                        const Epi& e, const Out& out,
                                        size_t m, uint32_t c0, size_t first) {
  uint32_t slow = 0;
  for (uint32_t j = 0; j < 16; ++j) {
    const LaneAt l = lane_at(g_pie_lane[j]);
    const size_t at = (size_t)(c0 + j) * m + first;
    const uint8_t* p = img + l.off;
    if (kMode == kOutCodes && chans[j].fx.guard != 0) {
      slow += lane_codes(p, fill, terms[j], chans[j], e, out.y8 + at);
    } else if (kMode == kOutRequant && chans[j].fx.guard != 0) {
      slow += l.shift > 12u ? lane_fixed<kMode, true>(p, fill, l.shift,
                                                      chans[j], e, out.y,
                                                      nullptr, at)
                            : lane_fixed<kMode, false>(p, fill, l.shift,
                                                       chans[j], e, out.y,
                                                       nullptr, at);
    } else {
      lane_float<kMode>(p, fill, l.shift, chans[j], e, out.y, out.y8, at);
    }
  }
  return slow;
}

/// Channels [c0, c0 + 16) of a 3x3 depthwise conv on PIE (the section above).
template <int kMode, typename Src>
[[gnu::noinline]] void depthwise_pie16(const Op& op, Src src, int32_t pad,
                                       const Columns& c, int32_t zx,
                                       const Epi& e, const Out& out,
                                       uint32_t c0, int8_t* scratch,
                                       Counts* counts) {
  const uint32_t Hp = read_extent(op.h_out, op.kh, op.stride_h, op.dil_h);
  const uint32_t Wp = read_extent(op.w_out, op.kw, op.stride_w, op.dil_w);
  const uint32_t rows_in = pie_rows_in(op, Hp, Wp);
  const uint32_t span = 2 * op.dil_h + 1;
  const uint32_t rows_out = (rows_in - span) / op.stride_h + 1;
  const size_t pitch = (size_t)Wp * 16u;
  int8_t* strip = scratch;
  int8_t* wpack = strip + align16((uint64_t)rows_in * pitch);
  uint8_t* img = reinterpret_cast<uint8_t*>(wpack + 144);
  // pie_dw_scratch's layout: the images, then a room sized for 16 runs of raw
  // sums, which holds the 16 channels' qacc_lane_codes_s3 terms (pie_finish16
  // reads the lanes in place), then the channels.
  Esp32LaneEpi* terms =
      reinterpret_cast<Esp32LaneEpi*>(img + (size_t)kPieRun * 64u);
  static_assert(16 * sizeof(Esp32LaneEpi) <= 16 * kPieRun * sizeof(int32_t),
                "the terms fit the raw runs' room");
  Channel* chans = reinterpret_cast<Channel*>(img + (size_t)kPieRun * 64u +
                                              16u * kPieRun * sizeof(int32_t));
  // The 9 taps' 16 weights, tap-major: the 16 channels' weight rows (k_pad
  // apart) transposed by transpose16_s3 where its reads (whole 16-byte tiles,
  // and the next aligned block) stay inside the op's weights, else one by one.
  const int8_t* wrows = c.wq + (size_t)c0 * op.k_pad;
  if (g_transpose_ok.load(std::memory_order_relaxed) != 0 &&
      wrows + 15 * (size_t)op.k_pad + 32u <= c.wq + (size_t)op.n * op.k_pad) {
    transpose16_s3(wrows, (int)op.k_pad, wpack, 16, 9);
  } else {
    for (uint32_t t = 0; t < 9; ++t) {
      for (uint32_t j = 0; j < 16; ++j) {
        wpack[t * 16 + j] = wrows[(size_t)j * op.k_pad + t];
      }
    }
  }
  for (uint32_t j = 0; j < 16; ++j) {
    chans[j] = channel(c, zx, c0 + j, e);
    if (kMode != kOutFloat && chans[j].fx.guard != 0) {
      counts->fixed += op.m;
      terms[j] = lane_epi(chans[j], e, lane_at(g_pie_lane[j]).shift);
    }
  }
  uint32_t fill = 0;
  size_t first = 0;
  for (uint32_t oh0 = 0; oh0 < op.h_out; oh0 += rows_out) {
    const uint32_t oh1 = oh0 + rows_out < op.h_out ? oh0 + rows_out : op.h_out;
    make_strip16(op, src, pad, c0, oh0 * op.stride_h,
                 (oh1 - 1 - oh0) * op.stride_h + span, Wp, strip);
    for (uint32_t oh = oh0; oh < oh1; ++oh) {
      const int8_t* row = strip + (size_t)(oh - oh0) * op.stride_h * pitch;
      for (uint32_t ow0 = 0; ow0 < op.w_out;) {
        const uint32_t left = op.w_out - ow0;
        const uint32_t cnt = left < kPieRun - fill ? left : kPieRun - fill;
        dw33_qacc16_s3(row + (size_t)ow0 * op.stride_w * 16u,
                       (int)(pitch * op.dil_h), (int)(op.stride_w * 16u),
                       wpack, img + (size_t)fill * 64u, (int)cnt);
        ow0 += cnt;
        fill += cnt;
        if (fill == kPieRun) {
          counts->fallback += pie_finish16<kMode>(img, fill, chans, terms, e,
                                                  out, op.m, c0, first);
          first += fill;
          fill = 0;
        }
      }
    }
  }
  if (fill > 0) {
    counts->fallback +=
        pie_finish16<kMode>(img, fill, chans, terms, e, out, op.m, c0, first);
  }
}

// ---- depthwise on PIE: one channel, 16 pixels a vector ----------------------
// A 5x5 depthwise conv, and the 3x3 channels the 16-channel path leaves (past
// the last full 16, or an op of fewer than 16), with no column dilation and
// a column stride of 1 or 2, run one channel at a time on the padded plane
// the scalar loop would read -- the same scratch, so no reservation grows --
// 16 output pixels of an output row a vector: dw5_plane_s3 (dw3_plane_s3)
// makes each run's 16 raw sums as one QACC image (one ee.vmulas.s8.qacc a
// tap: the pixels' codes, unaligned, times the tap's weight, broadcast),
// unpack16 reads its lanes, and the channel's runs take finish() as the
// scalar loop's do. A raw sum is the same exact integer sum the scalar loop
// makes (|sum| <= 25 * 128 * 128 < 2^19 fits QACC's lanes), so the outputs
// are the scalar path's bytes.
//
// The plane holds only the input rows the window reads (row r at r * Wp); a
// window row off the input reads a pad row of the code a tap off the input
// reads instead. A column stride of 2 stores each row's even columns, then
// its odd ones, so a tap's 16 pixels are 16 consecutive bytes either way.
// The rows left out (the padding's, at least pad_h of them) are the room the
// kernel's reads past the last row need: every tap reads its 16 codes as two
// aligned blocks, up to 31 bytes past the tap's address (plane_geo holds the
// furthest inside the scratch). The path runs where the plane is made (not
// Stage B codes read in place), PIE is on and its start-up check passed.

/// Pixels one kernel call makes: 2 runs of 16.
constexpr uint32_t kPlaneCall = 32;
/// The pad row's bytes (on the stack): the furthest a pad window row's taps
/// read.
constexpr uint32_t kPlanePadBytes = 256;

/// Whether dw5_plane_s3 and dw3_plane_s3 passed their start-up check
/// (pie_dw_init); until then, and if they failed, the scalar loop.
std::atomic<int> g_plane_ok{0};

/// One part of a stored row (the whole row, or its even or odd columns):
/// n bytes from `off`, element k padded column `first + step * k`, so
/// elements [lo, hi) are input columns (from c0, step apart) and the rest
/// the pad code.
struct PlanePart {
  uint32_t off, n, lo, hi, c0, step;
};

/// The plane path's layout for one op.
struct PlaneGeo {
  uint32_t Wp;       // the padded window's width (a stored row's bytes)
  uint32_t rows_in;  // the input rows the window reads, row r at r * Wp
  uint32_t boff;     // a window row's B taps after its A taps
  bool poly;         // column stride 2: even columns, then odd ones
  PlanePart part[2];  // the row (poly: its even columns, its odd ones)
};

/// Whether op takes the plane path, and its layout. Its reads stay inside
/// the plane's Hp * Wp bytes of scratch (scratch_need) and the pad row.
inline bool plane_geo(const Op& op, PlaneGeo* g) {
  if (!op.depthwise || op.kh != op.kw || (op.kh != 5 && op.kh != 3) ||
      op.dil_w != 1 || (op.stride_w != 1 && op.stride_w != 2) ||
      op.h_out == 0 || op.w_out == 0) {
    return false;
  }
  const uint32_t Hp = read_extent(op.h_out, op.kh, op.stride_h, op.dil_h);
  const uint32_t Wp = read_extent(op.w_out, op.kw, op.stride_w, op.dil_w);
  const uint32_t ka = (op.kw + 1) / 2, kb = op.kw / 2;
  g->Wp = Wp;
  g->poly = op.stride_w == 2;
  g->boff = g->poly ? (Wp + 1) / 2 : ka;
  const uint32_t below = Hp > op.pad_h ? Hp - op.pad_h : 0;
  g->rows_in = op.h_in < below ? op.h_in : below;
  // The columns inside the input: padded column q with q - pad_w in
  // [0, w_in), q in [q_lo, q_end).
  const uint32_t q_lo = op.pad_w < Wp ? op.pad_w : Wp;
  const uint32_t q_end = op.pad_w + op.w_in < Wp ? op.pad_w + op.w_in : Wp;
  if (!g->poly) {
    g->part[0] = PlanePart{0, Wp, q_lo, q_end > q_lo ? q_end : q_lo,
                           q_lo - op.pad_w, 1};
  } else {
    // Even columns q = 2k: inside for k in [ceil(q_lo/2), ceil(q_end/2));
    // odd q = 2k + 1: for k in [floor(q_lo/2), floor(q_end/2)).
    const uint32_t he = g->boff, ho = Wp - he;
    uint32_t lo = (q_lo + 1) / 2, hi = (q_end + 1) / 2;
    hi = hi < he ? hi : he;
    lo = lo < hi ? lo : hi;
    g->part[0] = PlanePart{0, he, lo, hi, 2 * lo - op.pad_w, 2};
    lo = q_lo / 2;
    hi = q_end / 2;
    hi = hi < ho ? hi : ho;
    lo = lo < hi ? lo : hi;
    g->part[1] = PlanePart{he, ho, lo, hi, 2 * lo + 1 - op.pad_w, 2};
  }
  // The furthest tap address past a window row's first byte, plus 16: the
  // second aligned block the kernel loads for it ends at (far | 15) + 1.
  const uint32_t tap = ka - 1 > g->boff + kb - 1 ? ka - 1 : g->boff + kb - 1;
  const uint64_t far = (uint64_t)(op.w_out - 1) / 16u * 16u + tap + 16u;
  if ((far | 15u) + 1u > kPlanePadBytes) {
    return false;
  }
  return g->rows_in == 0 ||
         ((((uint64_t)(g->rows_in - 1) * Wp + far) | 15u) + 1u <=
          (uint64_t)Hp * Wp);
}

/// One channel's input rows into the plane (PlaneGeo): row r the padded
/// window's row r + pad_h, Wp bytes, a column off the input holding `pad`
/// (fp32 quantised here, once); with poly its even columns, then its odd.
/// The pad runs are a few bytes a row: plain stores, which GCC would
/// otherwise make memset calls (loop distribution), one a run.
#if defined(__GNUC__) && !defined(__clang__)
#define ESP32_CONV_NO_LOOP_CALLS \
  __attribute__((optimize("no-tree-loop-distribute-patterns")))
#else
#define ESP32_CONV_NO_LOOP_CALLS
#endif
template <typename Src>
[[gnu::noinline]] ESP32_CONV_NO_LOOP_CALLS void plane_rows(
    const Op& op, Src src, int32_t pad, const PlaneGeo& g, int8_t* dst) {
  const int8_t p = (int8_t)pad;
  const uint32_t parts = g.poly ? 2u : 1u;
  const uint32_t rows = g.rows_in, Wp = g.Wp, w_in = op.w_in;
  for (uint32_t t = 0; t < parts; ++t) {
    // The part's terms in locals: an int8 store may alias the struct.
    const uint32_t off = g.part[t].off, n = g.part[t].n, lo = g.part[t].lo,
                   hi = g.part[t].hi, c0 = g.part[t].c0,
                   step = g.part[t].step;
    for (uint32_t r = 0; r < rows; ++r) {
      int8_t* __restrict o = dst + (size_t)r * Wp + off;
      uint32_t k = 0;
      for (; k < lo; ++k) {
        o[k] = p;
      }
      size_t at = (size_t)r * w_in + c0;
      for (; k < hi; ++k, at += step) {
        o[k] = src(at);
      }
      for (; k < n; ++k) {
        o[k] = p;
      }
    }
  }
}

/// A 20-bit lane's value, sign-extended.
inline __attribute__((always_inline)) int32_t sext20(uint32_t v) {
  return (int32_t)(v << 12) >> 12;
}

/// Bits [kShift, kShift + 32) of the little-endian word pair (lo, hi).
template <uint32_t kShift>
inline __attribute__((always_inline)) uint32_t funnel(uint32_t lo,
                                                      uint32_t hi) {
#if defined(__XTENSA__)
  uint32_t v;
  __asm__("ssai %1\n\tsrc %0, %2, %3"
          : "=a"(v)
          : "i"(kShift), "a"(hi), "a"(lo));
  return v;
#else
  return (lo >> kShift) | (hi << (32u - kShift));
#endif
}

/// A plane kernel's image as its 16 raw sums, lane l (pixel l) into raw[l]:
/// bits [20 l', 20 l' + 20) of QACC_L's 160 bits (l < 8) or QACC_H's, l' =
/// l mod 8, sign-extended. Lane l holding pixel l is what the start-up check
/// holds the kernels to.
inline __attribute__((always_inline)) void unpack8(const uint8_t* p,
                                                    int32_t* r) {
  const uint32_t w0 = image_word(p), w1 = image_word(p + 4),
                 w2 = image_word(p + 8), w3 = image_word(p + 12),
                 w4 = image_word(p + 16);
  r[0] = sext20(w0);
  r[1] = sext20(funnel<20>(w0, w1));
  r[2] = sext20(w1 >> 8);
  r[3] = sext20(funnel<28>(w1, w2));
  r[4] = sext20(funnel<16>(w2, w3));
  r[5] = sext20(w3 >> 4);
  r[6] = sext20(funnel<24>(w3, w4));
  r[7] = (int32_t)w4 >> 12;
}
inline void unpack16(const uint8_t* img, int32_t* raw) {
  unpack8(img, raw);
  unpack8(img + 32, raw + 8);
}

/// One channel on the plane path: each output row's window rows (the plane's,
/// or the pad row), its pixels kPlaneCall a kernel call, their raw sums
/// unpacked into a run and the run finished as depthwise_channel finishes
/// its own. wk is the channel's taps in the kernel's order. Returns the
/// elements the integer epilogue left to the float sequence.
template <int kMode>
[[gnu::noinline]] uint32_t depthwise_plane(const Op& op, const PlaneGeo& g,
                                           const int8_t* plane,
                                           const int8_t* pad_row,
                                           const int8_t* wk,
                                           const Channel& ch, const Epi& e,
                                           float* y, int8_t* y8) {
  alignas(16) uint8_t img[kPlaneCall / 16u * 64u];
  // A run, and room for a call's last 16 lanes past it.
  int32_t raw[kRun + 16];
  const int8_t* rows[5];
  const int8_t* moved[5];
  const uint32_t kh = op.kh;
  // A column stride of 1 puts the B taps right after the A taps: the
  // kernel without the step between them.
  void (*const kernel)(const int8_t* const*, int, const int8_t*, uint8_t*,
                       int) = kh == 5 ? (g.poly ? dw5_plane_s3 : dw5_plane1_s3)
                                      : (g.poly ? dw3_plane_s3 : dw3_plane1_s3);
  uint32_t slow = 0;
  uint32_t fill = 0;
  size_t first = 0;
  // Window row i of output row oh reads input row oh * stride_h + i * dil_h
  // - pad_h (in int32: the plane's rows are far fewer than 2^31). Where all
  // of an output row's window rows are stored rows and so were the last
  // output row's, each moves on stride_h rows.
  const int32_t sh = (int32_t)op.stride_h, dh = (int32_t)op.dil_h;
  const int32_t last = (int32_t)(kh - 1) * dh;
  const size_t step = (size_t)op.stride_h * g.Wp;
  bool stored = false;  // the last output row's window rows all stored
  int32_t ih0 = -(int32_t)op.pad_h;
  for (uint32_t oh = 0; oh < op.h_out; ++oh, ih0 += sh) {
    const bool all = ih0 >= 0 && ih0 + last < (int32_t)g.rows_in;
    if (all && stored) {
      for (uint32_t i = 0; i < kh; ++i) {
        rows[i] += step;
      }
    } else {
      int32_t ih = ih0;
      for (uint32_t i = 0; i < kh; ++i, ih += dh) {
        rows[i] = (uint32_t)ih < g.rows_in ? plane + (size_t)ih * g.Wp
                                           : pad_row;
      }
    }
    stored = all;
    if (op.w_out <= kPlaneCall) {
      // The whole row in one call (MobileNet's rows): no inner loop.
      const uint32_t cnt = op.w_out;
      if (fill + cnt > kRun) {
        slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
        first += fill;
        fill = 0;
      }
      const uint32_t chunks = (cnt + 15u) / 16u;
      kernel(rows, (int)g.boff, wk, img, (int)chunks);
      unpack8(img, raw + fill);
      if (cnt > 8u) {
        unpack8(img + 32u, raw + fill + 8u);
      }
      if (cnt > 16u) {
        unpack8(img + 64u, raw + fill + 16u);
        if (cnt > 24u) {
          unpack8(img + 96u, raw + fill + 24u);
        }
      }
      fill += cnt;
      continue;
    }
    for (uint32_t ow0 = 0; ow0 < op.w_out; ow0 += kPlaneCall) {
      const uint32_t left = op.w_out - ow0;
      const uint32_t cnt = left < kPlaneCall ? left : kPlaneCall;
      if (fill + cnt > kRun) {
        slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
        first += fill;
        fill = 0;
      }
      const int8_t* const* r = rows;
      if (ow0 != 0) {
        for (uint32_t i = 0; i < kh; ++i) {
          moved[i] = rows[i] + ow0;
        }
        r = moved;
      }
      const uint32_t chunks = (cnt + 15u) / 16u;
      kernel(r, (int)g.boff, wk, img, (int)chunks);
      for (uint32_t k = 0; k < chunks; ++k) {
        unpack8(img + 64u * k, raw + fill + 16u * k);
        if (cnt > 16u * k + 8u) {
          unpack8(img + 64u * k + 32u, raw + fill + 16u * k + 8u);
        }
      }
      fill += cnt;
    }
  }
  if (fill > 0) {
    slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
  }
  return slow;
}

/// A channel's weights in the plane kernels' order (per window row its A
/// taps, then its B taps): a column stride of 1 keeps the stored order (A
/// the first ka columns), so the row is used as it is; a stride of 2 puts
/// the even columns first.
inline const int8_t* plane_weights(const Op& op, const PlaneGeo& g,
                                   const int8_t* w, int8_t* buf) {
  if (!g.poly) {
    return w;
  }
  // Tap t of the kernel's order: the stored tap at kOrder[t].
  static const uint8_t k5[25] = {0,  2,  4,  1,  3,  5,  7,  9,  6,
                                 8,  10, 12, 14, 11, 13, 15, 17, 19,
                                 16, 18, 20, 22, 24, 21, 23};
  static const uint8_t k3[9] = {0, 2, 1, 3, 5, 4, 6, 8, 7};
  const uint8_t* order = op.kh == 5 ? k5 : k3;
  const uint32_t taps = op.kh * op.kw;
  for (uint32_t t = 0; t < taps; ++t) {
    buf[t] = w[order[t]];
  }
  return buf;
}

// ---- the plane path's two-rows-a-vector form ---------------------------------
// A 5x5 at a stride of 1 whose output rows are at most 8 pixels (MobileNet-v3
// Small's 7 x 7 layers) would fill 7 of a run's 16 lanes; dw5_pairs_s3 fills
// 14, two output rows a run. Its input is each padded row's five 8-byte
// windows (Z_j[t] = padded row t's bytes [j, j + 8)), plane j after plane
// j - 1, so a tap's two rows are 16 consecutive bytes: built per channel in
// a stack buffer (the plane path's pad row's), so no scratch at all. A raw
// sum is the same exact integer sum either way, so the bytes are the scalar
// loop's.

/// The stack buffer of the plane path: its pad row, or the pairs' windows.
constexpr uint32_t kPlaneBufBytes = 768;
static_assert(kPlaneBufBytes >= kPlanePadBytes, "the pad row fits");
/// The pairs' windows: kPairsRows padded rows a plane, 5 planes, a fixed
/// stride (constant offsets for the builder's stores).
constexpr uint32_t kPairsRows = 19;
constexpr uint32_t kPairsPlane = kPairsRows * 8;
static_assert(5 * kPairsPlane <= kPlaneBufBytes, "the windows fit");

/// Whether op takes dw5_pairs_s3, and *T, the padded rows its windows hold:
/// every row a run reads, the last run's second output row's too (even
/// when h_out is odd), at most kPairsRows.
inline bool pairs_geo(const Op& op, uint32_t* T) {
  if (!op.depthwise || op.kh != 5 || op.kw != 5 || op.dil_w != 1 ||
      op.stride_h != 1 || op.stride_w != 1 || op.h_out == 0 ||
      op.w_out == 0 || op.w_out > 8 || op.dil_h > 8) {
    return false;
  }
  *T = (op.h_out + 1) / 2 * 2 + 4 * op.dil_h;
  return *T <= kPairsRows;
}

/// One channel on the pairs form: its windows into z (T of a plane's
/// kPairsRows rows; a padded row off the input or past the window all pad),
/// then
/// dw5_pairs_s3 four runs a call, each run's two output rows' lanes
/// unpacked into a run of raw sums as depthwise_plane's, and finished.
/// Returns the elements the integer epilogue left to the float sequence.
template <int kMode, typename Src>
[[gnu::noinline]] uint32_t depthwise_pairs(const Op& op, Src src, int32_t pad,
                                           uint32_t T, int8_t* z,
                                           const int8_t* w,
                                           const Channel& ch, const Epi& e,
                                           float* y, int8_t* y8) {
  typedef uint32_t __attribute__((may_alias)) Word;
  constexpr uint32_t tw = kPairsPlane / 4;  // a plane, in words
  const uint32_t Wp = op.w_out + 4;
  const uint32_t q_lo = op.pad_w < Wp ? op.pad_w : Wp;
  const uint32_t q_end = op.pad_w + op.w_in < Wp ? op.pad_w + op.w_in : Wp;
  const uint32_t pad4 = (uint32_t)(uint8_t)pad * 0x01010101u;
  // The input rows the windows hold: padded rows [pad_h, pad_h + rows).
  const uint32_t t_lo = op.pad_h < T ? op.pad_h : T;
  const uint32_t rows = q_end > q_lo
                            ? (op.h_in < T - t_lo ? op.h_in : T - t_lo)
                            : 0;
  alignas(16) uint8_t prow[16];
  Word* const pw = reinterpret_cast<Word*>(prow);
  Word* zt = reinterpret_cast<Word*>(z);
  // A pad row's windows are all pad.
  for (uint32_t t = 0; t < T; ++t, zt += 2) {
    if (t - t_lo < rows) {
      continue;
    }
    for (uint32_t j = 0; j < 5; ++j) {
      zt[j * tw] = pad4;
      zt[j * tw + 1] = pad4;
    }
  }
  // An input row's: its padded bytes [0, 12), the windows' reach.
  zt = reinterpret_cast<Word*>(z) + 2 * t_lo;
  size_t at = (size_t)0 - op.pad_w;
  for (uint32_t r = 0; r < rows; ++r, zt += 2, at += op.w_in) {
    pw[0] = pad4;
    pw[1] = pad4;
    pw[2] = pad4;
    for (uint32_t q = q_lo; q < q_end; ++q) {
      prow[q] = (uint8_t)src(at + q);
    }
    const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
    zt[0] = w0;
    zt[1] = w1;
    zt[tw] = funnel<8>(w0, w1);
    zt[tw + 1] = funnel<8>(w1, w2);
    zt[2 * tw] = funnel<16>(w0, w1);
    zt[2 * tw + 1] = funnel<16>(w1, w2);
    zt[3 * tw] = funnel<24>(w0, w1);
    zt[3 * tw + 1] = funnel<24>(w1, w2);
    zt[4 * tw] = w1;
    zt[4 * tw + 1] = w2;
  }
  alignas(16) uint8_t img[4 * 64];
  int32_t raw[kRun + 8];
  const int8_t* win[5];
  const uint32_t runs = (op.h_out + 1) / 2;
  uint32_t slow = 0, fill = 0;
  size_t first = 0;
  for (uint32_t r0 = 0; r0 < runs; r0 += 4) {
    const uint32_t chunks = runs - r0 < 4 ? runs - r0 : 4;
    for (uint32_t i = 0; i < 5; ++i) {
      win[i] = z + (size_t)(i * op.dil_h + 2 * r0) * 8u;
    }
    dw5_pairs_s3(win, (int)kPairsPlane, w, img, (int)chunks);
    for (uint32_t k = 0; k < chunks; ++k) {
      if (fill + 2 * op.w_out > kRun) {
        slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
        first += fill;
        fill = 0;
      }
      unpack8(img + 64u * k, raw + fill);
      fill += op.w_out;
      if (2 * (r0 + k) + 1 < op.h_out) {
        unpack8(img + 64u * k + 32u, raw + fill);
        fill += op.w_out;
      }
    }
  }
  if (fill > 0) {
    slow += finish<kMode>(raw, fill, ch, e, y, y8, first);
  }
  return slow;
}

/// Channels [n, hi) of a depthwise conv in the plane path's pairs form, or
/// on the plane path itself, whichever applies (pairs_geo, plane_geo; the
/// plane path needs the plane's scratch, so not Stage B codes read in place,
/// `direct`). Returns the first channel it left to the scalar loop: hi, or
/// n when neither applies. Out of line, and called only when the plane
/// kernels may run, so its stack buffer is in no other conv's frames.
template <int kMode>
[[gnu::noinline]] uint32_t depthwise_kernels(const Op& op, const Act& a,
                                             const Columns& c, const Out& out,
                                             const Epi& e, const Scratch& sc,
                                             uint32_t n, uint32_t hi,
                                             bool direct, Counts* counts) {
  PlaneGeo pg;
  uint32_t pairs_t = 0;
  const bool pairs = pairs_geo(op, &pairs_t);
  const bool planes = !pairs && !direct && plane_geo(op, &pg);
  if (!pairs && !planes) {
    return n;
  }
  const size_t plane = (size_t)op.h_in * op.w_in;
  const Quant q = quant_of(a);
  alignas(16) int8_t buf[kPlaneBufBytes];  // the pad row, or the windows
  int8_t wk[25];
  counts->pie += (hi - n) * op.m;
  if (pairs) {
    for (; n < hi; ++n) {
      const Channel ch = channel(c, a.zx, n, e);
      float* const y = out.y != nullptr ? out.y + (size_t)n * op.m : nullptr;
      int8_t* const y8 =
          out.y8 != nullptr ? out.y8 + (size_t)n * op.m : nullptr;
      const int8_t* const w = c.wq + (size_t)n * op.k_pad;
      if (a.x8 != nullptr) {
        counts->fallback += depthwise_pairs<kMode>(
            op, Codes{a.x8 + (size_t)n * plane}, a.zx, pairs_t, buf, w, ch,
            e, y, y8);
      } else {
        counts->fallback += depthwise_pairs<kMode>(
            op, Quantised{a.x + (size_t)n * plane, q}, a.zx, pairs_t, buf, w,
            ch, e, y, y8);
      }
      if (kMode != kOutFloat && ch.fx.guard != 0) {
        counts->fixed += op.m;
      }
    }
    return hi;
  }
  int8_t* const pad_row = buf;
  fill_pad(pad_row, (uint32_t)(uint8_t)a.zx * 0x01010101u, kPlanePadBytes);
  for (; n < hi; ++n) {
    if (a.x8 != nullptr) {
      plane_rows(op, Codes{a.x8 + (size_t)n * plane}, a.zx, pg, sc.base);
    } else {
      plane_rows(op, Quantised{a.x + (size_t)n * plane, q}, a.zx, pg,
                 sc.base);
    }
    const Channel ch = channel(c, a.zx, n, e);
    counts->fallback += depthwise_plane<kMode>(
        op, pg, sc.base, pad_row,
        plane_weights(op, pg, c.wq + (size_t)n * op.k_pad, wk), ch, e,
        out.y != nullptr ? out.y + (size_t)n * op.m : nullptr,
        out.y8 != nullptr ? out.y8 + (size_t)n * op.m : nullptr);
    if (kMode != kOutFloat && ch.fx.guard != 0) {
      counts->fixed += op.m;
    }
  }
  return hi;
}

/// Output channels [part.lo, part.hi) of a depthwise conv: 16 at a time on
/// PIE where it applies, the rest one at a time.
template <int kMode>
void depthwise(const Op& op, const Act& a, const Columns& c, const Out& out,
               const Epi& e, const Scratch& sc, Part part, Counts* counts) {
  const size_t plane = (size_t)op.h_in * op.w_in;
  const uint32_t Hp = read_extent(op.h_out, op.kh, op.stride_h, op.dil_h);
  const uint32_t Wp = read_extent(op.w_out, op.kw, op.stride_w, op.dil_w);
  const bool direct = a.x8 != nullptr && all_interior(op);
  const Quant q = quant_of(a);
  uint32_t n = part.lo;
  if (c.pie && pie_dw_geometry(op) &&
      g_pie_dw_state.load(std::memory_order_acquire) == 2) {
    for (; n + 16 <= part.hi; n += 16) {
      if (a.x8 != nullptr) {
        depthwise_pie16<kMode>(op, Codes{a.x8}, a.zx, c, a.zx, e, out, n,
                               sc.base, counts);
      } else {
        depthwise_pie16<kMode>(op, Quantised{a.x, q}, a.zx, c, a.zx, e, out,
                               n, sc.base, counts);
      }
    }
    counts->pie += (n - part.lo) * op.m;
  }
  // The rest one channel at a time: on the plane path or in its pairs form
  // where one applies (depthwise_kernels), else the scalar loop.
  if (n < part.hi && c.pie &&
      g_plane_ok.load(std::memory_order_relaxed) != 0) {
    n = depthwise_kernels<kMode>(op, a, c, out, e, sc, n, part.hi, direct,
                                 counts);
  }
  counts->scalar += (part.hi - n) * op.m;
  for (; n < part.hi; ++n) {
    const int8_t* corner;
    size_t pitch;
    if (direct) {
      // Stage B codes, every tap inside: read in place.
      corner = a.x8 + (size_t)n * plane;
      pitch = op.w_in;
    } else {
      // This channel's codes, made once (fp32 quantised element by element,
      // as the gather would for every tap that reads it), in a plane padded
      // with the code a tap off the input reads.
      if (a.x8 != nullptr) {
        pad_channel(op, Codes{a.x8 + (size_t)n * plane}, a.zx, sc.base, Hp,
                    Wp);
      } else {
        pad_channel(op, Quantised{a.x + (size_t)n * plane, q}, a.zx, sc.base,
                    Hp, Wp);
      }
      corner = sc.base;
      pitch = Wp;
    }
    const Channel ch = channel(c, a.zx, n, e);
    counts->fallback += depthwise_channel<kMode>(
        op, corner, pitch, c.wq + (size_t)n * op.k_pad, ch, e,
        out.y != nullptr ? out.y + (size_t)n * op.m : nullptr,
        out.y8 != nullptr ? out.y8 + (size_t)n * op.m : nullptr);
    if (kMode != kOutFloat && ch.fx.guard != 0) {
      counts->fixed += op.m;
    }
  }
  counts->outputs += (uint64_t)(part.hi - part.lo) * op.m;
}

// ---- dense: the row gather --------------------------------------------------

/// Rows [r0, r0 + nrows) of a pointwise conv (1x1, no padding): row t holds
/// position r0 + t's c_in codes (its zeros to k_pad are dense_rows', made
/// once: no gather writes past column k).
template <typename Src>
[[gnu::noinline]] void gather_pointwise(const Op& op, Src src, uint32_t r0,
                                        uint32_t nrows, int8_t* rows) {
  const size_t plane = (size_t)op.h_in * op.w_in;
  const uint32_t kp = op.k_pad;
  if (op.stride_h == 1 && op.stride_w == 1) {
    // The output grid is the input's: position r reads element r of each
    // channel's plane, a transpose of a [c_in, nrows] tile.
    for (uint32_t ci = 0; ci < op.c_in; ++ci) {
      const size_t base = (size_t)ci * plane + r0;
      int8_t* d = rows + ci;
      for (uint32_t t = 0; t < nrows; ++t, d += kp) {
        *d = src(base + t);
      }
    }
  } else {
    uint32_t oh = r0 / op.w_out, ow = r0 - oh * op.w_out;
    for (uint32_t t = 0; t < nrows; ++t) {
      const size_t pos = (size_t)oh * op.stride_h * op.w_in +
          (size_t)ow * op.stride_w;
      int8_t* d = rows + (size_t)t * kp;
      for (uint32_t ci = 0; ci < op.c_in; ++ci) {
        d[ci] = src((size_t)ci * plane + pos);
      }
      if (++ow == op.w_out) {
        ow = 0;
        ++oh;
      }
    }
  }
}

/// gather_pointwise for Stage B codes (the op's c_in planes at x8): with PIE
/// on (Columns::pie) and at stride 1, each group of 16 channels transposed
/// into the rows by transpose16_s3 once it passed its check -- a [16, nrows]
/// tile of the planes to nrows rows' 16 columns -- and the channels past the
/// last 16 (or a group whose transposes would read past the last plane) as
/// gather_pointwise copies them. The rows are gather_pointwise's, byte for
/// byte; with PIE off every element is copied, as the scalar path's dots.
[[gnu::noinline]] void gather_pointwise_codes(const Op& op, const int8_t* x8,
                                              bool pie, uint32_t r0,
                                              uint32_t nrows, int8_t* rows) {
  if (!pie || op.stride_h != 1 || op.stride_w != 1 ||
      g_transpose_ok.load(std::memory_order_relaxed) == 0 || nrows == 0) {
    gather_pointwise(op, Codes{x8}, r0, nrows, rows);
    return;
  }
  const size_t plane = (size_t)op.h_in * op.w_in;
  const uint32_t kp = op.k_pad;
  const int8_t* const end = x8 + (size_t)op.c_in * plane;
  // The kernel's reads: whole 16-byte tiles from each channel's first
  // position, and the next aligned block.
  const size_t reach = (size_t)(nrows - 1) / 16u * 16u + 32u;
  uint32_t ci = 0;
  for (; ci + 16 <= op.c_in; ci += 16) {
    const int8_t* src = x8 + (size_t)ci * plane + r0;
    if (src + 15 * plane + reach > end) {
      break;
    }
    transpose16_s3(src, (int)plane, rows + ci, (int)kp, (int)nrows);
  }
  for (; ci < op.c_in; ++ci) {
    const int8_t* src = x8 + (size_t)ci * plane + r0;
    int8_t* d = rows + ci;
    for (uint32_t t = 0; t < nrows; ++t, d += kp) {
      *d = src[t];
    }
  }
}

/// gather_pointwise at stride 1 for an fp32 input by QuantFast: each
/// channel's run of positions, four a step written out lane by lane (GCC
/// keeps the source order on the LX7, so the four chains interleave), each
/// code stored to its row. The rows are gather_pointwise's, byte for byte.
[[gnu::noinline]] void gather_pointwise_fast(const Op& op, const float* x,
                                             const QuantFast& qf, uint32_t r0,
                                             uint32_t nrows, int8_t* rows) {
  ESP32_QUANT_FAST_CALL(g_fast_pointwise);
  const QuantFastRegs q(qf);
  const size_t plane = (size_t)op.h_in * op.w_in;
  const size_t kp = op.k_pad;
  for (uint32_t ci = 0; ci < op.c_in; ++ci) {
    const float* s = x + (size_t)ci * plane + r0;
    int8_t* d = rows + ci;
    uint32_t t = 0;
    for (; t + 4 <= nrows; t += 4, s += 4, d += 4 * kp) {
      float a = s[0] * q.inv, b = s[1] * q.inv;
      float c = s[2] * q.inv, e = s[3] * q.inv;
      a = quant_scaled(a, q.zero);
      b = quant_scaled(b, q.zero);
      c = quant_scaled(c, q.zero);
      e = quant_scaled(e, q.zero);
      a = a < q.lo ? q.lo : a;
      b = b < q.lo ? q.lo : b;
      c = c < q.lo ? q.lo : c;
      e = e < q.lo ? q.lo : e;
      a = a > q.hi ? q.hi : a;
      b = b > q.hi ? q.hi : b;
      c = c > q.hi ? q.hi : c;
      e = e > q.hi ? q.hi : e;
      d[0] = q.code(a);
      d[kp] = q.code(b);
      d[2 * kp] = q.code(c);
      d[3 * kp] = q.code(e);
    }
    for (; t < nrows; ++t, ++s, d += kp) {
      *d = q(*s);
    }
  }
}

/// The rows of a pointwise conv: gather_pointwise_codes for codes, else
/// (fp32 quantised as each element is read) gather_pointwise, at stride 1
/// gather_pointwise_fast where the record allows it.
inline void pointwise_rows(const Op& op, Codes src, bool pie, uint32_t r0,
                           uint32_t nrows, int8_t* rows) {
  gather_pointwise_codes(op, src.p, pie, r0, nrows, rows);
}
inline void pointwise_rows(const Op& op, Quantised src, bool pie, uint32_t r0,
                           uint32_t nrows, int8_t* rows) {
  gather_pointwise(op, src, r0, nrows, rows);
}
inline void pointwise_rows(const Op& op, QuantisedFast src, bool pie,
                           uint32_t r0, uint32_t nrows, int8_t* rows) {
  if (op.stride_h == 1 && op.stride_w == 1) {
    QuantFast qf;
    qf.inv = src.q.inv;
    qf.lo = src.q.lo;
    qf.hi = src.q.hi;
    qf.bias = src.q.bias;
    gather_pointwise_fast(op, src.x, qf, r0, nrows, rows);
  } else {
    gather_pointwise(op, src, r0, nrows, rows);
  }
}

/// Rows [r0, r0 + nrows) of a dense conv, in the gather's column order
/// (c, i, j): a tap off the input reads `pad` (the zeros to k_pad are
/// dense_rows').
template <typename Src>
[[gnu::noinline]] void gather_dense(const Op& op, Src src, int32_t pad,
                                    uint32_t r0, uint32_t nrows, int8_t* rows,
                                    uint32_t oh_lo, uint32_t oh_hi,
                                    uint32_t ow_lo, uint32_t ow_hi) {
  const size_t plane = (size_t)op.h_in * op.w_in;
  const uint32_t W = op.w_in, H = op.h_in;
  const uint32_t kh = op.kh, kw = op.kw, dh = op.dil_h, dw = op.dil_w;
  uint32_t oh = r0 / op.w_out, ow = r0 - oh * op.w_out;
  for (uint32_t t = 0; t < nrows; ++t) {
    int8_t* d = rows + (size_t)t * op.k_pad;
    const int64_t ih0 = (int64_t)oh * op.stride_h - op.pad_h;
    const int64_t iw0 = (int64_t)ow * op.stride_w - op.pad_w;
    if (oh >= oh_lo && oh < oh_hi && ow >= ow_lo && ow < ow_hi) {
      const size_t corner = (size_t)ih0 * W + (size_t)iw0;
      for (uint32_t ci = 0; ci < op.c_in; ++ci) {
        size_t rowp = (size_t)ci * plane + corner;
        for (uint32_t i = 0; i < kh; ++i, rowp += (size_t)dh * W) {
          if (kw == 3 && dw == 1) {
            d[0] = src(rowp);
            d[1] = src(rowp + 1);
            d[2] = src(rowp + 2);
            d += 3;
          } else {
            for (uint32_t j = 0; j < kw; ++j) {
              *d++ = src(rowp + (size_t)j * dw);
            }
          }
        }
      }
    } else {
      for (uint32_t ci = 0; ci < op.c_in; ++ci) {
        const size_t cbase = (size_t)ci * plane;
        for (uint32_t i = 0; i < kh; ++i) {
          const int64_t ih = ih0 + (int64_t)i * dh;
          const bool ok = ih >= 0 && ih < (int64_t)H;
          for (uint32_t j = 0; j < kw; ++j) {
            const int64_t iw = iw0 + (int64_t)j * dw;
            *d++ = (ok && iw >= 0 && iw < (int64_t)W)
                ? src(cbase + (size_t)ih * W + (size_t)iw)
                : (int8_t)pad;
          }
        }
      }
    }
    if (++ow == op.w_out) {
      ow = 0;
      ++oh;
    }
  }
}

/// gather_dense for an fp32 input by QuantFast, each input element
/// quantised once per run of a row's windows where they overlap. The rows
/// go a run at a time: the positions of one output row whose windows lie
/// inside the input, one corner pointer moving stride_w columns a position.
/// Within a run, position r's window is r - 1's moved stride_w columns on,
/// so where they overlap (no column dilation, stride_w < kw) the first
/// kw - stride_w columns of each (c, i) are r - 1's last ones, copied from
/// the row before, and only the others are quantised: a 3x3 stride-2 conv's
/// 27 taps a position take 18 quantisations, not 27. A code is a function of
/// its element alone, so the rows are gather_dense's, byte for byte; the
/// positions with a tap off the input take gather_dense's loop, by the same
/// quantiser.
[[gnu::noinline]] void gather_dense_fast(const Op& op, const float* x,
                                         const QuantFast& qf, int32_t pad,
                                         uint32_t r0, uint32_t nrows,
                                         int8_t* rows, uint32_t oh_lo,
                                         uint32_t oh_hi, uint32_t ow_lo,
                                         uint32_t ow_hi) {
  ESP32_QUANT_FAST_CALL(g_fast_dense);
  const QuantFastRegs q(qf);
  const size_t plane = (size_t)op.h_in * op.w_in;
  const uint32_t W = op.w_in, H = op.h_in;
  const uint32_t kh = op.kh, kw = op.kw, dh = op.dil_h, dw = op.dil_w;
  const uint32_t sw = op.stride_w;
  const size_t kp = op.k_pad;
  const size_t pitch = (size_t)dh * W;  // one kernel row to the next
  const uint32_t planes = op.c_in;
  const bool slide = dw == 1 && sw < kw;
  const uint32_t keep = slide ? kw - sw : 0;
  uint32_t oh = r0 / op.w_out, ow = r0 - oh * op.w_out;
  uint32_t t = 0;
  while (t < nrows) {
    int8_t* d = rows + (size_t)t * kp;
    if (oh >= oh_lo && oh < oh_hi && ow >= ow_lo && ow < ow_hi) {
      uint32_t run = ow_hi - ow;
      run = run < nrows - t ? run : nrows - t;
      const float* corner =
          x + (size_t)(oh * op.stride_h - op.pad_h) * W + (ow * sw - op.pad_w);
      for (uint32_t u = 0; u < run; ++u, corner += sw, d += kp) {
        int8_t* o = d;
        if (u == 0 || !slide) {
          // The run's first window (its row before is a border position or
          // not in this block), or no overlap: every tap.
          for (uint32_t ci = 0; ci < planes; ++ci) {
            const float* xr = corner + (size_t)ci * plane;
            for (uint32_t i = 0; i < kh; ++i, xr += pitch, o += kw) {
              for (uint32_t j = 0; j < kw; ++j) {
                o[j] = q(xr[(size_t)j * dw]);
              }
            }
          }
        } else if (kw == 3 && sw == 2) {
          // Column 0 is the row before's column 2; columns 1 and 2 new.
          const float* xc = corner;
          for (uint32_t ci = 0; ci < planes; ++ci, xc += plane) {
            const float* xr = xc;
            for (uint32_t i = 0; i < kh; ++i, xr += pitch, o += 3) {
              float a = xr[1] * q.inv, b = xr[2] * q.inv;
              a = quant_scaled(a, q.zero);
              b = quant_scaled(b, q.zero);
              a = a < q.lo ? q.lo : a;
              b = b < q.lo ? q.lo : b;
              a = a > q.hi ? q.hi : a;
              b = b > q.hi ? q.hi : b;
              o[0] = o[2 - (ptrdiff_t)kp];
              o[1] = q.code(a);
              o[2] = q.code(b);
            }
          }
        } else {
          for (uint32_t ci = 0; ci < planes; ++ci) {
            const float* xr = corner + (size_t)ci * plane;
            for (uint32_t i = 0; i < kh; ++i, xr += pitch, o += kw) {
              uint32_t j = 0;
              for (; j < keep; ++j) {
                o[j] = o[(ptrdiff_t)(j + sw) - (ptrdiff_t)kp];
              }
              for (; j < kw; ++j) {
                o[j] = q(xr[j]);
              }
            }
          }
        }
      }
      t += run;
      ow += run;
    } else {
      const int64_t ih0 = (int64_t)oh * op.stride_h - op.pad_h;
      const int64_t iw0 = (int64_t)ow * sw - op.pad_w;
      for (uint32_t ci = 0; ci < planes; ++ci) {
        const float* xc = x + (size_t)ci * plane;
        for (uint32_t i = 0; i < kh; ++i) {
          const int64_t ih = ih0 + (int64_t)i * dh;
          const bool ok = ih >= 0 && ih < (int64_t)H;
          for (uint32_t j = 0; j < kw; ++j) {
            const int64_t iw = iw0 + (int64_t)j * dw;
            *d++ = (ok && iw >= 0 && iw < (int64_t)W)
                ? q(xc[(size_t)ih * W + (size_t)iw])
                : (int8_t)pad;
          }
        }
      }
      ++t;
      ++ow;
    }
    if (ow == op.w_out) {
      ow = 0;
      ++oh;
    }
  }
}

/// A dense conv's rows (not pointwise): gather_dense, or for an fp32 input
/// by QuantFast, gather_dense_fast.
template <typename Src>
inline void dense_gather(const Op& op, Src src, int32_t pad, uint32_t r0,
                         uint32_t nrows, int8_t* rows, uint32_t oh_lo,
                         uint32_t oh_hi, uint32_t ow_lo, uint32_t ow_hi) {
  gather_dense(op, src, pad, r0, nrows, rows, oh_lo, oh_hi, ow_lo, ow_hi);
}
inline void dense_gather(const Op& op, QuantisedFast src, int32_t pad,
                         uint32_t r0, uint32_t nrows, int8_t* rows,
                         uint32_t oh_lo, uint32_t oh_hi, uint32_t ow_lo,
                         uint32_t ow_hi) {
  QuantFast qf;
  qf.inv = src.q.inv;
  qf.lo = src.q.lo;
  qf.hi = src.q.hi;
  qf.bias = src.q.bias;
  gather_dense_fast(op, src.x, qf, pad, r0, nrows, rows, oh_lo, oh_hi, ow_lo,
                    ow_hi);
}

// ---- dense: the columns -----------------------------------------------------

inline uint32_t dot_scalar(const int8_t* x, const int8_t* w, uint32_t len) {
  uint32_t acc = 0;
  for (uint32_t t = 0; t < len; ++t) {
    acc += (uint32_t)((int32_t)x[t] * (int32_t)w[t]);
  }
  return acc;
}

/// The most 16-byte weight chunks dot_rows_codes_s3 holds in registers
/// (q1-q7): k_pad up to 112.
constexpr uint32_t kRowsCodesChunks = 7;

/// Whether dot_rows_codes_s3 passed its start-up check (pie_dw_init); until
/// then, and if it failed, the columns take dot_i8_rows_s3 and finish.
std::atomic<int> g_rows_codes_ok{0};

/// The same for dot_rows_long_s3, which takes the rows past kRowsCodesChunks
/// (pie_dw_init's rows_long_kernel_ok).
std::atomic<int> g_rows_long_ok{0};

/// The most chunks dot_rows_long_s3 takes (k_pad up to 1024, past every conv
/// of the bench models: k=960 is 60): its boot check covers every count from
/// 8 to this, and a longer row keeps dot_i8_rows_s3 and finish.
constexpr uint32_t kRowsLongChunks = 64;

/// A channel's terms for dot_rows_codes_s3: lane_epi's, and its chunks.
inline Esp32RowEpi row_epi(const Channel& ch, const Epi& e, uint32_t chunks) {
  Esp32RowEpi t;
  t.g = ch.fx.g;
  t.c_lo = (uint32_t)(uint64_t)ch.fx.c_raw;
  t.c_hi = (int32_t)(ch.fx.c_raw >> 32);
  t.guard = ch.fx.guard;
  t.qlo = e.code_lo;
  t.qhi = e.code_hi;
  t.chunks = chunks;
  return t;
}

/// `cnt` (<= kRun) rows of one channel by dot_rows_codes_s3, codes out into
/// y8 (or, kOutRequant, into a run of codes then fp32 at y): the dot,
/// finish_fixed's integer epilogue and the clamp in one loop, with the
/// weights in registers; the elements it noted near a boundary then take
/// the float sequence on the same dot (dot_i8_s3). Returns how many did.
template <int kMode>
[[gnu::noinline]] uint32_t rows_codes(const int8_t* row, const int8_t* w,
                                      uint32_t kp, uint32_t cnt,
                                      const Esp32RowEpi& t, const Channel& ch,
                                      const Epi& e, float* y, int8_t* y8) {
  int8_t codes[kRun];
  int8_t* const o = kMode == kOutCodes ? y8 : codes;
  int8_t* slow[kRun];
  // Rows past the registers' chunks take dot_rows_long_s3, the same
  // arithmetic with the weights past the sixth chunk streamed.
  const int n = t.chunks > kRowsCodesChunks
      ? dot_rows_long_s3(row, w, o, (int)cnt, &t, slow)
      : dot_rows_codes_s3(row, w, o, (int)cnt, &t, slow);
  for (int i = 0; i < n; ++i) {
    const size_t u = (size_t)(slow[i] - o);
    const int32_t raw = dot_i8_s3(row + u * kp, w, (int)kp);
    const int32_t acc = (int32_t)((uint32_t)raw - ch.zxs);
    o[u] = (int8_t)float_code(acc, ch.s, ch.b, e);
  }
  if (kMode == kOutRequant) {
    const int32_t zo = e.zo;
    const float out_scale = e.out_scale;
    for (uint32_t u = 0; u < cnt; ++u) {
      y[u] = (float)((int32_t)o[u] - zo) * out_scale;
    }
  }
  return (uint32_t)n;
}

/// Every column over a block of rows: column n's outputs for positions
/// [r0, r0 + nrows) are contiguous in NCHW (n * m + r), so they are written
/// in place -- no staging, no scatter. The block's dots first, then their
/// epilogue (finish); or, for a channel with the integer form and its codes
/// within int8, both at once with the weights in registers (rows_codes).
template <int kMode>
[[gnu::noinline]] void columns(const Op& op, const Columns& c, int32_t zx,
                               const Epi& e, const int8_t* rows, uint32_t r0,
                               uint32_t nrows, const Out& out, Counts* counts) {
  const uint32_t kp = op.k_pad;
  int32_t raw[kRun];
  uint64_t slow = 0, fixed = 0;
  // dot_rows_codes_s3 applies: PIE, a short row, codes out (or fp32 from
  // codes that fit its int8 run), and the kernel checked at start-up; for a
  // longer row, dot_rows_long_s3 with its own check.
  const bool int8_run =
      kMode != kOutFloat && c.pie &&
      (kMode == kOutCodes || (e.qmin >= -128 && e.qmax <= 127));
  const bool fusable =
      int8_run &&
      (kp <= 16 * kRowsCodesChunks
           ? g_rows_codes_ok.load(std::memory_order_relaxed) != 0
           : kp <= 16 * kRowsLongChunks &&
                 g_rows_long_ok.load(std::memory_order_relaxed) != 0);
  for (uint32_t n = 0; n < op.n; ++n) {
    const int8_t* w = c.wq + (size_t)n * kp;
    const Channel ch = channel(c, zx, n, e);
    if (kMode != kOutFloat && ch.fx.guard != 0) {
      fixed += nrows;
    }
    if (fusable && ch.fx.guard != 0) {
      const Esp32RowEpi t = row_epi(ch, e, kp / 16u);
      for (uint32_t t0 = 0; t0 < nrows; t0 += kRun) {
        const uint32_t cnt = nrows - t0 < kRun ? nrows - t0 : kRun;
        const size_t at = (size_t)n * op.m + r0 + t0;
        slow += rows_codes<kMode>(rows + (size_t)t0 * kp, w, kp, cnt, t, ch,
                                  e, out.y != nullptr ? out.y + at : nullptr,
                                  out.y8 != nullptr ? out.y8 + at : nullptr);
      }
      continue;
    }
    for (uint32_t t0 = 0; t0 < nrows; t0 += kRun) {
      const uint32_t cnt = nrows - t0 < kRun ? nrows - t0 : kRun;
      const int8_t* row = rows + (size_t)t0 * kp;
      if (c.pie) {
        dot_i8_rows_s3(row, (int)kp, w, (int)kp, raw, (int)cnt);
      } else {
        for (uint32_t t = 0; t < cnt; ++t, row += kp) {
          raw[t] = (int32_t)dot_scalar(row, w, kp);
        }
      }
      slow += finish<kMode>(raw, cnt, ch, e, out.y, out.y8,
                            (size_t)n * op.m + r0 + t0);
    }
  }
  counts->fixed += fixed;
  counts->fallback += slow;
  if (c.pie) {
    counts->pie += op.n * nrows;
  } else {
    counts->scalar += op.n * nrows;
  }
  counts->outputs += (uint64_t)op.n * nrows;
}

/// Whether a dense conv's fp32 input is quantised whole, before its rows.
inline bool dense_plane(const Op& op) {
  const bool pointwise = op.kh == 1 && op.kw == 1 && op.pad_h == 0 &&
                         op.pad_w == 0;
  return !pointwise &&
         (uint64_t)op.c_in * op.h_in * op.w_in <= kPlaneBudget;
}

/// Output positions [part.lo, part.hi) of a dense conv, `sc.rows` a block.
template <int kMode, typename Src>
void dense_rows(const Op& op, Src src, int32_t pad, const Columns& c,
                int32_t zx, const Epi& e, const Out& out, const Scratch& sc,
                Part part, Counts* counts) {
  const bool pointwise = op.kh == 1 && op.kw == 1 && op.pad_h == 0 &&
                         op.pad_w == 0;
  uint32_t oh_lo = 0, oh_hi = 0, ow_lo = 0, ow_hi = 0;
  if (!pointwise) {
    interior(op.h_out, op.h_in, op.kh, op.stride_h, op.pad_h, op.dil_h,
             &oh_lo, &oh_hi);
    interior(op.w_out, op.w_in, op.kw, op.stride_w, op.pad_w, op.dil_w,
             &ow_lo, &ow_hi);
  }
  const uint32_t blk = sc.rows > 0 ? sc.rows : 1;
  // Each row's zeros from column k to k_pad, once for every block: the
  // gathers write columns [0, k) only, and nothing else writes the rows.
  if (op.k < op.k_pad) {
    const uint32_t tail = op.k_pad - op.k;
    const uint32_t first = part.hi - part.lo < blk ? part.hi - part.lo : blk;
    for (uint32_t t = 0; t < first; ++t) {
      int8_t* z = sc.base + (size_t)t * op.k_pad + op.k;
      for (uint32_t j = 0; j < tail; ++j) {
        z[j] = 0;
      }
    }
  }
  for (uint32_t r0 = part.lo; r0 < part.hi; r0 += blk) {
    const uint32_t nrows = (r0 + blk <= part.hi) ? blk : (part.hi - r0);
    if (pointwise) {
      pointwise_rows(op, src, c.pie, r0, nrows, sc.base);
    } else {
      dense_gather(op, src, pad, r0, nrows, sc.base, oh_lo, oh_hi, ow_lo,
                   ow_hi);
    }
    columns<kMode>(op, c, zx, e, sc.base, r0, nrows, out, counts);
  }
}

/// Output positions [part.lo, part.hi) of a dense conv: Stage B codes read
/// where they are, prepare's plane (an fp32 input quantised whole), or each
/// tap's code as its row is gathered -- once per element for a stride-1
/// pointwise conv, and no plane to hold for a large input.
template <int kMode>
void dense(const Op& op, const Act& a, const Columns& c, const Out& out,
           const Epi& e, const Scratch& sc, Part part, Counts* counts) {
  if (a.x8 != nullptr) {
    dense_rows<kMode>(op, Codes{a.x8}, a.zx, c, a.zx, e, out, sc, part,
                      counts);
  } else if (sc.plane != nullptr) {
    dense_rows<kMode>(op, Codes{sc.plane}, a.zx, c, a.zx, e, out, sc, part,
                      counts);
  } else if (quant_fast_ok(a)) {
    dense_rows<kMode>(op, QuantisedFast{a.x, QuantFastRegs(quant_fast_of(a))},
                      a.zx, c, a.zx, e, out, sc, part, counts);
  } else {
    dense_rows<kMode>(op, Quantised{a.x, quant_of(a)}, a.zx, c, a.zx, e, out,
                      sc, part, counts);
  }
}

template <int kMode>
void run_mode(const Op& op, const Act& a, const Columns& c, const Out& out,
              const Epi& e, const Scratch& sc, Part part, Counts* counts) {
  if (op.depthwise) {
    depthwise<kMode>(op, a, c, out, e, sc, part, counts);
  } else {
    dense<kMode>(op, a, c, out, e, sc, part, counts);
  }
}

/// The integer epilogue's per-op terms (Epi::fixed, code_lo, code_hi, amax):
/// a code out, positive in-range scales and an ordered clamp.
void fixed_epi(const Op& op, const Act& act, const Out& out, Epi* e) {
  e->fixed = (out.y8 != nullptr || out.requant) &&
             scale_in_range(e->sx) && scale_in_range(e->out_inv) &&
             e->lo <= e->hi;
  e->code_lo = code_of(e->lo, e->out_inv, e->zo, e->qmin, e->qmax);
  e->code_hi = code_of(e->hi, e->out_inv, e->zo, e->qmin, e->qmax);
  // |acc| = |sum (x - zx) * w| over op.k taps, every |x| <= 128 and
  // |w| <= 128 (int8 codes).
  const int32_t zx = act.zx < 0 ? -act.zx : act.zx;
  e->amax = (float)op.k * (float)(128 + zx) * 128.0f;
  // fixed_of's per-op factors: g~ = fl(s * fl(sx * out_inv)) has the two
  // roundings fl(fl(sx * s) * out_inv) would, and the guard is
  // 1.001 * (amax * 1795 * g~ + 771 * |c~| + amax + 4), its coefficients
  // here (each a few roundings off, far inside the 1.001).
  e->sx_out = e->sx * e->out_inv;
  e->guard_g = e->amax * 1795.0f * 1.001f;
  e->guard_c = 771.0f * 1.001f;
  e->guard_0 = (e->amax + 4.0f) * 1.001f;
}

}  // namespace

uint64_t scratch_need(const Op& op, bool fp32_in, uint32_t rows) {
  if (op.depthwise) {
    // One padded channel plane, unless Stage B codes are read in place; and
    // the PIE path's strip and buffers where it applies.
    uint64_t need = 0;
    if (fp32_in || !all_interior(op)) {
      need = (uint64_t)read_extent(op.h_out, op.kh, op.stride_h, op.dil_h) *
             read_extent(op.w_out, op.kw, op.stride_w, op.dil_w);
    }
    if (pie_dw_geometry(op)) {
      const uint64_t pie = pie_dw_scratch(op);
      need = pie > need ? pie : need;
    }
    return need;
  }
  uint64_t need = (uint64_t)rows * op.k_pad;
  if (fp32_in && dense_plane(op)) {
    need = align16(need) + (uint64_t)op.c_in * op.h_in * op.w_in;
  }
  return need;
}

uint32_t split_at(const Op& op) {
  const uint32_t all = units(op);
  uint32_t half = all / 2;
  if (pie_dw_geometry(op)) {
    // Whole groups of 16 on each side where there are two.
    const uint32_t even = (half + 8u) / 16u * 16u;
    if (even > 0 && even < all) {
      half = even;
    }
  }
  return half;
}

namespace {

/// lane_kernel_ok, after this buffer: qacc_lane_codes_s3 against the epilogue
/// it implements, computed from qacc_lanes' reading of the same bytes: 8 images
/// of pseudo-random bytes (every lane's 20 bits random: both signs, the extremes
/// reachable), every lane, and three sets of terms -- a typical channel; the
/// largest gain with a guard that sends most elements to slow[] and a clamp
/// that binds; and a constant whose low word sits next to a boundary. Codes and
/// slow[] must both match.
/// The buffer: the kernel checks' inputs, one after the other (16-byte aligned).
alignas(16) uint8_t g_kernel_check[1024];

bool lane_kernel_ok(uint32_t r) {
  uint8_t* const img = g_kernel_check;  // 8 images of 64 bytes
  constexpr uint32_t kImgBytes = 8 * 64;
  for (uint32_t i = 0; i < kImgBytes; ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    img[i] = (uint8_t)(r >> 24);
  }
  const Esp32LaneEpi sets[3] = {
      {0x00123456, 0x80000000u, -3, 1u << 20, -128, 127, 0},
      {0x3FFFFFFF, 0x00000000u, 0, 1u << 30, -5, 9, 0},
      {0x00000001, 0xFFFFFFF0u, -1, 0x11u, -128, 127, 0},
  };
  bool ok = true;
  for (uint32_t lane = 0; lane < 16 && ok; ++lane) {
    const LaneAt l = lane_at(lane);
    for (const Esp32LaneEpi& base : sets) {
      Esp32LaneEpi t = base;
      t.shift = l.shift;
      int8_t got[8];
      const uint8_t* slow[8];
      const int n = qacc_lane_codes_s3(img + l.off, 8, got, &t, slow);
      const int64_t c = (int64_t)(((uint64_t)(uint32_t)t.c_hi << 32) | t.c_lo);
      int want_n = 0;
      for (uint32_t u = 0; u < 8; ++u) {
        int32_t lanes[16];
        qacc_lanes(img + u * 64u, lanes);
        const int64_t y = (int64_t)lanes[lane] * t.g + c;
        const uint32_t lo = (uint32_t)y;
        const uint32_t d = (int32_t)lo < 0 ? 0u - lo : lo;
        if (d < t.guard) {
          ok = ok && want_n < n && slow[want_n] == img + u * 64u + l.off;
          ++want_n;
        }
        int32_t q = (int32_t)(y >> 32);
        q = q < t.qlo ? t.qlo : (q > t.qhi ? t.qhi : q);
        ok = ok && got[u] == (int8_t)q;
      }
      ok = ok && n == want_n;
    }
  }
  return ok;
}

/// dot_rows_codes_s3 against dot_i8_rows_s3 and the epilogue computed here,
/// for every chunk count it takes: 8 rows of pseudo-random codes, -128 every
/// seventh, against weights the same way, and three sets of terms as
/// lane_kernel_ok's (scaled to the longest dot). Codes and slow[] must match.
bool rows_kernel_ok(uint32_t r) {
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kRowBytes = kRows * 16 * kRowsCodesChunks;
  static_assert(kRowBytes + 16 * kRowsCodesChunks <= sizeof(g_kernel_check),
                "the rows and the weights fit the check's buffer");
  int8_t* const rows = reinterpret_cast<int8_t*>(g_kernel_check);
  int8_t* const w = rows + kRowBytes;
  int32_t raw[kRows];
  for (uint32_t i = 0; i < kRowBytes; ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    rows[i] = (int8_t)(i % 7 == 0 ? -128 : (int32_t)(r >> 24) - 128);
  }
  for (uint32_t i = 0; i < 16 * kRowsCodesChunks; ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    w[i] = (int8_t)(i % 5 == 0 ? -128 : (int32_t)(r >> 24) - 128);
  }
  const Esp32RowEpi sets[3] = {
      {0x00003456, 0x80000000u, -3, 1u << 20, -128, 127, 0},
      {0x00FFFFFF, 0x00000000u, 0, 1u << 30, -5, 9, 0},
      {0x00000001, 0xFFFFFFF0u, -1, 0x11u, -128, 127, 0},
  };
  bool ok = true;
  for (uint32_t chunks = 1; chunks <= kRowsCodesChunks && ok; ++chunks) {
    const uint32_t kp = 16u * chunks;
    dot_i8_rows_s3(rows, (int)kp, w, (int)kp, raw, (int)kRows);
    for (const Esp32RowEpi& base : sets) {
      Esp32RowEpi t = base;
      t.chunks = chunks;
      int8_t got[kRows];
      int8_t* slow[kRows];
      const int n = dot_rows_codes_s3(rows, w, got, (int)kRows, &t, slow);
      const int64_t c = (int64_t)(((uint64_t)(uint32_t)t.c_hi << 32) | t.c_lo);
      int want_n = 0;
      for (uint32_t u = 0; u < kRows; ++u) {
        const int64_t y = (int64_t)raw[u] * t.g + c;
        const uint32_t lo = (uint32_t)y;
        const uint32_t d = (int32_t)lo < 0 ? 0u - lo : lo;
        if (d < t.guard) {
          ok = ok && want_n < n && slow[want_n] == got + u;
          ++want_n;
        }
        int32_t q = (int32_t)(y >> 32);
        q = q < t.qlo ? t.qlo : (q > t.qhi ? t.qhi : q);
        ok = ok && got[u] == (int8_t)q;
      }
      ok = ok && n == want_n;
    }
  }
  return ok;
}

/// dot_rows_long_s3 against the dot computed here and the epilogue, for
/// every chunk count it takes (8 to kRowsLongChunks) and 1 to 4 rows:
/// pseudo-random codes, -128 every seventh, against weights the same way, on
/// a staging buffer of its
/// own (freed after; without it the kernel stays off), and three sets of
/// terms as rows_kernel_ok's. Codes and slow[] must match, and no code past
/// the rows may change.
bool rows_long_kernel_ok(uint32_t r) {
  constexpr uint32_t kRows = 4;
  constexpr uint32_t kMost = kRowsLongChunks;  // chunks
  constexpr uint32_t kRowBytes = kRows * 16 * kMost;
  uint8_t* const mem = static_cast<uint8_t*>(
      heap_caps_malloc(kRowBytes + 16 * kMost + 15, MALLOC_CAP_8BIT));
  if (mem == nullptr) {
    return false;
  }
  int8_t* const buf =
      reinterpret_cast<int8_t*>(((uintptr_t)mem + 15) & ~(uintptr_t)15);
  for (uint32_t i = 0; i < kRowBytes + 16 * kMost; ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    buf[i] = (int8_t)(i % 7 == 0 ? -128 : (int32_t)(r >> 24) - 128);
  }
  const Esp32RowEpi sets[3] = {
      {0x00000456, 0x80000000u, -3, 1u << 20, -128, 127, 0},
      {0x000FFFFF, 0x00000000u, 0, 1u << 30, -5, 9, 0},
      {0x00000001, 0xFFFFFFF0u, -1, 0x11u, -128, 127, 0},
  };
  bool ok = true;
  for (uint32_t chunks = 8; chunks <= kMost && ok; ++chunks) {
    const uint32_t kp = 16u * chunks;
    const int8_t* const w = buf + kRowBytes;
    for (uint32_t rows = 1; rows <= kRows && ok; ++rows) {
      for (const Esp32RowEpi& base : sets) {
        Esp32RowEpi t = base;
        t.chunks = chunks;
        int8_t got[kRows + 1];
        int8_t* slow[kRows];
        memset(got, 0x5A, sizeof(got));
        const int n = dot_rows_long_s3(buf, w, got, (int)rows, &t, slow);
        const int64_t c =
            (int64_t)(((uint64_t)(uint32_t)t.c_hi << 32) | t.c_lo);
        int want_n = 0;
        for (uint32_t u = 0; u < rows; ++u) {
          const int32_t raw =
              (int32_t)dot_scalar(buf + (size_t)u * kp, w, kp);
          const int64_t y = (int64_t)raw * t.g + c;
          const uint32_t lo = (uint32_t)y;
          const uint32_t d = (int32_t)lo < 0 ? 0u - lo : lo;
          if (d < t.guard) {
            ok = ok && want_n < n && slow[want_n] == got + u;
            ++want_n;
          }
          int32_t q = (int32_t)(y >> 32);
          q = q < t.qlo ? t.qlo : (q > t.qhi ? t.qhi : q);
          ok = ok && got[u] == (int8_t)q;
        }
        ok = ok && n == want_n && got[rows] == (int8_t)0x5A;
      }
    }
  }
  heap_caps_free(mem);
  return ok;
}

/// transpose16_s3 against its copy, element by element, on a staging buffer
/// of its own (freed after; without it the transpose stays off): 16 planes
/// of pseudo-random codes 37 bytes apart, so every channel starts at another
/// alignment, from two source alignments, into rows 16, 24 and 64 bytes
/// apart -- a strip's pixels and pointwise rows, whose tiles are 16 rows of
/// the stride apart -- at every pixel count from 1 to 17 and 23, 31, 32 and
/// 33 at stride 16, and 1, 7, 15, 16, 17, 23, 31, 32 and 33 at strides 24 and
/// 64: each partial tile, one full tile, a full tile and a tail, two full
/// tiles, two and a tail. Every byte the copy writes must match, and no
/// other byte of the destination -- the gaps between rows, 16 bytes before,
/// everything after -- may change.
bool transpose_kernel_ok(uint32_t r) {
  constexpr uint32_t kPlane = 37;
  constexpr uint32_t kMost = 33;     // pixels: two full tiles and a tail
  constexpr uint32_t kStride = 64;   // the widest row checked
  constexpr uint32_t kCanary = 16;
  // The source the copies read: offset 5, 15 planes on, tile 2 (32 pixels
  // in) and the 32 bytes a tile's loads reach, inside kSrc.
  constexpr uint32_t kSrc = 640;
  static_assert(5 + 15 * kPlane + 2 * 16 + 32 <= kSrc && kSrc % 16 == 0,
                "the copies' reads fit the source");
  constexpr uint32_t kDst = kCanary + kMost * kStride + kCanary;
  uint8_t* const mem =
      static_cast<uint8_t*>(heap_caps_malloc(kSrc + kDst + 15, MALLOC_CAP_8BIT));
  if (mem == nullptr) {
    return false;
  }
  int8_t* const src =
      reinterpret_cast<int8_t*>(((uintptr_t)mem + 15) & ~(uintptr_t)15);
  int8_t* const head = src + kSrc;       // the destination's first byte
  int8_t* const dst = head + kCanary;    // where the copy starts
  for (uint32_t i = 0; i < kSrc; ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    src[i] = (int8_t)(r >> 24);
  }
  static const uint8_t kAll[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                 12, 13, 14, 15, 16, 17, 23, 31, 32, 33};
  static const uint8_t kSome[] = {1, 7, 15, 16, 17, 23, 31, 32, 33};
  static const uint8_t kStrides[] = {16, 24, kStride};
  bool ok = true;
  for (const uint8_t row : kStrides) {
    const uint32_t stride = row;
    const uint8_t* const counts = stride == 16 ? kAll : kSome;
    const uint32_t n_counts = stride == 16 ? sizeof(kAll) : sizeof(kSome);
    for (uint32_t off = 0; off <= 5 && ok; off += 5) {
      for (uint32_t c = 0; c < n_counts && ok; ++c) {
        const uint32_t count = counts[c];
        memset(head, 0x5A, kDst);
        transpose16_s3(src + off, (int)kPlane, dst, (int)stride, (int)count);
        for (uint32_t b = 0; b < kCanary; ++b) {
          ok = ok && head[b] == 0x5A;
        }
        for (uint32_t p = 0; p < count; ++p) {
          for (uint32_t j = 0; j < stride; ++j) {
            const int8_t want =
                j < 16 ? src[off + j * kPlane + p] : (int8_t)0x5A;
            ok = ok && dst[p * stride + j] == want;
          }
        }
        for (uint32_t b = kCanary + count * stride; b < kDst; ++b) {
          ok = ok && head[b] == 0x5A;
        }
      }
    }
  }
  heap_caps_free(mem);
  return ok;
}

/// The plane kernels (dw5_plane_s3, dw3_plane_s3, their stride-1 entries
/// dw5_plane1_s3, dw3_plane1_s3, and dw5_pairs_s3), read by unpack16,
/// against the scalar sums:
/// window rows 37 bytes apart (every row at another alignment) of pseudo-
/// random codes, -128 every seventh, and weights the same way, -128 every
/// fifth; two runs of 16 pixels; B taps right after the A taps (a column
/// stride of 1: the stride-1 entries) and 13 bytes further on (a stride of
/// 2's odd columns: the general ones); the pairs kernel at the runtime's
/// plane stride, one to four runs and every row dilation it admits, its
/// images canaried (below); then the lanes' extremes, every code and weight
/// -128 (+409,600 for 25 taps) and -128 by 127 (-406,400). Every lane of
/// every run must match.
bool plane_kernel_ok(uint32_t r) {
  constexpr uint32_t kPitch = 37, kFirst = 3;
  constexpr uint32_t kCodes = 256;   // the window rows and their reads
  int8_t* const x = reinterpret_cast<int8_t*>(g_kernel_check);
  int8_t* const w = x + kCodes;               // 25 weights
  uint8_t* const img = g_kernel_check + 320;  // two 64-byte images
  // The furthest read: the last row's B taps at boff 16 in run 2, plus the
  // second aligned block.
  static_assert(kFirst + 4 * kPitch + 16 + (16 + 1) + 16 + 16 <= kCodes &&
                    kCodes + 32 <= 320 && 320 + 128 <= sizeof(g_kernel_check),
                "the check's reads and writes fit its buffer");
  // The pairs kernel's own buffer (pairs_kernel_ok): its windows at the
  // runtime's plane stride and canaried images, on the heap (freed after;
  // without it the plane kernels stay off).
  constexpr uint32_t kWindows = 5 * kPairsPlane;            // 760
  constexpr uint32_t kImagesAt = (kWindows + 15) / 16 * 16 + 16;
  constexpr uint32_t kStage = kImagesAt + 4 * 64 + 16;
  uint8_t* const mem =
      static_cast<uint8_t*>(heap_caps_malloc(kStage + 15, MALLOC_CAP_8BIT));
  if (mem == nullptr) {
    return false;
  }
  uint8_t* const stage =
      reinterpret_cast<uint8_t*>(((uintptr_t)mem + 15) & ~(uintptr_t)15);
  int8_t* const z = reinterpret_cast<int8_t*>(stage);
  bool ok = true;
  for (uint32_t pass = 0; pass < 3 && ok; ++pass) {
    for (uint32_t i = 0; i < kWindows; ++i) {
      r ^= r << 13;
      r ^= r >> 17;
      r ^= r << 5;
      z[i] = (int8_t)(pass == 0 ? (i % 7 == 0 ? -128 : (int32_t)(r >> 24) - 128)
                                : -128);
    }
    for (uint32_t i = 0; i < kCodes + 32; ++i) {
      r ^= r << 13;
      r ^= r >> 17;
      r ^= r << 5;
      const int32_t v = (int32_t)(r >> 24) - 128;
      if (i < kCodes) {
        x[i] = (int8_t)(pass == 0 ? (i % 7 == 0 ? -128 : v) : -128);
      } else {
        const uint32_t t = i - kCodes;
        w[t] = (int8_t)(pass == 0 ? (t % 5 == 0 ? -128 : v)
                                  : (pass == 1 ? -128 : 127));
      }
    }
    for (uint32_t kh = 3; kh <= 5 && ok; kh += 2) {
      const uint32_t ka = (kh + 1) / 2, kb = kh / 2;
      const int8_t* rows[5];
      for (uint32_t i = 0; i < kh; ++i) {
        rows[i] = x + kFirst + i * kPitch;
      }
      const uint32_t boffs[2] = {ka, ka + 13};
      for (const uint32_t boff : boffs) {
        memset(img, 0x5A, 128);
        // B right after A: the stride-1 entry; further on: the general one.
        if (kh == 5) {
          (boff == ka ? dw5_plane1_s3 : dw5_plane_s3)(rows, (int)boff, w, img, 2);
        } else {
          (boff == ka ? dw3_plane1_s3 : dw3_plane_s3)(rows, (int)boff, w, img, 2);
        }
        for (uint32_t u = 0; u < 2; ++u) {
          int32_t lanes[16];
          unpack16(img + 64u * u, lanes);
          for (uint32_t l = 0; l < 16; ++l) {
            int32_t want = 0;
            for (uint32_t i = 0; i < kh; ++i) {
              const int8_t* a = rows[i] + 16u * u + l;
              const int8_t* wr = w + i * kh;
              for (uint32_t j = 0; j < ka; ++j) {
                want += (int32_t)a[j] * (int32_t)wr[j];
              }
              for (uint32_t j = 0; j < kb; ++j) {
                want += (int32_t)a[boff + j] * (int32_t)wr[ka + j];
              }
            }
            ok = ok && lanes[l] == want;
          }
        }
      }
    }
    // dw5_pairs_s3 as depthwise_pairs calls it: planes kPairsPlane apart,
    // window row i at 8 * (i * dil + 2 * r0), one to four runs, at every row
    // dilation pairs_geo admits (1 to 4: T = 2 * runs + 4 * dil <=
    // kPairsRows) and a later call's offset (r0 = 3); each case's rows end
    // inside the windows. Every lane of every run must match, and no byte
    // around or after the images the call writes may change.
    static const uint8_t kCases[][3] = {  // dil, runs, r0
        {1, 1, 0}, {1, 2, 0}, {1, 3, 0}, {1, 4, 0}, {1, 4, 3},
        {2, 1, 0}, {2, 4, 0}, {3, 3, 0}, {4, 1, 0}};
    for (const auto& cs : kCases) {
      const uint32_t dil = cs[0], runs = cs[1], r0 = cs[2];
      if (!ok) {
        break;
      }
      const int8_t* rows[5];
      for (uint32_t i = 0; i < 5; ++i) {
        rows[i] = z + 8u * (i * dil + 2 * r0);
      }
      memset(stage + kImagesAt - 16, 0x5A, 16 + 4 * 64 + 16);
      dw5_pairs_s3(rows, (int)kPairsPlane, w, stage + kImagesAt, (int)runs);
      for (uint32_t b = 0; b < 16; ++b) {
        ok = ok && stage[kImagesAt - 16 + b] == 0x5A;
      }
      for (uint32_t b = 64 * runs; b < 4 * 64 + 16; ++b) {
        ok = ok && stage[kImagesAt + b] == 0x5A;
      }
      for (uint32_t u = 0; u < runs; ++u) {
        int32_t lanes[16];
        unpack16(stage + kImagesAt + 64u * u, lanes);
        for (uint32_t l = 0; l < 16; ++l) {
          int32_t want = 0;
          for (uint32_t i = 0; i < 5; ++i) {
            for (uint32_t j = 0; j < 5; ++j) {
              want += (int32_t)rows[i][kPairsPlane * j + 16u * u + l] *
                      (int32_t)w[5 * i + j];
            }
          }
          ok = ok && lanes[l] == want;
        }
      }
    }
  }
  heap_caps_free(mem);
  return ok;
}
static_assert(8 * (4 * 1 + 2 * 3) + 4 * kPairsPlane + 16 * 4 <= 5 * kPairsPlane &&
                  8 * (4 * 4) + 4 * kPairsPlane + 16 * 1 <= 5 * kPairsPlane &&
                  8 * (4 * 3) + 4 * kPairsPlane + 16 * 3 <= 5 * kPairsPlane,
              "the pairs check's reads stay inside its windows");

}  // namespace

bool pie_dw_init(void) {
  int state = g_pie_dw_state.load(std::memory_order_acquire);
  if (state >= 2) {
    return state == 2;
  }
  int expected = 0;
  if (!g_pie_dw_state.compare_exchange_strong(expected, 1)) {
    return false;  // another caller is measuring; the scalar path meanwhile
  }
  // The dense columns' fused kernel (rows_codes), on its own flag: until it
  // passes, and if it fails, they take dot_i8_rows_s3 and finish.
  const bool rows_ok = rows_kernel_ok(0x2545F491u);
  g_rows_codes_ok.store(rows_ok ? 1 : 0, std::memory_order_relaxed);
  // The long rows' fused kernel (rows_codes past kRowsCodesChunks), likewise.
  const bool long_ok = rows_long_kernel_ok(0x1B873593u);
  g_rows_long_ok.store(long_ok ? 1 : 0, std::memory_order_relaxed);
  printf("esp32: PIE long-row dot selftest: %s\n",
         long_ok ? "PASS" : "FAIL (dots then epilogue)");
  // The strips' transposes (strip16_codes), likewise on their own flag.
  const bool transpose_ok = transpose_kernel_ok(0x6C8E9CF5u);
  g_transpose_ok.store(transpose_ok ? 1 : 0, std::memory_order_relaxed);
  // The plane kernels (depthwise_plane), likewise.
  const bool plane_ok = plane_kernel_ok(0x1B873593u);
  g_plane_ok.store(plane_ok ? 1 : 0, std::memory_order_relaxed);
  printf("esp32: PIE conv kernels selftest: fused dot and epilogue %s, "
         "transpose %s, depthwise planes %s\n",
         rows_ok ? "PASS" : "FAIL (dots then epilogue)",
         transpose_ok ? "PASS" : "FAIL (element copies)",
         plane_ok ? "PASS" : "FAIL (scalar loop)");
  // A 3 x 3 window of one pixel a step (pitch 48), then four pixels.
  alignas(16) static int8_t x[6 * 3 * 16];
  alignas(16) static int8_t w[9 * 16];
  alignas(16) static uint8_t img[4 * 64];
  int32_t lanes[16];
  uint8_t lane_of[16];
  bool ok = true;
  // Which lane holds channel c: all-ones codes, one weight 1 at channel c.
  memset(x, 1, sizeof(x));
  for (uint32_t c = 0; c < 16 && ok; ++c) {
    memset(w, 0, sizeof(w));
    w[4 * 16 + c] = 1;  // the centre tap
    dw33_qacc16_s3(x, 48, 16, w, img, 1);
    qacc_lanes(img, lanes);
    int found = -1;
    for (uint32_t l = 0; l < 16; ++l) {
      if (lanes[l] == 1 && found < 0) {
        found = (int)l;
      } else if (lanes[l] != 0) {
        ok = false;
      }
    }
    ok = ok && found >= 0;
    lane_of[c] = (uint8_t)(found < 0 ? 0 : found);
    for (uint32_t d = 0; d < c && ok; ++d) {
      ok = lane_of[d] != lane_of[c];
    }
  }
  // Against the scalar sums on codes and weights across the int8 range, the
  // extremes included, for four pixels a stride of 16 bytes apart (window
  // rows 48 bytes apart in a 6 x 3 x 16 block: rows overlap, as in a plane).
  uint32_t r = 0x9E3779B9u;
  for (uint32_t i = 0; i < sizeof(x); ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    x[i] = (int8_t)(i % 7 == 0 ? -128 : (int32_t)(r & 0xFF) - 128);
  }
  for (uint32_t i = 0; i < sizeof(w); ++i) {
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    w[i] = (int8_t)(i % 5 == 0 ? -128 : (int32_t)(r & 0xFF) - 128);
  }
  if (ok) {
    dw33_qacc16_s3(x, 48, 16, w, img, 4);
    for (uint32_t u = 0; u < 4 && ok; ++u) {
      qacc_lanes(img + u * 64, lanes);
      for (uint32_t c = 0; c < 16; ++c) {
        int32_t want = 0;
        for (uint32_t i = 0; i < 3; ++i) {
          for (uint32_t j = 0; j < 3; ++j) {
            want += (int32_t)x[u * 16 + i * 48 + j * 16 + c] *
                    (int32_t)w[(i * 3 + j) * 16 + c];
          }
        }
        ok = ok && lanes[lane_of[c]] == want;
      }
    }
  }
  // The epilogue's kernel, which reads the lanes where dw33_qacc16_s3 left
  // them (lane_codes).
  ok = ok && lane_kernel_ok(r);
  if (ok) {
    memcpy(g_pie_lane, lane_of, sizeof(lane_of));
  }
  g_pie_dw_state.store(ok ? 2 : 3, std::memory_order_release);
  printf("esp32: PIE depthwise 16-lane selftest: %s (QACC lanes", ok ? "PASS" : "FAIL, scalar depthwise");
  for (uint32_t c = 0; c < 16; ++c) {
    printf(" %u", (unsigned)lane_of[c]);
  }
  printf(")\n");
  return ok;
}

#if defined(ESP32_HOST_BUILD)
QuantFastCalls quant_fast_calls() {
  QuantFastCalls c;
  c.span = g_fast_spans.load(std::memory_order_relaxed);
  c.pointwise = g_fast_pointwise.load(std::memory_order_relaxed);
  c.dense = g_fast_dense.load(std::memory_order_relaxed);
  return c;
}
#endif

uint64_t part_scratch_need(const Op& op, bool fp32_in, uint32_t rows) {
  if (op.depthwise) {
    return scratch_need(op, fp32_in, rows);
  }
  return (uint64_t)rows * op.k_pad;
}

int8_t* plane_of(const Op& op, const Act& act, const Scratch& scratch) {
  if (op.depthwise || act.x8 != nullptr || !dense_plane(op)) {
    return nullptr;
  }
  // The whole input's codes once, after the row block: a 3x3 reads each
  // element up to nine times.
  return scratch.base + align16((uint64_t)scratch.rows * op.k_pad);
}

void prepare_part(const Act& act, int8_t* plane, Part part) {
  quantize_span(act, act.x + part.lo, part.hi - part.lo, plane + part.lo);
}

const int8_t* prepare(const Op& op, const Act& act, const Scratch& scratch) {
  int8_t* plane = plane_of(op, act, scratch);
  if (plane != nullptr) {
    prepare_part(act, plane, Part{0, plane_elements(op)});
  }
  return plane;
}

void run(const Op& op, const Act& act, const Columns& cols, const Out& out,
         const Scratch& scratch, Counts* counts) {
  Scratch sc = scratch;
  sc.plane = prepare(op, act, scratch);
  run_part(op, act, cols, out, sc, Part{0, units(op)}, counts);
}

void run_part(const Op& op, const Act& act, const Columns& cols,
              const Out& out, const Scratch& scratch, Part part,
              Counts* counts) {
  Epi e;
  e.sx = act.sx;
  e.lo = op.clamp ? op.clamp_lo : -INFINITY;
  e.hi = op.clamp ? op.clamp_hi : INFINITY;
  e.out_inv = out.out_inv;
  e.out_scale = out.out_scale;
  e.zo = out.zo;
  e.qmin = out.qmin;
  e.qmax = out.qmax;
  fixed_epi(op, act, out, &e);
  if (out.y8 != nullptr) {
    run_mode<kOutCodes>(op, act, cols, out, e, scratch, part, counts);
  } else if (out.requant) {
    run_mode<kOutRequant>(op, act, cols, out, e, scratch, part, counts);
  } else {
    run_mode<kOutFloat>(op, act, cols, out, e, scratch, part, counts);
  }
}

EpilogueBench bench_epilogue(uint32_t (*cycles)(void), uint32_t reps) {
  // DS-CNN's pointwise shape: K = 64, codes after a ReLU (zx = -128), a
  // per-channel weight scale and a bias, codes out with a ReLU clamp.
  Op op;
  memset(&op, 0, sizeof(op));
  op.k = 64;
  Act act;
  memset(&act, 0, sizeof(act));
  act.zx = -128;
  act.sx = 0.0234f;
  Out out;
  memset(&out, 0, sizeof(out));
  int8_t sink = 0;
  out.y8 = &sink;
  Epi e;
  e.sx = act.sx;
  e.lo = 0.0f;
  e.hi = INFINITY;
  e.out_scale = 0.0517f;
  e.out_inv = 1.0f / e.out_scale;
  e.zo = -128;
  e.qmin = -128;
  e.qmax = 127;
  fixed_epi(op, act, out, &e);
  const float s = 0.00412f, b = -0.731f;
  const int32_t sum = 517;
  const Channel ch = channel_of(s, b, act.zx, (uint32_t)sum, e);
  // Accumulators across the codes: acc from -3,000 to 9,000, raw = acc + zx S'.
  int32_t raw[kRun];
  const uint32_t zxs = ch.zxs;
  uint32_t x = 12345u;
  for (uint32_t i = 0; i < kRun; ++i) {
    x = x * 1103515245u + 12345u;
    raw[i] = (int32_t)((uint32_t)((int32_t)(x >> 8) % 12000 - 3000) + zxs);
  }
  int8_t a[kRun], f[kRun];
  EpilogueBench r = {0, 0, 0, 0};
  uint32_t best[2] = {UINT32_MAX, UINT32_MAX};
  for (uint32_t trial = 0; trial < 3; ++trial) {
    uint32_t c0 = cycles();
    for (uint32_t i = 0; i < reps; ++i) {
      finish_float<kOutCodes>(raw, kRun, zxs, ch.s, ch.b, e, nullptr, a, 0);
    }
    uint32_t c1 = cycles();
    for (uint32_t i = 0; i < reps; ++i) {
      r.fallbacks = finish_fixed<kOutCodes>(raw, kRun, ch, e, nullptr, f, 0);
    }
    uint32_t c2 = cycles();
    best[0] = c1 - c0 < best[0] ? c1 - c0 : best[0];
    best[1] = c2 - c1 < best[1] ? c2 - c1 : best[1];
  }
  const uint64_t outputs = (uint64_t)reps * kRun;
  r.cycles_float = (uint32_t)((uint64_t)best[0] * 1000 / outputs);
  r.cycles_fixed = (uint32_t)((uint64_t)best[1] * 1000 / outputs);
  for (uint32_t i = 0; i < kRun; ++i) {
    r.mismatches += a[i] != f[i];
  }
  return r;
}

}  // namespace esp32_conv
