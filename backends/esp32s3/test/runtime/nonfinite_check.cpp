/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The non-finite quantize rule (runtime/quant_rules.h; ExecuTorch 1.6,
// #22811) through each quantiser's own entry point: the affine records'
// static int8 and int16 activations, output requantisation and Stage B
// codes (code_of), the backend's rows under PT2E's rounding -- the affine
// records' copy (rne_i), and esp32_backend.cpp's own quantize_row_i8 and
// quantize_conv_row through its C hooks, so round_half_even_i itself --
// the dynamic chooser, and the stock K/V cache's writer. Inputs: NaN of either sign, the infinities,
// +-FLT_MAX, values whose product lands far past every range, zeros and a
// few ordinary values. The table ops' quantize steps have their own sweeps
// (quant_lut_check.cpp, quant_lut16_check.cpp), specials included.
//
// The reference is a per-tensor quantize at the quantiser's scale, zero
// point and range: on the host ExecuTorch's own kernel (asym_conformance.cpp
// passes it in), which the transcription below is held to as well; on the
// device the transcription. So on the chip this shows that floor.s, trunc.s
// and the soft float64 give the portable kernel's codes. Test code: main.cpp's
// parity run and the host's --selftest call it; the story firmware never
// does, so it links none of this file. -ffp-contract=off, as the kernels it
// drives.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "esp32_asym.h"
#include "kv_affine.h"
#include "quant_rules.h"

// esp32_backend.cpp's C hooks into its own quantisers (the end of that file).
extern "C" float esp32_backend_check_quantize_row_i8(const float* x, uint32_t k,
                                                     uint32_t k_pad, int8_t* dst,
                                                     float fixed_sx,
                                                     int ties_to_even);
extern "C" float esp32_backend_check_quantize_conv_row(const float* x,
                                                       uint32_t n, uint32_t pad,
                                                       uint32_t k_pad,
                                                       int8_t* dst,
                                                       float fixed_sx,
                                                       int ties_to_even);

namespace espressif {
namespace nonfinite {

int32_t transcribed_code(float x, float scale, int32_t zero_point,
                         int32_t quant_min, int32_t quant_max) {
  // quantize_val: inv_scale = 1.0f / (float)scale, then
  // quantize_val_with_inv_scale.
  const float inv = 1.0f / scale;
  const float v = inv * x;
  const double q = std::isnan(v) ? (double)zero_point
                                 : (double)zero_point + (double)nearbyintf(v);
  const double c = q < (double)quant_min
                       ? (double)quant_min
                       : (q > (double)quant_max ? (double)quant_max : q);
  return (int32_t)c;
}

namespace {

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

const float kInputs[] = {kNan,   -kNan,  kInf,  -kInf, FLT_MAX, -FLT_MAX,
                         1e20f,  -1e20f, 3e9f,  -3e9f, 0.0f,    -0.0f,
                         1.0f,   -1.0f,  0.25f, 0.75f, 1.25f,   -2.5f,
                         1e-45f};
constexpr uint32_t kN = sizeof(kInputs) / sizeof(kInputs[0]);
constexpr uint32_t kPad = 32;  // k_pad: kN rounded up to 16
static_assert(kN <= kPad, "the inputs fit one padded row");

uint32_t bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}

struct Tally {
  Reference reference;
  uint32_t cases = 0, mismatches = 0, transcription = 0, printed = 0;

  // The reference's code, and on the host the transcription's against it.
  int32_t want(float x, float scale, int32_t zp, int32_t qmin, int32_t qmax) {
    const int32_t t = transcribed_code(x, scale, zp, qmin, qmax);
    if (reference == nullptr) {
      return t;
    }
    const int32_t r = reference(x, scale, zp, qmin, qmax);
    if (t != r) {
      ++transcription;
      report("transcription", x, scale, zp, t, r);
    }
    return r;
  }

  void check(const char* what, float x, float scale, int32_t zp, int32_t got,
             int32_t want_code) {
    ++cases;
    if (got != want_code) {
      ++mismatches;
      report(what, x, scale, zp, got, want_code);
    }
  }

  void report(const char* what, float x, float scale, int32_t zp, long got,
              long want_code) {
    if (printed++ < 8) {
      printf("  MISMATCH nonfinite %s: x %08lx scale %08lx zp %ld: %ld, want "
             "%ld\n", what, (unsigned long)bits(x), (unsigned long)bits(scale),
             (long)zp, got, want_code);
    }
  }
};

struct Grid {
  float scale;
  int32_t zp, qmin, qmax;
};

esp32_asym::ActQuant static_record(const Grid& g, uint32_t code_bits) {
  esp32_asym::ActQuant a{};
  a.in_dtype = esp32_asym::kDtypeFp32;
  a.code_bits = code_bits;
  a.granularity = esp32_asym::kGranularityStatic;
  a.chooser = esp32_asym::kChooserNone;
  a.rounding = esp32_asym::kRoundRneBeforeZp;
  a.qmin = g.qmin;
  a.qmax = g.qmax;
  a.scale = g.scale;
  a.zero_point = g.zp;
  return a;
}

esp32_asym::AffineWeights one_group(uint32_t k) {
  esp32_asym::AffineWeights w{};
  w.k = k;
  w.k_pad = kPad;
  w.gsize = kPad;
  w.ngw = 1;
  return w;
}

// The affine records' static quantisers and output records: an int8 grid at
// each end of its zero points, upstream's clamped-zero-point cases (the zero
// point past the range: NaN takes the nearer rail), and int16 grids.
void check_static(Tally& tally) {
  static const Grid kGrids8[] = {{0.5f, 5, -128, 127},
                                 {0.5f, 20, -10, 10},
                                 {0.5f, -20, -10, 10},
                                 {0.007f, -128, -128, 127},
                                 {3.7f, 127, -128, 127}};
  static const Grid kGrids16[] = {{0.0009765625f, 0, -32767, 32767},
                                  {0.0071f, -5, -32768, 32767}};
  const esp32_asym::AffineWeights w = one_group(kN);
  int8_t codes[kPad];
  int16_t codes16[kPad];
  int32_t xsum[1];
  for (const Grid& g : kGrids8) {
    const esp32_asym::ActQuant a = static_record(g, 8);
    const esp32_asym::ActParams p = esp32_asym::choose(a, kInputs, kN);
    esp32_asym::quantize_row(a, p, kInputs, kN, w, codes, xsum, nullptr);
    const esp32_asym::OutQuant o{esp32_asym::kDtypeFp32,
                                 esp32_asym::kRoundRneBeforeZp, g.scale, g.zp,
                                 g.qmin, g.qmax};
    esp32_asym::OpQuant q{};
    q.out = &o;
    q.out_inv = 1.0f / g.scale;
    for (uint32_t t = 0; t < kN; ++t) {
      const float x = kInputs[t];
      const int32_t want = tally.want(x, g.scale, g.zp, g.qmin, g.qmax);
      tally.check("static int8", x, g.scale, g.zp, codes[t], want);
      tally.check("requantize_code", x, g.scale, g.zp,
                  esp32_asym::requantize_code(q, x), want);
      // requantize's float is the code's dequantised value: compare bits.
      const float y = esp32_asym::requantize(q, x);
      const float back = (float)(want - g.zp) * g.scale;
      tally.check("requantize", x, g.scale, g.zp, (int32_t)bits(y),
                  (int32_t)bits(back));
    }
  }
  for (const Grid& g : kGrids16) {
    const esp32_asym::ActQuant a = static_record(g, 16);
    esp32_asym::quantize_row16(a, kInputs, kN, kPad, codes16);
    for (uint32_t t = 0; t < kN; ++t) {
      const float x = kInputs[t];
      tally.check("static int16", x, g.scale, g.zp, codes16[t],
                  tally.want(x, g.scale, g.zp, g.qmin, g.qmax));
    }
  }
}

// The backend's rows under PT2E's rounding, a folded per-tensor activation
// (esp32_backend.cpp's kFlagActPerTensor): zero point 0, +-127, at the scale
// fixed_activation_scale computes over the whole tensor -- absmax / 127, a
// NaN skipped, floored at FLT_EPSILON -- so every finite fl(x * inv) is
// within 127 (round_half_even_i's contract), and an infinity makes the scale
// infinite: every code is then 0, as the portable kernel gives at that scale.
void check_a0(Tally& tally) {
  static const float kOrdinary[] = {kNan, 0.5f, -1.5f, 2.5f, 3.5f, -kNan,
                                    127.0f, -0.0f, 1e-45f, 126.5f};
  static float no_inf[kN];
  uint32_t m = 0;
  for (float x : kInputs) {
    if (!std::isinf(x)) {
      no_inf[m++] = x;
    }
  }
  struct Set {
    const float* x;
    uint32_t n;
  };
  const Set kSets[] = {{kInputs, kN}, {no_inf, m}, {kOrdinary, 10}};
  int8_t codes[kPad];
  int32_t xsum[1];
  for (const Set& set : kSets) {
    float amax = 0.0f;
    for (uint32_t t = 0; t < set.n; ++t) {
      const float a = std::fabs(set.x[t]);
      amax = a > amax ? a : amax;
    }
    const float scale = amax / 127.0f;
    const float s = scale > 1.1920928955078125e-07f ? scale
                                                    : 1.1920928955078125e-07f;
    esp32_asym::quantize_row_a0(set.x, set.n, s, true, one_group(set.n), codes,
                                xsum, nullptr);
    for (uint32_t t = 0; t < set.n; ++t) {
      const float x = set.x[t];
      tally.check("a0 ties-to-even", x, s, 0, codes[t],
                  tally.want(x, s, 0, -127, 127));
    }
  }
}

// The same sets through esp32_backend.cpp's own quantisers: quantize_row_i8
// and quantize_conv_row (two padding taps each side, which read code 0) at
// the per-tensor scale, rounding ties to even -- round_half_even_i, which
// takes a NaN to 0 (quant_scaled) before floor_i converts it. Without that,
// floor_i's conversion of NaN is the chip's floor.s (a saturated code) or
// the host's (x86: INT_MIN, so -127; arm64 happens to give 0).
void check_backend_rows(Tally& tally) {
  static const float kOrdinary[] = {kNan, 0.5f, -1.5f, 2.5f, 3.5f, -kNan,
                                    127.0f, -0.0f, 1e-45f, 126.5f};
  static float no_inf[kN];
  uint32_t m = 0;
  for (float x : kInputs) {
    if (!std::isinf(x)) {
      no_inf[m++] = x;
    }
  }
  struct Set {
    const float* x;
    uint32_t n;
  };
  const Set kSets[] = {{kInputs, kN}, {no_inf, m}, {kOrdinary, 10}};
  constexpr uint32_t kConvPad = 2;
  static_assert(kN + 2 * kConvPad <= kPad, "the padded patch fits one row");
  int8_t codes[kPad];
  for (const Set& set : kSets) {
    float amax = 0.0f;
    for (uint32_t t = 0; t < set.n; ++t) {
      const float a = std::fabs(set.x[t]);
      amax = a > amax ? a : amax;
    }
    const float scale = amax / 127.0f;
    const float s = scale > 1.1920928955078125e-07f ? scale
                                                    : 1.1920928955078125e-07f;
    memset(codes, 0x55, sizeof codes);
    esp32_backend_check_quantize_row_i8(set.x, set.n, kPad, codes, s, 1);
    for (uint32_t t = 0; t < kPad; ++t) {
      const float x = t < set.n ? set.x[t] : 0.0f;
      tally.check("backend quantize_row_i8", x, s, 0, codes[t],
                  t < set.n ? tally.want(x, s, 0, -127, 127) : 0);
    }
    memset(codes, 0x55, sizeof codes);
    esp32_backend_check_quantize_conv_row(set.x, set.n, kConvPad, kPad, codes,
                                          s, 1);
    for (uint32_t t = 0; t < kPad; ++t) {
      const bool real = t >= kConvPad && t < kConvPad + set.n;
      const float x = real ? set.x[t - kConvPad] : 0.0f;
      tally.check("backend quantize_conv_row", x, s, 0, codes[t],
                  real ? tally.want(x, s, 0, -127, 127) : 0);
    }
  }
}

// The dynamic chooser: a NaN is skipped (the row's scale and zero point are
// those of its other values) and takes the zero point; an infinity makes the
// scale infinite and, affine, the zero point qmin (fl(lo / s) is 0, or NaN
// for lo = -inf, and either rounds to 0), so every code is the zero point.
void check_chooser(Tally& tally) {
  static const float kWithNan[] = {0.5f, kNan, -1.75f, 3.25f, -kNan, 2.0f};
  static const float kFinite[] = {0.5f, -1.75f, 3.25f, 2.0f};
  static const float kPosInf[] = {0.5f, kInf, -1.75f};
  static const float kNegInf[] = {0.5f, -kInf, 1.75f};
  esp32_asym::ActQuant affine{};
  affine.in_dtype = esp32_asym::kDtypeFp32;
  affine.code_bits = 8;
  affine.granularity = esp32_asym::kGranularityRow;
  affine.chooser = esp32_asym::kChooserAffineMinMax;
  affine.rounding = esp32_asym::kRoundRneBeforeZp;
  affine.qmin = -128;
  affine.qmax = 127;
  affine.eps = 0.000244140625f;  // 2^-12, XNNPACK's
  esp32_asym::ActQuant sym = affine;
  sym.chooser = esp32_asym::kChooserSymDiv;
  sym.qmin = -127;
  sym.sym_div = 127.0f;
  sym.eps = 1.1920928955078125e-07f;  // 2^-23
  int8_t codes[kPad];
  int32_t xsum[1];
  struct Row {
    const float* x;
    uint32_t n;
  };
  const Row kRows[] = {{kWithNan, 6}, {kPosInf, 3}, {kNegInf, 3}};
  for (const esp32_asym::ActQuant* a : {&affine, &sym}) {
    const esp32_asym::ActParams finite = esp32_asym::choose(*a, kFinite, 4);
    for (const Row& row : kRows) {
      const esp32_asym::ActParams p = esp32_asym::choose(*a, row.x, row.n);
      const bool infinite = row.x != kWithNan;
      const bool params_ok = infinite
          ? (std::isinf(p.sx) && p.sx > 0.0f &&
             p.zx == (a == &affine ? a->qmin : 0))
          : (bits(p.sx) == bits(finite.sx) && p.zx == finite.zx);
      tally.check("chooser params", row.x[1], p.sx, p.zx, params_ok ? 1 : 0, 1);
      esp32_asym::quantize_row(*a, p, row.x, row.n, one_group(row.n), codes,
                               xsum, nullptr);
      for (uint32_t t = 0; t < row.n; ++t) {
        const float x = row.x[t];
        tally.check("chooser codes", x, p.sx, p.zx, codes[t],
                    tally.want(x, p.sx, p.zx, a->qmin, a->qmax));
      }
    }
  }
}

// torch's quantize_per_token in float64, the zero point added before the
// rounding, with ExecuTorch 1.6's rule for a NaN product: the K/V writer's
// arithmetic for a finite element (kv_affine.h).
int32_t kv_code(float x, float scale, int32_t zp) {
  const double p = (double)x * (1.0 / (double)scale);
  if (std::isnan(p)) {
    return zp;
  }
  const double v = nearbyint(p + (double)zp);
  return (int32_t)(v < -128.0 ? -128.0 : (v > 127.0 ? 127.0 : v));
}

// The stock K/V cache's writer: each element against kv_code at the row's own
// scale and zero point, and a NaN or infinite one against the per-tensor
// reference too. A row holding +inf has scale inf and zero point 0 (its
// zero point's arithmetic is NaN), one whose only infinity is -inf zero
// point 127, and so does a row of NaNs (an all-zero row's): every code is
// the zero point. +-FLT_MAX overflow the span to inf, so that row's scale is
// infinite with every value finite.
void check_kv(Tally& tally) {
  static const float kRows[][6] = {{0.5f, kNan, -1.75f, 3.25f, 2.0f, -0.0f},
                                   {0.5f, kInf, -1.75f, 3.0f, 0.0f, 1.0f},
                                   {0.5f, -kInf, 1.75f, 3.0f, 0.0f, 1.0f},
                                   {kNan, -kNan, kNan, kNan, kNan, kNan},
                                   {FLT_MAX, -FLT_MAX, 1.0f, 0.0f, 2.0f, 3.0f},
                                   {kInf, -kInf, kNan, 1.0f, -1.0f, 0.0f}};
  static const int32_t kZeroPoints[] = {0, 0, 127, 127, 127, 0};
  int8_t codes[6];
  float scale;
  int8_t zp;
  for (uint32_t r = 0; r < 6; ++r) {
    kv_affine::quantize_row(kRows[r], 6, codes, &scale, &zp);
    if (r > 0) {
      // Every row but the first has a zero point of its own definition here.
      const bool scale_ok = r == 3 ? scale > 0.0f && !std::isinf(scale)
                                   : std::isinf(scale) && scale > 0.0f;
      tally.check("kv params", kRows[r][0], scale, zp,
                  scale_ok && zp == kZeroPoints[r] ? 1 : 0, 1);
    }
    for (uint32_t t = 0; t < 6; ++t) {
      const float x = kRows[r][t];
      tally.check("kv codes", x, scale, zp, codes[t], kv_code(x, scale, zp));
      if (!std::isfinite(x)) {
        tally.check("kv portable", x, scale, zp, codes[t],
                    tally.want(x, scale, zp, -128, 127));
      }
    }
  }
}

}  // namespace

uint32_t run_check(Reference reference) {
  Tally tally;
  tally.reference = reference;
  check_static(tally);
  check_a0(tally);
  check_backend_rows(tally);
  check_chooser(tally);
  check_kv(tally);
  const uint32_t mismatches = tally.mismatches + tally.transcription;
  if (reference != nullptr) {
    printf("nonfinite check: %lu case(s), %lu mismatch(es) (reference "
           "ExecuTorch's quantize_per_tensor; transcription %lu apart)\n",
           (unsigned long)tally.cases, (unsigned long)mismatches,
           (unsigned long)tally.transcription);
  } else {
    printf("nonfinite check: %lu case(s), %lu mismatch(es) (reference the "
           "transcription of ExecuTorch's quantize_val)\n",
           (unsigned long)tally.cases, (unsigned long)mismatches);
  }
  return mismatches;
}

}  // namespace nonfinite
}  // namespace espressif
