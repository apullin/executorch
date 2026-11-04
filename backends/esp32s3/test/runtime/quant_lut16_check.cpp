/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// espressif::lut16's conformance check (quant_lut16.h): every scheme (and
// flagged scheme word) on all 65,536 int16 codes, the op's quantize and
// dequantize steps, one case per malformed argument through the shim, and
// one per malformed flagged table through validate(), refused by its name,
// against mirrors.py's bits
// (quant_lut16_golden.h, written by quant_lut16_golden.py). Test code:
// main.cpp's parity run and the host's --selftest call it; the story firmware
// never does, so it links none of this file. -ffp-contract=off like the
// kernel.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include "quant_lut16.h"
#include "quant_lut16_golden.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;
using executorch::runtime::TensorShapeDynamism;

namespace espressif {
namespace lut16 {

namespace {

constexpr size_t kChunk = 256;

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

uint32_t fnv1a(uint32_t h, const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    h = (h ^ p[i]) * 16777619u;
  }
  return h;
}

struct Rng {
  uint32_t x;
  uint32_t next() {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
  }
};

// The log2 width of cubic row `row` (quant_lut16_golden.segment_shift).
int32_t row_shift(const golden::TableSpec& s, int32_t row) {
  if ((s.scheme & kBaseMask) == kUniformCubic) {
    return s.shift;
  }
  const NsauLayout l = nsau_layout(s.scheme, s.lo, s.hi, s.shift);
  int32_t n0 = 0;
  if (s.scheme & kOffset) {
    const int32_t a = -s.hi > 1 ? -s.hi : 1;
    int32_t sh = 31 - __builtin_clz((uint32_t)a | 1u) - s.shift;
    sh = sh < 0 ? 0 : sh;
    n0 = (sh << s.shift) + (a >> sh);
  }
  const int32_t k = (s.lo < 0 && row >= l.neg_bias + n0) ? row - l.neg_bias
                                                          : row - l.pos_bias;
  const int32_t F = (k >> s.shift) - 1;
  return F < 0 ? 0 : F;
}

// quant_lut16_golden.make_table, step for step, into `buf`.
void make_table(const golden::TableSpec& s, void* buf) {
  Rng rng{s.seed};
  if (s.scheme == kDirect) {
    int16_t* T = static_cast<int16_t*>(buf);
    for (int32_t i = 0; i < s.count; ++i) {
      T[i] = (int16_t)(uint16_t)(rng.next() >> 16);
    }
    return;
  }
  if ((s.scheme & kBaseMask) == kUniformLinear ||
      (s.scheme & kBaseMask) == kNsauLinear) {
    int32_t* V = static_cast<int32_t*>(buf);
    for (int32_t i = 0; i < s.count; ++i) {
      if (s.amp == -1) {
        V[i] = (i % 2 == 0) ? (1 << 30) : -(1 << 30);
      } else {
        V[i] = (int32_t)(rng.next() % (uint32_t)(2 * s.amp + 1)) - s.amp;
      }
    }
    return;
  }
  float* C = static_cast<float*>(buf);
  for (int32_t seg = 0; seg < s.count; ++seg) {
    float* c = C + 4 * seg;
    c[0] = c[1] = c[2] = c[3] = 0.0f;
    const int32_t kind = seg % 5;
    if (kind == 0) {
      c[0] = (float)((int32_t)(rng.next() % 60001u) - 30000) + 0.5f;
    } else if (kind == 1) {
      c[0] = (float)((int32_t)(rng.next() % 2001u) - 1000);
      const int32_t F = row_shift(s, seg);
      c[1] = F >= 1 ? bits_float((uint32_t)(127 + F - 1) << 23) : 0.0f;
    } else {
      for (int j = 0; j < 4; ++j) {
        const int32_t emin = golden::kCoefExp[j][0];
        const int32_t emax = golden::kCoefExp[j][1];
        const uint32_t r = rng.next();
        const uint32_t sign = r & 1u;
        const int32_t e = emin + (int32_t)((r >> 1) % (uint32_t)(emax - emin + 1));
        const uint32_t mant = rng.next() & 0x7FFFFFu;
        c[j] = bits_float((sign << 31) | ((uint32_t)(e + 127) << 23) | mant);
      }
    }
  }
  if (s.scheme & kTail) {
    // an edge row whose codes past it reach a segment or more is a line,
    // half a code per code (quant_lut16_golden.make_table)
    const int32_t width = 1 << s.shift;
    const int32_t last = s.lo + (s.count - 1) * width;
    const int32_t below = s.lo + 32768 > 0 ? s.lo + 32768 : 0;
    const int32_t above = 32767 - last > width - 1 ? 32767 - last : width - 1;
    const int32_t rows[2] = {0, s.count - 1};
    const int32_t reach[2] = {below, above};
    for (int e = 0; e < 2; ++e) {
      if (reach[e] >= width) {
        float* c = C + 4 * rows[e];
        c[1] = s.shift >= 1 ? (float)(1 << (s.shift - 1)) : 0.5f;
        c[2] = c[3] = 0.0f;
      }
    }
  }
}

size_t element_bytes(int32_t scheme) {
  const int32_t base = scheme & kBaseMask;
  return base == kDirect ? 2 : (base == kUniformLinear ||
                                base == kNsauLinear) ? 4 : 16;
}

struct Tally {
  uint64_t cases = 0;
  int tables = 0, generated = 0, device = 0, reference = 0, stage_a = 0;
  int inputs = 0;
};

// Every table on every int16 code: eval_codes (the device's loops) and
// eval_code (the C reference), each hashed against the mirror's.
void check_tables(Tally& tally, void* buf) {
  for (const golden::TableSpec& s : golden::kTables) {
    ++tally.tables;
    make_table(s, buf);
    const uint32_t table_fnv =
        fnv1a(2166136261u, buf, (size_t)s.count * element_bytes(s.scheme));
    const bool generated = table_fnv == s.table_fnv;
    tally.generated += generated;
    Table t;
    t.scheme = s.scheme;
    t.lo = s.lo;
    t.hi = s.hi;
    t.shift = s.shift;
    t.G = s.G;
    t.out_min = s.out_min;
    t.out_max = s.out_max;
    t.data = buf;
    t.count = s.count;
    const char* reason = validate(t);
    if (!generated || reason != nullptr) {
      printf("  MISMATCH lut16 table '%s': generated fnv %08lx (golden %08lx), "
             "validate %s\n", s.name, (unsigned long)table_fnv,
             (unsigned long)s.table_fnv, reason ? reason : "ok");
      continue;
    }
    uint32_t device = 2166136261u, reference = 2166136261u;
    int16_t in[kChunk], out[kChunk];
    for (int32_t first = -32768; first <= 32767; first += (int32_t)kChunk) {
      for (size_t i = 0; i < kChunk; ++i) {
        in[i] = (int16_t)(first + (int32_t)i);
      }
      eval_codes(t, in, out, kChunk);
      device = fnv1a(device, out, sizeof out);
      for (size_t i = 0; i < kChunk; ++i) {
        out[i] = (int16_t)eval_code(t, in[i]);
      }
      reference = fnv1a(reference, out, sizeof out);
    }
    tally.cases += 2 * 65536;
    tally.device += device == s.codes_fnv;
    tally.reference += reference == s.codes_fnv;
    if (device != s.codes_fnv || reference != s.codes_fnv) {
      printf("  MISMATCH lut16 '%s' codes: device %08lx, reference %08lx, "
             "golden %08lx\n", s.name, (unsigned long)device,
             (unsigned long)reference, (unsigned long)s.codes_fnv);
    }
  }
}

// The op's arithmetic (apply) on each case's inputs, generated as
// quant_lut16_golden.stage_a_inputs generates them.
void check_stage_a(Tally& tally, void* buf) {
  for (const golden::Case& c : golden::kCases) {
    const golden::TableSpec& s = golden::kTables[c.table];
    make_table(s, buf);
    Table t;
    t.scheme = s.scheme;
    t.lo = s.lo;
    t.hi = s.hi;
    t.shift = s.shift;
    t.G = s.G;
    t.out_min = s.out_min;
    t.out_max = s.out_max;
    t.data = buf;
    t.count = s.count;
    const float scale = bits_float(c.scale_bits);
    const float inv = bits_float(c.inv_bits);
    const float out_scale = bits_float(c.out_scale_bits);
    float xs[kChunk], ys[kChunk];
    size_t n = 0;
    uint32_t total = 0;
    uint32_t xh = 2166136261u, yh = 2166136261u;
    auto flush = [&]() {
      apply(t, xs, ys, n, inv, c.zero_point, c.quant_min, c.quant_max,
            out_scale, c.out_zero_point);
      xh = fnv1a(xh, xs, n * sizeof(float));
      yh = fnv1a(yh, ys, n * sizeof(float));
      total += (uint32_t)n;
      n = 0;
    };
    auto push = [&](float x) {
      xs[n++] = x;
      if (n == kChunk) {
        flush();
      }
    };
    for (const auto& window : golden::kWindows) {
      for (int32_t k = window[0]; k <= window[1]; ++k) {
        const float x0 = (float)(((double)k + 0.5) / (double)inv);
        const float d1 = nextafterf(x0, -INFINITY);
        const float u1 = nextafterf(x0, INFINITY);
        push(nextafterf(d1, -INFINITY));
        push(d1);
        push(x0);
        push(u1);
        push(nextafterf(u1, INFINITY));
      }
    }
    for (uint32_t special : golden::kSpecials) {
      push(bits_float(special));
    }
    for (int32_t k = golden::kSweepFirst; k < golden::kSweepEnd;
         k += golden::kSweepStep) {
      push((float)k * scale);
    }
    if (n > 0) {
      flush();
    }
    tally.cases += total;
    const bool inputs = total == c.inputs && xh == c.input_fnv;
    tally.inputs += inputs;
    tally.stage_a += yh == c.output_fnv;
    if (!inputs || yh != c.output_fnv) {
      printf("  MISMATCH lut16 stage A '%s' inv %08lx zp %ld: %lu inputs fnv "
             "%08lx (golden %lu, %08lx), outputs %08lx (golden %08lx)\n",
             s.name, (unsigned long)c.inv_bits, (long)c.zero_point,
             (unsigned long)total, (unsigned long)xh, (unsigned long)c.inputs,
             (unsigned long)c.input_fnv, (unsigned long)yh,
             (unsigned long)c.output_fnv);
    }
  }
}

// The malformed-argument cases, through the shim on hand-built tensors: a
// control (a uniform cubic table of four segments over [-8, 7]) and one edit
// per case, each of which must fail with its error and leave out as it was.
struct Malformed {
  int refused = 0, total = 0;
  bool untouched = true;
};

enum Edit {
  kControl,
  kSlots16,
  kXInt,
  kOutDouble,
  kInvInt,
  kInvZero,
  kInvNan,
  kInvNegative,
  kInvNotF32,
  kOutScaleInf,
  kOutScaleNotF32,
  kZpDouble,
  kQminEqQmax,
  kQminLow,
  kQmaxHigh,
  kZpAbove,
  kScheme5,
  kSchemeNeg,
  kLoAboveHi,
  kLoLow,
  kShift17,
  kG17,
  kGNeg,
  kOutRangeEmpty,
  kOutZpOutside,
  kOutMaxHigh,
  kTableInt,
  kTable1D,
  kTable3Cols,
  kTableRows,
  kCoefNan,
  kCoefBig,
  kNsauLo,
  kNsauM9,
  kLinearUnsafe,
  kDirectFloat,
  kOutShape,
  kXStrided,
  kTableStrided,
  kXChannelsLast,
  kEditCount,
};

const char* const kEditNames[kEditCount] = {
    "control", "sixteen slots", "x int32", "out fp64", "inv an int", "inv 0",
    "inv NaN", "inv negative", "inv not float32", "out scale inf",
    "out scale not float32", "zp a double", "qmin == qmax", "qmin -32769",
    "qmax 32768", "zp above", "scheme 5", "scheme -1", "lo > hi",
    "lo -32769", "shift 17", "G 17", "G -1", "out range empty",
    "out zp outside", "out qmax 32768", "cubic table int32", "cubic table 1-D",
    "cubic table [5, 3]", "cubic table [5, 4]", "coefficient NaN",
    "coefficient 2^20", "NSAU lo -32768", "NSAU m 9", "linear node 2^30 + 1",
    "direct table float", "out other shape", "x strided", "table strided",
    "x channels-last"};

void check_malformed(Malformed& result, bool& control_ok) {
  using Sizes = TensorImpl::SizesType;
  using Order = TensorImpl::DimOrderType;
  using Strides = TensorImpl::StridesType;
  alignas(16) static float x[16], out[16], coef[64];
  alignas(16) static int32_t nodes[16];
  for (int i = 0; i < 16; ++i) {
    x[i] = 0.75f * (float)(i - 8);
    nodes[i] = 100 * i;
  }
  for (int i = 0; i < 64; ++i) {
    coef[i] = (float)((i * 37) % 23) - 11.0f;
  }
  for (int e = 0; e < kEditCount; ++e) {
    // Each case is a program of its own, and some edit the control's table
    // in place: nothing checked() remembers carries over.
    forget_tables();
    // The control: x [2, 3], table float [4, 4], out [2, 3].
    Sizes xs[4] = {2, 3}, ts[4] = {4, 4}, os[4] = {2, 3};
    Order xo[4] = {0, 1}, to[4] = {0, 1}, oo[4] = {0, 1};
    Strides xst[4] = {3, 1}, tst[4] = {4, 1}, ost[4] = {3, 1};
    int8_t xdim = 2, tdim = 2, odim = 2;
    ScalarType xt = ScalarType::Float, tt = ScalarType::Float,
               ot = ScalarType::Float;
    void* tdata = coef;
    double inv = 0.5, out_scale = 0.25;
    int64_t zp = 0, qmin = -32767, qmax = 32767, scheme = kUniformCubic;
    int64_t lo = -8, hi = 7, shift = 2, G = 0, ozp = 0, omin = -32767,
            omax = 32767;
    bool inv_int = false, zp_double = false;
    size_t slots = 17;
    Error want = Error::InvalidArgument;
    float saved = 0.0f;
    int poked = -1;
    switch (e) {
      case kControl: want = Error::Ok; break;
      case kSlots16: slots = 16; want = Error::InvalidProgram; break;
      case kXInt: xt = ScalarType::Int; break;
      case kOutDouble: ot = ScalarType::Double; break;
      case kInvInt: inv_int = true; break;
      case kInvZero: inv = 0.0; break;
      case kInvNan: inv = NAN; break;
      case kInvNegative: inv = -0.5; break;
      case kInvNotF32: inv = 0.1; break;
      case kOutScaleInf: out_scale = INFINITY; break;
      case kOutScaleNotF32: out_scale = 0.1; break;
      case kZpDouble: zp_double = true; break;
      case kQminEqQmax: qmin = qmax = 0; break;
      case kQminLow: qmin = -32769; break;
      case kQmaxHigh: qmax = 32768; break;
      case kZpAbove: zp = 32768; qmax = 32767; break;
      case kScheme5: scheme = 5; break;
      case kSchemeNeg: scheme = -1; break;
      case kLoAboveHi: lo = 8; hi = 7; break;
      case kLoLow: lo = -32769; break;
      case kShift17: shift = 17; break;
      case kG17: G = 17; break;
      case kGNeg: G = -1; break;
      case kOutRangeEmpty: omin = omax = 0; break;
      case kOutZpOutside: ozp = 32767; omax = 100; break;
      case kOutMaxHigh: omax = 32768; break;
      case kTableInt: tt = ScalarType::Int; break;
      case kTable1D: tdim = 1; ts[0] = 16; tst[0] = 1; break;
      case kTable3Cols: ts[0] = 5; ts[1] = 3; tst[0] = 3; break;
      case kTableRows: ts[0] = 5; break;
      case kCoefNan: poked = 13; saved = coef[13]; coef[13] = NAN; break;
      case kCoefBig: poked = 6; saved = coef[6]; coef[6] = 1048576.0f; break;
      case kNsauLo: scheme = kNsauCubic; lo = -32768; hi = 7; break;
      case kNsauM9: scheme = kNsauCubic; shift = 9; break;
      case kLinearUnsafe:
        scheme = kUniformLinear;
        tt = ScalarType::Int;
        tdim = 1;
        ts[0] = 5;
        tst[0] = 1;
        tdata = nodes;
        nodes[3] = (1 << 30) + 1;
        break;
      case kDirectFloat: scheme = kDirect; tdim = 1; ts[0] = 16; tst[0] = 1; break;
      case kOutShape: os[0] = 3; os[1] = 2; ost[0] = 2; break;
      case kXStrided: xst[0] = 1; xst[1] = 2; break;
      case kTableStrided: tst[0] = 8; break;
      case kXChannelsLast:
        xdim = odim = 4;
        xs[0] = os[0] = 1; xs[1] = os[1] = 2; xs[2] = os[2] = 1; xs[3] = os[3] = 3;
        xo[0] = oo[0] = 0; xo[1] = oo[1] = 2; xo[2] = oo[2] = 3; xo[3] = oo[3] = 1;
        xst[0] = ost[0] = 6; xst[1] = ost[1] = 1; xst[2] = ost[2] = 6; xst[3] = ost[3] = 2;
        break;
    }
    TensorImpl xi(xt, xdim, xs, x, xo, xst, TensorShapeDynamism::STATIC);
    TensorImpl ti(tt, tdim, ts, tdata, to, tst, TensorShapeDynamism::STATIC);
    TensorImpl oi(ot, odim, os, out, oo, ost, TensorShapeDynamism::STATIC);
    for (int i = 0; i < 16; ++i) {
      out[i] = bits_float(0xA5A5A5A5u);
    }
    EValue ex{Tensor(&xi)}, et{Tensor(&ti)}, eo{Tensor(&oi)};
    EValue einv = inv_int ? EValue((int64_t)2) : EValue(inv);
    EValue ezp = zp_double ? EValue((double)zp) : EValue(zp);
    EValue eqmin{qmin}, eqmax{qmax}, escheme{scheme}, elo{lo}, ehi{hi},
        eshift{shift}, eG{G}, eoscale{out_scale}, eozp{ozp}, eomin{omin},
        eomax{omax};
    EValue* stack[17] = {&ex,    &et,     &einv,   &ezp,  &eqmin, &eqmax,
                         &escheme, &elo,  &ehi,    &eshift, &eG,  &eoscale,
                         &eozp,  &eomin,  &eomax,  &eo,   &eo};
    KernelRuntimeContext ctx;
    shim(ctx, Span<EValue*>(stack, slots));
    const Error got = ctx.failure_state();
    if (poked >= 0) {
      coef[poked] = saved;
    }
    nodes[3] = 300;
    if (want == Error::Ok) {
      Table t;
      t.scheme = kUniformCubic;
      t.lo = -8;
      t.hi = 7;
      t.shift = 2;
      t.G = 0;
      t.out_min = -32767;
      t.out_max = 32767;
      t.data = coef;
      t.count = 4;
      float expect[6];
      apply(t, x, expect, 6, 0.5f, 0, -32767, 32767, 0.25f, 0);
      control_ok = got == Error::Ok && memcmp(out, expect, sizeof expect) == 0;
      if (!control_ok) {
        printf("  MISMATCH lut16 malformed control: error %d\n", (int)got);
      }
      continue;
    }
    bool same = true;
    for (int i = 0; i < 16; ++i) {
      same &= float_bits(out[i]) == 0xA5A5A5A5u;
    }
    ++result.total;
    result.refused += got == want;
    result.untouched &= same;
    if (got != want || !same) {
      printf("  MISMATCH lut16 malformed '%s': error %d (want %d), out %s\n",
             kEditNames[e], (int)got, (int)want,
             same ? "untouched" : "WRITTEN");
    }
  }
  forget_tables();
}

// The flagged scheme words' malformed tables, each refused by
// validate() under its own name, and through the shim (InvalidArgument, out
// untouched): flags where they mean nothing, an offset table at the plain
// layout's size or one row short, nodes or a tail edge whose extrapolation
// leaves int32, a tail's edge cubic past 2^22.
struct Named {
  const char* name;
  int32_t scheme, lo, hi, shift, G, count;
  int poke;      // which entry to spoil (-1: none)
  double value;  // ... cubic: that coefficient; linear: that node
  const char* want;
};

const Named kNamed[] = {
    {"scheme 7", 7, -8, 7, 2, 0, 4, -1, 0.0, "scheme"},
    {"scheme flag 32", 32 | kUniformCubic, -8, 7, 2, 0, 4, -1, 0.0, "scheme"},
    {"tail on a direct table", kTail | kDirect, -8, 7, 0, 0, 16, -1, 0.0,
     "tail on a direct table"},
    {"tail on an NSAU table", kTail | kNsauCubic, 0, 7, 2, 0, 0, -1, 0.0f,
     "tail on an NSAU table"},
    {"offset on a uniform table", kOffset | kUniformCubic, -8, 7, 2, 0, 4, -1,
     0.0f, "offset on a uniform table"},
    {"offset on a direct table", kOffset | kDirect, -8, 7, 0, 0, 16, -1, 0.0f,
     "offset on a direct table"},
    {"offset table at the plain size", kOffset | kNsauCubic, 534, 32767, 5, 0,
     352, -1, 0.0f, "table size"},
    {"offset table a row short", kOffset | kNsauCubic, 534, 32767, 5, 0, 190, -1,
     0.0f, "table size"},
    {"offset node chain at the plain size", kOffset | kNsauLinear, -5000, -37, 3,
     4, 166, -1, 0.0f, "table size"},
    {"offset lo -32768", kOffset | kNsauCubic, -32768, 7, 2, 0, 0, -1, 0.0f,
     "domain"},
    {"offset node 2^30 + 1", kOffset | kNsauLinear, 534, 32767, 4, 6, 0, 7,
     1073741825.0, "nodes outside int32"},
    // inside int32 over its own segment, not out to code -32768
    {"tail linear edge past int32", kTail | kUniformLinear, -3000, 3000, 7, 4, 0,
     0, 1000000.0, "tail outside int32"},
    // c3 = 10^5 within 2^19, but 4^3 c3 past 2^22 at code -32768
    {"tail cubic edge past 2^22", kTail | kUniformCubic, -16382, 16385, 12, 0, 0,
     3, 100000.0, "tail past 2^22"},
    {"tail cubic edge coefficient NaN", kTail | kUniformCubic, -16382, 16385, 12,
     0, 0, 2, NAN, "coefficient"},
};

void check_named(Malformed& result) {
  // the tables on the heap, as check_tables' buffer: 8 KB of static storage
  // would take internal RAM from every test build for the whole run
  constexpr int kRows = 400;
  float* rows = static_cast<float*>(malloc(sizeof(float) * 4 * kRows));
  int32_t* nodes = static_cast<int32_t*>(malloc(sizeof(int32_t) * kRows));
  if (rows == nullptr || nodes == nullptr) {
    free(rows);
    free(nodes);
    result.total += (int)(sizeof kNamed / sizeof kNamed[0]);
    printf("  MISMATCH lut16 malformed: no memory for the named cases\n");
    return;
  }
  alignas(16) static int16_t entries[16];
  alignas(16) static float x[16], out[16];
  for (int i = 0; i < 16; ++i) {
    x[i] = 0.75f * (float)(i - 8);
    entries[i] = (int16_t)(3 * i);
  }
  using Sizes = TensorImpl::SizesType;
  using Order = TensorImpl::DimOrderType;
  using Strides = TensorImpl::StridesType;
  for (const Named& c : kNamed) {
    forget_tables();
    const int32_t base = c.scheme & kBaseMask;
    Table t;
    t.scheme = c.scheme;
    t.lo = c.lo;
    t.hi = c.hi;
    t.shift = c.shift;
    t.G = c.G;
    t.out_min = -32767;
    t.out_max = 32767;
    // count 0: the right size for the layout
    t.count = c.count != 0 ? c.count
              : (base == kNsauLinear || base == kNsauCubic)
                  ? nsau_layout(c.scheme, c.lo, c.hi, c.shift).entries
                  : ((c.hi - c.lo) >> c.shift) + 1 + (base == kUniformLinear ? 1 : 0);
    const bool linear = base == kUniformLinear || base == kNsauLinear;
    if (base == kDirect) {
      t.data = entries;
    } else if (linear) {
      for (int i = 0; i < kRows; ++i) {
        nodes[i] = 64 * (i % 7);
      }
      if (c.poke >= 0) {
        nodes[c.poke] = (int32_t)c.value;
      }
      t.data = nodes;
    } else {
      for (int i = 0; i < 4 * kRows; ++i) {
        rows[i] = (i % 4) == 0 ? (float)(i % 1000) : 0.0f;
      }
      if (c.poke >= 0) {
        rows[c.poke] = (float)c.value;
      }
      t.data = rows;
    }
    const char* why = (t.count >= 0 && t.count <= kRows) ? validate(t) : "count";
    // and through the shim
    Sizes xs[1] = {16}, ts[2] = {(Sizes)t.count, 4}, os[1] = {16};
    Order xo[1] = {0}, to[2] = {0, 1}, oo[1] = {0};
    Strides xst[1] = {1}, tst[2] = {linear || base == kDirect ? 1 : 4, 1},
            ost[1] = {1};
    const ScalarType tt = base == kDirect ? ScalarType::Short
                          : linear        ? ScalarType::Int
                                          : ScalarType::Float;
    TensorImpl xi(ScalarType::Float, 1, xs, x, xo, xst,
                  TensorShapeDynamism::STATIC);
    TensorImpl ti(tt, linear || base == kDirect ? 1 : 2, ts,
                  const_cast<void*>(t.data), to, tst,
                  TensorShapeDynamism::STATIC);
    TensorImpl oi(ScalarType::Float, 1, os, out, oo, ost,
                  TensorShapeDynamism::STATIC);
    for (int i = 0; i < 16; ++i) {
      out[i] = bits_float(0xA5A5A5A5u);
    }
    EValue ex{Tensor(&xi)}, et{Tensor(&ti)}, eo{Tensor(&oi)};
    EValue einv{(double)0.5}, ezp{(int64_t)0}, eqmin{(int64_t)-32767},
        eqmax{(int64_t)32767}, escheme{(int64_t)c.scheme}, elo{(int64_t)c.lo},
        ehi{(int64_t)c.hi}, eshift{(int64_t)c.shift}, eG{(int64_t)c.G},
        eoscale{(double)0.25}, eozp{(int64_t)0}, eomin{(int64_t)-32767},
        eomax{(int64_t)32767};
    EValue* stack[17] = {&ex,    &et,    &einv,    &ezp, &eqmin, &eqmax,
                         &escheme, &elo, &ehi,     &eshift, &eG, &eoscale,
                         &eozp,  &eomin, &eomax,   &eo,  &eo};
    KernelRuntimeContext ctx;
    shim(ctx, Span<EValue*>(stack, 17));
    bool same = true;
    for (int i = 0; i < 16; ++i) {
      same &= float_bits(out[i]) == 0xA5A5A5A5u;
    }
    const bool named = why != nullptr && strcmp(why, c.want) == 0;
    ++result.total;
    result.refused += named && ctx.failure_state() == Error::InvalidArgument;
    result.untouched &= same;
    if (!named || ctx.failure_state() != Error::InvalidArgument || !same) {
      printf("  MISMATCH lut16 malformed '%s': validate %s (want %s), error %d, "
             "out %s\n", c.name, why ? why : "ok", c.want,
             (int)ctx.failure_state(), same ? "untouched" : "WRITTEN");
    }
  }
  forget_tables();
  free(rows);
  free(nodes);
}

// A uniform cubic table over [-8, 7] (four segments) in the program's
// read-only data: flash on the device, so checked() copies it.
const float kFlashTable[16] = {-11.0f, 3.0f,  -6.0f, 8.0f,  -1.0f, -10.0f,
                               4.0f,   -5.0f, 9.0f,  0.0f,  -9.0f, 5.0f,
                               -4.0f,  10.0f, 1.0f,  -8.0f};

// checked(): a table is validated on its first use only and evaluates as it
// did in place (from its internal-RAM copy on the device); a refused table is
// never remembered; after forget_tables() an edited table is read again.
bool check_remembered(int* copied) {
  alignas(16) static float edited[16];
  memcpy(edited, kFlashTable, sizeof edited);
  Table t;
  t.scheme = kUniformCubic;
  t.lo = -8;
  t.hi = 7;
  t.shift = 2;
  t.G = 0;
  t.out_min = -32767;
  t.out_max = 32767;
  t.data = kFlashTable;
  t.count = 4;
  forget_tables();
  const Stats s0 = stats();
  Table first, second;
  bool ok = checked(t, &first) == nullptr && checked(t, &second) == nullptr &&
            first.data == second.data;
  const Stats s1 = stats();
  ok = ok && s1.validations - s0.validations == 1 &&
       s1.reused - s0.reused == 1;
#if defined(__XTENSA__)
  // copied: a different pointer to the same bytes
  ok = ok && first.data != t.data &&
       memcmp(first.data, t.data, sizeof kFlashTable) == 0 &&
       s1.copied - s0.copied == 1;
#else
  ok = ok && first.data == t.data && s1.in_place - s0.in_place == 1;
#endif
  *copied = (int)(s1.copied - s0.copied);
  if (ok) {
    int16_t in[16], a[16], b[16];
    for (int i = 0; i < 16; ++i) {
      in[i] = (int16_t)(i - 8);
    }
    eval_codes(t, in, a, 16);
    eval_codes(first, in, b, 16);
    ok = memcmp(a, b, sizeof a) == 0;
  }
  t.data = edited;
  edited[6] = 1048576.0f;  // past 2^19: refused on every call
  Table use;
  ok = ok && checked(t, &use) != nullptr && checked(t, &use) != nullptr;
  const Stats s2 = stats();
  ok = ok && s2.validations - s1.validations == 2 && s2.reused == s1.reused;
  edited[6] = kFlashTable[6];
  forget_tables();
  if (!ok) {
    printf("  MISMATCH lut16 checked(): a table remembered wrongly\n");
  }
  return ok;
}

}  // namespace

void run_conformance_check() {
  const Stats saved = stats();
  size_t biggest = 0;
  for (const golden::TableSpec& s : golden::kTables) {
    const size_t bytes = (size_t)s.count * element_bytes(s.scheme);
    biggest = bytes > biggest ? bytes : biggest;
  }
  void* buf = malloc(biggest);
  Tally tally;
  if (buf != nullptr) {
    check_tables(tally, buf);
    check_stage_a(tally, buf);
    free(buf);
  }
  const int tables = (int)(sizeof golden::kTables / sizeof golden::kTables[0]);
  const int cases = (int)(sizeof golden::kCases / sizeof golden::kCases[0]);
  Malformed malformed;
  bool control_ok = false;
  check_malformed(malformed, control_ok);
  check_named(malformed);
  int copied = 0;
  const bool remembered = check_remembered(&copied);
  const uint64_t mismatches =
      (uint64_t)(tables - tally.device) + (uint64_t)(tables - tally.reference) +
      (uint64_t)(tables - tally.generated) +
      (uint64_t)(cases - tally.stage_a) + (uint64_t)(cases - tally.inputs) +
      (uint64_t)(malformed.total - malformed.refused) +
      (malformed.untouched ? 0 : 1) + (control_ok ? 0 : 1) +
      (remembered ? 0 : 1) + (buf == nullptr ? 1 : 0);
  printf("lut16 check: %llu case(s), %llu mismatch(es) (tables %d/%d "
         "generated, codes device %d/%d reference %d/%d, stage A %d/%d inputs "
         "%d/%d), malformed %d/%d refused, out %s, control %s, validated once "
         "%s (copied %d)\n",
         (unsigned long long)tally.cases, (unsigned long long)mismatches,
         tally.generated, tables, tally.device, tables, tally.reference,
         tables, tally.stage_a, cases, tally.inputs, cases, malformed.refused,
         malformed.total, malformed.untouched ? "untouched" : "WRITTEN",
         control_ok ? "accepted" : "REFUSED", remembered ? "ok" : "FAILED",
         copied);
  restore_stats(saved);
}

}  // namespace lut16
}  // namespace espressif
