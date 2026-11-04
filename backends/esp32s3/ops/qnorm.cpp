/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// espressif::qnorm (qnorm.h): the kernel, its shim and counters. Built
// -ffp-contract=off on the host and the device: every product here is
// followed by a quantise or a separately rounded add in the graph it
// replaces, and GCC may not fuse them into madd.s. On the device qnorm.lf
// links this file after every other flash object, as quant_lut.lf does, so
// the story firmware's code keeps its addresses. The int16 kinds
// (2 and 3) use lut16's tables, so the story firmware, which links none of
// quant_lut16.cpp, builds without them.

#include "qnorm.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include "quant_lut16.h"
#include "quant_rules.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace espressif {
namespace qnorm {
namespace {

uint64_t g_calls = 0;
uint64_t g_rows = 0;
uint64_t g_refusals = 0;

// quant_lut's quantise step: v = fl(x * inv), NaN -> 0 (the zero point's
// code, quant_rules.h), clamped to [-512, 512], rounded half to even by
// 1.5 * 2^23, plus the zero point, clamped to the code range.
inline int32_t code_of(float x, const Quant& q) {
  float b = quant_scaled(x * q.inv);
  b = b > -512.0f ? b : -512.0f;
  b = b < 512.0f ? b : 512.0f;
  const float r = (b + 12582912.0f) - 12582912.0f;
  int32_t c = (int32_t)r + q.zero_point;
  c = c < q.quant_min ? q.quant_min : c;
  return c > q.quant_max ? q.quant_max : c;
}

// dequantize_per_tensor: fl(fl(c - zp) * scale); c - zp is exact.
inline float value_of(int32_t c, const Quant& q) {
  return (float)(c - q.zero_point) * q.scale;
}

// A slot's pair, or nothing when the slot is absent.
inline float fq(float v, const Quant& q) {
  return q.present ? value_of(code_of(v, q), q) : v;
}

inline const float* table(const float* tables, int32_t row) {
  return tables + (size_t)row * kTableLen;
}

void apply_rms(int32_t flags, const float* x, float* out, size_t rows,
               size_t n, const Quant* quant, const float* tables,
               const float* weight, float eps) {
  const Quant& qx = quant[kRmsX];
  const Quant& qsq = quant[kRmsSq];
  const Quant& qred = quant[kRmsRed];
  const Quant& qm = quant[kRmsM];
  const float* tsq = table(tables, kRmsTSq) - qx.quant_min;
  const float* tr = table(tables, kRmsTR) - qred.quant_min;
  const float count = (float)n;
  for (size_t row = 0; row < rows; ++row) {
    const float* xr = x + row * n;
    float* yr = out + row * n;
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      float s;
      if (qx.present) {
        s = tsq[code_of(xr[i], qx)];
      } else {
        const float square = xr[i] * xr[i];
        s = fq(square, qsq);
      }
      acc += s;
    }
    float v = acc;
    if (flags & kMean) {
      v = v / count;
    }
    if (flags & kEps) {
      v = v + eps;
    }
    const float r = tr[code_of(v, qred)];
    for (size_t i = 0; i < n; ++i) {
      const float a = qx.present ? value_of(code_of(xr[i], qx), qx) : xr[i];
      const float m = a * r;
      if (weight != nullptr) {
        const float mq = fq(m, qm);
        yr[i] = mq * weight[i];
      } else {
        yr[i] = m;
      }
    }
  }
}

void apply_layer(const float* x, float* out, size_t rows, size_t n,
                 const Quant* quant, const float* tables,
                 const float* weight, const float* bias) {
  const Quant& qx = quant[kLnX];
  const Quant& qred1 = quant[kLnRed1];
  const Quant& qred2 = quant[kLnRed2];
  const Quant& qc1 = quant[kLnC1];
  const Quant& qc2 = quant[kLnC2];
  const Quant& qred3 = quant[kLnRed3];
  const Quant& qn = quant[kLnN];
  const Quant& qw = quant[kLnW];
  const float* tm1 = table(tables, kLnTM1) - qred1.quant_min;
  const float* tm2 = table(tables, kLnTM2) - qred2.quant_min;
  const float* tsq = table(tables, kLnTSq) - qc2.quant_min;
  const float* tr = table(tables, kLnTR) - qred3.quant_min;
  for (size_t row = 0; row < rows; ++row) {
    const float* xr = x + row * n;
    float* yr = out + row * n;
    float acc1 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      acc1 += value_of(code_of(xr[i], qx), qx);
    }
    const float mu1 = tm1[code_of(acc1, qred1)];
    const float mu2 = tm2[code_of(acc1, qred2)];
    float acc2 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      const float a = value_of(code_of(xr[i], qx), qx);
      const float d2 = a - mu2;
      acc2 += tsq[code_of(d2, qc2)];
    }
    const float r = tr[code_of(acc2, qred3)];
    for (size_t i = 0; i < n; ++i) {
      const float a = value_of(code_of(xr[i], qx), qx);
      const float d1 = fq(a - mu1, qc1);
      const float m = d1 * r;
      float y = m;
      if (weight != nullptr) {
        const float w = fq(m, qn) * weight[i];
        y = bias != nullptr ? fq(w, qw) + bias[i] : w;
      } else if (bias != nullptr) {
        y = fq(m, qn) + bias[i];
      }
      yr[i] = y;
    }
  }
}

// Contiguous strides in the default dim order (quant_lut.cpp's dense): size-1
// axes may carry any stride.
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

// A quant row: integral zero point and range, a float32 range inside
// [low, high] of at most `codes` codes holding the zero point (int8: [-128,
// 255] and 256; int16: [-32768, 32767] and 65,536), a positive finite
// multiplier and scale. Absent rows (quant_min > quant_max) are skipped.
bool read_quant(const float* row, Quant* q, float low = -128.0f,
                float high = 255.0f, float codes = (float)kTableLen) {
  const float qmin = row[3];
  const float qmax = row[4];
  q->present = !(qmin > qmax);
  q->inv = row[0];
  q->scale = row[1];
  q->zero_point = 0;
  q->quant_min = 0;
  q->quant_max = 0;
  if (!q->present) {
    return true;
  }
  const float zp = row[2];
  if (!(qmin >= low && qmax <= high && qmin < qmax &&
        qmax - qmin + 1.0f <= codes && zp >= qmin && zp <= qmax &&
        std::floor(zp) == zp && std::floor(qmin) == qmin &&
        std::floor(qmax) == qmax && std::isfinite(q->inv) && q->inv > 0.0f &&
        std::isfinite(q->scale) && q->scale > 0.0f)) {
    return false;
  }
  q->zero_point = (int32_t)zp;
  q->quant_min = (int32_t)qmin;
  q->quant_max = (int32_t)qmax;
  return true;
}

void refuse(KernelRuntimeContext& ctx, Error error, const char* why) {
  printf("  [qnorm] refused: %s\n", why);
  ++g_refusals;
  ctx.fail(error);
}

bool optional_vector(const EValue* value, size_t n, const Tensor** out) {
  *out = nullptr;
  if (value->isNone()) {
    return true;
  }
  if (!value->isTensor()) {
    return false;
  }
  const Tensor& t = value->toTensor();
  if (t.scalar_type() != ScalarType::Float || (size_t)t.numel() != n ||
      !dense(t)) {
    return false;
  }
  *out = &t;
  return true;
}

#if !defined(ET_PLE_ONLY)
// --- int16 codes (kinds 2 and 3) -------------------------------------------

constexpr float kRail16 = 65536.0f;

// lut16's quantise step: v = fl(x * inv), NaN -> 0 (quant_rules.h), clamped
// to [-2^16, 2^16], rounded half to even, plus the zero point, clamped to
// the range.
inline int32_t code16(float x, float inv, int32_t zp, int32_t qmin,
                      int32_t qmax) {
  float b = quant_scaled(x * inv);
  b = b > -kRail16 ? b : -kRail16;
  b = b < kRail16 ? b : kRail16;
  const float r = (b + 12582912.0f) - 12582912.0f;
  int32_t c = (int32_t)r + zp;
  c = c < qmin ? qmin : c;
  return c > qmax ? qmax : c;
}

inline float fq16(float v, const Quant& q) {
  return q.present
             ? value_of(code16(v, q.inv, q.zero_point, q.quant_min,
                               q.quant_max),
                        q)
             : v;
}

inline float word_float(int32_t w) {
  float f;
  memcpy(&f, &w, sizeof f);
  return f;
}

// One TABLE step's table: as lut16::checked() gave it, and its exceptions.
struct Lut {
  lut16::Table table;
  const int32_t* exceptions;  // (input code, output code) pairs
  int32_t count;
};

struct Program {
  const int32_t* steps;
  int32_t count;
};

int32_t lookup(const Lut& l, int32_t c) {
  int32_t lo = 0;
  int32_t hi = l.count - 1;
  while (lo <= hi) {
    const int32_t mid = (lo + hi) >> 1;
    const int32_t key = l.exceptions[2 * mid];
    if (key == c) {
      return l.exceptions[2 * mid + 1];
    }
    if (key < c) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return lut16::eval_code(l.table, c);
}

float run(const Program& p, const Lut* luts, float v) {
  int32_t c = 0;
  for (int32_t s = 0; s < p.count; ++s) {
    const int32_t* a = p.steps + kStepWords * s;
    switch (a[0]) {
      case kCode:
        c = code16(v, word_float(a[1]), a[3], a[4], a[5]);
        break;
      case kTable:
        c = lookup(luts[a[1]], c);
        break;
      case kDeq:
        v = (float)(c - a[3]) * word_float(a[2]);
        break;
      case kMul:
        v = v * word_float(a[1]);
        break;
      case kAdd:
        v = v + word_float(a[1]);
        break;
      case kDiv:
        v = v / word_float(a[1]);
        break;
      case kRsub:
        v = word_float(a[1]) - v;
        break;
      default:  // kRdiv (parse16 accepts nothing else)
        v = word_float(a[1]) / v;
        break;
    }
  }
  return v;
}

void apply_rms16(const float* x, float* out, size_t rows, size_t n,
                 const Quant* quant, const Program* programs, const Lut* luts,
                 const float* weight) {
  const Quant& qx = quant[kR16X];
  const Quant& qsq = quant[kR16Sq];
  const Quant& qm = quant[kR16M];
  for (size_t row = 0; row < rows; ++row) {
    const float* xr = x + row * n;
    float* yr = out + row * n;
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      const float a = fq16(xr[i], qx);
      const float square = a * a;
      acc += fq16(square, qsq);
    }
    const float r = run(programs[0], luts, acc);
    for (size_t i = 0; i < n; ++i) {
      const float a = fq16(xr[i], qx);
      const float m = a * r;
      if (weight != nullptr) {
        const float mq = fq16(m, qm);
        yr[i] = mq * weight[i];
      } else {
        yr[i] = m;
      }
    }
  }
}

void apply_layer16(const float* x, float* out, size_t rows, size_t n,
                   const Quant* quant, const Program* programs,
                   const Lut* luts, const float* weight, const float* bias) {
  const Quant& qx = quant[kL16X];
  const Quant& qc1 = quant[kL16C1];
  const Quant& qc2 = quant[kL16C2];
  const Quant& qsq = quant[kL16Sq];
  const Quant& qn = quant[kL16N];
  const Quant& qw = quant[kL16W];
  for (size_t row = 0; row < rows; ++row) {
    const float* xr = x + row * n;
    float* yr = out + row * n;
    float acc1 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      acc1 += fq16(xr[i], qx);
    }
    const float mu1 = run(programs[0], luts, acc1);
    const float mu2 = run(programs[1], luts, acc1);
    float acc2 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      const float a = fq16(xr[i], qx);
      const float c = fq16(a - mu2, qc2);
      const float square = c * c;
      acc2 += fq16(square, qsq);
    }
    const float r = run(programs[2], luts, acc2);
    for (size_t i = 0; i < n; ++i) {
      const float a = fq16(xr[i], qx);
      const float d1 = fq16(a - mu1, qc1);
      const float m = d1 * r;
      float y = m;
      if (weight != nullptr) {
        const float w = fq16(m, qn) * weight[i];
        y = bias != nullptr ? fq16(w, qw) + bias[i] : w;
      } else if (bias != nullptr) {
        y = fq16(m, qn) + bias[i];
      }
      yr[i] = y;
    }
  }
}

bool int16_code(int32_t v) {
  return v >= -32768 && v <= 32767;
}

bool finite_positive(float f) {
  return std::isfinite(f) && f > 0.0f;
}

// int32 words of a lut16 table's data, or -1 for an unknown scheme (a
// scheme word's flags are lut16::validate()'s to judge).
int64_t data_words(int32_t scheme, int32_t count) {
  switch (scheme & lut16::kBaseMask) {
    case lut16::kDirect:
      return ((int64_t)count + 1) / 2;
    case lut16::kUniformLinear:
    case lut16::kNsauLinear:
      return count;
    case lut16::kUniformCubic:
    case lut16::kNsauCubic:
      return 4 * (int64_t)count;
    default:
      return -1;
  }
}

// A kind 2 or 3 tables tensor (qnorm.h): every word it names inside its
// `words`, `want` programs of known steps whose arguments hold (a TABLE or
// DEQ only after a CODE, a program ending on a value, finite constants), and
// each table through lut16::checked() with its exceptions in increasing
// input order inside the table's output range. nullptr, or what is wrong.
const char* parse16(const int32_t* w, int64_t words, int32_t want,
                    Program* programs, Lut* luts) {
  if (words < 2 || w[0] != want || w[1] < 0 || w[1] > kMaxLuts) {
    return "program or table count";
  }
  const int32_t tables = w[1];
  const int64_t head = 2 + 2 * (int64_t)want + (int64_t)kTableWords * tables;
  if (head > words) {
    return "header";
  }
  for (int32_t t = 0; t < tables; ++t) {
    const int32_t* h = w + 2 + 2 * want + kTableWords * t;
    const int64_t need = data_words(h[0], h[7]);
    if (need < 1 || h[7] < 1 || h[8] < head || h[8] + need > words) {
      return "table data";
    }
    if (h[9] < 0 || h[9] > kMaxExceptions || h[10] < head ||
        h[10] + 2 * (int64_t)h[9] > words) {
      return "exceptions";
    }
    lut16::Table table;
    table.scheme = h[0];
    table.lo = h[1];
    table.hi = h[2];
    table.shift = h[3];
    table.G = h[4];
    table.out_min = h[5];
    table.out_max = h[6];
    table.data = w + h[8];
    table.count = h[7];
    const char* why = lut16::checked(table, &luts[t].table);
    if (why != nullptr) {
      return why;
    }
    luts[t].exceptions = w + h[10];
    luts[t].count = h[9];
    for (int32_t e = 0; e < h[9]; ++e) {
      const int32_t in = luts[t].exceptions[2 * e];
      const int32_t to = luts[t].exceptions[2 * e + 1];
      if (!int16_code(in) || to < table.out_min || to > table.out_max ||
          (e > 0 && in <= luts[t].exceptions[2 * e - 2])) {
        return "exceptions";
      }
    }
  }
  for (int32_t p = 0; p < want; ++p) {
    const int32_t first = w[2 + 2 * p];
    const int32_t steps = w[3 + 2 * p];
    if (steps < 1 || steps > kMaxSteps || first < head ||
        first + (int64_t)kStepWords * steps > words) {
      return "program";
    }
    bool code = false;
    for (int32_t s = 0; s < steps; ++s) {
      const int32_t* a = w + first + kStepWords * s;
      bool ok = true;
      switch (a[0]) {
        case kCode:
          ok = finite_positive(word_float(a[1])) && int16_code(a[4]) &&
               int16_code(a[5]) && a[4] < a[5] && a[4] <= a[3] && a[3] <= a[5];
          code = true;
          break;
        case kTable:
          ok = code && a[1] >= 0 && a[1] < tables;
          break;
        case kDeq:
          ok = code && finite_positive(word_float(a[2])) && int16_code(a[3]);
          break;
        case kMul:
        case kAdd:
        case kDiv:
        case kRsub:
        case kRdiv:
          ok = std::isfinite(word_float(a[1]));
          break;
        default:
          ok = false;
      }
      if (!ok) {
        return "program step";
      }
    }
    const int32_t last = w[first + kStepWords * (steps - 1)];
    if (last == kCode || last == kTable) {
      return "program ends on a code";
    }
    programs[p].steps = w + first;
    programs[p].count = steps;
  }
  return nullptr;
}

// The shim's kinds 2 and 3, from the argument checks on.
void shim16(KernelRuntimeContext& ctx, Span<EValue*> st, int64_t kind,
            int64_t flags, double eps) {
  const Tensor& x = st[0]->toTensor();
  const Tensor& tables = st[1]->toTensor();
  const Tensor& quant = st[2]->toTensor();
  Tensor& out = st[8]->toTensor();
  const int32_t slots = kind == kRms16 ? kR16Slots : kL16Slots;
  if (x.scalar_type() != ScalarType::Float || x.dim() < 1 || x.numel() < 1 ||
      !dense(x) || out.scalar_type() != ScalarType::Float ||
      tables.scalar_type() != ScalarType::Int || tables.dim() != 1 ||
      !dense(tables) || quant.scalar_type() != ScalarType::Float ||
      quant.dim() != 2 || quant.size(0) != slots || quant.size(1) != 5 ||
      !dense(quant) || flags != 0 || !std::isfinite(eps) ||
      (double)(float)eps != eps) {
    refuse(ctx, Error::InvalidArgument, "tensor shapes, dtypes or flags");
    return;
  }
  const size_t n = (size_t)x.size(x.dim() - 1);
  const Tensor* weight = nullptr;
  const Tensor* bias = nullptr;
  if (!optional_vector(st[3], n, &weight) ||
      !optional_vector(st[4], n, &bias) ||
      (kind == kRms16 && bias != nullptr)) {
    refuse(ctx, Error::InvalidArgument, "weight or bias");
    return;
  }
  Quant q[kL16Slots];
  const float* rows = quant.const_data_ptr<float>();
  for (int32_t s = 0; s < slots; ++s) {
    if (!read_quant(rows + 5 * s, &q[s], -32768.0f, 32767.0f, 65536.0f)) {
      refuse(ctx, Error::InvalidArgument, "quantiser");
      return;
    }
  }
  // The layer norm's sums read x's codes' values.
  if (kind == kLayer16 && !q[kL16X].present) {
    refuse(ctx, Error::InvalidArgument, "the input's quantiser is absent");
    return;
  }
  Program programs[3];
  Lut luts[kMaxLuts];
  const char* why =
      parse16(tables.const_data_ptr<int32_t>(), (int64_t)tables.numel(),
              kind == kRms16 ? 1 : 3, programs, luts);
  if (why != nullptr) {
    refuse(ctx, Error::InvalidArgument, why);
    return;
  }
  if (executorch::runtime::resize_tensor(out, x.sizes()) != Error::Ok ||
      !dense(out)) {
    refuse(ctx, Error::InvalidArgument, "out");
    return;
  }
  const size_t rows_count = (size_t)x.numel() / n;
  const float* w = weight ? weight->const_data_ptr<float>() : nullptr;
  if (kind == kRms16) {
    apply_rms16(x.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                rows_count, n, q, programs, luts, w);
  } else {
    apply_layer16(x.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                  rows_count, n, q, programs, luts, w,
                  bias ? bias->const_data_ptr<float>() : nullptr);
  }
  ++g_calls;
  g_rows += rows_count;
}
#endif  // !ET_PLE_ONLY

}  // namespace

void apply(int32_t kind, int32_t flags, const float* x, float* out,
           size_t rows, size_t n, const Quant* quant, const float* tables,
           const float* weight, const float* bias, float eps) {
  if (kind == kRms) {
    apply_rms(flags, x, out, rows, n, quant, tables, weight, eps);
  } else {
    apply_layer(x, out, rows, n, quant, tables, weight, bias);
  }
}

void shim(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Eight arguments, out, and the returned alias of out.
  if (st.size() != 10) {
    refuse(ctx, Error::InvalidProgram, "argument count");
    return;
  }
  // Types first: an EValue accessor aborts on the wrong tag.
  if (!st[0]->isTensor() || !st[1]->isTensor() || !st[2]->isTensor() ||
      !st[5]->isInt() || !st[6]->isInt() || !st[7]->isDouble() ||
      !st[8]->isTensor()) {
    refuse(ctx, Error::InvalidArgument, "argument types");
    return;
  }
  const Tensor& x = st[0]->toTensor();
  const Tensor& tables = st[1]->toTensor();
  const Tensor& quant = st[2]->toTensor();
  const int64_t kind = st[5]->toInt();
  const int64_t flags = st[6]->toInt();
  const double eps = st[7]->toDouble();
  Tensor& out = st[8]->toTensor();
#if !defined(ET_PLE_ONLY)
  if (kind == kRms16 || kind == kLayer16) {
    shim16(ctx, st, kind, flags, eps);
    return;
  }
#endif
  if (kind != kRms && kind != kLayer) {
    refuse(ctx, Error::InvalidArgument, "kind");
    return;
  }
  const int32_t slots = kind == kRms ? kRmsSlots : kLnSlots;
  const int32_t rows_needed = kind == kRms ? kRmsTables : kLnTables;
  const int32_t known = kind == kRms ? (kMean | kEps) : 0;
  if (x.scalar_type() != ScalarType::Float || x.dim() < 1 || x.numel() < 1 ||
      !dense(x) || out.scalar_type() != ScalarType::Float ||
      tables.scalar_type() != ScalarType::Float || tables.dim() != 2 ||
      tables.size(0) != rows_needed || tables.size(1) != kTableLen ||
      !dense(tables) || quant.scalar_type() != ScalarType::Float ||
      quant.dim() != 2 || quant.size(0) != slots || quant.size(1) != 5 ||
      !dense(quant) || (flags & ~(int64_t)known) != 0 || !std::isfinite(eps) ||
      (double)(float)eps != eps) {
    refuse(ctx, Error::InvalidArgument, "tensor shapes, dtypes or flags");
    return;
  }
  const size_t n = (size_t)x.size(x.dim() - 1);
  const Tensor* weight = nullptr;
  const Tensor* bias = nullptr;
  if (!optional_vector(st[3], n, &weight) ||
      !optional_vector(st[4], n, &bias) || (kind == kRms && bias != nullptr)) {
    refuse(ctx, Error::InvalidArgument, "weight or bias");
    return;
  }
  Quant q[kLnSlots];
  const float* rows = quant.const_data_ptr<float>();
  for (int32_t s = 0; s < slots; ++s) {
    if (!read_quant(rows + 5 * s, &q[s])) {
      refuse(ctx, Error::InvalidArgument, "quantiser");
      return;
    }
  }
  // The slots whose codes key a table must be present, and so must the
  // layer norm's input quantiser (its sums read the codes' values).
  const bool keyed = kind == kRms
      ? q[kRmsRed].present
      : q[kLnX].present && q[kLnRed1].present && q[kLnRed2].present &&
        q[kLnC2].present && q[kLnRed3].present;
  if (!keyed) {
    refuse(ctx, Error::InvalidArgument, "a table's quantiser is absent");
    return;
  }
  if (executorch::runtime::resize_tensor(out, x.sizes()) != Error::Ok ||
      !dense(out)) {
    refuse(ctx, Error::InvalidArgument, "out");
    return;
  }
  const size_t rows_count = (size_t)x.numel() / n;
  apply((int32_t)kind, (int32_t)flags, x.const_data_ptr<float>(),
        out.mutable_data_ptr<float>(), rows_count, n, q,
        tables.const_data_ptr<float>(),
        weight ? weight->const_data_ptr<float>() : nullptr,
        bias ? bias->const_data_ptr<float>() : nullptr, (float)eps);
  ++g_calls;
  g_rows += rows_count;
}

Stats stats() {
  return {g_calls, g_rows, g_refusals};
}

void report_dispatch() {
  if (g_calls != 0 || g_refusals != 0) {
    printf("qnorm dispatch: calls %llu, rows %llu, refusals %llu\n",
           (unsigned long long)g_calls, (unsigned long long)g_rows,
           (unsigned long long)g_refusals);
  }
}

}  // namespace qnorm
}  // namespace espressif
