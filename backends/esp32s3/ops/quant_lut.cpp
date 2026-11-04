/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// espressif::quant_lut (quant_lut.h): the kernel, its shim and counters (the
// conformance check is quant_lut_check.cpp, so a build that never runs it --
// the story's -- links none of it, its strings included). Built
// -ffp-contract=off on the host and the device:
// the quantize step has no multiply-add to fuse (the product's uses are a
// comparison and a select), but the flag keeps it that way, and run_parity
// checks the object code for madd.s/msub.s. On the device quant_lut.lf links
// this file after every other flash object, so the story firmware's code
// keeps its addresses.

#include "quant_lut.h"

#include <cmath>
#include <cstdio>

#include "elementwise.h"

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include "quant_rules.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace espressif {
namespace quant_lut {
namespace {

// Table entries are copied as words: no FP instruction touches them.
typedef uint32_t __attribute__((__may_alias__)) word_t;

// GCC at -Os re-materialises float constants on every loop iteration (l32r +
// wfr each); an empty asm keeps each in its register. No-op off Xtensa.
#if defined(__XTENSA__)
#define QUANT_LUT_PIN(f) __asm__("" : "+f"(f))
#else
#define QUANT_LUT_PIN(f) ((void)0)
#endif

uint64_t g_calls = 0;
uint64_t g_elements = 0;
uint64_t g_refusals = 0;
// The code-to-code variant's (Stage B).
uint64_t g_codes_calls = 0;
uint64_t g_codes_elements = 0;
uint64_t g_codes_refusals = 0;

// Contiguous strides in the default dim order (espressif_ops.cpp's
// dense_tensor): size-1 axes may carry any stride.
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

// Whether a loop writing out[i] right after reading x[i] gives what the
// element loops below give: out is x itself, or apart from it. A partly
// overlapping pair stays on the element loops, whose order defines it.
bool same_or_apart(const void* x, const void* out, size_t bytes) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(x);
  const uintptr_t b = reinterpret_cast<uintptr_t>(out);
  return a == b || a + bytes <= b || b + bytes <= a;
}

// Whether two byte ranges share no byte. The table must be apart from out:
// the four-lane loops read four entries before storing four results, where
// the element loops read each entry just before the store that may overwrite
// it, so an out over the table -- exactly or in part -- stays on the element
// loops.
bool apart(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  const uintptr_t x = reinterpret_cast<uintptr_t>(a);
  const uintptr_t y = reinterpret_cast<uintptr_t>(b);
  return x + a_bytes <= y || y + b_bytes <= x;
}

}  // namespace

void apply(const float* x, float* out, size_t n, const float* table,
           float inv_scale, int32_t zero_point, int32_t quant_min,
           int32_t quant_max) {
  // The four-lane loop (elementwise.h) rounds v by the shifter first and
  // clamps the sum's bits to the codes' range, where this loop clamps v to
  // [-512, 512], rounds and clamps the code: with the range inside
  // [-512, 512] both give clamp(rne(v), quant_min - zero_point, quant_max -
  // zero_point) for every v -- NaN's 0 included (elementwise.cpp says why).
  // The shim's conditions keep the range within [-255, 255]. The loop's table
  // base also needs quant_min <= zero_point <= quant_max (it adds zero_point -
  // quant_min unsigned: on a 64-bit host a negative difference is off by
  // 2^34), which the shim's conditions hold and a direct
  // call need not.
  if (elementwise::enabled() && quant_min <= zero_point &&
      zero_point <= quant_max && quant_min - zero_point >= -512 &&
      quant_max - zero_point <= 512 &&
      same_or_apart(x, out, n * sizeof(float)) &&
      apart(table, (size_t)(quant_max - quant_min + 1) * sizeof(float), out,
            n * sizeof(float))) {
    elementwise::quant_lut_f32(x, out, n, table, inv_scale, zero_point,
                               quant_min, quant_max);
    return;
  }
  float inv = inv_scale;
  float lo = -512.0f;
  float hi = 512.0f;
  // 1.5 * 2^23: for |b| <= 2^22, (b + magic) lands where the float spacing is
  // 1, so the add rounds b half to even and the subtract is exact.
  float magic = 12582912.0f;
  float zero = 0.0f;
  QUANT_LUT_PIN(inv);
  QUANT_LUT_PIN(lo);
  QUANT_LUT_PIN(hi);
  QUANT_LUT_PIN(magic);
  QUANT_LUT_PIN(zero);
  // clamp(r + zp, qmin, qmax) - qmin == clamp(r, qmin - zp, qmax - zp) +
  // (zp - qmin), and zp - qmin indexes the table itself.
  const int32_t low = quant_min - zero_point;
  const int32_t high = quant_max - zero_point;
  const word_t* entries =
      reinterpret_cast<const word_t*>(table) + (zero_point - quant_min);
  word_t* dst = reinterpret_cast<word_t*>(out);
  for (size_t i = 0; i < n; ++i) {
    // NaN: 0, the zero point's code (quant_rules.h); infinities: the rails.
    float b = quant_scaled(x[i] * inv, zero);
    b = b > lo ? b : lo;
    b = b < hi ? b : hi;
    const float r = (b + magic) - magic;
    int32_t c = (int32_t)r;
    c = c < low ? low : c;
    c = c > high ? high : c;
    dst[i] = entries[c];
  }
}

void shim(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Seven arguments plus the returned alias of out.
  if (st.size() != 8) {
    refuse(ctx, Error::InvalidProgram);
    return;
  }
  // Types first: an EValue accessor aborts on the wrong tag.
  if (!st[0]->isTensor() || !st[1]->isTensor() || !st[2]->isDouble() ||
      !st[3]->isInt() || !st[4]->isInt() || !st[5]->isInt() ||
      !st[6]->isTensor()) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const Tensor& x = st[0]->toTensor();
  const Tensor& table = st[1]->toTensor();
  const double inv_scale = st[2]->toDouble();
  const int64_t zero_point = st[3]->toInt();
  const int64_t quant_min = st[4]->toInt();
  const int64_t quant_max = st[5]->toInt();
  Tensor& out = st[6]->toTensor();
  const float inv = (float)inv_scale;
  const bool ok =
      x.scalar_type() == ScalarType::Float &&
      table.scalar_type() == ScalarType::Float &&
      out.scalar_type() == ScalarType::Float &&
      -128 <= quant_min && quant_min < quant_max && quant_max <= 255 &&
      quant_max - quant_min + 1 <= 256 && quant_min <= zero_point &&
      zero_point <= quant_max && table.dim() == 1 &&
      table.numel() == quant_max - quant_min + 1 && dense(table) &&
      std::isfinite(inv_scale) && inv_scale > 0.0 &&
      (double)inv == inv_scale && dense(x) && dense(out);
  if (!ok) {
    printf("  [quant_lut] refused: dtypes %d/%d/%d, range [%lld, %lld] zero "
           "point %lld, table %d-D of %d, inv_scale %g\n",
           (int)x.scalar_type(), (int)table.scalar_type(),
           (int)out.scalar_type(), (long long)quant_min, (long long)quant_max,
           (long long)zero_point, (int)table.dim(), (int)table.numel(),
           inv_scale);
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  if (executorch::runtime::resize_tensor(out, x.sizes()) != Error::Ok ||
      !dense(out)) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  apply(x.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
        (size_t)x.numel(), table.const_data_ptr<float>(), inv,
        (int32_t)zero_point, (int32_t)quant_min, (int32_t)quant_max);
  ++g_calls;
  g_elements += (uint64_t)x.numel();
}

void apply_codes(const int8_t* x, int8_t* out, size_t n, const int8_t* table,
                 int32_t quant_min, int32_t quant_max) {
  // The whole int8 range clamps nothing: four codes a word (elementwise.h).
  // Its 256-entry table apart from out, as apply's.
  if (elementwise::enabled() && quant_min == -128 && quant_max == 127 &&
      same_or_apart(x, out, n) && apart(table, 256, out, n)) {
    elementwise::quant_lut_codes_i8(x, out, n, table);
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    int32_t c = x[i];
    c = c < quant_min ? quant_min : c;
    c = c > quant_max ? quant_max : c;
    out[i] = table[c - quant_min];
  }
}

void codes_shim(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Four arguments plus the returned alias of out.
  if (st.size() != 6) {
    ++g_codes_refusals;
    ctx.fail(Error::InvalidProgram);
    return;
  }
  if (!st[0]->isTensor() || !st[1]->isTensor() || !st[2]->isInt() ||
      !st[3]->isInt() || !st[4]->isTensor()) {
    ++g_codes_refusals;
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const Tensor& x = st[0]->toTensor();
  const Tensor& table = st[1]->toTensor();
  const int64_t quant_min = st[2]->toInt();
  const int64_t quant_max = st[3]->toInt();
  Tensor& out = st[4]->toTensor();
  const bool ok = x.scalar_type() == ScalarType::Char &&
      table.scalar_type() == ScalarType::Char &&
      out.scalar_type() == ScalarType::Char && -128 <= quant_min &&
      quant_min < quant_max && quant_max <= 127 && table.dim() == 1 &&
      table.numel() == quant_max - quant_min + 1 && dense(table) &&
      dense(x) && dense(out);
  if (!ok || executorch::runtime::resize_tensor(out, x.sizes()) != Error::Ok ||
      !dense(out)) {
    printf("  [quant_lut_codes] refused: dtypes %d/%d/%d, range [%lld, %lld], "
           "table %d-D of %d\n", (int)x.scalar_type(),
           (int)table.scalar_type(), (int)out.scalar_type(),
           (long long)quant_min, (long long)quant_max, (int)table.dim(),
           (int)table.numel());
    ++g_codes_refusals;
    ctx.fail(Error::InvalidArgument);
    return;
  }
  apply_codes(x.const_data_ptr<int8_t>(), out.mutable_data_ptr<int8_t>(),
              (size_t)x.numel(), table.const_data_ptr<int8_t>(),
              (int32_t)quant_min, (int32_t)quant_max);
  ++g_codes_calls;
  g_codes_elements += (uint64_t)x.numel();
}

Stats codes_stats() {
  return {g_codes_calls, g_codes_elements, g_codes_refusals};
}

Stats stats() {
  return {g_calls, g_elements, g_refusals};
}

void restore_stats(const Stats& saved) {
  g_calls = saved.calls;
  g_elements = saved.elements;
  g_refusals = saved.refusals;
}

void report_dispatch() {
  if (g_calls != 0 || g_refusals != 0) {
    printf("quant_lut dispatch: calls %llu, elements %llu, refusals %llu\n",
           (unsigned long long)g_calls, (unsigned long long)g_elements,
           (unsigned long long)g_refusals);
  }
  if (g_codes_calls != 0 || g_codes_refusals != 0) {
    printf("quant_lut_codes dispatch: calls %llu, elements %llu, refusals "
           "%llu\n", (unsigned long long)g_codes_calls,
           (unsigned long long)g_codes_elements,
           (unsigned long long)g_codes_refusals);
  }
}

}  // namespace quant_lut
}  // namespace espressif
