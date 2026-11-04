/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Exact fast paths for quantize_per_tensor.out, dequantize_per_tensor.out and
// avg_pool2d.out (qdq_pool.h). Built -O2 -ffp-contract=off on the device (the
// backend's CMakeLists.txt) and on the host, so each float operation below is
// one IEEE operation, as in the portable kernels.
//
// quantize. ExecuTorch 1.6's scalar kernel (op_quantize.cpp,
// quantize_val_with_inv_scale) computes, per element,
//
//   s = fl(inv * x),            inv = fl(1 / float(scale))
//   q = isnan(s) ? zp : zp + nearbyint(s)        in float64, exact here
//   q = min(qmax, max(qmin, q))
//
// and the loop below computes the same code as
//
//   b = fl(x * inv); b = isnan(b) ? 0 : b
//   b = min(max(b, L), H)       L = qmin - zp, H = qmax - zp, exact floats
//   q = rne(b) + zp             rne by the 1.5 * 2^23 shifter
//
// Why they agree. The products are the same IEEE multiply (commutative; a NaN
// product's payload is never used). rne is monotone and fixes integers, so
// clamping b to the integers [L, H] before rounding equals clamping rne(b)
// after it, and zp + clamp(r, L, H) = clamp(zp + r, qmin, qmax): the portable
// kernel's clamp. A NaN takes code zp both ways (b = 0, and L <= 0 <= H); an
// infinity takes its rail (it clamps to L or H). The shifter: for |b| <= 2^22,
// b + 12582912 lands in [2^23, 2^24], where floats are the integers, so the
// add rounds b half to even, exactly as nearbyint does in the default
// rounding mode, and the sum's bits are 0x4B400000 + rne(b). A subnormal b
// rounds to 0 either way, flushed or not. The conditions: a 1- or 2-byte code
// type (|qmin|, |qmax| <= 65535 by the portable kernel's own bound check) and
// qmin <= zp <= qmax, so |L|, |H| < 2^17. ExecuTorch's aarch64 NEON kernel
// (the host's, for 8-bit codes) saturates through int16 and agrees with the
// scalar one exactly when zp is that close to the range, so the host's
// checks hold the loop to both.
//
// dequantize. The portable kernel's expression, (code - int32(zp)) *
// float(scale), with float(scale) hoisted: one int-to-float conversion (exact,
// |code - zp| < 2^17) and one multiply.
//
// avg_pool2d. The portable kernel (kernel_reduction_then_map_2d) seeds each
// window's sum with its first in-bound element and adds every next one as
// in_val + accum, row by row, then divides by float(count) or
// float(divisor_override). Here the same additions run in the same order per
// window, without its per-element int64 index arithmetic; a pool whose every
// window is a whole plane (the global pools CNNs end with) sums four planes
// at a time, each in its own order. Float addition is exactly commutative
// except for which NaN's payload survives a NaN + NaN, and compilers order
// commutative operands as they like, so a call that produces any NaN output is
// redone by the portable kernel: no NaN, no NaN operand anywhere in a sum.

#include "qdq_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include "Esp32KernelVariant.h"
#include "quant_rules.h"

#ifndef ESP32_GENERATED_FAST_QUANTIZE
#define ESP32_GENERATED_FAST_QUANTIZE 0
#endif
#ifndef ESP32_GENERATED_FAST_DEQUANTIZE
#define ESP32_GENERATED_FAST_DEQUANTIZE 0
#endif
#ifndef ESP32_GENERATED_FAST_AVG_POOL2D
#define ESP32_GENERATED_FAST_AVG_POOL2D 0
#endif

// A program that names none of the three operators compiles nothing here (the
// story's): its firmware is what it was without this file.
#if ESP32_GENERATED_FAST_QUANTIZE || ESP32_GENERATED_FAST_DEQUANTIZE || \
    ESP32_GENERATED_FAST_AVG_POOL2D

using executorch::aten::ArrayRef;
using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

// The portable kernels: every input outside a fast path goes to them.
namespace torch {
namespace executor {
namespace native {
Tensor& quantize_per_tensor_out(KernelRuntimeContext&, const Tensor&, double,
                                int64_t, int64_t, int64_t, ScalarType, Tensor&);
Tensor& dequantize_per_tensor_out(KernelRuntimeContext&, const Tensor&, double,
                                  int64_t, int64_t, int64_t, ScalarType,
                                  std::optional<ScalarType>, Tensor&);
Tensor& avg_pool2d_out(KernelRuntimeContext&, const Tensor&, IntArrayRef,
                       IntArrayRef, IntArrayRef, bool, bool,
                       std::optional<int64_t>, Tensor&);
}  // namespace native
}  // namespace executor
}  // namespace torch

namespace espressif {
namespace qdq_pool {
namespace {

// GCC re-materialises float constants inside loops (l32r + wfr per use); an
// empty asm keeps each in its register. QDQ_STAGE4 ends a step of the four
// lanes: GCC otherwise pulls each lane's next instruction up against the
// one whose result it waits for. No-ops off Xtensa.
#if defined(__XTENSA__)
#define QDQ_PIN(f) __asm__("" : "+f"(f))
#define QDQ_STAGE4(a, b, c, d) \
  __asm__("" : "+f"(a), "+f"(b), "+f"(c), "+f"(d))
#else
#define QDQ_PIN(f) ((void)0)
#define QDQ_STAGE4(a, b, c, d) ((void)0)
#endif

// quant_scaled (quant_rules.h) on four lanes: NaN -> 0, all else unchanged.
// GCC accepts only b0 as an asm clobber, so each lane's compare and select
// stay a pair; the stage before them keeps them clear of the multiplies.
inline __attribute__((always_inline)) void nan_to_zero4(float& a, float& b,
                                                        float& c, float& d,
                                                        float zero) {
  a = quant_scaled(a, zero);
  b = quant_scaled(b, zero);
  c = quant_scaled(c, zero);
  d = quant_scaled(d, zero);
}

// A float's bits: rfr on the LX7 (GCC otherwise moves them through the stack).
inline __attribute__((always_inline)) int32_t bits_of(float value) {
#if defined(__XTENSA__)
  int32_t bits;
  __asm__("rfr %0, %1" : "=a"(bits) : "f"(value));
  return bits;
#else
  const union {
    float value;
    int32_t bits;
  } pun = {value};
  return pun.bits;
#endif
}

Counters g_counters = {};
bool g_enabled = true;

// The quantize loop's per-call constants.
struct QuantizeConstants {
  float inv, lo, hi, magic, zero;
  int32_t bias;  // zp - bits(1.5 * 2^23): the shifted sum's bits to the code
};

// One code (the header comment's derivation).
inline __attribute__((always_inline)) int32_t code_of(float x,
                                                      const QuantizeConstants& k) {
  float b = quant_scaled(x * k.inv, k.zero);
  b = b > k.lo ? b : k.lo;
  b = b < k.hi ? b : k.hi;
  return bits_of(b + k.magic) + k.bias;
}

// Four lanes written out step by step, each step closed by QDQ_STAGE4, so the
// independent chains interleave: per four codes 4 lsi, 4 mul.s, 4 NaN selects,
// 8 clamps (16 instructions), 4 add.s, 4 rfr, 4 add, 4 stores and the loop's 4.
template <typename T>
void quantize_loop(const float* __restrict__ x, T* __restrict__ out, size_t n,
                   float inv, int32_t zero_point, int32_t quant_min,
                   int32_t quant_max) {
  QuantizeConstants k;
  k.inv = inv;
  k.lo = static_cast<float>(quant_min - zero_point);
  k.hi = static_cast<float>(quant_max - zero_point);
  k.magic = 12582912.0f;
  k.zero = 0.0f;
  k.bias = zero_point - 0x4B400000;
  QDQ_PIN(k.inv);
  QDQ_PIN(k.lo);
  QDQ_PIN(k.hi);
  QDQ_PIN(k.magic);
  QDQ_PIN(k.zero);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float b0 = x[i + 0];
    float b1 = x[i + 1];
    float b2 = x[i + 2];
    float b3 = x[i + 3];
    QDQ_STAGE4(b0, b1, b2, b3);
    b0 = b0 * k.inv;
    b1 = b1 * k.inv;
    b2 = b2 * k.inv;
    b3 = b3 * k.inv;
    QDQ_STAGE4(b0, b1, b2, b3);
    nan_to_zero4(b0, b1, b2, b3, k.zero);
    b0 = b0 > k.lo ? b0 : k.lo;
    b1 = b1 > k.lo ? b1 : k.lo;
    b2 = b2 > k.lo ? b2 : k.lo;
    b3 = b3 > k.lo ? b3 : k.lo;
    b0 = b0 < k.hi ? b0 : k.hi;
    b1 = b1 < k.hi ? b1 : k.hi;
    b2 = b2 < k.hi ? b2 : k.hi;
    b3 = b3 < k.hi ? b3 : k.hi;
    QDQ_STAGE4(b0, b1, b2, b3);
    b0 = b0 + k.magic;
    b1 = b1 + k.magic;
    b2 = b2 + k.magic;
    b3 = b3 + k.magic;
    QDQ_STAGE4(b0, b1, b2, b3);
    out[i + 0] = static_cast<T>(bits_of(b0) + k.bias);
    out[i + 1] = static_cast<T>(bits_of(b1) + k.bias);
    out[i + 2] = static_cast<T>(bits_of(b2) + k.bias);
    out[i + 3] = static_cast<T>(bits_of(b3) + k.bias);
  }
  for (; i < n; ++i) {
    out[i] = static_cast<T>(code_of(x[i], k));
  }
}

// One-byte codes from a table of all 256 results, each computed by the
// loop's own expression and copied as a word: no FP instruction touches the
// copies, so the bits (a NaN's payload included) are the loop's. The table
// lives in the last 256 outputs, which the loop computes directly once the
// lookups are done: no stack, nothing shared between calls. Worth its 512
// computed values from this many codes on.
constexpr size_t kDequantizeTableFrom = 2048;
constexpr size_t kTable = 256;

typedef uint32_t __attribute__((__may_alias__)) word_t;

template <typename T>
__attribute__((noinline)) void dequantize_table(const T* __restrict__ codes,
                                                float* __restrict__ out,
                                                size_t n, int32_t zero_point,
                                                float scale) {
  static_assert(sizeof(T) == 1, "one-byte codes");
  QDQ_PIN(scale);
  float* const table = out + (n - kTable);
  for (size_t byte = 0; byte < kTable; ++byte) {
    const T code = static_cast<T>(byte);
    table[byte] = static_cast<float>(static_cast<int32_t>(code) - zero_point) * scale;
  }
  const word_t* entries = reinterpret_cast<const word_t*>(table);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(codes);
  word_t* dst = reinterpret_cast<word_t*>(out);
  const size_t looked_up = n - kTable;
  size_t i = 0;
  for (; i + 4 <= looked_up; i += 4) {
    const word_t w0 = entries[bytes[i + 0]];
    const word_t w1 = entries[bytes[i + 1]];
    const word_t w2 = entries[bytes[i + 2]];
    const word_t w3 = entries[bytes[i + 3]];
    dst[i + 0] = w0;
    dst[i + 1] = w1;
    dst[i + 2] = w2;
    dst[i + 3] = w3;
  }
  for (; i < looked_up; ++i) {
    dst[i] = entries[bytes[i]];
  }
  for (; i < n; ++i) {  // over the table: the loop's expression itself
    out[i] = static_cast<float>(static_cast<int32_t>(codes[i]) - zero_point) * scale;
  }
}

template <typename T>
void dequantize_loop(const T* __restrict__ codes, float* __restrict__ out,
                     size_t n, int32_t zero_point, float scale) {
  if constexpr (sizeof(T) == 1) {
    if (n >= kDequantizeTableFrom) {
      dequantize_table(codes, out, n, zero_point, scale);
      return;
    }
  }
  QDQ_PIN(scale);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float f0 = static_cast<float>(static_cast<int32_t>(codes[i + 0]) - zero_point);
    const float f1 = static_cast<float>(static_cast<int32_t>(codes[i + 1]) - zero_point);
    const float f2 = static_cast<float>(static_cast<int32_t>(codes[i + 2]) - zero_point);
    const float f3 = static_cast<float>(static_cast<int32_t>(codes[i + 3]) - zero_point);
    out[i + 0] = f0 * scale;
    out[i + 1] = f1 * scale;
    out[i + 2] = f2 * scale;
    out[i + 3] = f3 * scale;
  }
  for (; i < n; ++i) {
    out[i] = static_cast<float>(static_cast<int32_t>(codes[i]) - zero_point) * scale;
  }
}

// The code types the fast loops take: 1 or 2 bytes, whose bounds keep the
// quantize's L and H within the shifter's 2^22.
struct CodeType {
  size_t bytes;
  bool is_signed;
  int64_t lowest, highest;  // the portable kernel's bounds check
};

bool code_type(ScalarType type, CodeType* code) {
  switch (type) {
    case ScalarType::Char:
      *code = {1, true, -128, 127};
      return true;
    case ScalarType::Byte:
      *code = {1, false, 0, 255};
      return true;
    case ScalarType::Short:
      *code = {2, true, -32768, 32767};
      return true;
    case ScalarType::Bits16:
    case ScalarType::UInt16:
      *code = {2, false, 0, 65535};
      return true;
    default:
      return false;
  }
}

// The zero point inside the code range: a NaN's code is then zp unclamped,
// and every difference code - zp is far inside float32's exact integers.
bool zero_point_in_range(int64_t zero_point, int64_t quant_min,
                         int64_t quant_max) {
  return quant_min <= zero_point && zero_point <= quant_max;
}

// Whether two tensors' bytes are apart. The fast loops read and write in an
// order of their own -- four lanes at a time, a table in the output's last
// 256 floats, a NaN call redone on the input -- and over shared bytes only
// the portable kernel's own order gives its bytes (a method may be handed an
// output that overlaps an input: Method::set_output_data_ptr). Checked after
// the output's resize, as its extent is then final. An extent is numel()
// times the element size the shim has already checked, computed in line:
// TensorImpl::nbytes() is an out-of-line call that switches on the dtype,
// and two of them came to about 55 LX7 instructions on every call. An empty
// extent counts as apart unless it starts inside the other, which at worst
// sends a call with nothing to compute to the portable kernel.
inline __attribute__((always_inline)) bool disjoint(const Tensor& a,
                                                    size_t a_element,
                                                    const Tensor& b,
                                                    size_t b_element) {
  const uintptr_t a0 = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const uintptr_t b0 = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  return a0 + static_cast<size_t>(a.numel()) * a_element <= b0 ||
      b0 + static_cast<size_t>(b.numel()) * b_element <= a0;
}

// The default dim order, read without the portable helper's error log.
bool default_order(const Tensor& tensor) {
  const auto order = tensor.dim_order();
  for (size_t dim = 0; dim < order.size(); ++dim) {
    if (order[dim] != dim) {
      return false;
    }
  }
  return true;
}

#if ESP32_GENERATED_FAST_AVG_POOL2D

// The parameter checks of check_avg_pool2d_args, answered without its logs:
// a call they reject goes to the portable kernel, which logs and fails. Only
// float32 in the default dim order, every axis non-empty.
bool pool_args(const Tensor& in, IntArrayRef kernel_size, IntArrayRef stride,
               IntArrayRef padding, const std::optional<int64_t>& divisor,
               const Tensor& out) {
  if (in.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float ||
      (in.dim() != 3 && in.dim() != 4) || !default_order(in) ||
      !default_order(out)) {
    return false;
  }
  for (ssize_t dim = 0; dim < in.dim(); ++dim) {
    if (in.size(dim) <= 0) {
      return false;
    }
  }
  if (kernel_size.size() != 1 && kernel_size.size() != 2) {
    return false;
  }
  for (size_t i = 0; i < kernel_size.size(); ++i) {
    if (kernel_size[i] < 1) {
      return false;
    }
  }
  if (stride.size() > 2) {
    return false;
  }
  for (size_t i = 0; i < stride.size(); ++i) {
    if (stride[i] < 1) {
      return false;
    }
  }
  if (padding.size() != 1 && padding.size() != 2) {
    return false;
  }
  for (size_t i = 0; i < padding.size(); ++i) {
    if (padding[i] < 0 ||
        padding[i] > torch::executor::val_at(kernel_size, i) / 2) {
      return false;
    }
  }
  return !divisor.has_value() || divisor.value() != 0;
}

// One window of one plane, in the portable kernel's order. Not inlined, nor
// is global_pool: inside the shim the sums would share its registers (and
// its calls' FP spills) with everything else live there.
__attribute__((noinline)) float window_sum(const float* plane, size_t width,
                                           size_t y0, size_t y1, size_t x0,
                                           size_t x1) {
  const float* row = plane + y0 * width;
  float accum = row[x0];
  for (size_t x = x0 + 1; x < x1; ++x) {
    accum = row[x] + accum;
  }
  for (size_t y = y0 + 1; y < y1; ++y) {
    row += width;
    for (size_t x = x0; x < x1; ++x) {
      accum = row[x] + accum;
    }
  }
  return accum;
}

// Every window a whole plane of `count` elements: four planes at a time, each
// summed from its first element on. Returns whether any output is NaN.
__attribute__((noinline)) bool global_pool(const float* in, float* out,
                                           size_t planes, size_t count,
                                           float divisor) {
  bool nan = false;
  size_t c = 0;
  for (; c + 4 <= planes; c += 4) {
    const float* p0 = in + c * count;
    const float* p1 = p0 + count;
    const float* p2 = p1 + count;
    const float* p3 = p2 + count;
    float s0 = *p0;
    float s1 = *p1;
    float s2 = *p2;
    float s3 = *p3;
    for (size_t i = 1; i < count; ++i) {
      s0 = *++p0 + s0;
      s1 = *++p1 + s1;
      s2 = *++p2 + s2;
      s3 = *++p3 + s3;
    }
    const float o0 = s0 / divisor;
    const float o1 = s1 / divisor;
    const float o2 = s2 / divisor;
    const float o3 = s3 / divisor;
    out[c + 0] = o0;
    out[c + 1] = o1;
    out[c + 2] = o2;
    out[c + 3] = o3;
    nan |= (o0 != o0) | (o1 != o1) | (o2 != o2) | (o3 != o3);
  }
  for (; c < planes; ++c) {
    const float* p = in + c * count;
    float s = p[0];
    for (size_t i = 1; i < count; ++i) {
      s = p[i] + s;
    }
    const float o = s / divisor;
    out[c] = o;
    nan |= o != o;
  }
  return nan;
}

#endif  // ESP32_GENERATED_FAST_AVG_POOL2D

}  // namespace

void quantize_f32(const float* x, void* out, size_t n, size_t bytes,
                  bool is_signed, float inv, int32_t zero_point,
                  int32_t quant_min, int32_t quant_max) {
  if (bytes == 1 && is_signed) {
    quantize_loop(x, static_cast<int8_t*>(out), n, inv, zero_point, quant_min,
                  quant_max);
  } else if (bytes == 1) {
    quantize_loop(x, static_cast<uint8_t*>(out), n, inv, zero_point, quant_min,
                  quant_max);
  } else if (is_signed) {
    quantize_loop(x, static_cast<int16_t*>(out), n, inv, zero_point, quant_min,
                  quant_max);
  } else {
    quantize_loop(x, static_cast<uint16_t*>(out), n, inv, zero_point, quant_min,
                  quant_max);
  }
}

void dequantize_f32(const void* codes, size_t bytes, bool is_signed, float* out,
                    size_t n, int32_t zero_point, float scale) {
  if (bytes == 1 && is_signed) {
    dequantize_loop(static_cast<const int8_t*>(codes), out, n, zero_point, scale);
  } else if (bytes == 1) {
    dequantize_loop(static_cast<const uint8_t*>(codes), out, n, zero_point, scale);
  } else if (is_signed) {
    dequantize_loop(static_cast<const int16_t*>(codes), out, n, zero_point, scale);
  } else {
    dequantize_loop(static_cast<const uint16_t*>(codes), out, n, zero_point, scale);
  }
}

#if ESP32_GENERATED_FAST_QUANTIZE
// quantized_decomposed::quantize_per_tensor.out(Tensor input, float scale,
//     int zero_point, int quant_min, int quant_max, ScalarType dtype, *,
//     Tensor(a!) out): seven arguments plus the returned alias of out.
void shim_quantize_per_tensor_out(KernelRuntimeContext& ctx, Span<EValue*> stack) {
  if (stack.size() != 8) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& input = stack[0]->toTensor();
  const double scale = stack[1]->toDouble();
  const int64_t zero_point = stack[2]->toInt();
  const int64_t quant_min = stack[3]->toInt();
  const int64_t quant_max = stack[4]->toInt();
  const ScalarType dtype = stack[5]->toScalarType();
  Tensor& out = stack[6]->toTensor();
  CodeType code;
  // The portable kernel resizes first, then checks its arguments (aborting
  // on a bad one), then converts: a call any check would stop runs there.
  const bool fast = g_enabled && input.scalar_type() == ScalarType::Float &&
      code_type(dtype, &code) && out.scalar_type() == dtype &&
      quant_min >= code.lowest && quant_max <= code.highest &&
      zero_point_in_range(zero_point, quant_min, quant_max) &&
      executorch::runtime::resize_tensor(out, input.sizes()) == Error::Ok &&
      disjoint(input, sizeof(float), out, code.bytes);
  if (!fast) {
    ++g_counters.quantize_fallback;
    torch::executor::native::quantize_per_tensor_out(
        ctx, input, scale, zero_point, quant_min, quant_max, dtype, out);
    return;
  }
  // The portable kernel's reciprocal, the same IEEE operations (GCC hoists
  // it out of that kernel's loop as well).
  const float inv = 1.0f / static_cast<float>(scale);
  quantize_f32(input.const_data_ptr<float>(), out.mutable_data_ptr(),
               static_cast<size_t>(input.numel()), code.bytes, code.is_signed,
               inv, static_cast<int32_t>(zero_point),
               static_cast<int32_t>(quant_min), static_cast<int32_t>(quant_max));
  ++g_counters.quantize_fast;
}
#endif  // ESP32_GENERATED_FAST_QUANTIZE

#if ESP32_GENERATED_FAST_DEQUANTIZE
// quantized_decomposed::dequantize_per_tensor.out(Tensor input, float scale,
//     int zero_point, int quant_min, int quant_max, ScalarType dtype, *,
//     ScalarType? out_dtype=None, Tensor(a!) out): eight plus the alias.
void shim_dequantize_per_tensor_out(KernelRuntimeContext& ctx,
                                    Span<EValue*> stack) {
  if (stack.size() != 9) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& input = stack[0]->toTensor();
  const double scale = stack[1]->toDouble();
  const int64_t zero_point = stack[2]->toInt();
  const int64_t quant_min = stack[3]->toInt();
  const int64_t quant_max = stack[4]->toInt();
  const ScalarType dtype = stack[5]->toScalarType();
  const std::optional<ScalarType> out_dtype =
      stack[6]->toOptional<ScalarType>();
  Tensor& out = stack[7]->toTensor();
  CodeType code;
  // The portable kernel's checks (dtype, out_dtype, quant_min <= quant_max)
  // abort on failure: a call that would fail one runs there.
  const bool fast = g_enabled && code_type(input.scalar_type(), &code) &&
      input.scalar_type() == dtype && out.scalar_type() == ScalarType::Float &&
      (!out_dtype.has_value() || out_dtype.value() == ScalarType::Float) &&
      zero_point_in_range(zero_point, quant_min, quant_max) &&
      executorch::runtime::resize_tensor(out, input.sizes()) == Error::Ok &&
      disjoint(input, code.bytes, out, sizeof(float));
  if (!fast) {
    ++g_counters.dequantize_fallback;
    torch::executor::native::dequantize_per_tensor_out(
        ctx, input, scale, zero_point, quant_min, quant_max, dtype, out_dtype,
        out);
    return;
  }
  dequantize_f32(input.const_data_ptr(), code.bytes, code.is_signed,
                 out.mutable_data_ptr<float>(),
                 static_cast<size_t>(input.numel()),
                 static_cast<int32_t>(zero_point), static_cast<float>(scale));
  ++g_counters.dequantize_fast;
}
#endif  // ESP32_GENERATED_FAST_DEQUANTIZE

#if ESP32_GENERATED_FAST_AVG_POOL2D
// aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[],
//     int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True,
//     int? divisor_override=None, *, Tensor(a!) out): eight plus the alias.
void shim_avg_pool2d_out(KernelRuntimeContext& ctx, Span<EValue*> stack) {
  if (stack.size() != 9) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& in = stack[0]->toTensor();
  const IntArrayRef kernel_size = stack[1]->toIntList();
  const IntArrayRef stride = stack[2]->toIntList();
  const IntArrayRef padding = stack[3]->toIntList();
  const bool ceil_mode = stack[4]->toBool();
  const bool count_include_pad = stack[5]->toBool();
  const std::optional<int64_t> divisor = stack[6]->toOptional<int64_t>();
  Tensor& out = stack[7]->toTensor();
  auto portable = [&] {
    ++g_counters.pool_fallback;
    torch::executor::native::avg_pool2d_out(ctx, in, kernel_size, stride,
                                            padding, ceil_mode,
                                            count_include_pad, divisor, out);
  };
  if (!g_enabled || !pool_args(in, kernel_size, stride, padding, divisor, out)) {
    portable();
    return;
  }
  SizesType sizes[executorch::runtime::kTensorDimensionLimit];
  size_t ndim = 0;
  torch::executor::get_avg_pool2d_out_target_size(
      in, kernel_size, stride, padding, ceil_mode, sizes, &ndim);
  for (size_t dim = 0; dim < ndim; ++dim) {
    if (sizes[dim] <= 0) {  // output_size_is_valid would refuse (and log)
      portable();
      return;
    }
  }
  if (executorch::runtime::resize_tensor(out, {sizes, ndim}) != Error::Ok ||
      !disjoint(in, sizeof(float), out, sizeof(float))) {
    portable();
    return;
  }

  const int64_t height = in.size(in.dim() - 2);
  const int64_t width = in.size(in.dim() - 1);
  const int64_t out_height = sizes[ndim - 2];
  const int64_t out_width = sizes[ndim - 1];
  const size_t planes = static_cast<size_t>(in.numel() / (height * width));
  const int64_t k_h = torch::executor::val_at(kernel_size, 0);
  const int64_t k_w = torch::executor::val_at(kernel_size, 1);
  const int64_t s_h = torch::executor::val_at(stride, 0, k_h);
  const int64_t s_w = torch::executor::val_at(stride, 1, k_w);
  const int64_t p_h = torch::executor::val_at(padding, 0, 0);
  const int64_t p_w = torch::executor::val_at(padding, 1, 0);
  const float* in_data = in.const_data_ptr<float>();
  float* out_data = out.mutable_data_ptr<float>();

  // One window's bounds, clipped to the plane, and its count, exactly as
  // kernel_reduction_then_map_2d computes them (in int64).
  struct Span1d {
    int64_t begin, end, padded;  // [begin, end) in the plane; padded extent
  };
  auto window = [](int64_t o, int64_t s, int64_t p, int64_t k, int64_t size) {
    const int64_t begin = o * s - p;
    const int64_t end = std::min(begin + k, size + p);
    return Span1d{std::max(begin, int64_t{0}), std::min(end, size), end - begin};
  };
  auto count_of = [&](const Span1d& rows, const Span1d& cols) {
    return count_include_pad ? rows.padded * cols.padded
                             : (rows.end - rows.begin) * (cols.end - cols.begin);
  };

  bool nan = false;
  if (out_height == 1 && out_width == 1) {
    const Span1d rows = window(0, s_h, p_h, k_h, height);
    const Span1d cols = window(0, s_w, p_w, k_w, width);
    if (rows.begin == 0 && rows.end == height && cols.begin == 0 &&
        cols.end == width) {
      const int64_t count = divisor.has_value() ? divisor.value()
                                                : count_of(rows, cols);
      nan = global_pool(in_data, out_data, planes,
                        static_cast<size_t>(height * width),
                        static_cast<float>(count));
      if (nan) {
        portable();
        return;
      }
      ++g_counters.pool_global;
      return;
    }
  }
  const size_t plane_in = static_cast<size_t>(height * width);
  const size_t plane_out = static_cast<size_t>(out_height * out_width);
  for (size_t c = 0; c < planes; ++c) {
    const float* plane = in_data + c * plane_in;
    float* dst = out_data + c * plane_out;
    for (int64_t oy = 0; oy < out_height; ++oy) {
      const Span1d rows = window(oy, s_h, p_h, k_h, height);
      for (int64_t ox = 0; ox < out_width; ++ox) {
        const Span1d cols = window(ox, s_w, p_w, k_w, width);
        if (rows.begin >= rows.end || cols.begin >= cols.end) {
          continue;  // the portable kernel writes nothing here either
        }
        const int64_t count = divisor.has_value() ? divisor.value()
                                                  : count_of(rows, cols);
        const float o =
            window_sum(plane, static_cast<size_t>(width),
                       static_cast<size_t>(rows.begin),
                       static_cast<size_t>(rows.end),
                       static_cast<size_t>(cols.begin),
                       static_cast<size_t>(cols.end)) /
            static_cast<float>(count);
        dst[oy * out_width + ox] = o;
        nan |= o != o;
      }
    }
  }
  if (nan) {
    portable();
    return;
  }
  ++g_counters.pool_windowed;
}
#endif  // ESP32_GENERATED_FAST_AVG_POOL2D

Counters& counters() {
  return g_counters;
}

void set_enabled(bool enabled) {
  g_enabled = enabled;
}

bool enabled() {
  return g_enabled;
}

void report_dispatch() {
  const Counters& c = g_counters;
  if (c.quantize_fast != 0 || c.quantize_fallback != 0) {
    printf("quantize_per_tensor.out dispatch: fp32 %llu, fallback %llu\n",
           (unsigned long long)c.quantize_fast,
           (unsigned long long)c.quantize_fallback);
  }
  if (c.dequantize_fast != 0 || c.dequantize_fallback != 0) {
    printf("dequantize_per_tensor.out dispatch: to-fp32 %llu, fallback %llu\n",
           (unsigned long long)c.dequantize_fast,
           (unsigned long long)c.dequantize_fallback);
  }
  if (c.pool_global != 0 || c.pool_windowed != 0 || c.pool_fallback != 0) {
    printf("avg_pool2d.out dispatch: global %llu, windowed %llu, fallback %llu\n",
           (unsigned long long)c.pool_global,
           (unsigned long long)c.pool_windowed,
           (unsigned long long)c.pool_fallback);
  }
}

}  // namespace qdq_pool
}  // namespace espressif

#endif  // any of the three
