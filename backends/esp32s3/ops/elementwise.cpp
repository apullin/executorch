/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The element-wise fast loops (elementwise.h). Built -O2 -ffp-contract=off on
// the device (the backend's CMakeLists.txt) and -ffp-contract=off on the
// host, so each float operation below is one IEEE operation, as in the
// kernels these replace. elementwise.lf links this file after every other
// flash object, as qdq_pool.lf does qdq_pool.cpp; the story firmware calls
// none of it (its shims compile only where the variant header asks, and the
// table loops are reached only through quant_lut.cpp's, which the story's
// programs never run), so the linker keeps none of it there.

#include "elementwise.h"

#include <cstdio>
#include <cstring>
#include <optional>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include "Esp32KernelVariant.h"
#include "quant_rules.h"

#ifndef ESP32_GENERATED_EW_MUL
#define ESP32_GENERATED_EW_MUL 0
#endif
#ifndef ESP32_GENERATED_EW_CONSTANT_PAD
#define ESP32_GENERATED_EW_CONSTANT_PAD 0
#endif

#if ESP32_GENERATED_EW_CONSTANT_PAD
#include <executorch/kernels/portable/cpu/scalar_utils.h>

// The portable kernel: every call outside the row copy goes to it.
namespace torch {
namespace executor {
namespace native {
executorch::aten::Tensor& constant_pad_nd_out(
    executorch::runtime::KernelRuntimeContext&, const executorch::aten::Tensor&,
    executorch::aten::IntArrayRef, const executorch::aten::Scalar&,
    executorch::aten::Tensor&);
}  // namespace native
}  // namespace executor
}  // namespace torch
#endif

using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace espressif {
namespace elementwise {
namespace {

// GCC re-materialises float constants inside loops (l32r + wfr per use); an
// empty asm keeps each in its register. EW_STAGE4 ends one step of the four
// lanes: GCC otherwise pulls each lane's next instruction up against the one
// whose result it waits for. No-ops off Xtensa.
#if defined(__XTENSA__)
#define EW_PIN(f) __asm__("" : "+f"(f))
#define EW_STAGE4(a, b, c, d) \
  __asm__("" : "+f"(a), "+f"(b), "+f"(c), "+f"(d))
#else
#define EW_PIN(f) ((void)0)
#define EW_STAGE4(a, b, c, d) ((void)0)
#endif

// Table entries and copied outputs move as words: no FP instruction touches
// them, so denormals and NaN payloads survive.
typedef uint32_t __attribute__((__may_alias__)) word_t;

// A float's bits: rfr on the LX7 (GCC otherwise moves them through the stack).
inline __attribute__((always_inline)) uint32_t bits_of(float value) {
#if defined(__XTENSA__)
  uint32_t bits;
  __asm__("rfr %0, %1" : "=a"(bits) : "f"(value));
  return bits;
#else
  const union {
    float value;
    uint32_t bits;
  } pun = {value};
  return pun.bits;
#endif
}

Counters g_counters = {};

// quant_lut's per-call constants. The shifter: for |v| <= 2^22 the bits of
// v + 1.5 * 2^23 are 0x4B400000 + rne(v). Every larger v lands past every
// code's bits on its own side: a sum of 2^24 or more has larger bits, and a
// sum below 2^23 -- or zero, or negative, whose bits are negative as int32 --
// smaller ones. So clamping the sum's bits (signed) to [0x4B400000 + lo,
// 0x4B400000 + hi], lo = quant_min - zero_point and hi = quant_max -
// zero_point, gives 0x4B400000 + clamp(rne(v), lo, hi) for every v but NaN,
// which quant_scaled has made 0 (|lo|, |hi| <= 512, so the codes' bits are
// inside the exact span). That is quant_lut.cpp's code: clamp(rne(v)) over
// [-512, 512] and then the range, for the same reason. The entry of code
// rne(v) + zero_point then sits at base + 4 * bits, base = table + 4 *
// (zero_point - quant_min) - 4 * 0x4B400000, in unsigned pointer arithmetic
// (which wraps on the 32-bit device and on a 64-bit host alike).
struct LutConstants {
  float inv, magic, zero;
  int32_t lo_bits, hi_bits;
  uintptr_t base;
};

inline __attribute__((always_inline)) int32_t lut_bits(float sum,
                                                       const LutConstants& k) {
  int32_t bits = static_cast<int32_t>(bits_of(sum));
  bits = bits < k.lo_bits ? k.lo_bits : bits;
  bits = bits > k.hi_bits ? k.hi_bits : bits;
  return bits;
}

inline __attribute__((always_inline)) word_t lut_entry(int32_t bits,
                                                       uintptr_t base) {
  return *reinterpret_cast<const word_t*>(
      base + (static_cast<uintptr_t>(static_cast<uint32_t>(bits)) << 2));
}

// One element: fl(x * inv), quant_scaled's NaN -> 0, the shifter, the clamp
// on its bits, the entry.
inline __attribute__((always_inline)) word_t lut_one(float x,
                                                     const LutConstants& k) {
  const float b = quant_scaled(x * k.inv, k.zero);
  return lut_entry(lut_bits(b + k.magic, k), k.base);
}

}  // namespace

void mul_channels_f32(const float* scale, const float* x, float* out,
                      size_t outer, size_t inner) {
  for (size_t o = 0; o < outer; ++o) {
    float s = scale[o];
    EW_PIN(s);
    const float* src = x + o * inner;
    float* dst = out + o * inner;
    size_t j = 0;
    // Loads first, then the products, then the stores: out may be x itself,
    // and each output is still its own input times the scale. No EW_STAGE4
    // here: an asm statement costs the loop its zero-overhead LOOP, and
    // GCC's own order (each product two instructions after its load) loses
    // less than the counter and the taken branch would.
    for (; j + 4 <= inner; j += 4) {
      float a = src[j + 0];
      float b = src[j + 1];
      float c = src[j + 2];
      float d = src[j + 3];
      a = s * a;
      b = s * b;
      c = s * c;
      d = s * d;
      dst[j + 0] = a;
      dst[j + 1] = b;
      dst[j + 2] = c;
      dst[j + 3] = d;
    }
    for (; j < inner; ++j) {
      dst[j] = s * src[j];
    }
  }
}

void quant_lut_f32(const float* x, float* out, size_t n, const float* table,
                   float inv, int32_t zero_point, int32_t quant_min,
                   int32_t quant_max) {
  ++g_counters.lut_f32;
  LutConstants k;
  k.inv = inv;
  k.magic = 12582912.0f;  // 1.5 * 2^23
  k.zero = 0.0f;
  k.lo_bits = 0x4B400000 + (quant_min - zero_point);
  k.hi_bits = 0x4B400000 + (quant_max - zero_point);
  k.base = reinterpret_cast<uintptr_t>(table) +
      static_cast<uintptr_t>(static_cast<uint32_t>(zero_point - quant_min)) * 4u -
      static_cast<uintptr_t>(0x4B400000u) * 4u;
  EW_PIN(k.inv);
  EW_PIN(k.magic);
  EW_PIN(k.zero);
  word_t* dst = reinterpret_cast<word_t*>(out);
  size_t i = 0;
  // Four lanes written out step by step, each step closed by EW_STAGE4, so
  // the independent chains interleave: per four elements 4 lsi, 4 mul.s, 4
  // NaN selects, 4 add.s, 4 rfr, 8 integer clamps (max, min), 4 addx4, 4 l32i
  // and 4 s32i. All four lanes are loaded before any is stored, so out may
  // be x itself.
  for (; i + 4 <= n; i += 4) {
    float b0 = x[i + 0];
    float b1 = x[i + 1];
    float b2 = x[i + 2];
    float b3 = x[i + 3];
    EW_STAGE4(b0, b1, b2, b3);
    b0 = b0 * k.inv;
    b1 = b1 * k.inv;
    b2 = b2 * k.inv;
    b3 = b3 * k.inv;
    EW_STAGE4(b0, b1, b2, b3);
    b0 = quant_scaled(b0, k.zero);
    b1 = quant_scaled(b1, k.zero);
    b2 = quant_scaled(b2, k.zero);
    b3 = quant_scaled(b3, k.zero);
    EW_STAGE4(b0, b1, b2, b3);
    b0 = b0 + k.magic;
    b1 = b1 + k.magic;
    b2 = b2 + k.magic;
    b3 = b3 + k.magic;
    EW_STAGE4(b0, b1, b2, b3);
    const int32_t c0 = lut_bits(b0, k);
    const int32_t c1 = lut_bits(b1, k);
    const int32_t c2 = lut_bits(b2, k);
    const int32_t c3 = lut_bits(b3, k);
    const word_t w0 = lut_entry(c0, k.base);
    const word_t w1 = lut_entry(c1, k.base);
    const word_t w2 = lut_entry(c2, k.base);
    const word_t w3 = lut_entry(c3, k.base);
    dst[i + 0] = w0;
    dst[i + 1] = w1;
    dst[i + 2] = w2;
    dst[i + 3] = w3;
  }
  for (; i < n; ++i) {
    dst[i] = lut_one(x[i], k);
  }
}

void quant_lut_codes_i8(const int8_t* x, int8_t* out, size_t n,
                        const int8_t* table) {
  ++g_counters.lut_codes;
  // A code c's entry is table[c + 128]: its byte with the top bit flipped.
  const uint8_t* entries = reinterpret_cast<const uint8_t*>(table);
  const uint8_t* src = reinterpret_cast<const uint8_t*>(x);
  uint8_t* dst = reinterpret_cast<uint8_t*>(out);
  size_t i = 0;
  for (; i < n && (reinterpret_cast<uintptr_t>(src + i) & 3u) != 0; ++i) {
    dst[i] = entries[src[i] ^ 0x80u];
  }
  // Four codes a word: each lane's index, then its entry, then the stores,
  // so no load waits on the instruction before it. The word is read before
  // any of its bytes is written, so out may be x itself. Pointers rather
  // than an index: GCC otherwise rebuilds the word's address every step.
  const word_t* words = reinterpret_cast<const word_t*>(src + i);
  uint8_t* d = dst + i;
  const size_t count = (n - i) / 4;
  for (size_t k = 0; k < count; ++k) {
    const uint32_t w = words[k] ^ 0x80808080u;
    const uint32_t i0 = w & 0xFFu;
    const uint32_t i1 = (w >> 8) & 0xFFu;
    const uint32_t i2 = (w >> 16) & 0xFFu;
    const uint32_t i3 = w >> 24;
    const uint8_t c0 = entries[i0];
    const uint8_t c1 = entries[i1];
    const uint8_t c2 = entries[i2];
    const uint8_t c3 = entries[i3];
    d[0] = c0;
    d[1] = c1;
    d[2] = c2;
    d[3] = c3;
    d += 4;
  }
  i += count * 4;
  for (; i < n; ++i) {
    dst[i] = entries[src[i] ^ 0x80u];
  }
}

namespace {

#if ESP32_GENERATED_EW_MUL || ESP32_GENERATED_EW_CONSTANT_PAD

bool same_shape(const Tensor& a, const Tensor& b) {
  if (a.dim() != b.dim()) {
    return false;
  }
  for (ssize_t dim = 0; dim < a.dim(); ++dim) {
    if (a.size(dim) != b.size(dim)) {
      return false;
    }
  }
  return true;
}

// The default dim order and contiguous strides (size-1 axes may carry any
// stride), as espressif_ops.cpp's dense_tensor.
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

// Whether two fp32 tensors' bytes overlap (numel times the element size:
// Tensor::nbytes() is an out-of-line call).
bool overlap(const Tensor& a, const Tensor& b) {
  const uintptr_t a0 = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const uintptr_t b0 = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  const uintptr_t a1 = a0 + static_cast<size_t>(a.numel()) * sizeof(float);
  const uintptr_t b1 = b0 + static_cast<size_t>(b.numel()) * sizeof(float);
  return a0 < b1 && b0 < a1;
}

#endif  // ESP32_GENERATED_EW_MUL || ESP32_GENERATED_EW_CONSTANT_PAD

#if ESP32_GENERATED_EW_CONSTANT_PAD

// The default dim order and the canonical contiguous stride on every axis,
// size-1 axes included. dense()'s rule (a singleton axis may carry any stride)
// is not enough here: the portable pad kernel copies stride x size elements
// of each axis, so a stride-0 singleton input copies nothing there (an
// input [1] of stride 0, pad (1,1), gives [7,7,9] from it and [7,5,7]
// from a row copy). Any other stride goes to it.
bool canonical(const Tensor& tensor) {
  const auto order = tensor.dim_order();
  for (size_t dim = 0; dim < order.size(); ++dim) {
    if (order[dim] != dim) {
      return false;
    }
  }
  ssize_t expected = 1;
  for (ssize_t dim = tensor.dim(); dim-- > 0;) {
    if (tensor.strides()[dim] != expected) {
      return false;
    }
    expected *= tensor.size(dim);
  }
  return true;
}

// One plane after another: `top` rows of the value, each input row between
// `left` and `right` values, `bottom` rows of the value. T is the element's
// bits (uint8_t or uint32_t), so no FP instruction touches a value or an
// input. memcpy for the rows: IDF builds with -fno-builtin-memcpy, so it is a
// call, which the shortest rows (6 bytes) still repay against the portable
// kernel's recursion per row.
template <typename T>
void pad_planes(const T* in, T* out, size_t planes, size_t height,
                size_t width, size_t top, size_t bottom, size_t left,
                size_t right, T value) {
  const size_t out_width = left + width + right;
  for (size_t plane = 0; plane < planes; ++plane) {
    for (size_t i = 0; i < top * out_width; ++i) {
      *out++ = value;
    }
    for (size_t row = 0; row < height; ++row) {
      for (size_t i = 0; i < left; ++i) {
        *out++ = value;
      }
      memcpy(out, in, width * sizeof(T));
      out += width;
      in += width;
      for (size_t i = 0; i < right; ++i) {
        *out++ = value;
      }
    }
    for (size_t i = 0; i < bottom * out_width; ++i) {
      *out++ = value;
    }
  }
}

#endif  // ESP32_GENERATED_EW_CONSTANT_PAD

#if ESP32_GENERATED_EW_MUL

// Whether `small`, right-aligned against `big`, equals it on the leading
// axes and is 1 on the trailing ones ([1,C,1,1] against [1,C,H,W]), with
// more than one element on each side: outer leading elements, each repeated
// over inner trailing ones. (One element on the small side is the shim's
// scalar path; one on the trailing side, its same-shape path.)
bool channel_split(const Tensor& big, const Tensor& small, size_t* outer,
                   size_t* inner) {
  const ssize_t rank = big.dim();
  if (small.dim() > rank) {
    return false;
  }
  const ssize_t shift = rank - small.dim();
  auto small_size = [&](ssize_t dim) -> int64_t {
    return dim < shift ? 1 : small.size(dim - shift);
  };
  ssize_t split = rank;
  while (split > 0 && small_size(split - 1) == 1) {
    --split;
  }
  int64_t leading = 1;
  for (ssize_t dim = 0; dim < split; ++dim) {
    if (small_size(dim) != big.size(dim)) {
      return false;
    }
    leading *= big.size(dim);
  }
  int64_t trailing = 1;
  for (ssize_t dim = split; dim < rank; ++dim) {
    trailing *= big.size(dim);
  }
  if (leading <= 1 || trailing <= 1) {
    return false;
  }
  *outer = static_cast<size_t>(leading);
  *inner = static_cast<size_t>(trailing);
  return true;
}

#endif  // ESP32_GENERATED_EW_MUL

bool g_enabled = true;

}  // namespace

#if ESP32_GENERATED_EW_MUL
// aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out): three
// arguments plus the returned alias of out.
bool try_mul_out(KernelRuntimeContext& ctx, Span<EValue*> stack) {
  (void)ctx;
  if (!g_enabled || stack.size() != 4 || !stack[0]->isTensor() ||
      !stack[1]->isTensor() || !stack[2]->isTensor()) {
    return false;
  }
  const Tensor& left = stack[0]->toTensor();
  const Tensor& right = stack[1]->toTensor();
  const Tensor& out = stack[2]->toTensor();
  if (left.scalar_type() != ScalarType::Float ||
      right.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float) {
    return false;
  }
  // The operand of the output's shape streams; the other holds the scales.
  const bool left_streams = same_shape(left, out);
  if (left_streams == same_shape(right, out)) {
    return false;  // both (same shapes) or neither (a two-sided broadcast)
  }
  const Tensor& big = left_streams ? left : right;
  const Tensor& small = left_streams ? right : left;
  size_t outer = 0;
  size_t inner = 0;
  if (!channel_split(big, small, &outer, &inner) || !dense(big) ||
      !dense(small) || !dense(out) ||
      (big.const_data_ptr() != out.const_data_ptr() && overlap(big, out)) ||
      overlap(small, out)) {
    return false;
  }
  const float* scale = small.const_data_ptr<float>();
  for (size_t o = 0; o < outer; ++o) {
    if (scale[o] != scale[o]) {
      return false;  // a NaN scale: the shim's (elementwise.h says why)
    }
  }
  mul_channels_f32(scale, big.const_data_ptr<float>(),
                   out.mutable_data_ptr<float>(), outer, inner);
  ++g_counters.mul_channels;
  return true;
}
#else
bool try_mul_out(KernelRuntimeContext&, Span<EValue*>) {
  return false;
}
#endif  // ESP32_GENERATED_EW_MUL


#if ESP32_GENERATED_EW_CONSTANT_PAD
// aten::constant_pad_nd.out(Tensor self, SymInt[] pad, Scalar value=0, *,
// Tensor(a!) out): three arguments plus the returned alias of out. The row
// copy takes int8 and fp32 tensors in the default dim order with canonical
// strides on every axis (canonical()), of rank 1 or more, padded by 2 or 4 non-negative amounts (the last axis, or
// the last two), whose static out already has the padded shape and lies
// apart from the input; the value is the portable kernel's own cast of the
// scalar (check_overflow_scalar_cast), so a value it refuses goes to it too.
void shim_constant_pad_nd_out(KernelRuntimeContext& ctx, Span<EValue*> stack) {
  if (stack.size() != 5) {
    ctx.fail(executorch::runtime::Error::InvalidProgram);
    return;
  }
  const Tensor& in = stack[0]->toTensor();
  const executorch::aten::IntArrayRef pad = stack[1]->toIntList();
  const Scalar value = stack[2]->toScalar();
  Tensor& out = stack[3]->toTensor();
  auto portable = [&] {
    ++g_counters.pad_fallback;
    torch::executor::native::constant_pad_nd_out(ctx, in, pad, value, out);
  };
  const ScalarType type = in.scalar_type();
  const ssize_t rank = in.dim();
  if (!g_enabled || (type != ScalarType::Char && type != ScalarType::Float) ||
      out.scalar_type() != type || out.dim() != rank || rank < 1 ||
      (pad.size() != 2 && pad.size() != 4) ||
      static_cast<ssize_t>(pad.size() / 2) > rank || in.numel() <= 0 ||
      !canonical(in) || !canonical(out)) {
    portable();
    return;
  }
  for (size_t i = 0; i < pad.size(); ++i) {
    if (pad[i] < 0 || pad[i] > 0xFFFF) {
      portable();
      return;
    }
  }
  // pad = [left, right(, top, bottom)]: the last axis, then the one before.
  const size_t left = static_cast<size_t>(pad[0]);
  const size_t right = static_cast<size_t>(pad[1]);
  const size_t top = pad.size() == 4 ? static_cast<size_t>(pad[2]) : 0;
  const size_t bottom = pad.size() == 4 ? static_cast<size_t>(pad[3]) : 0;
  const size_t width = static_cast<size_t>(in.size(rank - 1));
  const size_t height = rank >= 2 ? static_cast<size_t>(in.size(rank - 2)) : 1;
  bool shaped = static_cast<size_t>(out.size(rank - 1)) == left + width + right;
  if (rank >= 2) {
    shaped &= static_cast<size_t>(out.size(rank - 2)) == top + height + bottom;
  } else {
    shaped &= top == 0 && bottom == 0;
  }
  for (ssize_t dim = 0; dim + 2 < rank; ++dim) {
    shaped &= out.size(dim) == in.size(dim);
  }
  if (!shaped) {
    portable();
    return;
  }
  const size_t element = type == ScalarType::Char ? 1 : 4;
  const uintptr_t in0 = reinterpret_cast<uintptr_t>(in.const_data_ptr());
  const uintptr_t out0 = reinterpret_cast<uintptr_t>(out.const_data_ptr());
  const size_t in_bytes = static_cast<size_t>(in.numel()) * element;
  const size_t out_bytes = static_cast<size_t>(out.numel()) * element;
  if (in0 < out0 + out_bytes && out0 < in0 + in_bytes) {
    portable();
    return;
  }
  const bool padded = left + right + top + bottom > 0;
  const size_t planes = static_cast<size_t>(in.numel()) / (height * width);
  if (type == ScalarType::Char) {
    const std::optional<int8_t> fill =
        torch::executor::native::utils::internal::check_overflow_scalar_cast<
            int8_t>(padded ? value : Scalar(0));
    if (!fill.has_value()) {
      portable();
      return;
    }
    pad_planes(static_cast<const uint8_t*>(in.const_data_ptr()),
               static_cast<uint8_t*>(out.mutable_data_ptr()), planes, height,
               width, top, bottom, left, right,
               static_cast<uint8_t>(fill.value()));
  } else {
    const std::optional<float> fill =
        torch::executor::native::utils::internal::check_overflow_scalar_cast<
            float>(padded ? value : Scalar(0));
    if (!fill.has_value()) {
      portable();
      return;
    }
    uint32_t bits;
    const float f = fill.value();
    memcpy(&bits, &f, sizeof(bits));
    pad_planes(static_cast<const uint32_t*>(in.const_data_ptr()),
               static_cast<uint32_t*>(out.mutable_data_ptr()), planes, height,
               width, top, bottom, left, right, bits);
  }
  ++g_counters.pad_rows;
}
#endif  // ESP32_GENERATED_EW_CONSTANT_PAD

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
  if (c.mul_channels != 0) {
    printf("mul.out elementwise dispatch: channels %llu\n",
           (unsigned long long)c.mul_channels);
  }
  if (c.lut_f32 != 0 || c.lut_codes != 0) {
    printf("quant_lut elementwise dispatch: fp32 %llu, codes %llu\n",
           (unsigned long long)c.lut_f32, (unsigned long long)c.lut_codes);
  }
  if (c.pad_rows != 0 || c.pad_fallback != 0) {
    printf("constant_pad_nd.out elementwise dispatch: rows %llu, fallback "
           "%llu\n", (unsigned long long)c.pad_rows,
           (unsigned long long)c.pad_fallback);
  }
}

}  // namespace elementwise
}  // namespace espressif
