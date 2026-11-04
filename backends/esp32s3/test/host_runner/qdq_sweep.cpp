/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// esp32_host_runner --qdq-sweep quick|full: the fast paths of ops/qdq_pool.h
// against ExecuTorch's portable kernels, far past the conformance cases.
//
//   quantize    every float32 bit pattern (full; quick: a prime stride of
//               them) plus every rounding tie's neighbourhood, at 21 scale and
//               zero-point sets (typical, power-of-two, extreme, zero,
//               infinite, NaN and negative scales; 8- and 16-bit codes), and
//               random scales. Each code is compared with the portable
//               kernel (on aarch64 its NEON loop for 8-bit codes, its scalar
//               loop for 16-bit ones) and with a transcription of the scalar
//               loop, op_quantize.cpp's quantize_val_with_inv_scale, which is
//               what the S3 runs.
//   dequantize  every code of every type at the specials and random scales.
//   avg_pool2d  random shapes, kernels, strides, paddings, ceil modes,
//               divisors and data (some with NaNs and infinities) through
//               the shim and the portable kernel: every byte and the failure
//               state.
//
// Prints one "qdq sweep:" line per part and a total; a mismatch prints its
// first few cases. Returns the mismatch count (0 passes).

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

#include "qdq_pool.h"

using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::BoxedEvalueList;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace torch {
namespace executor {
namespace native {
Tensor& quantize_per_tensor_out(const Tensor&, double, int64_t, int64_t,
                                int64_t, ScalarType, Tensor&);
Tensor& dequantize_per_tensor_out(const Tensor&, double, int64_t, int64_t,
                                  int64_t, ScalarType,
                                  std::optional<ScalarType>, Tensor&);
Tensor& avg_pool2d_out(KernelRuntimeContext&, const Tensor&, IntArrayRef,
                       IntArrayRef, IntArrayRef, bool, bool,
                       std::optional<int64_t>, Tensor&);
}  // namespace native
}  // namespace executor
}  // namespace torch

namespace {

// A 1-D tensor over caller-owned memory.
struct Flat {
  TensorImpl::SizesType size;
  TensorImpl::DimOrderType order = 0;
  TensorImpl::StridesType stride = 1;
  TensorImpl impl;
  Flat(ScalarType type, size_t n, void* data)
      : size(static_cast<TensorImpl::SizesType>(n)),
        impl(type, 1, &size, data, &order, &stride) {}
  Tensor tensor() { return Tensor(&impl); }
};

// op_quantize.cpp's quantize_val_with_inv_scale, transcribed (this file
// builds -ffp-contract=off, as nothing there fuses).
template <typename T>
T transcribed_quantize(float inv_scale, int64_t zero_point, float value,
                       int64_t quant_min, int64_t quant_max) {
  const float scaled_value = static_cast<float>(inv_scale * value);
  const double qvalue = std::isnan(scaled_value)
      ? static_cast<double>(zero_point)
      : static_cast<double>(zero_point) + std::nearbyint(scaled_value);
  const double clamped_value = std::min(
      static_cast<double>(quant_max),
      std::max(static_cast<double>(quant_min), qvalue));
  return static_cast<T>(clamped_value);
}

struct CodeInfo {
  size_t bytes;
  bool is_signed;
};

CodeInfo code_info(ScalarType type) {
  switch (type) {
    case ScalarType::Char:
      return {1, true};
    case ScalarType::Byte:
      return {1, false};
    case ScalarType::Short:
      return {2, true};
    default:
      return {2, false};
  }
}

int64_t code_at(const void* codes, const CodeInfo& code, size_t i) {
  if (code.bytes == 1) {
    return code.is_signed ? static_cast<const int8_t*>(codes)[i]
                          : static_cast<const uint8_t*>(codes)[i];
  }
  return code.is_signed ? static_cast<const int16_t*>(codes)[i]
                        : static_cast<const uint16_t*>(codes)[i];
}

struct QuantizeSet {
  const char* name;
  ScalarType dtype;
  double scale;
  int64_t zero_point, quant_min, quant_max;
};

float float_of(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t bits_of(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// One quantize set over `x`: the fast loop, the portable kernel and the
// transcription. Returns the mismatches; prints the first few.
uint64_t quantize_chunk(const QuantizeSet& set, const float* x, size_t n,
                        std::vector<uint8_t>& fast, std::vector<uint8_t>& portable,
                        std::atomic<int>& printed) {
  const CodeInfo code = code_info(set.dtype);
  fast.resize(n * code.bytes);
  portable.resize(n * code.bytes);
  const float inv = 1.0f / static_cast<float>(set.scale);
  espressif::qdq_pool::quantize_f32(
      x, fast.data(), n, code.bytes, code.is_signed, inv,
      static_cast<int32_t>(set.zero_point), static_cast<int32_t>(set.quant_min),
      static_cast<int32_t>(set.quant_max));
  Flat input(ScalarType::Float, n, const_cast<float*>(x));
  Flat out(set.dtype, n, portable.data());
  Tensor out_tensor = out.tensor();
  torch::executor::native::quantize_per_tensor_out(
      input.tensor(), set.scale, set.zero_point, set.quant_min, set.quant_max,
      set.dtype, out_tensor);
  uint64_t mismatches = 0;
  for (size_t i = 0; i < n; ++i) {
    const int64_t got = code_at(fast.data(), code, i);
    const int64_t kernel = code_at(portable.data(), code, i);
    int64_t scalar;
    if (code.bytes == 1) {
      scalar = code.is_signed
          ? transcribed_quantize<int8_t>(inv, set.zero_point, x[i],
                                         set.quant_min, set.quant_max)
          : transcribed_quantize<uint8_t>(inv, set.zero_point, x[i],
                                          set.quant_min, set.quant_max);
    } else {
      scalar = code.is_signed
          ? transcribed_quantize<int16_t>(inv, set.zero_point, x[i],
                                          set.quant_min, set.quant_max)
          : transcribed_quantize<uint16_t>(inv, set.zero_point, x[i],
                                           set.quant_min, set.quant_max);
    }
    if (got != kernel || got != scalar) {
      ++mismatches;
      if (printed.fetch_add(1) < 8) {
        std::printf("qdq sweep: MISMATCH quantize %s: x %08" PRIx32
                    " (%.9g): fast %" PRId64 ", portable kernel %" PRId64
                    ", scalar transcription %" PRId64 "\n",
                    set.name, bits_of(x[i]), x[i], got, kernel, scalar);
      }
    }
  }
  return mismatches;
}

// Values within three ulps of each rounding tie (k + 1/2) * scale for k
// across the code range and two past it.
std::vector<float> tie_values(const QuantizeSet& set) {
  std::vector<float> values;
  const float s = static_cast<float>(set.scale);
  for (int64_t k = set.quant_min - set.zero_point - 2;
       k <= set.quant_max - set.zero_point + 2; ++k) {
    const uint32_t tie = bits_of((static_cast<float>(k) + 0.5f) * s);
    for (int j = -3; j <= 3; ++j) {
      values.push_back(float_of(tie + static_cast<uint32_t>(j)));
    }
  }
  return values;
}

uint64_t sweep_quantize(bool full, int threads, uint64_t* checked) {
  // 21 sets: typical activation scales, powers of two, zero points at both
  // ends of the range, extreme scales (the reciprocal near FLT_MAX, infinite,
  // zero, NaN), a negative scale, and 16-bit codes.
  static const QuantizeSet sets[] = {
      {"int8 0.0235 zp -3", ScalarType::Char, 0.0235, -3, -128, 127},
      {"int8 2^-7 zp 0", ScalarType::Char, 0x1p-7, 0, -128, 127},
      {"int8 0.1 zp -128", ScalarType::Char, 0.1, -128, -128, 127},
      {"int8 0.1 zp 127", ScalarType::Char, 0.1, 127, -128, 127},
      {"int8 1.0 symmetric", ScalarType::Char, 1.0, 0, -127, 127},
      {"int8 2^-15 zp 5", ScalarType::Char, 0x1p-15, 5, -128, 127},
      {"int8 1e5", ScalarType::Char, 1e5, 0, -128, 127},
      {"int8 1e-38", ScalarType::Char, 1e-38, 0, -128, 127},
      {"int8 1e-39 (subnormal)", ScalarType::Char, 1e-39, 0, -128, 127},
      {"int8 scale 0", ScalarType::Char, 0.0, 0, -128, 127},
      {"int8 scale inf", ScalarType::Char, INFINITY, 1, -128, 127},
      {"int8 scale NaN", ScalarType::Char, NAN, 2, -128, 127},
      {"int8 -0.05 zp -1", ScalarType::Char, -0.05, -1, -128, 127},
      {"int8 0.3 zp 10", ScalarType::Char, 0.3, 10, -128, 127},
      {"uint8 0.0235 zp 128", ScalarType::Byte, 0.0235, 128, 0, 255},
      {"uint8 0.5 zp 8 [0,15]", ScalarType::Byte, 0.5, 8, 0, 15},
      {"uint8 2^-8 zp 0", ScalarType::Byte, 0x1p-8, 0, 0, 255},
      {"int16 2^-12 zp 0", ScalarType::Short, 0x1p-12, 0, -32768, 32767},
      {"int16 0.0003 zp -5", ScalarType::Short, 0.0003, -5, -32767, 32767},
      {"uint16 0.001 zp 32768", ScalarType::UInt16, 0.001, 32768, 0, 65535},
      {"bits16 7.0 zp 0", ScalarType::Bits16, 7.0, 0, 0, 65535},
  };
  constexpr size_t kChunk = size_t{1} << 16;
  // quick: one float in 65521 (a prime, so every exponent and mantissa
  // pattern is visited), 65,551 per set.
  const uint64_t step = full ? 1 : 65521;
  std::atomic<uint64_t> mismatches{0};
  std::atomic<uint64_t> count{0};
  std::atomic<int> printed{0};
  for (const QuantizeSet& set : sets) {
    std::vector<float> ties = tie_values(set);
    std::vector<uint8_t> fast, portable;
    mismatches += quantize_chunk(set, ties.data(), ties.size(), fast, portable,
                                 printed);
    count += ties.size();
    const uint64_t total = (uint64_t{1} << 32) / step + 1;
    const uint64_t chunks = (total + kChunk - 1) / kChunk;
    std::atomic<uint64_t> next{0};
    auto worker = [&] {
      std::vector<float> x(kChunk);
      std::vector<uint8_t> f, p;
      for (uint64_t chunk; (chunk = next.fetch_add(1)) < chunks;) {
        const uint64_t first = chunk * kChunk;
        const size_t n = static_cast<size_t>(std::min<uint64_t>(kChunk, total - first));
        for (size_t i = 0; i < n; ++i) {
          x[i] = float_of(static_cast<uint32_t>((first + i) * step));
        }
        mismatches += quantize_chunk(set, x.data(), n, f, p, printed);
        count += n;
      }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back(worker);
    }
    for (std::thread& t : pool) {
      t.join();
    }
  }
  // Random scales and zero points: 4096 input patterns each, plus the ties.
  std::mt19937_64 random(0x5eed);
  const int random_sets = full ? 2000 : 300;
  for (int index = 0; index < random_sets; ++index) {
    const ScalarType types[] = {ScalarType::Char, ScalarType::Byte,
                                ScalarType::Short, ScalarType::UInt16};
    const ScalarType dtype = types[random() % 4];
    int64_t lo = 0, hi = 0;
    switch (dtype) {
      case ScalarType::Char: lo = -128; hi = 127; break;
      case ScalarType::Byte: lo = 0; hi = 255; break;
      case ScalarType::Short: lo = -32768; hi = 32767; break;
      default: lo = 0; hi = 65535; break;
    }
    if (random() % 4 == 0) {  // a narrower range inside the type's
      lo += static_cast<int64_t>(random() % 8);
      hi -= static_cast<int64_t>(random() % 8);
    }
    const int64_t zp = lo + static_cast<int64_t>(random() % static_cast<uint64_t>(hi - lo + 1));
    // log-uniform over 2^-40..2^40, sometimes negative
    const double exponent = static_cast<double>(random() % 80000) / 1000.0 - 40.0;
    const double scale = std::exp2(exponent) * ((random() % 16) == 0 ? -1.0 : 1.0);
    const std::string name = "random set " + std::to_string(index);
    const QuantizeSet set = {name.c_str(), dtype, scale, zp, lo, hi};
    std::vector<float> x;
    if (dtype == ScalarType::Char || dtype == ScalarType::Byte) {
      x = tie_values(set);
    }
    for (int i = 0; i < 4096; ++i) {
      const uint64_t r = random();
      if (r % 3 == 0) {
        x.push_back(float_of(static_cast<uint32_t>(r >> 8)));
      } else {  // around the range
        const double t = static_cast<double>(lo - zp - 3) +
            static_cast<double>((r >> 8) % 1000000) / 1000000.0 *
                static_cast<double>(hi - lo + 6);
        x.push_back(static_cast<float>(t * scale));
      }
    }
    std::vector<uint8_t> fast, portable;
    mismatches += quantize_chunk(set, x.data(), x.size(), fast, portable, printed);
    count += x.size();
  }
  *checked = count.load();
  return mismatches.load();
}

uint64_t sweep_dequantize(bool full, uint64_t* checked) {
  std::mt19937_64 random(0xd5eed);
  std::vector<double> scales = {0.0235, 0x1p-7, 0.1, 1.0, 1e-38, 1e-39, 0.0,
                                INFINITY, -INFINITY, NAN, -0.05, 3e38, 1e-45,
                                0x1p-12, 0.001};
  const int random_scales = full ? 20000 : 1000;
  for (int i = 0; i < random_scales; ++i) {
    const double exponent = static_cast<double>(random() % 250000) / 1000.0 - 125.0;
    scales.push_back(std::exp2(exponent) * ((random() % 8) == 0 ? -1.0 : 1.0));
  }
  struct Type {
    ScalarType dtype;
    int64_t lo, hi;
  };
  static const Type types[] = {{ScalarType::Char, -128, 127},
                               {ScalarType::Byte, 0, 255},
                               {ScalarType::Short, -32768, 32767},
                               {ScalarType::UInt16, 0, 65535},
                               {ScalarType::Bits16, 0, 65535}};
  uint64_t mismatches = 0, count = 0;
  int printed = 0;
  for (const Type& type : types) {
    const CodeInfo code = code_info(type.dtype);
    // One-byte codes: every code nine times and a tail, 2,307 in all, past
    // the table the fast path builds from 2,048 codes on.
    const size_t n = code.bytes == 1 ? 256 * 9 + 3 : 65536;
    std::vector<uint8_t> codes(n * code.bytes);
    for (size_t i = 0; i < n; ++i) {
      if (code.bytes == 1) {
        codes[i] = static_cast<uint8_t>(i * 7);
      } else {
        const uint16_t v = static_cast<uint16_t>(i);
        std::memcpy(&codes[2 * i], &v, 2);
      }
    }
    std::vector<float> fast(n), portable(n);
    // 16-bit codes take a sample of the scales: each one is 65,536 codes.
    const size_t scale_step = code.bytes == 1 ? 1 : (full ? 4 : 40);
    for (size_t s = 0; s < scales.size(); s += scale_step) {
      const double scale = scales[s];
      const int64_t zps[] = {type.lo, type.hi, std::max<int64_t>(type.lo, 0),
                             type.lo + static_cast<int64_t>(random() % static_cast<uint64_t>(type.hi - type.lo + 1))};
      for (int64_t zp : zps) {
        espressif::qdq_pool::dequantize_f32(codes.data(), code.bytes,
                                            code.is_signed, fast.data(), n,
                                            static_cast<int32_t>(zp),
                                            static_cast<float>(scale));
        Flat input(type.dtype, n, codes.data());
        Flat out(ScalarType::Float, n, portable.data());
        Tensor out_tensor = out.tensor();
        torch::executor::native::dequantize_per_tensor_out(
            input.tensor(), scale, zp, type.lo, type.hi, type.dtype,
            std::nullopt, out_tensor);
        for (size_t i = 0; i < n; ++i) {
          const float scalar =
              (static_cast<int32_t>(code_at(codes.data(), code, i)) -
               static_cast<int32_t>(zp)) *
              static_cast<float>(scale);
          if (bits_of(fast[i]) != bits_of(portable[i]) ||
              bits_of(fast[i]) != bits_of(scalar)) {
            ++mismatches;
            if (printed++ < 8) {
              std::printf("qdq sweep: MISMATCH dequantize code %" PRId64
                          " scale %.9g zp %" PRId64 ": fast %08" PRIx32
                          ", portable kernel %08" PRIx32
                          ", scalar transcription %08" PRIx32 "\n",
                          code_at(codes.data(), code, i), scale, zp,
                          bits_of(fast[i]), bits_of(portable[i]),
                          bits_of(scalar));
            }
          }
        }
        count += n;
      }
    }
  }
  *checked = count;
  return mismatches;
}

// A tensor of up to four dimensions over caller-owned memory.
struct Shaped {
  TensorImpl::SizesType sizes[4];
  TensorImpl::DimOrderType order[4] = {0, 1, 2, 3};
  TensorImpl::StridesType strides[4];
  TensorImpl impl;
  Shaped(ScalarType type, int dim, const int64_t* shape, void* data)
      : impl(type, dim, init(dim, shape), data, order, strides) {}
  TensorImpl::SizesType* init(int dim, const int64_t* shape) {
    TensorImpl::StridesType stride = 1;
    for (int d = dim; d-- > 0;) {
      sizes[d] = static_cast<TensorImpl::SizesType>(shape[d]);
      strides[d] = stride;
      stride *= sizes[d];
    }
    return sizes;
  }
  Tensor tensor() { return Tensor(&impl); }
};

uint64_t sweep_pool(bool full, uint64_t* checked, uint64_t* took_fast,
                    uint64_t* global_padded) {
  std::mt19937_64 random(0x9001);
  const int cases = full ? 50000 : 3000;
  uint64_t mismatches = 0, count = 0, fast_count = 0, padded_count = 0;
  int printed = 0;
  auto& counters = espressif::qdq_pool::counters();
  for (int index = 0; index < cases; ++index) {
    const int dim = random() % 3 == 0 ? 3 : 4;
    int64_t shape[4];
    int d = 0;
    if (dim == 4) {
      shape[d++] = 1 + static_cast<int64_t>(random() % 2);
    }
    shape[d++] = 1 + static_cast<int64_t>(random() % 9);
    const int64_t height = 1 + static_cast<int64_t>(random() % 10);
    const int64_t width = 1 + static_cast<int64_t>(random() % 10);
    shape[d++] = height;
    shape[d++] = width;
    const bool global = random() % 3 == 0;
    int64_t kernel[2], stride[2], padding[2];
    size_t kernel_count, stride_count, padding_count;
    if (global) {
      // A window over the whole plane, padding included: per axis a padding
      // p <= the extent (half the time 0) and a kernel in [max(extent, 2p),
      // extent + 2p]. From extent + p on, one window covers the plane, and
      // with count_include_pad its count is the padded kernel's, not the
      // plane's; below it the window is clipped (the windowed path).
      const int64_t extents[2] = {height, width};
      for (int a = 0; a < 2; ++a) {
        padding[a] = random() % 2
            ? 0 : static_cast<int64_t>(random() % static_cast<uint64_t>(extents[a] + 1));
        const int64_t lo = std::max(extents[a], 2 * padding[a]);
        const int64_t hi = extents[a] + 2 * padding[a];
        kernel[a] = lo + static_cast<int64_t>(random() % static_cast<uint64_t>(hi - lo + 1));
      }
      kernel_count = 2;
      padding_count = 2;
      stride_count = random() % 2 ? 0 : 2;  // the kernel, or both given
      stride[0] = random() % 2 ? kernel[0] : 1 + static_cast<int64_t>(random() % 3);
      stride[1] = random() % 2 ? kernel[1] : 1 + static_cast<int64_t>(random() % 3);
    } else {
      kernel[0] = 1 + static_cast<int64_t>(random() % height);
      kernel[1] = 1 + static_cast<int64_t>(random() % width);
      kernel_count = (kernel[0] == kernel[1] && random() % 2) ? 1 : 2;
      stride_count = random() % 3;  // 0: the kernel, 1: one value, 2
      stride[0] = 1 + static_cast<int64_t>(random() % 3);
      stride[1] = stride_count == 1 ? stride[0] : 1 + static_cast<int64_t>(random() % 3);
      padding_count = 1 + random() % 2;
      padding[0] = static_cast<int64_t>(random() % (kernel[0] / 2 + 1));
      padding[1] = padding_count == 1
          ? padding[0]
          : static_cast<int64_t>(random() % (kernel[kernel_count == 1 ? 0 : 1] / 2 + 1));
      if (padding_count == 1 && padding[0] > kernel[kernel_count == 1 ? 0 : 1] / 2) {
        padding[0] = 0;
        padding[1] = 0;
      }
    }
    const bool ceil_mode = random() % 4 == 0;
    const bool count_include_pad = random() % 2;
    std::optional<int64_t> divisor;
    if (index % 997 == 0) {
      divisor = 0;  // refused alike (and logged by each side)
    } else if (random() % 5 == 0) {
      divisor = static_cast<int64_t>(random() % 14) - 7;  // -7..6
      if (*divisor == 0) {
        divisor = 7;
      }
    }
    const IntArrayRef kernel_ref(kernel, kernel_count);
    const IntArrayRef stride_ref(stride, stride_count);
    const IntArrayRef padding_ref(padding, padding_count);
    size_t in_numel = 1;
    for (int i = 0; i < dim; ++i) {
      in_numel *= static_cast<size_t>(shape[i]);
    }
    std::vector<float> x(in_numel);
    const int special = random() % 20;
    for (size_t i = 0; i < in_numel; ++i) {
      const uint64_t r = random();
      switch (r % 8) {
        case 0: x[i] = (r & 8) ? -0.0f : 0.0f; break;
        case 1: x[i] = float_of(static_cast<uint32_t>((r >> 8) & 0x807fffffu)); break;
        case 2: x[i] = static_cast<float>(static_cast<int32_t>(r >> 16)) * 0x1p+40f; break;
        default:
          x[i] = static_cast<float>(static_cast<int32_t>(r >> 16)) * 0x1p-31f *
              static_cast<float>(1u << ((r >> 4) % 16));
      }
    }
    if (special == 0) {
      x[random() % in_numel] = float_of(0x7fc00000u | static_cast<uint32_t>(random() % 0x3fffff));
      x[random() % in_numel] = float_of(0xffc00000u | static_cast<uint32_t>(random() % 0x3fffff));
    } else if (special == 1) {
      x[random() % in_numel] = INFINITY;
      x[random() % in_numel] = -INFINITY;
    } else if (special == 2) {
      x[random() % in_numel] = INFINITY;
    }
    // The output size, as both compute it; a refused size is still a case
    // (both must fail alike), with a one-element output tensor.
    int64_t out_shape[4];
    {
      Shaped probe(ScalarType::Float, dim, shape, x.data());
      executorch::aten::SizesType sizes[4];
      size_t ndim = 0;
      torch::executor::get_avg_pool2d_out_target_size(
          probe.tensor(), kernel_ref, stride_ref, padding_ref, ceil_mode, sizes,
          &ndim);
      for (int i = 0; i < dim; ++i) {
        out_shape[i] = std::max<int64_t>(sizes[i], 1);
      }
    }
    size_t out_numel = 1;
    for (int i = 0; i < dim; ++i) {
      out_numel *= static_cast<size_t>(out_shape[i]);
    }
    std::vector<float> fast_out(out_numel, float_of(0xa5a5a5a5u));
    std::vector<float> portable_out(out_numel, float_of(0xa5a5a5a5u));
    Shaped input(ScalarType::Float, dim, shape, x.data());
    KernelRuntimeContext fast_ctx, portable_ctx;
    const uint64_t fast_before = counters.pool_global + counters.pool_windowed;
    const uint64_t global_before = counters.pool_global;
    {
      Shaped out(ScalarType::Float, dim, out_shape, fast_out.data());
      EValue kernel_values[2] = {EValue(kernel[0]), EValue(kernel[1])};
      EValue stride_values[2] = {EValue(stride[0]), EValue(stride[1])};
      EValue padding_values[2] = {EValue(padding[0]), EValue(padding[1])};
      EValue* kernel_boxed[2] = {&kernel_values[0], &kernel_values[1]};
      EValue* stride_boxed[2] = {&stride_values[0], &stride_values[1]};
      EValue* padding_boxed[2] = {&padding_values[0], &padding_values[1]};
      int64_t kernel_unboxed[2], stride_unboxed[2], padding_unboxed[2];
      BoxedEvalueList<int64_t> kernel_list(kernel_boxed, kernel_unboxed,
                                           static_cast<int>(kernel_count));
      BoxedEvalueList<int64_t> stride_list(stride_boxed, stride_unboxed,
                                           static_cast<int>(stride_count));
      BoxedEvalueList<int64_t> padding_list(padding_boxed, padding_unboxed,
                                            static_cast<int>(padding_count));
      EValue in_value(input.tensor()), kernel_value(&kernel_list);
      EValue stride_value(&stride_list), padding_value(&padding_list);
      EValue ceil_value(ceil_mode), pad_value(count_include_pad), divisor_value;
      if (divisor.has_value()) {
        divisor_value = EValue(divisor.value());
      }
      EValue out_value(out.tensor());
      EValue* stack[9] = {&in_value,   &kernel_value, &stride_value,
                          &padding_value, &ceil_value, &pad_value,
                          &divisor_value, &out_value,  &out_value};
      espressif::qdq_pool::shim_avg_pool2d_out(fast_ctx, Span<EValue*>(stack, 9));
    }
    fast_count += counters.pool_global + counters.pool_windowed - fast_before;
    // The global branch dividing by a padded count: the case the windowed
    // path's clipping never shows.
    if (counters.pool_global != global_before && count_include_pad &&
        !divisor.has_value() && (padding[0] > 0 || padding[1] > 0)) {
      ++padded_count;
    }
    {
      Shaped out(ScalarType::Float, dim, out_shape, portable_out.data());
      Tensor out_tensor = out.tensor();
      torch::executor::native::avg_pool2d_out(
          portable_ctx, input.tensor(), kernel_ref, stride_ref, padding_ref,
          ceil_mode, count_include_pad, divisor, out_tensor);
    }
    ++count;
    if (fast_ctx.failure_state() != portable_ctx.failure_state() ||
        std::memcmp(fast_out.data(), portable_out.data(),
                    out_numel * sizeof(float)) != 0) {
      ++mismatches;
      if (printed++ < 8) {
        std::printf("qdq sweep: MISMATCH avg_pool2d case %d: dim %d "
                    "[%lld,%lld,%lld,%lld] kernel %lldx%lld stride %zu:%lldx%lld "
                    "padding %lldx%lld ceil %d pad %d divisor %lld\n",
                    index, dim, (long long)shape[0], (long long)shape[1],
                    (long long)shape[2], (long long)(dim == 4 ? shape[3] : 0),
                    (long long)kernel[0], (long long)kernel[1], stride_count,
                    (long long)stride[0], (long long)stride[1],
                    (long long)padding[0], (long long)padding[1], ceil_mode,
                    count_include_pad,
                    (long long)(divisor.has_value() ? divisor.value() : -99));
      }
    }
  }
  *checked = count;
  *took_fast = fast_count;
  *global_padded = padded_count;
  return mismatches;
}

}  // namespace

int run_qdq_sweep(const std::string& mode) {
  if (mode != "quick" && mode != "full") {
    std::fprintf(stderr, "--qdq-sweep takes quick or full\n");
    return 2;
  }
  const bool full = mode == "full";
  const int threads =
      full ? std::max(1u, std::min(4u, std::thread::hardware_concurrency())) : 1;
  uint64_t quantized = 0, dequantized = 0, pools = 0, pools_fast = 0, pools_padded = 0;
  const uint64_t q = sweep_quantize(full, threads, &quantized);
  std::printf("qdq sweep: quantize %" PRIu64 " value(s), %" PRIu64
              " mismatch(es)\n", quantized, q);
  const uint64_t dq = sweep_dequantize(full, &dequantized);
  std::printf("qdq sweep: dequantize %" PRIu64 " code(s), %" PRIu64
              " mismatch(es)\n", dequantized, dq);
  uint64_t p = sweep_pool(full, &pools, &pools_fast, &pools_padded);
  std::printf("qdq sweep: avg_pool2d %" PRIu64 " case(s), %" PRIu64
              " fast (%" PRIu64 " global with a counted padding), %" PRIu64
              " mismatch(es)\n", pools, pools_fast, pools_padded, p);
  // Beside four fixed check cases, only this sampling reaches the global
  // branch that divides by a counted padding. The seed is fixed, so the
  // count is too (148 quick, 2,380 full): none means the sampling or the
  // shim's global condition changed and the branch went untested.
  if (pools_padded == 0) {
    std::printf("qdq sweep: MISMATCH avg_pool2d: no global pool with a "
                "counted padding\n");
    ++p;
  }
  std::printf("qdq sweep: %s, %" PRIu64 " mismatch(es)\n", mode.c_str(),
              q + dq + p);
  return q + dq + p == 0 ? 0 : 1;
}
