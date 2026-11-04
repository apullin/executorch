/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The stock quantised K/V cache's kernels outside the delegate (kv_affine.h).
// Built -ffp-contract=off on the host and the device: the writer's float32
// steps and float64 quantise are each one IEEE operation in torch's order,
// with nothing fused. On the device kv_affine.lf links this file after every
// other flash object, so the story firmware's code keeps its addresses.

#include "kv_affine.h"

#include <cmath>
#include <cstdio>
#include <cstring>

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
namespace kv_affine {
namespace {

// torch.finfo(torch.float32).eps, the per-token chooser's scale floor.
constexpr float kFltEpsilon = 1.1920928955078125e-07f;  // 2^-23

uint64_t g_quantized_rows = 0;
uint64_t g_dequantized_rows = 0;
uint64_t g_cache_writes = 0;
uint64_t g_refusals = 0;

void refuse(KernelRuntimeContext& ctx, Error error) {
  ++g_refusals;
  ctx.fail(error);
}

// Contiguous strides in the default dim order: size-1 axes may carry any
// stride (quant_lut.cpp's dense).
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

// x[..., D] beside a per-row side tensor [..., 1]: same leading sizes.
bool row_side(const Tensor& x, const Tensor& side) {
  if (x.dim() < 1 || side.dim() != x.dim() || side.size(side.dim() - 1) != 1) {
    return false;
  }
  for (ssize_t dim = 0; dim + 1 < x.dim(); ++dim) {
    if (side.size(dim) != x.size(dim)) {
      return false;
    }
  }
  return true;
}

// Round half to even in float32: for |v| <= 2^22, v + 1.5 * 2^23 lands where
// the spacing is 1, so the add rounds v and the subtract is exact.
float rne_f(float v) {
  const float magic = 12582912.0f;
  return (v + magic) - magic;
}

// rne(fl64(p + zd)) for p = fl64(x * inv), in float64: 1.5 * 2^52 rounds
// exactly for |v| < 2^51 (any value a row's own scale quantises lies within
// a few hundred). Past that v is already integral or infinite, and returned
// for the caller's clamp, or NaN: then p was NaN (zd never is), and the code
// is the zero point, ExecuTorch 1.6's rule (quant_rules.h). Only this branch,
// which a finite row never takes, tests for NaN.
double code_d(double p, double zd) {
  const double magic = 6755399441055744.0;
  const double v = p + zd;
  if (std::fabs(v) < 2251799813685248.0) {
    return (v + magic) - magic;
  }
  return quant_scaled(p) + zd;  // v, or zd for NaN
}

}  // namespace

void quantize_row(const float* x, size_t n, int8_t* codes, float* scale,
                  int8_t* zero_point) {
  float lo = 0.0f;  // min(amin, 0)
  float hi = 0.0f;  // max(amax, 0)
  for (size_t i = 0; i < n; ++i) {
    lo = x[i] < lo ? x[i] : lo;
    hi = x[i] > hi ? x[i] : hi;
  }
  const float span = hi - lo;
  float s = span / 255.0f;
  s = s < kFltEpsilon ? kFltEpsilon : s;
  const float dmin = lo / s;
  const float dmax = hi / s;
  const float from_min_error = -128.0f + dmin;
  const float from_max_error = 127.0f + dmax;
  float z = (from_min_error + from_max_error) > 0.0f ? -128.0f - dmin
                                                      : 127.0f - dmax;
  // A row holding +inf has s = inf and dmax = inf / inf: its zero point is
  // NaN in torch too, and takes 0 here (quant_rules.h), before the clamp.
  z = quant_scaled(z);
  z = z < -128.0f ? -128.0f : (z > 127.0f ? 127.0f : z);
  const float zr = rne_f(z);
  // quantize_per_token: 1.0 / scales in float64 (the scale a float32 value),
  // the float32 input promoted to float64 by the multiply, then + zp.
  const double inv = 1.0 / (double)s;
  const double zd = (double)zr;
  for (size_t i = 0; i < n; ++i) {
    double v = code_d((double)x[i] * inv, zd);
    v = v < -128.0 ? -128.0 : (v > 127.0 ? 127.0 : v);
    codes[i] = (int8_t)(int32_t)v;
  }
  *scale = s;
  *zero_point = (int8_t)(int32_t)zr;
}

void dequantize_row(const int8_t* codes, size_t n, float scale,
                    int8_t zero_point, float* out) {
  const int32_t z = zero_point;
  for (size_t i = 0; i < n; ++i) {
    out[i] = (float)((int32_t)codes[i] - z) * scale;
  }
}

void shim_quantize(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Four arguments plus the returned list of the three outs.
  if (st.size() != 5) {
    refuse(ctx, Error::InvalidProgram);
    return;
  }
  if (!st[0]->isTensor() || !st[1]->isTensor() || !st[2]->isTensor() ||
      !st[3]->isTensor()) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const Tensor& x = st[0]->toTensor();
  Tensor& codes = st[1]->toTensor();
  Tensor& scales = st[2]->toTensor();
  Tensor& zps = st[3]->toTensor();
  const bool ok = x.scalar_type() == ScalarType::Float &&
      codes.scalar_type() == ScalarType::Char &&
      scales.scalar_type() == ScalarType::Float &&
      zps.scalar_type() == ScalarType::Char && x.dim() >= 1 &&
      x.size(x.dim() - 1) > 0 && row_side(x, scales) && row_side(x, zps) &&
      dense(x);
  if (!ok || executorch::runtime::resize_tensor(codes, x.sizes()) != Error::Ok ||
      !dense(codes) || !dense(scales) || !dense(zps)) {
    printf("  [kv_affine] quantize refused: dtypes %d/%d/%d/%d\n",
           (int)x.scalar_type(), (int)codes.scalar_type(),
           (int)scales.scalar_type(), (int)zps.scalar_type());
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const size_t width = (size_t)x.size(x.dim() - 1);
  const size_t rows = (size_t)x.numel() / width;
  const float* xd = x.const_data_ptr<float>();
  int8_t* cd = codes.mutable_data_ptr<int8_t>();
  float* sd = scales.mutable_data_ptr<float>();
  int8_t* zd = zps.mutable_data_ptr<int8_t>();
  for (size_t r = 0; r < rows; ++r) {
    quantize_row(xd + r * width, width, cd + r * width, sd + r, zd + r);
  }
  g_quantized_rows += rows;
}

void shim_dequantize(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Four arguments plus the returned alias of out.
  if (st.size() != 5) {
    refuse(ctx, Error::InvalidProgram);
    return;
  }
  if (!st[0]->isTensor() || !st[1]->isTensor() || !st[2]->isTensor() ||
      !st[3]->isTensor()) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const Tensor& codes = st[0]->toTensor();
  const Tensor& scales = st[1]->toTensor();
  const Tensor& zps = st[2]->toTensor();
  Tensor& out = st[3]->toTensor();
  const bool ok = codes.scalar_type() == ScalarType::Char &&
      scales.scalar_type() == ScalarType::Float &&
      zps.scalar_type() == ScalarType::Char &&
      out.scalar_type() == ScalarType::Float && codes.dim() >= 1 &&
      codes.size(codes.dim() - 1) > 0 && row_side(codes, scales) &&
      row_side(codes, zps) && dense(codes) && dense(scales) && dense(zps);
  if (!ok || executorch::runtime::resize_tensor(out, codes.sizes()) != Error::Ok ||
      !dense(out)) {
    printf("  [kv_affine] dequantize refused: dtypes %d/%d/%d/%d\n",
           (int)codes.scalar_type(), (int)scales.scalar_type(),
           (int)zps.scalar_type(), (int)out.scalar_type());
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const size_t width = (size_t)codes.size(codes.dim() - 1);
  const size_t rows = (size_t)codes.numel() / width;
  const int8_t* cd = codes.const_data_ptr<int8_t>();
  const float* sd = scales.const_data_ptr<float>();
  const int8_t* zd = zps.const_data_ptr<int8_t>();
  float* od = out.mutable_data_ptr<float>();
  for (size_t r = 0; r < rows; ++r) {
    dequantize_row(cd + r * width, width, sd[r], zd[r], od + r * width);
  }
  g_dequantized_rows += rows;
}

void shim_update_cache(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // Four arguments plus the returned alias of out.
  if (st.size() != 5) {
    refuse(ctx, Error::InvalidProgram);
    return;
  }
  if (!st[0]->isTensor() || !st[1]->isTensor() || !st[2]->isInt() ||
      !st[3]->isTensor()) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const Tensor& value = st[0]->toTensor();
  Tensor& cache = st[1]->toTensor();
  const int64_t start_pos = st[2]->toInt();
  // op_update_cache.cpp's validate_cache_params and update_cache_impl checks:
  // 4-D, contiguous, the same batch, heads, width and element size, and the
  // rows written within the cache's sequence axis.
  bool ok = value.dim() == 4 && cache.dim() == 4 && dense(value) &&
      dense(cache) && value.element_size() == cache.element_size() &&
      value.size(0) == cache.size(0) && value.size(2) == cache.size(2) &&
      value.size(3) == cache.size(3) && start_pos >= 0 &&
      start_pos < cache.size(1) && start_pos + value.size(1) <= cache.size(1);
  if (!ok) {
    printf("  [kv_affine] update_cache refused: start_pos %lld\n",
           (long long)start_pos);
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  const size_t elem = value.element_size();
  const size_t per_batch = (size_t)(value.numel() / value.size(0)) * elem;
  const size_t cache_batch = (size_t)cache.strides()[0] * elem;
  const size_t cache_seq = (size_t)cache.strides()[1] * elem;
  const uint8_t* src = static_cast<const uint8_t*>(value.const_data_ptr());
  uint8_t* dst = static_cast<uint8_t*>(cache.mutable_data_ptr());
  for (ssize_t b = 0; b < value.size(0); ++b) {
    std::memcpy(dst + (size_t)b * cache_batch + (size_t)start_pos * cache_seq,
                src + (size_t)b * per_batch, per_batch);
  }
  ++g_cache_writes;
}

void shim_local_scalar_dense(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 2 || !st[0]->isTensor()) {
    refuse(ctx, Error::InvalidProgram);
    return;
  }
  const Tensor& self = st[0]->toTensor();
  if (self.numel() < 1) {
    refuse(ctx, Error::InvalidArgument);
    return;
  }
  EValue& out = *st[1];
  switch (self.scalar_type()) {
    case ScalarType::Long:
      out = EValue(executorch::aten::Scalar(self.const_data_ptr<int64_t>()[0]));
      return;
    case ScalarType::Int:
      out = EValue(executorch::aten::Scalar(
          (int64_t)self.const_data_ptr<int32_t>()[0]));
      return;
    case ScalarType::Bool:
      out = EValue(executorch::aten::Scalar(self.const_data_ptr<bool>()[0]));
      return;
    case ScalarType::Float:
      out = EValue(executorch::aten::Scalar(
          (double)self.const_data_ptr<float>()[0]));
      return;
    case ScalarType::Double:
      out = EValue(executorch::aten::Scalar(self.const_data_ptr<double>()[0]));
      return;
    default:
      refuse(ctx, Error::InvalidArgument);
      return;
  }
}

Stats stats() {
  return {g_quantized_rows, g_dequantized_rows, g_cache_writes, g_refusals};
}

void report_dispatch() {
  if (g_quantized_rows == 0 && g_dequantized_rows == 0 &&
      g_cache_writes == 0 && g_refusals == 0) {
    return;
  }
  printf("kv_affine dispatch: quantized rows %llu, dequantized rows %llu, "
         "cache writes %llu, refusals %llu\n",
         (unsigned long long)g_quantized_rows,
         (unsigned long long)g_dequantized_rows,
         (unsigned long long)g_cache_writes, (unsigned long long)g_refusals);
}

}  // namespace kv_affine
}  // namespace espressif
