/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// espressif::layer_norm2d (layer_norm2d.h): the kernel, its refusal check and
// its counter. Built -ffp-contract=off on the host and the device: the graph
// it replaces rounds pow(d, 2) before its mean sums it and w * q before it
// adds b, so neither product may become a madd.s. In espressif_ops.cpp, which
// the device builds with GCC's default contraction, both did (sq += d * d and
// w[c] * q + bs[c]): one ulp off the graph in a fifth of the outputs, which
// the host, built without contraction, could not see.

#include "layer_norm2d.h"

#include <cmath>
#include <cstddef>
#include <cstdio>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include "espressif_ops.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

namespace espressif {
namespace layer_norm2d {
namespace {

uint64_t g_cols = 0;

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

bool contiguous_tensor(const Tensor& tensor) {
  ssize_t expected = 1;
  for (ssize_t dim = tensor.dim(); dim-- > 0;) {
    if (tensor.size(dim) > 1 && tensor.strides()[dim] != expected) {
      return false;
    }
    expected *= tensor.size(dim);
  }
  return true;
}

}  // namespace

uint64_t columns() {
  return g_cols;
}

// LayerNorm2d: normalise over the CHANNEL axis of an NCHW tensor.
//
// The reduction runs DOWN a column of channels, whose elements are H*W apart --
// at a 32x32 feature map that is a stride of 1024 floats. There is no contiguous
// version of this to reach for: making it contiguous is a transpose, which costs
// two full passes over the tensor, which is what the fourteen portable ops this
// replaces pay.
//
// Two passes over each column instead of one Welford pass, deliberately. Welford
// is a different expression tree and this has to agree bit-for-bit with the
// unfused graph, which computes mean first and then the mean of the squared
// deviations. Numerical stability is not the constraint here -- C is 16 to 256,
// not a million.
//
// Each mean DIVIDES its sum by C, as the graph's mean.dim does (portable
// op_mean.cpp: sum / num). A multiply by 1/C gives the same bits only where
// 1/C is exact, a power of two; at C = 3 it moves the mean by an ulp, and a
// column whose channels are nearly equal divides x - u by sqrt(eps) ~ 1e-3,
// which makes that ulp 1.3e-4 of the output.
void shim(KernelRuntimeContext& ctx, Span<EValue*> st) {
  const Tensor& tx = st[0]->toTensor();
  const Tensor& tw = st[1]->toTensor();
  const Tensor& tb = st[2]->toTensor();
  const float eps = (float)st[3]->toDouble();
  Tensor& tout = st[4]->toTensor();

  // Every refusal FAILS the instruction. Returning alone reports success, and
  // execute() then hands back whatever the planned output buffer held. The
  // dtype check matters most: const_data_ptr<float>() is an unchecked cast,
  // so an fp16 tensor would be read as fp32 and the output written to twice
  // its storage.
  if (tx.scalar_type() != ScalarType::Float ||
      tw.scalar_type() != ScalarType::Float ||
      tb.scalar_type() != ScalarType::Float ||
      tout.scalar_type() != ScalarType::Float) {
    printf("  [ln2d] expected fp32 tensors, got dtypes %d/%d/%d -> %d\n",
           (int)tx.scalar_type(), (int)tw.scalar_type(),
           (int)tb.scalar_type(), (int)tout.scalar_type());
    ctx.fail(Error::InvalidArgument);
    return;
  }
  if (tx.dim() != 4 || !same_shape(tx, tout) || !contiguous_tensor(tx) ||
      !contiguous_tensor(tout)) {
    printf("  [ln2d] expected a contiguous 4-D NCHW input and a same-shape "
           "output, got %d dims\n", (int)tx.dim());
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int B = (int)tx.size(0);
  const int C = (int)tx.size(1);
  const int HW = (int)tx.size(2) * (int)tx.size(3);
  if (C <= 0 || (int)tw.numel() != C || (int)tb.numel() != C ||
      !contiguous_tensor(tw) || !contiguous_tensor(tb)) {
    printf("  [ln2d] affine length %d/%d does not match C=%d\n",
           (int)tw.numel(), (int)tb.numel(), C);
    ctx.fail(Error::InvalidArgument);
    return;
  }

  const float* x = tx.const_data_ptr<float>();
  const float* w = tw.const_data_ptr<float>();
  const float* bs = tb.const_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();
  const float count = (float)C;

  for (int b = 0; b < B; ++b) {
    const float* xb = x + (size_t)b * C * HW;
    float* ob = out + (size_t)b * C * HW;
    for (int p = 0; p < HW; ++p) {
      float sum = 0.0f;
      for (int c = 0; c < C; ++c) {
        sum += xb[(size_t)c * HW + p];
      }
      const float u = sum / count;
      float sq = 0.0f;
      for (int c = 0; c < C; ++c) {
        const float d = xb[(size_t)c * HW + p] - u;
        sq += d * d;   // two roundings: this unit is -ffp-contract=off
      }
      // DIVIDE by sqrt, not multiply by an inverse: the graph this replaces
      // divides, and the two differ in the last ulp.
      const float den = sqrtf(sq / count + eps);
      for (int c = 0; c < C; ++c) {
        const size_t i = (size_t)c * HW + p;
        ob[i] = w[c] * ((xb[i] - u) / den) + bs[c];   // two roundings too
      }
      ++g_cols;
    }
  }
}

}  // namespace layer_norm2d

// The kernel's refusals, driven directly: no exporter emits a malformed
// layer_norm2d, so a program cannot reach these branches. Each case reports
// whether the kernel context recorded a failure. Buffers are oversized so a
// kernel that ignores the dtype writes into slack rather than past an end.
void run_ln2d_refusal_check() {
  using executorch::aten::TensorImpl;
  alignas(16) static float x[64], weight[4], bias[4], out[64];
  for (int i = 0; i < 64; ++i) x[i] = 0.125f * (float)(i % 7) - 0.25f;
  for (int i = 0; i < 4; ++i) {
    weight[i] = 1.0f;
    bias[i] = 0.0f;
  }
  auto refused = [&](ScalarType x_type, int x_dim) {
    TensorImpl::SizesType nchw[4] = {1, 4, 2, 2};
    TensorImpl::SizesType* x_sizes = nchw + (4 - x_dim);
    TensorImpl::DimOrderType order[4] = {0, 1, 2, 3};
    TensorImpl::StridesType nchw_strides[4] = {16, 4, 2, 1};
    TensorImpl::StridesType* x_strides = nchw_strides + (4 - x_dim);
    TensorImpl::SizesType c_size[1] = {4};
    TensorImpl::DimOrderType c_order[1] = {0};
    TensorImpl::StridesType c_stride[1] = {1};
    TensorImpl xi(x_type, x_dim, x_sizes, x, order, x_strides);
    TensorImpl oi(x_type, x_dim, x_sizes, out, order, x_strides);
    TensorImpl wi(ScalarType::Float, 1, c_size, weight, c_order, c_stride);
    TensorImpl bi(ScalarType::Float, 1, c_size, bias, c_order, c_stride);
    EValue ex{Tensor(&xi)}, ew{Tensor(&wi)}, eb{Tensor(&bi)};
    EValue eeps{1e-6}, eo{Tensor(&oi)};
    EValue* stack[5] = {&ex, &ew, &eb, &eeps, &eo};
    KernelRuntimeContext ctx;
    layer_norm2d::shim(ctx, Span<EValue*>(stack, 5));
    return ctx.failure_state() != Error::Ok;
  };
  const bool control = refused(ScalarType::Float, 4);
  const bool half = refused(ScalarType::Half, 4);
  const bool rank3 = refused(ScalarType::Float, 3);
  printf("  ln2d refusal: control=%s fp16=%s rank3=%s\n",
         control ? "refused" : "accepted", half ? "refused" : "accepted",
         rank3 ? "refused" : "accepted");
}

}  // namespace espressif
