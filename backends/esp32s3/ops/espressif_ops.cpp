/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Espressif model operators and target-specialized portable-kernel overrides.
// KernelRuntimeContext/Span<EValue*> shims use schema argument order, with
// the out tensor last. Registration is explicit so static archives retain it.

#include "espressif_ops.h"
#include "newlib_expf.h"
#include "quant_rules.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>

#include "Esp32KernelVariant.h"
#include "elementwise.h"
#include "fast_math.h"
#include "kv_affine.h"
#include "layer_norm2d.h"
#include "qdq_pool.h"
#include "qnorm.h"
#include "quant_lut.h"
#include "quant_lut16.h"
#include "esp32_hot.h"

// Registration and implementation are generated/configured in two different
// build steps. Refuse the two dangerous mixed states: no LayerNorm registration,
// or both the portable and target registrations for the same operator.
#if defined(ESP32_FAST_LAYERNORM) && !ESP32_GENERATED_FAST_LAYERNORM
#error "fast LayerNorm compile requested without matching kernel codegen"
#elif !defined(ESP32_FAST_LAYERNORM) && ESP32_GENERATED_FAST_LAYERNORM
#error "fast LayerNorm codegen requested without matching compile option"
#endif

// One switch per operator the library registers (gen_kernels.py
// LIBRARY_OPS): Esp32KernelVariant.h defines ESP32_GENERATED_OP_<NS>_<NAME>
// as 1 where one of the firmware's programs names the operator, so a
// firmware registers -- and so links -- only the kernels its programs run.
// The host's variant header registers every one. A variant header written
// before these switches would register nothing silently: refused.
#if defined(ESP32_GENERATED_OPS_ALL) && ESP32_GENERATED_OPS_ALL
#define ESP32_OP(name) 1
#elif defined(ESP32_GENERATED_OPS_SELECTED)
#define ESP32_OP(name) ESP32_GENERATED_OP_##name
#else
#error "Esp32KernelVariant.h predates per-operator registration: rerun gen_kernels.py"
#endif

using executorch::aten::IntArrayRef;
using executorch::aten::Scalar;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Span;

// Selected standard aten operators are registered to target-specialized shims
// below, each where a program names it (ESP32_OP). Unsupported shapes/dtypes
// fall through to ExecuTorch's exact portable implementations, which
// gen_kernels.py compiles while it suppresses their generated registration
// entries, to avoid duplicate operators.
namespace torch {
namespace executor {
namespace native {
#ifdef ESP32_FAST_LAYERNORM
std::tuple<Tensor&, Tensor&, Tensor&> native_layer_norm_out(
    KernelRuntimeContext&, const Tensor&, IntArrayRef,
    const std::optional<Tensor>&, const std::optional<Tensor>&, double,
    Tensor&, Tensor&, Tensor&);
#endif
Tensor& softmax_out(
    KernelRuntimeContext&, const Tensor&, int64_t, bool, Tensor&);
Tensor& add_out(
    KernelRuntimeContext&, const Tensor&, const Tensor&, const Scalar&, Tensor&);
Tensor& bmm_out(
    KernelRuntimeContext&, const Tensor&, const Tensor&, Tensor&);
Tensor& div_out(
    KernelRuntimeContext&, const Tensor&, const Tensor&, Tensor&);
Tensor& index_put_(
    KernelRuntimeContext&, Tensor&,
    executorch::aten::ArrayRef<std::optional<Tensor>>, const Tensor&, bool);
Tensor& le_tensor_out(
    KernelRuntimeContext&, const Tensor&, const Tensor&, Tensor&);
Tensor& mul_out(
    KernelRuntimeContext&, const Tensor&, const Tensor&, Tensor&);
Tensor& permute_copy_out(
    KernelRuntimeContext&, const Tensor&, IntArrayRef, Tensor&);
Tensor& remainder_Scalar_out(
    KernelRuntimeContext&, const Tensor&, const Scalar&, Tensor&);
void split_with_sizes_copy_out(
    KernelRuntimeContext&, const Tensor&, IntArrayRef, int64_t,
    executorch::aten::ArrayRef<Tensor>);
Tensor& where_out(
    KernelRuntimeContext&, const Tensor&, const Tensor&, const Tensor&, Tensor&);
// qdq_pool.h's fallbacks, linked where a program names them; for the checks.
Tensor& quantize_per_tensor_out(
    KernelRuntimeContext&, const Tensor&, double, int64_t, int64_t, int64_t,
    executorch::aten::ScalarType, Tensor&);
Tensor& dequantize_per_tensor_out(
    KernelRuntimeContext&, const Tensor&, double, int64_t, int64_t, int64_t,
    executorch::aten::ScalarType, std::optional<executorch::aten::ScalarType>,
    Tensor&);
Tensor& avg_pool2d_out(
    KernelRuntimeContext&, const Tensor&, IntArrayRef, IntArrayRef,
    IntArrayRef, bool, bool, std::optional<int64_t>, Tensor&);
}  // namespace native
}  // namespace executor
}  // namespace torch

static uint64_t g_add_same = 0;
static uint64_t g_add_leading_broadcast = 0;
static uint64_t g_add_fallback = 0;
static uint64_t g_permute_adjacent_block = 0;
static uint64_t g_permute_adjacent_element = 0;
static uint64_t g_permute_fallback = 0;
static uint64_t g_bmm_p_le_n = 0;
static uint64_t g_bmm_p_gt_n = 0;
static uint64_t g_bmm_fallback = 0;
static uint64_t g_mul_same = 0;
static uint64_t g_mul_broadcast = 0;
static uint64_t g_mul_scalar = 0;
static uint64_t g_mul_fallback = 0;
static uint64_t g_where_mask_scalar = 0;
static uint64_t g_where_fallback = 0;
static uint64_t g_div_scalar_denominator = 0;
static uint64_t g_div_fallback = 0;
static uint64_t g_le_scalar = 0;
static uint64_t g_le_fallback = 0;
static uint64_t g_index_put_rows = 0;
static uint64_t g_index_put_fallback = 0;
static uint64_t g_split_chunks = 0;
static uint64_t g_split_fallback = 0;
static uint64_t g_remainder_non_negative = 0;
static uint64_t g_remainder_fallback = 0;
static uint64_t g_softmax_last_dim = 0;
static uint64_t g_softmax_fallback = 0;
#ifdef ESP32_FAST_LAYERNORM
static uint64_t g_fast_layer_norm_calls = 0;
static uint64_t g_fast_layer_norm_rows = 0;
static uint64_t g_fast_layer_norm_fallback = 0;
#endif

ESP32_HOT_CODE
static bool same_shape(const Tensor& a, const Tensor& b) {
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

ESP32_HOT_CODE
static bool contiguous_tensor(const Tensor& tensor) {
  ssize_t expected = 1;
  for (ssize_t dim = tensor.dim(); dim-- > 0;) {
    if (tensor.size(dim) > 1 && tensor.strides()[dim] != expected) {
      return false;
    }
    expected *= tensor.size(dim);
  }
  return true;
}

// Contiguous strides and the default dim order. Strides alone are not enough
// for an exact fallback contract: a size-1 axis may carry any stride, and the
// portable kernels refuse operands whose dim orders disagree (a channels-last
// [1,C,1,1] times a 0-d tensor fails there), so a fast path must refuse too.
ESP32_HOT_CODE
static bool dense_tensor(const Tensor& tensor) {
  const auto order = tensor.dim_order();
  for (size_t dim = 0; dim < order.size(); ++dim) {
    if (order[dim] != dim) {
      return false;
    }
  }
  return contiguous_tensor(tensor);
}

// Whether `small` broadcasts over only leading dimensions of `big`. For
// [1,H,Q,K] + [1,1,Q,K], the small contiguous Q*K block simply repeats H
// times; no per-element multidimensional indexer is needed.
static bool leading_broadcast(const Tensor& big, const Tensor& small) {
  if (big.dim() < small.dim() || small.numel() <= 0 ||
      small.numel() > big.numel()) {
    return false;
  }
  const ssize_t shift = big.dim() - small.dim();
  ssize_t trailing = small.dim();
  while (trailing > 0 &&
         small.size(trailing - 1) == big.size(shift + trailing - 1)) {
    --trailing;
  }
  for (ssize_t dim = 0; dim < trailing; ++dim) {
    if (small.size(dim) != 1) {
      return false;
    }
  }
  return big.numel() % small.numel() == 0;
}

// Fast float32 alpha=1 cases used by transformer residual, position and
// additive-mask paths. This stays an aten override rather than a custom op, so
// ordinary exported graphs benefit without an architecture-specific FX pass.
static void shim_add_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 5) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& a = st[0]->toTensor();
  const Tensor& b = st[1]->toTensor();
  const Scalar& alpha = st[2]->toScalar();
  Tensor& out = st[3]->toTensor();
  const bool alpha_one =
      (alpha.isIntegral(false) && alpha.to<int64_t>() == 1) ||
      (alpha.isFloatingPoint() && alpha.to<double>() == 1.0) ||
      (alpha.isBoolean() && alpha.to<bool>());
  const bool floats =
      a.scalar_type() == executorch::aten::ScalarType::Float &&
      b.scalar_type() == executorch::aten::ScalarType::Float &&
      out.scalar_type() == executorch::aten::ScalarType::Float;
  const bool contiguous = contiguous_tensor(a) && contiguous_tensor(b) &&
      contiguous_tensor(out);
  const bool output_is_a = same_shape(out, a);

  if (alpha_one && floats && contiguous && output_is_a && same_shape(a, b)) {
    const float* lhs = a.const_data_ptr<float>();
    const float* rhs = b.const_data_ptr<float>();
    float* dst = out.mutable_data_ptr<float>();
    for (ssize_t i = 0; i < out.numel(); ++i) {
      dst[i] = lhs[i] + rhs[i];
    }
    ++g_add_same;
    return;
  }

  if (alpha_one && floats && contiguous && output_is_a &&
      leading_broadcast(a, b) &&
      out.mutable_data_ptr<float>() != b.const_data_ptr<float>()) {
    const float* large_data = a.const_data_ptr<float>();
    const float* small_data = b.const_data_ptr<float>();
    float* dst = out.mutable_data_ptr<float>();
    const ssize_t block = b.numel();
    for (ssize_t i = 0; i < out.numel(); ++i) {
      dst[i] = large_data[i] + small_data[i % block];
    }
    ++g_add_leading_broadcast;
    return;
  }

  ++g_add_fallback;
  torch::executor::native::add_out(ctx, a, b, alpha, out);
}

// Recognize one adjacent-axis swap in an otherwise identity permutation. This
// covers both [B,T,H,D] -> [B,H,T,D] and batched matrix transpose while keeping
// every other spelling on ET's portable validation/indexing implementation.
static bool adjacent_swap(
    const Tensor& input, IntArrayRef dims, const Tensor& output,
    ssize_t* swap_axis) {
  if (input.dim() != output.dim() ||
      dims.size() != static_cast<size_t>(input.dim())) {
    return false;
  }
  ssize_t found = -1;
  for (ssize_t dim = 0; dim < input.dim(); ++dim) {
    int64_t source_dim = dims[dim];
    if (source_dim < 0) {
      source_dim += input.dim();
    }
    if (source_dim == dim) {
      if (output.size(dim) != input.size(dim)) {
        return false;
      }
      continue;
    }
    if (found >= 0 || dim + 1 >= input.dim() || source_dim != dim + 1) {
      return false;
    }
    int64_t next_source = dims[dim + 1];
    if (next_source < 0) {
      next_source += input.dim();
    }
    if (next_source != dim || output.size(dim) != input.size(dim + 1) ||
        output.size(dim + 1) != input.size(dim)) {
      return false;
    }
    found = dim;
    ++dim;
  }
  if (found < 0) {
    return false;
  }
  *swap_axis = found;
  return true;
}

ESP32_HOT_CODE
static bool tensor_buffers_overlap(const Tensor& a, const Tensor& b) {
  const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  const uintptr_t a_end =
      a_begin + static_cast<size_t>(a.numel() * a.element_size());
  const uintptr_t b_end =
      b_begin + static_cast<size_t>(b.numel() * b.element_size());
  return a_begin < b_end && b_begin < a_end;
}

// Elementwise operators may legally write directly over an input. Accept that
// exact alias, or fully disjoint buffers, but never a shifted partial overlap:
// a flat forward loop would otherwise overwrite values it has not read yet.
ESP32_HOT_CODE
static bool tensor_output_alias_safe(const Tensor& input, const Tensor& output) {
  return input.const_data_ptr() == output.const_data_ptr() ||
      !tensor_buffers_overlap(input, output);
}

static bool is_broadcast_target(
    const Tensor& left, const Tensor& right, const Tensor& output) {
  const ssize_t rank = std::max(left.dim(), right.dim());
  if (output.dim() != rank) {
    return false;
  }
  const ssize_t left_shift = rank - left.dim();
  const ssize_t right_shift = rank - right.dim();
  for (ssize_t dim = 0; dim < rank; ++dim) {
    const ssize_t left_size =
        dim < left_shift ? 1 : left.size(dim - left_shift);
    const ssize_t right_size =
        dim < right_shift ? 1 : right.size(dim - right_shift);
    if (left_size != right_size && left_size != 1 && right_size != 1) {
      return false;
    }
    if (output.size(dim) != std::max(left_size, right_size)) {
      return false;
    }
  }
  return true;
}

// A broadcast input may alias the output only when it is actually the complete
// output tensor. A smaller exact-pointer view would be overwritten before its
// values were reused across the broadcast dimensions.
static bool broadcast_output_alias_safe(
    const Tensor& input, const Tensor& output) {
  if (!tensor_buffers_overlap(input, output)) {
    return true;
  }
  return input.const_data_ptr() == output.const_data_ptr() &&
      same_shape(input, output);
}

static void mul_contiguous_broadcast_f32(
    const Tensor& left, const Tensor& right, Tensor& output) {
  // Advance aligned input offsets with an odometer rather than delinearizing
  // every element through division/modulo. A zero step denotes a missing or
  // singleton broadcast dimension. ExecuTorch caps tensor rank at 16.
  constexpr size_t kMaxDims = 16;
  ssize_t coordinate[kMaxDims] = {};
  ssize_t left_step[kMaxDims] = {};
  ssize_t right_step[kMaxDims] = {};
  const ssize_t rank = output.dim();
  const ssize_t left_shift = rank - left.dim();
  const ssize_t right_shift = rank - right.dim();
  for (ssize_t dim = 0; dim < rank; ++dim) {
    if (dim >= left_shift && left.size(dim - left_shift) != 1) {
      left_step[dim] = left.strides()[dim - left_shift];
    }
    if (dim >= right_shift && right.size(dim - right_shift) != 1) {
      right_step[dim] = right.strides()[dim - right_shift];
    }
  }

  const float* left_data = left.const_data_ptr<float>();
  const float* right_data = right.const_data_ptr<float>();
  float* output_data = output.mutable_data_ptr<float>();
  ssize_t left_offset = 0;
  ssize_t right_offset = 0;
  for (ssize_t index = 0; index < output.numel(); ++index) {
    output_data[index] = left_data[left_offset] * right_data[right_offset];
    for (ssize_t dim = rank; dim-- > 0;) {
      ++coordinate[dim];
      if (coordinate[dim] < output.size(dim)) {
        left_offset += left_step[dim];
        right_offset += right_step[dim];
        break;
      }
      coordinate[dim] = 0;
      left_offset -= left_step[dim] * (output.size(dim) - 1);
      right_offset -= right_step[dim] * (output.size(dim) - 1);
    }
  }
}

// The exact LM's decomposed SiLU and gated MLP both reach same-shape fp32
// aten::mul.out. ET's generic implementation pays dtype dispatch plus a
// multidimensional broadcast indexer for each scalar. The flat path handles
// exact shapes; the scalar path a one-element operand; the odometer path
// handles arbitrary right-aligned contiguous broadcasting without
// per-element division or dtype dispatch. All other semantics remain in
// portable mul_out.
ESP32_HOT_CODE
static void shim_mul_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& left = st[0]->toTensor();
  const Tensor& right = st[1]->toTensor();
  Tensor& output = st[2]->toTensor();
  const bool floats =
      left.scalar_type() == executorch::aten::ScalarType::Float &&
      right.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float;
  const bool same = left.numel() > 0 && floats &&
      same_shape(left, right) && same_shape(left, output) &&
      dense_tensor(left) && dense_tensor(right) &&
      dense_tensor(output) && tensor_output_alias_safe(left, output) &&
      tensor_output_alias_safe(right, output);
  if (same) {
    const float* left_data = left.const_data_ptr<float>();
    const float* right_data = right.const_data_ptr<float>();
    float* output_data = output.mutable_data_ptr<float>();
    for (ssize_t index = 0; index < output.numel(); ++index) {
      output_data[index] = left_data[index] * right_data[index];
    }
    ++g_mul_same;
    return;
  }

  // A one-element operand of no higher rank than the other (the decode's 0-d
  // query, embedding and per-layer scales) leaves the other's shape as the
  // output's. The odometer below computes these too, but on the chip each
  // element pays an out-of-line Tensor::size() call and counter spills: ~70
  // cycles for one multiply. Here the scale is loaded once.
  //
  // Only a non-NaN scale: when both operands are NaN, which payload the
  // product keeps can depend on the operand order the FPU sees (IEEE 754
  // leaves it open; arm64 keeps the first), and compilers reorder a
  // commutative multiply freely (clang differently in one loop's vector body
  // and its tail). With one NaN operand at most, every order gives the same
  // bits; a NaN scale goes to the portable kernel, whose instructions are the
  // reference.
  const bool scalar_right = right.numel() == 1 && right.dim() <= left.dim();
  const bool scalar_left =
      !scalar_right && left.numel() == 1 && left.dim() <= right.dim();
  const Tensor& tensor = scalar_right ? left : right;
  const Tensor& scalar = scalar_right ? right : left;
  const bool one_element = (scalar_right || scalar_left) && floats &&
      tensor.numel() > 0 && same_shape(tensor, output) &&
      dense_tensor(tensor) && dense_tensor(scalar) && dense_tensor(output) &&
      tensor_output_alias_safe(tensor, output) &&
      !tensor_buffers_overlap(scalar, output);
  if (one_element) {
    const float scale = scalar.const_data_ptr<float>()[0];
    if (std::isnan(scale)) {
      ++g_mul_fallback;
      torch::executor::native::mul_out(ctx, left, right, output);
      return;
    }
    const float* tensor_data = tensor.const_data_ptr<float>();
    float* output_data = output.mutable_data_ptr<float>();
    const ssize_t count = output.numel();
    // Four elements a step, loads first: GCC at -Os neither unrolls nor
    // schedules for the LX7, and one element at a time stalls on its load and
    // its multiply (4 instructions an element here, 8 there). Each output is
    // still its own input times the scale, so in place stays exact too.
    ssize_t index = 0;
    for (; index + 4 <= count; index += 4) {
      const float a = tensor_data[index];
      const float b = tensor_data[index + 1];
      const float c = tensor_data[index + 2];
      const float d = tensor_data[index + 3];
      output_data[index] = a * scale;
      output_data[index + 1] = b * scale;
      output_data[index + 2] = c * scale;
      output_data[index + 3] = d * scale;
    }
    for (; index < count; ++index) {
      output_data[index] = tensor_data[index] * scale;
    }
    ++g_mul_scalar;
    return;
  }

  const bool broadcast = left.numel() > 0 && right.numel() > 0 &&
      output.numel() > 0 && output.dim() <= 16 && floats &&
      is_broadcast_target(left, right, output) &&
      dense_tensor(left) && dense_tensor(right) &&
      dense_tensor(output) &&
      broadcast_output_alias_safe(left, output) &&
      broadcast_output_alias_safe(right, output);
  if (broadcast) {
    mul_contiguous_broadcast_f32(left, right, output);
    ++g_mul_broadcast;
    return;
  }

  ++g_mul_fallback;
  torch::executor::native::mul_out(ctx, left, right, output);
}

static bool broadcasts_to(const Tensor& input, const Tensor& output) {
  if (input.dim() > output.dim()) {
    return false;
  }
  const ssize_t shift = output.dim() - input.dim();
  for (ssize_t dim = 0; dim < output.dim(); ++dim) {
    const ssize_t input_size =
        dim < shift ? 1 : input.size(dim - shift);
    if (input_size != 1 && input_size != output.size(dim)) {
      return false;
    }
  }
  return true;
}

static void where_contiguous_mask_scalar_f32(
    const Tensor& condition, const Tensor& value, const Tensor& other,
    Tensor& output) {
  constexpr size_t kMaxDims = 16;
  ssize_t coordinate[kMaxDims] = {};
  ssize_t condition_step[kMaxDims] = {};
  const ssize_t rank = output.dim();
  const ssize_t shift = rank - condition.dim();
  for (ssize_t dim = 0; dim < rank; ++dim) {
    if (dim >= shift && condition.size(dim - shift) != 1) {
      condition_step[dim] = condition.strides()[dim - shift];
    }
  }

  const bool* condition_data = condition.const_data_ptr<bool>();
  const float* value_data = value.const_data_ptr<float>();
  const float other_value = other.const_data_ptr<float>()[0];
  float* output_data = output.mutable_data_ptr<float>();
  ssize_t condition_offset = 0;
  for (ssize_t index = 0; index < output.numel(); ++index) {
    output_data[index] =
        condition_data[condition_offset] ? value_data[index] : other_value;
    for (ssize_t dim = rank; dim-- > 0;) {
      ++coordinate[dim];
      if (coordinate[dim] < output.size(dim)) {
        condition_offset += condition_step[dim];
        break;
      }
      coordinate[dim] = 0;
      condition_offset -=
          condition_step[dim] * (output.size(dim) - 1);
    }
  }
}

// Glimmer's causal mask is bool [1,1,S] over fp32 scores [1,H,S], with a
// scalar -inf false value. The portable tritensor iterator repeatedly performs
// dtype and broadcast-index work for this simple form. Keep the specialization
// deliberately narrow; every other where.self_out reaches ET unchanged.
static void shim_where_self_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 5) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& condition = st[0]->toTensor();
  const Tensor& value = st[1]->toTensor();
  const Tensor& other = st[2]->toTensor();
  Tensor& output = st[3]->toTensor();
  const bool types =
      condition.scalar_type() == executorch::aten::ScalarType::Bool &&
      value.scalar_type() == executorch::aten::ScalarType::Float &&
      other.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float;
  const bool fast = condition.numel() > 0 && value.numel() > 0 &&
      other.numel() == 1 && output.numel() > 0 && output.dim() <= 16 &&
      types && same_shape(value, output) && broadcasts_to(condition, output) &&
      contiguous_tensor(condition) && contiguous_tensor(value) &&
      contiguous_tensor(other) && contiguous_tensor(output) &&
      tensor_output_alias_safe(value, output) &&
      !tensor_buffers_overlap(condition, output) &&
      !tensor_buffers_overlap(other, output);
  if (fast) {
    where_contiguous_mask_scalar_f32(condition, value, other, output);
    ++g_where_mask_scalar;
    return;
  }

  ++g_where_fallback;
  torch::executor::native::where_out(ctx, condition, value, other, output);
}

// Attention scaling exports as fp32 tensor / zero-dimensional fp32 tensor.
// Keep real division in the loop (rather than precomputing a reciprocal) so
// each result has the same IEEE rounding boundary as portable div_out.
static void shim_div_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& numerator = st[0]->toTensor();
  const Tensor& denominator = st[1]->toTensor();
  Tensor& output = st[2]->toTensor();
  const bool floats =
      numerator.scalar_type() == executorch::aten::ScalarType::Float &&
      denominator.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float;
  const bool fast = numerator.numel() > 0 && denominator.dim() == 0 &&
      denominator.numel() == 1 && floats && same_shape(numerator, output) &&
      contiguous_tensor(numerator) && contiguous_tensor(denominator) &&
      contiguous_tensor(output) &&
      tensor_output_alias_safe(numerator, output) &&
      !tensor_buffers_overlap(denominator, output);
  if (!fast) {
    ++g_div_fallback;
    torch::executor::native::div_out(ctx, numerator, denominator, output);
    return;
  }

  const float* numerator_data = numerator.const_data_ptr<float>();
  const float divisor = denominator.const_data_ptr<float>()[0];
  float* output_data = output.mutable_data_ptr<float>();
  for (ssize_t index = 0; index < output.numel(); ++index) {
    output_data[index] = numerator_data[index] / divisor;
  }
  ++g_div_scalar_denominator;
}

// The decode's attention masks compare a constant int64 position table with
// the current 0-d position: int64 [N] <= int64 0-d -> bool [N], twice per
// step. The portable kernel's generic loop (a function-pointer load, load,
// store and broadcast-iterator step per element) takes ~5 us per element on
// the chip. Other dtypes, broadcasts and layouts stay portable.
ESP32_HOT_CODE
static void shim_le_tensor_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& self = st[0]->toTensor();
  const Tensor& other = st[1]->toTensor();
  Tensor& out = st[2]->toTensor();
  const bool fast =
      self.scalar_type() == executorch::aten::ScalarType::Long &&
      other.scalar_type() == executorch::aten::ScalarType::Long &&
      out.scalar_type() == executorch::aten::ScalarType::Bool &&
      self.numel() > 0 && other.numel() == 1 && other.dim() <= self.dim() &&
      same_shape(self, out) && dense_tensor(self) && dense_tensor(other) &&
      dense_tensor(out) && !tensor_buffers_overlap(self, out) &&
      !tensor_buffers_overlap(other, out);
  if (!fast) {
    ++g_le_fallback;
    torch::executor::native::le_tensor_out(ctx, self, other, out);
    return;
  }

  const int64_t* values = self.const_data_ptr<int64_t>();
  const int64_t bound = other.const_data_ptr<int64_t>()[0];
  bool* result = out.mutable_data_ptr<bool>();
  for (ssize_t index = 0; index < out.numel(); ++index) {
    result[index] = values[index] <= bound;
  }
  ++g_le_scalar;
}

// The decode writes each layer's new K/V rows into its persistent caches in
// place: index_put_(cache [1,S,...], [None, int64 [1]], row, accumulate=False),
// three per layer. The portable kernel already ends in a row memcpy, but its
// checks first print both shapes into strings (snprintf, with ET_LOG on):
// 36-185 us a call on the chip. These are the same checks, unformatted, and
// then the same copies in the same order (the last of repeated indices
// wins, as there). As there since #23529, an index may count from the end
// (-1 is the last row), and values may leave out leading size-1 dimensions
// of self's shape (a Mistral sliding-window cache write passes [2,1,8] for
// [1,2,1,8]); neither changes the bytes copied. Anything outside them
// (accumulate, several or out-of-range indices, int32 indices, other
// layouts) stays portable.
ESP32_HOT_CODE
static void shim_index_put_(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 5) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  Tensor& self = st[0]->toTensor();
  const auto indices = st[1]->toListOptionalTensor();
  const Tensor& values = st[2]->toTensor();
  const bool accumulate = st[3]->toBool();
  bool fast = !accumulate && self.numel() > 0 &&
      indices.size() <= static_cast<size_t>(self.dim()) &&
      values.scalar_type() == self.scalar_type() &&
      values.dim() <= self.dim() && dense_tensor(self) &&
      dense_tensor(values) && !tensor_buffers_overlap(values, self);
  ssize_t axis = -1;
  for (size_t position = 0; fast && position < indices.size(); ++position) {
    if (indices[position].has_value()) {
      fast = axis < 0;  // exactly one index tensor
      axis = static_cast<ssize_t>(position);
    }
  }
  const Tensor* index = fast && axis >= 0 ? &indices[axis].value() : nullptr;
  fast = index != nullptr &&
      index->scalar_type() == executorch::aten::ScalarType::Long &&
      index->dim() == 1 && index->numel() > 0 && dense_tensor(*index) &&
      !tensor_buffers_overlap(*index, self);
  // values' shape is self's with the index's length on the axis, less the
  // first `omitted` dimensions, which must each be 1 there.
  const ssize_t omitted = fast ? self.dim() - values.dim() : 0;
  for (ssize_t dim = 0; fast && dim < self.dim(); ++dim) {
    const ssize_t expected = dim == axis ? index->size(0) : self.size(dim);
    fast = dim < omitted ? expected == 1
                         : values.size(dim - omitted) == expected;
  }
  const int64_t* rows = fast ? index->const_data_ptr<int64_t>() : nullptr;
  const int64_t length = fast ? static_cast<int64_t>(self.size(axis)) : 0;
  for (ssize_t row = 0; fast && row < index->numel(); ++row) {
    fast = rows[row] >= -length && rows[row] < length;
  }
  if (!fast) {
    ++g_index_put_fallback;
    Tensor& result = torch::executor::native::index_put_(
        ctx, self, indices, values, accumulate);
    *st[4] = EValue(result);  // as the generated portable shim returns it
    return;
  }

  size_t leading = 1;
  size_t row_bytes = self.element_size();
  for (ssize_t dim = 0; dim < self.dim(); ++dim) {
    if (dim < axis) {
      leading *= static_cast<size_t>(self.size(dim));
    } else if (dim > axis) {
      row_bytes *= static_cast<size_t>(self.size(dim));
    }
  }
  const size_t self_rows = static_cast<size_t>(length);
  const size_t count = static_cast<size_t>(index->numel());
  const char* source = values.const_data_ptr<char>();
  char* cache = self.mutable_data_ptr<char>();
  for (size_t outer = 0; outer < leading; ++outer) {
    for (size_t row = 0; row < count; ++row) {
      const int64_t position = rows[row] < 0 ? rows[row] + length : rows[row];
      memcpy(cache + (outer * self_rows + static_cast<size_t>(position)) *
                         row_bytes,
             source + (outer * count + row) * row_bytes, row_bytes);
    }
  }
  ++g_index_put_rows;
  *st[4] = EValue(self);
}

// The decode's per-layer embeddings leave the delegate as one fp32 [1,8,256]
// split on dim 1 into eight [1,1,256] pieces. The portable kernel copies them
// an element at a time through two dtype switches; these are the same fp32
// bytes by memcpy, one per piece and leading index in portable's order.
// Other dtypes, conversions, layouts and malformed splits stay portable.
ESP32_HOT_CODE
static void shim_split_with_sizes_copy_out(
    KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 5) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& self = st[0]->toTensor();
  const IntArrayRef split_sizes = st[1]->toIntList();
  const int64_t dim = st[2]->toInt();
  const auto out = st[3]->toTensorList();
  const int64_t axis = dim < 0 ? dim + self.dim() : dim;
  bool fast = axis >= 0 && axis < self.dim() &&
      self.scalar_type() == executorch::aten::ScalarType::Float &&
      dense_tensor(self) && !out.empty() && split_sizes.size() == out.size();
  int64_t total = 0;
  for (size_t piece = 0; fast && piece < out.size(); ++piece) {
    const Tensor& target = out[piece];
    fast = split_sizes[piece] >= 0 &&
        target.scalar_type() == executorch::aten::ScalarType::Float &&
        target.dim() == self.dim() && dense_tensor(target) &&
        !tensor_buffers_overlap(target, self);
    for (ssize_t d = 0; fast && d < self.dim(); ++d) {
      fast = target.size(d) == (d == axis ? split_sizes[piece] : self.size(d));
    }
    total += split_sizes[piece];
  }
  if (!fast || total != self.size(axis)) {
    ++g_split_fallback;
    torch::executor::native::split_with_sizes_copy_out(
        ctx, self, split_sizes, dim, out);
    st[4] = st[3];  // as the generated portable shim returns it
    return;
  }

  size_t leading = 1;
  size_t trailing = 1;
  for (ssize_t d = 0; d < self.dim(); ++d) {
    if (d < axis) {
      leading *= static_cast<size_t>(self.size(d));
    } else if (d > axis) {
      trailing *= static_cast<size_t>(self.size(d));
    }
  }
  const size_t step = static_cast<size_t>(self.size(axis)) * trailing;
  const float* source = self.const_data_ptr<float>();
  for (size_t piece = 0; piece < out.size(); ++piece) {
    const size_t chunk = static_cast<size_t>(split_sizes[piece]) * trailing;
    float* target = out[piece].mutable_data_ptr<float>();
    for (size_t outer = 0; chunk > 0 && outer < leading; ++outer) {
      memcpy(target + outer * chunk, source + outer * step,
             chunk * sizeof(float));
    }
    source += chunk;
  }
  ++g_split_chunks;
  st[4] = st[3];
}

// The decode's sliding-window slot: int64 0-d position % window, once a
// step, 77 us through the portable kernel's generic loop. Only a
// non-negative dividend and a positive integral divisor: there the C++ %
// that ExecuTorch's portable kernel computes agrees with ATen's floor-mod
// sign rule as well. With a negative operand the two differ (portable gives
// -3 % 64 == -3, torch.remainder 61), and such inputs stay with the portable
// kernel, which is this override's contract.
ESP32_HOT_CODE
static void shim_remainder_scalar_out(
    KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& self = st[0]->toTensor();
  const Scalar& other = st[1]->toScalar();
  Tensor& out = st[2]->toTensor();
  const int64_t divisor = other.isIntegral(false) ? other.to<int64_t>() : 0;
  bool fast = divisor > 0 &&
      self.scalar_type() == executorch::aten::ScalarType::Long &&
      out.scalar_type() == executorch::aten::ScalarType::Long &&
      self.numel() > 0 && same_shape(self, out) && dense_tensor(self) &&
      dense_tensor(out) && tensor_output_alias_safe(self, out);
  const int64_t* values = fast ? self.const_data_ptr<int64_t>() : nullptr;
  for (ssize_t index = 0; fast && index < self.numel(); ++index) {
    fast = values[index] >= 0;
  }
  if (!fast) {
    ++g_remainder_fallback;
    torch::executor::native::remainder_Scalar_out(ctx, self, other, out);
    return;
  }

  int64_t* result = out.mutable_data_ptr<int64_t>();
  for (ssize_t index = 0; index < out.numel(); ++index) {
    result[index] = values[index] % divisor;
  }
  ++g_remainder_non_negative;
}

// newlib's expf and its host transcription: newlib_expf.h.
// Contiguous last-axis fp32 softmax. ET's portable kernel evaluates exp(x-max)
// once while summing and then evaluates the same exp a second time while
// writing the output. Store the first deterministic result in the output and
// reuse it for the division pass: identical arithmetic, half as many expf
// calls. Other axes, dtypes and layouts retain the portable implementation.
static void shim_softmax_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 5) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& input = st[0]->toTensor();
  const int64_t dim = st[1]->toInt();
  const bool half_to_float = st[2]->toBool();
  Tensor& output = st[3]->toTensor();
  const int64_t normalized_dim =
      dim < 0 ? dim + static_cast<int64_t>(input.dim()) : dim;
  const bool last_axis = input.dim() > 0 &&
      normalized_dim == static_cast<int64_t>(input.dim()) - 1;
  const ssize_t width = last_axis ? input.size(input.dim() - 1) : 0;
  const bool fast = !half_to_float && width > 0 && input.numel() > 0 &&
      input.numel() % width == 0 && same_shape(input, output) &&
      input.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float &&
      contiguous_tensor(input) && contiguous_tensor(output) &&
      !tensor_buffers_overlap(input, output);
  if (!fast) {
    ++g_softmax_fallback;
    torch::executor::native::softmax_out(
        ctx, input, dim, half_to_float, output);
    return;
  }

  const float* input_data = input.const_data_ptr<float>();
  float* output_data = output.mutable_data_ptr<float>();
  const ssize_t rows = input.numel() / width;
  for (ssize_t row = 0; row < rows; ++row) {
    const float* row_input = input_data + row * width;
    float* row_output = output_data + row * width;
    float max_input = row_input[0];
    for (ssize_t column = 1; column < width; ++column) {
      max_input = std::max(row_input[column], max_input);
    }

    float mapped = espressif_softmax_expf(row_input[0] - max_input);
    row_output[0] = mapped;
    float sum = mapped;
    for (ssize_t column = 1; column < width; ++column) {
      mapped = espressif_softmax_expf(row_input[column] - max_input);
      row_output[column] = mapped;
      sum += mapped;
    }
    for (ssize_t column = 0; column < width; ++column) {
      row_output[column] = row_output[column] / sum;
    }
  }
  ++g_softmax_last_dim;
}

// Explicit approximate-performance rung: the portable kernel's reductions,
// but one reciprocal per row multiplies each output where portable divides
// it by the standard deviation. The reductions are ExecuTorch 1.6's
// (layer_norm_scalar in normalization_ops_util.h): eight independent partial
// sums, then the deviations from that mean, whose own sum corrects it, so a
// row far from zero keeps its variance where E[x^2] - mean^2 cancelled it
// away. Written out the same way, the mean and rstd are portable's bits;
// run_fast_layer_norm_check holds the row to that. The registration is
// opt-in at code generation and compile time, so the default bit-exact build
// cannot silently inherit different arithmetic; the row function is built
// in every variant so the check runs on the host and in every parity build.
__attribute__((noinline)) static void fast_layer_norm_f32(
    const float* input_data, const float* weight_data, const float* bias_data,
    float* output_data, float* mean_data, float* rstd_data, size_t rows,
    size_t width, float eps) {
  const float count = static_cast<float>(width);
  constexpr size_t kLanes = 8;
  const size_t vectorized_end = width - width % kLanes;
  for (size_t row = 0; row < rows; ++row) {
    const float* x = input_data + row * width;
    float* y = output_data + row * width;

    std::array<float, kLanes> sums{};
    for (size_t column = 0; column < vectorized_end; column += kLanes) {
      for (size_t lane = 0; lane < kLanes; ++lane) {
        sums[lane] += x[column + lane];
      }
    }
    float sum = std::accumulate(sums.begin(), sums.end(), 0.0f);
    for (size_t column = vectorized_end; column < width; ++column) {
      sum += x[column];
    }
    float mean_value = sum / count;

    std::array<float, kLanes> deviations{};
    std::array<float, kLanes> squares{};
    for (size_t column = 0; column < vectorized_end; column += kLanes) {
      for (size_t lane = 0; lane < kLanes; ++lane) {
        const float d = x[column + lane] - mean_value;
        deviations[lane] += d;
        squares[lane] += d * d;
      }
    }
    float d_sum = std::accumulate(deviations.begin(), deviations.end(), 0.0f);
    float sq_sum = std::accumulate(squares.begin(), squares.end(), 0.0f);
    for (size_t column = vectorized_end; column < width; ++column) {
      const float d = x[column] - mean_value;
      d_sum += d;
      sq_sum += d * d;
    }
    const float correction = d_sum / count;
    const float variance =
        std::max((sq_sum - d_sum * correction) / count, 0.0f);
    mean_value += correction;
    const float inverse_standard_deviation =
        1.0f / std::sqrt(variance + eps);

    for (size_t column = 0; column < width; ++column) {
      const float w = weight_data ? weight_data[column] : 1.0f;
      const float b = bias_data ? bias_data[column] : 0.0f;
      y[column] =
          ((x[column] - mean_value) * inverse_standard_deviation) * w + b;
    }
    mean_data[row] = mean_value;
    rstd_data[row] = inverse_standard_deviation;
  }
}

#ifdef ESP32_FAST_LAYERNORM
static void shim_fast_layer_norm_out(
    KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 9) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& input = st[0]->toTensor();
  const IntArrayRef normalized_shape = st[1]->toIntList();
  const auto weight = st[2]->toOptional<Tensor>();
  const auto bias = st[3]->toOptional<Tensor>();
  const double eps = st[4]->toDouble();
  Tensor& output = st[5]->toTensor();
  Tensor& mean = st[6]->toTensor();
  Tensor& rstd = st[7]->toTensor();

  const bool one_axis = input.dim() > 0 && normalized_shape.size() == 1 &&
      normalized_shape[0] > 0 &&
      normalized_shape[0] == input.size(input.dim() - 1);
  const ssize_t width = one_axis ? normalized_shape[0] : 0;
  const ssize_t rows = width > 0 ? input.numel() / width : 0;
  const bool floats =
      input.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float &&
      mean.scalar_type() == executorch::aten::ScalarType::Float &&
      rstd.scalar_type() == executorch::aten::ScalarType::Float &&
      (!weight.has_value() || weight->scalar_type() ==
          executorch::aten::ScalarType::Float) &&
      (!bias.has_value() || bias->scalar_type() ==
          executorch::aten::ScalarType::Float);
  bool statistic_shapes = one_axis &&
      mean.dim() == input.dim() && rstd.dim() == input.dim();
  if (statistic_shapes) {
    for (ssize_t dimension = 0; dimension < input.dim(); ++dimension) {
      const ssize_t expected =
          dimension == input.dim() - 1 ? 1 : input.size(dimension);
      if (mean.size(dimension) != expected ||
          rstd.size(dimension) != expected) {
        statistic_shapes = false;
        break;
      }
    }
  }
  const bool shapes = one_axis && rows > 0 &&
      input.numel() == rows * width && same_shape(input, output) &&
      statistic_shapes && mean.numel() == rows && rstd.numel() == rows &&
      (!weight.has_value() ||
       (weight->dim() == 1 && weight->size(0) == width)) &&
      (!bias.has_value() ||
       (bias->dim() == 1 && bias->size(0) == width));
  const bool contiguous = shapes && contiguous_tensor(input) &&
      contiguous_tensor(output) && contiguous_tensor(mean) &&
      contiguous_tensor(rstd) &&
      (!weight.has_value() || contiguous_tensor(*weight)) &&
      (!bias.has_value() || contiguous_tensor(*bias));
  const bool default_order = contiguous &&
      torch::executor::tensor_is_default_dim_order(input) &&
      torch::executor::tensors_have_same_dim_order(
          input, output, mean, rstd) &&
      (!weight.has_value() ||
       torch::executor::tensors_have_same_dim_order(input, *weight)) &&
      (!bias.has_value() ||
       torch::executor::tensors_have_same_dim_order(input, *bias));
  const bool disjoint = default_order &&
      !tensor_buffers_overlap(input, output) &&
      !tensor_buffers_overlap(mean, input) &&
      !tensor_buffers_overlap(mean, output) &&
      !tensor_buffers_overlap(rstd, input) &&
      !tensor_buffers_overlap(rstd, output) &&
      !tensor_buffers_overlap(mean, rstd) &&
      (!weight.has_value() ||
       (!tensor_buffers_overlap(*weight, output) &&
        !tensor_buffers_overlap(*weight, mean) &&
        !tensor_buffers_overlap(*weight, rstd))) &&
      (!bias.has_value() ||
       (!tensor_buffers_overlap(*bias, output) &&
        !tensor_buffers_overlap(*bias, mean) &&
        !tensor_buffers_overlap(*bias, rstd)));
  if (!(floats && disjoint)) {
    ++g_fast_layer_norm_fallback;
    torch::executor::native::native_layer_norm_out(
        ctx, input, normalized_shape, weight, bias, eps,
        output, mean, rstd);
    return;
  }

  const float* input_data = input.const_data_ptr<float>();
  const float* weight_data =
      weight.has_value() ? weight->const_data_ptr<float>() : nullptr;
  const float* bias_data =
      bias.has_value() ? bias->const_data_ptr<float>() : nullptr;
  float* output_data = output.mutable_data_ptr<float>();
  float* mean_data = mean.mutable_data_ptr<float>();
  float* rstd_data = rstd.mutable_data_ptr<float>();
  const float eps_float = static_cast<float>(eps);
  fast_layer_norm_f32(
      input_data, weight_data, bias_data, output_data, mean_data, rstd_data,
      static_cast<size_t>(rows), static_cast<size_t>(width), eps_float);
  ++g_fast_layer_norm_calls;
  g_fast_layer_norm_rows += static_cast<uint64_t>(rows);
}
#endif

// A contiguous adjacent transpose consists of outer*left*right contiguous
// trailing blocks. ET's generic kernel recomputes a full coordinate and source
// index for every scalar; this loop does that arithmetic once per block.
static void shim_permute_copy_out(
    KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& input = st[0]->toTensor();
  const IntArrayRef dims = st[1]->toIntList();
  Tensor& output = st[2]->toTensor();
  ssize_t axis = -1;
  const bool fast =
      input.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float &&
      contiguous_tensor(input) && contiguous_tensor(output) &&
      !tensor_buffers_overlap(input, output) &&
      adjacent_swap(input, dims, output, &axis);
  if (!fast) {
    ++g_permute_fallback;
    torch::executor::native::permute_copy_out(ctx, input, dims, output);
    return;
  }

  size_t outer = 1;
  size_t inner = 1;
  for (ssize_t dim = 0; dim < axis; ++dim) {
    outer *= static_cast<size_t>(input.size(dim));
  }
  for (ssize_t dim = axis + 2; dim < input.dim(); ++dim) {
    inner *= static_cast<size_t>(input.size(dim));
  }
  const size_t left = static_cast<size_t>(input.size(axis));
  const size_t right = static_cast<size_t>(input.size(axis + 1));
  const float* src = input.const_data_ptr<float>();
  float* dst = output.mutable_data_ptr<float>();
  for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (size_t right_index = 0; right_index < right; ++right_index) {
      for (size_t left_index = 0; left_index < left; ++left_index) {
        const size_t src_base =
            ((outer_index * left + left_index) * right + right_index) * inner;
        const size_t dst_base =
            ((outer_index * right + right_index) * left + left_index) * inner;
        if (inner == 1) {
          dst[dst_base] = src[src_base];
        } else {
          memcpy(dst + dst_base, src + src_base, inner * sizeof(float));
        }
      }
    }
  }
  if (inner == 1) {
    ++g_permute_adjacent_element;
  } else {
    ++g_permute_adjacent_block;
  }
}

// Contiguous fp32 BMM with four output columns live at once. Each accumulator
// visits k in the same order as ET's portable i/j/k loop, loading the
// left-hand value once for four columns. Each step is std::fma, one rounding:
// GCC's default contraction made a madd.s of `sum += lhs * rhs` on the chip
// (five in this function), while the host runner builds with
// -ffp-contract=off and rounded twice, so the host and test/reference.py's
// bmm_f32_portable disagreed with the chip on prefill's P@V. Spelled out, the
// chip's arithmetic is the same on both, whatever the flags.
static void shim_bmm_out(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& left = st[0]->toTensor();
  const Tensor& right = st[1]->toTensor();
  Tensor& output = st[2]->toTensor();
  const bool shapes =
      left.dim() == 3 && right.dim() == 3 && output.dim() == 3 &&
      left.size(0) == right.size(0) && left.size(2) == right.size(1) &&
      output.size(0) == left.size(0) &&
      output.size(1) == left.size(1) &&
      output.size(2) == right.size(2);
  const bool floats =
      left.scalar_type() == executorch::aten::ScalarType::Float &&
      right.scalar_type() == executorch::aten::ScalarType::Float &&
      output.scalar_type() == executorch::aten::ScalarType::Float;
  const bool fast = shapes && floats && contiguous_tensor(left) &&
      contiguous_tensor(right) && contiguous_tensor(output) &&
      !tensor_buffers_overlap(left, output) &&
      !tensor_buffers_overlap(right, output);
  if (!fast) {
    ++g_bmm_fallback;
    torch::executor::native::bmm_out(ctx, left, right, output);
    return;
  }

  const size_t batches = static_cast<size_t>(left.size(0));
  const size_t m = static_cast<size_t>(left.size(1));
  const size_t n = static_cast<size_t>(left.size(2));
  const size_t p = static_cast<size_t>(right.size(2));
  const float* left_data = left.const_data_ptr<float>();
  const float* right_data = right.const_data_ptr<float>();
  float* output_data = output.mutable_data_ptr<float>();
  for (size_t batch = 0; batch < batches; ++batch) {
    const float* batch_left = left_data + batch * m * n;
    const float* batch_right = right_data + batch * n * p;
    float* batch_output = output_data + batch * m * p;
    for (size_t row = 0; row < m; ++row) {
      const float* row_left = batch_left + row * n;
      size_t column = 0;
      for (; column + 4 <= p; column += 4) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (size_t inner = 0; inner < n; ++inner) {
          const float lhs = row_left[inner];
          const float* rhs = batch_right + inner * p + column;
          sum0 = std::fma(lhs, rhs[0], sum0);
          sum1 = std::fma(lhs, rhs[1], sum1);
          sum2 = std::fma(lhs, rhs[2], sum2);
          sum3 = std::fma(lhs, rhs[3], sum3);
        }
        float* dst = batch_output + row * p + column;
        dst[0] = sum0;
        dst[1] = sum1;
        dst[2] = sum2;
        dst[3] = sum3;
      }
      for (; column < p; ++column) {
        float sum = 0.0f;
        for (size_t inner = 0; inner < n; ++inner) {
          sum = std::fma(row_left[inner], batch_right[inner * p + column], sum);
        }
        batch_output[row * p + column] = sum;
      }
    }
  }
  if (p <= n) {
    ++g_bmm_p_le_n;
  } else {
    ++g_bmm_p_gt_n;
  }
}

// ---- fused RMSNorm, backed by rsqrt0.s ------------------------------------
// The five-op decomposition (pow, mean, add, rsqrt, two mul) is fused AOT by
// fuse_rmsnorm_pass.py into espressif::rmsnorm; this is where it lands. The reason to
// bother is not flops -- one norm over D=96 is ~300 of them -- it is that this
// toolchain's GCC has no sqrtsf2/divsf3 pattern for Xtensa, so C's 1.0f/sqrtf(x)
// becomes two soft-float library calls and the hardware rsqrt0.s is unreachable
// from C. See the header of rmsnorm_s3.S for the Newton-Raphson error sweep.
extern "C" int rmsnorm_f32_s3(float* out, const float* x, const float* w, int d,
                              float inv_d, float eps);

// Atomic: two handles on two cores may run norms at once, and the A/B
// setter may run on a third task. Relaxed is enough for counters and a flag.
static std::atomic<int> g_rms_asm{0}, g_rms_scalar{0};
static std::atomic<int> g_rms_force_scalar{0};

// Same shape as esp32_backend_set_force_scalar: one binary, both paths, so hardware
// can A/B them without a rebuild. It exists for a second reason here -- QEMU does not
// implement rsqrt0.s (see app_main) -- but the A/B is the reason to keep it.
extern "C" void rmsnorm_set_force_scalar(int on) {
  g_rms_force_scalar.store(on, std::memory_order_relaxed);
}

// The fallback, and also the reference the assembly is judged against. Deliberately
// NOT the same summation order: this is sequential, the kernel uses four accumulators,
// so the two differ in the last ulp or two. test_rmsnorm.py measures that gap at
// ~2e-07 relative rather than assuming it away.
static void rmsnorm_scalar(float* out, const float* x, const float* w, int d,
                           float inv_d, float eps) {
  float sum = 0.0f;
  for (int i = 0; i < d; i++) sum += x[i] * x[i];
  const float s = 1.0f / sqrtf(sum * inv_d + eps);
  if (w) {
    for (int i = 0; i < d; i++) out[i] = (w[i] * x[i]) * s;
  } else {
    for (int i = 0; i < d; i++) out[i] = x[i] * s;
  }
}

// One row: the assembly kernel when allowed, else -- and for any row the
// kernel refuses because mean(x^2) + eps is not a positive normal float, where
// its Newton refinement turns into NaN -- the scalar reference.
ESP32_HOT_CODE
static void rmsnorm_row(bool use_asm, float* out, const float* x,
                        const float* w, int d, float inv_d, float eps) {
  if (use_asm && rmsnorm_f32_s3(out, x, w, d, inv_d, eps)) {
    g_rms_asm.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  rmsnorm_scalar(out, x, w, d, inv_d, eps);
  g_rms_scalar.fetch_add(1, std::memory_order_relaxed);
}

ESP32_HOT_CODE
static void run_rmsnorm(const Tensor& tx, const float* w, float eps, Tensor& tout) {
  // The AOT pass only fuses last-axis fp32 normalization. Derive width from x
  // so the same executor handles scaled and Glimmer's weightless form.
  const int d = (int)tx.size(tx.dim() - 1);
  const int rows = (d > 0) ? (int)(tx.numel() / d) : 0;

  const float* x = tx.const_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();
  const float inv_d = 1.0f / (float)d;

  // Unlike dot_i8_s3, this kernel uses scalar LSI/SSI rather than EE.LDF.128.IP, so
  // 4-byte alignment is the entire contract -- no 16-byte requirement, no d%4 split.
  const uintptr_t ptrs = (uintptr_t)out | (uintptr_t)x | (uintptr_t)w;
  const bool aligned = ((ptrs & 3u) == 0);
  const bool use_asm = aligned && !g_rms_force_scalar.load(std::memory_order_relaxed);

  for (int r = 0; r < rows; r++) {
    rmsnorm_row(use_asm, out + (size_t)r * d, x + (size_t)r * d, w, d, inv_d,
                eps);
  }
}

// espressif::rmsnorm.out(Tensor x, Tensor weight, float eps, *, Tensor(a!) out)
ESP32_HOT_CODE
static void shim_rmsnorm(KernelRuntimeContext& ctx, Span<EValue*> st) {
  (void)ctx;
  const Tensor& tx = st[0]->toTensor();
  const Tensor& tw = st[1]->toTensor();
  const float eps = (float)st[2]->toDouble();
  Tensor& tout = st[3]->toTensor();
  run_rmsnorm(tx, tw.const_data_ptr<float>(), eps, tout);
}

// espressif::rmsnorm_noscale.out(Tensor x, float eps, *, Tensor(a!) out)
ESP32_HOT_CODE
static void shim_rmsnorm_noscale(KernelRuntimeContext& ctx, Span<EValue*> st) {
  (void)ctx;
  const Tensor& tx = st[0]->toTensor();
  const float eps = (float)st[1]->toDouble();
  Tensor& tout = st[2]->toTensor();
  run_rmsnorm(tx, nullptr, eps, tout);
}

// espressif::rmsnorm_add.out(Tensor x, Tensor weight, float eps, Tensor residual,
//                            *, Tensor(a!) out)
// residual + rmsnorm(x, weight): a post-norm sublayer's residual add in one
// call (fuse_rmsnorm_add_pass.py). The norm is written exactly as rmsnorm.out
// writes it, then the residual is added: the same two float roundings as the
// rmsnorm.out + aten::add.out pair it replaces, so the result is bit-identical.
ESP32_HOT_CODE
static void shim_rmsnorm_add(KernelRuntimeContext& ctx, Span<EValue*> st) {
  const Tensor& tx = st[0]->toTensor();
  const Tensor& tw = st[1]->toTensor();
  const float eps = (float)st[2]->toDouble();
  const Tensor& tr = st[3]->toTensor();
  Tensor& tout = st[4]->toTensor();
  const auto f32 = executorch::aten::ScalarType::Float;
  if (tx.scalar_type() != f32 || tw.scalar_type() != f32 ||
      tr.scalar_type() != f32 || tout.scalar_type() != f32 ||
      tx.numel() != tout.numel() || tr.numel() != tout.numel() ||
      !contiguous_tensor(tx) || !contiguous_tensor(tr) ||
      !contiguous_tensor(tout)) {
    printf("  [rmsnorm_add] expected contiguous fp32 x, residual and output of one size\n");
    ctx.fail(Error::InvalidArgument);
    return;
  }
  // The norm is written into out before the residual is read, so a residual
  // sharing out's storage would be overwritten first. The memory planner never
  // aliases them (inputs live through the instruction); refuse a hand-built call.
  if (tensor_buffers_overlap(tr, tout)) {
    printf("  [rmsnorm_add] residual and output share storage\n");
    ctx.fail(Error::InvalidArgument);
    return;
  }
  run_rmsnorm(tx, tw.const_data_ptr<float>(), eps, tout);
  float* out = tout.mutable_data_ptr<float>();
  const float* r = tr.const_data_ptr<float>();
  for (ssize_t i = 0; i < tout.numel(); ++i) {
    out[i] = r[i] + out[i];
  }
}

static int g_rope_rows = 0;
static uint64_t g_cached_attention_calls = 0;
static uint64_t g_cached_attention_positions = 0;
static uint64_t g_ple_prepare_calls = 0;
static uint64_t g_ple_prepare_values = 0;

// espressif::ple_prepare.out(
//     Tensor projected, Tensor table, Tensor weight, float input_scale,
//     float table_scale, float output_scale, float eps, *, Tensor(a!) out)
//
// projected/table/out are [...,P], and weight is [P]. This is the literal PLE
// expression streamed once: scaled projection -> per-row RMSNorm -> scaled
// table add -> output scale. The unfused portable graph makes four full passes
// over L*P: 48.7 ms/token for only 9,216 scalar multiplies.
static void shim_ple_prepare(KernelRuntimeContext& ctx, Span<EValue*> st) {
  // The unboxed stack includes a trailing return EValue after the explicit
  // out tensor, just like cached_attention.out below. The kernel writes st[7];
  // ExecuTorch propagates that value through st[8].
  if (st.size() != 9) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& tprojected = st[0]->toTensor();
  const Tensor& ttable = st[1]->toTensor();
  const Tensor& tweight = st[2]->toTensor();
  const float input_scale = (float)st[3]->toDouble();
  const float table_scale = (float)st[4]->toDouble();
  const float output_scale = (float)st[5]->toDouble();
  const float eps = (float)st[6]->toDouble();
  Tensor& tout = st[7]->toTensor();

  const bool floats =
      tprojected.scalar_type() == executorch::aten::ScalarType::Float &&
      ttable.scalar_type() == executorch::aten::ScalarType::Float &&
      tweight.scalar_type() == executorch::aten::ScalarType::Float &&
      tout.scalar_type() == executorch::aten::ScalarType::Float;
  const bool shapes = tprojected.dim() >= 1 && tweight.dim() == 1 &&
      same_shape(tprojected, ttable) && same_shape(tprojected, tout) &&
      tprojected.size(tprojected.dim() - 1) == tweight.size(0);
  const bool contiguous = contiguous_tensor(tprojected) &&
      contiguous_tensor(ttable) && contiguous_tensor(tweight) &&
      contiguous_tensor(tout);
  if (!floats || !shapes || !contiguous || !std::isfinite(input_scale) ||
      !std::isfinite(table_scale) || !std::isfinite(output_scale) ||
      !(eps >= 0.0f)) {
    ctx.fail(Error::InvalidArgument);
    return;
  }

  const int width = (int)tweight.size(0);
  const int rows = width > 0 ? (int)(tprojected.numel() / width) : 0;
  if (width <= 0 || rows <= 0) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const float* projected = tprojected.const_data_ptr<float>();
  const float* table = ttable.const_data_ptr<float>();
  const float* weight = tweight.const_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();
  const float inv_width = 1.0f / (float)width;

  for (int row = 0; row < rows; ++row) {
    const size_t base = (size_t)row * width;
    float sum = 0.0f;
    for (int lane = 0; lane < width; ++lane) {
      const float scaled = projected[base + lane] * input_scale;
      sum += scaled * scaled;
    }
    const float inverse = 1.0f / sqrtf(sum * inv_width + eps);
    for (int lane = 0; lane < width; ++lane) {
      const size_t index = base + lane;
      const float scaled = projected[index] * input_scale;
      const float normalized = (weight[lane] * scaled) * inverse;
      out[index] =
          (normalized + table[index] * table_scale) * output_scale;
    }
  }
  ++g_ple_prepare_calls;
  g_ple_prepare_values += (uint64_t)tprojected.numel();
}

// espressif::cached_attention.out(
//     Tensor q, Tensor k, Tensor v, Tensor(a!) kcache, Tensor(b!) vcache,
//     Tensor pos, float scale_denominator, *, Tensor(c!) out) -> Tensor(c!)
//
// q/k/v and out are [H,Dh]; caches are contiguous [H,S,Dh].  Only one cache
// row is updated.  A single [S] score vector comes from ET's per-instruction
// temp allocator and is reused across heads; the runtime resets that allocator
// after this kernel returns.  Thus scratch is 1 KiB at S=256 and 64 KiB at S=16K,
// independent of layer count.
static void shim_cached_attention(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 9) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& tq = st[0]->toTensor();
  const Tensor& tk = st[1]->toTensor();
  const Tensor& tv = st[2]->toTensor();
  Tensor& tkh = st[3]->toTensor();
  Tensor& tvh = st[4]->toTensor();
  const Tensor& tpos = st[5]->toTensor();
  const float scale_denominator = (float)st[6]->toDouble();
  Tensor& tout = st[7]->toTensor();

  const bool floats =
      tq.scalar_type() == executorch::aten::ScalarType::Float &&
      tk.scalar_type() == executorch::aten::ScalarType::Float &&
      tv.scalar_type() == executorch::aten::ScalarType::Float &&
      tkh.scalar_type() == executorch::aten::ScalarType::Float &&
      tvh.scalar_type() == executorch::aten::ScalarType::Float &&
      tout.scalar_type() == executorch::aten::ScalarType::Float;
  const bool shapes = tq.dim() == 2 && tk.dim() == 2 && tv.dim() == 2 &&
      tkh.dim() == 3 && tvh.dim() == 3 && tout.dim() == 2 &&
      same_shape(tq, tk) && same_shape(tq, tv) && same_shape(tq, tout) &&
      same_shape(tkh, tvh) && tkh.size(0) == tq.size(0) &&
      tkh.size(2) == tq.size(1);
  const bool contiguous = contiguous_tensor(tq) && contiguous_tensor(tk) &&
      contiguous_tensor(tv) && contiguous_tensor(tkh) &&
      contiguous_tensor(tvh) && contiguous_tensor(tout);
  if (!floats || !shapes || !contiguous || tpos.numel() != 1 ||
      tpos.scalar_type() != executorch::aten::ScalarType::Long ||
      !(scale_denominator > 0.0f)) {
    ctx.fail(Error::InvalidArgument);
    return;
  }

  const int heads = (int)tq.size(0);
  const int width = (int)tq.size(1);
  const int context = (int)tkh.size(1);
  const int64_t position64 = tpos.const_data_ptr<int64_t>()[0];
  if (heads <= 0 || width <= 0 || context <= 0 || position64 < 0 ||
      position64 >= context) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int position = (int)position64;

  auto scratch_result = ctx.allocate_temp(
      (size_t)context * sizeof(float), alignof(float));
  if (!scratch_result.ok()) {
    ctx.fail(scratch_result.error());
    return;
  }
  float* scores = static_cast<float*>(scratch_result.get());
  const float* q = tq.const_data_ptr<float>();
  const float* k = tk.const_data_ptr<float>();
  const float* v = tv.const_data_ptr<float>();
  float* kh = tkh.mutable_data_ptr<float>();
  float* vh = tvh.mutable_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();

  // Cache row update: 2*H*Dh floats, rather than two H*S*Dh copies.
  for (int head = 0; head < heads; ++head) {
    const size_t source = (size_t)head * width;
    const size_t cache = ((size_t)head * context + position) * width;
    memcpy(kh + cache, k + source, (size_t)width * sizeof(float));
    memcpy(vh + cache, v + source, (size_t)width * sizeof(float));
  }

  // The probabilities are normalised BEFORE P@V, as softmax then matmul (the
  // Composite) and shim_attn_tail do: one reciprocal of the denominator, a
  // multiply per weight, then P@V. Dividing the finished sum instead lets the
  // unnormalised sum overflow where the attention output is finite (three V
  // rows of FLT_MAX/2 with equal scores: a sum of 1.5 FLT_MAX). Both dots are
  // std::fma, the madd.s GCC's default contraction already made of them on the
  // chip, spelled out so the host runs the chip's arithmetic whatever its
  // -ffp-contract.
  for (int head = 0; head < heads; ++head) {
    const float* qh = q + (size_t)head * width;
    const float* kh_head = kh + (size_t)head * context * width;
    const float* vh_head = vh + (size_t)head * context * width;
    float maximum = -INFINITY;
    for (int token = 0; token <= position; ++token) {
      const float* kt = kh_head + (size_t)token * width;
      float dot = 0.0f;
      for (int lane = 0; lane < width; ++lane) {
        dot = std::fma(qh[lane], kt[lane], dot);
      }
      const float score = dot / scale_denominator;
      scores[token] = score;
      maximum = std::max(maximum, score);
    }
    float denominator = 0.0f;
    for (int token = 0; token <= position; ++token) {
      const float weight = expf(scores[token] - maximum);
      scores[token] = weight;
      denominator += weight;
    }
    const float inverse = 1.0f / denominator;
    for (int token = 0; token <= position; ++token) {
      scores[token] *= inverse;
    }
    float* output = out + (size_t)head * width;
    for (int lane = 0; lane < width; ++lane) {
      float sum = 0.0f;
      for (int token = 0; token <= position; ++token) {
        sum = std::fma(scores[token], vh_head[(size_t)token * width + lane],
                       sum);
      }
      output[lane] = sum;
    }
  }
  ++g_cached_attention_calls;
  g_cached_attention_positions += (uint64_t)position + 1;
}

static uint64_t g_attn_tail_calls = 0;
static uint64_t g_attn_tail_rows = 0;

// espressif::attn_tail.out(Tensor scores, Tensor mask, float scale,
//     Tensor values, *, Tensor(a!) out) -> Tensor(a!)
//
// softmax(where(mask, scores * scale, -inf), -1) @ values: the decode tail after
// the delegated q@k^T, fused by fuse_attn_tail_pass.py. scores [B,M,L] and out
// [B,M,N] are fp32, mask is the L bools of one row shared by every head, and
// values [B,L,N] is the KV head's cache. As four portable calls (mul, where,
// _softmax, bmm) it costs 19.6 ms/token on the story model, mostly dispatch,
// newlib expf and a software divide per probability.
//
// Per score row: one pass scales, masks and takes the max; fast_exp_neg (the
// 64-entry table exp, for s - max <= 0) maps s - max; ONE reciprocal of the sum
// (the row's only division) normalises by multiplication; then P@V runs in
// shim_bmm_out's order. That is not bit-identical to the four ops (newlib expf, a
// divide per lane); models.py's attn_tail_f32_reference transcribes this
// arithmetic for the host goldens. A fully masked row comes out NaN, as from the
// graph it replaces: exp(-inf - -inf) is exp(NaN) there and fast_exp_neg(NaN) here.
ESP32_HOT_CODE
static void shim_attn_tail(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 6) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& tscores = st[0]->toTensor();
  const Tensor& tmask = st[1]->toTensor();
  const float scale = (float)st[2]->toDouble();
  const Tensor& tvalues = st[3]->toTensor();
  Tensor& tout = st[4]->toTensor();

  const bool types =
      tscores.scalar_type() == executorch::aten::ScalarType::Float &&
      tmask.scalar_type() == executorch::aten::ScalarType::Bool &&
      tvalues.scalar_type() == executorch::aten::ScalarType::Float &&
      tout.scalar_type() == executorch::aten::ScalarType::Float;
  const bool ranks = tscores.dim() == 3 && tvalues.dim() == 3 &&
      tout.dim() == 3 && tmask.dim() >= 1;
  if (!types || !ranks) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const ssize_t batches = tscores.size(0);
  const ssize_t rows = tscores.size(1);
  const ssize_t width = tscores.size(2);
  const ssize_t columns = tvalues.size(2);
  const bool shapes = batches > 0 && rows > 0 && width > 0 && columns > 0 &&
      tvalues.size(0) == batches && tvalues.size(1) == width &&
      tout.size(0) == batches && tout.size(1) == rows &&
      tout.size(2) == columns && tmask.size(tmask.dim() - 1) == width &&
      tmask.numel() == width;
  const bool contiguous = contiguous_tensor(tscores) &&
      contiguous_tensor(tmask) && contiguous_tensor(tvalues) &&
      contiguous_tensor(tout);
  const bool disjoint = !tensor_buffers_overlap(tout, tscores) &&
      !tensor_buffers_overlap(tout, tmask) &&
      !tensor_buffers_overlap(tout, tvalues);
  if (!shapes || !contiguous || !disjoint) {
    ctx.fail(Error::InvalidArgument);
    return;
  }

  // One row of weights from the per-instruction temp allocator, reused for
  // every row: 1 KiB at L=256, which the runners' internal-SRAM tier holds.
  auto scratch = ctx.allocate_temp((size_t)width * sizeof(float), alignof(float));
  if (!scratch.ok()) {
    ctx.fail(scratch.error());
    return;
  }
  float* weights = static_cast<float*>(scratch.get());
  const float* scores = tscores.const_data_ptr<float>();
  const bool* mask = tmask.const_data_ptr<bool>();
  const float* values = tvalues.const_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();

  for (ssize_t batch = 0; batch < batches; ++batch) {
    const float* batch_values = values + (size_t)batch * width * columns;
    for (ssize_t row = 0; row < rows; ++row) {
      const size_t index = (size_t)batch * rows + row;
      const float* row_scores = scores + index * width;
      float* row_out = out + index * columns;

      // std::max(value, maximum) is the softmax shim's comparison, so even a
      // NaN score propagates as it does there.
      float maximum = -INFINITY;
      for (ssize_t k = 0; k < width; ++k) {
        const float value = mask[k] ? row_scores[k] * scale : -INFINITY;
        weights[k] = value;
        maximum = std::max(value, maximum);
      }
      // Two lanes per step, interleaved (fast_exp_neg_lanes), with the
      // constants held in registers; the sum still adds in index order. A pair
      // that both flush (masked lanes, -inf) skips the arithmetic: its +0s are
      // what fast_exp_neg returns for them.
      const FastExpNeg exp_constants = fast_exp_neg_hoisted();
      float sum = 0.0f;
      ssize_t k = 0;
      for (; k + 2 <= width; k += 2) {
        const float shifted[2] = {weights[k] - maximum, weights[k + 1] - maximum};
        float mapped[2] = {0.0f, 0.0f};
        if (!(shifted[0] < exp_constants.flush &&
              shifted[1] < exp_constants.flush)) {
          fast_exp_neg_lanes<2>(exp_constants, shifted, mapped);
        }
        weights[k] = mapped[0];
        weights[k + 1] = mapped[1];
        sum += mapped[0];
        sum += mapped[1];
      }
      if (k < width) {
        const float shifted[1] = {weights[k] - maximum};
        float mapped[1];
        fast_exp_neg_lanes<1>(exp_constants, shifted, mapped);
        weights[k] = mapped[0];
        sum += mapped[0];
      }
      const float inverse = 1.0f / sum;
      for (ssize_t k = 0; k < width; ++k) {
        weights[k] *= inverse;
      }

      // shim_bmm_out's loop: four columns live, k ascending from zero. fma is
      // what GCC's default contraction makes of that shim's `sum += a * b` on
      // this target (madd.s); spelled out so this arithmetic, and the host
      // mirror of it, cannot drift with -ffp-contract.
      ssize_t column = 0;
      for (; column + 4 <= columns; column += 4) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (ssize_t k = 0; k < width; ++k) {
          const float p = weights[k];
          const float* v = batch_values + (size_t)k * columns + column;
          sum0 = std::fma(p, v[0], sum0);
          sum1 = std::fma(p, v[1], sum1);
          sum2 = std::fma(p, v[2], sum2);
          sum3 = std::fma(p, v[3], sum3);
        }
        row_out[column] = sum0;
        row_out[column + 1] = sum1;
        row_out[column + 2] = sum2;
        row_out[column + 3] = sum3;
      }
      for (; column < columns; ++column) {
        float acc = 0.0f;
        for (ssize_t k = 0; k < width; ++k) {
          acc = std::fma(weights[k],
                         batch_values[(size_t)k * columns + column], acc);
        }
        row_out[column] = acc;
      }
    }
  }
  ++g_attn_tail_calls;
  g_attn_tail_rows += (uint64_t)(batches * rows);
}

// espressif::conv_step.out(Tensor(a!) window, Tensor x, Tensor weight,
//     Tensor bias, *, Tensor(b!) out) -> Tensor(b!)
//
// One token through a scan layer's causal depthwise conv: window is the last
// three inputs [d, 3], x the current one [d], weight [d, 4]. Writes the conv
// output and shifts the window left by one, in place. The tap sum runs oldest
// to newest, then adds the bias, matching the eager oracle's order.
static void shim_conv_step(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 6) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  Tensor& twin = st[0]->toTensor();
  const Tensor& tx = st[1]->toTensor();
  const Tensor& tw = st[2]->toTensor();
  const Tensor& tb = st[3]->toTensor();
  Tensor& tout = st[4]->toTensor();
  const bool floats =
      twin.scalar_type() == executorch::aten::ScalarType::Float &&
      tx.scalar_type() == executorch::aten::ScalarType::Float &&
      tw.scalar_type() == executorch::aten::ScalarType::Float &&
      tb.scalar_type() == executorch::aten::ScalarType::Float &&
      tout.scalar_type() == executorch::aten::ScalarType::Float;
  const bool shapes = twin.dim() == 2 && twin.size(1) == 3 && tw.dim() == 2 &&
      tw.size(0) == twin.size(0) && tw.size(1) == 4 &&
      tx.numel() == twin.size(0) && tb.numel() == twin.size(0) &&
      tout.numel() == twin.size(0);
  const bool contiguous = contiguous_tensor(twin) && contiguous_tensor(tx) &&
      contiguous_tensor(tw) && contiguous_tensor(tb) && contiguous_tensor(tout);
  if (!floats || !shapes || !contiguous) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int channels = (int)twin.size(0);
  float* window = twin.mutable_data_ptr<float>();
  const float* x = tx.const_data_ptr<float>();
  const float* weight = tw.const_data_ptr<float>();
  const float* bias = tb.const_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();
  for (int c = 0; c < channels; ++c) {
    float* taps = window + (size_t)c * 3;
    const float* w = weight + (size_t)c * 4;
    float acc = taps[0] * w[0];
    acc += taps[1] * w[1];
    acc += taps[2] * w[2];
    acc += x[c] * w[3];
    out[c] = acc + bias[c];
    taps[0] = taps[1];
    taps[1] = taps[2];
    taps[2] = x[c];
  }
}

// espressif::ssm_step.out(Tensor dt, Tensor x, Tensor A, Tensor B, Tensor C,
//     Tensor D, Tensor(a!) state, *, Tensor(b!) out) -> Tensor(b!)
//
// One token through a scan layer's selective recurrence, state [d, s] updated
// in place:  dA = exp(max(dt*A, -30)),  h = state*dA + (dt*x)*B,  state <- h,
// y = sum_s(h*C) + D*x.  Nothing of size [d, s] is materialised: with the eager
// spelling's whole [d_inner, d_state] working set, a hybrid's decode plan does
// not fit QEMU's internal RAM. Scalar fp32: PIE has no float vector unit, and
// the exp is the cost that matters.
static void shim_ssm_step(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 9) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& tdt = st[0]->toTensor();
  const Tensor& tx = st[1]->toTensor();
  const Tensor& ta = st[2]->toTensor();
  const Tensor& tb = st[3]->toTensor();
  const Tensor& tc = st[4]->toTensor();
  const Tensor& td = st[5]->toTensor();
  Tensor& tstate = st[6]->toTensor();
  Tensor& tout = st[7]->toTensor();
  const bool floats =
      tdt.scalar_type() == executorch::aten::ScalarType::Float &&
      tx.scalar_type() == executorch::aten::ScalarType::Float &&
      ta.scalar_type() == executorch::aten::ScalarType::Float &&
      tb.scalar_type() == executorch::aten::ScalarType::Float &&
      tc.scalar_type() == executorch::aten::ScalarType::Float &&
      td.scalar_type() == executorch::aten::ScalarType::Float &&
      tstate.scalar_type() == executorch::aten::ScalarType::Float &&
      tout.scalar_type() == executorch::aten::ScalarType::Float;
  const bool shapes = tstate.dim() == 2 && ta.dim() == 2 &&
      same_shape(ta, tstate) && tdt.numel() == tstate.size(0) &&
      tx.numel() == tstate.size(0) && td.numel() == tstate.size(0) &&
      tout.numel() == tstate.size(0) && tb.numel() == tstate.size(1) &&
      tc.numel() == tstate.size(1);
  const bool contiguous = contiguous_tensor(tdt) && contiguous_tensor(tx) &&
      contiguous_tensor(ta) && contiguous_tensor(tb) && contiguous_tensor(tc) &&
      contiguous_tensor(td) && contiguous_tensor(tstate) &&
      contiguous_tensor(tout);
  if (!floats || !shapes || !contiguous) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const int channels = (int)tstate.size(0);
  const int states = (int)tstate.size(1);
  const float* dt = tdt.const_data_ptr<float>();
  const float* x = tx.const_data_ptr<float>();
  const float* A = ta.const_data_ptr<float>();
  const float* B = tb.const_data_ptr<float>();
  const float* C = tc.const_data_ptr<float>();
  const float* D = td.const_data_ptr<float>();
  float* state = tstate.mutable_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();
  for (int c = 0; c < channels; ++c) {
    const float decay_rate = dt[c];
    const float input = x[c];
    const float update = decay_rate * input;
    float* h = state + (size_t)c * states;
    const float* a = A + (size_t)c * states;
    float readout = 0.0f;
    for (int s = 0; s < states; ++s) {
      float exponent = decay_rate * a[s];
      if (exponent < -30.0f) {
        exponent = -30.0f;
      }
      const float next = h[s] * expf(exponent) + update * B[s];
      h[s] = next;
      readout += next * C[s];
    }
    out[c] = readout + D[c] * input;
  }
}

// espressif::rope.out(Tensor x, Tensor cos, Tensor sin, int rot_dim, *,
//                     Tensor(a!) out)
//
// Rotary position embedding, the eight-op chain fused by fuse_rope_pass.py: one
// read and one write instead of eight of each. Unfused, that chain is 34% of
// a speech encoder's portable output traffic.
//
// Operates on the last dimension. `rot_dim` lanes rotate in half-pairs and the
// remainder passes through (partial rotary); rot == width is the same loop with
// an empty tail.
//
// cos/sin are indexed by POSITION -- the second-to-last axis of x -- and
// broadcast over every leading axis (batch, heads). That is the layout the AOT
// reference produces, [T, rot] against [B, H, T, rot], and the pass checks it
// rather than this kernel assuming it.
ESP32_HOT_CODE
static void shim_rope(KernelRuntimeContext& ctx, Span<EValue*> st) {
  (void)ctx;
  const Tensor& tx = st[0]->toTensor();
  const Tensor& tcos = st[1]->toTensor();
  const Tensor& tsin = st[2]->toTensor();
  const int rot = (int)st[3]->toInt();
  Tensor& tout = st[4]->toTensor();

  const int width = (int)tx.size(tx.dim() - 1);
  const int rows = (width > 0) ? (int)(tx.numel() / width) : 0;
  const int half = rot / 2;
  const int t_len = (tcos.dim() >= 2) ? (int)tcos.size(tcos.dim() - 2) : 1;
  const int cs_w = (int)tcos.size(tcos.dim() - 1);

  const float* x = tx.const_data_ptr<float>();
  const float* cs = tcos.const_data_ptr<float>();
  const float* sn = tsin.const_data_ptr<float>();
  float* out = tout.mutable_data_ptr<float>();

  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (size_t)r * width;
    float* orow = out + (size_t)r * width;
    // Rows walk the leading axes with the position axis innermost, so the
    // position cycles with period t_len however many heads or batches sit above.
    const int pos = (t_len > 0) ? (r % t_len) : 0;
    const float* c = cs + (size_t)pos * cs_w;
    const float* s = sn + (size_t)pos * cs_w;
    for (int i = 0; i < half; ++i) {
      const float a = xr[i];
      const float b = xr[i + half];
      // Full-width HF tables supply one value per output lane. Half-width
      // pairwise tables deliberately share one value across the two halves.
      const int j = (cs_w == rot) ? i + half : i;
      // One rounding per output, in the order GCC's default contraction
      // chose on the chip: mul.s a*c[i], then msub.s -b*s[i]; mul.s a*s[j],
      // then madd.s +b*c[j] (it fuses the source's FIRST product of the
      // second line). Spelled out as std::fma so the host runner, built with
      // -ffp-contract=off, runs the same arithmetic, and test/reference.py's
      // rope_f32_device mirrors it; the torch spelling the fusion pass
      // matches (and the op's Composite) rounds each product, which is 1 ulp
      // off in about a quarter of outputs.
      orow[i] = std::fma(-b, s[i], a * c[i]);
      orow[i + half] = std::fma(b, c[j], a * s[j]);
    }
    for (int i = rot; i < width; ++i) {
      orow[i] = xr[i];
    }
    g_rope_rows++;
  }
}

static uint64_t g_quantize_rows_calls = 0;
static uint64_t g_quantize_rows_rows = 0;

// espressif::quantize_rows_i8.out(Tensor x, *, Tensor(a!) codes,
//                                 Tensor(b!) scales) -> (Tensor(a!), Tensor(b!))
//
// The write side of the persistent int8 K cache (kcache_i8.py): every row of x
// [..., K] becomes int8 codes [..., K] at its own absmax/127 scale [...]. The
// arithmetic is the delegate's quantize_row_i8, operation for operation --
// amax / 127.0f, a0_inverse_scale (1.0f / scale, or 0 for a zero or
// subnormal scale: that row's codes are all 0), x * inv, round half away
// from zero on a float +0.5f, clamp +-127 -- because the q@kT op that later
// reads these codes must see exactly the codes it would make itself from the
// fp32 row on every step. floorf rather than the delegate's FLOOR.S: both are
// exact floors of a finite float, which |x * inv| <= 127.5 is for every
// finite row, and libm keeps this file portable.
ESP32_HOT_CODE
static void shim_quantize_rows_i8(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (st.size() != 4) {
    ctx.fail(Error::InvalidProgram);
    return;
  }
  const Tensor& tx = st[0]->toTensor();
  Tensor& tcodes = st[1]->toTensor();
  Tensor& tscales = st[2]->toTensor();
  const bool types =
      tx.scalar_type() == executorch::aten::ScalarType::Float &&
      tcodes.scalar_type() == executorch::aten::ScalarType::Char &&
      tscales.scalar_type() == executorch::aten::ScalarType::Float;
  const ssize_t width = tx.dim() > 0 ? tx.size(tx.dim() - 1) : 0;
  bool shapes = width > 0 && same_shape(tx, tcodes) &&
      tscales.dim() == tx.dim() - 1;
  for (ssize_t dim = 0; shapes && dim < tscales.dim(); ++dim) {
    shapes = tscales.size(dim) == tx.size(dim);
  }
  if (!types || !shapes || !contiguous_tensor(tx) ||
      !contiguous_tensor(tcodes) || !contiguous_tensor(tscales)) {
    ctx.fail(Error::InvalidArgument);
    return;
  }
  const ssize_t rows = tx.numel() / width;
  const float* x = tx.const_data_ptr<float>();
  int8_t* codes = tcodes.mutable_data_ptr<int8_t>();
  float* scales = tscales.mutable_data_ptr<float>();
  for (ssize_t r = 0; r < rows; ++r) {
    const float* xr = x + (size_t)r * width;
    int8_t* cr = codes + (size_t)r * width;
    float amax = 0.0f;
    for (ssize_t t = 0; t < width; ++t) {
      const float a = std::fabs(xr[t]);
      if (a > amax) {
        amax = a;
      }
    }
    const float scale = amax / 127.0f;
    const float inv = a0_inverse_scale(scale);
    for (ssize_t t = 0; t < width; ++t) {
      const float v = xr[t] * inv;
      int32_t q = v < 0.0f ? -(int32_t)floorf(-v + 0.5f)
                           : (int32_t)floorf(v + 0.5f);
      q = q > 127 ? 127 : (q < -127 ? -127 : q);
      cr[t] = (int8_t)q;
    }
    scales[r] = scale;
  }
  ++g_quantize_rows_calls;
  g_quantize_rows_rows += (uint64_t)rows;
}

// The squeeze-excite scale's loop (elementwise.h), in firmware built with it
// (gen_kernels.py --elementwise-fast): it is tried first and every other call
// goes to shim_mul_out, unchanged. Firmware built without it -- the story's --
// registers shim_mul_out itself.
#if !defined(ESP32_GENERATED_EW_MUL)
#define ESP32_GENERATED_EW_MUL 0
#endif
#if !defined(ESP32_GENERATED_EW_CONSTANT_PAD)
#define ESP32_GENERATED_EW_CONSTANT_PAD 0
#endif
#if ESP32_GENERATED_EW_MUL
static void shim_mul_out_ew(KernelRuntimeContext& ctx, Span<EValue*> st) {
  if (!espressif::elementwise::try_mul_out(ctx, st)) {
    shim_mul_out(ctx, st);
  }
}
#define ESP32_MUL_SHIM shim_mul_out_ew
#else
#define ESP32_MUL_SHIM shim_mul_out
#endif

static constexpr Kernel kCustomKernels[] = {
#ifdef ESP32_FAST_LAYERNORM
    Kernel("aten::native_layer_norm.out", shim_fast_layer_norm_out),
#endif
#if ESP32_OP(ATEN_SOFTMAX)
    Kernel("aten::_softmax.out", shim_softmax_out),
#endif
#if ESP32_OP(ATEN_ADD)
    Kernel("aten::add.out", shim_add_out),
#endif
#if ESP32_OP(ATEN_BMM)
    Kernel("aten::bmm.out", shim_bmm_out),
#endif
#if ESP32_OP(ATEN_DIV)
    Kernel("aten::div.out", shim_div_out),
#endif
#if ESP32_OP(ATEN_INDEX_PUT)
    Kernel("aten::index_put_", shim_index_put_),
#endif
#if ESP32_OP(ATEN_LE)
    Kernel("aten::le.Tensor_out", shim_le_tensor_out),
#endif
#if ESP32_OP(ATEN_MUL)
    Kernel("aten::mul.out", ESP32_MUL_SHIM),
#endif
#if ESP32_OP(ATEN_PERMUTE_COPY)
    Kernel("aten::permute_copy.out", shim_permute_copy_out),
#endif
#if ESP32_OP(ATEN_REMAINDER)
    Kernel("aten::remainder.Scalar_out", shim_remainder_scalar_out),
#endif
#if ESP32_OP(ATEN_SPLIT_WITH_SIZES_COPY)
    Kernel("aten::split_with_sizes_copy.out", shim_split_with_sizes_copy_out),
#endif
#if ESP32_OP(ATEN_WHERE)
    Kernel("aten::where.self_out", shim_where_self_out),
#endif
    // ExecuTorch's per-tensor quantize and dequantize and the average pool,
    // each only where the program names it (qdq_pool.h): its portable kernel
    // is then linked as the fallback.
#if ESP32_GENERATED_FAST_QUANTIZE
    Kernel("quantized_decomposed::quantize_per_tensor.out",
           espressif::qdq_pool::shim_quantize_per_tensor_out),
#endif
#if ESP32_GENERATED_FAST_DEQUANTIZE
    Kernel("quantized_decomposed::dequantize_per_tensor.out",
           espressif::qdq_pool::shim_dequantize_per_tensor_out),
#endif
#if ESP32_GENERATED_FAST_AVG_POOL2D
    Kernel("aten::avg_pool2d.out", espressif::qdq_pool::shim_avg_pool2d_out),
#endif
#if ESP32_GENERATED_EW_CONSTANT_PAD
    // The row copy (elementwise.h), where the program names the operator in
    // a firmware built --elementwise-fast: its portable kernel is then linked
    // as the fallback.
    Kernel("aten::constant_pad_nd.out",
           espressif::elementwise::shim_constant_pad_nd_out),
#endif
#if ESP32_OP(ESPRESSIF_RMSNORM)
    Kernel("espressif::rmsnorm.out", shim_rmsnorm),
#endif
#if ESP32_OP(ESPRESSIF_RMSNORM_NOSCALE)
    Kernel("espressif::rmsnorm_noscale.out", shim_rmsnorm_noscale),
#endif
#if ESP32_OP(ESPRESSIF_RMSNORM_ADD)
    Kernel("espressif::rmsnorm_add.out", shim_rmsnorm_add),
#endif
#if ESP32_OP(ESPRESSIF_PLE_PREPARE)
    Kernel("espressif::ple_prepare.out", shim_ple_prepare),
#endif
#if ESP32_OP(ESPRESSIF_CACHED_ATTENTION)
    Kernel("espressif::cached_attention.out", shim_cached_attention),
#endif
#if ESP32_OP(ESPRESSIF_ATTN_TAIL)
    Kernel("espressif::attn_tail.out", shim_attn_tail),
#endif
#if ESP32_OP(ESPRESSIF_CONV_STEP)
    Kernel("espressif::conv_step.out", shim_conv_step),
#endif
#if ESP32_OP(ESPRESSIF_SSM_STEP)
    Kernel("espressif::ssm_step.out", shim_ssm_step),
#endif
#if ESP32_OP(ESPRESSIF_ROPE)
    Kernel("espressif::rope.out", shim_rope),
#endif
#if ESP32_OP(ESPRESSIF_LAYER_NORM2D)
    Kernel("espressif::layer_norm2d.out", espressif::layer_norm2d::shim),
#endif
#if ESP32_OP(ESPRESSIF_QUANTIZE_ROWS_I8)
    Kernel("espressif::quantize_rows_i8.out", shim_quantize_rows_i8),
#endif
#if ESP32_OP(ESPRESSIF_QUANT_LUT)
    Kernel("espressif::quant_lut.out", espressif::quant_lut::shim),
#endif
#if ESP32_OP(ESPRESSIF_QUANT_LUT_CODES)
    // Its code-to-code variant, Stage B's.
    Kernel("espressif::quant_lut_codes.out", espressif::quant_lut::codes_shim),
#endif
#if !defined(ET_PLE_ONLY) && ESP32_OP(ESPRESSIF_LUT16)
    // The int16 table op: a fitted table per a16w8 site.
    // The story firmware (ET_PLE_ONLY) runs no int16 table, so it links none
    // of quant_lut16.cpp, and its code and timing do not depend on it.
    Kernel("espressif::lut16.out", espressif::lut16::shim),
#endif
#if ESP32_OP(ESPRESSIF_QNORM)
    // A decomposed quantised norm as one op (qnorm.h; opt-in).
    Kernel("espressif::qnorm.out", espressif::qnorm::shim),
#endif
    // ExecuTorch's stock quantised K/V cache (kv_affine.h).
#if ESP32_OP(ESPRESSIF_QUANTIZE_PER_TOKEN_XNN)
    Kernel("espressif::quantize_per_token_xnn.out",
           espressif::kv_affine::shim_quantize),
#endif
#if ESP32_OP(ESPRESSIF_DEQUANTIZE_PER_TOKEN_I8)
    Kernel("espressif::dequantize_per_token_i8.out",
           espressif::kv_affine::shim_dequantize),
#endif
#if ESP32_OP(LLAMA_UPDATE_CACHE)
    Kernel("llama::update_cache.out", espressif::kv_affine::shim_update_cache),
#endif
#if !defined(ESP32_HOST_BUILD) && ESP32_OP(ATEN_LOCAL_SCALAR_DENSE)
    // The host links ExecuTorch's own prim op of this name.
    Kernel("aten::_local_scalar_dense",
           espressif::kv_affine::shim_local_scalar_dense),
#endif
    // Never registered: the table ends here, so a firmware whose programs
    // name none of the library's operators still declares an array.
    Kernel("", nullptr),
};
// The registered entries: every one but the terminator.
constexpr size_t kCustomKernelCount =
    sizeof(kCustomKernels) / sizeof(kCustomKernels[0]) - 1;

namespace espressif {

Error register_ops() {
  if (kCustomKernelCount == 0) {
    return Error::Ok;
  }
  return executorch::runtime::register_kernels(
      {kCustomKernels, kCustomKernelCount});
}

size_t kernel_count() {
  return kCustomKernelCount;
}

DispatchStats dispatch_stats() {
  return {
      g_add_same,
      g_add_leading_broadcast,
      g_add_fallback,
      g_permute_adjacent_block,
      g_permute_adjacent_element,
      g_permute_fallback,
      g_bmm_p_le_n,
      g_bmm_p_gt_n,
      g_bmm_fallback,
      g_mul_same,
      g_mul_broadcast,
      g_mul_fallback,
      g_mul_scalar,
      g_where_mask_scalar,
      g_where_fallback,
      g_div_scalar_denominator,
      g_div_fallback,
      g_le_scalar,
      g_le_fallback,
      g_index_put_rows,
      g_index_put_fallback,
      g_split_chunks,
      g_split_fallback,
      g_remainder_non_negative,
      g_remainder_fallback,
      g_softmax_last_dim,
      g_softmax_fallback,
      g_rms_asm.load(std::memory_order_relaxed),
      g_rms_scalar.load(std::memory_order_relaxed),
      g_rope_rows,
      layer_norm2d::columns(),
      g_cached_attention_calls,
      g_cached_attention_positions,
      g_ple_prepare_calls,
      g_ple_prepare_values,
      g_attn_tail_calls,
      g_attn_tail_rows,
      g_quantize_rows_calls,
      g_quantize_rows_rows,
#ifdef ESP32_FAST_LAYERNORM
      g_fast_layer_norm_calls,
      g_fast_layer_norm_rows,
      g_fast_layer_norm_fallback,
#else
      0, 0, 0,
#endif
  };
}

void report_override_dispatch() {
  const DispatchStats stats = dispatch_stats();
  printf("add.out dispatch: same %llu, leading-broadcast %llu, fallback %llu\n",
         (unsigned long long)stats.add_same,
         (unsigned long long)stats.add_leading_broadcast,
         (unsigned long long)stats.add_fallback);
  printf("permute_copy dispatch: adjacent-block %llu, adjacent-element %llu, "
         "fallback %llu\n",
         (unsigned long long)stats.permute_adjacent_block,
         (unsigned long long)stats.permute_adjacent_element,
         (unsigned long long)stats.permute_fallback);
  printf("bmm dispatch: p<=n %llu, p>n %llu, fallback %llu\n",
         (unsigned long long)stats.bmm_p_le_n,
         (unsigned long long)stats.bmm_p_gt_n,
         (unsigned long long)stats.bmm_fallback);
  printf("mul.out dispatch: same %llu, fallback %llu, broadcast %llu, "
         "scalar %llu\n",
         (unsigned long long)stats.mul_same,
         (unsigned long long)stats.mul_fallback,
         (unsigned long long)stats.mul_broadcast,
         (unsigned long long)stats.mul_scalar);
  printf("where.self_out dispatch: mask-scalar %llu, fallback %llu\n",
         (unsigned long long)stats.where_mask_scalar,
         (unsigned long long)stats.where_fallback);
  printf("div.out dispatch: scalar-denominator %llu, fallback %llu\n",
         (unsigned long long)stats.div_scalar_denominator,
         (unsigned long long)stats.div_fallback);
  printf("le.Tensor_out dispatch: scalar %llu, fallback %llu\n",
         (unsigned long long)stats.le_scalar,
         (unsigned long long)stats.le_fallback);
  printf("index_put_ dispatch: row-copy %llu, fallback %llu\n",
         (unsigned long long)stats.index_put_rows,
         (unsigned long long)stats.index_put_fallback);
  printf("split_with_sizes_copy.out dispatch: chunk-copy %llu, fallback %llu\n",
         (unsigned long long)stats.split_chunks,
         (unsigned long long)stats.split_fallback);
  printf("remainder.Scalar_out dispatch: non-negative %llu, fallback %llu\n",
         (unsigned long long)stats.remainder_non_negative,
         (unsigned long long)stats.remainder_fallback);
  printf("_softmax.out dispatch: last-axis-fp32 %llu, fallback %llu\n",
         (unsigned long long)stats.softmax_last_dim,
         (unsigned long long)stats.softmax_fallback);
  quant_lut::report_dispatch();
#if !defined(ET_PLE_ONLY)
  lut16::report_dispatch();
#endif
  kv_affine::report_dispatch();
  qnorm::report_dispatch();
#if ESP32_GENERATED_FAST_QUANTIZE || ESP32_GENERATED_FAST_DEQUANTIZE || \
    ESP32_GENERATED_FAST_AVG_POOL2D
  qdq_pool::report_dispatch();
#endif
#if ESP32_GENERATED_EW_MUL || ESP32_GENERATED_EW_CONSTANT_PAD || \
    ESP32_OP(ESPRESSIF_QUANT_LUT) || ESP32_OP(ESPRESSIF_QUANT_LUT_CODES)
  elementwise::report_dispatch();
#endif
}


// ---- device self-checks, driven by the test runner (main.cpp) ----

// The kernel's domain refusals, driven directly. Each row's mean(x^2) + eps
// leaves the positive normal floats -- finite activations whose squares
// overflow, a NaN, an all-zero row with eps = 0 -- so the kernel must write
// nothing and return 0, and the row path must give the scalar reference's
// bits. The refusal precedes rsqrt0.s, so this runs under QEMU too.
void run_rmsnorm_domain_check() {
  struct Case {
    float x[4];
    float eps;
  };
  const Case cases[] = {
      {{1e19f, 1e19f, 1e19f, 1e19f}, 1e-6f},  // sum of squares -> +inf
      {{NAN, 1.0f, 1.0f, 1.0f}, 1e-6f},
      {{0.0f, 0.0f, 0.0f, 0.0f}, 0.0f},       // t = 0
  };
  bool refused = true, untouched = true, fallback = true;
  for (const Case& c : cases) {
    float out[4] = {7.0f, 7.0f, 7.0f, 7.0f};
    refused &= rmsnorm_f32_s3(out, c.x, nullptr, 4, 0.25f, c.eps) == 0;
    for (float v : out) {
      untouched &= v == 7.0f;
    }
    float got[4], want[4];
    rmsnorm_row(true, got, c.x, nullptr, 4, 0.25f, c.eps);
    rmsnorm_scalar(want, c.x, nullptr, 4, 0.25f, c.eps);
    fallback &= memcmp(got, want, sizeof(got)) == 0;
  }
  printf("rmsnorm domain: refused %s, untouched %s, scalar fallback %s\n",
         refused ? "yes" : "NO", untouched ? "yes" : "NO",
         fallback ? "yes" : "NO");
}

// ---- the standard-op fast paths against ExecuTorch's portable kernels ----
//
// A fast path must give the portable kernel's bytes for every input it takes
// and leave every other input to that kernel. Each case runs the override
// and the portable kernel on their own copies of the same buffers, compares
// every byte either may have written and the two failure states, and names
// the path the override must take, read from its dispatch counters: a near
// miss wrongly taken, or a story shape wrongly refused, fails even when the
// two results agree. The counters are restored afterwards, so a run's
// dispatch report still counts only its programs' calls.
namespace {

using executorch::aten::ScalarType;
using executorch::aten::TensorImpl;

constexpr int kCheckDims = 4;
constexpr int kCheckCounters = 4;

struct CheckShape {
  int dim;
  int sizes[kCheckDims];

  size_t numel() const {
    size_t count = 1;
    for (int axis = 0; axis < dim; ++axis) {
      count *= static_cast<size_t>(sizes[axis]);
    }
    return count;
  }
};

// kSingletonStride0: the default dim order and contiguous strides, but every
// size-1 axis with stride 0, which ExecuTorch accepts (a singleton axis may
// carry any stride) and the portable pad kernel reads.
enum class CheckLayout { kDense, kReversed, kChannelsLast, kSingletonStride0 };

// A tensor over caller-owned bytes. TensorImpl points at these arrays, so a
// CheckTensor stays where it was constructed.
class CheckTensor {
 public:
  CheckTensor(ScalarType type, const CheckShape& shape, void* data,
              CheckLayout layout = CheckLayout::kDense)
      : impl_(type, describe(shape, layout), sizes_, data, order_, strides_) {}
  CheckTensor(const CheckTensor&) = delete;
  CheckTensor& operator=(const CheckTensor&) = delete;
  Tensor tensor() { return Tensor(&impl_); }

 private:
  ssize_t describe(const CheckShape& shape, CheckLayout layout) {
    static const TensorImpl::DimOrderType kNhwc[4] = {0, 2, 3, 1};
    for (int axis = 0; axis < shape.dim; ++axis) {
      sizes_[axis] = shape.sizes[axis];
      order_[axis] = layout == CheckLayout::kReversed ? shape.dim - 1 - axis
          : layout == CheckLayout::kChannelsLast      ? kNhwc[axis]
                                                      : axis;
    }
    TensorImpl::StridesType stride = 1;
    for (int position = shape.dim; position-- > 0;) {
      strides_[order_[position]] = stride;
      stride *= sizes_[order_[position]];
    }
    if (layout == CheckLayout::kSingletonStride0) {
      for (int axis = 0; axis < shape.dim; ++axis) {
        if (sizes_[axis] == 1) {
          strides_[axis] = 0;
        }
      }
    }
    return shape.dim;
  }

  TensorImpl::SizesType sizes_[kCheckDims];
  TensorImpl::DimOrderType order_[kCheckDims];
  TensorImpl::StridesType strides_[kCheckDims];
  TensorImpl impl_;
};

uint32_t g_check_random = 0x2545f491u;

uint32_t check_random() {
  uint32_t x = g_check_random;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return g_check_random = x;
}

// The IEEE special values: signed zeros and infinities, quiet and signalling
// NaNs with different payloads, denormals, the extremes, +-1.
const uint32_t kCheckF32Special[16] = {
    0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u,
    0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0xffa00001u,
    0x7f800001u, 0x00000001u, 0x807fffffu, 0x00800000u,
    0x7f7fffffu, 0xff7fffffu, 0x3f800000u, 0xbf800000u};

// fp32 bit patterns after the specials: specials again, arbitrary patterns
// and finite values of moderate size.
uint32_t check_f32_bits() {
  const uint32_t r = check_random();
  switch (r & 7u) {
    case 0:
      return kCheckF32Special[(r >> 3) & 15u];
    case 1:
    case 2:
      return check_random();
    default:
      return (r & 0x80000000u) | ((100u + (r >> 3) % 56u) << 23) |
          (check_random() & 0x007fffffu);
  }
}

// Integers in [low, high], with the type's extremes mixed in when asked.
int64_t check_integer(int64_t low, int64_t high, bool extremes) {
  const uint32_t r = check_random();
  if (extremes && (r & 15u) == 0) {
    return (r & 16u) ? INT64_MAX : INT64_MIN;
  }
  const uint64_t span = static_cast<uint64_t>(high - low) + 1;
  return low + static_cast<int64_t>(
      ((static_cast<uint64_t>(check_random()) << 32) | check_random()) % span);
}

void check_fill(ScalarType type, void* data, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    switch (type) {
      case ScalarType::Float:  // every special value first, at least once
        static_cast<uint32_t*>(data)[index] =
            index < 16 ? kCheckF32Special[index] : check_f32_bits();
        break;
      case ScalarType::Double:
        static_cast<double*>(data)[index] =
            static_cast<double>(check_integer(-1000, 1000, false)) / 7.0;
        break;
      case ScalarType::Long:
        static_cast<int64_t*>(data)[index] = check_integer(-130, 130, true);
        break;
      case ScalarType::Int:
        static_cast<int32_t*>(data)[index] =
            static_cast<int32_t>(check_integer(-130, 130, false));
        break;
      case ScalarType::Bool:
        static_cast<uint8_t*>(data)[index] = check_random() & 1u;
        break;
      default:  // one-byte and other types: arbitrary bytes
        for (size_t byte = 0; byte < executorch::runtime::elementSize(type);
             ++byte) {
          static_cast<unsigned char*>(data)
              [index * executorch::runtime::elementSize(type) + byte] =
              static_cast<unsigned char>(check_random());
        }
        break;
    }
  }
}

size_t check_bytes(ScalarType type, const CheckShape& shape) {
  return shape.numel() * executorch::runtime::elementSize(type);
}

// One operator's counters and results.
struct CheckOp {
  const char* name;
  uint64_t* counters[kCheckCounters];
  uint64_t* fallback;
  uint64_t saved[kCheckCounters] = {};
  int cases = 0;
  int fast = 0;
  int fallbacks = 0;
  int mismatches = 0;
};

class FastpathCheck {
 public:
  explicit FastpathCheck(size_t bytes)
      : arena_(static_cast<unsigned char*>(malloc(bytes))),
        capacity_(arena_ != nullptr ? bytes : 0) {}
  ~FastpathCheck() { free(arena_); }
  FastpathCheck(const FastpathCheck&) = delete;
  FastpathCheck& operator=(const FastpathCheck&) = delete;

  bool ok() const { return arena_ != nullptr; }

  // Every case starts with an empty arena and a seed of its own, so a
  // failing case can be reproduced alone.
  void begin_case() {
    used_ = 0;
    g_check_random = 0x2545f491u ^ (0x9e3779b9u * ++seed_);
  }

  void* take(size_t bytes) {
    const size_t start = (used_ + 15u) & ~static_cast<size_t>(15u);
    if (arena_ == nullptr || start + bytes > capacity_) {
      exhausted_ = true;
      return nullptr;
    }
    used_ = start + bytes;
    return arena_ + start;
  }

  // Bytes the call may write, initialised to a sentinel; a case then copies
  // any input that lives in them (in-place operands, aliases).
  unsigned char* take_written(size_t bytes) {
    unsigned char* data = static_cast<unsigned char*>(take(bytes));
    if (data != nullptr) {
      memset(data, 0xa5, bytes);
    }
    return data;
  }

  // One case. `run(fast, written, ctx)` builds the case's tensors with every
  // byte the call may write inside `written` and calls the override (fast)
  // or the portable kernel; both sides start from `initial`.
  template <typename Run>
  void compare(CheckOp& op, const char* what, uint64_t* expect,
               const unsigned char* initial, size_t bytes, Run run) {
    unsigned char* sides[2] = {static_cast<unsigned char*>(take(bytes)),
                               static_cast<unsigned char*>(take(bytes))};
    ++op.cases;
    ++(expect == op.fallback ? op.fallbacks : op.fast);
    if (initial == nullptr || sides[0] == nullptr || sides[1] == nullptr) {
      mismatch(op, what, "check arena exhausted");
      return;
    }
    memcpy(sides[0], initial, bytes);
    memcpy(sides[1], initial, bytes);
    uint64_t before[kCheckCounters] = {};
    for (int index = 0; index < kCheckCounters && op.counters[index]; ++index) {
      before[index] = *op.counters[index];
    }
    KernelRuntimeContext fast_ctx;
    KernelRuntimeContext portable_ctx;
    run(true, sides[0], fast_ctx);
    run(false, sides[1], portable_ctx);
    uint64_t moved = 0;
    bool expected = false;
    for (int index = 0; index < kCheckCounters && op.counters[index]; ++index) {
      const uint64_t delta = *op.counters[index] - before[index];
      moved += delta;
      expected |= op.counters[index] == expect && delta == 1;
    }
    if (moved != 1 || !expected) {
      mismatch(op, what, expect == op.fallback
                             ? "took a fast path, expected the portable kernel"
                             : "did not take its fast path");
    } else if (fast_ctx.failure_state() != portable_ctx.failure_state()) {
      mismatch(op, what, "failure states differ");
    } else if (memcmp(sides[0], sides[1], bytes) != 0) {
      size_t offset = 0;
      while (sides[0][offset] == sides[1][offset]) {
        ++offset;
      }
      printf("fastpath check: MISMATCH %s \"%s\": byte %u of %u differs "
             "(%02x fast, %02x portable)\n",
             op.name, what, (unsigned)offset, (unsigned)bytes,
             sides[0][offset], sides[1][offset]);
      ++op.mismatches;
    }
  }

  void mismatch(CheckOp& op, const char* what, const char* why) {
    printf("fastpath check: MISMATCH %s \"%s\": %s\n", op.name, what, why);
    ++op.mismatches;
  }

  bool exhausted() const { return exhausted_; }

 private:
  unsigned char* arena_;
  size_t capacity_;
  size_t used_ = 0;
  uint32_t seed_ = 0;
  bool exhausted_ = false;
};

// mul.out. A one-element operand takes the value `scale` (fp32 bits).
enum class MulAlias { kNone, kOutIsLeft, kRightInOut };

struct MulCase {
  const char* what;
  uint64_t* expect;
  ScalarType left_type, right_type, out_type;
  CheckShape left, right, out;
  uint32_t scale;
  CheckLayout left_layout;
  MulAlias alias;
};

void check_mul(FastpathCheck& check, CheckOp& op, const MulCase& c) {
  check.begin_case();
  const size_t left_bytes = check_bytes(c.left_type, c.left);
  const size_t right_bytes = check_bytes(c.right_type, c.right);
  void* left_data = check.take(left_bytes);
  void* right_data = check.take(right_bytes);
  unsigned char* initial = check.take_written(check_bytes(c.out_type, c.out));
  if (left_data == nullptr || right_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  check_fill(c.left_type, left_data, c.left.numel());
  check_fill(c.right_type, right_data, c.right.numel());
  if (c.left.numel() == 1 && c.left_type == ScalarType::Float) {
    memcpy(left_data, &c.scale, sizeof(c.scale));
  }
  if (c.right.numel() == 1 && c.right_type == ScalarType::Float) {
    memcpy(right_data, &c.scale, sizeof(c.scale));
  }
  if (c.alias == MulAlias::kOutIsLeft) {
    memcpy(initial, left_data, left_bytes);
  } else if (c.alias == MulAlias::kRightInOut) {
    memcpy(initial, right_data, right_bytes);
  }
  check.compare(
      op, c.what, c.expect, initial, check_bytes(c.out_type, c.out),
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor left(c.left_type, c.left,
                         c.alias == MulAlias::kOutIsLeft ? written : left_data,
                         c.left_layout);
        CheckTensor right(
            c.right_type, c.right,
            c.alias == MulAlias::kRightInOut ? written : right_data);
        CheckTensor out(c.out_type, c.out, written);
        if (fast) {
          EValue left_value(left.tensor()), right_value(right.tensor());
          EValue out_value(out.tensor());
          EValue* stack[4] = {&left_value, &right_value, &out_value, &out_value};
          shim_mul_out(ctx, Span<EValue*>(stack, 4));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::mul_out(
              ctx, left.tensor(), right.tensor(), out_tensor);
        }
      });
}

void check_mul_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  constexpr MulAlias kNone = MulAlias::kNone;
  const CheckShape scalar = {0, {}};
  const CheckShape small = {2, {5, 7}};
  // The scales: 1/sqrt(32), signed zeros, infinities, NaNs with payloads
  // (quiet and signalling), a denormal and the largest finite value.
  const uint32_t kScales[] = {0x3e3504f3u, 0x80000000u, 0x00000000u,
                              0x7f800000u, 0xff800000u, 0x7fc12345u,
                              0xff800001u, 0x00000003u, 0x7f7fffffu};
  const MulCase story[] = {
      {"story query [8,32] x 0-d", &g_mul_scalar, F32, F32, F32,
       {2, {8, 32}}, scalar, {2, {8, 32}}, 0x3e3504f3u, kDense, kNone},
      {"story [1,2048] x 0-d", &g_mul_scalar, F32, F32, F32,
       {2, {1, 2048}}, scalar, {2, {1, 2048}}, 0x41800000u, kDense, kNone},
      {"story [1,8,256] x 0-d", &g_mul_scalar, F32, F32, F32,
       {3, {1, 8, 256}}, scalar, {3, {1, 8, 256}}, 0x3d800000u, kDense,
       kNone},
  };
  for (const MulCase& c : story) {
    check_mul(check, op, c);
  }
  // Every scale against every special element; a NaN scale stays portable.
  for (uint32_t scale : kScales) {
    uint64_t* path =
        (scale & 0x7fffffffu) > 0x7f800000u ? &g_mul_fallback : &g_mul_scalar;
    check_mul(check, op, {"[5,7] x 0-d special scale", path, F32, F32, F32,
                          small, scalar, small, scale, kDense, kNone});
    check_mul(check, op, {"0-d x [5,7] special scale", path, F32, F32, F32,
                          scalar, small, small, scale, kDense, kNone});
  }
  const MulCase cases[] = {
      {"0-d x [4,6]", &g_mul_scalar, F32, F32, F32, scalar, {2, {4, 6}},
       {2, {4, 6}}, 0x3f000001u, kDense, kNone},
      {"[9] x [1]", &g_mul_scalar, F32, F32, F32, {1, {9}}, {1, {1}},
       {1, {9}}, 0xc0490fdbu, kDense, kNone},
      {"[2,3,4] x [1,1]", &g_mul_scalar, F32, F32, F32, {3, {2, 3, 4}},
       {2, {1, 1}}, {3, {2, 3, 4}}, 0x3eaaaaabu, kDense, kNone},
      {"[4,6] x 0-d in place", &g_mul_scalar, F32, F32, F32, {2, {4, 6}},
       scalar, {2, {4, 6}}, 0x3e3504f3u, kDense, MulAlias::kOutIsLeft},
      {"[3,5] x [3,5]", &g_mul_same, F32, F32, F32, {2, {3, 5}},
       {2, {3, 5}}, {2, {3, 5}}, 0, kDense, kNone},
      {"[2,3,4] x [3,1]", &g_mul_broadcast, F32, F32, F32, {3, {2, 3, 4}},
       {2, {3, 1}}, {3, {2, 3, 4}}, 0, kDense, kNone},
      {"[2,3] x [1,1,1] (one element, higher rank)", &g_mul_broadcast, F32,
       F32, F32, {2, {2, 3}}, {3, {1, 1, 1}}, {3, {1, 2, 3}}, 0x3e3504f3u,
       kDense, kNone},
      {"[4,6] x f64 0-d", &g_mul_fallback, F32, ScalarType::Double, F32,
       {2, {4, 6}}, scalar, {2, {4, 6}}, 0, kDense, kNone},
      {"f64 [4,6] x f64 0-d", &g_mul_fallback, ScalarType::Double,
       ScalarType::Double, ScalarType::Double, {2, {4, 6}}, scalar,
       {2, {4, 6}}, 0, kDense, kNone},
      {"i32 [4,6] x i32 0-d", &g_mul_fallback, ScalarType::Int,
       ScalarType::Int, ScalarType::Int, {2, {4, 6}}, scalar, {2, {4, 6}}, 0,
       kDense, kNone},
      {"[4,6] x 0-d stored in the output", &g_mul_fallback, F32, F32, F32,
       {2, {4, 6}}, scalar, {2, {4, 6}}, 0x3e3504f3u, kDense,
       MulAlias::kRightInOut},
      {"[4,3] column-major x 0-d", &g_mul_fallback, F32, F32, F32,
       {2, {4, 3}}, scalar, {2, {4, 3}}, 0x3e3504f3u, CheckLayout::kReversed,
       kNone},
      {"channels-last [1,4,1,1] x 0-d", &g_mul_fallback, F32, F32, F32,
       {4, {1, 4, 1, 1}}, scalar, {4, {1, 4, 1, 1}}, 0x3e3504f3u,
       CheckLayout::kChannelsLast, kNone},
      {"channels-last [1,4,1,1] x [1,4,1,1]", &g_mul_fallback, F32, F32, F32,
       {4, {1, 4, 1, 1}}, {4, {1, 4, 1, 1}}, {4, {1, 4, 1, 1}}, 0,
       CheckLayout::kChannelsLast, kNone},
      {"[4,6] x 0-d into [4,7]", &g_mul_fallback, F32, F32, F32,
       {2, {4, 6}}, scalar, {2, {4, 7}}, 0x3e3504f3u, kDense, kNone},
  };
  for (const MulCase& c : cases) {
    check_mul(check, op, c);
  }
}

// le.Tensor_out. A one-element int64 `other` holds `bound`, and an int64
// `self` starts with the bound and its neighbours. With `out_over_self`,
// out's bytes start at self's.
struct LeCase {
  const char* what;
  uint64_t* expect;
  ScalarType self_type, other_type, out_type;
  CheckShape self, other, out;
  int64_t bound;
  CheckLayout self_layout;
  bool out_over_self;
};

void check_le(FastpathCheck& check, CheckOp& op, const LeCase& c) {
  check.begin_case();
  const size_t self_bytes = check_bytes(c.self_type, c.self);
  const size_t written_bytes = c.out_over_self
      ? std::max(self_bytes, check_bytes(c.out_type, c.out))
      : check_bytes(c.out_type, c.out);
  void* self_data = check.take(self_bytes);
  void* other_data = check.take(check_bytes(c.other_type, c.other));
  unsigned char* initial = check.take_written(written_bytes);
  if (self_data == nullptr || other_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  check_fill(c.self_type, self_data, c.self.numel());
  check_fill(c.other_type, other_data, c.other.numel());
  if (c.self_type == ScalarType::Long) {
    const int64_t near[3] = {c.bound, c.bound > INT64_MIN ? c.bound - 1 : c.bound,
                             c.bound < INT64_MAX ? c.bound + 1 : c.bound};
    for (size_t index = 0; index < 3 && index < c.self.numel(); ++index) {
      static_cast<int64_t*>(self_data)[index] = near[index];
    }
  }
  if (c.other_type == ScalarType::Long && c.other.numel() == 1) {
    memcpy(other_data, &c.bound, sizeof(c.bound));
  }
  if (c.out_over_self) {
    memcpy(initial, self_data, self_bytes);
  }
  check.compare(
      op, c.what, c.expect, initial, written_bytes,
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor self(c.self_type, c.self,
                         c.out_over_self ? written : self_data, c.self_layout);
        CheckTensor other(c.other_type, c.other, other_data);
        CheckTensor out(c.out_type, c.out, written);
        if (fast) {
          EValue self_value(self.tensor()), other_value(other.tensor());
          EValue out_value(out.tensor());
          EValue* stack[4] = {&self_value, &other_value, &out_value, &out_value};
          shim_le_tensor_out(ctx, Span<EValue*>(stack, 4));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::le_tensor_out(
              ctx, self.tensor(), other.tensor(), out_tensor);
        }
      });
}

void check_le_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType I64 = ScalarType::Long;
  constexpr ScalarType B = ScalarType::Bool;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  const CheckShape scalar = {0, {}};
  const CheckShape window = {1, {64}};
  const LeCase cases[] = {
      {"story window [64] <= 0-d", &g_le_scalar, I64, I64, B, window, scalar,
       window, 37, kDense, false},
      {"story context [256] <= 0-d", &g_le_scalar, I64, I64, B, {1, {256}},
       scalar, {1, {256}}, 200, kDense, false},
      {"[64] <= INT64_MIN", &g_le_scalar, I64, I64, B, window, scalar, window,
       INT64_MIN, kDense, false},
      {"[64] <= INT64_MAX", &g_le_scalar, I64, I64, B, window, scalar, window,
       INT64_MAX, kDense, false},
      {"[64] <= -1", &g_le_scalar, I64, I64, B, window, scalar, window, -1,
       kDense, false},
      {"[2,3,4] <= 0-d", &g_le_scalar, I64, I64, B, {3, {2, 3, 4}}, scalar,
       {3, {2, 3, 4}}, 0, kDense, false},
      {"[5] <= [1]", &g_le_scalar, I64, I64, B, {1, {5}}, {1, {1}}, {1, {5}},
       3, kDense, false},
      {"[2,5] <= [1,1]", &g_le_scalar, I64, I64, B, {2, {2, 5}}, {2, {1, 1}},
       {2, {2, 5}}, -4, kDense, false},
      {"prefill [1,4] <= [4,1]", &g_le_fallback, I64, I64, B, {2, {1, 4}},
       {2, {4, 1}}, {2, {4, 4}}, 0, kDense, false},
      {"i32 [8] <= i32 0-d", &g_le_fallback, ScalarType::Int, ScalarType::Int,
       B, {1, {8}}, scalar, {1, {8}}, 0, kDense, false},
      {"f32 [20] <= f32 0-d", &g_le_fallback, ScalarType::Float,
       ScalarType::Float, B, {1, {20}}, scalar, {1, {20}}, 0, kDense, false},
      {"[8] <= 0-d into int64", &g_le_fallback, I64, I64, I64, {1, {8}},
       scalar, {1, {8}}, 2, kDense, false},
      {"[8] <= 0-d into [9]", &g_le_fallback, I64, I64, B, {1, {8}}, scalar,
       {1, {9}}, 2, kDense, false},
      {"[4,3] column-major <= 0-d", &g_le_fallback, I64, I64, B, {2, {4, 3}},
       scalar, {2, {4, 3}}, 2, CheckLayout::kReversed, false},
      {"[8] <= [8]", &g_le_fallback, I64, I64, B, {1, {8}}, {1, {8}},
       {1, {8}}, 0, kDense, false},
      {"[4] <= [1,1] (one element, higher rank)", &g_le_fallback, I64, I64, B,
       {1, {4}}, {2, {1, 1}}, {2, {1, 4}}, 1, kDense, false},
      {"[8] <= 0-d written over self", &g_le_fallback, I64, I64, B, {1, {8}},
       scalar, {1, {8}}, 1, kDense, true},
  };
  for (const LeCase& c : cases) {
    check_le(check, op, c);
  }
}

// index_put_ (in place, so the written bytes are self's, which start
// random). `axis` is the index's position in a list of `list_size`, or -1 for
// two index tensors at 0 and 1; `positions` are the index values. With
// `values_in_self`, values are self's last bytes.
struct IndexPutCase {
  const char* what;
  uint64_t* expect;
  ScalarType self_type, values_type, index_type;
  CheckShape self, values, index;
  int axis;
  int list_size;
  int64_t positions[3];
  bool accumulate;
  CheckLayout self_layout;
  bool values_in_self;
};

void check_index_put(FastpathCheck& check, CheckOp& op, const IndexPutCase& c) {
  check.begin_case();
  const size_t self_bytes = check_bytes(c.self_type, c.self);
  const size_t values_bytes = check_bytes(c.values_type, c.values);
  void* values_data = check.take(values_bytes);
  void* index_data = check.take(check_bytes(c.index_type, c.index));
  unsigned char* initial = check.take_written(self_bytes);
  if (values_data == nullptr || index_data == nullptr || initial == nullptr ||
      values_bytes > self_bytes) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  check_fill(c.self_type, initial, c.self.numel());
  check_fill(c.values_type, values_data, c.values.numel());
  for (size_t index = 0; index < c.index.numel() && index < 3; ++index) {
    if (c.index_type == ScalarType::Long) {
      static_cast<int64_t*>(index_data)[index] = c.positions[index];
    } else {
      static_cast<int32_t*>(index_data)[index] =
          static_cast<int32_t>(c.positions[index]);
    }
  }
  check.compare(
      op, c.what, c.expect, initial, self_bytes,
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor self(c.self_type, c.self, written, c.self_layout);
        CheckTensor values(
            c.values_type, c.values,
            c.values_in_self ? written + self_bytes - values_bytes
                             : values_data);
        CheckTensor index(c.index_type, c.index, index_data);
        std::optional<Tensor> list[kCheckDims];
        for (int position = 0; position < c.list_size; ++position) {
          if (position == c.axis || (c.axis < 0 && position < 2)) {
            list[position] = index.tensor();
          }
        }
        Tensor self_tensor = self.tensor();
        if (fast) {
          // The program's form: a boxed list over None and Tensor values.
          EValue none_value;
          EValue index_value(index.tensor());
          EValue* boxed[kCheckDims];
          std::optional<Tensor> unboxed[kCheckDims];
          for (int position = 0; position < c.list_size; ++position) {
            boxed[position] = list[position].has_value() ? &index_value
                                                         : &none_value;
          }
          executorch::runtime::BoxedEvalueList<std::optional<Tensor>> boxed_list(
              boxed, unboxed, c.list_size);
          EValue self_value(self_tensor), list_value(&boxed_list);
          EValue values_value(values.tensor()), accumulate_value(c.accumulate);
          EValue* stack[5] = {&self_value, &list_value, &values_value,
                              &accumulate_value, &self_value};
          shim_index_put_(ctx, Span<EValue*>(stack, 5));
        } else {
          torch::executor::native::index_put_(
              ctx, self_tensor,
              executorch::aten::ArrayRef<std::optional<Tensor>>(
                  list, static_cast<size_t>(c.list_size)),
              values.tensor(), c.accumulate);
        }
      });
}

void check_index_put_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType I8 = ScalarType::Char;
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr ScalarType I64 = ScalarType::Long;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  const CheckShape one = {1, {1}};
  const CheckShape keys = {3, {1, 64, 32}};
  const CheckShape key_row = {3, {1, 1, 32}};
  const IndexPutCase cases[] = {
      {"story int8 K [1,64,32] row 17", &g_index_put_rows, I8, I8, I64, keys,
       key_row, one, 1, 2, {17}, false, kDense, false},
      {"story K scales [1,64] row 63", &g_index_put_rows, F32, F32, I64,
       {2, {1, 64}}, {2, {1, 1}}, one, 1, 2, {63}, false, kDense, false},
      {"story V [1,64,32] row 0", &g_index_put_rows, F32, F32, I64, keys,
       key_row, one, 1, 2, {0}, false, kDense, false},
      {"story global int8 K [1,256,32] row 255", &g_index_put_rows, I8, I8,
       I64, {3, {1, 256, 32}}, key_row, one, 1, 2, {255}, false, kDense,
       false},
      {"prefill rows 5,1,5 of [1,8,4]", &g_index_put_rows, F32, F32, I64,
       {3, {1, 8, 4}}, {3, {1, 3, 4}}, {1, {3}}, 1, 2, {5, 1, 5}, false,
       kDense, false},
      {"rows 3,6 under two leading [2,8,4]", &g_index_put_rows, F32, F32, I64,
       {3, {2, 8, 4}}, {3, {2, 2, 4}}, {1, {2}}, 1, 2, {3, 6}, false, kDense,
       false},
      {"row 2 on axis 0 of [8,4]", &g_index_put_rows, F32, F32, I64,
       {2, {8, 4}}, {2, {1, 4}}, one, 0, 1, {2}, false, kDense, false},
      {"column 3 of [2,8]", &g_index_put_rows, F32, F32, I64, {2, {2, 8}},
       {2, {2, 1}}, one, 1, 2, {3}, false, kDense, false},
      {"int64 cache [1,4,2] row 1", &g_index_put_rows, I64, I64, I64,
       {3, {1, 4, 2}}, {3, {1, 1, 2}}, one, 1, 2, {1}, false, kDense, false},
      // Indices from the end, as portable takes them since #23529.
      {"row -1 (the last)", &g_index_put_rows, I8, I8, I64, keys, key_row, one,
       1, 2, {-1}, false, kDense, false},
      {"row -64 of 64 (the first)", &g_index_put_rows, F32, F32, I64, keys,
       key_row, one, 1, 2, {-64}, false, kDense, false},
      {"rows -1,3,-5 of [1,8,4] (-5 is row 3 again)", &g_index_put_rows, F32,
       F32, I64, {3, {1, 8, 4}}, {3, {1, 3, 4}}, {1, {3}}, 1, 2, {-1, 3, -5},
       false, kDense, false},
      {"rows -2,6 under two leading [2,8,4]", &g_index_put_rows, F32, F32,
       I64, {3, {2, 8, 4}}, {3, {2, 2, 4}}, {1, {2}}, 1, 2, {-2, 6}, false,
       kDense, false},
      // Values without leading size-1 dimensions (#23529): the same bytes.
      {"Mistral window [1,2,6,8] row -1, values [2,1,8]", &g_index_put_rows,
       F32, F32, I64, {4, {1, 2, 6, 8}}, {3, {2, 1, 8}}, one, 2, 3, {-1},
       false, kDense, false},
      {"row [1,32] for [1,64,32]", &g_index_put_rows, I8, I8, I64, keys,
       {2, {1, 32}}, one, 1, 2, {17}, false, kDense, false},
      {"row [32] for [1,64,32] (the axis left out too)", &g_index_put_rows,
       F32, F32, I64, keys, {1, {32}}, one, 1, 2, {40}, false, kDense, false},
      {"rows 5,1,-1 of [1,8,4] from [3,4]", &g_index_put_rows, F32, F32, I64,
       {3, {1, 8, 4}}, {2, {3, 4}}, {1, {3}}, 1, 2, {5, 1, -1}, false, kDense,
       false},
      {"0-d value into [4,1] row 2", &g_index_put_rows, F32, F32, I64,
       {2, {4, 1}}, {0, {}}, one, 0, 1, {2}, false, kDense, false},
      {"accumulate", &g_index_put_fallback, I8, I8, I64, keys, key_row, one, 1,
       2, {17}, true, kDense, false},
      {"row 64 of 64", &g_index_put_fallback, I8, I8, I64, keys, key_row, one,
       1, 2, {64}, false, kDense, false},
      {"row -65 of 64", &g_index_put_fallback, I8, I8, I64, keys, key_row, one,
       1, 2, {-65}, false, kDense, false},
      {"values [2,4] for rows 3,6 of [2,8,4] (a leading 2 left out)",
       &g_index_put_fallback, F32, F32, I64, {3, {2, 8, 4}}, {2, {2, 4}},
       {1, {2}}, 1, 2, {3, 6}, false, kDense, false},
      {"values [32] for rows 5,1 of [1,8,32] (an axis of 2 left out)",
       &g_index_put_fallback, F32, F32, I64, {3, {1, 8, 32}}, {1, {32}},
       {1, {2}}, 1, 2, {5, 1}, false, kDense, false},
      {"values [1,1,32] for [64,32] (one dimension too many)",
       &g_index_put_fallback, F32, F32, I64, {2, {64, 32}}, {3, {1, 1, 32}},
       one, 0, 1, {9}, false, kDense, false},
      {"int32 index", &g_index_put_fallback, F32, F32, ScalarType::Int, keys,
       key_row, one, 1, 2, {9}, false, kDense, false},
      {"fp32 row into an int8 cache", &g_index_put_fallback, I8, F32, I64,
       keys, key_row, one, 1, 2, {9}, false, kDense, false},
      {"two index tensors", &g_index_put_fallback, F32, F32, I64,
       {2, {8, 4}}, {1, {1}}, one, -1, 2, {2}, false, kDense, false},
      {"row [1,1,16] for [1,64,32]", &g_index_put_fallback, F32, F32, I64,
       keys, {3, {1, 1, 16}}, one, 1, 2, {9}, false, kDense, false},
      {"2-D index", &g_index_put_fallback, F32, F32, I64, keys, key_row,
       {2, {1, 1}}, 1, 2, {9}, false, kDense, false},
      {"index list longer than the rank", &g_index_put_fallback, F32, F32,
       I64, {2, {8, 4}}, {2, {8, 1}}, one, 1, 3, {2}, false, kDense, false},
      {"column-major [8,4]", &g_index_put_fallback, F32, F32, I64,
       {2, {8, 4}}, {2, {1, 4}}, one, 0, 1, {2}, false,
       CheckLayout::kReversed, false},
      {"row read from inside the cache", &g_index_put_fallback, F32, F32, I64,
       keys, key_row, one, 1, 2, {9}, false, kDense, true},
  };
  for (const IndexPutCase& c : cases) {
    check_index_put(check, op, c);
  }
}

// split_with_sizes_copy.out into `pieces` outputs of `sizes` along `dim`,
// laid out one after another in the written bytes. `last_piece_delta`
// changes the last output's size on that axis (a wrong output shape); with
// `piece_over_self`, the input is copied into the written bytes and the
// first output starts at it.
constexpr int kCheckPieces = 8;

struct SplitCase {
  const char* what;
  uint64_t* expect;
  ScalarType self_type, out_type;
  CheckShape self;
  int dim;
  int pieces;
  int sizes[kCheckPieces];
  CheckLayout self_layout;
  int last_piece_delta;
  bool piece_over_self;
};

void check_split(FastpathCheck& check, CheckOp& op, const SplitCase& c) {
  check.begin_case();
  const int axis = c.dim < 0 ? c.dim + c.self.dim : c.dim;
  const int shape_axis = axis >= 0 && axis < c.self.dim ? axis : 0;
  CheckShape shapes[kCheckPieces];
  size_t offsets[kCheckPieces];
  const size_t self_bytes = check_bytes(c.self_type, c.self);
  size_t written_bytes = c.piece_over_self ? self_bytes : 0;
  for (int piece = 0; piece < c.pieces; ++piece) {
    shapes[piece] = c.self;
    shapes[piece].sizes[shape_axis] = std::max(
        0, c.sizes[piece] + (piece == c.pieces - 1 ? c.last_piece_delta : 0));
    offsets[piece] = piece == 0 && c.piece_over_self ? 0 : written_bytes;
    if (!(piece == 0 && c.piece_over_self)) {
      written_bytes += (check_bytes(c.out_type, shapes[piece]) + 15u) & ~15u;
    }
  }
  void* self_data = check.take(self_bytes);
  unsigned char* initial = check.take_written(written_bytes);
  if (self_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  check_fill(c.self_type, self_data, c.self.numel());
  if (c.piece_over_self) {
    memcpy(initial, self_data, self_bytes);
  }
  check.compare(
      op, c.what, c.expect, initial, written_bytes,
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor self(c.self_type, c.self,
                         c.piece_over_self ? written : self_data,
                         c.self_layout);
        // TensorImpls, Tensors and the boxed lists as a program holds them:
        // a Tensor has no default constructor, so its storage is raw.
        alignas(CheckTensor) unsigned char piece_storage[kCheckPieces]
                                                        [sizeof(CheckTensor)];
        alignas(Tensor) unsigned char tensor_storage[kCheckPieces][sizeof(Tensor)];
        Tensor* tensors = reinterpret_cast<Tensor*>(tensor_storage);
        EValue piece_values[kCheckPieces];
        EValue* boxed_pieces[kCheckPieces];
        EValue size_values[kCheckPieces];
        EValue* boxed_sizes[kCheckPieces];
        int64_t sizes[kCheckPieces];
        for (int piece = 0; piece < c.pieces; ++piece) {
          CheckTensor* tensor = new (piece_storage[piece])
              CheckTensor(c.out_type, shapes[piece], written + offsets[piece]);
          new (&tensors[piece]) Tensor(tensor->tensor());
          piece_values[piece] = EValue(tensors[piece]);
          boxed_pieces[piece] = &piece_values[piece];
          sizes[piece] = c.sizes[piece];
          size_values[piece] = EValue(sizes[piece]);
          boxed_sizes[piece] = &size_values[piece];
        }
        if (fast) {
          int64_t unboxed_sizes[kCheckPieces];
          executorch::runtime::BoxedEvalueList<int64_t> size_list(
              boxed_sizes, unboxed_sizes, c.pieces);
          executorch::runtime::BoxedEvalueList<Tensor> piece_list(
              boxed_pieces, tensors, c.pieces);
          EValue self_value(self.tensor()), sizes_value(&size_list);
          EValue dim_value(static_cast<int64_t>(c.dim)), out_value(&piece_list);
          EValue* stack[5] = {&self_value, &sizes_value, &dim_value, &out_value,
                              &out_value};
          shim_split_with_sizes_copy_out(ctx, Span<EValue*>(stack, 5));
        } else {
          torch::executor::native::split_with_sizes_copy_out(
              ctx, self.tensor(),
              IntArrayRef(sizes, static_cast<size_t>(c.pieces)), c.dim,
              executorch::aten::ArrayRef<Tensor>(
                  tensors, static_cast<size_t>(c.pieces)));
        }
      });
}

void check_split_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  const SplitCase cases[] = {
      {"story [1,8,256] into eight on dim 1", &g_split_chunks, F32, F32,
       {3, {1, 8, 256}}, 1, 8, {1, 1, 1, 1, 1, 1, 1, 1}, kDense, 0, false},
      {"glimmer [1,4,24] into four on dim 1", &g_split_chunks, F32, F32,
       {3, {1, 4, 24}}, 1, 4, {1, 1, 1, 1}, kDense, 0, false},
      {"[2,6] into 2,4 on dim -1", &g_split_chunks, F32, F32, {2, {2, 6}}, -1,
       2, {2, 4}, kDense, 0, false},
      {"[3,5,2] into 1,3,1 on dim 1", &g_split_chunks, F32, F32,
       {3, {3, 5, 2}}, 1, 3, {1, 3, 1}, kDense, 0, false},
      {"[4,3] into 0,4 on dim 0", &g_split_chunks, F32, F32, {2, {4, 3}}, 0, 2,
       {0, 4}, kDense, 0, false},
      {"[6] into one piece", &g_split_chunks, F32, F32, {1, {6}}, 0, 1, {6},
       kDense, 0, false},
      {"int32 [2,6]", &g_split_fallback, ScalarType::Int, ScalarType::Int,
       {2, {2, 6}}, 1, 2, {2, 4}, kDense, 0, false},
      {"fp32 into fp64 pieces", &g_split_fallback, F32, ScalarType::Double,
       {2, {2, 6}}, 1, 2, {2, 4}, kDense, 0, false},
      {"sizes 2,3 for 6", &g_split_fallback, F32, F32, {2, {2, 6}}, 1, 2,
       {2, 3}, kDense, 0, false},
      {"sizes -1,7", &g_split_fallback, F32, F32, {2, {2, 6}}, 1, 2, {-1, 7},
       kDense, 0, false},
      {"dim 2 of a 2-D tensor", &g_split_fallback, F32, F32, {2, {2, 6}}, 2, 2,
       {1, 1}, kDense, 0, false},
      {"column-major [4,6]", &g_split_fallback, F32, F32, {2, {4, 6}}, 1, 2,
       {2, 4}, CheckLayout::kReversed, 0, false},
      {"a piece of the wrong shape", &g_split_fallback, F32, F32,
       {3, {1, 4, 24}}, 1, 4, {1, 1, 1, 1}, kDense, 1, false},
      {"first piece written over the input", &g_split_fallback, F32, F32,
       {3, {1, 4, 24}}, 1, 4, {1, 1, 1, 1}, kDense, 0, true},
  };
  for (const SplitCase& c : cases) {
    check_split(check, op, c);
  }
}

// remainder.Scalar_out. `self` starts with `values` (the rest random in
// [0, 1000]); the divisor is an int, double or bool Scalar.
enum class DivisorKind { kInt, kDouble, kBool };

struct RemainderCase {
  const char* what;
  uint64_t* expect;
  ScalarType self_type, out_type;
  CheckShape self, out;
  int64_t values[4];
  int value_count;
  DivisorKind kind;
  int64_t divisor;
  CheckLayout self_layout;
  bool in_place;
};

void check_remainder(FastpathCheck& check, CheckOp& op, const RemainderCase& c) {
  check.begin_case();
  const size_t self_bytes = check_bytes(c.self_type, c.self);
  const size_t written_bytes = c.in_place
      ? self_bytes : check_bytes(c.out_type, c.out);
  void* self_data = check.take(self_bytes);
  unsigned char* initial = check.take_written(written_bytes);
  if (self_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  for (size_t index = 0; index < c.self.numel(); ++index) {
    const int64_t value = static_cast<int>(index) < c.value_count
        ? c.values[index] : check_integer(0, 1000, false);
    if (c.self_type == ScalarType::Long) {
      static_cast<int64_t*>(self_data)[index] = value;
    } else {
      static_cast<int32_t*>(self_data)[index] = static_cast<int32_t>(value);
    }
  }
  if (c.in_place) {
    memcpy(initial, self_data, self_bytes);
  }
  const Scalar divisor = c.kind == DivisorKind::kDouble
      ? Scalar(static_cast<double>(c.divisor))
      : c.kind == DivisorKind::kBool ? Scalar(c.divisor != 0)
                                     : Scalar(c.divisor);
  check.compare(
      op, c.what, c.expect, initial, written_bytes,
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor self(c.self_type, c.self,
                         c.in_place ? written : self_data, c.self_layout);
        CheckTensor out(c.out_type, c.out, written);
        if (fast) {
          EValue self_value(self.tensor()), divisor_value(divisor);
          EValue out_value(out.tensor());
          EValue* stack[4] = {&self_value, &divisor_value, &out_value,
                              &out_value};
          shim_remainder_scalar_out(ctx, Span<EValue*>(stack, 4));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::remainder_Scalar_out(
              ctx, self.tensor(), divisor, out_tensor);
        }
      });
}

void check_remainder_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType I64 = ScalarType::Long;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  constexpr DivisorKind kInt = DivisorKind::kInt;
  const CheckShape scalar = {0, {}};
  const CheckShape eight = {1, {8}};
  const RemainderCase cases[] = {
      {"story slot 0-d 70 % 64", &g_remainder_non_negative, I64, I64, scalar,
       scalar, {70}, 1, kInt, 64, kDense, false},
      {"0-d 0 % 64", &g_remainder_non_negative, I64, I64, scalar, scalar, {0},
       1, kInt, 64, kDense, false},
      {"0-d 63 % 64", &g_remainder_non_negative, I64, I64, scalar, scalar,
       {63}, 1, kInt, 64, kDense, false},
      {"0-d 64 % 64", &g_remainder_non_negative, I64, I64, scalar, scalar,
       {64}, 1, kInt, 64, kDense, false},
      {"0-d INT64_MAX % 64", &g_remainder_non_negative, I64, I64, scalar,
       scalar, {INT64_MAX}, 1, kInt, 64, kDense, false},
      {"[8] % 7", &g_remainder_non_negative, I64, I64, eight, eight,
       {0, 6, 7, 8}, 4, kInt, 7, kDense, false},
      {"[8] % 1", &g_remainder_non_negative, I64, I64, eight, eight, {5}, 1,
       kInt, 1, kDense, false},
      {"[4] % INT64_MAX", &g_remainder_non_negative, I64, I64, {1, {4}},
       {1, {4}}, {INT64_MAX, INT64_MAX - 1, 0, 1}, 4, kInt, INT64_MAX, kDense,
       false},
      {"[6] % 10 in place", &g_remainder_non_negative, I64, I64, {1, {6}},
       {1, {6}}, {19, 20, 21}, 3, kInt, 10, kDense, true},
      {"0-d -3 % 64", &g_remainder_fallback, I64, I64, scalar, scalar, {-3},
       1, kInt, 64, kDense, false},
      {"0-d INT64_MIN % 64", &g_remainder_fallback, I64, I64, scalar, scalar,
       {INT64_MIN}, 1, kInt, 64, kDense, false},
      {"[8] with one negative % 64", &g_remainder_fallback, I64, I64, eight,
       eight, {5, 70, -70, 3}, 4, kInt, 64, kDense, false},
      {"0-d 70 % -64", &g_remainder_fallback, I64, I64, scalar, scalar, {70},
       1, kInt, -64, kDense, false},
      {"0-d 70 % 0", &g_remainder_fallback, I64, I64, scalar, scalar, {70}, 1,
       kInt, 0, kDense, false},
      {"0-d 70 % 64.0", &g_remainder_fallback, I64, I64, scalar, scalar, {70},
       1, DivisorKind::kDouble, 64, kDense, false},
      {"0-d 70 % true", &g_remainder_fallback, I64, I64, scalar, scalar, {70},
       1, DivisorKind::kBool, 1, kDense, false},
      {"int32 0-d 70 % 64", &g_remainder_fallback, ScalarType::Int,
       ScalarType::Int, scalar, scalar, {70}, 1, kInt, 64, kDense, false},
      {"int64 into fp32", &g_remainder_fallback, I64, ScalarType::Float,
       eight, eight, {70}, 1, kInt, 64, kDense, false},
      {"[4] into [5]", &g_remainder_fallback, I64, I64, {1, {4}}, {1, {5}},
       {70}, 1, kInt, 64, kDense, false},
      {"[4,3] column-major", &g_remainder_fallback, I64, I64, {2, {4, 3}},
       {2, {4, 3}}, {70}, 1, kInt, 64, CheckLayout::kReversed, false},
  };
  for (const RemainderCase& c : cases) {
    check_remainder(check, op, c);
  }
}

#if ESP32_GENERATED_FAST_QUANTIZE || ESP32_GENERATED_FAST_DEQUANTIZE || \
    ESP32_GENERATED_FAST_AVG_POOL2D
// ---- qdq_pool.h: per-tensor quantize and dequantize, avg_pool2d ----

// A float32 input for a quantize at `scale`: the IEEE specials (check_fill's
// first 16), arbitrary bit patterns, values across the code range and a little
// past it, and values within two ulps of a rounding tie (k + 1/2) * scale.
void qdq_quantize_fill(float* x, int count, double scale, int64_t zero_point,
                       int64_t quant_min, int64_t quant_max) {
  check_fill(ScalarType::Float, x, static_cast<size_t>(count));
  const float s = static_cast<float>(scale);
  const uint32_t span = static_cast<uint32_t>(quant_max - quant_min) + 5u;
  for (int index = 16; index < count; ++index) {
    const uint32_t r = check_random();
    const int64_t k = quant_min - zero_point - 2 +
        static_cast<int64_t>(check_random() % span);
    if ((r & 3u) == 0) {
      continue;  // check_fill's value
    } else if ((r & 3u) == 1) {
      x[index] = s * (static_cast<float>(k) +
                      static_cast<float>(check_random() & 0xffffu) * 0x1p-16f);
    } else {
      const float tie = (static_cast<float>(k) + 0.5f) * s;
      uint32_t bits;
      memcpy(&bits, &tie, sizeof(bits));
      bits += static_cast<uint32_t>(static_cast<int32_t>(check_random() % 5u) - 2);
      memcpy(&x[index], &bits, sizeof(bits));
    }
  }
}

#if ESP32_GENERATED_FAST_QUANTIZE
struct QuantizeCase {
  // In line, not a pointer: a static table's pointed-to names sit in the
  // file's shared string pool beside the kernel registration names, which
  // every image with these overrides keeps. In line, they go with the table,
  // which a build without the check drops (see run_fast_layer_norm_check).
  char what[64];
  bool fast;
  ScalarType in_type, dtype;
  int count;
  double scale;
  int64_t zero_point, quant_min, quant_max;
};

void check_quantize(FastpathCheck& check, CheckOp& op, const QuantizeCase& c) {
  check.begin_case();
  const CheckShape shape = {1, {c.count}};
  void* in_data = check.take(check_bytes(c.in_type, shape));
  unsigned char* initial = check.take_written(check_bytes(c.dtype, shape));
  if (in_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  if (c.in_type == ScalarType::Float) {
    qdq_quantize_fill(static_cast<float*>(in_data), c.count, c.scale,
                      c.zero_point, c.quant_min, c.quant_max);
  } else {
    check_fill(c.in_type, in_data, shape.numel());
  }
  check.compare(
      op, c.what, c.fast ? op.counters[0] : op.fallback, initial,
      check_bytes(c.dtype, shape),
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor input(c.in_type, shape, in_data);
        CheckTensor out(c.dtype, shape, written);
        if (fast) {
          EValue in_value(input.tensor()), scale_value(c.scale);
          EValue zero_value(c.zero_point), min_value(c.quant_min);
          EValue max_value(c.quant_max);
          EValue dtype_value(static_cast<int64_t>(c.dtype));
          EValue out_value(out.tensor());
          EValue* stack[8] = {&in_value,  &scale_value, &zero_value,
                              &min_value, &max_value,   &dtype_value,
                              &out_value, &out_value};
          qdq_pool::shim_quantize_per_tensor_out(ctx, Span<EValue*>(stack, 8));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::quantize_per_tensor_out(
              ctx, input.tensor(), c.scale, c.zero_point, c.quant_min,
              c.quant_max, c.dtype, out_tensor);
        }
      });
}

void check_quantize_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr ScalarType I8 = ScalarType::Char;
  constexpr ScalarType U8 = ScalarType::Byte;
  // A named static, names in line: a build without the check drops it (see
  // run_fast_layer_norm_check).
  static const QuantizeCase cases[] = {
      {"int8 [1031] at 0.0235, zp -3", true, F32, I8, 1031, 0.0235, -3, -128, 127},
      {"uint8 [515] at 0.0235, zp 128", true, F32, U8, 515, 0.0235, 128, 0, 255},
      {"int8 symmetric at 1/64", true, F32, I8, 257, 0.015625, 0, -127, 127},
      {"int8 zp at quant_min", true, F32, I8, 130, 0.1, -128, -128, 127},
      {"int8 zp at quant_max", true, F32, I8, 130, 0.1, 127, -128, 127},
      {"uint8 4-bit range", true, F32, U8, 130, 0.5, 8, 0, 15},
      {"int8 a one-code range", true, F32, I8, 66, 0.25, 7, 7, 7},
      {"int16 [258] at 2^-12", true, F32, ScalarType::Short, 258, 0x1p-12, 0,
       -32768, 32767},
      {"int16 at 0.0003, zp -5", true, F32, ScalarType::Short, 131, 0.0003, -5,
       -32767, 32767},
      {"uint16 at 0.001, zp 32768", true, F32, ScalarType::UInt16, 131, 0.001,
       32768, 0, 65535},
      {"bits16 at 3.0, zp 9", true, F32, ScalarType::Bits16, 67, 3.0, 9, 0,
       65535},
      {"int8 scale 0: an infinite reciprocal", true, F32, I8, 67, 0.0, 3, -128,
       127},
      {"int8 scale inf: a zero reciprocal", true, F32, I8, 67, INFINITY, 3,
       -128, 127},
      {"int8 scale NaN", true, F32, I8, 67, NAN, 3, -128, 127},
      {"int8 a negative scale", true, F32, I8, 67, -0.05, -1, -128, 127},
      {"int8 scale 1e-39, subnormal as float32", true, F32, I8, 67, 1e-39, 0,
       -128, 127},
      {"int8 scale 1e-38", true, F32, I8, 67, 1e-38, 0, -128, 127},
      {"int8 scale 3e38", true, F32, I8, 67, 3e38, 0, -128, 127},
      {"int8 scale 1e-300 (float32 0)", true, F32, I8, 67, 1e-300, 0, -128,
       127},
      {"int8 one element", true, F32, I8, 1, 0.0235, -3, -128, 127},
      {"int8 three elements", true, F32, I8, 3, 0.0235, -3, -128, 127},
      {"int8 four elements", true, F32, I8, 4, 0.0235, -3, -128, 127},
      {"int8 seventeen elements", true, F32, I8, 17, 0.0235, -3, -128, 127},
      // The bench models' own quantize shapes (their element counts).
      {"DS-CNN's input [1,1,49,10]", true, F32, I8, 490, 0.0472, 12, -128, 127},
      {"VWW's tail [1,256,1,1]", true, F32, I8, 256, 0.0117, -128, -128, 127},
      {"MobileNetV2-96's [1,64,6,6]", true, F32, I8, 2304, 0.0961, 4, -128, 127},
      {"ResNet-8's [1,64,8,8]", true, F32, I8, 4096, 0.0518, -7, -128, 127},
      {"ResNet-8's [1,32,16,16]", true, F32, I8, 8192, 0.0644, -11, -128, 127},
      {"int8 zp past quant_max", false, F32, I8, 67, 0.1, 128, -128, 127},
      {"uint8 zp below quant_min", false, F32, U8, 67, 0.1, -1, 0, 255},
      {"int8 quant_min above quant_max", false, F32, I8, 67, 0.1, 5, 10, 3},
      {"int32 codes", false, F32, ScalarType::Int, 67, 0.1, 0, INT32_MIN,
       INT32_MAX},
      {"fp16 input", false, ScalarType::Half, I8, 67, 0.1, 0, -128, 127},
      {"fp64 input", false, ScalarType::Double, I8, 67, 0.1, 0, -128, 127},
  };
  for (const QuantizeCase& c : cases) {
    check_quantize(check, op, c);
  }
  // Input and output over one buffer (a method may be handed an output that
  // overlaps its input): the codes written `out_offset` bytes into the
  // floats. Only the portable kernel's own order gives its bytes there.
  static const struct {
    char what[64];
    int count;
    size_t out_offset;
  } aliases[] = {
      {"codes at the floats' start", 67, 0},
      {"codes 4 bytes into the floats", 67, 4},
      {"codes 100 bytes into the floats", 67, 100},
  };
  for (const auto& a : aliases) {
    check.begin_case();
    const CheckShape shape = {1, {a.count}};
    const size_t in_bytes = check_bytes(ScalarType::Float, shape);
    const size_t bytes = std::max(in_bytes, a.out_offset + a.count);
    unsigned char* initial = check.take_written(bytes);
    if (initial == nullptr) {
      check.mismatch(op, a.what, "check arena exhausted");
      continue;
    }
    qdq_quantize_fill(reinterpret_cast<float*>(initial), a.count, 0.0235, -3,
                      -128, 127);
    check.compare(
        op, a.what, op.fallback, initial, bytes,
        [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
          CheckTensor input(ScalarType::Float, shape, written);
          CheckTensor out(ScalarType::Char, shape, written + a.out_offset);
          if (fast) {
            EValue in_value(input.tensor()), scale_value(0.0235);
            EValue zero_value(int64_t{-3}), min_value(int64_t{-128});
            EValue max_value(int64_t{127});
            EValue dtype_value(static_cast<int64_t>(ScalarType::Char));
            EValue out_value(out.tensor());
            EValue* stack[8] = {&in_value,  &scale_value, &zero_value,
                                &min_value, &max_value,   &dtype_value,
                                &out_value, &out_value};
            qdq_pool::shim_quantize_per_tensor_out(ctx, Span<EValue*>(stack, 8));
          } else {
            Tensor out_tensor = out.tensor();
            torch::executor::native::quantize_per_tensor_out(
                ctx, input.tensor(), 0.0235, -3, -128, 127, ScalarType::Char,
                out_tensor);
          }
        });
  }
}
#endif  // ESP32_GENERATED_FAST_QUANTIZE

#if ESP32_GENERATED_FAST_DEQUANTIZE
struct DequantizeCase {
  char what[64];  // in line, as QuantizeCase's
  bool fast;
  ScalarType dtype, out_type;
  int count;
  double scale;
  int64_t zero_point, quant_min, quant_max;
  int out_dtype;  // -1: None; else the ScalarType passed as out_dtype
};

void check_dequantize(FastpathCheck& check, CheckOp& op,
                      const DequantizeCase& c) {
  check.begin_case();
  const CheckShape shape = {1, {c.count}};
  void* in_data = check.take(check_bytes(c.dtype, shape));
  unsigned char* initial = check.take_written(check_bytes(c.out_type, shape));
  if (in_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  check_fill(c.dtype, in_data, shape.numel());  // every code, arbitrary bytes
  std::optional<ScalarType> out_dtype;
  if (c.out_dtype >= 0) {
    out_dtype = static_cast<ScalarType>(c.out_dtype);
  }
  check.compare(
      op, c.what, c.fast ? op.counters[0] : op.fallback, initial,
      check_bytes(c.out_type, shape),
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor input(c.dtype, shape, in_data);
        CheckTensor out(c.out_type, shape, written);
        if (fast) {
          EValue in_value(input.tensor()), scale_value(c.scale);
          EValue zero_value(c.zero_point), min_value(c.quant_min);
          EValue max_value(c.quant_max);
          EValue dtype_value(static_cast<int64_t>(c.dtype));
          EValue out_dtype_value;
          if (c.out_dtype >= 0) {
            out_dtype_value = EValue(static_cast<int64_t>(c.out_dtype));
          }
          EValue out_value(out.tensor());
          EValue* stack[9] = {&in_value,    &scale_value,     &zero_value,
                              &min_value,   &max_value,       &dtype_value,
                              &out_dtype_value, &out_value,   &out_value};
          qdq_pool::shim_dequantize_per_tensor_out(ctx, Span<EValue*>(stack, 9));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::dequantize_per_tensor_out(
              ctx, input.tensor(), c.scale, c.zero_point, c.quant_min,
              c.quant_max, c.dtype, out_dtype, out_tensor);
        }
      });
}

void check_dequantize_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr ScalarType I8 = ScalarType::Char;
  constexpr ScalarType U8 = ScalarType::Byte;
  constexpr int kNone = -1;
  constexpr int kF32 = static_cast<int>(ScalarType::Float);
  static const DequantizeCase cases[] = {
      {"int8 [1031] at 0.0235, zp -3", true, I8, F32, 1031, 0.0235, -3, -128,
       127, kNone},
      {"uint8 [515] at 0.0235, zp 128", true, U8, F32, 515, 0.0235, 128, 0, 255,
       kNone},
      {"int8 [2051]: the 256-entry table", true, I8, F32, 2051, 0.0235, -3, -128,
       127, kNone},
      {"uint8 [2048] table at scale NaN", true, U8, F32, 2048, NAN, 128, 0, 255,
       kNone},
      {"int8 out_dtype fp32", true, I8, F32, 130, 0.1, 0, -128, 127, kF32},
      {"int8 zp at quant_min", true, I8, F32, 130, 0.1, -128, -128, 127, kNone},
      {"int16 [258] at 2^-12", true, ScalarType::Short, F32, 258, 0x1p-12, 0,
       -32768, 32767, kNone},
      {"uint16 zp 32768", true, ScalarType::UInt16, F32, 131, 0.001, 32768, 0,
       65535, kNone},
      {"bits16 zp 9", true, ScalarType::Bits16, F32, 67, 3.0, 9, 0, 65535,
       kNone},
      {"int8 scale 0", true, I8, F32, 67, 0.0, 3, -128, 127, kNone},
      {"int8 scale inf", true, I8, F32, 67, INFINITY, 3, -128, 127, kNone},
      {"int8 scale NaN", true, I8, F32, 67, NAN, 3, -128, 127, kNone},
      {"int8 a negative scale", true, I8, F32, 67, -0.05, -1, -128, 127, kNone},
      {"int8 scale 1e-39, subnormal as float32", true, I8, F32, 67, 1e-39, 0,
       -128, 127, kNone},
      {"int8 scale 3e38: products overflow", true, I8, F32, 67, 3e38, 0, -128,
       127, kNone},
      {"int8 one element", true, I8, F32, 1, 0.0235, -3, -128, 127, kNone},
      {"int8 three elements", true, I8, F32, 3, 0.0235, -3, -128, 127, kNone},
      {"int8 seventeen elements", true, I8, F32, 17, 0.0235, -3, -128, 127,
       kNone},
      // The bench models' own dequantize shapes.
      {"DS-CNN's input [1,1,49,10]", true, I8, F32, 490, 0.0472, 12, -128,
       127, kNone},
      {"MobileNetV2-96's [1,64,6,6]: the table", true, I8, F32, 2304, 0.0961,
       4, -128, 127, kNone},
      {"ResNet-8's [1,64,8,8]: the table", true, I8, F32, 4096, 0.0518, -7,
       -128, 127, kNone},
      {"int8 zp past quant_max", false, I8, F32, 67, 0.1, 128, -128, 127,
       kNone},
      {"int32 codes", false, ScalarType::Int, F32, 67, 0.1, 0, INT32_MIN,
       INT32_MAX, kNone},
      {"int8 into fp16", false, I8, ScalarType::Half, 67, 0.1, 0, -128, 127,
       kNone},
      {"int8 into fp64", false, I8, ScalarType::Double, 67, 0.1, 0, -128, 127,
       kNone},
  };
  for (const DequantizeCase& c : cases) {
    check_dequantize(check, op, c);
  }
  // Input and output over one buffer: the codes `in_offset` bytes into the
  // floats the call writes. The table in the output's last 256 floats would
  // overwrite unread codes (the 2,048 sevens at byte 6,144), the four-lane
  // loop reads codes the portable kernel reads after its own writes.
  static const struct {
    char what[64];
    ScalarType dtype;
    int count;
    size_t in_offset;
    bool sevens;  // every code 7 (else arbitrary bytes)
  } aliases[] = {
      {"2,048 uint8 sevens at byte 6,144 of the output", ScalarType::Byte, 2048,
       6144, true},
      {"int8 codes at the output's start", ScalarType::Char, 67, 0, false},
      {"int8 codes 100 bytes into the output", ScalarType::Char, 67, 100,
       false},
  };
  for (const auto& a : aliases) {
    check.begin_case();
    const CheckShape shape = {1, {a.count}};
    const size_t out_bytes = check_bytes(ScalarType::Float, shape);
    const size_t bytes = std::max(out_bytes, a.in_offset + a.count);
    unsigned char* initial = check.take_written(bytes);
    if (initial == nullptr) {
      check.mismatch(op, a.what, "check arena exhausted");
      continue;
    }
    if (a.sevens) {
      memset(initial + a.in_offset, 7, (size_t)a.count);
    } else {
      check_fill(a.dtype, initial + a.in_offset, shape.numel());
    }
    const int64_t lo = a.dtype == ScalarType::Byte ? 0 : -128;
    const int64_t hi = a.dtype == ScalarType::Byte ? 255 : 127;
    const int64_t zp = a.dtype == ScalarType::Byte ? 0 : -3;
    const double scale = a.sevens ? 1.0 : 0.0235;
    check.compare(
        op, a.what, op.fallback, initial, bytes,
        [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
          CheckTensor input(a.dtype, shape, written + a.in_offset);
          CheckTensor out(ScalarType::Float, shape, written);
          if (fast) {
            EValue in_value(input.tensor()), scale_value(scale);
            EValue zero_value(zp), min_value(lo), max_value(hi);
            EValue dtype_value(static_cast<int64_t>(a.dtype));
            EValue out_dtype_value;
            EValue out_value(out.tensor());
            EValue* stack[9] = {&in_value,    &scale_value,     &zero_value,
                                &min_value,   &max_value,       &dtype_value,
                                &out_dtype_value, &out_value,   &out_value};
            qdq_pool::shim_dequantize_per_tensor_out(ctx,
                                                     Span<EValue*>(stack, 9));
          } else {
            Tensor out_tensor = out.tensor();
            torch::executor::native::dequantize_per_tensor_out(
                ctx, input.tensor(), scale, zp, lo, hi, a.dtype, std::nullopt,
                out_tensor);
          }
        });
  }
}
#endif  // ESP32_GENERATED_FAST_DEQUANTIZE

#if ESP32_GENERATED_FAST_AVG_POOL2D
// A pool input: finite values of every magnitude (signed zeros, subnormals,
// large ones whose sums round), plus, by `special`, one +inf, or a NaN, or a
// +inf and a -inf in the same plane (their sum is NaN).
enum class PoolSpecial { kNone, kInf, kNaN, kInfMinusInf };

void pool_fill(float* x, size_t count, PoolSpecial special) {
  for (size_t index = 0; index < count; ++index) {
    const uint32_t r = check_random();
    switch (r & 7u) {
      case 0:
        x[index] = (r & 8u) ? -0.0f : 0.0f;
        break;
      case 1: {  // a subnormal
        const uint32_t bits = (r & 0x80000000u) | (check_random() & 0x007fffffu);
        memcpy(&x[index], &bits, sizeof(bits));
        break;
      }
      case 2:  // large: sums round
        x[index] = static_cast<float>(static_cast<int32_t>(check_random())) * 0x1p+40f;
        break;
      default:
        x[index] = static_cast<float>(static_cast<int32_t>(check_random())) *
            0x1p-31f * static_cast<float>(1u << (r >> 28));
        break;
    }
  }
  if (special == PoolSpecial::kInf) {
    x[count / 2] = INFINITY;
  } else if (special == PoolSpecial::kNaN) {
    const uint32_t nan = 0x7fc12345u;
    memcpy(&x[count / 3], &nan, sizeof(nan));
  } else if (special == PoolSpecial::kInfMinusInf) {
    x[0] = INFINITY;
    x[1] = -INFINITY;
  }
}

constexpr int kPoolNoDivisor = INT32_MIN;  // divisor_override None

struct PoolCase {
  char what[64];  // in line, as QuantizeCase's
  int expect;  // 0 global, 1 windowed, 2 the portable kernel
  ScalarType type;
  CheckShape in, out;  // out: its sizes before the call (resized by both)
  int kernel_count, kernel[2];
  int stride_count, stride[2];
  int padding_count, padding[2];
  bool ceil_mode, count_include_pad;
  int divisor;  // or kPoolNoDivisor
  CheckLayout layout;
  PoolSpecial special;
};

void check_pool(FastpathCheck& check, CheckOp& op, const PoolCase& c) {
  check.begin_case();
  void* in_data = check.take(check_bytes(c.type, c.in));
  unsigned char* initial = check.take_written(check_bytes(c.type, c.out));
  if (in_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  if (c.type == ScalarType::Float) {
    pool_fill(static_cast<float*>(in_data), c.in.numel(), c.special);
  } else {
    check_fill(c.type, in_data, c.in.numel());
  }
  int64_t kernel[2] = {c.kernel[0], c.kernel[1]};
  int64_t stride[2] = {c.stride[0], c.stride[1]};
  int64_t padding[2] = {c.padding[0], c.padding[1]};
  check.compare(
      op, c.what, op.counters[c.expect], initial, check_bytes(c.type, c.out),
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor input(c.type, c.in, in_data, c.layout);
        CheckTensor out(c.type, c.out, written, c.layout);
        const std::optional<int64_t> divisor =
            c.divisor == kPoolNoDivisor
            ? std::nullopt
            : std::optional<int64_t>(static_cast<int64_t>(c.divisor));
        if (fast) {
          EValue kernel_values[2] = {EValue(kernel[0]), EValue(kernel[1])};
          EValue stride_values[2] = {EValue(stride[0]), EValue(stride[1])};
          EValue padding_values[2] = {EValue(padding[0]), EValue(padding[1])};
          EValue* kernel_boxed[2] = {&kernel_values[0], &kernel_values[1]};
          EValue* stride_boxed[2] = {&stride_values[0], &stride_values[1]};
          EValue* padding_boxed[2] = {&padding_values[0], &padding_values[1]};
          int64_t kernel_unboxed[2], stride_unboxed[2], padding_unboxed[2];
          executorch::runtime::BoxedEvalueList<int64_t> kernel_list(
              kernel_boxed, kernel_unboxed, c.kernel_count);
          executorch::runtime::BoxedEvalueList<int64_t> stride_list(
              stride_boxed, stride_unboxed, c.stride_count);
          executorch::runtime::BoxedEvalueList<int64_t> padding_list(
              padding_boxed, padding_unboxed, c.padding_count);
          EValue in_value(input.tensor()), kernel_value(&kernel_list);
          EValue stride_value(&stride_list), padding_value(&padding_list);
          EValue ceil_value(c.ceil_mode), pad_value(c.count_include_pad);
          EValue divisor_value;
          if (divisor.has_value()) {
            divisor_value = EValue(divisor.value());
          }
          EValue out_value(out.tensor());
          EValue* stack[9] = {&in_value,   &kernel_value, &stride_value,
                              &padding_value, &ceil_value, &pad_value,
                              &divisor_value, &out_value,  &out_value};
          qdq_pool::shim_avg_pool2d_out(ctx, Span<EValue*>(stack, 9));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::avg_pool2d_out(
              ctx, input.tensor(),
              IntArrayRef(kernel, static_cast<size_t>(c.kernel_count)),
              IntArrayRef(stride, static_cast<size_t>(c.stride_count)),
              IntArrayRef(padding, static_cast<size_t>(c.padding_count)),
              c.ceil_mode, c.count_include_pad, divisor, out_tensor);
        }
      });
}

void check_pool_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  constexpr PoolSpecial kNo = PoolSpecial::kNone;
  constexpr int G = 0, W = 1, P = 2;
  static const PoolCase cases[] = {
      {"DS-CNN's global pool, kernel 25x5", G, F32, {4, {1, 6, 25, 5}},
       {4, {1, 6, 1, 1}}, 2, {25, 5}, 2, {25, 5}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"ResNet-8's global pool, kernel 8", G, F32, {4, {1, 9, 8, 8}},
       {4, {1, 9, 1, 1}}, 2, {8, 8}, 2, {8, 8}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"global, seven planes (a tail of three)", G, F32, {4, {1, 7, 3, 3}},
       {4, {1, 7, 1, 1}}, 2, {3, 3}, 2, {3, 3}, 1, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"global, batch 2", G, F32, {4, {2, 5, 4, 4}}, {4, {2, 5, 1, 1}}, 1,
       {4, 4}, 0, {0, 0}, 1, {0, 0}, false, false, kPoolNoDivisor, kDense,
       kNo},
      {"global 3-D [5,4,4], kernel given once", G, F32, {3, {5, 4, 4}},
       {3, {5, 1, 1}}, 1, {4, 4}, 0, {0, 0}, 1, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"global, divisor_override 7", G, F32, {4, {1, 4, 3, 3}},
       {4, {1, 4, 1, 1}}, 2, {3, 3}, 2, {3, 3}, 2, {0, 0}, false, true, 7,
       kDense, kNo},
      {"global, one element a plane", G, F32, {4, {1, 6, 1, 1}},
       {4, {1, 6, 1, 1}}, 2, {1, 1}, 2, {1, 1}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      // The bench models' own global pools, whole.
      {"DS-CNN's [1,64,25,5]", G, F32, {4, {1, 64, 25, 5}}, {4, {1, 64, 1, 1}},
       2, {25, 5}, 2, {25, 5}, 2, {0, 0}, false, true, kPoolNoDivisor, kDense,
       kNo},
      {"VWW's [1,256,3,3]", G, F32, {4, {1, 256, 3, 3}}, {4, {1, 256, 1, 1}},
       2, {3, 3}, 2, {3, 3}, 2, {0, 0}, false, true, kPoolNoDivisor, kDense,
       kNo},
      {"ResNet-8's [1,64,8,8]", G, F32, {4, {1, 64, 8, 8}}, {4, {1, 64, 1, 1}},
       2, {8, 8}, 2, {8, 8}, 2, {0, 0}, false, true, kPoolNoDivisor, kDense,
       kNo},
      {"MobileNetV2-96's [1,1280,3,3]", G, F32, {4, {1, 1280, 3, 3}},
       {4, {1, 1280, 1, 1}}, 2, {3, 3}, 2, {3, 3}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"MobileNetV3-Small's SE pool [1,288,7,7]", G, F32, {4, {1, 288, 7, 7}},
       {4, {1, 288, 1, 1}}, 2, {7, 7}, 2, {7, 7}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"global with an infinity", G, F32, {4, {1, 5, 4, 4}}, {4, {1, 5, 1, 1}},
       2, {4, 4}, 2, {4, 4}, 2, {0, 0}, false, true, kPoolNoDivisor, kDense,
       PoolSpecial::kInf},
      // One window over the whole plane with its padding: the global path,
      // dividing by the padded kernel's count where the padding is counted
      // (25 here, not the plane's 16) and by the plane's where it is not.
      {"global, kernel 5 padding 1 on 4x4, padding counted (25)", G, F32,
       {4, {1, 6, 4, 4}}, {4, {1, 6, 1, 1}}, 1, {5, 5}, 0, {0, 0}, 1, {1, 1},
       false, true, kPoolNoDivisor, kDense, kNo},
      {"global, kernel 5 padding 1 on 4x4, padding not counted (16)", G, F32,
       {4, {1, 6, 4, 4}}, {4, {1, 6, 1, 1}}, 1, {5, 5}, 0, {0, 0}, 1, {1, 1},
       false, false, kPoolNoDivisor, kDense, kNo},
      {"global, kernel 5 padding 1 on 4x4, ceil_mode, counted", G, F32,
       {4, {1, 6, 4, 4}}, {4, {1, 6, 1, 1}}, 1, {5, 5}, 0, {0, 0}, 1, {1, 1},
       true, true, kPoolNoDivisor, kDense, kNo},
      {"global, kernel 5x6 padding 1x2 on 4x3, counted (30)", G, F32,
       {4, {1, 3, 4, 3}}, {4, {1, 3, 1, 1}}, 2, {5, 6}, 0, {0, 0}, 2, {1, 2},
       false, true, kPoolNoDivisor, kDense, kNo},
      {"3x3 stride 2 padding 1", W, F32, {4, {1, 3, 7, 7}}, {4, {1, 3, 4, 4}},
       2, {3, 3}, 2, {2, 2}, 2, {1, 1}, false, true, kPoolNoDivisor, kDense,
       kNo},
      {"3x3 stride 2 padding 1, pad not counted", W, F32, {4, {1, 3, 7, 7}},
       {4, {1, 3, 4, 4}}, 2, {3, 3}, 2, {2, 2}, 2, {1, 1}, false, false,
       kPoolNoDivisor, kDense, kNo},
      {"ceil_mode 3x3 stride 2 on 6x6", W, F32, {4, {1, 2, 6, 6}},
       {4, {1, 2, 3, 3}}, 2, {3, 3}, 2, {2, 2}, 2, {0, 0}, true, true,
       kPoolNoDivisor, kDense, kNo},
      {"ceil_mode with padding, pad not counted", W, F32, {4, {1, 2, 5, 6}},
       {4, {1, 2, 3, 4}}, 2, {3, 2}, 2, {2, 2}, 2, {1, 1}, true, false,
       kPoolNoDivisor, kDense, kNo},
      {"2x3 stride 1x2 padding 1x1", W, F32, {4, {1, 2, 5, 7}},
       {4, {1, 2, 6, 4}}, 2, {2, 3}, 2, {1, 2}, 2, {1, 1}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"windowed, divisor_override -3", W, F32, {4, {1, 2, 6, 6}},
       {4, {1, 2, 3, 3}}, 2, {2, 2}, 0, {0, 0}, 1, {0, 0}, false, true, -3,
       kDense, kNo},
      {"windowed 3-D", W, F32, {3, {3, 5, 5}}, {3, {3, 2, 3}}, 2, {2, 3}, 2,
       {2, 2}, 2, {0, 1}, false, true, kPoolNoDivisor, kDense, kNo},
      {"global-sized kernel with padding: not the whole plane", W, F32,
       {4, {1, 3, 3, 3}}, {4, {1, 3, 1, 1}}, 2, {3, 3}, 2, {3, 3}, 2, {1, 1},
       false, true, kPoolNoDivisor, kDense, kNo},
      {"global with a NaN: redone by the portable kernel", P, F32,
       {4, {1, 5, 4, 4}}, {4, {1, 5, 1, 1}}, 2, {4, 4}, 2, {4, 4}, 2, {0, 0},
       false, true, kPoolNoDivisor, kDense, PoolSpecial::kNaN},
      {"global with +inf and -inf: redone", P, F32, {4, {1, 5, 4, 4}},
       {4, {1, 5, 1, 1}}, 2, {4, 4}, 2, {4, 4}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, PoolSpecial::kInfMinusInf},
      {"windowed with a NaN: redone", P, F32, {4, {1, 3, 7, 7}},
       {4, {1, 3, 4, 4}}, 2, {3, 3}, 2, {2, 2}, 2, {1, 1}, false, true,
       kPoolNoDivisor, kDense, PoolSpecial::kNaN},
      {"fp16", P, ScalarType::Half, {4, {1, 4, 4, 4}}, {4, {1, 4, 1, 1}}, 2,
       {4, 4}, 2, {4, 4}, 2, {0, 0}, false, true, kPoolNoDivisor, kDense, kNo},
      {"channels-last", P, F32, {4, {1, 4, 4, 4}}, {4, {1, 4, 1, 1}}, 2,
       {4, 4}, 2, {4, 4}, 2, {0, 0}, false, true, kPoolNoDivisor,
       CheckLayout::kChannelsLast, kNo},
      {"padding past half the kernel: refused", P, F32, {4, {1, 2, 4, 4}},
       {4, {1, 2, 2, 2}}, 2, {2, 2}, 2, {2, 2}, 2, {2, 2}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"a kernel past the plane: refused", P, F32, {4, {1, 2, 3, 3}},
       {4, {1, 2, 1, 1}}, 2, {5, 5}, 2, {1, 1}, 2, {0, 0}, false, true,
       kPoolNoDivisor, kDense, kNo},
      {"divisor_override 0: refused", P, F32, {4, {1, 2, 3, 3}},
       {4, {1, 2, 1, 1}}, 2, {3, 3}, 2, {3, 3}, 2, {0, 0}, false, true, 0,
       kDense, kNo},
  };
  for (const PoolCase& c : cases) {
    check_pool(check, op, c);
  }
  // Input and output over one buffer, the output `out_offset` floats into the
  // input. The fast path writes in its own order (four planes at a time) and
  // would redo a NaN call on the input it had overwritten: [2,4,6,NaN]
  // pooled in place by [1,2] gives [3,5,NaN] in one portable pass, [4,NaN,NaN]
  // through a redo.
  static const struct {
    char what[64];
    CheckShape in, out;
    int kernel[2];
    size_t out_offset;  // floats
    bool nan_run;  // the input is [2,4,6,NaN] (else pool_fill's finite values)
  } aliases[] = {
      {"[2,4,6,NaN] pooled in place by [1,2]", {4, {1, 1, 1, 4}},
       {4, {1, 1, 1, 3}}, {1, 2}, 0, true},
      {"a global pool written over its input", {4, {1, 8, 2, 2}},
       {4, {1, 8, 1, 1}}, {2, 2}, 0, false},
      {"a global pool written 20 floats into its input", {4, {1, 8, 2, 2}},
       {4, {1, 8, 1, 1}}, {2, 2}, 20, false},
  };
  for (const auto& a : aliases) {
    check.begin_case();
    const size_t in_bytes = check_bytes(ScalarType::Float, a.in);
    const size_t bytes = std::max(
        in_bytes, (a.out_offset + a.out.numel()) * sizeof(float));
    unsigned char* initial = check.take_written(bytes);
    if (initial == nullptr) {
      check.mismatch(op, a.what, "check arena exhausted");
      continue;
    }
    float* x = reinterpret_cast<float*>(initial);
    if (a.nan_run) {
      const float values[4] = {2.0f, 4.0f, 6.0f, NAN};
      memcpy(x, values, sizeof(values));
    } else {
      pool_fill(x, a.in.numel(), PoolSpecial::kNone);
    }
    int64_t kernel[2] = {a.kernel[0], a.kernel[1]};
    int64_t stride[2] = {a.nan_run ? 1 : a.kernel[0], a.nan_run ? 1 : a.kernel[1]};
    int64_t padding[2] = {0, 0};
    check.compare(
        op, a.what, op.fallback, initial, bytes,
        [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
          CheckTensor input(ScalarType::Float, a.in, written);
          CheckTensor out(ScalarType::Float, a.out,
                          written + a.out_offset * sizeof(float));
          if (fast) {
            EValue kernel_values[2] = {EValue(kernel[0]), EValue(kernel[1])};
            EValue stride_values[2] = {EValue(stride[0]), EValue(stride[1])};
            EValue padding_values[2] = {EValue(padding[0]), EValue(padding[1])};
            EValue* kernel_boxed[2] = {&kernel_values[0], &kernel_values[1]};
            EValue* stride_boxed[2] = {&stride_values[0], &stride_values[1]};
            EValue* padding_boxed[2] = {&padding_values[0], &padding_values[1]};
            int64_t kernel_unboxed[2], stride_unboxed[2], padding_unboxed[2];
            executorch::runtime::BoxedEvalueList<int64_t> kernel_list(
                kernel_boxed, kernel_unboxed, 2);
            executorch::runtime::BoxedEvalueList<int64_t> stride_list(
                stride_boxed, stride_unboxed, 2);
            executorch::runtime::BoxedEvalueList<int64_t> padding_list(
                padding_boxed, padding_unboxed, 2);
            EValue in_value(input.tensor()), kernel_value(&kernel_list);
            EValue stride_value(&stride_list), padding_value(&padding_list);
            EValue ceil_value(false), pad_value(true), divisor_value;
            EValue out_value(out.tensor());
            EValue* stack[9] = {&in_value,   &kernel_value, &stride_value,
                                &padding_value, &ceil_value, &pad_value,
                                &divisor_value, &out_value,  &out_value};
            qdq_pool::shim_avg_pool2d_out(ctx, Span<EValue*>(stack, 9));
          } else {
            Tensor out_tensor = out.tensor();
            torch::executor::native::avg_pool2d_out(
                ctx, input.tensor(), IntArrayRef(kernel, 2),
                IntArrayRef(stride, 2), IntArrayRef(padding, 2), false, true,
                std::nullopt, out_tensor);
          }
        });
  }
}
#endif  // ESP32_GENERATED_FAST_AVG_POOL2D
#endif  // any of the three

#if ESP32_GENERATED_EW_MUL
// Finite values of every size, signed zeros and subnormals among them: the
// element-wise loops' operands where a case wants no NaN on that side.
void ew_finite_fill(float* x, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    const uint32_t r = check_random();
    uint32_t bits;
    switch (r & 7u) {
      case 0:
        bits = r & 0x80000000u;  // a signed zero
        break;
      case 1:
        bits = (r & 0x80000000u) | (check_random() & 0x007fffffu);
        break;
      default:  // exponents 2^-27 .. 2^28
        bits = (r & 0x80000000u) | ((100u + (r >> 3) % 56u) << 23) |
            (check_random() & 0x007fffffu);
        break;
    }
    memcpy(&x[index], &bits, sizeof(bits));
  }
}
#endif

#if ESP32_GENERATED_EW_MUL
// mul.out's channel-scale loop (elementwise.h) through the wrapper a firmware
// built with it registers, against the portable kernel. The streaming
// operand starts with every special value (NaNs too: one NaN operand gives
// the same bits in either order); the scales are every special value but
// NaN, then finite ones. A case may put one NaN among the scales, which sends
// the call to the shim's broadcast path; its streaming operand is then
// finite. That path is the shim's general one, and keeps an operand order of
// its own: on the chip it need not keep the portable kernel's payload when
// both factors are NaNs (under QEMU run_parity sees a byte of this case's NaN
// channel differ, where check_fill's arbitrary bit patterns include NaNs;
// the host, which keeps the first operand's, agrees), and no program here
// makes such a pair.
struct MulChannelCase {
  char what[64];  // in line, as PoolCase's
  int expect;  // the op's counters: 0 the loop, 1 broadcast, 2 portable, 3 scalar
  CheckShape left, right, out;
  bool scale_left;      // which operand holds the scales
  CheckLayout layout;   // the streaming operand's
  int nan_scale;        // a scale set to a NaN, or -1
  int alias;  // 0 apart; 1 out is the streaming operand; 2 the scales are in
              // out; 3 out starts one element past the streaming operand
  CheckLayout out_layout;    // out's
  CheckLayout small_layout;  // the scales'
};

void check_mul_channels(FastpathCheck& check, CheckOp& op,
                        const MulChannelCase& c) {
  check.begin_case();
  const CheckShape& big_shape = c.scale_left ? c.right : c.left;
  const CheckShape& small_shape = c.scale_left ? c.left : c.right;
  const size_t big_bytes = check_bytes(ScalarType::Float, big_shape);
  const size_t small_bytes = check_bytes(ScalarType::Float, small_shape);
  const size_t out_bytes = check_bytes(ScalarType::Float, c.out);
  // alias 3: the streaming operand at the written bytes' start, out one
  // element further, so the two overlap in part.
  const size_t shift = c.alias == 3 ? sizeof(float) : 0;
  void* big_data = check.take(big_bytes);
  void* small_data = check.take(small_bytes);
  unsigned char* initial = check.take_written(out_bytes + shift);
  if (big_data == nullptr || small_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  if (c.nan_scale >= 0) {
    ew_finite_fill(static_cast<float*>(big_data), big_shape.numel());
  } else {
    check_fill(ScalarType::Float, big_data, big_shape.numel());
  }
  float* scales = static_cast<float*>(small_data);
  static const uint32_t kScaleSpecial[11] = {
      0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x00000001u,
      0x807fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu, 0x3f800000u,
      0xbf800000u};
  ew_finite_fill(scales, small_shape.numel());
  for (size_t index = 0; index < small_shape.numel() && index < 11; ++index) {
    memcpy(&scales[index], &kScaleSpecial[index], sizeof(uint32_t));
  }
  if (c.nan_scale >= 0) {
    const uint32_t nan = 0x7fc12345u;
    memcpy(&scales[c.nan_scale], &nan, sizeof(nan));
  }
  if (c.alias == 1 || c.alias == 3) {
    memcpy(initial, big_data, big_bytes);
  } else if (c.alias == 2) {
    memcpy(initial, small_data, small_bytes);
  }
  check.compare(
      op, c.what, op.counters[c.expect], initial, out_bytes + shift,
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor big(ScalarType::Float, big_shape,
                        c.alias == 1 || c.alias == 3 ? written : big_data,
                        c.layout);
        CheckTensor small(ScalarType::Float, small_shape,
                          c.alias == 2 ? written : small_data, c.small_layout);
        CheckTensor out(ScalarType::Float, c.out, written + shift,
                        c.out_layout);
        Tensor left = c.scale_left ? small.tensor() : big.tensor();
        Tensor right = c.scale_left ? big.tensor() : small.tensor();
        if (fast) {
          EValue left_value(left), right_value(right), out_value(out.tensor());
          EValue* stack[4] = {&left_value, &right_value, &out_value, &out_value};
          shim_mul_out_ew(ctx, Span<EValue*>(stack, 4));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::mul_out(ctx, left, right, out_tensor);
        }
      });
}

void check_mul_channels_cases(FastpathCheck& check, CheckOp& op) {
  constexpr CheckLayout kDense = CheckLayout::kDense;
  constexpr int L = 0, B = 1, P = 2, S = 3;
  static const MulChannelCase cases[] = {
      {"MobileNetV3 SE scale [1,16,1,1] x [1,16,7,7]", L, {4, {1, 16, 1, 1}},
       {4, {1, 16, 7, 7}}, {4, {1, 16, 7, 7}}, true, kDense, -1, 0},
      {"[1,6,5,5] x [1,6,1,1], the scales right", L, {4, {1, 6, 5, 5}},
       {4, {1, 6, 1, 1}}, {4, {1, 6, 5, 5}}, false, kDense, -1, 0},
      {"[1,24,1,1] x [1,24,3,5] in place", L, {4, {1, 24, 1, 1}},
       {4, {1, 24, 3, 5}}, {4, {1, 24, 3, 5}}, true, kDense, -1, 1},
      {"[2,3,1,1] x [2,3,4,5]", L, {4, {2, 3, 1, 1}}, {4, {2, 3, 4, 5}},
       {4, {2, 3, 4, 5}}, true, kDense, -1, 0},
      {"[5,7] x [5,1]", L, {2, {5, 7}}, {2, {5, 1}}, {2, {5, 7}}, false,
       kDense, -1, 0},
      {"[1,5,1,1] x [1,5,3,3], planes of 9", L, {4, {1, 5, 1, 1}},
       {4, {1, 5, 3, 3}}, {4, {1, 5, 3, 3}}, true, kDense, -1, 0},
      {"[1,4,1,1] x [1,4,2,2], planes of 4", L, {4, {1, 4, 1, 1}},
       {4, {1, 4, 2, 2}}, {4, {1, 4, 2, 2}}, true, kDense, -1, 0},
      {"[6,1,1] x [1,6,5,5], the scales of lower rank", L, {3, {6, 1, 1}},
       {4, {1, 6, 5, 5}}, {4, {1, 6, 5, 5}}, true, kDense, -1, 0},
      {"[1,6,5,1] x [1,6,1,1]", L, {4, {1, 6, 5, 1}}, {4, {1, 6, 1, 1}},
       {4, {1, 6, 5, 1}}, false, kDense, -1, 0},
      {"[1,16,1,1] x [1,16,7,7], a NaN scale", B, {4, {1, 16, 1, 1}},
       {4, {1, 16, 7, 7}}, {4, {1, 16, 7, 7}}, true, kDense, 9, 0},
      {"[3,1,1] x [2,3,4,4], a batch the scales repeat over", B,
       {3, {3, 1, 1}}, {4, {2, 3, 4, 4}}, {4, {2, 3, 4, 4}}, true, kDense, -1,
       0},
      {"[1,6,1,1] x [1,1,5,5], both broadcast", B, {4, {1, 6, 1, 1}},
       {4, {1, 1, 5, 5}}, {4, {1, 6, 5, 5}}, true, kDense, -1, 0},
      {"[1,8,1,1] x channels-last [1,8,4,4]", P, {4, {1, 8, 1, 1}},
       {4, {1, 8, 4, 4}}, {4, {1, 8, 4, 4}}, true, CheckLayout::kChannelsLast,
       -1, 0, CheckLayout::kChannelsLast},
      // Each of the loop's refusals alone: the streaming
      // operand channels-last, out channels-last, the scales channels-last,
      // and out partly over the streaming operand.
      {"[1,8,1,1] x channels-last [1,8,4,4] into a dense out", P,
       {4, {1, 8, 1, 1}}, {4, {1, 8, 4, 4}}, {4, {1, 8, 4, 4}}, true,
       CheckLayout::kChannelsLast, -1, 0},
      {"[1,8,1,1] x [1,8,4,4] into a channels-last out", P, {4, {1, 8, 1, 1}},
       {4, {1, 8, 4, 4}}, {4, {1, 8, 4, 4}}, true, kDense, -1, 0,
       CheckLayout::kChannelsLast},
      {"channels-last scales [1,8,1,1] x [1,8,4,4]", P, {4, {1, 8, 1, 1}},
       {4, {1, 8, 4, 4}}, {4, {1, 8, 4, 4}}, true, kDense, -1, 0, kDense,
       CheckLayout::kChannelsLast},
      {"[1,6,1,1] x [1,6,5,5], out one element past it", P, {4, {1, 6, 1, 1}},
       {4, {1, 6, 5, 5}}, {4, {1, 6, 5, 5}}, true, kDense, -1, 3},
      {"[1,6,1,1] x [1,6,5,5], the scales in the output", P,
       {4, {1, 6, 1, 1}}, {4, {1, 6, 5, 5}}, {4, {1, 6, 5, 5}}, true, kDense,
       -1, 2},
      {"[1,1,1,1] x [1,1,5,5], one scale", S, {4, {1, 1, 1, 1}},
       {4, {1, 1, 5, 5}}, {4, {1, 1, 5, 5}}, true, kDense, -1, 0},
  };
  for (const MulChannelCase& c : cases) {
    check_mul_channels(check, op, c);
  }
}
#endif  // ESP32_GENERATED_EW_MUL


#if ESP32_GENERATED_EW_CONSTANT_PAD
}  // namespace
}  // namespace espressif
namespace torch {
namespace executor {
namespace native {
Tensor& constant_pad_nd_out(KernelRuntimeContext&, const Tensor&, IntArrayRef,
                            const Scalar&, Tensor&);
}  // namespace native
}  // namespace executor
}  // namespace torch
namespace espressif {
namespace {

// constant_pad_nd.out's row copy (elementwise.h) against the portable
// kernel: every byte of the output, the failure state, the path. Inputs are
// arbitrary bytes (fp32: every special value first); the value is given as
// fp32 bits (fp32) or an integer (int8), or as a double where `value_double`.
struct PadCase {
  char what[64];  // in line, as PoolCase's
  int expect;     // the op's counters: 0 the row copy, 1 portable
  ScalarType type;
  CheckShape in, out;
  int pad_count, pad[4];
  double value;
  bool value_double;
  CheckLayout layout;      // the input's
  CheckLayout out_layout;  // out's
  bool in_over_out;        // the input's bytes start at out's
};

void check_pad(FastpathCheck& check, CheckOp& op, const PadCase& c) {
  check.begin_case();
  const size_t in_bytes = check_bytes(c.type, c.in);
  const size_t written_bytes = c.in_over_out
      ? std::max(in_bytes, check_bytes(c.type, c.out))
      : check_bytes(c.type, c.out);
  void* in_data = check.take(in_bytes);
  unsigned char* initial = check.take_written(written_bytes);
  if (in_data == nullptr || initial == nullptr) {
    check.mismatch(op, c.what, "check arena exhausted");
    return;
  }
  check_fill(c.type, in_data, c.in.numel());
  if (c.in_over_out) {
    memcpy(initial, in_data, in_bytes);
  }
  int64_t pad[4] = {c.pad[0], c.pad[1], c.pad[2], c.pad[3]};
  check.compare(
      op, c.what, op.counters[c.expect], initial, written_bytes,
      [&](bool fast, unsigned char* written, KernelRuntimeContext& ctx) {
        CheckTensor input(c.type, c.in, c.in_over_out ? written : in_data,
                          c.layout);
        CheckTensor out(c.type, c.out, written, c.out_layout);
        const Scalar value =
            c.value_double ? Scalar(c.value)
                           : Scalar(static_cast<int64_t>(c.value));
        if (fast) {
          EValue pad_values[4] = {EValue(pad[0]), EValue(pad[1]),
                                  EValue(pad[2]), EValue(pad[3])};
          EValue* pad_boxed[4] = {&pad_values[0], &pad_values[1],
                                  &pad_values[2], &pad_values[3]};
          int64_t pad_unboxed[4];
          executorch::runtime::BoxedEvalueList<int64_t> pad_list(
              pad_boxed, pad_unboxed, c.pad_count);
          EValue in_value(input.tensor()), pad_value(&pad_list);
          EValue fill(value), out_value(out.tensor());
          EValue* stack[5] = {&in_value, &pad_value, &fill, &out_value,
                              &out_value};
          elementwise::shim_constant_pad_nd_out(ctx, Span<EValue*>(stack, 5));
        } else {
          Tensor out_tensor = out.tensor();
          torch::executor::native::constant_pad_nd_out(
              ctx, input.tensor(),
              IntArrayRef(pad, static_cast<size_t>(c.pad_count)), value,
              out_tensor);
        }
      });
}

void check_pad_cases(FastpathCheck& check, CheckOp& op) {
  constexpr ScalarType F32 = ScalarType::Float;
  constexpr ScalarType I8 = ScalarType::Char;
  constexpr CheckLayout kDense = CheckLayout::kDense;
  constexpr int R = 0, P = 1;
  static const PadCase cases[] = {
      {"MobileNet-v1 int8 [1,4,6,6] (0,1,0,1) -128", R, I8, {4, {1, 4, 6, 6}},
       {4, {1, 4, 7, 7}}, 4, {0, 1, 0, 1}, -128, false, kDense},
      {"MobileNet-v1 fp32 [1,3,9,9] (0,1,0,1) 0.0", R, F32,
       {4, {1, 3, 9, 9}}, {4, {1, 3, 10, 10}}, 4, {0, 1, 0, 1}, 0.0, true,
       kDense},
      {"DS-CNN fp32 [1,1,7,10] (1,1,4,5)", R, F32, {4, {1, 1, 7, 10}},
       {4, {1, 1, 16, 12}}, 4, {1, 1, 4, 5}, 0.0, true, kDense},
      {"int8 [2,3,5,7] (2,3,1,0) 5", R, I8, {4, {2, 3, 5, 7}},
       {4, {2, 3, 6, 12}}, 4, {2, 3, 1, 0}, 5, false, kDense},
      {"fp32 [2,5,7], the last axis only (2,3)", R, F32, {3, {2, 5, 7}},
       {3, {2, 5, 12}}, 2, {2, 3, 0, 0}, 0.0, true, kDense},
      {"fp32 [9], rank 1 (1,2) -0.0", R, F32, {1, {9}}, {1, {12}}, 2,
       {1, 2, 0, 0}, -0.0, true, kDense},
      {"fp32 [1,2,3,3] (1,1,1,1) NaN", R, F32, {4, {1, 2, 3, 3}},
       {4, {1, 2, 5, 5}}, 4, {1, 1, 1, 1}, NAN, true, kDense},
      {"int8 [1,2,4,4] (0,0,0,0)", R, I8, {4, {1, 2, 4, 4}},
       {4, {1, 2, 4, 4}}, 4, {0, 0, 0, 0}, 300, false, kDense},
      {"fp32 [1,2,4,4] (0,1,0,1) 1e300 (no fp32)", P, F32,
       {4, {1, 2, 4, 4}}, {4, {1, 2, 5, 5}}, 4, {0, 1, 0, 1}, 1e300, true,
       kDense},
      {"int8 [1,2,4,4] (0,1,0,1) 300 (no int8)", P, I8, {4, {1, 2, 4, 4}},
       {4, {1, 2, 5, 5}}, 4, {0, 1, 0, 1}, 300, false, kDense},
      {"fp32 [1,2,5,5] (-1,1,0,1), a crop", P, F32, {4, {1, 2, 5, 5}},
       {4, {1, 2, 6, 5}}, 4, {-1, 1, 0, 1}, 0.0, true, kDense},
      {"fp32 channels-last [1,4,3,3] (0,1,0,1)", P, F32, {4, {1, 4, 3, 3}},
       {4, {1, 4, 4, 4}}, 4, {0, 1, 0, 1}, 0.0, true,
       CheckLayout::kChannelsLast, CheckLayout::kChannelsLast},
      {"int16 [1,2,4,4] (0,1,0,1)", P, ScalarType::Short, {4, {1, 2, 4, 4}},
       {4, {1, 2, 5, 5}}, 4, {0, 1, 0, 1}, 7, false, kDense},
      // A singleton axis's stride counts: the portable kernel copies
      // stride x size elements of it, so a stride-0 input [1] copies none
      // ([7,7,9] portable, [7,5,7] by a row copy).
      {"int8 [1] stride 0 (1,1) 7", P, I8, {1, {1}}, {1, {3}}, 2,
       {1, 1, 0, 0}, 7, false, CheckLayout::kSingletonStride0},
      {"fp32 [1,2,3,3] (0,1,0,1) into an out of stride-0 batch", P, F32,
       {4, {1, 2, 3, 3}}, {4, {1, 2, 4, 4}}, 4, {0, 1, 0, 1}, 0.0, true,
       kDense, CheckLayout::kSingletonStride0},
      {"fp32 [1,2,3,3] stride-0 batch (0,1,0,1)", P, F32, {4, {1, 2, 3, 3}},
       {4, {1, 2, 4, 4}}, 4, {0, 1, 0, 1}, 0.0, true,
       CheckLayout::kSingletonStride0},
      // Each of the row copy's other refusals: an input
      // over its output and a static out of the wrong shape (the portable
      // kernel fails both), an empty input.
      {"int8 [1,2,4,4] (0,1,0,1) over its output", P, I8, {4, {1, 2, 4, 4}},
       {4, {1, 2, 5, 5}}, 4, {0, 1, 0, 1}, -128, false, kDense, kDense, true},
      {"fp32 [1,2,4,4] (0,1,0,1) into an out one row short", P, F32,
       {4, {1, 2, 4, 4}}, {4, {1, 2, 4, 5}}, 4, {0, 1, 0, 1}, 0.0, true},
      {"int8 [1,2,0,4] (0,1,0,1), an empty input", P, I8, {4, {1, 2, 0, 4}},
       {4, {1, 2, 1, 5}}, 4, {0, 1, 0, 1}, -128, false},
  };
  for (const PadCase& c : cases) {
    check_pad(check, op, c);
  }
}
#endif  // ESP32_GENERATED_EW_CONSTANT_PAD

}  // namespace

void run_portable_fastpath_check() {
  CheckOp mul = {"mul.out",
                 {&g_mul_same, &g_mul_broadcast, &g_mul_scalar, &g_mul_fallback},
                 &g_mul_fallback};
  CheckOp le = {"le.Tensor_out", {&g_le_scalar, &g_le_fallback}, &g_le_fallback};
  CheckOp index_put = {"index_put_", {&g_index_put_rows, &g_index_put_fallback},
                       &g_index_put_fallback};
  CheckOp split = {"split_with_sizes_copy.out",
                   {&g_split_chunks, &g_split_fallback}, &g_split_fallback};
  CheckOp remainder = {"remainder.Scalar_out",
                       {&g_remainder_non_negative, &g_remainder_fallback},
                       &g_remainder_fallback};
#if ESP32_GENERATED_FAST_QUANTIZE || ESP32_GENERATED_FAST_DEQUANTIZE || \
    ESP32_GENERATED_FAST_AVG_POOL2D
  qdq_pool::Counters& qdq = qdq_pool::counters();
#endif
#if ESP32_GENERATED_FAST_QUANTIZE
  CheckOp quantize = {"quantize_per_tensor.out",
                      {&qdq.quantize_fast, &qdq.quantize_fallback},
                      &qdq.quantize_fallback};
#endif
#if ESP32_GENERATED_FAST_DEQUANTIZE
  CheckOp dequantize = {"dequantize_per_tensor.out",
                        {&qdq.dequantize_fast, &qdq.dequantize_fallback},
                        &qdq.dequantize_fallback};
#endif
#if ESP32_GENERATED_FAST_AVG_POOL2D
  CheckOp pool = {"avg_pool2d.out",
                  {&qdq.pool_global, &qdq.pool_windowed, &qdq.pool_fallback},
                  &qdq.pool_fallback};
#endif
#if ESP32_GENERATED_EW_MUL
  elementwise::Counters& ew = elementwise::counters();
  CheckOp mul_channels = {"mul.out-channels",
                          {&ew.mul_channels, &g_mul_broadcast, &g_mul_fallback,
                           &g_mul_scalar},
                          &g_mul_fallback};
#endif
#if ESP32_GENERATED_EW_CONSTANT_PAD
  elementwise::Counters& ew_pad = elementwise::counters();
  CheckOp pad = {"constant_pad_nd.out", {&ew_pad.pad_rows, &ew_pad.pad_fallback},
                 &ew_pad.pad_fallback};
#endif
  CheckOp* ops[] = {&mul, &le, &index_put, &split, &remainder,
#if ESP32_GENERATED_EW_CONSTANT_PAD
                    &pad,
#endif
#if ESP32_GENERATED_EW_MUL
                    &mul_channels,
#endif
#if ESP32_GENERATED_FAST_QUANTIZE
                    &quantize,
#endif
#if ESP32_GENERATED_FAST_DEQUANTIZE
                    &dequantize,
#endif
#if ESP32_GENERATED_FAST_AVG_POOL2D
                    &pool,
#endif
  };
  for (CheckOp* op : ops) {
    for (int index = 0; index < kCheckCounters && op->counters[index]; ++index) {
      op->saved[index] = *op->counters[index];
    }
  }
  // 64 KB: the bench models' own tensors, among them MobileNetV2-96's
  // [1,1280,3,3] global pool, fit whole.
  FastpathCheck check(64 * 1024);
  if (!check.ok()) {
    printf("fastpath check: no memory for the check arena\n");
    return;
  }
  check_mul_cases(check, mul);
  check_le_cases(check, le);
  check_index_put_cases(check, index_put);
  check_split_cases(check, split);
  check_remainder_cases(check, remainder);
#if ESP32_GENERATED_FAST_QUANTIZE
  check_quantize_cases(check, quantize);
#endif
#if ESP32_GENERATED_FAST_DEQUANTIZE
  check_dequantize_cases(check, dequantize);
#endif
#if ESP32_GENERATED_FAST_AVG_POOL2D
  check_pool_cases(check, pool);
#endif
#if ESP32_GENERATED_EW_MUL || ESP32_GENERATED_EW_CONSTANT_PAD
  const elementwise::Counters ew_saved = elementwise::counters();
#endif
#if ESP32_GENERATED_EW_CONSTANT_PAD
  check_pad_cases(check, pad);
#endif
#if ESP32_GENERATED_EW_MUL
  check_mul_channels_cases(check, mul_channels);
#endif
#if ESP32_GENERATED_EW_MUL || ESP32_GENERATED_EW_CONSTANT_PAD
  // The loops' own counts too.
  elementwise::counters() = ew_saved;
#endif
  int cases = 0;
  int mismatches = check.exhausted() ? 1 : 0;
  for (CheckOp* op : ops) {
    printf("fastpath check: %s %d case(s), %d fast, %d fallback, "
           "%d mismatch(es)\n",
           op->name, op->cases, op->fast, op->fallbacks, op->mismatches);
    cases += op->cases;
    mismatches += op->mismatches;
    for (int index = 0; index < kCheckCounters && op->counters[index]; ++index) {
      *op->counters[index] = op->saved[index];
    }
  }
  printf("fastpath check: %d case(s), %d mismatch(es)\n", cases, mismatches);
}

// The opt-in fast LayerNorm's rows (fast_layer_norm_f32) against the portable
// kernel's, ExecuTorch's layer_norm_scalar<float>, on the same inputs. The
// reductions are the same arithmetic, so each row's mean and rstd must be the
// portable kernel's bits, and an output may differ from portable's only by
// the reciprocal multiply: n = (x - mean) * rstd against (x - mean) / std is
// three roundings of 2^-24 apart at most, and w * n + b adds one rounding
// each side, so |fast - portable| <= 2^-21 (|w n| + |b|) bounds every finite
// output (2^-149 more where it underflows). Widths cover the eight-lane
// blocks and their tails; rows sit far from zero (where E[x^2] - mean^2
// cancelled the variance away), spread from 1e-30 to 1e19, are constant
// (variance 0, also with eps 0, where 0/0 and 0*inf are both NaN), or hold
// a NaN or an infinity, whose outputs must both be NaN.
void run_fast_layer_norm_check() {
  struct Case {
    char what[40];
    int width;
    float offset;  // each x is offset + scale * u, u uniform in [-1, 1)
    float scale;
    double eps;
    bool affine;  // a weight and a bias, or neither
    uint32_t special;  // bits written over x[3] of each row, when nonzero
  };
  // A named static holding its names, in a section of its own: a build
  // that drops this check (the story firmware's) drops the table with it,
  // where a plain const table sat in the file's shared .rodata, and names
  // behind pointers in its shared string pool, both of which it keeps.
  static const Case cases[] = {
      {"width 1", 1, 0.5f, 1.0f, 1e-5, true, 0},
      {"width 7, the tail alone", 7, 0.0f, 1.0f, 1e-5, false, 0},
      {"width 8, one block", 8, 0.0f, 1.0f, 1e-5, true, 0},
      {"width 9", 9, -2.0f, 3.0f, 1e-5, true, 0},
      {"width 31", 31, 0.0f, 1.0f, 1e-6, false, 0},
      {"width 256", 256, 0.0f, 1.0f, 1e-5, true, 0},
      {"offset 1e4, spread 1", 64, 1e4f, 1.0f, 1e-5, true, 0},
      {"offset -3e5, spread 10", 33, -3e5f, 10.0f, 1e-5, false, 0},
      {"spread 1e-30", 16, 1e-30f, 1e-30f, 1e-5, true, 0},
      {"spread 1e19, squares past FLT_MAX", 16, 0.0f, 1e19f, 1e-5, true, 0},
      {"a constant row", 24, 5.0f, 0.0f, 1e-5, true, 0},
      {"a constant row, eps 0", 24, -7.0f, 0.0f, 0.0, true, 0},
      {"a NaN", 16, 0.0f, 1.0f, 1e-5, true, 0x7fc12345u},
      {"an infinity", 16, 0.0f, 1.0f, 1e-5, true, 0x7f800000u},
  };
  constexpr int kRows = 3;
  constexpr int kWidest = 256;
  // x, both outputs, weight, bias; then both sides' mean and rstd.
  float* memory = static_cast<float*>(
      malloc(sizeof(float) * (3 * kRows * kWidest + 2 * kWidest + 4 * kRows)));
  if (memory == nullptr) {
    printf("fast_layer_norm check: no memory for the check rows\n");
    return;
  }
  float* x = memory;
  float* y_fast = x + kRows * kWidest;
  float* y_portable = y_fast + kRows * kWidest;
  float* weight = y_portable + kRows * kWidest;
  float* bias = weight + kWidest;
  float* stats = bias + kWidest;  // mean, rstd: fast then portable
  auto uniform = [] {
    return static_cast<float>(static_cast<int32_t>(check_random())) * 0x1p-31f;
  };
  int rows = 0;
  int mismatches = 0;
  double widest_gap = 0.0;  // the largest |fast - portable| / bound
  uint32_t seed = 0;
  for (const Case& c : cases) {
    g_check_random = 0x2545f491u ^ (0x9e3779b9u * ++seed);
    const size_t width = static_cast<size_t>(c.width);
    for (size_t index = 0; index < kRows * width; ++index) {
      x[index] = c.offset + c.scale * uniform();
    }
    if (c.special != 0) {
      for (size_t row = 0; row < kRows; ++row) {
        memcpy(&x[row * width + 3], &c.special, sizeof(c.special));
      }
    }
    for (size_t column = 0; column < width; ++column) {
      weight[column] = 0.5f + uniform();
      bias[column] = uniform();
    }
    const float* w = c.affine ? weight : nullptr;
    const float* b = c.affine ? bias : nullptr;
    fast_layer_norm_f32(x, w, b, y_fast, stats, stats + kRows, kRows, width,
                        static_cast<float>(c.eps));
    torch::executor::layer_norm_scalar<float>(
        x, w, b, y_portable, stats + 2 * kRows, stats + 3 * kRows, kRows,
        width, c.eps);
    for (size_t row = 0; row < kRows; ++row) {
      ++rows;
      const float mean = stats[2 * kRows + row];
      const float rstd = stats[3 * kRows + row];
      const char* why = nullptr;
      size_t at = 0;
      uint32_t fast_bits = 0, portable_bits = 0;
      auto differ = [&](const float* fast, const float* portable) {
        memcpy(&fast_bits, fast, sizeof(float));
        memcpy(&portable_bits, portable, sizeof(float));
        return fast_bits != portable_bits;
      };
      if (differ(&stats[row], &stats[2 * kRows + row])) {
        why = "mean differs";
      } else if (differ(&stats[kRows + row], &stats[3 * kRows + row])) {
        why = "rstd differs";
      }
      for (size_t column = 0; why == nullptr && column < width; ++column) {
        const size_t index = row * width + column;
        const float fast = y_fast[index];
        const float portable = y_portable[index];
        if (!differ(&fast, &portable) ||
            (std::isnan(fast) && std::isnan(portable))) {
          continue;
        }
        const double n = std::fabs(static_cast<double>(x[index]) - mean) * rstd;
        const double scale = w ? std::fabs(w[column]) : 1.0;
        const double shift = b ? std::fabs(b[column]) : 0.0;
        const double bound = 0x1p-21 * (scale * n + shift) + 0x1p-149;
        const double gap = std::fabs(static_cast<double>(fast) - portable);
        if (!std::isfinite(fast) || !std::isfinite(portable) ||
            !(gap <= bound)) {
          why = "an output is outside the reciprocal's rounding";
          at = column;
        } else {
          widest_gap = std::max(widest_gap, gap / bound);
        }
      }
      if (why != nullptr) {
        if (mismatches < 4) {
          printf("fast_layer_norm check: MISMATCH \"%s\" row %u column %u: %s "
                 "(fast %08x, portable %08x)\n",
                 c.what, (unsigned)row, (unsigned)at, why,
                 (unsigned)fast_bits, (unsigned)portable_bits);
        }
        ++mismatches;
      }
    }
  }
  free(memory);
  printf("fast_layer_norm check: %d case(s), %d mismatch(es); %d rows, "
         "widest output gap %.3f of its bound\n",
         static_cast<int>(sizeof(cases) / sizeof(cases[0])), mismatches, rows,
         widest_gap);
}

}  // namespace espressif
