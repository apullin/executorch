/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Weight zero points, float offsets and dynamic activations: the
// runtime's arithmetic, case by case, for test_asym_conformance.py and
// test_asym_act_conformance.py to hold against quantizer/mirrors.py.
//
//   esp32_host_runner --asym-conformance CASES OUT
//
// Every case runs esp32_asym.cpp's own functions -- affine_weights, codes_row,
// accumulate / accumulate_fused (with the C references of the PIE kernels),
// fold_zero_points, dequant_row -- on activation codes, weights and tables the
// test chose to be adversarial (zero points at the code-range ends, unsigned
// codes, ragged last groups, a K at the exactness bound), and writes every
// result's bytes. The loops around these calls in esp32_backend.cpp are the
// asym parity fixtures' to prove; this is the arithmetic at its limits.
//
// CASES (little-endian): "ASCF", u32 version 1, u32 count, then per case
//   u32 kind (1 matmul, 2 embedding gather, 3 staged fold), u32 flags,
//   u32 k_pad, u32 gsize (0: one group), u32 n, u32 rows, u32 k,
//   u32 zp_kind, u32 offset_kind, i32 storage_bias,
//   then eight arrays, each a u32 byte count and the bytes: activation codes
//   (rows x k_pad int8), activation scales (rows f32), weight codes (n x k_pad
//   int8, or packed nibbles for the fused int4 path and i4 tables), scales
//   (n x groups, f32 or f16), zero points (int8 or f32), offsets (f32 or
//   f16), bias (n f32), gather indices (rows u32).
// flags: 1 fused int4 kernel, 2 PIE dot (else the scalar loop), 4 fp16
//   scales, 8 an int4 table (embedding), 16 a bias, 32/64 int16/int32 weight
//   sums, 128 the float-offset terms T_g wanted (kinds 4 and 5).
// Kinds 4-6 (dynamic activations) read four more arrays: the
//   EXT_ACT_QUANT record (44 B), the float activations, the weight sums
//   S' (n x groups) and a ConvGather (kind 5):
//   4 quantise rows x k activations with the record (per row, or over all of
//     them per tensor): codes (rows x k_pad int8), scales (rows f32), zero
//     points (rows i32), X_g (rows x groups i32), and T_g with flag 128;
//   5 the conv gather of `rows` output positions over the activations at the
//     per-tensor parameters: codes, the one scale and zero point, X_g, T_g;
//   6 a matmul from float activations: quantise each row, then every
//     column's corrected dot and epilogue: rows x n floats.
// Kind 7 (the output quantisation) reads two more arrays: the
//   EXT_OUT_QUANT record (24 B) and the float values, and writes each value
//   requantised (esp32_asym::requantize): as many floats.
// Kinds 8-10 (ExecuTorch's stock quantised K/V cache) take their
//   operands from the common arrays (the activation slots stay empty):
//   8 the K/V row writer (kv_affine::quantize_row) on rows x k floats, read from
//     the activation-scales slot: codes (rows x k int8), scales (rows f32),
//     zero points (rows int8);
//   9 OP_BMM_KCACHE_AFFINE_I8's arithmetic (esp32_asym::kcache_query_row and
//     kcache_score): query floats (rows x k, the activation-scales slot)
//     against n key rows -- codes (n x k int8, staged zero-padded to k_pad),
//     scales (n f32), zero points (n int8) -- with the PIE dot under flag 2:
//     rows x n scores;
//   10 the cache read (kv_affine::dequantize_row): codes (rows x k int8),
//     scales (rows f32), zero points (rows int8): rows x k floats.
// Kinds 11 and 12 (16-bit activations) read three more arrays: the
//   EXT_ACT_QUANT record (code_bits 16), the float activations and a
//   ConvGather (kind 12):
//   11 quantise rows x k activations to int16 codes (esp32_asym::
//     quantize_row16), rows x k_pad int16; then, with n > 0, every column's
//     int64 dot and epilogue against the weight codes (a16_column): rows x n
//     floats;
//   12 the int16 conv gather of `rows` output positions (quantize_conv_row16):
//     rows x k_pad int16.
// OUT: per case, a u32 byte count and the result: rows x n floats (matmul),
//   rows x k floats (gather), n x k_pad int8 (fold); kinds 4, 5 and 7 write
//   each of their results as its own counted array.
#include "esp32_asym.h"
#include "esp32_kernels.h"
#include "kv_affine.h"
#include "quant_rules.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

// ExecuTorch's own quantize kernel (kernels/quantized/cpu/op_quantize.cpp,
// linked into the host runner): the non-finite check's reference.
namespace torch {
namespace executor {
namespace native {
executorch::aten::Tensor& quantize_per_tensor_out(
    const executorch::aten::Tensor& input, double scale, int64_t zero_point,
    int64_t quant_min, int64_t quant_max, executorch::aten::ScalarType dtype,
    executorch::aten::Tensor& out);
}  // namespace native
}  // namespace executor
}  // namespace torch

namespace {

using esp32_asym::ActParams;
using esp32_asym::ActQuant;
using esp32_asym::AffineWeights;
using esp32_asym::ConvGather;
using esp32_asym::OpQuant;
using esp32_asym::OutQuant;
using esp32_asym::WeightAffine;

constexpr uint32_t kFused = 1, kPie = 2, kFp16Scales = 4, kI4Table = 8,
                   kBias = 16, kSums16 = 32, kSums32 = 64, kTerms = 128;

struct Reader {
  std::vector<uint8_t> data;
  size_t at = 0;
  uint32_t u32() {
    if (at + 4 > data.size()) throw std::runtime_error("truncated cases");
    uint32_t v;
    std::memcpy(&v, data.data() + at, 4);
    at += 4;
    return v;
  }
  std::vector<uint8_t> bytes() {
    const uint32_t n = u32();
    if (at + n > data.size()) throw std::runtime_error("truncated array");
    std::vector<uint8_t> out(data.begin() + at, data.begin() + at + n);
    at += n;
    return out;
  }
};

template <typename T>
void put(std::vector<uint8_t>* out, const std::vector<T>& v) {
  const uint32_t n = (uint32_t)(v.size() * sizeof(T));
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&n);
  out->insert(out->end(), p, p + 4);
  const uint8_t* q = reinterpret_cast<const uint8_t*>(v.data());
  out->insert(out->end(), q, q + n);
}

// prepare_i4_activation's layout: per 32 codes, the 16 even then the 16 odd.
void deinterleave(const int8_t* x, uint32_t k_pad, int8_t* xd) {
  for (uint32_t c = 0; c < k_pad; c += 32) {
    for (uint32_t t = 0; t < 16; ++t) {
      xd[c + t] = x[c + 2 * t];
      xd[c + 16 + t] = x[c + 2 * t + 1];
    }
  }
}

}  // namespace

int run_asym_conformance(const char* cases_path, const char* out_path) {
  Reader in;
  {
    std::ifstream f(cases_path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot read ") + cases_path);
    in.data.assign(std::istreambuf_iterator<char>(f), {});
  }
  if (in.data.size() < 12 || std::memcmp(in.data.data(), "ASCF", 4) != 0) {
    throw std::runtime_error("not an asym conformance case file");
  }
  in.at = 4;
  if (in.u32() != 1) throw std::runtime_error("case file version");
  const uint32_t count = in.u32();
  std::vector<uint8_t> out;
  for (uint32_t c = 0; c < count; ++c) {
    const uint32_t kind = in.u32(), flags = in.u32(), k_pad = in.u32(),
                   gsize = in.u32(), n = in.u32(), rows = in.u32(),
                   k = in.u32(), zp_kind = in.u32(), offset_kind = in.u32();
    const int32_t storage_bias = (int32_t)in.u32();
    const std::vector<uint8_t> xq = in.bytes(), sx = in.bytes(),
                               codes = in.bytes(), scales = in.bytes(),
                               zps = in.bytes(), offsets = in.bytes(),
                               bias = in.bytes(), indices = in.bytes();
    WeightAffine record{zp_kind, offset_kind, esp32_asym::kSumNone,
                        storage_bias, 0, 0, 0};
    OpQuant q{nullptr, &record, nullptr,
              zps.empty() ? nullptr : zps.data(),
              offsets.empty() ? nullptr : offsets.data(), nullptr, false};
    const bool fp16 = (flags & kFp16Scales) != 0;
    if (kind == 1) {
      const AffineWeights w = esp32_asym::affine_weights(
          q, k, k_pad, gsize, fp16, scales.data());
      std::vector<float> y((size_t)rows * n);
      std::vector<int32_t> xsum(w.ngw), raw(w.ngw);
      std::vector<float> toff(w.ngw);
      std::vector<int8_t> copy(k_pad);
      alignas(16) static int8_t xd[1 << 18];
      const int8_t* x8 = reinterpret_cast<const int8_t*>(xq.data());
      const float* sxv = reinterpret_cast<const float*>(sx.data());
      const float* b = reinterpret_cast<const float*>(bias.data());
      for (uint32_t r = 0; r < rows; ++r) {
        const int8_t* xr = x8 + (size_t)r * k_pad;
        // The case's codes, as Stage B's arrive: X_g (and T_g) from codes made
        // elsewhere, zero point 0, the row's scale.
        ActQuant made{};
        made.scale = sxv[r];
        esp32_asym::codes_row(made, xr, k, w, copy.data(), xsum.data(),
                              w.offsets ? toff.data() : nullptr);
        if (flags & kFused) deinterleave(xr, k_pad, xd);
        for (uint32_t col = 0; col < n; ++col) {
          const float init = (flags & kBias) ? b[col] : 0.0f;
          float v;
          if (flags & kFused) {
            dot_i4_groups_s3(codes.data() + (size_t)col * (k_pad / 2), xd,
                             raw.data(), (int)w.ngw, (int)(w.gsize / 2));
            v = esp32_asym::accumulate_fused(w, col, raw.data(), sxv[r], 0,
                                             xsum.data(), toff.data(), init);
          } else {
            v = esp32_asym::accumulate(
                w, col, xr,
                reinterpret_cast<const int8_t*>(codes.data()) +
                    (size_t)col * k_pad,
                (flags & kPie) != 0, sxv[r], 0, xsum.data(), toff.data(),
                init);
          }
          y[(size_t)r * n + col] = v;
        }
      }
      put(&out, y);
    } else if (kind == 2) {
      const bool i4 = (flags & kI4Table) != 0;
      const uint32_t g = gsize ? gsize : k;
      const uint32_t ng = (k + g - 1) / g;
      const size_t stride = i4 ? k_pad / 2 : k_pad;
      const uint32_t* idx = reinterpret_cast<const uint32_t*>(indices.data());
      std::vector<float> y((size_t)rows * k);
      for (uint32_t j = 0; j < rows; ++j) {
        esp32_asym::dequant_row(q, i4, codes.data() + (size_t)idx[j] * stride,
                                k, g, ng, scales.data(), fp16,
                                (size_t)idx[j] * ng, y.data() + (size_t)j * k);
      }
      put(&out, y);
    } else if (kind >= 4 && kind <= 6) {
      const std::vector<uint8_t> act_b = in.bytes(), xf_b = in.bytes(),
                                 sums_b = in.bytes(), gather_b = in.bytes();
      ActQuant act;
      if (act_b.size() != sizeof(act)) throw std::runtime_error("act record");
      std::memcpy(&act, act_b.data(), sizeof(act));
      record.sum_kind = (flags & kSums16) ? esp32_asym::kSumInt16
                        : (flags & kSums32) ? esp32_asym::kSumInt32
                                            : esp32_asym::kSumNone;
      q.sums = sums_b.empty() ? nullptr : sums_b.data();
      q.act = &act;
      const bool weightless = zp_kind == esp32_asym::kZpNone &&
          offset_kind == esp32_asym::kOffsetNone &&
          record.sum_kind == esp32_asym::kSumNone && storage_bias == 0;
      if (weightless) q.weight = nullptr;   // symmetric codes, no record
      const AffineWeights w = esp32_asym::affine_weights(
          q, k, k_pad, gsize, fp16, scales.data());
      const float* xf = reinterpret_cast<const float*>(xf_b.data());
      const size_t n_x = xf_b.size() / sizeof(float);
      // Per tensor, or static (the record's constants): chosen once.
      const bool tensor = act.granularity != esp32_asym::kGranularityRow;
      const ActParams whole = tensor ? esp32_asym::choose(act, xf, n_x)
                                     : ActParams{0.0f, 0};
      std::vector<int8_t> codes8((size_t)rows * k_pad);
      std::vector<float> sxv(rows), toff((size_t)rows * w.ngw);
      std::vector<int32_t> zxv(rows), xsum((size_t)rows * w.ngw);
      const bool terms = (flags & kTerms) != 0 || w.offsets != nullptr;
      for (uint32_t r = 0; r < rows; ++r) {
        int8_t* dst = codes8.data() + (size_t)r * k_pad;
        int32_t* xs = xsum.data() + (size_t)r * w.ngw;
        float* tf = terms ? toff.data() + (size_t)r * w.ngw : nullptr;
        ActParams p = whole;
        if (kind == 5) {
          ConvGather g;
          if (gather_b.size() != sizeof(g)) throw std::runtime_error("gather");
          std::memcpy(&g, gather_b.data(), sizeof(g));
          esp32_asym::quantize_conv_row(act, p, xf, g, r, w, dst, xs, tf);
        } else {
          if (!tensor) p = esp32_asym::choose(act, xf + (size_t)r * k, k);
          esp32_asym::quantize_row(act, p, xf + (size_t)r * k, k, w, dst, xs,
                                   tf);
        }
        sxv[r] = p.sx;
        zxv[r] = p.zx;
      }
      if (kind != 6) {
        put(&out, codes8);
        put(&out, sxv);
        put(&out, zxv);
        put(&out, xsum);
        if (flags & kTerms) put(&out, toff);
        continue;
      }
      std::vector<float> y((size_t)rows * n);
      std::vector<int32_t> raw(w.ngw);
      alignas(16) static int8_t xd6[1 << 18];
      const float* b = reinterpret_cast<const float*>(bias.data());
      for (uint32_t r = 0; r < rows; ++r) {
        const int8_t* xr = codes8.data() + (size_t)r * k_pad;
        const int32_t* xs = xsum.data() + (size_t)r * w.ngw;
        const float* tf = toff.data() + (size_t)r * w.ngw;
        if (flags & kFused) deinterleave(xr, k_pad, xd6);
        for (uint32_t col = 0; col < n; ++col) {
          const float init = (flags & kBias) ? b[col] : 0.0f;
          float v;
          if (flags & kFused) {
            dot_i4_groups_s3(codes.data() + (size_t)col * (k_pad / 2), xd6,
                             raw.data(), (int)w.ngw, (int)(w.gsize / 2));
            v = esp32_asym::accumulate_fused(w, col, raw.data(), sxv[r],
                                             zxv[r], xs, tf, init);
          } else {
            v = esp32_asym::accumulate(
                w, col, xr,
                reinterpret_cast<const int8_t*>(codes.data()) +
                    (size_t)col * k_pad,
                (flags & kPie) != 0, sxv[r], zxv[r], xs, tf, init);
          }
          y[(size_t)r * n + col] = v;
        }
      }
      put(&out, y);
    } else if (kind == 7) {
      const std::vector<uint8_t> out_b = in.bytes(), yf_b = in.bytes();
      OutQuant o;
      if (out_b.size() != sizeof(o)) throw std::runtime_error("out record");
      std::memcpy(&o, out_b.data(), sizeof(o));
      const OpQuant oq{nullptr, nullptr, &o, nullptr, nullptr, nullptr, false,
                       1.0f / o.scale};
      std::vector<float> y(yf_b.size() / sizeof(float));
      for (size_t t = 0; t < y.size(); ++t) {
        float v;
        std::memcpy(&v, yf_b.data() + t * sizeof(float), sizeof(v));
        y[t] = esp32_asym::requantize(oq, v);
      }
      put(&out, y);
    } else if (kind == 8) {
      const float* xf = reinterpret_cast<const float*>(sx.data());
      std::vector<int8_t> codes8((size_t)rows * k), zp8(rows);
      std::vector<float> sv(rows);
      for (uint32_t r = 0; r < rows; ++r) {
        espressif::kv_affine::quantize_row(xf + (size_t)r * k, k,
                                           codes8.data() + (size_t)r * k,
                                           sv.data() + r, zp8.data() + r);
      }
      put(&out, codes8);
      put(&out, sv);
      put(&out, zp8);
    } else if (kind == 9) {
      const float* qf = reinterpret_cast<const float*>(sx.data());
      const int8_t* kc = reinterpret_cast<const int8_t*>(codes.data());
      const float* ks = reinterpret_cast<const float*>(scales.data());
      const int8_t* kz = reinterpret_cast<const int8_t*>(zps.data());
      alignas(16) static int8_t staged[1 << 22];
      alignas(16) static int8_t qq[1 << 18];
      if ((size_t)n * k_pad > sizeof(staged) || k_pad > sizeof(qq)) {
        throw std::runtime_error("kcache case too large");
      }
      for (uint32_t c2 = 0; c2 < n; ++c2) {
        std::memcpy(staged + (size_t)c2 * k_pad, kc + (size_t)c2 * k, k);
        std::memset(staged + (size_t)c2 * k_pad + k, 0, k_pad - k);
      }
      std::vector<float> y((size_t)rows * n);
      for (uint32_t r = 0; r < rows; ++r) {
        int32_t qsum = 0;
        const float sq = esp32_asym::kcache_query_row(qf + (size_t)r * k, k,
                                                      k_pad, qq, &qsum);
        for (uint32_t c2 = 0; c2 < n; ++c2) {
          y[(size_t)r * n + c2] = esp32_asym::kcache_score(
              qq, qsum, sq, staged + (size_t)c2 * k_pad, k_pad, ks[c2],
              kz[c2], (flags & kPie) != 0);
        }
      }
      put(&out, y);
    } else if (kind == 10) {
      const int8_t* c8 = reinterpret_cast<const int8_t*>(codes.data());
      const float* sv = reinterpret_cast<const float*>(scales.data());
      const int8_t* z8 = reinterpret_cast<const int8_t*>(zps.data());
      std::vector<float> y((size_t)rows * k);
      for (uint32_t r = 0; r < rows; ++r) {
        espressif::kv_affine::dequantize_row(c8 + (size_t)r * k, k, sv[r],
                                             z8[r], y.data() + (size_t)r * k);
      }
      put(&out, y);
    } else if (kind == 11 || kind == 12) {
      const std::vector<uint8_t> act_b = in.bytes(), xf_b = in.bytes(),
                                 gather_b = in.bytes();
      ActQuant act;
      if (act_b.size() != sizeof(act)) throw std::runtime_error("act record");
      std::memcpy(&act, act_b.data(), sizeof(act));
      const float* xf = reinterpret_cast<const float*>(xf_b.data());
      std::vector<int16_t> codes16((size_t)rows * k_pad);
      for (uint32_t r = 0; r < rows; ++r) {
        int16_t* dst = codes16.data() + (size_t)r * k_pad;
        if (kind == 12) {
          ConvGather g;
          if (gather_b.size() != sizeof(g)) throw std::runtime_error("gather");
          std::memcpy(&g, gather_b.data(), sizeof(g));
          esp32_asym::quantize_conv_row16(act, xf, g, r, k_pad, dst);
        } else {
          esp32_asym::quantize_row16(act, xf + (size_t)r * k, k, k_pad, dst);
        }
      }
      put(&out, codes16);
      if (kind == 12 || n == 0) continue;
      // Symmetric int8 weights: no weight record.
      const OpQuant none{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         false, 0.0f};
      const AffineWeights w =
          esp32_asym::affine_weights(none, k, k_pad, 0, fp16, scales.data());
      const int8_t* w8 = reinterpret_cast<const int8_t*>(codes.data());
      const float* b = reinterpret_cast<const float*>(bias.data());
      std::vector<float> y((size_t)rows * n);
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t col = 0; col < n; ++col) {
          y[(size_t)r * n + col] = esp32_asym::a16_column(
              w, col, codes16.data() + (size_t)r * k_pad,
              w8 + (size_t)col * k_pad, act.scale,
              (flags & kBias) ? b[col] : 0.0f);
        }
      }
      put(&out, y);
    } else if (kind == 3) {
      const AffineWeights w = esp32_asym::affine_weights(
          q, k_pad, k_pad, gsize, fp16, scales.data());
      std::vector<int8_t> folded(
          reinterpret_cast<const int8_t*>(codes.data()),
          reinterpret_cast<const int8_t*>(codes.data()) + codes.size());
      for (uint32_t row = 0; row < n; ++row) {
        esp32_asym::fold_zero_points(w, row, folded.data() + (size_t)row * k_pad);
      }
      put(&out, folded);
    } else {
      throw std::runtime_error("unknown case kind");
    }
  }
  std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(out.data()), (std::streamsize)out.size());
  if (!f) throw std::runtime_error(std::string("cannot write ") + out_path);
  std::printf("asym conformance: %u case(s)\n", (unsigned)count);
  return 0;
}

namespace {

uint32_t g_portable_lanes_apart = 0;

// x's code from ExecuTorch's quantize_per_tensor, on nine copies of it: on an
// aarch64 host eight take the kernel's NEON block (FCVTNS, then saturating
// narrows) and the ninth its scalar tail, which must agree.
int32_t portable_quantize(float x, float scale, int32_t zero_point,
                          int32_t quant_min, int32_t quant_max) {
  using executorch::aten::ScalarType;
  using executorch::aten::Tensor;
  using executorch::aten::TensorImpl;
  constexpr int kCopies = 9;
  float in[kCopies];
  for (float& v : in) {
    v = x;
  }
  int32_t out32[kCopies] = {};
  int16_t out16[kCopies] = {};
  int8_t out8[kCopies] = {};
  uint8_t outu8[kCopies] = {};
  ScalarType dtype = ScalarType::Int;
  void* out = out32;
  if (quant_min >= -128 && quant_max <= 127) {
    dtype = ScalarType::Char;
    out = out8;
  } else if (quant_min >= 0 && quant_max <= 255) {
    dtype = ScalarType::Byte;
    out = outu8;
  } else if (quant_min >= -32768 && quant_max <= 32767) {
    dtype = ScalarType::Short;
    out = out16;
  }
  TensorImpl::SizesType sizes[1] = {kCopies};
  TensorImpl::DimOrderType order[1] = {0};
  TensorImpl::StridesType strides[1] = {1};
  TensorImpl in_impl(ScalarType::Float, 1, sizes, in, order, strides);
  TensorImpl out_impl(dtype, 1, sizes, out, order, strides);
  Tensor input(&in_impl);
  Tensor output(&out_impl);
  torch::executor::native::quantize_per_tensor_out(
      input, (double)scale, zero_point, quant_min, quant_max, dtype, output);
  int32_t codes[kCopies];
  for (int i = 0; i < kCopies; ++i) {
    codes[i] = dtype == ScalarType::Char    ? out8[i]
               : dtype == ScalarType::Byte  ? outu8[i]
               : dtype == ScalarType::Short ? out16[i]
                                            : out32[i];
    g_portable_lanes_apart += codes[i] != codes[0];
  }
  return codes[0];
}

}  // namespace

// The non-finite rule's host check (--selftest): nonfinite_check.cpp's
// quantisers against ExecuTorch's own kernel, and the transcription the
// device holds them to against the same kernel.
int run_nonfinite_check() {
  const uint32_t mismatches =
      espressif::nonfinite::run_check(portable_quantize);
  if (g_portable_lanes_apart != 0) {
    std::printf("  MISMATCH nonfinite portable kernel: %u lane(s) apart from "
                "their scalar tail\n", g_portable_lanes_apart);
  }
  return mismatches != 0 || g_portable_lanes_apart != 0;
}
