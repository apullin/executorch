/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "esp32_kernels.h"
#include "esp32_platform.h"
#include "quant_rules.h"

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
  ++failures; } } while (0)

int main() {
  alignas(16) std::array<int8_t, 512> a{}, b{};
  uint32_t random = 123;
  for (int trial = 0; trial < 30; ++trial) {
    for (size_t i = 0; i < a.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      a[i] = static_cast<int8_t>(random >> 24);
      random = random * 1664525u + 1013904223u;
      b[i] = static_cast<int8_t>(random >> 24);
    }
    for (int n = 0; n <= 512; ++n) {
      int32_t scalar = 0;
      for (int i = 0; i < n - n % 16; ++i) {
        scalar += static_cast<int32_t>(a[i]) * b[i];
      }
      CHECK(dot_i8_s3(a.data(), b.data(), n) == scalar);
    }
  }
  // The assembly exposes ACCX's low word even when the mathematical sum
  // exceeds int32. Include -128, which the delegate's symmetric codes omit.
  std::vector<int8_t> extreme(131088, -128);
  const uint32_t bits = static_cast<uint32_t>(extreme.size() * uint64_t{16384});
  int32_t wrapped;
  std::memcpy(&wrapped, &bits, sizeof(wrapped));
  CHECK(dot_i8_s3(extreme.data(), extreme.data(), extreme.size()) == wrapped);

  alignas(16) std::array<uint8_t, 256> packed{};
  alignas(16) std::array<int8_t, 512> unpacked{};
  for (size_t i = 0; i < packed.size(); ++i) packed[i] = i;
  for (int n = 0; n <= 256; ++n) {
    unpacked.fill(42);
    unpack_i4_s3(packed.data(), unpacked.data(), n);
    for (int i = 0; i < n - n % 16; ++i) {
      CHECK(unpacked[2 * i] == (packed[i] & 15) - 8);
      CHECK(unpacked[2 * i + 1] == (packed[i] >> 4) - 8);
    }
    for (int i = 2 * (n - n % 16); i < 512; ++i) CHECK(unpacked[i] == 42);
  }

  // The fused int4 dot against unpack + dot: per group, the raw sum of the
  // stored codes less 8 * sum(x) must equal the dot of the true codes, over
  // activations deinterleaved per 32 as prepare_i4_activation lays them out.
  // Include every nibble, -128 and 127, and a row that ends mid-buffer.
  alignas(16) std::array<uint8_t, 512> row{};
  alignas(16) std::array<int8_t, 1024> x{}, xd{}, codes{};
  for (int trial = 0; trial < 20; ++trial) {
    for (size_t i = 0; i < row.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      row[i] = i < 16 ? static_cast<uint8_t>(i | (i << 4))
                      : static_cast<uint8_t>(random >> 24);
    }
    for (size_t i = 0; i < x.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      x[i] = static_cast<int8_t>(random >> 24);
    }
    x[0] = 127;
    x[1] = -128;
    if (trial == 0) {  // Every code 15 against -128: the largest group sums.
      row.fill(0xFF);
      x.fill(-128);
    }
    for (size_t c = 0; c < x.size(); c += 32) {
      for (int t = 0; t < 16; ++t) {
        xd[c + t] = x[c + 2 * t];
        xd[c + 16 + t] = x[c + 2 * t + 1];
      }
    }
    unpack_i4_s3(row.data(), codes.data(), static_cast<int>(row.size()));
    for (int group_bytes : {16, 32, 64, 128, 256, 512}) {
      for (int n_groups = 0; n_groups * group_bytes <= 512; ++n_groups) {
        std::array<int32_t, 33> sums;
        sums.fill(0x5A5A5A5A);
        dot_i4_groups_s3(row.data(), xd.data(), sums.data(), n_groups, group_bytes);
        for (int g = 0; g < n_groups; ++g) {
          const int a = 2 * g * group_bytes;
          int32_t correction = 0;
          for (int i = a; i < a + 2 * group_bytes; ++i) correction += 8 * x[i];
          CHECK(sums[g] - correction ==
                dot_i8_s3(x.data() + a, codes.data() + a, 2 * group_bytes));
        }
        CHECK(sums[n_groups] == 0x5A5A5A5A);
      }
    }
  }
  // Raw sums past int32: the assembly returns ACCX's low word, as the
  // reference's wrap does (the delegate bounds groups far below this).
  const int huge_bytes = 16 * 70000;
  std::vector<uint8_t> huge_row(huge_bytes, 0xFF);
  std::vector<int8_t> huge_x(2 * huge_bytes, -128);
  int32_t huge_sum = 0;
  dot_i4_groups_s3(huge_row.data(), huge_x.data(), &huge_sum, 1, huge_bytes);
  const uint32_t huge_bits = static_cast<uint32_t>(
      static_cast<uint64_t>(-int64_t{2} * huge_bytes * 15 * 128));
  std::memcpy(&wrapped, &huge_bits, sizeof(wrapped));
  CHECK(huge_sum == wrapped);

  for (int i = -256; i <= 256; ++i) {
    const float tie = i + 0.5f;
    for (float value : {std::nextafter(tie, -INFINITY), tie,
                        std::nextafter(tie, INFINITY)}) {
      const int32_t got = value < 0.0f
          ? -esp32_platform_floor_i(-value + 0.5f)
          : esp32_platform_floor_i(value + 0.5f);
      const int32_t scalar = value < 0.0f
          ? -static_cast<int32_t>(std::floor(-value + 0.5f))
          : static_cast<int32_t>(std::floor(value + 0.5f));
      CHECK(got == scalar);
    }
  }
  CHECK(esp32_platform_floor_i(-1.25f) == -2);
  CHECK(esp32_platform_floor_i(1.25f) == 1);

  // ExecuTorch 1.6's rule for non-finite values (quant_rules.h): quant_scaled,
  // then the rails, ties-to-even rounding, zero point and clamp of code_of
  // (esp32_asym.cpp), on upstream's own vectors with the codes its test
  // expects (kernels/quantized/test/op_quantize_test.cpp: NonFiniteValues...,
  // NaNMapsToClampedZeroPoint, LargeFiniteValuesSaturate...; scale 0.5).
  {
    const auto code = [](float x, int32_t zp, int32_t qmin, int32_t qmax) {
      float v = quant_scaled(x * 2.0f);
      v = v < -4194304.0f ? -4194304.0f : v;
      v = v > 4194304.0f ? 4194304.0f : v;
      const int32_t f = esp32_platform_floor_i(v);
      const float frac = v - static_cast<float>(f);
      const int32_t r =
          (frac > 0.5f || (frac == 0.5f && (f & 1))) ? f + 1 : f;
      return r + zp < qmin ? qmin : (r + zp > qmax ? qmax : r + zp);
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float values[11] = {nan, inf, -inf, 0.0f, 1.0f, -1.0f,
                              0.25f, 0.75f, nan, inf, -inf};
    const int32_t codes[11] = {5, 10, 0, 5, 7, 3, 5, 7, 5, 10, 0};
    for (int i = 0; i < 11; ++i) CHECK(code(values[i], 5, 0, 10) == codes[i]);
    for (int32_t zp : {-20, 20}) {
      CHECK(code(nan, zp, -10, 10) == (zp < 0 ? -10 : 10));
      CHECK(code(-nan, zp, -10, 10) == (zp < 0 ? -10 : 10));
    }
    const float large[10] = {FLT_MAX, -FLT_MAX, 1e20f, -1e20f, 1e10f,
                             -1e10f,  0.0f,     1.0f,  FLT_MAX, -FLT_MAX};
    const int32_t large_codes[10] = {10, -10, 10, -10, 10, -10, 5, 7, 10, -10};
    for (int i = 0; i < 10; ++i) {
      CHECK(code(large[i], 5, -10, 10) == large_codes[i]);
    }
    // The helper alone: NaN of either sign becomes +0; every other value,
    // a signed zero included, passes bit for bit.
    for (float x : {nan, -nan}) {
      CHECK(quant_scaled(x) == 0.0f && !std::signbit(quant_scaled(x)));
      CHECK(quant_scaled(x, 0.0f) == 0.0f && !std::signbit(quant_scaled(x, 0.0f)));
      CHECK(quant_scaled(static_cast<double>(x)) == 0.0);
    }
    for (float x : {inf, -inf, FLT_MAX, -FLT_MAX, 0.0f, -0.0f, 1e-45f, 0.5f}) {
      uint32_t in_bits, out_bits, pinned_bits;
      const float out = quant_scaled(x), pinned = quant_scaled(x, 0.0f);
      std::memcpy(&in_bits, &x, sizeof in_bits);
      std::memcpy(&out_bits, &out, sizeof out_bits);
      std::memcpy(&pinned_bits, &pinned, sizeof pinned_bits);
      CHECK(out_bits == in_bits && pinned_bits == in_bits);
      CHECK(quant_scaled(static_cast<double>(x)) == static_cast<double>(x));
    }
  }

  for (int width : {1, 2, 3, 4, 7, 16, 31, 32, 63, 128, 255, 256}) {
    std::vector<float> x(width), w(width), out(width), scalar(width);
    for (int i = 0; i < width; ++i) {
      x[i] = std::sin(static_cast<float>(i) * 1.7f) * 3.0f;
      w[i] = 0.7f + i * 0.01f;
    }
    for (bool weighted : {false, true}) {
      float sum = 0.0f;
      for (float value : x) sum += value * value;
      const float scale = 1.0f / std::sqrt(sum * (1.0f / width) + 1e-5f);
      CHECK(rmsnorm_f32_s3(out.data(), x.data(), weighted ? w.data() : nullptr,
                           width, 1.0f / width, 1e-5f) == 1);
      for (int i = 0; i < width; ++i) {
        scalar[i] = (weighted ? w[i] * x[i] : x[i]) * scale;
        CHECK(std::fabs(out[i] - scalar[i]) <= 8e-7f * (1.0f + std::fabs(scalar[i])));
      }
      auto aliased = x;
      CHECK(rmsnorm_f32_s3(aliased.data(), aliased.data(), weighted ? w.data() : nullptr,
                           width, 1.0f / width, 1e-5f) == 1);
      CHECK(aliased == out);
    }
  }
  float untouched = 9.0f;
  CHECK(rmsnorm_f32_s3(&untouched, nullptr, nullptr, 0, 0.0f, 1e-5f) == 1);
  CHECK(untouched == 9.0f);
  // The assembly's refusals before rsqrt0.s: when mean(x^2) + eps is not a
  // positive normal float, nothing is written and the result is 0.
  struct Refusal {
    std::array<float, 4> x;
    float eps;
  };
  const float big = 1e19f, nan = std::nanf("");
  for (const Refusal& c : {Refusal{{big, big, big, big}, 1e-6f},  // sum -> +inf
                           Refusal{{nan, 1.0f, 1.0f, 1.0f}, 1e-6f},
                           Refusal{{0.0f, 0.0f, 0.0f, 0.0f}, 0.0f},  // t = 0
                           Refusal{{0.0f, 0.0f, 0.0f, 0.0f}, FLT_MIN / 2}}) {
    std::array<float, 4> out;
    out.fill(7.0f);
    CHECK(rmsnorm_f32_s3(out.data(), c.x.data(), nullptr, 4, 0.25f, c.eps) == 0);
    for (float value : out) CHECK(value == 7.0f);
  }
  std::array<float, 4> zero{}, smallest;
  smallest.fill(7.0f);  // t = FLT_MIN is the domain's lower edge, still written.
  CHECK(rmsnorm_f32_s3(smallest.data(), zero.data(), nullptr, 4, 0.25f, FLT_MIN) == 1);
  CHECK(smallest == zero);
  void* allocation = heap_caps_malloc(17, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  CHECK(allocation != nullptr);
  heap_caps_free(allocation);
  const int64_t first_tick = esp_timer_get_time();
  const int64_t second_tick = esp_timer_get_time();
  CHECK(first_tick <= second_tick);
  StaticSemaphore_t storage;
  const SemaphoreHandle_t mutex = xSemaphoreCreateMutexStatic(&storage);
  CHECK(xSemaphoreTake(mutex, portMAX_DELAY) == pdTRUE);
  // Held: another task's zero-wait take fails, as a second split lease must.
  std::thread([&] { CHECK(xSemaphoreTake(mutex, 0) == pdFALSE); }).join();
  xSemaphoreGive(mutex);
  CHECK(xSemaphoreTake(mutex, 0) == pdTRUE);
  xSemaphoreGive(mutex);
  CHECK(xPortGetCoreID() == 0);

  // The host's floor conversion saturates as the chip's FLOOR.S does: past the
  // int32 range, INT32_MAX (+inf and NaN too) or INT32_MIN (-inf). A plain cast
  // is undefined there (x86 gives INT32_MIN for all of them, ARM 0 for NaN).
  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  CHECK(esp32_platform_floor_i(3e9f) == INT32_MAX);
  CHECK(esp32_platform_floor_i(2147483648.0f) == INT32_MAX);
  CHECK(esp32_platform_floor_i(infinity) == INT32_MAX);
  CHECK(esp32_platform_floor_i(quiet_nan) == INT32_MAX);
  CHECK(esp32_platform_floor_i(-3e9f) == INT32_MIN);
  CHECK(esp32_platform_floor_i(-infinity) == INT32_MIN);
  CHECK(esp32_platform_floor_i(-2147483648.0f) == INT32_MIN);
  CHECK(esp32_platform_floor_i(2147483520.0f) == 2147483520);
  CHECK(esp32_platform_floor_i(-2.5f) == -3);
  CHECK(esp32_platform_floor_i(2.5f) == 2);
  CHECK(esp32_platform_floor_i(-0.0f) == 0);

  // A0's inverse scale: fl(1 / s) for a normal scale, 0 below FLT_MIN, so a
  // row like [FLT_MIN, 0] (s = FLT_MIN / 127, subnormal) gets zero codes
  // instead of inf * x. Every scale it inverts has a finite reciprocal.
  CHECK(a0_inverse_scale(0.0f) == 0.0f);
  CHECK(a0_inverse_scale(FLT_MIN / 127.0f) == 0.0f);
  CHECK(a0_inverse_scale(std::nextafter(FLT_MIN, 0.0f)) == 0.0f);
  CHECK(a0_inverse_scale(FLT_MIN) == 0x1p126f);
  CHECK(a0_inverse_scale(1.0f) == 1.0f);
  CHECK(a0_inverse_scale(FLT_MAX) == 1.0f / FLT_MAX);
  for (float s = FLT_MIN; s < 1e-30f; s *= 1.3f) {
    CHECK(std::isfinite(a0_inverse_scale(s)));
  }
  // The static int8 conv path's kernels (esp32_conv.cpp), against their
  // definitions computed here.
  //
  // qacc_lane_codes_s3: images packed here from known raw sums (the layout
  // dw33_qacc16_s3 leaves: lane l's 20 bits at bit 20 * (l % 8) of the
  // 20-byte half at 32 * (l / 8)), every lane, raw sums over the whole 20-bit
  // range, and terms whose clamp binds and whose guard sends some to slow[].
  {
    constexpr int kImages = 64;
    alignas(16) std::array<uint8_t, kImages * 64> img{};
    std::vector<int32_t> raws(kImages * 16);
    for (int u = 0; u < kImages; ++u) {
      for (int l = 0; l < 16; ++l) {
        random = random * 1664525u + 1013904223u;
        int32_t v = static_cast<int32_t>(random >> 12) - (1 << 19);
        if (u == 0) v = l % 2 ? (1 << 19) - 1 : -(1 << 19);  // the extremes
        raws[u * 16 + l] = v;
        uint8_t* half = img.data() + u * 64 + 32 * (l / 8);
        for (int b = 0; b < 20; ++b) {
          const int at = 20 * (l % 8) + b;
          half[at / 8] |= static_cast<uint8_t>(((static_cast<uint32_t>(v) >> b) & 1u) << (at % 8));
        }
      }
    }
    const Esp32LaneEpi sets[3] = {
        {0x00123456, 0x80000000u, -3, 1u << 22, -128, 127, 0},
        {0x3FFFFFFF, 0x00000000u, 0, 1u << 30, -5, 9, 0},
        {0x00000001, 0xFFFFFFF0u, -1, 0x11u, -128, 127, 0},
    };
    for (int l = 0; l < 16; ++l) {
      const uint32_t bit = 20u * (l % 8);
      const uint8_t* p = img.data() + 32 * (l / 8) + 4 * (bit / 32);
      for (const Esp32LaneEpi& base : sets) {
        Esp32LaneEpi t = base;
        t.shift = bit % 32;
        int8_t out[kImages];
        const uint8_t* slow[kImages];
        const int n = qacc_lane_codes_s3(p, kImages, out, &t, slow);
        const int64_t c = static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(t.c_hi)) << 32) | t.c_lo);
        int want = 0;
        for (int u = 0; u < kImages; ++u) {
          const int64_t y = static_cast<int64_t>(raws[u * 16 + l]) * t.g + c;
          const int32_t lo = static_cast<int32_t>(static_cast<uint32_t>(y));
          if ((lo < 0 ? -static_cast<int64_t>(lo) : lo) < t.guard) {
            CHECK(want < n && slow[want] == p + 64 * u);
            ++want;
          }
          const int64_t q = std::min<int64_t>(std::max<int64_t>(y >> 32, t.qlo), t.qhi);
          CHECK(out[u] == static_cast<int8_t>(q));
        }
        CHECK(n == want);
      }
    }
  }
  // dot_rows_codes_s3: every chunk count 1-7 on rows of random codes with
  // -128 every seventh (the extremes), the same terms; chunk counts 0 and 8
  // are out of contract and write nothing.
  {
    constexpr int kRows = 40;
    alignas(16) std::array<int8_t, kRows * 112> rows{};
    alignas(16) std::array<int8_t, 112> w{};
    for (size_t i = 0; i < rows.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      rows[i] = i % 7 == 0 ? -128 : static_cast<int8_t>(random >> 24);
    }
    for (size_t i = 0; i < w.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      w[i] = i % 5 == 0 ? -128 : static_cast<int8_t>(random >> 24);
    }
    const Esp32RowEpi sets[3] = {
        {0x00003456, 0x80000000u, -3, 1u << 22, -128, 127, 0},
        {0x00FFFFFF, 0x00000000u, 0, 1u << 30, -5, 9, 0},
        {0x00000001, 0xFFFFFFF0u, -1, 0x11u, -128, 127, 0},
    };
    for (uint32_t chunks = 0; chunks <= 8; ++chunks) {
      const int kp = 16 * static_cast<int>(chunks);
      for (const Esp32RowEpi& base : sets) {
        Esp32RowEpi t = base;
        t.chunks = chunks;
        int8_t out[kRows];
        std::memset(out, 0x5A, sizeof(out));
        int8_t* slow[kRows];
        const int n = dot_rows_codes_s3(rows.data(), w.data(), out, kRows, &t, slow);
        if (chunks < 1 || chunks > 7) {
          CHECK(n == 0);
          for (int8_t o : out) CHECK(o == 0x5A);
          continue;
        }
        const int64_t c = static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(t.c_hi)) << 32) | t.c_lo);
        int want = 0;
        for (int u = 0; u < kRows; ++u) {
          int32_t raw = 0;
          for (int i = 0; i < kp; ++i) raw += rows[u * kp + i] * w[i];
          CHECK(raw == dot_i8_s3(rows.data() + u * kp, w.data(), kp));
          const int64_t y = static_cast<int64_t>(raw) * t.g + c;
          const int32_t lo = static_cast<int32_t>(static_cast<uint32_t>(y));
          if ((lo < 0 ? -static_cast<int64_t>(lo) : lo) < t.guard) {
            CHECK(want < n && slow[want] == out + u);
            ++want;
          }
          const int64_t q = std::min<int64_t>(std::max<int64_t>(y >> 32, t.qlo), t.qhi);
          CHECK(out[u] == static_cast<int8_t>(q));
        }
        CHECK(n == want);
      }
    }
  }
  // transpose16_s3: every pixel count to 40, strides 16, 24 and 64, planes
  // 37 bytes apart from an odd start; nothing past the copy is written.
  {
    alignas(16) std::array<int8_t, 16 * 37 + 96> src{};
    alignas(16) std::array<int8_t, 40 * 64> dst{};
    for (size_t i = 0; i < src.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      src[i] = static_cast<int8_t>(random >> 24);
    }
    for (int stride : {16, 24, 64}) {
      for (int count = 0; count <= 40; ++count) {
        if (count * stride > static_cast<int>(dst.size())) continue;
        dst.fill(0x5A);
        transpose16_s3(src.data() + 3, 37, dst.data(), stride, count);
        for (int i = 0; i < static_cast<int>(dst.size()); ++i) {
          const int p = i / stride, j = i % stride;
          const int8_t want = p < count && j < 16 ? src[3 + j * 37 + p] : 0x5A;
          CHECK(dst[i] == want);
        }
      }
    }
  }
  // dot_rows_long_s3: chunk counts 8-72 (past the 64 the conv path gives it)
  // on rows of random codes with -128 every seventh, the same terms as
  // dot_rows_codes_s3's; chunk counts below 8 are out of contract and write
  // nothing.
  {
    constexpr int kRows = 6;
    constexpr int kMost = 72;
    alignas(16) std::array<int8_t, kRows * 16 * kMost> rows{};
    alignas(16) std::array<int8_t, 16 * kMost> w{};
    for (size_t i = 0; i < rows.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      rows[i] = i % 7 == 0 ? -128 : static_cast<int8_t>(random >> 24);
    }
    for (size_t i = 0; i < w.size(); ++i) {
      random = random * 1664525u + 1013904223u;
      w[i] = i % 5 == 0 ? -128 : static_cast<int8_t>(random >> 24);
    }
    const Esp32RowEpi sets[3] = {
        {0x00000456, 0x80000000u, -3, 1u << 22, -128, 127, 0},
        {0x000FFFFF, 0x00000000u, 0, 1u << 30, -5, 9, 0},
        {0x00000001, 0xFFFFFFF0u, -1, 0x11u, -128, 127, 0},
    };
    for (uint32_t chunks = 6; chunks <= static_cast<uint32_t>(kMost); ++chunks) {
      const int kp = 16 * static_cast<int>(chunks);
      for (const Esp32RowEpi& base : sets) {
        Esp32RowEpi t = base;
        t.chunks = chunks;
        int8_t out[kRows + 1];
        std::memset(out, 0x5A, sizeof(out));
        int8_t* slow[kRows];
        const int n = dot_rows_long_s3(rows.data(), w.data(), out, kRows, &t, slow);
        CHECK(out[kRows] == 0x5A);
        if (chunks < 8) {
          CHECK(n == 0);
          for (int8_t o : out) CHECK(o == 0x5A);
          continue;
        }
        const int64_t c = static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(t.c_hi)) << 32) | t.c_lo);
        int want = 0;
        for (int u = 0; u < kRows; ++u) {
          int32_t raw = 0;
          for (int i = 0; i < kp; ++i) raw += rows[u * kp + i] * w[i];
          CHECK(raw == dot_i8_s3(rows.data() + u * kp, w.data(), kp));
          const int64_t y = static_cast<int64_t>(raw) * t.g + c;
          const int32_t lo = static_cast<int32_t>(static_cast<uint32_t>(y));
          if ((lo < 0 ? -static_cast<int64_t>(lo) : lo) < t.guard) {
            CHECK(want < n && slow[want] == out + u);
            ++want;
          }
          const int64_t q = std::min<int64_t>(std::max<int64_t>(y >> 32, t.qlo), t.qhi);
          CHECK(out[u] == static_cast<int8_t>(q));
        }
        CHECK(n == want);
      }
    }
  }
  // dw5_plane_s3 and dw3_plane_s3: each run's image (dw33_qacc16_s3's layout,
  // read back bit by bit here) against the sums of their definition, over
  // window rows at every alignment (37 bytes apart), B taps right after the
  // A taps and further on, one to three runs, codes and weights across the
  // int8 range; then the lanes' extremes, every code and weight -128
  // (+409,600 for 25 taps) and -128 by 127 (-406,400), which a 20-bit lane
  // holds and a 19-bit one would not.
  {
    alignas(16) std::array<int8_t, 512> codes{};
    alignas(16) std::array<uint8_t, 3 * 64> img{};
    std::array<int8_t, 25> w{};
    for (int pass = 0; pass < 3; ++pass) {
      for (size_t i = 0; i < codes.size(); ++i) {
        random = random * 1664525u + 1013904223u;
        codes[i] = pass ? int8_t{-128} : static_cast<int8_t>(random >> 24);
      }
      for (size_t i = 0; i < w.size(); ++i) {
        random = random * 1664525u + 1013904223u;
        w[i] = pass == 1 ? int8_t{-128}
                         : (pass == 2 ? int8_t{127}
                                      : static_cast<int8_t>(random >> 24));
      }
      for (int kh : {3, 5}) {
        const int ka = (kh + 1) / 2, kb = kh / 2;
        for (int boff : {ka, ka + 29}) {
          for (int runs = 1; runs <= 3; ++runs) {
            const int8_t* rows[5];
            for (int i = 0; i < kh; ++i) rows[i] = codes.data() + 5 + 37 * i;
            img.fill(0x5A);
            if (kh == 5) {
              (boff == ka ? dw5_plane1_s3 : dw5_plane_s3)(rows, boff, w.data(),
                                                          img.data(), runs);
            } else {
              (boff == ka ? dw3_plane1_s3 : dw3_plane_s3)(rows, boff, w.data(),
                                                          img.data(), runs);
            }
            for (int u = 0; u < runs; ++u) {
              for (int l = 0; l < 16; ++l) {
                int32_t want = 0;
                for (int i = 0; i < kh; ++i) {
                  const int8_t* a = rows[i] + 16 * u + l;
                  for (int j = 0; j < ka; ++j) want += a[j] * w[i * kh + j];
                  for (int j = 0; j < kb; ++j) {
                    want += a[boff + j] * w[i * kh + ka + j];
                  }
                }
                const uint8_t* half = img.data() + 64 * u + 32 * (l / 8);
                uint32_t v = 0;
                for (int b = 0; b < 20; ++b) {
                  const int at = 20 * (l % 8) + b;
                  v |= static_cast<uint32_t>((half[at / 8] >> (at % 8)) & 1u) << b;
                }
                CHECK(static_cast<int32_t>(v << 12) >> 12 == want);
              }
            }
          }
        }
      }
    }
  }
  // dw5_pairs_s3: each run's image against its definition, the window rows
  // 8 and 24 bytes apart, planes 104 bytes apart, one to three runs, codes
  // and weights across the int8 range and at -128.
  {
    alignas(16) std::array<int8_t, 640> z{};
    alignas(16) std::array<uint8_t, 3 * 64> img{};
    std::array<int8_t, 25> w{};
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t i = 0; i < z.size(); ++i) {
        random = random * 1664525u + 1013904223u;
        z[i] = pass ? int8_t{-128} : static_cast<int8_t>(random >> 24);
      }
      for (size_t i = 0; i < w.size(); ++i) {
        random = random * 1664525u + 1013904223u;
        w[i] = pass ? int8_t{-128} : static_cast<int8_t>(random >> 24);
      }
      for (int gap : {8, 24}) {
        for (int runs = 1; runs <= 3; ++runs) {
          const int8_t* rows[5];
          for (int i = 0; i < 5; ++i) rows[i] = z.data() + gap * i;
          img.fill(0x5A);
          dw5_pairs_s3(rows, 104, w.data(), img.data(), runs);
          for (int u = 0; u < runs; ++u) {
            for (int l = 0; l < 16; ++l) {
              int32_t want = 0;
              for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                  want += rows[i][104 * j + 16 * u + l] * w[5 * i + j];
                }
              }
              const uint8_t* half = img.data() + 64 * u + 32 * (l / 8);
              uint32_t v = 0;
              for (int b = 0; b < 20; ++b) {
                const int at = 20 * (l % 8) + b;
                v |= static_cast<uint32_t>((half[at / 8] >> (at % 8)) & 1u) << b;
              }
              CHECK(static_cast<int32_t>(v << 12) >> 12 == want);
            }
          }
        }
      }
    }
  }
  std::printf("portable kernels: %d failures\n", failures);
  return failures ? 1 : 0;
}
