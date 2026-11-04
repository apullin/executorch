/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Portable implementations of the ESP32-S3 kernel entry points. Integer
// results and complete-block/tail behavior match the assembly exactly.
#include "esp32_kernels.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

// Calls per entry point, so a host run can show which kernels a model
// reached. Relaxed atomic adds: the host's split worker calls them from a
// thread of its own while the caller does. The firmware's kernels count
// nothing.
static Esp32ReferenceKernelCalls g_calls;
#define COUNT_CALL(field) __atomic_fetch_add(&g_calls.field, 1, __ATOMIC_RELAXED)

Esp32ReferenceKernelCalls esp32_reference_kernel_calls(void) {
  Esp32ReferenceKernelCalls calls;
  calls.dot_i8 = __atomic_load_n(&g_calls.dot_i8, __ATOMIC_RELAXED);
  calls.unpack_i4 = __atomic_load_n(&g_calls.unpack_i4, __ATOMIC_RELAXED);
  calls.dot_i4_groups = __atomic_load_n(&g_calls.dot_i4_groups, __ATOMIC_RELAXED);
  calls.rmsnorm = __atomic_load_n(&g_calls.rmsnorm, __ATOMIC_RELAXED);
  calls.qacc_lane_codes =
      __atomic_load_n(&g_calls.qacc_lane_codes, __ATOMIC_RELAXED);
  calls.dot_rows_codes =
      __atomic_load_n(&g_calls.dot_rows_codes, __ATOMIC_RELAXED);
  calls.transpose16 = __atomic_load_n(&g_calls.transpose16, __ATOMIC_RELAXED);
  calls.dw_plane = __atomic_load_n(&g_calls.dw_plane, __ATOMIC_RELAXED);
  return calls;
}

void dot_i8_rows_s3(const int8_t* rows, int stride, const int8_t* w, int n,
                    int32_t* out, int count) {
  for (int t = 0; t < count; ++t) {
    out[t] = dot_i8_s3(rows + (ptrdiff_t)t * stride, w, n);
  }
}

/* dw33_qacc16_s3's arithmetic: the exact sums, packed as 20-bit lanes in
 * channel order (lane c holds channel c), QACC_L's 160 bits for channels 0-7
 * at +0 and QACC_H's for 8-15 at +32 of each 64-byte image. */
void dw33_qacc16_s3(const int8_t* r0, int pitch, int step, const int8_t* w,
                    uint8_t* img, int count) {
  for (int u = 0; u < count; ++u) {
    int32_t acc[16] = {0};
    const int8_t* p = r0 + (ptrdiff_t)u * step;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int c = 0; c < 16; ++c) {
          acc[c] += (int32_t)p[i * pitch + j * 16 + c] *
                    (int32_t)w[(i * 3 + j) * 16 + c];
        }
      }
    }
    uint8_t* out = img + (ptrdiff_t)u * 64;
    memset(out, 0, 64);
    for (int h = 0; h < 2; ++h) {
      uint32_t words[5] = {0, 0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        const uint32_t v = (uint32_t)acc[8 * h + i] & 0xFFFFFu;
        const int bit = 20 * i;
        words[bit >> 5] |= v << (bit & 31);
        if ((bit & 31) > 12) {
          words[(bit >> 5) + 1] |= v >> (32 - (bit & 31));
        }
      }
      memcpy(out + 32 * h, words, sizeof(words));
    }
  }
}

/* dw5_plane_s3's and dw3_plane_s3's arithmetic: per run of 16 pixels, the
 * exact sums over the window rows' A and B taps, packed as dw33_qacc16_s3's
 * reference packs them (lane l holds pixel l). */
/* The assembly's alignment contract, which the C here would not need: a
 * call that breaks it computes other sums on the chip (an ee.vld or ee.st
 * address drops its low bits), so the host stops instead of agreeing. */
static void contract(int ok, const char* what) {
  if (!ok) {
    fprintf(stderr, "esp32 reference kernels: %s\n", what);
    abort();
  }
}

static void dw_plane(const int8_t* const* rows, int kh, int ka, int kb,
                     int boff, const int8_t* w, uint8_t* img, int chunks) {
  contract(((uintptr_t)img & 15) == 0, "dw*_plane_s3: img not 16-byte aligned");
  for (int u = 0; u < chunks; ++u) {
    int32_t acc[16] = {0};
    for (int i = 0; i < kh; ++i) {
      const int8_t* a = rows[i] + 16 * u;
      const int8_t* b = a + boff;
      const int8_t* wr = w + i * (ka + kb);
      for (int l = 0; l < 16; ++l) {
        for (int j = 0; j < ka; ++j) {
          acc[l] += (int32_t)a[l + j] * (int32_t)wr[j];
        }
        for (int j = 0; j < kb; ++j) {
          acc[l] += (int32_t)b[l + j] * (int32_t)wr[ka + j];
        }
      }
    }
    uint8_t* out = img + (ptrdiff_t)u * 64;
    memset(out, 0, 64);
    for (int h = 0; h < 2; ++h) {
      uint32_t words[5] = {0, 0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        const uint32_t v = (uint32_t)acc[8 * h + i] & 0xFFFFFu;
        const int bit = 20 * i;
        words[bit >> 5] |= v << (bit & 31);
        if ((bit & 31) > 12) {
          words[(bit >> 5) + 1] |= v >> (32 - (bit & 31));
        }
      }
      memcpy(out + 32 * h, words, sizeof(words));
    }
  }
}

void dw5_plane_s3(const int8_t* const* rows, int boff, const int8_t* w,
                  uint8_t* img, int chunks) {
  COUNT_CALL(dw_plane);
  dw_plane(rows, 5, 3, 2, boff, w, img, chunks);
}

void dw3_plane_s3(const int8_t* const* rows, int boff, const int8_t* w,
                  uint8_t* img, int chunks) {
  COUNT_CALL(dw_plane);
  dw_plane(rows, 3, 2, 1, boff, w, img, chunks);
}

/* The stride-1 entries: the B taps right after the A taps whatever boff
 * says (the assembly never reads it). */
void dw5_plane1_s3(const int8_t* const* rows, int boff, const int8_t* w,
                   uint8_t* img, int chunks) {
  (void)boff;
  COUNT_CALL(dw_plane);
  dw_plane(rows, 5, 3, 2, 3, w, img, chunks);
}

void dw3_plane1_s3(const int8_t* const* rows, int boff, const int8_t* w,
                   uint8_t* img, int chunks) {
  (void)boff;
  COUNT_CALL(dw_plane);
  dw_plane(rows, 3, 2, 1, 2, w, img, chunks);
}

/* dw5_pairs_s3's arithmetic: per run, lane l sums the bytes at rows[i] +
 * j * tstride + 16 u + l times w[5 i + j], packed as dw33's reference packs
 * them. */
void dw5_pairs_s3(const int8_t* const* rows, int tstride, const int8_t* w,
                  uint8_t* img, int chunks) {
  COUNT_CALL(dw_plane);
  contract(((uintptr_t)img & 15) == 0, "dw5_pairs_s3: img not 16-byte aligned");
  contract((tstride & 7) == 0, "dw5_pairs_s3: tstride not a multiple of 8");
  for (int i = 0; i < 5; ++i) {
    contract(((uintptr_t)rows[i] & 7) == 0,
             "dw5_pairs_s3: a window row not 8-byte aligned");
  }
  for (int u = 0; u < chunks; ++u) {
    int32_t acc[16] = {0};
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        const int8_t* a = rows[i] + (ptrdiff_t)j * tstride + 16 * u;
        for (int l = 0; l < 16; ++l) {
          acc[l] += (int32_t)a[l] * (int32_t)w[5 * i + j];
        }
      }
    }
    uint8_t* out = img + (ptrdiff_t)u * 64;
    memset(out, 0, 64);
    for (int h = 0; h < 2; ++h) {
      uint32_t words[5] = {0, 0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        const uint32_t v = (uint32_t)acc[8 * h + i] & 0xFFFFFu;
        const int bit = 20 * i;
        words[bit >> 5] |= v << (bit & 31);
        if ((bit & 31) > 12) {
          words[(bit >> 5) + 1] |= v >> (32 - (bit & 31));
        }
      }
      memcpy(out + 32 * h, words, sizeof(words));
    }
  }
}

/* qacc_lane_codes_s3's arithmetic, operation for operation. */
int qacc_lane_codes_s3(const uint8_t* p, int count, int8_t* out,
                       const Esp32LaneEpi* t, const uint8_t** slow) {
  COUNT_CALL(qacc_lane_codes);
  const uint8_t** next = slow;
  const int64_t c = (int64_t)(((uint64_t)(uint32_t)t->c_hi << 32) | t->c_lo);
  for (int u = 0; u < count; ++u, p += 64) {
    uint32_t w0, w1;
    memcpy(&w0, p, sizeof(w0));
    memcpy(&w1, p + 4, sizeof(w1));
    const uint64_t pair = ((uint64_t)w1 << 32) | w0;
    const uint32_t v = (uint32_t)(pair >> (t->shift & 31u));
    const int32_t raw = (int32_t)(v << 12) >> 12;
    const int64_t y = (int64_t)raw * t->g + c;
    const uint32_t lo = (uint32_t)y;
    const uint32_t d = (int32_t)lo < 0 ? 0u - lo : lo;
    if (d < t->guard) {
      *next++ = p;
    }
    int32_t q = (int32_t)(y >> 32);
    q = q < t->qlo ? t->qlo : q;
    q = q > t->qhi ? t->qhi : q;
    out[u] = (int8_t)q;
  }
  return (int)(next - slow);
}

/* dot_rows_codes_s3's arithmetic, operation for operation. */
int dot_rows_codes_s3(const int8_t* rows, const int8_t* w, int8_t* out,
                      int count, const Esp32RowEpi* t, int8_t** slow) {
  COUNT_CALL(dot_rows_codes);
  int8_t** next = slow;
  if (t->chunks < 1 || t->chunks > 7) {
    return 0;
  }
  const int n = (int)t->chunks * 16;
  const int64_t c = (int64_t)(((uint64_t)(uint32_t)t->c_hi << 32) | t->c_lo);
  for (int u = 0; u < count; ++u) {
    uint32_t sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += (uint32_t)((int32_t)rows[(ptrdiff_t)u * n + i] * (int32_t)w[i]);
    }
    int32_t raw;
    memcpy(&raw, &sum, sizeof(raw));
    const int64_t y = (int64_t)raw * t->g + c;
    const uint32_t lo = (uint32_t)y;
    const uint32_t d = (int32_t)lo < 0 ? 0u - lo : lo;
    if (d < t->guard) {
      *next++ = out + u;
    }
    int32_t q = (int32_t)(y >> 32);
    q = q < t->qlo ? t->qlo : q;
    q = q > t->qhi ? t->qhi : q;
    out[u] = (int8_t)q;
  }
  return (int)(next - slow);
}

/* dot_rows_long_s3's arithmetic: dot_rows_codes_s3's for chunks >= 8. */
static uint64_t g_dot_rows_long_calls;

uint64_t esp32_reference_dot_rows_long_calls(void) {
  return __atomic_load_n(&g_dot_rows_long_calls, __ATOMIC_RELAXED);
}

int dot_rows_long_s3(const int8_t* rows, const int8_t* w, int8_t* out,
                     int count, const Esp32RowEpi* t, int8_t** slow) {
  __atomic_fetch_add(&g_dot_rows_long_calls, 1, __ATOMIC_RELAXED);
  int8_t** next = slow;
  if (t->chunks < 8) {
    return 0;
  }
  const int n = (int)t->chunks * 16;
  const int64_t c = (int64_t)(((uint64_t)(uint32_t)t->c_hi << 32) | t->c_lo);
  for (int u = 0; u < count; ++u) {
    uint32_t sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += (uint32_t)((int32_t)rows[(ptrdiff_t)u * n + i] * (int32_t)w[i]);
    }
    int32_t raw;
    memcpy(&raw, &sum, sizeof(raw));
    const int64_t y = (int64_t)raw * t->g + c;
    const uint32_t lo = (uint32_t)y;
    const uint32_t d = (int32_t)lo < 0 ? 0u - lo : lo;
    if (d < t->guard) {
      *next++ = out + u;
    }
    int32_t q = (int32_t)(y >> 32);
    q = q < t->qlo ? t->qlo : q;
    q = q > t->qhi ? t->qhi : q;
    out[u] = (int8_t)q;
  }
  return (int)(next - slow);
}

/* transpose16_s3's copy. */
void transpose16_s3(const int8_t* src, int plane, int8_t* dst, int stride,
                    int count) {
  COUNT_CALL(transpose16);
  for (int p = 0; p < count; ++p) {
    for (int j = 0; j < 16; ++j) {
      dst[(ptrdiff_t)p * stride + j] = src[(ptrdiff_t)j * plane + p];
    }
  }
}

int32_t dot_i8_s3(const int8_t* a, const int8_t* b, int n) {
  COUNT_CALL(dot_i8);
  uint32_t sum = 0;
  const int end = n > 0 ? n - n % 16 : 0;
  for (int i = 0; i < end; ++i) {
    // Unsigned accumulation defines the low-word wrap without signed C UB.
    sum += (uint32_t)((int32_t)a[i] * (int32_t)b[i]);
  }
  int32_t result;
  memcpy(&result, &sum, sizeof(result));
  return result;
}

void unpack_i4_s3(const uint8_t* packed, int8_t* out, int n_bytes) {
  COUNT_CALL(unpack_i4);
  const int end = n_bytes > 0 ? n_bytes - n_bytes % 16 : 0;
  for (int i = 0; i < end; ++i) {
    out[2 * i] = (int8_t)((packed[i] & 15) - 8);
    out[2 * i + 1] = (int8_t)((packed[i] >> 4) - 8);
  }
}

void dot_i4_groups_s3(const uint8_t* w, const int8_t* xd, int32_t* sums,
                      int n_groups, int group_bytes) {
  COUNT_CALL(dot_i4_groups);
  // Like the assembly: whole 16-byte steps only, and both pointers run on
  // from one group into the next (16 packed bytes against 32 activations).
  const int steps = group_bytes > 0 ? group_bytes / 16 : 0;
  for (int g = 0; g < n_groups; ++g) {
    uint32_t sum = 0;
    for (int step = 0; step < steps; ++step, w += 16, xd += 32) {
      for (int t = 0; t < 16; ++t) {
        // Stored codes, still biased: low nibble 2t (even), high 2t+1 (odd).
        sum += (uint32_t)((int32_t)(w[t] & 15) * (int32_t)xd[t]);
        sum += (uint32_t)((int32_t)(w[t] >> 4) * (int32_t)xd[16 + t]);
      }
    }
    int32_t result;
    memcpy(&result, &sum, sizeof(result));
    sums[g] = result;
  }
}

int rmsnorm_f32_s3(float* out, const float* x, const float* w, int d,
                   float inv_d, float eps) {
  COUNT_CALL(rmsnorm);
  if (d < 1) {
    return 1;
  }
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int end = d - d % 4;
  for (int i = 0; i < end; i += 4) {
    for (int lane = 0; lane < 4; ++lane) {
      sums[lane] = fmaf(x[i + lane], x[i + lane], sums[lane]);
    }
  }
  float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  for (int i = end; i < d; ++i) {
    sum = fmaf(x[i], x[i], sum);
  }
  const float mean = sum * inv_d;
  const float t = mean + eps;
  // The assembly's domain check, before its seed: ordered compares, so NaN is
  // refused as well as zero, subnormal and infinite t.
  if (!(t >= FLT_MIN && t <= FLT_MAX)) {
    return 0;
  }
  float scale = (float)(1.0 / sqrt((double)t));
  const float half_t = 0.5f * t;
  for (int step = 0; step < 2; ++step) {
    const float square = scale * scale;
    const float correction = fmaf(-half_t, square, 1.5f);
    scale = scale * correction;
  }
  for (int i = 0; i < d; ++i) {
    const float item = w ? w[i] * x[i] : x[i];
    out[i] = item * scale;
  }
  return 1;
}
