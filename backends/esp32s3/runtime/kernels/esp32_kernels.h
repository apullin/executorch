/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Firmware links the .S implementations; the host links esp32_reference_kernels.c.
// Both integer kernels consume complete 16-byte blocks only. Callers handle
// alignment and tails. The dot returns the low 32 bits of the signed sum.
int32_t dot_i8_s3(const int8_t* a, const int8_t* b, int n);
// dot_i8_s3 for `count` rows against one weight row: out[t] = the dot of
// rows + t * stride (stride a multiple of 16) with w (dot_i8_rows_s3.S).
void dot_i8_rows_s3(const int8_t* rows, int stride, const int8_t* w, int n,
                    int32_t* out, int count);
void unpack_i4_s3(const uint8_t* packed, int8_t* out, int n_bytes);
// One raw sum per scale group: the stored (+8-biased) codes of one packed row
// against activations deinterleaved per 32 (16 even, then 16 odd). Low 32 bits.
// group_bytes is a nonzero multiple of 16; the caller subtracts 8 * sum(x).
void dot_i4_groups_s3(const uint8_t* w, const int8_t* xd, int32_t* sums,
                      int n_groups, int group_bytes);
// A 3x3 depthwise convolution's multiply-adds for 16 channels over `count`
// output pixels of a channels-last plane (taps r0 + i * pitch + j * 16, the
// next pixel `step` on; w = 9 taps x 16 weights), each pixel's 16 raw sums left
// as a 64-byte QACC image (QACC_L's 160 bits at +0, QACC_H's at +32; 20-bit
// lanes). Every pointer and stride a multiple of 16 (dw33_qacc16_s3.S).
void dw33_qacc16_s3(const int8_t* r0, int pitch, int step, const int8_t* w,
                    uint8_t* img, int count);
// One channel of a 5x5 (3x3) depthwise convolution, 16 output pixels of one
// output row a vector: for each of `chunks` runs, lane l (pixel l of the run,
// the run 16 bytes on from the last) sums, over window rows i < 5 (3), its A
// taps rows[i][l + j] (j < 3, or 2) and its B taps rows[i][boff + l + j]
// (j < 2, or 1) times w (per window row its A taps' weights, then its B
// taps'), as a 64-byte QACC image at img + 64 * run, dw33_qacc16_s3's layout,
// lane l at lane l. img 16-byte aligned; every tap's 16 codes are read as
// two aligned blocks, [a & ~15, (a & ~15) + 32) for a tap address a
// (dw_plane_s3.S).
void dw5_plane_s3(const int8_t* const* rows, int boff, const int8_t* w,
                  uint8_t* img, int chunks);
void dw3_plane_s3(const int8_t* const* rows, int boff, const int8_t* w,
                  uint8_t* img, int chunks);
// The same for a column stride of 1 only (boff = 3, or 2: the B taps right
// after the A taps), without the step between them.
void dw5_plane1_s3(const int8_t* const* rows, int boff, const int8_t* w,
                   uint8_t* img, int chunks);
void dw3_plane1_s3(const int8_t* const* rows, int boff, const int8_t* w,
                   uint8_t* img, int chunks);
// A 5x5 at a stride of 1 with output rows of at most 8 pixels, two output
// rows a vector: for each of `chunks` runs (16 bytes on from the last), lane
// l sums, over window rows i and columns j < 5, the byte at rows[i] +
// j * tstride + 16 * run + l times w[5 i + j], as dw5_plane_s3's image. Lanes
// 0-7 are one output row's pixels, 8-15 the next row's, when rows[i] points
// at 8-byte padded-row windows (dw_plane_s3.S). rows[i] and tstride multiples
// of 8, img of 16.
void dw5_pairs_s3(const int8_t* const* rows, int tstride, const int8_t* w,
                  uint8_t* img, int chunks);
// The static int8 conv path's integer epilogue (codes out) for one QACC lane
// over `count` such images, 64 bytes apart from p (the word holding the
// lane's bit 0): raw = bits [shift, shift + 20) of the word pair at p, sign-
// extended; Y = raw * g + (c_hi * 2^32 + c_lo); out[u] = clamp(Y >> 32, qlo,
// qhi), and the image's address into slow[] where Y's low word (signed) is
// within guard of 0. Returns the slow[] entries (qacc_lane_codes_s3.S).
typedef struct {  // fixed offsets: qacc_lane_codes_s3.S reads each field at +0..+24
  int32_t g;
  uint32_t c_lo;
  int32_t c_hi;
  uint32_t guard;
  int32_t qlo, qhi;
  uint32_t shift;
} Esp32LaneEpi;
int qacc_lane_codes_s3(const uint8_t* p, int count, int8_t* out,
                       const Esp32LaneEpi* t, const uint8_t** slow);
// The static int8 conv path's dense columns (codes out) for one output
// channel over `count` rows 16 * chunks apart: raw = dot_i8_s3(row u, w,
// 16 * chunks), then qacc_lane_codes_s3's epilogue on it into out[u], &out[u]
// into slow[] near a boundary. The weights stay in registers: 1 <= chunks <= 7.
// Returns the slow[] entries (dot_rows_codes_s3.S).
typedef struct {  // fixed offsets: dot_rows_codes_s3.S, dot_rows_long_s3.S read +0..+24
  int32_t g;
  uint32_t c_lo;
  int32_t c_hi;
  uint32_t guard;
  int32_t qlo, qhi;
  uint32_t chunks;
} Esp32RowEpi;
int dot_rows_codes_s3(const int8_t* rows, const int8_t* w, int8_t* out,
                      int count, const Esp32RowEpi* t, int8_t** slow);
// dot_rows_codes_s3 for rows too long for its registers, chunks >= 8 (k_pad
// past 112): the first six weight chunks held, the rest streamed beside each
// row's. The same arithmetic, contract and return (dot_rows_long_s3.S).
int dot_rows_long_s3(const int8_t* rows, const int8_t* w, int8_t* out,
                     int count, const Esp32RowEpi* t, int8_t** slow);
// 16 channel planes into channels-last vectors: dst[p * stride + j] =
// src[j * plane + p] for p in [0, count), j in [0, 16). dst and stride
// multiples of 8. For each 16-pixel tile t it reads [a & ~15, a + 32), a =
// src + j * plane + 16 * t: up to 15 bytes before a (the aligned block
// holding it, always mapped on the S3) and up to 32 past it, which the
// caller keeps inside its buffers (transpose16_s3.S).
void transpose16_s3(const int8_t* src, int plane, int8_t* dst, int stride,
                    int count);

// Floating reference preserves the assembly's four fused accumulators and two
// Newton steps. Its correctly rounded seed replaces the unavailable rsqrt0.s.
// Returns 1 when the row was written (d < 1 writes nothing and returns 1), and
// 0, having written nothing, when mean(x^2) + eps is not a positive normal float.
int rmsnorm_f32_s3(float* out, const float* x, const float* w, int d,
                   float inv_d, float eps);

#if defined(ESP32_HOST_BUILD)
// Host only: calls into each C reference entry point since start-up (the
// conv path's kernels too: their outputs equal the C loops they replace, so
// only these counts show they ran).
typedef struct {
  uint64_t dot_i8, unpack_i4, dot_i4_groups, rmsnorm;
  uint64_t qacc_lane_codes, dot_rows_codes, transpose16;
  uint64_t dw_plane;  // dw5_plane_s3 and dw3_plane_s3
} Esp32ReferenceKernelCalls;
Esp32ReferenceKernelCalls esp32_reference_kernel_calls(void);
// Host only: calls into dot_rows_long_s3's C reference since start-up.
uint64_t esp32_reference_dot_rows_long_calls(void);
#endif

#ifdef __cplusplus
}
#endif
