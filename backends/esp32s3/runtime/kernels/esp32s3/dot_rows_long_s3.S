/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 PIE: the static int8 conv path's dense columns, codes out, for
 * one output channel over a block of rows whose k_pad is past what
 * dot_rows_codes_s3 holds in registers (esp32_conv.cpp, columns).
 *
 *   int dot_rows_long_s3(const int8_t *rows, const int8_t *w, int8_t *out,
 *                        int count, const Esp32RowEpi *t, int8_t **slow);
 *
 * For u in [0, count), row u at rows + u * 16 * chunks (k_pad = 16 * chunks
 * apart, as the row block holds them):
 *
 *   raw = dot(row u, w, k_pad)                (ACCX's low 32 bits: dot_i8_s3)
 *   Y   = raw * g + (c_hi * 2^32 + c_lo)       (64-bit, exact)
 *   out[u] = clamp(Y >> 32, qlo, qhi)
 *
 * and where Y's low word, as a signed int32, is within `guard` of 0 (|f| <
 * guard), &out[u] goes to slow[] for the caller's float sequence. Returns the
 * number of slow[] entries: dot_rows_codes_s3's contract for any chunks >= 8.
 * esp32_reference_kernels.c has it in C for the host, and esp32_conv.cpp's
 * start-up check holds the two equal on the chip.
 *
 * The weights' first six chunks stay in q2..q7 for the whole block; the rest
 * stream through q1 a chunk at a time beside the row's (a zero-overhead loop
 * of the fused multiply-and-load and one load). So a row costs one
 * instruction for each of its first six chunks and two for each later one,
 * then the epilogue -- where dot_i8_rows_s3 and finish take two a chunk, a
 * store and a reload of each raw sum, and the epilogue's own loop.
 *
 * Contract, not checked here: rows and w 16-byte aligned (a misaligned
 * ee.vld.128 silently truncates the address), chunks >= 8, count >= 0, slow[]
 * room for count entries; the dot within int32 (kMaxDotLen).
 *
 * Esp32RowEpi: g at +0, c_lo +4, c_hi +8, guard +12, qlo +16, qhi +20,
 * chunks +24 (esp32_kernels.h). q0-q7, ACCX and the address registers here
 * are caller-saved (abikind=0); the calling task is pinned to its core.
 * The slow[] base and the next entry live in the frame (the epilogue needs
 * every other register).
 */

	.text
	.align	4
	.global	dot_rows_long_s3
	.type	dot_rows_long_s3, @function

/* windowed ABI: a2 rows, a3 w, a4 out, a5 count, a6 terms, a7 slow */
dot_rows_long_s3:
	entry	a1, 48
	s32i	a7, a1, 0		/* slow[]'s base, for the count */
	s32i	a7, a1, 4		/* the next slow[] entry */
	l32i	a8, a6, 0		/* g */
	l32i	a9, a6, 4		/* c_lo */
	l32i	a10, a6, 8		/* c_hi */
	l32i	a11, a6, 12		/* guard */
	l32i	a12, a6, 16		/* qlo */
	l32i	a13, a6, 20		/* qhi */
	l32i	a14, a6, 24		/* chunks */
	addi	a4, a4, -1		/* out, one before: advanced before each store */
	addi	a14, a14, -7		/* the streamed loop's trips: chunks - 7 */
	blti	a14, 1, .Ldone		/* out of contract (chunks < 8): nothing written */
	blti	a5, 1, .Ldone
	/* The first six weight chunks, held. a3 then points at chunk 6. */
	ee.vld.128.ip	q2, a3, 16
	ee.vld.128.ip	q3, a3, 16
	ee.vld.128.ip	q4, a3, 16
	ee.vld.128.ip	q5, a3, 16
	ee.vld.128.ip	q6, a3, 16
	ee.vld.128.ip	q7, a3, 16
.Lrow:
	ee.zero.accx
	mov	a6, a3			/* the streamed weights, from chunk 6 */
	ee.vld.128.ip	q0, a2, 16	/* row chunk 0 */
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q2
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q3
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q4
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q5
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q6
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q7	/* q0 = row chunk 6 */
	ee.vld.128.ip	q1, a6, 16	/* weight chunk 6 */
	loopgtz	a14, .Lmac_end
	/* chunk j times weight j, row chunk j + 1 loaded; weight j + 1 */
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vld.128.ip	q1, a6, 16
.Lmac_end:
	ee.vmulas.s8.accx	q0, q1		/* the last chunk; a2 is the next row */
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	addi	a5, a5, -1		/* rows left (the second) */
	rur.accx_0	a6		/* raw */
	mull	a7, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a7, a7, a9		/* Y's low word */
	saltu	a15, a7, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a15		/* Y's high word: the code before the clamp */
	abs	a7, a7			/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a7, a11, .Lslow
.Lstore:
	s8i	a6, a4, 0
	bnez	a5, .Lrow
.Ldone:
	l32i	a2, a1, 4
	l32i	a3, a1, 0
	sub	a2, a2, a3
	srai	a2, a2, 2
	retw.n
.Lslow:
	l32i	a7, a1, 4
	s32i	a4, a7, 0
	addi	a7, a7, 4
	s32i	a7, a1, 4
	j	.Lstore

	.size	dot_rows_long_s3, . - dot_rows_long_s3
