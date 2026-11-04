/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 PIE: the static int8 conv path's dense columns, codes out, for
 * one output channel over a block of rows, the int8 dot and the integer
 * epilogue in one loop (esp32_conv.cpp, columns).
 *
 *   int dot_rows_codes_s3(const int8_t *rows, const int8_t *w, int8_t *out,
 *                         int count, const Esp32RowEpi *t, int8_t **slow);
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
 * number of slow[] entries. This is dot_i8_rows_s3 then finish_fixed (codes
 * out) operation for operation; esp32_reference_kernels.c has it in C for the
 * host, and esp32_conv.cpp's start-up check holds the two equal on the chip.
 *
 * The weights stay in q1..q<chunks> for the whole block (chunks in [1, 7]:
 * k_pad <= 112), so a row costs its chunks' fused multiply-and-load and the
 * epilogue: one zero-overhead loop over the rows, a variant per chunk count.
 *
 * Contract, not checked here: rows and w 16-byte aligned (a misaligned
 * ee.vld.128 silently truncates the address), 1 <= chunks <= 7, count >= 0,
 * slow[] room for count entries; the dot within int32 (kMaxDotLen).
 *
 * Esp32RowEpi: g at +0, c_lo +4, c_hi +8, guard +12, qlo +16, qhi +20,
 * chunks +24 (esp32_kernels.h). q0-q7, ACCX and the address registers here
 * are caller-saved (abikind=0); the calling task is pinned to its core.
 */

	.text
	.align	4
	.global	dot_rows_codes_s3
	.type	dot_rows_codes_s3, @function

/* windowed ABI: a2 rows, a3 w, a4 out, a5 count, a6 terms, a7 slow */
dot_rows_codes_s3:
	entry	a1, 32
	l32i	a8, a6, 0		/* g */
	l32i	a9, a6, 4		/* c_lo */
	l32i	a10, a6, 8		/* c_hi */
	l32i	a11, a6, 12		/* guard */
	l32i	a12, a6, 16		/* qlo */
	l32i	a13, a6, 20		/* qhi */
	l32i	a14, a6, 24		/* chunks */
	mov	a15, a7			/* the next slow[] entry */
	addi	a4, a4, -1		/* out, one before: advanced before each store */

	beqi	a14, 1, .Lw1
	beqi	a14, 2, .Lw2
	beqi	a14, 3, .Lw3
	beqi	a14, 4, .Lw4
	beqi	a14, 5, .Lw5
	beqi	a14, 6, .Lw6
	beqi	a14, 7, .Lw7
	j	.Ldone			/* out of contract: nothing written */

/* 1 chunk(s): k_pad = 16 */
.Lw1:
	ee.vld.128.ip	q1, a3, 16
	loopgtz	a5, .Lend1
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx	q0, q1
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow1
.Lst1:
	s8i	a6, a4, 0
.Lend1:
	j	.Ldone
.Lslow1:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst1

/* 2 chunk(s): k_pad = 32 */
.Lw2:
	ee.vld.128.ip	q1, a3, 16
	ee.vld.128.ip	q2, a3, 16
	loopgtz	a5, .Lend2
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vmulas.s8.accx	q0, q2
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow2
.Lst2:
	s8i	a6, a4, 0
.Lend2:
	j	.Ldone
.Lslow2:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst2

/* 3 chunk(s): k_pad = 48 */
.Lw3:
	ee.vld.128.ip	q1, a3, 16
	ee.vld.128.ip	q2, a3, 16
	ee.vld.128.ip	q3, a3, 16
	loopgtz	a5, .Lend3
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q2
	ee.vmulas.s8.accx	q0, q3
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow3
.Lst3:
	s8i	a6, a4, 0
.Lend3:
	j	.Ldone
.Lslow3:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst3

/* 4 chunk(s): k_pad = 64 */
.Lw4:
	ee.vld.128.ip	q1, a3, 16
	ee.vld.128.ip	q2, a3, 16
	ee.vld.128.ip	q3, a3, 16
	ee.vld.128.ip	q4, a3, 16
	loopgtz	a5, .Lend4
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q2
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q3
	ee.vmulas.s8.accx	q0, q4
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow4
.Lst4:
	s8i	a6, a4, 0
.Lend4:
	j	.Ldone
.Lslow4:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst4

/* 5 chunk(s): k_pad = 80 */
.Lw5:
	ee.vld.128.ip	q1, a3, 16
	ee.vld.128.ip	q2, a3, 16
	ee.vld.128.ip	q3, a3, 16
	ee.vld.128.ip	q4, a3, 16
	ee.vld.128.ip	q5, a3, 16
	loopgtz	a5, .Lend5
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q2
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q3
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q4
	ee.vmulas.s8.accx	q0, q5
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow5
.Lst5:
	s8i	a6, a4, 0
.Lend5:
	j	.Ldone
.Lslow5:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst5

/* 6 chunk(s): k_pad = 96 */
.Lw6:
	ee.vld.128.ip	q1, a3, 16
	ee.vld.128.ip	q2, a3, 16
	ee.vld.128.ip	q3, a3, 16
	ee.vld.128.ip	q4, a3, 16
	ee.vld.128.ip	q5, a3, 16
	ee.vld.128.ip	q6, a3, 16
	loopgtz	a5, .Lend6
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q2
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q3
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q4
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q5
	ee.vmulas.s8.accx	q0, q6
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow6
.Lst6:
	s8i	a6, a4, 0
.Lend6:
	j	.Ldone
.Lslow6:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst6

/* 7 chunk(s): k_pad = 112 */
.Lw7:
	ee.vld.128.ip	q1, a3, 16
	ee.vld.128.ip	q2, a3, 16
	ee.vld.128.ip	q3, a3, 16
	ee.vld.128.ip	q4, a3, 16
	ee.vld.128.ip	q5, a3, 16
	ee.vld.128.ip	q6, a3, 16
	ee.vld.128.ip	q7, a3, 16
	loopgtz	a5, .Lend7
	ee.zero.accx
	ee.vld.128.ip	q0, a2, 16
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q2
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q3
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q4
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q5
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q6
	ee.vmulas.s8.accx	q0, q7
	addi	a4, a4, 1		/* the next code (and the first slot ACCX needs) */
	nop			/* the second */
	rur.accx_0	a6		/* raw */
	mull	a3, a6, a8
	mulsh	a6, a6, a8		/* raw * g, low and high words */
	add	a3, a3, a9		/* Y's low word */
	saltu	a14, a3, a9		/* its carry */
	add	a6, a6, a10
	add	a6, a6, a14		/* Y's high word: the code before the clamp */
	abs	a3, a3		/* the low word's distance from a boundary */
	max	a6, a6, a12
	min	a6, a6, a13
	bltu	a3, a11, .Lslow7
.Lst7:
	s8i	a6, a4, 0
.Lend7:
	j	.Ldone
.Lslow7:
	s32i	a4, a15, 0
	addi	a15, a15, 4
	j	.Lst7

.Ldone:
	sub	a2, a15, a7
	srai	a2, a2, 2
	retw.n

	.size	dot_rows_codes_s3, . - dot_rows_codes_s3
