/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3: the static int8 conv path's integer epilogue for one QACC lane
 * over a run of raw QACC images (esp32_conv.cpp, the PIE depthwise path).
 *
 *   int qacc_lane_codes_s3(const uint8_t *p, int count, int8_t *out,
 *                          const Esp32LaneEpi *t, const uint8_t **slow);
 *
 * For u in [0, count), the image at p + 64 * u (p points at the 32-bit word
 * holding the lane's bit 0, lane_at in esp32_conv.cpp):
 *
 *   raw = bits [shift, shift + 20) of the word pair at p, sign-extended
 *   Y   = raw * g + (c_hi * 2^32 + c_lo)            (64-bit, exact)
 *   out[u] = clamp(Y >> 32, qlo, qhi)
 *
 * and where Y's low word, as a signed int32, is within `guard` of 0 (|f| <
 * guard), the image's address goes to slow[] for the caller's float sequence.
 * Returns the number of slow[] entries. This is esp32_conv.cpp's lane_fixed
 * (codes out) operation for operation; esp32_reference_kernels.c has it in C
 * for the host, and pie_dw_init holds the two equal on the chip.
 *
 * Two loops: a lane inside its first word (shift <= 12) takes one load and a
 * shift (16 instructions an element), a lane running into the second word
 * two loads and a funnel shift (17).
 *
 * Contract, not checked here: p 4-byte aligned, the word after each image's
 * lane word readable when shift > 12, 0 <= shift <= 31, count >= 0, slow[]
 * room for count entries.
 *
 * Esp32LaneEpi: g at +0, c_lo +4, c_hi +8, guard +12, qlo +16, qhi +20,
 * shift +24 (esp32_kernels.h).
 */

	.text
	.align	4
	.global	qacc_lane_codes_s3
	.type	qacc_lane_codes_s3, @function

/* windowed ABI: a2 p, a3 count, a4 out, a5 terms, a6 slow */
qacc_lane_codes_s3:
	entry	a1, 32
	l32i	a8, a5, 0		/* g */
	l32i	a9, a5, 4		/* c_lo */
	l32i	a10, a5, 8		/* c_hi */
	l32i	a11, a5, 12		/* guard */
	l32i	a12, a5, 16		/* qlo */
	l32i	a13, a5, 20		/* qhi */
	l32i	a14, a5, 24		/* shift */
	mov	a15, a6			/* the next slow[] entry */
	ssr	a14			/* SAR: the lane's bit position, for src/srl */
	movi	a7, 12
	blt	a7, a14, .Lspan	/* shift > 12: the lane spans two words */
	/* The lane inside its first word. */
	loopgtz	a3, .Lend1
	l32i	a5, a2, 0		/* the lane's word */
	addi	a2, a2, 64		/* the next image (and a slot for the load) */
	srl	a5, a5		/* w0 >> shift */
	sext	a5, a5, 19		/* raw */
	mull	a7, a5, a8
	mulsh	a5, a5, a8		/* raw * g, low and high words */
	add	a7, a7, a9		/* Y's low word */
	saltu	a14, a7, a9		/* its carry */
	add	a5, a5, a10
	add	a5, a5, a14		/* Y's high word: the code before the clamp */
	abs	a7, a7			/* the low word's distance from a boundary */
	max	a5, a5, a12
	min	a5, a5, a13
	bltu	a7, a11, .Lslow1
.Lstore1:
	s8i	a5, a4, 0
	addi	a4, a4, 1
.Lend1:
	j	.Ldone
.Lspan:
	/* The lane running into the second word. */
	loopgtz	a3, .Lend2
	l32i	a7, a2, 4		/* the second word */
	l32i	a5, a2, 0		/* the lane's word */
	addi	a2, a2, 64		/* the next image (and a slot for the load) */
	src	a5, a7, a5		/* (w1:w0) >> shift */
	sext	a5, a5, 19		/* raw */
	mull	a7, a5, a8
	mulsh	a5, a5, a8		/* raw * g, low and high words */
	add	a7, a7, a9		/* Y's low word */
	saltu	a14, a7, a9		/* its carry */
	add	a5, a5, a10
	add	a5, a5, a14		/* Y's high word: the code before the clamp */
	abs	a7, a7			/* the low word's distance from a boundary */
	max	a5, a5, a12
	min	a5, a5, a13
	bltu	a7, a11, .Lslow2
.Lstore2:
	s8i	a5, a4, 0
	addi	a4, a4, 1
.Lend2:
.Ldone:
	sub	a2, a15, a6
	srai	a2, a2, 2
	retw.n
.Lslow1:
	/* The image's address (a2 has moved on by one image). */
	addi	a7, a2, -64
	s32i	a7, a15, 0
	addi	a15, a15, 4
	j	.Lstore1
.Lslow2:
	addi	a7, a2, -64
	s32i	a7, a15, 0
	addi	a15, a15, 4
	j	.Lstore2

	.size	qacc_lane_codes_s3, . - qacc_lane_codes_s3
