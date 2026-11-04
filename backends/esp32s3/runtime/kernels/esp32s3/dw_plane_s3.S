/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 PIE: one channel of a 5x5 (or 3x3) depthwise convolution, 16
 * output pixels of one output row a vector, the sums left raw in QACC images.
 *
 *   void dw5_plane_s3(const int8_t *const *rows, int boff, const int8_t *w,
 *                     uint8_t *img, int chunks);
 *   void dw3_plane_s3(const int8_t *const *rows, int boff, const int8_t *w,
 *                     uint8_t *img, int chunks);
 *   dw5_plane1_s3, dw3_plane1_s3: the same for boff = 3 (2) only, a column
 *   stride of 1, without the step from the A taps to the B taps.
 *
 * The input is one channel's codes, row by row (esp32_conv.cpp's plane:
 * the padded window's rows, a column stride of 2 split into its even and odd
 * columns). rows[i] (i < 5, or 3) is window row i's first byte for the
 * output row's first pixel. Each window row's taps are its A taps at
 * rows[i] + j (j < 3, or 2) and its B taps at rows[i] + boff + j (j < 2, or
 * 1); the next pixel's taps are one byte on. A column stride of 1 has
 * boff = 3 (or 2), the taps in order; a stride of 2 has the even columns
 * (taps 0, 2, 4) as A and the odd ones (taps 1, 3) as B. w holds the 25 (or
 * 9) weights in that order: per window row, its A taps, then its B taps.
 *
 * For each of `chunks` runs of 16 pixels (the next run 16 bytes on), QACC is
 * zeroed, takes one ee.vmulas.s8.qacc a tap -- lane l the pixel's code at
 * the tap times the tap's weight, broadcast (16 signed 20-bit accumulators,
 * exact while |sum| < 2^19: 25 products are at most 25 * 128 * 128 =
 * 409,600) -- and is stored raw into a 64-byte image at img + 64 * chunk, as
 * dw33_qacc16_s3 stores it: QACC_L's 160 bits at +0 (low 128, then high 32 at
 * +16), QACC_H's at +32 (+48). Lane l holds pixel l (esp32_conv.cpp reads it
 * so, and its start-up check holds the lanes to the scalar sums).
 *
 * Each tap's 16 codes come by an unaligned load (EE.LD.128.USAR.XP, the next
 * aligned block, EE.SRC.Q); its weight by a broadcast load, fused into the
 * multiply-accumulate of the tap before (EE.VMULAS.S8.QACC.LDBC.INCP).
 *
 * Contract, not checked here: img 16-byte aligned; chunks >= 0; for every
 * tap address a the run reads, every byte of [a & ~15, (a & ~15) + 32)
 * readable -- up to 15 bytes before a (its aligned block) and up to 31 past
 * it. esp32_conv.cpp keeps both inside the plane it builds or its pad row.
 *
 * q0-q5, QACC, SAR_BYTE and the address registers here are caller-saved
 * (abikind=0), as for dot_i8_s3; the calling task is pinned to its core.
 */

	.text

/* One tap: its two aligned blocks and SAR_BYTE (the address registers step
 * one byte), the tap before's multiply-accumulate with this tap's weight
 * broadcast into \wn, then this tap's codes into \uc. The source operands
 * of each instruction were written at least two instructions before. */
.macro	TAP	wn, up, wp, uc
	ee.vld.128.xp	q1, a13, a14
	ee.ld.128.usar.xp	q0, a12, a14
	ee.vmulas.s8.qacc.ldbc.incp	\wn, a4, \up, \wp
	ee.src.q	\uc, q0, q1
.endm

/* A run's first tap: no tap before; its weight by a plain broadcast load. */
.macro	TAP0
	ee.vld.128.xp	q1, a13, a14
	ee.ld.128.usar.xp	q0, a12, a14
	ee.vldbc.8.ip	q2, a4, 1
	ee.src.q	q4, q0, q1
.endm

/* Window row \r's taps from its A part: the tap pointers. */
.macro	ROW	r
	mov	a12, \r
	addi	a13, \r, 16
.endm

/* From the A part's end to the B part. */
.macro	TO_B
	add	a12, a12, a15
	add	a13, a13, a15
.endm

/* The run's image, after the last multiply-accumulate (at least two
 * instructions before). */
.macro	STORE_IMAGE
	ee.st.qacc_l.l.128.ip	a5, 16
	ee.st.qacc_l.h.32.ip	a5, 16
	ee.st.qacc_h.l.128.ip	a5, 16
	ee.st.qacc_h.h.32.ip	a5, 16
.endm

/* From the A part's end to the B part, unless the B taps follow the A taps
 * (\jump 0: a column stride of 1, boff = 3 or 2). */
.macro	TO_B_IF	jump
.if \jump
	TO_B
.endif
.endm

/* windowed ABI: a2 rows, a3 boff, a4 w, a5 img, a6 chunks */
.macro	DW5	name, jump
	.align	4
	.global	\name
	.type	\name, @function
\name:
	entry	a1, 32
	blti	a6, 1, .Ldone_\name
	l32i	a7, a2, 0
	l32i	a8, a2, 4
	l32i	a9, a2, 8
	l32i	a10, a2, 12
	l32i	a11, a2, 16
	movi	a14, 1
	addi	a15, a3, -3		/* B's start less the A taps' end */
	/* A branch, not a zero-overhead loop: the body is past LOOP's reach. */
.Lchunk_\name:
	ee.zero.qacc
	/* Taps 0-24; the weight and codes registers alternate (q2/q3, q4/q5). */
	ROW	a7
	TAP0
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	TO_B_IF	\jump
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	ROW	a8
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	TO_B_IF	\jump
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	ROW	a9
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	TO_B_IF	\jump
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	ROW	a10
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	TO_B_IF	\jump
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	ROW	a11
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	TO_B_IF	\jump
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	/* The next run's rows around tap 24's multiply-accumulate: room after
	 * its codes are made and before QACC is read. */
	addi	a7, a7, 16
	addi	a8, a8, 16
	ee.vmulas.s8.qacc	q4, q2
	addi	a9, a9, 16
	addi	a10, a10, 16
	addi	a11, a11, 16
	addi	a4, a4, -25
	STORE_IMAGE
	addi	a6, a6, -1
	bnez	a6, .Lchunk_\name
.Ldone_\name:
	retw.n
	.size	\name, . - \name
.endm

/* windowed ABI: a2 rows, a3 boff, a4 w, a5 img, a6 chunks */
.macro	DW3	name, jump
	.align	4
	.global	\name
	.type	\name, @function
\name:
	entry	a1, 32
	blti	a6, 1, .Ldone_\name
	l32i	a7, a2, 0
	l32i	a8, a2, 4
	l32i	a9, a2, 8
	movi	a14, 1
	addi	a15, a3, -2		/* B's start less the A taps' end */
.Lchunk_\name:
	ee.zero.qacc
	/* Taps 0-8, as above. */
	ROW	a7
	TAP0
	TAP	q3, q4, q2, q5
	TO_B_IF	\jump
	TAP	q2, q5, q3, q4
	ROW	a8
	TAP	q3, q4, q2, q5
	TAP	q2, q5, q3, q4
	TO_B_IF	\jump
	TAP	q3, q4, q2, q5
	ROW	a9
	TAP	q2, q5, q3, q4
	TAP	q3, q4, q2, q5
	TO_B_IF	\jump
	TAP	q2, q5, q3, q4
	addi	a7, a7, 16
	addi	a8, a8, 16
	ee.vmulas.s8.qacc	q4, q2	/* tap 8 */
	addi	a9, a9, 16
	addi	a4, a4, -9
	STORE_IMAGE
	addi	a6, a6, -1
	bnez	a6, .Lchunk_\name
.Ldone_\name:
	retw.n
	.size	\name, . - \name
.endm

	DW5	dw5_plane_s3, 1
	DW3	dw3_plane_s3, 1
	/* A column stride of 1: B right after A (boff 3, or 2), no jump. */
	DW5	dw5_plane1_s3, 0
	DW3	dw3_plane1_s3, 0

/* Two output rows a vector, for a 5x5 at a stride of 1 whose output rows are
 * at most 8 pixels wide (esp32_conv.cpp's depthwise_pairs):
 *
 *   void dw5_pairs_s3(const int8_t *const *rows, int tstride, const int8_t *w,
 *                     uint8_t *img, int chunks);
 *
 * The input is one channel's window columns, 8 bytes a padded row: Z_j[t] =
 * the padded row t's bytes [j, j + 8), plane j (j < 5) tstride bytes after
 * plane j - 1, rows 8 bytes apart. rows[i] is Z_0 at window row i of the
 * first run's first output row. For each of `chunks` runs (the next run 16
 * bytes on: two output rows), lane l (l < 8) is output row 2u's pixel l and
 * lane 8 + l output row 2u + 1's: tap (i, j) is the 16 bytes at
 * rows[i] + j * tstride + 16 u (Z_j's rows 2u + i and 2u + i + 1), times the
 * tap's weight, broadcast; w the 25 weights row by row. The image is
 * dw5_plane_s3's (lane l at lane l). Each tap's 16 bytes come by two 8-byte
 * loads (EE.VLD.L.64, EE.VLD.H.64), its weight by the broadcast load fused
 * into the multiply-accumulate before it.
 *
 * Contract, not checked here: rows[i] and tstride multiples of 8, img a
 * multiple of 16; chunks >= 0; every byte of each tap's 16 readable.
 */

/* One tap: its 16 bytes into \xc (the address register steps to the next
 * plane), the tap before's multiply-accumulate with this tap's weight
 * broadcast into \wn. */
.macro	PAIR	wn, xp, wp, xc
	ee.vld.l.64.ip	\xc, a12, 8
	ee.vld.h.64.xp	\xc, a12, a13
	ee.vmulas.s8.qacc.ldbc.incp	\wn, a4, \xp, \wp
.endm

	.align	4
	.global	dw5_pairs_s3
	.type	dw5_pairs_s3, @function

/* windowed ABI: a2 rows, a3 tstride, a4 w, a5 img, a6 chunks */
dw5_pairs_s3:
	entry	a1, 32
	blti	a6, 1, .Ldone_pairs
	l32i	a7, a2, 0
	l32i	a8, a2, 4
	l32i	a9, a2, 8
	l32i	a10, a2, 12
	l32i	a11, a2, 16
	addi	a13, a3, -8		/* the next plane, less the high half's 8 */
.Lchunk_pairs:
	ee.zero.qacc
	/* Taps 0-24; codes q0/q1 and weights q2/q3 alternate. */
	mov	a12, a7
	ee.vld.l.64.ip	q0, a12, 8
	ee.vld.h.64.xp	q0, a12, a13
	ee.vldbc.8.ip	q2, a4, 1
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	mov	a12, a8
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	mov	a12, a9
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	mov	a12, a10
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	mov	a12, a11
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	PAIR	q3, q0, q2, q1
	PAIR	q2, q1, q3, q0
	/* Tap 24's multiply-accumulate (codes q0, weight q2) among the next
	 * run's rows: room after its load and before QACC is read. */
	addi	a7, a7, 16
	addi	a8, a8, 16
	ee.vmulas.s8.qacc	q0, q2
	addi	a9, a9, 16
	addi	a10, a10, 16
	addi	a11, a11, 16
	addi	a4, a4, -25
	STORE_IMAGE
	addi	a6, a6, -1
	bnez	a6, .Lchunk_pairs
.Ldone_pairs:
	retw.n

	.size	dw5_pairs_s3, . - dw5_pairs_s3
