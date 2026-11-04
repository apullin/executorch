/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 PIE: 16 channel planes' codes into channels-last vectors, the
 * static int8 conv path's gathers (esp32_conv.cpp: the PIE depthwise strip,
 * a pointwise conv's rows).
 *
 *   void transpose16_s3(const int8_t *src, int plane, int8_t *dst,
 *                       int stride, int count);
 *
 * dst[p * stride + j] = src[j * plane + p] for p in [0, count), j in [0, 16):
 * pixel p's 16 channels (each channel's codes `plane` bytes after the last's)
 * as 16 bytes at dst + p * stride. Tiles of 16 pixels: each channel's 16
 * codes by an unaligned load (EE.LD.128.USAR.IP, the next aligned block,
 * EE.SRC.Q), the 16 x 16 byte block transposed eight channels at a time by
 * three rounds of EE.VZIP (8, 16, 32 bits), and each pixel's eight bytes
 * stored with EE.VST.L/H.64.XP; the last tile stores only its pixels.
 *
 * Contract, not checked here: dst and stride multiples of 8 (a misaligned
 * ee.vst.64 silently truncates the address); count >= 0; and for each
 * channel j and tile t, with a = src + j * plane + 16 * t (the tile's first
 * code), every byte of [a & ~15, a + 32) readable. Each unaligned load reads
 * the whole aligned 16-byte block holding a -- up to 15 bytes BEFORE a --
 * and the next block, up to 32 bytes past a; the bytes outside the codes
 * the call copies are shifted out by EE.SRC.Q or never stored, only read.
 * The callers keep the forward end inside their buffers (esp32_conv.cpp
 * checks it at each call). The backward bytes need no check on the S3: they
 * share a's aligned 16-byte block, so a's 64-byte cache line, and every
 * mapped region (internal SRAM, the MMU's 64 KB flash and PSRAM pages)
 * starts on a 16-byte boundary, so the block is mapped whenever a is. A
 * memory with finer protection than 16 bytes would need a guard. One
 * 16-byte spill slot on the stack, aligned inside the frame whatever the
 * stack pointer's alignment.
 *
 * q0-q7, SAR_BYTE and the address registers here are caller-saved
 * (abikind=0); the calling task is pinned to its core.
 */

	.text
	.align	4
	.global	transpose16_s3
	.type	transpose16_s3, @function

/* windowed ABI: a2 src, a3 plane, a4 dst, a5 stride, a6 count */
transpose16_s3:
	entry	a1, 64
	blti	a6, 1, .Ldone
	slli	a13, a5, 4		/* 16 pixels of dst */
	addi	a14, a1, 31
	srli	a14, a14, 4
	slli	a14, a14, 4		/* a 16-byte aligned spill slot inside the frame */
.Ltile:
	movi	a7, 16
	min	a7, a6, a7		/* this tile's pixels */
	/* channels 0-7: bytes 0-7 of each pixel */
	mov	a8, a2
	mov	a9, a4
	mov	a10, a8
	ee.ld.128.usar.ip	q0, a10, 16
	ee.vld.128.ip	q1, a10, -16
	ee.src.q	q0, q0, q1
	add	a10, a10, a3
	ee.ld.128.usar.ip	q1, a10, 16
	ee.vld.128.ip	q2, a10, -16
	ee.src.q	q1, q1, q2
	add	a10, a10, a3
	ee.vzip.8	q0, q1
	ee.ld.128.usar.ip	q2, a10, 16
	ee.vld.128.ip	q3, a10, -16
	ee.src.q	q2, q2, q3
	add	a10, a10, a3
	ee.ld.128.usar.ip	q3, a10, 16
	ee.vld.128.ip	q4, a10, -16
	ee.src.q	q3, q3, q4
	add	a10, a10, a3
	ee.vzip.8	q2, q3
	ee.vzip.16	q0, q2
	ee.vzip.16	q1, q3
	ee.ld.128.usar.ip	q4, a10, 16
	ee.vld.128.ip	q5, a10, -16
	ee.src.q	q4, q4, q5
	add	a10, a10, a3
	ee.ld.128.usar.ip	q5, a10, 16
	ee.vld.128.ip	q6, a10, -16
	ee.src.q	q5, q5, q6
	add	a10, a10, a3
	ee.vzip.8	q4, q5
	ee.ld.128.usar.ip	q6, a10, 16
	ee.vld.128.ip	q7, a10, -16
	ee.src.q	q6, q6, q7
	add	a10, a10, a3
	ee.vst.128.ip	q3, a14, 0		/* the eighth channel's temporary */
	ee.ld.128.usar.ip	q7, a10, 16
	ee.vld.128.ip	q3, a10, -16
	ee.src.q	q7, q7, q3
	ee.vld.128.ip	q3, a14, 0
	ee.vzip.8	q6, q7
	ee.vzip.16	q4, q6
	ee.vzip.16	q5, q7
	ee.vzip.32	q0, q4
	ee.vzip.32	q2, q6
	ee.vzip.32	q1, q5
	ee.vzip.32	q3, q7
	mov	a11, a9
	bnei	a7, 16, .Lpart0
	ee.vst.l.64.xp	q0, a11, a5
	ee.vst.h.64.xp	q0, a11, a5
	ee.vst.l.64.xp	q4, a11, a5
	ee.vst.h.64.xp	q4, a11, a5
	ee.vst.l.64.xp	q2, a11, a5
	ee.vst.h.64.xp	q2, a11, a5
	ee.vst.l.64.xp	q6, a11, a5
	ee.vst.h.64.xp	q6, a11, a5
	ee.vst.l.64.xp	q1, a11, a5
	ee.vst.h.64.xp	q1, a11, a5
	ee.vst.l.64.xp	q5, a11, a5
	ee.vst.h.64.xp	q5, a11, a5
	ee.vst.l.64.xp	q3, a11, a5
	ee.vst.h.64.xp	q3, a11, a5
	ee.vst.l.64.xp	q7, a11, a5
	ee.vst.h.64.xp	q7, a11, a5
	j	.Lhalf0
.Lpart0:
	mov	a12, a7
	ee.vst.l.64.xp	q0, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q0, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q4, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q4, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q2, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q2, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q6, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q6, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q1, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q1, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q5, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q5, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q3, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.h.64.xp	q3, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
	ee.vst.l.64.xp	q7, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf0
.Lhalf0:
	/* channels 8-15: bytes 8-15 */
	addx8	a8, a3, a2
	addi	a9, a4, 8
	mov	a10, a8
	ee.ld.128.usar.ip	q0, a10, 16
	ee.vld.128.ip	q1, a10, -16
	ee.src.q	q0, q0, q1
	add	a10, a10, a3
	ee.ld.128.usar.ip	q1, a10, 16
	ee.vld.128.ip	q2, a10, -16
	ee.src.q	q1, q1, q2
	add	a10, a10, a3
	ee.vzip.8	q0, q1
	ee.ld.128.usar.ip	q2, a10, 16
	ee.vld.128.ip	q3, a10, -16
	ee.src.q	q2, q2, q3
	add	a10, a10, a3
	ee.ld.128.usar.ip	q3, a10, 16
	ee.vld.128.ip	q4, a10, -16
	ee.src.q	q3, q3, q4
	add	a10, a10, a3
	ee.vzip.8	q2, q3
	ee.vzip.16	q0, q2
	ee.vzip.16	q1, q3
	ee.ld.128.usar.ip	q4, a10, 16
	ee.vld.128.ip	q5, a10, -16
	ee.src.q	q4, q4, q5
	add	a10, a10, a3
	ee.ld.128.usar.ip	q5, a10, 16
	ee.vld.128.ip	q6, a10, -16
	ee.src.q	q5, q5, q6
	add	a10, a10, a3
	ee.vzip.8	q4, q5
	ee.ld.128.usar.ip	q6, a10, 16
	ee.vld.128.ip	q7, a10, -16
	ee.src.q	q6, q6, q7
	add	a10, a10, a3
	ee.vst.128.ip	q3, a14, 0		/* the eighth channel's temporary */
	ee.ld.128.usar.ip	q7, a10, 16
	ee.vld.128.ip	q3, a10, -16
	ee.src.q	q7, q7, q3
	ee.vld.128.ip	q3, a14, 0
	ee.vzip.8	q6, q7
	ee.vzip.16	q4, q6
	ee.vzip.16	q5, q7
	ee.vzip.32	q0, q4
	ee.vzip.32	q2, q6
	ee.vzip.32	q1, q5
	ee.vzip.32	q3, q7
	mov	a11, a9
	bnei	a7, 16, .Lpart1
	ee.vst.l.64.xp	q0, a11, a5
	ee.vst.h.64.xp	q0, a11, a5
	ee.vst.l.64.xp	q4, a11, a5
	ee.vst.h.64.xp	q4, a11, a5
	ee.vst.l.64.xp	q2, a11, a5
	ee.vst.h.64.xp	q2, a11, a5
	ee.vst.l.64.xp	q6, a11, a5
	ee.vst.h.64.xp	q6, a11, a5
	ee.vst.l.64.xp	q1, a11, a5
	ee.vst.h.64.xp	q1, a11, a5
	ee.vst.l.64.xp	q5, a11, a5
	ee.vst.h.64.xp	q5, a11, a5
	ee.vst.l.64.xp	q3, a11, a5
	ee.vst.h.64.xp	q3, a11, a5
	ee.vst.l.64.xp	q7, a11, a5
	ee.vst.h.64.xp	q7, a11, a5
	j	.Lhalf1
.Lpart1:
	mov	a12, a7
	ee.vst.l.64.xp	q0, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q0, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q4, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q4, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q2, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q2, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q6, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q6, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q1, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q1, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q5, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q5, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q3, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.h.64.xp	q3, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
	ee.vst.l.64.xp	q7, a11, a5
	addi	a12, a12, -1
	beqz	a12, .Lhalf1
.Lhalf1:
	addi	a2, a2, 16
	add	a4, a4, a13
	sub	a6, a6, a7
	bnez	a6, .Ltile
.Ldone:
	retw.n

	.size	transpose16_s3, . - transpose16_s3
