/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 PIE (128-bit vector) fused int4 x int8 dot, one sum per scale group.
 *
 *   void dot_i4_groups_s3(const uint8_t *w, const int8_t *xd, int32_t *sums,
 *                         int n_groups, int group_bytes);
 *
 * w is one packed int4 weight row in the stored layout: two codes per byte,
 * code 2t in the low nibble and 2t+1 in the high, each biased by +8 (0..15).
 * xd is the activation row DEINTERLEAVED per 32 elements: for every 32 inputs
 * x[32c .. 32c+31], first the 16 even ones x[32c+0,2,..,30], then the 16 odd
 * ones x[32c+1,3,..,31] (prepare_i4_activation in esp32_backend.cpp). With that
 * order the low nibbles of 16 packed bytes multiply the even half and the high
 * nibbles the odd half directly, so a row needs no unpack into SRAM, no byte
 * zip and no second kernel call: per 32 codes one weight load, two masks, one
 * lane shift, two activation loads and two MACs.
 *
 * sums[g] receives the RAW dot of group g: the biased codes times x. The caller
 * subtracts 8 * sum(x over the group) once per group, which is exact integer
 * arithmetic, so the result equals the unpack + dot_i8_s3 path bit for bit.
 *
 * Contract, NOT checked here:
 *   - w and xd MUST be 16-byte aligned (a misaligned ee.vld.128.ip silently
 *     truncates the address);
 *   - group_bytes MUST be a nonzero multiple of 16 (32 codes per step);
 *     n_groups may be 0.
 * Range: ACCX is 40 bits; a group of G codes sums at most G * 15 * 128, so
 * the low 32 bits read by rur.accx_0 are exact for any G below 1.1 million.
 *
 * Same clobber rules as dot_i8_s3.S: q0-q7, ACCX and SAR are caller-saved and
 * the calling task is core-pinned.
 */
	.section .rodata
	.align	4
.Lmask0f:
	.byte	0x0F
	.text
	.align	4
	.global	dot_i4_groups_s3
	.type	dot_i4_groups_s3, @function
/* windowed ABI: a2 = w, a3 = xd, a4 = sums, a5 = n_groups, a6 = group_bytes */
dot_i4_groups_s3:
	entry	a1, 32
	beqz	a5, .Ldone
	movi	a7, .Lmask0f
	ee.vldbc.8	q4, a7		/* q4 = 16 x 0x0F */
	movi	a7, 4
	ssr	a7			/* SAR = 4 for the lane shift */
	srli	a7, a6, 4		/* a7 = 16-byte steps per group */
.Lgroup:
	ee.zero.accx			/* ACCX is global caller-saved state; never implicitly 0 */
	loopgtz	a7, .Lstep_end
	ee.vld.128.ip	q0, a2, 16	/* 16 packed bytes = 32 codes */
	ee.vld.128.ip	q6, a3, 16	/* the 16 even activations */
	ee.andq		q1, q0, q4	/* low nibbles: codes 2t (biased) */
	ee.vsr.32	q2, q0		/* each 32-bit lane >> 4 */
	ee.vld.128.ip	q7, a3, 16	/* the 16 odd activations */
	ee.andq		q2, q2, q4	/* high nibbles: codes 2t+1; mask drops the neighbour's bits */
	ee.vmulas.s8.accx	q1, q6
	ee.vmulas.s8.accx	q2, q7
.Lstep_end:
	/* ACCX write -> rur read needs a 2-instruction separation. */
	nop
	nop
	rur.accx_0	a8
	s32i	a8, a4, 0
	addi	a4, a4, 4
	addi	a5, a5, -1
	bnez	a5, .Lgroup
.Ldone:
	retw.n
	.size	dot_i4_groups_s3, . - dot_i4_groups_s3
