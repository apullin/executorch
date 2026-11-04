/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 PIE (128-bit vector) int4 -> int8 unpack.
 *
 *   void unpack_i4_s3(const uint8_t *packed, int8_t *out, int n_bytes);
 *
 * Expands n_bytes of packed nibbles (two codes per byte, low nibble first,
 * stored biased by +8) into 2*n_bytes signed int8 codes, exactly as the scalar
 * unpack_i4_row does: out[2t] = (packed[t] & 0x0F) - 8, out[2t+1] =
 * (packed[t] >> 4) - 8. Contract, NOT checked here:
 *   - packed and out MUST be 16-byte aligned (a misaligned ee.vld.128.ip
 *     silently truncates the address; the caller checks and falls back);
 *   - n_bytes MUST be a multiple of 16 (16 packed bytes = 32 codes per step).
 *
 * Packing note: the stored layout interleaves low/high nibbles per byte,
 * which is why the zip is needed. dot_i4_s3.S needs no unpack at all: it
 * reads the packed row against activations deinterleaved per 32, so the low
 * and high nibbles multiply the even and odd halves directly. The backend
 * owns its packing (serialization/weight_formats.py).
 *
 * Per 16 packed bytes: one load, an AND for the low nibbles, a 32-bit lane
 * shift right by 4 plus an AND for the high nibbles (the shift drags the next
 * byte's low nibble into the high half of each byte; the mask removes it and
 * makes the shift's sign fill irrelevant), two saturating subtracts of 8 (no
 * saturation can occur: 0..15 - 8), a byte zip to interleave low/high, two
 * stores. Same clobber rules as dot_i8_s3.S: q0-q7 and SAR are caller-saved
 * and the calling task is core-pinned.
 */
	.section .rodata
	.align	4
.Lmask0f:
	.byte	0x0F
.Lbias8:
	.byte	0x08

	.text
	.align	4
	.global	unpack_i4_s3
	.type	unpack_i4_s3, @function
/* windowed ABI: a2 = packed, a3 = out, a4 = n_bytes */
unpack_i4_s3:
	entry	a1, 32
	srli	a5, a4, 4		/* a5 = n_bytes / 16 = chunk count */
	beqz	a5, .Ldone
	movi	a6, .Lmask0f
	ee.vldbc.8	q4, a6		/* q4 = 16 x 0x0F */
	movi	a6, .Lbias8
	ee.vldbc.8	q5, a6		/* q5 = 16 x 0x08 */
	movi	a6, 4
	ssr	a6			/* SAR = 4 for the lane shift */
	loopgtz	a5, .Lloop_end
	ee.vld.128.ip	q0, a2, 16
	ee.andq		q1, q0, q4	/* low nibbles */
	ee.vsr.32	q2, q0		/* each 32-bit lane >> 4 */
	ee.andq		q2, q2, q4	/* high nibbles */
	ee.vsubs.s8	q1, q1, q5	/* - 8 */
	ee.vsubs.s8	q2, q2, q5
	ee.vzip.8	q1, q2		/* interleave: q1 = codes 0..15, q2 = 16..31 */
	ee.vst.128.ip	q1, a3, 16
	ee.vst.128.ip	q2, a3, 16
.Lloop_end:
.Ldone:
	retw.n
	.size	unpack_i4_s3, . - unpack_i4_s3
