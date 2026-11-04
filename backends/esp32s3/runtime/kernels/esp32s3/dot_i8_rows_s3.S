/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * The row loop below is dot_i8_s3.S's instruction sequence, from esp32-ai's
 * dot_i8_s3.S, under its MIT notice:
 *
 * MIT License
 *
 * Copyright (c) 2026 Viacheslav Sierbov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* ESP32-S3 PIE: int8 dot products of several rows against one weight row.
 *
 *   void dot_i8_rows_s3(const int8_t *rows, int stride, const int8_t *w,
 *                       int n, int32_t *out, int count);
 *
 * out[t] = dot(rows + t * stride, w, n) for t in [0, count): dot_i8_s3's
 * arithmetic and contract row by row (ACCX, the low 32 bits of the exact sum,
 * which kMaxDotLen keeps within int32; 16-byte aligned rows and weights; only
 * floor(n / 16) * 16 elements; n < 16 gives 0), in one call: the static int8
 * conv path's columns call it once per channel and row block instead of once
 * per output, which saves the call and the argument setup around each dot.
 *
 * q0, q1, ACCX and the address registers here are caller-saved (abikind=0);
 * the calling task is pinned to its core, as for dot_i8_s3.
 */

	.text
	.align	4
	.global	dot_i8_rows_s3
	.type	dot_i8_rows_s3, @function

/* windowed ABI: a2 rows, a3 stride, a4 w, a5 n, a6 out, a7 count */
dot_i8_rows_s3:
	entry	a1, 32
	blti	a7, 1, .Ldone
	srli	a8, a5, 4		/* chunks = n / 16 */
	addi	a9, a8, -1		/* the fused loop's trips */
	beqz	a8, .Lzeros
.Lrow:
	ee.zero.accx
	mov	a10, a2
	mov	a11, a4
	ee.vld.128.ip	q0, a10, 16
	ee.vld.128.ip	q1, a11, 16
	loopgtz	a9, .Lmac_end
	ee.vmulas.s8.accx.ld.ip	q0, a10, 16, q0, q1
	ee.vld.128.ip	q1, a11, 16
.Lmac_end:
	ee.vmulas.s8.accx	q0, q1
	/* The next row, and the two instructions ACCX needs before rur. */
	add	a2, a2, a3
	addi	a7, a7, -1
	rur.accx_0	a12
	s32i	a12, a6, 0
	addi	a6, a6, 4
	bnez	a7, .Lrow
	retw.n
.Lzeros:
	movi	a12, 0
.Lzero_row:
	s32i	a12, a6, 0
	addi	a6, a6, 4
	addi	a7, a7, -1
	bnez	a7, .Lzero_row
.Ldone:
	retw.n

	.size	dot_i8_rows_s3, . - dot_i8_rows_s3
