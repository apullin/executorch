/*
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

/* ESP32-S3 PIE (128-bit vector) int8 dot product.
 *
 *   int32_t dot_i8_s3(const int8_t *a, const int8_t *b, int n);
 *
 * Contract, and it is NOT checked here:
 *   - a and b MUST be 16-byte aligned. A misaligned ee.vld.128.ip silently
 *     truncates the low address bits and returns wrong data rather than
 *     faulting, so the caller checks alignment (the backend's
 *     pie_row_aligned).
 *   - only floor(n/16)*16 elements are consumed, and n < 16 returns 0. The
 *     backend calls this only with n a multiple of 16, so there is no tail.
 *
 * Exactness: ACCX is 40 bits but rur.accx_0 reads back its low 32, so the
 * result is the exact sum only while that sum fits int32. The backend's
 * activation codes are clamped to +/-127 and weight codes are int8, so no
 * product exceeds 127*128 and any n <= 132,096 is exact; esp32_backend.cpp
 * refuses longer rows at load (kMaxDotLen). Within that bound the result is
 * bit-identical to the scalar loop.
 *
 * Provenance: the instructions are esp32-ai's dot_i8_s3.S
 * (firmware/idf_esp32_llm/main/, MIT, (c) 2026 Viacheslav Sierbov) unchanged;
 * only this comment differs. That kernel was in turn modelled on esp-nn's
 * esp_nn_dot_s8_aligned_esp32s3 (v1.2.5). Note ACCX, not QACC:
 * ee.vmulas.s8.accx horizontally reduces all 16 lanes into one accumulator,
 * whereas ee.vmulas.s8.qacc keeps 16 independent 20-bit lanes and would need
 * a separate cross-lane sum.
 *
 * q0-q7, ACCX and SAR_BYTE are all caller-saved (abikind=0 in the S3 tie.h), so
 * this leaf routine may clobber them freely. FreeRTOS preserves PIE state across
 * preemption via the lazy CP3-disabled exception path, but coprocessor ownership
 * is tracked per core, so the calling task must be pinned to a core -- both the
 * inference task and the head worker are.
 */

	.text
	.align	4
	.global	dot_i8_s3
	.type	dot_i8_s3, @function

/* windowed ABI: a2 = a, a3 = b, a4 = n; returns in a2 */
dot_i8_s3:
	entry	a1, 32

	ee.zero.accx			/* ACCX is global caller-saved state; never implicitly 0 */
	srli	a5, a4, 4		/* a5 = n / 16 = chunk count */
	beqz	a5, .Ldone		/* n < 16: nothing to accumulate, return 0 */

	/* Software-pipelined prologue: chunk 0 in flight before the loop. */
	ee.vld.128.ip	q0, a2, 16
	ee.vld.128.ip	q1, a3, 16
	addi	a5, a5, -1		/* loop body handles all but the last chunk */

	/* Zero-overhead hardware loop. The fused MAC multiplies the CURRENT
	 * q0/q1 (16 MACs) and in the same slot loads the next 16 a-bytes into
	 * q0, post-incrementing a2 -- the load destination may alias a
	 * multiply operand because the MAC reads pre-load values. */
	loopgtz	a5, .Lloop_end
	ee.vmulas.s8.accx.ld.ip	q0, a2, 16, q0, q1
	ee.vld.128.ip	q1, a3, 16
.Lloop_end:

	/* Last chunk with the non-fused form, so we never read past either
	 * array: a2 and a3 have advanced exactly n bytes at this point. */
	ee.vmulas.s8.accx	q0, q1

.Ldone:
	/* ACCX write -> rur read needs a 2-instruction separation. */
	nop
	nop
	rur.accx_0	a2
	retw.n

	.size	dot_i8_s3, . - dot_i8_s3
