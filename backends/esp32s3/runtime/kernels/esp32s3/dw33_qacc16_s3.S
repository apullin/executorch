/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Written after esp-nn's esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S
 * (esp-nn 1.4.1, https://github.com/espressif/esp-nn), whose notice follows:
 *
 * Copyright 2021-2022 Espressif Systems (Shanghai) PTE LTD
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * (a copy is LICENSE-Apache-2.0, beside this file).
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MODIFIED for this backend (Andrew Pullin, 2026): rewritten as a leaf kernel
 * that leaves each output pixel's raw sums in a QACC image -- separate
 * ee.vld.128.ip loads, the nine ee.vmulas.s8.qacc products of one 3x3 window
 * over 16 channels and four raw QACC stores a pixel, with this backend's
 * arguments, registers and loop; esp-nn's requantisation, activation clamp
 * and int8 output stores are removed.
 */

/* ESP32-S3 PIE: a 3x3 depthwise convolution's multiply-adds for 16 channels,
 * over a run of output pixels, the sums left raw in QACC images.
 *
 *   void dw33_qacc16_s3(const int8_t *r0, int pitch, int step, const int8_t *w,
 *                       uint8_t *img, int count);
 *
 * The input is a channels-last plane of int8 codes: 16 bytes a pixel, one
 * channel a byte. r0 is the first output pixel's top-left tap; its other taps
 * are r0 + i * pitch + j * 16 (i, j in 0..2), and the next pixel's taps are
 * `step` bytes on. w holds the 9 taps' 16 weights each, tap-major (row, then
 * column). For each of `count` pixels, QACC is zeroed, takes the nine
 * ee.vmulas.s8.qacc products -- lane-wise int8 x int8 into 16 signed 20-bit
 * accumulators, exact while |sum| < 2^19 (nine products are at most
 * 9 * 128 * 128 = 147,456) -- and is stored raw into a 64-byte image: QACC_L's
 * 160 bits at +0 (low 128, then high 32 at +16) and QACC_H's at +32 (+48).
 * The C side (esp32_conv.cpp) unpacks the 20-bit lanes; which lane holds
 * which channel it measures once at start-up (pie_dw_init), so nothing here
 * depends on the lane order.
 *
 * Contract, not checked here: r0, pitch, step, w and img are multiples of 16
 * (a misaligned ee.vld.128 silently truncates the address); count >= 0.
 *
 * Written for this backend after esp-nn's padded 3x3 depthwise kernel
 * (esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S, Apache-2.0, (c)
 * Espressif): the same instructions for the products and the raw QACC
 * stores; its per-channel requantisation is not used.
 *
 * q0-q7, QACC and the address registers here are caller-saved (abikind=0),
 * as for dot_i8_s3; the calling task is pinned to its core.
 */

	.text
	.align	4
	.global	dw33_qacc16_s3
	.type	dw33_qacc16_s3, @function

/* windowed ABI: a2 r0, a3 pitch, a4 step, a5 w, a6 img, a7 count */
dw33_qacc16_s3:
	entry	a1, 32
	blti	a7, 1, .Ldone
	loopgtz	a7, .Lpixel_end
	ee.zero.qacc
	mov	a8, a2			/* row 0 */
	add	a9, a2, a3		/* row 1 */
	add	a10, a9, a3		/* row 2 */
	mov	a11, a5			/* the weights */
	ee.vld.128.ip	q0, a8, 16
	ee.vld.128.ip	q1, a11, 16
	ee.vld.128.ip	q2, a8, 16
	ee.vld.128.ip	q3, a11, 16
	ee.vmulas.s8.qacc	q0, q1
	ee.vld.128.ip	q4, a8, 16
	ee.vld.128.ip	q5, a11, 16
	ee.vmulas.s8.qacc	q2, q3
	ee.vld.128.ip	q0, a9, 16
	ee.vld.128.ip	q1, a11, 16
	ee.vmulas.s8.qacc	q4, q5
	ee.vld.128.ip	q2, a9, 16
	ee.vld.128.ip	q3, a11, 16
	ee.vmulas.s8.qacc	q0, q1
	ee.vld.128.ip	q4, a9, 16
	ee.vld.128.ip	q5, a11, 16
	ee.vmulas.s8.qacc	q2, q3
	ee.vld.128.ip	q0, a10, 16
	ee.vld.128.ip	q1, a11, 16
	ee.vmulas.s8.qacc	q4, q5
	ee.vld.128.ip	q2, a10, 16
	ee.vld.128.ip	q3, a11, 16
	ee.vmulas.s8.qacc	q0, q1
	ee.vld.128.ip	q4, a10, 16
	ee.vld.128.ip	q5, a11, 16
	ee.vmulas.s8.qacc	q2, q3
	ee.vmulas.s8.qacc	q4, q5
	/* The next pixel, and room before QACC is read. */
	add	a2, a2, a4
	nop
	ee.st.qacc_l.l.128.ip	a6, 16
	ee.st.qacc_l.h.32.ip	a6, 16
	ee.st.qacc_h.l.128.ip	a6, 16
	ee.st.qacc_h.h.32.ip	a6, 16
.Lpixel_end:
.Ldone:
	retw.n

	.size	dw33_qacc16_s3, . - dw33_qacc16_s3
