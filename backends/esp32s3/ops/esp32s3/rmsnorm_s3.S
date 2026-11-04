/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* ESP32-S3 fp32 RMSNorm over one row.
 *
 *   int rmsnorm_f32_s3(float *out, const float *x, const float *w,
 *                      int d, float inv_d, float eps);
 *
 * Computes, for i in [0, d):
 *
 *   s      = rsqrt(sum(x[j]^2) * inv_d + eps)
 *   out[i] = (w[i] * x[i]) * s   when w != NULL
 *          = x[i] * s            when w == NULL
 *
 * The caller passes inv_d = 1.0f/(float)d rather than d alone, because this core has no
 * float divide that GCC will emit (see "Why assembly" below) and a C-side divide is
 * correctly rounded for free. For d a power of two inv_d is exact and `sum*inv_d` is
 * bit-identical to the portable kernel's `sum/d`; for any other d (96, say)
 * they differ by at most one rounding of the mean, i.e. ~6e-8 relative.
 *
 * Contract, and NONE of it is checked here -- the shim in espressif_ops.cpp checks and falls back:
 *   - out, x and non-null w are 4-byte aligned. That is all: this kernel uses scalar LSI/SSI, not
 *     EE.LDF.128.IP, so there is no 16-byte requirement and no aligned/misaligned split.
 *     See "Why not the 128-bit float loads" below; it is a deliberate trade.
 *   - any d >= 1 is handled: floor(d/4)*4 elements go through the unrolled body and the
 *     remaining d&3 through a scalar tail. d < 1 returns without writing anything.
 *   - out may alias x exactly (the second pass reads x[i] before writing out[i] at the
 *     same index), but must not partially overlap it at a different offset.
 *   - x, non-null w and out are fp32 and contiguous over the row; the shim loops rows itself.
 *
 * Returns 1 when the row was written, and 0 -- having written nothing -- when
 * t = sum*inv_d + eps is not a positive normal float, the domain the refinement below
 * needs. Finite activations reach that easily: four values of 1e19 square to 4e38, the
 * sum overflows to +inf, rsqrt0.s seeds 0, and the first Newton step computes inf*0 =
 * NaN for the whole row, where the scalar reference's 1/sqrtf(inf) = 0 gives zeros.
 * eps = 0 on an all-zero row (t = 0) and NaN input break it the same way. The check
 * runs before rsqrt0.s, so the caller can hand such a row to the scalar reference, and
 * the refusal is exercisable even under QEMU, which lacks that instruction.
 *
 * Clobbers: a2-a12 (all caller-saved inside our own window), f0-f9, b0, and
 * LCOUNT/LBEG/LEND via LOOPNEZ. FR[0]-FR[15] and BR[0] are call-clobbered in both Xtensa
 * ABIs -- see
 * CALL_REALLY_USED_REGISTERS in gcc/config/xtensa/xtensa.h, where all sixteen FR entries
 * are 1 -- so a leaf routine may use them freely. No AR is read back from an FR, so there
 * is no RFR hazard to pad. Reads no memory except x[0..d) and w[0..d); writes only
 * out[0..d).
 *
 * FPU is coprocessor 0 on the S3 (XCHAL_CP_ID_FPU in xtensa/config/tie.h; PIE is CP3).
 * FreeRTOS saves FPU state lazily per task via the CP-disabled exception, and coprocessor
 * ownership is tracked per core, so the calling task must be pinned to a core -- the same
 * constraint dot_i8_s3.S already documents, and the inference task already satisfies it.
 *
 * ---- rsqrt: rsqrt0.s + 2 Newton-Raphson iterations ------------------------
 *
 * RSQRT0.S is an APPROXIMATION, not a rounded reciprocal square root. Cadence documents
 * the single-precision seed as a piecewise-linear table lookup accurate to ~14 bits.
 * Newton-Raphson for 1/sqrt(t) squares the error each step:
 *
 *     y <- y * (1.5 - (0.5*t) * y*y)
 *
 * Measured (test_rmsnorm.py, which simulates this exact instruction sequence in fp32
 * over 2e4 log-uniform magnitudes spanning 1e-8..1e8 plus the denormal/huge edges, and
 * sweeping the seed's worst-case relative error rather than trusting one figure):
 *
 *     seed err | 1 iter    | 2 iters   | 3 iters
 *     2^-8     | 2.30e-05  | 1.42e-07  | 1.21e-07
 *     2^-10    | 1.57e-06  | 1.32e-07  | 1.21e-07
 *     2^-12    | 2.29e-07  | 1.21e-07  | 1.13e-07
 *     2^-14    | 1.37e-07  | 1.21e-07  | 1.21e-07
 *
 * So: TWO iterations. That reaches 1.42e-07 relative -- 1.19 ulp, essentially the rounding
 * floor of the three-op refinement itself -- for any seed accurate to 8 bits or better,
 * which is a 64x margin over the documented 14. A third iteration buys nothing in fp32. One
 * iteration would be enough IF the 14-bit figure is exact, but it degrades to 2.3e-05 if
 * the seed is worse than advertised, and the second iteration costs 3 instructions once
 * per row of ~300 flops. Reference `1/sqrtf(t)` is <=0.5 ulp (5.96e-08), so we are within
 * one ulp of exactly rounded, and the rsqrt is not the accuracy-limiting step: the mean's
 * own summation accounts for 2.45e-07 (measured, see "Summation order" below).
 *
 * End to end, test_rmsnorm.py transcribes this whole routine into fp32 numpy -- same
 * roundings, same accumulator layout, same two NR steps -- and gets 1.96e-07 worst-case
 * relative error against espressif::rmsnorm over the real model's 200 activation rows.
 * That ~2e-07 per element is the parity bound the device run is judged against. It
 * validates the arithmetic only; the encoding, register allocation and ABI are
 * exercised on silicon only (by the parity rows that reach this kernel), since
 * QEMU cannot execute rsqrt0.s.
 *
 * Why assembly, and why rsqrt0.s at all: this toolchain's GCC has no divsf3/sqrtsf2
 * pattern whatsoever (grep gcc/config/xtensa/xtensa.md for "sqrt" -- nothing), so C
 * `1.0f/sqrtf(x)` compiles to a `sqrtf` call followed by a `__divsf3` call, two soft-float
 * library calls, even though XCHAL_HAVE_FP_RSQRT is 1 on this core. rsqrt0.s is
 * unreachable from C here, which is the whole justification for hand-writing this.
 * [INFERENCE: the kernel runs on silicon, but this comparison has not been timed] the
 * refinement is ~9 FPU ops versus hundreds of cycles for two soft-float calls.
 *
 * Why not the 128-bit float loads: EE.LDF.128.IP / EE.STF.128.IP move four floats per
 * instruction (esp-dsp's dsps_dotprod_f32_aes3 uses them) and would cut this kernel from
 * ~2.25 to ~1.25 instructions per element in pass 1. They also demand 16-byte alignment
 * and d%4==0, which means a guard plus a second scalar body for every pass -- four loop
 * bodies, none of which can be unit-tested on the host. One RMSNorm is ~300 flops against
 * the model's ~393k MACs per token, so the entire kernel is <0.1% of the work and a 1.4x
 * speedup on it is unmeasurable. Correctness of one body beats speed of four.
 *
 * MADD.S is FUSED: "the computation is performed with no intermediate round" (Xtensa ISA
 * Reference Manual 8.3, MADD.S). So sum(x^2) here rounds once per term where the
 * decomposed graph rounds twice (pow writes an fp32 temp, then mean adds it) -- this
 * kernel is slightly MORE accurate than the ops it replaces, not less. The same fusion is
 * used deliberately in the MSUB.S of the refinement.
 *
 * Summation order: four independent accumulators, reduced f0+f1, +f2, +f3, then the tail
 * added into f0 -- exactly the shape esp-dsp's fp32 dot products use, to keep the madd.s
 * dependency chains four deep instead of d deep. This is NOT the portable kernel's
 * sequential order, so the mean differs from mean.dim's in the last ulp or two. It is the
 * more accurate order (shorter chains), and test_rmsnorm.py reports the size of the
 * difference so the parity threshold is chosen from a measurement.
 */

	.text
	.literal_position
	.literal .Lc_half, 0x3f000000	/* 0.5f */
	.literal .Lc_1p5,  0x3fc00000	/* 1.5f */
	.literal .Lc_flt_min, 0x00800000	/* smallest positive normal float */
	.literal .Lc_flt_max, 0x7f7fffff	/* largest finite float */
	.align	4
	.global	rmsnorm_f32_s3
	.type	rmsnorm_f32_s3, @function

/* windowed ABI: a2 = out, a3 = x, a4 = w, a5 = d, a6 = inv_d, a7 = eps.
 * Floats arrive in AR registers on this target -- GCC passes them in a2.. and moves them
 * with WFR (see the codegen for `float f(float,float,float){return a*b+c;}`), so a6/a7
 * hold raw fp32 bit patterns. */
rmsnorm_f32_s3:
	entry	a1, 32
	blti	a5, 1, .Ldone		/* d < 1: nothing to normalise, out untouched */

	/* ---- pass 1: sum of x[j]^2 into four accumulators ---- */
	movi.n	a8, 0
	wfr	f0, a8
	wfr	f1, a8
	wfr	f2, a8
	wfr	f3, a8

	mov.n	a9, a3			/* cursor over x */
	srai	a10, a5, 2		/* full 4-element blocks */
	loopnez	a10, .Lsq_end
	lsi	f4, a9, 0
	lsi	f5, a9, 4
	lsi	f6, a9, 8
	lsi	f7, a9, 12
	madd.s	f0, f4, f4
	madd.s	f1, f5, f5
	madd.s	f2, f6, f6
	madd.s	f3, f7, f7
	addi	a9, a9, 16
.Lsq_end:

	add.s	f0, f0, f1
	add.s	f0, f0, f2
	add.s	f0, f0, f3

	extui	a11, a5, 0, 2		/* d & 3 leftover elements */
	loopnez	a11, .Lsq_tail_end
	lsi	f4, a9, 0
	madd.s	f0, f4, f4
	addi	a9, a9, 4
.Lsq_tail_end:

	/* ---- mean + eps ---- */
	wfr	f5, a6			/* inv_d */
	wfr	f6, a7			/* eps   */
	mul.s	f0, f0, f5		/* mean = sum * inv_d */
	add.s	f0, f0, f6		/* t    = mean + eps  */

	/* ---- domain: FLT_MIN <= t <= FLT_MAX. Ordered compares, so NaN fails too. ---- */
	l32r	a8, .Lc_flt_min
	wfr	f7, a8
	ole.s	b0, f7, f0
	bf	b0, .Lrefuse
	l32r	a8, .Lc_flt_max
	wfr	f7, a8
	ole.s	b0, f0, f7
	bf	b0, .Lrefuse

	/* ---- s = rsqrt(t): ~14-bit seed, then 2 Newton-Raphson iterations ---- */
	rsqrt0.s	f1, f0
	l32r	a8, .Lc_half
	wfr	f2, a8
	mul.s	f2, f2, f0		/* h = 0.5 * t, hoisted out of both iterations */
	l32r	a8, .Lc_1p5

	mul.s	f3, f1, f1		/* iteration 1: y*y            */
	wfr	f4, a8
	msub.s	f4, f2, f3		/*              1.5 - h*(y*y)  (fused) */
	mul.s	f1, f1, f4		/*              y *= that      */

	mul.s	f3, f1, f1		/* iteration 2 */
	wfr	f4, a8
	msub.s	f4, f2, f3
	mul.s	f1, f1, f4		/* f1 = s, held across pass 2 */

	beqz	a4, .Lnoscale		/* Glimmer weightless RMSNorm */

	/* ---- pass 2: out[i] = (w[i] * x[i]) * s ---- */
	mov.n	a9, a3			/* x   */
	mov.n	a10, a4			/* w   */
	mov.n	a11, a2			/* out */
	srai	a12, a5, 2
	loopnez	a12, .Lmul_end
	lsi	f2, a9, 0
	lsi	f3, a9, 4
	lsi	f4, a9, 8
	lsi	f5, a9, 12
	lsi	f6, a10, 0
	lsi	f7, a10, 4
	lsi	f8, a10, 8
	lsi	f9, a10, 12
	mul.s	f2, f2, f6
	mul.s	f3, f3, f7
	mul.s	f4, f4, f8
	mul.s	f5, f5, f9
	mul.s	f2, f2, f1
	mul.s	f3, f3, f1
	mul.s	f4, f4, f1
	mul.s	f5, f5, f1
	ssi	f2, a11, 0
	ssi	f3, a11, 4
	ssi	f4, a11, 8
	ssi	f5, a11, 12
	addi	a9, a9, 16
	addi	a10, a10, 16
	addi	a11, a11, 16
.Lmul_end:

	extui	a12, a5, 0, 2
	loopnez	a12, .Lmul_tail_end
	lsi	f2, a9, 0
	lsi	f6, a10, 0
	mul.s	f2, f2, f6
	mul.s	f2, f2, f1
	ssi	f2, a11, 0
	addi	a9, a9, 4
	addi	a10, a10, 4
	addi	a11, a11, 4
.Lmul_tail_end:
	j	.Ldone

	/* ---- weightless pass 2: out[i] = x[i] * s ---- */
.Lnoscale:
	mov.n	a9, a3			/* x   */
	mov.n	a11, a2			/* out */
	srai	a12, a5, 2
	loopnez	a12, .Lnoscale_end
	lsi	f2, a9, 0
	lsi	f3, a9, 4
	lsi	f4, a9, 8
	lsi	f5, a9, 12
	mul.s	f2, f2, f1
	mul.s	f3, f3, f1
	mul.s	f4, f4, f1
	mul.s	f5, f5, f1
	ssi	f2, a11, 0
	ssi	f3, a11, 4
	ssi	f4, a11, 8
	ssi	f5, a11, 12
	addi	a9, a9, 16
	addi	a11, a11, 16
.Lnoscale_end:

	extui	a12, a5, 0, 2
	loopnez	a12, .Lnoscale_tail_end
	lsi	f2, a9, 0
	mul.s	f2, f2, f1
	ssi	f2, a11, 0
	addi	a9, a9, 4
	addi	a11, a11, 4
.Lnoscale_tail_end:

.Ldone:
	movi.n	a2, 1			/* the row is written (or had no elements) */
	retw.n

.Lrefuse:
	movi.n	a2, 0			/* t outside the domain: nothing written */
	retw.n

	.size	rmsnorm_f32_s3, . - rmsnorm_f32_s3
