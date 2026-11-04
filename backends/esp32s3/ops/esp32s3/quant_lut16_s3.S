/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The hand-scheduled table steps of espressif::lut16 (quant_lut16.cpp calls
// them on the device; its C loops are the host's and the reference, and the
// device check runs both against the mirrors in quantizer/mirrors.py).
// The loops take the op's runtime parameters and follow the mirror's
// conventions:
//  - every parameter comes from a Lut16LoopArgs (quant_lut16.cpp; offsets
//    below), nothing is folded into the program's table;
//  - the input is already in the table's domain (apply() clamps while it
//    quantises): u = c - lo as uint16 for the uniform loops, c as int16 for
//    NSAU; the output is clamped to [out_min, out_max] with MAX/MIN;
//  - a zero-overhead loop (LOOPGTZ); one SAR setting per loop where the
//    scheme allows: the uniform linear index is (u * 2^G) >> (F + G), so the
//    index and the final shift share F + G;
//  - each multiply and add of the cubic rounds on its own (no MADD.S), and
//    round half to even is (y + 1.5 * 2^23) - 1.5 * 2^23, as in the mirror.
// Windowed ABI: arguments in a2..a5, ENTRY/RETW. Static loop bodies: uniform
// linear 18 instructions, uniform cubic 25 (29 with kTail), NSAU cubic 38
// (kOffset only moves its two base pointers).

// Lut16LoopArgs
#define ARG_DATA     0
#define ARG_OUT_MIN  4
#define ARG_OUT_MAX  8
#define ARG_SAR      12
#define ARG_MASK     16
#define ARG_POW2G    20
#define ARG_POW2F    24
#define ARG_RND      28
#define ARG_SCALE    32
#define ARG_M31      36
#define ARG_POW2M    40
#define ARG_NEGOFF   44
#define ARG_POW2NEG  48
#define ARG_LO       52
#define ARG_KMAX     56

    .text
    .literal_position

// void lut16_uniform_linear_s3(const uint16_t* u, int16_t* y, int n,
//                              const Lut16LoopArgs* a)
// y = (V[k] * 2^F + (V[k+1] - V[k]) * t + 2^(F+G-1)) >> (F + G),
// k = u >> F, t = u & (2^F - 1).
    .align 4
    .global lut16_uniform_linear_s3
    .type lut16_uniform_linear_s3, @function
lut16_uniform_linear_s3:
    entry   a1, 32
    l32i    a10, a5, ARG_DATA       // V
    l32i    a13, a5, ARG_OUT_MIN
    l32i    a14, a5, ARG_OUT_MAX
    l32i    a6, a5, ARG_SAR         // F + G
    l32i    a9, a5, ARG_MASK        // 2^F - 1
    l32i    a8, a5, ARG_POW2G       // 2^G
    l32i    a11, a5, ARG_POW2F      // 2^F
    l32i    a12, a5, ARG_RND        // 2^(F+G-1), or 0
    ssr     a6
    loopgtz a4, .Lul_end
    l16ui   a15, a2, 0              // u
    addi.n  a2, a2, 2
    mull    a4, a15, a8             // u * 2^G (< 2^32, unsigned)
    and     a5, a15, a9             // t
    srl     a4, a4                  // k
    addx4   a4, a4, a10             // &V[k]
    l32i.n  a15, a4, 0              // V[k]
    l32i.n  a4, a4, 4               // V[k+1]
    sub     a4, a4, a15
    mull    a4, a4, a5              // (V[k+1] - V[k]) * t
    mull    a15, a15, a11           // V[k] * 2^F
    add.n   a15, a15, a4
    add.n   a15, a15, a12
    sra     a15, a15                // >> (F + G)
    max     a15, a15, a13
    min     a15, a15, a14
    s16i    a15, a3, 0
    addi.n  a3, a3, 2
.Lul_end:
    retw.n
    .size lut16_uniform_linear_s3, . - lut16_uniform_linear_s3

// void lut16_uniform_cubic_s3(const uint16_t* u, int16_t* y, int n,
//                             const Lut16LoopArgs* a)
// tf = t * 2^-F, y = c0 + tf * (c1 + tf * (c2 + tf * c3)), one rounding
// per operation; C[k] is 16 B: c0, c1, c2, c3.
    .align 4
    .global lut16_uniform_cubic_s3
    .type lut16_uniform_cubic_s3, @function
lut16_uniform_cubic_s3:
    entry   a1, 32
    l32i    a12, a5, ARG_DATA       // C
    l32i    a13, a5, ARG_OUT_MIN
    l32i    a14, a5, ARG_OUT_MAX
    l32i    a6, a5, ARG_SAR         // F
    l32i    a11, a5, ARG_MASK       // 2^F - 1
    lsi     f8, a5, ARG_SCALE       // 2^-F
    movi    a7, 0x4b400000          // 1.5 * 2^23
    wfr     f9, a7
    ssr     a6
    loopgtz a4, .Luc_end
    l16ui   a8, a2, 0               // u
    addi.n  a2, a2, 2
    srl     a10, a8                 // k
    and     a8, a8, a11             // t
    float.s f0, a8, 0
    mul.s   f0, f0, f8              // tf, exact
    slli    a10, a10, 4
    add.n   a10, a10, a12           // &C[k]
    lsi     f1, a10, 12             // c3
    lsi     f2, a10, 8              // c2
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    lsi     f2, a10, 4              // c1
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    lsi     f2, a10, 0              // c0
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    add.s   f1, f1, f9
    sub.s   f1, f1, f9              // round half to even (|y| < 2^22)
    trunc.s a8, f1, 0
    max     a8, a8, a13
    min     a8, a8, a14
    s16i    a8, a3, 0
    addi.n  a3, a3, 2
.Luc_end:
    retw.n
    .size lut16_uniform_cubic_s3, . - lut16_uniform_cubic_s3

// void lut16_uniform_cubic_tail_s3(const int16_t* c, int16_t* y, int n,
//                                  const Lut16LoopArgs* a)
// kTail: u = c - lo, k = clamp(u >> F, 0, K - 1), t = u - k * 2^F (past the
// table t < 0 or t >= 2^F: the edge segment goes on), tf = t * 2^-F (exact,
// |t| < 2^17), then the cubic as above. c is any int16 code.
    .align 4
    .global lut16_uniform_cubic_tail_s3
    .type lut16_uniform_cubic_tail_s3, @function
lut16_uniform_cubic_tail_s3:
    entry   a1, 32
    l32i    a12, a5, ARG_DATA       // C
    l32i    a13, a5, ARG_OUT_MIN
    l32i    a14, a5, ARG_OUT_MAX
    l32i    a6, a5, ARG_SAR         // F
    l32i    a15, a5, ARG_POW2F      // 2^F
    l32i    a9, a5, ARG_LO
    l32i    a11, a5, ARG_KMAX       // K - 1
    lsi     f8, a5, ARG_SCALE       // 2^-F
    movi    a7, 0x4b400000          // 1.5 * 2^23
    wfr     f9, a7
    movi    a7, 0
    ssr     a6
    loopgtz a4, .Lut_end
    l16si   a8, a2, 0               // c
    addi.n  a2, a2, 2
    sub     a8, a8, a9              // u
    sra     a10, a8                 // u >> F, a floor
    max     a10, a10, a7
    min     a10, a10, a11           // k
    mull    a6, a10, a15
    sub     a8, a8, a6              // t
    float.s f0, a8, 0
    mul.s   f0, f0, f8              // tf, exact
    slli    a10, a10, 4
    add.n   a10, a10, a12           // &C[k]
    lsi     f1, a10, 12             // c3
    lsi     f2, a10, 8              // c2
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    lsi     f2, a10, 4              // c1
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    lsi     f2, a10, 0              // c0
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    add.s   f1, f1, f9
    sub.s   f1, f1, f9              // round half to even (|y| < 2^22)
    trunc.s a8, f1, 0
    max     a8, a8, a13
    min     a8, a8, a14
    s16i    a8, a3, 0
    addi.n  a3, a3, 2
.Lut_end:
    retw.n
    .size lut16_uniform_cubic_tail_s3, . - lut16_uniform_cubic_tail_s3

// void lut16_nsau_cubic_s3(const int16_t* c, int16_t* y, int n,
//                          const Lut16LoopArgs* a)
// a = |c|, sh = (31 - m) - nsau(a | 2^m) = max(floor(log2(a | 1)) - m, 0),
// q = a >> sh, k = (sh << m) + q, read at data + 16 k (c >= 0) or data +
// negoff + 16 k (c < 0) -- quant_lut16.cpp biases both for the layout --
// tf = a * 2^-sh - q = t * 2^-sh (both steps exact), then the cubic.
    .align 4
    .global lut16_nsau_cubic_s3
    .type lut16_nsau_cubic_s3, @function
lut16_nsau_cubic_s3:
    entry   a1, 32
    l32i    a12, a5, ARG_DATA       // C
    l32i    a13, a5, ARG_OUT_MIN
    l32i    a14, a5, ARG_OUT_MAX
    l32i    a9, a5, ARG_M31         // 31 - m
    l32i    a10, a5, ARG_POW2M      // 2^m
    l32i    a11, a5, ARG_NEGOFF     // bytes to the negative half's base
    l32i    a15, a5, ARG_POW2NEG    // 2^-sh, sh = 0..15
    movi    a7, 0x4b400000          // 1.5 * 2^23
    wfr     f9, a7
    loopgtz a4, .Lnc_end
    l16si   a4, a2, 0               // c
    addi.n  a2, a2, 2
    srai    a5, a4, 31
    and     a5, a5, a11
    add.n   a5, a5, a12             // the base of c's sign
    abs     a4, a4                  // a
    or      a6, a4, a10
    nsau    a6, a6
    sub     a6, a9, a6              // sh
    ssr     a6
    float.s f0, a4, 0               // a
    srl     a4, a4                  // q
    float.s f1, a4, 0               // q
    addx4   a7, a6, a15
    lsi     f2, a7, 0               // 2^-sh
    mul.s   f0, f0, f2
    sub.s   f0, f0, f1              // tf
    mull    a6, a6, a10             // sh << m
    add.n   a6, a6, a4              // k
    slli    a6, a6, 4
    add.n   a6, a6, a5              // &C[k]
    lsi     f1, a6, 12              // c3
    lsi     f2, a6, 8               // c2
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    lsi     f2, a6, 4               // c1
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    lsi     f2, a6, 0               // c0
    mul.s   f1, f0, f1
    add.s   f1, f2, f1
    add.s   f1, f1, f9
    sub.s   f1, f1, f9              // round half to even (|y| < 2^22)
    trunc.s a4, f1, 0
    max     a4, a4, a13
    min     a4, a4, a14
    s16i    a4, a3, 0
    addi.n  a3, a3, 2
.Lnc_end:
    retw.n
    .size lut16_nsau_cubic_s3, . - lut16_nsau_cubic_s3
