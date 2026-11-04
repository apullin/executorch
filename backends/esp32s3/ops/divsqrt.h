/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The device's float division and square root against exact rounding
// (every exact row between delegates that divides or takes
// a root assumes they round correctly to nearest even).
//
// On the ESP32-S3 neither is one instruction: float division is ROM
// __divsf3 (DIV0.S, a Newton refinement in MADDN.S, DIVN.S), sqrtf is
// libgcc's __ieee754_sqrtf (SQRT0.S ... DIVN.S), and ExecuTorch's portable
// reciprocal and rsqrt compute 1.0 / x in double (ROM __divdf3) and narrow.
// run_check computes divsqrt_golden.h's cases with those operations, spelled
// as the kernels spell them: the hardest quotients, reciprocals and roots
// (nearest a midpoint or a float), subnormal ties, the edges, and two random
// streams held by hash. One "divsqrt check:" line; run_parity gates on it.
// Under QEMU it proves QEMU's model of those FPU instructions; on the chip
// (run_hardware's parity run) the silicon's. Defined in
// divsqrt_check.cpp, which only test builds link.

namespace espressif {
namespace divsqrt {

void run_check();

}  // namespace divsqrt
}  // namespace espressif
