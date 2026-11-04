/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// espressif::layer_norm2d: SAM/ConvNeXt's LayerNorm2d, normalised over the
// channel axis of an NCHW tensor, as one op (fuse_layer_norm2d_pass.py
// replaces the fourteen portable ops of the graph with it).
//
//   espressif::layer_norm2d.out(Tensor x, Tensor weight, Tensor bias,
//                               float eps, *, Tensor(a!) out) -> Tensor(a!)
//
// It computes the graph's arithmetic bit for bit, on the host and on the
// chip: each mean divides its sum by C, the deviations' squares and the
// weight's products are rounded before they are added, as the graph's
// separate pow, mul and add kernels round them. layer_norm2d.cpp is built
// -ffp-contract=off on both, so GCC fuses none of them into madd.s.

#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace layer_norm2d {

// The op's boxed kernel: x [B, C, H, W] float32 contiguous, weight and bias
// [C], out x's shape. Every refusal fails the instruction.
void shim(executorch::runtime::KernelRuntimeContext& ctx,
          executorch::runtime::Span<executorch::runtime::EValue*> stack);

// Columns (one per batch and spatial position) normalised since boot.
uint64_t columns();

}  // namespace layer_norm2d
}  // namespace espressif
