# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused preparation of the per-layer embedding tensor.

The literal PyTorch spelling makes three separate scalar-multiply passes over
the full [layers, ple_dim] tensor, plus RMSNorm and add. On ESP32-S3 the generic
portable multiplies cost far more than the arithmetic. This opaque op preserves
the exact expression while streaming projected and table values once.
"""

import torch
from torch.library import Library


lib = Library("espressif", "FRAGMENT")
lib.define(
    "ple_prepare(Tensor projected, Tensor table, Tensor weight, "
    "float input_scale, float table_scale, float output_scale, float eps) "
    "-> Tensor")
lib.define(
    "ple_prepare.out(Tensor projected, Tensor table, Tensor weight, "
    "float input_scale, float table_scale, float output_scale, float eps, "
    "*, Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")


def _ple_prepare(
        projected, table, weight, input_scale, table_scale, output_scale,
        eps):
    scaled = projected * input_scale
    normalized = (
        weight * scaled
        * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps))
    return (normalized + table * table_scale) * output_scale


def _ple_prepare_out(
        projected, table, weight, input_scale, table_scale, output_scale,
        eps, *, out):
    out.copy_(_ple_prepare(
        projected, table, weight, input_scale, table_scale, output_scale,
        eps))
    return out


impl.impl("ple_prepare", _ple_prepare, "CompositeExplicitAutograd")
impl.impl("ple_prepare.out", _ple_prepare_out, "CompositeExplicitAutograd")

PLE_PREPARE = torch.ops.espressif.ple_prepare.default
