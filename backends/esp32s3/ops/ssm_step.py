# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opaque mutable scan-state ops for the hybrid's static-memory decode step.

The ordinary PyTorch spelling of one Mamba step (`h = state * dA + dBx;
state.copy_(h)`) is semantically right but expensive after functionalization:
ExecuTorch realises the mutation as an output copied back at the end of the
program, so every scan layer keeps BOTH its [d_inner, d_state] state buffer
and its freshly computed value alive to the end, plus the elementwise
temporaries (`dA`, `dBx`, `h * C`) at 12 KB each. Five layers plan 176 KB
for a model whose attention-only twin plans 42 KB, more than QEMU's internal
RAM holds.

These two ops keep the state update inside one kernel, exactly as
`cached_attention` keeps the KV write inside its kernel: the mutable argument
aliases its buffer, nothing is copied back, and the [d_inner, d_state]
temporaries never exist as tensors. The CompositeExplicitAutograd
implementations are the eager/export oracle and repeat the eager decode
step's arithmetic order.

  conv_step   the causal depthwise conv over [w0 w1 w2 | x], then shift the
              three-tap window buffer left by one.
  ssm_step    dA = exp(clamp(dt * A, -30)), h = state * dA + (dt * x) * B,
              state <- h, y = sum_s(h * C) + D * x.
"""

import torch
from torch.library import Library


lib = Library("espressif", "FRAGMENT")
lib.define(
    "conv_step(Tensor(a!) window, Tensor x, Tensor weight, Tensor bias) "
    "-> Tensor")
lib.define(
    "conv_step.out(Tensor(a!) window, Tensor x, Tensor weight, Tensor bias, "
    "*, Tensor(b!) out) -> Tensor(b!)")
lib.define(
    "ssm_step(Tensor dt, Tensor x, Tensor A, Tensor B, Tensor C, Tensor D, "
    "Tensor(a!) state) -> Tensor")
lib.define(
    "ssm_step.out(Tensor dt, Tensor x, Tensor A, Tensor B, Tensor C, "
    "Tensor D, Tensor(a!) state, *, Tensor(b!) out) -> Tensor(b!)")

impl = Library("espressif", "IMPL")


def _conv_step(window, x, weight, bias):
    """window [d, 3], x [d], weight [d, 4], bias [d] -> conv output [d]."""
    taps = torch.cat([window, x.reshape(-1, 1)], dim=1)     # [d, 4]
    out = (taps * weight).sum(dim=1) + bias
    window.copy_(taps[:, 1:])                                # drop the oldest
    return out


def _conv_step_out(window, x, weight, bias, *, out):
    out.copy_(_conv_step(window, x, weight, bias))
    return out


def _ssm_step(dt, x, A, B, C, D, state):
    """dt, x, D [d]; A, state [d, s]; B, C [s] -> y [d], state advanced."""
    # Same -30 clamp as training: identical arithmetic, so a decode step
    # cannot diverge from the sequence path by a different decay floor.
    dA = torch.exp((dt.reshape(-1, 1) * A).clamp(min=-30.0))
    dBx = (dt * x).reshape(-1, 1) * B.reshape(1, -1)
    h = state * dA + dBx
    state.copy_(h)
    return (h * C.reshape(1, -1)).sum(dim=-1) + D * x


def _ssm_step_out(dt, x, A, B, C, D, state, *, out):
    out.copy_(_ssm_step(dt, x, A, B, C, D, state))
    return out


impl.impl("conv_step", _conv_step, "CompositeExplicitAutograd")
impl.impl("conv_step.out", _conv_step_out, "CompositeExplicitAutograd")
impl.impl("ssm_step", _ssm_step, "CompositeExplicitAutograd")
impl.impl("ssm_step.out", _ssm_step_out, "CompositeExplicitAutograd")

CONV_STEP = torch.ops.espressif.conv_step.default
SSM_STEP = torch.ops.espressif.ssm_step.default
