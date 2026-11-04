# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opaque mutable cached attention for static-memory ExecuTorch decode.

The ordinary PyTorch spelling is semantically right but pathological after
functionalization: one cache-row update becomes a full-cache index_put + copy,
and static causal masking materializes several more full-cache tensors.  This
custom op keeps the cache shape static for AOT memory planning while preserving
the operation the firmware actually needs: update one row, then scan 0..pos.

Both cache arguments are explicitly mutable.  ExecuTorch's auto-functionalized
edge form records them as BUFFER_MUTATION outputs, then ToOutVarPass emits the
single ``espressif::cached_attention.out`` instruction registered by the ESP32
runner.  The CompositeExplicitAutograd implementation is the eager/export
oracle and deliberately retains the ordinary spelling's operation order.
"""

import torch
from torch.library import Library


lib = Library("espressif", "FRAGMENT")
lib.define(
    "cached_attention(Tensor q, Tensor k, Tensor v, Tensor(a!) kcache, "
    "Tensor(b!) vcache, Tensor pos, float scale_denominator) -> Tensor")
lib.define(
    "cached_attention.out(Tensor q, Tensor k, Tensor v, Tensor(a!) kcache, "
    "Tensor(b!) vcache, Tensor pos, float scale_denominator, *, "
    "Tensor(c!) out) -> Tensor(c!)")

impl = Library("espressif", "IMPL")


def _cached_attention(q, k, v, kcache, vcache, pos, scale_denominator):
    index = pos.reshape(1)
    kcache.index_copy_(1, index, k.unsqueeze(1))
    vcache.index_copy_(1, index, v.unsqueeze(1))
    scores = torch.matmul(
        q.unsqueeze(1), kcache.transpose(1, 2)).squeeze(1)
    scores = scores / scale_denominator
    valid = torch.arange(kcache.shape[1], device=q.device) <= pos
    scores = torch.where(valid.unsqueeze(0), scores, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(
        probabilities.unsqueeze(1), vcache).squeeze(1)


def _cached_attention_out(
        q, k, v, kcache, vcache, pos, scale_denominator, *, out):
    out.copy_(_cached_attention(
        q, k, v, kcache, vcache, pos, scale_denominator))
    return out


impl.impl("cached_attention", _cached_attention, "CompositeExplicitAutograd")
impl.impl(
    "cached_attention.out", _cached_attention_out,
    "CompositeExplicitAutograd")

CACHED_ATTENTION = torch.ops.espressif.cached_attention.default
