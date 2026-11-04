# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-quantised weights as modules: int4 codes and fp16 group scales held as
buffers, read through torchao's dequantize_affine so the partitioner passes the codes
through instead of quantising the weight again."""

import numpy as np
import torch


# ---- reading the artifact ---------------------------------------------------
class QTensor:
    def __init__(self, codes, scales, group):
        self.codes = codes      # int8 [rows, cols], values in [-7, 7]
        self.scales = scales    # float16 [rows, n_groups]
        self.group = group

    def dequant(self):
        """Mirror the checkpoint exporter's arithmetic exactly: per FIXED group of
        `group`, short tail allowed, qi * fp16-rounded-scale in float32."""
        rows, cols = self.codes.shape
        out = np.empty((rows, cols), dtype=np.float32)
        for gi in range(self.scales.shape[1]):
            a, b = gi * self.group, min((gi + 1) * self.group, cols)
            out[:, a:b] = (self.codes[:, a:b].astype(np.float32)
                           * self.scales[:, gi:gi + 1].astype(np.float32))
        return out


# ---- the pass-through module ------------------------------------------------
class PTQuant(torch.nn.Module):
    """Codes and scales as buffers; `weight` dequantises THROUGH the torchao op so
    export leaves a dequantize_affine the partitioner turns into a PassThroughWeight.
    One implementation serves Linears and Embeddings, and the tied pair shares the
    very same buffer objects."""

    def __init__(self, qt: QTensor):
        super().__init__()
        rows, cols = qt.codes.shape
        self.group = min(qt.group, cols)
        self.logical_cols = cols
        # torchao's block dequant requires complete rectangular blocks, while
        # a pass-through format may end a row in a short group.  Pad that
        # final group with zero codes for eager/export, then slice back to the
        # model width.  The ESP32 backend folds the slice and packs only the
        # logical columns, so this costs no PTD bytes or device arithmetic.
        padded_cols = ((cols + self.group - 1) // self.group) * self.group
        codes = np.zeros((rows, padded_cols), dtype=qt.codes.dtype)
        codes[:, :cols] = qt.codes
        self.register_buffer("codes", torch.from_numpy(codes))
        self.register_buffer("scales",
                             torch.from_numpy(qt.scales.astype(np.float32).copy()))
        self.register_buffer("zeros", torch.zeros_like(self.scales))

    @property
    def weight(self):
        weight = torch.ops.torchao.dequantize_affine(
            self.codes, [1, self.group], self.scales, self.zeros,
            torch.int8, -7, 7)
        # A syntactic full-width slice exports as aten.alias_copy.  That alias
        # is harmless numerically but hides the dequant constant from the
        # embedding partition gate, leaving torchao::dequantize_affine in the
        # portable graph.  Emit a slice only for the ragged case where it carries
        # the real logical-width contract that the backend must fold.
        if self.logical_cols == self.codes.shape[1]:
            return weight
        return weight[:, :self.logical_cols]


class PTLinear(PTQuant):
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class PTTiedPrefixLinear(PTLinear):
    """A logical output-row prefix backed by the full tied embedding buffers."""

    @property
    def weight(self):
        return PTQuant.weight.fget(self)[:self.output_rows]


class PTEmbedding(PTQuant):
    def forward(self, idx):
        return torch.nn.functional.embedding(idx, self.weight)
