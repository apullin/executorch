# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent int8 K cache: each key row is quantised once, when it is written.

The opt-in int8 q@kT (`OP_BMM_DYN_I8`) quantises BOTH operands at execute, so
every decode step re-quantises every cached key row -- 896 rows a token on the
story Glimmer, each read back from an fp32 cache four times the size it needs
to be. Each row gets its own absmax/127 scale, which depends on nothing but
that row, so quantising a key once at write time and keeping its codes and
scale gives the same scores bit for bit.

Two halves, one arithmetic:

  write  `espressif::quantize_rows_i8` turns each row of the new keys into int8
         codes plus an fp32 scale: per-row absmax/127, round half away from
         zero, clamp +-127. The CompositeExplicitAutograd body below is the
         reference and repeats `dynamic_i8_qk`'s key quantisation operation for
         operation; the runtime kernel (espressif_ops.cpp) repeats the
         delegate's `quantize_row_i8`. The graph then writes the codes and the
         scale into their two cache buffers with ordinary index_copy_, which the
         reinplace pass turns into in-place index_put_.

  read   plain ATen, `codes.to(float32) * scales.view(..., 1)` -- the
         dequantised keys the cache means -- so that without the backend the
         graph is still an ordinary, correct program. Esp32Partitioner(
         quantize_qk=True) folds the cast, the scale multiply and the key
         transpose into one OP_BMM_KCACHE_I8, which dots the stored codes
         directly and applies the stored scales in the device's
         `(acc * sq) * sk` order. `cached_i8_qk` is that op's host mirror.

V stays fp32: post-softmax att@v has no quantisation story worth its error.
"""

import torch
from torch.library import Library

# FRAGMENT, not DEF: the "espressif" namespace is shared with rmsnorm_fusion,
# rope_fusion, cached_attention and the other custom ops.
# FLT_MIN: A0's smallest scale with a reciprocal (quant_rules.h kA0MinScale).
A0_MIN_SCALE = 2.0 ** -126

lib = Library("espressif", "FRAGMENT")
lib.define("quantize_rows_i8(Tensor x) -> (Tensor codes, Tensor scales)")
lib.define("quantize_rows_i8.out(Tensor x, *, Tensor(a!) codes, "
           "Tensor(b!) scales) -> (Tensor(a!), Tensor(b!))")

impl = Library("espressif", "IMPL")


def _codes_and_scale(x):
    """`dynamic_i8_qk`'s per-vector quantiser, operation for operation.

    Float codes and a keepdim scale, exactly as the story model
    (story/glimmer_model.py) computes them. float32 `amax / 127.0` is a true
    division in PyTorch (not a multiply
    by a rounded reciprocal), which is what C's `amax / 127.0f` computes too.
    """
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    # 0 for a zero or subnormal scale, so that row's codes are all 0: the
    # device's a0_inverse_scale (runtime/quant_rules.h). A finite row like
    # [FLT_MIN, 0] would otherwise take fl(1 / s) = inf.
    inv = torch.where(scale >= A0_MIN_SCALE, scale.reciprocal(),
                      torch.zeros_like(scale))
    z = x * inv
    codes = torch.sign(z) * torch.floor(torch.abs(z) + 0.5)
    return codes.clamp(-127, 127), scale


def quantize_rows_i8(x):
    """Rows of fp32 ``x[..., K]`` -> (int8 codes ``[..., K]``, fp32 scales ``[...]``).

    An all-zero row gets scale 0 and zero codes, exactly as the delegate's
    quantize_row_i8 treats it, so an unwritten cache row (zeros in both
    buffers) scores 0 just as a zero fp32 row does.
    """
    codes, scale = _codes_and_scale(x)
    return codes.to(torch.int8), scale.squeeze(-1)


def _quantize_rows_i8_out(x, *, codes, scales):
    c, s = quantize_rows_i8(x)
    codes.copy_(c)
    scales.copy_(s)
    return codes, scales


impl.impl("quantize_rows_i8", quantize_rows_i8, "CompositeExplicitAutograd")
impl.impl("quantize_rows_i8.out", _quantize_rows_i8_out,
          "CompositeExplicitAutograd")

QUANTIZE_ROWS_I8 = torch.ops.espressif.quantize_rows_i8.default


def dequantized_keys(codes, scales):
    """The keys an int8 cache means, spelled the way the partitioner folds it.

    ``view`` rather than ``unsqueeze`` for the scale: a portable view_copy is a
    memory-planning alias, an unsqueeze_copy a real copy, should this ever run
    without the delegate.
    """
    return codes.to(torch.float32) * scales.view(*scales.shape, 1)


def cached_i8_qk(q, codes, scales):
    """Host mirror of OP_BMM_KCACHE_I8: `dynamic_i8_qk` with stored key codes.

    The query side is quantised exactly as `dynamic_i8_qk` quantises it. The
    integer products sum in int64, exactly at any length, as the device's
    int32 dot sums them within the gate's dot-length bound (a float32 sum
    would be exact only below 2**24), the sum converts to float32 once, and
    the two scales apply in the device's `(acc * sq) * sk` order. So given
    `(codes, scales) = quantize_rows_i8(k)` this is bit-identical to
    `dynamic_i8_qk(q, k)` (test_kcache_i8.py), the sign of an exact-zero
    score included: +0.0, as the device's int32 zero converts.
    """
    q_codes, q_scale = _codes_and_scale(q)
    acc = torch.matmul(q_codes.to(torch.int64),
                       codes.to(torch.int64).transpose(-1, -2))
    # The low 32 bits, as dot_i8_s3.S returns them: within the gate's bound
    # (MAX_DOT_LEN * 127 * 127 < 2**31) this changes nothing.
    acc = torch.remainder(acc + 2**31, 2**32) - 2**31
    return (acc.to(torch.float32) * q_scale) * scales.unsqueeze(-2)
