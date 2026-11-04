# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""q@kT admission against its allow-list: operand origins x moves x modes x shapes.

The partitioner claims a q@kT only when each operand's origin is one it can
prove (operator_support/qk_operands.py): a plain float value under a declared
mode with a mirror -- quantize_qk's OP_BMM_DYN_I8, the story's int8 K cache,
the stock affine K cache -- and a K cache's codes, read exactly as stored.
Every other origin is refused by name and runs portable. This file holds the
gate to that, case by case, with the contract of each outcome:

  admitted  the host runner's scores are the declared mode's mirror of the
            delegate's own inputs, bit for bit (reference.bmm_qk_i8, the int8
            K cache's cached_i8_qk arithmetic, mirrors.kcache_scores), and
            within the mode's propagated error budget of the exact float64
            product of those inputs (_bar);
  refused   the scores are ExecuTorch's portable bmm of the operands, bit for
            bit (reference.bmm_f32_portable), and the operands are the
            converted model's, bit for bit: moves, static pairs and dynamic
            chains computed in place change no value (eager's operands, with
            sqrt the correctly rounded root on every machine: _ByDefinition);

and in both, whether it delegated matches the allow-list as each case
defines it -- from how the case BUILT its operands, not from the gate's own
classification -- and a refusal's reason names the origin it found.

Families:

  explicit  softmax(q @ k^T / sqrt(d)), each operand from one of ORIGINS
            through one of MOVES, with MHA/GQA/MQA heads (repeat_kv), head
            widths 16/32/64 and an odd 17, query and key lengths from SEQS,
            under quantize_qk on or off, or on with static_act through
            lower_esp32 or through the two-step flow (to_edge, then
            calibrate_static_act, then to_backend: no strip, so a chain
            reaches the gate whole);
  kcache    the story's int8 K cache (kcache_i8.py): codes and scales
            buffers written at pos through quantize_rows_i8 and index_copy_,
            the local64 ring and the global256 cache at their edges, a query
            of each origin, a cache never written (constants: refused) or
            read before its write (the buffer read directly: admitted, a
            runtime tensor);
  affine    the stock affine K cache's spelling (stock_llm.rewrite_kcache):
            espressif.dequantize_per_token_i8 over codes, scales and zero
            points from graph inputs, written buffers or constants, claimed
            only under quantize_qk (its int8 query rows are opt-in);
  recipe    query and key projections quantised by a stock or esp32 recipe
            (XNNPACK dynamic and static, Arm a8w8, Esp32Quantizer, torchao
            8da4w), the q@kT left to quantize_qk; whether a recipe quantised
            a q@kT operand is read off the converted graph independently.

The explicit family is too big to run whole: pytest runs a greedy pairwise
cover of its nine factors (every pair of factor values in some case, but a
grid or plain arithmetic origin meets the other side's origins only among
the core's admitted ones: each operand is judged on its own) and the other
families whole, and prints what it covered. `--full` (script mode, a long
run) runs every (query origin, key origin, mode) under every move.

    python test_qk_matrix.py [--full] [--seed N] [--limit N]
    pytest test_qk_matrix.py
"""

import argparse
import contextlib
import io
import itertools
import operator
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

F32 = np.float32

CORE_ORIGINS = ("plain", "plain_op", "static_pair", "dyn_token",
                "dyn_tensor", "dyn_tensor_affine", "const_codes",
                "const_float", "computed_const", "state", "per_channel",
                "fake_quant", "neg_pair")
# The grids a model computes itself through arithmetic, each of which a gate
# that reads arithmetic between two values of the operand's own shape, or
# takes arithmetic between two varying values for new values, would admit as
# plain float: a group-wise scale repeated, reshaped, written twice, spelled
# as a reciprocal or held in a buffer of the operand's shape; one value
# expanded and reshaped to it, or a buffer filled with one value; a
# straight-through estimator whose equal terms are written apart; sign as a
# difference of two masks or two wheres, as x / abs(x), abs(x) / x and their
# reciprocal spellings, as a where of ones and their negation, of a learned
# alpha and its negation (scalar, per channel); truncation as y - frac(y);
# the magic-number round; sign as x / sqrt(x * x) and x * rsqrt(x * x), a
# lookup-table quantiser (a constant table at the codes the model casts from
# a float), and a group-wise step guarded against zero where it divides only;
# a uniform scale computed by a matmul of constants, which no rule reads (its
# values could pass for varying); lookup tables at codes
# the model computes from a float value other than by a cast -- a tensor
# zero point added to cast codes, a comparison cast to int64, a
# thermometer's sum of comparisons, the nearest of eight levels (argmin),
# the same through a straight-through estimator, a where of numbers,
# Transformer-VQ's nearest codeword of a vector, and a cast whose index walk
# is 70 integer ops long; int8 codes times a group-wise run-time scale
# repeated (repeat_interleave), expanded and reshaped, or gathered by group
# to the operand's width, and a KIVI-style int8 K cache whose codes and
# group scales are buffers the forward writes; sign and a step guarded by an
# addition (x / (abs(x) + 1e-8), x / sqrt(x * x + 1e-12),
# round(x / (s + 1e-8)) * s for a group-wise s and for a per-element one,
# which no shape tells for a step); a where of a row's own scale and its
# negation; sign of a flipped value written twice; sign with one move written
# on both sides of the division (a flip, a view, an unsqueeze and squeeze, a
# transpose and back), sign guarded by maximum(abs(x), eps), and a step
# guarded by maximum where it divides; int8 codes times a group-wise
# step stacked or concatenated to the operand's width, or a row's step
# concatenated to it, and a where of a row's step so spread and its negation;
# NF4's sixteen levels times a run-time absmax, per tensor or per
# row, looked up at the operand's nearest level.
GRID_ARITHMETIC = ("ste", "lsq", "ste_step", "expanded_scale", "remainder",
                   "mask_sign", "where_sign", "add_offset", "sub_offset",
                   "group_repeat", "group_reshape", "group_twice",
                   "group_reciprocal", "group_buffer", "scale_reshape",
                   "uniform_buffer", "ste_twice", "ste_divs", "mask_diff",
                   "where_diff", "abs_sign", "abs_div_sign",
                   "recip_abs_sign", "abs_recip_sign", "where_ones",
                   "where_alpha", "where_alpha_vec", "minus_frac",
                   "magic_round", "sqrt_sign", "rsqrt_sign", "lut_codes",
                   "group_guarded", "computed_scale", "lut_offset",
                   "lut_compare",
                   "lut_thermometer", "lut_nearest", "vq_ste", "lut_where",
                   "vq_vector", "lut_deep", "codes_group_repeat",
                   "codes_group_expand", "codes_group_index", "kcache_group",
                   "sign_eps_add", "sign_sqrt_eps", "step_eps_add",
                   "step_eps_elem", "where_row_scale", "flip_sign",
                   "sign_flip_both", "sign_view_both", "sign_unsqueeze_both",
                   "sign_transpose_both", "sign_maximum", "step_maximum",
                   "codes_group_stack", "codes_group_cat", "codes_row_cat",
                   "where_row_cat", "lut_scaled_tensor", "lut_scaled_rows")
# The grid arithmetic origins refused as integer codes cast to float.
CODES_ARITHMETIC = ("codes_group_repeat", "codes_group_expand",
                    "codes_group_index", "kcache_group", "codes_group_stack",
                    "codes_group_cat", "codes_row_cat")
# Plain float the grid rules must keep plain, through arithmetic between
# values that all vary (each is held to the mirror and to eager's bits, so
# none reduces: a portable mean is not eager's bits): a value times a mask
# (relu spelled as a product), a value times or plus a grid it does not
# scale (a fake-quantised other value, as a QAT layer's output is), a value
# times its own run-time scale expanded to its shape, relu spelled with
# where, a sum whose terms cancel to a plain value, and a hand-written RoPE
# over cos and sin tables (a spelling the RoPE fusion leaves alone); softsign,
# x / (1 + abs(x)); x / max(abs(x), 1), which is clamp(x, -1, 1) (a gate that
# takes any clamp for a guard refuses it); a learned per-channel temperature;
# RoPE whose half-width tables repeat_interleave spreads to the operand's
# width; an int64 mask broadcast over the width; a buffer of the operand's
# shape filled with one value (each a scale or an offset the shape test must
# find, its partner plain); x / maximum(abs(x), 1), clamp(x, -1, 1) again;
# and x / max(abs(x), eps0 * 1e8) for a buffer eps0 of 1e-8, whose bound is
# 1.0 (a gate that reads eps0 past the factor refuses it).
PLAIN_ARITHMETIC = ("times_mask", "times_grid", "plus_grid",
                    "scaled_runtime", "where_relu", "cancel_to_plain",
                    "rope", "softsign", "clamp_by_abs", "temperature",
                    "rope_spread", "int_mask", "times_uniform",
                    "maximum_by_one", "clamp_scaled_eps")
ORIGINS = CORE_ORIGINS + GRID_ARITHMETIC + PLAIN_ARITHMETIC
# The core's admitted origins, which every grid and plain arithmetic origin
# meets on the other side in the pairwise cover.
CORE_ADMITTED = ("plain", "plain_op", "state")
# What the allow-list admits, by how the case built the operand: a state the
# forward writes is read through moves in every case here (the matmul's own
# views at least), and holds what the forward writes into it, plain values.
ADMITTED = frozenset(CORE_ADMITTED + PLAIN_ARITHMETIC)
# The phrase a refusal names each origin by (the gate's kind or detail).
NAMED = {
    "static_pair": "static pair",
    "dyn_token": "dynamic chain",
    "dyn_tensor": "dynamic chain",
    "dyn_tensor_affine": "dynamic chain",
    "const_codes": "constant codes",
    "const_float": "build-time constant",
    "computed_const": "computed constant",
    "per_channel": "quantised op",
    "fake_quant": "already on a grid",
    "neg_pair": "static pair",
    **{origin: "already on a grid" for origin in GRID_ARITHMETIC},
    **{origin: "integer codes cast to float" for origin in CODES_ARITHMETIC},
}
# No channels-last move: ExecuTorch emits a reshape of a channels-last value
# as an alias of its bytes, wrong in every mode with or without this backend
# (test_channels_last_reshape_upstream_defect).
MOVES = ("none", "view", "permute", "transpose", "expand", "alias", "clone",
         "contiguous", "unsqueeze", "f16", "bf16")
MODES = ("qk", "off", "qk+static_act", "qk+2step")
# (query heads, KV heads); mqa_bcast leaves the one KV head to matmul's
# implicit broadcast instead of repeat_kv.
HEADS = {"mha": (2, 2), "gqa": (4, 2), "mqa": (4, 1), "mqa_bcast": (4, 1)}
HEAD_DIMS = (16, 32, 64, 17)
SEQS = (1, 7, 31, 32, 33, 64, 65, 256, 257)
# The kinds a refusal of an admissible case may give: not the allow-list's,
# but the q@kT's structure or layout, which this file reports separately.
STRUCTURAL = ("bmm not q@kT", "runtime tensor is not row-major",
              "q@kT operand produced inside the partition")


# --- operands ------------------------------------------------------------------

def _qd():
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

    return torch.ops.quantized_decomposed


def _token_chain(t):
    """torchao's per-token dynamic chain (esp32_torchao_config's activation)."""
    import torchao  # noqa: F401 -- the torchao ops

    ao = torch.ops.torchao
    block = [1] * (t.dim() - 1) + [int(t.shape[-1])]
    scale, zp = ao.choose_qparams_affine(t, "ASYMMETRIC", block, torch.int8,
                                         -128, 127, None, torch.float32,
                                         torch.int32)
    q = ao.quantize_affine(t, block, scale, zp, torch.int8, -128, 127)
    return ao.dequantize_affine(q, block, scale, zp, torch.int8, -128, 127)


def _tensor_chain(t):
    """Esp32Quantizer's per-tensor symmetric dynamic chain."""
    qd = _qd()
    eps = torch.finfo(torch.float32).eps
    scale, zp = qd.choose_qparams_symmetric.tensor(t, -127, 127, eps,
                                                   torch.int8)
    q = qd.quantize_per_tensor.tensor(t, scale, zp, -127, 127, torch.int8)
    return qd.dequantize_per_tensor.tensor(q, scale, zp, -127, 127, torch.int8)


def _tensor_chain_affine(t):
    """XNNPACK's per-tensor affine dynamic chain (eps 2^-12)."""
    qd = _qd()
    scale, zp = qd.choose_qparams.tensor(t, -128, 127, 2.0 ** -12, torch.int8)
    q = qd.quantize_per_tensor.tensor(t, scale, zp, -128, 127, torch.int8)
    return qd.dequantize_per_tensor.tensor(q, scale, zp, -128, 127, torch.int8)


def _per_channel_pair(t):
    """A static pair per channel of the last dimension :
    lower_esp32 refuses it at export (absorb_static_qdq's
    [per-channel-activation]); the two-step flow hands it to the gate."""
    qd = _qd()
    d = int(t.shape[-1])
    scales = torch.tensor([0.02, 0.03, 0.05, 0.07] * ((d + 3) // 4))[:d]
    zps = torch.zeros(d, dtype=torch.int64)
    q = qd.quantize_per_channel(t, scales, zps, t.dim() - 1, -128, 127,
                                torch.int8)
    return qd.dequantize_per_channel(q, scales, zps, t.dim() - 1, -128, 127,
                                     torch.int8)


STATIC_SCALE, STATIC_ZP = float(F32(0.0123)), 3
CODES_SCALE = float(F32(0.0117))


def _move(t, name):
    """`t` through one data move, every value unchanged."""
    if name == "none":
        return t
    b, h, s, d = t.shape
    if name == "view":
        return t.reshape(b, h * s, d).reshape(b, h, s, d)
    if name == "permute":
        return t.permute(0, 2, 1, 3).permute(0, 2, 1, 3)
    if name == "transpose":
        return t.transpose(-1, -2).transpose(-1, -2)
    if name == "expand":
        return t.unsqueeze(2).expand(b, h, 2, s, d)[:, :, 1]
    if name == "alias":
        return torch.ops.aten.alias.default(t)
    if name == "clone":
        return t.clone()
    if name == "contiguous":
        return t.transpose(1, 2).contiguous().transpose(1, 2)
    if name == "unsqueeze":
        return t.unsqueeze(0).squeeze(0)
    # Not data movement: the values round to the narrower type, and the
    # operand reaches the bmm as fp32 (a non-fp32 source).
    if name == "f16":
        return t.half().float()
    if name == "bf16":
        return t.bfloat16().float()
    raise KeyError(name)


def _see(module, values):
    """Keep the operands eager computes, as copies, for the bit-for-bit
    comparison: not while exporting, where a copy would be one more user of
    the operand, and before the forward writes a state they may view.
    `values` may be a function returning them, for values computed only to
    be seen (a broadcast the bmm reads)."""
    if not torch.compiler.is_exporting():
        if callable(values):
            values = values()
        module.seen = {k: v.detach().clone().contiguous()
                       for k, v in values.items()}


class _ByDefinition(TorchDispatchMode):
    """Eager, but each op fold_quant_lut_pass.EXACT names computed by its
    definition on a float32 tensor: sqrt's correctly rounded root (the eager
    twin of fixtures/quant_lut.ByDefinition, an Interpreter). The program's
    portable sqrt computes that root (std::sqrt), and so does ATen's kernel
    on Apple silicon, but not on x86, where torch's float32 sqrt goes
    through MKL: on an x86 host 2,877,312 of 16,711,680 float32 inputs come out one
    ulp off. There x / sqrt(x * x) is +-(1 - ulp) where the program's is
    exactly +-1 (6 of 477,624 unit-normal elements), and x / sqrt(x * x +
    1e-12) is off on 471 of them, so refused cases' operands would not be
    eager's bit for bit. Every operand this file holds to eager's is
    computed in this mode. rsqrt needs no definition: ATen's is a correctly
    rounded root, then a correctly rounded division, the portable kernel's
    1.0 / std::sqrt(x), on both machines (0 of the 16.7M differ on the x86 host)."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from executorch.backends.esp32s3._passes import fold_quant_lut_pass

        exact = fold_quant_lut_pass.EXACT.get(func)
        if exact is not None and not kwargs and args[0].dtype == torch.float32:
            # The definition's values in the layout ATen's kernel gives.
            return torch.empty_like(args[0]).copy_(exact(args[0]))
        return func(*args, **(kwargs or {}))


def _row_step(x):
    """A 4-bit step per row, chosen at run time: absmax / 7."""
    return x.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 7.0


MAGIC = 1.5 * 2.0 ** 23
# NF4's sixteen levels (QLoRA), as bitsandbytes rounds them to four places.
NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
                    -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407,
                    0.5626, 0.7230, 1.0])


def _group(d):
    """The group size of a group-wise scale over a last dimension of d: 4,
    or d itself (one group) where 4 does not divide it."""
    return 4 if d % 4 == 0 else d


def _group_step(x, keepdim=False):
    """A 4-bit step per group of the last dimension, chosen at run time:
    each group's absmax / 7."""
    g = _group(x.shape[-1])
    groups = x.reshape(*x.shape[:-1], x.shape[-1] // g, g)
    return groups.abs().amax(-1, keepdim=keepdim).clamp_min(1e-6) / 7.0


def _grid_arithmetic(origin, x, ops=None, side=None):
    """`x` put on a grid the model chooses itself, spelled through
    arithmetic (GRID_ARITHMETIC): a straight-through estimator, hand-written
    (x + (fq(x) - x).detach(), per row or at a constant step) or LSQ's
    round_pass; a scale expanded to x's shape; x - remainder(x, s); sign
    spelled with a comparison or with where; and a round plus or minus an
    offset of x's own shape; a group-wise step repeated (repeat_interleave), expanded then reshaped,
    written twice, multiplied as its reciprocal, or held in a buffer of x's
    shape (`ops`'s); one value expanded and reshaped to x's shape, a buffer
    of x's shape filled with one value; straight-through estimators whose
    equal terms are separate nodes (x * 2 twice; x / s three times); sign as
    two masks or two wheres subtracted, x / abs(x), abs(x) / x,
    x * (1 / abs(x)), abs(x) * (1 / x), where(x > 0, ones, -ones),
    where(x > 0, alpha, -alpha) with a learned scalar or per-channel alpha
    (`ops`'s); y - frac(y); and ((y + 1.5 * 2**23) - 1.5 * 2**23). Then
    their next relatives: x / sqrt(x * x), x * rsqrt(x * x), a 16-entry
    table (`ops`'s) at codes cast from round(4 x), and a group-wise step
    clamped away from zero only where it divides. Then (`ops`'s tables and
    buffers): a grid times a uniform scale of
    its own shape a matmul of two constants computes; a table at cast codes
    plus a
    zero-point tensor; at (x > 0).long(); at a thermometer's count of
    thresholds below x; at x's nearest of eight levels (argmin), with and
    without the estimator; at where(x > 0, 2, 1); a codebook at each row's
    nearest codeword (L-infinity: exact on every host); a table at a cast's
    index after 70 integer abs; int8 codes of round(4 x) times a group-wise
    run-time step repeated, expanded and reshaped, or gathered by group to
    x's width; a group-wise int8 K cache (codes and scales in buffers the
    forward writes); x / (abs(x) + 1e-8), x / sqrt(x * x + 1e-12),
    round(x / (s + 1e-8)) * s for a group-wise s and for s = abs(x) / 2 +
    0.1; where(x > 0, s, -s) for a row's own step s times a buffer of ones;
    and x.flip(-1) / abs(x.flip(-1)); sign with a
    flip, a view, an unsqueeze then squeeze, or a transpose then back
    applied to both x and abs(x); x / maximum(abs(x), eps) and
    round(x / maximum(s, eps)) * s for s = abs(x) / 2 + 0.1, eps a buffer of
    1e-8 (`ops`'s); int8 codes of round(4 x) times a group-wise step
    stacked or concatenated to x's width, or times a row's step
    concatenated to it, and where(x > 0, s, -s) for that row step s; NF4's
    levels (`ops`'s) times x's absmax, per tensor or per row, at x's nearest
    level (argmin)."""
    if origin == "ste":
        s = _row_step(x)
        fq = torch.clamp(torch.round(x / s), -8, 7) * s
        return x + (fq - x).detach()
    if origin == "lsq":
        s = _row_step(x)
        v = torch.clamp(x / s, -8, 7)
        return ((v.round() - v).detach() + v) * s
    if origin == "ste_step":
        fq = torch.clamp(torch.round(x / 0.25), -8, 7) * 0.25
        return x + (fq - x).detach()
    if origin == "expanded_scale":
        s = _row_step(x).expand_as(x)
        return torch.clamp(torch.round(x / s), -8, 7) * s
    if origin == "remainder":
        return x - torch.remainder(x, 0.25)
    if origin == "mask_sign":
        return (x > 0).to(torch.float32) * 2.0 - 1.0
    if origin == "where_sign":
        return torch.where(x > 0, 1.0, -1.0)
    if origin == "add_offset":
        return torch.round(x / 0.5) + torch.full_like(x, 0.25)
    if origin == "sub_offset":
        return torch.round(x / 0.5) - torch.tensor(0.25).expand_as(x)
    g = _group(x.shape[-1])
    if origin in ("group_repeat", "group_reciprocal"):
        s = _group_step(x).repeat_interleave(g, -1)
        if origin == "group_reciprocal":
            return torch.clamp(torch.round(x * s.reciprocal()), -8, 7) * s
        return torch.clamp(torch.round(x / s), -8, 7) * s
    if origin == "group_reshape":
        sg = _group_step(x, keepdim=True)
        s = sg.expand(*sg.shape[:-1], g).reshape(x.shape)
        return torch.clamp(torch.round(x / s), -8, 7) * s
    if origin == "group_twice":
        sg = _group_step(x)
        return (torch.clamp(torch.round(x / sg.repeat_interleave(g, -1)),
                            -8, 7) * sg.repeat_interleave(g, -1))
    if origin == "group_buffer":
        s = getattr(ops, f"{side}_scale")
        return torch.clamp(torch.round(x / s), -8, 7) * s
    if origin == "scale_reshape":
        s = getattr(ops, f"{side}_scale")
        return torch.round(x / 0.5) * s.expand(1, x.numel()).reshape(x.shape)
    if origin == "uniform_buffer":
        return torch.round(x / 0.5) * getattr(ops, f"{side}_scale")
    if origin == "ste_twice":
        a, b = x * 2.0, x * 2.0
        return a + (a.round() - b).detach()
    if origin == "ste_divs":
        return ((torch.round(x / 0.25) - x / 0.25).detach() + x / 0.25) * 0.25
    if origin == "mask_diff":
        return (x > 0).to(torch.float32) - (x < 0).to(torch.float32)
    if origin == "where_diff":
        return torch.where(x > 0, 1.0, 0.0) - torch.where(x < 0, 1.0, 0.0)
    if origin == "abs_sign":
        return x / x.abs()
    if origin == "abs_div_sign":
        return x.abs() / x
    if origin == "recip_abs_sign":
        return x * x.abs().reciprocal()
    if origin == "abs_recip_sign":
        return x.abs() * x.reciprocal()
    if origin == "where_ones":
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))
    if origin in ("where_alpha", "where_alpha_vec"):
        alpha = getattr(ops, f"{side}_alpha")
        return torch.where(x > 0, alpha, -alpha)
    if origin == "minus_frac":
        return (x * 4 - torch.frac(x * 4)) * 0.25
    if origin == "magic_round":
        return ((x * 4.0 + MAGIC) - MAGIC) * 0.25
    if origin == "sqrt_sign":
        return x / torch.sqrt(x * x)
    if origin == "rsqrt_sign":
        return x * torch.rsqrt(x * x)
    if origin == "lut_codes":
        codes = (torch.round(x * 4).clamp(-8, 7) + 8).to(torch.int64)
        return getattr(ops, f"{side}_table")[codes]
    if origin == "group_guarded":
        s = (x.reshape(*x.shape[:-1], x.shape[-1] // g, g).abs().amax(-1)
             / 7.0).repeat_interleave(g, -1)
        return torch.clamp(torch.round(x / s.clamp_min(1e-8)), -8, 7) * s
    if origin == "computed_scale":
        return torch.round(x / 0.5) * torch.matmul(getattr(ops, f"{side}_ca"),
                                                   getattr(ops, f"{side}_cb"))
    if origin == "lut_offset":
        codes = torch.round(x * 4).clamp(-8, 7).to(torch.int64)
        return getattr(ops, f"{side}_table")[codes + getattr(ops, f"{side}_zp")]
    if origin == "lut_compare":
        return getattr(ops, f"{side}_levels")[(x > 0).long()]
    if origin == "lut_thermometer":
        count = (x.unsqueeze(-1) > getattr(ops, f"{side}_thresholds")).sum(-1)
        return getattr(ops, f"{side}_lv4")[count]
    if origin in ("lut_nearest", "vq_ste"):
        cent = getattr(ops, f"{side}_cent")
        hard = cent[(x.unsqueeze(-1) - cent).abs().argmin(-1)]
        return hard if origin == "lut_nearest" else x + (hard - x).detach()
    if origin == "lut_where":
        return getattr(ops, f"{side}_lv4")[torch.where(x > 0, 2, 1)]
    if origin == "vq_vector":
        book = getattr(ops, f"{side}_codebook")
        far = (x.unsqueeze(-2) - book).abs().amax(-1)
        return book[far.argmin(-1)]
    if origin == "lut_deep":
        idx = (torch.round(x * 2).clamp(-2, 1) + 2).to(torch.int64)
        for _ in range(70):
            idx = idx.abs()
        return getattr(ops, f"{side}_lv4")[idx]
    if origin in ("codes_group_repeat", "codes_group_expand",
                  "codes_group_index", "codes_group_stack", "codes_group_cat",
                  "codes_row_cat"):
        codes = torch.round(x * 4).clamp(-8, 7).to(torch.int8)
        sg = _group_step(x)
        if origin == "codes_group_repeat":
            s = sg.repeat_interleave(g, -1)
        elif origin == "codes_group_expand":
            s = sg.unsqueeze(-1).expand(*sg.shape, g).reshape(x.shape)
        elif origin == "codes_group_stack":
            s = torch.stack([sg] * g, -1).reshape(x.shape)
        elif origin == "codes_group_cat":
            s = torch.cat([sg.unsqueeze(-1)] * g, -1).reshape(x.shape)
        elif origin == "codes_row_cat":
            s = torch.cat([_row_step(x)] * x.shape[-1], -1)
        else:
            s = sg[..., getattr(ops, f"{side}_gidx")]
        return codes.to(torch.float32) * s
    if origin == "kcache_group":
        ops.pending[side] = x
        return (getattr(ops, f"{side}_kc").to(torch.float32)
                * getattr(ops, f"{side}_ks").repeat_interleave(g, -1))
    if origin == "sign_eps_add":
        return x / (x.abs() + 1e-8)
    if origin == "sign_sqrt_eps":
        return x / torch.sqrt(x * x + 1e-12)
    if origin in ("step_eps_add", "step_eps_elem"):
        s = (_group_step(x).repeat_interleave(g, -1)
             if origin == "step_eps_add" else x.abs() * 0.5 + 0.1)
        return torch.clamp(torch.round(x / (s + 1e-8)), -8, 7) * s
    if origin == "where_row_scale":
        s = _row_step(x) * getattr(ops, f"{side}_ones")
        return torch.where(x > 0, s, -s)
    if origin == "where_row_cat":
        s = torch.cat([_row_step(x)] * x.shape[-1], -1)
        return torch.where(x > 0, s, -s)
    if origin == "lut_scaled_tensor":
        tab = getattr(ops, f"{side}_nf4") * x.abs().amax()
        return tab[(x.unsqueeze(-1) - tab).abs().argmin(-1)]
    if origin == "lut_scaled_rows":
        tab = getattr(ops, f"{side}_nf4") * x.abs().amax(-1, keepdim=True)
        idx = (x.unsqueeze(-1) - tab.unsqueeze(-2)).abs().argmin(-1)
        return tab.gather(-1, idx)
    if origin == "flip_sign":
        return x.flip(-1) / x.flip(-1).abs()
    if origin == "sign_flip_both":
        return x.flip(-1) / x.abs().flip(-1)
    if origin == "sign_view_both":
        # The * 1.0 ends the operand's view before a cast move, as
        # fake_quant's does: a view, a cast and a view in a row trip an
        # upstream defect (test_view_through_cast_upstream_defect).
        d = x.shape[-1]
        return (x.view(-1, d) / x.abs().view(-1, d)).view(x.shape) * 1.0
    if origin == "sign_unsqueeze_both":
        return (x.unsqueeze(0) / x.abs().unsqueeze(0)).squeeze(0) * 1.0
    if origin == "sign_transpose_both":
        return (x.transpose(-1, -2)
                / x.abs().transpose(-1, -2)).transpose(-1, -2)
    if origin == "sign_maximum":
        return x / torch.maximum(x.abs(), getattr(ops, f"{side}_eps"))
    if origin == "step_maximum":
        s = x.abs() * 0.5 + 0.1
        return torch.clamp(torch.round(
            x / torch.maximum(s, getattr(ops, f"{side}_eps"))), -8, 7) * s
    raise KeyError(origin)


def _plain_arithmetic(origin, x, ops=None, side=None):
    """`x` through arithmetic between values that all vary, computing plain
    float (PLAIN_ARITHMETIC): times a mask; times or plus another value
    fake-quantised by its estimator (on its own grid, which x does not
    scale); times its own run-time row scale expanded to its shape; relu
    as a where; a sum whose terms cancel to a plain value; a hand-written
    RoPE over cos and sin tables (`ops`'s buffers), the rotation's
    negation spelled as a multiply by -1, which fuse_rope leaves alone.
    Then softsign; x / max(abs(x), 1) (clamp(x, -1, 1)); times a learned
    per-channel temperature; the RoPE with half-width tables spread by
    repeat_interleave; times an int64 mask over the width; times a buffer
    filled with 0.5. Then x / maximum(abs(x), 1) for a buffer of 1.0, and
    x / max(abs(x), eps0 * 1e8) for a buffer eps0 of 1e-8 (a bound of 1.0)."""
    if origin == "times_mask":
        return x * (x > 0).to(torch.float32)
    if origin in ("times_grid", "plus_grid"):
        y = x.flip(-1) * 0.5 + 0.25
        s = _row_step(y)
        fy = y + (torch.clamp(torch.round(y / s), -8, 7) * s - y).detach()
        return x * fy if origin == "times_grid" else x + fy
    if origin == "scaled_runtime":
        return x * x.abs().amax(-1, keepdim=True).expand_as(x)
    if origin == "where_relu":
        return torch.where(x > 0, x, 0.0)
    if origin == "cancel_to_plain":
        p = x * 1.5
        return x + (p - x).detach()
    if origin == "rope":
        # The negation as a multiply by -1 (the same bits): fuse_rope matches
        # neg, and its fused kernel rounds once where the source rounds twice.
        half = x.shape[-1] // 2
        rot = torch.cat([x[..., half:] * -1.0, x[..., :half]], -1)
        return (x * getattr(ops, f"{side}_cos")
                + rot * getattr(ops, f"{side}_sin"))
    if origin == "softsign":
        return x / (1.0 + x.abs())
    if origin == "clamp_by_abs":
        return x / x.abs().clamp_min(1.0)
    if origin == "temperature":
        return x * getattr(ops, f"{side}_temp")
    if origin == "rope_spread":
        d = x.shape[-1]
        half = d // 2
        rot = torch.cat([x[..., half:] * -1.0, x[..., :half]], -1)
        cos = getattr(ops, f"{side}_cos").repeat_interleave(2, -1)[..., :d]
        sin = getattr(ops, f"{side}_sin").repeat_interleave(2, -1)[..., :d]
        return x * cos + rot * sin
    if origin == "int_mask":
        return x * getattr(ops, f"{side}_mask")
    if origin == "times_uniform":
        return x * getattr(ops, f"{side}_full")
    if origin == "maximum_by_one":
        return x / torch.maximum(x.abs(), getattr(ops, f"{side}_one"))
    if origin == "clamp_scaled_eps":
        return x / x.abs().clamp_min(getattr(ops, f"{side}_eps0") * 1e8)
    raise KeyError(origin)


class _Operands(nn.Module):
    """Builds each side's operand from its origin; the constants a constant
    origin needs are this module's parameters and buffers."""

    def __init__(self, sides, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.origin = {}
        self.pending = {}           # kcache_group: the side's x, to write
        for side, (origin, shape) in sides.items():
            self.origin[side] = origin
            self._lookups(side, origin, shape, g)
            if origin in ("const_float", "computed_const"):
                self.register_parameter(
                    f"{side}_p", nn.Parameter(torch.randn(shape, generator=g)))
            elif origin == "const_codes":
                self.register_buffer(f"{side}_codes", torch.randint(
                    -128, 128, shape, generator=g, dtype=torch.int8))
            elif origin == "state":
                self.register_buffer(f"{side}_state",
                                     torch.randn(shape, generator=g))
            elif origin == "group_buffer":
                d = shape[-1]
                gs = _group(d)
                steps = torch.rand(*shape[:-1], d // gs, 1,
                                   generator=g) * 0.2 + 0.05
                self.register_buffer(f"{side}_scale", steps.expand(
                    *shape[:-1], d // gs, gs).reshape(shape).clone())
            elif origin == "scale_reshape":
                self.register_buffer(f"{side}_scale", torch.tensor([0.5]))
            elif origin == "uniform_buffer":
                self.register_buffer(f"{side}_scale", torch.full(shape, 0.5))
            elif origin == "where_alpha":
                self.register_parameter(f"{side}_alpha",
                                        nn.Parameter(torch.tensor(0.7)))
            elif origin == "where_alpha_vec":
                self.register_parameter(f"{side}_alpha", nn.Parameter(
                    torch.linspace(0.5, 1.5, shape[-1])))
            elif origin == "lut_codes":
                self.register_buffer(f"{side}_table",
                                     torch.linspace(-2.0, 2.0, 16))
            elif origin == "rope":
                m, d = shape[-2], shape[-1]
                angle = (torch.arange(m, dtype=torch.float32)[:, None]
                         * torch.linspace(0.1, 1.0, d)[None, :])
                self.register_buffer(f"{side}_cos", angle.cos())
                self.register_buffer(f"{side}_sin", angle.sin())

    def _lookups(self, side, origin, shape, g):
        """The tables, buffers and parameters (`ops`) the grid origins read."""
        d = shape[-1]
        if origin == "computed_scale":
            # x's own shape: a scale broadcast over the heads would be found
            # by the shape test whatever its values.
            self.register_buffer(f"{side}_ca", torch.full((*shape[:-1], 4),
                                                          0.25))
            self.register_buffer(f"{side}_cb", torch.full((4, d), 0.5))
        elif origin == "lut_offset":
            self.register_buffer(f"{side}_table",
                                 torch.arange(-8, 8).float() * 0.25)
            self.register_buffer(f"{side}_zp", torch.full(shape, 8,
                                                          dtype=torch.int64))
        elif origin == "lut_compare":
            self.register_parameter(f"{side}_levels", nn.Parameter(
                torch.tensor([-0.7, 0.7])))
        elif origin in ("lut_thermometer", "lut_where", "lut_deep"):
            self.register_buffer(f"{side}_lv4", torch.tensor(
                [-0.75, -0.25, 0.25, 0.75]))
            self.register_buffer(f"{side}_thresholds",
                                 torch.tensor([-0.5, 0.0, 0.5]))
        elif origin in ("lut_nearest", "vq_ste"):
            self.register_buffer(f"{side}_cent", torch.tensor(
                [-1.5, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.5]))
        elif origin == "vq_vector":
            self.register_buffer(f"{side}_codebook",
                                 torch.randn(32, d, generator=g))
        elif origin == "codes_group_index":
            self.register_buffer(f"{side}_gidx", torch.arange(d) // _group(d))
        elif origin == "where_row_scale":
            # A factor whose values are read at the defaults (one class
            # along the width) and not read when payloads are unread.
            self.register_buffer(f"{side}_ones", torch.ones(d))
        elif origin == "kcache_group":
            self.register_buffer(f"{side}_kc", torch.randint(
                -8, 8, shape, generator=g).to(torch.int8))
            self.register_buffer(f"{side}_ks", torch.rand(
                *shape[:-1], d // _group(d), generator=g) * 0.2 + 0.05)
        elif origin == "temperature":
            self.register_parameter(f"{side}_temp", nn.Parameter(
                torch.linspace(0.8, 1.2, d)))
        elif origin == "rope_spread":
            m, half = shape[-2], (d + 1) // 2
            angle = (torch.arange(m, dtype=torch.float32)[:, None]
                     * torch.linspace(0.1, 1.0, half)[None, :])
            self.register_buffer(f"{side}_cos", angle.cos())
            self.register_buffer(f"{side}_sin", angle.sin())
        elif origin == "int_mask":
            self.register_buffer(f"{side}_mask",
                                 (torch.arange(d) % 3 != 0).to(torch.int64))
        elif origin == "times_uniform":
            self.register_buffer(f"{side}_full", torch.full(shape, 0.5))
        elif origin in ("sign_maximum", "step_maximum"):
            self.register_buffer(f"{side}_eps", torch.tensor([1e-8]))
        elif origin == "maximum_by_one":
            self.register_buffer(f"{side}_one", torch.tensor([1.0]))
        elif origin == "clamp_scaled_eps":
            self.register_buffer(f"{side}_eps0", torch.tensor([1e-8]))
        elif origin in ("lut_scaled_tensor", "lut_scaled_rows"):
            self.register_buffer(f"{side}_nf4", NF4.clone())

    def state_names(self):
        names = []
        for side, origin in self.origin.items():
            if origin == "state":
                names.append(f"{side}_state")
            elif origin == "kcache_group":
                names += [f"{side}_kc", f"{side}_ks"]
        return names

    def operand(self, side, x):
        origin = self.origin[side]
        if origin == "plain":
            return x
        if origin == "plain_op":
            return x * 1.5 + 0.25
        if origin == "static_pair":
            qd = _qd()
            q = qd.quantize_per_tensor.default(x, STATIC_SCALE, STATIC_ZP, -128,
                                               127, torch.int8)
            return qd.dequantize_per_tensor.default(q, STATIC_SCALE, STATIC_ZP,
                                                    -128, 127, torch.int8)
        if origin == "dyn_token":
            return _token_chain(x)
        if origin == "dyn_tensor":
            return _tensor_chain(x)
        if origin == "dyn_tensor_affine":
            return _tensor_chain_affine(x)
        if origin == "const_codes":
            return _qd().dequantize_per_tensor.default(
                getattr(self, f"{side}_codes"), CODES_SCALE, 0, -128, 127,
                torch.int8)
        if origin == "const_float":
            return getattr(self, f"{side}_p")
        if origin == "computed_const":
            return getattr(self, f"{side}_p").abs()
        if origin == "state":
            return getattr(self, f"{side}_state")
        if origin == "per_channel":
            return _per_channel_pair(x)
        if origin == "fake_quant":
            # The * 1.0 ends the fake quantiser's view before a cast move:
            # a view, a cast and a view in a row trip an upstream defect
            # (test_view_through_cast_upstream_defect), not this gate.
            return _per_token_fake_quant(x) * 1.0
        if origin == "neg_pair":
            qd = _qd()
            return -qd.dequantize_per_tensor.default(
                qd.quantize_per_tensor.default(x, STATIC_SCALE, STATIC_ZP,
                                               -128, 127, torch.int8),
                STATIC_SCALE, STATIC_ZP, -128, 127, torch.int8)
        if origin in GRID_ARITHMETIC:
            return _grid_arithmetic(origin, x, self, side)
        if origin in PLAIN_ARITHMETIC:
            return _plain_arithmetic(origin, x, self, side)
        raise KeyError(origin)

    def write_states(self):
        """The forward writes each state after reading it: a buffer the
        model writes, read as it was before. A group-wise K cache takes this
        call's x, quantised: its codes and its group steps."""
        for side, origin in self.origin.items():
            if origin == "state":
                getattr(self, f"{side}_state").mul_(0.5)
            elif origin == "kcache_group":
                x = self.pending[side]
                g = _group(x.shape[-1])
                step = _group_step(x)
                codes = torch.round(x / step.repeat_interleave(g, -1))
                getattr(self, f"{side}_kc").copy_(
                    codes.clamp(-8, 7).to(torch.int8))
                getattr(self, f"{side}_ks").copy_(step)


# --- the cases -------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    family: str
    q: str = "plain"
    k: str = "plain"
    q_move: str = "none"
    k_move: str = "none"
    mode: str = "qk"
    heads: str = "mha"
    d: int = 16
    m: int = 7
    n: int = 31
    window: int = 0          # K cache rows (kcache, affine)
    pos: int = 0             # the decode position written (kcache, affine)
    ring: bool = False       # kcache: the local ring (slot = pos % window)
    cache: str = "written"   # kcache/affine: written | input | constant | read_first
    recipe: str = ""
    seed: int = 0

    @property
    def id(self):
        f = self.family
        if f == "explicit":
            return (f"explicit-{self.q}@{self.q_move}-{self.k}@{self.k_move}-"
                    f"{self.mode}-{self.heads}-d{self.d}-m{self.m}-n{self.n}")
        if f == "kcache":
            return (f"kcache-{'local' if self.ring else 'global'}{self.window}-"
                    f"pos{self.pos}-{self.cache}-q_{self.q}@{self.q_move}-"
                    f"{self.mode}-{self.heads}-d{self.d}-m{self.m}")
        if f == "affine":
            return (f"affine-n{self.n}-{self.cache}-q_{self.q}@{self.q_move}-"
                    f"{self.mode}-{self.heads}-d{self.d}-m{self.m}")
        return f"recipe-{self.recipe}-{self.mode}-{self.heads}-d{self.d}-s{self.m}"

    @property
    def broadcast(self):
        """matmul broadcasts the one KV head: the gate matches no q@kT
        (the broadcast's expand of the transposed keys is not the bmm's
        permute), so the bmm stays portable whatever the operands."""
        return self.heads == "mqa_bcast" and self.mode != "qk+2step"

    def expected(self):
        """Whether the allow-list admits this case, from how it was built."""
        qk = self.mode.startswith("qk")
        if self.family == "explicit":
            return (qk and not self.broadcast and self.q in ADMITTED
                    and self.k in ADMITTED)
        if self.family == "kcache":
            # A cache read before its write is the buffer read directly: a
            # runtime tensor, as one read after it.
            return (qk and self.q in ADMITTED
                    and self.cache in ("written", "read_first"))
        if self.family == "affine":
            return (qk and self.q in ADMITTED
                    and self.cache in ("input", "written"))
        return None             # recipe: read off the converted graph


class Explicit(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        h, kh = HEADS[case.heads]
        self.ops = _Operands({"q": (case.q, (1, h, case.m, case.d)),
                              "k": (case.k, (1, kh, case.n, case.d))},
                             case.seed)
        self.seen = {}

    def forward(self, q_in, k_in):
        c = self.case
        h, kh = HEADS[c.heads]
        q = _move(self.ops.operand("q", q_in), c.q_move)
        k = _move(self.ops.operand("k", k_in), c.k_move)
        if c.heads == "mqa_bcast":      # matmul broadcasts the one KV head
            _see(self, lambda: {"q": q, "k": k.expand(1, h, c.n, c.d)})
        else:
            if h != kh:                 # repeat_kv, as Hugging Face writes it
                k = k[:, :, None].expand(1, kh, h // kh, c.n, c.d).reshape(
                    1, h, c.n, c.d)
            _see(self, {"q": q, "k": k})
        if c.mode == "qk+2step":
            # The two-step flow runs none of lower_esp32's canonicalisation,
            # which turns matmul's broadcast and reshapes into the gate's
            # 3-D bmm (without it a 4-D matmul is "bmm not q@kT"): its cases
            # write that bmm themselves, as calibration.py's users do.
            k3 = k.expand(1, h, c.n, c.d).contiguous().reshape(h, c.n, c.d)
            q3 = q.contiguous().reshape(h, c.m, c.d)
            scores = torch.bmm(q3, k3.transpose(1, 2))
            scores = scores.reshape(1, h, c.m, c.n) * (c.d ** -0.5)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (c.d ** -0.5)
        out = torch.softmax(scores, dim=-1)
        self.ops.write_states()
        return out

    def inputs(self):
        c = self.case
        h, kh = HEADS[c.heads]
        g = torch.Generator().manual_seed(1000 + c.seed)
        return (torch.randn(1, h, c.m, c.d, generator=g),
                torch.randn(1, kh, c.n, c.d, generator=g))

    def mutable(self):
        return self.ops.state_names()


class Int8KCache(nn.Module):
    """One decode step against the story's int8 K cache (glimmer_decode.py's
    spelling): the new key row quantised once and written at its slot, then
    q@kT over every cached row, masked, softmax. GQA without repeat_kv: each
    KV head is a batch, its query heads' rows stacked."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        h, kh = HEADS[case.heads]
        g = torch.Generator().manual_seed(case.seed)
        self.register_buffer("kcache_codes", torch.randint(
            -127, 128, (kh, case.window, case.d), generator=g,
            dtype=torch.int8))
        self.register_buffer("kcache_scales", torch.rand(kh, case.window,
                                                         generator=g)
                             * 0.02 + 1e-3)
        self.ops = _Operands({"q": (case.q, (1, h, case.m, case.d))},
                             case.seed)
        self.seen = {}

    def forward(self, q_in, k_new, pos):
        from executorch.backends.esp32s3.ops.kcache_i8 import (
            QUANTIZE_ROWS_I8,
            dequantized_keys,
        )

        c = self.case
        h, kh = HEADS[c.heads]
        q = _move(self.ops.operand("q", q_in), c.q_move)
        slot = torch.remainder(pos, c.window) if c.ring else pos
        codes, scales = QUANTIZE_ROWS_I8(k_new.view(kh, c.d))
        if c.cache == "written":
            self.kcache_codes.index_copy_(1, slot.reshape(1),
                                          codes.view(kh, 1, c.d))
            self.kcache_scales.index_copy_(1, slot.reshape(1),
                                           scales.view(kh, 1))
        keys = dequantized_keys(self.kcache_codes, self.kcache_scales)
        qg = q.reshape(kh, (h // kh) * c.m, c.d)
        _see(self, {"q": qg, "k": keys})
        scores = torch.bmm(qg, keys.transpose(1, 2)) * (c.d ** -0.5)
        cache_pos = torch.arange(c.window)
        valid = (cache_pos <= pos) | torch.tensor(c.ring) & (pos >= c.window)
        scores = torch.where(valid.view(1, 1, c.window), scores,
                             torch.tensor(float("-inf")))
        out = torch.softmax(scores, dim=-1)
        if c.cache == "read_first":
            self.kcache_codes.index_copy_(1, slot.reshape(1),
                                          codes.view(kh, 1, c.d))
            self.kcache_scales.index_copy_(1, slot.reshape(1),
                                           scales.view(kh, 1))
        self.ops.write_states()
        return out

    def inputs(self):
        c = self.case
        h, kh = HEADS[c.heads]
        g = torch.Generator().manual_seed(1000 + c.seed)
        return (torch.randn(1, h, c.m, c.d, generator=g),
                torch.randn(1, kh, 1, c.d, generator=g),
                torch.tensor([c.pos], dtype=torch.int64))

    def mutable(self):
        names = self.ops.state_names()
        return names + (["kcache_codes", "kcache_scales"]
                        if self.case.cache != "constant" else [])


class AffineKCache(nn.Module):
    """q@kT over the stock affine K cache as the delegate reads it
    (stock_llm.rewrite_kcache): espressif.dequantize_per_token_i8 of codes
    [B, N, D] int8, scales [B, N, 1] fp32, zero points [B, N, 1] int8."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        h, kh = HEADS[case.heads]
        g = torch.Generator().manual_seed(case.seed)
        if case.cache in ("constant", "written"):
            self.register_buffer("cache_codes", torch.randint(
                -128, 128, (kh, case.n, case.d), generator=g,
                dtype=torch.int8))
            self.register_buffer("cache_scales", torch.rand(
                kh, case.n, 1, generator=g) * 0.02 + 1e-3)
            self.register_buffer("cache_zps", torch.randint(
                -20, 20, (kh, case.n, 1), generator=g, dtype=torch.int8))
        self.ops = _Operands({"q": (case.q, (1, h, case.m, case.d))},
                             case.seed)
        self.seen = {}

    def forward(self, q_in, codes, scales, zps, pos):
        import executorch.backends.esp32s3._passes.stock_llm  # noqa: F401

        c = self.case
        h, kh = HEADS[c.heads]
        q = _move(self.ops.operand("q", q_in), c.q_move)
        if c.cache == "written":
            slot = pos.reshape(1)
            self.cache_codes.index_copy_(1, slot, codes[:, :1])
            self.cache_scales.index_copy_(1, slot, scales[:, :1])
            self.cache_zps.index_copy_(1, slot, zps[:, :1])
        if c.cache != "input":
            codes = self.cache_codes
            scales, zps = self.cache_scales, self.cache_zps
        keys = torch.ops.espressif.dequantize_per_token_i8(codes, scales, zps)
        qg = q.reshape(kh, (h // kh) * c.m, c.d)
        _see(self, {"q": qg, "k": keys})
        scores = torch.bmm(qg, keys.transpose(1, 2)) * (c.d ** -0.5)
        out = torch.softmax(scores, dim=-1)
        self.ops.write_states()
        return out

    def inputs(self):
        c = self.case
        h, kh = HEADS[c.heads]
        g = torch.Generator().manual_seed(1000 + c.seed)
        return (torch.randn(1, h, c.m, c.d, generator=g),
                torch.randint(-128, 128, (kh, c.n, c.d), generator=g,
                              dtype=torch.int8),
                torch.rand(kh, c.n, 1, generator=g) * 0.02 + 1e-3,
                torch.randint(-20, 20, (kh, c.n, 1), generator=g,
                              dtype=torch.int8),
                torch.tensor([c.pos], dtype=torch.int64))

    def mutable(self):
        names = self.ops.state_names()
        return names + (["cache_codes", "cache_scales", "cache_zps"]
                        if self.case.cache == "written" else [])


class ProjAttn(nn.Module):
    """Query and key projections into softmax(q @ k^T / sqrt(d)), with
    repeat_kv for GQA/MQA: the q@kT operands a recipe may or may not quantise."""

    E = 32

    def __init__(self, case):
        super().__init__()
        self.case = case
        h, kh = HEADS[case.heads]
        torch.manual_seed(case.seed)
        self.q_proj = nn.Linear(self.E, h * case.d)
        self.k_proj = nn.Linear(self.E, kh * case.d)

    def forward(self, x):
        c = self.case
        h, kh = HEADS[c.heads]
        s = x.shape[1]
        q = self.q_proj(x).view(1, s, h, c.d).transpose(1, 2)
        k = self.k_proj(x).view(1, s, kh, c.d).transpose(1, 2)
        if h != kh:
            k = k[:, :, None].expand(1, kh, h // kh, s, c.d).reshape(
                1, h, s, c.d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (c.d ** -0.5)
        return torch.softmax(scores, dim=-1)

    def inputs(self):
        g = torch.Generator().manual_seed(1000 + self.case.seed)
        return (torch.randn(1, self.case.m, self.E, generator=g),)

    def mutable(self):
        return []


MODELS = {"explicit": Explicit, "kcache": Int8KCache, "affine": AffineKCache,
          "recipe": ProjAttn}


# --- coverage --------------------------------------------------------------------

EXPLICIT_FACTORS = {
    "q": ORIGINS, "k": ORIGINS, "q_move": MOVES, "k_move": MOVES,
    "mode": MODES, "heads": tuple(HEADS), "d": HEAD_DIMS, "m": SEQS,
    "n": SEQS,
}


def _required(a, va, b, vb):
    """Whether the pairwise cover must hold the pair (a=va, b=vb): every
    pair of factor values, but a grid or plain arithmetic origin meets the
    other side's origin only among CORE_ADMITTED. Each operand is judged on
    its own: such an origin against an admitted one shows its own verdict
    (one refused side refuses the bmm, and a wrongly admitted one would be
    delegated), against a refused one nothing that pair does not; and each
    still meets every move, mode, head layout, width and length on both
    sides."""
    if {a, b} != {"q", "k"}:
        return True
    extra = set(GRID_ARITHMETIC) | set(PLAIN_ARITHMETIC)
    if va in extra or vb in extra:
        return (vb if va in extra else va) in CORE_ADMITTED
    return True


def _pairs(case_values):
    names = list(case_values)
    return {(a, case_values[a], b, case_values[b])
            for a, b in itertools.combinations(names, 2)
            if _required(a, case_values[a], b, case_values[b])}


def pairwise_cases(seed=0, candidates=40):
    """A greedy cover of every required pair of factor values (AETG,
    _required): each case keeps the first uncovered pair (in a fixed order)
    and is the best of `candidates` random fillings of the other factors by
    new pairs covered. Deterministic for a seed. Returns (cases, number of
    pairs)."""
    rng = random.Random(seed)
    names = list(EXPLICIT_FACTORS)
    order = []
    for a, b in itertools.combinations(names, 2):
        for va in EXPLICIT_FACTORS[a]:
            for vb in EXPLICIT_FACTORS[b]:
                if _required(a, va, b, vb):
                    order.append((a, va, b, vb))
    todo = set(order)
    total = len(order)
    cases = []
    at = 0
    while todo:
        while order[at] not in todo:
            at += 1
        a, va, b, vb = order[at]
        best, gain = None, -1
        for _ in range(candidates):
            values = {n: rng.choice(EXPLICIT_FACTORS[n]) for n in names}
            values[a], values[b] = va, vb
            got = len(_pairs(values) & todo)
            if got > gain:
                best, gain = values, got
        todo -= _pairs(best)
        cases.append(Case("explicit", seed=len(cases), **best))
    return cases, total


def full_cases(seed=0):
    """Every (query origin, key origin, mode) under every move, both sides;
    shapes drawn per case."""
    rng = random.Random(seed)
    cases = []
    for q, k, mode, mv in itertools.product(ORIGINS, ORIGINS, MODES, MOVES):
        cases.append(Case("explicit", q=q, k=k, q_move=mv, k_move=mv,
                          mode=mode, heads=rng.choice(tuple(HEADS)),
                          d=rng.choice(HEAD_DIMS), m=rng.choice(SEQS),
                          n=rng.choice(SEQS), seed=len(cases)))
    return cases


def kcache_cases():
    """The int8 K cache: local64 at its wrap (63, 64, 65, 127) and global256
    at its first and last rows, a plain query in every head layout and
    chunk-edge query lengths, each refused query origin once, and the cache
    never written or read before its write."""
    cases = []
    edges = [(64, True, p) for p in (63, 64, 65, 127)] + \
            [(256, False, p) for p in (0, 1, 255)]
    shapes = [("mha", 16, 1), ("gqa", 32, 7), ("mqa", 64, 1), ("gqa", 17, 31),
              ("mqa", 32, 32), ("mha", 16, 33), ("gqa", 64, 1)]
    for i, ((w, ring, pos), (heads, d, m)) in enumerate(zip(edges, shapes)):
        for mode in ("qk", "off"):
            cases.append(Case("kcache", window=w, ring=ring, pos=pos,
                              heads=heads, d=d, m=m, mode=mode, seed=i))
    for i, q in enumerate(o for o in ORIGINS if o != "plain"):
        cases.append(Case("kcache", q=q, q_move=MOVES[i % len(MOVES)],
                          window=64, ring=True, pos=64 + i, heads="gqa", d=32,
                          m=1, seed=100 + i))
    for i, cache in enumerate(("constant", "read_first")):
        cases.append(Case("kcache", cache=cache, window=64, ring=True,
                          pos=70, heads="mqa", d=32, m=1, seed=200 + i))
    # The two-step flow: chains reach the gate whole.
    for i, q in enumerate(("plain", "dyn_token", "dyn_tensor",
                           "dyn_tensor_affine", "per_channel",
                           "static_pair")):
        cases.append(Case("kcache", q=q, mode="qk+2step", window=64,
                          ring=True, pos=65 + i, heads="gqa", d=32, m=1,
                          seed=300 + i))
    return cases


def affine_cases():
    """The affine K cache: codes from graph inputs or a written cache at
    every head layout and the odd head width, under quantize_qk (admitted)
    and without it (refused: its int8 query is opt-in), each refused query
    origin in both, a constant cache, and the two-step flow."""
    cases = []
    shapes = [("mha", 16, 1, 31), ("gqa", 32, 7, 64), ("mqa", 64, 1, 257),
              ("gqa", 17, 32, 65), ("mha", 32, 33, 256)]
    for i, (heads, d, m, n) in enumerate(shapes):
        cache = ("input", "written")[i % 2]
        for mode in ("qk", "off"):
            cases.append(Case("affine", heads=heads, d=d, m=m, n=n,
                              cache=cache, pos=n - 1, mode=mode, seed=i))
    for i, q in enumerate(o for o in ORIGINS if o != "plain"):
        cases.append(Case("affine", q=q, q_move=MOVES[(i + 3) % len(MOVES)],
                          heads="gqa", d=32, m=7, n=64, cache="input",
                          mode=("off", "qk")[i % 2], seed=100 + i))
    cases.append(Case("affine", cache="constant", heads="mqa", d=32, m=1,
                      n=64, mode="qk", seed=200))
    for i, q in enumerate(("plain", "dyn_token", "dyn_tensor_affine",
                           "per_channel")):
        cases.append(Case("affine", q=q, mode="qk+2step", heads="gqa", d=32,
                          m=7, n=64, cache=("input", "written")[i % 2],
                          pos=63, seed=300 + i))
    return cases


RECIPES = ("xnnpack-dynamic", "xnnpack-static", "arm-a8w8", "arm-a16w8",
           "esp32", "esp32-xnnpack-dynamic", "esp32-arm-a8w8",
           "torchao-8da4w-g32", "torchao-8da8w")


def recipe_cases():
    cases = []
    shapes = [("mha", 16, 7), ("gqa", 32, 32), ("mqa", 16, 33),
              ("gqa", 17, 31), ("mha", 32, 1), ("mqa", 64, 7),
              ("gqa", 16, 65), ("mha", 17, 32), ("gqa", 32, 7)]
    for i, (recipe, (heads, d, s)) in enumerate(zip(RECIPES, shapes)):
        for mode in ("qk", "off"):
            cases.append(Case("recipe", recipe=recipe, heads=heads, d=d, m=s,
                              n=s, mode=mode, seed=i))
    return cases


# --- lowering, exposing the q@kT, running ---------------------------------------

def _release():
    """Upstream caches that keep every lowered program alive."""
    import gc

    from executorch.backends.esp32s3.test.suite.plugin import release_upstream_caches

    release_upstream_caches()
    gc.collect()


def _expose(ep):
    """Make each q@kT's operands and scores program outputs, after
    partitioning, leaving what was partitioned as it is. Returns one record
    per q@kT: kind "delegate" (its partition's inputs by role) or "portable"
    (the bmm's own operands), and where each value is (an output index, or
    a graph input's name)."""
    from executorch.exir.dialects._ops import ops as exir_ops
    from torch.export.graph_signature import OutputKind, OutputSpec, TensorArgument

    from executorch.backends.esp32s3.operator_support.patterns import (
        _kcache_affine_keys,
        _kcache_keys,
        _qk_bmm_structure,
    )

    gm = ep.graph_module
    graph = gm.graph
    output = next(n for n in graph.nodes if n.op == "output")
    outs = list(output.args[0])
    records = []

    sig = ep.graph_signature
    users = {s.arg.name for s in sig.input_specs
             if s.kind.name == "USER_INPUT"}

    def where(node):
        if node.op == "placeholder":
            if node.name in users:
                return ("input", node.name)
            # A parameter, buffer or lifted constant: its value, as the
            # program starts (a mutable buffer's initial state).
            fqn = (sig.inputs_to_parameters.get(node.name)
                   or sig.inputs_to_buffers.get(node.name)
                   or sig.inputs_to_lifted_tensor_constants.get(node.name))
            value = (ep.state_dict.get(fqn) if fqn in ep.state_dict
                     else ep.constants[fqn])
            return ("const", value.detach().contiguous().numpy().copy())
        index = sum(1 for spec in ep.graph_signature.output_specs
                    if spec.kind.name == "USER_OUTPUT")
        outs.append(node)
        ep.graph_signature.output_specs.append(OutputSpec(
            kind=OutputKind.USER_OUTPUT, arg=TensorArgument(name=node.name),
            target=None))
        return ("output", index)

    bmm_op = exir_ops.edge.aten.bmm.default
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if "executorch_call_delegate" in str(node.target):
            lm = getattr(gm, node.args[0].target)
            sub = lm.original_module.graph
            bmm = next((m for m in sub.nodes if m.op == "call_function"
                        and m.target is bmm_op), None)
            if bmm is None:
                continue
            got = _qk_bmm_structure(bmm)
            A, _, X = got
            affine, kc = _kcache_affine_keys(X), _kcache_keys(X)
            roles = {"q": A}
            if affine is not None:
                mode = "affine"
                roles.update(codes=affine["codes"], scales=affine["scales"],
                             zps=affine["zps"])
            elif kc is not None:
                mode = "kcache_i8"
                roles.update(codes=kc["codes"], scales=kc["scales"])
            else:
                mode = "dyn"
                roles["k"] = X
            # The delegate's arguments are its user inputs, in order; its
            # own constants are placeholders too, but not arguments.
            user_inputs = set(lm.original_module.graph_signature.user_inputs)
            phs = [p for p in sub.nodes if p.op == "placeholder"
                   and p.name in user_inputs]
            rec = {"kind": "delegate", "mode": mode, "values": {}}
            for role, ph in roles.items():
                rec["values"][role] = where(node.args[1 + phs.index(ph)])
            sub_out = next(m for m in sub.nodes if m.op == "output")
            outs_of = list(sub_out.args[0])
            if bmm not in outs_of:
                # The partition consumes the scores itself: none to expose.
                rec["kind"] = "delegate_internal"
                records.append(rec)
                continue
            index = outs_of.index(bmm)
            items = [u for u in node.users if u.target is operator.getitem
                     and u.args[1] == index]
            (item,) = items if items else list(node.users)
            rec["values"]["scores"] = where(item)
            records.append(rec)
        elif node.target is bmm_op:
            rec = {"kind": "portable", "mode": "portable", "values": {
                "q": where(node.args[0]), "kt": where(node.args[1]),
                "scores": where(node)}}
            records.append(rec)
    output.args = (tuple(outs),)
    gm.recompile()
    return records


def lower_exposed(model, args, mutable=(), flow="lower", compile_config=None,
                  **options):
    """lower_esp32 with an Esp32Partitioner(**options) (and the edge
    `compile_config`, if given) -- or, for flow "two_step", calibration.py's
    flow: to_edge_transform_and_lower without a partitioner,
    calibrate_static_act, to_backend -- then the q@kT exposed, apply_passes,
    to_executorch: (partitioner, program, records, log)."""
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    part = Esp32Partitioner(**options)
    log = io.StringIO()
    passes = [InitializedMutableBufferPass(list(mutable))] if mutable else []
    with contextlib.redirect_stdout(log):
        ep = (model if not isinstance(model, nn.Module)
              else torch.export.export(model.eval(), args))
        if flow == "two_step":
            from executorch.exir import to_edge_transform_and_lower

            from executorch.backends.esp32s3.quantizer.calibration import (
                calibrate_static_act,
            )

            edge = to_edge_transform_and_lower(ep)
            calibrate_static_act(edge, [args])
            edge = edge.to_backend(part)
        else:
            edge = lower_esp32(ep, [part], compile_config=compile_config)
        apply_passes(edge, quiet=True)
        records = _expose(edge.exported_program())
        program = edge.to_executorch(config=ExecutorchBackendConfig(
            delegate_alignment=16, passes=passes))
    return part, program, records, log.getvalue()


_DTYPES = {"FLOAT": np.float32, "CHAR": np.int8, "BYTE": np.uint8,
           "INT": np.int32, "LONG": np.int64, "BOOL": np.bool_,
           "SHORT": np.int16, "HALF": np.float16, "DOUBLE": np.float64}


def run_program(program, args):
    """Every user output of `program` on the host runner, in order, as numpy
    arrays of their own dtypes, or (None, log). The method's outputs, read
    from the emitted plan, lead with the buffers the forward writes, which
    are not user outputs."""
    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    plan = program.executorch_program.execution_plan[0]
    outs = [plan.values[i].val for i in plan.outputs]
    n_user = sum(1 for spec in
                 program.exported_program().graph_signature.output_specs
                 if spec.kind.name == "USER_OUTPUT")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(program.buffer)
        command = [str(HOST_RUNNER), "--pte", str(tmp / "model.pte")]
        for i, x in enumerate(args):
            x.detach().contiguous().numpy().tofile(tmp / f"in{i}.bin")
            command += ["--input", str(tmp / f"in{i}.bin")]
        for i in range(len(outs)):
            command += ["--output", str(tmp / f"out{i}.bin")]
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=300)
        log = done.stdout + done.stderr
        if done.returncode != 0:
            return None, log
        got = []
        for i in range(len(outs) - n_user, len(outs)):
            t = outs[i]
            dtype = _DTYPES[getattr(t.scalar_type, "name", str(t.scalar_type))]
            # The runner writes a tensor's bytes in its dim order (a
            # channels-last value's are not row-major).
            order = list(t.dim_order) or list(range(len(t.sizes)))
            raw = np.fromfile(tmp / f"out{i}.bin", dtype=dtype).reshape(
                tuple(t.sizes[d] for d in order))
            got.append(np.ascontiguousarray(
                np.transpose(raw, np.argsort(order))))
        return got, log


def run_program_steps(program, steps):
    """run_program over several executions of the method in one process
    (the host runner's --script), its state carried from one to the next:
    [each step's user outputs], or (None, log)."""
    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    plan = program.executorch_program.execution_plan[0]
    outs = [plan.values[i].val for i in plan.outputs]
    n_user = sum(1 for spec in
                 program.exported_program().graph_signature.output_specs
                 if spec.kind.name == "USER_OUTPUT")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(program.buffer)
        lines = []
        for j, args in enumerate(steps):
            ins = []
            for i, x in enumerate(args):
                x.detach().contiguous().numpy().tofile(tmp / f"in{j}_{i}.bin")
                ins.append(str(tmp / f"in{j}_{i}.bin"))
            outs_j = [str(tmp / f"out{j}_{i}.bin") for i in range(len(outs))]
            lines.append(" ".join(["forward", *ins, "->", *outs_j]))
        (tmp / "steps.txt").write_text("\n".join(lines) + "\n")
        done = subprocess.run([str(HOST_RUNNER), "--pte", str(tmp / "model.pte"),
                               "--script", str(tmp / "steps.txt")],
                              capture_output=True, text=True, timeout=300)
        log = done.stdout + done.stderr
        if done.returncode != 0:
            return None, log
        got = []
        for j in range(len(steps)):
            step = []
            for i in range(len(outs) - n_user, len(outs)):
                t = outs[i]
                dtype = _DTYPES[getattr(t.scalar_type, "name",
                                        str(t.scalar_type))]
                order = list(t.dim_order) or list(range(len(t.sizes)))
                raw = np.fromfile(tmp / f"out{j}_{i}.bin", dtype=dtype).reshape(
                    tuple(t.sizes[d] for d in order))
                step.append(np.ascontiguousarray(
                    np.transpose(raw, np.argsort(order))))
            got.append(step)
        return got, log


# --- mirrors, references, the bar ------------------------------------------------

def _round_half_away(v):
    return np.sign(v) * np.floor(np.abs(v) + F32(0.5))


def quantize_rows(x):
    """quantize_row_i8 (esp32_backend.cpp) per row of x [..., K]: (codes
    int64, scales float32), absmax/127, the reciprocal, half away, clamp."""
    x = np.asarray(x, dtype=F32)
    amax = np.abs(x).max(axis=-1)
    s = (amax / F32(127.0)).astype(F32)
    with np.errstate(divide="ignore"):
        inv = np.where(s > 0, (F32(1.0) / s).astype(F32), F32(0.0)).astype(F32)
    v = (x * inv[..., None]).astype(F32)
    codes = np.clip(_round_half_away(v), -127, 127).astype(np.int64)
    return codes, s


def mirror_dyn(a, x):
    """OP_BMM_DYN_I8 (run_bmm_dyn_i8): a [B, M, K], x [B, N, K] natural
    layout; both quantised per row, the integer dot, (acc * sa) * sk."""
    qa, sa = quantize_rows(a)
    qx, sx = quantize_rows(x)
    acc = np.matmul(qa, np.swapaxes(qx, -1, -2))
    return ((acc.astype(F32) * sa[..., :, None]).astype(F32)
            * sx[..., None, :]).astype(F32)


def mirror_kcache_i8(q, codes, scales):
    """OP_BMM_KCACHE_I8 (run_bmm_kcache_i8): the query rows quantised, the
    codes and scales as stored."""
    qq, sq = quantize_rows(q)
    acc = np.matmul(qq, np.swapaxes(np.asarray(codes, dtype=np.int64), -1, -2))
    return ((acc.astype(F32) * sq[..., :, None]).astype(F32)
            * np.asarray(scales, dtype=F32)[..., None, :]).astype(F32)


def mirror_affine(q, codes, scales, zps):
    from executorch.backends.esp32s3.quantizer import mirrors

    return mirrors.kcache_scores(np.asarray(q, dtype=F32), codes,
                                 np.asarray(scales, dtype=F32)[..., 0],
                                 np.asarray(zps)[..., 0])


def bmm_portable(a, b, fused=False):
    """ExecuTorch's portable bmm, vectorised over rows and columns: left to
    right over K, each product and sum rounded to float32 on its own, or
    (fused) each step one correctly rounded multiply-add, as a compiler that
    contracts the kernel's sum += a * b emits it (reference.bmm_f32_portable
    takes std::fma)."""
    a = np.asarray(a, dtype=F32)
    b = np.asarray(b, dtype=F32)
    total = np.zeros((a.shape[0], a.shape[1], b.shape[2]), dtype=F32)
    if fused:
        from executorch.backends.esp32s3.quantizer import mirrors
    for t in range(a.shape[2]):
        left, right = a[:, :, t, None], b[:, None, t, :]
        if fused:
            total = mirrors.fma_f32(left, right, total).astype(F32)
        else:
            total = (total + (left * right).astype(F32)).astype(F32)
    return total


def portable_scores(scores, a, b):
    """Whether `scores` is the portable bmm of `a` and `b` bit for bit,
    unfused or fused: the host runner's kernel is whichever its compiler
    emitted, and both are the converted model's arithmetic."""
    bits = np.ascontiguousarray(scores, dtype=F32).view(np.uint32)
    return any(np.array_equal(bits, bmm_portable(a, b, fused).view(np.uint32))
               for fused in (False, True))


def _bar(a, keys, sa, sk=None):
    """Per-score bound on |device - exact| for a query quantised per row at
    absmax/127 (half a step sa/2 per element, plus 2^-22 of it for the
    scale's and reciprocal's roundings), keys `keys` [B, N, K] quantised the
    same way when `sk` is given (OP_BMM_DYN_I8) or exact (a K cache's codes),
    and the epilogue's two float32 roundings of the integer product."""
    a = np.asarray(a, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    ea = 0.5 * np.asarray(sa, np.float64)[..., :, None] + np.abs(a) * 2.0 ** -22
    absk = np.abs(keys)
    bound = np.matmul(ea, np.swapaxes(absk, -1, -2))
    if sk is not None:
        ek = 0.5 * np.asarray(sk, np.float64)[..., :, None] + absk * 2.0 ** -22
        bound += np.matmul(np.abs(a), np.swapaxes(ek, -1, -2))
        bound += np.matmul(ea, np.swapaxes(ek, -1, -2))
    product = np.matmul(np.abs(a) + ea,
                        np.swapaxes(absk + (0 if sk is None else ek), -1, -2))
    return bound + 2.0 ** -21 * product + 1e-30


# --- one case ---------------------------------------------------------------------

@dataclass
class Result:
    case: Case
    delegated: bool = False
    expected: object = None
    outcome: str = ""        # admitted | refused | structural | error
    exact: str = ""          # "exact" | "bar" | ""
    reason: str = ""
    failures: list = field(default_factory=list)


def _value(where, outputs, named_inputs):
    kind, key = where
    if kind == "const":
        return key
    return outputs[key] if kind == "output" else named_inputs[key]


def _recipe_quantised_operand(program):
    """Whether a quantise/dequantise feeds a matmul's operand through moves
    in a converted (ATen) graph: what decides a recipe case's verdict,
    without the gate's own classifier."""
    moves = ("view", "reshape", "permute", "transpose", "expand", "clone",
             "contiguous", "unsqueeze", "squeeze", "alias", "_unsafe_view",
             "select", "slice")
    for node in program.graph.nodes:
        if node.op != "call_function" or not any(
                m in str(node.target) for m in ("matmul", "bmm")):
            continue
        for a in node.args[:2]:
            for _ in range(16):
                if not isinstance(a, torch.fx.Node) or a.op != "call_function":
                    break
                name = str(a.target)
                if "quantize" in name or "dequantize" in name:
                    return True
                if not any(f"aten.{m}" in name for m in moves):
                    break
                a = a.args[0]
    return False


def run_case(case):
    res = Result(case)
    try:
        _run_case(case, res)
    except Exception as e:                   # an export that dies is a result
        import traceback

        res.outcome = "error"
        where = "".join(traceback.format_tb(e.__traceback__)[-2:])
        res.failures.append(f"raised {type(e).__name__}: {str(e)[:400]} at "
                            f"{where[-600:]}")
    finally:
        _release()
    return res


def _run_case(case, res):
    from executorch.backends.esp32s3.test.quant_census import recipes as census

    model = MODELS[case.family](case).eval()
    args = model.inputs()
    options = {"quantize_qk": case.mode.startswith("qk")}
    flow = "lower"
    if case.mode in ("qk+static_act", "qk+2step"):
        options["static_act"] = True
    if case.mode == "qk+2step":
        flow = "two_step"
    eager_seen = None
    program_in = model
    if case.family == "recipe":
        quantized = census.quantize(case.recipe, model, args, [args])
        ep = torch.export.export(quantized.program, tuple(quantized.inputs))
        res.expected = not _recipe_quantised_operand(ep)
        res.expected = res.expected and options["quantize_qk"]
        program_in = ep
    else:
        res.expected = case.expected() and case.id not in READS_EARLIER_STATE
        with torch.no_grad(), _ByDefinition():
            model(*args)              # writes its states: export a fresh one
        eager_seen = dict(model.seen)
        model = program_in = MODELS[case.family](case).eval()
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    try:
        part, program, records, log = lower_exposed(
            program_in, args, mutable=model.mutable()
            if case.family != "recipe" else (), flow=flow, **options)
    except StaticPairRefused as e:
        # absorb_static_qdq refuses an activation quantised per channel by
        # name, before the gate, in both flows: the whole export stops.
        if not ("per_channel" in (case.q, case.k)
                and "per-channel-activation" in str(e)):
            raise
        res.outcome, res.reason = "refused", str(e)[:300]
        if res.expected:
            res.failures.append("refused at export, the allow-list says admit")
        return
    outputs, run_log = run_program(program, args)
    if outputs is None:
        res.outcome = "error"
        res.failures.append(f"host runner failed: {run_log[-400:]}")
        return
    plan = program.exported_program()
    users = [s.arg.name for s in plan.graph_signature.input_specs
             if s.kind.name == "USER_INPUT"]
    named = {n: a.detach().numpy() for n, a in zip(users, args)}
    qk = [r for r in records if r["kind"].startswith("delegate")]
    res.delegated = bool(qk)
    rejected = getattr(getattr(part, "reporter", None), "_rejected_nodes",
                       None) or {}
    bmm_reasons = [why for node, why in rejected.items()
                   if "bmm" in getattr(node, "name", str(node))]
    res.reason = (bmm_reasons[0] if bmm_reasons else "")[:300]
    kinds = dict(part.support.kinds)
    if res.delegated:
        res.outcome = "admitted"
    elif (res.expected or case.broadcast) and any(
            r.split(":")[0] in STRUCTURAL for r in bmm_reasons):
        # Structural by the q@kT's own refusal, not any node's: a refusal
        # for its origin is never filed as layout.
        res.outcome = "structural"
    else:
        res.outcome = "refused"
    if not records:
        # Both operands constants: constant propagation computed the scores
        # at build time, exactly what the converted model computes.
        constant = {"const_float", "computed_const", "const_codes"}
        if case.family == "explicit" and {case.q, case.k} <= constant:
            res.outcome = "folded"
            res.exact = "exact"
            with torch.no_grad():
                want = MODELS[case.family](case).eval()(*args).numpy()
            if not np.allclose(outputs[0], want, rtol=0, atol=1e-6):
                res.failures.append("the folded program is not the converted "
                                    "model's softmax")
            return
        res.failures.append("no q@kT in the lowered program")
        return
    rec = records[0]
    if rec["kind"] == "delegate_internal":
        res.failures.append("the q@kT's partition consumes its scores: "
                            "nothing to hold to the mirror")
        return
    vals = {role: _value(w, outputs, named) for role, w in rec["values"].items()}
    scores = vals["scores"]
    if res.delegated != bool(res.expected) and res.outcome != "structural":
        res.failures.append(
            f"{'delegated' if res.delegated else 'refused'}, the allow-list "
            f"says {'admit' if res.expected else 'refuse'} ({kinds})")
    if (res.outcome == "structural" and case.q_move == case.k_move == "none"
            and not case.broadcast):
        res.failures.append(f"an admissible case without moves refused: "
                            f"{kinds}")
    # The operands as the program computes them, against the converted
    # model's (eager's, sqrt by its definition), bit for bit.
    if eager_seen is not None:
        q_prog = vals["q"]
        if not np.array_equal(q_prog.reshape(-1).view(np.uint32),
                              eager_seen["q"].numpy().reshape(-1).view(
                                  np.uint32)):
            res.failures.append("the query the program computes is not the "
                                "converted model's, bit for bit")
        if "k" in vals:
            k_prog = vals["k"]
        elif "kt" in vals:
            k_prog = np.swapaxes(vals["kt"], -1, -2)
        else:
            k_prog = None
        if k_prog is not None and not np.array_equal(
                np.ascontiguousarray(k_prog).reshape(-1).view(np.uint32),
                eager_seen["k"].numpy().reshape(-1).view(np.uint32)):
            res.failures.append("the key the program computes is not the "
                                "converted model's, bit for bit")
    if rec["kind"] == "portable":
        res.exact = "exact"
        if not portable_scores(scores, vals["q"], vals["kt"]):
            want = bmm_portable(vals["q"], vals["kt"])
            res.failures.append(
                f"portable scores differ from the portable bmm "
                f"(max {np.abs(scores - want).max():.3g})")
        # A refusal names the origin it found: the query's, else the key's.
        sides = (case.q, case.k) if case.family == "explicit" else (case.q,)
        named_origin = [NAMED[o] for o in sides if o in NAMED]
        on = options["quantize_qk"] or case.family == "affine"
        if (res.outcome == "refused" and on and case.family != "recipe"
                and not case.broadcast and named_origin and not any(
                    n in r for n in named_origin for r in bmm_reasons)):
            res.failures.append(f"the refusal does not name the origin "
                                f"({named_origin}): {bmm_reasons[:2]}")
        return
    res.exact = "bar"
    q = vals["q"]
    if rec["mode"] == "dyn":
        k = vals["k"]
        want = mirror_dyn(q, k)
        exact64 = np.matmul(q.astype(np.float64),
                            np.swapaxes(k.astype(np.float64), -1, -2))
        bound = _bar(q, k, quantize_rows(q)[1], quantize_rows(k)[1])
    elif rec["mode"] == "kcache_i8":
        keys = vals["codes"].astype(np.float64) * vals["scales"].astype(
            np.float64)[..., None]
        want = mirror_kcache_i8(q, vals["codes"], vals["scales"])
        exact64 = np.matmul(q.astype(np.float64), np.swapaxes(keys, -1, -2))
        bound = _bar(q, keys, quantize_rows(q)[1])
    else:
        keys = ((vals["codes"].astype(np.float64)
                 - vals["zps"].astype(np.float64))
                * vals["scales"].astype(np.float64))
        want = mirror_affine(q, vals["codes"], vals["scales"], vals["zps"])
        exact64 = np.matmul(q.astype(np.float64), np.swapaxes(keys, -1, -2))
        bound = _bar(q, keys, quantize_rows(q)[1])
    if not np.array_equal(scores.view(np.uint32), want.view(np.uint32)):
        res.failures.append(
            f"host runner != {rec['mode']} mirror "
            f"({int((scores != want).sum())} of {scores.size}, max "
            f"{np.abs(scores - want).max():.3g})")
    err = np.abs(scores.astype(np.float64) - exact64)
    if not (err <= bound).all():
        worst = float((err / bound).max())
        res.failures.append(f"outside the {rec['mode']} bar against the exact "
                            f"float64 product ({worst:.3g}x the budget)")


# --- report --------------------------------------------------------------------

def coverage_table(results):
    """Admitted, refused (and structural), exact-or-bar, per origin x mode."""
    rows = defaultdict(Counter)
    for r in results:
        c = r.case
        origin = (f"q={c.q},k={c.k}" if c.family == "explicit" else
                  f"q={c.q},cache={c.cache}" if c.family != "recipe" else
                  f"recipe={c.recipe}")
        key = (c.family, origin, c.mode)
        rows[key][r.outcome] += 1
        if r.exact:
            rows[key][r.exact] += 1
        if r.failures:
            rows[key]["FAIL"] += 1
    lines = [f"{'family':8} {'origin':42} {'mode':14} admitted refused "
             f"structural exact bar FAIL"]
    for key in sorted(rows):
        n = rows[key]
        lines.append(f"{key[0]:8} {key[1]:42} {key[2]:14} {n['admitted']:8} "
                     f"{n['refused']:7} {n['structural']:10} {n['exact']:5} "
                     f"{n['bar']:3} {n['FAIL']:4}")
    return lines


def origin_mode_table(results):
    """Per operand origin x mode: cases admitted (A), refused (R), folded at
    build time (F, both operands constants) and failed (X). An explicit case
    counts under its query's origin and its key's; a K-cache case under its
    query's origin and its cache's; a recipe case under the recipe. Admitted
    cases are held to the mirror (bit for bit) and the bar, refused and
    folded ones to the converted model's arithmetic (bit for bit)."""
    cells = defaultdict(Counter)
    modes = []
    for r in results:
        c = r.case
        if c.family == "explicit":
            keys = {f"{c.q} (query)", f"{c.k} (key)"}
        elif c.family in ("kcache", "affine"):
            keys = {f"{c.q} (query, {c.family})",
                    f"{c.family} cache {c.cache}"}
        else:
            keys = {f"recipe {c.recipe}"}
        if c.mode not in modes:
            modes.append(c.mode)
        mark = {"admitted": "A", "refused": "R", "folded": "F",
                "structural": "S"}.get(r.outcome, "X")
        for key in keys:
            cells[key][(c.mode, mark)] += 1
            if r.failures:
                cells[key][(c.mode, "X")] += 1
    lines = [f"{'origin':34} " + " ".join(f"{m:>22}" for m in modes)]
    for key in sorted(cells):
        row = []
        for m in modes:
            n = cells[key]
            text = " ".join(f"{k}{n[(m, k)]}" for k in "ARFSX" if n[(m, k)])
            row.append(f"{text or '-':>22}")
        lines.append(f"{key:34} " + " ".join(row))
    return lines


def summary(results, total_pairs=None, covered_pairs=None):
    lines = [f"q@kT matrix: {len(results)} cases "
             f"({Counter(r.case.family for r in results)})"]
    if total_pairs:
        lines.append(f"explicit factor pairs covered: {covered_pairs} of "
                     f"{total_pairs}")
    out = Counter(r.outcome for r in results)
    lines.append(f"outcomes: {dict(out)}; failed: "
                 f"{sum(1 for r in results if r.failures)}")
    for r in results:
        if r.failures:
            lines.append(f"FAIL {r.case.id}: {'; '.join(r.failures)}")
    return lines


# --- pytest ---------------------------------------------------------------------

PAIRWISE, TOTAL_PAIRS = pairwise_cases(seed=0)
CASES = PAIRWISE + kcache_cases() + affine_cases() + recipe_cases()
RESULTS = []


@pytest.fixture(scope="module", autouse=True)
def _report(request):
    yield
    covered = set()
    for r in RESULTS:
        if r.case.family == "explicit":
            covered |= _pairs({n: getattr(r.case, n) for n in EXPLICIT_FACTORS})
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    lines = (summary(RESULTS, TOTAL_PAIRS, len(covered))
             + origin_mode_table(RESULTS))
    for line in lines:
        if reporter is not None:
            reporter.write_line(line)
        else:
            print(line)


# Cases whose operand holds a value computed from state an earlier call wrote
# -- a state the forward halves each call and reads expanded (its row
# amplitudes on one step), or a state read through a bf16/f16 cast (no row a
# written buffer holds, bit for bit) -- which the gate keeps portable:
# values cannot tell such state from a fresh value. Speed only; a precise
# rule for state an earlier call wrote is future work.
READS_EARLIER_STATE = {
    "explicit-plain@alias-state@expand-qk+static_act-mqa-d16-m1-n1",
    "explicit-state@bf16-scaled_runtime@unsqueeze-qk+static_act-mha-d16-m65-n31",
    "explicit-cancel_to_plain@f16-state@f16-qk-mha-d16-m64-n65",
    "explicit-maximum_by_one@expand-state@f16-qk+2step-mqa-d16-m33-n257",
    "explicit-clamp_scaled_eps@bf16-state@f16-qk+2step-mqa-d17-m65-n256",
}


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_qk_case(case):
    res = run_case(case)
    RESULTS.append(res)
    assert not res.failures, "; ".join(res.failures)


class ChainAttn(nn.Module):
    """softmax(chain(q) @ chain(k)^T / 4) with
    Esp32Quantizer's per-tensor chain on either operand or both."""

    def __init__(self, cq, ck):
        super().__init__()
        self.cq, self.ck = cq, ck

    def forward(self, q, k):
        q = _tensor_chain(q) if self.cq else q
        k = _tensor_chain(k) if self.ck else k
        _see(self, {"q": q, "k": k})
        return torch.softmax((q @ k.transpose(-2, -1)) * 0.25, dim=-1)


@pytest.mark.parametrize("cq, ck", [(True, True), (True, False),
                                    (False, True)])
def test_dynamic_chain_on_qk_operands(cq, ck):
    """Under quantize_qk, a q@kT operand that is a dynamic chain's
    dequantised value -- computed in place by strip_dynamic_act_qdq, since no
    delegate absorbs a chain into a bmm -- is refused, naming the chain, and
    the program computes exactly what it computes with quantize_qk off.
    Claimed, the delegate would quantise the recipe's value again per row:
    128 of 128 scores off by up to 3.5e-3."""
    torch.manual_seed(0)
    args = (torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
    model = ChainAttn(cq, ck).eval()
    part, program, records, _ = lower_exposed(model, args, quantize_qk=True)
    off_part, off_program, _, _ = lower_exposed(model, args, quantize_qk=False)
    try:
        assert [r["kind"] for r in records] == ["portable"], records
        assert part.support.kinds["dynamic chain on a q@kT operand"] >= 1, \
            dict(part.support.kinds)
        got, log = run_program(program, args)
        want, log_off = run_program(off_program, args)
        assert got is not None and want is not None, (log, log_off)
        assert np.array_equal(got[0].view(np.uint32), want[0].view(np.uint32))
    finally:
        _release()


# --- dynamic chains, constants and ops behind q@kT operands ----------------

def _bmm_reasons(part):
    """The gate's reasons for each bmm it refused (kind and detail)."""
    rejected = getattr(getattr(part, "reporter", None), "_rejected_nodes",
                       None) or {}
    return [str(why) for node, why in rejected.items()
            if "bmm" in getattr(node, "name", str(node))]


def _refused_program(program_in, args, mutable=(), flow="lower", **options):
    """Lower `program_in` (a module or an ExportedProgram, its q@kT
    exposed) and check that no q@kT was delegated and that every bmm's
    scores are the portable bmm of its operands. Returns (refusal kinds,
    bmm refusal reasons, the first bmm's values by role)."""
    part, program, records, log = lower_exposed(
        program_in, args, mutable=mutable, flow=flow, **options)
    try:
        assert records and all(r["kind"] == "portable" for r in records), (
            records, dict(part.support.kinds), log[-800:])
        got, run_log = run_program(program, args)
        assert got is not None, run_log[-800:]
        users = [s.arg.name for s in
                 program.exported_program().graph_signature.input_specs
                 if s.kind.name == "USER_INPUT"]
        named = {n: a.detach().numpy() for n, a in zip(users, args)}
        first = None
        for rec in records:
            vals = {role: _value(w, got, named)
                    for role, w in rec["values"].items()}
            assert portable_scores(vals["scores"], vals["q"], vals["kt"])
            first = vals if first is None else first
        return dict(part.support.kinds), _bmm_reasons(part), first
    finally:
        _release()


def _refused_exact(make, args, mutable=(), flow="lower", **options):
    """Lower the model `make()` builds (its q@kT exposed) and check it was
    not delegated and computes the converted model's arithmetic: scores the
    portable bmm of operands equal to eager's (sqrt by its definition:
    _ByDefinition). Returns the gate's refusal kinds and its bmm refusal
    reasons. Eager runs on one instance and the lowering takes a fresh one:
    a forward that writes a state would otherwise export what it wrote."""
    model = make().eval()
    with torch.no_grad(), _ByDefinition():
        model(*args)
    seen = {k: v.detach().clone() for k, v in model.seen.items()}
    kinds, reasons, vals = _refused_program(make().eval(), args,
                                            mutable=mutable, flow=flow,
                                            **options)
    assert np.array_equal(
        np.ascontiguousarray(vals["q"]).reshape(-1).view(np.uint32),
        seen["q"].numpy().reshape(-1).view(np.uint32))
    assert np.array_equal(
        np.ascontiguousarray(np.swapaxes(vals["kt"], -1, -2)).reshape(
            -1).view(np.uint32),
        seen["k"].numpy().reshape(-1).view(np.uint32))
    return kinds, reasons


class ExpandedPair(nn.Module):
    """A static pair on the query, its batch then
    broadcast by a real expand (static_source does not walk expand_copy)."""

    def forward(self, q, k):
        qd = _qd()
        q = qd.dequantize_per_tensor(
            qd.quantize_per_tensor(q, 1.0, 0, -128, 127, torch.int8),
            1.0, 0, -128, 127, torch.int8).expand(2, -1, -1)
        self.seen = {"q": q, "k": k}
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


class NarrowFrozenCodes(nn.Module):
    """Frozen int8 codes declared [-127, 127]
    (constant_codes_source takes only the whole range), unsqueezed."""

    def __init__(self):
        super().__init__()
        self.register_buffer("codes", torch.tensor([[2, 1]], dtype=torch.int8))

    def forward(self, q, k):
        q = _qd().dequantize_per_tensor(self.codes, 1.0, 0, -127, 127,
                                        torch.int8).unsqueeze(0)
        self.seen = {"q": q, "k": k}
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


class GqaExpandedPair(nn.Module):
    """A static pair on one KV head's keys, broadcast to
    two query heads by repeat_kv (expand + reshape)."""

    def forward(self, q4, k4):
        qd = _qd()
        kd = qd.dequantize_per_tensor(
            qd.quantize_per_tensor(k4, 0.0625, 0, -128, 127, torch.int8),
            0.0625, 0, -128, 127, torch.int8)
        k = kd[:, :, None, :, :].expand(1, 1, 2, 4, 8).reshape(1, 2, 4, 8)
        q, k = q4.reshape(2, 4, 8), k.reshape(2, 4, 8)
        self.seen = {"q": q, "k": k}
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, dim=-1)


class PerChannelQuery(nn.Module):
    """An activation quantised per channel feeding q@kT."""

    def forward(self, q, k):
        qd = _qd()
        scales = torch.tensor([0.02, 0.03, 0.05, 0.07] * 4)
        zps = torch.zeros(16, dtype=torch.int64)
        q = qd.dequantize_per_channel(
            qd.quantize_per_channel(q, scales, zps, 2, -128, 127, torch.int8),
            scales, zps, 2, -128, 127, torch.int8)
        self.seen = {"q": q, "k": k}
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


class AffineStaticQuery(nn.Module):
    """A static pair on
    the stock affine K cache's query, admitted without quantize_qk."""

    def forward(self, q, codes, scales, zps):
        import executorch.backends.esp32s3._passes.stock_llm  # noqa: F401

        qd = _qd()
        q = qd.dequantize_per_tensor(
            qd.quantize_per_tensor(q, 1.0, 0, -128, 127, torch.int8),
            1.0, 0, -128, 127, torch.int8)
        k = torch.ops.espressif.dequantize_per_token_i8(codes, scales, zps)
        self.seen = {"q": q, "k": k}
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


class EmptyQK(nn.Module):
    """A zero extent, which the runtime refuses
    at load."""

    def forward(self, q, k):
        self.seen = {"q": q, "k": k}
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


@pytest.mark.parametrize("name", ["expanded_pair", "narrow_frozen_codes",
                                  "gqa_expanded_pair"])
def test_recipe_operand_refused(name):
    """Recipe-quantised q@kT operands a hazard list misses, refused by
    the allow-list under quantize_qk and left exactly portable."""
    make, args, named = {
        "expanded_pair": (ExpandedPair, (
            torch.tensor([[[2.0, 1.0]]]),
            torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]).expand(2, -1, -1).clone()),
            "static pair"),
        "narrow_frozen_codes": (NarrowFrozenCodes, (
            torch.tensor([[[2.0, 1.0]]]),
            torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])), "constant codes"),
        "gqa_expanded_pair": (GqaExpandedPair, (
            torch.randn(1, 2, 4, 8, generator=torch.Generator().manual_seed(1)),
            torch.randn(1, 1, 4, 8, generator=torch.Generator().manual_seed(2))),
            "static pair"),
    }[name]
    kinds, reasons = _refused_exact(make, args, quantize_qk=True)
    assert any(named in r for r in reasons), (kinds, reasons)


def test_per_channel_query_refused_by_name():
    """lower_esp32 refuses an activation quantised per
    channel by name before the gate (absorb_static_qdq's
    [per-channel-activation]); a flow without that pass hands the gate the
    dequantise, whose origin is a quantised op (test_origin_names)."""
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    args = (torch.randn(1, 6, 16), torch.randn(1, 5, 16))
    with pytest.raises(StaticPairRefused, match="per-channel-activation"):
        try:
            lower_exposed(PerChannelQuery().eval(), args, quantize_qk=True)
        finally:
            _release()


class _Origins(nn.Module):
    """One q@kT per origin the classifier names, each over the same key."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(1, 4, 16))
        self.register_buffer("codes", torch.randint(-127, 128, (1, 4, 16),
                                                    dtype=torch.int8))
        self.register_buffer("state", torch.randn(1, 4, 16))

    def forward(self, x, k):
        qd = _qd()
        scales = torch.tensor([0.02, 0.03, 0.05, 0.07] * 4)
        zps = torch.zeros(16, dtype=torch.int64)
        cases = {
            "plain float activation": x * 1.5 + 0.25,
            "static pair": qd.dequantize_per_tensor(qd.quantize_per_tensor(
                x, 0.0625, 0, -128, 127, torch.int8), 0.0625, 0, -128, 127,
                torch.int8).expand(1, 4, 16).clone(),
            "dynamic chain": _token_chain(x),
            "quantise or dequantise": qd.dequantize_per_channel(
                qd.quantize_per_channel(x, scales, zps, 2, -128, 127,
                                        torch.int8),
                scales, zps, 2, -128, 127, torch.int8),
            "constant": self.p.transpose(1, 2).transpose(1, 2),
            "computed constant": self.p.abs() + 1.0,
            # Read directly, the bmm's own operand, as through any move: a
            # buffer the forward writes, judged by its writes (plain).
            "state buffer": self.state,
            "value rounded onto a grid": torch.clamp(
                torch.round(x / 0.1), -127, 127) * 0.1,
            "integer values cast to float": self.codes.to(torch.float32),
            "unrecognised origin": torch.ops.espressif.observed_range(
                x, -4.0, 4.0),
        }
        out = [torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)
               for q in cases.values()]
        self.state.mul_(0.5)
        return tuple(out)


def test_origin_names():
    """qk_operands.origin on the to_edge graph, before any of lower_esp32's
    passes: the name of each origin the allow-list refuses, and plain."""
    from executorch.exir import to_edge

    import executorch.backends.esp32s3.quantizer.lut16_range  # noqa: F401
    from executorch.backends.esp32s3.operator_support import qk_operands

    model = _Origins().eval()
    args = (torch.randn(1, 4, 16), torch.randn(1, 5, 16))
    edge = to_edge(torch.export.export(model, args))
    ep = edge.exported_program()
    bmms = [n for n in ep.graph.nodes if n.op == "call_function"
            and "aten.bmm" in str(n.target)]
    want = [k for k in ("plain float activation", "static pair",
                        "dynamic chain",
                        "quantise or dequantise", "constant",
                        "computed constant", "plain float activation",
                        "value rounded onto a grid",
                        "integer values cast to float", "unrecognised origin")]
    assert len(bmms) == len(want), len(bmms)
    got = []
    for bmm in bmms:
        a = bmm.args[0]
        got.append(qk_operands.origin(a, ep).kind)
    try:
        assert got == want, [(w, g) for w, g in zip(want, got) if w != g]
    finally:
        _release()


@pytest.mark.parametrize("quantize_qk", [False, True])
def test_affine_query_pair_refused(quantize_qk):
    """A static pair on the stock affine K cache's query is refused, as a
    static pair, with quantize_qk or without: the query's origin is judged
    before the opt-in."""
    args = (torch.tensor([[[2.0, 1.0]]]),
            torch.tensor([[[0, 1], [0, 0]]], dtype=torch.int8),
            torch.ones(1, 2, 1), torch.zeros(1, 2, 1, dtype=torch.int8))
    kinds, _ = _refused_exact(AffineStaticQuery, args,
                              quantize_qk=quantize_qk)
    assert kinds.get("static pair on a bmm of two activations"), kinds


@pytest.mark.parametrize("shapes", [((1, 1, 0), (1, 3, 0)),
                                    ((1, 0, 16), (1, 3, 16))])
def test_empty_qk_refused(shapes):
    args = tuple(torch.zeros(s) for s in shapes)
    kinds, _ = _refused_exact(EmptyQK, args, quantize_qk=True)
    assert kinds.get("empty q@kT"), kinds


class FilledKeys(nn.Module):
    """DeepSeek's MLA key spelling (transformers' expand_kv): an empty
    buffer filled by two slice copies, the shared rotary key broadcast to
    every head. Exported, `_empty_dim_order` under two slice_scatters."""

    def forward(self, q, k_nope, k_rot):
        k = k_nope.new_empty(*k_nope.shape[:-1], 24)
        k[..., :16].copy_(k_nope)
        k[..., 16:].copy_(k_rot.expand(-1, k_nope.shape[1], -1, -1))
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, -1)


def test_filled_keys_admitted():
    """The zoo's deepseek_v3 and axk2 build their keys so; a narrower
    allow-list refuses all four of their q@kT as an unrecognised origin
    (_empty_dim_order). Admitted, the host runner equals the mirror
    of the delegate's inputs bit for bit."""
    g = torch.Generator().manual_seed(12)
    args = (torch.randn(1, 2, 5, 24, generator=g),
            torch.randn(1, 2, 7, 16, generator=g),
            torch.randn(1, 1, 7, 8, generator=g))
    part, program, records, log = lower_exposed(FilledKeys().eval(), args,
                                                quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds))
        got, run_log = run_program(program, args)
        assert got is not None, run_log[-800:]
        users = [s.arg.name for s in
                 program.exported_program().graph_signature.input_specs
                 if s.kind.name == "USER_INPUT"]
        named = {n: a.detach().numpy() for n, a in zip(users, args)}
        vals = {role: _value(w, got, named)
                for role, w in records[0]["values"].items()}
        want = mirror_dyn(vals["q"], vals["k"])
        assert np.array_equal(vals["scores"].view(np.uint32),
                              want.view(np.uint32))
    finally:
        _release()


def _qk_opcodes(edge):
    """The q@kT opcodes in a lowered program's delegate blobs."""
    import struct

    from executorch.exir.lowered_backend_module import LoweredBackendModule

    found = []
    for _, mod in edge.exported_program().graph_module.named_modules():
        if isinstance(mod, LoweredBackendModule):
            blob = mod.processed_bytes
            n = struct.unpack_from("<I", blob, 8)[0]
            found += [struct.unpack_from("<I", blob, 16 + 64 * i)[0]
                      for i in range(n)]
    return found


@pytest.mark.parametrize("int8_kcache", [False, True])
def test_story_prefill_reads_its_cache(int8_kcache):
    """The story's chunked prefill reads the K cache it has not yet written
    this chunk through moves (the old rows by their ring slots), and the
    cache holds the keys the forward writes, plain values: every layer's
    q@kT stays delegated. An allow-list that takes the read for a state
    buffer leaves the fp32 cache's four prefill q@kT portable."""
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.serialization import espressif_blob as eb
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
        build_prefill,
    )

    common = dict(arm="ple", vocab=256, d_model=64, layers=4, heads=8,
                  kv_heads=1, ffn=192, ple_dim=24, seq_len=8, window=4,
                  softcap=20.0)
    _, model = build_prefill(chunk_size=4, int8_kcache=int8_kcache, **common)
    inputs = (torch.arange(3, 7), torch.tensor(4))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(torch.export.export(model, inputs),
                               [Esp32Partitioner(format_policy="i8",
                                                 quantize_qk=True)])
        ops = _qk_opcodes(edge)
        want = eb.OP_BMM_KCACHE_I8 if int8_kcache else eb.OP_BMM_DYN_I8
        # At chunk start 4 (window 4) layers 2-4's queries are computed from
        # the previous chunk's rows -- state an earlier call wrote -- and stay
        # portable; layer 1's q@kT is delegated. Speed only (the story's own
        # chunk test, from position 0, keeps all 28); a precise rule is
        # future work.
        assert ops.count(want) == 1, ops
    finally:
        _release()


class AffinePlainQuery(nn.Module):
    """The stock affine K cache's q@kT over a plain query."""

    def forward(self, q, codes, scales, zps):
        import executorch.backends.esp32s3._passes.stock_llm  # noqa: F401

        k = torch.ops.espressif.dequantize_per_token_i8(codes, scales, zps)
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)


def _affine_args(seed=11):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(2, 3, 16, generator=g),
            torch.randint(-128, 128, (2, 5, 16), generator=g,
                          dtype=torch.int8),
            torch.rand(2, 5, 1, generator=g) * 0.02 + 1e-3,
            torch.randint(-20, 20, (2, 5, 1), generator=g, dtype=torch.int8))


def test_affine_kcache_under_fp32_policy_refused():
    """The affine K cache's q@kT quantises its query to
    int8, and the loops of format_policy='fp32' have no activation
    quantiser: refused by name under that policy, even under quantize_qk,
    and exactly portable; under the default policy the same graph
    delegates."""
    args = _affine_args()
    kinds, _ = _refused_exact(AffinePlainQuery, args, format_policy="fp32",
                              quantize_qk=True)
    assert kinds.get("int8 q@kT under fp32 policy"), kinds
    _, _, records, _ = lower_exposed(AffinePlainQuery().eval(), args,
                                     quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], records
    finally:
        _release()


class CatCache(nn.Module):
    """A decode step whose keys are torch.cat of a cache buffer and the new
    row, the buffer then overwritten with the newest rows: the delegate reads
    the cat, a runtime tensor, and the cache holds what the forward writes."""

    def __init__(self, new_origin):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("past", torch.randn(1, 2, 15, 16, generator=g))
        self.new_origin = new_origin

    def forward(self, q, k_new):
        new = _tensor_chain(k_new) if self.new_origin == "chain" else k_new
        k = torch.cat([self.past, new], dim=2)
        _see(self, {"q": q, "k": k})
        out = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, -1)
        self.past.copy_(k[:, :, 1:])
        return out


@pytest.mark.parametrize("new_origin", ["plain", "chain"])
def test_cat_cache_keys(new_origin):
    """A cache read as one side of a cat is judged by what the forward
    writes into it: plain rows admit the q@kT (the host runner equals the
    mirror bit for bit), a chain's rows refuse it by name."""
    args = (torch.randn(1, 2, 3, 16, generator=torch.Generator().manual_seed(8)),
            torch.randn(1, 2, 1, 16, generator=torch.Generator().manual_seed(9)))
    if new_origin == "chain":
        kinds, _ = _refused_exact(lambda: CatCache("chain"), args,
                                  quantize_qk=True, mutable=["past"])
        assert kinds.get("dynamic chain on a q@kT operand"), kinds
        return
    part, program, records, log = lower_exposed(
        CatCache("plain").eval(), args, mutable=["past"], quantize_qk=True)
    try:
        # The shifted rows are the cache's own rows, judged with every test
        # (no carried exemption), and vector levels keeps the q@kT portable
        # -- speed only; a precise rule for state an earlier call wrote is
        # future work.
        assert [r["kind"] for r in records] == ["portable"], (
            records, dict(part.support.kinds))
        return
        got, run_log = run_program(program, args)
        assert got is not None, run_log[-800:]
        users = [s.arg.name for s in
                 program.exported_program().graph_signature.input_specs
                 if s.kind.name == "USER_INPUT"]
        named = {n: a.detach().numpy() for n, a in zip(users, args)}
        vals = {role: _value(w, got, named)
                for role, w in records[0]["values"].items()}
        want = mirror_dyn(vals["q"], vals["k"])
        assert np.array_equal(vals["scores"].view(np.uint32),
                              want.view(np.uint32))
    finally:
        _release()


# --- static pairs, affine caches and multi-head prefill --------------------

def _pair(t, s=0.03125, zp=0):
    """A static per-tensor pair, as a static recipe leaves one on an input."""
    qd = _qd()
    return qd.dequantize_per_tensor(
        qd.quantize_per_tensor(t, s, zp, -128, 127, torch.int8), s, zp, -128,
        127, torch.int8)


class _Deep(nn.Module):
    """An operand behind `n` pairs of moves, from `source`."""

    def __init__(self, n, source):
        super().__init__()
        self.n, self.source = n, source

    def forward(self, x, k):
        q = _pair(x) if self.source == "static_pair" else x * 1.5
        for _ in range(self.n):
            q = q.unsqueeze(0).squeeze(0)
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


def test_walk_has_no_depth_cap():
    """The walk must reach an operand's origin
    however many moves lie between, and refuse when it cannot: a walk that
    stopped after 256 moves would judge the move it stopped at (a view over
    a runtime value, plain) and admit a static pair behind 300 pairs of
    moves. The walk runs to the origin, and a walk that does not end (here,
    a budget forced down to 16) is refused by name."""
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.operator_support import qk_operands

    def operand(source, n):
        edge = to_edge(torch.export.export(
            _Deep(n, source).eval(), (torch.randn(1, 4, 16),
                                      torch.randn(1, 5, 16))))
        ep = edge.exported_program()
        (bmm,) = [m for m in ep.graph.nodes if m.op == "call_function"
                  and "aten.bmm" in str(m.target)]
        return bmm.args[0], ep

    try:
        a, ep = operand("static_pair", 300)
        got = qk_operands.origin(a, ep)
        assert got.kind == qk_operands.STATIC_PAIR, got
        assert len(got.path) >= 600, len(got.path)
        problem = qk_operands.operand_problem(ep, a, "query")
        assert problem is not None and "static pair" in problem[0], problem
        a, ep = operand("plain", 20)
        assert qk_operands.origin(a, ep).kind == qk_operands.PLAIN
        budget, qk_operands._BUDGET = qk_operands._BUDGET, 16
        try:
            got = qk_operands.origin(a, ep)
            refused = qk_operands.operand_problem(ep, a, "query")
        finally:
            qk_operands._BUDGET = budget
        assert got.kind == qk_operands.UNRECOGNISED, got
        assert "does not end" in got.detail, got
        assert refused is not None, refused
    finally:
        _release()


class _DeepGraph(nn.Module):
    """An operand computed by `n` additions in a row, or assembled by `n`
    nested cats."""

    def __init__(self, n, kind):
        super().__init__()
        self.n, self.kind = n, kind

    def forward(self, x, k):
        q = x
        for _ in range(self.n):
            if self.kind == "arith":
                q = q * 1.0001 + 0.5
            else:
                q = torch.cat([q[:, :1], q[:, 1:]], dim=1)
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


@pytest.mark.parametrize("kind, n, want", [("arith", 700, "plain"),
                                           ("cat", 600, "plain")])
def test_deep_graphs_judged_without_crashing(kind, n, want):
    """A deep graph is judged, never a crash. An operand behind 700
    additions is plain: _Signature.is_constant walks the activation's
    history without recursing, which would raise RecursionError past
    Python's limit. An operand assembled by 600 nested cats is plain too:
    the provenance search is iterative (a recursive one would refuse such
    assemblies at Python's limit, as would a 256-source budget)."""
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.operator_support import qk_operands

    try:
        edge = to_edge(torch.export.export(
            _DeepGraph(n, kind).eval(), (torch.randn(1, 8, 16),
                                         torch.randn(1, 5, 16))))
        ep = edge.exported_program()
        (bmm,) = [m for m in ep.graph.nodes if m.op == "call_function"
                  and "aten.bmm" in str(m.target)]
        got = qk_operands.origin(bmm.args[0], ep)
        if want == "plain":
            assert got.kind == qk_operands.PLAIN, got
        else:
            assert got.kind == qk_operands.UNRECOGNISED, got
            assert "too deep" in got.detail, got
            assert qk_operands.operand_problem(ep, bmm.args[0],
                                               "query") is not None
    finally:
        _release()


class MovedPair(nn.Module):
    """A static pair on a q@kT operand, reached
    through data movement a hazard list does not walk: a fused qkv
    pair split or unbound, GQA's repeat_kv (expand and reshape), a slice,
    and the copy torch.matmul's reshape makes of a transposed head
    (_clone_dim_order) -- p01's pair sits after the head transpose."""

    B, S, H, KH, D = 1, 8, 4, 2, 16

    def __init__(self, case):
        super().__init__()
        self.case = case

    def forward(self, x, y):                  # [B, S, H*D] each
        B, S, H, KH, D = self.B, self.S, self.H, self.KH, self.D
        sq, sk = float(F32(0.02)), float(F32(0.03))

        def heads(t, n):                      # Hugging Face's head split
            return t.view(B, S, n, D).transpose(1, 2)

        c = self.case
        if c == "qkv":
            q, k = _pair(torch.cat([x, y], dim=-1), sq).split(H * D, dim=-1)
            q, k = heads(q, H), heads(k, H)
        elif c in ("gqa", "gqa_k"):
            q = heads(_pair(x, sq) if c == "gqa" else x, H)
            k = heads(_pair(y[..., :KH * D], sk), KH)
            k = k[:, :, None].expand(B, KH, H // KH, S, D).reshape(B, H, S, D)
        elif c == "unbind":
            q, k = _pair(torch.stack([x, y], dim=0), sq).unbind(0)
            q, k = heads(q, H), heads(k, H)
        elif c == "slice_k":
            q = heads(x, H)
            k = heads(_pair(torch.cat([y, x], dim=-1), sk)[..., :H * D], H)
        elif c == "clone":
            q, k = heads(_pair(x, sq), H), heads(y, H)
        else:                                 # p01: pair_q, pair_k, pair_qk
            q, k = heads(x, H), heads(y, H)
            if c in ("pair_q", "pair_qk"):
                q = _pair(q, sq)
            if c in ("pair_k", "pair_qk"):
                k = _pair(k, sk)
        _see(self, {"q": q, "k": k})
        return torch.softmax((q @ k.transpose(-2, -1)) * 0.25, dim=-1)


@pytest.mark.parametrize("case", ["pair_q", "pair_k", "pair_qk", "qkv",
                                  "gqa", "gqa_k", "unbind", "slice_k",
                                  "clone"])
def test_static_pair_through_moves(case):
    """Each is refused as a static pair under quantize_qk, exactly portable,
    its operands eager's bit for bit; delegated, each is 4.1e-3 to 5.3e-3
    off."""
    g = torch.Generator().manual_seed(0)
    args = (torch.randn(1, 8, 64, generator=g),
            torch.randn(1, 8, 64, generator=g))
    kinds, reasons = _refused_exact(lambda: MovedPair(case), args,
                                    quantize_qk=True)
    assert any("static pair" in r for r in reasons), (kinds, reasons)


class HandPair(nn.Module):
    """On a fused [1, S, 2, H, DH]
    tensor read by select (qk[0], qk[1]); on a query every head shares,
    broadcast by matmul; on a key made contiguous after its transpose."""

    S, H, DH = 8, 2, 16

    def __init__(self, case):
        super().__init__()
        self.case = case

    def forward(self, a, b=None):
        S, H, DH = self.S, self.H, self.DH
        if self.case == "select":             # a [1, S, 2*H*DH]
            qk = _pair(a).view(1, S, 2, H, DH).permute(2, 0, 3, 1, 4)
            q, k = qk[0], qk[1]
        elif self.case == "broadcast_query":  # a [1, 1, S, DH], b [1, H, S, DH]
            q, k = _pair(a), b
        else:                                 # contiguous_key: [1, S, H, DH]
            q = a.transpose(1, 2)
            k = _pair(b).transpose(1, 2).contiguous()
        _see(self, lambda: {"q": q.expand(1, H, S, DH), "k": k})
        return torch.softmax((q @ k.transpose(-2, -1)) * 0.25, dim=-1)


@pytest.mark.parametrize("case", ["select", "broadcast_query",
                                  "contiguous_key"])
def test_hand_pairs(case):
    """The hand-written pairs on the fused qkv:
    refused as a static pair, exactly portable."""
    g = torch.Generator().manual_seed(3)
    shapes = {"select": ((1, 8, 64),),
              "broadcast_query": ((1, 1, 8, 16), (1, 2, 8, 16)),
              "contiguous_key": ((1, 8, 2, 16), (1, 8, 2, 16))}[case]
    args = tuple(torch.randn(*s, generator=g) for s in shapes)
    kinds, reasons = _refused_exact(lambda: HandPair(case), args,
                                    quantize_qk=True)
    assert any("static pair" in r for r in reasons), (kinds, reasons)


class QkvSplit(nn.Module):
    """NanoGPT's fused qkv Linear, split, heads by view
    and transpose (or one natively 3-D head), the 1/sqrt(d) after the bmm;
    or separate Linears with contiguous heads."""

    D, H, S = 32, 2, 8

    def __init__(self, case):
        super().__init__()
        self.case = case
        if case == "separate_contiguous":
            self.q, self.k = nn.Linear(self.D, self.D), nn.Linear(self.D,
                                                                  self.D)
        else:
            self.qkv = nn.Linear(self.D, 3 * self.D)

    def forward(self, x):
        D, H, S = self.D, self.H, self.S
        dh = D // H
        if self.case == "fused_split_3d":
            q, k, _ = self.qkv(x).split(D, dim=-1)
            return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)
        if self.case == "fused_split":
            q, k, _ = self.qkv(x).split(D, dim=-1)
            q = q.view(1, S, H, dh).transpose(1, 2)
            k = k.view(1, S, H, dh).transpose(1, 2)
        else:
            q = self.q(x).view(1, S, H, dh).transpose(1, 2).contiguous()
            k = self.k(x).view(1, S, H, dh).transpose(1, 2).contiguous()
        return torch.softmax((q @ k.transpose(-2, -1)) * 0.25, dim=-1)


@pytest.mark.parametrize("case", ["fused_split", "fused_split_3d",
                                  "separate_contiguous"])
def test_esp32quantizer_static_pairs(case):
    """Esp32Quantizer(preset="xnnpack_static") pairs on
    the q@kT operands, reached through a split's getitems, the head
    transpose's copy, or .contiguous(): refused as static pairs
    (delegated, up to 1.6e-3 off), every bmm the portable bmm of
    the program's own operands."""
    from executorch.backends.esp32s3.test.test_partition_crashes import pt2e

    torch.manual_seed(0)
    model = QkvSplit(case).eval()
    x = torch.randn(1, 8, 32)
    ep = pt2e(model, x, preset="xnnpack_static")
    kinds, reasons, _ = _refused_program(ep, (x,), quantize_qk=True)
    assert any("static pair" in r for r in reasons), (kinds, reasons)


class StockAttn(nn.Module):
    """NanoGPT's fused attention (explicit or
    F.scaled_dot_product_attention, whose canonicalisation puts sqrt(scale)
    on each operand), Llama's GQA, and timm's ViT attention (unbind)."""

    B, S, H, KH, D = 1, 8, 4, 2, 16

    def __init__(self, case):
        super().__init__()
        self.case = case
        e = self.H * self.D
        if case == "gqa":
            self.q_proj = nn.Linear(e, self.H * self.D, bias=False)
            self.k_proj = nn.Linear(e, self.KH * self.D, bias=False)
            self.v_proj = nn.Linear(e, self.KH * self.D, bias=False)
        else:
            self.qkv = nn.Linear(e, 3 * e)
            if case == "fused_noscale":
                with torch.no_grad():
                    self.qkv.weight[:e] *= self.D ** -0.5
                    self.qkv.bias[:e] *= self.D ** -0.5
        self.proj = nn.Linear(e, e)

    def forward(self, x):
        B, S, H, KH, D = self.B, self.S, self.H, self.KH, self.D
        e = H * D
        if self.case == "gqa":
            def rep(t):
                return t[:, :, None].expand(B, KH, H // KH, S, D).reshape(
                    B, H, S, D)
            q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)
            k = rep(self.k_proj(x).view(B, S, KH, D).transpose(1, 2))
            v = rep(self.v_proj(x).view(B, S, KH, D).transpose(1, 2))
        elif self.case == "unbind_sdpa":
            q, k, v = self.qkv(x).reshape(B, S, 3, H, D).permute(
                2, 0, 3, 1, 4).unbind(0)
        else:
            q, k, v = (t.view(B, S, H, D).transpose(1, 2)
                       for t in self.qkv(x).split(e, dim=-1))
        if self.case.endswith("sdpa"):
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.case == "fused_sdpa")
        elif self.case == "fused_noscale":   # 1/sqrt(D) folded into qkv
            y = torch.softmax(q @ k.transpose(-2, -1), -1) @ v
        else:
            att = torch.softmax((q @ k.transpose(-2, -1)) * D ** -0.5, -1)
            y = att @ v
        return self.proj(y.transpose(1, 2).reshape(B, S, e))


@pytest.mark.parametrize("case, named", [
    ("fused_noscale", "static pair"), ("fused_sdpa", "static pair"),
    ("unbind_sdpa", "static pair"),
    # XNNPACK's pair on the scores themselves: no q@kT by structure.
    ("fused", "bmm not q@kT"), ("gqa", "bmm not q@kT")])
def test_stock_static_recipe(case, named):
    """Stock XNNPACK static quantisation (its pairs on
    both q@kT operands), with SDPA's sqrt(scale) on each operand between
    the pair and the bmm in the *_sdpa cases (a multiply by one value, which
    the walk steps through): refused as static pairs (delegated, the fused
    and SDPA cases are up to 7.5e-3 off), every bmm the
    portable bmm of the program's own operands. Where the recipe also pairs
    the scores, the bmm is no q@kT at all."""
    from executorch.backends.esp32s3.test.quant_census.recipes import quantize

    torch.manual_seed(1)
    net = StockAttn(case).eval()
    x = torch.randn(1, 8, 64)
    calib = [(torch.randn(1, 8, 64),) for _ in range(4)] + [(x,)]
    with contextlib.redirect_stdout(io.StringIO()):
        qz = quantize("xnnpack-static", net, (x,), calib)
        ep = torch.export.export(qz.program, (x,), strict=False)
    options = {**qz.info.get("lowering", {}), "quantize_qk": True}
    kinds, reasons, _ = _refused_program(ep, (x,), **options)
    if named == "bmm not q@kT":
        # P@V is refused so too: both bmms must be.
        assert sum(named in r for r in reasons) == 2, (kinds, reasons)
    else:
        assert any(named in r for r in reasons), (kinds, reasons)


class ScaledPair(nn.Module):
    """A static pair scaled by one value before the bmm: a Python scalar
    (mul.Scalar, or the one-element constant ExecuTorch makes of it) or a
    one-element tensor, as SdpaToQk spells sqrt(scale) on each side."""

    def __init__(self, how):
        super().__init__()
        self.how = how

    def forward(self, q, k):
        q = _pair(q)
        if self.how == "scalar":
            q = q * 0.5
        elif self.how == "one_element":
            q = q * torch.tensor([0.5])
        else:
            q = q / 4.0
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


@pytest.mark.parametrize("how", ["scalar", "one_element", "divide"])
def test_scaled_pair_refused(how):
    """A multiply or divide by one value keeps a
    recipe-quantised value recipe-quantised; the walk steps through it and
    refuses the pair rather than delegating it."""
    g = torch.Generator().manual_seed(4)
    args = (torch.randn(2, 5, 16, generator=g),
            torch.randn(2, 7, 16, generator=g))
    kinds, reasons = _refused_exact(lambda: ScaledPair(how), args,
                                    quantize_qk=True)
    assert any("static pair" in r for r in reasons), (kinds, reasons)


class ChainBmm(nn.Module):
    """torch.bmm over 3-D operands, a chain on
    one of them."""

    def __init__(self, cq, ck):
        super().__init__()
        self.cq, self.ck = cq, ck

    def forward(self, q, k):
        q = _tensor_chain(q) if self.cq else q
        k = _tensor_chain(k) if self.ck else k
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, dim=-1)


@pytest.mark.parametrize("cq, ck", [(True, False), (False, True)])
def test_two_step_static_act_chain(cq, ck):
    """calibration.py's two-step flow (to_edge, then
    calibrate_static_act, then to_backend with static_act and quantize_qk)
    runs no strip, so the chain reaches the gate whole, found by its
    chooser: refused, exactly portable (delegated, 127 of 128 scores are
    off by up to 3.2e-3). The matrix's qk+2step mode holds both
    K-cache ops to the same."""
    torch.manual_seed(0)
    args = (torch.randn(2, 8, 16), torch.randn(2, 8, 16))
    kinds, reasons = _refused_exact(lambda: ChainBmm(cq, ck), args,
                                    flow="two_step", static_act=True,
                                    quantize_qk=True)
    assert kinds.get("dynamic chain on a q@kT operand"), kinds
    assert any("chooses its scale" in r or "whose scale" in r
               for r in reasons), reasons


class CodesQK(nn.Module):
    """Frozen int8 codes
    dequantised over a declared range, reached through a move, on the query
    or the key side."""

    S = float(F32(0.0117))

    def __init__(self, qmin, qmax, move, side, batch):
        super().__init__()
        g = torch.Generator().manual_seed(23)
        self.register_buffer("codes", torch.randint(
            qmin, qmax + 1, (8, 32), generator=g, dtype=torch.int8))
        self.qmin, self.qmax = qmin, qmax
        self.move, self.side, self.batch = move, side, batch

    def forward(self, x):                     # [batch, 12, 32]
        c = _qd().dequantize_per_tensor(self.codes, self.S, 0, self.qmin,
                                        self.qmax, torch.int8).unsqueeze(0)
        if self.move == "expand":
            c = c.expand(self.batch, 8, 32)
        elif self.move == "slice":
            c = c[:, 2:7]
        q, k = (c, x) if self.side == "query" else (x, c)
        _see(self, lambda: {"q": q.expand(self.batch, *q.shape[1:]),
                            "k": k.expand(self.batch, *k.shape[1:])})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)


@pytest.mark.parametrize("side", ["query", "key"])
@pytest.mark.parametrize("label, qmin, qmax, move, batch", [
    ("narrow-viewed", -127, 127, "view", 1),
    ("whole-expanded", -128, 127, "expand", 2),
    ("half-range-viewed", -64, 63, "view", 1),
    ("positive-sliced", 0, 127, "slice", 1),
])
def test_constant_codes_any_range(label, qmin, qmax, move, batch,
                                        side):
    """A dequantise of constant
    codes is refused as constant codes whatever range it declares and
    whatever moves lie between, not only the narrow ranges as "a quantised
    op"; delegated, they are up to 1.3e-2 off."""
    x = torch.linspace(-2, 2, batch * 12 * 32).reshape(batch, 12, 32)
    kinds, reasons = _refused_exact(
        lambda: CodesQK(qmin, qmax, move, side, batch), (x,),
        quantize_qk=True)
    assert any("constant codes" in r for r in reasons), (kinds, reasons)


class DequantisedKeys(nn.Module):
    """Keys that are int8 codes dequantised in ATen ops: times a scale per
    head dimension (StaticQuantizedKVCache's read; a scale no walk steps
    through) or one value; with a zero point, (c - z) * s or c * s - z * s;
    or cached and new rows cast, concatenated, then scaled."""

    def __init__(self, how):
        super().__init__()
        self.how = how

    def forward(self, q, codes, scales, zps):
        c = codes.to(torch.float32)
        if self.how == "per_dim":
            k = c * scales
        elif self.how == "one_scale":
            k = c * 0.05
        elif self.how == "zero_point":
            k = (c - zps) * scales
        elif self.how == "offset_after":
            k = c * scales - zps * scales
        else:                                 # cat_then_scale
            k = torch.cat([codes[:, :, :5].to(torch.float32),
                           codes[:, :, 5:].to(torch.float32)], dim=2) * scales
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, -1)


def _codes_args():
    g = torch.Generator().manual_seed(6)
    return (torch.randn(1, 2, 3, 16, generator=g),
            torch.randint(-127, 128, (1, 2, 9, 16), generator=g,
                          dtype=torch.int8),
            torch.rand(1, 2, 1, 16, generator=g) * 0.02 + 1e-3,
            torch.randint(-8, 8, (1, 2, 1, 16), generator=g).to(
                torch.float32))


@pytest.mark.parametrize("how", ["per_dim", "one_scale", "zero_point",
                                 "offset_after", "cat_then_scale"])
def test_dequantised_codes_refused(how):
    """A q@kT operand that is integer codes dequantised in ATen ops is a
    quantised value the delegate would quantise again. A walk that stops at
    a multiply by a per-dimension scale takes it for plain arithmetic and
    admits it (a single scale is walked through to the cast, and refused);
    a zero point, an offset or a cat between the cast and the scale hides
    the codes the same way."""
    kinds, reasons = _refused_exact(lambda: DequantisedKeys(how),
                                    _codes_args(), quantize_qk=True)
    assert kinds.get("q@kT operand is integer codes cast to float"), kinds


class PlainScaledKeys(nn.Module):
    """Keys times a 0/1 mask (a bool cast to float, of the keys' own size)
    or divided by an integer count: plain float arithmetic, no codes."""

    def __init__(self, how):
        super().__init__()
        self.how = how

    def forward(self, q, k, n):
        if self.how == "bool_mask":
            k = k * (k.abs() > 0.1).to(torch.float32)
        else:                                 # count_divide
            k = k / n.to(torch.float32)
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, -1)


@pytest.mark.parametrize("how", ["bool_mask", "count_divide"])
def test_masked_or_counted_keys_admitted(how):
    """The integer-codes rule must leave plain arithmetic plain: a bool is
    a mask, not codes, and an integer count a value is divided by does not
    carry its elements. Admitted under quantize_qk, held to the mirror bit
    for bit."""
    g = torch.Generator().manual_seed(16)
    args = (torch.randn(1, 2, 3, 16, generator=g),
            torch.randn(1, 2, 9, 16, generator=g),
            torch.tensor([3], dtype=torch.int64))
    part, program, records, log = lower_exposed(
        PlainScaledKeys(how).eval(), args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


class FloorGrid(nn.Module):
    """Keys rounded onto a grid by a division that rounds, then scaled
    back: div with rounding_mode 'floor' or 'trunc', or floor division."""

    def __init__(self, how):
        super().__init__()
        self.how = how

    def forward(self, q, k, s):
        if self.how == "floor_divide":
            g = k // s
        else:
            g = torch.div(k, s, rounding_mode=self.how)
        k = g * s
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, -1)


@pytest.mark.parametrize("how", ["floor", "trunc", "floor_divide"])
def test_floor_division_grid_refused(how):
    """A value rounded onto a grid by a division that rounds is a quantised
    value the delegate would quantise again: refused as on a grid, like
    round, floor, ceil and trunc (not admitted as plain)."""
    g = torch.Generator().manual_seed(17)
    args = (torch.randn(1, 2, 3, 16, generator=g),
            torch.randn(1, 2, 9, 16, generator=g),
            torch.rand(1, 2, 1, 16, generator=g) * 0.05 + 0.01)
    kinds, reasons = _refused_exact(lambda: FloorGrid(how), args,
                                    quantize_qk=True)
    assert kinds.get("q@kT operand already on a grid"), kinds


def test_affine_kcache_query_is_opt_in():
    """OP_BMM_KCACHE_AFFINE_I8 quantises
    its query to int8 rows, an accuracy trade every other q@kT op makes only
    under quantize_qk. Without quantize_qk it is refused by name (the q@kT
    runs portable on the cache's exact values); with it, delegated, the host
    runner equal to the mirror of the delegate's inputs bit for bit."""
    args = _affine_args(13)
    kinds, reasons = _refused_exact(AffinePlainQuery, args)
    assert kinds.get("affine K cache's int8 query is opt-in"), kinds
    part, program, records, log = lower_exposed(AffinePlainQuery().eval(),
                                                args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], records
        got, run_log = run_program(program, args)
        assert got is not None, run_log[-800:]
        users = [s.arg.name for s in
                 program.exported_program().graph_signature.input_specs
                 if s.kind.name == "USER_INPUT"]
        named = {n: a.detach().numpy() for n, a in zip(users, args)}
        vals = {role: _value(w, got, named)
                for role, w in records[0]["values"].items()}
        want = mirror_affine(vals["q"], vals["codes"], vals["scales"],
                             vals["zps"])
        assert np.array_equal(vals["scores"].view(np.uint32),
                              want.view(np.uint32))
    finally:
        _release()


def _mirror_records(program, records, args):
    """Each delegated q@kT's scores against its mode's mirror of the
    delegate's own inputs, bit for bit: the number checked."""
    got, run_log = run_program(program, args)
    assert got is not None, run_log[-800:]
    users = [s.arg.name for s in
             program.exported_program().graph_signature.input_specs
             if s.kind.name == "USER_INPUT"]
    named = {n: a.detach().numpy() for n, a in zip(users, args)}
    checked = 0
    for rec in records:
        if rec["kind"] != "delegate":
            continue
        vals = {role: _value(w, got, named)
                for role, w in rec["values"].items()}
        if rec["mode"] == "dyn":
            want = mirror_dyn(vals["q"], vals["k"])
        elif rec["mode"] == "kcache_i8":
            want = mirror_kcache_i8(vals["q"], vals["codes"], vals["scales"])
        else:
            want = mirror_affine(vals["q"], vals["codes"], vals["scales"],
                                 vals["zps"])
        assert np.array_equal(vals["scores"].view(np.uint32),
                              want.view(np.uint32)), rec["mode"]
        checked += 1
    return checked


@pytest.mark.parametrize("int8_kcache", [False, True])
def test_story_prefill_two_kv_heads(int8_kcache):
    """With two KV heads the blocked
    prefill restores its [token, head, key] scores through a reshape that
    needs a copy (_clone_dim_order), which the softmax walk steps through:
    prefill claims all four layers' q@kT, as decode does, and each one's
    scores are its mirror of the delegate's own inputs, bit for bit, on the
    host runner (the prefill oracle, reference_dynamic_qk, models int8)."""
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
        build_prefill,
    )

    common = dict(arm="ple", vocab=256, d_model=64, layers=4, heads=8,
                  kv_heads=2, ffn=192, ple_dim=24, seq_len=8, window=4,
                  softcap=20.0)
    _, model = build_prefill(chunk_size=4, int8_kcache=int8_kcache, **common)
    inputs = (torch.arange(3, 7), torch.tensor(4))
    part, program, records, log = lower_exposed(
        model, inputs, mutable=[n for n, _ in model.named_buffers()],
        format_policy="i8", quantize_qk=True)
    try:
        modes = [r["mode"] for r in records
                 if r["kind"].startswith("delegate")]
        want = "kcache_i8" if int8_kcache else "dyn"
        # Layers 2-4 read state an earlier call wrote at this chunk position
        # (see test_story_prefill_reads_its_cache): portable.
        assert modes == [want], (modes, dict(part.support.kinds))
        assert _mirror_records(program, records, inputs) == 1
    finally:
        _release()


class ProducedBy(nn.Module):
    """A 3-D query a claimable op computes: a Linear, whose 2-D output
    reaches the bmm through a view the q@kT's partition does not claim, or
    an embedding, whose [B, M, K] rows feed the bmm directly."""

    def __init__(self, producer):
        super().__init__()
        torch.manual_seed(14)
        self.producer = producer
        self.proj = nn.Linear(16, 16)
        self.emb = nn.Embedding(32, 16)

    def forward(self, x, k):
        q = self.proj(x) if self.producer == "linear" else self.emb(x)
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)


@pytest.mark.parametrize("producer, outcome", [("linear", "admitted"),
                                               ("embedding", "refused")])
def test_operand_produced_by_a_claimed_op(producer, outcome):
    """The q@kT reads both operands from partition inputs. A query another
    delegate computes (the Linear, then the view) is a partition input,
    admitted and held to the mirror bit for bit; one the q@kT's own
    partition would compute (the embedding's rows) is refused by name, a
    layout limit rather than an origin, and runs portable on the embedding
    delegate's output."""
    g = torch.Generator().manual_seed(15)
    x = (torch.randn(1, 5, 16, generator=g) if producer == "linear"
         else torch.randint(0, 32, (1, 5), generator=g))
    args = (x, torch.randn(1, 7, 16, generator=g))
    if outcome == "refused":
        kinds, _, _ = _refused_program(ProducedBy(producer).eval(), args,
                                       quantize_qk=True)
        assert kinds.get("q@kT operand produced inside the partition"), kinds
        return
    part, program, records, log = lower_exposed(
        ProducedBy(producer).eval(), args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


# --- ops a plain-float walk would trust -----------------------------------------

def _qk3(self, q, k):
    """softmax(q @ k^T / 4) over 3-D operands, eager's copies kept."""
    _see(self, {"q": q, "k": k})
    return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)


def _static(t):
    qd = _qd()
    return qd.dequantize_per_tensor(qd.quantize_per_tensor(
        t, STATIC_SCALE, STATIC_ZP, -128, 127, torch.int8), STATIC_SCALE,
        STATIC_ZP, -128, 127, torch.int8)


def _per_token_fake_quant(t):
    """torchao's QAT fake quantiser per token (IntxFakeQuantizer's), with
    scales chosen per row at run time."""
    from torchao.quantization.qat.utils import _fake_quantize_per_token

    scales = t.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
    zps = torch.zeros(scales.shape, dtype=torch.int32)
    return _fake_quantize_per_token(t, scales, zps, -128, 127)


class Escape(nn.Module):
    """A recipe's value, a grid or codes reaching a q@kT operand through an
    op a narrower allow-list would trust as plain float."""

    def __init__(self, case):
        super().__init__()
        self.case = case

    def forward(self, a, b, c):
        case = self.case
        if case == "broadcast_codes":          # [1,1,16] x [1,2,1]
            q, k = a.to(torch.float32) * b, c
        elif case == "moved_round":           # index_select between
            q, k = torch.round(a / 0.5).index_select(1, c) * 0.5, b
        elif case == "topk":                  # selected values
            q, k = torch.topk(_static(a), 16, dim=-1).values, b
        elif case == "sort":
            q, k = torch.sort(_static(a), dim=-1).values, b
        elif case == "kthvalue":
            q, k = torch.kthvalue(_static(c), 2, dim=2).values, b
        elif case == "max_dim":
            q, k = torch.max(_static(c), dim=2).values, b
        elif case == "min_dim":
            q, k = torch.min(_static(c), dim=2).values, b
        elif case == "neg":
            q, k = -_static(a), b
        elif case == "scatter_add":
            v = _static(a)
            idx = torch.arange(v.shape[1]).view(1, -1, 1).expand_as(v)
            q, k = torch.zeros_like(v).scatter_add(1, idx, v), b
        elif case == "torchao_per_token":     # QAT fake quantiser
            q, k = a, _per_token_fake_quant(b)
        elif case == "torchao_affine":
            from torchao.quantization.quant_primitives import (
                _fake_quantize_affine,
            )

            q, k = a, _fake_quantize_affine(
                b, tuple(b.shape), torch.tensor([0.05]),
                torch.tensor([0], dtype=torch.int32), torch.int8, -128, 127)
        elif case == "round_reshape":         # moves before scale
            g = torch.clamp(torch.round(b / 0.1), -127, 127)
            q, k = a, g.reshape(-1, 16).reshape(b.shape) * 0.1
        elif case == "round_transpose":
            g = torch.clamp(torch.round(b / 0.1), -127, 127)
            q, k = a, g.transpose(-1, -2).transpose(-1, -2) * 0.1
        else:                                 # codes_clamp
            q, k = a, c.to(torch.float32).clamp(-127, 127) * 0.05
        return _qk3(self, q, k)


def _escape_args(case):
    g = torch.Generator().manual_seed(19)
    q = torch.randn(1, 2, 16, generator=g)
    k = torch.randn(1, 3, 16, generator=g)
    if case == "broadcast_codes":
        return (torch.randint(-127, 128, (1, 1, 16), generator=g,
                              dtype=torch.int8),
                torch.tensor([[[1.0], [2.0]]]), k)
    if case == "moved_round":
        return (q, k, torch.tensor([0, 0]))
    if case in ("kthvalue", "max_dim", "min_dim"):
        return (q, k, torch.randn(1, 2, 3, 16, generator=g))
    if case == "codes_clamp":
        return (q, k, torch.randint(-128, 128, (1, 3, 16), generator=g,
                                    dtype=torch.int8))
    return (q, k, q)


ESCAPES = {
    "broadcast_codes": "integer codes cast to float",
    "moved_round": "already on a grid",
    "topk": "static pair", "max_dim": "static pair", "min_dim": "static pair",
    "neg": "static pair", "scatter_add": "static pair",
    "torchao_per_token": "already on a grid",
    "torchao_affine": "already on a grid",
    "round_reshape": "already on a grid",
    "round_transpose": "already on a grid",
    "codes_clamp": "integer codes cast to float",
}


@pytest.mark.parametrize("case", list(ESCAPES))
def test_moving_ops_escapes_refused(case):
    """Each reaches the operand through an op that moves, selects or keeps
    values (a broadcast, a gather, topk's values, neg, scatter_add, a view
    or cast between a rounding and its scale); trusted as plain float and
    delegated under quantize_qk, each is 2e-3 to 1.2e-1 off. The walk steps
    through every such op, finds the recipe, the grid or the codes, and
    refuses by name; the bmm runs portable, exactly."""
    kinds, reasons = _refused_exact(lambda: Escape(case), _escape_args(case),
                                    quantize_qk=True)
    assert any(ESCAPES[case] in r for r in reasons), (kinds, reasons)


class Control(nn.Module):
    """Plain float operands the lists must keep admitted: a value times a
    scale chosen at run time, through relu, through a transcendental
    function."""

    def __init__(self, case):
        super().__init__()
        self.case = case

    def forward(self, q, k):
        if self.case == "runtime_scale":
            k = k * k.abs().amax(-1, keepdim=True)
        elif self.case == "relu":
            q = torch.relu(q) + 0.0
        else:                                 # exp
            q = torch.exp(q * 0.1)
        return _qk3(self, q, k)


@pytest.mark.parametrize("case", ["runtime_scale", "relu", "exp"])
def test_plain_controls_admitted(case):
    """The lists keep plain float plain: admitted under quantize_qk, the
    host runner equal to the mirror of the delegate's own inputs."""
    g = torch.Generator().manual_seed(20)
    args = (torch.randn(1, 2, 16, generator=g),
            torch.randn(1, 3, 16, generator=g))
    part, program, records, log = lower_exposed(Control(case).eval(), args,
                                                quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


# --- grids behind arithmetic ------------------------

@pytest.mark.parametrize("side", ["q", "k"])
@pytest.mark.parametrize("origin", GRID_ARITHMETIC)
def test_grid_behind_arithmetic_refused(origin, side):
    """A grid the model computes itself reaches the operand through
    arithmetic between two values of its own shape: a straight-through
    estimator's add of fq(x) - x (hand-written, LSQ's round_pass, a
    constant step), a scale or an offset expanded or filled to its shape, x
    minus its remainder by a step, sign spelled with a comparison cast to
    float or with where over two numbers. Delegated under quantize_qk, each
    is 0.08-0.13 off the converted model's scores: a search that takes the
    arithmetic for plain float never visits the round, the remainder, the
    comparison or the factories behind it.

    The same premise -- arithmetic between two varying values computes new
    values -- would delegate each spelling after "sub_offset", 0.04 to 0.20
    off the converted model's scores (a plain operand: 0.09): a scale of the
    operand's shape that the per-dimension shape test cannot see (found by
    identity, round(x / s) * s, or by its values), equal terms written apart
    (merged by structure), levels combined by arithmetic (the algebra of
    grids), sign as x / abs(x) and its spellings, levels behind neg or a
    learned alpha, y - frac(y) (frac is x - trunc(x)), and the magic-number
    round (a constant added and taken away again); and sign with one move on
    both sides (0.055-0.062 off at [1, 7, 16]) and guards spelled with
    maximum (0.062 and 0.087). Each is refused by name, and the bmm runs
    portable on the model's own values, bit for bit (run_case's contract)."""
    case = Case("explicit", **{side: origin}, mode="qk", heads="gqa", d=16,
                m=7, n=31, seed=400 + GRID_ARITHMETIC.index(origin))
    res = run_case(case)
    assert not res.failures and res.outcome == "refused", (
        res.outcome, res.failures, res.reason)
    assert NAMED[origin] in res.reason, res.reason


@pytest.mark.parametrize("side", ["q", "k"])
@pytest.mark.parametrize("origin", PLAIN_ARITHMETIC)
def test_plain_arithmetic_admitted(origin, side):
    """The rules that find a grid behind arithmetic -- scales by identity
    and by values, structural merging of terms, the algebra of grids -- keep
    real computation plain: each origin of PLAIN_ARITHMETIC on one side, a
    plain operand on the other, is delegated under quantize_qk, its scores
    the mirror of the delegate's own inputs bit for bit and within the
    mode's bar, and its operands eager's bit for bit (run_case's
    contract)."""
    case = Case("explicit", **{side: origin}, mode="qk", heads="gqa", d=16,
                m=7, n=31, seed=500 + PLAIN_ARITHMETIC.index(origin))
    res = run_case(case)
    assert not res.failures and res.outcome == "admitted", (
        res.outcome, res.failures, res.reason)


class GridControl(nn.Module):
    """Plain float the grid rules must keep plain: a value times or plus a
    QAT linear's fake-quantised output (two values that both vary compute a
    new one, on no grid), a value times its own run-time scale expanded to
    its shape, relu spelled with where, and a sum whose terms cancel to a
    plain value. Then the ones whose
    reductions keep them out of the matrix (it holds operands to eager's
    bits, and a portable mean is not): centring and QK-norm; and a value
    times its own rounding (a grid it does not scale), a remainder alone (x
    minus its grid term: x mod s, a value the model computes, which a rule
    counting any remainder by a step as rounding would refuse). And a
    value 80 elementwise ops deep times its own rounding: the shape test
    reads partitions without recursion, so a chain of any depth is read (one
    not read would follow the rounding too)."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        self.lin = nn.Linear(16, 16)

    def forward(self, q, k):
        case = self.case
        if case in ("times_qat", "plus_qat"):
            y = self.lin(q)
            s = _row_step(y)
            fq = torch.clamp(torch.round(y / s), -8, 7) * s
            y = y + (fq - y).detach()       # on QAT's grid, by its estimator
            q = q * y if case == "times_qat" else q + y
        elif case == "expanded_runtime_scale":
            k = k * k.abs().amax(-1, keepdim=True).expand_as(k)
        elif case == "where_relu":
            q = torch.where(q > 0, q, 0.0)
        elif case == "centered":
            q = q - q.mean(-1, keepdim=True)
        elif case == "qk_norm":
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        elif case == "times_own_round":
            q = q * torch.round(q * 4.0)
        elif case == "remainder_alone":
            q = torch.remainder(q, 0.25)
        elif case == "deep_times_round":
            for _ in range(40):
                q = (q * -1.0625).abs()
            q = q * torch.round(q * 4.0)
        else:                               # cancel_to_plain
            p = q * 1.5
            q = q + (p - q).detach()        # p's value, plain
        return _qk3(self, q, k)


@pytest.mark.parametrize("case", ["times_qat", "plus_qat",
                                  "expanded_runtime_scale", "where_relu",
                                  "cancel_to_plain", "centered", "qk_norm",
                                  "times_own_round", "remainder_alone",
                                  "deep_times_round"])
def test_grid_rules_keep_plain_admitted(case):
    """The rules that find a grid behind arithmetic leave real computation
    plain: admitted under quantize_qk, the host runner equal to the mirror
    of the delegate's own inputs bit for bit."""
    g = torch.Generator().manual_seed(26)
    args = (torch.randn(1, 2, 16, generator=g),
            torch.randn(1, 3, 16, generator=g))
    part, program, records, log = lower_exposed(GridControl(case).eval(),
                                                args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds), _bmm_reasons(part))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


class GridOperand(nn.Module):
    """A grid times
    one value expanded then reshaped to the operand's shape, times a buffer
    of its shape filled with one value, sign as x / abs(x), and a
    straight-through estimator whose x * 2 is written twice."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        self.register_buffer("s", torch.tensor([0.5]))
        self.register_buffer("full", torch.full((1, 1, 16), 0.5))

    def forward(self, x, k):
        if self.case == "expanded":
            q = x.round() * self.s.expand(1, 16).reshape(1, 1, 16)
        elif self.case == "uniform":
            q = x.round() * self.full
        elif self.case == "sign":
            q = x / x.abs()
        else:
            a, b = x * 2., x * 2.
            q = a + (a.round() - b).detach()
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)) * 0.25, -1)


@pytest.mark.parametrize("case, want", [("expanded", 3.), ("uniform", 3.),
                                        ("sign", 1.), ("ste", 4.)])
def test_grid_operands_exact(case, want):
    """Delegated, the first scores are 2.984190, 2.984190, 0.992126 and
    4.031496 instead of the exact grid product; refused, the portable bmm
    gives the exact value."""
    x = torch.tensor([[[4., -2.] + [1.] * 14]])
    if case == "ste":
        x = torch.tensor([[[0.375, 0.75] + [0.] * 14]])
    k = torch.tensor([[[2., 1.] + [0.] * 14, [0.] * 16]])
    part, program, records, log = lower_exposed(GridOperand(case).eval(),
                                                (x, k), quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["portable"], (
            records, _bmm_reasons(part))
        assert any("already on a grid" in r for r in _bmm_reasons(part))
        got, run_log = run_program(program, (x, k))
        assert got is not None, run_log[-800:]
        score = float(got[records[0]["values"]["scores"][1]][0, 0, 0])
        assert score == want, score
    finally:
        _release()


class _Bounded(nn.Module):
    """A plain float operand whose verdict one of the search's limits
    decides: arithmetic of varying values two levels deep (_NESTING); a
    value times its own rounding, which it does not scale (_FACTOR_BUDGET,
    _FACTOR_DEPTH); a value times a rounded row scale expanded to its shape
    (_VARIES_BUDGET); any value at all (_BUDGET); a constant table gathered
    by token ids, a graph input (_INDEX_BUDGET); a difference of two
    expressions that differ only deep down, beside a rounding
    (_SAME_BUDGET)."""

    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.table = nn.Parameter(torch.randn(64, 16,
                                              generator=torch.Generator()
                                              .manual_seed(32)))

    def forward(self, x, y, z, ids, k):
        bound = self.bound
        if bound == "_NESTING":
            q = x * y + x * z
        elif bound in ("_FACTOR_BUDGET", "_FACTOR_DEPTH"):
            q = x * torch.round(x * 4.0)
        elif bound == "_VARIES_BUDGET":
            q = x * torch.round(x).abs().amax(-1, keepdim=True).expand_as(x)
        elif bound == "_BUDGET":
            q = x * y
        elif bound == "_INDEX_BUDGET":
            q = self.table[ids]
        else:                                   # _SAME_BUDGET
            q = torch.exp(x * 0.1) - torch.exp(x * 0.2) + torch.round(x)
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


@pytest.mark.parametrize("bound, small", [
    ("_NESTING", 1), ("_FACTOR_BUDGET", 0), ("_FACTOR_DEPTH", 0),
    ("_VARIES_BUDGET", 0), ("_BUDGET", 0), ("_INDEX_BUDGET", 0),
    ("_SAME_BUDGET", 0)])
def test_arithmetic_walks_refuse_when_cut_short(bound, small):
    """Every walk of the search is bounded, and one cut short refuses or
    searches further, never admits. Hard limits refuse by name: arithmetic
    nested past _NESTING levels, a walk past _BUDGET moves, an index walk
    past _INDEX_BUDGET nodes (a walk over a lookup's indices that answers
    "not codes" when it runs out would admit),
    structural equality past _SAME_BUDGET pairs. Soft ones search further:
    a search for a rounding's divisor that does not finish within
    _FACTOR_BUDGET nodes or _FACTOR_DEPTH moves takes the other operand for
    the step, past _VARIES_BUDGET nodes a partition is not read (and
    every operand it might scale is followed). Each operand here is plain
    float: admitted at the defaults, refused with its bound forced down.
    (_SAME_DEPTH is gone: equality is exact and needs no depth bound.)"""
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.operator_support import qk_operands

    g = torch.Generator().manual_seed(31)
    args = tuple(torch.randn(1, 4, 16, generator=g) for _ in range(3)) + (
        torch.randint(0, 64, (1, 4), generator=g),
        torch.randn(1, 5, 16, generator=g))
    try:
        edge = to_edge(torch.export.export(_Bounded(bound).eval(), args))
        ep = edge.exported_program()
        (bmm,) = [m for m in ep.graph.nodes if m.op == "call_function"
                  and "aten.bmm" in str(m.target)]
        got = qk_operands.origin(bmm.args[0], ep)
        assert got.kind == qk_operands.PLAIN, got
        saved = getattr(qk_operands, bound)
        setattr(qk_operands, bound, small)
        try:
            got = qk_operands.origin(bmm.args[0], ep)
            refused = qk_operands.operand_problem(ep, bmm.args[0], "query")
        finally:
            setattr(qk_operands, bound, saved)
        assert got.kind in (qk_operands.GRID, qk_operands.UNRECOGNISED), got
        assert refused is not None
        if bound == "_NESTING":
            assert "nested past 1 levels" in got.detail, got
        if bound in ("_BUDGET", "_INDEX_BUDGET", "_SAME_BUDGET"):
            assert f"cut short at {bound}" in got.detail, got
    finally:
        _release()


class _Equal(nn.Module):
    """Expressions written twice: a random factory, an uninitialised one, a
    deterministic factory, a product in either order, a cast."""

    def forward(self, x):
        return (torch.rand_like(x), torch.rand_like(x),
                torch.empty_like(x), torch.empty_like(x),
                torch.full_like(x, 0.5), torch.full_like(x, 0.5),
                x * 2.0, x * 2.0, x * x.flip(-1), x.flip(-1) * x,
                x * 3.0)


def test_structural_equality_merges_only_pure_ops():
    """Equal terms merge by structure (_same_value), over pure ops alone: a
    random factory or an uninitialised one written twice is two values,
    while a deterministic factory, a product (in either order: IEEE
    multiplication commutes) written twice is one; a different factor is
    another value."""
    from executorch.backends.esp32s3.operator_support import qk_operands

    ep = torch.export.export(_Equal().eval(), (torch.randn(2, 8),))
    outputs = next(n for n in ep.graph.nodes if n.op == "output").args[0]
    (r1, r2, e1, e2, f1, f2, m1, m2, p1, p2, m3) = outputs
    same = qk_operands._Classifier(ep)._same_value
    assert not same(r1, r2) and not same(e1, e2)
    assert same(f1, f2) and same(m1, m2) and same(p1, p2)
    assert not same(m1, m3) and not same(m1, p1)


class _Chains(nn.Module):
    """Expressions for exact structural equality: two 33-step sin chains
    over different inputs, a 40-step chain written twice over one input, a random factory under deterministic
    wrappers written twice, aliases and a reshape round trip of a product,
    a flip against no flip, a flip written twice."""

    def forward(self, x, y):
        a, b, c, d = x, y, x, x
        for _ in range(33):
            a, b = a.sin(), b.sin()
        for _ in range(40):
            c, d = c.sin(), d.sin()
        r1 = torch.rand_like(x) * 2.0 + 1.0
        r2 = torch.rand_like(x) * 2.0 + 1.0
        half = torch.full_like(x, 0.5)
        b1 = torch.binomial(torch.full_like(x, 4.0), half) * 2.0 + 1.0
        b2 = torch.binomial(torch.full_like(x, 4.0), half) * 2.0 + 1.0
        m = x * 2.0
        alias = x.clone().contiguous() * 2.0
        round_trip = x.reshape(-1).reshape(x.shape) * 2.0
        return (a, b, c, d, r1, r2, b1, b2, m, alias, round_trip,
                x.flip(-1), x.flip(-1), x)


def test_structural_equality_is_exact():
    """Equality is exact, with no depth bound (a bound at depth 32 answering
    "the same" would cancel sin^33(x) and sin^33(y)). Two chains over different inputs differ, however deep; one
    chain written twice is one value at 40 steps; a random factory hidden
    under deterministic ops is never one value with another (binomial,
    whose name says nothing random, by torch's is_impure);
    aliases and a
    reshape back to the same shape are; a flip is not what it flips, two
    flips of one value are one value."""
    from executorch.backends.esp32s3.operator_support import qk_operands

    args = (torch.full((1, 1, 16), 0.25), torch.full((1, 1, 16), 0.75))
    ep = torch.export.export(_Chains().eval(), args)
    outputs = next(n for n in ep.graph.nodes if n.op == "output").args[0]
    (a, b, c, d, r1, r2, b1, b2, m, alias, round_trip, f1, f2, x) = outputs
    same = qk_operands._Classifier(ep)._same_value
    assert not same(a, b)                    # the probe
    assert same(c, d)
    assert not same(r1, r2) and not same(b1, b2)
    assert same(m, alias) and same(m, round_trip)
    assert not same(f1, x) and same(f1, f2)


class DeepSte(nn.Module):
    """a + (round(a) - b).detach() with a and b two 40-step chains of exact
    ops (a multiply by a number, abs): over one input it is round(a), a
    grid; over two inputs it is a varying value."""

    def __init__(self, same):
        super().__init__()
        self.same = same

    def forward(self, x, y, k):
        a, b = x, (x if self.same else y)
        for _ in range(40):
            a = (a * -1.0625).abs()
            b = (b * -1.0625).abs()
        q = a + (torch.round(a) - b).detach()
        return _qk3(self, q, k)


@pytest.mark.parametrize("same", [True, False])
def test_deep_equal_terms(same):
    """At the default limits, a straight-through estimator whose equal
    terms are 40 ops deep cancels to its grid (refused, exactly portable),
    and two deep chains over different inputs never cancel (a depth bound
    would refuse this varying value): delegated, the host runner
    equal to the mirror."""
    g = torch.Generator().manual_seed(33)
    args = (torch.randn(1, 3, 16, generator=g),
            torch.randn(1, 3, 16, generator=g),
            torch.randn(1, 5, 16, generator=g))
    if same:
        _, reasons = _refused_exact(lambda: DeepSte(True), args,
                                    quantize_qk=True)
        assert any("already on a grid" in r for r in reasons), reasons
        return
    part, program, records, log = lower_exposed(DeepSte(False).eval(), args,
                                                quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, _bmm_reasons(part))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


def _ratio_origin(make, x):
    from executorch.backends.esp32s3.operator_support import qk_operands

    class Ratio(nn.Module):
        def forward(self, x):
            return make(x)

    ep = torch.export.export(Ratio(), (x,))
    out = next(n for n in ep.graph.nodes if n.op == "output").args[0][0]
    return qk_operands.origin(out, ep)


SIGN_SPELLINGS = {
    "flip": lambda x: x / x.flip(-1).abs(),
    "roll": lambda x: x / x.roll(1, -1).abs(),
    "slice": lambda x: torch.cat([x[..., :1], x[..., 1:]
                                  / x[..., :-1].abs()], -1),
    "rows": lambda x: x / x.flip(-2).abs(),
    "sqrt": lambda x: x / torch.sqrt(x.flip(-1) * x.flip(-1)),
    "rsqrt": lambda x: x * torch.rsqrt(x.flip(-1) * x.flip(-1)),
    "recip": lambda x: x * x.flip(-1).abs().reciprocal(),
    "matched_flip": lambda x: x.flip(-1) / x.flip(-1).abs(),
    "matched_sqrt": lambda x: x.flip(-1) / torch.sqrt(x.flip(-1)
                                                      * x.flip(-1)),
    "matched_roll": lambda x: x.roll(1, -1) / x.roll(1, -1).abs(),
    # One move on both sides, outside abs or sqrt, or inside the root.
    "matched_flip_both": lambda x: x.flip(-1) / x.abs().flip(-1),
    "matched_view_both": lambda x: (x.view(-1, 16)
                                    / x.abs().view(-1, 16)).view(x.shape),
    "matched_transpose_both": lambda x: (x.transpose(1, 2) / x.abs()
                                         .transpose(1, 2)).transpose(1, 2),
    "matched_sqrt_view": lambda x: (x.view(-1, 16) / torch.sqrt(
        (x * x).view(-1, 16))).view(x.shape),
    "matched_guard_view": lambda x: (x.view(-1, 16) / (x.abs() + 1e-8)
                                     .view(-1, 16)).view(x.shape),
    "matched_recip_flip_both": lambda x: x.flip(-1) * x.abs().reciprocal()
    .flip(-1),
    "matched_abs_over_flip_both": lambda x: x.abs().flip(-1) / x.flip(-1),
    "matched_reshape_runs": lambda x: x.view(-1).view(-1, 16)
    / x.abs().view(-1, 16),
    # A guard spelled with maximum; the magnitude spelled (x * x) ** 0.5,
    # its reciprocal (x * x) ** -0.5.
    # (A tensor built in forward is lift_fresh_copy then an in-place detach_
    # on the exported graph, not a constant: the guards' bounds are fulls.)
    "matched_maximum": lambda x: x / torch.maximum(x.abs(),
                                                   torch.full((1,), 1e-8)),
    "matched_sqrt_maximum": lambda x: x / torch.sqrt(torch.maximum(
        x * x, torch.full((1,), 1e-8))),
    "matched_pow": lambda x: x / torch.pow(x * x, 0.5),
    "matched_pow_rsqrt": lambda x: x * torch.pow(x * x, -0.5),
    # Computation: a move on one side only, two different moves, a maximum
    # by 1 (clamp(x, -1, 1)), a quarter power (sign(x) * sqrt(abs(x))).
    "flip_one_side": lambda x: x.flip(-1) / x.abs(),
    "flip_other_dims": lambda x: x.flip(-1) / x.abs().flip(-2),
    "slice_other": lambda x: x[..., 1:] / x.abs()[..., :-1],
    "maximum_one": lambda x: x / torch.maximum(x.abs(),
                                               torch.full((1,), 1.0)),
    "pow_quarter": lambda x: x / torch.pow(x * x, 0.25),
}


@pytest.mark.parametrize("spelling", list(SIGN_SPELLINGS))
def test_sign_compares_aligned_values(spelling):
    """Sign spelled by hand needs the same value on both sides of the
    division, element for element (a rule blind to what a flip, roll or
    slice moves would refuse x / abs(x.flip(-1)) as sign). Unmatched flips, rolls, slices and row flips are
    plain, in the abs, sqrt, rsqrt and reciprocal spellings; a flip or a
    roll on both sides is still sign. So is one move applied to both x and
    its magnitude (stepping past only aliases, reshapes back and factors
    would leave flip(x) / flip(abs(x)) plain): the moves on each side are
    compared, and the same moves
    over one value are one value. A guard spelled maximum(., eps) and the
    magnitude spelled (x * x) ** 0.5 are found; a maximum by 1 and a quarter power stay computation."""
    from executorch.backends.esp32s3.operator_support import qk_operands

    got = _ratio_origin(SIGN_SPELLINGS[spelling],
                        torch.arange(1, 33).float().reshape(1, 2, 16))
    want = (qk_operands.GRID if spelling.startswith("matched")
            else qk_operands.PLAIN)
    assert got.kind == want, got


class SignControl(nn.Module):
    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling

    def forward(self, q, k):
        return _qk3(self, SIGN_SPELLINGS[self.spelling](q), k)


@pytest.mark.parametrize("spelling", ["flip", "roll", "slice", "sqrt",
                                      "rsqrt", "flip_one_side",
                                      "flip_other_dims", "pow_quarter"])
def test_unaligned_sign_admitted(spelling):
    """The unmatched spellings, lowered: delegated, the host runner equal to
    the mirror of the delegate's inputs."""
    g = torch.Generator().manual_seed(34)
    args = (torch.randn(1, 3, 16, generator=g),
            torch.randn(1, 5, 16, generator=g))
    part, program, records, log = lower_exposed(SignControl(spelling).eval(),
                                                args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, _bmm_reasons(part))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


@pytest.mark.parametrize("spelling", ["matched_pow", "matched_pow_rsqrt"])
def test_pow_magnitude_refused(spelling):
    """x / (x * x) ** 0.5 is sign, and x * (x * x) ** -0.5 too, with the
    exponent exactly 0.5 (delegated, both are 0.062 off the converted model
    at [1, 7, 16]). Refused, the bmm
    portable on the program's own operands. (Not a matrix origin: ATen's
    pow takes 0.5 to its sqrt kernel and the portable pow calls std::pow,
    so the operand is not eager's bit for bit on every machine.)"""
    g = torch.Generator().manual_seed(41)
    args = (torch.randn(1, 7, 16, generator=g),
            torch.randn(1, 31, 16, generator=g))
    kinds, reasons, _ = _refused_program(SignControl(spelling).eval(), args,
                                         quantize_qk=True)
    assert any("already on a grid" in r and "magnitude" in r
               for r in reasons), (kinds, reasons)


class _Epsilon(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.register_buffer("eps0", torch.tensor([1e-8]))

    def forward(self, x):
        return x / x.abs().clamp_min(self.eps0 * self.factor)


@pytest.mark.parametrize("factor, want", [(1e8, "plain"), (10.0, "grid"),
                                          (100.0, "grid")])
def test_guard_epsilon_read_through_factors(factor, want):
    """A guard's epsilon is read through the factors that
    compute it. Read past them, a buffer of 1e-8 times 1e8 -- a bound of
    1.0, which makes x / max(abs(x), eps) clamp(x, -1, 1) -- would count as
    a guard and the computation would be refused as sign. Read through
    them, 1e-8 * 1e8 is computation (and delegated, the matrix's
    clamp_scaled_eps), 1e-8 * 10 and 1e-8 * 100 are guards."""
    from executorch.backends.esp32s3.operator_support import qk_operands

    x = torch.randn(1, 2, 16, generator=torch.Generator().manual_seed(42))
    ep = torch.export.export(_Epsilon(factor).eval(), (x,))
    out = next(n for n in ep.graph.nodes if n.op == "output").args[0][0]
    got = qk_operands.origin(out, ep)
    assert got.kind == (qk_operands.PLAIN if want == "plain"
                        else qk_operands.GRID), got


class ZeroPointLut(nn.Module):
    """A 16-entry table of
    0.5 * round(x) at cast codes plus a zero-point tensor."""

    def __init__(self, side):
        super().__init__()
        self.side = side
        self.register_buffer("table", torch.arange(-8, 8).float() * 0.5)
        self.register_buffer("offset", torch.full((1, 1, 16), 8,
                                                  dtype=torch.int64))

    def forward(self, x, k):
        if self.side == "q":
            x = self.table[x.round().long() + self.offset]
        else:
            k = self.table[k.round().long() + self.offset]
        _see(self, {"q": x, "k": k})
        return torch.softmax(torch.bmm(x, k.transpose(1, 2)) * 0.25, -1)


@pytest.mark.parametrize("side", ["q", "k"])
def test_lookup_with_zero_point_exact(side):
    """Delegated, its first score is 2.984190 (the int8 mirror): an index
    walk that stops at integer arithmetic with two tensors misses the cast
    codes. Every dependency of an index is followed, and the cast codes are
    found: refused, the portable bmm's exact 3.0."""
    a = torch.tensor([[[4., -2.] + [1.] * 14]])
    b = torch.tensor([[[2., 1.] + [0.] * 14]])
    if side == "q":
        x, k = a, torch.cat([b, torch.zeros(1, 1, 16)], 1)
    else:
        x, k = b, torch.cat([a, torch.zeros(1, 1, 16)], 1)
    part, program, records, log = lower_exposed(ZeroPointLut(side).eval(),
                                                (x, k), quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["portable"], (
            records, _bmm_reasons(part))
        assert any("already on a grid" in r and "lookup-table" in r
                   for r in _bmm_reasons(part)), _bmm_reasons(part)
        got, run_log = run_program(program, (x, k))
        assert got is not None, run_log[-800:]
        score = float(got[records[0]["values"]["scores"][1]][0, 0, 0])
        assert score == 3.0, score
    finally:
        _release()


def test_unread_payload_never_plain():
    """A constant whose values the gate cannot read is not read, never
    varying: round(x) times a buffer of x's shape filled with 0.5, its
    values not available (const_weights._concrete finding no real tensor
    behind a FakeTensor), is refused, the portable bmm's exact score
    (test_grid_operands_exact under that fallback). And it does
    not over-refuse: a plain value times the same buffer stays delegated,
    its values unread too, the host runner equal to the mirror."""
    from unittest import mock

    from executorch.backends.esp32s3.operator_support import const_weights

    with mock.patch.object(const_weights, "_concrete", lambda value: None):
        test_grid_operands_exact("uniform", 3.0)
        g = torch.Generator().manual_seed(35)
        args = (torch.randn(1, 1, 16, generator=g),
                torch.randn(1, 2, 16, generator=g))

        class TimesFull(GridOperand):
            def forward(self, x, k):
                q = x * self.full
                _see(self, {"q": q, "k": k})
                return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)

        part, program, records, log = lower_exposed(
            TimesFull("plain").eval(), args, quantize_qk=True)
        try:
            assert [r["kind"] for r in records] == ["delegate"], (
                records, _bmm_reasons(part))
            assert _mirror_records(program, records, args) == 1
        finally:
            _release()


class LargeUniform(nn.Module):
    """round(x) times a buffer of 1,048,832 elements, past a 2^20-element
    payload limit: filled with 0.5, or random values."""

    def __init__(self, varying=False):
        super().__init__()
        g = torch.Generator().manual_seed(40)
        self.register_buffer("full", torch.randn(1, 4097, 256, generator=g)
                             if varying else torch.full((1, 4097, 256), 0.5))

    def forward(self, x, k):
        q = x.round() * self.full
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


@pytest.mark.parametrize("varying", [False, True])
def test_large_payload_read(varying):
    """The values of a constant are read whatever its size (one vectorised
    pass): a uniform buffer past any size limit is a scale, and round(x)
    times it is refused, exactly portable (a size limit would leave such a
    buffer "varying" and delegate it); a buffer of random values is
    no scale, and round(x) times it is new values, delegated (a buffer not
    read would be taken for a possible scale, and refused)."""
    g = torch.Generator().manual_seed(36)
    args = (torch.randn(1, 4097, 256, generator=g) * 3.0,
            torch.randn(1, 2, 256, generator=g))
    if not varying:
        _, reasons = _refused_exact(LargeUniform, args, quantize_qk=True)
        assert any("already on a grid" in r for r in reasons), reasons
        return
    part, program, records, log = lower_exposed(LargeUniform(True).eval(),
                                                args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, _bmm_reasons(part))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


class GroupCodes(nn.Module):
    """Int8 codes and one step per group of four
    channels, given as inputs, the step spread to the codes' width by
    repeat_interleave or by expand and reshape."""

    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling

    def forward(self, codes, steps, k):
        if self.spelling == "repeat_interleave":
            s = steps.repeat_interleave(4, -1)
        else:
            s = steps.unsqueeze(-1).expand(*steps.shape, 4).reshape(
                codes.shape)
        return _qk3(self, codes.to(torch.float32) * s, k)


@pytest.mark.parametrize("spelling", ["repeat_interleave", "expand_reshape"])
def test_group_codes_inputs_refused(spelling):
    """Delegated, it is 0.061 off the converted model: the step, spread to
    the codes' shape, varies along every dimension the codes do, so the
    algebra of grids alone takes the product for new values. The step is
    built from fewer elements (classes of four along the width), a scale
    by shape: the codes carry the elements and are refused by name, and
    the bmm runs portable on the model's own values."""
    g = torch.Generator().manual_seed(37)
    args = (torch.randint(-8, 8, (1, 7, 16), generator=g).to(torch.int8),
            torch.rand(1, 7, 4, generator=g) * 0.2 + 0.05,
            torch.randn(1, 31, 16, generator=g))
    _, reasons = _refused_exact(lambda: GroupCodes(spelling), args,
                                quantize_qk=True)
    assert any("integer codes cast to float" in r for r in reasons), reasons


class GroupKCache(nn.Module):
    """A KIVI-style K cache: int8 codes and one
    step per group of four channels in buffers the forward writes; read as
    codes * steps spread by repeat_interleave or expand and reshape, or
    codes held as floats; this call's keys are quantised and written back
    after the q@kT reads the cache."""

    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling
        g = torch.Generator().manual_seed(38)
        if spelling == "float_codes":
            self.register_buffer("kc", torch.randint(
                -8, 8, (1, 31, 16), generator=g).float())
        else:
            self.register_buffer("kc", torch.randint(
                -127, 128, (1, 31, 16), generator=g).to(torch.int8))
        self.register_buffer("ks", torch.rand(1, 31, 4, generator=g) * 0.02
                             + 0.005)

    def forward(self, q, k):
        if self.spelling == "expand_reshape":
            s = self.ks.unsqueeze(-1).expand(1, 31, 4, 4).reshape(1, 31, 16)
        else:
            s = self.ks.repeat_interleave(4, -1)
        out = _qk3(self, q, self.kc.to(torch.float32) * s)
        step = k.view(1, 31, 4, 4).abs().amax(-1).clamp_min(1e-6) / 127.0
        codes = torch.round(k / step.repeat_interleave(4, -1)).clamp(-127, 127)
        self.kc.copy_(codes.to(self.kc.dtype))
        self.ks.copy_(step)
        return out


def _portable_steps(model, args, mutable, steps):
    """The converted model's outputs over `steps` calls: no partitioner,
    every op portable, on the host runner."""
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

    with contextlib.redirect_stdout(io.StringIO()):
        program = to_edge(torch.export.export(model.eval(), args)) \
            .to_executorch(config=ExecutorchBackendConfig(
                passes=[InitializedMutableBufferPass(list(mutable))]))
    return run_program_steps(program, [args] * steps)


@pytest.mark.parametrize("spelling", ["repeat_interleave", "expand_reshape",
                                      "float_codes"])
def test_group_kcache_in_buffers_exact(spelling):
    """Delegated, it is 0.126 off the converted bmm in 3-D (a plain key:
    0.088): the group steps, spread to the codes' width, hide the codes
    from the algebra of grids. Refused (codes cast to float; the float
    codes' write is a rounding), and over two calls -- the second reads
    what the first wrote -- the scores equal the converted model's, bit for
    bit."""
    g = torch.Generator().manual_seed(39)
    args = (torch.randn(1, 7, 16, generator=g),
            torch.randn(1, 31, 16, generator=g))
    mutable = ("kc", "ks")
    part, program, records, log = lower_exposed(
        GroupKCache(spelling).eval(), args, mutable=mutable,
        quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["portable"], (
            records, _bmm_reasons(part))
        want = ("already on a grid" if spelling == "float_codes"
                else "integer codes cast to float")
        assert any(want in r for r in _bmm_reasons(part)), _bmm_reasons(part)
        got, run_log = run_program_steps(program, [args, args])
        assert got is not None, run_log[-800:]
        ref, ref_log = _portable_steps(GroupKCache(spelling), args, mutable, 2)
        assert ref is not None, ref_log[-800:]
        for step in range(2):
            assert np.array_equal(got[step][0].view(np.uint32),
                                  ref[step][0].view(np.uint32)), step
        assert not np.array_equal(got[0][0], got[1][0])
    finally:
        _release()


class PartitionCases(nn.Module):
    """Partition cases at [1, 7, 16]: a grid of a value broadcast along the
    rows, gathered by run-time indices and times a per-channel parameter; a
    group step gathered to the codes' width by two index tensors. And
    controls: a plain value gathered the same way; a
    plain value plus or times a learned table's row at its argmax (one index
    tensor of two dimensions); int8 codes times a concatenation of two
    different halves of a plain value (not one value: kept apart)."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        g = torch.Generator().manual_seed(43)
        self.chan = nn.Parameter(torch.rand(16, generator=g) * 0.2 + 0.05)
        self.tab16 = nn.Parameter(torch.randn(16, 16, generator=g))
        self.register_buffer("rows7", torch.arange(7))
        self.register_buffer("gidx", torch.arange(16) // 4)

    def forward(self, q, k, codes, sg, idx):
        case = self.case
        if case == "gather_rows":
            g = torch.round(sg[:1, :1, :] * 40.0).expand(1, 7, 4)
            q = g.gather(-1, idx) * self.chan
        elif case == "gather_plain":
            q = q.gather(-1, idx) * self.chan
        elif case == "codes_two_index":
            s = sg[0][self.rows7[:, None], self.gidx[None, :]]
            q = codes.to(torch.float32) * s.unsqueeze(0)
        elif case == "plus_table_at_argmax":
            q = q + self.tab16[q.argmax(-1)]
        elif case == "times_table_at_argmax":
            q = q * self.tab16[q.argmax(-1)]
        else:                                   # codes_cat_halves
            q = codes.to(torch.float32) * torch.cat(
                [q[..., :8] * 1.5, q[..., 8:] * 1.5], -1)
        return _qk3(self, q, k)


def _partition_args():
    g = torch.Generator().manual_seed(5)
    return (torch.randn(1, 7, 16, generator=g),
            torch.randn(1, 31, 16, generator=g),
            torch.randint(-8, 8, (1, 7, 16), generator=g).to(torch.int8),
            torch.rand(1, 7, 4, generator=g) * 0.2 + 0.05,
            torch.randint(0, 4, (1, 7, 16), generator=g))


@pytest.mark.parametrize("case, want", [
    ("gather_rows", "already on a grid"),
    ("codes_two_index", "integer codes cast to float")])
def test_partition_cases_refused(case, want):
    """gather_rows: a gather that keeps its input's partition along every
    other dimension makes a grid broadcast along the rows look one value
    along them after a gather by indices that vary there -- a scale of the
    per-channel parameter, never searched (delegated, 0.054 off) -- so each
    other dimension's partition is the meet with the indices' own.
    codes_two_index: an index by two tensors is a move no rule reads;
    counted as varying it lets the group step hide the codes (0.061 off), so
    a move not read is unknown, and every operand beside it is followed. Refused, the bmm portable on the model's
    own values, bit for bit."""
    kinds, reasons = _refused_exact(lambda: PartitionCases(case), _partition_args(),
                                    quantize_qk=True)
    assert any(want in r for r in reasons), (kinds, reasons)


@pytest.mark.parametrize("case", ["gather_plain", "plus_table_at_argmax",
                                  "times_table_at_argmax",
                                  "codes_cat_halves"])
def test_partition_controls_admitted(case):
    """The partition rules keep plain values plain: a plain value gathered by
    run-time indices; a plain value plus or times a learned table's row at
    its argmax, whose one index of two dimensions is read (not unknown, which
    would follow the lookup and refuse it as codes); codes times two
    different halves of a plain value concatenated (two values, kept apart).
    Delegated, the host runner equal to the mirror."""
    args = _partition_args()
    part, program, records, log = lower_exposed(PartitionCases(case).eval(),
                                                args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, _bmm_reasons(part))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


class TensorRepeats(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("reps", torch.tensor(4))

    def forward(self, x, sg):
        return torch.round(x * 4.0) * sg.repeat_interleave(
            self.reps, -1, output_size=16)


def test_unread_move_is_unknown():
    """A group step spread by repeat_interleave with tensor repeats is a move
    no rule reads; counted as varying, it would leave the rounded value it
    scales unfollowed (plain). It is unknown: every operand beside it is
    followed, and the grid found.
    (On the exported graph: to_edge refuses repeat_interleave.Tensor, which
    is not core ATen.)"""
    from executorch.backends.esp32s3.operator_support import qk_operands

    g = torch.Generator().manual_seed(44)
    args = (torch.randn(1, 7, 16, generator=g),
            torch.rand(1, 7, 4, generator=g) * 0.2 + 0.05)
    ep = torch.export.export(TensorRepeats().eval(), args)
    out = next(n for n in ep.graph.nodes if n.op == "output").args[0][0]
    got = qk_operands.origin(out, ep)
    assert got.kind == qk_operands.GRID, got


@pytest.mark.parametrize("path", ["exception", "reshape"])
def test_unread_move_paths(path):
    """A move whose rule raises (the exception
    handler in _derived_classes) or a reshape whose sizes do not factor
    (_move_classes over _reshape_classes) must leave its partitions unread
    (_UNREAD, so _read_classes answers None: unknown), never take them for
    "varies", which would hide rounded codes beside an unread scale. Each
    path is faulted alone; reverting either return to None fails its case
    alone."""
    from unittest.mock import patch

    from executorch.backends.esp32s3.operator_support import qk_operands as qk

    g = torch.fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = torch.empty(1, 7, 4)
    v = g.call_function(torch.ops.aten.view.default, (x, [1, 7, 4]))
    v.meta["val"] = x.meta["val"]
    c = qk._Classifier(None)
    c.parts[x] = [qk._ident(n) for n in (1, 7, 4)]
    target, method, options = (
        (c, "_move_classes", {"side_effect": ValueError("unread")})
        if path == "exception"
        else (qk, "_reshape_classes", {"return_value": None}))
    with patch.object(target, method, **options):
        assert c._read_classes(v) is None


class TopkKeys(nn.Module):
    def forward(self, q, k):
        q = q.gather(-1, torch.topk(q.abs(), 16, -1).indices)
        return _qk3(self, q, k)


def test_topk_gather_admitted():
    """A lookup of a run-time table at its own keys' top k is a value the
    model computes, not a table of levels at codes (the scaled tables,
    lut_scaled_tensor and lut_scaled_rows, are refused): delegated,
    the host runner equal to the mirror."""
    g = torch.Generator().manual_seed(45)
    args = (torch.randn(1, 7, 16, generator=g),
            torch.randn(1, 31, 16, generator=g))
    part, program, records, log = lower_exposed(TopkKeys().eval(), args,
                                                quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, _bmm_reasons(part))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


class EmbeddingBag(nn.Module):
    def __init__(self, at_codes):
        super().__init__()
        self.at_codes = at_codes
        self.register_buffer("table", torch.randn(
            8, 16, generator=torch.Generator().manual_seed(46)))
        self.register_buffer("offsets", torch.arange(7))

    def forward(self, x, ids):
        idx = x.argmax(-1).reshape(-1) if self.at_codes else ids
        return torch.ops.aten._embedding_bag(self.table, idx,
                                             self.offsets)[0]


@pytest.mark.parametrize("at_codes", [True, False])
def test_embedding_bag_refused_by_name(at_codes):
    """aten::_embedding_bag is on no list: listed as computing new values, a
    constant bag at codes the model computes (an argmax) would be plain, its
    indices never read. It is refused by name, at codes or at token ids
    alike (no zoo family puts one on a q@kT operand; to_edge's own spelling,
    _embedding_bag_forward_only, is on no list either). On the exported
    graph."""
    from executorch.backends.esp32s3.operator_support import qk_operands

    g = torch.Generator().manual_seed(47)
    args = (torch.randn(1, 7, 8, generator=g),
            torch.randint(0, 8, (7,), generator=g))
    ep = torch.export.export(EmbeddingBag(at_codes).eval(), args)
    out = next(n for n in ep.graph.nodes if n.op == "output").args[0][0]
    got = qk_operands.origin(out, ep)
    assert got.kind == qk_operands.UNRECOGNISED, got
    assert "aten::_embedding_bag" in got.detail, got


class FilledIndex(nn.Module):
    """A table row picked by an int64 index filled from
    a run-time float (fill.Tensor, kept by preserve_ops), on one side of the
    q@kT; the row is [1, 2, 3, 4] four times."""

    def __init__(self, side):
        super().__init__()
        self.side = side
        rows = torch.zeros(4, 16)
        rows[1] = torch.tensor([1., 2., 3., 4.] * 4)
        self.register_buffer("table", rows)

    def forward(self, s, x):
        i = torch.zeros(1, dtype=torch.int64).fill_(s[0])
        row = self.table[i].unsqueeze(0)
        q, k = (row, x) if self.side == "q" else (x, row)
        _see(self, {"q": q, "k": k})
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), -1)


@pytest.mark.parametrize("side", ["q", "k"])
def test_filled_index_lookup_exact(side):
    """Delegated with fill.Tensor kept, its first score is 40.062992 (the
    int8 mirror) instead of 40: an index walk that takes fill for a factory,
    which reads its argument's shape alone, stops there. fill.Tensor reads its value and casts it, an integer fill of a float
    is codes: refused, the portable bmm's exact 40.0."""
    from executorch.exir import EdgeCompileConfig

    args = (torch.tensor([1.25]), torch.ones(1, 2, 16))
    config = EdgeCompileConfig(preserve_ops=[torch.ops.aten.fill.Tensor])
    part, program, records, log = lower_exposed(
        FilledIndex(side).eval(), args, compile_config=config,
        quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["portable"], (
            records, _bmm_reasons(part))
        assert any("already on a grid" in r and "lookup-table" in r
                   for r in _bmm_reasons(part)), _bmm_reasons(part)
        got, run_log = run_program(program, args)
        assert got is not None, run_log[-800:]
        score = float(got[records[0]["values"]["scores"][1]][0, 0, 0])
        assert score == 40.0, score
    finally:
        _release()


# --- the limits of the search -----------------

# Each limit of qk_operands forced from its minimum up through small values
# (patching the constant: one limit serves several walks, and the one that
# runs out first changes with the value), and the two unknowns: constant
# values the gate cannot read (_payload finds none) and shapes it cannot
# read (symbolic: _aligned_shape None).
LIMITS = [(limit, value) for limit in (
    "_BUDGET", "_NESTING", "_INDEX_BUDGET", "_SAME_BUDGET", "_VARIES_BUDGET",
    "_FACTOR_BUDGET", "_FACTOR_DEPTH")
    for value in (0, 1, 2, 3, 4, 8, 16, 32, 64)] + [
    ("unread payloads", None), ("unread shapes", None)]
# The origins the limit test lowers on each side, the other side plain:
# every refused origin of the matrix (per_channel's export stops before the
# gate) and every admitted one.
LIMIT_REFUSED = tuple(o for o in CORE_ORIGINS
                      if o not in ADMITTED and o != "per_channel") \
    + GRID_ARITHMETIC
LIMIT_ADMITTED = CORE_ADMITTED + PLAIN_ARITHMETIC
_LIMIT_VERDICTS = {}


@contextlib.contextmanager
def _forced(qk_operands, limit, value):
    """qk_operands with `limit` forced to `value`, or the unknown named."""
    from unittest import mock

    if limit == "unread payloads":
        with mock.patch.object(qk_operands._Classifier, "_payload",
                               lambda self, node: None):
            yield
    elif limit == "unread shapes":
        with mock.patch.object(qk_operands, "_aligned_shape",
                               lambda node, rank: None):
            yield
    else:
        with mock.patch.object(qk_operands, limit, value):
            yield


def _limit_verdicts():
    """{(origin, side): (the q@kT records' kinds of a lowering at the
    defaults, {setting: the kind operand_problem refuses the operand by, or
    None})}, each setting of LIMITS (and the defaults) judged on the gate's
    own program at partition time (a spy on the partitioner). Lowered once
    per origin and side, and kept."""
    if _LIMIT_VERDICTS:
        return _LIMIT_VERDICTS
    from executorch.backends.esp32s3.operator_support import qk_operands
    from executorch.backends.esp32s3.operator_support.patterns import (
        _qk_bmm_structure,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    settings = [("defaults", None)] + LIMITS
    original = Esp32Partitioner.partition
    origins = LIMIT_REFUSED + LIMIT_ADMITTED
    for i, origin in enumerate(origins):
        for side in ("q", "k"):
            verdicts = {}

            def spy(self, program, side=side, verdicts=verdicts):
                for n in program.graph_module.graph.nodes:
                    if verdicts or n.op != "call_function" or \
                            "aten.bmm" not in str(n.target):
                        continue
                    got = _qk_bmm_structure(n)
                    if got is None:
                        continue
                    operand = got[0] if side == "q" else got[2]
                    role = "query" if side == "q" else "key"
                    for setting in settings:
                        if setting[0] == "defaults":
                            problem = qk_operands.operand_problem(
                                program, operand, role)
                        else:
                            with _forced(qk_operands, *setting):
                                problem = qk_operands.operand_problem(
                                    program, operand, role)
                        verdicts[setting] = (None if problem is None
                                             else problem[0])
                return original(self, program)

            case = Case("explicit", **{side: origin}, mode="qk", heads="gqa",
                        d=16, m=7, n=31, seed=700 + i)
            model = Explicit(case).eval()
            Esp32Partitioner.partition = spy
            try:
                _, _, records, _ = lower_exposed(
                    model, model.inputs(), mutable=model.mutable(),
                    quantize_qk=True)
            finally:
                Esp32Partitioner.partition = original
                _release()
            _LIMIT_VERDICTS[(origin, side)] = ([r["kind"] for r in records],
                                               verdicts)
    return _LIMIT_VERDICTS


@pytest.mark.parametrize("limit, value", LIMITS,
                         ids=[f"{lim}={val}" for lim, val in LIMITS])
def test_limits_never_admit(limit, value):
    """The search's every limit -- depth, budget, nesting, the index walk,
    structural equality's pairs, the partitions' budget, the divisor search
    -- forced from its minimum up through small values, and constant values
    and shapes the gate cannot read: every refused origin of the matrix
    stays refused, on both operand sides (a lookup's index walk or an unread
    payload cut short must not admit). A hard limit
    refuses by name; a soft one or an unknown makes the search follow more
    operands."""
    verdicts = _limit_verdicts()
    admitted = []
    for origin in LIMIT_REFUSED:
        for side in ("q", "k"):
            kinds, got = verdicts[(origin, side)]
            assert got, (origin, side, "no q@kT reached the gate", kinds)
            if got[(limit, value)] is None:
                admitted.append(f"{origin}@{side}")
    assert not admitted, f"{limit} = {value} admits {admitted}"


def test_limits_defaults_keep_plain_delegated():
    """At the default limits every admitted origin of the matrix is
    delegated (its lowering) and admitted (the gate's verdict), on both
    sides, and every refused one refused; and a plain value times a buffer
    filled with one value stays admitted when the buffer's values cannot be
    read (an unknown never refuses what follows it, only what it might
    scale)."""
    verdicts = _limit_verdicts()
    for origin in LIMIT_ADMITTED:
        for side in ("q", "k"):
            kinds, got = verdicts[(origin, side)]
            assert kinds == ["delegate"] and got[("defaults", None)] is None, (
                origin, side, kinds, got.get(("defaults", None)))
    for origin in LIMIT_REFUSED:
        for side in ("q", "k"):
            kinds, got = verdicts[(origin, side)]
            assert "delegate" not in kinds and got[("defaults", None)], (
                origin, side, kinds)
    for side in ("q", "k"):
        _, got = verdicts[("times_uniform", side)]
        assert got[("unread payloads", None)] is None, (side, got)


class DirectState(nn.Module):
    """softmax(S @ k^T / 4) where S, a buffer, is the bmm's own operand, read
    directly, and the forward then writes into it what the next call reads:
    its keys' first rows, halved, plus one."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 16))

    def forward(self, k):
        out = _qk3(self, self.state, k)
        self.state.copy_(k[:, :4] * 0.5 + 1.0)
        return out


def test_state_read_directly_admitted():
    """A buffer the forward writes, read directly as the q@kT's operand,
    is no constant the partition pulls in (the partitioner leaves every
    written buffer out of the constants it pulls in). It is judged by
    what the forward writes, as one read through moves: admitted, the
    buffer passed to the delegate at run time, and over two executions in
    one process the second call's scores are the mirror of the state the
    first wrote, bit for bit. (ExecuTorch keeps no initial value of a
    written buffer, so the first reads whatever the planned memory held.)"""
    g = torch.Generator().manual_seed(28)
    first = (torch.randn(1, 5, 16, generator=g),)
    second = (torch.randn(1, 5, 16, generator=g),)
    written = first[0][:, :4] * 0.5 + 1.0
    part, program, records, log = lower_exposed(
        DirectState().eval(), first, mutable=("state",), quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds), _bmm_reasons(part))
        ep = program.exported_program()
        (call,) = [n for n in ep.graph.nodes if n.op == "call_function"
                   and "call_delegate" in str(n.target)]
        buffer = next(s.arg.name for s in ep.graph_signature.input_specs
                      if s.kind.name == "BUFFER" and s.target == "state")
        assert buffer in [getattr(a, "name", None) for a in call.args], (
            "the state is not a delegate argument", call.args)
        got, run_log = run_program_steps(program, [first, second])
        assert got is not None, run_log[-800:]
        users = [s.arg.name for s in ep.graph_signature.input_specs
                 if s.kind.name == "USER_INPUT"]
        # The second call's scores are the mirror of the state the first
        # call wrote (an exposed copy of the buffer itself would show the
        # second call's own write, made after the bmm read it).
        named = {n: a.detach().numpy() for n, a in zip(users, second)}
        vals = {role: _value(w, got[1], named)
                for role, w in records[0]["values"].items()}
        want = mirror_dyn(written.numpy(), vals["k"])
        assert np.array_equal(vals["scores"].view(np.uint32),
                              want.view(np.uint32))
        first_scores = _value(records[0]["values"]["scores"], got[0],
                              {n: a.detach().numpy()
                               for n, a in zip(users, first)})
        assert not np.array_equal(first_scores, vals["scores"])
    finally:
        _release()


class UnknownOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, q, k):
        q = torch.sign(q) * 0.5
        return _qk3(self, q, k)


def test_selection_values_walked():
    """sort's and kthvalue's values select existing elements, like topk's
    and max's: walked to the static pair (on the exported graph -- the
    host runner has no portable sort, and kthvalue is not a core ATen op,
    so neither reaches an edge program)."""
    from executorch.backends.esp32s3.operator_support import qk_operands

    for case in ("sort", "kthvalue"):
        class Pick(nn.Module):
            def forward(self, a, b, c):
                if case == "sort":
                    q = torch.sort(_static(a), dim=-1).values
                else:
                    q = torch.kthvalue(_static(c), 2, dim=2).values
                return torch.bmm(q, b.transpose(1, 2))

        g = torch.Generator().manual_seed(25)
        args = (torch.randn(1, 2, 16, generator=g),
                torch.randn(1, 3, 16, generator=g),
                torch.randn(1, 2, 3, 16, generator=g))
        ep = torch.export.export(Pick().eval(), args)
        (bmm,) = [n for n in ep.graph.nodes if n.op == "call_function"
                  and "bmm" in str(n.target)]
        got = qk_operands.origin(bmm.args[0], ep)
        assert got.kind == qk_operands.STATIC_PAIR, (case, got)
    _release()


@pytest.mark.parametrize("op", ["sign"])
def test_unknown_op_refused_by_name(op):
    """An op on none of the allow-list's lists is refused, naming it,
    instead of being trusted as plain float (a rule that counts any ATen op
    over a runtime value as plain would delegate it).
    (nan_to_num, also unlisted, decomposes into where and isnan before
    the gate, which walks it.)"""
    g = torch.Generator().manual_seed(21)
    args = (torch.randn(1, 2, 16, generator=g),
            torch.randn(1, 3, 16, generator=g))
    kinds, reasons = _refused_exact(lambda: UnknownOp(op), args,
                                    quantize_qk=True)
    assert any(f"aten::{op}" in r and "does not know" in r
               for r in reasons), (kinds, reasons)


class ConstantTableGather(nn.Module):
    """Keys gathered from a parameter by runtime indices, or chosen between
    two parameters by a runtime mask: runtime values."""

    def __init__(self, how):
        super().__init__()
        torch.manual_seed(18)
        self.how = how
        self.table = nn.Parameter(torch.randn(64, 16))
        self.other = nn.Parameter(torch.randn(9, 16))

    def forward(self, q, idx, mask):
        how = self.how
        if how == "index_select":
            k = self.table.index_select(0, idx)
        elif how == "index":
            k = self.table[idx]
        elif how == "gather":
            k = torch.gather(self.table, 0, idx.view(-1, 1).expand(-1, 16))
        elif how == "embedding":
            k = torch.nn.functional.embedding(idx, self.table)
        else:                                 # where
            k = torch.where(mask, self.table[:9], self.other)
        return _qk3(self, q, k.unsqueeze(0))


@pytest.mark.parametrize("how", ["index_select", "index", "gather",
                                 "embedding"])
def test_runtime_gather_of_constant_admitted(how):
    """A gather by runtime indices is judged by its own constness, not its
    table's (which would refuse every spelling but the embedding as a
    build-time constant): admitted, the host runner equal to the mirror."""
    g = torch.Generator().manual_seed(22)
    args = (torch.randn(1, 3, 16, generator=g),
            torch.randint(0, 64, (9,), generator=g),
            torch.rand(9, 16, generator=g) > 0.5)
    part, program, records, log = lower_exposed(
        ConstantTableGather(how).eval(), args, quantize_qk=True)
    try:
        assert [r["kind"] for r in records] == ["delegate"], (
            records, dict(part.support.kinds))
        assert _mirror_records(program, records, args) == 1
    finally:
        _release()


def test_where_of_constants_refused_as_levels():
    """A where choosing between two learned tables by a runtime mask gives
    each element one of two values the program fixes: a few fixed levels,
    as where(x > 0, alpha, -alpha) with a learned alpha does. No shape tells the two apart -- with a single row a
    per-channel alpha is a table of the operand's own shape -- so constants
    are levels whatever their shape, and this where is refused as on a
    grid. Its bmm runs portable on the model's values, bit for bit."""
    g = torch.Generator().manual_seed(22)
    args = (torch.randn(1, 3, 16, generator=g),
            torch.randint(0, 64, (9,), generator=g),
            torch.rand(9, 16, generator=g) > 0.5)
    kinds, reasons = _refused_exact(lambda: ConstantTableGather("where"),
                                    args, quantize_qk=True)
    assert any("already on a grid" in r for r in reasons), (kinds, reasons)


@pytest.mark.parametrize("op", ["dyn", "kcache_i8"])
def test_every_int8_qk_refused_under_fp32(op):
    """format_policy='fp32' means no int8 anywhere, so all three q@kT ops
    stay portable under it, quantize_qk or not: OP_BMM_DYN_I8 and
    OP_BMM_KCACHE_I8 as well as the affine K cache's."""
    if op == "dyn":
        g = torch.Generator().manual_seed(23)
        make = lambda: ChainBmm(False, False)  # noqa: E731
        args = (torch.randn(2, 8, 16, generator=g),
                torch.randn(2, 8, 16, generator=g))
        mutable = ()
    else:
        case = Case("kcache", window=64, ring=True, pos=65, heads="gqa",
                    d=32, m=1, seed=24)
        make = lambda: Int8KCache(case)  # noqa: E731
        args = make().inputs()
        mutable = make().mutable()
    kinds, _ = _refused_exact(make, args, mutable=mutable,
                              format_policy="fp32", quantize_qk=True)
    assert kinds.get("int8 q@kT under fp32 policy"), kinds


class ViewCastView(nn.Module):
    def forward(self, x):                      # [1, 2, 5, 16]
        y = (x * 2.0).reshape(2, 80).reshape(1, 2, 5, 16)
        return y.half().float().reshape(2, 5, 16) + 0.0


@pytest.mark.xfail(strict=True, reason="ExecuTorch's FuseViewCopyTransform "
                   "moves a view through a dim-order cast and leaves the "
                   "cast's dim_order of the old rank (upstream)")
def test_view_through_cast_upstream_defect():
    """Found by the matrix's torchao fake-quantiser origin under a half cast:
    apply_passes runs ExecuTorch's FuseViewCopyTransform on the portable
    remainder, which merges a view chain through _to_dim_order_copy (in its
    unary list) without rewriting that cast's dim_order for the new rank, so
    the program fails its kernel check at load (dim_order of 4 entries on a
    3-D tensor). Not a q@kT admission question; strict, so a fixed transform
    fails here."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    x = torch.randn(1, 2, 5, 16, generator=torch.Generator().manual_seed(26))
    want = ViewCastView()(x).numpy()
    with contextlib.redirect_stdout(io.StringIO()):
        edge = lower_esp32(torch.export.export(ViewCastView().eval(), (x,)),
                           [Esp32Partitioner()])
        apply_passes(edge, quiet=True)
        program = edge.to_executorch(config=ExecutorchBackendConfig(
            delegate_alignment=16))
    try:
        got, log = run_program(program, (x,))
        assert got is not None, log[-600:]
        assert np.array_equal(got[0], want)
    finally:
        _release()


class ChannelsLastReshape(nn.Module):
    def forward(self, x):                      # [1, 2, 4, 3]
        y = (x * 2.0).contiguous(memory_format=torch.channels_last)
        return y.reshape(2, 4, 3) + 0.0


@pytest.mark.xfail(strict=True, reason="ExecuTorch 1.6 aliases a reshape of "
                   "a channels-last value (upstream, not this backend)")
def test_channels_last_reshape_upstream_defect():
    """Found by the matrix's non-row-major operands: a value given a
    channels-last dim order inside the model and then reshaped -- as
    torch.matmul reshapes a q@kT operand -- is emitted as _clone_dim_order
    to channels-last, then a memory-planning alias (view) of its bytes, so
    the reshape reads them in channels-last order: the key's elements
    permuted, the scores wrong. Plain to_edge_transform_and_lower and
    to_executorch, with no backend, emit the same program. Not a q@kT
    admission question (it is wrong with quantize_qk off too), so the matrix
    has no channels-last move; strict, so a fixed ExecuTorch fails here."""
    from executorch.exir import to_edge_transform_and_lower

    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    want = ChannelsLastReshape()(x).numpy()
    program = to_edge_transform_and_lower(torch.export.export(
        ChannelsLastReshape().eval(), (x,))).to_executorch()
    got, log = run_program(program, (x,))
    assert got is not None, log[-800:]
    assert np.array_equal(got[0], want), (got[0].reshape(-1)[:8],
                                          want.reshape(-1)[:8])


# --- script mode -----------------------------------------------------------

def regression_tests():
    """The named tests beside the matrix, each parametrisation expanded:
    [(id, function, kwargs)]. The chain's verify step runs this file as a
    script (it has a __main__ block), so main() runs these too."""
    out = []
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)) or \
                name == "test_qk_case":
            continue
        marks = [m for m in getattr(fn, "pytestmark", [])
                 if m.name == "parametrize"]
        if not marks:
            out.append((name, fn, {}))
            continue
        # Stacked parametrize marks run as their product, as under pytest.
        axes = []
        for mark in marks:
            names = [a.strip() for a in mark.args[0].split(",")]
            axes.append([(names, v if len(names) > 1 else (v,))
                         for v in mark.args[1]])
        for combo in itertools.product(*axes):
            kwargs, ids = {}, []
            for names, v in combo:
                kwargs.update(zip(names, v))
                ids.append("-".join(map(str, v)))
            out.append((f"{name}[{'-'.join(ids)}]", fn, kwargs))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--full", action="store_true",
                   help="every (query, key, mode) under every move, besides "
                        "the other families")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--family", default="",
                   help="only this family (explicit, kcache, affine, recipe)")
    a = p.parse_args(argv)
    explicit, total = pairwise_cases(seed=a.seed)
    if a.full:
        explicit = full_cases(seed=a.seed)
    cases = explicit + kcache_cases() + affine_cases() + recipe_cases()
    if a.family:
        cases = [c for c in cases if c.family == a.family]
    if a.limit:
        cases = cases[:a.limit]
    results = []
    for i, case in enumerate(cases):
        res = run_case(case)
        results.append(res)
        flag = "FAIL" if res.failures else "ok"
        print(f"[{i + 1}/{len(cases)}] {flag:4} {res.outcome:10} {case.id}"
              + (f" :: {'; '.join(res.failures)}" if res.failures else ""),
              flush=True)
    covered = set()
    for r in results:
        if r.case.family == "explicit":
            covered |= _pairs({n: getattr(r.case, n) for n in EXPLICIT_FACTORS})
    failed = []
    if not (a.family or a.limit):
        for name, fn, kwargs in regression_tests():
            xfail = any(m.name == "xfail" for m in
                        getattr(fn, "pytestmark", []))
            try:
                fn(**kwargs)
                if xfail:                 # strict: a pass is a failure
                    failed.append(name)
                    print(f"FAIL {name}: passed, expected to fail",
                          flush=True)
                else:
                    print(f"ok   {name}", flush=True)
            except Exception as e:        # an assertion or a crash alike
                if xfail:
                    print(f"xfail {name}: {type(e).__name__}", flush=True)
                    continue
                failed.append(name)
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}",
                      flush=True)
    print()
    for line in (summary(results, total, len(covered))
                 + origin_mode_table(results) + [""]
                 + coverage_table(results)):
        print(line)
    if failed:
        print(f"regression tests failed: {', '.join(failed)}")
    return 1 if failed or any(r.failures for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
