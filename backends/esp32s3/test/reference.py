# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host reference arithmetic for the delegate's integer kernels."""

import numpy as np
import torch
import torch.nn as nn

from executorch.backends.esp32s3.operator_support.conv import (
    ConvGeom,
    _src_index,
    split_groups,
)
from executorch.backends.esp32s3.quantizer import mirrors
from executorch.backends.esp32s3.quantizer.mirrors import a0_inverse, fma_f32
from executorch.backends.esp32s3.serialization.weight_formats import (
    SCALE_GROUP,
    quantize_absmax_rows,
)


def addmm_k_outer(x, W, b):
    """Opcode 1 (inline fp32 [K,N] weights), bit for bit: the bias, then a
    k-outer pass per row whose `yrow[n] += xk * wrow[n]` the S3 build fuses
    into one madd.s (run_addmm_fp32 in the parity firmware), so each
    step rounds once. fp32 addition is not associative, so order is part of
    the contract.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    y = np.broadcast_to(np.asarray(b, dtype=np.float32),
                        (x.shape[0], W.shape[1])).copy()
    for k in range(x.shape[1]):
        y = fma_f32(x[:, k:k + 1], W[k][None, :], y)
    return y


def dense_act(act_qmax=127, sx=None, ties_to_even=False):
    """The activation quantiser of the delegate's own quantised matmul, as a
    mirrors.py scheme: A0 -- each row's absmax / act_qmax, x * fl(1/sx)
    rounded half away from zero and clipped to +/-act_qmax, as quantize_row_t
    does -- or one scale `sx` for every row: FLAG_STATIC_ACT's calibrated one,
    or a PT2E per-tensor scale (FLAG_ACT_PER_TENSOR), which the device rounds
    ties to even as torch.round does. mirrors.affine_matmul then dots
    exactly (int32 for int8 codes, int64 for i16's) and fuses the epilogue
    with fused_madd, as every symmetric loop does (device_epilogue).
    """
    from dataclasses import replace

    import numpy as np

    act = replace(mirrors.A0, qmin=-act_qmax, qmax=act_qmax)
    if sx is not None:
        act = act.static(float(np.float32(sx)))
    if ties_to_even:
        act = replace(act, rounding="rne_before_zp")
    return act


def addmm_iq_absmax(x, w_nk, b, fmt="i8", sx_fixed=None, ties_to_even=False,
                    group=None, fp16_scales=False):
    """The device quantised path, modelled exactly, at any rung of the format ladder.

    w_nk is the Linear's own [N,K] weight (which is why the backend permutes back to
    it -- addmm wants [K,N], the kernel wants [N,K], and nn.Linear stores [N,K]).
    Accumulation is exact integer and the float epilogue fused, mirrors.py's
    arithmetic under dense_act's scheme, the device's to the bit. `group` and
    `fp16_scales` are the backend's group_weights format: one scale per
    SCALE_GROUP codes, stored as halves.

    Weight precision and ACTIVATION precision are separate, and conflating them is a
    mistake: the i4 rung stores 4-bit weights
    but still quantises activations to int8, because the nibbles are unpacked into int8
    and multiplied by the int8 PIE kernel. Narrowing the activation to match the weight
    would discard precision for no saving -- activations are quantised per row as the op
    runs (Stage B's static int8 codes between delegates aside), not streamed like
    weights, so there is no traffic argument for them at any rung. Hence f.qmax for the weights
    and f.act_qmax for the activation, which are equal at i8 and i16 and differ at i4.

    The accumulator width follows: i16 needs int64, since 32767^2 overflows int32 after
    two terms -- wrong for every K > 1, not merely for a large one.

    Row padding is not modelled because it cannot matter: the pad is zero on both sides
    so it contributes nothing to any dot. i4's nibble unpack is likewise invisible here,
    because it reconstructs exactly the values quantize_absmax_rows produced -- which is
    precisely the property the device run is checking.
    """
    import numpy as np

    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        quantize_absmax_rows,
    )

    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    qw, sw = quantize_absmax_rows(w_nk, f, group=group, fp16_scales=fp16_scales)
    # sx_fixed models FLAG_STATIC_ACT: one calibrated per-tensor scale instead
    # of per-row absmax. Same rounding, same clip, same everything after --
    # unless ties_to_even, for a PT2E per-tensor scale (FLAG_ACT_PER_TENSOR),
    # which the device rounds as torch.round does. No exactness bound: the i16
    # rung dots in int64, and the export refuses an int8 row too long for int32.
    return mirrors.affine_matmul(
        np.asarray(x, dtype=np.float32), qw, sw, b,
        act=dense_act(f.act_qmax, sx_fixed, ties_to_even),
        gsize=None if sw.ndim == 1 else group, fused_madd=True,
        check_bound=False).y


def addmm_i8_absmax(x, w_nk, b):
    """The i8 rung, kept under its own name because it is what most callers mean."""
    return addmm_iq_absmax(x, w_nk, b, "i8")


def embedding_iq_absmax(table, idx, fmt="i8"):
    """The device embedding path: gather a quantised row and dequantise it.

    No arithmetic to model -- a gather reads one row per index and scales it, so unlike
    the matmul path there is no activation quantisation, no accumulator and no rounding
    beyond the weight quantisation itself. Which is also why the reference is this
    short: quantise the table exactly as the backend does, then index it.

    Scales are grouped by SCALE_GROUP along the row and, at i4/i8, stored fp16, as the
    backend stores them. The fp16 rounding happens *before* quantisation and has to here
    too: the device dequantises with the stored half, so a reference that divided by the
    fp32 scale and multiplied back by the half would disagree by the rounding error of
    every scale.
    """
    import numpy as np

    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        SCALE_GROUP,
        quantize_absmax_rows,
    )
    from executorch.backends.esp32s3.espressif_preprocess import embedding_scale_layout

    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    table = np.ascontiguousarray(table)
    k = table.shape[1]
    # The backend's own rule, not a copy of it: group only when a row holds more than
    # one group, fp16 scales only when grouping and never at i16. A single-group table
    # keeps fp32 so that a tied tensor is byte-identical to the matmul path's copy.
    group, fp16 = embedding_scale_layout(k, f)
    q, scale = quantize_absmax_rows(table, f, group=group, fp16_scales=fp16)
    if scale.ndim == 1:
        return (q[idx].astype(np.float32)
                * scale[idx][:, None].astype(np.float32)).astype(np.float32)
    # Expand each row's per-group scales back to per-element using FIXED groups of
    # SCALE_GROUP with a short final one -- the quantiser's own boundaries. Deriving a
    # group size from the count instead (ceil(k/n_groups)) reconstructs different
    # boundaries whenever k is not a multiple of SCALE_GROUP, and dequantises every
    # element in the straddled region with the wrong group's scale. A device with the
    # same mistake would agree with such a reference, so parity cannot see it;
    # test_formats.py pins the boundary definition against the quantiser directly.
    per_elem = np.repeat(scale, SCALE_GROUP, axis=1)[:, :k]
    return (q[idx].astype(np.float32) * per_elem[idx].astype(np.float32)
            ).astype(np.float32)


def pt2e_tensor_scale(amax):
    """PT2E's dynamic per-tensor activation scale, as the delegate folds it.

    choose_qparams_symmetric's absmax / 127 in float32, floored at the
    observer's eps. strip_dynamic_act_qdq folds only chains at the default eps
    (float32's), which is the value the runtime hard-codes. It is scheme A1,
    so the arithmetic is mirrors.py's.
    """
    return mirrors.symmetric_scale(amax, -127, 127,
                                         mirrors.FLT_EPSILON)


def addmm_codes_ref(x, codes_nk, scales, bias, gsize, per_tensor=False,
                    tensor_amax=None, fused_madd=True):
    """Device-exact reference for a PASS-THROUGH quantised matmul.

    Unlike addmm_iq_absmax this never quantises weights -- the codes and scales arrive
    from the graph (torchao's, or eventually a frontend loader's) and are consumed
    verbatim, which is the entire point of the pass-through path. Only the activation
    is quantised, dynamically per row, exactly as the device does it.

    `per_tensor=True` models a PT2E dynamic per-TENSOR activation scale, which the
    delegate honours instead of substituting its own per-row one: one absmax
    over the whole activation, exactly as `choose_qparams_symmetric` computes it.
    The two differ by ~1.9% on this model, so which one the reference uses is not
    a detail -- it is the difference between checking the device and checking a
    scheme the device does not use. The per-tensor scheme is PT2E's throughout:
    the scale is floored at its eps (pt2e_tensor_scale) and rounding is
    torch.round's ties-to-even, where the delegate's own per-row scale rounds
    half away from zero.

    Accumulation is per GROUP with the partial sums scaled and added in group order,
    mirroring the runtime's loop -- bias first, then += (float)acc_g * sx * s[n,g],
    whose multiply-add the device fuses -- so the float expression tree per output
    element is the device's, not a mathematically-equal rearrangement of it
    (mirrors.device_epilogue). fused_madd=False is the same sum rounded
    twice per group, the host runner's (built -ffp-contract=off), for a
    reference held to the host's bits rather than the chip's.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    codes = np.asarray(codes_nk)
    scales = np.atleast_2d(np.asarray(scales, dtype=np.float32))
    if scales.shape[0] != codes.shape[0]:
        scales = scales.T
    if per_tensor:
        # `tensor_amax` lets a caller supply the per-tensor absmax over a tensor
        # WIDER than x: a conv's device scale is the absmax of the whole NCHW
        # input, while x here is its im2col matrix, which can omit input
        # elements no patch reads (stride 2 under a 1x1 kernel, say).
        sx = pt2e_tensor_scale(np.abs(x).max() if tensor_amax is None
                               else tensor_amax)
    else:
        sx = None
    return mirrors.affine_matmul(
        x, codes, scales, bias, act=dense_act(127, sx, ties_to_even=per_tensor),
        gsize=gsize, fused_madd=fused_madd, check_bound=False).y


def conv_codes_ref(x, codes_nk, scales, bias, geom, per_tensor=True):
    """Device-exact reference for a PASS-THROUGH quantised convolution.

    The conv delegate is the matmul delegate behind an im2col gather, so its
    pass-through reference is `addmm_codes_ref` behind the same gather
    (`conv_support.im2col_reference`). The codes are the [C_out, K] the
    resolver reshaped from PT2E's [C_out, C_in, kh, kw]; only the activation
    is quantised, at the ONE absmax of the whole NCHW input when the model
    quantised per tensor, which is how the runtime computes it for a conv.
    Returns [1, C_out, d_out, h_out, w_out]; callers reshape to their rank.
    """
    import numpy as np
    import torch

    from executorch.backends.esp32s3.test.reference import im2col_reference

    cols = im2col_reference(torch.as_tensor(x).detach(), geom).numpy()
    amax = float(np.abs(np.asarray(x, dtype=np.float32)).max()) if per_tensor else None
    y = addmm_codes_ref(cols.astype(np.float32), codes_nk, scales, bias,
                        codes_nk.shape[1], per_tensor=per_tensor,
                        tensor_amax=amax)                     # [m, N]
    return torch.from_numpy(np.ascontiguousarray(y.T)).reshape(
        1, geom.n, geom.d_out, geom.h_out, geom.w_out).float()


def sdpa_qk_i8(q, k, scale=None):
    """OP_BMM_DYN_I8 in an SDPA-spelled graph (SdpaToQk's scaled rows): the
    raw scores the delegate writes for F.scaled_dot_product_attention(q, k,
    ...) under quantize_qk, bit for bit.

    ATen's math decomposition multiplies BOTH operands by s = sqrt(scale)
    before the bmm (sdp::calculate_scale: scale, or 1/sqrt(D), and its square
    root, in float64), the portable mul rounds each product to float32 at
    fl32(s), and SdpaToQk only moves the key's multiply above its transpose.
    So the delegate quantises the rows of fl(q*s) and fl(k*s), not of q and
    k: other codes and scales than bmm_qk_i8(q, k) * scale, and the scores
    come out already scaled. q [..., M, D]; k [..., N, D] in its natural
    layout with q's leading dims (repeat GQA heads first, as enable_gqa's
    repeat_interleave does). Returns [..., M, N] float32."""
    import math

    import numpy as np

    if scale is not None and not scale > 0:
        raise ValueError("a negative or zero SDPA scale negates the query; "
                         "not modelled")
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q {q.shape} and k {k.shape} do not pair")
    width = q.shape[-1]
    factor = np.float32(math.sqrt(1.0 / math.sqrt(width) if scale is None
                                  else scale))
    qs = (q * factor).astype(np.float32)
    ks = (k * factor).astype(np.float32)
    out = np.empty(q.shape[:-1] + (k.shape[-2],), dtype=np.float32)
    for index in np.ndindex(*q.shape[:-2]):
        out[index] = bmm_qk_i8(qs[index], ks[index])
    return out


def bmm_qk_i8(q, kx):
    """The device OP_BMM_DYN_I8 path, modelled exactly: both operands int8 at
    per-vector absmax, integer accumulate, scores scaled (acc*sx)*sk in that
    order. q is [M,K]; kx is the K side in its NATURAL [N,K] layout -- the
    transpose was folded at export, so key vectors are contiguous rows on both
    sides, which is also the K-cache layout a real board would read."""
    import numpy as np

    from executorch.backends.esp32s3.serialization.weight_formats import round_half_away

    M, K = q.shape
    N = kx.shape[0]

    def qrow(row):
        amax = np.abs(row).max()
        sx = np.float32(amax / np.float32(127.0))
        inv = a0_inverse(sx)  # 0 for a zero or subnormal scale
        return np.clip(round_half_away(row * inv), -127, 127).astype(np.int32), sx

    kq = np.empty((N, K), dtype=np.int32)
    sk = np.empty(N, dtype=np.float32)
    for n in range(N):
        kq[n], sk[n] = qrow(kx[n])
    y = np.empty((M, N), dtype=np.float32)
    for m in range(M):
        qq, sx = qrow(q[m])
        for n in range(N):
            acc = int(np.dot(qq, kq[n]))
            y[m, n] = (np.float32(acc) * sx) * sk[n]
    return y


def bmm_f32_portable(lhs, rhs):
    """The device's fp32 bmm (ops/espressif_ops.cpp shim_bmm_out), bit for bit.

    PyTorch is free to dispatch eager ``torch.bmm`` to an optimized BLAS-style
    kernel whose reduction tree differs from the device's, and stateful decode
    magnifies those one-op rounding differences over later tokens, so the
    device oracles need the device's order: each output a float32 accumulator
    from 0, k ascending, every step ONE rounding, fl(sum + lhs * rhs) as an
    fma (madd.s on the chip, std::fma in the shim so the host runner computes
    the same). Rounding the product and the sum separately would be the host
    runner's -ffp-contract=off arithmetic, not the chip's (three
    probabilities of fl(1/3) over V = [0.1, 0.2, 0.3] give 0x3e4ccccd fused,
    0x3e4cccce unfused).

    ``lhs`` and ``rhs`` use ordinary bmm layouts [B,M,K] and [B,K,N].
    """
    import numpy as np

    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    if lhs.ndim != 3 or rhs.ndim != 3:
        raise ValueError("portable bmm expects two rank-three tensors")
    batches, rows, inner = lhs.shape
    if rhs.shape[:2] != (batches, inner):
        raise ValueError("portable bmm operands have incompatible shapes")
    columns = rhs.shape[2]
    output = np.zeros((batches, rows, columns), dtype=np.float32)
    for index in range(inner):
        output = fma_f32(lhs[:, :, index:index + 1], rhs[:, index:index + 1, :],
                         output)
    return output


def rope_f32_device(x, cos, sin, rot_dim=None):
    """The device's RoPE (ops/espressif_ops.cpp shim_rope, espressif::rope),
    bit for bit: per rotated pair (a, b) = (x[i], x[i + half]),

        out[i]        = fma(-b, s[i], fl(a * c[i]))      mul.s, then msub.s
        out[i + half] = fma(b, c[j], fl(a * s[j]))       mul.s, then madd.s

    one rounding per output (fma_f32), with j = i + half for full-width
    [.., rot] tables and j = i for this repo's pairwise [.., rot/2] tables,
    whose one half both output halves share; lanes past rot_dim pass
    through. cos and sin broadcast against x[..., :half]. torch's spelling
    (and the op's Composite) rounds each product: 1 ulp off in about a
    quarter of outputs.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    cos = np.asarray(cos, dtype=np.float32)
    sin = np.asarray(sin, dtype=np.float32)
    rot = x.shape[-1] if rot_dim is None else int(rot_dim)
    half = rot // 2
    a, b = x[..., :half], x[..., half:2 * half]
    c0, s0 = cos[..., :half], sin[..., :half]
    if cos.shape[-1] == rot:
        c1, s1 = cos[..., half:rot], sin[..., half:rot]
    else:
        c1, s1 = c0, s0
    first = fma_f32(-b, s0, (a * c0).astype(np.float32))
    second = fma_f32(b, c1, (a * s1).astype(np.float32))
    return np.concatenate([first, second, x[..., 2 * half:]], axis=-1).astype(
        np.float32)


def softmax_f32_portable(value, dim=-1):
    """The device's fp32 softmax (ops/espressif_ops.cpp shim_softmax_out),
    bit for bit.

    Per row: the maximum as std::max(x, max) folds it from the first element,
    then fl(x - max) through newlib's expf (mirrors.expf_newlib, the chip's
    libm; the host runner's shim runs the same transcription), the sum in
    index order from the first exponential, and each exponential divided by
    it. np.exp is not the chip's expf: it differs in the last bit of about one
    exponential in ten, which is enough to flip prefill's int8 codes.
    """
    import numpy as np

    value = np.asarray(value, dtype=np.float32)
    axis = dim if dim >= 0 else value.ndim + dim
    if axis < 0 or axis >= value.ndim:
        raise ValueError("softmax dimension is out of range")
    if value.size == 0:
        # No element to reduce or write, as the portable kernel the shim falls
        # back to for an empty tensor: an empty result of the same shape.
        # (reshape(-1, 0) cannot infer its row count, so this comes first.)
        return np.empty_like(value)
    moved = np.moveaxis(value, axis, -1)
    width = moved.shape[-1]
    rows = moved.reshape(-1, width)
    output = np.empty_like(rows)
    if rows.size:
        maximum = rows[:, 0].copy()
        for index in range(1, width):
            column = rows[:, index]
            maximum = np.where(column < maximum, maximum, column)
        with np.errstate(invalid="ignore"):
            exponentials = mirrors.expf_newlib(
                (rows - maximum[:, None]).astype(np.float32))
        total = exponentials[:, 0].copy()
        for index in range(1, width):
            total = (total + exponentials[:, index]).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            output = (exponentials / total[:, None]).astype(np.float32)
    return np.moveaxis(output.reshape(moved.shape), -1, axis)


def rmsnorm_f32_s3_reference(value, weight, eps):
    """Host transcription of ``ops/esp32s3/rmsnorm_s3.S``.

    It preserves the four fused sum-of-squares accumulators, reduction order,
    two fused Newton refinements, and ``(weight*x)*scale`` multiplication order.
    ``rsqrt0.s`` itself is unavailable on the host; starting from correctly
    rounded reciprocal sqrt is sufficient after the target's two refinements
    and leaves at most its documented last-ulp seed uncertainty.
    """
    import numpy as np

    value = np.asarray(value, dtype=np.float32)
    width = value.shape[-1]
    rows = value.reshape(-1, width)
    scale_weight = (None if weight is None
                    else np.asarray(weight, dtype=np.float32).reshape(width))
    output = np.empty_like(rows)
    inverse_width = np.float32(np.float32(1.0) / np.float32(width))
    epsilon = np.float32(eps)

    def fused_add_product(accumulator, left, right):
        return np.float32(
            np.float64(accumulator)
            + np.float64(left) * np.float64(right))

    def fused_sub_product(accumulator, left, right):
        return np.float32(
            np.float64(accumulator)
            - np.float64(left) * np.float64(right))

    for row_index, row in enumerate(rows):
        accumulators = [np.float32(0.0) for _ in range(4)]
        vector_width = width - width % 4
        for index in range(0, vector_width, 4):
            for lane in range(4):
                item = row[index + lane]
                accumulators[lane] = fused_add_product(
                    accumulators[lane], item, item)
        total = accumulators[0]
        for lane in (1, 2, 3):
            total = np.float32(total + accumulators[lane])
        for index in range(vector_width, width):
            total = fused_add_product(total, row[index], row[index])

        mean = np.float32(total * inverse_width)
        radicand = np.float32(mean + epsilon)
        reciprocal = np.float32(
            1.0 / np.sqrt(np.float64(radicand)))
        half_radicand = np.float32(np.float32(0.5) * radicand)
        for _ in range(2):
            square = np.float32(reciprocal * reciprocal)
            correction = fused_sub_product(
                np.float32(1.5), half_radicand, square)
            reciprocal = np.float32(reciprocal * correction)

        for index in range(width):
            item = row[index]
            if scale_weight is not None:
                item = np.float32(scale_weight[index] * item)
            output[row_index, index] = np.float32(item * reciprocal)
    return output.reshape(value.shape)


def im2col_reference(x, geom: ConvGeom):
    """The exact gather the runtime will do, as a [m, K] host tensor.

    Column order is (c, i, j) -- the flattening order of the [C_out,C_in,kh,kw]
    weight -- so the packed weight and the gathered row agree by construction
    rather than by a convention written down in two places. In NCHW that makes
    the innermost run `kw` contiguous elements; NHWC would give runs of `C_in`
    and is reachable later by permuting the weight at build time, which is a
    change to a constant rather than to kernel code.
    """
    import torch

    # Normalise every rank to [1, C, D, H, W]. Views, so free, and it means the
    # loop below is written once.
    if x.dim() == 3:                # [1,C,L]
        x = x.unsqueeze(2).unsqueeze(2)
    elif x.dim() == 4:              # [1,C,H,W]
        x = x.unsqueeze(2)
    cols = torch.zeros(geom.m, geom.k, dtype=x.dtype)
    for od in range(geom.d_out):
        for oh in range(geom.h_out):
            for ow in range(geom.w_out):
                row = (od * geom.h_out + oh) * geom.w_out + ow
                t = 0
                for c in range(geom.c_in):
                    for dd in range(geom.kd):
                        for i in range(geom.kh):
                            for j in range(geom.kw):
                                idd, ih, iw = _src_index(geom, od, oh, ow,
                                                         dd, i, j)
                                if (idd is not None and ih is not None
                                        and iw is not None):
                                    cols[row, t] = x[0, c, idd, ih, iw]
                                t += 1
    return cols


def conv_iq_absmax(x, weight, bias, geom: ConvGeom, fmt="i8",
                   sx_fixed=None):
    """The device's quantised conv path, modelled exactly.

    The whole point of the im2col design is that this reduces to the matmul
    reference already proven bit-exact against the device: gather the patches,
    reshape the weight to [N,K], and hand both to `addmm_iq_absmax`. If this
    needed its own arithmetic, conv would need its own kernels -- that it does
    not is the design working.

    So the only conv-specific thing here is the gather, which is why the gather
    is what has to be right and nothing else is.
    """
    from executorch.backends.esp32s3.test.reference import addmm_iq_absmax

    import torch

    # detach(): the reference quantiser drops to numpy, and an nn.Module's
    # parameters carry requires_grad even under no_grad.
    import numpy as np

    # The reference quantiser works in numpy; converting at the boundary keeps
    # that its business rather than making every caller know it.
    cols = im2col_reference(x.detach(), geom).numpy()     # [m, K]
    w_nk = weight.detach().reshape(geom.n, geom.k).numpy()  # [N, K]
    b = (bias.detach().numpy() if bias is not None
         else np.zeros(geom.n, dtype=np.float32))
    # sx_fixed threads the CALIBRATED activation scale through unchanged: a
    # conv under static_act uses one scale for the whole op, which is exactly
    # what the matmul reference already models.
    y = np.asarray(addmm_iq_absmax(cols, w_nk, b, fmt=fmt,
                                   sx_fixed=sx_fixed))   # [m, N]
    # Always [1, N, d_out, h_out, w_out]. Callers reshape to their rank rather
    # than this guessing it: for a conv2d d_out is 1, for a conv1d h_out is too,
    # and a size-1 spatial extent is also legitimate at any rank, so squeezing
    # here would be ambiguous exactly where it mattered.
    return torch.from_numpy(y.T.copy()).reshape(
        1, geom.n, geom.d_out, geom.h_out, geom.w_out).float()


def depthwise_iq_absmax(x, weight, bias, geom: ConvGeom, fmt="i8",
                        group_weights=False, fp16_scales=True):
    """The device's DEPTHWISE arithmetic, modelled exactly.

    Separate from `conv_iq_absmax` because the arithmetic genuinely differs, and
    that difference is the whole reason depthwise gets its own runtime path:
    every output channel reads its OWN input channel, so the activation being
    quantised varies with the output column as well as the row. A matmul's does
    not -- one row of activation serves every column -- which is precisely what
    `addmm_iq_absmax` assumes and why depthwise cannot borrow it.

    Weight quantisation is per row of `[C, kd*kh*kw]`, which for depthwise is
    per CHANNEL: exactly the granularity the op wants, for free.

    Every float step is the device's, in float32 and in its order:
    quantize_conv_row's `sx = amax / QMAX` and `1 / sx` are float32 divisions
    and each code is `x * (1/sx)` rounded half away, never a division by sx;
    then accumulate_groups' `y += (float)acc * sx * scale`, whose final
    multiply-add the S3 build fuses into one madd.s (the disassembly of the
    i8/i4 loop and of run_addmm_i16_depthwise both show `float.s`/`__floatdisf`,
    `mul.s`, `madd.s`), so it rounds once (fma_f32). Dividing in float64, or
    adding the bias after one double expression, would change tens to hundreds
    of outputs per parity row; this reproduces the device's output bits on all
    six depthwise rows (test_conv_support.py).
    """
    import numpy as np
    import torch

    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        SCALE_GROUP,
        quantize_absmax_rows,
        round_half_away,
    )
    from executorch.backends.esp32s3.quantizer.mirrors import fma_f32

    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    if x.dim() == 3:
        x = x.unsqueeze(2).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    x = x.detach()
    c = geom.c_out
    w_nk = weight.detach().float().reshape(c, geom.k).numpy()
    group = (SCALE_GROUP
             if group_weights and geom.k > SCALE_GROUP else None)
    wq, sw = quantize_absmax_rows(
        w_nk, f, group=group,
        fp16_scales=bool(group is not None and fp16_scales))
    b = (bias.detach().float().numpy() if bias is not None
         else np.zeros(c, dtype=np.float32))

    # int32 holds every i8/i4 dot (the device's ACCX reads back 32 bits); the
    # i16 loop accumulates in int64, and its __floatdisf rounds like this cast.
    acc_dtype = np.int64 if f.bits > 8 else np.int32
    qmax = np.float32(f.act_qmax)
    scales = sw.reshape(c, -1)
    gsize = SCALE_GROUP if sw.ndim == 2 else geom.k
    out = np.zeros((c, geom.m), dtype=np.float32)
    for ch in range(c):
        # One channel's geometry is a 1-channel conv, so the shared gather
        # serves it: this is why depthwise needs a new KERNEL but no new
        # geometry, no new packing and no new format handling.
        one = ConvGeom(**{**vars(geom), "c_in": 1, "c_out": 1, "groups": 1,
                          "group_idx": 0})
        cols = im2col_reference(x[:, ch:ch + 1], one).numpy()      # [m, K]
        cols = cols.astype(np.float32)
        # Out-of-bounds taps gather as 0, so they neither raise the absmax
        # nor get a nonzero code -- as the device skips them and writes 0.
        sx = (np.abs(cols).max(axis=1) / qmax).astype(np.float32)
        inv = a0_inverse(sx)  # 0 for a zero or subnormal scale
        q = np.clip(round_half_away((cols * inv[:, None]).astype(np.float32)),
                    -f.act_qmax, f.act_qmax).astype(acc_dtype)
        value = np.full(geom.m, b[ch], dtype=np.float32)
        for gi, start in enumerate(range(0, geom.k, gsize)):
            stop = min(start + gsize, geom.k)
            acc = q[:, start:stop] @ wq[ch, start:stop].astype(acc_dtype)
            term = (acc.astype(np.float32) * sx).astype(np.float32)
            value = fma_f32(term, scales[ch, gi], value)
        out[ch] = value
    return torch.from_numpy(out).reshape(
        1, c, geom.d_out, geom.h_out, geom.w_out)


def grouped_iq_absmax(x, weight, bias, geom: ConvGeom, fmt="i8"):
    """Reference for a grouped conv: the dense reference, run per group."""
    import torch

    per_in = geom.c_in
    per_out = geom.c_out // geom.groups
    if x.dim() == 3:
        x = x.unsqueeze(2).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    parts = []
    for gi, sub in enumerate(split_groups(geom)):
        xs = x[:, gi * per_in:(gi + 1) * per_in]
        ws = weight[gi * per_out:(gi + 1) * per_out]
        bs = None if bias is None else bias[gi * per_out:(gi + 1) * per_out]
        parts.append(conv_iq_absmax(xs, ws, bs, sub, fmt=fmt))
    return torch.cat(parts, dim=1)


def _device_quantized_reference(model, *, embedding_format="i8",
                                dense_format="i8"):
    """Monkeypatch learned ops to the grouped runtime arithmetic.

    dense_format is the rung every nn.Linear executes at on the device: "i8"
    for format_policy="i8" exports, "i4" for the story Glimmer's
    format_policy="i4" (every projection and the head at 4 bits).
    """

    def quantized(weight, fmt="i8", *, table=False):
        weight = weight.detach().numpy().astype(np.float32)
        width = weight.shape[1]
        if table:
            # The sparse tables reach the device through
            # _replace_glimmer_tables_i4, which quantises every row with
            # group=min(128, width) and fp16 scales whatever the width. The
            # matmul rule below (per-row fp32 scales under 128 wide) would give
            # an 80-wide token table different int4 codes from the ones the
            # device gathers.
            group = min(SCALE_GROUP, width)
            codes, scales = quantize_absmax_rows(
                weight, fmt, group=group, fp16_scales=True)
            if scales.ndim == 1:
                scales = scales[:, None]
            return codes, scales, group
        group = SCALE_GROUP if width > SCALE_GROUP else None
        codes, scales = quantize_absmax_rows(
            weight, fmt, group=group, fp16_scales=group is not None)
        return codes, scales, (SCALE_GROUP if group is not None else width)

    def linear_forward(module):
        codes, scales, group = quantized(module.weight, dense_format)
        bias = (module.bias.detach().numpy().astype(np.float32)
                if module.bias is not None
                else np.zeros(codes.shape[0], dtype=np.float32))

        def forward(inp):
            width = inp.shape[-1]
            flat = inp.detach().reshape(-1, width).contiguous().numpy().astype(
                np.float32)
            out = addmm_codes_ref(flat, codes, scales, bias, group)
            return torch.from_numpy(out).reshape(
                *inp.shape[:-1], codes.shape[0])

        return forward

    def embedding_forward(module, fmt):
        codes, scales, group = quantized(module.weight, fmt, table=True)

        def forward(indices):
            flat = indices.detach().reshape(-1).numpy()
            out = np.empty((flat.size, codes.shape[1]), dtype=np.float32)
            for row_index, token in enumerate(flat):
                row = np.zeros(codes.shape[1], dtype=np.float32)
                for group_index in range(scales.shape[1]
                                         if scales.ndim == 2 else 1):
                    first = group_index * group
                    last = min(first + group, codes.shape[1])
                    scale = (scales[token, group_index]
                             if scales.ndim == 2 else scales[token])
                    row[first:last] = (
                        codes[token, first:last].astype(np.float32)
                        * np.float32(scale))
                out[row_index] = row
            return torch.from_numpy(out).reshape(
                *indices.shape, codes.shape[1])

        return forward

    def conv1d_forward(module):
        """Mirror the backend's dynamic-i8 depthwise Conv1d arithmetic."""
        if (module.groups != module.in_channels
                or module.out_channels != module.in_channels):
            raise ValueError(
                "architecture reference currently expects depthwise Conv1d")

        def forward(inp):
            from executorch.backends.esp32s3.operator_support.conv import ConvGeom
            from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax

            kernel = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]
            dilation = module.dilation[0]
            length = inp.shape[-1]
            output_length = (
                (length + 2 * padding - dilation * (kernel - 1) - 1)
                // stride + 1)
            geometry = ConvGeom(
                h_in=1, w_in=length, c_in=1,
                h_out=1, w_out=output_length,
                c_out=module.out_channels,
                kh=1, kw=kernel,
                stride_h=1, stride_w=stride,
                pad_h=0, pad_w=padding,
                dil_h=1, dil_w=dilation,
                groups=module.groups,
                c_out_total=module.out_channels,
            )
            result = depthwise_iq_absmax(
                inp, module.weight, module.bias, geometry, "i8",
                group_weights=True)
            return result.reshape(
                inp.shape[0], module.out_channels, output_length)

        return forward

    def rmsnorm_forward(module):
        from executorch.backends.esp32s3.test.reference import rmsnorm_f32_s3_reference

        weight = module.weight.detach().numpy().astype(np.float32)
        eps = module.eps

        def forward(inp):
            return torch.from_numpy(rmsnorm_f32_s3_reference(
                inp.detach().numpy(), weight, eps))

        return forward

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.forward = linear_forward(module)
        elif isinstance(module, nn.Embedding):
            if isinstance(embedding_format, set):
                fmt = "i4" if name in embedding_format else "i8"
            else:
                fmt = embedding_format
            module.forward = embedding_forward(module, fmt)
        elif isinstance(module, nn.Conv1d):
            module.forward = conv1d_forward(module)
        elif module.__class__.__name__ == "RMSNorm" and all(
                hasattr(module, attribute) for attribute in ("weight", "eps")):
            module.forward = rmsnorm_forward(module)
    return model
