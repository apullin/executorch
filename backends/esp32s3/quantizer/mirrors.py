# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bit-exact references for every quantisation scheme the backend must reproduce.

The runtime must match THESE bit for bit. So every function is a transcription
of what its source of truth computes, rounding step by rounding step, not of
the mathematics it approximates:

  activations  the choosers and quantisers A0-A7 (ActScheme, below) --
               torch.ao's quantized_decomposed ops, torchao's quant_primitives,
               XNNPACK's per-token chooser, the backend's own absmax rows --
               plus uint8 model IO;
  weights      W-sym, W-zp (integer zero point), W-off (tinygemm/HQQ and GGUF
               float offsets) and the unsigned-code normalisation;
  matmul       the integer dot with zero-point corrections,
               acc = raw - zw*X_g - zx*S'_ng, per row and per group (ragged
               last group), wrapping modulo 2^32 as the PIE kernels' 32-bit
               readback does, then the float epilogue in the device's order;
  conv         the im2col gather with padding written as zx, for dense, grouped,
               depthwise and transposed convolutions;
  embeddings   ExecuTorch's 2/4/8-bit tables (and its nibble order) and W-off;
  K cache      the asymmetric writer (XNNPACK's chooser, quantised in float64)
               and the score formula the K-cache op computes;
  table op     espressif::quant_lut (fold_quant_lut_pass.py): the input
               quantiser, then a lookup in a table built at export;
  fused norm   espressif::qnorm (fuse_qnorm_pass.py): a decomposed quantised
               RMSNorm or LayerNorm, every pair applied, tables of one code,
               reductions in index order;
  float        the float-weight loops (esp32_float.h): fp16/bf16 widened
               exactly, then opcode 1's sum -- the bias, then x * w in k
               order -- for a matmul and for the fp32 convolution's gather.

It also transcribes ExecuTorch's portable choose_qparams kernel, which agrees
with none of the PyTorch choosers, so the disagreement is on record.

numpy only at import time. Helpers that already exist elsewhere -- the
backend's round_half_away, operator_support.conv's geometry -- are
imported where they are used rather than copied. test_mirrors.py holds
every function here to torch's own ops on adversarial inputs.

Conventions. fl() means one IEEE float32 rounding. numpy float32 arithmetic is
exactly that (numpy 2 keeps Python scalars weak, so f32_array * 0.5 stays
float32), and every step that could promote to float64 is cast explicitly.

Non-finite inputs follow ExecuTorch 1.6's quantize rule (op_quantize.cpp,
#22811) wherever the device's codes have a portable reference: a NaN's code is
the zero point, clamped to the range, and an infinity's the rail
(nan_to_zero, the device's quant_rules.h). The choosers skip a NaN, as the
device's min/max comparisons do, and an infinity gives an infinite scale; a
NaN zero point from one rounds as 0. The backend's own rows (A0) and the float
loops keep NaN out of contract, as the device does.
"""

from dataclasses import dataclass, replace

import numpy as np

F32 = np.float32
F64 = np.float64

# The eps floors of the stock dynamic choosers.
EPS_XNNPACK = 2.0 ** -12    # xnnpack_quantizer.py's dynamic config
EPS_ARM = 2.0 ** -16        # arm_quantizer.py's dynamic config
FLT_EPSILON = 2.0 ** -23    # PT2E's default; XNNPACK's per-token chooser
FLT_MIN = 2.0 ** -126       # torchao's default, finfo(float32).smallest_normal

INT32_MAX = 2 ** 31 - 1
# The PIE accumulator ACCX's signed 40 bits: the exactness limit of a 16-bit
# activation's dot. The device's scalar int64
# dot does not need it; the runtime bounds 16-bit dots by it all the same.
ACCX_MAX = 2 ** 39 - 1
# max |xq - zx| the runtime assumes for every 16-bit code (they are
# symmetric, zero point 0, and int16 holds -32768).
WIDE_SPAN = 32768

# The fused int4 kernel's own per-group limit (esp32_backend.cpp
# kMaxFusedI4GroupSize), tighter than any int4 exactness bound.
FUSED_I4_MAX_GROUP = 1 << 16


def _f32(x):
    return np.asarray(x, dtype=F32)


# --- Rounding -----------------------------------------------------------------

def rne(v):
    """Round half to even in v's own precision: torch.round, C nearbyint.

    np.rint keeps float32 float32 and float64 float64, which matters: A4 rounds
    a float64 value, every other scheme a float32 one.
    """
    return np.rint(v)


def nan_to_zero(v):
    """The device's quant_scaled (runtime/quant_rules.h): a scaled value
    v = fl(x * inv) as a quantiser with a portable reference rounds it. NaN
    becomes 0, so its code is the zero point (ExecuTorch 1.6's quantize_val);
    every other value, the infinities included, is left for the quantiser's
    clamps, which put the infinities on the rails. Keeps v's float dtype."""
    v = np.asarray(v)
    return np.where(np.isnan(v), v.dtype.type(0), v)


def round_half_away(v):
    """The backend's own rounding (A0), from the backend itself."""
    from executorch.backends.esp32s3.serialization.weight_formats import (
        round_half_away as backend_round,
    )

    return backend_round(v)


def reciprocal(scale):
    """The float32 multiplier every quantiser applies: fl(1 / s).

    torch spells it three ways: quantize_per_tensor divides 1.0 by its float
    scale in float64 and the multiply casts the quotient to float32; the
    per-channel, per-group and torchao paths take Tensor.reciprocal in float32;
    the device divides 1.0f / s. For a float32 s all three are the correctly
    rounded float32 reciprocal: the float64 quotient of two float32 values rounds
    to float32 without double-rounding error (53 >= 2*24 + 2), and
    test_mirrors.py checks torch's float32 reciprocal on all 2^32 inputs.
    A scale that is NOT float32 (a hand-written 0.1 in a static graph) gets
    quantize_per_tensor's float64 quotient, which is what torch computes there.
    """
    return (F64(1.0) / np.asarray(scale, dtype=F64)).astype(F32)


# --- Activation schemes ---------------------------------------------------------

@dataclass(frozen=True)
class ActScheme:
    """One activation quantiser: the fields EXT_ACT_QUANT carries (its code
    width and sym_div follow from qmin and qmax).

    chooser      "absmax"     A0: amax / 127 per row, no eps
                 "symmetric"  A1, A7: max(-lo, hi) / ((qmax - qmin) / 2)
                 "affine"     A2, A3: (hi - lo) / (qmax - qmin), integer zp
                 "token_xnn"  A4: XNNPACK's per-token chooser
                 "static"     A5, A6: constants from the graph
    granularity  "row" (per token), "tensor", or "static"
    rounding     "half_away"         A0: half away from zero, clamp +-qmax
                 "rne_before_zp"     round, then add the zero point (float32)
                 "rne_after_zp_f64"  add the zero point, then round (float64)
    """

    name: str
    chooser: str
    granularity: str
    rounding: str
    qmin: int
    qmax: int
    eps: float = 0.0
    scale: float = 0.0
    zero_point: int = 0

    @property
    def affine(self):
        """Whether the zero point can be nonzero, which is what widens the
        integer operand from the code range to qmax - qmin."""
        return self.chooser in ("affine", "token_xnn") or self.zero_point != 0

    @property
    def span(self):
        """max |xq - zx| over every code this scheme can produce."""
        if self.affine:
            return self.qmax - self.qmin
        return max(-self.qmin, self.qmax)

    @property
    def code_bits(self):
        """EXT_ACT_QUANT's code width: 8 for an int8 or a uint8 range (uint8
        rides shifted onto int8 codes), 16 past it (A6)."""
        return code_bits(self.qmin, self.qmax)

    def static(self, scale, zero_point=0):
        """The same scheme with graph constants, as a static quantize carries."""
        return replace(self, chooser="static", granularity="static",
                       scale=scale, zero_point=int(zero_point))


A0 = ActScheme("A0", "absmax", "row", "half_away", -127, 127)
A1 = ActScheme("A1", "symmetric", "tensor", "rne_before_zp", -127, 127,
               eps=FLT_EPSILON)
A2_XNNPACK = ActScheme("A2/xnnpack", "affine", "tensor", "rne_before_zp",
                       -128, 127, eps=EPS_XNNPACK)
A2_ARM = ActScheme("A2/arm", "affine", "tensor", "rne_before_zp", -128, 127,
                   eps=EPS_ARM)
A3_TORCHAO = ActScheme("A3/torchao", "affine", "row", "rne_before_zp",
                       -128, 127, eps=FLT_MIN)
# Int8DynActInt4WeightQuantizer: eps FLT_EPSILON and a float32 zero point
# holding the same integer values.
A3_LEGACY = ActScheme("A3/legacy", "affine", "row", "rne_before_zp",
                      -128, 127, eps=FLT_EPSILON)
A4 = ActScheme("A4", "token_xnn", "row", "rne_after_zp_f64", -128, 127,
               eps=FLT_EPSILON)
# Static schemes carry their constants: A5.static(scale, zp), A6.static(scale).
A5 = ActScheme("A5", "static", "static", "rne_before_zp", -128, 127)
A5_UINT8 = ActScheme("A5/uint8", "static", "static", "rne_before_zp", 0, 255)
A6 = ActScheme("A6", "static", "static", "rne_before_zp", -32767, 32767)
A7_TORCHAO = ActScheme("A7/torchao", "symmetric", "row", "rne_before_zp",
                       -128, 127, eps=FLT_MIN)


def code_bits(qmin, qmax):
    """8 when [qmin, qmax] fits int8 codes, directly or shifted by -128 from
    uint8; 16 otherwise (static int16 activations, A6)."""
    if 0 <= qmin and qmax <= 255:
        return 8
    return 8 if -128 <= qmin and qmax <= 127 else 16


def _lo_hi(x, axis):
    """min(amin, 0) and max(amax, 0): every affine and symmetric chooser widens
    the observed range to contain 0 before doing anything else. A NaN is
    skipped, as the device's comparisons skip it (fmin and fmax from 0)."""
    lo = np.fmin.reduce(x, axis=axis, initial=F32(0)).astype(F32)
    hi = np.fmax.reduce(x, axis=axis, initial=F32(0)).astype(F32)
    return lo, hi


def symmetric_scale(amax, qmin, qmax, eps):
    """fl(amax / ((qmax - qmin) / 2)) floored at fl(eps).

    PT2E's choose_qparams_symmetric (utils.py determine_qparams) and torchao's
    SYMMETRIC chooser are this one expression: the divisor is a float32
    constant (127 for [-127, 127], 127.5 for [-128, 127]) and eps reaches the
    comparison as float32 (torch.Tensor([eps]), or clamp's cast).
    """
    scale = (_f32(amax) / F32((qmax - qmin) / 2)).astype(F32)
    return np.maximum(scale, F32(eps))


def choose_absmax(x, qmax=127):
    """A0, per row of x[..., K]: fl(amax / qmax), no eps, zero point 0.

    The device's quantize_row_t: a float32 division, so an all-zero row gets
    scale 0, which its quantiser maps to all-zero codes.
    """
    x = _f32(x)
    amax = np.fmax.reduce(np.abs(x), axis=-1, initial=F32(0))  # NaN skipped
    scale = (amax / F32(qmax)).astype(F32)
    return scale, np.zeros(scale.shape, dtype=np.int64)


def choose_symmetric(x, qmin, qmax, eps, axis=None):
    """A1 / A7: symmetric scale over `axis` (None = whole tensor), zero point 0.

    Signed ranges only. For an unsigned symmetric range PT2E puts the zero point
    at 128 and torchao at the range's midpoint, which disagree for any range but
    [0, 255]; no stock activation recipe asks for one, so refuse it by name.

    torchao's SYMMETRIC chooser takes the midpoint on signed ranges too,
    int((qmax + qmin + 1) / 2) (quant_primitives.py:1574): 0 for every stock
    range ([-128, 127], [-127, 127]), but -13 for [-128, 100]. This mirror's
    zero point is 0 (PT2E's for any signed range), so the exporter refuses a
    torchao symmetric range whose midpoint is not 0 (strip_dynamic_act_qdq,
    [act-symmetric-midpoint]) rather than run it here.
    """
    if qmin >= 0:
        raise ValueError(f"unsigned symmetric activations [{qmin}, {qmax}]: "
                         f"PT2E and torchao disagree on the zero point")
    lo, hi = _lo_hi(_f32(x), axis)
    scale = symmetric_scale(np.maximum(-lo, hi), qmin, qmax, eps)
    return scale, np.zeros(np.shape(scale), dtype=np.int64)


def choose_affine(x, qmin, qmax, eps, axis=None):
    """A2 / A3: the affine min/max chooser over `axis` (None = whole tensor).

    PT2E's choose_qparams.tensor (utils.py determine_qparams) and torchao's
    ASYMMETRIC chooser, operation for operation:

        scale = max(fl(fl(hi - lo) / (qmax - qmin)), fl(eps))
        zp    = clamp(qmin - rne(fl(lo / scale)), qmin, qmax)

    hi - lo is its own float32 rounding before the divide (two operations,
    not one), and lo / scale is a true division. An infinite range has an
    infinite scale, and lo / scale is 0 or (lo = -inf) NaN, which rounds as
    0: zp = qmin.
    """
    lo, hi = _lo_hi(_f32(x), axis)
    span = (hi - lo).astype(F32)
    scale = np.maximum((span / F32(qmax - qmin)).astype(F32), F32(eps))
    zp = np.clip(F32(qmin) - rne(nan_to_zero((lo / scale).astype(F32))), qmin,
                 qmax)
    return scale, zp.astype(np.int64)


def choose_affine_token_xnn(x, qmin=-128, qmax=127, eps=FLT_EPSILON):
    """A4, per row of x[..., K]: quantized_decomposed's
    _choose_qparams_per_token_asymmetric_impl (after XNNPACK's quantization.h).

        scale = max(fl(fl(hi - lo) / 255), FLT_EPSILON)
        dmin, dmax = fl(lo / scale), fl(hi / scale)
        zp = fl(qmin - dmin)  if fl(fl(qmin + dmin) + fl(qmax + dmax)) > 0
             fl(qmax - dmax)  otherwise
        zp = rne(clamp(zp, qmin, qmax))

    All float32, including the sum the test compares with 0; the clamp comes
    BEFORE the rounding, the reverse of choose_affine. A row holding +inf has
    scale inf and dmax = inf / inf: its zp is NaN, in torch too, and is 0
    here, as the device's writer takes it (kv_affine.cpp).
    """
    lo, hi = _lo_hi(_f32(x), -1)
    span = (hi - lo).astype(F32)
    scale = np.maximum((span / F32(qmax - qmin)).astype(F32), F32(eps))
    dmin = (lo / scale).astype(F32)
    dmax = (hi / scale).astype(F32)
    error_min = (F32(qmin) + dmin).astype(F32)
    error_max = (F32(qmax) + dmax).astype(F32)
    from_min = (F32(qmin) - dmin).astype(F32)
    from_max = (F32(qmax) - dmax).astype(F32)
    zp = np.where((error_min + error_max).astype(F32) > 0, from_min, from_max)
    return scale, rne(np.clip(nan_to_zero(zp), qmin, qmax)).astype(np.int64)


# FLT_MIN: A0's smallest scale with a reciprocal (quant_rules.h kA0MinScale).
A0_MIN_SCALE = F32(2.0 ** -126)


def a0_inverse(scale):
    """A0's inverse scale, the device's a0_inverse_scale (quant_rules.h):
    fl(1 / s) for a normal scale, 0 for a zero or subnormal one, so that row's
    codes are all 0 (an absmax below 127 * 2^-126). Without the cut a finite
    row like [FLT_MIN, 0] had fl(1 / s) = inf and codes from inf and NaN."""
    s = _f32(scale)
    normal = s >= A0_MIN_SCALE
    with np.errstate(divide="ignore", over="ignore"):
        inv = F32(1.0) / np.where(normal, s, F32(1.0))
    return np.where(normal, inv, F32(0.0)).astype(F32)


def quantize_half_away(x, scale, qmax=127):
    """A0's quantiser, the device's quantize_row_t: fl(x * a0_inverse(s))
    rounded half away from zero, clamped to +-qmax. A zero or subnormal scale
    gives zero codes."""
    v = (_f32(x) * a0_inverse(scale)).astype(F32)
    return np.clip(round_half_away(v), -qmax, qmax).astype(np.int64)


def quantize_before_zp(x, scale, zp, qmin, qmax):
    """Round, then add the zero point, in float32:

        clamp(rne(fl(x * fl(1/s))) + zp, qmin, qmax)

    quantize_per_tensor(.tensor) and quantize_per_channel (_decomposed.py),
    torchao's quantize_affine, and ExecuTorch's portable quantize_val. The add
    is exact (an integral float32 plus a small integer), or saturates. NaN
    takes the zero point (nan_to_zero), as ExecuTorch 1.6's kernel gives it;
    torch's own code for it is platform-defined.
    """
    v = nan_to_zero((_f32(x) * reciprocal(scale)).astype(F32))
    q = (rne(v) + np.asarray(zp, dtype=F32)).astype(F32)
    return np.clip(q, qmin, qmax).astype(np.int64)


def quantize_after_zp_f64(x, scale, zp, qmin, qmax):
    """Add the zero point, then round, in float64 -- quantize_per_token
    (_decomposed.py) with the float64 scales its chooser returns:

        clamp(rne(fl64(fl64(x * fl64(1/s)) + zp)), qmin, qmax)

    The multiply promotes the float32 input to float64 (scales is a float64
    tensor), so both the product and the zero-point add round in float64; with
    an odd zero point a tie lands on the other side of choose_affine's. A NaN
    product takes the zero point (nan_to_zero).
    """
    inv = F64(1.0) / np.asarray(scale, dtype=F64)
    v = nan_to_zero(np.asarray(x, dtype=F32).astype(F64) * inv)
    v = v + np.asarray(zp, dtype=F64)
    return np.clip(rne(v), qmin, qmax).astype(np.int64)


def quantize_activation(x, scheme):
    """x[M, K] float32 -> (codes [M, K] int64, scale [M] float32, zp [M] int64).

    Per-tensor and static schemes broadcast one scale and zero point to every
    row; the chooser of a per-tensor scheme sees the whole tensor.
    """
    x = _f32(x)
    rows = x.shape[0]
    if scheme.chooser == "absmax":
        scale, zp = choose_absmax(x, scheme.qmax)
    elif scheme.chooser == "static":
        scale = np.full(rows, scheme.scale, dtype=F32)
        zp = np.full(rows, scheme.zero_point, dtype=np.int64)
    else:
        axis = -1 if scheme.granularity == "row" else None
        if scheme.chooser == "symmetric":
            scale, zp = choose_symmetric(x, scheme.qmin, scheme.qmax,
                                         scheme.eps, axis)
        elif scheme.chooser == "affine":
            scale, zp = choose_affine(x, scheme.qmin, scheme.qmax, scheme.eps,
                                      axis)
        elif scheme.chooser == "token_xnn":
            if axis is None:
                raise ValueError("the XNNPACK-flavour chooser is per token only")
            scale, zp = choose_affine_token_xnn(x, scheme.qmin, scheme.qmax,
                                                scheme.eps)
        else:
            raise ValueError(f"unknown chooser {scheme.chooser!r}")
        scale = np.broadcast_to(scale, (rows,)).astype(F32)
        zp = np.broadcast_to(zp, (rows,)).astype(np.int64)
    if scheme.chooser == "static" and scheme.rounding == "rne_before_zp":
        # The graph's float scale goes through quantize_per_tensor unchanged:
        # reciprocal() takes its float64 quotient, which for a float32 scale
        # is fl(1/s) and for any other double is what torch multiplies by.
        codes = quantize_before_zp(x, F64(scheme.scale), scheme.zero_point,
                                   scheme.qmin, scheme.qmax)
    elif scheme.rounding == "half_away":
        codes = quantize_half_away(x, scale[:, None], scheme.qmax)
    elif scheme.rounding == "rne_before_zp":
        codes = quantize_before_zp(x, scale[:, None], zp[:, None],
                                   scheme.qmin, scheme.qmax)
    elif scheme.rounding == "rne_after_zp_f64":
        codes = quantize_after_zp_f64(x, scale[:, None], zp[:, None],
                                      scheme.qmin, scheme.qmax)
    else:
        raise ValueError(f"unknown rounding {scheme.rounding!r}")
    return codes, scale, zp


def dequantize_per_tensor(q, scale, zp):
    """dequantize_per_tensor: fl(fl(q - zp) * fl(s)) -- the float scale is cast
    to float32 before the multiply, the subtraction exact for codes."""
    return ((np.asarray(q).astype(F32) - F32(zp)).astype(F32)
            * F32(scale)).astype(F32)


def fake_quant_output(y, scale, zp, qmin, qmax):
    """Stage A's output side: quantise y with the model's (s, zp) and write the
    dequantised value back, so a downstream quantize sees the identical code."""
    return dequantize_per_tensor(quantize_before_zp(y, F64(scale), zp, qmin,
                                                    qmax), scale, zp)


def uint8_to_int8(codes, zp):
    """uint8 model IO onto the int8 kernels: codes and zero point shift by -128.

    Exact: clamp(r + zp, 0, 255) - 128 == clamp(r + (zp - 128), -128, 127) for
    the rounded product r, and (q - zp) is unchanged, so every dequantised value
    and every corrected dot is the same.
    """
    return (np.asarray(codes, dtype=np.int64) - 128,
            np.asarray(zp, dtype=np.int64) - 128)


# --- Weight dequantisation ----------------------------------------------------

def expand_groups(params, k, gsize=None):
    """Per-group parameters [N, G] as per-element [N, k]: FIXED groups of gsize
    along the row with a short final one, the quantiser's own boundaries (as
    reference.embedding_iq_absmax lays them out). One column means one group."""
    params = np.asarray(params)
    if params.ndim == 1:
        params = params[:, None]
    if params.shape[1] == 1:
        return np.repeat(params, k, axis=1)
    return params[:, np.arange(k) // gsize]


def normalize_unsigned(q, zw, bits):
    """Unsigned codes onto the signed kernel: c = q - 2^(bits-1) and
    zw' = zw - 2^(bits-1). Integer subtraction, so c - zw' == q - zw exactly."""
    base = 1 << (bits - 1)
    return (np.asarray(q, dtype=np.int64) - base,
            np.asarray(zw, dtype=np.int64) - base)


def dequant_w_sym(codes, scales, gsize=None):
    """W-sym: fl(c * s)."""
    codes = np.asarray(codes)
    s = expand_groups(_f32(scales), codes.shape[-1], gsize)
    return (codes.astype(F32) * s).astype(F32)


def dequant_w_zp(codes, zero_points, scales, gsize=None):
    """W-zp: fl(fl(c - zw) * s).

    torchao's dequantize_affine subtracts in float32 before scaling, and so does
    ExecuTorch's embedding kernel; PT2E's dequantize_per_channel(_group)
    subtracts in integers. For an integer zero point the subtraction is exact
    and all three agree; a FRACTIONAL zero point (a float zero-point tensor)
    keeps this two-rounding form, which is not W-off's fl(fl(c * s) + m).
    """
    codes = np.asarray(codes)
    k = codes.shape[-1]
    z = expand_groups(np.asarray(zero_points, dtype=F32), k, gsize)
    s = expand_groups(_f32(scales), k, gsize)
    return ((codes.astype(F32) - z).astype(F32) * s).astype(F32)


def dequant_w_off_tinygemm(q, scales, offsets, gsize=None, qmin=0, qmax=15):
    """W-off, tinygemm/HQQ (torchao _dequantize_affine_tinygemm_no_dtype_check):

        fl(fl(fl(q - mid) * s) + m),   mid = (qmax + qmin + 1) / 2

    q are the unsigned codes; q - mid is exact. HQQ's own format converts to
    this one (_convert_to_affinequantized_format), so it is this formula too.
    """
    q = np.asarray(q)
    k = q.shape[-1]
    mid = F32((qmax + qmin + 1) / 2)
    v = (q.astype(F32) - mid).astype(F32)
    v = (v * expand_groups(_f32(scales), k, gsize)).astype(F32)
    return (v + expand_groups(_f32(offsets), k, gsize)).astype(F32)


def gguf_block_params(super_scale, quant_scale, super_min, quant_min):
    """d = fl(ss * qs) and mn = fl(sm * qm), as torchao's _dequantize_gguf
    recovers each block's scale and min from their 6-bit quantisation."""
    d = (_f32(super_scale) * _f32(quant_scale)).astype(F32)
    mn = (_f32(super_min) * _f32(quant_min)).astype(F32)
    return d, mn


def dequant_w_off_gguf(q, d, mn, gsize=None):
    """W-off, GGUF (torchao _dequantize_gguf): fl(fl(q * d) + mn), unsigned q,
    per-block d and mn from gguf_block_params."""
    q = np.asarray(q)
    k = q.shape[-1]
    v = (q.astype(F32) * expand_groups(_f32(d), k, gsize)).astype(F32)
    return (v + expand_groups(_f32(mn), k, gsize)).astype(F32)


def dequantize_int32_bias(bias, scales):
    """An int32 bias with its derived scale, as PT2E's own dequant evaluates it:
    fl(fl(b) * s). The int-to-float conversion rounds once |b| > 2^24, so the
    fold is two roundings, not one."""
    b = np.asarray(bias, dtype=np.int64).astype(F32)
    return (b * _f32(scales)).astype(F32)


# --- Integer matmul with zero-point corrections ----------------------------------

class ExactnessRefused(ValueError):
    """A group long enough that its exact dot might not fit int32. The runtime
    refuses it at load and the AOT side at export; so does the mirror."""


def wrap_i32(v):
    """The low 32 bits as a signed value: what rur.accx_0 reads back, and what
    uint32 corrections leave once reinterpreted."""
    v = np.asarray(v, dtype=np.int64)
    return ((v + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def dot_length_bound(act_span, weight_span, limit=INT32_MAX):
    """Longest group (a multiple of 32) whose exact dot always fits int32 (or
    `limit`) when |xq - zx| <= act_span and |c - zw| <= weight_span:

        127 x 128  132,096   the symmetric rows (kMaxDotLen)
        128 x 128  131,040   symmetric codes reaching -128
        255 x 128   65,792   affine activations x symmetric int8 weights
        255 x 255   33,024   affine x affine int8
        255 x 15   561,408   affine x int4 (the fused kernel stops at 65,536)
      32768 x 128  131,040   16-bit activations x int8 weights, against the
                             40-bit ACCX (limit=ACCX_MAX)
    """
    n = limit // (act_span * weight_span)
    return n - n % 32


def act_dot_bound(act, bits, has_zero_point):
    """The runtime's dot-bound rule for an op with this activation scheme:
    int32 for 8-bit codes, the 40-bit ACCX at WIDE_SPAN for 16-bit ones."""
    if act.code_bits == 16:
        return dot_length_bound(WIDE_SPAN, weight_span(bits, has_zero_point),
                                ACCX_MAX)
    return dot_length_bound(act.span, weight_span(bits, has_zero_point))


def weight_span(bits, has_zero_point):
    """max |c - zw| for stored weight codes of `bits` (int8 storage can hold
    -128 even for symmetric weights, which is why the symmetric bound uses
    128)."""
    if bits == 4:
        return 15 if has_zero_point else 8
    if bits == 8:
        return 255 if has_zero_point else 128
    raise ValueError(f"no int32 bound for {bits}-bit weights; they accumulate "
                     f"in int64")


def group_spans(k, gsize=None):
    """[(a, b)] per group: fixed multiples of gsize, a short (ragged) last one."""
    if not gsize or gsize >= k:
        return [(0, k)]
    return [(a, min(a + gsize, k)) for a in range(0, k, gsize)]


def exact_dots(xq, zx, codes, zero_points=None, gsize=None):
    """sum over each group of (xq - zx)(c - zw), unwrapped: the true value the
    corrected kernel dot must equal. [M, N, G] int64."""
    xq = np.asarray(xq, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.int64)
    spans = group_spans(xq.shape[1], gsize)
    zw = _zero_points(zero_points, codes.shape[0], len(spans))
    xs = xq - np.asarray(zx, dtype=np.int64).reshape(-1, 1)
    return np.stack([xs[:, a:b] @ (codes[:, a:b] - zw[:, [g]]).T
                     for g, (a, b) in enumerate(spans)], axis=2)


def _zero_points(zero_points, n, groups):
    if zero_points is None:
        return np.zeros((n, groups), dtype=np.int64)
    zw = np.asarray(zero_points, dtype=np.int64).reshape(n, -1)
    return np.repeat(zw, groups, axis=1) if zw.shape[1] == 1 else zw


def weight_sums(codes, zero_points=None, gsize=None):
    """S'_ng = sum over group g of (c - zw), over the REAL elements only: the
    k_pad tail is outside the group's count. Stored at export. [N, G] int64."""
    codes = np.asarray(codes, dtype=np.int64)
    spans = group_spans(codes.shape[1], gsize)
    zw = _zero_points(zero_points, codes.shape[0], len(spans))
    return np.stack([(codes[:, a:b] - zw[:, [g]]).sum(axis=1)
                     for g, (a, b) in enumerate(spans)], axis=1)


def corrected_dots(xq, zx, codes, zero_points=None, gsize=None, *,
                   i4_fused=False, bound=None, wide=False):
    """Each group's dot the way the device computes it.

        acc = raw - zw * X_g - zx * S'_ng      (modulo 2^32)

    raw is the kernel's dot of the STORED codes -- int8 codes, or for the fused
    int4 kernel the nibbles u = c + 8, whose correction is then (8 + zw) * X_g
    -- read back as its low 32 bits; X_g = sum of xq over the group (the k_pad
    tail is 0, conv padding is zx); S'_ng from weight_sums. Everything wraps, so
    the result is exact exactly when the true dot fits int32, which `bound`
    (dot_length_bound) guarantees per group: a longer group raises
    ExactnessRefused, as the runtime must refuse it.

    wide: 16-bit activation codes (A6), which the device dots in int64 --
    nothing wraps, and acc is the exact dot.

    xq [M, K], zx [M], codes [N, K], zero_points [N, G] or [N] or None.
    Returns (acc [M, N, G] int64 holding int32 values -- the exact int64 dot
    when wide -- X [M, G], S' [N, G]).
    """
    wrap = (lambda v: np.asarray(v, dtype=np.int64)) if wide else wrap_i32
    xq = np.asarray(xq, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.int64)
    zx = np.asarray(zx, dtype=np.int64).reshape(-1)
    spans = group_spans(xq.shape[1], gsize)
    if bound is not None:
        longest = max(b - a for a, b in spans)
        if longest > bound:
            raise ExactnessRefused(f"group of {longest} exceeds the exactness "
                                   f"bound {bound}")
    if i4_fused and max(b - a for a, b in spans) > FUSED_I4_MAX_GROUP:
        raise ExactnessRefused(f"fused int4 groups stop at {FUSED_I4_MAX_GROUP}")
    zw = _zero_points(zero_points, codes.shape[0], len(spans))
    bias = 8 if i4_fused else 0
    stored = codes + bias
    raw = np.stack([wrap(xq[:, a:b] @ stored[:, a:b].T)
                    for a, b in spans], axis=2)
    X = np.stack([xq[:, a:b].sum(axis=1) for a, b in spans], axis=1)
    S = weight_sums(codes, zw, gsize)
    acc = raw - (zw + bias)[None] * X[:, None, :] - zx[:, None, None] * S[None]
    return wrap(acc), X, S


def offset_terms(X, zx, sx, spans):
    """T_g = fl(fl(X'_g) * sx) with X'_g = X_g - n_g * zx: the activation side
    of a float offset's contribution, sum over g of (xq - zx) * sx, computed
    once per row and group. n_g counts real elements only. [M, G] float32."""
    n = np.asarray([b - a for a, b in spans], dtype=np.int64)
    xprime = np.asarray(X, dtype=np.int64) - n[None] * np.asarray(
        zx, dtype=np.int64).reshape(-1, 1)
    return (xprime.astype(F32) * _f32(sx).reshape(-1, 1)).astype(F32)


def device_epilogue(acc, sx, scales, bias=None, offsets=None, T=None, *,
                    fused_madd=False):
    """The float side, in the device's order, for acc [M, N, G]:

        y = bias[n]                              (0 without one)
        for each group g, in order:
            y = fl(y + fl(fl(fl(acc) * sx) * s[n, g]))
            y = fl(y + fl(m[n, g] * T[m, g]))    with float offsets (W-off)

    Two roundings per term: the affine code is built -ffp-contract=off. The
    offset term follows its group's scale term, the order the runtime sums
    them in. fused_madd=True models the symmetric loops instead,
    y = fma(fl(fl(acc) * sx), s, y): the S3 build compiles each one's
    `y += (float)acc * sx * scale` -- addmm_q_row's three plain-i8 loops,
    addmm_q_row_range, the blocked plain-i8 range, addmm_q_selected_rows, the
    depthwise loops and both i16 loops, as the parity firmware's disassembly
    shows -- to float.s (__floatdisf in the i16 loops), mul.s and ONE madd.s,
    which QEMU rounds once (fma_f32).
    """
    acc = np.asarray(acc)
    M, N, G = acc.shape
    scales = expand_groups(_f32(scales).reshape(N, -1), G, 1)
    sx = _f32(sx).reshape(M, 1)
    y = np.zeros((M, N), dtype=F32) if bias is None else np.broadcast_to(
        _f32(bias), (M, N)).astype(F32)
    if offsets is not None:
        offsets = expand_groups(_f32(offsets).reshape(N, -1), G, 1)
        T = _f32(T).reshape(M, G)
    if fused_madd:
        from executorch.backends.esp32s3.quantizer.mirrors import fma_f32
    for g in range(G):
        term = (acc[:, :, g].astype(F32) * sx).astype(F32)
        s = scales[None, :, g]
        if fused_madd:
            y = fma_f32(term, s, y)
        else:
            y = (y + (term * s).astype(F32)).astype(F32)
        if offsets is not None:
            m_t = (offsets[None, :, g] * T[:, g, None]).astype(F32)
            y = (y + m_t).astype(F32)
    return y


@dataclass
class MatmulTrace:
    """Every intermediate of affine_matmul, for tests and goldens."""
    xq: np.ndarray
    sx: np.ndarray
    zx: np.ndarray
    acc: np.ndarray
    X: np.ndarray
    S: np.ndarray
    y: np.ndarray


def affine_matmul(x, codes, scales, bias=None, *, act, bits=8, gsize=None,
                  zero_points=None, offsets=None, i4_fused=False,
                  fused_madd=False, check_bound=True):
    """y = x @ W^T the device's way, for any activation scheme and W-sym / W-zp /
    W-off weights. x [M, K] float32; codes [N, K] the signed stored codes
    (normalize_unsigned first); scales [N] or [N, G]; zero_points zw [N] or
    [N, G]; offsets m [N] or [N, G] float32 (W-off). GGUF's fl(q*d)+mn is
    codes = q - 8 with zw = -8, scale d and offset mn; tinygemm's is codes =
    q - 8, zw = 0, offset m. 16-bit activation codes (A6) dot in int64, as
    the device's scalar a16 loop does: fl(acc) is then one rounding of an
    exact integer of up to 40 bits. Returns a MatmulTrace; .y is the output.
    """
    xq, sx, zx = quantize_activation(x, act)
    bound = None
    if check_bound:
        bound = act_dot_bound(act, bits, zero_points is not None)
    acc, X, S = corrected_dots(xq, zx, codes, zero_points, gsize,
                               i4_fused=i4_fused, bound=bound,
                               wide=act.code_bits == 16)
    T = None
    if offsets is not None:
        T = offset_terms(X, zx, sx, group_spans(np.shape(x)[1], gsize))
    y = device_epilogue(acc, sx, scales, bias, offsets, T,
                        fused_madd=fused_madd)
    return MatmulTrace(xq, sx, zx, acc, X, S, y)


# --- Convolution: im2col with padding written as zx ------------------------------

def as_cdhw(x):
    """A batch-1 conv input [1, C, L], [1, C, H, W] or [1, C, D, H, W] as
    [C, D, H, W], the one shape the gather indexes (operator_support.conv's
    convention: degenerate axes, never a second gather)."""
    x = np.asarray(x)
    if x.shape[0] != 1 or x.ndim not in (3, 4, 5):
        raise ValueError(f"expected a batch-1 conv input, got {x.shape}")
    x = x[0]
    if x.ndim == 2:
        return x[:, None, None, :]
    if x.ndim == 3:
        return x[:, None, :, :]
    return x


def im2col_codes(xq, geom, zx, k_pad=None, channel_offset=0):
    """One conv's patch rows as activation CODES: [m, k_pad] int64, from the
    input's codes xq [C, D, H, W] (as_cdhw).

    Column order (c, d, i, j), the gather's (operator_support.conv._src_index
    is the geometry, shared rather than re-derived). A tap outside the input,
    or off the lattice of a transposed conv's zero-stuffed input, reads a real
    0.0, whose code is zx -- so it is written as zx and contributes
    (zx - zx) * w = 0 to the corrected dot, and zx to X_g, which S' then
    cancels. The k_pad tail after K stays 0: outside every count.
    channel_offset selects a grouped slice's (or a depthwise channel's) input.
    """
    from executorch.backends.esp32s3.operator_support.conv import _src_index

    k = geom.k
    cols = np.zeros((geom.m, k_pad or k), dtype=np.int64)
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
                                if idd is None or ih is None or iw is None:
                                    cols[row, t] = zx
                                else:
                                    cols[row, t] = xq[channel_offset + c,
                                                      idd, ih, iw]
                                t += 1
    return cols


def affine_conv(x, codes_nk, scales, bias, geom, act, *, bits=8, gsize=None,
                zero_points=None, offsets=None, check_bound=True):
    """A dense, grouped-slice or depthwise convolution with a per-tensor or
    static activation scheme (A1, A2, A5, A6): quantise the whole input once,
    gather codes with zx padding, then the matmul's corrected dots and epilogue.

    x is the full input [1, C_total, ...]; codes_nk [N, K] this op's weight
    codes (a transposed conv's already flipped and transposed, as the backend
    packs it); returns [1, N, d_out, h_out, w_out] float32.
    """
    if act.granularity == "row":
        raise ValueError("a conv's activation scale is per tensor or static")
    x4 = as_cdhw(_f32(x))
    xq, sx, zx = quantize_activation(x4.reshape(1, -1), act)
    xq = xq.reshape(x4.shape)
    zx = int(zx[0])
    codes_nk = np.asarray(codes_nk, dtype=np.int64)
    n = geom.c_out
    bound = None
    if check_bound:
        bound = act_dot_bound(act, bits, zero_points is not None)
    spans = group_spans(geom.k, gsize)
    zxv = np.full(geom.m, zx, dtype=np.int64)
    sxv = np.full(geom.m, sx[0], dtype=F32)
    if geom.depthwise:
        # Every output channel reads its own input channel: one 1-channel
        # gather per channel, as the device's depthwise loop does.
        from executorch.backends.esp32s3.operator_support.conv import ConvGeom

        one = ConvGeom(**{**vars(geom), "c_in": 1, "c_out": 1, "groups": 1,
                          "group_idx": 0})
        patches = [([ch], im2col_codes(xq, one, zx, channel_offset=ch))
                   for ch in range(n)]
    else:
        patches = [(list(range(n)), im2col_codes(
            xq, geom, zx, channel_offset=geom.group_idx * geom.c_in))]
    out = np.zeros((n, geom.m), dtype=F32)
    for chans, cols in patches:
        zw = None if zero_points is None else _zero_points(
            zero_points, n, len(spans))[chans]
        acc, X, _ = corrected_dots(cols, zxv, codes_nk[chans], zw, gsize,
                                   bound=bound, wide=act.code_bits == 16)
        T = off = None
        if offsets is not None:
            off = _f32(offsets).reshape(n, -1)[chans]
            T = offset_terms(X, zxv, sxv, spans)
        b = None if bias is None else _f32(bias)[chans]
        out[chans] = device_epilogue(acc, sxv, _f32(scales).reshape(n, -1)[
            chans], b, off, T).T
    return out.reshape(1, n, geom.d_out, geom.h_out, geom.w_out)


# --- Float weights: fp16/bf16 widening and the fp32 loop (esp32_float.h) ------

def widen_f16_bits(bits):
    """IEEE binary16 bit patterns as the fp32 bit patterns (uint32) the
    runtime widens them to (esp32_float::f16_bits): every finite half exact,
    an infinity kept, a NaN's sign and payload shifted up 13 bits, a
    signalling NaN left signalling. Integer arithmetic, not a float cast: a
    hardware half-to-single conversion may quiet a signalling NaN."""
    h = np.asarray(bits, dtype=np.uint16).astype(np.uint32)
    sign = (h & 0x8000) << 16
    em = h & 0x7FFF
    normal = (em << 13) + ((127 - 15) << 23)
    special = 0x7F800000 | ((em & 0x3FF) << 13)
    # A subnormal is em * 2^-24 with em < 2^10: a power-of-two scaling of a
    # small integer, exact and normal in fp32.
    sub = (em.astype(F32) * F32(2.0 ** -24)).view(np.uint32)
    out = np.where(em >= 0x7C00, special,
                   np.where(em >= 0x0400, normal, np.where(em == 0, 0, sub)))
    return (out | sign).astype(np.uint32)


def widen_bf16_bits(bits):
    """bfloat16 bit patterns as fp32 bit patterns: the top half of fp32."""
    return np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16


def widen(bits, fmt):
    """A stored fp16/bf16 weight as the float32 values the loop reads."""
    wide = widen_f16_bits(bits) if fmt == "fp16" else widen_bf16_bits(bits)
    return wide.view(F32)


def float_dot_rows(x, w_nk, bias=None, *, fused=True):
    """y[r, j] = bias[j] (0 without one) + x[r, 0] * w[j, 0] + ... in k
    order: opcode 1's sum (reference.addmm_k_outer), which the fp32 convolution
    and the fp16/bf16 rungs form too, on the widened weight. Each step rounds
    once on the device, whose build fuses it into madd.s (fused=True), and
    twice on the host, built with -ffp-contract=off (fused=False)."""
    from executorch.backends.esp32s3.quantizer.mirrors import fma_f32

    x = _f32(x)
    w = _f32(w_nk)
    y = np.zeros((x.shape[0], w.shape[0]), dtype=F32)
    if bias is not None:
        y += _f32(bias)[None, :]
    for t in range(x.shape[1]):
        if fused:
            y = fma_f32(x[:, t:t + 1], w[None, :, t], y)
        else:
            y = (y + x[:, t:t + 1] * w[None, :, t]).astype(F32)
    return y


def im2col_float(x4, geom, channel_offset=0):
    """One conv's patch rows as float32 [m, k] from its input [C, D, H, W]
    (as_cdhw): im2col_codes' gather, a tap outside the input or off a
    transposed conv's lattice reading a real 0.0 (esp32_asym::
    gather_conv_row)."""
    from executorch.backends.esp32s3.operator_support.conv import _src_index

    cols = np.zeros((geom.m, geom.k), dtype=F32)
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
                                if not (idd is None or ih is None
                                        or iw is None):
                                    cols[row, t] = x4[channel_offset + c,
                                                      idd, ih, iw]
                                t += 1
    return cols


def float_conv(x, w_nk, bias, geom, *, fused=True):
    """The float-weight convolution (esp32_float::conv): a dense conv, a
    grouped conv's slice (geom.group_idx) or a depthwise conv, every output
    position's patch gathered with 0.0 padding and dotted with each output
    channel's row of w_nk [N, K] (a transposed conv's flipped and transposed,
    as the backend packs it), float_dot_rows' arithmetic. x is the full input
    [1, C_total, ...]; returns [1, N, d_out, h_out, w_out] float32."""
    x4 = as_cdhw(_f32(x))
    w = _f32(w_nk)
    n = geom.c_out
    b = None if bias is None else _f32(bias)
    out = np.zeros((n, geom.m), dtype=F32)
    if geom.depthwise:
        from executorch.backends.esp32s3.operator_support.conv import ConvGeom

        one = ConvGeom(**{**vars(geom), "c_in": 1, "c_out": 1, "groups": 1,
                          "group_idx": 0})
        for ch in range(n):
            cols = im2col_float(x4, one, channel_offset=ch)
            out[ch] = float_dot_rows(cols, w[ch:ch + 1],
                                     None if b is None else b[ch:ch + 1],
                                     fused=fused)[:, 0]
    else:
        cols = im2col_float(x4, geom,
                            channel_offset=geom.group_idx * geom.c_in)
        out[:] = float_dot_rows(cols, w, b, fused=fused).T
    return out.reshape(1, n, geom.d_out, geom.h_out, geom.w_out)


# --- Embeddings ---------------------------------------------------------------

def unpack_et_4bit(packed):
    """ExecuTorch's 4-bit table (embeddingxb.cpp weight_value): element 2i in
    the HIGH nibble, 2i+1 in the low one, each stored as code + 8."""
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (2 * p.shape[-1],), dtype=np.int64)
    out[..., 0::2] = (p >> 4).astype(np.int64) - 8
    out[..., 1::2] = (p & 0x0F).astype(np.int64) - 8
    return out


def unpack_et_2bit(packed):
    """ExecuTorch's 2-bit table: four codes per byte from the LOW bits up, each
    stored as code + 2."""
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (4 * p.shape[-1],), dtype=np.int64)
    for i in range(4):
        out[..., i::4] = ((p >> (2 * i)) & 3).astype(np.int64) - 2
    return out


def swap_nibbles(packed):
    """ExecuTorch's 4-bit layout <-> the backend's (pack_rows: element 2i in
    the LOW nibble); both bias by +8, so the swap is the whole conversion."""
    p = np.asarray(packed, dtype=np.uint8)
    return ((p << 4) | (p >> 4)).astype(np.uint8)


def unpack_backend_i4(packed):
    """The backend's i4 rows (pack_rows): element 2i low, 2i+1 high, + 8."""
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (2 * p.shape[-1],), dtype=np.int64)
    out[..., 0::2] = (p & 0x0F).astype(np.int64) - 8
    out[..., 1::2] = (p >> 4).astype(np.int64) - 8
    return out


def embedding_dequant(codes, scales, zero_points, indices):
    """quantized_decomposed.embedding_{byte,2bit,4bit} with float32 scales and
    output: gather rows, then fl(fl(c - zp) * s) per element over groups of
    D / G (ExecuTorch's embeddingxb.cpp and its Python reference agree).
    codes [V, D] signed; scales and zero_points [V] or [V, G]."""
    codes = np.asarray(codes, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)
    d = codes.shape[1]
    s = np.asarray(scales, dtype=F32).reshape(codes.shape[0], -1)
    gsize = d // s.shape[1]
    z = (np.zeros_like(s) if zero_points is None
         else np.asarray(zero_points, dtype=F32).reshape(s.shape))
    flat = idx.reshape(-1)
    out = dequant_w_zp(codes[flat], z[flat], s[flat], gsize)
    return out.reshape(idx.shape + (d,))


def embedding_w_off(q, scale, offset, indices, gsize=None, gguf=False):
    """A W-off table's rows: gather, then the reference's two roundings in
    order -- tinygemm's fl(fl((q - 8) * s) + m), or with gguf=True
    fl(fl(q * d) + mn) (scale = d, offset = mn from gguf_block_params).
    q [V, D] are the UNSIGNED codes; a runtime storing signed c = q - 8
    reconstructs q as c + code_base with code_base 8."""
    idx = np.asarray(indices, dtype=np.int64)
    q = np.asarray(q)
    flat = idx.reshape(-1)
    s = np.asarray(scale, dtype=F32).reshape(q.shape[0], -1)[flat]
    m = np.asarray(offset, dtype=F32).reshape(q.shape[0], -1)[flat]
    rows = (dequant_w_off_gguf(q[flat], s, m, gsize) if gguf
            else dequant_w_off_tinygemm(q[flat], s, m, gsize))
    return rows.reshape(idx.shape + (q.shape[1],))


# --- Asymmetric K cache --------------------------------------------------------

def kcache_write(k):
    """What ExecuTorch's QuantizedKVCache stores for keys (or values) k[..., D]:
    choose_qparams_per_token_asymmetric + quantize_per_token (A4, float64), the
    scale kept as float32 (exact: it is a float32 value) and the zero point as
    int8. Returns (codes int8 [..., D], scales float32 [...], zps int8 [...])."""
    k = _f32(k)
    scale, zp = choose_affine_token_xnn(k)
    codes = quantize_after_zp_f64(k, scale[..., None], zp[..., None], -128, 127)
    return codes.astype(np.int8), scale.astype(F32), zp.astype(np.int8)


def kcache_scores(q, codes, scales, zero_points):
    """q @ k^T against an asymmetric int8 key cache, the K-cache op's way.

    Queries stay the delegate's own symmetric A0 rows (kcache_i8's choice);
    with Q = sum(qq) per query row,

        score = fl(fl(fl(sum(qq * kq) - zk * Q) * sq) * sk)

    q [..., M, D]; codes [..., L, D]; scales and zero_points [..., L]. The
    integer side is exact while D * 127 * 255 fits int32 (D <= 66,304).
    Returns [..., M, L] float32.
    """
    sq, _ = choose_absmax(q)
    qq = quantize_half_away(q, sq[..., None])
    kq = np.asarray(codes, dtype=np.int64)
    zk = np.asarray(zero_points, dtype=np.int64)
    raw = wrap_i32(np.matmul(qq, np.swapaxes(kq, -1, -2)))
    acc = wrap_i32(raw - zk[..., None, :] * qq.sum(axis=-1)[..., :, None])
    term = (acc.astype(F32) * sq[..., :, None]).astype(F32)
    return (term * _f32(scales)[..., None, :]).astype(F32)


def vcache_dequant(codes, scales, zero_points):
    """dequantize_per_token on a cache row: fl((vq - zv) * sv), one rounding of
    the exact product, as its float64 multiply then cast computes it."""
    v = np.asarray(codes, dtype=np.int64) - np.asarray(
        zero_points, dtype=np.int64)[..., None]
    return (v.astype(F32) * _f32(scales)[..., None]).astype(F32)


# --- The table op (espressif::quant_lut) ---------------------------------------

LUT_RAIL = F32(512.0)


def quant_lut_codes(x, inv_scale, zero_point, quant_min, quant_max):
    """quant_lut's quantize step: q1's code for every x, the zero point for
    NaN (ExecuTorch 1.6's rule).

        v = fl(x * inv), NaN -> 0, clamped to [-512, 512] in float
        c = clamp(rne(v) + zp, qmin, qmax)

    The clamp changes no code: with |zp| <= 255 any |v| past 512 lands on a
    rail anyway, as quantize_before_zp's would."""
    v = nan_to_zero((_f32(x) * F32(inv_scale)).astype(F32))
    v = np.minimum(np.where(v > -LUT_RAIL, v, -LUT_RAIL), LUT_RAIL)
    q = (rne(v) + F32(zero_point)).astype(F32)
    return np.clip(q, quant_min, quant_max).astype(np.int64)


def quant_lut(x, table, inv_scale, zero_point, quant_min, quant_max):
    """espressif::quant_lut: table[code - qmin] for each element's code. The
    entries are indexed, not computed, so their bits (NaN payloads,
    denormals) come through unchanged, as the kernel's word copy does."""
    codes = quant_lut_codes(x, inv_scale, zero_point, quant_min, quant_max)
    return np.asarray(table, dtype=F32)[codes - quant_min]


# --- The int16 table op (espressif::lut16; lut_fit.py fits its tables) ------------

LUT16_SCHEMES = ("direct", "uniform_linear", "uniform_cubic", "nsau_linear",
                 "nsau_cubic")
# Flags on the scheme word (scheme = base | flags):
#   OFFSET (8)  NSAU only: each sign's half holds only the segments its codes
#               reach, from the segment of its first code in [lo, hi]
#               (lut16_nsau_layout), not from code 0 with both halves the
#               larger one's size
#   TAIL (16)   uniform only: a code past [lo, hi] is not clamped; it reads the
#               edge segment (the first or the last) at its own offset, so the
#               edge segment's line -- or cubic -- continues
# The valid ids: 0-4, 11, 12, 17, 18 (lut16_scheme_error names the rest).
LUT16_OFFSET = 8
LUT16_TAIL = 16
LUT16_BASE = 7
# The quantize step's rails: with int16 codes and zero points any |v| past
# 2^16 lands on a rail, and inside them 1.5 * 2^23 rounds half to even.
LUT16_RAIL = F32(65536.0)
LUT16_MAGIC = F32(12582912.0)
# Cubic coefficients and, with a tail, every value the edge cubic reaches
# past [lo, hi] stay below these (quant_lut16.cpp's validate): |y| < 2^22
# keeps the 1.5 * 2^23 rounding exact.
LUT16_COEF_MAX = 2.0 ** 19
LUT16_TAIL_MAX = 2.0 ** 22


def lut16_scheme_error(scheme):
    """None for a valid scheme word, else the runtime's refusal name."""
    base, flags = scheme & LUT16_BASE, scheme & ~LUT16_BASE
    if scheme < 0 or base > 4 or flags & ~(LUT16_OFFSET | LUT16_TAIL):
        return "scheme"
    if flags & LUT16_TAIL:
        return ("tail on a direct table" if base == 0 else
                "tail on an NSAU table" if base in (3, 4) else None)
    if flags & LUT16_OFFSET and base not in (3, 4):
        return "offset on a direct table" if base == 0 else "offset on a uniform table"
    return None


def lut16_nsau_k(a, m):
    """The NSAU segment of |code| a: (sh << m) + (a >> sh), sh = max(floor(
    log2(a | 1)) - m, 0)."""
    a = int(a)
    sh = max((a | 1).bit_length() - 1 - m, 0)
    return (sh << m) + (a >> sh)


def lut16_nsau_layout(scheme, lo, hi, m):
    """(positive bias, negative bias, entries): an NSAU table's code c, of
    segment k, reads entry k + (bias of c's sign) -- its row (cubic) or its
    first node (linear) -- and the table holds `entries` of them.

      plain   the positive half from segment 0, then the negative half when
              lo < 0, each lut16_nsau_half segments (+ 1 node, linear):
              biases 0 and half (+ 1)
      OFFSET  the positive half (when hi >= 0) holds segments p0 = k(max(lo,
              0)) .. k(hi), the negative half (when lo < 0) n0 = k(max(1,
              -hi)) .. k(-lo), each + 1 node when linear: biases -p0 and
              P - n0, P the positive half's entries"""
    base, linear = scheme & LUT16_BASE, (scheme & LUT16_BASE) == 3
    extra = 1 if linear else 0
    if not scheme & LUT16_OFFSET:
        half = lut16_nsau_half(lo, hi, m)
        return 0, half + extra, (half + extra) * (2 if lo < 0 else 1)
    pos = neg = 0
    p0 = n0 = 0
    if hi >= 0:
        p0 = lut16_nsau_k(max(lo, 0), m)
        pos = lut16_nsau_k(hi, m) - p0 + 1 + extra
    if lo < 0:
        n0 = lut16_nsau_k(max(1, -hi), m)
        neg = lut16_nsau_k(-lo, m) - n0 + 1 + extra
    assert base in (3, 4)
    return -p0, pos - n0, pos + neg


def lut16_quantize(x, inv_scale, zero_point, quant_min, quant_max):
    """lut16's quantize step, quant_lut's with int16 rails:
    v = fl(x * inv), NaN -> 0, clamped to [-2^16, 2^16],
    c = clamp(rne(v) + zp, qmin, qmax)."""
    v = nan_to_zero((_f32(x) * F32(inv_scale)).astype(F32))
    v = np.minimum(np.where(v > -LUT16_RAIL, v, -LUT16_RAIL), LUT16_RAIL)
    q = (rne(v) + F32(zero_point)).astype(F32)
    return np.clip(q, quant_min, quant_max).astype(np.int64)


def lut16_nsau_half(lo, hi, m):
    """Segments in one sign's half of an NSAU table over [lo, hi]: one past
    the segment holding max(|lo|, |hi|)."""
    a = max(abs(int(lo)), abs(int(hi)), 1)
    sh = max(a.bit_length() - 1 - m, 0)
    return (sh << m) + (a >> sh) + 1


def lut16_segments(lo, hi, F):
    """Segments of a uniform table over [lo, hi]: ceil((hi - lo + 1) / 2^F)."""
    return ((hi - lo) >> F) + 1


def lut16_index(c, scheme, lo, hi, shift):
    """(j, t, F) of each code: its table entry (the row of a cubic, the first
    node of a linear segment), its offset into the segment, and the
    segment's log2 width. The code is clamped to [lo, hi] first, except
    with TAIL, where it is only clamped to int16.

      uniform (1, 2)  u = c - lo; j = u >> F; t = u & (2^F - 1)
        + TAIL        j = clamp(u >> F, 0, K - 1); t = u - (j << F) (past
                      the ends, t < 0 or t >= 2^F: the edge segment goes on)
      NSAU (3, 4)     a = |c|; sh = max(floor(log2(a | 1)) - m, 0);
                      q = a >> sh; k = (sh << m) + q; t = a - (q << sh);
                      j = k + the bias of c's sign (lut16_nsau_layout)"""
    base = scheme & LUT16_BASE
    c = np.asarray(c, dtype=np.int64)
    if base in (1, 2):
        if scheme & LUT16_TAIL:
            u = np.clip(c, -32768, 32767) - lo
            j = np.clip(u >> shift, 0, lut16_segments(lo, hi, shift) - 1)
            return j, u - (j << shift), np.full(u.shape, shift, np.int64)
        u = np.clip(c, lo, hi) - lo
        return u >> shift, u & ((1 << shift) - 1), np.full(u.shape, shift, np.int64)
    c = np.clip(c, lo, hi)
    a = np.abs(c)
    e = np.frexp((a | 1).astype(F64))[1].astype(np.int64) - 1
    sh = np.maximum(e - shift, 0)
    q = a >> sh
    k = (sh << shift) + q
    t = a - (q << sh)
    pos, neg, _ = lut16_nsau_layout(scheme, lo, hi, shift)
    return k + np.where(c < 0, neg, pos), t, sh


def lut16_codes(c, scheme, table, lo, hi, shift, G, out_qmin, out_qmax):
    """lut16's table step: output codes for int codes c, at
    lut16_index's entry j and offset t.

      direct (0)          y = T[clamp(c, lo, hi) - lo]
      uniform linear (1)  y = ((V[j] << F) + (V[j+1] - V[j]) * t + 2^(F+G-1))
                              >> (F + G)
      NSAU linear (3)     y = (V[j] + (((V[j+1] - V[j]) * t) >> F) + 2^(G-1))
                              >> G
      cubic (2, 4)        tf = fl(t) * 2^-F (exact); y = fl(c0 + fl(tf *
                          fl(c1 + fl(tf * fl(c2 + fl(tf * c3)))))); y =
                          (y + 1.5 * 2^23) - 1.5 * 2^23, round half to even

    then clamp(y, out_qmin, out_qmax). Shifts are floors on two's complement;
    lut16_linear_safe bounds every intermediate to int32 (with TAIL, t too:
    one past the int16 codes at most)."""
    base = scheme & LUT16_BASE
    if base == 0:
        cc = np.clip(np.asarray(c, dtype=np.int64), lo, hi)
        return np.clip(np.asarray(table).astype(np.int64)[cc - lo], out_qmin, out_qmax)
    j, t, F = lut16_index(c, scheme, lo, hi, shift)
    if base in (1, 3):
        V = np.asarray(table).astype(np.int64)
        v0, v1 = V[j], V[j + 1]
        if base == 1:
            sh = shift + G
            rnd = (1 << (sh - 1)) if sh > 0 else 0
            y = ((v0 << shift) + (v1 - v0) * t + rnd) >> sh
        else:
            rnd = (1 << (G - 1)) if G > 0 else 0
            y = (v0 + (((v1 - v0) * t) >> F) + rnd) >> G
        return np.clip(y, out_qmin, out_qmax)
    C = np.asarray(table, dtype=F32).reshape(-1, 4)
    tf = (t.astype(F32) * np.exp2(-F).astype(F32)).astype(F32)
    c0, c1, c2, c3 = C[j, 0], C[j, 1], C[j, 2], C[j, 3]
    y = (c2 + (tf * c3).astype(F32)).astype(F32)
    y = (c1 + (tf * y).astype(F32)).astype(F32)
    y = (c0 + (tf * y).astype(F32)).astype(F32)
    r = ((y + LUT16_MAGIC).astype(F32) - LUT16_MAGIC).astype(F32)
    return np.clip(r.astype(np.int64), out_qmin, out_qmax)


def lut16_entries(scheme, lo, hi, shift):
    """The table's first dimension: entries (direct), nodes (linear) or rows
    of four floats (cubic)."""
    base = scheme & LUT16_BASE
    if base == 0:
        return hi - lo + 1
    if base in (1, 2):
        return lut16_segments(lo, hi, shift) + (1 if base == 1 else 0)
    return lut16_nsau_layout(scheme, lo, hi, shift)[2]


def lut16_tail_reach(lo, hi, F):
    """With TAIL: the largest |t| the first segment reaches below lo (codes
    down to -32768) and the largest t the last one reaches (codes up to
    32767), each at least the segment's own 2^F - 1."""
    K = lut16_segments(lo, hi, F)
    width = (1 << F) - 1
    return max(lo + 32768, 0), max(32767 - (lo + ((K - 1) << F)), width)


def _lut16_segment_shifts(scheme, lo, hi, shift):
    """(first entry, log2 width) of every segment an NSAU table holds."""
    linear = scheme & LUT16_BASE == 3
    pos, neg, _ = lut16_nsau_layout(scheme, lo, hi, shift)
    out = []
    halves = []
    if scheme & LUT16_OFFSET:
        if hi >= 0:
            halves.append((pos, lut16_nsau_k(max(lo, 0), shift), lut16_nsau_k(hi, shift)))
        if lo < 0:
            halves.append((neg, lut16_nsau_k(max(1, -hi), shift),
                           lut16_nsau_k(-lo, shift)))
    else:
        half = lut16_nsau_half(lo, hi, shift)
        halves.append((0, 0, half - 1))
        if lo < 0:
            halves.append((half + (1 if linear else 0), 0, half - 1))
    for bias, k0, k1 in halves:
        for k in range(k0, k1 + 1):
            out.append((bias + k, max((k >> shift) - 1, 0)))
    return out


def lut16_linear_safe(scheme, table, lo, hi, shift, G):
    """Whether a linear table's interpolation stays inside int32 for every
    segment the index reaches (quant_lut16.cpp refuses the table otherwise):
    |V| <= 2^30 for every node, and per segment
      uniform  |V0| * 2^F + |dV| * T + 2^(F+G-1) < 2^31, T = 2^F - 1, or the
               edge segments' reach with TAIL (lut16_tail_reach)
      NSAU     |dV| * (2^sh - 1) < 2^31 and |V0| + |dV| + 2^(G-1) < 2^31."""
    V = [int(v) for v in np.asarray(table).reshape(-1)]
    limit = 2 ** 31
    if any(abs(v) > 2 ** 30 for v in V):
        return False
    if scheme & LUT16_BASE == 1:
        F = shift
        rnd = (1 << (F + G - 1)) if F + G > 0 else 0
        reach = [(1 << F) - 1] * (len(V) - 1)
        if scheme & LUT16_TAIL:
            below, above = lut16_tail_reach(lo, hi, F)
            reach[0] = max(reach[0], below)
            reach[-1] = max(reach[-1], above)
        return all(abs(V[k]) * (1 << F) + abs(V[k + 1] - V[k]) * reach[k] + rnd
                   < limit for k in range(len(V) - 1))
    rnd = (1 << (G - 1)) if G > 0 else 0
    for at, sh in _lut16_segment_shifts(scheme, lo, hi, shift):
        v0, dv = V[at], V[at + 1] - V[at]
        if abs(dv) * ((1 << sh) - 1) >= limit or abs(v0) + abs(dv) + rnd >= limit:
            return False
    return True


def lut16_cubic_safe(scheme, table, lo, hi, shift):
    """Whether a cubic table holds finite coefficients within 2^19 and, with
    TAIL, its edge cubics stay below 2^22 everywhere past [lo, hi]:
    |c0| + |c1| T + |c2| T^2 + |c3| T^3 < 2^22 at the reach T (in segment
    widths) of each edge (quant_lut16.cpp's validate)."""
    C = np.asarray(table, dtype=np.float64).reshape(-1, 4)
    if not (np.isfinite(C).all() and np.abs(C).max(initial=0.0) <= LUT16_COEF_MAX):
        return False
    if scheme & LUT16_TAIL:
        below, above = lut16_tail_reach(lo, hi, shift)
        for row, reach in ((C[0], below), (C[-1], above)):
            T = reach / float(1 << shift)
            T2 = T * T
            bound = abs(row[0]) + abs(row[1]) * T + abs(row[2]) * T2 + abs(row[3]) * (T2 * T)
            if not bound < LUT16_TAIL_MAX:
                return False
    return True


def lut16(x, table, inv_scale, zero_point, quant_min, quant_max, scheme, lo, hi,
          shift, G, out_scale, out_zero_point, out_quant_min, out_quant_max):
    """espressif::lut16: q1's code (lut16_quantize), the table step
    (lut16_codes), then dequantize's arithmetic, fl(fl(y - z2) * fl(s2))."""
    c = lut16_quantize(x, inv_scale, zero_point, quant_min, quant_max)
    y = lut16_codes(c, scheme, table, lo, hi, shift, G, out_quant_min,
                    out_quant_max)
    return ((y - out_zero_point).astype(F32) * F32(out_scale)).astype(F32)


# --- The fused norm (espressif::qnorm) -------------------------------------------

def _qnorm_slot(quant, s):
    """(inv, scale, zero point, qmin, qmax) of quant row s, or None (absent:
    qmin > qmax)."""
    inv, scale, zp, qmin, qmax = (F32(v) for v in np.asarray(quant, F32)[s])
    if qmin > qmax:
        return None
    return inv, scale, int(zp), int(qmin), int(qmax)


def _qnorm_codes(v, q):
    inv, _, zp, qmin, qmax = q
    return quant_lut_codes(v, inv, zp, qmin, qmax)


def _qnorm_values(codes, q):
    """dequantize_per_tensor: fl(fl(c - zp) * scale)."""
    _, scale, zp, _, _ = q
    return ((codes - zp).astype(F32) * scale).astype(F32)


def _qnorm_fq(v, q):
    return v if q is None else _qnorm_values(_qnorm_codes(v, q), q)


def _qnorm_sum(s):
    """Each row's sum in index order from 0.0f, one float32 add per element
    (np.sum would sum pairwise)."""
    acc = np.zeros(s.shape[0], F32)
    for j in range(s.shape[1]):
        acc = (acc + s[:, j]).astype(F32)
    return acc[:, None]


QNORM16_STEP_WORDS = 6
QNORM16_TABLE_WORDS = 12


def _qnorm16_codes(v, q):
    inv, _, zp, qmin, qmax = q
    return lut16_quantize(v, inv, zp, qmin, qmax)


def _qnorm16_fq(v, q):
    return v if q is None else _qnorm_values(_qnorm16_codes(v, q), q)


def _word_f32(word):
    return np.asarray([word], np.int32).view(F32)[0]


def qnorm16_parse(tables):
    """The programs and lut16 tables of a kind 2 or 3 tables tensor (int32,
    qnorm.h's layout): ([[(op, five arguments)]], [dict])."""
    t = np.ascontiguousarray(np.asarray(tables).reshape(-1), dtype=np.int32)
    w = [int(v) for v in t]
    sw, tw = QNORM16_STEP_WORDS, QNORM16_TABLE_WORDS
    programs = [[(w[first + sw * s], w[first + sw * s + 1:first + sw * (s + 1)])
                 for s in range(steps)]
                for first, steps in (w[2 + 2 * p:4 + 2 * p] for p in range(w[0]))]
    luts = []
    base = 2 + 2 * w[0]
    for i in range(w[1]):
        (scheme, lo, hi, shift, G, out_min, out_max, count, at, exceptions,
         exc_at, _) = w[base + tw * i:base + tw * (i + 1)]
        if scheme & LUT16_BASE == 0:
            data = t[at:at + (count + 1) // 2].view(np.int16)[:count]
        elif scheme & LUT16_BASE in (1, 3):
            data = t[at:at + count]
        else:
            data = t[at:at + 4 * count].view(F32).reshape(count, 4)
        luts.append(dict(scheme=scheme, lo=lo, hi=hi, shift=shift, G=G,
                         out_min=out_min, out_max=out_max, data=data,
                         exceptions=[(w[exc_at + 2 * e], w[exc_at + 2 * e + 1])
                                     for e in range(exceptions)]))
    return programs, luts


def qnorm16_run(steps, luts, v):
    """One program (qnorm.h) on v (float32 [rows, 1]): its value, and the
    codes of its first CODE step."""
    c = first = None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for op, a in steps:
            if op == 1:                                        # CODE
                c = lut16_quantize(v, _word_f32(a[0]), a[2], a[3], a[4])
                first = c if first is None else first
            elif op == 2:                                      # TABLE
                lut = luts[a[0]]
                y = lut16_codes(c, lut["scheme"], lut["data"], lut["lo"],
                                lut["hi"], lut["shift"], lut["G"],
                                lut["out_min"], lut["out_max"])
                for code, out in lut["exceptions"]:
                    y = np.where(c == code, out, y)
                c = y
            elif op == 3:                                      # DEQ
                v = ((c - a[2]).astype(F32) * _word_f32(a[1])).astype(F32)
            else:
                k = _word_f32(a[0])
                v = (v * k if op == 4 else v + k if op == 5 else v / k
                     if op == 6 else k - v if op == 7 else k / v).astype(F32)
    return v, first


def qnorm16(x, tables, quant, weight, bias, kind):
    """espressif::qnorm's int16 kinds (2: RMSNorm, 3: LayerNorm): every pair
    applied per element, x^2 as one float32 multiply, the per-row chains as
    the tables tensor's programs."""
    x = _f32(x)
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    w = None if weight is None else np.asarray(weight, F32).reshape(1, n)
    b = None if bias is None else np.asarray(bias, F32).reshape(1, n)
    programs, luts = qnorm16_parse(tables)
    fq = _qnorm16_fq
    if kind == 2:
        qx, qsq, qm = (_qnorm_slot(quant, s) for s in range(3))
        a = rows if qx is None else _qnorm_values(_qnorm16_codes(rows, qx), qx)
        r, _ = qnorm16_run(programs[0], luts,
                           _qnorm_sum(fq((a * a).astype(F32), qsq)))
        y = (a * r).astype(F32)
        if w is not None:
            y = (fq(y, qm) * w).astype(F32)
    else:
        qx, qc1, qc2, qsq, qn, qw = (_qnorm_slot(quant, s) for s in range(6))
        a = _qnorm_values(_qnorm16_codes(rows, qx), qx)
        acc1 = _qnorm_sum(a)
        mu1, _ = qnorm16_run(programs[0], luts, acc1)
        mu2, _ = qnorm16_run(programs[1], luts, acc1)
        c2 = fq((a - mu2).astype(F32), qc2)
        r, _ = qnorm16_run(programs[2], luts,
                           _qnorm_sum(fq((c2 * c2).astype(F32), qsq)))
        y = (fq((a - mu1).astype(F32), qc1) * r).astype(F32)
        if w is not None:
            y = (fq(y, qn) * w).astype(F32)
            if b is not None:
                y = (fq(y, qw) + b).astype(F32)
        elif b is not None:
            y = (fq(y, qn) + b).astype(F32)
    return y.reshape(x.shape)


def qnorm(x, tables, quant, weight, bias, kind, flags, eps):
    """espressif::qnorm (qnorm.h): a decomposed quantised RMSNorm (kind 0) or
    LayerNorm (kind 1) over x's last axis, every pair of the pattern applied,
    tables indexed by the code that keys them, reductions in index order.
    `quant` is [slots, 5] float32 (inv, scale, zp, qmin, qmax), `tables`
    [tables, 256] float32. Kinds 2 and 3 (int16 codes): qnorm16."""
    if kind in (2, 3):
        return qnorm16(x, tables, quant, weight, bias, kind)
    x = _f32(x)
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    tables = np.asarray(tables, F32)
    w = None if weight is None else np.asarray(weight, F32).reshape(1, n)
    b = None if bias is None else np.asarray(bias, F32).reshape(1, n)

    def lookup(t, codes, q):
        return tables[t][codes - q[3]]

    if kind == 0:
        qx, qsq, qred, qm = (_qnorm_slot(quant, s) for s in range(4))
        if qx is not None:
            cx = _qnorm_codes(rows, qx)
            a = _qnorm_values(cx, qx)
            s = lookup(0, cx, qx)
        else:
            a = rows
            s = _qnorm_fq((rows * rows).astype(F32), qsq)
        v = _qnorm_sum(s)
        if flags & 1:
            v = (v / F32(n)).astype(F32)
        if flags & 2:
            v = (v + F32(eps)).astype(F32)
        r = lookup(1, _qnorm_codes(v, qred), qred)
        y = (a * r).astype(F32)
        if w is not None:
            y = (_qnorm_fq(y, qm) * w).astype(F32)
    else:
        qx, qr1, qr2, qc1, qc2, qr3, qn, qw = (_qnorm_slot(quant, s)
                                               for s in range(8))
        a = _qnorm_values(_qnorm_codes(rows, qx), qx)
        acc1 = _qnorm_sum(a)
        mu1 = lookup(0, _qnorm_codes(acc1, qr1), qr1)
        mu2 = lookup(1, _qnorm_codes(acc1, qr2), qr2)
        acc2 = _qnorm_sum(lookup(2, _qnorm_codes((a - mu2).astype(F32), qc2),
                                 qc2))
        r = lookup(3, _qnorm_codes(acc2, qr3), qr3)
        y = (_qnorm_fq((a - mu1).astype(F32), qc1) * r).astype(F32)
        if w is not None:
            y = (_qnorm_fq(y, qn) * w).astype(F32)
            if b is not None:
                y = (_qnorm_fq(y, qw) + b).astype(F32)
        elif b is not None:
            y = (_qnorm_fq(y, qn) + b).astype(F32)
    return y.reshape(x.shape)


# --- ExecuTorch's portable choose_qparams, for the record -------------------------

ET_SMALL_SCALE_THRESHOLD = F32(6.1e-5)


def et_calculate_scale_and_zero_point(lo, hi, qmin, qmax):
    """ExecuTorch's kernels/quantized/cpu/op_choose_qparams.cpp's
    calculate_scale_and_zero_point, transcribed with C's float and double
    types. It ignores eps, keeps the scale in float64 (PyTorch's is a float32
    value), floors it at 6.1e-5 by stretching the range, replaces a zero scale
    with 0.1, and picks the zero point from whichever end has the smaller
    error -- so it disagrees with every PyTorch chooser. Returns
    (scale float64, zero point int)."""
    lo = min(F32(lo), F32(0.0))
    hi = max(F32(hi), F32(0.0))
    scale = (F64(hi) - F64(lo)) / F64(qmax - qmin)
    with np.errstate(divide="ignore", over="ignore"):
        if F32(scale) == 0 or np.isinf(F32(1.0) / F32(scale)):
            scale = F64(0.1)
    if scale < F64(ET_SMALL_SCALE_THRESHOLD):
        org_scale = F32(scale)
        scale = F64(ET_SMALL_SCALE_THRESHOLD)
        if lo == 0:
            hi = F32(ET_SMALL_SCALE_THRESHOLD * F32(qmax - qmin))
        elif hi == 0:
            lo = F32(-ET_SMALL_SCALE_THRESHOLD * F32(qmax - qmin))
        else:
            amplifier = F32(ET_SMALL_SCALE_THRESHOLD / org_scale)
            lo = F32(lo * amplifier)
            hi = F32(hi * amplifier)
    from_min = F64(qmin) - F64(lo) / scale
    from_max = F64(qmax) - F64(hi) / scale
    error_min = F64(abs(qmin)) - abs(F64(lo) / scale)
    error_max = F64(abs(qmax)) - abs(F64(hi) / scale)
    initial = from_min if error_min < error_max else from_max
    if initial < qmin:
        zp = qmin
    elif initial > qmax:
        zp = qmax
    else:
        zp = int(np.rint(F32(initial)))
    return F64(scale), zp


def et_choose_qparams(x, qmin, qmax):
    """choose_qparams.tensor_out: one (scale, zp) over the whole tensor."""
    x = _f32(x)
    return et_calculate_scale_and_zero_point(x.min(), x.max(), qmin, qmax)


def et_choose_qparams_per_token_asymmetric(x):
    """choose_qparams_per_token_asymmetric.out: per row of x[..., K], [-128, 127].
    Returns (scales float64 [...], zps int64 [...])."""
    x = _f32(x)
    rows = x.reshape(-1, x.shape[-1])
    pairs = [et_calculate_scale_and_zero_point(r.min(), r.max(), -128, 127)
             for r in rows]
    shape = x.shape[:-1]
    return (np.asarray([p[0] for p in pairs], dtype=F64).reshape(shape),
            np.asarray([p[1] for p in pairs], dtype=np.int64).reshape(shape))


def expf_newlib(x):
    """newlib's expf as the ESP32-S3 libm runs it, bit for bit: the chip's exp
    in portable softmax (and the host's: ops/espressif_ops.cpp newlib_expf,
    transcribed from the same disassembly).

    fdlibm's float exp (libm/math/ef_exp.c, under wf_exp.c's errno wrapper):
    argument reduction x = hi - lo with k = trunc(fl1(invln2 * x + halF[sign]))
    and hi = fl1(x - t * ln2HI) (madd.s and msub.s on the chip: one rounding
    each, fma_f32), the polynomial c = x - t * (P1 + t * (P2 + ...)) as four
    madd.s and one msub.s, then 1 - ((x c) / (c - 2) - x) for k = 0, else
    1 - ((lo - (x c) / (2 - c)) - hi) with k added to the exponent. Every
    other operation is a separately rounded float32 one. np.exp is not this
    (a portable softmax mirror on np.exp flips prefill codes).
    """
    import numpy as np

    x = np.array(x, dtype=np.float32)
    shape = x.shape
    x = x.reshape(-1)
    bits = x.view(np.uint32).astype(np.int64)
    sign = bits >> 31
    magnitude = bits & 0x7FFFFFFF
    signed = x.view(np.int32).astype(np.int64)
    nan = magnitude > 0x7F800000
    inf = magnitude == 0x7F800000
    finite = ~(nan | inf)
    over = finite & (signed > 0x42B17217)
    under = finite & (sign == 1) & (magnitude > 0x42CFF1B5)
    ordinary = finite & ~over & ~under
    reduced = ordinary & (magnitude > 0x3EB17218)
    near = reduced & (magnitude < 0x3F851592)
    far = reduced & ~near
    tiny = ordinary & ~reduced & (magnitude < 0x34000000)
    primary = ordinary & ~reduced & ~tiny

    half = np.where(sign == 1, F32(-0.5), F32(0.5)).astype(F32)
    ln2_hi = F32(6.9313812256e-01)
    ln2_lo = F32(9.0580006145e-06)
    signed_hi = np.where(sign == 1, -ln2_hi, ln2_hi).astype(F32)
    signed_lo = np.where(sign == 1, -ln2_lo, ln2_lo).astype(F32)
    with np.errstate(all="ignore"):
        k_far = np.trunc(fma_f32(F32(1.4426950216e+00), x, half))
        k_far = np.where(far, k_far, 0).astype(np.int64)
        t_far = k_far.astype(F32)
        hi = np.where(far, fma_f32(-t_far, ln2_hi, x),
                      (x - signed_hi).astype(F32)).astype(F32)
        lo = np.where(far, (t_far * ln2_lo).astype(F32), signed_lo).astype(F32)
        k = np.where(far, k_far, np.where(near, 1 - 2 * sign, 0))
        r = np.where(reduced, (hi - lo).astype(F32), x).astype(F32)

        t = (r * r).astype(F32)
        p = fma_f32(t, F32(4.1381369442e-08), F32(-1.6533901999e-06))
        p = fma_f32(p, t, F32(6.6137559770e-05))
        p = fma_f32(p, t, F32(-2.7777778450e-03))
        p = fma_f32(p, t, F32(1.6666667163e-01))
        c = fma_f32(-p, t, r)
        rc = (r * c).astype(F32)
        one, two = F32(1.0), F32(2.0)
        k0 = (one - ((rc / (c - two)).astype(F32) - r).astype(F32)).astype(F32)
        y = (one - (((lo - (rc / (two - c)).astype(F32)).astype(F32) - hi)
                    .astype(F32))).astype(F32)
        ybits = y.view(np.uint32).astype(np.int64)
        low = k < -125
        ybits = (ybits + (np.where(low, k + 100, k) << 23)) & 0xFFFFFFFF
        y = ybits.astype(np.uint32).view(F32)
        y = np.where(low, (y * F32(2.0 ** -100)).astype(F32), y)

        out = np.where(primary, k0, y).astype(F32)
        out = np.where(tiny, (one + x).astype(F32), out)
        out = np.where(over, F32(np.inf), out)
        out = np.where(under, F32(0.0), out)
        out = np.where(inf, np.where(sign == 1, F32(0.0), x), out)
        out = np.where(nan, (x + x).astype(F32), out)
    return out.astype(F32).reshape(shape)


def fma_f32(left, right, addend):
    """float32 ``left * right + addend`` rounded ONCE, as Xtensa ``madd.s`` does.

    A product of two float32 values is exact in float64, so only the addition
    rounds there, and TwoSum recovers that rounding's exact error. Rounding the
    float64 sum to odd (an inexact result with an even last bit moves one ulp
    toward the error) and then to nearest float32 is a correct single rounding,
    because float64 carries more than two bits beyond float32 (Boldo and
    Melquiond's round-to-odd theorem). Casting the plain float64 sum is not:
    that double-rounds, which goes wrong exactly on float32 ties.
    """
    import numpy as np

    left = np.asarray(left, dtype=np.float32).astype(np.float64)
    right = np.asarray(right, dtype=np.float32).astype(np.float64)
    addend = np.asarray(addend, dtype=np.float32).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        product = left * right
        total = np.asarray(product + addend, dtype=np.float64)
        virtual = total - product
        error = (product - (total - virtual)) + (addend - virtual)
        shape = total.shape
        bits = total.reshape(-1).view(np.int64)
        error = np.broadcast_to(error, shape).reshape(-1)
        finite = np.isfinite(total).reshape(-1)
        inexact = finite & (error != 0) & ((bits & 1) == 0)
        # One step of the bit pattern grows the magnitude for either sign, which
        # is toward the exact value when the error has the sum's sign.
        step = np.where((error > 0) == (total.reshape(-1) > 0), 1, -1)
        bits = np.where(inexact, bits + step, bits)
    return bits.view(np.float64).astype(np.float32).reshape(shape)
