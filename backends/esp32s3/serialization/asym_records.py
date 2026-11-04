# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The blob's asymmetric-quantisation records: build them, pack them, and
refuse at export what the record cannot say.

espressif_blob.py names the wire constants (FLAG_ACT_QUANT, EXT_ACT_QUANT, the
field values) and runtime/esp32_asym.h defines what every field means;
esp32_asym.cpp holds the runtime's rules. This module is the exporter's side
of that contract: the emitters call it for weight zero points and float
offsets, for the dynamic activation quantisers an absorbed chain leaves on a
matmul (with the weight sums their zero points need and their exactness
bounds) and for the static records of static graphs. asym_blob_cases.py builds
the synthetic refusal blobs with it.

Five schemes are refused here, by name. Two have no record at all: the
backend's own quantiser (A0: absmax / 127, half away from zero) is what an op
WITHOUT a record does, and XNNPACK's per-token chooser (A4) makes the K
cache's codes (kv_affine), never an op's activation; the record's chooser and
rounding fields have no value for either. Three because the record cannot
express what is wrong with them and the runtime therefore cannot see it:

  - a static scale that is not a float32 value. torch quantises with
    fl32(fl64(1/s)) but dequantises with fl32(s), and one f32 field cannot
    reproduce both (226 of the 999 scales 0.001..0.999 differ). Observer-made
    scales are always float32 values.
  - unsigned symmetric activations. PT2E puts their zero point at 128,
    torchao at the range's midpoint; the two disagree for every range but
    [0, 255], and no stock activation recipe asks for one.
  - torchao's SYMMETRIC_NO_CLIPPING_ERR activations: the scale is the larger
    of min_neg/qmin and max_pos/qmax, a chooser no record describes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, fields

import numpy as np

from executorch.backends.esp32s3.serialization import espressif_blob as eb


class SchemeRefused(ValueError):
    """A quantisation scheme the backend refuses by name at export."""


def _pack(fmt, names, record):
    return struct.pack(fmt, *(getattr(record, name) for name in names))


@dataclass(frozen=True)
class ActQuant:
    """EXT_ACT_QUANT (44 B): how a quantised matmul quantises its activation."""

    in_dtype: int = eb.DTYPE_FP32
    code_bits: int = 8
    granularity: int = eb.GRANULARITY_ROW
    chooser: int = eb.CHOOSER_AFFINE_MINMAX
    rounding: int = eb.ROUND_RNE_BEFORE_ZP
    qmin: int = -128
    qmax: int = 127
    eps: float = 0.0
    sym_div: float = 0.0
    scale: float = 0.0
    zero_point: int = 0

    def pack(self) -> bytes:
        return _pack(eb.ACT_QUANT_FORMAT, eb.ACT_QUANT_FIELDS, self)

    @classmethod
    def unpack(cls, payload: bytes) -> "ActQuant":
        return cls(*struct.unpack(eb.ACT_QUANT_FORMAT, payload))

    @property
    def affine(self) -> bool:
        """Whether the zero point can be nonzero (the runtime's act_affine)."""
        return (self.chooser == eb.CHOOSER_AFFINE_MINMAX
                or (self.granularity == eb.GRANULARITY_STATIC
                    and self.zero_point != 0))


@dataclass(frozen=True)
class WeightAffine:
    """EXT_WEIGHT_AFFINE (28 B): zero points, float offsets and weight sums,
    one entry per scale group; the offsets are blob offsets or, under
    FLAG_EXTERNAL_WEIGHTS, the offsets of NUL-terminated named-data keys."""

    zp_kind: int = eb.ZP_NONE
    offset_kind: int = eb.OFFSET_NONE
    sum_kind: int = eb.SUM_NONE
    storage_bias: int = 0
    zp_off: int = 0
    offset_off: int = 0
    sum_off: int = 0

    def pack(self) -> bytes:
        return _pack(eb.WEIGHT_AFFINE_FORMAT, eb.WEIGHT_AFFINE_FIELDS, self)

    @classmethod
    def unpack(cls, payload: bytes) -> "WeightAffine":
        return cls(*struct.unpack(eb.WEIGHT_AFFINE_FORMAT, payload))


@dataclass(frozen=True)
class OutQuant:
    """EXT_OUT_QUANT (24 B): the model's quantize of an op's output."""

    out_dtype: int = eb.DTYPE_FP32
    rounding: int = eb.ROUND_RNE_BEFORE_ZP
    scale: float = 1.0
    zero_point: int = 0
    qmin: int = -128
    qmax: int = 127

    def pack(self) -> bytes:
        return _pack(eb.OUT_QUANT_FORMAT, eb.OUT_QUANT_FIELDS, self)

    @classmethod
    def unpack(cls, payload: bytes) -> "OutQuant":
        return cls(*struct.unpack(eb.OUT_QUANT_FORMAT, payload))


# The dataclasses ARE the wire order: a field added to one side only fails here.
for _record, _names in ((ActQuant, eb.ACT_QUANT_FIELDS),
                        (WeightAffine, eb.WEIGHT_AFFINE_FIELDS),
                        (OutQuant, eb.OUT_QUANT_FIELDS)):
    assert tuple(f.name for f in fields(_record)) == _names, _record


def require_float32_scale(scale, what="static activation scale"):
    """The scale as the float32 value it must already be, or SchemeRefused."""
    as_f32 = float(np.float32(scale))
    if not np.isfinite(as_f32) or as_f32 != float(scale):
        raise SchemeRefused(
            f"{what} {float(scale)!r} is not a float32 value: torch quantises "
            f"with fl32(fl64(1/s)) but dequantises with fl32(s), and the "
            f"record carries one f32 (observer-made scales are float32 values)")
    return as_f32


def activation_chooser(mapping_type):
    """A torchao MappingType (the enum or its name) -> the mirrors' chooser
    name. SYMMETRIC_NO_CLIPPING_ERR is refused by name."""
    name = getattr(mapping_type, "name", str(mapping_type)).rsplit(".", 1)[-1]
    if name == "SYMMETRIC_NO_CLIPPING_ERR":
        raise SchemeRefused(
            "torchao SYMMETRIC_NO_CLIPPING_ERR activations are refused: their "
            "scale is the larger of min_neg/qmin and max_pos/qmax, a chooser "
            "no blob record describes")
    try:
        return {"SYMMETRIC": "symmetric", "ASYMMETRIC": "affine"}[name]
    except KeyError:
        raise SchemeRefused(f"torchao mapping type {name} is not supported") \
            from None


_CHOOSERS = {"symmetric": eb.CHOOSER_SYM_DIV,
             "affine": eb.CHOOSER_AFFINE_MINMAX, "static": eb.CHOOSER_NONE}
_GRANULARITIES = {"row": eb.GRANULARITY_ROW, "tensor": eb.GRANULARITY_TENSOR,
                  "static": eb.GRANULARITY_STATIC}


def act_quant(scheme) -> ActQuant:
    """The EXT_ACT_QUANT record for a mirrors.ActScheme (the reference
    the runtime must reproduce bit for bit).

    uint8 ranges arrive shifted by -128 onto int8 codes, zero point included
    (mirrors.uint8_to_int8: exact). A symmetric chooser divides by
    (qmax - qmin) / 2 -- 127 for A1's [-127, 127], 127.5 for torchao's
    [-128, 127] -- which is what the choosers do. A0 and A4 are no record's
    (the module docstring) and are refused by name.
    """
    if scheme.chooser == "absmax" or scheme.rounding == "half_away":
        raise SchemeRefused(
            f"{scheme.name} is the backend's own activation quantiser (A0): an "
            f"op without an activation record quantises so; no record does")
    if scheme.chooser == "token_xnn" or scheme.rounding != "rne_before_zp":
        raise SchemeRefused(
            f"{scheme.name} ({scheme.chooser}, {scheme.rounding}) is XNNPACK's "
            f"per-token chooser (A4), which makes the K cache's own codes, "
            f"never an op's activation: no record carries it")
    qmin, qmax, zero_point = int(scheme.qmin), int(scheme.qmax), \
        int(scheme.zero_point)
    symmetric = scheme.chooser == "symmetric"
    if symmetric and qmin >= 0:
        raise SchemeRefused(
            f"unsigned symmetric activations [{qmin}, {qmax}] are refused: "
            f"PT2E puts the zero point at 128 and torchao at the midpoint")
    if qmin >= 0 and qmax <= 255:
        qmin, qmax, zero_point = qmin - 128, qmax - 128, zero_point - 128
    code_bits = 8 if -128 <= qmin and qmax <= 127 else 16
    fixed = scheme.chooser == "static"
    return ActQuant(
        in_dtype=eb.DTYPE_FP32, code_bits=code_bits,
        granularity=_GRANULARITIES[scheme.granularity],
        chooser=_CHOOSERS[scheme.chooser], rounding=eb.ROUND_RNE_BEFORE_ZP,
        qmin=qmin, qmax=qmax,
        eps=0.0 if fixed else float(np.float32(scheme.eps)),
        sym_div=float(np.float32((qmax - qmin) / 2)) if symmetric else 0.0,
        scale=require_float32_scale(scheme.scale) if fixed else 0.0,
        zero_point=zero_point if fixed else 0)


def out_quant(scale, zero_point, qmin, qmax, codes=False) -> OutQuant:
    """The EXT_OUT_QUANT record for a static output quantize: Stage A
    fake-quant (the codes never leave the op, so the range is kept as is), or
    with `codes` Stage B's int8 codes out (out_dtype 1; an int8 range)."""
    if codes and not (-128 <= int(qmin) < int(qmax) <= 127):
        raise SchemeRefused(
            f"int8 output codes over [{qmin}, {qmax}]: Stage B writes int8 "
            f"codes only")
    return OutQuant(out_dtype=eb.DTYPE_INT8 if codes else eb.DTYPE_FP32,
                    rounding=eb.ROUND_RNE_BEFORE_ZP,
                    scale=require_float32_scale(scale, "static output scale"),
                    zero_point=int(zero_point), qmin=int(qmin), qmax=int(qmax))


# --- weight zero points and float offsets -----------------------------------------
# What a pass-through weight carries beyond codes and scales, normalised to the
# record's conventions: signed stored codes c, integer zero points zw in the
# same (biased) domain, a fractional float zero point on the source's own code,
# float offsets. const_weights.py reads them off the graph; this turns them into
# the record and its tables.

@dataclass(frozen=True)
class WeightScheme:
    """A pass-through weight's codes and zero points in the stored domain.

    codes          signed stored codes c = q - storage_bias (int8 container)
    qmin, qmax     their declared range, shifted the same way
    zero_points    int64 zw = z - storage_bias per scale group, or None when
                   every group's zero point is the same: 0 with storage_bias 0
                   (symmetric), or -storage_bias (a source without one)
    float_zero_points  float32 per group, applied to q = c + storage_bias:
                   a fractional zero point (embeddings only), else None
    storage_bias   0 (signed source), 8 (uint4) or 128 (uint8)
    """

    codes: np.ndarray
    qmin: int
    qmax: int
    zero_points: np.ndarray | None
    float_zero_points: np.ndarray | None
    storage_bias: int

    @property
    def affine(self) -> bool:
        return (self.zero_points is not None
                or self.float_zero_points is not None
                or self.storage_bias != 0)


def storage_bias_for(qmin: int, qmax: int) -> int:
    """What an unsigned range's codes are shifted by to store them signed:
    8 onto int4's [-8, 7], 128 onto int8's [-128, 127]; 0 for a signed one."""
    if qmin < 0:
        return 0
    if qmax <= 15:
        return 8
    if qmax <= 255:
        return 128
    raise SchemeRefused(f"unsigned codes [{qmin}, {qmax}] are wider than "
                        f"uint8; no stored code holds them")


def normalize_weight(codes, qmin: int, qmax: int, zero_points=None) -> WeightScheme:
    """Unsigned codes onto the signed kernel and zero points into the record's
    domain: c = q - bias, zw = z - bias, both exact.

    `zero_points` is None (none), an integer array, or a float array; a float
    one whose values are all integers is an integer zero point (torchao's legacy
    quantiser stores them as floats), a fractional one stays float.
    """
    bias = storage_bias_for(int(qmin), int(qmax))
    c = np.asarray(codes).astype(np.int64) - bias
    qmin, qmax = int(qmin) - bias, int(qmax) - bias
    if not (-128 <= qmin and qmax <= 127):
        raise SchemeRefused(f"weight codes [{qmin + bias}, {qmax + bias}] do "
                            f"not fit int8 storage")
    stored = c.astype(np.int8)
    if zero_points is None:
        # No zero point: the stored codes' is -bias (GGUF's fl(q * d)).
        return WeightScheme(stored, qmin, qmax, None, None, bias)
    z = np.asarray(zero_points)
    if z.dtype.kind == "f":
        z32 = z.astype(np.float32)
        if not np.all(np.isfinite(z32)):
            raise SchemeRefused("a weight zero point is not finite")
        if not np.array_equal(z32, np.round(z32)):
            return WeightScheme(stored, qmin, qmax, None, z32, bias)
        z = z32.astype(np.int64)
    zw = z.astype(np.int64) - bias
    if not np.any(zw):
        # Zero in the stored domain: c are symmetric codes, nothing to record.
        return WeightScheme(stored, qmin, qmax, None, None, 0)
    if bias and np.all(zw == -bias):
        return WeightScheme(stored, qmin, qmax, None, None, bias)
    lo, hi = (-8, 7) if (qmin >= -8 and qmax <= 7) else (-128, 127)
    if int(zw.min()) < lo or int(zw.max()) > hi:
        raise SchemeRefused(
            f"weight zero points span [{int(zw.min()) + bias}, "
            f"{int(zw.max()) + bias}], outside the stored codes' range "
            f"[{lo + bias}, {hi + bias}]; the exactness bound assumes "
            f"|c - zw| within the code width")
    return WeightScheme(stored, qmin, qmax, zw, None, bias)


def weight_dot_bound(bits: int, affine: bool) -> int:
    """The longest group (or row) an exact int32 dot holds for the backend's
    own activations (|x| <= 127) against these weights."""
    from executorch.backends.esp32s3.quantizer.mirrors import (
        dot_length_bound,
        weight_span,
    )

    return dot_length_bound(127, weight_span(bits, affine))


def check_weight_bounds(bits: int, longest: int, *, zero_point: bool,
                        offsets: bool, what: str = "weight") -> None:
    """The runtime's dot-bound and offset-bound rules for one weight, at
    export, by name (esp32_asym.cpp check_bounds)."""
    bound = weight_dot_bound(bits, zero_point)
    if longest > bound:
        raise SchemeRefused(
            f"{what}: a dot over {longest} elements against "
            f"{'zero-point ' if zero_point else ''}int{bits} codes; an exact "
            f"int32 accumulator holds {bound}")
    if offsets and longest * 127 > (1 << 24):
        raise SchemeRefused(
            f"{what}: float offsets over groups of {longest}, where X_g can "
            f"pass 2^24 and its float conversion is no longer exact")


def table_payloads(rows: int, groups: int, *, zero_points=None,
                   float_zero_points=None, offsets=None):
    """The record's tables as bytes, [rows, groups] each: int8 integer zero
    points, fp32 float zero points, and offsets in fp16 when every value
    survives the round trip (the scales' rule), else fp32. Returns
    (zp_kind, zp_bytes, offset_kind, offset_bytes); a kind of 0 has no bytes."""
    def shaped(table, dtype):
        a = np.asarray(table)
        if a.ndim == 0:
            a = np.broadcast_to(a, (rows, groups))   # one value for the tensor
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.shape[1] == 1 and groups > 1:
            a = np.repeat(a, groups, axis=1)
        if a.shape != (rows, groups):
            raise SchemeRefused(f"a weight table is {a.shape}, expected "
                                f"({rows}, {groups})")
        return np.ascontiguousarray(a.astype(dtype))

    zp_kind, zp_bytes = eb.ZP_NONE, b""
    if zero_points is not None:
        zp_kind, zp_bytes = eb.ZP_INT8, shaped(zero_points, np.int8).tobytes()
    elif float_zero_points is not None:
        zp_kind = eb.ZP_FP32
        zp_bytes = shaped(float_zero_points, np.float32).tobytes()
    offset_kind, offset_bytes = eb.OFFSET_NONE, b""
    if offsets is not None:
        m = shaped(offsets, np.float32)
        if np.array_equal(m.astype(np.float16).astype(np.float32), m):
            offset_kind, offset_bytes = eb.OFFSET_FP16, m.astype(np.float16).tobytes()
        else:
            offset_kind, offset_bytes = eb.OFFSET_FP32, m.tobytes()
    return zp_kind, zp_bytes, offset_kind, offset_bytes


# --- dynamic activations ----------------------------------------------------------
# An op whose activation the model quantises itself (EXT_ACT_QUANT, from an
# absorbed dynamic chain): the record's code span decides the exactness bound,
# and an activation that can have a zero point needs the weight sums S'.

def act_span(act: ActQuant) -> int:
    """max |xq - zx| over every code the scheme makes (the runtime's act_span,
    mirrors.py ActScheme.span): qmax - qmin once a zero point can be
    nonzero, the larger code magnitude otherwise."""
    if act.code_bits == 16:
        return 32768
    if act.affine:
        return act.qmax - act.qmin
    return max(-act.qmin, act.qmax)


def check_act_bounds(act: ActQuant, bits: int, longest: int, *,
                     zero_point: bool, offsets: bool, what: str = "op") -> None:
    """The runtime's dot-bound and offset-bound rules for an op with an
    activation record, at export, by name (esp32_asym.cpp check_bounds):
    65,792 for affine int8 activations against symmetric int8 weights, 33,024
    against zero-point ones, wider for int4 (where the fused kernel's 65,536
    binds first); 131,040 for 16-bit codes against int8 weights, the 40-bit
    ACCX's bound, which the device's int64 dot keeps although it
    does not need it."""
    from executorch.backends.esp32s3.quantizer.mirrors import (
        ACCX_MAX,
        INT32_MAX,
        dot_length_bound,
        weight_span,
    )

    span = act_span(act)
    wide = act.code_bits == 16
    bound = dot_length_bound(span, weight_span(bits, zero_point),
                             ACCX_MAX if wide else INT32_MAX)
    if longest > bound:
        raise SchemeRefused(
            f"[act-dot-bound] {what}: a dot over {longest} elements of "
            f"activation codes spanning {span} against "
            f"{'zero-point ' if zero_point else ''}int{bits} weights; an "
            f"exact {'40-bit' if wide else 'int32'} accumulator holds {bound}")
    if offsets and longest * span > (1 << 24):
        raise SchemeRefused(
            f"[act-offset-bound] {what}: float offsets over groups of "
            f"{longest}, where X'_g can pass 2^24 and its float conversion is "
            f"no longer exact")


def weight_sums_table(codes, rows: int, groups: int, k: int, gsize: int,
                      zero_points=None, storage_bias: int = 0) -> np.ndarray:
    """S'[n, g] = sum over group g's REAL elements (columns < k; the k_pad
    tail is outside every count) of (c - zw), int64
    [rows, groups]. `gsize` 0 is one group per row; zero points are the
    record's zw ([rows] or [rows, groups]), or -storage_bias for all without
    a table. mirrors.weight_sums is the reference."""
    c = np.asarray(codes, dtype=np.int64)[:rows, :k]
    g = gsize if 0 < gsize < k else k
    if zero_points is None:
        zw = np.full((rows, groups), -int(storage_bias), dtype=np.int64)
    else:
        zw = np.asarray(zero_points, dtype=np.int64).reshape(rows, -1)
        if zw.shape[1] == 1:
            zw = np.repeat(zw, groups, axis=1)
    out = np.zeros((rows, groups), dtype=np.int64)
    for j in range(groups):
        a, b = j * g, min((j + 1) * g, k)
        if a < b:
            out[:, j] = (c[:, a:b] - zw[:, j:j + 1]).sum(axis=1)
    return out


def sums_payload(codes, rows: int, groups: int, k: int, gsize: int, bits: int,
                 *, zero_points=None, storage_bias: int = 0):
    """(sum_kind, bytes, element size) of the S' table: int16 when the
    longest group times the widest |c - zw| provably fits (the runtime's
    weight-sums-width rule, the same arithmetic), else int32."""
    from executorch.backends.esp32s3.quantizer.mirrors import weight_span

    table = weight_sums_table(codes, rows, groups, k, gsize, zero_points,
                              storage_bias)
    longest = gsize if 0 < gsize < k else k
    zero_point = zero_points is not None or storage_bias != 0
    if longest * weight_span(bits, zero_point) <= 32767:
        return eb.SUM_INT16, table.astype(np.int16).tobytes(), 2
    return eb.SUM_INT32, table.astype(np.int32).tobytes(), 4


def _act_record(node):
    """The EXT_ACT_QUANT record (asym_records.ActQuant) of the model's own
    activation quantiser on this op (node.meta["esp32_act_quant"], a
    mirrors.ActScheme): the dynamic one an absorbed chain left
    (strip_dynamic_act_qdq), or the static pair it reads
    (Esp32OperatorSupport._static_act_problem, static_qdq.py). None for the
    backend's own activations."""
    scheme = node.meta.get("esp32_act_quant")
    if scheme is None:
        return None
    from executorch.backends.esp32s3.serialization.asym_records import act_quant

    return act_quant(scheme)
