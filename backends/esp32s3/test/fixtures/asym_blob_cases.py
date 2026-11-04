# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic blobs, one per refusal rule, for the host runner and QEMU.

The runtime computes the blob's weight records (zero points, float offsets),
its dynamic activation records (per row or per tensor, affine or symmetric,
with their weight sums), the static int8 and 16-bit ones, the output records
and the affine K-cache opcode; the rest are refused. The runtime's rules
(runtime/esp32_asym.cpp) cover far more than any exporter emits,
so this module builds blobs by hand: a few VALID blobs, each either
computed (rule ACCEPTED: it must load, refused by nothing) or refused only
as "unimplemented", and for each rule a copy of one of them with exactly one
field wrong, which must be refused for THAT rule. Each case names its rule,
and both harnesses read the rule the runtime names -- from the log line on the
host, from esp32_asym::last_refusal() on the device -- so a case refused by
some other rule fails, which a bare "rejected" could never show.

A blob has to travel inside a program, and a delegate payload's length is fixed
by the program around it. So the cases ride in a CARRIER: a real exported
program with one delegate (a 64x64 Linear, inline int8 weights) whose payload
is bigger than any case. A case is written over the start of that payload and
the rest zeroed; the runtime sees the payload's length as the blob size and
reads nothing past the case's own offsets. The unmodified carrier is the
control: it must load and run.

    export_carrier(path)                  the carrier .pte (imports torch)
    CASES                                 [Case(name, rule, blob)]
    write_cases(path, carrier_pte)        the device's case file (below)
    patched(carrier_bytes, span, blob)    one case's program bytes

Case file layout (little-endian; main.cpp run_asym_refusal_check reads it):
    "ASYC", u32 version (1), u32 payload offset in the carrier, u32 payload
    length, u32 count, then per case: u16 name length, u16 rule length,
    u32 blob length, name, rule, blob, zero padding to 4.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, replace

from executorch.backends.esp32s3.serialization import espressif_blob as eb
from executorch.backends.esp32s3.serialization.asym_records import (
    ActQuant,
    OutQuant,
    WeightAffine,
)
from executorch.backends.esp32s3.quantizer.mirrors import (
    ACCX_MAX,
    FLT_MIN,
    dot_length_bound,
)

CASES_MAGIC = b"ASYC"
CASES_VERSION = 1

# The runtime's rule names (esp32_asym.cpp kRuleNames), every one a case below
# except kcache-zero-points (an execute-time check; esp32_asym::
# run_kcache_zero_point_check drives it with hand-built tensors) and none.
RULES = (
    "record-size", "record-without-flag", "flag-without-record",
    "duplicate-record", "act-opcode", "act-scheme-flags", "act-field-value",
    "act-stage-b", "act-code-bits", "act-range", "act-combination",
    "act-unsigned-symmetric", "act-static-scale", "act-static-zero-point",
    "act-eps", "act-sym-div", "act-unused-field",
    "act-zero-point-without-sums", "weight-opcode", "weight-field-value",
    "weight-empty", "weight-float-zp-matmul", "weight-combination",
    "weight-sums-unused", "weight-sums-width", "weight-staged-offsets",
    "weight-unused-field", "weight-table-bounds", "weight-table-external",
    "weight-zero-point-range", "out-opcode", "out-fused-operand",
    "out-field-value", "out-stage-b", "out-range",
    "out-scale", "out-zero-point", "dot-bound", "offset-bound",
    "kcache-args", "kcache-descriptor", "kcache-dot-bound", "unimplemented",
)
EXECUTE_TIME_RULES = ("kcache-zero-points",)
# A valid case this runtime computes: it must load, refused by no rule.
ACCEPTED = "accepted"

# Argument indices the synthetic ops name. The runtime checks arguments only at
# execute, which a refused blob never reaches.
IN, OUT, RES, CODES, SCALES, ZPS, KOUT = 0, 1, 2, 1, 2, 3, 4


@dataclass(frozen=True)
class Op:
    """One synthetic op, before placement: its descriptor fields and, for an
    asymmetric op, its records (ActQuant/WeightAffine/OutQuant or None)."""

    opcode: int
    m: int = 4
    k: int = 32
    n: int = 8
    k_pad: int = 32
    flags: int = 0
    scale_group: int = 0
    in_arg: int = IN
    out_arg: int = OUT
    res_arg: int = 0
    act_scale: float = 0.0
    # OP_BMM_KCACHE_AFFINE_I8's other ARG indices (in scale_off and wt_off).
    scales_arg: int = SCALES
    zps_arg: int = ZPS
    act: ActQuant | None = None
    weight: WeightAffine | None = None
    out: OutQuant | None = None
    # Table contents for EXT_WEIGHT_AFFINE (bytes, or None for a zero table),
    # and deliberate damage for the record-level rules.
    zp_bytes: bytes | None = None
    record_len_delta: int = 0       # EXT_ACT_QUANT payload shortened by this
    drop_records: tuple = ()        # record types left out (flags kept)
    duplicate_records: tuple = ()   # record types emitted twice
    clear_flags: int = 0            # flags removed (records kept)
    table_offset_delta: int = 0     # added to every placed table offset
    no_weights: bool = False        # descriptor only: its rule fires first


def _pad(n, a):
    return -(-n // a) * a


def _groups(op: Op) -> int:
    span = op.k if op.opcode in (eb.OP_EMBED_I8, eb.OP_EMBED_I4) else op.k_pad
    g = op.scale_group or span
    return -(-span // g)


def _row_bytes(op: Op) -> int:
    if op.opcode in (eb.OP_ADDMM_I4, eb.OP_EMBED_I4):
        return op.k_pad // 2
    if op.opcode in (eb.OP_ADDMM_I16, eb.OP_EMBED_I16):
        return op.k_pad * 2
    return op.k_pad


class _Blob:
    """Header, descriptors, body, extension table: _BlobWriter's layout."""

    def __init__(self, n_ops):
        self.offset = eb.BLOB_HEADER_BYTES + eb.OP_DESC_BYTES * n_ops
        self.body = bytearray()
        self.descs = []
        self.records = []

    def place(self, payload, align=4):
        pad = _pad(self.offset, align) - self.offset
        self.body.extend(b"\0" * pad)
        self.offset += pad
        at = self.offset
        self.body.extend(payload)
        self.offset += len(payload)
        return at

    def key(self, name):
        return self.place(name.encode() + b"\0", align=1)

    def finish(self):
        ext_off, ext = 0, b""
        if self.records:
            ext_off = _pad(self.offset, 4)
            self.body.extend(b"\0" * (ext_off - self.offset))
            parts = [struct.pack(eb.EXT_COUNT_FORMAT, len(self.records))]
            for op_index, ext_type, payload in self.records:
                parts.append(struct.pack(eb.EXT_RECORD_HEADER_FORMAT, op_index,
                                         ext_type, len(payload)))
                parts.append(payload + b"\0" * (-len(payload) % 4))
            ext = b"".join(parts)
        header = struct.pack(eb.BLOB_HEADER_FORMAT, eb.MAGIC, eb.VERSION,
                             len(self.descs), ext_off)
        return header + b"".join(self.descs) + bytes(self.body) + ext


def _place_tables(blob: _Blob, op: Op, external: bool) -> WeightAffine:
    """Place EXT_WEIGHT_AFFINE's tables and return the record pointing at
    them (keys under FLAG_EXTERNAL_WEIGHTS)."""
    w = op.weight
    entries = op.n * _groups(op)
    sizes = {"zp": {eb.ZP_INT8: 1, eb.ZP_FP32: 4},
             "offset": {eb.OFFSET_FP32: 4, eb.OFFSET_FP16: 2},
             "sum": {eb.SUM_INT16: 2, eb.SUM_INT32: 4}}
    offs = {}
    for table, kind in (("zp", w.zp_kind), ("offset", w.offset_kind),
                        ("sum", w.sum_kind)):
        elem = sizes[table].get(kind)
        if elem is None:
            continue
        if external:
            offs[table] = blob.key(f"asym_{table}_table")
            continue
        payload = (op.zp_bytes if table == "zp" and op.zp_bytes is not None
                   else bytes(entries * elem))
        offs[table] = blob.place(payload, align=elem) + op.table_offset_delta
    return replace(w, zp_off=offs.get("zp", w.zp_off),
                   offset_off=offs.get("offset", w.offset_off),
                   sum_off=offs.get("sum", w.sum_off))


def build(*ops: Op) -> bytes:
    """A blob of these ops, each placing what its descriptor stores."""
    blob = _Blob(len(ops))
    for index, op in enumerate(ops):
        flags = op.flags
        external = bool(flags & eb.FLAG_EXTERNAL_WEIGHTS)
        bias_off = wt_off = scale_off = 0
        groups = _groups(op)
        if op.opcode in (eb.OP_BMM_DYN_I8, eb.OP_BMM_KCACHE_I8,
                         eb.OP_BMM_KCACHE_AFFINE_I8):
            bias_off = 0
        elif op.no_weights:
            pass
        elif external:
            wt_off = blob.key("asym_weights")
            scale_off = blob.key("asym_scales")
        elif op.opcode == eb.OP_ADDMM:
            bias_off = blob.place(bytes(4 * op.n))
            wt_off = blob.place(bytes(4 * op.k * op.n))
        else:
            if op.opcode not in (eb.OP_EMBED_I8, eb.OP_EMBED_I4,
                                 eb.OP_EMBED_I16):
                bias_off = blob.place(bytes(4 * op.n))
            wt_off = blob.place(bytes(op.n * _row_bytes(op)), align=16)
            scale_bytes = 2 if flags & eb.FLAG_FP16_SCALES else 4
            scale_off = blob.place(bytes(op.n * groups * scale_bytes))
        weight = None
        if op.weight is not None:
            weight = _place_tables(blob, op, external)
        if op.opcode == eb.OP_BMM_KCACHE_AFFINE_I8:
            wt_off, scale_off = op.zps_arg, op.scales_arg
        desc = dict(opcode=op.opcode, flags=flags & ~op.clear_flags, m=op.m,
                    k=op.k, n=op.n, k_pad=op.k_pad, bias_off=bias_off,
                    wt_off=wt_off, scale_off=scale_off, in_arg=op.in_arg,
                    out_arg=op.out_arg, scale_group=op.scale_group,
                    clamp_lo=0.0, clamp_hi=0.0, res_arg=op.res_arg,
                    act_scale=op.act_scale)
        blob.descs.append(struct.pack(eb.OP_DESC_FORMAT,
                                      *(desc[f] for f in eb.OP_DESC_FIELDS)))
        for ext_type, record in ((eb.EXT_ACT_QUANT, op.act),
                                 (eb.EXT_WEIGHT_AFFINE, weight),
                                 (eb.EXT_OUT_QUANT, op.out)):
            if record is None or ext_type in op.drop_records:
                continue
            payload = record.pack()
            if ext_type == eb.EXT_ACT_QUANT and op.record_len_delta:
                payload = payload[:len(payload) - op.record_len_delta]
            copies = 2 if ext_type in op.duplicate_records else 1
            blob.records.extend([(index, ext_type, payload)] * copies)
    return blob.finish()


# --- the valid blobs -------------------------------------------------------------
# Torchao's per-token activation chooser (A3) and its weight sums: the shape of
# an 8da8w projection's activation side.
A3 = ActQuant(granularity=eb.GRANULARITY_ROW, chooser=eb.CHOOSER_AFFINE_MINMAX,
              rounding=eb.ROUND_RNE_BEFORE_ZP, qmin=-128, qmax=127,
              eps=FLT_MIN)
SUMS16 = WeightAffine(sum_kind=eb.SUM_INT16)
F_ACT = eb.FLAG_ACT_QUANT
F_W = eb.FLAG_WEIGHT_AFFINE
F_OUT = eb.FLAG_OUT_QUANT

V_ACT = Op(eb.OP_ADDMM_I8, flags=F_ACT | F_W, act=A3, weight=SUMS16)
# torchao's symmetric per-token chooser on its own grid (A7): no zero point,
# so no weight sums and no weight record.
A7 = ActQuant(granularity=eb.GRANULARITY_ROW, chooser=eb.CHOOSER_SYM_DIV,
              rounding=eb.ROUND_RNE_BEFORE_ZP, qmin=-128, qmax=127,
              eps=FLT_MIN, sym_div=127.5)
V_SYM = Op(eb.OP_ADDMM_I4, k=64, k_pad=64, scale_group=32, flags=F_ACT, act=A7)
# Asymmetric int4 group-32 weights (W-zp), backend activations. Kept small:
# the case file is embedded in the parity firmware's app image, and executing
# an affine op is the asym fixtures' job, so an accepted case only loads.
V_WZP_I4 = Op(eb.OP_ADDMM_I4, k=64, k_pad=64, scale_group=32, flags=F_W,
              weight=WeightAffine(zp_kind=eb.ZP_INT8),
              zp_bytes=bytes([0x05, 0xF8] * 8))      # zw 5 and -8
# GGUF's fl(fl(q*d) + mn): uint4 codes stored as q - 8 (storage bias 8, so the
# stored codes' zero point is -8) and fp16 offsets, per group of 32.
V_GGUF_I4 = Op(eb.OP_ADDMM_I4, k=64, k_pad=64, scale_group=32,
               flags=F_W | eb.FLAG_FP16_SCALES,
               weight=WeightAffine(offset_kind=eb.OFFSET_FP16, storage_bias=8))
# An embedding with fractional float zero points (fl(fl(q - z) * s)).
V_EMBED_FZP = Op(eb.OP_EMBED_I8, m=3, k=32, n=16, k_pad=32, flags=F_W,
                 weight=WeightAffine(zp_kind=eb.ZP_FP32))
# A static output quantize, as fake-quant (Stage A).
OUT_Q = OutQuant(out_dtype=eb.DTYPE_FP32, rounding=eb.ROUND_RNE_BEFORE_ZP,
                 scale=0.0625, zero_point=3, qmin=-128, qmax=127)
V_OUT = Op(eb.OP_ADDMM_I8, flags=F_OUT, out=OUT_Q)
# Static affine activations (XNNPACK/Arm a8w8): a nonzero zero point needs S'.
STATIC_A = ActQuant(granularity=eb.GRANULARITY_STATIC,
                    chooser=eb.CHOOSER_NONE, rounding=eb.ROUND_RNE_BEFORE_ZP,
                    qmin=-128, qmax=127, scale=0.03125, zero_point=-5)
V_STATIC = Op(eb.OP_ADDMM_I8, flags=F_ACT | F_W, act=STATIC_A, weight=SUMS16)
# Arm a16w8: static symmetric int16 activations on an ungrouped int8 matmul.
A16 = ActQuant(code_bits=16, granularity=eb.GRANULARITY_STATIC,
               chooser=eb.CHOOSER_NONE, rounding=eb.ROUND_RNE_BEFORE_ZP,
               qmin=-32767, qmax=32767, scale=0.0009765625, zero_point=0)
V_A16 = Op(eb.OP_ADDMM_I8, flags=F_ACT, act=A16)
# q@kT against an asymmetric int8 K cache: batch 2, 3 queries, head 32, 5 keys.
V_KCACHE = Op(eb.OP_BMM_KCACHE_AFFINE_I8, m=3, k=32, n=5, k_pad=32,
              scale_group=2, in_arg=IN, out_arg=KOUT, res_arg=CODES)


@dataclass(frozen=True)
class Case:
    name: str
    rule: str
    blob: bytes


def _cases():
    c = []

    def add(name, rule, *ops):
        c.append(Case(name, rule, build(*ops)))

    # Valid: every rule passes. Weight records, dynamic activation records,
    # static int8 and 16-bit ones, output records and the affine K cache are
    # computed; the rest is refused as "unimplemented".
    add("valid A3 activations with weight sums", ACCEPTED, V_ACT)
    add("valid A7 symmetric activations on int4 groups", ACCEPTED, V_SYM)
    add("valid dynamic 16-bit activation record", "unimplemented",
        replace(V_A16, act=replace(A16, granularity=eb.GRANULARITY_ROW,
                                   chooser=eb.CHOOSER_SYM_DIV, eps=FLT_MIN,
                                   sym_div=32767.0, scale=0.0)))
    add("valid int4 weight zero points", ACCEPTED, V_WZP_I4)
    add("valid GGUF storage bias and fp16 offsets", ACCEPTED, V_GGUF_I4)
    add("valid embedding float zero points", ACCEPTED, V_EMBED_FZP)
    add("valid output fake-quant", ACCEPTED, V_OUT)
    add("valid static affine activations", ACCEPTED, V_STATIC)
    add("valid a16w8 static int16 activations", ACCEPTED, V_A16)
    add("valid affine kcache q@kT", ACCEPTED, V_KCACHE)

    # Records and flags.
    add("record shorter than its type", "record-size",
        replace(V_ACT, record_len_delta=4))
    add("record without its flag", "record-without-flag",
        replace(V_OUT, clear_flags=F_OUT))
    add("flag without its record", "flag-without-record",
        replace(V_OUT, drop_records=(eb.EXT_OUT_QUANT,)))
    add("two records of one type", "duplicate-record",
        replace(V_OUT, duplicate_records=(eb.EXT_OUT_QUANT,)))

    # EXT_ACT_QUANT.
    add("activation record on an embedding", "act-opcode",
        replace(V_EMBED_FZP, flags=F_ACT, act=A3, weight=None))
    add("activation record beside FLAG_STATIC_ACT", "act-scheme-flags",
        replace(V_ACT, flags=F_ACT | F_W | eb.FLAG_STATIC_ACT,
                act_scale=0.01))
    add("activation chooser out of range", "act-field-value",
        replace(V_ACT, act=replace(A3, chooser=7)))
    # The backend's own A0 and XNNPACK's per-token A4: no record carries them,
    # so their chooser and rounding values (3; 0 and 2) are out of range too.
    add("half-away rounding on a record (A0's)", "act-field-value",
        replace(V_SYM, act=replace(A7, rounding=0, eps=0.0, sym_div=127.0,
                                   qmin=-127)))
    add("XNNPACK's per-token chooser on a record (A4's)", "act-field-value",
        replace(V_ACT, act=replace(A3, chooser=3, rounding=2)))
    add("activation codes across the boundary", "act-stage-b",
        replace(V_ACT, act=replace(A3, in_dtype=eb.DTYPE_INT8)))
    add("16-bit activations on a grouped matmul", "act-code-bits",
        replace(V_A16, k=64, k_pad=64, scale_group=32))
    add("activation range below int8", "act-range",
        replace(V_ACT, act=replace(A3, qmin=-129)))
    add("dynamic activations without a chooser", "act-combination",
        replace(V_ACT, act=replace(A3, chooser=eb.CHOOSER_NONE)))
    add("unsigned symmetric activations", "act-unsigned-symmetric",
        Op(eb.OP_ADDMM_I8, flags=F_ACT,
           act=ActQuant(chooser=eb.CHOOSER_SYM_DIV, qmin=0, qmax=127,
                        eps=FLT_MIN, sym_div=63.5)))
    add("static activation scale 0", "act-static-scale",
        replace(V_STATIC, act=replace(STATIC_A, scale=0.0)))
    add("static zero point above qmax", "act-static-zero-point",
        replace(V_STATIC, act=replace(STATIC_A, zero_point=128)))
    add("negative eps", "act-eps",
        replace(V_ACT, act=replace(A3, eps=-1.0)))
    add("symmetric divisor 0", "act-sym-div",
        Op(eb.OP_ADDMM_I8, flags=F_ACT,
           act=ActQuant(chooser=eb.CHOOSER_SYM_DIV, qmin=-127, qmax=127,
                        eps=FLT_MIN, sym_div=0.0)))
    add("divisor set on an affine chooser", "act-unused-field",
        replace(V_ACT, act=replace(A3, sym_div=127.0)))
    add("activation zero point without weight sums",
        "act-zero-point-without-sums",
        replace(V_ACT, flags=F_ACT, weight=None))

    # EXT_WEIGHT_AFFINE.
    add("weight record on an int16 matmul", "weight-opcode",
        Op(eb.OP_ADDMM_I16, flags=F_W,
           weight=WeightAffine(zp_kind=eb.ZP_INT8)))
    add("storage bias 4", "weight-field-value",
        replace(V_GGUF_I4, weight=replace(V_GGUF_I4.weight, storage_bias=4)))
    add("weight record describing a symmetric weight", "weight-empty",
        replace(V_WZP_I4, weight=WeightAffine(), zp_bytes=None))
    add("float zero point on a matmul", "weight-float-zp-matmul",
        Op(eb.OP_ADDMM_I8, flags=F_W,
           weight=WeightAffine(zp_kind=eb.ZP_FP32)))
    add("float zero point with float offsets", "weight-combination",
        replace(V_EMBED_FZP, weight=WeightAffine(zp_kind=eb.ZP_FP32,
                                                 offset_kind=eb.OFFSET_FP32)))
    add("weight sums under backend activations", "weight-sums-unused",
        Op(eb.OP_ADDMM_I8, flags=F_W, weight=SUMS16))
    add("int16 weight sums over 144 asymmetric codes", "weight-sums-width",
        replace(V_ACT, k=144, k_pad=144,
                weight=WeightAffine(zp_kind=eb.ZP_INT8,
                                    sum_kind=eb.SUM_INT16)))
    add("float offsets on a weight staged to int8", "weight-staged-offsets",
        replace(V_GGUF_I4, flags=F_W | eb.FLAG_EXTERNAL_WEIGHTS |
                eb.FLAG_STAGE_I8))
    add("offset set for an absent zero-point table", "weight-unused-field",
        replace(V_GGUF_I4, weight=replace(V_GGUF_I4.weight, zp_off=4)))
    add("zero-point table past the blob", "weight-table-bounds",
        replace(V_WZP_I4, table_offset_delta=1 << 20))
    add("external table with no named data", "weight-table-external",
        replace(V_WZP_I4, flags=F_W | eb.FLAG_EXTERNAL_WEIGHTS))
    add("int4 zero point 9", "weight-zero-point-range",
        replace(V_WZP_I4, zp_bytes=bytes([0x05, 0x09] * 8)))

    # EXT_OUT_QUANT.
    add("output quantize on an embedding", "out-opcode",
        replace(V_EMBED_FZP, flags=F_OUT, weight=None, out=OUT_Q))
    add("output quantize after a fused add", "out-fused-operand",
        replace(V_OUT, flags=F_OUT | eb.FLAG_ADD, res_arg=RES))
    add("output rounding out of range", "out-field-value",
        replace(V_OUT, out=replace(OUT_Q, rounding=9)))
    add("output codes across the boundary", "out-stage-b",
        replace(V_OUT, out=replace(OUT_Q, out_dtype=eb.DTYPE_INT8)))
    add("output rounding half away", "out-field-value",
        replace(V_OUT, out=replace(OUT_Q, rounding=0)))
    add("output range empty", "out-range",
        replace(V_OUT, out=replace(OUT_Q, qmin=7, qmax=7, zero_point=7)))
    add("output scale infinite", "out-scale",
        replace(V_OUT, out=replace(OUT_Q, scale=float("inf"))))
    add("output zero point outside the range", "out-zero-point",
        replace(V_OUT, out=replace(OUT_Q, zero_point=300)))

    # Exactness bounds. The descriptor alone: the bound's rule
    # fires before validate_blob would look for weights.
    k = dot_length_bound(255, 128) + 32               # 65,792 + 32
    add(f"affine x int8 dot of {k}", "dot-bound",
        replace(V_ACT, m=1, n=1, k=k, k_pad=k, no_weights=True,
                weight=WeightAffine(sum_kind=eb.SUM_INT32)))
    k = dot_length_bound(255, 255) + 32               # 33,024 + 32
    add(f"affine x zero-point int8 dot of {k}", "dot-bound",
        replace(V_ACT, m=1, n=1, k=k, k_pad=k, no_weights=True,
                weight=WeightAffine(zp_kind=eb.ZP_INT8,
                                    sum_kind=eb.SUM_INT32),
                zp_bytes=b"\x05"))
    # Weight zero points: int8 zero points under the backend's +-127 rows.
    k = dot_length_bound(127, 255) + 32               # 66,304 + 32
    add(f"int8 zero points x backend rows, dot of {k}", "dot-bound",
        Op(eb.OP_ADDMM_I8, m=1, n=1, k=k, k_pad=k, flags=F_W,
           no_weights=True, weight=WeightAffine(zp_kind=eb.ZP_INT8),
           zp_bytes=b"\x05"))
    # 16-bit codes against int8 weights: the 40-bit ACCX's bound.
    k = dot_length_bound(32768, 128, ACCX_MAX) + 32   # 131,040 + 32
    add(f"16-bit activations x int8 dot of {k}", "dot-bound",
        replace(V_A16, m=1, n=1, k=k, k_pad=k, no_weights=True))
    k = _pad((1 << 24) // 255 + 1, 32)                # 255 * k past 2^24
    add(f"int4 float offsets over {k}", "offset-bound",
        Op(eb.OP_ADDMM_I4, m=1, n=1, k=k, k_pad=k, flags=F_ACT | F_W,
           act=A3, no_weights=True,
           weight=WeightAffine(zp_kind=eb.ZP_INT8, offset_kind=eb.OFFSET_FP32,
                               sum_kind=eb.SUM_INT32),
           zp_bytes=b"\0"))

    # OP_BMM_KCACHE_AFFINE_I8.
    add("affine kcache zero points in the scales' arg", "kcache-args",
        replace(V_KCACHE, zps_arg=SCALES))
    add("affine kcache with batch 0", "kcache-descriptor",
        replace(V_KCACHE, scale_group=0))
    add("affine kcache with a fused clamp", "kcache-descriptor",
        replace(V_KCACHE, flags=eb.FLAG_CLAMP))
    k = dot_length_bound(127, 255) + 32               # 66,304 + 32
    add(f"affine kcache rows of {k}", "kcache-dot-bound",
        replace(V_KCACHE, k=k, k_pad=k))
    return c


CASES = _cases()


# --- the carrier --------------------------------------------------------------------

CARRIER_FEATURES = 64


def export_carrier(path):
    """A 64x64 Linear through the backend, inline int8 weights: one delegate
    whose payload (about 4.7 KB) holds any case. Imports torch."""
    import torch
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import lower_esp32

    torch.manual_seed(0)
    model = torch.nn.Linear(CARRIER_FEATURES, CARRIER_FEATURES).eval()
    x = torch.linspace(-1.0, 1.0, CARRIER_FEATURES).reshape(1, -1)
    ep = torch.export.export(model, (x,))
    lowered = lower_esp32(ep, [Esp32Partitioner(format_policy="i8")])
    et = lowered.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16))
    with open(path, "wb") as f:
        f.write(et.buffer)
    return et.buffer


class _FlatBuffer:
    """Just enough flatbuffer reading to find a program's delegate payload."""

    def __init__(self, data):
        self.data = data

    def u32(self, at):
        return struct.unpack_from("<I", self.data, at)[0]

    def field(self, table, index):
        vtable = table - struct.unpack_from("<i", self.data, table)[0]
        slot = 4 + 2 * index
        if slot >= struct.unpack_from("<H", self.data, vtable)[0]:
            return None
        offset = struct.unpack_from("<H", self.data, vtable + slot)[0]
        return table + offset if offset else None

    def tables(self, table, index):
        at = self.field(table, index)
        if at is None:
            return []
        start = at + self.u32(at)
        return [start + 4 + 4 * i + self.u32(start + 4 + 4 * i)
                for i in range(self.u32(start))]


def payload_span(pte: bytes):
    """(offset, length) of the program's one Esp32Backend payload: inline
    delegate data (Program field 3) or a segment (field 4)."""
    fb = _FlatBuffer(pte)
    program = fb.u32(0)
    spans = []
    for entry in fb.tables(program, 3):           # BackendDelegateInlineData
        at = fb.field(entry, 0)
        if at is not None:
            start = at + fb.u32(at)
            spans.append((start + 4, fb.u32(start)))
    base = struct.unpack_from("<Q", pte, 24)[0] if pte[8:12] == b"eh00" else 0
    for segment in fb.tables(program, 4):         # DataSegment {offset, size}
        # A flatbuffer omits a scalar equal to its default (0): the first
        # segment's offset is usually absent.
        values = []
        for index in (0, 1):
            at = fb.field(segment, index)
            values.append(0 if at is None else
                          struct.unpack_from("<Q", pte, at)[0])
        spans.append((base + values[0], values[1]))
    magic = struct.pack("<I", eb.MAGIC)
    # An empty segment may start where the payload does.
    found = [(o, n) for o, n in spans
             if n >= eb.BLOB_HEADER_BYTES and pte[o:o + 4] == magic]
    if len(found) != 1:
        raise ValueError(f"expected one Esp32Backend payload, found {found}")
    return found[0]


def patched(carrier: bytes, span, blob: bytes) -> bytes:
    """The carrier with its payload replaced by `blob` (then zeros)."""
    offset, length = span
    if len(blob) > length:
        raise ValueError(f"a {len(blob)} B case does not fit the carrier's "
                         f"{length} B payload")
    data = bytearray(carrier)
    data[offset:offset + length] = blob + bytes(length - len(blob))
    return bytes(data)


def write_cases(path, carrier: bytes, cases=None):
    """The device's case file (layout in the module docstring)."""
    cases = CASES if cases is None else cases
    offset, length = payload_span(carrier)
    parts = [CASES_MAGIC, struct.pack("<IIII", CASES_VERSION, offset, length,
                                      len(cases))]
    for case in cases:
        if len(case.blob) > length:
            raise ValueError(f"case {case.name!r} ({len(case.blob)} B) does "
                             f"not fit the {length} B carrier payload")
        name, rule = case.name.encode(), case.rule.encode()
        record = struct.pack("<HHI", len(name), len(rule), len(case.blob)) + \
            name + rule + case.blob
        parts.append(record + bytes(-len(record) % 4))
    data = b"".join(parts)
    with open(path, "wb") as f:
        f.write(data)
    return data


def covered_rules(cases=None):
    """The runtime rules the refusal cases cover (ACCEPTED is no rule)."""
    return sorted({case.rule for case in (CASES if cases is None else cases)}
                  - {ACCEPTED})


def accepted_cases(cases=None):
    """The valid cases this runtime computes: each must load."""
    return [case for case in (CASES if cases is None else cases)
            if case.rule == ACCEPTED]


if __name__ == "__main__":
    missing = set(RULES) - set(covered_rules())
    print(f"{len(CASES)} cases over {len(covered_rules())} rules; "
          f"largest blob {max(len(c.blob) for c in CASES)} B"
          + (f"; MISSING {sorted(missing)}" if missing else ""))
    for case in CASES:
        print(f"  {case.rule:28s} {len(case.blob):6d} B  {case.name}")
    raise SystemExit(1 if missing else 0)
