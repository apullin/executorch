# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The delegate blob's wire format; runtime/esp32_backend.cpp is the reader's contract."""

import hashlib
import struct

from executorch.backends.esp32s3.target.chips import ESP32S3 as CHIP


BACKEND_NAME = "Esp32Backend"


MAGIC = 0x45333244  # 'E32D'

# The serialized ABI is deliberately named in one place.  Keep field order here
# beside the format: test_blob_abi.py compares both against the C++ declaration,
# so changing one half of the AOT/runtime contract fails before a model runs.
BLOB_HEADER_FORMAT = "<IIII"
EXT_COUNT_FORMAT = "<I"
EXT_RECORD_HEADER_FORMAT = "<HHI"
OP_DESC_FORMAT = "<IIIIIIIIIIIIffIf"
OP_DESC_FIELDS = (
    "opcode", "flags", "m", "k", "n", "k_pad", "bias_off", "wt_off",
    "scale_off", "in_arg", "out_arg", "scale_group", "clamp_lo",
    "clamp_hi", "res_arg", "act_scale",
)
BLOB_HEADER_BYTES = struct.calcsize(BLOB_HEADER_FORMAT)
EXT_COUNT_BYTES = struct.calcsize(EXT_COUNT_FORMAT)
EXT_RECORD_HEADER_BYTES = struct.calcsize(EXT_RECORD_HEADER_FORMAT)
OP_DESC_BYTES = struct.calcsize(OP_DESC_FORMAT)

assert BLOB_HEADER_BYTES == 16
assert EXT_COUNT_BYTES == 4
assert EXT_RECORD_HEADER_BYTES == 8
assert OP_DESC_BYTES == 64
# The format version. The runtime accepts exactly this version and refuses any
# other at init. Beyond the header and the 64-byte OpDesc it covers:
# - the extension table at ext_off: [count u32], then per record
#   [op_index u16][type u16][len u32][payload, padded to 4]; an unknown type
#   with the EXT_OPTIONAL bit is skipped, an unknown required one refused;
# - k_pad in elements (its byte count set by the opcode's weight format) and
#   scale_group as a group size in elements (0: per row);
# - in_arg/out_arg sentinels 0xFFFFFFFD to 0xFFFFFFFF naming the delegate's
#   three scratch slots, and bias_off 0 for a bias-free projection;
# - the epilogue flags FLAG_CLAMP, the FLAG_ACT_* activations, FLAG_ADD,
#   FLAG_MUL and FLAG_SOFTCAP (clamp_lo/clamp_hi are a tagged union under
#   FLAG_SOFTCAP, FLAG_ACT_LEAKY and the hard activations), and
#   FLAG_STATIC_ACT with act_scale;
# - FLAG_STAGE_I8 and FLAG_HEAD_SPLIT (a tied head staged in PSRAM as int8, and
#   split across both cores) and EXT_SELECTED_ROWS (a selected-row head);
# - the asymmetric-quantisation pairs, each flag requiring its record and each
#   record its flag: FLAG_ACT_QUANT + EXT_ACT_QUANT, FLAG_WEIGHT_AFFINE +
#   EXT_WEIGHT_AFFINE, FLAG_OUT_QUANT + EXT_OUT_QUANT; and
#   OP_BMM_KCACHE_AFFINE_I8 (runtime/esp32_asym.h).
VERSION = 24


# The fp32 loop: fp32 weights, no scales (inline, or one PTD entry under
# FLAG_EXTERNAL_WEIGHTS: wt_off its key, scale_off 0). A matmul's are [K,N], walked
# k-outer. Under FLAG_CONV (the fp32 convolution) they are [N,K], the torch
# conv weight flattened (output channel rows of the gather's (c, d, i, j)
# patch), k_pad = k: each output element is bias + the patch's dot with its
# row, in k order -- the same sum, step for step, as the matmul's.
OP_ADDMM = 1
OP_ADDMM_I8 = 2
OP_ADDMM_I4 = 3
OP_ADDMM_I16 = 4
# Embedding at each rung. A separate opcode per (op kind, format) is redundant and will
# not scale past a third op kind -- at that point opcode should become the kind and the
# format should move into the descriptor's spare field. Two kinds is not yet worth the
# churn on a runtime that is verified.
OP_EMBED_I8 = 5
OP_EMBED_I4 = 6
OP_EMBED_I16 = 7
# Opt-in int8 q@kT: a bmm with BOTH operands quantised dynamically at execute --
# an op with no weights at all. Descriptor reuse, documented because it is
# reuse: res_arg carries the SECOND INPUT's arg index (a bmm has no residual) and
# scale_group carries the BATCH count (a dynamic bmm has no stored scales). The
# claimed transpose is folded, so the runtime reads the K side in its natural
# [B,N,K] layout -- key vectors contiguous on both sides of the dot, which is
# also the K-cache layout a real board would read.
OP_BMM_DYN_I8 = 8
# Opt-in int8 q@kT against a PERSISTENT int8 K cache (kcache_i8.py): the key
# side arrives already quantised, int8 codes [B,N,K] plus one fp32 scale per
# key row [B,N], so only the query rows are quantised at execute. A key's
# absmax scale depends on nothing but that key, so the scores are bit-identical
# to OP_BMM_DYN_I8 on the keys the codes came from -- the keys were simply
# quantised once, when written, instead of on every step. Descriptor reuse on
# top of OP_BMM_DYN_I8's: res_arg carries the CODES input, scale_off the
# SCALES input's arg index (a runtime tensor, not stored scales), scale_group
# the batch. An opcode rather than a flag because the opcode is what selects
# an operand's format, as across the weight ladder.
OP_BMM_KCACHE_I8 = 9
# q@kT against an asymmetric int8 K cache -- ExecuTorch's stock
# QuantizedKVCache, which stores keys per token with XNNPACK's affine chooser.
# OP_BMM_KCACHE_I8's descriptor reuse plus one more: wt_off carries the zero
# points' ARG index (an int8 tensor, one per key row, like the scales at
# scale_off). Queries stay the delegate's own symmetric rows.
OP_BMM_KCACHE_AFFINE_I8 = 10
# Its exactness bound on k_pad (esp32_asym.cpp kcache-dot-bound): A0 queries
# |q| <= 127 against key codes with a zero point |k - z| <= 255, so an int32
# dot holds floor((2^31 - 1) / (127 * 255)) rounded down to a multiple of 32.
KCACHE_AFFINE_DOT_BOUND = 66_304
# Stored float weights (runtime/esp32_float.h): fp16 and bf16 kept
# as shipped, two bytes an element, and widened to fp32 as the loop reads
# them -- bf16 by a 16-bit shift, fp16 exactly -- so the arithmetic is the
# fp32 loop's on the widened value, bit for bit. A matmul or convolution
# (FLAG_CONV with its EXT_CONV3D, as on any rung) and a gather each; [n, k]
# row-major (inline, or one PTD entry under FLAG_EXTERNAL_WEIGHTS), k_pad = k,
# no scales, scale_off 0. Opcodes rather than a flag, as across the integer
# ladder.
OP_ADDMM_F16 = 11
OP_ADDMM_BF16 = 12
OP_EMBED_F16 = 13
OP_EMBED_BF16 = 14


EXT_OPTIONAL = 0x8000
# Required extension for ``index_select(full_weight, 0, rows)`` feeding a
# Linear. Payload is (runtime rows arg, fixed selected-row count), both uint32.
EXT_SELECTED_ROWS = 7
SELECTED_ROWS_FORMAT = "<II"
SELECTED_ROWS_BYTES = struct.calcsize(SELECTED_ROWS_FORMAT)
assert SELECTED_ROWS_BYTES == 8

# The three asymmetric-quantisation records, all REQUIRED, all fields 4 bytes
# little-endian. Their meaning, field by field, is runtime/esp32_asym.h;
# asym_records.py builds and checks them. Each travels with its flag
# (FLAG_ACT_QUANT etc.).
#
# EXT_ACT_QUANT: how a quantised matmul quantises its activation, replacing
# the backend's per-row absmax -- dynamic per row or per tensor through one of
# three choosers, or static constants.
EXT_ACT_QUANT = 8
ACT_QUANT_FORMAT = "<IIIIIiifffi"
ACT_QUANT_FIELDS = (
    "in_dtype", "code_bits", "granularity", "chooser", "rounding", "qmin",
    "qmax", "eps", "sym_div", "scale", "zero_point",
)
# EXT_WEIGHT_AFFINE: zero points (integer, or float for embeddings), float
# offsets and the weight sums S', one entry per scale group; the storage bias
# says what an unsigned source's codes were shifted by to store them signed.
# The tables are inline (blob offsets) or, under FLAG_EXTERNAL_WEIGHTS, named
# data keys, as wt_off and scale_off are.
EXT_WEIGHT_AFFINE = 9
WEIGHT_AFFINE_FORMAT = "<IIIiIII"
WEIGHT_AFFINE_FIELDS = (
    "zp_kind", "offset_kind", "sum_kind", "storage_bias", "zp_off",
    "offset_off", "sum_off",
)
# EXT_OUT_QUANT: the model's quantize of the op's output, applied after the
# fused epilogue as fake-quant (the dequantised value is what is stored).
EXT_OUT_QUANT = 10
OUT_QUANT_FORMAT = "<IIfiii"
OUT_QUANT_FIELDS = ("out_dtype", "rounding", "scale", "zero_point", "qmin",
                    "qmax")
ACT_QUANT_BYTES = struct.calcsize(ACT_QUANT_FORMAT)
WEIGHT_AFFINE_BYTES = struct.calcsize(WEIGHT_AFFINE_FORMAT)
OUT_QUANT_BYTES = struct.calcsize(OUT_QUANT_FORMAT)
assert (ACT_QUANT_BYTES, WEIGHT_AFFINE_BYTES, OUT_QUANT_BYTES) == (44, 28, 24)
# The records' field values. EXT_ACT_QUANT.in_dtype and EXT_OUT_QUANT.out_dtype:
# what crosses the delegate boundary (codes only in Stage B).
DTYPE_FP32 = 0
DTYPE_INT8 = 1
DTYPE_INT16 = 2
GRANULARITY_ROW = 0
GRANULARITY_TENSOR = 1
GRANULARITY_STATIC = 2
CHOOSER_NONE = 0
CHOOSER_SYM_DIV = 1
CHOOSER_AFFINE_MINMAX = 2
# The one rounding both records take: rne before the zero point (torch's
# quantize ops). 0 and 2 are not used: they would name A0's half-away and A4's
# float64 rounding, which no record carries.
ROUND_RNE_BEFORE_ZP = 1
ZP_NONE = 0
ZP_INT8 = 1
ZP_FP32 = 2
OFFSET_NONE = 0
OFFSET_FP32 = 1
OFFSET_FP16 = 2
SUM_NONE = 0
SUM_INT16 = 1
SUM_INT32 = 2

FLAG_EXTERNAL_WEIGHTS = 1
# Scales at scale_off are IEEE half, not float. Explicit rather than derived from
# n_groups: the two happen to coincide today (grouping is what makes fp16 worth its
# decode cost) but tying the storage type to the group count would break silently the
# first time that stops being true.
FLAG_FP16_SCALES = 2
# Apply clamp(y, lo, hi) between accumulate and store. THE backend-fusion case: it
# runs in registers, so a whole portable-kernel dispatch and the activation tensor's
# round trip through memory disappear -- per claimed clamp-family op, and accidental
# CHAINS (relu into a defensive clamp, relu6 after relu) collapse into one interval.
FLAG_CLAMP = 4
# Fused output activations, applied BEFORE any fused clamp (matching the composition
# order clamp(act(mm)) that produces them). SiLU is the mechanically interesting one:
# it decomposes to sigmoid+mul with the matmul output feeding BOTH -- a diamond, a
# fused pattern where the producer legitimately has two users.
FLAG_ACT_SILU = 8
# GELU, exact ('none') approximate only -- the tanh variant is different numerics and
# has its own flag (FLAG_ACT_GELU_TANH) rather than being silently substituted.
FLAG_ACT_GELU = 16
# Fused residual add: y += args[res_arg], applied AFTER the act/clamp epilogue --
# the composition add(clamp(act(mm)), r). fp32 a+b is symmetric in its operands, so
# fusing changes no bit regardless of which side of the add the matmul sat on. The
# residual must be a partition INPUT: an add whose other operand is itself claimable
# is a join of two chains and is refused at the gate (see _residual_pattern).
FLAG_ADD = 32

# Static activation scales: quantise activations at a scale CALIBRATED ahead of
# time (descriptor act_scale) instead of per-row absmax at execute. What it buys is
# not speed -- absmax is K ops against K*N MACs -- but REPRODUCIBILITY: a decode
# loop's per-row scale shifts with every KV-cache rounding tie, so the same prompt
# can drift run to run; a calibrated scale cannot. The cost is honesty about range:
# an activation exceeding calibration clips, which is the standard static-quant
# trade, made here per-model and opt-in rather than silently.
FLAG_STATIC_ACT = 64
# A convolution, executed as an im2col matmul. A flag rather than an opcode: the
# opcode already selects the weight format, so conv gets i4/i8/i16 for free.
# Always accompanied by an EXT_CONV3D record; the runtime refuses one without
# the other in either direction.
FLAG_CONV = 128
# The activation arrives through a PT2E dynamic per-TENSOR quantise/dequantise
# pair, and the delegate honours that scale instead of its own per-row one.
#
# Absorbing this is not an optimisation dressed up as fidelity; it is both. A
# PT2E-quantised model spells its activation quantisation as
#
#     choose_qparams_symmetric(x) -> quantize_per_tensor -> dequantize_per_tensor
#
# and without this flag all three run as PORTABLE ops -- two full passes over the
# activation to quantise and immediately undo it -- after which the delegate
# quantises the result a THIRD time at its own per-row absmax. So the model would
# pay for a quantisation scheme it does not get.
#
# Per-row is strictly finer than per-tensor, so the substitution would be more
# accurate, not less. It would still be the wrong answer: a model QAT'd against a
# per-tensor scale was trained with that error, and silently giving it a
# different one is the same objection as re-quantising an fp8 checkpoint.
FLAG_ACT_PER_TENSOR = 256
# Terminal Muse Glimmer logit scaling/softcap. The two descriptor clamp floats
# are otherwise unused and carry (output_multiplier, positive cap) under this
# mutually-exclusive flag. Keeping the written operation order is part of the
# contract: y*=scale; y/=cap; y=tanh(y); y*=cap.
FLAG_SOFTCAP = 512
# Expand this external i4 addmm once into int8 PSRAM at method load. This is a
# storage-tier decision, not a quantisation change: every nibble becomes the
# identical signed code the ordinary per-row scratch unpack would produce.
FLAG_STAGE_I8 = 1024
# Split only this addmm's output rows across the two LX7 cores. Kept separate
# from min_latency, which splits every eligible projection, so the second core
# can be borrowed for the tied output head alone.
FLAG_HEAD_SPLIT = 2048
# Fused multiply: y = epilogue(mm) * args[res_arg], applied where a residual
# would be added and never together with one -- the composition
# mul(clamp(act(mm)), R). R is either a partition input (an output gate's
# attention value) or, for a gate*up pair, the op's OWN output: the pair's
# earlier projection writes act(gate) there and the later one multiplies it in
# place, so the [M, F] product never needs a buffer of its own -- nor a
# portable mul.out. fp32 * is commutative, so which factor the eager graph put
# first changes no bit.
FLAG_MUL = 4096
# More fused activations, exclusive with SiLU/GELU and each other. Each repeats
# its portable kernel's operation order in fp32 (fast_math.h supplies expf and
# tanhf). ReLU^2 is relu then square.
FLAG_ACT_SIGMOID = 8192
FLAG_ACT_GELU_TANH = 16384
# torch's hardswish/hardsigmoid decompositions, (x * clamp(x + a, 0, 6)) / d and
# clamp(x + a, 0, 6) / d, with a and d in clamp_lo/clamp_hi. They travel in the
# descriptor because the partitioner cannot read them: torch lifts the 3 and
# the 6 to tensor constants, and partitioning sees FakeTensors. Carrying them
# makes any constants exact, where a fixed 3/6 could only be checked -- and
# refused, fatally -- in preprocess.
FLAG_ACT_HARDSWISH = 32768
FLAG_ACT_HARDSIGMOID = 65536
# leaky_relu. Its slope rides clamp_lo. As under FLAG_SOFTCAP the clamp fields
# are a tagged union, so these never coexist with FLAG_CLAMP. (No gate produces
# act-then-clamp anyway: a clamp is only claimed directly over the matmul.)
FLAG_ACT_LEAKY = 131072
FLAG_ACT_RELU2 = 262144
# The asymmetric-quantisation flags, each requiring its record (EXT_ACT_QUANT,
# EXT_WEIGHT_AFFINE, EXT_OUT_QUANT below) and refused at load without it: an
# activation quantiser other than the backend's per-row absmax, weight zero
# points/offsets/sums, and an output quantize.
FLAG_ACT_QUANT = 524288
FLAG_WEIGHT_AFFINE = 1048576
FLAG_OUT_QUANT = 2097152
# act_fn names -> flag. One table so the gate, the walk and the descriptor
# writer cannot disagree about which activations exist.
ACT_FLAGS = {
    "silu": FLAG_ACT_SILU,
    "gelu": FLAG_ACT_GELU,
    "sigmoid": FLAG_ACT_SIGMOID,
    "gelu_tanh": FLAG_ACT_GELU_TANH,
    "hardswish": FLAG_ACT_HARDSWISH,
    "hardsigmoid": FLAG_ACT_HARDSIGMOID,
    "leaky": FLAG_ACT_LEAKY,
    "relu2": FLAG_ACT_RELU2,
}

# Intra-partition dataflow: in_arg/out_arg values name scratch slots rather than
# boundary tensors.
# An intermediate that is ALSO a partition output needs no slot at all: in-partition
# consumers read it back through its output arg index, since reading args is free.
# THREE scratch slots. The sentinels are the top of the arg space, so adding one
# moves the base and changes what 0xFFFFFFFE means -- a version bump rather than
# a compatible addition. Fusing activations onto convolutions needs three.
NUM_SCRATCH = 3
ARG_SCRATCH0 = 0xFFFFFFFF - (NUM_SCRATCH - 1)

# PIE loads 128 bits at a time and ee.vld.128.ip silently truncates the low address
# bits when misaligned, so rows are padded to this and the blob is aligned to it.
VEC_BYTES = CHIP.vec_bytes

# The longest int8 dot one int32 accumulator holds exactly (esp32_backend.cpp
# kMaxDotLen): activation codes |x| <= 127 against stored weight codes
# |w| <= 128, floor((2^31 - 1) / (127 * 128)) rounded down to a multiple of 32.
# The runtime's init refuses an i8 or i4 matmul or conv, or a q@kT, whose k_pad
# passes it; the partition gate refuses the op first, which leaves it portable.
MAX_DOT_LEN = 132_096


def external_data_key(kind: str, payload: bytes) -> str:
    """Return the runtime key for one content-addressed backend payload.

    Export-side native controls use this too. Keeping the spelling here makes
    it impossible for their PTD manifest to drift from preprocess while still
    validating every emitted key against the final NamedDataStore.
    """
    # w weights, s scales; z zero points, m float offsets and u weight sums.
    if kind not in ("w", "s", "z", "m", "u"):
        raise ValueError(f"external data kind must be one of w/s/z/m/u, got "
                         f"{kind!r}")
    return f"esp32_{kind}_{hashlib.sha256(payload).hexdigest()[:16]}"


# ---- reading blobs back -----------------------------------------------------
# One reader for every tool that looks inside a lowered program: the parity
# firmware's artifact census (test/parity/artifact_census.py) and the
# weight-tier report (utils/weight_tiers.py). It reads a blob the way
# runtime/esp32_backend.cpp does: the header, the 64-byte descriptors, then
# the extension table at ext_off.

# Short names for reports, one per opcode and flag.
OPCODE_NAMES = {
    OP_ADDMM: "f32",
    OP_ADDMM_I8: "i8",
    OP_ADDMM_I4: "i4",
    OP_ADDMM_I16: "i16",
    OP_EMBED_I8: "embed_i8",
    OP_EMBED_I4: "embed_i4",
    OP_EMBED_I16: "embed_i16",
    OP_BMM_DYN_I8: "bmm_dyn_i8",
    OP_BMM_KCACHE_I8: "bmm_kcache_i8",
    OP_BMM_KCACHE_AFFINE_I8: "bmm_kcache_affine_i8",
    OP_ADDMM_F16: "f16",
    OP_ADDMM_BF16: "bf16",
    OP_EMBED_F16: "embed_f16",
    OP_EMBED_BF16: "embed_bf16",
}
FLAG_NAMES = {
    FLAG_EXTERNAL_WEIGHTS: "external",
    FLAG_FP16_SCALES: "fp16_scale",
    FLAG_CLAMP: "clamp",
    FLAG_ACT_SILU: "silu",
    FLAG_ACT_GELU: "gelu",
    FLAG_ADD: "residual",
    FLAG_STATIC_ACT: "static_act",
    FLAG_CONV: "conv",
    FLAG_ACT_PER_TENSOR: "act_per_tensor",
    FLAG_SOFTCAP: "softcap",
    FLAG_STAGE_I8: "stage_i8",
    FLAG_HEAD_SPLIT: "head_split",
    FLAG_MUL: "mul",
    FLAG_ACT_SIGMOID: "sigmoid",
    FLAG_ACT_GELU_TANH: "gelu_tanh",
    FLAG_ACT_HARDSWISH: "hardswish",
    FLAG_ACT_HARDSIGMOID: "hardsigmoid",
    FLAG_ACT_LEAKY: "leaky",
    FLAG_ACT_RELU2: "relu2",
    FLAG_ACT_QUANT: "act_quant",
    FLAG_WEIGHT_AFFINE: "weight_affine",
    FLAG_OUT_QUANT: "out_quant",
}
ALL_FLAGS = 0
for _flag in FLAG_NAMES:
    ALL_FLAGS |= _flag
del _flag
# A blob scan gives up on a header claiming more ops than this.
MAX_SCANNED_OPS = 4096


def read_extensions(data, blob_at: int, ext_off: int) -> dict:
    """{op_index: [(type, payload), ...]} of the blob at `blob_at`, in table
    order, EXT_OPTIONAL stripped from each type. A truncated table yields the
    records before the cut; ext_off 0 means no table."""
    records = {}
    if ext_off == 0:
        return records
    cursor = blob_at + ext_off
    if cursor + EXT_COUNT_BYTES > len(data):
        return records
    count, = struct.unpack_from(EXT_COUNT_FORMAT, data, cursor)
    cursor += EXT_COUNT_BYTES
    for _ in range(count):
        if cursor + EXT_RECORD_HEADER_BYTES > len(data):
            break
        op_index, ext_type, length = struct.unpack_from(
            EXT_RECORD_HEADER_FORMAT, data, cursor)
        cursor += EXT_RECORD_HEADER_BYTES
        if cursor + length > len(data):
            break
        payload = bytes(data[cursor:cursor + length])
        cursor += length + (-length % 4)
        records.setdefault(op_index, []).append(
            (ext_type & ~EXT_OPTIONAL, payload))
    return records


class Blob:
    """One delegate blob read back from `data` at `offset`: its descriptors
    (a dict per op, keyed by OP_DESC_FIELDS) and its extension records."""

    def __init__(self, data, offset: int, descs: tuple, records: dict):
        self.data = data
        self.offset = offset
        self.descs = descs
        self.records = records

    def ext_types(self, i: int) -> tuple:
        return tuple(t for t, _ in self.records.get(i, ()))

    def record(self, i: int, ext_type: int, length: int):
        """Op i's record of this type and payload length, or None."""
        for t, payload in self.records.get(i, ()):
            if t == ext_type and len(payload) == length:
                return payload
        return None

    def geometry(self, i: int):
        """Op i's convolution geometry (EXT_CONV3D), or None."""
        from executorch.backends.esp32s3.operator_support.conv import (
            CONV_GEOM_BYTES,
            ConvGeom,
            EXT_CONV3D,
        )

        payload = self.record(i, EXT_CONV3D, CONV_GEOM_BYTES)
        return ConvGeom.unpack(payload) if payload is not None else None

    def selected(self, i: int):
        """Op i's selected-row record: (runtime rows arg, row count), or None."""
        payload = self.record(i, EXT_SELECTED_ROWS, SELECTED_ROWS_BYTES)
        return (struct.unpack(SELECTED_ROWS_FORMAT, payload)
                if payload is not None else None)

    def act_quant(self, i: int):
        """Op i's EXT_ACT_QUANT record as a dict, or None."""
        payload = self.record(i, EXT_ACT_QUANT, ACT_QUANT_BYTES)
        return (dict(zip(ACT_QUANT_FIELDS, struct.unpack(ACT_QUANT_FORMAT, payload)))
                if payload is not None else None)

    def weight_affine(self, i: int):
        """Op i's EXT_WEIGHT_AFFINE record as a dict, or None."""
        payload = self.record(i, EXT_WEIGHT_AFFINE, WEIGHT_AFFINE_BYTES)
        return (dict(zip(WEIGHT_AFFINE_FIELDS,
                         struct.unpack(WEIGHT_AFFINE_FORMAT, payload)))
                if payload is not None else None)

    def key(self, off: int) -> bytes:
        """The NUL-terminated named-data key at blob offset `off`, without its
        terminator: an external op's wt_off and scale_off name its keys."""
        start = self.offset + off
        end = bytes(self.data[start:start + 4096]).find(b"\0")
        if off == 0 or end < 0:
            raise ValueError(f"no NUL-terminated key at blob offset {off}")
        return bytes(self.data[start:start + end])


def read_blob(data, at: int = 0, known_flags: int = ALL_FLAGS):
    """The blob at `at` in `data`, or None when it is not one this runtime
    would read: another magic or version, no ops (or more than
    MAX_SCANNED_OPS), descriptors past the end, an unknown opcode or a flag
    outside `known_flags`, or a zero m, k or n."""
    if at + BLOB_HEADER_BYTES > len(data):
        return None
    magic, version, n_ops, ext_off = struct.unpack_from(BLOB_HEADER_FORMAT, data, at)
    desc_end = at + BLOB_HEADER_BYTES + n_ops * OP_DESC_BYTES
    if (magic != MAGIC or version != VERSION
            or not 0 < n_ops < MAX_SCANNED_OPS or desc_end > len(data)):
        return None
    descs = []
    for index in range(n_ops):
        values = struct.unpack_from(
            OP_DESC_FORMAT, data, at + BLOB_HEADER_BYTES + index * OP_DESC_BYTES)
        desc = dict(zip(OP_DESC_FIELDS, values))
        if (desc["opcode"] not in OPCODE_NAMES or desc["flags"] & ~known_flags
                or not desc["m"] or not desc["k"] or not desc["n"]):
            return None
        descs.append(desc)
    return Blob(data, at, tuple(descs), read_extensions(data, at, ext_off))


def scan_blobs(data, known_flags: int = ALL_FLAGS):
    """Every blob read_blob accepts at any offset of `data` that holds the
    magic: the delegates of a program or of a packed image of programs, found
    without parsing the container around them."""
    magic_bytes = struct.pack("<I", MAGIC)
    start = 0
    while True:
        at = data.find(magic_bytes, start)
        if at < 0:
            return
        start = at + len(magic_bytes)
        blob = read_blob(data, at, known_flags)
        if blob is not None:
            yield blob
