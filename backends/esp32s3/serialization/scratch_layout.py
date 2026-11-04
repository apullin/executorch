# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A delegate's per-call scratch, read from its blob as the runtime lays it out.

Esp32Backend::init (runtime/esp32_backend.cpp) gives every execute() call
one block of working memory, `scratch_bytes`, bump-allocated in four
reservations, each piece aligned to 16 bytes (ScratchLayout):

  row block  xq, one block of quantised activation rows (or a float loop's
             patch and widened weight rows, or the static int8 conv path's
             row block and code plane), conv_stage, a block of conv output
             before its NCHW scatter, whose bytes the split worker also takes
             for its part of a conv on that path (its own row block or
             channel plane), sxv, the block's row scales,
             wu and wu2, each core's int4 unpack row, and xq16, the int16
             rung's row: reserve_row_block_scratch;
  chain      one slot per chain sentinel the blob writes, each as wide as
             the widest output written to it: reserve_chain_scratch;
  q@kT       one batch of quantised keys and their scales:
             reserve_bmm_scratch;
  affine     the asymmetric matmuls' per-row terms: reserve_affine_scratch.

`layout(blob)` repeats that arithmetic from the same descriptors and
extension records the runtime reads, so preprocess can declare the total to
ExecuTorch's memory planner (DelegateScratchSpec, with blob_writer's
SCRATCH_SLACK for the alignment the runtime makes itself) and the runtime can
take the planned bytes instead of its temp allocator. Each function here names
its C++ twin. test_scratch_layout.py holds the two to each other through the
host runner's init report ("init: N op(s), B blob, T B temp"), and a runtime
handed fewer bytes than its own sum falls back to the temp allocator and
counts the call, so a drift is reported rather than overrun.
"""

import struct
from collections import namedtuple
from dataclasses import dataclass, field

from executorch.backends.esp32s3.operator_support.conv import (
    ConvGeom,
    EXT_CONV3D,
    runtime_block,
)
from executorch.backends.esp32s3.serialization.espressif_blob import (
    ACT_QUANT_FIELDS,
    ACT_QUANT_FORMAT,
    ARG_SCRATCH0,
    BLOB_HEADER_BYTES,
    BLOB_HEADER_FORMAT,
    DTYPE_INT8,
    EXT_ACT_QUANT,
    EXT_COUNT_BYTES,
    EXT_COUNT_FORMAT,
    EXT_OPTIONAL,
    EXT_OUT_QUANT,
    EXT_RECORD_HEADER_BYTES,
    EXT_RECORD_HEADER_FORMAT,
    EXT_SELECTED_ROWS,
    EXT_WEIGHT_AFFINE,
    FLAG_ADD,
    FLAG_MUL,
    FLAG_STAGE_I8,
    GRANULARITY_ROW,
    MAGIC,
    OFFSET_NONE,
    OP_ADDMM,
    OP_ADDMM_I8,
    OP_ADDMM_BF16,
    OP_ADDMM_F16,
    OP_ADDMM_I16,
    OP_ADDMM_I4,
    OP_BMM_DYN_I8,
    OP_BMM_KCACHE_AFFINE_I8,
    OP_BMM_KCACHE_I8,
    OP_DESC_BYTES,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
    OP_EMBED_BF16,
    OP_EMBED_F16,
    OP_EMBED_I16,
    OP_EMBED_I4,
    OP_EMBED_I8,
    OUT_QUANT_FIELDS,
    OUT_QUANT_FORMAT,
    VEC_BYTES,
    WEIGHT_AFFINE_FIELDS,
    WEIGHT_AFFINE_FORMAT,
    ZP_NONE,
)
from executorch.backends.esp32s3.serialization import espressif_blob as _blob

# esp32_float.h kWidenBytes: the budget for one block of a 2-byte weight's
# rows widened to fp32.
WIDEN_BYTES = 16 * 1024
# ScratchLayout's limit: offsets are uint32 on the target.
_LIMIT = 0xFFFFFFFF

_EMBED_OPS = (OP_EMBED_I8, OP_EMBED_I4, OP_EMBED_I16, OP_EMBED_F16, OP_EMBED_BF16)
_BMM_OPS = (OP_BMM_DYN_I8, OP_BMM_KCACHE_I8, OP_BMM_KCACHE_AFFINE_I8)
_FLOAT_MATMUL_OPS = (OP_ADDMM_F16, OP_ADDMM_BF16)

_Desc = namedtuple("_Desc", OP_DESC_FIELDS)


@dataclass
class _Op:
    """One descriptor and the records init attaches to it."""

    desc: _Desc
    conv: ConvGeom | None = None  # EXT_CONV3D
    selected: bool = False        # EXT_SELECTED_ROWS
    act: dict | None = None       # EXT_ACT_QUANT
    weight: dict | None = None    # EXT_WEIGHT_AFFINE
    out: dict | None = None       # EXT_OUT_QUANT


@dataclass
class ScratchLayout:
    """The runtime's ScratchLayout: pieces handed out in order, each at the
    next 16-byte boundary. `pieces` holds (name, offset, bytes) per piece."""

    bytes: int = 0
    pieces: list = field(default_factory=list)

    def reserve(self, name, need):
        """ScratchLayout::reserve: nothing for 0 bytes, else the piece's
        offset. Past the 32-bit limit the runtime refuses the blob at init."""
        if need == 0:
            return None
        at = (self.bytes + VEC_BYTES - 1) & ~(VEC_BYTES - 1)
        if at + need > _LIMIT:
            raise ValueError(
                f"delegate scratch {name} would end at {at + need:,} B, past the "
                f"runtime's 32-bit arena (init refuses this blob)")
        self.bytes = at + need
        self.pieces.append((name, at, need))
        return at


def read_ops(blob):
    """The blob's descriptors with their conv, selected-row and asymmetric
    records attached, as parse_extension_records and esp32_asym::attach
    attach them."""
    blob = bytes(blob)
    magic, _, n_ops, ext_off = struct.unpack_from(BLOB_HEADER_FORMAT, blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a delegate blob: magic {magic:#x}")
    ops = [_Op(_Desc(*struct.unpack_from(OP_DESC_FORMAT, blob,
                                         BLOB_HEADER_BYTES + i * OP_DESC_BYTES)))
           for i in range(n_ops)]
    if ext_off:
        (count,) = struct.unpack_from(EXT_COUNT_FORMAT, blob, ext_off)
        at = ext_off + EXT_COUNT_BYTES
        for _ in range(count):
            index, kind, length = struct.unpack_from(EXT_RECORD_HEADER_FORMAT, blob, at)
            start = at + EXT_RECORD_HEADER_BYTES
            payload = blob[start:start + length]
            at = start + (length + 3) // 4 * 4
            op = ops[index]
            kind &= ~EXT_OPTIONAL & 0xFFFF
            if kind == EXT_CONV3D:
                op.conv = ConvGeom.unpack(payload)
            elif kind == EXT_SELECTED_ROWS:
                op.selected = True
            elif kind == EXT_ACT_QUANT:
                op.act = dict(zip(ACT_QUANT_FIELDS, struct.unpack(ACT_QUANT_FORMAT, payload)))
            elif kind == EXT_WEIGHT_AFFINE:
                op.weight = dict(zip(WEIGHT_AFFINE_FIELDS,
                                     struct.unpack(WEIGHT_AFFINE_FORMAT, payload)))
            elif kind == EXT_OUT_QUANT:
                op.out = dict(zip(OUT_QUANT_FIELDS, struct.unpack(OUT_QUANT_FORMAT, payload)))
    return ops


# ---- the runtime's predicates (esp32_backend.cpp, esp32_asym.h, esp32_float.h)


def _is_embed(opcode):
    return opcode in _EMBED_OPS


def _is_bmm(opcode):
    return opcode in _BMM_OPS


def _executes_i4(d):
    """executes_i4: an int4 matmul not expanded to int8 at load."""
    return d.opcode == OP_ADDMM_I4 and not d.flags & FLAG_STAGE_I8


def _is_depthwise(g):
    return g.groups > 1 and g.c_in == 1 and g.c_out == g.groups and g.c_out_total == g.c_out


def _output_elems(op):
    """op_output_elems: a conv writes its whole NCHW output, a gather m rows
    of k, a matmul m rows of n."""
    d = op.desc
    if op.conv is not None:
        return op.conv.c_out_total * d.m
    if _is_embed(d.opcode):
        return d.m * d.k
    return d.m * d.n


def _codes_out(op):
    """esp32_asym::codes_out: Stage B's int8 codes written."""
    return op.out is not None and op.out["out_dtype"] == DTYPE_INT8


def _affine_op(op):
    """esp32_asym::affine_op: a weight, activation or output record."""
    return op.weight is not None or op.act is not None or op.out is not None


def _a16_op(op):
    """esp32_asym::a16_op: a 16-bit activation record."""
    return op.act is not None and op.act["code_bits"] == 16


def _widen_block(m, k, n):
    """esp32_float::widen_block."""
    row = k * 4
    if m < 2 or row == 0 or row > WIDEN_BYTES:
        return 0
    return min(WIDEN_BYTES // row, n)


def _float_scratch(opcode, conv, m, k, n):
    """esp32_float::scratch_bytes: a conv's patch (k words), then a 2-byte
    weight's widened block."""
    words = k if conv else 0
    block = 0 if opcode == OP_ADDMM else _widen_block(m, k, n) * k
    return (words + block) * 4


# ---- the static int8 convolution path (runtime/esp32_conv.h)
# esp32_backend.cpp's conv_fast_op / conv_fast_rows / conv_fast_scratch and
# esp32_conv::scratch_need: the xq region also holds the path's row block and
# its code plane. Declared whether or not the target firmware builds the path
# (ESP32_CONV_FAST): a runtime without it needs less, never more.

# kActFlags: the fused activations, at most one per op.
_ACT_FLAGS = (_blob.FLAG_ACT_SILU | _blob.FLAG_ACT_GELU | _blob.FLAG_SOFTCAP |
              _blob.FLAG_ACT_SIGMOID | _blob.FLAG_ACT_GELU_TANH |
              _blob.FLAG_ACT_HARDSWISH | _blob.FLAG_ACT_HARDSIGMOID |
              _blob.FLAG_ACT_LEAKY | _blob.FLAG_ACT_RELU2)
CONV_FAST_BLOCK_BYTES = 16 * 1024   # kConvFastBlockBytes
CONV_FAST_BLOCK_MAX = 64            # kConvFastBlockMax
CONV_FAST_PLANE_BUDGET = 32 * 1024  # esp32_conv::kPlaneBudget


def _conv_fast_op(op):
    """conv_fast_op: an int8 2-D conv (or a 3-D one whose depth axis is one
    plane, one tap and no padding) with an activation record whose codes
    depend on each element alone, one scale group, no weight zero points or
    offsets, nothing after the accumulate but a clamp."""
    d, g = op.desc, op.conv
    if (g is None or g.transposed or g.d_in != 1 or g.d_out != 1 or g.kd != 1
            or g.pad_d != 0):
        return False
    if (d.opcode != OP_ADDMM_I8 or d.flags & FLAG_STAGE_I8
            or d.flags & (_ACT_FLAGS | FLAG_ADD | FLAG_MUL)
            or d.scale_group not in (0, d.k_pad)):
        return False
    dw = _is_depthwise(g)
    if (d.m != g.h_out * g.w_out or d.k != (1 if dw else g.c_in) * g.kh * g.kw
            or d.k > d.k_pad):
        return False
    a = op.act
    if a is None or a["code_bits"] != 8 or a["granularity"] == GRANULARITY_ROW:
        return False
    w = op.weight
    return w is None or (w["zp_kind"] == ZP_NONE and w["offset_kind"] == OFFSET_NONE
                         and w["storage_bias"] == 0)


def _read_extent(out, ksize, stride, dil):
    return (out - 1) * stride + (ksize - 1) * dil + 1


def _align16(n):
    return (n + VEC_BYTES - 1) // VEC_BYTES * VEC_BYTES


# esp32_conv.cpp's PIE depthwise path: kPieRun output pixels a run,
# kPieStripBytes of padded channels-last rows, sizeof(Channel) (the same on the
# chip and the host: three 4-byte fields, then two int64 and two 4-byte ones).
PIE_RUN = 32
PIE_STRIP_BYTES = 8 * 1024
PIE_CHANNEL_BYTES = 40


def _pie_dw(op):
    """pie_dw_geometry: a 3x3 depthwise conv, dilation 1 along w, at least
    16 channels (it runs them 16 a vector)."""
    g = op.conv
    return g.kh == 3 and g.kw == 3 and g.dil_w == 1 and op.desc.n >= 16


def _pie_dw_need(op):
    """pie_dw_scratch: the strip (as many padded rows as its budget holds, at
    least the 2 * dil_h + 1 one output row reads, at most the padded plane's),
    the 9 x 16 packed weights, kPieRun QACC images, 16 runs of raw sums and
    16 channels' terms."""
    g = op.conv
    hp = _read_extent(g.h_out, g.kh, g.stride_h, g.dil_h)
    wp = _read_extent(g.w_out, g.kw, g.stride_w, g.dil_w)
    rows = min(max(PIE_STRIP_BYTES // (wp * 16), 2 * g.dil_h + 1), hp)
    return (_align16(rows * wp * 16) + 144 + PIE_RUN * 64 + 16 * PIE_RUN * 4
            + _align16(16 * PIE_CHANNEL_BYTES))


def _conv_fast_need(op):
    """conv_fast_scratch: the path's xq bytes for one op (0 off the path)."""
    if not _conv_fast_op(op):
        return 0
    d, g = op.desc, op.conv
    fp32_in = op.act["in_dtype"] != DTYPE_INT8
    hp = _read_extent(g.h_out, g.kh, g.stride_h, g.dil_h)
    wp = _read_extent(g.w_out, g.kw, g.stride_w, g.dil_w)
    if _is_depthwise(g):
        # esp32_conv::scratch_need: the padded channel plane (none when Stage B
        # codes are read in place), and the PIE path's scratch where it runs.
        interior = g.pad_h == 0 and g.pad_w == 0 and hp <= g.h_in and wp <= g.w_in
        need = 0 if (not fp32_in and interior) else hp * wp
        return max(need, _pie_dw_need(op)) if _pie_dw(op) else need
    rows = max(1, min(CONV_FAST_BLOCK_BYTES // d.k_pad, CONV_FAST_BLOCK_MAX))
    need = min(rows, d.m) * d.k_pad
    pointwise = g.kh == 1 and g.kw == 1 and g.pad_h == 0 and g.pad_w == 0
    plane = g.c_in * g.h_in * g.w_in
    if fp32_in and not pointwise and plane <= CONV_FAST_PLANE_BUDGET:
        need = (need + VEC_BYTES - 1) // VEC_BYTES * VEC_BYTES + plane
    return need


# ---- the four reservations, in init's order


# esp32_backend.cpp kConvSplit: whether a conv on the path splits across the
# cores under min-latency. Off: on the board a split conv gave wrong outputs
# and cache-error panics (the backend README's Known limits), so the runtime
# reserves no scratch for a worker's part and neither does this mirror.
CONV_SPLIT = False
# esp32_backend.cpp kConvSplitMinOutputs: the fewest outputs a conv on the
# path would split across the cores at (min-latency).
CONV_SPLIT_MIN_OUTPUTS = 1024


def _conv_split_need(op):
    """conv_split_scratch: the split worker's own scratch for its part of
    one op (0 off the path, when it cannot split, or needs none; always 0
    while CONV_SPLIT is off). The runtime would reserve it only for a handle
    that can split from init (min_latency in its compile spec, or the runtime
    option on at init): _reserve_row_block's min_latency."""
    if not CONV_SPLIT or not _conv_fast_op(op):
        return 0
    d, g = op.desc, op.conv
    dw = _is_depthwise(g)
    units = d.n if dw else d.m
    if units < 2 or d.n * d.m < CONV_SPLIT_MIN_OUTPUTS:
        return 0
    if dw:
        # esp32_conv::part_scratch_need: a depthwise part's padded plane.
        return _conv_fast_need(op)
    rows = max(1, min(CONV_FAST_BLOCK_BYTES // d.k_pad, CONV_FAST_BLOCK_MAX))
    return min(rows, d.m) * d.k_pad


def _reserve_row_block(layout, ops, min_latency=False):
    """reserve_row_block_scratch, for a handle that can split from init or
    not (`min_latency`)."""
    max_kpad, wants_i4, wants_i16 = 0, False, False
    for op in ops:
        d = op.desc
        wants_i4 |= _executes_i4(d)
        wants_i16 |= d.opcode == OP_ADDMM_I16
        if not _is_embed(d.opcode) and d.k_pad > max_kpad:
            max_kpad = d.k_pad
    if max_kpad == 0:
        return
    block_rows, xq, conv_stage = 1, max_kpad, 0
    for op in ops:
        d, cg = op.desc, op.conv
        if _is_embed(d.opcode) or _is_bmm(d.opcode):
            continue
        if (cg is not None and d.opcode == OP_ADDMM) or d.opcode in _FLOAT_MATMUL_OPS:
            xq = max(xq, _float_scratch(d.opcode, cg is not None, d.m, d.k, d.n))
            continue
        b = runtime_block(d.m, d.k_pad, d.n, conv=cg is not None)
        block_rows = max(block_rows, b)
        need = b * d.k_pad
        if cg is not None and _is_depthwise(cg):
            # One row of codes, then one position's tap offsets.
            taps = (d.k_pad + VEC_BYTES - 1) // VEC_BYTES * VEC_BYTES + d.k * 4
            need = max(need, taps)
        need = max(need, _conv_fast_need(op))
        xq = max(xq, need)
        if cg is not None and not _is_depthwise(cg):
            conv_stage = max(conv_stage, b * d.n * 4)
    layout.reserve("xq", xq)
    # The split worker's region takes the conv staging buffer's bytes (the
    # general loops stage there and the path never does): one region, in the
    # buffer's place after xq, as large as either needs.
    split = max((_conv_split_need(op) for op in ops), default=0) if min_latency else 0
    if max(conv_stage, split) > 0:
        layout.reserve("conv_stage", max(conv_stage, split))
    if block_rows > 1 or any(op.conv is not None for op in ops):
        layout.reserve("sxv", 4 * block_rows)
    if wants_i4:
        layout.reserve("wu", max_kpad)
        layout.reserve("wu2", max_kpad)
    if wants_i16:
        layout.reserve("xq16", max_kpad * 2)


def _reserve_chain(layout, ops):
    """reserve_chain_scratch: the slots the blob writes, each the widest
    output written to it, in bytes (Stage B codes take one byte each)."""
    need, used, slot_bytes = 0, 0, {}
    for op in ops:
        d = op.desc
        if d.out_arg < ARG_SCRATCH0:
            continue
        s = d.out_arg - ARG_SCRATCH0
        used = max(used, s + 1)
        e = _output_elems(op)
        need = max(need, e)
        slot_bytes[s] = max(slot_bytes.get(s, 0), e * (1 if _codes_out(op) else 4))
    if need > 0:
        for s in range(used):
            layout.reserve(f"chain{s}", slot_bytes.get(s, 0))


def _reserve_bmm(layout, ops):
    """reserve_bmm_scratch: a dynamic q@kT's keys and scales, and a cached-K
    one's ragged rows (k < k_pad) staged zero-padded."""
    kq_need = sk_need = 0
    for op in ops:
        d = op.desc
        if not _is_bmm(d.opcode) or (d.opcode != OP_BMM_DYN_I8 and d.k == d.k_pad):
            continue
        kq_need = max(kq_need, d.n * d.k_pad)
        if d.opcode == OP_BMM_DYN_I8:
            sk_need = max(sk_need, d.n)
    if kq_need > 0:
        layout.reserve("bmm_kq", kq_need)
        layout.reserve("bmm_sk", sk_need * 4)


def _reserve_affine(layout, ops):
    """reserve_affine_scratch: X_g (and T_g with float offsets) for every row
    of the largest block an affine op runs, then each row's zero point; the
    int16 loop's one row of codes instead."""
    need = 0
    for op in ops:
        d, cg = op.desc, op.conv
        if _is_embed(d.opcode) or not _affine_op(op):
            continue
        one_row = op.selected or (cg is not None and _is_depthwise(cg))
        rows = 1 if one_row else runtime_block(d.m, d.k_pad, d.n, conv=cg is not None)
        gsize = d.scale_group or d.k_pad
        ngw = -(-d.k_pad // gsize) if gsize else 0
        offsets = op.weight is not None and op.weight["offset_kind"] != OFFSET_NONE
        nbytes = rows * ngw * (4 + (4 if offsets else 0)) + rows * 4
        if _a16_op(op):
            nbytes = d.k_pad * 2
        need = max(need, nbytes)
    if need > 0:
        layout.reserve("affine_terms", need)


def layout(blob, min_latency=False):
    """The ScratchLayout Esp32Backend::init builds for this blob, for a
    handle that can split across the cores from init or not (`min_latency`:
    its compile spec's min_latency; the runtime also counts the option set at
    init, which preprocess cannot know: such a handle needs more than was
    declared and falls back to the temp allocator)."""
    ops = read_ops(blob)
    out = ScratchLayout()
    _reserve_row_block(out, ops, min_latency)
    _reserve_chain(out, ops)
    _reserve_bmm(out, ops)
    _reserve_affine(out, ops)
    return out


def scratch_bytes(blob, min_latency=False):
    """The bytes one execute() call of this blob's delegate works in: the
    runtime's CompiledBlob::scratch_bytes (`min_latency` as layout's)."""
    return layout(blob, min_latency).bytes
