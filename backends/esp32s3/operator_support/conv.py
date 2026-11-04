# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which convolutions the delegate accepts, and their im2col geometry."""

import math
import struct
from dataclasses import dataclass

from executorch.backends.esp32s3.target.chips import ESP32S3 as CHIP


# Extension record type 6, REQUIRED (high bit clear): a runtime that does not
# understand conv geometry cannot execute a conv op, and running the descriptor
# alone would do a plain matmul over whatever happened to be in the input. A
# different payload is a different record with a different type, never a
# silently different length; types 1 to 5 are not assigned, and the runtime
# refuses them like any unknown REQUIRED record.
EXT_CONV3D = 6          # payload: 24 x uint16, see ConvGeom
CONV_GEOM_FORMAT = "<24H"
CONV_GEOM_BYTES = struct.calcsize(CONV_GEOM_FORMAT)
assert CONV_GEOM_BYTES == 48

# int8 activations in the im2col buffer. The buffer is written once per block and
# read N times, so it wants to be in internal SRAM; this is what bounds the block.
_ACT_BYTES = 1


@dataclass(frozen=True)
class ConvGeom:
    """Everything the gather needs that m/k/n do not already say.

    ONE geometry for 1-D, 2-D and 3-D. The rank is expressed by degenerate
    axes -- a conv2d has d_in=d_out=kd=1, a conv1d additionally h_in=h_out=kh=1 --
    and the gather's loops then run once over those axes and contribute nothing.
    This is the conv1d insight applied one level up: a second (and third) gather
    would be more code to write, verify and keep in step for arithmetic that is
    identical.
    """

    h_in: int
    w_in: int
    c_in: int
    h_out: int
    w_out: int
    c_out: int
    kh: int
    kw: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dil_h: int
    dil_w: int
    d_in: int = 1
    d_out: int = 1
    kd: int = 1
    stride_d: int = 1
    pad_d: int = 0
    dil_d: int = 1
    groups: int = 1
    # Which group this op computes. A grouped conv becomes `groups` ops, each a
    # dense conv over its own channel slice, so the only thing they need beyond
    # the shared geometry is where their slice starts. Depthwise does NOT use
    # this -- it stays one op whose loop walks every channel -- because g ops of
    # one channel each would be C descriptors and C dispatches for a kernel that
    # is already the cheapest thing here.
    group_idx: int = 0
    # The FULL output channel count of the tensor this op writes into, which is
    # not derivable from c_out and groups in every case: a grouped slice has
    # c_out*groups, a depthwise op has c_out (it covers every channel in one
    # op), and guessing between them by inspecting the other fields is the
    # derivation a shape check gets wrong. Stored.
    c_out_total: int = 0
    # A TRANSPOSED convolution, expressed as a dense one over a zero-stuffed
    # input. The equivalence is exact (verified to 0.0), and the zero-stuffing
    # is never materialised: it costs 4.84x input inflation on a 2x2/stride-2
    # up-conv if built, and nothing at all if the gather simply treats
    # off-lattice positions as zero -- which it already does for padding.
    # Output padding needs no field of its own: the output extents above are
    # the padded ones, and a position past the computed extent reads its taps
    # through the same mapping as every other (see `_axis_index`).
    transposed: int = 0

    # `c_in` is ALWAYS PER GROUP, and the total input channels are
    # `c_in * groups`. One meaning for the field in all three cases -- dense
    # (c_in=C, groups=1), depthwise (c_in=1, groups=C), grouped slice
    # (c_in=C/g, groups=g) -- so no consumer has to ask which kind it is before
    # reading it. A meaning that differed by kind would make the runtime's
    # input-size check count depthwise channels C times over.

    @property
    def c_in_total(self) -> int:
        return self.c_in * self.groups

    @property
    def depthwise(self) -> bool:
        """One input channel per output channel: the case where im2col is the
        WRONG transform, not merely a slower one. Measured on a 3x3 128->128
        layer: weights are 1.1 KB against 205 KB of activation (1:178), and
        im2col would materialise a 373 KB buffer to save re-streaming those
        1.1 KB. A dense conv of the same shape is 1:1.4, where the same buffer
        buys back 147 KB per pass."""
        return (self.groups > 1 and self.c_in == 1 and
                self.c_out == self.groups and self.c_out_total == self.c_out)

    @property
    def m(self) -> int:
        return self.d_out * self.h_out * self.w_out

    @property
    def k(self) -> int:
        return self.c_in * self.kd * self.kh * self.kw

    @property
    def n(self) -> int:
        return self.c_out

    def pack(self) -> bytes:
        """The extension payload. uint16 throughout: every field is a spatial
        extent or a channel count, and 65535 of any of them is far past what
        fits on this part -- `admit()` enforces the bound rather than letting a
        silent truncation happen here."""
        return struct.pack(
            CONV_GEOM_FORMAT, self.h_in, self.w_in, self.c_in, self.h_out, self.w_out,
            self.c_out, self.kh, self.kw, self.stride_h, self.stride_w,
            self.pad_h, self.pad_w, self.dil_h, self.dil_w,
            self.d_in, self.d_out, self.kd, self.stride_d, self.pad_d,
            self.dil_d, self.groups, self.group_idx,
            self.c_out_total or self.c_out, self.transposed)

    @classmethod
    def unpack(cls, payload: bytes) -> "ConvGeom":
        return cls(*struct.unpack(CONV_GEOM_FORMAT, payload))


# The runtime's row-block policy, mirrored. `row_block_for()` in
# esp32_backend.cpp sizes one op's block as
#
#     min(m, kRowBlockMax, kRowBlockScratchBytes // (k_pad + 4*n if conv else k_pad))
#
# and both the activation scratch at init and `blk` at execute come from that
# one function, so the block the DEVICE uses is exactly this and nothing the
# AOT computes can raise it. The denominator is the per-row scratch the block
# costs: k_pad bytes of quantised activation, plus for a conv the fp32 staging
# row the NCHW scatter needs. So xq plus conv_stage together never exceed the
# budget, which is what keeps a delegate call inside the fast temp tier.
#
# Two numbers, both reported, neither pretending to be the other: this block is
# what runs, and `block_rows(...)` is the budget-derived ideal; reporting the
# ideal as what ran would understate flash traffic (7x on some prefill shapes).
# If the runtime constants change, these have to change with them;
# test_conv_support.py reads them out of the C++ to check.
RUNTIME_ROW_BLOCK_MAX = 32
RUNTIME_ROW_BLOCK_SCRATCH_BYTES = 16 * 1024


def runtime_block(m: int, k_pad: int, n: int = 0, conv: bool = False) -> int:
    """The block size the DEVICE will actually use for one op.

    Mirrors `row_block_for()` in esp32_backend.cpp exactly, including the
    conv staging term; `n` is only consulted when `conv` is set.
    """
    per_row = k_pad + (4 * n if conv else 0)
    if m <= 1 or per_row <= 0:
        return 1
    b = RUNTIME_ROW_BLOCK_SCRATCH_BYTES // per_row
    return max(1, min(m, RUNTIME_ROW_BLOCK_MAX, b))


def block_rows(k: int, sram_budget: int = None, m: int = None) -> int:
    """How many output rows share one pass over the weights.

    NOT WHAT THE DEVICE DOES. Read `runtime_block` above first: the
    runtime caps its block at `RUNTIME_ROW_BLOCK_MAX` rows and at
    `RUNTIME_ROW_BLOCK_SCRATCH_BYTES` of per-call scratch, so this function
    describes the block the policy WOULD pick given the whole SRAM-residency
    budget. Every figure below is headroom, not a result.

    THE parameter, not a tuning knob. Weight traffic is `K*N*ceil(m/B)` and the
    weights live in SPI flash, the slowest tier; the im2col scratch is `B*K`
    bytes and lives in internal SRAM, the fastest and smallest. On one mid-depth
    layer (m=324, K=1152, N=128), B=1 would stream 46,656 KB from flash and B=m
    144 KB -- 324x across the choice.

    So: take the largest B whose scratch fits, capped at m because more blocks
    than rows buys nothing.
    """
    budget = sram_budget if sram_budget is not None else CHIP.sram_resident_bytes
    if not budget or k <= 0:
        return 1
    b = max(1, budget // (k * _ACT_BYTES))
    return min(b, m) if m else b


def flash_traffic(geom: ConvGeom, b: int, weight_bytes_per_elem: float = 1.0):
    """(bytes streamed from flash, number of weight passes) for a block size.

    Reported at export rather than trusted, because a bad B is a 300x regression
    that PARITY CANNOT SEE -- the arithmetic is identical either way and every
    checksum still matches. Silence would be indistinguishable from health.
    """
    passes = -(-geom.m // b)
    # Whole bytes per pass, rounded up: a pass streams every weight, so a
    # non-empty conv never costs 0 B (a 1->1 1x1 conv at i4 is 0.5 B).
    per_pass = math.ceil(geom.k * geom.n * weight_bytes_per_elem)
    return per_pass * passes, passes


class ConvRefused(Exception):
    """Why a convolution is not claimable. Carries the specific failed check."""


def transposed_extent(e_in: int, k: int, stride: int, pad: int,
                      dil: int) -> int:
    """A transposed convolution's output extent on one axis before output
    padding: torch's and ExecuTorch's `(in - 1)*stride - 2*pad + dil*(k - 1)
    + 1`. Output padding then adds up to `max(stride, dil) - 1` positions at
    the far end, the window the runtime's `conv_axis_matches` accepts."""
    return (e_in - 1) * stride - 2 * pad + dil * (k - 1) + 1


def admit(node, allow_1d: bool = False) -> ConvGeom:
    """Validate a `convolution.default` node, or raise ConvRefused.

    Every limit here is a stage boundary rather than a permanent rule, and each
    says which, so the refusal reasons a user sees distinguish "not yet" from
    "never".
    """
    args = list(node.args)
    if len(args) < 9:
        raise ConvRefused(f"convolution has {len(args)} args, expected 9")
    inp, wt, bias, stride, padding, dilation, transposed, out_pad, groups = args[:9]

    if transposed and int(groups) != 1:
        raise ConvRefused("grouped transposed convolution is not implemented")
    if not transposed and any(int(p) for p in out_pad):
        raise ConvRefused(f"output_padding {list(out_pad)} is only meaningful "
                          f"for transposed convolution")
    iv = inp.meta.get("val") if hasattr(inp, "meta") else None
    wv = wt.meta.get("val") if hasattr(wt, "meta") else None
    ov = node.meta.get("val")
    if iv is None or wv is None or ov is None:
        raise ConvRefused("missing shape metadata")

    # Checked here, not with the other flags above, because it needs the
    # resolved channel counts: "grouped" and "depthwise" differ only by whether
    # groups equals them.
    g_int = int(groups)
    if g_int != 1:
        c_in_t, c_out_t = int(iv.shape[1]), int(ov.shape[1])
        if c_in_t % g_int or c_out_t % g_int:
            raise ConvRefused(
                f"groups={g_int} does not divide C_in={c_in_t} / "
                f"C_out={c_out_t}")
    spatial = len(iv.shape) - 2
    if spatial not in (1, 2, 3):
        raise ConvRefused(f"{spatial}-D convolution; 1-D, 2-D and 3-D are "
                          f"implemented")
    if int(iv.shape[0]) != 1:
        raise ConvRefused(f"batch {int(iv.shape[0])}: m would be B*H_out*W_out "
                          f"and the gather indexes one image (stage 4)")

    # The weight must be a build-time constant, exactly as for a matmul, but
    # that is the GATE's question (`_is_build_time_const`, which needs the
    # exported program this function does not have). Requiring a bare
    # placeholder here would refuse every PT2E-quantised conv, whose weight is a
    # dequantize_per_channel chain over constant codes -- a constant in every
    # sense that matters, and one the delegate consumes verbatim.

    if spatial == 3:
        # [B,C,D,H,W]; weight [C_out,C_in,kd,kh,kw]. The flattening order the
        # packer sees is (c, d, i, j), which is what the gather walks.
        geom = ConvGeom(
            d_in=int(iv.shape[2]), h_in=int(iv.shape[3]), w_in=int(iv.shape[4]),
            c_in=int(iv.shape[1]) // g_int,
            transposed=1 if transposed else 0,
            d_out=int(ov.shape[2]), h_out=int(ov.shape[3]),
            w_out=int(ov.shape[4]), c_out=int(ov.shape[1]),
            kd=int(wv.shape[2]), kh=int(wv.shape[3]), kw=int(wv.shape[4]),
            stride_d=int(stride[0]), stride_h=int(stride[1]),
            stride_w=int(stride[2]),
            pad_d=int(padding[0]), pad_h=int(padding[1]), pad_w=int(padding[2]),
            dil_d=int(dilation[0]), dil_h=int(dilation[1]),
            dil_w=int(dilation[2]), groups=g_int,
            c_out_total=int(ov.shape[1]),
        )
    elif spatial == 1:
        # A conv1d IS a conv2d with a degenerate height axis. Synthesising
        # h=1/kh=1/stride_h=1/pad_h=0/dil_h=1 makes every downstream consumer --
        # the packer, the geometry record, and the DEVICE GATHER -- work
        # unchanged: the gather's row loop runs once and `xc + ih*w_in` collapses
        # to `xc`. Handling 1-D as its own case would have meant a second gather
        # to write, verify and keep in step with the first, for no difference in
        # the arithmetic.
        geom = ConvGeom(
            h_in=1, w_in=int(iv.shape[2]), c_in=int(iv.shape[1]) // g_int,
            transposed=1 if transposed else 0,
            h_out=1, w_out=int(ov.shape[2]), c_out=int(ov.shape[1]),
            kh=1, kw=int(wv.shape[2]),
            stride_h=1, stride_w=int(stride[0]),
            pad_h=0, pad_w=int(padding[0]),
            dil_h=1, dil_w=int(dilation[0]), groups=g_int,
            c_out_total=int(ov.shape[1]),
        )
    else:
        geom = ConvGeom(
            h_in=int(iv.shape[2]), w_in=int(iv.shape[3]),
            c_in=int(iv.shape[1]) // g_int,
            transposed=1 if transposed else 0,
            h_out=int(ov.shape[2]), w_out=int(ov.shape[3]), c_out=int(ov.shape[1]),
            kh=int(wv.shape[2]), kw=int(wv.shape[3]),
            stride_h=int(stride[0]), stride_w=int(stride[-1]),
            pad_h=int(padding[0]), pad_w=int(padding[-1]),
            dil_h=int(dilation[0]), dil_w=int(dilation[-1]), groups=g_int,
            c_out_total=int(ov.shape[1]),
        )
    if geom.transposed:
        # A transposed conv stores its weight [C_in, C_out, ...]; the dense
        # equivalent wants [C_out, C_in, ...], which is a build-time transpose
        # of a constant and therefore free. Only the CHECK differs here.
        if int(wv.shape[0]) != geom.c_in or int(wv.shape[1]) != geom.c_out:
            raise ConvRefused(
                f"transposed weight {list(wv.shape)[:2]} does not match "
                f"C_in={geom.c_in}, C_out={geom.c_out}")
        # Output padding lengthens the far end of an axis by fewer than
        # max(stride, dilation) positions; torch refuses more. The record
        # carries the padded extent, and this is the check the runtime makes
        # of it at load (conv_axis_matches), made here too so that a geometry
        # the runtime would refuse stays portable instead of failing to load.
        # A degenerate axis (depth on a conv2d, height on a conv1d) has
        # extent 1 at kernel 1 and no output padding.
        op = [int(p) for p in out_pad]
        if len(op) == 1:
            op = op * spatial
        if len(op) != spatial:
            raise ConvRefused(f"output_padding {op} for a {spatial}-D "
                              f"convolution")
        op = [0] * (3 - spatial) + op
        for name, e_in, e_out, k, s, p, d, pad_out in (
                ("depth", geom.d_in, geom.d_out, geom.kd, geom.stride_d,
                 geom.pad_d, geom.dil_d, op[0]),
                ("height", geom.h_in, geom.h_out, geom.kh, geom.stride_h,
                 geom.pad_h, geom.dil_h, op[1]),
                ("width", geom.w_in, geom.w_out, geom.kw, geom.stride_w,
                 geom.pad_w, geom.dil_w, op[2])):
            extent = transposed_extent(e_in, k, s, p, d)
            if e_out - extent != pad_out or not 0 <= pad_out < max(s, d):
                raise ConvRefused(
                    f"transposed {name} extent {e_out} is not {extent} plus "
                    f"an output_padding below max(stride {s}, dilation {d}) "
                    f"(output_padding {list(out_pad)})")
    elif int(wv.shape[1]) != geom.c_in:
        raise ConvRefused(f"weight C_in {int(wv.shape[1])} != input C_in/groups "
                          f"{geom.c_in}")
    # PADDING may legitimately be zero; extents, kernel sizes, strides and
    # dilations may not. A blanket `> 0` would reject every unpadded convolution
    # -- including 1x1, the most common shape in a detector neck.
    # Enumerate what must be POSITIVE, not what may be zero. Offsets and flags
    # (padding, group_idx, transposed) default to 0, and positivity is the
    # property that belongs to a closed set (extents, kernel sizes, strides,
    # dilations, channel counts); everything else is allowed to be zero and a
    # new field is right by default.
    MUST_BE_POSITIVE = {
        "h_in", "w_in", "c_in", "h_out", "w_out", "c_out",
        "kh", "kw", "kd", "d_in", "d_out",
        "stride_h", "stride_w", "stride_d",
        "dil_h", "dil_w", "dil_d", "groups",
    }
    for name, v in vars(geom).items():
        floor = 1 if name in MUST_BE_POSITIVE else 0
        if not (floor <= v < 65536):
            raise ConvRefused(f"{name}={v} outside [{floor}, 65535]; the "
                              f"geometry record is uint16")
    return geom


def _axis_index(geom, o, kk, stride, pad, dil, extent, ksize):
    """The input index one output position reads for kernel tap `kk`, or None.

    Two modes, one formula each. A DENSE conv walks the input with the stride:
    `o*stride - pad + kk*dil`. A TRANSPOSED conv is the dense conv, with the
    taps flipped, over a ZERO-STUFFED input, so the same tap maps to
    `z = o + kk*dil - ((k-1)*dil - pad)` in the stuffed grid and only lands on
    a real sample when that is divisible by the stride -- off-lattice taps
    contribute zero, exactly as out-of-bounds ones already do. Expressing the
    stuffing as this divisibility test is what avoids materialising it:
    building it costs 4.84x input inflation on a 2x2/stride-2 up-conv, and
    this costs a modulo.

    The transposed form is torch's definition read backwards: input i feeds
    output `i*stride - pad + t*dil` through tap t, and the flipped tap
    kk = k-1-t of output o therefore reads input z / stride. Nothing in it
    depends on the output extent, so output padding (positions past the
    computed extent) needs nothing here.
    """
    if geom.transposed:
        z = o + kk * dil - ((ksize - 1) * dil - pad)
        if z < 0 or z % stride:
            return None
        idx = z // stride
        return idx if 0 <= idx < extent else None
    idx = o * stride - pad + kk * dil
    return idx if 0 <= idx < extent else None


def _src_index(geom, od, oh, ow, dd, i, j):
    return (_axis_index(geom, od, dd, geom.stride_d, geom.pad_d, geom.dil_d,
                        geom.d_in, geom.kd),
            _axis_index(geom, oh, i, geom.stride_h, geom.pad_h, geom.dil_h,
                        geom.h_in, geom.kh),
            _axis_index(geom, ow, j, geom.stride_w, geom.pad_w, geom.dil_w,
                        geom.w_in, geom.kw))


def split_groups(geom: ConvGeom):
    """A grouped conv as a list of per-group geometries.

    Each group is an ordinary dense conv over its own `C_in/groups` input
    channels producing its own `C_out/groups` output channels, which is why this
    needs no new gather and no new kernel: it is the conv path already verified,
    run g times against sliced weights and offset channel pointers.

    Depthwise deliberately does NOT come through here. It is the limit case
    g == C, where this would emit C descriptors and C dispatches for a kernel
    whose whole point is being cheap; it keeps its single-op loop instead.
    """
    if geom.groups <= 1 or geom.depthwise:
        return [geom]
    per_out = geom.c_out // geom.groups
    out = []
    for gi in range(geom.groups):
        # `groups` is KEPT, not reset to 1: it is what makes c_in*groups the
        # total input channel count, which the runtime needs to size the input
        # and to offset this slice.
        out.append(ConvGeom(**{**vars(geom), "c_out": per_out,
                               "group_idx": gi}))   # c_out_total carried
    return out
