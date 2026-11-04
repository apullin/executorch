# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The weight-format ladder: quantisation, packing and per-format row sizes."""

import numpy as np

from executorch.backends.esp32s3.serialization.espressif_blob import (
    BACKEND_NAME,
    OP_ADDMM_BF16,
    OP_ADDMM_F16,
    OP_ADDMM_I16,
    OP_ADDMM_I4,
    OP_ADDMM_I8,
    OP_EMBED_BF16,
    OP_EMBED_F16,
    OP_EMBED_I16,
    OP_EMBED_I4,
    OP_EMBED_I8,
    VEC_BYTES,
)
from executorch.backends.esp32s3.target.chips import ESP32S3 as CHIP


# Format name -> (matmul/conv opcode, embedding opcode).
HALF_FORMATS = {
    "fp16": (OP_ADDMM_F16, OP_EMBED_F16),
    "bf16": (OP_ADDMM_BF16, OP_EMBED_BF16),
}
# ... and its descriptor text, beside addmm_f32's: addmm_f16, embed_bf16.
HALF_TAGS = {"fp16": "f16", "bf16": "bf16"}


class Fmt:
    """One rung of the weight-format ladder.

    The ladder exists because this chip has no NPU. On an Ethos-U55 the partitioner's
    dominant constraint is staying inside the accelerator's domain -- an unsupported op
    means Vela cuts the graph and a tensor round-trips through shared memory, which is
    expensive enough to distort every other decision. PIE is a SIMD extension of the same
    LX7 core that runs the portable kernels: same registers, same SRAM, no command
    stream. Changing precision between two ops costs the arithmetic and nothing else, so
    format can be chosen **per op** instead of once per subgraph.

    What each rung is for, from the measured roofline (intensity is exactly M, so a
    decode matvec is memory-bound and int8's win is 4x less traffic, not the vector
    unit):

      i4   Half the traffic of i8. For weights that live in PSRAM or flash, where the
           op is bandwidth-bound and the nibble unpack runs in cycles that were going to
           be spent waiting anyway. This is the rung that matters for the 2.43 MB head:
           40.9 ms of PSRAM floor becomes 20.9.
      i8   The default (ee.vmulas.s8.accx, 16 lanes).
      i16  Twice the traffic and half the SIMD width of i8 (ee.vmulas.s16.accx is 8
           lanes x 40-bit in the 320-bit QACC), so it is never a speed choice. It is an
           *accuracy* choice, for the ops where i8's error is the binding constraint
           rather than bandwidth.

    `qmax` is deliberately symmetric and one short of the type's range: -127..127 rather
    than -128..127. An asymmetric range would make the negative extreme unrepresentable
    after negation and is not worth the one extra code.
    """

    def __init__(self, name, opcode, embed_opcode, bits, qmax, act_qmax, dtype,
                 bytes_per_row_elem, k_align):
        self.name = name
        self.opcode = opcode
        self.embed_opcode = embed_opcode
        self.bits = bits
        # Weight precision -- what the rung is named for, and the only thing that
        # changes how many bytes are stored and streamed.
        self.qmax = qmax
        # ACTIVATION precision, which is a separate decision and deliberately not the
        # same number. Activations are quantised per row as the op runs (Stage B's
        # static int8 codes between delegates aside), not streamed like weights, so
        # there is no traffic to save by quantising them harder; the only constraint
        # is the width the kernel multiplies in. i4 weights therefore keep int8
        # activations -- they are unpacked into int8 and fed to the int8 PIE kernel, so
        # narrowing the activation would discard precision and buy nothing at all.
        self.act_qmax = act_qmax
        self.dtype = dtype
        self.bytes_per_row_elem = bytes_per_row_elem
        # Elements each row is padded to. Chosen so that row_bytes is always a multiple
        # of VEC_BYTES and every row therefore starts 16 B aligned -- ee.vld.128.ip
        # truncates the low address bits silently rather than faulting, so a row that
        # merely *happens* to be aligned is not good enough. i4 needs 32 elements to
        # reach 16 bytes because it stores two per byte.
        self.k_align = k_align

    def row_bytes(self, k_pad: int) -> int:
        return int(k_pad * self.bytes_per_row_elem)


FORMATS = {
    "i4": Fmt("i4", OP_ADDMM_I4, OP_EMBED_I4, 4, 7, 127, np.int8, 0.5, 2 * VEC_BYTES),
    "i8": Fmt("i8", OP_ADDMM_I8, OP_EMBED_I8, 8, 127, 127, np.int8, 1, VEC_BYTES),
    "i16": Fmt("i16", OP_ADDMM_I16, OP_EMBED_I16, 16, 32767, 32767, np.int16, 2,
               VEC_BYTES),
}


# Weights at or above this many bytes (as i8) are assumed not to be internal-SRAM
# resident, so their op is bandwidth-bound and the cheaper format wins. The number
# lives in the chip descriptor; see chips.py for how it was chosen.
SRAM_RESIDENT_BYTES = CHIP.sram_resident_bytes


def round_half_away(x: np.ndarray) -> np.ndarray:
    """Match the device's roundf(). np.round is banker's and would differ on ties."""
    return np.floor(np.abs(x) + 0.5) * np.sign(x)


# One scale per this many consecutive weights along a row, when grouping is used.
# 128: the group of the int4 format the story model trains with, so its weights
# pass through as trained.
SCALE_GROUP = 128


def quantize_absmax_rows(w_nk: np.ndarray, fmt="i8", group=None, fp16_scales=False):
    """Symmetric absmax at the given rung. Returns (q[N,K], scale[N] or [N,n_groups]).

    Data-free: no calibration set and no observers, which is why this needs nothing
    from PT2E. A row of all zeros gets scale 0 and quantises to zeros, which
    dequantises back to zero correctly.

    `group=None` is per-row and is the default, which the parity references depend
    on. Grouping only changes
    rows wider than the group: the 768-wide `ple_table`, and with `group_weights=True`
    the same group-128 QAT format on ordinary i4/i8 weights (wide FFN/down and PLE
    projections).

    A short final group is allowed (pass-through formats may be ragged too) and takes
    its amax over the real elements only.

    `fp16_scales` rounds each scale to IEEE half BEFORE quantising, so the stored scale
    and the scale the quantiser divided by are the same number. Doing it afterwards
    would leave the device dequantising with a value the host never used.
    """
    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    n, k = w_nk.shape

    FP16_MAX = np.float32(65504.0)

    def _scales_from(amax):
        s = (amax / float(f.qmax)).astype(np.float32)
        if fp16_scales:
            # An overflowing scale becomes inf, and the device then computes
            # code * inf = NaN for the whole row and everything downstream of
            # it. numpy only warns. Needs |w| > ~4.6e5 at i4, so trained weights
            # will not reach it -- but a raw lookup table could, and a NaN model
            # that built cleanly is the worst possible way to find out.
            if np.any(s > FP16_MAX):
                worst = float(s.max())
                raise RuntimeError(
                    f"{BACKEND_NAME}: a group scale of {worst:.4g} overflows "
                    f"fp16 (max {float(FP16_MAX):.0f}); the device would "
                    f"dequantise that row to NaN. Rescale the tensor or use "
                    f"fp32 scales."
                )
            s = s.astype(np.float16).astype(np.float32)
        return s

    if group is None or group >= k:
        scale = _scales_from(np.abs(w_nk).max(axis=1))
        safe = np.where(scale > 0, scale, np.float32(1.0))
        q = round_half_away(w_nk / safe[:, None])
        return np.clip(q, -f.qmax, f.qmax).astype(f.dtype), scale

    n_groups = (k + group - 1) // group
    scales = np.zeros((n, n_groups), dtype=np.float32)
    q = np.zeros((n, k), dtype=np.float32)
    for g in range(n_groups):
        a, b = g * group, min((g + 1) * group, k)
        blk = w_nk[:, a:b]
        s = _scales_from(np.abs(blk).max(axis=1))
        scales[:, g] = s
        safe = np.where(s > 0, s, np.float32(1.0))
        q[:, a:b] = round_half_away(blk / safe[:, None])
    return np.clip(q, -f.qmax, f.qmax).astype(f.dtype), scales


def pack_rows(q: np.ndarray, k_pad: int, fmt) -> bytes:
    """Zero-pad each row to k_pad and serialise it in the format's storage layout.

    Nibble order (i4): low nibble = even column, high nibble = odd column, per
    byte. The device unpacks this with a vector shift, two masks and a byte zip
    (unpack_i4_s3.S), or dots it in place against an activation row
    deinterleaved to match (dot_i4_s3.S).

    Zeros contribute nothing to a dot product, which is what lets the kernel read k_pad
    elements with no tail case. For i4 the pad keeps the row an even number of nibbles,
    so packing never straddles a row boundary.

    i4 packs two weights per byte, element 2i in the low nibble and 2i+1 in the high,
    each stored biased by +8. The bias is what makes both halves ordinary unsigned
    nibbles, so the device unpack is a mask and a subtract with no sign extension to get
    wrong -- the backend's own codes live in [-7,7] and pass-through int4 codes in
    [-8,7], so biased they live in [0,15] and never overflow the nibble.
    """
    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    n, k = q.shape
    padded = np.zeros((n, k_pad), dtype=f.dtype)
    padded[:, :k] = q
    if f.bits != 4:
        return padded.tobytes()
    lo = (padded[:, 0::2].astype(np.int16) + 8).astype(np.uint8)
    hi = (padded[:, 1::2].astype(np.int16) + 8).astype(np.uint8)
    return ((hi << 4) | (lo & 0x0F)).astype(np.uint8).tobytes()


def encode_pass_through_embedding_scales(scales):
    """Store incoming grouped scales at their existing precision."""
    sc = np.asarray(scales, dtype=np.float32)
    fp16 = bool(np.array_equal(sc.astype(np.float16).astype(np.float32), sc))
    payload = sc.astype(np.float16 if fp16 else np.float32).tobytes()
    return payload, fp16


def format_error(w_nk: np.ndarray, fmt) -> float:
    """Relative error of reconstructing the weights at this rung.

    Measured on the actual weights rather than assumed from bit width, because the
    thing that decides is the per-row dynamic range: a row with one outlier loses far
    more at i4 than a uniform one, and only the weights know which it is.
    """
    q, scale = quantize_absmax_rows(w_nk, fmt)
    recon = q.astype(np.float32) * scale[:, None]
    denom = float(np.abs(w_nk).max())
    if denom == 0.0:
        return 0.0
    return float(np.abs(recon - w_nk).max() / denom)


def choose_format(w_nk: np.ndarray, policy: str = "i8"):
    """Pick a rung for one op's weights. Returns (Fmt, reason string).

    Policies:
      "i8"    every op at i8. The default, and what the parity references model.
      "i4" / "i16"
              force one rung everywhere. For A/B runs, not for shipping.
      "fp32"  is handled by preprocess rather than this quantised-format
              chooser. It keeps inline [K,N] weights as an exact backend
              diagnostic/control; it is not a flash-scale deployment policy.
      "auto"  per op, decided by the weight's size -- the one thing the backend can
              actually know at preprocess time. Weights at or above the chip's
              SRAM-residency threshold (SRAM_RESIDENT_BYTES as i8) are taken to be read
              from flash or PSRAM, so their op is bandwidth-bound and the cheaper format
              wins; smaller weights have little traffic to save, so they stay at i8.
              Where the weights are read from at run time is the application's
              weight-tier budget, which preprocess cannot see. Reconstruction error is
              REPORTED, never gated on.

    "auto" is not the default. It changes numerics per op, which means the host
    references have to model the same choice, and a policy that silently disagreed
    with them would look exactly like a backend bug.
    """
    if policy in FORMATS:
        return FORMATS[policy], "forced"
    if policy in HALF_FORMATS:
        # Reached only for a weight an integer format_overrides entry takes
        # off the 2-byte policy (_emit_half has every other); the override
        # replaces this.
        return FORMATS["i8"], f"{policy} policy"
    if policy != "auto":
        raise RuntimeError(f"unknown format policy {policy!r}")

    i8_bytes = w_nk.shape[0] * w_nk.shape[1]
    if i8_bytes >= SRAM_RESIDENT_BYTES:
        return FORMATS["i4"], (f"bandwidth-bound ({i8_bytes:,} B at i8), "
                               f"i4 recon err {format_error(w_nk, 'i4'):.2e}")
    return FORMATS["i8"], (f"SRAM-resident ({i8_bytes:,} B), "
                           f"i8 recon err {format_error(w_nk, 'i8'):.2e}")


# Elements half_bits narrows at a time: its temporaries stay a few MB whatever
# the weight's size.
_HALF_CHUNK = 1 << 20


def half_bits(w: np.ndarray, fmt: str, name: str = "weight") -> np.ndarray:
    """fp32 values as fp16 or bf16 bit patterns (uint16), rounded to nearest
    even: exact for a value that came from that type (a promoted checkpoint's,
    NaN payloads included), so a weight is stored as it shipped. A finite
    value the type cannot hold (|w| past fp16's 65504) is refused by name
    rather than stored as an infinity.

    _HALF_CHUNK elements at a time, bf16's rounding in place in uint32 (the
    sum wraps only for a NaN, whose bits are replaced): beside the result,
    the temporaries stay a few MB whatever the weight's size, where widening
    the whole weight to uint64 took about eight times it (2 GB for a 268 MB
    weight)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    if fmt not in ("fp16", "bf16"):
        raise RuntimeError(f"{BACKEND_NAME}: {fmt!r} is not a 2-byte format")
    flat = w.reshape(-1)
    bits = np.empty(flat.shape, dtype=np.uint16)
    for start in range(0, flat.size, _HALF_CHUNK):
        part = flat[start:start + _HALF_CHUNK]
        wide = _half_chunk(part, bits[start:start + _HALF_CHUNK], fmt)
        lost = np.isfinite(part) & ~np.isfinite(wide)
        if lost.any():
            raise RuntimeError(
                f"{BACKEND_NAME}: {name} holds {float(part[lost][0])!r}, past "
                f"{fmt}'s range: stored as {fmt} it would be an infinity")
    return bits.reshape(w.shape)


def _half_chunk(w: np.ndarray, out: np.ndarray, fmt: str) -> np.ndarray:
    """half_bits on one chunk: the fp16 or bf16 bits of `w` into `out`;
    returns them widened back to float32, for the range check."""
    u = w.view(np.uint32)
    if fmt == "fp16":
        with np.errstate(over="ignore", invalid="ignore"):
            out[...] = w.astype(np.float16).view(np.uint16)
            wide = out.view(np.float16).astype(np.float32)
        # A NaN by integer moves: a hardware narrowing may quiet a signalling
        # one. Sign and top payload kept; a payload only in the low 13 bits
        # stays a NaN (quiet), never an infinity.
        nan = np.isnan(w)
        if nan.any():
            v = u[nan]
            mant = ((v >> 13) & 0x3FF).astype(np.uint16)
            out[nan] = (((v >> 16) & 0x8000).astype(np.uint16) | 0x7C00 | mant
                        | np.where(mant == 0, 0x200, 0).astype(np.uint16))
        return wide
    # Nearest even: add 0x7FFF and the lowest bit kept, drop the low half.
    t = u >> 16
    t &= 1
    t += u
    t += 0x7FFF
    t >>= 16
    out[...] = t
    # A NaN keeps its sign and top payload; one whose payload lay only in the
    # low half stays a NaN (quiet), never an infinity.
    nan = np.isnan(w)
    if nan.any():
        top = (u[nan] >> 16).astype(np.uint16)
        out[nan] = top | np.where((top & 0x7F) == 0, 0x40, 0).astype(np.uint16)
    t = out.astype(np.uint32)
    t <<= 16
    return t.view(np.float32)
