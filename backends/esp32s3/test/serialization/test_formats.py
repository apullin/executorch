# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The weight-format ladder, tested against INTENT rather than against a twin.

Every row in run_parity.py compares two implementations of the same idea -- a host
reference against the device -- which catches divergence but is structurally blind to
shared error: when both sides derive from the same wrong assumption they agree
perfectly and the check passes. Two such assumptions are easy to share:

  * Weight precision setting activation precision: the i4 rung keeps int8
    ACTIVATIONS, and a host reference that quantised them to 4 bits would be
    caught only if the device happened to disagree.
  * Group boundaries from a scale-group COUNT: ceil(k/count) groups match the
    quantiser, which uses fixed groups of SCALE_GROUP with a short tail, only
    when k % 128 == 0. A device and a host reference that both reconstructed
    them that way would agree while every element in the straddled region of a
    ragged checkpoint is dequantised with the wrong group's scale; the blob
    carries the group SIZE for that reason.

So this file checks the arithmetic against first principles: the packer against a
bit-level statement of the layout, the quantiser against its own stated boundaries,
reconstruction against the error bound absmax quantisation promises. No device, no
twin reference, no tolerance wider than the mathematics requires.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test fails
naming its failed checks (@case).

    python test_formats.py
"""

import functools
import sys

import numpy as np

from executorch.backends.esp32s3.serialization.weight_formats import (
    FORMATS,
    SCALE_GROUP,
    encode_pass_through_embedding_scales,
    pack_rows,
    quantize_absmax_rows,
    round_half_away,
)
from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
    _pad_to,
)

RNG = np.random.default_rng(7)
FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def case(test):
    """One test. check() records a failure and goes on, so a script run
    reports every one (main); a test that only did that would pass under
    pytest whatever its checks found. Called, it fails if any check it made
    failed, naming them. main() calls the test itself (__wrapped__), so the
    script still runs every test."""

    @functools.wraps(test)
    def run():
        start = len(FAILS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


# ---------------------------------------------------------------------------
@case
def test_i4_pack_layout():
    """The nibble layout, stated bit-by-bit and checked against pack_rows.

    Element 2i in the LOW nibble, 2i+1 in the HIGH, each stored biased by +8 so both
    halves are ordinary unsigned nibbles. This is the statement the device unpack
    (`dst[j] = code - 8`) depends on.
    """
    print("\n-- i4 packing layout --")
    q = np.array([[-7, -1, 0, 1, 7, -4, 3, 2, -6]], dtype=np.int8)  # k=9, odd
    k_pad = _pad_to(9, 2)                                            # -> 10
    blob = np.frombuffer(pack_rows(q, k_pad, "i4"), dtype=np.uint8)
    check("row is k_pad/2 bytes", blob.size == k_pad // 2)
    ok = True
    padded = np.zeros(k_pad, dtype=np.int8)
    padded[:9] = q[0]
    for i in range(0, k_pad, 2):
        # int() before subtracting: numpy uint8 arithmetic wraps, so (nibble - 8) on a
        # uint8 is 249 rather than -7.
        byte = int(blob[i // 2])
        lo, hi = (byte & 0x0F) - 8, (byte >> 4) - 8
        ok &= (lo == padded[i]) and (hi == padded[i + 1])
    check("low nibble = element 2i, high = 2i+1, bias +8", ok)
    check("odd-k tail pads with biased zero (0x8 nibble)", (blob[-1] >> 4) == 8)


# ---------------------------------------------------------------------------
@case
def test_rung_roundtrip_bound():
    """Per-row absmax at each rung honours its reconstruction bound, and the row
    maximum maps to exactly +/-qmax.

    The bound is s/2 in exact arithmetic PLUS the fp32 division's own error: w/s is
    computed in float32, so the quotient carries up to |q|*2^-24 relative error before
    rounding, and a quotient that lands within that distance of a half-integer can
    round to the neighbouring code. At i4 that term is invisible; at i16 (|q| up to
    32767) it is ~0.002*s and a naive s/2 assertion fails on real data.
    """
    print("\n-- per-rung reconstruction bound --")
    w = RNG.standard_normal((8, 96)).astype(np.float32)
    for name in ("i4", "i8", "i16"):
        f = FORMATS[name]
        q, s = quantize_absmax_rows(w, name)
        recon = q.astype(np.float32) * s[:, None]
        bound = s[:, None] * (0.5 + f.qmax * 2.0 ** -23) + 1e-12
        check(f"{name}: |w - q*s| <= s*(1/2 + qmax*2^-23)",
              bool(np.all(np.abs(w - recon) <= bound)))
        rows = np.arange(w.shape[0])
        argm = np.abs(w).argmax(axis=1)
        check(f"{name}: row absmax maps to +/-{f.qmax}",
              bool(np.all(np.abs(q[rows, argm]) == f.qmax)))


# ---------------------------------------------------------------------------
@case
def test_weight_vs_activation_qmax():
    """Weight precision and activation precision are separate axes. i4 weights keep
    int8 activations -- the nibbles are unpacked into an int8 kernel, so narrowing the
    activation would discard precision to save traffic that does not exist."""
    print("\n-- weight/activation precision split --")
    check("i4: qmax 7, act_qmax 127",
          FORMATS["i4"].qmax == 7 and FORMATS["i4"].act_qmax == 127)
    check("i8: 127/127",
          FORMATS["i8"].qmax == 127 and FORMATS["i8"].act_qmax == 127)
    check("i16: 32767/32767",
          FORMATS["i16"].qmax == 32767 and FORMATS["i16"].act_qmax == 32767)


# ---------------------------------------------------------------------------
@case
def test_group_boundaries_ragged():
    """The headline: grouped quantisation uses FIXED groups of SCALE_GROUP with a
    short tail, and dequantisation must use the same boundaries. Checked directly
    against the quantiser's arithmetic for widths that do and do not divide evenly."""
    print("\n-- ragged group boundaries (the v7 bug class) --")
    g = SCALE_GROUP
    for k in (129, 200, 255, 256, 257, 384, 768):
        w = RNG.standard_normal((4, k)).astype(np.float32)
        q, s = quantize_absmax_rows(w, "i4", group=g, fp16_scales=True)
        check(f"k={k}: n_groups is ceil(k/{g})", s.shape[1] == -(-k // g))

        # Reconstruction with the quantiser's own boundaries...
        manual = np.empty_like(w)
        ok_bound = True
        for gi in range(s.shape[1]):
            a, b = gi * g, min((gi + 1) * g, k)
            blk = q[:, a:b].astype(np.float32) * s[:, gi, None]
            manual[:, a:b] = blk
            ok_bound &= bool(np.all(
                np.abs(w[:, a:b] - blk) <= 0.5 * s[:, gi, None] * (1 + 1e-6) + 1e-12))
        check(f"k={k}: per-group bound holds incl. short tail", ok_bound)

        # ...must equal the repeat-and-truncate expansion the host reference uses.
        per_elem = np.repeat(s, g, axis=1)[:, :k]
        check(f"k={k}: fixed-size expansion == quantiser boundaries",
              bool(np.array_equal(q.astype(np.float32) * per_elem, manual)))

    # And the hazard, demonstrated so it stays visible: derive a group size from
    # the COUNT and watch the reconstruction diverge.
    k = 200
    w = np.concatenate([RNG.uniform(-1e-3, 1e-3, (2, 128)),
                        RNG.uniform(-1.0, 1.0, (2, 72))], axis=1).astype(np.float32)
    q, s = quantize_absmax_rows(w, "i4", group=g, fp16_scales=True)
    right = np.repeat(s, g, axis=1)[:, :k]
    gsz_wrong = -(-k // s.shape[1])                      # ceil(200/2) = 100
    wrong = np.repeat(s, gsz_wrong, axis=1)[:, :k]
    diff = np.abs(q.astype(np.float32) * right - q.astype(np.float32) * wrong).max()
    # The straddled elements carry group 0's codes (values ~1e-3, codes up to +/-7)
    # but group 1's scale (~0.14), so the damage is ~7 * 0.14 ~ 1 -- three orders of
    # magnitude above the values themselves. Assert half that, not a round number the
    # construction only approaches.
    check("count-derived boundaries misattribute the straddled region",
          diff > 0.5, f"(max element error {diff:.3g} -- this is what v7 shipped)")


# ---------------------------------------------------------------------------
@case
def test_fp16_rounds_before_quantise():
    """Scales are rounded to fp16 BEFORE the division, so the stored half is the
    number the quantiser actually used. If rounding happened after, q would have
    been computed against a scale the device never sees."""
    print("\n-- fp16 scale ordering --")
    w = RNG.standard_normal((4, 300)).astype(np.float32)
    q, s = quantize_absmax_rows(w, "i4", group=SCALE_GROUP, fp16_scales=True)
    check("stored scales are exactly representable in fp16",
          bool(np.array_equal(s, s.astype(np.float16).astype(np.float32))))
    q2 = np.empty_like(q)
    for gi in range(s.shape[1]):
        a, b = gi * SCALE_GROUP, min((gi + 1) * SCALE_GROUP, 300)
        safe = np.where(s[:, gi] > 0, s[:, gi], np.float32(1.0))
        q2[:, a:b] = np.clip(round_half_away(w[:, a:b] / safe[:, None]), -7, 7)
    check("codes were computed against the fp16-rounded scale",
          bool(np.array_equal(q, q2)))


# ---------------------------------------------------------------------------
@case
def test_short_tail_amax_is_local():
    """The short final group takes its absmax over its real elements only. A tail
    outlier must set the tail's scale and leave every full group untouched."""
    print("\n-- short-tail locality --")
    k = 200
    w = RNG.uniform(-0.1, 0.1, (2, k)).astype(np.float32)
    w[0, 190] = 50.0                                     # outlier in the 72-wide tail
    q, s = quantize_absmax_rows(w, "i8", group=SCALE_GROUP, fp16_scales=True)
    exp_head = np.float16(np.abs(w[0, :128]).max() / 127.0)
    exp_tail = np.float16(np.float32(50.0) / 127.0)
    check("tail scale reflects the tail outlier",
          s[0, 1] == np.float32(exp_tail), f"({s[0,1]:.4g})")
    check("head group unaffected by tail outlier",
          s[0, 0] == np.float32(exp_head))


# ---------------------------------------------------------------------------
@case
def test_embedding_gather_matches_intent():
    """models.embedding_iq_absmax gathers rows of the quantised table; the result must
    equal dequantising the whole table with the quantiser's boundaries and indexing.
    On a ragged table this checks the group expansion against the quantiser's
    boundaries."""
    print("\n-- embedding gather vs whole-table dequant (ragged table) --")
    from executorch.backends.esp32s3.test.reference import embedding_iq_absmax

    table = RNG.standard_normal((40, 200)).astype(np.float32)
    idx = np.array([0, 7, 39, 7, 13])
    got = embedding_iq_absmax(table, idx, "i8")
    q, s = quantize_absmax_rows(table, "i8", group=SCALE_GROUP, fp16_scales=True)
    full = np.empty_like(table)
    for gi in range(s.shape[1]):
        a, b = gi * SCALE_GROUP, min((gi + 1) * SCALE_GROUP, 200)
        full[:, a:b] = q[:, a:b].astype(np.float32) * s[:, gi, None]
    check("gather == full-dequant[idx]", bool(np.array_equal(got, full[idx])))


# ---------------------------------------------------------------------------
@case
def test_dedup_precondition():
    """The tied-weight dedup fires only when the embed and addmm paths produce
    byte-identical payloads for the same tensor. That holds at D=96 because
    96 % 32 == 0 (the addmm i4 row alignment) -- and NOT at, say, D=80, where the
    deployed model would silently grow ~1.6 MB. Pinned so the fragility is a stated
    property rather than a surprise."""
    print("\n-- tied-weight dedup precondition --")
    f = FORMATS["i4"]
    for d, expect in ((96, True), (80, False)):
        w = RNG.standard_normal((64, d)).astype(np.float32)
        q, _ = quantize_absmax_rows(w, "i4")
        embed = pack_rows(q, _pad_to(d, 2), f)
        addmm = pack_rows(q, _pad_to(d, f.k_align), f)
        same = embed == addmm
        check(f"D={d}: payloads {'identical' if expect else 'differ'} "
              f"(dedup {'fires' if expect else 'cannot fire'})", same == expect)


# ---------------------------------------------------------------------------
@case
def test_pass_through_embedding_scale_storage():
    """Scale bytes and the storage flag must describe the same precision."""
    print("\n-- pass-through embedding scale storage --")
    half_exact = np.array([[0.125, 0.25], [0.5, 1.0]], dtype=np.float32)
    payload, fp16 = encode_pass_through_embedding_scales(half_exact)
    check("fp16-exact scales use fp16 storage", fp16)
    check("fp16 payload preserves the incoming values",
          bool(np.array_equal(
              np.frombuffer(payload, dtype=np.float16).astype(np.float32),
              half_exact.reshape(-1))))

    fp32 = np.array([[0.1, 0.2], [0.3, 0.7]], dtype=np.float32)
    payload, fp16 = encode_pass_through_embedding_scales(fp32)
    check("genuine fp32 scales keep the fp16 flag clear", not fp16)
    check("fp32 payload preserves the incoming values exactly",
          bool(np.array_equal(np.frombuffer(payload, dtype=np.float32),
                              fp32.reshape(-1))))


# ---------------------------------------------------------------------------
@case
def test_i16_embedding_scales_stay_fp32():
    """An i16 table quantised by the backend decodes within half an i16 step.

    The rung exists for accuracy, so its bound is the one absmax promises:
    |w - q*s| <= s/2 with s = amax/32767. fp16 cannot hold that scale -- it is
    subnormal for any group with amax below 2 -- so a grouped (k > 128) i16
    table must not take fp16 scales, which would make it decode worse than i8.
    Measured on the emitted blob rather than the quantiser, so what is checked
    is the choice the backend actually makes.
    """
    print("\n-- i16 embedding scale precision --")
    import contextlib
    import io
    import struct

    import torch
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.serialization.espressif_blob import (
        BLOB_HEADER_BYTES,
        FLAG_FP16_SCALES,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
        OP_EMBED_I16,
    )
    from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
    from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec

    n, k = 5, 300                                  # three groups, ragged tail
    amax = (1e-3, 0.02, 0.1)
    table = RNG.uniform(-1.0, 1.0, (n, k)).astype(np.float32)
    for g, a in enumerate(amax):
        blk = table[:, g * SCALE_GROUP:(g + 1) * SCALE_GROUP]
        blk *= np.float32(a) / np.abs(blk).max(axis=1, keepdims=True)

    class Table(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(n, k)
            self.emb.weight.data = torch.from_numpy(table.copy())

        def forward(self, idx):
            return self.emb(idx)

    edge = to_edge(torch.export.export(Table().eval(), (torch.arange(n),)))
    with contextlib.redirect_stdout(io.StringIO()):
        blob = Esp32Backend.preprocess(
            edge.exported_program(),
            Esp32CompileSpec(format_policy="i16").to_list()).processed_bytes
    desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES)))
    check("one grouped i16 gather",
          (desc["opcode"], desc["scale_group"]) == (OP_EMBED_I16, SCALE_GROUP),
          f"(opcode {desc['opcode']}, group {desc['scale_group']})")
    fp16 = bool(desc["flags"] & FLAG_FP16_SCALES)
    check("i16 scales are stored fp32", not fp16)
    n_groups = -(-k // SCALE_GROUP)
    q = np.frombuffer(blob, np.int16, n * desc["k_pad"], desc["wt_off"])
    q = q.reshape(n, desc["k_pad"])[:, :k].astype(np.float32)
    s = np.frombuffer(blob, np.float16 if fp16 else np.float32,
                      n * n_groups, desc["scale_off"])
    s = s.astype(np.float32).reshape(n, n_groups)
    recon = q * np.repeat(s, SCALE_GROUP, axis=1)[:, :k]
    worst = 0.0
    for g, a in enumerate(amax):
        cols = slice(g * SCALE_GROUP, min((g + 1) * SCALE_GROUP, k))
        err = float(np.abs(recon[:, cols] - table[:, cols]).max())
        worst = max(worst, err / (0.5 * a / 32767))
    check("every group decodes within half an i16 step", worst <= 1.01,
          f"(worst {worst:.3g} half-steps)")


# ---------------------------------------------------------------------------
def main():
    for t in (test_i4_pack_layout, test_rung_roundtrip_bound,
              test_weight_vs_activation_qmax, test_group_boundaries_ragged,
              test_fp16_rounds_before_quantise, test_short_tail_amax_is_local,
              test_embedding_gather_matches_intent, test_dedup_precondition,
              test_pass_through_embedding_scale_storage,
              test_i16_embedding_scales_stay_fp32):
        t.__wrapped__()
    print()
    if FAILS:
        print(f"FAILURES: {len(FAILS)}")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("OK: format arithmetic matches intent at every rung, grouped and ragged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
