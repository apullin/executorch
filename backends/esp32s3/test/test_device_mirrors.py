# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every device mirror reproduces the device's output bits, without QEMU.

run_parity.py holds each delegated row to its mirror's output bits on a fresh
QEMU run. This pins the mirrors that need no export to the bits the parity
firmware printed, so a change to a mirror -- or to fma_f32, or to a fast_math.h
transcription -- fails here in seconds instead of at the next parity run. The
depthwise rows are pinned in test_conv_support.py; the rows whose references
need an export (static scales, embeddings, the selected head, pass-through
codes) are held by run_parity alone.

    python test_device_mirrors.py
"""

import sys

import numpy as np

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def device_bits():
    """(row, mirror, the `output bits` main.cpp printed for it)."""
    from executorch.backends.esp32s3.test import models as m

    return (
        # The dense loops: plain i8 (addmm_q_row), the i4 unpack, i16's int64
        # accumulator, and opcode 1's k-outer fp32 loop.
        ("mlp_delegated", m.mlp_i8_out, "9b04d25e"),
        ("mlp_i4", lambda: m.mlp_iq_out("i4"), "86461967"),
        ("mlp_i16", lambda: m.mlp_iq_out("i16"), "269b96ed"),
        ("mlp_fp32_backend", m.mlp_fp32_backend_out, "12bea8bb"),
        # The fused epilogues: clamp, SiLU, GELU, the five one-op activations,
        # and espressif::attn_tail behind a portable x * x (its values PCG64's
        # integers, the same bytes on every machine).
        ("mlpr_delegated", m.mlpr_i8_out, "cb92a373"),
        ("mlps_delegated", m.mlps_i8_out, "62f504a4"),
        ("epiacts_delegated", m.epiacts_i8_out, "9a321ea8"),
        ("attntail", m.attntail_mirror_out, "f14a29bd"),
        # Convolutions through the im2col gather, row-blocked: 1-D, 2-D, 3-D,
        # transposed, grouped, the residual scatter and fused activations.
        ("convd_delegated", m.convd_i8_out, "5f7ec1e0"),
        ("conv1d_delegated", m.conv1d_i8_out, "20011f16"),
        ("conv3d_delegated", m.conv3d_i8_out, "76f8018b"),
        ("conv_i4", lambda: m.convd_iq_out("i4"), "2f4ce6cc"),
        ("conv_i16", lambda: m.convd_iq_out("i16"), "96c4ee46"),
        ("convt_delegated", m.convt_i8_out, "77f8cbc6"),
        ("convt_i4", lambda: m.convt_iq_out("i4"), "310b65e3"),
        ("convt_i16", lambda: m.convt_iq_out("i16"), "91e19d1b"),
        # Output padding (the decoder's 2x up-conv) and dilation with per-axis
        # geometry, and the int16 rung on the gather's other instantiation.
        ("convtop_delegated", lambda: m.convtop_iq_out("i8"), "7b17037a"),
        ("convtdil_delegated", lambda: m.convtdil_iq_out("i8"), "02252b61"),
        ("convtdil_i16", lambda: m.convtdil_iq_out("i16"), "1b53337d"),
        ("grpconv_delegated", m.groupconv_i8_out, "70f571fb"),
        ("grpconvedge_delegated", m.groupconvedge_i8_out, "203b449e"),
        ("grpconv8_delegated", m.groupconv8_i8_out, "bc9d76d2"),
        ("grpconv8_i4", lambda: m.groupconv8_iq_out("i4"), "5e466aba"),
        ("grpconv8_i16", lambda: m.groupconv8_iq_out("i16"), "e1bef054"),
        ("grpconvrelu_delegated", m.groupconvrelu_i8_out, "684fa6b6"),
        ("convres_delegated", m.convres_i8_out, "80cd50fb"),
        ("convresview_delegated", m.convresview_i8_out, "80cd50fb"),
        ("convres_i16", lambda: m.convres_iq_out("i16"), "289832c1"),
        ("convrelu_delegated", m.convrelu_i8_out, "3fb21af2"),
        ("convsilu_delegated", m.convsilu_i8_out, "b4cb277c"),
    )


def two_roundings(left, right, addend):
    """fl(fl(left * right) + addend): the multiply-add rounded twice."""
    f32 = np.float32
    product = (np.asarray(left, f32) * np.asarray(right, f32)).astype(f32)
    return (product + np.asarray(addend, f32)).astype(f32)


def main():
    from executorch.backends.esp32s3.quantizer import mirrors
    from executorch.backends.esp32s3.test import models

    print("each mirror reproduces the device's output bits:")
    rows = device_bits()
    for row, mirror, want in rows:
        got = models.output_bits(mirror())
        check(f"{row}: {got}", got == want, f"device {want}")

    # The negative control: the same mirrors with the multiply-add rounded
    # twice, as the device never does. Every dense row must then miss, or the
    # pins above could not see the fusion they claim to.
    print("rounding the device's madd.s twice misses its bits:")
    fused = mirrors.fma_f32
    mirrors.fma_f32 = two_roundings
    try:
        for row, mirror, want in rows[:3]:
            got = models.output_bits(mirror())
            check(f"{row}: {got}", got != want, f"device {want}")
    finally:
        mirrors.fma_f32 = fused

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL PASS")


def test_main():
    """Under pytest, what a script run checks: main() records every failed
    check and goes on; this fails if any did, naming them."""
    start = len(FAILS)
    rc = main()
    failed = FAILS[start:]
    assert not failed and rc in (None, 0), f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    main()
