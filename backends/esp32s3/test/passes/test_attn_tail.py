# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The decode attention tail fuses into one espressif::attn_tail, and its mirror is exact.

Claims, each tested as a function so pytest collects them:

  1. The tiny Glimmer decode step (2 layers, d64) fuses exactly one tail per layer,
     host output is BIT-IDENTICAL before and after (the op's CompositeExplicitAutograd
     implementation is the replaced graph), and the emitted program calls
     espressif::attn_tail.out in place of mul/where/_softmax/bmm.
  2. Anything but the exact chain is refused with a reason, and expect= catches it.
  3. A fully masked row is NaN, like the graph it replaces.
  4. The host mirror is the device arithmetic: fma_f32 is a correctly rounded
     single-rounding FMA (checked against exact rationals, including the tie a
     float64 sum gets wrong), and fast_expf_f32_reference loads the constants the
     device binary loads.
  5. fast_exp_neg, the table exp the kernel maps s - max with: fast_math.h's
     constants and table are the mirror's and are what they claim (the table
     re-derived exactly), NaN/-inf/+-0 and the flush threshold behave as
     fast_expf's do, and every float32 in [-87.3365447, 0] is within 1.004 ulp
     of float64 exp (an exhaustive sweep, about two minutes).

    python -m pytest test_attn_tail.py    (or run it as a script)
"""

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---- the mirror -------------------------------------------------------------
def _round_f32(q: Fraction) -> np.float32:
    """Exact rational -> nearest float32, ties to even (the reference rounding)."""
    if q == 0:
        return np.float32(0.0)
    sign = -1 if q < 0 else 1
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    ulp = max(e - 23, -149)
    scaled = q / Fraction(2) ** ulp
    n, rem = divmod(scaled.numerator, scaled.denominator)
    half = Fraction(rem, scaled.denominator)
    if half > Fraction(1, 2) or (half == Fraction(1, 2) and n % 2):
        n += 1
    return np.float32(sign * n * 2.0 ** ulp)


def test_fma_f32_is_one_rounding():
    from executorch.backends.esp32s3.quantizer.mirrors import fma_f32

    # The trap: 1+2^-11+2^-24 is a float32 midpoint, and 2^-80 above it is
    # below float64's resolution, so a float64 sum lands ON the tie and
    # rounds to even -- down. One rounding of the exact value goes up.
    a = np.float32(1 + 2 ** -12)
    c = np.float32(2.0 ** -80)
    want = np.float32(1 + 2 ** -11 + 2 ** -23)
    naive = np.float32(np.float64(a) * np.float64(a) + np.float64(c))
    assert naive != want, "the adversarial case no longer exercises the tie"
    assert fma_f32(a, a, c) == want

    rng = np.random.default_rng(0)
    left = (rng.standard_normal(3000) * 2.0 ** rng.integers(-20, 20, 3000))
    right = (rng.standard_normal(3000) * 2.0 ** rng.integers(-20, 20, 3000))
    addend = (rng.standard_normal(3000) * 2.0 ** rng.integers(-40, 40, 3000))
    left, right, addend = (v.astype(np.float32) for v in (left, right, addend))
    got = fma_f32(left, right, addend)
    for x, y, z, g in zip(left, right, addend, got):
        exact = Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z))
        assert g == _round_f32(exact), (x, y, z, g)
    assert fma_f32(np.float32(np.inf), np.float32(1), np.float32(1)) == np.inf
    assert np.isnan(fma_f32(np.float32(np.nan), np.float32(1), np.float32(1)))


# fast_math.h's constants as the device binary loads them (l32r literals in
# objdump of et_runner.elf's fast_expf). A float32 literal that rounded
# differently through Python's double would show up here, not as a golden
# that is one ulp off on the board.
DEVICE_EXPF_BITS = {
    1.44269504088896341: 0x3FB8AA3B, 0.693359375: 0x3F318000,
    2.12194440e-4: 0x395E8083, 1.9875691500e-4: 0x39506967,
    1.3981999507e-3: 0x3AB743CE, 8.3334519073e-3: 0x3C088908,
    4.1665795894e-2: 0x3D2AA9C1, 1.6666665459e-1: 0x3E2AAAAA,
    5.0000001201e-1: 0x3F000000, 88.7228317: 0x42B17217,
    -87.3365447: 0xC2AEAC50,
}


def test_fast_expf_mirror():
    from executorch.backends.esp32s3.test.models import (
        fast_expf_f32_reference as fast_expf,
    )

    for literal, bits in DEVICE_EXPF_BITS.items():
        assert int(np.float32(literal).view(np.uint32)) == bits, literal

    special = np.array([0.0, -0.0, -np.inf, np.inf, np.nan, -88.0, 89.0,
                        -87.3365447, 88.7228317], dtype=np.float32)
    got = fast_expf(special)
    assert got[0] == 1.0 and got[1] == 1.0 and got[2] == 0.0
    assert got[3] == np.inf and np.isnan(got[4])
    assert got[5] == 0.0 and got[6] == np.inf

    x = np.linspace(-87.0, 88.0, 200_001, dtype=np.float32)
    y = fast_expf(x).astype(np.float64)
    exact = np.exp(x.astype(np.float64))
    ulp = np.spacing(exact.astype(np.float32)).astype(np.float64)
    err = float(np.max(np.abs(y - exact) / ulp))
    print(f"  fast_expf (device FMA arithmetic) max error {err:.3f} ulp")
    assert err <= 1.5, err


# ---- fast_exp_neg, the table exp attn_tail uses ------------------------------
FAST_MATH_H = Path(__file__).resolve().parents[2] / "runtime" / "fast_math.h"

# fast_exp_neg_constants() in fast_math.h, as float32 bits.
DEVICE_EXP_NEG_BITS = {
    "shifter": 0x4B400000, "inv_ln2_64": 0x42B8AA3B, "ln2_64_hi": 0x3C318000,
    "ln2_64_lo": 0xB65E8083, "c3": 0x3E2AAAAB, "c2": 0x3F000000,
    "flush": 0xC2AEAC50, "zero": 0x00000000,
}


def _header_exp_neg():
    """fast_exp_neg_constants()'s literals and kFastExpNegTable, from the header."""
    import re

    text = FAST_MATH_H.read_text()
    init = re.search(r"fast_exp_neg_constants\(\) \{\s*return \{([^}]*)\};", text)
    literals = [item.strip() for item in init.group(1).split(",")]
    assert literals[-1] == "kFastExpNegTable", literals
    values = []
    for literal in literals[:-1]:
        assert literal.endswith("f"), literal
        body = literal[:-1]
        values.append(float.fromhex(body) if "0x" in body else float(body))
    table = re.search(r"kFastExpNegTable\[64\] FAST_MATH_DRAM = \{([^}]*)\};", text)
    entries = [int(item, 16) for item in table.group(1).replace("\n", " ").split(",")
               if item.strip()]
    return dict(zip(DEVICE_EXP_NEG_BITS, values)), entries


def _exp2_64_f32_bits(j):
    """2^(j/64) correctly rounded to float32, from exact integer comparisons:
    the significand m (2^23 <= m < 2^24) whose half-ulp neighbours bracket it."""
    target = 1 << (23 * 64 + j + 64)       # (2 * 2^(23 + j/64))^64
    m = round(2 ** (23 + j / 64))
    while (2 * m + 1) ** 64 < target:
        m += 1
    while (2 * m - 1) ** 64 > target:
        m -= 1
    assert (2 * m - 1) ** 64 < target < (2 * m + 1) ** 64
    return (127 << 23) | (m - (1 << 23))


def test_fast_exp_neg_constants():
    """The header's literals and table, the mirror's, and exact derivations agree."""
    from executorch.backends.esp32s3.test import models

    literals, table = _header_exp_neg()
    for name, bits in DEVICE_EXP_NEG_BITS.items():
        assert int(np.float32(literals[name]).view(np.uint32)) == bits, name
    mirror = {"shifter": models.FAST_EXP_NEG_SHIFTER,
              "inv_ln2_64": models.FAST_EXP_NEG_INV_LN2_64,
              "ln2_64_hi": models.FAST_EXP_NEG_LN2_64_HI,
              "ln2_64_lo": models.FAST_EXP_NEG_LN2_64_LO,
              "c3": models.FAST_EXP_NEG_C3, "c2": 0.5,
              "flush": models.FAST_EXP_NEG_FLUSH, "zero": 0.0}
    for name, value in mirror.items():
        assert int(np.float32(value).view(np.uint32)) == DEVICE_EXP_NEG_BITS[name], name
    # What the constants are: hi + lo is ln2/64 to 2.6e-14, hi has 9 significant
    # bits (k * hi exact for |k| < 2^13), and every unflushed input has |k| < 2^13.
    ln2 = Fraction("0.6931471805599453094172321214581765680755")
    hi, lo = Fraction(literals["ln2_64_hi"]), Fraction(float(np.float32(literals["ln2_64_lo"])))
    assert abs(float(hi + lo - ln2 / 64)) < 3e-14
    assert (hi * 2 ** 17).denominator == 1 and (hi * 2 ** 17).numerator == 1420
    assert abs(literals["flush"]) * float(literals["inv_ln2_64"]) < 2 ** 13 - 0.5
    exact = [(_exp2_64_f32_bits(j) - (j << 17)) & 0xFFFFFFFF for j in range(64)]
    assert table == exact
    assert [int(v) for v in models.fast_exp_neg_table()] == exact


def test_fast_exp_neg_specials():
    from executorch.backends.esp32s3.test.models import (
        FAST_EXP_NEG_FLUSH,
        fast_exp_neg_f32_reference as fast_exp_neg,
    )

    flush = np.float32(FAST_EXP_NEG_FLUSH)
    below = np.nextafter(flush, np.float32(-np.inf))
    special = np.array([0.0, -0.0, -np.inf, np.nan, flush, below,
                        -np.finfo(np.float32).tiny, -2.0 ** -149],
                       dtype=np.float32)
    got = fast_exp_neg(special)
    bits = got.view(np.uint32)
    assert bits[0] == bits[1] == 0x3F800000          # +-0 -> exactly 1
    assert bits[2] == 0                               # -inf -> +0
    assert np.isnan(got[3])                           # NaN in, NaN out
    # The threshold itself is kept, as in fast_expf: e^flush is just below
    # FLT_MIN, so its one unflushed input gives a subnormal.
    assert 0 < got[4] < np.finfo(np.float32).tiny, got[4]
    assert bits[5] == 0                               # first input below flushes
    assert bits[6] == bits[7] == 0x3F800000
    # Everything below the threshold, down to -FLT_MAX, is exactly +0.
    rng = np.random.default_rng(3)
    deep = -np.exp(rng.uniform(np.log(-float(below)), np.log(3.4e38), 100_000))
    deep = np.concatenate([deep.astype(np.float32), [below, -np.finfo(np.float32).max]])
    assert np.all(deep < flush)
    assert np.all(fast_exp_neg(deep).view(np.uint32) == 0)
    # Shape is kept for scalars and arrays alike.
    assert fast_exp_neg(np.float32(-1.0)).shape == ()


def test_fast_exp_neg_exhaustive_ulp():
    """Every float32 in [flush, 0] against float64 exp, in units of the exact
    value's ulp (2^(e-23) for 2^e <= |exp(x)| < 2^(e+1), never below 2^-149)."""
    import time

    from executorch.backends.esp32s3.test.models import (
        FAST_EXP_NEG_FLUSH,
        fast_exp_neg_f32_reference as fast_exp_neg,
    )

    start = time.time()
    first = int(np.float32(-0.0).view(np.uint32))
    last = int(np.float32(FAST_EXP_NEG_FLUSH).view(np.uint32))
    worst, worst_x, count = 0.0, None, 0
    step = 1 << 21
    for begin in range(first, last + 1, step):
        x = np.arange(begin, min(begin + step, last + 1), dtype=np.uint32).view(np.float32)
        if begin == first:
            x = np.concatenate([np.float32([0.0]), x])
        exact = np.exp(x.astype(np.float64))
        _, exponent = np.frexp(exact)
        ulp = np.ldexp(1.0, np.maximum(exponent - 24, -149))
        err = np.abs(fast_exp_neg(x).astype(np.float64) - exact) / ulp
        index = int(np.argmax(err))
        if err[index] > worst:
            worst, worst_x = float(err[index]), float(x[index])
        count += x.size
    print(f"  fast_exp_neg: {count:,} float32 inputs in [{FAST_EXP_NEG_FLUSH}, 0], "
          f"max error {worst:.6f} ulp at x={worst_x!r} ({time.time() - start:.0f} s)")
    assert count == (last - first + 1) + 1
    assert worst <= 1.004, (worst, worst_x)


def _random_tail(rows=8, width=37, columns=6, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((1, rows, width)).astype(np.float32) * 6
    mask = rng.random(width) < 0.7
    mask[3] = True
    values = rng.standard_normal((1, width, columns)).astype(np.float32)
    return scores, mask, values


def test_reference_tracks_the_op():
    from executorch.backends.esp32s3._passes.fuse_attn_tail_pass import ATTN_TAIL
    from executorch.backends.esp32s3.test.models import attn_tail_f32_reference

    scores, mask, values = _random_tail()
    probs, out = attn_tail_f32_reference(scores, mask, 0.17677669, values)
    eager = ATTN_TAIL(torch.from_numpy(scores), torch.from_numpy(mask).view(1, 1, -1),
                      0.17677669, torch.from_numpy(values)).numpy()
    assert np.allclose(probs.sum(-1), 1.0, atol=1e-6)
    assert np.all(probs[..., ~mask] == 0)
    rel = np.abs(out - eager).max() / np.abs(eager).max()
    print(f"  device arithmetic vs eager softmax/bmm: {rel:.2e} relative")
    assert 0 < rel < 1e-5      # close, and NOT identical: fast_exp_neg, reciprocal


def test_vectorised_reference_is_the_stepwise_loop():
    """attn_tail_softmax_pv_f32_reference vectorises the loop that defines it;
    every bit must survive, NaN rows and double-rounding ties included."""
    from executorch.backends.esp32s3.test.models import (
        _attn_tail_softmax_pv_f32_stepwise,
        _pv_fma_chain_f32,
        attn_tail_softmax_pv_f32_reference,
    )
    from executorch.backends.esp32s3.quantizer.mirrors import fma_f32

    def same(a, b):
        return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))

    rng = np.random.default_rng(7)
    cases = 0
    # Decode (one query row per head) and prefill (a block of rows) shapes,
    # masked lanes, and scales from tiny to overflowing.
    # (8, 16, 40, 32) is past the chain's lane limit: the vectorised max and
    # sum ahead of the fma_f32 loop.
    for batches, rows, width, columns in ((8, 1, 64, 32), (8, 1, 256, 32), (2, 32, 96, 32),
                                          (8, 16, 40, 32), (1, 3, 1, 5), (3, 4, 37, 6)):
        for scale in (1e-30, 1e-3, 1.0, 30.0, 1e30):
            masked = (rng.standard_normal((batches, rows, width)) * scale).astype(np.float32)
            masked[rng.random(masked.shape) < 0.3] = -np.inf
            values = (rng.standard_normal((batches, width, columns)) * scale).astype(np.float32)
            got = attn_tail_softmax_pv_f32_reference(masked, values)
            want = _attn_tail_softmax_pv_f32_stepwise(masked, values)
            assert same(got[0], want[0]) and same(got[1], want[1]), (batches, rows, width, scale)
            cases += 1
    # Signed-zero maxima (the max keeps the LAST of equal values), a fully
    # masked row (NaN), and a NaN score (the loop's NaN rule).
    masked = np.array([[[-0.0, 0.0, -1.0, -0.0], [0.0, -0.0, -np.inf, -2.0],
                        [-np.inf] * 4, [1.0, np.nan, 0.5, 2.0]]], dtype=np.float32)
    values = rng.standard_normal((1, 4, 3)).astype(np.float32)
    got = attn_tail_softmax_pv_f32_reference(masked, values)
    want = _attn_tail_softmax_pv_f32_stepwise(masked, values)
    assert same(got[0], want[0]) and same(got[1], want[1])
    # The tie a float64 sum gets wrong: acc = 1 + 2^-23, then p*v = 2^-24 - 2^-70
    # puts the float64 sum exactly on the midpoint 1 + 3*2^-24, and ties-to-even
    # would round UP where the single rounding goes down. The chain must refuse
    # (None), and the reference must still equal the loop.
    p = np.array([[[1.0, 1.0 + 2.0 ** -23]]], dtype=np.float32)
    v = np.array([[[1.0 + 2.0 ** -23], [2.0 ** -24 * (1.0 - 2.0 ** -23)]]], dtype=np.float32)
    assert _pv_fma_chain_f32(p, v) is None
    acc = np.zeros((1, 1, 1), dtype=np.float32)
    for k in range(2):
        acc = fma_f32(p[:, :, k:k + 1], v[:, k:k + 1, :], acc)
    assert acc.item() == np.float32(1.0 + 2.0 ** -23)
    naive = np.float32(np.float64(np.float32(1.0 + 2.0 ** -23))
                       + np.float64(p[0, 0, 1]) * np.float64(v[0, 1, 0]))
    assert naive != acc.item()  # so the refusal is load-bearing
    print(f"  vectorised attn_tail reference == stepwise loop on {cases + 1} inputs, "
          f"and refuses the double-rounding tie")


def test_fully_masked_row_is_nan():
    """Documented semantics: the replaced graph gives NaN, and so does the op."""
    from executorch.backends.esp32s3._passes.fuse_attn_tail_pass import ATTN_TAIL
    from executorch.backends.esp32s3.test.models import attn_tail_f32_reference

    scores, _, values = _random_tail(width=5)
    mask = np.zeros(5, dtype=bool)
    eager = ATTN_TAIL(torch.from_numpy(scores), torch.from_numpy(mask), 0.5,
                      torch.from_numpy(values))
    graph = torch.bmm(torch.softmax(torch.where(
        torch.from_numpy(mask), torch.from_numpy(scores) * 0.5,
        torch.tensor(float("-inf"))), -1), torch.from_numpy(values))
    _, mirror = attn_tail_f32_reference(scores, mask, 0.5, values)
    assert torch.isnan(graph).all() and torch.isnan(eager).all()
    assert np.isnan(mirror).all()


# ---- the pass on small graphs -----------------------------------------------
class Tail(nn.Module):
    """The chain as glimmer_decode.py writes it, with one knob per near-miss."""

    def __init__(self, dim=-1, fill=float("-inf"), per_head_mask=False,
                 leak=False, swap=False):
        super().__init__()
        self.dim, self.fill = dim, fill
        self.per_head_mask, self.leak, self.swap = per_head_mask, leak, swap

    def forward(self, scores, values, pos):
        width = scores.shape[-1]
        valid = (torch.arange(width) <= pos).view(1, 1, width)
        if self.per_head_mask:
            valid = valid & (torch.arange(scores.shape[1]) <= pos).view(1, -1, 1)
        fill = torch.full((), self.fill, dtype=torch.float32)
        s = scores * 0.125
        if self.swap:
            s = torch.where(~valid, fill, s)
        else:
            s = torch.where(valid, s, fill)
        probs = torch.softmax(s, dim=self.dim)
        out = torch.bmm(probs, values)
        return (out, s.amax()) if self.leak else out


def _tail_inputs():
    torch.manual_seed(0)
    return (torch.randn(1, 4, 6), torch.randn(1, 6, 5), torch.tensor(3))


def _fuse_small(module, expect=None):
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.fuse_attn_tail_pass import FuseAttnTailPass
    from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import constant_map

    args = _tail_inputs()
    edge = to_edge(torch.export.export(module.eval(), args))
    ep = edge.exported_program()
    with torch.no_grad():
        before = ep.module()(*args)
    p = FuseAttnTailPass(constants=constant_map(ep), expect=expect)
    p.call(ep.graph_module)
    with torch.no_grad():
        after = ep.module()(*args)
    return p, before, after


def test_exact_chain_fuses_bit_identically():
    p, before, after = _fuse_small(Tail(), expect=1)
    assert p.fused == 1 and not p.rejected
    assert torch.equal(before, after)


def test_near_misses_are_refused_with_reasons():
    cases = {
        "softmax over the heads": (Tail(dim=1), "not the last axis"),
        "finite masked fill": (Tail(fill=-1e9), "not -inf"),
        "per-head mask": (Tail(per_head_mask=True), "single mask row"),
        "intermediate escapes": (Tail(leak=True), "escapes the pattern"),
        "fill selected by the mask": (Tail(swap=True), "not a constant"),
    }
    for label, (module, reason) in cases.items():
        p, before, after = _fuse_small(module)
        got = [why for _, why in p.rejected]
        assert p.fused == 0, label
        assert any(reason in why for why in got), (label, got)
        before = before[0] if isinstance(before, tuple) else before
        after = after[0] if isinstance(after, tuple) else after
        # Untouched graphs, so exactly equal -- NaN included: a softmax over
        # the heads is all-NaN in every column the mask hides from all heads.
        torch.testing.assert_close(after, before, rtol=0, atol=0,
                                   equal_nan=True, msg=label)

    try:
        _fuse_small(Tail(fill=-1e9), expect=1)
    except RuntimeError as error:
        assert "expected 1 sites, fused 0" in str(error)
    else:
        raise AssertionError("expect=1 did not catch an unfused site")


# ---- the tiny decode step ---------------------------------------------------
def _decode(local_ring):
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import build

    cfg, model = build(arm="ple", vocab=128, d_model=64, layers=2, heads=8,
                       kv_heads=1, ffn=128, ple_dim=16, seq_len=8, window=4,
                       local_ring=local_ring)
    assert sum(p.numel() for p in model.parameters()) < 200_000
    return cfg, model


def _steps(module, steps=6):
    # ExportedProgram.module() shares its buffers with the program, so a second
    # run would start from the first run's KV state; start every run empty.
    with torch.no_grad():
        for name, buffer in module.named_buffers():
            if "cache_" in name:
                buffer.zero_()
        return [module(torch.tensor([(7 * t + 3) % 128]), torch.tensor(t))
                for t in range(steps)]


def test_tiny_decode_step_fuses_every_layer():
    import contextlib
    import io

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    for local_ring in (True, False):
        cfg, model = _decode(local_ring)
        eager = _steps(model)
        _, fresh = _decode(local_ring)      # zero caches for the export
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(
                torch.export.export(fresh, (torch.tensor([3]), torch.tensor(0))),
                [Esp32Partitioner(format_policy="i8", quantize_qk=True,
                                  group_weights=False)])
        unfused = _steps(edge.exported_program().module())

        rep = apply_passes(edge, quiet=True, attn_tail_expect=cfg.n_layers)
        assert rep["attn_tail_fused"] == cfg.n_layers == 2
        gm = edge.exported_program().graph_module
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        assert sum("espressif.attn_tail" in t for t in targets) == 2
        for gone in ("aten._softmax.", "aten.where.", "aten.bmm."):
            assert not any(gone in t for t in targets), (gone, targets)

        fused = _steps(edge.exported_program().module())
        for step, (a, b, e) in enumerate(zip(unfused, fused, eager)):
            assert torch.equal(a, b), f"ring={local_ring} step {step} moved"
            assert torch.allclose(b, e, rtol=1e-5, atol=1e-5), step

        program = edge.to_executorch()
        operators = {op.name + (f".{op.overload}" if op.overload else "")
                     for plan in program.executorch_program.execution_plan
                     for op in plan.operators}
        assert "espressif::attn_tail.out" in operators, operators
        assert not operators & {"aten::_softmax.out", "aten::where.self_out",
                                "aten::bmm.out"}, operators
        print(f"  ring={local_ring}: {rep['attn_tail_fused']} tails fused, "
              f"bit-identical over {len(fused)} steps; emitted {sorted(operators)}")


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as error:  # noqa: BLE001 -- report every failure
            failed.append(name)
            print(f"  FAIL {name}: {type(error).__name__}: {error}")
    print("FAILED: " + ", ".join(failed) if failed else "all attn_tail checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
