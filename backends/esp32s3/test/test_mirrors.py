# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""quantizer/mirrors.py against torch's own quantisation ops, bit for bit.

The mirrors are the references the runtime's asymmetric quantisation must
match exactly, so each is held to its source of truth -- codes, scales, zero
points and outputs bit-exact -- on the inputs where implementations part ways:

  1. torch's float32 reciprocal is correctly rounded (every quantiser
     multiplies by fl(1/s)), as are the divisions the choosers make, over all
     2^32 float32 inputs;
  2. the activation choosers and quantisers A0-A7 and uint8 model IO against
     quantized_decomposed and torchao: exact ties with odd and even zero
     points, all-positive / all-negative / constant / all-zero rows, each eps
     floor (2^-12, 2^-16, FLT_EPSILON, FLT_MIN), codes reaching -128;
  3. weight dequantisation W-sym, W-zp, W-off (tinygemm, GGUF), the unsigned
     normalisation and int32 bias folding against torch's dequantize ops;
  4. the corrected integer dot against an exact integer matmul: -128 x -128,
     ragged groups, the fused-int4 spelling, K at each exactness bound and one
     group past it; the float epilogue against a torch replay;
  5. conv, grouped, depthwise and transposed im2col with padding written as
     zx against torch's own convolutions over exact integers, with padding at
     every edge, stride and dilation, and a transposed conv's output padding;
  6. embeddings against quantized_decomposed.embedding_{byte,2bit,4bit}, and
     ExecuTorch's nibble order against the backend's packer;
  7. the asymmetric K cache against QuantizedKVCache's ops;
  8. ExecuTorch's portable choose_qparams: the transcription against the
     kernel itself (compiled from the pinned source), and inputs where it
     disagrees with PyTorch's choose_qparams.tensor, with both answers;
  9. models.py's link to the module.

Every check is printed and a failed one recorded, so a script run reports
them all and exits 1 if any failed; under pytest each section is a test_*
that fails naming its failed checks (@case), as in test_partition_crashes.py.
QUANT_MIRRORS_SKIP_SWEEP=1 skips section 1's 2^32 sweep in either.

    python test_mirrors.py
"""

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace

import numpy as np
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  (registers the ops)

from executorch.backends.esp32s3.quantizer import mirrors as qm

FAILS = []
SKIPS = []
QD = torch.ops.quantized_decomposed
F32 = np.float32


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def case(section):
    """A section as a pytest case. check() records a failure and goes on, so
    a script run reports every one (main); a section that only did that
    would pass under pytest whatever its checks found. Called, it fails if
    any check it made failed, naming them; main() runs the section itself."""

    @functools.wraps(section)
    def run():
        start = len(FAILS)
        section()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


def same(a, b):
    """Bit-identical values (and shapes); float arrays compare their bits."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        if a.dtype != b.dtype:
            return False
        width = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
        return np.array_equal(a.view(width), b.view(width))
    return np.array_equal(a.astype(np.int64), b.astype(np.int64))


def t(x):
    return torch.from_numpy(np.ascontiguousarray(x))


# --- 1. reciprocal and division ------------------------------------------------

def reciprocal_and_division():
    print("torch's float32 reciprocal and chooser divisions, all 2^32 inputs:")
    chunk = 1 << 24
    names = ("Tensor.reciprocal", "1.0 / tensor", "numpy 1/x",
             "tensor / 255", "tensor / 127.5", "tensor / 127")
    bad = dict.fromkeys(names, 0)
    first = {}
    base = np.arange(chunk, dtype=np.uint32)
    for start in range(0, 1 << 32, chunk):
        x = (base + np.uint32(start)).view(F32)
        tx = t(x)
        with np.errstate(all="ignore"):       # NaNs, infs and overflow, as meant
            x64 = x.astype(np.float64)
            # The float64 quotient of float32 operands rounds to float32
            # correctly (53 >= 2*24 + 2), so this is the exact reference.
            ref = (1.0 / x64).astype(F32)
            got = {"Tensor.reciprocal": tx.reciprocal().numpy(),
                   "1.0 / tensor": (1.0 / tx).numpy(),
                   "numpy 1/x": F32(1.0) / x}
            refs = dict.fromkeys(got, ref)
            for d in (255.0, 127.5, 127.0):
                name = f"tensor / {d:g}"
                got[name] = (tx / d).numpy()
                refs[name] = (x64 / d).astype(F32)
        for name in names:
            r, g = refs[name], got[name]
            nan = np.isnan(r)
            wrong = (r.view(np.uint32) != g.view(np.uint32)) & ~nan
            wrong |= nan != np.isnan(g)
            count = int(wrong.sum())
            if count and name not in first:
                first[name] = float(x[np.argmax(wrong)])
            bad[name] += count
    for name in names:
        check(f"{name} is correctly rounded on every float32",
              bad[name] == 0,
              f"{bad[name]} wrong" + (f", first at {first[name]!r}"
                                      if name in first else ""))
    # The zero-point division lo / scale is tensor / tensor.
    rng = np.random.default_rng(1)
    a = (rng.standard_normal(1 << 22) * 2.0 ** rng.integers(
        -60, 60, 1 << 22)).astype(F32)
    b = (rng.random(1 << 22) * 2.0 ** rng.integers(
        -60, 60, 1 << 22)).astype(F32) + F32(2.0 ** -70)
    with np.errstate(over="ignore"):          # quotients past float32's range
        ref = (a.astype(np.float64) / b.astype(np.float64)).astype(F32)
    check("tensor / tensor is correctly rounded (4M random pairs)",
          same((t(a) / t(b)).numpy(), ref))


# --- 2. activations -----------------------------------------------------------

def adversarial_rows(k=32, seed=0):
    """Rows (k values each) on every edge the choosers and quantisers have."""
    rng = np.random.default_rng(seed)
    step = 2.0 ** -4
    ties = np.asarray([2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 1.5, -1.5]) * step
    near = np.asarray([np.nextafter(F32(2.5 * step), F32(1)),
                       np.nextafter(F32(2.5 * step), F32(-1)),
                       np.nextafter(F32(-0.5 * step), F32(1)),
                       np.nextafter(F32(-0.5 * step), F32(-1))])
    rows = {
        # scale 2^-4 exactly, zero point 0 (affine and XNNPACK alike): every
        # tie below is an exact half after the multiply by 16.
        "tie, even zp": [-8.0, 7.9375, *ties, *near],
        # scale 2^-4, zero point 1: rounding after the zero point (A4) sends
        # each tie the other way from rounding before it (A2/A3).
        "tie, odd zp": [-8.0625, 7.875, *ties, *near],
        # lo / scale = -100.5 exactly: the zero point itself is a tie.
        "tie in the zero point": [-6.28125, 9.65625, *ties],
        # amax 127 * 2^-3: A0/A1 scale 0.125, ties at odd multiples of 1/16.
        "symmetric tie": [15.875, -15.875, 0.3125, -0.3125, 0.0625, -0.1875],
        # A7 over [-128, 127]: scale 0.125 and -amax -> -127.5 -> -128.
        "reaches -128": [-15.9375, 7.0, 0.3125, -0.3125, 15.9375],
        "all positive": rng.uniform(0.1, 3.0, k),
        "all negative": rng.uniform(-3.0, -0.1, k),
        "constant positive": [0.7],
        "constant negative": [-1.3],
        "all zero": [0.0],
        "signed zeros": [-0.0, 0.0, 1.0, -0.5],
        "huge": rng.standard_normal(k) * 1e30,
        "tiny": rng.standard_normal(k) * 1e-30,
        "normal": rng.standard_normal(k),
        "normal x1e-3": rng.standard_normal(k) * 1e-3,
        "normal x1e3": rng.standard_normal(k) * 1e3,
        "one outlier": [*(rng.standard_normal(k - 1) * 0.01), 50.0],
    }
    # Each eps floor binds on a row whose range is under 255 * eps, and only
    # just fails to on one whose range is a little over it.
    for name, eps in (("2^-12", qm.EPS_XNNPACK), ("2^-16", qm.EPS_ARM),
                      ("FLT_EPSILON", qm.FLT_EPSILON),
                      ("FLT_MIN", qm.FLT_MIN)):
        rows[f"under the {name} floor"] = np.asarray(
            [0.0, 3.25, -7.5, 100.0, 37.0]) * eps
        rows[f"over the {name} floor"] = np.asarray([-100.0, 155.25]) * eps
    return {name: np.resize(np.asarray(v, dtype=F32), k)
            for name, v in rows.items()}


def torch_scheme(x2d, kind, eps=None):
    """(scale float32 [M], zp int64 [M], codes int64 [M, K]) from torch's own
    ops for one activation scheme."""
    from torchao.quantization.quant_primitives import (
        MappingType, choose_qparams_affine, quantize_affine)

    tx = t(x2d)
    m, k = x2d.shape
    if kind in ("A1", "A2", "A7/pt2e", "uint8"):
        qmin, qmax, dtype = {"A1": (-127, 127, torch.int8),
                             "A2": (-128, 127, torch.int8),
                             "A7/pt2e": (-128, 127, torch.int8),
                             "uint8": (0, 255, torch.uint8)}[kind]
        chooser = (QD.choose_qparams_symmetric.tensor
                   if kind in ("A1", "A7/pt2e") else QD.choose_qparams.tensor)
        s, z = chooser(tx, qmin, qmax, eps, dtype)
        q = QD.quantize_per_tensor.tensor(tx, s, z, qmin, qmax, dtype)
        # float64 out, holding a float32 value: check it, then compare as one.
        assert float(s) == float(F32(float(s)))
        return (np.full(m, float(s), dtype=F32),
                np.full(m, int(z), dtype=np.int64), q.numpy().astype(np.int64))
    if kind == "A4":
        s, z = QD.choose_qparams_per_token_asymmetric(tx, torch.int8)
        q = QD.quantize_per_token(tx, s, z, -128, 127, torch.int8)
        assert torch.equal(s, s.float().double())
        return (s.float().numpy().reshape(m), z.numpy().reshape(m),
                q.numpy().astype(np.int64))
    block = (1, k)
    if kind == "A3/torchao":
        s, z = choose_qparams_affine(tx, MappingType.ASYMMETRIC, block,
                                     torch.int8, -128, 127,
                                     zero_point_dtype=torch.int8)
    elif kind == "A3/legacy":
        s, z = choose_qparams_affine(tx, MappingType.ASYMMETRIC, block,
                                     torch.int8, -128, 127,
                                     eps=qm.FLT_EPSILON,
                                     scale_dtype=torch.float32,
                                     zero_point_dtype=torch.float32)
    elif kind == "A7/torchao":
        s, z = choose_qparams_affine(tx, MappingType.SYMMETRIC, block,
                                     torch.int8, -128, 127,
                                     zero_point_dtype=torch.int8)
    else:
        raise ValueError(kind)
    q = quantize_affine(tx, block, s, z, torch.int8, -128, 127)
    return (s.numpy().astype(F32).reshape(m), z.numpy().astype(
        np.int64).reshape(m), q.numpy().astype(np.int64))


def activations():
    print("activation choosers and quantisers against torch's ops:")
    rows = adversarial_rows()
    names = list(rows)
    stack = np.stack([rows[n] for n in names])
    per_token = (("A3/torchao", qm.A3_TORCHAO), ("A3/legacy", qm.A3_LEGACY),
                 ("A4", qm.A4), ("A7/torchao", qm.A7_TORCHAO))
    per_tensor = (("A1", qm.A1, qm.FLT_EPSILON),
                  ("A2", qm.A2_XNNPACK, qm.EPS_XNNPACK),
                  ("A2", qm.A2_ARM, qm.EPS_ARM),
                  ("A2", replace(qm.A2_XNNPACK, name="A2/FLT_EPSILON",
                                 eps=qm.FLT_EPSILON), qm.FLT_EPSILON),
                  ("A2", replace(qm.A2_XNNPACK, name="A2/FLT_MIN",
                                 eps=qm.FLT_MIN), qm.FLT_MIN),
                  ("A7/pt2e", qm.ActScheme(
                      "A7/pt2e", "symmetric", "tensor", "rne_before_zp",
                      -128, 127, eps=qm.FLT_EPSILON), qm.FLT_EPSILON))
    for kind, scheme in per_token:
        ts, tz, tq = torch_scheme(stack, kind)
        q, s, z = qm.quantize_activation(stack, scheme)
        bad = [names[i] for i in range(len(names))
               if not (same(s[i], ts[i]) and z[i] == tz[i]
                       and same(q[i], tq[i]))]
        check(f"{scheme.name} per token: scales, zero points, codes exact on "
              f"{len(names)} rows", not bad, ", ".join(bad))
    for kind, scheme, eps in per_tensor:
        bad = []
        # Every row as its own tensor, then the whole stack as one.
        for i, name in enumerate(names + ["the stacked tensor"]):
            x = stack[i:i + 1] if i < len(names) else stack
            ts, tz, tq = torch_scheme(x, kind, eps)
            q, s, z = qm.quantize_activation(x, scheme)
            if not (same(s, ts) and same(z, tz) and same(q, tq)):
                bad.append(name)
        check(f"{scheme.name} (eps {eps:.3g}) per tensor: exact on every row "
              f"and the stack", not bad, ", ".join(bad))

    # Positive controls: the rows really sit on the edges they are named for.
    odd = rows["tie, odd zp"]
    # x * 16 is exact (scale 2^-4); its fraction is taken in float64, where it
    # is exact too -- in float32, -0.5 + 2^-25 minus its floor rounds to 0.5.
    v = (odd * F32(16)).astype(np.float64)
    exact_ties = int(np.sum(v - np.floor(v) == 0.5))
    q2, _, z2 = qm.quantize_activation(odd[None], qm.A3_TORCHAO)
    q4, _, z4 = qm.quantize_activation(odd[None], qm.A4)
    check("the odd-zp row has zero point 1 under both conventions",
          z2[0] == 1 and z4[0] == 1, f"A3 {z2[0]}, A4 {z4[0]}")
    check("odd zp: rounding after the zero point moves every exact tie and "
          "nothing else", int(np.sum(q2 != q4)) == exact_ties > 0,
          f"{int(np.sum(q2 != q4))} codes differ, {exact_ties} exact ties")
    q2e, _, _ = qm.quantize_activation(rows["tie, even zp"][None], qm.A3_TORCHAO)
    q4e, _, _ = qm.quantize_activation(rows["tie, even zp"][None], qm.A4)
    check("even zp: the two conventions agree on the same ties",
          same(q2e, q4e))
    _, s7, _ = qm.quantize_activation(rows["reaches -128"][None], qm.A7_TORCHAO)
    q7, _, _ = qm.quantize_activation(rows["reaches -128"][None], qm.A7_TORCHAO)
    check("A7 over [-128, 127] produces -128", q7.min() == -128,
          f"scale {float(s7[0])}")
    for scheme, eps, label in ((qm.A2_XNNPACK, qm.EPS_XNNPACK, "2^-12"),
                               (qm.A2_ARM, qm.EPS_ARM, "2^-16"),
                               (qm.A4, qm.FLT_EPSILON, "FLT_EPSILON"),
                               (qm.A3_TORCHAO, qm.FLT_MIN, "FLT_MIN")):
        _, under, _ = qm.quantize_activation(
            rows[f"under the {label} floor"][None], scheme)
        _, over, _ = qm.quantize_activation(
            rows[f"over the {label} floor"][None], scheme)
        check(f"{scheme.name}: the {label} floor binds under it and not over",
              under[0] == F32(eps) and over[0] > F32(eps),
              f"{float(under[0])!r} / {float(over[0])!r}")

    # A0 is the backend's own: hold it to kcache_i8's spelling, which
    # test_kcache_i8.py holds to dynamic_i8_qk, which the device matches.
    from executorch.backends.esp32s3.ops.kcache_i8 import _codes_and_scale

    codes, scale = _codes_and_scale(t(stack))
    s0, _ = qm.choose_absmax(stack)
    q0 = qm.quantize_half_away(stack, s0[:, None])
    check("A0 matches the delegate's quantiser (kcache_i8) on every row",
          same(s0, scale.numpy().reshape(-1))
          and same(q0, codes.numpy().astype(np.int64)))
    sym = rows["symmetric tie"]
    q1, s1, _ = qm.quantize_activation(sym[None], qm.A1)
    q0t = qm.quantize_half_away(sym[None], s1[:, None])
    v = (sym * F32(8)).astype(np.float64)               # exact: scale 1/8
    lower = np.floor(v)
    # Away from zero is the upper neighbour of a positive tie and the lower
    # of a negative one; ties to even disagrees when that neighbour is odd.
    away = int(np.sum((v - lower == 0.5)
                      & np.where(v > 0, lower % 2 == 0, lower % 2 == 1)))
    check("A0 rounds a tie away from zero exactly where A1's even neighbour "
          "is the other one", float(s1[0]) == 0.125
          and int(np.sum(q0t != q1)) == away > 0,
          f"{int(np.sum(q0t != q1))} differ, expected {away}")

    static_schemes()


def static_schemes():
    print("static schemes (A5, A6) and uint8 model IO:")
    rows = adversarial_rows(seed=3)
    x = np.concatenate([v for v in rows.values() if np.abs(v).max() < 1e4])
    x = x.reshape(1, -1)
    cases = ((qm.A5, 2.0 ** -5, 3, torch.int8),
             (qm.A5, float(F32(0.0123)), -17, torch.int8),
             (qm.A5, float(F32(0.73)), 127, torch.int8),
             (qm.A5_UINT8, float(F32(0.02)), 128, torch.uint8),
             (qm.A5_UINT8, float(F32(0.011)), 3, torch.uint8),
             (qm.A6, float(F32(1.7e-4)), 0, torch.int16))
    for scheme, s, z, dtype in cases:
        want = QD.quantize_per_tensor(t(x), s, z, scheme.qmin, scheme.qmax,
                                      dtype).numpy().astype(np.int64)
        got, _, _ = qm.quantize_activation(x, scheme.static(s, z))
        check(f"{scheme.name} static (s={s:.4g}, zp={z}) codes exact",
              same(got, want))
        back = QD.dequantize_per_tensor(t(want.astype(
            {torch.int8: np.int8, torch.uint8: np.uint8,
             torch.int16: np.int16}[dtype])), s, z, scheme.qmin, scheme.qmax,
            dtype).numpy()
        check(f"{scheme.name} static dequantize exact",
              same(qm.dequantize_per_tensor(got, s, z), back))

    # uint8 IO onto the int8 kernels: the -128 shift changes no value.
    s, z = float(F32(0.02)), 131
    q8, _, _ = qm.quantize_activation(x, qm.A5_UINT8.static(s, z))
    qi, _, _ = qm.quantize_activation(x, qm.A5.static(s, z - 128))
    shifted, zs = qm.uint8_to_int8(q8, z)
    check("uint8 codes shifted by -128 are the int8 codes at zp - 128",
          same(shifted, qi) and zs == z - 128)
    check("so (q - zp) is unchanged by the shift", same(q8 - z, qi - (z - 128)))
    for eps in (qm.EPS_XNNPACK, qm.EPS_ARM):
        su, zu = qm.choose_affine(x, 0, 255, eps)
        si, zi = qm.choose_affine(x, -128, 127, eps)
        tsu, tzu = QD.choose_qparams.tensor(t(x), 0, 255, eps, torch.uint8)
        check(f"dynamic uint8 chooser exact and the int8 one shifted "
              f"(eps {eps:g})",
              same(su, F32(float(tsu))) and zu == int(tzu)
              and same(si, su) and zi == zu - 128)

    # Stage A's output fake-quant: re-quantising its output gives the codes.
    y = x * F32(3.1)
    fq = qm.fake_quant_output(y, float(F32(0.05)), -7, -128, 127)
    tq = QD.quantize_per_tensor(t(y), float(F32(0.05)), -7, -128, 127,
                                torch.int8)
    tfq = QD.dequantize_per_tensor(tq, float(F32(0.05)), -7, -128, 127,
                                   torch.int8).numpy()
    requant = QD.quantize_per_tensor(t(fq), float(F32(0.05)), -7, -128, 127,
                                     torch.int8)
    check("output fake-quant equals quantize+dequantize, and re-quantises "
          "to the same codes", same(fq, tfq) and torch.equal(requant, tq))

    # A static scale that is not a float32 value: torch multiplies by the
    # float64 quotient's rounding, not by fl(1 / fl(s)). The mirror follows
    # torch; a runtime storing only fl(s) would not.
    found = next(s for s in (i / 1000.0 for i in range(1, 1000))
                 if F32(1.0 / s) != F32(1.0) / F32(s))
    # Odd multiples of half a step: products land on ties, and which side of
    # one depends on the multiplier's last bit.
    xs = ((np.arange(-128, 128) + 0.5) * F32(found)).astype(F32).reshape(1, -1)
    want = QD.quantize_per_tensor(t(xs), found, 0, -128, 127,
                                  torch.int8).numpy().astype(np.int64)
    got, _, _ = qm.quantize_activation(xs, qm.A5.static(found, 0))
    f32_spelling = qm.quantize_before_zp(xs, F32(found), 0, -128, 127)
    check(f"a float64 static scale ({found}) is mirrored as torch computes it",
          same(got, want))
    check(f"  ... and fl(1/fl(s)) would differ on "
          f"{int(np.sum(f32_spelling != want))} of {xs.size} codes",
          bool(np.any(f32_spelling != want)))


# --- 3. weights -----------------------------------------------------------------

def weights():
    from torchao.quantization.quant_primitives import (
        _dequantize_affine_tinygemm, _dequantize_gguf, dequantize_affine)

    print("weight dequantisation against torch's dequantize ops:")
    rng = np.random.default_rng(2)
    n, k, gsize = 6, 256, 64
    g = k // gsize
    c8 = rng.integers(-128, 128, (n, k)).astype(np.int8)
    c4 = rng.integers(-8, 8, (n, k)).astype(np.int8)
    s_row = (rng.random(n) * 0.05 + 1e-3).astype(F32)
    s_grp = (rng.random((n, g)) * 0.05 + 1e-3).astype(F32)
    z_row = rng.integers(-128, 128, n)
    z_grp = rng.integers(-8, 8, (n, g))

    want = QD.dequantize_per_channel(t(c8), t(s_row), t(np.zeros(n, np.int64)),
                                     0, -128, 127, torch.int8).numpy()
    check("W-sym per channel == dequantize_per_channel",
          same(qm.dequant_w_sym(c8, s_row), want))
    want = QD.dequantize_per_channel(t(c8), t(s_row), t(z_row), 0, -128, 127,
                                     torch.int8).numpy()
    check("W-zp per channel == dequantize_per_channel",
          same(qm.dequant_w_zp(c8, z_row, s_row), want))
    want = QD.dequantize_per_channel_group(
        t(c4), t(s_grp), t(z_grp.astype(np.int32)), -8, 7, torch.int8, gsize,
        torch.float32).numpy()
    check("W-zp per group == dequantize_per_channel_group",
          same(qm.dequant_w_zp(c4, z_grp, s_grp, gsize), want))
    want = dequantize_affine(t(c4), (1, gsize), t(s_grp),
                             t(z_grp.astype(np.int8)), torch.int8, -8,
                             7).numpy()
    check("W-zp per group == torchao dequantize_affine",
          same(qm.dequant_w_zp(c4, z_grp, s_grp, gsize), want))
    zf = rng.uniform(-8, 7, (n, g)).astype(F32)
    want = dequantize_affine(t(c4), (1, gsize), t(s_grp), t(zf), torch.int8,
                             -8, 7).numpy()
    check("a fractional float zero point == dequantize_affine (fl(fl(c-z)*s))",
          same(qm.dequant_w_zp(c4, zf, s_grp, gsize), want))
    # ... which is not W-off's formula with m = -z*s: the two are different
    # roundings, so a fractional zero point is its own scheme, not W-off.
    alt = qm.dequant_w_off_tinygemm(c4.astype(np.int64) + 8, s_grp,
                                    (-zf * s_grp).astype(F32), gsize)
    check("  ... and differs from W-off with m = fl(-z*s) on some elements",
          bool(np.any(alt.view(np.uint32) != want.view(np.uint32))),
          f"{int(np.sum(alt.view(np.uint32) != want.view(np.uint32)))} of "
          f"{alt.size}")

    for bits in (4, 8):
        top = (1 << bits) - 1
        q = rng.integers(0, top + 1, (n, k)).astype(np.uint8)
        zq = rng.integers(0, top + 1, (n, g)).astype(np.uint8)
        want = dequantize_affine(t(q), (1, gsize), t(s_grp), t(zq), torch.uint8,
                                 0, top).numpy()
        c, zw = qm.normalize_unsigned(q, zq, bits)
        check(f"unsigned uint{bits} codes normalised to signed: same values",
              same(qm.dequant_w_zp(c, zw, s_grp, gsize), want)
              and c.min() >= -(1 << (bits - 1)) and c.max() < 1 << (bits - 1))

    q = rng.integers(0, 16, (n, k)).astype(np.uint8)
    m_off = rng.standard_normal((n, g)).astype(F32) * F32(0.1)
    want = _dequantize_affine_tinygemm(t(q), (1, gsize), t(s_grp), t(m_off),
                                       torch.uint4, 0, 15).numpy()
    check("W-off tinygemm == torchao _dequantize_affine_tinygemm",
          same(qm.dequant_w_off_tinygemm(q, s_grp, m_off, gsize), want))

    kg = 512                                     # two GGUF super-blocks
    qg = rng.integers(0, 16, (n, kg)).astype(np.uint8)
    ss = (rng.random((n, kg // 256)) * 0.01).astype(F32)
    sm = (rng.random((n, kg // 256)) * 0.01).astype(F32)
    qs = rng.integers(0, 64, (n, kg // 32)).astype(F32)
    qmn = rng.integers(0, 64, (n, kg // 32)).astype(F32)
    want = _dequantize_gguf(t(qg), [1, 32], torch.uint4, t(ss), t(sm), t(qs),
                            t(qmn), torch.float32).numpy()
    d, mn = qm.gguf_block_params(np.repeat(ss, 8, axis=1), qs,
                                 np.repeat(sm, 8, axis=1), qmn)
    check("W-off GGUF == torchao _dequantize_gguf",
          same(qm.dequant_w_off_gguf(qg, d, mn, 32), want))

    b = rng.integers(-2 ** 31, 2 ** 31, n).astype(np.int32)
    b[:2] = [2 ** 31 - 1, -2 ** 31]
    s_b = (rng.random(n) * 1e-6).astype(F32)
    want = QD.dequantize_per_channel(t(b), t(s_b), t(np.zeros(n, np.int64)), 0,
                                     -2 ** 31, 2 ** 31 - 1,
                                     torch.int32).numpy()
    check("int32 bias fold == dequantize_per_channel (two roundings past 2^24)",
          same(qm.dequantize_int32_bias(b, s_b), want))
    want = QD.dequantize_per_tensor(t(b), float(s_b[0]), 0, -2 ** 31,
                                    2 ** 31 - 1, torch.int32).numpy()
    check("int32 bias fold == dequantize_per_tensor",
          same(qm.dequantize_int32_bias(b, s_b[0]), want))


# --- 4. the corrected integer dot ------------------------------------------------

def torch_exact_dots(xq, zx, codes, zw, spans):
    """sum (xq - zx)(c - zw) per group in float64 matmuls: every partial sum is
    an integer below 2^53, so this is exact whatever the BLAS order."""
    xs = t(np.asarray(xq, np.float64) - np.asarray(zx, np.float64)[:, None])
    out = []
    for gi, (a, b) in enumerate(spans):
        w = np.asarray(codes, np.float64)[:, a:b] - zw[:, gi:gi + 1]
        out.append((xs[:, a:b] @ t(w).T).numpy())
    return np.stack(out, axis=2).astype(np.int64)


def matmul():
    print("the corrected integer dot, acc = raw - zw*X - zx*S':")
    rng = np.random.default_rng(4)
    m, n = 5, 7
    configs = ((8, True, None, 300), (8, True, 128, 300), (8, False, 64, 256),
               (4, True, 32, 200), (4, True, 128, 384), (4, False, 32, 160))
    for bits, has_zw, gsize, k in configs:
        lo, hi = (-128, 128) if bits == 8 else (-8, 8)
        spans = qm.group_spans(k, gsize)
        xq = rng.integers(-128, 128, (m, k))
        xq[0] = -128                                  # -128 x -128 below
        zx = rng.integers(-128, 128, m)
        codes = rng.integers(lo, hi, (n, k))
        codes[0] = lo
        zw = (rng.integers(lo, hi, (n, len(spans))) if has_zw
              else np.zeros((n, len(spans)), np.int64))
        label = (f"int{bits}{' W-zp' if has_zw else ''}, K={k}, "
                 f"groups {[b - a for a, b in spans]}")
        acc, _, _ = qm.corrected_dots(xq, zx, codes, zw if has_zw else None,
                                      gsize)
        exact = torch_exact_dots(xq, zx, codes, zw, spans)
        check(f"{label}: corrected == exact integer matmul", same(acc, exact))
        if bits == 4:
            fused, _, _ = qm.corrected_dots(xq, zx, codes,
                                            zw if has_zw else None, gsize,
                                            i4_fused=True)
            check(f"{label}: the fused-int4 spelling (nibbles, (8+zw)*X) agrees",
                  same(fused, exact))
    # A ragged last group as the device runs it: the kernel dots k_pad codes
    # (a zero tail on both sides, so raw and X are unchanged) while S' was
    # summed at export over the group's REAL elements only.
    k, k_pad = 200, 224
    xq = rng.integers(-128, 128, (m, k))
    codes = rng.integers(-8, 8, (n, k))
    zw = rng.integers(-8, 8, (n, 2))
    zx = rng.integers(-128, 128, m)
    acc, _, _ = qm.corrected_dots(xq, zx, codes, zw, 128)
    exact = torch_exact_dots(xq, zx, codes, zw, [(0, 128), (128, k)])
    xs = np.pad(xq, ((0, 0), (0, k_pad - k)))
    cs = np.pad(codes, ((0, 0), (0, k_pad - k)))
    raw = np.stack([qm.wrap_i32(xs[:, a:b] @ cs[:, a:b].T)
                    for a, b in ((0, 128), (128, k_pad))], axis=2)
    X = np.stack([xs[:, :128].sum(1), xs[:, 128:].sum(1)], axis=1)
    s_real = qm.weight_sums(codes, zw, 128)
    device = qm.wrap_i32(raw - zw[None] * X[:, None] - zx[:, None, None]
                         * s_real[None])
    check("ragged last group (72 of 96 padded): exact, S' over real elements",
          same(acc, exact) and same(device, exact))
    tail_counted, _, _ = qm.corrected_dots(xs, zx, cs, zw, 128)
    check("  ... and an S' that counted the k_pad tail would not be",
          not same(tail_counted, exact))

    # -128 x -128 with zero points -128: every term is 0 while raw overflows
    # int32 -- modular corrections recover the small true value.
    k = 140_000
    acc, _, _ = qm.corrected_dots(np.full((1, k), -128), [-128],
                                  np.full((1, k), -128), [[-128]], None)
    raw = k * 128 * 128
    check("-128 x -128 at zx = zw = -128: raw wraps, the corrected dot is 0",
          raw > qm.INT32_MAX and int(acc[0, 0, 0]) == 0, f"raw {raw:,}")

    bounds()
    epilogue()


def bounds():
    print("exactness bounds: K at the bound, then one group of 32 past it:")
    cases = (
        # (label, act span, weight span, xq, zx, code, zw)
        ("symmetric +-127 x int8 (today)", 127, 128, -127, 0, -128, None),
        ("symmetric reaching -128 x int8", 128, 128, -128, 0, -128, None),
        ("affine activation x symmetric int8", 255, 128, -128, 127, -128, None),
        ("affine x affine int8", 255, 255, -128, 127, 127, -128),
    )
    expected = {"symmetric +-127 x int8 (today)": 132_096,
                "symmetric reaching -128 x int8": 131_040,
                "affine activation x symmetric int8": 65_792,
                "affine x affine int8": 33_024}
    for label, aspan, wspan, xv, zx, cv, zw in cases:
        bound = qm.dot_length_bound(aspan, wspan)
        check(f"{label}: bound {bound:,}", bound == expected[label])
        for k, fits in ((bound, True), (bound + 32, False)):
            xq = np.full((1, k), xv)
            codes = np.full((1, k), cv)
            zws = None if zw is None else [[zw]]
            exact = int(qm.exact_dots(xq, [zx], codes, zws)[0, 0, 0])
            acc, _, _ = qm.corrected_dots(xq, [zx], codes, zws)
            in_range = -2 ** 31 <= exact <= qm.INT32_MAX
            check(f"  K={k:,}: worst case {exact:,} "
                  f"{'fits' if fits else 'overflows'} int32 and the "
                  f"wrapped dot {'equals' if fits else 'differs from'} it",
                  in_range == fits and (int(acc[0, 0, 0]) == exact) == fits)
            try:
                qm.corrected_dots(xq, [zx], codes, zws, bound=bound)
                refused = False
            except qm.ExactnessRefused:
                refused = True
            check(f"  K={k:,}: {'accepted' if fits else 'refused'}",
                  refused != fits)
    # int4: the math allows 561k, the fused kernel's 65,536 binds first.
    check("affine x int4: bound 561,408", qm.dot_length_bound(255, 15) == 561_408)
    xq = np.full((1, qm.FUSED_I4_MAX_GROUP + 32), -128)
    codes = np.full((1, xq.shape[1]), 7)
    try:
        qm.corrected_dots(xq, [127], codes, [[-8]], i4_fused=True)
        refused = False
    except qm.ExactnessRefused:
        refused = True
    acc, _, _ = qm.corrected_dots(xq[:, :qm.FUSED_I4_MAX_GROUP], [127],
                                  codes[:, :qm.FUSED_I4_MAX_GROUP], [[-8]],
                                  i4_fused=True)
    check("fused int4: a 65,536 group is exact, one of 65,568 refused",
          refused and int(acc[0, 0, 0]) == -255 * 15 * qm.FUSED_I4_MAX_GROUP)


def epilogue():
    print("the float epilogue against a torch replay of the device's order:")
    rng = np.random.default_rng(5)
    m, n, k, gsize = 4, 6, 96, 32
    x = (rng.standard_normal((m, k)) * 2).astype(F32)
    bias = rng.standard_normal(n).astype(F32)
    g = k // gsize
    for label, bits, codes, scales, zw, off, act in (
            ("A3 x int4 W-zp groups", 4, rng.integers(-8, 8, (n, k)),
             (rng.random((n, g)) * 0.1).astype(F32),
             rng.integers(-8, 8, (n, g)), None, qm.A3_TORCHAO),
            ("A2 x int8 per channel", 8, rng.integers(-128, 128, (n, k)),
             (rng.random(n) * 0.01).astype(F32), None, None, qm.A2_XNNPACK),
            ("A4 x int8 W-zp", 8, rng.integers(-128, 128, (n, k)),
             (rng.random(n) * 0.01).astype(F32), rng.integers(-128, 128, n),
             None, qm.A4),
            ("A3 x int4 W-off (tinygemm)", 4, rng.integers(-8, 8, (n, k)),
             (rng.random((n, g)) * 0.1).astype(F32), None,
             rng.standard_normal((n, g)).astype(F32) * F32(0.1),
             qm.A3_TORCHAO),
            ("A3 x int4 W-off (GGUF, zw = -8)", 4, rng.integers(-8, 8, (n, k)),
             (rng.random((n, g)) * 0.1).astype(F32), np.full((n, g), -8),
             rng.standard_normal((n, g)).astype(F32) * F32(0.1),
             qm.A3_TORCHAO)):
        gs = gsize if np.ndim(scales) == 2 else None
        tr = qm.affine_matmul(x, codes, scales, bias, act=act, bits=bits,
                              gsize=gs, zero_points=zw, offsets=off)
        spans = qm.group_spans(k, gs)
        sc = qm.expand_groups(scales, len(spans), 1)
        y = t(bias).expand(m, n).clone()
        sx = t(tr.sx)[:, None]
        for gi, (a, b) in enumerate(spans):
            y = y + (t(tr.acc[:, :, gi]).float() * sx) * t(sc[:, gi])[None]
            if off is not None:
                xp = t(tr.X[:, gi] - (b - a) * tr.zx).float()
                y = y + t(qm.expand_groups(off, len(spans), 1)[:, gi])[None] * (
                    xp * t(tr.sx))[:, None]
        check(f"{label}: epilogue == torch replay", same(tr.y, y.numpy()))
        # Sanity against the real-valued product: catches a wrong sign or a
        # wrong term rather than a rounding.
        wz = qm.expand_groups(np.zeros((n, 1)) if zw is None else
                              np.asarray(zw).reshape(n, -1), k, gs)
        w = (np.asarray(codes) - wz) * qm.expand_groups(
            np.asarray(scales, np.float64).reshape(n, -1), k, gs)
        if off is not None:
            w = w + qm.expand_groups(np.asarray(off, np.float64), k, gs)
        xr = (tr.xq - tr.zx[:, None]) * tr.sx.astype(np.float64)[:, None]
        real = xr @ w.T + bias
        rel = np.abs(tr.y - real).max() / np.abs(real).max()
        check(f"{label}: within float rounding of the exact product",
              rel < 1e-5, f"rel {rel:.2e}")
    # The symmetric rows fuse the final multiply-add into one madd.s.
    from executorch.backends.esp32s3.quantizer.mirrors import fma_f32

    codes = rng.integers(-127, 128, (n, k))
    scales = (rng.random(n) * 0.01).astype(F32)
    fused = qm.affine_matmul(x, codes, scales, bias, act=qm.A0,
                             fused_madd=True)
    split = qm.affine_matmul(x, codes, scales, bias, act=qm.A0)
    term = (fused.acc[:, :, 0].astype(F32) * fused.sx[:, None]).astype(F32)
    check("fused_madd: y = fma(fl(fl(acc) * sx), s, bias), one rounding",
          same(fused.y, fma_f32(term, scales[None], bias[None])))
    check("  ... which the two-rounding epilogue differs from somewhere",
          not same(fused.y, split.y),
          f"{int(np.sum(fused.y != split.y))} of {fused.y.size}")


# --- 5. convolution -----------------------------------------------------------

def conv_geom(c_in, c_out, size, k, stride, pad, dil=1, transposed=False,
              groups=1, group_idx=0, rank=2, out_pad=0):
    """A ConvGeom with the output extent torch computes (`out_pad`, a
    transposed conv's output padding, on every spatial axis)."""
    from executorch.backends.esp32s3.operator_support.conv import ConvGeom

    def out_extent(e):
        if transposed:
            return (e - 1) * stride - 2 * pad + dil * (k - 1) + 1 + out_pad
        return (e + 2 * pad - dil * (k - 1) - 1) // stride + 1

    d = h = 1
    w = size[-1]
    kw = k
    kd = kh = 1
    if rank >= 2:
        h, kh = size[-2], k
    if rank == 3:
        d, kd = size[-3], k
    return ConvGeom(
        h_in=h, w_in=w, c_in=c_in, h_out=out_extent(h) if rank >= 2 else 1,
        w_out=out_extent(w), c_out=c_out, kh=kh, kw=kw,
        stride_h=stride if rank >= 2 else 1, stride_w=stride,
        pad_h=pad if rank >= 2 else 0, pad_w=pad,
        dil_h=dil if rank >= 2 else 1, dil_w=dil,
        d_in=d, d_out=out_extent(d) if rank == 3 else 1, kd=kd,
        stride_d=stride if rank == 3 else 1, pad_d=pad if rank == 3 else 0,
        dil_d=dil if rank == 3 else 1, groups=groups, group_idx=group_idx,
        c_out_total=c_out,
        transposed=1 if transposed else 0)


def padded_edges(geom):
    """Which edges of the H/W plane have an out-of-bounds tap."""
    from executorch.backends.esp32s3.operator_support.conv import _axis_index

    edges = set()
    for oh in range(geom.h_out):
        for i in range(geom.kh):
            idx = oh * geom.stride_h - geom.pad_h + i * geom.dil_h
            if _axis_index(geom, oh, i, geom.stride_h, geom.pad_h, geom.dil_h,
                           geom.h_in, geom.kh) is None:
                edges.add("top" if idx < 0 else "bottom")
    for ow in range(geom.w_out):
        for j in range(geom.kw):
            idx = ow * geom.stride_w - geom.pad_w + j * geom.dil_w
            if _axis_index(geom, ow, j, geom.stride_w, geom.pad_w, geom.dil_w,
                           geom.w_in, geom.kw) is None:
                edges.add("left" if idx < 0 else "right")
    return edges


def convolutions():
    import torch.nn.functional as F

    from executorch.backends.esp32s3.test.reference import im2col_reference

    print("conv im2col with padding written as zx, against torch's convs:")
    rng = np.random.default_rng(6)
    act = qm.A2_XNNPACK
    cases = (
        ("conv2d 3x3 s1 p1", 2, dict(c_in=3, c_out=4, size=(7, 6), k=3,
                                     stride=1, pad=1)),
        ("conv2d 3x3 s2 p1", 2, dict(c_in=3, c_out=4, size=(7, 9), k=3,
                                     stride=2, pad=1)),
        ("conv2d 3x3 s1 p2 dil2", 2, dict(c_in=2, c_out=3, size=(6, 7), k=3,
                                          stride=1, pad=2, dil=2)),
        ("conv1d k5 s2 p2", 1, dict(c_in=3, c_out=4, size=(11,), k=5,
                                    stride=2, pad=2)),
        ("conv3d 3x3x3 s1 p1", 3, dict(c_in=2, c_out=3, size=(4, 5, 5), k=3,
                                       stride=1, pad=1)),
    )
    for label, rank, kw in cases:
        geom = conv_geom(rank=rank, **kw)
        shape = (1, kw["c_in"]) + kw["size"]
        x = (rng.standard_normal(shape) + 0.4).astype(F32)
        codes = rng.integers(-128, 128, (geom.c_out, geom.k))
        zw = rng.integers(-128, 128, geom.c_out)
        scales = (rng.random(geom.c_out) * 0.01).astype(F32)
        bias = rng.standard_normal(geom.c_out).astype(F32)
        xq, sx, zx = qm.quantize_activation(x.reshape(1, -1), act)
        zx = int(zx[0])
        xq4 = xq.reshape(qm.as_cdhw(x).shape)
        cols = qm.im2col_codes(xq4, geom, zx)
        acc, _, _ = qm.corrected_dots(cols, np.full(geom.m, zx), codes,
                                      zw[:, None])
        conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
        wt = (codes - zw[:, None]).reshape(
            (geom.c_out, kw["c_in"]) + (kw["k"],) * rank)
        ref = conv(t(xq.reshape(shape) - zx).double(), t(wt).double(),
                   stride=kw["stride"], padding=kw["pad"],
                   dilation=kw.get("dil", 1)).numpy()
        check(f"{label}: corrected dots == torch {conv.__name__} over integers",
              same(acc[:, :, 0].T.reshape(ref.shape[1:]),
                   ref[0].astype(np.int64)))
        if rank >= 2:
            edges = padded_edges(geom)
            check(f"{label}: zx padding reached every edge",
                  edges == {"top", "bottom", "left", "right"},
                  ", ".join(sorted(edges)))
        floats = im2col_reference(t(x), geom).numpy()
        requant = qm.quantize_before_zp(floats, sx[0], zx, act.qmin, act.qmax)
        check(f"{label}: the zx gather == quantising the zero-padded input",
              same(cols, requant))
        y = qm.affine_conv(x, codes, scales, bias, geom, act,
                           zero_points=zw)
        yr = t(bias)[:, None] + (t(acc[:, :, 0].T).float() * float(sx[0])
                                 ) * t(scales)[:, None]
        check(f"{label}: affine_conv's epilogue == torch replay",
              same(y.reshape(geom.c_out, -1), yr.numpy()))

    # Depthwise: every channel its own 1-channel gather, S' per channel.
    c, size = 5, (6, 7)
    geom = conv_geom(c_in=1, c_out=c, size=size, k=3, stride=2, pad=1,
                     groups=c)
    check("depthwise geometry is depthwise", geom.depthwise)
    x = (rng.standard_normal((1, c) + size) - 0.3).astype(F32)
    codes = rng.integers(-128, 128, (c, geom.k))
    zw = rng.integers(-128, 128, c)
    scales = (rng.random(c) * 0.01).astype(F32)
    st = qm.A5.static(float(F32(0.021)), 9)
    y = qm.affine_conv(x, codes, scales, None, geom, st, zero_points=zw)
    xq, sx, _ = qm.quantize_activation(x.reshape(1, -1), st)
    ref = F.conv2d(t(xq.reshape(x.shape) - 9).double(),
                   t((codes - zw[:, None]).reshape(c, 1, 3, 3)).double(),
                   stride=2, padding=1, groups=c).numpy()[0]
    yr = (t(ref.reshape(c, -1)).float() * float(sx[0])) * t(scales)[:, None]
    check("depthwise s2 p1 (A5 static, W-zp) == torch grouped conv over "
          "integers, then the epilogue", same(y.reshape(c, -1), yr.numpy()))

    # A grouped slice reads its own channel block.
    from executorch.backends.esp32s3.operator_support.conv import split_groups

    # admit()'s convention: c_in per group, c_out the total, then split.
    full = conv_geom(c_in=2, c_out=6, size=(5, 5), k=3, stride=1, pad=1,
                     groups=2)
    x = (rng.standard_normal((1, 4, 5, 5))).astype(F32)
    codes = rng.integers(-128, 128, (6, full.k))
    xq, sx, zx = qm.quantize_activation(x.reshape(1, -1), act)
    zx = int(zx[0])
    ref = F.conv2d(t(xq.reshape(x.shape) - zx).double(),
                   t(codes.reshape(6, 2, 3, 3)).double(), padding=1,
                   groups=2).numpy()[0]
    ok = True
    for gi, sub in enumerate(split_groups(full)):
        cols = qm.im2col_codes(xq.reshape(4, 1, 5, 5), sub, zx,
                               channel_offset=gi * sub.c_in)
        acc, _, _ = qm.corrected_dots(cols, np.full(sub.m, zx),
                                      codes[gi * 3:(gi + 1) * 3])
        ok &= same(acc[:, :, 0].T.reshape(3, 5, 5),
                   ref[gi * 3:(gi + 1) * 3].astype(np.int64))
    check("grouped conv slices == torch groups=2 over integers", ok)

    # Transposed: off-lattice taps of the zero-stuffed input are zx too. With
    # output padding (positions past the computed extent, read through the
    # same mapping) and dilation (taps spaced dil apart on the stuffed grid);
    # the last pads by 2 at stride 2, which only its dilation of 3 allows.
    for label, cin, cout, size, k, stride, pad, dil, out_pad in (
            ("convT 3x3 s2 p1", 3, 4, (4, 5), 3, 2, 1, 1, 0),
            ("convT 2x2 s2 p0", 2, 3, (3, 3), 2, 2, 0, 1, 0),
            ("convT 3x3 s2 p1 output_padding 1", 3, 4, (4, 5), 3, 2, 1, 1, 1),
            ("convT 3x3 s1 p1 dil 2", 2, 3, (4, 5), 3, 1, 1, 2, 0),
            ("convT 3x3 s2 p2 dil 3 output_padding 2", 2, 3, (3, 4), 3, 2, 2,
             3, 2)):
        geom = conv_geom(c_in=cin, c_out=cout, size=size, k=k, stride=stride,
                         pad=pad, dil=dil, transposed=True, out_pad=out_pad)
        x = (rng.standard_normal((1, cin) + size) + 0.2).astype(F32)
        wt = rng.integers(-128, 128, (cin, cout, k, k))
        zw = rng.integers(-128, 128, cout)
        dense = np.ascontiguousarray(
            np.flip(wt, axis=(2, 3)).transpose(1, 0, 2, 3)).reshape(cout, -1)
        xq, sx, zx = qm.quantize_activation(x.reshape(1, -1), act)
        zx = int(zx[0])
        cols = qm.im2col_codes(xq.reshape(cin, 1, *size), geom, zx)
        acc, _, _ = qm.corrected_dots(cols, np.full(geom.m, zx), dense,
                                      zw[:, None])
        ref = F.conv_transpose2d(
            t(xq.reshape(x.shape) - zx).double(),
            t(wt - zw[None, :, None, None]).double(), stride=stride,
            padding=pad, output_padding=out_pad, dilation=dil).numpy()[0]
        check(f"{label}: corrected dots == torch conv_transpose2d over "
              f"integers", same(acc[:, :, 0].T.reshape(ref.shape),
                                ref.astype(np.int64)))
        floats = im2col_reference(t(x), geom).numpy()
        check(f"{label}: off-lattice taps gather as zx",
              same(cols, qm.quantize_before_zp(floats, sx[0], zx, -128, 127)))


# --- 6. embeddings --------------------------------------------------------------

def embeddings():
    import executorch.exir.passes._quant_patterns_and_replacements  # noqa: F401

    from executorch.backends.esp32s3.serialization.weight_formats import pack_rows

    print("embeddings against quantized_decomposed.embedding_*:")
    rng = np.random.default_rng(7)
    v, d, g = 11, 64, 4
    idx = np.asarray([0, 3, 3, 10, 7, 1], dtype=np.int64)
    scales = (rng.random((v, g)) * 0.05 + 1e-3).astype(F32)
    for label, zps in (("no zero points", None),
                       ("integer zero points (float dtype)",
                        rng.integers(-4, 4, (v, g)).astype(F32)),
                       ("fractional zero points",
                        rng.uniform(-4, 4, (v, g)).astype(F32))):
        tz = None if zps is None else t(zps)
        c4 = rng.integers(-8, 8, (v, d))
        packed = (((c4[:, 0::2] + 8) << 4) | (c4[:, 1::2] + 8)).astype(np.uint8)
        want = QD.embedding_4bit(t(packed), t(scales), tz, -8, 7,
                                 t(idx)).numpy()
        got = qm.embedding_dequant(qm.unpack_et_4bit(packed), scales, zps, idx)
        check(f"embedding_4bit, {label}", same(got, want))
        c8 = rng.integers(-128, 128, (v, d)).astype(np.int8)
        want = QD.embedding_byte(t(c8), t(scales), tz, -128, 127,
                                 t(idx)).numpy()
        check(f"embedding_byte, {label}",
              same(qm.embedding_dequant(c8, scales, zps, idx), want))
        c2 = rng.integers(-2, 2, (v, d))
        u = (c2 + 2).astype(np.uint8)
        packed2 = (u[:, 0::4] | (u[:, 1::4] << 2) | (u[:, 2::4] << 4)
                   | (u[:, 3::4] << 6)).astype(np.uint8)
        want = QD.embedding_2bit(t(packed2), t(scales), tz, -2, 1,
                                 t(idx)).numpy()
        check(f"embedding_2bit, {label}",
              same(qm.embedding_dequant(qm.unpack_et_2bit(packed2), scales,
                                        zps, idx), want))
    from torchao.quantization.quant_primitives import (
        _dequantize_affine_tinygemm, _dequantize_gguf)

    q = rng.integers(0, 16, (v, 256)).astype(np.uint8)
    s_g = (rng.random((v, 8)) * 0.05 + 1e-3).astype(F32)
    m_g = (rng.standard_normal((v, 8)) * 0.1).astype(F32)
    table = _dequantize_affine_tinygemm(t(q), (1, 32), t(s_g), t(m_g),
                                        torch.uint4, 0, 15)
    check("W-off tinygemm embedding == torchao's dequantised table, gathered",
          same(qm.embedding_w_off(q, s_g, m_g, idx, 32),
               torch.nn.functional.embedding(t(idx), table).numpy()))
    ss, sm = (rng.random((v, 1)) * 0.01).astype(F32), (rng.random(
        (v, 1)) * 0.01).astype(F32)
    qs = rng.integers(0, 64, (v, 8)).astype(F32)
    qmn = rng.integers(0, 64, (v, 8)).astype(F32)
    table = _dequantize_gguf(t(q), [1, 32], torch.uint4, t(ss), t(sm), t(qs),
                             t(qmn), torch.float32)
    dd, mn = qm.gguf_block_params(np.repeat(ss, 8, axis=1), qs,
                                  np.repeat(sm, 8, axis=1), qmn)
    check("W-off GGUF embedding == torchao's dequantised table, gathered",
          same(qm.embedding_w_off(q, dd, mn, idx, 32, gguf=True),
               torch.nn.functional.embedding(t(idx), table).numpy()))
    backend = np.frombuffer(pack_rows(c4.astype(np.int8), d, "i4"),
                            dtype=np.uint8).reshape(v, d // 2)
    check("ExecuTorch's 4-bit table, nibbles swapped, is the backend's packing",
          same(qm.swap_nibbles(packed), backend)
          and same(qm.unpack_backend_i4(backend), c4)
          and same(qm.swap_nibbles(qm.swap_nibbles(packed)), packed))


# --- 7. K cache ---------------------------------------------------------------

def kcache():
    from executorch.backends.esp32s3.ops.kcache_i8 import _codes_and_scale

    print("the asymmetric K cache against QuantizedKVCache's ops:")
    rng = np.random.default_rng(8)
    rows = adversarial_rows(k=32, seed=9)
    k = np.concatenate([np.stack(list(rows.values()))[None, None],
                        (rng.standard_normal((1, 1, 6, 32))).astype(F32)],
                       axis=2)
    k = np.concatenate([k, k[:, :, ::-1] * F32(0.5)], axis=1)  # two KV heads
    s, z = QD.choose_qparams_per_token_asymmetric(t(k), torch.int8)
    q = QD.quantize_per_token(t(k), s, z, -128, 127, torch.int8)
    codes, scales, zps = qm.kcache_write(k)
    check("writer: codes, float32 scales and int8 zero points exact",
          same(codes, q.numpy()) and same(scales, s.float().numpy()[..., 0])
          and same(zps, z.to(torch.int8).numpy()[..., 0]))
    v = QD.dequantize_per_token(t(codes), t(scales[..., None]).double(),
                                t(zps[..., None]).long(), -128, 127,
                                torch.int8, torch.float32).numpy()
    check("V side: (vq - zv) * sv == dequantize_per_token",
          same(qm.vcache_dequant(codes, scales, zps), v))
    qry = (rng.standard_normal((1, 2, 3, 32)) * 1.5).astype(F32)
    qry[0, 0, 0] = 0.0                                 # an all-zero query row
    got = qm.kcache_scores(qry, codes, scales, zps)
    qc, qs = _codes_and_scale(t(qry))
    acc = torch.matmul(qc.double(), (t(codes).double() - t(zps).double()[
        ..., None]).transpose(-1, -2))
    want = (acc.float() * qs) * t(scales).unsqueeze(-2)
    check("scores: fl(fl(fl(sum qq*kq - zk*Q) * sq) * sk) == exact integers "
          "replayed in torch", same(got, want.numpy()))
    real = torch.matmul(qc.double() * qs.double(), t(qm.vcache_dequant(
        codes, scales, zps)).double().transpose(-1, -2))
    rel = float((t(got).double() - real).abs().max() / real.abs().max())
    check("scores track the dequantised keys", rel < 1e-6, f"rel {rel:.2e}")


# --- 8. ExecuTorch's portable choose_qparams --------------------------------------

HARNESS = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define ET_CHECK_MSG(cond, ...) do {{ if (!(cond)) std::abort(); }} while (0)
namespace et_kernel {{
{threshold}
{body}
}}  // namespace et_kernel
int main() {{
  float lo, hi;
  int qmin, qmax;
  while (std::scanf("%a %a %d %d", &lo, &hi, &qmin, &qmax) == 4) {{
    double scale;
    int32_t zero_point;
    et_kernel::calculate_scale_and_zero_point(lo, hi, qmin, qmax, scale,
                                              zero_point);
    std::printf("%a %d\n", scale, (int)zero_point);
  }}
  return 0;
}}
"""


def et_kernel(cases):
    """Run the pinned kernel's own calculate_scale_and_zero_point on
    (lo, hi, qmin, qmax) cases; None when it cannot be built here."""
    # backends/esp32s3/test/ -> the root of the ExecuTorch tree
    ET_ROOT = os.environ.get("ET_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

    src = os.path.join(ET_ROOT, "kernels", "quantized", "cpu",
                       "op_choose_qparams.cpp")
    cxx = shutil.which("c++")
    if not os.path.isfile(src) or cxx is None:
        return None, f"needs {src} (set ET_ROOT) and a C++ compiler"
    text = open(src).read()
    threshold = next(line for line in text.splitlines()
                     if "SMALL_SCALE_THRESHOLD =" in line)
    start = text.index("void calculate_scale_and_zero_point(")
    body = text[start:text.index("void choose_qparams(", start)]
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "et_choose_qparams.cpp")
        exe = os.path.join(tmp, "et_choose_qparams")
        with open(cpp, "w") as f:
            f.write(HARNESS.format(threshold=threshold, body=body))
        subprocess.run([cxx, "-std=c++17", "-O2", "-ffp-contract=off", cpp,
                        "-o", exe], check=True)
        stdin = "".join(f"{float(F32(lo)).hex()} {float(F32(hi)).hex()} "
                        f"{qmin} {qmax}\n" for lo, hi, qmin, qmax in cases)
        out = subprocess.run([exe], input=stdin, capture_output=True,
                             text=True, check=True).stdout.split()
    return [(float.fromhex(out[i]), int(out[i + 1]))
            for i in range(0, len(out), 2)], src


def et_choose_qparams():
    print("ExecuTorch's portable choose_qparams:")
    rng = np.random.default_rng(10)
    cases = []
    for _ in range(1500):
        a, b = (rng.standard_normal(2) * 10.0 ** rng.integers(-8, 4)).astype(F32)
        cases.append((min(a, b), max(a, b), -128, 127))
    for _ in range(300):
        hi = F32(rng.random() * 10.0 ** rng.integers(-6, 3))
        cases.append((F32(0), hi, 0, 255))
        cases.append((-hi, F32(0), -128, 127))
    edge = F32(qm.ET_SMALL_SCALE_THRESHOLD * F32(255))
    for steps in range(-3, 4):
        hi = edge
        for _ in range(abs(steps)):
            hi = np.nextafter(hi, F32(np.sign(steps) * np.inf))
        cases += [(F32(0), hi, -128, 127), (-hi / 2, hi / 2, -128, 127)]
    cases += [(F32(0), F32(0), -128, 127), (F32(0), F32(0), 0, 255),
              (F32(-1e-30), F32(1e-30), -128, 127),
              (F32(-3.4e38), F32(3.4e38), -128, 127)]
    kernel, where = et_kernel(cases)
    if kernel is None:
        print(f"  SKIP the kernel comparison: {where}")
        SKIPS.append("ExecuTorch choose_qparams kernel")
    else:
        mirror = [qm.et_calculate_scale_and_zero_point(*c) for c in cases]
        bad = [i for i, (a, b) in enumerate(zip(mirror, kernel))
               if not (float(a[0]) == b[0] and a[1] == b[1])]
        check(f"the transcription == the kernel compiled from the pinned "
              f"source, on {len(cases)} ranges", not bad,
              f"{len(bad)} differ" + (f", first {cases[bad[0]]}" if bad else ""))

    # The record: the same inputs through PyTorch's choose_qparams.tensor (eps
    # FLT_EPSILON) and through the portable kernel.
    record = [
        ("[-1, 0.3, 2]: the scale kept in float64", [-1.0, 0.3, 2.0]),
        ("all zeros: scale 0.1, zero point from the other end",
         [0.0, 0.0, 0.0]),
        ("range under 6.1e-5 * 255: its floor, not eps", [0.0, 1e-6, 5e-7]),
    ]
    # A range where the zero point itself differs (the kernel picks the end
    # with the smaller error and rounds a float64 quotient): find one with the
    # mirrors, then put it to torch below.
    for _ in range(100000):
        x = np.asarray([-rng.random() * 5, rng.random() * 5], dtype=F32)
        if (qm.et_choose_qparams(x, -128, 127)[1]
                != qm.choose_affine(x, -128, 127, qm.FLT_EPSILON)[1]):
            record.append(("the zero point", [float(v) for v in x]))
            break
    check("a range with a different zero point exists", len(record) == 4)
    for label, values in record:
        x = np.asarray(values, dtype=F32)
        ts, tz = QD.choose_qparams.tensor(t(x), -128, 127, qm.FLT_EPSILON,
                                          torch.int8)
        es, ez = qm.et_choose_qparams(x, -128, 127)
        differ = float(ts) != float(es) or int(tz) != ez
        check(f"{label}: {values} -> PyTorch ({float(ts)!r}, {int(tz)}) vs "
              f"ExecuTorch ({float(es)!r}, {ez})", differ)
    rows = np.stack([adversarial_rows()[n] for n in
                     ("all zero", "tiny", "normal", "constant positive")])
    s4, z4 = QD.choose_qparams_per_token_asymmetric(t(rows), torch.int8)
    es, ez = qm.et_choose_qparams_per_token_asymmetric(rows)
    differ = [i for i in range(len(rows))
              if float(s4[i]) != float(es[i]) or int(z4[i]) != int(ez[i])]
    check(f"per token: the portable chooser disagrees with A4 on "
          f"{len(differ)} of {len(rows)} rows", len(differ) == len(rows),
          "; ".join(f"({float(s4[i]):.9g}, {int(z4[i])}) vs "
                    f"({float(es[i]):.9g}, {int(ez[i])})" for i in differ))


# --- 9. models.py --------------------------------------------------------------

def models_link():
    from executorch.backends.esp32s3.test import reference

    print("models.py imports the module rather than copying it:")
    rng = np.random.default_rng(11)
    amax = np.concatenate([rng.random(2000) * 10.0 ** rng.integers(-9, 4, 2000),
                           [0.0, 127 * qm.FLT_EPSILON, 1e-12, 3.0]])
    old = [max(F32(F32(a) / F32(127.0)), F32(np.finfo(F32).eps)) for a in amax]
    new = [reference.pt2e_tensor_scale(a) for a in amax]
    check("pt2e_tensor_scale through mirrors.py is bit-identical",
          same(np.asarray(old, F32), np.asarray(new, F32)))
    # run_parity holds the symmetric rows to models' dense mirrors, which are
    # this module's A0 path with the fused epilogue (models.dense_act): bit for
    # bit on per-row scales, ragged groups, an all-zero row (scale 0, codes 0)
    # and halves on every tie.
    m, n, k, gsize = 5, 7, 300, 128
    x = (rng.standard_normal((m, k)) * 3).astype(F32)
    x[1] = 0.0
    x[2, :4] = np.asarray([127.0, 0.5, -0.5, 1.5], F32)
    codes = rng.integers(-127, 128, (n, k))
    scales = (rng.random((n, -(-k // gsize))) * 0.01).astype(F32)
    bias = rng.standard_normal(n).astype(F32)
    ref = qm.affine_matmul(x, codes, scales, bias, act=qm.A0, gsize=gsize,
                           fused_madd=True)
    check("models.addmm_codes_ref is the A0 path with fused_madd, bit for bit",
          same(reference.addmm_codes_ref(x, codes, scales, bias, gsize), ref.y))


SECTIONS = [activations, weights, matmul, convolutions, embeddings, kcache,
            et_choose_qparams, models_link]


def _sweep_skipped():
    return os.environ.get("QUANT_MIRRORS_SKIP_SWEEP") == "1"


@case
def test_reciprocal_and_division():
    if _sweep_skipped():
        import pytest

        pytest.skip("QUANT_MIRRORS_SKIP_SWEEP=1")
    reciprocal_and_division()


test_activations = case(activations)
test_weights = case(weights)
test_matmul = case(matmul)
test_convolutions = case(convolutions)
test_embeddings = case(embeddings)
test_kcache = case(kcache)
test_et_choose_qparams = case(et_choose_qparams)
test_models_link = case(models_link)


def main():
    sections = list(SECTIONS)
    if not _sweep_skipped():
        sections.insert(0, reciprocal_and_division)
    for section in sections:
        # A section that raises is a failure; the others still run.
        try:
            section()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            check(f"{section.__name__} ran to completion", False, repr(exc))
    print()
    if SKIPS:
        print(f"SKIPPED: {', '.join(SKIPS)}")
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
