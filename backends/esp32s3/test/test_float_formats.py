# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float weights in the delegate: the fp32 convolution, and fp16/bf16 as
stored weight formats for matmul, convolution and embedding (opcode 1 under
FLAG_CONV, opcodes 11-14; esp32_float.h).

- A convolution a recipe left float stays fp32 and delegated (_KEEP_FLOAT_OPS),
  in the delegate's fp32 convolution: dense, grouped, depthwise, transposed.
- A 2-byte weight is stored as shipped and widened to fp32 as it is read:
  bf16 by a 16-bit shift, fp16 exactly. Every one of the 65,536 patterns of
  each widens as the mirror says (mirrors.widen_*), through the real
  gather, NaN payloads and signalling NaNs included; and the arithmetic after
  widening is the fp32 loop's, bit for bit.
- Policy: a half checkpoint's weight a recipe left alone keeps its type; a
  float model's is the policy's (i8); format_policy or format_overrides
  "fp16"/"bf16" select them for any weight.
- Under external_weights a float weight goes to the PTD as an integer one
  does (one key, no scale table), and the loops read it there.
- The runtime refuses an unknown opcode, and the 2-byte rungs' descriptors
  that the format does not allow, by name.

The host runner builds with -ffp-contract=off, so its fp32 sums round each
step twice (mirrors.py's fused=False); the device's round once. The
device rows are export_asym_static's st_xnn_conv_f32 and st_xnn_half.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import re
import struct
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.serialization import weight_formats
from executorch.backends.esp32s3.serialization import espressif_blob
from executorch.backends.esp32s3 import _passes
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
ALL = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def lower(model, inputs, partitioner, dynamic=False):
    """(lowered program, backend log). A dynamic recipe's activation chains
    are absorbed by lower_esp32's own strip right before partitioning, as
    the census lowers them (strip_before_partitioning checks that it ran)."""
    ep = torch.export.export(model, inputs, strict=False)
    log = io.StringIO()
    strip = contextlib.nullcontext()
    if dynamic:
        from executorch.backends.esp32s3.test.quant_census.lower import (
            strip_before_partitioning,
        )

        strip = strip_before_partitioning({})
    with contextlib.redirect_stdout(log), strip:
        lowered = _passes.lower_esp32(ep, [partitioner])
    with contextlib.redirect_stdout(log):
        _passes.apply_passes(lowered, quiet=True)
    return lowered, log.getvalue()


def descriptors(log):
    """Every descriptor the blobs carry, by kind (addmm_f32, embed_f16, ...)."""
    return re.findall(r"(?:: |, )([A-Za-z][A-Za-z0-9_]*)\[", "\n".join(
        l for l in log.splitlines() if " B blob" in l))


def conv_descriptors(log):
    """The descriptor kind of every convolution (its emit text has a conv
    note)."""
    return re.findall(r"(?:: |, )([A-Za-z][A-Za-z0-9_]*)\[[^,]* conv\d", "\n".join(
        l for l in log.splitlines() if " B blob" in l))


def portable(lowered):
    return [str(n.target) for n in lowered.exported_program().graph.nodes
            if n.op == "call_function" and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)]


def run_host(runner, tmp_path, pte_bytes, inputs, name="m", n_out=1,
             ptd=None):
    """The host runner's outputs (one float32 vector, in output order), or its
    stderr if the program is refused: (returncode, outputs or None, stderr).
    `ptd`: the program's named data (bytes)."""
    pte = tmp_path / f"{name}.pte"
    pte.write_bytes(pte_bytes)
    command = [str(runner), "--pte", str(pte)]
    if ptd is not None:
        (tmp_path / f"{name}.ptd").write_bytes(ptd)
        command += ["--ptd", str(tmp_path / f"{name}.ptd")]
    for i, x in enumerate(inputs):
        x.numpy().tofile(tmp_path / f"{name}_in{i}.bin")
        command += ["--input", str(tmp_path / f"{name}_in{i}.bin")]
    outs = [tmp_path / f"{name}_out{i}.bin" for i in range(n_out)]
    for out in outs:
        if out.exists():
            out.unlink()
        command += ["--output", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    got = None
    if done.returncode == 0 and all(o.exists() for o in outs):
        got = np.concatenate([np.fromfile(o, dtype=F32) for o in outs])
    return done.returncode, got, done.stderr


def dyadic(g, *shape, den, lim=8):
    return torch.randint(-lim, lim + 1, shape, generator=g).float() / den


# --- widening: the mirror against numpy, and the store-as-shipped round trip --

def test_widening_mirrors_cover_every_pattern():
    """mirrors.widen_f16_bits agrees with numpy's cast on every
    non-NaN half and keeps each NaN's sign and payload (a signalling one
    signalling); bf16 is the top half. half_bits, the exporter's narrowing,
    returns every pattern from its widened value: a 2-byte weight is stored
    as it shipped."""
    wide = qm.widen_f16_bits(ALL)
    halves = ALL.view(np.float16)
    finite = ~np.isnan(halves)
    cast = halves[finite].astype(F32).view(np.uint32)
    assert np.array_equal(wide[finite], cast)
    nan = ALL[~finite].astype(np.uint32)
    assert nan.size == 2046
    assert np.array_equal(wide[~finite],
                          ((nan & 0x8000) << 16) | 0x7F800000
                          | ((nan & 0x3FF) << 13))
    assert np.array_equal(qm.widen_bf16_bits(ALL), ALL.astype(np.uint32) << 16)
    for fmt in ("fp16", "bf16"):
        back = weight_formats.half_bits(qm.widen(ALL, fmt), fmt)
        assert np.array_equal(back, ALL), fmt


def test_half_bits_rounds_and_refuses_overflow():
    """An fp32 weight narrowed by the policy rounds to nearest even; one past
    fp16's range is refused by name, not stored as an infinity."""
    w = np.array([1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 65504.0], F32)
    assert list(weight_formats.half_bits(w, "fp16")) == [0x3C00, 0x3C02, 0x7BFF]
    w = np.array([1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8], F32)
    assert list(weight_formats.half_bits(w, "bf16")) == [0x3F80, 0x3F82]
    with pytest.raises(RuntimeError, match="past fp16's range"):
        weight_formats.half_bits(np.array([70000.0], F32), "fp16")


def _half_bits_widening(w, fmt):
    """half_bits narrowing the whole weight at once, the reference for the
    chunked narrowing: the whole weight at once, bf16's rounding
    widened to uint64, about eight times the weight in temporaries."""
    w = np.ascontiguousarray(w, dtype=F32)
    u = w.view(np.uint32)
    if fmt == "fp16":
        with np.errstate(over="ignore", invalid="ignore"):
            bits = w.astype(np.float16).view(np.uint16)
            wide = bits.view(np.float16).astype(F32)
        mant = ((u >> 13) & 0x3FF).astype(np.uint16)
        nan = (((u >> 16) & 0x8000).astype(np.uint16) | 0x7C00 | mant
               | np.where(mant == 0, 0x200, 0).astype(np.uint16))
        bits = np.where(np.isnan(w), nan, bits).astype(np.uint16)
    else:
        u = u.astype(np.uint64)
        rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        top = (u >> 16).astype(np.uint16)
        nan = top | np.where((top & 0x7F) == 0, 0x40, 0).astype(np.uint16)
        bits = np.where(np.isnan(w), nan, rounded).astype(np.uint16)
        wide = (bits.astype(np.uint32) << 16).view(F32)
    lost = np.isfinite(w) & ~np.isfinite(wide)
    if lost.any():
        raise RuntimeError(
            f"Esp32Backend: weight holds {float(w[lost].flat[0])!r}, past "
            f"{fmt}'s range: stored as {fmt} it would be an infinity")
    return bits


def _half_corpus(fmt, n, seed):
    """n random float32 bit patterns the format holds, more than one of
    half_bits' chunks: every sign and payload, exponents from float32's
    zero and subnormals through infinity and NaN (for fp16 the ones near
    its range: its subnormals up to its largest finite value), then the
    edge cases by name -- signed zeros, infinities, quiet and signalling
    NaNs with payloads only in the bits narrowing drops, the smallest and
    largest subnormals and normals, and values exactly halfway between two
    halves with an even and an odd lower neighbour, either side of each."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32)
    if fmt == "fp16":
        exp = rng.choice(np.r_[0, np.arange(100, 143), 255], n)
        u = (u & 0x807FFFFF) | (exp.astype(np.uint32) << 23)
    named = np.array([
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000,         # +-0, +-inf
        0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001,         # NaNs
        0x7F801FFF, 0x7FA00000, 0x7F80FFFF, 0x7FFF0000, 0xFFFFFFFF,
        0x00000001, 0x807FFFFF, 0x00800000, 0x7F7FFFFF,         # subnormals
        0x3F808000, 0x3F818000, 0x3F807FFF, 0x3F808001,         # bf16 ties
        0x3F801000, 0x3F803000, 0x3F800FFF, 0x3F801001,         # fp16 ties
        0x477FE000, 0x477FEFFF, 0x387FC000, 0x38800000,         # fp16 edges
        0x33800000, 0x33000000, 0x33000001, 0x33400000,
        0x7F7F7FFF, 0xFF7F7FFF], dtype=np.uint32)
    w = np.concatenate([u, named]).view(F32)
    if fmt == "fp16":
        held = ~(np.isfinite(w) & (np.abs(w) >= 65520.0))
    else:
        held = ~(np.isfinite(w)
                 & ((w.view(np.uint32) & 0x7FFFFFFF) >= 0x7F7F8000))
    return w[held]


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_half_bits_matches_the_widening_narrowing(fmt):
    """half_bits narrows a chunk at a time, bf16's rounding in place in
    uint32, and gives the whole-weight narrowing's bits exactly: on 2.5
    million random patterns that cross several chunks (NaNs, infinities,
    subnormals and signed zeros among them), on the named edge cases, on a
    transposed view, and in its refusal of a finite value past the range."""
    w = _half_corpus(fmt, 5 * 2 ** 19, seed=42 if fmt == "fp16" else 43)
    assert w.size > 2 * weight_formats._HALF_CHUNK
    assert np.isnan(w).sum() > 1000 and (w == 0).sum() >= 2
    got, want = weight_formats.half_bits(w, fmt), _half_bits_widening(w, fmt)
    assert got.dtype == np.uint16 and got.shape == w.shape
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (f"{bad.size} differ, first at {w[bad[0]]!r}: "
                           f"{got[bad[0]]:#06x} != {want[bad[0]]:#06x}")
    view = w[:12].reshape(3, 4).T
    assert np.array_equal(weight_formats.half_bits(view, fmt),
                          _half_bits_widening(view, fmt))
    past = np.array([1.0, 3.4e38, -1.0e30, 70000.0], F32)
    with pytest.raises(RuntimeError) as old:
        _half_bits_widening(past, fmt)
    with pytest.raises(RuntimeError) as new:
        weight_formats.half_bits(past, fmt, "weight")
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_half_bits_temporaries_stay_small(fmt):
    """Narrowing a 64 MB fp32 weight allocates its 32 MB of half bits and a
    few MB at a time besides (tracemalloc sees numpy's buffers), where the
    widening narrowing takes 320 MB (fp16) and 512 MB (bf16): a 268 MB
    weight would take about 2 GB."""
    import tracemalloc

    w = np.random.default_rng(7).standard_normal(16 * 2 ** 20).astype(F32)
    tracemalloc.start()
    try:
        weight_formats.half_bits(w, fmt)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < w.nbytes, f"{peak / 2 ** 20:.0f} MB for a {w.nbytes >> 20} MB weight"


class Table(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.emb = nn.Embedding(*values.shape)
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(values))

    def forward(self, ids):
        return self.emb(ids)


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_runtime_widens_every_pattern(runner, tmp_path, fmt):
    """All 65,536 patterns as one 256 x 256 table, stored as fmt
    (format_overrides) and gathered row by row by the delegate: the host
    runner's output bits are the mirror's, NaN payloads included."""
    table = qm.widen(ALL, fmt).reshape(256, 256)
    ids = (torch.arange(256, dtype=torch.int64),)
    lowered, log = lower(Table(table).eval(), ids, Esp32Partitioner(
        format_overrides={"emb.weight": fmt}))
    assert descriptors(log) == [f"embed_{weight_formats.HALF_TAGS[fmt]}"], log
    code, out, err = run_host(runner, tmp_path,
                              lowered.to_executorch().buffer, ids, fmt)
    assert code == 0, err[-1500:]
    want = (qm.widen_f16_bits(ALL) if fmt == "fp16"
            else qm.widen_bf16_bits(ALL))
    assert np.array_equal(out.view(np.uint32), want)


# --- the fp32 convolution ---------------------------------------------------------

class ConvMix(nn.Module):
    """The conv kinds XNNPACK's dynamic recipe leaves float, each reading the
    input: dense and grouped (g=2, dilated) each into a relu, depthwise
    strided. (Its dynamic config quantises a conv with no relu after it,
    depthwise aside, and a transposed conv -- test_asym_act's
    test_recipe_transposed_conv_is_delegated; the fp16 test below takes the
    float transposed gather.) And a Linear the recipe quantises, so the
    program is recipe-quantised and the convs keep their precision."""

    def __init__(self, dyadic_weights):
        super().__init__()
        g = torch.Generator().manual_seed(29)
        self.dense = nn.Conv2d(4, 8, 3, padding=1)
        self.dw = nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=4)
        self.grp = nn.Conv2d(4, 6, 3, padding=2, dilation=2, groups=2)
        self.fc = nn.Linear(256, 4)
        with torch.no_grad():
            for conv in (self.dense, self.dw, self.grp):
                if dyadic_weights:
                    conv.weight.copy_(dyadic(g, *conv.weight.shape, den=16))
                    conv.bias.copy_(dyadic(g, *conv.bias.shape, den=16))
                else:
                    conv.weight.copy_(torch.randn(*conv.weight.shape,
                                                  generator=g))
                    conv.bias.copy_(torch.randn(*conv.bias.shape, generator=g))

    def forward(self, x):                       # x [1, 4, 8, 8]
        return (torch.relu(self.dense(x)), self.dw(x),
                torch.relu(self.grp(x)), self.fc(x.reshape(1, 256)))


def conv_program(dyadic_weights, x):
    from executorch.backends.esp32s3.test.quant_census import recipes

    net = ConvMix(dyadic_weights).eval()
    q = recipes.quantize("xnnpack-dynamic", net, (x,), [(x,)])
    part = Esp32Partitioner()
    lowered, log = lower(q.program, q.inputs, part, dynamic=True)
    return net, q, part, lowered, log


def ramp():
    from executorch.backends.esp32s3.test.models import ramp_input

    return ramp_input((1, 4, 8, 8))


def test_recipe_left_float_convs_run_the_fp32_convolution(runner, tmp_path):
    """XNNPACK's dynamic recipe leaves the convolutions float: each is
    claimed by the delegate's fp32 convolution (opcode 1 under FLAG_CONV; a
    grouped conv one descriptor per group), named in the kept-float report,
    and with dyadic weights and input the host runner's output is eager's,
    exactly -- every conv kind."""
    x = ramp()
    net, q, part, lowered, log = conv_program(True, x)
    assert part.support.keep_float
    assert sorted(conv_descriptors(log)) == ["addmm_f32"] * 4, log
    assert not any("convolution" in t for t in portable(lowered))
    kept = [l for l in log.splitlines() if l.startswith(
        "esp32 partitioner: kept float")]
    assert len(kept) == 3 and all("fp32 loop in the delegate" in l
                                  for l in kept), kept
    code, out, err = run_host(runner, tmp_path,
                              lowered.to_executorch().buffer, (x,), n_out=4)
    assert code == 0, err[-1500:]
    with torch.no_grad():
        eager = torch.cat([t.reshape(-1) for t in net(x)[:3]]).numpy()
    convs = out[:eager.size]
    assert np.abs(convs - eager).max() == 0.0


def test_fp32_convolution_is_the_mirrors_bit_for_bit(runner, tmp_path):
    """Random weights and input: the host runner's dense, depthwise and
    grouped fp32 convolutions are mirrors.float_conv's sum to the bit
    (the host rounds each step twice, fused=False), and differ from eager's
    by rounding only."""
    from executorch.backends.esp32s3.operator_support.conv import ConvGeom, split_groups

    torch.manual_seed(3)
    x = torch.randn(1, 4, 8, 8)
    net, q, part, lowered, log = conv_program(False, x)
    code, out, err = run_host(runner, tmp_path,
                              lowered.to_executorch().buffer, (x,), n_out=4)
    assert code == 0, err[-1500:]

    def geom(conv, h_out):
        kh = conv.weight.shape[2]
        return ConvGeom(h_in=8, w_in=8, c_in=conv.weight.shape[1], h_out=h_out,
                        w_out=h_out, c_out=conv.weight.shape[0], kh=kh, kw=kh,
                        stride_h=conv.stride[0], stride_w=conv.stride[0],
                        pad_h=conv.padding[0], pad_w=conv.padding[0],
                        dil_h=conv.dilation[0], dil_w=conv.dilation[0],
                        groups=conv.groups, c_out_total=conv.weight.shape[0])

    xs = x.numpy()
    at = 0
    for conv, h_out, relu in ((net.dense, 8, True), (net.dw, 4, False),
                              (net.grp, 8, True)):
        g = geom(conv, h_out)
        w = conv.weight.detach().numpy().reshape(g.c_out, -1)
        b = conv.bias.detach().numpy()
        subs = split_groups(g)
        per = g.c_out // len(subs)
        parts = [qm.float_conv(xs, w[s.group_idx * per:(s.group_idx + 1) * per]
                               if len(subs) > 1 else w,
                               b[s.group_idx * per:(s.group_idx + 1) * per]
                               if len(subs) > 1 else b, s, fused=False)
                 for s in subs]
        want = np.concatenate(parts, axis=1).reshape(-1)
        if relu:
            want = np.where(want < 0, F32(0), want).astype(F32)
        got = out[at:at + want.size]
        at += want.size
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), conv
        with torch.no_grad():
            ref = conv(x).numpy().reshape(-1)
        if relu:
            ref = np.maximum(ref, 0)
        assert np.abs(got - ref).max() < 1e-4
    assert at == 8 * 64 + 4 * 16 + 6 * 64


# --- fp16/bf16: policy, and the arithmetic after widening ------------------------

def half_inputs():
    from executorch.backends.esp32s3.test.fixtures import asym_static as eas

    row = next(r for r in eas.SPECS if r.name == "st_xnn_half")
    return eas.row_inputs(row)


def test_recipe_keeps_a_half_checkpoints_type():
    """export_asym_static.HalfLeft (QEMU row st_xnn_half): in a
    recipe-quantised program each half checkpoint weight keeps its type --
    the bf16 and fp16 matmuls, the bf16 conv, the fp16 and bf16 gathers --
    delegated and named in the report; in the same model unquantised (a
    float model) they are the policy's, int8."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as eas
    from executorch.backends.esp32s3.test.quant_census import recipes

    x = half_inputs()
    net = eas.HalfLeft().eval()
    q = recipes.quantize("xnnpack-dynamic", net, x, [x])
    part = Esp32Partitioner()
    lowered, log = lower(q.program, q.inputs, part, dynamic=True)
    assert part.support.keep_float
    ops = sorted(descriptors(log))
    assert ops == ["addmmPT_i8", "addmm_bf16", "addmm_bf16", "addmm_f16",
                   "embed_bf16", "embed_f16"], log
    assert conv_descriptors(log) == ["addmm_bf16"], log
    kept = [l for l in log.splitlines() if l.startswith(
        "esp32 partitioner: kept float")]
    assert sorted(l.split(";")[0].split("] ")[1] for l in kept) == [
        "bf16 256 B", "bf16 288 B", "bf16 512 B", "fp16 256 B",
        "fp16 512 B"], kept
    assert not any("embedding" in t or "convolution" in t or ".mm" in t
                   for t in portable(lowered))
    _, log = lower(net, x, Esp32Partitioner())
    assert sorted(descriptors(log)) == ["addmm_i8"] * 4 + ["embed_i8"] * 2, log


def test_policy_and_override_select_half_for_any_weight():
    """format_policy="bf16" stores every weight it claims as bf16 -- matmul,
    conv and table; a format_overrides value "fp16" stores the one it names."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as eas

    x = half_inputs()
    net = eas.HalfLeft().eval()
    _, log = lower(net, x, Esp32Partitioner(format_policy="bf16"))
    assert sorted(descriptors(log)) == ["addmm_bf16"] * 4 + ["embed_bf16"] * 2, log
    _, log = lower(net, x, Esp32Partitioner(
        format_overrides={"fc.weight": "fp16", "table": "fp16"}))
    assert sorted(descriptors(log)) == ["addmm_f16"] + ["addmm_i8"] * 3 + [
        "embed_f16", "embed_i8"], log
    # The float policies' weights go to the PTD under external_weights.
    Esp32Partitioner(format_policy="fp16", external_weights=True)
    with pytest.raises(RuntimeError, match="cannot combine with static_act"):
        Esp32Partitioner(format_policy="fp16", static_act=True)


class Lin(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.fc = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            self.fc.weight.copy_(w)
            self.fc.bias.copy_(b)

    def forward(self, x):
        return torch.relu(self.fc(x))


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_half_matmul_is_the_fp32_loop_on_the_widened_weight(runner, tmp_path,
                                                             fmt):
    """Random weights the format holds and a random input: the fmt Linear's
    output (relu fused) equals, bit for bit, the fp32 loop's (opcode 1,
    format_policy="fp32") on the same values, and mirrors.py's sum."""
    torch.manual_seed(11)
    dtype = torch.float16 if fmt == "fp16" else torch.bfloat16
    w = torch.randn(24, 40).to(dtype).float()
    b = torch.randn(24)
    x = (torch.randn(3, 40),)
    model = Lin(w, b).eval()
    half, log = lower(model, x, Esp32Partitioner(
        format_overrides={"fc.weight": fmt}))
    assert descriptors(log) == [f"addmm_{weight_formats.HALF_TAGS[fmt]}"], log
    fp32, log = lower(model, x, Esp32Partitioner(format_policy="fp32"))
    assert descriptors(log) == ["addmm_f32"], log
    outs = []
    for name, prog in (("half", half), ("fp32", fp32)):
        code, out, err = run_host(runner, tmp_path, prog.to_executorch().buffer,
                                  x, name)
        assert code == 0, err[-1500:]
        outs.append(out)
    assert np.array_equal(outs[0].view(np.uint32), outs[1].view(np.uint32))
    want = qm.float_dot_rows(x[0].numpy(), w.numpy(), b.numpy(), fused=False)
    want = np.where(want < 0, F32(0), want).astype(F32).reshape(-1)
    assert np.array_equal(outs[0].view(np.uint32), want.view(np.uint32))


def test_half_conv_is_the_fp32_convolution_on_the_widened_weight(runner,
                                                                  tmp_path):
    """A bf16 conv (format_overrides) on a random input is
    mirrors.float_conv on the widened weight, the sum the host's fp32
    convolution forms (test_fp32_convolution_is_the_mirrors_bit_for_bit)."""
    from executorch.backends.esp32s3.operator_support.conv import ConvGeom

    torch.manual_seed(13)
    conv = nn.Conv2d(4, 5, 3, padding=1).eval()
    with torch.no_grad():
        conv.weight.copy_(conv.weight.bfloat16().float())
    x = (torch.randn(1, 4, 6, 6),)
    lowered, log = lower(conv, x, Esp32Partitioner(
        format_overrides={"weight": "bf16"}))
    assert conv_descriptors(log) == ["addmm_bf16"], log
    code, out, err = run_host(runner, tmp_path,
                              lowered.to_executorch().buffer, x)
    assert code == 0, err[-1500:]
    g = ConvGeom(h_in=6, w_in=6, c_in=4, h_out=6, w_out=6, c_out=5, kh=3, kw=3,
                 stride_h=1, stride_w=1, pad_h=1, pad_w=1, dil_h=1, dil_w=1,
                 groups=1, c_out_total=5)
    want = qm.float_conv(x[0].numpy(), conv.weight.detach().numpy().reshape(5, -1),
                         conv.bias.detach().numpy(), g, fused=False)
    assert np.array_equal(out.view(np.uint32), want.reshape(-1).view(np.uint32))


@pytest.mark.parametrize("geometry", [
    dict(kernel_size=2, stride=2),
    dict(kernel_size=3, stride=2, padding=1, output_padding=1),
    # Output padding 2 at stride 2, which only the dilation of 3 allows.
    dict(kernel_size=3, stride=(2, 1), padding=(1, 2), dilation=(3, 2),
         output_padding=(2, 1)),
], ids=["2x2 s2", "3x3 s2 p1 output_padding 1",
        "dil (3,2) output_padding (2,1)"])
def test_half_transposed_conv_is_exact(runner, tmp_path, geometry):
    """The float gather's transposed walk (a zero-stuffed input, never built),
    with output padding and dilation: an fp16 ConvTranspose2d with dyadic
    weights on the ramp is eager's, exactly."""
    g = torch.Generator().manual_seed(19)
    up = nn.ConvTranspose2d(4, 3, **geometry).eval()
    with torch.no_grad():
        up.weight.copy_(dyadic(g, *up.weight.shape, den=16))
        up.bias.copy_(dyadic(g, *up.bias.shape, den=16))
    x = (ramp(),)
    lowered, log = lower(up, x, Esp32Partitioner(
        format_overrides={"weight": "fp16"}))
    assert conv_descriptors(log) == ["addmm_f16"], log
    code, out, err = run_host(runner, tmp_path,
                              lowered.to_executorch().buffer, x)
    assert code == 0, err[-1500:]
    with torch.no_grad():
        eager = up(x[0]).numpy().reshape(-1)
    assert np.abs(out - eager).max() == 0.0


# --- refusals ---------------------------------------------------------------------

def _field(name):
    return 4 * ("opcode flags m k n k_pad bias_off wt_off scale_off in_arg "
                "out_arg scale_group clamp_lo clamp_hi res_arg "
                "act_scale").split().index(name)


def mutate(pte, **fields):
    """The program with its one blob's first descriptor changed."""
    from executorch.backends.esp32s3.test.fixtures.asym_blob_cases import payload_span

    offset, _ = payload_span(pte)
    data = bytearray(pte)
    for name, value in fields.items():
        struct.pack_into("<I", data, offset + espressif_blob.BLOB_HEADER_BYTES
                         + _field(name), value)
    return bytes(data)


def test_runtime_refuses_by_name(runner, tmp_path):
    """One past the opcode table is an unknown opcode; a 2-byte rung's
    descriptor with what its format has no room for -- padding, a scale
    group or offset, a flag on its gather, a quantiser's flag or a fused
    multiply on its matmul -- is refused with its own message. The
    unmodified programs run."""
    torch.manual_seed(17)
    lin = Lin(torch.randn(8, 16).bfloat16().float(), torch.randn(8)).eval()
    x = (torch.randn(2, 16),)
    mm, log = lower(lin, x, Esp32Partitioner(
        format_overrides={"fc.weight": "bf16"}))
    assert descriptors(log) == ["addmm_bf16"], log
    table = qm.widen(ALL[:64], "fp16").reshape(8, 8)
    ids = (torch.arange(8, dtype=torch.int64),)
    emb, log = lower(Table(table).eval(), ids, Esp32Partitioner(
        format_overrides={"emb.weight": "fp16"}))
    assert descriptors(log) == ["embed_f16"], log
    mm, emb = mm.to_executorch().buffer, emb.to_executorch().buffer
    for name, prog, inputs in (("mm", mm, x), ("emb", emb, ids)):
        code, _, err = run_host(runner, tmp_path, prog, inputs, name)
        assert code == 0, err[-1500:]
    cases = [
        (mm, x, dict(opcode=15), "op 0 has unknown opcode 15"),
        (emb, ids, dict(opcode=15), "op 0 has unknown opcode 15"),
        (mm, x, dict(k_pad=32), "fp16/bf16 op 0 has k_pad 32 (k 16)"),
        (mm, x, dict(scale_group=8), "fp16/bf16 op 0 has k_pad 16 (k 16), "
                                     "scale_group 8"),
        (emb, ids, dict(scale_off=16), "scale_off 16"),
        (emb, ids, dict(flags=espressif_blob.FLAG_CLAMP),
         "fp16/bf16 embedding op 0 has flags 0x00000004"),
        (mm, x, dict(flags=espressif_blob.FLAG_STATIC_ACT),
         "fp32 addmm op 0 has quantised-only flags 0x00000040"),
        (mm, x, dict(flags=espressif_blob.FLAG_MUL), "op 0 has an invalid fused multiply"),
    ]
    for i, (prog, inputs, fields, message) in enumerate(cases):
        code, _, err = run_host(runner, tmp_path, mutate(prog, **fields),
                                inputs, f"case{i}")
        assert code != 0 and message in err, (fields, err[-800:])


# --- external storage ---------------------------------------------------------------

PTD_TAG = "wfmt_test"


class Mixed(nn.Module):
    """A dense conv (relu fused), a depthwise conv, a Linear and a table."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(23)
        self.conv = nn.Conv2d(4, 6, 3, padding=1)
        self.dw = nn.Conv2d(4, 4, 3, padding=1, groups=4)
        self.fc = nn.Linear(16, 5)
        self.emb = nn.Embedding(10, 16)
        with torch.no_grad():
            for t in (self.conv.weight, self.dw.weight, self.fc.weight,
                      self.emb.weight):
                t.copy_(t.half().float())

    def forward(self, x, ids):                  # x [1, 4, 4, 4]
        return (torch.relu(self.conv(x)), self.dw(x),
                self.fc(x.reshape(4, 16)), self.emb(ids))


def payloads(pte):
    """Every Esp32Backend payload's descriptors, as dicts: inline delegate
    data or a segment, found as asym_blob_cases.payload_span finds its one."""
    from executorch.backends.esp32s3.test.fixtures.asym_blob_cases import _FlatBuffer

    fb = _FlatBuffer(pte)
    program = fb.u32(0)
    spans = []
    for entry in fb.tables(program, 3):           # BackendDelegateInlineData
        at = fb.field(entry, 0)
        if at is not None:
            start = at + fb.u32(at)
            spans.append(start + 4)
    base = struct.unpack_from("<Q", pte, 24)[0] if pte[8:12] == b"eh00" else 0
    for segment in fb.tables(program, 4):         # DataSegment {offset, size}
        at = fb.field(segment, 0)
        spans.append(base + (0 if at is None else
                             struct.unpack_from("<Q", pte, at)[0]))
    descs = []
    for at in sorted(set(spans)):
        if pte[at:at + 4] != struct.pack("<I", espressif_blob.MAGIC):
            continue
        _, _, n_ops, _ = struct.unpack_from(espressif_blob.BLOB_HEADER_FORMAT, pte, at)
        for i in range(n_ops):
            descs.append(dict(zip(espressif_blob.OP_DESC_FIELDS, struct.unpack_from(
                espressif_blob.OP_DESC_FORMAT, pte,
                at + espressif_blob.BLOB_HEADER_BYTES + i * espressif_blob.OP_DESC_BYTES))))
    return descs


def mixed_program(tmp_path, name, **options):
    """(pte, ptd or None, backend log) for Mixed under these options."""
    net = Mixed().eval()
    inputs = (torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(5)),
              torch.tensor([3, 0, 9, 4], dtype=torch.int64))
    lowered, log = lower(net, inputs, Esp32Partitioner(**options))
    from executorch.exir import ExecutorchBackendConfig

    config = (ExecutorchBackendConfig(external_constants=lambda node: PTD_TAG)
              if options.get("external_weights") else ExecutorchBackendConfig())
    et = lowered.to_executorch(config=config)
    out = tmp_path / name
    out.mkdir()
    et.write_tensor_data_to_file(str(out))
    ptd = out / f"{PTD_TAG}.ptd"
    return (bytes(et.buffer), ptd.read_bytes() if ptd.is_file() else None, log,
            inputs)


@pytest.mark.parametrize("options,kinds", [
    (dict(format_policy="fp32"), ["addmm_f32"] * 3),
    (dict(format_overrides={"conv.weight": "bf16", "dw.weight": "fp16",
                            "fc.weight": "fp16", "emb.weight": "fp16"}),
     ["addmm_bf16", "addmm_f16", "addmm_f16", "embed_f16"]),
], ids=["fp32", "half"])
def test_external_float_weights_are_read_from_the_ptd(runner, tmp_path,
                                                      options, kinds):
    """Under external_weights every float weight -- the fp32 matmul and
    convolutions, the 2-byte rungs' matmul, convolutions and table -- goes to
    the PTD as one content-addressed entry: FLAG_EXTERNAL_WEIGHTS, wt_off a
    key, scale_off 0. The program keeps none of their bytes, and the host
    runner's outputs are the inline program's, bit for bit."""
    inline, none, log, inputs = mixed_program(tmp_path, "inline", **options)
    assert none is None and "(weights inline)" in log
    assert sorted(descriptors(log)) == sorted(kinds), log
    ext, ptd, log, _ = mixed_program(tmp_path, "external",
                                     external_weights=PTD_TAG, **options)
    assert ptd is not None
    stored = sum(int(b.replace(",", "")) for b in
                 re.findall(r"\+ ([\d,]+) B external", log))
    net = Mixed()
    elem = 4 if options.get("format_policy") == "fp32" else 2
    params = [net.conv.weight, net.dw.weight, net.fc.weight]
    if elem == 2:
        params.append(net.emb.weight)
    assert stored == elem * sum(t.numel() for t in params), log
    assert len(inline) - len(ext) > stored - 1024, (len(inline), len(ext))
    descs = [d for d in payloads(ext)
             if d["opcode"] in (espressif_blob.OP_ADDMM, espressif_blob.OP_ADDMM_F16, espressif_blob.OP_ADDMM_BF16,
                                espressif_blob.OP_EMBED_F16)]
    assert len(descs) == len(kinds)
    assert all(d["flags"] & espressif_blob.FLAG_EXTERNAL_WEIGHTS and d["scale_off"] == 0
               for d in descs), descs
    n_out = 4
    got = []
    for name, prog, data in (("inline", inline, None), ("external", ext, ptd)):
        code, out, err = run_host(runner, tmp_path, prog, inputs, name,
                                  n_out=n_out, ptd=data)
        assert code == 0, err[-1500:]
        got.append(out)
    assert np.array_equal(got[0].view(np.uint32), got[1].view(np.uint32))


def test_external_fp32_names_no_scale_table(runner, tmp_path):
    """An external fp32 descriptor names its weight's key and nothing else:
    a scale offset is refused by name."""
    torch.manual_seed(31)
    lin = Lin(torch.randn(8, 16), torch.randn(8)).eval()
    x = (torch.randn(2, 16),)
    lowered, log = lower(lin, x, Esp32Partitioner(format_policy="fp32",
                                                  external_weights=PTD_TAG))
    assert descriptors(log) == ["addmm_f32"], log
    et = lowered.to_executorch()
    et.write_tensor_data_to_file(str(tmp_path))
    ptd = (tmp_path / f"{PTD_TAG}.ptd").read_bytes()
    pte = bytes(et.buffer)
    code, out, err = run_host(runner, tmp_path, pte, x, "ok", ptd=ptd)
    assert code == 0, err[-1500:]
    code, _, err = run_host(runner, tmp_path, mutate(pte, scale_off=4), x,
                            "bad", ptd=ptd)
    assert code != 0 and ("external fp32 op 0 names a scale table "
                          "(scale_off 4)") in err, err[-800:]


# --- fp16 widening on the dot path -------------------------------------------------

class Column(nn.Module):
    """y = x @ w.T with w one column of 65,536 rows: each output is 0 + x * w_j."""

    def __init__(self, w):
        super().__init__()
        self.fc = nn.Linear(1, w.shape[0], bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(w))

    def forward(self, x):
        return self.fc(x)


@pytest.mark.parametrize("rows", [1, 2], ids=["as-read", "widened-block"])
def test_fp16_dot_widens_every_pattern(runner, tmp_path, rows):
    """All 65,536 fp16 patterns as a [65536, 1] weight, x = 1: every output
    is 0 + 1 * w, so a finite or infinite pattern comes out as its exact
    value (a negative zero as +0) and a NaN as a NaN -- through the loop that
    widens as it reads (one row) and the one that widens a block of rows
    first (two)."""
    w = qm.widen(ALL, "fp16").reshape(-1, 1)
    x = (torch.ones(rows, 1),)
    lowered, log = lower(Column(w).eval(), x, Esp32Partitioner(
        format_overrides={"fc.weight": "fp16"}))
    assert descriptors(log) == ["addmm_f16"], log
    code, out, err = run_host(runner, tmp_path,
                              lowered.to_executorch().buffer, x, f"col{rows}")
    assert code == 0, err[-1500:]
    want = np.tile((F32(0) + w.reshape(-1)).astype(F32), rows)
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(out), nan)
    assert np.array_equal(out[~nan].view(np.uint32), want[~nan].view(np.uint32))
