# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every spelling of RMSNorm fuses; functions that only look like it do not.

The fusion's canonical spelling is ``x * rsqrt(mean(x^2) + eps)``, with the
weight on either side. Real checkpoints write the same operation several other
ways -- Hugging Face's Muse Glimmer reference RMSNorm writes ``pow(v, -0.5)``
for its QK-norms, its embedding norm and its final norm -- and a norm left
unfused stays portable and needs ``pow``, which the firmware op list does not
register.

Claims:

  1. each spelling of the same operation (pow -0.5, reciprocal or division by
     sqrt, ``1 / sqrt``, ``x * x``, the weight before, after or inside the
     inverse root) fuses exactly one site into the right op, and the fused
     graph agrees with the unfused one to rounding (relative error < 1e-6:
     the fused host op computes ``(w * x) * rsqrt(v)``, so pow and sqrt
     spellings are not bit-identical, the same way the device kernel is not),
  2. Muse Glimmer's reference norm fuses, scaled and weightless (its 0-d
     ``weight = 1.0`` buffer multiplies by exactly 1.0, an identity),
  3. QK-norms over the head dim fuse before and after the head transpose,
     weightless with Muse's q gain and weighted Qwen3-style,
  4. different functions stay portable: eps outside the root, an L2 norm
     (sum, not mean), another exponent, a hand-written LayerNorm; and a scalar
     gain other than 1.0 stays a portable multiply over a fused norm,
  5. a toy Muse Glimmer layer in the reference spelling loses every pow, sqrt
     and mean through the production pipeline,
  6. ExecuTorch's own RMSNorm fusion test models (rc1's
     backends/transforms/test/test_fuse_rms_norm.py, loaded from this tree:
     the fp32 result exposed beside the rounded one, explicit casts around a
     half-precision model, Gemma's fp32 (1 + w), the weight from a parameter,
     a buffer, a lifted constant, a graph input or a computed expression, the
     weight or an fp32 value escaping as an output) each fuse their fp32 core
     into one site, keep every cast and every escaping value, and agree with
     the unfused program at that file's tolerances; a weight the program
     writes is read before the write, call after call,
  7. that file's near misses stay portable (a cube, exponent -0.4, a mean over
     other axes or without keepdim, another tensor normalised, eps a graph
     input, add's alpha, a float64 mean, the inverse root cast to half, an
     intermediate escaping at each stage, a half or double core), except eps
     negative or infinite, which fuse here: the op is defined as exactly
     x * rsqrt(mean(x^2) + eps), as the device's scalar path computes it for
     a row outside its kernel's domain, where ExecuTorch's aten.rms_norm
     rewrite refuses them; a scale the kernel cannot take (a scalar,
     [1, D], [D, 1], double) fuses only the core under it; and a half
     weight, which the program casts to fp32 before the multiply, is the
     weight of one fused site, which that rewrite keeps outside too.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test fails
naming its failed checks (@case).

    python test_rmsnorm_spellings.py
"""

import contextlib
import copy
import functools
import importlib.util
import io
import sys
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

FAILS = []
D = 16
SCALE, NOSCALE = "espressif.rmsnorm.default", "espressif.rmsnorm_noscale.default"


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


def ms(x):
    return x.pow(2).mean(-1, keepdim=True)


def fuse(mod, *args):
    """(pass, unfused edge, fused edge) for one module. Each program is
    exported from its own copy, so one that writes a buffer leaves the
    other's, and the caller's, alone."""
    from executorch.exir import EdgeCompileConfig, to_edge

    from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import fuse_rmsnorm

    cfg = EdgeCompileConfig(_check_ir_validity=False)
    before = to_edge(torch.export.export(copy.deepcopy(mod).eval(), args),
                     compile_config=cfg)
    after = to_edge(torch.export.export(copy.deepcopy(mod).eval(), args),
                    compile_config=cfg)
    return fuse_rmsnorm(after), before, after


def targets(edge):
    return [str(n.target) for n in edge.exported_program().graph_module.graph.nodes
            if n.op == "call_function"]


def count(edge, name):
    return sum(name in t for t in targets(edge))


def rel_err(before, after, *args):
    with torch.no_grad():
        ref = before.exported_program().module()(*args)
        got = after.exported_program().module()(*args)
    return float((got - ref).abs().max() / ref.abs().max())


class Spelling(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.w = nn.Parameter(1.0 + 0.1 * torch.randn(D))

    def forward(self, x):
        return self.f(x, self.w)


class MuseRMSNorm(nn.Module):
    """Muse Glimmer's reference RMSNorm (Gemma 3n's form): ``pow(., -0.5)``,
    and when unscaled a 0-d buffer ``weight = 1.0`` that the output is still
    multiplied by."""

    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        if with_scale:
            self.weight = nn.Parameter(1.0 + 0.1 * torch.randn(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)

    def forward(self, x):
        return (self._norm(x.float()) * self.weight.float()).type_as(x)


class CenteredRMSNorm(nn.Module):
    """Glimmer's sandwich norms: rsqrt with a (1 + weight) scale."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(0.1 * torch.randn(dim))

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * (1.0 + self.weight)


class ScalarGain(nn.Module):
    """A weightless norm times a learned scalar: not a per-column weight."""

    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return (x * torch.pow(ms(x) + 1e-6, -0.5)) * self.g


POSITIVE = {
    "pow -0.5": (lambda x, w: x * torch.pow(ms(x) + 1e-6, -0.5), NOSCALE),
    "x / sqrt": (lambda x, w: x / torch.sqrt(ms(x) + 1e-6), NOSCALE),
    "reciprocal(sqrt)": (
        lambda x, w: x * torch.reciprocal(torch.sqrt(ms(x) + 1e-6)), NOSCALE),
    "1 / sqrt": (lambda x, w: x * (1 / torch.sqrt(ms(x) + 1e-6)), NOSCALE),
    "x * x square": (
        lambda x, w: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6),
        NOSCALE),
    "pow, weight after": (
        lambda x, w: (x * torch.pow(ms(x) + 1e-6, -0.5)) * w, SCALE),
    "pow, weight before": (
        lambda x, w: (w * x) * torch.pow(ms(x) + 1e-6, -0.5), SCALE),
    "sqrt, weight after": (
        lambda x, w: w * (x / torch.sqrt(ms(x) + 1e-6)), SCALE),
    "sqrt, weight before": (
        lambda x, w: (w * x) / torch.sqrt(ms(x) + 1e-6), SCALE),
    "rsqrt, weight inside the root": (
        lambda x, w: x * (w * torch.rsqrt(ms(x) + 1e-6)), SCALE),
    "pow, centred (1 + w) after": (
        lambda x, w: (x * torch.pow(ms(x) + 1e-6, -0.5)) * (1.0 + w), SCALE),
}

NEGATIVE = {
    "eps outside the root": lambda x, w: x / (torch.sqrt(ms(x)) + 1e-6),
    "L2 norm (sum, not mean)": (
        lambda x, w: x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)),
    "exponent -0.25": lambda x, w: x * torch.pow(ms(x) + 1e-6, -0.25),
    "LayerNorm by hand": lambda x, w: (
        (x - x.mean(-1, keepdim=True))
        / torch.sqrt((x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True)
                     + 1e-6)),
}


@case
def test_spellings():
    print("same operation, other spellings -- each fuses one site:")
    x = torch.randn(3, D)
    for name, (f, op) in POSITIVE.items():
        p, before, after = fuse(Spelling(f), x)
        err = rel_err(before, after, x)
        check(f"{name}: one {op.split('.')[1]} site", p.fused == 1
              and count(after, op) == 1 and not p.rejected,
              f"fused={p.fused} rejected={p.rejected[:1]}")
        check(f"{name}: agrees to rounding", err < 1e-6, f"rel {err:.1e}")

    print("Muse Glimmer's reference norm:")
    for with_scale, op in ((True, SCALE), (False, NOSCALE)):
        p, before, after = fuse(MuseRMSNorm(D, with_scale=with_scale), x)
        err = rel_err(before, after, x)
        check(f"with_scale={with_scale}: one {op.split('.')[1]} site, no pow left",
              p.fused == 1 and count(after, op) == 1
              and count(after, "aten.pow") == 0 and count(after, "aten.mul") == 0,
              f"fused={p.fused} rejected={p.rejected[:1]}")
        check(f"with_scale={with_scale}: agrees to rounding", err < 1e-6,
              f"rel {err:.1e}")


class QKNorm(nn.Module):
    """q/k normalised over the head dim, before or after the head transpose."""

    def __init__(self, muse, transpose_first):
        super().__init__()
        self.q = nn.Linear(32, 32, bias=False)
        self.k = nn.Linear(32, 8, bias=False)
        if muse:
            self.q_norm = MuseRMSNorm(8, with_scale=False)
            self.k_norm = MuseRMSNorm(8, with_scale=False)
        else:   # Qwen3: weighted, per head dim
            self.q_norm, self.k_norm = nn.RMSNorm(8, eps=1e-6), nn.RMSNorm(8, eps=1e-6)
        self.muse, self.transpose_first = muse, transpose_first

    def forward(self, x):
        b, t, _ = x.shape
        q, k = self.q(x).view(b, t, 4, 8), self.k(x).view(b, t, 1, 8)
        if self.transpose_first:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.muse:
            q = q * 3.87
        if not self.transpose_first:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        return torch.matmul(q, k.transpose(-1, -2))


@case
def test_qk_norms():
    print("QK-norms over the head dim:")
    x = torch.randn(1, 4, 32)
    for muse in (True, False):
        for transpose_first in (False, True):
            name = (f"{'Muse weightless + q gain' if muse else 'Qwen3 weighted'}, "
                    f"{'after' if transpose_first else 'before'} the transpose")
            p, before, after = fuse(QKNorm(muse, transpose_first), x)
            op = NOSCALE if muse else SCALE
            err = rel_err(before, after, x)
            check(f"{name}: q and k fuse", p.fused == 2 and count(after, op) == 2,
                  f"fused={p.fused} rejected={p.rejected[:1]}")
            if muse:
                check(f"{name}: the 3.87 gain stays a portable mul",
                      count(after, "aten.mul") == 1)
            check(f"{name}: agrees to rounding", err < 1e-6, f"rel {err:.1e}")


@case
def test_near_misses():
    print("different functions stay portable:")
    x = torch.randn(3, D)
    for name, f in NEGATIVE.items():
        p, before, after = fuse(Spelling(f), x)
        check(f"{name}: not fused", p.fused == 0
              and len(targets(after)) == len(targets(before)),
              f"fused={p.fused}" + (f" reason: {p.rejected[0][1][:60]}"
                                    if p.rejected else " (not a candidate)"))
    p, before, after = fuse(ScalarGain(), x)
    check("learned scalar gain: the norm under it fuses, the gain stays",
          p.fused == 1 and count(after, NOSCALE) == 1
          and count(after, "aten.mul") == 1
          and any("scalar" in r for _, r in p.rejected),
          f"fused={p.fused} rejected={[r[:40] for _, r in p.rejected]}")


class MuseLayer(nn.Module):
    """One Muse Glimmer decoder layer plus its embedding and head, in the
    reference spelling, at toy width: normed embedding (weightless), centred
    sandwich norms, weightless QK-norms with the q gain, scaled final norm."""

    def __init__(self, vocab=64, d=32, heads=4, hd=8, ffn=48):
        super().__init__()
        self.heads, self.hd = heads, hd
        self.embed = nn.Embedding(vocab, d)
        self.embed_norm = MuseRMSNorm(d, with_scale=False)
        self.input_norm, self.post_attn_norm = CenteredRMSNorm(d), CenteredRMSNorm(d)
        self.pre_ffn_norm, self.post_ffn_norm = CenteredRMSNorm(d), CenteredRMSNorm(d)
        self.q = nn.Linear(d, heads * hd, bias=False)
        self.k = nn.Linear(d, hd, bias=False)
        self.v = nn.Linear(d, hd, bias=False)
        self.o = nn.Linear(heads * hd, d, bias=False)
        self.q_norm = MuseRMSNorm(hd, with_scale=False)
        self.k_norm = MuseRMSNorm(hd, with_scale=False)
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)
        self.norm = MuseRMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed_norm(self.embed(ids))
        b, t, _ = x.shape
        h = self.input_norm(x)
        q = self.q_norm(self.q(h).view(b, t, self.heads, self.hd)) * 3.87
        k = self.k_norm(self.k(h).view(b, t, 1, self.hd))
        v = self.v(h).view(b, t, 1, self.hd)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) * self.hd ** -0.5
        o = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(b, t, -1)
        x = x + self.post_attn_norm(self.o(o))
        f = self.pre_ffn_norm(x)
        x = x + self.post_ffn_norm(
            self.down(nn.functional.silu(self.gate(f)) * self.up(f)))
        return self.head(self.norm(x))


@case
def test_muse_layer():
    print("a toy Muse Glimmer layer through the production pipeline:")
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    torch.manual_seed(0)
    model = MuseLayer().eval()
    n_params = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, 64, (1, 8))
    sites = 1 + 2 + 4 + 1          # embedding, q/k, sandwich, final
    with contextlib.redirect_stdout(io.StringIO()):
        edge = lower_esp32(torch.export.export(model, (ids,)),
                           [Esp32Partitioner(quantize_qk=True)])
        try:
            rep = apply_passes(edge, rmsnorm_expect=sites, quiet=True)
        except RuntimeError as e:
            check(f"all {sites} norms fuse ({n_params} params)", False, str(e)[:160])
            return
        et = edge.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    check(f"all {sites} norms fuse ({n_params} params)", rep["fused"] == sites,
          f"fused={rep['fused']}")
    table = {f"{op.name}.{op.overload}" for plan in et.executorch_program.execution_plan
             for op in plan.operators}
    left = sorted(t for t in table
                  if any(k in t for k in ("::pow", "::sqrt", "::mean", "::rsqrt")))
    check("no pow, sqrt, rsqrt or mean kernel left in the emitted program",
          not left, f"{left}")


# ---- ExecuTorch's own RMSNorm fusion test models ---------------------------
# rc1's tolerances by output dtype (test_fuse_rms_norm.py's _assert_close); a
# double output here still holds an fp32 core.
TOLERANCE = {torch.float32: 2e-6, torch.float64: 2e-6, torch.float16: 1e-3,
             torch.bfloat16: 8e-3}
CORE = ("aten.pow", "aten.mean", "aten.rsqrt", "aten.sqrt", "aten.reciprocal")
CAST = "_to_dim_order_copy"


@functools.lru_cache(maxsize=None)
def upstream():
    """ExecuTorch's RMSNorm fusion test module from this tree
    (backends/transforms/test/test_fuse_rms_norm.py), whose models are the
    norms real checkpoints export. Loaded by path: an installed executorch
    carries the pass, not its tests."""
    path = (Path(__file__).resolve().parents[3] / "transforms" / "test"
            / "test_fuse_rms_norm.py")
    spec = importlib.util.spec_from_file_location(
        "_executorch_test_fuse_rms_norm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dtype_name(dtype):
    return str(dtype).split(".")[-1]


def short(target):
    """aten.pow.Tensor_Scalar for an edge op's long printed form."""
    return target.split(">")[0].split(": ")[-1] if target.startswith("<") else target


def outputs(module, *args):
    with torch.no_grad():
        out = module(*args)
    return out if isinstance(out, tuple) else (out,)


def run(edge, *args):
    return outputs(edge.exported_program().module(), *args)


def agree(got, ref):
    """(ok, detail): the same number of outputs, each of the same dtype and
    shape and within rc1's tolerance for its dtype, relative and absolute
    (NaN where both are NaN)."""
    if len(got) != len(ref):
        return False, f"{len(got)} outputs, not {len(ref)}"
    for index, (g, r) in enumerate(zip(got, ref)):
        if g.dtype != r.dtype or g.shape != r.shape:
            return False, (f"output {index} is {g.dtype} {tuple(g.shape)}, "
                           f"not {r.dtype} {tuple(r.shape)}")
        tol = TOLERANCE[r.dtype]
        if not torch.isclose(g.double(), r.double(), rtol=tol, atol=tol,
                             equal_nan=True).all():
            off = (g.double() - r.double()).abs().nan_to_num(0.0).max().item()
            return False, f"output {index} off by {off:.1e}"
    return True, ""


def upstream_cases():
    """(name, model, inputs, op) for rc1's models, in the spellings and dtypes
    its tests export; each has one fp32 core, scaled when op is SCALE."""
    U = upstream()
    f16, bf16, f32 = torch.float16, torch.bfloat16, torch.float32
    x32, xh, xb = (U._rmsnorm_input(t) for t in (f32, f16, bf16))
    cases = []
    for with_scale, rsqrt, core, scale in product((False, True), repeat=4):
        if scale and not with_scale:
            continue
        cases.append((f"_FP32RMSNorm(with_scale={with_scale}, use_rsqrt={rsqrt}, "
                      f"commute_core={core}, commute_scale={scale})",
                      U._FP32RMSNorm(with_scale, rsqrt, core, scale), (x32,),
                      SCALE if with_scale else NOSCALE))
    for dtype, x in ((f16, xh), (bf16, xb)):
        for with_scale in (False, True):
            op = SCALE if with_scale else NOSCALE
            cases.append((f"{dtype_name(dtype)} _FP32RMSNorm(with_scale={with_scale})",
                          U._FP32RMSNorm(with_scale).to(dtype), (x,), op))
            cases.append((f"{dtype_name(dtype)} _CastRMSNorm(with_scale={with_scale})",
                          U._CastRMSNorm(with_scale).to(dtype), (x,), op))
        for commute in (False, True):
            cases.append((f"{dtype_name(dtype)} _OffsetRMSNorm(commute={commute})",
                          U._OffsetRMSNorm(commute).to(dtype), (x,), SCALE))
    for escape in ("input", "weight", "output"):
        cases.append((f"bfloat16 _CastRMSNorm(escape={escape!r})",
                      U._CastRMSNorm(escape=escape).to(bf16), (xb,), SCALE))
    for escape in ("input", "gamma", "scaled", "weight"):
        cases.append((f"bfloat16 _OffsetRMSNorm(escape={escape!r})",
                      U._OffsetRMSNorm(escape=escape).to(bf16), (xb,), SCALE))
    for output_dtype, weight_dtype in ((bf16, f16), (f16, bf16), (f16, f32)):
        cases.append((f"float16 in, {dtype_name(weight_dtype)} weight: _CastRMSNorm("
                      f"output_dtype={dtype_name(output_dtype)})",
                      U._CastRMSNorm(output_dtype=output_dtype).to(weight_dtype),
                      (xh,), SCALE))
    for source, expression in product(("parameter", "buffer", "constant"),
                                      ("direct", "add", "lifted_left",
                                       "lifted_right")):
        weight_dtype = f32 if expression == "direct" else bf16
        cases.append((f"_LossyWeightRMSNorm({dtype_name(weight_dtype)}, source={source!r}, "
                      f"expression={expression!r})",
                      U._LossyWeightRMSNorm(weight_dtype, source=source,
                                            expression=expression,
                                            shared=expression != "direct"),
                      (xb,), SCALE))
    for expression in ("unknown", "alpha"):
        cases.append((f"_LossyWeightRMSNorm(expression={expression!r})",
                      U._LossyWeightRMSNorm(expression=expression), (xb,), SCALE))
    cases.append(("_LossyWeightRMSNorm, the weight a graph input",
                  U._LossyWeightRMSNorm(), (xb, torch.linspace(-1.3, 1.7, 16)),
                  SCALE))
    return cases


@case
def test_upstream_models():
    print("ExecuTorch's RMSNorm fusion test models (rc1) -- each fuses its fp32 core:")
    for name, model, inputs, op in upstream_cases():
        model = model.eval()
        eager = outputs(copy.deepcopy(model), *inputs)
        p, before, after = fuse(model, *inputs)
        left = [short(t) for t in targets(after) if any(c in t for c in CORE)]
        check(f"{name}: one {op.split('.')[1]} site, the core gone, every cast kept",
              p.fused == 1 and count(after, op) == 1 and not left
              and count(after, CAST) == count(before, CAST),
              f"fused={p.fused} rejected={p.rejected[:1]} left={left} casts "
              f"{count(before, CAST)} -> {count(after, CAST)}")
        got = run(after, *inputs)
        ok, why = agree(got, run(before, *inputs))
        if ok:
            ok, why = agree(got, eager)
            why = why and f"against eager: {why}"
        check(f"{name}: agrees with the unfused program and eager", ok, why)


@case
def test_upstream_written_weight():
    print("rc1's weight in a buffer the program writes -- read before the write:")
    U = upstream()
    x = U._rmsnorm_input(torch.bfloat16)
    model = U._LossyWeightRMSNorm(source="mutable").eval()
    eager = copy.deepcopy(model)
    p, before, after = fuse(model, x)
    program = after.exported_program()
    check("one rmsnorm site, the buffer write still in the signature",
          p.fused == 1 and count(after, SCALE) == 1
          and bool(program.graph_signature.buffers_to_mutate),
          f"fused={p.fused} rejected={p.rejected[:1]}")
    module = program.module()
    results = []
    for call in (1, 2):
        got = outputs(module, x)
        ok, why = agree(got, outputs(eager, x))
        check(f"call {call}: agrees with eager", ok, why)
        results.append(got[0])
    check("the second call reads the weight the first wrote",
          not torch.equal(results[0], results[1]))


class Weighted(nn.Module):
    """rc1's WeightedNorm: a 16-element weight times the function under test."""

    def __init__(self, f):
        super().__init__()
        self.f = f
        self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

    def forward(self, *inputs):
        return self.weight * self.f(*inputs)


class Fanout(nn.Module):
    """rc1's intermediate fanout: the scaled norm, and one stage of it."""

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

    def forward(self, x):
        square = x.pow(2)
        mean = square.mean(-1, keepdim=True)
        added = mean + 1e-6
        inverse = added.pow(-0.5)
        normalized = x * inverse
        return normalized * self.weight, (square, mean, added, inverse)[self.stage]


class Native(nn.Module):
    """rc1's NativeDtypeNorm: the norm in the input's own dtype."""

    def forward(self, x):
        return x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()


class Scaled(nn.Module):
    """rc1's ScaledNorm: a weightless norm times a learned scale."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        normalized = x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        return normalized * self.weight


class NormalizedTwice(nn.Module):
    """rc1's normalized-output fanout: the norm returned, and scaled too."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-1.7, 2.3, 16))

    def forward(self, x):
        f32 = x.float()
        normalized = f32 * (f32.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
        return normalized, normalized * self.weight, f32


def upstream_near_misses(x):
    """rc1's near misses (test_near_match_semantics_are_not_fused) on its
    square input x, where a wrong reduction still broadcasts: (f, inputs)."""
    return {
        "x cubed": (
            lambda x: x * (x.pow(3).mean(-1, keepdim=True) + 1e-6).rsqrt(), (x,)),
        "exponent -0.4": (
            lambda x: x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.4), (x,)),
        "mean over axis 0": (
            lambda x: x * (x.pow(2).mean(0, keepdim=True) + 1e-6).rsqrt(), (x,)),
        "mean over both axes": (
            lambda x: x * (x.pow(2).mean([0, 1], keepdim=True) + 1e-6).rsqrt(),
            (x,)),
        "keepdim=False": (
            lambda x: x * (x.pow(2).mean(-1, keepdim=False) + 1e-6).rsqrt(), (x,)),
        "another tensor normalised": (
            lambda x, other: other * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt(),
            (x, x.flip(0))),
        "eps a 0-d graph input": (
            lambda x, eps: x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt(),
            (x, torch.tensor(1e-6))),
        "eps a [16] graph input": (
            lambda x, eps: x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt(),
            (x, torch.full((16,), 1e-6))),
        "add's alpha=2": (
            lambda x: x * torch.add(x.pow(2).mean(-1, keepdim=True), 1e-6,
                                    alpha=2).rsqrt(), (x,)),
        "a float64 mean": (
            lambda x: x * (x.pow(2).mean(-1, keepdim=True, dtype=torch.float64)
                           + 1e-6).rsqrt(), (x,)),
        "the inverse root cast to half": (
            lambda x: x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt().to(
                torch.float16), (x,)),
    }


@case
def test_upstream_near_misses():
    print("ExecuTorch's near misses (rc1) -- different functions stay portable:")
    x = torch.linspace(0.1, 2.5, 256).reshape(16, 16)
    near = [(name, Weighted(f), inputs)
            for name, (f, inputs) in upstream_near_misses(x).items()]
    near += [(f"{what} escapes", Fanout(stage), (upstream()._rmsnorm_input(torch.float32),))
             for stage, what in enumerate(("x^2", "its mean", "mean + eps",
                                           "the inverse root"))]
    near += [(f"a {dtype_name(dtype)} core", Native(),
              (torch.linspace(0.1, 2.5, 32).reshape(2, 16).to(dtype),))
             for dtype in (torch.float16, torch.bfloat16, torch.float64)]
    for name, model, inputs in near:
        p, before, after = fuse(model, *inputs)
        check(f"{name}: not fused",
              p.fused == 0 and len(targets(after)) == len(targets(before)),
              f"fused={p.fused}" + (f" reason: {p.rejected[0][1][:60]}"
                                    if p.rejected else " (not a candidate)"))


@case
def test_upstream_scales():
    print("ExecuTorch's eps extremes and scales (rc1) -- what fuses, and how much:")
    U = upstream()
    x = torch.linspace(0.1, 2.5, 256).reshape(16, 16)
    # Fused here, refused by ExecuTorch's aten.rms_norm rewrite: the op is
    # defined as this arithmetic, which the device's scalar path computes for
    # a row outside its kernel's domain (mean + eps not a positive normal).
    extremes = {
        "eps -1e-6": lambda x: x * (x.pow(2).mean(-1, keepdim=True) + (-1e-6)).rsqrt(),
        "eps inf": lambda x: x * (x.pow(2).mean(-1, keepdim=True)
                                  + float("inf")).rsqrt(),
    }
    for name, f in extremes.items():
        p, before, after = fuse(Weighted(f), x)
        ok, why = agree(run(after, x), run(before, x))
        check(f"{name}: one rmsnorm site, agrees with the unfused program",
              p.fused == 1 and count(after, SCALE) == 1 and ok,
              f"fused={p.fused} rejected={p.rejected[:1]} {why}")
    line = torch.linspace(-1.7, 2.3, 16)
    x = U._rmsnorm_input(torch.float32, seq=16)
    # A half-precision weight reaches the multiply through the program's own
    # cast to fp32, and the kernel takes that cast's output as its weight:
    # one rmsnorm site and no multiply left, the cast kept (ExecuTorch's
    # rewrite keeps such a scale outside the norm). A double weight makes the
    # product double, so only the core under it fuses, as below.
    for dtype in (torch.float16, torch.bfloat16):
        p, before, after = fuse(Scaled(line.to(dtype)), x)
        ok, why = agree(run(after, x), run(before, x))
        check(f"a {dtype_name(dtype)} (16,) scale: the weight, through the "
              f"program's cast to fp32",
              p.fused == 1 and count(after, SCALE) == 1
              and count(after, "aten.mul") == 0
              and count(before, CAST) == count(after, CAST) == 1 and ok,
              f"fused={p.fused} muls={count(after, 'aten.mul')} casts "
              f"{count(before, CAST)} -> {count(after, CAST)} {why}")
    scales = (torch.tensor(1.3), torch.tensor([1.3]), line.reshape(1, 16),
              line.reshape(16, 1), line.to(torch.float64))
    cases = [(f"a {dtype_name(weight.dtype)} {tuple(weight.shape)} scale", Scaled(weight), x)
             for weight in scales]
    cases.append(("the norm returned and scaled", NormalizedTwice(),
                  U._rmsnorm_input(torch.bfloat16)))
    for name, model, inputs in cases:
        p, before, after = fuse(model, inputs)
        ok, why = agree(run(after, inputs), run(before, inputs))
        check(f"{name}: only the core fuses, the scale a portable mul",
              p.fused == 1 and count(after, NOSCALE) == 1
              and count(after, "aten.mul") == 1 and ok,
              f"fused={p.fused} muls={count(after, 'aten.mul')} {why}")


def main():
    torch.manual_seed(0)
    for test in (test_spellings, test_qk_norms, test_near_misses, test_muse_layer,
                 test_upstream_models, test_upstream_written_weight,
                 test_upstream_near_misses, test_upstream_scales):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("every spelling of RMSNorm fuses into the kernel; look-alikes stay portable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
