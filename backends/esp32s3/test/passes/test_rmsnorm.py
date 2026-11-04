# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prove the RMSNorm fusion pass on a PLE language model, and measure what it buys.

Run:  python backends/esp32s3/test/passes/test_rmsnorm.py

Four things, in order of how much they matter:

  1. The pass fires on every RMSNorm in the real model and on nothing else. The expected
     count is derived from the module tree (3 per Block + ple_proj_norm + out_norm), not
     assumed, and `expect=` makes a miscount an exception rather than a printed warning.
  2. It refuses to fire on subgraphs that only look like RMSNorm, while accepting
     Glimmer's weightless form.
  3. The fused program is numerically equivalent to the unfused one on the same weights.
  4. The actual justification: node count, memory plan and .pte size before vs after.
     Plus the two numbers rmsnorm_s3.S's header comment cites -- the Newton-Raphson
     convergence table and the cost of its four-accumulator summation order.

Reported, not asserted, where the number is the point (4). Asserted where a regression
would be a silent bug (1, 2, 3).
"""

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from executorch.exir import EdgeCompileConfig, to_edge  # noqa: E402
from executorch.examples.espressif.esp32s3.story.glimmer_model import Config, RMSNorm, TinyLM  # noqa: E402

from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import fuse_rmsnorm  # noqa: E402

# A small PLE model. d_model=96 is deliberately NOT a power of two, which is the case
# where the kernel's `sum * inv_d` and mean.dim's `sum / d` can differ.
CFG = dict(vocab_size=4096, d_model=96, n_layers=6, n_heads=4, ffn_hidden=66,
           ple_dim=64, seq_len=64)
TOKENS = 8
EDGE = EdgeCompileConfig(_check_ir_validity=False)


class LogitsOnly(nn.Module):
    """torch.export dislikes a None in the output pytree; the runner wants one tensor."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, idx):
        logits, _ = self.inner(idx)
        return logits


def build():
    torch.manual_seed(0)
    cfg = Config(arm="ple", **CFG)
    return cfg, LogitsOnly(TinyLM(cfg).eval())


def graph_stats(ep):
    g = ep.graph_module.graph
    calls = [n for n in g.nodes if n.op == "call_function"]
    placeholders = [n for n in g.nodes if n.op == "placeholder"]
    return {
        "nodes": len(g.nodes),
        "call_function": len(calls),
        "placeholders": len(placeholders),
        "dead_placeholders": sum(1 for n in placeholders if not n.users),
        # Every call_function feeds a tensor the planner must place, so this is the
        # intermediate-buffer count the memory plan works from.
        "intermediates": len(calls),
    }


def emit(edge, tag):
    """to_executorch() and pull out the numbers a 512 KB part cares about.

    Two different memory numbers, because they answer different questions:
      peak   -- non_const_buffer_sizes, i.e. the arena main.cpp must reserve.
      demand -- every non-const planned tensor summed, i.e. how much the planner had to
                find room for. Fusion always cuts demand; it only cuts the peak if the
                buffers it deleted were part of the critical high-water mark.
    """
    et = edge.to_executorch()
    plan = et.executorch_program.execution_plan[0]
    planned = [int(v) for v in plan.non_const_buffer_sizes]
    ops = [f"{o.name}.{o.overload}" if o.overload else o.name for o in plan.operators]

    seen, demand, biggest = set(), 0, []
    for node in et.exported_program().graph_module.graph.nodes:
        spec = node.meta.get("spec")
        if spec is None:
            continue
        for s in spec if isinstance(spec, (list, tuple)) else [spec]:
            if getattr(s, "const", True) or getattr(s, "mem_id", None) is None:
                continue
            if id(s) in seen:
                continue          # a node and its alloc_ share one spec object
            seen.add(id(s))
            nbytes = int(getattr(s, "allocated_memory", 0) or 0)
            demand += nbytes
            biggest.append((nbytes, int(getattr(s, "mem_offset", 0) or 0), node.name))
    biggest.sort(reverse=True)
    return {
        "tag": tag,
        "peak_bytes": sum(planned[1:]),      # [0] is the (unused) constant pool slot
        "planned_buffers": planned[1:],
        "demand_bytes": demand,
        "planned_tensors": len(seen),
        "biggest": biggest[:4],
        "values": len(plan.values),
        "instructions": sum(len(c.instructions) for c in plan.chains),
        "pte_bytes": len(et.buffer),
        "ops": ops,
    }


def block_memory_plan(cfg):
    """The same measurement on one standalone Block.

    The whole model's arena is dominated by the output head, which no norm fusion can
    touch, so the whole-model peak is the wrong place to look for the win. One Block is
    the right place -- and it is also what block.pte already is on the device.
    """
    from executorch.examples.espressif.esp32s3.story.glimmer_model import Block, build_rope

    torch.manual_seed(0)
    block = Block(cfg).eval()
    cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
    args = (torch.randn(1, TOKENS, cfg.d_model), cos, sin,
            torch.randn(1, TOKENS, cfg.ple_dim))
    ep = torch.export.export(block, args)
    out = {}
    for tag, expect in (("unfused", None), ("fused", 3)):
        edge = to_edge(copy.deepcopy(ep), compile_config=EDGE)
        if expect is not None:
            p = fuse_rmsnorm(edge, expect=expect)
            assert p.rejected == [], p.rejected
        out[tag] = emit(edge, tag)
    return out


# ---------------------------------------------------------------------------
# 1 + 2: does the pass fire exactly where it should
# ---------------------------------------------------------------------------
def negative_cases():
    """Subgraphs that look like RMSNorm but are not. Every one must be refused."""
    D = 16

    class NotSameTensor(nn.Module):
        """y * rsqrt(mean(x^2)+eps): the scaled tensor is not the squared one."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(D))

        def forward(self, x, y):
            return (self.w * y) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    class Escapes(nn.Module):
        """The mean is also an output, so fusing would delete a live value."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(D))

        def forward(self, x):
            m = x.pow(2).mean(-1, keepdim=True)
            return (self.w * x) * torch.rsqrt(m + 1e-6), m

    class WrongExponent(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(D))

        def forward(self, x):
            return (self.w * x) * torch.rsqrt(x.pow(3).mean(-1, keepdim=True) + 1e-6)

    class FullWeight(nn.Module):
        """weight is the full activation shape, so the kernel's per-column w is wrong."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(4, D))

        def forward(self, x):
            return (self.w * x) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    class NonLastDim(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(D))

        def forward(self, x):
            return (self.w * x) * torch.rsqrt(x.pow(2).mean(-2, keepdim=True) + 1e-6)

    class MutableEps(nn.Module):
        """eps is a buffer this forward then updates. A fused literal would keep
        the export-time value on every later call."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(D))
            self.register_buffer("eps", torch.tensor(0.01))

        def forward(self, x):
            y = (self.w * x) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            self.eps.copy_(self.eps * 100.0)
            return y

    x = torch.randn(4, D)
    cases = [
        (NotSameTensor(), (x, torch.randn(4, D))),
        (Escapes(), (x,)),
        (WrongExponent(), (x,)),
        (FullWeight(), (x,)),
        (NonLastDim(), (x,)),
        (MutableEps(), (x,)),
    ]
    print("\n-- negative cases (each must be refused, with a reason) --")
    for mod, args in cases:
        edge = to_edge(torch.export.export(mod.eval(), args), compile_config=EDGE)
        p = fuse_rmsnorm(edge)
        name = type(mod).__name__
        assert p.fused == 0, f"{name}: pass fused {p.fused} sites and should have fused 0"
        assert p.candidates >= 1, f"{name}: no candidate site was even considered"
        assert p.rejected, f"{name}: refused silently, with no recorded reason"
        print(f"   {name:14s} candidates={p.candidates} fused=0  "
              f"reason: {p.rejected[0][1]}")

    # Positive control on the same tiny harness: the real shape must still fuse.
    class Real(nn.Module):
        def __init__(self):
            super().__init__()
            self.n = RMSNorm(D)

        def forward(self, x):
            return self.n(x)

    edge = to_edge(torch.export.export(Real().eval(), (x,)), compile_config=EDGE)
    p = fuse_rmsnorm(edge, expect=1)
    assert p.rejected == [], p.rejected
    print(f"   {'Real (control)':14s} candidates={p.candidates} fused={p.fused}")

    class BufferEps(nn.Module):
        """eps in a buffer nothing writes: a constant, so it still fuses."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(D))
            self.register_buffer("eps", torch.tensor(1e-6))

        def forward(self, x):
            return (self.w * x) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    edge = to_edge(torch.export.export(BufferEps().eval(), (x,)), compile_config=EDGE)
    p = fuse_rmsnorm(edge, expect=1)
    print(f"   {'BufferEps':14s} candidates={p.candidates} fused={p.fused}  "
          "(constant buffer control)")

    class RealNoScale(nn.Module):
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    mod = RealNoScale().eval()
    before = to_edge(torch.export.export(mod, (x,)), compile_config=EDGE)
    after = to_edge(torch.export.export(mod, (x,)), compile_config=EDGE)
    p = fuse_rmsnorm(after, expect=1)
    assert p.rejected == [], p.rejected
    with torch.no_grad():
        ref = before.exported_program().module()(x)
        got = after.exported_program().module()(x)
    assert torch.equal(ref, got), "weightless fusion is not bit-exact"
    emitted = emit(after, "weightless control")
    assert "espressif::rmsnorm_noscale.out" in emitted["ops"], emitted["ops"]
    print(f"   {'RealNoScale':14s} candidates={p.candidates} fused={p.fused}  "
          "bit-exact, emitted noscale.out")


# ---------------------------------------------------------------------------
# 4: the two numbers rmsnorm_s3.S cites
# ---------------------------------------------------------------------------
def _f32(v):
    return np.float32(v)


def _msub(acc, a, b):
    """MSUB.S: acc - a*b with a single rounding (ISA 8.3: 'no intermediate round')."""
    return _f32(np.float64(acc) - np.float64(a) * np.float64(b))


def _nr(t, y0, iters):
    """The exact instruction sequence in rmsnorm_s3.S."""
    t = _f32(t)
    y = _f32(y0)
    h = _f32(_f32(0.5) * t)
    for _ in range(iters):
        yy = _f32(y * y)
        u = _msub(_f32(1.5), h, yy)
        y = _f32(y * u)
    return y


def rsqrt_table():
    rng = np.random.default_rng(0)
    ts = np.concatenate([
        _f32(10.0 ** rng.uniform(-8, 8, 20000)),
        np.array([1.0, 4.0, 0.25, 1e-6, 4e-38, 8.5e37], dtype=np.float32),
    ]).astype(np.float32)
    exact = 1.0 / np.sqrt(ts.astype(np.float64))
    print("\n-- rsqrt0.s + Newton-Raphson, worst-case relative error --")
    print("   (rsqrt0.s seed is documented ~14 bits; the row is swept because that")
    print("    figure is Cadence's, not something measurable from the host)")
    print(f"   {'seed err':>9} | " + " | ".join(f"{k} iter".rjust(10) for k in (1, 2, 3)))
    best = {}
    for e0 in (2.0**-8, 2.0**-10, 2.0**-12, 2.0**-14):
        cells = []
        for k in (1, 2, 3):
            worst = 0.0
            for sign in (+1.0, -1.0):
                y0 = (exact * (1.0 + sign * e0)).astype(np.float32)
                y = np.array([_nr(a, b, k) for a, b in zip(ts, y0)], dtype=np.float32)
                rel = np.abs(y.astype(np.float64) - exact) / exact
                worst = max(worst, float(rel.max()))
            cells.append(worst)
            best[(e0, k)] = worst
        print(f"   2^-{int(round(-np.log2(e0))):<6d} | "
              + " | ".join(f"{c:.2e}".rjust(10) for c in cells))
    ulp = float(np.finfo(np.float32).eps)
    print(f"   fp32 1 ulp at 1.0 = {ulp:.3e}; a correctly rounded 1/sqrtf is <= "
          f"{ulp/2:.2e}")
    # The kernel ships 2 iterations. It must hold up even if the seed is 64x worse than
    # documented, which is the entire reason it is 2 and not 1.
    assert best[(2.0**-8, 2)] < 2e-7, best[(2.0**-8, 2)]
    assert best[(2.0**-8, 1)] > 1e-5, "1 iteration would have been enough after all"
    print(f"   -> kernel uses 2 iterations: <= {best[(2.0**-8, 2)]:.2e} relative "
          f"({best[(2.0**-8, 2)]/ulp:.2f} ulp) even at a 2^-8 seed")
    return best[(2.0**-8, 2)]


def _sum_squares_kernel(row):
    """Pass 1 of rmsnorm_s3.S: 4 accumulators, fused madd.s, then f0+f1,+f2,+f3, tail."""
    acc = [_f32(0.0)] * 4
    n4 = (len(row) // 4) * 4
    for i in range(0, n4, 4):
        for j in range(4):
            acc[j] = _f32(np.float64(acc[j]) + np.float64(row[i + j]) ** 2)
    s = acc[0]
    for j in (1, 2, 3):
        s = _f32(s + acc[j])
    for v in row[n4:]:
        s = _f32(np.float64(s) + np.float64(v) ** 2)
    return s


def summation_order(sites):
    """Cost of the kernel's 4 accumulators + fused madd vs mean.dim's sequential sum."""
    worst, nrows = 0.0, 0
    for x, _w in sites:
        for row in x.reshape(-1, x.shape[-1]).numpy().astype(np.float32):
            nrows += 1
            # portable: pow writes an fp32 temp, mean adds them in order, then / d
            seq = _f32(0.0)
            for v in row:
                seq = _f32(seq + _f32(v * v))
            mean_ref = _f32(seq / _f32(len(row)))
            mean_k = _f32(_sum_squares_kernel(row)
                          * _f32(_f32(1.0) / _f32(len(row))))
            eps = _f32(1e-6)
            r_ref = _f32(1.0) / np.sqrt(_f32(mean_ref + eps))
            r_k = _f32(1.0) / np.sqrt(_f32(mean_k + eps))
            worst = max(worst, abs(float(r_k) - float(r_ref)) / float(r_ref))
    print("\n-- summation order: 4 fused accumulators vs mean.dim's sequential sum --")
    print(f"   worst relative difference in the rsqrt scale, over {nrows} real "
          f"activation rows: {worst:.2e}")
    print("   (not an error -- shorter accumulation chains are the more accurate order;")
    print("    it is simply not bit-identical to mean.dim)")
    return worst


def kernel_transcription(sites, seed_rel=2.0**-14):
    """Run a faithful fp32 transcription of rmsnorm_s3.S against the op it implements.

    This is as close to testing the kernel as is possible without hardware: it is the
    same instruction sequence with the same roundings (fused madd.s/msub.s, four
    accumulators, sum*inv_d, rsqrt0.s seeded at its documented accuracy, two NR steps,
    (w*x)*s), just executed by numpy. It does not validate the encoding, register
    allocation or the ABI -- only the arithmetic. The number it prints is the parity
    threshold the device run should be judged against.
    """
    worst, worst_noscale, nrows = 0.0, 0.0, 0
    eps = _f32(1e-6)
    for x, w in sites:
        rows = x.reshape(-1, x.shape[-1])
        ref = torch.ops.espressif.rmsnorm(rows, w, float(eps))
        ref_noscale = torch.ops.espressif.rmsnorm_noscale(rows, float(eps))
        wn = w.numpy().astype(np.float32)
        for r, row in enumerate(rows.numpy().astype(np.float32)):
            nrows += 1
            d = len(row)
            t = _f32(_sum_squares_kernel(row) * _f32(_f32(1.0) / _f32(d)) + eps)
            for sign in (+1.0, -1.0):
                y0 = _f32(float(1.0 / np.sqrt(np.float64(t))) * (1.0 + sign * seed_rel))
                s = _nr(t, y0, 2)
                got = np.array([_f32(_f32(wn[i] * row[i]) * s) for i in range(d)])
                got_noscale = np.array([_f32(row[i] * s) for i in range(d)])
                exp = ref[r].numpy().astype(np.float32)
                exp_noscale = ref_noscale[r].numpy().astype(np.float32)
                scale = float(np.abs(exp).max()) or 1.0
                worst = max(worst, float(np.abs(got - exp).max()) / scale)
                scale_noscale = float(np.abs(exp_noscale).max()) or 1.0
                worst_noscale = max(
                    worst_noscale,
                    float(np.abs(got_noscale - exp_noscale).max()) / scale_noscale,
                )
    print(f"\n-- rmsnorm_s3.S transcribed to fp32 numpy, vs espressif::rmsnorm --")
    print(f"   worst relative error over {nrows} real activation rows "
          f"(seed error +/-2^-14): {worst:.2e}")
    print(f"   weightless output loop over the same rows: {worst_noscale:.2e}")
    print(f"   -> the device kernel should agree with the host op to ~{worst:.0e}")
    print("   relative per element. Arithmetic only: this does not exercise the")
    print("   encoding, the register allocation or the ABI.")
    return max(worst, worst_noscale)


def capture_norm_sites(model, idx):
    """(input, weight) for every RMSNorm, from a real forward pass."""
    sites, handles = [], []

    def make_hook(mod):
        def hook(_m, args):
            sites.append((args[0].detach().clone(), mod.weight.detach().clone()))
        return hook

    for m in model.modules():
        if isinstance(m, RMSNorm):
            handles.append(m.register_forward_pre_hook(make_hook(m)))
    with torch.no_grad():
        model(idx)
    for h in handles:
        h.remove()
    return sites


# ---------------------------------------------------------------------------
def main():
    cfg, model = build()
    idx = torch.arange(TOKENS, dtype=torch.long).remainder(cfg.vocab_size)[None, :]

    norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
    formula = 3 * cfg.n_layers + 2   # attn_norm, ffn_norm, ple_norm per block; + 2
    assert len(norms) == formula, (
        f"module tree has {len(norms)} RMSNorms, formula says {formula}"
    )
    dims = sorted({int(n.weight.numel()) for n in norms})
    print(f"model: arm={cfg.arm} D={cfg.d_model} L={cfg.n_layers} P={cfg.ple_dim} "
          f"T={TOKENS}")
    print(f"RMSNorm modules: {len(norms)}  (= 3*n_layers + 2)  row lengths {dims}")

    negative_cases()

    # -- export twice so the fused and unfused edge programs are independent ----
    with torch.no_grad():
        eager = model(idx)
    ep_a = torch.export.export(model, (idx,))
    ep_b = copy.deepcopy(ep_a)
    edge_a = to_edge(ep_a, compile_config=EDGE)
    edge_b = to_edge(ep_b, compile_config=EDGE)

    before = graph_stats(edge_a.exported_program())
    p = fuse_rmsnorm(edge_b, expect=len(norms))
    after = graph_stats(edge_b.exported_program())

    fused_nodes = [
        n for n in edge_b.exported_program().graph_module.graph.nodes
        if n.op == "call_function" and "rmsnorm" in str(n.target)
    ]
    eps_seen = sorted({float(n.args[2]) for n in fused_nodes})
    print(f"\n-- fusion on the real model --")
    print(f"   RMSNorms fused: {p.fused}   (checked against {len(norms)} RMSNorm "
          f"modules in the module tree)")
    print(f"   candidate sites considered: {p.candidates}   refused: {len(p.rejected)}")
    assert p.rejected == [], p.rejected
    assert len(fused_nodes) == len(norms)
    # eps was lifted as an fp32 tensor, so the scalar recovered from it is the fp32
    # neighbour of 1e-6, not the double. The C++ shim takes it as a double and casts
    # back to float, which round-trips exactly.
    assert eps_seen == [float(np.float32(1e-6))], eps_seen
    print(f"   eps recovered from lifted constants: {eps_seen} "
          f"(= fp32(1e-6), exactly what RMSNorm was built with)")

    # apply_passes runs this AFTER to_edge_transform_and_lower with Esp32Partitioner, so
    # verify that exact ordering rather than only the bare to_edge path. The partitioner
    # claims addmm/mm and their foldable feeders (operator_support/patterns.py's
    # SUPPORTED), none of which are in the RMSNorm chain, so every site must still be
    # visible and fusable.
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    def lower(model, idx):
        return to_edge_transform_and_lower(
            torch.export.export(model, (idx,)),
            partitioner=[Esp32Partitioner()],
            compile_config=EDGE,
        )

    deleg_a = lower(model, idx)
    deleg_b = lower(model, idx)
    pl = fuse_rmsnorm(deleg_b, expect=len(norms))
    assert pl.rejected == [], pl.rejected
    print(f"   after to_edge_transform_and_lower(Esp32Partitioner): fused {pl.fused} "
          f"of {len(norms)} -- the delegate does not swallow any norm")

    print(f"\n-- graph size --")
    for k, label in (("nodes", "total nodes"),
                     ("call_function", "call_function"),
                     ("intermediates", "intermediate tensors"),
                     ("placeholders", "placeholders")):
        d = after[k] - before[k]
        print(f"   {label:22s} {before[k]:5d} -> {after[k]:5d}  ({d:+d})")
    assert after["call_function"] < before["call_function"]
    # 6 nodes out, 1 in, per site.
    assert before["call_function"] - after["call_function"] == 5 * len(norms), (
        before["call_function"], after["call_function"])
    assert after["dead_placeholders"] - before["dead_placeholders"] == len(norms), (
        before["dead_placeholders"], after["dead_placeholders"])
    print(f"   eps constants now dead placeholders: "
          f"{after['dead_placeholders'] - before['dead_placeholders']} of {len(norms)}, "
          f"{4 * len(norms)} B of constant segment left behind "
          f"(see fuse_rmsnorm's note on why the signature is not rewritten)")

    # -- numerics ------------------------------------------------------------
    with torch.no_grad():
        out_a = edge_a.exported_program().module()(idx)
        out_b = edge_b.exported_program().module()(idx)
    out_a = out_a[0] if isinstance(out_a, (tuple, list)) else out_a
    out_b = out_b[0] if isinstance(out_b, (tuple, list)) else out_b

    def agree(u, v):
        u, v = u.double(), v.double()
        amax = float((u - v).abs().max())
        scale = float(v.abs().max()) or 1.0
        return amax, amax / scale, int((u != v).sum()), u.numel()

    a_abs, a_rel, a_ne, n = agree(out_b, out_a)
    e_abs, e_rel, _, _ = agree(out_b, eager[0])
    u_abs, u_rel, _, _ = agree(out_a, eager[0])
    print(f"\n-- numerical agreement, fused vs unfused edge program --")
    print(f"   max |diff| {a_abs:.3e}   relative to max|out| {a_rel:.3e}   "
          f"differing elements {a_ne}/{n}")
    print(f"   unfused vs eager: rel {u_rel:.3e}     fused vs eager: rel {e_rel:.3e}")
    # The custom op's CompositeExplicitAutograd impl replays the same ops in the same
    # order, so this is exact. Tolerance is stated as 0 and asserted as 0; anything
    # nonzero means the impl and the pattern have drifted apart.
    assert a_ne == 0 and a_abs == 0.0, (
        f"fused and unfused disagree on {a_ne} elements (max {a_abs:.3e}); the "
        "espressif::rmsnorm impl no longer matches the subgraph it replaces"
    )
    print("   tolerance: 0 (bit-exact). The op's impl replays the identical op order,")
    print("   so any nonzero difference is a real divergence, not float noise.")

    # -- what it costs the program -------------------------------------------
    ea = emit(edge_a, "unfused")
    eb = emit(edge_b, "fused")
    print(f"\n-- emitted program (whole model) --")
    for k, label, unit in (("peak_bytes", "memory-plan peak", "B"),
                           ("demand_bytes", "planned tensor bytes", "B"),
                           ("planned_tensors", "planned tensors", ""),
                           ("values", "EValues in the plan", ""),
                           ("instructions", "instructions", ""),
                           ("pte_bytes", ".pte size", "B")):
        d = eb[k] - ea[k]
        pct = 100.0 * d / ea[k] if ea[k] else 0.0
        print(f"   {label:22s} {ea[k]:9,d} -> {eb[k]:9,d} {unit:1s} "
              f"({d:+,d}, {pct:+.1f}%)")

    out_op = "espressif::rmsnorm.out"
    assert out_op in eb["ops"], (
        f"to_executorch() did not produce {out_op}; got {sorted(set(eb['ops']))}"
    )
    n_out = sum(1 for c in eb["ops"] if c == out_op)
    print(f"   {out_op} in the emitted operator table: yes "
          f"({n_out} entr{'y' if n_out == 1 else 'ies'})")
    gone = sorted(set(ea["ops"]) - set(eb["ops"]))
    print(f"   operators no longer needed at all: {gone if gone else 'none'}")

    blk = block_memory_plan(cfg)
    bu, bf = blk["unfused"], blk["fused"]
    print(f"\n-- emitted program (one Block: 3 RMSNorms, no output head) --")
    print("   Measured separately because the whole model's arena is dominated by the")
    print("   output head.")
    for k, label in (("peak_bytes", "memory-plan peak"),
                     ("demand_bytes", "planned tensor bytes"),
                     ("planned_tensors", "planned tensors"),
                     ("instructions", "instructions")):
        d = bf[k] - bu[k]
        pct = 100.0 * d / bu[k] if bu[k] else 0.0
        print(f"   {label:22s} {bu[k]:9,d} -> {bf[k]:9,d}   ({d:+,d}, {pct:+.1f}%)")
    assert bf["peak_bytes"] <= bu["peak_bytes"], (bu["peak_bytes"], bf["peak_bytes"])
    assert bf["demand_bytes"] < bu["demand_bytes"], (bu, bf)

    # ---- the build that actually ships -------------------------------------
    # The two measurements above use plain to_edge, where the arena is dominated by
    # permute_copy of the weight matrices. Esp32Partitioner folds those transposes in at
    # build time (in preprocess: applied at build time, never emitted), so the
    # delegated program is the only one whose arena number means anything on device.
    da = emit(deleg_a, "delegated unfused")
    db = emit(deleg_b, "delegated fused")
    print(f"\n-- emitted program (delegated: Esp32Partitioner + fusion) --")
    for k, label in (("peak_bytes", "memory-plan peak"),
                     ("demand_bytes", "planned tensor bytes"),
                     ("planned_tensors", "planned tensors"),
                     ("instructions", "instructions"),
                     ("pte_bytes", ".pte size")):
        d = db[k] - da[k]
        pct = 100.0 * d / da[k] if da[k] else 0.0
        print(f"   {label:22s} {da[k]:9,d} -> {db[k]:9,d}   ({d:+,d}, {pct:+.1f}%)")

    # ---- what sets each memory-plan peak -----------------------------------
    # Fusion might be expected to shrink the memory plan. Report where it does and
    # where it does not, rather than only the flattering half.
    print(f"\n-- what sets each memory-plan peak --")
    for tag, un, fu in (("whole model", ea, eb), ("one Block", bu, bf),
                        ("delegated", da, db)):
        moved = fu["peak_bytes"] - un["peak_bytes"]
        print(f"   {tag:12s} {un['peak_bytes']:9,d} -> {fu['peak_bytes']:9,d} B "
              f"({moved:+,d}); high-water mark is set by:")
        for nbytes, off, name in un["biggest"][:3]:
            print(f"      {nbytes:9,d} B at offset {off:9,d}  {name}")
    print("   Greedy planning packs each norm's ~6 KB of temporaries into space that")
    print("   longer-lived tensors need anyway, so on the non-delegated programs")
    print("   deleting them cuts total demand without cutting the arena.")

    # -- kernel-side numbers the .S header cites -----------------------------
    rsqrt_table()
    sites = capture_norm_sites(model, idx)
    summation_order(sites)
    kerr = kernel_transcription(sites)
    # Loose bound: this is one rounding of the mean plus ~1 ulp of rsqrt, so anything
    # above 1e-5 means the transcription and the assembly have diverged.
    assert kerr < 1e-5, kerr

    print(f"\nOK: {p.fused} RMSNorms fused (expected {len(norms)}), "
          f"{before['call_function']} -> {after['call_function']} call_function nodes, "
          f"bit-exact against the unfused program, kernel arithmetic within "
          f"{kerr:.1e} relative.")


def test_main():
    """Under pytest, what a script run checks: main() asserts each check
    (or returns non-zero), and this fails with it."""
    assert main() in (None, 0)


if __name__ == "__main__":
    main()
