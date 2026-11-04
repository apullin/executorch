# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion passes, tested as functional claims: an op VANISHES and the numbers hold.

Two fusions of two different kinds, which is the point of testing them side by side:

  conv+bn fold      constant algebra. A graph pass rewrites arithmetic on constant
                    operands; constant_prop materialises it; the emitted program has
                    one conv with pre-multiplied weights and no batch-norm anywhere.
                    Nothing at runtime changes.

  matmul+relu       backend fusion. The op survives to the DELEGATE as a flag, and
                    the clamp runs in registers between accumulate and store. The
                    graph claim here is absence-from-the-remainder, not absence.

Numerics are checked against the UNFUSED eager module. Not bit-exact by design:
folding computes W*s in fp32, which rounds differently from applying bn after the
conv -- so the bound is a tolerance, and the tolerance is stated where it is used.
The relu half's numerics are covered by the parity rows (every delegated MLP
carries the flag), so here it is checked structurally.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test fails
naming its failed checks (@case).

    python test_fusions.py
"""

import functools
import sys
from collections import Counter

import torch
import torch.nn as nn

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


def _fn(f):
    """Wrap a plain function as a Module."""

    class F(nn.Module):
        def forward(self, x):
            return f(x)

    return F()


def _edge(model, x):
    from executorch.exir import to_edge_transform_and_lower

    ep = torch.export.export(model.eval(), (x,))
    return to_edge_transform_and_lower(ep)


def _bn_count(edge):
    gm = edge.exported_program().graph_module
    return sum(1 for n in gm.graph.nodes
               if n.op == "call_function" and "batch_norm" in str(n.target))


def _fold(edge):
    from executorch.backends.esp32s3._passes import apply_passes

    return apply_passes(edge, quiet=True)


def _run_edge(edge, x):
    out = edge.exported_program().module()(x)
    return out[0] if isinstance(out, (tuple, list)) else out


# ---------------------------------------------------------------------------
@case
def test_fold_variants():
    """conv->bn folds across the argument space, matching unfused eager."""
    print("\n-- conv+bn fold: variants --")
    cases = {
        "conv_bias+affine": (nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8)),
        "conv_nobias": (nn.Conv2d(4, 8, 3, padding=1, bias=False),
                        nn.BatchNorm2d(8)),
        "bn_no_affine": (nn.Conv2d(4, 8, 3, padding=1),
                         nn.BatchNorm2d(8, affine=False)),
        "conv_grouped": (nn.Conv2d(8, 8, 3, padding=1, groups=2),
                         nn.BatchNorm2d(8)),
        "conv1d": (nn.Conv1d(4, 8, 3, padding=1), nn.BatchNorm1d(8)),
    }
    for name, (conv, bn) in cases.items():
        torch.manual_seed(0)
        # Non-trivial running stats, or the fold is multiplying by ~1.
        with torch.no_grad():
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
        m = nn.Sequential(conv, bn).eval()
        cin = conv.in_channels
        x = (torch.randn(1, cin, 10, 10) if isinstance(conv, nn.Conv2d)
             else torch.randn(1, cin, 10))
        with torch.no_grad():
            ref = m(x)
        edge = _edge(m, x)
        before = _bn_count(edge)
        rep = _fold(edge)
        after = _bn_count(edge)
        got = _run_edge(edge, x)
        err = float((got - ref).abs().max())
        check(f"{name}: bn {before}->0, folded=1",
              before == 1 and after == 0 and rep["convbn"] == 1)
        check(f"{name}: matches unfused eager", err < 1e-5, f"(max {err:.2e})")
        # And the folded program must still EMIT -- the signature-surgery failure
        # mode dies here, which is exactly where ExecuTorch's own pass falls over.
        try:
            edge.to_executorch()
            check(f"{name}: emits", True)
        except Exception as e:
            check(f"{name}: emits", False, f"{type(e).__name__}")


@case
def test_fold_refusals():
    """Sites the fold must NOT touch, each for a stated reason."""
    print("\n-- conv+bn fold: refusals --")
    import contextlib
    import io

    torch.manual_seed(0)

    # Standalone bn: no conv producer.
    m = nn.Sequential(nn.BatchNorm2d(4)).eval()
    x = torch.randn(1, 4, 8, 8)
    edge = _edge(m, x)
    rep = _fold(edge)
    check("standalone bn survives", _bn_count(edge) == 1 and rep["convbn"] == 0)

    # Transposed conv: channel axis differs; folding would scale the wrong dim.
    m = nn.Sequential(nn.ConvTranspose2d(4, 8, 2, stride=2),
                      nn.BatchNorm2d(8)).eval()
    edge = _edge(m, torch.randn(1, 4, 8, 8))
    rep = _fold(edge)
    check("transposed conv refused, reason recorded",
          _bn_count(edge) == 1 and rep["convbn"] == 0
          and any("transposed" in r for _, r in rep["convbn_rejected"]))

    # Conv whose pre-bn output is also consumed directly.
    class MultiUse(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(4, 4, 3, padding=1)
            self.b = nn.BatchNorm2d(4)

        def forward(self, x):
            y = self.c(x)
            return self.b(y) + y

    m = MultiUse().eval()
    x = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        ref = m(x)
    edge = _edge(m, x)
    rep = _fold(edge)
    got = _run_edge(edge, x)
    check("multi-use conv refused, numerics untouched",
          _bn_count(edge) == 1 and rep["convbn"] == 0
          and float((got - ref).abs().max()) < 1e-6)

    # The fold inserts arithmetic on its operands and relies on constant
    # propagation to materialise it, so every operand must be a build-time
    # constant -- and folding cannot run without the propagation.
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3._passes import apply_passes

    class RuntimeWeightConvBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.BatchNorm2d(4)

        def forward(self, x, w):
            return self.b(torch.nn.functional.conv2d(x, w, padding=1))

    m = RuntimeWeightConvBN().eval()
    args = (torch.randn(1, 4, 8, 8), torch.randn(4, 4, 3, 3))
    with torch.no_grad():
        ref = m(*args)
    edge = to_edge_transform_and_lower(torch.export.export(m, args))
    try:
        rep = apply_passes(edge, quiet=True)
        got = edge.exported_program().module()(*args)
        edge.exported_program().validate()
        ok = rep["convbn"] == 0 and float((got - ref).abs().max()) < 1e-5
        detail = f"folded {rep['convbn']}"
    except Exception as e:  # noqa: BLE001 -- any failure is the bug
        ok, detail = False, f"{type(e).__name__}: {str(e)[:70]}"
    check("a runtime conv weight is refused; the program stays valid", ok,
          detail)
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import lower_esp32

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lower_esp32(torch.export.export(m, args), [Esp32Partitioner()])
        ok, detail = True, ""
    except Exception as e:  # noqa: BLE001 -- any failure is the bug
        ok, detail = False, f"{type(e).__name__}: {str(e)[:70]}"
    check("and lower_esp32 lowers it (bn left portable)", ok, detail)

    edge = _edge(nn.Sequential(nn.Conv2d(4, 4, 3, padding=1),
                               nn.BatchNorm2d(4)).eval(), x)
    try:
        apply_passes(edge, fold=False, quiet=True)
        edge.exported_program().validate()
        ok, detail = True, ""
    except Exception as e:  # noqa: BLE001 -- any failure is the bug
        ok, detail = False, f"{type(e).__name__}: {str(e)[:70]}"
    check("fold=False leaves conv+bn a valid program", ok, detail)


def _lower_esp32(m, args, **options):
    """lower_esp32 with an Esp32Partitioner: (edge or None, failure text)."""
    import contextlib
    import io

    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return lower_esp32(torch.export.export(m, args),
                               [Esp32Partitioner(**options)]), ""
    except Exception as e:  # noqa: BLE001 -- any failure is the bug
        return None, f"{type(e).__name__}: {str(e)[:90]}"


def _valid(edge):
    """Whether the program validates (its signature against its graph
    among the rest): (ok, failure text)."""
    try:
        edge.exported_program().validate()
        return True, ""
    except Exception as e:  # noqa: BLE001 -- any failure is the bug
        return False, f"{type(e).__name__}: {str(e)[:90]}"


@case
def test_fold_at_graph_outputs():
    """A folded bn whose result is a graph output: the conv's result takes
    its place, and the signature, which names every output by its node,
    names the conv's. Left naming the erased getitem, the program would
    fail to_backend's validation before partitioning (SpecViolationError,
    'User output ... is not in the correct order')."""
    print("\n-- conv+bn fold: the bn's result is a graph output --")

    class Twice(nn.Module):
        """bn(conv(x)) returned twice beside another output, as a generated
        graph can return it."""

        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(4, 8, 3, padding=1)
            self.b = nn.BatchNorm2d(8)

        def forward(self, x):
            y = self.b(self.c(x))
            return y, x.abs(), y

    x = torch.randn(1, 4, 8, 8)
    for name, make in (
            ("the output", lambda: nn.Sequential(
                nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8))),
            ("twice among the outputs", Twice)):
        torch.manual_seed(0)
        m = make().eval()
        bn = next(b for b in m.modules() if isinstance(b, nn.BatchNorm2d))
        with torch.no_grad():
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
            ref = m(x)
        ref = ref if isinstance(ref, tuple) else (ref,)
        # conv=False keeps the folded conv portable, so the lowered program
        # runs here and its numbers can be compared.
        for options, how in (({}, "conv delegated"), ({"conv": False},
                                                      "conv portable")):
            edge, why = _lower_esp32(m, (x,), **options)
            check(f"bn result {name}: lower_esp32 lowers it ({how})",
                  edge is not None, why)
            if edge is None:
                continue
            ok, why = _valid(edge)
            check(f"bn result {name}: folded, the signature names the "
                  f"outputs ({how})", ok and _bn_count(edge) == 0, why)
            if options:
                got = edge.exported_program().module()(x)
                got = (got,) if isinstance(got, torch.Tensor) else tuple(got)
                err = max(float((g - r).abs().max()) for g, r in zip(got, ref))
                check(f"bn result {name}: matches unfused eager ({how})",
                      err < 1e-5, f"(max {err:.2e})")
        # The same fold after lowering, in apply_passes.
        edge = _edge(m, x)
        rep = _fold(edge)
        ok, why = _valid(edge)
        check(f"bn result {name}: apply_passes' fold leaves a valid program",
              ok and rep["convbn"] == 1, why)


@case
def test_constant_output_keeps_the_dim_order():
    """A parameter returned as it is, and an output computed from it alone,
    in a program the fold changes: the constant propagation that
    materialises the fold folds the second to a constant and clones both,
    in the program's own dim-order mode. Its default makes the clone an
    aten.clone, which the edge verifier refuses in a program with dim order,
    and to_backend validates before partitioning (SpecViolationError,
    'aten.clone.default ... should not be used in dim_order mode');
    apply_passes' propagation must not leave the same op behind."""
    print("\n-- conv+bn fold: a constant graph output --")

    class ConvBnAndParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(4, 4, 3, padding=1)
            self.b = nn.BatchNorm2d(4)
            self.p = nn.Parameter(torch.randn(3))

        def forward(self, x):
            return torch.relu(self.b(self.c(x))), self.p, self.p.abs()

    torch.manual_seed(0)
    m = ConvBnAndParam().eval()
    x = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        ref = m(x)
    for options, how in (({}, "conv delegated"), ({"conv": False},
                                                  "conv portable")):
        edge, why = _lower_esp32(m, (x,), **options)
        check(f"lower_esp32 lowers it ({how})", edge is not None, why)
        if edge is None:
            continue
        ok, why = _valid(edge)
        check(f"the lowered program is valid ({how})", ok, why)
        if options:
            got = edge.exported_program().module()(x)
            check(f"matches eager, the constant outputs exactly ({how})",
                  float((got[0] - ref[0]).abs().max()) < 1e-5
                  and torch.equal(got[1], ref[1])
                  and torch.equal(got[2], ref[2]))
    edge = _edge(m, x)
    _fold(edge)
    ok, why = _valid(edge)
    clones = [str(n.target) for n in edge.exported_program().graph.nodes
              if n.op == "call_function" and "aten.clone" in str(n.target)]
    check("apply_passes' propagation clones it in the dim-order spelling",
          ok and not clones, why or f"{clones}")
    got = edge.exported_program().module()(x)
    check("and the folded program returns the constant outputs exactly",
          torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2]))
    try:
        edge.to_executorch()
        check("and emits", True)
    except Exception as e:  # noqa: BLE001 -- any failure is the bug
        check("and emits", False, f"{type(e).__name__}: {str(e)[:70]}")


@case
def test_relu_into_delegate():
    """The relu vanishes from the portable remainder and lands in the delegate."""
    print("\n-- matmul+relu into the delegate --")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.models import MLP, ramp_input

    from executorch.backends.esp32s3.test.models import MLPR

    def portable_ops(model, x):
        ep = torch.export.export(model.eval(), (x,))
        edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
        gm = edge.exported_program().graph_module
        c = Counter()
        for n in gm.graph.nodes:
            if n.op == "call_function":
                t = getattr(n.target, "_name", None) or str(n.target)
                c["delegate" if "call_delegate" in str(n.target) else t] += 1
        return c

    # Mid-chain relu feeds another claimable matmul: intra-partition dataflow
    # claims it AND the weld is the point -- ONE delegate, the relu riding the
    # first matmul's epilogue and the intermediate a scratch slot, never a
    # portable round trip.
    torch.manual_seed(0)
    c = portable_ops(MLP(), ramp_input((1, 96)))
    check("MLP: ONE delegate, mid-chain relu chained inside it",
          c["delegate"] == 1 and sum(v for k, v in c.items() if "relu" in k) == 0,
          f"({dict(c)})")

    # Both relus of MLPR land inside the same single delegate: the mid one
    # chains, the final one fuses into the last matmul's epilogue.
    torch.manual_seed(0)
    c = portable_ops(MLPR(), ramp_input((1, 96)))
    check("MLPR: ONE delegate, both relus inside",
          c["delegate"] == 1 and sum(v for k, v in c.items() if "relu" in k) == 0,
          f"({dict(c)})")

    # A matmul whose pre-relu value is ALSO consumed must keep its relu outside.
    class PreActUsed(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            y = self.fc(x)
            return torch.relu(y) + y

    ep = torch.export.export(PreActUsed().eval(), (torch.randn(1, 16),))
    edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
    gm = edge.exported_program().graph_module
    relus = [n for n in gm.graph.nodes
             if n.op == "call_function" and "relu" in str(n.target)]
    check("multi-use pre-activation keeps its relu portable", len(relus) == 1)


def main():
    for t in (test_fold_variants, test_fold_refusals,
              test_fold_at_graph_outputs,
              test_constant_output_keeps_the_dim_order,
              test_relu_into_delegate,
              test_clamp_canonicalisation, test_clamp_chain_into_delegate,
              test_act_into_delegate, test_view_transparent_fusion,
              test_shared_cleanup, test_pass_switches, test_chained_linears,
              test_qk_optin, test_gated_mul_into_delegate,
              test_acts_into_delegate):
        t.__wrapped__()
    print()
    if FAILS:
        print(f"FAILURES: {len(FAILS)}")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("OK: both fusions fire where they should, refuse where they must, "
          "and the numbers hold.")
    return 0



@case
def test_clamp_canonicalisation():
    """Accidental clamp chains collapse to ONE clamp, bit-exactly.

    Bit-exact is the whole claim: composition is pure min/max, so unlike the bn fold
    there is no rounding to tolerate -- torch.equal or bust. The composition rule is
    lo' = clamp(l1, l2, h2), hi' = clamp(h1, l2, h2), which the crossed-interval case
    below exists to defend: the naive (max(lo), min(hi)) gets it wrong.
    """
    print("\n-- clamp canonicalisation --")
    from executorch.backends.esp32s3._passes.canonicalize_clamps_pass import (
        canonicalize_clamps,
    )

    def collapse(fn, x, expect_removed):
        m = _fn(fn)
        with torch.no_grad():
            ref = m(x)
        edge = _edge(m, x)
        removed = canonicalize_clamps(edge)
        got = _run_edge(edge, x)
        fam = sum(1 for n in edge.exported_program().graph_module.graph.nodes
                  if n.op == "call_function"
                  and any(t in str(n.target) for t in ("relu", "hardtanh", "clamp")))
        return removed, fam, bool(torch.equal(got, ref))

    x = torch.randn(2, 64) * 3

    r, fam, exact = collapse(lambda t: torch.relu(torch.relu(t)), x, 1)
    check("relu(relu(x)): 1 removed, 1 clamp left, bit-exact",
          r == 1 and fam == 1 and exact)

    r, fam, exact = collapse(
        lambda t: torch.clamp(torch.relu(t), max=4.0), x, 1)
    check("clamp(relu(x), max=4): composed to [0,4], bit-exact",
          r == 1 and fam == 1 and exact)

    r, fam, exact = collapse(
        lambda t: torch.nn.functional.relu6(torch.relu(t)), x, 1)
    check("relu6(relu(x)) [hardtanh chain]: composed, bit-exact",
          r == 1 and fam == 1 and exact)

    r, fam, exact = collapse(
        lambda t: torch.clamp(torch.clamp(torch.relu(t), max=5.0), 1.0, 3.0), x, 2)
    check("triple chain: 2 removed, one [1,3] clamp, bit-exact",
          r == 2 and fam == 1 and exact)

    # The crossed case the composition rule exists for: inner pins to [2,inf),
    # outer to (-inf,1] -- the result is CONSTANT 1, not the empty naive [2,1].
    r, fam, exact = collapse(
        lambda t: torch.clamp(torch.clamp(t, min=2.0), max=1.0), x, 1)
    check("crossed intervals collapse to the constant, bit-exact",
          r == 1 and exact)

    # Only +inf is an open upper bound. max=-inf pins everything to -inf, and
    # min=+inf everything to +inf; neither is the identity.
    r, fam, exact = collapse(
        lambda t: torch.clamp(torch.relu(t), max=float("-inf")), x, 1)
    check("clamp(relu(x), max=-inf) stays -inf, bit-exact", r == 1 and exact)
    r, fam, exact = collapse(
        lambda t: torch.clamp(torch.relu(t), min=float("inf")), x, 1)
    check("clamp(relu(x), min=+inf) stays +inf, bit-exact", r == 1 and exact)

    # A lone relu is not a chain; churn would buy nothing.
    r, fam, exact = collapse(lambda t: torch.relu(t), x, 0)
    check("single relu untouched", r == 0 and fam == 1 and exact)

    # Property check: random 3-deep chains vs their composed single clamp, exact.
    import math
    import random

    from executorch.backends.esp32s3.operator_support.patterns import compose_clamp

    random.seed(7)
    ok = True
    for _ in range(200):
        iv = []
        for _k in range(3):
            lo = random.choice([-math.inf, round(random.uniform(-3, 1), 2)])
            hi = random.choice([math.inf, round(random.uniform(-1, 3), 2)])
            iv.append((lo, hi))
        def apply(t, lo, hi):
            # A fully-open interval is the identity; torch.clamp refuses it rather
            # than no-op, so the identity is applied by hand.
            if lo == -math.inf and hi == math.inf:
                return t
            return torch.clamp(t, None if lo == -math.inf else lo,
                               None if hi == math.inf else hi)

        y = x.clone()
        for lo, hi in iv:
            y = apply(y, lo, hi)
        comp = iv[0]
        for nxt in iv[1:]:
            comp = compose_clamp(nxt, comp)
        z = apply(x, *comp)
        ok &= bool(torch.equal(y, z))
    check("200 random 3-chains == composed single clamp, bit-exact", ok)


@case
def test_clamp_chain_into_delegate():
    """The accidental chain rides into the delegate as ONE composed interval."""
    print("\n-- clamp chain into the delegate --")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.models import MLPR, ramp_input

    torch.manual_seed(0)
    ep = torch.export.export(MLPR().eval(), (ramp_input((1, 96)),))
    edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
    gm = edge.exported_program().graph_module
    fam = [n for n in gm.graph.nodes
           if n.op == "call_function"
           and any(t in str(n.target) for t in ("relu", "hardtanh", "aten::clamp"))]
    # Nothing survives: the final relu->clamp pair fuses as one [0,4] interval, and
    # the mid-chain relu chains the two matmuls inside one delegate.
    check("no clamp-family op remains portable (mid relu chains)", len(fam) == 0,
          f"({[str(n.target) for n in fam]})")


@case
def test_act_into_delegate():
    """The silu diamond and exact gelu fuse; the tanh-gelu and merge cases refuse."""
    print("\n-- silu / gelu into the delegate --")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.models import MLPS, ramp_input

    def census(model, x):
        ep = torch.export.export(model.eval(), (x,))
        edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
        c = Counter()
        for n in edge.exported_program().graph_module.graph.nodes:
            if n.op == "call_function":
                t = getattr(n.target, "_name", None) or str(n.target)
                c["delegate" if "call_delegate" in str(n.target) else t] += 1
        return c

    torch.manual_seed(0)
    c = census(MLPS(), ramp_input((1, 96)))
    # ONE delegate: the trunk relu chains, so fc1 and both heads land in a
    # single 3-op partition -- the heads reading the relu'd trunk from a scratch
    # slot (fan-out), each fused act binding to its own matmul inside it. Pinned
    # deliberately: if this splits, chaining regressed and the multi-op scratch
    # fan-out lost a fixture.
    check("MLPS: trunk+heads chain into one 3-op delegate; all acts gone",
          c["delegate"] == 1
          and not any("sigmoid" in k or "gelu" in k for k in c)
          and sum(v for k, v in c.items() if "relu" in k) == 0,
          f"({dict(c)})")

    # tanh-approximate gelu fuses too: the epilogue runs ExecuTorch's own
    # tanh formula, so it is the same op, not a substitute.
    class TanhGelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            return torch.nn.functional.gelu(self.fc(x), approximate="tanh")

    c = census(TanhGelu(), torch.randn(1, 16))
    check("tanh-approximate gelu fuses",
          not any("gelu" in k for k in c) and c["delegate"] == 1,
          f"({dict(c)})")

    # A silu whose mul feeds another claimable matmul: with intra-partition
    # dataflow the intermediate rides a scratch slot, the whole chain is ONE
    # delegate and the diamond vanishes from the remainder.
    class SiluMid(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(16, 16), nn.Linear(16, 16)

        def forward(self, x):
            return self.fc2(torch.nn.functional.silu(self.fc1(x)))

    c = census(SiluMid(), torch.randn(1, 16))
    check("mid-chain silu chains into ONE delegate",
          not any("sigmoid" in k for k in c) and c["delegate"] == 1,
          f"({dict(c)})")

    # The runtime applies the activation and THEN the clamp, so a clamp over a
    # fused silu or gelu belongs in the same epilogue rather than portable.
    class ActThenRelu(nn.Module):
        def __init__(self, act):
            super().__init__()
            self.fc, self.act = nn.Linear(16, 16), act

        def forward(self, x):
            return torch.relu(self.act(self.fc(x)))

    for name, act in (("silu", nn.SiLU()), ("gelu", nn.GELU()),
                      ("sigmoid", nn.Sigmoid()),
                      ("tanh-gelu", nn.GELU(approximate="tanh"))):
        c = census(ActThenRelu(act), torch.randn(1, 16))
        check(f"relu({name}(linear)): ONE delegate, no portable act or relu",
              c["delegate"] == 1 and not any(
                  t in k for k in c for t in ("relu", "sigmoid", "gelu")),
              f"({dict(c)})")

    # Except leaky ReLU, whose slope rides the clamp's own field: the leaky
    # fuses and the relu over it stays portable.
    c = census(ActThenRelu(nn.LeakyReLU(0.2)), torch.randn(1, 16))
    check("relu(leaky(linear)): leaky fused, relu portable",
          c["delegate"] == 1 and not any("leaky" in k for k in c)
          and sum(v for k, v in c.items() if "relu" in k) == 1,
          f"({dict(c)})")

    # The residual walk roots exactly the chains the gates claim: through a
    # clamp over a clamp-compatible activation, never through one over leaky.
    class ActThenReluRes(ActThenRelu):
        def forward(self, x, r):
            return torch.relu(self.act(self.fc(x))) + r

    for name, act, fused in (("sigmoid", nn.Sigmoid(), True),
                             ("leaky", nn.LeakyReLU(0.2), False)):
        ep = torch.export.export(ActThenReluRes(act).eval(),
                                 (torch.randn(1, 16), torch.randn(1, 16)))
        edge = to_edge_transform_and_lower(
            ep, partitioner=[Esp32Partitioner()])
        c = Counter(
            "delegate" if "call_delegate" in str(n.target)
            else getattr(n.target, "_name", None) or str(n.target)
            for n in edge.exported_program().graph_module.graph.nodes
            if n.op == "call_function")
        adds = sum(v for k, v in c.items() if "add" in k and "addmm" not in k)
        check(f"relu({name}(linear)) + r: lowers, add "
              f"{'fused' if fused else 'portable'}",
              c["delegate"] == 1 and adds == (0 if fused else 1),
              f"({dict(c)})")


@case
def test_view_transparent_fusion():
    """3-D F.linear traces as view->mm->view; fusion must see through the reshape.

    F.linear on a BUFFER weight is the forcing function -- nn.Linear's module trace
    lands activations on a naked addmm, so fixtures built from nn.Linear alone
    cannot show whether a model written with F.linear fuses. The pattern sees
    through the reshape, not the model.
    """
    print("\n-- view-transparent fusion (3-D F.linear) --")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    def census(model, x):
        ep = torch.export.export(model.eval(), (x,))
        edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
        c = Counter()
        for n in edge.exported_program().graph_module.graph.nodes:
            if n.op == "call_function":
                t = getattr(n.target, "_name", None) or str(n.target)
                c["delegate" if "call_delegate" in str(n.target) else t] += 1
        return c

    class FL(nn.Module):
        def __init__(self, act):
            super().__init__()
            g = torch.Generator().manual_seed(3)
            self.register_buffer("w", torch.randn(24, 16, generator=g) * 0.1)
            self.act = act

        def forward(self, x):
            return self.act(torch.nn.functional.linear(x, self.w))

    x = torch.randn(1, 8, 16)

    c = census(FL(torch.nn.functional.silu), x)
    check("3-D silu diamond fuses through the view",
          not any("sigmoid" in k for k in c), f"({dict(c)})")

    c = census(FL(torch.nn.functional.gelu), x)
    check("3-D exact gelu fuses through the view",
          not any("gelu" in k for k in c), f"({dict(c)})")

    c = census(FL(lambda t: torch.clamp(torch.relu(t), max=4.0)), x)
    check("3-D relu->clamp chain fuses through the view",
          not any(("relu" in k or "aten::clamp" in k) for k in c), f"({dict(c)})")

    # Refusal carries over: a fused tail feeding another claimable matmul would
    # still merge partitions, view or no view.
    class Mid(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(4)
            self.register_buffer("w1", torch.randn(16, 16, generator=g) * 0.1)
            self.register_buffer("w2", torch.randn(8, 16, generator=g) * 0.1)

        def forward(self, x):
            h = torch.nn.functional.silu(torch.nn.functional.linear(x, self.w1))
            return torch.nn.functional.linear(h, self.w2)

    c = census(Mid(), x)
    # The 2-D case must REFUSE because claiming the act would weld two matmul
    # partitions. In 3-D the next linear's input view is unclaimed and provides the
    # partition break itself -- so the silu fuses AND the partitions stay separate,
    # which is strictly better than refusing. The requirement is not "refuse";
    # it is "never merge": assert both halves of that directly.
    check("mid-chain 3-D silu fuses WITHOUT merging (view is the break)",
          not any("sigmoid" in k for k in c) and c["delegate"] == 2,
          f"({dict(c)})")


@case
def test_pass_switches():
    """apply_passes(fuse=, fold=) exists so one pass can be measured against the
    other, which only works if every combination runs and reports."""
    print("\n-- apply_passes fuse/fold switches --")
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes import apply_passes

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(16))

        def forward(self, x):
            return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    for fuse in (True, False):
        for fold in (True, False):
            edge = to_edge(torch.export.export(Norm().eval(), (torch.randn(2, 16),)))
            try:
                rep = apply_passes(edge, fuse=fuse, fold=fold, quiet=True)
                ok = rep["fused"] == (1 if fuse else 0)
                detail = f"fused {rep['fused']}"
            except Exception as e:  # noqa: BLE001 -- any failure is the bug
                ok, detail = False, f"{type(e).__name__}: {e}"
            check(f"fuse={fuse}, fold={fold}: runs and reports", ok, detail)


@case
def test_shared_cleanup():
    """The adopted executorch.backends.transforms hygiene, as a functional claim:
    identity clones and stacked reshapes on the PORTABLE side vanish under
    apply_passes, and the numbers hold. Post-partition by construction -- every
    delegated exporter partitions first -- so the claim is about the remainder,
    not about pattern visibility (see the note in _passes.apply_passes)."""
    print("\n-- shared clone/view cleanup (executorch.backends.transforms) --")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes

    class Messy(nn.Module):
        """A delegated linear surrounded by portable mess: an identity clone on
        its input and a two-step reshape chain on its output."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 24)

        def forward(self, x):
            h = torch.tanh(x).clone()      # identity clone, portable side
            y = self.fc(h)                 # delegated
            y = y.reshape(2, 4, 24)        # stacked reshapes, portable side
            y = y.reshape(8, 24)
            return torch.tanh(y)

    torch.manual_seed(0)
    m = Messy().eval()
    x = torch.randn(8, 16)
    ep = torch.export.export(m, (x,))
    edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
    gm = edge.exported_program().graph_module
    pre = Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
    y_pre = edge.exported_program().module()(x)[0]

    apply_passes(edge, quiet=True)
    gm = edge.exported_program().graph_module
    post = Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
    y_post = edge.exported_program().module()(x)[0]

    n_clone_pre = sum(v for k, v in pre.items() if "clone" in k)
    n_view_pre = sum(v for k, v in pre.items() if "view_copy" in k)
    n_clone_post = sum(v for k, v in post.items() if "clone" in k)
    n_view_post = sum(v for k, v in post.items() if "view_copy" in k)
    check("portable clone and view chain existed before cleanup",
          n_clone_pre >= 1 and n_view_pre >= 2,
          f"(clone {n_clone_pre}, view {n_view_pre})")
    check("cleanup removed them all",
          n_clone_post == 0 and n_view_post == 0,
          f"(clone {n_clone_post}, view {n_view_post})")
    # Bit-exact, not approximately equal: the passes rewire and rename, they do
    # not compute.
    check("outputs bit-identical across the cleanup",
          bool(torch.equal(y_pre, y_post)))


@case
def test_chained_linears():
    """Intra-partition dataflow: mm -> act -> mm chains run inside ONE delegate,
    the intermediate riding a scratch slot (or its own output arg) instead of a
    portable round trip. Census-level here; the chained NUMBERS are held by
    run_parity (the mlp/mlpr references do not depend on how the program is
    partitioned, so the chain must reproduce them bit-for-bit)."""
    print("\n-- intra-partition chained linears --")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    def census(model, x):
        ep = torch.export.export(model.eval(), (x,))
        edge = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
        c = Counter()
        for n in edge.exported_program().graph_module.graph.nodes:
            if n.op == "call_function":
                t = getattr(n.target, "_name", None) or str(n.target)
                c["delegate" if "call_delegate" in str(n.target) else t] += 1
        return c

    # Three matmuls, two different act links: needs BOTH scratch slots (s0, s1).
    class Chain3(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(16, 24)
            self.b = nn.Linear(24, 20)
            self.c = nn.Linear(20, 8)

        def forward(self, x):
            return self.c(torch.nn.functional.gelu(self.b(torch.relu(self.a(x)))))

    torch.manual_seed(0)
    c = census(Chain3(), torch.randn(4, 16))
    check("3-deep chain (relu, gelu links) is ONE delegate, no portable acts",
          c["delegate"] == 1
          and not any("relu" in k or "gelu" in k for k in c), f"({dict(c)})")

    # A tail that is ALSO a partition output: in-partition readers use its output
    # arg index, so the value leaves the delegate exactly once and no scratch is
    # spent on it.
    class TailOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(16, 24), nn.Linear(24, 8)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            return h, self.fc2(h)

    torch.manual_seed(0)
    c = census(TailOut(), torch.randn(4, 16))
    check("chained tail that is also an output stays ONE delegate",
          c["delegate"] == 1
          and not any("relu" in k for k in c), f"({dict(c)})")

    # Fan-out from one internal intermediate: two readers of the same slot. Legal
    # -- the slot is not rewritten between them -- and the join add stays portable
    # (both operands are fusable tails, which _residual_pattern refuses).
    class FanOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 24)
            self.fc2, self.fc3 = nn.Linear(24, 8), nn.Linear(24, 8)

        def forward(self, x):
            t = torch.relu(self.fc1(x))
            return self.fc2(t) + self.fc3(t)

    torch.manual_seed(0)
    c = census(FanOut(), torch.randn(4, 16))
    check("scratch fan-out: one delegate, portable join add",
          c["delegate"] == 1
          and not any("relu" in k for k in c)
          and sum(v for k, v in c.items() if "add" in k) == 1, f"({dict(c)})")


@case
def test_qk_optin():
    """int8 q@kT is OPT-IN: default partitioning leaves both bmms portable;
    quantize_qk=True claims exactly the pre-softmax one (its transpose folds
    into the delegate, which then reads keys in their natural row layout), and
    p@v stays portable by the softmax-consumer gate."""
    print("\n-- opt-in int8 q@kT --")
    import contextlib
    import io

    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.models import AttnQK, ramp_input

    def census(qk):
        torch.manual_seed(0)
        ep = torch.export.export(AttnQK().eval(), (ramp_input((1, 8, 32)),))
        edge = to_edge_transform_and_lower(
            ep, partitioner=[Esp32Partitioner(quantize_qk=qk)])
        c = Counter()
        for n in edge.exported_program().graph_module.graph.nodes:
            if n.op == "call_function":
                t = getattr(n.target, "_name", None) or str(n.target)
                c["delegate" if "call_delegate" in str(n.target) else t] += 1
        return c

    c = census(False)
    check("default OFF: both bmms portable, k-transpose too",
          sum(v for k, v in c.items() if "bmm" in k) == 2
          and sum(v for k, v in c.items() if "permute" in k) >= 1,
          f"({dict(c)})")

    c = census(True)
    check("opt-in: q@kT claimed, p@v portable, transpose folded",
          sum(v for k, v in c.items() if "bmm" in k) == 1
          and sum(v for k, v in c.items() if "permute" in k) == 0,
          f"({dict(c)})")

    # q@kT reads both operands from partition inputs. An embedding's [B,M,K]
    # output goes straight into it with no reshape to break the partition, so
    # claiming both would leave q on a scratch slot the bmm cannot read.
    class EmbedIntoQK(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(16, 8)

        def forward(self, ids, k):
            return torch.softmax(torch.bmm(self.emb(ids), k.transpose(1, 2)), -1)

    try:
        ep = torch.export.export(EmbedIntoQK().eval(), (
            torch.randint(0, 16, (2, 4)), torch.randn(2, 5, 8)))
        with contextlib.redirect_stdout(io.StringIO()):
            edge = to_edge_transform_and_lower(
                ep, partitioner=[Esp32Partitioner(quantize_qk=True)])
            edge.to_executorch()
        gm = edge.exported_program().graph_module
        left = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        check("embedding into q@kT lowers, the gather delegated",
              any("call_delegate" in t for t in left)
              and not any("embedding" in t for t in left), f"{len(left)} nodes")
    except RuntimeError as e:
        check("embedding into q@kT lowers, the gather delegated", False,
              str(e)[:80])


def _lower_census(model, inputs, **partitioner_kw):
    """(portable-op census, [descriptor dicts]) for a model lowered the way
    the exporters do it. The descriptors are read back out of the delegate
    blobs, so a claim about a flag is a claim about the bytes."""
    import struct

    from executorch.backends.esp32s3.serialization import espressif_blob as eb
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import lower_esp32

    ep = torch.export.export(model.eval(), inputs)
    edge = lower_esp32(ep, [Esp32Partitioner(**partitioner_kw)])
    gm = edge.exported_program().graph_module
    c = Counter()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            t = getattr(n.target, "_name", None) or str(n.target)
            c["delegate" if "call_delegate" in str(n.target) else t] += 1
    descs = []
    for _, mod in gm.named_children():
        blob = getattr(mod, "processed_bytes", None)
        if blob is None:
            continue
        _magic, _version, n_ops, _ext = struct.unpack_from(
            eb.BLOB_HEADER_FORMAT, blob, 0)
        for i in range(n_ops):
            descs.append(dict(zip(eb.OP_DESC_FIELDS, struct.unpack_from(
                eb.OP_DESC_FORMAT, blob,
                eb.BLOB_HEADER_BYTES + i * eb.OP_DESC_BYTES))))
    return c, descs


def _portable(c, *names):
    return sum(v for k, v in c.items() if any(n in k for n in names))


@case
def test_gated_mul_into_delegate():
    """FLAG_MUL: SwiGLU/GeGLU gate*up and output gates never become a portable
    aten::mul.out. A pair multiplies in place -- the later projection reads
    and writes the destination the earlier one filled -- and a gate reads a
    partition input. Refusals keep the mul portable without losing the
    matmuls."""
    print("\n-- gated multiply into the delegate (FLAG_MUL) --")
    from executorch.backends.esp32s3.serialization import espressif_blob as eb

    F = torch.nn.functional

    class GLU(nn.Module):
        def __init__(self, act, down=True):
            super().__init__()
            self.g = nn.Linear(16, 32, bias=False)
            self.u = nn.Linear(16, 32, bias=False)
            self.d = nn.Linear(32, 16, bias=False) if down else None
            self.act = act

        def forward(self, x):
            y = self.act(self.g(x)) * self.u(x)
            return self.d(y) if self.d is not None else y

    def mul_descs(descs):
        return [d for d in descs if d["flags"] & eb.FLAG_MUL]

    for name, act, flag in (
            ("SwiGLU", F.silu, eb.FLAG_ACT_SILU),
            ("GeGLU", F.gelu, eb.FLAG_ACT_GELU),
            ("GeGLU-tanh", lambda t: F.gelu(t, approximate="tanh"),
             eb.FLAG_ACT_GELU_TANH),
            ("ReGLU", F.relu, eb.FLAG_CLAMP)):
        torch.manual_seed(0)
        c, descs = _lower_census(GLU(act), (torch.randn(1, 16),))
        muls = mul_descs(descs)
        gate = [d for d in descs if d["flags"] & flag]
        check(f"{name}: one 3-op delegate, no portable mul or activation",
              c.get("delegate") == 1 and len(descs) == 3
              and _portable(c, "mul", "sigmoid", "gelu", "relu") == 0,
              f"({c})")
        # The later projection carries the flag; its operand IS its output
        # (in place), the slot the earlier projection wrote and down reads.
        check(f"{name}: up multiplies in place what gate wrote",
              len(muls) == 1 and len(gate) == 1
              and muls[0]["res_arg"] == muls[0]["out_arg"]
              == gate[0]["out_arg"] >= eb.ARG_SCRATCH0
              and descs.index(gate[0]) < descs.index(muls[0])
              and descs[2]["in_arg"] == muls[0]["out_arg"],
              f"({[(hex(d['flags']), d['in_arg'], d['out_arg'], d['res_arg']) for d in descs]})")

    # The product as a partition OUTPUT: in place on a boundary arg.
    torch.manual_seed(0)
    c, descs = _lower_census(GLU(F.silu, down=False), (torch.randn(1, 16),))
    muls = mul_descs(descs)
    check("SwiGLU product as an output: in place on the output arg",
          c.get("delegate") == 1 and _portable(c, "mul") == 0
          and len(muls) == 1 and muls[0]["res_arg"] == muls[0]["out_arg"]
          < eb.ARG_SCRATCH0, f"({c})")

    # Glimmer's attention output gate and its PLE gate: the other factor is a
    # runtime partition input, read by index.
    class Gates(nn.Module):
        def __init__(self):
            super().__init__()
            self.gp = nn.Linear(16, 32, bias=False)
            self.o = nn.Linear(32, 16, bias=False)
            self.pg = nn.Linear(16, 24, bias=False)
            self.pp = nn.Linear(24, 16, bias=False)

        def forward(self, x, out, ple):
            a = self.o(out * torch.sigmoid(self.gp(x)))
            return a + self.pp(F.gelu(self.pg(a)) * ple)

    torch.manual_seed(0)
    c, descs = _lower_census(
        Gates(), (torch.randn(1, 16), torch.randn(1, 32), torch.randn(1, 24)))
    muls = mul_descs(descs)
    check("output gates: no portable sigmoid or mul",
          _portable(c, "sigmoid", "mul", "gelu") == 0
          and len(muls) == 2
          and all(d["res_arg"] < eb.ARG_SCRATCH0 and d["res_arg"] != d["out_arg"]
                  for d in muls), f"({c})")

    # Refusals. Each keeps the multiply portable and the matmuls delegated.
    class Bcast(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Linear(16, 32, bias=False)

        def forward(self, x, s):
            return torch.sigmoid(self.g(x)) * s       # s is [1, 1]: broadcast

    c, descs = _lower_census(Bcast(), (torch.randn(1, 16), torch.randn(1, 1)))
    check("broadcast factor: mul stays portable, sigmoid still fuses",
          _portable(c, "mul") == 1 and _portable(c, "sigmoid") == 0
          and not mul_descs(descs), f"({c})")

    class ConstGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Linear(16, 32, bias=False)
            self.scale = nn.Parameter(torch.randn(1, 32))

        def forward(self, x):
            return self.g(x) * self.scale            # learned, lifted constant

    c, descs = _lower_census(ConstGate(), (torch.randn(1, 16),))
    check("constant factor (layer scale): mul stays portable",
          _portable(c, "mul") == 1 and c.get("delegate") == 1
          and not mul_descs(descs), f"({c})")

    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Linear(16, 32, bias=False)
            self.u = nn.Linear(16, 32, bias=False)

        def forward(self, x):
            u = self.u(x)
            return torch.sigmoid(self.g(x)) * u + u   # up also used raw

    c, descs = _lower_census(Reused(), (torch.randn(1, 16),))
    check("factor with a second user: mul stays portable",
          _portable(c, "mul") == 1 and not mul_descs(descs), f"({c})")

    torch.manual_seed(0)
    c, descs = _lower_census(GLU(F.silu), (torch.randn(1, 16),),
                             format_policy="fp32")
    check("fp32 policy: no fused multiply (its loop accumulates in place)",
          _portable(c, "mul") == 1 and not mul_descs(descs), f"({c})")

    # The tail walk mirrors the gates: an activation under a clamp is not
    # claimed, so a residual on top of it must not be either (otherwise this
    # spelling reaches preprocess with no tail to ride and fails the export).
    class ActUnderClamp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            return x + F.gelu(torch.relu(self.fc(x)))

    c, _ = _lower_census(ActUnderClamp(), (torch.randn(1, 16),))
    check("residual over act-over-clamp: lowers, add and gelu portable",
          _portable(c, "add") == 1 and _portable(c, "gelu") == 1,
          f"({c})")


@case
def test_acts_into_delegate():
    """The one-op activations: each vanishes from the remainder into
    its matmul's descriptor, in every spelling torch produces; non-standard
    constants and act-over-clamp chains stay portable."""
    print("\n-- one-op activations into the delegate --")
    from executorch.backends.esp32s3.serialization import espressif_blob as eb

    F = torch.nn.functional

    class Lin(nn.Module):
        def __init__(self, act):
            super().__init__()
            self.fc = nn.Linear(16, 32)
            self.act = act

        def forward(self, x):
            return self.act(self.fc(x))

    x2, x3 = torch.randn(2, 16), torch.randn(1, 4, 16)
    fused = (
        ("sigmoid", torch.sigmoid, eb.FLAG_ACT_SIGMOID, x2),
        ("tanh-gelu", lambda t: F.gelu(t, approximate="tanh"),
         eb.FLAG_ACT_GELU_TANH, x2),
        ("hardswish", F.hardswish, eb.FLAG_ACT_HARDSWISH, x2),
        ("hardswish 3-D", F.hardswish, eb.FLAG_ACT_HARDSWISH, x3),
        ("hardswish, x*relu6(x+3)/6", lambda t: t * F.relu6(t + 3) / 6,
         eb.FLAG_ACT_HARDSWISH, x2),
        ("hardsigmoid", F.hardsigmoid, eb.FLAG_ACT_HARDSIGMOID, x2),
        ("leaky relu", lambda t: F.leaky_relu(t, 0.2), eb.FLAG_ACT_LEAKY, x2),
        ("relu^2 as pow", lambda t: F.relu(t) ** 2, eb.FLAG_ACT_RELU2, x2),
        ("relu^2 as square", lambda t: torch.square(F.relu(t)),
         eb.FLAG_ACT_RELU2, x2),
        ("relu^2 as r*r", lambda t: (lambda r: r * r)(F.relu(t)),
         eb.FLAG_ACT_RELU2, x2),
    )
    for name, act, flag, x in fused:
        torch.manual_seed(0)
        c, descs = _lower_census(Lin(act), (x,))
        rest = {k: v for k, v in c.items()
                if k not in ("delegate", "aten::view_copy")
                and "getitem" not in k}
        check(f"{name}: fused, nothing else portable",
              c.get("delegate") == 1 and not rest and len(descs) == 1
              and descs[0]["flags"] & flag
              and not descs[0]["flags"] & eb.FLAG_CLAMP, f"({c})")
    torch.manual_seed(0)
    _, descs = _lower_census(Lin(lambda t: F.leaky_relu(t, 0.2)), (x2,))
    check("leaky relu: slope rides clamp_lo",
          abs(descs[0]["clamp_lo"] - 0.2) < 1e-7)

    # The add/divide constants ride the descriptor (partitioning sees only
    # FakeTensors), so a non-standard pair is exact rather than refused late.
    torch.manual_seed(0)
    _, descs = _lower_census(Lin(lambda t: t * F.relu6(t + 2) / 5), (x2,))
    check("hardswish constants ride clamp_lo/clamp_hi",
          len(descs) == 1 and descs[0]["flags"] & eb.FLAG_ACT_HARDSWISH
          and (descs[0]["clamp_lo"], descs[0]["clamp_hi"]) == (2.0, 5.0),
          f"({descs})")
    torch.manual_seed(0)
    _, descs = _lower_census(Lin(F.hardswish), (x2,))
    check("torch's hardswish: (3, 6)",
          (descs[0]["clamp_lo"], descs[0]["clamp_hi"]) == (3.0, 6.0))

    refused = (
        ("clamp to [0,5] is not hardswish",
         lambda t: t * torch.clamp(t + 3, 0, 5) / 6, ("mul", "div")),
        ("square of relu6 is not relu^2", lambda t: F.relu6(t) ** 2, ("pow",)),
        ("act over a clamp stays portable",
         lambda t: torch.sigmoid(torch.relu(t)), ("sigmoid",)),
    )
    for name, act, left in refused:
        torch.manual_seed(0)
        c, _ = _lower_census(Lin(act), (x2,))
        check(f"{name}: matmul delegated, {'/'.join(left)} portable",
              c.get("delegate") == 1
              and all(_portable(c, n) >= 1 for n in left), f"({c})")


if __name__ == "__main__":
    sys.exit(main())
