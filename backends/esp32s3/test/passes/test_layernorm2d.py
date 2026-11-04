# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LayerNorm2d fuses to one op, exactly, and only where it should.

The op replaces FOURTEEN portable ops, each reading and writing a full
[B,C,H,W] temporary. What makes that safe is that the fused op is not an
approximation: it repeats the source's operation order, including dividing by
`sqrt(s + eps)` rather than multiplying by an `rsqrt`, so the fused and unfused
programs agree bit for bit and the device kernel has an unambiguous reference.

So the checks are exactness and SELECTIVITY. A fusion matcher that is merely
eager is worse than none: every shape it accepts wrongly becomes a silently
different operator, and the graphs here (mean, sub, pow, mean, add, sqrt, div,
mul, add) are common enough to appear by accident.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* fails
naming its failed checks (@case).

    python test_layernorm2d.py
"""

import functools
import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def case(test):
    """One test_* case. check() records a failure and goes on, so a script
    run reports every one (main); a case that only did that would pass
    under pytest whatever its checks found. Called, it fails if any check
    it made failed, naming them. main() calls the case itself
    (__wrapped__), so the script still runs every case."""

    @functools.wraps(test)
    def run():
        start = len(FAILS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


class LN2d(nn.Module):
    """The SAM/ConvNeXt spelling."""

    def __init__(self, c=16, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c))
        self.bias = nn.Parameter(torch.randn(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LN2dTypedEps(nn.Module):
    """eps held in the model's own dtype. A Python-float eps reaches an fp16
    graph through a cast the matcher does not walk; a typed one does not."""

    def __init__(self, c=16, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c))
        self.bias = nn.Parameter(torch.randn(c))
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LastDimNorm(nn.Module):
    """Same arithmetic over the LAST axis: a different operator, must NOT fuse."""

    def __init__(self, c=16, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c))
        self.bias = nn.Parameter(torch.randn(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class NoAffine(nn.Module):
    """Normalise over C with no weight/bias. Nothing to match on; stays portable."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(s + self.eps)


# Look-alikes: the LayerNorm2d graph shape, computing something else.
class AffineOverWidth(LN2d):
    """`w.unsqueeze(0).unsqueeze(0)` is [1, 1, C]: it scales WIDTH."""

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return (self.weight.unsqueeze(0).unsqueeze(0) * x
                + self.bias.unsqueeze(0).unsqueeze(0))


class ScaledBias(LN2d):
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return torch.add(self.weight[:, None, None] * x,
                         self.bias[:, None, None], alpha=2)


class ScaledCentre(LN2d):
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        d = torch.sub(x, u, alpha=2)
        x = d / torch.sqrt(d.pow(2).mean(1, keepdim=True) + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ScaledEps(LN2d):
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(torch.add(s, self.eps, alpha=2))
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class WideMean(LN2d):
    """Both means accumulated in fp64: different rounding, an fp64 result."""

    def forward(self, x):
        u = x.mean(1, keepdim=True, dtype=torch.float64)
        s = (x - u).pow(2).mean(1, keepdim=True, dtype=torch.float64)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MutableEps(nn.Module):
    """eps is a buffer the forward updates; a fused literal would freeze it."""

    def __init__(self, c=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c))
        self.bias = nn.Parameter(torch.randn(c))
        self.register_buffer("eps", torch.tensor(0.01))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        y = (x - u) / torch.sqrt(s + self.eps)
        self.eps.copy_(self.eps * 100.0)
        return self.weight[:, None, None] * y + self.bias[:, None, None]


def n_calls(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function")


def n_ln2d(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function"
               and "layer_norm2d" in str(n.target))


def _sample():
    """The SAM/ConvNeXt module at seed 0 and its input."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 9, 9)
    return LN2d().eval(), x


@case
def test_same_arithmetic():
    from executorch.backends.esp32s3._passes.fuse_layer_norm2d_pass import _ln2d

    m, x = _sample()
    with torch.no_grad():
        ref = m(x)
    print("the replacement op is the same arithmetic, not an approximation:")
    got = _ln2d(x, m.weight, m.bias, m.eps)
    # torch.equal, not allclose. There is no rounding to tolerate: the op repeats
    # the source's expression tree, so anything but an exact match is a bug.
    check("bit-identical to eager", torch.equal(got, ref),
          f"max abs diff {(got - ref).abs().max().item():.3e}")


@case
def test_fuses_whole_chain():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.fuse_layer_norm2d_pass import (
        fuse_layer_norm2d,
    )

    m, x = _sample()
    print("it fuses, and collapses the whole chain:")
    edge = to_edge(torch.export.export(m, (x,)))
    before = n_calls(edge.exported_program().graph_module)
    p = fuse_layer_norm2d(edge)
    gm = edge.exported_program().graph_module
    check("one site fused", p.fused == 1, f"{p.fused}")
    check("14 ops -> 1", (before, n_calls(gm)) == (14, 1),
          f"{before} -> {n_calls(gm)}")
    check("and it is the custom op", n_ln2d(gm) == 1)


@case
def test_selective():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.fuse_layer_norm2d_pass import (
        fuse_layer_norm2d,
    )

    print("selectivity -- these must NOT fuse:")
    for name, mod, shape, dtype in (
        ("normalising over the LAST axis", LastDimNorm(), (1, 4, 8, 16),
         torch.float32),
        ("no affine weight/bias", NoAffine(), (1, 16, 9, 9), torch.float32),
        # 3-D: the kernel indexes NCHW and would read a plane that is not there.
        ("a 3-D input", LN2d(), (16, 9, 9), torch.float32),
        # fp16: the kernel reads float pointers, so it would read half storage
        # as fp32 and write the output past its end. Typed eps, because that
        # is the spelling that otherwise matches.
        ("an fp16 graph", LN2dTypedEps().half(), (1, 16, 9, 9),
         torch.float16),
    ):
        try:
            e = to_edge(torch.export.export(
                mod.eval(), (torch.randn(*shape, dtype=dtype),)))
            q = fuse_layer_norm2d(e)
            check(f"{name}", q.fused == 0, f"fused {q.fused}")
        except Exception as exc:  # noqa: BLE001 -- refusing to export is also fine
            check(f"{name}", True, f"did not export ({type(exc).__name__})")
    # channels-last: the kernel reads and writes contiguous NCHW, and would
    # refuse the fused op at execute (a Cortex-M image through a reflect pad).
    x_cl = torch.randn(1, 16, 9, 9).contiguous(memory_format=torch.channels_last)
    e = to_edge(torch.export.export(LN2d().eval(), (x_cl,)))
    q = fuse_layer_norm2d(e)
    check("a channels-last input", q.fused == 0, f"fused {q.fused}")


@case
def test_lookalikes_left_alone():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.fuse_layer_norm2d_pass import (
        fuse_layer_norm2d,
    )

    _, x = _sample()
    print("look-alikes that compute something else are left alone:")
    for name, mod, inp in (
        # W == C, so [1, 1, C] broadcasts -- over width.
        ("affine unsqueezed to [1, 1, C]", AffineOverWidth(c=2),
         torch.randn(1, 2, 2, 2)),
        ("bias added with alpha=2", ScaledBias(), x),
        ("mean subtracted with alpha=2", ScaledCentre(), x),
        ("eps added with alpha=2", ScaledEps(), x),
        ("means accumulated in fp64", WideMean(), x),
    ):
        e = to_edge(torch.export.export(mod.eval(), (inp,)))
        with torch.no_grad():
            want = e.exported_program().module()(inp)
        q = fuse_layer_norm2d(e)
        with torch.no_grad():
            got = e.exported_program().module()(inp)
        same = got.dtype == want.dtype and torch.equal(got, want)
        check(f"{name}: not fused, output unchanged", q.fused == 0 and same,
              f"fused {q.fused}, output same: {same}")
    e = to_edge(torch.export.export(MutableEps().eval(), (x,)))
    q = fuse_layer_norm2d(e)
    check("an eps buffer the program mutates is not baked in", q.fused == 0,
          f"fused {q.fused}")


def main():
    for test in (test_same_arithmetic, test_fuses_whole_chain, test_selective,
                 test_lookalikes_left_alone):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("LayerNorm2d fuses exactly, and only where it means the same thing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
