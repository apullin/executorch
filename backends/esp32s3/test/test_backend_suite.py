# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Out-of-box robustness: a wide span of patterns, lowered through the shared harness.

The bar ExecuTorch's own backend suite sets is that a backend "is able to
handle a given pattern without erroring out (not partitioning is fine)" and
yields reasonable outputs -- robustness across inputs nobody wrote a fixture
for. This file asks that question systematically, for patterns the backend's
own rewrites look for; upstream's operator suite itself runs through
test/suite/ (its README).

Run through `Esp32Tester`, this repo's subclass of `backends/test/harness`'s
Tester -- the same base XNNPACK, Arm, CoreML and Vulkan subclass -- so the shape
is what an upstream reviewer expects to see. Each case is lowered the README's
way, run by esp32_host_runner (the delegate's runtime and operator library
built for the build machine), and its outputs compared with eager at the
upstream suite's tolerance: a delegated float model runs with int8 weights,
so the numbers are close, not equal. The device's own bits are proven
elsewhere, by QEMU and silicon parity (`run_parity.py`).

A pattern that lowers with ZERO delegated ops is a PASS, deliberately. Refusing
to claim an op is a supported outcome; crashing while deciding is not, and
neither is a wrong number.

    python test_backend_suite.py
"""

import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

FAILS = []


class Seq(nn.Module):
    def __init__(self, *mods):
        super().__init__()
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class Attn(nn.Module):
    """4-D multi-head attention: the spelling QkTo3D rewrites."""

    def __init__(self, d=32, h=4):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))

    def forward(self, x):
        B, N, D = x.shape
        s = lambda t: t.view(B, N, self.h, self.dh).transpose(1, 2)
        q, k, v = s(self.q(x)), s(self.k(x)), s(self.v(x))
        a = torch.softmax(q @ k.transpose(-1, -2) / self.dh ** 0.5, -1)
        return (a @ v).transpose(1, 2).reshape(B, N, D)


class Residual(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return x + self.fc(x)


class Reduce(nn.Module):
    def forward(self, x):
        return x.mean(-1) + x.sum(-1) - x.amax(-1)


class Broadcast(nn.Module):
    def forward(self, a, b):
        return a * b + a.mean(-1, keepdim=True)


class Chunky(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc = nn.Linear(d, d * 3)

    def forward(self, x):
        a, b, c = self.fc(x).chunk(3, -1)
        return torch.cat([a * b, c], -1)


class EmbedLM(nn.Module):
    def __init__(self, v=128, d=32):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.fc = nn.Linear(d, v)

    def forward(self, ids):
        return self.fc(self.emb(ids))


class ConvStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, 2, 1)
        self.bn = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 1)

    def forward(self, x):
        return self.c2(torch.nn.functional.silu(self.bn(self.c1(x))))


class RuntimeWeightMM(nn.Module):
    def forward(self, activation, runtime_weight):
        return torch.mm(activation, runtime_weight)


class ScaledAddmmRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 24))
        self.bias = nn.Parameter(torch.randn(24))

    def forward(self, x):
        return torch.relu(torch.addmm(
            self.bias, x, self.weight, beta=2, alpha=3))


class LearnedOffset(nn.Module):
    """Qwen DeltaNet's projected decay plus a same-shaped lifted parameter."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32, bias=False)
        self.offset = nn.Parameter(torch.ones(32))

    def forward(self, x):
        return F.softplus(self.fc(x).reshape(-1) + self.offset)


class RuntimeBiasAddmm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 24))

    def forward(self, bias, x):
        return torch.addmm(bias, x, self.weight)


class RuntimeConvWeight(nn.Module):
    def forward(self, x, weight):
        return F.conv2d(x, weight, padding=1)


class RuntimeEmbeddingTable(nn.Module):
    def forward(self, table, indices):
        return F.embedding(indices, table)


def pass_through_table(rows=64, cols=16):
    """A torchao-quantised table, consumed verbatim as a pass-through weight."""
    import numpy as np

    from executorch.backends.esp32s3.utils.pass_through import PTEmbedding, QTensor

    rng = np.random.default_rng(0)
    codes = rng.integers(-7, 8, size=(rows, cols), dtype=np.int8)
    scales = np.full((rows, 1), 0.02, dtype=np.float16)
    return PTEmbedding(QTensor(codes, scales, cols))


# A gather has no accumulator: an activation after an embedding must stay
# portable while the embedding itself still delegates.
EMBED_IDS = torch.randint(0, 64, (1, 6))


class ResidualOverActClamp(nn.Module):
    """x + clamp(act(fc(x))): the runtime's own epilogue order."""

    def __init__(self, clamp_first=False):
        super().__init__()
        self.fc = nn.Linear(32, 32)
        self.clamp_first = clamp_first

    def forward(self, x):
        if self.clamp_first:        # act(clamp(...)): not representable
            return x + F.gelu(F.relu6(self.fc(x)))
        return x + F.relu6(F.gelu(self.fc(x)))


class GroupedIntoConv(nn.Module):
    """A grouped conv whose output feeds another delegated conv."""

    def __init__(self, act=None):
        super().__init__()
        self.grouped = nn.Conv2d(4, 4, 1, groups=2)
        self.dense = nn.Conv2d(4, 4, 1)
        self.act = act or nn.Identity()

    def forward(self, x):
        return self.dense(self.act(self.grouped(x)))


class EmbedIntoQK(nn.Module):
    """An embedding's [B, M, K] output straight into the int8 q@kT."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 8)

    def forward(self, ids, k):
        return torch.softmax(torch.bmm(self.emb(ids), k.transpose(1, 2)), -1)


class SelectedReluLinear(nn.Module):
    """A selected-row head whose relu'd output feeds another Linear."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(64, 16))
        self.fc = nn.Linear(4, 8)

    def forward(self, x, rows):
        logits = F.linear(x, torch.index_select(self.table, 0, rows))
        return self.fc(torch.relu(logits))


class ReluViewRelu(nn.Module):
    """relu(view(relu(fc(x)))): a reshape between two clamp-family links."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return torch.relu(torch.relu(self.fc(x)).reshape(4, 4, 8))


CASES = [
    # name,                    module,                              inputs
    ("linear",                 Seq(nn.Linear(32, 64)),              (torch.randn(4, 32),)),
    ("linear+relu+linear",     Seq(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32)),
                                                                    (torch.randn(4, 32),)),
    ("linear 3-D input",       Seq(nn.Linear(32, 64)),              (torch.randn(1, 6, 32),)),
    ("linear no bias",         Seq(nn.Linear(32, 64, bias=False)),  (torch.randn(4, 32),)),
    ("gelu exact",             Seq(nn.Linear(32, 32), nn.GELU()),   (torch.randn(4, 32),)),
    ("gelu tanh",              Seq(nn.Linear(32, 32), nn.GELU(approximate="tanh")),
                                                                    (torch.randn(4, 32),)),
    ("silu",                   Seq(nn.Linear(32, 32), nn.SiLU()),   (torch.randn(4, 32),)),
    ("relu6",                  Seq(nn.Linear(32, 32), nn.ReLU6()),  (torch.randn(4, 32),)),
    ("hardtanh",               Seq(nn.Linear(32, 32), nn.Hardtanh(-2, 3)),
                                                                    (torch.randn(4, 32),)),
    ("layernorm",              Seq(nn.Linear(32, 32), nn.LayerNorm(32)),
                                                                    (torch.randn(4, 32),)),
    ("residual add",           Residual(),                          (torch.randn(4, 32),)),
    ("residual over clamp(gelu)", ResidualOverActClamp(),           (torch.randn(4, 32),)),
    ("residual over gelu(clamp)", ResidualOverActClamp(clamp_first=True),
                                                                    (torch.randn(4, 32),)),
    ("relu(view(relu(linear)))", ReluViewRelu(),                    (torch.randn(4, 32),)),
    ("attention 4-D",          Attn(),                              (torch.randn(1, 8, 32),)),
    ("chunk + cat",            Chunky(),                            (torch.randn(1, 6, 32),)),
    ("embedding + tied-ish head", EmbedLM(),                        (torch.randint(0, 128, (1, 6)),)),
    ("embedding + relu",       Seq(nn.Embedding(64, 16), nn.ReLU()), (EMBED_IDS,)),
    ("embedding + gelu",       Seq(nn.Embedding(64, 16), nn.GELU()), (EMBED_IDS,)),
    ("embedding + silu",       Seq(nn.Embedding(64, 16), nn.SiLU()), (EMBED_IDS,)),
    ("pass-through embedding + relu", Seq(pass_through_table(), nn.ReLU()),
                                                                    (EMBED_IDS,)),
    ("pass-through embedding + silu", Seq(pass_through_table(), nn.SiLU()),
                                                                    (EMBED_IDS,)),
    ("reductions",             Reduce(),                            (torch.randn(2, 5, 16),)),
    ("broadcast elementwise",  Broadcast(),                         (torch.randn(2, 5, 16), torch.randn(2, 5, 16))),
    ("conv2d + bn + silu",     ConvStack(),                         (torch.randn(1, 3, 16, 16),)),
    ("conv1d",                 Seq(nn.Conv1d(4, 8, 3, padding=1)),  (torch.randn(1, 4, 20),)),
    ("selected head -> relu -> linear", SelectedReluLinear(),
                                (torch.randn(2, 16), torch.tensor([5, 1, 60, 7]))),
    ("grouped conv -> conv",   GroupedIntoConv(),                   (torch.randn(1, 4, 2, 2),)),
    ("grouped conv -> relu -> conv", GroupedIntoConv(nn.ReLU()),    (torch.randn(1, 4, 2, 2),)),
    ("maxpool + linear",       Seq(nn.MaxPool2d(2), nn.Flatten(), nn.Linear(3 * 8 * 8, 16)),
                                                                    (torch.randn(1, 3, 16, 16),)),
    ("lstm",                   nn.LSTM(16, 16, batch_first=True),   (torch.randn(1, 5, 16),)),
    ("gru",                    nn.GRU(16, 16, batch_first=True),    (torch.randn(1, 5, 16),)),
    ("scalar mul chain",       Seq(nn.Linear(32, 32)),              (torch.randn(4, 32),)),
    # These are intentionally portable.  They exercise the partitioner's
    # contract: refusing an ineligible core must also refuse its fused/folded
    # neighbours, and lowering must still complete.
    ("runtime-weight mm",      RuntimeWeightMM(),
                                (torch.randn(4, 32), torch.randn(32, 24))),
    ("runtime-bias addmm",     RuntimeBiasAddmm(),
                                (torch.randn(24), torch.randn(4, 32))),
    ("runtime conv weight",    RuntimeConvWeight(),
                                (torch.randn(1, 3, 8, 8),
                                 torch.randn(5, 3, 3, 3))),
    ("runtime embedding table", RuntimeEmbeddingTable(),
                                 (torch.randn(64, 16),
                                  torch.randint(0, 64, (2, 5)))),
    ("int32 embedding indices", nn.Embedding(64, 16),
                                 (torch.randint(0, 64, (2, 5),
                                                dtype=torch.int32),)),
    ("scaled addmm + relu",    ScaledAddmmRelu(),
                                (torch.randn(4, 32),)),
    ("linear + learned offset", LearnedOffset(),
                                (torch.randn(1, 32),)),
    ("fp16 linear",            Seq(nn.Linear(32, 24).half()),
                                (torch.randn(4, 32).half(),)),
]

EXPECTED_PORTABLE = {
    "runtime-weight mm", "runtime-bias addmm", "runtime conv weight",
    "runtime embedding table", "int32 embedding indices",
    "scaled addmm + relu", "fp16 linear",
}

# backends/test/suite/runner.py's comparison.
ATOL, RTOL = 1e-1, 4e-2


def host_error(tester, module, inputs):
    """The lowered program on the host runner against eager `module`: the
    largest absolute difference over every output, flattened as the program
    returns them. Raises if a shape or dtype differs or an output is outside
    the suite's tolerance."""
    from executorch.backends.test.harness.stages import StageType
    from torch.utils._pytree import tree_flatten

    got = tester.stages[StageType.SERIALIZE].run_artifact(inputs)
    with torch.no_grad():
        want = tree_flatten(module(*inputs))[0]
    if len(got) != len(want):
        raise AssertionError(f"{len(got)} outputs for eager's {len(want)}")
    worst = 0.0
    for i, (g, w) in enumerate(zip(got, want)):
        if g.shape != w.shape or g.dtype != w.dtype:
            raise AssertionError(f"output {i}: {g.dtype}{list(g.shape)} for "
                                 f"eager's {w.dtype}{list(w.shape)}")
        if w.numel() == 0:
            continue
        diff = (g.double() - w.double()).abs()
        worst = max(worst, float(diff.max()))
        if not torch.allclose(g.double(), w.double(), atol=ATOL, rtol=RTOL):
            raise AssertionError(f"output {i} off by up to {float(diff.max()):.3g}")
    return worst


def main():
    from executorch.backends.test.harness.stages import StageType

    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    print(f"{'pattern':30s} {'runs':>7s} {'delegate':>9s}  notes")
    print("-" * 66)
    lowered = 0
    for name, module, inputs in CASES:
        torch.manual_seed(0)
        note = ""
        try:
            t = Esp32Tester(module, inputs)
            t.export().to_edge_transform_and_lower().to_executorch().serialize()
            gm = t.get_artifact(
                StageType.TO_EDGE_TRANSFORM_AND_LOWER
            ).exported_program().graph_module
            calls = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                        and "executorch_call_delegate" in str(n.target))
            if name in EXPECTED_PORTABLE and calls != 0:
                raise AssertionError(
                    f"ineligible pattern produced {calls} delegate call(s)")
            size = len(t.get_artifact(StageType.SERIALIZE))
            err = host_error(t, module, inputs)
            ok, lowered = True, lowered + 1
            note = f"{size:,} B .pte, max err {err:.2g}"
        except Exception as e:  # noqa: BLE001 -- the point is to catalogue these
            ok, calls = False, 0
            note = f"{type(e).__name__}: {str(e).splitlines()[0][:44]}"
            FAILS.append(name)
        print(f"{name:30s} {'yes' if ok else 'NO':>7s} {calls:>9d}  {note}")

    print("-" * 66)
    print(f"{lowered}/{len(CASES)} patterns lowered, ran and matched eager "
          f"(atol {ATOL}, rtol {RTOL})")

    # The quantized flow, through the harness's own Quantize stage, both ways
    # the quantizer annotates: dynamic int8 activations (its default) and
    # weight-only. The converted model is exported non-strictly, as the
    # README exports it: strict export cannot re-trace the dynamic chooser's
    # tensor argument (espressif_tester.Export). Its reference is the converted
    # model, as the harness's always is.
    print()
    print("quantized flow (harness Quantize stage):")
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    from executorch.backends.esp32s3.test.tester.espressif_tester import (
        Quantize as Esp32Quantize,
    )

    for label, quantizer in (
        ("dynamic act (default)", Esp32Quantizer(weight_bits=4)),
        ("weight-only", Esp32Quantizer(weight_bits=4, annotate_act=False)),
    ):
        torch.manual_seed(0)
        mod = Seq(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        x = (torch.randn(4, 32),)
        try:
            t = Esp32Tester(mod, x, strict_export=False)
            t.quantize(Esp32Quantize(quantizer=quantizer)).export()
            t.to_edge_transform_and_lower().to_executorch().serialize()
            gm = t.get_artifact(
                StageType.TO_EDGE_TRANSFORM_AND_LOWER
            ).exported_program().graph_module
            calls = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                        and "executorch_call_delegate" in str(n.target))
            err = host_error(t, t.get_artifact(StageType.EXPORT).module(), x)
            print(f"  {label:24s} runs, {calls} delegate call(s), max err {err:.2g}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24s} does NOT run: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:48]}")
            FAILS.append(f"quantized {label} flow")

    # Zero delegated ops is a pass. Erroring while deciding is not.
    if FAILS:
        print(f"\nFAILED: {', '.join(FAILS)}")
        return 1
    print("every pattern lowers and runs within tolerance; delegation counts are "
          "information, not the bar")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() records every failed
    check and goes on; this fails if any did, naming them."""
    start = len(FAILS)
    rc = main()
    failed = FAILS[start:]
    assert not failed and rc in (None, 0), f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(main())
