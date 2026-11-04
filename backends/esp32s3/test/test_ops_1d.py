# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every 1-D op someone might build a network from, lowered. No landmines.

1-D CONVOLUTION is first class (test_conv_support.py). That is not the same
claim as "1-D is safe": an audio or timeseries model is conv1d plus pooling plus
normalisation plus resampling plus the elementwise glue, and a single op that
crashes the lowering makes the rest irrelevant.

So this is a robustness sweep, on the shared harness's bar: a pattern must
export, lower, and serialize WITHOUT ERRORING. Delegating nothing is a pass --
plenty of these have no matmul in them and correctly get portable kernels. The
failure this exists to catch is the one that stops a user dead.

Reported per op rather than asserted in aggregate, because "23 of 25 lower" is
useless without knowing which two.

    python test_ops_1d.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

FAILS = []


class Fn(nn.Module):
    """Wraps a plain function so the battery can hold both."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *a):
        return self.fn(*a)


C, L = 8, 32
X = (1, C, L)


def cases():
    return [
        # --- convolution family
        ("conv1d", nn.Conv1d(C, 16, 3, padding=1), X),
        ("conv1d depthwise", nn.Conv1d(C, C, 3, padding=1, groups=C), X),
        ("conv1d dilated", nn.Conv1d(C, 16, 3, dilation=4), X),
        ("conv1d stride", nn.Conv1d(C, 16, 5, stride=3, padding=2), X),
        ("conv_transpose1d", nn.ConvTranspose1d(C, 16, 4, stride=2, padding=1), X),
        # --- pooling. min-pool is not a torch op; it is -maxpool(-x), and that
        # spelling is what a model author actually writes, so it is what is
        # tested.
        ("max_pool1d", nn.MaxPool1d(2), X),
        ("max_pool1d stride1", nn.MaxPool1d(3, stride=1, padding=1), X),
        ("avg_pool1d", nn.AvgPool1d(2), X),
        ("min-pool via -max(-x)", Fn(lambda x: -F.max_pool1d(-x, 2)), X),
        ("adaptive_avg_pool1d", nn.AdaptiveAvgPool1d(8), X),
        ("adaptive_max_pool1d", nn.AdaptiveMaxPool1d(8), X),
        ("global avg (mean over L)", Fn(lambda x: x.mean(-1)), X),
        # --- normalisation
        ("batch_norm1d", nn.BatchNorm1d(C).eval(), X),
        ("instance_norm1d", nn.InstanceNorm1d(C), X),
        ("group_norm", nn.GroupNorm(2, C), X),
        ("layer_norm over L", nn.LayerNorm(L), X),
        ("rms-ish norm over C", Fn(
            lambda x: x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-5)), X),
        # --- resampling and shape
        ("upsample nearest", Fn(lambda x: F.interpolate(x, scale_factor=2.0,
                                                        mode="nearest")), X),
        ("upsample linear", Fn(lambda x: F.interpolate(x, scale_factor=2.0,
                                                       mode="linear",
                                                       align_corners=False)), X),
        ("pad (causal left)", Fn(lambda x: F.pad(x, (4, 0))), X),
        ("pad reflect", Fn(lambda x: F.pad(x, (2, 2), mode="reflect")), X),
        ("slice / stride", Fn(lambda x: x[..., ::2]), X),
        ("cat along L", Fn(lambda x: torch.cat([x, x], -1)), X),
        ("chunk + stack", Fn(lambda x: torch.stack(x.chunk(2, -1), 0).sum(0)), X),
        ("transpose C<->L", Fn(lambda x: x.transpose(1, 2).contiguous()), X),
        ("flatten + linear", nn.Sequential(nn.Flatten(), nn.Linear(C * L, 16)), X),
        # --- pointwise / activations a 1-D net actually uses
        ("silu", nn.SiLU(), X),
        ("gelu", nn.GELU(), X),
        ("hardswish", nn.Hardswish(), X),
        ("softmax over L", Fn(lambda x: x.softmax(-1)), X),
        ("clamp chain", Fn(lambda x: x.clamp(-3, 3).relu()), X),
        # --- recurrent, which is what a lot of timeseries code still is
        ("gru", nn.GRU(C, 16, batch_first=True), (1, L, C)),
        ("lstm", nn.LSTM(C, 16, batch_first=True), (1, L, C)),
        # --- attention over a 1-D sequence
        ("sdpa-ish attention", Fn(
            lambda q: torch.softmax(q @ q.transpose(-1, -2) / 4.0, -1) @ q),
         (1, L, C)),
    ]


def main():
    from executorch.backends.test.harness.stages import StageType

    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    print(f"{'op':28s} {'lowers':>7s} {'deleg':>6s}  notes")
    print("-" * 64)
    ok_n = 0
    for name, mod, shape in cases():
        torch.manual_seed(0)
        x = torch.randn(*shape)
        try:
            t = Esp32Tester(mod.eval(), (x,))
            t.export().to_edge_transform_and_lower().to_executorch().serialize()
            gm = t.get_artifact(
                StageType.TO_EDGE_TRANSFORM_AND_LOWER
            ).exported_program().graph_module
            calls = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                        and "executorch_call_delegate" in str(n.target))
            note = f"{len(t.get_artifact(StageType.SERIALIZE)):,} B"
            ok_n += 1
            good = True
        except Exception as e:  # noqa: BLE001 -- cataloguing these is the point
            good, calls = False, 0
            note = f"{type(e).__name__}: {str(e).splitlines()[0][:38]}"
            FAILS.append(name)
        print(f"{name:28s} {'yes' if good else 'NO':>7s} {calls:>6d}  {note}")

    print("-" * 64)
    print(f"{ok_n}/{len(cases())} 1-D patterns lower without error")
    if FAILS:
        print(f"\nLANDMINES: {', '.join(FAILS)}")
        return 1
    print("no landmines: every 1-D pattern lowers")
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
