# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every 2-D op someone might build an image network from. No landmines.

The 1-D sweep exists because audio and timeseries models are conv1d plus
pooling plus normalisation plus glue, and one op that stops the lowering makes
the rest irrelevant. Image models are the same claim with a different op set.
"A user can build almost any network for image data" is a claim about
COVERAGE, and coverage cannot be argued from the ops that are implemented. It
has to be argued from the ops that are not.

Same bar as the 1-D sweep: a pattern must export, lower and serialize WITHOUT
ERRORING. Delegating nothing is a pass -- a max-pool has no matmul in it and
correctly gets a portable kernel. The failure this catches is the one that stops
a user dead, which is a different and much worse thing than a refusal.

Reported per op rather than asserted in aggregate: "31 of 34 lower" is useless
without knowing which three.

    python test_ops_2d.py
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


C, H, W = 8, 16, 16
X = (1, C, H, W)


def cases():
    return [
        # --- convolution family (the delegated ones; here for regression)
        ("conv2d", nn.Conv2d(C, 16, 3, padding=1), X),
        ("conv2d dilated", nn.Conv2d(C, 16, 3, dilation=2), X),
        ("conv2d asymmetric k", nn.Conv2d(C, 16, (1, 5), padding=(0, 2)), X),
        ("conv2d no bias", nn.Conv2d(C, 16, 3, padding=1, bias=False), X),
        ("depthwise separable", nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C), nn.Conv2d(C, 16, 1)), X),
        ("conv_transpose2d", nn.ConvTranspose2d(C, 16, 2, stride=2), X),
        ("conv2d + bn + relu", nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1), nn.BatchNorm2d(16).eval(),
            nn.ReLU()), X),
        # --- pooling, the part every detector and segmenter is made of
        ("max_pool2d", nn.MaxPool2d(2), X),
        ("max_pool2d s1 p1", nn.MaxPool2d(3, stride=1, padding=1), X),
        ("avg_pool2d", nn.AvgPool2d(2), X),
        ("min-pool via -max(-x)", Fn(lambda x: -F.max_pool2d(-x, 2)), X),
        ("adaptive_avg_pool2d", nn.AdaptiveAvgPool2d(1), X),
        ("adaptive_avg_pool2d 4x4", nn.AdaptiveAvgPool2d((4, 4)), X),
        ("adaptive_max_pool2d", nn.AdaptiveMaxPool2d((4, 4)), X),
        ("global avg (mean HW)", Fn(lambda x: x.mean((-2, -1))), X),
        ("SPPF-style pool cat", Fn(lambda x: torch.cat(
            [x, F.max_pool2d(x, 5, 1, 2), F.max_pool2d(x, 9, 1, 4)], 1)), X),
        # --- normalisation
        ("batch_norm2d", nn.BatchNorm2d(C).eval(), X),
        ("instance_norm2d", nn.InstanceNorm2d(C), X),
        ("group_norm", nn.GroupNorm(4, C), X),
        ("layer_norm over C (NHWC)", nn.LayerNorm(C), (1, H, W, C)),
        # LayerNorm2d as SAM2 and ConvNeXt write it: normalise over the CHANNEL
        # axis of an NCHW tensor, which is a stride-HW reduction rather than a
        # contiguous one.
        ("layer_norm2d over NCHW C", Fn(
            lambda x: (x - x.mean(1, keepdim=True))
            * torch.rsqrt(x.var(1, keepdim=True, unbiased=False) + 1e-6)), X),
        # --- resampling: the whole decoder half of any segmenter
        ("upsample nearest", Fn(lambda x: F.interpolate(x, scale_factor=2.0,
                                                        mode="nearest")), X),
        ("upsample bilinear", Fn(lambda x: F.interpolate(
            x, scale_factor=2.0, mode="bilinear", align_corners=False)), X),
        ("upsample to size", Fn(lambda x: F.interpolate(x, size=(20, 20),
                                                        mode="nearest")), X),
        ("pixel_shuffle", nn.PixelShuffle(2), (1, C * 4, H, W)),
        ("pixel_unshuffle", nn.PixelUnshuffle(2), X),
        # --- shape plumbing
        ("pad zeros", Fn(lambda x: F.pad(x, (1, 1, 1, 1))), X),
        ("pad reflect", Fn(lambda x: F.pad(x, (2, 2, 2, 2), mode="reflect")), X),
        ("pad replicate", Fn(lambda x: F.pad(x, (1, 1, 1, 1),
                                             mode="replicate")), X),
        ("crop / slice", Fn(lambda x: x[..., 2:-2, 2:-2]), X),
        ("stride-2 slice (focus)", Fn(lambda x: torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2]], 1)), X),
        ("cat along C", Fn(lambda x: torch.cat([x, x], 1)), X),
        ("chunk + add", Fn(lambda x: sum(x.chunk(2, 1))), X),
        ("permute NCHW->NHWC", Fn(lambda x: x.permute(0, 2, 3, 1).contiguous()),
         X),
        ("flatten + linear", nn.Sequential(nn.Flatten(),
                                           nn.Linear(C * H * W, 16)), X),
        ("channel shuffle", Fn(
            lambda x: x.view(1, 2, 4, H, W).transpose(1, 2).reshape(1, 8, H, W)),
         X),
        # --- attention over a feature map, as every ViT-ish vision model does
        ("flatten + self-attention", Fn(
            lambda x: (lambda t: torch.softmax(
                t @ t.transpose(-1, -2) / 8.0, -1) @ t)(
                    x.flatten(2).transpose(1, 2))), X),
        ("squeeze-excite", Fn(
            lambda x: x * torch.sigmoid(x.mean((-2, -1), keepdim=True))), X),
        # --- elementwise glue
        ("residual add", Fn(lambda x: x + x * 0.5), X),
        ("hardswish", nn.Hardswish(), X),
        ("softmax over C", Fn(lambda x: x.softmax(1)), X),
        ("clamp chain", Fn(lambda x: x.clamp(-3, 3).relu()), X),
    ]


def main():
    from executorch.backends.test.harness.stages import StageType

    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    print(f"{'op':28s} {'lowers':>7s} {'deleg':>6s}  notes")
    print("-" * 66)
    ok_n = 0
    for name, mod, shape in cases():
        torch.manual_seed(0)
        x = torch.randn(*shape)
        try:
            t = Esp32Tester(mod.eval(), (x,))
            t.export().to_edge_transform_and_lower()
            gm = t.get_artifact(
                StageType.TO_EDGE_TRANSFORM_AND_LOWER
            ).exported_program().graph_module
            calls = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                        and "executorch_call_delegate" in str(n.target))
            t.to_executorch().serialize()
            note = f"{len(t.get_artifact(StageType.SERIALIZE)):,} B"
            ok_n += 1
            good = True
        except Exception as e:  # noqa: BLE001 -- cataloguing these is the point
            good, calls = False, 0
            note = f"{type(e).__name__}: {str(e).splitlines()[0][:38]}"
            FAILS.append(name)
        print(f"{name:28s} {'yes' if good else 'NO':>7s} {calls:>6d}  {note}")

    print("-" * 66)
    print(f"{ok_n}/{len(cases())} 2-D patterns lower without error")
    if FAILS:
        print(f"\nLANDMINES: {', '.join(FAILS)}")
        return 1
    print("no landmines: every 2-D pattern lowers")
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
