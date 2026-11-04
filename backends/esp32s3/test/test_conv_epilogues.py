# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every conv variant, every epilogue, both spellings. No landmines.

The activation gates accept a convolution as their producer, and so must every
gate on the path from it -- including the gate for the *view* between a conv
and its activation, where a plain `target in SUPPORTED` membership test is not
enough: conv is claimed by its own branch, not by membership. A gate that
disagrees with the walk is a CRASH, not a missed fusion: `conv -> reshape ->
relu` would claim the relu, refuse the view, and preprocess would walk from the
relu to a producer nobody claimed:

    RuntimeError: claimed clamp's producer was never claimed; gate and walk
    disagree

That is not a graceful refusal, it is a stop-dead for anyone who reshapes
between a convolution and its activation, which is an ordinary thing to write.

Hence this file sweeps rather than spot-checks: whenever an epilogue gate
accepts a new producer kind, EVERY gate on the path from that producer to the
epilogue has to accept it too. A gate that lags behind does not under-claim
safely; it desynchronises the gate from the walk. So the axes are crossed
exhaustively instead of sampled -- a failure can sit in one cell of the grid.

Bar: lower and serialize without erroring, AND leave nothing portable behind --
an unfused epilogue after a claimed conv is a real regression, because the
runtime applies conv epilogues.

    python test_conv_epilogues.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# (name, conv factory, input shape, the view's target shape)
VARIANTS = [
    ("conv2d", lambda: nn.Conv2d(4, 8, 3, padding=1), (1, 4, 8, 8), (1, 8, -1)),
    ("conv1d", lambda: nn.Conv1d(4, 8, 3, padding=1), (1, 4, 32), (1, 8, -1)),
    ("conv3d", lambda: nn.Conv3d(4, 8, 3, padding=1), (1, 4, 4, 6, 6), (1, 8, -1)),
    ("depthwise", lambda: nn.Conv2d(8, 8, 3, padding=1, groups=8),
     (1, 8, 8, 8), (1, 8, -1)),
    ("conv_t2d", lambda: nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1),
     (1, 4, 8, 8), (1, 8, -1)),
    ("pointwise", lambda: nn.Conv2d(4, 8, 1), (1, 4, 8, 8), (1, 8, -1)),
    ("strided", lambda: nn.Conv2d(4, 8, 3, stride=2, padding=1),
     (1, 4, 8, 8), (1, 8, -1)),
    ("grouped", lambda: nn.Conv2d(8, 16, 3, padding=1, groups=2),
     (1, 8, 8, 8), (1, 16, -1)),
]

EPILOGUES = [
    ("relu", F.relu),
    ("gelu", F.gelu),
    ("clamp", lambda t: t.clamp(-2.0, 2.0)),
    ("silu", F.silu),
    ("hardtanh", F.hardtanh),
    ("relu6", lambda t: F.hardtanh(t, 0.0, 6.0)),
    # The residual is the one epilogue that is not pointwise-on-self: it takes
    # a second SAME-SHAPED tensor, so it exercises the "exactly one fusable
    # side" rule as well as the producer test. `t + 1.0` would not do -- a
    # scalar add is correctly refused as a residual, and writing it that way
    # measures the test's mistake rather than the backend's behaviour.
    ("residual", None),
]


# Cells that are refused rather than fused, deliberately, with the reason. An
# entry here is a claim about the BACKEND, so the test fails both ways: if an
# unlisted cell stops fusing, and if a listed cell starts. A silent allowance
# that outlives its cause is how a coverage table starts lying.
# Keys are (variant, epilogue, via_view); `None` in a slot means "any".
KNOWN_UNFUSED = {
    # Grouped fuses POINTWISE epilogues: the g ops own disjoint output
    # channels and relu/gelu/clamp/silu are functions of one element, so setting
    # the flag on all g is exactly equivalent to applying it to the whole
    # output. The residual is the one that does not follow, and the reason is
    # indexing rather than difficulty: the runtime adds it during the NCHW
    # scatter, indexed as the destination is, and a group's destination is
    # offset into a tensor the residual pointer does not share an origin with.
    # Cheap to refuse, silently wrong to get slightly right.
    ("grouped", "residual", None),
    # `conv(x).reshape(s) + r.reshape(s)`. The conv side fuses; what stays
    # portable is the RESIDUAL OPERAND's own reshape, and it has to -- the
    # delegate reads that tensor as an input, so something must materialise it.
    # One node, correct rather than missed. Listed so the count is asserted
    # rather than assumed. Not conv1d: its (1, 8, 32) output reshaped to
    # (1, 8, -1) is the identity, which apply_passes removes.
    *((variant, "residual", True) for variant in (
        "conv2d", "conv3d", "depthwise", "conv_t2d", "pointwise", "strided")),
}


def is_known(variant, epilogue, via):
    """Does any KNOWN_UNFUSED pattern cover this cell? `None` matches anything."""
    return any(
        (v is None or v == variant) and (e is None or e == epilogue)
        and (w is None or w == via)
        for v, e, w in KNOWN_UNFUSED
    )


class Case(nn.Module):
    """`ep is None` means the residual: a second same-shaped tensor is added."""

    def __init__(self, conv, ep, shape, via_view):
        super().__init__()
        self.conv, self.ep, self.shape, self.via = conv, ep, shape, via_view

    def forward(self, x, r=None):
        y = self.conv(x)
        if self.via:
            y = y.reshape(*self.shape)
        if self.ep is None:
            return y + (r.reshape(*self.shape) if self.via else r)
        return self.ep(y)


def portable_count(gm):
    """call_functions the delegate did NOT take. getitem is bookkeeping."""
    return sum(1 for n in gm.graph.nodes
               if n.op == "call_function"
               and "executorch_call_delegate" not in str(n.target)
               and "getitem" not in str(n.target))


def main():
    from executorch.backends.test.harness.stages import StageType

    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    crashes, unfused, stale = [], [], []
    print(f"{'variant':11s} {'epilogue':10s} {'direct':>7s} {'via view':>9s}"
          "   (portable call_functions left; 0 == fully fused)")
    print("-" * 74)
    for vname, mk, shape, vshape in VARIANTS:
        for ename, ep in EPILOGUES:
            cells = []
            for via in (False, True):
                torch.manual_seed(0)
                mod = Case(mk(), ep, vshape, via).eval()
                x = torch.randn(*shape)
                args = (x,)
                if ep is None:  # residual: shape the skip like the conv output
                    with torch.no_grad():
                        args = (x, torch.randn_like(mod.conv(x)))
                try:
                    t = Esp32Tester(mod, args)
                    t.export().to_edge_transform_and_lower()
                    # Count BEFORE to_executorch. That stage mutates the edge
                    # program in place, so reading the artifact afterwards
                    # attributes its bookkeeping ops to the partitioner.
                    # Serialize after, so the row still proves the whole
                    # pipeline runs.
                    n = portable_count(t.get_artifact(
                        StageType.TO_EDGE_TRANSFORM_AND_LOWER
                    ).exported_program().graph_module)
                    t.to_executorch().serialize()
                    cells.append(str(n))
                    kn = is_known(vname, ename, via)
                    if n and not kn:
                        unfused.append(
                            f"{vname}/{ename}{' via view' if via else ''} ({n})")
                    elif not n and kn:
                        stale.append(f"{vname}/{ename}")
                except Exception as e:  # noqa: BLE001 -- cataloguing is the point
                    cells.append("CRASH")
                    crashes.append(
                        f"{vname}/{ename}{' via view' if via else ''}: "
                        f"{type(e).__name__}: {str(e).splitlines()[0][:44]}")
            print(f"{vname:11s} {ename:10s} {cells[0]:>7s} {cells[1]:>9s}")

    known = sum(1 for vn, _m, _s, _vs in VARIANTS for e, _f in EPILOGUES
                for v in (False, True) if is_known(vn, e, v))
    total = len(VARIANTS) * len(EPILOGUES) * 2
    print("-" * 74)
    print(f"{total - len(crashes)}/{total} lower; "
          f"{total - len(crashes) - len(unfused) - known}/{total - known} fuse "
          f"completely ({known} known-unfused excluded, see KNOWN_UNFUSED)")
    if crashes:
        print("\nLANDMINES (these stop a user dead):")
        for c in crashes:
            print(f"  {c}")
    if unfused:
        print("\nUNFUSED (epilogue left portable behind a claimed conv):")
        for u in unfused:
            print(f"  {u}")
    if stale:
        print("\nSTALE EXCEPTION -- these now fuse, drop them from "
              "KNOWN_UNFUSED:")
        for t_ in stale:
            print(f"  {t_}")
    if crashes or unfused or stale:
        return 1
    print("every conv variant fuses every epilogue, both spellings")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() asserts each check
    (or returns non-zero), and this fails with it."""
    assert main() in (None, 0)


if __name__ == "__main__":
    sys.exit(main())
