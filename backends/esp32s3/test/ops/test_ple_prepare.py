# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host gates for fused per-layer embedding preparation."""

import torch

from executorch.backends.esp32s3.ops.ple_prepare import PLE_PREPARE


class Prepare(torch.nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width))

    def forward(self, projected, table):
        return PLE_PREPARE(
            projected, table, self.weight,
            96**-0.5, 64**0.5, 2**-0.5, 1e-6)


def reference(projected, table, weight):
    scaled = projected * (96**-0.5)
    normalized = (
        weight * scaled
        * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + 1e-6))
    return (normalized + table * (64**0.5)) * (2**-0.5)


def main():
    torch.manual_seed(23)
    model = Prepare().eval()
    projected = torch.randn(1, 6, 64)
    table = torch.randn_like(projected)
    actual = model(projected, table)
    expected = reference(projected, table, model.weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("PASS eager PLE preparation is operation-order exact")

    exported = torch.export.export(model, (projected, table))
    from executorch.exir import to_edge

    program = to_edge(exported).to_executorch()
    operators = {
        operator.name + (f".{operator.overload}" if operator.overload else "")
        for plan in program.executorch_program.execution_plan
        for operator in plan.operators
    }
    assert operators == {"espressif::ple_prepare.out"}, operators
    print("PASS one emitted op replaces RMSNorm + three muls + add")


def test_main():
    """Under pytest, what a script run checks: main() asserts each check
    (or returns non-zero), and this fails with it."""
    assert main() in (None, 0)


if __name__ == "__main__":
    main()
