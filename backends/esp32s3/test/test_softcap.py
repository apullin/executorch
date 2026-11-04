# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selective fusion coverage for Glimmer's scaled logit softcap.

The supported spelling is exact and deliberately narrow:

    cap * tanh((scale * linear(x)) / cap)

with the pre-scale optional (Gemma 2/3n write ``tanh(linear(x) / cap) * cap``;
the descriptor then carries scale 1.0, which the runtime's ``y *= 1.0f``
applies exactly). Both cap uses must be the same positive build-time scalar,
every link must be single-use, and division order matters: multiplying by a
reciprocal is different arithmetic. Near misses stay portable rather than
being silently replaced by a similar nonlinear function.
"""

import contextlib
import io
import sys

import torch
import torch.nn as nn


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


class Head(nn.Module):
    def __init__(self, variant="exact"):
        super().__init__()
        self.linear = nn.Linear(16, 32, bias=False)
        self.variant = variant

    def forward(self, x, cap=None):
        if self.variant == "unscaled":
            return torch.tanh(self.linear(x) / 20.0) * 20.0
        if self.variant == "reciprocal":
            return torch.tanh(self.linear(x) * 0.05) * 20.0
        y = self.linear(x) * 0.196116
        if self.variant == "exact":
            return torch.tanh(y / 20.0) * 20.0
        if self.variant == "mismatched":
            return torch.tanh(y / 20.0) * 19.0
        if self.variant == "reversed":
            return torch.tanh(20.0 / y) * 20.0
        if self.variant == "runtime":
            return torch.tanh(y / cap) * cap
        if self.variant == "multiuse":
            return torch.tanh(y / 20.0) * 20.0 + y
        raise AssertionError(self.variant)


def lower(variant, shape=(2, 16)):
    from executorch.backends.test.harness.stages import StageType

    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    args = [torch.randn(*shape)]
    if variant == "runtime":
        args.append(torch.tensor(20.0))
    tester = Esp32Tester(Head(variant).eval(), tuple(args))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        tester.export().to_edge_transform_and_lower()
    edge = tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    targets = [str(n.target) for n in edge.exported_program().graph.nodes
               if n.op == "call_function"]
    portable = [t for t in targets
                if any(op in t for op in
                       ("aten.tanh.", "aten.div.", "aten.mul."))]
    delegates = sum("executorch_call_delegate" in t for t in targets)
    return portable, delegates, output.getvalue()


def lower_pass_through():
    """Exercise the already-quantized weight packer, not only plain fp32 weights."""
    from executorch.exir import to_edge
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    x = torch.randn(2, 16)
    gm = torch.export.export(Head("exact").eval(), (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=4))
    prepared(x)
    converted = convert_pt2e(prepared)
    edge = to_edge(torch.export.export(converted, (x,)))
    strip_dynamic_act_qdq(edge)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        edge = edge.to_backend(Esp32Partitioner())
    targets = [str(n.target) for n in edge.exported_program().graph.nodes
               if n.op == "call_function"]
    portable = [t for t in targets
                if any(op in t for op in
                       ("aten.tanh.", "aten.div.", "aten.mul."))]
    delegates = sum("executorch_call_delegate" in t for t in targets)
    return portable, delegates, output.getvalue()


def main():
    print("positive spellings:")
    for variant, suffix in (("exact", "+softcap[0.196116,20]"),
                            ("unscaled", "+softcap[1,20]")):
        for shape in ((2, 16), (1, 3, 16)):
            portable, delegates, log = lower(variant, shape)
            tag = f"{variant} {len(shape)}-D"
            check(f"{tag} head fuses", not portable, f"portable={portable}")
            check(f"{tag} emits one delegated head", delegates == 1,
                  f"delegates={delegates}")
            check(f"{tag} descriptor records both scalars", suffix in log)

    portable, delegates, log = lower_pass_through()
    check("exact quantized head fuses", not portable,
          f"portable={portable}")
    check("exact quantized head emits one delegate", delegates == 1,
          f"delegates={delegates}")
    check("softcap reaches pass-through i4 descriptor",
          "addmmPT_i4" in log and "+softcap[0.196116,20]" in log)

    print("negative spellings:")
    for variant in ("mismatched", "reversed", "reciprocal", "runtime",
                    "multiuse"):
        portable, delegates, log = lower(variant)
        check(f"{variant} stays portable", bool(portable),
              f"portable={portable}")
        check(f"{variant} does not set the softcap flag",
              "+softcap[" not in log)
        check(f"{variant} still delegates its linear", delegates == 1,
              f"delegates={delegates}")

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("softcap fusion is exact on its intended spelling and selective otherwise")
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
