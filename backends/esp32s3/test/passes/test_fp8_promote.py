# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp8-distributed checkpoints load, delegate, and are bit-exact.

The claim being tested is narrow and worth stating precisely: this backend
supports fp8 as a FILE FORMAT and not as an arithmetic. An fp8 checkpoint is
transcoded once on the host -- promoted to fp32, then quantised to whichever rung
the format policy chose -- and the device never sees an fp8 byte. So there is no
"is fp8 fast enough here" question to answer; nothing fp8 executes.

What could go wrong, and what each check is for:

  1. `to_edge` REJECTS fp8 outright (SpecViolationError on _to_dim_order_copy
     taking float8_e4m3fn and returning float32). That is upstream of every
     partitioner, so the positive control matters: without promotion this must
     fail, or the test proves nothing about promotion.
  2. Promotion must be LOSSLESS. fp8 -> fp32 is exact -- every fp8 value is
     representable in fp32 -- so the promoted model must agree with the fp8
     model bit for bit, not approximately. A tolerance here would hide a real
     bug, because there is no rounding to tolerate.
  3. The weight must still DELEGATE. The failure mode after a naive promotion is
     an fp32->fp32 identity cast left between the weight and its consumer:
     `_resolve_const` does not fold a cast, so the weight resolves to None and
     the op silently stays portable. Loading but not delegating is worse than
     the loud rejection it replaces, because nothing reports it.

    python test_fp8_promote.py
"""

import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


class Fp8Linear(nn.Module):
    """Weights stored fp8, promoted in forward: how a loaded fp8 checkpoint looks."""

    def __init__(self, k=64, n=32, dt=torch.float8_e4m3fn):
        super().__init__()
        torch.manual_seed(0)
        self.register_buffer("w8", torch.randn(n, k).to(dt))
        self.b = nn.Parameter(torch.randn(n))

    def forward(self, x):
        return x @ self.w8.to(torch.float32).t() + self.b


def delegate_calls(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function"
               and "executorch_call_delegate" in str(n.target))


def main():
    import contextlib
    import io

    from executorch.exir import to_edge, to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes.promote_fp8_constants_pass import (
        FP8_DTYPES,
        promote_fp8_constants,
    )

    check("torch exposes fp8 dtypes", len(FP8_DTYPES) >= 2,
          f"{[str(d) for d in FP8_DTYPES]}")

    print("the positive control -- to_edge must REJECT fp8 without promotion:")
    m = Fp8Linear().eval()
    x = torch.randn(8, 64)
    try:
        to_edge(torch.export.export(m, (x,)))
        check("unpromoted fp8 is rejected by to_edge", False,
              "to_edge ACCEPTED it, so this test no longer proves anything -- "
              "either ExecuTorch learned fp8 or the model stopped using it")
    except Exception as e:  # noqa: BLE001 -- the rejection IS the control
        check("unpromoted fp8 is rejected by to_edge",
              "float8" in str(e) or "SpecViolation" in type(e).__name__,
              f"{type(e).__name__}")

    print("promotion, per fp8 dtype:")
    for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        name = str(dt).replace("torch.", "")
        m = Fp8Linear(dt=dt).eval()
        with torch.no_grad():
            ref = m(x)
        ep = torch.export.export(m, (x,))
        n = promote_fp8_constants(ep)
        check(f"{name}: one constant promoted", n == 1, f"{n} promoted")
        # Lossless, not approximate: every fp8 value is exactly representable in
        # fp32, so anything but an exact match is a bug rather than rounding.
        got = ep.module()(x)
        check(f"{name}: bit-exact after promotion",
              torch.equal(got, ref),
              f"max abs diff {(got - ref).abs().max().item():.3e}")
        check(f"{name}: no fp8 left in the graph",
              all(not (isinstance(nd.meta.get("val"), torch.Tensor)
                       and nd.meta["val"].dtype in FP8_DTYPES)
                  for nd in ep.graph.nodes),
              "a promoted graph still carrying fp8 will be rejected downstream")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            edge = to_edge_transform_and_lower(
                ep, partitioner=[Esp32Partitioner(requantize_promoted="all")])
        gm = edge.exported_program().graph_module
        # The point of erasing the identity cast: the weight must reach the
        # delegate as a constant, not sit behind a cast that _resolve_const
        # cannot fold.
        # requantize_promoted="all" ON PURPOSE. Under the DEFAULT policy an fp8
        # weight is deliberately NOT delegated -- int8 on top of fp8's 141 levels
        # is a second quantisation the training never saw. What is checked here
        # is the PROMOTION mechanism, not the policy: the weight must reach the
        # delegate as a constant rather than sit behind a cast _resolve_const
        # cannot fold. Asking for "all" isolates that.
        check(f"{name}: the fp8 weight can delegate", delegate_calls(gm) == 1,
              f"{delegate_calls(gm)} delegate call(s) -- 0 means it loaded but "
              f"quietly stayed portable")

    print("fp16/bf16 checkpoints, which crashed rather than refused:")
    for dt in (torch.float16, torch.bfloat16):
        name = str(dt).replace("torch.", "")
        m = Fp8Linear(dt=dt).eval()
        with torch.no_grad():
            ref = m(x)
        ep = torch.export.export(m, (x,))
        check(f"{name}: promoted", promote_fp8_constants(ep) == 1)
        got = ep.module()(x)
        check(f"{name}: bit-exact", torch.equal(got, ref),
              f"max abs diff {(got - ref).abs().max().item():.3e}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            edge = to_edge_transform_and_lower(
                ep, partitioner=[Esp32Partitioner()])
        check(f"{name}: delegates",
              delegate_calls(edge.exported_program().graph_module) == 1)

    # The rule that keeps half-precision promotion honest: it is constant
    # FOLDING, not a numerics policy. A half constant whose value is consumed at
    # half precision must be left alone -- promoting it would double its memory
    # and silently change what the portable op computes.
    class HalfKept(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.randn(4, 4).to(torch.float16))

        def forward(self, t):
            return (t.to(torch.float16) * self.w).sum()

    ep3 = torch.export.export(HalfKept().eval(), (torch.randn(4, 4),))
    check("a half constant used AT half precision is left alone",
          promote_fp8_constants(ep3) == 0,
          "promoting it would change what the portable op computes")

    # fp8 is promoted unconditionally because to_edge refuses it otherwise --
    # but a use that READS the fp8 value (here: returning it) would have its
    # contract silently changed to fp32. Refused by name instead.
    class Fp8Returned(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w8", torch.randn(32, 64).to(
                torch.float8_e4m3fn))

        def forward(self, t):
            return t @ self.w8.to(torch.float32).t(), self.w8

    ep4 = torch.export.export(Fp8Returned().eval(), (x,))
    try:
        promote_fp8_constants(ep4)
        out = next(n for n in ep4.graph.nodes if n.op == "output")
        dtypes = [a.meta["val"].dtype for a in out.args[0]]
        check("an fp8 buffer that is also an output is refused", False,
              f"promoted; outputs are now {dtypes}")
    except RuntimeError as e:
        check("an fp8 buffer that is also an output is refused",
              "w8" in str(e), str(e)[:90])

    # A promoted PARAMETER must stay a Parameter: the lowered program's
    # verifier expects one under every inputs_to_parameters entry.
    class HalfParam(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.w = nn.Parameter(torch.randn(32, 64).to(torch.float16))

        def forward(self, t):
            return t @ self.w.to(torch.float32).t()

    ep5 = torch.export.export(HalfParam().eval(), (x,))
    promote_fp8_constants(ep5)
    detail = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            e5 = to_edge_transform_and_lower(
                ep5, partitioner=[Esp32Partitioner()])
        n5 = delegate_calls(e5.exported_program().graph_module)
        detail = f"{n5} delegate call(s)"
    except Exception as e:  # noqa: BLE001 -- any failure here is the bug
        n5, detail = 0, f"{type(e).__name__}: {str(e)[:70]}"
    check("an fp16 nn.Parameter promotes and still lowers", n5 == 1, detail)

    print("re-quantisation policy -- fp8 and fp16 are NOT the same case:")

    class Mixed(nn.Module):
        """One layer from a pre-quantised checkpoint, one ordinary fp32 layer."""

        def __init__(self, dt):
            super().__init__()
            torch.manual_seed(0)
            self.register_buffer("w", torch.randn(32, 64).to(dt))
            self.b = nn.Parameter(torch.randn(32))
            self.fc = nn.Linear(32, 16)

        def forward(self, t):
            return self.fc(t @ self.w.to(torch.float32).t() + self.b)

    def lower(dt, policy):
        ep_ = torch.export.export(Mixed(dt).eval(), (x,))
        promote_fp8_constants(ep_)
        buf_ = io.StringIO()
        with contextlib.redirect_stdout(buf_):
            e = to_edge_transform_and_lower(
                ep_, partitioner=[
                    Esp32Partitioner(requantize_promoted=policy)])
        g = e.exported_program().graph_module
        mm = sum(1 for n in g.graph.nodes if n.op == "call_function"
                 and ("addmm" in str(n.target) or "aten.mm" in str(n.target)))
        return delegate_calls(g), mm

    # (2, 0) == both layers delegated. (1, 1) == the pre-quantised layer stayed
    # portable at fp32 while the ORDINARY layer still delegated, which is the
    # property that makes refusing cheap: a subgraph break costs nothing here.
    DELEGATED, KEPT = (2, 0), (1, 1)
    EXPECT = {
        # fp8 has 3 mantissa bits and 141 distinct values in 4096 samples: it is
        # a quantisation scheme, so int8 on top is a SECOND lossy step the
        # training never saw. Not done by default.
        (torch.float8_e4m3fn, "half"): KEPT,
        (torch.float8_e4m3fn, "all"): DELEGATED,
        (torch.float8_e4m3fn, "none"): KEPT,
        # fp16/bf16 are storage precision, which is how nearly every checkpoint
        # ships. int8 there is the FIRST quantisation and the point of this
        # backend -- refusing them by default would make it useless for most
        # real models, so the default ("half") requantises them, and not fp8.
        (torch.float16, "half"): DELEGATED,
        (torch.float16, "all"): DELEGATED,
        (torch.float16, "none"): KEPT,
        (torch.bfloat16, "half"): DELEGATED,
        (torch.bfloat16, "all"): DELEGATED,
        (torch.bfloat16, "none"): KEPT,
    }
    for (dt, policy), want in EXPECT.items():
        name = str(dt).replace("torch.", "")
        got = lower(dt, policy)
        check(f"{name} + {policy!r}", got == want,
              f"delegate {got[0]}, portable matmuls {got[1]} "
              f"(want {want[0]}/{want[1]})")

    # The policy follows the WEIGHT -- the operand that gets quantised -- and
    # nothing else: a conv's fp16 bias does not answer for its fp8 weight, and
    # a selected-row head is admitted only after the policy is consulted.
    class Fp8ConvFp16Bias(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.register_buffer("w", torch.randn(8, 4, 3, 3).to(
                torch.float8_e4m3fn))
            self.register_buffer("b", torch.randn(8).to(torch.float16))

        def forward(self, t):
            return torch.nn.functional.conv2d(
                t, self.w.to(torch.float32), self.b.to(torch.float32), padding=1)

    class Fp8SelectedHead(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.register_buffer("table", torch.randn(64, 32).to(
                torch.float8_e4m3fn))

        def forward(self, value, rows):
            selected = torch.index_select(self.table.to(torch.float32), 0, rows)
            return torch.nn.functional.linear(value, selected)

    def delegates(model, args, policy):
        ep_ = torch.export.export(model.eval(), args)
        promote_fp8_constants(ep_)
        with contextlib.redirect_stdout(io.StringIO()):
            e = to_edge_transform_and_lower(ep_, partitioner=[
                Esp32Partitioner(requantize_promoted=policy)])
        return delegate_calls(e.exported_program().graph_module)

    conv_args = (torch.randn(1, 4, 8, 8),)
    head_args = (torch.randn(2, 32), torch.tensor([5, 1, 60, 7]))
    for name, model, args in (
            ("conv: fp8 weight, fp16 bias", Fp8ConvFp16Bias(), conv_args),
            ("selected-row head over an fp8 table", Fp8SelectedHead(),
             head_args)):
        got = (delegates(model, args, "half"), delegates(model, args, "all"))
        check(f"{name}: kept under 'half', delegated under 'all'",
              got == (0, 1), f"delegate calls {got}, want (0, 1)")

    try:
        Esp32Partitioner(requantize_promoted="yes-please")
        check("an unknown policy is rejected", False, "it was accepted")
    except ValueError:
        check("an unknown policy is rejected", True)

    print("models without fp8 are untouched:")
    plain = nn.Linear(64, 32).eval()
    ep2 = torch.export.export(plain, (x,))
    before = len(list(ep2.graph.nodes))
    check("nothing promoted", promote_fp8_constants(ep2) == 0)
    check("graph unchanged", len(list(ep2.graph.nodes)) == before,
          f"{before} -> {len(list(ep2.graph.nodes))} nodes")

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("fp8 checkpoints load, promote losslessly, and delegate")
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
