# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every refusal states a reason, through ExecuTorch's own reporting channel.

Why this exists. This backend refuses a lot on purpose -- dynamic shapes, scaled
addmm, a fusion whose producer has a second consumer -- and without a stated
reason every one of those refusals looks identical from outside: the op simply
is not in a delegate. "Nothing delegated" is the same observation whether the
cause is a stated precondition or the pattern matcher being broken.

`WhyNoPartitionReporter` is ExecuTorch's existing answer (the Arm backend reports
through it, which is what makes its table the format a reviewer already knows how
to read), so the backend reports through it rather than inventing a channel.

Four claims:

  1. a refusal carries a reason NAMING ITS CAUSE, not a generic "unsupported" --
     checked on the dynamic dim, the refusal a user is most likely to hit by
     accident, and on the gated multiply the fp32 diagnostic policy declines,
  2. the reason is ACTIONABLE where an action exists (the multiply one names
     the fix),
  3. CLAIMED nodes are absent from the report -- a reporter that accuses
     everything is as useless as one that accuses nothing, and this is the half
     that silently rots,
  4. structural nodes (placeholder/output) are never listed, so the table shows
     ops a user could plausibly have expected to delegate.

    python test_why_no_partition.py
"""

import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


class MLP(nn.Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.approximate = approximate

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x),
                                                 approximate=self.approximate))


class ConvAfterActivationPermute(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(8, 8, 3, padding=1, groups=8)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class Linear3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x):
        return self.fc(x)


class Linear3Residual(Linear3):
    def forward(self, x, r):
        return self.fc(x) + r


class Conv1x1(nn.Module):
    def __init__(self, channels_last_copy=False):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 1)
        self.channels_last_copy = channels_last_copy

    def forward(self, x):
        if self.channels_last_copy:
            x = x.to(memory_format=torch.channels_last)
        return self.conv(x)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(16, 32, bias=False)
        self.u = nn.Linear(16, 32, bias=False)

    def forward(self, x):
        return torch.nn.functional.silu(self.g(x)) * self.u(x)


def lower(model, x, dynamic_shapes=None, **partitioner_kw):
    """Lower and hand back the partitioner, which holds the reasons."""
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    part = Esp32Partitioner(**partitioner_kw)
    args = x if isinstance(x, tuple) else (x,)
    ep = torch.export.export(model.eval(), args, dynamic_shapes=dynamic_shapes)
    edge = to_edge_transform_and_lower(ep, partitioner=[part])
    return part, edge


def reasons(part):
    """{node name: reason}. Reads the reporter's own store, so the test covers
    what a user actually receives rather than a parallel bookkeeping path."""
    return {n.name: r for n, r in part.reporter._rejected_nodes.items()}


def main():
    print("dynamic dim is refused BY NAME:")
    batch = torch.export.Dim("b", min=2, max=64)
    part, _ = lower(MLP(), torch.randn(4, 16), dynamic_shapes={"x": {0: batch}})
    rs = reasons(part)
    mm = [r for n, r in rs.items() if "addmm" in n or "mm" in n]
    check("the matmuls are reported", bool(mm), f"{len(mm)} matmul node(s)")
    check("reason names the dynamic dim, not a generic 'unsupported'",
          bool(mm) and all("dynamic shape" in r for r in mm))
    check("reason says static shapes are the precondition",
          bool(mm) and all("static shapes only" in r for r in mm))
    # The counter behind the partitioner's unconditional summary line.
    check("summary counter agrees with the report",
          sum(part.support.kinds.values()) == len(rs),
          f"{sum(part.support.kinds.values())} counted, {len(rs)} reported")

    # The tanh-approximate gelu fuses (the epilogue runs ExecuTorch's own tanh
    # formula), so the stated decision here is the fp32 policy's gated multiply.
    print("a gated multiply under the fp32 policy is a stated decision, "
          "with the fix:")
    part, _ = lower(SwiGLU(), torch.randn(4, 16), format_policy="fp32")
    rs = reasons(part)
    mul = [r for n, r in rs.items() if "mul" in n]
    check("the gate*up mul is reported", len(mul) == 1,
          f"{len(mul)} mul node(s)")
    check("reason distinguishes it from an unimplemented op",
          bool(mul) and "multiply unavailable under fp32 policy" in mul[0])
    check("reason names the fix (a quantised format policy)",
          bool(mul) and "quantised format policy" in mul[0])

    print("claimed nodes are NOT accused:")
    # The static, exact-gelu twin: every matmul here IS delegated, so a matmul
    # appearing in the report would mean the reporter fires on success -- the
    # failure this half of the test exists to catch.
    part, edge = lower(MLP(approximate="none"), torch.randn(4, 16))
    rs = reasons(part)
    gm = edge.exported_program().graph_module
    n_delegate = sum(1 for n in gm.graph.nodes
                     if n.op == "call_function"
                     and "executorch_call_delegate" in str(n.target))
    check("the control model actually delegates", n_delegate > 0,
          f"{n_delegate} delegate call(s)")
    accused_mm = [n for n in rs if "mm" in n]
    check("no delegated matmul appears in the report", not accused_mm,
          f"accused: {accused_mm}" if accused_mm else "")
    check("the exact gelu is not accused either",
          not [n for n in rs if "gelu" in n])
    part, _ = lower(MLP(approximate="tanh"), torch.randn(4, 16))
    check("nor the tanh gelu, which fuses",
          not [n for n in reasons(part) if "gelu" in n])

    print("structural nodes are never listed:")
    ep = torch.export.export(MLP().eval(), (torch.randn(4, 16),))
    names = {n.name for n in ep.graph.nodes if n.op in ("placeholder", "output")}
    check("no placeholder/output in the report", not (names & set(rs)),
          f"leaked: {sorted(names & set(rs))}" if (names & set(rs)) else "")
    # get_table_report() skips these itself, but the reporter is fed by
    # is_node_supported: keeping them out at the SOURCE is what makes the raw
    # store usable by anything other than the table renderer.
    table = part.reporter.get_table_report()
    check("the table renders", isinstance(table, str) and len(table) > 0)

    print("activation permutes stay outside claimed conv partitions:")
    part, edge = lower(ConvAfterActivationPermute(), torch.randn(1, 16, 8))
    gm = edge.exported_program().graph_module
    n_delegate = sum(1 for n in gm.graph.nodes
                     if n.op == "call_function"
                     and "executorch_call_delegate" in str(n.target))
    check("conv still delegates across a portable activation transpose",
          n_delegate == 1, f"{n_delegate} delegate call(s)")
    rs = reasons(part)
    permutes = [r for n, r in rs.items() if "permute" in n]
    check("the activation transpose is explicitly refused as non-constant",
          bool(permutes) and all("activation transpose" in r for r in permutes),
          f"{len(permutes)} reported")

    print("a runtime tensor that is not row-major is refused BY NAME:")
    # ExecuTorch keeps a tensor's dim order and the delegate indexes every
    # activation row-major, so claiming any of these reads the wrong elements.
    nchw = torch.randn(1, 4, 5, 5)
    for name, model, args in (
            ("a transposed matmul input", Linear3(), torch.randn(3, 2).t()),
            ("a channels-last conv input", Conv1x1(),
             nchw.to(memory_format=torch.channels_last)),
            ("a conv after an in-graph channels-last copy",
             Conv1x1(channels_last_copy=True), nchw)):
        part, edge = lower(model, args)
        gm = edge.exported_program().graph_module
        n_delegate = sum(1 for n in gm.graph.nodes
                         if n.op == "call_function"
                         and "executorch_call_delegate" in str(n.target))
        layout = [r for r in reasons(part).values() if "row-major" in r]
        check(f"{name}: not delegated, and the reason names the layout",
              n_delegate == 0 and bool(layout),
              f"{n_delegate} delegate call(s), {len(layout)} layout reason(s)")
    part, _ = lower(Linear3Residual(), (torch.randn(2, 3), torch.randn(4, 2).t()))
    adds = [r for n, r in reasons(part).items() if n.startswith("aten_add")]
    check("a transposed residual stays out of the matmul's epilogue",
          bool(adds) and all("row-major" in r for r in adds),
          adds[0][:90] if adds else "the add was claimed")

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("every refusal states a cause, and only refusals do")
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
