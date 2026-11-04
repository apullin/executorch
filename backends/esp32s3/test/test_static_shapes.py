# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static shapes are a declared precondition, not an accident.

The delegate resolves every extent at export: `OpDesc` carries m/k/n as uint32,
preprocess pads and packs weights to exactly that k_pad, and init sizes the
activation, unpack and chain scratch from those numbers. A symbolic dim has no
value to write into any of them, so the failure mode WITHOUT this check is the
worst kind -- the tracing example's shape gets baked in, parity passes on that
example, and the next input silently reads the wrong number of bytes.

So the claim under test is a refusal, in three parts:

  1. a dynamic-batch model still LOWERS and still RUNS (portable kernels handle
     dynamic shapes; refusing to delegate must not break the export),
  2. its matmuls are NOT claimed -- zero delegate calls, with a stated reason,
  3. the byte-identical STATIC twin is still fully delegated, which is what
     makes (2) evidence about dynamism rather than about the matcher.

Part (3) is the control: without it, "0 delegate calls" could just as well mean
the pattern matcher broke.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* fails
naming its failed checks (@case).

    python test_static_shapes.py
"""

import functools
import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
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


class MLP(nn.Module):
    """Two Linears with a relu -- the shape the partitioner is most eager to claim."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _lower(model, x, dynamic_shapes=None):
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes

    ep = torch.export.export(model, (x,), dynamic_shapes=dynamic_shapes)
    lowered = to_edge_transform_and_lower(ep, partitioner=[Esp32Partitioner()])
    apply_passes(lowered)
    gm = lowered.exported_program().graph_module
    n_delegate = sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and "call_delegate" in str(n.target)
    )
    # to_executorch is where a half-claimed dynamic graph would blow up, so it is
    # part of the claim that this succeeds rather than being skipped.
    et = lowered.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16)
    )
    return n_delegate, gm, len(et.buffer)


def _mlp():
    """The MLP at seed 0, and an example batch of 8."""
    torch.manual_seed(0)
    return MLP().eval(), torch.randn(8, 32)


def _lower_static():
    """The static control, lowered: _lower's (delegate calls, graph, .pte
    size)."""
    model, x_static = _mlp()
    return _lower(model, x_static)


def _lower_dynamic():
    """The same MLP with a dynamic batch, lowered: _lower's (delegate calls,
    graph, .pte size)."""
    model, x_dyn = _mlp()
    batch = torch.export.Dim("b", min=2, max=64)
    return _lower(model, x_dyn, dynamic_shapes={"x": {0: batch}})


@case
def test_static_control():
    print("static control (batch 8):")
    n_static, gm_static, size_static = _lower_static()
    check("static model delegates", n_static > 0, f"{n_static} delegate call(s)")
    static_mms = sum(
        1 for n in gm_static.graph.nodes
        if n.op == "call_function" and "addmm" in str(n.target)
    )
    check("no addmm left in the remainder", static_mms == 0,
          f"{static_mms} portable addmm")


@case
def test_dynamic_batch():
    print("dynamic batch (Dim('b', 2..64)):")
    n_dyn, gm_dyn, size_dyn = _lower_dynamic()
    check("dynamic model still lowers to a program", size_dyn > 0,
          f"{size_dyn} B .pte")
    check("dynamic matmuls are NOT delegated", n_dyn == 0,
          f"{n_dyn} delegate call(s)")
    dyn_mms = sum(
        1 for n in gm_dyn.graph.nodes
        if n.op == "call_function" and "addmm" in str(n.target)
    )
    check("they stayed on portable kernels instead", dyn_mms > 0,
          f"{dyn_mms} portable addmm")


@case
def test_refusal_reported():
    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        Esp32OperatorSupport,
    )
    from executorch.backends.esp32s3.operator_support.patterns import _is_symbolic_shaped

    print("the refusal is reported, not silent:")
    _, gm_static, _ = _lower_static()
    _, gm_dyn, _ = _lower_dynamic()
    sym_nodes = [n for n in gm_dyn.graph.nodes if _is_symbolic_shaped(n)]
    check("symbolic nodes detected in the dynamic graph", len(sym_nodes) > 0,
          f"{len(sym_nodes)} node(s)")
    check("static graph has no symbolic node",
          not any(_is_symbolic_shaped(n) for n in gm_static.graph.nodes))

    # The counter that matters lives on the support object partition() creates
    # internally and then discards, so assert on a partition run rather than on
    # a fresh object (which trivially reads zero and proves nothing).
    support = Esp32OperatorSupport()
    for n in gm_dyn.graph.nodes:
        if n.op == "call_function":
            support.is_node_supported(None, n)
    check("the support gate refuses symbolic nodes", support.dynamic_refused > 0,
          f"{support.dynamic_refused} refusal(s) counted")

    static_support = Esp32OperatorSupport()
    for n in gm_static.graph.nodes:
        if n.op == "call_function":
            static_support.is_node_supported(None, n)
    check("and refuses nothing in the static twin",
          static_support.dynamic_refused == 0)


@case
def test_dynamic_qk_static_key():
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes

    # The q@kT route is a THIRD way a node can be claimed because of a consumer:
    # a permute of a fixed-length key is static even when the query's length is
    # dynamic, so the permute could be claimed while its bmm is refused -- an
    # empty partition, which kills the export rather than leaving the subgraph
    # portable.
    class QK(nn.Module):
        def forward(self, q, k):
            scores = torch.bmm(q, k.transpose(1, 2)) / 8.0
            return torch.softmax(scores, dim=-1)

    torch.manual_seed(0)
    qk = QK().eval()
    q = torch.randn(1, 6, 16)
    k = torch.randn(1, 12, 16)
    t = torch.export.Dim("t", min=2, max=64)
    ep_qk = torch.export.export(qk, (q, k), dynamic_shapes={"q": {1: t}, "k": None})
    try:
        low = to_edge_transform_and_lower(
            ep_qk, partitioner=[Esp32Partitioner(quantize_qk=True)])
        apply_passes(low)
        check("dynamic q@kT with a static key lowers without an empty partition",
              True, "no RuntimeError")
    except RuntimeError as e:
        check("dynamic q@kT with a static key lowers without an empty partition",
              False, str(e)[:70])


@case
def test_selected_head_dynamic_batch():
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes

    # A fourth route: the selected-row head
    # (`x @ index_select(W, 0, rows).T`) is claimed by its own branch of
    # `_core_problem`, which must apply the symbolic-dim test too: int() on a
    # backed SymInt specialises silently, so a dynamic-batch selected head
    # would delegate with the example batch baked in.
    print("selected-row head with a dynamic batch:")

    class SelectedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(257, 64))

        def forward(self, value, rows):
            return torch.nn.functional.linear(
                value, torch.index_select(self.weight, 0, rows))

    torch.manual_seed(0)
    sel = SelectedHead().eval()
    value = torch.randn(2, 64)
    rows = torch.tensor([1, 5, 8, 13, 21, 144, 233])
    b = torch.export.Dim("b", min=1, max=8)

    def _lower_selected(dynamic_shapes):
        from executorch.exir import to_edge_transform_and_lower

        ep_sel = torch.export.export(sel, (value, rows),
                                     dynamic_shapes=dynamic_shapes)
        part = Esp32Partitioner()
        low = to_edge_transform_and_lower(ep_sel, partitioner=[part])
        apply_passes(low)
        gm_sel = low.exported_program().graph_module
        n_del = sum(1 for n in gm_sel.graph.nodes
                    if n.op == "call_function"
                    and "call_delegate" in str(n.target))
        return n_del, part.support.kinds

    n_sel_static, _ = _lower_selected(None)
    check("static selected head delegates (control)", n_sel_static == 1,
          f"{n_sel_static} delegate call(s)")
    n_sel_dyn, kinds = _lower_selected({"value": {0: b}, "rows": None})
    check("dynamic-batch selected head is NOT delegated", n_sel_dyn == 0,
          f"{n_sel_dyn} delegate call(s)")
    check("and the refusal names the dynamic shape",
          kinds.get("dynamic shape", 0) > 0, str(dict(kinds)))


def main():
    for test in (test_static_control, test_dynamic_batch,
                 test_refusal_reported, test_dynamic_qk_static_key,
                 test_selected_head_dynamic_batch):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("static-shape precondition holds: dynamic refused, static claimed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
