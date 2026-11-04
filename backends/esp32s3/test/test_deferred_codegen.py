# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioner defers code generation only on to_backend's throwaway copy.

ExecuTorch's to_backend partitions a fake-weight copy of the program and fuses
one partition at a time; every fusion re-sorts the graph and, through the
GraphModule.graph setter, regenerates the source of the whole module. The
partitioner makes that copy generate its code lazily (espressif_partitioner
_defer_codegen). Three claims keep that safe:

  1. a deferred module still runs and prints the CURRENT graph: calling it or
     reading .code after an edit generates the code first,
  2. only the module it was applied to changes (GraphModule gives every
     instance its own class),
  3. only to_backend's fake copy qualifies: a caller's real program handed to
     partition() directly is left alone, and the lowered result is an ordinary
     GraphModule.

    python test_deferred_codegen.py
"""

import operator
import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


class AddMul(nn.Module):
    def forward(self, x):
        return (x + 1.0) * 2.0


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def deferred_module_semantics():
    from executorch.backends.esp32s3.partition.espressif_partitioner import _defer_codegen

    print("a deferred module runs and prints its current graph:")
    gm = torch.fx.symbolic_trace(AddMul())
    other = torch.fx.symbolic_trace(AddMul())
    x = torch.arange(4.0)
    _defer_codegen(gm)
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is operator.mul:
            node.target = operator.sub
    gm.recompile()  # deferred: only marks the code stale
    check("the edit is visible to a call", torch.equal(gm(x), (x + 1.0) - 2.0))
    check("and to .code", "- 2.0" in gm.code and "* 2.0" not in gm.code)
    # A second edit through the graph setter, the path legalize_graph takes.
    new_graph = torch.fx.Graph()
    env = {}
    for node in gm.graph.nodes:
        env[node] = new_graph.node_copy(node, lambda n: env[n])
        if node.op == "call_function" and node.target is operator.add:
            env[node].target = operator.mul
    gm.graph = new_graph
    check("a reassigned graph is what runs", torch.equal(gm(x), (x * 1.0) - 2.0))
    check("other modules keep their eager codegen",
          "_esp32_codegen" not in vars(type(other))
          and torch.equal(other(x), (x + 1.0) * 2.0))


def only_the_fake_copy():
    from executorch.exir import to_edge
    from executorch.exir.program._fake_program import get_fake_program

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
        _is_to_backend_fake_copy,
    )

    print("only to_backend's fake copy qualifies:")
    x = torch.randn(2, 16)
    edge = to_edge(torch.export.export(MLP().eval(), (x,)))
    real = edge.exported_program()
    check("a real program does not", not _is_to_backend_fake_copy(real))
    check("its fake copy does", _is_to_backend_fake_copy(get_fake_program(real)))
    Esp32Partitioner().partition(real)
    check("partition() on a real program leaves its module eager",
          "_esp32_codegen" not in vars(type(real.graph_module)))
    lowered = edge.to_backend(Esp32Partitioner())
    gm = lowered.exported_program().graph_module
    delegates = sum(1 for n in gm.graph.nodes
                    if n.op == "call_function" and "call_delegate" in str(n.target))
    check("the lowered result is an ordinary GraphModule",
          "_esp32_codegen" not in vars(type(gm)) and "executorch_call_delegate" in gm.code,
          f"{delegates} delegate call(s)")


def main():
    deferred_module_semantics()
    only_the_fake_copy()
    if FAILS:
        print(f"\nFAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("\nall deferred-codegen checks passed")
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
