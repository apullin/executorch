# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deleting an identity expand_copy must change nothing but the byte count.

`expand_copy(x, shape)` with `shape == x.shape` broadcasts no dimension, but it
is a `_copy` variant, so it materialises a full tensor holding exactly the bytes
it was handed. `aten::expand_copy.out` IS in the emitted operator table -- unlike
`view_copy`, which ET dissolves into memory-planning aliases -- so these really
execute on device.

Deleting a copy is the kind of change that is right until it is not, so the
claims are:

  1. outputs are BIT-IDENTICAL before and after, on batch-1 attention scores
     (torch.matmul broadcasts both operands to the batch shape). Not close:
     the pass either removed a no-op or it did not, and "nearly the same" would
     mean it removed something that mattered.
  2. a REAL broadcast is never touched,
  3. an expand fed by a dim-order-changing op is refused with a reason -- that
     is the one case where the input's strides are genuinely not what the
     expand's output promised,
  4. lowered through the backend, every identity expand is gone and the op
     leaves the runtime operator table entirely.

    python test_remove_identity_expand.py
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


def main():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.remove_identity_expand_pass import (
        RemoveIdentityExpand,
    )

    print("outputs are bit-identical where the pass fires:")

    class Scores(nn.Module):
        def forward(self, q, k):
            # Written for any batch; at batch 1 every expand is the identity.
            q = q.expand(1, 2, 8, 4)
            return torch.matmul(q, k.expand(1, 2, 8, 4).transpose(-1, -2))

    torch.manual_seed(0)
    qk = (torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))
    before = to_edge(torch.export.export(Scores().eval(), qk))
    with torch.no_grad():
        y_before = before.exported_program().module()(*qk)

    p = RemoveIdentityExpand()
    after = to_edge(torch.export.export(Scores().eval(), qk)).transform([p])
    with torch.no_grad():
        y_after = after.exported_program().module()(*qk)

    check("the pass actually fired", p.removed > 0,
          f"{p.removed} removed, {p.bytes_saved:,} B")
    check("no expand_copy survives", not [
        n for n in after.exported_program().graph_module.graph.nodes
        if n.op == "call_function" and "expand_copy" in str(n.target)])
    check("outputs bit-identical", torch.equal(y_before, y_after),
          f"max abs {(y_before - y_after).abs().max().item():.2e}")
    check("nothing was skipped without a reason",
          all(len(r) == 2 and r[1] for r in p.skipped))

    print("a REAL broadcast is left alone:")

    class Broadcast(nn.Module):
        def forward(self, a, b):
            # b is [1,1,4]; reaching [2,3,4] is a genuine broadcast.
            return a + b.expand(2, 3, 4)

    p2 = RemoveIdentityExpand()
    to_edge(torch.export.export(Broadcast().eval(),
                                (torch.randn(2, 3, 4), torch.randn(1, 1, 4)))
            ).transform([p2])
    check("a widening expand is not removed", p2.removed == 0,
          f"{p2.removed} removed")

    print("an expand fed by a dim-order change is refused, with a reason:")

    class ChannelsLast(nn.Module):
        def forward(self, x):
            return x.to(memory_format=torch.channels_last).expand(1, 3, 4, 4) * 2.0

    p3 = RemoveIdentityExpand()
    to_edge(torch.export.export(ChannelsLast().eval(), (torch.randn(1, 3, 4, 4),))
            ).transform([p3])
    check("the expand is kept", p3.removed == 0, f"{p3.removed} removed")
    check("and the reason is recorded",
          len(p3.skipped) == 1 and "not contiguous" in p3.skipped[0][1],
          str(p3.skipped))

    print("lowered, the op leaves the runtime operator table:")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3 import Esp32Partitioner
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    p4 = RemoveIdentityExpand()
    lowered = to_edge_transform_and_lower(
        torch.export.export(Scores().eval(), qk), transform_passes=[p4],
        partitioner=[Esp32Partitioner()])
    table = ops_of_program(lowered.to_executorch())
    check("identity expands removed", p4.removed > 0, f"{p4.removed} ops")
    check("expand_copy gone from the runtime table",
          not [o for o in table if "expand_copy" in o], str(sorted(table)))

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("identity expands are deletable, and deleted")
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
