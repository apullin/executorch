# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delete `expand_copy` nodes that broadcast nothing.

`expand_copy(x, shape)` with `shape == x.shape` broadcasts no dimension. It is
not free, though, because it is a `_copy` variant: it materialises a whole new
tensor holding exactly the bytes it was given. On a speech encoder every
one of the 24 expand_copy nodes is this case, and together they copy **8.9 MB**
per forward for no effect; on a small segmentation model, 46 of 49.

They are not a modelling mistake. They come from broadcasting a 2-D weight or a
head-split activation up to a batched matmul's operand rank, and when the batch
is 1 the broadcast is the identity. Nothing upstream removes them: ET's
`RemoveNoopPass` covers `to.dtype`, `dropout` and `slice_copy` only, and runs on
the aten dialect rather than the edge one. `remove_view_copy` does dissolve
`view_copy` into memory-planning aliases, which is why views cost nothing at
runtime and expands do -- `aten::expand_copy.out` is in the emitted operator
table, so these really execute.

SAFETY, since deleting a copy is exactly the kind of change that is right until
it is not. An identity `expand_copy` differs from its input in one respect: its
output is contiguous, and its input might not have been. So removal is only
sound when the input is contiguous by construction, which is checked rather than
assumed -- the input must be a placeholder or the output of another `_copy` op
(all of which materialise contiguously). Anything else is left alone and
recorded in `skipped`.
"""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult

EXPAND = exir_ops.edge.aten.expand_copy.default

# Stated as a DENYLIST: an allowlist of reshape-family producers would skip
# every expand fed by `_softmax` -- which is as contiguous as any of them. The
# edge dialect is functional and its operators write freshly-allocated tensors;
# the `_copy` suffixes exist precisely to guarantee that no operator returns an
# alias. So the question is
# not "which ops produce contiguous output" (all of them) but "which ops
# deliberately produce a DIFFERENT dim order", and that is a short list.
#
# `to_dim_order_copy` is the one that matters, and it is not hypothetical: it is
# the op a channels-last conversion would use, which is exactly the layout
# machinery a conv delegate would introduce. Dropping an expand fed by one would
# hand its consumer a tensor whose strides are not what the expand promised.
DIM_ORDER_CHANGING = tuple(
    op for op in (
        getattr(exir_ops.edge.dim_order_ops, "_to_dim_order_copy", None),
        getattr(exir_ops.edge.dim_order_ops, "_clone_dim_order", None),
    ) if op is not None
)
DIM_ORDER_TARGETS = tuple(
    getattr(op, "default", op) for op in DIM_ORDER_CHANGING
)


def _val(n):
    return n.meta.get("val") if isinstance(n, torch.fx.Node) else None


def _contiguous_by_construction(node):
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "placeholder":
        return True
    if node.op != "call_function":
        return False
    return node.target not in DIM_ORDER_TARGETS


class RemoveIdentityExpand(ExportPass):
    """`removed` counts deletions; `skipped` records every decline with a reason."""

    def __init__(self, expect=None):
        super().__init__()
        self.removed = 0
        self.bytes_saved = 0
        self.skipped = []
        self.expect = expect

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        changed = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not EXPAND:
                continue
            src = node.args[0]
            vi, vo = _val(src), _val(node)
            if vi is None or vo is None:
                continue
            if list(vi.shape) != list(vo.shape):
                continue        # a real broadcast; leave it alone silently
            if not _contiguous_by_construction(src):
                self.skipped.append(
                    (node.name, f"input {getattr(src, 'name', src)} is not "
                                f"contiguous by construction"))
                continue
            self.bytes_saved += vo.numel() * vo.element_size()
            node.replace_all_uses_with(src)
            graph.erase_node(node)
            self.removed += 1
            changed = True

        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()
        if self.expect is not None and self.removed != self.expect:
            raise RuntimeError(
                f"RemoveIdentityExpand removed {self.removed}, expected "
                f"{self.expect}. Skipped: {self.skipped}")
        return PassResult(graph_module, changed)
