# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrite clamp spellings to one canonical clamp."""

from executorch.backends.esp32s3.operator_support.patterns import (
    _clamp_bounds,
    compose_clamp,
)


def canonicalize_clamps(edge_manager, method_name: str = "forward") -> int:
    """Collapse chains of clamp-family ops into one clamp (or nothing). Returns the
    number of ops removed.

    The chains are almost never written on purpose -- they arrive by composition:
    relu6 decomposing to hardtanh after an explicit relu, a residual branch clamped
    on both sides of a merge, a defensive clamp around a block whose last layer
    already ends in one. Each link is a whole kernel dispatch and a full activation
    round-trip through memory for an op that does nothing the neighbouring clamp
    was not already doing.

    Only CHAINS are touched (a clamp whose input is another clamp-family op with no
    other consumer). A lone relu stays a relu -- rewriting it to clamp would churn
    kernels for zero ops saved. A fully open composed interval deletes the pair
    outright.
    """
    import math

    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    removed = 0
    changed = True
    while changed:
        changed = False
        for outer in list(graph.nodes):
            if outer.op != "call_function":
                continue
            ob = _clamp_bounds(outer)
            if ob is None:
                continue
            inner = outer.args[0]
            if not isinstance(inner, torch.fx.Node) or inner.op != "call_function":
                continue
            ib = _clamp_bounds(inner)
            if ib is None or len(inner.users) != 1:
                continue
            dtype = getattr(inner.args[0].meta.get("val"), "dtype", None) \
                if isinstance(inner.args[0], torch.fx.Node) else None
            if dtype == torch.int8:
                continue    # a clamp of codes (stage_b.py): integer bounds

            lo, hi = compose_clamp(ob, ib)
            x = inner.args[0]
            # Open means -inf below and +inf ABOVE: max=-inf is a bound that
            # pins everything to -inf, not an absent one.
            if lo == -math.inf and hi == math.inf:
                outer.replace_all_uses_with(x)     # fully open: both were no-ops
            else:
                with graph.inserting_before(outer):
                    new = graph.call_function(
                        exir_ops.edge.aten.clamp.default,
                        (x,
                         None if lo == -math.inf else lo,
                         None if hi == math.inf else hi))
                new.meta = dict(outer.meta)
                outer.replace_all_uses_with(new)
            graph.erase_node(outer)
            graph.erase_node(inner)
            removed += 1
            changed = True
            break  # node list is stale; rescan
    if removed:
        graph.eliminate_dead_code()
        graph.lint()
        ep.graph_module.recompile()
    return removed
