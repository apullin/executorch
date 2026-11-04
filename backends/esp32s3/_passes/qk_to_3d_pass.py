# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrite 4-D multi-head q@kT into the 3-D spelling the delegate's gate matches.

The opt-in int8 q@kT path (`OP_BMM_DYN_I8`) is gated on a structural pattern:
`bmm(A, permute_copy(X, (0,2,1)))` with 3-D operands, where the permute is
folded so the runtime reads X in its natural `[B,N,K]` layout. That is the shape
this repo's own attention produces, and nothing else's.

Standard multi-head attention keeps heads as a separate dimension --
`[B,H,T,Dh]` -- and transposes the key with `permute(0,1,3,2)`. Export then
collapses the batch and head dims for the batched matmul, so what reaches the
bmm is a `view_copy`, not a permute:

    bmm(view[8,40,36], view[8,36,40])      <- measured, a speech encoder
        where each view sources an expand_copy of a 4-D permute

Every one of that encoder's 12 bmms is refused as "bmm not q@kT" for that
reason alone, with `quantize_qk=True` set. The arithmetic is identical; the
spelling is not.

The rewrite moves the adjacent B/H collapse inside the transpose:

    view(permute(K[B,H,S,Dh], (0,1,3,2)), [B*H,Dh,S])
      ->  permute(view(K[B,H,S,Dh], [B*H,S,Dh]), (0,2,1))

Both produce `[B*H,Dh,S]` from the same bytes: B and H are adjacent axes that
the permutation leaves in place, so collapsing them commutes with swapping S
and Dh. This is a spelling change, not a numerics change, and the proof is blob
identity against a natively-3-D attention at B=2, H=2
(`test_qk_canonicalize.py`).

Same technique and same reasoning as `bmm_to_mm_pass.py`: canonicalise BEFORE
partitioning rather than teach the gate a second pattern. A gate that matches two
spellings has to keep both correct forever, and preprocess re-checks the
structure inside the partition subprogram where it must match a third time.

lower_esp32 runs it on every program, before partitioning (`canonical_passes`):
the rewrite is exact, and whether the q@kT it exposes delegates is still the
choice of `quantize_qk=True`, which is opt-in and off by default.
"""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult

BMM = exir_ops.edge.aten.bmm.default
VIEW = exir_ops.edge.aten.view_copy.default
EXPAND = exir_ops.edge.aten.expand_copy.default
PERMUTE = exir_ops.edge.aten.permute_copy.default
_CLONE_PACKET = getattr(exir_ops.edge.dim_order_ops, "_clone_dim_order", None)
CLONE_DIM_ORDER = getattr(_CLONE_PACKET, "default", _CLONE_PACKET)


def _val(n):
    return n.meta.get("val") if isinstance(n, torch.fx.Node) else None


def _dim_order(value):
    fn = getattr(value, "dim_order", None)
    return list(fn()) if callable(fn) else None


def _skip_identity_materializations(node, hops=4):
    """Step past copies that preserve shape, values, and dimension order.

    Multi-head matmul export inserts ``expand_copy`` and then
    ``_clone_dim_order`` before flattening B/H.  The clone is needed for the
    *original* transpose-then-view spelling, but becomes redundant after this
    pass moves the view before the transpose.  Refuse a clone that actually
    requests a different dimension order; bypassing that would be a layout
    change rather than canonicalisation.
    """
    for _ in range(hops):
        if not (isinstance(node, torch.fx.Node)
                and node.op == "call_function" and len(node.users) == 1):
            return node, None
        source = node.args[0] if node.args else None
        v_in, v_out = _val(source), _val(node)
        if node.target is EXPAND:
            if (v_in is None or v_out is None
                    or list(v_in.shape) != list(v_out.shape)):
                return node, "expand is not shape-preserving"
            node = source
            continue
        if CLONE_DIM_ORDER is not None and node.target is CLONE_DIM_ORDER:
            in_order, out_order = _dim_order(v_in), _dim_order(v_out)
            requested = node.kwargs.get("dim_order")
            requested = list(requested) if requested is not None else out_order
            if (v_in is None or v_out is None
                    or list(v_in.shape) != list(v_out.shape)
                    or in_order is None or in_order != out_order
                    or requested != out_order):
                return node, "clone changes shape or dimension order"
            node = source
            continue
        return node, None
    return node, "materialization chain exceeds four nodes"


class QkTo3D(ExportPass):
    """Rewrites the key operand of a 4-D q@kT into the 3-D permute spelling."""

    def __init__(self, expect=None):
        super().__init__()
        self.rewritten = 0
        self.skipped = []
        self.expect = expect

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        changed = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not BMM:
                continue
            k_side = node.args[1]
            if not (isinstance(k_side, torch.fx.Node)
                    and k_side.op == "call_function" and k_side.target is VIEW
                    and len(k_side.users) == 1):
                continue
            src, reason = _skip_identity_materializations(k_side.args[0])
            if reason is not None:
                self.skipped.append((node.name, reason))
                continue
            if not (isinstance(src, torch.fx.Node) and src.op == "call_function"
                    and src.target is PERMUTE):
                continue
            # The 4-D key transpose, and only that one. permute(0,1,3,2) on
            # [B,H,S,Dh] is the transpose; any other permutation means the axes
            # do not mean what this rewrite assumes.
            if list(src.args[1]) != [0, 1, 3, 2]:
                self.skipped.append((node.name, f"permute {list(src.args[1])} "
                                                f"is not the 4-D key transpose"))
                continue
            k4 = src.args[0]
            v4, vout = _val(k4), _val(k_side)
            if v4 is None or vout is None or len(v4.shape) != 4:
                continue
            B, H, S, Dh = (int(d) for d in v4.shape)
            BH = B * H
            if list(vout.shape) != [BH, Dh, S]:
                self.skipped.append((node.name, f"view target {list(vout.shape)} "
                                                f"is not [B*H,Dh,S]"))
                continue

            with graph.inserting_before(node):
                flat = graph.call_function(VIEW, (k4, [BH, S, Dh]))
                flat.meta = dict(k4.meta)
                flat.meta["val"] = v4.reshape(BH, S, Dh)

                perm = graph.call_function(PERMUTE, (flat, [0, 2, 1]))
                perm.meta = dict(k_side.meta)
                perm.meta["val"] = vout

            node.update_arg(1, perm)
            self.rewritten += 1
            changed = True

        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()
        if self.expect is not None and self.rewritten != self.expect:
            raise RuntimeError(
                f"QkTo3D rewrote {self.rewritten}, expected {self.expect}. "
                f"Skipped: {self.skipped}")
        return PassResult(graph_module, changed)
