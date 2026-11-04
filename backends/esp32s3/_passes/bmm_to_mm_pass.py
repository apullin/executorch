# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Canonicalise a batch-1 `bmm` against a constant weight into the `mm` spelling.

`bmm(x[1,M,K], W[1,K,N])` computes exactly what `mm(x[M,K], W[K,N])` computes:
same arithmetic, same weights, same output bytes. This backend claims `mm` and
not `bmm`, so the identical model delegates or does not depending on which one
torch happened to emit -- and that is not the model author's choice. Measured on
a speech encoder: from identical `nn.Linear` calls, `q/k/v` and `fc1` came
out `bmm` while `o_proj` and `fc2` came out `mm`. 24 of 36 projections lost.

WHY A PASS, and not a partitioner rule. Claiming `bmm` directly takes a gate
rule for the op, a second gate rule for the weight's `view -> expand -> permute`
chain, `_resolve_const` replaying reshapes, and a batch-stripping branch in
preprocess -- and still fails: `generate_pattern_op_partitions` puts the weight
chain in a DIFFERENT partition from the bmm it feeds, producing a partition
holding nothing but reshapes. The chain is three hops long, where the `mm`
path's folded permute is adjacent to its matmul, and nothing makes the grouper
keep a longer chain together.

Canonicalising before partitioning avoids the whole question. The graph reaching
the partitioner contains only the spelling the backend implements, packs, tests
and has device parity for. The pass needs no opcode, descriptor field or kernel
of its own and leaves the numerics path alone -- which is why the test for this
is blob identity against the `mm` spelling rather than a parity row.

Run it BEFORE lowering, which `to_edge_transform_and_lower` supports directly:

    to_edge_transform_and_lower(ep, transform_passes=[BmmToMm()],
                                partitioner=[Esp32Partitioner()])

A batch of B > 1 is the same Linear when every batch reads ONE matrix: torch's
matmul spells an nn.Linear over an input it cannot view as rows (a
channels-last image, a transposed attention output) as `expand -> view -> bmm`
against the weight broadcast over the batch, `bmm(x[B,M,K], expand(W[K,N]))`,
and that is `mm(x[B*M,K], W)` viewed back to [B,M,N]: each output row is still
its own row times W. Left alone, the same nn.Linear would delegate over a
row-major input and run portable over a channels-last one.
The weight chain must keep [K,N] as its last two dims at every step, so that
only whole copies of the matrix are broadcast or regrouped.

Scope, stated so the silence is not mistaken for coverage: the weight side must
reach a placeholder through reshapes alone, and past batch 1 it must be one
matrix broadcast over the batch. A real batched matmul (attention's q@kT, batch
= heads) is untouched -- it is not a Linear and has no constant operand.
"""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult

BMM = exir_ops.edge.aten.bmm.default
VIEW = exir_ops.edge.aten.view_copy.default
EXPAND = exir_ops.edge.aten.expand_copy.default
PERMUTE = exir_ops.edge.aten.permute_copy.default
UNSQUEEZE = exir_ops.edge.aten.unsqueeze_copy.default
MM = exir_ops.edge.aten.mm.default
ADDMM = exir_ops.edge.aten.addmm.default
ADD = exir_ops.edge.aten.add.Tensor
# unsqueeze_copy belongs here for the same reason expand_copy does: a 2-D weight
# reaching a batched matmul is broadcast, and `w.unsqueeze(0).expand(...)` is the
# other spelling of the same broadcast.
RESHAPES = (VIEW, EXPAND, PERMUTE, UNSQUEEZE)


def _val(node):
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else None


def _two_d_source(node, want, hops=6):
    """Walk the weight chain back to a node already holding `want` = [K,N].

    Returning an EXISTING node rather than inserting a reshape is the point: the
    node found this way is the `permute_copy` that `nn.Linear` emits, which the
    partitioner already folds (it is in `FOLDED`) and `_resolve_const` already
    replays. Inserting a fresh `view_copy` instead would hand the matmul a weight
    operand of a kind the constant folder does not accept, and the delegate would
    refuse the very op this pass exists to hand it.

    The walk passes only through ORDER-PRESERVING steps: view, unsqueeze and an
    expand that broadcasts nothing (numel is K*N at both ends, and expand never
    shrinks it). A permute may be the node returned, never a step walked
    through. Walking through one would match a square weight by shape alone:
    `bmm(x, W.unsqueeze(0).transpose(1, 2))` has W itself at [K,K], so the
    rewrite would compute x @ W where the model computes x @ W.T.
    """
    for _ in range(hops):
        v = _val(node)
        if v is None:
            return None
        if list(v.shape) == list(want):
            return node
        if not (node.op == "call_function"
                and node.target in (VIEW, EXPAND, UNSQUEEZE)):
            return None
        node = node.args[0]
    return None


def _broadcast_source(node, want, hops=6):
    """For a batch past 1: the node holding `want` = [K,N] whose copies, and
    only whole copies, make up every batch of the weight chain ending at
    `node`, else None.

    Each step walked is a view, an expand or an unsqueeze whose output still
    ends in [K,N]: then the step changes only the leading dims, so it regroups
    or broadcasts whole [K,N] blocks of a row-major tensor and never mixes rows
    of the matrix (an unsqueeze between K and N, or a view through [K*N], is
    refused). The node returned holds exactly K*N elements, so every block is
    that one matrix. As for batch 1, a permute is never walked through."""
    for _ in range(hops):
        v = _val(node)
        if v is None or list(v.shape[-2:]) != list(want):
            return None
        if list(v.shape) == list(want):
            return node
        if not (node.op == "call_function"
                and node.target in (VIEW, EXPAND, UNSQUEEZE)):
            return None
        node = node.args[0]
    return None


def _linear_bias(node, n):
    """torch's bias add after the batched bmm it spells an nn.Linear as,
    add(view(bmm), bias[N]) with alpha 1 (either operand order), the view
    keeping N last and each link single-use: (view, add, bias), else None.
    Folded into an addmm, the bias rides in the delegate's epilogue, as for
    the row-major input's addmm, and the add leaves the graph."""
    users = list(node.users)
    if len(users) != 1 or users[0].op != "call_function" or users[0].target is not VIEW:
        return None
    view = users[0]
    vv = _val(view)
    if vv is None or len(view.users) != 1 or not len(vv.shape) or int(vv.shape[-1]) != n:
        return None
    add = next(iter(view.users))
    if (add.op != "call_function" or add.target is not ADD or len(add.args) != 2
            or add.kwargs.get("alpha", 1) != 1):
        return None
    bias = (add.args[1] if add.args[0] is view
            else add.args[0] if add.args[1] is view else None)
    bv, sv = _val(bias), _val(add)
    if (bv is None or sv is None or list(bv.shape) != [n] or bv.dtype != vv.dtype
            or list(sv.shape) != list(vv.shape) or not _traces_to_placeholder(bias)):
        return None
    return view, add, bias


def _traces_to_placeholder(node, hops=6):
    for _ in range(hops):
        if not isinstance(node, torch.fx.Node):
            return False
        if node.op == "placeholder":
            return True
        if not (node.op == "call_function" and node.target in RESHAPES):
            return False
        node = node.args[0]
    return False


class BmmToMm(ExportPass):
    """Rewrites qualifying bmm nodes in place. `rewritten` counts them."""

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
            a, b = node.args[0], node.args[1]
            av, bv, ov = _val(a), _val(b), _val(node)
            if av is None or bv is None or ov is None:
                continue
            if len(av.shape) != 3 or len(bv.shape) != 3:
                continue
            batch = int(av.shape[0])
            M, K = int(av.shape[1]), int(av.shape[2])
            N = int(bv.shape[2])
            if batch != 1 or int(bv.shape[0]) != 1:
                # A genuine batched matmul -- attention's q@kT lands here
                # (batch = heads) and is deliberately left alone -- unless its
                # weight is one constant matrix broadcast over the batch: an
                # nn.Linear over an input torch could not view as rows.
                w2d = (_broadcast_source(b, [K, N])
                       if (int(bv.shape[0]) == batch
                           and _traces_to_placeholder(b)) else None)
                if w2d is None:
                    self.skipped.append((node.name, "batch != 1 and the weight "
                                                    "is not one constant matrix "
                                                    "broadcast over it"))
                    continue
            elif not _traces_to_placeholder(b):
                # Both operands are activations: not a Linear.
                self.skipped.append((node.name, "weight side is not constant"))
                continue
            else:
                w2d = _two_d_source(b, [K, N])
                if w2d is None:
                    self.skipped.append((node.name, f"no [{K},{N}] node in the "
                                                    f"weight chain"))
                    continue

            # Past batch 1 only, so that the batch-1 rewrite's programs stay as
            # they were: torch's bias add folds into an addmm, and a bmm read
            # only through views gives them the matmul's [B*M, N] directly, as
            # torch's own Linear spelling does (a view of the [B, M, N] view
            # would hide the matmul from the epilogue gates).
            fold = _linear_bias(node, N) if batch != 1 else None
            direct = batch != 1 and all(
                u.op == "call_function" and u.target is VIEW for u in node.users)
            with graph.inserting_before(node):
                a2 = graph.call_function(VIEW, (a, [batch * M, K]))
                a2.meta = dict(a.meta)
                a2.meta["val"] = av.reshape(batch * M, K)

                mm = graph.call_function(
                    MM if fold is None else ADDMM,
                    (a2, w2d) if fold is None else (fold[2], a2, w2d))
                mm.meta = dict(node.meta)
                mm.meta["val"] = ov.reshape(batch * M, N)

                if not direct:
                    out = graph.call_function(VIEW, (mm, [batch, M, N]))
                    out.meta = dict(node.meta)
                    out.meta["val"] = ov

            if direct:
                for user in list(node.users):
                    user.replace_input_with(node, mm)
            else:
                node.replace_all_uses_with(out)
            graph.erase_node(node)
            if fold is not None:
                view, add, _ = fold
                add.replace_all_uses_with(view)
                graph.erase_node(add)
            self.rewritten += 1
            changed = True

        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()
        if self.expect is not None and self.rewritten != self.expect:
            raise RuntimeError(
                f"BmmToMm rewrote {self.rewritten} bmm node(s), expected "
                f"{self.expect}. Skipped: {self.skipped}"
            )
        return PassResult(graph_module, changed)
