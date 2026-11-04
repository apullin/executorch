# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrite decomposed scaled_dot_product_attention into the explicit q@kT spelling.

Hugging Face's default attention (``attn_implementation="sdpa"``), timm and
``nn.MultiheadAttention(need_weights=False)`` call
``F.scaled_dot_product_attention``. ExecuTorch lowers it through ATen's math
decomposition (``_scaled_dot_product_attention_math`` plus ``_safe_softmax``),
which is ordinary explicit attention in a different spelling:

    s = sqrt(scale)
    bmm(view(q4 * s), view(permute(k4, (0,1,3,2)) * s)) -> view [-> + mask]
        -> _softmax ----------------------------------------------.
        -> eq(-inf) -> logical_not -> any -> logical_not -> where(rows, 0, softmax)

Two things in that spelling keep its bmms from the opt-in int8 q@kT gate
(``quantize_qk=True``):

  1. the key's scale sits between the transpose and the head-collapsing view,
     so `QkTo3D` never sees ``view(permute(K))`` and the bmm's key operand is
     not the folded permute the gate matches;
  2. the NaN guard reads the scores a second time (``eq(scores, -inf)``), so
     the scores do not reach the softmax through single-use hops.

Left as it is, on 4-layer Hugging Face models, that is 18 delegate calls
instead of the eager spelling's 22, and about 35 more portable ops.

One rewrite, and it is exact:

  ``mul(permute(k, p), s) -> permute(mul(k, s), p)``. A scalar multiply
  commutes with a permutation element for element: the same products in the
  same bytes, so not one bit changes. (Explicit attention written as
  ``q @ (k.transpose(-1, -2) * s)`` gets the same rewrite.) The replacement is
  inserted at the multiply, where the key and the scale both exist.

The NaN guard ``where(all(t == -inf, -1), 0, softmax(t))`` always stays, even
where no row "could" be entirely -inf: no row is provably finite without
bounds on the scores. Finite q and k can overflow to -inf (q = 1e20,
k = -1e20), a literal-looking mask can be built from rand, and a softmax over
another axis needs the check along that axis. Keeping it
costs a few portable ops per attention; q@kT still delegates, because the
gate's softmax test steps over a guard (``_leads_to_softmax``).
"""

import math

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_base import PassResult

from executorch.backends.esp32s3._passes.qk_to_3d_pass import (
    _skip_identity_materializations,
)

BMM = exir_ops.edge.aten.bmm.default
VIEW = exir_ops.edge.aten.view_copy.default
PERMUTE = exir_ops.edge.aten.permute_copy.default
MUL_SCALAR = exir_ops.edge.aten.mul.Scalar
MUL_TENSOR = exir_ops.edge.aten.mul.Tensor
ADD = exir_ops.edge.aten.add.Tensor
SOFTMAX = exir_ops.edge.aten._softmax.default
EQ = exir_ops.edge.aten.eq.Scalar
NOT = exir_ops.edge.aten.logical_not.default
ANY = exir_ops.edge.aten.any.dim
FULL_LIKE = exir_ops.edge.aten.full_like.default
WHERE = exir_ops.edge.aten.where.self

# A mask subgraph this large is not a mask built from literals.


def _val(n):
    return n.meta.get("val") if isinstance(n, torch.fx.Node) else None


def _is(n, target, users=None):
    """`n` is a call to `target` (with exactly `users` as its users, if given)."""
    return (isinstance(n, torch.fx.Node) and n.op == "call_function"
            and n.target is target
            and (users is None or set(n.users) == set(users)))


def safe_softmax_guard(softmax):
    """The nodes of ATen's ``_safe_softmax`` NaN guard around `softmax`, or None.

    ``where(logical_not(any(logical_not(eq(t, -inf)), dim, True)),
            full_like(softmax, 0), softmax)`` with every link single-use.
    """
    if not _is(softmax, SOFTMAX) or len(softmax.args) < 2:
        return None
    t, dim = softmax.args[0], softmax.args[1]
    users = list(softmax.users)
    where = next((u for u in users if _is(u, WHERE)), None)
    zeros = next((u for u in users if _is(u, FULL_LIKE)), None)
    if len(users) != 2 or where is None or zeros is None:
        return None
    if not (len(where.args) == 3 and where.args[1] is zeros
            and where.args[2] is softmax and set(zeros.users) == {where}
            and zeros.args[1] == 0):
        return None
    rows = where.args[0]
    if not _is(rows, NOT, [where]):
        return None
    reduced = rows.args[0]
    if not (_is(reduced, ANY, [rows]) and len(reduced.args) == 3
            and reduced.args[2] is True):
        return None
    tv = _val(t)
    if tv is None or reduced.args[1] % tv.dim() != dim % tv.dim():
        return None
    open_ = reduced.args[0]
    if not _is(open_, NOT, [reduced]):
        return None
    eq = open_.args[0]
    if not (_is(eq, EQ, [open_]) and eq.args[0] is t
            and eq.args[1] == -math.inf):
        return None
    return {"where": where, "zeros": zeros, "rows": rows, "any": reduced,
            "open": open_, "eq": eq}


def _scaled_operand(mul):
    """The tensor `mul` multiplies by a single value, or None.

    ``mul.Scalar`` (the SDPA decomposition's spelling) or ``mul.Tensor`` whose
    other operand has one element and no more dims (a Python float in model
    code, lifted to a 0-d constant). Such a multiply is the same arithmetic on
    every element whatever their order, so it commutes with a permutation.
    """
    if _is(mul, MUL_SCALAR) and len(mul.args) == 2:
        return mul.args[0]
    if _is(mul, MUL_TENSOR) and len(mul.args) == 2 and not mul.kwargs:
        for t, factor in (mul.args, mul.args[::-1]):
            tv, fv = _val(t), _val(factor)
            if (tv is not None and fv is not None and fv.numel() == 1
                    and fv.dim() <= tv.dim()):
                return t
    return None


class SdpaToQk(ExportPass):
    """Moves the key scale above the transpose; keeps every NaN guard."""

    def __init__(self):
        super().__init__()
        self.swapped = 0        # key scales moved above the key transpose
        self.skipped = []       # (node, reason) for sites left as they were

    def _swap_key_scale(self, graph, bmm):
        view = bmm.args[1]
        if not _is(view, VIEW, [bmm]):
            return False
        # matmul's broadcast leaves an identity expand (and possibly a
        # layout-preserving clone) above the view; QkTo3D steps over the same.
        mul, reason = _skip_identity_materializations(view.args[0])
        if reason is not None or len(getattr(mul, "users", ())) != 1:
            return False
        perm = _scaled_operand(mul)
        if not _is(perm, PERMUTE, [mul]):
            return False
        if list(perm.args[1]) != [0, 1, 3, 2]:
            self.skipped.append((bmm.name, f"scaled permute {list(perm.args[1])} "
                                           "is not the 4-D key transpose"))
            return False
        k4 = perm.args[0]
        if _val(k4) is None or _val(mul) is None:
            return False
        with graph.inserting_before(mul):
            scaled = graph.call_function(
                mul.target, tuple(k4 if a is perm else a for a in mul.args),
                dict(mul.kwargs))
            scaled.meta = dict(mul.meta)
            scaled.meta["val"] = _val(mul).permute(0, 1, 3, 2).contiguous()
            moved = graph.call_function(PERMUTE, (scaled, [0, 1, 3, 2]))
            moved.meta = dict(perm.meta)
            moved.meta["val"] = _val(mul)
        mul.replace_all_uses_with(moved)
        self.swapped += 1
        return True

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        changed = False
        for node in list(graph.nodes):
            if _is(node, BMM):
                changed |= self._swap_key_scale(graph, node)
        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, changed)
