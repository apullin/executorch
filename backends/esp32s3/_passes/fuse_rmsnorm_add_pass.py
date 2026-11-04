# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse a post-norm residual add, residual + rmsnorm(y), into one Espressif op.

Post-norm transformers (Glimmer, Gemma 2/3) write every sublayer as

    x = x + post_norm(sublayer(...))

which after `fuse_rmsnorm` is an `espressif::rmsnorm` feeding one `aten::add`. The
add is a full portable kernel call for a [1, D] elementwise sum -- 24 of them per
story-Glimmer decode step, about 70 us each on the chip, almost all call overhead.
`espressif::rmsnorm_add(x, weight, eps, residual)` computes the same norm and then
adds the residual in the same expression order, so the result is bit-identical to
the two ops it replaces: the norm's float output, rounded, plus the residual, rounded
once.

Matching is structural, like fuse_rmsnorm: an fp32 `aten::add.Tensor` with alpha 1,
one operand a fused weighted `espressif::rmsnorm` whose only user is this add, the
other operand the same shape and dtype (no broadcasting). Anything else is left
alone. Run after `fuse_rmsnorm`.
"""

from typing import Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch.library import Library

from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import RMSNORM  # declares espressif::rmsnorm

lib = Library("espressif", "FRAGMENT")
lib.define("rmsnorm_add(Tensor x, Tensor weight, float eps, Tensor residual) -> Tensor")
lib.define("rmsnorm_add.out(Tensor x, Tensor weight, float eps, Tensor residual, *, "
           "Tensor(a!) out) -> Tensor(a!)")
impl = Library("espressif", "IMPL")


def _rmsnorm_add(x, weight, eps, residual):
    return torch.add(residual, torch.ops.espressif.rmsnorm(x, weight, eps))


def _rmsnorm_add_out(x, weight, eps, residual, *, out):
    out.copy_(_rmsnorm_add(x, weight, eps, residual))
    return out


impl.impl("rmsnorm_add", _rmsnorm_add, "CompositeExplicitAutograd")
impl.impl("rmsnorm_add.out", _rmsnorm_add_out, "CompositeExplicitAutograd")

RMSNORM_ADD = torch.ops.espressif.rmsnorm_add.default
_ADD = exir_ops.edge.aten.add.Tensor


def _fp32_val(node):
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return val if val is not None and val.dtype == torch.float32 else None


def fuse_rmsnorm_add(edge_manager, method_name: str = "forward", expect: Optional[int] = None):
    """Rewrite residual + rmsnorm(y) in place; returns the number of sites fused."""
    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    fused = 0
    for add in list(graph.nodes):
        if add.op != "call_function" or add.target is not _ADD:
            continue
        if add.kwargs.get("alpha", 1) != 1 or len(add.args) != 2:
            continue
        a, b = add.args
        for norm, residual in ((b, a), (a, b)):
            # residual is norm: h + h reads the norm twice, so it is not the
            # norm's only USE even though it is its only user node.
            if not (isinstance(norm, torch.fx.Node) and norm.op == "call_function"
                    and norm.target is RMSNORM and len(norm.users) == 1
                    and isinstance(residual, torch.fx.Node)
                    and residual is not norm):
                continue
            nv, rv, ov = _fp32_val(norm), _fp32_val(residual), _fp32_val(add)
            if nv is None or rv is None or ov is None:
                continue
            if tuple(nv.shape) != tuple(rv.shape) or tuple(ov.shape) != tuple(nv.shape):
                continue
            x, weight, eps = norm.args
            with graph.inserting_before(add):
                node = graph.call_function(RMSNORM_ADD, (x, weight, eps, residual))
            node.meta = dict(add.meta)
            add.replace_all_uses_with(node)
            graph.erase_node(add)
            graph.erase_node(norm)
            fused += 1
            break
    if fused:
        graph.lint()
        ep.graph_module.recompile()
    if expect is not None and fused != expect:
        raise RuntimeError(f"rmsnorm_add fusion expected {expect} sites, fused {fused}")
    return fused
