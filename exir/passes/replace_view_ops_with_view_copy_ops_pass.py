# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    get_view_copy_of_view_op,
)
from torch._ops import HigherOrderOperator
from torch.fx.passes.infra.pass_base import PassBase, PassResult


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
    torch.ops.aten.squeeze_.dim: torch.ops.aten.squeeze_copy.dim,
    exir_ops.edge.aten.squeeze_.dim: exir_ops.edge.aten.squeeze_copy.dim,
}
_PRESERVED_VIEW_OPS = {
    torch.ops.aten.contiguous.default,
}


class ReplaceViewOpsWithViewCopyOpsPass(PassBase):
    """
    Replace view ops with their view_copy variants using direct FX rewrites.

    The upstream torch pass does this through ExportPass retracing. ExecuTorch's
    edge conversion only needs a target substitution, so a direct graph walk is
    substantially cheaper while preserving the same operator mapping.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            module_modified = False
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue

                target = node.target
                if target in _PRESERVED_VIEW_OPS:
                    continue

                replacement = _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS.get(target)
                if replacement is None:
                    if isinstance(target, HigherOrderOperator):
                        continue
                    schema = getattr(target, "_schema", None)
                    if schema is None:
                        continue
                    replacement = get_view_copy_of_view_op(schema)

                if replacement is None or replacement == target:
                    continue

                node.target = replacement
                module_modified = True

            if module_modified:
                module.recompile()
                modified = True

        return PassResult(graph_module, modified)
