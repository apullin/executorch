# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult


class NormalizeTransposePass(ExportPass):
    """
    Even with functionalization on, we still get graph with
    torch.ops.aten.t.default op. Ideally we should fix functionalization.
    TODO: once we have that, we should remove this pass.
    Check test_normalize_transpose_op in test_passes.py for more details
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if not any(
            node.op == "call_function" and node.target == torch.ops.aten.t.default
            for module in graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
        ):
            return PassResult(graph_module, False)
        return super().call(graph_module)

    def call_operator(self, op, args, kwargs, meta):
        if op == torch.ops.aten.t.default:
            return super().call_operator(
                torch.ops.aten.t_copy.default, (args[0],), kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)
