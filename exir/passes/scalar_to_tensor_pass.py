# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, map_args, PassResult


class ScalarToTensorPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()
        self._modified: bool = False

    @staticmethod
    def _has_scalar_tensor_candidate(graph_module: torch.fx.GraphModule) -> bool:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function" or not hasattr(node.target, "_schema"):
                    continue

                found = False

                # pyre-ignore
                def try_find(value, arg):
                    nonlocal found
                    if isinstance(value, (float, int, bool)) and isinstance(
                        arg.type, torch.TensorType
                    ):
                        found = True
                    return value

                try:
                    map_args(node.target, try_find, node.args, dict(node.kwargs))
                except Exception:
                    return True
                if found:
                    return True
        return False

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        # pyre-ignore
        def try_coerce(value, arg):
            # Note: we want to create tensor constants instead of
            # FakeTensor or ProxyTensor. If python_dispatcher is enabled,
            # the fake_tensor_mode of inputs will be used so that we won't
            # get a constant tensor with torch.tensor() call but instead
            # a fake tensor is created.
            with torch.utils._python_dispatch._disable_current_modes():
                if isinstance(value, (float, int, bool)) and isinstance(
                    arg.type, torch.TensorType
                ):
                    self._modified = True
                    return torch.tensor(value)

                return value

        args, kwargs = map_args(op, try_coerce, args, kwargs)
        return super().call_operator(op, args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if not self._has_scalar_tensor_candidate(graph_module):
            return PassResult(graph_module, False)

        self._modified = False
        graph_module = super().call(graph_module).graph_module
        modified = self._modified
        self._modified = False
        return PassResult(graph_module, modified)
