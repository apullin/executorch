# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Union

import torch
from executorch.exir.pass_base import (
    ExportPass,
    map_args,
    NodeMetadata,
    PassResult,
    ProxyValue,
)
from torch import SymBool, SymFloat, SymInt
from torch.utils._pytree import PyTree, tree_flatten


_SYM_TYPES = {SymInt, SymFloat, SymBool}


class SymToTensorPass(ExportPass):
    """
    The dispatcher implicitly converts SymInt/SymFloats to tensors, but
    sometimes this doesn't comply with the operator's schema which ExecuTorch
    heavily relies on. So this pass inserts a
    torch.ops.aten.scalar_tensor.default operator before these SymInts are used
    so that it matches the schema of the operator.
    """

    @staticmethod
    def _contains_symbolic_value(value: object) -> bool:
        if type(value) in _SYM_TYPES:
            return True
        if isinstance(value, ProxyValue):
            return type(value.data) in _SYM_TYPES
        if isinstance(value, torch.fx.Node):
            return SymToTensorPass._contains_symbolic_value(value.meta.get("val"))

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return any(type(dim) in _SYM_TYPES for dim in shape)
            except TypeError:
                return False
        return False

    @staticmethod
    def _has_symbolic_candidate(graph_module: torch.fx.GraphModule) -> bool:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                leaves, _ = tree_flatten((node.args, node.kwargs, node.meta.get("val")))
                if any(
                    SymToTensorPass._contains_symbolic_value(leaf) for leaf in leaves
                ):
                    return True
        return False

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta: NodeMetadata):
        # pyre-ignore
        def is_sym(value, arg) -> bool:
            if isinstance(value, ProxyValue) and not value.is_tensor():
                if isinstance(arg.type, torch.TensorType) and type(value.data) in {
                    SymInt,
                    SymFloat,
                    SymBool,
                }:
                    return True
            return False

        def corresponding_dtype(
            symbol: Union[SymInt, SymFloat, SymBool]
        ) -> torch.dtype:
            if isinstance(symbol, SymInt):
                return torch.int32
            elif isinstance(symbol, SymFloat):
                return torch.float32
            elif isinstance(symbol, SymBool):
                return torch.bool
            else:
                raise AssertionError(f"Unsupported data type: {type(symbol)}")

        def try_coerce(value: PyTree, arg: torch.Argument) -> PyTree:
            if is_sym(value, arg):
                return self.call_operator(
                    torch.ops.aten.scalar_tensor.default,
                    (value,),
                    {"dtype": corresponding_dtype(value.data)},
                    meta,
                )
            else:
                return value

        args, kwargs = map_args(op, try_coerce, args, kwargs)

        return super().call_operator(op, args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if not self._has_symbolic_candidate(graph_module):
            return PassResult(graph_module, False)
        return super().call(graph_module)
