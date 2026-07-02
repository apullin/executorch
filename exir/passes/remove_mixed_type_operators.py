# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Iterable, Optional

import torch
from executorch.exir.pass_base import ExportPass, map_args, NodeMetadata, ProxyValue
from torch.fx.passes.infra.pass_base import PassResult
from torch import SymBool, SymFloat, SymInt
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils._pytree import PyTree


_PROMOTION_TYPE_ALLOW_LIST = {
    torch.ops.aten.add.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.mul.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.sub.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    # The correct promotion for div depends on the mode. If there is no mode,
    # it is INT_TO_FLOAT; otherwise it is default.
    torch.ops.aten.div.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    torch.ops.aten.div.Tensor_mode: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.minimum.default: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
}


class RemoveMixedTypeOperators(ExportPass):
    def _arg_value(self, arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            return arg.meta.get("val")
        return arg

    def _is_tensor_like(self, value: Any) -> bool:
        return isinstance(value, torch.Tensor)

    def _candidate_values(self, args: Iterable[Any]) -> Optional[tuple[Any, ...]]:
        values = tuple(self._arg_value(arg) for arg in args)
        if any(value is None for value in values):
            return None
        return values

    def _node_requires_mixed_type_rewrite(self, node: torch.fx.Node) -> bool:
        promotion_kind = _PROMOTION_TYPE_ALLOW_LIST.get(node.target)
        if promotion_kind is None:
            return False
        if (
            node.target == torch.ops.aten.div.Tensor_mode
            and node.kwargs.get("rounding_mode") is None
        ):
            promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT

        values = self._candidate_values(node.args)
        if values is None:
            return True

        try:
            promoted_dtype = elementwise_dtypes(
                *values,
                type_promotion_kind=promotion_kind,
            )[1]
        except Exception:
            return True

        return any(
            self._is_tensor_like(value) and value.dtype != promoted_dtype
            for value in values
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if not any(
            node.op == "call_function" and self._node_requires_mixed_type_rewrite(node)
            for module in graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
        ):
            return PassResult(graph_module, False)
        return super().call(graph_module)

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta: NodeMetadata):  # noqa: C901
        if len(args) <= 1:
            # Unary Operators are not mixed type
            return super().call_operator(op, args, kwargs, meta)

        if op in _PROMOTION_TYPE_ALLOW_LIST:
            promotion_kind = _PROMOTION_TYPE_ALLOW_LIST[op]
            if (
                op == torch.ops.aten.div.Tensor_mode
                and kwargs.get("rounding_mode") is None
            ):
                promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        else:
            # Not in allow list, do nothing
            return super().call_operator(op, args, kwargs, meta)

        # Using tensors for type information only
        arg_tensor = []
        for arg in args:
            if isinstance(arg, ProxyValue) and arg.is_tensor():
                arg_tensor.append(arg.to_tensor())
            elif isinstance(arg, ProxyValue) and isinstance(
                arg.data,
                (
                    SymFloat,
                    SymInt,
                    SymBool,
                ),
            ):
                arg_tensor.append(torch.tensor(arg.data))
            # Note: this case can happen after scarlar_to_tensor pass
            # where we convert a scalar to a tensor.
            elif isinstance(arg, torch.Tensor):
                arg_tensor.append(arg)
            else:
                arg_tensor.append(arg.data)
        arg_tensor = tuple(arg_tensor)

        # Computes type for computation
        promote_dtype: torch.dtype = elementwise_dtypes(
            *arg_tensor,
            type_promotion_kind=promotion_kind,
        )[1]

        def try_coerce(value: PyTree, arg: torch.Argument) -> PyTree:
            if not isinstance(arg.type, torch.TensorType):
                return value

            if isinstance(value, ProxyValue):
                if not value.is_tensor():
                    return value
                if value.to_tensor().dtype == promote_dtype:
                    return value

            if isinstance(value, torch.Tensor) and value.dtype == promote_dtype:
                return value

            return self.call_operator(
                torch.ops.aten._to_copy.default,
                (value,),
                {"dtype": promote_dtype},
                meta,
            )

        args, kwargs = map_args(op, try_coerce, args, kwargs)

        return super().call_operator(op, args, kwargs, meta)
