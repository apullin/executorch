# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class RemovePermutesAroundElementwiseTosaOps(RemovePermutesAroundElementwiseOps):
    # TABLE looks each input element up in its rank-1 table, so it never
    # broadcasts. Listing it keeps the base pass's mixed-rank check from
    # counting the table, which permute_subgraph drops from the constant edges.
    _NON_BROADCASTING_OPS = (
        *RemovePermutesAroundElementwiseOps._NON_BROADCASTING_OPS,
        exir_ops.backend.tosa.TABLE.default,
    )

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__(
            extra_permutable_ops={
                *TableOps.unary_table_ops.keys(),
                *TableOps.special_table_ops,
                exir_ops.backend.tosa.RESCALE.default,
                exir_ops.backend.tosa.TABLE.default,
            }
        )
        self.exported_program = exported_program
        self._constant_input_names: set[str] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # The ARM pass manager updates exported_program before each invocation.
        gs = self.exported_program.graph_signature
        self._constant_input_names = (
            set(gs.inputs_to_parameters)
            | set(gs.inputs_to_buffers)
            | set(gs.inputs_to_lifted_tensor_constants)
        )
        return super().call(graph_module)

    def _is_constant(self, node: torch.fx.Node) -> bool:
        # get_attr nodes are handled by super()._is_constant; set membership
        # here is equivalent to is_param_node for placeholder inputs.
        return super()._is_constant(node) or node.name in self._constant_input_names

    def permute_subgraph(self, subgraph) -> bool:
        # A TABLE's lookup table is indexed by value, so it keeps its layout.
        new_constant_edges_in = set()
        for const_node, user_node in subgraph.constant_edges_in:
            if (
                user_node.target == exir_ops.backend.tosa.TABLE.default
                and const_node is user_node.args[1]
            ):
                continue
            new_constant_edges_in.add((const_node, user_node))

        subgraph.constant_edges_in = new_constant_edges_in
        return super().permute_subgraph(subgraph)
