# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_cf_extension,
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import (
    map_dtype,
    TOSA_CONTROL_FLOW_REGION_NAME_META,
    TOSA_TENSOR_NAME_META,
    TosaArg,
)
from executorch.backends.arm.tosa.utils import normalize_symint

from torch.fx import Node


@register_node_visitor
class WhileLoopVisitor(NodeVisitor):
    target = "while_loop"

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        validate_num_inputs(self.target, inputs, 4)
        validate_cf_extension(self.target, self.tosa_spec)

        carried_inputs = inputs[2].special if hasattr(inputs[2], "special") else None
        if carried_inputs is None:
            raise ValueError(f"{self.target}: Expected loop input arguments to be set.")

        additional_inputs = inputs[3].special if hasattr(inputs[3], "special") else None
        if additional_inputs:
            raise ValueError(
                "Additional inputs is not supported, use carried inputs instead."
            )

        attr = ts.TosaSerializerAttribute()
        cond_graph, body_graph = (
            cast(Node, arg).meta.get(
                TOSA_CONTROL_FLOW_REGION_NAME_META, str(cast(Node, arg).target)
            )
            for arg in node.args[:2]
        )
        attr.WhileLoopAttribute(cond_graph, body_graph)

        input_names: list[str] = []
        for loop_input in carried_inputs:
            if not isinstance(loop_input, Node):
                raise ValueError(
                    f"{self.target}: Unsupported carried input type {type(loop_input)}."
                )
            input_names.append(
                loop_input.name + loop_input.meta.get(TOSA_TENSOR_NAME_META, "")
            )

        num_inputs = len(input_names)
        if num_inputs < len(output.multiple_output_names):
            # This is a strange case, if we reach it something bad has happened.
            raise ValueError(
                f"TOSA specifies that the number of inputs, {input_names}, need to be the "
                f"same as the number of outputs, {output.multiple_output_names}."
            )

        # Wire each output by its getitem index. An output that is not used after
        # the loop has no getitem, so it gets a dummy tensor.
        suffix = node.meta.get(TOSA_TENSOR_NAME_META, "")
        output_names = {
            cast(int, user.args[1]): user.name + suffix for user in node.users
        }
        if len(output_names) < num_inputs:
            body_module = getattr(node.graph.owning_module, body_graph)
            body_outputs = body_module.graph.output_node().args[0]
            for i, body_output in enumerate(body_outputs):
                if i not in output_names:
                    output_names[i] = f"{node.name}_dummy{i}{suffix}"
                    tosa_graph.currRegion.currBasicBlock.addTensor(
                        output_names[i],
                        normalize_symint(body_output.meta["val"].shape),
                        map_dtype(body_output.meta["val"].dtype),
                    )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.WHILE_LOOP,
            input_names,
            [output_names[i] for i in range(num_inputs)],
            attr,
        )
