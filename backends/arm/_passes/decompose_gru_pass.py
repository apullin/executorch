# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeGruPass(ArmPass):
    """Decomposes aten.gru.input into elementary ops supported by TOSA.

    GRU cell equations per timestep:
        r_t = sigmoid(x_t @ W_ir.T + b_ir + h_{t-1} @ W_hr.T + b_hr)
        z_t = sigmoid(x_t @ W_iz.T + b_iz + h_{t-1} @ W_hz.T + b_hz)
        n_t = tanh(x_t @ W_in.T + b_in + r_t * (h_{t-1} @ W_hn.T + b_hn))
        h_t = n_t + z_t * (h_{t-1} - n_t)

    The weights are batched: one mm computes all three gates at once, then the
    result is sliced into r/z/n components. This yields 2 mm ops per timestep
    instead of 6.

    Supports multi-layer, with/without bias, and batch_first.
    """

    _passes_required_after: Set[Type[ExportPass]] = {InsertTableOpsPass}

    _TARGET = torch.ops.aten.gru.input

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        made_changes = False

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != self._TARGET
                or not self.allowed_to_transform(node.meta)
            ):
                continue

            args = node.args
            input_node = args[0]
            hx_node = args[1]
            params = args[2]
            has_biases = args[3]
            num_layers = args[4]
            # dropout (args[5]) and train (args[6]) are unused at inference
            bidirectional = args[7]
            batch_first = args[8]

            if bidirectional:
                # Bidirectional not yet supported; leave node for fallback.
                continue

            input_val = input_node.meta["val"]
            hx_val = hx_node.meta["val"]

            if batch_first:
                seq_len = input_val.shape[1]
                time_dim = 1
            else:
                seq_len = input_val.shape[0]
                time_dim = 0

            hidden_size = hx_val.shape[-1]
            step = 4 if has_biases else 2

            # Ops namespace — always aten since GRU has no edge dialect variant
            mm_op = torch.ops.aten.mm.default
            t_op = torch.ops.aten.t.default
            add_op = torch.ops.aten.add.Tensor
            sub_op = torch.ops.aten.sub.Tensor
            mul_op = torch.ops.aten.mul.Tensor
            sigmoid_op = torch.ops.aten.sigmoid.default
            tanh_op = torch.ops.aten.tanh.default
            slice_op = torch.ops.aten.slice_copy.Tensor
            unsqueeze_op = torch.ops.aten.unsqueeze.default
            cat_op = torch.ops.aten.cat.default
            select_op = torch.ops.aten.select_copy.int

            with graph.inserting_before(node):
                current_input = input_node
                layer_final_hiddens = []

                for layer_idx in range(num_layers):
                    offset = layer_idx * step
                    weight_ih = params[offset]
                    weight_hh = params[offset + 1]
                    bias_ih = params[offset + 2] if has_biases else None
                    bias_hh = params[offset + 3] if has_biases else None

                    # Initial hidden state for this layer: hx[layer_idx]
                    h_prev = create_node(
                        graph,
                        select_op,
                        args=(hx_node, 0, layer_idx),
                        from_node=node,
                    )

                    # Transpose weights once outside the timestep loop
                    w_ih_t = create_node(
                        graph, t_op, args=(weight_ih,), from_node=node
                    )
                    w_hh_t = create_node(
                        graph, t_op, args=(weight_hh,), from_node=node
                    )

                    timestep_outputs = []

                    for t_idx in range(seq_len):
                        # Extract x_t: select along time dimension
                        x_t = create_node(
                            graph,
                            select_op,
                            args=(current_input, time_dim, t_idx),
                            from_node=node,
                        )

                        # Batched gate computation: 2 mm ops instead of 6
                        # gates_x: [batch, 3*H], gates_h: [batch, 3*H]
                        gates_x = create_node(
                            graph, mm_op, args=(x_t, w_ih_t), from_node=node
                        )
                        gates_h = create_node(
                            graph, mm_op, args=(h_prev, w_hh_t), from_node=node
                        )

                        if bias_ih is not None:
                            gates_x = create_node(
                                graph,
                                add_op,
                                args=(gates_x, bias_ih),
                                from_node=node,
                            )
                        if bias_hh is not None:
                            gates_h = create_node(
                                graph,
                                add_op,
                                args=(gates_h, bias_hh),
                                from_node=node,
                            )

                        # Slice gates into r, z, n components
                        H = hidden_size
                        r_x = create_node(
                            graph,
                            slice_op,
                            args=(gates_x, 1, 0, H),
                            from_node=node,
                        )
                        z_x = create_node(
                            graph,
                            slice_op,
                            args=(gates_x, 1, H, 2 * H),
                            from_node=node,
                        )
                        n_x = create_node(
                            graph,
                            slice_op,
                            args=(gates_x, 1, 2 * H, 3 * H),
                            from_node=node,
                        )

                        r_h = create_node(
                            graph,
                            slice_op,
                            args=(gates_h, 1, 0, H),
                            from_node=node,
                        )
                        z_h = create_node(
                            graph,
                            slice_op,
                            args=(gates_h, 1, H, 2 * H),
                            from_node=node,
                        )
                        n_h = create_node(
                            graph,
                            slice_op,
                            args=(gates_h, 1, 2 * H, 3 * H),
                            from_node=node,
                        )

                        # Reset gate: r = sigmoid(r_x + r_h)
                        r_pre = create_node(
                            graph, add_op, args=(r_x, r_h), from_node=node
                        )
                        r_t = create_node(
                            graph, sigmoid_op, args=(r_pre,), from_node=node
                        )

                        # Update gate: z = sigmoid(z_x + z_h)
                        z_pre = create_node(
                            graph, add_op, args=(z_x, z_h), from_node=node
                        )
                        z_t = create_node(
                            graph, sigmoid_op, args=(z_pre,), from_node=node
                        )

                        # New gate: n = tanh(n_x + r * n_h)
                        r_n_h = create_node(
                            graph, mul_op, args=(r_t, n_h), from_node=node
                        )
                        n_pre = create_node(
                            graph, add_op, args=(n_x, r_n_h), from_node=node
                        )
                        n_t = create_node(
                            graph, tanh_op, args=(n_pre,), from_node=node
                        )

                        # Hidden state: h_t = n_t + z_t * (h_prev - n_t)
                        diff = create_node(
                            graph, sub_op, args=(h_prev, n_t), from_node=node
                        )
                        z_diff = create_node(
                            graph, mul_op, args=(z_t, diff), from_node=node
                        )
                        h_t = create_node(
                            graph, add_op, args=(n_t, z_diff), from_node=node
                        )

                        h_prev = h_t

                        # Unsqueeze for concatenation along time dim
                        h_t_expanded = create_node(
                            graph,
                            unsqueeze_op,
                            args=(h_t, time_dim),
                            from_node=node,
                        )
                        timestep_outputs.append(h_t_expanded)

                    # Concatenate timestep outputs into layer output
                    layer_output = create_node(
                        graph,
                        cat_op,
                        args=(timestep_outputs, time_dim),
                        from_node=node,
                    )

                    current_input = layer_output

                    # Unsqueeze final hidden for stacking across layers
                    h_final_expanded = create_node(
                        graph,
                        unsqueeze_op,
                        args=(h_prev, 0),
                        from_node=node,
                    )
                    layer_final_hiddens.append(h_final_expanded)

                # Build h_n: [num_layers, batch, hidden_size]
                if num_layers == 1:
                    h_n = layer_final_hiddens[0]
                else:
                    h_n = create_node(
                        graph,
                        cat_op,
                        args=(layer_final_hiddens, 0),
                        from_node=node,
                    )

                # Output is the sequence output of the last layer
                output_node = current_input

            # Replace getitem users: GRU returns (output, h_n)
            getitem_nodes = []
            for user in list(node.users.keys()):
                if user.target == operator.getitem:
                    idx = user.args[1]
                    if idx == 0:
                        user.replace_all_uses_with(output_node)
                    elif idx == 1:
                        user.replace_all_uses_with(h_n)
                    getitem_nodes.append(user)

            # Erase getitem nodes then the GRU node explicitly;
            # eliminate_dead_code does not remove GRU because aten
            # considers it impure (may have dropout side-effects).
            for gi in getitem_nodes:
                graph.erase_node(gi)
            graph.erase_node(node)
            made_changes = True

        if not made_changes:
            return PassResult(graph_module, False)

        graph_module.recompile()
        return PassResult(graph_module, True)
