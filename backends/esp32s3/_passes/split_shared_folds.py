# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Give each core op its own copy of the ops folded into a weight it shares."""


def split_shared_folds(edge_manager, method_name: str = "forward") -> int:
    """Give each matmul, convolution or embedding its own copy of the ops
    preprocess folds into a constant operand of it -- a transpose, a slice, a
    weight dequantise, a float-offset chain -- where several such ops read
    them. Returns how many ops got a copy.

    Preprocess applies those ops at build time (_resolve_const) and has no
    runtime form for them, so the partition of every op reading one has to
    own it, and the gate claims one only with all its readers
    (Esp32OperatorSupport._fold_problem). But the readers' partitions are
    grouped after the gate, and grouping merges two claimed ops only when no
    unclaimed op lies between them: in t = W.t(); h = cos(x @ t); h @ t (an
    RNN hoisting W.t() out of its loop does the same) the transpose would
    join one matmul's partition, the other would receive the transposed
    weight as a runtime input, and preprocess would fail the whole export
    ("needs the weight to be a constant"). With a copy each,
    every reader's fold lands in its own partition, and ExecuTorch gives each
    partition its own copy of the constant placeholder under them, as for
    any shared weight. Readers that land in one partition are unchanged:
    each op folds its own weight there anyway.

    Only ops whose readers all read them into a constant operand of a core
    op, through more such ops or directly, are copied. One a portable op or
    a graph output also reads stays shared, and the gate leaves its readers
    portable on it (a transposed parameter the program also returns).

    lower_esp32 runs this right before the chain strip, on the canonical
    graph, and static_act's two-step lowering in calibrate_static_act; a
    graph where no core op shares one is not touched. A graph partitioned
    without either (to_edge_transform_and_lower) meets the partitioner's
    own net instead: a fold grouping splits stays portable, with its
    readers (Esp32Partitioner._partition)."""
    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    from executorch.backends.esp32s3.operator_support.patterns import SUPPORTED

    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    e = exir_ops.edge.aten
    # The operand slots preprocess resolves as constants (_collect_ops):
    # mm's and convolution's weight, addmm's bias and weight, conv's bias,
    # an embedding's table.
    slots = {e.mm.default: (1,), e.addmm.default: (0, 2),
             e.convolution.default: (1, 2), e.embedding.default: (0,)}
    assert set(slots) >= set(SUPPORTED), "a core op without its constant slots"

    sig = ep.graph_signature
    written = set(sig.buffers_to_mutate.values())
    fixed = (set(sig.inputs_to_parameters)
             | set(sig.inputs_to_lifted_tensor_constants)
             | {n for n, fqn in sig.inputs_to_buffers.items() if fqn not in written})
    # The nodes the program's constants alone determine.
    derived = set()
    for node in graph.nodes:
        if node.op == "placeholder":
            if node.name in fixed:
                derived.add(node)
        elif (node.op == "call_function" and not node.is_impure()
              and node.all_input_nodes
              and all(a in derived for a in node.all_input_nodes)):
            derived.add(node)

    def constant_reads(user, node):
        """Whether `user` reads `node` only in constant slots of a core op."""
        index = slots.get(user.target) if user.op == "call_function" else None
        if index is None:
            return False
        args = list(user.args) + list(user.kwargs.values())
        return all(i in index for i, a in enumerate(args) if a is node)

    def into_constants(user, node, depth=0):
        """Whether `user` takes `node` into constant operands only: a core
        op's constant slot, or an op on constants whose every reader
        does."""
        if constant_reads(user, node):
            return True
        return (depth < 16 and user in derived and user.op == "call_function"
                and bool(user.users)
                and all(into_constants(u, user, depth + 1) for u in user.users))

    copied = 0
    for reader in list(graph.nodes):
        index = slots.get(reader.target) if reader.op == "call_function" else None
        if index is None:
            continue
        for i in index:
            operand = reader.args[i] if i < len(reader.args) else None
            if not (isinstance(operand, torch.fx.Node)
                    and operand.op == "call_function" and operand in derived
                    and constant_reads(reader, operand)):
                continue
            # The operand and every op under it, in graph order.
            closure, stack = set(), [operand]
            while stack:
                n = stack.pop()
                if n.op == "call_function" and n not in closure:
                    closure.add(n)
                    stack.extend(n.all_input_nodes)
            others = [(n, u) for n in closure for u in n.users
                      if u not in closure and u is not reader]
            if not others or not all(into_constants(u, n) for n, u in others):
                continue
            order = [n for n in graph.nodes if n in closure]
            mapping = {}
            with graph.inserting_before(reader):
                for n in order:
                    mapping[n] = graph.node_copy(n, lambda a: mapping.get(a, a))
            reader.replace_input_with(operand, mapping[operand])
            # The originals only this reader read go, so the next reader of
            # the rest sees no reader that is gone.
            for n in reversed(order):
                if not n.users:
                    graph.erase_node(n)
            copied += 1
    if copied:
        graph.lint()
        ep.graph_module.recompile()
    return copied
