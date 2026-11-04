# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fold batch norm into the preceding convolution's weights."""


# ---- conv + batch-norm folding ----------------------------------------------
def fold_conv_bn(edge_manager, method_name: str = "forward"):
    """Fold eval-mode BatchNorm into a preceding convolution. Returns (folded, rejected).

    The classic:  bn(conv(x, W, b)) == conv(x, W', b')  with, per output channel,

        s  = gamma * rsqrt(running_var + eps)
        W' = W * s          (broadcast over the non-channel dims)
        b' = (b - running_mean) * s + beta

    The implementation choice is the part worth reading: rather than computing W'/b'
    here and surgically editing the state_dict -- which means reproducing
    constant-tensor plumbing that already exists elsewhere -- the pass inserts the
    scale/shift arithmetic as ORDINARY GRAPH OPS on the constant operands and rewires
    the conv. Every inserted op has all-constant inputs, so the `constant_prop_pass`
    that apply_passes already runs immediately afterwards materialises them into plain
    folded tensors and deletes the arithmetic. Composition instead of duplication; the
    emitted program carries exactly one conv with pre-multiplied weights and no trace
    of the bn.

    Why not ExecuTorch's own FuseBatchNormWithConvPass (backends/transforms/), which
    wraps torch's canonical fuse_conv_bn_weights: it commits the folded tensors via
    graph_module.register_parameter + get_attr nodes, which on a LIFTED exported
    program leaves the graph disagreeing with its signature -- the original
    parameter placeholders survive in the calling convention while the graph reads
    new get_attrs. That works inside XNNPACK's pass manager, which re-handles
    lifting afterwards; in this pipeline it breaks the module's calling convention
    outright. Inserting ops and letting constant_prop materialise them never
    touches the signature's inputs, so the entire failure class is absent by
    construction. The one name the fold changes is an output's: where the bn's
    result is a graph output, the conv's takes its place in the output list,
    and the signature, which names every output by its node, is renamed with
    it (to_backend validates the two against each other before partitioning).

    Strictness, as in the rmsnorm pass -- every reject is recorded, and a site
    is folded only when the algebra is exactly the one above:
      * eval-mode `_native_batch_norm_legit_no_training` only. The training variants
        update running stats; folding them would silently change semantics.
      * the conv must have the bn as its ONLY user (otherwise the unfused output is
        still live), and only the bn's output 0 may be consumed.
      * transposed convolution is rejected: its weight lays channels out [in, out/g]
        and the broadcast below would scale the wrong axis.
      * every operand -- weight, bias, gamma, beta, mean, var -- must be a
        build-time constant. The inserted ops carry no fake-tensor meta and are
        valid only once constant propagation has replaced them; on a runtime
        weight they would survive as invalid IR ("Node.meta ... is missing val
        field") and fail lower_esp32.
      * "build-time constant" is what constant propagation folds, by its own
        rules, as the gate decides it (_folded_constants): a parameter, an
        unmutated buffer or a lifted constant, and each pure op over them --
        but never a quantize or dequantize primitive, nor aten.full, which it
        leaves (constant_prop_pass's _DEFAULT_SKIP_TARGETS). A QAT conversion
        folds a conv's own BatchNorm into int codes and leaves the weight as a
        dequantise of them; a second BatchNorm after that conv would fold onto
        the dequantise, a product constant propagation never computes.
        It stays portable instead, on the recipe's codes. The set is read
        before the fold, so a product the fold inserts is not in it: a second
        BatchNorm on a conv whose first one folded stays portable too.
    """
    import operator

    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        _folded_constants,
    )

    CONV = exir_ops.edge.aten.convolution.default
    BN = exir_ops.edge.aten._native_batch_norm_legit_no_training.default
    MUL = exir_ops.edge.aten.mul.Tensor
    ADD = exir_ops.edge.aten.add.Tensor
    ADD_S = exir_ops.edge.aten.add.Scalar
    SUB = exir_ops.edge.aten.sub.Tensor
    RSQRT = exir_ops.edge.aten.rsqrt.default
    VIEW = exir_ops.edge.aten.view_copy.default
    MUL_S = exir_ops.edge.aten.mul.Scalar

    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    folded, rejected = 0, []
    # What constant propagation folds, by its own rules: the gate's set
    # (_folded_constants), so the fold and the gate agree on every node, and
    # an ExecuTorch whose skip set the gate cannot read is refused by name.
    constant = _folded_constants(ep)

    def build_time_constant(node):
        return not isinstance(node, torch.fx.Node) or node in constant

    for bn in list(graph.nodes):
        if not (bn.op == "call_function" and bn.target == BN):
            continue
        conv = bn.args[0]
        if not (isinstance(conv, torch.fx.Node) and conv.op == "call_function"
                and conv.target == CONV):
            rejected.append((bn.name, "input is not a convolution"))
            continue
        if len(conv.users) != 1:
            rejected.append((bn.name, "conv output is used beyond the bn"))
            continue
        if bool(conv.args[6]):
            rejected.append((bn.name, "transposed conv lays channels out [in, out/g]"))
            continue
        getitems = list(bn.users)
        if not all(u.op == "call_function" and u.target == operator.getitem
                   for u in getitems):
            rejected.append((bn.name, "bn consumed other than through getitem"))
            continue
        if any(u.args[1] != 0 and u.users for u in getitems):
            rejected.append((bn.name, "bn's saved-stats outputs are consumed"))
            continue

        gamma, beta, mean, var = bn.args[1], bn.args[2], bn.args[3], bn.args[4]
        eps = float(bn.args[6])
        w_node, b_node = conv.args[1], conv.args[2]
        if not all(build_time_constant(n)
                   for n in (w_node, b_node, gamma, beta, mean, var)):
            rejected.append((bn.name, "an operand is not a build-time constant"))
            continue
        w_val = w_node.meta.get("val", None)
        if w_val is None:
            rejected.append((bn.name, "conv weight has no fake-tensor meta"))
            continue

        with graph.inserting_before(conv):
            s = graph.call_function(RSQRT, (graph.call_function(ADD_S, (var, eps)),))
            if gamma is not None:
                s = graph.call_function(MUL, (s, gamma))
            s_view = graph.call_function(
                VIEW, (s, [int(w_val.shape[0])] + [1] * (w_val.dim() - 1)))
            new_w = graph.call_function(MUL, (w_node, s_view))
            if b_node is not None:
                shifted = graph.call_function(SUB, (b_node, mean))
                new_b = graph.call_function(MUL, (shifted, s))
            else:
                new_b = graph.call_function(MUL_S,
                                            (graph.call_function(MUL, (mean, s)),
                                             -1.0))
            if beta is not None:
                new_b = graph.call_function(ADD, (new_b, beta))

        args = list(conv.args)
        args[1], args[2] = new_w, new_b
        conv.args = tuple(args)
        for u in getitems:
            if u.args[1] == 0:
                if any(user.op == "output" for user in u.users):
                    for spec in ep.graph_signature.output_specs:
                        if getattr(spec.arg, "name", None) == u.name:
                            spec.arg.name = conv.name
                u.replace_all_uses_with(conv)
        for u in getitems:
            graph.erase_node(u)
        graph.erase_node(bn)
        folded += 1

    if folded:
        graph.eliminate_dead_code()
        graph.lint()
        ep.graph_module.recompile()
    return folded, rejected
