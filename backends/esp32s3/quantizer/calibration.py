# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static activation calibration for FLAG_STATIC_ACT."""

import torch
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.backends.esp32s3.operator_support.const_weights import _const_tensor
from executorch.backends.esp32s3.serialization.espressif_blob import BACKEND_NAME


def calibrate_static_act(edge_manager, batches, method_name: str = "forward"):
    """Record per-matmul activation absmax into node meta, for FLAG_STATIC_ACT.

    Runs the edge graph -- BEFORE lowering, while the matmuls are still ordinary
    nodes -- over the given calibration batches with an fx Interpreter, recording
    max(|activation|) for every mm/addmm/conv into node.meta['esp32_act_amax'] (max across
    batches). The partitioner copies node meta into each partition subprogram,
    which is how the numbers reach preprocess without any side channel.

    Returns the recorded amaxes as a list of (node_name, amax) in graph order --
    the export scripts feed these straight into the host reference, so the
    reference and the blob share ONE set of calibration numbers rather than two
    hopefully-identical replays.

    Two-step lowering is required, then the post-lowering passes as for
    every program:

        edge = to_edge_transform_and_lower(ep)      # no partitioner yet
        calibrate_static_act(edge, [(x,)])
        edge = edge.to_backend(Esp32Partitioner(static_act=True))
        apply_passes(edge)

    A dynamically quantised graph's ops (a dynamic recipe's chooser chain in
    front of the activation) are refused under static_act and stay portable
    on the recipe's value (KIND_STATIC_ACT_DYNAMIC); apply_passes computes
    those chains in portable ops, so the program still emits and runs.

    First, each matmul, convolution or embedding gets its own copy of a
    transpose, slice or weight dequantise others share (split_shared_folds),
    as lower_esp32 does before partitioning.
    """
    # The partitioner runs next, on this graph: a transpose, slice or weight
    # dequantise several matmuls share gets a copy per reader first, as
    # lower_esp32 gives it (split_shared_folds.py), so readers grouped into
    # different partitions each fold their own weight.
    from executorch.backends.esp32s3._passes.split_shared_folds import split_shared_folds

    split_shared_folds(edge_manager, method_name)
    ep = edge_manager.exported_program(method_name)
    gm = ep.graph_module
    MM = (exir_ops.edge.aten.mm.default, exir_ops.edge.aten.addmm.default)
    CONV = exir_ops.edge.aten.convolution.default

    # CONV is calibrated too: preprocess REQUIRES esp32_act_amax on every
    # quantised op it emits under static_act, and convolutions are delegated by
    # default, so without it `static_act=True` on any model containing a
    # convolution would fail at export ("... has no esp32_act_amax meta").
    #
    # A conv's activation is args[0], the input tensor, and one scale for the
    # whole op is exactly what `quantize_conv_row`'s `fixed_sx` consumes. Taking
    # the max over the whole input rather than per patch is conservative by
    # construction: every gathered patch is a subset of it, so nothing clips
    # that would not have clipped anyway.
    def _act_of(n):
        if n.target is CONV:
            return n.args[0]
        return n.args[0] if n.target is MM[0] else n.args[1]

    def _is_quantised(n):
        return n.op == "call_function" and (n.target in MM or n.target is CONV)

    class _Cal(torch.fx.Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if _is_quantised(n):
                mat1 = _act_of(n)
                val = self.env.get(mat1)
                if isinstance(val, torch.Tensor):
                    amax = float(val.detach().abs().max())
                    prev = n.meta.get("esp32_act_amax", 0.0)
                    n.meta["esp32_act_amax"] = max(prev, amax)
            return out

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for batch in batches:
        batch = list(batch)
        args = []
        for ph in placeholders:
            const = _const_tensor(ep, ph)
            args.append(const if const is not None else batch.pop(0))
        if batch:
            raise RuntimeError(
                f"{BACKEND_NAME}: calibration batch has {len(batch)} unused "
                f"input(s); the graph takes fewer"
            )
        _Cal(gm).run(*args)

    return [(n.name, n.meta["esp32_act_amax"])
            for n in gm.graph.nodes
            if _is_quantised(n) and "esp32_act_amax" in n.meta]
