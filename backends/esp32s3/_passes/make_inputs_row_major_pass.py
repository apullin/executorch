# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph inputs in another dim order (channels-last images), made row-major
once, right after the input.

The delegate reads and writes row-major tensors only, and ExecuTorch keeps
every tensor's dim order: export takes it from the example input's strides,
every op after it inherits it (a convolution's output follows its input), and
to_executorch emits each tensor with the dim order its strides give
(exir/tensor.py dim_order_from_stride). So an image fed channels-last, the
Cortex-M flow's NHWC, reaches every convolution channels-last and the gate
refuses it ("runtime tensor is not row-major"). Portable kernels mind it too:
add, mul and div refuse operands whose dim orders differ
(tensors_have_same_dim_order, tensor_util_portable.cpp:133), and a 0-dim
constant (hardswish's + 3) is default-order while a channels-last [N, C, 1, 1]
is not, as in the census's Cortex-M squeeze-excite block.

make_inputs_row_major copies each such input to the default dim order right
after its placeholder, re-derives every downstream tensor's metadata from it
with each op's own fake kernel (so each layout follows as eager's would, the
way ExecuTorch's own LegalizePortableDimOrderPass re-derives what follows the
copy it puts in front of avg_pool2d), and copies a graph output whose dim
order thereby changed back to the one the program declared. The program's
inputs and outputs keep their dim orders; everything between is row-major,
so the partitioner claims what it claims for the row-major program and the
delegate computes the same bits: the copies move values, they never round
them. The price is one copy of each such input (and of each such output),
paid once, where it is least, instead of at every delegate boundary.

A dim order changed inside the model (an explicit
x.contiguous(memory_format=torch.channels_last)) is the model's own and is
kept: the convolutions after it stay portable, refused by name.

A copy a decomposition wrote is not. torch decomposes a reflect or
replicate pad into index ops followed by
`.contiguous(memory_format=suggest_memory_format(...))`, which returns the
tensor in the input's format; traced with a channels-last input, that copy
is frozen as a _clone_dim_order to channels-last, and everything after it
(the convolution the pad feeds) would stay channels-last and portable,
and a portable kernel after it could meet two dim orders. Such a copy --
one whose node meta names another ATen op as the op it was decomposed
from (`original_aten`), not a clone, contiguous or to-copy the model
wrote, and whose target is the dim order a copied input had -- writes
the default dim order instead: what the decomposition would have written
for the row-major input. `released` in the report lists them.

An output declared in a dim order that is neither the default nor
channels-last stays row-major. Eager declares such orders: torch's add of
a channels-last [1, C, H, 1] input and a row-major [1, C, H, W] tensor
has strides in dim order [0, 2, 1, 3]. No ExecuTorch kernel writes one:
_to_dim_order_copy maps its dim order to a memory format, at export
(dim_order_utils.get_memory_format) and at run time
(check__to_dim_order_copy_args), and the elementwise kernels take operands
all row-major or all channels-last. The values are the same in any order,
and the emitted program declares the order its output is in.

A no-op, graph untouched, when every user input already has the default dim
order, as most programs have. Imports torch and executorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.tensor import dim_order_from_stride

_TO_DIM_ORDER_COPY = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
_CLONE_DIM_ORDER = exir_ops.edge.dim_order_ops._clone_dim_order.default
# The ATen ops a model changes a layout with: a dim-order copy decomposed from
# one of these is the model's own (kept); from any other op, a decomposition's.
_MODEL_COPIES = {"aten::clone", "aten::contiguous", "aten::_to_copy", "aten::to",
                 "aten::copy", "aten::_copy_from", "aten::empty_like",
                 "dim_order_ops::_clone_dim_order", "dim_order_ops::_to_dim_order_copy"}


@dataclass
class RowMajorReport:
    """What make_inputs_row_major did: each input it copied and each output
    it restored, with the dim order each has; `skipped` says why it left a
    graph with such an input untouched."""

    inputs: list = field(default_factory=list)    # [(name, dim order)]
    outputs: list = field(default_factory=list)   # [(index, dim order)]
    # [(index, declared dim order)]: no ExecuTorch copy writes that order
    row_major_outputs: list = field(default_factory=list)
    # [(node name, the op decomposed into it)]: a decomposition's copy back to
    # an input's dim order, rewritten to write the default one
    released: list = field(default_factory=list)
    skipped: str = ""

    def summary(self) -> str:
        if self.skipped:
            return f"row-major inputs: left as they are ({self.skipped})"
        text = ("row-major inputs: copied " + ", ".join(
            f"{name} {list(order)}" for name, order in self.inputs)
            + " to the default dim order")
        if self.released:
            text += "; " + ", ".join(
                f"{name} (from {op}'s decomposition)" for name, op in self.released) + \
                " writes the default dim order"
        if self.outputs:
            text += "; output " + ", ".join(
                f"{i} copied back to {list(order)}" for i, order in self.outputs)
        if self.row_major_outputs:
            text += "; output " + ", ".join(
                f"{i} left row-major (declared {list(order)}, neither "
                f"row-major nor channels-last, which no ExecuTorch copy "
                f"writes)" for i, order in self.row_major_outputs)
        return text


def dim_order(value):
    """The dim order ExecuTorch emits for a tensor value (TensorSpec), or
    None for anything else."""
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(d) for d in dim_order_from_stride(value.stride()))


def _is_default(order) -> bool:
    return order is None or tuple(order) == tuple(range(len(order)))


def _has_memory_format(order) -> bool:
    """Whether _to_dim_order_copy can write this dim order: the default or
    channels-last, the two get_memory_format maps."""
    from executorch.exir.dim_order_utils import get_memory_format

    try:
        get_memory_format(list(order))
    except AssertionError:
        return False
    return True


def _is_row_major_copy(node) -> bool:
    """A _to_dim_order_copy to the default dim order (this pass's own)."""
    if not (node.op == "call_function" and node.target is _TO_DIM_ORDER_COPY):
        return False
    order = node.kwargs.get("dim_order")
    return order is not None and _is_default(order)


def make_inputs_row_major(edge_manager, method_name: str = "forward"):
    """Copy every user input whose dim order is not the default to the
    default one, right after its placeholder, and keep the program's outputs
    in the dim orders it declared; returns a RowMajorReport."""
    ep = edge_manager.exported_program(method_name)
    gm = ep.graph_module
    graph = gm.graph
    report = RowMajorReport()
    user_inputs = set(ep.graph_signature.user_inputs)
    todo = [n for n in graph.nodes if n.op == "placeholder"
            and n.name in user_inputs
            and not _is_default(dim_order(n.meta.get("val")))
            and not (len(n.users) == 1 and _is_row_major_copy(next(iter(n.users))))]
    if not todo:
        return report
    output = next(n for n in graph.nodes if n.op == "output")
    outs, spec = torch.utils._pytree.tree_flatten(output.args[0])
    declared = [dim_order(o.meta.get("val")) if isinstance(o, torch.fx.Node)
                else None for o in outs]
    mode = todo[0].meta["val"].fake_mode
    saved = {n: n.meta["val"] for n in graph.nodes
             if n.op == "call_function" and "val" in n.meta}
    copies = []
    for p in todo:
        with graph.inserting_after(p):
            copy = graph.call_function(
                _TO_DIM_ORDER_COPY, (p,),
                {"dim_order": list(range(p.meta["val"].dim()))})
        copy.meta = {k: v for k, v in p.meta.items() if k != "val"}
        p.replace_all_uses_with(copy,
                                delete_user_cb=lambda u, c=copy: u is not c)
        copies.append((p, copy))
    released = _release_decomposition_copies(
        graph, {c for _, c in copies}, {dim_order(p.meta["val"]) for p in todo})
    try:
        _repropagate(graph, {c for _, c in copies}, mode)
    except Exception as why:   # an op whose metadata cannot be re-derived
        for n, val in saved.items():
            n.meta["val"] = val
        for n, kwargs, _ in released:
            n.kwargs = kwargs
        for p, copy in copies:
            copy.replace_all_uses_with(p)
            graph.erase_node(copy)
        graph.lint()
        gm.recompile()
        report.skipped = (f"cannot re-derive the metadata after the copy: "
                          f"{type(why).__name__}: {why}")
        return report
    report.inputs = [(p.name, dim_order(p.meta["val"])) for p, _ in copies]
    report.released = [(n.name, source) for n, _, source in released]
    restored = []
    for i, (o, order) in enumerate(zip(outs, declared)):
        if (isinstance(o, torch.fx.Node) and order is not None
                and dim_order(o.meta.get("val")) != order):
            if not _has_memory_format(order):
                report.row_major_outputs.append((i, order))
                restored.append(o)
                continue
            with graph.inserting_before(output):
                back = graph.call_function(_TO_DIM_ORDER_COPY, (o,),
                                           {"dim_order": list(order)})
            back.meta = {k: v for k, v in o.meta.items() if k != "val"}
            with mode:
                back.meta["val"] = _TO_DIM_ORDER_COPY(o.meta["val"],
                                                      dim_order=list(order))
            # The signature names every output by its node.
            for out_spec in ep.graph_signature.output_specs:
                if getattr(out_spec.arg, "name", None) == o.name:
                    out_spec.arg.name = back.name
            report.outputs.append((i, order))
            restored.append(back)
        else:
            restored.append(o)
    if report.outputs:
        output.args = (torch.utils._pytree.tree_unflatten(restored, spec),)
    graph.lint()
    gm.recompile()
    return report


def _decomposed_from(node):
    """The ATen op `node` was decomposed from, by name ('aten::reflection_pad2d'),
    or None when its meta does not say (original_aten)."""
    schema = getattr(node.meta.get("original_aten"), "_schema", None)
    return getattr(schema, "name", None)


def _release_decomposition_copies(graph, copies, orders):
    """Make each dim-order copy downstream of `copies` that a decomposition
    wrote back to one of `orders` (the copied inputs' dim orders) write the
    default dim order instead, as everything after `copies` is. A copy decomposed
    from a clone, contiguous or to-copy is the model's own and stays, as does
    one whose meta does not say where it came from. Returns [(node, its old
    kwargs, the op it was decomposed from)], for the caller to undo."""
    downstream, released = set(copies), []
    for n in graph.nodes:
        if (n.op != "call_function" or n in downstream
                or not any(a in downstream for a in n.all_input_nodes)):
            continue
        downstream.add(n)
        if n.target not in (_CLONE_DIM_ORDER, _TO_DIM_ORDER_COPY):
            continue
        order = n.kwargs.get("dim_order")
        if order is None or _is_default(order) or tuple(order) not in orders:
            continue
        source = _decomposed_from(n)
        if source is None or source in _MODEL_COPIES:
            continue
        released.append((n, dict(n.kwargs), source))
        n.kwargs = {**n.kwargs, "dim_order": list(range(len(order)))}
    return released


def _repropagate(graph, changed, mode):
    """Recompute node.meta['val'] of every call_function node that reads,
    transitively, one in `changed` (those included), in graph order, with
    each op's own fake kernel."""
    dirty = set(changed)

    def val(a):
        return a.meta["val"]

    with mode:
        for n in graph.nodes:
            if n.op != "call_function" or not (
                    n in dirty or any(a in dirty for a in n.all_input_nodes)):
                continue
            args = torch.fx.node.map_arg(n.args, val)
            kwargs = torch.fx.node.map_arg(n.kwargs, val)
            n.meta["val"] = n.target(*args, **kwargs)
            dirty.add(n)
