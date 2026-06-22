# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torch

from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceBrokenOpsWithFunctionalOpsPass(ExportPass):
    """
    TODO: Our backend expects pure functions. However, some operators
    are not functionalized properly. This pass intends to replace
    non-functionalized operators with their functionalized variant.

    TODO: this can be refactors into a general OpReplacementPass
    """

    _VIEW_COPY_FALLBACK_PREFIXES = (
        "aten::reshape",
        "aten::unflatten",
    )
    _PRESERVE_VIEW_OP_PREFIXES = ("aten::contiguous",)

    @staticmethod
    def _functionalized_view_target(op):
        namespace, op_full_name = op.name().split("::")
        split = op_full_name.split(".")
        if len(split) == 2:
            op_name, overload_name = split[0], split[1]
        elif len(split) == 1:
            # Add default overload if no overload listed
            op_name = op_full_name
            overload_name = "default"
        else:
            raise RuntimeError(
                f"Invalid op name expected only one '.' to be present: {op_full_name}"
            )

        if op_name == "to":
            return getattr(getattr(torch.ops, namespace), "_to_copy").default

        functional_name = f"{op_name}_copy"
        return getattr(getattr(getattr(torch.ops, namespace), functional_name), overload_name)

    @staticmethod
    def _normalize_to_copy_args(op, args, kwargs):
        normalized_kwargs = dict(kwargs)

        for schema_arg, value in zip(op._schema.arguments[1:], args[1:]):
            normalized_kwargs.setdefault(schema_arg.name, value)

        normalized_kwargs.pop("copy", None)
        return (args[0],), normalized_kwargs

    @staticmethod
    def _fallback_view_copy_rewrite(op, args, meta):
        op_name = op.name()
        if any(
            op_name.startswith(prefix)
            for prefix in ReplaceBrokenOpsWithFunctionalOpsPass._PRESERVE_VIEW_OP_PREFIXES
        ):
            return op, args, {}

        if not any(op_name.startswith(prefix) for prefix in ReplaceBrokenOpsWithFunctionalOpsPass._VIEW_COPY_FALLBACK_PREFIXES):
            return None

        output_val = meta["val"] if "val" in meta else None
        output_shape = getattr(output_val, "shape", None)
        if output_shape is None:
            raise RuntimeError(f"Missing output shape metadata for {op_name}")

        return (
            torch.ops.aten.view_copy.default,
            (args[0], list(output_shape)),
            {},
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            module_modified = False
            for node in module.graph.nodes:
                if node.op != "call_function" or not getattr(
                    node.target, "is_view", False
                ):
                    continue

                try:
                    replacement_op = self._functionalized_view_target(node.target)
                    args, kwargs = node.args, node.kwargs
                    if node.target.name().startswith("aten::to."):
                        args, kwargs = self._normalize_to_copy_args(
                            node.target, node.args, node.kwargs
                        )
                except (AttributeError, RuntimeError):
                    fallback = self._fallback_view_copy_rewrite(
                        node.target, node.args, node.meta
                    )
                    if fallback is None:
                        raise
                    replacement_op, args, kwargs = fallback

                node.target = replacement_op
                node.args = args
                node.kwargs = kwargs
                module_modified = True

            if module_modified:
                module.recompile()
                modified = True

        return PassResult(graph_module, modified)

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op.is_view:
            try:
                view_copy_op = self._functionalized_view_target(op)
                if op.name().startswith("aten::to."):
                    args, kwargs = self._normalize_to_copy_args(op, args, kwargs)
                return super().call_operator(view_copy_op, args, kwargs, meta)
            except (AttributeError, RuntimeError):
                fallback = self._fallback_view_copy_rewrite(op, args, meta)
                if fallback is None:
                    raise
                replacement_op, args, kwargs = fallback
                return super().call_operator(replacement_op, args, kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)
