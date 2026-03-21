# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torch

from executorch.exir.pass_base import ExportPass


class ReplaceBrokenOpsWithFunctionalOpsPass(ExportPass):
    """
    TODO: Our backend expects pure functions. However, some operators
    are not functionalized properly. This pass intends to replace
    non-functionalized operators with their functionalized variant.

    TODO: this can be refactors into a general OpReplacementPass
    """

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

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op.is_view:
            view_copy_op = self._functionalized_view_target(op)
            if op.name().startswith("aten::to."):
                args, kwargs = self._normalize_to_copy_args(op, args, kwargs)
            return super().call_operator(view_copy_op, args, kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)
