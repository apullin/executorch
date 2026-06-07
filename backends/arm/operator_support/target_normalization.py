from __future__ import annotations

import typing
from collections.abc import Sequence

_TARGET_ALIASES = {
    "aten::clone.default": "dim_order_ops::_clone_dim_order.default",
    "aten::conv1d.default": "aten::convolution.default",
    "aten::conv2d.default": "aten::convolution.default",
    "aten::copy_.default": "aten::copy.default",
    "aten::expand.default": "aten::expand_copy.default",
    "aten::fill_.Scalar": "aten::full_like.default",
    "aten::flatten.using_ints": "aten::view_copy.default",
    "aten::permute.default": "aten::permute_copy.default",
    "aten::reshape.default": "aten::view_copy.default",
    "aten::slice.Tensor": "aten::slice_copy.Tensor",
    "aten::_to_copy.default": "dim_order_ops::_to_dim_order_copy.default",
    "aten::_unsafe_view.default": "aten::view_copy.default",
    "aten::view.default": "aten::view_copy.default",
}


def target_key(target: typing.Any) -> str:
    """Return a canonical key for matching raw and Edge operator targets."""
    if isinstance(target, str):
        return _TARGET_ALIASES.get(target, target)

    name = target.name() if hasattr(target, "name") else str(target)
    overload = getattr(target, "_overloadname", "")
    if overload and not name.endswith(f".{overload}"):
        name = f"{name}.{overload}"
    return _TARGET_ALIASES.get(name, name)


def target_keys(targets: Sequence[typing.Any]) -> frozenset[str]:
    return frozenset(target_key(target) for target in targets)
