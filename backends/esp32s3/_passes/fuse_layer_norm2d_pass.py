# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse SAM/ConvNeXt-style LayerNorm2d into one `espressif::layer_norm2d`.

WHAT IT REPLACES. Normalising over the CHANNEL axis of an NCHW tensor is written
the same way in SAM2 and ConvNeXt:

    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    return weight[:, None, None] * x + bias[:, None, None]

which lowers to FOURTEEN portable ops -- two means, two subs, a pow, an add, a
sqrt, a div, four unsqueezes, a mul and an add. Every one of them reads and
writes a full [B,C,H,W] temporary, so the cost is not the arithmetic, it is
fourteen passes over the tensor and thirteen intermediates the memory planner has
to find room for. In a small SAM2-style encoder, three of these are about 42 of
its 383 portable nodes.

Fused, it is two passes: one to accumulate mean and sum-of-squares down each
channel column, one to write the normalised result.

WHY IT IS NOT `native_layer_norm`. That op normalises over the LAST dimensions of
a contiguous tensor. This reduces over dim 1 of NCHW, so the elements of one
reduction are `H*W` apart -- the stride-1024 case at SAM2's resolutions. It is a
genuinely different memory access pattern, which is why writing the model with
`permute -> nn.LayerNorm -> permute` gives a different graph.

BIT-EXACTNESS. The reference below repeats the source's operation order exactly,
including dividing by `sqrt(s + eps)` rather than multiplying by `rsqrt`. Those
differ in the last ulp, and the kernel divides for the same reason: the fused and
unfused programs have to agree so the device has an unambiguous reference. If
profiling ever makes the divide matter, the change is one line in two places and
a new reference -- not a silent substitution.
"""

from typing import Optional

import torch
from torch.fx.passes.infra.pass_base import PassResult
from torch.library import Library

lib = Library("espressif", "FRAGMENT")
lib.define(
    "layer_norm2d(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor")
lib.define(
    "layer_norm2d.out(Tensor x, Tensor weight, Tensor bias, float eps, *, "
    "Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")


def _ln2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
          eps: float) -> torch.Tensor:
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    y = (x - u) / torch.sqrt(s + eps)
    return weight[:, None, None] * y + bias[:, None, None]


def _ln2d_out(x, weight, bias, eps, *, out):
    out.copy_(_ln2d(x, weight, bias, eps))
    return out


impl.impl("layer_norm2d", _ln2d, "CompositeExplicitAutograd")
impl.impl("layer_norm2d.out", _ln2d_out, "CompositeExplicitAutograd")

LN2D = torch.ops.espressif.layer_norm2d.default


def _opname(target) -> str:
    s = str(target)
    return s.split("::")[-1].split("(")[0].strip()


def _is(node, *names) -> bool:
    return (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and _opname(node.target) in names)


def _through_unsqueeze(node, want: int = 2):
    """Peel `want` unsqueeze_copy hops, returning the tensor underneath.

    `weight[:, None, None]` is two unsqueezes on a 1-D parameter. Peeled rather
    than pattern-matched as a unit so a model that writes `weight.view(-1, 1, 1)`
    (one reshape, same meaning) simply does not match and stays portable, instead
    of matching partially and losing a dimension.

    Every hop must insert its axis AFTER the parameter's own (dim >= 1), which
    is what makes the result [C, 1, 1] and lines C up with the channel axis.
    `w.unsqueeze(0).unsqueeze(0)` is the same two hops giving [1, 1, C], which
    broadcasts the parameter over WIDTH.
    """
    n = node
    for _ in range(want):
        if not _is(n, "unsqueeze_copy"):
            return None
        out, dim = n.meta.get("val"), n.args[1]
        if out is None or not isinstance(dim, int):
            return None
        if (dim + out.dim() if dim < 0 else dim) == 0:
            return None
        n = n.args[0]
    return n if isinstance(n, torch.fx.Node) else None


def _unit_alpha(node) -> bool:
    """add/sub with alpha 1 (keyword or third positional), i.e. a plain sum."""
    alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
    return alpha == 1


def _mean_over_c(node, src):
    """Is `node` mean(src, dim=[1], keepdim=True)?"""
    if not _is(node, "mean", "mean.dim"):
        return False
    if node.args[0] is not src:
        return False
    dims = node.args[1] if len(node.args) > 1 else None
    keep = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim")
    return list(dims or []) == [1] and bool(keep)


class FuseLayerNorm2dPass:
    """Match the LayerNorm2d chain from its final add and replace it."""

    def __init__(self, expect: Optional[int] = None, constants=None):
        self.fused = 0
        self.rejected = []
        self.expect = expect
        # Placeholder name -> tensor, for lifted constants. `eps` is lifted to a
        # `_lifted_tensor_constantN` INPUT rather than surviving as a Python
        # float, so without this every site is rejected for having a Node where
        # a number was expected.
        self.constants = constants or {}

    def __call__(self, graph_module):
        return self.call(graph_module)

    def _scalar(self, v):
        """A Python float from a literal or a lifted 0-d constant, else None."""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.fx.Node):
            t = self.constants.get(v.name)
            if isinstance(t, torch.Tensor) and t.numel() == 1:
                return float(t.reshape(-1)[0])
        return None

    def _match(self, add):
        """Return (x, weight, bias, eps) or None. Walks backward from the add."""
        if not _is(add, "add", "add.Tensor"):
            return None
        mul, bias_u = add.args[0], add.args[1]
        if not _is(mul, "mul", "mul.Tensor"):
            return None
        bias = _through_unsqueeze(bias_u)
        if bias is None:
            return None
        # mul(weight[:,None,None], normalised) -- either operand order.
        w_u, q = mul.args[0], mul.args[1]
        weight = _through_unsqueeze(w_u)
        if weight is None:
            weight = _through_unsqueeze(q)
            q, w_u = w_u, q
            if weight is None:
                return None
        if not _is(q, "div", "div.Tensor"):
            return None
        num, den = q.args[0], q.args[1]
        if not _is(den, "sqrt"):
            return None
        eps_add = den.args[0]
        if not _is(eps_add, "add", "add.Tensor"):
            return None
        s, eps = eps_add.args[0], eps_add.args[1]
        eps = self._scalar(eps)
        if eps is None:
            return None
        # s = mean((x-u)^2) over C
        if not _is(s, "mean", "mean.dim"):
            return None
        pw = s.args[0]
        if not _is(pw, "pow", "pow.Tensor_Scalar") or pw.args[1] != 2:
            return None
        sub2 = pw.args[0]
        if not _is(sub2, "sub", "sub.Tensor") or not _is(num, "sub", "sub.Tensor"):
            return None
        # alpha scales the second operand: 2*bias, x - 2*u, s + 2*eps are all
        # different arithmetic from the op's.
        if not all(_unit_alpha(n) for n in (add, eps_add, sub2, num)):
            return None
        x, u = num.args[0], num.args[1]
        # The numerator and the variance must be built from the SAME x and the
        # SAME mean. The source writes `x - u` twice and nothing guarantees CSE
        # merged them, so both spellings are accepted -- but they must agree.
        if sub2.args[0] is not x or sub2.args[1] is not u:
            return None
        if not _mean_over_c(u, x) or not _mean_over_c(s, pw):
            return None
        if not (isinstance(weight, torch.fx.Node)
                and isinstance(bias, torch.fx.Node)):
            return None
        wv, bv = weight.meta.get("val"), bias.meta.get("val")
        xv = x.meta.get("val")
        if wv is None or bv is None or xv is None:
            return None
        # 4-D NCHW with 1-D affine params of length C. Anything else is a
        # different operator wearing the same shape of graph.
        if xv.dim() != 4 or wv.dim() != 1 or bv.dim() != 1:
            return None
        if wv.shape[0] != xv.shape[1] or bv.shape[0] != xv.shape[1]:
            return None
        # fp32 on every tensor the device kernel touches: it reads and writes
        # through float pointers, so a fused fp16 graph would have its storage
        # read as fp32 and its output written past the end.
        ov = add.meta.get("val")
        if any(v is None or v.dtype != torch.float32
               for v in (xv, wv, bv, ov)):
            return None
        # Contiguous NCHW in and out, as the kernel requires (espressif_ops.cpp,
        # contiguous_tensor on both, which like is_contiguous() ignores the
        # strides of size-1 axes): fused over a channels-last x -- a Cortex-M
        # image, which a reflect pad's decomposition hands on channels-last
        # -- the program would fail at execute.
        if not (xv.is_contiguous() and ov.is_contiguous()):
            return None
        # Every intermediate must be single-use, or fusing would delete a value
        # something else still reads.
        for mid in (mul, q, den, eps_add, s, pw, sub2, num, u):
            if isinstance(mid, torch.fx.Node) and len(mid.users) != 1:
                if mid is not num and mid is not u:
                    return None
        return x, weight, bias, float(eps)

    def call(self, graph_module):
        graph = graph_module.graph
        changed = False
        for node in list(graph.nodes):
            got = self._match(node)
            if got is None:
                continue
            x, weight, bias, eps = got
            with graph.inserting_before(node):
                new = graph.call_function(LN2D, (x, weight, bias, eps))
            new.meta = dict(node.meta)
            node.replace_all_uses_with(new)
            self.fused += 1
            changed = True
        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()
        if self.expect is not None and self.fused != self.expect:
            raise RuntimeError(
                f"FuseLayerNorm2dPass fused {self.fused}, expected {self.expect}")
        return PassResult(graph_module, changed)


def fuse_layer_norm2d(edge_manager, method_name: str = "forward",
                      expect: Optional[int] = None):
    """Run over an EdgeProgramManager in place; return the pass for its counts."""
    from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import constant_map

    ep = edge_manager.exported_program(method_name)
    p = FuseLayerNorm2dPass(expect=expect, constants=constant_map(ep))
    p.call(ep.graph_module)
    return p
