# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""residual + rmsnorm(y) fuses into espressif::rmsnorm_add, bit-identically.

A two-sublayer post-norm block (x = x + norm(W x), twice, the second with the
residual on the right) lowers through the standard path: both adds fuse, no
aten add remains, and the program's output equals the SAME lowering with this
fusion switched off, bit for bit (comparing against the fp32 eager module would
measure the int8 delegate, not this rewrite). Near misses stay unfused: a norm
with a second user, a broadcast residual, and an add with alpha != 1.
"""
import torch
import torch.nn as nn

from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3._passes.fuse_rmsnorm_add_pass import RMSNORM_ADD


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(d) + 0.5)

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class PostNorm(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)
        self.na, self.nb = RMSNorm(d), RMSNorm(d)

    def forward(self, x):
        x = x + self.na(self.a(x))
        return self.nb(self.b(x)) + x          # residual on the right


class NearMisses(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)

    def forward(self, x):
        h = self.n1(self.a(x))
        y = x + h + h.sum()                     # the norm has a second user
        z = self.n2(y)[:, :1] + y               # broadcast residual (shape differs)
        return torch.add(z, self.n3(z), alpha=2.0)   # alpha != 1


class Doubled(nn.Module):
    """h + h with h = norm(W x): one user node, two uses of the norm."""

    def __init__(self, d=64):
        super().__init__()
        self.a, self.n = nn.Linear(d, d), RMSNorm(d)

    def forward(self, x):
        h = self.n(self.a(x))
        return h + h


def lowered(model, x, fuse_adds=True):
    from executorch.backends.esp32s3._passes import fuse_rmsnorm_add_pass
    real = fuse_rmsnorm_add_pass.fuse_rmsnorm_add
    if not fuse_adds:
        fuse_rmsnorm_add_pass.fuse_rmsnorm_add = lambda *a, **k: 0
    try:
        ep = torch.export.export(model.eval(), (x,))
        em = lower_esp32(ep, [Esp32Partitioner()])
        rep = apply_passes(em, quiet=True)
    finally:
        fuse_rmsnorm_add_pass.fuse_rmsnorm_add = real
    return em, rep


def count(em, target):
    return sum(1 for n in em.exported_program().graph.nodes
               if n.op == "call_function" and n.target is target)


def test_post_norm_adds_fuse_bit_exact():
    torch.manual_seed(0)
    model, x = PostNorm(), torch.randn(1, 4, 64)
    em, rep = lowered(model, x)
    assert rep["rms_add_fused"] == 2, rep["rms_add_fused"]
    assert count(em, RMSNORM_ADD) == 2
    adds = [n for n in em.exported_program().graph.nodes
            if n.op == "call_function" and "aten.add" in str(n.target)]
    assert not adds, adds
    reference, ref_rep = lowered(model, x, fuse_adds=False)
    assert ref_rep["rms_add_fused"] == 0
    with torch.no_grad():
        want = reference.exported_program().module()(x)
        got = em.exported_program().module()(x)
    assert torch.equal(got, want), (got - want).abs().max()


def test_norm_added_to_itself_stays_unfused():
    torch.manual_seed(2)
    em, rep = lowered(Doubled(), torch.randn(1, 4, 64))
    assert rep["rms_add_fused"] == 0, rep["rms_add_fused"]


def test_near_misses_stay_unfused():
    torch.manual_seed(1)
    em, rep = lowered(NearMisses(), torch.randn(1, 4, 64))
    assert rep["rms_add_fused"] == 0, rep["rms_add_fused"]


if __name__ == "__main__":
    test_post_norm_adds_fuse_bit_exact()
    test_norm_added_to_itself_stays_unfused()
    test_near_misses_stay_unfused()
    print("PASS")
