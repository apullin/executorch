# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RoPE fuses to one op, from either spelling, bit-exactly.

Eight edge ops per application, each writing a full-width temporary. The
claims:

  1. BOTH published spellings reduce to the same canonical op. HuggingFace
     writes `x*cos + rotate_half(x)*sin`; this repo's own model writes
     `cat([x1*cos - x2*sin, x2*cos + x1*sin])`. Which one a model author used is
     not something a backend should get to care about -- the same principle as
     the bmm-as-linear and 4-D q@kT canonicalisations.
  2. Output is BIT-IDENTICAL, on real models, not merely close. The fused op
     performs the same IEEE operations in the same order; anything less than
     equality would mean it does not.
  3. Partial rotary (rot < width, with pass-through lanes) and full rotary both
     work.
  4. A near-miss is REFUSED with a reason rather than mangled.

    python test_rope_fusion.py
"""

import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def main():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.fuse_rope_pass import FuseRoPEPass, _rope

    print("both spellings reduce to the canonical op:")
    torch.manual_seed(0)
    x = torch.randn(1, 4, 6, 8)
    half = 4
    cos = torch.randn(6, 8)
    sin = torch.randn(6, 8)
    x1, x2 = x.chunk(2, -1)
    hf = x * cos + torch.cat([-x2, x1], -1) * sin
    check("HF rotate-half, non-duplicated full table",
          torch.equal(hf, _rope(x, cos, sin, 8)))
    c, s = cos[:, :half], sin[:, :half]
    pairwise = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    check("pairwise cat (this repo's model)",
          torch.equal(pairwise, _rope(x, c, s, 8)))

    class HFFullTable(nn.Module):
        def forward(self, y, cs, sn):
            a, b = y.chunk(2, -1)
            return y * cs + torch.cat([-b, a], -1) * sn

    hfmod = HFFullTable().eval()
    he = to_edge(torch.export.export(hfmod, (x, cos, sin)))
    hgm = he.exported_program().graph_module
    with torch.no_grad():
        h0 = he.exported_program().module()(x, cos, sin)
    hp = FuseRoPEPass(expect=1)
    hp.call(hgm)
    with torch.no_grad():
        h1 = he.exported_program().module()(x, cos, sin)
    check("HF non-duplicated table graph fuses bit-identically", torch.equal(h0, h1))

    print("real models, bit-identical after fusing:")
    from executorch.examples.espressif.esp32s3.story.glimmer_model import Config, TinyLM

    lm = TinyLM(Config(vocab_size=512, d_model=96, n_layers=2, n_heads=4,
                       ffn_hidden=66, seq_len=32, ple_dim=64, arm="ple")).eval()
    ids = torch.randint(0, 512, (1, 32))
    e2 = to_edge(torch.export.export(lm, (ids,)))
    gm2 = e2.exported_program().graph_module
    with torch.no_grad():
        z0 = e2.exported_program().module()(ids)
    p2 = FuseRoPEPass()
    p2.call(gm2)
    with torch.no_grad():
        z1 = e2.exported_program().module()(ids)
    z0 = z0[0] if isinstance(z0, tuple) else z0
    z1 = z1[0] if isinstance(z1, tuple) else z1
    check("PLE model: 4 sites fused (full rotary, other spelling)",
          p2.fused == 4, f"{p2.fused} fused")
    check("PLE model: output bit-identical", torch.equal(z0, z1))

    print("a near-miss is refused with a reason:")

    class NotRope(nn.Module):
        """Halves swapped: cat([x1*cos + x2*sin, ...]) is not a rotation."""

        def forward(self, y, cs, sn):
            a, b = y.chunk(2, -1)
            return torch.cat([a * cs + b * sn, b * cs + a * sn], -1)

        
    p3 = FuseRoPEPass()
    e4 = to_edge(torch.export.export(
        NotRope().eval(), (torch.randn(1, 4, 8), torch.randn(4, 4),
                           torch.randn(4, 4))))
    p3.call(e4.exported_program().graph_module)
    check("wrong sign structure not fused", p3.fused == 0, f"{p3.fused} fused")
    check("and the refusal states a cause", bool(p3.rejected),
          p3.rejected[0][1][:52] if p3.rejected else "no reason recorded")

    class UnevenSlices(nn.Module):
        def forward(self, y, cs, sn):
            # Still width-preserving (5+3), but not rotate_half's 4+4 split.
            rotated = torch.cat([-y[..., 3:], y[..., :3]], -1)
            return y * cs + rotated * sn

    p4 = FuseRoPEPass()
    e5 = to_edge(torch.export.export(UnevenSlices().eval(), (x, cos, sin)))
    p4.call(e5.exported_program().graph_module)
    check("uneven slice halves not fused", p4.fused == 0, f"{p4.fused} fused")
    reasons = [why for _, why in p4.rejected if "slice bounds" in why]
    check("uneven refusal identifies the bounds", bool(reasons),
          reasons[0] if reasons else "no slice-bounds rejection")

    print("look-alikes that compute something else are left alone:")

    def fused_and_preserved(mod, args):
        e = to_edge(torch.export.export(mod.eval(), args))
        with torch.no_grad():
            before = e.exported_program().module()(*args)
        p = FuseRoPEPass()
        p.call(e.exported_program().graph_module)
        try:
            with torch.no_grad():
                after = e.exported_program().module()(*args)
            same = after.shape == before.shape and torch.equal(after, before)
        except RuntimeError:        # the rewritten graph does not even run
            same = False
        return p.fused, same

    def rotate(r, cs, sn):
        a, b = r.chunk(2, -1)
        return r * cs + torch.cat([-b, a], -1) * sn

    class SuffixRotation(nn.Module):
        """Rotates the LAST four lanes and moves the first four after them."""

        def forward(self, y, cs, sn):
            return torch.cat([rotate(y[..., 4:], cs, sn), y[..., :4]], -1)

    class PositionSlice(nn.Module):
        """Rotates the first four POSITIONS and passes the rest through."""

        def forward(self, y, cs, sn):
            return torch.cat([rotate(y[:, :4], cs, sn), y[:, 4:]], 1)

    class PairwiseOnRows(nn.Module):
        """The pairwise spelling with its two halves stacked on dim 0."""

        def forward(self, y, cs, sn):
            a, b = y.chunk(2, -1)
            return torch.cat([a * cs - b * sn, b * cs + a * sn], 0)

    class ScaledAdd(nn.Module):
        def forward(self, y, cs, sn):
            a, b = y.chunk(2, -1)
            return torch.add(y * cs, torch.cat([-b, a], -1) * sn, alpha=2)

    class ScaledSub(nn.Module):
        def forward(self, y, cs, sn):
            a, b = y.chunk(2, -1)
            return torch.cat([torch.sub(a * cs, b * sn, alpha=2),
                              b * cs + a * sn], -1)

    y3 = torch.randn(1, 6, 8)
    for name, mod, args in (
            ("rotating the suffix, passing the prefix", SuffixRotation(),
             (y3, torch.randn(6, 4), torch.randn(6, 4))),
            ("slicing and joining the position axis", PositionSlice(),
             (y3, torch.randn(4, 8), torch.randn(4, 8))),
            ("pairwise halves joined on dim 0", PairwiseOnRows(),
             (torch.randn(2, 4), torch.randn(2, 2), torch.randn(2, 2))),
            ("rotate-half add with alpha=2", ScaledAdd(), (x, cos, sin)),
            ("pairwise sub with alpha=2", ScaledSub(), (x, c, s))):
        n_fused, same = fused_and_preserved(mod, args)
        check(f"{name}: not fused, output unchanged",
              n_fused == 0 and same, f"{n_fused} fused, output same: {same}")

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("RoPE fuses to one op from either spelling, bit-for-bit")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() records every failed
    check and goes on; this fails if any did, naming them."""
    start = len(FAILS)
    rc = main()
    failed = FAILS[start:]
    assert not failed and rc in (None, 0), f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(main())
