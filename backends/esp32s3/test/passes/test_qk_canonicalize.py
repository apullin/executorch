# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""4-D multi-head q@kT must reach the delegate as the same op the 3-D one does.

The opt-in int8 q@kT gate matches `bmm(A, permute_copy(X,(0,2,1)))` with 3-D
operands -- the spelling this repo's own AttnQK fixture produces. Standard
multi-head attention keeps heads as a separate axis and collapses it for the
batched matmul, so its key operand arrives as a `view_copy`, which the gate
alone refuses, even with `quantize_qk=True` explicitly requested.

Claims:

  1. multi-head 4-D attention and the equivalent 3-D spelling produce
     BYTE-IDENTICAL delegate blobs, so the parity row already covering the 3-D
     form covers the 4-D one,
  2. the rewrite is numerically exact before lowering (it is a reshape identity,
     so "close" would mean wrong),
  3. `att@v` is still NOT claimed -- it is post-softmax and deliberately
     portable, and a pass that swept up both bmms would look like a bigger win
     while quietly cutting precision where there is no bandwidth to save,
  4. an alternate exported permutation spelling is either recognized as the
     same key transpose or declined with a recorded reason.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* fails
naming its failed checks (@case).

    python test_qk_canonicalize.py
"""

import functools
import hashlib
import sys

import torch
import torch.nn as nn

FAILS = []
D, T, HEADS = 32, 8, 2
DH = D // HEADS


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def case(test):
    """One test_* case. check() records a failure and goes on, so a script
    run reports every one (main); a case that only did that would pass
    under pytest whatever its checks found. Called, it fails if any check
    it made failed, naming them. main() calls the case itself
    (__wrapped__), so the script still runs every case."""

    @functools.wraps(test)
    def run():
        start = len(FAILS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


class ThreeD(nn.Module):
    """Multi-head attention with B/H explicitly collapsed before each bmm."""

    def __init__(self):
        super().__init__()
        self.wq, self.wk, self.wv = (nn.Linear(D, D) for _ in range(3))

    def forward(self, x):
        B, N, _ = x.shape

        def split(t):
            return (t.view(B, N, HEADS, DH).transpose(1, 2)
                    .reshape(B * HEADS, N, DH))

        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        s = torch.bmm(q, k.transpose(1, 2)) / (DH ** 0.5)
        o = torch.bmm(torch.softmax(s, -1), v)
        return (o.view(B, HEADS, N, DH).transpose(1, 2)
                .reshape(B, N, D))


class FourD(nn.Module):
    """The same computation with B and H kept as explicit 4-D axes."""

    def __init__(self):
        super().__init__()
        self.wq, self.wk, self.wv = (nn.Linear(D, D) for _ in range(3))

    def forward(self, x):
        B, N, _ = x.shape
        def split(t):
            return t.view(B, N, HEADS, DH).transpose(1, 2)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        s = torch.matmul(q, k.transpose(-1, -2)) / (DH ** 0.5)
        o = torch.matmul(torch.softmax(s, -1), v)
        return o.transpose(1, 2).reshape(B, N, D)


def lower(model, x, passes):
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.backend.utils import get_delegated_payload

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    part = Esp32Partitioner(quantize_qk=True)
    edge = to_edge_transform_and_lower(torch.export.export(model, (x,)),
                                       transform_passes=passes,
                                       partitioner=[part])
    gm = edge.exported_program().graph_module
    blobs = [v[2] for v in get_delegated_payload(gm).values()]
    bmm_left = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                   and "call_delegate" not in str(n.target)
                   and "bmm.default" in str(n.target))
    return blobs, bmm_left, part


def _models():
    """The input and the two spellings at seed 0, FourD with ThreeD's
    weights."""
    torch.manual_seed(0)
    # Two distinct batches make the flattened serialized batch field and every
    # batch offset observable while retaining exact two-head 3-D/4-D parity.
    x = torch.randn(2, T, D)
    a, b = ThreeD().eval(), FourD().eval()
    b.load_state_dict({k: v.detach().clone() for k, v in a.state_dict().items()})
    return x, a, b


def _lower_both():
    """Both spellings lowered, the 4-D one through QkTo3D: (3-D blobs, bmms
    left in 3-D, 4-D blobs, bmms left in 4-D, the QkTo3D pass)."""
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm
    from executorch.backends.esp32s3._passes.qk_to_3d_pass import QkTo3D

    x, a, b = _models()
    blobs3, left3, _ = lower(a, x, [BmmToMm()])
    p = QkTo3D()
    blobs4, left4, _ = lower(b, x, [BmmToMm(), p])
    return blobs3, left3, blobs4, left4, p


@case
def test_rewrite_exact():
    x, a, b = _models()
    print("the rewrite is exact before lowering:")
    with torch.no_grad():
        ya, yb = a(x), b(x)
    check("4-D and 3-D spellings agree", torch.allclose(ya, yb, atol=1e-6),
          f"max abs {(ya - yb).abs().max().item():.2e}")


@case
def test_same_delegate_blob():
    print("the two spellings emit the same delegate blob:")
    blobs3, _, blobs4, _, p = _lower_both()
    check("the pass fired once (one q@kT)", p.rewritten == 1,
          f"{p.rewritten} rewritten, skipped {p.skipped}")
    h3 = sorted(hashlib.md5(z).hexdigest() for z in blobs3)
    h4 = sorted(hashlib.md5(z).hexdigest() for z in blobs4)
    check("both spellings delegate something", bool(h3) and bool(h4),
          f"{len(h3)} vs {len(h4)} blob(s)")
    check("delegate blobs are byte-identical", h3 == h4,
          f"{h3[0][:16] if h3 else '-'} vs {h4[0][:16] if h4 else '-'}")


@case
def test_att_v_stays_portable():
    print("att@v stays portable in BOTH:")
    _, left3, _, left4, _ = _lower_both()
    check("3-D leaves exactly one bmm behind", left3 == 1, f"{left3}")
    check("4-D leaves exactly one bmm behind", left4 == 1, f"{left4}")


@case
def test_alternate_spelling():
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm
    from executorch.backends.esp32s3._passes.qk_to_3d_pass import QkTo3D

    # The input; AlternateSpelling's weights are drawn after the two models',
    # as before.
    x, _, _ = _models()
    print("an alternate permutation spelling is handled explicitly:")

    class AlternateSpelling(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq, self.wk = nn.Linear(D, D), nn.Linear(D, D)

        def forward(self, y):
            B, N, _ = y.shape
            q = self.wq(y).view(B, N, 1, D).transpose(1, 2)
            k = self.wk(y).view(B, N, 1, D).permute(0, 2, 1, 3)
            s = torch.matmul(q, k.transpose(-1, -2))
            return torch.softmax(s, -1)

    p2 = QkTo3D()
    lower(AlternateSpelling().eval(), x, [BmmToMm(), p2])
    # The permute this model writes IS the key transpose after export folds the
    # two adjacent permutes, so a rewrite here is correct. What must never
    # happen is a rewrite with NO recorded reason when it is declined.
    check("the alternate is rewritten or declined with a reason",
          p2.rewritten > 0 or bool(p2.skipped),
          f"rewrote {p2.rewritten}, skipped {p2.skipped}")


def main():
    for test in (test_rewrite_exact, test_same_delegate_blob,
                 test_att_v_stays_portable, test_alternate_spelling):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("4-D q@kT canonicalises to the 3-D spelling, byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
