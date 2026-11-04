# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A batch-1 bmm against a constant weight IS an mm, and must produce the same blob.

The claim is equivalence, and the strongest available evidence for it is not a
numerical tolerance but BYTE IDENTITY: lower the same computation written both
ways and compare the delegate payloads. If they differ by one byte, the two
spellings are not being packed as the same op and the rewrite is unsound --
regardless of how close the outputs happen to land. And if they match, the
device parity that already covers `mm` covers this for free, which is the whole
argument for doing this as a canonicalisation instead of a new delegate op.

Three claims:

  1. a bmm spelling and an mm spelling of one computation emit IDENTICAL
     delegate blobs, and both actually delegate,
  2. the rewrite preserves numerics on the portable path too (a view is free, so
     this must be exact, not close),
  3. a REAL batched matmul is left alone -- attention's q@kT has batch = heads
     and no constant operand, and rewriting it would be silently wrong,
  4. past batch 1, ONE weight matrix broadcast over the batch is still a Linear:
     torch's matmul spells nn.Linear over an input it cannot view as rows (a
     channels-last image) that way, and it emits the row-major input's
     blob; per-batch weights, or a broadcast that
     mixes the matrix's rows, are left alone,
  5. lower_esp32 canonicalises before the constant propagation its conv+bn
     fold needs, which would fold the weight chain into one [B, K, N]
     constant: behind a conv+bn, both bmm spellings still emit the mm
     spelling's blobs.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* fails
naming its failed checks (@case).

    python test_bmm_to_mm.py
"""

import functools
import hashlib
import sys

import torch
import torch.nn as nn

FAILS = []


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


def lower(model, args, passes=None):
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.backend.utils import get_delegated_payload

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    kw = {"partitioner": [Esp32Partitioner()]}
    if passes is not None:
        kw["transform_passes"] = passes
    edge = to_edge_transform_and_lower(torch.export.export(model, args), **kw)
    gm = edge.exported_program().graph_module
    calls = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                and "call_delegate" in str(n.target))
    payload = get_delegated_payload(gm)
    blobs = [v[2] for v in payload.values()]
    return calls, blobs, gm


K, N, M = 32, 48, 6
torch.manual_seed(0)
W = torch.randn(K, N)


class AsMM(nn.Module):
    """2-D activation: torch emits mm directly."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(W.clone())

    def forward(self, x):
        return x @ self.w


class AsBMM(nn.Module):
    """The same numbers, written so torch emits a batch-1 bmm."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(W.clone())

    def forward(self, x):
        return torch.bmm(x.reshape(1, M, K),
                         self.w.unsqueeze(0).expand(1, K, N))


class RealBatched(nn.Module):
    """q@kT: batch is the head count and BOTH operands are activations. Nothing
    here is a Linear, and a pass that rewrote it would be quietly wrong."""

    def forward(self, q, k):
        return torch.bmm(q, k.transpose(1, 2))


class SquareTransposed(nn.Module):
    """x @ W.T with a SQUARE W, the transpose taken after the batch dim was
    added. W and W.T have the same [K,K] shape, so a walk that matches the
    weight by shape alone cannot tell them apart."""

    def __init__(self, spelling):
        super().__init__()
        self.w = nn.Parameter(torch.randn(K, K))
        self.spelling = spelling

    def forward(self, x):
        x3 = x.reshape(1, M, K)
        if self.spelling == "unsqueeze":
            return torch.bmm(x3, self.w.unsqueeze(0).transpose(1, 2))
        if self.spelling == "expand":
            return torch.bmm(x3, self.w.expand(1, K, K).transpose(1, 2))
        return torch.einsum("bmk,nk->bmn", x3, self.w)


B = 2


class AsBatchedBMM(nn.Module):
    """The same numbers as AsMM over B * M rows, written as a batch-B bmm
    against the one weight broadcast over the batch."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(W.clone())

    def forward(self, x):
        return torch.bmm(x.reshape(B, M, K), self.w.unsqueeze(0).expand(B, K, N))


class LinearNoBias(nn.Module):
    """nn.Linear without bias, over whatever layout its input arrives in."""

    def __init__(self, bias=False):
        super().__init__()
        self.fc = nn.Linear(K, N, bias=bias)
        self.fc.weight.data = W.t().contiguous()
        if bias:
            self.fc.bias.data = torch.linspace(-1, 1, N)

    def forward(self, x):
        return self.fc(x)


class PerBatchWeights(nn.Module):
    """A batch-B bmm whose batches read different matrices: not a Linear."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(B, K, N))

    def forward(self, x):
        return torch.bmm(x.reshape(B, M, K), self.w)


class MixedBroadcast(nn.Module):
    """W broadcast between its own K and N and regrouped to [B, K, N]: each
    batch holds rows of W interleaved with copies, so no batch is W."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(K, N))

    def forward(self, x):
        wb = self.w.unsqueeze(1).expand(K, B, N).reshape(B, K, N)
        return torch.bmm(x.reshape(B, M, K), wb)


class BehindConvBn(nn.Module):
    """A conv + BatchNorm beside a Linear spelled one way or another. The
    bn's statistics are seeded so that the fold is not an identity."""

    def __init__(self, inner):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        with torch.no_grad():
            self.bn.running_mean.uniform_(-0.5, 0.5)
            self.bn.running_var.uniform_(0.5, 2.0)
            self.bn.weight.uniform_(0.5, 1.5)
            self.bn.bias.uniform_(-0.5, 0.5)
        self.inner = inner

    def forward(self, img, x):
        return self.bn(self.conv(img)), self.inner(x)


def lower_esp32_blobs(model, args):
    """lower_esp32, the default Esp32Partitioner: the delegate calls, the
    md5 of each blob (sorted), and the ops left outside the delegates."""
    from executorch.exir.backend.utils import get_delegated_payload

    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    edge = lower_esp32(torch.export.export(model, args), [Esp32Partitioner()])
    gm = edge.exported_program().graph_module
    calls = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                and "call_delegate" in str(n.target))
    blobs = sorted(hashlib.md5(v[2]).hexdigest()
                   for v in get_delegated_payload(gm).values())
    left = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"
            and "call_delegate" not in str(n.target)]
    return calls, blobs, left


def _x2():
    return torch.randn(M, K, generator=torch.Generator().manual_seed(1))


def _x4():
    """A rank-4 input, row-major and channels-last."""
    x4 = torch.randn(2, 3, 2, K, generator=torch.Generator().manual_seed(2))
    return x4, x4.contiguous(memory_format=torch.channels_last)


@case
def test_spellings_emit_one_blob():
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm

    x2 = _x2()
    print("the two spellings emit the same blob:")
    mm_calls, mm_blobs, _ = lower(AsMM().eval(), (x2,))
    p = BmmToMm()
    bmm_calls, bmm_blobs, _ = lower(AsBMM().eval(), (x2,), passes=[p])

    check("the pass fired exactly once", p.rewritten == 1,
          f"{p.rewritten} rewritten")
    check("the mm spelling delegates", mm_calls == 1, f"{mm_calls} call(s)")
    check("the bmm spelling delegates too", bmm_calls == 1,
          f"{bmm_calls} call(s)")
    hm = [hashlib.md5(b).hexdigest() for b in mm_blobs]
    hb = [hashlib.md5(b).hexdigest() for b in bmm_blobs]
    check("delegate blobs are byte-identical", hm == hb and bool(hm),
          f"{hm[0][:16] if hm else '-'} vs {hb[0][:16] if hb else '-'} "
          f"({len(mm_blobs[0]) if mm_blobs else 0} B)")


@case
def test_rewrite_exact():
    x2 = _x2()
    print("the rewrite is exact, not approximate:")
    a, b = AsMM().eval(), AsBMM().eval()
    b.load_state_dict({"w": a.w.detach().clone()})
    with torch.no_grad():
        ya, yb = a(x2), b(x2).reshape(M, N)
    check("outputs bit-identical before lowering", torch.equal(ya, yb),
          f"max abs diff {(ya - yb).abs().max().item():.1e}")


@case
def test_batched_matmul_left_alone():
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm

    print("a genuine batched matmul is left alone:")
    q = torch.randn(8, 12, 16)
    k = torch.randn(8, 12, 16)
    p2 = BmmToMm()
    lower(RealBatched().eval(), (q, k), passes=[p2])
    check("q@kT was not rewritten", p2.rewritten == 0,
          f"{p2.rewritten} rewritten; skipped {p2.skipped}")
    check("and the reason is recorded", any("batch != 1" in r
                                            for _, r in p2.skipped))


@case
def test_square_weight_keeps_transpose():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm

    x2 = _x2()
    print("a square weight transposed in 3-D keeps its transpose:")
    for spelling in ("unsqueeze", "expand", "einsum"):
        torch.manual_seed(1)
        model = SquareTransposed(spelling).eval()
        edge = to_edge(torch.export.export(model, (x2,)))
        p4 = BmmToMm()
        edge = edge.transform([p4])
        with torch.no_grad():
            want = model(x2)
            got = edge.exported_program().module()(x2)
        check(f"{spelling}: rewritten graph computes x @ W.T",
              torch.allclose(got, want, atol=1e-5),
              f"max abs diff {(got - want).abs().max().item():.1e}, "
              f"{p4.rewritten} rewritten")


@case
def test_broadcast_weight_is_a_linear():
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm

    print("one weight broadcast over a batch is still a Linear:")
    xb = torch.randn(B * M, K)
    mm_calls, mm_blobs, _ = lower(AsMM().eval(), (xb,))
    p5 = BmmToMm()
    bb_calls, bb_blobs, _ = lower(AsBatchedBMM().eval(), (xb,), passes=[p5])
    check("the batch-2 bmm was rewritten", p5.rewritten == 1,
          f"{p5.rewritten} rewritten; skipped {p5.skipped}")
    check("it delegates, with the mm spelling's blob byte for byte",
          bb_calls == 1 and mm_calls == 1 and bool(mm_blobs)
          and [hashlib.md5(b).hexdigest() for b in bb_blobs]
          == [hashlib.md5(b).hexdigest() for b in mm_blobs],
          f"{bb_calls} call(s)")
    x4, x4_cl = _x4()
    rm_calls, rm_blobs, _ = lower(LinearNoBias().eval(), (x4,))
    cl_ops = [str(n.target) for n in to_edge(torch.export.export(
        LinearNoBias().eval(), (x4_cl,))).exported_program().graph.nodes
        if n.op == "call_function"]
    check("over a channels-last input, torch spells nn.Linear as a batched bmm",
          any("aten.bmm" in t for t in cl_ops)
          and not any("aten.mm" in t for t in cl_ops))
    p6 = BmmToMm()
    cl_calls, cl_blobs, _ = lower(LinearNoBias().eval(), (x4_cl,), passes=[p6])
    check("and it delegates, with the row-major input's blob byte for byte",
          p6.rewritten == 1 and cl_calls == rm_calls == 1 and bool(rm_blobs)
          and [hashlib.md5(b).hexdigest() for b in cl_blobs]
          == [hashlib.md5(b).hexdigest() for b in rm_blobs],
          f"{p6.rewritten} rewritten, {cl_calls} call(s) against {rm_calls}; "
          f"skipped {p6.skipped}")
    rm_calls, rm_blobs, _ = lower(LinearNoBias(bias=True).eval(), (x4,))
    p8 = BmmToMm()
    cl_calls, cl_blobs, cl_gm = lower(LinearNoBias(bias=True).eval(), (x4_cl,),
                                      passes=[p8])
    left = [str(n.target) for n in cl_gm.graph.nodes if n.op == "call_function"
            and "aten.add" in str(n.target)]
    check("with its bias, torch's add folds in: the row-major input's blob, "
          "byte for byte, and no add left outside",
          p8.rewritten == 1 and cl_calls == rm_calls == 1 and bool(rm_blobs)
          and not left and [hashlib.md5(b).hexdigest() for b in cl_blobs]
          == [hashlib.md5(b).hexdigest() for b in rm_blobs],
          f"{p8.rewritten} rewritten, {cl_calls} call(s) against {rm_calls}; "
          f"left outside: {left}")
    for cls in (PerBatchWeights, MixedBroadcast):
        torch.manual_seed(2)
        model = cls().eval()
        edge = to_edge(torch.export.export(model, (xb,)))
        p7 = BmmToMm()
        edge = edge.transform([p7])
        with torch.no_grad():
            want = model(xb)
            got = edge.exported_program().module()(xb)
        check(f"{cls.__name__}: left alone, and the graph computes the model",
              p7.rewritten == 0 and torch.allclose(got, want, atol=1e-5),
              f"{p7.rewritten} rewritten; skipped {p7.skipped}; max abs diff "
              f"{(got - want).abs().max().item():.1e}")


@case
def test_behind_conv_bn():
    x2 = _x2()
    x4, x4_cl = _x4()
    print("behind a conv+bn, lower_esp32 canonicalises before it folds constants:")
    img = torch.randn(1, 3, 8, 8)

    def behind(inner):
        torch.manual_seed(3)
        return BehindConvBn(inner).eval()

    def stray(left):
        return [t for t in left if "bmm" in t or "aten.add" in t or "batch_norm" in t]

    mm_calls, mm_blobs, _ = lower_esp32_blobs(behind(AsMM()), (img, x2))
    b_calls, b_blobs, b_left = lower_esp32_blobs(behind(AsBMM()), (img, x2))
    check("a batch-1 bmm: the mm spelling's blobs byte for byte, no bmm left outside",
          b_calls == mm_calls and bool(mm_blobs) and b_blobs == mm_blobs
          and not stray(b_left),
          f"{b_calls} call(s) against {mm_calls}; left outside: {stray(b_left)}")
    rm_calls, rm_blobs, _ = lower_esp32_blobs(behind(LinearNoBias(bias=True)), (img, x4))
    cl_calls, cl_blobs, cl_left = lower_esp32_blobs(behind(LinearNoBias(bias=True)),
                                                    (img, x4_cl))
    check("nn.Linear over a channels-last input: the row-major input's blobs byte "
          "for byte, no bmm or add left outside",
          cl_calls == rm_calls and bool(rm_blobs) and cl_blobs == rm_blobs
          and not stray(cl_left),
          f"{cl_calls} call(s) against {rm_calls}; left outside: {stray(cl_left)}")


def main():
    for test in (test_spellings_emit_one_blob, test_rewrite_exact,
                 test_batched_matmul_left_alone,
                 test_square_weight_keeps_transpose,
                 test_broadcast_weight_is_a_linear, test_behind_conv_bn):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("bmm-as-linear canonicalises to mm, byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
