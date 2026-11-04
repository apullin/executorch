# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""F.scaled_dot_product_attention must reach the delegate as explicit attention does.

Hugging Face's default attention (``attn_implementation="sdpa"``), timm and
``nn.MultiheadAttention(need_weights=False)`` call SDPA, and ExecuTorch lowers
it through ATen's math decomposition, whose q@kT bmms the gate alone refuses
even with ``quantize_qk=True``: the key's scale sits between the transpose and
the view, and the NaN guard reads the scores twice.

Claims, all with ``quantize_qk=True`` unless stated:

  1. the canonicalisation is exact: the edge program's output is bit-identical
     with and without `SdpaToQk`, for every mask kind,
  2. every SDPA q@kT is delegated -- no mask, is_causal, GQA via enable_gqa,
     nn.MultiheadAttention -- and with no mask its delegate blobs are
     byte-identical to explicit attention written with SDPA's scale placement,
  3. p@v stays portable: exactly one bmm per attention is left behind,
  4. the NaN guard always stays (no row is provably finite before running:
     finite q and k can overflow to -inf), and q@kT is delegated anyway,
  5. with quantize_qk off, no bmm is claimed at all.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* fails
naming its failed checks (@case).

    python test_sdpa_canonicalize.py
"""

import contextlib
import functools
import hashlib
import io
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

FAILS = []
D, T, H = 32, 8, 2
DH = D // H
GUARD_OPS = ("aten.eq.", "aten.any.", "aten.logical_not.", "aten.full_like.")


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


class Attn(nn.Module):
    def __init__(self, mode="none", kv_heads=H):
        super().__init__()
        self.mode, self.kv = mode, kv_heads
        self.q = nn.Linear(D, H * DH)
        self.k = nn.Linear(D, kv_heads * DH)
        self.v = nn.Linear(D, kv_heads * DH)
        self.register_buffer("keep", torch.ones(T, T, dtype=torch.bool).tril(),
                             persistent=False)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, H, DH).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv, DH).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv, DH).transpose(1, 2)
        kw = {"enable_gqa": True} if self.kv != H else {}
        if self.mode == "causal":
            kw["is_causal"] = True
        elif self.mode == "buffer":
            kw["attn_mask"] = self.keep
        elif self.mode == "runtime":
            kw["attn_mask"] = mask
        o = F.scaled_dot_product_attention(q, k, v, **kw)
        return o.transpose(1, 2).reshape(b, t, D)


class Explicit(nn.Module):
    """The no-mask attention written out, scaling q and k by sqrt(scale) as
    the SDPA decomposition does."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        b, t, _ = x.shape
        s = DH ** -0.25
        q = self.q(x).view(b, t, H, DH).transpose(1, 2)
        k = self.k(x).view(b, t, H, DH).transpose(1, 2)
        v = self.v(x).view(b, t, H, DH).transpose(1, 2)
        p = torch.softmax(torch.matmul(q * s, k.transpose(-1, -2) * s), -1)
        return torch.matmul(p, v).transpose(1, 2).reshape(b, t, D)


class LateScale(nn.Module):
    """Explicit attention whose key scale is computed AFTER the key transpose
    (from q, at run time): the moved multiply must not use the scale above its
    definition."""

    def __init__(self):
        super().__init__()
        self.q, self.k = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, H, DH).transpose(1, 2)
        kt = self.k(x).view(b, t, H, DH).transpose(1, 2).transpose(-1, -2)
        s = q.abs().mean() + 1.0
        return torch.softmax(torch.matmul(q, kt * s), -1)


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(D, H, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


def passes(with_sdpa=True):
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm
    from executorch.backends.esp32s3._passes.qk_to_3d_pass import QkTo3D
    from executorch.backends.esp32s3._passes.remove_identity_expand_pass import (
        RemoveIdentityExpand,
    )
    from executorch.backends.esp32s3._passes.sdpa_to_qk_pass import SdpaToQk

    sdpa = SdpaToQk()
    middle = [sdpa] if with_sdpa else []
    return [BmmToMm(), *middle, QkTo3D(), RemoveIdentityExpand()], sdpa


def lower(model, args, quantize_qk=True):
    """Lower through the shared pipeline; returns (portable targets, blobs, pass)."""
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.backend.utils import get_delegated_payload

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    plist, sdpa = passes()
    with contextlib.redirect_stdout(io.StringIO()):
        edge = to_edge_transform_and_lower(
            torch.export.export(model.eval(), args), transform_passes=plist,
            partitioner=[Esp32Partitioner(quantize_qk=quantize_qk)])
    gm = edge.exported_program().graph_module
    portable = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"
                and "executorch_call_delegate" not in str(n.target)]
    blobs = sorted(hashlib.md5(v[2]).hexdigest()
                   for v in get_delegated_payload(gm).values())
    return portable, blobs, sdpa


def n_bmm(portable):
    return sum("aten.bmm." in t for t in portable)


def edge_output(model, args, with_sdpa):
    from executorch.exir import to_edge

    plist, _ = passes(with_sdpa)
    edge = to_edge(torch.export.export(model.eval(), args)).transform(plist)
    with torch.no_grad():
        return edge.exported_program().module()(*args)


def _cases():
    """The input at seed 0, and the attention cases by name, each a model
    and its inputs."""
    torch.manual_seed(0)
    x = torch.randn(1, T, D)
    runtime_mask = torch.zeros(1, 1, T, T).masked_fill(
        ~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    cases = {
        "no mask": (Attn("none"), (x,)),
        "is_causal": (Attn("causal"), (x,)),
        "GQA (enable_gqa, 1 kv head)": (Attn("none", kv_heads=1), (x,)),
        "bool mask buffer": (Attn("buffer"), (x,)),
        "runtime float mask": (Attn("runtime"), (x, runtime_mask)),
        "nn.MultiheadAttention": (MHA(), (x,)),
    }
    return x, cases


@case
def test_rewrite_exact():
    _, cases = _cases()
    print("the rewrite is exact:")
    for name, (model, args) in cases.items():
        a = edge_output(model, args, with_sdpa=False)
        b = edge_output(model, args, with_sdpa=True)
        check(f"{name}: bit-identical with and without SdpaToQk",
              torch.equal(a, b), f"max abs {(a - b).abs().max().item():.1e}")


@case
def test_qk_delegated_pv_not():
    _, cases = _cases()
    print("q@kT is delegated, p@v is not:")
    for name, (model, args) in cases.items():
        portable, _, sdpa = lower(model, args)
        check(f"{name}: one bmm left (p@v)", n_bmm(portable) == 1,
              f"{n_bmm(portable)} portable bmm; skipped {sdpa.skipped}")


@case
def test_nan_guard_stays():
    _, cases = _cases()
    print("the NaN guard always stays, and q@kT still delegates:")
    for name, (model, args) in cases.items():
        portable, _, sdpa = lower(model, args)
        guards = [t for t in portable if any(g in t for g in GUARD_OPS)]
        check(f"{name}: guard kept", bool(guards) and n_bmm(portable) == 1,
              f"guard ops {len(guards)}, portable bmm {n_bmm(portable)}")


@case
def test_late_key_scale():
    # The input; LateScale's weights are drawn after the cases', as before.
    x, _ = _cases()
    print("a key scale defined after the transpose:")
    late = LateScale()
    try:
        a = edge_output(late, (x,), with_sdpa=False)
        b = edge_output(late, (x,), with_sdpa=True)
        check("LateScale: rewrites without a use-before-definition, bit-identical",
              torch.equal(a, b), f"max abs {(a - b).abs().max().item():.1e}")
    except Exception as e:  # noqa: BLE001 -- use-before-definition fails graph.lint()
        check("LateScale: rewrites without a use-before-definition", False, repr(e)[:200])


@case
def test_blob_identity_with_explicit():
    x, _ = _cases()
    print("blob identity with the explicit spelling:")
    model, explicit = Attn("none"), Explicit()
    explicit.load_state_dict({k: v for k, v in model.state_dict().items()
                              if k != "keep"})
    _, blobs_sdpa, _ = lower(model, (x,))
    _, blobs_explicit, _ = lower(explicit, (x,))
    check("SDPA and explicit attention emit the same delegate blobs",
          bool(blobs_sdpa) and blobs_sdpa == blobs_explicit,
          f"{len(blobs_sdpa)} vs {len(blobs_explicit)} blob(s)")


@case
def test_quantize_qk_off():
    x, _ = _cases()
    print("quantize_qk off claims no bmm:")
    portable, _, _ = lower(Attn("causal"), (x,), quantize_qk=False)
    check("both bmms stay portable", n_bmm(portable) == 2, f"{n_bmm(portable)}")


def main():
    for test in (test_rewrite_exact, test_qk_delegated_pv_not,
                 test_nan_guard_stays, test_late_key_scale,
                 test_blob_identity_with_explicit, test_quantize_qk_off):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("SDPA reaches the q@kT delegate exactly as explicit attention does")
    return 0


if __name__ == "__main__":
    sys.exit(main())
