# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whatever the partition gate claims, preprocess lowers.

A raise in preprocess ("unsupported quantised-weight shape", as an Arm or
Cortex-M int32 bias would cause) fails the WHOLE lowering where a refusal
leaves one op portable. So the gate runs preprocess's own weight checks first
(espressif_operator_support._check_lowerable), and this holds for every
spelling here: each model lowers without an exception, and an op preprocess
cannot take is refused by the gate under a named kind instead. Preprocess is
also wrapped, so a raise there is reported as itself even if a later stage
would have masked it.

The same holds for the ops preprocess folds into a weight, bias or table at
build time (a transpose, a slice, a dequantise): the gate claims an op only
with all of them, so one that something unclaimed also reads leaves the op
portable rather than handing preprocess a runtime weight. One that only
matmuls read is copied for each of them before partitioning, so readers
grouped into different partitions each fold their own (an RNN hoisting
W.t() out of its loop).
"""

import contextlib
import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3 import espressif_preprocess
from executorch.backends.esp32s3.partition import espressif_partitioner
from executorch.backends.esp32s3.operator_support import const_weights


def _decomposed():
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    return torch.ops.quantized_decomposed


def _torchao():
    import torchao  # noqa: F401

    return torch.ops.torchao


class Weighted(nn.Module):
    """y = x @ W^T (+ b) with W (and b) built in forward from quantised
    buffers by `weight_fn(self)` (and `bias_fn(self)`)."""

    def __init__(self, buffers, weight_fn, bias_fn=None):
        super().__init__()
        for name, value in buffers.items():
            self.register_buffer(name, value)
        self.weight_fn, self.bias_fn = weight_fn, bias_fn

    def forward(self, x):
        b = self.bias_fn(self) if self.bias_fn else None
        return F.linear(x, self.weight_fn(self), b)


class SharedFold(nn.Module):
    """(use(self, x, f), also(self, x, f)) with f = fold(self): an op on the
    module's tensors (nn.Parameters stay parameters, the rest are buffers)
    that preprocess folds into the weight, bias or table `use` reads, and a
    second reader of it, by default the graph output itself."""

    def __init__(self, tensors, fold, use, also=None):
        super().__init__()
        for name, value in tensors.items():
            if isinstance(value, nn.Parameter):
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)
        self.fold, self.use = fold, use
        self.also = also or (lambda m, x, f: f)

    def forward(self, x):
        f = self.fold(self)
        return self.use(self, x, f), self.also(self, x, f)


def _codes(n, k, lo=-128, hi=127, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi + 1, (n, k), generator=g, dtype=torch.int8)


def _corpus():
    """(name, model, input, refusal kind the gate must name, or None)."""
    qd, ao = _decomposed(), _torchao()
    cases = []
    # Arm's int32 per-channel bias over per-channel int8 weights: delegated.
    cases.append(("Arm int32 bias", Weighted(
        dict(c=_codes(16, 32), s=torch.full((16,), 0.01),
             z=torch.zeros(16, dtype=torch.int64),
             bc=torch.randint(-2 ** 20, 2 ** 20, (16,), dtype=torch.int32),
             bs=torch.full((16,), 1e-6), bz=torch.zeros(16, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel(m.c, m.s, m.z, 0, -128, 127,
                                            torch.int8),
        lambda m: qd.dequantize_per_channel(m.bc, m.bs, m.bz, 0, -2 ** 31,
                                            2 ** 31 - 1, torch.int32)),
        torch.randn(2, 32), None))
    # A per-tensor int32 bias beside per-tensor weights (Cortex-M's Linear).
    cases.append(("per-tensor weight and int32 bias", Weighted(
        dict(c=_codes(16, 32), bc=torch.randint(-2 ** 20, 2 ** 20, (16,),
                                                dtype=torch.int32)),
        lambda m: qd.dequantize_per_tensor(m.c, 0.01, 3, -128, 127,
                                           torch.int8),
        lambda m: qd.dequantize_per_tensor(m.bc, 1e-6, 0, -2 ** 31,
                                           2 ** 31 - 1, torch.int32)),
        torch.randn(2, 32), None))
    # A fractional float zero point on a matmul: no exact integer form.
    cases.append(("fractional float zero point", Weighted(
        dict(c=_codes(16, 64, -8, 7), s=torch.full((16, 2), 0.01),
             z=torch.full((16, 2), 0.5)),
        lambda m: ao.dequantize_affine(m.c, [1, 32], m.s, m.z, torch.int8,
                                       -8, 7)),
        torch.randn(2, 64), "fractional float zero point on a matmul weight"))
    # Zero points outside the codes' range break the exactness bound's
    # |c - zw| <= 15.
    cases.append(("int4 zero point outside the code range", Weighted(
        dict(c=_codes(16, 64, -8, 7), s=torch.full((16, 2), 0.01),
             z=torch.full((16, 2), 12, dtype=torch.int8)),
        lambda m: ao.dequantize_affine(m.c, [1, 32], m.s, m.z, torch.int8,
                                       -8, 7)),
        torch.randn(2, 64), "weight code or zero-point range unsupported"))
    # Groups of 8 do not tile the int8 kernel's 16-byte steps.
    cases.append(("int8 groups of 8", Weighted(
        dict(c=_codes(16, 64), s=torch.full((16 * 8,), 0.01),
             z=torch.zeros(16 * 8, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel_group(m.c, m.s, m.z, -128, 127,
                                                  torch.int8, 8,
                                                  torch.float32)),
        torch.randn(2, 64), "pass-through group size unsupported"))
    # Per channel along the wrong axis.
    cases.append(("per-channel axis 1", Weighted(
        dict(c=_codes(16, 32), s=torch.full((32,), 0.01),
             z=torch.zeros(32, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel(m.c, m.s, m.z, 1, -128, 127,
                                            torch.int8)),
        torch.randn(2, 32), "quantised-weight layout unsupported"))
    # One group past the int8 zero-point exactness bound (66,304).
    k = 66_336
    cases.append(("int8 zero points one group past the bound", Weighted(
        dict(c=_codes(1, k), s=torch.full((1,), 0.01),
             z=torch.full((1,), 5, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel(m.c, m.s, m.z, 0, -128, 127,
                                            torch.int8)),
        torch.randn(1, k), "affine weight past the exactness bound"))
    # A tinygemm chain whose mid point is not an integer.
    def tinygemm_half(m):
        from torchao.quantization.quant_primitives import (
            _dequantize_affine_tinygemm)

        return _dequantize_affine_tinygemm(m.q, [1, 32], m.s, m.m,
                                           torch.int32, 0, 14)

    cases.append(("tinygemm with a fractional mid point", Weighted(
        dict(q=torch.randint(0, 15, (16, 64), dtype=torch.int32),
             s=torch.full((16, 2), 0.01), m=torch.zeros(16, 2)),
        tinygemm_half),
        torch.randn(2, 64), "float-offset weight layout unsupported"))
    # Codes declaring int4 but holding int8 values: wrapping them modulo 16
    # would be a different model.
    cases.append(("codes outside their declared range", Weighted(
        dict(c=_codes(16, 64), s=torch.full((16, 2), 0.01),
             z=torch.zeros(16, 2)),
        lambda m: ao.dequantize_affine(m.c, [1, 32], m.s, m.z, torch.int8,
                                       -8, 7)),
        torch.randn(2, 64), "weight codes outside their declared range"))
    # Scales that are not float32. The dequant multiplies in the
    # scale's dtype, so eager rounds each weight through it (fl32(fl64(...))
    # for float64, a float16 product for float16); one float32 scale per
    # group cannot reproduce that.
    cases.append(("torchao float64 weight scales", Weighted(
        dict(c=_codes(16, 64, -8, 7),
             s=torch.full((16, 2), 0.01, dtype=torch.float64),
             z=torch.zeros(16, 2, dtype=torch.int8)),
        lambda m: ao.dequantize_affine(m.c, [1, 32], m.s, m.z, torch.int8,
                                       -8, 7)),
        torch.randn(2, 64), "weight scale not float32"))
    cases.append(("PT2E per-channel float64 scales", Weighted(
        dict(c=_codes(16, 32), s=torch.full((16,), 0.01, dtype=torch.float64),
             z=torch.zeros(16, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel(m.c, m.s, m.z, 0, -128, 127,
                                            torch.int8)),
        torch.randn(2, 32), "weight scale not float32"))
    cases.append(("PT2E per-group float16 scales", Weighted(
        dict(c=_codes(16, 64), s=torch.full((16 * 4,), 0.01,
                                            dtype=torch.float16),
             z=torch.zeros(16 * 4, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel_group(m.c, m.s, m.z, -128, 127,
                                                  torch.int8, 16,
                                                  torch.float32)),
        torch.randn(2, 64), "weight scale not float32"))
    # int() would truncate a non-integral bound.
    cases.append(("torchao non-integral code range", Weighted(
        dict(c=_codes(16, 64, -8, 7), s=torch.full((16, 2), 0.01),
             z=torch.zeros(16, 2, dtype=torch.int8)),
        lambda m: ao.dequantize_affine(m.c, [1, 32], m.s, m.z, torch.int8,
                                       -8.5, 7.5)),
        torch.randn(2, 64), "weight code range not integral"))
    # An op folded into a constant operand that something unclaimed also
    # reads stays portable (the delegate cannot hand it out at run time), so
    # the op it folds into must too. First a parameter's transpose returned
    # beside the 1-D matmul it feeds; then the same transpose read by a
    # portable op instead.
    shared = "folded operand shared with an unclaimed consumer"
    cases.append(("transposed weight also an output", SharedFold(
        dict(w=nn.Parameter(torch.randn(23, 1))),
        lambda m: m.w.transpose(1, 0), lambda m, x, t: torch.matmul(x, t)),
        torch.randn(1), shared))
    cases.append(("transposed weight also read by abs", SharedFold(
        dict(w=nn.Parameter(torch.randn(23, 1))),
        lambda m: m.w.transpose(1, 0), lambda m, x, t: torch.matmul(x, t),
        lambda m, x, t: torch.abs(t)),
        torch.randn(1), shared))
    cases.append(("sliced bias also an output", SharedFold(
        dict(w=nn.Parameter(torch.randn(16, 32)),
             b=nn.Parameter(torch.randn(24))),
        lambda m: m.b[:16], lambda m, x, b: F.linear(x, m.w, b)),
        torch.randn(2, 32), shared))
    # The weight's transpose is claimable; the dequantise under it is not.
    cases.append(("dequantised weight also an output", SharedFold(
        dict(c=_codes(16, 32), s=torch.full((16,), 0.01),
             z=torch.zeros(16, dtype=torch.int64)),
        lambda m: qd.dequantize_per_channel(m.c, m.s, m.z, 0, -128, 127,
                                            torch.int8),
        lambda m, x, w: F.linear(x, w)),
        torch.randn(2, 32), shared))
    # A float-offset weight (codes * scale + offset): its root also an
    # output, then a member of the chain (the scaled codes) instead.
    cases.append(("float-offset weight also an output", SharedFold(
        dict(q=_codes(16, 32), s=torch.full((16, 1), 0.01),
             o=torch.full((16, 1), 0.5)),
        lambda m: m.q.float() * m.s + m.o, lambda m, x, w: F.linear(x, w)),
        torch.randn(2, 32), shared))
    cases.append(("float-offset scaled codes also an output", SharedFold(
        dict(q=_codes(16, 32), s=torch.full((16, 1), 0.01),
             o=torch.full((16, 1), 0.5)),
        lambda m: m.q.float() * m.s, lambda m, x, qs: F.linear(x, qs + m.o)),
        torch.randn(2, 32), shared))
    # Both readers are matmuls, so each gets its own transpose
    # (split_shared_folds): the plain one delegates with it, and the gate
    # refuses the scaled one for its beta alone.
    cases.append(("transpose shared with a scaled addmm", SharedFold(
        dict(w=nn.Parameter(torch.randn(16, 32)),
             b=nn.Parameter(torch.randn(16))),
        lambda m: m.w.t(), lambda m, x, t: x @ t,
        lambda m, x, t: torch.addmm(m.b, x, t, beta=0.5)),
        torch.randn(2, 32), "scaled addmm"))
    cases.append(("sliced conv weight also an output", SharedFold(
        dict(w=nn.Parameter(torch.randn(12, 4, 3, 3))), lambda m: m.w[:8],
        lambda m, x, w: F.conv2d(x, w, padding=1)),
        torch.randn(1, 4, 6, 6), shared))
    cases.append(("sliced embedding table also an output", SharedFold(
        dict(t=nn.Parameter(torch.randn(48, 16))), lambda m: m.t[:32],
        lambda m, ids, t: F.embedding(ids, t)),
        torch.randint(0, 32, (1, 5)), shared))
    return cases


CORPUS = _corpus()


@pytest.mark.parametrize("name,model,x,refusal", CORPUS,
                         ids=[c[0] for c in CORPUS])
def test_claimed_ops_lower(monkeypatch, name, model, x, refusal):
    from executorch.backends.esp32s3._passes import lower_esp32

    raised = []
    original = espressif_preprocess.Esp32Backend.preprocess

    def preprocess(edge_program, compile_specs):
        try:
            return original(edge_program, compile_specs)
        except Exception as error:   # the failure this test exists to catch
            raised.append(error)
            raise

    monkeypatch.setattr(espressif_preprocess.Esp32Backend, "preprocess", staticmethod(preprocess))
    partitioner = espressif_partitioner.Esp32Partitioner()
    ep = torch.export.export(model.eval(), (x,))
    with contextlib.redirect_stdout(io.StringIO()):
        lower_esp32(ep, [partitioner])
    assert not raised, f"{name}: preprocess raised {raised[0]!r}"
    kinds = partitioner.support.kinds
    if refusal is None:
        assert not any("unsupported" in k or "fractional" in k for k in kinds), \
            dict(kinds)
    else:
        assert kinds.get(refusal), (
            f"{name}: expected the gate to refuse it as {refusal!r}; "
            f"refusals {dict(kinds)}")


def test_shared_fold_claimed_with_its_readers():
    """The shared fold's controls: a weight's transpose that only claimed ops read
    -- two matmuls sharing it, or two slices of it, one per matmul -- is
    claimed with them, and they still delegate."""
    from executorch.backends.esp32s3._passes import lower_esp32

    folded = ("aten.mm.", "aten.permute_copy.", "aten.slice_copy.")
    for name, use, also in (
            ("two matmuls", lambda m, x, t: x @ t,
             lambda m, x, t: (2.0 * x) @ t),
            ("two slices", lambda m, x, t: x @ t[:, :8],
             lambda m, x, t: x @ t[:, 8:16])):
        model = SharedFold(dict(w=nn.Parameter(torch.randn(16, 32))),
                           lambda m: m.w.t(), use, also)
        partitioner = espressif_partitioner.Esp32Partitioner()
        ep = torch.export.export(model.eval(), (torch.randn(2, 32),))
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(ep, [partitioner])
        portable = [n.name for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                    and any(op in str(n.target) for op in folded)]
        assert not portable, f"{name}: left portable {portable}"
        assert not partitioner.support.kinds.get(
            "folded operand shared with an unclaimed consumer"), (
            f"{name}: {dict(partitioner.support.kinds)}")


class HoistedTranspose(nn.Module):
    """t = w.t() once, read by two matmuls with a portable op between them
    (a recurrent cell's shape: two partitions share the hoisted transpose);
    `hoist` False takes the transpose at each use instead."""

    def __init__(self, hoist=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(16, 16, generator=torch.Generator()
                                          .manual_seed(3)))
        self.hoist = hoist

    def forward(self, x):
        t = self.w.t()
        h = torch.cos(x @ t)
        return h @ (t if self.hoist else self.w.t())


class HoistedRNN(nn.Module):
    """An RNN cell unrolled three steps, h = tanh(x_t Wx^T + h Wh^T), with
    both transposes taken once before the loop (`hoist`) or at each step."""

    def __init__(self, hoist=True):
        super().__init__()
        g = torch.Generator().manual_seed(4)
        self.wx = nn.Parameter(torch.randn(16, 16, generator=g) * 0.3)
        self.wh = nn.Parameter(torch.randn(16, 16, generator=g) * 0.3)
        self.hoist = hoist

    def forward(self, xs, h):                         # [3, 1, 16], [1, 16]
        tx, th = self.wx.t(), self.wh.t()
        for step in range(3):
            if not self.hoist:
                tx, th = self.wx.t(), self.wh.t()
            h = torch.tanh(xs[step] @ tx + h @ th)
        return h


def test_shared_fold_across_partitions():
    """A transpose matmuls in different partitions share: each gets its own
    (split_shared_folds), so the export completes -- without the copies
    preprocess finds the second partition reading the transposed weight as
    a runtime input and raises "needs the weight to be a constant" -- every
    matmul delegates,
    and the delegate blobs are, byte for byte, the ones of the same model
    taking the transpose at each use. Through lower_esp32 and through
    static_act's two-step lowering (calibrate_static_act splits them)."""
    import hashlib

    from executorch.exir.backend.utils import get_delegated_payload

    from executorch.backends.esp32s3._passes import lower_esp32

    def lowered(model, args, flow):
        ep = torch.export.export(model.eval(), args)
        with contextlib.redirect_stdout(io.StringIO()):
            if flow == "lower_esp32":
                partitioner = espressif_partitioner.Esp32Partitioner()
                edge = lower_esp32(ep, [partitioner])
            else:
                # static_act's documented two-step lowering
                # (quantizer/calibration.py).
                from executorch.exir import to_edge_transform_and_lower

                from executorch.backends.esp32s3.quantizer.calibration import (
                    calibrate_static_act,
                )

                partitioner = espressif_partitioner.Esp32Partitioner(
                    static_act=True)
                edge = to_edge_transform_and_lower(ep)
                calibrate_static_act(edge, [args])
                edge = edge.to_backend(partitioner)
        gm = edge.exported_program().graph_module
        left = [n.name for n in gm.graph.nodes if n.op == "call_function"
                and "aten.mm" in str(n.target)]
        blobs = sorted(hashlib.md5(v[2]).hexdigest()
                       for v in get_delegated_payload(gm).values())
        return left, blobs, partitioner.support.kinds

    for name, cls, args, calls in (
            ("HoistedTranspose", HoistedTranspose, (torch.randn(2, 16),), 2),
            ("HoistedRNN", HoistedRNN, (torch.randn(3, 1, 16),
                                        torch.randn(1, 16)), 3)):
        for flow in ("lower_esp32", "static_act"):
            left, blobs, kinds = lowered(cls(hoist=True), args, flow)
            _, want, _ = lowered(cls(hoist=False), args, flow)
            assert not left, f"{name}, {flow}: matmuls left portable: {left}"
            assert len(blobs) >= calls - 1 and blobs == want, (
                f"{name}, {flow}: {len(blobs)} delegate(s), blobs equal the "
                f"unshared spelling's: {blobs == want}; refusals {dict(kinds)}")


def test_shared_fold_split_by_grouping_stays_portable():
    """Partitioned directly (to_edge_transform_and_lower, no
    split_shared_folds), the hoisted transpose's graph still groups the two matmuls
    into two partitions; the partitioner then refuses the transpose and
    both its readers by name and groups again, so the export completes and
    the program computes eager's numbers on the portable kernels instead of
    preprocess raising "needs the weight to be a constant"."""
    import numpy as np
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
        KIND_FOLD_SPLIT,
    )
    from executorch.backends.esp32s3.test.test_partition_crashes import run_host

    model, x = HoistedTranspose().eval(), torch.randn(2, 16)
    partitioner = espressif_partitioner.Esp32Partitioner()
    with contextlib.redirect_stdout(io.StringIO()):
        edge = to_edge_transform_and_lower(torch.export.export(model, (x,)),
                                           partitioner=[partitioner])
        et = edge.to_executorch(ExecutorchBackendConfig(delegate_alignment=16))
    kinds = partitioner.support.kinds
    assert kinds.get(KIND_FOLD_SPLIT, 0) >= 3, dict(kinds)
    got, log = run_host(et, (x,))
    assert got is not None, log[-1000:]
    with torch.no_grad():
        want = model(x).numpy().reshape(-1)
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max(), log[-500:]


def test_folding_skips_refuses_an_unknown_executorch(monkeypatch):
    """The gate's constant set (_folded_constants) is constant
    propagation's own, so it reads constant_prop_pass's
    _DEFAULT_SKIP_TARGETS; an ExecuTorch without it is refused by name, not
    guessed at by matching op names."""
    from executorch.exir.passes import constant_prop_pass

    from executorch.backends.esp32s3.operator_support import (
        espressif_operator_support as gate,
    )

    assert gate._folding_skips() == set(constant_prop_pass._DEFAULT_SKIP_TARGETS)
    monkeypatch.delattr(constant_prop_pass, "_DEFAULT_SKIP_TARGETS")
    with pytest.raises(ImportError, match="_DEFAULT_SKIP_TARGETS"):
        gate._folding_skips()


def test_conv_bn_fold_reads_the_gates_constants(monkeypatch):
    """fold_conv_bn asks what constant propagation computes
    through the gate's own set, _folded_constants: an ExecuTorch without
    _DEFAULT_SKIP_TARGETS is refused by name in the fold as in the gate. A
    float conv's first BatchNorm folds; a second reads the product the fold
    inserted, which the set (read before the fold) does not hold, so it
    stays portable."""
    from executorch.exir import to_edge
    from executorch.exir.passes import constant_prop_pass

    from executorch.backends.esp32s3._passes.fold_conv_bn_pass import fold_conv_bn

    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4),
                          nn.BatchNorm2d(4)).eval()
    x = torch.randn(1, 3, 8, 8)
    folded, rejected = fold_conv_bn(to_edge(torch.export.export(model, (x,))))
    assert folded == 1 and [why for _, why in rejected] == [
        "an operand is not a build-time constant"], (folded, rejected)
    edge = to_edge(torch.export.export(model, (x,)))
    monkeypatch.delattr(constant_prop_pass, "_DEFAULT_SKIP_TARGETS")
    with pytest.raises(ImportError, match="equivalent in _folding_skips"):
        fold_conv_bn(edge)


def _edge(model, x):
    from executorch.exir import to_edge

    return to_edge(torch.export.export(model.eval(), (x,))).exported_program()


def _node(ep, name):
    return next(n for n in ep.graph.nodes
                if n.op == "call_function" and name in str(n.target))


def test_weight_dequant_output_dtype_refused():
    """A weight dequantised to another dtype than float32 is
    refused by name (a matmul cannot take one beside float32 activations,
    so the gate's runtime-dtype rule usually names it first; the resolver
    refuses it on its own)."""
    ao = _torchao()
    model = Weighted(
        dict(c=_codes(16, 64, -8, 7), s=torch.full((16, 2), 0.01),
             z=torch.zeros(16, 2, dtype=torch.int8)),
        lambda m: ao.dequantize_affine(m.c, [1, 32], m.s, m.z, torch.int8,
                                       -8, 7, output_dtype=torch.bfloat16
                                       ).float())
    ep = _edge(model, torch.randn(2, 64))
    with pytest.raises(const_weights.LoweringRefused) as why:
        const_weights._resolve_const(ep, _node(ep, "dequantize_affine"))
    assert why.value.kind == "weight dequant output not float32", why.value


def test_pt2e_per_tensor_parameters_are_scalars():
    """A per-tensor scale or zero point is one value (eager's
    dequantize_per_tensor.tensor raises otherwise); the resolver refuses a
    longer one by name instead of taking its first element."""
    qd = _decomposed()
    model = Weighted(
        dict(c=_codes(16, 32), s=torch.tensor([0.01]),
             z=torch.tensor([3], dtype=torch.int64)),
        lambda m: qd.dequantize_per_tensor.tensor(m.c, m.s, m.z, -128, 127,
                                                  torch.int8))
    ep = _edge(model, torch.randn(2, 32))
    dq = _node(ep, "dequantize_per_tensor")
    assert isinstance(const_weights._resolve_const(ep, dq), const_weights.PassThroughWeight)
    sig = ep.graph_signature
    for arg, value in ((dq.args[1], torch.tensor([0.01, 0.02])),
                       (dq.args[2], torch.tensor([3, 4], dtype=torch.int64))):
        name = sig.inputs_to_buffers[arg.name]
        kept = ep.state_dict[name]
        ep.state_dict[name] = value
        try:
            with pytest.raises(const_weights.LoweringRefused) as why:
                const_weights._resolve_const(ep, dq)
            assert why.value.kind == "per-tensor weight parameter not a scalar"
        finally:
            ep.state_dict[name] = kept


class SlicedWeight(nn.Module):
    """x @ W.t()[:, cols]: a constant weight's transpose sliced along its
    output dim, or (`codes`) int8 codes dequantised per channel and sliced
    along K before F.linear's transpose."""

    def __init__(self, cols, codes=False):
        super().__init__()
        self.cols, self.codes = cols, codes
        if codes:
            self.register_buffer("c", _codes(16, 32))
            self.register_buffer("s", torch.full((16,), 0.01))
            self.register_buffer("z", torch.zeros(16, dtype=torch.int64))
        else:
            self.w = nn.Parameter(torch.randn(24, 16))

    def forward(self, x):
        if self.codes:
            w = _decomposed().dequantize_per_channel(self.c, self.s, self.z, 0,
                                                     -128, 127, torch.int8)
            return F.linear(x, w[:, self.cols])
        return x @ self.w.t()[:, self.cols]


SLICES = {"8:": slice(8, None), "-16:": slice(-16, None),
          "8:100": slice(8, 100), ":-8": slice(None, -8)}


@pytest.mark.parametrize("spelling", list(SLICES))
def test_constant_slice_bounds_read_as_aten_does(spelling):
    """_resolve_const reads a constant slice's bounds as aten.slice does: an
    open end is the dim's size, a negative bound counts from its end, and
    each is clamped into the dim: torch.export's end INT64_MAX for an
    open-ended slice (`W.t()[:, 8:]`) is not past the dim, nor is a negative
    end (`[:, :-8]`, which torch.export keeps) before the start. The gate
    does not resolve a float weight, so a misread bound fails the export in
    preprocess."""
    model = SlicedWeight(SLICES[spelling]).eval()
    ep = _edge(model, torch.randn(2, 16))
    weight = _node(ep, "aten.mm.").args[1]
    got = const_weights._resolve_const(ep, weight)
    assert torch.equal(got, model.w.detach().t()[:, SLICES[spelling]])


def test_pass_through_slice_with_a_negative_end_is_a_prefix():
    """The same for codes consumed verbatim: a column prefix spelled with a
    negative end (`[:, :-16]` of 32 columns) is the first 16 columns, as
    aten reads it, not an unsupported slice that would leave the op
    portable."""
    model = SlicedWeight(slice(None, -16), codes=True).eval()
    ep = _edge(model, torch.randn(2, 16))
    pt = const_weights._resolve_const(ep, _node(ep, "slice_copy"))
    assert isinstance(pt, const_weights.PassThroughWeight)
    assert pt.logical_cols == 16 and tuple(pt.codes.shape) == (16, 32)


def test_open_ended_constant_slice_lowers_and_runs(tmp_path):
    """x @ W.t()[:, 8:] lowers with its matmul delegated, under the default
    policy and the fp32 one, and the fp32 loop computes eager's numbers on
    the host runner (fp32 sums in another order)."""
    import subprocess

    import numpy as np
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.test.host import host_runner

    runner = host_runner()
    if not runner.is_file():
        pytest.fail(f"host runner not built: {runner}")
    torch.manual_seed(8)
    model, x = SlicedWeight(slice(8, None)).eval(), torch.randn(2, 16)
    with torch.no_grad():
        want = model(x).numpy().reshape(-1)
    for options in ({}, {"format_policy": "fp32"}):
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(torch.export.export(model, (x,)),
                               [espressif_partitioner.Esp32Partitioner(**options)])
            apply_passes(edge, quiet=True)
        ops = [str(n.target) for n in edge.exported_program().graph.nodes
               if n.op == "call_function"]
        assert (any("executorch_call_delegate" in t for t in ops)
                and not any("aten.mm." in t for t in ops)), (options, ops)
    pte = tmp_path / "model.pte"
    pte.write_bytes(edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16)).buffer)
    x.numpy().tofile(tmp_path / "in.bin")
    done = subprocess.run([str(runner), "--pte", str(pte), "--input",
                           str(tmp_path / "in.bin"), "--output",
                           str(tmp_path / "out.bin")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    got = np.fromfile(tmp_path / "out.bin", dtype=np.float32)
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()


class LongHeadQK(nn.Module):
    """Softmax over q@kT, which the opt-in quantize_qk gate claims as an
    int8 bmm whatever the head width."""

    def forward(self, query, key):
        return torch.softmax(torch.bmm(query, key.transpose(1, 2)), dim=-1)


def _long_dot(name, k):
    """A model whose one dot op has `k` terms, its inputs, and the partitioner
    options that let the gate claim it."""
    g = torch.Generator().manual_seed(9)
    if name == "linear":
        return nn.Linear(k, 2), (torch.randn(1, k, generator=g),), {}
    if name == "conv":
        c_in = -(-k // 9)                    # 3x3 taps: K = c_in * 9 >= k
        return (nn.Conv2d(c_in, 2, 3),
                (torch.randn(1, c_in, 3, 3, generator=g),), {})
    return (LongHeadQK(),
            (0.05 * torch.randn(1, 2, k, generator=g),
             0.05 * torch.randn(1, 3, k, generator=g)),
            {"quantize_qk": True})


def _lower_and_run(tmp_path, model, inputs, **options):
    """lower_esp32 and the post passes, then the program on the host runner:
    (partitioner, the ops left outside the delegates, delegate count, the
    host runner's output or its log when it fails)."""
    import subprocess

    import numpy as np
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.test.host import host_runner

    runner = host_runner()
    if not runner.is_file():
        pytest.fail(f"host runner not built: {runner}")
    partitioner = espressif_partitioner.Esp32Partitioner(**options)
    with contextlib.redirect_stdout(io.StringIO()):
        edge = lower_esp32(torch.export.export(model.eval(), inputs),
                           [partitioner])
        apply_passes(edge, quiet=True)
    ops = [str(n.target) for n in edge.exported_program().graph.nodes
           if n.op == "call_function"]
    delegates = sum("executorch_call_delegate" in t for t in ops)
    pte = tmp_path / "model.pte"
    pte.write_bytes(edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16)).buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, x in enumerate(inputs):
        x.numpy().tofile(tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    command += ["--output", str(tmp_path / "out.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    got = (np.fromfile(tmp_path / "out.bin", dtype=np.float32)
           if done.returncode == 0 else done.stdout[-1500:] + done.stderr[-1500:])
    return partitioner, ops, delegates, got


LONG_DOT = "dot past the exact int32 accumulator"


@pytest.mark.parametrize("name", ["linear", "conv", "q@kT"])
def test_int8_dot_past_the_exact_accumulator_stays_portable(tmp_path, name):
    """An int8 dot accumulates in int32, exact to MAX_DOT_LEN
    terms, and the runtime's init refuses an op with a longer one, which
    fails the whole program's load (ExecuTorch error 35) if the gate claimed
    it. The gate refuses the op by name instead: it stays portable,
    and the program loads and computes eager's numbers."""
    import numpy as np

    from executorch.backends.esp32s3.serialization.espressif_blob import MAX_DOT_LEN

    model, inputs, options = _long_dot(name, MAX_DOT_LEN + 32)
    with torch.no_grad():
        want = model.eval()(*inputs).numpy().reshape(-1)
    partitioner, ops, delegates, got = _lower_and_run(
        tmp_path, model, inputs, **options)
    assert partitioner.support.kinds.get(LONG_DOT), dict(partitioner.support.kinds)
    assert delegates == 0, ops
    assert isinstance(got, np.ndarray), got
    assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max()


@pytest.mark.parametrize("policy,claimed", [
    ("i4", False), ("auto", False), ("i16", True), ("fp32", True)])
def test_exact_dot_length_by_format(tmp_path, policy, claimed):
    """Only the loops that accumulate int8 codes in int32 are bounded: the
    i4 and auto policies' (i4 or i8 weights) are, the i16 loop (int64) and
    the fp32 one are not. Each program loads and runs."""
    import numpy as np

    from executorch.backends.esp32s3.serialization.espressif_blob import MAX_DOT_LEN

    model, inputs, _ = _long_dot("linear", MAX_DOT_LEN + 32)
    partitioner, ops, delegates, got = _lower_and_run(
        tmp_path, model, inputs, format_policy=policy)
    assert bool(partitioner.support.kinds.get(LONG_DOT)) != claimed, \
        dict(partitioner.support.kinds)
    assert delegates == int(claimed), ops
    assert isinstance(got, np.ndarray) and np.isfinite(got).all(), got


def test_exact_dot_length_is_inclusive(tmp_path):
    """A dot of exactly MAX_DOT_LEN terms is claimed, and the runtime runs
    it."""
    import numpy as np

    from executorch.backends.esp32s3.serialization.espressif_blob import MAX_DOT_LEN

    model, inputs, _ = _long_dot("linear", MAX_DOT_LEN)
    partitioner, ops, delegates, got = _lower_and_run(tmp_path, model, inputs)
    assert not partitioner.support.kinds.get(LONG_DOT)
    assert delegates == 1, ops
    assert isinstance(got, np.ndarray) and np.isfinite(got).all(), got


class TwoLinears(nn.Module):
    """(a(x), b(y)): two Linears, each over its own input."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(12)
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 4)

    def forward(self, x, y):
        return self.a(x), self.b(y)


def test_static_act_scale_that_is_not_finite_stays_portable(tmp_path):
    """Under static_act each quantised op scales its activation by
    the max|activation| calibrate_static_act recorded, and preprocess refuses
    one that is not finite -- after the gate claimed the op, the whole
    export would fail ("invalid calibrated activation absmax inf"). A calibration
    batch whose activation reaches inf records one. The gate refuses that op
    by name instead: it stays portable, the op calibrated on finite values
    is still claimed, and the program runs."""
    import subprocess

    import numpy as np
    from executorch.exir import ExecutorchBackendConfig, to_edge

    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.quantizer.calibration import calibrate_static_act
    from executorch.backends.esp32s3.test.host import host_runner

    runner = host_runner()
    if not runner.is_file():
        pytest.fail(f"host runner not built: {runner}")
    model = TwoLinears().eval()
    g = torch.Generator().manual_seed(13)
    x, y = torch.randn(2, 8, generator=g), torch.randn(2, 8, generator=g)
    with torch.no_grad():
        want_a, want_b = (t.numpy().reshape(-1) for t in model(x, y))
    edge = to_edge(torch.export.export(model, (x, y)))
    calibrate_static_act(edge, [(x, y), (x, torch.full((2, 8), float("inf")))])
    partitioner = espressif_partitioner.Esp32Partitioner(static_act=True)
    with contextlib.redirect_stdout(io.StringIO()):
        edge = edge.to_backend(partitioner)
        apply_passes(edge, quiet=True)
    assert partitioner.support.kinds.get(
        "calibrated activation absmax not finite") == 1, dict(
            partitioner.support.kinds)
    ops = [str(n.target) for n in edge.exported_program().graph.nodes
           if n.op == "call_function"]
    assert sum("executorch_call_delegate" in t for t in ops) == 1, ops
    assert sum("aten.addmm." in t for t in ops) == 1, ops
    pte = tmp_path / "model.pte"
    pte.write_bytes(edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16)).buffer)
    x.numpy().tofile(tmp_path / "x.bin")
    y.numpy().tofile(tmp_path / "y.bin")
    done = subprocess.run(
        [str(runner), "--pte", str(pte), "--input", str(tmp_path / "x.bin"),
         "--input", str(tmp_path / "y.bin"), "--output", str(tmp_path / "a.bin"),
         "--output", str(tmp_path / "b.bin")],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    got_a = np.fromfile(tmp_path / "a.bin", dtype=np.float32)
    got_b = np.fromfile(tmp_path / "b.bin", dtype=np.float32)
    # a: the delegate's static int8 Linear; b: the portable fp32 one
    assert np.abs(got_a - want_a).max() <= 0.05 * np.abs(want_a).max()
    assert np.abs(got_b - want_b).max() <= 1e-5 * np.abs(want_b).max()


class LinearReluLinear(nn.Module):
    """b(relu(a(x))): two Linears in a row."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(14)
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 8)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


STATIC_ACT_DYNAMIC = "static_act over a dynamic recipe"


def _dynamically_quantised(recipe, model, x, only=None):
    """`model` quantised by a dynamic recipe, then exported: Esp32Quantizer's
    default (per-tensor symmetric activations), its xnnpack_dynamic preset
    (per-tensor affine) or esp32_torchao_config (per-token affine; on the
    Linears named in `only` alone, when given)."""
    if recipe == "torchao":
        from torchao.quantization import quantize_

        from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
            esp32_torchao_config,
        )

        quantize_(model, esp32_torchao_config(4, None),
                  filter_fn=None if only is None else
                  (lambda m, fqn: isinstance(m, nn.Linear) and fqn in only))
        return torch.export.export(model, (x,))
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3 import Esp32Quantizer

    quantizer = (Esp32Quantizer() if recipe == "esp32"
                 else Esp32Quantizer(preset=recipe))
    prepared = prepare_pt2e(torch.export.export(model, (x,)).module(),
                            quantizer)
    prepared(x)
    return torch.export.export(convert_pt2e(prepared), (x,))


def _static_act_lowering(ep, x, flow):
    """`ep` lowered under Esp32Partitioner(static_act=True): `two-step` is
    quantizer/calibration.py's lowering (to_edge, calibrate_static_act,
    to_backend), `lower_esp32` the README's. Then apply_passes, as for every
    program. Returns (partitioner, edge, call_function targets)."""
    from executorch.exir import to_edge

    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.quantizer.calibration import calibrate_static_act

    partitioner = espressif_partitioner.Esp32Partitioner(static_act=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if flow == "two-step":
            edge = to_edge(ep)
            calibrate_static_act(edge, [(x,)])
            edge = edge.to_backend(partitioner)
        else:
            edge = lower_esp32(ep, [partitioner])
        apply_passes(edge, quiet=True)
    ops = [str(n.target) for n in edge.exported_program().graph.nodes
           if n.op == "call_function"]
    return partitioner, edge, ops


def _emit_and_run(tmp_path, edge, x):
    """`edge` emitted and run on the host runner over `x`: its one output."""
    import subprocess

    import numpy as np
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.test.host import host_runner

    runner = host_runner()
    if not runner.is_file():
        pytest.fail(f"host runner not built: {runner}")
    pte = tmp_path / "model.pte"
    pte.write_bytes(edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16)).buffer)
    x.numpy().tofile(tmp_path / "x.bin")
    done = subprocess.run(
        [str(runner), "--pte", str(pte), "--input", str(tmp_path / "x.bin"),
         "--output", str(tmp_path / "y.bin")],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    return np.fromfile(tmp_path / "y.bin", dtype=np.float32)


@pytest.mark.parametrize("flow", ["two-step", "lower_esp32"])
@pytest.mark.parametrize("recipe", ["esp32", "xnnpack_dynamic", "torchao"])
def test_static_act_over_a_dynamic_recipe_is_refused(tmp_path, recipe, flow):
    """static_act over a dynamic recipe: the delegate runs one activation
    scheme per op, so each op the recipe quantises is refused by name and
    stays portable on the recipe's value: the program lowers, emits and
    runs, equal to the converted model. Claimed, the op would be quantised
    again with the calibrated scale over the recipe's dequantised value: the
    two-step lowering runs no chain strip, so the recipe's chooser chain
    would reach to_executorch (no out variant for a symmetric chooser or any
    torchao op) or the runtime; lower_esp32 absorbs the chain, and
    preprocess would raise for the calibration static_act wants."""
    import numpy as np

    model = LinearReluLinear().eval()
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(15))
    ep = _dynamically_quantised(recipe, model, x)
    with torch.no_grad():
        want = ep.module()(x).numpy().reshape(-1)
    partitioner, edge, ops = _static_act_lowering(ep, x, flow)
    assert partitioner.support.kinds.get(STATIC_ACT_DYNAMIC) == 2, dict(
        partitioner.support.kinds)
    assert not any("executorch_call_delegate" in t for t in ops), ops
    assert not any("choose_qparams" in t or "torchao" in t for t in ops), ops
    got = _emit_and_run(tmp_path, edge, x)
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max(), (got, want)


def test_static_act_refuses_only_the_dynamically_quantised_op(tmp_path):
    """The refusal is per op. esp32_torchao_config on `a` alone: under
    static_act's two-step lowering `a` is refused by name and stays
    portable on the recipe's value, and `b`, which the recipe left float,
    is still claimed."""
    import numpy as np

    model = LinearReluLinear().eval()
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(16))
    ep = _dynamically_quantised("torchao", model, x, only={"a"})
    with torch.no_grad():
        want = ep.module()(x).numpy().reshape(-1)
    partitioner, edge, ops = _static_act_lowering(ep, x, "two-step")
    assert partitioner.support.kinds.get(STATIC_ACT_DYNAMIC) == 1, dict(
        partitioner.support.kinds)
    assert sum("executorch_call_delegate" in t for t in ops) == 1, ops
    assert sum("aten.addmm." in t for t in ops) == 1, ops
    assert not any("choose_qparams" in t or "torchao" in t for t in ops), ops
    got = _emit_and_run(tmp_path, edge, x)
    # b in the delegate (at worst a static int8 activation), a portable
    assert np.abs(got - want).max() <= 0.05 * np.abs(want).max(), (got, want)
