# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The standard PT2E flow reaches this backend.

The claims, in order:

  1. convert_pt2e runs at all (torchao 0.17 references an aten overload torch
     2.13 removed; torchao 0.18 does not),
  2. Esp32Quantizer's annotation produces per-channel int4 weight codes in
     [-7, 7] and a DYNAMIC activation chain,
  3. the activation chain strips, so no quantized_decomposed op is left stranded
     outside the delegate where it would fail at Method::init,
  4. the matmuls are delegated with the codes taken VERBATIM -- the point of
     pass-through, and the difference between honouring PT2E's numbers and
     re-quantising them,
  5. the delegated program's arithmetic matches a host mirror of the same int4
     math to within the tolerance the parity harness uses.
  6. ternary PT2E codes survive as a clearly-labelled i4 semantic carrier; this
     is the no-speed bring-up control for the future signed-t2 wire format.
  7. the README's flow, lower_esp32 with nothing added, absorbs the chains
     itself, and the host runner computes the converted model's numbers;
     a chain it leaves in place computes them bit for bit, on portable ops.
  8. lower_esp32's strip and a standalone one before partitioning give the
     same program where both absorb a chain.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* (one
per *_case) fails naming its failed checks (@case).

    python test_pt2e.py
"""

import functools
import sys

import numpy as np
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


class MLP(nn.Module):
    def __init__(self, d=32, h=64, o=16):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, h), nn.Linear(h, o)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def shared_activation_case():
    """One activation feeding three Linears -- separate q/k/v, not a fused qkv.

    Most published transformers project q, k and v separately, which makes
    PT2E emit one choose_qparams/quantize pair with three users, and torchao's
    PortNodeMetaForQDQ raises `Expecting single user for quantize_per_tensor_*`
    on exactly that (a fused qkv Linear gives the activation one consumer).

    The quantizer's answer is to annotate weight-only when it sees a shared
    activation. What that must NOT do is change deployed arithmetic, and it does
    not: the activation qdq chain is stripped before partitioning anyway. So the
    claim under test is that the artifact still converts, still lowers, and
    still lands its weights in the delegate as PASS-THROUGH codes.
    """
    import contextlib
    import io

    import torch.nn as nn
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import (convert_pt2e,
                                                         prepare_qat_pt2e)

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    class SeparateQKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(32, bias=False)
            self.q = nn.Linear(32, 32, bias=False)
            self.k = nn.Linear(32, 32, bias=False)
            self.v = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            h = self.norm(x)
            return self.q(h) + self.k(h) + self.v(h)

    print("shared activation (separate q/k/v projections):")
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    gm = torch.export.export(SeparateQKV().train(), (x,)).module()

    quant = Esp32Quantizer(weight_bits=4)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        prepared = prepare_qat_pt2e(gm, quant)
    notice = buf.getvalue()
    check("the shared activation is detected and announced",
          "feed more than one matmul" in notice)
    check("it degraded to weight-only rather than guessing",
          quant.act_annotated is False)

    # A QAT step, so the fake-quant machinery is genuinely exercised rather than
    # the graph merely being built and converted untouched.
    prepared(x).pow(2).mean().backward()
    try:
        converted = convert_pt2e(prepared)
        move_exported_model_to_eval(converted)
        ok = True
    except Exception as e:
        ok = False
        print(f"      convert_pt2e raised {type(e).__name__}")
    check("convert_pt2e survives a shared dynamically-quantised activation", ok)
    if not ok:
        return

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        edge = to_edge_transform_and_lower(
            torch.export.export(converted, (x,)),
            partitioner=[Esp32Partitioner()])
    desc = buf.getvalue()
    gmm = edge.exported_program().graph_module
    left = sum(1 for n in gmm.graph.nodes if n.op == "call_function"
               and ("aten.addmm.default" in str(n.target)
                    or "aten.mm.default" in str(n.target)))
    check("all three matmuls delegated", left == 0, f"{left} left portable")
    check("weights still consumed verbatim", desc.count("addmmPT_i4") == 3,
          f"{desc.count('addmmPT_i4')} pass-through op(s)")

    # The control. If forcing the annotation ALSO passed, the degrade would be
    # unnecessary and this whole path would be cargo cult.
    gm2 = torch.export.export(SeparateQKV().train(), (x,)).module()
    forced = prepare_qat_pt2e(gm2, Esp32Quantizer(weight_bits=4,
                                                  annotate_act=True))
    forced(x).pow(2).mean().backward()
    try:
        convert_pt2e(forced)
        still_broken = False
    except Exception:
        still_broken = True
    check("annotate_act=True still fails upstream (the degrade is load-bearing)",
          still_broken)


def strip_scope_case():
    """Which dynamic activation chains strip_dynamic_act_qdq absorbs, what it
    records, and what it refuses by name, called on its own (no partitioner).

    Every dynamic chain is absorbed with its exact scheme recorded on the
    consumer (node.meta "esp32_act_quant"), or the export stops with
    DynamicChainRefused naming its rule -- never left in place, where its
    chooser would run ExecuTorch's portable choose_qparams, which disagrees
    with PyTorch. Esp32Quantizer's own chain keeps FLAG_ACT_PER_TENSOR. The
    PT2E affine chooser and the torchao asymmetric chain are absorbed with
    their schemes recorded; the mismatched dequant and the portable consumers
    are refused by name, PT2E's and torchao's alike: without the partitioner's
    gate the strip computes no chain in place (lower_esp32's does,
    portable_chain_case); the torchao symmetric control records its own grid,
    never a substituted one.
    """
    import contextlib
    import io

    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops
    from executorch.exir import to_edge
    from torchao.quantization.quant_primitives import (MappingType,
                                                       choose_qparams_affine,
                                                       dequantize_affine,
                                                       quantize_affine)

    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        DynamicChainRefused,
        strip_dynamic_act_qdq,
    )

    qd = torch.ops.quantized_decomposed
    eps = torch.finfo(torch.float32).eps

    def pt2e_fq(x, choose=qd.choose_qparams_symmetric.tensor, dq_scale=1.0):
        scale, zp = choose(x, -127, 127, eps, torch.int8)
        q = qd.quantize_per_tensor.tensor(x, scale, zp, -127, 127, torch.int8)
        if dq_scale != 1.0:
            scale = scale * dq_scale
        return qd.dequantize_per_tensor.tensor(q, scale, zp, -127, 127,
                                               torch.int8)

    def torchao_fq(x, mapping, qmin, qmax):
        block = [1, x.shape[-1]]                   # per token, like torchao's
        scale, zp = choose_qparams_affine(
            x, mapping, block, torch.int8, qmin, qmax, eps=1e-6,
            scale_dtype=torch.float32, zero_point_dtype=torch.int32)
        q = quantize_affine(x, block, scale, zp, torch.int8, qmin, qmax)
        return dequantize_affine(q, block, scale, zp, torch.int8, qmin, qmax)

    class Net(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.fc = nn.Linear(16, 8)
            self.body = body

        def forward(self, x):
            return self.body(self, x)

    def also_output(m, x):
        d = pt2e_fq(x)                 # ONE chain, two uses
        return m.fc(d), d + 1

    def torchao_also_output(m, x):
        d = torchao_fq(x, MappingType.ASYMMETRIC, -128, 127)
        return m.fc(d), d + 1

    print("dynamic activation chains are absorbed and recorded, or refused:")
    torch.manual_seed(0)
    x = torch.rand(4, 16)          # one-sided, so an affine zero point is not 0
    asym = lambda x: torchao_fq(x, MappingType.ASYMMETRIC, -128, 127)  # noqa: E731
    # (name, body, expected): an int is how many were absorbed, with the
    # record each consumer must carry ("pt" = FLAG_ACT_PER_TENSOR, else
    # (chooser, granularity, qmin, qmax)); a str is the rule the refusal must
    # name.
    cases = (
        ("PT2E symmetric chooser, matmul only (control)",
         lambda m, x: m.fc(pt2e_fq(x)), (1, "pt")),
        ("PT2E affine chooser with [-127, 127] bounds",
         lambda m, x: m.fc(pt2e_fq(x, qd.choose_qparams.tensor)),
         (1, ("affine", "tensor", -127, 127))),
        ("PT2E dequant with a different scale from its quantise",
         lambda m, x: m.fc(pt2e_fq(x, dq_scale=2.0)), "act-chain-mismatch"),
        ("PT2E dequantised value also feeds a portable output",
         also_output, "act-chain-consumer"),
        ("PT2E dequantised value feeds only a portable op",
         lambda m, x: m.fc(x) + pt2e_fq(x).sum(-1, keepdim=True),
         "act-chain-consumer"),
        ("torchao symmetric [-127, 127] per-token chain (its own grid)",
         lambda m, x: m.fc(torchao_fq(x, MappingType.SYMMETRIC, -127, 127)),
         (1, ("symmetric", "row", -127, 127))),
        ("torchao asymmetric per-token chain",
         lambda m, x: m.fc(asym(x)), (1, ("affine", "row", -128, 127))),
        ("torchao chain that also feeds a portable output",
         torchao_also_output, "act-chain-consumer"),
    )
    for name, body, want in cases:
        edge = to_edge(torch.export.export(Net(body).eval(), (x,)))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = strip_dynamic_act_qdq(edge)
        except DynamicChainRefused as why:
            check(name, why.rule == want, f"refused [{why.rule}]")
            continue
        if isinstance(want, str):
            check(name, False, f"absorbed {got}, expected [{want}]")
            continue
        count, record = want
        mms = [n for n in edge.exported_program().graph.nodes
               if n.op == "call_function" and "mm" in str(n.target)]
        scheme = mms[0].meta.get("esp32_act_quant")
        if record == "pt":
            ok = bool(mms[0].meta.get("esp32_act_per_tensor")) and scheme is None
        else:
            ok = scheme is not None and (scheme.chooser, scheme.granularity,
                                         scheme.qmin, scheme.qmax) == record
        left = [n.name for n in edge.exported_program().graph.nodes
                if n.op == "call_function" and "choose_qparams" in str(n.target)]
        check(name, got == count and ok and not left,
              f"absorbed {got}, record {scheme or mms[0].meta.get('esp32_act_per_tensor')}")


def _pt2e_chain(x, symmetric, qmin, qmax, eps):
    """A PT2E dynamic chain written out: the symmetric spelling is
    Esp32Quantizer's ([-127, 127], FLT_EPSILON), the affine one XNNPACK's
    dynamic recipe's ([-128, 127], 2^-12)."""
    qd = torch.ops.quantized_decomposed
    choose = (qd.choose_qparams_symmetric.tensor if symmetric
              else qd.choose_qparams.tensor)
    scale, zp = choose(x, qmin, qmax, eps, torch.int8)
    q = qd.quantize_per_tensor.tensor(x, scale, zp, qmin, qmax, torch.int8)
    return qd.dequantize_per_tensor.tensor(q, scale, zp, qmin, qmax, torch.int8)


def _standalone_strip():
    """A context in which lower_esp32 also runs strip_dynamic_act_qdq with no
    partitioner's gate, right after to_edge (around its first step,
    fold_conv_bn), as the census did before lower_esp32 ran the gated strip
    itself: every matmul and convolution takes its chain there, and the
    gated strip finds nothing left. What the gated strip absorbs must come
    out as this flow makes it."""
    import contextlib

    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3._passes import fold_conv_bn_pass
    from executorch.backends.esp32s3._passes import strip_dynamic_act_qdq_pass

    @contextlib.contextmanager
    def standalone():
        fold, ran = fold_conv_bn_pass.fold_conv_bn, []

        def strip_then_fold(edge, method_name="forward"):
            ran.append(strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq(
                edge, method_name))
            return fold(edge, method_name)

        _passes.fold_conv_bn = strip_then_fold
        try:
            yield
        finally:
            _passes.fold_conv_bn = fold
        if not ran:
            raise RuntimeError("lower_esp32 no longer calls fold_conv_bn "
                               "after to_edge: the standalone strip did not run")

    return standalone()


def _torchao_chain(x, mapping, qmin, qmax, eps, zero_point_dtype):
    """torchao's dynamic chain written out, per token (blocks [1, ..., K]):
    ASYMMETRIC [-128, 127] with eps None is esp32_torchao_config's (A3); a
    float32 zero point, the legacy Int8DynActInt4WeightQuantizer's."""
    from torchao.quantization.quant_primitives import (MappingType,
                                                       choose_qparams_affine,
                                                       dequantize_affine,
                                                       quantize_affine)

    block = [1] * (x.dim() - 1) + [x.shape[-1]]
    scale, zp = choose_qparams_affine(
        x, getattr(MappingType, mapping), block, torch.int8, qmin, qmax,
        eps=eps, scale_dtype=torch.float32, zero_point_dtype=zero_point_dtype)
    q = quantize_affine(x, block, scale, zp, torch.int8, qmin, qmax)
    return dequantize_affine(q, block, scale, zp, torch.int8, qmin, qmax)


def _host_outputs(program, inputs, count):
    """The program's `count` outputs on the host runner, or (None, its log)."""
    import subprocess
    import tempfile
    from pathlib import Path

    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(program.buffer)
        command = [str(HOST_RUNNER), "--pte", str(tmp / "model.pte")]
        for i, x in enumerate(inputs):
            x.detach().contiguous().numpy().tofile(tmp / f"in{i}.bin")
            command += ["--input", str(tmp / f"in{i}.bin")]
        for i in range(count):
            command += ["--output", str(tmp / f"out{i}.bin")]
        run = subprocess.run(command, capture_output=True, text=True,
                             timeout=120)
        if run.returncode != 0:
            return None, (run.stdout + run.stderr)[-300:]
        return [np.fromfile(tmp / f"out{i}.bin", dtype=np.float32)
                for i in range(count)], ""


def portable_chain_case():
    """A dynamic chain whose dequantised value a portable op and a graph
    output read: the README flow (lower_esp32, which gives the strip its
    partitioner) leaves it in place, computed by float32 portable ops, and the
    program asks the runtime for no chooser, quantize or dequantize kernel --
    ExecuTorch's choose_qparams is not PyTorch's, it has no symmetric one,
    and torchao's ops have none at all. The host runner's values equal the
    converted model's bit for bit: each op is the one the recipe's chooser,
    quantize and dequantize compute with, in their order and precision, per
    tensor (PT2E) or per token (torchao), ties to even and zeros included,
    and values a division by the scale would round to other codes than the
    chain's multiply by its reciprocal."""
    import contextlib
    import io

    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import Esp32Partitioner
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    class Chain(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            d = self.body(x)
            return d, d * 3.0

    print("a dynamic chain left in place computes the converted model's bits:")
    torch.manual_seed(3)
    eps32 = torch.finfo(torch.float32).eps
    # (name, chain, the range's ends, per token)
    spellings = (
        ("symmetric [-127, 127]",
         lambda x: _pt2e_chain(x, True, -127, 127, eps32), [127.0, -127.0],
         False),
        ("affine [-128, 127]",
         lambda x: _pt2e_chain(x, False, -128, 127, 2 ** -12),
         [127.0, -128.0], False),
        ("torchao asymmetric per token",
         lambda x: _torchao_chain(x, "ASYMMETRIC", -128, 127, None,
                                  torch.int32), [127.0, -128.0], True),
        ("torchao symmetric per token [-127, 127]",
         lambda x: _torchao_chain(x, "SYMMETRIC", -127, 127, eps32,
                                  torch.int32), [127.0, -127.0], True),
        ("torchao asymmetric per token, float32 zero points",
         lambda x: _torchao_chain(x, "ASYMMETRIC", -128, 127, eps32,
                                  torch.float32), [127.0, -128.0], True))
    for name, body, ends, per_token in spellings:
        model = Chain(body).eval()
        # Unit scale (the range's own ends, in each row for a per-token
        # chain), so x * (1 / scale) lands on the half steps: ties to even,
        # signed zeros.
        halves = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -0.0, 0.0, 3.5,
                  -3.5, 4.5, 0.25, -0.75]
        ties = (torch.tensor([ends + halves[:6], ends + halves[6:12]])
                if per_token else torch.tensor(ends + halves).reshape(2, 8))
        # A scale whose reciprocal is inexact, with values where the
        # chain's x * fl(1 / scale) and a division x / scale round to
        # different codes (for the [-127, 127] grids: amax 2.8885069 in
        # each row, so per tensor and per token alike).
        a, r = 2.8885068893432617, [0.011372075416147709, 0.0341162234544754,
                                    0.05686037614941597, 0.07960452139377594]
        reciprocal = torch.tensor([[a, *r, -r[1], -r[0], 0.5],
                                   [a, r[2], -r[3], 0.0, 1.0, -2.0, r[3], r[0]]])
        inputs = {"random": torch.randn(2, 8), "ties": ties,
                  "zeros": torch.tensor([[0.0, -0.0] * 4] * 2),
                  "below eps": torch.randn(2, 8) * 1e-9,
                  "reciprocal": reciprocal}
        ep = torch.export.export(model, (inputs["random"],))
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                edge = lower_esp32(ep, [Esp32Partitioner()])
                apply_passes(edge, quiet=True)
                et = edge.to_executorch(ExecutorchBackendConfig(
                    delegate_alignment=16))
        except Exception as e:  # noqa: BLE001 -- the regression is "this raised"
            check(f"{name}: the README flow lowers it", False,
                  f"{type(e).__name__}: {str(e)[:200]}")
            continue
        check(f"{name}: the README flow lowers it, saying so",
              any(l.startswith("esp32: leaving") for l in
                  log.getvalue().splitlines()))
        table = sorted(f"{o.name}.{o.overload}" for plan in
                       et.executorch_program.execution_plan
                       for o in plan.operators
                       if o.name.startswith(("quantized_decomposed::",
                                             "torchao::")))
        check(f"{name}: no quantized_decomposed or torchao kernel asked for",
              not table, f"{table}")
        for kind, x in inputs.items():
            got, log_tail = _host_outputs(et, (x,), 2)
            with torch.no_grad():
                want = [t.numpy().reshape(-1) for t in model(x)]
            same = got is not None and all(
                g.shape == w.shape and np.array_equal(g.view(np.int32),
                                                      w.view(np.int32))
                for g, w in zip(got, want))
            check(f"{name}, {kind} input: bit for bit on the host runner",
                  same, log_tail if got is None else
                  f"max abs error {max(float(np.abs(g - w).max()) for g, w in zip(got, want)):.3g}")


def decomposed_matmul_case():
    """An explicit mm/addmm takes its weight already transposed, [K, N], so a
    per-output-channel (axis 0) spec on it would be per-K. Neither a square
    nor a rectangular one is annotated (a square weight passes a shape test
    and would then fail the backend's transpose check), and both lower.
    nn.Linear is the positive control."""
    import contextlib
    import io

    from executorch.exir import to_edge
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    class Explicit(nn.Module):
        def __init__(self, k, n, addmm):
            super().__init__()
            self.w = nn.Parameter(torch.randn(k, n))
            self.b = nn.Parameter(torch.randn(n))
            self.addmm = addmm

        def forward(self, x):
            if self.addmm:
                return torch.addmm(self.b, x, self.w)
            return torch.mm(x, self.w)

    print("explicit mm/addmm are not given a per-output-channel weight spec:")
    for name, model, want in (
            ("square mm", Explicit(16, 16, False), 0),
            ("square addmm", Explicit(16, 16, True), 0),
            ("rectangular mm", Explicit(16, 8, False), 0),
            ("nn.Linear (control)", nn.Linear(16, 16), 1)):
        x = torch.randn(4, 16)
        quant = Esp32Quantizer(weight_bits=4, annotate_act=False)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gm = torch.export.export(model.eval(), (x,)).module()
                prepared = prepare_pt2e(gm, quant)
                prepared(x)
                edge = to_edge(torch.export.export(convert_pt2e(prepared), (x,)))
                strip_dynamic_act_qdq(edge)
                edge.to_backend(Esp32Partitioner())
            check(f"{name}: {want} annotated, and it lowers",
                  quant.annotated == want, f"{quant.annotated} annotated")
        except Exception as e:  # noqa: BLE001 -- any failure is the bug
            check(f"{name}: {want} annotated, and it lowers", False,
                  f"{quant.annotated} annotated; {type(e).__name__}: "
                  f"{str(e)[:70]}")


def ternary_semantic_carrier_case():
    """Ternary PT2E codes reach the runtime without requantization.

    This is intentionally not a packed-t2 claim. The delegate sees qmax=1,
    preserves the codes, and stores them in its i4 carrier. A future t2
    opcode must keep this exact model as its same-codes parity control.
    """
    import contextlib
    import io

    from executorch.exir import ExecutorchBackendConfig, to_edge
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    print("ternary PT2E semantic carrier:")
    torch.manual_seed(11)
    model = MLP().eval()
    x = torch.randn(4, 32)
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=2))
    for _ in range(4):
        prepared(torch.randn(4, 32))
    converted = convert_pt2e(prepared)

    code_sets = []
    for node in converted.graph.nodes:
        if (node.op == "call_function"
                and "dequantize_per_channel" in str(node.target)):
            source = node.args[0]
            if source.op == "get_attr":
                codes = getattr(converted, source.target)
                code_sets.append(set(codes.unique().tolist()))
    check("PT2E emitted only {-1,0,+1} weight codes",
          bool(code_sets) and all(values <= {-1, 0, 1}
                                  for values in code_sets), str(code_sets))

    edge = to_edge(torch.export.export(converted, (x,)))
    stripped = strip_dynamic_act_qdq(edge)
    check("ternary dynamic activation chains stripped", stripped > 0,
          f"{stripped} chain(s)")
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        edge = edge.to_backend(Esp32Partitioner())
    descriptors = capture.getvalue()
    # addmmPT_i4 is the honest expected spelling: exact ternary codes, i4
    # storage. The packed-t2 implementation must change this assertion.
    carriers = descriptors.count("addmmPT_i4")
    check("both ternary Linears use the explicit i4 carrier", carriers == 2,
          f"{carriers} carrier op(s)")
    et = edge.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16))
    check("ternary carrier program emits", len(et.buffer) > 0,
          f"{len(et.buffer):,} B .pte")


class SmallCNN(nn.Module):
    """Three conv spellings the backend runs -- dense, depthwise, pointwise --
    then a Linear, so one graph exercises every pass-through conv shape."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, groups=8, padding=1)
        self.pw = nn.Conv2d(8, 4, 1)
        self.fc = nn.Linear(4 * 6 * 6, 5)

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.dw(h))
        h = self.pw(h)
        return self.fc(h.flatten(1))


def conv_pass_through_case(weight_bits=8):
    """PT2E-quantised convolutions reach the delegate with their codes intact.

    A conv whose weight arrives as a `dequantize_per_channel` chain is
    delegated, not refused ("conv weight is not a plain constant"): a refusal
    is a silent fallback that would run every convolution of a PT2E CNN
    portable and unquantised. The per-channel codes of a `[C_out, C_in, kh,
    kw]` weight ARE the `[N, K]` packer layout after a reshape, the same
    identity the whole conv delegate rests on, so they run on the same kernels.
    """
    print(f"\nPT2E convolutions, pass-through at i{weight_bits}:")
    import contextlib
    import io

    from executorch.exir import ExecutorchBackendConfig, to_edge
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    torch.manual_seed(0)
    model = SmallCNN().eval()
    x = torch.randn(1, 3, 6, 6)
    gm = torch.export.export(model, (x,)).module()
    quantizer = Esp32Quantizer(weight_bits=weight_bits)
    prepared = prepare_pt2e(gm, quantizer)
    prepared(x)
    converted = convert_pt2e(prepared)
    check("the quantizer annotated the three convs and the Linear",
          quantizer.annotated == 4, f"{quantizer.annotated} annotated")

    edge = to_edge(torch.export.export(converted, (x,)))
    stripped = strip_dynamic_act_qdq(edge)
    check("four dynamic activation chains stripped (three convs, one Linear)",
          stripped == 4, f"{stripped} stripped")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        edge = edge.to_backend(Esp32Partitioner())
    log = buf.getvalue()
    gmm = edge.exported_program().graph_module
    targets = [str(n.target) for n in gmm.graph.nodes
               if n.op == "call_function"]
    check("no convolution left portable",
          not any("convolution" in t for t in targets), str(targets))
    check("no quantized_decomposed op stranded",
          not any("quantized_decomposed" in t for t in targets))
    tag = f"addmmPT_i{weight_bits}"
    n_pt = log.count(tag)
    check("all four weights consumed verbatim", n_pt == 4,
          f"{n_pt} {tag} op(s)")
    check("dense conv geometry rides on the pass-through descriptor",
          " conv3x3[3->8 @6x6]" in log, log[-400:])
    check("depthwise geometry too", " conv3x3[1->8 @6x6]" in log)
    check("pointwise geometry too", " conv1x1[8->4 @6x6]" in log)
    check("the ReLUs fused onto the pass-through convs",
          log.count("+relu") == 2, f"{log.count('+relu')} +relu")
    check("every op honours PT2E's per-tensor activation scale",
          log.count("+pt") == 4, f"{log.count('+pt')} +pt")
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    check("program emits", len(et.buffer) > 0, f"{len(et.buffer):,} B")

    # Selectivity: a transposed convolution's per-channel axis is dim 1, and
    # the pass-through path has no layout for that. The quantizer therefore
    # does not annotate it, so it arrives as a plain fp32 weight and is
    # quantised by the backend itself -- delegated, not dropped. (The gate's
    # own refusal of a pass-through transposed weight is reachable only from a
    # hand-built graph.)
    class Up(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.ConvTranspose2d(3, 4, 2, stride=2)

        def forward(self, x):
            return self.up(x)

    part = Esp32Partitioner()
    up = Up().eval()
    gm_up = torch.export.export(up, (x,)).module()
    q_up = Esp32Quantizer(weight_bits=8)
    prepared = prepare_pt2e(gm_up, q_up)
    prepared(x)
    check("the quantizer leaves a transposed conv unannotated",
          q_up.annotated == 0, f"{q_up.annotated} annotated")
    edge_up = to_edge(torch.export.export(convert_pt2e(prepared), (x,)))
    strip_dynamic_act_qdq(edge_up)
    with contextlib.redirect_stdout(io.StringIO()):
        edge_up = edge_up.to_backend(part)
    n_up = sum(1 for n in edge_up.exported_program().graph_module.graph.nodes
               if n.op == "call_function" and "call_delegate" in str(n.target))
    check("and it still delegates as a backend-quantised transposed conv",
          n_up == 1, f"{n_up} delegate call(s)")


def gated_strip_case():
    """lower_esp32's strip asks the partitioner's gate which ops will take a
    chain, about the graph the partitioner will see: the op carrying the
    chain's record. A chain into a Linear whose weight the recipe left float
    is then absorbed there, as a standalone strip right after to_edge, with
    no gate, absorbs it (_standalone_strip: the census's flow before
    lower_esp32 stripped), and the two flows give one program, byte for
    byte.
    Asked without the record, the gate would see a weight kept float and
    leave the chain in place, and the partitioner -- no chain or record
    left, the program not recipe-quantised -- would claim the Linear and
    quantise its activation its own way."""
    import contextlib
    import io

    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import Esp32Partitioner
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    class FloatLinear(nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.fc = nn.Linear(16, 8)
            self.spec = spec

        def forward(self, x):
            return self.fc(_pt2e_chain(x, *self.spec))

    def program(model, x, standalone):
        ep = torch.export.export(model, (x,))
        with contextlib.redirect_stdout(io.StringIO()):
            with (_standalone_strip() if standalone
                  else contextlib.nullcontext()):
                edge = lower_esp32(ep, [Esp32Partitioner()])
            apply_passes(edge, quiet=True)
            return edge.to_executorch(ExecutorchBackendConfig(
                delegate_alignment=16)).buffer

    print("lower_esp32's strip and a standalone one give one program:")
    torch.manual_seed(4)
    x = torch.randn(4, 16)
    eps32 = torch.finfo(torch.float32).eps
    for name, spec in (("symmetric [-127, 127]", (True, -127, 127, eps32)),
                       ("affine [-128, 127]", (False, -128, 127, 2 ** -12))):
        model = FloatLinear(spec).eval()
        gated, standalone = program(model, x, False), program(model, x, True)
        check(f"{name} chain into a float-weight Linear: the same program",
              gated == standalone,
              f"{len(gated)} B through lower_esp32, {len(standalone)} B with "
              f"the strip before partitioning")


def shared_view_case():
    """A view of a chain's dequantised value that a Linear and a portable op
    both read (a hand-built graph: convert_pt2e and torchao give each
    consumer its own dequantize). lower_esp32's strip gives the Linear its
    own view of the tensor before the chain, so it takes the chain over with
    its record, and leaves the view the portable op reads on the dequantised
    value, computed in place. The Linear's output is then, bit for bit, the
    one it gives with no portable reader beside it, its chain absorbed by a
    standalone strip (_standalone_strip); the portable op's is the converted
    model's. Planning
    the view as a whole kept the chain for both, and the partitioner claimed
    the Linear over the computed value and quantised it again under its own
    per-row scheme."""
    import contextlib
    import io

    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3 import Esp32Partitioner
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    eps32 = torch.finfo(torch.float32).eps

    class SharedView(nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.fc = nn.Linear(16, 8)
            self.shared = shared

        def forward(self, x):
            v = _pt2e_chain(x, True, -127, 127, eps32).reshape(4, 16)
            return (self.fc(v), v * 2.0) if self.shared else self.fc(v)

    def run(model, x, count, standalone=False):
        ep = torch.export.export(model, (x,))
        with contextlib.redirect_stdout(io.StringIO()):
            with (_standalone_strip() if standalone
                  else contextlib.nullcontext()):
                edge = lower_esp32(ep, [Esp32Partitioner()])
            apply_passes(edge, quiet=True)
            et = edge.to_executorch(ExecutorchBackendConfig(
                delegate_alignment=16))
        return _host_outputs(et, (x,), count)

    print("a view a Linear and a portable op share:")
    torch.manual_seed(6)
    x = torch.randn(2, 2, 16)
    shared, alone = SharedView(True).eval(), SharedView(False).eval()
    alone.load_state_dict(shared.state_dict())
    got, log_tail = run(shared, x, 2)
    want, log_alone = run(alone, x, 1, standalone=True)
    if got is None or want is None:
        check("both programs run on the host runner", False,
              log_tail or log_alone)
        return
    check("the Linear computes what it does with its chain absorbed, alone",
          np.array_equal(got[0].view(np.int32), want[0].view(np.int32)),
          f"max abs difference {float(np.abs(got[0] - want[0]).max()):.3g}")
    with torch.no_grad():
        eager = shared(x)[1].numpy().reshape(-1)
    check("the portable op reads the converted model's value, bit for bit",
          np.array_equal(got[1].view(np.int32), eager.view(np.int32)),
          f"max abs error {float(np.abs(got[1] - eager).max()):.3g}")


def readme_flow_case():
    """The README's flow, quantised first: Esp32Quantizer(), convert_pt2e,
    then lower_esp32 -> apply_passes -> to_executorch and nothing else.
    lower_esp32 absorbs the dynamic activation chains itself, so no chooser,
    quantize or dequantize is left for to_executorch (ExecuTorch has no
    choose_qparams_symmetric out variant), and the host runner computes the
    converted model's numbers: the delegate reproduces the chooser, and the
    int4 dots are exact, so only fp32 accumulation order differs."""
    import contextlib
    import io
    import subprocess
    import tempfile
    from pathlib import Path

    from executorch.exir import ExecutorchBackendConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3 import Esp32Partitioner, Esp32Quantizer
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.memory_planning import planning_pass
    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    print("the README flow lowers a dynamically quantised model:")
    torch.manual_seed(5)
    model = MLP().eval()
    x = torch.randn(4, 32)
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer())
    for _ in range(4):
        prepared(torch.randn(4, 32))
    ep = torch.export.export(convert_pt2e(prepared), (x,))
    with torch.no_grad():
        want = ep.module()(x).numpy()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            edge = lower_esp32(ep, [Esp32Partitioner()])
            apply_passes(edge, quiet=True)
            et = edge.to_executorch(ExecutorchBackendConfig(
                delegate_alignment=16, constant_tensor_alignment=16,
                memory_planning_pass=planning_pass("esp32")))
    except Exception as e:  # noqa: BLE001 -- the regression is "this raised"
        check("lower_esp32 -> to_executorch completes", False,
              f"{type(e).__name__}: {str(e)[:200]}")
        return
    check("lower_esp32 -> to_executorch completes", True)
    outside = [str(n.target) for n in edge.exported_program().graph.nodes
               if n.op == "call_function"
               and "executorch_call_delegate" not in str(n.target)
               and "getitem" not in str(n.target)]
    check("both chains absorbed, nothing left outside the delegate",
          not outside and edge.stripped_chains == 2,
          f"{edge.stripped_chains} absorbed; outside: {outside}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(et.buffer)
        x.numpy().tofile(tmp / "in.bin")
        run = subprocess.run(
            [str(HOST_RUNNER), "--pte", str(tmp / "model.pte"), "--input",
             str(tmp / "in.bin"), "--output", str(tmp / "out.bin")],
            capture_output=True, text=True, timeout=120)
        got = (np.fromfile(tmp / "out.bin", dtype=np.float32)
               if run.returncode == 0 else None)
    check("the host runner runs it", got is not None,
          "" if got is not None else (run.stdout + run.stderr)[-300:])
    if got is not None:
        err = float(np.abs(got - want.reshape(-1)).max())
        scale = float(np.abs(want).max())
        check("and computes the converted model's numbers",
              err <= 1e-5 * scale, f"max abs error {err:.3g} of {scale:.3g}")


def pt2e_flow_case():
    """Claims 1-5: annotate, convert, strip, partition, codes verbatim."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    torch.manual_seed(0)
    model = MLP().eval()
    x = torch.randn(4, 32)

    print("PT2E prepare -> calibrate -> convert:")
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=4))
    for _ in range(4):
        prepared(torch.randn(4, 32))
    converted = convert_pt2e(prepared)
    check("convert_pt2e completed", converted is not None)

    targets = [str(n.target) for n in converted.graph.nodes
               if n.op == "call_function"]
    check("weights got a per-channel dequant",
          any("dequantize_per_channel" in t for t in targets))
    check("activations got a dynamic (choose_qparams) chain",
          any("choose_qparams" in t for t in targets))

    # The codes themselves: int4 range in an int8 container is the whole claim
    # of the i4 rung, and a quantizer that silently emitted int8 codes would
    # still pass every structural check above.
    code_ranges = []
    for n in converted.graph.nodes:
        if n.op == "call_function" and "dequantize_per_channel" in str(n.target):
            src = n.args[0]
            if src.op == "get_attr":
                t = getattr(converted, src.target)
                code_ranges.append((int(t.min()), int(t.max())))
    check("weight codes are int4-ranged", bool(code_ranges) and all(
        lo >= -7 and hi <= 7 for lo, hi in code_ranges), f"{code_ranges}")

    print("lowering the converted graph through the partitioner:")
    from executorch.exir import ExecutorchBackendConfig, to_edge

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    ep = torch.export.export(converted, (x,))
    edge = to_edge(ep)
    stripped = strip_dynamic_act_qdq(edge)
    check("dynamic activation chains stripped", stripped > 0, f"{stripped} chain(s)")

    # Capture preprocess's descriptor line: it names the opcode actually
    # emitted, and "addmmPT_*" is the delegate's own word for "these codes were
    # consumed verbatim". The blob's magic would prove only that A blob
    # exists -- every blob has the magic, re-quantised or not.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        edge = edge.to_backend(Esp32Partitioner())
    descriptors = buf.getvalue()
    print(descriptors, end="")
    gmm = edge.exported_program().graph_module
    n_delegate = sum(1 for n in gmm.graph.nodes
                     if n.op == "call_function" and "call_delegate" in str(n.target))
    check("matmuls landed in the delegate", n_delegate > 0,
          f"{n_delegate} delegate call(s)")

    leftover = [str(n.target) for n in gmm.graph.nodes
                if n.op == "call_function" and "quantized_decomposed" in str(n.target)]
    check("no quantized_decomposed op stranded outside the delegate",
          not leftover, f"{len(leftover)} left")

    et = edge.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    check("program emits", len(et.buffer) > 0, f"{len(et.buffer):,} B .pte")

    # Codes taken VERBATIM is the substantive claim: the delegate reports the
    # pass-through opcode (addmmPT_*) for weights it consumed as given, and
    # would report the plain opcode (addmm_i4) had it re-quantised them. So the
    # assertion is on the opcode, and the negative is asserted too -- otherwise
    # a build that re-quantised everything would still satisfy the positive.
    n_pt = descriptors.count("addmmPT_i4")
    check("weights consumed verbatim (pass-through opcode)", n_pt == 2,
          f"{n_pt} addmmPT_i4 op(s)")
    check("nothing was re-quantised", "addmm_i4[" not in descriptors,
          "no plain-i4 opcode emitted")

    # NUMERICS live in the QEMU parity row (mlp_pt2e), not here: this repo's
    # standard of proof for arithmetic is the emulated device against a host
    # mirror of the same int4 math.
    print("numerics: see the mlp_pt2e row in run_parity.py")


def tiny_eps_case():
    """A chain whose eps has no finite float32 reciprocal (1e-40, a
    subnormal): left in front of a portable reader it is refused by name,
    since computed in place a zero row would come out NaN where the
    converted model gives zeros. Read
    by a Linear the delegate claims, the chain is the runtime's quantiser,
    which takes the NaN to the zero point: that program lowers and computes
    the converted model's zeros."""
    import contextlib
    import io

    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

    from executorch.backends.esp32s3 import Esp32Partitioner
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        DynamicChainRefused,
    )

    class InPlace(nn.Module):
        def forward(self, x):
            return _pt2e_chain(x, True, -127, 127, 1e-40) * 2.0

    class Absorbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc(_pt2e_chain(x, True, -127, 127, 1e-40))

    print("a chain whose eps has no finite float32 reciprocal:")
    x = torch.zeros(2, 16)
    rule = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lower_esp32(torch.export.export(InPlace().eval(), (x,)),
                        [Esp32Partitioner()])
    except DynamicChainRefused as why:
        rule = why.rule
    check("left in place, it is refused by name", rule == "act-eps-in-place",
          f"rule {rule}")
    model = Absorbed().eval()
    keep_out = io.StringIO()
    try:
        with contextlib.redirect_stdout(keep_out):
            edge = lower_esp32(torch.export.export(model, (x,)),
                               [Esp32Partitioner()])
        ok = any("call_delegate" in str(n.target)
                 for n in edge.exported_program().graph.nodes)
        detail = ""
    except Exception as e:  # noqa: BLE001 -- the regression is "this raised"
        ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
    check("absorbed into a claimed Linear, it lowers and delegates", ok, detail)


# One pytest case per *_case, each from seed 0 (the script runs them in this
# order through main).
@case
def test_pt2e_flow():
    torch.manual_seed(0)
    pt2e_flow_case()


@case
def test_shared_activation():
    torch.manual_seed(0)
    shared_activation_case()


@case
def test_strip_scope():
    torch.manual_seed(0)
    strip_scope_case()


@case
def test_decomposed_matmul():
    torch.manual_seed(0)
    decomposed_matmul_case()


@case
def test_ternary_semantic_carrier():
    torch.manual_seed(0)
    ternary_semantic_carrier_case()


@case
def test_conv_pass_through_int8():
    torch.manual_seed(0)
    conv_pass_through_case(weight_bits=8)


@case
def test_conv_pass_through_int4():
    torch.manual_seed(0)
    conv_pass_through_case(weight_bits=4)


@case
def test_readme_flow():
    torch.manual_seed(0)
    readme_flow_case()


@case
def test_portable_chain():
    torch.manual_seed(0)
    portable_chain_case()


@case
def test_gated_strip():
    torch.manual_seed(0)
    gated_strip_case()


@case
def test_shared_view():
    torch.manual_seed(0)
    shared_view_case()


@case
def test_tiny_eps():
    torch.manual_seed(0)
    tiny_eps_case()


def main():
    for test in (test_pt2e_flow, test_shared_activation, test_strip_scope,
                 test_decomposed_matmul, test_ternary_semantic_carrier,
                 test_conv_pass_through_int8, test_conv_pass_through_int4,
                 test_readme_flow, test_portable_chain, test_gated_strip,
                 test_shared_view, test_tiny_eps):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("PT2E reaches the delegate: annotate -> convert -> partition -> match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
