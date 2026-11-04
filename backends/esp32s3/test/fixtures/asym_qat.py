# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QAT round trips: train a nano model with fake quantisation, convert,
export and lower it, and hold the device to the grid it trained on.

Each row trains a few steps on seeded random data, freezes its
fake-quantisers, converts and exports. Three bars, the first two exact:

  grid    every scale and zero point the exported program carries equals the
          trained fake-quantiser's, bit for bit -- calculate_qparams() (what
          convert_pt2e exports) AND the buffers the last training forward
          used (scale, zero_point). For torchao, the weight fake-quantiser's
          qparams on the final weights against the converted weights'.
  device  the device's output bits are the mirror's on the lowered program
          (run_parity), and so are the host runner's (test_qat.py).
  QAT     the trained model's fake-quant forward against the converted
          model's output: largest difference, and output codes that flip
          (static rows, in the output pair's steps) or elements that differ
          (dynamic rows). Reported, not a bar: the forward runs fake-quant
          kernels, the converted model quantize/dequantize ops.

  qat_pt2e_static    Esp32Quantizer(preset="arm_a8w8", is_qat=True) on a nano
                     CNN: conv 3x3 + batch norm + ReLU (QAT keeps the norm;
                     convert folds it into the conv), conv 3x3 stride 2 +
                     ReLU, Linear; int32 biases
  qat_pt2e_dynamic   Esp32Quantizer(preset="xnnpack_dynamic", is_qat=True) on a
                     nano MLP: dynamic affine activations (eps 2^-12) through
                     FakeQuantize over MovingAverageMinMaxObserver(
                     averaging_constant=1), int8 per-channel weights
  qat_pt2e_default   Esp32Quantizer(is_qat=True): the backend's own scheme --
                     int4 per-channel weights, FLAG_ACT_PER_TENSOR -- whose
                     device loop fuses its epilogue into madd.s, so the device
                     is held to the fused mirror and the host runner to the
                     unfused one
  qat_torchao_8da4w  esp32_torchao_qat_config(4, 32) on a nano language model
                     (token embedding, two residual MLP blocks, a head; no
                     attention, so every op the device runs has an exact
                     mirror), trained on random tokens; the row is its body,
                     everything after the embedding lookup, on embedded
                     activations

export_all(out_dir) writes every program and the packed image main.cpp reads
from the asymqat flash partition (asym_qat_fixtures.bin; asym.py's
layout); it returns {row: reference output}, and each row's report is on
export_all.reports. Imports torch.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.fixtures.asym import ramp
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32
IMAGE = "asym_qat_fixtures.bin"
ROWS = ("qat_pt2e_static", "qat_pt2e_dynamic", "qat_pt2e_default",
        "qat_torchao_8da4w")
QEMU_ROWS = ROWS
# Ops a row keeps outside its delegates on purpose.
PORTABLE = {"qat_pt2e_static": ("aten.view_copy",),
            "qat_torchao_8da4w": ("aten.add.Tensor", "aten.view_copy")}
# The symmetric path's epilogue: madd.s on the device, unfused on the host.
HOST_UNFUSED = ("qat_pt2e_default",)


# --- the nano models ------------------------------------------------------------------

class NanoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.fc = nn.Linear(8 * 5 * 5, 10)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.conv2(h))
        return self.fc(h.flatten(1))


class NanoMLP(nn.Module):
    def __init__(self, k=64, hidden=48, n=32):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(k, hidden), nn.Linear(hidden, n)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up, self.down = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return x + self.down(torch.relu(self.up(x)))


class NanoLM(nn.Module):
    """Token embedding, residual MLP blocks, a head: a language model with
    no attention."""

    def __init__(self, vocab=256, d=64, hidden=128, blocks=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.body = nn.Sequential(*(Block(d, hidden) for _ in range(blocks)),
                                  nn.Linear(d, vocab))

    def forward(self, tokens):
        return self.body(self.embed(tokens))


# --- the QAT flows ----------------------------------------------------------------------

@dataclass
class Report:
    """One row's QAT bars."""

    grid_compared: int = 0          # scales/zero points compared
    grid_mismatches: list = field(default_factory=list)
    buffer_mismatches: list = field(default_factory=list)
    max_abs: float = 0.0            # fake-quant forward vs converted output
    flips: int = 0                  # output codes (static) / elements (dynamic)
    elements: int = 0
    steps: int = 0                  # training steps
    loss: tuple = ()                # held-out loss before and after training

    def exact(self):
        return (self.grid_compared > 0 and not self.grid_mismatches
                and not self.buffer_mismatches)


@contextlib.contextmanager
def one_thread():
    """Training and calibration single-threaded: their float reductions then
    do not vary from run to run, so every export of a row is the same
    program."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _attr(gm, target):
    """A graph module's constant by its (possibly dotted) get_attr target."""
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _f32_bits(t):
    return np.asarray(t.detach().to(torch.float32).reshape(-1).numpy()).view(np.uint32)


def _fake_quantisers(prepared):
    """(call node, module) of every observer/fake-quantiser the prepared graph
    calls, in graph order."""
    return [(n, prepared.get_submodule(n.target)) for n in prepared.graph.nodes
            if n.op == "call_module"]


def _freeze(prepared):
    """Observers off: the grid stops moving, so the next forward and convert
    both use the trained qparams. Not on a dynamic activation's fake-
    quantiser: its range is the call's own, as the chooser's is at run time."""
    for _, m in _fake_quantisers(prepared):
        if hasattr(m, "disable_observer") and not getattr(m, "is_dynamic", False):
            m.disable_observer()


def _trained_grid(prepared):
    """[(kind, scale, zero point, scale buffer, zero-point buffer)] per fake-
    quantiser, graph order. kind: 'act' (per tensor), 'weight' (per channel),
    'bias' (derived int32)."""
    out = []
    for node, m in _fake_quantisers(prepared):
        name = type(m).__name__
        if "Derived" in name:
            s, z = m.calculate_qparams()
            out.append(("bias", s, z, None, None))
            continue
        if not hasattr(m, "calculate_qparams"):
            continue
        if getattr(m, "is_dynamic", False):
            # A dynamic activation: no trained constants, a trained scheme.
            obs = m.activation_post_process
            out.append(("dynamic", (int(obs.quant_min), int(obs.quant_max),
                                    float(obs.eps)), None, None, None))
            continue
        s, z = m.calculate_qparams()
        per_channel = getattr(m, "is_per_channel", False) or s.numel() > 1
        scale, zp = getattr(m, "scale", None), getattr(m, "zero_point", None)
        out.append(("weight" if per_channel else "act", s, z,
                    None if scale is None else scale.clone(),
                    None if zp is None else zp.clone()))
    return out


def _exported_grid(converted):
    """The converted graph's qparams, graph order: static activation pairs
    (scale, zero point), dynamic choosers (qmin, qmax, eps), per-channel
    weight/bias dequantisations (scales, zero points)."""
    acts, chans, choosers = [], [], []
    for n in converted.graph.nodes:
        if n.op != "call_function":
            continue
        t = str(n.target)
        if "choose_qparams" in t:
            _x, qmin, qmax, eps = n.args[:4]
            choosers.append((int(qmin), int(qmax), float(eps)))
        elif "quantize_per_tensor.default" in t and "dequantize" not in t:
            acts.append((torch.tensor([n.args[1]], dtype=torch.float32),
                         torch.tensor([int(n.args[2])])))
        elif "dequantize_per_channel" in t:
            get = (lambda a: _attr(converted, a.target)
                   if isinstance(a, torch.fx.Node) else a)
            chans.append((get(n.args[1]), get(n.args[2]),
                          get(n.args[0]).dtype))
    return acts, chans, choosers


def _compare_grids(trained, converted, report):
    """trained's qparams against the exported program's, bitwise."""
    acts, chans, choosers = _exported_grid(converted)
    t_acts = [t for t in trained if t[0] == "act"]
    t_chans = [t for t in trained if t[0] in ("weight", "bias")]
    t_dyn = [t[1] for t in trained if t[0] == "dynamic"]
    if t_dyn or choosers:
        report.grid_compared += len(t_dyn)
        if sorted(t_dyn) != sorted(choosers):
            report.grid_mismatches.append(
                f"dynamic schemes: trained {sorted(set(t_dyn))} exported "
                f"{sorted(set(choosers))}")

    def same(a, b):
        return (a.numel() == b.numel()
                and np.array_equal(_f32_bits(a), _f32_bits(b)))

    def same_int(a, b):
        return torch.equal(a.reshape(-1).to(torch.int64), b.reshape(-1).to(torch.int64))

    if len(t_acts) != len(acts):
        report.grid_mismatches.append(f"{len(t_acts)} activation fake-quantisers, "
                                      f"{len(acts)} exported pairs")
    for i, (t, e) in enumerate(zip(t_acts, acts)):
        report.grid_compared += 2
        if not (same(t[1], e[0]) and same_int(t[2], e[1])):
            report.grid_mismatches.append(
                f"activation {i}: trained ({float(t[1].reshape(-1)[0])!r}, "
                f"{int(t[2].reshape(-1)[0])}) exported ({float(e[0][0])!r}, "
                f"{int(e[1][0])})")
        if t[3] is not None and not (same(t[3], t[1]) and same_int(t[4], t[2])):
            report.buffer_mismatches.append(f"activation {i}")
    if len(t_chans) != len(chans):
        report.grid_mismatches.append(f"{len(t_chans)} weight/bias quantisers, "
                                      f"{len(chans)} exported dequantisations")
    for i, (t, e) in enumerate(zip(t_chans, chans)):
        report.grid_compared += 2
        if not (same(t[1], e[0]) and same_int(t[2], e[1])):
            report.grid_mismatches.append(f"{t[0]} {i}: scales or zero points")
        if t[3] is not None and not (same(t[3], t[1]) and same_int(t[4], t[2])):
            report.buffer_mismatches.append(f"{t[0]} {i}")


def _gap(fq_out, conv_out, report, out_qparams=None):
    a = fq_out.detach().to(torch.float32).reshape(-1).numpy()
    b = conv_out.detach().to(torch.float32).reshape(-1).numpy()
    report.max_abs = float(np.max(np.abs(a - b), initial=0.0))
    report.elements = int(a.size)
    if out_qparams is not None:
        s, z = out_qparams
        ca = np.rint(a / F32(s)).astype(np.int64)
        cb = np.rint(b / F32(s)).astype(np.int64)
        report.flips = int(np.count_nonzero(ca != cb))
    else:
        report.flips = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


def _output_qparams(converted):
    """(scale, zero point) of the pair the graph output leaves through."""
    out = next(n for n in converted.graph.nodes if n.op == "output").args[0]
    node = out[0] if isinstance(out, (tuple, list)) else out
    if "dequantize_per_tensor" in str(node.target):
        return float(node.args[1]), int(node.args[2])
    return None


def pt2e_qat(build, quantizer, shape, *, steps, data, loss_fn, seed, lr):
    """prepare_qat_pt2e -> train `steps` -> freeze -> (converted, report).
    `build()` makes the float model (after seeding, so its weights do not
    depend on what ran before in the process); `data(generator) -> (x,
    target)` draws one seeded batch of `shape`."""
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import (convert_pt2e,
                                                         prepare_qat_pt2e)

    report = Report(steps=steps)
    with one_thread():
        torch.manual_seed(seed)
        model = build()
        g = torch.Generator().manual_seed(seed + 1)
        example = (ramp(shape),)
        gm = torch.export.export(model.train(), example, strict=False).module(
            check_guards=False)
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = prepare_qat_pt2e(gm, quantizer)
        held_out = [data(torch.Generator().manual_seed(seed + 2 + i))
                    for i in range(16)]

        def held_out_loss():
            with torch.no_grad():
                return float(np.mean([float(loss_fn(prepared(x), t))
                                      for x, t in held_out]))

        before = held_out_loss()
        opt = torch.optim.SGD(prepared.parameters(), lr=lr, momentum=0.9)
        for _ in range(steps):
            x, target = data(g)
            loss = loss_fn(prepared(x), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        report.loss = (before, held_out_loss())
        _freeze(prepared)
        move_exported_model_to_eval(prepared)
        trained = _trained_grid(prepared)
        x = ramp(shape)
        with torch.no_grad():
            fq_out = prepared(x)
        # In place: a deep copy's annotations would key on the original nodes.
        converted = convert_pt2e(prepared)
        with torch.no_grad():
            conv_out = converted(x)
    _compare_grids(trained, converted, report)
    _gap(fq_out, conv_out, report, _output_qparams(converted))
    return converted, report


def _teacher_data(shape, n, seed, scale, shift):
    """Seeded random inputs over the device ramp's range and the targets a
    fixed random linear teacher gives them: something to learn, so the loss
    falls."""
    t = torch.Generator().manual_seed(seed)
    k = int(np.prod(shape[1:]))
    w = torch.randn((k, n), generator=t) / k ** 0.5

    def data(g):
        x = torch.randn(shape, generator=g) * scale + shift
        return x, torch.tanh(x.reshape(shape[0], -1) @ w)
    return data


def _static_row():
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    shape = (1, 4, 9, 9)
    return pt2e_qat(NanoCNN, Esp32Quantizer(preset="arm_a8w8", is_qat=True),
                    shape, steps=40,
                    data=_teacher_data(shape, 10, 191, 0.8, 0.75),
                    loss_fn=F.mse_loss, seed=91, lr=0.005)


def _dynamic_row():
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    shape = (4, 64)
    return pt2e_qat(lambda: NanoMLP(64, 48, 32),
                    Esp32Quantizer(preset="xnnpack_dynamic", is_qat=True),
                    shape, steps=40,
                    data=_teacher_data(shape, 32, 192, 1.2, 1.0),
                    loss_fn=F.mse_loss, seed=92, lr=0.02)


def _default_row():
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    shape = (4, 96)
    return pt2e_qat(lambda: NanoMLP(96, 64, 32), Esp32Quantizer(is_qat=True),
                    shape,
                    steps=40, data=_teacher_data(shape, 32, 193, 1.2, 1.0),
                    loss_fn=F.mse_loss, seed=93, lr=0.02)


def torchao_qat_lm(steps=40, seed=94, vocab=256, seq=8):
    """The nano LM through esp32_torchao_qat_config(4, 32): prepare, train on
    random next-token pairs, then (trained weight qparams on the final
    weights, the LM's fake-quant body, the converted LM)."""
    from torchao.quantization import quantize_

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
        esp32_torchao_qat_config,
    )

    report = Report(steps=steps)
    with one_thread():
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed + 1)
        lm = NanoLM(vocab).train()
        quantize_(lm, esp32_torchao_qat_config(4, 32))
        # Random tokens drawn from a fixed skewed distribution: a unigram to
        # learn, so the loss falls.
        weights = torch.rand(vocab, generator=torch.Generator().manual_seed(
            seed + 3)) ** 4

        def batch(gen):
            return torch.multinomial(weights, seq + 1, replacement=True,
                                     generator=gen).reshape(1, -1)

        def loss_of(tokens):
            logits = lm(tokens[:, :-1])
            return F.cross_entropy(logits.reshape(-1, vocab),
                                   tokens[:, 1:].reshape(-1))

        held_out = [batch(torch.Generator().manual_seed(seed + 4 + i))
                    for i in range(16)]

        def held_out_loss():
            with torch.no_grad():
                return float(np.mean([float(loss_of(t)) for t in held_out]))

        before = held_out_loss()
        opt = torch.optim.Adam(lm.parameters(), lr=5e-3)
        for _ in range(steps):
            loss = loss_of(batch(g))
            opt.zero_grad()
            loss.backward()
            opt.step()
        report.loss = (before, held_out_loss())
        lm.eval()
        x = ramp((1, seq, 64))
        with torch.no_grad():
            fq_out = lm.body(x)          # refreshes the weight qparams too
        trained = [(name, m.weight_fake_quantizer.scale.clone(),
                    m.weight_fake_quantizer.zero_point.clone())
                   for name, m in lm.named_modules()
                   if hasattr(m, "weight_fake_quantizer")]
        quantize_(lm, esp32_torchao_qat_config(4, 32, step="convert"))
        with torch.no_grad():
            conv_out = lm.body(x)
    named = dict(lm.named_modules())
    for name, s, z in trained:
        w = named[name].weight
        report.grid_compared += 2
        if not (np.array_equal(_f32_bits(w.scale.reshape(s.shape)), _f32_bits(s))
                and torch.equal(w.zero_point.reshape(z.shape).to(torch.int64),
                                z.to(torch.int64))):
            report.grid_mismatches.append(name)
    _gap(fq_out, conv_out, report)
    return lm, report


# --- the device's arithmetic ------------------------------------------------------------

def static_reference(converted, x):
    """StaticMirror on the converted program (no tables: nothing between the
    delegates needs one)."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as eas

    return eas.mirror(converted, (x,), luts=False)


def _weights(converted, node):
    """(codes [N, K] int64, scales [N], zero points or None) of a converted
    linear's dequantised weight."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as eas

    d = eas._behind_moves(node.args[1])
    codes = _attr(converted, d.args[0].target).to(torch.int64).numpy()
    scales = _attr(converted, d.args[1].target).to(torch.float32).numpy()
    zeros = _attr(converted, d.args[2].target).to(torch.int64).numpy()
    return codes, scales, (zeros if np.any(zeros) else None)


def _chain_scheme(node):
    """The ActScheme of the dynamic chain in front of a converted linear."""
    from executorch.backends.esp32s3.test.fixtures import asym_static as eas

    dq = eas._behind_moves(node.args[0])
    chooser = dq.args[1].args[0]
    _x, qmin, qmax, eps, _dtype = chooser.args[:5]
    symmetric = "symmetric" in str(chooser.target)
    return qm.ActScheme("pt2e/symmetric" if symmetric else "pt2e/affine",
                        "symmetric" if symmetric else "affine", "tensor",
                        "rne_before_zp", int(qmin), int(qmax), eps=float(eps))


def _linears(converted):
    return [n for n in converted.graph.nodes
            if n.op == "call_function" and "aten.linear" in str(n.target)]


def _bias(converted, node):
    b = node.args[2] if len(node.args) > 2 else None
    return None if b is None else _attr(converted, b.target).detach().numpy().astype(F32)


def dynamic_reference(converted, x):
    """Two Linears under the chain's PT2E affine scheme, ReLU between:
    mirrors.affine_matmul, the device's affine loop."""
    h = np.asarray(x, F32)
    layers = _linears(converted)
    for i, node in enumerate(layers):
        codes, scales, zeros = _weights(converted, node)
        h = qm.affine_matmul(h, codes, scales, _bias(converted, node),
                             act=_chain_scheme(node), bits=8,
                             zero_points=zeros).y
        if i < len(layers) - 1:
            h = np.maximum(h, F32(0))
    return h


def default_reference(converted, x, fused_madd=True):
    """The backend's own scheme (FLAG_ACT_PER_TENSOR, int4 per channel): the
    symmetric path's mirror, models.addmm_codes_ref -- its epilogue fused
    into madd.s on the device (fused_madd=True), unfused on the host."""
    from executorch.backends.esp32s3.test.reference import addmm_codes_ref

    h = np.asarray(x, F32)
    layers = _linears(converted)
    for i, node in enumerate(layers):
        codes, scales, zeros = _weights(converted, node)
        assert zeros is None and _chain_scheme(node).eps == qm.FLT_EPSILON
        h = addmm_codes_ref(h, codes, scales.reshape(-1, 1),
                            _bias(converted, node), codes.shape[1],
                            per_tensor=True, fused_madd=fused_madd)
        if i < len(layers) - 1:
            h = np.maximum(h, F32(0))
    return h


def lm_reference(lm, x):
    """The converted LM's body the device's way: A3 per token in front of
    every Linear (mirrors.affine_matmul over the int4 group-32 codes),
    ReLU, and each block's residual added in float32."""
    from executorch.backends.esp32s3.test.fixtures.asym_act import torchao_layer

    def mm(h, layer):
        codes, scales, zeros, group, bias = torchao_layer(layer)
        return qm.affine_matmul(h, codes.astype(np.int64), scales, bias,
                                act=qm.A3_TORCHAO, bits=4, gsize=group,
                                zero_points=zeros).y

    h = np.asarray(x, F32).reshape(-1, x.shape[-1])
    for block in lm.body[:-1]:
        h = (h + mm(np.maximum(mm(h, block.up), F32(0)), block.down)).astype(F32)
    return mm(h, lm.body[-1]).reshape(*x.shape[:-1], -1)


# --- the table --------------------------------------------------------------------------

@dataclass
class Fixture(ea.Fixture):
    report: Report | None = None
    host_reference: np.ndarray | None = None


def fixtures():
    out = []
    for name, build, ref in (("qat_pt2e_static", _static_row, static_reference),
                             ("qat_pt2e_dynamic", _dynamic_row, dynamic_reference),
                             ("qat_pt2e_default", _default_row, default_reference)):
        converted, report = build()
        x = ramp(tuple(converted.graph.find_nodes(op="placeholder")[0]
                       .meta["val"].shape))
        fixture = Fixture(name, converted, (x,), ref(converted, x), {},
                          report=report)
        if name in HOST_UNFUSED:
            fixture.host_reference = default_reference(converted, x, False)
        out.append(fixture)
    lm, report = torchao_qat_lm()
    x = ramp((1, 8, 64))
    out.append(Fixture("qat_torchao_8da4w", lm.body.eval(), (x,),
                       lm_reference(lm, x), {}, report=report))
    assert tuple(f.name for f in out) == ROWS
    return out


def lower(fixture):
    """The fixture through lower_esp32 (dynamic chains absorbed by
    strip_dynamic_act_qdq, static pairs by the pre-partition pass) and
    apply_passes; nothing left outside the delegates but what the row keeps.
    Returns (program, the backend's log)."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import fold_conv_bn_pass
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3._passes import strip_dynamic_act_qdq_pass
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = torch.export.export(fixture.model, fixture.inputs, strict=False)
    log = io.StringIO()
    fold = fold_conv_bn_pass.fold_conv_bn

    def absorb_then_fold(edge, method_name="forward"):
        strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq(edge, method_name)
        return fold(edge, method_name)

    _passes.fold_conv_bn = absorb_then_fold
    try:
        with contextlib.redirect_stdout(log):
            lowered = _passes.lower_esp32(ep, [Esp32Partitioner()],
                                            quant_luts=False)
            _passes.apply_passes(lowered, quiet=True)
    finally:
        _passes.fold_conv_bn = fold
    graph = lowered.exported_program().graph_module.graph
    keeps = PORTABLE.get(fixture.name, ())
    left = [str(n.target) for n in graph.nodes if n.op == "call_function"
            and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)
            and not any(k in str(n.target) for k in keeps)]
    if left:
        raise RuntimeError(f"{fixture.name}: ops left portable: {left}\n"
                           f"{log.getvalue()}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue()


def emitted_programs():
    """Every row's emitted program (dump_required_ops)."""
    return [lower(f)[0] for f in fixtures()]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte) and the QEMU image (asym_qat_fixtures.bin).
    Returns {row: reference}; each row's QAT report is on export_all.reports."""
    table = fixtures()
    programs, refs = {}, {}
    export_all.reports = {}
    for fixture in table:
        refs[fixture.name] = fixture.reference
        export_all.reports[fixture.name] = fixture.report
        et, log = lower(fixture)
        programs[fixture.name] = bytes(et.buffer)
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(et.buffer)
        if verbose:
            r = fixture.report
            ops = [line.split(": ", 1)[-1] for line in log.splitlines()
                   if "[Esp32Backend]" in line]
            print(f"  {fixture.name}.pte: {len(et.buffer):,} B; grid "
                  f"{'exact' if r.exact() else 'MISMATCH'} ({r.grid_compared} "
                  f"compared); fake-quant vs converted max abs {r.max_abs:.3g}, "
                  f"{r.flips} of {r.elements} differ; " + "; ".join(ops))
    ea.write_image(os.path.join(out_dir, IMAGE), table, programs)
    return refs


if __name__ == "__main__":
    for name, ref in export_all().items():
        r = export_all.reports[name]
        print(f"  {name}: reference {np.shape(ref)}; loss {r.loss[0]:.3f} -> "
              f"{r.loss[1]:.3f}; grid {r.exact()}")
