# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training through a fitted int16 table (quantizer/lut16_train.py).

  fit         fit_tables swaps a QAT- or PTQ-prepared model's int16 site for
              a Lut16 module, freezes the static activation observers (not
              the weights'), fits over the observer's window (or every code),
              and leaves a site the budget refuses as it was
  forward     Lut16 is espressif::lut16 on its own arguments: bit for bit
              mirrors.lut16
  backward    grad="true": the activation's derivative at dq1(c), 0 where x
              quantises outside q1's range; grad="table": the deployed
              function's slope; learn=True: the entries get gradients and
              `table` always holds what the forward ran
  round trip  train, convert_pt2e (lut16_N stays where it is), torch.export
              (espressif::lut16 on the module's buffer, the same bytes),
              lower_esp32 (the same bytes in the program, no refit), and the
              lowered program's torch reference == the trained graph
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.quantizer import lut_fit
from executorch.backends.esp32s3.quantizer import lut16_train as T
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

torch.set_num_threads(1)


class SiluMlp(nn.Module):
    def __init__(self, k=32, hidden=48, n=16):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(k, hidden), nn.Linear(hidden, n)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


def _prepared(is_qat=True, seed=0):
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e

    torch.manual_seed(seed)
    net = SiluMlp()
    x = torch.randn(4, 32)
    gm = torch.export.export(net, (x,), strict=False).module(check_guards=False)
    quantizer = Esp32Quantizer(preset="arm_a16w8", is_qat=is_qat)
    prepared = (prepare_qat_pt2e if is_qat else prepare_pt2e)(gm, quantizer)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _ in range(4):
            prepared(torch.randn(4, 32, generator=gen))
    return prepared


def _calls(gm, kind):
    return [n for n in gm.graph.nodes if n.op == "call_function" and kind in str(n.target)]


def _mirror(mod, x):
    return qm.lut16(x.detach().numpy().astype(np.float32), mod.table.detach().numpy(),
                    *mod.scalars)


@pytest.mark.parametrize("is_qat", [True, False])
def test_fit_tables_swaps_the_site(is_qat):
    prepared = _prepared(is_qat)
    report = T.fit_tables(prepared)
    assert len(report.tables) == 1 and not report.refused, report.summary()
    assert not _calls(prepared, "silu")
    mods = T.tables(prepared)
    assert list(mods) == ["lut16_0"]
    mod = mods["lut16_0"]
    lo, hi = mod.scalars[5], mod.scalars[6]
    assert mod.scalars[2] < lo <= hi < mod.scalars[3], "fitted over the window"
    # activation observers frozen, weight observers live
    for name, m in prepared.named_modules():
        if hasattr(m, "observer_enabled"):
            reads_weight = any(n.op == "call_module" and n.target == name
                               and n.args[0].op == "get_attr" for n in prepared.graph.nodes)
            assert int(m.observer_enabled[0]) == (1 if reads_weight else 0), name
    whole = _prepared(is_qat)
    T.fit_tables(whole, ranges=False)
    full = T.tables(whole)["lut16_0"]
    assert full.scalars[5] <= lo and full.scalars[6] >= hi
    assert (full.table.numel() * full.table.element_size()
            > mod.table.numel() * mod.table.element_size())


def test_budget_refusal_keeps_the_activation():
    prepared = _prepared()
    report = T.fit_tables(prepared, lut_fit.Spec(budget=8), ranges=False)
    assert [r[2] for r in report.refused] == ["lut16-budget"] and not report.tables
    assert len(_calls(prepared, "silu")) == 1


def _site_inputs(mod, n=4096, seed=0):
    s1 = mod.in_scale
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(0, 1, n), (np.arange(-50, 50) + 0.5) * s1,
                        np.array([1e6, -1e6, 9.0, -9.0]), (mod.scalars[5] + np.arange(-3, 4)) * s1,
                        (mod.scalars[6] + np.arange(-3, 4)) * s1])
    return torch.from_numpy(x.astype(np.float32))


@pytest.mark.parametrize("learn", [False, True])
def test_forward_is_the_mirror(learn):
    prepared = _prepared()
    T.fit_tables(prepared, learn=learn)
    mod = T.tables(prepared)["lut16_0"]
    x = _site_inputs(mod)
    with torch.no_grad():
        plain = mod(x)
    trained = mod(x.clone().requires_grad_(True))
    want = _mirror(mod, x)
    for got in (plain, trained):
        assert np.array_equal(got.detach().numpy().view(np.uint32), want.view(np.uint32))


def test_true_gradient_is_the_straight_through_estimator():
    prepared = _prepared()
    T.fit_tables(prepared)
    mod = T.tables(prepared)["lut16_0"]
    x = _site_inputs(mod).requires_grad_(True)
    mod(x).sum().backward()
    inv, zp, qmin, qmax = mod.scalars[:4]
    raw = torch.round(x.detach() * inv) + zp
    c = raw.clamp(qmin, qmax)
    xs = ((c - zp) * mod.in_scale).requires_grad_(True)
    F.silu(xs).sum().backward()
    want = xs.grad * ((raw >= qmin) & (raw <= qmax))
    assert torch.equal(x.grad, want)
    assert (x.grad[x.detach().abs() > 1e5] == 0).all()


def test_table_gradient_is_the_deployed_slope():
    prepared = _prepared()
    T.fit_tables(prepared, grad="table")
    mod = T.tables(prepared)["lut16_0"]
    lo, hi = mod.scalars[5], mod.scalars[6]
    c = torch.arange(lo + 160, hi - 160, 97, dtype=torch.float32)
    x = (c * mod.in_scale).requires_grad_(True)
    mod(x).sum().backward()
    # the table's own central difference over +-128 codes, in output units
    step = 128 * mod.in_scale
    with torch.no_grad():
        fd = (mod(x.detach() + step) - mod(x.detach() - step)) / (2 * step)
    assert torch.allclose(x.grad, fd, atol=0.05, rtol=0.05)
    beyond = (torch.tensor([hi + 50.0, lo - 50.0]) * mod.in_scale).requires_grad_(True)
    mod(beyond).sum().backward()
    assert (beyond.grad == 0).all()


def test_learned_entries_get_gradients_and_stay_device_typed():
    prepared = _prepared()
    T.fit_tables(prepared, learn=True)
    mod = T.tables(prepared)["lut16_0"]
    before = mod.table.clone()
    opt = torch.optim.SGD([mod.nodes], lr=1e4)
    x = _site_inputs(mod)
    loss = (mod(x) - F.silu(x) * 1.01).pow(2).sum()
    loss.backward()
    assert mod.nodes.grad is not None and mod.nodes.grad.abs().sum() > 0
    opt.step()
    with torch.enable_grad():
        y = mod(x.clone().requires_grad_(True))
    assert mod.table.dtype == before.dtype
    assert not torch.equal(mod.table, before)
    assert np.array_equal(y.detach().numpy().view(np.uint32), _mirror(mod, x).view(np.uint32))
    T.check_tables(prepared)


def _train(prepared, steps=20, seed=3):
    torch.manual_seed(seed)
    teacher = SiluMlp()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in prepared.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(steps):
        x = torch.randn(16, 32, generator=gen)
        with torch.no_grad():
            y = teacher(x)
        loss = (prepared(x) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


@pytest.mark.parametrize("learn", [False, True])
def test_round_trip_keeps_the_table(learn):
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.exir import ExecutorchBackendConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

    prepared = _prepared()
    report = T.fit_tables(prepared, learn=learn)
    _train(prepared)
    T.check_tables(prepared)
    trained = T.tables(prepared)["lut16_0"].table.clone()
    converted = convert_pt2e(prepared)
    assert [n.target for n in converted.graph.nodes if n.op == "call_module"] == ["lut16_0"]
    assert torch.equal(T.tables(converted)["lut16_0"].table, trained)
    x = torch.randn(4, 32, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        eager = converted(x)
    ep = torch.export.export(converted, (x,), strict=False)
    (node,) = _calls(ep.graph_module, "lut16")
    table_arg = node.args[1]
    spec = next(s for s in ep.graph_signature.input_specs if s.arg.name == table_arg.name)
    exported = (ep.state_dict.get(spec.target) if spec.target in ep.state_dict
                else ep.constants[spec.target])
    assert exported.dtype == trained.dtype
    assert exported.numpy().tobytes() == trained.numpy().tobytes()
    with torch.no_grad():
        assert torch.equal(ep.module()(x), eager)
    lowered = _passes.lower_esp32(ep, [Esp32Partitioner(luts16=True)])
    assert not lowered.lut16_report.tables, "no refit: nothing left to fold"
    _passes.apply_passes(lowered, quiet=True)
    edge = lowered.exported_program()
    (node,) = [n for n in edge.graph_module.graph.nodes
               if n.op == "call_function" and "lut16" in str(n.target)]
    name = node.args[1].name
    spec = next(s for s in edge.graph_signature.input_specs if s.arg.name == name)
    kept = (edge.state_dict.get(spec.target) if spec.target in edge.state_dict
            else edge.constants[spec.target])
    assert kept.numpy().tobytes() == trained.numpy().tobytes()
    program = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    assert trained.numpy().tobytes() in bytes(program.buffer)
    assert report.tables[0][1].table.scheme in lut_fit.SCHEMES
