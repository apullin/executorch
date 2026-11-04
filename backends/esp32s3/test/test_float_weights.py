# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weights a recipe leaves float stay float: the partitioner's default policy.

In a recipe-quantised program -- the recipe quantised at least one weight or
activation the backend sees (espressif_partitioner._recipe_quantised) -- a
weight the recipe left float keeps its precision: a matmul runs the delegate's
fp32 loop (addmm_f32), a convolution its fp32 convolution (opcode 1 under
FLAG_CONV; test_float_formats.py), a table stays portable at fp32, and the
partitioner names each weight in its report. An explicit format policy, or a
format_overrides entry for one weight, quantises; a float model's weights, and
those of a recipe that quantises no weight and no activation the backend sees,
are the format policy's to quantise.

The block is export_asym_static.FloatLeft (QEMU row st_xnn_float_w):
XNNPACK's static per-channel recipe quantises no embedding and no addmm, so
its table and its HF Conv1D reach the backend float.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import re
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn

from executorch.backends.esp32s3 import _passes
from executorch.backends.esp32s3.test.fixtures import asym_static as eas
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
    _recipe_quantised,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
ROW = next(r for r in eas.SPECS if r.name == "st_xnn_float_w")
KEPT = "esp32 partitioner: kept float, as the recipe left it: "


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@pytest.fixture(scope="module")
def quantised():
    """The row's block through the stock recipe: (module, inputs)."""
    model, inputs, _ = eas.quantized(ROW)
    return model, inputs


def lower(model, inputs, partitioner):
    ep = torch.export.export(model, inputs, strict=False)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        lowered = _passes.lower_esp32(ep, [partitioner])
        _passes.apply_passes(lowered, quiet=True)
    return lowered, log.getvalue()


def descriptors(log):
    """Every descriptor the blobs carry, by kind (addmm_f32, embed_i8, ...)."""
    return re.findall(r"(?:: |, )([A-Za-z][A-Za-z0-9_]*)\[", "\n".join(
        l for l in log.splitlines() if " B blob" in l))


def portable(lowered):
    return [str(n.target) for n in lowered.exported_program().graph.nodes
            if n.op == "call_function" and "call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)]


def kept(log):
    return [l[len(KEPT):] for l in log.splitlines() if l.startswith(KEPT)]


def test_recipe_left_float_weights_stay_float(runner, tmp_path, quantised):
    """The table is gathered portably at fp32 and the Conv1D runs the fp32
    loop, each named once; the host runner's bits are the mirror's, and the
    mirror's are eager's (max abs 0)."""
    model, inputs = quantised
    lowered, log = lower(model, inputs, Esp32Partitioner())
    assert sorted(descriptors(log)) == ["addmmPT_i8", "addmm_f32"], log
    assert any("aten.embedding" in t for t in portable(lowered))
    lines = kept(log)
    assert len(lines) == 2, log
    assert any(l.startswith("table.weight [16x32] fp32 2,048 B")
               and "portable fp32 gather" in l for l in lines), lines
    assert any(l.startswith("conv1d_w [32x32] fp32 4,096 B")
               and "fp32 loop in the delegate" in l for l in lines), lines

    ref = eas.mirror(model, inputs, ROW.luts)
    with torch.no_grad():
        eager = model(*inputs).numpy().astype(F32).reshape(-1)
    assert np.array_equal(ref.reshape(-1).view(np.uint32), eager.view(np.uint32))
    et = lowered.to_executorch()
    pte = tmp_path / "row.pte"
    pte.write_bytes(et.buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, x in enumerate(inputs):
        x.numpy().tofile(tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    command += ["--output", str(tmp_path / "out.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    out = np.fromfile(tmp_path / "out.bin", dtype=F32)
    assert np.array_equal(out.view(np.uint32), eager.view(np.uint32))


def test_explicit_policy_quantises_them(quantised):
    """format_policy given: every float weight it claims is quantised, the
    table included."""
    model, inputs = quantised
    _, log = lower(model, inputs, Esp32Partitioner(format_policy="i8"))
    assert sorted(descriptors(log)) == ["addmmPT_i8", "addmm_i8", "embed_i8"], log
    assert not kept(log)


def test_override_quantises_its_one_weight(quantised):
    """A format_overrides entry quantises the weight it names; the other
    weight the recipe left float stays float."""
    model, inputs = quantised
    _, log = lower(model, inputs,
                   Esp32Partitioner(format_overrides={"conv1d_w": "i4"}))
    assert sorted(descriptors(log)) == ["addmmPT_i8", "addmm_i4"], log
    lines = kept(log)
    assert len(lines) == 1 and lines[0].startswith("table.weight"), lines


def test_float_model_is_the_policys():
    """No recipe: the default policy quantises the float weights at i8 (the
    story and the zoo are such models)."""
    model = eas.FloatLeft().eval()
    inputs = eas.row_inputs(ROW)
    ep = torch.export.export(model, inputs, strict=False)
    assert not _recipe_quantised(ep)
    _, log = lower(model, inputs, Esp32Partitioner())
    assert sorted(descriptors(log)) == ["addmm_i8", "addmm_i8", "embed_i8"], log
    assert not kept(log)


class ConvHead(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.conv = nn.Conv2d(4, 8, 3)
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def lower_dynamic(program, inputs, partitioner):
    """The census's lowering of a dynamic recipe's program: lower_esp32's own
    strip absorbs its activation chains right before partitioning
    (quant_census.lower.strip_before_partitioning checks that it ran)."""
    from executorch.backends.esp32s3.test.quant_census.lower import (
        strip_before_partitioning,
    )

    ep = torch.export.export(program, inputs, strict=False)
    log = io.StringIO()
    with contextlib.redirect_stdout(log), strip_before_partitioning({}):
        lowered = _passes.lower_esp32(ep, [partitioner])
    _passes.apply_passes(lowered, quiet=True)
    return lowered, log.getvalue()


def test_recipe_left_float_conv_stays_float():
    """XNNPACK's dynamic recipe quantises the Linear and leaves the conv
    float. The program is recipe-quantised and convolution is in
    _KEEP_FLOAT_OPS: the conv keeps its fp32 weight in the delegate's fp32
    convolution, delegated, and is named in the report. The explicit i8
    policy quantises it."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    net, x = ConvHead().eval(), (torch.randn(1, 4, 6, 6),)
    q = recipes.quantize("xnnpack-dynamic", net, x, [x])
    default = Esp32Partitioner()
    lowered, log = lower_dynamic(q.program, q.inputs, default)
    assert default.support.keep_float
    lines = kept(log)
    assert len(lines) == 1 and lines[0].startswith("conv.weight [8x4x3x3] "
                                                   "fp32 1,152 B"), lines
    assert "fp32 loop in the delegate" in lines[0], lines
    assert sorted(descriptors(log)) == ["addmmPT_i8", "addmm_f32"], log
    assert " conv3x3[4->8 @4x4]" in log, log
    assert not any("convolution" in t for t in portable(lowered))
    explicit, log = lower_dynamic(q.program, q.inputs,
                                  Esp32Partitioner(format_policy="i8"))
    assert sorted(descriptors(log)) == ["addmmPT_i8", "addmm_i8"], log


def lower_census(recipe, model):
    """A census pair lowered as quant_census.lower does, with the default
    partitioner: (partitioner, backend log)."""
    from executorch.backends.esp32s3.test.quant_census import models, recipes

    net, inputs, calibration = models.model(model)
    q = recipes.quantize(recipe, net, inputs, calibration)
    part = Esp32Partitioner()
    _, log = lower_dynamic(q.program, q.inputs, part)
    return part, log


def test_kv_cache_recipe_alone_is_the_policys():
    """ExecuTorch's KV-cache recipe quantises no weight, and the backend sees
    no activation of its quantised: stock_llm.prepare respells the cache's
    quantize and dequantize as espressif ops before partitioning. So the
    program is not recipe-quantised, and the policy quantises its float
    Linears and table, as in a float model (the census's et-kv-quant).
    The same model with its table and Linears quantised by the recipe
    (--embedding_quantize with 8da4w) is recipe-quantised."""
    part, log = lower_census("et-kv-quant", "et_llama")
    assert not part.support.keep_float
    assert not kept(log), log
    ops = descriptors(log)
    assert ops.count("addmm_i8") == 15 and ops.count("embed_i8") == 1, ops
    part, _ = lower_census("et-emb-8da4w", "et_llama")
    assert part.support.keep_float


class PairOnly(nn.Module):
    """A static pair, hand-written as convert_pt2e spells it, in front of a
    float Linear: the recipe quantised an activation the backend sees and no
    weight."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.fc = nn.Linear(32, 16)

    def forward(self, x):
        qd = torch.ops.quantized_decomposed
        i8 = (-128, 127, torch.int8)
        s = float(F32(2.0 ** -5))
        return self.fc(qd.dequantize_per_tensor(
            qd.quantize_per_tensor(x, s, 0, *i8), s, 0, *i8))


def test_activation_alone_is_recipe_quantised():
    """A quantised activation the backend sees makes the program
    recipe-quantised: the Linear's float weight stays float, in the fp32
    loop, reading the pair's dequantised value."""
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

    model, x = PairOnly().eval(), (torch.randn(4, 32),)
    part = Esp32Partitioner()
    lowered, log = lower(model, x, part)
    assert part.support.keep_float
    assert descriptors(log) == ["addmm_f32"], log
    lines = kept(log)
    assert len(lines) == 1 and lines[0].startswith("fc.weight [16x32]"), lines
    assert any("quantize_per_tensor" in t for t in portable(lowered))
