# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QAT round trips: each row of test/fixtures/asym_qat.py -- PT2E static
(arm_a8w8), PT2E dynamic (xnnpack_dynamic), the backend's own scheme,
torchao 8da4w on a nano LM -- exports exactly the grid its fake-quantisers
trained, lowers with nothing portable but what the row keeps, and
runs on the host runner bit for bit as its mirror computes it (the default
scheme's against the unfused mirror: the host builds -ffp-contract=off, the
device fuses that loop's epilogue into madd.s). Prints each row's QAT report:
the trained model's fake-quant forward against the converted model.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import subprocess

import numpy as np
import pytest
import torch

from executorch.backends.esp32s3.test.fixtures import asym_qat as qat
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@pytest.fixture(scope="module")
def rows():
    return {f.name: f for f in qat.fixtures()}


def bits(a):
    return np.asarray(a, F32).reshape(-1).view(np.uint32)


@pytest.mark.parametrize("name", qat.ROWS)
def test_trained_grid_is_exported(rows, name):
    """Every exported scale and zero point is the trained fake-quantiser's,
    bit for bit (calculate_qparams and the buffers training used)."""
    r = rows[name].report
    print(f"\n{name}: {r.steps} steps, loss {r.loss[0]:.4f} -> {r.loss[1]:.4f}; "
          f"grid {r.grid_compared} compared, mismatches {r.grid_mismatches}, "
          f"buffers {r.buffer_mismatches}; fake-quant forward vs converted: "
          f"max abs {r.max_abs:.3g}, {r.flips} of {r.elements} differ")
    assert r.exact(), (r.grid_mismatches, r.buffer_mismatches)
    assert r.loss[1] < r.loss[0], "training did not lower the loss"


@pytest.mark.parametrize("name", qat.ROWS)
def test_row_host_bits_are_the_mirror(runner, tmp_path, rows, name):
    fixture = rows[name]
    et, _ = qat.lower(fixture)
    pte = tmp_path / f"{name}.pte"
    pte.write_bytes(et.buffer)
    command = [str(runner), "--pte", str(pte)]
    for i, x in enumerate(fixture.inputs):
        np.asarray(x, F32).tofile(tmp_path / f"in{i}.bin")
        command += ["--input", str(tmp_path / f"in{i}.bin")]
    command += ["--output", str(tmp_path / "out.bin")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    out = np.fromfile(tmp_path / "out.bin", dtype=F32)
    want = (fixture.host_reference if fixture.host_reference is not None
            else fixture.reference)
    assert np.array_equal(bits(out), bits(want)), (
        f"{np.count_nonzero(bits(out) != bits(want))} of {out.size} differ, "
        f"max abs {np.max(np.abs(out - np.asarray(want, F32).reshape(-1)))}")


def test_fixtures_are_reproducible():
    """Two builds give the same programs whatever ran before them (seeded
    before each model is made, single-threaded training): a row's reference
    does not move between exports."""
    a = [bytes(qat.lower(f)[0].buffer) for f in qat.fixtures()]
    torch.manual_seed(12345)
    torch.rand(1000)                   # disturb the global RNG
    b = [bytes(qat.lower(f)[0].buffer) for f in qat.fixtures()]
    assert a == b


def test_torchao_qat_on_a_nano_llama(tmp_path):
    """The torchao flow on a real transformer: the census's nano llama3_2
    (attention, RMSNorm, RoPE; 2 layers, width 64, vocab 256), prepared with
    esp32_torchao_qat_config(4, 32), trained on random tokens, converted,
    lowered and run on the host runner. Its exported weight grid is the
    trained one, bit for bit, and its output matches the converted model
    within the census bar. Not a QEMU row: its softmax and norms are portable
    float ops with no exact mirror (the nano LM row has none)."""
    import torch
    import torch.nn.functional as F
    from torchao.quantization import quantize_

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
        esp32_torchao_qat_config,
    )
    from executorch.backends.esp32s3.test.quant_census import lower as census_lower
    from executorch.backends.esp32s3.test.quant_census import models, recipes

    if not models.zoo_available():
        pytest.skip("needs the nano zoo and transformers")
    with qat.one_thread():
        net, inputs, _ = models.model("llama3_2", 0)
        keep = recipes._linear_filter(32)
        quantize_(net, esp32_torchao_qat_config(4, 32), filter_fn=keep)
        for p in net.parameters():
            p.requires_grad_(True)
        tokens = inputs[0]
        vocab = int(net(tokens).shape[-1])
        g = torch.Generator().manual_seed(95)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        net.train()
        for _ in range(20):
            batch = torch.randint(0, vocab, tokens.shape, generator=g)
            logits = net(batch)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab),
                                   batch[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            net(tokens)                  # the weight qparams of the final weights
        trained = {name: (m.weight_fake_quantizer.scale.clone(),
                          m.weight_fake_quantizer.zero_point.clone())
                   for name, m in net.named_modules()
                   if hasattr(m, "weight_fake_quantizer")}
        quantize_(net, esp32_torchao_qat_config(4, 32, step="convert"),
                  filter_fn=keep)
        with torch.no_grad():
            reference = net(tokens)
    assert trained, "nothing was QAT-prepared"
    named = dict(net.named_modules())
    for name, (s, z) in trained.items():
        w = named[name].weight
        assert np.array_equal(bits(w.scale.reshape(s.shape).float().numpy()),
                              bits(s.float().numpy())), name
        assert np.array_equal(w.zero_point.reshape(z.shape).long().numpy(),
                              z.long().numpy()), name
    exported = torch.export.export(net, inputs, strict=False)
    rec = {"seconds": {}}
    program = census_lower.lower(rec, exported, tmp_path)
    assert not rec["ops"]["no_kernel"], rec["ops"]["no_kernel"]
    actual = census_lower.run_host(rec, program, inputs, reference)
    got = census_lower.compare(actual, reference, logits=True)
    print(f"\nnano llama3_2 torchao 8da4w QAT: {len(trained)} Linears, grid "
          f"exact; host runner vs converted cosine {got['cosine']}, "
          f"{rec.get('delegates')} delegates")
    assert got["cosine"] >= 0.999, got
