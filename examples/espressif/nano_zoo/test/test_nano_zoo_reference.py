# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny semantic fixtures for the nano zoo's device-arithmetic reference
(nano_zoo/build.py: _device_quantized_model), and its q@kT probe
(build.check_qk)."""

from pathlib import Path

import torch

from executorch.examples.espressif.nano_zoo import build


class GroupedLinear(torch.nn.Linear):
    """Same grouped forward shape as DeepseekV4GroupedLinear, two groups."""

    def forward(self, x):
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        weight = self.weight.view(2, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, 2, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, weight).transpose(0, 1)
        return y.reshape(*input_shape, 2, -1)


class NormedEmbedding(torch.nn.Embedding):
    """A lookup followed by a learned norm, as in InklingNormedEmbedding."""

    def __init__(self):
        super().__init__(16, 8)
        self.embed_norm = torch.nn.LayerNorm(8)

    def forward(self, ids):
        return self.embed_norm(super().forward(ids))


class FunctionalLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 8))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        return torch.nn.functional.linear(input=x, weight=self.weight.float(), bias=self.bias)


def tiny(factory):
    # Count before constructing even these hand-written toy fixtures for real.
    with torch.device("meta"):
        params = build.count_params(factory())
    assert params < 1024
    torch.manual_seed(0)
    return factory().eval().requires_grad_(False)


def test_grouped_projection_keeps_group_axis_and_portable_arithmetic():
    net = tiny(lambda: GroupedLinear(8, 4, bias=False))
    x = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 8) / 7
    expected = net(x)
    build._device_quantized_model(net)
    actual = net(x)
    assert actual.shape == (1, 2, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_int4_embedding_still_applies_learned_post_normalization():
    net = tiny(NormedEmbedding)
    lookup = tiny(lambda: torch.nn.Embedding(16, 8))
    lookup.weight.copy_(net.weight)
    build._device_quantized_model(lookup)
    ids = torch.tensor([[1, 3, 7]])
    expected = net.embed_norm(lookup(ids))
    build._device_quantized_model(net)
    actual = net(ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Raw lookup values would fail both the value comparison and this check.
    torch.testing.assert_close(actual.mean(-1), torch.zeros(1, 3), atol=1e-6, rtol=0)


def test_llama_ci_still_matches_device_arithmetic():
    import json

    from executorch.examples.espressif.nano_zoo import run

    assert run.main(["llama3_2", "--size", "ci"]) == 0
    # Both layers' q@kT are delegated, and each one's scores are its exact
    # mirror's (build.check_qk): the cosine bar alone cannot see them.
    rec = json.loads((run.RESULTS / "llama3_2.ci.json").read_text())
    assert rec["qk"]["ops"] == rec["qk"]["exact_ops"] == 2, rec["qk"]
    assert rec["qk"]["same_output"] and rec["qk"]["exact"], rec["qk"]


def test_qk_probe_leaves_the_zoo_program_as_it_was(tmp_path):
    net, inputs = build.model("llama3_2", "ci")
    plain = build.lower(net, inputs, tmp_path / "plain", qk_probe=False)
    probed = build.lower(net, inputs, tmp_path / "probed")
    assert "qk_probe" not in plain and len(probed["qk_probe"]["ops"]) == 2
    for key in ("pte", "ptd"):
        assert Path(plain[key]).read_bytes() == Path(probed[key]).read_bytes(), key
    # The probe shares the zoo program's weights file when it is the same.
    assert probed["qk_probe"]["ptd"] == probed["ptd"]


def _qk_case(batches=2, m=3, n=5, k=17, seed=7):
    """An exposed q@kT as check_qk sees one: operands, scores, record."""
    import numpy as np

    from executorch.backends.esp32s3.test.reference import bmm_qk_i8

    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batches, m, k)).astype(np.float32)
    keys = (rng.standard_normal((batches, n, k)) * 0.25).astype(np.float32)
    scores = np.stack([bmm_qk_i8(q[b], keys[b]) for b in range(batches)])
    op = {"delegate": "d", "q_shape": [batches, m, k], "k_shape": [batches, n, k],
          "values": {"q": ["output", 1], "k": ["output", 2], "scores": ["output", 3]}}
    logits = np.zeros(4, np.float32)
    return [logits, q, keys, scores], op


def test_qk_check_sees_one_ulp_zeroed_scores_and_swapped_heads():
    import numpy as np

    outputs, op = _qk_case()
    good = build.compare_qk(outputs, [op], outputs[0])
    assert good["exact"] and good["exact_ops"] == 1 and good["rows"][0]["differ"] == 0

    def verdict(scores):
        return build.compare_qk(outputs[:3] + [scores], [op], outputs[0])

    ulp = outputs[3].copy()
    ulp[1, 2, 3] = np.nextafter(ulp[1, 2, 3], np.float32(np.inf))
    assert not verdict(ulp)["exact"] and verdict(ulp)["rows"][0]["differ"] == 1
    assert not verdict(np.zeros_like(outputs[3]))["exact"]
    assert not verdict(outputs[3][::-1].copy())["exact"]
    # A probe whose first output is not the zoo program's ran something else.
    assert not build.compare_qk(outputs, [op], outputs[0] + 1)["exact"]
    # A q@kT the probe could not reach is not a pass.
    unreached = {k: v for k, v in op.items() if k != "values"}
    unreached["unreachable"] = "scores"
    assert not build.compare_qk(outputs, [unreached], outputs[0])["exact"]


def test_functional_parameter_projection_matches_quantized_linear():
    net = tiny(FunctionalLinear)
    control = tiny(lambda: torch.nn.Linear(8, 4))
    control.weight.copy_(net.weight)
    control.bias.copy_(net.bias)
    x = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 7
    build._device_quantized_model(control)
    expected = control(x)
    build._device_quantized_model(net)
    actual = net(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
