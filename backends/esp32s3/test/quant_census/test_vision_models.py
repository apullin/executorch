# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The census's vision models: MobileNetV2 and MobileNetV3-Small as torchvision
builds them, the board benchmark's models
(examples/espressif/esp32s3/scripts/run_bench_hardware.py).

torchvision is not installed here, so the checks are torchvision's published
facts: each model's parameter count (torchvision.models' documentation), its
layer structure (depthwise convolutions, residual adds, strides), its
activations and its BatchNorm epsilons. The 96-pixel MobileNetV2 runs one
census cell, the benchmark's recipe; the two 224-pixel models run none.

    python -m pytest backends/esp32s3/test/quant_census/test_vision_models.py
"""

import pytest
import torch
import torch.nn as nn

from executorch.backends.esp32s3.test.quant_census import covered, pairs
from executorch.backends.esp32s3.test.quant_census import models

# torchvision.models.mobilenet_v2(), .mobilenet_v3_small(): num_params.
PARAMS = {"mobilenet_v2_96": 3_504_872, "mobilenet_v2_224": 3_504_872,
          "mobilenet_v3_small": 2_542_856}


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_parameter_count_is_torchvisions(name):
    assert models.meta_params(name) == PARAMS[name]
    assert models.cap(name) == models.VISION_CAP >= PARAMS[name]


def _convs(net):
    return [m for m in net.modules() if isinstance(m, nn.Conv2d)]


def test_mobilenet_v2_layers():
    net, inputs, calibration = models.model("mobilenet_v2_96", 0)
    convs = _convs(net)
    depthwise = [c for c in convs if c.groups > 1]
    # 17 inverted residual blocks, each with one 3x3 depthwise convolution;
    # 16 expand (all but the first), 17 project, the stem and the 1x1 head.
    assert len(depthwise) == 17 and all(c.groups == c.in_channels == c.out_channels
                                        and c.kernel_size == (3, 3) and c.padding == (1, 1)
                                        for c in depthwise)
    assert len(convs) == 17 + 16 + 17 + 2
    assert sum(c.stride == (2, 2) for c in convs) == 5          # 96 / 32 = 3 at the head
    assert sum(getattr(b, "residual", False) for b in net.modules()) == 10
    assert all(c.bias is None for c in convs)  # bias-free, a BatchNorm after each
    assert sum(isinstance(m, nn.ReLU6) for m in net.modules()) == 2 * 17 - 1 + 2
    assert [x.shape for x in inputs] == [torch.Size([1, 3, 96, 96])]
    assert len(calibration) == models.CALIBRATION_BATCHES
    assert float(inputs[0].min()) >= -1 and float(inputs[0].max()) <= 1
    with torch.no_grad():
        out = net(*inputs)
    assert out.shape == (1, 1000) and torch.isfinite(out).all()
    # Seeded BatchNorms: no identity left for conv+bn folding to hide behind.
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == len(convs) and all(float(b.running_var.min()) >= 0.5 for b in bns)
    assert all(b.eps == 1e-5 for b in bns)


def test_mobilenet_v3_small_layers():
    net, inputs, _ = models.model("mobilenet_v3_small", 0)
    convs = _convs(net)
    depthwise = [c for c in convs if c.groups > 1]
    assert len(depthwise) == 11
    assert [c.kernel_size[0] for c in depthwise] == [3, 3, 3] + [5] * 8
    assert all(c.padding[0] == c.kernel_size[0] // 2 for c in depthwise)
    # Squeeze-excite in 9 of the 11 blocks, gated by hardsigmoid.
    assert sum(isinstance(m, nn.Hardsigmoid) for m in net.modules()) == 9
    assert sum(getattr(b, "residual", False) for b in net.modules()) == 6
    assert sum(isinstance(m, nn.BatchNorm2d) and m.eps == 1e-3 for m in net.modules()) == \
        sum(isinstance(m, nn.BatchNorm2d) for m in net.modules()) > 0
    assert [x.shape for x in inputs] == [torch.Size([1, 3, 224, 224])]
    with torch.no_grad():
        out = net(*inputs)
    assert out.shape == (1, 1000) and torch.isfinite(out).all()


def test_census_cells():
    """mobilenet_v2_96 runs the benchmark's recipe in the census; the
    224-pixel models are the benchmark's only (cut first)."""
    cells = [(r, m) for r, m in pairs() if m.startswith("mobilenet_v")
             and m != "mobilenet_v1_025"]
    assert cells == [("arm-a8w8", "mobilenet_v2_96")]
    assert not covered("arm-a8w8", "mobilenet_v2_224")
    assert models.input_shape("mobilenet_v2_224") == (1, 3, 224, 224)
    with pytest.raises(KeyError):
        models.input_shape("lstm")
