# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The census models: tiny, random weights, seeded, nothing downloaded.

Five kinds:

- block: one activation, or one op an activation zero point touches, with a
  quantised op on each side, so under every static recipe it sits between
  quantised tensors. Linear blocks take [1, 8, 128] through Linear(128, 256)
  -> op -> Linear(256, 128): widths that are multiples of 128, so torchao's
  group-128 recipes quantise them as well as the group-32 ones. Conv blocks take
  [1, 8, 12, 12] images; the padded convolutions read a ReLU output, which is
  one-sided, so an affine recipe gives that activation a zero point of -128 and
  the padding must be the zero point, not the integer 0.
- mlperf: MLPerf Tiny's four models, in tree at
  executorch.examples.models.mlperf_tiny, at their benchmark input sizes. Their
  BatchNorms get seeded statistics and affine parameters: the defaults make
  every BatchNorm an identity, and conv+bn folding would then test nothing.
- rnn: one recurrent layer, nn.LSTM or nn.GRU, over [1, 8, 32] (8 steps of 32
  features), its last step's hidden state through a Linear head: a keyword
  spotter's shape. Export unrolls the layer on the layer's own weights (not
  nn.Linear modules): one matmul projects every step's input, then each step
  is a matmul on the shared recurrent weight, the gate activations and their
  products.
- transformer: the nano model zoo's llama3_2 and gpt2 at CI size (the zoo is
  examples/espressif/nano_zoo), and llama3_2 at device size for the RAM column and
  the stock LLM recipes
  (eight recipes only); and
  ExecuTorch's own llama (examples/models/llama) at CI size with its KV cache,
  one static decode step, for ExecuTorch's LLM recipes.
- vision: MobileNetV2 (at 96 and 224 pixels) and MobileNetV3-Small (at 224),
  1000 classes, written out here layer for layer as torchvision builds them
  (its widths, strides, paddings, activations and weight initialisation); their
  BatchNorms are seeded as the MLPerf models' are. They are the board
  benchmark's models (examples/espressif/esp32s3/scripts/run_bench_hardware.py)
  at their published sizes, 3.5M and 2.5M parameters, past SMALL_CAP: each
  runs one census recipe, the benchmark's (arm-a8w8), and the 224-pixel two
  none.

Each model draws its evaluation input and its calibration batches from
separately seeded generators over the same distribution: uniform [-1, 1] for
the MLPerf and vision models (the MLPerf models' get_example_inputs), standard
normal for the blocks and the recurrent models, uniform token ids for the
transformers.

Importing this module does not import torch.
"""

from dataclasses import dataclass

# Calibration batches for every static recipe.
CALIBRATION_BATCHES = 8
# Tokens per block input and per transformer prompt (the nano zoo's SEQ).
SEQ = 8
# The recurrent models: features per step, hidden width, classes.
RNN_IN, RNN_HIDDEN, RNN_CLASSES = 32, 64, 12
# Everything but the device-size llama and the vision models stays under this;
# it is counted on the meta device before a real byte is allocated.
SMALL_CAP = 1_000_000
# The vision models at their published sizes: MobileNetV2 3,504,872 parameters,
# MobileNetV3-Small 2,542,856.
VISION_CAP = 4_000_000


@dataclass(frozen=True)
class ModelSpec:
    kind: str           # "block", "mlperf", "rnn", "transformer" or "vision"
    linear: bool        # its work is nn.Linear (torchao's recipes apply)
    doc: str
    recipes: tuple = None   # the only recipes it runs (None: all)
    kv_cache: bool = False  # ExecuTorch's llama KVCache modules (et-kv-quant)


MODELS = {
    # Linear blocks: Linear(128, 256) -> op -> Linear(256, 128).
    "gelu_erf": ModelSpec("block", True, "GELU (erf) between two Linears"),
    "gelu_tanh": ModelSpec("block", True, "GELU (tanh approximation) between two Linears"),
    "sigmoid": ModelSpec("block", True, "sigmoid between two Linears"),
    "silu": ModelSpec("block", True, "SiLU between two Linears"),
    "tanh": ModelSpec("block", True, "tanh between two Linears"),
    "softmax": ModelSpec("block", True, "softmax over the last axis between two Linears"),
    "layernorm": ModelSpec("block", True, "LayerNorm between two Linears"),
    "linear_relu": ModelSpec("block", True, "Linear -> ReLU -> Linear"),
    "residual_add": ModelSpec("block", True, "Linear(x) + x, then a Linear"),
    "cat": ModelSpec("block", True, "two Linears concatenated, then a Linear"),
    # Conv blocks on [1, 8, 12, 12].
    "se_hardswish": ModelSpec("block", False, "MobileNetV3 block: conv3x3 -> hardswish -> "
                              "squeeze-excite (1x1 convs, ReLU, hardsigmoid, mul) -> conv1x1"),
    "avgpool": ModelSpec("block", False, "conv3x3 -> AvgPool2d(2) -> conv1x1"),
    "maxpool": ModelSpec("block", False, "conv3x3 -> MaxPool2d(2) -> conv1x1"),
    "conv_pad": ModelSpec("block", False, "conv1x1 -> ReLU -> conv3x3 padding 1 -> conv1x1"),
    "dwconv_pad": ModelSpec("block", False, "conv1x1 -> ReLU -> depthwise conv3x3 padding 1 "
                            "-> conv1x1"),
    # MLPerf Tiny.
    "ds_cnn": ModelSpec("mlperf", False, "keyword spotting DS-CNN, [1, 1, 49, 10]"),
    "mobilenet_v1_025": ModelSpec("mlperf", False, "visual wake words MobileNetV1 x0.25, "
                                  "[1, 3, 96, 96]"),
    "resnet8": ModelSpec("mlperf", False, "CIFAR-10 ResNet-8, [1, 3, 32, 32]"),
    "deep_autoencoder": ModelSpec("mlperf", True, "anomaly detection autoencoder, "
                                  "Linear+BatchNorm1d+ReLU, [1, 640]"),
    # Recurrent, on [1, 8, 32].
    "lstm": ModelSpec("rnn", False, "nn.LSTM(32, 64) over 8 steps, the last step's "
                      "hidden state through Linear(64, 12)"),
    "gru": ModelSpec("rnn", False, "nn.GRU(32, 64) over 8 steps, the last step's "
                     "hidden state through Linear(64, 12)"),
    # Transformers (nano zoo families).
    "llama3_2": ModelSpec("transformer", True, "nano zoo llama3_2 at CI size: 2 layers, "
                          "width 64, vocab 256"),
    "gpt2": ModelSpec("transformer", True, "nano zoo gpt2 at CI size: 1 layer, width 64, "
                      "vocab 256; projections are HF Conv1D (addmm), not nn.Linear"),
    # The RAM column's recipes, and the stock LLM recipes at device size:
    # 8da4w at group 32 and 128, 8da8w, 8da4w with asymmetric weights
    # (8da4w-asym) and weight-only asymmetric int4 (4w-asym).
    "llama3_2_device": ModelSpec("transformer", True, "nano zoo llama3_2 at device size: "
                                 "24 layers, width 256, vocab 8192 (24.9M parameters)",
                                 recipes=("none", "xnnpack-static", "arm-a8w8",
                                          "torchao-8da4w-g32", "torchao-8da4w-g128",
                                          "torchao-8da8w", "torchao-8da4w-asym",
                                          "torchao-4w-asym")),
    "et_llama": ModelSpec("transformer", True, "ExecuTorch's own llama at CI size: "
                          "2 layers, width 64, 4 query and 2 KV heads, vocab 256, "
                          "a 16-token KV cache, one static decode step",
                          recipes=("none", "torchao-8da4w-g32", "torchao-8da4w-asym",
                                   "et-kv-quant", "et-emb-8da4w"), kv_cache=True),
    # Vision, the board benchmark's models: the benchmark's recipe only.
    "mobilenet_v2_96": ModelSpec("vision", False, "MobileNetV2 x1.0 as torchvision builds it, "
                                 "1000 classes, [1, 3, 96, 96]", recipes=("arm-a8w8",)),
    "mobilenet_v2_224": ModelSpec("vision", False, "the same at [1, 3, 224, 224]", recipes=()),
    "mobilenet_v3_small": ModelSpec("vision", False, "MobileNetV3-Small x1.0 as torchvision "
                                    "builds it, 1000 classes, [1, 3, 224, 224]", recipes=()),
}

# ExecuTorch's llama behind et_llama (export_llama's static-shape decode).
ET_LLAMA = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=256,
                hidden_dim=128, max_seq_len=16, max_context_len=16,
                use_kv_cache=True, enable_dynamic_shape=False, max_batch_size=1)


def _et_llama():
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs

    return construct_transformer(ModelArgs(**ET_LLAMA))

# The nano zoo family and size behind each transformer.
_ZOO = {"llama3_2": ("llama3_2", "ci"), "gpt2": ("gpt2", "ci"),
        "llama3_2_device": ("llama3_2", "device")}


def zoo_available():
    """Whether the nano zoo behind the _ZOO models can build them: the zoo
    package (examples/espressif/nano_zoo) and the transformers it builds
    them with. Without transformers the zoo still imports (it reads its
    families with the standard library), but every _ZOO model would fail as
    an environment error instead of being skipped."""
    import importlib.util

    try:
        return (importlib.util.find_spec("executorch.examples.espressif.nano_zoo") is not None
                and importlib.util.find_spec("transformers") is not None)
    except ImportError:
        return False


def _blocks():
    """name -> constructor of the bare (unseeded) module. Imports torch."""
    import torch
    import torch.nn as nn

    class Between(nn.Module):
        """Linear(128, 256) -> op -> Linear(256, 128)."""

        def __init__(self, op, width=128, hidden=256):
            super().__init__()
            self.fc1 = nn.Linear(width, hidden)
            self.op = op
            self.fc2 = nn.Linear(hidden, width)

        def forward(self, x):
            return self.fc2(self.op(self.fc1(x)))

    class ResidualAdd(nn.Module):
        def __init__(self, width=128):
            super().__init__()
            self.fc1 = nn.Linear(width, width)
            self.fc2 = nn.Linear(width, width)

        def forward(self, x):
            return self.fc2(self.fc1(x) + x)

    class Cat(nn.Module):
        def __init__(self, width=128):
            super().__init__()
            self.a = nn.Linear(width, width)
            self.b = nn.Linear(width, width)
            self.fc = nn.Linear(2 * width, width)

        def forward(self, x):
            return self.fc(torch.cat((self.a(x), self.b(x)), dim=-1))

    class SqueezeExcite(nn.Module):
        """MobileNetV3's: average, squeeze through 1x1 convs, gate with
        hardsigmoid, scale the input."""

        def __init__(self, channels, squeezed):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, squeezed, 1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Conv2d(squeezed, channels, 1)
            self.gate = nn.Hardsigmoid()

        def forward(self, x):
            s = self.gate(self.fc2(self.relu(self.fc1(self.pool(x)))))
            return x * s

    class SeHardswish(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 16, 3, padding=1)
            self.act = nn.Hardswish()
            self.se = SqueezeExcite(16, 8)
            self.project = nn.Conv2d(16, 16, 1)

        def forward(self, x):
            return self.project(self.se(self.act(self.conv(x))))

    class ConvOpConv(nn.Module):
        def __init__(self, first, op, last):
            super().__init__()
            self.first, self.op, self.last = first, op, last

        def forward(self, x):
            return self.last(self.op(self.first(x)))

    class PaddedAfterRelu(nn.Module):
        """conv1x1 -> ReLU -> padded 3x3 (dense or depthwise) -> conv1x1."""

        def __init__(self, groups):
            super().__init__()
            self.expand = nn.Conv2d(8, 16, 1)
            self.relu = nn.ReLU()
            self.padded = nn.Conv2d(16, 16, 3, padding=1, groups=groups)
            self.project = nn.Conv2d(16, 16, 1)

        def forward(self, x):
            return self.project(self.padded(self.relu(self.expand(x))))

    return {
        "gelu_erf": lambda: Between(nn.GELU()),
        "gelu_tanh": lambda: Between(nn.GELU(approximate="tanh")),
        "sigmoid": lambda: Between(nn.Sigmoid()),
        "silu": lambda: Between(nn.SiLU()),
        "tanh": lambda: Between(nn.Tanh()),
        "softmax": lambda: Between(nn.Softmax(dim=-1)),
        "layernorm": lambda: Between(nn.LayerNorm(256)),
        "linear_relu": lambda: Between(nn.ReLU()),
        "residual_add": ResidualAdd,
        "cat": Cat,
        "se_hardswish": SeHardswish,
        "avgpool": lambda: ConvOpConv(nn.Conv2d(8, 16, 3, padding=1), nn.AvgPool2d(2),
                                      nn.Conv2d(16, 16, 1)),
        "maxpool": lambda: ConvOpConv(nn.Conv2d(8, 16, 3, padding=1), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 16, 1)),
        "conv_pad": lambda: PaddedAfterRelu(groups=1),
        "dwconv_pad": lambda: PaddedAfterRelu(groups=16),
    }


def _mlperf():
    """name -> (constructor, input shape). Imports executorch."""
    from executorch.examples.models import mlperf_tiny as mt

    return {
        "ds_cnn": (mt.DSCNNKWS, (1, 1, 49, 10)),
        "mobilenet_v1_025": (mt.MobileNetV1025, (1, 3, 96, 96)),
        "resnet8": (mt.ResNet8, (1, 3, 32, 32)),
        "deep_autoencoder": (mt.DeepAutoEncoder, (1, 640)),
    }


def _recurrent():
    """name -> constructor of the bare (unseeded) recurrent module. Imports
    torch."""
    import torch.nn as nn

    class LastStep(nn.Module):
        """One recurrent layer over the sequence, the hidden state of its last
        step through a Linear head. nn.LSTM and nn.GRU return (output, state);
        the head reads output's last step, so the result depends on every step
        of the recurrence."""

        def __init__(self, layer):
            super().__init__()
            self.rnn = layer(RNN_IN, RNN_HIDDEN, batch_first=True)
            self.head = nn.Linear(RNN_HIDDEN, RNN_CLASSES)

        def forward(self, x):
            y, _ = self.rnn(x)
            return self.head(y[:, -1, :])

    return {"lstm": lambda: LastStep(nn.LSTM), "gru": lambda: LastStep(nn.GRU)}


def _divisible(v, divisor=8):
    """torchvision's _make_divisible: `v` to a multiple of `divisor`, never
    more than 10% below it."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


def _vision():
    """name -> (constructor of the bare (unseeded) module, input shape).
    Imports torch."""
    import torch
    import torch.nn as nn

    def conv_bn(cin, cout, kernel=3, stride=1, groups=1, act=nn.ReLU6, eps=1e-5):
        """torchvision's Conv2dNormActivation: a bias-free convolution padded
        (kernel - 1) // 2 on every side, BatchNorm, then the activation."""
        layers = [nn.Conv2d(cin, cout, kernel, stride, (kernel - 1) // 2, groups=groups,
                            bias=False), nn.BatchNorm2d(cout, eps=eps)]
        return nn.Sequential(*layers, *([act()] if act else []))

    def init(net):
        """torchvision's initialisation of both MobileNets."""
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return net

    class InvertedResidual(nn.Module):
        """MobileNetV2's block: 1x1 expansion (none at ratio 1), 3x3
        depthwise, linear 1x1 projection; the input added back where the
        shape allows."""

        def __init__(self, cin, cout, stride, ratio):
            super().__init__()
            hidden = int(round(cin * ratio))
            self.residual = stride == 1 and cin == cout
            self.conv = nn.Sequential(
                *([conv_bn(cin, hidden, 1)] if ratio != 1 else []),
                conv_bn(hidden, hidden, 3, stride, groups=hidden),
                conv_bn(hidden, cout, 1, act=None))

        def forward(self, x):
            return x + self.conv(x) if self.residual else self.conv(x)

    class MobileNetV2(nn.Module):
        # (expansion t, channels c, repeats n, first stride s)
        SETTING = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
                   (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))

        def __init__(self, num_classes=1000):
            super().__init__()
            cin = 32
            features = [conv_bn(3, cin, 3, 2)]
            for t, c, n, s in self.SETTING:
                for i in range(n):
                    features.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                    cin = c
            features.append(conv_bn(cin, 1280, 1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
            init(self)

        def forward(self, x):
            x = nn.functional.adaptive_avg_pool2d(self.features(x), (1, 1))
            return self.classifier(torch.flatten(x, 1))

    class SqueezeExcitation(nn.Module):
        """torchvision's, as MobileNetV3 configures it: average, squeeze
        through 1x1 convolutions with a ReLU, gate with hardsigmoid, scale."""

        def __init__(self, channels, squeezed):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, squeezed, 1)
            self.fc2 = nn.Conv2d(squeezed, channels, 1)
            self.activation = nn.ReLU()
            self.scale_activation = nn.Hardsigmoid()

        def forward(self, x):
            s = self.fc2(self.activation(self.fc1(self.avgpool(x))))
            return self.scale_activation(s) * x

    class Bneck(nn.Module):
        """MobileNetV3's block, eps 0.001 BatchNorms throughout."""

        def __init__(self, cin, kernel, expanded, cout, se, hardswish, stride):
            super().__init__()
            act = nn.Hardswish if hardswish else nn.ReLU
            self.residual = stride == 1 and cin == cout
            self.block = nn.Sequential(
                *([conv_bn(cin, expanded, 1, act=act, eps=1e-3)] if expanded != cin else []),
                conv_bn(expanded, expanded, kernel, stride, groups=expanded, act=act, eps=1e-3),
                *([SqueezeExcitation(expanded, _divisible(expanded // 4))] if se else []),
                conv_bn(expanded, cout, 1, act=None, eps=1e-3))

        def forward(self, x):
            y = self.block(x)
            return y + x if self.residual else y

    class MobileNetV3Small(nn.Module):
        # (in, kernel, expanded, out, squeeze-excite, hardswish, stride)
        SETTING = ((16, 3, 16, 16, True, False, 2), (16, 3, 72, 24, False, False, 2),
                   (24, 3, 88, 24, False, False, 1), (24, 5, 96, 40, True, True, 2),
                   (40, 5, 240, 40, True, True, 1), (40, 5, 240, 40, True, True, 1),
                   (40, 5, 120, 48, True, True, 1), (48, 5, 144, 48, True, True, 1),
                   (48, 5, 288, 96, True, True, 2), (96, 5, 576, 96, True, True, 1),
                   (96, 5, 576, 96, True, True, 1))

        def __init__(self, num_classes=1000):
            super().__init__()
            self.features = nn.Sequential(
                conv_bn(3, 16, 3, 2, act=nn.Hardswish, eps=1e-3),
                *(Bneck(*row) for row in self.SETTING),
                conv_bn(96, 576, 1, act=nn.Hardswish, eps=1e-3))
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Linear(576, 1024), nn.Hardswish(),
                                            nn.Dropout(0.2), nn.Linear(1024, num_classes))
            init(self)

        def forward(self, x):
            return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))

    return {
        "mobilenet_v2_96": (MobileNetV2, (1, 3, 96, 96)),
        "mobilenet_v2_224": (MobileNetV2, (1, 3, 224, 224)),
        "mobilenet_v3_small": (MobileNetV3Small, (1, 3, 224, 224)),
    }


def _constructor(name):
    spec = MODELS[name]
    if spec.kind == "block":
        return _blocks()[name]
    if spec.kind == "mlperf":
        return _mlperf()[name][0]
    if spec.kind == "rnn":
        return _recurrent()[name]
    if spec.kind == "vision":
        return _vision()[name][0]
    raise KeyError(name)


def meta_params(name):
    """Parameter count of `name`, built on the meta device (nothing allocated)."""
    import torch

    if name == "et_llama":
        with torch.device("meta"):
            return sum(p.numel() for p in _et_llama().parameters())
    if MODELS[name].kind == "transformer":
        from executorch.examples.espressif.nano_zoo import build

        return build.meta_params(*_ZOO[name])[0]
    with torch.device("meta"):
        net = _constructor(name)()
    return sum(p.numel() for p in net.parameters())


def cap(name):
    if name == "llama3_2_device":
        from executorch.examples.espressif.nano_zoo import PARAM_CAP

        return PARAM_CAP
    if MODELS[name].kind == "vision":
        return VISION_CAP
    return SMALL_CAP


def input_shape(name):
    """The evaluation input's shape of an MLPerf or vision model."""
    kind = MODELS[name].kind
    table = _mlperf() if kind == "mlperf" else _vision() if kind == "vision" else {}
    if name not in table:
        raise KeyError(f"{name} is a {kind} model: its input is not an image or a spectrogram")
    return table[name][1]


def _generator(seed, stream):
    import torch

    return torch.Generator().manual_seed(1_000_003 * (seed + 1) + stream)


def _seed_batchnorms(net, generator):
    """Seeded running statistics and affine parameters for every BatchNorm."""
    import torch

    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                c = m.num_features
                m.running_mean.copy_(0.1 * torch.randn(c, generator=generator))
                m.running_var.copy_(0.5 + 1.5 * torch.rand(c, generator=generator))
                m.weight.copy_(0.5 + torch.rand(c, generator=generator))
                m.bias.copy_(0.1 * torch.randn(c, generator=generator))


def model(name, seed=0):
    """(eval-mode net, evaluation inputs, calibration batches) for `name`,
    after refusing it if its meta-device parameter count passes its cap."""
    net, draw = _net_and_draw(name, seed)
    inputs = draw(_generator(seed, 0))
    calibration = [draw(_generator(seed, 1 + i)) for i in range(CALIBRATION_BATCHES)]
    return net, inputs, calibration


def held_out(name, count, seed=0):
    """`count` more inputs for `name`'s seed-`seed` net, from the evaluation
    input's distribution on streams neither it nor the calibration batches
    use (no observer has seen them)."""
    _, draw = _net_and_draw(name, seed)
    return [draw(_generator(seed, 1000 + i)) for i in range(count)]


def _net_and_draw(name, seed):
    import torch

    spec = MODELS[name]
    params = meta_params(name)
    if params > cap(name):
        raise ValueError(f"{name}: {params:,} parameters > cap {cap(name):,}")
    if name == "et_llama":
        torch.manual_seed(seed)
        net = _et_llama().eval().requires_grad_(False)
        vocab = ET_LLAMA["vocab_size"]

        def draw(g):
            # export_llama's decode inputs: one token and its position.
            return (torch.randint(0, vocab, (1, 1), generator=g),
                    {"input_pos": torch.tensor([0], dtype=torch.long)})
    elif spec.kind == "transformer":
        from executorch.examples.espressif.nano_zoo import build

        family, size = _ZOO[name]
        net, _ = build.model(family, size, seed)
        vocab = net.net.config.get_text_config().vocab_size

        def draw(g):
            return (torch.randint(0, vocab, (1, SEQ), generator=g),)
    else:
        torch.manual_seed(seed)
        net = _constructor(name)()
        _seed_batchnorms(net, _generator(seed, 99))
        net = net.eval().requires_grad_(False)
        if spec.kind in ("mlperf", "vision"):
            shape = input_shape(name)

            def draw(g):
                return (torch.rand(shape, generator=g) * 2 - 1,)
        else:
            shape = ((1, SEQ, RNN_IN) if spec.kind == "rnn" else
                     (1, SEQ, 128) if spec.linear else (1, 8, 12, 12))

            def draw(g):
                return (torch.randn(shape, generator=g),)
    return net, draw
