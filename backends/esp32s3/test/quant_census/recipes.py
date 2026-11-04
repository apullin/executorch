# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The census recipes: stock quantisation flows, called as their owners ship them.

quantize(name, net, inputs, calibration) takes a float eval-mode model, its
evaluation inputs and its calibration batches, and returns a Quantized:

  program    the module to torch.export and lower through the backend
  reference  the same quantised model run eagerly in PyTorch (the PT2E
             converted GraphModule, or the torchao-quantised module): the
             semantics a user validates against
  inputs     the evaluation inputs in the layout the recipe feeds (Cortex-M's
             flow feeds images channels-last)
  info       what the recipe did: its quantise/dequantise ops and activation
             zero points, or how many Linears torchao replaced

Choices the owners leave open, made once here:

- PT2E graphs come from `torch.export.export(m, x).module(check_guards=False)`.
  torch 2.15 otherwise inserts a `_guards_fn` submodule, which CortexMQuantizer's
  annotation passes refuse ("call_module is not supported"). ExecuTorch's own
  tests never see it: torch skips the guard module for callers under executorch/
  and torchao/ paths (torch/export/_unlift.py, _ok_to_generate_guards_fn).
- QAT (`xnnpack-qat`) follows prepare_qat_pt2e's own docstring example: export
  the eval-mode model, prepare_qat_pt2e (its conv+bn QAT fusion matches the
  export's aten.batch_norm whatever its training flag), then
  move_exported_model_to_eval before the calibration batches: no training, and
  the BatchNorm statistics stay the model's own. (A train-mode export cannot
  take the batch-1 inputs: BatchNorm1d refuses one value per channel.)
- Arm: EthosUQuantizer(EthosUCompileSpec("ethos-u55-128")), the Corstone-300
  pairing MLPerf Tiny targets (TOSA-1.0+INT+int16+int4+u55, so int16 is legal).
  Arm's packages import tosa_serializer, which the environment may lack;
  quant_census/_stubs/tosa_serializer stands in for its names only,
  and anything that needs real TOSA work raises StubUsed (recorded as skipped).
- torchao 8da4w / 8da8w copy examples/models/llama/source_transformation/
  quantize.py: Int8DynamicActivationIntxWeightConfig with
  intx_choose_qparams_algorithm="hqq_scale_only" (export_llama's default
  use_hqq=True), on nn.Linear layers whose in_features divide by the group
  size; 8da8w at quantize.py's default group size, 128.
- torchao 4w / 4w-asym use IntxWeightOnlyConfig's own default algorithm
  (affine), not quantize.py's "4w" hqq_scale_only: HQQ scale-only always yields
  zero points of 0, so the asymmetric variant would carry none. Like
  quantize.py's 4w they take every nn.Linear (no filter), so a model with a
  Linear whose in_features 32 does not divide, which torchao asserts on, is n/a
  for them by that named precondition (deep_autoencoder's 8-wide bottleneck).
- torchao 8da4w-asym is the 8da4w recipe with asymmetric
  weights: 8da4w-g32 with weight_mapping_type=MappingType.ASYMMETRIC and the
  config's own default algorithm (affine; quantize.py's use_hqq=False), not
  hqq_scale_only, for the 4w-asym reason. quantize.py's 8da4w mode has no
  asymmetric option, so the config is torchao's, with quantize.py's filter;
  the activations are the stock per-token affine int8 of every 8da recipe.
- ExecuTorch's LLM recipes (export_llama) are its own source transforms, in
  _get_source_transforms' order and ending with its float32 cast:
  et-kv-quant is --quantize_kv_cache (QuantizedKVCache: XNNPACK's per-token
  affine chooser, written with llama.update_cache; and the quantised SDPA,
  which replaces only the custom-op SDPA this flow does not use);
  et-emb-8da4w is --embedding_quantize 4,32 with -qmode 8da4w -G 32
  (QuantizedGroupEmbedding's embedding_4bit, then 8da4w on the Linears).
- esp32-* are this backend's Esp32Quantizer presets: the stock config's
  scheme on the ops the backend computes in integers
  (espressif_quantizer.PRESETS; test_espressif_quantizer.py compares them with
  the stock quantizers). esp32-cortex-m feeds images contiguous, the delegate's
  layout: Cortex-M's channels-last checks are CMSIS-NN's, not its scheme's.

Importing this module does not import torch.
"""

import copy
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RecipeSpec:
    kind: str       # "control", "pt2e" or "torchao"
    owner: str
    doc: str
    stub: bool = False   # imports Arm's packages (tosa_serializer)


RECIPES = {
    "esp32": RecipeSpec(
        "control", "this backend",
        "Esp32Quantizer() defaults: int4 per-channel symmetric weights, dynamic int8 "
        "per-tensor symmetric activations (weight-only when one activation feeds "
        "several matmuls); prepare_pt2e, 8 batches, convert_pt2e"),
    "none": RecipeSpec(
        "control", "this backend",
        "the float model: the backend quantises weights itself (i8 per row) and "
        "activations per row at run time; the reference is the float model"),
    "xnnpack-static": RecipeSpec(
        "pt2e", "XNNPACK",
        "XNNPACKQuantizer().set_global(get_symmetric_quantization_config("
        "is_per_channel=True)); prepare_pt2e, 8 batches, convert_pt2e"),
    "xnnpack-static-pt": RecipeSpec(
        "pt2e", "XNNPACK",
        "the same with is_per_channel=False (per-tensor weights)"),
    "xnnpack-dynamic": RecipeSpec(
        "pt2e", "XNNPACK",
        "get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)"),
    "xnnpack-qat": RecipeSpec(
        "pt2e", "XNNPACK",
        "get_symmetric_quantization_config(is_per_channel=True, is_qat=True); "
        "prepare_qat_pt2e, move_exported_model_to_eval, 8 batches (no training), "
        "convert_pt2e"),
    "arm-a8w8": RecipeSpec(
        "pt2e", "Arm",
        "EthosUQuantizer(EthosUCompileSpec('ethos-u55-128')).set_global("
        "get_symmetric_quantization_config())", stub=True),
    "arm-a16w8": RecipeSpec(
        "pt2e", "Arm",
        "the same with get_symmetric_a16w8_quantization_config() (int16 symmetric "
        "activations, int8 weights)", stub=True),
    "cortex-m": RecipeSpec(
        "pt2e", "Arm",
        "CortexMQuantizer() defaults; image inputs channels-last, as its example and "
        "conv pattern checks require", stub=True),
    "torchao-8da4w-g32": RecipeSpec(
        "torchao", "torchao",
        "quantize_(Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4, "
        "weight_granularity=PerGroup(32), intx_choose_qparams_algorithm="
        "'hqq_scale_only'), filter_fn=quantize.py's)"),
    "torchao-8da4w-g128": RecipeSpec(
        "torchao", "torchao", "the same with PerGroup(128)"),
    "torchao-8da8w": RecipeSpec(
        "torchao", "torchao", "the same with weight_dtype=torch.int8, PerGroup(128)"),
    "torchao-8da4w-asym": RecipeSpec(
        "torchao", "torchao",
        "8da4w-g32 with weight_mapping_type=MappingType.ASYMMETRIC (weight zero "
        "points) and the config's default intx_choose_qparams_algorithm "
        "('affine'): HQQ scale-only gives zero points of 0"),
    "torchao-4w": RecipeSpec(
        "torchao", "torchao",
        "quantize_(IntxWeightOnlyConfig(weight_dtype=torch.int4, "
        "granularity=PerGroup(32))): symmetric weights, float activations"),
    "torchao-4w-asym": RecipeSpec(
        "torchao", "torchao",
        "the same with mapping_type=MappingType.ASYMMETRIC (weight zero points)"),
    "et-kv-quant": RecipeSpec(
        "executorch", "ExecuTorch",
        "export_llama --quantize_kv_cache: replace_kv_cache_with_quantized_kv_cache "
        "and replace_sdpa_with_quantized_sdpa, then the model cast to float32"),
    "et-emb-8da4w": RecipeSpec(
        "executorch", "ExecuTorch",
        "export_llama --embedding_quantize 4,32 -qmode 8da4w -G 32: "
        "get_quant_embedding_transform('4,32') then get_quant_weight_transform("
        "'8da4w', group_size=32), then the model cast to float32"),
    # Stage A, the explicit-off twins: Stage B (stage_b.py) is on by default,
    # so xnnpack-static and arm-a8w8 keep their codes as int8 between claimed
    # ops and across delegates wherever the ops between can run on codes;
    # these lower the same quantisation with Esp32Partitioner(stage_b=False),
    # fp32 between delegates.
    "xnnpack-static-stage-a": RecipeSpec(
        "pt2e", "XNNPACK",
        "xnnpack-static's quantisation, lowered with Esp32Partitioner("
        "stage_b=False): Stage A, fp32 between claimed ops and delegates"),
    "arm-a8w8-stage-a": RecipeSpec(
        "pt2e", "Arm",
        "arm-a8w8's quantisation, lowered with Esp32Partitioner(stage_b=False)",
        stub=True),
    # Esp32Quantizer's presets: the stock schemes, this backend's coverage.
    "esp32-xnnpack-static": RecipeSpec(
        "pt2e", "this backend",
        "Esp32Quantizer(preset='xnnpack_static'): XNNPACK's static per-channel "
        "scheme (int8 [-127, 127] weights, affine int8 activations, eps 2^-12) on "
        "Linear/conv inputs, weights and outputs; prepare_pt2e, 8 batches"),
    "esp32-xnnpack-dynamic": RecipeSpec(
        "pt2e", "this backend",
        "Esp32Quantizer(preset='xnnpack_dynamic'): XNNPACK's dynamic scheme "
        "(per-tensor affine, eps 2^-12)"),
    "esp32-arm-a8w8": RecipeSpec(
        "pt2e", "this backend",
        "Esp32Quantizer(preset='arm_a8w8'): Arm's a8w8 scheme (eps 2^-16, int32 "
        "per-channel biases)"),
    "esp32-arm-a8w4": RecipeSpec(
        "pt2e", "this backend",
        "Esp32Quantizer(preset='arm_a8w4'): Arm's a8w4 scheme (int4 [-7, 7] "
        "weights), PTQ"),
    "esp32-arm-a16w8": RecipeSpec(
        "pt2e", "this backend",
        "Esp32Quantizer(preset='arm_a16w8'): Arm's a16w8 scheme (static int16 "
        "activations, eps 2^-12; int8 per-channel weights, eps 2^-16)"),
    "esp32-cortex-m": RecipeSpec(
        "pt2e", "this backend",
        "Esp32Quantizer(preset='cortex_m'): Cortex-M's scheme (per-tensor Linear, "
        "per-channel conv, [-128, 127] weights, int32 biases); contiguous images"),
}
# The Stage A twins: the stock recipe each quantises with, lowered with
# Stage B off (lowering(name)).
STAGE_A_RECIPES = {"xnnpack-static-stage-a": "xnnpack-static",
                   "arm-a8w8-stage-a": "arm-a8w8"}
# The census name of each Esp32Quantizer preset.
ESP32_PRESETS = {"esp32-xnnpack-static": "xnnpack_static",
                 "esp32-xnnpack-dynamic": "xnnpack_dynamic",
                 "esp32-arm-a8w8": "arm_a8w8", "esp32-arm-a8w4": "arm_a8w4",
                 "esp32-arm-a16w8": "arm_a16w8", "esp32-cortex-m": "cortex_m"}

STUBS = Path(__file__).resolve().parent / "_stubs"


@dataclass
class Quantized:
    program: object
    reference: object
    inputs: tuple
    info: dict = field(default_factory=dict)


def tosa_serializer_source():
    """'real' if the environment has tosa_serializer, else 'stub' after putting
    quant_census/_stubs first on sys.path (this process only)."""
    import importlib.util

    if str(STUBS) not in sys.path and importlib.util.find_spec("tosa_serializer"):
        return "real"
    if str(STUBS) not in sys.path:
        sys.path.insert(0, str(STUBS))
    return "stub"


def is_stub_error(error):
    """Whether `error`, or an exception it was raised from, came from the
    tosa_serializer stand-in."""
    seen = set()
    while error is not None and id(error) not in seen:
        if type(error).__name__ == "StubUsed" or "tosa_serializer stub" in str(error):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def lowering(name):
    """The Esp32Partitioner options a recipe's pairs lower with beyond the
    census's own (quant_census.lower): stage_b=False for the Stage A twins."""
    return {"stage_b": False} if name in STAGE_A_RECIPES else {}


def quantize(name, net, inputs, calibration):
    # torch.histc's float32 overflow (histc_guard.py): the stock observers'
    # calibration must not write out of bounds on a masked-attention range.
    from executorch.backends.esp32s3.test import histc_guard

    histc_guard.install()
    if name in STAGE_A_RECIPES:
        # The stock recipe's own quantisation; only the lowering differs.
        quantized = quantize(STAGE_A_RECIPES[name], net, inputs, calibration)
        quantized.info["lowering"] = lowering(name)
        return quantized
    spec = RECIPES[name]
    if name == "none":
        return Quantized(net, net, inputs, {})
    if spec.kind == "torchao":
        return _torchao(name, net, inputs)
    if spec.kind == "executorch":
        return _executorch_llm(name, net, inputs)
    info = {}
    if spec.stub:
        info["tosa_serializer"] = tosa_serializer_source()
    qat = False
    if name == "esp32":
        from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

        quantizer = Esp32Quantizer()
    elif name in ESP32_PRESETS:
        from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

        quantizer = Esp32Quantizer(preset=ESP32_PRESETS[name])
    elif name.startswith("xnnpack"):
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        qat = name == "xnnpack-qat"
        config = get_symmetric_quantization_config(
            is_per_channel=name != "xnnpack-static-pt",
            is_dynamic=name == "xnnpack-dynamic", is_qat=qat)
        quantizer = XNNPACKQuantizer().set_global(config)
    elif name.startswith("arm"):
        from executorch.backends.arm.ethosu import EthosUCompileSpec
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer,
            get_symmetric_a16w8_quantization_config,
            get_symmetric_quantization_config,
        )

        config = (get_symmetric_a16w8_quantization_config() if name == "arm-a16w8"
                  else get_symmetric_quantization_config())
        quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u55-128")).set_global(config)
    elif name == "cortex-m":
        from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer

        quantizer = CortexMQuantizer()
        inputs = _channels_last(inputs)
        calibration = [_channels_last(batch) for batch in calibration]
        info["channels_last_inputs"] = any(x.dim() == 4 for x in inputs)
    else:
        raise KeyError(name)
    converted = _pt2e(net, inputs, calibration, quantizer, qat)
    info.update(pt2e_info(converted))
    if name == "esp32" or name in ESP32_PRESETS:
        info["act_annotated"] = quantizer.act_annotated
    return Quantized(converted, converted, inputs, info)


def _channels_last(batch):
    """The 4-D image tensors of a batch channels-last; anything else (a
    token tensor, ExecuTorch's llama's {"input_pos": ...}) as it is."""
    import torch

    return tuple(x.contiguous(memory_format=torch.channels_last)
                 if isinstance(x, torch.Tensor) and x.dim() == 4 else x
                 for x in batch)


def _pt2e(net, inputs, calibration, quantizer, qat):
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import (
        convert_pt2e,
        prepare_pt2e,
        prepare_qat_pt2e,
    )

    if qat:
        from torchao.quantization.pt2e import move_exported_model_to_eval

        gm = torch.export.export(net, inputs, strict=False).module(check_guards=False)
        prepared = prepare_qat_pt2e(gm, quantizer)
        move_exported_model_to_eval(prepared)
    else:
        gm = torch.export.export(net, inputs, strict=False).module(check_guards=False)
        prepared = prepare_pt2e(gm, quantizer)
    with torch.no_grad():
        for batch in calibration:
            prepared(*batch)
    return convert_pt2e(prepared)


# Pure data movement between a dequantise and the graph output.
_VIEWS = ("view", "reshape", "flatten", "squeeze", "unsqueeze", "permute", "clone",
          "contiguous", "_to_copy", "expand")


def pt2e_info(gm):
    """The converted graph's quantisation: op counts, the static activation
    quantisers' zero points and dtypes, and the output's (scale, zero point)
    when the output leaves through a static dequantise."""
    ops, zero_points, dtypes = Counter(), Counter(), Counter()
    batch_norms = 0
    for n in gm.graph.nodes:
        if n.op == "call_function" and "batch_norm" in str(n.target):
            batch_norms += 1
        if n.op != "call_function" or "quantized_decomposed" not in str(n.target):
            continue
        name = str(n.target).split("quantized_decomposed.")[-1]
        ops[name] += 1
        if name == "quantize_per_tensor.default":
            # (x, scale, zero_point, quant_min, quant_max, dtype)
            zero_points["nonzero" if n.args[2] != 0 else "zero"] += 1
            dtypes[str(n.args[5]).replace("torch.", "")] += 1
    # BatchNorms the recipe left unfolded (PTQ folds into the conv at prepare,
    # QAT at convert); the backend folds what remains before partitioning.
    info = {"qdq_ops": dict(ops), "static_act_zero_points": dict(zero_points),
            "static_act_dtypes": dict(dtypes), "batch_norms_left": batch_norms}
    out = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    node = out[0] if isinstance(out, (tuple, list)) else out
    while (getattr(node, "op", None) == "call_function"
           and any(v in str(node.target) for v in _VIEWS)):
        node = node.args[0]
    if (getattr(node, "op", None) == "call_function"
            and "dequantize_per_tensor.default" in str(node.target)
            and isinstance(node.args[1], float)):
        info["output_qparams"] = {"scale": node.args[1], "zero_point": int(node.args[2]),
                                  "quant_min": int(node.args[3]),
                                  "quant_max": int(node.args[4])}
    return info


def _linear_filter(group):
    """quantize.py's 8da4w filter_fn: an nn.Linear, not a LoRA adapter, whose
    in_features divide by the group size."""
    import torch.nn as nn

    def filter_fn(m, fqn):
        if not isinstance(m, nn.Linear):
            return False
        parts = fqn.split(".")
        if "lora_a" in parts or "lora_b" in parts:
            return False
        return group == 0 or m.weight.shape[1] % group == 0

    return filter_fn


def _torchao(name, net, inputs):
    import torch
    import torch.nn as nn
    from torchao.quantization import (
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
        MappingType,
        quantize_,
    )
    from torchao.quantization.granularity import PerGroup

    model = copy.deepcopy(net)
    linears = {fqn for fqn, m in model.named_modules() if isinstance(m, nn.Linear)}
    if name == "torchao-8da4w-asym":
        # The config's default algorithm (affine): hqq_scale_only would
        # leave every zero point 0.
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4, weight_granularity=PerGroup(32),
            weight_mapping_type=MappingType.ASYMMETRIC)
        quantize_(model, config, filter_fn=_linear_filter(32))
    elif name.startswith("torchao-8da"):
        group = 32 if name == "torchao-8da4w-g32" else 128
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8 if name == "torchao-8da8w" else torch.int4,
            weight_granularity=PerGroup(group),
            intx_choose_qparams_algorithm="hqq_scale_only")
        quantize_(model, config, filter_fn=_linear_filter(group))
    else:
        # No filter: quantize_ takes every nn.Linear, as ExecuTorch's
        # quantize.py "4w" mode calls it (quantize.py:177-190), and torchao
        # asserts on a Linear whose in_features the group does not divide
        # (quant_primitives.py:322): a model with one is not this recipe's.
        narrow = sorted(f"{fqn} (in_features {m.in_features})"
                        for fqn, m in model.named_modules()
                        if isinstance(m, nn.Linear) and m.in_features % 32)
        if narrow:
            return Quantized(model, model, inputs, {"recipe_limit": (
                f"IntxWeightOnlyConfig(PerGroup(32)) quantises every "
                f"nn.Linear (quantize_ with no filter, as ExecuTorch's "
                f"quantize.py 4w mode calls it, quantize.py:177-190), and "
                f"torchao refuses one whose in_features 32 does not divide "
                f"(quant_primitives.py:322): {', '.join(narrow)}")})
        config = IntxWeightOnlyConfig(
            weight_dtype=torch.int4, granularity=PerGroup(32),
            mapping_type=(MappingType.ASYMMETRIC if name == "torchao-4w-asym"
                          else MappingType.SYMMETRIC))
        quantize_(model, config)
    quantized = sorted(fqn for fqn, m in model.named_modules()
                       if fqn in linears and type(m.weight) is not nn.Parameter)
    weight_types = {type(m.weight).__name__ for fqn, m in model.named_modules()
                    if fqn in quantized}
    info = {"linears": len(linears), "quantized_linears": len(quantized),
            "left_float": sorted(linears - set(quantized))[:12],
            "weight_tensor": sorted(weight_types)}
    return Quantized(model, model, inputs, info)


def _executorch_llm(name, net, inputs):
    """export_llama's own source transforms, as _get_source_transforms lists
    them (embedding quantisation before the Linears' 8da4w; the KV cache's
    quantisation with the quantised SDPA), then its float32 cast."""
    import contextlib
    import io

    import torch
    import torch.nn as nn
    from executorch.examples.models.llama.source_transformation import (
        custom_kv_cache as ckv,
        quantize as qz,
        sdpa,
    )

    model = copy.deepcopy(net)
    linears = {fqn for fqn, m in model.named_modules() if isinstance(m, nn.Linear)}
    if name == "et-emb-8da4w":
        transforms = [qz.get_quant_embedding_transform("4,32"),
                      qz.get_quant_weight_transform("8da4w", group_size=32)]
    else:
        transforms = [ckv.replace_kv_cache_with_quantized_kv_cache,
                      sdpa.replace_sdpa_with_quantized_sdpa]
    transforms.append(lambda m: m.to(dtype=torch.float32))
    with contextlib.redirect_stdout(io.StringIO()):     # the handlers print
        for transform in transforms:
            model = transform(model)
    kinds = Counter(type(m).__name__ for m in model.modules())
    info = {"quantized_embeddings": kinds.get("QuantizedGroupEmbedding", 0),
            "quantized_kv_caches": kinds.get("QuantizedKVCache", 0)}
    if name == "et-emb-8da4w":
        quantized = sorted(fqn for fqn, m in model.named_modules()
                           if fqn in linears and type(m.weight) is not nn.Parameter)
        info.update(linears=len(linears), quantized_linears=len(quantized),
                    left_float=sorted(linears - set(quantized))[:12])
    return Quantized(model, model, inputs, info)
