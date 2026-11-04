# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantizer validation census: stock quantization recipes through this backend.

ExecuTorch users already quantize with somebody's recipe: XNNPACK's PT2E
quantizer, Arm's Ethos-U and Cortex-M quantizers, torchao's 8da4w / 4w (the
recipe ExecuTorch's own LLM export ships). This package applies each of them
exactly as its owner ships it to small random-weight models, lowers the result
through the ESP32-S3 backend, runs it on the host runner, and compares it with
the recipe's own eager reference: the PT2E converted GraphModule or the
torchao-quantized module, the semantics a user validates against.

Two controls run beside them: `esp32` (the backend's own Esp32Quantizer) and
`none` (the float model, quantized by the backend itself, data-free).

The backend runs a model's own dynamic activation quantiser
(strip_dynamic_act_qdq absorbs each dynamic chain and records its scheme, or
refuses it by name) and its static int8 pairs (absorbed by the delegates that
read or write them, static_qdq.py, or run on ExecuTorch's own
quantize/dequantize kernels between portable ops). The census records what
fails as strict expected failures (expect.py): a pair that starts passing
fails test_quant_census.py until its entry is removed.

    python -m executorch.backends.esp32s3.test.quant_census.run <recipe> <model>
    python -m pytest backends/esp32s3/test/quant_census/test_quant_census.py

The first runs one pair and writes its record into results_dir(); the second
runs every pair, one process each, its records in a fresh temporary directory.

Importing this package does not import torch.
"""

import os
import tempfile
from pathlib import Path

from executorch.backends.esp32s3.test.quant_census.models import _ZOO, MODELS, zoo_available
from executorch.backends.esp32s3.test.quant_census.recipes import RECIPES

ROOT = Path(__file__).resolve().parent.parent
# Provisional pass bar: cosine of the host runner's output against the recipe's
# eager reference. Every record keeps the raw numbers, so a stricter bar (bit
# exact, or within one rounding step of the output's quantization) can be
# applied to them.
MIN_COSINE = 0.999


def applicable(recipe, model):
    """None when the pair belongs in the census, else why it is n/a."""
    r, m = RECIPES[recipe], MODELS[model]
    if model in _ZOO and not zoo_available():
        return "needs the nano zoo and transformers"
    if r.kind == "torchao" and not m.linear:
        work = ("a recurrent layer's gates, on the layer's own weights"
                if m.kind == "rnn" else "convolutions")
        return f"torchao's LLM recipes quantize nn.Linear; this model's work is {work}"
    if recipe == "et-kv-quant" and not m.kv_cache:
        return ("export_llama's KV-cache quantisation replaces ExecuTorch's "
                "llama KVCache modules; this model has none")
    if r.kind == "executorch" and m.kind != "transformer":
        return ("export_llama's LLM recipes quantise an LLM's embedding and "
                "Linears; this model has no embedding")
    return None


def covered(recipe, model):
    """Whether the census has a cell for the pair at all. The device-size
    model runs only the recipes that measure its RAM column."""
    restrict = MODELS[model].recipes
    return restrict is None or recipe in restrict


def pairs():
    """Every (recipe, model) cell, in census order, n/a cells included."""
    return [(r, m) for m in MODELS for r in RECIPES if covered(r, m)]


def results_dir():
    """Where quant_census.run writes its records: $QUANT_CENSUS_RESULTS, else
    espressif_quant_census in the system's temporary directory, never the
    source tree."""
    return Path(os.environ.get("QUANT_CENSUS_RESULTS")
                or os.path.join(tempfile.gettempdir(), "espressif_quant_census"))


def result_path(recipe, model):
    return results_dir() / f"{recipe}.{model}.json"
