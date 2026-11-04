# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Known gaps: EXPECT[(recipe, model)] = "<stage>: <one-line cause>".

test_quant_census.py runs each listed pair as xfail(strict=True): the day a pair
starts passing, its test fails and asks for the entry to be removed. Written from
the status and first error of each pair's JSON record. Causes marked "recipe,
not backend" fail inside the stock quantizer before the backend sees the model;
"upstream bug" names the upstream file and line. Pure data: importing this
module imports nothing.
"""

ARM_PASS_HF = (
    "quantize: recipe, not backend (upstream bug) -- Arm's transform_for_annotation "
    "decomposes HF GPT-2's Conv1D addmm into mul.Scalar(bias, beta) on the bias "
    "parameter (backends/arm/_passes/decompose_addmm_pass.py:69), and "
    "ReplaceScalarWithTensorByProfilePass calls .to_tensor() on that operand "
    "(backends/transforms/replace_scalar_with_tensor.py:69), a raw Parameter "
    "because ExportPass hands get_attr values through unwrapped "
    "(exir/pass_base.py:590-593): 'Parameter' object has no attribute 'to_tensor'"
)

EXPECT = {}
EXPECT.update(dict.fromkeys([
    ('arm-a8w8', 'gpt2'),
    ('arm-a16w8', 'gpt2'),
    ('arm-a8w8-stage-a', 'gpt2'),
], ARM_PASS_HF))
