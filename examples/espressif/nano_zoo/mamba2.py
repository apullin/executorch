# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoMamba2: Mamba-2's chunked state-space mixer at ESP32-S3 size.

Released (state-spaces/mamba2-2.8b, about 2.8B parameters): the Mamba-2
2.8B scale is width 2560 and 64 layers, with 2x expansion, 64-channel SSM
heads (80 heads at that width), 128 state values, one shared B/C group,
a four-tap convolution, vocabulary 50,280 and tied embeddings.
SSD, the chunked state-space calculation, replaces Mamba-1's channel-wise
selective scan; it still has no ordinary attention or separate MLP.
transformers 5.19.0 configuration_mamba2.py names that checkpoint but its
defaults are width 4096, 128 heads, 8 B/C groups, 32,768 tokens and untied
embeddings; those are not the original state-spaces 2.8B checkpoint's shape.
This fixture explicitly retains the original model's group count and tying.

DEVICE: 27,001,056 parameters, ~13.92 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width 2560 -> 384 and SSM heads 80 -> 12: keep 64-channel heads and 2x
  expansion while shrinking group-128 weight rows.
- Layers 64 -> 24: retain recurrent depth within the flash budget.
- Vocabulary 50,280 -> 8,192: shrink the tied input/output table.
- Chunk size 256 -> 4: bound scratch space and exercise two SSD chunks in
  the eight-token prompt, including state transfer across their boundary.
- Kept: state size 128, SSM head size 64, expansion 2, four-tap biased conv,
  SiLU, gated mixer RMSNorm, fp32 residuals and bias-free projections;
  one shared B/C group and tied embeddings.
  CI uses one 64-wide layer, eight 16-channel heads and 256 tokens.

Other ways to fit it:
- Width 512 and 12 layers (~12.81 MB): 16 SSM heads with less recurrent depth.
- Width 256 and 32 layers (~8.76 MB): eight SSM heads and a deeper stack.
- Width 384 with 4096 tokens: save about 0.81 MB at int4, with a smaller
  deployment tokenizer and shorter weight traffic for the output head.
"""

MODEL_TYPE = "mamba2"
MODEL_CLASS = "Mamba2ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, num_hidden_layers=24,
    num_heads=12, head_dim=64, state_size=128, expand=2, n_groups=1,
    conv_kernel=4, chunk_size=4, hidden_act="silu", layer_norm_epsilon=1e-5,
    use_bias=False, use_conv_bias=True, residual_in_fp32=True,
    tie_word_embeddings=True, pad_token_id=1, bos_token_id=0, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_hidden_layers=1,
          num_heads=8, head_dim=16)
