# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGranite-3.0: Granite 3.0 2B's layout at ESP32-S3 size, random weights.

Released (ibm-granite/granite-3.0-2b-base, 2.5B parameters): 40 layers at
width 2048; 32 query heads sharing 8 key/value heads (4:1), head size 64;
SwiGLU MLP of 8192 (4x width); four scalar multipliers from its muP-style
training, the family's mark: the embedding times 12, each attention and MLP
output times 0.22 before it joins the residual, attention scores times 1/64
(instead of 1/sqrt(64)), and the logits divided by 8; RMSNorm eps 1e-6; RoPE
theta 10,000; vocabulary 49,155; input embedding tied to the output head; 4K
context. Granite 3.1 2B is the same network with a longer context (a larger
RoPE theta).

DEVICE: 24.9M parameters, ~12.8 MB at int4.
- Width 2048 -> 256, which leaves 4 query heads and 1 key/value head.
- MLP 8192 -> 1024, still 4x the width.
- Layers 40 -> 24: all 40 would be 20.7 MB.
- Vocabulary 49,155 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: the four multipliers (12, 0.22, 1/64, 8; muP would retune
  them for a width this small, but these are random weights), head size 64
  and 4:1 sharing, the 4x SwiGLU MLP, tied embeddings, RoPE theta 10k,
  RMSNorm eps 1e-6.

Other ways to fit it:
- Width 512, 8 query / 2 key-value heads, 6 layers (13.9 MB): the released
  head ratio at a quarter of the width, under a sixth of the depth.
- Keep the 49,155 vocabulary: its table is 12.6M parameters at width 256,
  which leaves 13 layers (12.9 MB).
"""

MODEL_TYPE = "granite"
MODEL_CLASS = "GraniteForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=24,
    num_attention_heads=4, num_key_value_heads=1,
    embedding_multiplier=12.0, residual_multiplier=0.22,
    attention_multiplier=0.015625, logits_scaling=8.0,
    max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

# attention_multiplier stays 1/64 although the CI heads are 16 wide: it is
# the released constant, not 1/head size.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256,
          num_hidden_layers=2)
