# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoOLMo2: OLMo 2 1B's layout at ESP32-S3 size, random weights.

Released (allenai/OLMo-2-0425-1B, 1.5B parameters, 0.41B of them in the two
embedding tables): 16 layers at width 2048; 16 heads, each with its own
key/value head (no sharing), head size 128; SwiGLU MLP of 8192 (4x width);
the RMSNorms sit after attention and after the MLP, on each branch before it
joins the residual, not before them (the family's mark); the q and k
projections are each RMSNorm'd across all heads at once (QK-norm 2048 wide,
not per head as in Qwen3); RMSNorm eps 1e-6; RoPE theta 500,000; vocabulary
100,352; separate input embedding and output head; 4K context.

DEVICE: 21.0M parameters, ~10.8 MB at int4.
- Width 2048 -> 256, which leaves 2 heads, each with its own key/value head.
- MLP 8192 -> 1024, still 4x the width.
- Vocabulary 100,352 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: all 16 layers, the norms after attention and MLP, QK-norm
  across the whole projection, head size 128 without sharing, the 4x SwiGLU
  MLP, untied embeddings, RoPE theta 500k, RMSNorm eps 1e-6.

Other ways to fit it:
- Width 256 with 20 layers (13.0 MB): four layers beyond the release, to
  fill the region.
- Width 384, 3 heads, MLP 1536, 8 layers (13.0 MB): half the depth for half
  as much again of width.
- Keep the 100,352 vocabulary: its two tables are 51.4M parameters at width
  256, three times the budget.
"""

MODEL_TYPE = "olmo2"
MODEL_CLASS = "Olmo2ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=16,
    num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 500_000.0},
    # The class's eos id (50279) lies outside the 8,192 vocabulary, and its
    # pad id 1 would zero the embedding row of a token the example prompt uses.
    pad_token_id=None, eos_token_id=None,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256,
          num_hidden_layers=2)
