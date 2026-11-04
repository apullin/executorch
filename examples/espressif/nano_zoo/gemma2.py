# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGemma2: Gemma 2 2B's layout at ESP32-S3 size, random weights.

Released (google/gemma-2-2b, 2.6B parameters, 0.59B of them in the
embedding): 26 layers at width 2304; 8 query heads sharing 4 key/value heads
(2:1), head size 256, so attention is 2048 wide; GeGLU MLP (tanh GELU) of
9216 (4x width); layers alternate a 4096-token sliding window and full
attention, sliding first; attention scores soft-capped at 50 and the final
logits at 30 (cap * tanh(x / cap)), the family's mark; four RMSNorms per
layer, before and after attention and before and after the MLP, in Gemma's
(1 + weight) form; the embedding multiplied by sqrt(width); queries scaled by
1/sqrt(256) (query_pre_attn_scalar); RoPE theta 10,000 in every layer;
vocabulary 256,000; input embedding tied to the output head; 8K context.

DEVICE: 25.7M parameters, ~13.3 MB at int4.
- Width 2304 -> 256, and the heads 8/4 -> 2/1, the fewest that keep the 2:1
  sharing: attention 512 wide, twice the width (0.9x released).
- MLP 9216 -> 1024, still 4x the width.
- Layers 26 -> 20, ten sliding/full pairs: all 26 would be 16.9 MB.
- Vocabulary 256,000 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: head size 256 and 2:1 sharing, both softcaps (50 and
  30), the alternating pattern and its 4096 window, the four norms per layer,
  the scaled embedding, query_pre_attn_scalar 256, RoPE theta 10k, tanh GELU,
  tied embeddings.

Other ways to fit it:
- Width 512 with the same 2/1 heads (attention as wide as the model, close
  to the released 0.9x), MLP 2048, 6 layers (14.3 MB): the released balance
  of attention to MLP, at under a quarter of the depth.
- Width 256, 4 query / 2 key-value heads (attention 1024), 14 layers
  (12.4 MB): twice the heads, six fewer layers.
- Keep the 256,000 vocabulary: its table alone is 65.5M parameters at width
  256, four times the budget.
"""

MODEL_TYPE = "gemma2"
MODEL_CLASS = "Gemma2ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=20,
    num_attention_heads=2, num_key_value_heads=1, head_dim=256,
    hidden_activation="gelu_pytorch_tanh", query_pre_attn_scalar=256,
    sliding_window=4096, layer_types=["sliding_attention", "full_attention"] * 10,
    attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
    max_position_embeddings=8192, rms_norm_eps=1e-6, tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

# One pair. A window shorter than the 8-token prompt, so the sliding mask
# differs from the causal one and the comparison can see it.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256,
          num_hidden_layers=2, head_dim=64, query_pre_attn_scalar=64,
          sliding_window=4, layer_types=["sliding_attention", "full_attention"])
