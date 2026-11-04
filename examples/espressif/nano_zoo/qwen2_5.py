# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoQwen2.5: Qwen2.5-0.5B's layout at ESP32-S3 size, random weights.

Released (Qwen/Qwen2.5-0.5B, 0.49B parameters, 0.36B of them outside the
embedding): 24 layers at width 896; 14 query heads sharing 2 key/value heads
(7:1), head size 64; SwiGLU MLP of 4864 (5.4x width); a bias on the q, k and v
projections (not on the output projection or the MLP), the family's mark;
RMSNorm eps 1e-6; RoPE theta 1,000,000; vocabulary 151,936; input embedding
tied to the output head; the sliding-window option is off.

DEVICE: 26.0M parameters, ~13.4 MB at int4.
- Width 896 -> 448 with the heads halved, 14/2 -> 7/1. Qwen2's head size is
  width / heads (the config has no head_dim), so 448 is the narrowest width
  that keeps both the 64-wide heads and the 7:1 sharing. It is not a multiple
  of 128: each row ends in a half group of 64.
- MLP 4864 -> 2432, still 5.4x the width.
- Layers 24 -> 6: at 3.7M parameters a layer (seven eighths of it MLP), six
  is what the region holds.
- Vocabulary 151,936 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: head size 64, the 7:1 head sharing, the q/k/v biases, the
  5.4x SwiGLU MLP, tied embeddings, RoPE theta 1M, RMSNorm eps 1e-6.

Other ways to fit it:
- Width 256, 4 query / 1 key-value heads (4:1), MLP 1408, 18 layers
  (12.6 MB): three quarters of the depth, but 4:1 sharing instead of 7:1.
- Width 448 with a 4x MLP (1792), 8 layers (13.7 MB): two more layers for a
  narrower MLP than the family's.
- Keep the 151,936 vocabulary: its table alone is 68M parameters at width
  448, four times the budget.
"""

MODEL_TYPE = "qwen2"
MODEL_CLASS = "Qwen2ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=448, intermediate_size=2432, num_hidden_layers=6,
    num_attention_heads=7, num_key_value_heads=1,
    max_position_embeddings=32768, rms_norm_eps=1e-6, tie_word_embeddings=True,
    use_sliding_window=False,
    rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
)

# Width 112, not 64: the 7 heads must divide it (head size 16).
CI = dict(DEVICE, vocab_size=256, hidden_size=112, intermediate_size=608,
          num_hidden_layers=2)
