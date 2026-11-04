# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoQwen3: Qwen3-0.6B's layout at ESP32-S3 size, random weights.

Released (Qwen/Qwen3-0.6B, 0.6B parameters, 0.44B of them outside the
embedding): 28 layers at width 1024; 16 query heads sharing 8 key/value heads
(2:1), head size 128, so attention is 2048 wide, twice the model width; SwiGLU
MLP of 3072 (3x width); RMSNorm before attention and MLP, plus an RMSNorm over
each query and key head before RoPE (QK-norm, new in Qwen3; it replaces
Qwen2's q/k/v biases); RoPE theta 1,000,000; vocabulary 151,936; input
embedding tied to the output head; no sliding window.

DEVICE: 25.7M parameters, ~13.3 MB at int4.
- Width 1024 -> 256, and the heads with it, 16/8 -> 4/2: attention stays
  twice the width, so each layer keeps the released balance of attention to
  MLP weights (2:3).
- MLP 3072 -> 768, still 3x the width.
- Layers 28 -> 24: all 28 would be 29.6M parameters, 15.3 MB, just past the
  15.27 MB model region.
- Vocabulary 151,936 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: head size 128, the 2:1 head sharing, QK-norm, the 3x
  SwiGLU MLP, tied embeddings, RoPE theta 1M, RMSNorm eps 1e-6.

Other ways to fit it:
- Width 256 with 2 query / 1 key-value heads, all 28 layers (12.4 MB): the
  released depth, but attention only as wide as the model, so a layer is
  three quarters MLP instead of three fifths.
- Width 512, 8 query / 4 key-value heads, 6 layers (14.3 MB): the released
  head counts halved and every ratio kept, at under a quarter of the depth.
- Keep the 151,936 vocabulary: its table alone is 38.9M parameters at width
  256, over the budget before the first layer.
"""

MODEL_TYPE = "qwen3"
MODEL_CLASS = "Qwen3ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=768, num_hidden_layers=24,
    num_attention_heads=4, num_key_value_heads=2, head_dim=128,
    max_position_embeddings=40960, rms_norm_eps=1e-6, tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=192,
          num_hidden_layers=2, head_dim=32)
