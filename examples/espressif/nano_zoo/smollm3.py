# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoSmolLM3: SmolLM3-3B's layout at ESP32-S3 size, random weights.

Released (HuggingFaceTB/SmolLM3-3B, 3.1B parameters, 0.26B of them in the
embedding): 36 layers at width 2048; 16 query heads sharing 4 key/value heads
(4:1), head size 128; SwiGLU MLP of 11008 (5.4x width); every 4th layer skips
RoPE (NoPE: layers 4, 8, ... 36 attend with no position signal at all), the
family's mark; RMSNorm eps 1e-6; vocabulary 128,256 (Llama 3's tokenizer);
input embedding tied to the output head; 64K context. RoPE theta: the class
default in transformers is 2,000,000; whether the checkpoint uses that value
is not confirmed offline.

DEVICE: 24.1M parameters, ~12.4 MB at int4.
- Width 2048 -> 512 with the heads 16/4 -> 4/1. SmolLM3's head size is
  width / heads (the config has no head_dim), so 512 is the narrowest width
  that keeps both the 128-wide heads and the 4:1 sharing.
- MLP 11008 -> 2816, 5.5x the width (5.4x released, rounded to whole groups
  of 128).
- Layers 36 -> 4: one period, three RoPE layers then one NoPE layer. A layer
  is 5.0M parameters here; a second period would need 44M.
- Vocabulary 128,256 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: NoPE every 4th layer, head size 128 and 4:1 sharing, the
  wide SwiGLU MLP, tied embeddings, RMSNorm eps 1e-6, the class's RoPE theta.

Other ways to fit it:
- Head size 64 at width 256 (4 query / 1 key-value heads), MLP 1408, 20
  layers (13.9 MB): five NoPE periods, for half the released head size.
- Width 512 with a 2.75x MLP (1408), 8 layers (13.8 MB): two NoPE periods,
  for half the MLP ratio.
- Keep the 128,256 vocabulary: its table alone is 65.7M parameters at width
  512, four times the budget.
"""

MODEL_TYPE = "smollm3"
MODEL_CLASS = "SmolLM3ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=512, intermediate_size=2816, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=1, no_rope_layer_interval=4,
    max_position_embeddings=65536, rms_norm_eps=1e-6, tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 2_000_000.0},
    # The class's special-token ids (128000-128004) lie outside the 8,192
    # vocabulary; nn.Embedding refuses a padding index past its table.
    pad_token_id=None, bos_token_id=None, eos_token_id=None,
)

# One RoPE layer and one NoPE layer.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=352,
          num_hidden_layers=2, no_rope_layer_interval=2)
