# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoLlama3.2: Llama 3.2 1B's layout at ESP32-S3 size, random weights.

Released (meta-llama/Llama-3.2-1B, 1.24B parameters): 16 layers at width 2048;
32 query heads sharing 8 key/value heads (4:1), head size 64; SwiGLU MLP of
8192 (4x width); RMSNorm; RoPE with llama3 frequency scaling (theta 500k, x32
over an 8K original window); vocabulary 128,256; input embedding tied to the
output head.

DEVICE: 24.9M parameters, ~12.8 MB at int4.
- Width 2048 -> 256, which leaves 4 query heads and 1 key/value head.
- Layers 16 -> 24. At width 256 the released 16 layers fill only 8.9 MB, and
  at this size depth buys more than width (MobileLLM, 2024), so the rest of
  the budget goes to layers.
- Vocabulary 128,256 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: head size 64, the 4:1 head sharing, the 4x MLP, tied
  embeddings, llama3 RoPE.

Other ways to fit it:
- Width 512, 8 query / 2 key-value heads, 6 layers (13.9 MB): the released
  head counts' ratio at width 512, but a third of the depth.
- Width 384, 6 query / 2 key-value heads (3:1), 10 layers (12.8 MB).
- Keep the 128K vocabulary: its embedding alone is 32.8M parameters at width
  256, over the budget before the first layer.

Llama 3.1 (8B) is the same block with head size 128, an MLP of 3.5x width and
an untied output head: llama3_1.py.
"""

MODEL_TYPE = "llama"
MODEL_CLASS = "LlamaForCausalLM"

ROPE = {"rope_type": "llama3", "rope_theta": 500_000.0, "factor": 32.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=24,
    num_attention_heads=4, num_key_value_heads=1, head_dim=64,
    max_position_embeddings=131072, rms_norm_eps=1e-5, tie_word_embeddings=True,
    rope_parameters=ROPE,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256,
          num_hidden_layers=2, head_dim=16)
