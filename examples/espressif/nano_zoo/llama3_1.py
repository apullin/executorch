# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoLlama3.1: Llama 3.1 8B's layout at ESP32-S3 size, random weights.

Released (meta-llama/Llama-3.1-8B, 8.03B parameters): 32 layers at width 4096;
32 query heads sharing 8 key/value heads (4:1), head size 128; SwiGLU MLP of
14,336 (3.5x width); RMSNorm; RoPE with llama3 frequency scaling (theta 500k,
x8 over an 8K original window); vocabulary 128,256; an output head separate
from the input embedding.

DEVICE: 25.4M parameters, ~13.1 MB at int4.
- Width 4096 -> 512, which leaves 4 query heads and 1 key/value head at the
  released head size of 128.
- Layers 32 -> 5: each layer is 3.4M parameters at this width, and the
  separate embedding and output head take 8.4M of the budget between them.
- Vocabulary 128,256 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: head size 128, the 4:1 head sharing, the 3.5x MLP, the
  untied output head, llama3 RoPE (x8).

Other ways to fit it:
- Width 256: 2 query heads and 1 key/value head (2:1) at head size 128,
  25 layers (13.6 MB). Close to the released depth, but half the head sharing.
- Tie the output head to the embedding, as Llama 3.2 does: 4.2M parameters
  freed, 7 layers (14.5 MB).
- Head size 64 (8 query / 2 key-value heads at width 512): the same parameter
  count with twice the attention heads; that is Llama 3.2's head.

Llama 3.2 (1B, 3B) is the same block with head size 64, a 4x MLP and tied
embeddings: llama3_2.py.
"""

MODEL_TYPE = "llama"
MODEL_CLASS = "LlamaForCausalLM"

ROPE = {"rope_type": "llama3", "rope_theta": 500_000.0, "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192}

DEVICE = dict(
    vocab_size=8192, hidden_size=512, intermediate_size=1792, num_hidden_layers=5,
    num_attention_heads=4, num_key_value_heads=1, head_dim=128,
    max_position_embeddings=131072, rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters=ROPE,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=128, intermediate_size=448,
          num_hidden_layers=2, head_dim=32)
