# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoYoutu: the dense MLA text model at ESP32-S3 size, random weights.

Released: tencent/Youtu-LLM-2B, using transformers 5.19.0 YoutuConfig defaults:
width 2048, 32 layers, 16 query and 16 expanded KV heads; query/key size
128 non-rotary + 64 rotary, value size 128. Query/KV latent ranks 1536/512,
dense SwiGLU MLP 6144 (3x width), interleaved RoPE, RMSNorm, vocabulary
128,256 and tied input/output embeddings. No routed experts.

DEVICE: 24,055,552 parameters, ~12.40 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 2048 -> 256, with 2 query and 2 expanded KV heads; MLA expands KV
  to every query head, so grouping KV heads would change the architecture.
- Layers 32 -> 26 fill flash after reducing width.
- Dense MLP 6144 -> 768 and query/KV ranks 1536/512 -> 192/64 scale with width.
- Vocabulary 128,256 -> 8,192 cuts table storage; bos/eos 128000/128001 -> 1/2
  stay in range.
- Kept: 1:1 query/KV sharing, 128+64 query/key and 128 value head sizes,
  the 3x MLP and 3/4 + 1/4 latent rank ratios, interleaved RoPE, RMSNorm,
  SwiGLU and tied embeddings.

Other ways to fit it (meta-counted int4 weight estimates):
- Width 512, four query/four KV heads and six layers: 24.26M parameters,
  12.51 MB; much wider projections with substantially less depth. MLP 1536,
  query/KV ranks 384/128 keep the released ratios.
- Width 384, three query/three KV heads and twelve layers: 25.79M, 13.30 MB;
  a middle ground between width and depth. MLP 1152, query/KV ranks 288/96.
"""

MODEL_TYPE = "youtu"
MODEL_CLASS = "YoutuForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=768, num_hidden_layers=26,
    num_attention_heads=2, num_key_value_heads=2, q_lora_rank=192, kv_lora_rank=64,
    qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
    max_position_embeddings=131072, hidden_act="silu", rms_norm_eps=1e-6,
    rope_interleave=True, tie_word_embeddings=True, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=192,
          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
          q_lora_rank=48, kv_lora_rank=16, qk_nope_head_dim=16,
          qk_rope_head_dim=8, v_head_dim=16)
