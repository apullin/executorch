# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoLongCatFlash: two MLA sublayers and shortcut experts, random weights.

Released: meituan-longcat/LongCat-Flash-Chat, from transformers 5.19.0
LongcatFlashConfig: width 6144, 28 decoder blocks, each with two attention
sublayers and two dense MLPs (56 attention sublayers in total). 64 query
and 64 expanded KV heads; query/key size 128 non-rotary + 64 rotary, value
size 128, query/KV ranks 1536/512. Dense MLP 12288 (2x width); shortcut MoE
has 512 learned experts of width 2048 (1/3x) plus 256 identity experts, and
selects 12 experts per token. RMSNorm, SwiGLU, RoPE theta 10 million,
vocabulary 131,072 and untied embeddings.

DEVICE: 25,503,040 parameters, ~13.15 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 6144 -> 128, with 2 query and 2 expanded KV heads (still 1:1).
- Blocks 28 -> 2 (four MLA sublayers) keep the two-sublayer shortcut structure.
- Dense MLP 12288 -> 256 retains 2x; expert width 2048 -> 43 rounds 128/3
  to an integer. Query/KV ranks 1536/512 -> 32/16 shrink the latent projections.
- Vocabulary 131,072 -> 8,192 cuts token-table storage.
- The source allocates unused gate/up rows for the identity experts too;
  those rows are included in the count, so three blocks would exceed flash.
- Kept: 512 learned + 256 identity experts, top-12, both dense MLPs per block,
  128+64 query/key and 128 value sizes, shortcut routing, norms, activation,
  RoPE base and untied embeddings.
Export gap: LongcatFlashExperts iterates over expert_hit.nonzero().
Its data-dependent length cannot be unbound during torch.export, so both
sizes fail before the ESP32 partitioner. Identity experts remain in the fixture.

Other ways to fit it (meta-counted int4 weight estimates):
- Width 128, one block and expert width 86: 25.07M parameters, 12.93 MB;
  half the depth, twice the per-expert width, still 512 learned/256 identity experts.
- Width 256, one block, expert width 85, 192 learned/96 identity experts:
  22.28M, 11.49 MB; wider attention/dense MLPs, a smaller 2:1 expert pool.
  Four query/four KV heads, dense MLP 512, query/KV ranks 64/32.
"""

MODEL_TYPE = "longcat_flash"
MODEL_CLASS = "LongcatFlashForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=128, num_layers=2,
    num_attention_heads=2, num_key_value_heads=2, ffn_hidden_size=256,
    expert_ffn_hidden_size=43, n_routed_experts=512, zero_expert_num=256, moe_topk=12,
    q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=128,
    qk_rope_head_dim=64, head_dim=64, v_head_dim=128,
    max_position_embeddings=131072, hidden_act="silu", rms_norm_eps=1e-5,
    routed_scaling_factor=6.0, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_layers=1,
          num_attention_heads=4, num_key_value_heads=4, ffn_hidden_size=128,
          expert_ffn_hidden_size=21, n_routed_experts=4, zero_expert_num=2,
          moe_topk=2, q_lora_rank=32, kv_lora_rank=32,
          qk_nope_head_dim=16, qk_rope_head_dim=16, head_dim=16, v_head_dim=16)

EXPECT = "export: expert_hit.nonzero() routing loop has data-dependent length (Eq(u0, 0))"
