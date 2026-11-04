# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoDeepSeek-V3: DeepSeek-V3's layout at ESP32-S3 size, random weights.

Released (deepseek-ai/DeepSeek-V3, 671B parameters, 37B used per token): 61
layers at width 7168. Multi-head latent attention (MLA): 128 heads whose
queries come through a 1,536-wide low-rank bottleneck and whose keys and
values are rebuilt from one shared 512-wide latent (what the cache stores),
plus a 64-wide rotary key shared by all heads; per head, 128 + 64 query/key
and 128 value. The first 3 layers have a dense SwiGLU MLP of 18,432 (2.57x
width); the other 58 a mixture of 256 routed experts of 2,048 (0.29x) plus 1
shared expert of the same size that every token uses. Routing: sigmoid
scores plus a learned per-expert bias choose the best 4 of 8 expert groups,
then the best 8 experts inside them; their weights are renormalized and
scaled by 2.5. YaRN RoPE (x40 over a 4K original window), on interleaved
pairs; vocabulary 129,280; untied embeddings; one extra multi-token-prediction
layer for speculative decoding. (transformers' docstring names a tiny test
checkpoint; the config defaults are DeepSeek-V3's.)

DEVICE: 21.0M parameters, ~10.8 MB at int4.
- Width 7168 -> 256 and heads 128 -> 2, each keeping the released 128 + 64
  query/key and 128 value widths.
- Query bottleneck 1,536 -> 128 and key/value latent 512 -> 128: one
  quantisation group each (scaled with the width they would be 55 and 18).
- Layers 61 -> 4: the released 3 dense layers, then 1 MoE layer. The 257
  experts cost 14.2M parameters at width 256, so one MoE layer fills the rest
  of the budget.
- Dense MLP 18,432 -> 640 (2.5x); experts 2,048 -> 72 wide (0.28x).
- Vocabulary 129,280 -> 8,192, as a deployment would retrain its tokenizer.
- The multi-token-prediction layer is not built: it is not part of the
  forward pass.
- Kept as released: 256 routed experts and 1 shared; the 8-group, best-4-
  groups, best-8-experts sigmoid routing with its bias, renormalization and
  2.5 scale; the 3 leading dense layers; MLA's head shape and its shared
  rotary key; YaRN on interleaved pairs; untied embeddings.

Other ways to fit it:
- 64 routed experts (8 groups of 8), 3 dense + 5 MoE layers (13.3 MB): depth
  instead of expert count.
- Width 128 (1 head), 36-wide experts, 3 dense + 6 MoE layers (12.9 MB): the
  routing as released over more layers, but MLA with a single head.
- Keep the 129,280 vocabulary: its two tables are 66.2M parameters at width
  256, over twice the budget.

On the board: the backend has no expert kernel. The 3-D expert weights (14.2M
of the 21.0M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 60.33 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "deepseek_v3"
MODEL_CLASS = "DeepseekV3ForCausalLM"
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 60.33 MB > 15.27 MB"}

YARN = {"rope_type": "yarn", "rope_theta": 10_000.0, "factor": 40.0,
        "original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0,
        "mscale": 1.0, "mscale_all_dim": 1.0}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, num_hidden_layers=4, first_k_dense_replace=3,
    intermediate_size=640, moe_intermediate_size=72,
    num_attention_heads=2, num_key_value_heads=2,
    q_lora_rank=128, kv_lora_rank=128, qk_nope_head_dim=128, qk_rope_head_dim=64,
    v_head_dim=128, rope_interleave=True,
    n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8,
    n_group=8, topk_group=4, routed_scaling_factor=2.5, norm_topk_prob=True,
    num_mtp_layers=0, max_position_embeddings=163840, rms_norm_eps=1e-6,
    tie_word_embeddings=False, rope_parameters=YARN,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_hidden_layers=2, first_k_dense_replace=1,
          intermediate_size=160, moe_intermediate_size=16, q_lora_rank=32, kv_lora_rank=32,
          qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)
