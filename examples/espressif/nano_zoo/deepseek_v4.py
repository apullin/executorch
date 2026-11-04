# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoDeepSeekV4: shared-KV compressed attention and MoE, random weights.

Released: deepseek-ai/DeepSeek-V4-Flash-Base. Shapes below are from
transformers 5.19.0 configuration_deepseek_v4.py, whose default attention
schedule is explicitly labelled V4-Pro; the Flash checkpoint's exact schedule
is not supplied there. Width 4096, 43 layers, 64 query heads and one shared
key/value head; head size 512 with 64 rotary coordinates; query bottleneck
1024; grouped output projection with 8 groups and rank 1024 per group. Each
layer has 256 routed SwiGLU experts of width 2048 (0.5x model width), top-6,
and one shared expert. First three layers use token-id hash routing, then
learned routing. The source schedule starts with two HCA layers, then repeats
HCA/CSA. CSA compresses 4 tokens, HCA 128; both keep a local window of 128.
The CSA indexer has 64 heads of size 128 and selects up to 512 entries.
Four residual streams use manifold-constrained hyper-connections (mHC) with
20 Sinkhorn normalization iterations. Vocabulary 129,280, untied embeddings,
RMSNorm, sqrt-softplus routing, clipped SwiGLU and learned attention sinks.

DEVICE: 24,748,381 parameters, ~12.76 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 4096 -> 128; query/output bottlenecks 1024 -> 32 scale with width.
- Layers 43 -> 4 retain the bootstrap, one HCA/CSA period and both routers.
- Experts 256 -> 128 keep the 0.5x expert width (64), top-6 and shared expert
  within flash while retaining the released attention head size and sharing.
- Vocabulary 129,280 -> 8,192 cuts token-table storage; special ids stay in range.
- Compression 4/128 -> 2/4, local window 128 -> 4 and index top-k 512 -> 2
  exercise both compressed branches and sparse selection in the 8-token run.
- Kept: 64:1 query/KV sharing with K=V, 512-wide heads, rotary fraction 1/8,
  eight output groups, 64x128 indexer, four mHC streams/20 Sinkhorn steps,
  untied embeddings, norms, gates, clipping and both MoE routing methods.
On the board: CI and DEVICE host parity pass, but DEVICE emits 72.57 MB.
The routed 3-D expert tables and grouped output projections remain fp32;
the DEVICE-only EXPECT records that flash-storage gap.

Other ways to fit it (parameter-based int4 estimates):
- Keep width 128 and four layers but use 96 experts: 21.59M parameters,
  11.13 MB; less expert capacity, with the full attention and routing layout.
- Keep 256 experts with query/output ranks 4 and vocabulary 2048: 28.13M,
  14.50 MB; the attention bottlenecks and tokenizer become much smaller. This
  estimate removes the embedding/head rows from the measured meta count.
- Keep vocabulary 8192 in that rank-4 model: 29.70M, 15.31 MB before the
  program, already over the model region; ranks alone do not make it fit.
"""

import os as _os

MODEL_TYPE = "deepseek_v4"
MODEL_CLASS = "DeepseekV4ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=128, num_hidden_layers=4,
    num_attention_heads=64, num_key_value_heads=1, head_dim=512,
    partial_rotary_factor=0.125, q_lora_rank=32, o_groups=8, o_lora_rank=32,
    moe_intermediate_size=64, n_routed_experts=128, num_experts_per_tok=6,
    n_shared_experts=1, scoring_func="sqrtsoftplus", norm_topk_prob=True,
    routed_scaling_factor=1.5, swiglu_limit=10.0,
    layer_types=["heavily_compressed_attention", "heavily_compressed_attention",
                 "heavily_compressed_attention", "compressed_sparse_attention"],
    mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe"],
    compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
    sliding_window=4, index_n_heads=64, index_head_dim=128, index_topk=2,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    max_position_embeddings=1048576, rope_theta=10000.0, compress_rope_theta=160000.0,
    hidden_act="silu", rms_norm_eps=1e-6, tie_word_embeddings=False,
    bos_token_id=0, eos_token_id=1, num_nextn_predict_layers=1,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64,
          head_dim=32, q_lora_rank=16, o_lora_rank=16, moe_intermediate_size=32,
          n_routed_experts=4, num_experts_per_tok=2,
          index_head_dim=16, hc_sinkhorn_iters=3)

# CI fits flash; preserve its ordinary parity gate while DEVICE is too large.
EXPECT = ("flash: routed 3-D expert tables and eight-group output BMM stay fp32 "
          "(72.57 MB emitted package)"
          if _os.environ.get("NANO_ZOO_SIZE", "device") == "device" else None)
