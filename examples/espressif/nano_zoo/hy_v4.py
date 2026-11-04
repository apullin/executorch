# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoHYV4: MLA, a shared sparse indexer and iHC, random weights.

Released: tencent/Hy4-preview, using transformers 5.19.0 HYV4Config defaults:
width 2816, 34 layers; 32 query and 32 expanded KV heads, 192 non-rotary +
64 rotary query/key coordinates and value size 256. MLA bottlenecks are
1536 for queries and 512 for KV. Dense MLP width 6912 (27/11x), expert
width 768 (3/11x), 256 routed experts, top-8 and one shared expert. Layer 0
is dense; later layers are MoE. Indexer layers 0 and 1 are full, followed
by three shared layers; the full/shared/shared/shared cycle then repeats.
The indexer has 16 heads of size 128 and top-k 2048. Four independent
hyper-connection (iHC) streams, learned attention sinks and attention output
gates; clamped SwiGLU, RMSNorm, vocabulary 120,832, untied embeddings.

DEVICE: 28,461,725 parameters, ~14.68 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 2816 -> 128, with 2 query and 2 expanded KV heads (still 1:1).
- Layers 34 -> 8 keep the initial full indexer, a full four-layer reuse
  period and the beginning of the next period.
- Dense/expert widths 6912/768 -> 314/35 and query/KV ranks 1536/512 -> 70/23
  round the released ratios to the nearest integer at width 128.
- Indexer heads 16 -> 2 and top-k 2048 -> 4 select half of the 8-token prompt.
- Vocabulary 120,832 -> 8,192 cuts table storage; pad/bos/eos
  120002/120000/120025 -> 0/1/2 stay in range.
- Kept: 256 routed experts, top-8, shared expert, first dense layer, indexer
  reuse period, 192+64 query/key and 256 value sizes, index head size 128,
  four iHC streams, attention sinks/gates, norms, clipping, untied embeddings.
On the board: both sizes pass host parity, but DEVICE emits 99.35 MB.
The two selected expert projections per sparse layer retain fp32 tables;
the DEVICE-only EXPECT records this flash-storage gap.

Other ways to fit it (meta-counted int4 weight estimates):
- Width 256, five layers and 64 experts: 23.03M parameters, 11.87 MB;
  wider projections, one full indexer-reuse period and a quarter of the pool.
  Dense/expert widths 628/70 and query/KV ranks 140/47 scale with width.
- Width 128, five layers and 384 experts: 24.26M, 12.51 MB; trade the later
  reuse layers for more expert capacity than the reference's 256-expert pool.
"""

import os as _os

MODEL_TYPE = "hy_v4"
MODEL_CLASS = "HYV4ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=128, intermediate_size=314, moe_intermediate_size=35,
    num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2,
    q_lora_rank=70, kv_lora_rank=23, qk_nope_head_dim=192, qk_rope_head_dim=64,
    v_head_dim=256, n_routed_experts=256, num_experts_per_tok=8, n_shared_experts=1,
    mlp_layer_types=["dense"] + ["sparse"] * 7,
    layer_types=["indexed_attention"] * 8,
    indexer_types=["full", "full", "shared", "shared", "shared",
                   "full", "shared", "shared"],
    index_topk=4, index_n_heads=2, index_head_dim=128,
    hc_mult=4, hc_magnitude=2.0, hc_eps=1e-6, learnable_sink_init=0.0,
    swiglu_limit=10.0, n_group=1, topk_group=1, norm_topk_prob=True,
    routed_scaling_factor=2.827, max_position_embeddings=262144,
    hidden_act="silu", rms_norm_eps=1e-5, initializer_range=0.006,
    tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=157,
          moe_intermediate_size=17, num_hidden_layers=5,
          num_attention_heads=8, num_key_value_heads=8, q_lora_rank=35,
          kv_lora_rank=12, qk_nope_head_dim=24, qk_rope_head_dim=8,
          v_head_dim=32, n_routed_experts=4, num_experts_per_tok=2, index_head_dim=16,
          mlp_layer_types=["dense"] + ["sparse"] * 4,
          layer_types=["indexed_attention"] * 5,
          indexer_types=["full", "full", "shared", "shared", "shared"])

# The toy tables fit; keep CI parity strict while DEVICE storage is unsupported.
EXPECT = ("flash: selected 3-D expert tables stay fp32 (99.35 MB emitted package)"
          if _os.environ.get("NANO_ZOO_SIZE", "device") == "device" else None)
