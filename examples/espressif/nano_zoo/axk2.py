# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoAXK2: gated MLA, sparse indexing and MoE, random weights.

Released: skt/A.X-K2 is the reference named by transformers 5.19.0, but
AXK2Config explicitly uses the A.X-K2-Light defaults: width 2048, 48 layers,
32 query and 32 expanded KV heads; query/key size 64 non-rotary + 32 rotary,
value size 64, query/KV ranks 384/128. Dense MLP 5120 (2.5x width), expert
MLP 512 (0.25x), 128 routed experts, top-8, one shared expert; layer 0 dense,
the rest sparse. Every attention layer has a DSA indexer (16 heads of size
128, top-k 2048). Low-rank gated RMSNorm uses rank 16; attention also has
an output gate. Ungrouped expert routing, SwiGLU, vocabulary 163,840,
untied embeddings. Exact larger A.X-K2 checkpoint dimensions are not in this
configuration file, so this nano follows the documented Light variant.

DEVICE: 24,688,128 parameters, ~12.73 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 2048 -> 256, with 4 query and 4 expanded KV heads, retaining 1:1.
- Layers 48 -> 4 retain the dense prefix and three sparse layers.
- Dense/expert MLP 5120/512 -> 640/64 and query/KV ranks 384/128 -> 48/16
  preserve their ratios to model width.
- Indexer heads 16 -> 2 and top-k 2048 -> 4 exercise sparse attention on 8 tokens.
- Vocabulary 163,840 -> 8,192 cuts table storage; bos/eos 163691 -> 1/2 stay in range.
- Kept: 64+32 query/key and 64 value sizes, indexer head size 128,
  128 routed experts, top-8, one shared expert, ungrouped routing, norm gate
  rank 16, attention output gate, SwiGLU, RMSNorm and untied embeddings.
On the board: both sizes pass host parity, but DEVICE emits 78.80 MB.
The two selected expert projections in each of its three sparse layers
retain fp32 tables; the DEVICE-only EXPECT records this flash-storage gap.

Other ways to fit it (meta-counted int4 weight estimates):
- Width 128, twelve layers, two query/two KV heads and 128 experts: 20.68M
  parameters, 10.66 MB; narrower attention but more sparse MLP/indexer depth.
  Dense/expert widths 320/32, query/KV ranks 24/8 and one indexer head.
- Width 256, eight layers and 64 experts: 28.94M, 14.92 MB; twice the depth
  of DEVICE with half its expert pool, keeping the same attention shape.
"""

import os as _os

MODEL_TYPE = "axk2"
MODEL_CLASS = "AXK2ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=640, moe_intermediate_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
    q_lora_rank=48, kv_lora_rank=16, qk_nope_head_dim=64, qk_rope_head_dim=32,
    v_head_dim=64, n_routed_experts=128, num_experts_per_tok=8, n_shared_experts=1,
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    layer_types=["indexed_attention"] * 4, n_group=None, topk_group=None,
    index_topk=4, index_n_heads=2, index_head_dim=128, gated_norm_rank=16,
    norm_topk_prob=True, routed_scaling_factor=2.5,
    max_position_embeddings=131072, hidden_act="silu", rms_norm_eps=1e-6,
    tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=160,
          moe_intermediate_size=16, num_hidden_layers=2,
          q_lora_rank=12, kv_lora_rank=4, qk_nope_head_dim=16,
          qk_rope_head_dim=8, v_head_dim=16, index_head_dim=16,
          n_routed_experts=4, num_experts_per_tok=2,
          mlp_layer_types=["dense", "sparse"],
          layer_types=["indexed_attention"] * 2)

# The toy tables fit; retain CI parity while DEVICE expert storage is unsupported.
EXPECT = ("flash: selected 3-D expert tables stay fp32 (78.80 MB emitted package)"
          if _os.environ.get("NANO_ZOO_SIZE", "device") == "device" else None)
