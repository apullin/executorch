# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGLM5Next: linear KDA, pooled indexed MLA and mHC, random weights.

Released: zai-org/GLM-5.3-Flash, from transformers 5.19.0 Glm5NextTextConfig:
width 4096, 45 layers, three linear KDA layers then one indexed MLA layer
per period. MLA has 64 query and 64 expanded KV heads, query/key and value
size 256 with no rotary coordinates; query/KV latent ranks 1536/512.
Linear attention has 64 heads of size 128 and a kernel-4 short convolution.
The first three MLPs are dense, width 12288 (3x); later ones use 288 routed
experts of width 2048 (0.5x), top-8 and one shared expert. DSA indexer:
32 heads of size 128, top-k 2048, pooled groups of 16 tokens with tail
inclusion. Four mHC residual streams with 20 Sinkhorn iterations, clipped
SwiGLU, RMSNorm, vocabulary 154,880 and untied embeddings. No text causal
LM class is exported in this version: Glm5NextForConditionalGeneration
also constructs a vision tower (24 blocks, width 1024, 16 heads, MLP 4096,
output width 1536 and merger MLP 10240).

DEVICE: 25,533,006 parameters, ~13.17 MB estimated int4 weights, including
the minimal vision tower. Counted on the meta device before allocating weights.
- Text width 4096 -> 128, layers 45 -> 6 preserve a full period, the three
  dense bootstrap layers and three sparse MLPs.
- MLA query/KV heads 64/64 -> 2/2 and linear heads 64 -> 2 shrink projections
  while keeping 1:1 MLA sharing and released 256/128 head sizes.
- Dense/expert MLP 12288/2048 -> 384/64, query/KV ranks 1536/512 -> 48/16
  preserve ratios to width.
- Indexer heads 32 -> 1, top-k 2048 -> 4, pool size 16 -> 4 exercise one
  selected compressed block from the two blocks of an 8-token prompt.
- Vocabulary 154,880 -> 8,192 cuts table storage, padding id 154820 -> 0;
  multimodal markers
  become 250..255 so the toy and DEVICE vocabularies both contain them.
- Vision tower -> one width-32 block, two heads, MLP 64, merger MLP 64,
  output width 128, 28x28 input with 14x14 patches. It is counted but receives
  no images in this text stress test.
- Kept: 288 experts/top-8/shared expert, attention/dense-prefix schedules,
  NoPE, index head size 128 and tail inclusion, four mHC streams/20 Sinkhorn
  steps, convolution size, norms, clipping, gates and untied embeddings.
Lowering gap: filtering valid indexer pools makes the pool count
data-dependent. The selection-budget minimum becomes torch.sym_min, which
the edge verifier rejects at verifier.py:312 before ESP32 partitioning.
Both sizes retain this indexed layer and fail there; the error is a bare
AssertionError, and its traceback (at CI size) names the rejected operator.

Other ways to fit it (meta-counted int4 weight estimates, including vision):
- Width 256, four layers and 144 experts: 23.37M parameters, 12.05 MB;
  wider projections with half the expert pool and one sparse MLP after the
  three dense bootstrap layers. MLA/linear heads 4/4, query/KV ranks 96/32.
- Width 192, four layers and all 288 experts: 22.14M, 11.41 MB; keep the
  released pool at a width that is not divisible by 128. MLA/linear heads
  3/3, dense/expert widths 576/96, query/KV ranks 72/24.
"""

MODEL_TYPE = "glm5_next"
MODEL_CLASS = "Glm5NextForConditionalGeneration"

TEXT = dict(
    vocab_size=8192, hidden_size=128, intermediate_size=384, moe_intermediate_size=64,
    num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=2,
    q_lora_rank=48, kv_lora_rank=16, qk_nope_head_dim=256, qk_rope_head_dim=0,
    v_head_dim=256, n_routed_experts=288, num_experts_per_tok=8, n_shared_experts=1,
    layer_types=["linear_attention", "linear_attention", "linear_attention",
                 "indexed_attention", "linear_attention", "linear_attention"],
    mlp_layer_types=["dense"] * 3 + ["sparse"] * 3,
    indexer_types=["full"] * 6, index_topk=4, index_kpool=4,
    index_kpool_always_select_tail=True, index_head_dim=128, index_n_heads=1,
    linear_head_dim=128, linear_num_heads=2, linear_conv_kernel_dim=4,
    linear_lower_bound=-5.0, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
    swiglu_limit=10.0, max_position_embeddings=1048576,
    hidden_act="silu", rms_norm_eps=1e-5, tie_word_embeddings=False, pad_token_id=0,
)

VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
              image_size=28, patch_size=14, out_hidden_size=128,
              projection_intermediate_size=64)

DEVICE = dict(text_config=TEXT, vision_config=VISION, image_token_id=250,
              video_token_id=251, image_start_token_id=252, image_end_token_id=253,
              video_start_token_id=254, video_end_token_id=255, tie_word_embeddings=False)

CI = dict(DEVICE,
          text_config=dict(TEXT, vocab_size=256, hidden_size=64,
                           intermediate_size=192, moe_intermediate_size=32,
                           num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=16,
                           q_lora_rank=24, kv_lora_rank=8, qk_nope_head_dim=16,
                           v_head_dim=16, n_routed_experts=4, num_experts_per_tok=2,
                           layer_types=["linear_attention"] * 3 + ["indexed_attention"],
                           mlp_layer_types=["dense"] * 3 + ["sparse"],
                           indexer_types=["full"] * 4, index_head_dim=16, index_n_heads=4,
                           linear_head_dim=16, linear_num_heads=8, hc_sinkhorn_iters=3),
          vision_config=dict(VISION, out_hidden_size=64))

EXPECT = "lower: pooled indexer's dynamic pool count emits torch.sym_min, rejected by the edge verifier"
