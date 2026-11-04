# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoStep3.7: Step-3.7-Flash's layout at ESP32-S3 size, random weights.

Released (stepfun-ai/Step-3.7-Flash), from transformers' config defaults and
the source's notes on the checkpoint: a text model of 45 layers at width
4096 (about 196B parameters, 11B used per token, by the defaults'
arithmetic) and a vision tower of 47 layers at width 1536 (about 2B). Text:
64 query heads sharing 8 key/value heads (8:1), head size 128, RMSNorm on
each query and key head (weights stored as 1 + w), and a sigmoid gate per
head on the attention output. Layers 0-2 have a dense SwiGLU MLP of 11,264
(2.75x width); layers 3-44 a mixture of 288 experts of 1,280 (0.31x), 8
chosen per token by sigmoid scores plus a learned per-expert bias and
renormalized, beside a shared expert of 1,280; the last two layers clamp
their SwiGLU (routed experts at 7, shared expert at 16). Vocabulary 128,815;
untied embeddings; three multi-token-prediction layers outside the forward
pass. The attention mixes sliding-window and full layers (the sliding ones
may use a different head count); taken here as three sliding layers per full
layer with a 512-token window, unverified: transformers' source does not carry
the pattern, and the hub config is not checked.

DEVICE: 26.3M parameters, ~13.6 MB at int4.
- Width 4096 -> 256: 8 query heads of 128 share 1 key/value head (query width
  4x the model width; released 2x).
- Layers 45 -> 4, the release's first four: 3 dense layers, then 1 MoE layer
  (the 288 experts cost 17.7M parameters at width 256); three sliding-window
  layers then a full one.
- Dense MLP 11,264 -> 768 (3x instead of 2.75x, a multiple of 128); experts
  and the shared expert 1,280 -> 80 wide (0.31x).
- The SwiGLU clamps (7 and 16) move to the last layer, the only MoE one.
- Vision tower 47 layers at width 1536 -> 1 layer at width 32 on 28-pixel
  images: the export feeds text only, so the tower is never run, but it is
  built and counted. Its default is the released size (43 GB), which a flat
  config that leaves it unset would build.
- Vocabulary 128,815 -> 8,192; the image placeholder token moves to 8,191.
- Kept as released: 288 experts with 8 per token plus the shared expert, the
  biased sigmoid routing, the 3 leading dense layers, per-head output gates,
  QK-norm, head size 128 and the 8:1 head sharing, untied embeddings.

Other ways to fit it:
- 72 experts (a quarter), 3 dense + 3 MoE layers (12.0 MB): depth over
  expert count.
- Width 128 with 40-wide experts, 3 dense + 4 MoE layers (11.7 MB): all 288
  experts over more MoE layers, but the query width becomes 8x the model
  width.
- Every MLP dense, no experts: that lowers, but it is not a released model.

On the board: not yet. Step3p7Experts loops over the experts the router hit
(a nonzero), and transformers gives it no batched path, so torch.export fails
at the first MoE layer.
"""

MODEL_TYPE = "step3p7"
MODEL_CLASS = "Step3p7ForConditionalGeneration"
EXPECT = ("export: Step3p7Experts loops over the experts the router hit (nonzero), a "
          "data-dependent loop torch.export cannot trace; it has no batched_mm path")

TEXT = dict(
    vocab_size=8192, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=8, num_sliding_attention_heads=8, num_key_value_heads=1, head_dim=128,
    layer_types=["sliding_attention"] * 3 + ["full_attention"], sliding_window=512,
    mlp_layer_types=["dense"] * 3 + ["sparse"], intermediate_size=768,
    # num_local_experts is an alias of n_routed_experts; both are set so
    # neither default (128, 288) wins by declaration order.
    n_routed_experts=288, num_local_experts=288, num_experts_per_tok=8,
    moe_intermediate_size=80, share_expert_dim=80, moe_router_scaling_factor=1.0,
    swiglu_limits=[0, 0, 0, 7], swiglu_limits_shared=[0, 0, 0, 16],
    rms_norm_eps=1e-5, tie_word_embeddings=False, pad_token_id=1,
)
VISION = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
              mlp_ratio=8960 / 1536, image_size=28, patch_size=14)

DEVICE = dict(text_config=TEXT, vision_config=VISION, image_token_id=8191)

CI = dict(
    text_config=dict(TEXT, vocab_size=256, hidden_size=64, num_hidden_layers=2, head_dim=16,
                     layer_types=["sliding_attention", "full_attention"],
                     mlp_layer_types=["dense", "sparse"], intermediate_size=192,
                     moe_intermediate_size=16, share_expert_dim=16,
                     swiglu_limits=[0, 7], swiglu_limits_shared=[0, 16]),
    vision_config=VISION, image_token_id=255)
