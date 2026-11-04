# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoSolarOpen: Solar-Open's routed experts at ESP32-S3 size, random weights.

Released (upstage/Solar-Open-100B): local SolarOpenConfig defaults describe
width 4096, 48 layers, 64 query / 8 KV heads (8:1), head size 128, vocabulary
196,608, context 131,072, and untied embeddings. Query projection width is
8192, twice the residual width. Each layer has 128 routed SwiGLU experts,
top-8 routing, one always-active shared expert, and expert MLP width 1280
(5/16 of residual width). Routing uses sigmoid scores, a correction-bias
buffer, group selection (one group), and normalized top-k weights. RMSNorm,
no projection biases, and RoPE theta 1M complete this default layout; an
independent checkpoint config is not included in the installed source.

DEVICE: 21,292,288 parameters, ~10.98 MB estimated at int4. The actual
program and weights occupy 65.84 MB: indexed 3-D expert banks stay fp32,
so this configuration cannot fit flash until the backend supports their
int4 storage. It lowers and matches device arithmetic; its flash check is
a strict expected failure. CI occupies 2.04 MB and passes normally.
- Residual width 4096 -> 256: leave room for all 128 routed experts.
- Heads 64/8 -> 8/1, retaining 8:1 and head size 128: eight queries are the
  minimum at that ratio. Query width is consequently 4x residual width here,
  rather than 2x; keeping 2x at width 512 exceeds the budget even at one layer.
- Expert width 1280 -> 80: preserve the exact 5/16 expansion.
- Layers 48 -> 2: the expert banks dominate the parameter budget.
- Vocabulary 196,608 -> 8,192: shrink both untied token matrices.
- Kept as released defaults: 128 routed experts, top-8, one shared expert,
  sigmoid/group/correction-bias routing, normalized weights, 8:1 heads, head
  size 128, RMSNorm, SwiGLU, no biases, untied embeddings, context, and RoPE.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 512, 8/1 heads, 1 layer, 64 routed experts, expert width 160
  (~13.19 MB): retain 2x query width but halve the expert count.
- Width 256, 8/1 heads, 3 layers, 64 routed experts, expert width 80
  (~9.28 MB): fewer experts and more depth, retaining top-8 routing.
- Width 128, 8/1 heads, 8 layers, 128 experts, expert width 40 (~10.54 MB):
  keep every expert and more layers, but query width becomes 8x residual width.
"""


MODEL_TYPE = "solar_open"
MODEL_CLASS = "SolarOpenForCausalLM"

# The flash gap affects DEVICE. A CI pytest run must still pass normally.
EXPECT = {"device": "flash: indexed 3-D expert banks stay fp32; "
                     "65.84 MB exceeds the 15.27 MB model region"}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, num_hidden_layers=2,
    num_attention_heads=8, num_key_value_heads=1, head_dim=128,
    moe_intermediate_size=80, n_routed_experts=128, num_experts_per_tok=8,
    n_shared_experts=1, n_group=1, topk_group=1, norm_topk_prob=True,
    routed_scaling_factor=1.0, max_position_embeddings=131072,
    hidden_act="silu", rms_norm_eps=1e-5, attention_bias=False,
    attention_dropout=0.0, tie_word_embeddings=False, output_router_logits=False,
    bos_token_id=None, eos_token_id=None, pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
)

# One layer contains the complete layout; retain DEVICE's 4x query width.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_hidden_layers=1,
          num_attention_heads=16, num_key_value_heads=2, head_dim=16,
          moe_intermediate_size=20, max_position_embeddings=128)
