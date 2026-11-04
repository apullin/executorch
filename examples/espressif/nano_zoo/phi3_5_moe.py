# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoPhi-3.5-MoE: Phi-3.5-MoE's layout at ESP32-S3 size, random weights.

Released (microsoft/Phi-3.5-MoE-instruct, 41.9B parameters, 6.6B used per
token): 32 layers at width 4096; 32 query heads sharing 8 key/value heads
(4:1), head size 128; LayerNorm (with a bias) where most models use RMSNorm;
every layer's MLP is a mixture of 16 SwiGLU experts of 6,400 (1.5625x width),
2 per token, routed by SparseMixer: the top expert, then the best of the
rest, each weighted by a softmax over only the experts whose score is within
a relative margin of it (twice the jitter of 0.01); LongRoPE (128K context
from a 4K original window, with per-frequency scale factors); vocabulary
32,064; untied embeddings. Biases on the attention projections and the
output head: the config class defaults both off; the release is taken to turn
both on (the class exists only for this model, and these switches with it),
unverified against its config.json.

DEVICE: 24.4M parameters, ~12.6 MB at int4.
- Width 4096 -> 256: 4 query heads of 128 share 1 key/value head, so the
  query width is now twice the model width (head_dim is set explicitly: the
  config class does not declare it).
- Experts 6,400 -> 384 wide: 1.5x instead of 1.5625x, a multiple of 128.
- Layers 32 -> 4: the 16 experts cost 4.7M parameters per layer at width 256.
- Plain RoPE at the release's theta (10,000) instead of LongRoPE: its 2 x 64
  scale factors are searched values that transformers' source does not carry,
  and a board context of a few hundred tokens would only ever use the short
  ones.
- Vocabulary 32,064 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: 16 experts with 2 per token through SparseMixer, its
  0.01 margin, LayerNorm, the attention and output-head biases, head size 128
  and the 4:1 head sharing, untied embeddings.

Other ways to fit it:
- Keep the 1.5625x experts (400 wide): 13.0 MB at 4 layers, but their down
  projections span 3 1/8 quantisation groups.
- Width 128 with 192-wide experts, 16 layers (12.2 MB): half the released
  depth, but the query width becomes 4x the model width.
- Keep the 32,064 vocabulary: its two tables are 16.4M parameters at width
  256, which leaves room for 2 layers (13.7 MB).

On the board: the backend has no expert kernel. The 3-D expert weights (18.9M
of the 24.4M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 78.48 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "phimoe"
MODEL_CLASS = "PhimoeForCausalLM"
# PhimoeConfig does not declare head_dim; the attention reads it with getattr.
UNDECLARED = ("head_dim",)
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 78.48 MB > 15.27 MB"}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=384, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=1, head_dim=128,
    num_local_experts=16, num_experts_per_tok=2, router_jitter_noise=0.01,
    attention_bias=True, lm_head_bias=True, sliding_window=None,
    max_position_embeddings=131072, rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=96, num_hidden_layers=1,
          head_dim=16)
