# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoPythia: Pythia-160M's parallel GPT-NeoX decoder at ESP32-S3 size, random weights.

Released (EleutherAI/pythia-160m, marketed as 160M): 12 layers at width 768,
12 query and 12 key/value heads (1:1), head size 64, MLP 3072 (4x width),
vocabulary 50,304, and context 2048. These are Pythia-160M's shapes; the local
GPTNeoXConfig docstring instead names gpt-neox-20b and defaults to its much
larger shape. Both use parallel attention and MLP branches from the same
layer input, LayerNorm, GELU, biased projections, untied embeddings, and
RoPE on only one quarter of each head (theta 10,000).

DEVICE: 27,585,792 parameters, ~14.22 MB estimated at int4.
- Width 768 -> 384 and heads 12 -> 6: keep 64-wide heads while halving width.
- MLP 3072 -> 1536: keep the 4x expansion.
- Vocabulary 50,304 -> 8,192: reduce both untied token matrices.
- Kept as released: 12 layers, 1:1 heads, head size 64, parallel residuals,
  partial RoPE (16 of 64 coordinates), 2048 context, LayerNorm, GELU, biases,
  untied embeddings, and BOS/EOS 0/2.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 256, 4 heads, 24 layers, MLP 1024 (~11.94 MB): deeper and narrower.
- Width 512, 8 heads, 6 layers, MLP 2048 (~14.08 MB): wider and shallower.
- Width 384, 6 heads, 8 layers, MLP 1536 (~10.56 MB): reserve flash space.
"""

MODEL_TYPE = "gpt_neox"
MODEL_CLASS = "GPTNeoXForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, intermediate_size=1536,
    num_hidden_layers=12, num_attention_heads=6, max_position_embeddings=2048,
    hidden_act="gelu", layer_norm_eps=1e-5, use_parallel_residual=True,
    attention_bias=True, attention_dropout=0.0, hidden_dropout=0.0,
    tie_word_embeddings=False, bos_token_id=0, eos_token_id=2, pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0,
                     "partial_rotary_factor": 0.25},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256,
          num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=128)
