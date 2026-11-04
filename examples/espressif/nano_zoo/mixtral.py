# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoMixtral: Mixtral 8x7B's layout at ESP32-S3 size, random weights.

Released (mistralai/Mixtral-8x7B-v0.1, 46.7B parameters, 12.9B used per
token): 32 layers at width 4096; 32 query heads sharing 8 key/value heads
(4:1), head size 128; every layer's MLP is a mixture of 8 SwiGLU experts of
14,336 (3.5x width), of which a softmax router picks 2 per token and
renormalizes their weights; RMSNorm; RoPE theta 1,000,000; 32K context, no
sliding window; vocabulary 32,000; untied input embedding and output head.

DEVICE: 27.5M parameters, ~14.2 MB at int4.
- Width 4096 -> 256: 4 query heads of 128 share 1 key/value head, so the
  query width is now twice the model width (released: equal).
- Experts 14,336 -> 896 wide, the released 3.5x of the width.
- Layers 32 -> 4: each layer's 8 experts cost 5.5M parameters at width 256.
- Vocabulary 32,000 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: 8 experts with 2 chosen per token, head size 128 and the
  4:1 head sharing, untied embeddings, RoPE theta 1,000,000.

Other ways to fit it:
- Head size 64 (query width equal to the model width, as released): 0.16M
  fewer parameters per layer, still 4 layers (13.9 MB).
- Width 128 with 448-wide experts, 14 layers (12.2 MB): closer to the
  released depth, but the query width becomes 4x the model width.
- Keep the 32,000 vocabulary: its two tables are 16.4M parameters at width
  256, which leaves room for 2 layers (14.5 MB).

On the board: the backend has no expert kernel. The 3-D expert weights (22.0M
of the 27.5M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 90.99 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "mixtral"
MODEL_CLASS = "MixtralForCausalLM"
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 90.99 MB > 15.27 MB"}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=896, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=1, head_dim=128,
    num_local_experts=8, num_experts_per_tok=2, sliding_window=None,
    max_position_embeddings=32768, rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=224, num_hidden_layers=1,
          head_dim=16)
