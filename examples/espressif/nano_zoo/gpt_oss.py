# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGPT-OSS: gpt-oss-20b's layout at ESP32-S3 size, random weights.

Released (openai/gpt-oss-20b, 20.9B parameters, 3.6B used per token): 24
layers at width 2880, alternating a 128-token sliding-window layer and a
full-attention layer; 64 query heads sharing 8 key/value heads (8:1), head
size 64, biases on the attention projections, and one learned "sink" logit
per head that joins every softmax, so a head can attend to nothing; every
layer's MLP is a mixture of 32 experts of 2,880 (1x width), of which a router
with a bias picks 4 per token (softmax over the 4); the experts' SwiGLU is
clamped (gate at most 7, up within +-7, gate * sigmoid(1.702 * gate) * (up +
1)) and has biases; RMSNorm; YaRN RoPE (theta 150,000, x32 over a 4K original
window); vocabulary 201,088; untied embeddings. (The config class defaults to
gpt-oss-120b's 36 layers and 128 experts.)

DEVICE: 17.4M parameters, ~9.0 MB at int4.
- Width 2880 -> 256: 8 query heads of 64 share 1 key/value head, so the query
  width is now twice the model width (released: 1.4x).
- Experts 2,880 -> 256 wide, 1x the width as released.
- Layers 24 -> 2: one sliding-window layer and one full layer, the released
  pattern's period. The 32 experts cost 6.3M parameters per layer at width
  256; a second pair would pass the region (15.8 MB).
- Vocabulary 201,088 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: attention sinks, the 128-token window alternating with
  full attention, attention biases, 32 experts with 4 per token and the
  router bias, the clamped SwiGLU with (up + 1) and expert biases, head size
  64 and the 8:1 head sharing, YaRN, untied embeddings.

Other ways to fit it:
- 4 layers with 192-wide experts (0.75x; 12.5 MB): two sliding/full pairs.
- 16 experts, 2 per token (the released 1:8 active share), 4 layers (9.3 MB).
- Width 128 with 128-wide experts, 12 layers (11.8 MB): six sliding/full
  pairs, but the query width becomes 4x the model width.

On the board: the backend has no expert kernel. The 3-D expert weights (12.6M
of the 17.4M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 52.96 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "gpt_oss"
MODEL_CLASS = "GptOssForCausalLM"
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 52.96 MB > 15.27 MB"}

YARN = {"rope_type": "yarn", "rope_theta": 150_000.0, "factor": 32.0, "beta_fast": 32.0,
        "beta_slow": 1.0, "truncate": False, "original_max_position_embeddings": 4096}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
    layer_types=["sliding_attention", "full_attention"], sliding_window=128,
    num_attention_heads=8, num_key_value_heads=1, head_dim=64, attention_bias=True,
    num_local_experts=32, num_experts_per_tok=4, swiglu_alpha=1.702, swiglu_limit=7.0,
    max_position_embeddings=131072, rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters=YARN,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=64, head_dim=16)
