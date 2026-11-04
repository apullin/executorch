# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoOLMoE: OLMoE-1B-7B's layout at ESP32-S3 size, random weights.

Released (allenai/OLMoE-1B-7B-0924, 6.9B parameters, 1.3B used per token): 16
layers at width 2048; 16 attention heads, each with its own key and value (no
sharing), head size 128; RMSNorm over the whole query vector and the whole key
vector (QK-norm across heads, not per head as in Qwen3); every layer's MLP is
a mixture of 64 SwiGLU experts of 1,024 (0.5x width), of which a softmax
router picks 8 per token without renormalizing; RoPE theta 10,000; 4,096
context; vocabulary 50,304; untied input embedding and output head. (The
config class defaults the expert width to 2,048; 1,024 is what matches the
released parameter counts.)

DEVICE: 23.9M parameters, ~12.3 MB at int4.
- Width 2048 -> 256: 2 heads of 128, each still with its own key and value.
- Experts 1,024 -> 128 wide, the released 0.5x of the width.
- Layers 16 -> 3: the 64 experts cost 6.3M parameters per layer at width 256.
- Vocabulary 50,304 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: 64 experts with 8 chosen per token, not renormalized;
  head size 128 without head sharing; QK-norm across heads; untied
  embeddings; RoPE theta 10,000; token 1 as padding (a zero embedding row).

Other ways to fit it:
- 16 experts, 2 chosen per token (the released 1:8 active share), 12 layers
  (13.5 MB): four times the depth, a quarter of the experts.
- Width 128 with 64-wide experts, 16 layers (14.7 MB): the released depth,
  but a single 128-wide head per layer.
- Keep the 50,304 vocabulary: its two tables are 25.8M parameters at width
  256, leaving no room for a single MoE layer.

On the board: the backend has no expert kernel. The 3-D expert weights (18.9M
of the 23.9M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 78.14 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "olmoe"
MODEL_CLASS = "OlmoeForCausalLM"
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 78.14 MB > 15.27 MB"}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=2, num_key_value_heads=2,
    num_experts=64, num_experts_per_tok=8, norm_topk_prob=False, clip_qkv=None,
    max_position_embeddings=4096, rms_norm_eps=1e-5, tie_word_embeddings=False,
    pad_token_id=1, rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=32, num_hidden_layers=1)
