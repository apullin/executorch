# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoQwen3-MoE: Qwen3-30B-A3B's layout at ESP32-S3 size, random weights.

Released (Qwen/Qwen3-30B-A3B, 30.5B parameters, 3.3B used per token): 48
layers at width 2048; 32 query heads sharing 4 key/value heads (8:1), head
size 128, so the query width (4096) is twice the model width; RMSNorm on each
query and key head (QK-norm); every layer's MLP is a mixture of 128 SwiGLU
experts of 768 (0.375x width), of which a softmax router picks 8 per token and
renormalizes their weights; RoPE theta 1,000,000; vocabulary 151,936; untied
input embedding and output head.

DEVICE: 24.3M parameters, ~12.5 MB at int4.
- Width 2048 -> 256. The 8 query heads of 128 still share 1 key/value head,
  so the query width is now 4x the model width (head_dim is set explicitly:
  the config class does not declare it, the release's config.json does).
- Experts 768 -> 96 wide, the released 0.375x of the width.
- Layers 48 -> 2: the 128 experts cost 9.4M parameters per layer at width
  256, so two layers fill the budget.
- Vocabulary 151,936 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: 128 experts with 8 chosen per token and renormalized,
  head size 128 and the 8:1 head sharing, QK-norm, untied embeddings, RoPE.

Other ways to fit it:
- 64 experts, 4 chosen per token (the released 1:16 active share), 4 layers
  (13.1 MB): twice the depth, half the experts.
- Width 128 with 48-wide experts, 8 layers (12.1 MB): more depth, but each
  expert is a sliver and the query width is 8x the model width.
- Keep the 151,936 vocabulary: its two tables alone are 77.8M parameters at
  width 256, five times the budget.

On the board: the backend has no expert kernel. The 3-D expert weights (18.9M
of the 24.3M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 78.3 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "qwen3_moe"
MODEL_CLASS = "Qwen3MoeForCausalLM"
# Qwen3MoeConfig does not declare head_dim; the attention reads it with getattr.
UNDECLARED = ("head_dim",)
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 78.35 MB > 15.27 MB"}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, num_hidden_layers=2,
    num_attention_heads=8, num_key_value_heads=1, head_dim=128,
    num_experts=128, num_experts_per_tok=8, moe_intermediate_size=96,
    norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[],
    max_position_embeddings=40960, rms_norm_eps=1e-6, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_hidden_layers=1, head_dim=16,
          moe_intermediate_size=24)
