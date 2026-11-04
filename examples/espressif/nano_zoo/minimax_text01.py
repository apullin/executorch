# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoMiniMax-Text-01: MiniMax-Text-01's layout at ESP32-S3 size, random weights.

Released (MiniMaxAI/MiniMax-Text-01, 456B parameters, 45.9B used per token):
80 layers at width 6144 in a repeating period of 7 "lightning" attention
layers then 1 softmax attention layer. Lightning attention is linear: instead
of a key/value cache it keeps a decaying 128 x 128 state per head, computed in
256-token blocks (inside a block: masked, decayed query-key scores; across
blocks: the state), with 64 heads of 128, an RMSNorm and a sigmoid output
gate. The softmax layers have 64 query heads sharing 8 key/value heads (8:1),
head size 128, RoPE theta 10,000,000. Every layer's MLP is a mixture of 32
SwiGLU experts of 9,216 (1.5x width), 2 chosen per token by a softmax router
and renormalized. Post-norm residuals, each branch scaled by a constant
(DeepNorm-style; 3.557 = (2 x 80 layers)^(1/4) in the hub config, unverified:
transformers' source does not carry it and defaults to 1). Vocabulary
200,064; untied embeddings.

DEVICE: 25.9M parameters, ~13.4 MB at int4.
- Width 6144 -> 128: the 7:1 period needs 8 layers, and 32 experts at 1.5x
  cost 2.4M parameters per layer even at width 128.
- Heads 64 -> 8 of 128 in both attention types; the softmax layer's 8 query
  heads share 1 key/value head, so its query width is 8x the model width.
- Experts 9,216 -> 192 wide, the released 1.5x of the width.
- Layers 80 -> 8: one period, 7 lightning layers then 1 softmax layer.
- Lightning blocks 256 -> 4 tokens: the export runs 8 tokens, so 256-token
  blocks would leave the across-block path out of the graph; 4 runs both.
- Residual scale 2.0 = (2 x 8 layers)^(1/4), the same rule at this depth.
- Vocabulary 200,064 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: the 7:1 lightning/softmax period, 32 experts with 2 per
  token, head size 128 and the 8:1 head sharing, the lightning output gate
  and norm, post-norm residuals, RoPE theta, untied embeddings.

Other ways to fit it:
- Width 256 with 8 experts and heads of 64, 8 layers (14.4 MB): a wider model
  with Mixtral's expert count.
- Width 128, two periods (16 layers) with 16 experts and heads of 64
  (13.3 MB): twice the depth, half the experts.
- Keep the 200,064 vocabulary: its two tables are 51.2M parameters even at
  width 128, nearly twice the budget.

On the board: the backend has no expert kernel. The 3-D expert weights (18.9M
of the 25.9M parameters) stay outside the delegates in fp32, gathered per
token by router index and multiplied by portable bmm, so the lowered program
is 79.38 MB and overflows the flash region; the rest lowers and matches the
device arithmetic.
"""

MODEL_TYPE = "minimax"
MODEL_CLASS = "MiniMaxForCausalLM"
EXPECT = {"device": "flash: the 3-D expert weights stay outside the delegates in fp32 "
                    "(portable index + bmm): 79.38 MB > 15.27 MB"}

PERIOD = ["linear_attention"] * 7 + ["full_attention"]
RESIDUAL = dict(full_attn_alpha_factor=2.0, full_attn_beta_factor=1.0,
                linear_attn_alpha_factor=2.0, linear_attn_beta_factor=1.0,
                mlp_alpha_factor=2.0, mlp_beta_factor=1.0)

DEVICE = dict(
    vocab_size=8192, hidden_size=128, intermediate_size=192, num_hidden_layers=8,
    layer_types=PERIOD, block_size=4,
    num_attention_heads=8, num_key_value_heads=1, head_dim=128,
    num_local_experts=32, num_experts_per_tok=2, **RESIDUAL,
    rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
          layer_types=["linear_attention", "full_attention"], head_dim=16)
