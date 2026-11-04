# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoPhi-4-mini: Phi-4-mini's layout at ESP32-S3 size, random weights.

Released (microsoft/Phi-4-mini-instruct, 3.8B parameters, 0.61B of them in
the embedding; transformers runs it as Phi3ForCausalLM): 32 layers at width
3072; 24 query heads sharing 8 key/value heads (3:1), head size 128; q, k and
v come out of one fused projection (qkv_proj) and the MLP's gate and up out of
another (gate_up_proj; SwiGLU of 8192, 2.67x width), the family's mark; RoPE
on only the first 96 of each head's 128 dims (partial_rotary_factor 0.75),
theta 10,000, with LongRoPE rescaling from its original 4K positions to a
128K context; RMSNorm eps 1e-5; vocabulary 200,064; input embedding tied to
the output head. Its sliding window is longer than its context, so it never
applies.

DEVICE: 25.2M parameters, ~13.0 MB at int4.
- Width 3072 -> 384 with the heads 24/8 -> 3/1. Phi-3's head size is
  width / heads (the config has no head_dim), so 384 is the narrowest width
  that keeps both the 128-wide heads and the 3:1 sharing.
- MLP 8192 -> 1024, still 2.67x the width.
- Layers 32 -> 14.
- Vocabulary 200,064 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Plain RoPE instead of LongRoPE: its 48 short and 48 long rescale factors are
  checkpoint data, and it picks between them with a branch on the position
  (past 4,096 or not) that torch.export cannot trace. Under 4K tokens, all a
  device holds, LongRoPE is RoPE with rescaled frequencies and cos/sin times a
  constant: the same graph with other constants.
- No sliding window (see above).
- Kept as released: the fused qkv and gate_up projections, partial RoPE (96 of
  128 dims), head size 128 and 3:1 sharing, the 2.67x MLP, tied embeddings,
  RoPE theta 10k, RMSNorm eps 1e-5.

Other ways to fit it:
- Width 768, 6 query / 2 key-value heads, MLP 2048, 3 layers (13.0 MB): the
  released proportions at a quarter of the width, a tenth of the depth.
- Width 384 with 16 layers (14.6 MB): two more layers, leaving little of the
  15.27 MB region for the program.
- Keep the 200,064 vocabulary: its table alone is 76.8M parameters at width
  384, five times the budget.
"""

MODEL_TYPE = "phi3"
MODEL_CLASS = "Phi3ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, intermediate_size=1024, num_hidden_layers=14,
    num_attention_heads=3, num_key_value_heads=1, hidden_act="silu",
    max_position_embeddings=131072, original_max_position_embeddings=4096,
    rms_norm_eps=1e-5, tie_word_embeddings=True, sliding_window=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0,
                     "partial_rotary_factor": 0.75},
    # The class's special-token ids (32000) lie outside the 8,192 vocabulary;
    # nn.Embedding refuses a padding index past its table.
    pad_token_id=None, bos_token_id=None, eos_token_id=None,
)

# Width 96, not 64: the 3 heads must divide it (head size 32, RoPE on 24).
CI = dict(DEVICE, vocab_size=256, hidden_size=96, intermediate_size=256,
          num_hidden_layers=2)
