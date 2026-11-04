# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGemma3: Gemma 3 270M's layout at ESP32-S3 size, random weights.

Released (google/gemma-3-270m, 268M parameters, 168M of them in the
embedding): 18 layers at width 640; 4 query heads sharing 1 key/value head,
head size 256, so attention is 1024 wide; GeGLU MLP (tanh GELU) of 2048 (3.2x
width); five sliding-window layers (window 512, RoPE theta 10,000) to every
full-attention layer (RoPE theta 1,000,000); RMSNorm on each query and key
head (QK-norm); four RMSNorms per layer, before and after attention and
before and after the MLP, all in Gemma's (1 + weight) form; the embedding
multiplied by sqrt(width); queries scaled by 1/sqrt(256) (query_pre_attn_scalar);
vocabulary 262,144; input embedding tied to the output head; no softcaps
(Gemma 2 had them; Gemma 3 has QK-norm instead). Gemma 3 1B is the same
design at width 1152, MLP 6912 (6x) and 26 layers. Both are text-only:
Gemma3ForCausalLM, not the vision-language Gemma3ForConditionalGeneration of
the 4B and up.

DEVICE: 24.5M parameters, ~12.7 MB at int4.
- Width 640 -> 256. The heads are already the fewest that keep the 4:1
  sharing, so attention stays 1024 wide, four times the width (1.6x
  released).
- MLP 2048 -> 768, 3x the width: 3.2x would be 819, and 768 keeps whole
  groups of 128.
- Vocabulary 262,144 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: all 18 layers (three periods of 5 sliding : 1 full), the
  512-token window, head size 256 and 4:1 sharing, QK-norm, the four norms per
  layer, the scaled embedding, query_pre_attn_scalar 256, both RoPE thetas,
  tanh GELU, tied embeddings.

Other ways to fit it:
- Width 384, MLP 1280, 6 layers (9.2 MB): one period of the pattern, a
  third of the depth, attention 2.7x the width.
- Head size 128 at width 256 (4 query / 1 key-value heads), 24 layers
  (12.5 MB, four periods): a third more depth for half the released head
  size and attention width.
- Keep the 262,144 vocabulary: its table alone is 67M parameters at width
  256, four times the budget.
"""

MODEL_TYPE = "gemma3_text"
MODEL_CLASS = "Gemma3ForCausalLM"

ROPE = {"sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0}}
PERIOD = ["sliding_attention"] * 5 + ["full_attention"]

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=768, num_hidden_layers=18,
    num_attention_heads=4, num_key_value_heads=1, head_dim=256,
    hidden_activation="gelu_pytorch_tanh", query_pre_attn_scalar=256,
    sliding_window=512, layer_types=PERIOD * 3, rope_parameters=ROPE,
    max_position_embeddings=32768, rms_norm_eps=1e-6, tie_word_embeddings=True,
    final_logit_softcapping=None, attn_logit_softcapping=None,
)

# One layer of each type. A window shorter than the 8-token prompt, so the
# sliding mask differs from the causal one and the comparison can see it.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=192,
          num_hidden_layers=2, head_dim=64, query_pre_attn_scalar=64,
          sliding_window=4, layer_types=["sliding_attention", "full_attention"])
