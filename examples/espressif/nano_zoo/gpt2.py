# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGPT2: GPT-2 124M at ESP32-S3 size, random weights.

Released (openai-community/gpt2): 12 layers at width 768, 12 query and
12 key/value heads (1:1), head size 64, MLP 3072 (4x width), vocabulary
50,257. GPT2Config's defaults in transformers 5.19.0 describe this checkpoint.
It uses LayerNorm, tanh-approximate GELU, learned positions for 1024 tokens,
biases, tied input/output embeddings, and HF Conv1D projections (ordinary
dense matrix multiplies with their weight axes reversed).

DEVICE: 24,833,280 parameters, ~12.80 MB estimated at int4.
- Width 768 -> 384 and heads 12 -> 6: halve width while keeping 64-wide heads.
- MLP 3072 -> 1536: preserve the 4x expansion.
- Vocabulary 50,257 -> 8,192: leave flash for the decoder rather than tokens.
- BOS/EOS 50,256 -> 0/1: keep special token IDs inside the smaller vocabulary.
- Kept as released: all 12 layers, 1:1 heads, head size 64, 1024 learned
  positions, LayerNorm, GELU, biases, tied embeddings, and Conv1D projections.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 256, 4 heads, 24 layers, MLP 1024 (~10.99 MB): more depth, less width.
- Width 512, 8 heads, 6 layers, MLP 2048 (~12.19 MB): wider, half the depth.
- Width 384, 6 heads, 10 layers, MLP 1536 (~10.97 MB): keep more flash free.
"""

MODEL_TYPE = "gpt2"
MODEL_CLASS = "GPT2LMHeadModel"

DEVICE = dict(
    vocab_size=8192, n_embd=384, n_layer=12, n_head=6, n_inner=1536,
    n_positions=1024, activation_function="gelu_new", layer_norm_epsilon=1e-5,
    resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, scale_attn_weights=True,
    scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False,
    add_cross_attention=False, tie_word_embeddings=True, bos_token_id=0,
    eos_token_id=1, pad_token_id=None,
)

CI = dict(DEVICE, vocab_size=256, n_embd=64, n_layer=1, n_head=4,
          n_inner=256, n_positions=128)
