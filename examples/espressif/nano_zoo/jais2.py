# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoJais2: Jais-2's LayerNorm decoder at ESP32-S3 size, random weights.

Released (inceptionai/Jais-2-8B-Chat): the local Jais2Config defaults give
width 3328, 32 layers, 26 query / 26 KV heads (1:1), head size 128, MLP
26624 (8x width), vocabulary 150,272, and context 8192. The example text
calls it a 7B-style model; the checkpoint named by the decorator is 8B-Chat.
It uses LayerNorm, RoPE, a two-projection squared-ReLU MLP (not a gated MLP),
biased attention and MLP projections, and untied input/output embeddings.

DEVICE: 25,235,968 parameters, ~13.01 MB estimated at int4.
- Width 3328 -> 256; heads 26/26 -> 2/2: preserve 128-wide, unshared heads.
- MLP 26624 -> 2048: retain the 8x expansion.
- Layers 32 -> 16: shrink depth after width to fit the flash region.
- Vocabulary 150,272 -> 8,192: reduce both untied token matrices.
- EOS 150,024 -> 1: put the special token inside the smaller vocabulary.
- Kept as released defaults: head size 128, 1:1 heads, LayerNorm, squared
  ReLU, all projection biases, untied embeddings, 8192 context, default RoPE,
  and BOS 0.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 384, 3 query / 3 KV heads, 7 layers, MLP 3072 (~13.91 MB): wider,
  with fewer layers and the same expansion.
- Width 256, 2/2 heads, 12 layers, MLP 2048 (~10.30 MB): leave flash free.
- Width 128, 2/2 heads of size 64, 64 layers, MLP 1024 (~11.97 MB): much
  deeper, but halve head size and double the released depth.
"""

MODEL_TYPE = "jais2"
MODEL_CLASS = "Jais2ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=2048,
    num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
    head_dim=128, max_position_embeddings=8192, hidden_act="relu2",
    layer_norm_eps=1e-5, attention_bias=True, attention_dropout=0.0,
    mlp_bias=True, tie_word_embeddings=False, bos_token_id=0, eos_token_id=1,
    pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=512,
          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
          head_dim=16, max_position_embeddings=128)
