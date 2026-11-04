# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoCosmos3Edge: NVIDIA Cosmos3 Edge's language model at ESP32-S3 size.

Released (nvidia/Cosmos3-Edge; the shape below is transformers' Cosmos3Edge
config defaults, documented against that checkpoint and not compared with
its config.json, which is not downloaded): a vision-language model. Its language
model: 28 layers at width 2048; 16 query heads sharing 8 key/value heads
(2:1), head size 128, no query/key norms; a two-matrix MLP of 9216 (4.5x)
with squared-ReLU activation (no gate); RMSNorm; untied output head;
vocabulary 131,072. Positions use 3-axis M-RoPE: each head's 64 rotary pairs
are dealt round-robin to time, height and width (24/20/20 pairs; a text token
uses the same position on all three axes), wavelength base 100,000,000. The
vision side is a SigLIP-2-style encoder (27 layers at 1152, 16-pixel
patches, 2x2 patch merging) and an 11,520-wide projector.

DEVICE: 26.3M parameters, ~13.5 MB at int4 (13.80 MB lowered).
- Width 2048 -> 256: 2 query heads sharing 1 key/value head (the released
  2:1 at head size 128; query width 1x the width, as released).
- MLP 9216 -> 1152 (4.5x the width, as released).
- Vocabulary 131,072 -> 8,192, for the embedding and the separate head.
- Vision encoder -> one layer, 32 wide, with a 64-wide projector: there is
  no text-only class in transformers, so the composite is built; the tower
  never runs on a text prompt and is not in the exported program.
- Kept as released: all 28 layers, head size 128 and 2:1 sharing, the 24/20/20
  M-RoPE split and its base, squared ReLU, no biases, untied head.

Other ways to fit it:
- Width 384 at head size 128: 3 query heads, so 1 key/value head (3:1), and
  12 layers: 27.0M, ~13.9 MB; wider, under half the depth.
- Width 512, 4 query / 2 key-value heads (the released 2:1), 6 layers: 27.3M,
  ~14.1 MB; a quarter of the released width, a fifth of the depth.
- Keep the 131K vocabulary: embedding and head are 33.6M parameters each at
  width 256; with 4 layers that is 70.3M, over twice the budget.
"""

MODEL_TYPE = "cosmos3_edge"
MODEL_CLASS = "Cosmos3EdgeForConditionalGeneration"

TEXT = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1152, num_hidden_layers=28,
    num_attention_heads=2, num_key_value_heads=1, head_dim=128, hidden_act="relu2",
    # The 24/20/20 rotary pairs must sum to head_dim / 2.
    rope_parameters={"rope_type": "default", "rope_theta": 100_000_000.0,
                     "mrope_section": [24, 20, 20]},
    max_position_embeddings=131072, rms_norm_eps=1e-5, attention_bias=False, mlp_bias=False,
    tie_word_embeddings=False,
)

VISION = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
              patch_size=4, num_patches=16, spatial_merge_size=2)

DEVICE = dict(text_config=TEXT, vision_config=VISION, projector_hidden_size=64,
              tie_word_embeddings=False)

CI = dict(DEVICE, text_config=dict(
    TEXT, vocab_size=256, hidden_size=64, intermediate_size=288, num_hidden_layers=2, head_dim=32,
    rope_parameters={"rope_type": "default", "rope_theta": 100_000_000.0,
                     "mrope_section": [6, 5, 5]}))
