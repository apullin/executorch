# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoLFM2: LFM2-350M's short-conv/attention layout at ESP32-S3 size.

Released (LiquidAI/LFM2-350M, about 350M parameters): width 1024, 16 layers,
16 query / 8 KV heads (2:1), head size 64, vocabulary 65,536 and tied
embeddings. Ten layers use double-gated, three-tap depthwise convolutions;
six use full attention with Q/K RMSNorm before full-head RoPE (theta 1M).
Every layer has SwiGLU with actual matrix width 4608 (4.5x hidden width).
The advertised pre-adjustment FFN width is about 6.5x: the implementation
applies 2/3 and rounds before building the matrices. The released small
shape and schedule are those of ExecuTorch's
examples/models/lfm2/config/lfm2_350m_config.json (its layer_types list has an
unused seventeenth entry after n_layers=16).
transformers 5.19.0 configuration_lfm2.py instead defaults to LFM2-1.2B;
using its unspecified layer_types would silently remove all short convs.

DEVICE: 20,070,912 parameters, ~10.35 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width 1024 -> 256, heads 16/8 -> 4/2: retain head size 64 and 2:1 sharing
  while fitting group-128 projection rows.
- FFN pre-adjustment width -> 1664; adjusted width 4608 -> 1152: retain
  the actual 4.5x-width SwiGLU rather than budgeting the advertised width.
- FFN rounding multiple 256 -> 128: keep the adjusted ratio at nano width.
- Vocabulary 65,536 -> 8,192: retrain a small deployment tokenizer.
- Kept as released: all 16 layers in their original ten-conv/six-attention
  order, three-tap bias-free double-gated convolution, Q/K norm, RMSNorm,
  full-head RoPE, SiLU and tied embeddings. CI has one conv and one full
  attention layer at width 64, head size 16, FFN 288 and 256 tokens.

Other ways to fit it:
- Width 384, actual FFN 1728 and the repository's eight-layer five-conv/
  three-attention schedule (~12.04 MB): wider channels, but discard half
  of the released schedule; FFN rounding uses 64 instead of 256.
- Width 256 and 20 layers (~12.68 MB), extending the schedule with its first
  four blocks: more depth, a different overall conv/attention mixture.
- Keep 65,536 tokens at width 128: the tied table costs about 4.33 MB at
  int4, but head size must fall to 32 to retain 2:1 sharing.
"""

MODEL_TYPE = "lfm2"
MODEL_CLASS = "Lfm2ForCausalLM"

LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv",
    "conv", "full_attention", "conv", "conv",
    "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv",
]

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1664,
    num_hidden_layers=16, num_attention_heads=4, num_key_value_heads=2,
    layer_types=LAYER_TYPES, conv_L_cache=3, conv_bias=False,
    block_multiple_of=128, block_ffn_dim_multiplier=1.0,
    block_auto_adjust_ff_dim=True, norm_eps=1e-5, tie_word_embeddings=True,
    max_position_embeddings=128000,
    rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=416,
          num_hidden_layers=2, layer_types=["conv", "full_attention"],
          block_multiple_of=16)
