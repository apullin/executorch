# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoBitNet: BitNet b1.58 2B4T at ESP32-S3 size, random weights.

Released (microsoft/bitnet-b1.58-2B-4T): BitNetConfig's local defaults give
30 layers at width 2560, 20 query / 5 key/value heads (4:1), head size 128,
MLP 6912 (2.7x width), vocabulary 128,256, context 2048, and RoPE theta 500k.
The family is trained for ternary weights, but the installed model uses
ordinary dense projections. This zoo starts with random fp32 weights and the
ESP32 backend quantizes them to int4 like every other family: it exercises
the architecture, not ternary storage or a ternary kernel. It keeps squared
ReLU gating, RMSNorm, extra norms before the attention output projection and
the MLP down projection, no biases, and untied embeddings.

DEVICE: 27,856,330 parameters, ~14.36 MB estimated at int4.
- Width 2560 -> 512; heads 20/5 -> 4/1: the smallest width that keeps both
  128-wide heads and the released 4:1 sharing.
- MLP 6912 -> 1382: round 2.7x width to the nearest integer (1382.4 -> 1382).
- Layers 30 -> 7: the minimum head-preserving width needs fewer layers to fit.
- Vocabulary 128,256 -> 8,192: shrink both untied token matrices.
- BOS/EOS 128,000/128,001 -> 0/1: valid IDs for the smaller vocabulary.
- Kept as released: head size 128, 4:1 sharing, squared-ReLU gated MLP,
  all four per-layer norms, no biases, untied embeddings, 2048 context, RoPE.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 256, 4 query / 1 KV head of size 64, 30 layers, MLP 692 (~12.94 MB):
  keep the depth by halving head size, with the MLP rounded to an even width.
- Width 512, 4/1 heads, 7 layers, MLP 1408 (~14.51 MB): align MLP width to
  128 too, changing the expansion from 2.7x to 2.75x.
- Width 512, 4/1 heads, 6 layers, MLP 1382 (~12.93 MB): leave flash free.
"""

MODEL_TYPE = "bitnet"
MODEL_CLASS = "BitNetForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=512, intermediate_size=1382,
    num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=1,
    max_position_embeddings=2048, hidden_act="relu2", rms_norm_eps=1e-5,
    attention_bias=False, attention_dropout=0.0, tie_word_embeddings=False,
    bos_token_id=0, eos_token_id=1, pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 500_000.0},
)

# Toy head size 16; round the 2.7x MLP to the nearest integer.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=173,
          num_hidden_layers=1, max_position_embeddings=128)
