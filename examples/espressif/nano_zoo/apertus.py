# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoApertus: Apertus's xIELU decoder at ESP32-S3 size, random weights.

Released: ApertusConfig names swiss-ai/Apertus-8B-Instruct-2509. Its source
defaults describe width 4096, 32 layers, 32 query heads, head size 128,
MLP 14336 (3.5x width), vocabulary 131,072, context 65,536, and untied
embeddings. KV heads default to the query count (1:1); the actual checkpoint's
KV count and MLP dimensions are not verified by the installed files. DEVICE
uses that explicit default layout. The model has QK RMSNorm, RoPE with llama3
scaling (theta 12M, factor 8 over 8192), and a two-projection xIELU MLP. xIELU
has learned positive/negative slopes stored as [1]-shaped parameters; these
remain in the exported model.

DEVICE: 27,287,872 parameters, ~14.07 MB estimated at int4.
- Width 4096 -> 256; heads 32/32 -> 2/2: preserve 128-wide heads and the
  default 1:1 sharing while shrinking width first.
- MLP 14336 -> 896: keep the 3.5x expansion with two projections.
- Vocabulary 131,072 -> 8,192: reduce both untied token matrices.
- Kept from the reference defaults: 32 layers, head size 128, 1:1 heads,
  QK-norm, RMSNorm, xIELU and its [1] parameters, no projection biases,
  untied embeddings, 65,536 context, scaled RoPE, and BOS/EOS/PAD 1/2/3.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 384, 3 query / 3 KV heads, 12 layers, MLP 1344 (~13.29 MB): wider,
  with fewer layers but the same head size and expansion.
- Width 256, 2/2 heads, 24 layers, MLP 896 (~11.09 MB): leave flash free.
- Width 512, 4/4 heads, 6 layers, MLP 1792 (~13.25 MB): wider and shallower.
"""

MODEL_TYPE = "apertus"
MODEL_CLASS = "ApertusForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=896,
    num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=65536, hidden_act="xielu", rms_norm_eps=1e-5,
    attention_bias=False, attention_dropout=0.0, tie_word_embeddings=False,
    bos_token_id=1, eos_token_id=2, pad_token_id=3,
    rope_parameters={"rope_type": "llama3", "rope_theta": 12_000_000.0,
                     "factor": 8.0, "original_max_position_embeddings": 8192,
                     "low_freq_factor": 1.0, "high_freq_factor": 4.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=224,
          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
