# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoExaone4: EXAONE's hybrid attention at ESP32-S3 size, random weights.

Released: Exaone4Config names LGAI-EXAONE/EXAONE-4.0-32B. Its local defaults
describe width 4096, 32 layers, 32 query / 32 KV heads (1:1), head size 128,
MLP 16384 (4x width), vocabulary 102,400, context 2048, and sliding window
4096. Those defaults are not a verified 32B checkpoint shape: the installed
files contain no checkpoint config, so DEVICE follows the explicit defaults.
Each four-layer period is three sliding layers then one full-attention layer.
Both have QK RMSNorm; RoPE is applied only in the sliding layers, while the
full layers omit position rotation (global NoPE). SwiGLU,
RMSNorm, no projection biases, and untied embeddings complete the layout.
This uses Exaone4ForCausalLM directly, the same text architecture used by the
EXAONE-4.5 vision-language composite, without constructing a vision tower.

DEVICE: 25,181,440 parameters, ~12.98 MB estimated at int4.
- Width 4096 -> 256; heads 32/32 -> 2/2: preserve 128-wide, unshared heads.
- MLP 16384 -> 1024: keep the 4x expansion.
- Layers 32 -> 20: five complete periods fit after the width reduction.
- Vocabulary 102,400 -> 8,192: shrink both untied token matrices.
- Sliding window 4096 -> 4: eight-token smoke inputs exercise restricted
  local attention; the full layers still use the entire input.
- Kept from the reference defaults: period 3 sliding : 1 full, head size 128,
  1:1 heads, QK-norm, sliding RoPE/full NoPE, RMSNorm, SwiGLU, no biases,
  untied embeddings, 2048 context, and BOS/EOS 0/2.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 384, 3 query / 3 KV heads, 8 layers, MLP 1536 (~12.98 MB): wider,
  with two complete periods and the same head size.
- Width 256, 2/2 heads, 16 layers, MLP 1024 (~10.82 MB): four periods and
  more flash space left over.
- Width 512, 4/4 heads, 4 layers, MLP 2048 (~12.98 MB): widest option,
  with only one complete period.
"""

MODEL_TYPE = "exaone4"
MODEL_CLASS = "Exaone4ForCausalLM"

PERIOD = ["sliding_attention"] * 3 + ["full_attention"]

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024,
    num_hidden_layers=20, num_attention_heads=2, num_key_value_heads=2,
    layer_types=PERIOD * 5, sliding_window_pattern=4, sliding_window=4,
    max_position_embeddings=2048, hidden_act="silu", rms_norm_eps=1e-5,
    attention_dropout=0.0, tie_word_embeddings=False, bos_token_id=0,
    eos_token_id=2, pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256,
          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
          layer_types=PERIOD, max_position_embeddings=128)
