# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoNanoChat: the installed nanochat decoder at ESP32-S3 size, random weights.

Released: NanoChatConfig names karpathy/nanochat-d32. Its local defaults are
12 layers at width 768, 6 query and 6 key/value heads (1:1), head size 128,
MLP 8192 (32/3 times width), vocabulary 50,304, and context 2048. The d32
checkpoint's actual dimensions are not supplied by these installed files;
these defaults are the reference layout here, not verified d32 metadata.
modeling_nanochat.py implements parameter-free RMSNorm, including a norm on
the input embeddings; clockwise RoPE with rotate_half=[x2, -x1]; Q/K
normalization AFTER RoPE; a two-projection squared-ReLU MLP, untied embeddings, no projection
biases, and tanh logit softcapping at 15. It does not implement the sliding
layers or value-embedding tables of other nanochat revisions.

DEVICE: 28,704,768 parameters, ~14.80 MB estimated at int4.
- Width 768 -> 384 and heads 6 -> 3: retain 128-wide, unshared heads.
- MLP 8192 -> 4096: preserve the exact 32/3 expansion of the local defaults.
- Layers 12 -> 6: after the width reduction, six layers fill the flash budget.
- Vocabulary 50,304 -> 8,192: shrink both untied token matrices.
- Kept from the installed reference layout: head size 128, 1:1 heads,
  parameter-free norms, input embedding norm, clockwise RoPE then QK-norm,
  squared ReLU, no biases, softcap 15,
  untied embeddings, 2048 context, and BOS/EOS/PAD 0/1/1.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 256, 2 heads, 12 layers, MLP 2731 (~12.44 MB): keep all default
  layers and round the MLP expansion to the nearest integer.
- Width 512, 4 heads, 3 layers, MLP 5461 (~14.60 MB): wider and shallower,
  with the MLP expansion rounded to the nearest integer.
- Width 384, 3 heads, 5 layers, MLP 4096 (~12.87 MB): leave more flash free.
"""

MODEL_TYPE = "nanochat"
MODEL_CLASS = "NanoChatForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, intermediate_size=4096,
    num_hidden_layers=6, num_attention_heads=3, num_key_value_heads=3,
    max_position_embeddings=2048, hidden_act="relu2", rms_norm_eps=1e-6,
    final_logit_softcapping=15.0, attention_bias=False, attention_dropout=0.0,
    tie_word_embeddings=False, bos_token_id=0, eos_token_id=1, pad_token_id=1,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=683,
          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
          max_position_embeddings=128)
