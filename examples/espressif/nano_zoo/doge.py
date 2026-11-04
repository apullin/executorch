# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoDoge: Doge-320M's dynamic-mask decoder at ESP32-S3 size, random weights.

Released (SmallDoge/Doge-320M): DogeConfig's defaults give width 1024,
32 layers, 8 query / 8 key/value heads (1:1), head size 128, MLP 2048
(2x width), vocabulary 32,768, context 2048, and untied embeddings. It uses
RMSNorm, QK-norm, RoPE, SwiGLU, learned residual multipliers, and attention
masks derived from projected values through softplus and exp. A top-k mask
keeps up to 2048 keys. This is the released dense variant (is_moe=False),
not Doge's separate cross-domain expert option. The checkpoint config itself
is not included locally; these configurations follow the source defaults.

DEVICE: 25,223,488 parameters, ~13.01 MB estimated at int4.
- Width 1024 -> 256; heads 8/8 -> 2/2: keep 128-wide heads and 1:1 sharing.
- MLP 2048 -> 512: preserve the 2x expansion.
- Vocabulary 32,768 -> 8,192: shrink both untied token matrices.
- Keep-window 2048 -> 4: make the eight-token export execute dynamic top-k
  pruning as well as constructing the mask from values.
- Kept as released defaults: 32 layers, head size 128, 1:1 heads, QK-norm, RMSNorm,
  SwiGLU, residual multipliers, no biases, untied embeddings, 2048 context,
  default RoPE, and the dense MLP. A has one entry per key/value head: [2]
  here.
- Smoke inputs mask the last four keys: untrained A=0 gives tied scores, so
  keeping at most four valid keys makes top-k membership deterministic while
  retaining the top-k/scatter pruning path in the graph.

Other ways to fit it (same int4 estimate, vocabulary 8,192):
- Width 384, 3 query / 3 KV heads, 12 layers, MLP 768 (~12.39 MB): wider,
  with fewer layers and the same head size.
- Width 256, 2/2 heads, 24 layers, MLP 512 (~10.30 MB): leave flash free.
- Width 512, 4/4 heads, 6 layers, MLP 1024 (~12.45 MB): wider and shallower.
"""

MODEL_TYPE = "doge"
MODEL_CLASS = "DogeForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=512,
    num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=2048, keep_window_size=4, sliding_window=None,
    hidden_act="silu", rms_norm_eps=1e-6, hidden_dropout=0.0,
    attention_bias=False, attention_dropout=0.0, mlp_bias=False,
    tie_word_embeddings=False, is_moe=False, bos_token_id=None,
    eos_token_id=None, pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=128,
          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
          max_position_embeddings=128)


def INPUTS(cfg):
    """Eight queries, causal attention, at most keep_window_size valid keys."""
    import torch

    from executorch.examples.espressif.nano_zoo import SEQ, build

    inputs = build.ids(cfg)
    query = torch.arange(SEQ)[:, None]
    key = torch.arange(SEQ)[None, :]
    mask = (key <= query) & (key < cfg.keep_window_size)
    inputs["attention_mask"] = mask[None, None]
    return inputs
