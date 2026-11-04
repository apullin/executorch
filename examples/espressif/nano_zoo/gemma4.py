# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGemma4: Gemma 4 E2B's on-device layout at ESP32-S3 size, random weights.

Released (google/gemma-4-E2B-it, the checkpoint transformers documents
Gemma4TextConfig against; this builds its text tower, Gemma4ForCausalLM,
without the vision and audio encoders): width 1536 (the backbone_hidden_size
of transformers' Gemma 4 assistant config); vocabulary 262,144, input
embedding tied to the output head. Sliding-window layers (window 512) attend
with head size 256 and RoPE over every dimension; full-attention layers use
head size 512 and "proportional" RoPE that rotates a quarter of each head.
Queries and keys are RMS-normalised per head, values too (no scale), and the
score scale is 1. Gated GELU MLP; four RMSNorms per layer. Per-layer
embeddings (PLE): a second table of 256 values per layer per token, mixed into
the residual stream after every layer through a gate, plus a projection of the
main embedding. KV sharing: the last layers reuse the keys/values of the last
earlier layer of their type, optionally with an MLP twice as wide. Not in the
transformers source, and not confirmed offline: the layer count (35,
unverified), the head counts and MLP width (the config defaults, 8 query / 4
key-value heads and an MLP 4x the width, describe a 2304-wide model, not this
one), how many layers share KV and whether they get the double-wide MLP, the
exact sliding/full order (the default is 5 sliding : 1 full, last layer full),
and the logit softcap (the config default is none; DiffusionGemma, built on
Gemma 4, hard-codes 30).

DEVICE: 27.7M parameters, ~14.3 MB at int4 (14.54 MB lowered).
- Width 1536 -> 256.
- 2 query heads sharing 1 key/value head on every layer: 8 query heads of the
  released sizes would make attention 1.5x (sliding) to 3x (full layers) the
  MLP at width 256.
- Layers -> 12: two periods of 5 sliding + 1 full, the second period sharing
  the first one's keys/values with double-wide MLPs (one period of each kind).
- MLP 1024 (4x the width; 2048 on the KV-sharing layers).
- PLE 256 -> 64 values per layer, from an 8,192-token table: at the released
  256 the table alone would be 25M parameters.
- Vocabulary 262,144 -> 8,192, for the main and the per-layer tables.
- Logit softcap 30 (see above).
- Kept as released: head sizes 256 and 512, window 512, the two RoPE kinds,
  QK/V norms with score scale 1, gated GELU, PLE, KV sharing (with the
  double-wide MLPs the config offers), tied embeddings.

Why E2B and not the 26B-A4B mixture of experts: E2B is the variant built for
devices (PLE, KV sharing), and the backend keeps 3-D expert weights in fp32
outside the delegates, so an MoE nano spends most of the flash on experts
(diffusion_gemma carries Gemma 4's MoE block and its key=value attention).

Other ways to fit it:
- Width 384, 2 query heads (query width 1.33x the width, as 8 x 256 is at
  1536), one period of 6 layers, PLE 64: 21.5M, ~11.1 MB; half the depth and
  no KV sharing (a single period has no earlier layer to share from).
- PLE at the released 256 per layer: the table is 2.1M parameters per layer,
  so one period of 6 layers without KV sharing: 23.3M, ~12.0 MB.
- The 26B-A4B layout (a dense MLP plus 8 experts on every layer, keys as
  values on full layers): at 12 layers and 8 experts of 64 the experts alone
  are 4.7M fp32 parameters, 18.9 MB: over the region before anything else.
"""

MODEL_TYPE = "gemma4_text"
MODEL_CLASS = "Gemma4ForCausalLM"

# Random weights leave q_norm/k_norm at scale 1 and the score scale is 1, so
# the scores are ~N(0, head size): near one-hot softmaxes, and the device
# arithmetic itself is that touchy (a 1e-4 nudge of the embedding table moves
# the int4/int8 reference to cosine 0.964). The delegate's int8 Q and K rows
# put ~1% error on every score. Measured at device size: cosine 0.948 against
# the device arithmetic, 0.998 with the q@kT left in fp32. CI's toy heads clear
# the gate (four seeds: 0.9966-0.9992).
EXPECT = {"device": "host: int8 q@kT moves near one-hot QK-normed attention "
                    "(cosine 0.948; 0.998 with q@kT in fp32)"}

# Sliding layers rotate every dimension; full layers rotate a quarter of each
# (wider) head, at a longer wavelength. The transformers defaults, written out.
ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                       "rope_theta": 1_000_000.0},
}


def _pattern(periods):
    """5 sliding-window layers then 1 full-attention layer, `periods` times."""
    return (["sliding_attention"] * 5 + ["full_attention"]) * periods


DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=12,
    num_attention_heads=2, num_key_value_heads=1, head_dim=256,
    # Builder-only kwarg: the head size of the full-attention layers.
    global_head_dim=512,
    layer_types=_pattern(2), sliding_window=512, rope_parameters=ROPE,
    num_kv_shared_layers=6, use_double_wide_mlp=True,
    vocab_size_per_layer_input=8192, hidden_size_per_layer_input=64,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    max_position_embeddings=131072, rms_norm_eps=1e-6, tie_word_embeddings=True,
)

# Own sliding, own full, then a shared sliding and a shared full layer.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256, num_hidden_layers=4,
          head_dim=32, global_head_dim=64, layer_types=["sliding_attention", "full_attention"] * 2,
          sliding_window=4, num_kv_shared_layers=2, vocab_size_per_layer_input=256,
          hidden_size_per_layer_input=16)
