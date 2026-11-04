# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGemma4Unified: the text tower of Gemma 4 12B "unified" at ESP32-S3 size.

Released (google/gemma-4-12B-it, the checkpoint transformers documents the
unified configs against): a decoder that takes images and audio without
separate encoders -- image patches and raw 40 ms audio frames are projected
straight into its residual stream -- so this text tower is nearly the whole
model. Width 3840 (the vision embedder's output width, which its docstring
says matches the text width); vocabulary 262,144, input embedding tied to the
output head. Gemma 4 attention as the config defaults have it: sliding-window
layers (window 1024) with head size 256 and full RoPE, full-attention layers
with head size 512 and proportional RoPE over a quarter of each head; queries
and keys RMS-normalised per head, values too, score scale 1; 262,144
positions. Gated GELU MLP, four RMSNorms per layer. No per-layer embeddings
and no experts. Not in the transformers source, and not confirmed offline:
the layer count, head counts, MLP width, the sliding/full order (the
default is 5 sliding : 1 full, last layer full), whether the 12B uses the
config's key=value, KV-sharing and double-wide-MLP options (all off by
default), and the logit softcap (default none; DiffusionGemma, built on
Gemma 4, hard-codes 30).

DEVICE: 24.5M parameters, ~12.7 MB at int4 (12.95 MB lowered).
- Width 3840 -> 256.
- 2 query heads sharing 1 key/value head on every layer, at the released
  head sizes: query width 512 on sliding layers, 1024 on full ones.
- Layers -> 18: three periods of 5 sliding + 1 full.
- MLP 1024 (4x the width, the ratio of the config defaults).
- Vocabulary 262,144 -> 8,192.
- Logit softcap 30 (see above).
- Kept as released: head sizes 256 and 512, window 1024, the two RoPE kinds,
  QK/V norms with score scale 1, gated GELU, tied embeddings; key=value,
  KV sharing and double-wide MLPs left off as the config defaults have them.

Other ways to fit it:
- Width 384 with 2 query heads, one period of 6 layers: 17.9M, ~9.2 MB; a
  second period would be 32.7M, over the region.
- 4 query heads per key/value head at width 256, 12 layers: 20.7M,
  ~10.7 MB; twice the attention, a third less depth.
- Keys as values on the full layers (attention_k_eq_v, the 26B-A4B's choice)
  and an MLP of 768 (3x): four periods, 24 layers, 26.8M, ~13.8 MB.
"""

MODEL_TYPE = "gemma4_unified_text"
MODEL_CLASS = "Gemma4UnifiedForCausalLM"

# Random weights leave q_norm/k_norm at scale 1 and the score scale is 1, so
# the scores are ~N(0, head size): near one-hot softmaxes, and the device
# arithmetic itself is that touchy (a 1e-4 nudge of the embedding table moves
# the int4/int8 reference to cosine 0.933). The delegate's int8 Q and K rows
# put ~1% error on every score. Measured at device size: cosine 0.965 against
# the device arithmetic, 0.9985 with the q@kT left in fp32. CI's toy heads clear
# the gate (four seeds: 0.9991-0.9996).
EXPECT = {"device": "host: int8 q@kT moves near one-hot QK-normed attention "
                    "(cosine 0.965; 0.9985 with q@kT in fp32)"}

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                       "rope_theta": 1_000_000.0},
}


def _pattern(periods):
    """5 sliding-window layers then 1 full-attention layer, `periods` times."""
    return (["sliding_attention"] * 5 + ["full_attention"]) * periods


DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=18,
    num_attention_heads=2, num_key_value_heads=1, head_dim=256,
    # Builder-only kwarg: the head size of the full-attention layers.
    global_head_dim=512,
    layer_types=_pattern(3), sliding_window=1024, rope_parameters=ROPE,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    max_position_embeddings=262144, rms_norm_eps=1e-6, tie_word_embeddings=True,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
          head_dim=32, global_head_dim=64, layer_types=["sliding_attention", "full_attention"],
          sliding_window=4)
