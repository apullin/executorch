# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoEmbeddingGemma2: EmbeddingGemma 2's text encoder at ESP32-S3 size.

Released (google/embeddinggemma-2; the shape below is transformers'
EmbeddingGemma2TextConfig defaults, documented against that checkpoint and
not compared with its config.json, which is not downloaded): a sentence-embedding
encoder, not a chatbot -- every token attends both ways, there is no output
head, and the model returns one 768-value vector per token (averaging them
gives the sentence's embedding). 24 layers at width 512; 4 query heads sharing
2 key/value heads (2:1) with head size 256 on sliding-window layers (a window
of 512 tokens each side), and head size 512 with 1 key/value head on the
full-attention layers, 5 sliding : 1 full; queries and keys RMS-normalised per
head, values too, score scale 1; RoPE at 10K/1M; gated GELU MLP of 2048 (4x);
four RMSNorms per layer. Per-layer embeddings, projection only: 512 values
per layer projected from the token embedding (no second table), gated into
the stream after every layer. Vocabulary 262,144.

DEVICE: 28.2M parameters, ~14.5 MB at int4 (14.82 MB lowered).
- Width 512 -> 256, which leaves 2 query heads sharing 1 key/value head
  (the released 2:1) on sliding layers.
- Layers 24 -> 18: three periods of 5 sliding + 1 full.
- MLP 1024 (4x the width, as released).
- PLE 512 -> 256 values per layer (the width, as released).
- Output vector 768 -> 384 (1.5x the width, as released).
- Vocabulary 262,144 -> 8,192.
- Kept as released: head sizes 256 and 512, 1 key/value head on full layers,
  bidirectional attention with a 512-token window, both RoPE wavelengths,
  QK/V norms with score scale 1, gated GELU, projection-only PLE.

Other ways to fit it:
- The released width 512 with its 4 query / 2 key-value heads: one period of
  6 layers is already 38.7M, ~20 MB, over the region.
- All 24 released layers at width 256: 36.8M, ~19 MB; with the MLP at 2x and
  the PLE at 128 per layer they fit, 25.1M, ~12.9 MB.
- Width 256 with the MLP and PLE as here but 12 layers (two periods):
  about 19.5M, ~10 MB; shallower, with room left over.
"""

MODEL_TYPE = "embedding_gemma2_text"
MODEL_CLASS = "EmbeddingGemma2TextModel"
# An encoder: no logits, and no KV cache to switch off.
OUTPUT = "last_hidden_state"
FORWARD_KWARGS = {}

# Random weights leave q_norm/k_norm at scale 1 and the score scale is 1, so
# the scores are ~N(0, head size): near one-hot softmaxes, and the device
# arithmetic itself is that touchy (a 1e-4 nudge of the embedding table moves
# the int4/int8 reference to cosine 0.894). The delegate's int8 Q and K rows
# put ~1% error on every score. Measured at device size: cosine 0.863 against
# the device arithmetic, 0.935 with the q@kT left in fp32 -- and 0.9999 with
# the RMSNorms also unfused, which differ from the fused ones only in rounding
# (the fused edge graph is exact in Python): at 18 layers a few flipped int8
# activation codes are enough. CI's toy heads clear the gate (four seeds:
# 0.9982-0.9991).
EXPECT = {"device": "host: int8 q@kT moves near one-hot QK-normed attention "
                    "(cosine 0.863; 0.935 with q@kT in fp32)"}

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
}


def _pattern(periods):
    """5 sliding-window layers then 1 full-attention layer, `periods` times."""
    return (["sliding_attention"] * 5 + ["full_attention"]) * periods


DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=1024, num_hidden_layers=18,
    num_attention_heads=2, num_key_value_heads=1, head_dim=256,
    # Builder-only kwargs: the full-attention layers' head size and KV heads.
    global_head_dim=512, num_global_key_value_heads=1,
    layer_types=_pattern(3), sliding_window=512, rope_parameters=ROPE,
    hidden_size_per_layer_input=256, embedding_dim=384,
    hidden_activation="gelu_pytorch_tanh", max_position_embeddings=262144, rms_norm_eps=1e-6,
)

# A window of 4 each side, so the band mask cuts into the 8-token prompt.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
          head_dim=32, global_head_dim=64, layer_types=["sliding_attention", "full_attention"],
          sliding_window=4, hidden_size_per_layer_input=64, embedding_dim=96)
