# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoGemma3n: Gemma 3n E4B's layout at ESP32-S3 size, random weights.

Released (google/gemma-3n-E4B; transformers' Gemma3nTextConfig defaults are
"E4B style", and this is its text tower, Gemma3nForCausalLM, without the
vision and audio encoders): 35 layers at width 2048; 8 query heads sharing 2
key/value heads (4:1), head size 256, queries and keys RMS-normalised per head,
values too, score scale 1; gated GELU MLP of 16,384 (8x) on every layer;
4 sliding-window layers (window 512) then 1 full-attention layer, RoPE at
10K/1M; vocabulary 262,400 tied to the output head, logits softcapped at 30.
On top of a plain decoder:
- AltUp: the residual stream is 4 copies of the width; each layer runs on one
  and predicts/corrects the other three with tiny learned mixes.
- LAuReL: a rank-64 linear shortcut beside every attention block.
- Per-layer embeddings (PLE): 256 values per layer per token from a
  262,144-token table, gated into the stream after every layer.
- KV sharing: the last 15 layers reuse the keys/values of the last earlier
  layer of their type.
- Activation sparsity: the first 10 layers keep only the MLP activations more
  than 1.645 standard deviations above their mean (the top 5% of a Gaussian).
Raw ~8B parameters, "effective 4B" (the PLE table can live outside fast
memory, and E2B is nested inside it).

DEVICE: 27.1M parameters, ~14.0 MB at int4 (it stops at lowering: see EXPECT).
- Width 2048 -> 256.
- 4 query heads sharing 1 key/value head: the released 4:1 and head size 256
  (query width 1024, 4x the width; 1x as released).
- Layers 35 -> 10: two periods of 4 sliding + 1 full, the second sharing the
  first one's keys/values (released: 3 of 7 periods share).
- Activation sparsity on the first 3 layers (released: the first 10 of 35).
- PLE 256 -> 32 values per layer (1/8 of the width, as released), from an
  8,192-token table.
- LAuReL rank 64 -> 8 (1/32 of the width, as released).
- Vocabulary 262,400 -> 8,192, for the main and the per-layer tables.
- Kept as released: head size 256 and 4:1 sharing, the 8x gated GELU MLP,
  window 512, 4 AltUp streams, LAuReL, PLE, KV sharing, activation sparsity at
  0.95, softcap 30, tied embeddings.

Other ways to fit it:
- E2B's proportions (MLP 4x, a third of the layers sharing KV; unverified)
  at width 256: the narrower MLP buys a third period, 15 layers at 27.9M,
  ~14.4 MB.
- 2 query heads per key/value head (query width 512): 24.4M, ~12.6 MB; not
  enough for a third period (15 layers would be 35.7M).
- Width 384, 4 query heads, 5 layers (one period, so no KV sharing), PLE 48:
  29.0M, ~14.9 MB; wider but half the depth.
"""

MODEL_TYPE = "gemma3n_text"
MODEL_CLASS = "Gemma3nForCausalLM"

# The Normal(0, 1) inverse CDF behind activation sparsity exports, under torch
# 2.15, as its argument check (aten._is_all_true feeding _assert_async,
# torch/distributions/distribution.py:15-19) and aten.erfinv; the ExecuTorch
# edge verifier rejects both (neither is Core ATen), _is_all_true first, before
# any partitioner runs: an inverse CDF of a constant (0.95 -> 1.645), foldable
# at export time. With the sparsity off this DEVICE config lowers (114
# delegates, 441 portable ops, 14.28 MB) at cosine 0.975 against the device
# arithmetic, 0.999999 with the q@kT in fp32: scale-1 QK-normed attention, the
# int8 q@kT gap gemma4 shows.
EXPECT = ("lower: activation sparsity's Gaussian cutoff exports aten._is_all_true "
          "(torch 2.15's validation assert) and aten.erfinv, neither Core ATen")

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
}


def _pattern(periods):
    """4 sliding-window layers then 1 full-attention layer, `periods` times."""
    return (["sliding_attention"] * 4 + ["full_attention"]) * periods


DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=2048, num_hidden_layers=10,
    num_attention_heads=4, num_key_value_heads=1, head_dim=256,
    layer_types=_pattern(2), sliding_window=512, rope_parameters=ROPE,
    num_kv_shared_layers=5, activation_sparsity_pattern=[0.95] * 3 + [0.0] * 7,
    vocab_size_per_layer_input=8192, hidden_size_per_layer_input=32,
    altup_num_inputs=4, altup_active_idx=0, altup_coef_clip=120.0, altup_correct_scale=True,
    laurel_rank=8, hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    max_position_embeddings=32768, rms_norm_eps=1e-6, tie_word_embeddings=True,
)

# Own sliding (sparse MLP), own full, then a shared sliding and a shared full.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=512, num_hidden_layers=4,
          num_attention_heads=4, head_dim=16,
          layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=4,
          num_kv_shared_layers=2, activation_sparsity_pattern=[0.95, 0.0, 0.0, 0.0],
          vocab_size_per_layer_input=256, hidden_size_per_layer_input=8, laurel_rank=2)
