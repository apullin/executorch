# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoRecurrentGemma: Griffin RG-LRU plus local attention at board size.

Released (google/recurrentgemma-2b, about 2B parameters): width 2560,
26 layers, RG-LRU width 2560, ten query heads of size 256, MLP config width
7680 (3x), vocabulary 256,000 and tied embeddings. The implementation halves
that MLP field: its two GELU-gated branches each have width 3840 (1.5x).
Two recurrent blocks then
one local-attention block form the repeated three-layer pattern. Recurrent
blocks combine a four-tap convolution, learned block-diagonal gates and a
real-gated linear recurrent unit (RG-LRU). Attention has a 2048-token window
and half-head RoPE; GELU gates the MLP, RMSNorm uses 1+weight, embeddings
scale by sqrt(width), and logits soft-cap at 30. These shapes/mechanisms
come from transformers 5.19.0 configuration/modeling_recurrent_gemma.py.
Its num_key_value_heads default resolves to ten, whereas the released 2B
uses one shared KV head (10:1); the fixture sets that explicitly.

DEVICE: 23,400,320 parameters, ~12.07 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width and RG-LRU width 2560 -> 640: keep ten-way recurrent gate blocks
  and group-128 projections without disabling the recurrent branch.
- Layers 26 -> 6: retain two complete recurrent/recurrent/attention periods.
- Attention head size 256 -> 64: preserving size 256 and 10:1 sharing would
  force width 2560, which cannot fit even one complete period.
- MLP field 7680 -> 1920, actual branches 3840 -> 960: retain the released
  1.5x-width GELU-gated branches and their 3x combined config field.
- Vocabulary 256,000 -> 8,192: shrink the tied deployment tokenizer table.
- Kept as released: ten query / one KV head, the three-block period,
  four-tap convolution, RG-LRU recurrence and gates, 2048-token window,
  half-head RoPE, 1+weight RMSNorm, embedding scaling, soft-cap and tying.
  CI keeps all three blocks at width 64, using four query / one KV
  head (size 16); ten gate blocks cannot evenly divide width 64. It uses one
  three-layer period rather than DEVICE's two.

Other ways to fit it:
- Width 512, two query / one KV head of size 256, and six layers (~8.81 MB):
  keep the released head size but change sharing to 2:1 and gate partitioning.
- Width 640 with a 4096-token vocabulary and six layers (~10.71 MB): save
  about 1.35 MB at int4 for runtime context or program overhead.
- Width 1280 and three layers keeps ten 128-channel query heads, but its
  46.7M parameters (~24.10 MB) exceed both flash and the zoo's real-build cap.
"""

MODEL_TYPE = "recurrent_gemma"
MODEL_CLASS = "RecurrentGemmaForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=640, lru_width=640, intermediate_size=1920,
    num_hidden_layers=6, num_attention_heads=10, num_key_value_heads=1,
    block_types=["recurrent", "recurrent", "attention"],
    attention_window_size=2048, conv1d_width=4, logits_soft_cap=30.0,
    hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6,
    tie_word_embeddings=True, attention_bias=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.5},
    pad_token_id=0, bos_token_id=2, eos_token_id=1,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, lru_width=64,
          intermediate_size=192, num_hidden_layers=3, num_attention_heads=4)
