# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoLlama4: Llama 4 Scout's text layout at ESP32-S3 size, random weights.

Released (meta-llama/Llama-4-Scout-17B-16E, 109B parameters with its vision
tower, 17B used per token): 48 text layers at width 5120; 40 query heads
sharing 8 key/value heads (5:1), head size 128. Every fourth layer is a
"NoPE" layer: no rotary embedding, attention over the whole context, and
queries scaled up with position (temperature tuning). The other three use
RoPE, attend only within 8,192-token chunks, and normalize queries and keys
by their RMS without weights (the code calls it an L2 norm). Every layer's
MLP is a mixture of 16 SwiGLU experts of 8,192 (1.6x width), 1 chosen per
token by a sigmoid router, plus a shared expert of the same size that every
token uses. RMSNorm; RoPE theta 500,000
with llama3-style frequency scaling (x16 over an 8K original window,
unverified: transformers' source does not carry it); a
10M-token context; vocabulary 202,048; untied embeddings. (Llama 4 Maverick
instead alternates dense and MoE layers, with 128 experts; Scout has MoE in
every layer.)

DEVICE: 25.8M parameters, ~13.3 MB at int4.
- The text model alone: Llama4ForCausalLM is the same text architecture
  without the vision tower.
- Width 5120 -> 256: 5 query heads of 128 share 1 key/value head, so the
  query width is 2.5x the model width (released: 1x).
- Experts and the shared expert 8,192 -> 384 wide: 1.5x instead of 1.6x, a
  multiple of 128.
- Layers 48 -> 4: one period, three chunked RoPE layers then a NoPE layer.
  The 16 experts and the shared one cost 5.0M parameters per layer at width
  256.
- Vocabulary 202,048 -> 8,192, as a deployment would retrain its tokenizer.
- Kept as released: 16 experts with 1 per token plus the shared expert, the
  sigmoid router, MoE in every layer, NoPE every fourth layer with its
  temperature tuning, 8,192-token chunks, the weightless QK norm, head size
  128 and the 5:1 head sharing, the RoPE scaling, untied embeddings.

Other ways to fit it:
- 8 experts plus the shared one, 8 layers (14.8 MB): two NoPE periods at half
  the expert count.
- Width 128 with 192-wide experts, 16 layers (13.1 MB): four NoPE periods,
  but the query width becomes 5x the model width.
- Maverick's interleave (dense and MoE layers alternating) over two NoPE
  periods, 8 layers: 15.4 MB, just over the region. Maverick's 128 experts
  would not fit even once (37.7M parameters per layer at width 256).

On the board: not yet. Llama 4 computes RoPE in complex numbers (torch.polar,
view_as_complex, a complex multiply), and ExecuTorch's edge dialect has no
complex ops: lowering stops at aten.polar before the partitioner runs. Past
that, the experts would meet the gap the other MoE nanos show: Llama 4 runs
every expert on every token (a bmm over the 3-D weights, the router's scores
zeroing the rest), and nothing delegates a 3-D weight (18.9M of the 25.8M
parameters).
"""

MODEL_TYPE = "llama4_text"
MODEL_CLASS = "Llama4ForCausalLM"
EXPECT = ("lower: RoPE in complex numbers; to_edge refuses aten.polar (not in the Core "
          "ATen opset) and no pass rewrites it into real cos/sin arithmetic")

ROPE = {"rope_type": "llama3", "rope_theta": 500_000.0, "factor": 16.0,
        "low_freq_factor": 1.0, "high_freq_factor": 1.0,
        "original_max_position_embeddings": 8192}

DEVICE = dict(
    vocab_size=8192, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=5, num_key_value_heads=1, head_dim=128,
    intermediate_size=384, intermediate_size_mlp=768,  # the latter: dense layers only, none here
    num_local_experts=16, num_experts_per_tok=1, interleave_moe_layer_step=1,
    no_rope_layer_interval=4, attention_chunk_size=8192, use_qk_norm=True,
    attn_temperature_tuning=True, floor_scale=8192, attn_scale=0.1,
    max_position_embeddings=10_485_760, rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters=ROPE,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, head_dim=16, intermediate_size=64,
          intermediate_size_mlp=192)
