# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoQwen3Next: gated DeltaNet, gated attention and MoE at board size.

Released (Qwen/Qwen3-Next-80B-A3B-Instruct): about 80B total parameters,
about 3B active per token, width 2048 and 48 layers. Three gated DeltaNet
linear-attention layers then one gated full-attention layer form the period.
Full attention has 16 query / 2 KV heads (8:1), head size 256 and a separate
output gate; the query width is 2x the residual width. DeltaNet has sixteen
128-channel key heads and thirty-two 128-channel value heads (2:1), a
four-tap convolution and a gated normalized recurrent update. Every layer
routes to ten of 512 SwiGLU experts, each width 512 (width/4), and adds a
gated shared expert of width 512. The dense-only MLP field is 5632 (2.75x),
but there are no dense-only layers in this layout. Vocabulary is 151,936;
embeddings are untied, norms use 1+weight RMSNorm and RoPE covers a quarter
of each full-attention head. All these fields are in transformers 5.19.0
configuration_qwen3_next.py; exact released RoPE theta is not certified by
that source, so this fixture explicitly uses the class default 10,000.

DEVICE: 28,083,788 parameters, ~14.48 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction. These are weight
estimates; the emitted program is also measured, including any fp32 experts
that the backend cannot yet quantize.
- Width 2048 -> 128: keep all 512 experts and top-10 routing inside the
  estimated weight budget instead of removing the MoE mechanism.
- Layers 48 -> 4: retain one complete three-linear/one-full period.
- Full query/KV heads 16/2 -> 8/1, head size 256 -> 32: retain 8:1 sharing,
  2x query width and the separate gate; keeping head size 256 cannot fit.
- Linear key/value heads 16/32 -> 1/2: retain head size 128, 2:1 sharing,
  full-width keys and double-width values.
- Routed and shared MLP width 512 -> 32: retain width/4; dense-only width
  5632 -> 352 retains its unused 2.75x field for an explicit config.
- Vocabulary 151,936 -> 8,192: shrink both untied tables.
- Kept as released: the four-layer period, all 512 experts/top-10, shared
  expert gating, four-tap conv, DeltaNet, gated full attention, RMSNorm,
  quarter-head RoPE and untied embeddings. CI retains the four-layer period
  at width 64 with 16 experts/top-10, head sizes 16 and vocabulary 256;
  the expert bank must shrink to keep CI below one million parameters.
- Known backend gap: indexing the 3-D expert banks before batched matmul
  leaves their 25,165,824 weights in fp32 (~100.66 MB). The full program
  and weights emit 103.05 MB, beyond the 15.27 MB model region, although
  host cosine vs device arithmetic is 0.998533. CI fits at 1.72 MB;
  EXPECT applies only to DEVICE.

Other ways to fit it:
- Width 256, four layers and 64 experts, top-10 (~9.83 MB): wider residuals,
  fewer specialist weights, still a shared expert and both mixers.
- Width 128, four layers and 256 experts (~7.93 MB): halve expert storage
  and leave more flash slack while preserving top-10 and both mixers.
- Width 128, four layers and 1024-token vocabulary: save about 0.95 MB at
  int4; the much smaller tokenizer changes the deployment's token lengths.
"""


MODEL_TYPE = "qwen3_next"
MODEL_CLASS = "Qwen3NextForCausalLM"

LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]

DEVICE = dict(
    vocab_size=8192, hidden_size=128, intermediate_size=352,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=1,
    head_dim=32, layer_types=LAYER_TYPES,
    linear_conv_kernel_dim=4, linear_key_head_dim=128,
    linear_value_head_dim=128, linear_num_key_heads=1, linear_num_value_heads=2,
    moe_intermediate_size=32, shared_expert_intermediate_size=32,
    num_experts=512, num_experts_per_tok=10, norm_topk_prob=True,
    decoder_sparse_step=1, mlp_only_layers=[], hidden_act="silu",
    rms_norm_eps=1e-6, tie_word_embeddings=False, attention_bias=False,
    max_position_embeddings=32768,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.25},
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=176,
          head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
          linear_num_key_heads=4, linear_num_value_heads=8,
          moe_intermediate_size=16, shared_expert_intermediate_size=16,
          num_experts=16)

EXPECT = {"device": "flash: indexed 3-D expert banks stay fp32; DEVICE emits 103.05 MB"}
