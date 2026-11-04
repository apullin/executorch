# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoFalconH1: parallel attention and Mamba-2 at ESP32-S3 size.

Released (tiiuae/Falcon-H1-0.5B-Base, about 0.5B parameters): every layer
runs attention and an SSM branch in parallel, adds their outputs, then uses
a SwiGLU MLP. The 0.5B checkpoint's exact width, depth, head counts, MLP,
vocabulary and scaling multipliers are not confirmed in this offline source.
transformers 5.19.0 configuration_falcon_h1.py instead names the 1.5B-Deep
variant and defaults to width 4096, 32 layers, 32 query / 8 KV heads, head
size 128, MLP 14,336, vocabulary 128,000 and untied embeddings. Its SSM
defaults are width 1024, 128 heads of size 8, state size 256, one B/C group,
four-tap convolution, SiLU and chunked SSD. Those explicit defaults supply
the ratios below; they are not claimed as the 0.5B checkpoint's real shape.

DEVICE: 23,905,472 parameters, ~12.33 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width 4096 (class default) -> 512: fit group-128 projections while keeping
  the source's 128-channel attention heads and 4:1 query/KV sharing.
- Layers 32 (class default) -> 4: budget the wider parallel branches and MLP;
  every layer still contains both attention and the SSM.
- Query/KV heads 32/8 -> 4/1: retain head size 128 and 4:1 sharing.
- MLP 14,336 -> 1792: keep the source default's 3.5x-width SwiGLU ratio.
- SSM width 1024 -> 128, heads 128 -> 16: preserve the source's quarter-width
  independent SSM branch and its eight-channel heads.
- Vocabulary 128,000 (class default) -> 8,192: shrink the untied tables.
- Chunk size 256 -> 4: exercise cross-chunk SSD with bounded prompt scratch.
- Kept: both mixers in every layer, state size 256, one B/C group, four-tap
  biased convolution, SiLU SSM output gating, 128-channel attention heads,
  block RMSNorm, bias-free
  projections, untied embeddings and explicit branch scaling multipliers.
  CI has one 64-wide hybrid layer, a 16-wide SSM and state size 16.

Other ways to fit it:
- Width 256 and 20 layers (~12.86 MB): more recurrent depth, but attention
  head size falls to 64 to keep the 4:1 sharing ratio.
- Width 128 and 48 layers (~8.37 MB): keep both branches while shrinking
  attention head size to 32; recurrence gets more of the budget.
- Tie the tables at width 512: save about 2.16 MB at int4, giving up separate
  output weights. A released 0.5B config could replace the disclosed source
  defaults once its exact ratios and multipliers are available locally.
"""

MODEL_TYPE = "falcon_h1"
MODEL_CLASS = "FalconH1ForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=512, intermediate_size=1792,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    mamba_d_ssm=128, mamba_n_heads=16, mamba_d_head=8, mamba_n_groups=1,
    mamba_d_state=256, mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=4,
    mamba_conv_bias=True, mamba_proj_bias=False, mamba_norm_before_gate=True,
    mamba_rms_norm=False, projectors_bias=False,
    hidden_act="silu", rms_norm_eps=1e-5, tie_word_embeddings=False,
    attention_bias=False, mlp_bias=False, max_position_embeddings=8192,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    lm_head_multiplier=1.0, embedding_multiplier=1.0,
    mlp_multipliers=[1.0, 1.0], key_multiplier=1.0,
    attention_out_multiplier=1.0, attention_in_multiplier=1.0,
    ssm_multipliers=[1.0, 1.0, 1.0, 1.0, 1.0],
    ssm_in_multiplier=1.0, ssm_out_multiplier=1.0,
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=224,
          num_hidden_layers=1, mamba_d_ssm=16, mamba_n_heads=2,
          mamba_d_state=16)
