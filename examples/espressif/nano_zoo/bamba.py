# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoBamba: Mamba-2 layers with occasional attention at ESP32-S3 size.

Released (ibm-fms/Bamba-9.8b-2.2T-hf, about 9.8B parameters): width 4096,
32 layers, 32 query / 8 KV heads (4:1), attention head size 128, SwiGLU
width 14,336 (3.5x), vocabulary 128,000 and untied embeddings. Mamba-2
layers use 128 heads of size 64, expansion 2, state size 256, one B/C group,
a four-tap convolution and chunked SSD. These are the fields in transformers
5.19.0 configuration_bamba.py. The checkpoint places attention at layers
9, 18 and 27 (zero based): nine initial SSM layers, then attention every
nine layers, with eight SSM layers between attention layers. The local class
leaves attn_layer_indices unspecified rather than copying this released list;
this fixture writes out the retained prefix explicitly. The released RoPE
theta is not verified in this offline source; this fixture uses its 10,000
default and keeps half-head rotation.

DEVICE: 26,650,264 parameters, ~13.74 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width 4096 -> 256: group-128 projections dominate the size reduction.
- Layers 32 -> 19: stop after the second attention layer; retain both initial
  attention-containing intervals, with attention at layers 9 and 18.
- Attention heads 32/8 -> 4/1 and head size 128 -> 64: preserve 4:1 sharing;
  retaining size 128 at 4:1 would require width 512 and exceed this budget.
- MLP 14,336 -> 896: keep the 3.5x-width SwiGLU ratio.
- SSM heads 128 -> 8: retain 64-channel heads and expansion 2.
- Vocabulary 128,000 -> 8,192: shrink the two untied tables.
- SSD chunk size 256 -> 4: exercise two chunks without large padded scratch.
- Kept as released: state size 256, one B/C group, four-tap biased conv,
  SiLU, RMSNorm, half-head RoPE, bias-free projections and untied embeddings.
  CI has one SSM and one attention layer at width 64; its SSM state is 16
  and both head sizes are 16 to keep the fixture below one million parameters.

Other ways to fit it:
- Width 256 and 20 layers, attention at 9/18 (~14.37 MB): keep two attention
  layers, with less flash slack before program overhead.
- Width 128 and 40 layers, attention at 9/18/27/36 (~7.85 MB): more periods,
  but heads shrink to size 32 to preserve 4:1 sharing.
- Tie the tables at width 256: save about 1.08 MB at int4 while changing the
  released independent input/output embeddings.
"""

MODEL_TYPE = "bamba"
MODEL_CLASS = "BambaForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=896,
    num_hidden_layers=19, num_attention_heads=4, num_key_value_heads=1,
    attn_layer_indices=[9, 18], mamba_n_heads=8, mamba_d_head=64,
    mamba_n_groups=1, mamba_d_state=256, mamba_d_conv=4, mamba_expand=2,
    mamba_chunk_size=4, mamba_conv_bias=True, mamba_proj_bias=False,
    hidden_act="silu", rms_norm_eps=1e-5, tie_word_embeddings=False,
    attention_bias=False, mlp_bias=False, max_position_embeddings=262144,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.5},
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=224,
          num_hidden_layers=2, attn_layer_indices=[1], mamba_n_heads=8,
          mamba_d_head=16, mamba_d_state=16)
