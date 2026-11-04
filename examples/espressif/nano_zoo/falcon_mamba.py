# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoFalconMamba: Falcon Mamba's normalized scan at ESP32-S3 size.

Released (tiiuae/falcon-mamba-7b, about 7B parameters): width 4096, 64 layers,
2x mixer expansion, 16 state values per channel and a four-tap depthwise
convolution. Falcon adds RMS normalization of the input-dependent B, C and
time-step vectors to Mamba-1's selective scan. There is no attention head
split or separate MLP; gates use SiLU and blocks use RMSNorm. The released
vocabulary is 65,024. transformers 5.19.0 configuration_falcon_mamba.py
names this checkpoint but defaults to width 768, 32 layers and 50,280 tokens;
those defaults are not its released shape. This offline source does not
confirm release embedding tying; this fixture explicitly uses untied tables.

DEVICE: 25,576,320 parameters, ~13.19 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width 4096 -> 384: group-128 projections fit the small flash region.
- Layers 64 -> 20: retain depth; the measured program and weights total
  15.12 MB, including the scan's fp32 constants.
- Vocabulary 65,024 -> 8,192: shrink both input and output tables.
- Automatic time-step rank 256 -> 24 follows width/16.
- Untied embeddings are an explicit fixture choice; the local class default
  is tied, and the release value is not certified by that source.
- Kept as released: selective scan, state size 16, expansion 2, four-tap
  biased convolution, SiLU, block RMSNorm and B/C/dt RMS normalization
  (epsilon 1e-6). CI uses one 64-wide layer and 256 tokens.

Other ways to fit it:
- Width 512 and 10 layers (~13.07 MB): more channels but half the recurrent depth.
- Width 256 and 48 layers (~13.00 MB): spend the budget on recurrent layers.
- Tie the two 8,192-token tables at width 384: save about 1.62 MB at int4
  without changing the scan, but force the input/output weights to agree.
"""

MODEL_TYPE = "falcon_mamba"
MODEL_CLASS = "FalconMambaForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, num_hidden_layers=20,
    state_size=16, expand=2, conv_kernel=4, time_step_rank="auto",
    hidden_act="silu", layer_norm_epsilon=1e-5, mixer_rms_eps=1e-6,
    use_bias=False, use_conv_bias=True, residual_in_fp32=True,
    tie_word_embeddings=False, use_mambapy=False, use_associative_scan=True,
    pad_token_id=0, bos_token_id=0, eos_token_id=0,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_hidden_layers=1)
