# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoMamba: Mamba-1 selective scan at ESP32-S3 size, random weights.

Released (state-spaces/mamba-2.8b, about 2.8B parameters): width 2560 and
64 layers; a 2x-width selective state-space mixer with 16 state values per
channel, a four-tap depthwise convolution, SiLU gates and RMSNorm. There is
no attention, no query/KV head split, and no separate MLP. The vocabulary is
50,280 and the input/output embeddings are tied. The checkpoint name and
mixer fields come from transformers 5.19.0 configuration_mamba.py; its
768-wide, 32-layer constructor defaults are not the 2.8B checkpoint's shape.

DEVICE: 24,359,040 parameters, ~12.56 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width 2560 -> 384: reduce the dense projections while retaining group-128
  weight rows; the mixer remains 2x width, or 768 channels.
- Layers 64 -> 22: leave room for the program and the scan's fp32 constants;
  a 24-layer package is 15.85 MB, beyond the 15.27 MB flash region.
- Vocabulary 50,280 -> 8,192: use a small deployment tokenizer.
- Automatic time-step rank 160 -> 24 follows width/16, as in the release.
- Kept as released: 16-state selective scan, four-tap biased depthwise conv,
  SiLU, RMSNorm, fp32 residuals, bias-free projections and tied embeddings.
  Export unrolls the scan over the eight prompt tokens; it does not replace
  the recurrence with attention. CI uses one 64-wide layer and 256 tokens.

Other ways to fit it:
- Width 512 and 12 layers (~12.65 MB): wider projections and a shallower
  recurrent stack.
- Width 256 and 48 layers (~11.92 MB): more recurrence depth with fewer channels.
- Keep 50,280 tokens at width 256: the table alone costs about 6.6 MB at
  int4, leaving substantially less flash for the recurrent stack.
"""

MODEL_TYPE = "mamba"
MODEL_CLASS = "MambaForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, num_hidden_layers=22,
    state_size=16, expand=2, conv_kernel=4, time_step_rank="auto",
    hidden_act="silu", layer_norm_epsilon=1e-5,
    use_bias=False, use_conv_bias=True, residual_in_fp32=True,
    tie_word_embeddings=True, use_mambapy=False, use_associative_scan=True,
    pad_token_id=0, bos_token_id=0, eos_token_id=0,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, num_hidden_layers=1)
