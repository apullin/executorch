# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoRWKV: RWKV-4 time/channel mixing at ESP32-S3 size, random weights.

Released (RWKV/rwkv-4-169m-pile, about 169M parameters): width 768, 12 layers,
time-mixing width 768, squared-ReLU channel MLP width 3072 (4x), vocabulary
50,277, context length 1024 and untied input/output embeddings. There are
no attention heads or KV groups: learned time-decay and time-first vectors
drive a numerically stable recurrent weighted sum, and sigmoid receptance
gates control both the time and channel branches. Blocks use LayerNorm.
transformers 5.19.0 configuration_rwkv.py names the 169M checkpoint but its
width 4096 / 32-layer defaults describe a larger scale, not that checkpoint.

DEVICE: 25,504,512 parameters, ~13.15 MB estimated at int4 (33/64 byte per
parameter); counted on the meta device before construction.
- Width and time-mixing width 768 -> 384: shrink dense matrices in multiples
  of 128 while preserving a full-width recurrent branch.
- Layers 12 -> 10: leave room for both untied vocabulary tables.
- Channel MLP 3072 -> 1536: retain the released 4x ratio and squared ReLU.
- Vocabulary 50,277 -> 8,192: shrink both tables for a deployment tokenizer.
- rescale_every 6 -> 0: disable the fp16 overflow trick that otherwise
  mutates output weights during the first eval forward and confuses export.
- Kept as released: recurrent time decay/time first, sigmoid receptance,
  time/channel mixing, LayerNorm, context 1024 and untied embeddings. CI has
  two 64-wide layers and 256 tokens; the initialization divides by layers-1,
  so a one-layer random fixture is not valid.

Other ways to fit it:
- Width 256 and 24 layers (~12.74 MB): more time/channel recurrence with fewer
  channels.
- Width 512 and 5 layers (~13.13 MB): wider channels and a much shallower stack.
- Tie the tables at width 384: save about 1.62 MB at int4, but change the
  released independent embedding/output weights.
"""

MODEL_TYPE = "rwkv"
MODEL_CLASS = "RwkvForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=384, attention_hidden_size=384,
    intermediate_size=1536, num_hidden_layers=10, context_length=1024,
    layer_norm_epsilon=1e-5, rescale_every=0, tie_word_embeddings=False,
    bos_token_id=0, eos_token_id=0,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, attention_hidden_size=64,
          intermediate_size=256, num_hidden_layers=2)
