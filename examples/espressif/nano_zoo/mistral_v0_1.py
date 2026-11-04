# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoMistral-v0.1: Mistral 7B v0.1's layout at ESP32-S3 size, random weights.

Released (mistralai/Mistral-7B-v0.1, 7.2B parameters): 32 layers at width
4096; 32 query heads sharing 8 key/value heads (4:1), head size 128; SwiGLU
MLP of 14336 (3.5x width); every layer attends through a 4,096-token sliding
window, the family's mark (v0.2 and v0.3 turned it off; Ministral 8B
interleaves it with full attention and has its own class); RoPE theta 10,000;
RMSNorm eps 1e-5; vocabulary 32,000; separate input embedding and output head.

DEVICE: 24.5M parameters, ~12.6 MB at int4.
- Width 4096 -> 256. The heads go to 4/1, the fewest that keep the 4:1
  sharing; at head size 128 that makes attention 512 wide, twice the width
  (MistralConfig takes head_dim apart from the width, as Mistral NeMo does).
- MLP 14336 -> 896, still 3.5x the width.
- Layers 32 -> 20.
- Vocabulary 32,000 -> 8,192, as a deployment would retrain its tokenizer
  (the story model's is 8,192).
- Kept as released: the sliding window on every layer and its 4,096 length,
  head size 128 and 4:1 sharing, the 3.5x SwiGLU MLP, untied embeddings, RoPE
  theta 10k, RMSNorm eps 1e-5.

Other ways to fit it:
- Width 512, attention as wide as the model (4 query / 1 key-value heads),
  MLP 1792, 5 layers (13.1 MB): the released proportions, a sixth of the
  depth.
- Keep the 32,000 vocabulary: its two tables are 16.4M parameters at width
  256, which leaves 8 layers (12.6 MB).
- Head size 64 at width 256 (attention as wide as the model), 24 layers
  (12.7 MB): nanoLlama's heads with Mistral's window and MLP.
"""

MODEL_TYPE = "mistral"
MODEL_CLASS = "MistralForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=896, num_hidden_layers=20,
    num_attention_heads=4, num_key_value_heads=1, head_dim=128,
    sliding_window=4096, max_position_embeddings=32768, rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
)

# A window shorter than the 8-token prompt, so the sliding mask differs from
# the causal one and the comparison can see it.
CI = dict(DEVICE, vocab_size=256, hidden_size=64, intermediate_size=224,
          num_hidden_layers=2, head_dim=32, sliding_window=4)
