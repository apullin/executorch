# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoT5Gemma2: T5Gemma 2 270M-270M's encoder-decoder at ESP32-S3 size.

Released (google/t5gemma-2-270m-270m, the checkpoint transformers documents
T5Gemma 2 against): an encoder-decoder (T5-style) made from two Gemma 3 270M
decoders. The encoder reads the whole input with attention both ways; the
decoder writes the output with "merged" attention -- one softmax over its own
earlier tokens and all encoder tokens, through one set of query/key/value
weights, instead of separate self- and cross-attention. The token embedding is
shared by encoder, decoder and output head. Each stack, as Gemma 3 270M
(unverified; transformers' defaults describe a 2B-class model): 18 layers at
width 640, 4 query heads sharing 1 key/value head, head size 256, scores
scaled by 1/16 (query_pre_attn_scalar 256), queries and keys RMS-normalised
per head; gated GELU MLP of 2048 (3.2x); 5 sliding-window (512) : 1 full
layers, RoPE at 10K/1M; vocabulary 262,144 (transformers' default 262,208).
Images enter the encoder through a SigLIP vision encoder, 256 tokens per image.

DEVICE: 17.1M parameters, ~8.8 MB at int4 (8.98 MB lowered): under the
zoo's usual 10-15 MB, because a second period per stack at 4:1 would be 32M.
- Width 640 -> 256, at the released 4:1 heads of size 256 (query width
  1024, 4x the width; 1.6x as released).
- Layers 18 -> 6 per stack: one period of 5 sliding + 1 full each.
- MLP 2048 -> 768 (3x the width; 3.2x released, rounded to whole groups
  of 128).
- Vocabulary 262,144 -> 8,192.
- SigLIP -> one layer, 32 wide, 4 tokens per image: T5Gemma 2 always builds
  its vision tower; it never runs on a text prompt and is not in the
  exported program.
- Kept as released: merged decoder attention, bidirectional encoder, the
  shared embedding, head size 256 and 4:1 sharing, the 1/16 score scale, QK
  norms, window 512, both RoPE wavelengths, gated GELU.

Other ways to fit it:
- 2 query heads per key/value head (query width 512) frees room for two
  periods per stack: 12 + 12 layers, 25.8M, ~13.3 MB; 2:1 instead of 4:1.
- An unbalanced 12-layer encoder and 6-layer decoder at 4:1: 24.6M,
  ~12.7 MB; T5Gemma (the first one) shipped unbalanced pairs, version 2 did
  not.
- MLP 1024 (4x) at 6 + 6 layers: 19.5M, ~10.0 MB; a wider MLP than released.
"""

MODEL_TYPE = "t5gemma2"
MODEL_CLASS = "T5Gemma2ForConditionalGeneration"

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
}


def _pattern(periods):
    """5 sliding-window layers then 1 full-attention layer, `periods` times."""
    return (["sliding_attention"] * 5 + ["full_attention"]) * periods


def INPUTS(cfg):
    """The same 8 ids into the encoder and, as the output so far, the decoder."""
    from executorch.examples.espressif.nano_zoo import build

    source = build.ids(cfg)["input_ids"]
    return {"input_ids": source, "decoder_input_ids": source.clone()}


TEXT = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=768, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=1, head_dim=256, query_pre_attn_scalar=256,
    layer_types=_pattern(1), sliding_window=512, rope_parameters=ROPE,
    hidden_activation="gelu_pytorch_tanh", max_position_embeddings=131072, rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)

VISION = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
              image_size=32, patch_size=16)


def _composite(text):
    """Image token ids inside the vocabulary, clear of the prompt's ids 1..8."""
    vocab = text["vocab_size"]
    return dict(encoder=dict(text_config=text, vision_config=VISION, mm_tokens_per_image=4,
                             boi_token_index=vocab - 4, eoi_token_index=vocab - 3),
                decoder=dict(text), image_token_index=vocab - 2)


DEVICE = _composite(TEXT)

CI = _composite(dict(TEXT, vocab_size=256, hidden_size=64, intermediate_size=192,
                     num_hidden_layers=2, head_dim=16, query_pre_attn_scalar=16,
                     layer_types=["sliding_attention", "full_attention"], sliding_window=4))
