# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoDiffusionGemma: DiffusionGemma's block diffusion at ESP32-S3 size.

Released (google/diffusiongemma-26B-A4B-it, the checkpoint transformers
documents DiffusionGemma against): a text generator that writes a block of
tokens at a time instead of one. An encoder (causal, like a normal LM) reads
the prompt and leaves its keys/values in a cache; a decoder then refines a
"canvas" of 256 tokens over several denoising steps, attending both ways over
the cached prompt and the canvas. Encoder and decoder share every weight; the
decoder adds a self-conditioning gated MLP that feeds the previous step's
predictions back in. The layers are Gemma 4 26B-A4B's: every layer runs a
dense gated GELU MLP and a mixture of experts side by side (the router picks a
few experts per token and weights them); sliding-window layers attend with
head size 256, full-attention layers with head size 512, proportional RoPE
and no value projection (the keys double as values), with 2 key/value heads
(transformers' expert-parallel plan); queries and keys RMS-normalised, values
too, score scale 1; logits softcapped at 30; vocabulary 262,144 tied to the
output head. 26B parameters, ~4B active per token. Not in the transformers
source, and not confirmed offline: the width, layer count, head counts, MLP
and expert sizes, expert count and top-k, and the sliding/full order (the
default is 5 sliding : 1 full, last layer full).

DEVICE: 16.3M parameters, ~8.4 MB at int4 -- but 13.02 MB lowered, because
the experts' 1.2M parameters stay fp32: the backend keeps 3-D expert weights
outside the delegates.
- Width -> 256; 2 query heads sharing 1 key/value head on every layer.
- Layers -> 12: two periods of 5 sliding + 1 full.
- Dense MLP 768 (3x the width) beside 8 experts of 16, 2 per token: the
  experts are the flash budget's hardest limit at 4 bytes a parameter.
- Canvas 256 -> 8 tokens, the exported program's block (one denoising step).
- Vocabulary 262,144 -> 8,192.
- A one-layer, 32-wide vision tower: DiffusionGemma always builds one; it
  never runs on a text prompt and is not in the exported program.
- Kept as released: the encoder/decoder weight sharing, self-conditioning,
  bidirectional canvas attention over the cached prompt, the dense-MLP +
  experts block on every layer, keys as values on full layers, head sizes 256
  and 512, the two RoPE kinds, QK/V norms, softcap 30, tied embeddings.

Other ways to fit it:
- 8 experts of 64 (a quarter of the width each): 4.7M fp32 expert
  parameters, 18.9 MB: over the region on experts alone.
- One period of 6 layers with 8 experts of 32 and a dense MLP of 1536:
  14.2M, ~11.4 MB in flash; half the depth, experts twice as wide (but
  ~7.3 MB at int4, under the zoo's 8 MB floor).
- 4 experts of 32: the same flash as 8 of 16; fewer, wider experts.
"""

MODEL_TYPE = "diffusion_gemma"
MODEL_CLASS = "DiffusionGemmaForBlockDiffusion"

# Random weights leave q_norm/k_norm at scale 1 and the score scale is 1, so
# the scores are ~N(0, head size): near one-hot softmaxes, and the device
# arithmetic itself is that touchy (a 1e-4 nudge of the embedding table moves
# the int4/int8 reference to cosine 0.629). The delegate's int8 Q and K rows
# put ~1% error on every score. Measured at device size: cosine 0.774 against
# the device arithmetic, 0.9999 with the q@kT left in fp32. CI's toy heads clear
# the gate (four seeds: 0.9933-0.9993).
# The routers then pick other experts for close calls, which is why this one
# falls furthest.
EXPECT = {"device": "host: int8 q@kT moves near one-hot QK-normed attention and "
                    "the expert choice (cosine 0.774; 0.9999 with q@kT in fp32)"}

# The decoder reads the cache the encoder writes, so use_cache keeps its default.
FORWARD_KWARGS = {}

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                       "rope_theta": 1_000_000.0},
}


def _pattern(periods):
    """5 sliding-window layers then 1 full-attention layer, `periods` times."""
    return (["sliding_attention"] * 5 + ["full_attention"]) * periods


def INPUTS(cfg):
    """The prompt for the encoder and an 8-token canvas for the decoder."""
    from executorch.examples.espressif.nano_zoo import build

    prompt = build.ids(cfg)["input_ids"]
    return {"input_ids": prompt, "decoder_input_ids": prompt.clone()}


TEXT = dict(
    vocab_size=8192, hidden_size=256, intermediate_size=768, num_hidden_layers=12,
    num_attention_heads=2, num_key_value_heads=1, head_dim=256,
    # Builder-only kwarg: the head size of the full-attention layers.
    global_head_dim=512,
    layer_types=_pattern(2), sliding_window=512, rope_parameters=ROPE,
    num_experts=8, top_k_experts=2, moe_intermediate_size=16,
    hidden_activation="gelu_pytorch_tanh", max_position_embeddings=131072, rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)

VISION = dict(model_type="gemma4_vision", hidden_size=32, intermediate_size=64,
              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
              patch_size=16, position_embedding_size=64)


def _composite(text):
    """Image token ids inside the vocabulary, clear of the prompt's ids 1..8."""
    vocab = text["vocab_size"]
    return dict(text_config=text, vision_config=VISION, canvas_length=8,
                boi_token_id=vocab - 4, eoi_token_id=vocab - 3, image_token_id=vocab - 2,
                tie_word_embeddings=True)


DEVICE = _composite(TEXT)

CI = _composite(dict(TEXT, vocab_size=256, hidden_size=64, intermediate_size=192,
                     num_hidden_layers=2, head_dim=32, global_head_dim=64,
                     layer_types=["sliding_attention", "full_attention"], sliding_window=4,
                     num_experts=4, moe_intermediate_size=16))
