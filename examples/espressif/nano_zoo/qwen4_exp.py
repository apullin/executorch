# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoQwen4Exp: delta-net, indexed attention, PLE and HC, random weights.

Released: Qwen/Qwen4-Exp, using transformers 5.19.0 Qwen4ExpTextConfig:
width 2048, 40 layers repeating three linear (gated delta-net) and one
indexed (QSA) layer. Indexed attention has 16 query/2 KV heads (8:1), head
size 256; linear attention has 16 key/32 value heads of size 128 and a
kernel-4 convolution. Every MLP has 512 routed experts, top-10, expert and
shared-expert widths 512 (0.25x). Four hyper-connection streams, mixer rank
320, RMSNorm/SwiGLU, untied embeddings, vocabulary 248,320. PLE supports
2- and 3-grams with eight independently hashed heads per order, prime
vocabularies starting at 20 million, kernel-4 dilated depthwise convolution.
The source leaves PLE layer ids and QSA sizing fields unset: the checkpoint's
table placement, indexer heads, budget and compression ratio are unknown.
Uses Qwen4ExpForCausalLM, the existing text class, without the vision tower.

DEVICE: 28,010,636 parameters, ~14.44 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 2048 -> 128, layers 40 -> 4 keep the full three-linear/one-QSA period.
- Query/KV heads 16/2 -> 8/1 is the smallest pair retaining 8:1 and size 256.
- Linear heads 16/32 -> 1/2 keep their 2:1 ratio and head size 128.
- Expert/shared widths 512 -> 32 keep 0.25x; HC mixer rank 320 -> 20 scales with width.
- Experts 512 -> 480 reserve about 0.8 MB for the program and metadata; 512
  leaves only 7 KB beyond the nominal int4 weights at this width and period.
- Vocabulary 248,320 -> 8,192 cuts table storage; eos id 2 lets PLE reset n-grams.
- PLE is enabled on the three linear layers; dimension 128 keeps 16 hash
  heads, while base table vocabulary 20 million -> 512 makes the tables fit.
- QSA uses 2 index query heads, one KV head, dimension 128, budget 4 and
  compression 2 to exercise sparse block selection and tails on 8 tokens.
  These are nano choices because the released values are absent from source.
- RoPE uses the source's [11,11,10] multimodal sections over 64 rotary
  coordinates (1/4 of the 256-wide head); checkpoint RoPE settings are unknown.
- Kept: top-10 routing, shared expert, four HC streams, attention
  period, head sizes/ratios, n-gram orders/hash heads, kernel-4 convolutions,
  gating, norms, activation and untied embeddings.
Export gap: QSA extracts visible indices with nonzero(), then tests
whether their data-dependent count contains any complete compressed blocks.
Both sizes fail at that branch before the ESP32 partitioner; QSA and PLE
remain enabled in this fixture.
A second gap lies behind it (an all-linear CI configuration with PLE reaches
it): PLE's EOS-aware n-gram shifts use cummax, which is outside the Core ATen
opset.

Other ways to fit it (meta-counted int4 weight estimates):
- Width 256, four layers and 64 experts: 22.74M parameters, 11.72 MB;
  wider projections with fewer experts. Linear heads 2/4, expert/shared
  widths 64/64, HC rank 40 and PLE dimension 256 scale with width.
- Width 128, 256 experts and PLE vocabulary base 16,384: 23.02M, 11.87 MB;
  trade expert capacity for much larger n-gram tables and fewer hash collisions.
"""

MODEL_TYPE = "qwen4_exp_text"
MODEL_CLASS = "Qwen4ExpForCausalLM"

DEVICE = dict(
    vocab_size=8192, hidden_size=128, num_hidden_layers=4,
    num_attention_heads=8, num_key_value_heads=1, head_dim=256,
    layer_types=["linear_attention"] * 3 + ["indexed_attention"],
    linear_key_head_dim=128, linear_value_head_dim=128,
    linear_num_key_heads=1, linear_num_value_heads=2, linear_conv_kernel_dim=4,
    moe_intermediate_size=32, shared_expert_intermediate_size=32,
    num_experts=480, num_experts_per_tok=10, norm_topk_prob=True,
    hc_count=4, hc_lowrank=20,
    ple_layer_ids=[1, 2, 3], ple_embed_dim=128, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=512,
    make_ngram_vocab_size_divisible_by=128, seed=1234, split_ngram_parts=512,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=128,
    indexer_budget=4, indexer_compress_ratio=2,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
    max_position_embeddings=32768, hidden_act="silu", rms_norm_eps=1e-6,
    tie_word_embeddings=False, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64, head_dim=16,
          linear_key_head_dim=16, linear_value_head_dim=16,
          linear_num_key_heads=4, linear_num_value_heads=8,
          moe_intermediate_size=16, shared_expert_intermediate_size=16,
          num_experts=4, num_experts_per_tok=2, hc_lowrank=10,
          ple_embed_dim=64, ngram_vocab_size_base=64, indexer_head_dim=16,
          rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                           "mrope_section": [4, 2, 2]})

EXPECT = "export: QSA nonzero visible-token count makes the complete-block branch data-dependent"
