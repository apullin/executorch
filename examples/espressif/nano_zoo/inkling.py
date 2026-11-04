# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""nanoInkling: hybrid attention, short convolutions and MoE, random weights.

Released: thinkingmachines/Inkling; numbers from transformers 5.19.0
InklingTextConfig: width 6144, 66 layers, 64 query/8 KV heads for global
attention (8:1), 64 query/16 KV for sliding attention (4:1), all head sizes
128. A five-sliding/one-global period, window 512; per-head relative states
of size 16 with extent 1024, and four short convolutions of kernel size 4
per layer. Dense MLP 24576 (4x), expert MLP 3072 (0.5x); 256 routed experts,
top-6, two shared experts that also act as router softmax sinks. RMSNorm,
SwiGLU, learned MLP/router scale, muP output divisor 24, vocabulary 201,024.
The default is all sparse MLPs; the source supports a dense prefix but does
not give the checkpoint's prefix length. The model source explicitly says
its embeddings are untied. The unpadded output row count and any MTP layers
are not known here.
This uses the existing InklingForCausalLM text class, leaving out the audio
and vision towers of InklingForConditionalGeneration.

DEVICE: 27,444,166 parameters, ~14.15 MB estimated int4 weights. Counted on
the meta device before allocating weights; program bytes are measured separately.
- Width 6144 -> 128; global heads 64/8 -> 8/1, sliding 64/16 -> 4/1 retain ratios.
- Layers 66 -> 6 keep the full five-sliding/one-global period.
- Dense/expert widths 24576/3072 -> 512/64 preserve 4x and 0.5x.
- Experts 256 -> 192 fit that six-layer period while keeping top-6 and two shared.
- One dense prefix layer exercises the supported dense path; the actual
  released prefix is unspecified by this source, so this is a coverage choice.
- Vocabulary 201,024 -> 8,192 cuts table storage, with a full 8,192-row
  output head and no MTP (no released MTP count is supplied by these defaults).
- Window 512 -> 4 and relative extent 1024 -> 8 exercise local boundaries
  and relative bias in the 8-token prompt.
- Kept: 128-wide heads, relative state size 16, all four kernel-4 convolutions,
  norm/activation, shared expert sinks, learned scales, muP divisor, untied embeddings.
On the board: both sizes pass host parity, but DEVICE emits 97.49 MB.
Routed expert tables and the shared experts' three grouped projections
remain fp32; the DEVICE-only EXPECT records this flash-storage gap.

Other ways to fit it (meta-counted int4 weight estimates):
- Width 128, six layers and 128 experts: 19.54M parameters, 10.07 MB;
  the same full hybrid period with less expert capacity.
- Width 256, six layers and 32 experts: 26.45M, 13.64 MB; wider attention
  and dense projections, a much smaller expert pool. Global heads 16/2,
  sliding heads 8/2, dense/expert widths 1024/128 preserve ratios.
"""

import os as _os

MODEL_TYPE = "inkling_text"
MODEL_CLASS = "InklingForCausalLM"

DEVICE = dict(
    vocab_size=8192, unpadded_vocab_size=None, hidden_size=128, num_hidden_layers=6,
    num_attention_heads=8, num_key_value_heads=1, head_dim=128,
    swa_num_attention_heads=4, swa_num_key_value_heads=1, swa_head_dim=128,
    layer_types=["hybrid_sliding"] * 5 + ["hybrid"],
    mlp_layer_types=["dense"] + ["sparse"] * 5,
    sliding_window_size=4, d_rel=16, rel_extent=8, conv_kernel_size=4,
    intermediate_size=512, moe_intermediate_size=64, n_routed_experts=192,
    num_experts_per_tok=6, n_shared_experts=2, shared_expert_sink=True, route_scale=8.0,
    max_position_embeddings=131072, rms_norm_eps=1e-6, rms_norm_eps_moe_gate=1e-6,
    hidden_act="silu", logits_mup_width_multiplier=24.0,
    num_mtp_layers=None, bos_token_id=1, eos_token_id=2,
)

CI = dict(DEVICE, vocab_size=256, hidden_size=64,
          num_attention_heads=8, num_key_value_heads=1, head_dim=16,
          swa_num_attention_heads=4, swa_num_key_value_heads=1, swa_head_dim=16,
          intermediate_size=256, moe_intermediate_size=32,
          n_routed_experts=4, num_experts_per_tok=2, d_rel=4)

# The toy tables fit; retain its parity gate while DEVICE storage is unsupported.
EXPECT = ("flash: routed and shared 3-D expert tables stay fp32 "
          "(97.49 MB emitted package)"
          if _os.environ.get("NANO_ZOO_SIZE", "device") == "device" else None)
