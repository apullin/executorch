# MIT License
#
# Copyright (c) 2026 Viacheslav Sierbov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# An excerpt of src/model.py from esp32-ai (https://github.com/slvDev/esp32-ai).

"""Muse Glimmer, the decoder the story demo exports, and the tinylm backbone.

The Glimmer path of the model, with Per-Layer Embeddings: per-layer
inputs are (RMSNorm(proj(embed)) + table) / sqrt(2), gated into each layer's
residual. The TinyLM backbone stays for ``Config(arm="ple")``, which the
backend's fusion tests build. The fatembed and bigcore arms, the looped and
Mamba/ABC variants, landmark attention and the long-context training paths
are left out.

Module and attribute names are the checkpoint's state_dict keys, and the story
program's non-delegated constants are named after them; the order modules are
created in fixes what a seeded random initialisation produces.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    arm: str = "baseline"
    # "glimmer" swaps only the text backbone; PLE is selected independently
    # by arm.
    architecture: str = "tinylm"
    vocab_size: int = 4096
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 256
    seq_len: int = 512
    ple_dim: int = 64
    rope_theta: float = 10000.0
    # Muse Glimmer text invariants. At this scale 8:1 is the strongest useful
    # Q:KV ratio for the default d_model=96/n_heads=8; the released model is
    # 16:1. Local layers repeat [local, local, local, global].
    n_kv_heads: int = 1
    glimmer_window: int = 64
    glimmer_rope_theta: float = 500_000.0
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    # Evaluation/deployment experiment matching the ExecuTorch backend's
    # opt-in OP_BMM_DYN_I8 path. Training keeps this false.
    dynamic_qk: bool = False

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def uses_per_layer(self):
        return self.arm in ("ple", "ple_notable")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class WeightlessRMSNorm(nn.Module):
    """Glimmer's embedding and Q/K normalization."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class CenteredRMSNorm(nn.Module):
    """Glimmer scale parameterization: effective weight is 1 + weight."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


def build_rope(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config, idx: int = -1):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        if cfg.uses_per_layer:
            # Gemma 4's Gemma4TextDecoderLayer, minus AltUp: squeeze the hidden
            # state to ple_dim, gate it elementwise *by* the per-layer input,
            # project back, norm, add to the residual. The multiply is what makes
            # this conditioning rather than a per-token bias -- either factor can
            # suppress the other.
            self.ple_gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
            self.ple_proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
            self.ple_norm = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, ple=None):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        if ple is not None:
            g = F.gelu(self.ple_gate(x))
            x = x + self.ple_norm(self.ple_proj(g * ple))
        return x


def repeat_kv(x, repeats):
    """[B, KVH, T, Dh] -> [B, QH, T, Dh] without learned parameters."""
    if repeats == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, repeats, T, D).reshape(B, H * repeats, T, D)


def dynamic_i8_qk(q, k):
    """Backend-exact dynamic int8 Q@K.

    Each vector gets an absmax/127 scale and round-half-away codes. The
    integer products sum in int64, exactly at any length, as the device's
    int32 dot sums them within the gate's dot-length bound (a float32 matmul
    is exact only while every partial sum stays below 2**24: K ~ 1040 codes
    near 127), then the sum converts to float32 once and the two scales are
    applied in the device's `(acc * sq) * sk` order. An exact-zero score is
    +0.0, as the device's int32 zero converts.
    """
    def codes(x):
        scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
        # 0 below FLT_MIN (a zero or subnormal scale): that row's codes are
        # all 0, as the device's a0_inverse_scale (runtime/quant_rules.h).
        inv = torch.where(scale >= 2.0 ** -126, scale.reciprocal(),
                          torch.zeros_like(scale))
        z = x * inv
        qx = torch.sign(z) * torch.floor(torch.abs(z) + 0.5)
        return qx.clamp(-127, 127), scale

    qq, sq = codes(q)
    kk, sk = codes(k)
    acc = torch.matmul(qq.to(torch.int64), kk.to(torch.int64).transpose(-1, -2))
    # The low 32 bits, as dot_i8_s3.S returns them: within the gate's bound
    # (MAX_DOT_LEN * 127 * 127 < 2**31) this changes nothing.
    acc = torch.remainder(acc + 2**31, 2**32) - 2**31
    return (acc.to(torch.float32) * sq) * sk.transpose(-1, -2)


class GlimmerAttention(nn.Module):
    """Released Muse Glimmer attention, scaled in width but not in structure."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("Glimmer n_heads must be divisible by n_kv_heads")
        self.cfg = cfg
        self.local = layer_idx % 4 != 3
        q_width = cfg.n_heads * cfg.head_dim
        kv_width = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, q_width, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, q_width, bias=False)
        self.o_proj = nn.Linear(q_width, cfg.d_model, bias=False)
        self.qk_norm = WeightlessRMSNorm(cfg.rms_norm_eps)

    def attention_mask(self, length, device):
        """Build the training mask on demand, not once per model layer.

        Decode does not use this path: it carries a KV cache and constructs a
        one-row validity mask.  Eagerly allocating an SxS mask in every layer
        would nevertheless make even constructing an 8K--16K decode model
        consume gigabytes before export.  Keeping the quadratic training mask lazy
        separates that training/prefill problem from the O(S) decode graph.
        """
        row = torch.arange(length, device=device)[:, None]
        col = torch.arange(length, device=device)[None, :]
        keep = col <= row
        if self.local:
            keep = keep & ((row - col) < self.cfg.glimmer_window)
        mask = torch.zeros(length, length, device=device)
        return mask.masked_fill(~keep, float("-inf"))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, KH, Dh = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KH, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KH, Dh).transpose(1, 2)
        q = self.qk_norm(q) * self.cfg.qk_scale_factor
        k = self.qk_norm(k)
        if self.local:  # global layers are intentionally NoPE
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        repeats = H // KH
        k, v = repeat_kv(k, repeats), repeat_kv(v, repeats)
        mask = self.attention_mask(T, x.device)
        if self.cfg.dynamic_qk:
            scores = dynamic_i8_qk(q, k) * (Dh ** -0.5)
            scores = scores + mask
            o = torch.matmul(torch.softmax(scores, dim=-1), v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        o = o * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(o)


class GlimmerBlock(nn.Module):
    def __init__(self, cfg: Config, idx: int):
        super().__init__()
        self.input_norm = CenteredRMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GlimmerAttention(cfg, idx)
        self.post_attn_norm = CenteredRMSNorm(cfg.d_model, cfg.post_norm_eps)
        self.pre_ffn_norm = CenteredRMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)
        self.post_ffn_norm = CenteredRMSNorm(cfg.d_model, cfg.post_norm_eps)
        if cfg.uses_per_layer:
            self.ple_gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
            self.ple_proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
            self.ple_norm = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, ple=None):
        x = x + self.post_attn_norm(self.attn(self.input_norm(x), cos, sin))
        x = x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))
        if ple is not None:
            g = F.gelu(self.ple_gate(x))
            x = x + self.ple_norm(self.ple_proj(g * ple))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.architecture not in ("tinylm", "glimmer"):
            raise ValueError(f"unknown architecture {cfg.architecture!r}")
        if cfg.arm not in ("baseline", "ple", "ple_notable"):
            raise ValueError(f"unknown arm {cfg.arm!r}")
        if cfg.architecture == "glimmer":
            if cfg.d_model % cfg.n_heads or cfg.head_dim % 2:
                raise ValueError("Glimmer requires d_model/n_heads and an even head_dim")
            if not 0 < cfg.glimmer_window <= cfg.seq_len:
                raise ValueError("glimmer_window must be in [1, seq_len]")

        if cfg.architecture == "glimmer":
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.tok_emb_norm = WeightlessRMSNorm(cfg.rms_norm_eps)
            # Muse Glimmer does not tie input and output embeddings.
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # tied

        if cfg.uses_per_layer:
            # Context-aware half of the per-layer input: one projection of the
            # token embedding, reshaped to (layers, ple_dim) and normed per slice.
            self.ple_model_proj = nn.Linear(cfg.d_model, cfg.n_layers * cfg.ple_dim, bias=False)
            self.ple_proj_norm = RMSNorm(cfg.ple_dim)
        if cfg.arm == "ple":
            self.ple_table = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)

        block_cls = GlimmerBlock if cfg.architecture == "glimmer" else Block
        self.blocks = nn.ModuleList([block_cls(cfg, i) for i in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model, eps=(cfg.rms_norm_eps
                                                  if cfg.architecture == "glimmer"
                                                  else 1e-6))

        self.apply(self._init)
        for n, p in self.named_parameters():
            # Scale down residual-writing projections, GPT-2 style. The tinylm
            # backbone keeps its broad suffix rule (its seeded initialisation
            # depends on it); Glimmer's q/k/v/gate all end in `_proj` and must
            # not be swept into it.
            residual = ((n.endswith("attn.o_proj.weight") or n.endswith("ffn.down.weight"))
                        if cfg.architecture == "glimmer"
                        else (n.endswith("proj.weight") or n.endswith("down.weight")))
            if residual:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        for block in self.blocks:
            # The per-layer branch ends in an RMSNorm, which would undo the small
            # residual init above and inject unit-scale noise from step 0. Zeroing
            # the norm gain makes the branch start as an exact no-op, so every arm
            # begins from the same function and the comparison measures learning
            # rather than initialisation luck.
            if cfg.uses_per_layer:
                nn.init.zeros_(block.ple_norm.weight)

        rope_theta = (cfg.glimmer_rope_theta if cfg.architecture == "glimmer"
                      else cfg.rope_theta)
        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        x = self.tok_emb(idx)
        if cfg.architecture == "glimmer":
            x = self.tok_emb_norm(x)

        ple = None
        if cfg.uses_per_layer:
            B, T = idx.shape
            ple = self.ple_model_proj(x) * (cfg.d_model**-0.5)
            ple = self.ple_proj_norm(ple.view(B, T, cfg.n_layers, cfg.ple_dim))
            if cfg.arm == "ple":
                # embed_scale = sqrt(ple_dim), applied to the table rather than
                # baked into init -- undocumented in Gemma's config but load-bearing.
                table = self.ple_table(idx).view(B, T, cfg.n_layers, cfg.ple_dim)
                ple = (ple + table * (cfg.ple_dim**0.5)) * (2**-0.5)

        for i, block in enumerate(self.blocks):
            block_ple = None if ple is None else ple[:, :, i]
            x = block(x, self.cos, self.sin, block_ple)

        x = self.out_norm(x)
        logits = self.head(x)
        if cfg.architecture == "glimmer":
            logits = logits * cfg.output_multiplier
            cap = cfg.final_logit_softcapping
            if cap > 0:
                logits = torch.tanh(logits / cap) * cap
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss
