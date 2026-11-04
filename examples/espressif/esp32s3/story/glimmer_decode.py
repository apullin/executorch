# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-token decode for the Muse Glimmer architecture (story/glimmer_model.py).

Generation does not execute the full-sequence graph: it projects one token,
appends compact K/V state, and attends against the persistent cache. This
module wraps the model's own modules, so training and deployment share
weights and arithmetic rather than two look-alike models.

GQA stays compact throughout.  The cache is ``[KVH, S, Dh]`` and Q is viewed as
``[KVH, QH/KVH, Dh]`` for both batched matmuls.  Repeating K/V to Q-head width
would use eight times the state (or materialise an equally large temporary) in
the story configuration (8 query heads, 1 KV head).

``int8_kcache=True`` (off by default) stores K as int8 codes ``[KH, S, Dh]``
plus an fp32 per-row scale ``[KH, S]`` (``kcache_{li}`` / ``kcache_scale_{li}``)
instead of fp32. Each new key row is quantised once, when it is written, with
the arithmetic the int8 q@kT delegate otherwise repeats on every cached row at
every step, so the scores are bit-identical and K takes a quarter of the bytes.
See kcache_i8.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.espressif.esp32s3.story.glimmer_model import (
    Config,
    TinyLM,
    apply_rope,
    dynamic_i8_qk,
)

from executorch.backends.esp32s3.ops.kcache_i8 import (
    QUANTIZE_ROWS_I8,
    cached_i8_qk,
    dequantized_keys,
)  # noqa: E402
import numpy as np
from executorch.backends.esp32s3.serialization.weight_formats import (
    SCALE_GROUP,
    quantize_absmax_rows,
)


def _portable_softmax(scores):
    """The device's float32 softmax (shim_softmax_out: its order, newlib's
    expf), bit for bit (test/reference.py softmax_f32_portable)."""
    from executorch.backends.esp32s3.test.reference import softmax_f32_portable
    return torch.from_numpy(softmax_f32_portable(
        scores.detach().float().numpy(), dim=-1))


def _portable_bmm(lhs, rhs):
    """The device's float32 bmm (shim_bmm_out: its order, one fma per step),
    bit for bit (test/reference.py bmm_f32_portable)."""
    from executorch.backends.esp32s3.test.reference import bmm_f32_portable
    return torch.from_numpy(bmm_f32_portable(
        lhs.detach().float().numpy(), rhs.detach().float().numpy())).to(lhs.dtype)


def _device_rope(x, cos, sin):
    """RoPE as the device's espressif::rope computes it (host mirror):
    x [..., Dh] split into halves, the pairwise [.., Dh/2] tables shared by
    both, one rounding per output in the chip's order
    (test/reference.py rope_f32_device). The torch spelling the exported
    program carries (and the fusion pass matches) rounds each product."""
    from executorch.backends.esp32s3.test.reference import rope_f32_device
    return torch.from_numpy(rope_f32_device(
        x.detach().float().numpy(), cos.detach().float().numpy(),
        sin.detach().float().numpy())).to(x.dtype)


def _attn_tail_pv(scores, values):
    """softmax + P@V in espressif::attn_tail's arithmetic (host mirror).

    ``scores`` arrive scaled and masked, which is exactly the kernel's first
    pass (one fp32 multiply, then the -inf select), so the mirror starts after it.
    """
    from executorch.backends.esp32s3.test.models import attn_tail_softmax_pv_f32_reference
    probs, out = attn_tail_softmax_pv_f32_reference(
        scores.detach().float().numpy(), values.detach().float().numpy())
    return torch.from_numpy(probs), torch.from_numpy(out).to(values.dtype)


def _register_kv(module, li, cfg, cache_len, int8_kcache):
    """Register one layer's K/V state; ``"cache_"`` names every piece of it."""
    shape = (cfg.n_kv_heads, cache_len, cfg.head_dim)
    if int8_kcache:
        module.register_buffer(f"kcache_{li}", torch.zeros(shape, dtype=torch.int8))
        module.register_buffer(f"kcache_scale_{li}", torch.zeros(shape[:2]))
    else:
        module.register_buffer(f"kcache_{li}", torch.zeros(shape))
    module.register_buffer(f"vcache_{li}", torch.zeros(shape))


class GlimmerDecodeStep(nn.Module):
    """``forward(token, pos) -> logits[V]``, mutating compact per-layer KV.

    ``token`` is int64 ``[1]`` and ``pos`` is an int64 scalar.  Shapes remain
    static: every step scans ``seq_len`` cache rows and masks invalid/local-old
    positions, which keeps the ExecuTorch memory plan fixed.
    """

    def __init__(self, lm: TinyLM, *, reference_dynamic_qk: bool = False,
                 local_ring: bool = False, cache_context: int | None = None,
                 selected_rows: int = 0,
                 head_token_ids: torch.Tensor | list[int] | None = None,
                 int8_kcache: bool = False):
        super().__init__()
        if lm.cfg.architecture != "glimmer":
            raise ValueError("GlimmerDecodeStep requires architecture='glimmer'")
        if lm.cfg.dynamic_qk:
            raise ValueError(
                "dynamic_qk is a backend lowering policy for decode; keep the "
                "source model fp32 and pass quantize_qk=True to the partitioner"
            )
        self.lm = lm
        # Host-reference only.  The exported graph must leave this false so the
        # raw q@k BMM reaches Esp32Partitioner; a second eager instance turns it
        # on to mirror OP_BMM_DYN_I8 numerically for QEMU checksums.
        self.reference_dynamic_qk = reference_dynamic_qk
        self.local_ring = local_ring
        self.int8_kcache = int8_kcache
        if selected_rows < 0:
            raise ValueError("selected_rows must be non-negative")
        self.selected_rows = selected_rows
        cfg = lm.cfg
        self.head_token_ids = None
        if head_token_ids is not None:
            ids = torch.as_tensor(
                head_token_ids, dtype=torch.long,
                device=lm.head.weight.device)
            if (ids.ndim != 1 or ids.numel() == 0
                    or int(ids.min()) < 0 or int(ids.max()) >= cfg.vocab_size
                    or ids.unique().numel() != ids.numel()):
                raise ValueError(
                    "head_token_ids must be a non-empty unique 1-D subset of "
                    "the model vocabulary")
            if selected_rows == 0:
                raise ValueError("a pruned head requires selected_rows > 0")
            if selected_rows > ids.numel():
                raise ValueError("selected_rows exceeds the pruned head width")
            # A bounded grammar can name only a small union of output tokens.
            # Copy those learned rows once and omit the
            # unused full head from the exported graph. Runtime row IDs are then
            # local indices; the host grammar maps them back to tokenizer IDs.
            self.register_buffer(
                "head_table",
                lm.head.weight.detach().index_select(0, ids).clone(),
            )
            self.head_token_ids = tuple(int(value) for value in ids)
        else:
            self.register_buffer("head_table", None)
        self.cache_context = cfg.seq_len if cache_context is None else cache_context
        if not cfg.glimmer_window <= self.cache_context <= cfg.seq_len:
            raise ValueError("cache_context must be between window and seq_len")
        for li, block in enumerate(lm.blocks):
            cache_len = (cfg.glimmer_window
                         if local_ring and block.attn.local
                         else self.cache_context)
            _register_kv(self, li, cfg, cache_len, int8_kcache)

    def forward(self, token: torch.Tensor, pos: torch.Tensor,
                head_rows: torch.Tensor | None = None) -> torch.Tensor:
        lm, cfg = self.lm, self.lm.cfg
        H, KH, Dh, S = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.seq_len
        repeats = H // KH

        x = lm.tok_emb_norm(lm.tok_emb(token))                 # [1, D]

        ple = None
        if cfg.uses_per_layer:
            ple = lm.ple_model_proj(x) * (cfg.d_model**-0.5)
            ple = lm.ple_proj_norm(ple.view(1, cfg.n_layers, cfg.ple_dim))
            if cfg.arm == "ple":
                table = lm.ple_table(token).view(1, cfg.n_layers, cfg.ple_dim)
                ple = (ple + table * (cfg.ple_dim**0.5)) * (2**-0.5)
            # One split for all layers rather than a select per layer: it
            # exports as a single split_with_sizes_copy, and the per-layer
            # views below are free (remove_view_copy aliases them).
            ple = torch.split(ple, 1, dim=1)

        # Local layers use slice-half RoPE; global layers deliberately use NoPE.
        # The tables have width Dh/2 in the trainable model.
        cos = lm.cos.index_select(0, pos.reshape(1))            # [1, Dh/2]
        sin = lm.sin.index_select(0, pos.reshape(1))
        neg_inf = torch.full((), float("-inf"), dtype=torch.float32,
                             device=token.device)
        # Layers of one kind attend over the same cache slots, so the ring slot
        # and each validity mask are built once per token and shared, not once
        # per layer: on the chip each mask is ~0.6 ms of portable ops.
        ring_slot = torch.remainder(pos, cfg.glimmer_window) if self.local_ring else None
        masks: dict[tuple[int, bool], torch.Tensor] = {}

        for li, block in enumerate(lm.blocks):
            h = block.input_norm(x)
            q = block.attn.q_proj(h).view(H, Dh)
            k = block.attn.k_proj(h).view(KH, Dh)
            v = block.attn.v_proj(h).view(KH, Dh)
            q = block.attn.qk_norm(q) * cfg.qk_scale_factor
            k = block.attn.qk_norm(k)

            if block.attn.local and self.reference_dynamic_qk:
                q = _device_rope(q, cos, sin)
                k = _device_rope(k, cos, sin)
            elif block.attn.local:
                q1, q2 = q.chunk(2, dim=-1)
                k1, k2 = k.chunk(2, dim=-1)
                q = torch.cat((q1 * cos - q2 * sin,
                               q2 * cos + q1 * sin), dim=-1)
                k = torch.cat((k1 * cos - k2 * sin,
                               k2 * cos + k1 * sin), dim=-1)

            kh = getattr(self, f"kcache_{li}")                # [KH, S, Dh]
            vh = getattr(self, f"vcache_{li}")
            cache_len = kh.shape[1]
            # Global layers write the cache at the absolute position; only local layers wrap the ring.
            slot = ring_slot if self.local_ring and block.attn.local else pos
            # view, not unsqueeze: ExecuTorch removes a view_copy (it aliases the
            # buffer) but runs unsqueeze_copy as a portable copy, 16 per token.
            if self.int8_kcache:
                # Quantised ONCE, here; every later step dots these codes.
                ks = getattr(self, f"kcache_scale_{li}")
                codes, scales = QUANTIZE_ROWS_I8(k)          # [KH, Dh], [KH]
                kh.index_copy_(1, slot.reshape(1), codes.view(KH, 1, Dh))
                ks.index_copy_(1, slot.reshape(1), scales.view(KH, 1))
            else:
                kh.index_copy_(1, slot.reshape(1), k.view(KH, 1, Dh))
            vh.index_copy_(1, slot.reshape(1), v.view(KH, 1, Dh))

            # GQA without repeat_kv: KV head is the batch, its Q heads are M.
            qg = q.view(KH, repeats, Dh)
            # Spell these as bmm explicitly.  aten.matmul decomposes even an
            # already-3D pair through redundant views, hiding the folded K
            # transpose from the backend's deliberately strict Q@K matcher.
            if self.int8_kcache and self.reference_dynamic_qk:
                scores = cached_i8_qk(qg, kh, ks) * (Dh**-0.5)
            elif self.int8_kcache:
                keys = dequantized_keys(kh, ks)
                scores = torch.bmm(qg, keys.transpose(1, 2)) * (Dh**-0.5)
            elif self.reference_dynamic_qk:
                scores = dynamic_i8_qk(qg, kh) * (Dh**-0.5)
            else:
                scores = torch.bmm(qg, kh.transpose(1, 2)) * (Dh**-0.5)
            windowed = bool(block.attn.local and not self.local_ring)
            valid = masks.get((cache_len, windowed))
            if valid is None:
                cache_pos = torch.arange(cache_len, device=token.device)
                valid = cache_pos <= pos
                if windowed:
                    # Inclusive lower bound, written tensor-vs-tensor.  The direct
                    # `(pos-cache_pos) < window` exports aten::lt.Scalar_out, which
                    # is absent from the selective ExecuTorch runtime; causal above
                    # already proves aten::le.Tensor_out is available.
                    lower = pos - (cfg.glimmer_window - 1)
                    valid = valid & (lower <= cache_pos)
                masks[(cache_len, windowed)] = valid
            # Direct selection is the device-proven mask path. The Q@K matcher
            # explicitly walks through this pre-softmax where, so the
            # dynamic-i8 BMM still delegates.
            scores = torch.where(valid.view(1, 1, cache_len), scores, neg_inf)
            if self.reference_dynamic_qk:
                # Device-faithful reference: the exported tail fuses into one
                # espressif::attn_tail (fuse_attn_tail_pass.py), so mirror its
                # fast_exp_neg, per-row reciprocal and index-order fused P@V. Over
                # a 256-wide cache the accumulation order alone is worth 1e-3
                # of the selected logits, which the strict parity gate can see.
                probs, out = _attn_tail_pv(scores, vh)
            else:
                probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.bmm(probs, vh)
            out = out.reshape(1, cfg.d_model)
            out = out * torch.sigmoid(block.attn.gate_proj(h))
            x = x + block.post_attn_norm(block.attn.o_proj(out))

            x = x + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(x)))
            if ple is not None:
                g = F.gelu(block.ple_gate(x))
                x = x + block.ple_norm(block.ple_proj(g * ple[li].view(1, cfg.ple_dim)))

        x = lm.out_norm(x)
        if self.selected_rows:
            if head_rows is None:
                raise ValueError("selected-head decode requires head_rows")
            # Keep the full constant head table in the graph and make only its
            # row IDs dynamic.  The ESP32 backend fuses this gather + linear
            # into selected-row matvec; portable execution is merely the eager
            # reference spelling.
            table = (self.head_table if self.head_table is not None
                     else lm.head.weight)
            rows = torch.index_select(table, 0, head_rows)
            logits = F.linear(x, rows)
        else:
            logits = lm.head(x)
        logits = logits * cfg.output_multiplier
        cap = cfg.final_logit_softcapping
        if cap > 0:
            logits = torch.tanh(logits / cap) * cap
        return logits


class GlimmerPrefillChunk(nn.Module):
    """Vectorized cached prefill: ``forward(tokens[C], start_pos) -> logits[V]``.

    Unlike calling decode C times, every projection sees C activation rows at
    once. That reaches the backend's row-blocked addmm path, which makes one
    pass over a weight tile for the whole block. Attention reads the OLD cache
    plus the new block and commits K/V only after computing the block, so a
    local ring cannot overwrite history needed by an earlier query in the same
    block.
    """

    def __init__(self, lm: TinyLM, chunk_size: int, *,
                 reference_dynamic_qk: bool = False,
                 cache_context: int | None = None,
                 selected_rows: int = 0,
                 head_token_ids: torch.Tensor | list[int] | None = None,
                 int8_kcache: bool = False):
        super().__init__()
        if lm.cfg.architecture != "glimmer":
            raise ValueError("GlimmerPrefillChunk requires architecture='glimmer'")
        if lm.cfg.dynamic_qk:
            raise ValueError("keep source Q@K fp32; quantize in the backend")
        if not 0 < chunk_size <= lm.cfg.glimmer_window:
            raise ValueError("chunk_size must be in [1, glimmer_window]")
        self.lm = lm
        self.chunk_size = chunk_size
        self.reference_dynamic_qk = reference_dynamic_qk
        self.int8_kcache = int8_kcache
        if selected_rows < 0:
            raise ValueError("selected_rows must be non-negative")
        self.selected_rows = selected_rows
        cfg = lm.cfg
        self.head_token_ids = None
        if head_token_ids is not None:
            ids = torch.as_tensor(
                head_token_ids, dtype=torch.long,
                device=lm.head.weight.device)
            if (ids.ndim != 1 or ids.numel() == 0
                    or int(ids.min()) < 0 or int(ids.max()) >= cfg.vocab_size
                    or ids.unique().numel() != ids.numel()):
                raise ValueError(
                    "head_token_ids must be a non-empty unique 1-D subset of "
                    "the model vocabulary")
            if selected_rows == 0:
                raise ValueError("a pruned head requires selected_rows > 0")
            if selected_rows > ids.numel():
                raise ValueError("selected_rows exceeds the pruned head width")
            self.register_buffer(
                "head_table",
                lm.head.weight.detach().index_select(0, ids).clone(),
            )
            self.head_token_ids = tuple(int(value) for value in ids)
        else:
            self.register_buffer("head_table", None)
        self.cache_context = cfg.seq_len if cache_context is None else cache_context
        if not cfg.glimmer_window <= self.cache_context <= cfg.seq_len:
            raise ValueError("cache_context must be between window and seq_len")
        for li, block in enumerate(lm.blocks):
            cache_len = (cfg.glimmer_window if block.attn.local
                         else self.cache_context)
            _register_kv(self, li, cfg, cache_len, int8_kcache)

    def forward(self, tokens: torch.Tensor, start_pos: torch.Tensor,
                head_rows: torch.Tensor | None = None) -> torch.Tensor:
        lm, cfg, C = self.lm, self.lm.cfg, self.chunk_size
        H, KH, Dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        repeats = H // KH
        positions = start_pos + torch.arange(C, device=tokens.device)

        x = lm.tok_emb_norm(lm.tok_emb(tokens))                # [C, D]
        ple = None
        if cfg.uses_per_layer:
            ple = lm.ple_model_proj(x) * (cfg.d_model**-0.5)
            ple = lm.ple_proj_norm(ple.view(C, cfg.n_layers, cfg.ple_dim))
            if cfg.arm == "ple":
                table = lm.ple_table(tokens).view(C, cfg.n_layers, cfg.ple_dim)
                ple = (ple + table * (cfg.ple_dim**0.5)) * (2**-0.5)

        cos = lm.cos.index_select(0, positions)                # [C, Dh/2]
        sin = lm.sin.index_select(0, positions)
        neg_inf = torch.full((), float("-inf"), dtype=torch.float32,
                             device=tokens.device)

        for li, block in enumerate(lm.blocks):
            h = block.input_norm(x)
            q = block.attn.q_proj(h).view(C, H, Dh)
            k = block.attn.k_proj(h).view(C, KH, Dh)
            v = block.attn.v_proj(h).view(C, KH, Dh)
            q = block.attn.qk_norm(q) * cfg.qk_scale_factor
            k = block.attn.qk_norm(k)

            if block.attn.local:
                # Use the sequence model's exact [B,H,T,D] spelling so the
                # RoPE fusion covers prefill as well as decode.
                q4 = q.transpose(0, 1).unsqueeze(0)
                k4 = k.transpose(0, 1).unsqueeze(0)
                if self.reference_dynamic_qk:
                    # The device's espressif::rope arithmetic on the same
                    # [1, H, C, Dh] rows, the tables' first C positions.
                    rows = q4.shape[2]
                    q = _device_rope(q4, cos[:rows], sin[:rows]).squeeze(0).transpose(0, 1)
                    k = _device_rope(k4, cos[:rows], sin[:rows]).squeeze(0).transpose(0, 1)
                else:
                    q = apply_rope(q4, cos, sin).squeeze(0).transpose(0, 1)
                    k = apply_rope(k4, cos, sin).squeeze(0).transpose(0, 1)

            kh = getattr(self, f"kcache_{li}")
            vh = getattr(self, f"vcache_{li}")
            cache_len = kh.shape[1]
            cache_slot = torch.arange(cache_len, device=tokens.device)
            if block.attn.local:
                # Keep the remainder dividend non-negative.  PyTorch defines
                # remainder using floor division, but ExecuTorch's portable
                # integer kernel on ESP32 currently follows C/C++ truncating
                # `%` for a negative dividend.  The tempting spelling
                #
                #   last - remainder(last - cache_slot, cache_len)
                #
                # therefore labels several empty ring slots as positions 0, 1,
                # ... during the first chunk on device.  Adding one complete
                # ring before taking remainder is algebraically identical and
                # never presents that backend-sensitive case.
                last = start_pos - 1
                distance = torch.remainder(
                    last + cache_len - cache_slot, cache_len)
                old_pos = last - distance
            else:
                old_pos = cache_slot

            knew = k.transpose(0, 1)                           # [KH, C, Dh]
            vnew = v.transpose(0, 1)
            if self.int8_kcache:
                # The block's keys are quantised once, before attention, and
                # these same codes are what the commit below stores: the old
                # rows were quantised identically when they were written.
                ks = getattr(self, f"kcache_scale_{li}")
                kcodes, kscales = QUANTIZE_ROWS_I8(knew)     # [KH, C, Dh], [KH, C]
                kall = torch.cat((kh, kcodes), dim=1)
                kall_scales = torch.cat((ks, kscales), dim=1)
            else:
                kall = torch.cat((kh, knew), dim=1)
            vall = torch.cat((vh, vnew), dim=1)
            source_pos = torch.cat((old_pos, positions))

            query_pos = positions[:, None]
            if block.attn.local:
                valid = ((source_pos[None, :] >= 0)
                         & (source_pos[None, :] <= query_pos)
                         & ((query_pos - source_pos[None, :]) < cache_len))
            else:
                old_valid = (old_pos < start_pos)[None, :].expand(C, cache_len)
                new_valid = positions[None, :] <= query_pos
                valid = torch.cat((old_valid, new_valid), dim=1)

            # KV head is the BMM batch; token and repeated query-head axes are M.
            qg = (q.view(C, KH, repeats, Dh).permute(1, 0, 2, 3)
                    .reshape(KH, C * repeats, Dh))
            if self.int8_kcache and self.reference_dynamic_qk:
                scores = cached_i8_qk(qg, kall, kall_scales) * (Dh**-0.5)
            elif self.int8_kcache:
                keys = dequantized_keys(kall, kall_scales)
                scores = torch.bmm(qg, keys.transpose(1, 2)) * (Dh**-0.5)
            elif self.reference_dynamic_qk:
                scores = dynamic_i8_qk(qg, kall) * (Dh**-0.5)
            else:
                scores = torch.bmm(qg, kall.transpose(1, 2)) * (Dh**-0.5)
            scores = (scores.view(KH, C, repeats, cache_len + C)
                      .permute(1, 0, 2, 3).reshape(C, H, cache_len + C))
            scores = torch.where(valid[:, None, :], scores, neg_inf)
            if self.reference_dynamic_qk:
                probs = _portable_softmax(scores).to(q.dtype)
            else:
                probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            pg = (probs.view(C, KH, repeats, cache_len + C)
                  .permute(1, 0, 2, 3).reshape(KH, C * repeats, cache_len + C))
            if self.reference_dynamic_qk:
                out = _portable_bmm(pg, vall).view(KH, C, repeats, Dh)
            else:
                out = torch.bmm(pg, vall).view(KH, C, repeats, Dh)
            out = out.permute(1, 0, 2, 3).reshape(C, cfg.d_model)

            # Commit only after every query consumed the pre-block cache.
            slots = (torch.remainder(positions, cache_len)
                     if block.attn.local else positions)
            if self.int8_kcache:
                kh.index_copy_(1, slots, kcodes)
                ks.index_copy_(1, slots, kscales)
            else:
                kh.index_copy_(1, slots, knew)
            vh.index_copy_(1, slots, vnew)

            out = out * torch.sigmoid(block.attn.gate_proj(h))
            x = x + block.post_attn_norm(block.attn.o_proj(out))
            x = x + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(x)))
            if ple is not None:
                g = F.gelu(block.ple_gate(x))
                x = x + block.ple_norm(block.ple_proj(g * ple[:, li]))

        # Prefill needs only the next-token distribution after the final prompt
        # token; avoiding C full-vocabulary head rows matters more than any cache.
        x = lm.out_norm(x[-1:])
        if self.selected_rows:
            if head_rows is None:
                raise ValueError("selected-head prefill requires head_rows")
            table = (self.head_table if self.head_table is not None
                     else lm.head.weight)
            rows = torch.index_select(table, 0, head_rows)
            logits = F.linear(x, rows)
        else:
            logits = lm.head(x)
        logits = logits * cfg.output_multiplier
        cap = cfg.final_logit_softcapping
        if cap > 0:
            logits = torch.tanh(logits / cap) * cap
        return logits


def _shared_lm(lm, cfg):
    """Seed-0 TinyLM for cfg, or the given one after checking it is that shape.

    Passing ``lm`` lets an exporter that loads a checkpoint build the decode
    and prefill wrappers around one model: each fresh TinyLM costs a random
    initialisation (~0.5 s at the story Glimmer's 29M parameters) that
    load_state_dict then overwrites. The wrappers keep their KV state on
    themselves and never modify the model, so sharing it is safe; the caller
    owns whatever it does to the model afterwards (it is seen by both).
    """
    if lm is None:
        torch.manual_seed(0)
        return TinyLM(cfg).eval()
    if lm.cfg != cfg:
        raise ValueError(f"shared lm has config {lm.cfg}, expected {cfg}")
    return lm


def build(*, arm="baseline", vocab=256, d_model=64, layers=4, heads=8,
          kv_heads=1, ffn=192, ple_dim=24, seq_len=16, window=8,
          softcap=20.0, reference_dynamic_qk=False, local_ring=False,
          cache_context=None, selected_rows=0, head_token_ids=None,
          int8_kcache=False, lm=None):
    cfg = Config(
        architecture="glimmer", arm=arm, vocab_size=vocab,
        d_model=d_model, n_layers=layers, n_heads=heads,
        n_kv_heads=kv_heads, ffn_hidden=ffn, ple_dim=ple_dim,
        seq_len=seq_len, glimmer_window=window,
        final_logit_softcapping=softcap,
    )
    model = GlimmerDecodeStep(
        _shared_lm(lm, cfg), reference_dynamic_qk=reference_dynamic_qk,
        local_ring=local_ring, cache_context=cache_context,
        selected_rows=selected_rows,
        head_token_ids=head_token_ids, int8_kcache=int8_kcache).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return cfg, model


def build_prefill(*, chunk_size=4, arm="baseline", vocab=256, d_model=64,
                  layers=4, heads=8, kv_heads=1, ffn=192, ple_dim=24,
                  seq_len=16, window=8, softcap=20.0,
                  reference_dynamic_qk=False, cache_context=None,
                  selected_rows=0, head_token_ids=None, int8_kcache=False,
                  lm=None):
    cfg = Config(
        architecture="glimmer", arm=arm, vocab_size=vocab,
        d_model=d_model, n_layers=layers, n_heads=heads,
        n_kv_heads=kv_heads, ffn_hidden=ffn, ple_dim=ple_dim,
        seq_len=seq_len, glimmer_window=window,
        final_logit_softcapping=softcap,
    )
    model = GlimmerPrefillChunk(
        _shared_lm(lm, cfg), chunk_size,
        reference_dynamic_qk=reference_dynamic_qk,
        cache_context=cache_context,
        selected_rows=selected_rows,
        head_token_ids=head_token_ids,
        int8_kcache=int8_kcache,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return cfg, model


def _replace_glimmer_tables_i4(model):
    """Use trained-format i4 pass-through for both sparse Glimmer tables."""
    from executorch.backends.esp32s3.utils.pass_through import PTEmbedding, QTensor

    converted = set()
    for name in ("tok_emb", "ple_table"):
        module = getattr(model.lm, name)
        weight = module.weight.detach().numpy().astype(np.float32)
        group = min(SCALE_GROUP, weight.shape[1])
        codes, scales = quantize_absmax_rows(
            weight, "i4", group=group, fp16_scales=True)
        if scales.ndim == 1:
            scales = scales[:, None]
        setattr(model.lm, name, PTEmbedding(QTensor(codes, scales, group)))
        converted.add(f"lm.{name}")
    return model, converted
