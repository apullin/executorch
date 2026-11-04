# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch's own LLM recipes on the device: the parity rows of stock_llm.py,
their references, and the packed QEMU image.

Every row is built from ExecuTorch's own modules (examples/models/llama) with
random weights, lowered through the backend with nothing left portable but
the plumbing each row names, and run as a SEQUENCE: one program, its mutable
buffers zeroed once, executed `steps` times with step-dependent inputs (main.cpp
run_stock_llm_fixtures, and step_float / step_long below are its rules). Rows:

  kv_gqa_seq, kv_mqa_seq   QuantizedKVCache, the class export_llama's
          --quantize_kv_cache installs (affine per token, llama.update_cache),
          read the way SDPA's math decomposition reads it -- repeat_interleave,
          q * D^-1/4 against k^T * D^-1/4 -- returning the scores and the read
          V cache: two KV heads x two query heads each at D = 16 (16 rows),
          and one KV head x four at D = 24 (8 rows: ragged, k_pad 32). Each
          step writes one K and V row and scores against every row, so the
          cache holds rows of differing scales and zero points (step 2 writes
          all-zero rows, the eps floor; step 3 constant rows; step 4 rows
          whose every value is an exact rounding tie, under an even zero point
          for K and an odd one for V). Exact: the
          writer, OP_BMM_KCACHE_AFFINE_I8 and the read against
          mirrors.kcache_write / kcache_scores / vcache_dequant.
  emb_*   quantized_decomposed.embedding_{byte,4bit,2bit}: export_llama's
          QuantizedGroupEmbedding (no zero points, 8 / 4 at group 32 / 2 at
          group 32), and byte and 4-bit tables with integral and fractional
          float zero points (the ops' own signature, as ExecuTorch's
          quantisation fusion emits them for float-qparams embeddings). Exact
          against mirrors.embedding_dequant and the ops themselves.
  nano_llama_stock   the full recipe: a nano llama (2 layers, width 64, 4
          query and 2 KV heads, vocab 256) with --embedding_quantize 4,32,
          -X / 8da4w at group 32, --quantize_kv_cache, in export_llama's
          transform order and its strict-export flow; 32 teacher-forced decode
          steps. Held to eager at TOL, with max-abs-diff and top-1 agreement
          reported: the K cache's symmetric query rows and the key just
          written being read from its codes are the delegate's arithmetic,
          not eager's (stock_llm.py).

Image (little-endian): "STKF", u32 version, u32 count, then per row 48 B:
char name[24], u32 offset, u32 size, u32 steps, u32 vocab, u32 flags
(FLAG_POSITION: the last int64 input is input_pos, the step index), u32
reserved; then the programs, 16-byte aligned. Partition stockllm (0x50).
"""

import contextlib
import io
import os
import struct
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
F32 = np.float32

IMAGE = "stock_llm_fixtures.bin"
FIXTURES_MAGIC = b"STKF"
FIXTURES_VERSION = 1
ENTRY = struct.Struct("<24sIIIIII")
FLAG_POSITION = 1

ROWS = ("kv_gqa_seq", "kv_mqa_seq", "emb_byte", "emb_4bit", "emb_2bit",
        "emb_byte_zp", "emb_byte_fzp", "emb_4bit_zp", "emb_4bit_fzp",
        "nano_llama_stock")
TOL_ROWS = ("nano_llama_stock",)
EXACT_ROWS = tuple(r for r in ROWS if r not in TOL_ROWS)
LLAMA_STEPS = 32


# ---- main.cpp's input rules ------------------------------------------------------

def step_float(shape, step, index):
    """Float input `index` at `step` (main.cpp stock_step_float): all zeros at
    step 2, all 0.75 at step 3; at step 4 exact A4 rounding ties -- every 16
    values hold lo = -128/64 and hi = 127/64 (index 2: -127/64 and 128/64), so
    the row's scale is 2^-6 and its zero point 0 (index 2: -1), and the rest
    are ((37 j mod 254) - 127 + 1/2) / 64, so x / s + z = m + z + 1/2 exactly
    and round-half-to-even goes either way; else element j is

        (((37 j + 101 step + 13 index) mod (17 + 6 (step mod 5)))
         - (3 + 2 (step mod 7))) * 2^-4

    exact in float32 under any contraction (a small integer times a power of
    two), and a different range at every step."""
    n = int(np.prod(shape))
    if step == 2:
        vals = np.zeros(n, dtype=F32)
    elif step == 3:
        vals = np.full(n, F32(0.75), dtype=F32)
    elif step == 4:
        j = np.arange(n, dtype=np.int64)
        lo = -127 if index == 2 else -128
        half = ((((37 * j) % 254) - 127).astype(F32) + F32(0.5)) * F32(0.015625)
        vals = np.where(j % 16 == 0, F32(lo) * F32(0.015625),
                        np.where(j % 16 == 1, F32(lo + 255) * F32(0.015625),
                                 half)).astype(F32)
    else:
        j = np.arange(n, dtype=np.int64)
        m = 17 + 6 * (step % 5)
        vals = (((37 * j + 101 * step + 13 * index) % m) - (3 + 2 * (step % 7))
                ).astype(F32) * F32(0.0625)
    return torch.from_numpy(vals.astype(F32)).reshape(shape)


def step_long(shape, step, vocab):
    """An int64 input that is not input_pos: element j is (7 j + 3 + 37 step)
    mod vocab (main.cpp stock_step_long)."""
    n = int(np.prod(shape))
    return torch.tensor([(7 * j + 3 + 37 * step) % vocab for j in range(n)],
                        dtype=torch.int64).reshape(shape)


# ---- the modules ---------------------------------------------------------------------

def _stock_ops():
    """llama.update_cache (ExecuTorch's custom-ops AOT library) and the
    quantized_decomposed .out variants to_executorch needs."""
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import torch.ao.quantization.fx._decomposed  # noqa: F401


class KvScores(nn.Module):
    """ExecuTorch's QuantizedKVCache as replace_kv_cache_with_quantized_kv_cache
    builds it, read the way SDPA's math decomposition reads it; returns the
    scores q @ k^T (before the mask and softmax) and the read V cache."""

    def __init__(self, heads, rep, dim, seq):
        super().__init__()
        _stock_ops()
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (  # noqa: E501
            QuantizedCacheType,
            QuantizedKVCache,
        )

        self.kv_cache = QuantizedKVCache(
            1, seq, heads, dim, QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op=True)
        self.heads, self.rep, self.dim, self.seq = heads, rep, dim, seq
        self.scale = dim ** -0.25

    def forward(self, q, k, v, input_pos):
        k_out, v_out = self.kv_cache.update(input_pos, k, v)
        k_out = k_out.repeat_interleave(self.rep, dim=1)
        scores = (q * self.scale) @ (k_out.transpose(-2, -1) * self.scale)
        return scores, v_out

    def step_inputs(self, step):
        hq = self.heads * self.rep
        return (step_float((1, hq, 1, self.dim), step, 0),
                step_float((1, self.heads, 1, self.dim), step, 1),
                step_float((1, self.heads, 1, self.dim), step, 2),
                torch.tensor([step], dtype=torch.int64))


def kv_reference(net, steps):
    """What the device computes at each step: [(scores, v_out)] float32."""
    from executorch.backends.esp32s3._passes import stock_llm

    h, r, d, s = net.heads, net.rep, net.dim, net.seq
    # Mutable buffers start at zero on the device (ExecuTorch does not store
    # their initial values): codes, scales and zero points all 0.
    kc = np.zeros((s, h, d), np.int8)
    ks = np.zeros((s, h), F32)
    kz = np.zeros((s, h), np.int8)
    vc, vs, vz = kc.copy(), ks.copy(), kz.copy()
    c = F32(stock_llm._query_scale(net.scale, net.scale))
    out = []
    for t in range(steps):
        q, k, v, _ = (x.numpy() for x in net.step_inputs(t))
        codes, scales, zps = qm.kcache_write(k[0, :, 0, :])
        kc[t], ks[t], kz[t] = codes, scales, zps
        codes, scales, zps = qm.kcache_write(v[0, :, 0, :])
        vc[t], vs[t], vz[t] = codes, scales, zps
        qs = (q * c).astype(F32)
        scores = np.zeros((1, h * r, 1, s), F32)
        for hq in range(h * r):
            scores[0, hq] = qm.kcache_scores(qs[0, hq], kc[:, hq // r, :],
                                             ks[:, hq // r], kz[:, hq // r])
        v_out = np.stack([qm.vcache_dequant(vc[:, j, :], vs[:, j], vz[:, j])
                          for j in range(h)])[None]
        v_out[0, :, t, :] = v[0, :, 0, :]     # the token just written, as given
        out.append((scores, v_out))
    return out


class StockEmbedding(nn.Module):
    """export_llama's --embedding_quantize BITS,GROUP on an nn.Embedding:
    QuantizedGroupEmbedding, then the model cast to float32 as export_llama
    casts it after its quantisation transforms."""

    def __init__(self, vocab, dim, bits, group):
        super().__init__()
        from executorch.examples.models.llama.source_transformation.quantize import (  # noqa: E501
            get_quant_embedding_transform,
        )

        self.tok_embeddings = nn.Embedding(vocab, dim)
        with contextlib.redirect_stdout(io.StringIO()):
            get_quant_embedding_transform(f"{bits},{group}")(self)
        self.to(dtype=torch.float32)
        self.vocab = vocab

    def forward(self, tokens):
        return self.tok_embeddings(tokens)

    def table(self):
        e = self.tok_embeddings
        return e.weight, e.scales, None


class EmbeddingOp(nn.Module):
    """quantized_decomposed.embedding_{byte,4bit}.dtype over constant tables,
    with float zero points: the signature ExecuTorch's quantisation fusion
    emits for float-qparams embeddings."""

    def __init__(self, bits, weight, scales, zero_points, qmin, qmax):
        super().__init__()
        _stock_ops()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)
        self.bits, self.qmin, self.qmax = bits, qmin, qmax
        self.vocab = int(weight.shape[0])

    def forward(self, tokens):
        qd = torch.ops.quantized_decomposed
        op = qd.embedding_byte.dtype if self.bits == 8 else qd.embedding_4bit.dtype
        return op(self.weight, self.scales, self.zero_points, self.qmin,
                  self.qmax, tokens, dtype=torch.float32)

    def table(self):
        return self.weight, self.scales, self.zero_points


def embedding_reference(net, tokens):
    """mirrors.embedding_dequant on the table's signed codes."""
    from executorch.backends.esp32s3._passes import stock_llm

    weight, scales, zps = net.table()
    bits = (net.tok_embeddings.bitwidth if isinstance(net, StockEmbedding)
            else net.bits)
    codes = stock_llm.unpack_codes(weight, bits).to(torch.int64)
    return qm.embedding_dequant(codes.numpy(), scales.numpy(),
                                None if zps is None else zps.numpy(),
                                tokens.numpy())


def nano_llama(seed=0):
    """The full stock recipe on a nano ExecuTorch llama, in export_llama's
    transform order (_get_source_transforms), ending with its float32 cast."""
    _stock_ops()
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.source_transformation import (
        custom_kv_cache as ckv,
        quantize as qz,
        sdpa,
    )

    args = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     vocab_size=256, hidden_dim=128, max_seq_len=LLAMA_STEPS,
                     max_context_len=LLAMA_STEPS, use_kv_cache=True,
                     enable_dynamic_shape=False, max_batch_size=1)
    torch.manual_seed(seed)
    model = construct_transformer(args).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        for transform in (qz.get_quant_embedding_transform("4,32"),
                          qz.get_quant_weight_transform("8da4w", group_size=32),
                          ckv.replace_kv_cache_with_quantized_kv_cache,
                          sdpa.replace_sdpa_with_quantized_sdpa,
                          lambda m: m.to(dtype=torch.float32)):
            model = transform(model)
    model.vocab = args.vocab_size
    return model


def llama_inputs(step, vocab):
    return (step_long((1, 1), step, vocab),
            {"input_pos": torch.tensor([step], dtype=torch.int64)})


def llama_eager(model, steps):
    """Eager logits per step, every MUTATED buffer starting at zero as on the
    device (ExecuTorch stores no initial value for one): a cache's codes, and
    QuantizedKVCache's per-token scales and zero points -- not
    StaticQuantizedKVCache's scales, which are the model's constants."""
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (  # noqa: E501
        QuantizedKVCache,
    )

    with torch.no_grad():
        for module in model.modules():
            names = ("k_cache", "v_cache")
            if isinstance(module, QuantizedKVCache):
                names += ("k_cache_scales", "v_cache_scales",
                          "k_cache_zero_points", "v_cache_zero_points")
            for name in names:
                buffer = getattr(module, name, None)
                if isinstance(buffer, torch.Tensor):
                    buffer.zero_()
        return [model(*llama_inputs(t, model.vocab)).numpy().astype(F32)
                for t in range(steps)]


def llm_export(model, inputs):
    """export_llama's own export (LLMEdgeManager._export, twice: export, then
    the re-export to_edge_transform_and_lower makes): strict, under
    sdpa_kernel(MATH) and no_grad, run_decompositions({})."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        ep = torch.export.export(model, inputs, strict=True).run_decompositions({})
        return torch.export.export(ep.module(), inputs,
                                   strict=True).run_decompositions({})


# ---- the fixture table ----------------------------------------------------------------

@dataclass
class Fixture:
    name: str
    model: nn.Module
    inputs: tuple          # the step-0 inputs, for export
    steps: int
    vocab: int = 0
    flags: int = 0
    references: list = field(default_factory=list)  # per step, a tuple of arrays
    portable: tuple = ()   # op names the row keeps outside the delegate
    strict: bool = True    # nothing else may stay portable (lower checks)


# The KV rows keep the cache's own plumbing portable: the writes
# (llama.update_cache), the writer and the V read (espressif's custom ops),
# input_pos to a SymInt, the q scale and the reshapes around the delegate.
KV_PORTABLE = ("update_cache", "auto_functionalized", "quantize_per_token_xnn",
               "dequantize_per_token_i8", "_local_scalar_dense", "select_copy",
               "_assert_scalar", "mul.Scalar", "view_copy", "permute_copy",
               "unsqueeze_copy", "expand_copy", "_clone_dim_order", "ge", "le",
               "lt")


def _embedding_fixtures():
    out = []
    g = torch.Generator().manual_seed(8)
    for name, bits, group in (("emb_byte", 8, 0), ("emb_4bit", 4, 32),
                              ("emb_2bit", 2, 32)):
        torch.manual_seed(bits)
        net = StockEmbedding(32, 64, bits, group).eval()
        out.append((name, net))
    for name, bits, fractional in (("emb_byte_zp", 8, False),
                                   ("emb_byte_fzp", 8, True),
                                   ("emb_4bit_zp", 4, False),
                                   ("emb_4bit_fzp", 4, True)):
        vocab, dim, group = 32, 64, 32
        weight = torch.randint(0, 256, (vocab, dim if bits == 8 else dim // 2),
                               generator=g, dtype=torch.int32).to(torch.uint8)
        scales = (torch.rand((vocab, dim // group), generator=g) * 0.02
                  + 0.001).float()
        if bits == 8:
            zps = torch.randint(96, 160, (vocab, dim // group), generator=g)
            qmin, qmax = 0, 255
        else:
            zps = torch.randint(-3, 4, (vocab, dim // group), generator=g)
            qmin, qmax = -8, 7
        zps = zps.float()
        if fractional:
            zps = zps + (torch.rand(zps.shape, generator=g) - 0.5).float()
        out.append((name, EmbeddingOp(bits, weight, scales, zps, qmin,
                                      qmax).eval()))
    return out


def fixtures():
    out = []
    for name, (heads, rep, dim, seq, steps) in (
            ("kv_gqa_seq", (2, 2, 16, 16, 6)), ("kv_mqa_seq", (1, 4, 24, 8, 8))):
        torch.manual_seed(0)
        net = KvScores(heads, rep, dim, seq).eval()
        out.append(Fixture(name, net, net.step_inputs(0), steps,
                           flags=FLAG_POSITION,
                           references=kv_reference(net, steps),
                           portable=KV_PORTABLE))
    for name, net in _embedding_fixtures():
        tokens = step_long((1, 6), 0, net.vocab)
        ref = embedding_reference(net, tokens)
        with torch.no_grad():
            eager = net(tokens).numpy()
        if not np.array_equal(ref.view(np.uint32), eager.view(np.uint32)):
            raise SystemExit(f"{name}: the embedding mirror differs from "
                             "the op itself")
        out.append(Fixture(name, net, (tokens,), 1, vocab=net.vocab,
                           references=[(ref,)]))
    model = nano_llama()
    out.append(Fixture("nano_llama_stock", model, llama_inputs(0, model.vocab),
                       LLAMA_STEPS, vocab=model.vocab, flags=FLAG_POSITION,
                       references=[(x,) for x in llama_eager(model, LLAMA_STEPS)],
                       strict=False))
    assert tuple(f.name for f in out) == ROWS
    return out


def lower(fixture, quantize_qk=True, format_policy=None):
    """The fixture through export_llama's export and the backend, its dynamic
    chains absorbed as every stock flow does: (ExecutorchProgramManager, log,
    stock_llm report). Nothing may stay portable but the row's plumbing.

    The rows hold OP_BMM_KCACHE_AFFINE_I8 to its mirror, so they opt in to
    its int8 query rows (quantize_qk, opt-in like every int8 q@kT); without
    it the K cache's q @ k^T stays portable in fp32."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import fold_conv_bn_pass
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3._passes import strip_dynamic_act_qdq_pass
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ep = llm_export(fixture.model, fixture.inputs)
    log = io.StringIO()
    fold = fold_conv_bn_pass.fold_conv_bn

    def strip_then_fold(edge, method_name="forward"):
        strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq(edge, method_name)
        return fold(edge, method_name)

    _passes.fold_conv_bn = strip_then_fold
    try:
        with contextlib.redirect_stdout(log):
            lowered = _passes.lower_esp32(
                ep, [Esp32Partitioner(quantize_qk=quantize_qk,
                                      format_policy=format_policy)])
    finally:
        _passes.fold_conv_bn = fold
    with contextlib.redirect_stdout(log):
        _passes.apply_passes(lowered, quiet=True)
    if fixture.strict and quantize_qk and format_policy != "fp32":
        graph = lowered.exported_program().graph_module.graph
        left = [str(n.target) for n in graph.nodes if n.op == "call_function"
                and "call_delegate" not in str(n.target)
                and "getitem" not in str(n.target)
                and not any(p in str(n.target) for p in fixture.portable)]
        if left:
            raise RuntimeError(f"{fixture.name}: ops left portable: {left}\n"
                               f"{log.getvalue()}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, log.getvalue(), lowered.stock_llm_report


def write_image(path, table, programs):
    header = len(FIXTURES_MAGIC) + 8 + ENTRY.size * len(table)
    start = -(-header // 16) * 16
    body, entries = bytearray(), []
    for f in table:
        body += bytes(-(start + len(body)) % 16)
        entries.append(ENTRY.pack(f.name.encode(), start + len(body),
                                  len(programs[f.name]), f.steps, f.vocab,
                                  f.flags, 0))
        body += programs[f.name]
    data = (FIXTURES_MAGIC + struct.pack("<II", FIXTURES_VERSION, len(table))
            + b"".join(entries))
    data += bytes(start - len(data)) + bytes(body)
    with open(path, "wb") as f:
        f.write(data)
    return data


def read_image(data):
    """[(name, program bytes, steps, vocab, flags)] from a packed image."""
    if data[:4] != FIXTURES_MAGIC:
        raise ValueError("not a stock LLM fixtures image")
    version, count = struct.unpack_from("<II", data, 4)
    if version != FIXTURES_VERSION:
        raise ValueError(f"stock LLM fixtures image version {version}")
    out = []
    for i in range(count):
        name, offset, size, steps, vocab, flags, _ = ENTRY.unpack_from(
            data, 12 + i * ENTRY.size)
        out.append((name.rstrip(b"\0").decode(), data[offset:offset + size],
                    steps, vocab, flags))
    return out


def step_inputs(fixture, step):
    """The inputs main.cpp writes at `step`, flattened in method order."""
    if isinstance(fixture.model, KvScores):
        return list(fixture.model.step_inputs(step))
    if isinstance(fixture.inputs[-1], dict):          # a llama's decode step
        tokens, kwargs = llama_inputs(step, fixture.vocab)
        return [tokens, kwargs["input_pos"]]
    return [step_long(tuple(fixture.inputs[0].shape), step, fixture.vocab)]


def export_all(out_dir=MAIN, verbose=True):
    """Every program (<name>.pte) and the QEMU image. Returns the fixture
    table, each fixture with its per-step references and `report` (what
    stock_llm.prepare rewrote in it)."""
    table = fixtures()
    programs = {}
    for fixture in table:
        et, log, report = lower(fixture)
        programs[fixture.name] = bytes(et.buffer)
        fixture.report = report
        fixture.log = log
        with open(os.path.join(out_dir, f"{fixture.name}.pte"), "wb") as f:
            f.write(programs[fixture.name])
        if verbose:
            print(f"  {fixture.name}.pte: {len(programs[fixture.name]):,} B, "
                  f"{fixture.steps} step(s); kcache {len(report['kcache'])}, "
                  f"writers {report['writers']}, reads {report['reads']}, "
                  f"embeddings {len(report['embeddings'])}")
    data = write_image(os.path.join(out_dir, IMAGE), table, programs)
    if verbose:
        print(f"  {IMAGE}: {len(data):,} B, {len(table)} row(s)")
    return table


if __name__ == "__main__":
    export_all()
