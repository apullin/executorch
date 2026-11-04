# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch's own LLM recipes on this backend: the stock quantised K/V cache
and the stock quantised embeddings.

ExecuTorch's llama export (examples/models/llama, export_llama) quantises two
things the delegate consumes as they are:

  the K/V cache   source_transformation/custom_kv_cache.py's QuantizedKVCache
                  (`--quantize_kv_cache`): every new key and value row is
                  quantised per token and head with XNNPACK's affine chooser
                  (design A4, the quantise in float64), written into int8 code,
                  fp32 scale and int8 zero-point buffers by llama.update_cache,
                  and read back through quantized_decomposed.dequantize_per_token;
  the embeddings  source_transformation/quantize.py's QuantizedGroupEmbedding
                  (`--embedding_quantize 8,0 | 4,32 | 2,32`):
                  quantized_decomposed.embedding_{byte,4bit,2bit}.dtype.

What reaches the edge graph, per layer, from export_llama's own flow (strict
export, run_decompositions({}), static shapes, one decode token):

  writer   choose_qparams_per_token_asymmetric(x) -> (scale f64, zp i64),
           quantize_per_token(x, scale, zp, -128, 127, int8), scale -> f32,
           zp -> int8, then three auto_functionalized llama.update_cache into
           the code, scale and zero-point buffers -- for K and again for V;
  K read   dequantize_per_token(codes, scales -> f64, zps -> i64), a fourth
           update_cache putting the token just written back in as its FLOAT
           value, transpose, repeat_interleave (GQA), and
           scaled_dot_product_attention's math decomposition: q * D^-1/4 and
           k^T * D^-1/4 into one bmm, + mask, softmax;
  V read   the same dequant and float override, then p @ v.

`prepare(edge)` runs right after to_edge (lower_esp32), before the
backend does anything else, and makes four rewrites, each only where its whole
pattern is present:

  1. writers   choose + quantise + both casts become
               espressif::quantize_per_token_xnn(x) -> (codes, scales, zero
               points): torch's own ops are its reference body, and the device
               repeats them exactly (A4: the zero-point test
               fl(fl(qmin + dmin) + fl(qmax + dmax)) > 0, then
               rne(fl64(fl64(x * fl64(1/s)) + z)) and the clamp, in float64).
               ExecuTorch's portable per-token chooser agrees with neither
               (op_choose_qparams.cpp), so a chooser must never run portable;
               strip_dynamic_act_qdq's final sweep refuses any left behind.
  2. reads     every int8 dequantize_per_token over cast scales and zero points
               becomes espressif::dequantize_per_token_i8(codes, scales, zps):
               torch's value bit for bit -- the float64 product (q - z) * s of
               an integer below 2^8 and a float32 is exact, so its one rounding
               to float32 is fl((q - z) * s) -- with no float64 or int64
               tensors left on the device. This is how V is read; p @ v stays
               portable fp32.
  3. the K side  q @ k^T over a K read becomes
               bmm(q', permute(dequantize_per_token_i8(codes', scales',
               zps'), (0, 2, 1))) over [B*H, S, D], which the partitioner claims
               as OP_BMM_KCACHE_AFFINE_I8: the delegate dots the stored codes
               with its own symmetric query rows. Three numerics choices come
               with it, and mirrors.kcache_scores is what the device then
               computes:
                 - every row is read from its codes, the one written this step
                   too, not taken as its float value (llama.custom_quantized_
                   sdpa, ExecuTorch's own fused consumer of this cache, reads
                   it from codes as well);
                 - SDPA's key scale moves onto the query, q' = fl(q * c) with
                   c = fl(s_q * s_k) (exact at D = 16 and 256, where s is a
                   power of two);
                 - GQA reads each KV head once: the R query heads sharing it
                   form one batch of R*M rows instead of R copies of the keys.
               Shapes the delegate cannot take (symbolic, a head past the dot
               bound) are left as they were, on the exact portable path.
  4. embeddings  embedding_{byte,4bit,2bit} become embedding(
               dequantize_per_channel_group(codes, ...)), their own
               CompositeExplicitAutograd bodies, with packed codes unpacked at
               export exactly as those bodies unpack them. ExecuTorch's 4-bit
               table puts the EVEN element in the HIGH nibble and the
               backend's packer writes it LOW, so the unpack-and-repack is the
               nibble swap; 2-bit codes in [-2, 1] are stored as int4, exactly.
               The pass-through embedding then consumes the codes, their
               scales and any zero points (integer, integral float, or
               fractional float: fl(fl(q - z) * s)).

In a program with the stock cache, the llama's own position gathers
(mask[input_pos], the RoPE tables' freqs[input_pos]: index.Tensor with one 1-D
index on dim 0) become index_select, the same rows: the firmware carries
index_select's kernel and not index.Tensor's (3 KB of app image).

Refused by name (StockRecipeRefused), never left to fail at Method::init:
float16/bfloat16 embedding scales (eager multiplies in that dtype, ExecuTorch's
kernel in float32 -- cast the model to float32 as export_llama does); torchao's
_embedding_Nbit (it needs the torchao kernel library at export);
llama.custom_sdpa and llama.custom_quantized_sdpa (ExecuTorch's fused
attention kernels, not in this runtime: export without use_sdpa_with_kv_cache).
After partitioning, `check_claimed` refuses a rewritten K-cache bmm or
embedding the partitioner did not claim ([kcache-unclaimed],
[embedding-unclaimed]).

StaticQuantizedKVCache (`--static_quantize_kv_cache`) needs none of this: its
quantise and dequantise are plain ATen ops (true division, round, clamp,
casts, one multiply), exact as portable kernels, around llama.update_cache. Its
per-head-dim key scales lie along the q @ k^T contraction, which no per-row
K-cache opcode expresses, so its attention stays portable fp32.

The custom ops' device kernels are ops/kv_affine.cpp, with
llama::update_cache.out for the cache writes.
"""

import operator

import torch
from torch.library import Library

# FRAGMENT: the "espressif" namespace is shared with kcache_i8,
# fuse_rmsnorm_pass, cached_attention, fold_quant_lut_pass and the other custom
# ops.
lib = Library("espressif", "FRAGMENT")
lib.define("quantize_per_token_xnn(Tensor x) -> "
           "(Tensor codes, Tensor scales, Tensor zero_points)")
lib.define("quantize_per_token_xnn.out(Tensor x, *, Tensor(a!) codes, "
           "Tensor(b!) scales, Tensor(c!) zero_points) -> "
           "(Tensor(a!), Tensor(b!), Tensor(c!))")
lib.define("dequantize_per_token_i8(Tensor codes, Tensor scales, "
           "Tensor zero_points) -> Tensor")
lib.define("dequantize_per_token_i8.out(Tensor codes, Tensor scales, "
           "Tensor zero_points, *, Tensor(a!) out) -> Tensor(a!)")

impl = Library("espressif", "IMPL")


def _qd():
    """torch.ops.quantized_decomposed, registered (the namespace exists only
    once torch.ao.quantization.fx._decomposed is imported)."""
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    return torch.ops.quantized_decomposed


def quantize_per_token_xnn(x):
    """x[..., D] float32 -> (codes int8 [..., D], scales float32 [..., 1],
    zero points int8 [..., 1]): exactly what QuantizedKVCache._quantize and
    its two casts compute, with the very same ops."""
    qd = _qd()
    scales, zero_points = qd.choose_qparams_per_token_asymmetric.default(
        x, torch.int8)
    codes = qd.quantize_per_token.default(x, scales, zero_points, -128, 127,
                                          torch.int8)
    return codes, scales.to(torch.float32), zero_points.to(torch.int8)


def _quantize_per_token_xnn_out(x, *, codes, scales, zero_points):
    c, s, z = quantize_per_token_xnn(x)
    codes.copy_(c)
    scales.copy_(s)
    zero_points.copy_(z)
    return codes, scales, zero_points


def dequantize_per_token_i8(codes, scales, zero_points):
    """codes int8 [..., D], scales float32 [..., 1], zero points int8
    [..., 1] -> float32 [..., D]: QuantizedKVCache's read, the same op over the
    same casts."""
    return _qd().dequantize_per_token.default(
        codes, scales.to(torch.float64), zero_points.to(torch.int64), -128, 127,
        torch.int8, torch.float32)


def _dequantize_per_token_i8_out(codes, scales, zero_points, *, out):
    out.copy_(dequantize_per_token_i8(codes, scales, zero_points))
    return out


impl.impl("quantize_per_token_xnn", quantize_per_token_xnn,
          "CompositeExplicitAutograd")
impl.impl("quantize_per_token_xnn.out", _quantize_per_token_xnn_out,
          "CompositeExplicitAutograd")
impl.impl("dequantize_per_token_i8", dequantize_per_token_i8,
          "CompositeExplicitAutograd")
impl.impl("dequantize_per_token_i8.out", _dequantize_per_token_i8_out,
          "CompositeExplicitAutograd")


@torch.library.register_fake("espressif::quantize_per_token_xnn")
def _quantize_per_token_xnn_fake(x):
    side = (*x.shape[:-1], 1)
    return (torch.empty(x.shape, dtype=torch.int8, device=x.device),
            torch.empty(side, dtype=torch.float32, device=x.device),
            torch.empty(side, dtype=torch.int8, device=x.device))


@torch.library.register_fake("espressif::dequantize_per_token_i8")
def _dequantize_per_token_i8_fake(codes, scales, zero_points):
    return torch.empty(codes.shape, dtype=torch.float32, device=codes.device)


# The K-cache opcode's exactness bound on a key row (esp32_asym.cpp
# kKcacheQuerySpan * kKcacheKeySpan: A0 queries |q| <= 127 against key codes
# with a zero point, |k - z| <= 255): the dot runs over k_pad.
KCACHE_DOT_BOUND = 66_304


class StockRecipeRefused(RuntimeError):
    """A piece of an ExecuTorch LLM recipe this backend refuses, by rule.

    Raised at export so the program never reaches a device that would fail at
    Method::init with a missing kernel instead."""

    def __init__(self, rule: str, where: str, detail: str):
        self.rule = rule
        super().__init__(f"esp32: stock LLM recipe at {where} refused "
                         f"[{rule}]: {detail}")


# ---- matching ----------------------------------------------------------------

def _edge():
    from executorch.exir.dialects._ops import ops as exir_ops

    return exir_ops.edge


def _overload(node):
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        _overload as overload,
    )

    return overload(node)


def _schema(node):
    """("namespace::name", overload) of an op node, else (None, None)."""
    op = _overload(node)
    return (op._schema.name, op._overloadname) if op is not None else (None, None)


def _is(node, name, overload=None):
    got, ov = _schema(node)
    return got == name and (overload is None or ov == overload)


def _val(node):
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else None


def _static(*nodes):
    """Every node's value is a tensor of concrete shape (no SymInt)."""
    return all(isinstance(_val(n), torch.Tensor)
               and all(type(d) is int for d in _val(n).shape) for n in nodes)


def _cast_source(node, dtype):
    """x when `node` casts x to `dtype` and nothing else (no layout change),
    else None."""
    e = _edge()
    casts = [e.aten._to_copy.default]
    to_dim_order = getattr(getattr(e.dim_order_ops, "_to_dim_order_copy", None),
                           "default", None)
    if to_dim_order is not None:
        casts.append(to_dim_order)
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target in casts and len(node.args) == 1):
        return None
    if node.kwargs.get("dtype") != dtype:
        return None
    order = node.kwargs.get("dim_order")
    v = _val(node.args[0])
    if v is None or (order is not None and list(order) != list(range(v.dim()))):
        return None
    if node.kwargs.get("memory_format") not in (None, torch.preserve_format,
                                               torch.contiguous_format):
        return None
    return node.args[0]


def _getitems(node):
    """{index: getitem node} over a multi-output node's users, or None when a
    user is anything but a getitem."""
    out = {}
    for u in node.users:
        if not (u.op == "call_function" and u.target is operator.getitem):
            return None
        out[u.args[1]] = u
    return out


def _new(graph, target, args, like, val, kwargs=None):
    node = graph.call_function(target, tuple(args), kwargs or {})
    node.meta = {k: v for k, v in like.meta.items()
                 if k in ("stack_trace", "nn_module_stack", "source_fn_stack",
                          "from_node", "seq_nr", "torch_fn")}
    node.meta["val"] = val
    return node


def _replace(ep, old, new):
    """old's uses become new's, the program's output signature included (a
    graph output is named by its node)."""
    old.replace_all_uses_with(new)
    for spec in ep.graph_signature.output_specs:
        if getattr(spec.arg, "name", None) == old.name:
            spec.arg.name = new.name


def _erase_dead(graph, nodes):
    """Erase each node once nothing reads it, users before producers."""
    changed = True
    nodes = [n for n in nodes if n is not None]
    while changed:
        changed = False
        for n in list(nodes):
            if n.graph is graph and not n.users:
                graph.erase_node(n)
                nodes.remove(n)
                changed = True


# ---- 1. writers ----------------------------------------------------------------

def _writer(chooser):
    """QuantizedKVCache._quantize and its two casts, from the chooser:
    (x, quantize, scale cast, zero-point cast, getitems), or None."""
    if not _is(chooser, "quantized_decomposed::choose_qparams_per_token_asymmetric"):
        return None
    if len(chooser.args) < 2 or chooser.args[1] != torch.int8 or chooser.kwargs:
        return None
    x = chooser.args[0]
    xv = _val(x)
    if not isinstance(xv, torch.Tensor) or xv.dtype != torch.float32:
        return None
    items = _getitems(chooser)
    if items is None or set(items) != {0, 1}:
        return None
    scale, zp = items[0], items[1]
    quants = [u for u in scale.users if _is(u, "quantized_decomposed::quantize_per_token")]
    if len(quants) != 1:
        return None
    quant = quants[0]
    if (list(quant.args) != [x, scale, zp, -128, 127, torch.int8]
            or quant.kwargs):
        return None
    rest_s = [u for u in scale.users if u is not quant]
    rest_z = [u for u in zp.users if u is not quant]
    if (len(rest_s) != 1 or len(rest_z) != 1
            or _cast_source(rest_s[0], torch.float32) is not scale
            or _cast_source(rest_z[0], torch.int8) is not zp):
        return None
    return x, quant, rest_s[0], rest_z[0], (scale, zp)


def absorb_writers(ep) -> int:
    graph = ep.graph_module.graph
    target = _edge().espressif.quantize_per_token_xnn.default
    count = 0
    for chooser in list(graph.nodes):
        parts = _writer(chooser)
        if parts is None:
            continue
        x, quant, scale_cast, zp_cast, items = parts
        codes_v, scale_v, zp_v = _val(quant), _val(scale_cast), _val(zp_cast)
        with graph.inserting_before(quant):
            w = _new(graph, target, (x,), quant, (codes_v, scale_v, zp_v))
            outs = [_new(graph, operator.getitem, (w, i), quant, v)
                    for i, v in enumerate((codes_v, scale_v, zp_v))]
        _replace(ep, quant, outs[0])
        _replace(ep, scale_cast, outs[1])
        _replace(ep, zp_cast, outs[2])
        _erase_dead(graph, [quant, scale_cast, zp_cast, *items, chooser])
        count += 1
    return count


# ---- 2. reads ------------------------------------------------------------------

def _read(dq):
    """(codes, scales f32, zero points int8) of an int8 dequantize_per_token
    over cast scales and zero points, else None."""
    if not _is(dq, "quantized_decomposed::dequantize_per_token"):
        return None
    args = list(dq.args)
    out_dtype = args[6] if len(args) > 6 else dq.kwargs.get("output_dtype",
                                                           torch.float32)
    if len(args) < 6 or args[3:6] != [-128, 127, torch.int8]:
        return None
    if out_dtype not in (torch.float32, None):
        return None
    codes, s64, z64 = args[:3]
    scales = _cast_source(s64, torch.float64)
    zps = _cast_source(z64, torch.int64)
    cv, sv, zv = _val(codes), _val(scales), _val(zps)
    if (any(not isinstance(v, torch.Tensor) for v in (cv, sv, zv))
            or cv.dtype != torch.int8 or sv.dtype != torch.float32
            or zv.dtype != torch.int8 or cv.dim() < 1
            or list(sv.shape) != [*cv.shape[:-1], 1]
            or list(zv.shape) != list(sv.shape)):
        return None
    return codes, scales, zps, (s64, z64)


def absorb_reads(ep) -> int:
    graph = ep.graph_module.graph
    target = _edge().espressif.dequantize_per_token_i8.default
    count = 0
    for dq in list(graph.nodes):
        parts = _read(dq)
        if parts is None:
            continue
        codes, scales, zps, casts = parts
        with graph.inserting_before(dq):
            new = _new(graph, target, (codes, scales, zps), dq, _val(dq))
        _replace(ep, dq, new)
        _erase_dead(graph, [dq, *casts])
        count += 1
    return count


# ---- 3. the K side ---------------------------------------------------------------

def _single(node):
    return isinstance(node, torch.fx.Node) and len(node.users) == 1


def _step(node, target, users=1):
    """node.args[0] when node calls `target` and has `users` users."""
    if (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target is target and len(node.users) == users):
        return node.args[0]
    return None


def _identity_expand(node):
    """Step over an expand_copy / _clone_dim_order that keeps shape and order."""
    e = _edge()
    clone = getattr(getattr(e.dim_order_ops, "_clone_dim_order", None),
                    "default", None)
    while (_single(node) and node.op == "call_function"
           and node.target in (e.aten.expand_copy.default, clone)):
        src = node.args[0]
        if _val(src) is None or list(_val(src).shape) != list(_val(node).shape):
            break
        if node.target is clone and node.kwargs.get("dim_order") not in (
                None, list(range(_val(node).dim()))):
            break
        node = src
    return node


def _scaled(node, ep):
    """(source, scalar) through an optional multiply by one number -- SDPA's
    decomposition writes mul.Scalar, a module's own `x * self.scale` a
    mul.Tensor by a lifted one-element constant -- else (node, None)."""
    e = _edge()
    if not (_single(node) and node.op == "call_function"):
        return node, None
    if node.target is e.aten.mul.Scalar and isinstance(node.args[1], (int, float)):
        return node.args[0], float(node.args[1])
    if node.target is e.aten.mul.Tensor and len(node.args) == 2:
        for x, c in (node.args, node.args[::-1]):
            if not (isinstance(c, torch.fx.Node) and c.op == "placeholder"):
                continue
            real = _const(ep, c)
            xv = _val(x)
            if (real is not None and real.numel() == 1 and real.is_floating_point()
                    and xv is not None and list(_val(node).shape) == list(xv.shape)):
                return x, float(real.reshape(-1)[0])
    return node, None


def _k_read(node):
    """The K read under a transposed key operand: from KT = the [B, H, S, D]
    transpose of the cache read, back through the float override to
    dequantize_per_token_i8. (dequant, override node or None, override getitem
    or None)."""
    e = _edge()
    if not (_single(node) and node.op == "call_function"
            and node.target is e.aten.permute_copy.default
            and list(node.args[1]) == [0, 2, 1, 3]):
        return None
    src = node.args[0]
    override = item = None
    if (_single(src) and src.op == "call_function"
            and src.target is operator.getitem and src.args[1] == 1):
        af = src.args[0]
        if not (af.op == "call_function"
                and "auto_functionalized" in str(af.target)
                and af.args and "llama.update_cache" in str(af.args[0])
                and len(af.users) == 1):
            return None
        bases = af.kwargs.get("_all_bases")
        if not (isinstance(bases, (list, tuple)) and len(bases) == 1
                and af.kwargs.get("_cache_base_index") == 0):
            return None
        override, item, src = af, src, bases[0]
    if not (_single(src) and src.op == "call_function"
            and src.target is e.espressif.dequantize_per_token_i8.default):
        return None
    return src, override, item, node


def _gqa(node):
    """(KT, R, nodes) through repeat_interleave's spelling
    view(clone(expand(unsqueeze(KT, 2), [B,H,R,S,D])), [B,H*R,S,D]); R = 1
    and no nodes when there is none."""
    e = _edge()
    view = node
    if not (_single(view) and view.op == "call_function"
            and view.target is e.aten.view_copy.default):
        return node, 1, []
    src = view.args[0]
    clone = getattr(getattr(e.dim_order_ops, "_clone_dim_order", None),
                    "default", None)
    chain = [view]
    if _single(src) and src.op == "call_function" and src.target is clone:
        chain.append(src)
        src = src.args[0]
    if not (_single(src) and src.op == "call_function"
            and src.target is e.aten.expand_copy.default):
        return node, 1, []
    expand = src
    unsq = expand.args[0]
    if not (_single(unsq) and unsq.op == "call_function"
            and unsq.target is e.aten.unsqueeze_copy.default and unsq.args[1] == 2):
        return node, 1, []
    kt = unsq.args[0]
    kv, ev, vv = _val(kt), _val(expand), _val(view)
    if kv is None or ev is None or vv is None or kv.dim() != 4:
        return node, 1, []
    b, h, s, d = (int(x) for x in kv.shape)
    r = int(ev.shape[2])
    if (list(ev.shape) != [b, h, r, s, d]
            or list(vv.shape) != [b, h * r, s, d]):
        return node, 1, []
    return kt, r, [*chain, expand, unsq]


def match_kcache_bmm(bmm, ep):
    """The q @ k^T over a stock K read, as the rewrite needs it, or None."""
    e = _edge()
    if not (bmm.op == "call_function" and bmm.target is e.aten.bmm.default):
        return None
    qv, kv = bmm.args[:2]
    # Key side: view(expand?(mul?(permute(KR, (0,1,3,2)), s_k)), [B*Hq, D, S]).
    if not (_single(kv) and kv.target is e.aten.view_copy.default):
        return None
    k_nodes = [kv]
    node = _identity_expand(kv.args[0])
    if node is not kv.args[0]:
        k_nodes.append(kv.args[0])
    src, s_k = _scaled(node, ep)
    if s_k is not None:
        k_nodes.append(node)
    if not (_single(src) and src.op == "call_function"
            and src.target is e.aten.permute_copy.default
            and list(src.args[1]) == [0, 1, 3, 2]):
        return None
    k_nodes.append(src)
    kt, r, gqa_nodes = _gqa(src.args[0])
    k_nodes += gqa_nodes
    read = _k_read(kt)
    if read is None:
        return None
    dq, override, item, kt_node = read
    k_nodes.append(kt_node)
    # Query side: view(expand?(mul?(Q, s_q)), [B*Hq, M, D]).
    if not (_single(qv) and qv.target is e.aten.view_copy.default):
        return None
    q_nodes = [qv]
    node = _identity_expand(qv.args[0])
    if node is not qv.args[0]:
        q_nodes.append(qv.args[0])
    q, s_q = _scaled(node, ep)
    if s_q is not None:
        q_nodes.append(node)
    codes, scales, zps = dq.args
    cv, qval, bv = _val(codes), _val(q), _val(bmm)
    if cv is None or qval is None or bv is None or cv.dim() != 4 or qval.dim() != 4:
        return None
    b, s, h, d = (int(x) for x in cv.shape)
    bq, hq, m, dq_ = (int(x) for x in qval.shape)
    if (bq != b or dq_ != d or hq != h * r or qval.dtype != torch.float32
            or list(bv.shape) != [b * hq, m, s]):
        return None
    return dict(bmm=bmm, q=q, s_q=s_q, s_k=s_k, dq=dq, codes=codes,
                scales=scales, zps=zps, override=override, item=item,
                b=b, s=s, h=h, d=d, r=r, m=m, k_nodes=k_nodes, q_nodes=q_nodes)


def _query_scale(s_q, s_k):
    """The one multiplier the rewritten query takes: fl32(s_q * s_k), each a
    float32 value first (portable mul.Scalar casts its scalar to float)."""
    import numpy as np

    if s_q is None and s_k is None:
        return None
    c = np.float32(1.0)
    for s in (s_q, s_k):
        if s is not None:
            c = np.float32(c * np.float32(s))
    return float(c)


def rewrite_kcache(ep, report) -> int:
    graph = ep.graph_module.graph
    e = _edge()
    count = 0
    for bmm in list(graph.nodes):
        got = match_kcache_bmm(bmm, ep)
        if got is None:
            continue
        b, s, h, d, r, m = (got[k] for k in "bshdrm")
        # The delegate quantises the query itself, so it must be a plain
        # float value (operator_support/qk_operands.py): a query a recipe
        # quantised keeps the recipe's portable arithmetic, as the gate
        # would leave it, rather than a rewrite nothing claims.
        from executorch.backends.esp32s3.operator_support import qk_operands

        query = qk_operands.origin(got["q"], ep)
        if query.kind != qk_operands.PLAIN:
            report["kcache_left"].append(
                (bmm.name, f"its query is a {query.kind} ({query.detail})"))
            continue
        # The bound is on k_pad, the head padded to whole 16-byte vectors.
        if not _static(got["codes"], got["q"], bmm):
            report["kcache_left"].append((bmm.name, "a symbolic shape"))
            continue
        if -(-d // 16) * 16 > KCACHE_DOT_BOUND:
            report["kcache_left"].append(
                (bmm.name, f"head width {d} pads past the exact-dot bound "
                           f"{KCACHE_DOT_BOUND}"))
            continue
        bh = b * h
        codes, scales, zps, q = got["codes"], got["scales"], got["zps"], got["q"]
        cv, sv, zv, qv = _val(codes), _val(scales), _val(zps), _val(q)
        with graph.inserting_before(bmm):
            def to3(t, tv, last):
                """[B, S, H, last] -> [B*H, S, last]: a view for one head, a
                transpose then a view for several."""
                if h > 1:
                    pv = tv.permute(0, 2, 1, 3).contiguous()
                    t = _new(graph, e.aten.permute_copy.default, (t, [0, 2, 1, 3]),
                             bmm, pv)
                    tv = pv
                return _new(graph, e.aten.view_copy.default, (t, [bh, s, last]),
                            bmm, tv.reshape(bh, s, last))

            c3, s3, z3 = to3(codes, cv, d), to3(scales, sv, 1), to3(zps, zv, 1)
            k3v = torch.empty_like(_val(c3), dtype=torch.float32)
            k3 = _new(graph, e.espressif.dequantize_per_token_i8.default,
                      (c3, s3, z3), got["dq"], k3v)
            kt = _new(graph, e.aten.permute_copy.default, (k3, [0, 2, 1]), bmm,
                      k3v.permute(0, 2, 1).contiguous())
            scale = _query_scale(got["s_q"], got["s_k"])
            qs = q
            if scale is not None:
                qs = _new(graph, e.aten.mul.Scalar, (q, scale), bmm,
                          torch.empty_like(qv))
            q3 = _new(graph, e.aten.view_copy.default, (qs, [bh, r * m, d]), bmm,
                      qv.reshape(bh, r * m, d))
            # Every value here is a FakeTensor made from the graph's own:
            # to_backend inserts a partition's getitem only for one.
            out3 = _new(graph, e.aten.bmm.default, (q3, kt), bmm,
                        qv.new_empty((bh, r * m, s)))
            out3.meta["esp32_kcache_affine"] = True
            out = _new(graph, e.aten.view_copy.default, (out3, [b * h * r, m, s]),
                       bmm, _val(bmm))
        _replace(ep, bmm, out)
        consts = {a for n in (*got["k_nodes"], *got["q_nodes"]) for a in n.args
                  if isinstance(a, torch.fx.Node) and a.op == "placeholder"}
        _erase_dead(graph, [bmm, *got["k_nodes"], *got["q_nodes"], got["item"],
                            got["override"], got["dq"]])
        from executorch.backends.transforms.utils import delete_constant_placeholder

        for c in consts:
            if not c.users and c.name in ep.graph_signature.inputs_to_lifted_tensor_constants:
                delete_constant_placeholder(ep, c)
        report["kcache"].append(out3.name)
        count += 1
    return count


# ---- 4. embeddings -----------------------------------------------------------------

_EMBEDDINGS = {
    "quantized_decomposed::embedding_byte": 8,
    "quantized_decomposed::embedding_4bit": 4,
    "quantized_decomposed::embedding_2bit": 2,
}


def _const(ep, node):
    """The real tensor behind a parameter/buffer/constant placeholder."""
    sig = ep.graph_signature
    name = (sig.inputs_to_parameters.get(node.name)
            or sig.inputs_to_buffers.get(node.name)
            or sig.inputs_to_lifted_tensor_constants.get(node.name))
    if name is None:
        return None
    if name in ep.state_dict:
        return ep.state_dict[name]
    return ep.constants.get(name)


def unpack_codes(weight, bits):
    """The signed codes embedding_{byte,4bit,2bit}'s own bodies unpack
    (exir/passes/_quant_patterns_and_replacements.py), with their very ops:
    4-bit even = high nibble, 2-bit from the low bits up, each - 2^(bits-1)."""
    if bits == 8:
        return weight
    if bits == 4:
        even = weight.div(16, rounding_mode="trunc")
        odd = weight.remainder(16)
        unpacked = torch.stack((even, odd), dim=-1)
        return unpacked.view(weight.shape[0], -1).view(torch.int8).add(-8)
    parts = [weight & 3, (weight & 12) >> 2, (weight & 48) >> 4,
             (weight & 192) >> 6]
    unpacked = torch.stack(parts, dim=-1)
    return unpacked.view(weight.shape[0], -1).view(torch.int8).add(-2)


def _fold_constant_gather(ep, graph, node, report):
    """An embedding_Nbit whose table AND indices are program constants (GPT-2's
    position table read by its position ids) needs no kernel at all: its value
    is a constant, computed here by the op itself on the real tensors -- so
    exact by construction -- and stored as a lifted constant. A gather of
    constants by constants is constant propagation's job (the partitioner
    refuses to claim one); True when folded."""
    from executorch.backends.transforms.utils import (
        create_constant_placeholder,
        delete_constant_placeholder,
    )
    from torch.export.graph_signature import InputKind

    tensors = [a for a in node.args if isinstance(a, torch.fx.Node)]
    if not all(a.op == "placeholder" for a in tensors):
        return False
    real = {a: _const(ep, a) for a in tensors}
    if any(v is None for v in real.values()):
        return False
    args = [real.get(a, a) if isinstance(a, torch.fx.Node) else a
            for a in node.args]
    with torch.no_grad():
        value = _overload(node)(*args, **node.kwargs)
    first = next(n for n in graph.nodes if n.op == "placeholder")
    with graph.inserting_before(first):
        const = create_constant_placeholder(
            ep, graph, f"{node.name}_folded", InputKind.CONSTANT_TENSOR,
            value.contiguous())
    _replace(ep, node, const)
    graph.erase_node(node)
    for a in tensors:
        if not a.users:
            delete_constant_placeholder(ep, a)
    report["embeddings_folded"].append(node.name)
    return True


def rewrite_embeddings(ep, report) -> int:
    from executorch.backends.transforms.utils import (
        create_constant_placeholder,
        delete_constant_placeholder,
    )
    from torch.export.graph_signature import InputKind

    graph = ep.graph_module.graph
    e = _edge()
    count = 0
    for node in list(graph.nodes):
        name, overload = _schema(node)
        if name == "torchao::_embedding_Nbit" or (
                name is not None and name.startswith("torchao::")
                and "embedding" in name):
            raise StockRecipeRefused(
                "torchao-embedding", node.name,
                f"{name} needs the torchao kernel library at export and has "
                "no ExecuTorch kernel; quantise embeddings with export_llama's "
                "legacy --embedding_quantize (quantized_decomposed.embedding_*)")
        bits = _EMBEDDINGS.get(name)
        if bits is None:
            continue
        weight, scales, zps, qmin, qmax, indices = node.args[:6]
        if _fold_constant_gather(ep, graph, node, report):
            continue
        sv = _val(scales)
        out_dtype = (node.kwargs.get("dtype") if overload == "dtype"
                     else (sv.dtype if sv is not None else None))
        if sv is None or sv.dtype != torch.float32:
            raise StockRecipeRefused(
                "embedding-scale-dtype", node.name,
                f"{name} with {getattr(sv, 'dtype', None)} scales: eager "
                "dequantises in the scales' dtype, ExecuTorch's kernel in "
                "float32, so neither value is the other's; cast the model to "
                "float32 after quantising, as export_llama does")
        if out_dtype not in (torch.float32,):
            raise StockRecipeRefused(
                "embedding-dtype", node.name,
                f"{name} producing {out_dtype}; the delegate gathers float32")
        if zps is not None and (_val(zps) is None
                                or _val(zps).dtype != torch.float32):
            raise StockRecipeRefused(
                "embedding-zero-point-dtype", node.name,
                f"{name} zero points must share the float32 scales' dtype")
        real = _const(ep, weight) if weight.op == "placeholder" else None
        if real is None:
            raise StockRecipeRefused(
                "embedding-table", node.name,
                f"{name}'s table is not a constant of the program")
        codes_t = unpack_codes(real, bits)
        rows, cols = (int(x) for x in codes_t.shape)
        groups = int(sv.shape[1]) if sv.dim() == 2 else 1
        if cols % groups:
            raise StockRecipeRefused(
                "embedding-groups", node.name,
                f"{groups} scale groups do not divide {cols} columns")
        gsize = cols // groups
        codes = weight
        if bits != 8:
            first = next(n for n in graph.nodes if n.op == "placeholder")
            with graph.inserting_before(first):
                codes = create_constant_placeholder(
                    ep, graph, f"{weight.name}_codes{bits}",
                    InputKind.CONSTANT_TENSOR, codes_t.contiguous())
        with graph.inserting_before(node):
            fake = _val(node)
            dq_val = fake.new_empty((rows, cols))
            dq = _new(graph, e.quantized_decomposed.dequantize_per_channel_group.default,
                      (codes, scales, zps, int(qmin), int(qmax), codes_t.dtype,
                       gsize, torch.float32), node, dq_val)
            emb = _new(graph, e.aten.embedding.default, (dq, indices), node, fake)
        emb.meta["esp32_stock_embedding"] = name
        _replace(ep, node, emb)
        graph.erase_node(node)
        if bits != 8 and not weight.users:
            delete_constant_placeholder(ep, weight)
        report["embeddings"].append((emb.name, name, bits))
        count += 1
    return count


# ---- the entry points ---------------------------------------------------------------

def _refuse_custom_sdpa(graph):
    for node in graph.nodes:
        name, _ = _schema(node)
        if name in ("llama::custom_sdpa", "llama::custom_quantized_sdpa",
                    "llama::sdpa_with_kv_cache"):
            raise StockRecipeRefused(
                "custom-sdpa", node.name,
                f"{name} is ExecuTorch's fused attention kernel "
                "(extension/llm/custom_ops), which this runtime does not "
                "have, and its queries are quantised per token (A4) where the "
                "delegate's K-cache op keeps symmetric query rows; export "
                "without use_sdpa_with_kv_cache to delegate the cache")


def canonicalize_position_gathers(ep, report) -> int:
    """ExecuTorch's llama at static shapes gathers its mask and RoPE tables by
    position, mask[input_pos], spelled index.Tensor(x, [input_pos]): one 1-D
    int64 index on dim 0, which is exactly index_select(x, 0, input_pos) --
    the same rows, the same bytes. The firmware already carries
    index_select's kernel; index.Tensor's would cost 3 KB of app image for
    these gathers alone. Only in programs with the stock cache (prepare
    calls this once it has absorbed a writer)."""
    graph = ep.graph_module.graph
    e = _edge()
    count = 0
    for node in list(graph.nodes):
        if not (node.op == "call_function"
                and node.target is e.aten.index.Tensor and len(node.args) == 2):
            continue
        x, indices = node.args
        if not (isinstance(indices, (list, tuple)) and len(indices) == 1
                and isinstance(indices[0], torch.fx.Node)):
            continue
        iv, xv = _val(indices[0]), _val(x)
        if (iv is None or xv is None or iv.dtype != torch.int64 or iv.dim() != 1
                or xv.dim() < 1):
            continue
        with graph.inserting_before(node):
            new = _new(graph, e.aten.index_select.default, (x, 0, indices[0]),
                       node, _val(node))
        _replace(ep, node, new)
        graph.erase_node(node)
        count += 1
    report["position_gathers"] = count
    return count


def new_report():
    return {"writers": 0, "reads": 0, "kcache": [], "kcache_left": [],
            "embeddings": [], "embeddings_folded": [], "position_gathers": 0}


def prepare(edge_manager, method_name="forward", kcache=True,
            skip_reason="the int8 query is opt-in (quantize_qk)"):
    """The four rewrites on one method of an edge program, before anything
    else touches it. Returns a report; a program with none of the stock ops
    is left untouched (and an empty report).

    `kcache` is whether the K side may become OP_BMM_KCACHE_AFFINE_I8, whose
    int8 query rows are opt-in like every int8 q@kT (Esp32Partitioner's
    quantize_qk, and not under format_policy='fp32'). Without it the K
    cache's q @ k^T is left as the recipe wrote it, read through the exact
    dequantize_per_token_i8 like V, and runs portable in fp32: each match
    goes in report["kcache_left"] with `skip_reason` (lower_esp32 names
    format_policy='fp32' when that is why)."""
    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    report = new_report()
    if not any(_schema(n)[0] is not None and (
            _schema(n)[0].startswith(("quantized_decomposed::", "llama::",
                                      "torchao::_embedding")))
               for n in graph.nodes):
        return report
    _refuse_custom_sdpa(graph)
    report["writers"] = absorb_writers(ep)
    report["reads"] = absorb_reads(ep)
    if report["writers"]:
        canonicalize_position_gathers(ep, report)
    if kcache:
        rewrite_kcache(ep, report)
    else:
        for bmm in graph.nodes:
            if match_kcache_bmm(bmm, ep) is not None:
                report["kcache_left"].append((bmm.name, skip_reason))
    rewrite_embeddings(ep, report)
    if (report["writers"] or report["reads"] or report["kcache"]
            or report["embeddings"] or report["embeddings_folded"]
            or report["position_gathers"]):
        graph.eliminate_dead_code()
        graph.lint()
        ep.graph_module.recompile()
    return report


def check_claimed(graph_module):
    """After partitioning: every rewritten K-cache bmm and stock embedding
    must have gone into a delegate. Left portable, the K-cache rewrite would
    have changed the numerics for nothing, and the embedding's dequant has no
    kernel at all."""
    for node in graph_module.graph.nodes:
        # A delegate's outputs are getitems that inherit the claimed node's
        # meta: only the op itself, still in the top-level graph, is unclaimed.
        if node.op != "call_function" or node.target is operator.getitem:
            continue
        if node.meta.get("esp32_kcache_affine"):
            raise StockRecipeRefused(
                "kcache-unclaimed", node.name,
                "the stock K cache's q @ k^T was rewritten for "
                "OP_BMM_KCACHE_AFFINE_I8, but the partitioner did not claim it")
        if node.meta.get("esp32_stock_embedding"):
            raise StockRecipeRefused(
                "embedding-unclaimed", node.name,
                f"{node.meta['esp32_stock_embedding']} was rewritten for the "
                "pass-through embedding, but the partitioner did not claim it")
