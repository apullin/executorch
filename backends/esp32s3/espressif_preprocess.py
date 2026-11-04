# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Esp32Backend: lowers a partition to the delegate blob."""

import math
import struct
from typing import List, final

import numpy as np
import torch
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram

from executorch.backends.esp32s3.operator_support.const_weights import (
    FOLDED,
    LoweringRefused,
    PassThroughWeight,
    _check_pass_through_matmul,
    _const_fqn,
    _const_tensor,
    _is_torchao_dequant,
    _resolve_const,
    _shape_of,
    _slice_pass_through,
    _w_off_member,
)
from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
    KEEP_FLOAT,
    _a16_problem,
    _affine_bounds,
    _check_pass_through_conv,
    _pad_to,
    _pass_through_matmul_layout,
    _pass_through_table_layout,
    _residual_pattern,
)
from executorch.backends.esp32s3.operator_support.patterns import (
    SUPPORTED,
    _hard_pattern,
    _kcache_affine_keys,
    _kcache_affine_member_bmm,
    _kcache_keys,
    _kcache_member_bmm,
    _qk_bmm_structure,
    _relu2_pattern,
    _selected_mm_from_member,
    _selected_mm_pattern,
    _silu_pattern,
    _softcap_pattern,
    _unary_act,
)
from executorch.backends.esp32s3.serialization.asym_records import _act_record
from executorch.backends.esp32s3.serialization.blob_writer import _BlobWriter
from executorch.backends.esp32s3.serialization.espressif_blob import (
    ACT_FLAGS,
    ARG_SCRATCH0,
    BACKEND_NAME,
    EXT_SELECTED_ROWS,
    FLAG_ACT_PER_TENSOR,
    FLAG_ACT_QUANT,
    FLAG_ADD,
    FLAG_CLAMP,
    FLAG_FP16_SCALES,
    FLAG_HEAD_SPLIT,
    FLAG_MUL,
    FLAG_SOFTCAP,
    FLAG_STAGE_I8,
    FLAG_STATIC_ACT,
    NUM_SCRATCH,
    OP_ADDMM,
    OP_BMM_DYN_I8,
    OP_BMM_KCACHE_AFFINE_I8,
    OP_BMM_KCACHE_I8,
    SELECTED_ROWS_FORMAT,
    VEC_BYTES,
)
from executorch.backends.esp32s3.serialization.weight_formats import (
    FORMATS,
    HALF_FORMATS,
    HALF_TAGS,
    SCALE_GROUP,
    choose_format,
    encode_pass_through_embedding_scales,
    half_bits,
    pack_rows,
    quantize_absmax_rows,
)


def _clamp_suffix(interval):
    import math

    if interval is None:
        return ""
    lo, hi = interval
    if lo == 0.0 and hi == math.inf:
        return "+relu"
    f = lambda v: "-inf" if v == -math.inf else ("inf" if v == math.inf else f"{v:g}")  # noqa: E731
    return f"+clamp[{f(lo)},{f(hi)}]"


def _softcap_suffix(softcap):
    if softcap is None:
        return ""
    scale, cap = softcap
    return f"+softcap[{scale:g},{cap:g}]"


# ===========================================================================
# preprocess(), as three stages. Each is a module-level function so the
# backend entry point reads top to bottom and each stage can be exercised on
# its own:
#
#   1. _collect_ops       walk the claimed partition into op records
#   2. _assign_dataflow   boundary arg indices and chain scratch slots
#   3. _BlobWriter        pack the records into the blob and its PTD store
#
# An op record is a dict. The keys the collector writes:
#   kind          "addmm" | "embedding" | "bmm_qk"
#   node          the fx node the op was claimed at
#   act           the activation input node (the indices, for an embedding)
#   m, k, n       the [M,K] x [K,N] problem (embedding: rows gathered, D, V)
#   bias          fp32 vector or None
#   weight        fp32 [K,N] ([V,D] for an embedding), or None with `pt` set
#   pt            PassThroughWeight: already-quantised codes taken verbatim
#   weight_node   the constant's node, for format overrides by name
#   conv          ConvGeom for a convolution (one record per group slice)
#   siblings      the other slices of a grouped conv, kept on its last slice
#   act2, batch   q@kT's key operand and batch count
#   act3, kcache  an int8 K cache's scales (act2 is then its codes)
#   act4, kcache_affine   the stock affine K cache's zero points (act2 its
#                 codes, act3 its scales)
#   selected_rows, selected_count   the legal-token head's ID input and count
#   out_node      the fusion tail once an epilogue is fused
#   act_fn, clamp, softcap, residual   the fused epilogues
#
# _assign_dataflow adds _in_arg, _out_arg, _chain_src, _in2_arg, _in3_arg,
# _in4_arg, _rows_arg.
# ===========================================================================


# --- stage 1: collect op records ------------------------------------------

def _fuse_into(entry, **kw):
    """Write a pointwise epilogue to an op record and its siblings.

    Only a grouped conv has siblings. For everything else this is a plain
    update, which is why it is safe to route every pointwise epilogue through
    it rather than remembering which ones can be grouped.

    `out_node` goes through here too, and must: the g ops share one output
    tensor and therefore one tail. Left on the primary alone, each sibling's
    tail would be the CONV node, whose user is the activation -- which is not
    a chained op, so the partition-output check would refuse the whole graph.
    """
    for e in (entry, *entry.get("siblings", ())):
        e.update(kw)


def _claimed(out_map, producer, what):
    """The op record whose fusion tail is `producer`, or a loud failure.

    Fusion targets are resolved by NODE IDENTITY, never by "the op appended
    last": a multi-op partition interleaves freely (the partitioner merges
    matmuls that merely share an input -- SwiGLU's gate and up arrive as one
    partition with the diamond's corners scheduled between them), so the
    last op appended is whichever matmul happened to sort earlier. A claimed
    node whose producer was never recorded means the gate and this walk
    disagree, which is a bug here and never in the model.
    """
    entry = out_map.get(producer)
    if entry is None:
        raise RuntimeError(f"{BACKEND_NAME}: {what}; gate and walk disagree")
    return entry


def _skipped_in_walk(node, edge_program):
    """Nodes the walk steps over: folded into a neighbour's record, so they
    emit nothing of their own."""
    selected_member = _selected_mm_from_member(node, edge_program)
    if (selected_member is not None
            and node in (selected_member["select"],
                         selected_member["permute"])):
        return True          # fused into the selected-row matvec
    if node.target in FOLDED:
        return True          # applied at build time, never emitted
    if _kcache_member_bmm(node) is not None:
        return True          # read in place by OP_BMM_KCACHE_I8
    if _kcache_affine_member_bmm(node) is not None:
        return True          # read in place by OP_BMM_KCACHE_AFFINE_I8
    from executorch.backends.esp32s3._passes.stage_b import IN, STAGE_B

    if node.meta.get(STAGE_B) == IN:
        return True          # its consumer reads the codes (_activation_of)
    from executorch.backends.esp32s3._passes.static_qdq import static_pair

    if static_pair(node) is not None:
        return False         # an op's output quantisation (_fuse_requant)
    # A torchao/PT2E dequant, or a float-offset chain, is absorbed into its
    # consumer as a pass-through weight.
    return _is_torchao_dequant(node) or _w_off_member(node) is not None


def _fuse_softcap(node, edge_program, out_map):
    softcap = _softcap_pattern(node, edge_program)
    if softcap is None or node not in softcap[2:6]:
        return False
    root, value, scaled, divided, tanh, final, scale, cap = softcap
    if node is not final:
        return True          # the final corner does the bookkeeping
    entry = _claimed(out_map, value, "softcap head was never claimed")
    if any(entry.get(k) is not None
           for k in ("act_fn", "clamp", "residual", "softcap")):
        raise RuntimeError(
            f"{BACKEND_NAME}: softcap must be the head's sole epilogue")
    _fuse_into(entry, softcap=(scale, cap), out_node=final)
    out_map[final] = entry
    return True


def _fuse_view(node, edge_program, out_map):
    if node.target is not exir_ops.edge.aten.view_copy.default:
        return False
    # A reshape merely renames the fusion tail; nothing is emitted.
    out_map[node] = _claimed(out_map, node.args[0],
                             "claimed view has no claimed producer")
    return True


def _fuse_silu(node, edge_program, out_map):
    # allow_conv=True: preprocess only ever sees nodes the GATE claimed, so
    # it has to recognise everything the gate does. The default here would
    # make the two disagree ("handed an unsupported op" on the sigmoid).
    pat = _silu_pattern(node, allow_conv=True)
    if pat is None or node not in (pat[1], pat[2]):
        return False
    # Only the sigmoid/mul corners short-circuit here; the matmul corner of
    # the diamond falls through to ordinary addmm handling first.
    mm, sig, mul, P = pat
    entry = _claimed(out_map, P, "silu diamond's matmul was never claimed")
    if node is mul:          # the mul corner does the bookkeeping
        _fuse_into(entry, act_fn="silu", out_node=mul)
        out_map[mul] = entry
    return True


def _fuse_hard(node, edge_program, out_map):
    hard = _hard_pattern(node, edge_program)
    if hard is None or not (node in hard["nodes"]
                            or node in hard["feeders"]):
        return False
    if node is not hard["div"]:
        return True          # the div corner does the bookkeeping
    # The gate saw FakeTensors; here the constants are real, and they go into
    # the descriptor as they are.
    if hard["offset"] is None or hard["divisor"] is None:
        raise RuntimeError(
            f"{BACKEND_NAME}: {hard['kind']}'s constants are not readable in "
            f"preprocess; gate and walk disagree")
    entry = _claimed(out_map, hard["P"],
                     f"{hard['kind']}'s matmul was never claimed")
    _fuse_into(entry, act_fn=hard["kind"],
               act_consts=(hard["offset"], hard["divisor"]),
               out_node=hard["div"])
    out_map[hard["div"]] = entry
    return True


def _fuse_relu2(node, edge_program, out_map):
    square = _relu2_pattern(node)
    if square is None or node is not square[2]:
        return False
    _mm, relu, _sq = square
    entry = _claimed(out_map, relu, "relu^2's relu was never claimed")
    # The relu arrived as a clamp interval; with the square it becomes one
    # activation, applied where the interval would have been.
    if (entry.get("clamp") != (0.0, math.inf)
            or any(entry.get(k) is not None
                   for k in ("act_fn", "softcap", "residual", "mul_operand"))
            or entry.get("mul_inplace")):
        raise RuntimeError(
            f"{BACKEND_NAME}: relu^2 over something other than a bare relu; "
            f"gate and walk disagree")
    _fuse_into(entry, clamp=None, act_fn="relu2", out_node=node)
    out_map[node] = entry
    return True


def _fuse_act(node, edge_program, out_map):
    act = _unary_act(node)
    if act is None:
        return False
    entry = _claimed(out_map, node.args[0],
                     f"claimed {act[0]}'s producer was never claimed")
    _fuse_into(entry, act_fn=act[0], slope=act[1], out_node=node)
    out_map[node] = entry
    return True


def _fuse_clamp(node, edge_program, out_map):
    from executorch.backends.esp32s3.operator_support.patterns import (
        _clamp_bounds,
        compose_clamp,
    )

    bounds = _clamp_bounds(node)
    if bounds is None:
        return False
    entry = _claimed(out_map, node.args[0],
                     "claimed clamp's producer was never claimed")
    prev = entry.get("clamp", (-math.inf, math.inf))
    _fuse_into(entry, clamp=compose_clamp(bounds, prev), out_node=node)
    out_map[node] = entry
    return True


def _fuse_residual(node, edge_program, out_map):
    if node.target is not exir_ops.edge.aten.add.Tensor:
        return False
    # allow_conv=True for the same reason as the silu walk: preprocess only
    # sees what the gate claimed.
    got = _residual_pattern(node, allow_conv=True)
    if got is None:
        raise RuntimeError(
            f"{BACKEND_NAME}: claimed add is not a fusable residual; "
            f"gate and walk disagree")
    t, r = got
    entry = _claimed(out_map, t, "residual add's matmul chain was never claimed")
    if entry.get("residual") is not None:
        raise RuntimeError(
            f"{BACKEND_NAME}: two residual adds landed on one matmul")
    if entry.get("selected_rows") is not None:
        # Mirrors the runtime, which refuses kFlagAdd on a selected-row op at
        # load. The gate refuses this add; a partition that still carries it
        # is a gate/walk disagreement.
        raise RuntimeError(
            f"{BACKEND_NAME}: residual add landed on a selected-row head; "
            f"the gate should have refused it")
    entry["residual"] = r
    entry["out_node"] = node
    out_map[node] = entry
    return True


def _fuse_mul(node, edge_program, out_map):
    """A gated multiply, as FLAG_MUL. Every other claimed mul -- the SiLU,
    hardswish and ReLU^2 corners -- was consumed earlier in the walk.

    Both factors claimed here: a gate*up pair. The LATER op multiplies, in
    place, what the earlier one wrote to the shared destination, so op order
    is the only thing that decides which projection carries the flag.
    One factor claimed, the other a partition input: an operand multiply.
    """
    if node.target is not exir_ops.edge.aten.mul.Tensor:
        return False
    args = node.args
    if not (len(args) == 2 and all(isinstance(a, torch.fx.Node)
                                   for a in args)):
        raise RuntimeError(f"{BACKEND_NAME}: claimed mul is not a gated "
                           f"multiply; gate and walk disagree")
    a, b = args
    ea, eb = out_map.get(a), out_map.get(b)

    def check(entry):
        if (entry["kind"] != "addmm" or entry.get("conv") is not None
                or entry.get("selected_rows") is not None):
            raise RuntimeError(
                f"{BACKEND_NAME}: a fused multiply needs a plain matmul "
                f"factor; gate and walk disagree")

    if ea is not None and eb is not None:
        if ea is eb:
            raise RuntimeError(f"{BACKEND_NAME}: both factors of a gated "
                               f"multiply come from one op")
        earlier, owner = sorted((ea, eb), key=lambda e: e["_seq"])
        check(earlier)
        check(owner)
        if (any(owner.get(k) is not None
                for k in ("residual", "softcap", "mul_operand"))
                or owner.get("mul_inplace") or earlier.get("mul_owner")):
            raise RuntimeError(f"{BACKEND_NAME}: gated multiply landed on an "
                               f"op that already carries an operand")
        owner["mul_inplace"] = True
        owner["out_node"] = node
        earlier["mul_owner"] = owner
        earlier["out_node"] = node
        out_map[node] = owner
        return True
    entry, operand = (ea, b) if ea is not None else (eb, a)
    if entry is None:
        raise RuntimeError(f"{BACKEND_NAME}: claimed mul has no claimed "
                           f"factor; gate and walk disagree")
    if not (operand.op == "placeholder"
            and _const_tensor(edge_program, operand) is None):
        raise RuntimeError(
            f"{BACKEND_NAME}: gated multiply's other factor {operand.name} is "
            f"not a runtime partition input; gate and walk disagree")
    check(entry)
    if (any(entry.get(k) is not None
            for k in ("residual", "softcap", "mul_operand"))
            or entry.get("mul_inplace")):
        raise RuntimeError(f"{BACKEND_NAME}: gated multiply landed on an op "
                           f"that already carries an operand")
    entry["mul_operand"] = operand
    entry["out_node"] = node
    out_map[node] = entry
    return True


def _fuse_requant(node, edge_program, out_map):
    """A static pair the gate claimed as an op's output quantisation
    (Esp32OperatorSupport._requant_problem): the quantize puts the pair's
    EXT_OUT_QUANT on the op whose tail it reads, after every epilogue fused
    so far, and the dequantise becomes the op's tail -- the value the
    partition outputs, or a chained op reads."""
    from executorch.backends.esp32s3._passes.stage_b import OUT, STAGE_B
    from executorch.backends.esp32s3._passes.static_qdq import (
        is_static_q,
        out_record,
        static_pair,
    )

    if is_static_q(node):
        entry = _claimed(out_map, node.args[0],
                         "requantize's producer was never claimed")
        if (entry["kind"] != "addmm" or entry.get("out_quant") is not None
                or any(entry.get(k) is not None
                       for k in ("residual", "mul_operand", "softcap"))
                or entry.get("mul_inplace") or entry.get("mul_owner")):
            raise RuntimeError(
                f"{BACKEND_NAME}: requantize {node.name} landed on an op it "
                f"cannot follow; gate and walk disagree")
        if node.meta.get(STAGE_B) == OUT:
            # Stage B: the op writes the quantize's int8 codes, and the codes
            # are its tail -- a partition output, or a chain slot of codes.
            _fuse_into(entry, out_quant=out_record(node, codes=True),
                       out_node=node)
        else:
            _fuse_into(entry, out_quant=out_record(node))
        out_map[node] = entry
        return True
    q = static_pair(node)
    if q is None:
        return False
    entry = _claimed(out_map, q, "requantize's dequantise has no quantize")
    _fuse_into(entry, out_node=node)
    out_map[node] = entry
    return True


# In walk order. Each returns True when it consumed the node.
_EPILOGUE_FUSIONS = (_fuse_softcap, _fuse_view, _fuse_silu, _fuse_hard,
                     _fuse_relu2, _fuse_act, _fuse_clamp, _fuse_residual,
                     _fuse_mul, _fuse_requant)


def _collect_bmm(node, ops, out_map):
    """q@kT. Structure only: the softmax that justified claiming this bmm is
    a portable op and does not travel into the partition."""
    got = _qk_bmm_structure(node)
    if got is None:
        raise RuntimeError(
            f"{BACKEND_NAME}: claimed bmm is not the q@kT pattern; "
            f"gate and walk disagree")
    A, P, X = got
    b_, m_, k_ = _shape_of(A)
    _, n_, _ = _shape_of(X)
    affine = _kcache_affine_keys(X)
    if affine is not None:
        # ExecuTorch's stock affine K cache: codes, scales and zero points.
        ops.append(dict(
            kind="bmm_qk", node=node, act=A, act2=affine["codes"],
            act3=affine["scales"], act4=affine["zps"], kcache_affine=True,
            m=m_, k=k_, n=n_, batch=b_, bias=None, weight=None,
        ))
        out_map[node] = ops[-1]
        return
    kcache = _kcache_keys(X)
    ops.append(dict(
        kind="bmm_qk", node=node, act=A,
        act2=X if kcache is None else kcache["codes"],
        act3=None if kcache is None else kcache["scales"],
        kcache=kcache is not None,
        m=m_, k=k_, n=n_, batch=b_, bias=None, weight=None,
    ))
    out_map[node] = ops[-1]


def _conv_weight(edge_program, node, geom):
    """A conv's weight as pass-through codes OR a dense fp32 tensor in
    [C_out, C_in/g, taps...] layout, and its fp32 bias vector."""
    w_node, b_node = node.args[1], node.args[2]
    w = _resolve_const(edge_program, w_node)
    if w is None:
        raise RuntimeError(
            f"{BACKEND_NAME}: conv weight must be a constant; "
            f"got {getattr(w_node, 'name', w_node)}")
    pt_full = w if isinstance(w, PassThroughWeight) else None
    w_full = None
    if pt_full is not None:
        _check_pass_through_conv(pt_full, geom)
    else:
        w_full = w.detach().float()
        if geom.transposed:
            # [C_in, C_out, ...] -> [C_out, C_in, ...] with the spatial taps
            # flipped. A build-time transform of a CONSTANT, so it costs
            # nothing at runtime and the packer sees the ordinary dense
            # layout it already handles.
            spatial = list(range(2, w_full.dim()))
            w_full = w_full.flip(spatial).transpose(0, 1).contiguous()
    if isinstance(b_node, torch.fx.Node):
        b = _resolve_const(edge_program, b_node)
        if b is None:
            raise RuntimeError(f"{BACKEND_NAME}: conv bias must be a constant")
        bias_full = b.detach().float().numpy()
    else:
        bias_full = np.zeros(geom.c_out, dtype=np.float32)
    return pt_full, w_full, bias_full


def _collect_conv(node, edge_program, ops, out_map):
    """conv as im2col: m = H_out*W_out, k = C_in*kh*kw, n = C_out, and the
    weight reshapes to exactly the [N,K] the packer takes. Handed to the SAME
    path as addmm -- transposed to [K,N] first because that is the orientation
    that path expects, and it transposes back. Round-tripping rather than
    special-casing keeps one packing implementation.

    A grouped conv becomes `groups` records, each a dense conv over its own
    channel slice -- no new gather, no new kernel, just the verified path run
    g times against sliced weights. Depthwise is deliberately NOT split
    (split_groups returns it whole): at g == C that would be C descriptors
    and C dispatches for the cheapest kernel here.
    """
    from executorch.backends.esp32s3.operator_support.conv import admit, split_groups

    geom = admit(node)
    pt_full, w_full, bias_full = _conv_weight(edge_program, node, geom)
    w_node = node.args[1]
    act, codes_in = _activation_of(node.args[0])
    subs = split_groups(geom)
    per_out = geom.c_out // max(geom.groups, 1)
    for sub in subs:
        if len(subs) > 1:
            lo = sub.group_idx * per_out
            b_slice = bias_full[lo:lo + per_out]
        else:
            lo, b_slice = 0, bias_full
        if pt_full is not None:
            # Same slicing of rows the float path does, on codes and scales;
            # the emit path packs them verbatim.
            pt_slice = (_slice_pass_through(pt_full, lo, lo + per_out)
                        if len(subs) > 1 else pt_full)
            ops.append(dict(
                kind="addmm", node=node, act=act,
                m=sub.m, k=sub.k, n=sub.n,
                bias=b_slice, weight=None, conv=sub, pt=pt_slice,
                weight_node=w_node, codes_in=codes_in,
            ))
            continue
        w_slice = w_full[lo:lo + per_out] if len(subs) > 1 else w_full
        w_nk = w_slice.reshape(sub.n, sub.k).numpy()
        ops.append(dict(
            kind="addmm", node=node, act=act,
            m=sub.m, k=sub.k, n=sub.n,
            bias=b_slice, weight=w_nk.T, conv=sub,
            weight_node=w_node, codes_in=codes_in,
        ))
    # A grouped conv has g partial outputs and no single node an epilogue
    # could attach to -- so it registers the LAST record and links the others
    # as siblings. `_fuse_into` then writes a pointwise epilogue to all of
    # them, which is exact: the groups own disjoint output channels and
    # relu/gelu/clamp/silu are functions of one element. Fusing onto the last
    # group alone would activate one slice of the output and leave the rest
    # raw.
    out_map[node] = ops[-1]
    if len(subs) > 1:
        ops[-1]["siblings"] = ops[-len(subs):-1]


def _index_count(idx_node):
    """How many rows an embedding gathers: the index tensor's element count."""
    idx_val = idx_node.meta.get("val", None)
    if idx_val is None:
        raise RuntimeError(
            f"{BACKEND_NAME}: embedding indices have no fake-tensor meta")
    n_idx = 1
    for d in idx_val.shape:
        n_idx *= int(d)
    return n_idx


def _collect_embedding(node, edge_program, ops):
    """embedding(weight[V,D], indices[...]) -> [..., D]

    Shaped into the same (m, k, n) slots the matmul path uses so one
    descriptor layout serves both: k is the row width D, n is the table
    height V, and m is how many rows get gathered. Nothing is computed -- a
    gather touches exactly one row per index -- so unlike addmm there is no
    bias, no activation quantisation and no accumulator.
    """
    w_node, idx_node = node.args[0], node.args[1]
    # The runtime reads the index tensor as int64 through an UNCHECKED
    # static_cast (ExecuTorch's const_data_ptr<T>), so an int32 index tensor
    # is not a type error on device -- it fuses adjacent indices into one
    # 64-bit value and overreads the buffer. Usually that trips the t >= V
    # bounds check loudly, but when the odd-position index is 0 (a padding
    # token, entirely realistic) the fused value passes the check and the
    # wrong rows are gathered. Refused here, where it is still a build error.
    idx_meta = (idx_node.meta.get("val", None)
                if isinstance(idx_node, torch.fx.Node) else None)
    if idx_meta is not None and idx_meta.dtype != torch.int64:
        raise RuntimeError(
            f"{BACKEND_NAME}: embedding indices must be int64 to be "
            f"delegated; got {idx_meta.dtype}. The runtime reads them as "
            f"int64 without a dtype check, so a narrower index type would "
            f"be silently reinterpreted."
        )
    table = _resolve_const(edge_program, w_node)
    if table is None:
        raise RuntimeError(
            f"{BACKEND_NAME} needs the embedding table to be a constant; "
            f"got {w_node.name} ({w_node.op})"
        )
    if isinstance(table, PassThroughWeight):
        # An already-quantised table (pass-through weights; a tied tok_emb
        # arrives as the SAME codes the head consumes).
        logical_k = (int(table.codes.shape[1])
                     if table.logical_cols is None else table.logical_cols)
        ops.append(dict(
            kind="embedding", node=node, act=idx_node,
            m=_index_count(idx_node), k=logical_k,
            n=int(table.codes.shape[0]), bias=None, weight=None,
            pt=table, weight_node=w_node,
        ))
        return
    if table.dim() != 2:
        raise RuntimeError(
            f"{BACKEND_NAME}: embedding table is {tuple(table.shape)}, "
            f"expected 2-D"
        )
    ops.append(dict(
        kind="embedding", node=node, act=idx_node,
        m=_index_count(idx_node), k=int(table.shape[1]),
        n=int(table.shape[0]), bias=None,
        weight=table.detach().float().numpy(),
        weight_node=w_node,
    ))


def _collect_selected_head(node, selected, edge_program, ops, out_map):
    """The legal-token head: the full table stays packed once, and the
    runtime streams only the rows named by a second partition input."""
    table = _resolve_const(edge_program, selected["table"])
    if table is None:
        raise RuntimeError(
            f"{BACKEND_NAME}: selected-row head needs a constant full table")
    expected = (selected["vocab"], selected["k"])
    if isinstance(table, PassThroughWeight):
        # Codes [V, K] verbatim, rows gathered by ID at execute: the table is
        # not transposed (the folded permute is of the gathered rows).
        if (table.permuted or table.logical_rows is not None
                or table.logical_cols is not None
                or tuple(table.codes.shape) != expected):
            raise LoweringRefused(
                "selected-row table layout unsupported",
                f"codes {tuple(table.codes.shape)} (permuted "
                f"{table.permuted}), expected {expected}")
        ops.append(dict(
            kind="addmm", node=node, act=selected["act"],
            m=selected["m"], k=selected["k"], n=selected["vocab"], bias=None,
            weight=None, pt=table,
            selected_rows=selected["rows"],
            selected_count=selected["selected"],
            weight_node=selected["table"],
        ))
        out_map[node] = ops[-1]
        return
    if tuple(table.shape) != expected:
        raise RuntimeError(
            f"{BACKEND_NAME}: selected-row table is "
            f"{tuple(table.shape)}, expected {expected}")
    # The ordinary packer consumes [K,N] and transposes it to contiguous
    # output rows. N remains the complete vocabulary; only execute's column
    # walk is selected dynamically.
    ops.append(dict(
        kind="addmm", node=node, act=selected["act"],
        m=selected["m"], k=selected["k"],
        n=selected["vocab"], bias=None,
        weight=table.detach().float().numpy().T,
        selected_rows=selected["rows"],
        selected_count=selected["selected"],
        weight_node=selected["table"],
    ))
    out_map[node] = ops[-1]


def _matmul_bias(edge_program, bias_n, n):
    """addmm's bias as an fp32 vector of exactly N, or None for mm."""
    if bias_n is None:
        return None
    bias = _resolve_const(edge_program, bias_n)
    if bias is None:
        raise RuntimeError(
            f"{BACKEND_NAME} needs addmm's bias to be a constant; "
            f"got {bias_n.name}"
        )
    # The runtime reads exactly bias[0..N). addmm accepts anything
    # broadcastable to [M,N], so an [M,N] bias would silently apply row 0 to
    # every row, and a 0-dim one would send the reader off the end of the
    # bias payload into the weight bytes as fp32.
    if tuple(bias.shape) != (n,):
        raise RuntimeError(
            f"{BACKEND_NAME}: addmm bias must be shape ({n},) to be "
            f"delegated; got {tuple(bias.shape)}. A broadcast bias has "
            f"no representation in the op descriptor."
        )
    return bias.detach().float().numpy()


def _activation_of(act):
    """(the tensor an op reads, whether it reads codes): a Stage B consumer
    reads the int8 codes its own dequantise (stage_b.py) dequantises -- a
    partition input, or an earlier op's codes -- not the dequantised value."""
    from executorch.backends.esp32s3._passes.stage_b import IN, STAGE_B

    if isinstance(act, torch.fx.Node) and act.meta.get(STAGE_B) == IN:
        return act.args[0], True
    return act, False


def _collect_matmul(node, edge_program, ops, out_map):
    """mm(mat1[M,K], mat2[K,N]) or addmm(bias, mat1, mat2) -> [M,N]."""
    if node.target is exir_ops.edge.aten.mm.default:
        mat1, mat2 = node.args[:2]
        bias_n = None
    else:
        bias_n, mat1, mat2 = node.args[:3]
    m, k = _shape_of(mat1)
    k2, n = _shape_of(mat2)
    if k != k2:
        raise RuntimeError(f"{node.target} inner dims disagree: {k} vs {k2}")
    weight = _resolve_const(edge_program, mat2)
    if weight is None:
        raise RuntimeError(
            f"{BACKEND_NAME} needs the weight to be a constant; "
            f"got {mat2.name} ({mat2.op})"
        )
    if isinstance(weight, PassThroughWeight):
        _check_pass_through_matmul(weight, k, n)
    bias = _matmul_bias(edge_program, bias_n, n)
    act, codes_in = _activation_of(mat1)
    if isinstance(weight, PassThroughWeight):
        ops.append(dict(
            kind="addmm", node=node, act=act, m=m, k=k, n=n,
            bias=bias, weight=None, pt=weight, weight_node=mat2,
            codes_in=codes_in,
        ))
    else:
        ops.append(dict(
            kind="addmm", node=node, act=act, m=m, k=k, n=n,
            bias=bias, weight=weight.detach().float().numpy(),
            weight_node=mat2, codes_in=codes_in,
        ))
    out_map[node] = ops[-1]


def _collect_ops(edge_program):
    """One record per op the blob will carry, in graph order."""
    ops = []
    out_map = {}   # fusion tail node -> the record it extends
    numbered = 0
    for node in edge_program.graph.nodes:
        # Each record knows its position: a gated pair's flag goes on the
        # later of its two ops.
        for seq in range(numbered, len(ops)):
            ops[seq]["_seq"] = seq
        numbered = len(ops)
        if node.op != "call_function" or _skipped_in_walk(node, edge_program):
            continue
        if any(fuse(node, edge_program, out_map) for fuse in _EPILOGUE_FUSIONS):
            continue
        if node.target is exir_ops.edge.aten.bmm.default:
            _collect_bmm(node, ops, out_map)
            continue
        if node.target is exir_ops.edge.aten.convolution.default:
            _collect_conv(node, edge_program, ops, out_map)
            continue
        if SUPPORTED.get(node.target) is None:
            raise RuntimeError(
                f"{BACKEND_NAME} was handed an unsupported op: {node.target}"
            )
        if node.target is exir_ops.edge.aten.embedding.default:
            _collect_embedding(node, edge_program, ops)
            continue
        selected = _selected_mm_pattern(node, edge_program)
        if selected is not None:
            _collect_selected_head(node, selected, edge_program, ops, out_map)
            continue
        _collect_matmul(node, edge_program, ops, out_map)
    for seq in range(numbered, len(ops)):
        ops[seq]["_seq"] = seq
    if not ops:
        raise RuntimeError(f"{BACKEND_NAME} got a partition with no ops in it")
    return ops


# --- stage 2: boundary args and chain scratch -----------------------------

class _Boundary:
    """The partition's runtime arguments: args = [activation inputs...,
    outputs...]. Constants were absorbed into the blob, so the surviving
    placeholders are exactly the runtime inputs."""

    def __init__(self, edge_program):
        placeholders = [n for n in edge_program.graph.nodes
                        if n.op == "placeholder"]
        self.act_in = [n for n in placeholders
                       if _const_tensor(edge_program, n) is None]
        out_node = next(n for n in edge_program.graph.nodes
                        if n.op == "output")
        self.outs = list(out_node.args[0])
        self.n_in = len(self.act_in)


def _resolve_activation_inputs(ops, act_in, tails):
    """Each op's activation is a partition input (an arg index now) or an
    earlier op's tail (a chain, resolved once slots are assigned). Returns
    the last reader of every chained producer, which is what slot liveness
    is measured against."""
    chain_last_reader = {}
    for idx, op in enumerate(ops):
        act = op["act"]
        if act in act_in:
            op["_in_arg"] = act_in.index(act)
            continue
        pidx = tails.get(act)
        if pidx is None or pidx >= idx:
            raise RuntimeError(
                f"{BACKEND_NAME}: {op['node'].name}'s activation {act.name} is "
                f"neither a partition input nor an earlier claimed op's tail"
            )
        if op["kind"] == "embedding":
            raise RuntimeError(
                f"{BACKEND_NAME}: an embedding cannot chain from another op; "
                f"indices are integers and never a matmul's output"
            )
        op["_chain_src"] = pidx
        chain_last_reader[pidx] = idx
    return chain_last_reader


def _assign_scratch_slots(ops, boundary, chain_last_reader, tails):
    """Every op's output: a partition output (an arg index) or an internal
    intermediate, which takes one of the delegate's scratch slots by
    liveness. Returns producer op index -> slot."""
    scratch_slot = {}
    slot_busy_until = [-1] * NUM_SCRATCH  # per slot, last reader's op idx
    for idx, op in enumerate(ops):
        tail = op.get("out_node", op["node"])
        if tail in boundary.outs:
            op["_out_arg"] = boundary.n_in + boundary.outs.index(tail)
            continue
        # The record readers chain from owns the value's slot. A grouped
        # conv's g records all write ONE tensor (the runtime requires
        # identical args across the group), and only the last is its tail. A
        # gated pair shares one destination too: its earlier half writes
        # act(gate) there, and `tails` names the owner, whose readers the slot
        # is reserved for. The first record through plans the slot, so it stays
        # live from that op on; the others find it taken and write the same
        # one. Planned per record, the earlier records would have no readers
        # and the export would fail.
        producer = tails[tail]
        if producer in scratch_slot:
            op["_out_arg"] = ARG_SCRATCH0 + scratch_slot[producer]
            continue
        # An embedding's output may ride a scratch slot like any other tail:
        # `nn.Embedding -> nn.Linear` is the first two layers of most models,
        # and refusing it here -- after the gate claimed both and the
        # partitioner grouped them -- would kill the export rather than leave
        # anything portable. The runtime's gather writes the slot,
        # sized at init from the gather's m*k rather than a matmul's m*n.
        #
        # Internal-only. Every user of the tail must be the matmul of an op
        # that chains from it -- anything else consumes a value the runtime
        # never materialises.
        readers = [j for j, o2 in enumerate(ops)
                   if o2.get("_chain_src") == producer]
        reader_nodes = {ops[j]["node"] for j in readers}
        from executorch.backends.esp32s3._passes.stage_b import IN, STAGE_B

        users = [w for u in tail.users
                 for w in (u.users if u.meta.get(STAGE_B) == IN else (u,))]
        for u in users:
            if u not in reader_nodes:
                raise RuntimeError(
                    f"{BACKEND_NAME}: {tail.name} is not a partition output "
                    f"but is consumed by {u.name}, which is not a chained op"
                )
        if not readers:
            raise RuntimeError(
                f"{BACKEND_NAME}: {tail.name} is neither a partition output "
                f"nor read by any chained op; nothing would ever see it"
            )
        free = [s for s in range(NUM_SCRATCH) if slot_busy_until[s] < idx]
        if not free:
            # More simultaneously-live intermediates than there are slots. A
            # linear chain cannot get here; a DAG that does is refused loudly
            # rather than silently mis-scheduled.
            raise RuntimeError(
                f"{BACKEND_NAME}: chain at {tail.name} needs more than "
                f"{NUM_SCRATCH} live intermediates"
            )
        s = free[0]
        scratch_slot[producer] = s
        slot_busy_until[s] = chain_last_reader[producer]
        op["_out_arg"] = ARG_SCRATCH0 + s
    return scratch_slot


def _resolve_side_inputs(ops, act_in):
    """q@kT's key operand and the selected-row head's ID vector: both must
    be partition inputs, read by arg index."""
    for op in ops:
        act2 = op.get("act2")
        if act2 is None:
            continue
        if act2 not in act_in:
            raise RuntimeError(
                f"{BACKEND_NAME}: q@kT's key operand {act2.name} must be a "
                f"partition input; a chained bmm is unimplemented"
            )
        op["_in2_arg"] = act_in.index(act2)
        act3 = op.get("act3")
        if act3 is not None:
            if act3 not in act_in:
                raise RuntimeError(
                    f"{BACKEND_NAME}: the int8 K cache's scales {act3.name} "
                    f"must be a partition input")
            op["_in3_arg"] = act_in.index(act3)
        act4 = op.get("act4")
        if act4 is not None:
            if act4 not in act_in:
                raise RuntimeError(
                    f"{BACKEND_NAME}: the affine K cache's zero points "
                    f"{act4.name} must be a partition input")
            op["_in4_arg"] = act_in.index(act4)
    for op in ops:
        rows = op.get("selected_rows")
        if rows is None:
            continue
        if rows not in act_in:
            raise RuntimeError(
                f"{BACKEND_NAME}: selected-row IDs {rows.name} must be a "
                "partition input")
        if op["_out_arg"] >= ARG_SCRATCH0:
            raise RuntimeError(
                f"{BACKEND_NAME}: selected-row output must be a partition "
                "output; chained selected heads are unimplemented")
        op["_rows_arg"] = act_in.index(rows)


def _assign_dataflow(ops, boundary):
    """Intra-partition dataflow: resolve each op's input and output to a
    boundary arg index or to a chain through another op in this partition.
    Three cases per value, in order of preference:
      1. partition boundary         -> ordinary arg index
      2. tail is ALSO an output     -> in-partition readers use its output
         arg index; reading args costs nothing, so no scratch is spent on it
      3. internal-only intermediate -> one of the delegate's scratch slots,
         assigned by liveness
    """
    tails = {}
    for idx, op in enumerate(ops):
        if op.get("mul_owner") is not None:
            continue   # a gated pair's earlier half writes its owner's tail
        tails[op.get("out_node", op["node"])] = idx
    chain_last_reader = _resolve_activation_inputs(ops, boundary.act_in, tails)
    scratch_slot = _assign_scratch_slots(ops, boundary, chain_last_reader,
                                         tails)
    for op in ops:
        src = op.get("_chain_src")
        if src is None:
            continue
        if src in scratch_slot:
            op["_in_arg"] = ARG_SCRATCH0 + scratch_slot[src]
        else:
            # The producer's tail is a partition output: read it back by index.
            op["_in_arg"] = ops[src]["_out_arg"]
    _resolve_side_inputs(ops, boundary.act_in)


# --- stage 3: the blob -----------------------------------------------------

class _Epilogue:
    """The pointwise epilogue fused onto one op: activation, clamp, softcap,
    residual. Built once per op so the flag bits, the descriptor text and the
    clamp/softcap parameters cannot disagree between the storage paths."""

    def __init__(self, op, act_in):
        self.clamp = op.get("clamp")
        self.act = op.get("act_fn")
        self.slope = op.get("slope")
        self.act_consts = op.get("act_consts")
        self.softcap = op.get("softcap")
        self.res_node = op.get("residual")
        self.mul_node = op.get("mul_operand")
        self.mul_inplace = bool(op.get("mul_inplace"))
        self.mul = self.mul_node is not None or self.mul_inplace
        if self.softcap is not None and (
                self.clamp is not None or self.act is not None
                or self.res_node is not None or op.get("conv") is not None):
            raise RuntimeError(
                f"{BACKEND_NAME}: softcap must be a plain matmul's sole epilogue")
        if self.act is not None and self.act not in ACT_FLAGS:
            raise RuntimeError(f"{BACKEND_NAME}: unknown activation {self.act}")
        if self.act == "leaky" and (self.clamp is not None
                                    or self.slope is None):
            # The slope rides clamp_lo; a clamp would need the same field.
            raise RuntimeError(
                f"{BACKEND_NAME}: leaky relu needs its slope and no clamp")
        if self.act in ("hardswish", "hardsigmoid") and (
                self.clamp is not None or self.act_consts is None):
            raise RuntimeError(
                f"{BACKEND_NAME}: {self.act} needs its constants and no clamp")
        if self.mul and (self.res_node is not None or self.softcap is not None
                         or op.get("conv") is not None):
            raise RuntimeError(
                f"{BACKEND_NAME}: a fused multiply cannot share an op with a "
                f"residual, a softcap or a convolution")
        # Fused residual: a partition input summed into the output after the
        # act/clamp epilogue. A fused multiply names its operand in the same
        # word: a partition input, or the op's own output for a gated pair.
        self.res_arg = 0
        if self.res_node is not None:
            if self.res_node not in act_in:
                raise RuntimeError(
                    f"{BACKEND_NAME}: residual {self.res_node.name} is not a "
                    f"partition input; the gate should have refused this add"
                )
            self.res_arg = act_in.index(self.res_node)
        if self.mul_node is not None:
            if self.mul_node not in act_in:
                raise RuntimeError(
                    f"{BACKEND_NAME}: multiply operand {self.mul_node.name} is "
                    f"not a partition input; the gate should have refused it")
            self.res_arg = act_in.index(self.mul_node)
        elif self.mul_inplace:
            self.res_arg = op["_out_arg"]

    def flags(self):
        flags = 0
        if self.clamp is not None:
            flags |= FLAG_CLAMP
        if self.act is not None:
            flags |= ACT_FLAGS[self.act]
        if self.softcap is not None:
            flags |= FLAG_SOFTCAP
        if self.res_node is not None:
            flags |= FLAG_ADD
        if self.mul:
            flags |= FLAG_MUL
        return flags

    def note(self):
        consts = ""
        if self.act == "leaky":
            consts = f"[{self.slope:g}]"
        elif self.act_consts is not None and self.act_consts != (3.0, 6.0):
            consts = f"[{self.act_consts[0]:g},{self.act_consts[1]:g}]"
        return (("+" + self.act if self.act else "") + consts
                + _clamp_suffix(self.clamp)
                + _softcap_suffix(self.softcap)
                + ("+res" if self.res_node is not None else "")
                + ("*in" if self.mul_node is not None else "")
                + ("*pair" if self.mul_inplace else ""))

    def bounds(self):
        """The descriptor's two float parameters: softcap's (scale, cap),
        leaky's slope, hardswish/hardsigmoid's (add, divide) constants, or the
        clamp interval; unused otherwise."""
        if self.softcap is not None:
            return self.softcap
        if self.act == "leaky":
            return (self.slope, 0.0)
        if self.act in ("hardswish", "hardsigmoid"):
            return self.act_consts
        if self.clamp is not None:
            return self.clamp
        return (0.0, 0.0)

    def refuse_on_embedding(self):
        """A gather has no accumulator for an epilogue to run on."""
        if self.clamp is not None:
            raise RuntimeError(f"{BACKEND_NAME}: clamp fused into an embedding")
        if self.act is not None:
            raise RuntimeError(f"{BACKEND_NAME}: act fused into an embedding")
        if self.softcap is not None:
            raise RuntimeError(
                f"{BACKEND_NAME}: softcap fused into an embedding")
        if self.res_node is not None:
            raise RuntimeError(
                f"{BACKEND_NAME}: residual fused into an embedding")
        if self.mul:
            raise RuntimeError(
                f"{BACKEND_NAME}: multiply fused into an embedding")


def _chain_marks(in_arg, out_arg):
    """Chain markers for the descriptor text: <sN reads a scratch slot, >sN
    writes one. Boundary args stay unmarked."""
    marks = ""
    if in_arg >= ARG_SCRATCH0:
        marks += f"<s{in_arg - ARG_SCRATCH0}"
    if out_arg >= ARG_SCRATCH0:
        marks += f">s{out_arg - ARG_SCRATCH0}"
    return marks


def _group_note(k, scale_group):
    if not scale_group:
        return ""
    return f" x{-(-k // scale_group)}g{scale_group}"


def _act_scale_note(flags):
    return (("+sa" if flags & FLAG_STATIC_ACT else "")
            + ("+pt" if flags & FLAG_ACT_PER_TENSOR else "")
            + ("+aq" if flags & FLAG_ACT_QUANT else ""))


def _register_selected_rows(w, i, op):
    """The EXT_SELECTED_ROWS record for a legal-token head; its log note."""
    selected_count = op.get("selected_count")
    if selected_count is None:
        return ""
    if op["_in_arg"] >= ARG_SCRATCH0 or op["_out_arg"] >= ARG_SCRATCH0:
        raise RuntimeError(
            f"{BACKEND_NAME}: selected-row projection cannot use chain scratch")
    w.ext_records.append((
        i, EXT_SELECTED_ROWS,
        struct.pack(SELECTED_ROWS_FORMAT, op["_rows_arg"], selected_count),
    ))
    return f" select{selected_count}/{op['n']}"


def _emit_bmm_qk(w, op, chn):
    """Q [batch, m, k] times untransposed keys [batch, n, k] (the permute is
    folded), scores [batch, m, n] out; batch counts the flattened heads.
    No bias, no weights, no scales: bias_off/wt_off/scale_off all 0 and
    unread. res_arg carries the SECOND INPUT, scale_group the batch count --
    see the OP_BMM_DYN_I8 note by the opcode. Static activation scales do
    not apply here; both sides stay dynamic."""
    m, k, n = op["m"], op["k"], op["n"]
    if op["_in_arg"] >= ARG_SCRATCH0 or op["_out_arg"] >= ARG_SCRATCH0:
        raise RuntimeError(f"{BACKEND_NAME}: chained q@kT is unimplemented")
    if op.get("kcache_affine"):
        # ExecuTorch's stock affine K cache: codes at res_arg, per-row scales
        # at scale_off, per-row zero points at wt_off (all three arg indices).
        w.emit(f"bmmqk_kca8[{op['batch']}x{m}x{k}x{n}]{chn}",
               opcode=OP_BMM_KCACHE_AFFINE_I8, flags=0, m=m, k=k, n=n,
               k_pad=_pad_to(k, VEC_BYTES),
               in_arg=op["_in_arg"], out_arg=op["_out_arg"],
               scale_group=op["batch"], res_arg=op["_in2_arg"],
               scale_off=op["_in3_arg"], wt_off=op["_in4_arg"])
        return
    if op.get("kcache"):
        # The int8 K cache: codes at res_arg, per-row scales at scale_off.
        w.emit(f"bmmqk_kc8[{op['batch']}x{m}x{k}x{n}]{chn}",
               opcode=OP_BMM_KCACHE_I8, flags=0, m=m, k=k, n=n,
               k_pad=_pad_to(k, VEC_BYTES),
               in_arg=op["_in_arg"], out_arg=op["_out_arg"],
               scale_group=op["batch"], res_arg=op["_in2_arg"],
               scale_off=op["_in3_arg"])
        return
    w.emit(f"bmmqk_i8[{op['batch']}x{m}x{k}x{n}]{chn}",
           opcode=OP_BMM_DYN_I8, flags=0, m=m, k=k, n=n,
           k_pad=_pad_to(k, VEC_BYTES),
           in_arg=op["_in_arg"], out_arg=op["_out_arg"],
           scale_group=op["batch"], res_arg=op["_in2_arg"])


def _emit_fp32(w, op, epi, chn, bias_off):
    """The fp32 runtime walks [K,N] front-to-back (k outer), exactly the
    orientation mm/addmm supplies before the quantised path transposes to
    [N,K]; a convolution's rows are its output channels, [N,K]. No scales and
    no padding: inline, or under external_weights one PTD entry
    (store_float_weights). The diagnostic policy's every matmul and
    convolution, and under the default policy each weight a recipe left
    float."""
    node, m, k, n = op["node"], op["m"], op["k"], op["n"]
    if op["kind"] == "embedding":
        raise RuntimeError(
            f"{BACKEND_NAME}: fp32 embedding storage is not implemented; "
            "use i4/i8/i16")
    if op.get("pt") is not None:
        raise RuntimeError(
            f"{BACKEND_NAME}: format_policy='fp32' cannot consume "
            "already-quantised pass-through weights")
    if node.meta.get("esp32_act_per_tensor"):
        raise RuntimeError(
            f"{BACKEND_NAME}: fp32 addmm cannot absorb a folded "
            "per-tensor activation quantiser")
    if node.meta.get("esp32_act_quant") is not None:
        from executorch.backends.esp32s3.serialization.asym_records import SchemeRefused

        raise SchemeRefused(
            f"[act-format] {node.name}: format_policy='fp32' cannot honour "
            f"the model's own activation quantiser, which the chain absorbed")
    if epi.mul:
        # The fp32 loop accumulates into its output row, where an in-place
        # factor lives; the gate refuses a multiply under the fp32 policy.
        raise RuntimeError(
            f"{BACKEND_NAME}: fp32 addmm cannot carry a fused multiply")
    w_kn = np.ascontiguousarray(op["weight"], dtype=np.float32)
    if w_kn.shape != (k, n):
        raise RuntimeError(
            f"{BACKEND_NAME}: fp32 weight is {w_kn.shape}, expected ({k}, {n})")
    conv = op.get("conv") is not None
    # A convolution's rows are its output channels, [N,K] (the OP_ADDMM note).
    wt_off, xflag = w.store_float_weights(
        (np.ascontiguousarray(w_kn.T) if conv else w_kn).tobytes())
    cflag, conv_note = w.float_conv_geometry(op, 4)
    w.emit(f"addmm_f32[{m}x{k}x{n}]" + epi.note() + chn + conv_note,
           opcode=OP_ADDMM, flags=epi.flags() | cflag | xflag, m=m, k=k, n=n,
           k_pad=k,
           in_arg=op["_in_arg"], out_arg=op["_out_arg"],
           bias_off=bias_off, wt_off=wt_off,
           bounds=epi.bounds(), res_arg=epi.res_arg)


def _half_format(w, op):
    """"fp16" or "bf16" when this op's weight is stored as one, else None: a
    format_overrides value first, then a half checkpoint's weight a recipe
    left alone (the gate's KEEP_FLOAT meta), then the policy. Never for
    pass-through codes, whose format is fixed, nor for a weight the gate kept
    at fp32."""
    if op.get("pt") is not None or op["kind"] == "bmm_qk":
        return None
    override = w.override_name(op)
    if override is not None:
        return override if override in HALF_FORMATS else None
    kept = op["node"].meta.get(KEEP_FLOAT)
    if kept is not None:
        return kept if kept in HALF_FORMATS else None
    policy = w.cfg.format_policy
    return policy if policy in HALF_FORMATS else None


def _emit_half(w, op, epi, chn, bias_off, fmt):
    """A weight stored as fp16 or bf16 (OP_ADDMM_F16/BF16, OP_EMBED_F16/BF16):
    [n, k] two bytes an element -- the value a half checkpoint shipped, or
    the policy's or an override's rounding of an fp32 weight -- widened to
    fp32 as the loop reads it, so the op computes what the fp32 loop computes
    on the widened weight. Inline, or under external_weights one PTD entry,
    as the fp32 loop's are (store_float_weights)."""
    node, m, k, n = op["node"], op["m"], op["k"], op["n"]
    mm_opcode, embed_opcode = HALF_FORMATS[fmt]
    tag = HALF_TAGS[fmt]
    source = op.get("weight_node")
    name = ((_const_fqn(w.edge_program, source) if source is not None else None)
            or node.name)
    if op["kind"] == "embedding":
        epi.refuse_on_embedding()
        table = np.ascontiguousarray(op["weight"], dtype=np.float32)
        if table.shape != (n, k):
            raise RuntimeError(
                f"{BACKEND_NAME}: {fmt} table is {table.shape}, expected "
                f"({n}, {k})")
        wt_off, xflag = w.store_float_weights(
            half_bits(table, fmt, name).tobytes())
        w.emit(f"embed_{tag}[{m}x{k}x{n}] rows->{2 * k}B" + chn,
               opcode=embed_opcode, flags=xflag, m=m, k=k, n=n, k_pad=k,
               in_arg=op["_in_arg"], out_arg=op["_out_arg"], wt_off=wt_off)
        return
    if op.get("selected_count") is not None:
        raise RuntimeError(
            f"{BACKEND_NAME}: the selected-row head {node.name} implements "
            f"i4/i8 weights only, not {fmt}")
    if node.meta.get("esp32_act_per_tensor"):
        raise RuntimeError(
            f"{BACKEND_NAME}: {fmt} addmm cannot absorb a folded per-tensor "
            "activation quantiser")
    if node.meta.get("esp32_act_quant") is not None:
        from executorch.backends.esp32s3.serialization.asym_records import SchemeRefused

        raise SchemeRefused(
            f"[act-format] {node.name}: a {fmt} weight cannot honour the "
            f"model's own activation quantiser, which the chain absorbed")
    if epi.mul:
        raise RuntimeError(
            f"{BACKEND_NAME}: {fmt} addmm cannot carry a fused multiply")
    w_kn = np.asarray(op["weight"], dtype=np.float32)
    if w_kn.shape != (k, n):
        raise RuntimeError(
            f"{BACKEND_NAME}: {fmt} weight is {w_kn.shape}, expected ({k}, {n})")
    wt_off, xflag = w.store_float_weights(half_bits(w_kn.T, fmt, name).tobytes())
    cflag, conv_note = w.float_conv_geometry(op, 2)
    w.emit(f"addmm_{tag}[{m}x{k}x{n}]" + epi.note() + chn + conv_note,
           opcode=mm_opcode, flags=epi.flags() | cflag | xflag, m=m, k=k, n=n,
           k_pad=k, in_arg=op["_in_arg"], out_arg=op["_out_arg"],
           bias_off=bias_off, wt_off=wt_off,
           bounds=epi.bounds(), res_arg=epi.res_arg)


def _pass_through_codes(pt):
    """A pass-through weight's codes narrowed to its logical columns, with
    its logical row/column counts and its storage row count.

    The FULL backing rows are kept even when the logical head is a row
    prefix: the descriptor's n limits execution, and the full bytes
    intentionally hash-identically to the embedding's payload so
    NamedDataStore emits them once.
    """
    q = pt.codes.detach().numpy().astype(np.int8)
    storage_n, storage_k = q.shape
    logical_n = storage_n if pt.logical_rows is None else pt.logical_rows
    logical_k = storage_k if pt.logical_cols is None else pt.logical_cols
    return np.ascontiguousarray(q[:, :logical_k]), logical_n, logical_k, storage_n


def _emit_pass_through_matmul(w, op, epi, chn, bias_off, selected_note=""):
    """PASS-THROUGH matmul: codes and scales consumed verbatim -- no
    re-quantisation anywhere. Re-quantising an already-quantised tensor is
    strictly destructive; taking it as-is is what makes the artifact
    bit-faithful to the model the user calibrated or trained (QAT). Zero
    points and float offsets ride in EXT_WEIGHT_AFFINE."""
    pt = op["pt"]
    node, m, k, n = op["node"], op["m"], op["k"], op["n"]
    w.override_for(op)      # refuses by name; a code's format is fixed
    act = _act_record(node)
    fmt, scale_group, storage_n = _pass_through_matmul_layout(pt, k, n, act)
    q, _, _, storage_n = _pass_through_codes(pt)
    k_pad = _pad_to(k, fmt.k_align)
    payload = pack_rows(q, k_pad, fmt)
    if scale_group == 0:
        scale_arr = (pt.scales.detach().numpy()
                     .reshape(storage_n).astype(np.float32))
    else:
        scale_arr = pt.scales.detach().numpy().astype(np.float32)
    # fp32 as given, never fp16-rounded: fidelity to the incoming model is
    # the entire point of this path.
    wt_off, scale_off, flags = w.store_weights(payload, scale_arr.tobytes())
    groups = 1 if scale_group == 0 else -(-k // scale_group)
    aflag, affine_note = w.weight_affine(pt, storage_n, groups, act=act,
                                         codes=q, k=k, gsize=scale_group,
                                         bits=fmt.bits)
    flags |= aflag
    cflag, conv_note = w.conv_geometry(op, fmt)
    flags |= cflag
    tied_prefix = pt.logical_rows is not None
    if tied_prefix and w.cfg.stage_tied_head:
        if fmt.name != "i4":
            raise RuntimeError(
                f"{BACKEND_NAME}: stage_tied_head currently expands external "
                f"i4 codes; got {fmt.name}")
        if pt.offsets is not None:
            raise RuntimeError(
                f"{BACKEND_NAME}: stage_tied_head expands codes to int8 "
                f"with their zero point folded in; a float offset has no "
                f"place there (the runtime refuses it as "
                f"weight-staged-offsets)")
        flags |= FLAG_STAGE_I8
    if tied_prefix and w.cfg.split_tied_head:
        flags |= FLAG_HEAD_SPLIT
    flags |= epi.flags()
    # A PT2E-quantised model takes THIS branch -- its weights are
    # pass-through codes -- so the folded per-tensor activation has to be
    # honoured here exactly as on the quantised-here path.
    act_scale, act_flags = w.activation_scale(
        node, fmt, codes_in=bool(op.get("codes_in")))
    flags |= act_flags
    oflag, out_note = w.out_quant(op, fmt)
    flags |= oflag
    text = (f"addmmPT_{fmt.name}[{m}x{k}x{n}] rows->"
            f"{fmt.row_bytes(k_pad)}B{_group_note(k, scale_group)}"
            + affine_note
            + epi.note()
            + out_note
            + _act_scale_note(flags)
            + ("+in8" if op.get("codes_in") else "")
            + ("+stage8" if flags & FLAG_STAGE_I8 else "")
            + ("+headsplit" if flags & FLAG_HEAD_SPLIT else "")
            + selected_note
            + chn
            + conv_note)
    w.emit(text, opcode=fmt.opcode, flags=flags, m=m, k=k, n=n, k_pad=k_pad,
           in_arg=op["_in_arg"], out_arg=op["_out_arg"],
           bias_off=bias_off, wt_off=wt_off, scale_off=scale_off,
           scale_group=scale_group, bounds=epi.bounds(),
           res_arg=epi.res_arg, act_scale=act_scale)


def _emit_pass_through_embedding(w, op, chn, bias_off):
    """PASS-THROUGH table: codes and scales verbatim, same contract as the
    matmul PT path. Grouped fp16 scales are stored back as the exact same
    fp16 bits; a single-group table stores fp32 per the dedup rule, which is
    what lets the tied tok_emb share one payload with the head. Zero points
    (integer or float) and float offsets ride in EXT_WEIGHT_AFFINE."""
    pt = op["pt"]
    m, k, n = op["m"], op["k"], op["n"]
    w.override_for(op)      # refuses by name, as the matmul path does
    fmt, scale_group = _pass_through_table_layout(pt, k, n)
    q, _, _, _ = _pass_through_codes(pt)
    k_pad = _pad_to(k, 2 if fmt.bits == 4 else 1)
    payload = pack_rows(q, k_pad, fmt)
    if scale_group == 0:
        scale_bytes = (pt.scales.detach().numpy().reshape(n)
                       .astype(np.float32).tobytes())
        fp16 = False
    else:
        # fp16 ONLY when the incoming scales already survive the round trip:
        # rounding genuine fp32 scales adds ~2.6e-04 of relative error to the
        # scales the codes were chosen against, the host/device mismatch
        # quantize_absmax_rows warns about. The matmul pass-through path keeps
        # its scales "fp32 as given, never fp16-rounded" for the same reason.
        scale_bytes, fp16 = encode_pass_through_embedding_scales(
            pt.scales.detach().numpy())
    wt_off, scale_off, flags = w.store_weights(payload, scale_bytes)
    if fp16:
        flags |= FLAG_FP16_SCALES
    groups = 1 if scale_group == 0 else -(-k // scale_group)
    aflag, affine_note = w.weight_affine(pt, n, groups)
    flags |= aflag
    text = (f"embedPT_{fmt.name}[{m}x{k}x{n}] rows->"
            f"{fmt.row_bytes(k_pad)}B{_group_note(k, scale_group)}"
            + affine_note)
    w.emit(text, opcode=fmt.embed_opcode, flags=flags, m=m, k=k, n=n,
           k_pad=k_pad, in_arg=op["_in_arg"], out_arg=op["_out_arg"],
           bias_off=bias_off, wt_off=wt_off, scale_off=scale_off,
           scale_group=scale_group)


def embedding_scale_layout(k: int, fmt):
    """(group, fp16_scales) for a table of row width `k` quantised here.

    Grouped where a row holds more than one group, with fp16 scales only when
    grouped (see _embedding_table_format) -- and never at i16. An i16 scale
    is amax/32767, which fp16 holds only as a subnormal for any group with
    amax below 2: at amax 0.1 the stored half is ~1% off, the codes clip at
    the group maximum, and the accuracy rung decodes worse than i8. The i16
    payload is twice the bytes of i8 already; fp32 scales are noise beside it.

    Shared with models.embedding_iq_absmax, so the parity reference cannot
    make a different choice from the backend.
    """
    f = FORMATS[fmt] if isinstance(fmt, str) else fmt
    group = SCALE_GROUP if k > SCALE_GROUP else None
    return group, group is not None and f.bits <= 8


def _embedding_table_format(w, op):
    """Format, row padding and scale grouping for a table quantised here.

    The table is already [V,D] -- rows are what a gather reads, so no
    transpose. Rows are padded only enough to keep i4 nibbles from straddling
    a row boundary: nothing here feeds a 128-bit vector load, so the 16-byte
    row alignment the matmul path needs would be pure waste (D=96 at i4 would
    round 48 B up to 64, a third of the table).

    Grouped fp16 scales, but ONLY where the row is wide enough to have more
    than one group. The two are coupled by the flash budget rather than by
    taste: at the deployed config ple_table needs 6 groups per 768-wide row,
    which in fp32 is 786,432 B of scales and puts the model at 101.7% of the
    15.66 MB partition, against 393,216 B and 99.2% in fp16.

    When there is only one group the whole justification evaporates -- there
    is nothing to make affordable -- and halving those scales costs something
    much more valuable. `tok_emb` is TIED: it appears both here as a gather
    and as the output matmul, in two partitions. The matmul path stores
    per-row fp32, so rounding the scale to fp16 here makes the quantised
    codes differ and the two copies stop being byte-identical, which defeats
    the content-addressed dedup. Matching the matmul's format costs 65,536 B
    of scales and saves 1,572,864 B of duplicated weights -- a 24x trade, and
    the difference between the deployed model fitting the part and not.
    """
    w_nk = np.ascontiguousarray(op["weight"])
    fmt, why = choose_format(w_nk, w.cfg.format_policy)
    forced = w.override_for(op)
    if forced is not None:
        fmt, why = forced, f"override {forced.name}"
    k_pad = _pad_to(op["k"], 2 if fmt.bits == 4 else 1)
    group, fp16 = embedding_scale_layout(op["k"], fmt)
    return w_nk, fmt, why, k_pad, group, fp16


def _matmul_weight_format(w, op):
    """Format, row padding and scale grouping for a matmul weight quantised
    here. [K,N] -> [N,K] so each output column is one contiguous dot product,
    then each row is padded up to the format's alignment with zeros. Zeros
    contribute nothing to the dot, which removes the kernel's tail case.

    Per-row fp32 is the artifact-stable default. Group-128/fp16 is opt-in
    for checkpoints whose QAT used that exact format; the runtime honours
    scale_group.
    """
    node, k = op["node"], op["k"]
    w_nk = np.ascontiguousarray(op["weight"].T)
    fmt, why = choose_format(w_nk, w.cfg.format_policy)
    forced = w.override_for(op)
    if forced is not None:
        fmt, why = forced, f"override {forced.name}"
    if op.get("selected_count") is not None and fmt.bits not in (4, 8):
        # The gate admits only i4/i8/auto policies here, but an override
        # bypasses that, and the runtime has no other selected-row walk.
        raise RuntimeError(
            f"{BACKEND_NAME}: the selected-row head {node.name} implements "
            f"i4/i8 weights only, not {fmt.name} ({why})")
    k_pad = _pad_to(k, fmt.k_align)
    group, fp16 = None, False
    if k > SCALE_GROUP:
        if w.cfg.group_weights:
            if fmt.bits > 8:
                raise RuntimeError(
                    f"{BACKEND_NAME}: group_weights=True is implemented "
                    f"for i4/i8, not {fmt.name} ({node.name}, K={k})")
            group, fp16 = SCALE_GROUP, True
        else:
            print(f"  [{BACKEND_NAME}] NOTE {node.name} K={k} > "
                  f"{SCALE_GROUP}: one scale for the whole row; set "
                  f"group_weights=True to match group-128 QAT")
    return w_nk, fmt, why, k_pad, group, fp16


def _emit_quantised(w, op, epi, chn, bias_off, selected_note):
    """A weight or table quantised HERE, from fp32, at the policy's format."""
    node, m, k, n = op["node"], op["m"], op["k"], op["n"]
    is_embed = op["kind"] == "embedding"
    if is_embed:
        w_nk, fmt, why, k_pad, group, fp16 = _embedding_table_format(w, op)
    else:
        w_nk, fmt, why, k_pad, group, fp16 = _matmul_weight_format(w, op)
    q, scale = quantize_absmax_rows(w_nk, fmt, group=group, fp16_scales=fp16)
    payload = pack_rows(q, k_pad, fmt)
    # The descriptor carries the group SIZE (0 = per-row), never the count:
    # the count is derivable from the size, the size is not derivable from
    # the count on a ragged row. See test_formats.py for the failure the
    # other choice produces.
    scale_group = 0 if scale.ndim == 1 else SCALE_GROUP
    scale_dtype = np.float16 if fp16 else np.float32
    wt_off, scale_off, flags = w.store_weights(
        payload, scale.astype(scale_dtype).tobytes())
    affine_note = ""
    if not is_embed:
        # The model's own activation quantiser over these symmetric codes:
        # an activation zero point needs their sums S'.
        act = _act_record(node)
        if act is not None:
            try:
                _a16_problem(None, fmt, scale_group, act)
            except LoweringRefused as why:
                # The gate passed the policy (i8); group_weights or a format
                # override picked something else after it.
                from executorch.backends.esp32s3.serialization.asym_records import SchemeRefused

                raise SchemeRefused(f"[act-code-bits] {node.name}: {why.kind}: "
                                    f"{why.detail}") from None
            _affine_bounds(None, fmt, k, scale_group, act)
        groups = 1 if scale_group == 0 else -(-k // scale_group)
        aflag, affine_note = w.weight_affine(None, q.shape[0], groups,
                                             act=act, codes=q, k=k,
                                             gsize=scale_group, bits=fmt.bits)
        flags |= aflag
    cflag, conv_note = w.conv_geometry(op, fmt)
    flags |= cflag
    if fp16:
        flags |= FLAG_FP16_SCALES
    if is_embed:
        epi.refuse_on_embedding()
    flags |= epi.flags()
    act_scale = 0.0
    out_note = ""
    if not is_embed:
        act_scale, act_flags = w.activation_scale(
            node, fmt, codes_in=bool(op.get("codes_in")))
        flags |= act_flags
        oflag, out_note = w.out_quant(op, fmt)
        flags |= oflag
    kind = "embed" if is_embed else "addmm"
    opcode = fmt.embed_opcode if is_embed else fmt.opcode
    why_note = ""
    if w.cfg.format_policy == "auto" or why.startswith("override"):
        why_note = f" ({why})"
    text = (f"{kind}_{fmt.name}[{m}x{k}x{n}] rows->"
            f"{fmt.row_bytes(k_pad)}B{_group_note(k, scale_group)}"
            + affine_note
            + epi.note()
            + out_note
            + _act_scale_note(flags)
            + ("+in8" if op.get("codes_in") else "")
            + selected_note
            + chn
            + conv_note
            + why_note)
    w.emit(text, opcode=opcode, flags=flags, m=m, k=k, n=n, k_pad=k_pad,
           in_arg=op["_in_arg"], out_arg=op["_out_arg"],
           bias_off=bias_off, wt_off=wt_off, scale_off=scale_off,
           scale_group=scale_group, bounds=epi.bounds(),
           res_arg=epi.res_arg, act_scale=act_scale)


def _emit_op(w, i, op):
    """One op record -> one descriptor, by storage path: dynamic q@kT, fp32
    diagnostic, pass-through codes (matmul or table), or quantised here."""
    chn = _chain_marks(op["_in_arg"], op["_out_arg"])
    selected_note = _register_selected_rows(w, i, op)
    if op["kind"] == "bmm_qk":
        _emit_bmm_qk(w, op, chn)
        return
    epi = _Epilogue(op, w.act_in)
    # Offset zero is the no-bias sentinel. In particular, do not put an
    # output-vocabulary-sized zero vector in the PTE for ``nn.Linear(
    # bias=False)`` while its real weights live in PTD.
    bias = op["bias"]
    bias_off = 0
    if bias is not None:
        bias_off = w.place(bias.astype(np.float32).tobytes())
    is_embed = op["kind"] == "embedding"
    # A 2-byte weight (_half_format), else the fp32 loop: the diagnostic
    # policy, or a weight a recipe left float (Esp32OperatorSupport.
    # _keeps_float marks the node).
    half = _half_format(w, op)
    fp32 = half is None and (w.cfg.format_policy == "fp32"
                             or op["node"].meta.get(KEEP_FLOAT) == "fp32")
    if op.get("out_quant") is not None and (is_embed or fp32 or half):
        raise RuntimeError(
            f"{BACKEND_NAME}: an output quantisation on "
            f"{'an embedding' if is_embed else 'a float-weight loop'}; the "
            f"gate should have refused its pair")
    if half is not None:
        _emit_half(w, op, epi, chn, bias_off, half)
    elif fp32:
        _emit_fp32(w, op, epi, chn, bias_off)
    elif op.get("pt") is not None and not is_embed:
        _emit_pass_through_matmul(w, op, epi, chn, bias_off, selected_note)
    elif op.get("pt") is not None:
        _emit_pass_through_embedding(w, op, chn, bias_off)
    else:
        _emit_quantised(w, op, epi, chn, bias_off, selected_note)


@final
class Esp32Backend(BackendDetails):
    @classmethod
    def copy_exported_program_for_preprocess(cls, edge_program, compile_specs):
        """The partition program itself: preprocess only reads it.

        The default deep-copies every partition's program, real weights
        included, in case a backend edits the graph inside preprocess (0.14 s
        and a copy of every weight on the story Glimmer). Nothing below writes
        to the program -- op records, blob and named data are all new objects
        -- and keeping it that way is the condition for skipping the copy.
        """
        return edge_program

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        ops = _collect_ops(edge_program)
        boundary = _Boundary(edge_program)
        _assign_dataflow(ops, boundary)
        writer = _BlobWriter(edge_program, boundary, compile_specs, len(ops))
        for i, op in enumerate(ops):
            _emit_op(writer, i, op)
        return writer.finish()
