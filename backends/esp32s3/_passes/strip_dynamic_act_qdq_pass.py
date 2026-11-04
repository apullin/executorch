# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remove dynamic activation quantize/dequantize pairs the delegate does itself."""

from typing import Optional

from executorch.backends.esp32s3.partition.espressif_partitioner import _overload


# ---- torchao QDQ handling ---------------------------------------------------
def _activation_arg_index(node):
    """Which arg of `node` is the ACTIVATION, or None if it has none.

    The weight side must never match: a weight arriving through a quantise pair
    is a pass-through weight, absorbed by `_resolve_const` on a different route.
    """
    t = str(node.target)
    if "addmm" in t:
        return 1                       # addmm(bias, activation, weight)
    if "aten.mm" in t or t.endswith("mm.default"):
        return 0
    if "convolution" in t:
        return 0
    return None


def _is_view(node) -> bool:
    t = str(node.target)
    return node.op == "call_function" and (
        "view_copy" in t or "_to_dim_order_copy" in t or "clone" in t)


def _walks(node, rows: bool = False) -> bool:
    """Whether the strip walks past `node` from a chain's dequantised value
    towards the op that reads it: a reshape, a copy or a dim-order change
    (_is_view), and for a scheme over the whole tensor static_qdq's other
    moves too -- a permute, an alias, a squeeze, an unsqueeze (to_edge
    spells an unbatched convolution unsqueeze -> convolution -> squeeze, and
    a Linear over a transposed tensor permute -> addmm). Each moves values
    without changing any, so one absmax over the op's activation is the
    absmax over the tensor the model quantised. A per-token scheme (`rows`)
    also needs the tokens to stay the op's rows, which a reshape keeps
    exactly when the row width stays (_check_rows) but a permute need not,
    even at the same width: it walks views only."""
    from executorch.backends.esp32s3._passes.static_qdq import is_move

    if node.op != "call_function":
        return False
    return _is_view(node) or (not rows and is_move(node))


def _ends(reader, src, depth: int = 4, rows: bool = False):
    """Where each path from `src` through `reader` and the data movement after
    it (_walks; `rows` for a per-token scheme) ends: [(the moves on the way,
    the node at the end, whether that node reads the path as its matmul or
    conv activation)]. Any other end -- a portable op, a weight operand, a
    graph output, a move the walk does not pass or one too deep -- needs the
    dequantised value itself.

    A reshape between the dequantise and the matmul is the normal case, not an
    exotic one -- `nn.Linear` on a 3-D input lowers to `view -> addmm -> view`.
    A view is numel-preserving, so an absmax over the matmul's input equals an
    absmax over the tensor the model quantised. The scheme survives the hop,
    which is what makes walking through it legitimate rather than convenient.
    """
    if _walks(reader, rows) and depth and reader.users:
        return [((reader, *views), end, takes)
                for user in reader.users
                for views, end, takes in _ends(user, reader, depth - 1, rows)]
    idx = (_activation_arg_index(reader) if reader.op == "call_function"
           else None)
    takes = (idx is not None and idx < len(reader.args)
             and reader.args[idx] is src
             and sum(a is src for a in reader.args) == 1)
    return [((), reader, takes)]


class DynamicChainRefused(RuntimeError):
    """A dynamic activation quantiser this backend refuses, by rule name.

    Never a substitute. A chain is absorbed exactly into the ops the delegate
    runs, and stays in place for a reader no delegate takes, computed by
    portable ops exactly as torch computes it. What can do neither -- a
    scheme no record describes, a chooser no chain this backend reads, a
    chain left in place without the partitioner's gate -- stops the export
    here, saying why.
    """

    def __init__(self, rule: str, where: str, detail: str):
        self.rule = rule
        super().__init__(f"esp32: dynamic activation chain {where} refused "
                         f"[{rule}]: {detail}")


def _op_is(node, name: str, overload: Optional[str] = None) -> bool:
    """Whether `node` calls exactly `name` ("torchao::quantize_affine") -- and
    that overload when given. A substring test of the target's name would
    also take quantize_affine_float8 for quantize_affine."""
    op = _overload(node)
    return (op is not None and op._schema.name == name
            and (overload is None or op._overloadname == overload))


def _is_chooser(node) -> bool:
    op = _overload(node)
    return op is not None and "choose_qparams" in op._schema.name


def _named_args(node) -> dict:
    """{argument name: value} of an op node; an absent argument takes its
    schema default (None where it has none)."""
    out = {}
    for i, arg in enumerate(_overload(node)._schema.arguments):
        if i < len(node.args):
            out[arg.name] = node.args[i]
        elif arg.name in node.kwargs:
            out[arg.name] = node.kwargs[arg.name]
        else:
            out[arg.name] = getattr(arg, "default_value", None)
    return out


def _chooser_output(value):
    """(chooser, index) when `value` is getitem(chooser, index), else (None, None)."""
    import operator

    import torch

    if (isinstance(value, torch.fx.Node) and value.op == "call_function"
            and value.target is operator.getitem and _is_chooser(value.args[0])):
        return value.args[0], value.args[1]
    return None, None


def dynamic_chooser(value, depth: int = 8):
    """The chooser whose dynamic chain `value` is the dequantised value of,
    through pure data movement (static_qdq.is_move), or None: a torchao
    dequantize_affine or a PT2E dequantize_per_tensor.tensor of the quantize
    whose scale the chooser computes. The first steps of _torchao_chain and
    _pt2e_chain without their checks of the scheme, so that the gate can ask
    without being refused (Esp32OperatorSupport._static_act_dynamic_problem)."""
    import torch

    from executorch.backends.esp32s3._passes.static_qdq import is_move

    for _ in range(depth):
        if not isinstance(value, torch.fx.Node):
            return None
        for dq, q, overload in (
                ("torchao::dequantize_affine", "torchao::quantize_affine", None),
                ("quantized_decomposed::dequantize_per_tensor",
                 "quantized_decomposed::quantize_per_tensor", "tensor")):
            if _op_is(value, dq, overload):
                source = value.args[0] if value.args else None
                if not _op_is(source, q, overload):
                    return None
                return _chooser_output(_named_args(source)["scale"])[0]
        if not is_move(value):
            return None
        value = value.args[0]
    return None


def _is_float32(dtype) -> bool:
    """A dtype argument naming float32: torch.float32, or the ScalarType code 6
    a schema default carries (dequantize_affine's output_dtype=6)."""
    import torch

    return dtype is None or dtype == torch.float32 or dtype == 6


class _Chain:
    """One dynamic chain, x -> chooser -> quantize -> dequantize, checked and
    read: `scheme` is the mirrors.ActScheme the consumers will carry,
    or `per_tensor` marks Esp32Quantizer's own spelling, which keeps
    FLAG_ACT_PER_TENSOR instead."""

    def __init__(self, dq, q, chooser, x, scheme=None, per_tensor=False):
        self.dq, self.q, self.chooser, self.x = dq, q, chooser, x
        self.scheme, self.per_tensor = scheme, per_tensor
        self.dqs = [dq]     # with its identical copies (_check_users)


def _same_call(a, b, skip_input=False):
    """Whether two nodes call the same op on the same arguments (the first
    argument aside when `skip_input`)."""
    return (a.op == b.op == "call_function" and a.target == b.target
            and (a.args[1:] == b.args[1:] if skip_input else a.args == b.args)
            and a.kwargs == b.kwargs)


def _check_users(chain):
    """The chooser's scale and zero point feed only this quantize and its
    dequantize, and the codes only the dequantize: anything else reading them
    would be handed values the absorbed chain no longer computes.

    Identical copies count as the chain itself -- another quantize of the
    same tensor with the same arguments (one chooser feeding a quantize per
    consumer), another dequantize of one of them with the same arguments
    (DuplicateDQPass copies a dequantize per user for every chooser but
    choose_qparams.tensor). Each copy computes the same codes and values, so
    absorbing them all with one scheme is exact; chain.dqs lists the
    dequantizes, the chain's own first."""
    where = chain.dq.name
    qs = [chain.q] + [u for out in chain.chooser.users for u in out.users
                      if u is not chain.q and _same_call(u, chain.q)]
    qs = list(dict.fromkeys(qs))
    dqs = [chain.dq] + [d for q in qs for d in q.users
                        if d is not chain.dq and _same_call(d, chain.dq, True)]
    dqs = list(dict.fromkeys(dqs))
    family = set(qs) | set(dqs)
    for out in chain.chooser.users:
        if not set(out.users) <= family:
            raise DynamicChainRefused(
                "act-chooser-users", where,
                f"{chain.chooser.name}'s output also feeds "
                f"{sorted(u.name for u in set(out.users) - family)}")
    for q in qs:
        if not set(q.users) <= set(dqs):
            raise DynamicChainRefused(
                "act-chooser-users", where,
                f"the codes of {q.name} also feed "
                f"{sorted(u.name for u in set(q.users) - set(dqs))}")
    chain.dqs = dqs
    val = chain.x.meta.get("val")
    if val is None or str(val.dtype) != "torch.float32":
        raise DynamicChainRefused(
            "act-input-dtype", where,
            f"the quantised activation is {getattr(val, 'dtype', 'unknown')}, "
            f"not float32")


def _code_range(where, dtype, qmin, qmax, symmetric):
    """The chain's code range, refusing what no dynamic scheme here covers:
    int16 (no stock recipe quantises activations dynamically to int16;
    static int16 is the a16w8 path), unsigned symmetric (PT2E puts the
    zero point at 128 and torchao at the midpoint) and anything but 8 bits."""
    import torch

    if dtype in (torch.int16, torch.uint16):
        raise DynamicChainRefused(
            "act-dynamic-int16", where,
            "dynamic int16 activations: no stock recipe quantises activations "
            "dynamically to int16 (static int16 activations, Arm a16w8, are "
            "supported)")
    if dtype not in (torch.int8, torch.uint8):
        raise DynamicChainRefused("act-dtype", where,
                                  f"{dtype} activation codes (int8 or uint8 only)")
    info = torch.iinfo(dtype)
    qmin = info.min if qmin is None else int(qmin)
    qmax = info.max if qmax is None else int(qmax)
    if symmetric and qmin >= 0:
        raise DynamicChainRefused(
            "act-unsigned-symmetric", where,
            f"symmetric activations over [{qmin}, {qmax}]: PT2E puts the zero "
            f"point at 128 and torchao at the midpoint")
    return qmin, qmax


def _check_eps(where, eps):
    """The eps floor under a dynamic chooser must be positive in float32
    (what the record carries). At 0 -- or -0.0, or a value float32
    rounds to 0 -- an all-zero row chooses scale 0 and torch's quantize
    multiplies by fl(1/0); the runtime refuses such a record at load
    (act-eps), and the export refuses it here, with the other rules."""
    import math

    import numpy as np

    e32 = float(np.float32(eps))
    if not (math.isfinite(e32) and e32 > 0.0):
        raise DynamicChainRefused(
            "act-eps", where,
            f"eps {eps!r} is {e32!r} in float32: an all-zero row would choose "
            f"scale 0, which torch's quantize divides by")


def _check_in_place_eps(where, eps):
    """A chain computed in place multiplies by fl(1 / scale), and the scale
    is at least eps: past fl(1 / eps) = inf (eps below ~2.9e-39, a float32
    subnormal) a zero in a row whose scale is eps gives 0 * inf = NaN,
    where the converted model casts that NaN code to the integer 0 and
    dequantises a zero (and mirrors.py takes NaN to the zero point's code
    before rounding). A chain absorbed into a claimed op is the runtime's
    quantiser, which takes NaN to the zero point (quant_rules.h); one left
    in front of a portable reader is refused here by name instead."""
    import math

    import numpy as np

    with np.errstate(over="ignore", divide="ignore"):
        inv = float(np.float32(1.0) / np.float32(eps))
    if not math.isfinite(inv):
        raise DynamicChainRefused(
            "act-eps-in-place", where,
            f"eps {eps!r}: its float32 reciprocal is {inv!r}, so a zero in a "
            f"row whose scale is eps would come out NaN where the converted "
            f"model gives 0; a reader that no delegate takes needs the chain "
            f"computed in place")


def _integral_range(where, qmin, qmax):
    """torchao takes float quant_min/quant_max and computes with them
    as floats ((qmax - qmin) / 2 for SYMMETRIC, the clamp); the record's range
    is integer, so a non-integral bound is refused rather than truncated."""
    for bound in (qmin, qmax):
        if isinstance(bound, float) and not bound.is_integer():
            raise DynamicChainRefused(
                "act-range-not-integral", where,
                f"quant_min {qmin!r}, quant_max {qmax!r}: torchao computes "
                f"with them as floats, and the record's range is integer")


def _torchao_chain(dq):
    """choose_qparams_affine -> quantize_affine -> dequantize_affine, or None
    for a dequant of something else (a weight's constant codes; a static
    scale, which static_qdq absorbs)."""
    import torch

    from executorch.backends.esp32s3.quantizer.mirrors import ActScheme

    q = dq.args[0]
    if not _op_is(q, "torchao::quantize_affine"):
        return None
    qa, da = _named_args(q), _named_args(dq)
    chooser, index = _chooser_output(qa["scale"])
    if chooser is None:
        return None
    where = dq.name
    if not _op_is(chooser, "torchao::choose_qparams_affine"):
        raise DynamicChainRefused(
            "act-chooser", where,
            f"{_overload(chooser)._schema.name} is not a chooser this backend "
            f"reproduces")
    ca = _named_args(chooser)
    zp_chooser, zp_index = _chooser_output(qa["zero_point"])
    same = (index == 0 and zp_chooser is chooser and zp_index == 1
            and ca["input"] is qa["input"]
            and da["scale"] is qa["scale"]
            and da["zero_point"] is qa["zero_point"]
            and list(da["block_size"]) == list(qa["block_size"])
            == list(ca["block_size"])
            and da["input_dtype"] == qa["output_dtype"] == ca["target_dtype"]
            and da["quant_min"] == qa["quant_min"] == ca["quant_min"]
            and da["quant_max"] == qa["quant_max"] == ca["quant_max"])
    if not same:
        raise DynamicChainRefused(
            "act-chain-mismatch", where,
            "the dequantize does not undo exactly this quantize (scale, zero "
            "point, block, range and dtype from one chooser over the same "
            "tensor)")
    if not _is_float32(da["output_dtype"]):
        raise DynamicChainRefused("act-input-dtype", where,
                                  f"dequantised to {da['output_dtype']}")
    chain = _Chain(dq, q, chooser, qa["input"])
    _check_users(chain)
    mapping = str(ca["mapping_type"]).rsplit(".", 1)[-1]
    if mapping == "SYMMETRIC_NO_CLIPPING_ERR":
        # Weights-only in torchao; its scale is the larger of
        # min_neg/qmin and max_pos/qmax, a chooser no record describes.
        raise DynamicChainRefused(
            "act-symmetric-no-clipping-err", where,
            "torchao SYMMETRIC_NO_CLIPPING_ERR activations: the scale is the "
            "larger of min_neg/qmin and max_pos/qmax, which no record describes")
    if mapping not in ("SYMMETRIC", "ASYMMETRIC"):
        raise DynamicChainRefused("act-chooser", where,
                                  f"torchao mapping type {mapping}")
    symmetric = mapping == "SYMMETRIC"
    _integral_range(where, ca["quant_min"], ca["quant_max"])
    qmin, qmax = _code_range(where, ca["target_dtype"], ca["quant_min"],
                             ca["quant_max"], symmetric)
    # torchao's SYMMETRIC chooser puts its zero point at the range's midpoint,
    # int((qmax + qmin + 1) / 2) (quant_primitives.py:1574): 0 for every stock
    # range, -13 for [-128, 100]. The SYM_DIV record's zero point is 0 (PT2E's
    # int8 symmetric chooser is 0 for any range, so the rule is torchao's only).
    if symmetric and int((qmax + qmin + 1) / 2) != 0:
        raise DynamicChainRefused(
            "act-symmetric-midpoint", where,
            f"torchao SYMMETRIC activations over [{qmin}, {qmax}]: the zero "
            f"point is the range's midpoint, {int((qmax + qmin + 1) / 2)}, "
            f"and the symmetric record's is 0")
    # choose_qparams_affine rounds its scale to scale_dtype (input's when
    # absent) and floors it at finfo(input).smallest_normal when eps is absent.
    if not _is_float32(ca["scale_dtype"]):
        raise DynamicChainRefused(
            "act-scale-dtype", where,
            f"{ca['scale_dtype']} scales: the chooser rounds its scale to that "
            f"dtype, and the runtime computes float32 ones")
    zp_dtype = ca["zero_point_dtype"]
    # Integer zero points, or float32 holding the same integers (the legacy
    # Int8DynActInt4WeightQuantizer's): torchao 0.18's chooser rounds every
    # zero point it makes (it has no preserve_zero/FLOAT-domain variant), so
    # a float32 one is integral by construction.
    if zp_dtype not in (None, torch.int8, torch.uint8, torch.int16, torch.int32,
                        torch.int64, torch.float32):
        raise DynamicChainRefused(
            "act-zero-point-dtype", where,
            f"{zp_dtype} zero points (integers, or float32 holding integers)")
    eps = ca["eps"]
    if eps is None:
        eps = float(torch.finfo(torch.float32).smallest_normal)
    _check_eps(where, eps)
    shape = list(chain.x.meta["val"].shape)
    block = [int(b) for b in qa["block_size"]]
    if block == shape:
        granularity = "tensor"
    elif (len(block) == len(shape) and all(b == 1 for b in block[:-1])
          and block[-1] == shape[-1]):
        granularity = "row"
    else:
        raise DynamicChainRefused(
            "act-granularity", where,
            f"blocks of {block} over {shape}: one scale per tensor or per token "
            f"only; per-channel or per-group scales along K cannot leave the "
            f"integer dot")
    chain.scheme = ActScheme(
        f"torchao/{mapping.lower()}", "symmetric" if symmetric else "affine",
        granularity, "rne_before_zp", qmin, qmax, eps=float(eps))
    return chain


def _pt2e_chain(dq):
    """choose_qparams(_symmetric).tensor -> quantize_per_tensor.tensor ->
    dequantize_per_tensor.tensor, PT2E's dynamic spelling (XNNPACK, Arm,
    Esp32Quantizer), or None for a static pair (static_qdq's)."""
    import torch

    from executorch.backends.esp32s3.quantizer.mirrors import ActScheme

    q = dq.args[0]
    if not _op_is(q, "quantized_decomposed::quantize_per_tensor", "tensor"):
        return None
    qa, da = _named_args(q), _named_args(dq)
    chooser, index = _chooser_output(qa["scale"])
    if chooser is None:
        return None
    where = dq.name
    affine = _op_is(chooser, "quantized_decomposed::choose_qparams", "tensor")
    symmetric = _op_is(chooser, "quantized_decomposed::choose_qparams_symmetric",
                       "tensor")
    if not (affine or symmetric):
        raise DynamicChainRefused(
            "act-chooser", where,
            f"{_overload(chooser)._schema.name}.{_overload(chooser)._overloadname}"
            f" is not a chooser this backend reproduces")
    ca = _named_args(chooser)
    zp_chooser, zp_index = _chooser_output(qa["zero_point"])
    fields = ("quant_min", "quant_max", "dtype")
    same = (index == 0 and zp_chooser is chooser and zp_index == 1
            and ca["input"] is qa["input"]
            and da["scale"] is qa["scale"]
            and da["zero_point"] is qa["zero_point"]
            and all(da[f] == qa[f] == ca[f] for f in fields))
    if not same:
        raise DynamicChainRefused(
            "act-chain-mismatch", where,
            "the dequantize does not undo exactly this quantize (scale, zero "
            "point, range and dtype from one chooser over the same tensor)")
    if not _is_float32(da.get("out_dtype")):
        raise DynamicChainRefused("act-input-dtype", where,
                                  f"dequantised to {da['out_dtype']}")
    chain = _Chain(dq, q, chooser, qa["input"])
    _check_users(chain)
    qmin, qmax = _code_range(where, ca["dtype"], ca["quant_min"],
                             ca["quant_max"], symmetric)
    eps = float(ca["eps"])
    _check_eps(where, eps)
    if (symmetric and ca["dtype"] == torch.int8 and (qmin, qmax) == (-127, 127)
            and eps == float(torch.finfo(torch.float32).eps)):
        # Esp32Quantizer's spelling: FLAG_ACT_PER_TENSOR, exactly this chooser
        # (absmax / 127 floored at FLT_EPSILON, rounded to even).
        chain.per_tensor = True
        return chain
    chain.scheme = ActScheme(
        "pt2e/symmetric" if symmetric else "pt2e/affine",
        "symmetric" if symmetric else "affine", "tensor", "rne_before_zp",
        qmin, qmax, eps=eps)
    return chain


_RECORDS = ("esp32_act_quant", "esp32_act_per_tensor")
# On a matmul or conv the gate refuses to give the chain's record: the
# (kind, detail) it refused it by. The chain is computed in place in front of
# it, and the partitioner leaves it portable on that value.
KEPT = "esp32_act_kept"
# On the value _computed_in_place computes: the name of the dequantize whose
# chain it is. An op reading it reads the recipe's dequantised value, and the
# q@kT gate (operator_support/qk_operands.py) refuses an operand that reaches
# it through data movement by that name, since its delegate would quantise
# the value again its own way.
CHAIN_VALUE = "esp32_chain_value"
# The gate's kind for a consumer that cannot take a per-token chain
# (_rows_problem), by the rule a standalone strip refuses the chain by.
_ROWS_KINDS = {
    "act-token-on-conv": "per-token activation scales on a convolution",
    "act-token-width": "per-token activation scales over other rows",
}


def _record(chain, node):
    """Put the chain's scheme on a consumer that takes it over."""
    if chain.per_tensor:
        node.meta["esp32_act_per_tensor"] = True
    else:
        node.meta["esp32_act_quant"] = chain.scheme


def _claim_problem(support, chain, dq, views, node):
    """Why the gate will not claim `node` to quantise the chain's activation
    itself (Esp32OperatorSupport._quantised_problem), or None, asked about
    the node the partitioner will see: one carrying the chain's record,
    which the gate reads its exactness bounds from ([act-dot-bound],
    [act-offset-bound], the 16-bit rules) and keeps no weight float under,
    and reading, through `views`, the tensor before the chain instead of
    `dq`, the copy it reads now. The gate's tests of the activation judge
    that tensor: a buffer the forward writes is no constant to fold, but
    the gate refuses a matmul over it. The record stays on a node the gate
    claims and is taken off one it refuses."""
    saved = {k: node.meta[k] for k in _RECORDS if k in node.meta}
    _record(chain, node)
    first = views[0] if views else node
    index = next(i for i, a in enumerate(first.args) if a is dq)
    first.update_arg(index, chain.x)
    try:
        problem = support._quantised_problem(node)
    finally:
        first.update_arg(index, dq)
    if problem is not None:
        for k in _RECORDS:
            node.meta.pop(k, None)
        node.meta.update(saved)
    return problem


def _keep_behind(end, chain):
    """Mark each matmul or convolution that reads `end` -- a data move left
    reading the chain's dequantised value, which the walk does not pass
    (_walks) or reached past its depth -- as its activation through pure data
    movement (_MOVES), with why it keeps that value (meta esp32_act_kept).
    The chain is computed in place in front of the move; claimed with no
    record, such an op would quantise the model's dequantised value a second
    time its own way, so the partitioner leaves it on that value as for an
    op the gate refuses the chain's record (_readers). Returns the marked
    ops."""
    marked = []

    def visit(node, path):
        for user in list(node.users):
            op = _overload(user)
            if op is not None and op._schema.name in _MOVES:
                visit(user, path + [user.name])
                continue
            idx = (_activation_arg_index(user) if user.op == "call_function"
                   else None)
            if (idx is None or idx >= len(user.args) or user.args[idx] is not node
                    or sum(a is node for a in user.args) != 1
                    or KEPT in user.meta):
                continue
            user.meta[KEPT] = (
                "dynamic chain computed in place",
                f"{user.name} reads {chain.dq.name}'s dequantised value "
                f"through {' -> '.join(path)}, which the strip does not walk: "
                f"the chain is computed in place in front of it, and the "
                f"quantised loop would quantise that value again its own way")
            marked.append(user)

    op = _overload(end)
    if op is not None and op._schema.name in _MOVES:
        visit(end, [end.name])
    return marked


def _readers(chain, support):
    """Each reader of the chain's dequantised value, every copy's, with the
    ends of the paths through it (_ends): [(the copy, the reader, [(moves,
    end, why the end keeps the dequantised value, or None when it takes the
    chain over)])].

    A matmul or convolution takes it over when the gate will claim it to
    quantise its activation itself (_claim_problem), and, under a per-token
    scheme, reads the tokens as its rows (_rows_problem: not a convolution,
    whose activation scale is per tensor). Without a gate (`support` None)
    each one is assumed to, the partitioner refuses one it does not claim
    ([act-consumer-unclaimed]), and _check_rows refuses the chain by name
    for one that cannot take its tokens. With one, a matmul or convolution
    behind an end that keeps the value through more data movement keeps it
    too (_keep_behind)."""
    rows = chain.scheme is not None and chain.scheme.granularity == "row"
    plan = []
    for dq in chain.dqs:
        for reader in list(dq.users):
            ends = []
            for views, end, takes in _ends(reader, dq, rows=rows):
                if not takes:
                    why = f"{end.name} reads the dequantised value"
                    behind = ([] if support is None
                              else _keep_behind(end, chain))
                    if behind:
                        why += (f" (and so does "
                                f"{', '.join(n.name for n in behind)} behind "
                                f"it)")
                else:
                    problem = None
                    if support is not None:
                        tokens = _rows_problem(chain, end)
                        problem = ((_ROWS_KINDS[tokens[0]], tokens[1])
                                   if tokens is not None else
                                   _claim_problem(support, chain, dq, views,
                                                  end))
                    why = (None if problem is None else
                           f"{end.name} stays portable ({problem[0]}: "
                           f"{problem[1]})")
                    if problem is not None:
                        # The gate refuses it again at partitioning, by the
                        # same reason (Esp32OperatorSupport._static_act_problem).
                        end.meta[KEPT] = problem
                ends.append((views, end, why))
            plan.append((dq, reader, ends))
    return plan


def _own_path(graph, x, src, views, end):
    """Point `end` at a copy of `views`, the path it reads `src` through,
    read from `x`: the views it shares with a reader that keeps the
    dequantised value stay on that value."""
    old, new = src, x
    for view in views:
        with graph.inserting_after(view):
            copy = graph.node_copy(
                view, lambda a, old=old, new=new: new if a is old else a)
        old, new = view, copy
    end.replace_input_with(old, new)


def _rows_problem(chain, u):
    """(rule, detail) when consumer `u` cannot take a per-token scheme, which
    must see the tokens as its rows, else None: a convolution, whose
    activation scale is per tensor, or an op reading rows of another width."""
    if chain.scheme is None or chain.scheme.granularity != "row":
        return None
    width = chain.x.meta["val"].shape[-1]
    if "convolution" in str(u.target):
        return ("act-token-on-conv",
                f"per-token scales on convolution {u.name} (a conv's "
                f"activation scale is per tensor)")
    act = u.args[_activation_arg_index(u)]
    if act.meta["val"].shape[-1] != width:
        return ("act-token-width",
                f"{u.name} reads rows of {act.meta['val'].shape[-1]}, not "
                f"the {width}-wide tokens the chooser scaled")
    return None


def _check_rows(chain, consumers):
    """A per-token scheme must see the tokens as its consumers' rows. With
    the partitioner's gate, _readers leaves a consumer that does not on the
    chain's value; a standalone strip refuses the chain by name."""
    for u in consumers:
        problem = _rows_problem(chain, u)
        if problem is not None:
            raise DynamicChainRefused(problem[0], chain.dq.name, problem[1])


def _computed_in_place(graph, chain):
    """The chain's dequantised value, computed by float32 portable ops: each
    one an op the recipe's chooser, quantize and dequantize compute with
    (PT2E's torch/ao/quantization/fx/_decomposed.py and determine_qparams,
    torchao's quant_primitives.py), in their order and precision, so the
    value is the converted model's bit for bit and the runtime runs no
    chooser. Over the whole tensor, or over each token's row for torchao's
    per-token blocks (amin and amax over the last dim):

        lo, hi = min(amin(x), 0), max(amax(x), 0)
        scale  = max(max(-lo, hi) / ((qmax - qmin) / 2), eps)    symmetric
                 max((hi - lo) / (qmax - qmin), eps)             affine
        zp     = 0                                               symmetric
                 clamp(qmin - round(lo / scale), qmin, qmax)     affine
        codes  = clamp(round(x * (1 / scale)) + zp, qmin, qmax)
        value  = (codes - zp) * scale

    PT2E takes 1 / scale in float64 from the float32 scale and multiplies by
    it in float32: that is the float32 reciprocal, since a quotient of float32
    values rounded to 53 bits and then 24 is the quotient rounded to 24.
    torchao's 1.0 / scale is Tensor.__rtruediv__, the reciprocal itself. The
    "+ zp" of the symmetric zero point, 0 (torchao's is the range's
    midpoint, which _torchao_chain requires to be 0), is kept: it turns
    round's -0.0 into the +0.0 an integer code dequantises to.

    That holds for finite activations and for NaN. An infinite activation
    is out of this contract, as it is for the quantised kernels: the scale
    is then infinite, the infinite element's code NaN, and every value
    NaN, where torch's affine chain casts its NaN codes and zero point to
    integers (0 where PyTorch runs; undefined in C++) and dequantises the
    infinite element to +inf, whatever its sign. The symmetric chains give
    NaN on both sides. Repeating torch's cast here is not worth it."""
    from torch.fx.node import map_arg

    from executorch.exir.dialects._ops import ops as exir_ops

    e = exir_ops.edge.aten
    if chain.scheme is None:        # Esp32Quantizer's own spelling
        args = _named_args(chain.chooser)
        qmin, qmax = int(args["quant_min"]), int(args["quant_max"])
        eps, symmetric, row = float(args["eps"]), True, False
    else:
        qmin, qmax, eps = chain.scheme.qmin, chain.scheme.qmax, chain.scheme.eps
        symmetric = chain.scheme.chooser == "symmetric"
        row = chain.scheme.granularity == "row"
    _check_in_place_eps(chain.dq.name, eps)
    x = chain.x
    mode = x.meta["val"].fake_mode

    def op(target, *operands):
        node = graph.call_function(target, operands)
        with mode:
            node.meta["val"] = target(*map_arg(operands,
                                               lambda n: n.meta["val"]))
        return node

    with graph.inserting_before(chain.chooser):
        rank = x.meta["val"].dim()
        dims = ([rank - 1], True) if row else (list(range(rank)),)
        lo = op(e.clamp.default, op(e.amin.default, x, *dims), None, 0.0)
        hi = op(e.clamp.default, op(e.amax.default, x, *dims), 0.0, None)
        if symmetric:
            top = op(e.maximum.default, op(e.neg.default, lo), hi)
            scale = op(e.clamp.default,
                       op(e.div.Scalar, top, (qmax - qmin) / 2), eps, None)
            zp = 0
        else:
            scale = op(e.clamp.default,
                       op(e.div.Scalar, op(e.sub.Tensor, hi, lo),
                          float(qmax - qmin)), eps, None)
            offset = op(e.round.default, op(e.div.Tensor, lo, scale))
            zp = op(e.clamp.default,
                    op(e.add.Scalar, op(e.neg.default, offset), qmin),
                    qmin, qmax)
        codes = op(e.round.default,
                   op(e.mul.Tensor, x, op(e.reciprocal.default, scale)))
        codes = op(e.clamp.default,
                   op(e.add.Scalar if symmetric else e.add.Tensor, codes, zp),
                   qmin, qmax)
        if not symmetric:
            codes = op(e.sub.Tensor, codes, zp)
        value = op(e.mul.Tensor, codes, scale)
        value.meta[CHAIN_VALUE] = chain.dq.name
        return value


def _constants(ep) -> set:
    """The nodes whose values the program's constants alone determine:
    parameters, lifted constants and unmutated buffers, and the pure ops
    over them."""
    sig = ep.graph_signature
    written = set(sig.buffers_to_mutate.values())
    fixed = (set(sig.inputs_to_parameters)
             | set(sig.inputs_to_lifted_tensor_constants)
             | {n for n, fqn in sig.inputs_to_buffers.items()
                if fqn not in written})
    constant = set()
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in fixed:
                constant.add(node)
        elif (node.op == "call_function" and not node.is_impure()
              and all(a in constant for a in node.all_input_nodes)):
            constant.add(node)
    return constant


def _lift(ep, nodes, constant) -> bool:
    """Replace `nodes` (constant ones, in graph order) by the values their own
    ops compute, with the ops on constants before them, on the real tensors
    -- exact by construction, as stock_llm._fold_constant_gather folds a
    gather of constants by constants: a lifted constant for each one that
    anything else reads, and a copy of it for a graph output, as
    constant_prop_pass gives one. False, changing nothing, when such a value
    is not a tensor."""
    import torch
    from executorch.backends.transforms.utils import (
        create_constant_placeholder,
        delete_constant_placeholder,
    )
    from executorch.exir.dialects._ops import ops as exir_ops
    from torch.export.graph_signature import InputKind
    from torch.fx.node import map_arg

    from executorch.backends.esp32s3.operator_support.const_weights import _const_tensor

    graph = ep.graph_module.graph
    frontier = [n for n in nodes if any(u not in nodes for u in n.users)]
    needed, stack = set(), list(nodes)
    while stack:
        node = stack.pop()
        if node not in needed:
            needed.add(node)
            stack += [a for a in node.all_input_nodes if a in constant]
    values = {}
    for node in graph.nodes:
        if node not in needed:
            continue
        if node.op == "placeholder":
            values[node] = _const_tensor(ep, node)
            continue
        with torch.no_grad():
            values[node] = node.target(*map_arg(node.args, values.__getitem__),
                                       **map_arg(node.kwargs, values.__getitem__))
    if not all(isinstance(values[n], torch.Tensor) for n in frontier):
        return False
    first = next(n for n in graph.nodes if n.op == "placeholder")
    output = next(n for n in graph.nodes if n.op == "output")
    for node in frontier:
        with graph.inserting_before(first):
            const = create_constant_placeholder(
                ep, graph, f"{node.name}_folded", InputKind.CONSTANT_TENSOR,
                values[node].contiguous())
        returned = node in output.all_input_nodes
        node.replace_all_uses_with(const)
        if returned:
            with graph.inserting_before(output):
                copy = graph.call_function(
                    exir_ops.edge.dim_order_ops._clone_dim_order.default,
                    (const,))
            copy.meta["val"] = const.meta["val"]
            output.replace_input_with(const, copy)
            for spec in ep.graph_signature.output_specs:
                if getattr(spec.arg, "name", None) == node.name:
                    spec.arg.name = copy.name
    for node in reversed(list(nodes)):
        if not node.users:
            graph.erase_node(node)
    graph.eliminate_dead_code()
    for node in needed:
        if node.op == "placeholder" and not node.users:
            delete_constant_placeholder(ep, node)
    graph.lint()
    ep.graph_module.recompile()
    return True


# Pure data movement, which a folded chain's value can reach its readers
# through: a Linear over rank-3 learned queries reads it through a view.
_MOVES = frozenset({
    "aten::view_copy", "aten::clone", "aten::alias_copy", "aten::permute_copy",
    "aten::squeeze_copy", "aten::unsqueeze_copy", "aten::expand_copy",
    "aten::repeat", "aten::slice_copy", "aten::select_copy",
    "dim_order_ops::_to_dim_order_copy", "dim_order_ops::_clone_dim_order"})


def _fold_constant_chains(ep) -> int:
    """Fold each chain over program constants into lifted constants (_lift):
    a chooser whose input the program's constants alone determine, the
    getitems after it, the quantizes and dequantizes reading them and the
    data movement after those (_MOVES). Returns how many choosers were
    folded.

    A chooser over constants chooses once: its chain is a constant, not a
    dynamic quantiser, whole or not. convert_pt2e folds such a chain's
    quantize into int8 codes and leaves its chooser (a Linear over learned
    query tokens, or over an op on them), so no chain is left for an op to
    absorb, and the chooser left in the program would run ExecuTorch's
    choose_qparams. A chain still whole (torchao's, which convert_pt2e does
    not touch, or convert_pt2e's with fold_quantize=False), absorbed, would
    hand its op a constant activation, which the gate refuses, and computed
    in place it would hand a claimed op a value the delegate then quantises
    its own way. Folded, the op reads a lifted constant and the gate leaves
    it to constant propagation. That needs the views after the chain folded
    too: nn.Linear over rank-3 queries is view -> addmm -> view, and the gate
    sees a view of a constant as a runtime value (it folds only a weight's
    transposes and slices), so it would claim the addmm with no record and
    quantise the model's dequantised constant again its own way."""
    import operator

    from executorch.backends.esp32s3.partition.espressif_partitioner import _recipe_qdq

    constant = _constants(ep)
    chain = {}      # the nodes to fold, in graph order
    for node in ep.graph_module.graph.nodes:
        if node not in constant or node.op != "call_function":
            continue
        op = _overload(node)
        if _is_chooser(node) or (
                any(a in chain for a in node.all_input_nodes)
                and (node.target is operator.getitem or _recipe_qdq(node)
                     or (op is not None and op._schema.name in _MOVES))):
            chain[node] = None
    if not chain or not _lift(ep, chain, constant):
        return 0
    folded = sum(1 for n in chain if _is_chooser(n))
    print(f"esp32: {folded} dynamic activation chain(s) over program "
          f"constants folded at build time")
    return folded


def fold_portable_torchao(ep) -> int:
    """Fold each torchao quantize or dequantize over program constants that
    partitioning left outside every delegate -- the weight of an op the gate
    refused -- into the lifted constant its own op computes (_lift):
    torchao's ops have no ExecuTorch kernels, and constant_prop_pass skips
    them, so left there the program would not emit. Constant propagation
    then folds what reads it. Returns how many were folded."""
    from executorch.backends.esp32s3.partition.espressif_partitioner import _recipe_qdq

    constant = _constants(ep)
    nodes = {n: None for n in ep.graph_module.graph.nodes
             if n in constant and n.op == "call_function" and _recipe_qdq(n)
             and _overload(n)._schema.name.startswith("torchao::")}
    if not nodes or not _lift(ep, nodes, constant):
        return 0
    print(f"esp32: {len(nodes)} torchao quantize/dequantize op(s) over "
          f"constants outside the delegates folded at build time (no "
          f"ExecuTorch kernel runs them)")
    return len(nodes)


def compute_left_chains(ep) -> int:
    """After partitioning: each dynamic chain still in the graph, computed in
    place by the float32 portable ops torch computes it with
    (_computed_in_place), every reader keeping its dequantised value; then
    torchao's quantizes and dequantizes over constants folded
    (fold_portable_torchao). Returns how many chains it computed.

    For static_act's two-step lowering (quantizer/calibration.py), which
    partitions the to_edge graph without lower_esp32's strip. Under
    static_act the gate refuses every op a dynamic recipe quantises
    (KIND_STATIC_ACT_DYNAMIC), and their chains would otherwise reach
    to_executorch, which has no kernel for a symmetric chooser or for any of
    torchao's ops, and the runtime, whose choose_qparams is not PyTorch's.
    The strip leaves no chooser behind (it absorbs a chain, computes it in
    place or refuses it), so for a program lower_esp32 lowered this changes
    nothing. A chain this backend cannot compute exactly is refused by the
    strip's rules (DynamicChainRefused), as it would be in lower_esp32."""
    graph = ep.graph_module.graph
    computed = 0
    if any(_is_chooser(n) for n in graph.nodes):
        _fold_constant_chains(ep)
        done = set()
        for dq in list(graph.nodes):
            if dq in done:
                continue
            if _op_is(dq, "torchao::dequantize_affine"):
                chain = _torchao_chain(dq)
            elif _op_is(dq, "quantized_decomposed::dequantize_per_tensor",
                        "tensor"):
                chain = _pt2e_chain(dq)
            else:
                continue
            if chain is None:
                continue
            done.update(chain.dqs)
            value = _computed_in_place(graph, chain)
            for copy in chain.dqs:
                for reader in list(copy.users):
                    reader.replace_input_with(copy, value)
                    if reader.op == "output":
                        for spec in ep.graph_signature.output_specs:
                            if getattr(spec.arg, "name", None) == copy.name:
                                spec.arg.name = value.name
            print(f"esp32: leaving {chain.dq.name} in place: no delegate "
                  f"reads it quantised; its chooser, quantize and dequantize "
                  f"run as the float32 portable ops torch computes them with")
            computed += 1
        if computed:
            graph.eliminate_dead_code()
            graph.lint()
            ep.graph_module.recompile()
    if any(n.op == "call_function" and "torchao" in str(n.target)
           for n in graph.nodes):
        fold_portable_torchao(ep)
    return computed


def strip_dynamic_act_qdq(edge_manager, method_name: str = "forward",
                          partitioner=None) -> int:
    """Absorb dynamic-activation fake-quant chains into the matmuls and convs
    that consume them, recording each chain's exact scheme on them; returns
    how many were absorbed. `partitioner` is the Esp32Partitioner about to
    run, whose gate decides which consumers will be claimed (lower_esp32
    passes its own).

    A dynamically quantised model carries, on every quantised layer's input,

        chooser(x) -> (scale, zp)                  torchao choose_qparams_affine,
                                                   or PT2E choose_qparams(_symmetric)
        quantize(x, scale, zp) -> dequantize(^)    -> x'

    which is FAKE quantisation of a dynamic scheme: the scale is recomputed
    from x at every call. The delegate quantises its activations for real at
    execute, so the chain comes out of the graph (left in, it would quantise
    twice and strand torchao ops that have no ExecuTorch kernels) and its
    SCHEME goes onto each consumer as node.meta["esp32_act_quant"], a
    mirrors.ActScheme the runtime reproduces operation for operation
    (EXT_ACT_QUANT): torchao per-token affine (A3) and symmetric
    (A7, its own grid: (qmax - qmin) / 2, ties to even, the eps floor), PT2E
    per-tensor affine (A2: XNNPACK's eps 2^-12, Arm's 2^-16) and symmetric.
    Esp32Quantizer's own chain ([-127, 127], FLT_EPSILON) keeps its own
    spelling, node.meta["esp32_act_per_tensor"] (FLAG_ACT_PER_TENSOR).

    Only chains whose scale comes from a chooser ON THE SAME TENSOR are
    dynamic; a static pair (a calibrated constant scale) is left for
    static_qdq.absorb_static_qdq, which lower_esp32 runs
    right after this pass. Nor is a chooser over program constants whose
    quantize convert_pt2e folded into codes: its chain folds to the constant
    it computes, first (_fold_constant_chains).

    A matmul or convolution reading a chain's dequantised value, through
    views and the other moves the walk passes (_walks: a permute or an
    unsqueeze too, for a scheme over the whole tensor), takes the chain over
    when the partitioner's gate will claim it to
    quantise its own activation, asked about the op as the partitioner will
    see it, with the chain's record on it and reading the tensor before the
    chain (_claim_problem); it then reads that tensor, through its own copy
    of any view it shares with another reader. Every other reader keeps the
    dequantised value: a portable op, a graph output, a weight operand, and
    an op the gate refuses (a convolution over a batch of images, a Linear
    torch spells as a bmm against a broadcast weight, a dot past its
    exactness bound, a Linear over a buffer the forward writes), and one
    that cannot take a per-token scheme (a convolution), which
    carries the gate's reason (meta esp32_act_kept) so that the partitioner
    leaves it on that value. So does a matmul or convolution behind a move
    the walk does not pass (an expand, a permute under a per-token scheme)
    or past its depth (_keep_behind).
    For those the chain stays in place, said so ("esp32: leaving ..."), its
    chooser, quantize and dequantize written out as the float32 portable ops
    the recipe computes them with (_computed_in_place): the value is the
    converted model's bit for bit, and the runtime runs no chooser
    (ExecuTorch's choose_qparams is not PyTorch's, and it has no symmetric
    one), nor any of torchao's ops, which have no ExecuTorch kernels. A
    chooser no chain here reads cannot stay: the final sweep refuses any
    chooser still in the graph, whatever its spelling
    ([act-chooser-unabsorbed]).

    Without a partitioner (a standalone call) the strip cannot know which
    matmuls and convolutions will be claimed: each one is assumed to be, and
    the partitioner refuses one it does not claim ([act-consumer-unclaimed]),
    since run portable it would compute on the unquantised tensor; and a
    chain any other reader needs is refused ([act-chain-consumer]) rather than
    computed in place.

    A chain read by several matmuls is shared: absorbed once, its scheme
    recorded on each, and the runtime quantises the row once for the ops
    whose records are identical. So are its identical copies: one chooser
    feeding a quantize per consumer, or one quantize a dequantize per
    consumer (DuplicateDQPass's), absorbed as the one chain they all
    compute; a copy that differs in any argument is refused
    ([act-chooser-users]).

    Must run BEFORE partitioning (the whole point is what the partitioner then
    sees), which is why it is not part of apply_passes -- those run on the
    lowered program. lower_esp32 runs it on the canonical graph the
    partitioner sees; a second run finds nothing to absorb.
    """
    ep = edge_manager.exported_program(method_name)
    graph = ep.graph_module.graph
    _fold_constant_chains(ep)
    support = None if partitioner is None else partitioner.gate(ep)
    absorbed = 0
    changed = False
    done = set()
    for dq in list(graph.nodes):
        if dq in done:
            continue            # a copy taken with its chain
        if _op_is(dq, "torchao::dequantize_affine"):
            chain = _torchao_chain(dq)
        elif _op_is(dq, "quantized_decomposed::dequantize_per_tensor", "tensor"):
            chain = _pt2e_chain(dq)
        else:
            continue
        if chain is None:
            continue
        done.update(chain.dqs)
        plan = _readers(chain, support)
        takers = [end for _, _, ends in plan for _, end, why in ends
                  if why is None]
        _check_rows(chain, takers)
        stays = [why for _, _, ends in plan for _, _, why in ends if why]
        if stays and support is None:
            raise DynamicChainRefused(
                "act-chain-consumer", chain.dq.name,
                f"its dequantised value feeds something other than a matmul "
                f"or conv activation ({stays[0]}); without the partitioner's "
                f"gate no chain is computed in place (lower_esp32's strip "
                f"computes it in portable ops)")
        value = _computed_in_place(graph, chain) if stays else None
        for copy, reader, ends in plan:
            for _, end, why in ends:
                if why is None:
                    _record(chain, end)
            if all(why is None for _, _, why in ends):
                reader.replace_input_with(copy, chain.x)
                continue
            # The ends that take the chain over get their own views of the
            # tensor before it; the reader keeps the dequantised value.
            for views, end, why in ends:
                if why is None:
                    _own_path(graph, chain.x, copy, views, end)
            reader.replace_input_with(copy, value)
            if reader.op == "output":
                for spec in ep.graph_signature.output_specs:
                    if getattr(spec.arg, "name", None) == copy.name:
                        spec.arg.name = value.name
        if takers:
            absorbed += 1
        if stays:
            print(f"esp32: leaving {chain.dq.name} in place: "
                  f"{'; '.join(stays)}; its chooser, quantize and dequantize "
                  f"run as the float32 portable ops torch computes them with")
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        ep.graph_module.recompile()
    left = [n for n in graph.nodes if _is_chooser(n)]
    if left:
        op = _overload(left[0])
        raise DynamicChainRefused(
            "act-chooser-unabsorbed", left[0].name,
            f"{op._schema.name}.{op._overloadname} feeds no chain this backend "
            f"absorbs; left in the program it would run ExecuTorch's portable "
            f"choose_qparams, which disagrees with PyTorch")
    return absorbed
