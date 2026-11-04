# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Esp32Quantizer: its defaults, its refusals, its presets
against the stock quantizers on the census models, and the torchao helper.

The presets' claim (quantizer/espressif_quantizer.py's docstring): the same SCHEME as each
stock config on the ops this backend computes in integers. Checked here on
every census model (quant_census.models, the census's own exports), one test
case per preset and model, and both ways of preparing (PTQ; QAT where the
stock config has one):

  1. every edge Esp32Quantizer annotates -- a matmul or convolution's
     activation, weight and bias, and under a static scheme its output, on the
     activation the stock quantizer fuses -- carries the same EFFECTIVE
     observer or fake-quantiser as the stock quantizer's annotation of that
     edge: built by PT2E's own _create_obs_or_fq_from_qspec and compared by
     class, dtype, range, scheme, eps, axis and averaging constant; an int32
     bias by its derived spec and by what its derive function returns;
  2. what only the stock quantizer annotates is named, op by op (STOCK_ONLY),
     and anything unnamed fails: those are the tensors the backend computes in
     float between its delegates (Stage A), which the preset leaves in float;
  3. after prepare_pt2e, the same calibration and convert_pt2e, the two
     converted programs carry bit-identical weights, biases and activation
     qparams at every matmul and convolution, except where the stock
     quantizer's observer is shared with a tensor the preset leaves in float
     (it then observes more values), which is named per edge.

The one deliberate difference in a spec: under QAT the stock configs' fused
fake-quantiser (FusedMovingAvgObsFakeQuantize) becomes FakeQuantize over the
same observer (QAT_SWAP), because the fused kernel does not train on the grid
convert_pt2e exports -- test_fused_fake_quant_trains_another_grid measures it.

A pair is left uncompared only where the stock quantizer itself cannot
annotate the census model (STOCK_FAILS), and then its case checks that the
stock quantizer still raises its named cause and that the preset annotates the
model: a listed pair that starts annotating fails, as the census's strict
expectations do, so the comparison resumes the day upstream fixes it.

Arm's and Cortex-M's quantizers import tosa_serializer; where it is missing,
quant_census's stand-in supplies the names (they do no TOSA work here). A case
whose stock quantizer would need the real package is skipped, by name.
"""

import copy
import io
import contextlib
from collections import Counter, defaultdict

import pytest
import torch
import torch.nn as nn

from executorch.backends.esp32s3.quantizer import espressif_quantizer as eq
from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
    Esp32Quantizer,
    SchemeRefused,
)

# One thread: two calibrations are compared bit for bit, and multi-threaded
# float reductions differ from run to run (and the Arm packages load a second
# OpenMP runtime, under which torch.histc can abort in a fresh process).
torch.set_num_threads(1)

# --- the stock quantizers ---------------------------------------------------------------

STOCK_QAT = ("xnnpack_static", "xnnpack_dynamic", "arm_a8w8", "arm_a8w4",
             "arm_a16w8")   # Cortex-M ships no QAT config


def stock_quantizer(preset, is_qat=False):
    """The stock quantizer each preset reproduces, as the census builds it."""
    if preset.startswith("xnnpack"):
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=preset == "xnnpack_dynamic",
            is_qat=is_qat)
        return XNNPACKQuantizer().set_global(config)
    from executorch.backends.esp32s3.test.quant_census import recipes

    recipes.tosa_serializer_source()
    if preset.startswith("arm"):
        from executorch.backends.arm.ethosu import EthosUCompileSpec
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer, get_symmetric_a16w8_quantization_config,
            get_symmetric_quantization_config)
        from executorch.backends.arm.quantizer.arm_quantizer import (
            get_symmetric_a8w4_quantization_config)
        make = {"arm_a8w8": get_symmetric_quantization_config,
                "arm_a8w4": get_symmetric_a8w4_quantization_config,
                "arm_a16w8": get_symmetric_a16w8_quantization_config}[preset]
        return EthosUQuantizer(EthosUCompileSpec("ethos-u55-128")).set_global(
            make(is_qat=is_qat))
    assert not is_qat
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    return CortexMQuantizer()


def preset_quantizer(preset, is_qat=False):
    """The preset, built without its construction checks."""
    return Esp32Quantizer.from_scheme(
        eq.resolve_scheme(preset, is_qat=is_qat), check=False)


def census_graph(name, channels_last=False):
    from executorch.backends.esp32s3.test.quant_census import models, recipes

    if name in models._ZOO and not models.zoo_available():
        pytest.skip("needs the nano zoo and transformers")
    net, inputs, calibration = models.model(name, 0)
    if channels_last:
        inputs = recipes._channels_last(inputs)
        calibration = [recipes._channels_last(b) for b in calibration]
    gm = torch.export.export(net, inputs, strict=False).module(check_guards=False)
    return gm, inputs, calibration


def annotate_like_prepare(gm, quantizer, is_qat):
    """What prepare_pt2e / prepare_qat_pt2e do before inserting observers:
    PTQ folds conv + batch norm first; both then transform and annotate."""
    from torchao.quantization.pt2e.utils import _fuse_conv_bn_

    gm = copy.deepcopy(gm)
    if not is_qat:
        _fuse_conv_bn_(gm)
    with contextlib.redirect_stdout(io.StringIO()):
        gm = quantizer.transform_for_annotation(gm)
        quantizer.annotate(gm)
    return gm


# --- effective specs -------------------------------------------------------------------

_ATTRS = ("dtype", "qscheme", "quant_min", "quant_max", "ch_axis",
          "averaging_constant", "reduce_range", "bins", "upsample_rate",
          "is_dynamic")
QAT_SWAP = {"FusedMovingAvgObsFakeQuantize": "FakeQuantize"}


def describe(module, swap=False):
    """An observer or fake-quantiser as the numbers it will produce depend on.
    DynamicFakeQuantize is FakeQuantize exposing its observer's eps (compared
    in "observer"); test_dynamic_qat_exports_its_eps covers the difference."""
    out = {"class": type(module).__name__}
    if swap:
        out["class"] = QAT_SWAP.get(out["class"], out["class"])
    if out["class"] == "DynamicFakeQuantize":
        out["class"] = "FakeQuantize"
    if out["class"] == "CallRangeObserver":    # exact averaging_constant=1
        out["class"] = "MovingAverageMinMaxObserver"
    for a in _ATTRS:
        if hasattr(module, a):
            v = getattr(module, a)
            out[a] = v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
    fake_quant = hasattr(module, "activation_post_process")
    if hasattr(module, "eps") and not fake_quant:
        out["eps"] = float(module.eps)
    if fake_quant:
        out["observer"] = describe(module.activation_post_process)
    return out


def effective(spec, node, is_qat, swap=False):
    """The spec at one edge of `node` as PT2E will build it."""
    from torchao.quantization.pt2e.prepare import _create_obs_or_fq_from_qspec
    from torchao.quantization.pt2e.quantizer import (
        DerivedQuantizationSpec, FixedQParamsQuantizationSpec, QuantizationSpec,
        SharedQuantizationSpec)

    if spec is None:
        return None
    if isinstance(spec, QuantizationSpec):
        return describe(_create_obs_or_fq_from_qspec(spec, {}, is_qat), swap)
    if isinstance(spec, DerivedQuantizationSpec):
        roles = []
        for producer, consumer in spec.derived_from:
            role = {id(node.args[0]): "activation",
                    id(node.args[1]): "weight"}.get(id(producer), producer.name)
            roles.append((role, consumer is node))
        return {"class": "derived", "dtype": spec.dtype, "qmin": spec.quant_min,
                "qmax": spec.quant_max, "qscheme": spec.qscheme,
                "ch_axis": spec.ch_axis, "from": tuple(roles)}
    if isinstance(spec, SharedQuantizationSpec):
        return {"class": "shared"}
    if isinstance(spec, FixedQParamsQuantizationSpec):
        return {"class": "fixed", "scale": spec.scale, "zp": spec.zero_point}
    return {"class": type(spec).__name__}


def derive_fns_agree(stock_spec, preset_spec):
    """The two int32-bias derive functions on the same observers, bitwise."""
    from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver

    g = torch.Generator().manual_seed(3)
    act = MinMaxObserver(dtype=torch.int8, qscheme=torch.per_tensor_affine)
    act(torch.randn(4, 16, generator=g))
    per_channel = preset_spec.qscheme == torch.per_channel_symmetric
    weight = (PerChannelMinMaxObserver(dtype=torch.int8, ch_axis=0,
                                       qscheme=torch.per_channel_symmetric)
              if per_channel else
              MinMaxObserver(dtype=torch.int8, qscheme=torch.per_tensor_symmetric))
    weight(torch.randn(8, 16, generator=g))
    a = stock_spec.derive_qparams_fn([act, weight])
    b = preset_spec.derive_qparams_fn([act, weight])
    return all(torch.equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b))


# --- comparing the annotated graphs ----------------------------------------------------

_WEIGHTED = ("aten.linear", "aten.conv1d", "aten.conv2d", "aten.conv3d",
             "aten.convolution")

# What only a stock quantizer annotates, op by op: the tensors the backend
# computes in float between its delegates (Stage A), which the preset leaves
# in float. A key is (preset family, op name as str(target) shortened).
_FLOAT = ("the backend computes this op in float between its delegates "
          "(Stage A); the preset quantises only matmul and convolution "
          "inputs, weights, biases and (static) outputs")
STOCK_ONLY = {
    "xnnpack": {
        **dict.fromkeys(("add.Tensor", "mul.Tensor", "cat"),
                        "XNNPACK's add/mul/cat annotators; " + _FLOAT),
        "adaptive_avg_pool2d": "XNNPACK shares the pool's qparams with its "
                               "input; " + _FLOAT,
        **dict.fromkeys(("relu", "relu_", "hardtanh", "max_pool2d", "mean.dim",
                         "permute", "transpose.int", "squeeze.dim", "view",
                         "reshape", "slice.Tensor", "flatten.using_ints",
                         "dropout"),
                        "XNNPACK's propagate_annotation shares its input's "
                        "output qparams through this op; " + _FLOAT),
        "batch_norm": "XNNPACK's conv + batch norm QAT patterns put the pair "
                      "after the norm where the conv is shared; " + _FLOAT,
    },
    "arm": {"*": "Arm's support table annotates nearly every op (one-to-one "
                 "activations, binaries, pools, shape ops, the products of its "
                 "transform_for_annotation decompositions -- layer norm, "
                 "softmax, batch norm -- and the model's inputs); " + _FLOAT},
    "cortex_m": {"*": "Cortex-M's support table (binaries, LUT activations, "
                      "softmax at CMSIS-NN's fixed 1/256 grid, pools, shape "
                      "ops, the model's inputs); " + _FLOAT},
}
# What only the preset annotates, with why.
PRESET_ONLY = {
    "cortex_m": {"conv2d": "CortexMConv2DCheck annotates a convolution only "
                           "when its tensors are channels-last (CMSIS-NN's "
                           "layout); the delegate runs convolutions on "
                           "contiguous tensors, so the preset annotates every "
                           "conv2d"},
}


def family(preset):
    return preset.split("_")[0] if not preset.startswith("cortex") else "cortex_m"


def short(target):
    return (str(target).replace("aten.", "").replace(".default", "")
            .replace("<built-in function getitem>", "getitem"))


def weighted_ops(gm, by_name=False):
    """{key: node} for every matmul and convolution, keyed by its weight's
    parameter name (annotated graphs: stable through the stock quantizers'
    transforms), or by its own name (converted graphs, whose weights are
    dequantize nodes)."""
    out = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and any(str(n.target).startswith(w)
                                           for w in _WEIGHTED):
            w = n.args[1]
            key = (n.name if by_name or getattr(w, "op", None) != "get_attr"
                   else w.target)
            out[key] = n
    return out


def annotation(n):
    a = n.meta.get("quantization_annotation")
    if a is None or (not a.input_qspec_map and a.output_qspec is None):
        return None
    return a


def output_node(n):
    """The node carrying `n`'s output pair: `n`, or the chain of single users
    (batch norm, activation) ending at the first node with an output spec."""
    end = n
    for _ in range(3):
        a = annotation(end)
        if a is not None and a.output_qspec is not None:
            return end
        if len(end.users) != 1:
            return None
        end = next(iter(end.users))
        if not (end.op == "call_function"
                and short(end.target).split(".")[0] in (
                    "batch_norm", "relu", "relu_", "hardtanh", "hardtanh_",
                    "clamp", "clamp_", "hardsigmoid", "hardsigmoid_")):
            return None
    return None


def annotated_pair(preset, model, is_qat):
    """The census model's graph, its calibration, and the stock and preset
    annotations of it."""
    gm, _inputs, calibration = census_graph(model,
                                            channels_last=preset == "cortex_m")
    stock = annotate_like_prepare(gm, stock_quantizer(preset, is_qat), is_qat)
    mine = annotate_like_prepare(gm, preset_quantizer(preset, is_qat), is_qat)
    return gm, calibration, stock, mine


def stock_only_names(stock, mine):
    """Annotated stock nodes the preset's annotations do not cover: every
    stock node with an annotation but the matmuls and convolutions the preset
    annotates and the node carrying each one's output pair."""
    s_ops, m_ops = weighted_ops(stock, True), weighted_ops(mine, True)
    covered = set()
    for k, m in m_ops.items():
        if k in s_ops and annotation(m) is not None:
            covered.add(k)
            end = output_node(s_ops[k])
            if end is not None:
                covered.add(end.name)
    return {n.name for n in stock.graph.nodes
            if annotation(n) is not None and n.name not in covered}


def compare_annotations(preset, model, is_qat):
    """(edges compared, spec differences, stock-only {op: n}, preset-only {op: n}).
    Ops are matched by name: the stock quantizers' transforms rename weights
    (_param_constantN), not ops."""
    _gm, _cal, stock, mine = annotated_pair(preset, model, is_qat)
    s_ops, m_ops = weighted_ops(stock, True), weighted_ops(mine, True)
    compared, diffs = 0, []
    preset_only, moved = Counter(), Counter()
    for key, m in m_ops.items():
        ma = annotation(m)
        if ma is None:
            continue
        s = s_ops.get(key)
        sa = annotation(s) if s is not None else None
        if sa is None:
            preset_only[short(m.target)] += 1
            continue
        for arg_i, role in ((0, "activation"), (1, "weight"), (2, "bias")):
            if arg_i >= len(m.args) or not isinstance(m.args[arg_i], torch.fx.Node):
                continue
            m_spec = ma.input_qspec_map.get(m.args[arg_i])
            s_spec = sa.input_qspec_map.get(s.args[arg_i])
            if m_spec is None and s_spec is None:
                continue
            compared += 1
            me = effective(m_spec, m, is_qat)
            se = effective(s_spec, s, is_qat, swap=is_qat)
            if me != se:
                diffs.append((key, role, se, me))
            elif role == "bias" and m_spec is not None and not derive_fns_agree(
                    s_spec, m_spec):
                diffs.append((key, "bias derive", "stock", "preset"))
        m_end, s_end = output_node(m), output_node(s)
        if m_end is not None or s_end is not None:
            compared += 1
            where_m = None if m_end is None else short(m_end.target)
            where_s = None if s_end is None else short(s_end.target)
            me = None if m_end is None else effective(
                annotation(m_end).output_qspec, m_end, is_qat)
            se = None if s_end is None else effective(
                annotation(s_end).output_qspec, s_end, is_qat, swap=is_qat)
            if (where_m, me) != (where_s, se):
                placement = (where_m != where_s and me == se and
                             (family(preset), is_qat) in OUTPUT_PLACEMENT)
                if placement:
                    moved[(where_s, where_m)] += 1
                else:
                    diffs.append((key, "output", (where_s, se), (where_m, me)))
    stock_only = Counter()
    names = stock_only_names(stock, mine)
    for n in stock.graph.nodes:
        if n.name in names:
            stock_only["input" if n.op == "placeholder" else short(n.target)] += 1
    for (where_s, where_m), n in moved.items():
        preset_only[f"output pair on {where_m}, not {where_s}"] += n
    return compared, diffs, dict(stock_only), dict(preset_only)


def unnamed(preset, stock_only, preset_only):
    """The differences with no entry in STOCK_ONLY / PRESET_ONLY."""
    named = STOCK_ONLY[family(preset)]
    bad = [f"stock-only {op}" for op in stock_only
           if "*" not in named and op not in named]
    bad += [f"preset-only {op}" for op in preset_only
            if op not in PRESET_ONLY.get(family(preset), {})
            and not op.startswith("output pair on ")]
    return bad


def census_models():
    """The models every census recipe runs, the esp32-* presets' included: not
    llama3_2_device or et_llama, which run only the few recipes their spec
    names. The nano zoo's are listed whether or not it imports; their cases
    skip, by name, where it does not (census_graph)."""
    from executorch.backends.esp32s3.test.quant_census.models import MODELS

    return [m for m, spec in MODELS.items() if spec.recipes is None]


# Census models the stock quantizer itself cannot annotate, with the error it
# raises: they fail inside the recipe, before the backend sees the model.
# Arm's transform_for_annotation fails on HF GPT-2's Conv1D addmm, the census's
# ARM_PASS_HF (quant_census/expect.py), for each of its configs. XNNPACK's
# quantizer annotates llama3_2 and gpt2 since 1.6: it lifts only float32
# scalars to tensors and leaves integer adds and muls alone (#22065, #22804),
# so their integer position arithmetic does not trip it.
STOCK_FAILS = dict.fromkeys(
    [("arm_a8w8", "gpt2"), ("arm_a8w4", "gpt2"), ("arm_a16w8", "gpt2")],
    "'Parameter' object has no attribute 'to_tensor'")


def _error_chain(error):
    """`error` and every exception it was raised from, as messages."""
    out, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        out.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__
    return out


def _skip_if_stub(error, preset):
    from executorch.backends.esp32s3.test.quant_census import recipes

    if recipes.is_stub_error(error):
        pytest.skip(f"the stock {preset} quantizer needs the real tosa_serializer "
                    f"here (quant_census's stand-in raised): {error}")


def check_stock_fails(preset, model, is_qat):
    """A STOCK_FAILS pair: the stock quantizer still raises its named cause,
    and the preset annotates the model it cannot."""
    cause = STOCK_FAILS[(preset, model)]
    gm, _inputs, _calibration = census_graph(model,
                                             channels_last=preset == "cortex_m")
    try:
        annotate_like_prepare(gm, stock_quantizer(preset, is_qat), is_qat)
    except Exception as e:  # noqa: BLE001 -- the cause is checked below
        _skip_if_stub(e, preset)
        chain = _error_chain(e)
        assert any(cause in m for m in chain), (
            f"the stock {preset} quantizer raises on {model}, but not its named "
            f"cause {cause!r}: " + " <- ".join(m[:300] for m in chain))
    else:
        pytest.fail(f"the stock {preset} quantizer annotates {model} on this "
                    f"tree: remove ({preset!r}, {model!r}) from STOCK_FAILS so "
                    f"the pair is compared")
    mine = annotate_like_prepare(gm, preset_quantizer(preset, is_qat), is_qat)
    annotated = sum(annotation(n) is not None for n in mine.graph.nodes)
    assert annotated, f"the {preset} preset annotates nothing in {model}"
    print(f"\n{preset}{' (QAT)' if is_qat else ''} {model}: the stock quantizer "
          f"raises {cause!r}; the preset annotates {annotated} node(s)")


# Output pairs placed differently, with why: (family, is_qat) -> reason.
OUTPUT_PLACEMENT = {
    ("arm", True): "Arm's conv + batch norm patterns do not match the batch norm "
                   "of an eval-mode export (the census exports eval-mode models "
                   "for QAT, as prepare_qat_pt2e's docstring does), so it puts "
                   "the pair before the norm; the preset puts it after the norm "
                   "and the activation, where prepare_qat_pt2e folds the norm "
                   "into the conv (Arm does too on a train-mode export)",
    ("cortex_m", False): "Cortex-M's transform_for_annotation inserts a clamp "
                         "in front of hardswish (its LUT's input range) and its "
                         "(conv2d, clamp) pattern fuses it: the pair moves past "
                         "an op the preset's graph does not have",
}


# Census models whose converted programs a preset's family cannot match
# (claim 3), with why: (family, model) -> reason. Run as strict expected
# failures: the day a pair matches, its entry has to go.
CONVERTED_GAPS = dict.fromkeys(
    [("arm", "lstm"), ("arm", "gru")],
    "Arm's transform_for_annotation decomposes nn.LSTM and nn.GRU "
    "(DecomposeLstmPass, DecomposeGruPass) and quantises every gate matmul; "
    "the preset annotates the export as it is, where the layer is one "
    "aten.lstm or aten.gru node, so it quantises the head alone and the "
    "gates stay float (claims 1 and 2 pass for these pairs only because "
    "both graphs name a matmul `linear`: the preset's head, stock's first "
    "gate projection)")


def _cases(qat=(False, True), gaps=None):
    """(preset, is_qat, model) for every preset, each way of preparing its
    stock config ships, and every census model; a pair `gaps` names is a
    strict expected failure."""
    out = []
    for preset in eq.PRESETS:
        for is_qat in qat:
            if is_qat and preset not in STOCK_QAT:
                continue
            for model in census_models():
                reason = (gaps or {}).get((family(preset), model))
                marks = ((pytest.mark.xfail(strict=True, reason=reason),)
                         if reason else ())
                out.append(pytest.param(preset, is_qat, model, id=(
                    f"{preset}{'-qat' if is_qat else ''}-{model}"), marks=marks))
    return out


@pytest.mark.parametrize("preset, is_qat, model", _cases())
def test_preset_annotations_match_stock(preset, is_qat, model):
    """Claims 1 and 2."""
    if (preset, model) in STOCK_FAILS:
        check_stock_fails(preset, model, is_qat)
        return
    try:
        compared, diffs, stock_only, preset_only = compare_annotations(
            preset, model, is_qat)
    except Exception as e:  # noqa: BLE001
        _skip_if_stub(e, preset)
        raise
    bad = unnamed(preset, stock_only, preset_only)
    row = (f"{compared} edges equal" if not diffs
           else f"{len(diffs)} DIFFER of {compared}")
    if stock_only:
        row += "; stock only: " + ", ".join(
            f"{op} {n}" for op, n in sorted(stock_only.items()))
    if preset_only:
        row += "; preset only: " + ", ".join(
            f"{op} {n}" for op, n in sorted(preset_only.items()))
    print(f"\n{preset}{' (QAT)' if is_qat else ''} {model}: {row}")
    failures = [f"{model}/{key}/{role}: stock {stock_side} != preset {preset_side}"
                for key, role, stock_side, preset_side in diffs]
    failures += [f"{model}: unnamed {b}" for b in bad]
    assert not failures, "\n".join(failures)


# --- claim 3: the converted programs ------------------------------------------------

def _behind_views(n):
    while (isinstance(n, torch.fx.Node) and n.op == "call_function"
           and any(v in str(n.target) for v in ("view", "reshape", "flatten",
                                                 "permute", "clone"))):
        n = n.args[0]
    return n


def _value(gm, n):
    if isinstance(n, torch.fx.Node) and n.op == "get_attr":
        return getattr(gm, n.target)
    return n


def _qparams(gm, dq):
    """What a dequantize node carries: its constants (static), its chooser's
    arguments (dynamic), or None."""
    if not (isinstance(dq, torch.fx.Node) and "dequantize" in str(dq.target)):
        return None
    vals = [_value(gm, a) for a in dq.args[1:]]
    if "per_tensor.tensor" in str(dq.target):
        chooser = dq.args[1].args[0]
        return ("dynamic", str(chooser.target),
                tuple(a for a in chooser.args[1:] if not isinstance(a, torch.fx.Node)))
    return tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in vals)


def _same(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    return a == b


def converted(gm, quantizer, calibration, is_qat=False):
    from torchao.quantization.pt2e.quantize_pt2e import (convert_pt2e,
                                                         prepare_pt2e)

    with contextlib.redirect_stdout(io.StringIO()):
        prepared = prepare_pt2e(copy.deepcopy(gm), quantizer)
        with torch.no_grad():
            for batch in calibration:
                prepared(*batch)
        return convert_pt2e(prepared)


def shared_tensors(gm):
    """Producer names of the tensors whose stock observer is shared with
    another tensor (SharedQuantizationSpec): their qparams fold in more values
    than the tensor itself."""
    from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec

    out = set()
    for n in gm.graph.nodes:
        a = annotation(n)
        if a is None:
            continue
        specs = list(a.input_qspec_map.items()) + [(n, a.output_qspec)]
        for edge_src, spec in specs:
            if isinstance(spec, SharedQuantizationSpec):
                ref = spec.edge_or_node
                out.add(ref[0].name if isinstance(ref, tuple) else ref.name)
                out.add(edge_src.name if edge_src is not n else n.name)
    return out


def _output_qparams(gm, end_name):
    """The output pair after the node named `end_name`: the quantize reading it."""
    node = next((n for n in gm.graph.nodes if n.name == end_name), None)
    if node is None:
        return None
    for u in node.users:
        if "quantize_per_tensor" in str(u.target) and "dequantize" not in str(u.target):
            return tuple(_value(gm, a) for a in u.args[1:])
    return None


def compare_converted(preset, model):
    """(edges compared, unexplained mismatches, {named cause: n})."""
    gm, calibration, stock_ann, mine_ann = annotated_pair(preset, model, False)
    sharing = shared_tensors(stock_ann)
    only = stock_only_names(stock_ann, mine_ann)
    s_named = {n.name: n for n in stock_ann.graph.nodes}
    m_ops_by_name = {n.name: short(n.target) for n in mine_ann.graph.nodes
                     if n.op == "call_function"}

    def rewritten_upstream(node):
        """An op upstream of `node` that the stock quantizer's transform put
        in place of the model's own (Cortex-M's avg_pool2d for
        adaptive_avg_pool2d: another kernel, last-ulp different sums)."""
        seen, stack = set(), [node]
        while stack:
            for a in stack.pop().all_input_nodes:
                if a.name in seen:
                    continue
                seen.add(a.name)
                if (a.op == "call_function"
                        and m_ops_by_name.get(a.name) != short(a.target)):
                    return short(a.target)
                stack.append(a)
        return None

    s_conv = converted(gm, stock_quantizer(preset), calibration)
    m_conv = converted(gm, preset_quantizer(preset), calibration)
    s_ops, m_ops = (weighted_ops(s_conv, by_name=True),
                    weighted_ops(m_conv, by_name=True))
    compared, bad, allowed = 0, [], Counter()
    for key, m in m_ops.items():
        s = s_ops.get(key)
        if s is None:
            continue
        s_node = s_named.get(key)
        act_ok = True
        for i, role in ((0, "activation"), (1, "weight"), (2, "bias")):
            if i >= len(m.args):
                continue
            mq = _qparams(m_conv, _behind_views(m.args[i]))
            sq = _qparams(s_conv, _behind_views(s.args[i]))
            if mq is None and sq is None:
                continue
            compared += 1
            if _same(mq, sq):
                continue
            if sq is None:
                allowed["preset-only (the stock quantizer leaves it float)"] += 1
                act_ok = act_ok and role != "activation"
            elif role == "activation":
                act_ok = False
                producer = _behind_views(s_node.args[0]) if s_node is not None else None
                name = getattr(producer, "name", "")
                rewritten = rewritten_upstream(s_node) if s_node is not None else None
                if name in only or name in sharing:
                    allowed["activation: the stock quantizer quantises its "
                            "producer (" + (short(producer.target)
                                            if producer.op == "call_function"
                                            else "input") + ")"] += 1
                elif rewritten:
                    allowed[f"activation: the stock transform rewrote an op "
                            f"upstream ({rewritten})"] += 1
                else:
                    bad.append((key, f"{role} (producer {name})"))
            elif role == "bias" and not act_ok:
                allowed["bias: derived from the activation scale above"] += 1
            else:
                bad.append((key, role))
        m_end = output_node(next(n for n in mine_ann.graph.nodes if n.name == key))
        if m_end is not None:
            mq = _output_qparams(m_conv, m_end.name)
            sq = _output_qparams(s_conv, m_end.name)
            compared += 1
            if not _same(mq, sq):
                if sq is None:
                    allowed["preset-only (the stock quantizer leaves it float)"] += 1
                elif m_end.name in sharing or any(u.name in only for u in
                                                  s_named[m_end.name].users):
                    allowed["output: the stock observer also sees a tensor the "
                            "preset leaves float"] += 1
                elif rewritten_upstream(s_named[m_end.name]):
                    allowed["output: the stock transform rewrote an op upstream "
                            f"({rewritten_upstream(s_named[m_end.name])})"] += 1
                else:
                    bad.append((key, "output"))
    return compared, bad, dict(allowed)


@pytest.mark.parametrize("preset, is_qat, model",
                         _cases(qat=(False,), gaps=CONVERTED_GAPS))
def test_preset_converted_programs_match_stock(preset, is_qat, model):
    """Claim 3: PTQ with the same calibration gives the same numbers."""
    if (preset, model) in STOCK_FAILS:
        check_stock_fails(preset, model, is_qat)
        return
    try:
        compared, bad, allowed = compare_converted(preset, model)
    except Exception as e:  # noqa: BLE001
        _skip_if_stub(e, preset)
        raise
    print(f"\n{preset} converted {model}: {compared} compared, "
          f"{compared - len(bad) - sum(allowed.values())} identical"
          + "".join(f"; {n} {cause}" for cause, n in allowed.items()))
    assert not bad, "unexplained: " + ", ".join(f"{model}/{key}/{role}"
                                                 for key, role in bad)


# --- defaults, refusals, QAT, shared activations -----------------------------------------

def default_specs(bits=4):
    """Esp32Quantizer()'s (weight, activation) specs: weights symmetric per
    output channel at the rung's range in an int8 container (weight_bits=2 the
    ternary [-1, 1]), activations dynamic per-tensor symmetric int8 over
    [-127, 127]."""
    from torchao.quantization.pt2e.observer import (MinMaxObserver,
                                                    PerChannelMinMaxObserver)
    from torchao.quantization.pt2e.quantizer.quantizer import QuantizationSpec

    q = (1 << (bits - 1)) - 1
    weight = QuantizationSpec(
        dtype=torch.int8, quant_min=-q, quant_max=q,
        qscheme=torch.per_channel_symmetric, ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver)
    act = QuantizationSpec(
        dtype=torch.int8, quant_min=-127, quant_max=127,
        qscheme=torch.per_tensor_symmetric, is_dynamic=True,
        observer_or_fake_quant_ctr=MinMaxObserver)
    return weight, act


def test_default_specs():
    """Esp32Quantizer()'s specs for every weight_bits, and no bias spec."""
    for bits in (2, 4, 8):
        q = Esp32Quantizer(weight_bits=bits)
        weight, act = default_specs(bits)
        assert q.specs.weight["linear"] == weight
        assert q.specs.weight["conv"] == weight
        assert q.specs.act == act
        assert q.specs.bias(None, "linear") is None
    assert Esp32Quantizer().scheme == eq.Scheme()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(32, 16), nn.Linear(16, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_default_annotations():
    """The default annotates the activation and weight of each Linear, in that
    order (the one prepare_qat_pt2e's conv + batch norm rewrite re-keys by),
    no output and no bias."""
    gm = torch.export.export(MLP().eval(), (torch.randn(2, 32),)).module()
    q = Esp32Quantizer()
    q.annotate(gm)
    weight, act = default_specs()
    linears = [n for n in gm.graph.nodes if n.target == torch.ops.aten.linear.default]
    for n in linears:
        a = n.meta["quantization_annotation"]
        assert a.output_qspec is None
        assert list(a.input_qspec_map.items()) == [(n.args[0], act),
                                                   (n.args[1], weight)]
    assert q.annotated == 2 and q.act_annotated


@pytest.mark.parametrize("kwargs, rule", [
    (dict(preset="arm_a16w8", weight_bits=4), "int16-weights"),
    (dict(activation="static_per_tensor_symmetric_int16",
          weight_mapping="asymmetric"), "int16-weights"),
    (dict(activation="dynamic_per_tensor_affine", bias="int32_derived"),
     "bias-int32-dynamic"),
    (dict(activation="none", bias="int32_derived"), "bias-int32-dynamic"),
    (dict(activation="dynamic_per_tensor_symmetric", act_qmin=0, act_qmax=255),
     "act-unsigned-symmetric"),
    (dict(activation="static_per_tensor_affine", act_qmin=-200, act_qmax=100),
     "act-range"),
    (dict(eps=0.0), "eps"),
    (dict(weight_eps=-1.0), "eps"),
    (dict(weight_bits=2, weight_mapping="asymmetric"), "weight-bits"),
    (dict(weight_bits=4, weight_qmin=-9, weight_qmax=7), "weight-range"),
    (dict(weight_mapping="symmetric", weight_qmin=0, weight_qmax=7), "weight-range"),
    (dict(weight_granularity="per_group"), "weight-granularity"),
    (dict(weight_granularity={"linear": "per_channel"}), "weight-granularity"),
    (dict(weight_mapping="affine"), "weight-mapping"),
    (dict(activation="dynamic_per_token_affine"), "activation"),
    (dict(bias="int8"), "bias"),
    (dict(preset="tflite"), "preset"),
])
def test_refused_by_name(kwargs, rule):
    with pytest.raises(SchemeRefused) as err:
        Esp32Quantizer(**kwargs)
    assert err.value.rule == rule, str(err.value)
    assert f"[{rule}]" in str(err.value)


def test_presets_construct():
    for preset in eq.PRESETS:
        for qat in (False, True):
            Esp32Quantizer(preset=preset, is_qat=qat)
    # keywords override the preset
    q = Esp32Quantizer(preset="xnnpack_static", weight_granularity="per_tensor")
    assert q.specs.weight["linear"].qscheme == torch.per_tensor_symmetric
    assert q.specs.act.qscheme == torch.per_tensor_affine


def arm_dynamic_quantizer():
    """Arm's dynamic config, get_symmetric_quantization_config(is_dynamic=True),
    on the census's EthosUQuantizer."""
    from executorch.backends.esp32s3.test.quant_census import recipes

    recipes.tosa_serializer_source()
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer, get_symmetric_quantization_config)
    return EthosUQuantizer(EthosUCompileSpec("ethos-u55-128")).set_global(
        get_symmetric_quantization_config(is_dynamic=True))


@pytest.mark.parametrize("model", ["linear_relu", "conv_pad", "deep_autoencoder",
                                   "llama3_2"])
def test_arm_dynamic_config_is_arm_a8w8_made_dynamic(model):
    """check_scheme's bias-int32-dynamic hint: Arm's dynamic config, which
    since 1.6 drops its int32 bias under a dynamic activation (#22879), is
    arm_a8w8 with activation="dynamic_per_tensor_affine" and bias="fp32".
    Every matmul and convolution input carries the same effective spec, and
    after the same calibration and convert_pt2e the same weights and
    choosers. (Arm also shares a fused ReLU's output with the matmul's, which
    a dynamic config gives no qparams.)"""
    def mine():
        return Esp32Quantizer(preset="arm_a8w8",
                              activation="dynamic_per_tensor_affine", bias="fp32")

    gm, _inputs, calibration = census_graph(model)
    stock_ann = annotate_like_prepare(gm, arm_dynamic_quantizer(), False)
    mine_ann = annotate_like_prepare(gm, mine(), False)
    s_ops, m_ops = weighted_ops(stock_ann, True), weighted_ops(mine_ann, True)
    assert set(m_ops) <= set(s_ops), set(m_ops) - set(s_ops)
    edges = 0
    for key, m in m_ops.items():
        s = s_ops[key]
        for i in range(min(3, len(m.args))):
            if not isinstance(m.args[i], torch.fx.Node):
                continue
            specs = [None if annotation(n) is None else
                     annotation(n).input_qspec_map.get(n.args[i]) for n in (s, m)]
            assert effective(specs[0], s, False) == effective(specs[1], m, False), (
                f"{model}/{key}/arg {i}")
            edges += any(sp is not None for sp in specs)
    s_conv = converted(gm, arm_dynamic_quantizer(), calibration)
    m_conv = converted(gm, mine(), calibration)
    s_ops, m_ops = (weighted_ops(s_conv, by_name=True),
                    weighted_ops(m_conv, by_name=True))
    assert set(m_ops) <= set(s_ops), set(m_ops) - set(s_ops)
    qparams = 0
    for key, m in m_ops.items():
        for i in range(min(3, len(m.args))):
            mq = _qparams(m_conv, _behind_views(m.args[i]))
            sq = _qparams(s_conv, _behind_views(s_ops[key].args[i]))
            assert _same(mq, sq), f"{model}/{key}/arg {i}: stock {sq} != preset {mq}"
            qparams += mq is not None
    print(f"\narm dynamic {model}: {edges} annotated edges equal, {qparams} "
          f"converted qparams identical")
    assert edges and qparams


def _fake_quantisers(prepared):
    """The observers and fake-quantisers the prepared graph calls."""
    return Counter(type(prepared.get_submodule(n.target)).__name__
                   for n in prepared.graph.nodes if n.op == "call_module")


@pytest.mark.parametrize("kwargs", [
    dict(), dict(preset="xnnpack_static"), dict(preset="xnnpack_dynamic"),
    dict(preset="arm_a8w8"), dict(preset="cortex_m"),
    dict(activation="static_per_tensor_affine", weight_bits=8),
])
def test_is_qat_inserts_fake_quantisers(kwargs):
    """is_qat=True: every annotated edge gets a FakeQuantize (never the fused
    one); without it prepare_qat_pt2e inserts plain observers, which
    fake-quantise nothing."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e

    x = torch.randn(2, 32)
    gm = torch.export.export(MLP().train(), (x,)).module(check_guards=False)
    with contextlib.redirect_stdout(io.StringIO()):
        qat = _fake_quantisers(prepare_qat_pt2e(
            copy.deepcopy(gm), Esp32Quantizer(is_qat=True, **kwargs)))
        plain = _fake_quantisers(prepare_qat_pt2e(
            copy.deepcopy(gm), Esp32Quantizer(**kwargs)))
    assert qat and set(qat) <= {"FakeQuantize", "DynamicFakeQuantize",
                                "DerivedObserverOrFakeQuantize"}, qat
    assert not any("FakeQuantize" == k for k in plain), plain


def test_fused_fake_quant_trains_another_grid():
    """Why QAT_SWAP: FusedMovingAvgObsFakeQuantize fake-quantises with the
    qparams its C++ kernel computes, convert_pt2e exports calculate_qparams.
    FakeQuantize over the same observer uses one function for both."""
    from torchao.quantization.pt2e import (
        FakeQuantize, FusedMovingAvgObsFakeQuantize,
        MovingAveragePerChannelMinMaxObserver)

    def run(fq):
        g = torch.Generator().manual_seed(5)
        differing = 0
        for step in range(6):
            fq(torch.randn(8, 16, generator=g) * (0.5 + step))
            s, _ = fq.calculate_qparams()
            differing += not torch.equal(fq.scale, s.reshape(fq.scale.shape))
        return differing

    kw = dict(dtype=torch.int8, quant_min=-127, quant_max=127,
              qscheme=torch.per_channel_symmetric, ch_axis=0)
    fused = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver, eps=2 ** -12)(**kw)
    plain = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver, eps=2 ** -12)(**kw)
    assert run(fused) == 6       # XNNPACK's QAT weights: 255/254 wider
    assert run(plain) == 0


def test_call_range_observer_is_the_chooser():
    """CallRangeObserver's qparams are choose_qparams.tensor's on every call;
    MovingAverageMinMaxObserver(averaging_constant=1)'s are not always."""
    import torch.ao.quantization.fx._decomposed  # noqa: F401
    from torchao.quantization.pt2e import MovingAverageMinMaxObserver

    kw = dict(dtype=torch.int8, qscheme=torch.per_tensor_affine,
              quant_min=-128, quant_max=127, eps=eq.EPS_XNNPACK)
    exact = eq.CallRangeObserver(**kw)
    moving = MovingAverageMinMaxObserver(averaging_constant=1, **kw)
    torch.manual_seed(0)
    off = {"exact": 0, "moving": 0}
    for i in range(200):
        x = torch.randn(4, 64) * (1 + i % 7) + (i % 5) * 0.3
        s, z = torch.ops.quantized_decomposed.choose_qparams.tensor(
            x, -128, 127, eq.EPS_XNNPACK, torch.int8)
        for name, obs in (("exact", exact), ("moving", moving)):
            obs(x)
            os_, oz = obs.calculate_qparams()
            off[name] += not (torch.equal(os_.reshape(()), s.reshape(()))
                              and int(oz) == int(z))
    assert off["exact"] == 0 and off["moving"] > 0, off


def test_dynamic_qat_exports_its_eps():
    """convert_pt2e takes a dynamic chain's eps from the fake-quantiser
    module: XNNPACK's FakeQuantize has none, so its dynamic QAT exports
    FLT_EPSILON although it trained at 2^-12; DynamicFakeQuantize exports
    2^-12."""
    from torchao.quantization.pt2e.quantize_pt2e import (convert_pt2e,
                                                         prepare_qat_pt2e)

    def chooser_eps(quantizer):
        x = torch.randn(2, 32)
        gm = torch.export.export(MLP().train(), (x,)).module(check_guards=False)
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = prepare_qat_pt2e(gm, quantizer)
            prepared(x)
            out = convert_pt2e(prepared)
        return {float(n.args[3]) for n in out.graph.nodes
                if "choose_qparams" in str(n.target)}

    assert chooser_eps(stock_quantizer("xnnpack_dynamic", True)) == {eq.FLT_EPSILON}
    assert chooser_eps(Esp32Quantizer(preset="xnnpack_dynamic", is_qat=True)) == {
        eq.EPS_XNNPACK}


class SeparateQKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(16, 16) for _ in range(3))

    def forward(self, x):
        return self.q(x) + self.k(x) * self.v(x)


@pytest.mark.parametrize("preset, degrades", [
    (None, True), ("xnnpack_dynamic", False), ("xnnpack_static", False),
    ("arm_a8w8", False), ("cortex_m", False)])
def test_shared_activation(preset, degrades):
    """One activation feeding three Linears: the dynamic SYMMETRIC scheme (the
    default) degrades to weight-only (PortNodeMetaForQDQ fails on its shared
    chain); the affine dynamic and the static schemes keep the annotation and
    convert."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    x = torch.randn(2, 16)
    gm = torch.export.export(SeparateQKV().eval(), (x,)).module()
    q = Esp32Quantizer(preset=preset) if preset else Esp32Quantizer()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        prepared = prepare_pt2e(gm, q)
        prepared(x)
        out = convert_pt2e(prepared)
    assert q.act_annotated is (not degrades)
    assert ("WEIGHT-ONLY" in log.getvalue()) is degrades
    targets = [str(n.target) for n in out.graph.nodes]
    assert any("quantize_per_tensor" in t for t in targets) is (not degrades)


def test_shared_symmetric_chain_fails_upstream():
    """The control: forcing the default's annotation still fails in
    PortNodeMetaForQDQ (the degrade is load-bearing)."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    x = torch.randn(2, 16)
    gm = torch.export.export(SeparateQKV().eval(), (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(annotate_act=True))
    prepared(x)
    with pytest.raises(Exception, match="PortNodeMetaForQDQ"):
        convert_pt2e(prepared)


# --- the torchao helper ------------------------------------------------------------------

def test_torchao_defaults_as_read():
    """torchao 0.18's defaults the helper departs from: 8da is PerGroup(32),
    AFFINE; int4 weight-only is group 128, TINYGEMM."""
    from torchao.quantization import (Int4WeightOnlyConfig,
                                      Int8DynamicActivationIntxWeightConfig)
    from torchao.quantization.granularity import PerGroup

    da = Int8DynamicActivationIntxWeightConfig()
    assert da.weight_granularity == PerGroup(32)
    assert str(da.intx_choose_qparams_algorithm).endswith("AFFINE") or \
        da.intx_choose_qparams_algorithm == "affine"
    assert da.weight_dtype == torch.int8 and da.weight_scale_dtype is None
    w4 = Int4WeightOnlyConfig()
    assert w4.group_size == 128
    assert "TINYGEMM" in str(w4.int4_choose_qparams_algorithm).upper()


def test_torchao_config_fields():
    from torchao.quantization import (Int8DynamicActivationIntxWeightConfig,
                                      IntxWeightOnlyConfig, MappingType)
    from torchao.quantization.granularity import PerAxis, PerGroup

    c = eq.esp32_torchao_config(4, 32)
    assert isinstance(c, Int8DynamicActivationIntxWeightConfig)
    assert c.weight_dtype == torch.int4 and c.weight_granularity == PerGroup(32)
    assert c.weight_mapping_type == MappingType.SYMMETRIC
    assert c.act_mapping_type == MappingType.ASYMMETRIC
    assert c.weight_scale_dtype == torch.float32
    c = eq.esp32_torchao_config(8, None, "asymmetric", activations=False)
    assert isinstance(c, IntxWeightOnlyConfig)
    assert c.granularity == PerAxis(0) and c.scale_dtype == torch.float32
    assert c.mapping_type == MappingType.ASYMMETRIC


@pytest.mark.parametrize("args, kwargs, rule", [
    ((3, 32), {}, "torchao-bits"),
    ((4, 48), {}, "torchao-group"),
    ((8, 24), {}, "torchao-group"),
    ((4, 32, "affine"), {}, "weight-mapping"),
])
def test_torchao_refused_by_name(args, kwargs, rule):
    with pytest.raises(SchemeRefused) as err:
        eq.esp32_torchao_config(*args, **kwargs)
    assert err.value.rule == rule


def test_torchao_qat_config():
    from torchao.quantization.qat import QATConfig

    prep = eq.esp32_torchao_qat_config(4, 32)
    assert isinstance(prep, QATConfig) and prep.step == "prepare"
    assert prep.base_config is None and prep.weight_config.scale_precision == torch.float32
    conv = eq.esp32_torchao_qat_config(4, 32, step="convert")
    assert conv.base_config.weight_scale_dtype == torch.float32
    with pytest.raises(SchemeRefused) as err:
        eq.esp32_torchao_qat_config(4, 32, "asymmetric")
    assert err.value.rule == "torchao-qat-asymmetric"


def test_torchao_qat_weight_grid_is_the_exported_one():
    """The prepare step's weight fake-quantiser computes the scales the
    convert step exports, bit for bit, on a Linear's final weights."""
    from torchao.quantization import quantize_

    torch.manual_seed(4)
    lin = nn.Sequential(nn.Linear(64, 32))
    quantize_(lin, eq.esp32_torchao_qat_config(4, 32))
    fq = lin[0].weight_fake_quantizer
    lin(torch.randn(3, 64))
    trained = (fq.scale.clone(), fq.zero_point.clone())
    quantize_(lin, eq.esp32_torchao_qat_config(4, 32, step="convert"))
    w = lin[0].weight
    assert torch.equal(w.scale.reshape(trained[0].shape).float(), trained[0].float())
    assert torch.equal(w.zero_point.reshape(trained[1].shape).to(torch.int64),
                       trained[1].to(torch.int64))


# --- every accepted scheme lowers whole ----------------------------------------------

def _accepted_schemes():
    """Every combination of the public options Esp32Quantizer accepts (the
    weight range and eps at their defaults)."""
    out = []
    for bits in (2, 4, 8):
        for mapping in ("symmetric", "asymmetric"):
            for gran in ("per_channel", "per_tensor"):
                for act in ("dynamic_per_tensor_symmetric",
                            "dynamic_per_tensor_affine",
                            "static_per_tensor_affine",
                            "static_per_tensor_symmetric_int16", "none"):
                    for bias in ("fp32", "int32_derived"):
                        kw = dict(weight_bits=bits, weight_mapping=mapping,
                                  weight_granularity=gran, activation=act,
                                  bias=bias)
                        try:
                            Esp32Quantizer(**kw)
                        except SchemeRefused:
                            continue
                        out.append(pytest.param(kw, id="-".join(
                            [f"w{bits}", mapping[:4], gran[4:8], act, bias])))
    return out


class ConvMLP(nn.Module):
    """A padded conv, then two Linears with a ReLU: the shapes every scheme
    must delegate."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(4 * 4 * 8, 64), nn.Linear(64, 32)

    def forward(self, x):
        h = torch.relu(self.conv(x)).flatten(1)
        return self.fc2(torch.relu(self.fc1(h)))


@pytest.fixture(scope="module")
def host_runner():
    from executorch.backends.esp32s3.test import host

    path = host.host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path


@pytest.mark.parametrize("kwargs", _accepted_schemes())
def test_accepted_scheme_lowers_whole(kwargs, host_runner, tmp_path):
    """Whatever Esp32Quantizer accepts, the backend takes whole: after
    prepare/calibrate/convert and lowering, no matmul or convolution is left
    portable, and the host runner's output matches the converted model's."""
    import subprocess

    import numpy as np
    from executorch.exir import ExecutorchBackendConfig
    from torchao.quantization.pt2e.quantize_pt2e import (convert_pt2e,
                                                         prepare_pt2e)

    from executorch.backends.esp32s3._passes import fold_conv_bn_pass
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3._passes import strip_dynamic_act_qdq_pass
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    torch.manual_seed(7)
    x = torch.randn(1, 2, 4, 8)
    gm = torch.export.export(ConvMLP().eval(), (x,)).module(check_guards=False)
    with contextlib.redirect_stdout(io.StringIO()):
        prepared = prepare_pt2e(gm, Esp32Quantizer(**kwargs))
        g = torch.Generator().manual_seed(8)
        for _ in range(4):
            prepared(torch.randn(1, 2, 4, 8, generator=g))
        converted = convert_pt2e(prepared)
    ep = torch.export.export(converted, (x,), strict=False)
    with torch.no_grad():
        reference = ep.module()(x).reshape(-1).numpy()
    fold = fold_conv_bn_pass.fold_conv_bn

    def absorb_then_fold(edge, method_name="forward"):
        strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq(edge, method_name)
        return fold(edge, method_name)

    _passes.fold_conv_bn = absorb_then_fold
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lowered = _passes.lower_esp32(ep, [Esp32Partitioner()])
            _passes.apply_passes(lowered, quiet=True)
    finally:
        _passes.fold_conv_bn = fold
    left = [str(n.target) for n in
            lowered.exported_program().graph_module.graph.nodes
            if n.op == "call_function" and any(
                k in str(n.target) for k in ("linear", "addmm", "mm.default",
                                              "convolution"))]
    assert not left, left
    et = lowered.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    pte = tmp_path / "m.pte"
    pte.write_bytes(et.buffer)
    np.asarray(x, np.float32).tofile(tmp_path / "in.bin")
    done = subprocess.run([str(host_runner), "--pte", str(pte), "--input",
                           str(tmp_path / "in.bin"), "--output",
                           str(tmp_path / "out.bin")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    out = np.fromfile(tmp_path / "out.bin", dtype=np.float32)
    cos = float(np.dot(out, reference) / (np.linalg.norm(out)
                                           * np.linalg.norm(reference)))
    assert cos >= 0.9999, cos


# --- carry_observer_ranges (quantizer/lut16_range.py) -------------------------------------

RANGE_CASES = [("xnnpack_static", False, "silu"), ("xnnpack_static", True, "se_hardswish"),
               ("xnnpack_dynamic", False, "silu"), ("arm_a8w8", False, "se_hardswish"),
               ("arm_a16w8", False, "silu"), ("arm_a16w8", True, "silu"),
               ("arm_a16w8", False, "llama3_2"), ("cortex_m", False, "se_hardswish"),
               ("esp32-arm_a16w8", False, "silu")]


def _calibrated(stock, is_qat, model):
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e

    gm, inputs, calibration = census_graph(model, channels_last=stock == "cortex_m")
    if stock.startswith("esp32-"):
        quantizer = Esp32Quantizer(preset=stock[len("esp32-"):], is_qat=is_qat)
    else:
        quantizer = stock_quantizer(stock, is_qat)
    with contextlib.redirect_stdout(io.StringIO()):
        prepared = (prepare_qat_pt2e if is_qat else prepare_pt2e)(gm, quantizer)
        if is_qat:
            move_exported_model_to_eval(prepared)
        with torch.no_grad():
            for batch in calibration:
                prepared(*batch)
    return prepared, inputs


@pytest.mark.parametrize("stock, is_qat, model", RANGE_CASES,
                         ids=[f"{s}{'-qat' if q else ''}-{m}" for s, q, m in RANGE_CASES])
def test_carry_observer_ranges_converts_stock_recipes(stock, is_qat, model):
    """The helper on each stock recipe: convert_pt2e gives the same program
    plus one observed_range per static activation quantize, each holding
    its observer's min/max; eager outputs bit for bit; after the lowering's
    strip, the same exported graph."""
    from executorch.backends.esp32s3.quantizer import lut16_range as R
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

    plain_prepared, inputs = _calibrated(stock, is_qat, model)
    prepared, _ = _calibrated(stock, is_qat, model)
    expected = {}     # one entry per observer object (shared ones have several names)
    for n in prepared.graph.nodes:
        if n.op != "call_module" or not n.args:
            continue
        mod = prepared.get_submodule(n.target)
        if (type(mod) in R.carriers() and R._static_per_tensor(mod)
                and isinstance(n.args[0], torch.fx.Node) and n.args[0].op != "get_attr"):
            expected[id(mod)] = R.observer_range(mod)
    carried = eq.carry_observer_ranges(prepared)
    assert carried == len(expected)
    with contextlib.redirect_stdout(io.StringIO()):
        plain = convert_pt2e(plain_prepared)
        converted = convert_pt2e(prepared)
    ranges = [n for n in converted.graph.nodes if R._is_range(n)]
    statics = [n for n in plain.graph.nodes if fold_quant_lut_pass._is(n, fold_quant_lut_pass.Q)
               and isinstance(n.args[0], torch.fx.Node) and n.args[0].op != "get_attr"]
    print(f"{stock}{'-qat' if is_qat else ''} {model}: {carried} observer(s) carried, "
          f"{len(ranges)} observed_range op(s), {len(statics)} static activation quantize(s)")
    if stock == "xnnpack_dynamic":
        assert not ranges and not carried
    else:
        assert ranges and carried
    for r in ranges:
        assert fold_quant_lut_pass._is(r.args[0], fold_quant_lut_pass.Q)
        assert all(fold_quant_lut_pass._is(u, fold_quant_lut_pass.DQ) for u in r.users)
        assert tuple(r.args[1:]) in {v for v in expected.values() if v is not None}
    with torch.no_grad():
        a, b = plain(*inputs), converted(*inputs)
    for x, y in zip(torch.utils._pytree.tree_leaves(a), torch.utils._pytree.tree_leaves(b)):
        assert torch.equal(x, y)
    ep_plain = torch.export.export(plain, inputs, strict=False)
    ep = torch.export.export(converted, inputs, strict=False)
    assert R.strip_observed_ranges(ep) == len(ranges)
    assert str(ep.graph_module.graph) == str(ep_plain.graph_module.graph)


def test_histc_guard_bins_an_overflowing_range_and_nothing_else():
    """histc_guard (conftest.py installs it): torch.histc's float32 kernel
    writes out of bounds -- and drops -- the elements whose bin overflows past
    FLT_MAX / bins, the range a masked attention score gives HistogramObserver;
    the guard bins those calls in float64 and leaves every other call torch's
    own, bit for bit. The raw kernel is never called on an overflowing range
    here: it would corrupt this process's heap."""
    from executorch.backends.esp32s3.test import histc_guard

    fmin = torch.finfo(torch.float32).min
    x = torch.tensor([fmin, 0.0, 0.5, 1.0])
    counts = histc_guard.histc(x, 2048, fmin, 1.0)
    assert counts.sum() == 4 and counts[0] == 1 and counts[-1] == 3
    assert histc_guard._overflows(fmin, 1.0, 2048)
    assert not histc_guard._overflows(-1e30, 1.0, 2048)
    g = torch.Generator().manual_seed(0)
    for lo, hi in ((-3.0, 4.0), (-1e30, 1.0), (0.0, 1e-20), (-1.0, -1.0)):
        y = torch.rand(10_000, generator=g) * (hi - lo) + lo
        assert torch.equal(histc_guard.histc(y, 2048, lo, hi),
                           histc_guard._HISTC(y, 2048, lo, hi))
    assert torch.histc is histc_guard.histc
