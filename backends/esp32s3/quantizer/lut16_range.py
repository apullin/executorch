# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The observer's range carried to lowering.

A fitted int16 table (fold_lut16_pass.py) built from the graph alone must hold
every input code of its quantiser: 65,535 of them under Arm a16w8, whose eps
floor fixes s1 = 2^-12 while the activations sit in a few hundred or
thousand codes. The input observer saw where they sit -- its [min_val,
max_val] -- and convert_pt2e throws that away. Those two floats buy tables
2-30x smaller on the quantizer census's a16w8 sites, 16 B - 1 KB,
direct and exact for the rsqrt-like ones. Node meta does not survive
torch.export, so the range travels as an op:

    X -> q1 -> espressif::observed_range(codes, min, max) -> dq1 -> ...

an identity on q1's codes (between q1 and dq1, where torchao's metadata
pass looks for neither, so convert warns about nothing), with the two floats
as arguments. Three pieces:

  carry_observer_ranges(prepared)
                  (esp32_quantizer's helper) after calibration, before
                  convert_pt2e: every stock static per-tensor activation
                  observer -- HistogramObserver, MinMaxObserver,
                  MovingAverageMinMaxObserver, and the FakeQuantize
                  modules of a QAT model -- becomes its RangeCarrying
                  subclass (carriers()), in place, state and all. Its
                  convert() (torchao/quantization/pt2e/convert.py's hook:
                  "if hasattr(activation_post_process, 'convert')") runs
                  torchao's own replacement, then puts observed_range
                  between the q and dq it emitted. Weight observers and
                  constants are left alone.
  strip_observed_ranges(ep)
                  the lowering's first step (_passes.lower_esp32):
                  every observed_range goes, its (min, max) kept on the
                  quantize it read, as node.meta["observed_range"]. A
                  program without one is not touched.
  window(...)     fold_lut16_pass.fold_fitted_luts' codes for a site whose
                  q1 carries a range: the observed codes
                  [floor(min / s1) + z1, ceil(max / s1) + z1] widened by
                  12.5% of their span (at least 2 codes) each way. The
                  table holds those codes: the smallest candidate whose
                  flips against the eager graph there stay within their
                  float32 floor (lut_fit.fit_smallest; the flips bar over
                  the observer's range), or the graph-only table where that is
                  no larger. A code outside clamps to the window's end. A
                  window's NSAU table holds segments from the window's first
                  one (lut_fit's OFFSET). lut16's TAIL extrapolates the
                  edge segment only where that is exact on every code past
                  the table -- where f is a line toward the rails -- so it
                  shortens graph-only tables (lut_fit.fit_tail), which then
                  compete with the window's; codes past a window still
                  clamp.

No histogram payload: it does not beat the range. With none of this used,
nothing changes: no op is emitted, the strip finds nothing.
"""

from __future__ import annotations

import copy
import math

import torch
from torch.library import Library

lib = Library("espressif", "FRAGMENT")
lib.define("observed_range(Tensor x, float min, float max) -> Tensor")
impl = Library("espressif", "IMPL")


def _observed_range(x, min, max):
    return x.clone()


impl.impl("observed_range", _observed_range, "CompositeExplicitAutograd")


@torch.library.register_fake("espressif::observed_range")
def _observed_range_fake(x, min, max):
    return torch.empty_like(x)


OBSERVED_RANGE = torch.ops.espressif.observed_range.default
SCHEMA = ("espressif::observed_range", "default")
MARGIN = 0.125
META = "observed_range"


# --- carrying: the observers' convert() --------------------------------------------

def _is_quantize(node):
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    return fold_quant_lut_pass._is(node, fold_quant_lut_pass.Q)


def observer_range(module):
    """(min_val, max_val) of an observer or fake-quantiser as floats, or
    None when it holds no finite, ordered pair."""
    obs = getattr(module, "activation_post_process", module)
    lo, hi = getattr(obs, "min_val", None), getattr(obs, "max_val", None)
    if not (isinstance(lo, torch.Tensor) and isinstance(hi, torch.Tensor)
            and lo.numel() == 1 and hi.numel() == 1):
        return None
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        return None
    return lo, hi


def _convert(self, model, node):
    """convert_pt2e's hook (convert.py: an activation_post_process with a
    convert() method converts itself): torchao's replacement exactly as it
    runs for the stock class -- same qparams, same nodes, same metadata --
    then observed_range(codes, min_val, max_val) between each quantize it
    emitted and that quantize's dequantizes."""
    from torchao.quantization.pt2e.convert import (
        _replace_observer_with_quantize_dequantize_node_decomposed as replace)

    attrs = model.meta.get("_observed_graph_module_attrs")
    scope = getattr(attrs, "node_name_to_scope", None) or {}
    qconfig = getattr(attrs, "node_name_to_qconfig", None) or {}
    X = node.args[0]
    before = set(X.users) if isinstance(X, torch.fx.Node) else set()
    stock = copy.copy(self)          # shares every buffer and submodule
    stock.__class__ = type(self).__base__
    replace(model, node, {node.target: stock}, scope, qconfig, None)
    rng = observer_range(self)
    if rng is None or not isinstance(X, torch.fx.Node) or X.op == "get_attr":
        return
    graph = model.graph
    for q in [u for u in X.users if u not in before and _is_quantize(u)]:
        readers = list(q.users)
        with graph.inserting_after(q):
            carrier = graph.call_function(OBSERVED_RANGE, (q, *rng))
        for user in readers:
            user.replace_input_with(q, carrier)


_CARRIERS = {}


def carriers():
    """{stock class: its RangeCarrying subclass}, built on first use (it
    imports torchao), each subclass also a module attribute so that a
    carried observer pickles by name."""
    if not _CARRIERS:
        from torchao.quantization.pt2e import (
            FakeQuantize,
            FusedMovingAvgObsFakeQuantize,
            HistogramObserver,
            MovingAverageMinMaxObserver,
        )
        from torchao.quantization.pt2e.observer import MinMaxObserver

        for cls in (HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver,
                    FakeQuantize, FusedMovingAvgObsFakeQuantize):
            # single inheritance, nothing but a method: CPython lets __class__
            # change in place only between layout-identical classes (a mixin
            # base breaks that for FusedMovingAvgObsFakeQuantize)
            carrier = type(f"RangeCarrying{cls.__name__}", (cls,),
                           {"__module__": __name__, "convert": _convert, "__doc__":
                            f"{cls.__name__} whose convert() also emits "
                            f"espressif::observed_range (lut16_range.py)."})
            _CARRIERS[cls] = carrier
            globals()[carrier.__name__] = carrier
    return _CARRIERS


def __getattr__(name):
    if name.startswith("RangeCarrying"):
        for carrier in carriers().values():
            if carrier.__name__ == name:
                return carrier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _static_per_tensor(module):
    if getattr(module, "is_dynamic", False):
        return False
    if module.dtype not in (torch.int8, torch.uint8, torch.int16, torch.uint16):
        return False
    return module.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)


def carry_observer_ranges(prepared):
    """Make every stock static per-tensor activation observer of a prepared
    (or QAT-prepared) PT2E model emit its range at convert_pt2e: its class
    becomes the RangeCarrying subclass in place, so its statistics, its
    qparams and every node calling it stay exactly as they were. Observers
    that only watch constants (weights) are left as they are. Call it after
    calibration (or training), right before convert_pt2e. Returns the
    number of observers changed."""
    readers = {}
    for node in prepared.graph.nodes:
        if node.op == "call_module":
            readers.setdefault(node.target, []).append(node)
    table = carriers()
    changed = 0
    for name, module in prepared.named_modules(remove_duplicate=False):
        cls = type(module)
        if cls not in table or not _static_per_tensor(module):
            continue
        sites = readers.get(name, [])
        if not any(isinstance(n.args[0], torch.fx.Node) and n.args[0].op != "get_attr"
                   for n in sites if n.args):
            continue
        module.__class__ = table[cls]
        changed += 1
    return changed


# --- lowering: the strip and the window ----------------------------------------------

def _is_range(node):
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    return node.op == "call_function" and fold_quant_lut_pass._schema(node.target) == SCHEMA


def strip_observed_ranges(exported_program):
    """Remove every espressif::observed_range from an ATen-dialect
    ExportedProgram (or a dict of them), in place, keeping its (min, max)
    on the node it read as meta["observed_range"] (the union when several
    read one node). Returns how many went; a program with none is not
    touched at all."""
    if isinstance(exported_program, dict):
        return sum(strip_observed_ranges(ep) for ep in exported_program.values())
    graph = exported_program.graph_module.graph
    found = [n for n in graph.nodes if _is_range(n)]
    for node in found:
        source = node.args[0]
        lo, hi = float(node.args[1]), float(node.args[2])
        if isinstance(source, torch.fx.Node):
            old = source.meta.get(META)
            source.meta[META] = (lo, hi) if old is None else (min(old[0], lo),
                                                             max(old[1], hi))
        node.replace_all_uses_with(source)
        graph.erase_node(node)
    if found:
        graph.lint()
        exported_program.graph_module.recompile()
    return len(found)


def window(rng, scale, zero_point, quant_min, quant_max, margin=MARGIN):
    """[lo, hi], the input codes a table for this quantiser holds given the
    observed range rng = (min, max): the codes the range covers,
    floor(min / s) + z to ceil(max / s) + z, widened each way by `margin` of
    that span, at least 2 codes, inside [quant_min, quant_max]. None without
    a range."""
    if rng is None:
        return None
    lo, hi = rng
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        return None
    olo = math.floor(lo / scale) + zero_point
    ohi = math.ceil(hi / scale) + zero_point
    pad = max(2, int(math.floor((ohi - olo) * margin)))
    return (min(max(olo - pad, quant_min), quant_max),
            min(max(ohi + pad, quant_min), quant_max))
