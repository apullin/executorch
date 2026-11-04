# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A PT2E Quantizer expressing the formats this backend runs exactly.

Why this exists at all: the delegate can quantise weights itself (data-free
absmax in `preprocess`) and can consume torchao's
`Int8DynActInt4WeightQuantizer` output verbatim. Neither is the flow ExecuTorch
users are told to use. The standard one is PT2E -- annotate with a Quantizer,
`prepare_pt2e` (or `prepare_qat_pt2e`), calibrate or train, `convert_pt2e` --
and a backend that cannot participate in it is a backend nobody can reach with
their existing pipeline. That, not kernel coverage, is the gap this closes.

WHAT IT EXPRESSES, precisely, by default (`Esp32Quantizer()`):

  weights      int4 symmetric, per output channel, codes in [-7, 7]
  activations  int8 symmetric, per tensor, DYNAMIC (scale chosen per call),
               [-127, 127], scale floored at FLT_EPSILON -- the delegate's
               FLAG_ACT_PER_TENSOR quantiser, exactly

which is the delegate's per-channel i4 rung exactly. Where every Linear has
K <= 128, per-channel and group-128 are the same numbers. That equivalence does
NOT extend to wider projections. Their PT2E artifacts remain exact because
pass-through preserves the per-channel codes and scales, but they are a
different accuracy/storage choice from this backend's optional group-128 AOT
policy. Per-group weights and per-token activations cannot be expressed in
PT2E at all: those schemes come through torchao, configured by
`esp32_torchao_config` below.

Embedding tables never pass through this Linear quantizer; they arrive as
pass-through weights or through their own graph op. Their group-128 format is
therefore outside this quantizer by construction rather than silently covered.

THE OTHER SCHEMES. The delegate runs a model's own
quantisers exactly: weight zero points, dynamic
per-tensor affine activations, static int8 graphs with their int32 biases.
Every keyword below selects one of those, and nothing else:

  weight_bits         2, 4 or 8 (2: the ternary carrier, below)
  weight_granularity  "per_channel" or "per_tensor", or a dict giving
                      {"linear": ..., "conv": ...} separately (Cortex-M
                      quantises Linear per tensor and convolutions per channel)
  weight_mapping      "symmetric" (per_channel/per_tensor_symmetric) or
                      "asymmetric" (*_affine: weight zero points)
  weight_qmin/qmax    the weight code range; default [-q, q] symmetric
                      (q = 2^(bits-1) - 1), the full range asymmetric
  activation          "dynamic_per_tensor_symmetric" (the default),
                      "dynamic_per_tensor_affine", "static_per_tensor_affine",
                      "static_per_tensor_symmetric_int16" or "none"
                      (weight-only)
  act_qmin/act_qmax   the activation code range; [0, 255] (affine only)
                      means uint8 activations
  eps                 the scale floor of every observer (weights too, as the
                      stock configs do); weight_eps overrides the weights'
  bias                "fp32" (the delegate adds it after the scale) or
                      "int32_derived" (int32, scale = activation scale x
                      weight scale: Arm's and Cortex-M's)
  is_qat              fake-quantisers for prepare_qat_pt2e (QAT, below)

and a preset sets them all to a stock recipe's own values -- `preset=` one of
PRESETS: "xnnpack_static", "xnnpack_dynamic", "arm_a8w8", "arm_a8w4",
"arm_a16w8", "cortex_m". Keywords given alongside a preset override its
values. A static configuration also annotates each op's output (after the
activation the delegate fuses, by the preset's rule), which is the pair the
delegate absorbs as its output quantisation (static_qdq.py, Stage A).

The point is not to replace the stock quantizers: the backend accepts their
graphs as they are (quant_census). A preset is the same SCHEME as its stock
config -- every spec, eps, observer and QAT fake-quantiser identical, which
test_espressif_quantizer.py checks against the stock quantizer on the census
models -- on the ops this backend computes in integers. What a stock
quantizer annotates beyond those (adds, muls, concatenations, pools; Arm and
Cortex-M nearly every op) this quantizer leaves in float: the delegate runs
every op outside a matmul or convolution in float between its delegates
(Stage A), so those pairs would add roundings and portable kernel calls on the
device and save nothing. test_espressif_quantizer.py names each such difference.

REFUSED, by name, at construction (SchemeRefused, with its rule): any scheme
the backend cannot run exactly. Static int16 activations (Arm a16w8) take
symmetric int8 weights only (the runtime's int16 loop); an int32 bias
with dynamic activations (upstream convert_pt2e cannot derive a scale from a
per-call one); a symmetric unsigned activation range (the backend refuses
both choosers' midpoint zero points); an eps that is not positive.

QAT. `is_qat=True` puts real fake-quantisers in the specs, each computing its
qparams with the very function convert_pt2e exports them with, so a trained
model exports the grid it trained on, bit for bit: dynamic activations
FakeQuantize over a fresh range per call, as the dynamic chooser computes at
run time -- XNNPACK's and Arm's swap, MovingAverageMinMaxObserver(
averaging_constant=1), made exact twice: CallRangeObserver takes the call's
own aminmax (the moving average's min + 1 * (cur - min) is not cur in
float), and DynamicFakeQuantize shows convert its eps (convert_pt2e reads a
dynamic chain's eps from the fake-quantiser itself, which plain FakeQuantize
lacks, so the stock configs export FLT_EPSILON whatever eps they trained
with); static activations FakeQuantize over MovingAverageMinMaxObserver,
weights FakeQuantize over each preset's stock observer (Arm's per-channel
min/max; XNNPACK's moving average). XNNPACK and Arm put
FusedMovingAvgObsFakeQuantize on static activations (XNNPACK on weights too),
and that one does NOT export what it trains: its C++ kernel
(ChooseQuantizationParams) computes the qparams the training forward uses --
a symmetric grid over (qmax - qmin)/2 + 1 negative codes, a 6.1e-5 scale
floor instead of eps, a nudged zero point -- while convert_pt2e exports the
observer's calculate_qparams. Measured: per-channel weight scales
255/254 wider in training than exported on every step, activation scales 1
ulp apart on some. The observer statistics are the same either way; only the
fused kernel is swapped out, and test_espressif_quantizer.py names it.

WITHOUT is_qat the specs hold plain observers, and prepare_qat_pt2e then
inserts observers that pass values through untouched: such a "QAT" trains in
full precision. Swapping prepare_pt2e for prepare_qat_pt2e is not enough:
prepare_qat_pt2e(Esp32Quantizer()) inserts no fake-quantiser at all.

There is also a deliberately limited ``weight_bits=2`` semantic rung. PT2E
uses an int8 container with range [-1, 1], so training/export sees exact
ternary codes. Until the backend gains a signed-t2 wire opcode those codes
travel in the existing i4 container and execute through the i4-unpack -> int8
PIE path. This proves the framework path but saves ZERO bytes; descriptor/tests
call it a carrier rather than pretending packed-ternary performance already
exists.

SHARED ACTIVATIONS, and why the default's activation annotation is
conditional. An architecture with SEPARATE q/k/v projections feeds one
activation to three annotated matmuls. PT2E then inserts ONE choose_qparams/
quantize pair for that activation and gives it three users -- and torchao's
`PortNodeMetaForQDQ` pass raises (`Expecting single user for
quantize_per_tensor_tensor`) on exactly that shape when the chooser is the
SYMMETRIC one, choose_qparams_symmetric.tensor: measured 5 of 5 failures on
three and on two consumers, while the affine choose_qparams.tensor chain of
the same graph converts 5 of 5 (XNNPACK's dynamic config, the
xnnpack_dynamic preset). It is not reachable from this repo's own models,
which use a FUSED qkv Linear (one consumer), and it is not fixable from here:
the assertion lives upstream and a fresh spec object per node does not avoid
it, because PT2E keys the deduplication on the activation NODE.

So when a shared activation is present under the dynamic SYMMETRIC scheme
(the default), the activation annotation is dropped and the model is
annotated weight-only, and the notice says so: the delegate then quantises
those activations per row, its own scheme, instead of per tensor (an
annotated chain's scheme is what `strip_dynamic_act_qdq` records and the
device honours), and QAT trains quantised weights over full-precision
activations. Models with no shared activation keep the annotation (the
`mlp_pt2e` parity row is one). The
affine dynamic and the static configurations keep the annotation: a static
pair is found per consumer (DuplicateDQPass gives each its own dequantize),
and the backend absorbs a shared dynamic chain once for all its consumers.

    from executorch.backends.esp32s3 import Esp32Quantizer
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer())
    prepared(calibration_batch)
    converted = convert_pt2e(prepared)
    # then to_edge_transform_and_lower(..., partitioner=[Esp32Partitioner()])

FITTED TABLES. An int16 activation between two static quantisers (Arm
a16w8's) becomes an espressif::lut16 table at lowering (Esp32Partitioner's
luts16, on by default; fold_lut16_pass.py), fitted from the graph over all
65,535 codes. Two opt-in helpers here use what calibration knows:

  carry_observer_ranges(prepared)   right before convert_pt2e: each static
      activation observer's [min_val, max_val] travels to the lowering as an
      identity op (lut16_range.py), and the table is fitted over the codes it
      covers (arm-a16w8 llama3_2: 27.8 -> 16.4 KB over 16 tables; its rsqrt
      tables 1.8 KB -> 16-26 B)
  fit_tables(prepared, spec)        after calibration: each site becomes a
      Lut16 module (lut16_train.py) whose forward is the device's table, bit
      for bit, with a training backward; train, convert_pt2e, lower: the
      lowering takes the trained tables as they are
"""

import dataclasses
from typing import Optional

import torch
from torchao.quantization.pt2e import (
    FakeQuantize,
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torchao.quantization.pt2e.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer.quantizer import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)

# The i4 rung's code range. -7..7 rather than -8..7: symmetric, so the negative
# end is clipped to match, which is what quantize_absmax_rows does AOT and what
# the runtime's qmax of 7 assumes.
I4_QMIN, I4_QMAX = -7, 7
T2_QMIN, T2_QMAX = -1, 1
I8_QMIN, I8_QMAX = -127, 127

# Scale floors of the stock configs (verified in the installed sources):
# XNNPACK's get_symmetric_quantization_config 2^-12 (activations and
# weights); Arm's get_symmetric_quantization_config 2^-16 (both), its a16w8
# activations 2^-12 with the base config's 2^-16 weights; Cortex-M and
# torch's observers FLT_EPSILON (their default).
EPS_XNNPACK = 2.0 ** -12
EPS_ARM = 2.0 ** -16
FLT_EPSILON = float(torch.finfo(torch.float32).eps)

# Targets to annotate: `linear`, which is what a pre-dispatch export leaves,
# and nothing that decomposed it. ch_axis=0 means "per OUTPUT channel", which
# holds for nn.Linear's [N, K] weight; addmm/mm take theirs ALREADY TRANSPOSED,
# [K, N], whatever the shapes, so axis 0 there is the contraction axis and the
# backend would refuse the codes for arriving without their transpose. A shape
# test (dim 0 != N) cannot stand in for that: it lets a square K == N weight
# through. Export preserves aten.linear, so an nn.Linear model never meets an
# mm here; an explicit torch.mm/addmm stays fp32 and the backend quantises it.
_LINEAR_TARGETS = (torch.ops.aten.linear.default,)
# Convolutions, per OUTPUT channel on the [C_out, C_in/g, ...] weight -- which
# is the same ch_axis=0 spec as a Linear's [N, K], and reshapes to the same
# [N, K] the delegate packs. Transposed convolutions are deliberately absent:
# their weight is [C_in, C_out/g, ...], so per-output-channel is axis 1 and the
# backend has no pass-through layout for it. An unannotated ConvTranspose stays
# fp32 through PT2E and is quantised by the backend itself at export, like
# any op this quantizer leaves unannotated.
_CONV_TARGETS = (
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.convolution.default,
)
# XNNPACK's CONV_TARGETS (no conv3d); Arm's _conv_ops (with the padding="same"
# spellings); Cortex-M's CONV_OP_PATTERNS (conv2d only).
_XNNPACK_CONV = (torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default,
                 torch.ops.aten.convolution.default)
_ARM_CONV = (torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default,
             torch.ops.aten.conv2d.padding, torch.ops.aten.conv3d.default,
             torch.ops.aten.conv3d.padding)
_CORTEX_M_CONV = (torch.ops.aten.conv2d.default,)

_RELU = (torch.ops.aten.relu.default, torch.ops.aten.relu_.default)
_HARDTANH = (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default)
_CLAMP = (torch.ops.aten.clamp.default, torch.ops.aten.clamp_.default)
_HARDSIGMOID = (torch.ops.aten.hardsigmoid.default,
                torch.ops.aten.hardsigmoid_.default)
_BATCH_NORM = (torch.ops.aten.batch_norm.default,)

ACTIVATIONS = ("dynamic_per_tensor_symmetric", "dynamic_per_tensor_affine",
               "static_per_tensor_affine", "static_per_tensor_symmetric_int16",
               "none")
GRANULARITIES = ("per_channel", "per_tensor")
MAPPINGS = ("symmetric", "asymmetric")
BIASES = ("fp32", "int32_derived")


class CallRangeObserver(MovingAverageMinMaxObserver):
    """MovingAverageMinMaxObserver(averaging_constant=1) -- XNNPACK's and
    Arm's dynamic QAT observer -- done exactly: the range is the call's own
    torch.aminmax, which is what the dynamic chooser computes at run time.
    The moving average's min + 1 * (cur - min) is not cur in float (26 of
    200 calls off, measured), so the stock observer trains on a per-call
    grid an ulp away from the deployed one on those calls."""

    def __init__(self, **kwargs):
        kwargs["averaging_constant"] = 1
        super().__init__(**kwargs)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach().to(self.min_val.dtype)
        min_val, max_val = torch.aminmax(x)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class DynamicFakeQuantize(FakeQuantize):
    """FakeQuantize for a DYNAMIC activation spec that exports the eps it
    trains with. convert_pt2e builds a dynamic chain's chooser from
    getattr(fake_quantiser, "eps", FLT_EPSILON) -- the module's own eps, which
    FakeQuantize does not have (its observer does) -- so XNNPACK's dynamic
    QAT config trains at 2^-12 and exports FLT_EPSILON. This class shows
    convert its observer's eps; the arithmetic is FakeQuantize's."""

    @property
    def eps(self):
        # A Python float: a tensor eps lands in the converted graph as a
        # literal its generated code cannot spell (NameError 'float32').
        # The observer's float32 buffer converts exactly.
        return float(self.activation_post_process.eps)


class SchemeRefused(ValueError):
    """A configuration the backend cannot run exactly, named by its rule."""

    def __init__(self, rule, message):
        self.rule = rule
        super().__init__(f"Esp32Quantizer [{rule}]: {message}")


@dataclasses.dataclass(frozen=True)
class Scheme:
    """Every choice a configuration makes, resolved. The public keywords of
    Esp32Quantizer, then the stock flavour a preset pins (which observer, which
    fake-quantiser, which ops and fusions its stock quantizer annotates)."""

    weight_bits: int = 4
    weight_granularity: tuple = (("conv", "per_channel"),
                                 ("linear", "per_channel"))
    weight_mapping: str = "symmetric"
    weight_qmin: Optional[int] = None
    weight_qmax: Optional[int] = None
    activation: str = "dynamic_per_tensor_symmetric"
    act_qmin: Optional[int] = None
    act_qmax: Optional[int] = None
    eps: Optional[float] = None
    weight_eps: Optional[float] = None
    bias: str = "fp32"
    is_qat: bool = False
    preset: Optional[str] = None
    # -- flavour (set by presets) --------------------------------------------
    qat_weights: str = "fake_quantize"     # Arm's; "moving_average" XNNPACK's
    conv_targets: tuple = _CONV_TARGETS
    fusion: str = "esp32"                  # "xnnpack", "arm", "cortex_m"
    dynamic_conv: str = "all"              # "xnnpack": 2-D, not depthwise, no ReLU
    # Cortex-M's spelling of the same observers: quant_min/quant_max None for
    # the dtype's full range, ch_axis None on a per-tensor weight, and the
    # int32 bias of arm_quantizer_utils' helpers rather than Arm's config.
    cortex_m_spelling: bool = False

    def granularity(self, kind):
        return dict(self.weight_granularity)[kind]

    @property
    def dynamic(self):
        return self.activation.startswith("dynamic")

    @property
    def static(self):
        return self.activation.startswith("static")


# The stock configs, as the census runs them (quant_census/recipes.py). Each
# value is verified against the installed sources; test_espressif_quantizer.py
# compares the resulting specs with the stock quantizer's on the census
# models.
PRESETS = {
    # XNNPACKQuantizer().set_global(get_symmetric_quantization_config(
    #     is_per_channel=True))
    "xnnpack_static": dict(
        weight_bits=8, activation="static_per_tensor_affine", eps=EPS_XNNPACK,
        qat_weights="moving_average", conv_targets=_XNNPACK_CONV, fusion="xnnpack"),
    # ... get_symmetric_quantization_config(is_per_channel=True,
    #     is_dynamic=True)
    "xnnpack_dynamic": dict(
        weight_bits=8, activation="dynamic_per_tensor_affine", eps=EPS_XNNPACK,
        qat_weights="moving_average", conv_targets=_XNNPACK_CONV, fusion="xnnpack",
        dynamic_conv="xnnpack"),
    # EthosUQuantizer(...).set_global(get_symmetric_quantization_config())
    "arm_a8w8": dict(
        weight_bits=8, activation="static_per_tensor_affine", eps=EPS_ARM,
        bias="int32_derived", conv_targets=_ARM_CONV, fusion="arm"),
    # get_symmetric_a8w4_quantization_config(): weights [-7, 7]. Arm's helper
    # defaults to is_qat=True; the preset keeps is_qat's own default (False).
    "arm_a8w4": dict(
        weight_bits=4, activation="static_per_tensor_affine", eps=EPS_ARM,
        bias="int32_derived", conv_targets=_ARM_CONV, fusion="arm"),
    # get_symmetric_a16w8_quantization_config(): int16 [-32767, 32767]
    # symmetric activations at 2^-12, the base config's int8 weights at 2^-16.
    "arm_a16w8": dict(
        weight_bits=8, activation="static_per_tensor_symmetric_int16",
        eps=2.0 ** -12, weight_eps=EPS_ARM, bias="int32_derived",
        conv_targets=_ARM_CONV, fusion="arm"),
    # CortexMQuantizer(): convolutions per channel, everything else (Linear)
    # per tensor, both symmetric over int8's full range [-128, 127]; int32
    # biases; observers at their default eps (FLT_EPSILON).
    "cortex_m": dict(
        weight_bits=8, weight_granularity={"linear": "per_tensor",
                                           "conv": "per_channel"},
        weight_qmin=-128, weight_qmax=127,
        activation="static_per_tensor_affine", bias="int32_derived",
        conv_targets=_CORTEX_M_CONV,
        fusion="cortex_m", cortex_m_spelling=True),
}


def resolve_scheme(preset=None, **options):
    """The Scheme for a preset (or the backend's default) with `options`
    overriding it; None means "not given". No refusal here: the specs of a
    refused scheme can still be built and compared (Esp32Quantizer refuses)."""
    if preset is not None and preset not in PRESETS:
        raise SchemeRefused("preset", f"unknown preset {preset!r}; one of "
                            f"{', '.join(PRESETS)}")
    values = dict(PRESETS.get(preset, {}))
    values.update({k: v for k, v in options.items() if v is not None})
    gran = values.get("weight_granularity", "per_channel")
    if isinstance(gran, str):
        gran = {"linear": gran, "conv": gran}
    if not isinstance(gran, dict) or set(gran) != {"linear", "conv"}:
        raise SchemeRefused(
            "weight-granularity", f"{gran!r}: 'per_channel', 'per_tensor', or a "
            f"dict with exactly the keys 'linear' and 'conv'")
    values["weight_granularity"] = tuple(sorted(gran.items()))
    return Scheme(preset=preset, **values)


def check_scheme(scheme):
    """Raise SchemeRefused unless the backend runs `scheme` exactly."""
    if scheme.weight_bits not in (2, 4, 8):
        raise SchemeRefused(
            "weight-bits", f"{scheme.weight_bits}-bit weights: the backend "
            f"stores 4- and 8-bit codes (2 rides the i4 container as the "
            f"ternary carrier)")
    for kind, g in scheme.weight_granularity:
        if g not in GRANULARITIES:
            raise SchemeRefused("weight-granularity",
                                f"{kind}: {g!r}, not one of {GRANULARITIES}")
    if scheme.weight_mapping not in MAPPINGS:
        raise SchemeRefused("weight-mapping", f"{scheme.weight_mapping!r}, not "
                            f"one of {MAPPINGS}")
    if scheme.activation not in ACTIVATIONS:
        raise SchemeRefused("activation", f"{scheme.activation!r}, not one of "
                            f"{ACTIVATIONS}")
    if scheme.bias not in BIASES:
        raise SchemeRefused("bias", f"{scheme.bias!r}, not one of {BIASES}")
    if (scheme.activation == "static_per_tensor_symmetric_int16"
            and (scheme.weight_bits != 8 or scheme.weight_mapping != "symmetric")):
        raise SchemeRefused(
            "int16-weights", f"{scheme.weight_bits}-bit {scheme.weight_mapping} "
            f"weights under int16 activations: the int16 loop dots int16 "
            f"codes against symmetric int8 weights only")
    wmin, wmax = weight_range(scheme)
    lo, hi = -(1 << (scheme.weight_bits - 1)), (1 << (scheme.weight_bits - 1)) - 1
    if not lo <= wmin < wmax <= hi:
        raise SchemeRefused(
            "weight-range", f"[{wmin}, {wmax}] is not inside the "
            f"{scheme.weight_bits}-bit range [{lo}, {hi}]")
    if scheme.weight_mapping == "symmetric" and not wmin < 0 < wmax:
        raise SchemeRefused("weight-range", f"symmetric weights need a signed "
                            f"range, got [{wmin}, {wmax}]")
    if scheme.weight_bits == 2 and scheme.weight_mapping != "symmetric":
        raise SchemeRefused(
            "weight-bits", "2-bit weights are the symmetric ternary carrier "
            "(codes {-1, 0, 1} in the i4 container); asymmetric 2-bit codes "
            "have no backend path of their own")
    if scheme.activation != "none":
        qmin, qmax, dtype = act_range(scheme)
        if dtype is None:
            raise SchemeRefused(
                "act-range", f"[{qmin}, {qmax}] is neither int8 nor uint8 "
                f"(int16 only as static_per_tensor_symmetric_int16)")
        symmetric = scheme.activation.endswith(("symmetric", "symmetric_int16"))
        if symmetric and not qmin < 0 < qmax:
            raise SchemeRefused(
                "act-unsigned-symmetric", f"[{qmin}, {qmax}]: a symmetric "
                f"chooser over an unsigned range puts the zero point at the "
                f"midpoint, which the backend refuses")
    for name, eps in (("eps", scheme.eps), ("weight_eps", scheme.weight_eps)):
        if eps is not None and not eps > 0:
            raise SchemeRefused("eps", f"{name}={eps!r}: a scale floor must be "
                                f"positive")
    if scheme.bias == "int32_derived" and not scheme.static:
        # Since 1.6 Arm's QuantizationConfig drops its int32 bias spec under a
        # dynamic activation (get_bias_qspec, #22879): an Arm preset made
        # dynamic, with bias="fp32", is the scheme of Arm's own dynamic config.
        raise SchemeRefused(
            "bias-int32-dynamic", "an int32 bias takes the activation scale "
            "times the weight scale, and a dynamic (or absent) activation has "
            "no scale at export (convert_pt2e raises on it); pass bias='fp32' "
            "to keep the bias in float, as Arm's dynamic configs do")


def weight_range(scheme):
    if scheme.weight_qmin is not None or scheme.weight_qmax is not None:
        if scheme.weight_qmin is None or scheme.weight_qmax is None:
            raise SchemeRefused("weight-range",
                                "give weight_qmin and weight_qmax together")
        return scheme.weight_qmin, scheme.weight_qmax
    q = (1 << (scheme.weight_bits - 1)) - 1
    if scheme.weight_mapping == "symmetric":
        return -q, q
    return -q - 1, q


def act_range(scheme):
    """(qmin, qmax, dtype) of the activation codes; dtype None if neither
    int8 nor uint8 (int16 for the int16 scheme)."""
    if scheme.activation == "static_per_tensor_symmetric_int16":
        qmin = -32767 if scheme.act_qmin is None else scheme.act_qmin
        qmax = 32767 if scheme.act_qmax is None else scheme.act_qmax
        ok = -32768 <= qmin < qmax <= 32767
        return qmin, qmax, torch.int16 if ok else None
    if (scheme.act_qmin is None) != (scheme.act_qmax is None):
        raise SchemeRefused("act-range", "give act_qmin and act_qmax together")
    if scheme.act_qmin is None:
        if scheme.activation == "dynamic_per_tensor_symmetric":
            return I8_QMIN, I8_QMAX, torch.int8
        return -128, 127, torch.int8
    qmin, qmax = scheme.act_qmin, scheme.act_qmax
    if -128 <= qmin < qmax <= 127:
        return qmin, qmax, torch.int8
    if 0 <= qmin < qmax <= 255:
        return qmin, qmax, torch.uint8
    return qmin, qmax, None


def _derive_bias_qparams(obs_or_fqs):
    """An int32 bias's qparams: activation scale x weight scale, zero point 0
    (Arm's derive function, operation for operation; Cortex-M's omits the
    float32 cast, a no-op on float32 scales)."""
    if len(obs_or_fqs) != 2:
        raise ValueError(f"Expecting two obs/fqs, one for activation and one "
                         f"for weight, got: {len(obs_or_fqs)}")
    act_scale, _ = obs_or_fqs[0].calculate_qparams()
    weight_scale, _ = obs_or_fqs[1].calculate_qparams()
    return (act_scale * weight_scale).to(torch.float32), torch.full_like(
        weight_scale, fill_value=0, dtype=torch.int32)


class SchemeSpecs:
    """The QuantizationSpecs of one Scheme."""

    def __init__(self, scheme):
        self.scheme = scheme
        self.act = self._act()
        self.weight = {kind: self._weight(kind) for kind in ("linear", "conv")}

    # -- activations ------------------------------------------------------------
    def _act(self):
        s = self.scheme
        if s.activation == "none":
            return None
        qmin, qmax, dtype = act_range(s)
        qscheme = (torch.per_tensor_symmetric
                   if s.activation in ("dynamic_per_tensor_symmetric",
                                       "static_per_tensor_symmetric_int16")
                   else torch.per_tensor_affine)
        extra = {} if s.eps is None else {"eps": s.eps}
        if s.dynamic:
            if s.is_qat:
                ctor = DynamicFakeQuantize.with_args(observer=CallRangeObserver,
                                                     **extra)
            else:
                # The default's MinMaxObserver; the stock configs spell a
                # dynamic observer PlaceholderObserver(eps=...).
                base = PlaceholderObserver if s.preset else MinMaxObserver
                ctor = base.with_args(**extra) if extra else base
        elif s.is_qat:
            # Not FusedMovingAvgObsFakeQuantize, which XNNPACK and Arm put
            # here: its C++ kernel computes the qparams the training forward
            # uses (ChooseQuantizationParams), and convert_pt2e exports the
            # observer's calculate_qparams instead -- measured 1 ulp apart on
            # some steps. FakeQuantize over the same moving-average observer
            # trains on the grid convert exports.
            ctor = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                          **extra)
        else:
            # Every stock static config observes activations by histogram.
            ctor = (HistogramObserver.with_args(**extra) if extra
                    else HistogramObserver)
        full = (qmin, qmax) == (-128, 127) and dtype == torch.int8
        implicit = s.cortex_m_spelling and full
        return QuantizationSpec(
            dtype=dtype, quant_min=None if implicit else qmin,
            quant_max=None if implicit else qmax, qscheme=qscheme,
            is_dynamic=s.dynamic, observer_or_fake_quant_ctr=ctor)

    # -- weights ----------------------------------------------------------------
    def _weight(self, kind):
        s = self.scheme
        per_channel = s.granularity(kind) == "per_channel"
        symmetric = s.weight_mapping == "symmetric"
        qscheme = {(True, True): torch.per_channel_symmetric,
                   (True, False): torch.per_channel_affine,
                   (False, True): torch.per_tensor_symmetric,
                   (False, False): torch.per_tensor_affine}[(per_channel,
                                                             symmetric)]
        qmin, qmax = weight_range(s)
        eps = s.weight_eps if s.weight_eps is not None else s.eps
        extra = {} if eps is None else {"eps": eps}
        if s.is_qat and s.qat_weights == "moving_average":
            # XNNPACK's moving-average observer, under FakeQuantize rather than
            # its FusedMovingAvgObsFakeQuantize: the fused kernel trains a
            # symmetric weight on a grid 255/254 wider than the one
            # convert_pt2e exports (measured on every step).
            ctor = FakeQuantize.with_args(
                observer=(MovingAveragePerChannelMinMaxObserver if per_channel
                          else MovingAverageMinMaxObserver), **extra)
        elif s.is_qat and per_channel:
            # Arm: FakeQuantize over the per-channel min/max observer, its
            # range and scheme spelled into the constructor as Arm does.
            ctor = FakeQuantize.with_args(
                observer=PerChannelMinMaxObserver, quant_min=qmin,
                quant_max=qmax, dtype=torch.qint8, qscheme=qscheme,
                reduce_range=False, ch_axis=0, **extra)
        elif s.is_qat:
            ctor = FakeQuantize.with_args(**extra)
        else:
            base = PerChannelMinMaxObserver if per_channel else MinMaxObserver
            ctor = base.with_args(**extra) if extra else base
        implicit = s.cortex_m_spelling and (qmin, qmax) == (-128, 127)
        return QuantizationSpec(
            dtype=torch.int8, quant_min=None if implicit else qmin,
            quant_max=None if implicit else qmax, qscheme=qscheme,
            ch_axis=None if s.cortex_m_spelling and not per_channel else 0,
            is_dynamic=False, observer_or_fake_quant_ctr=ctor)

    # -- bias -------------------------------------------------------------------
    def bias(self, node, kind):
        """The int32 bias spec of `node` (activation x weight), or None.

        Two stock spellings of one rule, scale = activation scale x weight
        scale with zero point 0: Arm's config (QuantizationConfig.
        get_bias_qspec) takes [-2^31 + 1, 2^31 - 1] and the weight's own
        symmetric scheme; Cortex-M's (arm_quantizer_utils' _get_int32_*
        helpers) [-2^31, 2^31 - 2], per channel or no scheme at all."""
        if self.scheme.bias != "int32_derived":
            return None
        per_channel = self.scheme.granularity(kind) == "per_channel"
        i32 = torch.iinfo(torch.int32)
        if self.scheme.cortex_m_spelling:
            qmin, qmax = i32.min, i32.max - 1
            extra = (dict(qscheme=torch.per_channel_symmetric, ch_axis=0)
                     if per_channel else {})
        else:
            qmin, qmax = i32.min + 1, i32.max
            extra = (dict(qscheme=torch.per_channel_symmetric, ch_axis=0)
                     if per_channel else
                     dict(qscheme=torch.per_tensor_symmetric, ch_axis=None))
        return DerivedQuantizationSpec(
            derived_from=((node.args[0], node), (node.args[1], node)),
            derive_qparams_fn=_derive_bias_qparams, dtype=torch.int32,
            quant_min=qmin, quant_max=qmax, **extra)


class Esp32Quantizer(Quantizer):
    """Annotate Linear-family ops for the ESP32-S3 delegate's exact formats.

    With no arguments: int4 per-channel symmetric weights and dynamic int8
    per-tensor symmetric activations. `weight_bits` selects 4,
    8, or the ternary semantic carrier 2 (still i4 on the wire); `preset` and
    the keyword options select the other schemes the backend runs exactly
    (module docstring). The i16 rung is deliberately not offered -- it exists
    for ops the format ladder sends there by policy, not as something a user
    asks a quantizer for.
    """

    def __init__(self, weight_bits: Optional[int] = None,
                 annotate_act="auto", *, preset: Optional[str] = None,
                 weight_granularity=None, weight_mapping: Optional[str] = None,
                 weight_qmin: Optional[int] = None,
                 weight_qmax: Optional[int] = None,
                 activation: Optional[str] = None,
                 act_qmin: Optional[int] = None, act_qmax: Optional[int] = None,
                 eps: Optional[float] = None, weight_eps: Optional[float] = None,
                 bias: Optional[str] = None, is_qat: Optional[bool] = None) -> None:
        self.annotated = 0
        if weight_bits is not None and weight_bits not in (2, 4, 8):
            raise ValueError(
                f"weight_bits must be 2, 4 or 8 (2 is a ternary i4 carrier), "
                f"got {weight_bits}"
            )
        if annotate_act not in ("auto", True, False):
            raise ValueError(
                f"annotate_act must be 'auto', True or False, got {annotate_act!r}"
            )
        self.scheme = resolve_scheme(
            preset, weight_bits=weight_bits,
            weight_granularity=weight_granularity, weight_mapping=weight_mapping,
            weight_qmin=weight_qmin, weight_qmax=weight_qmax,
            activation=activation, act_qmin=act_qmin, act_qmax=act_qmax,
            eps=eps, weight_eps=weight_eps, bias=bias, is_qat=is_qat)
        check_scheme(self.scheme)
        # "auto" annotates activations unless a DYNAMIC configuration shares
        # one between matmuls, which upstream cannot convert (see the module
        # docstring). True forces it and lets convert_pt2e raise; False is
        # weight-only.
        self.annotate_act = annotate_act
        self._init_specs()

    @classmethod
    def from_scheme(cls, scheme, annotate_act="auto", check=True):
        """A quantizer for a resolved Scheme. check=False skips the refusal,
        so that the specs of a scheme the backend does not run yet can be
        compared with its stock config (test_espressif_quantizer.py) -- not a
        way to export one."""
        self = cls.__new__(cls)
        Quantizer.__init__(self)
        self.annotated = 0
        self.scheme = scheme
        if check:
            check_scheme(scheme)
        self.annotate_act = annotate_act
        self._init_specs()
        return self

    def _init_specs(self):
        self.specs = SchemeSpecs(self.scheme)
        self.weight_bits = self.scheme.weight_bits
        self.act_annotated = None   # what "auto" resolved to, for tests

    def _shared_activations(self, targets):
        """Activation nodes feeding more than one matmul this quantizer claims."""
        feeds = {}
        for node, act, _wt in targets:
            feeds.setdefault(act, []).append(node)
        return {a: ns for a, ns in feeds.items() if len(ns) > 1}

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        scheme, specs = self.scheme, self.specs
        # Two passes: the activation decision is a property of the WHOLE graph,
        # so it cannot be made while walking it. Annotating some matmuls with an
        # activation spec and others without would be worse than either choice --
        # QAT would then simulate different arithmetic for different layers of
        # the same model.
        targets = []
        for node in model.graph.nodes:
            got = self._target_of(node)
            if got is not None:
                targets.append(got)

        # Only the symmetric chooser's shared chain trips upstream (see the
        # module docstring); an affine one converts, as XNNPACK's does.
        shared = (self._shared_activations(targets)
                  if scheme.activation == "dynamic_per_tensor_symmetric"
                  else {})
        if specs.act is None:
            use_act = False
        elif self.annotate_act == "auto":
            use_act = not shared
            if shared:
                first = next(iter(shared.items()))
                print(
                    f"esp32 quantizer: {len(shared)} activation(s) feed more "
                    f"than one matmul (e.g. {first[0].name} -> "
                    f"{len(first[1])} matmuls), which torchao's "
                    f"PortNodeMetaForQDQ cannot convert. Annotating WEIGHT-ONLY: "
                    f"the delegate then quantises those activations per row, "
                    f"its own scheme, instead of per tensor, and QAT will not "
                    f"fake-quantise them. Pass annotate_act=True to force it "
                    f"and see the upstream error instead."
                )
        else:
            use_act = bool(self.annotate_act)
        self.act_annotated = use_act
        a_spec = specs.act if use_act else None

        for node, act, wt in targets:
            kind = "linear" if node.target in _LINEAR_TARGETS else "conv"
            # Activation, weight, bias: prepare_qat_pt2e's conv + batch norm
            # rewrite re-keys this map BY POSITION in that order
            # (_update_conv_input_qspec_map_after_replacement); weight first,
            # the activation would get the weight's spec.
            qmap = {act: a_spec} if a_spec is not None else {}
            qmap[wt] = specs.weight[kind]
            bias = node.args[2] if len(node.args) > 2 else None
            b_spec = specs.bias(node, kind) if a_spec is not None else None
            if isinstance(bias, torch.fx.Node) and b_spec is not None:
                qmap[bias] = b_spec
            output = None
            if a_spec is not None and scheme.static:
                end = self._epilogue_end(node, a_spec)
                if end is node:
                    output = a_spec
                else:
                    # The output pair after the activation the delegate fuses
                    # (the stock quantizers' conv/linear + relu patterns).
                    end.meta["quantization_annotation"] = QuantizationAnnotation(
                        output_qspec=a_spec, _annotated=True)
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=qmap,
                # Dynamic configs: no output spec. The delegate writes fp32 and
                # the next op's own annotation decides what happens to it.
                # Annotating the output would insert a q/dq pair at every
                # partition boundary, which is precisely the round trip the
                # intra-partition chain avoids. Static configs annotate it: the
                # delegate absorbs that pair as its output quantisation.
                output_qspec=output,
                _annotated=True,
            )
            self.annotated += 1
        return model

    def _target_of(self, node):
        """(node, act, wt) if this node is one we annotate, else None."""
        scheme = self.scheme
        if node.op != "call_function" or (
                node.target not in _LINEAR_TARGETS
                and node.target not in scheme.conv_targets):
            return None
        if node.meta.get("quantization_annotation") is not None:
            return None
        if (node.target is torch.ops.aten.convolution.default
                and len(node.args) > 6 and bool(node.args[6])):
            return None      # transposed: see _CONV_TARGETS
        # The bias is left in fp32 (unless the scheme derives an int32 one):
        # the delegate adds it after the int32 accumulation is scaled back, so
        # quantising it would only lose range for no traffic saved.
        act, wt = node.args[0], node.args[1]
        if not (isinstance(act, torch.fx.Node) and isinstance(wt, torch.fx.Node)):
            return None
        if (scheme.dynamic_conv == "xnnpack" and node.target in _CONV_TARGETS
                and not _xnnpack_dynamic_conv(node, scheme.is_qat)):
            return None
        return (node, act, wt)

    def _epilogue_end(self, node, out_spec):
        """The node whose output carries `node`'s output pair: `node` itself,
        or the activation (after a batch norm QAT leaves in place) the
        preset's stock quantizer fuses into it."""
        fusion = self.scheme.fusion
        end = node
        conv = node.target in self.scheme.conv_targets
        # A batch norm after a conv survives to prepare only in QAT (PTQ
        # folds it first), and the pair goes after it, as XNNPACK's QAT
        # patterns put it: prepare_qat_pt2e then folds the norm into the conv.
        # (Arm's support table lists Linear -> batch norm too, but its
        # transform decomposes a BatchNorm1d before it can match.)
        if conv and _sole_user(end, _BATCH_NORM):
            end = _sole_user(end, _BATCH_NORM)
        act = _sole_user(end, _fusable(fusion))
        if act is None:
            return end
        if fusion == "arm":
            # ReluFusedPatternCheck: hardtanh only from 0, and no fusion when
            # the output is quantised symmetrically.
            if act.target in _HARDTANH and _min_val(act) != 0.0:
                return end
            if out_spec.qscheme in (torch.per_tensor_symmetric,
                                    torch.per_channel_symmetric):
                return end
        return act

    def validate(self, model: torch.fx.GraphModule) -> None:
        """Complain if nothing was annotated.

        Silence here is the dangerous outcome: prepare/convert both succeed, the
        weights reach the backend as plain fp32 to be re-quantised data-free,
        and a user who ran QAT through this quantizer ships a model whose
        matmuls were never fake-quantised -- without ever being told. The most
        likely cause is a graph spelling matmul as `aten.matmul.default`
        (`x @ W`) rather than nn.Linear, which is deliberately not annotated:
        its second operand is already [K,N], so a ch_axis=0 per-channel spec
        would scale along the wrong axis.
        """
        if self.annotated == 0:
            print(f"esp32: {type(self).__name__} annotated NOTHING -- no "
                  f"linear or convolution found. If this model uses `x @ W` "
                  f"(aten.matmul) or an explicit torch.mm/addmm, rewrite it as "
                  f"nn.Linear: their weight is already transposed and cannot "
                  f"take a per-output-channel spec.")
        return None


def _fusable(fusion):
    """The activations each stock quantizer fuses into a Linear or conv."""
    if fusion == "xnnpack":
        return _RELU                                   # linear_relu, conv_relu
    if fusion == "arm":
        return _RELU + _HARDTANH                       # FUSED_ACTIVATION_OPS
    if fusion == "cortex_m":
        return _RELU + _HARDTANH + _HARDSIGMOID + _CLAMP  # its *_OP_PATTERNS
    # The backend's own: the clamps its epilogue applies before an output
    # quantisation (static_qdq.py).
    return _RELU + _HARDTANH + _CLAMP


def _sole_user(node, targets):
    """`node`'s only user if it is a call to one of `targets`, else None."""
    if len(node.users) != 1:
        return None
    user = next(iter(node.users))
    if user.op == "call_function" and user.target in targets:
        return user
    return None


def _min_val(node):
    return float(node.args[1]) if len(node.args) > 1 else -1.0


def _xnnpack_dynamic_conv(node, is_qat):
    """XNNPACK's dynamic config annotates a convolution only if its weight is
    4-D, it is not depthwise, and a single ReLU does not consume it
    (_do_annotate_conv; conv+relu has no dynamic pattern) -- or, under QAT,
    if a batch norm does (its conv_bn[_relu] patterns are QAT patterns, which
    it applies to a dynamic QAT config too)."""
    if is_qat and _sole_user(node, _BATCH_NORM) is not None:
        return True
    weight = node.args[1].meta.get("val")
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) != 4:
        return False
    if shape is not None and shape[1] == 1:       # is_depthwise_conv(shape, 1)
        return False
    if len(node.users) == 1 and next(iter(node.users)).target in _RELU:
        return False
    return True



# --- torchao: per-group weights and per-token activations ----------------------------
#
# PT2E cannot express per-group weights or per-token activations; torchao can,
# and the delegate runs its spelling exactly. torchao 0.18's
# defaults, read from quant_api.py: Int8DynamicActivationIntxWeightConfig is
# int8 weights, PerGroup(32), SYMMETRIC, algorithm AFFINE, activations
# per-token ASYMMETRIC; IntxWeightOnlyConfig is int8, PerAxis(0), SYMMETRIC;
# Int4WeightOnlyConfig is group 128 with TINYGEMM qparams and an opaque
# packed layout (not the unpacked chain the delegate reads, so not offered).

TORCHAO_BITS = (4, 8)


def _torchao_weights(bits, group, weight_mapping):
    """(weight dtype, granularity, MappingType) after the backend's checks."""
    from torchao.quantization import MappingType
    from torchao.quantization.granularity import PerAxis, PerGroup

    if bits not in TORCHAO_BITS:
        raise SchemeRefused(
            "torchao-bits", f"int{bits} weights: the delegate stores int4 and "
            f"int8 codes; ask for 4 or 8")
    if weight_mapping not in MAPPINGS:
        raise SchemeRefused("weight-mapping", f"{weight_mapping!r}, not one of "
                            f"{MAPPINGS}")
    if group in (None, 0, "per_channel"):
        granularity = PerAxis(0)
    else:
        multiple = 32 if bits == 4 else 16
        if not isinstance(group, int) or group <= 0 or group % multiple:
            raise SchemeRefused(
                "torchao-group", f"group {group!r}: int{bits} groups are "
                f"multiples of {multiple} (or None for one scale per channel)")
        granularity = PerGroup(group)
    mapping = (MappingType.SYMMETRIC if weight_mapping == "symmetric"
               else MappingType.ASYMMETRIC)
    return getattr(torch, f"int{bits}"), granularity, mapping


def esp32_torchao_config(bits=4, group=32, weight_mapping="symmetric", *,
                         activations=True):
    """The torchao config of one scheme this backend runs exactly.

    activations=True: Int8DynamicActivationIntxWeightConfig -- per-token
    affine int8 activations (choose_qparams_affine, eps None = float32's
    smallest normal, round to even before the zero point: A3) over int`bits`
    weights per group of `group` (None: per channel), symmetric or with zero
    points (weight_mapping="asymmetric"), qparams by AFFINE min/max (not
    HQQ: that is an export-time choice the census also covers) in the
    unpacked int8 layout ExecuTorch exports. activations=False: the
    weight-only IntxWeightOnlyConfig with the same weights; the delegate then
    quantises activations its own way (per row).

    Weight scales are pinned to float32 (`weight_scale_dtype`/`scale_dtype`):
    the exporter refuses any other scale dtype by name, since
    choose_qparams_affine rounds a scale to its dtype. The activation chain
    has no scale-dtype or eps knob in torchao 0.18: it takes the input's
    dtype, so the model must run in float32 (a bf16 model's chain is refused
    at export as act-scale-dtype). torchao's QATConfig refuses a pinned
    scale dtype; use esp32_torchao_qat_config for QAT.
    """
    from torchao.quantization import (Int8DynamicActivationIntxWeightConfig,
                                      IntxWeightOnlyConfig)
    from torchao.quantization.quant_api import (IntxChooseQParamsAlgorithm,
                                                IntxPackingFormat)

    dtype, granularity, mapping = _torchao_weights(bits, group, weight_mapping)
    if activations:
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=dtype, weight_granularity=granularity,
            weight_mapping_type=mapping, weight_scale_dtype=torch.float32,
            intx_packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
            intx_choose_qparams_algorithm=IntxChooseQParamsAlgorithm.AFFINE)
    return IntxWeightOnlyConfig(
        weight_dtype=dtype, granularity=granularity, mapping_type=mapping,
        scale_dtype=torch.float32,
        intx_packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
        intx_choose_qparams_algorithm=IntxChooseQParamsAlgorithm.AFFINE)


def esp32_torchao_qat_config(bits=4, group=32, weight_mapping="symmetric", *,
                             activations=True, step="prepare"):
    """torchao QATConfig for esp32_torchao_config's scheme.

    step="prepare": fake-quantisers that compute exactly the grid the convert
    step exports -- IntxFakeQuantizeConfig per token (int8, asymmetric,
    choose_qparams_affine with eps None and float32 scales) on activations,
    and on weights int`bits` per group, symmetric: get_group_qparams_symmetric
    calls choose_qparams_affine with the same range [-2^(b-1), 2^(b-1)-1] and
    eps (float32's smallest normal) as Int8DynamicActivationIntxWeightConfig's
    AFFINE algorithm. They are spelled out rather than inferred from the base
    config (QATConfig(base_config) asserts weight_scale_dtype is None).
    step="convert": the base config, applied to the trained weights.

    Refused: asymmetric weights. torchao's asymmetric weight fake-quantiser
    (get_groupwise_affine_qparams) computes an unsigned [0, 2^b - 1] grid with
    eps 1e-6 and tinygemm float zero points, not the signed grid with int8
    zero points the convert step exports, so the model would not export the
    grid it trained.
    """
    from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

    if step not in ("prepare", "convert"):
        raise ValueError(f"step must be 'prepare' or 'convert', got {step!r}")
    if weight_mapping != "symmetric":
        raise SchemeRefused(
            "torchao-qat-asymmetric", "torchao's asymmetric weight fake-"
            "quantiser trains an unsigned grid with float zero points "
            "(get_groupwise_affine_qparams, eps 1e-6), not the signed grid "
            "with int8 zero points the convert step exports")
    base = esp32_torchao_config(bits, group, weight_mapping,
                                activations=activations)
    if step == "convert":
        return QATConfig(base, step="convert")
    dtype, granularity, mapping = _torchao_weights(bits, group, weight_mapping)
    act = (IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False,
                                  scale_precision=torch.float32)
           if activations else None)
    weight = IntxFakeQuantizeConfig(dtype, granularity, mapping,
                                    scale_precision=torch.float32)
    return QATConfig(activation_config=act, weight_config=weight,
                     step="prepare")


# --- fitted int16 tables --------------------------------------------------------------------

def carry_observer_ranges(prepared):
    """Carry every static activation observer's [min_val, max_val] through
    convert_pt2e to the lowering (lut16_range.py), where
    fold_fitted_luts fits each int16 table over the codes the range covers,
    12.5% wider each way, instead of every code. After calibration, right
    before convert_pt2e:

        prepared = prepare_pt2e(gm, quantizer)   # any stock recipe's
        ...calibrate...
        carry_observer_ranges(prepared)
        converted = convert_pt2e(prepared)
        lower_esp32(torch.export.export(converted, inputs),
                    [Esp32Partitioner(luts16=True)])

    Each stock HistogramObserver, MinMaxObserver or MovingAverageMinMaxObserver
    (a QAT model's FakeQuantize modules too) becomes, in place, a subclass
    whose convert() emits torchao's own quantize/dequantize and then
    espressif::observed_range(codes, min, max) between them, an identity the
    lowering strips first. Weight observers are left alone. Returns the
    number of observers changed."""
    from executorch.backends.esp32s3.quantizer.lut16_range import (
        carry_observer_ranges as carry,
    )

    return carry(prepared)


def fit_tables(prepared, spec=None, **options):
    """Fit an espressif::lut16 table for every int16 activation site of a
    calibrated PT2E model (prepare_pt2e or prepare_qat_pt2e) and train
    through it (lut16_train.py): the static activation observers
    are frozen, and each site's activation becomes a Lut16 module whose
    forward is the device's table and whose backward is the straight-through
    estimator with the activation's own derivative (grad="table": the
    table's slope; learn=True: the entries learn too). convert_pt2e keeps the
    modules; the lowering takes their tables byte for byte, with no refit.
    spec: lut_fit.Spec (byte budget); ranges=True fits over each observer's
    range (as carry_observer_ranges does at lowering). Returns a
    fold_lut16_pass.Lut16Report."""
    from executorch.backends.esp32s3.quantizer.lut16_train import fit_tables as fit

    return fit(prepared, spec, **options)
