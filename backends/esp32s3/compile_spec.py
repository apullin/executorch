# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Esp32CompileSpec: the options the partitioner passes to preprocess."""

import re
from typing import List, final

from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.backends.esp32s3.serialization.espressif_blob import BACKEND_NAME
from executorch.backends.esp32s3.serialization.weight_formats import (
    FORMATS,
    HALF_FORMATS,
)


# When set, weight and scale payloads leave the blob and become named data with this
# external tag, which lands them in a .ptd instead of the .pte. That is the difference
# between a 1.5 MB app-partition ceiling and a flash partition: the real PLE head is
# 12.58 MB and can never be serialised into a program.
DEFAULT_EXTERNAL_TAG = "esp32_weights"


# Compile-spec key selecting the policy; see choose_format.
SPEC_FORMAT_POLICY = "format_policy"


# NO error budget, deliberately. Weight reconstruction error is not activation error
# through a matmul: an output head at 7.12e-02 relative reconstruction error costs
# +0.063/+0.089 nats end to end over two seeds as per-row int4, which is acceptable.
#
# No static weight-error threshold can make this call. int4 always
# looks bad by reconstruction error; what makes it safe is that a large redundant table
# is robust, which is a property of the model and the data, not of the weights alone. The
# only real gate is an end-to-end eval, and a backend at preprocess time has neither the
# data nor the metric.
#
# So "auto" decides on the thing it *can* know, the weight's size: at or above the chip's
# SRAM-residency threshold (chips.py sram_resident_bytes, 128 KiB as int8) it takes the
# weight to be read from flash or PSRAM, so its op bandwidth-bound, and picks i4; below
# it, i8. Where the weights are actually read from is the application's weight-tier
# budget, a runtime option preprocess cannot see. It reports the reconstruction error per
# rung without gating on it. Escalating to i16 is left to an explicit policy, because
# nothing measured in this model asks for it.

# Compile-spec key the partitioner sets to request external weights.
SPEC_EXTERNAL = "external_weights"

# Compile-spec key requesting static (calibrated) activation scales. Preprocess
# then REQUIRES every quantised matmul to carry esp32_act_amax meta -- an
# uncalibrated graph fails loudly at export, not quietly at runtime.
SPEC_STATIC_ACT = "static_act"

# Opt-in 128-element group scales for ordinary quantised weights. Embeddings
# already select this automatically when their rows are wide; Linears keep the
# per-row default until model quality and silicon timing justify the extra scale
# loads. This option lets a group-128 QAT checkpoint reach the exact
# format it trained against instead of silently coarsening wide projections.
SPEC_GROUP_WEIGHTS = "group_weights"

# Compile-spec key asking the RUNTIME for the min-latency posture: wide
# quantised addmms (projections) split their output columns across both cores;
# no convolution splits (the README's Known limits). Unlike every other spec
# here it is not consumed by preprocess -- it survives into the .pte and is
# read by the delegate's init(), because it configures execution rather
# than the artifact. Per-model rather than global precisely so a latency-
# critical model and a background one can be loaded at the same time.
# The name is a contract with kSpecMinLatency in esp32_backend.cpp.
SPEC_MIN_LATENCY = "min_latency"

# Per-op weight-format overrides, keyed by the weight's parameter name (the
# FQN torch.export records in the graph signature, e.g. "fc2.weight"). The
# policy is otherwise global; this lets one sensitive projection or table take
# a different rung (the i16 accuracy rung, say) without forcing the whole model
# onto it. Serialised as "fqn=fmt;fqn=fmt". Not consumed
# by the runtime: an overridden op is an ordinary op at that rung.
SPEC_FORMAT_OVERRIDES = "format_overrides"

# Tied-head deployment options. The frontend represents a logical output prefix
# with PassThroughWeight.logical_rows while retaining the complete backing
# allocation for PTD deduplication; that is an unambiguous AOT marker for a tied
# output head and avoids guessing from a vocabulary-sized dimension.
SPEC_STAGE_TIED_HEAD = "stage_tied_head"
SPEC_SPLIT_TIED_HEAD = "split_tied_head"

# Stage B of static int8 graphs (stage_b.py; on by default): codes, not fp32,
# between claimed ops and across delegate boundaries
# wherever the graph between them can run on codes. The runtime's init accepts
# int8 activation and output records (EXT_ACT_QUANT in_dtype 1, EXT_OUT_QUANT
# out_dtype 1) only in a delegate whose compile specs carry it, and the
# partitioner puts it only on a delegate that holds such a record (a program
# with no code region is byte for byte the same with Stage B on or off).
# The name is a contract with kSpecStageB in esp32_backend.cpp.
SPEC_STAGE_B = "stage_b"

# Planned delegate scratch (ExecuTorch #21985; Esp32Partitioner's default): preprocess
# declares each delegate's per-call working memory, as the runtime will lay it
# out (serialization/scratch_layout.py), as a DelegateScratchSpec. ExecuTorch's
# planner then places it in the method's arena, where it may reuse the bytes
# of whatever is dead during the call, and the runtime finds it through
# BackendExecutionContext::scratch_buffers() instead of taking it from the
# temp allocator, so a program states its whole memory need at load. Consumed
# by preprocess only; the runtime ignores the key and uses a planned buffer
# wherever one arrives. Upstream marks the API experimental.
SPEC_PLAN_SCRATCH = "plan_scratch"


@final
class Esp32CompileSpec:
    """The AOT decisions the graph cannot express, as a validated object.

    Compile specs are a list of (str, bytes) on the wire, which is fine for the
    wire and poor everywhere else: encoding them by hand in the partitioner and
    decoding them by hand in preprocess would repeat the key strings, the byte
    encodings and the defaults in both places, an untyped duplicated contract
    that drifts silently. Arm's `EthosUCompileSpec` is the in-tree precedent for
    this shape.

    Validation lives HERE rather than in preprocess because a bad format policy
    is a build error the caller can fix, and discovering it after partitioning
    means the traceback points at the backend instead of at the call.
    """

    def __init__(self, external_weights: bool | str = False,
                 format_policy: str = "i8",
                 static_act: bool = False,
                 min_latency: bool = False,
                 group_weights: bool = False,
                 stage_tied_head: bool = False,
                 split_tied_head: bool = False,
                 format_overrides=None,
                 stage_b: bool | None = None,
                 plan_scratch: bool = False) -> None:
        valid_policies = set(FORMATS) | {"auto", "fp32"} | set(HALF_FORMATS)
        if format_policy not in valid_policies:
            raise RuntimeError(
                f"unknown format policy {format_policy!r}; "
                f"expected one of {sorted(valid_policies)}"
            )
        overrides = dict(format_overrides or {})
        for fqn, fmt in overrides.items():
            if not isinstance(fqn, str) or not fqn or ";" in fqn or "=" in fqn:
                raise RuntimeError(
                    f"format_overrides key {fqn!r} must be a parameter name")
            if fmt not in FORMATS and fmt not in HALF_FORMATS:
                raise RuntimeError(
                    f"format_overrides[{fqn!r}] = {fmt!r}; expected one of "
                    f"{sorted(set(FORMATS) | set(HALF_FORMATS))}")
        if overrides and format_policy == "fp32":
            raise RuntimeError(
                "format_policy='fp32' cannot carry format_overrides: the fp32 "
                "diagnostic opcode has no quantised rung to override to")
        self.format_overrides = overrides
        if isinstance(external_weights, str):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", external_weights):
                raise RuntimeError(
                    "external_weights tag must contain only letters, digits, "
                    f"underscore or hyphen, got {external_weights!r}"
                )
            self.external_tag = external_weights
        elif isinstance(external_weights, bool):
            self.external_tag = DEFAULT_EXTERNAL_TAG if external_weights else None
        else:
            raise TypeError("external_weights must be bool or a PTD tag string")
        self.external_weights = self.external_tag is not None
        self.format_policy = format_policy
        self.static_act = bool(static_act)
        self.min_latency = bool(min_latency)
        self.group_weights = bool(group_weights)
        self.stage_tied_head = bool(stage_tied_head)
        self.split_tied_head = bool(split_tied_head)
        self.plan_scratch = bool(plan_scratch)
        # None, the default, is on, except under a float policy, whose
        # loops quantise nothing for Stage B to keep as codes; an explicit
        # True there is refused below.
        float_policy = (self.format_policy == "fp32"
                        or self.format_policy in HALF_FORMATS)
        self.stage_b = (not float_policy) if stage_b is None else bool(stage_b)
        if self.min_latency and self.split_tied_head:
            raise RuntimeError(
                "min_latency and split_tied_head are mutually exclusive: "
                "the former splits every eligible projection, while the latter "
                "exists to preserve a head-only second-core posture")
        if self.stage_tied_head and not self.external_weights:
            raise RuntimeError(
                "stage_tied_head requires external_weights: staging an inline "
                "weight would duplicate it between the PTE and PSRAM")
        if self.format_policy == "i16" and self.static_act:
            # Unconditional, so refused here rather than at the first matmul's
            # preprocess. (An i16 OVERRIDE under static_act depends on the op
            # it names, and preprocess refuses that one by name.)
            raise RuntimeError(
                "format_policy='i16' cannot combine with static_act: static "
                "activation scales are int8-range, and the i16 rung quantises "
                "activations to int16 per row")
        if float_policy:
            # The float-weight loops quantise nothing. (Their weights go to
            # the PTD under external_weights, as the integer rungs' do.)
            incompatible = []
            if self.static_act:
                incompatible.append("static_act")
            if self.min_latency:
                incompatible.append("min_latency")
            if self.group_weights:
                incompatible.append("group_weights")
            if self.stage_tied_head:
                incompatible.append("stage_tied_head")
            if self.split_tied_head:
                incompatible.append("split_tied_head")
            if self.stage_b:
                incompatible.append("stage_b")
            if incompatible:
                raise RuntimeError(
                    f"format_policy={self.format_policy!r} cannot combine with "
                    + ", ".join(incompatible)
                )

    def to_list(self) -> List[CompileSpec]:
        """Only flags that are on are emitted (absent means off on the wire),
        so an option left at its default adds nothing to the program.
        stage_b is on by default, and Esp32Partitioner emits it only on the
        delegates that hold int8 codes (_partition)."""
        specs = []
        if self.external_weights:
            value = (bytes([1]) if self.external_tag == DEFAULT_EXTERNAL_TAG
                     else self.external_tag.encode())
            specs.append(CompileSpec(SPEC_EXTERNAL, value))
        specs.append(CompileSpec(SPEC_FORMAT_POLICY, self.format_policy.encode()))
        if self.static_act:
            specs.append(CompileSpec(SPEC_STATIC_ACT, bytes([1])))
        if self.min_latency:
            specs.append(CompileSpec(SPEC_MIN_LATENCY, bytes([1])))
        if self.group_weights:
            specs.append(CompileSpec(SPEC_GROUP_WEIGHTS, bytes([1])))
        if self.stage_tied_head:
            specs.append(CompileSpec(SPEC_STAGE_TIED_HEAD, bytes([1])))
        if self.split_tied_head:
            specs.append(CompileSpec(SPEC_SPLIT_TIED_HEAD, bytes([1])))
        if self.format_overrides:
            wire = ";".join(f"{fqn}={fmt}" for fqn, fmt
                            in sorted(self.format_overrides.items()))
            specs.append(CompileSpec(SPEC_FORMAT_OVERRIDES, wire.encode()))
        if self.stage_b:
            specs.append(CompileSpec(SPEC_STAGE_B, bytes([1])))
        if self.plan_scratch:
            specs.append(CompileSpec(SPEC_PLAN_SCRATCH, bytes([1])))
        return specs

    @classmethod
    def from_list(cls, compile_specs) -> "Esp32CompileSpec":
        """The decode half. An unknown key is REFUSED rather than ignored: specs
        are routed per backend id, so one arriving here was addressed to this
        backend, and silently dropping it means a user believes a setting is on
        when it is not -- the same reasoning as the runtime option API."""
        flags = {
            SPEC_STATIC_ACT: False,
            SPEC_MIN_LATENCY: False,
            SPEC_GROUP_WEIGHTS: False,
            SPEC_STAGE_TIED_HEAD: False,
            SPEC_SPLIT_TIED_HEAD: False,
            SPEC_STAGE_B: False,
            SPEC_PLAN_SCRATCH: False,
        }
        external_weights = False
        policy = "i8"
        overrides = {}
        for spec in compile_specs:
            if spec.key == SPEC_EXTERNAL:
                value = bytes(spec.value)
                external_weights = (True if value == bytes([1])
                                    else value.decode().rstrip("\0"))
            elif spec.key in flags:
                flags[spec.key] = bool(spec.value and spec.value[0])
            elif spec.key == SPEC_FORMAT_POLICY and spec.value:
                policy = bytes(spec.value).decode().rstrip("\0")
            elif spec.key == SPEC_FORMAT_OVERRIDES:
                wire = bytes(spec.value).decode().rstrip("\0")
                for item in filter(None, wire.split(";")):
                    fqn, _, fmt = item.partition("=")
                    overrides[fqn] = fmt
            else:
                raise RuntimeError(
                    f"{BACKEND_NAME}: unknown compile spec {spec.key!r}")
        return cls(external_weights=external_weights, format_policy=policy,
                   static_act=flags[SPEC_STATIC_ACT],
                   min_latency=flags[SPEC_MIN_LATENCY],
                   group_weights=flags[SPEC_GROUP_WEIGHTS],
                   stage_tied_head=flags[SPEC_STAGE_TIED_HEAD],
                   split_tied_head=flags[SPEC_SPLIT_TIED_HEAD],
                   format_overrides=overrides,
                   stage_b=flags[SPEC_STAGE_B],
                   plan_scratch=flags[SPEC_PLAN_SCRATCH])

    def __repr__(self) -> str:
        external = self.external_tag if self.external_tag is not None else False
        return (f"Esp32CompileSpec(external_weights={external!r}, "
                f"format_policy={self.format_policy!r}, "
                f"static_act={self.static_act}, "
                f"min_latency={self.min_latency}, "
                f"group_weights={self.group_weights}, "
                f"stage_tied_head={self.stage_tied_head}, "
                f"split_tied_head={self.split_tied_head}, "
                f"format_overrides={self.format_overrides!r}, "
                f"stage_b={self.stage_b}, "
                f"plan_scratch={self.plan_scratch})")

    def __eq__(self, other) -> bool:
        return isinstance(other, Esp32CompileSpec) and vars(self) == vars(other)
