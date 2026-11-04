# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Esp32Partitioner."""

import copy
import logging
import os
from collections import Counter
from typing import Dict, final

import torch
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram

from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec
from executorch.backends.esp32s3.operator_support.const_weights import (
    FOLDED,
    _const_fqn,
    _const_tensor,
    _is_torchao_dequant,
    _placeholder_index,
    _w_off_member,
    _w_off_pattern,
)
from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
    Esp32OperatorSupport,
    REQUANTIZE_POLICIES,
    _written_buffer,
)
from executorch.backends.esp32s3.operator_support.patterns import (
    _selected_mm_pattern,
    _weight_operand,
)
from executorch.backends.esp32s3.serialization.espressif_blob import BACKEND_NAME


logger = logging.getLogger(__name__)

def _plan_scratch_default() -> bool:
    """What Esp32Partitioner(plan_scratch=None), the default, means: on. On
    the board the story decodes as fast with planned scratch as on the temp
    tiers (+0.09% at one PSRAM timing point, every value identical), and
    a planned method states its whole memory need at load; ExecuTorch still
    marks the scratch API (DelegateScratchSpec) experimental.
    ESP32_PLAN_SCRATCH=0 (or 1) in the environment says otherwise for a whole
    run, read once per process: the census, run_parity and the board
    scripts lower in child processes, which inherit it, so an A/B needs no
    flag on every exporter. A planned program says so in its compile specs
    and its emit log."""
    value = os.environ.get("ESP32_PLAN_SCRATCH", "")
    if value not in ("", "0", "1"):
        raise ValueError(f"ESP32_PLAN_SCRATCH={value!r}: expected 0 or 1")
    return value != "0"


PLAN_SCRATCH_DEFAULT = _plan_scratch_default()


# The meta an op carries once it has absorbed one of a recipe's activation
# chains (strip_dynamic_act_qdq, static_qdq.absorb_static_qdq), or once a
# dynamic chain it could not absorb is computed in place in front of it.
_ABSORBED_CHAIN = ("esp32_act_quant", "esp32_act_per_tensor",
                   "esp32_act_absorbed", "esp32_act_kept")


def _recipe_qdq(node) -> bool:
    """A quantize, dequantize or choose_qparams op of PT2E's
    quantized_decomposed or torchao's namespace: what a quantisation recipe
    leaves in a graph, on activations or on weights."""
    from executorch.backends.esp32s3.partition.espressif_partitioner import _overload

    op = _overload(node)
    if op is None:
        return False
    namespace, _, name = op._schema.name.partition("::")
    return (namespace in ("quantized_decomposed", "torchao")
            and ("quantize" in name or "choose_qparams" in name))


def _recipe_quantised(ep: ExportedProgram) -> bool:
    """Whether this program is recipe-quantised: what decides, under the
    default format policy, that a weight the recipe left float stays float
    (Esp32Partitioner).

    "Recipe-quantised" means the recipe quantised at least one weight or
    activation the backend sees. A recipe leaves its quantize/dequantize ops
    on the activations and weights it quantised; some have left the graph by
    the time the partitioner runs: strip_dynamic_act_qdq absorbs a dynamic
    activation chain into the op it feeds, and absorb_static_qdq points a
    consumer past its static pair, each recording the scheme in that op's
    meta. A float-offset weight (tinygemm/HQQ, GGUF) is a dequantise spelled
    in elementwise ops.

    A recipe that quantises no weight and no activation the backend sees
    gets the format policy, as a float model does. ExecuTorch's KV-cache
    recipe alone (--quantize_kv_cache, the census's et-kv-quant) is one:
    stock_llm.prepare respells its cache's quantize and dequantize as the
    backend's own espressif ops before partitioning, so its float Linears
    and table are quantised by the policy."""
    for node in ep.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if (_recipe_qdq(node) or any(key in node.meta for key in _ABSORBED_CHAIN)
                or _w_off_member(node) is not None):
            return True
    return False


def _kept_float_lines(ep, support, reporter):
    """The partitioner's report of the weights a recipe left float
    (Esp32OperatorSupport._keeps_float): one line per weight, with its shape,
    its fp32 bytes and where each op reading it runs."""
    uses = {}
    for node in ep.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        half = support._keeps_half(node)
        if half is None and not support._keeps_float(node):
            continue
        weight = _weight_operand(node)
        selected = _selected_mm_pattern(node, ep)
        if selected is not None:
            weight = selected["table"]
        while weight.op == "call_function" and weight.target in FOLDED:
            weight = weight.args[0]
        if half is not None and "delegation_tag" in node.meta:
            where = f"{half} in the delegate"
        elif node.target is exir_ops.edge.aten.embedding.default:
            where = "portable fp32 gather"
        elif "delegation_tag" in node.meta:
            where = "fp32 loop in the delegate"
        else:
            where = "portable fp32: " + reporter._rejected_nodes.get(
                node, "no partition took it")
        uses.setdefault((weight, half), []).append(f"{node.name} -> {where}")
    lines = []
    for (weight, half), where in uses.items():
        val = weight.meta["val"]
        size = val.numel() * (2 if half is not None else val.element_size())
        lines.append(
            f"esp32 partitioner: kept float, as the recipe left it: "
            f"{_const_fqn(ep, weight) or weight.name} "
            f"[{'x'.join(str(d) for d in val.shape)}] {half or 'fp32'} "
            f"{size:,} B; " + "; ".join(where))
    return lines


def _qk_value_lines(support):
    """The partitioner's report of the q@kT value check
    (operator_support/qk_values.py): one line per q@kT it kept portable, with
    the operand and the test that found the grid; a fallback it took; and,
    when the program could not be run, that the rules alone decided. Nothing
    when no q@kT reached the check, or it ran and found no grid."""
    from executorch.backends.esp32s3.operator_support.qk_values import _CHECKS

    lines = [f"esp32 partitioner: q@kT {node.name} left portable by its "
             f"operand values: {why}"
             for node, why in support.value_refusals.items()]
    lines += [f"esp32 partitioner: q@kT {node.name}: no verdict from values "
              f"({'; '.join(dict.fromkeys(whys))}); the rules alone decided it"
              for node, whys in support.value_silent.items()
              if node not in support.value_refusals]
    ep = support.exported_program
    got = _CHECKS.get(ep.graph_module) if ep is not None else None
    check = got[1] if got is not None and got[1].ep is ep else None
    if check is None or check.samples is None:
        return lines
    for note in check.notes:
        lines.append(f"esp32 partitioner: q@kT value check: {note}")
    if check.over_budget is not None:
        lines.append(f"esp32 partitioner: q@kT value check passed its budget "
                     f"({check.over_budget}); every q@kT it would judge "
                     f"stays portable")
    if check.failure is not None:
        lines.append(f"esp32 partitioner: q@kT value check did not run "
                     f"({check.failure}); the rules alone decided every q@kT")
    # Keep the verdicts, drop the samples: this program is the lowered
    # program's root, which upstream caches keep alive.
    check.release()
    return lines


@final
class Esp32Partitioner(Partitioner):
    # Partition granularity: one delegate per matched pattern today. On the
    # story Glimmer (d256/L8) that is 116 delegate blobs across two methods,
    # each with its own header, init and PSRAM staging pass, and a 170 KB
    # program (170,100 B on ExecuTorch 1.6 with planned scratch), too big for
    # the default 148 KB partition. The MCU-shaped answer, not implemented here yet, is one
    # delegate per method (whole-graph lowering with portable islands
    # inlined).
    def __init__(self, external_weights: bool | str = False,
                 format_policy: str | None = None,
                 static_act: bool = False,
                 quantize_qk: bool = False,
                 min_latency: bool = False,
                 group_weights: bool = False,
                 stage_tied_head: bool = False,
                 split_tied_head: bool = False,
                 conv: bool = True,
                 requantize_promoted: str = "half",
                 format_overrides=None,
                 stage_b: bool | None = None,
                 luts16: bool = True,
                 fused_norms: bool = True,
                 plan_scratch: bool | None = None) -> None:
        # Compile specs are how a partitioner tells the backend about a decision it
        # cannot see from the graph. Here: whether this model's weights are big enough
        # to belong outside the program, and which rung of the format ladder to use.
        #
        # stage_b (on by default; None resolves to off only under a
        # float format_policy, Esp32CompileSpec): a static int8 graph keeps
        # its activations as int8 codes between claimed ops and across
        # delegate boundaries wherever the ops between can run on codes
        # (stage_b.py), instead of Stage A's fp32 -- the same codes, a quarter
        # of the activation memory. The spec rides only on the delegates that
        # hold codes, so a program with no code region is byte for byte
        # Stage A's. stage_b=False: the rewrite never runs and no delegate
        # carries the spec.
        #
        # Both are per-model rather than global because both are properties of the
        # deployment, not of the graph: the same block wants i8 inline in a smoke test
        # and i4 from a flash partition in the shipped model.
        # Gate-only: it decides what is CLAIMED, so it never reaches the blob
        # or the runtime and is not part of the compile spec. Validated HERE as
        # well as in the gate, because the gate is built lazily in partition() --
        # a typo'd policy would otherwise pass the constructor and silently mean
        # "half", the same failure Esp32CompileSpec exists to prevent for
        # format_policy.
        if requantize_promoted not in REQUANTIZE_POLICIES:
            raise ValueError(
                f"requantize_promoted must be one of {REQUANTIZE_POLICIES}, "
                f"got {requantize_promoted!r}")
        self.requantize_promoted = requantize_promoted
        # format_policy None is the default: i8 rows for the float weights of
        # a float model, while in a program a recipe quantised
        # (_recipe_quantised) a weight the recipe left float keeps that
        # precision -- the fp32 loop for a matmul, portable fp32 for a
        # gather, the fp32 convolution for a conv
        # (Esp32OperatorSupport._keeps_float, _KEEP_FLOAT_OPS), and a half
        # checkpoint's weight its fp16/bf16 rung (_keeps_half). fp32 costs
        # little on this part, and it is what the recipe's own model computes. An explicit
        # policy quantises every float weight it claims; so does a
        # format_overrides entry, for its one.
        self.keep_recipe_float = format_policy is None
        # plan_scratch (None: PLAN_SCRATCH_DEFAULT, on): each delegate
        # declares its per-call working memory to ExecuTorch's memory planner
        # (compile_spec.SPEC_PLAN_SCRATCH), which places it in the method's
        # arena; off, the runtime takes it from the temp allocator at every
        # call. A preprocess decision, so it rides in the compile spec, and
        # off adds nothing to the program.
        if plan_scratch is None:
            plan_scratch = PLAN_SCRATCH_DEFAULT
        self.compile_spec = Esp32CompileSpec(
            external_weights=external_weights,
            format_policy="i8" if format_policy is None else format_policy,
            static_act=static_act, min_latency=min_latency,
            group_weights=group_weights, stage_tied_head=stage_tied_head,
            split_tied_head=split_tied_head, format_overrides=format_overrides,
            stage_b=stage_b, plan_scratch=plan_scratch)
        self.delegation_spec = DelegationSpec(BACKEND_NAME,
                                              self.compile_spec.to_list())
        # The spec of a delegate that holds no Stage B codes: the runtime
        # needs "stage_b" only where a record says int8 in or out, and a
        # program with no code region keeps Stage A's bytes.
        self._codeless_spec = self.delegation_spec
        if self.compile_spec.stage_b:
            codeless = copy.copy(self.compile_spec)
            codeless.stage_b = False
            self._codeless_spec = DelegationSpec(BACKEND_NAME,
                                                 codeless.to_list())
        # A partition-shape decision, not a preprocess one, so it rides on the
        # partitioner instance rather than a compile spec: preprocess handles
        # whatever bmm the gate admitted, unconditionally.
        self.quantize_qk = quantize_qk
        self.conv = conv
        # luts16 (on by default): lower_esp32 also folds the int16
        # activations quant_lut refuses as lut-wide-codes into fitted
        # espressif::lut16 tables (fold_lut16_pass.py). A lowering decision the
        # partitioner only carries: a program with no int16 activation is
        # untouched, and luts16=False changes nothing by a byte.
        self.luts16 = luts16
        # fused_norms (on by default): lower_esp32 rewrites a stock
        # recipe's decomposed quantised RMSNorm or LayerNorm into one
        # espressif::qnorm (fuse_qnorm_pass.py) before to_edge. A graph
        # rewrite, not a claim: it never reaches the blob or the compile
        # spec; a program with no quantised norm is untouched, and off
        # nothing runs.
        self.fused_norms = fused_norms
        # The exported program's example inputs, the first input set of the
        # q@kT value check (operator_support/qk_values.py): lower_esp32 sets
        # them from the program it is given (to_edge drops them); a caller
        # partitioning some other way may set them, and None draws every set.
        # A partition-time input only: never in the blob or the compile spec.
        self.example_inputs = None

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        with _placeholder_index(exported_program):
            return self._partition(exported_program)

    def gate(self, exported_program: ExportedProgram) -> "Esp32OperatorSupport":
        """A reporter-free gate with this partitioner's policies, for a pass
        that must know before partitioning what partition() will claim
        (static_qdq.absorb_static_qdq)."""
        return Esp32OperatorSupport(
            quantize_qk=self.quantize_qk, conv=self.conv,
            requantize_promoted=self.requantize_promoted,
            format_policy=self.compile_spec.format_policy,
            exported_program=exported_program,
            keep_float=self._keep_float(exported_program),
            overrides=self.compile_spec.format_overrides,
            static_act=self.compile_spec.static_act,
            example_inputs=self.example_inputs)

    def _keep_float(self, exported_program: ExportedProgram) -> bool:
        """Whether a weight a recipe left float keeps its precision in this
        program: the default format policy, and a recipe-quantised program.
        The one place that decides it; the absorption pre-pass and the
        partition each ask, and the meta the first leaves keeps the second's
        answer the same."""
        return self.keep_recipe_float and _recipe_quantised(exported_program)

    def _partition(self, exported_program: ExportedProgram) -> PartitionResult:
        from executorch.backends.esp32s3._passes.stage_b import STAGE_B

        gm = exported_program.graph_module
        partition_tags: Dict[str, DelegationSpec] = {}
        # An override naming no weight is a typo, and a typo that silently
        # leaves the intended op at the policy rung is the failure this
        # object exists to prevent (same reasoning as unknown spec keys).
        if self.compile_spec.format_overrides:
            sig = exported_program.graph_signature
            known = set(sig.inputs_to_parameters.values())
            known |= set(sig.inputs_to_buffers.values())
            known |= set(sig.inputs_to_lifted_tensor_constants.values())
            unknown = sorted(set(self.compile_spec.format_overrides) - known)
            if unknown:
                raise RuntimeError(
                    f"{BACKEND_NAME}: format_overrides name(s) {unknown} match "
                    f"no parameter, buffer or constant of this program; "
                    f"known: {sorted(known)[:12]}{'...' if len(known) > 12 else ''}")
        # Kept on the instance so a caller -- or a test -- can read the reasons
        # after the fact instead of scraping stdout.
        # A fold (a weight's transpose, slice, dequantise, float-offset
        # chain) must land in the partition of every op reading it:
        # preprocess has no runtime form for it. The gate claims one only
        # with all its readers, but grouping runs after the gate and merges
        # two claimed readers only when no unclaimed op lies between them
        # (t = w.t(); cos(x @ t) @ t), so a fold whose readers it puts in
        # different partitions is refused, with its readers, and the graph
        # grouped again; until nothing is split. lower_esp32 and
        # calibrate_static_act give each reader its own copy first
        # (split_shared_folds), so this only catches a graph partitioned
        # without them (to_edge_transform_and_lower).
        refused = set()
        while True:
            self.reporter = WhyNoPartitionReporter()
            support = Esp32OperatorSupport(
                quantize_qk=self.quantize_qk, reporter=self.reporter,
                conv=self.conv, requantize_promoted=self.requantize_promoted,
                format_policy=self.compile_spec.format_policy,
                exported_program=exported_program,
                keep_float=self._keep_float(exported_program),
                overrides=self.compile_spec.format_overrides,
                static_act=self.compile_spec.static_act,
                refused_folds=refused,
                example_inputs=self.example_inputs)
            partitions = list(generate_pattern_op_partitions(gm, op_support=support))
            split = _split_folds(partitions) - refused
            if not split:
                break
            refused |= split
        self.support = support
        for partition in partitions:
            tag = f"tag{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
                # Pull constant operands in so preprocess() owns their bytes. Tagging a
                # non-constant placeholder -- an activation input -- is rejected by
                # backend_api, and rightly: those must be passed in at call time.
                # Nor a buffer the forward writes: pulled in, its first value would
                # be packed into the blob and read on every call while the program
                # updated its own copy. The gate refuses an op that reads one as a
                # constant (_written_operand_problem); left out, one that slipped
                # through reaches preprocess as a runtime input and is refused there
                # by name, never frozen.
                for arg in node.all_input_nodes:
                    if arg.op == "placeholder" and (
                        _const_tensor(exported_program, arg) is not None
                    ) and not _written_buffer(exported_program.graph_signature, arg):
                        arg.meta["delegation_tag"] = tag
            partition_tags[tag] = (
                self.delegation_spec
                if any(STAGE_B in node.meta for node in partition.nodes)
                else self._codeless_spec)
        # A matmul or conv that absorbed a dynamic activation chain
        # (strip_dynamic_act_qdq) has to be delegated: the chain is gone from
        # the graph, so run portable it would compute on the unquantised
        # tensor, silently. Refused by name instead, with the gate's reason.
        # The same for one pointed at the tensor before its static pair
        # (static_qdq.absorb_static_qdq); a static record on an op that still
        # reads the pair's dequantised value needs no delegate to be right.
        for node in gm.graph.nodes:
            if node.op != "call_function" or "delegation_tag" in node.meta:
                continue
            scheme = node.meta.get("esp32_act_quant")
            static = scheme is not None and scheme.chooser == "static"
            if (node.meta.get("esp32_act_per_tensor")
                    or (scheme is not None and not static)
                    or node.meta.get("esp32_act_absorbed")):
                why = self.reporter._rejected_nodes.get(
                    node, "accepted by the gate, but no partition took it")
                raise RuntimeError(
                    f"{BACKEND_NAME}: {node.name} absorbed a "
                    f"{'static' if static else 'dynamic'} activation "
                    f"quantiser but is not delegated "
                    f"[act-consumer-unclaimed]: {why}")
        _check_requant_pairs(gm)
        _check_stage_b(gm)
        if support.keep_float:
            for line in _kept_float_lines(exported_program, support,
                                          self.reporter):
                print(line)
        for line in _qk_value_lines(support):
            print(line)
        if support.dynamic_refused:
            # Not an error: the portable kernels handle dynamic shapes, so the
            # model still lowers and runs. Said out loud because a silent
            # zero-partition result is indistinguishable from a matcher bug.
            print(
                f"esp32 partitioner: {support.dynamic_refused} node(s) left "
                "portable for carrying dynamic (symbolic) dimensions; this "
                "backend claims static shapes only"
            )
        # The full per-node table goes to logging at INFO, the same level and the
        # same format the Arm backend uses, so it is there when a user turns
        # logging up and silent when they have not asked. The summary below is
        # unconditional but bounded: a model with 800 unclaimed nodes prints a
        # handful of lines, not 800 rows. Built only when it will be logged:
        # tabulate's text wrapping costs ~0.2 s per 1,000 rows.
        if logger.isEnabledFor(logging.INFO):
            logger.info("%s", self.reporter.get_table_report())
        if support.kinds:
            total = sum(support.kinds.values())
            head = ", ".join(f"{n}x {kind}"
                             for kind, n in support.kinds.most_common(4))
            print(f"esp32 partitioner: {total} node(s) not claimed ({head}"
                  f"{', ...' if len(support.kinds) > 4 else ''}); set "
                  f"logging.getLogger('{__name__}').setLevel('INFO') for the "
                  f"per-node table")

        # THE OTHER HALF. Everything above reports the gate's refusals. This
        # reports nodes the gate ACCEPTED that no partition took -- the failure
        # that reads as success, because a clean refusal list looks like a
        # healthy lowering even when most of the model's matmuls went portable.
        #
        # A model can have no refused matmul and still delegate only a third
        # of them. Grouping runs
        # after the gate and answers to its own constraints (single-entry
        # regions, no cycles through unclaimed nodes), so "supported" and
        # "delegated" are genuinely different questions and both need answering.
        self.dropped = sorted(
            (n for n in support.accepted if "delegation_tag" not in n.meta),
            key=lambda n: n.name)
        if self.dropped:
            kinds = Counter(
                str(n.target).split("aten.")[-1].split(".")[0].strip("<> ")
                for n in self.dropped)
            mm = sum(v for k, v in kinds.items() if k in ("addmm", "mm", "bmm"))
            head = ", ".join(f"{n}x {k}" for k, n in kinds.most_common(4))
            print(f"esp32 partitioner: {len(self.dropped)} SUPPORTED node(s) "
                  f"reached no partition ({head}"
                  f"{', ...' if len(kinds) > 4 else ''})")
            if mm:
                # Matmuls are the whole point of this backend, so losing one to
                # grouping is a different severity from losing a view.
                print(f"  WARNING: {mm} of them are matmuls -- the gate claimed "
                      f"them and grouping did not; delegate coverage is lower "
                      f"than the refusal list above implies")
            logger.info("supported but unpartitioned: %s",
                        ", ".join(n.name for n in self.dropped))
        if _is_to_backend_fake_copy(exported_program):
            _defer_codegen(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


def _split_folds(partitions):
    """The folds grouping put in one partition while a reader of theirs is in
    another, or in none: a transpose or slice of a constant, a weight
    dequantise, a float-offset chain's root."""
    where = {node: i for i, partition in enumerate(partitions)
             for node in partition.nodes}
    split = set()
    for node, i in where.items():
        if not (node.op == "call_function"
                and (node.target in FOLDED or _is_torchao_dequant(node)
                     or _w_off_pattern(node) is not None)):
            continue
        if any(where.get(user) != i for user in node.users):
            split.add(node)
    return split


def _check_requant_pairs(gm) -> None:
    """A static pair claimed as an op's output quantisation lands whole, in
    its op's partition: q's int8 codes never cross a delegate boundary
    (Stage B's, which _check_stage_b holds), and a d whose q stayed portable
    would read them."""
    from executorch.backends.esp32s3._passes.stage_b import STAGE_B
    from executorch.backends.esp32s3._passes.static_qdq import (
        is_static_q,
        requant_dq,
        static_pair,
    )

    for node in gm.graph.nodes:
        q = node if is_static_q(node) else static_pair(node)
        if (q is None or "delegation_tag" not in node.meta
                or STAGE_B in node.meta or STAGE_B in q.meta):
            continue
        d = requant_dq(q)
        tags = {n.meta.get("delegation_tag") for n in (q.args[0], q, d)
                if n is not None}
        if d is None or len(tags) != 1:
            raise RuntimeError(
                f"{BACKEND_NAME}: static pair {q.name} was claimed as an "
                f"output quantisation but did not land whole in its op's "
                f"partition (tags {sorted(map(str, tags))}); gate and grouping "
                f"disagree")


def _check_stage_b(gm) -> None:
    """Stage B's codes (stage_b.py) land with their ops: a producer's quantize
    in the partition of the op that writes its codes, a consumer's dequantise
    in the partition of the op that reads them. Either left portable, or in
    another partition, would compute with no op behind it."""
    from executorch.backends.esp32s3._passes.stage_b import IN, OUT, STAGE_B

    for node in gm.graph.nodes:
        stage = node.meta.get(STAGE_B)
        if stage is None:
            continue
        if stage == OUT:
            peer = node.args[0]
        elif stage == IN:
            peer = next(iter(node.users), None)
        else:
            continue
        tags = {getattr(n, "meta", {}).get("delegation_tag")
                for n in (node, peer)}
        if None in tags or len(tags) != 1:
            raise RuntimeError(
                f"{BACKEND_NAME}: Stage B {'quantize' if stage == OUT else 'dequantise'} "
                f"{node.name} did not land in the partition of "
                f"{getattr(peer, 'name', peer)} [stage-b-unclaimed] (tags "
                f"{sorted(map(str, tags))}); gate and grouping disagree")


def _is_to_backend_fake_copy(exported_program) -> bool:
    """True for the program ExecuTorch's to_backend hands a partitioner: its
    own copy of the graph with fake weights (get_fake_program), which it
    fuses, rebuilds the result from, and drops. A caller's real program has
    real weights, so it never qualifies."""
    from torch._subclasses.fake_tensor import FakeTensor

    weights = list(exported_program.state_dict.values())
    return bool(weights) and all(isinstance(w, FakeTensor) for w in weights)


def _defer_codegen(gm: torch.fx.GraphModule) -> None:
    """Generate gm's Python source on first call or .code access, not on
    every graph edit.

    For to_backend's fake copy only (_is_to_backend_fake_copy). It fuses one
    partition at a time, and each fusion ends in legalize_graph, whose
    `gm.graph = sorted_graph` makes the GraphModule regenerate and compile the
    source of the WHOLE top-level graph: partitions x nodes, 2.8 s of the
    story Glimmer's 6.5 s to_backend. Nothing runs the copy before to_backend
    builds the result's own GraphModule from its graph, so that code is only
    ever thrown away; the graph edits themselves are untouched. str(gm) shows
    the last generated source -- ExecuTorch formats it into a debug message
    after every fusion, logged or not, so it must not trigger codegen.

    GraphModule gives every instance its own class, so patching type(gm)
    touches this module alone.
    """
    cls = type(gm)
    if "_esp32_codegen" in vars(cls):
        return
    generate = torch.fx.GraphModule.recompile

    def recompile(self):
        cls.forward = forward_after_codegen

    def forward_after_codegen(self, *args, **kwargs):
        generate(self)  # replaces cls.forward with the compiled one
        return self(*args, **kwargs)

    def code(self):
        if cls.forward is forward_after_codegen:
            generate(self)
        return self._code

    cls.recompile = recompile
    cls.code = property(code)
    cls._esp32_codegen = True


def _overload(node):
    """The OpOverload behind a call_function node (edge or ATen), or None."""
    import torch

    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        return target
    op = getattr(target, "_op", None)       # an EdgeOpOverload's OpOverload
    return op if isinstance(op, torch._ops.OpOverload) else None
