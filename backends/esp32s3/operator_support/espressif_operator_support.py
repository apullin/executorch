# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which nodes the partitioner may claim, and why it refuses the rest."""

import math
from collections import Counter

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase

from executorch.backends.esp32s3.operator_support.const_weights import (
    FOLDED,
    LoweringRefused,
    PassThroughWeight,
    _check_pass_through_matmul,
    _const_fqn,
    _is_build_time_const,
    _is_pass_through_const_path,
    _is_torchao_affine,
    _is_torchao_dequant,
    _pass_through_source,
    _pt2e_dequant_kind,
    _resolve_const,
    _shape_of,
    _slice_pass_through,
    _torchao_code_range,
    _unsplittable_constant,
    _w_off_member,
    _w_off_pattern,
)
from executorch.backends.esp32s3.operator_support import qk_operands
from executorch.backends.esp32s3.operator_support import qk_values
from executorch.backends.esp32s3.operator_support.patterns import (
    SUPPORTED,
    _act_admits_clamp,
    _bmm_is_linear,
    _from_promoted_checkpoint,
    _hard_pattern,
    _is_symbolic_shaped,
    _kcache_affine_bmm,
    _kcache_affine_member_bmm,
    _kcache_keys,
    _kcache_member_bmm,
    _qk_bmm_pattern,
    _relu2_pattern,
    _selected_mm_from_member,
    _selected_mm_pattern,
    _silu_pattern,
    _softcap_pattern,
    _tail_root_mm,
    _through_view,
    _unary_act,
    _weight_operand,
)
from executorch.backends.esp32s3.serialization.asym_records import _act_record
from executorch.backends.esp32s3.serialization.espressif_blob import (
    KCACHE_AFFINE_DOT_BOUND,
    MAX_DOT_LEN,
    VEC_BYTES,
)
from executorch.backends.esp32s3.serialization.weight_formats import (
    FORMATS,
    HALF_FORMATS,
)


# node.meta key: the gate claimed this op with a float-weight loop because its
# weight is one a recipe left as the model had it -- "fp32" for a plain float
# weight (Esp32OperatorSupport._keeps_float), "fp16"/"bf16" for a half
# checkpoint's (_keeps_half) -- so preprocess emits that loop whatever the
# format policy.
KEEP_FLOAT = "esp32_keep_float"
# The ops whose weight keeps its precision when a recipe left it float
# (Esp32OperatorSupport._keeps_float): a matmul in the delegate's fp32 loop,
# a convolution in its fp32 convolution (opcode 1 under FLAG_CONV), a gather
# portable at fp32 -- or, for a half checkpoint's weight, the fp16/bf16 rung
# of each, the gather included (_keeps_half).
_KEEP_FLOAT_OPS = frozenset({
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
    exir_ops.edge.aten.embedding.default,
    exir_ops.edge.aten.convolution.default,
})


def _gated_mul_pattern(node, support=None):
    """If `node` is ``mul(A, B)`` fusable as FLAG_MUL, return
    ``("pair", A, B, root_a, root_b)`` or ``("operand", tail, R, root)``;
    else None.

    pair      both factors are single-use tails of DIFFERENT matmuls: SwiGLU's
              silu(x@Wg) * (x@Wu), GeGLU, ReGLU. The later projection
              multiplies, in place, what the earlier one wrote.
    operand   one factor is such a tail and the other, R, is a runtime value
              this partitioner would not claim: an output gate's
              ``out * sigmoid(x@Wg)``, Glimmer's ``gelu(x@Wp) * ple[:, l]``.

    Shapes must match outright -- the runtime walks both operands row-major,
    with no broadcasting -- and R may not be a build-time constant (a lifted
    parameter cannot occupy the runtime operand slot; it is the residual
    gate's learned-offset case). `support` is the asking gate, whose probe
    answers "is R claimable" under the same policies, exactly as for
    `_residual_pattern`; the in-progress set breaks the same reentrancy.
    """
    MUL = exir_ops.edge.aten.mul.Tensor
    if node in _MUL_IN_PROGRESS:
        return None
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.target is MUL and len(node.args) == 2):
        return None
    a, b = node.args
    if not (isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)) \
            or a is b:
        return None
    av, bv = a.meta.get("val"), b.meta.get("val")
    if (av is None or bv is None or tuple(av.shape) != tuple(bv.shape)
            or av.dtype != torch.float32 or bv.dtype != torch.float32):
        return None

    def tail_root(t):
        return _tail_root_mm(t) if set(t.users) == {node} else None

    ra, rb = tail_root(a), tail_root(b)
    if ra is not None and rb is not None:
        return ("pair", a, b, ra, rb) if ra is not rb else None
    if ra is None and rb is None:
        return None
    t, R, root = (a, b, ra) if ra is not None else (b, a, rb)
    ep = support.exported_program if support is not None else None
    if ep is not None and _is_build_time_const(ep, R):
        return None
    _MUL_IN_PROGRESS.add(node)
    try:
        probe = (support.probe_twin() if support is not None
                 else Esp32OperatorSupport())
        if probe.is_node_supported(None, R):
            return None
    finally:
        _MUL_IN_PROGRESS.discard(node)
    return ("operand", t, R, root)


_MUL_IN_PROGRESS = set()


def _residual_pattern(node, allow_conv: bool = False, support=None):
    """If `node` is  add.Tensor(tail, R)  fusable as a residual, return (tail, R);
    else None.

    `support` is the gate asking, when there is one: its `probe_twin()` answers
    the is-R-claimable query below under the SAME policies. Preprocess passes
    nothing -- inside a partition subprogram R is a placeholder, and a default
    gate refuses those.

    Fusable means: alpha is 1, EXACTLY one operand is the single-use tail of a
    matmul chain, shapes match outright (the runtime walks both row-major, no
    broadcasting), and R is nothing this partitioner would claim. That last gate is
    load-bearing twice over: add(mm1, mm2) with both sides fusable is a JOIN that
    would weld two partitions, and in a residual CHAIN (transformers) R is the
    previous block's fused add -- claiming both merges every block into one
    partition through the residual stream. Refusing when R is claimable keeps every
    partition sound; along a chain the adds then fuse alternately, because a
    fused residual reads R as a partition input (the descriptor's residual word
    indexes the partition's inputs, never a chain slot).

    Reentrancy: the is-R-claimable query below can re-reach this very add through
    the view gate (is R a view whose sole user is this add?). The in-progress set
    makes the nested query answer None -- conservatively "not fusable" -- which
    resolves the fixpoint the right way around: R stays portable, the add fuses."""
    if node in _RESIDUAL_IN_PROGRESS:
        return None
    if not (node.op == "call_function"
            and node.target is exir_ops.edge.aten.add.Tensor
            and len(node.args) == 2):
        return None
    alpha = node.kwargs.get("alpha", 1)
    if alpha != 1:
        return None
    a, b = node.args
    cands = []
    for t, r in ((a, b), (b, a)):
        if (isinstance(t, torch.fx.Node) and set(t.users) == {node}
                and _tail_root_mm(t, allow_conv=allow_conv) is not None):
            cands.append((t, r))
    if len(cands) != 1:
        return None
    t, r = cands[0]
    if not isinstance(r, torch.fx.Node):
        return None
    tv, rv = t.meta.get("val"), r.meta.get("val")
    if tv is None or rv is None or tuple(tv.shape) != tuple(rv.shape):
        return None
    if not _row_major(r):
        return None          # the runtime reads R row-major, like the output
    _RESIDUAL_IN_PROGRESS.add(node)
    try:
        # Every policy, not just conv. This asks whether R is itself claimable
        # under the SAME settings as the gate that is asking; otherwise an R
        # that the real gate refuses (a conv under fp32, a promoted mm under
        # requantize 'none') looks claimable here and the fusion is lost, or
        # an R the real gate claims looks portable and welds two partitions.
        probe = (support.probe_twin() if support is not None
                 else Esp32OperatorSupport(conv=allow_conv))
        if probe.is_node_supported(None, r):
            return None
    finally:
        _RESIDUAL_IN_PROGRESS.discard(node)
    return (t, r)


_RESIDUAL_IN_PROGRESS = set()


def _row_major(node) -> bool:
    """Whether a runtime tensor's memory is plain row-major.

    It is the only layout the delegate reads or writes, and ExecuTorch keeps a
    tensor's dim order: an input exported channels-last or transposed, or a
    `_to_dim_order_copy` into channels-last, reaches the delegate in that
    physical order, where row-major indexing reads the wrong elements.
    lower_esp32 copies a channels-last graph input to row-major before
    partitioning (make_inputs_row_major_pass.py), so what still arrives here
    is a dim order the model sets itself, or a flow that skips lower_esp32.
    """
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return not isinstance(val, torch.Tensor) or val.is_contiguous()


def _fp32_qk_refusal(op):
    """Every q@kT op quantises its query (OP_BMM_DYN_I8 its key too) to int8
    rows, the backend's own quantisation, which format_policy='fp32' -- no
    int8 anywhere -- leaves out, quantize_qk or not."""
    return ("int8 q@kT under fp32 policy",
            f"format_policy='fp32' runs float loops with no activation "
            f"quantiser, and {op} quantises its query to int8 rows: the q@kT "
            f"stays portable on the model's float arithmetic, quantize_qk or "
            f"not")


def _layout_problem(nodes):
    """(kind, detail) when one of these runtime tensors is not row-major."""
    bad = [getattr(n, "name", "input") for n in nodes if not _row_major(n)]
    if not bad:
        return None
    return ("runtime tensor is not row-major",
            f"{', '.join(bad)}: the delegate reads and writes row-major "
            "only, and ExecuTorch keeps a tensor's dim order (channels-last, "
            "transposed); the op stays portable. lower_esp32 makes "
            "channels-last graph inputs row-major; a dim "
            "order set inside the model would need the gather to read "
            "through strides and the outputs to scatter back")


# Named because two places agree on it: the refusal below and the partitioner's
# dynamic-shape notice, which is a stated precondition rather than a bucket.
KIND_DYNAMIC = "dynamic shape"


def _carries_symbolic_dim(node) -> bool:
    return _is_symbolic_shaped(node) or any(
        _is_symbolic_shaped(a) for a in node.all_input_nodes)


# A fold (a weight's transpose, slice, dequantise or float-offset chain)
# whose claimed readers grouping put in different partitions: preprocess has
# no runtime form for it, so the fold and its readers stay portable
# (Esp32Partitioner._partition regroups without them).
KIND_FOLD_SPLIT = "folded operand read from two partitions"
# Its own name because it is the refusal that means "implemented, wrong
# spelling" rather than "not implemented". Counted separately so a coverage
# comparison between two architectures cannot silently attribute this to either.
KIND_BMM_LINEAR = "bmm is a Linear in disguise"
# What may be quantised AGAIN after arriving already quantised.
#
# fp8 and fp16 are not the same case, so one flag cannot serve both. Over
# 4096 normal samples:
#
#     fp8 e4m3   3 mantissa bits    141 distinct values   3.5e-02 max rel err
#     bf16       7                 1352                   2.2e-03
#     fp16      10                 3319                   2.7e-04
#
# fp8 is a QUANTISATION SCHEME -- 141 levels -- so int8 on top of it is a second
# lossy step the training never saw. fp16/bf16 are storage PRECISION, which is
# how nearly every checkpoint ships, and quantising one to int8 is the FIRST
# quantisation and the entire point of this backend. Refusing those would make it
# useless for most real models.
#
#   "half"  requantise fp16/bf16, leave fp8 alone. The default.
#   "all"   requantise everything, fp8 included. Kept because the measurement
#           (~0.4% output error on trained weights) says it is usually fine
#           and it buys the int8 kernel -- but it is opt-in, because "usually
#           fine" is not something to impose on a model someone trained at fp8.
#   "none"  leave anything pre-quantised alone; only ordinary fp32 weights are
#           quantised.
REQUANTIZE_POLICIES = ("half", "all", "none")
_HALF_SOURCES = ("torch.float16", "torch.bfloat16")

# Two conv kinds, not one: "you did not ask for it" and "this shape is not
# supported yet" are different actions for the reader.
KIND_CONV_OFF = "convolution, but conv delegation is off"
KIND_CONV_SHAPE = "convolution shape not supported yet"
# A recipe left this table float and the delegate has no fp32 loop for its
# op, a gather or a selected-row head (Esp32OperatorSupport._keeps_float): it
# stays portable at fp32.
KIND_KEPT_FLOAT = "weight a recipe left float"
# Under static_act, an op whose activation a dynamic recipe quantises at
# every call (Esp32OperatorSupport._static_act_dynamic_problem): it stays
# portable on the recipe's value.
KIND_STATIC_ACT_DYNAMIC = "static_act over a dynamic recipe"
# A q@kT whose operand the sample runs show on a grid
# (Esp32OperatorSupport._qk_value_problem, qk_values.py): the rules found no
# grid, the values did, and the bmm stays portable on the model's own values.
KIND_QK_VALUES = "q@kT operand on a grid by its values"
# The value check passed its memory or time budget (qk_values.MAX_BYTES,
# MAX_SECONDS) before it could judge: the q@kT stays portable -- exact is
# safe, and a grid not judged must not be admitted.
KIND_QK_VALUES_BUDGET = "q@kT value check past its budget"
# The value check could not class an operand's rows (its run with the
# written buffers' initial content drawn afresh raised): the q@kT stays
# portable, as past the budget -- a grid not judged must not be admitted.
KIND_QK_VALUES_UNJUDGED = "q@kT operand its values could not judge"
# An operand has rows that depend on state an earlier call wrote (replaying
# run 0's inputs on the state that enters run 1 changes them): from values
# such state cannot be told from a fresh value, so the q@kT stays portable
# -- speed only (a cache shifted a row a call, a cache written at a counter,
# a state the forward scales each call).
KIND_QK_VALUES_STATE = "q@kT operand reads state an earlier call wrote"
# The model quantises an op's activation itself under format_policy='fp32',
# whose loop has no activation quantiser
# (Esp32OperatorSupport._record_loop_problem).
_FP32_RECORD = ("activation quantiser under fp32 policy",
                "format_policy='fp32' runs the fp32 loop, which has no "
                "activation quantiser: the model's stays in the graph")


def _folding_skips():
    """The ops ExecuTorch's constant propagation never folds by default
    (constant_prop_pass's _DEFAULT_SKIP_TARGETS: aten.full, and PT2E's and
    torchao's quantize and dequantize primitives), which apply_passes runs
    it with. An ExecuTorch without that set is refused by name rather than
    guessed at: the gate's constant set has to be propagation's own, or an
    op over constants is claimed and its value quantised per row at run
    time, or a runtime op is refused."""
    try:
        from executorch.exir.passes.constant_prop_pass import _DEFAULT_SKIP_TARGETS
    except ImportError as error:
        raise ImportError(
            "Esp32 partitioner: executorch.exir.passes.constant_prop_pass "
            "has no _DEFAULT_SKIP_TARGETS, the ops constant propagation "
            "leaves (aten.full and the quantize/dequantize primitives in "
            "ExecuTorch 1.5 and 1.6). The gate decides which activations are "
            "constants by that set (_folded_constants): name this "
            "ExecuTorch's equivalent in _folding_skips") from error
    return set(_DEFAULT_SKIP_TARGETS)


def _folded_constants(ep) -> set:
    """The nodes constant propagation folds into constants (apply_passes'
    constant_prop_pass, by its own rules): the parameters, the buffers the
    forward does not write and the lifted constants, and each pure op whose
    inputs are all among them -- data movement (a rank-3 Linear's view of
    a parameter, a batch of queries expanded) and arithmetic alike (its
    abs, negation or scale) -- but not a quantize or dequantize primitive,
    nor aten.full, which it leaves: a dequantise of constant codes stays a
    pair, and anything computed from it a runtime value."""
    sig = ep.graph_signature
    written = set(sig.buffers_to_mutate.values())
    fixed = (set(sig.inputs_to_parameters)
             | set(sig.inputs_to_lifted_tensor_constants)
             | {n for n, fqn in sig.inputs_to_buffers.items()
                if fqn not in written})
    skips = _folding_skips()
    constant = set()
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in fixed:
                constant.add(node)
        elif (node.op == "call_function" and node.target not in skips
              and not node.is_impure()
              and all(a in constant for a in node.all_input_nodes)):
            constant.add(node)
    return constant


def _constant_source(node, constant):
    """A placeholder `node`, one of the constant set, is computed from (or
    the op with no tensor input it starts from, an arange)."""
    while node.op != "placeholder":
        inputs = [a for a in node.all_input_nodes if a in constant]
        if not inputs:
            break
        node = inputs[0]
    return node


def _written_buffer(sig, node) -> bool:
    """Whether placeholder `node` is a buffer the forward writes."""
    return sig.inputs_to_buffers.get(node.name) in sig.buffers_to_mutate.values()


def _reads_written_buffer(sig, node) -> bool:
    """Whether `node` is a buffer the forward writes, itself or through the
    transposes and slices preprocess folds (FOLDED): no constant, though
    _is_build_time_const counts every buffer. The partitioner never pulls
    one in, so an op reading it as a runtime operand reads it at run time:
    directly a partition input, through a fold that op left outside the
    partition (the FOLDED gate takes one only into a weight, bias or
    table)."""
    while isinstance(node, torch.fx.Node) and node.op == "call_function" \
            and node.target in FOLDED:
        node = node.args[0]
    return (isinstance(node, torch.fx.Node) and node.op == "placeholder"
            and _written_buffer(sig, node))


class Esp32OperatorSupport(OperatorSupportBase):
    def __init__(self, quantize_qk: bool = False, reporter=None,
                 conv: bool = True,
                 requantize_promoted: str = "half",
                 format_policy: str = "i8",
                 exported_program: ExportedProgram = None,
                 keep_float: bool = False,
                 overrides=(),
                 static_act: bool = False,
                 refused_folds=(),
                 example_inputs=None) -> None:
        # Opt-in, default OFF: int8 q@kT is an accuracy trade (the standard
        # attention-quantisation one), not a pure win, so a model author turns it
        # on rather than discovering it.
        self.quantize_qk = quantize_qk
        # DEFAULT ON: conv has device parity rows of its own -- conv2d/1d/3d,
        # depthwise, depthwise-1d, grouped g=2 and g=8, transposed, conv+relu,
        # conv+silu, and the i4/i16 rungs. It is NOT paired with quantize_qk,
        # and the difference matters: int8 q@kT is an accuracy TRADE a model
        # author should choose, while conv delegation is the same arithmetic the
        # portable kernel does, only without re-streaming the weights per output
        # row. There is no trade to opt into.
        self.conv = conv
        # Which weights that ARRIVED already quantised may be quantised AGAIN
        # (REQUANTIZE_POLICIES says why fp8 and fp16 do not share an answer).
        # The default 'half' requantises fp16/bf16 weights and keeps fp8
        # weights on their trained grid; 'all' requantises fp8 too (~0.4%
        # output error measured on trained weights, for the int8 kernel at
        # 10-30x the scalar fp32 loop); 'none' is for a model whose QAT was
        # done AT fp8 or fp16 and which the author does not want adapted a
        # second time to suit this backend. Refused ops run portable at fp32
        # on exactly the checkpoint's values -- and because a subgraph break
        # costs nothing on this part (no accelerator domain to leave, no
        # shared-memory round trip), that refusal is local. The rest of the
        # model still delegates.
        if requantize_promoted not in REQUANTIZE_POLICIES:
            raise ValueError(
                f"requantize_promoted must be one of {REQUANTIZE_POLICIES}, "
                f"got {requantize_promoted!r}")
        self.requantize_promoted = requantize_promoted
        valid_policies = set(FORMATS) | {"auto", "fp32"} | set(HALF_FORMATS)
        if format_policy not in valid_policies:
            raise ValueError(
                f"format_policy must be one of {sorted(valid_policies)}, "
                f"got {format_policy!r}")
        # Admission depends on storage policy. Under 'fp32' the delegate runs
        # matmuls in opcode 1's loop and convolutions in its fp32 convolution,
        # with no fp32 gather: claiming an embedding and discovering that in
        # preprocess would turn a local portable fallback into a hard
        # model-export failure.
        self.format_policy = format_policy
        # The graph alone cannot distinguish a lifted parameter placeholder
        # from a runtime input placeholder.  Preprocess can, because it has the
        # ExportedProgram; the gate keeps the same program, so eligibility and
        # preprocessing answer the same question and preprocess never meets the
        # difference after the gate has claimed the op.
        self.exported_program = exported_program
        # The program is recipe-quantised and the format policy is the
        # default (Esp32Partitioner): a weight the recipe left float keeps
        # its precision (_keeps_float), except one an explicit
        # format_overrides entry names (`overrides`).
        self.keep_float = keep_float
        self.overrides = frozenset(overrides)
        self.override_formats = (dict(overrides)
                                 if isinstance(overrides, dict) else {})
        # Esp32Partitioner(static_act=True): each quantised op's activation
        # scale is the one calibrate_static_act recorded (esp32_act_amax),
        # which preprocess refuses unless it is finite (_static_scale_problem).
        self.static_act = static_act
        # Folds (a transpose, slice, weight dequantise or float-offset root)
        # whose readers a previous grouping put in different partitions
        # (Esp32Partitioner._partition): refused, with their readers, so
        # that the next grouping cannot split them.
        self.refused_folds = frozenset(refused_folds)
        # The exported program's example inputs (Esp32Partitioner, from
        # lower_esp32), the first of the q@kT value check's input sets
        # (qk_values.py); None draws every set.
        self.example_inputs = example_inputs
        # Each q@kT bmm the value check kept portable, with its reason, and
        # each it could give no verdict on, with why, for the partition log
        # (Esp32Partitioner._partition).
        self.value_refusals = {}
        self.value_silent = {}
        # Every refusal below states a REASON, collected two ways: into
        # ExecuTorch's WhyNoPartitionReporter (the per-node table a user asks for
        # when their op did not delegate) and into a Counter of coarse kinds (the
        # one-glance summary the partitioner prints unconditionally).
        #
        # This is not decoration. A partition gate's failures present
        # identically -- "nothing delegated" -- whether the cause is a symbolic
        # dim, a consumer-verdict asymmetry, or the matcher simply missing; the
        # stated reason is what tells them apart.
        self.reporter = reporter
        self.kinds = Counter()
        # is_node_supported may be asked about the same node more than once
        # (the partitioner walks candidates, not just nodes). The reporter keeps
        # the FIRST reason per node and drops the rest; the counter has to match
        # that, or the summary reports more refusals than there are nodes.
        self._reported = set()
        # Nodes the gate accepted. Diffed against what actually got tagged, so
        # "supported but never partitioned" is visible.
        self.accepted = set()
        # The core ops and folded ops whose joint verdict _fold_problem is
        # deciding right now, each counted as claimed while it is open.
        self._deciding = set()
        # _folded_constants(exported_program), made on first use: the graph
        # as the gate first saw it. The passes that ask while they rewrite
        # (the chain strip, the static absorption) only point an op at
        # another runtime value, which leaves every node's constness as it
        # was; a node made since is a runtime value.
        self._constants = None
    @property
    def dynamic_refused(self) -> int:
        """Nodes refused for a symbolic dim. Its own accessor because the
        partitioner's dynamic-shape notice is a documented contract ("this
        backend claims static shapes only"), not one bucket among many -- but
        DERIVED from the counter rather than tallied separately, because a second
        tally is a second thing to get out of step on a re-visited node."""
        return self.kinds[KIND_DYNAMIC]

    @staticmethod
    def _dtype(node):
        if not isinstance(node, torch.fx.Node):
            return None
        val = node.meta.get("val")
        return getattr(val, "dtype", None)

    def _folded_constant(self, node) -> bool:
        """Whether constant propagation folds `node` into a constant
        (_folded_constants, made on first use)."""
        if self.exported_program is None or not isinstance(node, torch.fx.Node):
            return False
        if self._constants is None:
            self._constants = _folded_constants(self.exported_program)
        return node in self._constants

    def _runtime_operand_problem(self, operand, reads="its activation"):
        """Why `operand`, which the delegate reads at run time (`reads`), is
        not a runtime tensor, or None: a build-time constant.

        The partitioner pulls a claimed op's constant operands into its
        partition, a constant activation too, and preprocess would find the
        op with no runtime input for it and fail the whole export; an op over
        constants is constant propagation's to fold, as for an embedding's
        constant indices. A buffer the forward writes (an RNN's state) is no
        constant, however it is read: the partitioner never pulls one in
        (espressif_partitioner._partition), so read
        directly it is a partition input, and read through the transposes and
        slices preprocess folds (FOLDED) those stay outside the partition
        (the FOLDED gate takes one only into a weight, bias or table), as a
        view does; the delegate reads the state at run time, as it was
        before the forward writes it (ExecuTorch writes a buffer back after
        its last reader). As a weight, bias or table it is refused by
        _written_operand_problem."""
        ep = self.exported_program
        if ep is None:
            return None
        sig = ep.graph_signature
        root = source = operand
        if _is_build_time_const(ep, operand):
            from executorch.backends.esp32s3._passes.static_qdq import (
                constant_codes_source,
            )

            if constant_codes_source(operand, ep) is operand:
                # A PT2E dequantise of constant codes is a pair whose
                # quantize convert_pt2e folded: constant
                # propagation leaves the dequantise, which stays outside
                # the partition (the dequant gate takes one only into a
                # weight, bias or table), and the op carries the pair's
                # record -- read directly as behind a view. torchao's
                # dequantise is still refused: fold_portable_torchao folds
                # it into a constant.
                return None
        else:
            # Computed from constants alone, by data movement (a rank-3
            # Linear's view of a parameter) or any other pure op (its abs,
            # negation or scale), it is as constant as they are, and
            # constant propagation folds the op over it all the same. Not a
            # buffer the forward writes, which propagation leaves (nor is
            # the state read through a view: the view stays outside the
            # partition), nor a dequantise of constant codes, which it
            # skips: that op keeps the delegate and carries the pair's
            # record (static_qdq.constant_codes_source).
            if not self._folded_constant(operand):
                return None
            root = source = _constant_source(operand, self._constants)
        if _reads_written_buffer(sig, root):
            return None             # a runtime tensor, read where it is
        via = "" if source is operand else f" (computed from {source.name})"
        return ("activation is a build-time constant",
                f"{operand.name}{via} is a constant: the delegate reads "
                f"{reads} at run time; an op over constants is "
                "left for constant propagation to fold")

    def _written_operand_problem(self, operands):
        """Why one of a core op's constant `operands` ((role, node) pairs: its
        weight, bias or table) is no constant, or None: it reads a buffer the
        forward writes, itself or through the transposes, slices, dequantises
        and float-offset chains preprocess folds into it. The partition would
        pull that buffer in, preprocess would pack its value at build time,
        and every call would read the first one while the program updated its
        own copy (a fast-weight or memory state: y = x @ S.t(), then S
        updated)."""
        ep = self.exported_program
        if ep is None:
            return None
        sig = ep.graph_signature
        for role, operand in operands:
            if not (isinstance(operand, torch.fx.Node)
                    and _is_build_time_const(ep, operand)):
                continue
            stack, seen = [operand], set()
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                if n.op == "placeholder":
                    if _written_buffer(sig, n):
                        return ("activation is a buffer the model writes",
                                f"{n.name} is a state the forward writes: "
                                f"this op reads it as its {role}, which "
                                "preprocess would pack at build time, and "
                                "every call would read that first value")
                elif n.op == "call_function":
                    stack.extend(n.all_input_nodes)
        return None

    def _core_problem(self, node):
        """Return ``(kind, detail)`` when a runtime op cannot be emitted.

        This is the single source of truth used by the op gate and every
        epilogue/folded-node gate.  A target-name membership test is not enough:
        mm/addmm/embedding require build-time constants, addmm has scalar and
        bias-layout restrictions, and the runtime ABI is fp32 for activations.
        """
        if not (isinstance(node, torch.fx.Node)
                and node.op == "call_function"):
            return ("not a delegate core op", "not a call_function node")
        is_conv = node.target is exir_ops.edge.aten.convolution.default
        if node.target not in SUPPORTED and not is_conv:
            return ("not a delegate core op", str(node.target))

        if self.format_policy == "fp32":
            # mm/addmm in opcode 1's loop, a convolution in the fp32
            # convolution (opcode 1 under FLAG_CONV); no fp32 gather.
            if node.target is exir_ops.edge.aten.embedding.default:
                return (
                    "embedding unavailable under fp32 policy",
                    "format_policy='fp32' implements mm/addmm and "
                    "convolution only; this embedding stays portable")
            if (self.exported_program is not None
                    and _is_pass_through_const_path(_weight_operand(node))):
                return (
                    f"pass-through {'conv' if is_conv else 'matmul'} weight "
                    f"under fp32 policy",
                    "format_policy='fp32' stores float weights and cannot "
                    "consume already-quantised codes; this "
                    f"{'convolution' if is_conv else 'matmul'} stays "
                    "portable")
        # Learned query tokens through a Linear, a convolution over a learned
        # map (_runtime_operand_problem).
        if node.target is not exir_ops.edge.aten.embedding.default:
            problem = self._runtime_operand_problem(
                node.args[1 if node.target is exir_ops.edge.aten.addmm.default
                          else 0])
            if problem is not None:
                return problem
        keep = self._keeps_float(node)
        half = self._keeps_half(node)
        if keep and node.target is exir_ops.edge.aten.embedding.default:
            return (KIND_KEPT_FLOAT,
                    "a float table in a recipe-quantised program keeps its "
                    "precision: ExecuTorch's fp32 gather reads it exactly")

        # Before the selected-row branch, which returns: its table is
        # quantised here like any other weight. A half weight kept as it
        # shipped is not quantised again, whatever requantize_promoted says.
        if half is None and self._promoted_refusal(node):
            return (
                "weight arrived pre-quantised",
                f"requantize_promoted={self.requantize_promoted!r}, and this "
                f"op's weight arrived as {_from_promoted_checkpoint(node)} -- "
                "it stays portable on the checkpoint's own values")

        selected = _selected_mm_pattern(node, self.exported_program)
        if selected is not None:
            if self.format_policy not in ("i4", "i8", "auto"):
                return (
                    "selected-row head format unsupported",
                    "selected-row projection currently implements i4/i8 "
                    f"weights, not format_policy={self.format_policy!r}")
            if keep or half is not None:
                return (KIND_KEPT_FLOAT,
                        "a float table in a recipe-quantised program keeps "
                        "its precision, and the selected-row walk implements "
                        "i4/i8 tables: the head stays portable")
            rows_dtype = self._dtype(selected["rows"])
            if rows_dtype not in (None, torch.int64):
                return ("selected-row indices are not int64",
                        f"got {rows_dtype}")
            if selected["selected"] <= 0:
                return ("selected-row head is empty",
                        "the fixed selected width must be positive")
            if selected["selected"] > selected["vocab"]:
                # Legal index_select (IDs may repeat), but the runtime's walk
                # takes at most n IDs and init refuses the record otherwise.
                return ("selected-row head wider than its table",
                        f"{selected['selected']} row IDs over a "
                        f"{selected['vocab']}-row table; the runtime walks at "
                        "most as many IDs as the table has rows")
            bad = [(value.name, self._dtype(value))
                   for value in (selected["act"], node)
                   if self._dtype(value) not in (None, torch.float32)]
            if bad:
                return (
                    "runtime tensor dtype is not fp32",
                    ", ".join(f"{name}={dtype}" for name, dtype in bad))
            problem = _layout_problem(
                (selected["act"], selected["rows"], node))
            if problem is not None:
                return problem
            # The runtime walks a selected head from a boundary input to a
            # boundary output only (validate refuses scratch sentinels on it).
            # Preprocess raises on either chain direction, so refuse them here
            # where a refusal is a portable fallback instead of a dead export.
            act = selected["act"]
            root = _tail_root_mm(act, allow_conv=self.conv)
            if self._core_claimable(act) or (
                    root is not None and root is not node
                    and self._core_claimable(root)):
                return ("selected-row head chained from a claimed producer",
                        "the selected-row walk reads a partition input only; "
                        "chaining into it is unimplemented")
            if self._feeds_claimed_core(node):
                return ("selected-row head feeds a claimed consumer",
                        "the selected-row output must be a partition output; "
                        "chaining out of it is unimplemented")
            problem = (self._written_operand_problem(
                (("table", selected["table"]),))
                or self._fold_problem(node, (("table", selected["table"]),)))
            if problem is not None:
                return problem
            return (self._static_act_problem(node, selected["act"])
                    or self._lowering_problem(node))

        if node.target is exir_ops.edge.aten.addmm.default:
            beta = node.kwargs.get("beta", 1)
            alpha = node.kwargs.get("alpha", 1)
            if beta != 1 or alpha != 1:
                return (
                    "scaled addmm",
                    f"beta={beta} alpha={alpha}; the descriptor represents "
                    "bias + x@W only")

        if _carries_symbolic_dim(node):
            return (
                KIND_DYNAMIC,
                "this node or one of its inputs carries a symbolic dimension; "
                "the backend claims static shapes only")

        # All runtime arithmetic/output pointers are float*.  Constants may
        # arrive in another dtype because preprocessing converts their values.
        if is_conv:
            runtime_inputs = (node.args[0],)
        elif node.target is exir_ops.edge.aten.embedding.default:
            runtime_inputs = ()
        elif node.target is exir_ops.edge.aten.mm.default:
            runtime_inputs = (node.args[0],)
        else:
            runtime_inputs = (node.args[1],)
        bad = [(getattr(n, "name", "input"), self._dtype(n))
               for n in (*runtime_inputs, node)
               if self._dtype(n) not in (None, torch.float32)]
        if bad:
            return (
                "runtime tensor dtype is not fp32",
                ", ".join(f"{name}={dtype}" for name, dtype in bad))
        # The gather's int64 indices are read in memory order too.
        layout = runtime_inputs
        if node.target is exir_ops.edge.aten.embedding.default:
            layout = (node.args[1],)
        problem = _layout_problem((*layout, node))
        if problem is not None:
            return problem

        if self.exported_program is not None:
            problem = self._written_operand_problem(_constant_operands(node))
            if problem is not None:
                return problem
            shared = _unsplittable_constant(self.exported_program, node)
            if shared is not None:
                const, user = shared
                return (
                    "constant shared with a container argument",
                    f"{const.name} is also an operand of {user.name} inside a "
                    "list or tuple (a torch.cond branch, a cat); ExecuTorch "
                    "duplicates a shared constant only for top-level "
                    "arguments, so the partition cannot own a private copy")
            problem = self._fold_problem(node, _constant_operands(node))
            if problem is not None:
                return problem

        if is_conv:
            if not self.conv:
                return (KIND_CONV_OFF, "pass Esp32Partitioner(conv=True)")
            from executorch.backends.esp32s3.operator_support.conv import ConvRefused, admit
            try:
                admit(node)
            except ConvRefused as why:
                return (KIND_CONV_SHAPE, str(why))
            if self.exported_program is not None:
                weight_arg = node.args[1]
                if not _is_build_time_const(self.exported_program, weight_arg):
                    return ("conv weight is not constant",
                            "the runtime has no dynamic-weight conv path")
                if _is_pass_through_const_path(weight_arg):
                    # Codes consumed verbatim, exactly as for a Linear (a
                    # TRANSPOSED conv's turned into its dense equivalent's on
                    # the way in, _resolve_pt2e_dequant). torchao's
                    # block-quantised form is refused by name rather than
                    # discovered in preprocess: it has no per-output-channel
                    # meaning on a 4-D tensor.
                    if _pt2e_dequant_kind(_pass_through_source(
                            weight_arg)) not in ("per_channel", "per_tensor"):
                        return ("pass-through conv weight is not per-channel",
                                "only PT2E dequantize_per_channel or "
                                "dequantize_per_tensor codes reshape to the "
                                "conv packer's [N, K]")
                bias = node.args[2]
                if (isinstance(bias, torch.fx.Node)
                        and not _is_build_time_const(
                            self.exported_program, bias)):
                    return ("conv bias is not constant",
                            "the descriptor stores bias in the blob")
            if keep or half is not None:
                # The float convolution (_emit_fp32 / _emit_half), reading a
                # static pair's dequantised value as the matmul's loop does.
                node.meta[KEEP_FLOAT] = half or "fp32"
                node.meta.pop("esp32_act_quant", None)
                return self._lowering_problem(node)
            node.meta.pop(KEEP_FLOAT, None)
            return (self._static_act_problem(node, node.args[0])
                    or self._lowering_problem(node))

        if node.target is exir_ops.edge.aten.embedding.default:
            idx_dtype = self._dtype(node.args[1])
            if idx_dtype not in (None, torch.int64):
                return ("embedding indices are not int64",
                        f"got {idx_dtype}")
            if self.exported_program is not None:
                if not _is_build_time_const(
                        self.exported_program, node.args[0]):
                    return ("embedding table is not constant",
                            "the table must be packed at build time")
                # SigLIP/CLIP/BERT positions: Embedding(position_ids buffer).
                # Claiming it would pull the index buffer into the partition as
                # a constant, leaving the gather with no runtime input, and
                # preprocess would fail the whole export. A gather of constants
                # by constants is constant_prop's job, not a kernel's.
                # So for indices computed from constants alone (GPT-2's
                # positions, wpe(arange(T)); a view of the buffer), which
                # constant propagation folds the gather over all the same --
                # over a float table, which the delegate would quantise. A
                # pass-through table's codes are gathered verbatim, exactly,
                # and a stock recipe's dequantise has no kernel to run it
                # portable (stock_llm.check_claimed), so that gather stays.
                # Indices the forward writes (a token history) are no
                # constant: read at run time (_reads_written_buffer).
                indices = node.args[1]
                written = _reads_written_buffer(
                    self.exported_program.graph_signature, indices)
                if not written and (
                        _is_build_time_const(self.exported_program, indices)
                        or (not _is_pass_through_const_path(node.args[0])
                            and self._folded_constant(indices))):
                    return ("embedding indices are a build-time constant",
                            "the delegate gathers rows by runtime indices; a "
                            "constant lookup is left for constant propagation "
                            "to fold")
            if half is not None:
                node.meta[KEEP_FLOAT] = half      # the fp16/bf16 gather
            else:
                node.meta.pop(KEEP_FLOAT, None)
            return self._lowering_problem(node)

        weight_arg = (node.args[1] if node.target is exir_ops.edge.aten.mm.default
                      else node.args[2])
        if self.exported_program is not None:
            if not _is_build_time_const(self.exported_program, weight_arg):
                return ("matmul weight is not constant",
                        f"{getattr(weight_arg, 'name', weight_arg)} is a runtime input")
            if node.target is exir_ops.edge.aten.addmm.default:
                bias_arg = node.args[0]
                if not _is_build_time_const(
                        self.exported_program, bias_arg):
                    return ("addmm bias is not constant",
                            "the descriptor stores bias in the blob")
                n = int(node.meta["val"].shape[-1])
                bias_val = bias_arg.meta.get("val")
                bias_shape = tuple(bias_val.shape) if bias_val is not None else ()
                if bias_shape != (n,):
                    return ("addmm bias is not a vector",
                            f"expected ({n},), got {bias_shape}")
        act = (node.args[0] if node.target is exir_ops.edge.aten.mm.default
               else node.args[1])
        if keep or half is not None:
            # The fp32 loop (_emit_fp32), or its fp16/bf16 rung (_emit_half).
            # A static pair in front of it stays in the graph and the loop
            # reads its dequantised value, which is the model's own
            # arithmetic: no activation record.
            node.meta[KEEP_FLOAT] = half or "fp32"
            node.meta.pop("esp32_act_quant", None)
            return self._lowering_problem(node)
        node.meta.pop(KEEP_FLOAT, None)
        return (self._static_act_problem(node, act)
                or self._lowering_problem(node))

    def _keeps_float(self, node) -> bool:
        """Whether this core op's weight is one a recipe left float, which
        keeps the recipe's precision rather than the format policy's: the
        program is recipe-quantised (keep_float) and the weight reaches the
        op as a plain fp32 constant -- no dequantise or float-offset chain in
        front of it, no fp8/fp16 checkpoint upcast (requantize_promoted's
        question), no format_overrides entry. A matmul keeps it in the fp32
        loop and a convolution in the fp32 convolution; a gather or a
        selected-row head, which the delegate has no fp32 loop for, stays
        portable. Only the ops in _KEEP_FLOAT_OPS.

        Not an op that absorbed a dynamic activation chain, or was pointed
        past its static pair: that chain has left the graph, so only the
        quantised loop computes what the model does, and the weight is
        quantised under the format policy."""
        if (not self.keep_float or self.exported_program is None
                or node.target not in _KEEP_FLOAT_OPS):
            return False
        scheme = node.meta.get("esp32_act_quant")
        if (node.meta.get("esp32_act_per_tensor")
                or node.meta.get("esp32_act_absorbed")
                or (scheme is not None and scheme.chooser != "static")):
            return False
        weight = _weight_operand(node)
        if node.target is exir_ops.edge.aten.mm.default:
            selected = _selected_mm_pattern(node, self.exported_program)
            if selected is not None:
                weight = selected["table"]
        ep = self.exported_program
        return (weight is not None
                and not _is_pass_through_const_path(weight)
                and _is_build_time_const(ep, weight)
                and _from_promoted_checkpoint(node) is None
                and _const_fqn(ep, weight) not in self.overrides)

    def _keeps_half(self, node):
        """"fp16" or "bf16" when this core op's weight is a half-precision
        checkpoint's a recipe left alone: _keeps_float's conditions, but the
        weight reaches the op through promote_fp8_constants' upcast of an
        fp16/bf16 constant. It keeps its type -- the 2-byte rung of the op's
        loop, the gather included -- rather than the format policy's, under
        the floatw rule. None otherwise (a float model's half weight follows
        the format policy)."""
        if (not self.keep_float or self.exported_program is None
                or node.target not in _KEEP_FLOAT_OPS):
            return None
        source = _from_promoted_checkpoint(node)
        if source not in _HALF_SOURCES:
            return None
        scheme = node.meta.get("esp32_act_quant")
        if (node.meta.get("esp32_act_per_tensor")
                or node.meta.get("esp32_act_absorbed")
                or (scheme is not None and scheme.chooser != "static")):
            return None
        weight = _weight_operand(node)
        if node.target is exir_ops.edge.aten.mm.default:
            selected = _selected_mm_pattern(node, self.exported_program)
            if selected is not None:
                weight = selected["table"]
        ep = self.exported_program
        if (weight is None or _is_pass_through_const_path(weight)
                or not _is_build_time_const(ep, weight)
                or _const_fqn(ep, weight) in self.overrides):
            return None
        return "fp16" if source == "torch.float16" else "bf16"

    def _float_loop(self, node) -> bool:
        """Whether this core op runs a float-weight loop (fp32, or a stored
        fp16/bf16 weight), which applies no output quantisation and no fused
        multiply: a weight kept float or kept half, or one the policy or an
        override stores as fp16/bf16."""
        if self._keeps_float(node) or self._keeps_half(node) is not None:
            return True
        weight = _weight_operand(node)
        if (weight is None or self.exported_program is None
                or _is_pass_through_const_path(weight)):
            return False
        fmt = self.override_formats.get(
            _const_fqn(self.exported_program, weight), self.format_policy)
        return fmt in HALF_FORMATS

    def _quantised_problem(self, node):
        """_core_problem, for an op that is to quantise its activation itself:
        a static pair absorbed into it (static_qdq.absorb_static_qdq), a
        dynamic chain (strip_dynamic_act_qdq) or Stage B codes it reads. A
        weight kept float is the fp32 loop's, which reads the dequantised
        value instead; under format_policy='fp32' every op runs that loop,
        which has no activation quantiser (_emit_fp32)."""
        problem = self._core_problem(node)
        if problem is None and self.format_policy == "fp32":
            return _FP32_RECORD
        if problem is None and self._float_loop(node):
            return ("weight kept float reads fp32",
                    "a recipe left this op's weight float: the fp32 loop "
                    "reads the dequantised value, so the pair stays in the "
                    "graph")
        if problem is None:
            problem = self._record_loop_problem(node)
        return problem

    def _record_loop_problem(self, node):
        """Why the loop this op's weight runs cannot carry the model's own
        activation quantiser -- a static pair's scheme, a dynamic chain's, its
        folded per-tensor scale -- or None. The fp32 loop has no activation
        quantiser (_emit_fp32), nor has a weight stored as fp16 or bf16
        (_emit_half), and a weight quantised here at the i16 rung quantises
        its activation to int16 codes itself (BlobWriter.activation_scale):
        preprocess refuses a record on any of them. A weight a recipe left
        float never gets this far (the keep path in _core_problem reads the
        dequantised value). The quantiser stays in the graph and the op
        portable on the value it computes, the model's own arithmetic."""
        if self.format_policy == "fp32":
            return _FP32_RECORD
        weight = _weight_operand(node)
        if (weight is None or self.exported_program is None
                or _is_pass_through_const_path(weight)):
            return None
        fmt = self.override_formats.get(
            _const_fqn(self.exported_program, weight), self.format_policy)
        if fmt in HALF_FORMATS:
            return ("activation quantiser on a half-precision weight",
                    f"this weight is stored as {fmt}, whose float loop has no "
                    f"activation quantiser: the model's stays in the graph")
        if fmt in FORMATS and FORMATS[fmt].bits == 16:
            return ("activation quantiser on an i16 weight",
                    f"this weight is quantised here at {fmt}, whose loop "
                    f"quantises its activation to int16 codes itself: the "
                    f"model's stays in the graph")
        return None

    def _static_act_problem(self, node, act):
        """The model's static input pair on a claimable op (static_qdq.py):
        its scheme recorded as node.meta["esp32_act_quant"] -- the static
        EXT_ACT_QUANT, which requantises the pair's dequantised value
        to the pair's own codes -- else why the record cannot carry it, which
        leaves the op portable on the dequantised value: the model's own
        arithmetic, never a substitute. An op static_qdq.absorb_static_qdq
        already pointed past its pair carries its record from there. So does
        an op whose dynamic chain strip_dynamic_act_qdq computed in place in
        front of it, the chain's record refused here (its reason in
        node.meta["esp32_act_kept"]): the op reads the model's dequantised
        value, which the quantised loop would quantise again its own way."""
        from executorch.backends.esp32s3._passes.static_qdq import (
            act_scheme,
            constant_codes_source,
            per_channel_source,
            scheme_problem,
            static_source,
        )

        kept = node.meta.get("esp32_act_kept")
        if kept is not None:
            return kept
        if self.static_act:
            problem = self._static_act_dynamic_problem(node, act)
            if problem is not None:
                return problem
        if node.meta.get("esp32_act_absorbed"):
            # Pointed past its pair, whose scheme it carries: what it reads
            # now may be another pair's value (q2(d1(..)), a requantization),
            # which its own record quantises exactly as the model's q2 did.
            return None
        if per_channel_source(act) is not None:
            return ("per-channel activation quantiser",
                    "one scale per channel along K cannot leave the integer "
                    "dot; no stock default config emits one")
        from executorch.backends.esp32s3._passes.stage_b import IN, STAGE_B

        if isinstance(act, torch.fx.Node) and act.meta.get(STAGE_B) == IN:
            # Stage B: this op's own dequantise of codes (stage_b.py), whose
            # constants are the quantize's that wrote them -- the codes may
            # come from a max_pool or a cat of them, with no quantize behind.
            q = act
        else:
            # A dequantize of constant codes is a pair whose quantize
            # convert_pt2e folded: its record is the same, and without it the
            # op would quantise a value on the pair's grid its own way.
            q = static_source(act) or (
                constant_codes_source(act, self.exported_program)
                if self.exported_program is not None else None)
        if q is None:
            return None
        problem = scheme_problem(q)
        if problem is not None:
            return problem
        scheme = act_scheme(q)
        prior = node.meta.get("esp32_act_quant")
        if node.meta.get("esp32_act_per_tensor") or (
                prior is not None and prior != scheme):
            return ("two activation schemes",
                    f"{node.name} reads a static pair ({q.name}) and already "
                    f"carries another activation scheme")
        if scheme.code_bits == 16:
            problem = self._a16_weight_problem(node)
            if problem is not None:
                return problem
        problem = self._record_loop_problem(node)
        if problem is not None:
            return problem
        node.meta["esp32_act_quant"] = scheme
        return None

    def _static_act_dynamic_problem(self, node, act):
        """Under static_act, why an op whose activation a dynamic recipe
        quantises is not claimed, or None. static_act scales each claimed
        op's activation by one calibrated max|activation|
        (BlobWriter.static_scale); the recipe computes the scale from the
        activation at every call. The delegate runs one scheme per op, so
        claiming it would drop one of the two: quantising the recipe's
        dequantised value again with the calibrated scale is a substitute
        for the recipe's arithmetic, and taking the recipe's scheme ignores
        what static_act asked for. The op stays portable on the recipe's
        value, the converted model's own arithmetic.

        Two spellings reach here. lower_esp32's strip asks with the chain's
        record on the op (strip_dynamic_act_qdq._claim_problem), and then
        computes the chain in place in front of the refused op. static_act's
        two-step lowering (quantizer/calibration.py) partitions the to_edge
        graph, which runs no strip, so the chain itself is behind the
        activation; apply_passes computes it in place after partitioning
        (strip_dynamic_act_qdq_pass.compute_left_chains)."""
        from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
            dynamic_chooser,
        )

        scheme = node.meta.get("esp32_act_quant")
        if node.meta.get("esp32_act_per_tensor") or (
                scheme is not None and scheme.chooser != "static"):
            how = "its dynamic chain's scheme is recorded on it"
        else:
            chooser = dynamic_chooser(act)
            if chooser is None:
                return None
            how = f"{chooser.name} computes its scale"
        return (KIND_STATIC_ACT_DYNAMIC,
                f"static_act would scale {node.name}'s activation by one "
                f"calibrated max|activation|, but the recipe quantises it at "
                f"every call ({how}): the delegate cannot run both, so the op "
                f"stays portable on the recipe's value. Lower without "
                f"static_act (lower_esp32) to delegate it with the recipe's "
                f"own scheme")

    def _a16_weight_problem(self, node):
        """What the int16 loop cannot take about a 16-bit record's weight
        before preprocess sees it (_a16_problem checks the packed layout):
        int16 codes (a16w16; the affine weight reader refuses codes past int8
        storage under another name), or a weight quantised here at a policy
        that may not pick int8 rows."""
        from executorch.backends.esp32s3.quantizer.mirrors import code_bits

        weight = _weight_operand(node)
        if _is_pass_through_const_path(weight):
            declared = _declared_code_range(weight)
            if declared is not None and code_bits(*declared) == 16:
                return ("int16 weights with static int16 activations",
                        f"weight codes declared {list(declared)} under 16-bit "
                        f"activation codes (a16w16): the int16 loop dots int8 "
                        f"weight codes, and the backend's int16 rung takes no "
                        f"activation record")
            return None
        if self.format_policy != "i8":
            return ("16-bit activation codes on a weight quantised here",
                    f"format_policy={self.format_policy!r} may pick a rung "
                    f"other than int8 rows, the only weights the int16 loop "
                    f"dots")
        return None

    def _requant_problem(self, q):
        """Why static pair q -> d is not claimed as the output quantisation
        (EXT_OUT_QUANT) of the op whose fused epilogue q reads, or None.

        The fake-quant runs after the op's activation and clamp, which is the
        graph's order only when the activation comes BEFORE the q: the op's
        fused tail must be q's input and
        read by q alone. An activation after the pair reads the fake-quantised
        value, and nothing fuses onto d: it is not an epilogue node. The
        activation must be one the device computes as torch does -- a clamp,
        leaky ReLU, hardswish or hardsigmoid's decomposition, ReLU^2 -- not
        one it approximates (static_qdq.APPROXIMATE_ACTS), whose requantised
        codes could differ by more than a tie. A fused residual or multiply
        is refused (the runtime refuses FLAG_ADD/FLAG_MUL beside it); the
        eager graph of a residual between two static pairs adds dequantised
        values, which stays a portable add."""
        from executorch.backends.esp32s3._passes.static_qdq import requant_dq, scheme_problem

        if requant_dq(q) is None:
            return ("static pair shared",
                    "its quantize feeds more than its one dequantise")
        problem = scheme_problem(q)
        if problem is not None:
            return problem
        return self._code_out_tail_problem(q.args[0])

    def _code_out_problem(self, q):
        """Why the claimed op whose fused epilogue static quantize q reads
        cannot write q's int8 codes (Stage B's EXT_OUT_QUANT, out_dtype 1;
        stage_b.py), or None: _requant_problem's rules on the op and its
        epilogue, whatever reads q, and an op that dots 16-bit codes writes
        none (Stage B carries int8 codes only)."""
        from executorch.backends.esp32s3._passes.static_qdq import scheme_problem

        problem = scheme_problem(q) or self._code_out_tail_problem(q.args[0])
        if problem is not None:
            return problem
        root, _ = self._epilogue_chain(q.args[0])
        scheme = root.meta.get("esp32_act_quant")
        if scheme is not None and scheme.code_bits == 16:
            return ("int16 activations writing codes",
                    f"{root.name} quantises 16-bit codes; Stage B v1 carries "
                    f"int8 codes only")
        return None

    def _code_in_problem(self, d):
        """Why a Stage B dequantise (stage_b.py: one per consumer) cannot be
        claimed with the op that reads its codes, or None: its one reader
        must be a claimable matmul or conv taking it as its activation."""
        from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
            _activation_arg_index,
        )

        users = list(d.users)
        if len(users) != 1:
            return ("codes dequantised for several readers",
                    "a Stage B dequantise belongs to one consumer")
        user = users[0]
        idx = _activation_arg_index(user)
        if idx is None or idx >= len(user.args) or user.args[idx] is not d:
            return ("codes not an op's activation",
                    f"{user.name} does not read them as its activation")
        return self._quantised_problem(user)

    def _code_out_tail_problem(self, tail):
        """Why a claimed op cannot apply a static quantize reading `tail`
        after its epilogue, or None (_requant_problem; stage_b.py asks it of
        a quantize and of a table op)."""
        from executorch.backends.esp32s3._passes.static_qdq import APPROXIMATE_ACTS

        if not isinstance(tail, torch.fx.Node) or len(tail.users) != 1:
            return ("requantized value has other readers",
                    "the op would have to output its value both before and "
                    "after the requantization")
        if self.format_policy == "fp32":
            return ("requantize under fp32 policy",
                    "the fp32 diagnostic loop applies no output quantisation")
        root, act = self._epilogue_chain(tail)
        if root is None:
            return ("requantize not over a claimed op's epilogue", act)
        if self._float_loop(root):
            return ("requantize on a weight kept float",
                    "a recipe left this op's weight float, and the fp32 loop "
                    "applies no output quantisation: the pair stays portable")
        if (self.format_policy not in ("i4", "i8")
                and not _is_pass_through_const_path(_weight_operand(root))):
            return ("requantize on a weight quantised here",
                    f"format_policy={self.format_policy!r} may pick a rung "
                    f"wider than int8, whose loops apply no output "
                    f"quantisation")
        if act in APPROXIMATE_ACTS:
            return ("requantize after an approximated activation",
                    f"{act}: the device approximates it, so its requantised "
                    f"code could differ from eager's by more than a tie")
        return None

    def _epilogue_chain(self, tail):
        """(root op, activation) when `tail` ends a fused epilogue chain on a
        claimed matmul or conv -- the op itself, the linear trace's reshape
        over it, its fused activation or clamp chain -- else (None, why).
        Mirrors what the epilogue gates claim, read from the tail down."""
        from executorch.backends.esp32s3.operator_support.patterns import _clamp_bounds

        VIEW = exir_ops.edge.aten.view_copy.default
        if self._claimed_producer(tail):
            return tail, None
        if tail.op != "call_function":
            return None, "its input is not an op"
        if tail.target is VIEW:
            src = tail.args[0]
            if (isinstance(src, torch.fx.Node) and self._claimed_producer(src)
                    and len(src.users) == 1):
                return src, None
            return None, "a reshape not directly over a claimed op"
        act = _unary_act(tail)
        if act is not None:
            if not self._fusable_activation(tail):
                return None, f"its {act[0]} does not fuse"
            return _through_view(tail.args[0]), act[0]
        silu = _silu_pattern(tail, allow_conv=self.conv)
        if silu is not None and tail is silu[2]:
            return ((silu[0], "silu") if self._core_claimable(silu[0])
                    else (None, "its silu does not fuse"))
        hard = _hard_pattern(tail, self.exported_program)
        if hard is not None and tail is hard["div"]:
            return ((hard["mm"], hard["kind"])
                    if self._core_claimable(hard["mm"])
                    else (None, f"its {hard['kind']} does not fuse"))
        square = _relu2_pattern(tail)
        if square is not None and tail is square[2]:
            return ((square[0], "relu2") if self._core_claimable(square[0])
                    else (None, "its relu^2 does not fuse"))
        softcap = _softcap_pattern(tail, self.exported_program)
        if softcap is not None and tail is softcap[5]:
            return ((softcap[0], "softcap")
                    if self._core_claimable(softcap[0])
                    else (None, "its softcap does not fuse"))
        if _clamp_bounds(tail) is not None:
            # The clamp gate's walk: clamp links, then at most one activation
            # directly over the producer, or the producer through one view.
            a0 = tail.args[0]
            while (isinstance(a0, torch.fx.Node) and a0.op == "call_function"
                   and _clamp_bounds(a0) is not None and len(a0.users) == 1):
                a0 = a0.args[0]
            if (isinstance(a0, torch.fx.Node) and len(a0.users) == 1
                    and self._fusable_activation(a0, under_clamp=True)):
                below = _unary_act(a0)
                if below is not None:
                    return _through_view(a0.args[0]), below[0]
                return _silu_pattern(a0, allow_conv=self.conv)[0], "silu"
            root = _through_view(a0)
            if self._claimed_producer(root) and len(root.users) == 1:
                return root, "clamp"
            return None, "a clamp chain over no claimed op"
        if tail.target in (exir_ops.edge.aten.add.Tensor,
                           exir_ops.edge.aten.mul.Tensor):
            return None, ("a fused residual add or multiply: v1 refuses an "
                          "output quantisation beside FLAG_ADD/FLAG_MUL")
        return None, "not a claimed op's epilogue"

    def _lowering_problem(self, node):
        """Why preprocess would refuse this core op's quantised weight or
        bias, as (kind, detail), or None.

        The same checks preprocess runs (_check_lowerable), so anything the
        gate claims lowers: a refusal here leaves the op portable where a
        refusal in preprocess would fail the whole export (an Arm int32 bias,
        say). The partition sees parameters and buffers as FakeTensors;
        values are read where the real constant can be found (_concrete), and
        otherwise the weight is judged by structure alone, conservatively.
        Cached per node: the gate asks about one node many times."""
        if self.exported_program is None:
            return None
        cache = self.__dict__.setdefault("_lowering_cache", {})
        if node not in cache:
            try:
                _check_lowerable(self.exported_program, node)
                cache[node] = (self._dot_length_problem(node)
                               or self._static_scale_problem(node))
            except LoweringRefused as why:
                cache[node] = (why.kind, why.detail)
        return cache[node]

    def _dot_length_problem(self, node):
        """Why an int32 accumulator cannot hold this core op's dot exactly,
        as (kind, detail), or None: an int8 dot -- an i8 or i4 weight,
        quantised here (the i8, i4 and auto policies, or an override) or
        passed through as codes -- longer than MAX_DOT_LEN, which the
        runtime's init refuses (kMaxDotLen), failing the program's load. The
        fp32, fp16/bf16 and i16 loops accumulate in float or int64, and a
        gather has no dot. A convolution's dot is one group's C_in * taps."""
        if node.target is exir_ops.edge.aten.embedding.default:
            return None
        if self.format_policy == "fp32" or self._float_loop(node):
            return None
        ep = self.exported_program
        is_conv = node.target is exir_ops.edge.aten.convolution.default
        selected = None if is_conv else _selected_mm_pattern(node, ep)
        weight = (selected["table"] if selected is not None
                  else _weight_operand(node))
        if (weight is not None and not _is_pass_through_const_path(weight)
                and FORMATS.get(self.override_formats.get(
                    _const_fqn(ep, weight), self.format_policy),
                    FORMATS["i8"]).bits == 16):
            return None
        if is_conv:
            from executorch.backends.esp32s3.operator_support.conv import admit, split_groups

            k = max(sub.k for sub in split_groups(admit(node)))
        elif selected is not None:
            k = selected["k"]
        else:
            act = node.args[0 if node.target is exir_ops.edge.aten.mm.default
                            else 1]
            k = _shape_of(act)[-1]
        return _dot_length(k)

    def _static_scale_problem(self, node):
        """Why preprocess would refuse this op's calibrated static activation
        scale, as (kind, detail), or None: under static_act, an op whose
        int8 loop scales its activation by the calibrated max|activation|
        (BlobWriter.static_scale), which must be finite and non-negative --
        a calibration batch whose activation reaches inf or NaN records an
        inf or NaN one. Not a float loop or a gather, which take no
        activation scale; an op with no calibration at all is an
        export-order error preprocess reports itself."""
        if (not self.static_act
                or node.target is exir_ops.edge.aten.embedding.default
                or self.format_policy == "fp32" or self._float_loop(node)):
            return None
        amax = node.meta.get("esp32_act_amax")
        if amax is None or (math.isfinite(float(amax)) and float(amax) >= 0.0):
            return None
        return ("calibrated activation absmax not finite",
                f"{node.name}'s calibrated max|activation| is {float(amax)}: "
                f"a static int8 activation scale needs a finite one")

    def _core_claimable(self, node) -> bool:
        return self._core_problem(node) is None

    def _feeds_claimed_core(self, node) -> bool:
        """Whether a claimable core op reads `node`, or reads anything an
        epilogue fused onto `node` would produce.

        Walked structurally, through every epilogue-shaped user rather than
        only those the gates would claim: the epilogue gates ask whether
        `node` is claimable, so asking them back from here would recurse. The
        over-approximation only ever leaves a head portable. Checking direct
        users alone would miss selected-head -> relu -> Linear, where the relu
        fuses and its output lands on a scratch slot.
        """
        from executorch.backends.esp32s3.operator_support.patterns import _clamp_bounds

        EPILOGUE_SHAPED = (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.gelu.default,
            exir_ops.edge.aten.sigmoid.default,
            exir_ops.edge.aten.leaky_relu.default,
            exir_ops.edge.aten.pow.Tensor_Scalar,   # ReLU^2's square
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.tanh.default,
        )
        frontier, seen = [node], set()
        while frontier:
            n = frontier.pop()
            for u in n.users:
                if u in seen:
                    continue
                seen.add(u)
                if self._core_claimable(u):
                    return True
                if u.op == "call_function" and (
                        u.target in EPILOGUE_SHAPED
                        or _clamp_bounds(u) is not None):
                    frontier.append(u)
        return False

    def _qk_problem(self, node):
        """Why this bmm is not a q@kT the delegate computes, as (kind,
        detail), or None.

        Admission is a positive proof of each operand's origin
        (qk_operands.py): the operand the delegate quantises itself -- the
        query, and under OP_BMM_DYN_I8 the key -- is a plain float value
        under a declared mode with a mirror (quantize_qk; the int8 K caches),
        and a K cache's codes, scales and zero points are runtime tensors the
        delegate reads exactly as stored. Every other origin refuses by name
        and the bmm runs portable on the model's own values."""
        affine = _kcache_affine_bmm(node)
        got = affine[0] if affine is not None else _qk_bmm_pattern(node)
        if got is None:
            return ("bmm not q@kT",
                    "only the pre-softmax q@kT is claimable")
        A, _permute, X = got
        # strip_dynamic_act_qdq's reason for leaving an op on a chain's
        # dequantised value, which no quantised loop may quantise again.
        kept = node.meta.get("esp32_act_kept")
        if kept is not None:
            return kept
        # Nor one a pass recorded a recipe's activation scheme on (pointed
        # past its pair, or past a chain): the q@kT ops apply no record, so
        # the recipe's quantisation would be lost.
        records = [k for k in ("esp32_act_quant", "esp32_act_per_tensor",
                               "esp32_act_absorbed") if node.meta.get(k)]
        if records:
            return ("activation record on a q@kT",
                    f"{node.name} carries {', '.join(records)}: a recipe's "
                    f"scheme for an operand, which no q@kT op applies (each "
                    f"quantises its query its own way)")
        # The runtime's init refuses a zero batch, row count, key count or
        # width (validate_blob), which would fail the program's load; the
        # portable bmm computes the empty product.
        extents = [int(d) for d in (*A.meta["val"].shape,
                                    *X.meta["val"].shape[1:2])
                   if isinstance(d, int)]
        if 0 in extents:
            return ("empty q@kT",
                    f"{node.name} has a zero extent ({list(A.meta['val'].shape)}"
                    f" x {list(X.meta['val'].shape)}): the runtime refuses an "
                    f"empty op at load, and the portable bmm computes it")
        if affine is not None:
            return self._kcache_affine_problem(node, A, affine[1])
        bad = [(n.name, self._dtype(n)) for n in (A, X, node)
               if self._dtype(n) not in (None, torch.float32)]
        if bad:
            return ("runtime tensor dtype is not fp32",
                    ", ".join(f"{name}={dtype}" for name, dtype in bad))
        # The runtime reads both operands from partition inputs -- for an int8
        # K cache, the codes and scales its folded dequantisation reads (X
        # itself is claimed with the bmm, and asking about it would recurse
        # into the probe below).
        kcache = _kcache_keys(X)
        keys = (X,) if kcache is None else (kcache["codes"], kcache["scales"])
        ep = self.exported_program
        problem = qk_operands.operand_problem(ep, A, "query")
        if problem is not None:
            return problem
        if kcache is None:
            problem = qk_operands.operand_problem(ep, X, "key")
        else:
            problem = (qk_operands.kcache_part_problem(
                           ep, kcache["codes"], "K cache's codes")
                       or qk_operands.kcache_part_problem(
                           ep, kcache["scales"], "K cache's scales"))
        if problem is not None:
            return problem
        if self.format_policy == "fp32":
            return _fp32_qk_refusal("OP_BMM_KCACHE_I8" if kcache is not None
                                    else "OP_BMM_DYN_I8")
        if _is_symbolic_shaped(node) or any(
                _is_symbolic_shaped(a) for a in node.all_input_nodes):
            return (KIND_DYNAMIC,
                    "q@kT carries a symbolic dimension")
        if kcache is not None and any(
                _is_symbolic_shaped(kcache[k]) for k in ("codes", "scales")):
            return (KIND_DYNAMIC,
                    "the int8 K cache carries a symbolic dimension")
        problem = _dot_length(int(A.meta["val"].shape[-1]))
        if problem is not None:
            return problem
        # An operand the gate also claims -- an embedding's [B,M,K] output
        # needs no reshape to feed the bmm -- lands in the same partition, on
        # a scratch slot q@kT cannot read.
        probe = self.probe_twin()
        if any(probe.is_node_supported(None, n) for n in (A, *keys)):
            return ("q@kT operand produced inside the partition",
                    "q@kT reads both operands from partition inputs; "
                    "chaining into it is unimplemented")
        problem = _layout_problem((A, *keys, node))
        if problem is not None:
            return problem
        # Last, on a q@kT the rules would delegate: the operands the delegate
        # quantises itself, judged by their values on sample runs.
        return self._qk_value_problem(
            node, [(A, "query")] + ([(X, "key")] if kcache is None else []))

    def value_check(self):
        """This program's q@kT value check (qk_values.for_program): one set
        of sample runs, shared by every gate over the same graph."""
        return qk_values.for_program(self.exported_program,
                                     self.example_inputs)

    def _qk_value_problem(self, node, operands):
        """(kind, detail) keeping q@kT `node` portable when the sample runs
        show an operand the delegate would quantise itself on a grid, else
        None. Behind the rules, never instead of them: values never admit
        what the rules refuse, and an operand the runs cannot judge (no
        finite values, a program that cannot run) is left to the rules."""
        if self.exported_program is None:
            return None
        check = self.value_check()
        for operand, role in operands:
            got = check.grid(operand)
            why = check.state_read.get(operand.name)
            if why is not None:
                # Before any grid its values show: a state the forward
                # carries shows structure across runs whatever it holds.
                problem = (KIND_QK_VALUES_STATE,
                           f"{operand.name}, the {role}, reads state an "
                           f"earlier call wrote ({why}); the q@kT stays "
                           f"portable")
                self.value_refusals.setdefault(node, problem[1])
                return problem
            if check.over_budget is not None:
                problem = (KIND_QK_VALUES_BUDGET,
                           f"{operand.name}, the {role}, was not judged: the "
                           f"value check passed its budget "
                           f"({check.over_budget}), and a grid not judged is "
                           f"never admitted: the q@kT stays portable")
                self.value_refusals.setdefault(node, problem[1])
                return problem
            if got is None:
                why = check.unjudged.get(operand.name)
                if why is not None:
                    problem = (KIND_QK_VALUES_UNJUDGED,
                               f"{operand.name}, the {role}, was not judged: "
                               f"{why}; a grid not judged is never admitted: "
                               f"the q@kT stays portable")
                    self.value_refusals.setdefault(node, problem[1])
                    return problem
                v = check.verdict(operand)
                if check.failure is None and (v is None or v[0].grid is None):
                    why = check.why_silent(operand) if v is None else \
                        v[0].detail
                    self.value_silent.setdefault(node, []).append(
                        f"{operand.name}, the {role}: {why}")
                continue
            test, detail, regime, runs = got
            problem = (KIND_QK_VALUES,
                       f"{operand.name}, the {role}, is on a grid on {runs} "
                       f"sample runs ({regime}): {test}, {detail}; the model "
                       f"rounds the {role} itself, and the q@kT delegate "
                       f"would quantise that value again its own way")
            self.value_refusals.setdefault(node, problem[1])
            return problem
        return None

    def _kcache_affine_problem(self, node, A, parts):
        """Why the stock affine K cache's q@kT cannot be OP_BMM_KCACHE_AFFINE_I8:
        fp32 queries and scores, a plain float query (the delegate quantises
        it itself), the int8 codes, fp32 scales and int8 zero points read in
        place from partition inputs, static shapes, row-major, and key rows
        within the exact-dot bound (esp32_asym kcache-dot-bound)."""
        keys = (parts["codes"], parts["scales"], parts["zps"])
        # The query's origin first, so that a query the recipe quantised is
        # refused as that, whatever the options.
        ep = self.exported_program
        problem = qk_operands.operand_problem(ep, A, "query")
        if problem is not None:
            return problem
        if not self.quantize_qk:
            # Its int8 query rows are the backend's own quantisation: opt-in,
            # as for every q@kT. Without it the cache is read exactly
            # (dequantize_per_token_i8) and the q@kT runs portable in fp32.
            return ("affine K cache's int8 query is opt-in",
                    "OP_BMM_KCACHE_AFFINE_I8 quantises its query to int8 "
                    "rows, an accuracy trade every int8 q@kT leaves to "
                    "Esp32Partitioner(quantize_qk=True); the q@kT stays "
                    "portable on the cache's exact values")
        if self.format_policy == "fp32":
            return _fp32_qk_refusal("OP_BMM_KCACHE_AFFINE_I8")
        bad = [(n.name, self._dtype(n)) for n in (A, node)
               if self._dtype(n) not in (None, torch.float32)]
        if bad:
            return ("runtime tensor dtype is not fp32",
                    ", ".join(f"{name}={dtype}" for name, dtype in bad))
        for part, role in zip(keys, ("codes", "scales", "zero points")):
            problem = problem or qk_operands.kcache_part_problem(
                ep, part, f"K cache's {role}")
        if problem is not None:
            return problem
        if any(_is_symbolic_shaped(n) for n in (node, A, *keys)):
            return (KIND_DYNAMIC,
                    "the affine K cache's q@kT carries a symbolic dimension")
        k = int(parts["codes"].meta["val"].shape[2])
        if _pad_to(k, VEC_BYTES) > KCACHE_AFFINE_DOT_BOUND:
            return ("affine K-cache row past the exact-dot bound",
                    f"head width {k} pads past {KCACHE_AFFINE_DOT_BOUND}, where "
                    "an int32 dot of symmetric queries and zero-pointed codes "
                    "stops being exact")
        probe = self.probe_twin()
        if any(probe.is_node_supported(None, n) for n in (A, *keys)):
            return ("q@kT operand produced inside the partition",
                    "q@kT reads both operands from partition inputs; "
                    "chaining into it is unimplemented")
        problem = _layout_problem((A, *keys, node))
        if problem is not None:
            return problem
        return self._qk_value_problem(node, [(A, "query")])

    def _weight_path_reaches_core(self, node, seen=None) -> bool:
        """Every consumer is a folded/dequant chain ending in an eligible core.
        A consumer whose verdict _fold_problem is deciding counts as one."""
        if seen is None:
            seen = set()
        if node in seen or not node.users:
            return False
        seen = seen | {node}
        for user in node.users:
            if user in self._deciding or self._core_claimable(user):
                # Into a weight, bias or table only: preprocess folds the
                # chain into the constant it computes, and an op reading it
                # as its activation (a dequantise of constant codes, read
                # directly) reads it from outside the partition.
                if not _reads_as_constant(user, node):
                    return False
                continue
            if (isinstance(user, torch.fx.Node) and user.op == "call_function"
                    and (user.target in FOLDED or _is_torchao_dequant(user))
                    and self._weight_path_reaches_core(user, seen)):
                continue
            # A selected-row head gathers rows of this very table: the
            # gather is folded into its matvec, which claims the table.
            selected = (_selected_mm_from_member(user, self.exported_program)
                        if user.target is exir_ops.edge.aten.index_select.default
                        else None)
            if (selected is not None and selected["select"] is user
                    and selected["table"] is node
                    and (selected["mm"] in self._deciding
                         or self._core_claimable(selected["mm"]))):
                continue
            return False
        return True

    def _fold_problem(self, node, operands):
        """Why an op folded into one of this core op's constant `operands`
        ((role, node) pairs) would not be claimed with it, or None.

        A weight, bias or table may reach its op through ops on the constant:
        nn.Linear's transpose, a slice, a weight dequantise, a float-offset
        chain. Preprocess applies them at build time (_resolve_const), so the
        partition has to own every one, and each one's own gate claims it
        only when everything reading it is claimed (_weight_path_reaches_core,
        _fold_claimed). One read by anything else as well -- a graph output,
        a portable op -- stays portable: the delegate has no runtime form for
        it, and ExecuTorch gives a partition its own copy of a shared constant
        placeholder only, never of an op. This op, claimed anyway, would
        reach preprocess reading that operand as a runtime input and fail the
        whole export (a transposed parameter returned beside the matmul it
        feeds, for one); refused, it runs portable.

        The two gates ask each other -- the folded op whether its readers are
        claimable, a reader whether its folded ops are claimed -- so each op
        whose verdict is being decided counts as claimed while the question
        is open (self._deciding). Whatever the outer question decides, it
        decides for all of them; a folded op is asked about once however many
        readers share it."""
        if self.exported_program is None:
            return None
        added = node not in self._deciding
        self._deciding.add(node)
        try:
            folds = self._operand_folds(operands)
            for fold, role in folds.items():
                if self._split_fold(fold):
                    return (KIND_FOLD_SPLIT,
                            f"{fold.name} is applied to this op's {role} at "
                            "build time, and grouping put its readers in "
                            "different partitions, one of which would read it "
                            "as a runtime input; they stay portable "
                            "(lower_esp32 gives each reader its own copy "
                            "instead: split_shared_folds)")
                if fold in self._deciding:
                    continue
                self._deciding.add(fold)
                try:
                    claimed = self._fold_claimed(fold)
                finally:
                    self._deciding.discard(fold)
                if claimed:
                    continue
                others = [u.name for u in fold.users
                          if u is not node and u not in folds]
                return (
                    "folded operand shared with an unclaimed consumer",
                    f"{fold.name} is applied to this op's {role} at build "
                    f"time, and {', '.join(others) or 'another op'} also "
                    "reads it, which the delegate cannot hand a value to at "
                    "run time (ExecuTorch copies a shared constant into a "
                    "partition, never an op on one); the op stays portable")
        finally:
            if added:
                self._deciding.discard(node)
        return None

    def _operand_folds(self, operands):
        """{op: role}: the ops preprocess folds into the constant `operands`,
        every transpose, slice and weight dequantise down to the constants,
        and a float-offset chain as its root. A runtime operand has none:
        the gate refuses it as not constant."""
        folds = {}
        for role, operand in operands:
            if not (isinstance(operand, torch.fx.Node) and _is_build_time_const(
                    self.exported_program, operand)):
                continue
            stack = [operand]
            while stack:
                n = stack.pop()
                if n in folds or n.op != "call_function":
                    continue
                if n.target in FOLDED or _is_torchao_dequant(n):
                    folds[n] = role
                    stack.extend(n.all_input_nodes)
                elif _w_off_pattern(n) is not None:
                    folds[n] = role
        return folds

    def _split_fold(self, fold) -> bool:
        """Whether `fold`, or the float-offset chain it is a member of, is one
        a grouping split across partitions (refused_folds)."""
        if not self.refused_folds:
            return False
        if fold in self.refused_folds:
            return True
        woff = _w_off_member(fold)
        return woff is not None and woff["root"] in self.refused_folds

    def _fold_claimed(self, fold) -> bool:
        """What `fold`'s own gate in _supported decides: a transpose, a slice
        or a weight dequantise is claimed when everything reading it is; a
        float-offset chain when that gate recognises every member
        (_w_off_member) and everything reading its root is claimed."""
        if not (fold.target in FOLDED or _is_torchao_dequant(fold)):
            members = _w_off_pattern(fold)["members"]
            if any(_w_off_member(m) is None for m in members):
                return False
        return self._weight_path_reaches_core(fold)


    def probe_twin(self):
        """A reporter-free copy of this gate, for nested claimability queries.

        `_residual_pattern` asks "would the OTHER operand of this add be
        claimed?", which only a gate under the same policies can answer. A
        default-constructed gate (no exported program, `format_policy='i8'`,
        `requantize_promoted='half'`, `quantize_qk=False`) would disagree
        under any other policy -- a conv refused by the real gate under
        `fp32`, or a promoted mm refused under `'none'`, would look claimable
        to it and the residual fusion would be silently lost. The twin carries
        every policy and shares the exported program; it only drops the
        reporter and the refusal counter, so a nested query cannot pollute the
        per-node table.
        """
        twin = getattr(self, "_twin", None)
        if twin is None:
            twin = Esp32OperatorSupport(
                quantize_qk=self.quantize_qk, conv=self.conv,
                requantize_promoted=self.requantize_promoted,
                format_policy=self.format_policy,
                exported_program=self.exported_program,
                keep_float=self.keep_float, overrides=self.overrides,
                refused_folds=self.refused_folds,
                example_inputs=self.example_inputs)
            twin._twin = twin       # a probe's probe is itself
            self._twin = twin
        return twin

    def _residual_root_problem(self, root):
        """Why a residual cannot ride on `root`'s epilogue, or None.

        The runtime applies a fused residual inside `addmm_q_row_range`, and
        the selected-row walk (`addmm_q_selected_rows`) has no residual
        argument; `init` refuses `kFlagAdd` on a selected op. Fusing it
        anyway would let the export succeed and `load_method` fail.
        """
        if _selected_mm_pattern(root, self.exported_program) is not None:
            return ("residual on a selected-row head",
                    "the selected-row walk applies no residual; the runtime "
                    "refuses kFlagAdd on it at load, so the add stays portable")
        return None

    def _residual_add_problem(self, node):
        """Why the add `node` is not claimed as a residual epilogue, or None.

        The add gate's verdict, and the view gate's for the reshape between
        the matmul and this add: one function, because a view claimed for an
        add the gate refuses is a partition ending in a bare reshape.
        """
        got = _residual_pattern(node, allow_conv=self.conv, support=self)
        root = (_tail_root_mm(got[0], allow_conv=self.conv)
                if got is not None else None)
        if root is not None:
            problem = self._residual_root_problem(root)
            if problem is not None:
                return problem
        # A learned offset can have exactly the same shape as a flattened
        # matmul tail (Qwen3.5 spells its DeltaNet decay as
        # ``in_proj_a(x).reshape(-1) + dt_bias``).  It is not a runtime
        # residual: lifted parameters live inside the delegated program and
        # therefore cannot occupy the residual input slot the runtime
        # descriptor requires.  Keep the add portable while still claiming
        # the producer matmul.
        if (got is not None and self.exported_program is not None
                and _is_build_time_const(self.exported_program, got[1])):
            return ("add not a residual epilogue",
                    "the other operand is a build-time constant, not a runtime "
                    "partition input")
        if got is None or not self._core_claimable(root):
            return ("add not a residual epilogue",
                    "needs exactly one fusable side, an R that is not itself "
                    "claimable, equal shapes, a row-major R and alpha=1")
        # `_claimed_producer`, not `in SUPPORTED`: this asks whether the
        # add's output feeds something the delegate ALSO claims, and a conv
        # user is invisible to the membership test. Paired with rooting the
        # walk at a conv above -- enabling one without the other would let
        # an add feeding a conv fuse and weld two partitions.
        if any(self._claimed_producer(u) for u in node.users):
            return ("residual add feeds a claimed matmul or conv",
                    "fusing it would weld two partitions through the residual "
                    "stream")
        return None

    def _mul_problem(self, got):
        """Why a `_gated_mul_pattern` match cannot become FLAG_MUL, or None."""
        roots = got[3:5] if got[0] == "pair" else got[3:4]
        if self.format_policy == "fp32":
            return ("multiply unavailable under fp32 policy",
                    "the fp32 diagnostic loop accumulates into its output row, "
                    "where an in-place factor lives, so the mul stays portable "
                    "-- export with a quantised format policy to fuse it")
        for root in roots:
            problem = self._core_problem(root)
            if problem is not None:
                return ("mul factor's matmul is not claimed", problem[0])
            if self._float_loop(root):
                return ("multiply on a weight kept float",
                        "a recipe left this matmul's weight float, and the "
                        "fp32 loop accumulates into its output row, where an "
                        "in-place factor lives: the mul stays portable")
            if _selected_mm_pattern(root, self.exported_program) is not None:
                return ("multiply on a selected-row head",
                        "the selected-row walk applies no operand; the runtime "
                        "refuses kFlagMul on it at load")
        return None

    def _claimed_producer(self, node) -> bool:
        """Is `node` an op this partitioner claims and can fuse an epilogue on?

        Not `node.target in SUPPORTED`, which would silently exclude conv:
        conv is claimed by its own branch, not by membership in SUPPORTED, so
        every activation after a convolution would stay portable (14 ReLUs on a
        U-Net) although the runtime applies conv epilogues.

        Grouped convs are claimable producers for pointwise epilogues: preprocess
        mirrors the epilogue onto every disjoint output-channel descriptor.
        Residual fusion has its stricter `_conv_producer(grouped_ok=False)` gate
        because the serialized descriptor carries no residual channel offset.

        An embedding is claimable but is NOT a producer: a gather has no
        accumulator for an epilogue to run on, and preprocess records no
        fusion tail for it, so an activation claimed after one would fail the
        export with "producer was never claimed". It stays portable instead.
        """
        return (node.target is not exir_ops.edge.aten.embedding.default
                and self._core_claimable(node))

    def _fusable_activation(self, node, under_clamp: bool = False) -> bool:
        """A one-op activation (`_unary_act`), or a SiLU diamond's mul, that
        fuses onto a claimed producer. The activation gate's own test, and --
        `under_clamp` -- the clamp gate's test for the one activation the
        runtime applies before a clamp, which leaves out leaky ReLU (see
        `_act_admits_clamp`)."""
        if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
            return False
        if _unary_act(node) is not None:
            if under_clamp and not _act_admits_clamp(node):
                return False
            a0 = _through_view(node.args[0])
            return self._claimed_producer(a0) and len(a0.users) == 1
        pat = _silu_pattern(node, allow_conv=self.conv)
        return (pat is not None and node is pat[2]
                and self._core_claimable(pat[0]))

    def _promoted_refusal(self, node) -> bool:
        """Would this op be refused for consuming a pre-quantised weight?

        A predicate rather than an inline test because TWO gates need it. The
        FOLDED gate claims a permute when its users are claimable, and
        `u.target in SUPPORTED` stays True for an mm this refuses: decided that
        way, the permute would be claimed while its consumer is not, and
        preprocess would be handed a partition with a transpose and nothing to
        transpose for. Every refusal has to be visible everywhere a gate asks
        "will this op be claimed".
        """
        if not (isinstance(node, torch.fx.Node)
                and node.op == "call_function"):
            return False
        src = _from_promoted_checkpoint(node)
        if src is None:
            return False
        if self.requantize_promoted == "all":
            return False
        if self.requantize_promoted == "none":
            return True
        return src not in _HALF_SOURCES        # "half": fp8 is left alone

    def _no(self, node: torch.fx.Node, kind: str, detail: str = "") -> bool:
        """Record why `node` was refused, and return False so callers can
        `return self._no(...)` in place of a bare `return False`."""
        if node in self._reported:
            return False
        self._reported.add(node)
        self.kinds[kind] += 1
        if self.reporter is not None:
            self.reporter.report_reject(node, f"{kind}{': ' + detail if detail else ''}")
        return False

    def _gate(self, node: torch.fx.Node, ok: bool, kind: str,
              detail: str = "") -> bool:
        """`ok`, reporting `kind` when it is False. For the sites whose verdict is
        one boolean expression, so adding a reason does not restructure them."""
        return True if ok else self._no(node, kind, detail)

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        """The gate, plus a record of every node it said YES to.

        The verdict is only half the story. A node this returns True for can
        still fail to reach any partition, because grouping runs afterwards and
        makes its own decisions -- `partition()` reports that outcome by
        diffing these two sets. It is the more misleading of the two failures:
        every refusal reason reads clean while the model quietly loses most of
        its delegated matmuls.
        """
        ok = self._supported(submodules, node)
        if ok:
            self.accepted.add(node)
        return ok

    def _supported(self, submodules, node: torch.fx.Node) -> bool:
        # Not reported: placeholders, outputs and get_attr are not ops a user
        # could have expected to delegate, and listing them buries the ops that
        # were.
        if node.op != "call_function":
            return False
        # Persistent int8 K cache: the dequantisation feeding a q@kT folds into
        # OP_BMM_KCACHE_I8, so its cast, scale view and multiply are claimed
        # exactly when that bmm is. Ahead of every other gate: the view and the
        # multiply would otherwise be refused as a stray reshape and an
        # unsupported op, leaving the bmm to quantise materialised fp32 keys.
        if self.quantize_qk:
            kcache_bmm = _kcache_member_bmm(node)
            if kcache_bmm is not None:
                problem = self._qk_problem(kcache_bmm)
                return True if problem is None else self._no(node, *problem)
        # ExecuTorch's stock affine K cache (stock_llm.rewrite_kcache): the bmm,
        # its folded transpose and the dequantisation feeding it are one
        # OP_BMM_KCACHE_AFFINE_I8, claimed together -- the model itself
        # quantised the keys, and the delegate reads them as stored -- under
        # quantize_qk, since the delegate quantises the query to int8 rows
        # (_kcache_affine_problem).
        affine_bmm = (node if _kcache_affine_bmm(node) is not None
                      else _kcache_affine_member_bmm(node))
        if affine_bmm is not None:
            problem = self._qk_problem(affine_bmm)
            return True if problem is None else self._no(node, *problem)
        # Core ops have one eligibility predicate shared with every fusion and
        # folded-weight gate.  This prevents a refused producer from donating a
        # claimed ReLU/SiLU/view, which otherwise hands preprocess a partition
        # containing an epilogue and no op to attach it to.
        is_core = (node.target in SUPPORTED
                   or node.target is exir_ops.edge.aten.convolution.default)
        if is_core:
            problem = self._core_problem(node)
            if problem is not None:
                return self._no(node, *problem)
            return True

        # Static shapes are also a precondition for non-core patterns such as
        # q@kT and their folded transpose.
        if _carries_symbolic_dim(node):
            return self._no(
                node, KIND_DYNAMIC,
                "this node or one of its inputs carries a symbolic dimension; "
                "the backend claims static shapes only, and K in particular is "
                "what the weight packing is padded to")
        # Opt-in int8 q@kT: only the PRE-SOFTMAX bmm qualifies; p@v fails the
        # softmax-consumer test and stays portable, deliberately -- post-softmax
        # rows have absmax ~1, so int8 there cuts precision with no bandwidth
        # story behind it.
        if self.quantize_qk and node.target is exir_ops.edge.aten.bmm.default:
            problem = self._qk_problem(node)
            return True if problem is None else self._no(node, *problem)
        if (self.quantize_qk
                and node.target is exir_ops.edge.aten.permute_copy.default
                and len(node.users) == 1):
            u = next(iter(node.users))
            if isinstance(u, torch.fx.Node) and u.op == "call_function":
                got = _qk_bmm_pattern(u)
                # The consumer's verdict, not this node's: a permute of a
                # fixed-length KEY is fully static even when the QUERY carries a
                # dynamic prompt length, so without this the permute is claimed
                # while its bmm is refused -- a partition with no matmul in it,
                # which preprocess rejects and which kills the export outright
                # rather than leaving the subgraph portable as intended.
                if (got is not None and got[1] is node
                        and self._qk_problem(u) is None):
                    return True
        # Dynamic selected output head: the gather and transpose are folded
        # into the consuming quantised matvec.  This is deliberately separate
        # from the ordinary constant-weight FOLDED gate below: the gathered
        # tensor is dynamic, but its backing table is still a build-time
        # constant and the row IDs become a second delegate input.
        selected = _selected_mm_from_member(node, self.exported_program)
        if (selected is not None and node in
                (selected["select"], selected["permute"])):
            problem = self._core_problem(selected["mm"])
            return True if problem is None else self._no(node, *problem)
        # Stage B (stage_b.py): a producer's quantize, claimed with the op that
        # writes its codes, and a consumer's own dequantise of codes, claimed
        # with the op that reads them. Ahead of the pair branch below, whose
        # rules are Stage A's.
        from executorch.backends.esp32s3._passes.stage_b import IN, OUT, STAGE_B

        stage = node.meta.get(STAGE_B)
        if stage == OUT:
            problem = self._code_out_problem(node)
            return True if problem is None else self._no(node, *problem)
        if stage == IN:
            problem = self._code_in_problem(node)
            return True if problem is None else self._no(node, *problem)
        # A static pair after a claimed op's fused epilogue: that op's output
        # quantisation (EXT_OUT_QUANT), claimed with it -- the quantize and its
        # one dequantise, whose value the partition then outputs. Ahead of the
        # weight-dequant branch below, which would take the dequantise for a
        # weight's.
        from executorch.backends.esp32s3._passes.static_qdq import is_static_q, static_pair

        requant = node if is_static_q(node) else static_pair(node)
        if requant is not None:
            problem = self._requant_problem(requant)
            return True if problem is None else self._no(node, *problem)
        # A weight-side dequant -- constant codes in, consumed by an op we claim -- is
        # absorbed into the delegate as a pass-through weight. The placeholder test is
        # the cheap proxy for "constant" available without the ExportedProgram; a
        # non-constant placeholder reaching preprocess fails there with a clear error.
        # Activation-side dequants take a quantize_affine NODE as input, so they never
        # match -- and after strip_dynamic_act_qdq none remain anyway.
        if self.refused_folds and self._split_fold(node):
            return self._no(
                node, KIND_FOLD_SPLIT,
                "grouping put this fold's readers in different partitions; "
                "it stays portable, and so do they")
        if _is_torchao_dequant(node):
            w = node.args[0]
            const_ok = isinstance(w, torch.fx.Node) and w.op == "placeholder"
            if self.exported_program is not None:
                const_ok = _is_build_time_const(self.exported_program, node)
            return self._gate(
                node,
                const_ok and self._weight_path_reaches_core(node),
                "dequant not absorbable",
                "codes are not a constant placeholder, or the op consuming this "
                "dequant was itself refused (claiming it alone would leave a "
                "partition with no matmul in it)")
        # A float-offset weight (tinygemm/HQQ, GGUF), traced as elementwise
        # ops over constants: every node of the chain is claimed exactly when
        # the dequant op it stands for would be -- constant leaves, and a
        # consumer the gate claims.
        woff = _w_off_member(node)
        if woff is not None:
            root = woff["root"]
            const_ok = (self.exported_program is not None
                        and _is_build_time_const(self.exported_program, root))
            return self._gate(
                node,
                const_ok and self._weight_path_reaches_core(root),
                "float-offset weight not absorbable",
                "its codes, scales or offsets are not constants, or the op "
                "consuming the weight was itself refused")
        # relu fed by a claimed matmul that has no other consumer is absorbed: the
        # runtime clamps between accumulate and store. Gated on single-use because a
        # matmul whose pre-activation output is ALSO consumed elsewhere must remain a
        # partition output in its own right.
        # SiLU diamond: claim all corners (and the via-view, below) or none.
        # SiLU on a conv producer. Claiming it merges more ops per partition:
        # YOLO's C2f block then wants three live intermediates, which
        # NUM_SCRATCH = 3 holds (with two scratch slots preprocess would refuse
        # the partition and fail the whole export).
        pat = _silu_pattern(node, allow_conv=self.conv)
        if (pat is not None and node in (pat[1], pat[2])
                and self._core_claimable(pat[0])):
            # A diamond whose mul feeds another claimable matmul is claimed too:
            # with intra-partition dataflow the merged partition is exactly what
            # we want -- the intermediate rides a scratch slot or its own output
            # arg, never a portable round trip.
            return True
        softcap = _softcap_pattern(node, self.exported_program)
        if softcap is not None and node in softcap[1:6] and softcap[7] is None:
            return self._no(
                node, "softcap scalar not visible while partitioning",
                "its scale or cap is a parameter, which ExecuTorch hands the "
                "partitioner as a fake tensor, so the value the descriptor "
                "would carry cannot be checked; the softcap stays portable")
        if (softcap is not None and node in softcap[1:6]
                and self._core_claimable(softcap[0])):
            return True
        # hardswish / hardsigmoid, as torch decomposes them: claim every corner
        # (and the copies feeding its two constants) or none. Before the
        # residual and clamp gates, which would otherwise each refuse a corner.
        hard = _hard_pattern(node, self.exported_program)
        if hard is not None and (node in hard["nodes"]
                                 or node in hard["feeders"]):
            return self._gate(
                node, self._core_claimable(hard["mm"]),
                f"{hard['kind']} producer not claimable",
                "the activation runs in the matmul's epilogue, so its "
                "producer must be a claimed matmul")
        # ReLU^2: the square over a relu the clamp gate already claims.
        square = _relu2_pattern(node)
        if square is not None and node is square[2]:
            return self._gate(
                node, self._core_claimable(square[0]),
                "relu^2 producer not claimable",
                "the square runs in the matmul's epilogue, so its producer "
                "must be a claimed matmul")
        # Gated multiply: SwiGLU/GeGLU's gate*up and output gates, as FLAG_MUL.
        if node.target is exir_ops.edge.aten.mul.Tensor:
            got = _gated_mul_pattern(node, support=self)
            if got is not None:
                problem = self._mul_problem(got)
                return True if problem is None else self._no(node, *problem)
        # Residual add: claimed when it fuses into a matmul chain's epilogue. The
        # pattern itself (exactly one fusable side, R not claimable, shapes equal)
        # is _residual_pattern's job; here only the standard no-merge gate is added
        # on the add's OWN output.
        if node.target is exir_ops.edge.aten.add.Tensor:
            problem = self._residual_add_problem(node)
            return True if problem is None else self._no(node, *problem)
        # The reshape between a matmul and its fused consumers: claim it so the
        # pattern's nodes land in one partition. Numel-preserving by definition, it
        # costs the runtime nothing -- the delegate writes M*N floats either way.
        if node.target is exir_ops.edge.aten.view_copy.default:
            if (softcap is not None and node is softcap[1]
                    and self._core_claimable(softcap[0])):
                return True
            if (pat is not None and pat[3] is node
                    and self._core_claimable(pat[0])):
                return True
            if (hard is not None and hard["P"] is node
                    and self._core_claimable(hard["mm"])):
                return True
            src = node.args[0]
            # `_claimed_producer`, not `in SUPPORTED`. With the plain membership
            # test a conv here would not be a clean refusal but a CRASH: the
            # clamp gate below accepts a conv producer, so `relu` would be
            # claimed while the view feeding it is not, and preprocess would
            # walk from the clamp to an unclaimed producer -- "gate and walk
            # disagree". `conv -> reshape -> relu` is an ordinary thing to write.
            if (self._claimed_producer(src) and len(src.users) == 1
                    and len(node.users) == 1):
                from executorch.backends.esp32s3.operator_support.patterns import _clamp_bounds

                u = next(iter(node.users))
                # The consumer's verdict, not a copy of part of it: a view
                # claimed for a consumer the gate then refuses ends its
                # partition in a bare reshape, which preprocess cannot emit
                # ("<mm> is not a partition output but is consumed by <view>").
                # _supported refuses a non-core node carrying a symbolic dim
                # before any fusion branch, and an add's or mul's other
                # operand can carry one while this view does not.
                fusable_u = (u.op == "call_function"
                             and not _carries_symbolic_dim(u))
                if fusable_u and (
                        _unary_act(u) is not None
                        or _clamp_bounds(u) is not None):
                    return True
                if (fusable_u
                        and u.target is exir_ops.edge.aten.mul.Tensor):
                    got = _gated_mul_pattern(u, support=self)
                    if (got is not None and node in got[1:3]
                            and self._mul_problem(got) is None):
                        return True
                if (fusable_u
                        and u.target is exir_ops.edge.aten.add.Tensor):
                    got = _residual_pattern(u, allow_conv=self.conv,
                                            support=self)
                    if (got is not None and got[0] is node
                            and self._residual_add_problem(u) is None):
                        return True
                from executorch.backends.esp32s3._passes.stage_b import OUT, STAGE_B
                from executorch.backends.esp32s3._passes.static_qdq import is_static_q

                if is_static_q(u) and (
                        self._requant_problem(u) is None
                        or (u.meta.get(STAGE_B) == OUT
                            and self._code_out_problem(u) is None)):
                    # The op's output quantisation, after it -- Stage A's
                    # fake-quant, or Stage B's codes.
                    return True
            return self._no(
                node, "view not between a matmul and a fused consumer",
                "a view is only claimed to keep a fusion pattern's nodes in one "
                "partition; standing alone it would make the delegate owe a "
                "reshape it has no code for")
        # One-op activations -- GELU (exact or tanh), sigmoid, leaky ReLU --
        # behave like the clamp family: single-use producer.
        act = _unary_act(node)
        if act is not None:
            name = "gelu" if act[0].startswith("gelu") else act[0]
            return self._gate(
                node, self._fusable_activation(node),
                f"{name} producer not a single-use claimed matmul or conv",
                "the runtime applies the activation between accumulate and "
                "store, so it must own the value being activated")

        from executorch.backends.esp32s3.operator_support.patterns import _clamp_bounds

        if _clamp_bounds(node) is not None:
            # Walk down through single-use clamp-family links to the producer; the
            # whole chain is claimable iff it bottoms out in a claimed matmul. This
            # is how ACCIDENTAL chains (relu into a defensive clamp, relu6 after
            # relu) ride into the delegate as one composed interval.
            a0 = node.args[0]
            while (isinstance(a0, torch.fx.Node) and a0.op == "call_function"
                   and _clamp_bounds(a0) is not None and len(a0.users) == 1):
                a0 = a0.args[0]
            # Under the clamps: one fused activation, which the runtime applies
            # before the clamp (so relu(silu(x@W)) fuses whole), or the
            # producer itself through at most the one reshape the view gate
            # claims. Hopping a reshape BETWEEN clamp links would claim the
            # outer clamp of relu(view(relu(x@W))) while the view stays
            # portable, and preprocess would fail the export.
            if (isinstance(a0, torch.fx.Node) and len(a0.users) == 1
                    and self._fusable_activation(a0, under_clamp=True)):
                return True
            a0 = _through_view(a0)
            # fc2(relu(fc1(x))) is claimed whole: claiming the relu merges two
            # matmul partitions, and intra-partition dataflow routes the value
            # between them -- the merged partition emits fc1+relu as one op whose
            # output rides a scratch slot (or its own output arg) straight into
            # fc2's quantiser.
            return self._gate(
                node,
                (self._claimed_producer(a0) and len(a0.users) == 1),
                "clamp chain does not bottom out in a claimed matmul or conv",
                "same reason as gelu: the clamp interval is applied in the "
                "delegate's epilogue, which requires owning the producer")
        # Take a foldable op only on a BUILD-TIME CONSTANT path feeding an op we
        # claim.  Consumer structure alone is insufficient: Mamba's ordinary
        # `conv1d(x.transpose(1, 2))` exports the activation transpose as this
        # same permute_copy spelling.  Claiming it would weld the permute into
        # the conv partition, where preprocess quite correctly has no runtime
        # transpose implementation, and fail the entire export instead of
        # leaving the transpose portable.  Weight transposes remain folded;
        # dynamic Q@K's key transpose is admitted by its dedicated gate above.
        if node.target in FOLDED:
            const_path = (self.exported_program is not None
                          and _is_build_time_const(self.exported_program, node))
            return self._gate(
                node,
                const_path and self._weight_path_reaches_core(node),
                "foldable op has an unclaimed consumer",
                "only a build-time constant weight transpose is absorbed into "
                "the op it feeds; an activation transpose stays portable")
        # A Linear that lowered to a batch-1 bmm, in a graph partitioned without
        # canonical_passes (BmmToMm rewrites it as mm). Refused, but named and
        # counted, because it is the one refusal here that means
        # "this backend could take it and does not" rather than "this backend
        # has no kernel for it" -- and the two are indistinguishable in the
        # generic bucket, where a model's lost projections look like nothing.
        if _bmm_is_linear(node):
            av = node.args[0].meta.get("val")
            bv = node.args[1].meta.get("val")
            return self._no(
                node, KIND_BMM_LINEAR,
                f"bmm({list(av.shape)}, {list(bv.shape)}) against a constant is "
                f"mm({[int(av.shape[1]), int(av.shape[2])]}, "
                f"{[int(bv.shape[1]), int(bv.shape[2])]}) with a batch dim; the "
                f"delegate implements this arithmetic but claims only the mm "
                f"spelling, and which one torch emits is not the model author's "
                f"choice; canonical_passes() (BmmToMm) rewrites it as mm "
                f"before partitioning")
        # The Target column of the report already names the op, so the detail says
        # what to DO about it rather than repeating the schema.
        return self._no(
            node, "op not in the supported set",
            "this backend delegates quantised matmuls and the activation, "
            "residual and view nodes that fuse into them; everything else runs "
            "on portable kernels, which is correct but unaccelerated")


def _pad_to(n: int, mult: int) -> int:
    return (n + mult - 1) // mult * mult


def _dot_length(k):
    """(kind, detail) for an int8 dot of K terms an int32 accumulator cannot
    hold exactly, else None. The runtime bounds k_pad, but MAX_DOT_LEN is a
    multiple of 32, which every row alignment divides: k_pad passes it
    exactly when K does."""
    if k <= MAX_DOT_LEN:
        return None
    return ("dot past the exact int32 accumulator",
            f"K={k}: the int8 dot accumulates codes |x| <= 127 times "
            f"|w| <= 128 in int32, exact to {MAX_DOT_LEN} terms; the "
            f"runtime refuses a longer one at init")


def _check_pass_through_conv(pt, geom):
    """PT2E per-output-channel (or per-tensor) codes, already [N, K] after the
    resolver's reshape, checked against the geometry the gather will walk: K
    is C_in PER GROUP times taps, and N the full output channel count before
    any split."""
    if bool(geom.transposed) != pt.transposed:
        raise LoweringRefused(
            "pass-through conv codes mismatch",
            "a transposed convolution's codes must arrive turned into its "
            "dense equivalent's rows, and only its codes may")
    want = (geom.c_out, geom.k)
    if tuple(pt.codes.shape) != want:
        raise LoweringRefused(
            "pass-through conv codes mismatch",
            f"codes are {tuple(pt.codes.shape)} (from {pt.source_shape}), "
            f"expected {want}")
    if pt.permuted or pt.logical_rows is not None:
        raise LoweringRefused(
            "pass-through conv codes mismatch",
            "codes arrived through a transpose or row slice")


_CONSTANT_SLOTS = {
    exir_ops.edge.aten.mm.default: (1,),
    exir_ops.edge.aten.addmm.default: (0, 2),
    exir_ops.edge.aten.convolution.default: (1, 2),
    exir_ops.edge.aten.embedding.default: (0,),
}


def _reads_as_constant(user, node) -> bool:
    """Whether core op `user` reads `node` only where preprocess reads a
    constant (_constant_operands' slots: a weight, a bias, a table)."""
    slots = _CONSTANT_SLOTS.get(getattr(user, "target", None))
    if slots is None:
        return False
    where = [i for i, a in enumerate(user.args) if a is node]
    return bool(where) and all(i in slots for i in where)


def _constant_operands(node):
    """(role, operand) for each operand of a core op that preprocess reads as
    a constant: the weight, an addmm's or a convolution's bias, an
    embedding's table."""
    if node.target is exir_ops.edge.aten.addmm.default:
        return (("weight", node.args[2]), ("bias", node.args[0]))
    if node.target is exir_ops.edge.aten.convolution.default:
        return (("weight", node.args[1]), ("bias", node.args[2]))
    if node.target is exir_ops.edge.aten.embedding.default:
        return (("table", node.args[0]),)
    return (("weight", node.args[1]),)


def _fake_data_error(error) -> bool:
    """Whether an exception is torch refusing to read a FakeTensor's values
    -- the partition gate's lack of data, not a property of the weight."""
    from torch._subclasses import fake_tensor

    kinds = tuple(getattr(fake_tensor, name) for name in (
        "DataDependentOutputException", "DynamicOutputShapeException",
        "UnsupportedFakeTensorException") if hasattr(fake_tensor, name))
    text = str(error).lower()
    return isinstance(error, kinds) or "fake" in text or "data pointer" in text


def _check_lowerable(ep: ExportedProgram, node) -> None:
    """Raise LoweringRefused for whatever preprocess would refuse about this
    core op's quantised weight or bias: the collectors' own resolution and the
    emitters' own layout checks, everything but packing. The partition gate
    calls it for every core op (Esp32OperatorSupport._lowering_problem), so a
    model whose weight preprocess cannot take keeps that op portable.

    Two checks read VALUES, which the gate sees only where it can reach the
    real constant behind a FakeTensor (_concrete; every parameter, buffer and
    constant placeholder): _check_code_range (codes inside their
    declared range) and _resolve_w_off's layout and mid-point checks. Where
    it cannot -- codes computed by graph ops, or torch internals moved -- the
    weight is claimed by its structure alone and the same check refuses it
    in preprocess, failing the whole export, by name, instead of leaving the
    op portable."""
    target = node.target
    try:
        act = _act_record(node)
        if target is exir_ops.edge.aten.embedding.default:
            if _is_pass_through_const_path(node.args[0]):
                pt = _resolve_const(ep, node.args[0])
                if isinstance(pt, PassThroughWeight):
                    k = (int(pt.codes.shape[1]) if pt.logical_cols is None
                         else pt.logical_cols)
                    _pass_through_table_layout(pt, k, int(pt.codes.shape[0]))
            return
        if target is exir_ops.edge.aten.convolution.default:
            from executorch.backends.esp32s3.operator_support.conv import admit, split_groups

            geom = admit(node)
            pt_full = None
            if _is_pass_through_const_path(node.args[1]):
                pt_full = _resolve_const(ep, node.args[1])
                if isinstance(pt_full, PassThroughWeight):
                    _check_pass_through_conv(pt_full, geom)
                    per_out = geom.c_out // max(geom.groups, 1)
                    subs = split_groups(geom)
                    for sub in subs:
                        lo = sub.group_idx * per_out
                        pt = (_slice_pass_through(pt_full, lo, lo + per_out)
                              if len(subs) > 1 else pt_full)
                        _pass_through_matmul_layout(pt, sub.k, sub.n, act)
            if act is not None and not isinstance(pt_full, PassThroughWeight):
                # A weight quantised here: int8 whole rows at worst.
                for sub in split_groups(geom):
                    _affine_bounds(None, FORMATS["i8"], sub.k, 0, act)
            b_node = node.args[2]
            if (isinstance(b_node, torch.fx.Node)
                    and _is_pass_through_const_path(b_node)):
                bias = _resolve_const(ep, b_node)
                if (not isinstance(bias, torch.Tensor)
                        or tuple(bias.shape) != (geom.c_out,)):
                    raise LoweringRefused(
                        "quantised bias unsupported",
                        f"{b_node.name} is not a ({geom.c_out},) vector of "
                        f"codes")
            return
        selected = _selected_mm_pattern(node, ep)
        if selected is not None:
            if act is not None and act.code_bits == 16:
                raise LoweringRefused(
                    "16-bit activation codes on a selected-row head",
                    "the selected-row walk dots int8 activation codes; the "
                    "int16 loop runs whole matmuls and convs")
            table = _resolve_const(ep, selected["table"])
            if isinstance(table, PassThroughWeight):
                expected = (selected["vocab"], selected["k"])
                if (table.permuted or table.logical_rows is not None
                        or table.logical_cols is not None
                        or tuple(table.codes.shape) != expected):
                    raise LoweringRefused(
                        "selected-row table layout unsupported",
                        f"codes {tuple(table.codes.shape)}, expected "
                        f"{expected}")
                _pass_through_matmul_layout(table, selected["k"],
                                            selected["vocab"], act)
            elif act is not None:
                _affine_bounds(None, FORMATS["i8"], selected["k"], 0, act)
            return
        if target is exir_ops.edge.aten.mm.default:
            bias_n, mat1, mat2 = None, node.args[0], node.args[1]
        elif target is exir_ops.edge.aten.addmm.default:
            bias_n, mat1, mat2 = node.args[:3]
        else:
            return
        _, k = _shape_of(mat1)
        _, n = _shape_of(mat2)
        weight = None
        if _is_pass_through_const_path(mat2):
            weight = _resolve_const(ep, mat2)
            if isinstance(weight, PassThroughWeight):
                _check_pass_through_matmul(weight, k, n)
                _pass_through_matmul_layout(weight, k, n, act)
        if act is not None and not isinstance(weight, PassThroughWeight):
            # A weight quantised here: int8 whole rows at worst.
            _affine_bounds(None, FORMATS["i8"], k, 0, act)
        if (isinstance(bias_n, torch.fx.Node)
                and _is_pass_through_const_path(bias_n)):
            # A quantised bias (Arm's int32 per-channel): evaluated to fp32.
            bias = _resolve_const(ep, bias_n)
            if (not isinstance(bias, torch.Tensor)
                    or tuple(bias.shape) != (n,)):
                raise LoweringRefused(
                    "quantised bias unsupported",
                    f"{bias_n.name} is not a ({n},) vector of codes")
    except LoweringRefused:
        raise
    except Exception as why:   # what preprocess would raise on this weight
        if _fake_data_error(why):
            return
        raise LoweringRefused("quantised weight unsupported",
                              f"{type(why).__name__}: {why}") from None


def _affine_bounds(pt, fmt, k, scale_group, act=None):
    """The runtime's exactness bounds for an affine op (esp32_asym.cpp
    check_bounds), refused by name at export: the backend's own activations
    (|x| <= 127) against |c - zw| <= 15 (int4) or 255 (int8), or the model's
    own activation record (`act`) against any weight, symmetric included
    (`pt` None: a weight quantised here)."""
    from executorch.backends.esp32s3.serialization.asym_records import (
        SchemeRefused,
        check_act_bounds,
        check_weight_bounds,
    )

    longest = scale_group if 0 < scale_group < k else k
    zero_point = pt is not None and (pt.zero_points is not None
                                     or pt.storage_bias != 0)
    offsets = pt is not None and pt.offsets is not None
    try:
        if act is not None:
            check_act_bounds(act, fmt.bits, longest, zero_point=zero_point,
                             offsets=offsets)
        elif pt is not None and pt.affine:
            check_weight_bounds(fmt.bits, longest, zero_point=zero_point,
                                offsets=offsets)
    except SchemeRefused as why:
        raise LoweringRefused(
            "activation record past the exactness bound" if act is not None
            else "affine weight past the exactness bound", str(why)) from None


def _declared_code_range(weight):
    """The [qmin, qmax] a pass-through weight's dequant declares for its
    codes (PT2E or torchao), or None for any other weight."""
    src = _pass_through_source(weight)
    kind = _pt2e_dequant_kind(src)
    if kind == "per_channel" and len(src.args) > 5:
        return int(src.args[4]), int(src.args[5])
    if kind in ("per_tensor", "per_channel_group") and len(src.args) > 4:
        return int(src.args[3]), int(src.args[4])
    if _is_torchao_affine(src):
        return _torchao_code_range(src)
    return None


def _a16_problem(pt, fmt, scale_group, act):
    """What the int16 loop (16-bit activations, Arm a16w8)
    cannot dot, refused by name: it takes the model's 16-bit codes
    against ungrouped symmetric int8 weight codes only -- the runtime refuses
    16-bit codes on anything else as act-code-bits or unimplemented, and has
    no zero-point correction for them. `pt` is the pass-through weight, or
    None for one quantised here (the gate checks its policy)."""
    if act is None or act.code_bits != 16:
        return
    if fmt.bits != 8:
        raise LoweringRefused(
            f"16-bit activation codes on int{fmt.bits} weights",
            "the int16 loop dots int8 weight codes only")
    if scale_group:
        raise LoweringRefused(
            "16-bit activation codes on grouped weights",
            f"groups of {scale_group}: the int16 loop applies one scale per "
            f"output channel")
    if pt is not None and pt.affine:
        raise LoweringRefused(
            "16-bit activation codes on affine weights",
            "weight zero points or float offsets: the int16 loop corrects "
            "nothing")


def _pass_through_matmul_layout(pt, k, n, act=None):
    """Everything _emit_pass_through_matmul checks before it packs: returns
    (fmt, scale_group, storage_n), or raises LoweringRefused for what it
    cannot pack. The partition gate asks the same first
    (Esp32OperatorSupport._lowering_problem), so these refusals leave the op
    portable instead of failing the export."""
    fmt = _pass_through_format(pt)
    if fmt is None and act is not None and act.code_bits == 16:
        raise LoweringRefused(
            "int16 weights with static int16 activations",
            f"weight codes declared [{pt.qmin}, {pt.qmax}] under 16-bit "
            f"activation codes (a16w16): the runtime's int16 activations take "
            f"int8 weights, and its int16 weights (the backend's own rung) "
            f"take no activation record")
    if fmt is None:
        raise LoweringRefused(
            "pass-through codes wider than int8",
            f"codes declared [{pt.qmin}, {pt.qmax}]; i16 pass-through needs "
            f"grouped i16 accumulation, which the runtime refuses")
    storage_n, storage_k = (int(d) for d in pt.codes.shape)
    logical_n = storage_n if pt.logical_rows is None else pt.logical_rows
    logical_k = storage_k if pt.logical_cols is None else pt.logical_cols
    if logical_n != n or logical_k != k:
        raise LoweringRefused(
            "pass-through weight shape mismatch",
            f"logical rows/cols {logical_n}/{logical_k}, matmul N/K {n}/{k}")
    if pt.float_zero_points is not None:
        raise LoweringRefused(
            "fractional float zero point on a matmul weight",
            "the integer dot subtracts an integer zero point exactly; "
            "fl(fl(q - z) * s) with a fractional z has no exact form in it, "
            "and a float zero point the gate cannot "
            "read is not provably integral")
    if pt.gsize >= k:
        scale_group = 0
        if pt.scales.numel() != storage_n:
            raise LoweringRefused(
                "pass-through scales mismatch",
                f"{pt.scales.numel()} per-row scales for {storage_n} rows")
    else:
        scale_group = pt.gsize
        need = 32 if fmt.bits == 4 else 16
        if scale_group % need != 0:
            raise LoweringRefused(
                "pass-through group size unsupported",
                f"group size {scale_group} not a multiple of {need}; PIE "
                f"group offsets would misalign")
        want = (storage_n, -(-k // scale_group))
        if tuple(pt.scales.shape) != want:
            raise LoweringRefused(
                "pass-through scales mismatch",
                f"scales {tuple(pt.scales.shape)} != {want}")
    _a16_problem(pt, fmt, scale_group, act)
    _affine_bounds(pt, fmt, k, scale_group, act)
    return fmt, scale_group, storage_n


def _pass_through_table_layout(pt, k, n):
    """_pass_through_matmul_layout's twin for an embedding table: returns
    (fmt, scale_group)."""
    fmt = _pass_through_format(pt)
    if fmt is None:
        raise LoweringRefused(
            "pass-through table wider than int8",
            f"codes declared [{pt.qmin}, {pt.qmax}]; only i4/i8 tables pass "
            f"through")
    storage_k = int(pt.codes.shape[1])
    logical_k = storage_k if pt.logical_cols is None else pt.logical_cols
    if logical_k != k:
        raise LoweringRefused(
            "pass-through table shape mismatch",
            f"logical columns {logical_k} != K {k}")
    if pt.gsize >= k:
        if pt.scales.numel() != n:
            raise LoweringRefused(
                "pass-through scales mismatch",
                f"{pt.scales.numel()} per-row scales for {n} rows")
        return fmt, 0
    want = (n, (k + pt.gsize - 1) // pt.gsize)
    if tuple(pt.scales.shape) != want:
        # A scale array of the wrong shape reaches the runtime as a
        # differently-strided read, silently.
        raise LoweringRefused(
            "pass-through scales mismatch",
            f"grouped table scales are {tuple(pt.scales.shape)}, expected "
            f"{want} for k={k} group={pt.gsize}")
    return fmt, pt.gsize


def _pass_through_format(pt):
    """The narrowest rung whose stored code holds all of [qmin, qmax], or None.

    Both bounds decide it. pack_rows biases an i4 code by +8 into a nibble,
    so [-8, 7] is exactly what i4 holds; codes declared [-128, 7] are int8
    data even though qmax alone looks like a nibble, and a -9 among them
    would be packed as +7. _check_code_range has already proven the codes
    lie inside the declared range.
    """
    if pt.qmin >= -8 and pt.qmax <= 7:
        return FORMATS["i4"]
    if pt.qmin >= -128 and pt.qmax <= 127:
        return FORMATS["i8"]
    return None
