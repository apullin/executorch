# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Memory planning for ESP32 programs: memory_plan="esp32", the default of
every exporter here.

ExecuTorch's planner (exir/memory_planning.py: greedy, over the graph's own
order) never lets an operator's output take the storage of an input that dies
at that operator, so a residual add holds three full tensors; it plans the
order the graph happens to have; and its greedy packs lifetimes into shared
objects that fragment. This module plans each method several ways and keeps
the smallest arena:

  in-place reuse  an allowlisted elementwise operator's output takes the
                  storage of an input whose lifetime ends at that operator
                  (Regor's IFM reuse, compiler/live_range.cpp), through
                  TensorSpec.storage_base, which ExecuTorch's greedy and its
                  Verifier already honour for in-place operators;
  operator order  a topological order with the smallest peak of live bytes: a
                  search over the graph's frontiers (the sets of operators run
                  so far), exact while they stay few, a beam beyond; every
                  operator touching a tensor that some operator writes in place
                  (a KV cache's index_put_, a buffer's copy_ write-back) keeps
                  its order relative to the others;
  allocator       Regor's hill climb (compiler/hillclimb_allocator.cpp) beside
                  ExecuTorch's greedy.

Every candidate (order x in-place x allocator) is planned with ExecuTorch's
own lifetime and spec-collection functions, exactly as apply_algo will plan
the winner. ExecuTorch's own plan is the first candidate and wins every tie,
so no method's arena grows, and a method with nothing to gain is planned byte
for byte as ExecuTorch plans it.

memory_plan=None (`--memory-plan none` on the exporters' command lines) is
ExecuTorch's own MemoryPlanningPass; nothing in this module runs then.

In-place reuse is allowed only when all of these hold:
  - the operator is in INPLACE_OPS, whose kernels (host and device: our
    overrides in ops/ and ExecuTorch's portable kernels,
    every fast path in them) read element i of every operand before writing
    element i of the output, and choose no path by where the buffers lie;
  - every elementwise operand has the output's shape, dtype and dim order (no
    broadcast: a broadcast operand is read again after the output element
    over it is written), and the donor's byte size equals the output's;
  - the donor is planned memory that is not a graph input or output, a mutable
    buffer, a constant or a view (memory.view) alias, that no operator writes
    in place, and that no other storage alias involves;
  - the donor's lifetime ends at this operator.
test_memory_planning.py proves each kernel alias-safe on the host runner and
checks the refusals.

A delegate's planned scratch (exir.backend.backend_details.DelegateScratchSpec:
the working memory it declares for one execute() call) is a spec like any
other here, live at its call alone, as ExecuTorch's own lifetime rule makes
it: every candidate is planned with it, so scratch may take the bytes of
whatever is dead during the call.
"""

from __future__ import annotations

import itertools
import operator
import random
from dataclasses import dataclass, field

from executorch.exir import memory
from executorch.exir.error import InternalError
from executorch.exir.inplace_aliasing import (
    is_inplace_node,
    verify_inplace_result_aliases,
)
from executorch.exir.memory_planning import (
    _contains_xnnpack_delegate,
    _extend_storage_base_lifetimes,
    _is_mutable_buffer,
    apply_algo,
    collect_specs_from_nodes,
    filter_nodes,
    get_cond_nodes,
    get_map_nodes,
    get_node_tensor_specs,
    get_scan_nodes,
    get_while_nodes,
    greedy,
    MemoryAlgoResult,
    MemoryPlanningAlgorithmSuite,
    SpecAllocResult,
    update_tensor_lifetime,
    Verifier,
)
from executorch.exir.operator.convert import get_out_args_from_opoverload
from executorch.exir.pass_base import PassResult
from executorch.exir.passes.memory_planning_pass import (
    _check_default_mem_ids,
    _insert_mutable_buffer_specs,
    MemoryPlanningPass,
)
from executorch.exir.passes.replace_view_copy_with_view_pass import _ViewSpec
from executorch.exir.schema import DeviceType, TensorShapeDynamism
from executorch.exir.tensor import TensorSpec
from torch.utils._pytree import tree_flatten

MEMORY_PLANS = ("esp32",)

# Out-variant kernels whose output may overwrite an operand (schema name,
# overload) -> the elementwise operands' argument positions; any of them may
# donate. The evidence, per kernel (paths under backends/esp32s3/ops and
# kernels/portable/cpu; the same sources build the host runner
# and the firmware, and none qualifies a pointer __restrict):
#   add.out      shim_add_out (espressif_ops.cpp): same-shape fp32 alpha=1
#                loop dst[i] = lhs[i] + rhs[i]; its leading-broadcast path
#                needs operands of different shapes, which this module never
#                aliases; everything else op_add.cpp's add_out, whose
#                elementwise_util.h loop loads every operand at index i, then
#                stores the output at i (same-shape operands index i
#                everywhere: BroadcastIndexesRange's no-broadcast case).
#   sub.out      op_sub.cpp: the same elementwise_util.h loop.
#   mul.out      shim_mul_out: same-shape loop, entered for disjoint buffers
#                and an exact alias alike (tensor_output_alias_safe), so the
#                path never depends on where the output lies; a one-element
#                or broadcast path needs differing shapes; else op_mul.cpp.
#   relu.out     op_relu.cpp: apply_unary_map_fn, data_out[i] = f(data_in[i]).
#   clamp.out    op_clamp.cpp clamp_out (scalar bounds): elementwise_util.h.
#   hardtanh.out op_hardtanh.cpp: apply_unary_map_fn.
#   sigmoid.out  op_sigmoid.cpp: elementwise_util.h.
#   tanh.out     pattern/unary_ufunc_realhbbf16_to_floathbf16.cpp:
#                apply_unary_map_fn.
#   espressif::quant_lut.out  quant_lut.cpp apply(): dst[i] from x[i] alone
#                (the table is argument 1, a constant, never an operand).
# Not here: quantize/dequantize_per_tensor change the dtype (int8 codes over
# fp32 values, never the same tensor), so they could never qualify.
INPLACE_OPS = {
    ("aten::add", "out"): (0, 1),
    ("aten::sub", "out"): (0, 1),
    ("aten::mul", "out"): (0, 1),
    ("aten::relu", "out"): (0,),
    ("aten::clamp", "out"): (0,),
    ("aten::hardtanh", "out"): (0,),
    ("aten::sigmoid", "out"): (0,),
    ("aten::tanh", "out"): (0,),
    ("espressif::quant_lut", "out"): (0,),
}

# The order search's work cap, in frontier transitions per search: a graph
# whose frontiers stay under the beam width is searched exactly.
SEARCH_TRANSITIONS = 1_500_000
MAX_BEAM = 4096
# The hill climb's work cap per allocation problem, in neighbour visits.
HILL_CLIMB_WORK = 4_000_000


def planning_pass(memory_plan="esp32", **kwargs):
    """The memory planning pass for `memory_plan`, with MemoryPlanningPass's
    keyword arguments: "esp32" (the default) this module's, None ExecuTorch's
    own pass."""
    if memory_plan is None:
        return MemoryPlanningPass(**kwargs)
    if memory_plan not in MEMORY_PLANS:
        raise ValueError(f"memory_plan {memory_plan!r}: expected None or one of {MEMORY_PLANS}")
    return Esp32MemoryPlanningPass(**kwargs)


def install(memory_plan):
    """Make every EdgeProgramManager.to_executorch in this process plan with
    `memory_plan` (run_parity's --memory-plan, "esp32" by default: its
    exporters build their own configs). A plain MemoryPlanningPass in the
    config is swapped for this module's with the same settings; any other
    pass is left alone. None installs nothing."""
    if memory_plan is None:
        return
    planning_pass(memory_plan)  # validates the name
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.program._program import EdgeProgramManager

    original = getattr(EdgeProgramManager.to_executorch, "_memory_plan_original",
                       EdgeProgramManager.to_executorch)

    def swap(pass_):
        if type(pass_) is not MemoryPlanningPass:
            return pass_
        swapped = Esp32MemoryPlanningPass(
            allow_lifetime_and_storage_overlap=pass_.allow_lifetime_and_storage_overlap,
            alloc_graph_input=pass_.alloc_graph_input,
            alloc_graph_output=pass_.alloc_graph_output,
            alloc_mutable_buffers=pass_.alloc_mutable_buffers,
            share_mutable_buffers=pass_.share_mutable_buffers,
            alignment=pass_.alignment)
        return swapped

    def to_executorch(self, config=None):
        import dataclasses

        config = config if config else ExecutorchBackendConfig()
        planner = config.memory_planning_pass
        planner = ({k: swap(v) for k, v in planner.items()} if isinstance(planner, dict)
                   else swap(planner))
        return original(self, dataclasses.replace(config, memory_planning_pass=planner))

    to_executorch._memory_plan_original = original
    EdgeProgramManager.to_executorch = to_executorch


@dataclass
class MethodPlan:
    """What the planner did for one method (Esp32MemoryPlanningPass.plans)."""

    nodes: int = 0
    baseline: int | None = None        # ExecuTorch's arena: greedy, the graph's order
    baseline_bound: int | None = None  # live-bytes lower bound of that plan
    arena: int | None = None           # the chosen plan's arena
    bound: int | None = None           # live-bytes lower bound of the chosen plan
    choice: str = "ExecuTorch"         # which candidate won
    inplace: int = 0                   # outputs placed over a dying operand
    reordered: bool = False
    search: str = ""                   # how the order was searched, or why not
    note: str = ""                     # why the plan fell back, if it did
    candidates: list = field(default_factory=list)  # (label, arena, bound)
    # Delegate calls that declared planned scratch, and the largest one's
    # bytes: part of `arena`, live during its call alone.
    scratch_calls: int = 0
    scratch_max: int = 0

    def describe(self):
        if self.arena is None:
            return f"not planned: {self.note}"
        text = (f"arena {self.arena:,} B (ExecuTorch {self.baseline:,}; live bound "
                f"{self.bound:,}): {self.choice}")
        if self.scratch_calls:
            text += (f"; planned scratch for {self.scratch_calls} delegate call(s), "
                     f"largest {self.scratch_max:,} B")
        if self.search:
            text += f"; order search: {self.search}"
        if self.note:
            text += f"; {self.note}"
        return text


class Esp32MemoryPlanningPass(MemoryPlanningPass):
    """MemoryPlanningPass that plans each method the module docstring's ways
    and keeps the smallest arena (never larger than ExecuTorch's). `inplace`,
    `reorder` and `hill_climb` switch each tool off for measurement. After
    to_executorch, `plans` holds one MethodPlan per method, in method order."""

    def __init__(self, *, inplace=True, reorder=True, hill_climb=True, **kwargs):
        super().__init__(**kwargs)
        self.inplace = inplace
        self.reorder = reorder
        self.hill_climb = hill_climb
        self.plans: list[MethodPlan] = []

    def _alloc_mutable(self):
        return self.alloc_mutable_buffers and not self.share_mutable_buffers

    def run(self, graph_module, graph_signature=None):
        # MemoryPlanningPass.run, with the plan chosen before apply_algo. Its
        # first check too: an in-place op added after SpecPropPass would get a
        # buffer of its own that its kernel never writes.
        verify_inplace_result_aliases(graph_module)
        self._set_alloc_node_spec(graph_module)
        algo, plan = self._choose(graph_module, graph_signature)
        bufsizes = apply_algo(
            MemoryPlanningAlgorithmSuite([algo]), graph_module, self.alignment,
            graph_signature, self.alloc_graph_input, self.alloc_graph_output,
            self._alloc_mutable(), self.enable_non_cpu_memory_planning)
        if self.share_mutable_buffers and graph_signature is not None:
            self.state.graph_modules.append(graph_module)
            _check_default_mem_ids(graph_module)
            _insert_mutable_buffer_specs(self.state, graph_module, graph_signature)
        verifier = Verifier(graph_module, self.alloc_graph_input, self.alloc_graph_output,
                            self._alloc_mutable(), graph_signature)
        verifier.verify_graph_input_output()
        planned = list(collect_specs_from_nodes(
            graph_module.graph.nodes, graph_signature, do_assertion=False,
            ignore_graph_input=not self.alloc_graph_input,
            ignore_graph_output=not self.alloc_graph_output,
            ignore_mutable_buffers=not self._alloc_mutable()))
        check_storage(planned)
        scratch = [s.nbytes() for node in graph_module.graph.nodes
                   for s in memory.delegate_scratch_specs(node)]
        plan.scratch_calls = sum(1 for node in graph_module.graph.nodes
                                 if memory.delegate_scratch_specs(node))
        plan.scratch_max = max(scratch, default=0)
        arena = sum(bufsizes[1:])
        if plan.arena is None:  # refused: ExecuTorch's plan, measured here
            plan.baseline = arena
            plan.bound = plan.baseline_bound = live_bound(planned)
        elif arena != plan.arena:
            raise InternalError(f"memory plan: apply_algo planned {arena:,} B where the "
                                f"chosen candidate measured {plan.arena:,} B")
        plan.arena = arena
        self.plans.append(plan)
        return PassResult(graph_module, True)

    # ------------------------------------------------------------ choosing

    def _choose(self, gm, gs):
        nodes = list(gm.graph.nodes)
        plan = MethodPlan(nodes=len(nodes))
        refusal = _plan_refusal(gm, self.enable_non_cpu_memory_planning)
        if refusal:
            plan.note = f"ExecuTorch's plan ({refusal})"
            return greedy, plan
        snapshot = _Snapshot(gm)
        extra = 64 if _contains_xnnpack_delegate(gm) else 0
        candidates = []

        def consider(label, order, pairs, algo):
            arena, bound = self._evaluate(gm, gs, order, pairs, algo, snapshot, extra)
            candidates.append((arena, len(candidates), label, order, pairs, algo, bound))

        try:
            consider("ExecuTorch", nodes, {}, greedy)
            orders = [("graph order", nodes)]
            if self.reorder:
                refusal = _reorder_refusal(gm, gs)
                if refusal:
                    plan.search = f"not reordered ({refusal})"
                else:
                    found, plan.search = self._search(gm, gs, nodes, snapshot)
                    for label, order in found:
                        if all(order != other for _, other in orders):
                            orders.append((label, order))
            allocators = [greedy, hillclimb] if self.hill_climb else [greedy]
            for label, order in orders:
                pairs = self._inplace_pairs(gm, gs, order, snapshot) if self.inplace else {}
                for algo in allocators:
                    for chosen in ({}, pairs) if pairs else ({},):
                        if order is nodes and not chosen and algo is greedy:
                            continue  # the baseline itself
                        name = label + (f" + {len(chosen)} in place" if chosen else "")
                        consider(f"{name}, {algo.__name__}", order, chosen, algo)
        except Exception as e:  # noqa: BLE001 -- any surprise falls back to ExecuTorch's plan
            snapshot.restore()
            plan.note = f"ExecuTorch's plan (planner error {type(e).__name__}: {str(e)[:200]})"
            plan.candidates = []
            return greedy, plan
        snapshot.restore()

        # Equal arenas break by insertion order: ExecuTorch's own plan (candidate 0)
        # wins a tie, so a method no candidate improves keeps its bytes.
        arena, _, label, order, pairs, algo, bound = min(candidates, key=lambda c: c[:2])
        base = candidates[0]
        plan.baseline, plan.baseline_bound = base[0], base[6]
        plan.arena, plan.bound, plan.choice = arena, bound, label
        plan.inplace, plan.reordered = len(pairs), order is not nodes
        plan.candidates = [(c[2], c[0], c[6]) for c in candidates]
        if order is not nodes:
            _reorder_graph(gm, order)
        for out, donor in pairs.items():
            out.storage_base, out.storage_base_offset = donor, 0
        return algo, plan

    def _planned(self, order, gs):
        # apply_algo's spec collection, in `order`.
        return list(collect_specs_from_nodes(
            order, gs, do_assertion=False,
            ignore_graph_input=not self.alloc_graph_input,
            ignore_graph_output=not self.alloc_graph_output,
            ignore_mutable_buffers=not self._alloc_mutable()))

    def _evaluate(self, gm, gs, order, pairs, algo, snapshot, extra):
        """Plan `order` with `pairs` in place as apply_algo would: (arena, live
        bound). Leaves every spec as the snapshot holds it."""
        snapshot.restore()
        try:
            for out, donor in pairs.items():
                out.storage_base, out.storage_base_offset = donor, 0
            set_lifetimes(order, gs)
            specs = self._planned(order, gs)
            gm.input_mem_buffer_sizes = []
            result = algo(self.alignment, specs, gm, gs, extra)
            return sum(result.bufsizes[1:]), live_bound(specs) + extra
        finally:
            snapshot.restore()

    def _inplace_pairs(self, gm, gs, order, snapshot):
        """{output spec: donor spec} for every allowlisted operator in `order`
        whose output may overwrite a dying operand (module docstring)."""
        snapshot.restore()
        try:
            set_lifetimes(order, gs)
            planned = {id(s) for s in self._planned(order, gs)}
            index = {node: i for i, node in enumerate(order)}
            fixed, outputs = _fixed_specs(order, gs, snapshot)
            pairs = {}
            for node in order:
                operands = inplace_operands(node)
                if operands is None:
                    continue
                out = node.meta.get("spec")
                if (not isinstance(out, TensorSpec) or id(out) not in planned
                        or id(out) in fixed or out.storage_base is not None):
                    continue
                donor = None
                for arg in operands:
                    spec = _single_spec(arg)
                    if (donor is None and _donor_node(arg) and spec is not None
                            and id(spec) in planned and id(spec) not in fixed
                            and id(spec) not in outputs
                            and spec.storage_base is None and not spec.const
                            and spec.lifetime[1] == index[node]
                            and spec.allocated_memory == out.allocated_memory):
                        donor = spec
                if donor is not None:
                    pairs[out] = donor
            return pairs
        finally:
            snapshot.restore()

    # ------------------------------------------------------------ order search

    def _search(self, gm, gs, nodes, snapshot):
        """Minimum-peak orders: [(label, node order)], and how they were found."""
        snapshot.restore()
        try:
            set_lifetimes(nodes, gs)
            planned = self._planned(nodes, gs)
            fixed, outputs = _fixed_specs(nodes, gs, snapshot)
            model = _UnitModel.build(nodes, gs, planned, fixed, outputs, self.alignment)
        finally:
            snapshot.restore()
        if model is None:
            return [], "not reordered (graph shape the search does not model)"
        if len(model.units) < 2:
            return [], f"{len(model.units)} operator(s): nothing to order"
        found, notes = [], []
        for label, aware in (("min-peak order", False), ("min-peak order (in-place aware)", True)):
            if aware and not (self.inplace and model.has_donors):
                continue
            units, peak, how = model.search(aware)
            order = model.node_order(units)
            if order is None:
                notes.append(f"{label}: no valid node order")
                continue
            found.append((label, order))
            notes.append(f"{label} {how}, live peak {peak:,} B")
        graph_peak = model.peak(list(range(len(model.units))), False)
        kept = (f", {model.constraints} order edge(s) for in-place writes"
                if model.constraints else "")
        return found, (f"{len(model.units)} operators{kept}, graph order's live peak "
                       f"{graph_peak:,} B; " + "; ".join(notes))


# ------------------------------------------------------------ graph helpers


class _Snapshot:
    """Every spec's planning fields as the pass found them; restore() puts
    them back (dry runs set lifetimes, and ExecuTorch extends a lifetime
    from whatever it already holds). A delegate's scratch specs too: they
    sit beside the call's meta["spec"], not in it (exir.memory)."""

    FIELDS = ("mem_id", "mem_obj_id", "mem_offset", "storage_base", "storage_base_offset")

    def __init__(self, gm):
        self.saved = []
        seen = set()
        for node in gm.graph.nodes:
            specs = tree_flatten(node.meta.get("spec"))[0]
            for spec in itertools.chain(specs, memory.delegate_scratch_specs(node)):
                if not isinstance(spec, TensorSpec):
                    continue
                spec = _base_spec(spec)
                if id(spec) in seen:
                    continue
                seen.add(id(spec))
                self.saved.append((spec, list(spec.lifetime),
                                   tuple(getattr(spec, f) for f in self.FIELDS)))

    def restore(self):
        for spec, lifetime, values in self.saved:
            spec.lifetime = list(lifetime)
            for name, value in zip(self.FIELDS, values):
                setattr(spec, name, value)

    def related(self):
        """Specs in a storage relation before this pass: backed or backing."""
        out = set()
        for spec, _, values in self.saved:
            base = values[3]
            if base is not None:
                out.add(id(spec))
                out.add(id(_base_spec(base)))
        return out


def _base_spec(spec):
    while isinstance(spec, _ViewSpec):
        spec = spec._base
    return spec


def _single_spec(node):
    specs = get_node_tensor_specs(node) if hasattr(node, "op") else []
    return specs[0] if len(specs) == 1 and isinstance(specs[0], TensorSpec) else None


def _donor_node(node):
    """A node whose value may donate its storage: an operator's result (an
    out-variant op or a delegate's getitem), never a graph input, a constant
    or a view alias."""
    return (hasattr(node, "op") and node.op == "call_function"
            and node.target not in (memory.view, memory.alloc))


def _fixed_specs(order, gs, snapshot):
    """(specs that keep storage of their own: graph inputs, mutable buffers,
    in-place operators' results and any spec already in a storage relation;
    graph outputs, which may be an in-place result but never donate)."""
    fixed, outputs = snapshot.related() | written_specs(order), set()
    for node in order:
        if node.op in ("placeholder", "output") or is_inplace_node(node):
            for spec in tree_flatten(node.meta.get("spec"))[0]:
                if isinstance(spec, TensorSpec):
                    (outputs if node.op == "output" else fixed).add(id(_base_spec(spec)))
    return fixed, outputs


def inplace_operands(node):
    """The allowlisted elementwise operands of `node`, when its output may
    take one's storage as far as shapes go (INPLACE_OPS, all operands the
    output's shape, dtype and dim order); else None."""
    if node.op != "call_function" or not hasattr(node.target, "_schema"):
        return None
    schema = node.target._schema
    positions = INPLACE_OPS.get((schema.name, schema.overload_name))
    if positions is None:
        return None
    out = node.meta.get("spec")
    out_args = get_out_args_from_opoverload(node.target)
    if (not isinstance(out, TensorSpec) or len(out_args) != 1
            or getattr(node.kwargs.get(out_args[0]), "target", None) is not memory.alloc):
        return None
    operands = []
    for position in positions:
        arg = node.args[position] if position < len(node.args) else None
        spec = _single_spec(arg) if arg is not None else None
        if (spec is None or list(spec.shape) != list(out.shape) or spec.dtype != out.dtype
                or tuple(spec.dim_order) != tuple(out.dim_order)):
            return None
        operands.append(arg)
    # A donor read anywhere else by this node (not as an elementwise operand)
    # could be read after its elements are overwritten.
    others = [a for i, a in enumerate(node.args) if i not in positions]
    others += [v for k, v in node.kwargs.items() if k not in out_args]
    elsewhere = {id(n) for n in filter_nodes(others)}
    return [a for a in operands if id(a) not in elsewhere]


def inplace_kernels(program):
    """{operator: count} of the kernel calls in an emitted program whose output
    shares its planned bytes with one of its tensor operands: what in-place
    reuse did, read back from the program itself."""
    found = {}
    for plan in program.executorch_program.execution_plan:
        names = [op.name + (f".{op.overload}" if op.overload else "") for op in plan.operators]

        def where(index):
            value = plan.values[index].val
            info = getattr(value, "allocation_info", None)
            if info is None or 0 in value.sizes:
                return None  # an empty tensor may sit anywhere
            return info.memory_id, info.memory_offset_low, info.memory_offset_high

        for chain in plan.chains:
            for instruction in chain.instructions:
                call = instruction.instr_args
                if type(call).__name__ != "KernelCall":
                    continue
                tensors = [i for i in dict.fromkeys(call.args)
                           if type(plan.values[i].val).__name__ == "Tensor"]
                if len(tensors) >= 2 and where(tensors[-1]) is not None and any(
                        where(i) == where(tensors[-1]) for i in tensors[:-1]):
                    name = names[call.op_index]
                    found[name] = found.get(name, 0) + 1
    return found


def set_lifetimes(order, gs):
    """update_all_tensors_lifetime over `order` instead of the graph's own
    node order (the same calls: ExecuTorch's lifetime rule, verbatim),
    including its scratch rule: a delegate's scratch lives from the first to
    the last call naming it, not to the last reader of the call's outputs."""
    specs = set()
    last = len(order) - 1
    for i, node in enumerate(order):
        for spec in collect_specs_from_nodes(
                filter_nodes(itertools.chain([node], node.args, node.kwargs.values())), gs,
                ignore_graph_input=False, ignore_const=False, ignore_out_var_node=False,
                dedup=False, do_assertion=False, ignore_dynamic_unbound_tensor=False):
            update_tensor_lifetime(node, spec, i, last, gs)
            specs.add(spec)
    scratch = {}
    for i, node in enumerate(order):
        for spec in memory.delegate_scratch_specs(node):
            start, end = scratch.get(spec, (i, i))
            scratch[spec] = (min(start, i), max(end, i))
    for spec, (start, end) in scratch.items():
        spec.lifetime = [start, end]
    _extend_storage_base_lifetimes(specs)
    return specs


def live_bound(specs):
    """The largest total size of specs live at one node: no allocator can
    plan these lifetimes in less (Regor's minRequiredSize). Storage-backed
    specs live inside their bases."""
    deltas = {}
    for spec in specs:
        if spec.storage_base is not None or spec.lifetime[0] is None:
            continue
        deltas[spec.lifetime[0]] = deltas.get(spec.lifetime[0], 0) + spec.allocated_memory
        deltas[spec.lifetime[1] + 1] = deltas.get(spec.lifetime[1] + 1, 0) - spec.allocated_memory
    live = peak = 0
    for t in sorted(deltas):
        live += deltas[t]
        peak = max(peak, live)
    return peak


def check_storage(specs):
    """Every two planned specs live at once occupy disjoint bytes, unless one
    is storage-backed by the other (the Verifier's rule, by a sweep)."""

    def chain(spec):
        out = []
        while spec is not None:
            out.append(spec)
            spec = spec.storage_base
        return out

    active = []
    for spec in sorted(specs, key=lambda s: (s.lifetime[0], s.lifetime[1])):
        if spec.allocated_memory == 0:
            continue
        active = [a for a in active if a.lifetime[1] >= spec.lifetime[0]]
        lo, hi = spec.mem_offset, spec.mem_offset + spec.allocated_memory
        for other in active:
            if other.mem_id != spec.mem_id:
                continue
            if lo < other.mem_offset + other.allocated_memory and other.mem_offset < hi:
                if not (other in chain(spec) or spec in chain(other)):
                    raise InternalError(
                        f"memory plan: specs {spec} (life {spec.lifetime}, bytes {lo}-{hi}) "
                        f"and {other} (life {other.lifetime}, bytes {other.mem_offset}-"
                        f"{other.mem_offset + other.allocated_memory}) overlap")
        active.append(spec)


def _plan_refusal(gm, enable_non_cpu):
    """Why this method keeps ExecuTorch's plan untouched, or None."""
    if any(True for f in (get_cond_nodes, get_while_nodes, get_map_nodes, get_scan_nodes)
           for _ in f(gm)):
        return "control flow"
    for node in gm.graph.nodes:
        if node.op not in ("placeholder", "get_attr", "call_function", "output"):
            return f"{node.op} node"
        for spec in tree_flatten(node.meta.get("spec"))[0]:
            if isinstance(spec, TensorSpec):
                if spec.shape_dynamism == TensorShapeDynamism.DYNAMIC_UNBOUND:
                    return "dynamic unbound shapes"
                if enable_non_cpu and (spec.device, spec.device_index) != (DeviceType.CPU, 0):
                    return "non-CPU tensors"
    return None


def _reorder_refusal(gm, gs):
    """Why this method's operator order is not free to change, or None.
    Mutation is no reason: _UnitModel keeps every node that touches a tensor
    some operator writes in place in its own order."""
    if gs is None:
        return "no graph signature"
    return None


def written_specs(nodes):
    """Ids of the specs some operator writes in place: an in-place op's self,
    or any other `Tensor(a!)` argument that is not an out argument
    (index_put_, copy_ write-backs, llama::update_cache's cache)."""
    written = set()
    for node in nodes:
        schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
        if schema is None:
            continue
        for position, arg in enumerate(schema.arguments):
            info = arg.alias_info
            if info is None or not info.is_write or arg.is_out:
                continue
            value = node.args[position] if position < len(node.args) else node.kwargs.get(arg.name)
            for written_node in filter_nodes([value]):
                for spec in get_node_tensor_specs(written_node):
                    if isinstance(spec, TensorSpec):
                        written.add(id(_base_spec(spec)))
    return written


def _reorder_graph(gm, order):
    output = order[-1]
    for node in order[:-1]:
        if node.op != "placeholder":
            output.prepend(node)
    if list(gm.graph.nodes) != list(order):
        raise InternalError("memory plan: the graph did not take the chosen order")
    gm.graph.lint()
    gm.recompile()


# ------------------------------------------------------------ the order search


class _UnitModel:
    """The method as operators ("units") and planned specs, for the order
    search. A unit is a call that does work (an out-variant op, a delegate);
    an alloc travels with the unit that writes it, a getitem or view goes
    right after the value it reads. A spec lives from its producing unit
    through its last reader (from the start for graph inputs, to the end for
    graph outputs), exactly ExecuTorch's lifetime once node_order lays the
    units out."""

    @classmethod
    def build(cls, nodes, gs, planned, fixed, outputs, alignment):
        self = cls()
        self.nodes = nodes
        if nodes[-1].op != "output" or any(n.op == "output" for n in nodes[:-1]):
            return None
        movable = (memory.alloc, memory.view, operator.getitem)
        self.units = [n for n in nodes if n.op == "call_function" and n.target not in movable]
        unit_index = {n: i for i, n in enumerate(self.units)}
        owner = {}
        for unit in self.units:
            for arg in unit.all_input_nodes:
                if arg.op == "call_function" and arg.target is memory.alloc:
                    owner.setdefault(arg, unit)
        for node in nodes:
            if node.op == "call_function" and node.target is memory.alloc and node not in owner:
                return None  # an alloc no unit writes
        self.owner = owner

        def producer(node):
            while node.op == "call_function" and node.target in (memory.view, operator.getitem):
                node = node.args[0]
            if node.op != "call_function":
                return None
            return owner[node] if node.target is memory.alloc else node

        self.producer = producer
        n_units = len(self.units)
        self.preds = [0] * n_units
        self.succs = [[] for _ in range(n_units)]
        for i, unit in enumerate(self.units):
            for arg in unit.all_input_nodes:
                p = producer(arg)
                if p is not None and p is not unit:
                    j = unit_index[p]
                    if j >= i:
                        return None  # the graph's own order is not topological here
                    if not self.preds[i] >> j & 1:
                        self.preds[i] |= 1 << j
                        self.succs[j].append(i)

        # Which unit (or START / END) each node's spec references count at.
        START, END = -1, -2

        def slot(node):
            if node.op in ("placeholder", "get_attr"):
                return START
            if node.op == "output":
                return END
            if node.target is memory.alloc:
                return unit_index[owner[node]]
            if node.target in (memory.view, operator.getitem):
                p = producer(node)
                return START if p is None else unit_index[p]
            return unit_index[node]

        planned_ids = {id(s): s for s in planned}
        written = written_specs(nodes)
        mutable = {id(_base_spec(spec)) for node in nodes if _is_mutable_buffer(node, gs)
                   for spec in get_node_tensor_specs(node)}
        # A delegate's scratch is named by its call and, through the call, by
        # the getitems reading its outputs; it is live during the call alone
        # (set_lifetimes), so only the call itself counts.
        scratch = {id(s) for n in nodes for s in memory.delegate_scratch_specs(n)}
        refs, touch = {}, {}
        for node in nodes:
            where = slot(node)
            own_scratch = {id(s) for s in memory.delegate_scratch_specs(node)}
            for spec in collect_specs_from_nodes(
                    filter_nodes(itertools.chain([node], node.args, node.kwargs.values())), gs,
                    ignore_graph_input=False, ignore_const=False, ignore_out_var_node=False,
                    dedup=False, do_assertion=False, ignore_dynamic_unbound_tensor=False):
                if id(spec) in scratch and id(spec) not in own_scratch:
                    continue
                if id(spec) in planned_ids:
                    refs.setdefault(id(spec), set()).add(where)
                if id(spec) in written and where >= 0:
                    touch.setdefault(id(spec), set()).add(where)
        # A tensor written in place has readers and writers no data edge need
        # link (a read of the old value, then the write): every unit touching
        # it stays in the graph's order relative to the others.
        self.constraints = 0
        for units_here in touch.values():
            chain = sorted(units_here)
            for j, i in zip(chain, chain[1:]):
                if not self.preds[i] >> j & 1:
                    self.preds[i] |= 1 << j
                    self.succs[j].append(i)
                    self.constraints += 1
        self.size, self.prod, self.cons, self.to_end = [], [], [], []
        spec_index, consumers = {}, []
        for sid, where in refs.items():
            spec = planned_ids[sid]
            if spec.storage_base is not None:
                return None  # pre-existing storage aliases: not modelled
            spec.realign(alignment)
            units_here = sorted(w for w in where if w >= 0)
            prod = START if START in where else (units_here[0] if units_here else None)
            if prod is None:
                # Only the output node names it: on 1.5.1 an in-place
                # write-back's own result, which 1.6 gives its buffer's spec
                # (SpecPropPass, checked by verify_inplace_result_aliases).
                # Not modelled.
                return None
            cons, readers = 0, [w for w in units_here if w != prod]
            for w in readers:
                cons |= 1 << w
            spec_index[sid] = len(self.size)
            self.size.append(spec.allocated_memory)
            self.prod.append(prod)
            self.cons.append(cons)
            # ExecuTorch keeps a mutable buffer to the method's last node.
            self.to_end.append(END in where or sid in mutable)
            consumers.append(readers)
        self.outs = [[] for _ in range(n_units)]
        self.ins = [[] for _ in range(n_units)]
        for s, prod in enumerate(self.prod):
            if prod >= 0:
                self.outs[prod].append(s)
            for u in consumers[s]:
                self.ins[u].append(s)
        self.out_total = [sum(self.size[s] for s in self.outs[u]) for u in range(n_units)]
        # In-place candidates per unit: the static half of _inplace_pairs (the
        # dynamic half, dying here, depends on the order).
        self.donors = [() for _ in range(n_units)]
        for u, unit in enumerate(self.units):
            operands = inplace_operands(unit)
            out = unit.meta.get("spec")
            if not operands or id(out) not in spec_index or id(out) in fixed:
                continue
            cands = []
            for arg in operands:
                spec = _single_spec(arg)
                if (_donor_node(arg) and spec is not None and id(spec) in spec_index
                        and id(spec) not in fixed and id(spec) not in outputs
                        and not spec.const
                        and spec.allocated_memory == out.allocated_memory):
                    cands.append(spec_index[id(spec)])
            self.donors[u] = tuple(cands)
        self.has_donors = any(self.donors)
        inputs = [s for s, p in enumerate(self.prod) if p == START]
        self.peak0 = sum(self.size[s] for s in inputs)
        self.live0 = sum(self.size[s] for s in inputs if self.cons[s] or self.to_end[s])
        return self

    def _step(self, mask, live, u, aware):
        """Run unit u after the units in mask: (new mask, bytes live while u
        runs, bytes live after)."""
        new = mask | (1 << u)
        during = live + self.out_total[u]
        if aware:
            for s in self.donors[u]:
                if not self.to_end[s] and not (self.cons[s] & ~new):
                    during -= self.size[s]
                    break
        freed = 0
        for s in self.ins[u]:
            if not self.to_end[s] and not (self.cons[s] & ~new):
                freed += self.size[s]
        for s in self.outs[u]:
            if not self.to_end[s] and not self.cons[s]:
                freed += self.size[s]
        return new, during, live + self.out_total[u] - freed

    def peak(self, units, aware):
        mask, live, peak = 0, self.live0, self.peak0
        for u in units:
            mask, during, live = self._step(mask, live, u, aware)
            peak = max(peak, during)
        return max(peak, live)

    def search(self, aware):
        """A unit order with the smallest live peak: the frontier search,
        exact while every layer keeps all its frontiers, a beam otherwise.
        Returns (units, peak, how)."""
        n = len(self.units)
        # The graph order's ready-set sizes, to size the beam.
        waiting = [bin(p).count("1") for p in self.preds]
        ready_now = sum(1 for w in waiting if w == 0)
        widths = []
        for u in range(n):
            ready_now -= 1
            for v in self.succs[u]:
                waiting[v] -= 1
                ready_now += waiting[v] == 0
            widths.append(max(ready_now, 1))
        per_state = max(1.0, sum(widths) / max(n, 1))
        beam = int(max(16, min(MAX_BEAM, SEARCH_TRANSITIONS / (n * per_state))))
        ready0 = tuple(u for u in range(n) if not self.preds[u])
        layer = {0: (self.peak0, self.live0, ready0)}
        history, exact, states = [], True, 1
        for k in range(n):
            nxt = {}
            for mask, (peak, live, ready) in layer.items():
                for u in ready:
                    new, during, after = self._step(mask, live, u, aware)
                    npeak = peak if peak >= during else during
                    old = nxt.get(new)
                    if old is not None and (old[0], old[1]) <= (npeak, after):
                        continue
                    nready = [v for v in ready if v != u]
                    for v in self.succs[u]:
                        if not (self.preds[v] & ~new):
                            nready.append(v)
                    nready.sort()
                    nxt[new] = (npeak, after, tuple(nready), mask, u)
            if len(nxt) > beam:
                exact = False
                ranked = sorted(nxt.items(), key=lambda kv: (kv[1][0], kv[1][1]))
                kept = dict(ranked[:beam])
                # The graph order's own frontier always stays, so the beam
                # never ends above the graph order's peak.
                graph = (1 << (k + 1)) - 1
                if graph not in kept:
                    kept.pop(ranked[beam - 1][0])
                    kept[graph] = nxt[graph]
                nxt = kept
            states += len(nxt)
            history.append(nxt)
            layer = {m: (v[0], v[1], v[2]) for m, v in nxt.items()}
        full = (1 << n) - 1
        order, mask = [], full
        for k in range(n - 1, -1, -1):
            _, _, _, parent, u = history[k][mask]
            order.append(u)
            mask = parent
        order.reverse()
        final = history[-1][full]
        peak = max(final[0], final[1])
        how = (f"exact over {states:,} frontiers" if exact
               else f"beam {beam} ({states:,} frontiers)")
        return order, peak, how

    def node_order(self, units):
        """The graph's nodes with the units in this order: placeholders and
        get_attrs first, each unit right after its allocs, each getitem or
        view right after the node it reads, the output last. None if that is
        not a valid order (it always should be)."""
        nodes = self.nodes
        position = {n: i for i, n in enumerate(nodes)}
        order = [n for n in nodes if n.op in ("placeholder", "get_attr")]
        placed = set(order)
        allocs = {}
        for alloc, unit in self.owner.items():
            allocs.setdefault(unit, []).append(alloc)
        readers = {}
        for n in nodes:
            if n.op == "call_function" and n.target in (memory.view, operator.getitem):
                readers.setdefault(n.args[0], []).append(n)

        def place(node):
            order.append(node)
            placed.add(node)
            for reader in sorted(readers.get(node, ()), key=position.get):
                place(reader)

        for node in list(order):
            for reader in sorted(readers.get(node, ()), key=position.get):
                place(reader)
        for u in units:
            unit = self.units[u]
            for alloc in sorted(allocs.get(unit, ()), key=position.get):
                place(alloc)
            place(unit)
        order.append(nodes[-1])
        if len(order) != len(nodes) or set(order) != set(nodes):
            return None
        seen = set()
        for node in order:
            if any(arg not in seen for arg in node.all_input_nodes):
                return None
            seen.add(node)
        return order


# ------------------------------------------------------------ the hill climb


def hillclimb(alignment, specs, graph_module, graph_signature, extra_padding=0):
    """Regor's hill-climb allocator (compiler/hillclimb_allocator.cpp) as a
    MemoryPlanningAlgorithmSuite member, beside ExecuTorch's greedy. Each spec
    goes at the lowest address free of every spec it overlaps in time, in an
    order that starts from Regor's urgency heuristic; then pairs of specs that
    shaped the highest one swap places in that order while that lowers the
    top. Deterministic (a fixed seed and iteration caps)."""
    result = MemoryAlgoResult({}, [])
    roots, deferred = {}, []
    for spec in specs:
        mem_id = 1 if spec.mem_id is None else spec.mem_id
        result.spec_dict[spec] = SpecAllocResult(mem_id, 0, 0)
        spec.realign(alignment)
        if spec.storage_base is not None:
            deferred.append(spec)
        else:
            roots.setdefault(mem_id, []).append(spec)
    if not roots:
        result.bufsizes = [0, 0]
        return result
    reserved = getattr(graph_module, "input_mem_buffer_sizes", None) or []
    totals = [0] * (max(roots) + 1)
    for mem_id, group in roots.items():
        base = reserved[mem_id] if len(reserved) > mem_id else 0
        offsets, size = _hill_climb([(s.lifetime[0], s.lifetime[1], s.allocated_memory)
                                     for s in group])
        for spec, offset in zip(group, offsets):
            result.spec_dict[spec].mem_offset = base + offset
        totals[mem_id] = base + size + extra_padding
    pending = deferred
    while pending:
        rest = []
        for spec in pending:
            base = result.spec_dict.get(spec.storage_base)
            if base is None:
                raise InternalError("hillclimb: a storage base that is not planned")
            if spec.storage_base.storage_base is not None and spec.storage_base in pending:
                rest.append(spec)
                continue
            alloc = result.spec_dict[spec]
            alloc.mem_id = base.mem_id
            alloc.mem_offset = base.mem_offset + spec.storage_base_offset
        if len(rest) == len(pending):
            raise InternalError("hillclimb: circular storage bases")
        pending = rest
    # The Verifier's mem_obj_id rule: specs whose bytes overlap share an id.
    by_mem = {}
    for spec, alloc in result.spec_dict.items():
        by_mem.setdefault(alloc.mem_id, []).append((alloc.mem_offset, spec))
    obj = 0
    for items in by_mem.values():
        items.sort(key=lambda item: item[0])
        reach = -1
        for offset, spec in items:
            if offset >= reach:
                obj += 1
            reach = max(reach, offset + spec.allocated_memory)
            result.spec_dict[spec].mem_obj_id = obj - 1
    result.bufsizes = totals
    return result


def _hill_climb(ranges):
    """Offsets for (start, end, size) live ranges, ends inclusive: (offsets,
    total size)."""
    n = len(ranges)
    if n == 0:
        return [], 0
    start = [r[0] for r in ranges]
    end = [r[1] for r in ranges]
    size = [r[2] for r in ranges]
    neighbours = [[] for _ in range(n)]
    active = []
    for i in sorted(range(n), key=lambda i: (start[i], i)):
        active = [j for j in active if end[j] >= start[i]]
        for j in active:
            neighbours[i].append(j)
            neighbours[j].append(i)
        active.append(i)
    # Live bytes at each time: the lower bound, and each range's urgency.
    deltas = {}
    for i in range(n):
        deltas[start[i]] = deltas.get(start[i], 0) + size[i]
        deltas[end[i] + 1] = deltas.get(end[i] + 1, 0) - size[i]
    times = sorted(deltas)
    level, live = {}, 0
    for t in times:
        live += deltas[t]
        level[t] = live
    target = max(level.values())
    position = {t: k for k, t in enumerate(times)}
    levels = [level[t] for t in times]
    urgency = []
    for i in range(n):
        lo = position[start[i]]
        hi = position.get(end[i] + 1, len(times))
        urgency.append(max(levels[lo:hi]) if hi > lo else levels[lo])
    # Two starts: Regor's urgency order, and largest first (TFLite's greedy by
    # size, which on wide graphs often lands closer than the urgency order and
    # the few climbing steps a large graph can afford).
    starts = (sorted(range(n), key=lambda i: (-urgency[i], -(end[i] - start[i]), start[i],
                                              -size[i], i)),
              sorted(range(n), key=lambda i: (-size[i], start[i], -(end[i] - start[i]), i)))

    address = [0] * n
    top = [0] * n
    pred = [-1] * n
    turn = [0] * n

    def allocate(indices, limit):
        placed = [False] * n
        total = 0
        for k, i in enumerate(indices):
            blocks = sorted((address[j], top[j], j) for j in neighbours[i] if placed[j])
            a, p = 0, -1
            for b0, b1, j in blocks:
                if b0 >= a + size[i]:
                    break
                if b1 > a:
                    a, p = b1, j
            address[i], top[i], pred[i], turn[i] = a, a + size[i], p, k
            placed[i] = True
            total = max(total, a + size[i])
            if total > limit:
                return total
        return total

    best, order, best_state = float("inf"), None, None
    for candidate in starts:
        total = allocate(candidate, best)
        if total < best:
            best, order = total, candidate
            best_state = (list(address), list(top))
    address[:], top[:] = best_state
    allocate(order, float("inf"))  # pred and turn for the climb
    if best > target:
        work = n + sum(len(nb) for nb in neighbours)
        iterations = min(99999, HILL_CLIMB_WORK // max(work, 1))
        patience = min(5000, iterations)
        rng = random.Random(0)
        indices, best_indices = list(order), list(order)
        best_pred, best_turn = list(pred), list(turn)
        last_improvement = 0
        for it in range(iterations):
            # The bottleneck, and every range that shaped its placement.
            hi = max(range(n), key=lambda i: (top[i], -i))
            turns = set()

            def add_chain(i):
                turns.add(turn[i])
                while pred[i] != -1:
                    i = pred[i]
                    turns.add(turn[i])

            add_chain(hi)
            for j in neighbours[hi]:
                add_chain(j)
            turn_list = sorted(turns)
            if len(turn_list) < 2:
                break
            far = [t for t in turn_list
                   if not (start[indices[t]] <= end[hi] and start[hi] <= end[indices[t]])]
            if far and rng.randrange(100) < 30:
                a = far[rng.randrange(len(far))]
            else:
                a = turn_list[rng.randrange(len(turn_list))]
            b = turn_list[rng.randrange(len(turn_list))]
            if a == b:
                b = turn_list[-1]
            indices[a], indices[b] = indices[b], indices[a]
            size_now = allocate(indices, best)
            if size_now <= best:
                if size_now < best:
                    last_improvement = it
                best = size_now
                best_indices = list(indices)
                best_state = (list(address), list(top))
                best_pred, best_turn = list(pred), list(turn)
                if best <= target:
                    break
            else:
                indices = list(best_indices)
                address[:], top[:] = best_state[0], best_state[1]
                pred[:], turn[:] = best_pred, best_turn
            if it - last_improvement > patience:
                break
    return best_state[0], best
