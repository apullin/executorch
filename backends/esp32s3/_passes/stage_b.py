# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage B: int8 codes between claimed ops and across delegate boundaries.

On by default (Esp32Partitioner's stage_b; off under a float format_policy or
with stage_b=False). A program with no code region is byte for byte Stage A's.

Stage A (static_qdq.py) keeps fp32 wherever a delegate's tensor goes: a
claimed op applies its output quantize as fake-quant (EXT_OUT_QUANT writes
fl((yq - zo) * so)), the next op quantises that value again with its own
static record, and a portable op between two delegates reads and writes
fp32. Under the spec, codes that can travel as codes do: the producer writes
its int8 codes (EXT_OUT_QUANT, out_dtype 1), the consumer reads them
(EXT_ACT_QUANT, in_dtype 1) -- inside one delegate through an int8 chain
slot, across delegates as an int8 tensor the portable ops between them run
on. A quarter of the activation memory, no fp32 round trip, and Stage A's
codes bit for bit.

A code region starts at a PRODUCER: a static int8 quantize over a claimable
op's fused epilogue (the quantize the gate would take as that op's output
quantisation; Esp32OperatorSupport._code_out_problem), or an
espressif::quant_lut table over one, whose input quantize is recreated with
the scale its table was built from. Each reader of the dequantised value is
one of:

  sink   a claimable matmul or conv whose activation it is: the op reads the
         codes (its static record, in_dtype 1)
  move   view, permute, clone, squeeze/unsqueeze, dim-order copy: the same op
         on the codes
  clamp  relu, hardtanh or clamp, read by one static quantize: a clamp on the
         codes where Stage A's codes are one, else a code table
  pool   max_pool2d_with_indices whose values one quantize with the region's
         own constants reads: a max of codes
  pad    constant_pad_nd read by one quantize with the region's constants:
         the codes, padded with the value's code
  cat    cat whose every input is a code value on its quantize's grid: the
         codes concatenated
  table  espressif::quant_lut read by one quantize: espressif::
         quant_lut_codes, whose table is the code Stage A's quantize gives
         each entry (int8, at most 256 B)

and the code value a clamp, pool, pad, cat or table makes is followed the same
way. Anything else -- an add, a mean, a softmax, a graph output reading the
dequantised value -- needs fp32, and the region it touches keeps Stage A's
form whole, named in the report: converting part of it would put a portable
dequantise beside the producer's codes, more memory than Stage A.

The identity proof. Stage A hands the next op fl(fl(c - z) * s), which that
op quantises again; Stage B hands it c. Before a region converts, every edge
where Stage A requantises a dequantised code with the same constants -- each
sink, pool, pad and cat -- is checked over every code of the grid, with the
device's arithmetic (mirrors.py) and with torch's own dequantize and
quantize ops: q(fl(fl(c - z) * s)) must return c. A clamp's or a table's
codes are Stage A's by construction (evaluated by torch on every code). An
edge that fails keeps its region fp32, reported with the codes that differ.

Not converted, each reported by name with its region kept fp32: int16 codes (Arm
a16w8), uint8 codes (Arm's uint8 model IO, at the graph's boundary), a cat
input no claimed op writes, a selected-row head.

What the rewrite leaves: node.meta[STAGE_B] is "out" on each producer's
quantize (the gate claims it with its op: the op's output codes) and "in" on
each consumer's own dequantise, one per consumer (claimed with it: the op's
input codes). Stage A's absorption then runs on every pair this pass left
alone.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from dataclasses import dataclass, field

import numpy as np

STAGE_B = "esp32_stage_b"
OUT, IN = "out", "in"

LUT = ("espressif::quant_lut", "default")
CLAMPS = ("aten::relu", "aten::hardtanh", "aten::clamp")
POOL = ("aten::max_pool2d_with_indices", "default")
PAD = ("aten::constant_pad_nd", "default")
CAT = ("aten::cat", "default")
INT8 = (-128, 127)
# A claimable op's epilogue tail that no claimed op can write: a quantize the
# gate does not take as an output quantisation at all (a graph input's), not
# a Stage B refusal worth reporting.
NOT_A_PRODUCER = "requantize not over a claimed op's epilogue"
# Esp32OperatorSupport._code_out_problem's kind for an op dotting int16 codes.
INT16_OP = "int16 activations writing codes"


@dataclass(frozen=True)
class Grid:
    """A static int8 quantize's constants."""

    scale: float
    zero_point: int
    qmin: int
    qmax: int

    @classmethod
    def of(cls, node):
        """The grid of a quantize_per_tensor / dequantize_per_tensor node."""
        s, z, qmin, qmax = node.args[1:5]
        return cls(float(s), int(z), int(qmin), int(qmax))

    def args(self):
        import torch

        return (self.scale, self.zero_point, self.qmin, self.qmax, torch.int8)

    def codes(self):
        return np.arange(self.qmin, self.qmax + 1, dtype=np.int64)

    def __str__(self):
        return (f"({self.scale:.6g}, {self.zero_point}, [{self.qmin}, "
                f"{self.qmax}])")


def _schema(node):
    from executorch.backends.esp32s3._passes.static_qdq import _schema as schema

    return schema(node) if getattr(node, "op", None) == "call_function" else None


def identity_failures(grid):
    """The codes c of `grid` that Stage A's round trip does not return,
    q(fl(fl(c - z) * s)) != c: with the device's arithmetic (mirrors.py:
    what an output record writes and a static record quantises) or with
    torch's own dequantize_per_tensor and quantize_per_tensor. [] when it is
    the identity."""
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401  (the qd ops)

    from executorch.backends.esp32s3.quantizer import mirrors as qm

    codes = grid.codes()
    values = qm.dequantize_per_tensor(codes, grid.scale, grid.zero_point)
    back = qm.quantize_before_zp(values, np.float64(grid.scale),
                                 grid.zero_point, grid.qmin, grid.qmax)
    qd = torch.ops.quantized_decomposed
    t = torch.from_numpy(codes).to(torch.int8)
    with torch.no_grad():
        tv = qd.dequantize_per_tensor.default(t, *grid.args())
        tb = qd.quantize_per_tensor.default(tv, *grid.args())
    bad = (back != codes) | (tb.to(torch.int64).numpy() != codes)
    return [int(c) for c in codes[bad]]


def quantize_values(values, grid):
    """torch's quantize_per_tensor of float32 `values` onto `grid`: int64."""
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401  (the qd ops)

    qd = torch.ops.quantized_decomposed
    with torch.no_grad():
        t = torch.as_tensor(values, dtype=torch.float32)
        return qd.quantize_per_tensor.default(t, *grid.args()).to(
            torch.int64).numpy()


def code_map(target, grid_in, grid_out, args=(), kwargs=None):
    """Stage A's next code for every code of grid_in through elementwise op
    `target`: the graph's own dequantize, the op as torch evaluates it, the
    graph's own quantize onto grid_out. int64 [L]."""
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401  (the qd ops)

    qd = torch.ops.quantized_decomposed
    t = torch.from_numpy(grid_in.codes()).to(torch.int8)
    with torch.no_grad():
        x = qd.dequantize_per_tensor.default(t, *grid_in.args())
        y = target(x, *args, **(kwargs or {}))
    return quantize_values(y, grid_out)


def as_clamp(mapping, grid_in):
    """(lo, hi) when mapping[c - qmin] == min(max(c, lo), hi) for every code
    c of grid_in, else None."""
    lo, hi = int(mapping[0]), int(mapping[-1])
    if lo > hi or not np.array_equal(np.clip(grid_in.codes(), lo, hi),
                                     mapping):
        return None
    return lo, hi


def table_name(codes):
    """A content-addressed name for a code table (int8 [L])."""
    data = np.asarray(codes, dtype=np.int8).tobytes()
    digest = hashlib.sha256(data + struct.pack("<I", len(data))).hexdigest()
    return f"esp32_lutc_{digest[:16]}"


def _callable(target):
    """The ATen op an edge-dialect target wraps (callable on real tensors)."""
    return getattr(target, "_op", target)


# --- the analysis ------------------------------------------------------------------

@dataclass(eq=False)
class Code:
    """A code value on `grid`: `node` holds its codes once rewritten (None
    until the step that makes it has run), `views` read as its dequantised
    value in the graph as it stands."""

    node: object
    grid: Grid
    views: list
    region: int


@dataclass(eq=False)
class Step:
    """One rewrite of `node`, which reads code value `code` through `view`;
    `then` is the quantize it replaces and `new` the code value it makes."""

    kind: str
    node: object
    code: Code = None
    view: object = None
    then: object = None
    new: Code = None
    bounds: tuple = None       # clamp: (lo, hi) in codes
    table: object = None       # table: the codes; pool: its values' getitem
    inputs: list = None        # cat: [(view, Code)] in argument order
    pad_value: int = 0


@dataclass(eq=False)
class Producer:
    """A region's start: a quantize `q`, or a table op `lut` whose input
    quantize (constants `grid`, over `x`) the rewrite recreates."""

    q: object = None
    lut: object = None
    x: object = None
    grid: Grid = None
    made: object = None

    @property
    def name(self):
        return self.q.name if self.q is not None else self.lut.name


@dataclass
class StageBReport:
    """What rewrite_stage_b did: the producers that write codes, the
    consumers that read them, the ops between them now on codes and the code
    tables made; the regions kept fp32, each with its rule and where it fired;
    and every identity proof that failed, with its codes."""

    converted: list = field(default_factory=list)  # producer names
    consumers: list = field(default_factory=list)  # (dequantise, consumer)
    code_ops: list = field(default_factory=list)   # (node, kind)
    tables: list = field(default_factory=list)     # (node, constant name)
    kept: list = field(default_factory=list)       # (producer, rule, detail)
    outputs: list = field(default_factory=list)    # producers whose region
                                                   # ends at a graph output
    identity: list = field(default_factory=list)   # (edge, codes)
    proved: int = 0                                # edges proved identity

    def summary(self) -> str:
        parts = [f"stage B: {len(self.converted)} producer(s) write codes, "
                 f"{len(self.consumers)} consumer(s) read them, "
                 f"{len(self.code_ops)} op(s) between on codes"]
        if self.tables:
            parts.append(f"{len(self.tables)} code table(s)")
        if self.outputs:
            parts.append(f"{len(self.outputs)} region(s) end at a graph "
                         f"output, fp32 as the model gives it")
        if self.kept:
            parts.append(f"{len(self.kept)} kept fp32: " + "; ".join(
                f"{p} [{rule}] {detail}" for p, rule, detail in self.kept))
        return ", ".join(parts)


class _Plan:
    """Every producer's region, followed reader by reader in the graph as it
    stands (nothing changes until apply); regions a cat joins are one."""

    def __init__(self, ep, support, lut_scales):
        self.ep = ep
        self.support = support
        self.lut_scales = lut_scales
        self.parent = []           # union-find over region ids
        self.blocked = {}          # root -> [(rule, where, detail)]
        self.failures = {}         # root -> [(edge, codes)]
        self.edges = {}            # root -> edges proved
        self.steps = []            # every Step, in discovery order
        self.producers = {}        # region -> [Producer]
        self.cats = {}             # cat node -> ({view: Code}, region)
        self.cats_done = set()
        self.proved = {}           # Grid -> identity failures

    def new_region(self):
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def root(self, r):
        while self.parent[r] != r:
            self.parent[r] = self.parent[self.parent[r]]
            r = self.parent[r]
        return r

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[b] = a
            self.blocked.setdefault(a, []).extend(self.blocked.pop(b, []))
            self.failures.setdefault(a, []).extend(self.failures.pop(b, []))
            self.edges[a] = self.edges.get(a, 0) + self.edges.pop(b, 0)
        return a

    def block(self, region, rule, where, detail):
        self.blocked.setdefault(self.root(region), []).append(
            (rule, where, detail))

    def prove(self, region, grid, edge):
        """Stage A's requantise over `grid` returns every code, or the region
        keeps fp32 with the codes that differ."""
        if grid not in self.proved:
            self.proved[grid] = identity_failures(grid)
        root = self.root(region)
        self.edges[root] = self.edges.get(root, 0) + 1
        if self.proved[grid]:
            self.failures.setdefault(root, []).append(
                (edge, self.proved[grid]))

    def views_of(self, q):
        from executorch.backends.esp32s3._passes.static_qdq import static_pair

        return [u for u in q.users if static_pair(u) is q]

    def follow(self, code):
        """Every reader of code's dequantised views, depth first."""
        stack = [code]
        while stack:
            c = stack.pop()
            for view in list(c.views):
                for user in list(view.users):
                    made = self.reader(c, view, user)
                    if made is not None:
                        stack.append(made)

    def quantize_after(self, region, node, what):
        """The one static int8 quantize reading `node`, whose own readers are
        all its dequantises; else the region blocks (None)."""
        from executorch.backends.esp32s3._passes.static_qdq import (
            is_static_q,
            scheme_problem,
            static_pair,
        )

        import torch

        users = list(node.users)
        if len(users) == 1 and _schema(users[0]) == LUT:
            # fold_quant_luts took the quantize in front of the table into
            # the table: its input is no claimed op's output, and Stage B
            # makes no code table of an op on codes.
            self.block(region, "stage-b-table-input", users[0].name,
                       f"a table op reads {what}; a table after an op on "
                       f"codes is not part of v1")
            return None
        if len(users) != 1 or not is_static_q(users[0]):
            self.block(region, "stage-b-unquantised", node.name,
                       f"{what} is not read by one static quantize alone")
            return None
        q2 = users[0]
        problem = scheme_problem(q2)
        if problem is not None or q2.args[5] != torch.int8:
            self.block(region, "stage-b-dtype", q2.name,
                       problem[0] if problem else
                       f"{q2.args[5]} codes (Stage B v1 carries int8 only)")
            return None
        others = [u for u in q2.users if static_pair(u) is not q2]
        if others:
            self.block(region, "stage-b-codes-read", q2.name,
                       f"{others[0].name} reads its codes")
            return None
        return q2

    def reader(self, code, view, user):
        """Classify one reader of code's dequantised `view`: record its step
        and return the code value it makes (to follow), or block the region."""
        from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
            _activation_arg_index,
        )
        from executorch.backends.esp32s3._passes.static_qdq import is_move

        import torch

        region = code.region
        if user.op == "output":
            self.block(region, "stage-b-graph-output", view.name,
                       "the graph outputs the dequantised value (the "
                       "model's own outputs stay as it gives them)")
            return None
        schema = _schema(user)
        if schema is None:
            self.block(region, "stage-b-reader", user.name,
                       f"{user.op} reads the dequantised value")
            return None
        if schema == CAT and isinstance(user.args[0], (list, tuple)) and any(
                v is view for v in user.args[0]):
            got, cat_region = self.cats.setdefault(user, ({},
                                                          self.new_region()))
            got[view] = code
            self.join(cat_region, region)
            return None
        if sum(a is view for a in user.args) != 1 or any(
                v is view for v in user.kwargs.values()):
            self.block(region, "stage-b-reader", user.name,
                       f"{schema[0]} reads the dequantised value other than "
                       f"as its one input")
            return None
        idx = _activation_arg_index(user)
        if idx is not None and idx < len(user.args) and user.args[idx] is view:
            return self._sink(code, view, user)
        if is_move(user) and user.args[0] is view:
            dtype = user.kwargs.get("dtype")
            if dtype not in (None, torch.float32):
                self.block(region, "stage-b-reader", user.name,
                           f"a copy to {dtype}")
                return None
            new = Code(None, code.grid, [user], region)
            self.steps.append(Step("move", user, code, view, new=new))
            return new
        if user.args[0] is not view:
            self.block(region, "stage-b-reader", user.name,
                       f"{schema[0]} reads the dequantised value as an "
                       f"operand other than its input")
            return None
        if schema[0] in CLAMPS:
            return self._clamp(code, view, user)
        if schema == POOL:
            return self._pool(code, view, user)
        if schema == PAD:
            return self._pad(code, view, user)
        self.block(region, "stage-b-reader", user.name,
                   f"{schema[0].split('::')[-1]} needs fp32 (not an op on "
                   f"codes)")
        return None

    def _sink(self, code, view, user):
        from executorch.backends.esp32s3.operator_support.patterns import _selected_mm_pattern

        region = code.region
        if _selected_mm_pattern(user, self.ep) is not None:
            self.block(region, "stage-b-selected-head", user.name,
                       "a selected-row head reads a partition input only; "
                       "codes into one are not implemented")
            return None
        problem = self.support._quantised_problem(user)
        if problem is not None:
            self.block(region, "stage-b-consumer", user.name,
                       f"{problem[0]}: {problem[1]}")
            return None
        self.prove(region, code.grid, f"{view.name} -> {user.name}")
        self.steps.append(Step("sink", user, code, view))
        return None

    def _clamp(self, code, view, user):
        region = code.region
        q2 = self.quantize_after(region, user, f"{user.name}'s value")
        if q2 is None:
            return None
        grid2 = Grid.of(q2)
        mapping = code_map(_callable(user.target), code.grid, grid2,
                           tuple(user.args[1:]), dict(user.kwargs))
        new = Code(None, grid2, self.views_of(q2), region)
        bounds = as_clamp(mapping, code.grid)
        if bounds is not None:
            self.steps.append(Step("clamp", user, code, view, then=q2, new=new,
                                   bounds=bounds))
        else:
            self.steps.append(Step("table", user, code, view, then=q2, new=new,
                                   table=mapping))
        return new

    def _pool(self, code, view, pool):
        region = code.region
        values = [u for u in pool.users
                  if u.target is operator.getitem and u.args[1] == 0]
        others = [u for u in pool.users if u not in values]
        if len(values) != 1 or any(u.target is not operator.getitem or u.users
                                   for u in others):
            self.block(region, "stage-b-reader", pool.name,
                       "its indices are read, or its values more than once")
            return None
        q2 = self.quantize_after(region, values[0], "the pooled value")
        if q2 is None:
            return None
        grid2 = Grid.of(q2)
        if grid2 != code.grid:
            self.block(region, "stage-b-requantize", pool.name,
                       f"max_pool between two grids, {code.grid} -> {grid2}")
            return None
        self.prove(region, code.grid, f"{view.name} -> {pool.name}")
        new = Code(None, grid2, self.views_of(q2), region)
        self.steps.append(Step("pool", pool, code, view, then=q2, new=new,
                               table=values[0]))
        return new

    def _pad(self, code, view, pad):
        region = code.region
        q2 = self.quantize_after(region, pad, "the padded value")
        if q2 is None:
            return None
        grid2 = Grid.of(q2)
        if grid2 != code.grid:
            self.block(region, "stage-b-requantize", pad.name,
                       f"pad between two grids, {code.grid} -> {grid2}")
            return None
        value = (pad.args[2] if len(pad.args) > 2
                 else pad.kwargs.get("value", 0.0))
        self.prove(region, code.grid, f"{view.name} -> {pad.name}")
        new = Code(None, grid2, self.views_of(q2), region)
        self.steps.append(Step(
            "pad", pad, code, view, then=q2, new=new,
            pad_value=int(quantize_values([float(value)], grid2)[0])))
        return new

    def finish_cats(self):
        """A cat whose every input is a code value on its quantize's grid
        becomes a cat of codes, followed like any other code value; one with
        an input no claimed op writes keeps its region fp32."""
        progress = True
        while progress:
            progress = False
            for cat, (got, cat_region) in list(self.cats.items()):
                if cat in self.cats_done or not all(
                        v in got for v in cat.args[0]):
                    continue
                self.cats_done.add(cat)
                progress = True
                q2 = self.quantize_after(cat_region, cat, "the concatenation")
                if q2 is None:
                    continue
                grid2 = Grid.of(q2)
                if any(code.grid != grid2 for code in got.values()):
                    self.block(cat_region, "stage-b-requantize", cat.name,
                               "a cat input on another grid than its output")
                    continue
                self.prove(cat_region, grid2, f"{cat.name}'s inputs")
                new = Code(None, grid2, self.views_of(q2), cat_region)
                self.steps.append(Step("cat", cat, then=q2, new=new, inputs=[
                    (v, got[v]) for v in cat.args[0]]))
                self.follow(new)
        for cat, (got, cat_region) in self.cats.items():
            if cat not in self.cats_done:
                missing = [v.name for v in cat.args[0] if v not in got]
                self.block(cat_region, "stage-b-cat-input", cat.name,
                           f"input(s) {', '.join(missing)} are not codes a "
                           f"claimed op writes")


def _lut_scale(lut, lut_scales):
    """s1 for a quant_lut node: the scale its table was built from
    (fold_quant_lut_pass.LutBuild.scale, matched by the table constant and the
    node's quantize constants), provided it gives the node's own multiplier."""
    from executorch.backends.esp32s3.quantizer import mirrors as qm

    key = (lut.args[1].name, float(lut.args[2]), int(lut.args[3]),
           int(lut.args[4]), int(lut.args[5]))
    scales = {s for k, s in lut_scales if k == key}
    if len(scales) != 1:
        return None
    s1 = scales.pop()
    return s1 if float(qm.reciprocal(s1)) == key[1] else None


def lut_scales_of(report, ep):
    """[(key, s1)] from a fold_quant_luts report, keyed as _lut_scale reads a
    node: (table placeholder name, inv_scale, zero point, qmin, qmax)."""
    if report is None:
        return []
    names = {}
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            target = ep.graph_signature.inputs_to_lifted_tensor_constants.get(
                node.name)
            if target is not None:
                names.setdefault(target, node.name)
    return [((names.get(build.name, build.name), float(build.inv_scale),
              int(build.zero_point), int(build.quant_min),
              int(build.quant_max)), float(build.scale))
            for _, build in report.tables]


def find_producers(plan):
    """Start every region (and follow it); returns the quantizes over a
    claimable op's epilogue that start none, with why: int16 or uint8 codes,
    or what keeps the op from writing them."""
    from executorch.backends.esp32s3._passes.fold_quant_lut_pass import _constant
    from executorch.backends.esp32s3._passes.static_qdq import is_static_q

    import torch

    support = plan.support
    kept = []
    for node in list(plan.ep.graph_module.graph.nodes):
        if is_static_q(node):
            problem = support._code_out_problem(node)
            if problem is not None and problem[0] == NOT_A_PRODUCER:
                continue
            if node.args[5] == torch.int16 or (
                    problem is not None and problem[0] == INT16_OP):
                # Arm a16w8: int16 codes, or an op that dots them.
                kept.append((node.name, "stage-b-int16",
                             f"{node.args[5]} codes"
                             f"{' from an op dotting int16 codes' if problem else ''}"
                             f": Stage B v1 carries int8 codes only, so this "
                             f"boundary stays fp32"))
                continue
            if problem is not None:
                kept.append((node.name, "stage-b-producer",
                             f"{problem[0]}: {problem[1]}"))
                continue
            if node.args[5] != torch.int8:
                kept.append((node.name, "stage-b-dtype",
                             f"{node.args[5]} codes: Stage B v1 carries int8 "
                             f"codes only, so this boundary stays fp32"))
                continue
            region = plan.new_region()
            plan.producers.setdefault(region, []).append(Producer(q=node))
            views = plan.views_of(node)
            others = [u for u in node.users
                      if u not in views and u.op != "output"]
            if others:
                plan.block(region, "stage-b-codes-read", node.name,
                           f"{others[0].name} reads its codes")
                continue
            plan.follow(Code(node, Grid.of(node), views, region))
        elif _schema(node) == LUT:
            if support._code_out_tail_problem(node.args[0]) is not None:
                continue
            region = plan.new_region()
            qmin, qmax = int(node.args[4]), int(node.args[5])
            s1 = _lut_scale(node, plan.lut_scales)
            producer = Producer(lut=node, x=node.args[0])
            plan.producers.setdefault(region, []).append(producer)
            if s1 is None or not INT8[0] <= qmin < qmax <= INT8[1]:
                plan.block(region, "stage-b-lut-input", node.name,
                           "its input quantize is not an int8 one whose "
                           "scale its table's build recorded")
                continue
            producer.grid = Grid(s1, int(node.args[3]), qmin, qmax)
            q2 = plan.quantize_after(region, node, "the table's value")
            if q2 is None:
                continue
            grid2 = Grid.of(q2)
            mapping = quantize_values(_constant(plan.ep, node.args[1]), grid2)
            new = Code(None, grid2, plan.views_of(q2), region)
            plan.steps.append(Step("table", node,
                                   Code(None, producer.grid, [], region),
                                   then=q2, new=new, table=mapping))
            plan.follow(new)
    return kept


# --- the rewrite --------------------------------------------------------------------

def _val(node, dtype):
    val = node.meta.get("val")
    return val.to(dtype) if val is not None else None


def _code_table(ep, graph, codes):
    """The int8 table constant for `codes`, one per distinct table."""
    import torch
    from executorch.backends.transforms.utils import create_constant_placeholder
    from torch.export.graph_signature import InputKind

    data = torch.from_numpy(np.asarray(codes, dtype=np.int8).copy())
    first = next(n for n in graph.nodes if n.op == "placeholder")
    with graph.inserting_before(first):
        return create_constant_placeholder(
            ep, graph, table_name(codes), InputKind.CONSTANT_TENSOR, data)


def apply(ep, steps, producers, report):
    """Rewrite one convertible region, its steps in discovery order (each
    reads a code value an earlier step or a producer made)."""
    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    from executorch.backends.esp32s3._passes import fold_quant_lut_pass  # noqa: F401  (registers espressif::quant_lut_codes)

    graph = ep.graph_module.graph
    edge = exir_ops.edge
    replaced_tables = []
    for p in producers:
        if p.q is not None:
            p.q.meta[STAGE_B] = OUT
            p.made = p.q
        else:
            with graph.inserting_before(p.lut):
                p.made = graph.call_function(
                    edge.quantized_decomposed.quantize_per_tensor.default,
                    (p.x, *p.grid.args()))
            p.made.meta["val"] = _val(p.x, torch.int8)
            p.made.meta[STAGE_B] = OUT
            replaced_tables.append(p.lut.args[1])
        report.converted.append(p.name)

    def node_of(step):
        if step.code.node is not None:
            return step.code.node
        return next(p.made for p in producers if p.lut is step.node)

    for step in steps:
        n = step.node
        if step.kind == "sink":
            source = node_of(step)
            with graph.inserting_before(n):
                d = graph.call_function(
                    edge.quantized_decomposed.dequantize_per_tensor.default,
                    (source, *step.code.grid.args()))
            # fp32 values of the codes' shape (the view may be a move an
            # earlier step already turned into codes).
            d.meta["val"] = _val(source, torch.float32)
            d.meta[STAGE_B] = IN
            n.replace_input_with(step.view, d)
            report.consumers.append((d.name, n.name))
            continue
        if step.kind == "move":
            n.replace_input_with(step.view, node_of(step))
            if n.kwargs.get("dtype") is not None:
                n.kwargs = {**n.kwargs, "dtype": torch.int8}
            n.meta["val"] = _val(n, torch.int8)
            step.new.node = n
            report.code_ops.append((n.name, "move"))
            continue
        if step.kind in ("clamp", "table"):
            with graph.inserting_before(step.then):
                if step.kind == "clamp":
                    made = graph.call_function(
                        edge.aten.clamp.default, (node_of(step), *step.bounds))
                else:
                    table = _code_table(ep, graph, step.table)
                    grid = step.code.grid
                    made = graph.call_function(
                        edge.espressif.quant_lut_codes.default,
                        (node_of(step), table, grid.qmin, grid.qmax))
                    report.tables.append((made.name, table.name))
        elif step.kind == "pool":
            n.replace_input_with(step.view, node_of(step))
            values, indices = n.meta["val"]
            n.meta["val"] = (values.to(torch.int8), indices)
            made = step.table
            made.meta["val"] = _val(made, torch.int8)
        elif step.kind == "pad":
            n.replace_input_with(step.view, node_of(step))
            n.args = (*n.args[:2], step.pad_value)
            n.kwargs = {k: v for k, v in n.kwargs.items() if k != "value"}
            n.meta["val"] = _val(n, torch.int8)
            made = n
        else:  # cat
            n.args = ([code.node for _, code in step.inputs], *n.args[1:])
            n.meta["val"] = _val(n, torch.int8)
            made = n
        made.meta["val"] = step.then.meta.get("val")
        step.then.replace_all_uses_with(made)
        step.new.node = made
        report.code_ops.append((made.name, step.kind))
    for p in producers:
        if p.lut is not None and not p.lut.users:
            graph.erase_node(p.lut)
    return replaced_tables


def rewrite_stage_b(ep, support, lut_report=None) -> StageBReport:
    """Convert every code region of `ep` (an edge-dialect ExportedProgram
    whose duplicate pairs are merged: static_qdq.absorb_static_qdq calls this
    under Esp32Partitioner(stage_b=True)) whose readers all take codes, in
    place. `support` is the partitioner's gate, `lut_report` the
    fold_quant_luts report of the same export (each table's input scale)."""
    from executorch.backends.transforms.utils import delete_constant_placeholder

    graph = ep.graph_module.graph
    report = StageBReport()
    plan = _Plan(ep, support, lut_scales_of(lut_report, ep))
    report.kept += find_producers(plan)
    plan.finish_cats()
    regions, steps = {}, {}
    for region, producers in plan.producers.items():
        regions.setdefault(plan.root(region), []).extend(producers)
    for step in plan.steps:
        code = step.code if step.code is not None else step.new
        steps.setdefault(plan.root(code.region), []).append(step)
    tables = []
    for root, producers in regions.items():
        names = ", ".join(p.name for p in producers)
        if plan.blocked.get(root):
            # A region the graph's own output stops is the model's output,
            # not a refused boundary; any other reason is reported first.
            reasons = sorted(plan.blocked[root],
                             key=lambda b: b[0] == "stage-b-graph-output")
            rule, where, detail = reasons[0]
            if rule == "stage-b-graph-output":
                report.outputs.append(names)
            else:
                report.kept.append((names, rule, f"{where}: {detail}"))
            continue
        if plan.failures.get(root):
            report.identity += plan.failures[root]
            edge, codes = plan.failures[root][0]
            report.kept.append((
                names, "stage-b-identity",
                f"{edge}: q(fl(fl(c - z) * s)) != c for code(s) "
                f"{codes[:8]}{' ...' if len(codes) > 8 else ''}"))
            continue
        report.proved += plan.edges.get(root, 0)
        tables += apply(ep, steps.get(root, []), producers, report)
    if report.converted:
        graph.eliminate_dead_code()
        for table in tables:
            if not table.users:
                delete_constant_placeholder(ep, table)
        graph.lint()
        ep.graph_module.recompile()
    return report
