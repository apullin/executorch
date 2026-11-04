# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""memory_plan="esp32": in-place reuse, the order search and the hill climb.

Every in-place kernel is proven alias-safe on the host runner: a program
whose kernel writes its output over a dying operand (the emitted plan is
checked to really share the bytes) gives the same output bits as ExecuTorch's
plan, at shapes, dtypes and layouts that take each path of our override and
of the portable kernel. The planner's refusals (graph inputs and outputs,
views, mutable buffers, broadcasts, dtype changes, live operands) and its
never-worse rule are checked on the plans themselves.

    python -m pytest test_memory_planning.py -q

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import operator
import random
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    memory,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    DelegateScratchSpec,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.error import InternalError
from executorch.exir.memory_planning import greedy
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.tensor import TensorSpec

from executorch.backends.esp32s3 import memory_planning
from executorch.backends.esp32s3._passes import fold_quant_lut_pass  # noqa: F401 -- defines espressif::quant_lut
from executorch.backends.esp32s3.test.host import host_runner



@pytest.fixture(scope="session")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def lower(module, inputs, plan=None, **options):
    """to_edge (no partitioner: every op portable) -> to_executorch with
    `plan`'s memory planning. Returns (program, the planning pass)."""
    module.eval()
    edge = to_edge(torch.export.export(module, inputs))
    planner = memory_planning.planning_pass(plan, **options)
    return edge.to_executorch(ExecutorchBackendConfig(memory_planning_pass=planner)), planner


def arena(program):
    return sum(program.executorch_program.execution_plan[0].non_const_buffer_sizes[1:])


def aliased(program):
    """(operator, output value, operand value) for each kernel call whose
    output shares its planned bytes with one of its tensor operands."""
    plan = program.executorch_program.execution_plan[0]
    names = [op.name + (f".{op.overload}" if op.overload else "") for op in plan.operators]

    def where(index):
        value = plan.values[index].val
        info = getattr(value, "allocation_info", None)
        if info is None or 0 in value.sizes:
            return None  # an empty tensor may sit anywhere, over anything
        return info.memory_id, info.memory_offset_low, info.memory_offset_high

    found = []
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if type(call).__name__ != "KernelCall":
                continue
            tensors = [i for i in dict.fromkeys(call.args)
                       if type(plan.values[i].val).__name__ == "Tensor"]
            if len(tensors) < 2:
                continue
            out = tensors[-1]
            for i in tensors[:-1]:
                if where(i) is not None and where(i) == where(out):
                    found.append((names[call.op_index], out, i))
    return found


def run_host(runner, program, inputs, work):
    """Every output of forward on the host runner, as bytes (inputs written
    in the memory order the program declares)."""
    work.mkdir(parents=True, exist_ok=True)
    pte = work / "model.pte"
    pte.write_bytes(program.buffer)
    plan = program.executorch_program.execution_plan[0]
    command = [str(runner), "--pte", str(pte), "--method", "forward"]
    for i, (x, index) in enumerate(zip(inputs, plan.inputs)):
        order = list(plan.values[index].val.dim_order)
        if order and order != sorted(order):
            x = x.permute(*order)
        path = work / f"in{i}.bin"
        np.ascontiguousarray(x.contiguous().numpy()).tofile(path)
        command += ["--input", str(path)]
    outs = [work / f"out{k}.bin" for k in range(len(plan.outputs))]
    for out in outs:
        out.unlink(missing_ok=True)
        command += ["--output", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    return [out.read_bytes() for out in outs]


class Dying(torch.nn.Module):
    """op(cat([x, x]), *rest): the cat is a planned intermediate twice x's
    size that dies at op, so writing op's output over it lowers the peak and
    the planner takes it."""

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, *rest):
        return self.op(torch.cat([x, x]), *rest)


class DyingImage(Dying):
    """op(upsample(x), *rest): Dying for a channels-last x, whose nearest
    upsampling to twice its height is the dying intermediate. ExecuTorch's
    portable cat takes the default dim order only (op_cat.cpp, ExecuTorch
    #21865); upsample_nearest2d has a channels-last loop."""

    def forward(self, x, *rest):
        return self.op(F.interpolate(x, scale_factor=(2.0, 1.0), mode="nearest"), *rest)


class Lut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.linspace(-3.0, 5.0, 256) ** 3)

    def forward(self, x):
        return torch.ops.espressif.quant_lut(torch.cat([x, x]), self.table, 16.0, 3, -128, 127)


def _twice(shape):
    return (2 * shape[0],) + tuple(shape[1:])


def f32(*shape, double=False):
    return torch.randn(*(_twice(shape) if double else shape))


def i32(*shape, double=False):
    return torch.randint(-1000, 1000, _twice(shape) if double else shape, dtype=torch.int32)


def channels_last(*shape):
    return torch.randn(*shape).to(memory_format=torch.channels_last)


# (case, module, inputs, the operator that must write over neg's result).
# Shapes [5] and [3, 37] give short and odd-tail loops (at 16-byte alignment a
# [1] tensor and its double are one size, so nothing would gain); float32
# alpha=1 same-shape operands take shim_add_out's and shim_mul_out's own
# loops, alpha != 1, int32 and channels-last ones the portable kernels'.
KERNELS = []
for shape in ((5,), (3, 37)):
    tag = "x".join(map(str, shape))
    KERNELS += [
        (f"add lhs {tag}", Dying(lambda t, y: t + y), (f32(*shape), f32(*shape, double=True)), "aten::add.out"),
        (f"add rhs {tag}", Dying(lambda t, y: y + t), (f32(*shape), f32(*shape, double=True)), "aten::add.out"),
        (f"add twice {tag}", Dying(lambda t: t + t), (f32(*shape),), "aten::add.out"),
        (f"add alpha {tag}", Dying(lambda t, y: torch.add(t, y, alpha=2)),
         (f32(*shape), f32(*shape, double=True)), "aten::add.out"),
        (f"add int32 {tag}", Dying(lambda t, y: t + y), (i32(*shape), i32(*shape, double=True)), "aten::add.out"),
        (f"sub {tag}", Dying(lambda t, y: t - y), (f32(*shape), f32(*shape, double=True)), "aten::sub.out"),
        (f"sub rhs alpha {tag}", Dying(lambda t, y: torch.sub(y, t, alpha=3)),
         (f32(*shape), f32(*shape, double=True)), "aten::sub.out"),
        (f"sub int32 {tag}", Dying(lambda t, y: t - y), (i32(*shape), i32(*shape, double=True)), "aten::sub.out"),
        (f"mul {tag}", Dying(lambda t, y: t * y), (f32(*shape), f32(*shape, double=True)), "aten::mul.out"),
        (f"mul rhs {tag}", Dying(lambda t, y: y * t), (f32(*shape), f32(*shape, double=True)), "aten::mul.out"),
        (f"mul int32 {tag}", Dying(lambda t, y: t * y), (i32(*shape), i32(*shape, double=True)), "aten::mul.out"),
        (f"relu {tag}", Dying(torch.relu), (f32(*shape),), "aten::relu.out"),
        (f"relu int32 {tag}", Dying(torch.relu), (i32(*shape),), "aten::relu.out"),
        (f"clamp {tag}", Dying(lambda t: torch.clamp(t, -0.5, 0.25)), (f32(*shape),),
         "aten::clamp.out"),
        (f"clamp min {tag}", Dying(lambda t: torch.clamp(t, min=0.1)), (f32(*shape),),
         "aten::clamp.out"),
        (f"hardtanh {tag}", Dying(lambda t: F.hardtanh(t, -0.3, 0.7)), (f32(*shape),),
         "aten::hardtanh.out"),
        (f"sigmoid {tag}", Dying(torch.sigmoid), (f32(*shape),), "aten::sigmoid.out"),
        (f"tanh {tag}", Dying(torch.tanh), (f32(*shape),), "aten::tanh.out"),
        (f"quant_lut {tag}", Lut(), (f32(*shape),), "espressif::quant_lut.out"),
    ]
KERNELS += [
    ("add channels-last", DyingImage(lambda t, y: t + y),
     (channels_last(1, 4, 3, 5), channels_last(1, 4, 6, 5)), "aten::add.out"),
    ("mul channels-last", DyingImage(lambda t, y: t * y),
     (channels_last(1, 4, 3, 5), channels_last(1, 4, 6, 5)), "aten::mul.out"),
    ("sigmoid channels-last", DyingImage(torch.sigmoid), (channels_last(1, 4, 3, 5),),
     "aten::sigmoid.out"),
]


@pytest.mark.parametrize("case,module,inputs,op", KERNELS, ids=[k[0] for k in KERNELS])
def test_kernel_alias_safe(runner, tmp_path, case, module, inputs, op):
    """The kernel writes its output over its dying operand (the emitted plan
    says so), and the host runner's output bits equal ExecuTorch's plan's."""
    torch.manual_seed(0)
    off, _ = lower(module, inputs)
    on, planner = lower(module, inputs, "esp32", reorder=False)
    assert op in [name for name, _, _ in aliased(on)], (case, aliased(on), planner.plans)
    assert not aliased(off)
    assert arena(on) <= arena(off)
    got_off = run_host(runner, off, inputs, tmp_path / "off")
    got_on = run_host(runner, on, inputs, tmp_path / "on")
    assert got_on == got_off, case
    eager = module(*inputs)
    assert len(got_on[0]) == eager.numel() * eager.element_size()


def test_alias_chain(runner, tmp_path):
    """Three outputs in a row over one buffer: storage bases of storage bases."""
    module = Dying(lambda t: torch.relu(torch.sigmoid(torch.tanh(t))))
    inputs = (f32(4, 64),)
    off, _ = lower(module, inputs)
    on, planner = lower(module, inputs, "esp32")
    assert len(aliased(on)) == 3, aliased(on)
    assert planner.plans[0].inplace == 3
    assert run_host(runner, on, inputs, tmp_path / "on") == run_host(
        runner, off, inputs, tmp_path / "off")


# ------------------------------------------------------------ refusals


class TwoOut(torch.nn.Module):
    def forward(self, x):
        t = torch.neg(x)
        return t, torch.relu(t)


class ViewIn(torch.nn.Module):
    def forward(self, x):
        return torch.relu(torch.neg(x).view(-1))


class Broadcast(torch.nn.Module):
    def forward(self, x, b):
        return torch.neg(x) + b


class DtypeChange(torch.nn.Module):
    def forward(self, x, y):
        return torch.neg(x) + y  # int32 + float32 -> float32


class StillLive(torch.nn.Module):
    def forward(self, x):
        t = torch.neg(x)
        return torch.relu(t) + t


class NotAllowlisted(torch.nn.Module):
    def forward(self, x):
        return torch.exp(torch.neg(x))


class Constant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16))

    def forward(self, x):
        return torch.relu(self.weight) + x


class Buffer(torch.nn.Module):
    """A mutable buffer read (relu), then updated in place: nothing links the
    read to the write-back but the order."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(16))

    def forward(self, x):
        r = torch.relu(self.state)
        self.state.add_(x)
        return r + self.state


def _ops(program):
    return sorted(name for name, _, _ in aliased(program))


def test_refuses_graph_input():
    on, _ = lower(torch.nn.ReLU(), (f32(16),), "esp32")
    assert aliased(on) == []


def test_refuses_graph_output():
    on, _ = lower(TwoOut(), (f32(16),), "esp32")
    assert aliased(on) == []


def test_refuses_view():
    on, _ = lower(ViewIn(), (f32(4, 4),), "esp32")
    assert aliased(on) == []


def test_refuses_broadcast():
    on, _ = lower(Broadcast(), (f32(4, 8), f32(1, 8)), "esp32")
    assert aliased(on) == []


def test_refuses_dtype_change():
    on, _ = lower(DtypeChange(), (i32(16), f32(16)), "esp32")
    assert aliased(on) == []


def test_refuses_live_operand():
    """relu may not write over t (the add reads it later); the add may."""
    on, _ = lower(StillLive(), (f32(16),), "esp32")
    assert _ops(on) == ["aten::add.out"]


def test_refuses_unlisted_op_and_constants():
    on, _ = lower(NotAllowlisted(), (f32(16),), "esp32")
    assert aliased(on) == []
    # relu never writes over the weight; the add may write over relu's result.
    on, _ = lower(Constant(), (f32(16),), "esp32")
    assert _ops(on) == ["aten::add.out"]


def test_mutable_buffer(runner, tmp_path):
    """The buffer is never written over or written into (only the last add,
    over relu's dying result, is in place), the order search keeps the
    buffer's read before its write-back with an order edge, and the output
    bits are unchanged."""
    inputs = (f32(16),)
    off, _ = lower(Buffer(), inputs)
    on, planner = lower(Buffer(), inputs, "esp32")
    plan = planner.plans[0]
    assert "order edge" in plan.search, plan.describe()
    assert _ops(on) in ([], ["aten::add.out"]), aliased(on)
    assert run_host(runner, on, inputs, tmp_path / "on") == run_host(
        runner, off, inputs, tmp_path / "off")


def test_written_specs():
    """index_put_ (the KV cache write after the reinplace pass) and copy_
    write their self; out arguments are not in-place writes."""
    class Cache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(8, 4))

        def forward(self, row, pos):
            self.cache.index_put_((pos,), row)
            return self.cache.sum(0)

    edge = to_edge(torch.export.export(Cache(), (torch.randn(1, 4), torch.tensor([3]))))
    program = edge.to_executorch(ExecutorchBackendConfig(run_reinplace_pass=True))
    gm = program.exported_program().graph_module
    nodes = list(gm.graph.nodes)
    written = memory_planning.written_specs(nodes)
    cache = next(n for n in nodes if n.op == "placeholder" and "cache" in n.name)
    assert id(cache.meta["spec"]) in written
    outs = [n for n in nodes if n.op == "call_function" and "sum" in str(n.target)]
    assert outs and all(id(n.meta["spec"]) not in written for n in outs)


# ------------------------------------------------------------ the plan


class Branches(torch.nn.Module):
    """Two branches, each a big temporary then a small sum: the graph order
    holds both temporaries at once, branch after branch holds one."""

    def forward(self, x):
        a = torch.cat([x, x, x, x], 1)
        b = torch.cat([x, x, x, x], 1)
        return a.sum(1, keepdim=True) * 2 + b.sum(1, keepdim=True)


class Residual(torch.nn.Module):
    """A residual add whose skip value is read last in the graph's order."""

    def forward(self, x):
        h = torch.neg(x)
        y = torch.exp(h)
        z = torch.sin(h)
        return y + z


def test_reorder_finds_the_smaller_order(runner, tmp_path):
    inputs = (f32(4, 256),)
    off, _ = lower(Branches(), inputs)
    on, planner = lower(Branches(), inputs, "esp32")
    plan = planner.plans[0]
    assert plan.reordered and "exact" in plan.search, plan.describe()
    assert arena(on) < arena(off)
    assert arena(on) == plan.bound
    assert run_host(runner, on, inputs, tmp_path / "on") == run_host(
        runner, off, inputs, tmp_path / "off")


def test_never_worse_and_identical_when_nothing_gains():
    """With nothing to gain the program is ExecuTorch's, byte for byte."""
    for module, inputs in ((torch.nn.ReLU(), (f32(16),)), (TwoOut(), (f32(16),)),
                           (NotAllowlisted(), (f32(16),))):
        off, _ = lower(module, inputs)
        on, planner = lower(module, inputs, "esp32")
        assert planner.plans[0].choice == "ExecuTorch", planner.plans[0].describe()
        assert on.buffer == off.buffer


def test_every_case_never_worse():
    cases = [(m, i) for _, m, i, _ in KERNELS] + [
        (Branches(), (f32(4, 256),)), (Residual(), (f32(64),)), (StillLive(), (f32(16),)),
        (Buffer(), (f32(16),)), (TwoOut(), (f32(16),))]
    for module, inputs in cases:
        off, _ = lower(module, inputs)
        on, planner = lower(module, inputs, "esp32")
        assert arena(on) <= arena(off)
        assert planner.plans[0].arena == arena(on)
        assert planner.plans[0].baseline == arena(off)


def test_deterministic():
    inputs = (f32(4, 256),)
    a, _ = lower(Branches(), inputs, "esp32")
    b, _ = lower(Branches(), inputs, "esp32")
    assert a.buffer == b.buffer


def test_switch_off_is_executorch():
    planner = memory_planning.planning_pass(None)
    assert type(planner) is MemoryPlanningPass
    with pytest.raises(ValueError):
        memory_planning.planning_pass("fastest")


def test_esp32_is_the_default():
    """memory_plan="esp32" is the default: planning_pass(), the
    multimethod exporters' shared pass and the census lowering; None is
    ExecuTorch's own planner in each."""
    import inspect

    from executorch.backends.esp32s3.utils.multimethod import shared_memory_planning_pass
    from executorch.backends.esp32s3.test.quant_census import lower as census_lower

    assert type(memory_planning.planning_pass()) is memory_planning.Esp32MemoryPlanningPass
    assert type(shared_memory_planning_pass()) is memory_planning.Esp32MemoryPlanningPass
    assert type(shared_memory_planning_pass(None)) is MemoryPlanningPass
    assert inspect.signature(census_lower.lower).parameters["memory_plan"].default == "esp32"


def test_shared_mutable_buffers_pass_through():
    """share_mutable_buffers (the story's multimethod setting) still plans the
    buffers in run_multimethod."""
    off, _ = lower(Buffer(), (f32(16),), share_mutable_buffers=True)
    on, planner = lower(Buffer(), (f32(16),), "esp32", share_mutable_buffers=True)
    plan_off = off.executorch_program.execution_plan[0].non_const_buffer_sizes
    plan_on = on.executorch_program.execution_plan[0].non_const_buffer_sizes
    assert len(plan_on) == len(plan_off) == 3 and plan_on[2] == plan_off[2]


# ------------------------------------------------------------ the hill climb


def _ranges(rng, n):
    specs = []
    for _ in range(n):
        spec = TensorSpec(dtype=torch.uint8, shape=torch.Size([rng.choice((16, 48, 64, 96, 160, 256))]))
        start = rng.randrange(0, 40)
        spec.lifetime = [start, start + rng.randrange(0, 12)]
        specs.append(spec)
    return specs


def _check_disjoint(specs, offsets):
    for i, a in enumerate(specs):
        for b, ob in zip(specs[i + 1:], offsets[i + 1:]):
            oa = offsets[i]
            if a.lifetime[0] <= b.lifetime[1] and b.lifetime[0] <= a.lifetime[1]:
                assert oa + a.allocated_memory <= ob or ob + b.allocated_memory <= oa


def test_hill_climb_is_sound_and_finds_smaller_packings():
    """Random live ranges: never an overlap, never under the live bound, and
    on some of them smaller than ExecuTorch's greedy."""
    rng = random.Random(7)
    smaller = 0
    for trial in range(60):
        specs = _ranges(rng, rng.randrange(4, 30))
        bound = memory_planning.live_bound(specs)
        hill = memory_planning.hillclimb(16, specs, None, None)
        offsets = [hill.spec_dict[s].mem_offset for s in specs]
        _check_disjoint(specs, offsets)
        assert sum(hill.bufsizes) >= bound
        assert all(o + s.allocated_memory <= hill.bufsizes[1] for o, s in zip(offsets, specs))
        base = greedy(16, list(specs), None, None)
        smaller += sum(hill.bufsizes) < sum(base.bufsizes)
    assert smaller > 0


def test_hill_climb_storage_backed_specs():
    """A storage-backed spec lands at its base's offset, with its base's id."""
    a = TensorSpec(dtype=torch.float32, shape=torch.Size([8]))
    b = TensorSpec(dtype=torch.float32, shape=torch.Size([8]))
    c = TensorSpec(dtype=torch.float32, shape=torch.Size([8]))
    a.lifetime, b.lifetime, c.lifetime = [0, 3], [2, 3], [0, 1]
    b.storage_base = a
    result = memory_planning.hillclimb(16, [a, b, c], None, None)
    assert result.spec_dict[b].mem_offset == result.spec_dict[a].mem_offset
    assert result.spec_dict[b].mem_obj_id == result.spec_dict[a].mem_obj_id
    assert result.spec_dict[c].mem_offset != result.spec_dict[a].mem_offset


# ------------------------------------------------------------ the census switch


class ResidualCnn(torch.nn.Module):
    """Two delegated convolutions with a portable tanh between them (the
    partitioner fuses relu, silu, gelu and friends into its epilogues, not
    tanh). The 8 -> 32 -> 8 expansion makes tanh's step the peak, where its
    output may take the dying convolution output's bytes."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(8, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.tanh(self.conv1(x))) + x


def test_census_lowering_switch(runner, tmp_path, monkeypatch):
    """quant_census.lower.lower(memory_plan="esp32"): delegate outputs donate,
    the arena shrinks, the host runner's output bits are unchanged, and the
    record says what the planner did; without it the record has no entry."""
    from executorch.backends.esp32s3.test.quant_census import lower as census_lower

    monkeypatch.setattr(census_lower, "HOST_RUNNER", runner)
    x = torch.randn(1, 8, 8, 8)
    exported = torch.export.export(ResidualCnn().eval(), (x,))
    recs, programs = {}, {}
    for plan in (None, "esp32"):
        rec = {"seconds": {}}
        # Without planned scratch (on by default; test_planned_scratch_real_backend
        # covers it): in a model this small the delegates' scratch fills the
        # arena under either planner, which hides the planner's own saving.
        programs[plan] = census_lower.lower(rec, exported, tmp_path / str(plan), memory_plan=plan,
                                            plan_scratch=False)
        recs[plan] = rec
    assert "memory_plan" not in recs[None]
    assert recs["esp32"]["memory_plan"]["plan"] == "esp32"
    assert recs["esp32"]["delegates"] >= 1
    assert recs["esp32"]["arena_bytes"] < recs[None]["arena_bytes"], recs["esp32"]["memory_plan"]
    like = ResidualCnn().eval()(x)
    outs = {plan: census_lower.run_host(recs[plan], programs[plan], (x,), like)
            for plan in programs}
    assert outs[None].tobytes() == outs["esp32"].tobytes()


def test_planned_scratch_real_backend(runner, tmp_path, monkeypatch):
    """Esp32Partitioner(plan_scratch=True) on the census lowering: the host
    runner's bits are the same with and without planned scratch under either
    planner; ours is never worse than ExecuTorch's with the scratch in the
    arena (run() checks every pair of planned specs for overlap, the scratch
    included); planned scratch never costs more than the temp it replaces
    plus its alignment slack (blob_writer.SCRATCH_SLACK, rounded to the
    planner's 16 bytes); and under ExecuTorch's plan, where conv1's output is
    dead during conv2, conv2's scratch takes those bytes. (Ours already gives
    them to tanh's output, in place, so it has none dead there to reuse.)"""
    from executorch.backends.esp32s3.serialization.blob_writer import SCRATCH_SLACK
    from executorch.backends.esp32s3.test.quant_census import lower as census_lower

    def slack(scratch):  # what one planned scratch may cost over the temp
        return -(-(scratch + SCRATCH_SLACK) // 16) * 16 - scratch

    monkeypatch.setattr(census_lower, "HOST_RUNNER", runner)
    x = torch.randn(1, 8, 8, 8)
    exported = torch.export.export(ResidualCnn().eval(), (x,))
    like = ResidualCnn().eval()(x)
    recs, outs = {}, {}
    for plan, scratch in ((None, False), (None, True), ("esp32", False), ("esp32", True)):
        rec = {"seconds": {}}
        program = census_lower.lower(rec, exported, tmp_path / f"{plan}-{scratch}",
                                     memory_plan=plan, plan_scratch=scratch)
        calls = [i.instr_args for chain in program.executorch_program.execution_plan[0].chains
                 for i in chain.instructions if type(i.instr_args).__name__ == "DelegateCall"]
        assert calls and all(bool(c.scratch) == scratch for c in calls), (plan, scratch)
        outs[plan, scratch] = census_lower.run_host(rec, program, (x,), like).tobytes()
        recs[plan, scratch] = rec
    assert len(set(outs.values())) == 1
    assert recs["esp32", True]["arena_bytes"] <= recs[None, True]["arena_bytes"]
    assert "planned scratch for 2 delegate call(s)" in recs["esp32", True]["memory_plan"]["methods"][0]
    for plan in (None, "esp32"):
        without, planned = recs[plan, False], recs[plan, True]
        scratch = without["delegate_scratch_bytes"]
        assert (planned["arena_bytes"]
                <= without["arena_bytes"] + scratch + slack(scratch)), (plan, recs)
    without, planned = recs[None, False], recs[None, True]
    assert planned["arena_bytes"] < without["arena_bytes"] + without["delegate_scratch_bytes"]
    # The census record: planned scratch counted once, in the arena; the
    # runtime's counts say every call ran on it.
    for plan in (None, "esp32"):
        without, planned = recs[plan, False], recs[plan, True]
        assert planned["plan_scratch"] and not without["plan_scratch"]
        assert planned["delegate_scratch_bytes"] == 0 and without["planned_scratch_bytes"] == 0
        assert planned["peak_bytes"] == planned["arena_bytes"]
        assert planned["planned_scratch_bytes"] == without["delegate_scratch_bytes"] > 0
        scratch = without["delegate_scratch_bytes"]
        assert planned["peak_bytes"] <= without["peak_bytes"] + slack(scratch)
        assert planned["scratch_calls"] == [2, 0, 0] and without["scratch_calls"] == [0, 2, 0]


# ------------------------------------------------------------ planned delegate scratch


class PlannerScratchProbe(BackendDetails):
    """A delegate that declares the scratch its "nbytes" compile spec names
    (little-endian) and does nothing else: its blob is never run, so these
    tests plan around scratch without a backend's runtime."""

    @staticmethod
    def preprocess(edge_program, compile_specs):
        nbytes = next((int.from_bytes(s.value, "little") for s in compile_specs
                       if s.key == "nbytes"), 0)
        return PreprocessResult(processed_bytes=b"scratch-probe",
                                scratch_specs=[DelegateScratchSpec(nbytes)] if nbytes else None)


class PerAddPartitioner(Partitioner):
    """One PlannerScratchProbe delegate per add, each declaring `nbytes`."""

    def __init__(self, nbytes):
        self.delegation_spec = DelegationSpec(
            "PlannerScratchProbe", [CompileSpec("nbytes", nbytes.to_bytes(4, "little"))])

    def partition(self, exported_program):
        tags = {}
        for node in exported_program.graph_module.graph.nodes:
            if node.op == "call_function" and node.target is exir_ops.edge.aten.add.Tensor:
                tag = f"tag{len(tags)}"
                node.meta["delegation_tag"] = tag
                tags[tag] = self.delegation_spec
        return PartitionResult(tagged_exported_program=exported_program, partition_tags=tags)


def lower_scratch(module, inputs, nbytes, plan=None, **options):
    """lower() with every add a PlannerScratchProbe delegate declaring
    `nbytes` of scratch (none at 0)."""
    module.eval()
    edge = to_edge_transform_and_lower(
        torch.export.export(module, inputs), partitioner=[PerAddPartitioner(nbytes)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False))
    planner = memory_planning.planning_pass(plan, **options)
    return edge.to_executorch(ExecutorchBackendConfig(memory_planning_pass=planner)), planner


def delegate_calls(program):
    plan = program.executorch_program.execution_plan[0]
    return [i.instr_args for chain in plan.chains for i in chain.instructions
            if type(i.instr_args).__name__ == "DelegateCall"]


class DeadBeforeCall(torch.nn.Module):
    """A large temporary that dies before the delegated add: the add's scratch
    can take its bytes, so planning the scratch costs no arena at all."""

    def forward(self, x, y):
        big = torch.cat([x] * 8)
        return torch.sin(big.sum(0, keepdim=True) + y)


class TwoCalls(torch.nn.Module):
    """Two delegated adds, the second's operand computed between them."""

    def forward(self, x, y):
        a = torch.sin(x + y)
        return torch.cos(torch.cat([a, a]).sum(0, keepdim=True) + y)


class BranchCalls(torch.nn.Module):
    """Branches, each branch ending in a delegated add: the order search
    reorders it with scratch in the graph."""

    def forward(self, x):
        a = torch.cat([x, x, x, x], 1)
        b = torch.cat([x, x, x, x], 1)
        sa, sb = a.sum(1, keepdim=True), b.sum(1, keepdim=True)
        return torch.sin(sa + sa) * torch.sin(sb + sb)


SCRATCH_CASES = [
    ("dead before call", DeadBeforeCall(), (f32(4, 256), f32(1, 256))),
    ("two calls", TwoCalls(), (f32(1, 256), f32(1, 256))),
    ("branches", BranchCalls(), (f32(4, 256),)),
]


@pytest.mark.parametrize("case,module,inputs", SCRATCH_CASES, ids=[c[0] for c in SCRATCH_CASES])
def test_scratch_planned_at_its_call(case, module, inputs):
    """Every call's scratch is emitted at the declared size and planned live
    at its own call alone (ExecuTorch's rule: not to the last reader of the
    call's outputs), the plan records it, and the arena is never larger than
    ExecuTorch's plan of the same program."""
    nbytes = 3000  # not a multiple of the alignment
    on, planner = lower_scratch(module, inputs, nbytes, "esp32")
    off, _ = lower_scratch(module, inputs, nbytes)
    plan = planner.plans[0]
    assert "planner error" not in plan.note, plan.describe()
    calls = delegate_calls(on)
    assert calls and all([s.size for s in c.scratch] == [nbytes] for c in calls)
    assert plan.scratch_calls == len(calls) and plan.scratch_max == nbytes, plan.describe()
    gm = on.exported_program().graph_module
    for index, node in enumerate(gm.graph.nodes):
        for spec in memory.delegate_scratch_specs(node):
            assert spec.lifetime == [index, index], (case, node.name, spec.lifetime)
    assert arena(on) <= arena(off), (case, arena(on), arena(off))
    assert plan.arena == arena(on)
    memory_planning.check_storage(
        [s for n in gm.graph.nodes for s in memory.delegate_scratch_specs(n)]
        + [s for n in gm.graph.nodes for s in memory_planning.get_node_tensor_specs(n)
           if isinstance(s, TensorSpec) and s.mem_id is not None and s.lifetime[0] is not None])


def test_scratch_reuses_dead_bytes():
    """Scratch takes bytes nothing live at its call uses: the arena grows by
    less than the scratch, here not at all."""
    inputs = (f32(4, 256), f32(1, 256))
    none, _ = lower_scratch(DeadBeforeCall(), inputs, 0, "esp32")
    some, planner = lower_scratch(DeadBeforeCall(), inputs, 8192, "esp32")
    assert not delegate_calls(none)[0].scratch
    assert arena(some) == arena(none), planner.plans[0].describe()


def test_scratch_with_the_order_search():
    """A method the search reorders plans its scratch in the chosen order,
    and apply_algo agrees with the candidate's measurement (run() checks;
    a stale scratch lifetime from a dry run would make them differ)."""
    on, planner = lower_scratch(BranchCalls(), (f32(4, 256),), 2048, "esp32")
    plan = planner.plans[0]
    assert plan.reordered and "planner error" not in plan.note, plan.describe()
    assert plan.arena == arena(on) and plan.scratch_calls == 2


def test_set_lifetimes_scratch_rule():
    """set_lifetimes, which plans every candidate, gives scratch the call's
    lifetime alone, as apply_algo will: the walk alone would stretch it over
    the getitems that read the call's outputs."""
    on, _ = lower_scratch(TwoCalls(), (f32(1, 256), f32(1, 256)), 512, "esp32")
    ep = on.exported_program()
    nodes = list(ep.graph_module.graph.nodes)
    memory_planning.set_lifetimes(nodes, ep.graph_signature)
    calls = [(i, n) for i, n in enumerate(nodes) if memory.delegate_scratch_specs(n)]
    assert len(calls) == 2
    for index, node in calls:
        assert any(user.target is operator.getitem for user in node.users)
        for spec in memory.delegate_scratch_specs(node):
            assert spec.lifetime == [index, index], (node.name, spec.lifetime)


def test_snapshot_restores_scratch():
    """_Snapshot saves and restores a delegate's scratch specs, which live
    beside the call's meta["spec"]."""
    on, _ = lower_scratch(DeadBeforeCall(), (f32(4, 256), f32(1, 256)), 512, "esp32")
    gm = on.exported_program().graph_module
    spec = next(s for n in gm.graph.nodes for s in memory.delegate_scratch_specs(n))
    saved = (list(spec.lifetime), spec.mem_id, spec.mem_offset)
    snapshot = memory_planning._Snapshot(gm)
    spec.lifetime, spec.mem_id, spec.mem_offset = [0, 99], 7, 12345
    snapshot.restore()
    assert (spec.lifetime, spec.mem_id, spec.mem_offset) == saved


def test_write_backs_are_modelled():
    """A buffer's in-place write-back: on 1.6 its result shares the buffer's
    spec (SpecPropPass), so no spec is left that only the output names, and
    the order search models the method."""
    _, planner = lower(Buffer(), (f32(16),), "esp32")
    plan = planner.plans[0]
    assert "does not model" not in plan.search, plan.describe()


def test_alias_check_runs_first():
    """run() starts with ExecuTorch's verify_inplace_result_aliases: an
    in-place result with a spec of its own is refused before planning."""
    on, _ = lower(Buffer(), (f32(16),), "esp32")
    gm = on.exported_program().graph_module
    written = [n for n in gm.graph.nodes if memory_planning.is_inplace_node(n)]
    assert written, [n.target for n in gm.graph.nodes]
    node = written[0]
    spec = node.meta["spec"]
    node.meta["spec"] = TensorSpec(dtype=spec.dtype, shape=spec.shape)
    with pytest.raises(InternalError, match="does not share the TensorSpec"):
        memory_planning.Esp32MemoryPlanningPass().run(gm, on.exported_program().graph_signature)
