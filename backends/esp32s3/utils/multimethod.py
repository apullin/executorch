# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Combine separately lowered Glimmer entry points with shared KV state.

Decode and blocked prefill need different static input shapes, so they are two
ExecuTorch methods.  Their mutable buffers deliberately have identical FQNs and
shapes; ExecuTorch's multimethod memory planner can therefore assign them to one
mem-id-2 arena.  Learned named data is content-deduplicated across both methods
when the combined manager emits one PTE/PTD pair.
"""

from __future__ import annotations

import copy

import torch

from executorch.exir import EdgeCompileConfig, EdgeProgramManager


def mutable_buffer_specs(edge_manager) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return mutable FQN -> (shape, dtype), failing on incomplete metadata."""
    program = edge_manager.exported_program()
    signature = program.graph_signature
    mutable = set(signature.buffers_to_mutate.values())
    specs = {}
    for node in program.graph_module.graph.nodes:
        if node.op != "placeholder" or node.target not in signature.inputs_to_buffers:
            continue
        fqn = signature.inputs_to_buffers[node.target]
        if fqn not in mutable:
            continue
        value = node.meta.get("val")
        if value is None:
            raise ValueError(f"mutable buffer {fqn!r} has no fake-tensor metadata")
        specs[fqn] = (tuple(int(dim) for dim in value.shape), value.dtype)
    missing = mutable - specs.keys()
    if missing:
        raise ValueError(f"mutable buffers absent from graph inputs: {sorted(missing)}")
    return specs


def combine_shared_methods(methods: dict[str, object],
                           copy_programs: bool = True) -> EdgeProgramManager:
    """Merge one-method edge managers after proving their state ABI identical.

    ``copy_programs=False`` hands the managers' programs over instead of
    deep-copying them, for a caller that discards the source managers: the
    combined manager's to_executorch then mutates them (see below).
    """
    if len(methods) < 2:
        raise ValueError("shared multimethod export requires at least two methods")
    programs = {}
    reference = None
    for name, manager in methods.items():
        if not name or name == "forward":
            raise ValueError("combined methods need explicit non-forward names")
        if manager.methods != {"forward"}:
            raise ValueError(
                f"{name}: expected one lowered 'forward' method, got "
                f"{sorted(manager.methods)}")
        specs = mutable_buffer_specs(manager)
        if not specs:
            raise ValueError(f"{name}: method has no mutable state to share")
        if reference is None:
            reference = specs
        elif specs != reference:
            raise ValueError(
                f"{name}: mutable-state ABI differs from the first method\n"
                f"  first={reference}\n  {name}={specs}")
        # This ExecuTorch wheel's to_executorch path mutates graph outputs while
        # inserting buffer write-back.  A combined manager must own its methods;
        # otherwise emitting it invalidates the source managers (and vice versa).
        program = manager.exported_program()
        programs[name] = copy.deepcopy(program) if copy_programs else program

    # Each constituent manager was already verified as Edge IR before the
    # post-lowering RMSNorm/RoPE fusions ran.  Those registered espressif ops are
    # intentionally custom rather than Edge operators; reconstructing a manager
    # with the wheel's default verifier would reject them even though each source
    # manager can emit them.  Disable only this redundant second verification.
    return EdgeProgramManager(
        programs,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )


def shared_memory_planning_pass(memory_plan="esp32"):
    """Fresh pass instance; its cross-method state must never be reused.
    memory_plan="esp32" (the default) is memory_planning.py's planner,
    None ExecuTorch's own."""
    if memory_plan is not None:
        from executorch.backends.esp32s3.memory_planning import planning_pass

        return planning_pass(memory_plan, share_mutable_buffers=True)
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    return MemoryPlanningPass(share_mutable_buffers=True)
