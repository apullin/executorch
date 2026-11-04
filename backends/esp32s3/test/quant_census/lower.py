# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower one quantised program through the backend, run it, and measure it.

The pipeline is the README's -- lower_esp32, apply_passes, then
to_executorch with every constant in one PTD. lower_esp32 runs
strip_dynamic_act_qdq right before partitioning: every dynamic chain is
absorbed with its scheme recorded, left in place for an op no delegate takes
and computed there by float32 portable ops ("esp32: leaving ...", the
record's left_in_place), or refused by name; never substituted.

Partitioner: Esp32Partitioner(external_weights=PTD_TAG), every other option at
its default, chosen to keep each model's own quantisation:

- format_policy at its default. Weights a recipe quantised arrive as dequantise
  nodes over constant codes and are consumed verbatim (PassThroughWeight); a
  weight the recipe left float keeps its precision (the fp32 loop for a matmul,
  the fp32 convolution for a conv, portable fp32 for a table), as the
  recipe's reference computes it.
- quantize_qk off: no stock recipe quantises the attention matmuls, so the
  backend leaves them in fp32 too.
- group_weights, static_act, min_latency off: the backend's own weight format,
  calibration and scheduling, not the model's.

Imports torch and executorch.
"""

import contextlib
import io
import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from executorch.backends.esp32s3.test.host import (
    HOST_RUNNER,
    emitted_ops,
    op_base,
    op_name,
)

PTD_TAG = "census_weights"
# Lines of the backend's own output kept in a record, and their width.
LOG_LINES, LOG_WIDTH = 40, 400


@contextlib.contextmanager
def strip_before_partitioning(report):
    """Lower the README's way, with lower_esp32's own strip_dynamic_act_qdq
    right before partitioning, and put the count of the chains it absorbed
    in report["stripped"] (edge.stripped_chains).

    Nothing strips early: a strip right after to_edge, with no
    partitioner's gate, assumes every matmul and convolution takes its
    chain and refuses any other reader, so it would stop programs that
    lower_esp32's gated strip lowers (a convolution over a batch of images,
    a Linear torch spells as a bmm). This wraps lower_esp32's own call, and
    if lower_esp32 stops making it, raises instead of measuring another
    pipeline. Kept for its callers (the asym, Stage B, float-weight and
    transposed-convolution suites).
    """
    from executorch.backends.esp32s3._passes import strip_dynamic_act_qdq_pass

    strip = strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq

    def record(edge, method_name="forward", partitioner=None):
        report["stripped"] = strip(edge, method_name, partitioner)
        return report["stripped"]

    strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq = record
    try:
        yield
    finally:
        strip_dynamic_act_qdq_pass.strip_dynamic_act_qdq = strip
    if "stripped" not in report:
        raise RuntimeError("lower_esp32 did not run strip_dynamic_act_qdq "
                           "before partitioning")


def node_macs(node):
    """Multiply-accumulates of one matmul or convolution node, else 0."""
    if node.op != "call_function":
        return 0
    base = op_base(op_name(node.target))
    out = node.meta.get("val")
    if not isinstance(out, torch.Tensor):
        return 0

    def shape(i):
        arg = node.args[i] if i < len(node.args) else None
        val = arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None
        return tuple(val.shape) if isinstance(val, torch.Tensor) else None

    if base in ("mm", "bmm", "linear", "matmul"):
        x = shape(0)
        return out.numel() * x[-1] if x else 0
    if base == "addmm":
        x = shape(1)
        return out.numel() * x[-1] if x else 0
    if base in ("convolution", "conv1d", "conv2d"):
        w = shape(1)
        return out.numel() * math.prod(w[1:]) if w else 0
    return 0


def _is_qdq(name):
    return "quantize" in name or "choose_qparams" in name


def graph_report(gm):
    """Delegates, MACs inside and outside them, and the ops left outside."""
    delegates, inside, outside, ops = 0, 0, 0, Counter()
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        if "executorch_call_delegate" in str(n.target):
            delegates += 1
            lowered = getattr(gm, n.args[0].target)
            inside += sum(node_macs(m) for m in lowered.original_module.graph.nodes)
            continue
        name = op_name(n.target)
        if name == "getitem":
            continue
        ops[name] += 1
        outside += node_macs(n)
    qdq = Counter({o: c for o, c in ops.items() if _is_qdq(o)})
    rest = Counter({o: c for o, c in ops.items() if not _is_qdq(o)})
    total = inside + outside
    return {"delegates": delegates, "macs": total, "macs_delegated": inside,
            "mac_share": round(inside / total, 4) if total else None,
            "ops_outside": sum(rest.values()), "qdq_outside": sum(qdq.values()),
            "outside": dict(rest.most_common()), "qdq": dict(qdq.most_common())}


def backend_report(log):
    """What the backend printed: its messages (partitioner summaries, chains it
    left in place, notes), and per delegate op kind how many descriptors it
    emitted -- addmmPT_* is a recipe's weight consumed verbatim, addmm_* one
    the backend quantised itself."""
    from executorch.backends.esp32s3.serialization.espressif_blob import BACKEND_NAME

    lines = log.splitlines()
    messages = [l.strip()[:LOG_WIDTH] for l in lines
                if l.startswith("esp32") or f"[{BACKEND_NAME}] NOTE" in l or "WARNING" in l]
    kinds = Counter()
    blob = re.compile(rf"\s*\[{re.escape(BACKEND_NAME)}\] \d+ op\(s\), [\d,]+ B blob[^:]*: (.*)")
    for l in lines:
        m = blob.match(l)
        if m:
            kinds.update(re.findall(r"(?:^|, )([A-Za-z][A-Za-z0-9_]*)\[", m.group(1)))
    return {"backend_messages": messages[:LOG_LINES],
            "left_in_place": [l for l in messages if l.startswith("esp32: leaving")],
            "delegate_ops": dict(kinds.most_common())}


def lower(rec, exported, out_dir, memory_plan="esp32", **options):
    """Lower `exported`, write model.pte and its PTD into out_dir, and fill rec
    with what came out. Returns the ExecutorchProgramManager. `options`:
    Esp32Partitioner options beyond the census's own external weights, every
    other one at its default (recipes.lowering: stage_b=False for the Stage A
    twins); Stage B's report, where it ran, goes in rec["stage_b"].
    memory_plan "esp32" (the default) plans the arena with memory_planning.py,
    None with ExecuTorch's planner. Planned scratch (on by default;
    plan_scratch in `options`, or ESP32_PLAN_SCRATCH=0 or 1, which each pair's
    process inherits) puts each delegate's scratch in the arena; run_host's
    peak_bytes counts it there, and rec["plan_scratch"] says which."""
    # The .out variants of the quantized_decomposed ops, which to_executorch
    # needs for any quantise/dequantise left outside a delegate. Arm's
    # quantizer package imports this on its own; without it, those programs
    # stop at to_executorch ("Missing out variants") instead of emitting and
    # naming the runtime kernels they lack.
    import executorch.kernels.quantized  # noqa: F401
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    partitioner = Esp32Partitioner(external_weights=PTD_TAG, **options)
    rec["plan_scratch"] = partitioner.compile_spec.plan_scratch
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            edge = lower_esp32(exported, [partitioner])
            apply_passes(edge, quiet=True)
    finally:
        rec.update(backend_report(log.getvalue()))
    rec["stripped_chains"] = edge.stripped_chains
    rec.update(graph_report(edge.exported_program().graph_module))
    static = getattr(edge, "static_qdq_report", None)
    if static is not None and static.stage_b is not None:
        sb = static.stage_b
        rec["stage_b"] = {"summary": sb.summary(),
                          "producers": len(sb.converted),
                          "consumers": len(sb.consumers),
                          "code_ops": sorted({k for _, k in sb.code_ops}),
                          "kept": [[p, rule] for p, rule, _ in sb.kept],
                          "identity_failures": len(sb.identity)}
    planning = {}
    if memory_plan is not None:
        from executorch.backends.esp32s3.memory_planning import planning_pass

        planning["memory_planning_pass"] = planning_pass(memory_plan)
    program = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, external_constants=lambda node: PTD_TAG, **planning))
    if planning:
        rec["memory_plan"] = {"plan": memory_plan, "methods": [
            plan.describe() for plan in planning["memory_planning_pass"].plans]}
    pte = out_dir / "model.pte"
    pte.write_bytes(program.buffer)
    program.write_tensor_data_to_file(str(out_dir))
    ptd = out_dir / f"{PTD_TAG}.ptd"
    rec["pte"], rec["ptd"] = str(pte), str(ptd) if ptd.is_file() else None
    rec["pte_bytes"] = pte.stat().st_size
    rec["ptd_bytes"] = ptd.stat().st_size if ptd.is_file() else 0
    plan = program.executorch_program.execution_plan[0]
    # Buffer 0 stands for the constant segment; the rest are the planned arena.
    rec["arena_bytes"] = sum(plan.non_const_buffer_sizes[1:])
    # Each delegate's working memory (activation quantisation rows, unpacked
    # weight rows, chain slots) is sized by the runtime at init; the host run
    # reports it (run_host), and under plan_scratch it is in the arena above.
    rec["delegate_scratch_bytes"] = None
    rec["ops"] = emitted_ops(program)
    rec["seconds"]["lower"] = round(time.perf_counter() - started, 1)
    return program


def _dim_orders(program, which):
    plan = program.executorch_program.execution_plan[0]
    return [list(plan.values[i].val.dim_order)
            for i in (plan.inputs if which == "inputs" else plan.outputs)]


def run_host(rec, program, inputs, like, timeout=300):
    """Run the lowered program on the host runner; its first output, shaped
    and typed like `like`. Inputs are written and the output read in the
    memory order the program declares (a channels-last image input is NHWC
    bytes)."""
    if not HOST_RUNNER.is_file():
        raise FileNotFoundError(f"host runner not built: {HOST_RUNNER}; see test/host_runner/README.md")
    work = Path(rec["pte"]).parent
    # The program's inputs are the flattened ones (ExecuTorch's llama takes
    # (tokens, {"input_pos": ...})).
    inputs = torch.utils._pytree.tree_flatten(inputs)[0]
    command = [str(HOST_RUNNER), "--pte", rec["pte"], "--method", "forward"]
    if rec["ptd"]:
        command += ["--ptd", rec["ptd"]]
    for i, (x, order) in enumerate(zip(inputs, _dim_orders(program, "inputs"))):
        path = work / f"input{i}.bin"
        x = x.detach()
        if order and order != sorted(order):
            x = x.permute(*order)
        np.ascontiguousarray(x.contiguous().numpy()).tofile(path)
        command += ["--input", str(path)]
    out = work / "output0.bin"
    out.unlink(missing_ok=True)
    command += ["--output", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"host runner exit {result.returncode}: "
                           f"{(result.stdout + result.stderr)[-2000:]}")
    # Each delegate's init line gives its temp: the largest per-call scratch
    # (activation quantisation rows, conv staging, chain slots), in delegate
    # order. Taken from the temp allocator, which resets after every
    # instruction, so the method needs the largest one at once with the arena:
    # its activation peak. A delegate whose scratch is memory-planned
    # (plan_scratch: its call carries DelegateCall.scratch) has it in the
    # arena already, so it counts once, there.
    log = result.stdout + result.stderr
    temps = [int(t) for t in re.findall(r"init: \d+ op\(s\), \d+ B blob, (\d+) B temp", log)]
    plan = program.executorch_program.execution_plan[0]
    planned = {i.instr_args.delegate_index for chain in plan.chains for i in chain.instructions
               if type(i.instr_args).__name__ == "DelegateCall" and i.instr_args.scratch}
    rec["delegate_scratch_bytes"] = max(
        (t for d, t in enumerate(temps) if d not in planned), default=0)
    rec["planned_scratch_bytes"] = max(
        (t for d, t in enumerate(temps) if d in planned), default=0)
    rec["peak_bytes"] = rec["arena_bytes"] + rec["delegate_scratch_bytes"]
    # Where the calls got their scratch (planned, temp allocator, fallback):
    # a fallback means scratch_layout.py and the runtime disagree.
    calls = re.search(r"esp32 scratch: (\d+) call\(s\) on planned scratch, (\d+) on the "
                      r"temp allocator \(none planned\), (\d+) fallback", log)
    rec["scratch_calls"] = [int(g) for g in calls.groups()] if calls else None
    like = like.detach()
    order = _dim_orders(program, "outputs")[0]
    if order and order != sorted(order):
        stored = [like.shape[d] for d in order]
        data = np.fromfile(out, dtype=like.numpy().dtype).reshape(stored)
        return np.transpose(data, np.argsort(order))
    return np.fromfile(out, dtype=like.numpy().dtype).reshape(tuple(like.shape))


def compare(actual, reference, output_qparams=None, logits=False):
    """How close one output is to another: cosine, relative L2, max abs error,
    and that error in steps of the output's quantisation when it has one."""
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().numpy()
    a = actual.astype(np.float64).ravel()
    r = reference.astype(np.float64).ravel()
    cosine = float(a @ r / max(np.linalg.norm(a) * np.linalg.norm(r), 1e-30))
    err = np.abs(a - r)
    rec = {"finite": bool(np.isfinite(a).all()), "cosine": round(cosine, 6),
           "rel_l2": round(float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-30)), 6),
           "max_abs": float(f"{err.max():.4g}") if err.size else 0.0}
    if output_qparams:
        rec["max_steps"] = round(float(err.max()) / output_qparams["scale"], 2)
    if logits:  # [batch, positions, vocab]
        rec["top1_agree"] = round(float((actual.argmax(-1) == reference.argmax(-1)).mean()), 4)
    return rec
