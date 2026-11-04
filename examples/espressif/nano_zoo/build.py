# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build, lower and run one nano family. Imports torch and transformers.

    from executorch.examples.espressif.nano_zoo import build
    net, inputs = build.model("llama3_2", "device")
    lowered = build.lower(net, inputs, out_dir)
    output = build.run_host(lowered["pte"], lowered["ptd"], inputs, reference)

A family module provides:

  MODEL_TYPE       key into transformers.CONFIG_MAPPING
  MODEL_CLASS      the transformers class to instantiate
  DEVICE, CI       constructor kwargs: config fields, or builder-only kwargs
                   the config consumes (Gemma 4's global_head_dim);
                   sub-configs as nested dicts, and an AutoConfig-typed
                   sub-config names its own "model_type"
  INPUTS           optional: fn(cfg) -> {forward kwarg: example tensor};
                   default: input_ids [1, SEQ]
  FORWARD_KWARGS   optional: static forward kwargs; default {"use_cache": False}
  OUTPUT           optional: output attribute, or fn(output) -> tensor;
                   default "logits"
  EXPERTS          optional: the experts implementation; default "batched_mm"
                   (selected experts in batched matmuls, no data-dependent loop)
  EXPECT           optional: why the family is known not to lower or run yet,
                   or {size: reason} when only one size shows it;
                   test_nano_zoo.py marks it xfail(strict), so a fix shows up
  UNDECLARED       optional: top-level config keys the modeling code reads
                   with getattr but the config class does not declare
                   (Qwen3-MoE's head_dim: its released config.json sets it);
                   exempt from the unknown-key check
"""

import contextlib
import copy
import dataclasses
import io
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from executorch.examples.espressif.nano_zoo import (
    PARAM_CAP,
    SEQ,
    TRANSFORMERS_VERSION,
    family,
)
from executorch.backends.esp32s3.test.host import (
    HOST_RUNNER,
    op_name,
    op_base,
    emitted_ops,
)

PTD_TAG = "zoo_weights"

# Pure data movement: not counted as real portable work.
NOT_REAL = {"view_copy", "permute_copy", "squeeze_copy", "unsqueeze_copy",
            "expand_copy", "slice_copy", "select_copy", "clone", "getitem",
            "_to_dim_order_copy", "_clone_dim_order", "split_with_sizes_copy",
            "split_copy", "cat"}


# ---------------------------------------------------------------------- config


def _fields(cls):
    try:
        return {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        return set()


def _check_keys(cfg, kwargs, where, undeclared=()):
    """Every key must be a field of its config class, a builder-only kwarg the
    class's __post_init__ consumes (Gemma 4's global_head_dim shapes
    per_layer_config and is not kept), or named in the family's UNDECLARED (a
    key the modelling code reads but the config class does not declare).
    Checked on the built config: transformers 5.19 keeps any other kwarg as a
    plain attribute, so a misspelt size field would leave the real one at its
    released size without a word."""
    from transformers import PreTrainedConfig

    cls = type(cfg)
    subs = getattr(cls, "sub_configs", {})
    fields = _fields(cls) | set(undeclared)
    kept = sorted(k for k in kwargs if k not in fields and k not in subs
                  and k != "model_type" and k in vars(cfg))
    if kept:
        raise KeyError(f"{where}: not fields of {cls.__name__}: {kept}")
    for key, value in kwargs.items():
        if key in subs and isinstance(value, dict):
            sub = getattr(cfg, key)
            if not isinstance(sub, PreTrainedConfig):
                raise KeyError(f"{where}.{key}: given, but {cls.__name__} built no config from it")
            _check_keys(sub, value, f"{where}.{key}")


def config(name, size="device"):
    """The family's config at `size`, set up for export: eager attention,
    batched-matmul experts, no KV cache."""
    import transformers
    from transformers import CONFIG_MAPPING

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"nano zoo configs are written for transformers "
                           f"{TRANSFORMERS_VERSION}, found {transformers.__version__}")
    spec = family(name)
    kwargs = getattr(spec, size.upper())
    cfg = CONFIG_MAPPING[spec.MODEL_TYPE](**kwargs)
    _check_keys(cfg, kwargs, f"{name}.{size.upper()}", getattr(spec, "UNDECLARED", ()))
    cfg._attn_implementation = "eager"
    experts = getattr(spec, "EXPERTS", "batched_mm")
    if experts:
        with contextlib.suppress(Exception):  # configs without experts refuse it
            cfg._experts_implementation = experts
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    return cfg


# ----------------------------------------------------------------------- model


def count_params(net):
    """Parameter elements; a tied tensor counts once."""
    return sum(p.numel() for p in net.parameters())


def largest_params(net, k=4):
    rows = sorted(net.named_parameters(), key=lambda kv: -kv[1].numel())[:k]
    return [f"{n} {list(p.shape)}" for n, p in rows]


def meta_params(name, size="device"):
    """(parameter count, largest tensors) without allocating a byte."""
    import transformers

    cfg = config(name, size)
    with torch.device("meta"):
        net = getattr(transformers, family(name).MODEL_CLASS)(cfg)
    return count_params(net), largest_params(net)


def weight_bytes(params):
    """Estimated flash bytes at the backend's int4 group-128 format with fp16
    scales. The lowered PTE + PTD is the real number; this is for sizing."""
    return int(params * (0.5 + 2 / 128))


def ids(cfg):
    """input_ids [1, SEQ] = 1..SEQ, inside the text vocabulary."""
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    vocab = text.vocab_size
    return {"input_ids": (torch.arange(SEQ) % (vocab - 1) + 1).unsqueeze(0)}


class Wrapped(torch.nn.Module):
    """Positional tensors in, one tensor out: what torch.export and the host
    runner want from a model whose forward takes keyword inputs."""

    def __init__(self, net, names, kwargs, output):
        super().__init__()
        self.net = net
        self.names = names
        self.kwargs = kwargs
        self.output = output

    def forward(self, *inputs):
        out = self.net(**dict(zip(self.names, inputs)), **self.kwargs)
        if callable(self.output):
            return self.output(out)
        if isinstance(out, (tuple, list)):
            return out[0]
        return getattr(out, self.output)


def model(name, size="device", seed=0):
    """(wrapped module, example inputs): the family at `size` with seeded random
    weights. Refuses anything over PARAM_CAP before allocating it."""
    import transformers

    params, largest = meta_params(name, size)
    if params > PARAM_CAP:
        raise ValueError(f"{name} {size}: {params:,} parameters on the meta device "
                         f"> PARAM_CAP {PARAM_CAP:,}; largest: {largest}")
    spec = family(name)
    cfg = config(name, size)
    torch.manual_seed(seed)
    net = getattr(transformers, spec.MODEL_CLASS)(cfg).eval().requires_grad_(False)
    if count_params(net) != params:
        raise RuntimeError(f"{name}: built {count_params(net):,} parameters, "
                           f"the meta device counted {params:,}")
    inputs = getattr(spec, "INPUTS", ids)(cfg)
    wrapped = Wrapped(net, list(inputs), getattr(spec, "FORWARD_KWARGS", {"use_cache": False}),
                      getattr(spec, "OUTPUT", "logits"))
    return wrapped, tuple(inputs.values())


# -------------------------------------------------------------------- lowering


def graph_counts(gm):
    """Delegate calls and the portable ops left around them."""
    delegates, ops = 0, Counter()
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        if "executorch_call_delegate" in str(n.target):
            delegates += 1
        else:
            ops[op_name(n.target)] += 1
    real = Counter({o: c for o, c in ops.items() if op_base(o) not in NOT_REAL})
    return {"delegates": delegates, "portable_ops": sum(ops.values()),
            "real_portable_ops": sum(real.values()),
            "portable_bmm": sum(c for o, c in ops.items() if op_base(o) == "bmm"),
            "top_real_portable": dict(real.most_common(8))}


def portable_constant_elems(ep):
    """Constant elements left outside the delegates (portable weights,
    constant-folded tensors and buffers): what does not run on the fast path."""
    sig = ep.graph_signature
    names = list(sig.parameters) + list(sig.buffers) + list(sig.lifted_tensor_constants)
    total = 0
    for n in names:
        t = ep.state_dict.get(n)
        if t is None:
            t = ep.constants.get(n)
        if isinstance(t, torch.Tensor):
            total += t.numel()
    return total


def lower(net, inputs, out_dir, qk_probe=True):
    """export -> lower_esp32 with the story's partitioner settings -> the
    post-lowering passes -> to_executorch with the story's arena planning
    (memory_plan="esp32"), every weight in one PTD. Writes model.pte and
    zoo_weights.ptd into out_dir and returns what it measured.

    When a delegate computes a q@kT (OP_BMM_DYN_I8, claimed under
    quantize_qk) and `qk_probe` is set, also writes a second program into
    out_dir/qk_probe: the same lowered graph with each delegated q@kT's
    operands and scores added as outputs (rec["qk_probe"], for check_qk).
    model.pte stays byte for byte what it was without the probe."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.examples.espressif.esp32s3.scripts.flash_layout import layout_for
    from executorch.backends.esp32s3.memory_planning import planning_pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec, t0 = {"seconds": {}}, time.perf_counter()

    def lap(stage):
        nonlocal t0
        now = time.perf_counter()
        rec["seconds"][stage] = round(now - t0, 1)
        t0 = now

    exported = torch.export.export(net, inputs, strict=False)
    lap("export")
    partitioner = Esp32Partitioner(external_weights=PTD_TAG, format_policy="i4",
                                   group_weights=True, quantize_qk=True)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        edge = lower_esp32(exported, [partitioner])
        lap("lower")
        fusions = apply_passes(edge, quiet=True)
    lap("passes")
    ep = edge.exported_program()
    rec.update(graph_counts(ep.graph_module))
    rec["portable_constant_elems"] = portable_constant_elems(ep)
    rec["fusions"] = {"rmsnorm": fusions["fused"], "rope": fusions["rope_fused"],
                      "rmsnorm_add": fusions.get("rms_add_fused", 0),
                      "attn_tail": fusions.get("attn_tail_fused", 0)}
    summary = [l for l in log.getvalue().splitlines() if l.startswith("esp32 partitioner:")]
    rec["partitioner"] = summary[-1] if summary else ""
    # to_executorch rewrites the program it is given in place, so the probe
    # gets its own copy, taken before the program the zoo ships is emitted.
    probe_edge = copy.deepcopy(edge) if qk_probe and qk_delegates(ep) else None

    program = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, external_constants=lambda node: PTD_TAG,
        memory_planning_pass=planning_pass("esp32")))
    # Buffer 0 stands for the constant segment; the rest are the planned arena.
    rec["arena_bytes"] = sum(
        program.executorch_program.execution_plan[0].non_const_buffer_sizes[1:])
    pte = out_dir / "model.pte"
    pte.write_bytes(program.buffer)
    program.write_tensor_data_to_file(str(out_dir))
    ptd = out_dir / f"{PTD_TAG}.ptd"
    lap("emit")
    rec["pte"], rec["ptd"] = str(pte), str(ptd) if ptd.is_file() else None
    rec["pte_bytes"] = pte.stat().st_size
    rec["ptd_bytes"] = ptd.stat().st_size if ptd.is_file() else 0
    try:
        layout_for(rec["pte_bytes"], rec["ptd_bytes"])
        rec["fits_flash"] = True
    except ValueError as e:
        rec["fits_flash"] = False
        rec["flash_error"] = str(e)
    rec["ops"] = emitted_ops(program)
    if probe_edge is not None:
        del program
        try:
            rec["qk_probe"] = _emit_qk_probe(probe_edge, out_dir / "qk_probe", ptd)
        except Exception as e:  # noqa: BLE001 -- recorded; check_qk fails on it
            rec["qk_probe"] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
        lap("qk_probe")
    return rec


# ------------------------------------------------------------------ host runner


def run_host(pte, ptd, inputs, like, timeout=120):
    """Run the program on the host build of the delegate; returns the first
    output shaped and typed like `like`."""
    if not HOST_RUNNER.is_file():
        raise FileNotFoundError(f"host runner not built: {HOST_RUNNER}; see "
                                "backends/esp32s3/test/host_runner/README.md")
    work = Path(pte).parent
    command = [str(HOST_RUNNER), "--pte", str(pte), "--method", "forward"]
    if ptd:
        command += ["--ptd", str(ptd)]
    for i, x in enumerate(inputs):
        path = work / f"input{i}.bin"
        np.ascontiguousarray(x.numpy()).tofile(path)
        command += ["--input", str(path)]
    out = work / "output0.bin"
    out.unlink(missing_ok=True)
    command += ["--output", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"host runner exit {result.returncode}: "
                           f"{(result.stdout + result.stderr)[-2000:]}")
    like = like.detach().numpy()
    return np.fromfile(out, dtype=like.dtype).reshape(like.shape)


# -------------------------------------------------------------------- q@kT probe
#
# The cosine bar below cannot see the int8 q@kT: with HF's random init the
# scores are small and softmax near uniform, so zeroed scores or reversed heads
# keep llama3_2, mistral and smollm3 at cosine >= 0.9995. Modelling the int8
# q@kT in the eager reference would not fix
# that: the bar would still sit on the whole model's output. So each delegated
# q@kT is checked on its own, bit for bit: a second program exposes the q@kT
# delegates' operands and scores as outputs, the host runner computes them, and
# the scores must equal test/reference.py's bmm_qk_i8 of the operands the
# delegate received.

_DTYPES = {"FLOAT": np.float32, "CHAR": np.int8, "BYTE": np.uint8, "INT": np.int32,
           "LONG": np.int64, "BOOL": np.bool_, "SHORT": np.int16, "HALF": np.float16,
           "DOUBLE": np.float64}


def qk_delegates(ep):
    """[(call_delegate node, lowered module, bmm node, (A, P, X))] for every
    q@kT an Esp32Backend delegate computes (patterns._qk_bmm_structure: A @
    permute(X), the delegate reading X in its natural [B, N, K] layout)."""
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.lowered_backend_module import LoweredBackendModule

    from executorch.backends.esp32s3.operator_support.patterns import _qk_bmm_structure
    from executorch.backends.esp32s3.serialization.espressif_blob import BACKEND_NAME

    gm = ep.graph_module
    bmm_op = exir_ops.edge.aten.bmm.default
    found = []
    for node in gm.graph.nodes:
        if not (node.op == "call_function" and "executorch_call_delegate" in str(node.target)):
            continue
        lowered = getattr(gm, node.args[0].target)
        if not (isinstance(lowered, LoweredBackendModule)
                and lowered.backend_id == BACKEND_NAME):
            continue
        for inner in lowered.original_module.graph.nodes:
            if inner.op == "call_function" and inner.target is bmm_op:
                structure = _qk_bmm_structure(inner)
                if structure is not None:
                    found.append((node, lowered, inner, structure))
    return found


def _delegate_input(node, lowered, placeholder):
    """The outer value a delegate's user-input placeholder receives: call
    delegate arguments follow the subprogram's user inputs, in order."""
    from torch.export.graph_signature import InputKind

    users = [s.arg.name for s in lowered.original_module.graph_signature.input_specs
             if s.kind == InputKind.USER_INPUT]
    if placeholder.op != "placeholder" or placeholder.name not in users:
        return None
    outer = node.args[1 + users.index(placeholder.name)]
    want, got = placeholder.meta.get("val"), getattr(outer, "meta", {}).get("val")
    if want is None or got is None or tuple(want.shape) != tuple(got.shape):
        return None
    return outer


def _delegate_output(node, lowered, inner):
    """The outer getitem carrying `inner` (or a view_copy chain of it, same
    elements in the same order) out of the delegate, or None."""
    from executorch.exir.dialects._ops import ops as exir_ops

    view = exir_ops.edge.aten.view_copy.default
    outputs = next(n for n in lowered.original_module.graph.nodes if n.op == "output").args[0]
    for index, value in enumerate(outputs):
        while (isinstance(value, torch.fx.Node) and value is not inner
               and value.op == "call_function" and value.target is view):
            value = value.args[0]
        if value is inner:
            for user in node.users:
                if (user.op == "call_function" and user.target.__name__ == "getitem"
                        and user.args[1] == index):
                    return user
    return None


def _expose_qk(ep):
    """Add each delegated q@kT's query, keys and scores to the program's user
    outputs, after partitioning, without touching what was partitioned. Returns
    one record per q@kT: where each value is (["output", user output index]),
    and the shapes the delegate saw; a q@kT whose values cannot all be reached
    is recorded with a reason instead."""
    from torch.export.graph_signature import OutputKind, OutputSpec, TensorArgument

    gm = ep.graph_module
    output = next(n for n in gm.graph.nodes if n.op == "output")
    outs = list(output.args[0])
    specs = ep.graph_signature.output_specs
    user_index = {}

    def expose(value):
        if value.op == "placeholder":
            return None  # an input or a constant: the q@kT gate refuses both
        if value not in user_index:
            user_index[value] = sum(1 for s in specs if s.kind == OutputKind.USER_OUTPUT)
            outs.append(value)
            specs.append(OutputSpec(kind=OutputKind.USER_OUTPUT,
                                    arg=TensorArgument(name=value.name), target=None))
        return ["output", user_index[value]]

    records = []
    for node, lowered, inner, (A, _, X) in qk_delegates(ep):
        rec = {"delegate": node.name, "bmm": inner.name,
               "q_shape": [int(d) for d in A.meta["val"].shape],
               "k_shape": [int(d) for d in X.meta["val"].shape]}
        q, k = _delegate_input(node, lowered, A), _delegate_input(node, lowered, X)
        scores = _delegate_output(node, lowered, inner)
        if q is None or k is None or scores is None:
            rec["unreachable"] = ("query" if q is None else "keys" if k is None
                                  else "scores")
            records.append(rec)
            continue
        rec["values"] = {"q": expose(q), "k": expose(k), "scores": expose(scores)}
        if None in rec["values"].values():
            rec["unreachable"] = "a graph input or constant operand"
            del rec["values"]
        records.append(rec)
    output.args = (tuple(outs),)
    gm.recompile()
    return records


def _emit_qk_probe(edge, probe_dir, main_ptd):
    """to_executorch the exposed copy as the zoo's own program is emitted, into
    probe_dir; the weights file is shared with the main program when equal."""
    import filecmp

    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.memory_planning import planning_pass

    records = _expose_qk(edge.exported_program())
    program = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, external_constants=lambda node: PTD_TAG,
        memory_planning_pass=planning_pass("esp32")))
    probe_dir.mkdir(parents=True, exist_ok=True)
    pte = probe_dir / "model.pte"
    pte.write_bytes(program.buffer)
    program.write_tensor_data_to_file(str(probe_dir))
    ptd = probe_dir / f"{PTD_TAG}.ptd"
    if ptd.is_file() and main_ptd and Path(main_ptd).is_file() and filecmp.cmp(
            ptd, main_ptd, shallow=False):
        ptd.unlink()
        ptd = Path(main_ptd)
    plan = program.executorch_program.execution_plan[0]
    method_outputs = []
    for i in plan.outputs:
        t = plan.values[i].val
        method_outputs.append([getattr(t.scalar_type, "name", str(t.scalar_type)),
                               [int(d) for d in t.sizes]])
    n_user = sum(1 for spec in program.exported_program().graph_signature.output_specs
                 if spec.kind.name == "USER_OUTPUT")
    return {"pte": str(pte), "ptd": str(ptd) if ptd.is_file() else None,
            "method_outputs": method_outputs, "user_outputs": n_user, "ops": records}


def run_host_outputs(pte, ptd, inputs, method_outputs, timeout=300):
    """Every output of the method (method_outputs: [dtype name, sizes] each,
    in order) on the host runner, as numpy arrays."""
    work = Path(pte).parent
    command = [str(HOST_RUNNER), "--pte", str(pte), "--method", "forward"]
    if ptd:
        command += ["--ptd", str(ptd)]
    for i, x in enumerate(inputs):
        path = work / f"input{i}.bin"
        np.ascontiguousarray(x.numpy()).tofile(path)
        command += ["--input", str(path)]
    paths = [work / f"output{i}.bin" for i in range(len(method_outputs))]
    for path in paths:
        path.unlink(missing_ok=True)
        command += ["--output", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"host runner exit {result.returncode}: "
                           f"{(result.stdout + result.stderr)[-2000:]}")
    return [np.fromfile(path, dtype=_DTYPES[dtype]).reshape(sizes)
            for path, (dtype, sizes) in zip(paths, method_outputs)]


def compare_qk(user_outputs, ops, actual=None, first=None):
    """Each exposed q@kT's scores against bmm_qk_i8 of the operands the
    delegate received, bit for bit (NaN equal to NaN); and the probe's first
    output (`first`, by default its first user output) against the zoo
    program's (`actual`, what run_host read), which must be the same bits:
    the probe only adds outputs. Returns the record's "qk" entry; "exact" is
    the verdict."""
    from executorch.backends.esp32s3.test.reference import bmm_qk_i8

    rows, worst = [], 0.0
    for op in ops:
        row = {"delegate": op["delegate"], "q": op["q_shape"], "k": op["k_shape"]}
        if "values" not in op:
            row["unchecked"] = op["unreachable"]
            rows.append(row)
            continue
        q, k, scores = (user_outputs[op["values"][key][1]] for key in ("q", "k", "scores"))
        q = np.asarray(q, np.float32).reshape(op["q_shape"])
        k = np.asarray(k, np.float32).reshape(op["k_shape"])
        batches, m, n = q.shape[0], q.shape[1], k.shape[1]
        scores = np.asarray(scores, np.float32).reshape(batches, m, n)
        want = np.stack([bmm_qk_i8(q[b], k[b]) for b in range(batches)])
        nan = np.isnan(scores) & np.isnan(want)
        differ = (scores.view(np.uint32) != want.view(np.uint32)) & ~nan
        row["scores"] = int(scores.size)
        row["differ"] = int(differ.sum())
        if row["differ"]:
            gap = np.abs(scores.astype(np.float64) - want.astype(np.float64))[differ]
            row["max_abs_diff"] = float(np.nanmax(gap)) if np.isfinite(gap).any() else float("inf")
            worst = max(worst, row["max_abs_diff"])
        rows.append(row)
    checked = [r for r in rows if "unchecked" not in r]
    result = {"ops": len(rows), "checked": len(checked),
              "exact_ops": sum(1 for r in checked if r["differ"] == 0),
              "max_abs_diff": worst, "rows": rows}
    if actual is not None:
        first = user_outputs[0] if first is None else first
        result["same_output"] = bool(
            np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(actual).tobytes())
    result["exact"] = (result["exact_ops"] == result["checked"] == result["ops"]
                       and result.get("same_output", True))
    return result


def check_qk(probe, inputs, actual):
    """Run the q@kT probe (rec["qk_probe"]) on the host runner and compare_qk
    its outputs. A probe that could not be built checks nothing: not exact."""
    if "error" in probe:
        return {"ops": 0, "checked": 0, "exact_ops": 0, "exact": False,
                "error": probe["error"]}
    outputs = run_host_outputs(probe["pte"], probe["ptd"], inputs, probe["method_outputs"])
    user = outputs[len(outputs) - probe["user_outputs"]:]
    # run_host reads the method's first output; so does the comparison.
    return compare_qk(user, probe["ops"], actual, first=outputs[0])


def _weight_key(weight):
    """A parameter's identity under no-op views (.float() on fp32, .contiguous())."""
    return weight.data_ptr(), tuple(weight.shape), tuple(weight.stride())


def _device_quantized_model(net):
    """Put `net` on the device's arithmetic, in place, keeping each module's
    own semantics:

    - every nn.Linear and HF Conv1D at int4 (group-128 fp16 scales over 128
      wide) with int8 activations per row, every nn.Embedding an int4 table:
      the arithmetic of the backend's device reference (test/reference.py);
    - a Linear subclass keeps its own forward. Its F.linear calls take the
      arithmetic below; a forward that is not a linear (DeepSeek-V4's grouped
      projection, a portable fp32 bmm on the device) stays as it is;
    - an nn.Embedding subclass keeps its own forward over the int4 table
      (Gemma's sqrt(width) scale, Inkling's post-lookup norm, T5Gemma 2's
      end-of-image row);
    - F.linear on any of the model's own 2-D parameters (MoE routers, the
      hyper-connection mixers) runs at int4: the delegate claims those matmuls
      like any other. With fp32 routers the reference picks other experts for
      close calls (nanoGPT-OSS at CI size: 8 of 16 token expert sets, cosine
      0.989 against 0.99998 with the routers quantised).

    Attention scores stay fp32, so the int8 Q@K^T is the main difference left
    (check_qk holds each delegated q@kT to its exact mirror instead); anything
    else the patch does not model (3-D expert weights, non-depthwise
    convolutions) stays fp32 too. Returns net."""
    from unittest.mock import patch

    from executorch.backends.esp32s3.test.reference import _device_quantized_reference
    from transformers.pytorch_utils import Conv1D

    # GPT-2's Conv1D is a dense projection whose weight is [in, out]. Keep
    # Conv1D in the exported model, but use a transposed Linear view to apply
    # the same arithmetic patch in the reference. Meta construction avoids
    # initializing an extra copy of its weights.
    for module in net.modules():
        if isinstance(module, Conv1D):
            linear = torch.nn.Linear(*module.weight.shape,
                                     bias=module.bias is not None, device="meta")
            linear.weight = torch.nn.Parameter(module.weight.detach().T,
                                               requires_grad=False)
            if module.bias is not None:
                linear.bias = module.bias
            _device_quantized_reference(linear, dense_format="i4")
            module.forward = linear.forward
    _device_quantized_reference(net, embedding_format="i4", dense_format="i4")
    _embedding_subclass_forwards(net)
    for module in net.modules():
        if isinstance(module, torch.nn.Linear) and type(module).forward is not torch.nn.Linear.forward:
            module.__dict__.pop("forward", None)

    weights = {_weight_key(p) for p in net.parameters() if p.dim() == 2}
    plain, quantized = torch.nn.functional.linear, {}

    def linear(input, weight, bias=None):  # F.linear's own argument names
        key = _weight_key(weight)
        if key not in weights:
            return plain(input, weight, bias)
        if (key, id(bias)) not in quantized:
            lin = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                                  device="meta")
            lin.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
            if bias is not None:
                lin.bias = torch.nn.Parameter(bias.detach(), requires_grad=False)
            quantized[(key, id(bias))] = _device_quantized_reference(lin, dense_format="i4")
        return quantized[(key, id(bias))](input)

    forward = net.forward

    def patched(*args, **kwargs):
        with patch.object(torch.nn.functional, "linear", linear):
            return forward(*args, **kwargs)

    net.forward = patched
    return net


def _embedding_subclass_forwards(net):
    """Give an nn.Embedding subclass its own forward back, over the int4 table.
    The monkeypatch replaces every embedding's forward with a bare gather, which
    drops what a subclass adds: Gemma's scaled embeddings multiply by
    sqrt(width), T5Gemma 2's also swaps in an end-of-image row, Inkling's
    normalises after the lookup. Without this the reference is a different
    model (nanoGemma3 CI: cosine 0.982, 0.99972 with this). A tied output
    head keeps its own patched matmul; only the embedding gets the table."""
    for module in net.modules():
        if (isinstance(module, torch.nn.Embedding) and "forward" in vars(module)
                and type(module).forward is not torch.nn.Embedding.forward):
            table = module.forward(torch.arange(module.num_embeddings))
            del module.forward
            module.weight = torch.nn.Parameter(table, requires_grad=False)


def device_reference(name, size, inputs, seed=0):
    """The family's output under the device's arithmetic
    (_device_quantized_model)."""
    net, _ = model(name, size, seed)
    _device_quantized_model(net)
    with torch.no_grad():
        return net(*inputs)


def compare(actual, reference):
    """How close the delegate's output is to a reference output."""
    a = actual.astype(np.float64).ravel()
    r = reference.astype(np.float64).ravel()
    cosine = float(a @ r / max(np.linalg.norm(a) * np.linalg.norm(r), 1e-30))
    rel = float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-30))
    rec = {"finite": bool(np.isfinite(a).all()), "cosine": round(cosine, 6),
           "rel_l2": round(rel, 6)}
    if actual.ndim == 3:  # [batch, positions, vocab] logits
        rec["top1_agree"] = round(float(
            (actual.argmax(-1) == reference.argmax(-1)).mean()), 4)
    return rec
