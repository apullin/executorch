#!/usr/bin/env python3

# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export models for the bench firmware (run_bench_hardware.py): each one's two
flash partitions, checked on the host first.

    export_bench.py ds_cnn mobilenet_v1_025 resnet8 mobilenet_v2_96 --out /tmp/bench

Each model is a census model (backends/esp32s3/test/quant_census/models.py: its
seeded weights and BatchNorm statistics), quantised by Arm's a8w8 recipe as the
census quantises it -- static int8 per-tensor activations, int8 per-channel
symmetric weights -- and lowered as the CNN parity rows
are (fixtures/cnn.lower: Esp32Partitioner at its defaults, the weights tagged
external, memory_plan="esp32"). The recipe calibrates on the census's eight
batches and on the board's input, so the input sits inside every grid.

The board's input is generated (run_bench_hardware.bench_input): uniform in
[-1, 1), the census's distribution for these models. The reference is the
host runner's output on it, bit for bit (the delegate's and the portable
kernels' own C++; fixtures/cnn.py says why that is the device's arithmetic).

Each model's directory gets:

  bench_image.bin          the program partition (0x43): header, labels, program
  bench_weights.ptd        the weights partition (0x44)
  bench.pte                the program alone
  bench_required_ops.txt   the operators the firmware must register
  bench_model.json         what the export measured: MACs (each delegate's too),
                           arena, delegate scratch, ops left between the
                           delegates, the output against the recipe's eager
                           model, the host run of the image, and whether it
                           was exported for a dual-core build (min_latency)

`--min-latency` exports for a dual-core bench firmware (run_bench_hardware.py
build --dual-core, which refuses images exported otherwise, and a single-core
build these): every delegate is compiled for min_latency, so it splits its
projections' columns across both cores from init. No convolution splits (on
the board a split conv gave wrong outputs and cache-error panics: the backend
README's Known limits), so a delegate's planned scratch is the single-core
one's either way and nothing falls back to the temp allocator ("planned
scratch unusable"). The export holds the host runner's output to its
single-core bits (--single-core) and its every call to planned scratch.

The host run is the firmware's bench runner itself (main/bench_runner.inc),
built for the host as esp32_bench_host beside the host runner: both
measurements must be exact, every instruction labelled, every placement made.
Imports torch; several models export in a child process each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_bench_hardware as rbh  # noqa: E402

RECIPE = "arm-a8w8"
INPUT_SEED = 1


def ensure_host_bench(path=None):
    """The host build of the bench runner (esp32_bench_host), brought up to
    date as fixtures/cnn.host_runner() brings the host runner: beside the
    host runner unless `path` names it, rebuilt from this tree's sources when
    its CMake build directory is there. None when there is no such binary."""
    from executorch.backends.esp32s3.test.fixtures import cnn

    runner = cnn.host_runner()
    path = Path(path) if path else runner.parent / "esp32_bench_host"
    if (path.parent / "CMakeCache.txt").is_file():
        done = subprocess.run(["cmake", "--build", str(path.parent), "-j4", "--target",
                               "esp32_bench_host"], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"esp32_bench_host build FAILED:\n{done.stdout[-2000:]}"
                               f"{done.stderr[-2000:]}")
    return path if path.is_file() else None


def instruction_table(et, gm):
    """One label per instruction of the plan, "<kind> <text>": D a delegate
    (with its MACs), Q a portable quantize or dequantize, P any other
    portable operator (with its output's shape), M a move, free or jump.
    Returns (labels, the delegates' MACs in delegate order)."""
    from executorch.exir import schema

    from executorch.backends.esp32s3.test.quant_census.lower import node_macs

    delegate_macs = []
    for n in gm.graph.nodes:
        if n.op == "call_function" and "executorch_call_delegate" in str(n.target):
            lowered = getattr(gm, n.args[0].target)
            delegate_macs.append(sum(node_macs(m) for m in lowered.original_module.graph.nodes))
    plan = et.executorch_program.execution_plan[0]
    if len(plan.chains) != 1:
        raise RuntimeError(f"{len(plan.chains)} chains: the bench steps one")
    labels, seen = [], {}
    for instr in plan.chains[0].instructions:
        args = instr.instr_args
        if isinstance(args, schema.DelegateCall):
            index = args.delegate_index
            if index not in seen:  # delegate_macs is in graph order: match by first call
                seen[index] = len(seen)
            macs = delegate_macs[seen[index]] if seen[index] < len(delegate_macs) else 0
            labels.append(f"D delegate {index} ({plan.delegates[index].id}): {macs:,} MACs")
        elif isinstance(args, schema.KernelCall):
            op = plan.operators[args.op_index]
            name = f"{op.name}.{op.overload}" if op.overload else op.name
            kind = "Q" if "quantize_per_" in name and name.startswith("quantized_decomposed") else "P"
            shape = ""
            out = plan.values[args.args[-1]].val if args.args else None
            if isinstance(out, schema.Tensor):
                shape = " [" + ",".join(str(s) for s in out.sizes) + "]"
            labels.append(f"{kind} {name}{shape}")
        elif isinstance(args, schema.MoveCall):
            labels.append("M move")
        elif isinstance(args, schema.FreeCall):
            labels.append("M free")
        else:
            labels.append(f"M {type(args).__name__}")
    if len(seen) != len(delegate_macs):
        raise RuntimeError(f"{len(seen)} delegates called, {len(delegate_macs)} in the graph")
    return labels, [delegate_macs[seen[i]] for i in sorted(seen)]


def run_host_bench(path, model_dir, timed=2, stepped=1):
    """The image through esp32_bench_host: (its log, parsed)."""
    done = subprocess.run([str(path), "--image", str(model_dir / rbh.IMAGE_FILE),
                           "--ptd", str(model_dir / rbh.WEIGHTS_FILE),
                           "--timed-runs", str(timed), "--stepped-runs", str(stepped)],
                          capture_output=True, text=True, timeout=1800)
    log = done.stdout + done.stderr
    if done.returncode != 0:
        raise RuntimeError(f"esp32_bench_host exit {done.returncode}: {log[-3000:]}")
    return log, rbh.parse_log(log)


def check_host_bench(name, parsed, labels):
    """Both measurements exact, the split over every labelled instruction,
    the psram measurement's weights placed by the apply call, and no call
    off its planned scratch."""
    problems = list(parsed["errors"])
    if not parsed["done"]:
        problems.append("no BENCH_DONE")
    for phase in ("flash", "psram"):
        p = parsed["phases"].get(phase)
        if p is None:
            problems.append(f"no {phase} measurement")
            continue
        if p.get("check", {}).get("result") != "EXACT":
            problems.append(f"{phase}: {p.get('check')}")
        if len(p["instr"]) != len(labels):
            problems.append(f"{phase}: {len(p['instr'])} instructions timed, {len(labels)} labelled")
        if p.get("split", {}).get("same_bits") not in (None, "1/1", "2/2", "3/3"):
            problems.append(f"{phase}: stepped runs disagree {p['split']}")
    psram = parsed["phases"].get("psram", {}).get("load", {})
    if psram.get("weight_copies", 0) == 0:
        problems.append(f"psram: no weight copied {psram}")
    if problems:
        raise RuntimeError(f"{name}: the host run of its image: {problems}")


def export(name, out_dir, timed_runs=None, stepped_runs=None, host_bench=None,
           min_latency=False):
    """Quantise, lower, reference, label, pack and host-check one model; for
    a dual-core firmware with `min_latency` (the module's docstring)."""
    import numpy as np
    import torch

    from executorch.backends.esp32s3.test.fixtures import cnn
    from executorch.backends.esp32s3.test.quant_census import MIN_COSINE, models, recipes
    from executorch.backends.esp32s3.test.quant_census.lower import compare
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    t0 = time.time()
    shape = models.input_shape(name)
    params = models.meta_params(name)
    numel = int(np.prod(shape))
    x = torch.from_numpy(rbh.bench_input(numel, INPUT_SEED).reshape(shape))
    net, _, calibration = models.model(name, 0)
    with cnn.one_thread():
        q = recipes.quantize(RECIPE, net, (x,), [(x,)] + list(calibration))
    et, named, report, arena = cnn.lower(q.program, q.inputs, min_latency=min_latency)
    program = bytes(et.buffer)
    labels, delegate_macs = instruction_table(et, et.exported_program().graph_module)
    reference, temp = cnn.run_host(cnn.host_runner(), program, named, q.inputs[0],
                                   ("--single-core",))
    # The image's temp must hold a dual-core build's too (a handle initialised
    # under min_latency; it reserves no more, as no conv splits). Its
    # output must be the single-core run's, byte for byte.
    split, split_temp, scratch = cnn.run_host(cnn.host_runner(), program, named,
                                              q.inputs[0], ("--min-latency",),
                                              scratch=True)
    if split.tobytes() != reference.tobytes():
        raise RuntimeError(f"{name}: --min-latency changes the host runner's output")
    temp = max(temp, split_temp)
    if min_latency and scratch[2]:
        raise RuntimeError(f"{name}: exported for min_latency, yet {scratch[2]} split "
                           f"call(s) found their planned scratch too small")
    with torch.no_grad():
        eager = q.program(*q.inputs)
    gap = compare(reference, eager, q.info.get("output_qparams"))
    if not gap["finite"]:
        raise RuntimeError(f"{name}: the host runner's output is not finite")
    macs = report["macs"]
    if timed_runs is None:
        timed_runs = 5 if macs > 100_000_000 else 10
    if stepped_runs is None:
        stepped_runs = 1 if macs > 100_000_000 else 3
    image = rbh.pack_image(
        name=name, program=program, weights=named, input_numel=numel, input_seed=INPUT_SEED,
        input_bits=rbh.fnv1a(np.ascontiguousarray(x.numpy(), dtype="<f4").tobytes()),
        output=reference, timed_runs=timed_runs, stepped_runs=stepped_runs, temp_bytes=temp,
        labels=labels, macs=macs, macs_delegated=report["macs_delegated"])
    model_dir = Path(out_dir) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / rbh.IMAGE_FILE).write_bytes(image)
    (model_dir / rbh.WEIGHTS_FILE).write_bytes(named)
    (model_dir / "bench.pte").write_bytes(program)
    ops = sorted(ops_of_program(et))
    (model_dir / rbh.OPS_FILE).write_text(
        f"# Generated by export_bench.py for {name}.\n" + "".join(op + "\n" for op in ops))
    header = rbh.read_image(image)
    record = {
        "name": name, "recipe": RECIPE, "seed": 0, "params": params, "input_shape": list(shape),
        "input_seed": INPUT_SEED, "input_bits": f"{header['input_bits']:08x}",
        "output_numel": header["output_numel"], "output_bits": f"{header['output_bits']:08x}",
        "output_checksum": header["output_checksum"], "macs": macs,
        "macs_delegated": report["macs_delegated"], "delegate_macs": delegate_macs,
        "delegates": report["delegates"], "outside": report["outside"], "qdq": report["qdq"],
        "arena": arena, "temp": temp, "instructions": labels, "gap": gap,
        "census_bar": gap["cosine"] >= MIN_COSINE, "program_bytes": len(program),
        "weights_bytes": len(named), "image_bytes": len(image),
        "timed_runs": timed_runs, "stepped_runs": stepped_runs, "ops": ops,
        "min_latency": min_latency,
        "split_scratch": {"planned": scratch[0], "temp": scratch[1],
                          "unusable": scratch[2]},
        "sha256": {f: hashlib.sha256((model_dir / f).read_bytes()).hexdigest()
                   for f in (rbh.IMAGE_FILE, rbh.WEIGHTS_FILE, "bench.pte")},
        "tree": rbh.tree_state(HERE),
    }
    path = ensure_host_bench(host_bench)
    if path is not None:
        log, parsed = run_host_bench(path, model_dir)
        (model_dir / "host_bench.log").write_text(log)
        check_host_bench(name, parsed, labels)
        record["host_bench"] = {
            phase: {"check": p.get("check"), "split": p.get("split"), "load": p.get("load")}
            for phase, p in parsed["phases"].items()}
    else:
        record["host_bench"] = "not run: no esp32_bench_host (build the host runner's targets)"
    record["seconds"] = round(time.time() - t0, 1)
    (model_dir / rbh.MODEL_FILE).write_text(json.dumps(record, indent=1))
    print(f"  {name}: {params:,} parameters, {macs:,} MACs ({report['macs_delegated']:,} in "
          f"{report['delegates']} delegate(s)); program {len(program):,} B, weights "
          f"{len(named):,} B, image {len(image):,} B; arena {arena:,} B, delegate scratch "
          f"{temp:,} B; {len(labels)} instructions, outside {report['outside']} "
          f"{report['qdq']}; vs eager cosine {gap['cosine']}; host bench "
          f"{'EXACT both ways' if isinstance(record['host_bench'], dict) else record['host_bench']}"
          f"; {record['seconds']} s", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="census models (quant_census/models.py)")
    parser.add_argument("--out", required=True, help="one directory per model goes here")
    parser.add_argument("--timed-runs", type=int, help="default 10 (5 past 100M MACs)")
    parser.add_argument("--stepped-runs", type=int, help="default 3 (1 past 100M MACs)")
    parser.add_argument("--host-bench", help="esp32_bench_host (default: beside the host runner)")
    parser.add_argument("--min-latency", action="store_true",
                        help="for a dual-core firmware (build --dual-core): every delegate "
                             "splits from init, its planned scratch holding the worker's")
    args = parser.parse_args(argv)
    if len(args.models) == 1:
        export(args.models[0], args.out, args.timed_runs, args.stepped_runs, args.host_bench,
               args.min_latency)
        return 0
    # One process per model: a lowering's buffers stay reachable from upstream
    # caches for the life of the process. The children inherit this process's
    # environment, so they import the same executorch.
    failed = []
    for name in args.models:
        command = [sys.executable, os.path.abspath(__file__), name, "--out", args.out]
        for flag, value in (("--timed-runs", args.timed_runs), ("--stepped-runs", args.stepped_runs),
                            ("--host-bench", args.host_bench)):
            if value is not None:
                command += [flag, str(value)]
        command += ["--min-latency"] if args.min_latency else []
        if subprocess.run(command).returncode != 0:
            failed.append(name)
    if failed:
        print(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
