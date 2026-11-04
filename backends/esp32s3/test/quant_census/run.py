# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantise one model with one recipe, lower it, run it; one pair per process.

    python -m executorch.backends.esp32s3.test.quant_census.run xnnpack-static resnet8

Writes <recipe>.<model>.json into results_dir() ($QUANT_CENSUS_RESULTS, else a
temporary directory) whatever happens and prints one summary line. Status, by
the first stage that failed:

  quantize-fail  the recipe itself raised (prepare, calibrate, convert, quantize_)
  export-fail    torch.export of the quantised model raised
  lower-fail     to_edge, the strip, partitioning, preprocess or emission raised
  no-kernel      the program lowered but emits ops no kernel in the host runner
                 or firmware implements (quantized_decomposed and torchao ops)
  run-fail       the host runner failed
  mismatch       it ran, but cosine against the recipe's reference < MIN_COSINE
  ok             cosine against the reference >= MIN_COSINE
  n/a            the recipe is not meant for this model (written without torch),
                 or cannot quantise it as its owner ships it (a precondition
                 the recipe asserts, a filter that takes no layer)
  skipped        the recipe cannot run here (Arm needing real tosa_serializer)

--compare-off then lowers the same quantised model again with four defaults
turned off (Stage B, memory_plan="esp32", luts16, fused_norms), runs it, and
writes its arena, delegate temp, peak and output bits beside the pair's
programs (results/work/<recipe>.<model>/off.json; the record itself is written
first and is not touched).
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import traceback

from executorch.backends.esp32s3.test.quant_census import (
    MIN_COSINE,
    ROOT,
    applicable,
    covered,
    result_path,
    results_dir,
)
from executorch.backends.esp32s3.test.quant_census.models import MODELS
from executorch.backends.esp32s3.test.quant_census.recipes import RECIPES

FAILED = {"quantize": "quantize-fail", "export": "export-fail", "lower": "lower-fail",
          "run": "run-fail"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recipe", choices=sorted(RECIPES))
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-plan", choices=("esp32", "none"), default="esp32",
                    help="plan the arena with memory_planning.py (the default) or "
                         "ExecuTorch's own planner (none)")
    ap.add_argument("--compare-off", action="store_true",
                    help="also lower and run with Stage B, luts16, fused_norms "
                         "and memory_plan off (results/work/<pair>/off.json)")
    args = ap.parse_args(argv)
    if args.memory_plan == "none":
        args.memory_plan = None
    if not covered(args.recipe, args.model):
        ap.error(f"{args.model} runs only {', '.join(MODELS[args.model].recipes)}")

    results_dir().mkdir(parents=True, exist_ok=True)
    out_path = result_path(args.recipe, args.model)
    rec = {"recipe": args.recipe, "model": args.model, "seed": args.seed,
           "status": "running", "stage": "start", "seconds": {},
           "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip()}
    started = time.time()
    ran = None
    try:
        reason = applicable(args.recipe, args.model)
        if reason:
            rec.update(status="n/a", stage="done", reason=reason)
        else:
            ran = _run(args, rec)
    except BaseException as e:  # noqa: BLE001 -- the first error is the finding
        from executorch.backends.esp32s3.test.quant_census.recipes import is_stub_error

        rec["status"] = ("skipped" if is_stub_error(e)
                         else FAILED.get(rec["stage"], f"{rec['stage']}-fail"))
        rec["error"] = {"type": type(e).__name__, "message": str(e)[:3000],
                        "where": _frames(e.__traceback__), "causes": _causes(e)}
    finally:
        rec["total_seconds"] = round(time.time() - started, 1)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
        tmp.replace(out_path)
    print(summary(rec))
    if args.compare_off and ran is not None:
        print(compare_off(args, rec, *ran))
    return 0 if rec["status"] in ("ok", "n/a") else 1


# The lowering defaults --compare-off turns off (memory_plan goes separately).
OFF = {"stage_b": False, "luts16": False, "fused_norms": False}


def compare_off(args, rec, q, reference, actual, logits):
    """The pair lowered again with those defaults off (OFF, and ExecuTorch's
    memory planner), run on the host runner: its memory and output beside the
    record's, in results/work/<recipe>.<model>/off.json."""
    import numpy as np
    import torch

    from executorch.backends.esp32s3.test.quant_census import lower, recipes

    options = {**recipes.lowering(args.recipe), **OFF}
    off = {"recipe": args.recipe, "model": args.model, "seconds": {},
           "options": {**options, "memory_plan": None}}
    work = results_dir() / "work" / f"{args.recipe}.{args.model}"
    try:
        exported = torch.export.export(q.program, q.inputs, strict=False)
        program = lower.lower(off, exported, work / "off", memory_plan=None, **options)
        out = lower.run_host(off, program, q.inputs, reference)
        off["vs_reference"] = lower.compare(out, reference, q.info.get("output_qparams"),
                                            logits=logits)
        # A channels-last output comes back as a transposed view: compare bytes
        # of contiguous copies.
        off["same_bits"] = bool(np.array_equal(
            np.ascontiguousarray(out).view(np.uint8),
            np.ascontiguousarray(actual).view(np.uint8)))
    except Exception as e:  # noqa: BLE001 -- recorded, the record stands
        off["error"] = {"type": type(e).__name__, "message": str(e)[:1500]}
    keep = ("arena_bytes", "delegate_scratch_bytes", "peak_bytes", "delegates",
            "ops_outside", "qdq_outside", "macs", "macs_delegated", "vs_reference",
            "same_bits", "error", "options", "recipe", "model")
    off = {k: off[k] for k in keep if k in off}
    off["default"] = {k: rec.get(k) for k in ("arena_bytes", "delegate_scratch_bytes",
                                               "peak_bytes", "delegates", "macs_delegated")}
    (work / "off.json").write_text(json.dumps(off, indent=1, sort_keys=True, default=str))
    return (f"  off: arena {off.get('arena_bytes')} peak {off.get('peak_bytes')} "
            f"(default {rec.get('arena_bytes')} / {rec.get('peak_bytes')}), "
            f"same bits {off.get('same_bits')}"
            + (f", error {off['error']['type']}" if "error" in off else ""))


def _run(args, rec):
    import torch

    from executorch.backends.esp32s3.test.quant_census import lower, models, recipes
    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    torch.set_num_threads(4)
    rec["host_runner"] = {"path": str(HOST_RUNNER),
                          "built": time.strftime("%Y-%m-%d %H:%M", time.localtime(
                              os.path.getmtime(HOST_RUNNER))) if HOST_RUNNER.is_file() else None}

    rec["stage"] = "build"
    rec["params"] = models.meta_params(args.model)
    net, inputs, calibration = models.model(args.model, args.seed)
    rec["inputs"] = [list(x.shape) for x in torch.utils._pytree.tree_flatten(inputs)[0]]
    with torch.no_grad():
        float_out = net(*inputs)
    rec["output_shape"] = list(float_out.shape)
    logits = MODELS[args.model].kind == "transformer"

    rec["stage"] = "quantize"
    t0 = time.perf_counter()
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            q = recipes.quantize(args.recipe, net, inputs, calibration)
    finally:
        # What the quantizer printed (Esp32Quantizer's weight-only notice).
        rec["recipe_messages"] = [l.strip()[:400] for l in said.getvalue().splitlines()
                                  if l.strip()][-12:]
    rec["recipe_info"] = q.info
    if q.info.get("recipe_limit"):
        # A precondition the recipe itself asserts, checked first: it cannot
        # quantise this model as its owner ships it.
        rec.update(status="n/a", stage="done", reason=q.info["recipe_limit"])
        return
    if q.info.get("quantized_linears") == 0:
        # quantize.py's filter took no layer (in_features not a multiple of
        # the group size): the "quantised" model is the float model.
        rec.update(status="n/a", stage="done",
                   reason=f"the recipe quantises no layer of this model "
                          f"(left float: {', '.join(q.info['left_float'])})")
        return
    reference = None
    try:
        with torch.no_grad():
            reference = q.reference(*q.inputs)
        rec["reference_from"] = "the quantized model, eager"
    except Exception as e:  # noqa: BLE001
        # convert_pt2e leaves a dynamic chain's eps as a tensor argument that
        # the converted module's generated code cannot spell (NameError
        # 'float32'); export_pt2e.py runs the re-exported module for the same
        # reason. The same graph, so the same semantics.
        rec["reference_from"] = (f"torch.export of the quantized model: calling it "
                                 f"directly raised {type(e).__name__}: {str(e)[:160]}")
    rec["seconds"]["quantize"] = round(time.perf_counter() - t0, 1)

    rec["stage"] = "export"
    t0 = time.perf_counter()
    exported = torch.export.export(q.program, q.inputs, strict=False)
    if reference is None:
        with torch.no_grad():
            reference = exported.module()(*q.inputs)
    # What the recipe itself costs this random-weight model (information).
    rec["reference_vs_float"] = lower.compare(reference, float_out, logits=logits)
    rec["seconds"]["export"] = round(time.perf_counter() - t0, 1)

    rec["stage"] = "lower"
    work = results_dir() / "work" / f"{args.recipe}.{args.model}"
    program = lower.lower(rec, exported, work, memory_plan=args.memory_plan,
                          **recipes.lowering(args.recipe))
    if rec["ops"]["no_kernel"]:
        rec["status"], rec["stage"] = "no-kernel", "done"
        return

    rec["stage"] = "run"
    t0 = time.perf_counter()
    actual = lower.run_host(rec, program, q.inputs, reference)
    rec["seconds"]["run"] = round(time.perf_counter() - t0, 1)
    rec["vs_reference"] = lower.compare(actual, reference, q.info.get("output_qparams"),
                                        logits=logits)
    rec["vs_float"] = lower.compare(actual, float_out, logits=logits)
    good = rec["vs_reference"]["finite"] and rec["vs_reference"]["cosine"] >= MIN_COSINE
    rec["status"], rec["stage"] = ("ok" if good else "mismatch"), "done"
    return q, reference, actual, logits


def _causes(error, keep=3):
    """The exceptions `error` was raised from (pass managers wrap the real one)."""
    out, seen = [], {id(error)}
    error = error.__cause__ or error.__context__
    while error is not None and id(error) not in seen and len(out) < keep:
        seen.add(id(error))
        out.append({"type": type(error).__name__, "message": str(error)[:1500],
                    "where": _frames(error.__traceback__, keep=4)})
        error = error.__cause__ or error.__context__
    return out


def _frames(tb, keep=6):
    """'file:line function' for the frames in this checkout (the census and
    backend code, where the cause usually is), then the innermost."""
    rows = [f"{f.filename.replace(str(ROOT) + '/', '')}:{f.lineno} {f.name}"
            for f in traceback.extract_tb(tb)]
    ours = [r for r in rows if not r.startswith("/")]
    return ours[-keep:] + ["--- innermost:"] + rows[-keep:]


def summary(rec):
    parts = [f"census {rec['recipe']} {rec['model']}: {rec['status']}"]
    if "delegates" in rec:
        share = rec.get("mac_share")
        parts.append(f"{rec['delegates']} delegates"
                     + (f" ({100 * share:.0f}% of MACs)" if share is not None else "")
                     + f", {rec['ops_outside']} ops + {rec['qdq_outside']} q/dq outside")
    if rec.get("ops", {}).get("no_kernel"):
        parts.append("no kernel: " + ", ".join(rec["ops"]["no_kernel"]))
    if "arena_bytes" in rec:
        parts.append(f"arena {rec['arena_bytes']:,} B"
                     + (f", peak {rec['peak_bytes']:,} B" if "peak_bytes" in rec else ""))
    if "vs_reference" in rec:
        parts.append(f"cosine {rec['vs_reference']['cosine']} vs reference, "
                     f"{rec['vs_float']['cosine']} vs float")
    if "error" in rec:
        parts.append(f"at {rec['stage']}: {rec['error']['type']}: "
                     f"{rec['error']['message'][:200]}")
    if "reason" in rec:
        parts.append(rec["reason"])
    return "; ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
