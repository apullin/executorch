# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower one nano family and run it on the host runner; one family per process.

    python -m executorch.examples.espressif.nano_zoo.run llama3_2 \\
        [--size device|ci] [--no-host] [--output-hash]

Writes results/<family>.<size>.json beside this file whatever happens: the status
(ok, fail, refused, no-kernel, mismatch), the stage it reached, sizes,
delegation counts, timings and, when it ran, how close the host runner's output
is to the fp32 eager model's and to the device-arithmetic reference, and whether
each delegated q@kT's scores are its exact mirror's (build.check_qk). Prints one
summary line.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from executorch.examples.espressif.nano_zoo import PARAM_CAP, ROOT, family

RESULTS = Path(__file__).resolve().parent / "results"
# The gate compares the host runner with the device-arithmetic reference
# (build.device_reference), not with fp32: int4 weights alone move a random
# 24-layer nanoLlama3.2 to cosine 0.92 against fp32, while the delegate sits at
# 0.9998 against the reference. A wrong kernel or a dropped op lands far below.
# The q@kT is the exception: the scores barely move a random-init model's
# output, so build.check_qk holds each delegated q@kT to its mirror bit for bit.
MIN_COSINE = 0.99


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("family")
    ap.add_argument("--size", choices=("device", "ci"), default="device")
    ap.add_argument("--no-host", action="store_true", help="lower only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-hash", action="store_true",
                    help="also record the sha256 of the host runner's output bytes "
                         "(to check that a change keeps every output bit for bit)")
    args = ap.parse_args(argv)

    family(args.family)  # a typo fails here, before torch loads
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{args.family}.{args.size}.json"
    work = RESULTS / "work" / f"{args.family}.{args.size}"
    rec = {"family": args.family, "size": args.size, "status": "running", "stage": "start",
           "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip()}
    started = time.time()
    try:
        _run(args, rec, work)
    except BaseException as e:  # noqa: BLE001 -- the first error is the finding
        rec["status"] = "refused" if isinstance(e, ValueError) and "PARAM_CAP" in str(e) else "fail"
        rec["error"] = {"type": type(e).__name__, "message": str(e)[:3000],
                        "where": _frames(e.__traceback__)}
    finally:
        rec["total_seconds"] = round(time.time() - started, 1)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
        tmp.replace(out_path)
    print(summary(rec))
    return 0 if rec["status"] == "ok" else 1


def _run(args, rec, work):
    import torch
    import transformers

    from executorch.examples.espressif.nano_zoo import build

    torch.set_num_threads(4)
    rec["transformers"] = transformers.__version__

    rec["stage"] = "meta"
    params, largest = build.meta_params(args.family, args.size)
    rec.update(params=params, largest=largest, est_weight_bytes=build.weight_bytes(params))
    if params > PARAM_CAP:
        raise ValueError(f"{params:,} parameters > PARAM_CAP {PARAM_CAP:,}; largest: {largest}")

    rec["stage"] = "build"
    net, inputs = build.model(args.family, args.size, args.seed)
    rec["inputs"] = {n: list(x.shape) for n, x in zip(net.names, inputs)}

    rec["stage"] = "eager"
    with torch.no_grad():
        reference = net(*inputs)
    rec["output_shape"] = list(reference.shape)

    rec["stage"] = "lower"
    rec.update(build.lower(net, inputs, work))
    if rec["ops"]["no_kernel"]:
        rec["status"] = "no-kernel"
        return
    if args.no_host:
        rec["status"] = "ok"
        return

    rec["stage"] = "host"
    started = time.perf_counter()
    actual = build.run_host(rec["pte"], rec["ptd"], inputs, reference)
    rec["seconds"]["host"] = round(time.perf_counter() - started, 1)
    if args.output_hash:
        rec["host_output_sha256"] = hashlib.sha256(actual.tobytes()).hexdigest()
    # Against fp32: what int4/int8 costs this model (information, not a gate).
    rec["vs_fp32"] = build.compare(actual, reference.numpy())

    rec["stage"] = "reference"
    device = build.device_reference(args.family, args.size, inputs, args.seed)
    rec["vs_device"] = build.compare(actual, device.numpy())
    ok = rec["vs_device"]["finite"] and rec["vs_device"]["cosine"] >= MIN_COSINE

    # The cosine bar cannot see the int8 q@kT (build.py, "q@kT probe"): each
    # delegated q@kT must equal its exact mirror, bit for bit.
    if rec.get("qk_probe"):
        rec["stage"] = "qk"
        started = time.perf_counter()
        rec["qk"] = build.check_qk(rec["qk_probe"], inputs, actual)
        rec["seconds"]["qk"] = round(time.perf_counter() - started, 1)
        ok = ok and rec["qk"]["exact"]
    rec["status"] = "ok" if ok else "mismatch"
    rec["stage"] = "done"


def _frames(tb, keep=6):
    """'file:line function' for the frames outside torch and executorch (the
    model and backend code, where the cause usually is), then the innermost."""
    rows = [f"{f.filename.replace(str(ROOT) + '/', '')}:{f.lineno} {f.name}"
            for f in traceback.extract_tb(tb)]
    outside = [r for r in rows if "/torch/" not in r and "/executorch/" not in r]
    return outside[-keep:] + ["--- innermost:"] + rows[-keep:]


def summary(rec):
    size = rec.get("pte_bytes", 0) + rec.get("ptd_bytes", 0)
    parts = [f"zoo {rec['family']} {rec['size']}: {rec['status']}"]
    if "params" in rec:
        parts.append(f"{rec['params'] / 1e6:.1f}M params")
    if size:
        parts.append(f"{size / 1e6:.2f} MB" + ("" if rec.get("fits_flash") else " (too big)"))
    if "delegates" in rec:
        parts.append(f"{rec['delegates']} delegates, {rec['real_portable_ops']} portable ops")
    if "vs_device" in rec:
        parts.append(f"cosine {rec['vs_device']['cosine']} vs device arithmetic, "
                     f"{rec['vs_fp32']['cosine']} vs fp32")
    if "qk" in rec:
        qk = rec["qk"]
        if "error" in qk:
            parts.append(f"q@kT probe failed: {qk['error'][:120]}")
        else:
            parts.append(f"q@kT {qk['exact_ops']}/{qk['ops']} exact"
                         + ("" if qk.get("same_output", True) else ", probe output differs"))
    if "seconds" in rec:
        parts.append(" ".join(f"{k} {v}s" for k, v in rec["seconds"].items()))
    if "error" in rec:
        parts.append(f"at {rec['stage']}: {rec['error']['type']}: {rec['error']['message'][:200]}")
    return "; ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
