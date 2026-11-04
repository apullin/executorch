#!/usr/bin/env python3

# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time single models on the board: the bench firmware, its images, its logs.

    export_bench.py ds_cnn resnet8 --out /tmp/bench           # each model's two partitions
    run_bench_hardware.py build /tmp/bench/* --build build_bench
    run_bench_hardware.py run /tmp/bench/ds_cnn --build build_bench --serial <USB serial> --boots 3
    run_bench_hardware.py summary build_bench/logs/*.log

`build` makes one lean firmware (ET_PLE_ONLY, ET_BENCH_MODEL; main/bench_runner.inc)
whose selective build registers every operator the named models use. A model is
then two flash partitions written beside that firmware: its bench image in the
program partition (0x43: a header, a label per instruction, the program) and
its named data, the weights, in the weights partition (0x44). `run` flashes the
firmware with one model's partitions, captures `--boots` boots (the capture
resets the board between them), parses each log and prints the summary;
`summary` does the same for saved logs.

Each boot measures the model twice on the same buffers and at the same PSRAM
timing point: with every weight read from flash, then with every weight PSRAM
holds placed by the backend's ranked placement (weight_tier_apply, before any
timed execute). Each measurement checks the output bits against the host
runner's, times `timed_runs` executes and splits `stepped_runs` more by
instruction (delegates, portable operators, quantize/dequantize between them).

The firmware build reads what the story firmware's does: an ESP-IDF
environment (IDF_ENV_SCRIPT, or an activated shell), and for a tree without
submodules ESPRESSIF_FLATBUFFERS_INCLUDE and ESPRESSIF_LVGL_DIR (with
IDF_COMPONENT_MANAGER=0). Its profile -- -Os or -O2 for the backend, the
runtime's Kconfig profile, the caches -- is printed in every log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)  # examples/espressif/esp32s3
STORY_DEFAULTS = os.path.join(PROJECT, "sdkconfig.story.defaults")
CACHE_DEFAULTS = os.path.join(PROJECT, "sdkconfig.cache.defaults")
APP_LIMIT = 0x160000  # the lean table's factory partition

# The bench image (main/bench_runner.inc reads the same 128-byte header).
IMAGE_MAGIC = b"ESPBENCH"
IMAGE_VERSION = 1
IMAGE_HEADER = struct.Struct("<8s10Id6I2Q32s")
IMAGE_ALIGN = 64
# The most timed or stepped runs an image may ask for (bench_runner.inc
# kMaxRuns, which the firmware's parser checks).
MAX_RUNS = 1000
IMAGE_FILE = "bench_image.bin"
WEIGHTS_FILE = "bench_weights.ptd"
MODEL_FILE = "bench_model.json"
OPS_FILE = "bench_required_ops.txt"
assert IMAGE_HEADER.size == 128


def fnv1a(data: bytes, h: int = 0x811C9DC5) -> int:
    """FNV-1a 32 over `data`, as the firmware hashes outputs and weights."""
    for byte in data:
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def bench_input(numel: int, seed: int):
    """The firmware's input: element i is lowbias32(i + seed * 0x9E3779B9),
    its top 24 bits scaled into [-1, 1), exactly representable in float32
    (bench_runner.inc input_value)."""
    import numpy as np

    with np.errstate(over="ignore"):
        h = (np.arange(numel, dtype=np.uint32) + np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF))
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0 ** -23) - np.float32(1.0)


def output_checksum(values) -> float:
    """sum(v[i] * (i + 1)) in float64, in order, as the firmware sums it."""
    total = 0.0
    for i, v in enumerate(values.tolist()):
        total += v * (i + 1)
    return total


def pack_image(*, name: str, program: bytes, weights: bytes, input_numel: int, input_seed: int,
               input_bits: int, output, timed_runs: int, stepped_runs: int, temp_bytes: int,
               labels: list, macs: int, macs_delegated: int) -> bytes:
    """The bench image: the header, NUL-terminated labels, padding to
    IMAGE_ALIGN, the program."""
    import numpy as np

    for field, runs in (("timed_runs", timed_runs), ("stepped_runs", stepped_runs)):
        if not 0 <= runs <= MAX_RUNS:
            raise ValueError(f"{field} {runs} is out of range: the firmware takes "
                             f"0 to {MAX_RUNS} (bench_runner.inc kMaxRuns)")

    out = np.ascontiguousarray(output, dtype="<f4").ravel()
    label_blob = b"".join(label.encode("ascii", "replace") + b"\0" for label in labels)
    # The header and labels pad to IMAGE_ALIGN, so the program starts aligned in the partition.
    header_bytes = -(-(IMAGE_HEADER.size + len(label_blob)) // IMAGE_ALIGN) * IMAGE_ALIGN
    encoded = name.encode("ascii")
    if len(encoded) >= 32:
        raise ValueError(f"model name {name!r} too long for the image (31 bytes)")
    header = IMAGE_HEADER.pack(
        IMAGE_MAGIC, IMAGE_VERSION, header_bytes, len(program), len(weights), fnv1a(weights),
        input_numel, input_seed, input_bits, out.size, fnv1a(out.tobytes()),
        output_checksum(out), timed_runs, stepped_runs, temp_bytes, len(labels),
        IMAGE_HEADER.size, len(label_blob), macs, macs_delegated, encoded)
    body = header + label_blob
    return body + b"\0" * (header_bytes - len(body)) + program


def read_image(data: bytes) -> dict:
    """The header fields, the labels and the program of a bench image."""
    fields = ("magic", "version", "header_bytes", "program_bytes", "weights_bytes",
              "weights_bits", "input_numel", "input_seed", "input_bits", "output_numel",
              "output_bits", "output_checksum", "timed_runs", "stepped_runs", "temp_bytes",
              "instructions", "labels_offset", "labels_bytes", "macs", "macs_delegated", "name")
    h = dict(zip(fields, IMAGE_HEADER.unpack_from(data)))
    if h["magic"] != IMAGE_MAGIC or h["version"] != IMAGE_VERSION:
        raise ValueError("not a version-1 bench image")
    h["name"] = h["name"].rstrip(b"\0").decode()
    blob = data[h["labels_offset"]:h["labels_offset"] + h["labels_bytes"]]
    h["labels"] = [s.decode() for s in blob.split(b"\0")[:h["instructions"]]]
    h["program"] = data[h["header_bytes"]:h["header_bytes"] + h["program_bytes"]]
    return h


# --- the board's log ---------------------------------------------------------------------

_FIELD = re.compile(r"(\w+)=(\S+)")
_MSPI = re.compile(r"\[mspi\] psram din_mode (0x[0-9a-f]+) din_num (0x[0-9a-f]+)")
# The conv path's boot self-tests (esp32_conv.cpp pie_dw_init), by the name
# parse_log files each verdict under, with the text of the line that reports
# it: where the conv path ran, each must say PASS, or its convs took the code
# the kernels replace -- the same bits at that code's speed.
SELFTEST_LINES = (("depthwise", b"PIE depthwise 16-lane selftest"),
                  ("fused dot and epilogue", b"PIE conv kernels selftest"),
                  ("transpose", b"PIE conv kernels selftest"),
                  ("long-row dot", b"PIE long-row dot selftest"),
                  # The depthwise plane kernels, on the same line.
                  ("depthwise planes", b", depthwise planes %s"))
SELFTESTS = tuple(name for name, _ in SELFTEST_LINES)
_PIE_DW = re.compile(r"PIE depthwise 16-lane selftest: (\w+)")
_PIE_KERNELS = re.compile(
    r"PIE conv kernels selftest: fused dot and epilogue (\w+).*?, transpose (\w+)")
_PIE_LONG = re.compile(r"PIE long-row dot selftest: (\w+)")
_PIE_PLANES = re.compile(r"PIE conv kernels selftest: .*?, depthwise planes (\w+)")
_CONVFAST = re.compile(r"convfast: (\d+) conv op")
# One line of esp32_backend_report_timing's "timing by shape" table (--backend-timing).
_SHAPE = re.compile(r"(\S+)\s+m=(\d+)\s+k=(\d+)\s+n=(\d+)\s+x(\d+)\s+(\d+) call\(s\)\s+([\d.]+) us/call")


def _fields(text):
    out = {}
    for key, value in _FIELD.findall(text):
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def parse_log(text: str, required=SELFTESTS) -> dict:
    """Every BENCH_ line of one boot, by phase. Where the conv path ran, each
    self-test in `required` must have said PASS (firmware_selftests)."""
    log = {"phases": {}, "errors": [], "mem": [], "done": False, "model": {}, "profile": None,
           "mspi": None, "heap": {}, "micro": [], "shapes": {}}
    timing = None  # the phase whose backend timing report is being read
    selftests = {}     # the conv path's self-tests: their verdicts
    conv_ops = 0       # the most conv ops a convfast line counted
    for raw in text.splitlines():
        line = raw.strip()
        m = _PIE_DW.search(line)
        if m:
            selftests["depthwise"] = m.group(1)
        m = _PIE_KERNELS.search(line)
        if m:
            selftests["fused dot and epilogue"], selftests["transpose"] = m.groups()
        m = _PIE_LONG.search(line)
        if m:
            selftests["long-row dot"] = m.group(1)
        m = _PIE_PLANES.search(line)
        if m:
            selftests["depthwise planes"] = m.group(1)
        m = _CONVFAST.search(line)
        if m:
            conv_ops = max(conv_ops, int(m.group(1)))
        m = _MSPI.search(line)
        if m:
            log["mspi"] = f"din_mode {m.group(1)} din_num {m.group(2)}"
            continue
        if "planned scratch unusable (" in line:
            # The backend's one-time line: a delegate's planned scratch is too
            # small for this posture (a dual-core firmware running a program
            # exported without min_latency), so every call took the temp
            # allocator's: not the configuration being measured. Not the
            # counters' "esp32 scratch: ... 0 fallback(s) (planned scratch
            # unusable)" after the bench, which labels the count.
            log["errors"].append(line)
            continue
        if not line.startswith("BENCH_"):
            m = _SHAPE.match(line) if timing else None
            if m:
                op, mm, k, n, members, calls = m.group(1), *map(int, m.groups()[1:6])
                log["shapes"].setdefault(timing, []).append(
                    {"op": op, "m": mm, "k": k, "n": n, "members": members, "calls": calls,
                     "us": float(m.group(7))})
            continue
        tag, _, rest = line.partition(" ")
        timing = _fields(rest).get("phase") if tag == "BENCH_TIMING" else None
        if tag == "BENCH_ERROR":
            log["errors"].append(rest)
        elif tag == "BENCH_MODEL":
            log["model"] = _fields(rest)
        elif tag == "BENCH_PROFILE":
            log["profile"] = rest
        elif tag == "BENCH_MEM":
            log["mem"].append(rest)
        elif tag == "BENCH_MICRO":
            log["micro"].append(rest)
        elif tag == "BENCH_HEAP":
            when, _, values = rest.partition(" ")
            log["heap"][when] = _fields(values)
        elif tag == "BENCH_DONE":
            log["done"] = True
        elif tag in ("BENCH_PHASE", "BENCH_CHECK", "BENCH_RUN", "BENCH_TIME", "BENCH_INSTR",
                     "BENCH_SPLIT"):
            fields = _fields(rest)
            phase = log["phases"].setdefault(fields.get("phase", "?"), {
                "runs": [], "instr": [], "tiers": []})
            if tag == "BENCH_PHASE":
                phase["load"] = fields
                phase["skipped"] = "skipped" in rest
            elif tag == "BENCH_CHECK":
                phase["check"] = fields
            elif tag == "BENCH_RUN":
                phase["runs"].append(fields["us"])
            elif tag == "BENCH_TIME":
                phase["time"] = fields
            elif tag == "BENCH_SPLIT":
                phase["split"] = fields
            else:  # BENCH_INSTR phase=.. i=.. us=.. <kind> <label>
                m = re.match(r"phase=\S+ i=(\d+) us=(\S+) (\S)\s?(.*)$", rest)
                if m:
                    phase["instr"].append((int(m.group(1)), float(m.group(2)), m.group(3),
                                           m.group(4)))
    if conv_ops > 0:
        for name in required:
            if selftests.get(name) != "PASS":
                log["errors"].append(f"PIE {name} selftest: "
                                     f"{selftests.get(name, 'never printed')}")
    log["selftests"] = selftests
    return log


def _median(values):
    values = sorted(values)
    n = len(values)
    return None if n == 0 else values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def summarize(logs: list, macs: int | None = None) -> dict:
    """One model's boots: per phase the median of each boot's median, the
    spread, whether every check was exact, and the split of the first boot
    that has one."""
    out = {"boots": len(logs), "mspi": [l["mspi"] for l in logs],
           "profile": next((l["profile"] for l in logs if l["profile"]), None),
           "errors": [e for l in logs for e in l["errors"]],
           "done": all(l["done"] for l in logs), "phases": {},
           "micro": next((l["micro"] for l in logs if l.get("micro")), []),
           "shapes": next((l["shapes"] for l in logs if l.get("shapes")), {})}
    name = next((l["model"].get("name") for l in logs if l["model"]), None)
    macs = macs or next((l["model"].get("macs") for l in logs if l["model"]), None)
    out["name"], out["macs"] = name, macs
    for phase in ("flash", "psram"):
        boots = [l["phases"][phase] for l in logs
                 if phase in l["phases"] and not l["phases"][phase].get("skipped")]
        timed = [b["time"] for b in boots if "time" in b]
        if not boots:
            continue
        medians = [t["median_us"] for t in timed]
        row = {"boots": len(timed), "median_ms": None, "min_ms": None, "boot_medians_ms": [
            round(m / 1000, 3) for m in medians],
            "exact": all(b.get("check", {}).get("result") == "EXACT" for b in boots),
            "same_bits": all(t["same_bits"] == f"{t['runs']}/{t['runs']}" for t in timed),
            "weight_copies": [b.get("load", {}).get("weight_copies") for b in boots],
            "weight_resident": [b.get("load", {}).get("weight_resident") for b in boots]}
        if medians:
            row["median_ms"] = round(_median(medians) / 1000, 3)
            row["min_ms"] = round(min(t["min_us"] for t in timed) / 1000, 3)
            if macs:
                row["mmac_per_s"] = round(macs / (row["median_ms"] * 1000), 1)
        split = next((b["split"] for b in boots if "split" in b), None)
        if split:
            total = split["total_us"] or 1.0
            row["split_ms"] = {k: round(split[f"{k}_us"] / 1000, 3)
                               for k in ("delegate", "qdq", "portable", "other")}
            row["split_share"] = {k: round(split[f"{k}_us"] / total, 4)
                                  for k in ("delegate", "qdq", "portable", "other")}
            row["stepped_ms"] = round(split["total_us"] / 1000, 3)
            instr = next(b["instr"] for b in boots if b["instr"])
            row["top"] = sorted(instr, key=lambda r: -r[1])[:8]
        out["phases"][phase] = row
    return out


def format_summary(s: dict) -> str:
    lines = [f"--- {s['name']}: {s['boots']} boot(s); {s['profile'] or 'profile not printed'}",
             f"    [mspi] per boot: {', '.join(m or '?' for m in s['mspi'])}"]
    if s["macs"]:
        lines.append(f"    {s['macs']:,} MACs")
    for phase, row in s["phases"].items():
        verdict = "EXACT" if row["exact"] else "NOT EXACT"
        if row["median_ms"] is None:
            lines.append(f"    {phase}: no timing ({verdict})")
            continue
        rate = f", {row['mmac_per_s']} MMAC/s" if "mmac_per_s" in row else ""
        lines.append(f"    {phase}: median {row['median_ms']} ms (min {row['min_ms']}; boot medians "
                     f"{row['boot_medians_ms']}){rate}; outputs {verdict}, runs agree "
                     f"{'yes' if row['same_bits'] else 'NO'}; weight copies {row['weight_copies']}")
        if "split_ms" in row:
            share = row["split_share"]
            lines.append(f"      split (stepped {row['stepped_ms']} ms): delegates "
                         f"{row['split_ms']['delegate']} ms ({share['delegate']:.1%}), "
                         f"quantize/dequantize {row['split_ms']['qdq']} ms ({share['qdq']:.1%}), "
                         f"other portable ops {row['split_ms']['portable']} ms "
                         f"({share['portable']:.1%}), moves {row['split_ms']['other']} ms")
            for i, us, kind, label in row["top"][:5]:
                lines.append(f"        #{i} {us / 1000:.3f} ms {kind} {label}")
    for micro in s.get("micro", []):
        lines.append(f"    micro: {micro}")
    for phase in ("psram", "flash"):
        shapes = s.get("shapes", {}).get(phase)
        if not shapes:
            continue
        lines.append(f"    op shapes, {phase} phase, first boot (cycles an output at 240 MHz: "
                     f"us x 240 / (m x n)):")
        for t in shapes:
            group = f" x{t['members']}" if t["members"] > 1 else ""
            cycles = t["us"] * 240 / (t["m"] * t["n"]) if t["m"] * t["n"] else 0.0
            lines.append(f"      {t['op']} m={t['m']} k={t['k']} n={t['n']}{group}: "
                         f"{t['us']:.1f} us a call, {t['calls']} call(s); "
                         f"{cycles:.1f} cycles an output")
    if s["errors"]:
        lines.append(f"    ERRORS: {s['errors'][:4]}")
    if not s["done"]:
        lines.append("    a boot did not reach BENCH_DONE")
    return "\n".join(lines)


# --- the firmware ------------------------------------------------------------------------

def _board():
    from executorch.examples.espressif.esp32s3.scripts import board

    return board


def _flash_layout():
    from executorch.examples.espressif.esp32s3.scripts import flash_layout

    return flash_layout


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_state(path) -> str:
    """The commit of the tree holding `path`, "-dirty" when it has changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True).stdout.strip()
    return git("rev-parse", "--short=10", "HEAD") + ("-dirty" if git("status", "--porcelain",
                                                                    "--untracked-files=no") else "")


def load_model(model_dir: str) -> dict:
    """A model directory export_bench.py wrote: its record, its files' sizes,
    its operators."""
    record = json.load(open(os.path.join(model_dir, MODEL_FILE)))
    record["dir"] = os.path.abspath(model_dir)
    record["image_path"] = os.path.join(record["dir"], IMAGE_FILE)
    record["weights_path"] = os.path.join(record["dir"], WEIGHTS_FILE)
    record["image_bytes"] = os.path.getsize(record["image_path"])
    record["weights_bytes"] = os.path.getsize(record["weights_path"])
    with open(os.path.join(record["dir"], OPS_FILE)) as f:
        record["ops"] = sorted(l.strip() for l in f if l.strip() and not l.startswith("#"))
    return record


def model_layout(model: dict) -> dict:
    """Every bench model uses the lean table's default split: the image in
    the program partition, the weights after it."""
    fl = _flash_layout()
    layout = fl.layout_for(model["image_bytes"], model["weights_bytes"])
    if not layout["default"]:
        raise SystemExit(f"{model['name']}: image {model['image_bytes']:,} B and weights "
                         f"{model['weights_bytes']:,} B do not fit the default split; the bench "
                         "firmware's partition table is the default one")
    return layout


def build(model_dirs, build_dir, backend_o2=False, default_caches=False, sdkconfig_extra=None,
          dual_core=False, tier_bytes=0, conv_fast=True, qdq_fast=True,
          backend_timing=False, elementwise_fast=True):
    """One bench firmware for every named model: the union of their
    operators, selectively registered."""
    fl, board = _flash_layout(), _board()
    models = [load_model(d) for d in model_dirs]
    for model in models:
        model_layout(model)
    # A dual-core firmware turns min_latency on at boot (projections split,
    # every convolution on one core) and takes min_latency exports, compiled
    # for that posture (export_bench.py --min-latency). A single-core firmware
    # leaves each program's compile spec to decide, so a min_latency export
    # would split its projections there too.
    wrong = [m["name"] for m in models if bool(m.get("min_latency")) != dual_core]
    if wrong:
        raise SystemExit(
            f"{', '.join(wrong)}: exported {'without' if dual_core else 'with'} "
            f"--min-latency, for a {'single' if dual_core else 'dual'}-core firmware; "
            f"{'this build is dual-core: export them again with export_bench.py --min-latency' if dual_core else 'this build is single-core: export them again without --min-latency'}")
    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    ops = sorted({op for m in models for op in m["ops"]})
    required = os.path.join(build_dir, OPS_FILE)
    with open(required, "w") as f:
        f.write("# The union of the bench models' operators (run_bench_hardware.py build):\n")
        f.write("".join(f"#   {m['name']}\n" for m in models))
        f.write("".join(op + "\n" for op in ops))
    layout_defaults = fl.use_layout(build_dir, fl.layout_for(1, 1))  # the default split
    fl.pin_defaults(build_dir, STORY_DEFAULTS)
    for defaults in ([] if default_caches else [CACHE_DEFAULTS]) + ([sdkconfig_extra] if sdkconfig_extra else []):
        stale = fl.defaults_disagree(build_dir, defaults)
        if stale:
            print(f"  {os.path.basename(defaults)}: {', '.join(stale)} changed; "
                  f"regenerating {os.path.basename(build_dir)}/sdkconfig")
            os.remove(os.path.join(build_dir, "sdkconfig"))
            break
    board._run([sys.executable, os.path.join(HERE, "gen_kernels.py"), "--required", required,
                "--suffix", "_ple", *([] if qdq_fast else ["--no-qdq-fast"]),
                *(["--elementwise-fast"] if elementwise_fast else [])],
               "bench selective-kernel generation")
    defaults = ["sdkconfig.defaults", "sdkconfig.ple", "sdkconfig.hardware.defaults",
                layout_defaults, STORY_DEFAULTS]
    defaults += [] if default_caches else ["sdkconfig.cache.defaults"]
    defaults += [os.path.abspath(sdkconfig_extra)] if sdkconfig_extra else []
    flags = {"ESP32_BACKEND_O2": int(backend_o2), "ET_BENCH_DUAL_CORE": int(dual_core),
             "ET_BENCH_TIER_BYTES": int(tier_bytes), "ESP32_CONV_FAST": int(conv_fast),
             # The backend's micro-benchmarks at boot (BENCH_MICRO lines), with the path.
             "ET_BENCH_MICRO": int(conv_fast), "ESP32_BACKEND_TIMING": int(backend_timing)}
    t0 = time.time()
    board._run(board.idf(
        "-B", build_dir, "-C", PROJECT, f"-DSDKCONFIG={os.path.join(build_dir, 'sdkconfig')}",
        f"-DSDKCONFIG_DEFAULTS={';'.join(defaults)}",
        "-DET_PLE_ONLY=1", "-DET_BENCH_MODEL=1", "-DET_STORY=0", "-DET_QEMU_NO_RSQRT0=0",
        "-DET_PAPERCOLOR_DISPLAY=0", "-DESP32_OP_TIMING=0",
        *(f"-D{k}={v}" for k, v in flags.items()), "build"), "bench firmware build")
    app = os.path.join(build_dir, "et_runner.bin")
    size = os.path.getsize(app)
    if size > APP_LIMIT:
        raise RuntimeError(f"application exceeds its partition: {size:,} B")
    record = {"models": [m["name"] for m in models], "ops": ops, "flags": flags,
              "qdq_fast": qdq_fast, "elementwise_fast": elementwise_fast,
              "default_caches": default_caches, "sdkconfig_extra": sdkconfig_extra,
              "app_bytes": size, "app_free": APP_LIMIT - size, "app_sha256": _sha256(app),
              "seconds": round(time.time() - t0, 1),
              "tree": tree_state(PROJECT)}
    with open(os.path.join(build_dir, "bench_build.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"  bench firmware: app {size:,} B ({APP_LIMIT - size:,} B free), {len(ops)} operators "
          f"for {', '.join(record['models'])}; {build_dir}")
    return record


def flash_model(port, baud, build_dir, model, timeout=600):
    """The firmware, its table and bootloader, and one model's two partitions,
    in one esptool run."""
    from executorch.examples.espressif.esp32s3.scripts import runner_io

    board = _board()
    layout = model_layout(model)
    with open(os.path.join(build_dir, "flash_project_args")) as f:
        arguments = shlex.split(f.read())
    arguments += [hex(layout["pte_offset"]), model["image_path"],
                  hex(layout["ptd_offset"]), model["weights_path"]]
    command = board.esptool("--chip", "esp32s3", "--port", port, "--baud", str(baud),
                            "--before", "default_reset", "--after", "watchdog_reset",
                            "write_flash", *arguments)
    process = subprocess.Popen(command, cwd=build_dir, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        log = runner_io.capture_output(process, timeout, ())
        code = process.poll()
        if code is None:
            try:
                code = process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
    finally:
        runner_io.stop_process_group(process)
    if code is None:
        print(log[-4000:])
        raise RuntimeError(f"flashing took over {timeout} s")
    if code and not board._verified_post_reset_disconnect(log):
        print(log[-4000:])
        raise RuntimeError(f"flashing failed with status {code}")


def firmware_selftests(build_dir) -> tuple:
    """The conv path's self-tests the firmware in build_dir can report: each
    whose line its application holds. A firmware with every conv kernel holds
    all five; an A/B baseline built without some of them holds only the lines
    of the kernels it has, and is held to those."""
    with open(os.path.join(build_dir, "et_runner.bin"), "rb") as f:
        app = f.read()
    return tuple(name for name, line in SELFTEST_LINES if line in app)


def run_models(model_dirs, build_dir, serial, board_name="papercolor", boots=3, baud=2_000_000,
               timeout=1800, log_dir=None):
    board = _board()
    build_dir = os.path.abspath(build_dir)
    record = json.load(open(os.path.join(build_dir, "bench_build.json")))
    selftests = firmware_selftests(build_dir)
    log_dir = os.path.abspath(log_dir or os.path.join(build_dir, "logs"))
    os.makedirs(log_dir, exist_ok=True)
    results = []
    for model_dir in model_dirs:
        model = load_model(model_dir)
        missing = sorted(set(model["ops"]) - set(record["ops"]))
        if missing:
            raise SystemExit(f"{model['name']}: the firmware in {build_dir} does not register "
                             f"{missing}; build it with this model")
        port = board.find_port(board_name, serial)
        print(f"[{model['name']}] flashing the firmware, its image and its weights", flush=True)
        flash_model(port, baud, build_dir, model)
        logs = []
        for boot in range(boots):
            port = board.wait_for_port(board=board_name, serial=serial)
            try:
                text = board.run_device(port, timeout=timeout, build=build_dir, board=board_name,
                                        serial=serial)
            except RuntimeError as error:
                print(f"  boot {boot + 1}: capture failed ({error}); resetting and retrying once")
                subprocess.run(board.esptool("--port", port, "--after", "hard_reset", "chip_id"),
                               capture_output=True, timeout=60)
                port = board.wait_for_port(board=board_name, serial=serial)
                text = board.run_device(port, timeout=timeout, build=build_dir, board=board_name,
                                        serial=serial)
            path = os.path.join(log_dir, f"{model['name']}.boot{boot + 1}.log")
            with open(path, "w") as f:
                f.write(text)
            parsed = parse_log(text, selftests)
            logs.append(parsed)
            flash = parsed["phases"].get("flash", {}).get("time", {})
            print(f"  boot {boot + 1}: {parsed['mspi']}; flash median "
                  f"{flash.get('median_us', float('nan')) / 1000:.3f} ms; log {path}", flush=True)
        summary = summarize(logs, model.get("macs"))
        print(format_summary(summary), flush=True)
        results.append(summary)
    with open(os.path.join(log_dir, f"summary-{time.strftime('%Y%m%d-%H%M%S')}.json"), "w") as f:
        json.dump({"build": record, "results": results}, f, indent=1)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="one bench firmware for the named model directories")
    b.add_argument("models", nargs="+", help="directories export_bench.py wrote")
    b.add_argument("--build", required=True, help="the build directory (its own per profile)")
    b.add_argument("--backend-o2", action="store_true", help="the backend at -O2 (the rest -Os)")
    b.add_argument("--default-caches", action="store_true",
                   help="IDF's caches instead of sdkconfig.cache.defaults (the story's)")
    b.add_argument("--sdkconfig-extra", help="one more sdkconfig defaults file (e.g. the runtime's -O2)")
    b.add_argument("--dual-core", action="store_true",
                   help="projection columns split across both cores (no convolution splits)")
    b.add_argument("--tier-bytes", type=int, default=0,
                   help="the psram measurement's weight budget (default: every weight PSRAM holds)")
    b.add_argument("--backend-timing", action="store_true",
                   help="time every op shape (ESP32_BACKEND_TIMING): the boot ends with "
                        "esp32::addmm_i8 m=.. k=.. n=.. lines, us a call, for cycles per output")
    b.add_argument("--no-conv-fast", action="store_true",
                   help="leave the static int8 conv path (runtime/esp32_conv.h) out: every conv "
                        "takes the general loops, for an A/B against the default build")
    b.add_argument("--no-qdq-fast", action="store_true",
                   help="register ExecuTorch's portable quantize, dequantize and avg_pool2d "
                        "kernels instead of the op library's exact fast paths (ops/qdq_pool.h), "
                        "for an A/B against the default build")
    b.add_argument("--no-elementwise-fast", action="store_true",
                   help="register the op library's mul.out shim and the portable "
                        "constant_pad_nd without the element-wise loops (ops/elementwise.h), "
                        "for an A/B against the default build. quant_lut's two table loops "
                        "stay on: no device build turns them off (the host runners' "
                        "--no-ew-fast does)")
    r = sub.add_parser("run", help="flash and time models on the board")
    r.add_argument("models", nargs="+")
    r.add_argument("--build", required=True)
    r.add_argument("--serial", required=True, help="the board's USB serial number")
    r.add_argument("--board", default="papercolor")
    r.add_argument("--boots", type=int, default=3)
    r.add_argument("--baud", type=int, default=2_000_000)
    r.add_argument("--timeout", type=int, default=1800, help="seconds per boot")
    r.add_argument("--log-dir")
    s = sub.add_parser("summary", help="summarize saved boot logs, grouped by model")
    s.add_argument("logs", nargs="+")
    s.add_argument("--json", action="store_true")
    i = sub.add_parser("show", help="print a bench image's header and labels")
    i.add_argument("image")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.models, args.build, args.backend_o2, args.default_caches,
              args.sdkconfig_extra, args.dual_core, args.tier_bytes, not args.no_conv_fast,
              not args.no_qdq_fast, args.backend_timing, not args.no_elementwise_fast)
    elif args.command == "run":
        results = run_models(args.models, args.build, args.serial, args.board, args.boots,
                             args.baud, args.timeout, args.log_dir)
        bad = [r["name"] for r in results
               if r["errors"] or not r["done"] or not all(p["exact"] for p in r["phases"].values())]
        if bad:
            print(f"FAIL: {bad} not exact or not done")
            return 1
    elif args.command == "summary":
        by_model = {}
        for path in args.logs:
            parsed = parse_log(open(path, errors="replace").read())
            by_model.setdefault(parsed["model"].get("name", path), []).append(parsed)
        summaries = [summarize(logs) for logs in by_model.values()]
        if args.json:
            print(json.dumps(summaries, indent=1))
        else:
            print("\n".join(format_summary(s) for s in summaries))
    else:
        h = read_image(open(args.image, "rb").read())
        for key, value in h.items():
            if key not in ("labels", "program"):
                print(f"{key}: {value}")
        for index, label in enumerate(h["labels"]):
            print(f"  {index:4d} {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
