#!/usr/bin/env python3

# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Put the story Glimmer on the board, check it against the host, and talk to it.

    run_story_hardware.py --checkpoint /path/to/glimmer-ple-story.pt --tokenizer /path/to/tokenizer.json \\
        --serial <the board's USB serial number>
    run_story_hardware.py --reuse-artifact --serial <...> --request "Once upon a time, a little fox"

Steps: export (export_story_flash.py) into the example's main/, derive the
selective kernels, build with -DET_STORY=1, flash (--reuse-weights skips the
15 MB PTD; NVS is erased first unless --keep-saved-story), capture the boot log until === DONE ===, parse the STORY_GOLDEN
line (the device's greedy continuation of the exporter's prompt against the
host modules' tokens and checksums), then send each --request over the
console and print the generated story with its timing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from executorch.examples.espressif.esp32s3.scripts import board as hardware  # noqa: E402
from executorch.examples.espressif.esp32s3.scripts import board as bench  # noqa: E402
from executorch.examples.espressif.esp32s3.scripts import flash_layout  # noqa: E402

PROJECT = os.path.dirname(HERE)  # examples/espressif/esp32s3
MAIN = os.path.join(PROJECT, "main")
BUILD = os.path.join(PROJECT, "build_story_hw")
SDKCONFIG_EXTRA = None   # --sdkconfig-extra: an A/B config experiment, built in its own directory
DEFAULT_CACHES = False   # --default-caches: IDF's 16/32 KB caches instead of sdkconfig.cache.defaults
PLACEMENT = {}           # placement_defines()'s memory settings from the command line
REQUIRED = os.path.join(MAIN, "ple_required_ops.txt")
STORY_DEFAULTS = os.path.join(PROJECT, "sdkconfig.story.defaults")  # cold .bss in PSRAM, every story build
CACHE_DEFAULTS = os.path.join(PROJECT, "sdkconfig.cache.defaults")  # unless --default-caches
ARTIFACT_FILES = ("ple.pte", "ple_weights.ptd", "ple_required_ops.txt", "story_inputs.h", "story_vocab.h",
                  "story_bpe_tables.h", "story_artifact.json")
GOLDEN = re.compile(r"^STORY_GOLDEN (?P<fields>.*)$", re.M)
GOLDEN_KEYS = {"tokens_match": "matched", "first_diff": "first", "max_rel": "rel", "median_rel": "median",
               "prefill_blocks": "blocks", "prefill_us": "pus", "decode_steps": "steps", "decode_us": "dus"}
# The cold start, beside decode's clock (story_runner.inc): Program::load with the data map, then
# each method's first load_method and the ranked placement after it, microseconds.
COLD = re.compile(r"^STORY_COLD (?P<fields>.*)$", re.M)
RESULT = re.compile(r"STORY_RESULT prompt_tokens=(?P<prompt>\d+) generated=(?P<generated>\d+) stopped=(?P<stopped>[\w-]+) "
                    r"(?:pages=(?P<pages>\d+) )?prefill_blocks=(?P<blocks>\d+) "
                    r"prefill_us=(?P<pus>\d+) decode_steps=(?P<steps>\d+) decode_us=(?P<dus>\d+)")
PANEL = re.compile(r"PAPERCOLOR_STATUS seq=\d+ result=(?P<result>\w+) mode=(?P<mode>\w+) .*body_px=(?P<need>\d+)/(?P<avail>\d+)")
TEXT = re.compile(r'STORY_TEXT "(?P<text>.*)"$')
# The firmware's lines about where its memory went (story_runner.inc), echoed from the boot log and
# from request replies: a prompt of 33 tokens or more runs prefill, whose arena and temp peaks print then.
MEMORY_LINES = ("mem-id-", "method arena", "temp:", "[mem]", "[tier]")


def firmware_default(name):
    """story_runner.inc's own default for one of its settings (`#ifndef name`
    then `#define name <integer>`)."""
    with open(os.path.join(MAIN, "story_runner.inc")) as source:
        found = re.search(rf"^#ifndef {name}\n#define {name} (\d+)$", source.read(), re.M)
    if found is None:
        raise RuntimeError(f"story_runner.inc gives {name} no integer default")
    return int(found.group(1))


# Ranked weight placement (the backend's weight_tier_rank) for the story by
# default is one line of the firmware (story_runner.inc's
# ET_STORY_TIER_RANK): --tier-rank follows it, so builds made here and with
# idf.py alone agree.
TIER_RANK_DEFAULT = bool(firmware_default("ET_STORY_TIER_RANK"))


def unescape(text):
    return text.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8", errors="replace")


def export(checkpoint, tokenizer, context, chunk, prompt=None, golden="default"):
    bench._run([sys.executable, os.path.join(HERE, "export_story_flash.py"), "--checkpoint", os.path.abspath(checkpoint),
                   "--tokenizer", os.path.abspath(tokenizer), "--context", str(context), "--prefill-chunk", str(chunk),
                   "--emit-dir", MAIN, "--golden", golden] + (["--prompt", prompt] if prompt else []), "story export")


def stage_mb_arg(text):
    """--stage-mb: megabytes, or "auto"."""
    if text == "auto":
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--stage-mb takes megabytes or auto, not {text!r}") from None


def placement_defines(stage_mb=0.0, temp_mb=2.0, tier_rank=TIER_RANK_DEFAULT, stage_reserve_mb=0.5,
                      internal_floor_kb=40, decode_act_internal=True, hot_constants=True,
                      arena_kb=128):
    """The story firmware's -D settings for its weights and memory (main/CMakeLists.txt).

    Every one is given on every build: they are CMake cache variables, which
    would otherwise keep a previous build's value. --stage-mb auto is the
    backend's auto weight budget (-1), which keeps --stage-reserve-mb of PSRAM
    free for what is allocated after load, with ranked placement. The rest is
    the placement in story_runner.inc ("Where the story's memory goes"): data
    in the image, not code, so images that differ only in them run the same
    code at the same addresses."""
    auto = stage_mb == "auto"
    return [f"-DET_STORY_STAGE_BYTES={-1 if auto else int(stage_mb * 1048576)}",
            f"-DET_STORY_TIER_RESERVE={int(stage_reserve_mb * 1048576) if auto else 0}",
            f"-DET_STORY_TEMP_BYTES={int(temp_mb * 1048576)}",
            f"-DET_STORY_TIER_RANK={int(tier_rank or auto)}",
            f"-DET_STORY_INTERNAL_FLOOR={int(internal_floor_kb * 1024)}",
            f"-DET_STORY_DECODE_ACT_INTERNAL={int(decode_act_internal)}",
            f"-DET_STORY_HOT_CONSTANTS={int(hot_constants)}",
            f"-DET_STORY_ARENA_BYTES={int(arena_kb * 1024)}"]


def build(manifest, display, backend_timing=False, dual_core=False, stage_mb=0.0, temp_mb=2.0,
          backend_o2=False, op_timing=False, tier_rank=TIER_RANK_DEFAULT):
    # The story program (170 KB) does not fit the default split: this build
    # gets a partition table sized to its own program and weights.
    layout = flash_layout.layout_for(manifest["pte_bytes"], manifest["ptd_bytes"])
    print("  flash: " + flash_layout.describe(layout, manifest["pte_bytes"], manifest["ptd_bytes"]))
    layout_defaults = flash_layout.use_layout(BUILD, layout)
    flash_layout.pin_defaults(BUILD, STORY_DEFAULTS)  # a present sdkconfig outranks the defaults
    # Cache sizes are Kconfig choices: pinning one value would leave the old
    # choice set beside it. When a build directory's sdkconfig disagrees with
    # the cache defaults (or an --sdkconfig-extra file), regenerate it from the
    # defaults files, which is where all of its settings come from.
    for defaults in ([] if DEFAULT_CACHES else [CACHE_DEFAULTS]) + ([SDKCONFIG_EXTRA] if SDKCONFIG_EXTRA else []):
        stale = flash_layout.defaults_disagree(BUILD, defaults)
        if stale:
            print(f"  {os.path.basename(defaults)}: {', '.join(stale)} changed; regenerating {os.path.basename(BUILD)}/sdkconfig")
            os.remove(os.path.join(BUILD, "sdkconfig"))
            break
    bench._run([sys.executable, os.path.join(PROJECT, "scripts", "gen_kernels.py"), "--required", REQUIRED, "--suffix", "_ple"],
                  "story selective-kernel generation")
    bench._run(bench.idf(
        "-B", BUILD, "-C", PROJECT, f"-DSDKCONFIG={os.path.join(BUILD, 'sdkconfig')}",
        f"-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.ple;sdkconfig.hardware.defaults;{layout_defaults};{STORY_DEFAULTS}"
        + ("" if DEFAULT_CACHES else ";sdkconfig.cache.defaults")
        + (f";{SDKCONFIG_EXTRA}" if SDKCONFIG_EXTRA else ""),
        "-DET_PLE_ONLY=1", "-DET_QEMU_NO_RSQRT0=0", "-DET_STORY=1", f"-DET_STORY_DUAL_CORE={int(dual_core)}",
        *placement_defines(stage_mb, temp_mb, tier_rank, **PLACEMENT),
        f"-DET_PAPERCOLOR_DISPLAY={int(display)}", f"-DESP32_BACKEND_TIMING={int(backend_timing)}",
        f"-DESP32_BACKEND_O2={int(backend_o2)}", f"-DESP32_OP_TIMING={int(op_timing)}",
        # The story has no convolution: it leaves the static int8 conv path
        # (runtime/esp32_conv.h) out, so its image holds none of that code.
        "-DESP32_CONV_FAST=0", "build"),
        "story N16R8 hardware build")
    size = os.path.getsize(os.path.join(BUILD, "et_runner.bin"))
    if size > 0x160000:  # the lean table's factory partition (partitions_ple.csv)
        raise RuntimeError(f"application exceeds partition: {size:,} B")
    return size


def reset_board(port):
    """Hard-reset the chip through esptool (RTS on the native USB port)."""
    subprocess.run(hardware.esptool("--port", port, "--after", "hard_reset", "chip_id"),
                   capture_output=True, timeout=60)


def capture_boot(port, args):
    """The boot log from the host's GO onward. The story firmware starts on
    its own if no GO arrives within 8 s of reset (it is a product, and must run
    without a laptop); a capture that finds it already running resets it and
    catches the next boot itself. If the capture still fails, reset through
    esptool and try once more."""
    try:
        return hardware.run_device(port, timeout=args.timeout, build=BUILD, board=args.board,
                                   serial=args.serial)
    except RuntimeError as error:
        print(f"  capture failed ({error}); resetting the board through esptool and retrying once")
        reset_board(port)
        port = hardware.wait_for_port(board=args.board, serial=args.serial)
        return hardware.run_device(port, timeout=args.timeout, build=BUILD, board=args.board,
                                   serial=args.serial)


def parse_golden(log):
    match = GOLDEN.search(log)
    if not match:
        raise RuntimeError("no STORY_GOLDEN line in the device log")
    row = {}
    for field in match.group("fields").split():
        key, _, value = field.partition("=")
        key = GOLDEN_KEYS.get(key, key)
        if key == "matched":
            value, _, total = value.partition("/")
            row["total"] = int(total)
        # digest: the FNV-1a of every step's logits, hex (a firmware without it prints none)
        row[key] = (value if key in ("result", "values", "digest") else
                    float(value) if key in ("rel", "median", "early_max", "tol_early") else int(value))
    device = re.search(r'device greedy: "(.*)"', log)
    golden = re.search(r'golden greedy: "(.*)"', log)
    row["device_text"] = unescape(device.group(1)) if device else ""
    row["golden_text"] = unescape(golden.group(1)) if golden else ""
    return row


def parse_cold(log):
    """STORY_COLD's fields (story_runner.inc), or None for a firmware that prints no such line."""
    match = COLD.search(log)
    if not match:
        return None
    return {key: float(value) if key == "decode_ms_per_token" else int(value)
            for key, _, value in (field.partition("=") for field in match.group("fields").split())}


def erase_saved_story(port, baud):
    """Erase NVS, where the firmware keeps the last story it wrote. At boot it shows that
    story while the model loads and the golden runs: a different story every run, which
    can move the timed decode by up to 2%."""
    command = hardware.esptool("--chip", "esp32s3", "--port", port, "--baud", str(baud),
                               "--before", "default_reset", "--after", "no_reset", "erase_region",
                               hex(flash_layout.NVS_OFFSET), hex(flash_layout.NVS_SIZE))
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        print(result.stdout[-2000:] + result.stderr[-2000:])
        raise RuntimeError("erasing NVS (the saved story) failed")
    print("  NVS erased: no saved story at boot")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="the trained story Glimmer (README.md: the checkpoint contract); "
                        "required unless --reuse-artifact or --no-flash")
    parser.add_argument("--tokenizer", help="its tokenizer JSON; required with --checkpoint")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--golden", choices=("default", "long"), default="default",
                        help="which golden the image checks at boot (export_story_flash.py GOLDENS): default, the "
                        "19-token prompt and 24 steps every story image carries; long, the release check's second "
                        "golden: a 75-token prompt (two prefill chunks), then 24 greedy steps from position 74, past "
                        "the local layers' 64-slot ring, gated on tokens and on its first prefill block's value")
    parser.add_argument("--golden-prompt", help="the golden check's prompt (export_story_flash.py --prompt); "
                        "65 tokens or more give two prefill chunks, each checked against the golden")
    parser.add_argument("--reuse-artifact", action="store_true", help="skip the export; use what is in main/")
    parser.add_argument("--reuse-weights", action="store_true", help="flash everything but the PTD")
    parser.add_argument("--keep-saved-story", action="store_true",
                        help="keep the story the last run saved in NVS; by default NVS is erased before flashing, "
                        "so the panel shows nothing at boot and the timed golden runs with the display idle")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-flash", action="store_true", help="the board already runs this build: skip to the requests")
    parser.add_argument("--board", default="papercolor", choices=tuple(hardware.BOARD_PROFILES))
    parser.add_argument("--serial", help="the board's USB serial number (`python -m serial.tools.list_ports -v`): "
                        "finds the board, and finds it again after a flash or a reset")
    parser.add_argument("--port", help="the board's serial port, for --request alone (flashing needs --serial)")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend-timing", action="store_true", help="build with per-op timing and print the report")
    parser.add_argument("--op-timing", action="store_true", help="time every decode instruction (ExecuTorch EventTracer); instrumented, not a speed figure")
    parser.add_argument("--default-caches", action="store_true",
                        help="build with IDF's default 32 KB DCache / 32 B lines instead of "
                        "sdkconfig.cache.defaults (64 KB DCache with 64 B lines and a 32 KB ICache: 141 -> 97 ms/token); "
                        "builds in build_story_hw_defcache")
    parser.add_argument("--sdkconfig-extra", help="an extra sdkconfig defaults file for an A/B experiment "
                        "(e.g. cache sizes); builds in build_story_hw_<file stem> so the base build is untouched")
    parser.add_argument("--backend-o2", action="store_true", help="compile the backend at -O2 (the rest stays -Os)")
    parser.add_argument("--dual-core", action="store_true", help="split every projection across both cores (backend min_latency)")
    parser.add_argument("--stage-mb", type=stage_mb_arg, default=0.0,
                        help="MB of weights to copy into PSRAM at load (0 = read from flash), or auto: every weight "
                             "that fits while --stage-reserve-mb of PSRAM stays free, placed ranked")
    parser.add_argument("--stage-reserve-mb", type=float, default=0.5,
                        help="with --stage-mb auto, the PSRAM the copies leave free for what is allocated after load")
    parser.add_argument("--temp-mb", type=float, default=2.0, help="PSRAM temp arena for the methods")
    parser.add_argument("--tier-rank", action=argparse.BooleanOptionalAction, default=TIER_RANK_DEFAULT,
                        help="ranked weight placement: the --stage-mb copies are made after each load, the "
                             "weights worth most first, instead of inside it in load order (default: "
                             "story_runner.inc's ET_STORY_TIER_RANK)")
    parser.add_argument("--internal-floor-kb", type=int, default=40,
                        help="internal SRAM kept free after load: a placement that would leave less goes to PSRAM "
                             "or stays on flash (the [mem] lines print it after load and after the first story)")
    parser.add_argument("--hot-constants", action=argparse.BooleanOptionalAction, default=True,
                        help="serve the small constants outside the delegate (the RMSNorm weights) from internal "
                             "SRAM, within the floor, instead of reading them from flash every token")
    parser.add_argument("--decode-act-internal", action=argparse.BooleanOptionalAction, default=True,
                        help="give decode's activations their own buffer in internal SRAM, within the floor, "
                             "before the small constants, instead of sharing prefill's in PSRAM")
    parser.add_argument("--arena-kb", type=int, default=128, help="the method arena both methods use, in PSRAM")
    parser.add_argument("--request", action="append", default=[], help="a prompt; \\n escapes allowed; repeatable")
    parser.add_argument("--new", type=int, default=0, help="stories from random premises (key B), after any --request")
    parser.add_argument("--turn", action="store_true", help="after each story, page down to the end and back (keys C, A)")
    parser.add_argument("--font", type=int, default=0, help="story face in px (14, 16, 18, 20) before the stories")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=None, help="token guard; default: the firmware's (230)")
    parser.add_argument("--log", help="the boot log (default: story-hardware.log in the build directory)")
    parser.add_argument("--quit", action="store_true")
    args = parser.parse_args(argv)
    if not (args.reuse_artifact or args.no_flash) and not (args.checkpoint and args.tokenizer):
        parser.error("--checkpoint and --tokenizer are required to export (or pass --reuse-artifact)")
    if args.golden_prompt and (args.reuse_artifact or args.no_flash):
        parser.error("--golden-prompt is an export setting: it cannot apply to a reused artifact")
    if args.golden != "default" and (args.reuse_artifact or args.no_flash):
        parser.error("--golden is an export setting: it cannot apply to a reused artifact")
    global BUILD, SDKCONFIG_EXTRA, DEFAULT_CACHES, PLACEMENT
    PLACEMENT = dict(stage_reserve_mb=args.stage_reserve_mb, internal_floor_kb=args.internal_floor_kb,
                     decode_act_internal=args.decode_act_internal, hot_constants=args.hot_constants,
                     arena_kb=args.arena_kb)
    if args.default_caches:
        DEFAULT_CACHES = True
        BUILD = os.path.join(PROJECT, "build_story_hw_defcache")
    if args.sdkconfig_extra:
        SDKCONFIG_EXTRA = os.path.abspath(args.sdkconfig_extra)
        # sdkconfig.<stem>.defaults and <stem>.defaults both build in build_story_hw_<stem>.
        stem = os.path.basename(SDKCONFIG_EXTRA).removeprefix("sdkconfig.").split(".")[0]
        BUILD = os.path.join(PROJECT, "build_story_hw_" + stem)
    args.log = args.log or os.path.join(BUILD, "story-hardware.log")

    if not args.no_flash:
        if args.reuse_artifact:
            missing = [f for f in ARTIFACT_FILES if not os.path.exists(os.path.join(MAIN, f))]
            if missing:
                parser.error(f"--reuse-artifact but {missing} missing in {MAIN}")
            print("[1/4] reusing the exported artifact")
        else:
            print("[1/4] exporting the story Glimmer")
            export(args.checkpoint, args.tokenizer, args.context, args.prefill_chunk, args.golden_prompt,
                   args.golden)
        manifest = json.load(open(os.path.join(MAIN, "story_artifact.json")))
        # main/ is shared by every runner (the parity harness exports its
        # programs there too), so the files must be proven to be THIS model
        # before flashing.
        import hashlib
        for name, key in (("ple.pte", "pte_sha256"), ("ple_weights.ptd", "ptd_sha256")):
            path = os.path.join(MAIN, name)
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != manifest[key]:
                print(f"FAIL: {path} is not the exported story model (sha256 {digest.hexdigest()[:16]}..., "
                      f"manifest {manifest[key][:16]}...); re-export or restore it")
                return 1
        print(f"  PTE {manifest['pte_bytes']:,} B, PTD {manifest['ptd_bytes']:,} B; golden {manifest['golden']['text']!r}")
        print("[2/4] building the story firmware")
        size = build(manifest, args.display, args.backend_timing, args.dual_core, args.stage_mb, args.temp_mb,
                     args.backend_o2, args.op_timing, args.tier_rank)
        print(f"  app {size:,} B; {0x160000 - size:,} B free")
        if args.build_only:
            return 0
        port = args.port or hardware.find_port(args.board, args.serial)
        print(f"[3/4] flashing {args.board} at {args.baud:,} baud" + (" (weights kept)" if args.reuse_weights else ""))
        if not args.keep_saved_story:
            erase_saved_story(port, args.baud)
        hardware.flash_device(port, args.baud, build=BUILD,
                              excluded_suffixes=("ple_weights.ptd",) if args.reuse_weights else ())
        port = hardware.wait_for_port(board=args.board, serial=args.serial)
        log = capture_boot(port, args)
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "w") as out:
            out.write(log)
        print("[4/4] golden check")
        golden = parse_golden(log)
        for line in log.splitlines():
            if any(tag in line for tag in MEMORY_LINES + ("prefill block", "decode step", "staged", "PSRAM before")):
                print("  " + line.strip())
        if args.backend_timing:
            print("  --- backend timing report ---")
            report = log.split("STORY_GOLDEN", 1)[-1].split("=== DONE ===", 1)[0]
            print("\n".join("  " + l.rstrip() for l in report.splitlines()[1:] if l.strip()))
        print(f"  device: {golden['device_text']!r}\n  host:   {golden['golden_text']!r}")
        total_ms = (golden["pus"] + golden["dus"]) / 1000
        for line in log.splitlines():
            if line.strip().startswith(("choice ", "step drift")):
                print("  " + line.strip())
        print(f"  {golden['result']}: {golden['matched']}/{golden['total']} choices agree, {golden.get('near_ties', 0)} near-tie, "
              f"{golden.get('hard', 0)} wrong (first difference at {golden['first']}); "
              f"first {golden['early_steps']} steps drift at most {golden['early_max']:.1e} (limit {golden['tol_early']:.0e}): "
              f"values {golden['values']}; later drift from step {golden['onset']} (rounding flips, reported), "
              f"median {golden['median']:.1e}, max {golden['rel']:.2e}; prefill {golden['blocks']} blocks "
              f"{golden['pus'] / 1000:.0f} ms, decode {golden['steps']} steps {golden['dus'] / 1000:.0f} ms "
              f"({golden['dus'] / 1000 / max(1, golden['steps']):.0f} ms/token)")
        cold = parse_cold(log)
        if "digest" in golden or cold:
            print(f"  logits digest {golden.get('digest', '(none)')}" + (
                f"; cold start: program {cold['program_load_us'] / 1000:.0f} ms, prefill load "
                f"{cold['prefill_load_us'] / 1000:.0f} ms + placement {cold['prefill_place_us'] / 1000:.0f} ms, "
                f"decode load {cold['decode_load_us'] / 1000:.0f} ms + placement "
                f"{cold['decode_place_us'] / 1000:.0f} ms" if cold else ""))
        if golden["result"] != "MATCH":
            print("FAIL: the device does not reproduce the exporter's reference (log: " + args.log + ")")
            return 1
    if not args.request and not args.new and not args.font:
        return 0

    from executorch.examples.espressif.esp32s3.scripts import board as board_mod
    port = args.port or hardware.find_port(args.board, args.serial)
    board = board_mod.Board(port, None, False)
    print(f"board on {port}; waiting for the request prompt")
    board.ready(args.timeout)
    settings = [f"T {args.temperature}", f"K {args.top_k}", f"S {args.seed}"]
    if args.max_new is not None:
        settings.append(f"N {args.max_new}")
    for setting in settings:
        board.send(setting + "\n")
        board.wait_for("=== REQUEST DONE ===", 30)
    def panel(seen):
        """Refreshes the panel made during one command: mode and fit."""
        frames = [PANEL.search(line) for line in seen]
        return [f"{m['mode']} {m['need']}/{m['avail']}px" for m in frames if m]

    if args.font:
        board.send(f"F {args.font}\n")
        seen = board.wait_for("=== REQUEST DONE ===", 120)
        print("  " + next((l.strip() for l in seen if "STORY_SET font_px" in l), "font not set"))
    commands = [("R " + t.replace("\\", "\\\\").replace("\n", "\\n"), t) for t in args.request]
    commands += [("G", "(next story)")] * args.new
    for wire, label in commands:
        board.send(wire + "\n")
        seen = board.wait_for("=== REQUEST DONE ===", args.timeout)
        for line in seen:
            if any(tag in line for tag in MEMORY_LINES):
                print("  " + line.strip())
        story, result = "", None
        premise = next((l.split('"')[1] for l in seen if l.startswith("STORY_SHOWN")),
                       next((l.split('"')[1] for l in seen if l.startswith("STORY_PREMISE")), None))
        for line in seen:
            m = TEXT.search(line)
            if m:
                story = unescape(m.group("text"))
            m = RESULT.search(line)
            if m:
                result = {k: int(v) if v and v.isdigit() else v for k, v in m.groupdict().items()}
        if result is None:
            error = next((l for l in seen if "STORY_RESULT error" in l), "no result line")
            print(f"FAIL {label[:60]!r}: {error.strip()}")
            continue
        per_token = result["dus"] / 1000 / max(1, result["steps"])
        print(f"--- {(premise or label).splitlines()[0][:90]}\n{story.strip()}\n"
              f"    {result['prompt']} prompt tokens, {result['generated']} generated (stopped: {result['stopped']}), "
              f"{result.get('pages') or 1} page(s); decode {result['dus'] / 1000:.0f} ms = {per_token:.0f} ms/token; "
              f"panel: {', '.join(panel(seen))}")
        pages = int(result.get("pages") or 1)
        if args.turn and pages > 1:
            turns = []
            for key in ["D"] * (pages - 1) + ["U"] * (pages - 1):
                board.send(key + "\n")
                seen = board.wait_for("=== REQUEST DONE ===", 120)
                page = next((l.split()[1] for l in seen if l.startswith("STORY_PAGE")), "?")
                turns.append(f"{key}->{page} [{', '.join(panel(seen))}]")
            print("    page turns: " + "; ".join(turns))
    if args.quit:
        board.send("Q\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
