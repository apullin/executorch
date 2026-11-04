# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The same parity suite, on real silicon. Three things QEMU structurally cannot do.

    python backends/esp32s3/test/parity/run_hardware.py --board papercolor --serial SERIAL
        [--resume] [--port PORT] [--log FILE] [--replay FILE] [--cache-defaults]
    python backends/esp32s3/test/parity/run_hardware.py --rates LOG [--board B] [--caches C]

WHY A SEPARATE RUNNER, rather than a flag on run_parity.py. Three differences
are not cosmetic:

1. **`rmsnorm_s3.S` does not execute under QEMU.** QEMU faults with
   IllegalInstruction on `rsqrt0.s`, which is the single instruction that
   assembly file exists to reach, so `run_parity.py` builds with
   `-DET_QEMU_NO_RSQRT0=1` and the kernel silently takes its scalar reference
   path. It assembles, it links, its parity
   rows pass -- and none of that is evidence about the assembly. This runner
   builds WITHOUT that define, so the hand-written kernel runs, against
   references that already know the right answer.

2. **Memory bandwidth.** `chips.py` records measured SRAM, PSRAM and flash read
   bandwidth; the flash figure is what turns streamed weight bytes into time.
   Emulated flash has no relationship to a real SPI part, so this is the only
   place the number can come from. `-DET_BENCH_MEM=1` compiles the measurement
   in, and its SRAM/PSRAM rows are positive controls against reference figures
   measured independently on this board (240.0 / 60.7 MB/s). The same build
   times the weight tiers' kernels (main/bench_tiers.cpp: int4 and int8 decode
   rows from flash, PSRAM and SRAM, on one core and two) and the int4 decode
   row again with every weight copied into PSRAM; this prints both, and the
   kernel rates as chips.py MEASURED entries. `--cache-defaults` builds with
   the story's caches (sdkconfig.cache.defaults), in a directory of its own.
   `--rates LOG` reads those rows from any saved boot log -- a story built
   with sdkconfig.bench.defaults prints them too -- with no board and no
   build.

3. **Division and square root.** Float division (ROM `__divsf3`) and `sqrtf`
   (libgcc's `__ieee754_sqrtf`) are DIV0.S / SQRT0.S sequences ending in
   DIVN.S, and QEMU runs its own model of those instructions. The parity
   firmware's `divsqrt check` (divsqrt_golden.py: the hardest-to-round
   quotients and roots, subnormal ties, the edges) passes under QEMU; only
   here does it say the silicon rounds correctly, which every exact row
   between delegates that divides or takes a root assumes.

Everything else -- the models, the references, the comparison -- is
`run_parity.main` itself, called with this file's build and run steps: the
same exports, every partition image, the same exact-bit and tolerance rows and
dispatch checks. Nothing is copied. Two suites that could disagree about what the right
answer is would be worse than one suite that only runs under emulation.

WHAT A FAILURE HERE MEANS. A row that passes under QEMU and fails here is not a
flaky test. It is a real divergence between emulator and silicon, and the two
candidates are the assembly path this build enables and anything
timing-dependent. Both are worth a stop.
"""

import argparse
import os
import re
import subprocess
import sys

from executorch.examples.espressif.esp32s3.scripts import board as hardware
from executorch.backends.esp32s3.test.parity import run_parity as P

BUILD = os.path.join(P.PROJ, "build_hardware")
# --cache-defaults: the story's caches. Its own directory, because a present
# sdkconfig outranks the defaults files.
BUILD_CACHE = os.path.join(P.PROJ, "build_hardware_cache")


def build_hw(build=BUILD, cache_defaults=False):
    """Build for silicon: no QEMU workaround, with the bandwidth benchmark.

    `ET_QEMU_NO_RSQRT0=0` is passed EXPLICITLY, and that is the important line in
    this file. It is a CMake CACHE variable: `run_parity.py` sets it to 1, and a
    cache variable persists in a build directory until something overwrites it.
    So a hardware build that merely omitted the flag would inherit 1 from the
    last QEMU build, take the scalar path, pass every row, and prove nothing --
    while the whole reason this runner exists is to execute `rsqrt0.s`.
    Silently inheriting it is the exact failure the CMakeLists comment warns
    about.

    The device prints which path it took at boot, and `flash_and_run` checks it,
    because a define that is supposed to be off is worth verifying rather than
    assuming.

    `cache_defaults` adds the story's caches (sdkconfig.cache.defaults), so the
    rates are measured as the story reads its weights.
    """
    defaults = "sdkconfig.defaults;sdkconfig.hardware.defaults" + (
        ";sdkconfig.cache.defaults" if cache_defaults else "")
    r = subprocess.run(
        P.idf("-C", P.PROJ, "-B", build,
              f"-DSDKCONFIG={os.path.join(build, 'sdkconfig')}",
              f"-DSDKCONFIG_DEFAULTS={defaults}",
              "-DET_QEMU_NO_RSQRT0=0", "-DET_BENCH_MEM=1",
              "-DET_HARDWARE_RUNNER_GATE=1", "build"),
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise SystemExit("hardware build FAILED")
    size = re.search(r"et_runner.bin binary size (0x[0-9a-f]+)", r.stdout)
    if size:
        print(f"  built, image {int(size.group(1), 16):,} B "
              f"(rsqrt0.s ENABLED, bandwidth bench in)")
    return r.stdout


def flash_and_run(port=None, baud=2_000_000, timeout=300, board="nano", serial=None,
                  build=BUILD):
    """Flash, then use the deterministic READY/GO raw capture path."""
    port = hardware.validate_port(port, board, serial)
    # The whole parity table (every partition image, about 14 MB) outlasts
    # flash_device's 300 s default at the story's 921,600 baud.
    hardware.flash_device(port, baud, build=build, timeout=1200)
    print("  flashed")
    port = hardware.wait_for_port(board=board, serial=serial)
    log = hardware.run_device(port, timeout=timeout, build=build,
                              project=P.PROJ, board=board, serial=serial)
    print(log, end="" if log.endswith("\n") else "\n")
    return log


def _parse_bandwidth(log):
    rows = re.findall(
        r"(SRAM|PSRAM|FLASH)\s+seq read\s*:\s*([\d.]+)\s+MB/s", log)
    got = {kind: float(value) for kind, value in rows}
    method = re.search(
        r"SRAM\s+seq read \(internal, best of 16\):\s*([\d.]+)\s+MB/s", log)
    if method:
        got["SRAM_CONTROL"] = float(method.group(1))
    return got


def report_bandwidth(log):
    """Echo measurements; accept flash only when both controls reproduce."""
    got = _parse_bandwidth(log)
    if not got:
        print("\nno bandwidth rows -- was ET_BENCH_MEM compiled in?")
        return
    print("\n=== memory bandwidth ===")
    for kind, value in got.items():
        print(f"  {kind:6s} {value:8.1f} MB/s")
    # The controls: reference figures measured independently on this board. If
    # these two do not reproduce, the flash figure from the same loop is not
    # trustworthy either.
    controls_ok = True
    # The SRAM control is compared method for method with the reference
    # benchmark (MALLOC_CAP_INTERNAL, best of 16 passes) when the firmware
    # printed that row; the plain DRAM average is reported alongside.
    if "SRAM_CONTROL" in got:
        got = {**got, "SRAM": got["SRAM_CONTROL"]}
    for kind, want in (("SRAM", 240.0), ("PSRAM", 60.7)):
        if kind not in got:
            controls_ok = False
            print(f"  control: {kind} MISSING (expected about {want} MB/s)")
            continue
        off = abs(got[kind] - want) / want
        matches = off < 0.25
        controls_ok &= matches
        verdict = "matches" if matches else "DIVERGES from"
        print(f"  control: {kind} {verdict} the reference {want} MB/s "
              f"({off * 100:.0f}% off)")
    if "FLASH" in got:
        if controls_ok:
            print(f"\n  -> put flash_read_mb_s={got['FLASH']:.1f} in "
                  "chips.py's ESP32S3 entry")
            print("     so streamed weight bytes can be turned into time")
        else:
            print("\n  -> DO NOT record flash_read_mb_s: a positive control "
                  "is missing or diverged")


TIER_RATE = re.compile(
    r"TIER_RATE tier=(?P<tier>\w+) kernel=(?P<kernel>\w+) cores=(?P<cores>\d+) "
    r"k=(?P<k>\d+) rows=(?P<rows>\d+) bytes=(?P<bytes>\d+) us=(?P<us>[\d.]+) "
    r"best_us=(?P<best_us>\d+) mb_s=(?P<mb_s>[\d.]+) check=(?P<check>\S+)")
MSPI = re.compile(r"\[mspi\] psram din_mode (0x[0-9a-fA-F]+)")
DECODE_STEP = re.compile(
    r"^\s*(?P<label>decode_i4(?:_psram)?) step (?P<step>\d+):.*?time (?P<us>\d+) us"
    r"\s+bits (?P<bits>[0-9a-f]{8})", re.M)
DECODE_TIER = re.compile(
    r"decode_i4_psram weight tier: budget (?P<budget>-?\d+), (?P<resident>\d+) B")


def _parse_tier_rates(log):
    """The weight tiers' kernel-rate rows (main/bench_tiers.cpp), as dicts."""
    rows = []
    for m in TIER_RATE.finditer(log):
        row = m.groupdict()
        for key in ("cores", "k", "rows", "bytes", "best_us"):
            row[key] = int(row[key])
        for key in ("us", "mb_s"):
            row[key] = float(row[key])
        rows.append(row)
    return rows


def report_tier_rates(log, board, caches):
    """Echo the kernel rates, and print them as chips.py MEASURED entries
    when every tier and core computed the same rows (the checksums agree per
    kernel: every tier reads copies of the same bytes). Returns the entries
    printed, or [] when there was nothing to record."""
    rows = _parse_tier_rates(log)
    if not rows:
        print("\nno weight-tier rate rows -- was ET_BENCH_MEM compiled in?")
        return []
    mspi = MSPI.search(log)
    mspi = mspi.group(1).lower() if mspi else None
    print(f"\n=== weight tiers: kernel rates ({board}, {caches} caches, mspi "
          f"{mspi or 'not printed'}) ===")
    agree = True
    for kernel in dict.fromkeys(row["kernel"] for row in rows):
        checks = {row["check"] for row in rows if row["kernel"] == kernel}
        same = len(checks) == 1
        agree &= same
        for row in rows:
            if row["kernel"] == kernel:
                print(f"  {row['tier']:6s} {kernel} {row['cores']} core(s): "
                      f"{row['mb_s']:7.2f} MB/s ({row['rows']} rows, "
                      f"{row['bytes']:,} B, {row['us']:.0f} us)")
        print(f"  {kernel}: checksums " + ("agree" if same else
                                           "DIFFER: " + ", ".join(sorted(checks))))
    if not agree:
        print("\n  -> DO NOT record these rates: the tiers computed different rows")
        return []
    entries = [
        f'    _measured("{board}", "{row["tier"]}", "{row["kernel"]}", '
        f'{row["cores"]}, "{caches}", {row["mb_s"]:.1f},\n'
        f'              "bench_tiers, k={row["k"]}, {row["rows"]} rows, mean of 3",\n'
        f'              mspi={mspi!r}),'
        for row in rows]
    print("\n  -> chips.py MEASURED entries:")
    for entry in entries:
        print(entry)
    return entries


def report_decode_tiers(log):
    """The int4 decode row read from flash against the same row with every
    weight copied into PSRAM: their median step times, and whether every
    step's output bits agree (they must: a copy holds the stored bytes).
    Returns whether the bits agree; None when the PSRAM row did not run."""
    steps = {"decode_i4": {}, "decode_i4_psram": {}}
    for m in DECODE_STEP.finditer(log):
        steps[m["label"]][int(m["step"])] = (int(m["us"]), m["bits"])
    flash, psram = steps["decode_i4"], steps["decode_i4_psram"]
    if not psram:
        return None
    same = flash.keys() == psram.keys() and all(
        flash[i][1] == psram[i][1] for i in flash)

    def median(times):
        times = sorted(times)
        return times[len(times) // 2] if times else 0

    t_flash = median(us for us, _ in flash.values())
    t_psram = median(us for us, _ in psram.values())
    tier = DECODE_TIER.search(log)
    staged = int(tier["resident"]) if tier else 0
    print("\n=== int4 decode: weights from flash against PSRAM copies ===")
    print(f"  median step: flash {t_flash:,} us, PSRAM {t_psram:,} us "
          f"({staged:,} B copied)")
    if staged and t_flash > t_psram:
        print(f"  saving: {(t_flash - t_psram) / (staged / 1e6):.1f} us per MB "
              f"copied, per step")
    print("  output bits: " + ("identical at every step" if same else "DIFFER"))
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None,
                    help="the board's USB serial number (`python -m "
                         "serial.tools.list_ports -v`); the board is found by it")
    ap.add_argument("--port", default=None,
                    help="serial port, checked against --serial; found by it "
                         "if omitted")
    # The target is a board profile (scripts/board.py): the PaperColor and the
    # Nano are both ESP32-S3 N16R8 and build identically, but must never be
    # mistaken for each other.
    ap.add_argument("--board", choices=tuple(hardware.BOARD_PROFILES),
                    default="nano", help="board profile (default: nano)")
    ap.add_argument("--baud", type=int, default=2_000_000,
                    help="flash baud (default: 2000000)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse run_parity's export checkpoint (refused if "
                         "the export sources or artifacts changed)")
    ap.add_argument("--log", help="also write the whole device log here")
    ap.add_argument("--replay", metavar="LOG",
                    help="compare a saved device log (from --log) again, "
                         "with no board: needs --resume and the build "
                         "directory that produced it")
    ap.add_argument("--cache-defaults", action="store_true",
                    help="build with the story's caches (sdkconfig.cache."
                         "defaults: 64 KB data cache, 64 B lines), in "
                         "build_hardware_cache")
    ap.add_argument("--rates", metavar="LOG",
                    help="only read the bandwidth and weight-tier rows of a "
                         "saved boot log (a parity or a story bench build's): "
                         "no board, no build")
    ap.add_argument("--caches", choices=("idf", "story", "probe"),
                    help="with --rates: the build's cache configuration "
                         "(default: story when the log has the story's "
                         "golden line, else idf)")
    args = ap.parse_args()
    if args.rates:
        with open(args.rates, newline="") as f:
            log = f.read().replace("\r\n", "\n")
        caches = args.caches or ("story" if "STORY_GOLDEN" in log else "idf")
        report_bandwidth(log)
        report_tier_rates(log, args.board, caches)
        return 0 if report_decode_tiers(log) is not False else 1
    build_dir = BUILD_CACHE if args.cache_defaults else BUILD

    captured = {}
    if args.replay:
        if not args.resume:
            raise SystemExit("--replay compares against the checkpoint the "
                             "run used: pass --resume")

        def build():
            return build_dir

        def run():
            with open(args.replay, newline="") as f:
                captured["log"] = f.read().replace("\r\n", "\n")
            return captured["log"]
    else:
        port = hardware.validate_port(args.port, args.board, args.serial)
        print(f"port: {port} ({hardware.BOARD_PROFILES[args.board]['label']})")

        def build():
            build_hw(build_dir, args.cache_defaults)
            return build_dir

        def run():
            # The USB serial console ends lines with CRLF; run_parity's
            # line-anchored patterns (`...$` under MULTILINE) expect LF, as
            # QEMU prints.
            log = flash_and_run(port, args.baud, timeout=600, board=args.board,
                                serial=args.serial, build=build_dir)
            captured["log"] = log.replace("\r\n", "\n")
            return captured["log"]

    parity_args = ["--resume"] if args.resume else []
    if args.log:
        parity_args += ["--log", args.log]
    parity_rc = P.main(parity_args, build_fn=build, run_fn=run,
                       target="silicon")
    log = captured.get("log", "")

    fails = []
    # Verified, not assumed: the cache variable this build overrides is exactly
    # the kind that fails silently.
    m = re.search(r"rmsnorm path: (\w+)", log)
    if m:
        print(f"\nrmsnorm path taken: {m.group(1)}"
              + ("  <- rsqrt0.s executed" if m.group(1) != "SCALAR" else ""))
    if not m or m.group(1) == "SCALAR":
        print("WARNING: the device took the SCALAR rmsnorm path (or did not "
              "say) -- ET_QEMU_NO_RSQRT0 is still set in the build cache, so "
              "rmsnorm_s3.S did NOT run. Delete the build_hardware directory "
              "and retry.")
        fails.append("rmsnorm_path")

    report_bandwidth(log)
    report_tier_rates(log, args.board, "story" if args.cache_defaults else "idf")
    if report_decode_tiers(log) is False:
        fails.append("decode_i4_psram bits")

    print()
    if parity_rc or fails:
        print("HARDWARE FAILURES: "
              + ", ".join((["parity suite (above)"] if parity_rc else []) + fails))
        print("A row that passes under QEMU and fails here is a real "
              "emulator-vs-silicon divergence, not flake. The prime suspects "
              "are rsqrt0.s, which only this build reaches, and anything "
              "timing-dependent.")
        return 1
    print("ALL PASS on hardware")
    return 0


if __name__ == "__main__":
    sys.exit(main())
