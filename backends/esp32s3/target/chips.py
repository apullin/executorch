# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-target numbers as data: the chip descriptor the AOT layer consumes.

The backend's decisions that depend on the *part* rather than the *model* -- the
vector width, the SRAM-residency threshold -- read their numbers from one
descriptor: the `auto` format policy takes int4 for a weight at or above
sram_resident_bytes (as int8) and int8 below it. flash_model_budget records the
reference board's model region, which test_weight_tiers_report.py holds to the
example's flash layout; nothing in the backend plans against it.
The descriptor is also the honest unit of multi-chip support: a second chip is a new
entry here, not edits across the backend.

What multi-chip actually costs, so it is written down once:

  AOT (this file's consumers)  Already chip-agnostic. The partitioner, passes, format
                               ladder and blob layout use only these numbers.
  Runtime C++                  Already 90% portable: every quantised path in
                               esp32_backend.cpp has a scalar fallback (dot_i8_scalar,
                               the i16 loop, the embed gather) and the rmsnorm op
                               falls back to C. The S3-locked surface is the PIE
                               assembly (runtime/kernels/esp32s3, ops/esp32s3) and the
                               PIE alignment dispatch, all behind guards. A chip with
                               has_pie=False links no .S and always takes the scalar
                               arm -- a build-target change, not a port.
  Kernels per ISA              The esp-nn precedent: shared interface, per-target
                               implementation directory. Note their accumulator
                               geometries differ (S3 QACC is 320-bit as 16x20-bit; the
                               ESP32-P4's is 512-bit as 16x32-bit), so its dot_i8 is a
                               new bit-exactness study, not a translation.

Provenance discipline: the ESP32-S3 entry is MEASURED (a bandwidth bench on a real
N16R8; the flash budget from the partition table's arithmetic).
A second chip's entry carries only what is measured or stated plainly in its
datasheet, with None for anything else -- so building for it fails loudly on the
missing number instead of silently inheriting an S3 figure.

RATES, below the descriptors, is the board's measured read rates, keyed by
everything a rate depends on: the board, the memory tier, the loop reading it
(a plain word loop or a weight kernel), the cores, the cache configuration and
the PSRAM read-timing point the boot tuning chose. A key nobody measured is
None. The weight-tier report (utils/weight_tiers.py) turns them into time
saved per inference.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Chip:
    name: str
    isa: str
    cores: int
    # The S3's 128-bit integer SIMD (PIE, the ee.* opcodes). This is the whole reason
    # the backend targets the S3: no other Xtensa ESP32 has it.
    has_pie: bool
    fpu: str                      # "fp32-scalar" | "none"
    # SIMD load width the weight layout aligns rows to. Kept at 16 even for chips with
    # no SIMD, deliberately: alignment padding is harmless there, and one layout means
    # one .ptd can serve any chip.
    vec_bytes: int
    # Policy threshold: weights at/above this (as i8 bytes) are assumed not
    # internal-SRAM resident, so their op is bandwidth-bound and cheaper formats win.
    sram_resident_bytes: Optional[int]
    # Bytes the model partition can hold on the reference board for this chip.
    flash_model_budget: Optional[int]
    # Measured sequential read bandwidths, MB/s. None = not measured, never guessed.
    sram_read_mb_s: Optional[float]
    psram_read_mb_s: Optional[float]
    # Sequential flash read through esp_partition_mmap, the tier streamed weights
    # come from. Defaults to None because only the reference board has been measured.
    flash_read_mb_s: Optional[float] = None
    notes: str = ""


ESP32S3 = Chip(
    name="esp32s3", isa="xtensa-lx7", cores=2, has_pie=True, fpu="fp32-scalar",
    vec_bytes=16,
    sram_resident_bytes=128 * 1024,   # deliberate under-estimate of ~277 KB free at
                                      # boot; activations, the method arena and the KV
                                      # cache all want the same pool
    flash_model_budget=15_269_888,    # N16R8: the model region of the lean
                                      # table, 0x170000 to the end of flash
                                      # (examples/espressif/esp32s3/scripts/
                                      # flash_layout.py ET_PACKAGE_BUDGET):
                                      # 16 MiB less the bootloader, the
                                      # partition table, NVS, PHY and the app
    sram_read_mb_s=240.0,             # measured on a real N16R8
    psram_read_mb_s=60.7,             # measured on octal PSRAM; a quad R2 part would
                                      # roughly halve this
    flash_read_mb_s=26.6,             # measured on the M5Stack PaperColor
                                      # (QIO 80 MHz, 1 MB sequential via
                                      # esp_partition_mmap) by run_hardware.py, with
                                      # both SRAM and PSRAM controls reproducing the
                                      # figures above
    notes="The reference target. All numbers measured on a real N16R8.",
)


# ---- measured read rates ----------------------------------------------------
#
# How fast a weight reads from each memory tier, as the board measured it.
# The weight tiers copy weights from flash into PSRAM, and what a copy is worth
# is the difference between the two rates for the kernel that reads it, so a
# rate means nothing without what it was measured with:
#
#   board    the board (the PaperColor and the Nano are both N16R8 but are
#            different boards, scripts/board.py BOARD_PROFILES)
#   tier     "flash" (through esp_partition_mmap, as weights are read),
#            "psram" or "sram"
#   kernel   "seq": a word loop summing volatile uint32s, the bandwidth
#            bench's method; "i4": int4 rows through the fused PIE kernel
#            (dot_i4_groups_s3), the backend's decode loop; "i8": int8 rows
#            through dot_i8_s3; "f16", "bf16": the backend's float loop
#            (esp32_float::matmul) on 2-byte weights. A kernel rate counts the
#            bytes the rows hold, codes and scales.
#   cores    1, or 2 with the rows split between the cores as min_latency does
#   caches   the cache configuration (CACHES): IDF's defaults, the story's
#            sdkconfig.cache.defaults, or the kernel probe's
#   mspi     MSPI0's PSRAM din_mode after the boot-time tuning (the firmware
#            prints it as [mspi]): octal PSRAM at 80 MHz is tuned at every boot
#            and the story decodes 0.65% faster at one point than the other.
#            None when the run did not record it.
#
# MEASURED holds every rate with where it was measured; a key it lacks has
# not been measured, and rate() gives None for it. WANTED lists the keys
# main.cpp's bench_memory rows measure (run_hardware.py parses them) that
# are still missing.

CACHES = {
    "idf": "IDF defaults: 32 KB data cache with 32 B lines, 16 KB instruction cache",
    "story": "sdkconfig.cache.defaults: 64 KB data cache with 64 B lines, 32 KB "
             "instruction cache (every story build)",
    "probe": "the kernel probe's build: 64 KB data cache with 64 B lines, 16 KB "
             "instruction cache",
}
# Each configuration's data cache (KB, line bytes) and instruction cache (KB),
# which is what makes one nearer another (nearest_rate).
CACHE_SHAPES = {"idf": ((32, 32), 16), "story": ((64, 64), 32),
                "probe": ((64, 64), 16)}
TIERS = ("flash", "psram", "sram")
KERNELS = ("seq", "i4", "i8", "f16", "bf16")


@dataclass(frozen=True)
class RateKey:
    board: str
    tier: str
    kernel: str
    cores: int
    caches: str
    mspi: Optional[str] = None


@dataclass(frozen=True)
class Rate:
    mb_s: float   # decimal MB/s: bytes per microsecond
    source: str   # the log and line it came from


def _measured(board, tier, kernel, cores, caches, mb_s, source, mspi=None):
    return RateKey(board, tier, kernel, cores, caches, mspi), Rate(mb_s, source)


# The PSRAM timing point the story bench build's [mspi] line printed (din_mode
# 1 in every field).
STORY_BENCH_MSPI = "0x01249249"


MEASURED = dict([
    # main.cpp's bandwidth bench in the parity firmware, which run_hardware.py
    # builds with IDF's caches, on the PaperColor.
    _measured("papercolor", "flash", "seq", 1, "idf", 26.6,
              "bench_memory, 1 MB through esp_partition_mmap, QIO 80 MHz "
              "(run_hardware.py, 2026-09-09)"),
    _measured("papercolor", "psram", "seq", 1, "idf", 56.3,
              "bench_memory, 1 MB, two passes; 55.1 on the rerun "
              "(run_hardware.py, 2026-09-09)"),
    _measured("papercolor", "sram", "seq", 1, "idf", 239.6,
              "bench_memory, 128 KB of MALLOC_CAP_INTERNAL, best of 16; DRAM "
              "with MALLOC_CAP_8BIT averages 137.1 (run_hardware.py, "
              "2026-09-09)"),
    # A kernel probe on the same board: fused int4 rows of k = 256, group 128
    # with fp32 scales. Built with IDF's caches...
    _measured("papercolor", "sram", "i4", 1, "idf", 190.2,
              "kernel probe, k=256; 222.9 at k=768 (2026-10-10)"),
    _measured("papercolor", "psram", "i4", 1, "idf", 54.9,
              "kernel probe, k=256, scales in internal SRAM; 57.0 at k=768 "
              "(2026-10-10)"),
    # ... and with the story's data cache (its 16 KB instruction cache is not
    # the story's 32 KB).
    _measured("papercolor", "sram", "i4", 1, "probe", 190.2,
              "kernel probe, k=256; 220.1 at k=768 (2026-10-10)"),
    _measured("papercolor", "psram", "i4", 1, "probe", 79.6,
              "kernel probe, k=256, weights and fp32 scales in PSRAM; 80.5 "
              "with the scales internal (2026-10-10)"),
    _measured("papercolor", "psram", "i4", 2, "probe", 89.5,
              "kernel probe, k=256 split over both cores, weights and fp32 "
              "scales in PSRAM; 89.5 at k=768 (2026-10-10)"),
    # The weight tiers' kernel rows (bench_tiers.cpp) in a story bench build
    # (CONFIG_ET_BENCH_MEM, sdkconfig.bench.defaults: the story's caches and
    # its staged weight layout), on the PaperColor at PSRAM din_mode 1
    # (run_hardware.py --rates). Every tier read the same bytes and
    # the checksums agree. int4 and int8 are bound by memory in flash and PSRAM
    # (a second core adds 14% and 7% from flash, 24% and 4% from PSRAM); the
    # float loop (one core) is bound by its arithmetic for fp16, which reads
    # PSRAM only 6% faster than flash, and partly for bf16 (45% faster). The
    # same run's word-loop rows are not recorded: their PSRAM control was 29%
    # off the reference, which was measured with IDF's caches.
    _measured("papercolor", "flash", "i4", 1, "story", 28.4,
              "bench_tiers, k=256, 8192 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "flash", "i4", 2, "story", 32.4,
              "bench_tiers, k=256, 8192 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "psram", "i4", 1, "story", 73.3,
              "bench_tiers, k=256, 8192 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "psram", "i4", 2, "story", 91.1,
              "bench_tiers, k=256, 8192 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "sram", "i4", 1, "story", 163.3,
              "bench_tiers, k=256, 992 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "sram", "i4", 2, "story", 289.7,
              "bench_tiers, k=256, 992 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "flash", "i8", 1, "story", 30.2,
              "bench_tiers, k=256, 4096 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "flash", "i8", 2, "story", 32.4,
              "bench_tiers, k=256, 4096 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "psram", "i8", 1, "story", 87.9,
              "bench_tiers, k=256, 4096 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "psram", "i8", 2, "story", 91.4,
              "bench_tiers, k=256, 4096 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "sram", "i8", 1, "story", 341.6,
              "bench_tiers, k=256, 504 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "sram", "i8", 2, "story", 541.5,
              "bench_tiers, k=256, 504 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "flash", "f16", 1, "story", 24.8,
              "bench_tiers, k=256, 2048 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "psram", "f16", 1, "story", 26.3,
              "bench_tiers, k=256, 2048 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "sram", "f16", 1, "story", 26.9,
              "bench_tiers, k=256, 256 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "flash", "bf16", 1, "story", 31.2,
              "bench_tiers, k=256, 2048 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "psram", "bf16", 1, "story", 45.3,
              "bench_tiers, k=256, 2048 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
    _measured("papercolor", "sram", "bf16", 1, "story", 59.3,
              "bench_tiers, k=256, 256 rows, mean of 3 (2026-11-01)",
              mspi=STORY_BENCH_MSPI),
])


# What bench_memory's kernel rows measure in an ET_BENCH_MEM story build, with
# the story's caches (and the boot's PSRAM timing point, which the rows
# record): from every tier, both integer kernels on one core and on two, and
# the float loop on 2-byte weights on the one core the backend runs it on.
WANTED = tuple(RateKey("papercolor", tier, kernel, cores, "story")
               for tier in TIERS
               for kernel, cores in (("i4", 1), ("i4", 2), ("i8", 1),
                                     ("i8", 2), ("f16", 1), ("bf16", 1)))


def unmeasured() -> tuple:
    """The WANTED keys with no measurement at any PSRAM timing point."""
    have = {RateKey(k.board, k.tier, k.kernel, k.cores, k.caches)
            for k in MEASURED}
    return tuple(key for key in WANTED if key not in have)


def rate(board: str, tier: str, kernel: str, cores: int, caches: str,
         mspi: Optional[str] = None) -> Optional[Rate]:
    """The measured rate for exactly this key, or None: not measured."""
    return MEASURED.get(RateKey(board, tier, kernel, cores, caches, mspi))


def nearest_rate(board: str, tier: str, kernel: str, cores: int, caches: str,
                 mspi: Optional[str] = None):
    """(Rate, the key it was measured at) for this key, or the nearest
    measured one on the same board and tier; (None, None) when the tier has
    no measurement on the board. Nearer means, in order: the same kernel,
    then the nearest caches (the same data cache, then the same instruction
    cache: CACHE_SHAPES), the same cores, and the same PSRAM timing point. A
    report that uses a rate from another key says which: the gap between the
    key asked for and the key measured is a measurement still to make, not a
    rate."""
    asked = RateKey(board, tier, kernel, cores, caches, mspi)
    if asked in MEASURED:
        return MEASURED[asked], asked
    want = CACHE_SHAPES.get(caches)

    def distance(key):
        shape = CACHE_SHAPES.get(key.caches)
        return (key.kernel != kernel,
                shape is None or want is None or shape[0] != want[0],
                key.caches != caches and (shape is None or want is None
                                          or shape[1] != want[1]),
                key.cores != cores, key.mspi != mspi)

    near = [key for key in MEASURED if key.board == board and key.tier == tier]
    if not near:
        return None, None
    key = min(near, key=lambda k: (distance(k), k.kernel, k.caches, k.cores,
                                   k.mspi or ""))
    return MEASURED[key], key
