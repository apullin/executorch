# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The operator library's conformance checks (backends/esp32s3/test/runtime),
# beside their goldens: test code, linked only by builds that call it (the
# parity runner).
get_filename_component(ESPRESSIF_CHECKS
    "${CMAKE_CURRENT_LIST_DIR}/../../../../backends/esp32s3/test/runtime" ABSOLUTE)
set(ET_RUNNER_SRCS
    "main.cpp" "esp_pal.cpp" "esp_partition_loader.cpp" "pte_blobs.S"
    "${ESPRESSIF_CHECKS}/quant_lut_check.cpp" "${ESPRESSIF_CHECKS}/divsqrt_check.cpp"
    "${ESPRESSIF_CHECKS}/quant_lut16_check.cpp" "${ESPRESSIF_CHECKS}/nonfinite_check.cpp"
    "${ESPRESSIF_CHECKS}/expf_check.cpp")
if(ET_STORY)
    list(APPEND ET_RUNNER_SRCS "bpe.c")
endif()
if(ET_PAPERCOLOR_DISPLAY)
    list(APPEND ET_RUNNER_SRCS "papercolor_status.cpp")
endif()
# The bandwidth and weight-tier rate rows (below): -DET_BENCH_MEM=1, or
# CONFIG_ET_BENCH_MEM (Kconfig.projbuild) for a build whose command line is
# fixed -- the story's, with sdkconfig.bench.defaults as its extra defaults.
if(CONFIG_ET_BENCH_MEM)
    set(ET_BENCH_MEM 1)
endif()
if(ET_BENCH_MEM)
    list(APPEND ET_RUNNER_SRCS "bench_tiers.cpp")
endif()

idf_component_register(
    SRCS ${ET_RUNNER_SRCS}
    INCLUDE_DIRS "."
    REQUIRES executorch espressif_backend esp_timer esp_partition
    PRIV_REQUIRES papercolor nvs_flash esp_driver_gpio)

# .incbin is resolved by the assembler, not the preprocessor, so the search path
# has to reach `as` via -Wa,-I rather than a plain -I.
set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/pte_blobs.S" APPEND PROPERTY
             COMPILE_OPTIONS "-Wa,-I${CMAKE_CURRENT_LIST_DIR}")

# CMake cannot see through .incbin, so without this an updated .pte does NOT
# trigger a rebuild and the firmware silently keeps the previous model. The
# lean runner does not incbin any program (its sole program is a flash
# partition), so a clean ET_PLE_ONLY build must not demand the parity files.
if(NOT ET_PLE_ONLY)
    file(GLOB ET_ALL_GENERATED_PROGRAMS CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_LIST_DIR}/*.pte")
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/pte_blobs.S" APPEND PROPERTY
                 OBJECT_DEPENDS "${ET_ALL_GENERATED_PROGRAMS}"
                 "${CMAKE_CURRENT_LIST_DIR}/asym_refusal_cases.bin")
endif()

target_compile_options(${COMPONENT_LIB} PRIVATE -Os -fno-rtti -w)

# The checks' references compute each step as written, like the kernels they
# check: nothing fused.
set_source_files_properties("${ESPRESSIF_CHECKS}/quant_lut_check.cpp"
    "${ESPRESSIF_CHECKS}/divsqrt_check.cpp" "${ESPRESSIF_CHECKS}/quant_lut16_check.cpp"
    "${ESPRESSIF_CHECKS}/nonfinite_check.cpp" "${ESPRESSIF_CHECKS}/expf_check.cpp"
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# The lean build: the story's runtime/kernels/backend with zero parity blobs --
# the app must fit 0x150000 alongside the model's own 15 MB of partitions in a
# real 16 MB layout. Guards pte_blobs.S (preprocessed, so the .S sees it too)
# and app_main; the backend component reads it too.
if(ET_PLE_ONLY)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_PLE_ONLY=1)
endif()
if(ET_PAPERCOLOR_DISPLAY)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        ET_PAPERCOLOR_DISPLAY=1)
endif()
if(ET_STORY)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_STORY=1
        BPE_TABLES_HEADER="story_bpe_tables.h")
endif()
if(ET_STORY_DUAL_CORE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_STORY_DUAL_CORE=1)
endif()
if(ET_STORY_STAGE_BYTES)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_STORY_STAGE_BYTES=${ET_STORY_STAGE_BYTES})
endif()
# The story's weights and memory (story_runner.inc, "Where the story's memory
# goes"): ranked weight placement (the backend's weight_tier_rank), the PSRAM
# temp tier (0: none), the PSRAM an auto weight budget leaves free, the
# internal SRAM floor, decode's own activations in internal SRAM, the small
# constants outside the delegate in internal SRAM, and the method arena's
# size. Cache variables that run_story_hardware.py passes every build, so each
# is forwarded whenever it is set, zero included: a zero must not fall back to
# a nonzero default.
foreach(setting ET_STORY_TIER_RANK ET_STORY_TEMP_BYTES ET_STORY_TIER_RESERVE
                ET_STORY_INTERNAL_FLOOR ET_STORY_DECODE_ACT_INTERNAL ET_STORY_HOT_CONSTANTS
                ET_STORY_ARENA_BYTES)
    if(DEFINED ${setting})
        target_compile_definitions(${COMPONENT_LIB} PRIVATE ${setting}=${${setting}})
    endif()
endforeach()
# The budgeted silicon-parity pass: run_model gives the backend this many bytes
# of PSRAM for weight copies. A cache variable, so it persists in its build
# directory: build the pass in its own, or clear it with 0.
if(ET_PARITY_TIER_BYTES)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_PARITY_TIER_BYTES=${ET_PARITY_TIER_BYTES})
endif()
# With it, ranked placement (weight_tier_rank) made by each row's first
# execute. Also a cache variable: clear it with 0.
if(ET_PARITY_TIER_RANK)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_PARITY_TIER_RANK=1)
endif()
if(ET_PAPERCOLOR_DUMP_FRAME)  # the drawn frame as hex rows in the log: what the panel shows, no camera
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        ET_PAPERCOLOR_DUMP_FRAME=1)
endif()
if(ET_FORCE_SCALAR_BACKEND)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        ET_FORCE_SCALAR_BACKEND=1)
endif()
if(ET_HARDWARE_RUNNER_GATE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_HARDWARE_RUNNER_GATE=1)
endif()
# The bench build (scripts/run_bench_hardware.py, with ET_PLE_ONLY): one model,
# timed, read from the bench image in the program partition
# (bench_runner.inc). ET_BENCH_TIER_BYTES is its psram measurement's weight
# budget (default -1, every weight PSRAM holds), ET_BENCH_DUAL_CORE splits the
# matmuls across both cores. The build profile goes into every bench log.
if(ET_BENCH_MODEL)
    set(_et_bench_profile "backend -Os")
    if(ESP32_BACKEND_O2)
        set(_et_bench_profile "backend -O2")
    endif()
    if(CONFIG_ET_PORTABLE_O2)
        string(APPEND _et_bench_profile ", runtime and portable kernels -O2")
    else()
        string(APPEND _et_bench_profile ", runtime and portable kernels -Os")
    endif()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_BENCH_MODEL=1
        "ET_BENCH_PROFILE=\"${_et_bench_profile}\"")
endif()
if(ET_BENCH_TIER_BYTES)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_BENCH_TIER_BYTES=${ET_BENCH_TIER_BYTES})
endif()
if(ET_BENCH_DUAL_CORE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_BENCH_DUAL_CORE=1)
endif()
# ET_BENCH_MICRO runs the backend's micro-benchmarks at boot (BENCH_MICRO lines:
# esp32_backend_microbench, which only a build with the conv path has).
if(ET_BENCH_MICRO)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_BENCH_MICRO=1)
endif()

# QEMU has no RSQRT0.S -- it faults with IllegalInstruction -- so the parity harness
# builds with -DET_QEMU_NO_RSQRT0=1 and the runner takes the scalar path. rmsnorm_s3.S
# is still assembled and linked in that build, so the toolchain half stays covered.
#
# NOTE this is a CMake CACHE variable: once set in a build directory it persists until
# the cache is cleared, so a hardware build reusing a parity build directory would
# silently inherit it. main.cpp prints the selected path at boot for exactly this reason.
if(ET_QEMU_NO_RSQRT0)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_QEMU_NO_RSQRT0=1)
endif()

# The sequential scalar RMSNorm whatever the build.
if(ET_FORCE_SCALAR_RMS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_FORCE_SCALAR_RMS=1)
endif()

# Measurement build: per-opcode wall time from esp_timer, reported by
# esp32_backend_report_timing(). Off by default because a timer read per op is
# not free and every parity number here is a NUMERIC claim, not a timing one --
# the two builds must not be silently the same binary. The backend component
# reads it too.
if(ESP32_BACKEND_TIMING)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ESP32_BACKEND_TIMING=1)
endif()

# Fixed-memory EventTracer used only by explicit attribution builds. It groups
# portable OPERATOR_CALL events by runtime instruction/debug handle; ordinary
# timing builds keep every hook compiled out.
if(ESP32_OP_TIMING)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        ESP32_OP_TIMING=1 ET_EVENT_TRACER_ENABLED=1)
endif()

# Opt-in approximate-performance rung. The default exact build keeps ET's
# portable LayerNorm expression; this build substitutes one reciprocal followed
# by multiplies so its numerical and timing effects are measured explicitly.
if(ESP32_FAST_LAYERNORM)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        ESP32_FAST_LAYERNORM=1)
endif()

# Memory-bandwidth benchmark, compiled in only for hardware runs (run_hardware.py
# sets it): emulated flash has no relationship to a real SPI part, so measuring
# it under QEMU would produce a confident wrong answer. Its weight-tier rows
# (bench_tiers.cpp) are the backend's decode loops, so they are compiled as
# the backend is: -O2 under ESP32_BACKEND_O2.
if(ET_BENCH_MEM)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_BENCH_MEM=1)
    if(ESP32_BACKEND_O2)
        set_source_files_properties("bench_tiers.cpp" PROPERTIES
            COMPILE_OPTIONS "-O2")
    endif()
endif()
