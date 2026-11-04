# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The Espressif backend as an ESP-IDF component: the delegate (Esp32Backend),
# its PIE kernels, and the operator library (the espressif:: custom ops and the
# target overrides of portable kernels). Add this directory's parent
# (backends/esp32s3/esp-idf) to a project's EXTRA_COMPONENT_DIRS and require
# espressif_backend. It needs the executorch component
# (examples/espressif/esp32s3/components/executorch), whose generated kernel
# registration also supplies Esp32KernelVariant.h.

get_filename_component(ESPRESSIF_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(ESPRESSIF_RUNTIME "${ESPRESSIF_DIR}/runtime")
set(ESPRESSIF_KERNELS "${ESPRESSIF_DIR}/runtime/kernels")
set(ESPRESSIF_OPS "${ESPRESSIF_DIR}/ops")

# The static int8 convolution path (esp32_conv.h): on unless the application
# passes -DESP32_CONV_FAST=0. The story firmware does (it has no convolution),
# so its image carries none of the path's code or fields.
if(NOT DEFINED ESP32_CONV_FAST)
    set(ESP32_CONV_FAST 1)
endif()
set(ESPRESSIF_CONV_SRCS "")
set(ESPRESSIF_CONV_LF "")
if(ESP32_CONV_FAST)
    set(ESPRESSIF_CONV_SRCS "${ESPRESSIF_RUNTIME}/esp32_conv.cpp"
        "${ESPRESSIF_KERNELS}/esp32s3/dw33_qacc16_s3.S"
        "${ESPRESSIF_KERNELS}/esp32s3/dot_i8_rows_s3.S"
        "${ESPRESSIF_KERNELS}/esp32s3/qacc_lane_codes_s3.S"
        "${ESPRESSIF_KERNELS}/esp32s3/dot_rows_codes_s3.S"
        "${ESPRESSIF_KERNELS}/esp32s3/transpose16_s3.S"
        "${ESPRESSIF_KERNELS}/esp32s3/dot_rows_long_s3.S"
        "${ESPRESSIF_KERNELS}/esp32s3/dw_plane_s3.S")
    set(ESPRESSIF_CONV_LF "${ESPRESSIF_RUNTIME}/esp32_conv.lf")
endif()

idf_component_register(
    SRCS "${ESPRESSIF_OPS}/espressif_ops.cpp" "${ESPRESSIF_RUNTIME}/esp32_backend.cpp"
         "${ESPRESSIF_KERNELS}/esp32s3/dot_i8_s3.S" "${ESPRESSIF_KERNELS}/esp32s3/dot_i4_s3.S"
         "${ESPRESSIF_OPS}/esp32s3/rmsnorm_s3.S" "${ESPRESSIF_KERNELS}/esp32s3/unpack_i4_s3.S"
         "${ESPRESSIF_RUNTIME}/esp32_asym.cpp" "${ESPRESSIF_OPS}/quant_lut.cpp"
         "${ESPRESSIF_OPS}/kv_affine.cpp" "${ESPRESSIF_OPS}/quant_lut16.cpp"
         "${ESPRESSIF_OPS}/esp32s3/quant_lut16_s3.S" "${ESPRESSIF_OPS}/qnorm.cpp"
         "${ESPRESSIF_RUNTIME}/esp32_float.cpp" ${ESPRESSIF_CONV_SRCS}
         "${ESPRESSIF_OPS}/qdq_pool.cpp" "${ESPRESSIF_OPS}/layer_norm2d.cpp"
         "${ESPRESSIF_OPS}/elementwise.cpp"
    INCLUDE_DIRS "${ESPRESSIF_RUNTIME}" "${ESPRESSIF_KERNELS}" "${ESPRESSIF_OPS}"
    REQUIRES executorch esp_timer
    LDFRAGMENTS "${ESPRESSIF_RUNTIME}/esp32_asym.lf" "${ESPRESSIF_OPS}/quant_lut.lf"
                "${ESPRESSIF_OPS}/kv_affine.lf" "${ESPRESSIF_OPS}/quant_lut16.lf"
                "${ESPRESSIF_OPS}/qnorm.lf" "${ESPRESSIF_RUNTIME}/esp32_float.lf"
                "${ESPRESSIF_RUNTIME}/esp32_hot.lf"
                ${ESPRESSIF_CONV_LF} "${ESPRESSIF_OPS}/qdq_pool.lf"
                "${ESPRESSIF_OPS}/elementwise.lf")

target_compile_options(${COMPONENT_LIB} PRIVATE -Os -fno-rtti -w)

# Speed rung: the backend's hot loops at -O2 (inlining, unrolling) while the
# rest of the firmware stays -Os, because the app partition is the constraint.
# Source options follow the target's on the command line, so -O2 wins here.
if(ESP32_BACKEND_O2)
    set_source_files_properties("${ESPRESSIF_RUNTIME}/esp32_backend.cpp"
        PROPERTIES COMPILE_OPTIONS "-O2")
endif()
# The speed switches of this directory's Kconfig. The operator library at -O2
# (on by default), as ESP32_BACKEND_O2 does for the delegate:
if(CONFIG_ESP32_BACKEND_OPS_O2)
    set_source_files_properties("${ESPRESSIF_OPS}/espressif_ops.cpp"
        PROPERTIES COMPILE_OPTIONS "-O2")
endif()
# The narrowest projection the min-latency mode splits (kSplitMinCols):
if(CONFIG_ESP32_BACKEND_SPLIT_MIN_COLS)
    set_property(SOURCE "${ESPRESSIF_RUNTIME}/esp32_backend.cpp" APPEND PROPERTY
        COMPILE_DEFINITIONS "ESP32_SPLIT_MIN_COLS=${CONFIG_ESP32_BACKEND_SPLIT_MIN_COLS}")
endif()
# CONFIG_ESP32_BACKEND_HOT_IRAM needs no flag: esp32_hot.h reads it from
# sdkconfig.h for ESP32_HOT_CODE, and esp32_hot.lf from the configuration.

# The blob's asymmetric-quantisation half (esp32_asym.h). Its float epilogues
# must match the Python mirrors and the host bit for bit, so GCC may not fuse
# multiply-adds into madd.s here -- while esp32_backend.cpp keeps -ffp-contract
# at the default, which its symmetric outputs depend on. esp32_asym.lf links
# its code and constants after everything else. -fno-merge-constants keeps its
# strings in its own .rodata: as mergeable sections the linker would pool them
# with every other object's strings, in the middle of flash rodata.
set_source_files_properties("${ESPRESSIF_RUNTIME}/esp32_asym.cpp" PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-fno-merge-constants")
# The table op (quant_lut.h): its quantize step must give the host's codes, and
# quant_lut.lf places it after every other flash object, as esp32_asym.lf does.
set_source_files_properties("${ESPRESSIF_OPS}/quant_lut.cpp" PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-fno-merge-constants")
# ExecuTorch's stock quantised K/V cache kernels (kv_affine.h): the writer's
# float32 and float64 steps must give the host's bits, and kv_affine.lf links
# them after every other flash object, as esp32_asym.lf does.
set_source_files_properties("${ESPRESSIF_OPS}/kv_affine.cpp" PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-fno-merge-constants")
# The int16 table op (quant_lut16.h): every step of its quantize, cubic and
# dequantize arithmetic rounds on its own, as on the host and in the mirrors;
# quant_lut16.lf places it (and its LX7 loops) after every other flash object.
set_source_files_properties("${ESPRESSIF_OPS}/quant_lut16.cpp" PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-fno-merge-constants")
# The fused quantised norm (qnorm.h): every product is followed by a
# quantise or a separately rounded add, as in the graph it replaces, and
# qnorm.lf links it after every other flash object, as quant_lut.lf does.
set_source_files_properties("${ESPRESSIF_OPS}/qnorm.cpp" PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-fno-merge-constants")
# LayerNorm2d (layer_norm2d.h): bit for bit the fourteen-op graph it
# replaces, whose pow and mul kernels round each product before the next op
# adds it, so GCC may not fuse them into madd.s, as espressif_ops.cpp's
# default contraction did; at the operator library's own optimisation level.
if(CONFIG_ESP32_BACKEND_OPS_O2)
    set_source_files_properties("${ESPRESSIF_OPS}/layer_norm2d.cpp" PROPERTIES
        COMPILE_OPTIONS "-O2;-ffp-contract=off")
else()
    set_source_files_properties("${ESPRESSIF_OPS}/layer_norm2d.cpp" PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off")
endif()
# The float-weight loops (esp32_float.h): the fp32 convolution and the
# fp16/bf16 rungs. Unlike the files above they keep esp32_backend.cpp's
# contraction, because their sum must be opcode 1's (run_addmm_fp32), which
# the device fuses into madd.s: a 2-byte weight then computes exactly what the
# fp32 loop computes on its widened value. -O2 whatever the build: at -Os
# the dot is a branch, a jump and an index rebuilt per element (11
# instructions a MAC) where -O2 gives a zero-overhead LOOP of 5. esp32_float.lf
# places them after every other flash object.
set_source_files_properties("${ESPRESSIF_RUNTIME}/esp32_float.cpp" PROPERTIES
    COMPILE_OPTIONS "-O2;-fno-merge-constants")
# The static int8 convolution path (esp32_conv.h): its column math must be
# esp32_asym.cpp's rounding for rounding, so -ffp-contract=off as there, and
# -O2 whatever the build, as esp32_float.cpp: its loops are short and run per
# output element. esp32_conv.lf places it after every other flash object.
if(ESP32_CONV_FAST)
    set_source_files_properties("${ESPRESSIF_RUNTIME}/esp32_conv.cpp" PROPERTIES
        COMPILE_OPTIONS "-O2;-ffp-contract=off;-fno-merge-constants")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ESP32_CONV_FAST=1)
endif()

# The quantize, dequantize and average-pool overrides (qdq_pool.h): exact
# arithmetic, so -ffp-contract=off as esp32_asym.cpp, and -O2 whatever the
# build, as esp32_conv.cpp: their loops run once per tensor element.
# qdq_pool.lf places them after every other flash object. Each override
# compiles only where the program names its operator (Esp32KernelVariant.h),
# so a program without them (the story) gets an empty object.
set_source_files_properties("${ESPRESSIF_OPS}/qdq_pool.cpp" PROPERTIES
    COMPILE_OPTIONS "-O2;-ffp-contract=off;-fno-merge-constants")

# The element-wise loops (elementwise.h): the squeeze-excite scale, the
# pads' row copy and the table ops' loops. Exact arithmetic and -O2 whatever
# the build, as qdq_pool.cpp; elementwise.lf places them after every other
# flash object. The mul and pad paths compile only where the variant header
# asks (gen_kernels.py --elementwise-fast), and nothing reaches the table
# loops but quant_lut.cpp's, so the story firmware links none of this file.
set_source_files_properties("${ESPRESSIF_OPS}/elementwise.cpp" PROPERTIES
    COMPILE_OPTIONS "-O2;-ffp-contract=off;-fno-merge-constants")

# The application's build switches that the backend and the ops read; the
# application component sets the same ones for its own sources.
# The lean build (ET_PLE_ONLY): no parity fixtures, only what a deployed
# program needs.
if(ET_PLE_ONLY)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_PLE_ONLY=1)
endif()
# Measurement build: per-opcode wall time from esp_timer, reported by
# esp32_backend_report_timing(). Off by default: a timer read per op is not
# free, and a timing build must never be mistaken for the parity binary.
if(ESP32_BACKEND_TIMING)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ESP32_BACKEND_TIMING=1)
endif()
# Operator attribution: ExecuTorch's EventTracer hooks, compiled in here as
# they are in the executorch component and the application.
if(ESP32_OP_TIMING)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_EVENT_TRACER_ENABLED=1)
endif()
# Opt-in approximate-performance rung: the reciprocal-times-input LayerNorm.
if(ESP32_FAST_LAYERNORM)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ESP32_FAST_LAYERNORM=1)
endif()
