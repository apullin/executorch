# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16)

# The runner's own components (the ExecuTorch core, the story display) and the
# Espressif backend's ESP-IDF component, both from the tree this project sits in.
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/components"
    "${CMAKE_CURRENT_LIST_DIR}/../../../backends/esp32s3/esp-idf")

# LVGL, for the story's display (components/papercolor): by default the ESP
# Component Registry's lvgl/lvgl 9.5.0 (components/papercolor/idf_component.yml),
# which the component manager downloads into managed_components/. To build
# offline, name a local LVGL 9.5.0 checkout in ESPRESSIF_LVGL_DIR
# (-DESPRESSIF_LVGL_DIR=<path>, or the environment variable): it is added as
# the component lvgl and the component manager stays off.
if(NOT ESPRESSIF_LVGL_DIR)
    set(ESPRESSIF_LVGL_DIR "$ENV{ESPRESSIF_LVGL_DIR}")
endif()
if(ESPRESSIF_LVGL_DIR)
    list(APPEND EXTRA_COMPONENT_DIRS "${ESPRESSIF_LVGL_DIR}")
    set(ESPRESSIF_LVGL lvgl)
elseif("$ENV{IDF_COMPONENT_MANAGER}" STREQUAL "0")
    message(FATAL_ERROR "The component manager is off (IDF_COMPONENT_MANAGER=0) and cannot "
                        "fetch lvgl/lvgl: set ESPRESSIF_LVGL_DIR to a local LVGL 9.5.0")
else()
    set(ESPRESSIF_LVGL lvgl__lvgl)
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
if(ESPRESSIF_LVGL_DIR)
    idf_build_set_property(IDF_COMPONENT_MANAGER 0)
endif()
project(et_runner)

# LVGL reads components/papercolor/ui/lv_conf.h instead of its Kconfig menu.
idf_component_get_property(lvgl_lib ${ESPRESSIF_LVGL} COMPONENT_LIB)
target_compile_definitions(${lvgl_lib} PUBLIC LV_KCONFIG_IGNORE
    "LV_CONF_PATH=\"${CMAKE_CURRENT_LIST_DIR}/components/papercolor/ui/lv_conf.h\"")
# That configuration gives LVGL a PSRAM heap on the board (LV_STDLIB_CUSTOM);
# its allocator lives in the lvgl library itself, beside the calls it answers.
target_sources(${lvgl_lib} PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/components/papercolor/lv_mem_psram.c")

if(NOT ET_PLE_ONLY)
# Put the weight file into the 'weights' partition, so it is flashed (and assembled
# into the QEMU image) separately from the app rather than linked into it.
esptool_py_flash_to_partition(flash "weights"
    "${CMAKE_CURRENT_LIST_DIR}/main/block_weights.ptd")

# The delegate's own weights, tagged external by the backend.
esptool_py_flash_to_partition(flash "dweights"
    "${CMAKE_CURRENT_LIST_DIR}/main/esp32_weights.ptd")

# Cached i8 Glimmer exposes prefill and decode from one PTE. ExecuTorch
# content-deduplicates their learned tensors into this one named-data map while
# its shared memory plan assigns both methods the same mutable KV arena.
esptool_py_flash_to_partition(flash "glimmerw"
    "${CMAKE_CURRENT_LIST_DIR}/main/glimmer_weights.ptd")

# Packed-i4 cached Glimmer decode: separate program means a separate NamedDataMap.
esptool_py_flash_to_partition(flash "dcweights4"
    "${CMAKE_CURRENT_LIST_DIR}/main/decode_i4_weights.ptd")

# The table op's parity rows: a whole program (not a weight file), one method
# per row, mapped and loaded straight from its partition.
esptool_py_flash_to_partition(flash "luts"
    "${CMAKE_CURRENT_LIST_DIR}/main/luts.pte")

# The weight-affine parity rows (backends/esp32s3/test/fixtures/asym.py):
# small programs packed into one image, since the app partition has no room left
# for them, and the tied head's named data.
esptool_py_flash_to_partition(flash "asymprog"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_fixtures.bin")
esptool_py_flash_to_partition(flash "asymw"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_weights.ptd")
# ...and the dynamic-activation rows (fixtures/asym_act.py), the same way.
esptool_py_flash_to_partition(flash "asymact"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_act_fixtures.bin")
esptool_py_flash_to_partition(flash "asymactw"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_act_weights.ptd")
# ...and the static int8 graphs (fixtures/asym_static.py), the same way.
esptool_py_flash_to_partition(flash "asymstat"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_static_fixtures.bin")
# ExecuTorch's own LLM recipes (fixtures/stock_llm.py): one image, each row a
# program run as a sequence.
esptool_py_flash_to_partition(flash "stockllm"
    "${CMAKE_CURRENT_LIST_DIR}/main/stock_llm_fixtures.bin")
# The int16 activations (fixtures/asym_a16.py, Arm a16w8), the same way.
esptool_py_flash_to_partition(flash "asyma16"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_a16_fixtures.bin")
# The QAT round trips (fixtures/asym_qat.py), the same way.
esptool_py_flash_to_partition(flash "asymqat"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_qat_fixtures.bin")
# Stage B's rows (fixtures/asym_stage_b.py), each row's Stage A twin beside it.
esptool_py_flash_to_partition(flash "asymstgb"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_stageb_fixtures.bin")
# The ops between delegates (fixtures/asym_static.py's second image), the same way.
esptool_py_flash_to_partition(flash "asymops"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_ops_fixtures.bin")
# The int16 table op's rows (fixtures/lut16.py), one program per a16w8 site.
esptool_py_flash_to_partition(flash "asymlut16"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_lut16_fixtures.bin")
# The fused quantised norms (fixtures/qnorm.py), the same way.
esptool_py_flash_to_partition(flash "asymnorm"
    "${CMAKE_CURRENT_LIST_DIR}/main/asym_norm_fixtures.bin")
# The CNN and RNN rows (fixtures/cnn.py), the same way, and the named data
# every one of them reads.
esptool_py_flash_to_partition(flash "cnnrnn"
    "${CMAKE_CURRENT_LIST_DIR}/main/cnn_fixtures.bin")
esptool_py_flash_to_partition(flash "cnnrnnw"
    "${CMAKE_CURRENT_LIST_DIR}/main/cnn_weights.ptd")
endif()

# The lean PLE build only (ET_PLE_ONLY, sdkconfig.ple): the model's program and
# weights, each in its own partition. The parity build neither needs the files
# nor knows the partitions. The bench build (ET_BENCH_MODEL) flashes a model's
# two partitions itself (scripts/run_bench_hardware.py), from its own directory:
# one firmware serves every model whose operators it registers.
if(ET_PLE_ONLY AND NOT ET_BENCH_MODEL)
    esptool_py_flash_to_partition(flash "pleprog"
        "${CMAKE_CURRENT_LIST_DIR}/main/ple.pte")
    esptool_py_flash_to_partition(flash "pleweights"
        "${CMAKE_CURRENT_LIST_DIR}/main/ple_weights.ptd")
endif()
