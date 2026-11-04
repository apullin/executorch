# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(esp32_host_runner LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# backends/esp32s3/test/host_runner -> backends/esp32s3: the delegate and
# its C kernel references (runtime/, runtime/kernels/), the operator library
# (ops/) and the operator checks with their goldens (test/runtime/).
get_filename_component(ESPRESSIF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(ESPRESSIF_RUNTIME "${ESPRESSIF_ROOT}/runtime")
set(ESPRESSIF_KERNELS "${ESPRESSIF_ROOT}/runtime/kernels")
set(ESPRESSIF_OPS "${ESPRESSIF_ROOT}/ops")
set(ESPRESSIF_CHECKS "${ESPRESSIF_ROOT}/test/runtime")
set(ESPRESSIF_INCLUDE_DIRS "${ESPRESSIF_RUNTIME}" "${ESPRESSIF_KERNELS}"
  "${ESPRESSIF_OPS}" "${ESPRESSIF_CHECKS}")
add_library(esp32_reference_kernels STATIC
  "${ESPRESSIF_KERNELS}/esp32_reference_kernels.c")
target_include_directories(esp32_reference_kernels PUBLIC ${ESPRESSIF_INCLUDE_DIRS})
target_compile_definitions(esp32_reference_kernels PRIVATE ESP32_HOST_BUILD=1)
target_compile_options(esp32_reference_kernels PRIVATE -ffp-contract=off)

find_package(Threads REQUIRED)
add_executable(esp32_kernel_test kernel_test.cpp)
target_compile_definitions(esp32_kernel_test PRIVATE ESP32_HOST_BUILD=1)
target_compile_options(esp32_kernel_test PRIVATE -ffp-contract=off)
target_link_libraries(esp32_kernel_test PRIVATE esp32_reference_kernels Threads::Threads)

enable_testing()
add_test(NAME esp32_reference_kernels COMMAND esp32_kernel_test)

option(ESP32_HOST_BUILD_RUNTIME "Build the ExecuTorch PTE/PTD runner" ON)
if(NOT ESP32_HOST_BUILD_RUNTIME)
  return()
endif()

# The runtime and kernel sources: the ExecuTorch tree this backend sits in,
# unless ET_ROOT (cache or environment) names another source tree.
if(DEFINED ENV{ET_ROOT})
  set(_et_default "$ENV{ET_ROOT}")
else()
  get_filename_component(_et_default "${ESPRESSIF_ROOT}/../.." ABSOLUTE)
endif()
set(ET_ROOT "${_et_default}" CACHE PATH "ExecuTorch source tree (v1.6.0)")
if(NOT EXISTS "${ET_ROOT}/version.txt")
  message(FATAL_ERROR "Set ET_ROOT to an ExecuTorch v1.6.0 source tree")
endif()
get_filename_component(ET_ROOT "${ET_ROOT}" ABSOLUTE)
file(READ "${ET_ROOT}/version.txt" ET_VERSION_STRING)
string(STRIP "${ET_VERSION_STRING}" ET_VERSION_STRING)
if(NOT ET_VERSION_STRING STREQUAL "1.6.0")
  message(FATAL_ERROR "Host source list supports ExecuTorch v1.6.0, "
                      "not ${ET_VERSION_STRING}")
endif()
string(REPLACE "." ";" _et_version_parts "${ET_VERSION_STRING}")
list(GET _et_version_parts 0 ET_VERSION_MAJOR)
list(GET _et_version_parts 1 ET_VERSION_MINOR)
list(GET _et_version_parts 2 ET_VERSION_PATCH)
set(ET_HOST_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file("${ET_ROOT}/runtime/core/version.h.in"
  "${ET_HOST_GENERATED}/executorch/runtime/core/version.h" @ONLY)

set(PYTHON_EXECUTABLE "" CACHE FILEPATH "Python with torchgen and ExecuTorch installed")
if(NOT PYTHON_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

# Sources include <executorch/...>: the directory above a tree named
# executorch, else a link to the tree under that name in the build directory
# (a git worktree has a name of its own).
get_filename_component(_et_name "${ET_ROOT}" NAME)
if(_et_name STREQUAL "executorch")
  get_filename_component(ET_HOST_INCLUDE "${ET_ROOT}" DIRECTORY)
else()
  set(ET_HOST_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/include")
  file(MAKE_DIRECTORY "${ET_HOST_INCLUDE}")
  file(CREATE_LINK "${ET_ROOT}" "${ET_HOST_INCLUDE}/executorch" SYMBOLIC)
endif()

# Submodule headers: ET_ROOT's own submodule by default. A tree whose
# submodules are not initialised (a git worktree, say) can name another
# checkout's copy.
set(ESPRESSIF_FLATBUFFERS_INCLUDE "${ET_ROOT}/third-party/flatbuffers/include"
  CACHE PATH "FlatBuffers headers (the third-party/flatbuffers submodule)")
if(NOT EXISTS "${ESPRESSIF_FLATBUFFERS_INCLUDE}/flatbuffers/flatbuffers.h")
  message(FATAL_ERROR "No FlatBuffers headers in ${ESPRESSIF_FLATBUFFERS_INCLUDE}: "
                      "initialise third-party/flatbuffers or set "
                      "ESPRESSIF_FLATBUFFERS_INCLUDE")
endif()

# The program and FlatTensor schema headers, generated from ET_ROOT's own
# schemas by the flatc the ExecuTorch wheel ships, with the firmware's flags
# (no --gen-mutable: nothing here writes into a program).
set(FLATC_EXECUTABLE "" CACHE FILEPATH "flatc (default: the ExecuTorch wheel's)")
if(NOT FLATC_EXECUTABLE)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c [=[
import importlib.util, os
spec = importlib.util.find_spec("executorch")
for root in spec.submodule_search_locations if spec else ():
    if os.path.isfile(os.path.join(root, "data", "bin", "flatc")):
        print(os.path.join(root, "data", "bin", "flatc"))
        break
]=]
    OUTPUT_VARIABLE _wheel_flatc OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _wheel_flatc)
    message(FATAL_ERROR "No flatc in ${PYTHON_EXECUTABLE}'s ExecuTorch package: "
                        "set FLATC_EXECUTABLE")
  endif()
  set(FLATC_EXECUTABLE "${_wheel_flatc}" CACHE FILEPATH
    "flatc (default: the ExecuTorch wheel's)" FORCE)
endif()
set(ET_HOST_SCHEMA "${ET_HOST_GENERATED}/executorch/schema")
set(ET_HOST_FLAT_TENSOR "${ET_HOST_GENERATED}/executorch/extension/flat_tensor/serialize")
set(ET_HOST_SCHEMA_HEADERS
  "${ET_HOST_SCHEMA}/program_generated.h" "${ET_HOST_SCHEMA}/scalar_type_generated.h"
  "${ET_HOST_FLAT_TENSOR}/flat_tensor_generated.h"
  "${ET_HOST_FLAT_TENSOR}/scalar_type_generated.h")
set(_flatc_flags --cpp --cpp-std c++11 --scoped-enums)
add_custom_command(
  OUTPUT ${ET_HOST_SCHEMA_HEADERS}
  COMMAND "${FLATC_EXECUTABLE}" ${_flatc_flags} -o "${ET_HOST_SCHEMA}"
          "${ET_ROOT}/schema/program.fbs" "${ET_ROOT}/schema/scalar_type.fbs"
  COMMAND "${FLATC_EXECUTABLE}" ${_flatc_flags} -o "${ET_HOST_FLAT_TENSOR}"
          "${ET_ROOT}/extension/flat_tensor/serialize/flat_tensor.fbs"
          "${ET_ROOT}/extension/flat_tensor/serialize/scalar_type.fbs"
  DEPENDS "${ET_ROOT}/schema/program.fbs" "${ET_ROOT}/schema/scalar_type.fbs"
          "${ET_ROOT}/extension/flat_tensor/serialize/flat_tensor.fbs"
          "${ET_ROOT}/extension/flat_tensor/serialize/scalar_type.fbs"
  COMMENT "Generate the program and FlatTensor schema headers")
add_custom_target(executorch_host_schema DEPENDS ${ET_HOST_SCHEMA_HEADERS})

# The chip's GCC (gnu++2b, so -ffp-contract=fast) fuses some multiply-adds into
# madd.s/msub.s. "off" evaluates fp32 expressions as written, like the Python
# references; "fast" lets the host compiler fuse as well, to bracket that
# difference (it chooses its own fusions, not GCC's). The C kernel references
# model the assembly's explicit fused steps and always build with "off".
set(ESP32_HOST_FP_CONTRACT off CACHE STRING
  "-ffp-contract for the runtime, portable kernels, delegate and ops")
set_property(CACHE ESP32_HOST_FP_CONTRACT PROPERTY STRINGS off on fast)
add_library(esp32_host_config INTERFACE)
# Everything compiled against the runtime waits for the schema headers.
add_dependencies(esp32_host_config executorch_host_schema)
target_include_directories(esp32_host_config INTERFACE
  "${ET_HOST_GENERATED}"
  "${ET_HOST_INCLUDE}"
  "${ESPRESSIF_FLATBUFFERS_INCLUDE}"
  "${ET_ROOT}/runtime/core/portable_type/c10"
  ${ESPRESSIF_INCLUDE_DIRS})
target_compile_definitions(esp32_host_config INTERFACE
  ESP32_HOST_BUILD=1 C10_USING_CUSTOM_GENERATED_MACROS=1
  ET_ENABLE_PROGRAM_VERIFICATION=1 ET_LOG_ENABLED=1)
target_compile_options(esp32_host_config INTERFACE
  -ffp-contract=${ESP32_HOST_FP_CONTRACT} -Wno-deprecated-declarations)
target_link_libraries(esp32_host_config INTERFACE Threads::Threads)

include(executorch_sources.cmake)
# Object libraries retain the POSIX PAL and static delegate/prim registration
# without platform-specific whole-archive linker flags.
add_library(executorch_host OBJECT ${ET_HOST_RUNTIME_SRCS})
target_link_libraries(executorch_host PUBLIC esp32_host_config)
file(GLOB_RECURSE ET_PORTABLE_SRCS CONFIGURE_DEPENDS
  "${ET_ROOT}/kernels/portable/cpu/*.cpp")
list(FILTER ET_PORTABLE_SRCS EXCLUDE REGEX "/test/|/codegen/")
add_library(executorch_host_portable STATIC ${ET_PORTABLE_SRCS})
target_link_libraries(executorch_host_portable PUBLIC esp32_host_config)
# ExecuTorch's static quantize/dequantize kernels (per tensor, per channel):
# the safety net for the pairs a static int8 graph leaves between portable
# ops, and the per-tensor dequantize by one-element tensor qparams, as the
# firmware registers them for a program that names them
# (codegen_portable.py QUANTIZED registers them; never a chooser).
add_library(executorch_host_quantized STATIC
  "${ET_ROOT}/kernels/quantized/cpu/op_quantize.cpp"
  "${ET_ROOT}/kernels/quantized/cpu/op_dequantize.cpp")
# The per-channel kernels walk their channels with the portable kernels'
# reduce_util, so the portable library follows this one on every link line:
# GNU ld takes a symbol only from an archive after the one that needs it.
target_link_libraries(executorch_host_quantized PUBLIC esp32_host_config
  executorch_host_portable)

# The firmware's operator library, compiled unchanged. gen_kernels.py writes
# its variant header beside the firmware registration; the host builds the
# default variant (ExecuTorch's portable LayerNorm) with every portable
# kernel linked, so the overrides a firmware takes only when its program names
# their operators (qdq_pool.h), and every kernel of the library, which a
# firmware registers only for a program that names its operator
# (ESP32_GENERATED_OP_*), are all on here.
file(CONFIGURE OUTPUT "${ET_HOST_GENERATED}/espressif_ops/Esp32KernelVariant.h"
  CONTENT "#pragma once\n#define ESP32_GENERATED_FAST_LAYERNORM 0\n\
#define ESP32_GENERATED_FAST_QUANTIZE 1\n#define ESP32_GENERATED_FAST_DEQUANTIZE 1\n\
#define ESP32_GENERATED_FAST_AVG_POOL2D 1\n#define ESP32_GENERATED_EW_MUL 1\n\
#define ESP32_GENERATED_EW_CONSTANT_PAD 1\n#define ESP32_GENERATED_OPS_ALL 1\n")
add_library(esp32_host_ops OBJECT "${ESPRESSIF_OPS}/espressif_ops.cpp"
  "${ESPRESSIF_OPS}/quant_lut.cpp" "${ESPRESSIF_CHECKS}/quant_lut_check.cpp"
  "${ESPRESSIF_OPS}/kv_affine.cpp" "${ESPRESSIF_CHECKS}/divsqrt_check.cpp"
  "${ESPRESSIF_OPS}/quant_lut16.cpp" "${ESPRESSIF_CHECKS}/quant_lut16_check.cpp"
  "${ESPRESSIF_OPS}/qnorm.cpp" "${ESPRESSIF_CHECKS}/expf_check.cpp"
  "${ESPRESSIF_OPS}/qdq_pool.cpp" "${ESPRESSIF_OPS}/layer_norm2d.cpp"
  "${ESPRESSIF_OPS}/elementwise.cpp")
target_include_directories(esp32_host_ops PRIVATE
  "${ET_HOST_GENERATED}/espressif_ops")
target_link_libraries(esp32_host_ops PUBLIC esp32_host_config)
# The table op's quantize step must give the device's bits, whatever
# ESP32_HOST_FP_CONTRACT brackets the rest with (as on the device).
set_source_files_properties("${ESPRESSIF_OPS}/quant_lut.cpp"
  "${ESPRESSIF_CHECKS}/quant_lut_check.cpp" "${ESPRESSIF_OPS}/kv_affine.cpp"
  "${ESPRESSIF_CHECKS}/divsqrt_check.cpp" "${ESPRESSIF_OPS}/quant_lut16.cpp"
  "${ESPRESSIF_CHECKS}/quant_lut16_check.cpp" "${ESPRESSIF_OPS}/qnorm.cpp"
  "${ESPRESSIF_CHECKS}/expf_check.cpp" "${ESPRESSIF_OPS}/qdq_pool.cpp"
  "${ESPRESSIF_OPS}/layer_norm2d.cpp" "${ESPRESSIF_OPS}/elementwise.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
add_executable(esp32_host_overrides espressif_overrides.cpp)
target_link_libraries(esp32_host_overrides PRIVATE
  executorch_host esp32_host_ops executorch_host_portable
  executorch_host_quantized esp32_reference_kernels esp32_host_config)

set(ET_HOST_BINDINGS "${ET_HOST_GENERATED}/executorch/kernels/portable")
set(ET_HOST_OVERRIDES "${ET_HOST_BINDINGS}/espressif_overrides.txt")
# Upstream only rewrites changed files. A stamp prevents older, unchanged
# headers from retriggering this command (or running it twice in a Make build).
# Every portable op is registered except the standard ones the Espressif op
# library overrides, read from its compiled registration table.
add_custom_command(
  OUTPUT "${ET_HOST_BINDINGS}/codegen.stamp"
  BYPRODUCTS "${ET_HOST_BINDINGS}/RegisterKernelsEverything.cpp"
         "${ET_HOST_BINDINGS}/NativeFunctions.h"
         "${ET_HOST_BINDINGS}/Functions.h" "${ET_HOST_OVERRIDES}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${ET_HOST_BINDINGS}"
  COMMAND esp32_host_overrides "${ET_HOST_OVERRIDES}"
  COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/codegen_portable.py"
          --et-root "${ET_ROOT}" --output "${ET_HOST_BINDINGS}"
          --exclude "${ET_HOST_OVERRIDES}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${ET_HOST_BINDINGS}/codegen.stamp"
  DEPENDS codegen_portable.py esp32_host_overrides
          "${ET_ROOT}/kernels/portable/functions.yaml"
          "${ET_ROOT}/kernels/quantized/quantized.yaml" "${ET_ROOT}/codegen/gen.py"
  COMMENT "Generate portable-kernel registration without the Espressif overrides")
add_custom_target(executorch_host_codegen
  DEPENDS "${ET_HOST_BINDINGS}/codegen.stamp")
add_library(executorch_host_registration OBJECT
  "${ET_HOST_BINDINGS}/RegisterKernelsEverything.cpp"
  "${ET_ROOT}/kernels/prim_ops/register_prim_ops.cpp"
  "${ET_ROOT}/kernels/prim_ops/et_copy_index.cpp"
  "${ET_ROOT}/kernels/prim_ops/et_view.cpp")
add_dependencies(executorch_host_registration executorch_host_codegen)
target_link_libraries(executorch_host_registration PUBLIC esp32_host_config)

add_library(esp32_host_delegate OBJECT "${ESPRESSIF_RUNTIME}/esp32_backend.cpp"
  "${ESPRESSIF_RUNTIME}/esp32_asym.cpp" "${ESPRESSIF_RUNTIME}/esp32_float.cpp"
  "${ESPRESSIF_RUNTIME}/esp32_conv.cpp")
target_link_libraries(esp32_host_delegate PUBLIC esp32_host_config)
# The static int8 convolution path (esp32_conv.h), on as in the firmware's
# default build; --no-conv-fast sends every conv to the general loops.
target_compile_definitions(esp32_host_delegate PRIVATE ESP32_CONV_FAST=1)
# The affine code (esp32_asym.cpp) is -ffp-contract=off whatever ESP32_HOST_FP_CONTRACT
# brackets the rest with, as on the device (esp-idf/espressif_backend/CMakeLists.txt).
set_source_files_properties("${ESPRESSIF_RUNTIME}/esp32_asym.cpp"
  "${ESPRESSIF_RUNTIME}/esp32_conv.cpp" PROPERTIES
  COMPILE_OPTIONS "-ffp-contract=off")
add_executable(esp32_host_runner runner.cpp asym_conformance.cpp
  "${ESPRESSIF_CHECKS}/nonfinite_check.cpp" qdq_sweep.cpp)
set_source_files_properties(asym_conformance.cpp
  "${ESPRESSIF_CHECKS}/nonfinite_check.cpp" qdq_sweep.cpp PROPERTIES
  COMPILE_OPTIONS "-ffp-contract=off")
target_link_libraries(esp32_host_runner PRIVATE
  executorch_host esp32_host_delegate esp32_host_ops executorch_host_registration
  executorch_host_portable executorch_host_quantized esp32_reference_kernels
  esp32_host_config)
# The firmware's bench runner (examples/espressif/esp32s3/main/bench_runner.inc)
# on this runtime: an exported bench image runs through the board's code here
# first (examples/espressif/esp32s3/scripts/export_bench.py).
add_executable(esp32_bench_host bench_host.cpp)
target_include_directories(esp32_bench_host PRIVATE
  "${ET_ROOT}/examples/espressif/esp32s3/main")
target_link_libraries(esp32_bench_host PRIVATE
  executorch_host esp32_host_delegate esp32_host_ops executorch_host_registration
  executorch_host_portable executorch_host_quantized esp32_reference_kernels
  esp32_host_config)
# The runner with the delegate and itself under ThreadSanitizer (the rest of
# the runtime as above), for the ranked placement's two-thread check
# (test_weight_tiers.py, --placement-stress). Built when a program built so
# runs here -- compiling is not enough: some runtimes stop at start on a
# kernel that randomises more address bits than they map for --;
# ESP32_HOST_TSAN=OFF skips it.
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS -fsanitize=thread)
set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=thread)
check_cxx_source_runs("int main() { return 0; }" ESP32_HOST_TSAN_RUNS)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
option(ESP32_HOST_TSAN "Build esp32_host_runner_tsan" ${ESP32_HOST_TSAN_RUNS})
if(ESP32_HOST_TSAN)
  add_library(esp32_host_delegate_tsan OBJECT "${ESPRESSIF_RUNTIME}/esp32_backend.cpp"
    "${ESPRESSIF_RUNTIME}/esp32_asym.cpp" "${ESPRESSIF_RUNTIME}/esp32_float.cpp"
    "${ESPRESSIF_RUNTIME}/esp32_conv.cpp")
  target_compile_definitions(esp32_host_delegate_tsan PRIVATE ESP32_CONV_FAST=1)
  target_compile_options(esp32_host_delegate_tsan PRIVATE -fsanitize=thread)
  target_link_libraries(esp32_host_delegate_tsan PUBLIC esp32_host_config)
  add_executable(esp32_host_runner_tsan runner.cpp asym_conformance.cpp
    "${ESPRESSIF_CHECKS}/nonfinite_check.cpp" qdq_sweep.cpp)
  target_compile_options(esp32_host_runner_tsan PRIVATE -fsanitize=thread)
  target_link_options(esp32_host_runner_tsan PRIVATE -fsanitize=thread)
  target_link_libraries(esp32_host_runner_tsan PRIVATE
    executorch_host esp32_host_delegate_tsan esp32_host_ops
    executorch_host_registration executorch_host_portable executorch_host_quantized
    esp32_reference_kernels esp32_host_config)
endif()
# The conv path's integer epilogue against its float sequence (runtime/
# esp32_conv.cpp, included whole): random and adversarial channels, every
# accumulator in range.
add_executable(esp32_conv_epilogue_check
  "${ESPRESSIF_CHECKS}/conv_epilogue_check.cpp")
target_include_directories(esp32_conv_epilogue_check PRIVATE
  ${ESPRESSIF_INCLUDE_DIRS})
target_compile_definitions(esp32_conv_epilogue_check PRIVATE
  ESP32_HOST_BUILD=1 ESP32_CONV_FAST=1 ESP32_CONV_EPI_CHECK=1)
target_link_libraries(esp32_conv_epilogue_check PRIVATE esp32_reference_kernels
  esp32_host_config)
# After esp32_host_config's -ffp-contract, as the unit's own (above).
set_source_files_properties("${ESPRESSIF_CHECKS}/conv_epilogue_check.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
add_test(NAME esp32_conv_epilogue COMMAND esp32_conv_epilogue_check)
# The conv path's branch-free fp32 quantiser against code_of (runtime/
# esp32_conv.cpp, included whole): the sampled float sweep (--full: every
# float) and the gathers that use it against code_of's, byte for byte.
add_executable(esp32_conv_quant_check
  "${ESPRESSIF_CHECKS}/conv_quant_check.cpp")
target_include_directories(esp32_conv_quant_check PRIVATE
  ${ESPRESSIF_INCLUDE_DIRS})
target_compile_definitions(esp32_conv_quant_check PRIVATE
  ESP32_HOST_BUILD=1 ESP32_CONV_FAST=1)
target_link_libraries(esp32_conv_quant_check PRIVATE esp32_reference_kernels
  esp32_host_config)
set_source_files_properties("${ESPRESSIF_CHECKS}/conv_quant_check.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
add_test(NAME esp32_conv_quant COMMAND esp32_conv_quant_check)
add_test(NAME esp32_bench_host_help COMMAND esp32_bench_host --help)
add_test(NAME esp32_host_runner_help COMMAND esp32_host_runner --help)
add_test(NAME espressif_rmsnorm_domain COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_rmsnorm_domain PROPERTIES PASS_REGULAR_EXPRESSION
  "rmsnorm domain: refused yes, untouched yes, scalar fallback yes")
add_test(NAME espressif_ln2d_refusal COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_ln2d_refusal PROPERTIES PASS_REGULAR_EXPRESSION
  "ln2d refusal: control=accepted fp16=refused rank3=refused")
add_test(NAME esp32_asym_kcache_zero_points COMMAND esp32_host_runner --selftest)
set_tests_properties(esp32_asym_kcache_zero_points PROPERTIES PASS_REGULAR_EXPRESSION
  "asym kcache zp check: control=accepted dtype=refused shape=refused")
add_test(NAME espressif_portable_fastpaths COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_portable_fastpaths PROPERTIES
  PASS_REGULAR_EXPRESSION "fastpath check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH")
# The opt-in fast LayerNorm's rows against the portable kernel's: the host
# does not build its registration (ESP32_FAST_LAYERNORM), only its arithmetic.
add_test(NAME espressif_fast_layer_norm COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_fast_layer_norm PROPERTIES
  PASS_REGULAR_EXPRESSION "fast_layer_norm check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "fast_layer_norm check: MISMATCH")
add_test(NAME espressif_quant_lut COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_quant_lut PROPERTIES
  PASS_REGULAR_EXPRESSION "quant_lut check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH quant_lut")
add_test(NAME espressif_lut16 COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_lut16 PROPERTIES
  PASS_REGULAR_EXPRESSION "lut16 check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH lut16")
# The non-finite quantize rule (runtime/quant_rules.h): every quantiser with
# a portable reference against ExecuTorch's own quantize_per_tensor kernel.
add_test(NAME espressif_nonfinite COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_nonfinite PROPERTIES
  PASS_REGULAR_EXPRESSION "nonfinite check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH nonfinite")
# The backend's weight-tier options through ExecuTorch's option API.
add_test(NAME esp32_weight_tier_options COMMAND esp32_host_runner --selftest)
set_tests_properties(esp32_weight_tier_options PROPERTIES PASS_REGULAR_EXPRESSION
  "weight tier options: roundtrip yes, read-only refused yes, budget below -1 refused yes, rank 2 refused yes, defaults restored yes")
# The quantize/dequantize/avg_pool2d fast paths (ops/qdq_pool.h) against the
# portable kernels: a stride of every float32 and every tie's neighbourhood at
# 21 scales, random scales, every code, 3,000 random pools. `--qdq-sweep full`
# (every float32, minutes on four threads) is the exhaustive run.
add_test(NAME espressif_qdq_sweep COMMAND esp32_host_runner --qdq-sweep quick)
set_tests_properties(espressif_qdq_sweep PROPERTIES
  PASS_REGULAR_EXPRESSION "qdq sweep: quick, 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH")
# The next two tests. The softmax override's exp: the host's transcription of
# newlib's expf (newlib_expf.h) against mirrors.expf_newlib's bits (expf_golden.h);
# the parity firmware runs the same check against the chip's own expf. Then the
# division and square root check: on the host a positive control (its FPU rounds
# correctly), so a mismatch there is a golden or check bug.
add_test(NAME espressif_expf COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_expf PROPERTIES
  PASS_REGULAR_EXPRESSION "expf check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH expf")
add_test(NAME espressif_divsqrt COMMAND esp32_host_runner --selftest)
set_tests_properties(espressif_divsqrt PROPERTIES
  PASS_REGULAR_EXPRESSION "divsqrt check: [0-9]+ case\\(s\\), 0 mismatch\\(es\\)"
  FAIL_REGULAR_EXPRESSION "MISMATCH divsqrt")
