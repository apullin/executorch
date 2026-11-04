# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ExecuTorch portable runtime as an ESP-IDF component.
#
# Three build-plumbing facts:
#   1. ET's includes are <executorch/...>, so the include root is the directory
#      above a tree named executorch (or a link to the tree under that name in
#      the build directory: a git worktree has a name of its own).
#   2. The vendored c10 shim under runtime/core/portable_type/c10 must be on the
#      include path, or every file fails on <c10/util/irange.h>.
#   3. C10_USING_CUSTOM_GENERATED_MACROS tells the shim not to look for
#      torch/headeronly/macros/cmake_macros.h, which only exists in an installed
#      torch. The header documents this contract itself.
#
# generated/ holds the program and FlatTensor schema headers, which
# scripts/gen_kernels.py writes ahead of the build with the flatc the
# ExecuTorch wheel ships (24.3.25): a newer flatc's headers fail the bundled
# library's "Non-compatible flatbuffers version" static_assert.

# The runtime and kernel sources: the ExecuTorch tree this example sits in,
# unless ET_ROOT (cache or environment) names another source tree.
if(DEFINED ENV{ET_ROOT})
    set(_et_default "$ENV{ET_ROOT}")
else()
    get_filename_component(_et_default "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
endif()
set(ET_ROOT "${_et_default}" CACHE PATH "ExecuTorch source tree")
# Submodule headers: ET_ROOT's own FlatBuffers by default. A tree whose
# submodules are not initialised (a git worktree, say) can name another
# checkout's copy (cache or environment variable).
if(DEFINED ENV{ESPRESSIF_FLATBUFFERS_INCLUDE})
    set(_flatbuffers_default "$ENV{ESPRESSIF_FLATBUFFERS_INCLUDE}")
else()
    set(_flatbuffers_default "${ET_ROOT}/third-party/flatbuffers/include")
endif()
set(ESPRESSIF_FLATBUFFERS_INCLUDE "${_flatbuffers_default}"
    CACHE PATH "FlatBuffers headers (the third-party/flatbuffers submodule)")

set(ET_RUNTIME_SRCS
    ${ET_ROOT}/runtime/backend/interface.cpp
    ${ET_ROOT}/runtime/core/device_allocator.cpp
    ${ET_ROOT}/runtime/core/device_memory_buffer.cpp
    ${ET_ROOT}/runtime/core/evalue.cpp
    ${ET_ROOT}/runtime/core/tag.cpp
    ${ET_ROOT}/runtime/core/tensor_layout.cpp
    ${ET_ROOT}/runtime/core/exec_aten/util/tensor_shape_to_c_string.cpp
    ${ET_ROOT}/runtime/core/exec_aten/util/tensor_util_portable.cpp
    ${ET_ROOT}/runtime/core/portable_type/tensor_impl.cpp
    ${ET_ROOT}/runtime/executor/method.cpp
    ${ET_ROOT}/runtime/executor/method_meta.cpp
    ${ET_ROOT}/runtime/executor/program.cpp
    ${ET_ROOT}/runtime/executor/program_validation.cpp
    ${ET_ROOT}/runtime/executor/pte_data_map.cpp
    ${ET_ROOT}/runtime/executor/tensor_parser_exec_aten.cpp
    ${ET_ROOT}/runtime/executor/tensor_parser_portable.cpp
    ${ET_ROOT}/runtime/kernel/operator_registry.cpp
    ${ET_ROOT}/runtime/platform/abort.cpp
    ${ET_ROOT}/runtime/platform/log.cpp
    ${ET_ROOT}/runtime/platform/platform.cpp
    ${ET_ROOT}/runtime/platform/profiler.cpp
    ${ET_ROOT}/runtime/platform/runtime.cpp
    ${ET_ROOT}/runtime/platform/default/minimal.cpp
    ${ET_ROOT}/schema/extended_header.cpp
    # Program-data separation: a .pte carrying only the program, plus a .ptd of
    # weights supplied at load time through a NamedDataMap. Required for the PLE
    # model, whose 12.58 MB table cannot be serialised into the program.
    ${ET_ROOT}/extension/flat_tensor/flat_tensor_data_map.cpp
    ${ET_ROOT}/extension/flat_tensor/serialize/flat_tensor_header.cpp)

# Kernel registration comes from ExecuTorch's own codegen: gen_kernels.py runs
# codegen.tools.gen_oplist + codegen.gen and drops RegisterKernelsEverything.cpp
# here. It is generated ahead of the build rather than during it, so building
# firmware needs no torch/torchgen environment -- see the docstring of
# gen_kernels.py for the trade-off.
#
# The SOURCE list beside it is ours. ET's build compiles the whole portable
# kernel library and lets --gc-sections drop what is unused; compiling ~200 files for
# xtensa to throw most away is not free, so the op -> .cpp mapping stays explicit.
# The lean PLE build (ET_PLE_ONLY) swaps in a selective registration holding
# only the ple program's ops -- the difference between a 1.77 MB app and one that
# fits beside 15 MB of model in a real 16 MB layout.
# ESP-IDF evaluates component requirements in a helper CMake process that does
# not propagate project-specific cache variables such as ET_PLE_ONLY, so there
# a lean-only tree would select the full-runtime file, which does not exist.
# The generated files decide instead, since they are mutually diagnostic: a
# lean-only invocation creates the _ple variant first, while run_parity
# creates the ordinary variant first.
if(ET_PLE_ONLY OR
   (NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/generated_kernel_srcs.cmake" AND
    EXISTS "${CMAKE_CURRENT_LIST_DIR}/generated_kernel_srcs_ple.cmake"))
    set(ET_GEN_SUFFIX "_ple")
else()
    set(ET_GEN_SUFFIX "")
endif()
include(${CMAKE_CURRENT_LIST_DIR}/generated_kernel_srcs${ET_GEN_SUFFIX}.cmake)

set(ET_KERNEL_REGISTRATION
    ${CMAKE_CURRENT_LIST_DIR}/generated_kernels${ET_GEN_SUFFIX}/RegisterKernelsEverything.cpp)

file(GLOB ET_KERNEL_UTIL_SRCS ${ET_ROOT}/kernels/portable/cpu/util/*.cpp)

idf_component_register(
    SRCS ${ET_RUNTIME_SRCS} ${ET_GENERATED_KERNEL_SRCS} ${ET_KERNEL_UTIL_SRCS}
         ${ET_KERNEL_REGISTRATION}
    INCLUDE_DIRS "generated" "generated_kernels${ET_GEN_SUFFIX}"
    LDFRAGMENTS "linker.lf")

get_filename_component(_et_name "${ET_ROOT}" NAME)
if(_et_name STREQUAL "executorch")
    get_filename_component(ET_INCLUDE_ROOT "${ET_ROOT}" DIRECTORY)
else()
    set(ET_INCLUDE_ROOT "${CMAKE_BINARY_DIR}/executorch_include")
    file(MAKE_DIRECTORY "${ET_INCLUDE_ROOT}")
    file(CREATE_LINK "${ET_ROOT}" "${ET_INCLUDE_ROOT}/executorch" SYMBOLIC)
endif()
if(NOT EXISTS "${ESPRESSIF_FLATBUFFERS_INCLUDE}/flatbuffers/flatbuffers.h")
    message(FATAL_ERROR "No FlatBuffers headers in ${ESPRESSIF_FLATBUFFERS_INCLUDE}: "
                        "initialise third-party/flatbuffers or set "
                        "ESPRESSIF_FLATBUFFERS_INCLUDE")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/generated/executorch/schema/program_generated.h")
    message(FATAL_ERROR "No schema headers in ${CMAKE_CURRENT_LIST_DIR}/generated: "
                        "run scripts/gen_kernels.py first")
endif()

target_include_directories(${COMPONENT_LIB} PUBLIC
    ${ET_INCLUDE_ROOT}
    ${ESPRESSIF_FLATBUFFERS_INCLUDE}
    ${ET_ROOT}/runtime/core/portable_type/c10)

target_compile_definitions(${COMPONENT_LIB} PUBLIC C10_USING_CUSTOM_GENERATED_MACROS)

# Optional operator-attribution build. EventTracer hooks are compiled out of
# ExecuTorch unless this definition reaches the runtime and generated unboxing
# code themselves; defining it only on the application component produces a
# tracer object that receives no events.
if(ESP32_OP_TIMING)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ET_EVENT_TRACER_ENABLED=1)
endif()

# ET is not warning-clean under IDF's -Werror=all, and is size-sensitive here.
if(CONFIG_ET_PORTABLE_O2)
    target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -fno-rtti -w)
else()
    target_compile_options(${COMPONENT_LIB} PRIVATE -Os -fno-rtti -w)
endif()
