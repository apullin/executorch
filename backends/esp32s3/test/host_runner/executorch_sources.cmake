# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The same portable runtime sources as the IDF component, with the POSIX PAL
# and file loaders. Do not add upstream's top-level CMake: its FlatCC setup
# writes into the ExecuTorch source tree, which may be shared by worktrees.
set(ET_HOST_RUNTIME_SRCS
  runtime/backend/interface.cpp
  runtime/core/device_allocator.cpp
  runtime/core/device_memory_buffer.cpp
  runtime/core/evalue.cpp
  runtime/core/tag.cpp
  runtime/core/tensor_layout.cpp
  runtime/core/exec_aten/util/tensor_shape_to_c_string.cpp
  runtime/core/exec_aten/util/tensor_util_portable.cpp
  runtime/core/portable_type/tensor_impl.cpp
  runtime/executor/method.cpp
  runtime/executor/method_meta.cpp
  runtime/executor/program.cpp
  runtime/executor/program_validation.cpp
  runtime/executor/pte_data_map.cpp
  runtime/executor/tensor_parser_exec_aten.cpp
  runtime/executor/tensor_parser_portable.cpp
  runtime/kernel/operator_registry.cpp
  runtime/platform/abort.cpp
  runtime/platform/log.cpp
  runtime/platform/platform.cpp
  runtime/platform/profiler.cpp
  runtime/platform/runtime.cpp
  runtime/platform/default/posix.cpp
  schema/extended_header.cpp
  extension/flat_tensor/flat_tensor_data_map.cpp
  extension/flat_tensor/serialize/flat_tensor_header.cpp
  extension/data_loader/file_data_loader.cpp
  extension/data_loader/mmap_data_loader.cpp)
list(TRANSFORM ET_HOST_RUNTIME_SRCS PREPEND "${ET_ROOT}/")
