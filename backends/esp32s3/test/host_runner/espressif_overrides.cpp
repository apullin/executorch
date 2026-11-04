/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Writes the standard operators espressif::register_ops() registers, one per
// line. The host's portable registration omits exactly these, as
// gen_kernels.py's OVERRIDES does for the firmware: ExecuTorch aborts on a
// second registration of the same operator, and the list cannot drift from the
// compiled table because it is read from it.
#include "espressif_ops.h"

#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s overrides.txt\n", argv[0]);
    return 2;
  }
  executorch::runtime::runtime_init();
  if (espressif::register_ops() != executorch::runtime::Error::Ok) {
    std::fprintf(stderr, "espressif::register_ops failed\n");
    return 1;
  }
  std::FILE* out = std::fopen(argv[1], "w");
  if (out == nullptr) {
    std::perror(argv[1]);
    return 1;
  }
  // The standard namespaces the portable registration also covers: aten's
  // operators and ExecuTorch's quantize/dequantize kernels.
  for (const auto& kernel : executorch::runtime::get_registered_kernels()) {
    if (std::strncmp(kernel.name_, "aten::", 6) == 0 ||
        std::strncmp(kernel.name_, "quantized_decomposed::", 22) == 0) {
      std::fprintf(out, "%s\n", kernel.name_);
    }
  }
  return std::fclose(out) == 0 ? 0 : 1;
}
