# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch backend for Espressif microcontrollers (ESP32-S3)."""

from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec
from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

__all__ = [
    "Esp32Partitioner",
    "Esp32Backend",
    "Esp32CompileSpec",
    "Esp32Quantizer",
]
