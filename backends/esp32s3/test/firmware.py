# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The example firmware project the parity rows run in.

run_parity.py exports the parity programs into MAIN, derives the operator list
the selective build reads (REQUIRED_OPS), runs GEN_KERNELS, and builds PROJECT
for QEMU (run_hardware.py, for the board).
"""

from pathlib import Path

# backends/esp32s3/test/firmware.py -> the root of the ExecuTorch tree
ET_ROOT = Path(__file__).resolve().parents[3]
PROJECT = ET_ROOT / "examples" / "espressif" / "esp32s3"
MAIN = PROJECT / "main"
COMPONENT = PROJECT / "components" / "executorch"
REQUIRED_OPS = COMPONENT / "required_ops.txt"
GEN_KERNELS = PROJECT / "scripts" / "gen_kernels.py"
