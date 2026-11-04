# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export with program-data separation: a small .pte plus a separate .ptd of weights.

Why this matters beyond tidiness: constants are serialised INTO the .pte by default,
which is why mlp.pte (52 KB) is larger than the structurally bigger convnet.pte
(20 KB). The PLE model's table is 12.58 MB, so inlining is not an option -- it has to
live in flash and be mapped (a data partition read through esp_partition_mmap), and
ExecuTorch's .ptd is the supported form of that.

    python backends/esp32s3/test/parity/export_ptd.py
"""

import os
import torch

from executorch.backends.esp32s3._passes import apply_passes
from executorch.backends.esp32s3.test.models import SPECS, checksum, ramp_input
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
NAME = "block"          # the biggest of the three, so the split is obvious


def main():
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    factory, shape = SPECS[NAME]
    torch.manual_seed(0)
    model = factory().eval()
    x = ramp_input(shape)
    with torch.no_grad():
        eager = checksum(model(x))

    ep = torch.export.export(model, (x,))
    lowered = to_edge_transform_and_lower(ep)
    apply_passes(lowered)

    # A callable returning a filename tags every constant for that external file.
    et = lowered.to_executorch(
        config=ExecutorchBackendConfig(
            external_constants=lambda node: "block_weights",
        )
    )

    pte = os.path.join(MAIN, "block_split.pte")
    with open(pte, "wb") as f:
        f.write(et.buffer)
    et.write_tensor_data_to_file(MAIN)

    ptd = os.path.join(MAIN, "block_weights.ptd")
    inline = os.path.join(MAIN, "block.pte")
    print(f"inline  block.pte        {os.path.getsize(inline):>9,} B  (weights inside)")
    print(f"split   block_split.pte  {os.path.getsize(pte):>9,} B  (program only)")
    print(f"        block_weights.ptd {os.path.getsize(ptd):>8,} B  (weights)")
    print(f"        sum               {os.path.getsize(pte)+os.path.getsize(ptd):>8,} B")
    print(f"\neager checksum: {eager:.6f}   (device must match this)")


if __name__ == "__main__":
    main()
