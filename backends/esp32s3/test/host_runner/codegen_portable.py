# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run upstream's portable-op codegen, writing only into the host build dir.

Besides every portable op, ExecuTorch's own static quantize/dequantize kernels
(QUANTIZED, from kernels/quantized/quantized.yaml), as gen_kernels.py
registers them on the device for a program whose op list names them: the
safety net for the static int8 pairs a static graph leaves between two
portable ops (_passes/static_qdq.py), and the per-tensor dequantize by a scale
and zero point in one-element tensors. Never a chooser: ExecuTorch's
choose_qparams agrees with none of PyTorch's. inf and NaN inputs are out of
these kernels' contract.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


# The static pair kernels the host registers (and nothing else of
# quantized.yaml: no chooser, no per-token or embedding op). Tensor_out is the
# per-tensor dequantize with its scale and zero point in one-element tensors.
# ExecuTorch's kernel takes them as float64 and int64, as PyTorch's choosers
# return them, and aborts the process on any other dtype (op_dequantize.cpp,
# ET_CHECK), which torch's own .tensor overload accepts: the export must not
# leave it a float32 scale or an int32 zero point.
QUANTIZED = (
    "quantized_decomposed::quantize_per_tensor.out",
    "quantized_decomposed::dequantize_per_tensor.out",
    "quantized_decomposed::dequantize_per_tensor.Tensor_out",
    "quantized_decomposed::quantize_per_channel.out",
    "quantized_decomposed::dequantize_per_channel.out",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--et-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path,
                        help="operators registered by another library, one per line")
    args = parser.parse_args()

    # No bytecode in the source tree, as in gen_kernels.py.
    sys.dont_write_bytecode = True
    import torchgen
    import yaml

    torchgen_root = Path(torchgen.__file__).parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = output / "selected_operators.yaml"
    functions = args.et_root / "kernels/portable/functions.yaml"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, "-m", "codegen.tools.gen_oplist",
         f"--output_path={selected}", f"--ops_schema_yaml_path={functions}"],
        cwd=args.et_root, env=environment, check=True,
    )
    selection = yaml.safe_load(selected.read_text())
    excluded = set(args.exclude.read_text().split()) if args.exclude else set()
    # The firmware's gen_kernels.py drops its overrides from --root_ops; the
    # host selects every portable op, so drop them from that selection. A name
    # portable does not have needs no exclusion. The quantize/dequantize
    # kernels the op library overrides (qdq_pool.h) are not added below.
    for name in excluded:
        selection["operators"].pop(name, None)
        selection["et_kernel_metadata"].pop(name, None)
    template = next(iter(selection["operators"].values()))
    for name in QUANTIZED:
        if name in excluded:
            continue
        selection["operators"][name] = dict(template)
        selection["et_kernel_metadata"][name] = ["default"]
    selected.write_text(yaml.safe_dump(selection, default_flow_style=False))
    quantized = args.et_root / "kernels/quantized/quantized.yaml"
    subprocess.run(
        [sys.executable, "-m", "codegen.gen",
         f"--source-path={args.et_root / 'codegen'}",
         f"--install-dir={output}",
         f"--tags-path={torchgen_root / 'packaged/ATen/native/tags.yaml'}",
         f"--aten-yaml-path={torchgen_root / 'packaged/ATen/native/native_functions.yaml'}",
         f"--op-selection-yaml-path={selected}",
         f"--functions-yaml-path={functions}",
         f"--custom-ops-yaml-path={quantized}", "--manual-registration"],
        cwd=args.et_root, env=environment, check=True,
    )


if __name__ == "__main__":
    main()
