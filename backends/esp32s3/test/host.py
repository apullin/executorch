# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The host runner and the operator tables the host-side tests share.

esp32_host_runner (test/host_runner/) runs lowered programs on the build machine
with the firmware's delegate and kernels. Tests find it through host_runner():
$ESP32_HOST_RUNNER, else the build directory test/host_runner/README.md uses.
"""

import os
from pathlib import Path
import re

# backends/esp32s3/test/host.py -> the root of the ExecuTorch tree
ET_ROOT = Path(__file__).resolve().parents[3]
FUNCTIONS_YAML = ET_ROOT / "kernels" / "portable" / "functions.yaml"


def host_runner():
    """The esp32_host_runner binary (built or not)."""
    return Path(os.environ.get("ESP32_HOST_RUNNER")
                or ET_ROOT / "cmake-out-espressif" / "esp32_host_runner")


HOST_RUNNER = host_runner()


# -------------------------------------------------------------------- lowering


def op_name(target):
    """'mul.Tensor', 'espressif.rmsnorm.default', 'getitem'."""
    t = target._op if type(target).__name__ == "EdgeOpOverload" else target
    if getattr(t, "__name__", None) == "getitem":
        return "getitem"
    s = str(t)
    return s[len("aten."):] if s.startswith("aten.") else s


def op_base(name):
    parts = name.split(".")
    if parts[0] in ("dim_order_ops", "espressif", "quantized_decomposed") and len(parts) > 1:
        parts = parts[1:]
    return parts[0]


# ExecuTorch's static quantize/dequantize kernels, registered beside the
# portable ones on the host (test/host_runner/codegen_portable.py QUANTIZED) and
# the device (scripts/gen_kernels.py, from the program's own op list): the safety
# net for the static pairs a static int8 graph leaves between portable ops, and
# the per-tensor dequantize by a scale and zero point in one-element tensors.
# Never a chooser.
QUANTIZED_KERNELS = (
    "quantized_decomposed::quantize_per_tensor.out",
    "quantized_decomposed::dequantize_per_tensor.out",
    "quantized_decomposed::dequantize_per_tensor.Tensor_out",
    "quantized_decomposed::quantize_per_channel.out",
    "quantized_decomposed::dequantize_per_channel.out",
)


def portable_kernels():
    names = set(QUANTIZED_KERNELS)
    for line in open(FUNCTIONS_YAML):
        m = re.match(r"^- op:\s*(\S+)", line) or re.match(r"^- func:\s*([^(\s]+)", line)
        if m:
            names.add(m.group(1))
    return names


def emitted_ops(program):
    """The program's operator table, split into Espressif kernels, portable
    kernels and ops that no kernel in the tree implements."""
    kernels = portable_kernels()
    out = {"espressif": [], "portable": [], "no_kernel": []}
    for plan in program.executorch_program.execution_plan:
        for op in plan.operators:
            name = f"{op.name}.{op.overload}" if op.overload else op.name
            ns, _, short = name.rpartition("::")
            # Prim ops: ExecuTorch's on the host; on the device the ops library
            # registers _local_scalar_dense (a tensor's value as a Scalar).
            if (ns == "executorch_prim" or short.startswith("sym_")
                    or short == "_local_scalar_dense"):
                continue
            # The ops library's own kernels, ExecuTorch's llama::update_cache
            # (the stock LLM cache write) among them.
            if ns in ("espressif", "llama"):
                out["espressif"].append(name)
                continue
            base = short[:-4] + "_out" if short.endswith(".out") else short
            known = {short, base, f"{ns}::{short}", f"{ns}::{base}"} & kernels
            out["portable" if known else "no_kernel"].append(name)
    return {k: sorted(set(v)) for k, v in out.items()}
