# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The declared inline-fp32 backend opcode must be AOT-reachable and exact.

This is a diagnostic/control storage policy, not a deployment recommendation:
the ESP32-S3 has integer SIMD and no fp32 vector unit.  Its value is making the
runtime's opcode-1 implementation a tested part of the ABI.
"""

import contextlib
import io
import struct
import sys

import numpy as np
import torch
from executorch.exir import to_edge
from torch import nn

from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_BYTES,
    BLOB_HEADER_FORMAT,
    OP_ADDMM,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
    MAGIC,
    VERSION,
)
from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3._passes import lower_esp32

FAILS = []


class MixedStorageKinds(nn.Module):
    """Only the terminal Linear has an fp32 delegate implementation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.conv = nn.Conv1d(8, 8, 1)
        self.head = nn.Linear(8, 5, bias=False)

    def forward(self, indices):
        x = self.embedding(indices)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x)


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def main():
    torch.manual_seed(0)
    model = nn.Linear(5, 7, bias=False).eval()
    value = torch.randn(2, 5)
    edge = to_edge(torch.export.export(model, (value,)))
    spec = Esp32CompileSpec(format_policy="fp32")
    with contextlib.redirect_stdout(io.StringIO()):
        result = Esp32Backend.preprocess(
            edge.exported_program(), spec.to_list())

    blob = result.processed_bytes
    magic, version, n_ops, ext_off = struct.unpack_from(
        BLOB_HEADER_FORMAT, blob)
    desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES)))
    stored = np.frombuffer(
        blob, dtype="<f4", count=5 * 7, offset=desc["wt_off"]
    ).reshape(5, 7)
    expected = model.weight.detach().numpy().T

    check("compile spec round-trips fp32",
          Esp32CompileSpec.from_list(spec.to_list()) == spec)
    check("blob header is current and extension-free",
          (magic, version, n_ops, ext_off) == (MAGIC, VERSION, 1, 0),
          f"{magic:#x} v{version} ops={n_ops} ext={ext_off}")
    check("descriptor selects opcode 1 only",
          desc["opcode"] == OP_ADDMM and desc["flags"] == 0,
          str(desc))
    check("geometry is unpadded fp32",
          (desc["m"], desc["k"], desc["n"], desc["k_pad"])
          == (2, 5, 7, 5))
    check("bias-free and scale-free sentinels are zero",
          desc["bias_off"] == 0 and desc["scale_off"] == 0)
    check("stored [K,N] weights are bit-exact",
          np.array_equal(stored, expected))
    check("fp32 emits no external named data",
          not result.data_store_output.external_data)

    check("fp32 combines with external_weights (one PTD entry a weight)",
          Esp32CompileSpec(format_policy="fp32",
                           external_weights=True).external_weights)
    for option in ("static_act", "min_latency",
                   "group_weights", "stage_tied_head",
                   "split_tied_head"):
        kwargs = {option: True}
        try:
            Esp32CompileSpec(format_policy="fp32", **kwargs)
        except RuntimeError:
            rejected = True
        else:
            rejected = False
        check(f"fp32 refuses incompatible {option}", rejected)

    # A storage policy must narrow admission before partitioning: under fp32
    # an embedding stays portable rather than reaching preprocess as a
    # descriptor it cannot lower. A convolution runs in the fp32 convolution
    # (opcode 1 under FLAG_CONV). Keep an eligible Linear in the same graph so
    # a blanket disable cannot satisfy the check.
    mixed = MixedStorageKinds().eval()
    indices = torch.randint(0, 32, (1, 6))
    part = Esp32Partitioner(format_policy="fp32")
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            lowered = lower_esp32(
                torch.export.export(mixed, (indices,)), [part])
    except RuntimeError as exc:
        lowered = None
        lower_error = str(exc)
    else:
        lower_error = ""
    check("mixed fp32 graph lowers without a preprocess abort",
          lowered is not None, lower_error)
    if lowered is not None:
        gm = lowered.exported_program().graph_module
        targets = [str(node.target) for node in gm.graph.nodes
                   if node.op == "call_function"]
        delegates = [target for target in targets
                     if "executorch_call_delegate" in target]
        check("the fp32 conv and Linear delegate", len(delegates) == 2,
              f"{len(delegates)} delegate call(s)")
        check("embedding stays portable under fp32 policy",
              any("aten.embedding" in target for target in targets))
        check("convolution delegates under fp32 policy",
              not any("aten.convolution" in target for target in targets))
        reasons = list(part.reporter._rejected_nodes.values())
        check("fp32 embedding refusal is reported by policy",
              any("fp32" in reason and "embedding" in reason
                  for reason in reasons))
        kinds = [l for l in log.getvalue().splitlines() if " B blob" in l]
        check("the conv is opcode 1 under FLAG_CONV",
              any("addmm_f32[" in l and " conv1x1" in l for l in kinds),
              "; ".join(kinds))

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("inline fp32 is AOT-reachable with exact [K,N] storage")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() records every failed
    check and goes on; this fails if any did, naming them."""
    start = len(FAILS)
    rc = main()
    failed = FAILS[start:]
    assert not failed and rc in (None, 0), f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(main())
