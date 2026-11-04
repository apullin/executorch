# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""External PTD convolution must retain its runtime gather geometry."""

import contextlib
import io
import struct
import sys

import torch
import torch.nn as nn

from executorch.backends.esp32s3.operator_support.conv import (
    CONV_GEOM_BYTES,
    EXT_CONV3D,
    ConvGeom,
)
from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_BYTES,
    BLOB_HEADER_FORMAT,
    EXT_COUNT_FORMAT,
    EXT_RECORD_HEADER_BYTES,
    EXT_RECORD_HEADER_FORMAT,
    FLAG_CONV,
    FLAG_EXTERNAL_WEIGHTS,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
)
from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


class DepthwiseStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(8, 8, 3, groups=8, bias=False)

    def forward(self, value):
        return self.conv(value)


def main():
    from executorch.exir import to_edge

    torch.manual_seed(0)
    model = DepthwiseStep().eval()
    value = torch.randn(1, 8, 3)
    edge = to_edge(torch.export.export(model, (value,)))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = Esp32Backend.preprocess(
            edge.exported_program(),
            Esp32CompileSpec(
                external_weights="external_conv_test").to_list())

    blob = result.processed_bytes
    _magic, _version, op_count, extension_offset = struct.unpack_from(
        BLOB_HEADER_FORMAT, blob)
    descriptor = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES)))
    extension_count, = struct.unpack_from(
        EXT_COUNT_FORMAT, blob, extension_offset)
    record_at = extension_offset + struct.calcsize(EXT_COUNT_FORMAT)
    op_index, extension_type, length = struct.unpack_from(
        EXT_RECORD_HEADER_FORMAT, blob, record_at)
    payload_at = record_at + EXT_RECORD_HEADER_BYTES
    geometry = ConvGeom.unpack(blob[payload_at:payload_at + length])

    check("one external depthwise convolution is serialized", op_count == 1)
    check("descriptor carries both external-weight and convolution flags",
          descriptor["flags"] & (FLAG_EXTERNAL_WEIGHTS | FLAG_CONV)
          == FLAG_EXTERNAL_WEIGHTS | FLAG_CONV,
          f"flags=0x{descriptor['flags']:x}")
    check("required convolution extension remains attached",
          (extension_count, op_index, extension_type, length)
          == (1, 0, EXT_CONV3D, CONV_GEOM_BYTES),
          f"ext={extension_count}/{op_index}/{extension_type}/{length}")
    check("external Conv1d geometry is exact",
          (geometry.kh, geometry.kw, geometry.c_in, geometry.c_out,
           geometry.groups, geometry.m, geometry.k)
          == (1, 3, 1, 8, 8, 1, 3), str(geometry))
    named = result.data_store_output.external_data["external_conv_test"]
    check("weight and scale payloads live in PTD", len(named) == 2,
          f"named buffers={len(named)}")
    check("emit report still identifies the convolution",
          " conv1x3[1->8 @1x1]" in output.getvalue())

    print()
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("external convolution retains FLAG_CONV and EXT_CONV3D")
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
