# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bias-free projections must not carry vocabulary-sized zeroes in the PTE."""

import contextlib
import io
import struct

import torch
from torch import nn

from executorch.exir import to_edge

from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_BYTES,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
    MAGIC,
    VERSION,
)
from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec


class Projection(nn.Module):
    def __init__(self, weight, biased):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(torch.zeros(weight.shape[0])) if biased else None

    def forward(self, value):
        weight_kn = self.weight.t()
        if self.bias is None:
            return torch.mm(value, weight_kn)
        return torch.addmm(self.bias, value, weight_kn)


def blob(weight, biased):
    model = Projection(weight, biased).eval()
    edge = to_edge(torch.export.export(model, (torch.randn(2, weight.shape[1]),)))
    with contextlib.redirect_stdout(io.StringIO()):
        result = Esp32Backend.preprocess(
            edge.exported_program(),
            Esp32CompileSpec(external_weights="bias_test").to_list(),
        )
    return result


def main():
    torch.manual_seed(0)
    outputs, inputs = 4096, 16
    weight = torch.randn(outputs, inputs)
    plain = blob(weight, False)
    biased = blob(weight, True)

    magic, version = struct.unpack_from("<II", plain.processed_bytes)
    plain_desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, plain.processed_bytes, BLOB_HEADER_BYTES)))
    biased_desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, biased.processed_bytes, BLOB_HEADER_BYTES)))
    plain_bias = plain_desc["bias_off"]
    biased_bias = biased_desc["bias_off"]
    saved = len(biased.processed_bytes) - len(plain.processed_bytes)
    plain_named = plain.data_store_output.external_data["bias_test"]
    biased_named = biased.data_store_output.external_data["bias_test"]

    checks = {
        "blob version is current": (magic, version) == (MAGIC, VERSION),
        "bias-free descriptor uses offset-zero sentinel": plain_bias == 0,
        "real bias remains embedded": biased_bias != 0,
        "removed bytes equal the complete fp32 bias": saved == outputs * 4,
        "weight/scale PTD entries are unchanged": set(plain_named) == set(biased_named),
    }
    failed = []
    for name, passed in checks.items():
        print(f"  {'ok ' if passed else 'FAIL'} {name}")
        if not passed:
            failed.append(name)
    print(f"  saved {saved:,} PTE bytes at N={outputs:,}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("OK: bias-free projections carry no materialized zero bias.")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() asserts each check
    (or returns non-zero), and this fails with it."""
    assert main() in (None, 0)


if __name__ == "__main__":
    raise SystemExit(main())
