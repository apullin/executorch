# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-model delegate PTD naming; external_weights=True keeps the default name."""

import os
import sys
import tempfile

import torch
import torch.nn as nn


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def main():
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.compile_spec import (
        DEFAULT_EXTERNAL_TAG,
        Esp32CompileSpec,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import lower_esp32

    legacy = Esp32CompileSpec.from_list(
        Esp32CompileSpec(external_weights=True).to_list())
    named = Esp32CompileSpec.from_list(
        Esp32CompileSpec(external_weights="model_a_weights").to_list())
    check("legacy boolean external spec retains its filename",
          legacy.external_tag == DEFAULT_EXTERNAL_TAG, str(legacy))
    check("named external spec round-trips through bytes",
          named.external_tag == "model_a_weights", str(named))
    try:
        Esp32CompileSpec(external_weights="../collision")
    except RuntimeError:
        rejected = True
    else:
        rejected = False
    check("unsafe/path-like PTD tags are rejected", rejected)

    torch.manual_seed(0)
    model = nn.Linear(32, 24, bias=False).eval()
    model.requires_grad_(False)
    exported = torch.export.export(model, (torch.randn(1, 32),))
    edge = lower_esp32(
        exported, [Esp32Partitioner(external_weights="probe_weights")])
    program = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16,
        external_constants=lambda node: "probe_weights",
    ))
    with tempfile.TemporaryDirectory() as tmp:
        program.write_tensor_data_to_file(tmp)
        files = sorted(os.listdir(tmp))
    check("delegate and portable constants share the requested PTD",
          files == ["probe_weights.ptd"], str(files))

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("external delegate weights have collision-free per-model PTD names")
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
