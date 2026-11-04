# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decode and blocked prefill share one cache and one learned-weight image.

Every check is printed and a failed one recorded, so a script run reports
them all and exits 1 if any failed; under pytest, each test_* fails naming its
failed checks (@case). Both tests read one lowering, made once per process.

    python test_glimmer_multimethod.py
"""

import contextlib
import functools
import io
import os
import sys
import tempfile

import torch

from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
    _replace_glimmer_tables_i4,
)
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
    build,
    build_prefill,
)
from executorch.backends.esp32s3.utils.multimethod import (
    combine_shared_methods,
    shared_memory_planning_pass,
)


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def case(test):
    """One test_* case. check() records a failure and goes on, so a script
    run reports every one (main); a case that only did that would pass under
    pytest whatever its checks found. Called, it fails if any check it made
    failed, naming them. main() calls the case itself (__wrapped__)."""

    @functools.wraps(test)
    def run():
        start = len(FAILS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


def lower(model, inputs):
    model, _ = _replace_glimmer_tables_i4(model)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        edge = lower_esp32(
            torch.export.export(model, inputs),
            [Esp32Partitioner(
                external_weights="shared_glimmer",
                format_policy="i8",
                group_weights=True,
                quantize_qk=True,
            )],
        )
        fusions = apply_passes(edge, quiet=True)
    return edge, fusions, capture.getvalue()


def emit(edge, *, shared=False):
    from executorch.exir import ExecutorchBackendConfig

    options = dict(
        delegate_alignment=16,
        constant_tensor_alignment=16,
        external_constants=lambda node: "shared_glimmer",
    )
    if shared:
        options["memory_planning_pass"] = shared_memory_planning_pass()
    return edge.to_executorch(config=ExecutorchBackendConfig(**options))


def tensor_files(program):
    with tempfile.TemporaryDirectory() as directory:
        program.write_tensor_data_to_file(directory)
        return {name: os.path.getsize(os.path.join(directory, name))
                for name in os.listdir(directory)}


@functools.lru_cache(maxsize=None)
def artifacts():
    """The two lowered methods, the combined program and the single-method
    size controls: built once, read by every case."""
    common = dict(
        arm="ple", vocab=256, d_model=64, layers=4, heads=8,
        kv_heads=1, ffn=192, ple_dim=24, seq_len=16, window=8,
        softcap=20.0, cache_context=16, selected_rows=7,
        head_token_ids=list(range(9)),
    )
    _, decode = build(local_ring=True, **common)
    _, prefill = build_prefill(chunk_size=4, **common)
    rows = torch.arange(7, dtype=torch.long)
    decode_edge, decode_fusions, decode_log = lower(
        decode, (torch.tensor([3]), torch.tensor(0), rows))
    prefill_edge, prefill_fusions, prefill_log = lower(
        prefill, (torch.tensor([3, 4, 5, 6]), torch.tensor(0), rows))

    combined_edge = combine_shared_methods({
        "decode": decode_edge,
        "prefill": prefill_edge,
    })
    # to_executorch mutates portions of an ExportedProgram in this wheel. Emit
    # the combined manager first while its method references are pristine; the
    # single-method programs below are only size controls afterward.
    combined = emit(combined_edge, shared=True)
    decode_program = emit(decode_edge)
    prefill_program = emit(prefill_edge)
    cache_bytes = sum(
        value.numel() * value.element_size()
        for name, value in decode.named_buffers() if "cache_" in name)
    return dict(
        decode_fusions=decode_fusions, decode_log=decode_log,
        prefill_fusions=prefill_fusions, prefill_log=prefill_log,
        combined=combined, decode_program=decode_program,
        prefill_program=prefill_program, cache_bytes=cache_bytes,
        plans={plan.name: plan
               for plan in combined.executorch_program.execution_plan})


@case
def test_multimethod_structure():
    built = artifacts()
    plans = built["plans"]
    decode_fusions, prefill_fusions = (
        built["decode_fusions"], built["prefill_fusions"])
    print("multimethod structure:")
    check("one PTE exposes decode and prefill",
          set(plans) == {"decode", "prefill"}, str(sorted(plans)))
    check("both methods retain all architecture fusions",
          (decode_fusions["fused"], decode_fusions["rope_fused"],
           prefill_fusions["fused"], prefill_fusions["rope_fused"])
          == (31, 6, 31, 6))
    check("both selected output heads delegate",
          "select7/9" in built["decode_log"]
          and "select7/9" in built["prefill_log"])

    arenas = {name: list(plan.non_const_buffer_sizes)
              for name, plan in plans.items()}
    check("shared mutable state has its own mem-id-2 arena",
          all(len(sizes) == 3 and sizes[2] == built["cache_bytes"]
              for sizes in arenas.values()), str(arenas))


@case
def test_weight_deduplication():
    built = artifacts()
    combined = built["combined"]
    decode_program, prefill_program = (
        built["decode_program"], built["prefill_program"])
    combined_files = tensor_files(combined)
    decode_files = tensor_files(decode_program)
    prefill_files = tensor_files(prefill_program)

    print("weight deduplication:")
    check("combined program emits exactly one named-data file",
          set(combined_files) == {"shared_glimmer.ptd"}, str(combined_files))
    combined_ptd = sum(combined_files.values())
    single_ptd = max(sum(decode_files.values()), sum(prefill_files.values()))
    check("learned PTD is not duplicated across methods",
          combined_ptd == single_ptd,
          f"combined={combined_ptd:,} max-single={single_ptd:,}")
    check("combined PTE is smaller than two separate manifests",
          len(combined.buffer) < len(decode_program.buffer) + len(prefill_program.buffer),
          f"combined={len(combined.buffer):,} separate="
          f"{len(decode_program.buffer) + len(prefill_program.buffer):,}")


def main():
    for test in (test_multimethod_structure, test_weight_deduplication):
        test.__wrapped__()
    print()
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("nano-Glimmer decode/prefill share KV state and one deduplicated PTD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
