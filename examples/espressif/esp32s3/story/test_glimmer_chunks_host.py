# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prefill chunks past the first and decode past the ring's wrap, lowered and run.

test_glimmer_step_sequence.py holds the eager wrappers to the sequence model;
this runs what they export through the delegate's own C++ (the host runner):
one two-method program per K cache kind, shared KV and in-place writes as the
story exports it, driven through prefill chunks at positions 0, 4 and 8
(window 4, so the second and third chunks read and then overwrite ring slots
an earlier chunk wrote) and then decode at positions 12-15 (every step
wrapping the ring). The int8 K cache only re-encodes what the fp32 one holds
(kcache_i8.py): both programs' logits must be equal bit for bit at every
step, and each step's checksum within 1e-5 of the device-arithmetic
reference with aliased caches. No fixture covers this otherwise: the parity's
kc8 rows run one prefill chunk at position 0, and test_kcache_i8.py compares
two eager references.

Needs the host runner (backends/esp32s3/test/host_runner/README.md); skipped
without it.

    python test_glimmer_chunks_host.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.host import host_runner
from executorch.backends.esp32s3.test.parity.export_glimmer_decode import (
    quantize_reference_modules,
    token_at,
)
from executorch.backends.esp32s3.test.parity.export_glimmer_prefill import checksum
from executorch.backends.esp32s3.utils.multimethod import (
    combine_shared_methods,
    shared_memory_planning_pass,
)
from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
    build,
    build_prefill,
)

COMMON = dict(arm="ple", vocab=256, d_model=64, layers=4, heads=8, kv_heads=1,
              ffn=192, ple_dim=24, seq_len=16, window=4, softcap=20.0)
CHUNK = 4
TOKENS = [token_at(i, COMMON["vocab"]) for i in range(COMMON["seq_len"])]
STEPS = ([("prefill", (torch.tensor(TOKENS[s:s + CHUNK]), torch.tensor(s)))
          for s in (0, 4, 8)]
         + [("decode", (torch.tensor([TOKENS[p]]), torch.tensor(p)))
            for p in range(12, 16)])
TOL = 1e-5


def program(int8_kcache):
    from executorch.exir import ExecutorchBackendConfig

    _, decode = build(local_ring=True, int8_kcache=int8_kcache, **COMMON)
    _, prefill = build_prefill(chunk_size=CHUNK, int8_kcache=int8_kcache,
                               **COMMON)
    edges = {}
    for name, model, inputs, tails in (
            ("decode", decode, STEPS[3][1], COMMON["layers"]),
            ("prefill", prefill, STEPS[0][1], 0)):
        edge = lower_esp32(torch.export.export(model, inputs),
                           [Esp32Partitioner(format_policy="i8",
                                             quantize_qk=True)])
        apply_passes(edge, quiet=True, attn_tail_expect=tails)
        edges[name] = edge
    return combine_shared_methods(edges).to_executorch(
        config=ExecutorchBackendConfig(
            delegate_alignment=16,
            memory_planning_pass=shared_memory_planning_pass(),
            run_reinplace_pass=True))


def run_steps(runner, prog, work):
    """Every step through one runner process (the KV state carries over);
    the logits per step and the q@kT counter lines."""
    (work / "m.pte").write_bytes(prog.buffer)
    lines, outputs = [], []
    for index, (method, args) in enumerate(STEPS):
        paths = []
        for arg_index, value in enumerate(args):
            path = work / f"s{index}_in{arg_index}.bin"
            value.numpy().tofile(path)
            paths.append(str(path))
        outputs.append(work / f"s{index}_out.bin")
        lines.append(f"{method} {' '.join(paths)} -> {outputs[-1]}")
    (work / "steps.txt").write_text("\n".join(lines) + "\n")
    done = subprocess.run(
        [str(runner), "--pte", str(work / "m.pte"),
         "--script", str(work / "steps.txt")],
        capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    counters = [line for line in done.stdout.splitlines()
                if "bmmkc8" in line or "bmmqk" in line]
    return [np.fromfile(path, dtype=np.float32) for path in outputs], counters


def reference(int8_kcache):
    _, ref_prefill = build_prefill(chunk_size=CHUNK, reference_dynamic_qk=True,
                                   int8_kcache=int8_kcache, **COMMON)
    _, ref_decode = build(reference_dynamic_qk=True, local_ring=True,
                          int8_kcache=int8_kcache, **COMMON)
    quantize_reference_modules(ref_prefill, "i8")
    quantize_reference_modules(ref_decode, "i8")
    for name, buffer in ref_prefill.named_buffers():
        if "cache_" in name:
            setattr(ref_decode, name, buffer)
    with torch.no_grad():
        return [checksum((ref_prefill if method == "prefill" else ref_decode)(*args))
                for method, args in STEPS]


def test_chunks_and_wrapping_decode_on_host_runner():
    runner = host_runner()
    if not runner.is_file():
        pytest.skip(f"no host runner at {runner}")
    logits = {}
    for int8_kcache in (False, True):
        with tempfile.TemporaryDirectory() as work:
            got, counters = run_steps(runner, program(int8_kcache), Path(work))
        want = reference(int8_kcache)
        rels = [abs(checksum(torch.from_numpy(row)) - value) / max(abs(value), 1e-9)
                for row, value in zip(got, want)]
        print(f"int8_kcache={int8_kcache}: {counters}; checksum rel per step "
              f"{[f'{rel:.1e}' for rel in rels]}")
        # The int8 cache's q@kT ops read the cache's rows; the fp32 cache's
        # re-quantise them.
        kind = "bmmkc8: 28" if int8_kcache else "bmmqk: 28"
        assert any(kind in line for line in counters), counters
        assert max(rels) < TOL, (int8_kcache, rels)
        logits[int8_kcache] = got
    same = [np.array_equal(a.view(np.uint32), b.view(np.uint32))
            for a, b in zip(logits[False], logits[True])]
    assert all(same), f"int8 and fp32 K cache programs differ at steps {same}"


if __name__ == "__main__":
    test_chunks_and_wrapping_decode_on_host_runner()
    print("lowered prefill chunks and wrapping decode hold on the host runner")
    sys.exit(0)
