# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioner's signature cache changes how fast a program lowers, never what it lowers to.

Esp32Partitioner.partition() answers _const_tensor's placeholder lookups from
one name -> source map (const_weights._placeholder_index), and is held to the
same standard as any lowering-speed change: the emitted program is
byte-identical with and without it. Claims:

  1. _const_tensor answers every placeholder (parameter, persistent and
     non-persistent buffer, lifted constant, user input) the same from the
     map as from torch's lookups, the map is active inside partition()'s
     scope and only there, and it follows a replaced signature or spec list,
  2. lowering test/models.MLP builds the map once, in partition(), and not
     at all with the map turned off, and the .pte bytes agree either way,
  3. to_backend's graph and the .pte/.ptd bytes agree with and without the
     map on a 2-layer nanoLlama3.2 with a dozen partitions, and only the run
     with it builds the map. Only this case needs the nano zoo and
     transformers; without them it is skipped.

Every check is printed and a failed one recorded, so a run reports them all:
as a script, the file exits 1 if any failed; under pytest, each test_* fails
naming its failed checks, and the nanoLlama3.2 one is skipped without the
nano zoo and transformers (@case).

    python test_lowering_compat.py
"""

import contextlib
import functools
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

FAILS = []
SKIPS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def case(test):
    """One test_* case. check() records a failure and goes on, so a script
    run reports every one (main); a case that only did that would pass
    under pytest whatever its checks found. Called, it fails if any check
    it made failed, naming them, and is skipped if it recorded a skip.
    main() calls the case itself (__wrapped__), so the script still runs
    every case."""

    @functools.wraps(test)
    def run():
        start, skipped = len(FAILS), len(SKIPS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"
        if SKIPS[skipped:]:
            import pytest

            pytest.skip(", ".join(SKIPS[skipped:]))

    return run


def graph_key(gm):
    """The graph as (op, target, args with node references as positions)."""
    rows = []
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        index = {}
        for i, n in enumerate(module.graph.nodes):
            index[n] = i

            def m(a):
                return ("node", index[a]) if isinstance(a, torch.fx.Node) else a
            rows.append((n.op, str(n.target), torch.fx.node.map_aggregate(n.args, m),
                         torch.fx.node.map_aggregate(dict(n.kwargs), m)))
    return rows


class Constants(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("kept", torch.randn(4))
        self.register_buffer("scratch", torch.randn(4), persistent=False)
        self.lit = torch.tensor([0.5, 1.5, 2.5, 3.5])

    def forward(self, x):
        return (x @ self.w + self.kept) * self.scratch + self.lit


@case
def test_answers():
    from executorch.backends.esp32s3.operator_support import const_weights
    from torch.export.graph_signature import ExportGraphSignature

    print("the signature cache answers like torch, and follows a new signature:")
    ep = torch.export.export(Constants(), (torch.randn(2, 4),), strict=False)
    phs = [n for n in ep.graph_module.graph.nodes if n.op == "placeholder"]
    plain = [const_weights._const_tensor(ep, n) for n in phs]
    with const_weights._placeholder_index(ep):
        cached = [const_weights._const_tensor(ep, n) for n in phs]
        active = const_weights._placeholder_sources(ep) is not None
    check("every placeholder resolves to the same object",
          all(a is b for a, b in zip(plain, cached)) and len(phs) == 5,
          f"{len(phs)} placeholders, {sum(t is not None for t in plain)} constants")
    check("the index is active inside partition()'s scope, and only there",
          active and const_weights._placeholder_sources(ep) is None)

    sig = ep.graph_signature
    specs = list(sig.input_specs)
    w, kept = (next(s for s in specs if s.target == t) for t in ("w", "kept"))
    w_node = next(n for n in phs if n.name == w.arg.name)
    with const_weights._placeholder_index(ep):
        before = const_weights._const_tensor(ep, w_node)
        # Re-point the parameter's placeholder at the buffer, as a new list.
        swapped = [type(s)(kind=kept.kind, arg=s.arg, target=kept.target,
                           persistent=kept.persistent) if s is w else s for s in specs]
        sig.input_specs = swapped
        after_list = const_weights._const_tensor(ep, w_node)
        sig.input_specs = specs
        ep._graph_signature = ExportGraphSignature(input_specs=swapped,
                                                   output_specs=sig.output_specs)
        after_sig = const_weights._const_tensor(ep, w_node)
    ep._graph_signature = sig
    check("a replaced spec list is re-read", before is ep.state_dict["w"]
          and after_list is ep.state_dict["kept"])
    check("a replaced signature is re-read", after_sig is ep.state_dict["kept"])


@contextlib.contextmanager
def lookups(index):
    """partition()'s lookups from the map (index) or from torch's, with
    lowering's output muted; yields the mock that counts the maps built."""
    from executorch.backends.esp32s3.operator_support import const_weights
    from executorch.backends.esp32s3.partition import espressif_partitioner

    with contextlib.ExitStack() as stack:
        if not index:
            stack.enter_context(mock.patch.object(
                espressif_partitioner, "_placeholder_index", contextlib.nullcontext))
        built = stack.enter_context(mock.patch.object(
            const_weights, "_signature_sources", wraps=const_weights._signature_sources))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        yield built


@case
def test_partition_builds_map():
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.partition import espressif_partitioner
    from executorch.backends.esp32s3.test import models

    print("partition() answers from the map it builds:")
    runs = []
    for index in (True, False):
        torch.manual_seed(0)
        ep = torch.export.export(models.MLP().eval(), (torch.randn(2, 96),), strict=False)
        with lookups(index) as built:
            edge = lower_esp32(ep, [espressif_partitioner.Esp32Partitioner()])
        runs.append((built.call_count, edge.to_executorch().buffer))
    (cached, pte), (plain, plain_pte) = runs
    check("lowering test/models.MLP builds one map, and only with it",
          cached == 1 and plain == 0, f"{cached} vs {plain} maps built")
    check("byte-identical .pte", pte == plain_pte, f"{len(pte):,} B")


def lower_nano(index):
    """2-layer nanoLlama3.2 through lower_esp32, partition()'s lookups from the
    map (index) or from torch's: (to_backend's graph as graph_key, number of
    delegates, .pte bytes, .ptd bytes, maps built)."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.partition import espressif_partitioner
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.examples.espressif.nano_zoo import build

    net, inputs = build.model("llama3_2", "ci")
    ep = torch.export.export(net, inputs, strict=False)
    with lookups(index) as built:
        edge = lower_esp32(ep, [espressif_partitioner.Esp32Partitioner(
            external_weights="w", format_policy="i4", group_weights=True, quantize_qk=True)])
    gm = edge.exported_program().graph_module
    delegates = sum(1 for n in gm.graph.nodes if "call_delegate" in str(n.target))
    program = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, external_constants=lambda node: "w"))
    with tempfile.TemporaryDirectory() as out:
        program.write_tensor_data_to_file(out)
        ptd = Path(out, "w.ptd").read_bytes()
    return graph_key(gm), delegates, program.buffer, ptd, built.call_count


@case
def test_same_program():
    print("nanoLlama3.2 lowers to the same program with the map as without:")
    try:
        ready = all(importlib.util.find_spec(m) for m in (
            "executorch.examples.espressif.nano_zoo", "transformers"))
    except ImportError:
        ready = False
    if not ready:
        print("  SKIP needs the nano zoo and transformers")
        SKIPS.append("nanoLlama3.2 (needs the nano zoo and transformers)")
        return
    plain = lower_nano(False)
    cached = lower_nano(True)
    check("a dozen partitions or more", plain[1] >= 12, f"{plain[1]} delegates")
    check("only the run with the map builds it", cached[4] > 0 and plain[4] == 0,
          f"{cached[4]} vs {plain[4]} maps built")
    check("to_backend's graph is the same, node for node", cached[0] == plain[0],
          f"{len(cached[0])} vs {len(plain[0])} nodes")
    check("byte-identical .pte and .ptd", cached[2:4] == plain[2:4],
          f"{len(cached[2]):,} B + {len(cached[3]):,} B")


def main():
    for test in (test_answers, test_partition_builds_map, test_same_program):
        test.__wrapped__()
    print()
    if SKIPS:
        print(f"SKIPPED: {', '.join(SKIPS)}")
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("the signature cache answers like torch, and the bytes do not move")
    return 0


if __name__ == "__main__":
    sys.exit(main())
