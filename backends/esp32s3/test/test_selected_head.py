# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic legal-token output rows fuse into one full-table projection.

The graph keeps the vocabulary table constant and supplies only int64 row IDs
at runtime.  The descriptor still stores the complete table height; a required
extension changes execute's output walk to the fixed selected width.
"""

import contextlib
import io
import struct
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_FORMAT,
    EXT_COUNT_FORMAT,
    EXT_RECORD_HEADER_FORMAT,
    EXT_RECORD_HEADER_BYTES,
    EXT_SELECTED_ROWS,
    FLAG_SOFTCAP,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
    BLOB_HEADER_BYTES,
    SELECTED_ROWS_FORMAT,
    SELECTED_ROWS_BYTES,
)
from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


class SelectedHead(nn.Module):
    def __init__(self, vocab=257, width=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, width))

    def forward(self, value, rows):
        selected = torch.index_select(self.weight, 0, rows)
        logits = F.linear(value, selected) * 0.196116
        return torch.tanh(logits / 20.0) * 20.0


def main():
    from executorch.backends.test.harness.stages import StageType
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    torch.manual_seed(0)
    model = SelectedHead().eval()
    value = torch.randn(2, 64)
    rows = torch.tensor([1, 5, 8, 13, 21, 144, 233])

    print("partition and fusion:")
    tester = Esp32Tester(model, (value, rows))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        tester.export().to_edge_transform_and_lower()
    edge = tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    targets = [str(node.target) for node in edge.exported_program().graph.nodes
               if node.op == "call_function"]
    delegates = sum("executorch_call_delegate" in target for target in targets)
    check("selected gather, transpose, matmul and softcap form one delegate",
          delegates == 1, f"delegates={delegates}")
    check("no selected-head arithmetic remains portable",
          not any(fragment in target for target in targets for fragment in
                  ("index_select", "aten.mm.", "aten.tanh.", "aten.div.")),
          str(targets))
    check("lowering reports full and selected row counts",
          "select7/257" in log.getvalue())
    edge.to_executorch()

    print("serialized contract:")
    raw_edge = to_edge(torch.export.export(model, (value, rows)))
    with contextlib.redirect_stdout(io.StringIO()):
        result = Esp32Backend.preprocess(
            raw_edge.exported_program(),
            Esp32CompileSpec(external_weights="selected_test").to_list(),
        )
    blob = result.processed_bytes
    _magic, _version, n_ops, ext_off = struct.unpack_from(
        BLOB_HEADER_FORMAT, blob)
    desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES)))
    count, = struct.unpack_from(EXT_COUNT_FORMAT, blob, ext_off)
    op_index, ext_type, length = struct.unpack_from(
        EXT_RECORD_HEADER_FORMAT, blob, ext_off + struct.calcsize(EXT_COUNT_FORMAT))
    payload_at = (ext_off + struct.calcsize(EXT_COUNT_FORMAT)
                  + EXT_RECORD_HEADER_BYTES)
    rows_arg, selected_count = struct.unpack_from(
        SELECTED_ROWS_FORMAT, blob, payload_at)
    check("one selected-row extension is attached to the head",
          (n_ops, count, op_index, ext_type, length)
          == (1, 1, 0, EXT_SELECTED_ROWS, SELECTED_ROWS_BYTES),
          f"ops={n_ops} ext={count}/{ext_type}/{length}")
    check("descriptor retains complete table dimensions",
          (desc["m"], desc["k"], desc["n"]) == (2, 64, 257),
          f"{desc['m']}x{desc['k']}x{desc['n']}")
    check("extension names int64 rows input and seven-result output",
          (rows_arg, selected_count) == (1, 7),
          f"arg={rows_arg} count={selected_count}")
    check("softcap remains fused on selected outputs",
          bool(desc["flags"] & FLAG_SOFTCAP))
    named = result.data_store_output.external_data["selected_test"]
    sizes = sorted(len(result.data_store_output.buffers[entry.buffer_index])
                   for entry in named.values())
    check("external storage contains the full table, not seven copied rows",
          257 * 64 in sizes, str(sizes))

    # The runtime's selected-row walk reads one partition input, writes one
    # partition output and takes no residual; `init` refuses kFlagAdd and any
    # scratch sentinel on it. The gate has to refuse the same three shapes, or
    # the export succeeds and load_method fails (residual) or preprocess raises
    # (either chain direction). Each case here is refused BY NAME and the rest
    # of the graph still delegates -- a refusal that took the head's neighbours
    # down with it would be the empty-partition failure mode.
    print("contract refusals (gate mirrors the runtime):")
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes

    def lower_and_kinds(module, example):
        part = Esp32Partitioner()
        with contextlib.redirect_stdout(io.StringIO()):
            low = to_edge_transform_and_lower(
                torch.export.export(module, example), partitioner=[part])
            apply_passes(low)
        graph = low.exported_program().graph_module.graph
        delegates = sum("call_delegate" in str(n.target) for n in graph.nodes
                        if n.op == "call_function")
        return delegates, dict(part.support.kinds)

    class ResidualOnSelected(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(257, 64))

        def forward(self, value, rows, residual):
            selected = torch.index_select(self.weight, 0, rows)
            return F.linear(value, selected) + residual

    n_res, kinds = lower_and_kinds(
        ResidualOnSelected().eval(), (value, rows, torch.randn(2, 7)))
    check("residual onto a selected head is refused by name",
          kinds.get("residual on a selected-row head", 0) == 1, str(kinds))
    check("the selected head itself still delegates", n_res == 1,
          f"delegates={n_res}")

    class ChainedIntoSelected(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 64, bias=False)
            self.weight = nn.Parameter(torch.randn(257, 64))

        def forward(self, value, rows):
            hidden = self.proj(value)
            return F.linear(hidden, torch.index_select(self.weight, 0, rows))

    n_chain, kinds = lower_and_kinds(
        ChainedIntoSelected().eval(), (torch.randn(2, 32), rows))
    check("chaining a claimed projection into a selected head is refused",
          kinds.get("selected-row head chained from a claimed producer", 0)
          >= 1, str(kinds))
    check("the projection still delegates on its own", n_chain == 1,
          f"delegates={n_chain}")

    # The runtime's walk takes 1..n selected IDs (init refuses count > n), and
    # only the i4/i8 weight paths have a selected-row walk at all. Either
    # mismatch would emit a record that load_method rejects.
    class SmallTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 64))

        def forward(self, value, ids):
            return F.linear(value, torch.index_select(self.weight, 0, ids))

    n_wide, kinds = lower_and_kinds(
        SmallTable().eval(), (value, torch.tensor([0, 1, 2, 3, 1])))
    check("five IDs over a four-row table are refused by name",
          kinds.get("selected-row head wider than its table", 0) >= 1
          and n_wide == 0, f"{kinds}, delegates={n_wide}")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            to_edge_transform_and_lower(
                torch.export.export(SelectedHead().eval(), (value, rows)),
                partitioner=[Esp32Partitioner(
                    format_overrides={"weight": "i16"})])
        check("an i16 override on a selected head is refused by name", False,
              "emitted a record the runtime rejects")
    except RuntimeError as e:
        check("an i16 override on a selected head is refused by name",
              "selected-row" in str(e), str(e)[:90])

    print()
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("selected-row output head is fused, serialized, and lowerable")
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
