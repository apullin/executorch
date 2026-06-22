# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.partitioner import DirectLoweringResult, Partitioner
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from torch.export import export, ExportedProgram


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class _DirectLoweringPartitioner(Partitioner):
    def __init__(self) -> None:
        self.partition_called = False

    def partition(self, exported_program: ExportedProgram):
        self.partition_called = True
        raise AssertionError("legacy partition path reached")

    def direct_lower(self, exported_program, compile_config):
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_backend(AddMulPartitionerDemo()).exported_program()
        return DirectLoweringResult(exported_program=edge_program)


class TestDirectLowering(unittest.TestCase):
    def test_direct_lowering_takes_fast_path(self):
        partitioner = _DirectLoweringPartitioner()
        example_inputs = (torch.ones(1), torch.zeros(1))
        ep = export(AddModule(), example_inputs, strict=True).run_decompositions()
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _enable_direct_backend_lowering=True,
            ),
        )

        self.assertFalse(partitioner.partition_called)
        delegate_count = sum(
            1
            for node in edge.exported_program().graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.higher_order.executorch_call_delegate
        )
        self.assertEqual(delegate_count, 1)

    def test_direct_lowering_skips_when_transform_passes_are_requested(self):
        example_inputs = (torch.ones(1), torch.zeros(1))
        ep = export(AddModule(), example_inputs, strict=True).run_decompositions()

        with self.assertRaisesRegex(AssertionError, "legacy partition path reached"):
            to_edge_transform_and_lower(
                ep,
                transform_passes=[],
                partitioner=[_DirectLoweringPartitioner()],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _enable_direct_backend_lowering=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
