# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor, torch.Tensor]  # (input, h0)


def get_test_inputs():
    return (
        torch.randn(5, 3, 10),  # input: (seq_len=5, batch=3, input_size=10)
        torch.randn(2, 3, 20),  # h0: (num_layers=2, batch=3, hidden_size=20)
    )


class TestGRU:
    """Tests torch.nn.GRU via the DecomposeGruPass."""

    gru = torch.nn.GRU(10, 20, 2)
    gru = gru.eval()

    model_example_inputs = get_test_inputs()


def test_gru_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestGRU.gru,
        TestGRU.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", inputs=get_test_inputs(), atol=3e-1
    )
    pipeline.run()


def test_gru_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestGRU.gru,
        TestGRU.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_test_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_gru_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestGRU.gru,
        TestGRU.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_test_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_gru_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestGRU.gru,
        TestGRU.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_test_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()
