# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_lstm_pass import DecomposeLstmPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)
from executorch.backends.test.harness.stages.run_passes import RunPasses

input_t = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def get_test_inputs():
    return (
        torch.randn(5, 3, 10),  # input: (seq_len=5, batch=3, input_size=10)
        (
            torch.randn(2, 3, 20),  # h0: (num_layers=2, batch=3, hidden_size=20)
            torch.randn(2, 3, 20),  # c0: same shape as h0
        ),
    )


class TestLSTM:
    """Tests torch.nn.LSTM via the DecomposeLstmPass."""

    lstm = torch.nn.LSTM(10, 20, 2).eval()
    model_example_inputs = get_test_inputs()


def test_lstm_decompose_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "check.aten",
        pipeline.tester.run_passes,
        RunPasses(pass_manager_cls=None, pass_list=[DecomposeLstmPass]),
        suffix="decompose_lstm",
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", inputs=get_test_inputs(), atol=3e-1
    )
    pipeline.run()


def test_lstm_decompose_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
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
def test_lstm_decompose_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
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
def test_lstm_decompose_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestLSTM.lstm,
        TestLSTM.model_example_inputs,
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


# --- Bidirectional LSTM tests ---


def get_bidi_test_inputs():
    return (
        torch.randn(5, 3, 10),
        (
            torch.randn(4, 3, 20),  # h0: (2*num_layers=4, batch=3, hidden_size=20)
            torch.randn(4, 3, 20),  # c0
        ),
    )


class TestLSTMBidirectional:
    """Tests bidirectional torch.nn.LSTM via the DecomposeLstmPass."""

    lstm = torch.nn.LSTM(10, 20, 2, bidirectional=True).eval()
    model_example_inputs = get_bidi_test_inputs()


def test_lstm_bidirectional_decompose_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestLSTMBidirectional.lstm,
        TestLSTMBidirectional.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "check.aten",
        pipeline.tester.run_passes,
        RunPasses(pass_manager_cls=None, pass_list=[DecomposeLstmPass]),
        suffix="decompose_lstm",
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_bidi_test_inputs(),
        atol=3e-1,
    )
    pipeline.run()


def test_lstm_bidirectional_decompose_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestLSTMBidirectional.lstm,
        TestLSTMBidirectional.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_bidi_test_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_lstm_bidirectional_decompose_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestLSTMBidirectional.lstm,
        TestLSTMBidirectional.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_bidi_test_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_lstm_bidirectional_decompose_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestLSTMBidirectional.lstm,
        TestLSTMBidirectional.model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_bidi_test_inputs(),
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()
