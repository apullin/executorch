# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.quantizable.rnn import RNN
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor, torch.Tensor]


def get_test_inputs():
    return (
        torch.randn(5, 3, 10),  # input: (seq_len=5, batch=3, input_size=10)
        torch.randn(2, 3, 20),  # h0: (num_layers=2, batch=3, hidden_size=20)
    )


class TestRNNQuantizable:
    """Tests quantizable RNN module (QAT-capable frontend decomposition)."""

    rnn_tanh = RNN(10, 20, 2).eval()
    rnn_relu = RNN(10, 20, 2, nonlinearity="relu").eval()
    model_example_inputs = get_test_inputs()


# --- Tanh tests ---


def test_rnn_quantizable_tanh_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestRNNQuantizable.rnn_tanh,
        TestRNNQuantizable.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", inputs=get_test_inputs(), atol=3e-1
    )
    pipeline.run()


def test_rnn_quantizable_tanh_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestRNNQuantizable.rnn_tanh,
        TestRNNQuantizable.model_example_inputs,
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
def test_rnn_quantizable_tanh_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestRNNQuantizable.rnn_tanh,
        TestRNNQuantizable.model_example_inputs,
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
def test_rnn_quantizable_tanh_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestRNNQuantizable.rnn_tanh,
        TestRNNQuantizable.model_example_inputs,
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


# --- ReLU tests ---


def test_rnn_quantizable_relu_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestRNNQuantizable.rnn_relu,
        TestRNNQuantizable.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", inputs=get_test_inputs(), atol=3e-1
    )
    pipeline.run()


def test_rnn_quantizable_relu_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestRNNQuantizable.rnn_relu,
        TestRNNQuantizable.model_example_inputs,
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


# --- Bidirectional tests ---


def get_bidi_test_inputs():
    return (
        torch.randn(5, 3, 10),
        torch.randn(4, 3, 20),  # h0: (2*num_layers=4, batch=3, hidden_size=20)
    )


class TestRNNQuantizableBidirectional:
    """Tests bidirectional quantizable RNN module."""

    rnn = RNN(10, 20, 2, bidirectional=True).eval()
    model_example_inputs = get_bidi_test_inputs()


def test_rnn_quantizable_bidirectional_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        TestRNNQuantizableBidirectional.rnn,
        TestRNNQuantizableBidirectional.model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_bidi_test_inputs(),
        atol=3e-1,
    )
    pipeline.run()


def test_rnn_quantizable_bidirectional_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        TestRNNQuantizableBidirectional.rnn,
        TestRNNQuantizableBidirectional.model_example_inputs,
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
def test_rnn_quantizable_bidirectional_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        TestRNNQuantizableBidirectional.rnn,
        TestRNNQuantizableBidirectional.model_example_inputs,
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
def test_rnn_quantizable_bidirectional_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        TestRNNQuantizableBidirectional.rnn,
        TestRNNQuantizableBidirectional.model_example_inputs,
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
