# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end pipeline tests for quantizable recurrent modules.

Exercises the quantizable RNN and GRU implementations through TOSA reference
pipelines and Ethos-U targets, including the pipelines' quantization and
delegation assertions.
"""

from typing import Tuple

import torch
from executorch.backends.arm.quantizable import GRU as QuantizableGRU
from executorch.backends.arm.quantizable import RNN as QuantizableRNN
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)


# ──────────────────────────── Input type aliases ────────────────────────────

rnn_input_t = Tuple[torch.Tensor, torch.Tensor]  # (x, h)


# ──────────────────────────── Model definitions ─────────────────────────────


class RNNTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = QuantizableRNN(
            10, 20, 1, nonlinearity="tanh", batch_first=True
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h)

    @staticmethod
    def get_inputs() -> rnn_input_t:
        return (torch.randn(2, 5, 10), torch.randn(1, 2, 20))


class RNNRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = QuantizableRNN(
            10, 20, 1, nonlinearity="relu", batch_first=True
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h)

    @staticmethod
    def get_inputs() -> rnn_input_t:
        return (torch.randn(2, 5, 10), torch.randn(1, 2, 20))


class GRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = QuantizableGRU(10, 20, 1, batch_first=True)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gru(x, h)

    @staticmethod
    def get_inputs() -> rnn_input_t:
        return (torch.randn(2, 5, 10), torch.randn(1, 2, 20))


# ──────────────────────────── Helpers ───────────────────────────────────────


def _run_tosa_fp_pipeline(module, inputs, input_type):
    """Run TosaPipelineFP for a recurrent model."""
    pipeline = TosaPipelineFP[input_type](
        module,
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=3e-1,
        rtol=3e-1,
    )
    pipeline.pop_stage("check.aten")
    pipeline.run()


def _run_tosa_int_pipeline(module, inputs, input_type):
    """Run TosaPipelineINT for a recurrent model."""
    pipeline = TosaPipelineINT[input_type](
        module,
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.pop_stage("check.aten")
    pipeline.change_args(
        "run_method_and_compare_outputs",
        atol=3e-1,
        qtol=1.0,
    )
    pipeline.run()


def _run_ethos_u_pipeline(pipeline_cls, module, inputs, input_type):
    """Run an EthosU pipeline for a recurrent model."""
    pipeline = pipeline_cls[input_type](
        module,
        inputs,
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.pop_stage("check.aten")
    pipeline.run()


# ──────────────────────────── RNN FP tests ──────────────────────────────────


def test_decompose_rnn_tosa_FP_tanh_e2e():
    _run_tosa_fp_pipeline(RNNTanh(), RNNTanh.get_inputs(), rnn_input_t)


def test_decompose_rnn_tosa_FP_relu_e2e():
    _run_tosa_fp_pipeline(RNNRelu(), RNNRelu.get_inputs(), rnn_input_t)


# ──────────────────────────── RNN INT tests ─────────────────────────────────


def test_decompose_rnn_tosa_INT_tanh_e2e():
    _run_tosa_int_pipeline(RNNTanh(), RNNTanh.get_inputs(), rnn_input_t)


def test_decompose_rnn_tosa_INT_relu_e2e():
    _run_tosa_int_pipeline(RNNRelu(), RNNRelu.get_inputs(), rnn_input_t)


# ──────────────────────────── GRU tests ─────────────────────────────────────


def test_decompose_gru_tosa_FP_e2e():
    _run_tosa_fp_pipeline(GRU(), GRU.get_inputs(), rnn_input_t)


def test_decompose_gru_tosa_INT_e2e():
    _run_tosa_int_pipeline(GRU(), GRU.get_inputs(), rnn_input_t)


# ──────────────────────── EthosU55 INT probes ───────────────────────────────


@common.XfailIfNoCorstone300
def test_decompose_rnn_u55_INT_tanh_e2e():
    _run_ethos_u_pipeline(
        EthosU55PipelineINT, RNNTanh(), RNNTanh.get_inputs(), rnn_input_t
    )


@common.XfailIfNoCorstone300
def test_decompose_gru_u55_INT_e2e():
    _run_ethos_u_pipeline(EthosU55PipelineINT, GRU(), GRU.get_inputs(), rnn_input_t)


# ──────────────────────── EthosU85 INT probes ───────────────────────────────


@common.XfailIfNoCorstone320
def test_decompose_rnn_u85_INT_tanh_e2e():
    _run_ethos_u_pipeline(
        EthosU85PipelineINT, RNNTanh(), RNNTanh.get_inputs(), rnn_input_t
    )


@common.XfailIfNoCorstone320
def test_decompose_gru_u85_INT_e2e():
    _run_ethos_u_pipeline(EthosU85PipelineINT, GRU(), GRU.get_inputs(), rnn_input_t)
