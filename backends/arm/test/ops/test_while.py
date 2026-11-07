# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import torch
import torch.fx

from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.quantizer.arm_quantizer import (
    _TOSAQuantizerV1,
    _TOSAQuantizerV2,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa import TosaSpecification
from pytest import mark
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

input_single = Tuple[torch.Tensor]
input_double = Tuple[torch.Tensor, torch.Tensor]


class WhileTwoInputsTwoOutputs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, lhs: torch.Tensor, rhs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def cond_fn(lhs_val: torch.Tensor, rhs_val: torch.Tensor) -> torch.Tensor:
            total = torch.sum(rhs_val)
            zero = torch.zeros_like(total)
            return torch.gt(total, zero).squeeze()

        def body_fn(
            lhs_val: torch.Tensor, rhs_val: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            next_lhs = torch.add(lhs_val, rhs_val)
            next_rhs = torch.sub(rhs_val, torch.full((1,), 1.0))
            return (next_lhs, next_rhs)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (lhs, rhs),
            (),
        )
        return result  # type: ignore


class WhileOneInputOneBufferTwoOutputs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((30.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(
            value: torch.Tensor, limit: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return (torch.add(value, value), limit.clone())

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value, self.threshold),
            (),
        )
        return result  # type: ignore


class DecreasingOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.gt(total, torch.full((1,), 60.0)).squeeze()

        def body_fn(value: torch.Tensor) -> Tuple[torch.Tensor]:
            return (torch.div(value, torch.full((1,), 2.0)),)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value,),
            (),
        )
        return result[0]  # type: ignore


class WhileAdditionalArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((128.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(value: torch.Tensor, limit: torch.Tensor) -> tuple[torch.Tensor]:
            return (torch.add(value, value),)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value,),
            (self.threshold,),
        )
        return result  # type: ignore


class WhileRotateAdditionalArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((0.5,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            return torch.gt(value[:, :1].sum(), limit).squeeze()

        def body_fn(value: torch.Tensor, limit: torch.Tensor) -> tuple[torch.Tensor]:
            return (torch.cat((value[:, 1:], value[:, :1]), dim=1),)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value,),
            (self.threshold,),
        )
        return result  # type: ignore


class WhileCatAdditionalArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("fill", torch.full((1, 1), 0.25))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
            return torch.lt(value.sum(), torch.full((1,), 20.0)).squeeze()

        def body_fn(value: torch.Tensor, fill: torch.Tensor) -> tuple[torch.Tensor]:
            doubled = torch.add(value[:, 1:], value[:, 1:])
            return (torch.cat((doubled, fill), dim=1),)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value,),
            (self.fill,),
        )
        return result  # type: ignore


class WhileThreeCarriedAdditionalArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((0.5,)))

    def forward(
        self, value: torch.Tensor, source: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        def cond_fn(
            value: torch.Tensor,
            source: torch.Tensor,
            key: torch.Tensor,
            limit: torch.Tensor,
        ) -> torch.Tensor:
            return torch.gt(key[:, :1].sum(), limit).squeeze()

        def body_fn(
            value: torch.Tensor,
            source: torch.Tensor,
            key: torch.Tensor,
            limit: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.cat((value[:, 1:], source[:, :1]), dim=1),
                torch.cat((source[:, 1:], source[:, :1]), dim=1),
                torch.cat((key[:, 1:], key[:, :1]), dim=1),
            )

        result, _, _ = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value, source, key),
            (self.threshold,),
        )
        return result  # type: ignore


class WhileChainedAdditionalArgs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((10.0,)))
        self.register_buffer("step", torch.linspace(0.01, 0.08, 8).reshape(1, 8))
        self.register_buffer("limit", torch.tensor((60.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def first_cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            return torch.lt(value.sum(), limit).squeeze()

        def first_body_fn(
            value: torch.Tensor, limit: torch.Tensor
        ) -> tuple[torch.Tensor]:
            return (torch.mul(value, 1.5),)

        def second_cond_fn(
            value: torch.Tensor, step: torch.Tensor, limit: torch.Tensor
        ) -> torch.Tensor:
            return torch.lt(value.sum(), limit).squeeze()

        def second_body_fn(
            value: torch.Tensor, step: torch.Tensor, limit: torch.Tensor
        ) -> tuple[torch.Tensor]:
            return (torch.add(value, step),)

        (value,) = torch.ops.higher_order.while_loop(
            first_cond_fn,
            first_body_fn,
            (value,),
            (self.threshold,),
        )
        result = torch.ops.higher_order.while_loop(
            second_cond_fn,
            second_body_fn,
            (value,),
            (self.step, self.limit),
        )
        return result  # type: ignore


class WhileSingleCapturedOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((128.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(
            value: torch.Tensor, limit: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return (torch.add(value, value), limit.clone())

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value, self.threshold),
            (),
        )
        return result[0]  # type: ignore


class WhileLargeThreshold(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((400.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(value: torch.Tensor, limit: torch.Tensor) -> tuple[torch.Tensor]:
            return (torch.add(value, value),)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value,),
            (self.threshold,),
        )
        return result  # type: ignore


def _single_input_case(
    module_factory: Callable[[], torch.nn.Module],
) -> Callable[[], Tuple[torch.nn.Module, input_single]]:
    def _create() -> Tuple[torch.nn.Module, input_single]:
        return module_factory(), (torch.ones(2, 3, 2, 2),)

    return _create


def _dual_input_case(
    module_factory: Callable[[], torch.nn.Module],
) -> Callable[[], Tuple[torch.nn.Module, input_double]]:
    def _create() -> Tuple[torch.nn.Module, input_double]:
        return module_factory(), (torch.zeros(2, 3), torch.full((2, 3), -2.0))

    return _create


test_cases: dict[str, Callable[[], Tuple[torch.nn.Module, Tuple]]] = {
    "two_in_two_out": _dual_input_case(WhileTwoInputsTwoOutputs),
    "one_in_one_buffer_two_out": _single_input_case(WhileOneInputOneBufferTwoOutputs),
    "decreasing_output": _single_input_case(DecreasingOutput),
    "additional_arg": _single_input_case(WhileAdditionalArg),
    "two_in_one_captured_out": _single_input_case(WhileSingleCapturedOutput),
    "large_threshold": _single_input_case(WhileLargeThreshold),
    "additional_arg_ramp": lambda: (
        WhileAdditionalArg(),
        (torch.linspace(0.05, 1.0, 24).reshape(2, 3, 2, 2),),
    ),
    "additional_arg_rotate": lambda: (
        WhileRotateAdditionalArg(),
        (torch.tensor([[6.0, 3.0, 8.0, 0.25, 5.0, 1.0, 7.0, 2.0]]),),
    ),
    "additional_arg_cat": lambda: (
        WhileCatAdditionalArg(),
        (torch.linspace(0.1, 1.0, 8).reshape(1, 8),),
    ),
    "additional_arg_three_carried": lambda: (
        WhileThreeCarriedAdditionalArg(),
        (
            torch.linspace(0.1, 0.8, 8).reshape(1, 8),
            torch.linspace(1.0, 8.0, 8).reshape(1, 8),
            torch.tensor([[6.0, 3.0, 8.0, 0.25, 5.0, 1.0, 7.0, 2.0]]),
        ),
    ),
}


@common.parametrize(
    "case",
    test_cases,
)
def test_while_loop_tosa_FP(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    pipeline = TosaPipelineFP[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_extensions=["cf"],
    )
    pipeline.run()


@common.parametrize(
    "case",
    test_cases,
)
def test_while_loop_tosa_INT(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    pipeline = TosaPipelineINT[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_extensions=["cf"],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        ArmTester.check_not,
        pipeline.tester,
        ["torch.ops.higher_order.while_loop"],
    )
    pipeline.run()


def test_while_loop_tosa_INT_composable_large_threshold():
    module, example_inputs = test_cases["large_threshold"]()
    pipeline = TosaPipelineINT[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_extensions=["cf"],
    )

    composable_quantizer = _TOSAQuantizerV2(pipeline.tester.compile_spec)
    composable_quantizer.set_global(get_symmetric_quantization_config())
    pipeline.quantizer.quantizer = composable_quantizer

    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        ArmTester.check_not,
        pipeline.tester,
        ["torch.ops.higher_order.while_loop"],
    )
    pipeline.run()


@common.parametrize(
    "case",
    {
        name: test_cases[name]
        for name in (
            "additional_arg",
            "additional_arg_ramp",
            "additional_arg_cat",
            "additional_arg_three_carried",
        )
    },
)
def test_while_loop_tosa_INT_legacy_quantizer(
    case: Callable[[], Tuple[torch.nn.Module, Tuple]],
):
    module, example_inputs = case()
    pipeline = TosaPipelineINT[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_extensions=["cf"],
    )

    legacy_quantizer = _TOSAQuantizerV1(pipeline.tester.compile_spec)
    legacy_quantizer.set_global(get_symmetric_quantization_config())
    pipeline.quantizer.quantizer = legacy_quantizer

    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        ArmTester.check_not,
        pipeline.tester,
        ["torch.ops.higher_order.while_loop"],
    )
    pipeline.run()


def test_while_loop_chained_additional_inputs_do_not_share_qspec():
    """The additional inputs of a loop fed by another loop keep their own
    qparams rather than joining those of the loop state.
    """
    inputs = (torch.linspace(0.1, 1.0, 8).reshape(1, 8),)
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT+cf"))
    quantizer.set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(
        export(WhileChainedAdditionalArgs(), inputs, strict=True).module(), quantizer
    )
    prepared(*inputs)
    quantized = convert_pt2e(prepared)

    _, second_loop = quantized.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.while_loop
    )
    value_scale, *additional_scales = (
        dequantize.args[1] for dequantize in second_loop.args[2]
    )
    assert value_scale not in additional_scales


@common.parametrize(
    "case",
    test_cases,
)
def test_while_loop_u55_INT(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    OpNotSupportedPipeline[tuple](
        module,
        example_inputs,
        non_delegated_ops={"torch.ops.higher_order.while_loop": 1},
        u55_subset=True,
    ).run()


@common.parametrize(
    "case",
    test_cases,
)
@common.XfailIfNoCorstone320
def test_while_loop_u85_INT(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    EthosU85PipelineINT[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
    ).run()


@mark.skip("While not supported in model_converter.")
@common.parametrize(
    "case",
    test_cases,
)
@common.SkipIfNoModelConverter
def test_while_loop_vgf_FP(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    VgfPipeline[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_version="TOSA-1.0+FP",
    ).run()


@mark.skip("While not supported in model_converter.")
@common.parametrize(
    "case",
    test_cases,
)
@common.SkipIfNoModelConverter
def test_while_loop_vgf_INT(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    VgfPipeline[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
    ).run()
