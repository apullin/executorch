# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export tiny delegated PTEs and check every host output against QEMU references.

Build the runner first (test/host_runner/README.md).
ESP32_HOST_RUNNER can override the default cmake-out-espressif/esp32_host_runner path.
"""

import re
import struct
import subprocess

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.backend.utils import get_delegated_payload

from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_BYTES,
    BLOB_HEADER_FORMAT,
    FLAG_ADD,
    FLAG_CLAMP,
    FLAG_EXTERNAL_WEIGHTS,
    FLAG_FP16_SCALES,
    OP_ADDMM_I4,
    OP_ADDMM_I8,
    OP_BMM_DYN_I8,
    OP_BMM_KCACHE_I8,
    OP_DESC_BYTES,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
)
from executorch.backends.esp32s3.serialization.weight_formats import (
    SCALE_GROUP,
    quantize_absmax_rows,
)
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3._passes import lower_esp32
from executorch.backends.esp32s3.test.reference import (
    addmm_codes_ref,
    addmm_i8_absmax,
    bmm_qk_i8,
    sdpa_qk_i8,
    softmax_f32_portable,
)
from executorch.backends.esp32s3.test.host import host_runner

MODES = [(), ("--force-scalar",), ("--min-latency",),
         ("--force-scalar", "--min-latency")]


@pytest.fixture(scope="session")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def export_model(model, inputs, directory, **options):
    model.eval().requires_grad_(False)
    # A hard gate before export: these fixtures must never grow into real models.
    assert sum(p.numel() for p in model.parameters()) <= 10_000
    assert sum(x.numel() for x in inputs) <= 4096
    exported = torch.export.export(model, inputs)
    edge = lower_esp32(exported, [Esp32Partitioner(**options)])
    payloads = get_delegated_payload(edge.exported_program().graph_module)
    assert len(payloads) == 1, "fixture must actually exercise one delegate partition"
    blob = next(iter(payloads.values()))[2]
    _, _, count, _ = struct.unpack_from(BLOB_HEADER_FORMAT, blob)
    descriptors = [dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES + i * OP_DESC_BYTES)))
        for i in range(count)]
    tag = options.get("external_weights")
    if tag:
        config = ExecutorchBackendConfig(delegate_alignment=16,
                                          external_constants=lambda node: tag)
    else:
        config = ExecutorchBackendConfig(delegate_alignment=16)
    program = edge.to_executorch(config=config)
    pte = directory / "model.pte"
    pte.write_bytes(program.buffer)
    ptd = None
    if tag:
        program.write_tensor_data_to_file(str(directory))
        ptd = directory / f"{tag}.ptd"
        assert ptd.is_file()
    return pte, ptd, descriptors


def run_model(runner, pte, ptd, inputs, expected, flags=(), method="forward"):
    command = [str(runner), "--pte", str(pte), "--method", method, *flags]
    if ptd:
        command += ["--ptd", str(ptd)]
    for index, value in enumerate(inputs):
        path = pte.parent / f"input{index}.bin"
        np.ascontiguousarray(value.detach().numpy()).tofile(path)
        command += ["--input", str(path)]
    output_paths = [pte.parent / f"output{index}.bin" for index in range(len(expected))]
    for path in output_paths:
        # Do not let output from a previous run make a failing runner look green.
        path.unlink(missing_ok=True)
        command += ["--output", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = []
    for path, reference in zip(output_paths, expected):
        assert path.stat().st_size == reference.nbytes
        actual = np.fromfile(path, dtype=reference.dtype).reshape(reference.shape)
        # QEMU's checksum gate is 2e-4 relative. Host tests compare every element
        # at 2e-6; the integer kernels themselves have an exact CTest gate.
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
        outputs.append(actual)
    return outputs, result.stdout


@pytest.mark.parametrize("rows", [1, 3])
def test_int8_linear(runner, tmp_path, rows):
    torch.manual_seed(101)
    model = nn.Linear(33, 32)
    value = torch.randn(rows, 33)
    value[0].zero_()
    value[0, :7] = torch.tensor([127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
    if rows > 1:
        value[1].zero_()
    with torch.no_grad():
        model.weight[0].zero_()
    reference = addmm_i8_absmax(value.numpy(), model.weight.detach().numpy(),
                              model.bias.detach().numpy())
    pte, ptd, descriptors = export_model(model, (value,), tmp_path, format_policy="i8")
    assert len(descriptors) == 1 and descriptors[0]["opcode"] == OP_ADDMM_I8
    baseline = None
    for flags in MODES:
        outputs, log = run_model(runner, pte, ptd, (value,), (reference,), flags)
        if baseline is None:
            baseline = outputs[0]
        else:
            np.testing.assert_array_equal(outputs[0], baseline)
        match = re.search(r"columns PIE (\d+) / scalar (\d+)", log)
        assert match, log
        pie, scalar = map(int, match.groups())
        if "--force-scalar" in flags:
            assert scalar > 0 and pie == 0
        else:
            assert pie > 0
        split = re.search(r"esp32 split: (\d+) column-split", log)
        assert split and int(split[1]) == (rows if "--min-latency" in flags else 0), log


def test_grouped_int4_relu_external(runner, tmp_path):
    torch.manual_seed(102)
    model = nn.Sequential(nn.Linear(145, 32), nn.ReLU())
    value = torch.randn(3, 145)
    value[1].zero_()
    linear = model[0]
    codes, scales = quantize_absmax_rows(
        linear.weight.detach().numpy(), "i4", group=SCALE_GROUP, fp16_scales=True)
    reference = np.maximum(addmm_codes_ref(
        value.numpy(), codes, scales, linear.bias.detach().numpy(), SCALE_GROUP),
        np.float32(0.0))
    assert np.any(reference == 0.0) and np.any(reference > 0.0)
    pte, ptd, descriptors = export_model(
        model, (value,), tmp_path, format_policy="i4", group_weights=True,
        external_weights="host_weights")
    assert len(descriptors) == 1
    desc = descriptors[0]
    assert desc["opcode"] == OP_ADDMM_I4 and desc["scale_group"] == SCALE_GROUP
    assert desc["flags"] & FLAG_CLAMP
    assert desc["flags"] & FLAG_FP16_SCALES
    assert desc["flags"] & FLAG_EXTERNAL_WEIGHTS
    baseline = None
    for flags in MODES:
        outputs, log = run_model(runner, pte, ptd, (value,), (reference,), flags)
        assert "weights external 2 / inline 0" in log, log
        if baseline is None:
            baseline = outputs[0]
        else:
            np.testing.assert_array_equal(outputs[0], baseline)
        if "--min-latency" in flags:
            assert "esp32 split: 3 column-split" in log, log

    # File contract and loader failures must fail before writing output bytes.
    bad_input = tmp_path / "short.bin"
    bad_input.write_bytes(b"\0" * 4)
    result = subprocess.run(
        [str(runner), "--pte", str(pte), "--ptd", str(ptd),
         "--input", str(bad_input), "--output", str(tmp_path / "bad-output.bin")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "input byte count" in result.stderr
    assert not (tmp_path / "bad-output.bin").exists()
    result = subprocess.run(
        [str(runner), "--pte", str(pte), "--input", str(tmp_path / "input0.bin"),
         "--output", str(tmp_path / "missing-ptd-output.bin")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (tmp_path / "missing-ptd-output.bin").exists()


@pytest.mark.parametrize("act", ["sigmoid", "gelu_tanh", "gelu", "silu"])
def test_clamp_after_fused_activation(runner, tmp_path, act):
    """A clamp, and then a residual, over a fused activation ride the same
    descriptor in the runtime's order -- activation, clamp, operand -- so the
    output is the activation-only op's, clipped, plus r, bit for bit. (Leaky
    ReLU is the exception the gates refuse: its slope rides clamp_lo.)"""
    lo, hi = -0.1, 0.6
    fn = {"sigmoid": torch.sigmoid,
          "gelu_tanh": lambda t: torch.nn.functional.gelu(t, approximate="tanh"),
          "gelu": torch.nn.functional.gelu,
          "silu": torch.nn.functional.silu}[act]

    class Net(nn.Module):
        def __init__(self, fc, clamp, residual):
            super().__init__()
            self.fc, self.clamp, self.residual = fc, clamp, residual

        def forward(self, x, *r):
            y = fn(self.fc(x))
            if self.clamp:
                y = torch.clamp(y, lo, hi)
            return y + r[0] if self.residual else y

    torch.manual_seed(103)
    fc = nn.Linear(16, 32)
    value, residual = torch.randn(2, 16), torch.randn(2, 32)
    base_dir = tmp_path / "act"
    base_dir.mkdir()
    pte, _, _ = export_model(Net(fc, False, False), (value,), base_dir)
    np.ascontiguousarray(value.numpy()).tofile(base_dir / "input0.bin")
    result = subprocess.run(
        [str(runner), "--pte", str(pte), "--input", str(base_dir / "input0.bin"),
         "--output", str(base_dir / "output0.bin")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    base = np.fromfile(base_dir / "output0.bin", dtype=np.float32).reshape(2, 32)
    clipped = np.clip(base, np.float32(lo), np.float32(hi))
    assert np.any(clipped != base) and np.any(clipped == base)
    for residual_too in (False, True):
        directory = tmp_path / f"clamp{int(residual_too)}"
        directory.mkdir()
        inputs = (value, residual) if residual_too else (value,)
        pte, _, descriptors = export_model(
            Net(fc, True, residual_too), inputs, directory)
        assert len(descriptors) == 1
        assert descriptors[0]["flags"] & FLAG_CLAMP
        assert bool(descriptors[0]["flags"] & FLAG_ADD) == residual_too
        expected = clipped + residual.numpy() if residual_too else clipped
        outputs, _ = run_model(runner, pte, None, inputs, (expected,))
        np.testing.assert_array_equal(outputs[0], expected)


def kernel_calls(log):
    match = re.search(r"host kernels: dot_i8 (\d+), unpack_i4 (\d+), "
                      r"dot_i4_groups (\d+), rmsnorm (\d+) call", log)
    assert match, log
    return dict(zip(("dot_i8", "unpack_i4", "dot_i4_groups", "rmsnorm"),
                    map(int, match.groups())))


def test_int4_decode_row_fused(runner, tmp_path):
    # One activation row in whole 32-code groups with 16-byte packed rows: the
    # decode shape that takes the fused dot_i4_groups_s3 instead of unpack + dot.
    torch.manual_seed(105)
    model = nn.Linear(256, 32)
    value = torch.randn(1, 256)
    codes, scales = quantize_absmax_rows(
        model.weight.detach().numpy(), "i4", group=SCALE_GROUP, fp16_scales=True)
    reference = addmm_codes_ref(
        value.numpy(), codes, scales, model.bias.detach().numpy(), SCALE_GROUP)
    pte, ptd, descriptors = export_model(
        model, (value,), tmp_path, format_policy="i4", group_weights=True,
        external_weights="host_weights")
    assert len(descriptors) == 1 and descriptors[0]["opcode"] == OP_ADDMM_I4
    baseline, calls = None, {}
    for flags in MODES:
        outputs, log = run_model(runner, pte, ptd, (value,), (reference,), flags)
        assert "PIE fused int4 dot selftest: PASS" in log, log
        calls[flags] = kernel_calls(log)
        if baseline is None:
            baseline = outputs[0]
        else:
            np.testing.assert_array_equal(outputs[0], baseline)
    # Init selftests call every kernel in every mode; the rest are the model's.
    # The scalar dot unpacks each of the 32 weight rows (still with the PIE
    # unpack); the fused path makes one call per row instead and unpacks none.
    scalar = calls[("--force-scalar",)]
    for flags, counts in calls.items():
        fused = 0 if "--force-scalar" in flags else 32
        assert counts["dot_i4_groups"] - scalar["dot_i4_groups"] == fused, (flags, counts)
        assert scalar["unpack_i4"] - counts["unpack_i4"] == fused, (flags, counts)


class LinearOverConstant(nn.Module):
    """A Linear whose activation is a constant (a learned query row)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(13, 1)
        self.query = nn.Parameter(torch.randn(13))

    def forward(self, x):
        return self.fc(self.query)


def test_pt2e_dynamic_linear_over_a_constant(runner, tmp_path):
    """A Linear over a constant activation, quantised by the default
    Esp32Quantizer and lowered with the README's steps: the
    program keeps no chooser and runs here with the delegate's numbers. How
    the export disposes of the chain over the constant is the lowering's
    choice, and the test holds for each: one that leaves a dequantize of
    constant codes outside the delegate (convert_pt2e folded the quantize,
    constant propagation ran PyTorch's chooser) has it run on
    dequantize_per_tensor.Tensor_out; one that folds the whole chain at
    build time leaves no quantized op at all."""
    from torch.fx import Interpreter
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # to_executorch emits a quantized_decomposed op left portable only once
    # ExecuTorch's out variants are registered, which the lowering does not do
    # for every such op: imported here, as an application's export does.
    import executorch.kernels.quantized  # noqa: F401
    from executorch.backends.esp32s3 import Esp32Quantizer
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    torch.manual_seed(106)
    model = LinearOverConstant().eval().requires_grad_(False)
    example = (torch.randn(13, 1),)
    prepared = prepare_pt2e(torch.export.export(model, example).module(), Esp32Quantizer())
    prepared(*example)
    exported = torch.export.export(convert_pt2e(prepared), example)

    # The activation as PyTorch dequantises it, the weight's int4 codes and
    # scales, and the model's own output, from the program the lowering reads.
    seen = {}

    class Record(Interpreter):
        def run_node(self, node):
            out = super().run_node(node)
            if node.op == "call_function":
                seen[str(node.target)] = (self.fetch_args_kwargs_from_env(node)[0], out)
            return out

    want = Record(exported.module()).run(*example).numpy()
    activation = seen["quantized_decomposed.dequantize_per_tensor.tensor"][1]
    codes, scales = seen["quantized_decomposed.dequantize_per_channel.default"][0][:2]
    reference = addmm_codes_ref(
        activation.reshape(1, 13).numpy(), codes.numpy(), scales.numpy(),
        model.fc.bias.numpy(), 13).reshape(1)

    edge = lower_esp32(exported, [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    ops = ops_of_program(program)
    assert not any("choose_qparams" in op for op in ops), ops
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, example, (reference,))
    # The delegate quantises the dequantised row again, onto the same codes.
    np.testing.assert_allclose(outputs[0], want, rtol=1e-5, atol=1e-6)


class DequantizeTensorArgs(nn.Module):
    """x plus int8 codes dequantised by a scale and zero point held in
    one-element tensors."""

    def forward(self, x, codes, scale, zero_point):
        return x + torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            codes, scale, zero_point, -128, 127, torch.int8)


class DequantizeConstantCodes(DequantizeTensorArgs):
    """The same with codes, scale and zero point that are program constants."""

    def __init__(self, codes, scale, zero_point):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

    def forward(self, x):
        return super().forward(x, self.codes, self.scale, self.zero_point)


@pytest.mark.parametrize("qparams", ["constant", "input"])
def test_portable_dequantize_with_tensor_qparams(runner, tmp_path, qparams):
    """A per-tensor dequantize the delegate does not take, its scale and zero
    point in one-element tensors (float64 and int64, as PyTorch's choosers
    return them), runs on ExecuTorch's dequantize_per_tensor.Tensor_out and
    computes torch's bits. The firmware's selective build registers that
    kernel whenever a program names it (gen_kernels.py), and the host
    registers it beside the static pair kernels (codegen_portable.py
    QUANTIZED). The qparams are program constants, or inputs known only at
    run time."""
    import executorch.kernels.quantized  # noqa: F401  (as in the test above)
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    torch.manual_seed(107)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    codes[0, :2] = torch.tensor([-128, 127])
    # 0.0123 is not a float32 value: both sides compute in float32 from it.
    scale = torch.tensor([0.0123], dtype=torch.float64)
    zero_point = torch.tensor([3])
    x = torch.randn(4, 8)
    if qparams == "constant":
        model, inputs = DequantizeConstantCodes(codes, scale, zero_point), (x,)
    else:
        model, inputs = DequantizeTensorArgs(), (x, codes, scale, zero_point)
    want = model(*inputs).numpy()
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    if qparams == "input":
        # A lowering may fold constant qparams into the scalar overload; it
        # cannot fold inputs.
        ops = ops_of_program(program)
        assert "quantized_decomposed::dequantize_per_tensor.Tensor_out" in ops, ops
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, inputs, (want,))
    # Both compute (code - zero point) * float(scale) in float32.
    np.testing.assert_array_equal(outputs[0], want)


_ZERO_POINT_DTYPES = {"int32 zero point": torch.int32,
                      "int32 zero point input": torch.int32,
                      "float32 zero point": torch.float32,
                      "float64 zero point": torch.float64}


@pytest.mark.parametrize("case", ["float32 scale", "int32 zero point",
                                  "float32 scale input",
                                  "int32 zero point input",
                                  "float32 zero point", "float64 zero point"])
def test_portable_dequantize_widens_its_qparams(runner, tmp_path, case):
    """The same dequantize with a float32 scale, an int32 zero point or a
    float zero point holding an integer, which torch's op reads with .item()
    as readily as float64 and int64 ones, but ExecuTorch's Tensor_out kernel
    aborts the process on (op_dequantize.cpp checks both dtypes with
    ET_CHECK_MSG): static_qdq.widen_portable_zero_points casts them to
    float64 and int64 in front of it, exactly, folded into new constants
    where they are program constants and kept for run time where they are
    inputs. The program loads and runs, and computes torch's bits."""
    import executorch.kernels.quantized  # noqa: F401  (as in the test above)
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    torch.manual_seed(108)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    codes[0, :2] = torch.tensor([-128, 127])
    scale = torch.tensor([0.0123], dtype=(torch.float32 if "scale" in case
                                          else torch.float64))
    zero_point = torch.tensor([3], dtype=_ZERO_POINT_DTYPES.get(case,
                                                                torch.int64))
    x = torch.randn(4, 8)
    if case.endswith("input"):
        model, inputs = DequantizeTensorArgs(), (x, codes, scale, zero_point)
    else:
        model = DequantizeConstantCodes(codes, scale, zero_point)
        inputs = (x,)
    want = model(*inputs).numpy()
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, inputs, (want,))
    np.testing.assert_array_equal(outputs[0], want)
    ops = ops_of_program(program)
    assert "quantized_decomposed::dequantize_per_tensor.Tensor_out" in ops, ops
    if case.endswith("input"):
        assert "dim_order_ops::_to_dim_order_copy.out" in ops, ops


@pytest.mark.parametrize("case", ["float64 zero point 2^24",
                                  "float32 zero point 2^31 - 128",
                                  "float64 zero point -2^31 + 128"])
def test_portable_dequantize_widens_a_float_zero_point_to_its_edges(
        runner, tmp_path, case):
    """The largest float zero points the widening takes: integers float32
    holds exactly whose difference from every int8 code is still an int32
    (-128 - z and 127 - z, at INT32_MIN and INT32_MAX for the two outer ones).
    The kernel's int32 subtraction then rounds to the float32 torch's own
    subtraction gives, so the program computes torch's bits."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass

    torch.manual_seed(110)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    codes[0, :2] = torch.tensor([-128, 127])
    value = {"float64 zero point 2^24": 2.0 ** 24,
             "float32 zero point 2^31 - 128": 2.0 ** 31 - 128,
             "float64 zero point -2^31 + 128": -(2.0 ** 31) + 128}[case]
    dtype = torch.float32 if case.startswith("float32") else torch.float64
    scale = torch.tensor([2.0 ** -31 if "31" in case else 2.0 ** -24],
                         dtype=torch.float64)
    model = DequantizeConstantCodes(codes, scale,
                                    torch.tensor([value], dtype=dtype))
    x = torch.randn(4, 8)
    want = model(x).numpy()
    edge = lower_esp32(torch.export.export(model, (x,)), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, (x,), (want,))
    np.testing.assert_array_equal(outputs[0], want)


@pytest.mark.parametrize("case", ["float32 zero point input",
                                  "fractional float32 zero point",
                                  "float64 zero point 2^64",
                                  "float64 zero point 2^32",
                                  "float32 zero point 2^31",
                                  "float64 zero point 2^24 + 1"])
def test_portable_dequantize_refuses_a_float_zero_point_it_cannot_widen(case):
    """A float zero point that is a program input, or a constant that is not
    an integer, has no int64 that is exactly its value, and ExecuTorch's
    Tensor_out kernel takes nothing else (it aborts the process on a float
    one): the export refuses it by name (static_qdq.refuse_portable_scales,
    per-tensor-dequant-dtype) instead of emitting that program. So does an
    integer the kernel would subtract differently from torch's float32 op:
    2^64, whose int64 cast saturates (at code 0 torch gives -1.0, the
    kernel 5.4e-20); 2^32, which the kernel narrows to an int32 0 (torch
    -1.0, the kernel 0.0); 2^31, which narrows to INT32_MIN; and 2^24 + 1,
    which torch's float32 subtraction takes as 2^24. An int64 bound alone
    passes all four but the first."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    torch.manual_seed(109)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    codes[0, 0] = 0
    x = torch.randn(4, 8)
    if case.endswith("input"):
        scale = torch.tensor([0.0123], dtype=torch.float64)
        model = DequantizeTensorArgs()
        inputs = (x, codes, scale, torch.tensor([3.0]))
    else:
        value, step = {"fractional float32 zero point": (2.5, 0.0123),
                       "float64 zero point 2^64": (2.0 ** 64, 2.0 ** -64),
                       "float64 zero point 2^32": (2.0 ** 32, 2.0 ** -32),
                       "float32 zero point 2^31": (2.0 ** 31, 2.0 ** -31),
                       "float64 zero point 2^24 + 1": (2.0 ** 24 + 1, 2.0 ** -24),
                       }[case]
        dtype = torch.float64 if case.startswith("float64") else torch.float32
        model = DequantizeConstantCodes(
            codes, torch.tensor([step], dtype=torch.float64),
            torch.tensor([value], dtype=dtype))
        inputs = (x,)
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "per-tensor-dequant-dtype", str(why.value)


class DequantizeScalarQparams(nn.Module):
    """x plus int8 codes (an input) dequantised by a scalar scale and zero
    point: the default overload, as a static pair left portable carries
    them."""

    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale, self.zero_point = scale, zero_point

    def forward(self, x, codes):
        return x + torch.ops.quantized_decomposed.dequantize_per_tensor(
            codes, self.scale, self.zero_point, -128, 127, torch.int8)


class DequantizePerChannel(nn.Module):
    """x plus constant int8 codes dequantised per channel along axis 0."""

    def __init__(self, codes, scales, zero_points):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)

    def forward(self, x):
        return x + torch.ops.quantized_decomposed.dequantize_per_channel(
            self.codes, self.scales, self.zero_points, 0, -128, 127, torch.int8)


def _integer_zero_point_program(case, codes, x):
    """(model, inputs) for an integer zero point case: "int32 ..." and "int64
    ..." a one-element constant of that dtype (tensor qparams), "scalar ..."
    a Python int (scalar qparams), "per-channel int32/int64 ..." one per row
    of the codes; the value and its scale from the case."""
    value, step = {"2^24 + 1": (2 ** 24 + 1, 2.0 ** -24),
                   "2^24": (2 ** 24, 2.0 ** -24),
                   "2^31": (2 ** 31, 2.0 ** -31),
                   "2^32": (2 ** 32, 2.0 ** -32),
                   "-2^31 + 128": (-(2 ** 31) + 128, 2.0 ** -31),
                   "1000": (1000, 2.0 ** -10),
                   "128": (128, 2.0 ** -7),
                   "127": (127, 2.0 ** -7),
                   "-128": (-128, 2.0 ** -7)}[case.split("zero point ")[1]]
    if case.startswith("scalar"):
        return DequantizeScalarQparams(step, value), (x, codes)
    dtype = torch.int32 if " int32 " in f" {case} " else torch.int64
    if case.startswith("per-channel"):
        rows = codes.shape[0]
        return DequantizePerChannel(codes, torch.full((rows,), step, dtype=torch.float64),
                                    torch.full((rows,), value, dtype=dtype)), (x,)
    return DequantizeConstantCodes(codes, torch.tensor([step], dtype=torch.float64),
                                   torch.tensor([value], dtype=dtype)), (x,)


@pytest.mark.parametrize("case", ["int32 zero point 2^24 + 1",
                                  "int64 zero point 2^31",
                                  "int64 zero point 2^32",
                                  "scalar zero point 2^24 + 1",
                                  "scalar zero point 2^32"])
def test_portable_dequantize_refuses_an_integer_zero_point_it_cannot_subtract(case):
    """An integer zero point the export knows (a constant, or the scalar of a
    static pair) follows the float one's rule per tensor. Torch reads it as a
    Python int and subtracts it in float32, as fl32(z); ExecuTorch's kernel
    narrows it to int32 and subtracts in int32. 2^24 + 1 is 2^24 to torch,
    and from 2^31 on, c - z leaves int32. The export refuses each by name
    (dequant-zero-point-inexact) instead of widening it (int32) or passing it
    on (int64, scalar). Per channel the rule is the zero point's range
    (test_portable_dequantize_per_channel_refuses_a_zero_point_outside_its_range)."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    torch.manual_seed(111)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    model, inputs = _integer_zero_point_program(case, codes, torch.randn(4, 8))
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "dequant-zero-point-inexact", str(why.value)


@pytest.mark.parametrize("case", ["int32 zero point 2^24",
                                  "scalar zero point -2^31 + 128",
                                  "per-channel int64 zero point 127",
                                  "per-channel int32 zero point -128"])
def test_portable_dequantize_takes_an_integer_zero_point_to_its_edges(
        runner, tmp_path, case):
    """The largest integer zero points the rules take: per tensor 2^24 (an
    int32 constant, widened to int64) and -2^31 + 128 (a scalar), whose
    difference from every int8 code is still an int32 that float32 rounds as
    torch's subtraction does; per channel the ends of the op's [-128, 127].
    The program runs and computes torch's bits."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass

    torch.manual_seed(112)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    codes[0, :2] = torch.tensor([-128, 127])
    model, inputs = _integer_zero_point_program(case, codes, torch.randn(4, 8))
    want = model(*inputs).numpy()
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, inputs, (want,))
    np.testing.assert_array_equal(outputs[0], want)


@pytest.mark.parametrize("case", ["per-channel int64 zero point 128",
                                  "per-channel int64 zero point 1000",
                                  "per-channel int64 zero point 2^32",
                                  "per-channel int32 zero point 2^24 + 1"])
def test_portable_dequantize_per_channel_refuses_a_zero_point_outside_its_range(case):
    """Per channel, torch subtracts the zero points from the codes as integers
    and takes any; ExecuTorch's kernel, where it takes its int8 -> float32
    path (dequantize_optimized, on every target), checks quant_min <= z <=
    quant_max and aborts the process on one outside (z = 128 with [-128,
    127]: exit -6). The export refuses a per-channel zero point it knows to
    lie outside the op's range by name (per-channel-zero-point-range), the
    zero points 128 and 1000 as well as 2^32 and 2^24 + 1, which an int32
    constant keeps unwidened."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    torch.manual_seed(115)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    model, inputs = _integer_zero_point_program(case, codes, torch.randn(4, 8))
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "per-channel-zero-point-range", str(why.value)


class DequantizeChannelsInt32(nn.Module):
    """int32 codes given at run time dequantised per channel along axis 0
    over int32's whole range, with a constant zero point per channel."""

    def __init__(self, zero_points):
        super().__init__()
        self.register_buffer("scales", torch.full_like(zero_points, 2.0 ** -20,
                                                       dtype=torch.float32))
        self.register_buffer("zero_points", zero_points)

    def forward(self, codes):
        return torch.ops.quantized_decomposed.dequantize_per_channel(
            codes, self.scales, self.zero_points, 0, -(2 ** 31), 2 ** 31 - 1,
            torch.int32)


def test_portable_dequantize_per_channel_refuses_a_zero_point_32_bit_codes_overflow():
    """Within its range, a nonzero per-channel zero point with int32 codes
    still leaves some c - z outside int32, which the kernel's int32
    subtraction overflows where torch's integer one does not: refused by name
    (dequant-zero-point-inexact)."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    model = DequantizeChannelsInt32(torch.full((2,), 5, dtype=torch.int64))
    codes = torch.tensor([[-(2 ** 31), 7], [2 ** 31 - 1, -3]], dtype=torch.int32)
    edge = lower_esp32(torch.export.export(model, (codes,)), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "dequant-zero-point-inexact", str(why.value)


class DequantizeUnsigned(nn.Module):
    """x plus constant uint8 or uint16 codes dequantised by tensor qparams,
    over their whole range."""

    def __init__(self, codes, scale, zero_point):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.q = (0, 255 if codes.dtype == torch.uint8 else 65535, codes.dtype)

    def forward(self, x):
        return x + torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            self.codes, self.scale, self.zero_point, *self.q)


@pytest.mark.parametrize("case", ["float32 zero point 2^31, uint8 codes",
                                  "float32 zero point 2^31, uint16 codes",
                                  "int64 zero point 2^31, uint8 codes"])
def test_portable_dequantize_refuses_a_zero_point_past_int32(case):
    """With unsigned codes a zero point of 2^31 leaves every c - z inside int32
    (from -2^31 up), but not z itself: the kernel narrows it to INT32_MIN
    before it subtracts, and c - INT32_MIN overflows int32 for every code
    (undefined behaviour, which UBSan reports, though the host's wrap happens
    to give torch's bits). The export refuses it by name: a float zero point
    keeps its float dtype (per-tensor-dequant-dtype), an integer one is named
    (dequant-zero-point-inexact)."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    dtype = torch.uint16 if "uint16" in case else torch.uint8
    codes = torch.tensor([[0, 1, 2, 3, 128, 254, 255, 7]], dtype=torch.int32).to(dtype)
    zero = (torch.tensor([2.0 ** 31]) if case.startswith("float32")
            else torch.tensor([2 ** 31], dtype=torch.int64))
    model = DequantizeUnsigned(codes, torch.tensor([2.0 ** -31], dtype=torch.float64),
                               zero)
    x = torch.zeros(1, 8)
    edge = lower_esp32(torch.export.export(model, (x,)), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    want = ("per-tensor-dequant-dtype" if case.startswith("float32")
            else "dequant-zero-point-inexact")
    assert why.value.rule == want, str(why.value)


class DequantizeTo(nn.Module):
    """A dequantize of int8 codes given at run time, asking for an output
    dtype: per tensor with tensor qparams ("tensor"), with scalar ones
    ("scalar"), or per channel along axis 0 ("channel", its scales of dtype
    `scales`)."""

    def __init__(self, kind, out_dtype, rows=4, zero_point=3, scale=0.0123,
                 scales=torch.float64):
        super().__init__()
        self.kind, self.out_dtype = kind, out_dtype
        self.qparams = (scale, zero_point)
        self.register_buffer("scale", torch.tensor([scale], dtype=torch.float64))
        self.register_buffer("zero_point", torch.tensor([zero_point], dtype=torch.int64))
        self.register_buffer("scales", torch.full((rows,), scale, dtype=scales))
        self.register_buffer("zero_points",
                             torch.full((rows,), zero_point, dtype=torch.int64))

    def forward(self, codes):
        qd = torch.ops.quantized_decomposed
        if self.kind == "tensor":
            return qd.dequantize_per_tensor.tensor(
                codes, self.scale, self.zero_point, -128, 127, torch.int8,
                out_dtype=self.out_dtype)
        if self.kind == "scalar":
            return qd.dequantize_per_tensor.default(
                codes, *self.qparams, -128, 127, torch.int8,
                out_dtype=self.out_dtype)
        return qd.dequantize_per_channel(
            codes, self.scales, self.zero_points, 0, -128, 127, torch.int8,
            out_dtype=self.out_dtype)


_FL32_0_0123 = float(np.float32(0.0123))  # 0.0123 as float32 holds it


@pytest.mark.parametrize("case", ["tensor float64", "scalar float64",
                                  "channel float64",
                                  "tensor float16 zero point 2^24",
                                  "channel float16 float64 scales",
                                  "tensor bfloat16"])
def test_portable_dequantize_refuses_an_output_it_does_not_compute(case):
    """Per tensor, torch's dequantize computes in float32 and rounds to the
    output dtype after each step; per channel it multiplies in the scales'
    dtype and rounds once; ExecuTorch's kernels compute in float32 and cast.
    A float64 output already differs at zero point 3 (torch -0.0492, the
    kernel -0.04919999838); a float16 one at zero point 2^24 overflows to
    -inf in torch where the kernel gives -1.0; a float16 one per channel with
    float64 scales, even ones float32 holds, rounds the exact product once in
    torch and through float32 in the kernel; a bfloat16 one makes the kernel
    abort. Each is refused by name (dequant-out-dtype)."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    kind, out = case.split()[:2]
    options = {"tensor float16 zero point 2^24": dict(zero_point=2 ** 24,
                                                      scale=2.0 ** -24),
               "channel float16 float64 scales": dict(scale=_FL32_0_0123),
               }.get(case, {})
    torch.manual_seed(113)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    model = DequantizeTo(kind, getattr(torch, out), **options)
    edge = lower_esp32(torch.export.export(model, (codes,)), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "dequant-out-dtype", str(why.value)


@pytest.mark.parametrize("case", ["tensor zero point 3", "scalar zero point 3",
                                  "tensor zero point -128",
                                  "channel float32 scales"])
def test_portable_dequantize_takes_a_float16_output_it_computes_exactly(
        runner, tmp_path, case):
    """PT2E's convert asks every dequantize of a float16 model for a float16
    output. With 8-bit codes whose every c - z float16 holds (|c - z| <=
    2048), each of torch's float32 steps is exact up to the product, which
    torch and the kernel both round from float32 to float16; per channel
    with float32 scales, too. The program runs and gives torch's bits for
    all 256 int8 codes (scale 0.0123, which float16 does not hold)."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass

    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(4, 64)
    kind = case.split()[0]
    if kind == "channel":
        model = DequantizeTo(kind, torch.float16, scales=torch.float32)
    else:
        model = DequantizeTo(kind, torch.float16, zero_point=int(case.split()[-1]))
    want = model(codes).numpy()
    assert want.dtype == np.float16
    edge = lower_esp32(torch.export.export(model, (codes,)), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, (codes,), (want,))
    np.testing.assert_array_equal(outputs[0].view(np.uint16), want.view(np.uint16))


def test_portable_dequantize_takes_a_float32_output_asked_for(runner, tmp_path):
    """out_dtype=float32, said out loud, is the kernel's own arithmetic: the
    program runs and computes torch's bits."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass

    torch.manual_seed(114)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    model = DequantizeTo("tensor", torch.float32)
    want = model(codes).numpy()
    edge = lower_esp32(torch.export.export(model, (codes,)), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, (codes,), (want,))
    np.testing.assert_array_equal(outputs[0], want)


class DequantizeChannels(nn.Module):
    """int8 codes given at run time dequantised per channel along axis 0 by
    constant scales and zero points (None: none), or by scales given at run
    time too (`scales` None)."""

    def __init__(self, scales=None, zero_points=None):
        super().__init__()
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)

    def forward(self, codes, *scales):
        return torch.ops.quantized_decomposed.dequantize_per_channel(
            codes, scales[0] if scales else self.scales, self.zero_points, 0,
            -128, 127, torch.int8)


def _channels_program(case, codes):
    """(model, inputs) for a per-channel scales case: its scales' dtype and
    value, its zero points (3, 0 or none), given at run time or not."""
    rows = codes.shape[0]
    value = _FL32_0_0123 if "holding" in case else (0.1 if "0.1" in case else 0.0123)
    dtype = torch.float32 if case.startswith("float32") else torch.float64
    scales = torch.full((rows,), value, dtype=dtype)
    zero = None if "no zero points" in case else torch.full(
        (rows,), 0 if "zero points 0" in case else 3, dtype=torch.int64)
    if "constant codes" in case:
        return DequantizePerChannel(codes, scales, zero), (torch.randn(codes.shape),)
    if "at run time" in case:
        return DequantizeChannels(None, zero), (codes, scales)
    return DequantizeChannels(scales, zero), (codes,)


@pytest.mark.parametrize("case", ["float64 scales 0.0123, zero points 3",
                                  "float64 scales 0.0123, no zero points",
                                  "float64 scales 0.1, zero points 0",
                                  "float64 scales 0.0123, constant codes",
                                  "float64 scales 0.0123 given at run time"])
def test_portable_dequantize_per_channel_refuses_scales_float32_cannot_hold(case):
    """Per channel, torch multiplies the integer c - z by the scales in their
    own dtype and rounds to the float32 output once; ExecuTorch's kernels
    multiply by fl32(s) (get_scale), on every path. With float64 scales of
    0.0123, 89 of 256 int8 codes differ (code -124: torch -1.5621, the kernel
    -1.5620999); with 0.1 at zero point 0, 48. The export refuses float64
    scales float32 does not hold by name (per-channel-scale-inexact), and
    float64 scales known only at run time with them: nothing proves them."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    torch.manual_seed(116)
    codes = torch.randint(-128, 128, (4, 8), dtype=torch.int8)
    model, inputs = _channels_program(case, codes)
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "per-channel-scale-inexact", str(why.value)


@pytest.mark.parametrize("case", ["float32 scales 0.0123, zero points 3",
                                  "float64 scales holding fl32(0.0123), zero points 3",
                                  "float32 scales 0.0123 given at run time"])
def test_portable_dequantize_per_channel_takes_scales_float32_holds(
        runner, tmp_path, case):
    """Float32 scales, constant or given at run time, and float64 ones whose
    values float32 holds: torch's product is then the exact one, rounded to
    float32 once, as the kernel's. The program runs and computes torch's
    bits for all 256 int8 codes."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass

    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(4, 64)
    model, inputs = _channels_program(case, codes)
    want = model(*inputs).numpy()
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, inputs, (want,))
    np.testing.assert_array_equal(outputs[0].view(np.uint32), want.view(np.uint32))


class DequantizeWideChannels(nn.Module):
    """Codes of any integer dtype dequantised per channel along axis 0 over
    the dtype's whole range, by constant scales and zero points (None:
    none): codes given at run time, or held as a constant (`codes`) and
    added to x."""

    def __init__(self, scales, zero_points, dtype, codes=None):
        super().__init__()
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)
        self.register_buffer("codes", codes)
        info = torch.iinfo(dtype)
        self.q = (info.min, info.max, dtype)

    def forward(self, x):
        if self.codes is None:
            return torch.ops.quantized_decomposed.dequantize_per_channel(
                x, self.scales, self.zero_points, 0, *self.q)
        return x + torch.ops.quantized_decomposed.dequantize_per_channel(
            self.codes, self.scales, self.zero_points, 0, *self.q)


def _wide_codes_program(case):
    """(model, inputs) for a wide-codes case: its codes' dtype, constant or
    given at run time, within 2^24 or not; its scales float64 holding
    fl32(0.0123), or float32 0.0123; zero points none or 0."""
    g = torch.Generator().manual_seed(117)
    dtype = torch.int16 if case.startswith("int16") else torch.int32
    rows = 4
    reach = 2 ** 24 if "within 2^24" in case else (
        2 ** 15 if dtype == torch.int16 else 2 ** 30)
    codes = torch.randint(-reach, reach, (rows, 256), generator=g,
                          dtype=torch.int64).to(dtype)
    if "past 2^24" in case:
        codes[0, 0] = 2 ** 24 + 1      # a difference float32 rounds
    scales = (torch.full((rows,), 0.0123, dtype=torch.float32)
              if "float32 scales" in case else
              torch.full((rows,), _FL32_0_0123, dtype=torch.float64))
    zero = (torch.zeros(rows, dtype=torch.int64)
            if "zero points 0" in case else None)
    if "constant" in case:
        return (DequantizeWideChannels(scales, zero, dtype, codes),
                (torch.randn(codes.shape, generator=g),))
    return DequantizeWideChannels(scales, zero, dtype), (codes,)


@pytest.mark.parametrize("case", [
    "int32 codes, float64 scales holding fl32(0.0123), no zero points",
    "int32 codes, float64 scales holding fl32(0.0123), zero points 0",
    "int32 constant codes past 2^24, float64 scales holding fl32(0.0123)"])
def test_portable_dequantize_per_channel_refuses_float64_scales_for_wide_codes(
        case):
    """Per channel, torch subtracts the zero points from the codes as
    integers and multiplies the exact difference by a float64 scale, then
    rounds once to float32; ExecuTorch's kernels round c - z to float32
    first, then multiply by fl32(s). A float64 scale float32 holds makes the
    two agree only while every c - z is a value float32 holds, which every
    8- and 16-bit code's is and a 32-bit code's past 2^24 is not: on int32
    codes in +-2^30 at fl32(0.0123), 269 of 1024 outputs differed from
    torch's bits. The export refuses them
    by name (per-channel-scale-inexact), unless the codes are a constant
    whose every difference float32 holds."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3._passes.static_qdq import StaticPairRefused

    model, inputs = _wide_codes_program(case)
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    with pytest.raises(StaticPairRefused) as why:
        apply_passes(edge, quiet=True)
    assert why.value.rule == "per-channel-scale-inexact", str(why.value)


@pytest.mark.parametrize("case", [
    "int16 codes, float64 scales holding fl32(0.0123), no zero points",
    "int16 codes, float64 scales holding fl32(0.0123), zero points 0",
    "int32 codes, float32 scales, no zero points",
    "int32 constant codes within 2^24, float64 scales holding fl32(0.0123)"])
def test_portable_dequantize_per_channel_takes_float64_scales_where_exact(
        runner, tmp_path, case):
    """Where every c - z is a value float32 holds -- 16-bit codes, or
    constant 32-bit codes within 2^24 -- a float64 scale float32 holds gives
    torch's product, and float32 scales round c - z on both sides alike:
    the program runs and computes torch's bits for every output."""
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.memory_planning import planning_pass

    model, inputs = _wide_codes_program(case)
    want = model(*inputs).numpy()
    edge = lower_esp32(torch.export.export(model, inputs), [Esp32Partitioner()])
    apply_passes(edge, quiet=True)
    program = edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32")))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)
    outputs, _ = run_model(runner, pte, None, inputs, (want,))
    np.testing.assert_array_equal(outputs[0].view(np.uint32), want.view(np.uint32))


def assert_bits_equal(actual, expected):
    """Bit for bit, not to run_model's 2e-6: a device mirror reproduces the
    kernel's arithmetic exactly, so one ulp (or the sign of a zero) is a
    failure. The q@kT kernels' raw scores are held to bmm_qk_i8 this way:
    a change of their order of operations, e.g. acc * (sx * sk) for
    (acc * sx) * sk, moves a score by an ulp and stays inside 2e-6."""
    actual = np.ascontiguousarray(actual, dtype=np.float32)
    expected = np.ascontiguousarray(expected, dtype=np.float32)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    differ = actual.view(np.uint32) != expected.view(np.uint32)
    assert not differ.any(), (
        f"{int(differ.sum())} of {differ.size} values differ from the mirror; "
        f"first at {np.argwhere(differ)[0].tolist()}: "
        f"{actual[differ][0]!r} vs {expected[differ][0]!r}")


class QK(nn.Module):
    def forward(self, query, key):
        scores = torch.bmm(query, key.transpose(1, 2))
        # The opt-in gate only claims pre-softmax Q@K^T. Also return
        # its raw scores so the integer result has a direct output oracle.
        return scores, torch.softmax(scores, dim=-1)


def test_dynamic_int8_qk(runner, tmp_path):
    torch.manual_seed(103)
    query = torch.randn(2, 3, 17)
    key = torch.randn(2, 5, 17) * 0.25
    query[0, 0].zero_()
    key[1, 2].zero_()
    reference = np.stack([bmm_qk_i8(q.numpy(), k.numpy()) for q, k in zip(query, key)])
    probabilities = softmax_f32_portable(reference)
    pte, ptd, descriptors = export_model(
        QK(), (query, key), tmp_path, quantize_qk=True)
    assert len(descriptors) == 1 and descriptors[0]["opcode"] == OP_BMM_DYN_I8
    assert descriptors[0]["scale_group"] == 2  # Serialized batch count.
    baseline = None
    for flags in MODES:
        outputs, log = run_model(runner, pte, ptd, (query, key),
                                 (reference, probabilities), flags)
        assert "esp32 bmmqk: 1 dynamic int8 bmm" in log, log
        assert_bits_equal(outputs[0], reference)
        if baseline is None:
            baseline = outputs
        else:
            for output, previous in zip(outputs, baseline):
                np.testing.assert_array_equal(output, previous)


class KCacheQK(nn.Module):
    """QK with its keys arriving as an int8 cache: codes plus per-row scales."""

    def forward(self, query, codes, scales):
        from executorch.backends.esp32s3.ops.kcache_i8 import dequantized_keys

        scores = torch.bmm(query, dequantized_keys(codes, scales).transpose(1, 2))
        return scores, torch.softmax(scores, dim=-1)


@pytest.mark.parametrize("width", [17, 32])
def test_int8_kcache_qk(runner, tmp_path, width):
    # Keys quantised once by quantize_rows_i8 must score exactly as the dynamic
    # op scores the fp32 keys they came from: the same bmm_qk_i8 reference, and
    # the same bits as OP_BMM_DYN_I8 on this runner. Width 17 rows are ragged
    # (staged, zero-padded); width 32 rows are read in place from the codes.
    from executorch.backends.esp32s3.ops.kcache_i8 import quantize_rows_i8

    torch.manual_seed(103)
    query = torch.randn(2, 3, width)
    key = torch.randn(2, 5, width) * 0.25
    query[0, 0].zero_()
    key[1, 2].zero_()
    codes, scales = quantize_rows_i8(key)
    reference = np.stack([bmm_qk_i8(q.numpy(), k.numpy()) for q, k in zip(query, key)])
    probabilities = softmax_f32_portable(reference)
    (tmp_path / "cached").mkdir()
    (tmp_path / "dynamic").mkdir()
    pte, ptd, descriptors = export_model(
        KCacheQK(), (query, codes, scales), tmp_path / "cached", quantize_qk=True)
    assert len(descriptors) == 1 and descriptors[0]["opcode"] == OP_BMM_KCACHE_I8
    assert descriptors[0]["scale_group"] == 2  # Serialized batch count.
    dyn_pte, dyn_ptd, _ = export_model(
        QK(), (query, key), tmp_path / "dynamic", quantize_qk=True)
    dynamic, _ = run_model(runner, dyn_pte, dyn_ptd, (query, key),
                           (reference, probabilities))
    assert_bits_equal(dynamic[0], reference)
    for flags in MODES:
        outputs, log = run_model(runner, pte, ptd, (query, codes, scales),
                                 (reference, probabilities), flags)
        assert "esp32 bmmkc8: 1 int8-K-cache bmm op(s), 10 cached key row" in log, log
        assert_bits_equal(outputs[0], reference)
        for output, previous in zip(outputs, dynamic):
            np.testing.assert_array_equal(output, previous)


class GQAScores(nn.Module):
    """4-D attention scores, [B,Hq,S,D] queries against [B,Hkv,S,D] keys made
    [B,Hq,S,D] by transformers' repeat_kv (expand, then reshape), returned
    raw beside their softmax. QkTo3D collapses the q@kT to one 3-D
    OP_BMM_DYN_I8, whose batch b*Hq + h must read KV head h // (Hq/Hkv)."""

    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def forward(self, query, key):
        batch, kv_heads, length, width = key.shape
        if self.repeats > 1:
            key = (key[:, :, None, :, :]
                   .expand(batch, kv_heads, self.repeats, length, width)
                   .reshape(batch, kv_heads * self.repeats, length, width))
        scores = query @ key.transpose(-2, -1)
        return scores, torch.softmax(scores, dim=-1)


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2), (4, 1)])
def test_gqa_4d_int8_qk_exact(runner, tmp_path, heads, kv_heads):
    # Multi-head, GQA and MQA q@kT written 4-D, at batch 2, held to
    # bmm_qk_i8 per (batch, query head, its KV head) bit for bit in every
    # runner mode (the nano zoo holds its own q@kT bit for bit through its
    # probe; this adds MHA, GQA and MQA at batch 2).
    from executorch.backends.esp32s3.test.test_partition_crashes import lower

    torch.manual_seed(7)
    repeats = heads // kv_heads
    query = torch.randn(2, heads, 8, 16)
    key = torch.randn(2, kv_heads, 8, 16) * 0.5
    query[1, 3, 2].zero_()
    keep = {}
    _, ops, _, err = lower(GQAScores(repeats), (query, key), keep,
                           quantize_qk=True)
    assert err is None, err
    assert sum("executorch_call_delegate" in op for op in ops) == 1, ops
    assert not any("aten.bmm" in op for op in ops), ops
    reference = np.stack([np.stack([
        bmm_qk_i8(query[b, h].numpy(), key[b, h // repeats].numpy())
        for h in range(heads)]) for b in range(2)])
    probabilities = softmax_f32_portable(reference)
    pte = tmp_path / "model.pte"
    pte.write_bytes(keep["program"].buffer)
    for flags in MODES:
        outputs, log = run_model(runner, pte, None, (query, key),
                                 (reference, probabilities), flags)
        assert "esp32 bmmqk: 1 dynamic int8 bmm" in log, log
        assert_bits_equal(outputs[0], reference)


@pytest.mark.parametrize("name", ["attnqk", "attnkc8"])
def test_parity_attention_rows_print_raw_scores(runner, tmp_path, name):
    # run_parity holds the rows attnqk_qk and attnkc8_qk to
    # models.attnqk_qk_scores bit for bit: their models (AttnQKScores,
    # AttnKC8Scores) put the delegate's raw q@kT scores first, the output
    # main.cpp prints `output bits` for. That reference fuses the Linears'
    # epilogue (madd.s, as QEMU and the chip do); this runner does not
    # (-ffp-contract=off). So here the same models and lowering, against the
    # same mirrors with the host's unfused epilogue: output 0 is bmm_qk_i8
    # over the Linears' outputs, exactly, and the delegate is AttnQK's or
    # AttnKC8's descriptor for descriptor, so main.cpp's blob fixtures on
    # these rows still patch the blob they were written for.
    from executorch.backends.esp32s3.quantizer import mirrors
    from executorch.backends.esp32s3.serialization.weight_formats import (
        quantize_absmax_rows,
    )
    from executorch.backends.esp32s3.test import models
    from executorch.backends.esp32s3.test.reference import (
        bmm_f32_portable,
        dense_act,
    )

    factory, shape = models.SPECS[name]
    scored = {"attnqk": models.AttnQKScores,
              "attnkc8": models.AttnKC8Scores}[name]
    torch.manual_seed(0)
    base = factory().eval()
    torch.manual_seed(0)
    model = scored().eval()
    x = models.ramp_input(shape)

    def lower(module):
        # export_delegated.export_qk's lowering; AttnKC8 may take more than
        # one partition (the cache writer is portable), so every payload.
        edge = lower_esp32(torch.export.export(module, (x,)),
                           [Esp32Partitioner(quantize_qk=True)])
        found = []
        for _, _, blob in get_delegated_payload(
                edge.exported_program().graph_module).values():
            _, _, count, _ = struct.unpack_from(BLOB_HEADER_FORMAT, blob)
            found.append([struct.unpack_from(
                OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES + i * OP_DESC_BYTES)
                for i in range(count)])
        return edge, found

    _, base_descriptors = lower(base)
    edge, descriptors = lower(model)
    assert descriptors == base_descriptors
    pte, ptd = tmp_path / "model.pte", None
    pte.write_bytes(edge.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16)).buffer)

    x2d = x.reshape(-1, model.d).numpy()

    def linear(module):
        codes, scales = quantize_absmax_rows(module.weight.detach().numpy(), "i8")
        return mirrors.affine_matmul(
            x2d, codes, scales, module.bias.detach().numpy(),
            act=dense_act(127), fused_madd=False, check_bound=False).y.reshape(
                x.shape[0], x.shape[1], model.d)

    q, k, v = linear(model.wq), linear(model.wk), linear(model.wv)
    scores = np.stack([bmm_qk_i8(q[b], k[b]) for b in range(x.shape[0])])
    probabilities = softmax_f32_portable(
        (scores / np.float32(model.d ** 0.5)).astype(np.float32))
    attention = bmm_f32_portable(probabilities, v)
    for flags in MODES:
        outputs, _ = run_model(runner, pte, ptd, (x,), (scores, attention), flags)
        assert_bits_equal(outputs[0], scores)


class SdpaScores(nn.Module):
    """F.scaled_dot_product_attention on raw q, k and v (no projections, so
    the delegate's only op is the q@kT and its operands are exactly the
    decomposition's scaled rows)."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, query, key, value):
        if self.mode == "gqa":
            return F.scaled_dot_product_attention(query, key, value,
                                                  enable_gqa=True)
        if self.mode == "causal":
            return F.scaled_dot_product_attention(query, key, value,
                                                  is_causal=True)
        if self.mode == "scale":
            return F.scaled_dot_product_attention(query, key, value,
                                                  scale=0.3)
        return F.scaled_dot_product_attention(query, key, value)


def _expose_delegate_outputs(edge):
    """Append each delegate call's outputs to the program's user outputs,
    after partitioning, so what was claimed does not change. Returns the
    appended nodes' values (their shapes), in output order."""
    from torch.export.graph_signature import (
        OutputKind,
        OutputSpec,
        TensorArgument,
    )

    program = edge.exported_program()
    graph = program.graph_module.graph
    output = next(n for n in graph.nodes if n.op == "output")
    outs = list(output.args[0])
    added = []
    for node in list(graph.nodes):
        if (node.op != "call_function"
                or "executorch_call_delegate" not in str(node.target)):
            continue
        for item in sorted(node.users, key=lambda n: n.args[1]):
            outs.append(item)
            program.graph_signature.output_specs.append(OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=item.name), target=None))
            added.append(item.meta["val"])
    output.args = (tuple(outs),)
    program.graph_module.recompile()
    return added


@pytest.mark.parametrize("mode", ["plain", "causal", "gqa", "scale"])
def test_sdpa_int8_qk_scaled_rows_exact(runner, tmp_path, mode):
    # SdpaToQk's mode: under quantize_qk an SDPA-spelled q@kT quantises the
    # rows of q*sqrt(scale) and k*sqrt(scale) (the decomposition's multiply),
    # not those of q and k. reference.sdpa_qk_i8 is that arithmetic; the
    # delegate's raw scores (exposed as a program output) must equal it bit
    # for bit, and differ from bmm_qk_i8(q, k) * scale, the unscaled-rows
    # mirror's answer.
    from executorch.backends.esp32s3._passes import apply_passes
    from executorch.backends.esp32s3.test.reference import bmm_f32_portable

    # Head width 32: sqrt(1/sqrt(32)) is not a power of two. At 16 (or 256)
    # it is (0.5, 0.25), the scaled rows' codes are the unscaled rows' and
    # each score is exactly bmm_qk_i8(q, k) * scale, so the two mirrors
    # cannot be told apart there.
    torch.manual_seed(211)
    heads, kv_heads, width = 4, (2 if mode == "gqa" else 4), 32
    query = torch.randn(1, heads, 8, width)
    key = torch.randn(1, kv_heads, 8, width) * 0.5
    value = torch.randn(1, kv_heads, 8, width)
    query[0, 1, 3].zero_()
    model = SdpaScores(mode).eval()
    edge = lower_esp32(torch.export.export(model, (query, key, value)),
                       [Esp32Partitioner(quantize_qk=True)])
    apply_passes(edge, quiet=True)
    exposed = _expose_delegate_outputs(edge)
    assert len(exposed) == 1, exposed
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16))
    pte = tmp_path / "model.pte"
    pte.write_bytes(program.buffer)

    repeats = heads // kv_heads
    keys = key.repeat_interleave(repeats, dim=1).numpy()
    values = value.repeat_interleave(repeats, dim=1).numpy()
    scale = 0.3 if mode == "scale" else None
    scores = sdpa_qk_i8(query.numpy(), keys, scale)          # [1, H, 8, 8]
    masked = scores
    if mode == "causal":
        keep = np.tril(np.ones((8, 8), dtype=bool))
        masked = np.where(keep, scores, np.float32(-np.inf)).astype(np.float32)
    probabilities = softmax_f32_portable(masked)
    attention = bmm_f32_portable(probabilities.reshape(heads, 8, 8),
                                 values.reshape(heads, 8, width)).reshape(
                                     query.shape)
    want = scores.reshape(tuple(exposed[0].shape))
    for flags in MODES:
        outputs, log = run_model(runner, pte, None, (query, key, value),
                                 (attention, want), flags)
        assert "esp32 bmmqk: 1 dynamic int8 bmm" in log, log
        assert_bits_equal(outputs[1], want)
    # The unscaled-rows spelling is other arithmetic: it must not pass.
    factor = (0.3 if mode == "scale" else width ** -0.5)
    unscaled = np.stack([bmm_qk_i8(query[0, h].numpy(), keys[0, h])
                         for h in range(heads)]) * np.float32(factor)
    assert not np.array_equal(unscaled.reshape(want.shape).view(np.uint32),
                              want.view(np.uint32))


class CustomOps(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("rms_weight", torch.linspace(0.5, 1.5, 7))
        self.register_buffer("cos", torch.randn(3, 4))
        self.register_buffer("sin", torch.randn(3, 4))
        self.register_buffer("ln_weight", torch.linspace(0.75, 1.25, 4))
        self.register_buffer("ln_bias", torch.linspace(-0.3, 0.3, 4))

    def forward(self, value, image):
        return (
            torch.ops.espressif.rmsnorm(value, self.rms_weight, 1e-5),
            torch.ops.espressif.rmsnorm_noscale(value, 1e-5),
            torch.ops.espressif.rope(value, self.cos, self.sin, 4),
            torch.ops.espressif.layer_norm2d(image, self.ln_weight, self.ln_bias, 1e-5),
            torch.relu(image),  # Also prove upstream portable-op registration.
        )


def test_espressif_custom_ops(runner, tmp_path):
    # The same Python op definitions used by the fusion passes and QEMU runner.
    from executorch.backends.esp32s3._passes import fuse_layer_norm2d_pass  # noqa: F401
    from executorch.backends.esp32s3._passes import fuse_rmsnorm_pass  # noqa: F401
    from executorch.backends.esp32s3._passes import fuse_rope_pass  # noqa: F401
    from executorch.backends.esp32s3.test.reference import rmsnorm_f32_s3_reference

    torch.manual_seed(104)
    model = CustomOps().eval()
    value = torch.randn(2, 3, 7)  # Odd RMS width; partial rotary keeps the tail.
    image = torch.randn(2, 4, 2, 3)
    assert sum(p.numel() for p in model.parameters()) <= 10_000
    assert value.numel() + image.numel() <= 4096
    with torch.no_grad():
        expected = [output.numpy() for output in model(value, image)]
    expected[0] = rmsnorm_f32_s3_reference(value.numpy(), model.rms_weight.numpy(), 1e-5)
    expected[1] = rmsnorm_f32_s3_reference(value.numpy(), None, 1e-5)
    exported = torch.export.export(model, (value, image))
    edge = to_edge(exported, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    targets = {str(node.target) for node in edge.exported_program().graph.nodes
               if node.op == "call_function"}
    assert all(any(f"espressif.{name}." in target for target in targets)
               for name in ("rmsnorm", "rmsnorm_noscale", "rope", "layer_norm2d")), targets
    program = edge.to_executorch()
    pte = tmp_path / "custom_ops.pte"
    pte.write_bytes(program.buffer)
    for flags in ((), ("--force-scalar",)):
        run_model(runner, pte, None, (value, image), expected, flags)


class LayerNorm2dSource(nn.Module):
    """LayerNorm2d as SAM2 and ConvNeXt write it: the graph
    fuse_layer_norm2d_pass replaces."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.75, 1.25, channels))
        self.bias = nn.Parameter(torch.linspace(-0.3, 0.3, channels))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + 1e-6)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


@pytest.mark.parametrize("channels", [3, 5, 12, 16])
def test_layer_norm2d_kernel_is_the_unfused_graph(runner, tmp_path, channels):
    """espressif::layer_norm2d gives the bits of the fourteen portable ops it
    replaces, at channel counts that are and are not powers of two, on columns
    whose channels are equal or nearly so: their variance is ~0, the output
    divides x - mean by ~sqrt(eps), and an ulp of the mean shows. At C = 3
    a kernel that multiplies each sum by 1/C where the graph divides is off."""
    from executorch.backends.esp32s3._passes.fuse_layer_norm2d_pass import (
        fuse_layer_norm2d,
    )

    torch.manual_seed(105)
    model = LayerNorm2dSource(channels).eval()
    image = torch.randn(1, channels, 5, 6)
    image[0, :, 0, :] = image[0, 0, 0, :]
    image[0, :, 1, :] = image[0, 0, 1, :] + 1e-4 * torch.randn(channels, 6)
    image.numpy().tofile(tmp_path / "image.bin")
    outputs = {}
    for fused in (False, True):
        edge = to_edge(torch.export.export(model, (image,)))
        if fused:
            assert fuse_layer_norm2d(edge).fused == 1
        pte = tmp_path / f"ln2d_{'fused' if fused else 'unfused'}.pte"
        pte.write_bytes(edge.to_executorch().buffer)
        out = tmp_path / f"{pte.stem}.bin"
        result = subprocess.run(
            [str(runner), "--pte", str(pte), "--input", str(tmp_path / "image.bin"),
             "--output", str(out)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[fused] = np.fromfile(out, dtype=np.float32)
    differ = np.flatnonzero(outputs[False].view(np.uint32) != outputs[True].view(np.uint32))
    assert differ.size == 0, (
        f"{differ.size} of {outputs[False].size} differ, max abs "
        f"{np.abs(outputs[False] - outputs[True]).max():.3g}")


class SharedCache(nn.Module):
    """The story program's state in miniature: rows written at a position into
    a mutable buffer that both methods share (mem-id-2)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(8, 2))

    def forward(self, rows, pos):
        self.cache.index_copy_(0, pos + torch.arange(rows.shape[0]), rows)
        return self.cache * 1.0


@pytest.mark.parametrize("reinplace", [False, True])
def test_script_methods_share_state(runner, tmp_path, reinplace):
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program
    from executorch.backends.esp32s3.utils.multimethod import (
        combine_shared_methods,
        shared_memory_planning_pass,
    )

    methods = {name: to_edge(torch.export.export(
        SharedCache(), (torch.zeros(count, 2), torch.tensor(0))))
        for name, count in (("prefill", 3), ("decode", 1))}
    program = combine_shared_methods(methods).to_executorch(ExecutorchBackendConfig(
        memory_planning_pass=shared_memory_planning_pass(), run_reinplace_pass=reinplace))
    # In place, the cache write is index_put_ on the buffer itself; out of place
    # it is index_put into a new tensor and a copy back.
    assert ("aten::index_put_" in ops_of_program(program)) == reinplace
    pte = tmp_path / "shared.pte"
    pte.write_bytes(program.buffer)

    # Prefill rows 0-2, decode rows 3 and 4, prefill rows 5-7: every step must
    # see all earlier writes, across both changes of method.
    cache = np.zeros((8, 2), np.float32)
    lines, expected = [], []
    for index, (name, start, count) in enumerate(
            (("prefill", 0, 3), ("decode", 3, 1), ("decode", 4, 1), ("prefill", 5, 3))):
        rows = np.arange(2 * count, dtype=np.float32).reshape(count, 2) + 10 * (index + 1)
        rows.tofile(tmp_path / f"rows{index}.bin")
        np.asarray(start, dtype=np.int64).tofile(tmp_path / f"pos{index}.bin")
        cache[start:start + count] = rows
        expected.append(cache.copy())
        lines.append(f"{name} {tmp_path}/rows{index}.bin {tmp_path}/pos{index}.bin"
                     f" -> {tmp_path}/out{index}.bin  # step {index}")
    script = tmp_path / "steps.txt"
    script.write_text("# two methods, one cache\n\n" + "\n".join(lines) + "\n")
    command = [str(runner), "--pte", str(pte), "--script", str(script)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    for index, state in enumerate(expected):
        actual = np.fromfile(tmp_path / f"out{index}.bin", dtype=np.float32)
        np.testing.assert_array_equal(actual.reshape(8, 2), state)

    # Malformed lines and wrong file counts fail before any step runs.
    for line, message in ((f"decode {tmp_path}/rows1.bin {tmp_path}/pos1.bin",
                           "expected METHOD IN... -> OUT..."),
                          (f"decode {tmp_path}/rows1.bin -> {tmp_path}/bad.bin",
                           "method decode requires 2 input file(s)"),
                          (f"verify {tmp_path}/rows1.bin -> {tmp_path}/bad.bin",
                           "read method metadata for verify")):
        script.write_text(lines[0].replace("out0", "bad0") + "\n" + line + "\n")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and message in result.stderr, result.stderr
        assert not (tmp_path / "bad0.bin").exists() and not (tmp_path / "bad.bin").exists()
