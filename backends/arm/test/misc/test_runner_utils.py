# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, cast

import torch

from executorch.backends.arm.test import runner_utils


class _FakeExecutorchProgramManager:
    def __init__(self, buffer: bytes) -> None:
        self.buffer = buffer

    def exported_program(self):
        return object()


def test_run_corstone_uses_short_input_aliases_in_semihosting_cmd(
    monkeypatch, tmp_path: Path
) -> None:
    long_input_paths = [
        str(tmp_path / ("very_long_input_name_" * 6 + "0.bin")),
        str(tmp_path / ("very_long_input_name_" * 6 + "1.bin")),
    ]

    monkeypatch.setattr(
        runner_utils,
        "save_inputs_to_file",
        lambda exported_program, inputs, intermediate_path: long_input_paths,
    )

    copied_files: list[tuple[str, str]] = []

    def _fake_copyfile(src: str, dst: str) -> None:
        copied_files.append((src, dst))

    monkeypatch.setattr(runner_utils.shutil, "copyfile", _fake_copyfile)

    captured: dict[str, list[str]] = {}

    def _fake_run_cmd(cmd, check=True):
        captured["cmd"] = cmd
        return runner_utils.subprocess.CompletedProcess(
            cmd, 0, stdout=b"OK", stderr=b""
        )

    monkeypatch.setattr(runner_utils, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(
        runner_utils,
        "get_output_from_file",
        lambda exported_program, intermediate_path, output_base_name: ("sentinel",),
    )

    elf_path = tmp_path / "arm_executor_runner"
    elf_path.write_bytes(b"")

    output = runner_utils.run_corstone(
        executorch_program_manager=cast(
            Any, _FakeExecutorchProgramManager(buffer=b"pte")
        ),
        inputs=cast(Any, ()),
        intermediate_path=tmp_path,
        target_board="corstone-320",
        elf_path=elf_path,
        timeout=1,
    )

    assert output == ("sentinel",)
    assert [Path(dst).name for _, dst in copied_files] == ["i0.bin", "i1.bin"]

    semihosting_cmd_arg = next(
        arg for arg in captured["cmd"] if "semihosting-cmd_line" in arg
    )
    assert "-i i0.bin" in semihosting_cmd_arg
    assert "-i i1.bin" in semihosting_cmd_arg
    assert long_input_paths[0] not in semihosting_cmd_arg
    assert long_input_paths[1] not in semihosting_cmd_arg


def test_get_elf_path_uses_repo_root_candidates(monkeypatch, tmp_path: Path) -> None:
    elf_path = (
        tmp_path
        / "arm_test"
        / "arm_semihosting_executor_runner_corstone-300"
        / "arm_executor_runner"
    )
    elf_path.parent.mkdir(parents=True)
    elf_path.write_bytes(b"")

    monkeypatch.setattr(runner_utils, "_elf_search_roots", lambda: [tmp_path])
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    assert runner_utils.get_elf_path("corstone-300") == str(elf_path)


def test_get_elf_path_accepts_nested_runner_output(monkeypatch, tmp_path: Path) -> None:
    elf_path = (
        tmp_path
        / "arm_test"
        / "arm_semihosting_executor_runner_corstone-300"
        / "examples"
        / "arm"
        / "executor_runner"
        / "arm_executor_runner"
    )
    elf_path.parent.mkdir(parents=True)
    elf_path.write_bytes(b"")

    monkeypatch.setattr(runner_utils, "_elf_search_roots", lambda: [tmp_path])

    assert runner_utils.get_elf_path("corstone-300") == str(elf_path)


def test_quantized_decomposed_out_ops_preserve_channels_last_layout() -> None:
    x = torch.arange(1, 25, dtype=torch.float32).reshape((1, 2, 3, 4))
    x = x.to(memory_format=torch.channels_last)

    runner_utils._ensure_quantized_out_variant_registered(
        torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    runner_utils._ensure_quantized_out_variant_registered(
        torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )

    q_out = torch.empty_like(x, dtype=torch.int8)
    quantized = runner_utils._run_quantized_decomposed_out_op(
        torch.ops.quantized_decomposed.quantize_per_tensor.out,
        (x, 0.09407360851764679, -128, -128, 127, torch.int8),
        {"out": q_out},
    )

    assert quantized.dim_order() == x.dim_order()
    assert quantized.is_contiguous(memory_format=torch.channels_last)

    dq_out = torch.empty_like(x)
    dequantized = runner_utils._run_quantized_decomposed_out_op(
        torch.ops.quantized_decomposed.dequantize_per_tensor.out,
        (
            quantized,
            2.2577226161956787,
            -128,
            -128,
            127,
            torch.int8,
        ),
        {"out": dq_out},
    )

    assert dequantized.dim_order() == x.dim_order()
    assert dequantized.is_contiguous(memory_format=torch.channels_last)
