# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This backend, plugged into ExecuTorch's own backend Tester harness.

`backends/test/harness` is the shared Tester every in-tree backend subclasses
(XNNPACK, Arm, Cortex-M, CoreML, Vulkan, Qualcomm), and `backends/test/suite`
is the shared black-box battery that runs against any backend registering a
flow. test/suite/flow_esp32s3.py registers this one (test/suite/README.md
says how to run it).

The stages are the README's pipeline, so what the harness measures is what a
user ships:

- ToEdgeTransformAndLower runs `lower_esp32` (to_edge, the pre-partition
  folds, canonicalisation, the quantiser absorption, partitioning), then
  `apply_passes`. `lowering="stock"` runs ExecuTorch's own
  `to_edge_transform_and_lower` with the canonical passes as transform
  passes instead, which measures the partitioner alone.
- ToExecutorch plans memory with `planning_pass("esp32")` and aligns
  delegates and constants to 16 bytes, as the firmware loads them.
- Serialize runs the program with esp32_host_runner (test/host_runner/): the
  delegate's runtime and the operator library built for the build machine,
  with C reference kernels in place of the PIE assembly, and the portable
  kernels for whatever the delegate does not take. `run_artifact` writes the
  inputs, runs the runner and reads the outputs back, so
  `run_method_and_compare_outputs` compares numbers this backend computed with
  the eager reference. That is a host build: the device's own bits are proven
  by QEMU and silicon parity (test/parity/run_parity.py) against the host
  mirrors of the same arithmetic.
"""

import dataclasses
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.fx
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.backends.test.harness import stages as BaseStages
from torch.utils._pytree import tree_flatten

from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

LOWERINGS = ("shipped", "stock")


def default_passes():
    """The canonicalisation passes the shipped exports use, so the harness and
    the export scripts cannot disagree about what a lowering includes."""
    from executorch.backends.esp32s3._passes import canonical_passes

    return canonical_passes()


def shipped_backend_config():
    """The README's to_executorch config: delegates and constants on 16-byte
    boundaries, so the PIE kernels read weights in place, and the arena planned
    by memory_planning.py."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.memory_planning import planning_pass

    return ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=planning_pass("esp32"))


class Partition(BaseStages.Partition):
    def __init__(self, partitioner=None, quantize_qk: bool = False,
                 conv: bool = True, group_weights: bool = False,
                 external_weights: bool | str = False):
        super().__init__(
            partitioner=partitioner or Esp32Partitioner(quantize_qk=quantize_qk,
                                                        conv=conv,
                                                        group_weights=group_weights,
                                                        external_weights=external_weights))


class DelegatedOps:
    """The edge program the suite reads from the ETRecord, for this lowering.

    The suite's runner (run_test) counts the ETRecord edge program's ops, takes
    the lowered program's from them, and reports the rest as the ops the
    delegates took; it also checks that program's convolutions' input ranks.
    Upstream's to_edge_transform_and_lower records the program right after
    to_edge, but lower_esp32 and apply_passes fold, merge and drop ops either
    side of partitioning (constants, clamp chains, dim-order copies, identity
    views), and those would count as delegated. So the program recorded here
    is the lowered program's graph with each delegate's partition copied in
    beside it: the difference is exactly what the delegates took. It has a
    `graph` and nothing else; the runner reads nothing else."""

    def __init__(self, program):
        graph_module = program.graph_module
        self.graph = torch.fx.Graph()
        self.graph.graph_copy(graph_module.graph, {})
        for node in graph_module.graph.nodes:
            if (node.op == "call_function"
                    and node.target == torch._higher_order_ops.executorch_call_delegate):
                lowered = getattr(graph_module, node.args[0].target)
                self.graph.graph_copy(lowered.original_module.graph, {})


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(self, partitioners=None, edge_compile_config=None,
                 quantize_qk: bool = False, transform_passes=None,
                 conv: bool = True, group_weights: bool = False,
                 external_weights: bool | str = False,
                 lowering: str = "shipped"):
        if lowering not in LOWERINGS:
            raise ValueError(f"lowering {lowering!r}: expected one of {LOWERINGS}")
        if lowering == "shipped" and transform_passes is not None:
            raise ValueError("transform_passes applies to lowering='stock'; "
                             "lower_esp32 runs its own canonical passes")
        self.lowering = lowering
        # The canonicalisation passes belong in the DEFAULT path, not in an
        # opt-in one: whether a Linear lowers as `mm` or as a batch-1 `bmm` is
        # not the model author's choice, so a harness that omitted them would
        # measure the decomposition rather than the backend. lower_esp32 runs
        # them itself; the stock pipeline gets them as transform passes.
        self._passes = (default_passes() if transform_passes is None
                        else transform_passes)
        # `default_partitioner_cls` must be passed even though `partitioners` is
        # given. The base stage does:
        #     partitioners or [cls()] if cls is not None else []
        # and a conditional expression binds looser than `or`, so the whole
        # thing is `(partitioners or [cls()]) if cls is not None else []`.
        # Supplying only `partitioners=` therefore yields an EMPTY partitioner
        # list and a silently undelegated graph -- which reads as "the backend
        # claimed nothing" rather than as a wiring mistake.
        super().__init__(
            default_partitioner_cls=Esp32Partitioner,
            partitioners=partitioners or [Esp32Partitioner(quantize_qk=quantize_qk,
                                                          conv=conv,
                                                          group_weights=group_weights,
                                                          external_weights=external_weights)],
            edge_compile_config=edge_compile_config)

    def run(self, artifact, inputs=None, generate_etrecord: bool = False, **kwargs):
        if self.lowering == "stock":
            self._run_stock(artifact, generate_etrecord)
            return
        from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

        # The base stage's edge config (IR validity off) rather than the
        # README's default, so this flow lowers what every other flow lowers.
        edge = lower_esp32(artifact, self.partitioners,
                           compile_config=dataclasses.replace(self.edge_compile_conf))
        apply_passes(edge, quiet=True)
        if generate_etrecord:
            from executorch.exir.program._program import _create_empty_etrecord

            etrecord = _create_empty_etrecord()
            etrecord.edge_dialect_program = DelegatedOps(edge.exported_program())
            edge._etrecord = etrecord
        self.edge_dialect_program = edge

    def _run_stock(self, artifact, generate_etrecord):
        from executorch.exir import to_edge_transform_and_lower

        # Mirrors the base stage exactly except for `transform_passes`. The
        # attribute is `edge_compile_conf`, not `edge_compile_config`.
        # Same reason as `lower_esp32`: fp8 constants have to be promoted while
        # the ATen dialect still allows them.
        from executorch.backends.esp32s3._passes.promote_fp8_constants_pass import (
            promote_fp8_constants,
        )

        promote_fp8_constants(artifact)
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            transform_passes=self._passes,
            partitioner=self.partitioners,
            compile_config=self.edge_compile_conf,
            generate_etrecord=generate_etrecord,
        )


class Export(BaseStages.Export):
    """torch.export, strict as the base stage (the default) or not.

    A graph Esp32Quantizer converted with dynamic activations cannot be
    exported strictly: its chooser takes eps as a tensor argument, which the
    converted GraphModule's generated source spells
    `torch.Tensor(size=[1], dtype=float32)`, and dynamo, re-tracing that
    source, stops at the name `float32`. The README exports a converted model
    non-strictly (torch.export's default), and so does a tester built with
    strict_export=False."""

    def __init__(self, dynamic_shapes: Optional[Tuple[Any]] = None,
                 strict: bool = True):
        super().__init__(dynamic_shapes=dynamic_shapes)
        self.strict = strict

    def run(self, artifact, inputs) -> None:
        self.exported_program = torch.export.export(
            artifact, inputs, dynamic_shapes=self.dynamic_shapes, strict=self.strict)


class ToExecutorch(BaseStages.ToExecutorch):
    """to_executorch with shipped_backend_config() unless given another."""

    def __init__(self, config=None):
        super().__init__(config or shipped_backend_config())


class Quantize(BaseStages.Quantize):
    def __init__(self, quantizer=None, weight_bits: int = 4, is_qat: bool = False,
                 calibrate: bool = True, calibration_samples=None, **kwargs):
        super().__init__(
            quantizer=quantizer or Esp32Quantizer(weight_bits=weight_bits),
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )


# ExecuTorch's ScalarType codes (exir/scalar_type.py) for the dtypes a
# program's inputs and outputs can carry.
_DTYPES = {
    0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32,
    4: torch.int64, 5: torch.float16, 6: torch.float32, 7: torch.float64,
    11: torch.bool, 15: torch.bfloat16, 23: torch.float8_e5m2,
    24: torch.float8_e4m3fn, 27: torch.uint16, 28: torch.uint32,
    29: torch.uint64,
}


def _tensor_spec(evalue, what, index):
    """The schema Tensor of a method input or output; the runner reads and
    writes nothing else."""
    from executorch.exir.schema import Tensor

    if not isinstance(evalue.val, Tensor):
        raise NotImplementedError(
            f"esp32_host_runner runs tensor inputs and outputs only; {what} "
            f"{index} is a {type(evalue.val).__name__}")
    if evalue.val.scalar_type not in _DTYPES:
        raise NotImplementedError(
            f"{what} {index}: scalar type {evalue.val.scalar_type!r} has no "
            f"torch dtype here")
    return evalue.val


def _raw_bytes(tensor, spec):
    """`tensor` as the runner reads it: the program's dtype, laid out in its
    dim order, no header."""
    if list(tensor.shape) != list(spec.sizes):
        raise ValueError(f"input shape {list(tensor.shape)} is not the program's "
                         f"{list(spec.sizes)}")
    physical = tensor.detach().to(_DTYPES[spec.scalar_type])
    if spec.dim_order:  # a 0-d tensor has none
        physical = physical.permute(*spec.dim_order)
    return physical.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _from_raw(data, spec):
    """An output file as a logical tensor: the bytes are in the program's dim
    order, so they are reshaped in that order and permuted back."""
    dtype = _DTYPES[spec.scalar_type]
    physical_sizes = [spec.sizes[d] for d in spec.dim_order]
    numel = 1
    for size in physical_sizes:
        numel *= size
    itemsize = torch.empty((), dtype=dtype).element_size()
    if len(data) != numel * itemsize:
        raise RuntimeError(f"output holds {len(data)} B; sizes {list(spec.sizes)} "
                           f"of {dtype} need {numel * itemsize} B")
    flat = (torch.frombuffer(bytearray(data), dtype=dtype) if numel
            else torch.empty(0, dtype=dtype))
    if not spec.dim_order:  # a 0-d tensor
        return flat.reshape(())
    inverse = [0] * len(spec.dim_order)
    for axis, dim in enumerate(spec.dim_order):
        inverse[dim] = axis
    return flat.reshape(physical_sizes).permute(*inverse)


def run_on_host(pte: bytes, inputs, input_specs, output_specs, data_files=None,
                runner=None, timeout: float = 120.0, method: str = "forward"):
    """Run a serialized program with esp32_host_runner: (outputs, log).

    `inputs` are the method's flattened inputs, `input_specs` and
    `output_specs` its schema Tensors (Serialize.run reads them from the
    program), `data_files` its named data ({file name: bytes}; the runner maps
    one). Raises on a refused load, a failed execution or a timeout, with the
    runner's own message."""
    from executorch.backends.esp32s3.test.host import host_runner

    runner = Path(runner) if runner is not None else host_runner()
    if not runner.is_file():
        raise FileNotFoundError(
            f"esp32_host_runner is not built at {runner} (test/host_runner/README.md; "
            f"ESP32_HOST_RUNNER names another)")
    if len(inputs) != len(input_specs):
        raise ValueError(f"{len(inputs)} inputs for a method that takes "
                         f"{len(input_specs)}")
    data_files = data_files or {}
    if len(data_files) > 1:
        raise NotImplementedError(f"esp32_host_runner maps one .ptd; the program "
                                  f"names {sorted(data_files)}")
    with tempfile.TemporaryDirectory(prefix="esp32-host-") as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(pte)
        command = [str(runner), "--pte", str(tmp / "model.pte"), "--method", method]
        for name, payload in data_files.items():
            (tmp / name).write_bytes(payload)
            command += ["--ptd", str(tmp / name)]
        for i, (tensor, spec) in enumerate(zip(inputs, input_specs)):
            if not isinstance(tensor, torch.Tensor):
                raise NotImplementedError(
                    f"esp32_host_runner runs tensor inputs only; input {i} is a "
                    f"{type(tensor).__name__}")
            (tmp / f"in{i}.bin").write_bytes(_raw_bytes(tensor, spec))
            command += ["--input", str(tmp / f"in{i}.bin")]
        for i in range(len(output_specs)):
            command += ["--output", str(tmp / f"out{i}.bin")]
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout)
        log = done.stdout + done.stderr
        if done.returncode != 0:
            raise RuntimeError(f"esp32_host_runner exited {done.returncode}: "
                               f"{log.strip()[-2000:]}")
        outputs = [_from_raw((tmp / f"out{i}.bin").read_bytes(), spec)
                   for i, spec in enumerate(output_specs)]
    return outputs, log


class Serialize(BaseStages.Serialize):
    """The serialized program, run by esp32_host_runner.

    `run` keeps the program's bytes, its named data and its method's input
    and output specs; `run_artifact` runs it on the host runner (run_on_host)
    in place of the base stage's pybindings, which have no Esp32Backend.
    `log` keeps the runner's last output: its delegate and kernel counters."""

    def __init__(self, runner=None, timeout: float = 120.0, method: str = "forward"):
        super().__init__()
        self.runner = runner
        self.timeout = timeout
        self.method = method
        self.data_files = {}
        self.input_specs = []
        self.output_specs = []
        self.log = ""

    def run(self, artifact, inputs=None) -> None:
        super().run(artifact, inputs)
        tensor_data = getattr(artifact, "_tensor_data", None) or {}
        self.data_files = {
            (name if name.endswith(".ptd") else f"{name}.ptd"): bytes(cord)
            for name, cord in tensor_data.items()}
        plans = [p for p in artifact.executorch_program.execution_plan
                 if p.name == self.method]
        if not plans:
            raise RuntimeError(f"the program has no method {self.method!r}")
        self.input_specs = [_tensor_spec(plans[0].values[i], "input", n)
                            for n, i in enumerate(plans[0].inputs)]
        self.output_specs = [_tensor_spec(plans[0].values[i], "output", n)
                             for n, i in enumerate(plans[0].outputs)]

    def run_artifact(self, inputs):
        flat, _ = tree_flatten(inputs)
        outputs, self.log = run_on_host(
            self.buffer, flat, self.input_specs, self.output_specs,
            data_files=self.data_files, runner=self.runner, timeout=self.timeout,
            method=self.method)
        return outputs


class Esp32Tester(TesterBase):
    """`Tester` specialised for this backend. Same shape as XNNPACK's.

    Convolution defaults on here and in the shipped partitioner.  Passing
    ``conv=False`` keeps a portable-path comparison available. `lowering`:
    "shipped" (the default) lowers the README's way, "stock" through
    ExecuTorch's to_edge_transform_and_lower (module docstring); not
    `pipeline`, which is TesterBase's stage graph. `strict_export=False`
    exports as the README does (Export). `runner` names an esp32_host_runner
    other than test/host.py's.
    """

    __test__ = False  # not a pytest class despite the name

    def __init__(self, module: torch.nn.Module,
                 example_inputs: Tuple[torch.Tensor],
                 dynamic_shapes: Optional[Tuple[Any]] = None,
                 quantize_qk: bool = False,
                 group_weights: bool = False,
                 external_weights: bool | str = False,
                 lowering: str = "shipped",
                 strict_export: bool = True,
                 runner=None,
                 **kwargs):
        if lowering not in LOWERINGS:
            raise ValueError(f"lowering {lowering!r}: expected one of {LOWERINGS}")
        self.quantize_qk = quantize_qk
        self.group_weights = group_weights
        self.external_weights = external_weights
        self.lowering = lowering
        self.runner = runner
        stage_classes = TesterBase.default_stage_classes() | {
            StageType.EXPORT: lambda dynamic_shapes=None: Export(
                dynamic_shapes, strict=strict_export),
            StageType.PARTITION: Partition,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
            StageType.QUANTIZE: Quantize,
            StageType.SERIALIZE: lambda: Serialize(runner=self.runner),
        }
        if lowering == "shipped":
            stage_classes[StageType.TO_EXECUTORCH] = ToExecutorch
        super().__init__(
            module=module,
            example_inputs=example_inputs,
            stage_classes=stage_classes,
            dynamic_shapes=dynamic_shapes,
        )

    def partition(self, partition_stage=None):
        """Preserve the tester-level q@k policy in the default stage.

        TesterBase constructs registered stages with no arguments. Merely storing
        `self.quantize_qk` would therefore leave
        `Esp32Tester(..., quantize_qk=True)` on the default false path, a
        particularly bad silent failure for an opt-in. An explicitly supplied
        stage still wins.
        """
        return super().partition(
            partition_stage or Partition(quantize_qk=self.quantize_qk,
                                         group_weights=self.group_weights,
                                         external_weights=self.external_weights)
        )

    def to_edge_transform_and_lower(
        self, to_edge_and_transform_stage=None, generate_etrecord: bool = False
    ):
        return super().to_edge_transform_and_lower(
            to_edge_and_transform_stage
            or ToEdgeTransformAndLower(quantize_qk=self.quantize_qk,
                                       group_weights=self.group_weights,
                                       external_weights=self.external_weights,
                                       lowering=self.lowering),
            generate_etrecord=generate_etrecord,
        )


def delegate_count(tester: Esp32Tester) -> int:
    """Delegate calls in the lowered graph. `check_count` can assert this, but
    a plain number is what a coverage table wants."""
    gm = tester.get_artifact(
        StageType.TO_EDGE_TRANSFORM_AND_LOWER).exported_program().graph_module
    return sum(1 for n in gm.graph.nodes
               if n.op == "call_function"
               and "executorch_call_delegate" in str(n.target))
