# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ESP32-S3 flows for ExecuTorch's backend test suite (backends/test/suite).

Every case is lowered the README's way (lower_esp32, apply_passes, the esp32
memory plan), serialized, and run by esp32_host_runner: the delegate's runtime
and the operator library built for the build machine, with C reference kernels
in place of the PIE assembly and the portable kernels for whatever the
delegate does not take (tester/espressif_tester.py). So every case runs,
delegated or not (`is_delegated=False`, as Cortex-M's flow): an op the
partitioner refuses still has to lower, load and give the right numbers.

- `esp32s3`: the float model, partitioned with Esp32Partitioner's defaults.
  A float weight the delegate takes is stored as int8 and the activation is
  quantised per row at execute, so a delegated case is compared with fp32
  eager at the suite's fixed tolerance (atol 0.1, rtol 0.04).
- `esp32s3_8da4w`: quantised first by the default Esp32Quantizer(), int4
  per-channel symmetric weights with dynamic int8 activations (the README's
  quantised flow). The reference is the converted model, as for every
  quantised flow. The converted model is exported non-strictly, as the
  README exports it: strict export cannot re-trace its chooser's tensor
  argument (tester/espressif_tester.py's Export).

plugin.py registers both with the suite (test/suite/README.md). A flow whose
host runner is not built is not registered, as Cortex-M's is not without its
FVP: every case would fail the same way.

ESP32S3_SKIPS and ESP32S3_XFAILS map a test-name pattern (a substring of the
pytest item name, as TestFlow matches them) to the reason. A skipped case does
not run at all; an expected failure still runs, strictly, so one that starts
passing fails the run until its entry goes.
"""

from executorch.backends.test.suite.flow import TestFlow

# The skip for a case's dynamic-shape variant (the use_dynamic_shapes
# parameter, which only the model tests have).
DYNAMIC_SHAPES_REASON = (
    "esp32s3 lowers for one fixed shape: the delegate's descriptors and the "
    "memory plan take their dimensions from the graph")

# Expected failures every portable runtime shares. ExecuTorch's portable
# library pools in 2-D only (avg_pool2d, max_pool2d_with_indices,
# _adaptive_avg_pool2d: kernels/portable/functions.yaml) and the delegate does
# not pool, so a 3-D pool's program loads with "kernel ... not found". The
# suite's runner skips 3-D convolutions for that reason
# (_graph_has_unsupported_patterns), but not 3-D pools.
NO_3D_POOL_KERNEL = (
    "no ExecuTorch kernel pools in 3-D: the portable library has 2-D pools only "
    "and the delegate does not pool, so the program fails to load (kernel not "
    "found)")
PORTABLE_XFAILS = {
    "test_avgpool3d": NO_3D_POOL_KERNEL,
    "test_maxpool3d": NO_3D_POOL_KERNEL,
    "test_adaptive_avgpool3d": NO_3D_POOL_KERNEL,
    "test_adaptive_maxpool3d": NO_3D_POOL_KERNEL,
}

ESP32S3_SKIPS: dict[str, str] = {}

ESP32S3_XFAILS: dict[str, str] = {**PORTABLE_XFAILS}

ESP32S3_8DA4W_SKIPS: dict[str, str] = {}

ESP32S3_8DA4W_XFAILS: dict[str, str] = {**PORTABLE_XFAILS}


def _tester(model, inputs, **kwargs):
    from executorch.backends.esp32s3.test.tester.espressif_tester import Esp32Tester

    return Esp32Tester(model, inputs, **kwargs)


def _quantized_tester(model, inputs, **kwargs):
    return _tester(model, inputs, strict_export=False, **kwargs)


def _quantize_stage():
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
        Esp32Quantizer,
    )
    from executorch.backends.esp32s3.test.tester.espressif_tester import Quantize

    return Quantize(quantizer=Esp32Quantizer())


def _flow(name, skips, xfails, quantize_stage_factory=None) -> TestFlow:
    flow = TestFlow(
        name,
        backend="esp32s3",
        tester_factory=_quantized_tester if quantize_stage_factory else _tester,
        quantize=quantize_stage_factory is not None,
        quantize_stage_factory=quantize_stage_factory,
        is_delegated=False,
        param_skip_reasons={"use_dynamic_shapes": {True: DYNAMIC_SHAPES_REASON}},
        skip_patterns=list(skips),
        xfail_patterns=list(xfails),
    )
    # The reasons, for plugin.py to put on each skip and expected failure in
    # place of the suite's generic one.
    flow.skip_reasons = dict(skips)
    flow.xfail_reasons = dict(xfails)
    return flow


ESP32S3_TEST_FLOW = _flow("esp32s3", ESP32S3_SKIPS, ESP32S3_XFAILS)
ESP32S3_8DA4W_TEST_FLOW = _flow("esp32s3_8da4w", ESP32S3_8DA4W_SKIPS,
                                ESP32S3_8DA4W_XFAILS, _quantize_stage)


def esp32s3_flows() -> list[TestFlow]:
    """The flows to register: none when the host runner is not built."""
    from executorch.backends.esp32s3.test.host import host_runner

    if not host_runner().is_file():
        return []
    return [ESP32S3_TEST_FLOW, ESP32S3_8DA4W_TEST_FLOW]
