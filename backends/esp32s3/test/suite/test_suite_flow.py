# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The suite flows' own checks: the host runner stage reads back what the
program computes (dtype, shape, dim order), the suite's run_test passes a
delegated case on each flow, and the plugin registers the flows and reports.

Build the runner first (test/host_runner/README.md); ESP32_HOST_RUNNER
overrides cmake-out-espressif/esp32_host_runner. The whole suite is not run
here: test/suite/README.md says how.
"""

import json
import os
import re
import subprocess
import sys

import torch
from torch import nn

from executorch.backends.test.harness.stages import StageType

from executorch.backends.esp32s3.test.host import ET_ROOT
from executorch.backends.esp32s3.test.suite.flow_esp32s3 import (
    ESP32S3_8DA4W_TEST_FLOW,
    ESP32S3_TEST_FLOW,
)
from executorch.backends.esp32s3.test.tester.espressif_tester import (
    delegate_count,
    Esp32Tester,
)

# The suite's comparison (backends/test/suite/runner.py).
SUITE_ATOL, SUITE_RTOL = 1e-1, 4e-2


class LinearRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return torch.relu(self.fc(x))


class Layouts(nn.Module):
    """Portable ops whose outputs are channels-last float, int64, bool and a
    0-d float (no dim order at all)."""

    def forward(self, x):
        return x * 2.0 + 1.0, x.flatten(1).argmax(-1), x > 0, x.amax()


def lowered(model, inputs, **kwargs):
    tester = Esp32Tester(model.eval(), inputs, **kwargs)
    tester.export().to_edge_transform_and_lower().to_executorch().serialize()
    return tester


def test_host_stage_reads_dtypes_and_dim_orders():
    """A channels-last input and output, int64, bool and 0-d outputs, read
    back bit for bit: none of them is delegated, so the portable kernels'
    results are eager's."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)
    tester = lowered(Layouts(), (x,))
    serialize = tester.stages[StageType.SERIALIZE]
    assert list(serialize.input_specs[0].dim_order) == [0, 2, 3, 1]
    got = serialize.run_artifact((x,))
    want = Layouts()(x)
    assert [t.dtype for t in got] == [torch.float32, torch.int64, torch.bool,
                                      torch.float32]
    for g, w in zip(got, want):
        assert g.shape == w.shape and torch.equal(g, w)
    tester.run_method_and_compare_outputs(inputs=(x,), atol=0, rtol=0)


def test_delegated_linear_runs_on_the_host():
    """The delegate's int8 Linear and ReLU, inside the suite's tolerance of
    eager, and visibly not eager's bits (the weight is stored as int8)."""
    torch.manual_seed(0)
    model, x = LinearRelu(), (torch.randn(4, 16),)
    tester = lowered(model, x)
    assert delegate_count(tester) == 1
    got = tester.stages[StageType.SERIALIZE].run_artifact(x)[0]
    want = model(*x)
    assert torch.allclose(got, want, atol=SUITE_ATOL, rtol=SUITE_RTOL)
    assert not torch.equal(got, want)
    calls = re.search(r"delegate calls (\d+)", tester.stages[StageType.SERIALIZE].log)
    assert calls and int(calls.group(1)) == 1, tester.stages[StageType.SERIALIZE].log


def test_suite_run_test_passes_each_flow():
    """The suite's own run_test, as each upstream case calls it: lowered,
    counted (the ETRecord), serialized, run and compared."""
    from executorch.backends.test.suite.reporting import TestResult
    from executorch.backends.test.suite.runner import run_test

    for flow in (ESP32S3_TEST_FLOW, ESP32S3_8DA4W_TEST_FLOW):
        torch.manual_seed(0)
        summary = run_test(LinearRelu().eval(), (torch.randn(4, 16),), flow,
                           "test_linear_relu", "test_linear_relu", 0, None,
                           generate_random_test_inputs=False)
        assert summary.result == TestResult.SUCCESS, (flow.name, summary.error)
        assert summary.is_delegated(), flow.name
        assert summary.pte_size_bytes and summary.tensor_error_statistics


def test_suite_counts_only_what_the_delegates_took():
    """The suite's delegated-op count, on a case with no delegate, is empty:
    the ops apply_passes merges or drops (here hardswish's clamp chain and
    dim-order copies) are not counted as delegated (DelegatedOps). With a
    delegate, the count is the partition's ops and none of the portable ones."""
    from executorch.backends.test.suite.runner import run_test

    torch.manual_seed(0)
    summary = run_test(nn.Hardswish().eval(), (torch.randn(4, 16),),
                       ESP32S3_TEST_FLOW, "test_hardswish", "test_hardswish", 0,
                       None, generate_random_test_inputs=False)
    assert summary.result.is_success(), summary.error
    assert not summary.is_delegated(), summary.delegated_op_counts
    summary = run_test(LinearRelu().eval(), (torch.randn(4, 16),),
                       ESP32S3_TEST_FLOW, "test_linear_relu", "test_linear_relu",
                       0, None, generate_random_test_inputs=False)
    assert summary.delegated_op_counts["aten::addmm"] == 1, summary.delegated_op_counts
    assert not set(summary.delegated_op_counts) & set(summary.undelegated_op_counts)


def test_plugin_registers_flows_and_reports(tmp_path):
    """The plugin, as test/suite/README.md runs it, on one upstream file:
    both flows collected beside upstream's, each case reported with the
    suite's own metadata."""
    report = tmp_path / "report.json"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", os.devnull, "--rootdir", str(ET_ROOT),
         "--confcutdir", str(ET_ROOT),
         "-p", "executorch.backends.esp32s3.test.suite.plugin",
         "-p", "no:cacheprovider", "-q",
         "backends/test/suite/operators/test_neg.py", "-m", "backend_esp32s3",
         "--esp32-report", str(report)],
        cwd=ET_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    tests = json.loads(report.read_text())["tests"]
    assert {t["flow"] for t in tests} == {"esp32s3", "esp32s3_8da4w"}
    for test in tests:
        if "test_neg_edge_cases" in test["nodeid"]:  # unittest.skip upstream
            assert test["outcome"] == "skipped", test
            continue
        subtests = test["metadata"]["subtests"]
        assert test["outcome"] == "passed", test
        assert subtests and all(s["Result"] == "Pass" for s in subtests), test
    assert "esp32s3 suite flows" in done.stdout
