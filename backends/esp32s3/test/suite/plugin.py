# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A pytest plugin that runs ExecuTorch's backend test suite on this backend.

    pytest -c /dev/null -p executorch.backends.esp32s3.test.suite.plugin \\
        backends/test/suite/operators -m backend_esp32s3 \\
        [--esp32-report report.json] [--esp32-seed N]

The suite's conftest parametrizes every test over all_flows(), which it
imports by name from backends/test/suite/flow.py when pytest loads it. A
plugin given with -p is imported before any conftest, so this one wraps
flow.all_flows there and the suite sees this backend's flows
(flow_esp32s3.py) beside upstream's, with no upstream file changed. `-m
flow_esp32s3` or `-m flow_esp32s3_8da4w` selects one flow, `-m
backend_esp32s3` both.

It also:

- starts the suite's test session with a fixed seed (--esp32-seed, default
  0) and seeds torch per test from the test's file and name, so a run's
  inputs repeat, and with them its expected failures, which are strict;
- puts the flow's own reason on each skip and expected failure
  (flow_esp32s3.py), in place of the suite's generic one;
- writes --esp32-report: one entry per test, in pytest-json-report's layout
  with the suite conftest's own per-subtest metadata (result, error,
  delegated ops, output error statistics), so
  `python -m executorch.backends.test.suite.generate_markdown_summary_json`
  reads it, and prints a summary per flow;
- releases, after every test, the process-wide upstream caches that keep
  every program a process lowers alive until it exits (ExecuTorch's
  is_same_node, FakeTensorMode's cache, the torchao and torch.ao device caches,
  and fx's generated sources), so a whole suite runs in one process at the
  memory of its largest case.
"""

import gc
import hashlib
import json
import linecache
import os
import sys
from collections import Counter

import pytest

import executorch.backends.test.suite.flow as _suite_flow
from executorch.backends.esp32s3.test.suite.flow_esp32s3 import esp32s3_flows


def _with_esp32s3(upstream):
    def all_flows():
        flows = upstream()
        for flow in esp32s3_flows():
            flows[flow.name] = flow
        return flows

    all_flows.__wrapped__ = upstream
    all_flows._esp32s3 = True
    return all_flows


if not getattr(_suite_flow.all_flows, "_esp32s3", False):
    _suite_flow.all_flows = _with_esp32s3(_suite_flow.all_flows)


def pytest_addoption(parser):
    group = parser.getgroup("esp32s3", "ESP32-S3 backend suite flows")
    group.addoption("--esp32-seed", type=int, default=0,
                    help="seed for the suite's inputs (default 0)")
    group.addoption("--esp32-report", default=None,
                    help="write a JSON report of the esp32s3 cases here")


def _flow(item):
    callspec = getattr(item, "callspec", None)
    flow = callspec.params.get("test_runner") if callspec is not None else None
    return flow if getattr(flow, "backend", None) == "esp32s3" else None


def pytest_configure(config):
    from executorch.backends.test.suite.reporting import (
        begin_test_session,
        get_active_test_session,
    )

    if get_active_test_session() is None:
        begin_test_session(None, seed=config.getoption("esp32_seed"))
    config._esp32_records = []
    if not esp32s3_flows():
        from executorch.backends.esp32s3.test.host import host_runner

        print(f"esp32s3: no flows registered: esp32_host_runner is not built at "
              f"{host_runner()}", file=sys.stderr)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """The flow's reason on each skip and expected failure. The suite's
    conftest added them first, with a generic reason, from the same
    patterns; each is swapped for one with this flow's."""
    for item in items:
        flow = _flow(item)
        if flow is None:
            continue
        name = item.originalname or item.name
        for kind, reasons, match, generic in (
                ("skip", flow.skip_reasons, name,
                 f"Skipped by {flow.name} skip_patterns"),
                ("xfail", flow.xfail_reasons, item.name,
                 f"Expected to fail by {flow.name} xfail_patterns")):
            reason = next((r for p, r in reasons.items() if p in match), None)
            if reason is None:
                continue
            item.own_markers[:] = [
                m for m in item.own_markers
                if not (m.name == kind and generic in
                        (m.kwargs.get("reason"), *m.args[:1]))]
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True)
                            if kind == "xfail" else pytest.mark.skip(reason=reason))


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item):
    # The operator tests draw their inputs before the suite seeds anything.
    if _flow(item) is None:
        return
    import torch

    key = (f"{item.path.name}::{item.originalname or item.name}:"
           f"{item.config.getoption('esp32_seed')}")
    torch.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],
                                     "little"))


def _suite_metadata(config):
    """The suite conftest's pytest_json_runtest_metadata: called directly,
    since pytest-json-report (which would call it) is not required here."""
    for plugin in config.pluginmanager.get_plugins():
        hook = getattr(plugin, "pytest_json_runtest_metadata", None)
        path = getattr(plugin, "__file__", "") or ""
        if hook is not None and path.endswith(os.path.join("suite", "conftest.py")):
            return hook
    return None


def _outcome(report):
    if hasattr(report, "wasxfail"):
        return "xfailed" if report.skipped else "xpassed"
    if str(report.longrepr).startswith("[XPASS(strict)]"):
        return "xpassed-strict"
    return report.outcome


# tryfirst: the outermost wrapper, so the report it reads is final (the
# skipping plugin's wrapper, inside it, has already marked expected failures).
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    flow = _flow(item)
    if flow is None:
        return
    final = report.when == "call" or (report.when == "setup" and not report.passed)
    if not final:
        return
    metadata = {"subtests": []}
    hook = _suite_metadata(item.config)
    if hook is not None and "test_runner" in (getattr(item, "funcargs", None) or {}):
        metadata = hook(item, call)
    reason = getattr(report, "wasxfail", None)
    if reason is None and report.skipped and isinstance(report.longrepr, tuple):
        reason = report.longrepr[2]
    item.config._esp32_records.append({
        "nodeid": report.nodeid,
        "flow": flow.name,
        "outcome": _outcome(report),
        "reason": reason,
        "metadata": metadata,
    })


def release_upstream_caches():
    """Drop what keeps every lowered program alive in a long process
    (an upstream leak): ExecuTorch's is_same_node lru_cache (exir/backend/utils.py),
    filled by to_backend; torch's class-level FakeTensorMode cache, whose
    negative entries for executorch_call_delegate hold the lowered module; the
    torchao and torch.ao device caches prepare_pt2e and convert_pt2e fill;
    and the forward source fx keeps for every GraphModule it recompiles. Each
    of the first four holds the same objects, so all of them go."""
    from executorch.exir.backend.utils import is_same_node
    from torch._subclasses.fake_tensor import FakeTensorMode

    is_same_node.cache_clear()
    FakeTensorMode.cache_clear()
    for module, name in (("torchao.utils", "_assert_and_get_unique_device"),
                         ("torch.ao.quantization.fx.utils",
                          "assert_and_get_unique_device")):
        fn = getattr(sys.modules.get(module), name, None)
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    from torch.fx import graph_module

    sources = getattr(getattr(graph_module, "_loader", None), "eval_cache", None)
    if isinstance(sources, dict):
        for key in list(sources):
            linecache.cache.pop(key, None)
        sources.clear()


def pytest_runtest_logfinish(nodeid, location):
    # After the test's teardown, when pytest has dropped its fixtures.
    release_upstream_caches()
    gc.collect()


def _summary(records):
    """{flow: lines} for the terminal summary."""
    out = {}
    for flow in sorted({r["flow"] for r in records}):
        mine = [r for r in records if r["flow"] == flow]
        outcomes = Counter(r["outcome"] for r in mine)
        results = Counter(s.get("Result Detail") or s.get("Result")
                          for r in mine for s in r["metadata"]["subtests"])
        delegated = sum(1 for r in mine
                        if any(s.get("Delegated") == "True"
                               for s in r["metadata"]["subtests"]))
        lines = [", ".join(f"{n} {k}" for k, n in sorted(outcomes.items()))
                 + f"; {delegated} of {len(mine)} delegated",
                 "subtests: " + ", ".join(f"{n} {k}" for k, n in
                                          sorted(results.items(), key=str))]
        out[flow] = lines
    return out


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    records = getattr(config, "_esp32_records", [])
    if not records:
        return
    terminalreporter.section("esp32s3 suite flows")
    for flow, lines in _summary(records).items():
        terminalreporter.line(f"{flow}: {lines[0]}")
        terminalreporter.line(f"  {lines[1]}")


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("esp32_report")
    if not path:
        return
    with open(path, "w") as f:
        json.dump({"seed": session.config.getoption("esp32_seed"),
                   "exitstatus": int(exitstatus),
                   "tests": getattr(session.config, "_esp32_records", [])},
                  f, indent=1, default=str)
