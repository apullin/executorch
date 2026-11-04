# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The expf check (expf_golden.py, expf_check.cpp).

  golden    expf_golden.h is what the generator makes from mirrors.expf_newlib
  coverage  the boundary cases reach every branch of newlib's expf, on both
            sides of each threshold
  host      the host runner's --selftest line: newlib_expf.h's C transcription
            gives the numpy transcription's bits, 0 mismatches; the chip's own
            expf is run_parity's gate under QEMU and run_hardware's on the board

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import re
import subprocess

import numpy as np
import pytest

from executorch.backends.esp32s3.test.host import host_runner
from executorch.backends.esp32s3.test.runtime import expf_golden as eg


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def test_golden_header_is_current():
    assert eg.HEADER.read_text() == eg.golden_header()


def branch(bits):
    """Which path of newlib's __ieee754_expf an input takes (ef_exp.c)."""
    sign, magnitude = bits >> 31, bits & 0x7FFFFFFF
    signed = bits - (1 << 32) if bits >> 31 else bits
    if magnitude > 0x7F800000:
        return "nan"
    if magnitude == 0x7F800000:
        return "inf"
    if signed > 0x42B17217:
        return "overflow"
    if sign and magnitude > 0x42CFF1B5:
        return "underflow"
    if magnitude > 0x3EB17218:
        if magnitude < 0x3F851592:
            return "near"
        x = np.float32(np.uint32(bits).view(np.float32))
        k = int(np.trunc(np.float64(np.float32(1.4426950216)) * np.float64(x)
                         + (0.5 if not sign else -0.5)))
        return "far, k < -125" if k < -125 else "far"
    if magnitude < 0x34000000:
        return "tiny"
    return "primary"


def test_boundaries_reach_every_branch():
    cases = eg.boundary_inputs()
    reached = {branch(b) for b in cases}
    assert reached == {"inf", "overflow", "underflow", "near", "far",
                       "far, k < -125", "tiny", "primary"}, reached
    # Both sides of every threshold the branches read.
    for threshold in (0x34000000, 0x3EB17218, 0x3F851592, 0x42B17217):
        assert {threshold - 1, threshold, threshold + 1} <= set(cases), hex(threshold)
    assert {0xC2CFF1B4, 0xC2CFF1B5, 0xC2CFF1B6} <= set(cases)
    # And the streams draw what they say.
    rng = eg.stream_inputs(0, 0x12345678, 4096)
    assert rng.min() >= -104 and rng.max() < 24
    bits = eg.stream_inputs(1, 0x9E3779B9, 4096)
    assert np.isfinite(bits).all() and np.abs(bits).max() < 128
    assert (np.abs(bits) < np.float32(2.0 ** -126)).any()  # subnormals too


def test_host_selftest_line(runner):
    result = subprocess.run([str(runner), "--selftest"], capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    line = re.search(r"^expf check: (\d+) case\(s\), (\d+) mismatch\(es\).*$",
                     result.stdout, re.M)
    assert line, result.stdout
    boundary, streams = eg.expected()
    assert int(line.group(1)) == len(boundary) + sum(s[2] for s in streams)
    assert line.group(2) == "0", result.stdout
    for name, _, _, _, fnv in streams:
        assert f"{name} fnv {fnv:08x}" in line.group(0), line.group(0)
