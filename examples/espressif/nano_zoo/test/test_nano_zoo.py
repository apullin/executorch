# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every nano-zoo family fits the board, lowers, and runs on the host runner.

Each family runs in its own process (nano_zoo.run), so one model's memory
never stacks on the next. At device size a family takes about 40 s.
NANO_ZOO_SIZE=ci runs the toy-width configs (seconds each);
NANO_ZOO_FAMILIES=llama3_2,gemma4 picks families.

A family whose module sets EXPECT is a known gap and runs as
xfail(strict=True): the day it starts passing, this file fails and asks for its
EXPECT to be removed. EXPECT = {"device": reason} marks only that size (a
model that runs but overflows the flash region at device size).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from executorch.examples.espressif.nano_zoo import PARAM_CAP, families, family

ROOT = Path(__file__).resolve().parent
SIZE = os.environ.get("NANO_ZOO_SIZE", "device")
PICKED = [f for f in os.environ.get("NANO_ZOO_FAMILIES", "").split(",") if f] or families()
# The header every family carries: the released model, what DEVICE changed, and
# the alternatives that were not taken.
SECTIONS = ("Released", "DEVICE", "Other ways to fit it")


def _expected(name):
    reason = getattr(family(name), "EXPECT", None)
    if isinstance(reason, dict):  # {size: reason}: a gap only one size shows
        reason = reason.get(SIZE)
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize("name", PICKED)
def test_header_and_configs(name):
    spec = family(name)
    doc = spec.__doc__ or ""
    missing = [s for s in SECTIONS if s not in doc]
    assert not missing, f"{name}: header lacks {missing}"
    for attr in ("MODEL_TYPE", "MODEL_CLASS", "DEVICE", "CI"):
        assert hasattr(spec, attr), f"{name}: no {attr}"


@pytest.mark.parametrize("name", PICKED)
def test_device_config_is_device_sized(name):
    """DEVICE fills most of the board's model region without passing it.
    Counted on the meta device: nothing is allocated."""
    pytest.importorskip("transformers")
    from executorch.examples.espressif.esp32s3.scripts.flash_layout import ET_PACKAGE_BUDGET
    from executorch.examples.espressif.nano_zoo import build

    params, largest = build.meta_params(name, "device")
    estimate = build.weight_bytes(params)
    assert params <= PARAM_CAP, largest
    assert estimate <= ET_PACKAGE_BUDGET, (
        f"{estimate:,} B estimated > {ET_PACKAGE_BUDGET:,} B; largest: {largest}")
    assert estimate >= 8_000_000, f"{estimate:,} B: DEVICE should fill most of the region"


@pytest.mark.parametrize("name", [_expected(n) for n in PICKED])
def test_lowers_and_matches_device_arithmetic(name):
    pytest.importorskip("transformers")
    result = subprocess.run(
        [sys.executable, "-m", "executorch.examples.espressif.nano_zoo.run", name,
         "--size", SIZE], capture_output=True, text=True, timeout=900)
    record = ROOT.parent / "results" / f"{name}.{SIZE}.json"
    assert record.is_file(), result.stdout[-2000:] + result.stderr[-2000:]
    rec = json.loads(record.read_text())
    assert rec["status"] == "ok", result.stdout.strip().splitlines()[-1]
    if SIZE == "device":
        assert rec["fits_flash"], rec.get("flash_error")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
