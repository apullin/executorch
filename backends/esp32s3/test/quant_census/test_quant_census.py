# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stock quantisation recipes through the backend: one test per (recipe, model).

Each pair runs in its own process (quant_census.run), so one model's
memory never stacks on the next; a pair takes 3-15 s at CI size, the full
census about 25 minutes. Pick pairs with
QUANT_CENSUS_RECIPES=esp32,none and/or QUANT_CENSUS_MODELS=resnet8,gelu_erf.

A pair passes when the host runner's output matches the recipe's own eager
reference (cosine >= quant_census.MIN_COSINE, provisional). A pair that fails
today has an entry in quant_census/expect.py and runs as xfail(strict=True):
the day it starts passing, this file fails and asks for the entry to go. A pair
the recipe is not meant for (torchao on a convolutional model) is skipped.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from executorch.backends.esp32s3.test.quant_census import (
    applicable,
    pairs,
    result_path,
)
from executorch.backends.esp32s3.test.quant_census.expect import EXPECT
from executorch.backends.esp32s3.test.quant_census.models import _ZOO, zoo_available

ROOT = Path(__file__).resolve().parent
RECIPES = [r for r in os.environ.get("QUANT_CENSUS_RECIPES", "").split(",") if r]
MODELS = [m for m in os.environ.get("QUANT_CENSUS_MODELS", "").split(",") if m]
PICKED = [(r, m) for r, m in pairs()
          if (not RECIPES or r in RECIPES) and (not MODELS or m in MODELS)]
STAGES = ("quantize", "export", "lower", "no-kernel", "run", "mismatch")


@pytest.fixture(scope="session", autouse=True)
def results(tmp_path_factory):
    """The records go to a fresh temporary directory, never the source tree:
    result_path() and each pair's process read QUANT_CENSUS_RESULTS."""
    with pytest.MonkeyPatch.context() as patch:
        path = tmp_path_factory.mktemp("quant_census")
        patch.setenv("QUANT_CENSUS_RESULTS", str(path))
        yield path


def _param(recipe, model):
    marks = []
    reason = applicable(recipe, model)
    if reason:
        marks.append(pytest.mark.skip(reason=f"n/a: {reason}"))
    elif (recipe, model) in EXPECT:
        marks.append(pytest.mark.xfail(strict=True, reason=EXPECT[(recipe, model)]))
    return pytest.param(recipe, model, marks=marks, id=f"{recipe}-{model}")


def test_expect_names_census_pairs():
    """Every EXPECT entry is a pair the census runs, says which stage fails,
    and gives a cause."""
    census = set(pairs())
    for (recipe, model), reason in EXPECT.items():
        assert (recipe, model) in census, f"{recipe}/{model} is not a census pair"
        if model in _ZOO and not zoo_available():
            continue  # n/a without the nano zoo or transformers
        assert not applicable(recipe, model), f"{recipe}/{model} is n/a, not a failure"
        stage, _, cause = reason.partition(": ")
        assert stage in STAGES and cause, f"{recipe}/{model}: {reason!r}"


def test_tosa_stub_names_but_refuses_work():
    """The tosa_serializer stand-in gives Arm's quantizers distinct enumerant
    names to import, and raises on anything that would do TOSA work."""
    path = ROOT / "_stubs" / "tosa_serializer" / "__init__.py"
    spec = importlib.util.spec_from_file_location("tosa_serializer_stub", path)
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    assert ts.DType.INT8 is ts.DType.INT8 and ts.DType.INT8 is not ts.DType.INT16
    for make in (ts.TosaSerializer, ts.TosaSerializerAttribute, ts.Op.TILE):
        with pytest.raises(ts.StubUsed):
            make()
    with pytest.raises(AttributeError):
        ts.no_such_name  # noqa: B018


@pytest.mark.parametrize("recipe,model", [_param(r, m) for r, m in PICKED])
def test_pair_matches_its_reference(recipe, model):
    result = subprocess.run(
        [sys.executable, "-m", "executorch.backends.esp32s3.test.quant_census.run",
         recipe, model], capture_output=True, text=True, timeout=900)
    record = result_path(recipe, model)
    assert record.is_file(), result.stdout[-2000:] + result.stderr[-2000:]
    rec = json.loads(record.read_text())
    if rec["status"] in ("n/a", "skipped"):
        # Decided at run time: the recipe's filter took no layer of this
        # model, or the recipe needs real TOSA functionality.
        pytest.skip(f"{rec['status']}: {rec.get('reason') or rec['error']['message'][:200]}")
    lines = result.stdout.strip().splitlines()
    assert rec["status"] == "ok", lines[-1] if lines else result.stderr[-2000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
