# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The story firmware's weight and memory settings, end to end on the host.

run_story_hardware.py turns its flags into -D settings, main/CMakeLists.txt
forwards them to the compiler, and story_runner.inc gives each a default. The
settings are CMake cache variables: a build directory keeps whatever value it
was last given. So the script passes every one on every build, and CMake
forwards each whenever it is set, zero included, unless zero is also the
source's default; otherwise a flag set to zero would silently build with the
default instead.
"""

import argparse
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from executorch.examples.espressif.esp32s3.story import run_story_hardware as story  # noqa: E402

MAIN = os.path.join(os.path.dirname(HERE), "main")


def settings(*args, **kwargs):
    return dict(define[2:].split("=", 1) for define in story.placement_defines(*args, **kwargs))


def source_defaults():
    """story_runner.inc's `#ifndef X / #define X value` defaults: an integer,
    or a product of integers in parentheses."""
    with open(os.path.join(MAIN, "story_runner.inc")) as source:
        text = source.read()
    defaults = {}
    for name, value in re.findall(r"^#ifndef (ET_STORY_\w+)\n#define \1 (.+)$", text, re.M):
        product = 1
        for factor in value.strip().strip("()").split("*"):
            product *= int(factor)
        defaults[name] = product
    return defaults


def cmake_forwarding():
    """Each setting main/CMakeLists.txt forwards: "set" when any value it is
    given reaches the compiler, "true" when only a true one does."""
    with open(os.path.join(MAIN, "CMakeLists.txt")) as cmake:
        text = cmake.read()
    forwarded = {}
    for names, body in re.findall(r"foreach\(setting ([^)]*)\)(.*?)endforeach\(\)", text, re.S):
        if "if(DEFINED ${setting})" in body:
            forwarded.update(dict.fromkeys(names.split(), "set"))
    for name in re.findall(r"^if\((ET_STORY_\w+)\)", text, re.M):
        forwarded.setdefault(name, "true")
    return forwarded


# Ranked placement's default is story_runner.inc's ET_STORY_TIER_RANK.
RANK = str(int(story.TIER_RANK_DEFAULT))


def test_every_setting_on_every_build():
    plain = settings()
    assert plain == {
        "ET_STORY_STAGE_BYTES": "0", "ET_STORY_TIER_RESERVE": "0", "ET_STORY_TEMP_BYTES": "2097152",
        "ET_STORY_TIER_RANK": RANK, "ET_STORY_INTERNAL_FLOOR": "40960", "ET_STORY_DECODE_ACT_INTERNAL": "1",
        "ET_STORY_HOT_CONSTANTS": "1", "ET_STORY_ARENA_BYTES": "131072"}
    changed = settings("auto", 0, not story.TIER_RANK_DEFAULT, stage_reserve_mb=1, internal_floor_kb=64,
                       decode_act_internal=False, hot_constants=False, arena_kb=96)
    assert set(changed) == set(plain)
    # --stage-mb auto ranks, whatever the default.
    same = {"ET_STORY_TIER_RANK"} if story.TIER_RANK_DEFAULT else set()
    assert {name for name in plain if changed[name] == plain[name]} == same
    assert settings(6, 1, not story.TIER_RANK_DEFAULT)["ET_STORY_TIER_RANK"] != RANK


def test_the_board_build():
    # run_story_hardware.py --dual-core --stage-mb 6 --temp-mb 1 --backend-o2
    board = settings(6, 1)
    assert board["ET_STORY_STAGE_BYTES"] == str(6 << 20)
    assert board["ET_STORY_TEMP_BYTES"] == str(1 << 20)
    assert board["ET_STORY_TIER_RANK"] == RANK and board["ET_STORY_TIER_RESERVE"] == "0"
    assert board["ET_STORY_DECODE_ACT_INTERNAL"] == "1" and board["ET_STORY_HOT_CONSTANTS"] == "1"


def test_each_on_off_setting_has_both_flags(capsys):
    # Defaults move with the board's results: each on/off setting can be
    # given either way on the command line, whichever its default.
    with pytest.raises(SystemExit):
        story.main(["--help"])
    usage = capsys.readouterr().out
    for flag in ("--tier-rank", "--no-tier-rank", "--decode-act-internal", "--no-decode-act-internal",
                 "--hot-constants", "--no-hot-constants"):
        assert flag in usage, flag


def test_auto_stage_keeps_a_reserve_and_ranks():
    auto = settings("auto", 1, False, stage_reserve_mb=0.25)
    assert auto["ET_STORY_STAGE_BYTES"] == "-1"  # the backend's weight_tier_budget auto
    assert auto["ET_STORY_TIER_RESERVE"] == str(256 << 10)
    assert auto["ET_STORY_TIER_RANK"] == "1"
    # A fixed budget has no reserve, whatever the flag says.
    assert settings(4.5, 1, True, stage_reserve_mb=0.25)["ET_STORY_TIER_RESERVE"] == "0"


def test_stage_mb_takes_megabytes_or_auto():
    assert story.stage_mb_arg("auto") == "auto"
    assert story.stage_mb_arg("4.5") == 4.5
    with pytest.raises(argparse.ArgumentTypeError):
        story.stage_mb_arg("all of it")


def test_cmake_forwards_every_setting_and_zero_reaches_the_compiler():
    defaults = source_defaults()
    forwarded = cmake_forwarding()
    for name in settings():
        assert name in defaults, f"story_runner.inc gives {name} no default"
        assert name in forwarded, f"main/CMakeLists.txt does not forward {name}"
        if forwarded[name] == "true":
            assert defaults[name] == 0, f"{name}=0 would build with its default, {defaults[name]}"


def test_source_defaults_are_the_scripts():
    # A build without run_story_hardware.py (idf.py alone) gets what the
    # script gives when no flag is set.
    defaults = source_defaults()
    for name, value in settings().items():
        assert defaults[name] == int(value), name
