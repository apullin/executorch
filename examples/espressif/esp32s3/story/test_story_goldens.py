# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the docs say the long golden gates, against what it gates.

A golden's values gate covers its first early_steps checksums, and the steps
run prefill first (host_story_golden.py, story_runner.inc): early_steps=1
gates the long golden's first prefill block and nothing after it. A release
check reads values=OK by the README and by run_story_hardware.py's --golden
help, so both must name the blocks the gate covers.
"""

import os
import re

from executorch.examples.espressif.esp32s3.story import export_story_flash as export

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(os.path.dirname(HERE), "README.md")
SCRIPT = os.path.join(HERE, "run_story_hardware.py")

# How each text names the prefill blocks that a gate of early_steps covers.
README_BLOCKS = {1: "the value of the first prefill block", 2: "the values of both prefill blocks"}
HELP_BLOCKS = {1: "its first prefill block's value", 2: "both prefill blocks' values"}


def words(path):
    """The file's text with every run of whitespace one space."""
    with open(path) as source:
        return " ".join(source.read().split())


def test_readme_names_the_blocks_the_long_golden_gates():
    early = export.GOLDENS["long"]["early_steps"]
    gate = re.search(r"gated on the token choices and on (.*?) \(`early_steps=(\d+)`", words(README))
    assert gate, "the README no longer states the long golden's gate"
    assert int(gate.group(2)) == early, gate.group(0)
    assert gate.group(1) == README_BLOCKS[early], gate.group(0)


def test_help_names_the_blocks_the_long_golden_gates():
    early = export.GOLDENS["long"]["early_steps"]
    assert "gated on tokens and on " + HELP_BLOCKS[early] in words(SCRIPT)
