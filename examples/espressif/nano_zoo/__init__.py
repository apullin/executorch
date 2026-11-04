# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nano zoo: Hugging Face model families at ESP32-S3 size, with random weights.

One module per released model, named with its version (llama3_2, not llama:
Llama 3.1, 3.2 and 4 are different designs). Each names the transformers class
to build and gives two configs with the same layer pattern:

  DEVICE  sized to the board: the program and its int4 weights fit the 15.27 MB
          model region of the 16 MB flash (flash_layout.ET_PACKAGE_BUDGET).
  CI      the same layout at toy width, so the whole zoo lowers in minutes.

The module docstring says what the released model looks like, what was shrunk
to fit, and how else it could have been shrunk. Configs are written out in full
(every key must be a real config field, or a builder-only kwarg the config
consumes), nothing is downloaded, and the weights are random and seeded. The
backend quantises weights itself when it lowers (int4 in groups of 128;
activations to int8 per row at run time), so there is no calibration pass.

Every model is counted on the meta device before a real byte is allocated and
refused over PARAM_CAP. A composite config that silently keeps a sub-model at its
released size (step3p7's vision tower: 43 GB) fails that count instead of the
machine.

transformers is pinned to TRANSFORMERS_VERSION because config fields move between
releases: requirements.txt beside this file pins it, and README.md says how to
install it. Run one family with
`python -m executorch.examples.espressif.nano_zoo.run <family>` or all of them
with test/test_nano_zoo.py; census.py renders the results as CENSUS.md.

Importing this package (or a family module) does not import torch.
"""

import importlib
import pkgutil
from pathlib import Path

# examples/espressif/nano_zoo -> the root of the ExecuTorch tree
ROOT = Path(__file__).resolve().parents[3]
TRANSFORMERS_VERSION = "5.19.0"
SIZES = ("device", "ci")
# Device nanos land at 20-30M parameters; nothing legitimate comes near this.
PARAM_CAP = 40_000_000
# Prompt length of the exported forward pass.
SEQ = 8

# Modules and packages here that are not families: the tools and the tests.
_TOOLS = {"build", "run", "census", "test"}


def families():
    """Every family module in the package, sorted by name."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__)
                  if not m.name.startswith("_") and m.name not in _TOOLS)


def family(name):
    """The family's module: MODEL_TYPE, MODEL_CLASS, DEVICE, CI and the optional
    INPUTS, FORWARD_KWARGS, OUTPUT and EXPECT (see build.py)."""
    if name not in families():
        raise KeyError(f"no nano family {name!r}; have {', '.join(families())}")
    return importlib.import_module(f"{__name__}.{name}")
