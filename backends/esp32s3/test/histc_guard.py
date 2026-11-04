# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torch.histc without its float32 overflow, so the stock recipes'
calibration does not abort.

HistogramObserver (XNNPACK's, Arm's and Cortex-M's static activation
observer) calls torch.histc(x, 2048, min=lo, max=hi) on every observed
tensor. The CPU kernel bins each element in float32 as

    pos = (int64)((x - lo) * bins / (hi - lo))

and once hi - lo passes FLT_MAX / bins (about 1.7e35 at 2048 bins) the product
overflows to +inf for the elements near hi. On arm64 the conversion saturates
to INT64_MAX and the index arithmetic wraps to -1: the kernel writes 4 bytes
before its thread's slice of the per-thread histogram. Thread 0's slice starts
the heap block, so the write lands on whatever precedes it, and macOS's malloc
aborts later ("Fatal Python error: Aborted", malloc_zone_error) wherever it
next notices: in histc's own frees, in HistogramObserver's repeat_interleave,
in a BLAS allocation. Hugging Face's causal masks put torch.finfo(float32).min
into attention scores, so any recipe that observes the masked add sees such
a range: the aborts are intermittent (about one calibration process in twenty;
a single thread does not avoid it).
torch.histc(torch.tensor([finfo.min, 0., .5, 1.]), 2048, finfo.min, 1.)
counts 1 element of 4 here; in float64, 4.

install() replaces torch.histc with a function that is torch.histc itself
whenever the float32 kernel cannot overflow -- the same call, the same bits --
and bins in float64 when it can (the counts the kernel means: every element
in range, the upper ones in the top bins). Only observers whose range passes
1.7e35 see a different histogram, where the kernel's is wrong. Idempotent;
installed by quant_census.recipes.quantize and conftest.py.
"""

import numpy as np
import torch

_HISTC = torch.histc


def _overflows(lo, hi, bins):
    """Whether (x - lo) * bins can overflow float32 for some x in [lo, hi],
    computed in float32 as the kernel computes it (x - lo <= hi - lo)."""
    with np.errstate(over="ignore"):
        span = np.float32(hi) - np.float32(lo)
        return not np.isfinite(span * np.float32(bins))


def histc(input, bins=100, min=0, max=0, *, out=None):  # noqa: A002 -- torch's names
    lo, hi = float(min), float(max)
    if (out is None and input.dtype == torch.float32 and lo < hi
            and _overflows(lo, hi, bins)):
        return _HISTC(input.double(), bins, lo, hi).to(torch.float32)
    if out is None:
        return _HISTC(input, bins, min, max)
    return _HISTC(input, bins, min, max, out=out)


def install():
    """Make torch.histc the guarded function (once per process)."""
    if torch.histc is not histc:
        torch.histc = histc
