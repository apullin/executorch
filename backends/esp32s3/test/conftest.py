# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""pytest's session setup for this repository's tests.

histc_guard.install(): torch.histc's float32 kernel writes out of bounds when
a HistogramObserver range passes FLT_MAX / bins (Hugging Face's causal masks
put finfo.min into the scores the stock recipes observe), and the process
aborts later at some unrelated malloc; histc_guard.py explains how.
"""

from executorch.backends.esp32s3.test import histc_guard

histc_guard.install()
