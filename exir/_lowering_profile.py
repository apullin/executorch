# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator


_ENABLED = os.environ.get("ET_LOWERING_PROFILE") == "1"
_LOCAL = threading.local()


def lowering_profile_enabled() -> bool:
    return _ENABLED


@contextmanager
def profile_scope(name: str, **fields: Any) -> Generator[None, None, None]:
    if not _ENABLED:
        yield
        return

    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    depth = len(stack)
    stack.append(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        stack.pop()
        suffix = "".join(f" {key}={value}" for key, value in fields.items())
        indent = "  " * depth
        print(
            f"[et-lower-profile] {elapsed:9.3f}s {indent}{name}{suffix}",
            file=sys.stderr,
            flush=True,
        )
