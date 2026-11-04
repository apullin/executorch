# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded subprocess capture shared by QEMU and silicon runners."""

import codecs
import os
import selectors
import signal
import subprocess
import time


def capture_output(process, timeout, stop_phrases, on_output=None):
    """Read a child pipe without letting a silent child defeat ``timeout``.

    ``on_output`` may return true to stop after stateful inspection. This is
    used by the silicon monitor, where an initial USB reset is expected but a
    reset after the application banner terminates the run.
    """
    if process.stdout is None:
        raise RuntimeError("capture process has no stdout pipe")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks = []
    overlap = ""
    max_phrase = max((len(phrase) for phrase in stop_phrases), default=1)
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            events = selector.select(min(1.0, remaining))
            if not events:
                if process.poll() is not None:
                    break
                continue
            payload = os.read(process.stdout.fileno(), 65536)
            if not payload:
                break
            text = decoder.decode(payload)
            chunks.append(text)
            if on_output is not None and on_output(text):
                break
            scan = overlap + text
            if any(phrase in scan for phrase in stop_phrases):
                break
            overlap = scan[-max_phrase:]  # a stop phrase split across two reads is still found
        chunks.append(decoder.decode(b"", final=True))
        return "".join(chunks)
    finally:
        selector.close()


def stop_process_group(process):
    """Best-effort cleanup that never masks the runner's primary failure."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    if process.stdout is not None:
        try:
            process.stdout.close()
        except OSError:
            pass
