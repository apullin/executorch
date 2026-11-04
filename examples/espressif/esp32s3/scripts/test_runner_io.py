# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host gates for bounded QEMU and raw hardware capture."""

import io
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

from executorch.examples.espressif.esp32s3.scripts import board as arch_hardware
import executorch.examples.espressif.esp32s3.scripts.board as arch_hardware
from executorch.examples.espressif.esp32s3.scripts import hardware_capture
from executorch.examples.espressif.esp32s3.scripts import runner_io as runner
from executorch.examples.espressif.esp32s3.scripts.board import (
    _filtered_flash_arguments,
    _verified_post_reset_disconnect,
)


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


class FakeClock:
    def __init__(self, step=.001):
        self.now = 0.0
        self.step = step

    def __call__(self):
        result = self.now
        self.now += self.step
        return result


class FakeStream:
    def __init__(self, reads, refuse_go=False):
        self.reads = list(reads)
        self.writes = []
        self.refuse_go = refuse_go  # pySerial "Write timeout": not reading
        self.closed = False

    def write(self, payload):
        if self.refuse_go and payload == hardware_capture.GO:
            raise OSError("Write timeout")
        self.writes.append(payload)

    def flush(self):
        raise AssertionError("capture must not call blocking serial tcdrain")

    def read(self, _size):
        if not self.reads:
            return b""
        item = self.reads.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class FakeBoard:
    """reconnect()/reset() for capture_stream: each reset boots `consoles`."""

    def __init__(self, consoles):
        self.consoles = list(consoles)
        self.resets = []
        self.reconnects = 0

    def reconnect(self, _deadline):
        self.reconnects += 1
        return self.consoles.pop(0)

    def reset(self, stream):
        self.resets.append(stream)


def gated_capture(stream, board=None, clock=None, **deadlines):
    output = io.BytesIO()
    kwargs = dict(startup_timeout=30, workload_timeout=30,
                  clock=clock or FakeClock(), sleep=lambda _s: None)
    if board is not None:
        kwargs.update(reconnect=board.reconnect, reset=board.reset)
    kwargs.update(deadlines)
    final = hardware_capture.capture_stream(stream, output, **kwargs)
    return final, output.getvalue()


def capture_error(stream, board=None, clock=None, **deadlines):
    try:
        gated_capture(stream, board, clock, **deadlines)
    except hardware_capture.CaptureError as error:
        return str(error)
    return None


SERIAL = "02:00:00:00:00:01"  # a made-up board serial (locally administered MAC)


def main():
    nano = SimpleNamespace(device="/dev/cu.nano", serial_number=SERIAL)
    artix = SimpleNamespace(device="/dev/cu.artix", serial_number="ARTIX")
    check(
        "raw capture selects only immutable Nano identity",
        hardware_capture.select_port([artix, nano], SERIAL) == "/dev/cu.nano",
    )
    try:
        hardware_capture.select_port([nano, nano], SERIAL)
    except hardware_capture.CaptureError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    check("ambiguous Nano identity is rejected", duplicate_rejected)
    with mock.patch.object(
            arch_hardware, "find_port", return_value="/dev/cu.nano"):
        check(
            "hardware runner resolves and validates the immutable Nano port",
            arch_hardware.validate_port(serial=SERIAL) == "/dev/cu.nano"
            and arch_hardware.validate_port("/dev/cu.nano", serial=SERIAL)
            == "/dev/cu.nano",
        )
        try:
            arch_hardware.validate_port("/dev/cu.artix", serial=SERIAL)
        except SystemExit:
            wrong_port_rejected = True
        else:
            wrong_port_rejected = False
    check("explicit non-Nano hardware port is rejected", wrong_port_rejected)
    with mock.patch.object(
            arch_hardware, "find_port", return_value="/dev/cu.papercolor") \
            as find_profiled:
        check(
            "hardware runner selects the explicit PaperColor profile",
            arch_hardware.validate_port(
                "/dev/cu.papercolor", "papercolor", SERIAL) ==
            "/dev/cu.papercolor"
            and find_profiled.call_args.args == ("papercolor", SERIAL),
        )
    try:
        arch_hardware.find_port("nano")
    except SystemExit as error:
        unnamed = "--serial" in str(error)
    else:
        unnamed = False
    check("no default board: a port is found only by a named serial", unnamed)

    stream = FakeStream([
        b"boot\n=== HARDWARE READY ===\n",
        b"=== HARD",
        b"WARE RUNNING ===\nresult\n=== DO",
        b"NE ===\n",
    ])
    final, output = gated_capture(stream, startup_timeout=1, workload_timeout=1)
    check(
        "READY/GO capture answers READY once and spans split markers",
        final is stream and stream.writes == [b"\n", hardware_capture.GO]
        and b"=== HARDWARE RUNNING ===" in output
        and output.endswith(b"=== DONE ===\n"),
    )

    # One READY stays in the overlap the next reads are scanned with; it is
    # still one READY, and earns one GO however slowly the reads come.
    stream = FakeStream([b"=== HARDWARE READY ===\n", b"x", b"y", b"z",
                         b"=== HARDWARE RUNNING ===\n=== DONE ===\n"])
    gated_capture(stream, clock=FakeClock(step=.1))
    check("a READY earns one GO, not one per read",
          stream.writes.count(hardware_capture.GO) == 1)

    error = capture_error(FakeStream([b"stale === DONE ===\n=== HARD",
                                      b"WARE RUNNING ===\n=== DONE ===\n"]))
    check("DONE before RUNNING never passes", error is not None
          and "started without this host" in error)

    # The story firmware starts alone 8 s after boot: a host that opens the
    # port later resets the board and catches its next gate.
    alone = FakeStream([b"=== HARDWARE RUNNING (no host) ===\nloading\n"])
    rebooted = FakeStream([b"ESP-ROM:esp32s3-20210327\n",
                           b"=== HARDWARE READY ===\n",
                           b"=== HARDWARE RUNNING ===\nlog\n=== DONE ===\n"])
    board = FakeBoard([rebooted])
    final, output = gated_capture(alone, board)
    check(
        "a board that started without the host is reset and caught again",
        final is rebooted and board.resets == [alone] and alone.closed
        and alone.writes == [b"\n"]
        and rebooted.writes == [b"\n", hardware_capture.GO]
        and output.endswith(b"=== DONE ===\n"),
    )

    serving = FakeStream([b"=== HARDWARE READY ===\n=== REQUEST READY ===\n"])
    rebooted = FakeStream([b"=== HARDWARE READY ===\n",
                           b"=== HARDWARE RUNNING ===\n=== DONE ===\n"])
    board = FakeBoard([rebooted])
    gated_capture(serving, board)
    check("a READY older than a serving prompt gets no GO",
          serving.writes == [b"\n"] and board.resets == [serving])

    refusing = FakeStream([b"=== HARDWARE READY ===\n"], refuse_go=True)
    rebooted = FakeStream([b"=== HARDWARE READY ===\n",
                           b"=== HARDWARE RUNNING ===\n=== DONE ===\n"])
    board = FakeBoard([rebooted])
    final, _ = gated_capture(refusing, board)
    check("a write timeout resets and reopens the console",
          final is rebooted and board.resets == [refusing])

    vanished = FakeStream([OSError("device disconnected")])
    reopened = FakeStream([b"=== HARDWARE READY ===\n",
                           b"=== HARDWARE RUNNING ===\n=== DONE ===\n"])
    board = FakeBoard([reopened])
    final, _ = gated_capture(vanished, board)
    check("a vanished console is reopened without a reset",
          final is reopened and board.resets == [] and board.reconnects == 1)

    # A firmware that printed its only READY before the port opened stays
    # silent: GO goes out blind for a moment. If READY never comes the board
    # is past its gate and is reset.
    silent = FakeStream([])
    rebooted = FakeStream([b"=== HARDWARE READY ===\n",
                           b"=== HARDWARE RUNNING ===\n=== DONE ===\n"])
    board = FakeBoard([rebooted])
    gated_capture(silent, board, clock=FakeClock(step=.05))
    blind = silent.writes.count(hardware_capture.GO)
    check("a silent console gets a few blind GOs, then a reset",
          silent.writes[0] == b"\n" and 3 <= blind <= 8
          and board.resets == [silent])

    blind_start = FakeStream(
        [b""] * 6 + [b"=== HARDWARE RUNNING ===\n=== DONE ===\n"])
    gated_capture(blind_start, clock=FakeClock(step=.1))
    check("a blind GO starts a firmware that announced READY only once",
          hardware_capture.GO in blind_start.writes)

    always_alone = [FakeStream([b"=== HARDWARE RUNNING (no host) ===\n"])
                    for _ in range(hardware_capture.MAX_RESETS + 1)]
    board = FakeBoard(always_alone[1:])
    error = capture_error(always_alone[0], board)
    check("resets are bounded",
          error is not None and "after 3 resets" in error
          and len(board.resets) == hardware_capture.MAX_RESETS)

    error = capture_error(FakeStream([OSError("device disconnected")]))
    check("without reconnect a lost console fails at once",
          error is not None and "console lost" in error)

    class LineState:
        def __init__(self):
            self.changes = []
            self._rts = self._dtr = False

        @property
        def rts(self):
            return self._rts

        @rts.setter
        def rts(self, value):
            self._rts = value
            self.changes.append(("rts", value))

        @property
        def dtr(self):
            return self._dtr

        @dtr.setter
        def dtr(self, value):
            self._dtr = value
            self.changes.append(("dtr", value))

    lines = LineState()
    hardware_capture.hard_reset(lines, sleep=lambda _s: None)
    check(
        "hard reset pulses RTS with DTR low (application boot)",
        [c for c in lines.changes if c[0] == "rts"] == [("rts", True), ("rts", False)]
        and all(value is False for name, value in lines.changes if name == "dtr"),
    )

    class Port:
        device = "/dev/cu.nano"
        serial_number = SERIAL

    class Listing:
        def __init__(self):
            self.calls = 0

        def comports(self):
            self.calls += 1
            return [] if self.calls == 1 else [Port()]

    class OpenOnSecondTry:
        attempts = 0

        def __init__(self, **kwargs):
            self.port = None
            self.rts = self.dtr = False

        def open(self):
            OpenOnSecondTry.attempts += 1
            if OpenOnSecondTry.attempts == 1:
                raise OSError("Resource busy")

        def reset_input_buffer(self):
            pass

    listing = Listing()
    console = hardware_capture.reopen_serial(
        OpenOnSecondTry, listing, SERIAL, 115200,
        deadline=10, clock=FakeClock(), sleep=lambda _s: None)
    check("the console is reopened once it is enumerated and free",
          console.port == "/dev/cu.nano" and listing.calls == 3)

    class NeverThere:
        @staticmethod
        def comports():
            return []

    try:
        hardware_capture.reopen_serial(
            OpenOnSecondTry, NeverThere, SERIAL, 115200,
            deadline=1, clock=FakeClock(step=.5), sleep=lambda _s: None)
    except hardware_capture.CaptureError as error:
        gone = "did not come back" in str(error)
    else:
        gone = False
    check("reopening gives up at the deadline", gone)

    check("the capture helper runs in this interpreter when it has pySerial",
          arch_hardware._serial_python() == [sys.executable])

    startup_timeout = FakeStream([])
    try:
        hardware_capture.capture_stream(
            startup_timeout, io.BytesIO(), startup_timeout=.5,
            clock=FakeClock(step=1))
    except hardware_capture.CaptureError as error:
        startup_bounded = "acknowledge GO" in str(error)
    else:
        startup_bounded = False
    check("raw capture has a distinct startup deadline", startup_bounded)

    workload_timeout = FakeStream([b"=== HARDWARE RUNNING ===\n"])
    try:
        hardware_capture.capture_stream(
            workload_timeout, io.BytesIO(), startup_timeout=5,
            workload_timeout=.5, clock=FakeClock(step=1))
    except hardware_capture.CaptureError as error:
        workload_bounded = "after acknowledging GO" in str(error)
    else:
        workload_bounded = False
    check("raw capture has a distinct workload deadline", workload_bounded)

    rebooted = FakeStream([
        b"=== HARDWARE RUNNING ===\n",
        b"rst:0x15 (USB_UART_CHIP_RESET)\n",
    ])
    try:
        hardware_capture.capture_stream(
            rebooted, io.BytesIO(), startup_timeout=5,
            workload_timeout=5, clock=FakeClock())
    except hardware_capture.CaptureError as error:
        reboot_detected = "reset after" in str(error)
    else:
        reboot_detected = False
    check("raw capture fails immediately on a post-GO reset", reboot_detected)

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "flash_project_args"), "w") as out:
            out.write(
                "--flash_mode qio 0x10000 app.bin 0x170000 ple.pte "
                "0x194000 ple_weights.ptd")
        filtered = _filtered_flash_arguments(
            directory, ("ple_weights.ptd",))
        check(
            "selective flash excludes only named PTD",
            filtered == ["--flash_mode", "qio", "0x10000", "app.bin",
                         "0x170000", "ple.pte"],
        )
    verified_disconnect = (
        "Hash of data verified.\nLeaving...\n"
        "Hard resetting with a watchdog...\n"
        "A serial exception error occurred: Could not configure port")
    check(
        "verified post-reset USB disconnect classification",
        _verified_post_reset_disconnect(verified_disconnect)
        and not _verified_post_reset_disconnect(
            verified_disconnect.replace("Hash of data verified.", "write failed"))
        and not _verified_post_reset_disconnect(
            verified_disconnect.replace("Hard resetting with a watchdog...", "")),
    )
    reordered_macos_disconnect = (
        "Traceback: esp.watchdog_reset()\n"
        "OSError: [Errno 83] Device error\n"
        "Wrote 14078080 bytes\nHash of data verified.\nLeaving...\n"
        "Hard resetting with a watchdog...")
    check(
        "verified reordered macOS reset disconnect classification",
        _verified_post_reset_disconnect(reordered_macos_disconnect)
        and not _verified_post_reset_disconnect(
            reordered_macos_disconnect.replace("Hash of data verified.\n", ""))
        and not _verified_post_reset_disconnect(
            reordered_macos_disconnect.replace("esp.watchdog_reset()", "write")),
    )
    child = subprocess.Popen(
        [sys.executable, "-c", (
            "import os,time; os.write(1,b'=== DO'); time.sleep(.02); "
            "os.write(1,b'NE ===\\n'); time.sleep(10)")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        captured = runner.capture_output(child, 1, ("=== DONE ===",))
    finally:
        runner.stop_process_group(child)
    check(
        "runner capture stops across split output",
        "=== DONE ===" in captured and time.monotonic() - started < 1,
    )

    silent = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        captured = runner.capture_output(silent, .05, ("DONE",))
    finally:
        runner.stop_process_group(silent)
    check(
        "runner capture enforces a silent-child deadline",
        not captured and time.monotonic() - started < 1,
    )

    callback_child = subprocess.Popen(
        [sys.executable, "-c", (
            "import os,time; os.write(1,b'booted\\n'); time.sleep(10)")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    seen = []
    started = time.monotonic()
    try:
        captured = runner.capture_output(
            callback_child, 1, (),
            on_output=lambda chunk: seen.append(chunk) or "booted" in chunk,
        )
    finally:
        runner.stop_process_group(callback_child)
    check(
        "runner capture honors a stateful callback stop",
        "booted" in captured and seen and time.monotonic() - started < 1,
    )

    class PermissionDeniedProcess:
        pid = 123456789
        stdout = None

        @staticmethod
        def wait(timeout):
            del timeout
            return 0

    with mock.patch.object(runner.os, "killpg", side_effect=PermissionError):
        runner.stop_process_group(PermissionDeniedProcess())
    check("cleanup cannot mask a primary failure with EPERM", True)
    print("bounded runner I/O host gates PASS")


if __name__ == "__main__":
    main()
