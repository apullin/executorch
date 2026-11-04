# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The board: find it by its USB serial, flash it with esptool, run the firmware and
capture its output through hardware_capture.py."""

import os
import shutil
import subprocess
import shlex
import sys
import time

import serial

from executorch.examples.espressif.esp32s3.scripts import hardware_capture, runner_io


HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)  # examples/espressif/esp32s3

# Without an activated ESP-IDF, a script that sources $IDF_PATH/export.sh and
# exec's its arguments (see idf()).
IDFENV = os.environ.get("IDF_ENV_SCRIPT")


def idf(*args):
    if os.environ.get("IDF_PATH") and shutil.which("idf.py"):
        return ["idf.py", *args]
    if not IDFENV or not os.path.exists(IDFENV):
        raise SystemExit(f"no activated ESP-IDF and no IDF_ENV_SCRIPT helper ({IDFENV}): "
                         "source $IDF_PATH/export.sh, or set IDF_ENV_SCRIPT")
    return ["sh", IDFENV, "idf.py", *args]


def _run(command, label):
    result = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-3000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"{label} failed")
    return result.stdout


BUILD = os.path.join(PROJ, "build")
# The boards the scripts know. Each is found by its USB serial number, which the
# caller names (--serial): it identifies one physical board, so there is no default.
BOARD_PROFILES = {
    "nano": {"label": "Arduino Nano ESP32"},
    "papercolor": {"label": "M5Stack PaperColor"},
}


# Seconds the capture helper may spend getting GO acknowledged.
CAPTURE_STARTUP_S = 60


def _verified_post_reset_disconnect(log):
    """Recognize only esptool's harmless native-USB reset race.

    The Nano can vanish between a completely verified write and pySerial's
    final attempt to configure the just-reset USB device. Acceptance requires
    complete verify/exit/reset markers and one exact disconnect signature.
    """
    # Python buffers esptool stdout but not its traceback on some macOS paths,
    # so the traceback can appear textually before the final Wrote/Hash/Leaving
    # lines even though it happened afterward. Classify by the complete verified
    # transaction plus one of two exact native-USB reset-disconnect signatures.
    verified = all(marker in log for marker in (
        "Hash of data verified.",
        "Leaving...",
        "Hard resetting with a watchdog",
    ))
    serial_exception = (
        "A serial exception error occurred" in log
        and "Could not configure port" in log)
    macos_device_error = (
        "esp.watchdog_reset()" in log
        and "OSError: [Errno 83] Device error" in log)
    return verified and (serial_exception or macos_device_error)


def esptool(*args):
    """Run esptool in the same ESP-IDF environment as idf.py."""
    command = ["python", "-m", "esptool", *args]
    return (["sh", IDFENV, *command]
            if IDFENV and os.path.exists(IDFENV) else command)


def _filtered_flash_arguments(build, excluded_suffixes):
    with open(os.path.join(build, "flash_project_args")) as source:
        arguments = shlex.split(source.read())
    filtered = []
    excluded = set()
    index = 0
    while index < len(arguments):
        try:
            int(arguments[index], 0)
            is_offset = True
        except ValueError:
            is_offset = False
        if is_offset and index + 1 < len(arguments):
            filename = arguments[index + 1]
            matched = next(
                (suffix for suffix in excluded_suffixes
                 if filename.endswith(suffix)), None)
            if matched is not None:
                excluded.add(matched)
                index += 2
                continue
            filtered.extend(arguments[index:index + 2])
            index += 2
            continue
        filtered.append(arguments[index])
        index += 1
    missing = set(excluded_suffixes) - excluded
    if missing:
        raise RuntimeError(
            f"requested flash exclusions were not present: {sorted(missing)}")
    return filtered


def flash_device(
        port, baud, build=BUILD, timeout=300, *, app_only=False,
        excluded_suffixes=()):
    if app_only and excluded_suffixes:
        raise ValueError("app_only and excluded_suffixes are mutually exclusive")
    flash_arguments = (
        ["@flash_app_args"] if app_only
        else _filtered_flash_arguments(build, excluded_suffixes)
        if excluded_suffixes else ["@flash_project_args"]
    )
    command = esptool(
        "--chip", "esp32s3", "--port", port, "--baud", str(baud),
        # The Nano's native USB Serial/JTAG path takes default_reset at both
        # 1 and 2 Mbaud.
        "--before", "default_reset", "--after", "watchdog_reset", "write_flash",
        *flash_arguments,
    )
    process = subprocess.Popen(
        command,
        cwd=build,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        log = runner_io.capture_output(process, timeout, ())
        returncode = process.poll()
        if returncode is None:
            try:
                returncode = process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
    finally:
        runner_io.stop_process_group(process)
    if returncode is None:
        print(log[-5000:])
        raise RuntimeError(f"hardware flash exceeded {timeout} seconds")
    if returncode:
        if _verified_post_reset_disconnect(log):
            print(
                "  flash data hash-verified; native USB disconnected during "
                "post-flash reset (accepted)")
            return
        print(log[-5000:])
        raise RuntimeError(f"hardware flash failed with status {returncode}")


def _normalize(value):
    return "".join(character.lower() for character in (value or "")
                   if character.isalnum())


def _board_profile(board, serial):
    """The board's label and the USB serial number the caller named for it."""
    try:
        label = BOARD_PROFILES[board]["label"]
    except KeyError as error:
        raise ValueError(
            f"unknown board profile {board!r}; choose "
            f"{', '.join(BOARD_PROFILES)}") from error
    if not _normalize(serial):
        raise SystemExit(
            f"name the {label}'s USB serial number (--serial; "
            "`python -m serial.tools.list_ports -v` lists it)")
    return label, serial


def _serial_python():
    """The interpreter for pySerial work: this one when it has pySerial,
    else ESP-IDF's, which costs 2-3 s to source each time; after
    a flash or a reset those seconds come out of the story firmware's 8 s
    window for GO."""
    try:
        import serial.tools.list_ports  # noqa: F401
    except ImportError:
        return (["sh", IDFENV, "python"] if IDFENV and os.path.exists(IDFENV)
                else ["python"])
    return [sys.executable]


def _port_listing():
    try:
        from serial.tools import list_ports
    except ImportError:
        code = (
            "from serial.tools import list_ports; "
            "[print((p.device or '')+'\\t'+(p.serial_number or '')+'\\t'+"
            "(p.hwid or '')) for p in list_ports.comports()]"
        )
        listing = subprocess.run(
            [*_serial_python(), "-c", code], capture_output=True, text=True,
            check=True, timeout=10)
        return listing.stdout.splitlines()
    return [f"{p.device or ''}\t{p.serial_number or ''}\t{p.hwid or ''}"
            for p in list_ports.comports()]


def find_port(board="nano", serial=None):
    # Never infer the target from a generic serial-device glob: unrelated FPGA
    # and ESP boards can be attached simultaneously. Match the one immutable
    # USB serial number the caller named.
    label, serial = _board_profile(board, serial)
    ports = [line.split("\t", 1)[0] for line in _port_listing()
             if _normalize(serial) in _normalize(line)]
    if len(ports) != 1:
        raise SystemExit(
            f"{label} with USB serial {serial} was not "
            f"uniquely present ({ports}); pass --port only after checking it")
    return ports[0]


def validate_port(port=None, board="nano", serial=None):
    """Reject an explicit path unless it is the selected exact board."""
    label, serial = _board_profile(board, serial)
    resolved = find_port(board, serial)
    if port is not None and port != resolved:
        raise SystemExit(
            f"refusing port {port}: {label} serial {serial} is "
            f"currently {resolved}")
    return resolved


def wait_for_port(timeout=20, board="nano", serial=None):
    """Wait for the selected exact board after a native-USB reset."""
    label, serial = _board_profile(board, serial)
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            port = find_port(board, serial)
        except (SystemExit, subprocess.SubprocessError) as error:
            last_error = error
            time.sleep(.25)
            continue
        # Let macOS finish publishing the newly enumerated character device.
        time.sleep(.25)
        return port
    raise RuntimeError(
        f"{label} did not re-enumerate within {timeout}s: "
        f"{last_error}")


def run_device(port, timeout=900, build=BUILD, project=PROJ,
               board="nano", serial=None):
    # Hardware fixtures wait at a READY/GO gate, so automated capture needs no
    # reset and cannot miss a fast workload. Avoid idf.py monitor: it is an
    # interactive, nested PTY process that resets the target on startup.
    _, serial = _board_profile(board, serial)
    resolved = find_port(board, serial)
    if port != resolved:
        raise RuntimeError(
            f"capture port changed from {port} to {resolved}; call "
            "wait_for_port() after flashing")
    del build, project  # The helper independently rechecks USB identity.
    helper = os.path.join(HERE, "hardware_capture.py")
    command = [
        *_serial_python(), helper,
        "--serial-number", serial,
        # Room for the helper to reset a board that started without it and
        # catch the next boot, up to three times. A pass returns at once.
        "--startup-timeout", str(CAPTURE_STARTUP_S),
        "--workload-timeout", str(timeout),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        log = runner_io.capture_output(
            process, CAPTURE_STARTUP_S + timeout + 20, ())
        returncode = process.poll()
        if returncode is None:
            try:
                returncode = process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
    finally:
        runner_io.stop_process_group(process)
    if returncode is None:
        print(log[-6000:])
        raise RuntimeError("hardware capture exceeded its outer deadline")
    if returncode:
        print(log[-6000:])
        raise RuntimeError(
            f"hardware capture failed with status {returncode}")
    if "=== DONE ===" not in log:
        print(log[-6000:])
        raise RuntimeError("hardware capture exited without DONE")
    return log


class Board:
    def __init__(self, port: str, log, echo: bool):
        self.stream = hardware_capture.open_serial(serial.Serial, port)
        self.log = log
        self.echo = echo
        self.pending = b""

    def lines(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = self.stream.read(4096)
            if not chunk:
                continue
            self.pending += chunk
            while b"\n" in self.pending:
                raw, self.pending = self.pending.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").rstrip("\r")
                if self.log:
                    self.log.write(line + "\n")
                if self.echo:
                    print("   | " + line)
                yield line
        raise TimeoutError(f"no expected line within {timeout:.0f} s")

    def send(self, text: str) -> None:
        self.stream.write(text.encode("utf-8"))
        self.stream.flush()

    def wait_for(self, marker: str, timeout: float, nudge=None) -> list[str]:
        """Collect lines until `marker`; `nudge` is sent every 250 ms."""
        seen = []
        last_nudge = 0.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if nudge and time.monotonic() - last_nudge > 0.25:
                self.send(nudge)
                last_nudge = time.monotonic()
            try:
                for line in self.lines(0.3):
                    seen.append(line)
                    if "Guru Meditation" in line:
                        raise RuntimeError("board crashed: " + line)
                    if marker in line:
                        return seen
                    if "=== HARDWARE READY ===" in line and marker != "=== HARDWARE READY ===":
                        # A reset put the firmware back at the runner's gate.
                        nudge = "G\n"
            except TimeoutError:
                continue
        raise TimeoutError(f"{marker!r} not seen within {timeout:.0f} s")

    def ready(self, timeout: float) -> None:
        """Bring the firmware to the request prompt from any state."""
        self.send("\n")
        self.wait_for("=== REQUEST READY ===", timeout, nudge=None)

