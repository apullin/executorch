# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic raw-serial capture for ESP32-S3 hardware fixtures.

The firmware waits at a hardware-only READY gate and repeats READY while it
waits. This helper selects the known board by immutable USB serial, opens its
USB-Serial/JTAG console without resetting it, answers READY with GO, and
enforces separate startup and workload deadlines. It deliberately does not use
the interactive ESP-IDF monitor.

Within the startup deadline it recovers from the ways the handshake can
fail: a board that started without a host (the story firmware starts
alone 8 s after boot) or that never shows READY is reset through the console's
RTS line and caught at its next gate; a console that vanishes (the USB device
re-enumerates after a reset) is reopened; one that stops taking bytes (pySerial
"Write timeout": the firmware is past its gate and not reading) is reset.
"""

import argparse
import sys
import time


READY_MARKER = b"=== HARDWARE READY ==="
START_MARKER = b"=== HARDWARE RUNNING ==="
DONE_MARKER = b"=== DONE ==="
PANIC_MARKER = b"Guru Meditation"
REBOOT_MARKERS = (b"rst:0x", b"ESP-ROM:", READY_MARKER)
# Printed only past the gate: seen before RUNNING, the board started without
# this host. (A serving firmware answers the blank-line probe with REQUEST READY.)
STARTED_MARKERS = (b"=== HARDWARE RUNNING (no host) ===", DONE_MARKER,
                   b"=== REQUEST READY ===", b"=== REQUEST DONE ===")
MARKER_WIDTH = max(len(marker) for marker in (
    START_MARKER, PANIC_MARKER, *REBOOT_MARKERS, *STARTED_MARKERS))
GO = b"G\n"
GO_INTERVAL = .2        # at most one GO per READY (the gate repeats it every 0.5 s)
# A console silent since it opened gets GO every 0.25 s in this window (seconds
# after opening): a firmware whose only READY went out before the port opened.
BLIND_GO = (.3, 2.)
READY_WITHIN = 5.       # no READY this long after opening: past its gate; reset
MAX_RESETS = 3


class CaptureError(RuntimeError):
    pass


def normalize(value):
    return "".join(character.lower() for character in (value or "")
                   if character.isalnum())


def select_port(ports, serial_number):
    wanted = normalize(serial_number)
    matches = [port.device for port in ports
               if normalize(getattr(port, "serial_number", None)) == wanted]
    if len(matches) != 1:
        raise CaptureError(
            f"board serial {serial_number} did not resolve uniquely: "
            f"{matches}")
    return matches[0]


def open_serial(serial_class, port, baud=115200):
    """Open USB-Serial/JTAG without creating a reset edge."""
    stream = serial_class(
        port=None, baudrate=baud, timeout=.1, write_timeout=1)
    # This is ESP-IDF monitor's safe open sequence, minus its hard reset.
    stream.rts = True
    stream.dtr = True
    stream.port = port
    stream.open()
    stream.rts = False
    stream.dtr = False
    stream.reset_input_buffer()
    return stream


def reopen_serial(serial_class, list_ports, serial_number, baud, deadline,
                  clock=time.monotonic, sleep=time.sleep):
    """Open the board's console as soon as it is enumerated again."""
    while True:
        try:
            port = select_port(list_ports.comports(), serial_number)
            return open_serial(serial_class, port, baud)
        except (CaptureError, OSError) as error:
            if clock() >= deadline:
                raise CaptureError(
                    f"console did not come back: {error}") from error
            sleep(.1)


def hard_reset(stream, sleep=time.sleep):
    """esptool's HardReset for USB-Serial/JTAG: an RTS pulse with DTR low,
    so the chip boots the application, not the ROM download mode."""
    stream.rts = True
    stream.dtr = stream.dtr  # esptool re-sends DTR with each RTS change
    sleep(.2)
    stream.rts = False
    stream.dtr = stream.dtr
    sleep(.2)


def _last_new(scan, marker, fresh_from):
    """Start of the last copy of marker that ends in the new bytes, or -1."""
    at = scan.rfind(marker)
    return at if at >= 0 and at + len(marker) > fresh_from else -1


def capture_stream(
        stream, output, *, startup_timeout=10, workload_timeout=900,
        clock=time.monotonic, reconnect=None, reset=None, sleep=time.sleep):
    """Start one gated fixture and copy its bytes through ``output``.

    GO answers each READY; a console silent since it opened also gets GO for
    a moment, for a firmware whose single READY went out before the port
    opened. DONE counts only after HARDWARE RUNNING, so stale bytes from a
    prior boot cannot produce a false pass.

    ``reconnect(deadline)`` returns a newly opened console and
    ``reset(stream)`` restarts the board through it. With both, the startup
    phase survives a board that started without this host, one that never
    shows READY, a console that vanishes and one that stops taking bytes;
    without them each of those fails at once. Returns the console in use at
    the end, which need not be ``stream``.
    """
    startup_deadline = clock() + startup_timeout
    workload_deadline = None
    resets = 0
    overlap = b""
    opened = heard = ready_seen = last_go = next_blind = None

    def probe():
        # Gates ignore a blank line; a serving firmware answers REQUEST READY.
        nonlocal opened, heard, ready_seen, last_go, next_blind, overlap
        opened = clock()
        heard = ready_seen = False
        last_go = None
        next_blind = opened + BLIND_GO[0]
        overlap = b""
        try:
            stream.write(b"\n")
        except OSError:
            pass  # not reading: READY_WITHIN decides

    def recover(why, restart):
        nonlocal stream, resets
        if reconnect is None or (restart and reset is None):
            raise CaptureError(why)
        if restart:
            if resets == MAX_RESETS:
                raise CaptureError(f"{why} (after {resets} resets)")
            resets += 1
            print(f"[hardware-capture] {why}; resetting the board "
                  f"({resets}/{MAX_RESETS})", file=sys.stderr, flush=True)
            try:
                reset(stream)
            except OSError:
                pass  # the console is already gone: the board restarted
        try:
            stream.close()
        except Exception:
            pass
        # Let a resetting board leave USB before looking for it again.
        sleep(.5 if restart else .1)
        stream = reconnect(startup_deadline)
        probe()

    def send_go():
        try:
            stream.write(GO)
        except OSError as error:
            # A board at its gate reads every byte: this one is past it.
            recover(f"console stopped taking bytes ({error})", restart=True)
            return False
        return True

    probe()
    while True:
        try:
            payload = stream.read(65536)
        except OSError as error:
            if workload_deadline is not None:
                raise CaptureError(
                    f"console lost after acknowledging GO: {error}") from error
            recover(f"console lost before GO ({error})", restart=False)
            continue
        if payload:
            output.write(payload)
            output.flush()
            heard = True
            fresh_from = len(overlap)
            scan = overlap + payload
            if PANIC_MARKER in scan:
                raise CaptureError("device reported Guru Meditation")
            completion_scan = scan if workload_deadline is not None else b""
            if workload_deadline is None:
                start = scan.find(START_MARKER)
                if start >= 0:
                    workload_deadline = clock() + workload_timeout
                    completion_scan = scan[start + len(START_MARKER):]
                else:
                    started = max(_last_new(scan, marker, fresh_from)
                                  for marker in STARTED_MARKERS)
                    ready = _last_new(scan, READY_MARKER, fresh_from)
                    if started > ready:
                        recover("board started without this host",
                                restart=True)
                        continue
                    if ready >= 0:
                        ready_seen = True
                        if last_go is None or clock() - last_go >= GO_INTERVAL:
                            if not send_go():
                                continue
                            last_go = clock()
            if workload_deadline is not None and any(
                    marker in completion_scan for marker in REBOOT_MARKERS):
                raise CaptureError("device reset after acknowledging GO")
            if workload_deadline is not None and DONE_MARKER in completion_scan:
                return stream
            overlap = scan[-MARKER_WIDTH:]

        now = clock()
        if workload_deadline is None:
            if now >= startup_deadline:
                raise CaptureError(
                    f"fixture did not acknowledge GO within "
                    f"{startup_timeout:g}s")
            since = now - opened
            if not heard and BLIND_GO[0] <= since <= BLIND_GO[1] \
                    and now >= next_blind:
                # pySerial.flush() is tcdrain(2), which can block indefinitely
                # on macOS native USB Serial/JTAG when the target is not
                # consuming bytes; write_timeout bounds the write itself.
                if not send_go():
                    continue
                last_go = now
                next_blind = now + .25
            elif not ready_seen and since >= READY_WITHIN:
                recover(f"no READY within {READY_WITHIN:g}s", restart=True)
        elif now >= workload_deadline:
            raise CaptureError(
                f"fixture did not reach DONE within {workload_timeout:g}s "
                "after acknowledging GO")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-number", required=True,
                        help="the board's USB serial number")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--startup-timeout", type=float, default=10)
    parser.add_argument("--workload-timeout", type=float, default=900)
    args = parser.parse_args(argv)

    from serial import Serial
    from serial.tools import list_ports

    def reconnect(deadline):
        return reopen_serial(Serial, list_ports, args.serial_number,
                             args.baud, deadline)

    stream = None
    try:
        # The board may still be enumerating after a flash or a reset.
        stream = reconnect(time.monotonic() + args.startup_timeout)
        print(f"[hardware-capture] board {args.serial_number} at "
              f"{stream.port}", flush=True)
        stream = capture_stream(
            stream, sys.stdout.buffer,
            startup_timeout=args.startup_timeout,
            workload_timeout=args.workload_timeout,
            reconnect=reconnect, reset=hard_reset)
        return 0
    except Exception as error:
        print(f"[hardware-capture] ERROR: {error}", file=sys.stderr,
              flush=True)
        return 2
    finally:
        if stream is not None:
            try:
                stream.rts = False
                stream.dtr = False
                stream.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
