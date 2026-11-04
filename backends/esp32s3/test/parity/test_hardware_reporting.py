# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host gates for silicon-runner reporting and acceptance policy."""

import contextlib
import io
import os
import sys
import tempfile

from executorch.backends.esp32s3.test.parity.run_hardware import (
    _parse_bandwidth,
    _parse_tier_rates,
    main as run_hardware_main,
    report_bandwidth,
    report_decode_tiers,
    report_tier_rates,
)


LOG = (
    "  SRAM  seq read :   137.0 MB/s   (128 KB)\r\n"
    "  PSRAM seq read :    55.1 MB/s\r\n"
    "  FLASH seq read :    26.6 MB/s\r\n"
)


# The weight tiers' rows (main/bench_tiers.cpp) and the decode rows' tails.
TIERS = (
    "[mspi] psram din_mode 0x01249249 din_num 0x00000000 timing_cali 0x00000001\r\n"
    "  TIER_RATE tier=flash kernel=i4 cores=1 k=256 rows=8192 bytes=1081344 "
    "us=40652 best_us=40600 mb_s=26.60 check=1.234567e+03\r\n"
    "  TIER_RATE tier=psram kernel=i4 cores=2 k=256 rows=8192 bytes=1081344 "
    "us=12082 best_us=12050 mb_s=89.50 check=1.234567e+03\r\n"
)
DECODE = (
    "  decode_i4 step 0: chunk 1 pos 0  checksum 1.5  time 9000 us  bits 0000abcd\n"
    "  decode_i4 step 1: chunk 1 pos 1  checksum 2.5  time 9100 us  bits 0000abce\n"
    "  decode_i4_psram weight tier: budget -1, 500000 B of weights in PSRAM\n"
    "  decode_i4_psram step 0: chunk 1 pos 0  checksum 1.5  time 4000 us  bits 0000abcd\n"
    "  decode_i4_psram step 1: chunk 1 pos 1  checksum 2.5  time 4100 us  bits 0000abce\n"
)


def _report(log, report=report_bandwidth, *args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        report(log, *args)
    return output.getvalue()


def main():
    assert _parse_bandwidth(LOG) == {
        "SRAM": 137.0,
        "PSRAM": 55.1,
        "FLASH": 26.6,
    }
    rejected = _report(LOG)
    assert "SRAM DIVERGES" in rejected
    assert "DO NOT record flash_read_mb_s" in rejected
    assert "put flash_read_mb_s=26.6" not in rejected

    accepted = _report(LOG.replace("137.0", "240.0"))
    assert "put flash_read_mb_s=26.6" in accepted
    assert "DO NOT" not in accepted

    missing = _report(LOG.splitlines()[2] + "\n")
    assert "SRAM MISSING" in missing and "PSRAM MISSING" in missing
    assert "DO NOT record flash_read_mb_s" in missing

    rows = _parse_tier_rates(TIERS)
    assert [(r["tier"], r["kernel"], r["cores"], r["mb_s"]) for r in rows] == [
        ("flash", "i4", 1, 26.6), ("psram", "i4", 2, 89.5)]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        entries = report_tier_rates(TIERS, "papercolor", "story")
    assert len(entries) == 2 and "checksums agree" in output.getvalue()
    assert ('_measured("papercolor", "psram", "i4", 2, "story", 89.5,'
            in entries[1] and "mspi='0x01249249'" in entries[1])
    # Two tiers that computed different rows record nothing.
    with contextlib.redirect_stdout(io.StringIO()):
        assert report_tier_rates(TIERS.replace("check=1.234567e+03",
                                               "check=9.9e+03", 1),
                                 "papercolor", "story") == []
    assert "no weight-tier rate rows" in _report(LOG, report_tier_rates, "nano", "idf")

    decode = _report(DECODE, report_decode_tiers)
    assert "median step: flash 9,100 us, PSRAM 4,100 us (500,000 B copied)" in decode
    assert "identical at every step" in decode
    with contextlib.redirect_stdout(io.StringIO()):
        assert report_decode_tiers(DECODE) is True
        assert report_decode_tiers(DECODE.replace("4100 us  bits 0000abce",
                                                  "4100 us  bits 0000ffff")) is False
        assert report_decode_tiers(LOG) is None

    # --rates reads a saved log with no board: a story bench build's has the
    # story's golden line, so its rates are filed under the story's caches.
    with tempfile.TemporaryDirectory() as scratch:
        saved = os.path.join(scratch, "story_bench.log")
        with open(saved, "w") as f:
            f.write(TIERS + DECODE + "STORY_GOLDEN result=MATCH\n")
        argv, sys.argv = sys.argv, ["run_hardware.py", "--rates", saved,
                                    "--board", "papercolor"]
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                assert run_hardware_main() == 0
        finally:
            sys.argv = argv
    assert "(papercolor, story caches, mspi 0x01249249)" in output.getvalue()
    print("hardware reporting host gates: PASS")


def test_main():
    """Under pytest, what a script run checks: main() asserts each check
    (or returns non-zero), and this fails with it."""
    assert main() in (None, 0)


if __name__ == "__main__":
    main()
