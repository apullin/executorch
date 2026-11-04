# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The story golden's digest and cold-start line, off the board.

main/story_digest.h is what the story firmware folds every prefill block's and
decode step's logits into, in order, and prints as STORY_GOLDEN digest=0x...;
a board comparison holds it like every other field, which makes comparing two
silicon runs a bit-identity check. Here the header is built on the host with a
small driver and held to FNV-1a's published 64-bit values, and to a Python
transcription chained over several float rows as story_runner.inc chains it.
run_story_hardware.py must read a golden line with the digest (kept as text)
and one without (a firmware that prints no digest), and the STORY_COLD line.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "main")
sys.path.insert(0, HERE)
from executorch.examples.espressif.esp32s3.story import run_story_hardware as story  # noqa: E402

BASIS = 0xCBF29CE484222325
PRIME = 0x100000001B3

DRIVER = r"""
#include <cstdio>
#include <cstdlib>
#include "story_digest.h"
// Each argument is one buffer, as hex bytes; the buffers are chained in order.
int main(int argc, char** argv) {
  uint64_t hash = STORY_DIGEST_BASIS;
  for (int i = 1; i < argc; ++i) {
    static unsigned char buffer[1 << 16];
    size_t n = 0;
    for (const char* p = argv[i]; p[0] && p[1] && n < sizeof(buffer); p += 2) {
      const char pair[3] = {p[0], p[1], 0};
      buffer[n++] = (unsigned char)std::strtoul(pair, nullptr, 16);
    }
    hash = story_digest_bytes(hash, buffer, n);
  }
  std::printf("0x%016llx\n", (unsigned long long)hash);
  return 0;
}
"""


def fnv1a(data, value=BASIS):
    for byte in data:
        value = ((value ^ byte) * PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


@pytest.fixture(scope="module")
def digest(tmp_path_factory):
    build = tmp_path_factory.mktemp("story_digest")
    source = build / "driver.cpp"
    source.write_text(DRIVER)
    binary = build / "digest"
    subprocess.run(["c++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", MAIN,
                    str(source), "-o", str(binary)], check=True)

    def run(*buffers):
        out = subprocess.run([str(binary), *(bytes(b).hex() for b in buffers)],
                             check=True, capture_output=True, text=True).stdout
        return int(out, 16)

    return run


def test_fnv1a_published_values(digest):
    for text, want in ((b"", 0xCBF29CE484222325), (b"a", 0xAF63DC4C8601EC8C),
                       (b"foobar", 0x85944171F73967E8)):
        assert fnv1a(text) == want, text
        assert digest(text) == want, text


def test_steps_chain_in_order(digest):
    rng = np.random.default_rng(5)
    rows = [rng.standard_normal(64).astype("<f4") for _ in range(3)]
    chained = digest(*(row.tobytes() for row in rows))
    assert chained == fnv1a(b"".join(row.tobytes() for row in rows))
    # The order the steps ran in is part of the digest.
    assert digest(rows[1].tobytes(), rows[0].tobytes(), rows[2].tobytes()) != chained
    # One ulp in one logit moves it, and so does a zero's sign.
    nudged = [row.copy() for row in rows]
    nudged[2].view("<u4")[17] += 1
    assert digest(*(row.tobytes() for row in nudged)) != chained
    zeros = np.zeros(4, dtype="<f4")
    assert digest(zeros.tobytes()) != digest((-zeros).tobytes())


GOLDEN_NO_DIGEST = ("STORY_GOLDEN result=MATCH tokens_match=24/24 near_ties=0 hard=0 first_diff=-1 "
                    "early_max=6.758e-08 tol_early=1.0e-05 early_steps=4 onset=7 median_rel=1.003e-05 "
                    "max_rel=1.686e-02 values=OK prefill_blocks=0 prefill_us=0 decode_steps=42 "
                    "decode_us=3604440")


def test_run_story_hardware_reads_the_digest_and_the_cold_start():
    old = story.parse_golden(GOLDEN_NO_DIGEST + "\n")
    assert "digest" not in old and old["dus"] == 3604440
    log = (GOLDEN_NO_DIGEST + " digest=0x00c0ffee12345678\n"
           "STORY_COLD program_load_us=8123 prefill_load_us=0 prefill_place_us=0 "
           "decode_load_us=412345 decode_place_us=1534567 decode_ms_per_token=85.82\n")
    new = story.parse_golden(log)
    assert new["digest"] == "0x00c0ffee12345678" and new["dus"] == 3604440
    assert story.parse_cold(GOLDEN_NO_DIGEST) is None
    assert story.parse_cold(log) == {
        "program_load_us": 8123, "prefill_load_us": 0, "prefill_place_us": 0,
        "decode_load_us": 412345, "decode_place_us": 1534567, "decode_ms_per_token": 85.82}
    assert len(new["digest"]) == 2 + 16  # 64 bits, zero-padded as the firmware prints them
