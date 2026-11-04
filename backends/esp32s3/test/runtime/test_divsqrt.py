# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The division and square root check (divsqrt_golden.py, divsqrt_check.cpp).

  golden    divsqrt_golden.h is what the generator makes, and every case
            in it is what exact rational rounding gives (the generator also
            checks each against numpy's IEEE float32 / float64 on this machine)
  rounding  the exact rounding helpers on hand-checked values: subnormal ties
            to even, overflow at FLT_MAX + half an ulp, signed zeros, and the
            double rounding of the portable reciprocal (1.0 / x in double,
            narrowed) equal to a single float32 rounding on every hard case
  host      the host runner's --selftest line: 0 mismatches (the Mac rounds
            correctly, so this is the check's positive control); the device's
            line is run_parity's gate under QEMU and run_hardware's on the board

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner. The golden
search takes about 15 s and 1.1 GB (no torch).
"""

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from executorch.backends.esp32s3.test.runtime import divsqrt_golden as dg
from executorch.backends.esp32s3.test.host import host_runner

ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "divsqrt_golden.h"


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def test_golden_header_is_current():
    assert HEADER.read_text() == dg.golden_header()


def test_round_float_ties_overflow_and_zeros():
    # 2^-150: halfway between 0 and denorm_min, to even (0); 3 * 2^-150 to 2.
    assert dg.round_float(Fraction(1, 2 ** 150)) == 0
    assert dg.round_float(Fraction(3, 2 ** 150)) == 2
    assert dg.round_float(Fraction(-5, 2 ** 150)) == 0x80000002
    flt_max = dg.exact(0x7F7FFFFF)
    half_ulp = Fraction(2) ** 103
    assert dg.round_float(flt_max + half_ulp - Fraction(1, 2 ** 200)) == 0x7F7FFFFF
    assert dg.round_float(flt_max + half_ulp) == 0x7F800000
    assert dg.round_float(Fraction(0), negative_zero=True) == 0x80000000
    assert dg.div(0x80000000, 0x3F800000) == 0x80000000          # -0 / 1
    assert dg.div(0x3F800000, 0x80000000) == 0xFF800000          # 1 / -0
    assert dg.div(0, 0) == dg.ANY_NAN32
    assert dg.sqrt(0x80000000) == 0x80000000                     # sqrt(-0)
    assert dg.sqrt(0xBF800000) == dg.ANY_NAN32


def test_hardest_cases_are_hard_and_exact():
    """Each hardest quotient lies within 2^-20 of a unit in the bit past the
    last from a midpoint or a float; its golden is the exact rounding and the
    float32 division of this machine."""
    for A, B in dg.hardest_quotients():
        q = Fraction(A * 2 ** 24, B)
        frac = q - (q.numerator // q.denominator)
        assert min(frac, 1 - frac) < Fraction(1, 2 ** 20), (A, B)
    for a, b, q in dg.division_cases():
        if q == dg.ANY_NAN32:
            continue
        with np.errstate(all="ignore"):
            assert dg.bits32(np.float32(dg.f32(a)) / np.float32(dg.f32(b))) == q


def test_portable_reciprocal_double_rounding_is_one_rounding():
    """1.0 / x in double, narrowed, equals the float32 quotient on every
    finite hard case: the double rounding is harmless for division."""
    for x, r in dg.recip_cases():
        if dg.is_nan(x) or dg.is_inf(x) or dg.exact(x) == 0:
            continue
        assert r == dg.round_float(1 / dg.exact(x)), hex(x)


def test_host_selftest_line(runner):
    done = subprocess.run([str(runner), "--selftest"], capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(l for l in done.stdout.splitlines()
                if l.startswith("divsqrt check:"))
    assert " 0 mismatch(es)" in line and "HASH MISMATCH" not in line, line
    assert "MISMATCH divsqrt" not in done.stdout
