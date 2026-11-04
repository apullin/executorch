# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The CNN and RNN parity rows (fixtures/cnn.py) on the host build of the
runtime, without QEMU.

run_parity holds each row's device output bits to the host runner's. This
pins the host runner's bits to the ones the parity firmware printed for the
same programs (QEMU for every row it runs; ResNet-8, which needs PSRAM, has
no QEMU bits), so a change to the census models, the recipe, the lowering or
the runtime that moves a row fails here in about a minute instead of at the
next parity run. Building the rows also asserts what each leaves outside its
delegates, where it may run, and the census's bar against the recipe's eager
model (fixtures.cnn.fixtures).

Also: a row's export is deterministic (two exports, the same bytes: the
parity checkpoint and any artifact hash rely on it); the merged named
data holds every program's keys with their bytes; the image packs every row
with its flags.

The QEMU bits hold only for the programs Apple silicon exports, where they
were recorded: each row calibrates a census model through eager float32,
which another machine rounds otherwise (APPLE_SILICON below). Everything
else runs everywhere.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner; the
fixture brings that build up to date first.
"""

import platform
import sys

import numpy as np
import pytest

from executorch.backends.esp32s3.test.fixtures import asym as ea
from executorch.backends.esp32s3.test.fixtures import cnn
from executorch.backends.esp32s3.test.models import output_bits

# The `output bits` main.cpp prints for each row under QEMU (run_parity,
# ALL PASS); None for a row QEMU cannot run. The
# split-pass twin runs DS-CNN's program under min_latency, its convs on one
# core: the plain row's bits.
QEMU_BITS = {
    "cnn_arm_ds_cnn": "34f0229e",
    "cnn_arm_resnet8": None,
    "rnn_arm_lstm": "b6d346f6",
    "rnn_arm_gru": "cd1c737b",
    # The 5x5 depthwise rows (run_parity, ALL PASS): the plane path's and
    # the pairs form's assembly end to end.
    "cnn_arm_dw5_plane": "fb0335ee",
    "cnn_arm_dw5_pairs": "47fddef9",
    "cnn_arm_ds_cnn_split": "34f0229e",
}
# Those bits are the device's for the programs exported on Apple silicon. A
# row's program is a census model calibrated through eager float32 --
# convolutions, matmuls, sigmoid and tanh -- which x86 rounds otherwise: from
# the same parameters and the same calibration batches an AVX-512 Threadripper
# computes other activations, so other scales (on 18 of DS-CNN's 32 quantize
# and dequantize nodes, 73 of the LSTM's 349 and 113 of the GRU's 394, each in
# its last bit) and another program, whose device bits nobody has recorded.
# (The census's torch.randn draws, its BatchNorm statistics and recurrent
# batches, differ there too.)
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


@pytest.fixture(scope="module")
def rows():
    return {f.name: f for f in cnn.fixtures()}


def test_rows_are_the_fixture_table():
    assert tuple(QEMU_BITS) == cnn.ROWS
    assert [n for n, bits in QEMU_BITS.items() if bits is None] == [
        r.name for r in cnn.SPECS if r.board_only]


@pytest.mark.skipif(not APPLE_SILICON, reason=(
    "QEMU_BITS are the device's bits for the programs Apple silicon exports: "
    "each row calibrates a census model through eager float32, which this "
    "machine rounds otherwise, so it exports other programs"))
@pytest.mark.parametrize("name", cnn.QEMU_ROWS)
def test_host_bits_are_the_devices(rows, name):
    assert output_bits(rows[name].reference) == QEMU_BITS[name]


@pytest.mark.parametrize("name", cnn.ROWS)
def test_every_mac_delegated(rows, name):
    report = rows[name].report
    assert report["delegates"] > 0
    assert report["macs"] > 0 and report["macs_delegated"] == report["macs"]


def test_export_is_deterministic():
    row = cnn.SPECS[0]
    first = cnn.lower(*cnn.quantized(row)[:2])
    again = cnn.lower(*cnn.quantized(row)[:2])
    assert bytes(first[0].buffer) == bytes(again[0].buffer)
    assert first[1] == again[1]


def test_named_data_holds_every_program_once(rows):
    from executorch.exir._serialize._cord import Cord
    from executorch.extension.flat_tensor.serialize.serialize import (
        FlatTensorSerializer,
    )

    def entries(blob):
        payload = FlatTensorSerializer().deserialize(Cord(blob))
        return {key: bytes(payload.buffers[e.buffer_index])
                for key, e in payload.named_data.items()}

    merged = entries(cnn.fixtures.weights)
    union = {}
    for fixture in rows.values():
        own = entries(fixture.named)
        assert own, fixture.name
        union.update(own)
    assert merged == union


def test_image_packs_every_row(rows, tmp_path):
    table = list(rows.values())
    data = cnn.write_image(str(tmp_path / cnn.IMAGE), table)
    entries = ea.read_image(data)
    assert [e[0] for e in entries] == list(cnn.ROWS)
    for (name, program, modulus, flags), fixture in zip(entries, table):
        want = (ea.FIXTURE_PTD | cnn.FIXTURE_HEAP
                | (cnn.FIXTURE_PSRAM if fixture.row.board_only else 0)
                | (ea.FIXTURE_SPLIT_PASS if fixture.split_pass else 0))
        assert program == fixture.program
        assert modulus == 0 and flags == want, name


def test_split_twin_splits_no_conv(rows):
    """The min_latency twin runs DS-CNN's own program, and the host runner
    split none of its nine convs: no convolution splits across the cores
    (the backend README's Known limits). The export also held its bits to the
    plain row's; its temp is the plain row's (no split worker's scratch), within
    the parity firmware's temp arena."""
    twin = rows["cnn_arm_ds_cnn_split"]
    assert twin.split_pass and twin.program_of == "cnn_arm_ds_cnn"
    assert twin.program == rows["cnn_arm_ds_cnn"].program
    assert twin.report["convsplit"] == 0
    assert twin.report["temp"] <= cnn.PARITY_TEMP
    assert twin.report["temp"] == rows["cnn_arm_ds_cnn"].report["temp"]


def test_references_are_finite(rows):
    for name, fixture in rows.items():
        assert np.isfinite(fixture.reference).all(), name


@pytest.mark.parametrize("name", [r.name for r in cnn.SPECS
                                  if r.name.startswith("cnn_")])
def test_conv_rows_never_store_into_their_constants(rows, name):
    """The conv rows read every weight where the host runner maps the named
    data, read-only (MmapDataLoader maps the program and the named data
    PROT_READ), as the board's flash measurement reads its weights partition:
    a store into a constant on any conv path faults here instead of passing
    unseen in writable memory. On one core and under min-latency (where no
    conv splits), the weight tiers off (no copy is ever read), the row's own
    bits."""
    fixture = rows[name]
    runner = cnn.host_runner()
    in_place = ("--weight-tier-budget", "0", "--no-weight-cache")
    for flags in (("--single-core",), ("--min-latency",)):
        out, _, splits = cnn.run_host(runner, fixture.program, cnn.fixtures.weights,
                                      fixture.inputs[0], flags + in_place,
                                      splits=True)
        assert out.tobytes() == fixture.reference.tobytes(), (name, flags)
        assert splits == 0, (name, flags, splits)
