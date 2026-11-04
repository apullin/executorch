# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight zero points and float offsets on the host runner, bit for bit.

Every asym fixture -- PT2E int8 per-channel/per-group/per-tensor zero
points with an Arm int32 bias, torchao int4 at group 32/128 and per channel at
decode and prefill, the affine split, GGUF and tinygemm float offsets, conv and
depthwise conv, a float-offset embedding, a tied head (split, and staged), and
a selected-row head -- runs through the host runner built from this tree, and
its output must equal its reference, mirrors.py's device arithmetic, in every
bit (FNV-1a of the output bytes, as the parity firmware prints). The same
rows run under QEMU through run_parity.py.

Also here, the external-weights path test_asym_refusals.py cannot reach: the
head's zero-point tables are named data in a real .ptd; without it, with the
table missing from it, or with a short table, the method is refused for
weight-table-external.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import re
import subprocess

import numpy as np
import pytest

from executorch.backends.esp32s3.test.fixtures import asym
from executorch.backends.esp32s3.test.models import output_bits
from executorch.backends.esp32s3.test.host import host_runner

REFUSED = re.compile(r"esp32: v24 refused \[([a-z0-9-]+)\]")


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    work = tmp_path_factory.mktemp("asym_weights")
    refs = asym.export_all(str(work), verbose=False)
    table = {f.name: f for f in asym.fixtures()}
    return work, refs, table


def run(runner, work, fixture, table, *, ptd=True, extra=()):
    """(returncode, output floats or None, stderr, stdout) of one row."""
    source = table[fixture.program_of or fixture.name]
    cmd = [str(runner), "--pte", str(work / f"{source.name}.pte")]
    if ptd and source.partitioner.get("external_weights"):
        cmd += ["--ptd", str(work / f"{asym.PTD_TAG}.ptd")]
    for i, tensor in enumerate(source.inputs):
        path = work / f"{fixture.name}_in{i}.bin"
        tensor.numpy().tofile(path)
        cmd += ["--input", str(path)]
    out = work / f"{fixture.name}_out.bin"
    cmd += ["--output", str(out), *extra]
    if fixture.split_pass:
        cmd.append("--min-latency")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    got = np.fromfile(out, dtype=np.float32) if done.returncode == 0 else None
    return done.returncode, got, done.stderr, done.stdout


@pytest.mark.parametrize("name", asym.ROWS)
def test_bits_equal_the_mirror(runner, exported, name):
    work, refs, table = exported
    rc, got, err, out = run(runner, work, table[name], table)
    assert rc == 0, err[-1500:]
    want = np.asarray(refs[name], dtype=np.float32).reshape(-1)
    assert got.size == want.size
    bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    assert output_bits(got) == output_bits(want), (
        f"{name}: {bad} of {want.size} outputs differ, max abs "
        f"{float(np.abs(got - want).max()):.3e}")
    if name in ("asym_i4_split", "asym_head", "asym_head_staged"):
        forks = re.search(r"esp32 split: (\d+) column-split", out)
        assert forks and int(forks.group(1)) > 0, out[-800:]


def test_scalar_path_is_the_same(runner, exported):
    """PIE off (--force-scalar): every dot takes the scalar loop, same bits."""
    work, refs, table = exported
    for name in ("asym_i8_m4", "asym_i4_m1", "asym_woff"):
        rc, got, err, _ = run(runner, work, table[name], table,
                              extra=("--force-scalar",))
        assert rc == 0, err[-1500:]
        assert output_bits(got) == output_bits(refs[name]), name


def test_head_tables_are_named_data(exported):
    """The head's zero points travel as .ptd entries (esp32_z_*), keyed by
    content: the embedding and the head share one copy of each table."""
    work, _, _ = exported
    pte = (work / "asym_head.pte").read_bytes()
    ptd = (work / f"{asym.PTD_TAG}.ptd").read_bytes()
    keys = set(re.findall(rb"esp32_z_[0-9a-f]{16}", pte))
    assert len(keys) == 1, keys
    assert next(iter(keys)) in ptd


def test_external_tables_refused_without_named_data(runner, exported):
    work, _, table = exported
    rc, _, err, _ = run(runner, work, table["asym_head"], table, ptd=False)
    assert rc != 0 and REFUSED.findall(err)[:1] == ["weight-table-external"], \
        err[-800:]


@pytest.mark.parametrize("damage", ["missing", "short"])
def test_damaged_ptd_tables_refused(runner, exported, tmp_path, monkeypatch,
                                    damage):
    """A .ptd whose zero-point entry is absent, or shorter than the table the
    descriptor needs: refused at init for weight-table-external, not read."""
    from executorch.backends.esp32s3.serialization import blob_writer
    from executorch.backends.esp32s3.serialization import espressif_blob

    original = blob_writer._BlobWriter.store_table

    def store_table(self, kind, payload, elem):
        if kind != "z" or not self.cfg.external_weights:
            return original(self, kind, payload, elem)
        if damage == "short":
            return original(self, kind, payload[: len(payload) // 2], elem)
        key = espressif_blob.external_data_key(kind, payload)       # named, never stored
        return self.place(key.encode() + b"\0", align=1)

    monkeypatch.setattr(blob_writer._BlobWriter, "store_table", store_table)
    fixture = next(f for f in asym.fixtures() if f.name == "asym_head")
    et, _ = asym.lower(fixture)
    (tmp_path / "asym_head.pte").write_bytes(et.buffer)
    et.write_tensor_data_to_file(str(tmp_path))
    table = {"asym_head": fixture}
    rc, _, err, _ = run(runner, tmp_path, fixture, table)
    assert rc != 0 and REFUSED.findall(err)[:1] == ["weight-table-external"], \
        err[-800:]


def test_image_lists_the_qemu_rows(exported):
    """asym_fixtures.bin, the parity firmware's partition image, holds every
    row QEMU runs; the split row points at its program, the staged head (no
    PSRAM under QEMU) is left out."""
    work, _, table = exported
    rows = asym.read_image((work / "asym_fixtures.bin").read_bytes())
    names = [r[0] for r in rows]
    assert "asym_head_staged" not in names
    assert set(names) == {n for n, f in table.items() if f.qemu}
    by_name = {r[0]: r for r in rows}
    assert by_name["asym_i4_split"][1] == by_name["asym_i4_m1"][1]
    assert by_name["asym_i4_split"][3] & asym.FIXTURE_SPLIT_PASS
    assert by_name["asym_head"][3] & asym.FIXTURE_PTD
    assert by_name["asym_select"][2] == 257
    for name, program, _, _ in rows:
        assert program == (work / f"{(table[name].program_of or name)}.pte"
                           ).read_bytes()
