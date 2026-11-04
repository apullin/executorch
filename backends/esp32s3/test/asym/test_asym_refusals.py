# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The asymmetric-quantisation refusal rules on the host runner: one synthetic
blob per rule.

The asymmetric-quantisation records are validated in runtime/esp32_asym.cpp,
which computes the weight records, the dynamic int8 and the static activation
records, the output records and the affine K cache, and refuses the rest as
unimplemented.
asym_blob_cases.py builds a blob per rule -- a valid blob with exactly one
field wrong -- and this runs each through the host runner built from this
tree, inside the carrier program, and requires the refusal to name the case's
OWN rule: the runtime logs "esp32: v24 refused [<rule>] ...", and the first
such line must be it. A case refused by some other rule fails, which
"load_method failed" alone could not show. The valid blobs must pass every
rule: those the runtime computes must load, the rest be refused only as
"unimplemented"; the unmodified carrier must load and run. (An accepted case
then fails at execute, on the carrier's own tensor shapes: executing affine
ops is test_asym_weights.py's and test_asym_act.py's.) The device half is main.cpp's
run_asym_refusal_check under QEMU, through run_parity.py.

Also here: the records' wire round trip, the schemes refused by name at export
(asym_records.py; the backend's own A0 and XNNPACK's per-token A4 among them,
which no record carries), every other reference activation scheme of
mirrors.py as a record the runtime accepts, and the execute-time K-cache
zero-point check (esp32_host_runner --selftest).

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner. It imports
torch to export the carrier.
"""

import re
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from executorch.backends.esp32s3.test.fixtures import asym_blob_cases as cases
from executorch.backends.esp32s3.serialization import espressif_blob as eb
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.serialization.asym_records import (
    ActQuant,
    OutQuant,
    SchemeRefused,
    WeightAffine,
    act_quant,
    activation_chooser,
    out_quant,
)
from executorch.backends.esp32s3.test.host import host_runner

ROOT = Path(__file__).resolve().parent
REFUSED = re.compile(r"esp32: v24 refused \[([a-z0-9-]+)\]")


@pytest.fixture(scope="session")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@pytest.fixture(scope="session")
def carrier(tmp_path_factory):
    work = tmp_path_factory.mktemp("asym_carrier")
    data = cases.export_carrier(work / "asym_carrier.pte")
    return work, bytes(data), cases.payload_span(bytes(data))


def run_case(runner, carrier, name, blob):
    """(returncode, the rules the refusal lines name, stderr) for `blob` run
    in the carrier."""
    work, data, span = carrier
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
    pte = work / f"{slug}.pte"
    pte.write_bytes(data if blob is None else cases.patched(data, span, blob))
    x = np.linspace(-1.0, 1.0, cases.CARRIER_FEATURES, dtype=np.float32)
    x.tofile(work / "in.bin")
    done = subprocess.run(
        [str(runner), "--pte", str(pte), "--method", "forward",
         "--input", str(work / "in.bin"), "--output", str(work / "out.bin")],
        capture_output=True, text=True, timeout=120)
    return done.returncode, REFUSED.findall(done.stderr), done.stderr


def test_every_rule_has_a_case():
    covered = set(cases.covered_rules())
    assert cases.accepted_cases(), "no valid case is computed"
    assert covered == set(cases.RULES), sorted(set(cases.RULES) ^ covered)
    # The runtime's own list, in its order: none, every case rule, and the
    # execute-time rule driven by the selftest.
    text = (ROOT.parents[1] / "runtime" / "esp32_asym.cpp").read_text()
    body = re.search(r"kRuleNames\[\]\s*=\s*\{(.*?)\};", text, re.S).group(1)
    runtime = re.findall(r'"([a-z0-9-]+)"', body)
    assert runtime[0] == "none"
    assert set(runtime[1:]) == set(cases.RULES) | set(cases.EXECUTE_TIME_RULES)
    assert len(runtime) == len(set(runtime))


def test_records_round_trip():
    for record in (cases.A3, cases.STATIC_A, cases.A16, cases.OUT_Q,
                   WeightAffine(zp_kind=eb.ZP_INT8, offset_kind=eb.OFFSET_FP16,
                                sum_kind=eb.SUM_INT32, storage_bias=8,
                                zp_off=64, offset_off=96, sum_off=128)):
        payload = record.pack()
        assert len(payload) in (eb.ACT_QUANT_BYTES, eb.WEIGHT_AFFINE_BYTES,
                                eb.OUT_QUANT_BYTES)
        assert type(record).unpack(payload) == record


def test_export_refusals_by_name():
    # A static scale that is not a float32 value: 0.1 is not, 2^-5 is.
    with pytest.raises(SchemeRefused, match="not a float32 value"):
        act_quant(qm.A5.static(0.1, 3))
    assert act_quant(qm.A5.static(0.03125, 3)).scale == 0.03125
    with pytest.raises(SchemeRefused, match="not a float32 value"):
        out_quant(0.1, 0, -128, 127)
    # Unsigned symmetric activations.
    unsigned = replace(qm.A1, qmin=0, qmax=255)
    with pytest.raises(SchemeRefused, match="unsigned symmetric"):
        act_quant(unsigned)
    # torchao's SYMMETRIC_NO_CLIPPING_ERR, by enum name or string.
    with pytest.raises(SchemeRefused, match="SYMMETRIC_NO_CLIPPING_ERR"):
        activation_chooser("MappingType.SYMMETRIC_NO_CLIPPING_ERR")
    # No record carries the backend's own A0 (an op without a record
    # quantises so) or XNNPACK's per-token A4 (the K cache's own codes).
    with pytest.raises(SchemeRefused, match=r"backend's own"):
        act_quant(qm.A0)
    with pytest.raises(SchemeRefused, match="per-token chooser"):
        act_quant(qm.A4)
    assert activation_chooser("ASYMMETRIC") == "affine"
    assert activation_chooser("SYMMETRIC") == "symmetric"


# Every reference activation scheme (mirrors.py) a record carries, as the
# record exporters write it: each must pass every rule. The dynamic per-row
# and per-tensor schemes, the static int8 A5 (and its uint8 form) and the
# static int16 A6 are all computed (they load, then fail at execute on the
# carrier's shapes). A0 and A4 have no record (test_export_refusals_by_name).
SCHEMES = {
    "A1": qm.A1, "A2/xnnpack": qm.A2_XNNPACK,
    "A2/arm": qm.A2_ARM, "A3/torchao": qm.A3_TORCHAO,
    "A3/legacy": qm.A3_LEGACY,
    "A5": qm.A5.static(0.03125, -7), "A5/uint8": qm.A5_UINT8.static(0.03125, 131),
    "A6": qm.A6.static(2.0 ** -10), "A7/torchao": qm.A7_TORCHAO,
}


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_reference_schemes_are_valid_records(runner, carrier, name):
    record = act_quant(SCHEMES[name])
    op = cases.Op(eb.OP_ADDMM_I8, flags=eb.FLAG_ACT_QUANT, act=record)
    if record.affine:
        op = replace(op, flags=op.flags | eb.FLAG_WEIGHT_AFFINE,
                     weight=WeightAffine(sum_kind=eb.SUM_INT16))
    rc, refused, err = run_case(runner, carrier, f"scheme {name}",
                                cases.build(op))
    assert not refused and "execute forward" in err, (refused, err[-600:])


@pytest.mark.parametrize("case", cases.CASES, ids=lambda c: c.name)
def test_refused_for_its_own_rule(runner, carrier, case):
    rc, refused, err = run_case(runner, carrier, case.name, case.blob)
    if case.rule == cases.ACCEPTED:
        # Computed: no rule refuses it and the method loads (it
        # then fails at execute on the carrier's tensor shapes, which is not a
        # load error).
        assert not refused and "load method" not in err, (
            f"{case.name}: refused for {refused}\n{err[-800:]}")
        assert "execute forward" in err, err[-800:]
        return
    assert rc != 0, f"{case.name}: loaded\n{err[-600:]}"
    assert refused[:1] == [case.rule], (
        f"{case.name}: expected [{case.rule}], refused for {refused}\n"
        f"{err[-800:]}")


def test_carrier_control_runs(runner, carrier):
    rc, refused, err = run_case(runner, carrier, "carrier control", None)
    assert rc == 0 and not refused, err[-800:]
    work = carrier[0]
    out = np.fromfile(work / "out.bin", dtype=np.float32)
    assert out.shape == (cases.CARRIER_FEATURES,) and np.isfinite(out).all()


def test_carrier_is_a_symmetric_v24_blob(carrier):
    """The carrier comes from the normal export path: version 24, and none of
    the asymmetric-quantisation flags or records -- what every symmetric
    model's blob is."""
    _, data, (offset, length) = carrier
    magic, version, n_ops, ext_off = struct.unpack_from(
        eb.BLOB_HEADER_FORMAT, data, offset)
    assert (magic, version, ext_off) == (eb.MAGIC, 24, 0)
    v24 = eb.FLAG_ACT_QUANT | eb.FLAG_WEIGHT_AFFINE | eb.FLAG_OUT_QUANT
    for i in range(n_ops):
        desc = dict(zip(eb.OP_DESC_FIELDS, struct.unpack_from(
            eb.OP_DESC_FORMAT, data,
            offset + eb.BLOB_HEADER_BYTES + i * eb.OP_DESC_BYTES)))
        assert desc["opcode"] == eb.OP_ADDMM_I8 and not desc["flags"] & v24
    assert max(len(c.blob) for c in cases.CASES) <= length


def test_kcache_zero_point_check(runner):
    done = subprocess.run([str(runner), "--selftest"], capture_output=True,
                          text=True, timeout=300)
    assert "asym kcache zp check: control=accepted dtype=refused shape=refused" \
        in done.stdout, done.stdout[-1500:]
    assert REFUSED.findall(done.stderr).count("kcache-zero-points") == 2
