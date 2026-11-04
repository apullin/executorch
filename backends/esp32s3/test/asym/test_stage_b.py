# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stage B: int8 codes between claimed ops and across delegate boundaries, on
by default, with Stage A's bits as the bar.

stage_b.py describes the rewrite and fixtures/asym_stage_b.py the parity rows.
Here:

  - the default: Esp32Partitioner(stage_b=...) is on by default (off under a
    float format policy), its spec rides only on the delegates that hold
    codes, and with stage_b=False the rewrite never runs and no delegate
    carries the spec;
  - identity: every static int8 and int16 parity row (and the Stage B
    rows) lowered with the spec on runs on the host runner bit for bit as with
    it off; a row Stage B leaves alone keeps its delegate blobs byte for byte;
  - the chain scratch a chain of codes saves, from the runtime's own init
    report;
  - the named refusals: an edge whose identity proof fails (the codes that
    differ are reported), an op between delegates that needs fp32, int16
    codes under the spec;
  - the runtime's half: int8 codes refused without the stage_b compile spec
    (act-stage-b / out-stage-b), and field combinations Stage B does not
    define refused under it.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import re
import struct
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

from executorch.backends.esp32s3.serialization import espressif_blob
from executorch.backends.esp32s3 import compile_spec
from executorch.backends.esp32s3 import _passes
from executorch.backends.esp32s3.test.fixtures import asym_a16
from executorch.backends.esp32s3.test.fixtures import asym_stage_b as sb
from executorch.backends.esp32s3.test.fixtures import asym_static as st
from executorch.backends.esp32s3._passes import stage_b
from executorch.backends.esp32s3.serialization.asym_records import ActQuant, OutQuant
from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32
qd = torch.ops.quantized_decomposed


@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def bits(a):
    return np.asarray(a, F32).reshape(-1).view(np.uint32)


def run(runner, tmp_path, program, inputs, name="m", flags=(), outputs=1):
    """(output, log) of one program on the host runner; output None on a
    failed run. Each input is written in the memory order its strides give
    (a channels-last row's NHWC, as the device fills it), in its own dtype
    (float32, or int64 by the device's rule); the outputs are read one after
    another."""
    pte = tmp_path / f"{name}.pte"
    pte.write_bytes(program if isinstance(program, bytes) else program.buffer)
    command = [str(runner), "--pte", str(pte), *flags]
    for i, x in enumerate(inputs):
        x = st.memory_order(x).numpy() if isinstance(x, torch.Tensor) \
            else np.asarray(x, F32)
        x.tofile(tmp_path / f"{name}.in{i}.bin")
        command += ["--input", str(tmp_path / f"{name}.in{i}.bin")]
    outs = [tmp_path / f"{name}.out{k}.bin" for k in range(outputs)]
    for out in outs:
        command += ["--output", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    log = done.stdout + done.stderr
    if done.returncode != 0:
        return None, log
    return np.concatenate([np.fromfile(out, dtype=F32) for out in outs]), log


def temp_bytes(log):
    """Every delegate's per-call temporary scratch, from its init report."""
    return [int(m) for m in re.findall(r"init: \d+ op\(s\), \d+ B blob, (\d+) B "
                                       r"temp", log)]


def general_temp_bytes(log):
    """temp_bytes less what the static int8 conv path (runtime/esp32_conv.h)
    adds for itself (its init report line): the general loops' scratch. The
    path quantises an fp32 input once into a code plane, which a Stage A conv
    needs and its Stage B twin, reading codes, does not."""
    temps = []
    for line in log.splitlines():
        found = re.search(r"init: \d+ op\(s\), \d+ B blob, (\d+) B temp", line)
        if found:
            temps.append(int(found.group(1)))
            continue
        found = re.search(r"convfast: \d+ conv op\(s\) on the static int8 path, "
                          r"(\d+) B of the temp", line)
        if found and temps:
            temps[-1] -= int(found.group(1))
    return temps


def blobs(program):
    """Each delegate blob of a program (magic E32D, version 24), in order."""
    data = bytes(program.buffer if hasattr(program, "buffer") else program)
    out, at = [], 0
    magic = struct.pack("<II", espressif_blob.MAGIC, espressif_blob.VERSION)
    while True:
        at = data.find(magic, at)
        if at < 0:
            return out
        n_ops, ext_off = struct.unpack_from("<II", data, at + 8)
        end = at + espressif_blob.BLOB_HEADER_BYTES + espressif_blob.OP_DESC_BYTES * n_ops
        if ext_off:
            count, = struct.unpack_from("<I", data, at + ext_off)
            end = at + ext_off + 4
            for _ in range(count):
                _, _, length = struct.unpack_from(espressif_blob.EXT_RECORD_HEADER_FORMAT,
                                                  data, end)
                end += struct.calcsize(espressif_blob.EXT_RECORD_HEADER_FORMAT)
                end += length + (-length) % 4
        out.append((at, data[at:end]))
        at += 4


# --- the default ---------------------------------------------------------------------

def _keys(specs):
    return [s.key for s in specs]


def _delegate_specs(program):
    """Each delegate's compile-spec keys, as the program carries them."""
    plan = program.executorch_program.execution_plan[0]
    return [_keys(d.compile_specs) for d in plan.delegates]


def test_spec_is_on_by_default():
    on = Esp32Partitioner().compile_spec
    assert on.stage_b is True and Esp32CompileSpec().stage_b is True
    assert compile_spec.SPEC_STAGE_B in _keys(on.to_list())
    assert compile_spec.SPEC_STAGE_B in _keys(Esp32Partitioner().delegation_spec.compile_specs)
    assert Esp32CompileSpec.from_list(on.to_list()) == on
    off = Esp32Partitioner(stage_b=False)
    assert off.compile_spec.stage_b is False
    assert compile_spec.SPEC_STAGE_B not in _keys(off.delegation_spec.compile_specs)
    assert Esp32CompileSpec.from_list(off.compile_spec.to_list()) == off.compile_spec
    # Absent on the wire is off.
    assert Esp32CompileSpec.from_list([]).stage_b is False
    # A float policy quantises nothing: off by default, refused when asked.
    assert Esp32CompileSpec(format_policy="fp32").stage_b is False
    assert Esp32Partitioner(format_policy="bf16").compile_spec.stage_b is False
    with pytest.raises(RuntimeError, match="stage_b"):
        Esp32CompileSpec(format_policy="fp32", stage_b=True)


def test_spec_off_changes_no_byte():
    """Off -- stage_b=False, passed explicitly since the default is on --
    the rewrite never runs (no report) and no delegate carries the spec. On,
    the spec rides only on the delegates that hold codes, so a row Stage B
    converts nothing in is the off program byte for byte."""
    row = st.SPECS[st.ROWS.index("st_xnn_sig_lut")]
    model, inputs, _ = st.quantized(row)
    off, _, report_off = sb.lower_program(model, inputs, stage_b=False)
    assert report_off.stage_b is None
    assert not any(compile_spec.SPEC_STAGE_B in keys for keys in _delegate_specs(off))
    default, _, report = sb.lower_program(model, inputs, stage_b=None)
    assert report.stage_b is not None and report.stage_b.converted
    specs = _delegate_specs(default)
    assert any(compile_spec.SPEC_STAGE_B in keys for keys in specs), specs
    assert bytes(default.buffer) != bytes(off.buffer)
    # The residual add needs fp32: Stage B keeps that region, and the program.
    row = st.SPECS[st.ROWS.index("st_xnn_residual")]
    model, inputs, _ = st.quantized(row)
    off, _, _ = sb.lower_program(model, inputs, stage_b=False)
    default, _, report = sb.lower_program(model, inputs, stage_b=None)
    assert report.stage_b is not None and not report.stage_b.converted
    assert bytes(default.buffer) == bytes(off.buffer)


# --- identity ------------------------------------------------------------------------

def _static_fixtures():
    return {f.name: f for f in st.fixtures()}


def _a16_fixtures():
    return {f.name: f for f in asym_a16.fixtures()}


@pytest.fixture(scope="module")
def static_rows():
    return _static_fixtures()


@pytest.fixture(scope="module")
def a16_rows():
    return _a16_fixtures()


def _stripped(fixture):
    """A dynamic recipe's row is lowered with its activation chains absorbed
    by lower_esp32's own strip right before partitioning, checked to run
    (strip_before_partitioning), as asym_static.lower and the census do."""
    if not getattr(fixture, "dynamic", False):
        return contextlib.nullcontext()
    from executorch.backends.esp32s3.test.quant_census.lower import (
        strip_before_partitioning,
    )

    return strip_before_partitioning({})


def _identity(runner, tmp_path, fixture):
    with _stripped(fixture):
        a, _, ra = sb.lower_program(fixture.model, fixture.inputs, fixture.luts,
                                    partitioner=fixture.partitioner)
    with _stripped(fixture):
        b, _, rb = sb.lower_program(fixture.model, fixture.inputs, fixture.luts,
                                    stage_b=True, partitioner=fixture.partitioner)
    n = getattr(fixture, "outputs", 1)
    oa, la = run(runner, tmp_path, a, fixture.inputs, "a", outputs=n)
    ob, lb = run(runner, tmp_path, b, fixture.inputs, "b", outputs=n)
    assert oa is not None, la[-1500:]
    assert ob is not None, lb[-1500:]
    assert np.array_equal(bits(oa), bits(ob))
    want = np.asarray(fixture.reference, F32).reshape(-1)
    if getattr(fixture, "bar", "exact") == "code":
        # A float reduction: Stage A's bar, one code of the mirror.
        gap = st.code_gap(want[:fixture.measured], ob[:fixture.measured],
                          fixture.grid)
        assert gap.ok(), gap.summary()
    else:
        assert np.array_equal(bits(ob), bits(want))
    report = rb.stage_b
    if getattr(fixture, "dynamic", False):
        # A dynamic recipe leaves no static pair to convert: every delegate
        # blob is Stage A's.
        assert report is None
        assert [x for _, x in blobs(a)] == [x for _, x in blobs(b)]
        return report
    assert report is not None and not report.identity
    if not report.converted:
        # Nothing travels as codes: every delegate blob is Stage A's.
        assert [x for _, x in blobs(a)] == [x for _, x in blobs(b)]
    else:
        assert sum(temp_bytes(lb)) < sum(temp_bytes(la)) or report.code_ops
    return report


@pytest.mark.parametrize("name", st.ROWS)
def test_static_rows_bit_identical(runner, tmp_path, static_rows, name):
    """Static rows: Stage B == Stage A == the mirror, on the host runner."""
    _identity(runner, tmp_path, static_rows[name])


@pytest.mark.parametrize("name", asym_a16.ROWS)
def test_a16_rows_bit_identical_and_refused_by_name(runner, tmp_path, a16_rows,
                                                    name):
    """The int16 rows: int16 codes are not part of Stage B, so each region
    over an int16 pair stays fp32, by name, and the blobs are Stage A's."""
    report = _identity(runner, tmp_path, a16_rows[name])
    assert not report.converted
    rules = {rule for _, rule, _ in report.kept}
    assert "stage-b-int16" in rules, report.summary()


@pytest.fixture(scope="module")
def stageb_rows():
    return {f.name: f for f in sb.fixtures()}


@pytest.mark.parametrize("name", [r for r in sb.ROWS if r.startswith("sb_")
                                  and not r.endswith("_split")])
def test_stageb_rows(runner, tmp_path, stageb_rows, name):
    """The parity rows: each Stage B variant does what its row says, and its
    host bits are its Stage A program's and the mirror's -- also with its
    columns split across the cores (--min-latency)."""
    variant = stageb_rows[name]
    a, _, _ = sb.lower_program(variant.model, variant.inputs, variant.luts)
    b, _ = sb.lower(variant)
    oa, la = run(runner, tmp_path, a, variant.inputs, "a")
    ob, lb = run(runner, tmp_path, b, variant.inputs, "b")
    assert oa is not None, la[-1500:]
    assert ob is not None, lb[-1500:]
    assert np.array_equal(bits(oa), bits(ob))
    assert np.array_equal(bits(ob), bits(variant.reference))
    os_, ls = run(runner, tmp_path, b, variant.inputs, "split",
                  ("--min-latency",))
    assert os_ is not None, ls[-1500:]
    assert np.array_equal(bits(os_), bits(variant.reference))


def test_chain_scratch_of_codes_is_a_quarter(runner, tmp_path, static_rows):
    """st_xnn_conv's three convolutions chain inside one delegate: their two
    chain slots hold int8 codes under Stage B, a quarter of fp32's bytes, so
    the delegate's temporary scratch shrinks by three quarters of them."""
    fixture = static_rows["st_xnn_conv"]
    a, _, _ = sb.lower_program(fixture.model, fixture.inputs)
    b, _, _ = sb.lower_program(fixture.model, fixture.inputs, stage_b=True)
    _, la = run(runner, tmp_path, a, fixture.inputs, "a")
    _, lb = run(runner, tmp_path, b, fixture.inputs, "b")
    # Each chain slot is sized to its own writer: the first conv's 81
    # positions x 8 channels, and the stride-2 conv's 25 x 8.
    saved = sum(e * 4 - (e + 15) // 16 * 16 for e in (81 * 8, 25 * 8))
    assert general_temp_bytes(la)[0] - general_temp_bytes(lb)[0] == saved


# --- the named refusals ------------------------------------------------------------------

class Linear8(nn.Module):
    def __init__(self, k, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("c", torch.randint(-127, 128, (n, k), generator=g,
                                                dtype=torch.int8))
        self.register_buffer("s", torch.rand(n, generator=g) * 2e-3 + 1e-4)
        self.register_buffer("z", torch.zeros(n, dtype=torch.int64))

    def forward(self, x):
        w = qd.dequantize_per_channel(self.c, self.s, self.z, 0, -127, 127,
                                      torch.int8)
        return F.linear(x, w)


def pair(x, s, z, qmin=-128, qmax=127, dtype=torch.int8):
    q = qd.quantize_per_tensor(x, s, z, qmin, qmax, dtype)
    return qd.dequantize_per_tensor(q, s, z, qmin, qmax, dtype)


class Chain(nn.Module):
    """input pair -> Linear -> pair(S, Z) -> Linear -> output pair."""

    S, Z = float(F32(0.021)), -3

    def __init__(self):
        super().__init__()
        self.fc1 = Linear8(32, 32, 1)
        self.fc2 = Linear8(32, 16, 2)

    def forward(self, x):
        h = pair(self.fc1(pair(x, float(F32(0.02)), 0)), self.S, self.Z)
        return pair(self.fc2(h), float(F32(0.01)), 2)


class Huge(Chain):
    # (c - z) * s overflows float32 past code -128 + 170: Stage A's
    # requantise sends codes 43..126 to the rail, so their round trip fails.
    S, Z = float(F32(2e36)), -128


class ClampChain(nn.Module):
    """Linear -> pair -> relu -> pair on another grid -> hardtanh -> pair ->
    hardtanh -> pair -> Linear: the relu between two grids is a code table
    (its codes are no clamp), the hardtanhs on the second grid's own codes two
    clamps of codes in a row."""

    S, Z = float(F32(0.02)), -10
    S2, Z2 = float(F32(0.013)), -100

    def __init__(self):
        super().__init__()
        self.fc1 = Linear8(32, 32, 3)
        self.fc2 = Linear8(32, 16, 4)

    def forward(self, x):
        h = pair(self.fc1(pair(x, float(F32(0.02)), 0)), self.S, self.Z)
        h = pair(F.relu(h), self.S2, self.Z2)
        h = pair(F.hardtanh(h, -0.5, 1.0), self.S2, self.Z2)
        h = pair(F.hardtanh(h, -1.0, 0.75), self.S2, self.Z2)
        return pair(self.fc2(h), float(F32(0.01)), 2)


def test_clamps_and_table_on_codes(runner, tmp_path):
    """A relu between grids becomes a code table, hardtanhs on one grid two
    clamps of codes in a row (canonicalize_clamps leaves an int8 chain
    alone), and the host bits are Stage A's."""
    x = torch.randn(4, 32)
    a, _, _ = sb.lower_program(ClampChain().eval(), (x,))
    b, log, static = sb.lower_program(ClampChain().eval(), (x,), stage_b=True)
    report = static.stage_b
    kinds = [k for _, k in report.code_ops]
    assert kinds.count("clamp") == 2 and kinds.count("table") == 1, \
        report.summary()
    clamps = [n for n in b.exported_program().graph.nodes
              if n.op == "call_function" and "clamp" in str(n.target)]
    assert len(clamps) == 2
    oa, la = run(runner, tmp_path, a, (x,), "a")
    ob, lb = run(runner, tmp_path, b, (x,), "b")
    assert oa is not None and ob is not None, (la + lb)[-1500:]
    assert np.array_equal(bits(oa), bits(ob))


def _report(model, x):
    _, log, static = sb.lower_program(model.eval(), (x,), stage_b=True)
    return static.stage_b, log


def test_identity_failures_name_the_codes():
    assert stage_b.identity_failures(stage_b.Grid(Chain.S, Chain.Z, -128,
                                                  127)) == []
    bad = stage_b.identity_failures(stage_b.Grid(Huge.S, Huge.Z, -128, 127))
    assert bad == list(range(43, 127))


def test_non_identity_edge_keeps_fp32(runner, tmp_path):
    """The Linear -> Linear chain converts on a sound grid; on one whose
    round trip fails it keeps fp32, its report naming the edge and codes."""
    x = torch.randn(4, 32)
    good, _ = _report(Chain(), x)
    assert len(good.converted) == 1 and not good.kept
    bad, log = _report(Huge(), x)
    assert not bad.converted
    [(edge, codes)] = bad.identity
    assert codes == list(range(43, 127))
    [(_, rule, detail)] = bad.kept
    assert rule == "stage-b-identity" and "43" in detail
    assert "stage-b-identity" in log
    a, _, _ = sb.lower_program(Huge().eval(), (x,))
    b, _, _ = sb.lower_program(Huge().eval(), (x,), stage_b=True)
    assert [y for _, y in blobs(a)] == [y for _, y in blobs(b)]


def test_op_between_delegates_that_needs_fp32_is_named(static_rows):
    """st_xnn_residual: the add between its two delegates reads dequantised
    values, so that boundary stays fp32, named by the add."""
    fixture = static_rows["st_xnn_residual"]
    _, _, static = sb.lower_program(fixture.model, fixture.inputs,
                                    stage_b=True)
    [(_, rule, detail)] = static.stage_b.kept
    assert rule == "stage-b-reader" and "add" in detail


def test_code_table_is_stage_a_through_the_next_quantize(stageb_rows):
    """The table op's code-to-code variant holds, for each code, the code
    Stage A's next quantize gives the fp32 table's entry."""
    from executorch.backends.esp32s3._passes import fold_quant_lut_pass

    fixture = stageb_rows["sb_siglut"]
    ep = torch.export.export(fixture.model, fixture.inputs, strict=False)
    report = fold_quant_lut_pass.fold_quant_luts(ep)
    [(_, build)] = report.tables
    lowered = _passes.lower_esp32(
        torch.export.export(fixture.model, fixture.inputs, strict=False),
        [Esp32Partitioner(stage_b=True)])
    prog = lowered.exported_program()
    [lut] = [n for n in prog.graph.nodes if "quant_lut_codes" in str(n.target)]
    table = fold_quant_lut_pass._constant(prog, lut.args[1])
    want = stage_b.quantize_values(build.table, stage_b.Grid(
        *build.out_qparams[:4]))
    assert table.dtype == torch.int8
    assert np.array_equal(table.to(torch.int64).numpy(), want)


# --- the runtime's half --------------------------------------------------------------------

def _records(blob):
    """[(record offset in blob, op, type, length)] of a blob's extension table."""
    ext_off = struct.unpack_from("<I", blob, 12)[0]
    count, = struct.unpack_from("<I", blob, ext_off)
    at, out = ext_off + 4, []
    for _ in range(count):
        op, typ, length = struct.unpack_from(espressif_blob.EXT_RECORD_HEADER_FORMAT,
                                             blob, at)
        at += struct.calcsize(espressif_blob.EXT_RECORD_HEADER_FORMAT)
        out.append((at, op, typ, length))
        at += length + (-length) % 4
    return out


@pytest.fixture(scope="module")
def conv_programs():
    fixture = _static_fixtures()["st_xnn_conv"]
    b, _, _ = sb.lower_program(fixture.model, fixture.inputs, stage_b=True)
    return bytes(b.buffer), fixture.inputs


def _refused(runner, tmp_path, data, inputs):
    out, log = run(runner, tmp_path, data, inputs, "patched")
    found = re.findall(r"v24 refused \[([a-z0-9-]+)\]", log)
    return out, found


def test_codes_without_the_spec_are_refused(runner, tmp_path, conv_programs):
    """The same blob under a delegate whose compile specs lack stage_b (its
    key renamed in place): op 0 writes codes, refused as out-stage-b."""
    data, inputs = conv_programs
    key = compile_spec.SPEC_STAGE_B.encode()
    assert data.count(key) == 1
    patched = data.replace(key, b"stage_x")
    out, found = _refused(runner, tmp_path, patched, inputs)
    assert out is None and found[:1] == ["out-stage-b"]


@pytest.mark.parametrize("field,value,rule", [
    ("granularity", espressif_blob.GRANULARITY_TENSOR, "act-combination"),
    ("code_bits", 16, "act-combination"),
    ("in_dtype", espressif_blob.DTYPE_INT16, "act-stage-b"),
])
def test_codes_in_fields_stage_b_does_not_define(runner, tmp_path,
                                                 conv_programs, field, value,
                                                 rule):
    data, inputs = conv_programs
    at, blob = blobs(data)[0]
    act = [r for r in _records(blob)
           if r[1] == 1 and r[2] == espressif_blob.EXT_ACT_QUANT][0]
    rec = ActQuant.unpack(blob[act[0]:act[0] + act[3]])
    assert rec.in_dtype == espressif_blob.DTYPE_INT8
    from dataclasses import replace

    mutated = replace(rec, **{field: value}).pack()
    off = at + act[0]
    patched = data[:off] + mutated + data[off + len(mutated):]
    out, found = _refused(runner, tmp_path, patched, inputs)
    assert out is None and found[:1] == [rule]


@pytest.mark.parametrize("field,value,rule", [
    ("qmax", 255, "out-range"),
    ("out_dtype", espressif_blob.DTYPE_INT16, "out-stage-b"),
])
def test_codes_out_fields_stage_b_does_not_define(runner, tmp_path,
                                                  conv_programs, field, value,
                                                  rule):
    data, inputs = conv_programs
    at, blob = blobs(data)[0]
    rec_at = [r for r in _records(blob)
              if r[1] == 0 and r[2] == espressif_blob.EXT_OUT_QUANT][0]
    rec = OutQuant.unpack(blob[rec_at[0]:rec_at[0] + rec_at[3]])
    assert rec.out_dtype == espressif_blob.DTYPE_INT8
    from dataclasses import replace

    mutated = replace(rec, **{field: value}).pack()
    off = at + rec_at[0]
    patched = data[:off] + mutated + data[off + len(mutated):]
    out, found = _refused(runner, tmp_path, patched, inputs)
    assert out is None and found[:1] == [rule]


def test_chain_of_mismatched_dtypes_is_refused(runner, tmp_path, conv_programs):
    """op 1 reads op 0's chain slot: with op 0's output record back to fp32
    the slot holds floats op 1 would read as codes -- refused at init."""
    data, inputs = conv_programs
    at, blob = blobs(data)[0]
    rec_at = [r for r in _records(blob)
              if r[1] == 0 and r[2] == espressif_blob.EXT_OUT_QUANT][0]
    rec = OutQuant.unpack(blob[rec_at[0]:rec_at[0] + rec_at[3]])
    from dataclasses import replace

    mutated = replace(rec, out_dtype=espressif_blob.DTYPE_FP32).pack()
    off = at + rec_at[0]
    patched = data[:off] + mutated + data[off + len(mutated):]
    out, log = run(runner, tmp_path, patched, inputs, "mismatch")
    assert out is None and "chain dtype mismatch" in log
