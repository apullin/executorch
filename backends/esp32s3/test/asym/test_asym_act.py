# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic activation quantisers: the host runner bit for bit, and every
refusal by name.

Every asym_act fixture -- PT2E's affine chooser at XNNPACK's and Arm's
eps into int8 weights per channel and per tensor (symmetric and zero points),
torchao's per-token A3 and symmetric A7 on its own grid, the uint8 chooser,
8da4w heads at decode (the fused kernel) and prefill with a shared row, the
affine split, 8da8w zero points, float offsets under a zero point, the
mixed-scheme sharing group, every eps floor and rounding tie, every conv kind
with its padding written as zx, the selected and the tied head (split, and
staged), and mlp_ao_asym -- runs through the host runner
built from this tree, and its output must equal its reference, mirrors.py's
device arithmetic, in every bit (FNV-1a of the output bytes, as the parity
firmware prints). The same rows run under QEMU through run_parity.py.

Also here: what the exporter refuses by name (DynamicChainRefused,
asym_records' bound checks, the partitioner's unclaimed-consumer rule), each
with the smallest graph that reaches it -- torchao SYMMETRIC_NO_CLIPPING_ERR,
dynamic int16, unsigned symmetric ranges, per-channel or
per-group scales along K, non-float32 scales, a zero-point dtype that cannot
hold the codes, a chooser feeding anything but its own quantize, a chooser
no chain absorbs, the exactness bounds -- and that the activation-sharing
group forms only among identical records.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import re
import subprocess

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops
import torch.nn as nn

from executorch.backends.esp32s3.test.fixtures import asym_act as eaa
from executorch.backends.esp32s3.quantizer import mirrors as qm
from executorch.backends.esp32s3.test.models import output_bits
from executorch.backends.esp32s3.test.host import host_runner



@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    work = tmp_path_factory.mktemp("asym_act")
    refs = eaa.export_all(str(work), verbose=False)
    table = {f.name: f for f in eaa.fixtures()}
    return work, refs, table


def outputs_of(name):
    if name.startswith("asym_a3_i4"):
        return 4
    return {"asym_actshare": 5, "asym_ties": len(eaa.TIES),
            "asym_a2_conv13": 2, "asym_a3_s32": 2}.get(name, 1)


def run(runner, work, fixture, table, *, extra=()):
    """(returncode, every output as one flat float array or None, stderr,
    stdout) of one row."""
    source = table[fixture.program_of or fixture.name]
    cmd = [str(runner), "--pte", str(work / f"{source.name}.pte")]
    if source.partitioner.get("external_weights"):
        cmd += ["--ptd", str(work / f"{eaa.PTD_TAG}.ptd")]
    for i, tensor in enumerate(source.inputs):
        path = work / f"{fixture.name}_in{i}.bin"
        tensor.numpy().tofile(path)
        cmd += ["--input", str(path)]
    outs = [work / f"{fixture.name}_out{i}.bin"
            for i in range(outputs_of(fixture.name))]
    for out in outs:
        cmd += ["--output", str(out)]
    cmd += list(extra)
    if fixture.split_pass:
        cmd.append("--min-latency")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    got = None
    if done.returncode == 0:
        got = np.concatenate([np.fromfile(o, dtype=np.float32) for o in outs])
    return done.returncode, got, done.stderr, done.stdout


@pytest.mark.parametrize("name", eaa.ROWS)
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
    if name in ("asym_a3_i4_split", "asym_a3_head", "asym_a3_head_staged"):
        forks = re.search(r"esp32 split: (\d+) column-split", out)
        assert forks and int(forks.group(1)) > 0, out[-800:]


def test_int32_weight_sums_on_a_device_row():
    """asym_a3_s32's heads carry int32 weight sums (S' past
    int16: groups of 256 with zero points, a 512-long symmetric row), so
    the QEMU row runs the device's int32-sum path, which no other device
    row reaches."""
    fixture = next(f for f in eaa.fixtures() if f.name == "asym_a3_s32")
    _, log = eaa.lower(fixture)
    blob = next(line for line in log.splitlines() if " op(s), " in line)
    assert blob.count("+sum32") == 2 and "+sum16" not in blob, blob


def test_scalar_path_is_the_same(runner, exported):
    """PIE off (--force-scalar): every dot takes the scalar loop, same bits."""
    work, refs, table = exported
    for name in ("asym_a2x_i8", "asym_a3_i4_m1", "asym_a3_woff",
                 "asym_a2_conv"):
        rc, got, err, _ = run(runner, work, table[name], table,
                              extra=("--force-scalar",))
        assert rc == 0, err[-1500:]
        assert output_bits(got) == output_bits(refs[name]), name


@pytest.mark.parametrize("name, rows_saved", [("asym_actshare", 4),
                                              ("asym_a3_i4_m1", 1),
                                              ("asym_a3_i4_m8", 8)])
def test_identical_records_share_one_row(runner, exported, name, rows_saved):
    """The activation-group key: the two identical A3 records (and the two
    group-128 heads) quantise their row once; the A2, A7 and backend heads
    of asym_actshare, and the group-32 and per-channel heads, do not join."""
    work, _, table = exported
    rc, _, err, out = run(runner, work, table[name], table)
    assert rc == 0, err[-800:]
    assert "actq: 1 shared-input group(s)" in out, out[-1500:]
    saved = re.search(r"esp32 actq: (\d+) shared-input row", out)
    assert saved and int(saved.group(1)) == rows_saved, out[-800:]


# --- refused by name, at export ----------------------------------------------------------

def _lower(model, x, **partitioner):
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    edge = to_edge(torch.export.export(model.eval(), (x,)))
    with contextlib.redirect_stdout(io.StringIO()):
        strip_dynamic_act_qdq(edge)
        return edge.to_backend(Esp32Partitioner(**partitioner))


class Fc(nn.Module):
    """A Linear behind whatever activation chain `body` builds."""

    def __init__(self, body, k=16, n=8):
        super().__init__()
        self.fc = nn.Linear(k, n)
        self.body = body

    def forward(self, x):
        return self.fc(self.body(x))


def _torchao(x, mapping="ASYMMETRIC", dtype=torch.int8, qmin=-128, qmax=127,
             block=None, scale_dtype=None, zp_dtype=torch.int32, eps=None):
    import torchao  # noqa: F401

    block = block or [1, int(x.shape[-1])]
    ops = torch.ops.torchao
    s, z = ops.choose_qparams_affine(x, mapping, block, dtype, qmin, qmax, eps,
                                     scale_dtype, zp_dtype)
    q = ops.quantize_affine(x, block, s, z, dtype, qmin, qmax)
    return ops.dequantize_affine(q, block, s, z, dtype, qmin, qmax)


def _pt2e(x, dtype=torch.int8, qmin=-128, qmax=127, symmetric=False,
          eps=qm.EPS_XNNPACK):
    qd = torch.ops.quantized_decomposed
    choose = (qd.choose_qparams_symmetric.tensor if symmetric
              else qd.choose_qparams.tensor)
    s, z = choose(x, qmin, qmax, eps, dtype)
    q = qd.quantize_per_tensor.tensor(x, s, z, qmin, qmax, dtype)
    return qd.dequantize_per_tensor.tensor(q, s, z, qmin, qmax, dtype)


def _chooser_leak(x):
    """A chooser whose scale also feeds a portable op beside its quantize."""
    qd = torch.ops.quantized_decomposed
    s, z = qd.choose_qparams.tensor(x, -128, 127, qm.EPS_XNNPACK, torch.int8)
    q = qd.quantize_per_tensor.tensor(x, s, z, -128, 127, torch.int8)
    return qd.dequantize_per_tensor.tensor(q, s, z, -128, 127,
                                           torch.int8) * s.float()


def _token_chooser(x):
    """XNNPACK's per-token chooser (A4), quantized per token: no chain this
    backend absorbs."""
    qd = torch.ops.quantized_decomposed
    s, z = qd.choose_qparams_per_token_asymmetric(x, torch.int8)
    q = qd.quantize_per_token(x, s, z, -128, 127, torch.int8)
    return qd.dequantize_per_token(q, s, z, -128, 127, torch.int8,
                                   torch.float32)


REFUSALS = (
    ("act-symmetric-no-clipping-err",
     lambda x: _torchao(x, "SYMMETRIC_NO_CLIPPING_ERR")),
    ("act-dynamic-int16",
     lambda x: _pt2e(x, torch.int16, -32768, 32767)),
    ("act-dynamic-int16",
     lambda x: _torchao(x, dtype=torch.int16, qmin=-32768, qmax=32767)),
    ("act-unsigned-symmetric",
     lambda x: _pt2e(x, torch.uint8, 0, 255, symmetric=True)),
    ("act-unsigned-symmetric",
     lambda x: _torchao(x, "SYMMETRIC", torch.uint8, 0, 255)),
    ("act-symmetric-midpoint",                          # zero point -13
     lambda x: _torchao(x, "SYMMETRIC", qmin=-128, qmax=100)),
    ("act-granularity",
     lambda x: _torchao(x, block=[1, 4])),              # groups of 4 along K
    ("act-granularity",
     lambda x: _torchao(x, block=[4, 1])),              # per channel along K
    ("act-scale-dtype",
     lambda x: _torchao(x, scale_dtype=torch.bfloat16)),
    ("act-zero-point-dtype",
     lambda x: _torchao(x, zp_dtype=torch.float16)),
    ("act-chooser-users", _chooser_leak),
    ("act-chooser-unabsorbed", _token_chooser),
    # eps 0 lets an all-zero row choose scale 0 (the runtime
    # refuses it at load as act-eps); refused here, with the other rules.
    ("act-eps", lambda x: _pt2e(x, eps=0.0)),
    ("act-eps", lambda x: _torchao(x, eps=0.0)),
    ("act-eps", lambda x: _pt2e(x, eps=-0.0)),       # no signed-zero eps
    ("act-eps", lambda x: _pt2e(x, eps=1e-50)),      # 0 in float32
    # int() would truncate a non-integral bound.
    ("act-range-not-integral",
     lambda x: _torchao(x, qmin=-127.5, qmax=126.5)),
)


@pytest.mark.parametrize("rule, body", REFUSALS,
                         ids=[f"{r}-{i}" for i, (r, _) in enumerate(REFUSALS)])
def test_refused_by_name_at_export(rule, body):
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        DynamicChainRefused,
    )

    with pytest.raises(DynamicChainRefused) as why:
        _lower(Fc(body), torch.rand(4, 16) - 0.25)
    assert why.value.rule == rule, str(why.value)
    assert f"[{rule}]" in str(why.value)


# --- a chain's identical copies are the chain --------------------------------------------

class TwoHeads(nn.Module):
    """Two Linears over one activation: `body` returns the dequantised value
    each reads (a chain's copies, or the one value twice)."""

    def __init__(self, body):
        super().__init__()
        torch.manual_seed(41)
        self.fa, self.fb = nn.Linear(16, 8), nn.Linear(16, 4)
        self.body = body

    def forward(self, x):
        a, b = self.body(x)
        return self.fa(a), self.fb(b)


def _sibling_quantizes(x):
    """One chooser, a quantize -> dequantize copy per consumer."""
    qd = torch.ops.quantized_decomposed
    s, z = qd.choose_qparams.tensor(x, -128, 127, qm.EPS_XNNPACK, torch.int8)
    return [qd.dequantize_per_tensor.tensor(
        qd.quantize_per_tensor.tensor(x, s, z, -128, 127, torch.int8),
        s, z, -128, 127, torch.int8) for _ in range(2)]


def _sibling_dequantizes(x):
    """One quantize, a dequantize per consumer: DuplicateDQPass's copies of a
    symmetric chain (it keeps only choose_qparams.tensor's shared)."""
    qd = torch.ops.quantized_decomposed
    s, z = qd.choose_qparams_symmetric.tensor(x, -127, 127, qm.EPS_ARM,
                                              torch.int8)
    q = qd.quantize_per_tensor.tensor(x, s, z, -127, 127, torch.int8)
    return [qd.dequantize_per_tensor.tensor(q, s, z, -127, 127, torch.int8)
            for _ in range(2)]


def _shared_affine(x):
    d = _pt2e(x)
    return d, d


def _shared_symmetric(x):
    d = _pt2e(x, qmin=-127, qmax=127, symmetric=True, eps=qm.EPS_ARM)
    return d, d


def _different_copies(x):
    """One chooser, two quantizes that differ (another range): not copies."""
    qd = torch.ops.quantized_decomposed
    s, z = qd.choose_qparams.tensor(x, -128, 127, qm.EPS_XNNPACK, torch.int8)
    return [qd.dequantize_per_tensor.tensor(
        qd.quantize_per_tensor.tensor(x, s, z, lo, 127, torch.int8),
        s, z, lo, 127, torch.int8) for lo in (-128, -127)]


def _two_heads_on_host(runner, tmp_path, body, x):
    fixture = eaa.Fixture(f"r5_{body.__name__}", TwoHeads(body).eval(), (x,),
                          None, {})
    et, _ = eaa.lower(fixture)
    work = tmp_path / body.__name__
    work.mkdir()
    (work / "m.pte").write_bytes(et.buffer)
    x.numpy().tofile(work / "in.bin")
    outs = [work / f"out{i}.bin" for i in range(2)]
    cmd = [str(runner), "--pte", str(work / "m.pte"), "--input",
           str(work / "in.bin")]
    for out in outs:
        cmd += ["--output", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    return np.concatenate([np.fromfile(o, dtype=np.float32) for o in outs])


@pytest.mark.parametrize("copies, shared", [
    (_sibling_quantizes, _shared_affine),
    (_sibling_dequantizes, _shared_symmetric)])
def test_identical_copies_of_a_chain_are_absorbed(runner, tmp_path, copies,
                                                  shared):
    """A chain's identical copies are absorbed as the chain, and
    the program computes what the one shared chain's does, bit for bit."""
    x = torch.rand(4, 16) - 0.25
    got = _two_heads_on_host(runner, tmp_path, copies, x)
    want = _two_heads_on_host(runner, tmp_path, shared, x)
    assert output_bits(got) == output_bits(want)


def test_copies_that_differ_are_refused():
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        DynamicChainRefused,
    )

    with pytest.raises(DynamicChainRefused) as why:
        _lower(TwoHeads(_different_copies), torch.rand(4, 16) - 0.25)
    assert why.value.rule == "act-chooser-users", str(why.value)


def test_per_token_on_a_conv_is_refused():
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        DynamicChainRefused,
    )

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(4, 4, 3, padding=1)

        def forward(self, x):
            return self.c(_torchao(x, block=[1, 1, 1, 6]))

    with pytest.raises(DynamicChainRefused) as why:
        _lower(Conv(), torch.rand(1, 4, 6, 6))
    assert why.value.rule == "act-token-on-conv"


class TokenConv(nn.Module):
    """A convolution behind a per-token torchao chain, and the chain's value
    itself."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(31)
        self.c = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x):
        y = _torchao(x, block=[1, 1, 1, 6])
        return self.c(y), y


@pytest.mark.parametrize("policy", [None, "i8"])
def test_per_token_on_a_conv_runs_portable(runner, tmp_path, policy):
    """Through lower_esp32, whose strip asks the partitioner's gate, a
    per-token chain on a convolution is left on its value like any chain
    whose consumer cannot take it, not refused by name (act-token-on-conv,
    which stopped the lowering): the chain runs as the float32 portable ops
    torchao's arithmetic is, and the convolution reads its value. Under the
    default policy the program is recipe-quantised and the conv's weight
    kept float, so the gate claims the conv in the fp32 loop; under an
    explicit i8 policy the conv would quantise its activation again, and the
    gate refuses it by the chain's reason, so it runs portable. Either way
    the program runs, the chain's value is torchao's bit for bit and the
    conv is within float rounding of eager's."""
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    x = torch.rand(1, 4, 6, 6) - 0.25
    model = TokenConv().eval()
    with torch.no_grad():
        want_conv, want_chain = (t.numpy() for t in model(x))
    partitioner = Esp32Partitioner(format_policy=policy)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        edge = lower_esp32(torch.export.export(model, (x,)), [partitioner])
        apply_passes(edge, quiet=True)
    assert re.search(r"leaving \S+ in place: \S+ stays portable \(per-token "
                     r"activation scales on a convolution", log.getvalue()), \
        log.getvalue()
    ops = [str(n.target) for n in edge.exported_program().graph.nodes
           if n.op == "call_function"]
    assert not any("torchao" in t for t in ops), ops
    if policy is None:
        assert "fp32 loop" in log.getvalue(), log.getvalue()
        assert not any("convolution" in t for t in ops), ops
    else:
        assert partitioner.support.kinds.get(
            "per-token activation scales on a convolution"), dict(
                partitioner.support.kinds)
        assert any("convolution" in t for t in ops), ops
    pte = tmp_path / "model.pte"
    pte.write_bytes(edge.to_executorch(ExecutorchBackendConfig(
        delegate_alignment=16)).buffer)
    x.numpy().tofile(tmp_path / "in.bin")
    outs = [tmp_path / f"out{i}.bin" for i in range(2)]
    done = subprocess.run(
        [str(runner), "--pte", str(pte), "--input", str(tmp_path / "in.bin"),
         "--output", str(outs[0]), "--output", str(outs[1])],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    got_conv, got_chain = (np.fromfile(o, dtype=np.float32) for o in outs)
    assert output_bits(got_chain) == output_bits(want_chain.reshape(-1))
    assert np.abs(got_conv - want_conv.reshape(-1)).max() <= \
        1e-5 * np.abs(want_conv).max()


def test_unclaimed_consumer_is_refused():
    """A Linear that absorbed a chain but the partitioner cannot take: here a
    weight computed at runtime. Run portable it would see the raw tensor."""

    class Dyn(nn.Module):
        def forward(self, x, w):
            return torch.nn.functional.linear(_pt2e(x), w)

    from executorch.exir import to_edge

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    edge = to_edge(torch.export.export(
        Dyn().eval(), (torch.rand(2, 16), torch.rand(8, 16))))
    strip_dynamic_act_qdq(edge)
    with pytest.raises(RuntimeError, match=r"\[act-consumer-unclaimed\]"):
        with contextlib.redirect_stdout(io.StringIO()):
            edge.to_backend(Esp32Partitioner())


def test_records_need_an_int_format():
    """format_policy fp32 has no activation quantiser: refused by name."""
    from executorch.backends.esp32s3.serialization.asym_records import SchemeRefused

    with pytest.raises(SchemeRefused, match=r"\[act-format\]"):
        _lower(Fc(_torchao), torch.rand(2, 16), format_policy="fp32")


def test_exactness_bounds_by_name():
    """65,792 for affine int8 activations against symmetric int8 weights,
    33,024 against zero-point ones; one group of 32 past either is refused
    by name, as the runtime's dot-bound rule refuses it at load."""
    from executorch.backends.esp32s3.serialization.asym_records import (
        SchemeRefused,
        act_quant,
        check_act_bounds,
    )

    a3 = act_quant(qm.A3_TORCHAO)
    for zero_point, bound in ((False, 65_792), (True, 33_024)):
        check_act_bounds(a3, 8, bound, zero_point=zero_point, offsets=False)
        with pytest.raises(SchemeRefused, match=r"\[act-dot-bound\]"):
            check_act_bounds(a3, 8, bound + 32, zero_point=zero_point,
                             offsets=False)
        assert qm.dot_length_bound(255, 128 if not zero_point else 255) == bound
    # A symmetric record on the model's grid reaches -128: 131,040.
    a7 = act_quant(qm.A7_TORCHAO)
    check_act_bounds(a7, 8, 131_040, zero_point=False, offsets=False)
    with pytest.raises(SchemeRefused, match=r"\[act-dot-bound\]"):
        check_act_bounds(a7, 8, 131_072, zero_point=False, offsets=False)
    # Float offsets: X'_g converts exactly up to 2^24.
    with pytest.raises(SchemeRefused, match=r"\[act-offset-bound\]"):
        check_act_bounds(a3, 4, 65_824, zero_point=True, offsets=True)


def test_the_gate_refuses_a_dot_past_the_bound():
    """At export the bound is the gate's: a K one group past 65,792 behind an
    affine chain stops the export, naming both rules."""
    k = 65_792 + 32
    with pytest.raises(RuntimeError) as why:
        _lower(Fc(_torchao, k=k, n=1), torch.rand(1, k))
    text = str(why.value)
    assert "[act-consumer-unclaimed]" in text and "[act-dot-bound]" in text


@pytest.mark.parametrize("geometry", [
    dict(kernel_size=2, stride=2),
    dict(kernel_size=3, stride=2, padding=1, output_padding=1),
    # Output padding 2 at stride 2, which only the dilation of 3 allows.
    dict(kernel_size=3, stride=(2, 1), padding=(1, 2), dilation=(3, 2),
         output_padding=(2, 1)),
], ids=["2x2 s2", "3x3 s2 p1 output_padding 1",
        "dil (3,2) output_padding (2,1)"])
def test_recipe_transposed_conv_is_delegated(runner, tmp_path, geometry):
    """XNNPACK's dynamic recipe quantises a ConvTranspose2d with no relu
    after it: its weight per output channel, on axis 1, and its activation
    dynamically. The backend claims it with its own codes, turned into the
    dense equivalent's rows (const_weights._resolve_pt2e_dequant): one int8
    pass-through conv with the chain's record, nothing portable (the export
    must not stop at [act-consumer-unclaimed]), and the host runner's output
    is the recipe's eager model's to fp32 rounding -- with output padding and
    dilation too, through esp32_asym's gather."""
    from executorch.backends.esp32s3 import _passes
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.quant_census import recipes
    from executorch.backends.esp32s3.test.quant_census.lower import (
        strip_before_partitioning,
    )

    torch.manual_seed(3)
    net = nn.ConvTranspose2d(8, 4, **geometry).eval()
    x = torch.randn(1, 8, 5, 5)
    q = recipes.quantize("xnnpack-dynamic", net, (x,), [(x,)])
    ep = torch.export.export(q.program, q.inputs, strict=False)
    log = io.StringIO()
    with contextlib.redirect_stdout(log), strip_before_partitioning({}):
        lowered = _passes.lower_esp32(ep, [Esp32Partitioner()])
    with contextlib.redirect_stdout(log):
        _passes.apply_passes(lowered, quiet=True)
    blobs = [l for l in log.getvalue().splitlines() if " B blob" in l]
    k = geometry["kernel_size"]
    assert len(blobs) == 1 and "addmmPT_i8[" in blobs[0] \
        and f" conv{k}x{k}" in blobs[0], blobs
    left = [str(n.target) for n in lowered.exported_program().graph.nodes
            if n.op == "call_function" and "delegate" not in str(n.target)
            and "getitem" not in str(n.target)]
    assert left == [], left
    (tmp_path / "m.pte").write_bytes(lowered.to_executorch().buffer)
    x.numpy().tofile(tmp_path / "in.bin")
    done = subprocess.run([str(runner), "--pte", str(tmp_path / "m.pte"),
                           "--input", str(tmp_path / "in.bin"), "--output",
                           str(tmp_path / "out.bin")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    got = np.fromfile(tmp_path / "out.bin", dtype=np.float32)
    with torch.no_grad():
        want = q.reference(x).numpy().reshape(-1)
    assert np.abs(got - want).max() <= 1e-6
