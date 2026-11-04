# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight tiers: the external weights the backend copies from flash into PSRAM.

The backend's weight-tier options (runtime/esp32_backend.cpp, described with
kOptWeightTierBudget) copy every stored format's external weights -- int4,
int8, int16, fp32, fp16 and bf16, matmuls and convolutions -- into PSRAM within
a budget, in load order; tables (embeddings and selected-row heads) only when
asked; and an int4 head FLAG_STAGE_I8 expands always, counted first. Ranked
placement (weight_tier_rank) copies nothing new at load and leaves the
candidates of every loaded method to weight_tier_apply, or to the first
execute when nobody applies: score first, then format class, then size, the
larger first, first fit. The host runner sets them (--weight-tier-budget
N|auto, --weight-tier-reserve, --weight-tier-tables, --weight-tier-rank,
--weight-tier-apply, --no-weight-cache, --psram-bytes; --keep-methods holds
every method of a script loaded at once) and prints the cache after the run:
an "esp32 tiers:" summary, an "esp32 tiers rank:" line once ranking is used, a
line per format class, and one "esp32 tier:" line per weight the cache knows.

A copy holds the stored bytes, so every output below is bit-identical at budget
0, partial and full, in load order and ranked, applied or not, and while a
placement runs on another thread (--placement-stress, also under
ThreadSanitizer where the host build has esp32_host_runner_tsan). The asymmetric
z, m and u tables (zero points, offsets, weight sums) are not copied:
esp32_asym reads them in place, and test/asym covers their numerics.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import os
import re
import struct
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
)
from executorch.exir.backend.utils import get_delegated_payload

from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_BYTES,
    BLOB_HEADER_FORMAT,
    EXT_COUNT_FORMAT,
    EXT_RECORD_HEADER_BYTES,
    EXT_RECORD_HEADER_FORMAT,
    EXT_SELECTED_ROWS,
    FLAG_EXTERNAL_WEIGHTS,
    FLAG_FP16_SCALES,
    FLAG_STAGE_I8,
    OP_ADDMM,
    OP_ADDMM_BF16,
    OP_ADDMM_F16,
    OP_ADDMM_I16,
    OP_ADDMM_I4,
    OP_ADDMM_I8,
    OP_BMM_DYN_I8,
    OP_BMM_KCACHE_AFFINE_I8,
    OP_BMM_KCACHE_I8,
    OP_DESC_BYTES,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
    OP_EMBED_BF16,
    OP_EMBED_F16,
    OP_EMBED_I16,
    OP_EMBED_I4,
    OP_EMBED_I8,
)
from executorch.backends.esp32s3.test.host import host_runner

TAG = "tier_weights"
FULL = 64 << 20  # more PSRAM than any program here wants

HALF = (OP_ADDMM_F16, OP_ADDMM_BF16, OP_EMBED_F16, OP_EMBED_BF16)
EMBED = (OP_EMBED_I8, OP_EMBED_I4, OP_EMBED_I16, OP_EMBED_F16, OP_EMBED_BF16)
BMM = (OP_BMM_DYN_I8, OP_BMM_KCACHE_I8, OP_BMM_KCACHE_AFFINE_I8)
CLASS = {OP_ADDMM_I4: "int4", OP_EMBED_I4: "int4", OP_ADDMM_I8: "int8",
         OP_EMBED_I8: "int8", OP_ADDMM: "fp32", OP_ADDMM_BF16: "bf16",
         OP_EMBED_BF16: "bf16", OP_ADDMM_I16: "int16", OP_EMBED_I16: "int16",
         OP_ADDMM_F16: "fp16", OP_EMBED_F16: "fp16"}


@pytest.fixture(scope="module")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


# ---------------------------------------------------------------- the programs


def lower(model, inputs, max_params=60_000, **options):
    """The model through the backend, every weight external under TAG. A hard
    gate first: these fixtures must stay tiny."""
    model.eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) <= max_params
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        edge = lower_esp32(torch.export.export(model, inputs),
                           [Esp32Partitioner(external_weights=TAG, **options)])
        apply_passes(edge, quiet=True)
    return edge


def c_string(blob, offset):
    return bytes(blob[offset:blob.index(b"\0", offset)])


def descriptors(edge):
    """Every descriptor of the manager's delegates, with its op's place in its
    blob, its weight keys and whether a selected-row record is attached."""
    out = []
    payloads = get_delegated_payload(edge.exported_program().graph_module)
    for _, _, blob in payloads.values():
        _, _, count, ext_off = struct.unpack_from(BLOB_HEADER_FORMAT, blob)
        selected = set()
        if ext_off:
            (records,) = struct.unpack_from(EXT_COUNT_FORMAT, blob, ext_off)
            at = ext_off + struct.calcsize(EXT_COUNT_FORMAT)
            for _ in range(records):
                op, kind, length = struct.unpack_from(EXT_RECORD_HEADER_FORMAT, blob, at)
                if kind & 0x7FFF == EXT_SELECTED_ROWS:
                    selected.add(op)
                at += EXT_RECORD_HEADER_BYTES + (length + 3) // 4 * 4
        for i in range(count):
            d = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
                OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES + i * OP_DESC_BYTES)))
            d["selected"] = i in selected
            if d["flags"] & FLAG_EXTERNAL_WEIGHTS and d["opcode"] not in BMM:
                d["wkey"] = c_string(blob, d["wt_off"])
                d["skey"] = None if is_float(d) else c_string(blob, d["scale_off"])
            out.append(d)
    return out


def emit(edge, directory):
    """(pte, ptd) paths: the program and its named data, in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    pte = directory / "model.pte"
    pte.write_bytes(et.buffer)
    et.write_tensor_data_to_file(str(directory))
    ptd = directory / f"{TAG}.ptd"
    assert ptd.is_file()
    return pte, ptd


def multi_method(methods):
    """One manager from one-method managers, as the parity firmware's
    multi-method programs are made: each lowered alone, then combined."""
    return EdgeProgramManager(
        {name: edge.exported_program() for name, edge in methods.items()},
        compile_config=EdgeCompileConfig(_check_ir_validity=False))


# ------------------------------------------------- the runtime's arithmetic


def is_float(d):
    return d["opcode"] == OP_ADDMM or d["opcode"] in HALF


def weight_bytes(d):
    op = d["opcode"]
    if op in HALF:
        return d["k"] * d["n"] * 2
    if op == OP_ADDMM:
        return d["k"] * d["n"] * 4
    if op in (OP_ADDMM_I4, OP_EMBED_I4):
        return d["n"] * (d["k_pad"] // 2)
    if op in (OP_ADDMM_I16, OP_EMBED_I16):
        return d["n"] * d["k_pad"] * 2
    return d["n"] * d["k_pad"]


def scale_bytes(d):
    if is_float(d):
        return 0
    span = d["k"] if d["opcode"] in EMBED else d["k_pad"]
    group = d["scale_group"] or span
    return d["n"] * -(-span // group) * (2 if d["flags"] & FLAG_FP16_SCALES else 4)


def need(d):
    """A copy's bytes in the budget's unit: weights (an expanded int4 head at
    one byte a code) rounded up to 4, scales, and 15 B of alignment slack."""
    wbytes = d["n"] * d["k_pad"] if d["flags"] & FLAG_STAGE_I8 else weight_bytes(d)
    return (wbytes + 3) // 4 * 4 + scale_bytes(d) + 15


def tier_class(d):
    return "int8" if d["flags"] & FLAG_STAGE_I8 else CLASS[d["opcode"]]


def fnv1a(h, data):
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def names(d):
    """The digest an entry is named by: its weight key, then its scale key,
    each with its terminator (a float weight has no scale key)."""
    h = fnv1a(0xCBF29CE484222325, d["wkey"] + b"\0")
    return h if d["skey"] is None else fnv1a(h, d["skey"] + b"\0")


def candidates(descs, tables=False):
    """The descriptors whose weights a budget may copy, as the runtime picks
    them (tier_candidate): stored external weights, tables only if asked."""
    return [d for d in descs
            if d["flags"] & FLAG_EXTERNAL_WEIGHTS and d["opcode"] not in BMM
            and not d["flags"] & FLAG_STAGE_I8
            and (tables or not (d["opcode"] in EMBED or d["selected"]))]


# ------------------------------------------------------------- the host runner


def write_inputs(directory, stem, inputs):
    paths = []
    for i, x in enumerate(inputs):
        path = directory / f"{stem}_in{i}.bin"
        np.ascontiguousarray(x.detach().numpy()).tofile(path)
        paths.append(path)
    return paths


def run(runner, pte, ptd, inputs, *flags, n_out=1, ok=True):
    """One method's outputs (as uint32 bit patterns) and the runner's stdout."""
    command = [str(runner), "--pte", str(pte), "--ptd", str(ptd), *flags]
    for path in write_inputs(pte.parent, "run", inputs):
        command += ["--input", str(path)]
    outs = [pte.parent / f"run_out{i}.bin" for i in range(n_out)]
    for out in outs:
        out.unlink(missing_ok=True)
        command += ["--output", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if not ok:
        return done
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return [np.fromfile(o, dtype=np.uint32) for o in outs], done.stdout


def run_script(runner, pte, ptd, steps, *flags):
    """Steps (method, inputs, number of outputs) in one process, or directive
    lines ("set KEY VALUE", "unload"): each step's outputs as uint32 bit
    patterns (none for a directive), and the runner's stdout."""
    lines, outputs = [], []
    for index, step in enumerate(steps):
        if isinstance(step, str):
            lines.append(step)
            outputs.append([])
            continue
        method, inputs, n_out = step
        ins = write_inputs(pte.parent, f"step{index}", inputs)
        outs = [pte.parent / f"step{index}_out{i}.bin" for i in range(n_out)]
        for out in outs:
            out.unlink(missing_ok=True)
        lines.append(" ".join([method, *map(str, ins), "->", *map(str, outs)]))
        outputs.append(outs)
    script = pte.parent / "steps.txt"
    script.write_text("\n".join(lines) + "\n")
    command = [str(runner), "--pte", str(pte), "--ptd", str(ptd),
               "--script", str(script), *flags]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return [[np.fromfile(o, dtype=np.uint32) for o in outs] for outs in outputs], done.stdout


SUMMARY = re.compile(
    r"^esp32 tiers: budget (?P<budget>.+?), resident (?P<resident>\d+) B, "
    r"wanted (?P<wanted>\d+) B, entries (?P<entries>\d+); copies (?P<copies>\d+), "
    r"reuses (?P<reuses>\d+), evictions (?P<evictions>\d+), left on flash "
    r"(?P<skipped>\d+), expanded (?P<expanded>\d+)$", re.M)
PER_CLASS = re.compile(
    r"^esp32 tiers (?P<cls>\w+): staged (?P<staged>\d+) B, on flash (?P<flash>\d+) B$",
    re.M)
ENTRY = re.compile(
    r"^esp32 tier: (?P<cls>\w+) (?P<kind>\S+) (?P<place>psram|flash) (?P<bytes>\d+) B "
    r"score (?P<score>\d+) refs (?P<refs>\d+) names (?P<names>[0-9a-f]{16})$", re.M)
RANK = re.compile(
    r"^esp32 tiers rank: (?P<rank>on|off), placements (?P<placements>\d+) "
    r"\((?P<late>\d+) at a first execute\), delegates waiting (?P<waiting>\d+), "
    r"ops reading copies (?P<reading>\d+) of (?P<copied>\d+)$", re.M)
DIRECTIVE = re.compile(
    r"^(?P<what>unload|set \S+ \S+): resident (?P<resident>\d+) B, "
    r"entries (?P<entries>\d+)$", re.M)


def report(log):
    """The runner's cache report: (summary, {class: (staged, on flash)},
    {names digest: entry}); (None, {}, {}) when the cache knew no weight."""
    summary = SUMMARY.search(log)
    if summary is None:
        assert ENTRY.search(log) is None, log[-2000:]
        return None, {}, {}
    s = {k: (v if k == "budget" else int(v)) for k, v in summary.groupdict().items()}
    classes = {m["cls"]: (int(m["staged"]), int(m["flash"]))
               for m in PER_CLASS.finditer(log)}
    entries = {}
    for m in ENTRY.finditer(log):
        e = m.groupdict()
        for key in ("bytes", "score", "refs"):
            e[key] = int(e[key])
        e["names"] = int(e["names"], 16)
        entries[e["names"]] = e
    assert len(entries) == s["entries"], log[-2000:]
    return s, classes, entries


def rank_report(log):
    """The runner's ranked-placement line, {rank, placements, late, waiting,
    reading, copied} (late: placements made by an execute, nobody having
    applied; copied: the loaded delegates' ops whose weight has a copy, and
    reading: those that read it, which every delegate executed since the
    placement does); None when no ranking was used."""
    m = RANK.search(log)
    if m is None:
        return None
    return {k: (v if k == "rank" else int(v)) for k, v in m.groupdict().items()}


CLASS_ORDER = ("int4", "int8", "fp32", "bf16", "int16", "fp16")


def ranked_first_fit(entries, budget, resident=0):
    """The weights one ranked placement copies (their names digests), from
    the report's candidates in table order: by score, then format class,
    then size, the larger first, then table order -- load order -- each copied
    when it fits what the budget still has. `resident`: bytes copied before
    it, an expanded head's say."""
    order = sorted(enumerate(entries.values()), key=lambda t: (
        -t[1]["score"], CLASS_ORDER.index(t[1]["cls"]), -t[1]["bytes"], t[0]))
    staged = set()
    for _, e in order:
        if resident + e["bytes"] <= budget:
            resident += e["bytes"]
            staged.add(e["names"])
    return staged


def same_bits(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------- every stored format


class Ladder(nn.Module):
    """Five Linear layers, one per stored format the overrides give them."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.fc = nn.ModuleList(nn.Linear(64, 64) for _ in range(5))

    def forward(self, x):
        for fc in self.fc:
            x = torch.relu(fc(x))
        return x


class Convs(nn.Module):
    """A dense 3x3 convolution in int4, one in int8, a depthwise one in int8."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(8)
        self.c4 = nn.Conv2d(8, 16, 3, padding=1)
        self.c8 = nn.Conv2d(16, 16, 3, padding=1)
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)

    def forward(self, x):
        return self.dw(torch.relu(self.c8(torch.relu(self.c4(x)))))


class Float32(nn.Module):
    """A Linear and a convolution, both on the fp32 loops."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(9)
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.fc = nn.Linear(32, 16)

    def forward(self, x, v):
        return self.conv(x), self.fc(v)


FORMATS = {
    "ladder": (Ladder, lambda: (torch.randn(3, 64),),
               dict(format_overrides={"fc.0.weight": "i4", "fc.1.weight": "i8",
                                      "fc.2.weight": "i16", "fc.3.weight": "fp16",
                                      "fc.4.weight": "bf16"}),
               {"int4", "int8", "int16", "fp16", "bf16"}),
    "convs": (Convs, lambda: (torch.randn(1, 8, 8, 8),),
              dict(format_overrides={"c4.weight": "i4", "c8.weight": "i8",
                                     "dw.weight": "i8"}),
              {"int4", "int8"}),
    "fp32": (Float32, lambda: (torch.randn(1, 4, 6, 6), torch.randn(2, 32)),
             dict(format_policy="fp32"), {"fp32"}),
}


@pytest.fixture(scope="module")
def formats(tmp_path_factory):
    """name -> (pte, ptd, inputs, candidate descriptors, output count)."""
    out = {}
    for name, (cls, make_inputs, options, _) in FORMATS.items():
        torch.manual_seed(11)
        inputs = make_inputs()
        model = cls()
        edge = lower(model, inputs, **options)
        descs = candidates(descriptors(edge))
        with torch.no_grad():
            produced = model(*inputs)
        n_out = len(produced) if isinstance(produced, tuple) else 1
        pte, ptd = emit(edge, tmp_path_factory.mktemp(name))
        out[name] = (pte, ptd, inputs, descs, n_out)
    return out


@pytest.mark.parametrize("name", list(FORMATS))
def test_every_format_is_copied_bit_for_bit(runner, formats, name):
    """Every stored format's matmul and convolution weights take a copy at a
    full budget, named and sized as the runtime's arithmetic says; at a
    partial budget only what fits does, first fit; with none, nothing is
    resolved at all. The outputs never move, PIE or scalar, one core or two."""
    pte, ptd, inputs, descs, n_out = formats[name]
    want = FORMATS[name][3]
    assert {tier_class(d) for d in descs} == want, descs
    expected = {names(d): d for d in descs}
    total = sum(need(d) for d in expected.values())

    flash, log = run(runner, pte, ptd, inputs, n_out=n_out)
    assert report(log) == (None, {}, {})

    full, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", str(FULL),
                    n_out=n_out)
    same_bits(full, flash)
    s, classes, entries = report(log)
    assert set(entries) == set(expected), (entries, expected)
    for digest, d in expected.items():
        e = entries[digest]
        assert (e["cls"], e["kind"], e["place"], e["bytes"], e["refs"]) == (
            tier_class(d), "stored", "psram", need(d), 1), (e, d)
    assert s["resident"] == s["wanted"] == total
    assert (s["copies"], s["skipped"], s["evictions"]) == (len(expected), 0, 0)
    assert classes == {c: (sum(need(d) for d in expected.values()
                               if tier_class(d) == c), 0) for c in want}

    partial, log = run(runner, pte, ptd, inputs, "--weight-tier-budget",
                       str(total // 2), n_out=n_out)
    same_bits(partial, flash)
    s, _, entries = report(log)
    staged = [e for e in entries.values() if e["place"] == "psram"]
    left = [e for e in entries.values() if e["place"] == "flash"]
    assert staged and left and s["wanted"] == total
    assert s["resident"] == sum(e["bytes"] for e in staged) <= total // 2
    # First fit: whatever stayed on flash did not fit in what was left.
    assert all(e["bytes"] > total // 2 - s["resident"] for e in left)

    for flags in (("--force-scalar",), ("--min-latency",)):
        modes, _ = run(runner, pte, ptd, inputs, "--weight-tier-budget",
                       str(FULL), *flags, n_out=n_out)
        same_bits(modes, flash)


@pytest.mark.parametrize("name", list(FORMATS))
def test_ranked_placement_in_every_format(runner, formats, name):
    """Ranked, a load copies nothing new and one placement copies its
    candidates by score, class and size, first fit -- made by the apply call,
    or by the first execute when nobody applies. At budgets 0, partial and
    full the outputs never move, PIE or scalar, one core or two."""
    pte, ptd, inputs, descs, n_out = formats[name]
    total = sum(need(d) for d in {names(d): d for d in descs}.values())
    flash, _ = run(runner, pte, ptd, inputs, n_out=n_out)

    got, log = run(runner, pte, ptd, inputs, "--weight-tier-rank",
                   "--weight-tier-apply", n_out=n_out)
    same_bits(got, flash)
    assert report(log) == (None, {}, {}) and rank_report(log) is None

    for budget in (total // 2, FULL):
        for apply in (("--weight-tier-apply",), ()):
            got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget",
                           str(budget), "--weight-tier-rank", *apply, n_out=n_out)
            same_bits(got, flash)
            s, _, entries = report(log)
            staged = {k for k, e in entries.items() if e["place"] == "psram"}
            copied = sum(names(d) in staged for d in descs)
            assert copied == (len(descs) if budget == FULL else copied) > 0
            assert rank_report(log) == {"rank": "on", "placements": 1,
                                        "late": 0 if apply else 1, "waiting": 0,
                                        "reading": copied, "copied": copied}
            assert staged == ranked_first_fit(entries, budget), entries
            assert s["resident"] == sum(entries[k]["bytes"] for k in staged)
            assert (s["copies"], s["reuses"], s["skipped"], s["evictions"]) == (
                len(staged), 0, len(entries) - len(staged), 0)
            if budget == FULL:
                assert len(staged) == len(entries)

    for flags in (("--force-scalar",), ("--min-latency",)):
        modes, _ = run(runner, pte, ptd, inputs, "--weight-tier-budget",
                       str(total // 2), "--weight-tier-rank", *flags, n_out=n_out)
        same_bits(modes, flash)


def test_no_psram_leaves_every_weight_on_flash(runner, formats):
    """A board without PSRAM (QEMU's, --psram-bytes 0): every copy fails, every
    weight is read from flash, and the outputs do not move."""
    pte, ptd, inputs, descs, n_out = formats["ladder"]
    flash, _ = run(runner, pte, ptd, inputs, n_out=n_out)
    got, log = run(runner, pte, ptd, inputs, "--psram-bytes", "0",
                   "--weight-tier-budget", str(FULL), n_out=n_out)
    same_bits(got, flash)
    s, classes, entries = report(log)
    assert s["resident"] == s["copies"] == 0 and s["skipped"] == len(descs)
    assert {e["place"] for e in entries.values()} == {"flash"}
    assert sum(f for _, f in classes.values()) == s["wanted"]
    # Ranked, the placement's every allocation fails the same way.
    got, log = run(runner, pte, ptd, inputs, "--psram-bytes", "0",
                   "--weight-tier-budget", str(FULL), "--weight-tier-rank",
                   "--weight-tier-apply", n_out=n_out)
    same_bits(got, flash)
    s, _, entries = report(log)
    assert s["resident"] == s["copies"] == 0 and s["skipped"] == len(entries)
    assert {e["place"] for e in entries.values()} == {"flash"}


def test_auto_keeps_the_reserve_and_leaves_fp16_on_flash(runner, formats):
    """Auto (-1) copies every candidate while weight_tier_reserve bytes of
    PSRAM stay free, and leaves fp16 out: its loop is bound by its arithmetic
    even from flash."""
    pte, ptd, inputs, descs, n_out = formats["ladder"]
    flash, _ = run(runner, pte, ptd, inputs, n_out=n_out)
    by_class = {tier_class(d): d for d in descs}
    rest = [d for c, d in by_class.items() if c != "fp16"]

    got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", "auto",
                   n_out=n_out)
    same_bits(got, flash)
    s, _, entries = report(log)
    assert s["budget"] == "auto keeping 0 B free"
    assert set(entries) == {names(d) for d in rest}
    assert {e["place"] for e in entries.values()} == {"psram"}
    assert s["resident"] == sum(need(d) for d in rest)

    # PSRAM for about half of them past the reserve (the cache's own table
    # takes some too): what is copied never eats into the reserve.
    psram, reserve = 1 << 20, (1 << 20) - sum(need(d) for d in rest) // 2
    got, log = run(runner, pte, ptd, inputs, "--psram-bytes", str(psram),
                   "--weight-tier-budget", "auto", "--weight-tier-reserve",
                   str(reserve), n_out=n_out)
    same_bits(got, flash)
    s, _, entries = report(log)
    places = [e["place"] for e in entries.values()]
    assert "psram" in places and "flash" in places, entries
    assert s["resident"] <= psram - reserve

    # Ranked, auto takes the same candidates, and keeps the reserve too.
    got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", "auto",
                   "--weight-tier-rank", "--weight-tier-apply", n_out=n_out)
    same_bits(got, flash)
    s, _, entries = report(log)
    assert set(entries) == {names(d) for d in rest}
    assert {e["place"] for e in entries.values()} == {"psram"}
    got, log = run(runner, pte, ptd, inputs, "--psram-bytes", str(psram),
                   "--weight-tier-budget", "auto", "--weight-tier-reserve",
                   str(reserve), "--weight-tier-rank", "--weight-tier-apply",
                   n_out=n_out)
    same_bits(got, flash)
    s, _, entries = report(log)
    places = [e["place"] for e in entries.values()]
    assert "psram" in places and "flash" in places, entries
    assert s["resident"] <= psram - reserve


# ------------------------------------------------------------------- tables


class Tables(nn.Module):
    """An embedding table into a Linear, and a selected-row head over a
    257-row table (test_selected_head.py's). The head reads a graph input of
    its own: a selected-row head is not chained from a claimed producer."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(12)
        self.emb = nn.Embedding(48, 64)
        self.fc = nn.Linear(64, 64)
        self.head = nn.Parameter(torch.randn(257, 64))

    def forward(self, ids, value, rows):
        x = torch.relu(self.fc(self.emb(ids)))
        selected = torch.index_select(self.head, 0, rows)
        logits = F.linear(value, selected) * 0.196116
        return x, torch.tanh(logits / 20.0) * 20.0


def test_tables_stay_on_flash_unless_asked(runner, tmp_path):
    """Tables read a few of their rows per inference, so by default a budget
    copies the Linear only; weight_tier_tables copies the embedding table and
    the selected-row head too, scored by rows read over rows stored."""
    torch.manual_seed(13)
    inputs = (torch.tensor([3, 7, 11, 40]), torch.randn(2, 64),
              torch.tensor([1, 5, 8, 13, 21, 144, 233]))
    edge = lower(Tables(), inputs, format_policy="i8")
    descs = descriptors(edge)
    tables = [d for d in candidates(descs, tables=True) if d not in candidates(descs)]
    assert sorted((d["opcode"], d["selected"]) for d in tables) == [
        (OP_ADDMM_I8, True), (OP_EMBED_I8, False)], descs
    linear = candidates(descs)
    assert [d["opcode"] for d in linear] == [OP_ADDMM_I8]
    pte, ptd = emit(edge, tmp_path)

    flash, _ = run(runner, pte, ptd, inputs, n_out=2)
    got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", str(FULL),
                   n_out=2)
    same_bits(got, flash)
    _, _, entries = report(log)
    assert set(entries) == {names(linear[0])}

    got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", str(FULL),
                   "--weight-tier-tables", n_out=2)
    same_bits(got, flash)
    _, _, entries = report(log)
    assert set(entries) == {names(d) for d in linear + tables}
    assert {e["place"] for e in entries.values()} == {"psram"}
    for d in tables:
        # Rows read over rows stored, in 1/65536ths: 4 of the table's 48, and
        # 2 x 7 of the head's 257.
        read = d["m"] * (7 if d["selected"] else 1)
        assert entries[names(d)]["score"] == (read << 16) // d["n"], d


# ------------------------------------------------------- the expanded int4 head


def tied_head_program(tmp_path):
    """asym_head_staged's tied int4 head as two methods of different lengths
    sharing its weights: (pte, ptd, the fixture model, its descriptors)."""
    from executorch.backends.esp32s3.test.fixtures import asym

    model = asym.TiedHead().eval()
    options = dict(split_tied_head=True, stage_tied_head=True)
    methods = {name: lower(model, (asym.long_input(n, 96),), **options)
               for name, n in (("three", 3), ("five", 5))}
    descs = descriptors(methods["three"])
    pte, ptd = emit(multi_method(methods), tmp_path)
    return pte, ptd, model, descs


def test_expanded_head_counts_first_and_is_shared(runner, tmp_path):
    """FLAG_STAGE_I8's int8 expansion goes through the cache: made whatever
    the budget, named with its zero-point table, counted in the resident bytes
    before its init's candidates, and shared by the methods holding it at
    once -- which read it as already folded, bit for bit the mirror's. Without
    a budget it goes with its last method, as before the cache took it, so an
    application that sets no tier option gets that PSRAM back; under a budget
    the next method reuses it."""
    from executorch.backends.esp32s3.test.fixtures import asym

    pte, ptd, model, descs = tied_head_program(tmp_path)
    head = [d for d in descs if d["flags"] & FLAG_STAGE_I8]
    table = [d for d in candidates(descs, tables=True)]
    assert len(head) == 1 and [d["opcode"] for d in table] == [OP_EMBED_I4], descs
    head, table = head[0], table[0]
    three, five = asym.long_input(3, 96), asym.long_input(5, 96)
    want = [np.asarray(model.reference(x), np.float32).reshape(-1).view(np.uint32)
            for x in (three, five)]
    steps = [("three", (three,), 1), ("five", (five,), 1), ("three", (three,), 1)]

    # Budget 0: the expansion is made anyway, and freed with each method:
    # three loads, three copies, nothing kept to reuse.
    outs, log = run_script(runner, pte, ptd, steps)
    for (out,), expect in zip(outs, (want[0], want[1], want[0])):
        np.testing.assert_array_equal(out, expect)
    s, classes, entries = report(log)
    assert (s["copies"], s["reuses"], s["expanded"]) == (3, 0, 3)
    (entry,) = entries.values()
    assert (entry["cls"], entry["kind"], entry["place"], entry["bytes"]) == (
        "int8", "i4->i8", "psram", need(head))
    assert s["resident"] == need(head) and classes == {"int8": (need(head), 0)}
    assert entry["names"] != names(head)  # the zero-point table's key is in it

    # Budget 0 with both methods held at once: one copy, shared.
    outs, log = run_script(runner, pte, ptd, steps, "--keep-methods")
    for (out,), expect in zip(outs, (want[0], want[1], want[0])):
        np.testing.assert_array_equal(out, expect)
    s, _, entries = report(log)
    assert (s["copies"], s["reuses"], s["expanded"]) == (1, 1, 1)
    assert [e["refs"] for e in entries.values()] == [2]

    # Under a budget the next method reuses it: one copy, reused twice.
    outs, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                           str(need(head)))
    for (out,), expect in zip(outs, (want[0], want[1], want[0])):
        np.testing.assert_array_equal(out, expect)
    s, _, _ = report(log)
    assert (s["copies"], s["reuses"], s["expanded"]) == (1, 2, 1)

    # Counted first: one byte short of both, the table stays on flash.
    for budget, table_place in ((need(head) + need(table) - 1, "flash"),
                                (need(head) + need(table), "psram")):
        outs, log = run_script(runner, pte, ptd, steps[:2], "--weight-tier-budget",
                               str(budget), "--weight-tier-tables")
        for (out,), expect in zip(outs, want):
            np.testing.assert_array_equal(out, expect)
        _, _, entries = report(log)
        assert entries[names(table)]["place"] == table_place

    # No PSRAM (QEMU's case): a head that runs as int8 cannot load.
    done = run(runner, pte, ptd, (three,), "--method", "three", "--psram-bytes", "0",
               ok=False)
    assert done.returncode != 0 and "PSRAM" in done.stderr + done.stdout

    # Ranked, the expansion is still made at load, and counted first: the
    # table waits for the placement (here a first execute) and gets what is
    # left. Held, it is reused by the next load; left on flash, it goes with
    # its method and the next load's first execute places it again.
    for budget, table_place, placements in (
            (need(head) + need(table) - 1, "flash", 2),
            (need(head) + need(table), "psram", 1)):
        outs, log = run_script(runner, pte, ptd, steps[:2], "--weight-tier-budget",
                               str(budget), "--weight-tier-tables", "--weight-tier-rank")
        for (out,), expect in zip(outs, want):
            np.testing.assert_array_equal(out, expect)
        s, _, entries = report(log)
        assert entries[names(table)]["place"] == table_place
        assert (s["expanded"], s["evictions"]) == (1, 0)
        copied = 1 + (table_place == "psram")  # five's expansion, and its table
        assert rank_report(log) == {"rank": "on", "placements": placements,
                                    "late": placements, "waiting": 0,
                                    "reading": copied, "copied": copied}


def test_no_budget_again_frees_the_expansions_no_method_holds(runner, tmp_path):
    """Under a budget an expansion outlives its method, for the next method to
    reuse. Setting the budget back to none frees every expansion no loaded
    method holds (release_unused_expansions), so the application gets that
    PSRAM back without loading anything more; one a loaded method still holds
    stays, and goes with that method, as it does without a budget. The
    runner's script sets the option between loads, as an application does,
    and prints the cache's resident bytes after each directive."""
    from executorch.backends.esp32s3.test.fixtures import asym

    pte, ptd, model, descs = tied_head_program(tmp_path)
    (head,) = [d for d in descs if d["flags"] & FLAG_STAGE_I8]
    three = asym.long_input(3, 96)
    want = np.asarray(model.reference(three), np.float32).reshape(-1).view(np.uint32)
    budget = ("--weight-tier-budget", str(need(head)))

    def directives(steps):
        outs, log = run_script(runner, pte, ptd, [("three", (three,), 1), *steps],
                               *budget)
        np.testing.assert_array_equal(outs[0][0], want)
        return [(m["what"], int(m["resident"]), int(m["entries"]))
                for m in DIRECTIVE.finditer(log)], log

    # Destroyed under the budget, the expansion is kept; no budget frees it.
    lines, log = directives(["unload", "set weight_tier_budget 0"])
    assert lines == [("unload", need(head), 1),
                     ("set weight_tier_budget 0", 0, 0)], log[-2000:]
    assert report(log) == (None, {}, {})

    # Still held when the budget goes: it stays, and goes with its method.
    lines, log = directives(["set weight_tier_budget 0", "unload"])
    assert lines == [("set weight_tier_budget 0", need(head), 1),
                     ("unload", 0, 0)], log[-2000:]


class GroupedHead(nn.Module):
    """A one-row int4 head over the first row and first 192 columns of a
    two-row zero-point table: zero codes, unit scales, zero points 1 and 2
    by group, `cols` columns stored in groups of `gsize`."""

    def __init__(self, gsize, cols):
        super().__init__()
        self.gsize = gsize
        self.register_buffer("codes", torch.zeros(2, cols, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(2, cols // gsize))
        self.register_buffer("zeros", torch.tensor([[1, 2], [1, 2]],
                                                   dtype=torch.int8))

    def forward(self, x):
        import torchao  # noqa: F401 -- registers torch.ops.torchao

        table = torch.ops.torchao.dequantize_affine(
            self.codes, [1, self.gsize], self.scales, self.zeros, torch.int8,
            -8, 7)
        return F.linear(x, table[:1, :192])


def test_expansions_in_other_groups_are_not_shared(runner, tmp_path):
    """An expansion folds each scale group's zero point into its columns, so
    two heads over the same packed bytes, scales and zero points -- one
    external weight with the same keys -- in groups of 96 and of 128 need an
    expansion each: the cache names an expansion by its op's rows, row length
    and groups too. Columns 96-127 fold zero point 2 in groups of 96 and 1 in
    groups of 128; either method run after the other, or beside it, gives its
    own output bit for bit, never the other's expansion."""
    x = torch.zeros(1, 192)
    x[0, 96:128] = 1.0
    methods = {name: lower(GroupedHead(gsize, cols), (x,), stage_tied_head=True)
               for name, gsize, cols in (("g96", 96, 192), ("g128", 128, 256))}
    heads = {name: [d for d in descriptors(edge) if d["flags"] & FLAG_STAGE_I8]
             for name, edge in methods.items()}
    assert [len(h) for h in heads.values()] == [1, 1], heads
    a, b = heads["g96"][0], heads["g128"][0]
    same = ("opcode", "wkey", "skey", "n", "k", "k_pad")
    assert [a[f] for f in same] == [b[f] for f in same], (a, b)
    assert (a["scale_group"], b["scale_group"]) == (96, 128)
    pte, ptd = emit(multi_method(methods), tmp_path)
    alone = {name: run(runner, pte, ptd, (x,), "--method", name)[0][0]
             for name in methods}
    np.testing.assert_allclose(alone["g96"].view(np.float32), [-64.0], rtol=1e-5)
    np.testing.assert_allclose(alone["g128"].view(np.float32), [-32.0], rtol=1e-5)
    for order in (("g96", "g128"), ("g128", "g96")):
        for flags in ((), ("--keep-methods",)):
            steps = [(name, (x,), 1) for name in order]
            outs, log = run_script(runner, pte, ptd, steps, *flags)
            for name, (out,) in zip(order, outs):
                np.testing.assert_array_equal(out, alone[name], err_msg=name)
            s, _, entries = report(log)
            assert (s["copies"], s["reuses"], s["expanded"]) == (2, 0, 2), log[-2000:]
            if flags:
                assert len(entries) == 2


# ------------------------------------------------------------------- eviction


class One(nn.Module):
    def __init__(self, seed, n=32):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(64, n)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture(scope="module")
def eviction(tmp_path_factory):
    """Five methods with a weight each: low and low2 read theirs once per
    execute (score 1), high twice (64 rows in blocks of 32), big's is larger
    than the budget the tests set, and mid's is larger than low's but not
    than low's and low2's together. (pte, ptd, steps, descriptors)."""
    shapes = {"low": (1, 32, 1), "low2": (1, 32, 2), "high": (64, 32, 3),
              "big": (1, 128, 4), "mid": (1, 48, 5)}
    methods, inputs, descs = {}, {}, {}
    for name, (rows, n, seed) in shapes.items():
        torch.manual_seed(20 + seed)
        inputs[name] = (torch.randn(rows, 64),)
        methods[name] = lower(One(seed, n), inputs[name], format_policy="i8")
        (descs[name],) = candidates(descriptors(methods[name]))
    pte, ptd = emit(multi_method(methods), tmp_path_factory.mktemp("eviction"))
    return pte, ptd, inputs, descs


def test_eviction_takes_any_unused_copy_and_only_what_fits(runner, eviction):
    """An unused copy -- one no loaded method holds -- is evicted for a
    newcomer that fits once it is gone, whatever the two score: a destroyed
    program's copies give way to the next program's weights instead of
    pinning the budget. Never for a newcomer larger than the whole budget."""
    pte, ptd, inputs, descs = eviction
    low, high = need(descs["low"]), need(descs["high"])
    assert low == high == need(descs["low2"]) < need(descs["big"])
    budget = str(low + low // 2)  # room for one copy

    def sequence(*methods):
        steps = [(m, inputs[m], 1) for m in methods]
        flash, _ = run_script(runner, pte, ptd, steps)
        got, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget", budget)
        same_bits(got, flash)
        s, _, entries = report(log)
        return s, {m: entries.get(names(descs[m])) for m in methods}

    # Higher score: low's unused copy goes, high's comes.
    s, e = sequence("low", "high")
    assert s["evictions"] == 1 and e["low"] is None
    assert (e["high"]["place"], e["high"]["score"]) == ("psram", 2 << 16)

    # Lower score: high's unused copy goes all the same, low's comes.
    s, e = sequence("high", "low")
    assert s["evictions"] == 1 and e["high"] is None
    assert (e["low"]["place"], e["low"]["refs"]) == ("psram", 1)

    # The same score, as every op at decode: low's unused copy goes.
    s, e = sequence("low", "low2")
    assert s["evictions"] == 1 and e["low"] is None
    assert e["low2"]["place"] == "psram"

    # Larger than the budget: nothing to gain, nothing evicted.
    s, e = sequence("low", "big")
    assert s["evictions"] == 0
    assert (e["low"]["place"], e["big"]["place"]) == ("psram", "flash")


class TwoHeads(nn.Module):
    """Two Linear layers side by side on one input, concatenated: `first`'s
    op, then `second`'s, in the delegate."""

    def __init__(self, first, second):
        super().__init__()
        self.first, self.second = first, second

    def forward(self, x):
        return torch.cat([self.first(x), self.second(x)], dim=-1)


def test_a_load_holds_its_hits_before_it_evicts(runner, tmp_path):
    """A load takes its references on the copies it reuses before any of its
    newcomers evicts. Program A (big, then small) fills a budget of exactly
    its two copies and is destroyed; program B (other, then the same small)
    loads. B's first op is a newcomer and small is the unused copy worth
    least, yet B keeps small for its second op and evicts big alone: three
    copies, one reuse, one eviction -- not small evicted and copied again."""
    torch.manual_seed(31)
    big = nn.Linear(64, 64)
    torch.manual_seed(32)
    small = nn.Linear(64, 32)
    torch.manual_seed(33)
    other = nn.Linear(64, 32)
    inputs = (torch.randn(1, 64),)
    methods = {"a": lower(TwoHeads(big, small), inputs, format_policy="i8"),
               "b": lower(TwoHeads(other, small), inputs, format_policy="i8")}
    a, b = (candidates(descriptors(methods[m])) for m in ("a", "b"))
    assert a[1]["wkey"] == b[1]["wkey"] and a[0]["wkey"] != b[0]["wkey"]
    assert need(a[0]) > need(a[1]) == need(b[0])
    pte, ptd = emit(multi_method(methods), tmp_path)
    steps = [("a", inputs, 1), ("b", inputs, 1)]
    flash, _ = run_script(runner, pte, ptd, steps)
    got, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                          str(need(a[0]) + need(a[1])))
    same_bits(got, flash)
    s, _, e = report(log)
    assert (s["copies"], s["reuses"], s["evictions"]) == (3, 1, 1), log[-2000:]
    assert names(a[0]) not in e
    assert (e[names(a[1])]["place"], e[names(a[1])]["refs"]) == ("psram", 1)
    assert e[names(b[0])]["place"] == "psram"


def test_a_newcomer_psram_cannot_hold_evicts_nothing(runner, eviction):
    """A newcomer's PSRAM is allocated before anything is evicted for the
    budget, so no copy is evicted for PSRAM that is already free, and none at
    all when even every unused copy gone would not free enough. When PSRAM is
    short, unused copies are evicted until the allocation succeeds: here one
    larger than the newcomer (--psram-bytes: the host's PSRAM, which also
    holds the cache's table)."""
    from executorch.backends.esp32s3.utils import weight_tiers as wt

    pte, ptd, inputs, descs = eviction
    low, big = need(descs["low"]), need(descs["big"])
    table = wt.table_bytes(1, pointer=8)

    # big after low: the budget would take low out for big, but PSRAM cannot
    # hold big even then. low stays, unused, and big reads flash.
    steps = [("low", inputs["low"], 1), ("big", inputs["big"], 1)]
    flash, _ = run_script(runner, pte, ptd, steps)
    got, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                          str(low + big - 1), "--psram-bytes",
                          str(table + low + big // 2))
    same_bits(got, flash)
    s, _, e = report(log)
    assert s["evictions"] == 0
    assert (e[names(descs["low"])]["place"], e[names(descs["low"])]["refs"]) == (
        "psram", 0)
    assert e[names(descs["big"])]["place"] == "flash"

    # low after big: PSRAM is short of low, and big's unused copy is larger:
    # it goes, and low takes its block.
    steps = [("big", inputs["big"], 1), ("low", inputs["low"], 1)]
    flash, _ = run_script(runner, pte, ptd, steps)
    got, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                          str(FULL), "--psram-bytes", str(table + big + low // 2))
    same_bits(got, flash)
    s, _, e = report(log)
    assert s["evictions"] == 1 and names(descs["big"]) not in e
    assert e[names(descs["low"])]["place"] == "psram"


@pytest.mark.parametrize("budget", ["full", "both copies"])
def test_a_newcomer_evicts_several_smaller_copies_when_psram_is_short(
        runner, eviction, budget):
    """When PSRAM is short, unused copies are evicted, least worth first,
    until the newcomer's allocation succeeds: two smaller ones together where
    neither alone is large enough. low then low2 fill PSRAM; mid, larger
    than each but not than both, loads: both go and mid takes their room.
    (Evicting only one copy at least the newcomer's size finds none and
    leaves mid on flash beside two unused copies.) Under a budget of
    exactly both copies, the same."""
    from executorch.backends.esp32s3.utils import weight_tiers as wt

    pte, ptd, inputs, descs = eviction
    low, low2, mid = need(descs["low"]), need(descs["low2"]), need(descs["mid"])
    assert low == low2 < mid < low + low2
    table = wt.table_bytes(1, pointer=8)
    steps = [(m, inputs[m], 1) for m in ("low", "low2", "mid")]
    flash, _ = run_script(runner, pte, ptd, steps)
    got, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                          str(FULL if budget == "full" else low + low2),
                          "--psram-bytes", str(table + low + low2))
    same_bits(got, flash)
    s, _, e = report(log)
    assert (s["copies"], s["evictions"]) == (3, 2), log[-2000:]
    assert names(descs["low"]) not in e and names(descs["low2"]) not in e
    assert e[names(descs["mid"])]["place"] == "psram"


def test_cache_off_copies_again_and_cache_on_reuses(runner, eviction):
    """The same method loaded twice reuses its copy while the cache is on;
    with it off the copy dies with its method and is made again."""
    pte, ptd, inputs, _ = eviction
    steps = [("low", inputs["low"], 1), ("high", inputs["high"], 1),
             ("low", inputs["low"], 1)]
    flash, _ = run_script(runner, pte, ptd, steps)
    on, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget", str(FULL))
    same_bits(on, flash)
    s, _, _ = report(log)
    assert (s["copies"], s["reuses"]) == (2, 1)
    off, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget", str(FULL),
                          "--no-weight-cache")
    same_bits(off, flash)
    s, _, _ = report(log)
    assert (s["copies"], s["reuses"]) == (3, 0)


# ----------------------------------------------------------- ranked placement


class LastLarge(nn.Module):
    """Four small Linear layers, then one four times their size, all read
    once per execute: a decode step's shape, the largest weight loading last
    as an LLM's output head does."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(14)
        self.fc = nn.ModuleList(nn.Linear(64, 64) for _ in range(4))
        self.head = nn.Linear(64, 256)

    def forward(self, x):
        for fc in self.fc:
            x = torch.relu(fc(x))
        return self.head(x)


def test_ranked_placement_holds_a_large_last_weight(runner, tmp_path):
    """Load order fills the budget with the small weights that load first and
    leaves the large last one on flash, with room to spare; ranked placement
    copies the large one first and fills what is left, first fit, so the
    same budget holds more -- applied, or made by the first execute."""
    torch.manual_seed(15)
    inputs = (torch.randn(1, 64),)
    edge = lower(LastLarge(), inputs, format_policy="i8")
    descs = candidates(descriptors(edge))
    assert len(descs) == 5 and need(descs[-1]) > 3 * max(map(need, descs[:4]))
    small, big = descs[:4], descs[4]
    budget = need(big) + need(small[0]) * 3 // 2
    pte, ptd = emit(edge, tmp_path)
    flash, _ = run(runner, pte, ptd, inputs)

    got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", str(budget))
    same_bits(got, flash)
    in_order, _, entries = report(log)
    assert entries[names(big)]["place"] == "flash"
    assert [entries[names(d)]["place"] for d in small] == ["psram"] * 4
    assert rank_report(log) is None

    for apply in (("--weight-tier-apply",), ()):
        got, log = run(runner, pte, ptd, inputs, "--weight-tier-budget", str(budget),
                       "--weight-tier-rank", *apply)
        same_bits(got, flash)
        s, _, entries = report(log)
        # Every weight is read once: the largest goes first, then the first
        # small one (equal sizes keep load order); no second one fits.
        assert [entries[names(d)]["place"] for d in descs] == [
            "psram", "flash", "flash", "flash", "psram"]
        assert s["resident"] == need(big) + need(small[0]) > in_order["resident"]
        assert {k for k, e in entries.items() if e["place"] == "psram"} == \
            ranked_first_fit(entries, budget)
        rank = rank_report(log)
        assert rank["late"] == (0 if apply else 1)
        assert (rank["reading"], rank["copied"]) == (2, 2)


def test_ranked_placement_spans_every_loaded_method(runner, eviction):
    """Two methods held at once, as an application holds prefill and decode:
    load order gives the budget to the weight of the method loaded first,
    which no later load may evict while it is held; ranked placement, applied
    once after both loads or made by the first execute of either, gives it to
    the weight scoring higher, whichever method loads it."""
    pte, ptd, inputs, descs = eviction
    low = need(descs["low"])
    budget = str(low + low // 2)  # room for one copy
    steps = [("low", inputs["low"], 1), ("high", inputs["high"], 1),
             ("low", inputs["low"], 1)]
    flash, _ = run_script(runner, pte, ptd, steps)

    got, log = run_script(runner, pte, ptd, steps, "--keep-methods",
                          "--weight-tier-budget", budget)
    same_bits(got, flash)
    s, _, e = report(log)
    assert (e[names(descs["low"])]["place"], e[names(descs["high"])]["place"]) == (
        "psram", "flash")
    assert s["evictions"] == 0

    for apply in (("--weight-tier-apply",), ()):
        got, log = run_script(runner, pte, ptd, steps, "--keep-methods",
                              "--weight-tier-budget", budget, "--weight-tier-rank",
                              *apply)
        same_bits(got, flash)
        s, _, e = report(log)
        assert (e[names(descs["low"])]["place"], e[names(descs["high"])]["place"]) == (
            "flash", "psram")
        assert (s["copies"], s["evictions"], s["skipped"]) == (1, 0, 1)
        assert rank_report(log) == {"rank": "on", "placements": 1,
                                    "late": 0 if apply else 1, "waiting": 0,
                                    "reading": 1, "copied": 1}


def test_ranked_loads_reuse_copies_and_evict_unused_ones(runner, eviction):
    """One method at a time, ranked and applied after each load as the story
    does: a placement evicts an unused copy for its candidate whatever the two
    score, never for one larger than the whole budget, and a reload finds its
    copy at load, leaving nothing to place."""
    pte, ptd, inputs, descs = eviction
    low = need(descs["low"])

    def sequence(*methods, budget=low + low // 2):
        steps = [(m, inputs[m], 1) for m in methods]
        flash, _ = run_script(runner, pte, ptd, steps)
        got, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                              str(budget), "--weight-tier-rank", "--weight-tier-apply")
        same_bits(got, flash)
        s, _, entries = report(log)
        return s, rank_report(log), {m: entries.get(names(descs[m])) for m in methods}

    s, _, e = sequence("low", "high")
    assert s["evictions"] == 1 and e["low"] is None
    assert e["high"]["place"] == "psram"
    s, _, e = sequence("high", "low")
    assert s["evictions"] == 1 and e["high"] is None
    assert e["low"]["place"] == "psram"
    s, _, e = sequence("low", "low2")
    assert s["evictions"] == 1 and e["low"] is None
    assert e["low2"]["place"] == "psram"
    s, _, e = sequence("low", "big")
    assert s["evictions"] == 0
    assert (e["low"]["place"], e["big"]["place"]) == ("psram", "flash")
    s, rank, _ = sequence("low", "high", "low", budget=FULL)
    assert (s["copies"], s["reuses"], s["evictions"]) == (2, 1, 0)
    assert rank == {"rank": "on", "placements": 2, "late": 0, "waiting": 0,
                    "reading": 1, "copied": 1}


# ---------------------------------------- a placement while another method runs


class Stack(nn.Module):
    """Four int8 Linear layers with ReLU: four candidates, a few ops a call."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(21)
        self.layers = nn.Sequential(*[m for _ in range(4)
                                      for m in (nn.Linear(96, 96), nn.ReLU())])

    def forward(self, x):
        return self.layers(x)


STRESS = re.compile(
    r"^placement stress: (?P<rounds>\d+) round\(s\); run executed (?P<runs>\d+) "
    r"time\(s\), (?P<after>\d+) after a placement; (?P<partner>\w+) executed "
    r"(?P<places>\d+) time\(s\); placements (?P<placements>\d+), copies "
    r"(?P<copies>\d+); every copy read in (?P<read>\d+) round\(s\); "
    r"(?P<diffs>\d+) mismatch\(es\)$", re.M)
ROUNDS = 30


@pytest.fixture(scope="module")
def stress_program(runner, tmp_path_factory):
    """Method "run" (Stack); method "place" (One), whose weights are its own;
    method "share", Stack again, whose weights are run's (the same keys, so
    the same cache entries). Per partner, the copies one placement makes;
    each method's outputs run alone, as bit patterns."""
    directory = tmp_path_factory.mktemp("stress")
    torch.manual_seed(22)
    x = torch.randn(2, 96)
    inputs = {"run": (x,), "place": (torch.randn(1, 64),), "share": (x,)}
    methods = {"run": lower(Stack(), inputs["run"]),
               "place": lower(One(7), inputs["place"]),
               "share": lower(Stack(), inputs["share"])}
    weights = {name: {names(d) for d in candidates(descriptors(edge))}
               for name, edge in methods.items()}
    assert weights["share"] == weights["run"] and not weights["place"] & weights["run"]
    wanted = {partner: len(weights["run"] | weights[partner])
              for partner in ("place", "share")}
    pte, ptd = emit(multi_method(methods), directory)
    alone = {name: run(runner, pte, ptd, inputs[name], "--method", name)[0][0]
             for name in methods}
    return pte, ptd, inputs, wanted, alone


def placement_stress(path, program, partner, slots=None):
    """The stress (esp32_host_runner --placement-stress) of run against
    `partner` through the runner at `path`, with `slots` private placement
    marks (--tier-slots; 0 puts every handle on the overflow path): its
    result, checked."""
    pte, ptd, inputs, wanted, alone = program
    lines, outs = [], {}
    for name in ("run", partner):
        (given,) = write_inputs(pte.parent, f"stress_{name}", inputs[name])
        outs[name] = pte.parent / f"stress_{name}_out.bin"
        outs[name].unlink(missing_ok=True)
        lines.append(f"{name} {given} -> {outs[name]}")
    script = pte.parent / "stress.txt"
    script.write_text("\n".join(lines) + "\n")
    done = subprocess.run(
        [str(path), "--pte", str(pte), "--ptd", str(ptd), "--script", str(script),
         "--placement-stress", str(ROUNDS)]
        + ([] if slots is None else ["--tier-slots", str(slots)]),
        capture_output=True, text=True, timeout=1200,
        env={**os.environ, "TSAN_OPTIONS": "halt_on_error=1"})
    log = done.stdout + done.stderr
    assert done.returncode == 0 and "ThreadSanitizer" not in log, log[-6000:]
    assert "AddressSanitizer" not in log, log[-6000:]
    m = STRESS.search(done.stdout)
    assert m is not None, log[-3000:]
    got = {k: (v if k == "partner" else int(v)) for k, v in m.groupdict().items()}
    # Every round placed, and copied the running method's weights (with the
    # partner's own); the running method executed three times after each
    # placement; then every op of both, whose weight had a copy, read it.
    assert (got["rounds"], got["placements"], got["places"]) == (ROUNDS,) * 3, got
    assert got["copies"] == wanted[partner] * ROUNDS, got
    assert got["after"] == 3 * ROUNDS and got["read"] == ROUNDS, got
    assert got["diffs"] == 0, got
    for name in ("run", partner):
        np.testing.assert_array_equal(np.fromfile(outs[name], dtype=np.uint32),
                                      alone[name], err_msg=name)


@pytest.mark.parametrize("slots", [None, 0], ids=["slots", "overflow"])
@pytest.mark.parametrize("partner", ["place", "share"])
def test_placement_while_another_method_executes(runner, stress_program, partner,
                                                 slots):
    """A ranked placement applied on one thread while another thread executes
    a method whose weights it copies -- the placed method's weights its own,
    or the running method's: the placement moves no handle's pointers, and
    each execute after it repoints its own ops before reading any
    (weight_tier_sync), so every execute of both methods, before and after
    the copies, gives the bits each gives alone, and both read every copy
    afterwards. A shared copy outlives the placed method's destroy while the
    running method holds it. Also with every handle on the overflow path,
    whose placement marks live in the handles' links (--tier-slots 0)."""
    placement_stress(runner, stress_program, partner, slots)


@pytest.mark.parametrize("slots", [None, 0], ids=["slots", "overflow"])
@pytest.mark.parametrize("partner", ["place", "share"])
def test_placement_while_another_method_executes_under_tsan(runner, stress_program,
                                                            partner, slots):
    """The same under ThreadSanitizer (esp32_host_runner_tsan): no read of a
    pointer races a write. A placement that wrote the running handle's
    pointers -- release stores against the ops' plain loads -- is a data race,
    reported here, even where both pointers reach the same bytes."""
    tsan = runner.parent / "esp32_host_runner_tsan"
    if not tsan.is_file():
        pytest.skip(f"no ThreadSanitizer runner at {tsan} (no program built "
                    "with -fsanitize=thread runs on this host, or "
                    "ESP32_HOST_TSAN=OFF)")
    placement_stress(tsan, stress_program, partner, slots)


IDLE = re.compile(
    r"^placement idle: (?P<placements>\d+) placement\(s\) while run waited; then it "
    r"read (?P<reads>\d+) of (?P<copied>\d+) copied op\(s\); outputs (?P<outputs>.+)$",
    re.M)


@pytest.mark.parametrize("count, slots", [(1, None), (255, None), (256, None),
                                          (257, None), (256, 0)])
def test_an_idle_method_reads_its_copies_after_any_number_of_placements(
        runner, stress_program, count, slots):
    """A method loaded and left idle while `count` ranked placements run
    (esp32_host_runner --placement-idle) reads the copies the first of them
    made at its next execute, whatever the count. A placement marks every
    live handle, and the mark stays until the handle syncs. A wrapping counter
    would not do: with a byte that 256 placements bring back to the value the
    handle saved, it would read flash for good. With --tier-slots 0 the
    handle takes the overflow path, its mark in its link."""
    pte, ptd, inputs, wanted, _ = stress_program
    lines = []
    for name in ("run", "place"):
        (given,) = write_inputs(pte.parent, f"idle_{name}", inputs[name])
        out = pte.parent / f"idle_{name}_out.bin"
        out.unlink(missing_ok=True)
        lines.append(f"{name} {given} -> {out}")
    script = pte.parent / "idle.txt"
    script.write_text("\n".join(lines) + "\n")
    done = subprocess.run(
        [str(runner), "--pte", str(pte), "--ptd", str(ptd), "--script", str(script),
         "--placement-idle", str(count)]
        + ([] if slots is None else ["--tier-slots", str(slots)]),
        capture_output=True, text=True, timeout=600)
    log = done.stdout + done.stderr
    m = IDLE.search(done.stdout)
    assert m is not None, log[-3000:]
    run_ops = wanted["share"]  # share's weights are run's: run's candidate ops
    assert (int(m["placements"]), int(m["reads"]), int(m["copied"]), m["outputs"]) == (
        count, run_ops, run_ops, "the same"), log[-3000:]
    assert done.returncode == 0, log[-3000:]


# --------------------------------------------------------- the multi-method glimmer


def test_glimmer_decode_and_prefill_share_one_set_of_copies(runner, tmp_path):
    """The nano Glimmer's decode and prefill share one weights file: under a
    full budget the first method copies each weight once and the other reuses
    it, with the KV state carried across both and every output bit-identical
    to flash -- at full and at a partial budget, tables included."""
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
        _replace_glimmer_tables_i4,
        build,
        build_prefill,
    )
    from executorch.backends.esp32s3.utils.multimethod import (
        combine_shared_methods,
        shared_memory_planning_pass,
    )

    common = dict(arm="ple", vocab=256, d_model=64, layers=2, heads=8,
                  kv_heads=1, ffn=192, ple_dim=24, seq_len=16, window=8,
                  softcap=20.0, cache_context=16, selected_rows=7,
                  head_token_ids=list(range(9)))
    _, decode = build(local_ring=True, **common)
    _, prefill = build_prefill(chunk_size=4, **common)
    rows = torch.arange(7, dtype=torch.long)
    options = dict(format_policy="i8", group_weights=True, quantize_qk=True)
    edges = {}
    for name, model, tokens in (("decode", decode, torch.tensor([3])),
                                ("prefill", prefill, torch.tensor([3, 4, 5, 6]))):
        model, _ = _replace_glimmer_tables_i4(model)
        edges[name] = lower(model, (tokens, torch.tensor(0), rows),
                            max_params=250_000, **options)
    descs = {name: descriptors(edge) for name, edge in edges.items()}
    tables = {names(d) for ds in descs.values() for d in candidates(ds, tables=True)}
    plain = {names(d) for ds in descs.values() for d in candidates(ds)}
    assert plain < tables
    et = combine_shared_methods(edges).to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=shared_memory_planning_pass()))
    pte = tmp_path / "model.pte"
    pte.write_bytes(et.buffer)
    et.write_tensor_data_to_file(str(tmp_path))
    ptd = tmp_path / f"{TAG}.ptd"
    n_out = {plan.name: len(plan.outputs) for plan in et.executorch_program.execution_plan}

    steps = [("prefill", (torch.tensor([5, 9, 2, 7]), torch.tensor(0), rows),
              n_out["prefill"]),
             ("decode", (torch.tensor([11]), torch.tensor(4), rows), n_out["decode"]),
             ("decode", (torch.tensor([13]), torch.tensor(5), rows), n_out["decode"]),
             ("prefill", (torch.tensor([1, 2, 3, 4]), torch.tensor(6), rows),
              n_out["prefill"])]
    flash, _ = run_script(runner, pte, ptd, steps)

    full, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget", str(FULL))
    same_bits(full, flash)
    s, _, entries = report(log)
    assert set(entries) <= plain and s["copies"] == len(plain)
    assert s["reuses"] > 0 and s["evictions"] == 0 and s["skipped"] == 0

    with_tables, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                                  str(FULL), "--weight-tier-tables")
    same_bits(with_tables, flash)
    s, _, _ = report(log)
    assert s["copies"] == len(tables)

    third = s["resident"] // 3
    partial, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                              str(third), "--weight-tier-tables")
    same_bits(partial, flash)
    s, _, entries = report(log)
    assert {e["place"] for e in entries.values()} == {"psram", "flash"}

    # Ranked, applied after each load as the story firmware does, or left to
    # each load's first execute: the same bits at full and at a third.
    for budget in (FULL, third):
        for apply in (("--weight-tier-apply",), ()):
            ranked, log = run_script(runner, pte, ptd, steps, "--weight-tier-budget",
                                     str(budget), "--weight-tier-tables",
                                     "--weight-tier-rank", *apply)
            same_bits(ranked, flash)
            s, _, entries = report(log)
            rank = rank_report(log)
            assert rank["waiting"] == 0 and rank["placements"] >= 1
            assert rank["late"] == (0 if apply else rank["placements"])
            assert rank["reading"] == rank["copied"] > 0
            if budget == FULL:
                assert s["copies"] == len(tables) and s["skipped"] == 0
            else:
                assert {e["place"] for e in entries.values()} == {"psram", "flash"}
                assert s["resident"] <= third
