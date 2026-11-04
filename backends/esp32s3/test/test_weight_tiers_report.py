# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The weight-tier report against the backend: the copies it predicts are the
ones the host runner makes.

utils/weight_tiers.py reads a lowered program's delegate blobs and replays
the backend's weight cache (CacheModel): which weights each method load
copies into PSRAM, reuses, evicts or leaves on flash. Here the same programs
run on the host runner with the same options (--weight-tier-budget N|auto,
--weight-tier-reserve, --weight-tier-tables, --no-weight-cache,
--psram-bytes), and its report -- an "esp32 tier:" line for every weight the
cache knows, with its class, bytes, score, users and place, and the summary's
counts -- must equal the prediction entry for entry: every stored format
(int4, int8, int16, fp16, bf16, fp32), convolutions (whose scores differ),
budgets from none to full, auto with a reserve under a small PSRAM, no PSRAM
at all, methods in turn that evict each other, an int4 head expanded to int8,
and the nano Glimmer's decode and prefill sharing their weights. Then the
report's own arithmetic: the best packing against brute force, the curve, and
the measured rates it reads.

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import itertools
import json
import random
import re

import pytest
import torch

from executorch.backends.esp32s3.target import chips
from executorch.backends.esp32s3.test.host import host_runner
from executorch.backends.esp32s3.test.test_weight_tiers import (
    FORMATS,
    FULL,
    One,
    emit,
    lower,
    multi_method,
    run,
    run_script,
    tied_head_program,
)
from executorch.backends.esp32s3.utils import weight_tiers as wt

HOST_POINTER = 8  # the host runner's pointers size the cache's table entries

ENTRY = re.compile(
    r"^esp32 tier: (?P<cls>\w+) (?P<kind>\S+) (?P<place>psram|flash) (?P<bytes>\d+) B "
    r"score (?P<score>\d+) refs (?P<refs>\d+) names (?P<names>[0-9a-f]{16})$", re.M)
SUMMARY = re.compile(
    r"^esp32 tiers: budget .+?, resident (?P<resident>\d+) B, wanted (?P<wanted>\d+) B, "
    r"entries (?P<entries>\d+); copies (?P<copies>\d+), reuses (?P<reuses>\d+), "
    r"evictions (?P<evictions>\d+), left on flash (?P<skipped>\d+), "
    r"expanded (?P<expanded>\d+)$", re.M)


@pytest.fixture(scope="module")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


def flags(budget=0, reserve=0, tables=False, cache=True, psram=None):
    """The runner's flags for a CacheModel's options."""
    out = []
    if budget:
        out += ["--weight-tier-budget", "auto" if budget == "auto" else str(budget)]
    if reserve:
        out += ["--weight-tier-reserve", str(reserve)]
    if tables:
        out.append("--weight-tier-tables")
    if not cache:
        out.append("--no-weight-cache")
    if psram is not None:
        out += ["--psram-bytes", str(psram)]
    return out


def actual(log):
    """The runner's cache: its entries and its summary (None when it knew
    no weight)."""
    entries = sorted((int(m["names"], 16), m["cls"], m["kind"], m["place"],
                      int(m["bytes"]), int(m["score"]), int(m["refs"]))
                     for m in ENTRY.finditer(log))
    summary = SUMMARY.search(log)
    return entries, summary and {k: int(v) for k, v in summary.groupdict().items()}


def predicted(cache):
    """The same, from the model of the cache."""
    known = cache.known()
    entries = sorted((e.names, e.cls, e.kind, "psram" if e.raw else "flash",
                      e.alloc, e.score, e.refs) for e in known)
    if not known:
        return entries, None
    return entries, {
        "resident": cache.resident,
        "wanted": sum(e.alloc for e in known if e.refs > 0),
        "entries": len(known), "copies": cache.copies, "reuses": cache.hits,
        "evictions": cache.evictions, "skipped": cache.skipped,
        "expanded": cache.expanded}


def same_cache(log, cache):
    got, want = actual(log), predicted(cache)
    assert got == want, f"runner {got}\npredicted {want}"
    return got


# ------------------------------------------------- every stored format


@pytest.fixture(scope="module")
def programs(tmp_path_factory):
    """name -> (pte, ptd, inputs, outputs, the report's reading of the pte),
    for test_weight_tiers.py's three format programs: five Linear layers in
    int4, int8, int16, fp16 and bf16; int4, int8 and depthwise int8
    convolutions; an fp32 convolution and Linear."""
    out = {}
    for name, (cls, make_inputs, options, _) in FORMATS.items():
        torch.manual_seed(11)
        inputs = make_inputs()
        model = cls()
        edge = lower(model, inputs, **options)
        with torch.no_grad():
            produced = model(*inputs)
        n_out = len(produced) if isinstance(produced, tuple) else 1
        pte, ptd = emit(edge, tmp_path_factory.mktemp(name))
        out[name] = (pte, ptd, inputs, n_out, wt.read_program(pte.read_bytes()))
    return out


@pytest.mark.parametrize("name", list(FORMATS))
def test_predicted_copies_are_the_runners_at_every_budget(runner, programs, name):
    """At no budget, a third, a half, one byte short of every weight, every
    weight and more than any: the same entries, places, scores and counts."""
    pte, ptd, inputs, n_out, program = programs[name]
    total = sum(w.need for w in program.weights())
    assert total > 0
    places = {}
    for budget in (0, total // 3, total // 2, total - 1, total, FULL):
        _, log = run(runner, pte, ptd, inputs, *flags(budget), n_out=n_out)
        entries, _ = same_cache(log, wt.predict(program, budget=budget,
                                                pointer=HOST_POINTER))
        places[budget] = sorted(entry[3] for entry in entries)
    weights = len(program.weights())
    assert places[0] == []
    assert places[total] == places[FULL] == ["psram"] * weights
    assert "flash" in places[total - 1] and "psram" in places[total // 2]


@pytest.mark.parametrize("name", list(FORMATS))
def test_predicted_auto_and_psram_are_the_runners(runner, programs, name):
    """Auto with a reserve under a small PSRAM, where the cache's own table
    takes PSRAM too and fp16 stays on flash; a budget larger than the PSRAM;
    and no PSRAM at all, as under QEMU."""
    pte, ptd, inputs, n_out, program = programs[name]
    total = sum(w.need for w in program.weights())
    cases = [dict(budget="auto"),
             dict(budget="auto", psram=total, reserve=total // 3),
             dict(budget=FULL, psram=total // 2),
             dict(budget=FULL, psram=0)]
    for options in cases:
        _, log = run(runner, pte, ptd, inputs, *flags(**options), n_out=n_out)
        same_cache(log, wt.predict(program, pointer=HOST_POINTER, **options))


def test_scores_follow_each_loop(programs):
    """Matmuls at m rows read their weights once per row block (int8 and
    int4) or once per row (fp32, int16; fp16 and bf16 unless one block of
    rows is widened for all); a depthwise conv once per output position."""
    scores = {}
    for name in FORMATS:
        for weight in programs[name][4].weights():
            scores.setdefault((name, weight.cls), set()).add(weight.score)
    one = wt.SCORE_ONE
    # The ladder runs 3 rows: int8 and int4 in one block, int16 row by row,
    # fp16 and bf16 widened once.
    assert scores[("ladder", "int8")] == scores[("ladder", "int4")] == {one}
    assert scores[("ladder", "int16")] == {3 * one}
    assert scores[("ladder", "fp16")] == scores[("ladder", "bf16")] == {one}
    # 64 output positions: the dense convs in blocks, the depthwise one per
    # position.
    assert scores[("convs", "int4")] <= {(-(-64 // b)) * one for b in range(1, 33)}
    assert 64 * one in scores[("convs", "int8")]


# ------------------------------------------------- methods loaded in turn


@pytest.fixture(scope="module")
def eviction(tmp_path_factory):
    """test_weight_tiers.py's five methods: low and low2 read their weight
    once per execute, high twice, big's is the largest, and mid's is larger
    than low's but not than low's and low2's together."""
    shapes = {"low": (1, 32, 1), "low2": (1, 32, 2), "high": (64, 32, 3),
              "big": (1, 128, 4), "mid": (1, 48, 5)}
    methods, inputs = {}, {}
    for name, (rows, n, seed) in shapes.items():
        torch.manual_seed(20 + seed)
        inputs[name] = (torch.randn(rows, 64),)
        methods[name] = lower(One(seed, n), inputs[name], format_policy="i8")
    pte, ptd = emit(multi_method(methods), tmp_path_factory.mktemp("eviction"))
    return pte, ptd, inputs, wt.read_program(pte.read_bytes())


def test_predicted_evictions_are_the_runners(runner, eviction):
    """Methods loaded one at a time with room for one copy: a newcomer evicts
    an unused copy whatever the two score, and one larger than the budget
    evicts nothing; with the cache off each copy dies with its method. Under
    a small PSRAM the newcomer allocates first: nothing is evicted for one
    PSRAM cannot hold even then, and otherwise unused copies make room until
    it allocates -- one larger, or two smaller together."""
    pte, ptd, inputs, program = eviction
    low = program.methods["low"][0][0].need
    big = program.methods["big"][0][0].need
    mid = program.methods["mid"][0][0].need
    table = wt.table_bytes(1, pointer=HOST_POINTER)
    sequences = [("low", "high"), ("high", "low"), ("low", "low2"),
                 ("low", "big"), ("high", "low", "low2", "big", "high")]
    for options in [dict(budget=low + low // 2), dict(budget=FULL),
                    dict(budget=FULL, cache=False)]:
        for sequence in sequences:
            steps = [(m, inputs[m], 1) for m in sequence]
            _, log = run_script(runner, pte, ptd, steps, *flags(**options))
            same_cache(log, wt.predict(program, sequence, pointer=HOST_POINTER,
                                       **options))
    for sequence, options in [
            (("low", "big"), dict(budget=low + big - 1, psram=table + low + big // 2)),
            (("big", "low"), dict(budget=FULL, psram=table + big + low // 2)),
            (("big", "low", "big", "high"), dict(budget=low + big,
                                                 psram=table + big + low // 2)),
            (("low", "low2", "mid"), dict(budget=FULL, psram=table + 2 * low)),
            (("low", "low2", "mid"), dict(budget=2 * low, psram=table + 2 * low))]:
        steps = [(m, inputs[m], 1) for m in sequence]
        _, log = run_script(runner, pte, ptd, steps, *flags(**options))
        cache = wt.predict(program, sequence, pointer=HOST_POINTER, **options)
        same_cache(log, cache)
        if sequence[-1] == "mid":
            # Neither small copy holds mid alone; both go and mid is copied.
            assert low < mid < 2 * low
            assert (cache.evictions, [e.alloc for e in cache.staged()]) == (2, [mid])


def test_predicted_expanded_head_is_the_runners(runner, tmp_path):
    """An int4 head FLAG_STAGE_I8 expands is copied whatever the budget,
    named with its zero-point table, counted first and shared by both
    methods; a budget one byte short of it and the table leaves the table on
    flash."""
    from executorch.backends.esp32s3.test.fixtures import asym

    pte, ptd, _, _ = tied_head_program(tmp_path)
    program = wt.read_program(pte.read_bytes())
    weights = program.weights()
    (head,) = [w for w in weights if w.must]
    (table,) = [w for w in weights if w.table]
    assert head.cls == "int8" and head.kind == "i4->i8"
    three, five = asym.long_input(3, 96), asym.long_input(5, 96)
    steps = [("three", (three,), 1), ("five", (five,), 1), ("three", (three,), 1)]
    sequence = [name for name, _, _ in steps]
    for options in [dict(), dict(budget=head.need + table.need - 1, tables=True),
                    dict(budget=head.need + table.need, tables=True)]:
        _, log = run_script(runner, pte, ptd, steps, *flags(**options))
        same_cache(log, wt.predict(program, sequence, pointer=HOST_POINTER,
                                   **options))
    rep = wt.report(program, budgets=[0, head.need], tables=True)
    assert [row["load_order"]["bytes"] for row in rep.curve] == [head.need] * 2
    with pytest.raises(wt.TierInitError):
        wt.predict(program, ["three"], psram=0, pointer=HOST_POINTER)


def test_predicted_glimmer_is_the_runners(runner, tmp_path):
    """The nano Glimmer, prefill and decode in turn over one weights file:
    every weight shared, prefill's scores above decode's, tables only when
    asked; at a full budget and a third of it."""
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
        _replace_glimmer_tables_i4,
        build,
        build_prefill,
    )
    from executorch.exir import ExecutorchBackendConfig
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
    et = combine_shared_methods(edges).to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        memory_planning_pass=shared_memory_planning_pass()))
    pte = tmp_path / "model.pte"
    pte.write_bytes(et.buffer)
    et.write_tensor_data_to_file(str(tmp_path))
    ptd = tmp_path / "tier_weights.ptd"
    n_out = {plan.name: len(plan.outputs) for plan in et.executorch_program.execution_plan}
    program = wt.read_program(et.buffer)
    assert set(program.methods) == {"decode", "prefill"}

    steps = [("prefill", (torch.tensor([5, 9, 2, 7]), torch.tensor(0), rows),
              n_out["prefill"]),
             ("decode", (torch.tensor([11]), torch.tensor(4), rows), n_out["decode"]),
             ("prefill", (torch.tensor([1, 2, 3, 4]), torch.tensor(5), rows),
              n_out["prefill"])]
    sequence = [name for name, _, _ in steps]
    wanted = wt.report(program, budgets=[], tables=True).wanted
    for options in [dict(budget=FULL), dict(budget=FULL, tables=True),
                    dict(budget=wanted // 3, tables=True)]:
        _, log = run_script(runner, pte, ptd, steps, *flags(**options))
        same_cache(log, wt.predict(program, sequence, pointer=HOST_POINTER,
                                   **options))
    # Decode alone, as the story loads it: what pays is what it copies.
    rep = wt.report(program, budgets=[FULL], sequence=["decode"])
    _, log = run_script(runner, pte, ptd, steps[1:2], *flags(FULL))
    assert actual(log)[1]["resident"] == rep.bytes_that_pay
    assert rep.curve[0]["load_order"]["bytes"] == rep.bytes_that_pay


# ------------------------------------------------- the report's arithmetic


def fake(need, score=wt.SCORE_ONE, cls="int4", method="decode"):
    """A Weight for the arithmetic alone."""
    op = wt.TierOp(method=method, delegate=0, op=0, opcode=3, cls=cls,
                   kind="stored", names=random.getrandbits(64), wkey="w",
                   skey="s", wbytes=need - 15, sbytes=0, src_bytes=need - 15,
                   need=need, score=score, table=False, must=False)
    return wt.Weight(op)


def brute_best(weights, room):
    best = 0
    for n in range(len(weights) + 1):
        for subset in itertools.combinations(weights, n):
            if sum(w.need for w in subset) <= room:
                best = max(best, sum(w.score * w.need for w in subset))
    return best


@pytest.mark.parametrize("equal_scores", [True, False])
def test_best_packing_is_the_best_subset(equal_scores):
    """The subset sum (one score) and the knapsack (several) find the most
    score-weighted bytes any subset fits, at every room."""
    rng = random.Random(5 + equal_scores)
    for _ in range(20):
        weights = [fake(rng.randrange(100, 5000),
                        wt.SCORE_ONE if equal_scores else
                        rng.choice([1, 2, 3, 36]) * wt.SCORE_ONE)
                   for _ in range(rng.randrange(1, 10))]
        total = sum(w.need for w in weights)
        rooms = sorted(rng.randrange(0, total + 1) for _ in range(5)) + [total]
        sets, unit = wt.best_sets(weights, rooms)
        assert unit == 1
        for room, chosen in zip(rooms, sets):
            assert sum(w.need for w in chosen) <= room
            assert sum(w.score * w.need for w in chosen) == brute_best(weights, room)


def test_ranking_beats_load_order_when_the_large_weight_loads_last():
    """Three 30 KB weights, then a 64 KB one, all read once per execute: 100
    KB of budget copies 90 KB in load order and 94 KB ranked, the best
    there is; the rates price each."""
    sizes = (30_000, 30_000, 30_000, 64_000)
    ops = [wt.TierOp(method="decode", delegate=i, op=0, opcode=3, cls="int4",
                     kind="stored", names=i, wkey=f"w{i}", skey=f"s{i}",
                     wbytes=s - 15, sbytes=0, src_bytes=s - 15, need=s,
                     score=wt.SCORE_ONE, table=False, must=False)
           for i, s in enumerate(sizes)]
    program = wt.Program({"decode": [[op] for op in ops]})
    rep = wt.report(program, budgets=[100_000, 154_000])
    tight, full = rep.curve
    assert tight["load_order"]["bytes"] == 90_000
    assert tight["ranked"]["bytes"] == tight["best"]["bytes"] == 94_000
    assert {full[k]["bytes"] for k in ("load_order", "ranked", "best")} == {154_000}
    per_byte = rep.rates["int4"].saved_us_per_byte
    assert tight["ranked"]["ms"] == pytest.approx((94_000 - 30) * per_byte / 1000)
    assert rep.bytes_that_pay == 154_000


def test_curve_is_bounded_and_ordered(programs):
    """Every column fits its budget, the best is never beaten, and at a budget
    holding every candidate all three copy all of them."""
    for name in FORMATS:
        program = programs[name][4]
        rep = wt.report(program)
        assert rep.curve[-1]["budget"] >= rep.wanted
        for row in rep.curve:
            for column in ("load_order", "ranked", "best"):
                assert row[column]["bytes"] <= row["budget"]
            assert row["best"]["value"] >= row["ranked"]["value"]
            assert row["best"]["value"] >= row["load_order"]["value"]
        last = rep.curve[-1]
        assert {last[k]["bytes"] for k in ("load_order", "ranked", "best")} == {
            rep.wanted}
        text = wt.render(rep)
        assert "budget curve" in text and "bytes that pay" in text
        assert json.loads(json.dumps(rep.as_json()))["wanted"] == rep.wanted


def test_reading_the_flatbuffer_finds_what_a_scan_finds(programs):
    """The program's delegates, read from its flatbuffer in init order, hold
    the ops a magic scan of the file finds."""
    for name in FORMATS:
        pte = programs[name][0]
        read = programs[name][4]
        scanned = wt.scan_program(pte.read_bytes())
        assert scanned.source == "scan" and read.source == "flatbuffer"
        strip = lambda op: (op.names, op.need, op.score, op.cls, op.kind)  # noqa: E731
        assert [strip(op) for op in read.ops()] == [strip(op) for op in scanned.ops()]


def test_rates_are_the_measured_ones():
    """int4 reads flash and PSRAM at its kernel's own two-core rates (the
    story bench build's rows), the float loops at their one-core rates; every
    rate main.cpp's bench rows want is measured, at one PSRAM timing point;
    the flash budget is the lean table's model region."""
    from executorch.examples.espressif.esp32s3.scripts import flash_layout

    assert chips.ESP32S3.flash_model_budget == flash_layout.ET_PACKAGE_BUDGET
    rates = wt.class_rates("papercolor", cores=2, caches="story")
    int4 = rates["int4"]
    assert (int4.flash_key.kernel, int4.flash_key.cores, int4.flash.mb_s) == ("i4", 2, 32.4)
    assert (int4.psram_key.kernel, int4.psram_key.cores, int4.psram.mb_s) == (
        "i4", 2, 91.1)
    assert int4.saved_us_per_byte == pytest.approx(1 / 32.4 - 1 / 91.1)
    # The float loop runs on one core: its one-core rows are the rates.
    fp16 = rates["fp16"]
    assert (fp16.flash_key.cores, fp16.flash.mb_s, fp16.psram.mb_s) == (1, 24.8, 26.3)
    assert rates["bf16"].saved_us_per_byte == pytest.approx(1 / 31.2 - 1 / 45.3)
    # An exact lookup names the PSRAM timing point the rows were measured at.
    assert chips.rate("papercolor", "flash", "i4", 2, "story") is None
    assert chips.rate("papercolor", "flash", "i4", 2, "story",
                      chips.STORY_BENCH_MSPI).mb_s == 32.4
    assert chips.unmeasured() == ()
    assert chips.RateKey("papercolor", "flash", "f16", 1, "story") in chips.WANTED
    assert chips.RateKey("papercolor", "flash", "f16", 2, "story") not in chips.WANTED
    rate, key = chips.nearest_rate("papercolor", "psram", "i4", 1, "story")
    assert (key.caches, key.cores, key.mspi, rate.mb_s) == (
        "story", 1, chips.STORY_BENCH_MSPI, 73.3)
    assert chips.nearest_rate("nano", "psram", "i4", 1, "story") == (None, None)
