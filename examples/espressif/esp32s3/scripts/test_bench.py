# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bench: its image format against the firmware's reader, its log parser,
its input generator, and one model exported and run through the firmware's
bench runner on the host (esp32_bench_host).

    python -m pytest examples/espressif/esp32s3/scripts/test_bench.py

The end-to-end case lowers DS-CNN (about 30 s) and needs the host runner's
build directory with esp32_bench_host in it (ESP32_HOST_RUNNER, as the backend's
tests use it); it is skipped, by name, without one.
"""

import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_bench_hardware as rbh  # noqa: E402

RUNNER_INC = HERE.parent / "main" / "bench_runner.inc"


def test_header_matches_the_firmware_struct():
    """IMAGE_HEADER packs the fields bench_runner.inc's Header declares, in
    its order, with its types."""
    text = RUNNER_INC.read_text()
    body = re.search(r"struct Header \{(.*?)\n\};", text, re.S).group(1)
    codes = {"uint32_t": "I", "uint64_t": "Q", "double": "d"}
    fmt = "<"
    for ctype, name, count in re.findall(r"^\s*(\w+)\s+(\w+)(?:\[(\d+)\])?;", body, re.M):
        fmt += f"{count}s" if ctype == "char" else codes[ctype]
    collapsed = re.sub(r"(I+|Q+)", lambda m: f"{len(m.group(0))}{m.group(0)[0]}"
                       if len(m.group(0)) > 1 else m.group(0), fmt)
    assert collapsed == rbh.IMAGE_HEADER.format == "<8s10Id6I2Q32s"
    assert struct.calcsize(rbh.IMAGE_HEADER.format) == 128
    assert "static_assert(sizeof(Header) == 128" in text


def test_fnv1a_vectors():
    assert rbh.fnv1a(b"") == 0x811C9DC5
    assert rbh.fnv1a(b"a") == 0xE40C292C
    assert rbh.fnv1a(b"foobar") == 0xBF9CF968


def test_input_is_exact_uniform_and_seeded():
    x = rbh.bench_input(1 << 16, 1)
    assert x.dtype == np.float32 and x.min() >= -1.0 and x.max() < 1.0
    # Every value is k * 2^-23 - 1 for an integer k < 2^24: exact in float32.
    k = (x.astype(np.float64) + 1.0) * 2.0 ** 23
    assert np.array_equal(k, np.round(k)) and k.max() < 2 ** 24
    assert abs(float(x.mean())) < 0.01 and 0.57 < float(x.std()) < 0.58   # 2 / sqrt(12)
    assert not np.array_equal(x, rbh.bench_input(1 << 16, 2))
    # The firmware's arithmetic, step by step, for a few elements.
    for i in (0, 1, 12345, 65535):
        h = (i + 1 * 0x9E3779B9) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 0x7FEB352D) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 0x846CA68B) & 0xFFFFFFFF
        h ^= h >> 16
        assert x[i] == np.float32((h >> 8) * 2.0 ** -23 - 1.0)


def test_image_round_trip():
    program = bytes(range(256)) * 7
    weights = b"\x01\x02\x03" * 1000
    out = np.linspace(-3, 3, 37, dtype=np.float32)
    labels = ["D delegate 0 (Esp32Backend): 1,000 MACs", "Q quantized_decomposed::"
              "quantize_per_tensor.out [1,3,8,8]", "P aten::add.out [1,8,4,4]", "M move"]
    image = rbh.pack_image(name="toy", program=program, weights=weights, input_numel=192,
                           input_seed=1, input_bits=0x1234, output=out, timed_runs=7,
                           stepped_runs=2, temp_bytes=4096, labels=labels, macs=1000,
                           macs_delegated=900)
    h = rbh.read_image(image)
    assert h["header_bytes"] % rbh.IMAGE_ALIGN == 0 and h["program"] == program
    assert h["labels"] == labels and h["instructions"] == 4 and h["name"] == "toy"
    assert h["weights_bytes"] == len(weights) and h["weights_bits"] == rbh.fnv1a(weights)
    assert h["output_numel"] == 37 and h["output_bits"] == rbh.fnv1a(out.tobytes())
    # The firmware's checksum: in order, in float64.
    want = 0.0
    for i, v in enumerate(out):
        want += float(v) * (i + 1)
    assert h["output_checksum"] == want
    assert (h["timed_runs"], h["stepped_runs"], h["temp_bytes"], h["macs"]) == (7, 2, 4096, 1000)


def _toy_image(timed_runs=2, stepped_runs=1):
    return rbh.pack_image(name="toy", program=b"\0" * 64, weights=b"\0" * 64,
                          input_numel=4, input_seed=1, input_bits=0,
                          output=np.zeros(4, np.float32), timed_runs=timed_runs,
                          stepped_runs=stepped_runs, temp_bytes=0,
                          labels=["M move"], macs=0, macs_delegated=0)


def test_run_counts_are_bounded_as_the_firmware_bounds_them():
    """The firmware times `timed_runs` executes into one allocation of an
    int64_t per run, whose byte count is a 32-bit size_t on the chip:
    0x20000002 runs would wrap 8 * n to 16 bytes, and the third run would
    write past them. bench_runner.inc refuses more than kMaxRuns runs of
    either kind, and pack_image packs no more (MAX_RUNS, the same number)."""
    for timed, stepped in ((0x20000002, 1), (1, 0x20000002), (1001, 1), (-1, 1)):
        with pytest.raises(ValueError, match="out of range"):
            _toy_image(timed, stepped)
    text = RUNNER_INC.read_text()
    limit = re.search(r"constexpr uint32_t kMaxRuns = (\d+);", text)
    assert limit is not None and int(limit.group(1)) == rbh.MAX_RUNS == 1000
    h = rbh.read_image(_toy_image(rbh.MAX_RUNS, rbh.MAX_RUNS))
    assert (h["timed_runs"], h["stepped_runs"]) == (rbh.MAX_RUNS, rbh.MAX_RUNS)


BOOT = """\
[mspi] psram din_mode 0x00000001 din_num 0x00000000 timing_cali 0x00000001 core_clk 0x00000002 sram_clk 0x00000003
BENCH_PROFILE backend -Os, runtime and portable kernels -Os; data cache 64 KB, 64 B lines; instruction cache 32 KB; CPU 240 MHz
BENCH_MODEL name=toy instructions=3 program=1000 weights=3000 macs=2000000 macs_delegated=2000000 timed_runs=3 stepped_runs=2
BENCH_MEM temp: 4096 B scratch wanted, 32768 B internal, 0 B PSRAM
BENCH_PHASE phase=flash load_us=900 apply_us=3 weight_copies=0 weight_resident=0 method_arena_peak=5000
BENCH_CHECK phase=flash result=EXACT input_bits=0000abcd/0000abcd output_bits=12345678/12345678 checksum=1.5/1.5 outputs=10 temp_peak=4096 large_temp_calls=0
BENCH_RUN phase=flash run=0 us={a}
BENCH_RUN phase=flash run=1 us={b}
BENCH_RUN phase=flash run=2 us={c}
BENCH_TIME phase=flash runs=3 min_us={lo} median_us={mid}.0 mean_us={mid}.0 max_us={hi} same_bits=3/3
BENCH_INSTR phase=flash i=0 us=100.0 Q quantized_decomposed::quantize_per_tensor.out [1,10]
BENCH_INSTR phase=flash i=1 us=1800.0 D delegate 0 (Esp32Backend): 2,000,000 MACs
BENCH_INSTR phase=flash i=2 us=100.0 P aten::add.out [1,10]
BENCH_SPLIT phase=flash stepped_runs=2 delegate_us=1800.0 qdq_us=100.0 portable_us=100.0 other_us=0.0 total_us=2000.0 same_bits=2/2
BENCH_PHASE phase=psram skipped: no PSRAM
BENCH_DONE name=toy
=== DONE ===
"""


def test_parse_and_summarize_boots():
    logs = [rbh.parse_log(BOOT.format(a=a, b=b, c=c, lo=min(a, b, c), mid=sorted((a, b, c))[1],
                                      hi=max(a, b, c)))
            for a, b, c in ((2000, 2100, 1990), (2050, 2040, 2060), (1980, 2500, 2010))]
    assert logs[0]["done"] and not logs[0]["errors"]
    assert logs[0]["mspi"] == "din_mode 0x00000001 din_num 0x00000000"
    assert logs[0]["phases"]["flash"]["runs"] == [2000, 2100, 1990]
    assert logs[0]["phases"]["psram"]["skipped"]
    s = rbh.summarize(logs)
    flash = s["phases"]["flash"]
    assert flash["boot_medians_ms"] == [2.0, 2.05, 2.01] and flash["median_ms"] == 2.01
    assert flash["min_ms"] == 1.98 and flash["exact"] and flash["same_bits"]
    assert flash["mmac_per_s"] == round(2_000_000 / 2010, 1)
    assert flash["split_share"] == {"delegate": 0.9, "qdq": 0.05, "portable": 0.05, "other": 0.0}
    assert flash["top"][0][2] == "D"
    assert "psram" not in s["phases"]    # skipped where there is no PSRAM
    text = rbh.format_summary(s)
    assert "median 2.01 ms" in text and "delegates 1.8 ms (90.0%)" in text


def test_a_diff_is_not_exact():
    bad = BOOT.replace("result=EXACT", "result=DIFF").format(
        a=1, b=1, c=1, lo=1, mid=1, hi=1)
    assert not rbh.summarize([rbh.parse_log(bad)])["phases"]["flash"]["exact"]


def test_a_call_off_its_planned_scratch_is_an_error():
    """The backend's one-time "planned scratch unusable" line (a dual-core
    firmware running a program exported without min_latency: every split
    call then took the temp allocator's scratch) is an error of the boot,
    so `run` fails instead of reporting a time measured off the planned
    configuration."""
    line = ("E (2345) esp32: planned scratch unusable (1 buffer(s), the first "
            "51584 B at 0x3fc9a000; the layout needs 55440 B from a 16-byte "
            "boundary): using the temp allocator")
    boot = BOOT.format(a=1, b=1, c=1, lo=1, mid=1, hi=1).replace(
        "BENCH_PHASE phase=flash", line + "\nBENCH_PHASE phase=flash")
    log = rbh.parse_log(boot)
    assert log["errors"] == [line]
    assert rbh.summarize([log])["errors"] == [line]


def test_the_scratch_counters_are_not_an_error():
    """The backend's counters after the bench (the firmware's
    esp32_backend_report, and the host bench's) label their fallback count
    "(planned scratch unusable)": a count, zero here, not the one-time line,
    so a boot that printed them is not failed for it (taking the label for the
    line would fail every model, the exact ones included)."""
    report = ("esp32 scratch: 56 call(s) on planned scratch, 0 on the temp "
              "allocator (none planned), 0 fallback(s) (planned scratch unusable)")
    boot = BOOT.format(a=1, b=1, c=1, lo=1, mid=1, hi=1) + report + "\n"
    log = rbh.parse_log(boot)
    assert log["errors"] == []
    assert rbh.summarize([log])["errors"] == []


def test_a_conv_boot_needs_both_self_tests_passing():
    """Where the conv path ran (a "convfast" count above 0), both of its boot
    self-tests must say PASS: on a FAIL its convs take the code the kernels
    replace, with the same bits at that code's speed, so only these lines show
    it. A boot without the path needs neither."""
    boot = BOOT.format(a=1, b=1, c=1, lo=1, mid=1, hi=1)
    ran = "esp32 convfast: 9 conv op(s) on the static int8 path, 72000 output(s)\n"
    dw = ("esp32: PIE depthwise 16-lane selftest: PASS (QACC lanes 0 1 2 3 4 5 6 7 "
          "8 9 10 11 12 13 14 15)\n")
    kernels = ("esp32: PIE conv kernels selftest: fused dot and epilogue PASS, "
               "transpose PASS, depthwise planes PASS\n")
    long_rows = "esp32: PIE long-row dot selftest: PASS\n"
    log = rbh.parse_log(dw + long_rows + kernels + boot + ran)
    assert log["errors"] == []
    assert log["selftests"] == {"depthwise": "PASS", "fused dot and epilogue": "PASS",
                                "transpose": "PASS", "long-row dot": "PASS",
                                "depthwise planes": "PASS"}
    failed = kernels.replace("transpose PASS", "transpose FAIL (element copies)")
    assert rbh.parse_log(dw + long_rows + failed + boot + ran)["errors"] == [
        "PIE transpose selftest: FAIL"]
    long_failed = long_rows.replace("PASS", "FAIL (dots then epilogue)")
    assert rbh.parse_log(dw + long_failed + kernels + boot + ran)["errors"] == [
        "PIE long-row dot selftest: FAIL"]
    failed = kernels.replace("planes PASS", "planes FAIL (scalar loop)")
    assert rbh.parse_log(dw + long_rows + failed + boot + ran)["errors"] == [
        "PIE depthwise planes selftest: FAIL"]
    assert rbh.parse_log(dw + boot + ran)["errors"] == [
        "PIE fused dot and epilogue selftest: never printed",
        "PIE transpose selftest: never printed",
        "PIE long-row dot selftest: never printed",
        "PIE depthwise planes selftest: never printed"]
    assert rbh.summarize([rbh.parse_log(boot + ran)])["errors"] == [
        "PIE depthwise selftest: never printed",
        "PIE fused dot and epilogue selftest: never printed",
        "PIE transpose selftest: never printed",
        "PIE long-row dot selftest: never printed",
        "PIE depthwise planes selftest: never printed"]
    assert rbh.parse_log(boot)["errors"] == []


def test_a_firmware_is_held_to_the_self_tests_it_holds(tmp_path):
    """run holds a firmware to each conv self-test whose line its application
    holds: this tree's to all five; one built without the long-row and
    depthwise plane kernels to its three; one with only the 16-lane depthwise
    kernel's line (an A/B baseline) to that one, so its boots do not fail for
    a line it cannot print."""
    app = tmp_path / "et_runner.bin"
    app.write_bytes(b"\0esp32: PIE depthwise 16-lane selftest: %s (QACC lanes\0"
                    b"esp32: PIE long-row dot selftest: %s\n\0"
                    b"esp32: PIE conv kernels selftest: fused dot and epilogue %s, "
                    b"transpose %s, depthwise planes %s\n\0")
    assert rbh.firmware_selftests(tmp_path) == rbh.SELFTESTS
    # A firmware without the long-row and depthwise plane kernels is held to its three.
    app.write_bytes(b"\0esp32: PIE depthwise 16-lane selftest: %s (QACC lanes\0"
                    b"esp32: PIE conv kernels selftest: fused dot and epilogue %s, "
                    b"transpose %s\n\0")
    assert rbh.firmware_selftests(tmp_path) == ("depthwise", "fused dot and epilogue",
                                                "transpose")
    app.write_bytes(b"\0esp32: PIE depthwise 16-lane selftest: %s (QACC lanes\0")
    assert rbh.firmware_selftests(tmp_path) == ("depthwise",)
    boot = BOOT.format(a=1, b=1, c=1, lo=1, mid=1, hi=1)
    old = ("esp32: PIE depthwise 16-lane selftest: PASS (QACC lanes 0 1 2 3 4 5 6 7 "
           "8 9 10 11 12 13 14 15)\n" + boot
           + "esp32 convfast: 9 conv op(s) on the static int8 path, 72000 output(s)\n")
    assert rbh.parse_log(old, rbh.firmware_selftests(tmp_path))["errors"] == []
    assert len(rbh.parse_log(old)["errors"]) == 4
    app.write_bytes(b"\0no conv path\0")
    assert rbh.firmware_selftests(tmp_path) == ()


def _conv_kernel_calls(log):
    """The host runner's (and esp32_bench_host's) calls into the conv path's
    three kernels: their outputs equal the C loops they replace, so only these
    counts show they ran."""
    found = re.search(r"host conv kernels: qacc_lane_codes (\d+), "
                      r"dot_rows_codes (\d+), transpose16 (\d+) call", log)
    assert found is not None, log[-2000:]
    return dict(zip(("qacc_lane_codes", "dot_rows_codes", "transpose16"),
                    map(int, found.groups())))


def _fake_model_dir(root, name, min_latency):
    """A model directory as export_bench.py writes it, the files only as big
    as the layout check needs."""
    import json

    d = root / name
    d.mkdir()
    (d / rbh.IMAGE_FILE).write_bytes(_toy_image())
    (d / rbh.WEIGHTS_FILE).write_bytes(b"\0" * 64)
    (d / rbh.OPS_FILE).write_text("aten::add.out\n")
    record = {"name": name, "macs": 0}
    if min_latency is not None:
        record["min_latency"] = min_latency
    (d / rbh.MODEL_FILE).write_text(json.dumps(record))
    return d


@pytest.mark.parametrize("dual_core,exported", [(True, False), (True, None),
                                                (False, True)])
def test_build_takes_only_images_exported_for_its_posture(tmp_path, monkeypatch,
                                                          dual_core, exported):
    """A dual-core bench firmware takes only images export_bench.py
    --min-latency exported (an image without the field was exported without
    it), and a single-core one only images exported without it:
    refused before anything is generated or built."""
    def no_build(*args, **kwargs):
        raise AssertionError("a mismatched build went past the check")

    monkeypatch.setattr(rbh, "_board", lambda: type("B", (), {"_run": no_build,
                                                              "idf": no_build})())
    d = _fake_model_dir(tmp_path, "toy", exported)
    with pytest.raises(SystemExit, match="--min-latency"):
        rbh.build([str(d)], str(tmp_path / "build"), dual_core=dual_core)


# --- one model, end to end, on the host ------------------------------------------------

def _host_bench():
    runner = os.environ.get("ESP32_HOST_RUNNER")
    if runner:
        return Path(runner).parent / "esp32_bench_host"
    return HERE.parents[3] / "cmake-out-espressif" / "esp32_bench_host"


@pytest.mark.skipif(not _host_bench().is_file(),
                    reason="needs the host runner's build with esp32_bench_host "
                           "(backends/esp32s3/test/host_runner)")
def test_the_bench_runner_refuses_an_unbounded_run_count(tmp_path):
    """The firmware's parser (bench_runner.inc, built into esp32_bench_host)
    refuses an image asking for 0x20000002 timed runs before anything loads,
    and esp32_bench_host refuses the same count as an override."""
    import subprocess

    weights = tmp_path / "weights.ptd"
    weights.write_bytes(b"\0" * 64)
    image = bytearray(_toy_image())
    struct.pack_into("<I", image, struct.calcsize("<8s10Id"), 0x20000002)
    assert rbh.read_image(bytes(image))["timed_runs"] == 0x20000002
    patched = tmp_path / "patched.bin"
    patched.write_bytes(bytes(image))
    done = subprocess.run([str(_host_bench()), "--image", str(patched), "--ptd",
                           str(weights)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 1, done.stdout + done.stderr
    assert ("BENCH_ERROR bench image asks for 536870914 timed and 1 stepped runs, "
            f"at most {rbh.MAX_RUNS} each") in done.stdout, done.stdout
    valid = tmp_path / "valid.bin"
    valid.write_bytes(_toy_image())
    done = subprocess.run([str(_host_bench()), "--image", str(valid), "--ptd",
                           str(weights), "--timed-runs", str(0x20000002)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2, done.stdout + done.stderr
    assert f"take at most {rbh.MAX_RUNS} runs" in done.stderr, done.stderr


@pytest.mark.skipif(not _host_bench().is_file(),
                    reason="needs the host runner's build with esp32_bench_host "
                           "(backends/esp32s3/test/host_runner)")
def test_ds_cnn_exports_and_runs_exact_through_the_bench_runner(tmp_path):
    import export_bench

    record = export_bench.export("ds_cnn", tmp_path, timed_runs=2, stepped_runs=1,
                                 host_bench=str(_host_bench()))
    d = tmp_path / "ds_cnn"
    h = rbh.read_image((d / rbh.IMAGE_FILE).read_bytes())
    assert h["program"] == (d / "bench.pte").read_bytes()
    assert h["weights_bytes"] == len((d / rbh.WEIGHTS_FILE).read_bytes())
    # Every MAC delegated; one label per instruction, the delegates' MACs summing to
    # the model's; the quantize at the input and the pads between delegates labelled.
    assert record["macs"] == record["macs_delegated"] > 0
    assert sum(record["delegate_macs"]) == record["macs"]
    labels = record["instructions"]
    assert h["labels"] == labels and h["instructions"] == len(labels)
    assert sum(l.startswith("D ") for l in labels) == record["delegates"] > 0
    assert any(l.startswith("P aten::constant_pad_nd") for l in labels)
    # The host run of the image: exact both ways, the psram measurement's
    # weights placed by the apply call, every instruction timed.
    for phase in ("flash", "psram"):
        assert record["host_bench"][phase]["check"]["result"] == "EXACT"
    assert record["host_bench"]["psram"]["load"]["weight_copies"] > 0
    assert record["host_bench"]["flash"]["load"]["weight_copies"] == 0
    text = (d / "host_bench.log").read_text()
    log = rbh.parse_log(text)
    assert len(log["phases"]["flash"]["instr"]) == len(labels)
    # The conv path's kernels served its convs: both self-tests passed (or
    # parse_log would hold an error), and each kernel ran more often than in a
    # --force-scalar run of the same image, whose only calls are the boot
    # self-tests' (the bits are the same either way: test_conv_fast).
    assert log["errors"] == [] and log["selftests"]["transpose"] == "PASS"
    import subprocess

    scalar = subprocess.run([str(_host_bench()), "--image", str(d / rbh.IMAGE_FILE),
                             "--ptd", str(d / rbh.WEIGHTS_FILE), "--force-scalar",
                             "--timed-runs", "1", "--stepped-runs", "1"],
                            capture_output=True, text=True, timeout=120)
    assert scalar.returncode == 0, scalar.stdout[-2000:] + scalar.stderr
    on, off = _conv_kernel_calls(text), _conv_kernel_calls(scalar.stdout)
    for kernel, calls in on.items():
        assert calls > off[kernel], f"{kernel} never ran: {on} vs {off}"
    # The element-wise row copy served DS-CNN's input pad (ops/elementwise.h):
    # it gives the portable kernel's bytes, so only its count shows it ran.
    pad = re.search(r"^constant_pad_nd\.out elementwise dispatch: rows (\d+), "
                    r"fallback (\d+)$", text, re.M)
    assert pad and int(pad[1]) > 0 and pad[2] == "0", text[-2000:]
    # The board's flash layout takes it in the default split.
    model = rbh.load_model(d)
    assert rbh.model_layout(model)["default"]
    assert "quantized_decomposed::quantize_per_tensor.out" in model["ops"]


@pytest.mark.skipif(not _host_bench().is_file(),
                    reason="needs the host runner's build with esp32_bench_host "
                           "(backends/esp32s3/test/host_runner)")
def test_resnet8_for_two_cores_plans_the_same_scratch(tmp_path):
    """No convolution splits across the cores (the backend README's Known
    limits), so a dual-core firmware reserves no scratch for a split worker's
    part of a conv: ResNet-8 exported for one core, run with min_latency
    turned on at run time (a dual-core firmware), finds its planned scratch
    big enough, as the --min-latency export does; every call runs on planned
    scratch, both give the single-core run's bits (export_bench checks them
    against --single-core) and plan the same delegate temp."""
    import export_bench

    single = export_bench.export("resnet8", tmp_path / "one", timed_runs=2,
                                 stepped_runs=1, host_bench=str(_host_bench()))
    dual = export_bench.export("resnet8", tmp_path / "two", timed_runs=2,
                               stepped_runs=1, host_bench=str(_host_bench()),
                               min_latency=True)
    assert not single["min_latency"] and dual["min_latency"]
    for record in (single, dual):
        assert record["split_scratch"]["unusable"] == 0, record["split_scratch"]
        assert record["split_scratch"]["planned"] > 0, record["split_scratch"]
    assert dual["output_bits"] == single["output_bits"]
    assert dual["temp"] == single["temp"]
    assert rbh.load_model(tmp_path / "two" / "resnet8")["min_latency"] is True


# --- the constants stay read-only -------------------------------------------------------

# The postures the bench firmware's code can take: one core; min_latency (the
# dual-core firmware's: projections split, every conv on one core); and that
# with the scalar loops, with every conv on the general loops, and with the
# portable q/dq/avg_pool2d kernels.
READ_ONLY_POSTURES = (
    ("--single-core",),
    ("--min-latency",),
    ("--min-latency", "--force-scalar"),
    ("--min-latency", "--no-conv-fast"),
    ("--min-latency", "--no-qdq-fast"),
)


def _bench_read_only(model_dir, flags, runs=3):
    """The host bench on an exported image with the image and its weights in
    pages mapped read-only (--read-only), as the board maps partitions 0x43
    and 0x44 from flash: a store into the program or a weight faults here
    (BENCH_FAULT names the mapping and the offset), where the board's flash
    rejects it (a Cache error panic) or drops it. (log parsed, the conv ops
    it split across the cores, the projection rows it split)."""
    import subprocess

    done = subprocess.run(
        [str(_host_bench()), "--image", str(model_dir / rbh.IMAGE_FILE),
         "--ptd", str(model_dir / rbh.WEIGHTS_FILE), "--timed-runs", str(runs),
         "--stepped-runs", "1", "--read-only", *flags],
        capture_output=True, text=True, timeout=600)
    log = done.stdout + done.stderr
    assert "BENCH_FAULT" not in log and done.returncode == 0, (flags, log[-3000:])
    assert "BENCH_READONLY" in log, log[-3000:]
    split = re.search(r"^esp32 convsplit: (\d+) conv op", log, re.M)
    rows = re.search(r"^esp32 split: (\d+) column-split addmm row\(s\) forked",
                     log, re.M)
    assert split is not None and rows is not None, log[-3000:]
    return rbh.parse_log(log), int(split.group(1)), int(rows.group(1))


@pytest.mark.skipif(not _host_bench().is_file(),
                    reason="needs the host runner's build with esp32_bench_host "
                           "(backends/esp32s3/test/host_runner)")
def test_bench_models_never_store_into_their_constants(tmp_path):
    """No path the bench firmware runs stores into a program or a weight.
    The flash measurement reads every weight from the flash mapping in place,
    and on the board a store there is rejected or lost: the release line's
    first CNN bench came out wrong or panicking in its flash measurements, and
    in some psram ones (a two-core convolution split, since turned off: the
    backend README's Known limits).
    Here the image and the weights are mapped read-only, every posture of the
    bench firmware runs both measurements, and both give the exporter's bits
    on every run, with no fault. No posture splits a convolution; min_latency
    still splits MobileNetV2-96's 1,000-column classifier. DS-CNN (depthwise
    and pointwise on the path), ResNet-8 (3x3 and 1x1 dense convs) and
    MobileNetV2-96, exported for the dual-core firmware as the board's images
    are."""
    import export_bench

    runs = 3
    for name in ("ds_cnn", "resnet8", "mobilenet_v2_96"):
        record = export_bench.export(name, tmp_path, timed_runs=2, stepped_runs=1,
                                     host_bench=str(_host_bench()), min_latency=True)
        d = tmp_path / name
        for flags in READ_ONLY_POSTURES:
            parsed, splits, rows = _bench_read_only(d, flags, runs)
            phases = parsed["phases"]
            for phase in ("flash", "psram"):
                check = phases[phase]["check"]
                assert check["result"] == "EXACT", (name, flags, phase, check)
                assert check["output_bits"].split("/")[0] == record["output_bits"]
                assert phases[phase]["time"]["same_bits"] == f"{runs}/{runs}", (
                    name, flags, phase, phases[phase]["time"])
            # The flash measurement read the mapping, the psram one copies.
            assert phases["flash"]["load"]["weight_copies"] == 0
            assert phases["psram"]["load"]["weight_copies"] > 0
            assert splits == 0, (name, flags, splits)
            # Only MobileNetV2-96 has a projection wide enough to split.
            want_rows = name == "mobilenet_v2_96" and flags[0] == "--min-latency"
            assert (rows > 0) == want_rows, (name, flags, rows)
