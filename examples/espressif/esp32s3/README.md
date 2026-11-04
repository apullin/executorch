# ESP32-S3 example project

The ESP-IDF firmware that runs ExecuTorch programs delegated to the Espressif
backend (`backends/esp32s3`) on an ESP32-S3, the scripts that check it in
QEMU and on a board, and a story demo built on it.

- `main/`: the firmware. The parity build embeds the test programs and prints
  each one's checksums and output bits; the story build (`ET_STORY`) runs the
  story model from flash partitions; the bench build (`ET_BENCH_MODEL`) times
  one model from flash partitions.
- `components/executorch`: ExecuTorch's runtime and a selective kernel build,
  from the ExecuTorch tree this project sits in. `components/papercolor`: the
  M5Stack PaperColor e-paper driver and LVGL screens the story demo uses.
- `scripts/`: the selective build (`gen_kernels.py`), the flash layout, the
  board helpers and the bench (`export_bench.py`, `run_bench_hardware.py`).
  `story/`: the story model, its exporter and its board runner.

## Prerequisites

- ESP-IDF v5.5.5, activated (`. $IDF_PATH/export.sh`), with its QEMU for the
  parity run (`python $IDF_PATH/tools/idf_tools.py install qemu-xtensa`).
- ExecuTorch's Python environment (the `executorch` package with torch and
  torchao), from which the scripts below run.
- LVGL 9.5.0 for the display component: the ESP Component Registry's
  `lvgl/lvgl`, which the component manager fetches into `managed_components/`
  at the first build. To build offline, point `ESPRESSIF_LVGL_DIR` at an LVGL
  v9.5.0 checkout: the project then uses it and leaves the component manager
  off.
- The FlatBuffers headers of the ExecuTorch tree's `third-party/flatbuffers`
  submodule; `ESPRESSIF_FLATBUFFERS_INCLUDE` names another copy.

The commands run from the root of the ExecuTorch tree unless they say
otherwise.

## Parity in QEMU

```bash
python backends/esp32s3/test/parity/run_parity.py
```

exports every parity program with a fixed seed, records its references,
derives the selective kernel build from the programs (`scripts/gen_kernels.py`),
builds the firmware (`idf.py -C examples/espressif/esp32s3 build`), runs it in
QEMU and compares each row. A row is held either to the output bits of a
device mirror (the device's arithmetic, reproduced in Python bit for bit), to
one code of its output grid, or to a stated tolerance; the run ends with
`ALL PASS` only if every row passes and the firmware's conformance checks
report no mismatch: the portable fast paths against the portable kernels, the
fast LayerNorm rows, the activation tables (`quant_lut`, `lut16`), float
division and square root against exact rounding, and the rule for quantising
NaN and the infinities. `--resume` reuses the last complete export; `--log
FILE` keeps the device log. Every program is exported with planned
delegate scratch (`ESP32_PLAN_SCRATCH=0` in the environment exports them
without), and the backend's report line `esp32 scratch:` counts the calls
that ran on it. `--memory-plan none` exports every program with ExecuTorch's own
memory planner instead of the backend's (`memory_planning.py`: in-place
reuse, operator order and a hill-climbing allocator, the default), and
`--fast-layernorm` builds the firmware with the opt-in fast LayerNorm (a
reciprocal of the standard deviation times the input, registered in place of
the portable `native_layer_norm`) and runs the whole suite with it.

## On a board

```bash
python backends/esp32s3/test/parity/run_hardware.py --board papercolor --serial SERIAL
```

runs the same suite on silicon (an ESP32-S3 with 16 MB of flash and 8 MB of
octal PSRAM; `--board nano` for an Arduino Nano ESP32). It finds the board by
its USB serial number (`python -m serial.tools.list_ports -v` lists it; there
is no default), builds with the hardware configuration
(`sdkconfig.hardware.defaults`), flashes, captures the log through a READY/GO
handshake and compares every row. On the board it also runs the RMSNorm
kernel's `rsqrt0.s` path, which QEMU cannot execute, the rows that need PSRAM
(ResNet-8's), and measures memory read bandwidth. `--resume` reuses
`run_parity.py`'s export; `--cache-defaults` builds with the story's caches
(`sdkconfig.cache.defaults`) in a directory of its own; `--replay LOG`
compares a saved log again with no board; `--rates LOG` reads only the
bandwidth and weight-tier rate rows of a saved log (a parity log, or a story
build's with `sdkconfig.bench.defaults`) and prints them as `target/chips.py`
entries. A parity build configured with `-DET_PARITY_TIER_BYTES=N` runs every
row under an N-byte weight-tier budget, and `-DET_PARITY_TIER_RANK=1` ranks
it; every row must still be exact.

## Story demo

A Glimmer language model with Per-Layer Embeddings (int4 weights, an 8,192-token
vocabulary) writes short stories on an ESP32-S3 with 16 MB of flash and 8 MB of
PSRAM, and shows them on an M5Stack PaperColor panel. The model is one program
with two methods, `prefill` and `decode`, that share their KV cache; its weights
stay in flash, a few megabytes of them copied into PSRAM at load.

Neither the trained checkpoint nor its tokenizer is published: bring your own,
in the form below.

### The checkpoint contract

`story/export_story_flash.py` reads a `torch.save` file that holds a dict:

- `cfg`, the training configuration, with
  - `architecture: "glimmer"` and `arm: "ple"`: a Glimmer
    (`story/glimmer_model.py`) with Per-Layer Embeddings;
  - its shape: `vocab_size`, `seq_len`, `d_model`, `n_layers`, `n_heads`,
    `n_kv_heads`, `ffn_hidden`, `ple_dim`, `glimmer_window` and
    `final_logit_softcapping`;
  - `qk_scale_factor`, `output_multiplier`, `glimmer_rope_theta`,
    `rms_norm_eps` and `post_norm_eps` at `glimmer_model.Config`'s defaults
    (3.87, 0.19611613513818404, 500000, 1e-5 and 1e-8);
  - `loops` 1, or no `loops` at all;
- `quantization`: `{"dense_bits": 4, "table_bits": 4, "group": 128}`, the int4
  format the model was trained for (int4 dense weights and int4 embedding
  tables, in groups of 128);
- `state`: the float `state_dict` of `glimmer_model.TinyLM` for that `cfg`,
  which the exporter loads strictly.

The tokenizer is a Hugging Face `tokenizers` JSON file: a byte-level BPE (the
`ByteLevel` pre-tokenizer, without a prefix space) with an `<|endoftext|>`
token, whose vocabulary (added tokens included) has the checkpoint's
`vocab_size`. The exported context (`--context`, 256 by default) may not exceed
the checkpoint's `seq_len`.

### Export, build, flash

From this directory:

    python story/run_story_hardware.py --checkpoint /path/to/glimmer-ple-story.pt \
        --tokenizer /path/to/tokenizer.json --serial SERIAL \
        --dual-core --stage-mb 6 --temp-mb 1 --backend-o2

This exports the model into `main/` with `story/export_story_flash.py`
(`ple.pte`, `ple_weights.ptd`, the golden in `story_inputs.h`, the
detokenizer's `story_vocab.h`, the prompt encoder's `story_bpe_tables.h` and
`story_artifact.json`), derives the selective kernel build from the program
(`scripts/gen_kernels.py`), builds the firmware in `build_story_hw/`, flashes
the board, and checks the device's greedy continuation of the exporter's prompt
against the host's tokens and logit checksums (the `STORY_GOLDEN` line). The
line's `digest=` is a 64-bit FNV-1a of every prefill block's and decode step's
raw logits, in the order they ran (`main/story_digest.h`), computed outside the
timed steps: two runs, or two builds, that print the same digest computed the
same bits. The next line, `STORY_COLD`, gives the cold start: the program's
load, then each method's first load and the ranked weight placement after it,
in microseconds.
`--reuse-artifact` builds from what is already in `main/`, `--build-only` stops
before flashing, and `--request "<prompt>"` asks the board for a story.

The KV cache (146,944 B, read every token) takes internal SRAM first. The
first eight flags below place the rest. They are data in the image, not code,
so two builds that differ only in them run the same code at the same
addresses, and a board A/B of one sees only where the bytes are. The last
four change the code that is built:

| Flag | Default | Effect |
|---|---|---|
| `--stage-mb MB` or `auto` | 0 | the weight-tier budget: MB of weights copied into PSRAM (0 reads them all from flash). `auto` copies every weight that fits while `--stage-reserve-mb` of PSRAM stays free, ranked |
| `--stage-reserve-mb` | 0.5 | with `auto`, the PSRAM left for what is allocated after load |
| `--tier-rank`, `--no-tier-rank` | on | ranked placement: the copies are made after each load, the weights worth most first, instead of inside it in load order. The default is `main/story_runner.inc`'s `ET_STORY_TIER_RANK` |
| `--temp-mb` | 2 | the PSRAM tier of the methods' temp allocator, behind a 32 KB internal one; 0 means none |
| `--arena-kb` | 128 | the method arena both methods use, in PSRAM |
| `--internal-floor-kb` | 40 | internal SRAM kept free after load for what is allocated later; a placement that would leave less goes to PSRAM or stays on flash |
| `--decode-act-internal`, `--no-decode-act-internal` | on | decode's activations and planned scratch (34,576 B) in a buffer of their own in internal SRAM, within the floor, instead of sharing prefill's in PSRAM |
| `--hot-constants`, `--no-hot-constants` | on | the small constants outside the delegate (the RMSNorm weights and two mask tables, up to 45,568 B) copied into internal SRAM on first use, as far as the floor allows after decode's activations |
| `--dual-core` | off | every projection split across both cores (the backend's `min_latency`) |
| `--backend-o2` | off | the delegate at -O2 (the operator library is at -O2 by default) |
| `--sdkconfig-extra FILE` | | one more sdkconfig defaults file, built in `build_story_hw_<stem>`: `sdkconfig.hotiram.defaults` puts the per-token code in internal RAM (give `--no-decode-act-internal` with it, as both want the same SRAM), `sdkconfig.split64.defaults` raises the dual-core split to 64 columns, `sdkconfig.bench.defaults` adds the bandwidth and weight-tier rate rows at boot |
| `--default-caches` | off | IDF's default caches instead of `sdkconfig.cache.defaults` |

The firmware prints free internal SRAM, its largest block and the lowest
since boot after load and after the first story (`[mem]` lines, each `held`
or `BROKEN` against the floor), the weight tiers (`[tier]`), and the backend's
report, with where each delegate call's scratch came from (`esp32 scratch:`).
`--help` lists the rest: the sampling (`--temperature`, `--top-k`, `--seed`,
`--max-new`), the display, and the serial link.
`--op-timing` builds the firmware with ExecuTorch's EventTracer, which prints
decode's time per instruction (`ET_OP_PROFILE` lines): an instrumented build,
not a speed figure. `story/story_op_profile.py LOG` names each instruction's
operator or delegate from the exported program (`--pte`, `main/ple.pte` by
default) and prints the time per operator per token.

`story/host_story_golden.py --emit-dir main` runs the same golden check on the
build machine, through the delegate's own C++ (the host runner,
`backends/esp32s3/test/host_runner`).

The default golden runs neither prefill nor the local layers' ring past its
wrap: its prompt is 19 tokens (prefill needs 33) and its 24 steps end at
position 41 (the 64-slot ring first wraps at 64). `--golden long` checks both,
for a release: a 75-token prompt runs two prefill chunks (positions 0-31, then
32-63 against the first chunk's keys), then decode from position 64, whose
first step overwrites ring slot 0, and 24 greedy steps from position 74. It is
gated on the token choices and on the value of the first prefill block
(`early_steps=1`, 1e-5): the reference runs prefill's softmax, P@V and RoPE as
the chip does (newlib's expf, the fused multiply-adds of `shim_bmm_out` and
`shim_rope`). The second block's and the decode steps' drift is printed, not
gated: the reference does not mirror everything the chip rounds (its RMSNorm,
for one), a one-ulp difference can flip an int8 code downstream, and with this
prompt the chip's second block sits 3e-3 off. It is an export setting, so it
replaces `main/`'s export; the program and weights stay byte for byte the
default export's, and only `story_inputs.h` changes. Re-export without it
before a run that reuses the artifact.

## Timing a model

```bash
python examples/espressif/esp32s3/scripts/export_bench.py ds_cnn mobilenet_v2_96 --out /tmp/bench
python examples/espressif/esp32s3/scripts/run_bench_hardware.py build /tmp/bench/* --build /tmp/bench/fw
python examples/espressif/esp32s3/scripts/run_bench_hardware.py run /tmp/bench/ds_cnn \
    --build /tmp/bench/fw --serial SERIAL --boots 3
```

times models on the board, one at a time. The exporter quantises a census
model (`backends/esp32s3/test/quant_census/models.py`: the MLPerf Tiny models,
MobileNetV2 and MobileNetV3-Small among them) with Arm's static int8 recipe,
lowers it with the backend's defaults and its weights in a named-data file,
records the host runner's output bits as the reference, and writes the model's
two flash partitions: a bench image (a header, a label for every instruction
and the program) and the weights. It then runs the image through the
firmware's own bench runner built for the host (`esp32_bench_host`, beside the
host runner). `build` makes one lean firmware (`main/bench_runner.inc`) whose
selective build registers the operators of every model given; `run` flashes
it with one model's partitions and captures `--boots` boots. `build
--no-conv-fast` leaves the backend's static int8 convolution path out (every
convolution then takes the general loops, with the same output bits), for an
A/B on one tree; `build --no-qdq-fast` registers ExecuTorch's portable
quantize, dequantize and avg_pool2d kernels in place of the backend's exact
fast paths for them (`ops/qdq_pool.h`; same output bits), the same way;
`build --dual-core` sets `min_latency` at boot, which splits projections'
columns across both cores (no convolution splits: the backend README's Known
limits), and takes only images `export_bench.py --min-latency` exported
(each delegate compiled for `min_latency`), as a single-core `build` takes
only images exported without it; `build --backend-timing` adds
the backend's timing by op shape, reported after each measurement (below),
from which `summary` gives cycles an output for every shape. The firmware
sizes the method's arena by loading the method
once into a PSRAM probe before it measures, then places that size internal
first. With the convolution path built, a boot first runs the backend's
micro-benchmarks (`BENCH_MICRO` lines, in core cycles): `dot_i8_s3` a call at
five lengths, the path's epilogues an output, a fork and join through the split
worker (the projection split's hand-over), and a DS-CNN pointwise and
depthwise layer on one core, with PIE and without, the second run's outputs
compared byte by byte with the first's.

Each boot measures the model twice, on the same buffers and at the same PSRAM
timing point: with every weight read from flash, then with every weight PSRAM
holds, placed by the backend's ranked placement (`weight_tier_apply`) before
any timed run. Each measurement checks the output bits against the host
runner's, times its executes (a fresh input each, every output checked), and
splits a few more by instruction with `Method::step`: each delegate, each
portable operator, each quantize and dequantize between them. The log states
the build's profile (`--backend-o2`, the caches, the runtime's Kconfig
optimisation) and the PSRAM timing point, and `run_bench_hardware.py summary`
reads saved logs.

## Performance

Measured on an M5Stack PaperColor: ESP32-S3 (N16R8) at 240 MHz, 8 MB octal
PSRAM at 80 MHz, 16 MB quad flash at 80 MHz, ESP-IDF v5.5.5.

**Story decode.** The story model at d_model 256, 8 layers, 8 query heads and
1 KV head, a 64-token sliding window, FFN 768 and 256-wide Per-Layer
Embeddings, with int4 weights and the full 8,192-token head: a 170,100 B
program (162,932 B without planned scratch) and 15,033,216 B of weights.
Run by the command above: both cores, 5,205,198 B of weights in 82 copies
staged in PSRAM under the 6 MiB budget, the delegate at -O2, the 64 KB data
cache with 64-byte lines and the 32 KB instruction cache of
`sdkconfig.cache.defaults`, display on. With the operator library at -O2,
decode's activations in internal SRAM and as many of its small constants as
the floor leaves room for, decode takes 85.82 ms per token: the
`STORY_GOLDEN` line's decode time per step over its 42 greedy decode steps,
the median of the boots that tuned PSRAM to the same read-timing point (the
firmware prints it as `[mspi]`), on builds without `--backend-timing`. The
figure is for a build without ranked placement and planned scratch; at this
budget each of those defaults measured within 0.1% of it. Decode streams the int4 weights once
per token, so its speed depends on the model's shape and not on its trained
values.

Each speed and placement setting was measured the same way, against the same
image without it. The operator library at -O2 saves 1.74% of a decode step,
decode's activations in internal SRAM 1.81%, and the small constants in
internal SRAM 1.35% with all 45,568 B of them placed; by default the
activations go first and the constants take what the floor leaves. The
per-token code in internal RAM (`sdkconfig.hotiram.defaults`) saved 1.34% but
needs the internal SRAM the activations use, and splitting only projections
of 64 columns or more across the cores (`sdkconfig.split64.defaults`) gained
nothing (+0.14%). In place of decode's activations (with
`--no-decode-act-internal`), the per-token code in internal RAM measured
0.33% faster than the defaults, too little to change them. Ranked weight
placement at a 4.5 MiB budget took decode from 110.10 to 98.85 ms per token
against load order, and at the 6 MiB budget it costs 0.03%, so the story
ranks by default. Planned delegate scratch measured +0.09%.

**CNN bench.** The bench models under Arm's a8w8 recipe
(Bench flags, below), timed by `run_bench_hardware.py` with their weights
copied into PSRAM, on one core with the story's caches: the median over 3
boots, in ms per inference, every output bit for bit the host runner's. The
table's build has none of the backend's quantize, dequantize and avg_pool2d
fast paths, its integer epilogue or its PIE depthwise path, each of which
keeps every output bit. With them and the PIE kernels, on one core with the
backend at -O2 (flash, then PSRAM, in ms; every output exact and every run the
same over 3 boots): DS-CNN 10.29 (10.29), MobileNet-v1 0.25 57.3 (52.6),
ResNet-8 69.5 (67.5), MobileNetV2-96 340 (265), MobileNetV2-224 1,390 (1,249),
MobileNetV3-Small 526 (472). Convolutions run on one core: split across both, they gave
wrong outputs and cache-error panics on the board (the backend README's
Known limits).

| Model | General conv loops, -Os | Static int8 conv path, -Os | Static int8 conv path, -O2 |
|---|---|---|---|
| `ds_cnn` | 354.0 | 46.2 | 44.0 |
| `mobilenet_v1_025` (VWW) | 1,137.3 | 200.8 | 172.5 |
| `resnet8` | 599.5 | 219.0 | 159.6 |
| `mobilenet_v2_96` | 5,300.1 | 729.1 | 711.2 |

The first column is the same tree built with `build --no-conv-fast`, so every
convolution takes the general loops. -O2 is `--backend-o2` with ExecuTorch's
runtime and portable kernels at -O2 as well (`CONFIG_ET_PORTABLE_O2=y`,
through `--sdkconfig-extra`). The static path is built at -O2 in every
profile, so the delegates' time hardly moves between the last two columns;
what -O2 speeds up is the portable quantize and dequantize kernels between
delegates, which the bench's split by instruction puts at 141.9 ms of
ResNet-8's 218.9 at -Os and 83.4 of 159.4 at -O2. With the weights read from
flash the -O2 times are 44.0, 177.0, 161.1 and 782.0 ms.

**Memory read bandwidth**, from `run_hardware.py` (the parity firmware's
sequential 32-bit read loop, IDF's default caches): internal SRAM 137.1 MB/s,
PSRAM 56.3 MB/s, and flash 26.6 MB/s through the cache (1 MB mapped with
`esp_partition_mmap`). The delegate's weight loops, measured with the story's
caches, are in the backend's README (Weight tiers).

## Bench flags

`scripts/export_bench.py MODEL... --out DIR` exports census models
(`backends/esp32s3/test/quant_census/models.py`) for the bench; `--timed-runs`
and `--stepped-runs` set how many executes each measurement times and splits
by instruction (default 10 and 3; 5 and 1 past 100M MACs), and `--host-bench`
names an `esp32_bench_host` other than the one beside the host runner. The
models it was written for, under Arm's a8w8 recipe, every MAC delegated and
both measurements exact on the host:

| Model | MACs | Delegates | Planned arena | Largest scratch | Weights |
|---|---|---|---|---|---|
| `ds_cnn` | 2.66M | 2 | 34,784 B | 28,672 B | 30 KB |
| `mobilenet_v1_025` (VWW) | 7.49M | 6 | 223,504 B | 69,584 B | 241 KB |
| `resnet8` | 12.5M | 4 | 200,768 B | 44,416 B | 83 KB |
| `mobilenet_v2_96` | 56.3M | 12 | 221,184 B | 349,184 B | 3.65 MB |
| `mobilenet_v2_224` | 300.8M | 12 | 1.2 MB | 1.68 MB | 3.65 MB |
| `mobilenet_v3_small` | 56.5M | 36 | 803 KB | 313 KB | 2.64 MB |

The MLPerf Tiny graphs keep explicit float32 pads between delegates (five in
VWW, one before each of ResNet-8's stride-2 stages) that no convolution
absorbs, and their tensors add to those two arenas.

`scripts/run_bench_hardware.py` has four commands:

| Command | Flags |
|---|---|
| `build MODEL_DIR... --build DIR` | one firmware for every model given. `--backend-o2` (the delegate at -O2), `--default-caches` (IDF's caches instead of the story's), `--sdkconfig-extra FILE`, `--dual-core` (projection columns split across both cores, no convolution; images exported with `--min-latency`), `--tier-bytes N` (the PSRAM measurement's weight budget; by default every weight PSRAM holds) |
| `run MODEL_DIR... --build DIR --serial SERIAL` | flash one model at a time and capture its boots: `--boots` (3), `--board` (`papercolor`), `--baud`, `--timeout` (seconds a boot, 1800), `--log-dir` |
| `summary LOG...` | the saved boot logs, grouped by model; `--json` |
| `show IMAGE` | a bench image's header and instruction labels |
