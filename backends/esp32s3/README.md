# ExecuTorch Espressif Backend

This subtree is an ExecuTorch backend for the Espressif ESP32-S3: two Xtensa LX7
cores at 240 MHz, 512 KB of internal SRAM, external PSRAM and flash, and the
PIE 128-bit SIMD instructions. The partitioner claims the matrix work of a model
(matmuls, convolutions and embedding lookups with their epilogues) and
preprocess packs each partition into one delegate blob. On the chip the delegate
runs that blob with PIE kernels, reading weights in place from flash or from a
copy in PSRAM. A custom-op library (`espressif::*`) and exact fast paths for
some portable kernels cover the operators between delegates.

The ESP-IDF firmware that runs delegated programs, the parity suite in QEMU and
on a board, and a story-writing language model demo are in
`examples/espressif/esp32s3`.

## What it delegates

**Operators.** `aten.mm` and `aten.addmm` (Linear); `aten.embedding`;
convolutions (1-D, 2-D and 3-D, grouped, depthwise and transposed, the
transposed ones with output padding and dilation), lowered to the same matmul
through an im2col gather; and, opt-in (`quantize_qk=True`), the
batched `q @ k^T` of attention, quantised to int8 on both sides or read from an
int8 K cache. Fused into a matmul or convolution: the bias, ReLU and clamp
chains, SiLU, GELU (erf and tanh forms), sigmoid, hardswish, hardsigmoid, leaky
ReLU, ReLU², a residual add, a multiplicative gate (SwiGLU, GeGLU) and logit
soft-capping. An `index_select` feeding a Linear becomes a selected-row output
head. Consecutive claimed ops run in one delegate call, their intermediates kept
in delegate scratch.

**Weight formats.** int4, int8 and int16 codes with fp32 or fp16 scales per
output row or per group of 128 (`group_weights=True`); fp32; and fp16 or bf16
weights kept as stored. `format_policy` picks the format per op and
`format_overrides` names one for an op. `"auto"` decides by size alone: int4
for a weight of 128 KiB or more at int8, which it takes to be read from flash
or PSRAM and so bound by bandwidth, and int8 below that
(`target/chips.py`, `sram_resident_bytes`). Where the weights are read from at
run time is the application's choice (Weight tiers, below), which preprocess
cannot see.

**Quantization.** Without a quantizer the delegate quantises a float model
itself: absmax weights at export, int8 activations per row at run time. It also
runs a model's own quantisation exactly, as the PT2E or torchao flow left it:

- `Esp32Quantizer`, a PT2E quantizer: int4 per-channel weights and dynamic int8
  activations by default, or (`preset=`) the schemes of the stock quantizers,
  `xnnpack_static`, `xnnpack_dynamic`, `arm_a8w8`, `arm_a8w4`, `arm_a16w8` and
  `cortex_m`, with QAT (`is_qat=True`). Keywords given with a preset override
  it: `arm_a8w8` with `activation="dynamic_per_tensor_affine"` and
  `bias="fp32"` is Arm's dynamic configuration. A preset's specs, eps,
  observers and fake-quantisers equal its stock quantizer's;
  `test/quantizer/test_espressif_quantizer.py` checks that on every census
  model, pair by pair;
- graphs quantised by those stock quantizers themselves, and by torchao
  (`esp32_torchao_config`: per-group int4 or int8 weights with dynamic
  per-token activations);
- symmetric and asymmetric (zero-point) weights; dynamic per-row, per-token and
  per-tensor activations, symmetric or affine; static int8 graphs with int32
  biases; static int16 activations with int8 weights (a16w8); and the
  asymmetric int8 K cache of ExecuTorch's `QuantizedKVCache` (its writer and
  V read always, its `q @ k^T` under `quantize_qk`).

A scheme the runtime cannot run exactly is refused when the quantizer is
built or when the graph is partitioned, with the rule that refused it.

**Static int8 convolutions.** A convolution whose activation codes depend on
each input element alone -- a static int8 graph's, read as Stage B's codes or
quantised from fp32 at a static or per-tensor scale -- takes a direct path
(`runtime/esp32_conv.h`): every input element's code is made once per op, a
depthwise convolution runs on its channel's code plane, and a pointwise or
dense one runs as row blocks of the int8 dot written straight to NCHW, one
PIE call per channel per row block: for rows of up to 112 bytes the dot and
the code epilogue in one loop with the channel's weights held in registers
(`dot_rows_codes_s3`), else the dots (`dot_i8_rows_s3`) and then the
epilogue. A 3x3 depthwise convolution with 16 channels or more and no
dilation along its width runs each whole group of 16 channels as one PIE
vector (`dw33_qacc16_s3`), each channel's epilogue reading its sums from its
QACC lane where the kernel left them (`qacc_lane_codes_s3`). A 5x5 one, and
the 3x3 channels past the last whole 16 (or an op of fewer), with a column
stride of 1 or 2 and no dilation along its width, run one channel at a time
on the padded plane the scalar loop reads, 16 output pixels of an output row
a PIE vector (`dw5_plane_s3`, `dw3_plane_s3`; a 5x5 at a stride of 1 whose
output rows are at most 8 pixels, two output rows a vector: `dw5_pairs_s3`),
in the scratch the scalar loop already has; the rest take the scalar loop.
Below the delegate's `run_part`, the path's deepest stack is 2.5 KB on those
kernels (the pairs form: 2,528 B at -O2, `-fstack-usage`; the plane path
2,464 B), 1.2 KB on the scalar loop and 0.96 KB on the 16-lane path; an
application that runs inference from a task of its own sizes it for that, on
top of ExecuTorch's and the delegate's frames.
The 16 channels' strip of a depthwise convolution and a
pointwise convolution's rows are gathered from the NCHW codes by a PIE
transpose (`transpose16_s3`). Self-tests at the first convolution's init find
which QACC lane holds which channel and check each kernel against its C
computation; on a failure that step takes its scalar code (the depthwise
convolutions the scalar loop). A code output is computed in integers, by a
bound proved against the float requantisation it replaces, which still runs
for the few elements near a rounding boundary ("esp32 convepi" counts both). The path's output is the
general loops' bit for bit. Like every convolution, it runs on one core,
under `min_latency` too ("esp32 convsplit" stays 0; Known limits says why).
On the board, measured without the integer epilogue and the PIE kernels,
the path ran DS-CNN in 46.2 ms an inference where the general
loops take 354.0 (both builds at -Os), and in 44.0 ms with the delegate and
ExecuTorch's runtime at -O2. Against the general loops at -Os, that -O2
build took MobileNet-v1 0.25 from 1,137 to 172.5 ms, ResNet-8 from 600 to
159.6 and MobileNetV2-96 from 5,300 to 711.2, every output unchanged
(`examples/espressif/esp32s3/README.md`, Performance).
`esp32_backend_set_conv_fast(0)` (the host runner's `--no-conv-fast`) sends
every convolution back to those loops, and an application without
convolutions can leave the path out (`-DESP32_CONV_FAST=0`, as the story
firmware does).

**Between delegates.** ExecuTorch's per-tensor quantize and dequantize
(`quantized_decomposed::quantize_per_tensor.out`, `dequantize_per_tensor.out`)
and `avg_pool2d.out`, the portable operators a quantised CNN leaves between its
delegates, take exact fast paths (`ops/qdq_pool.h`): the quantize does the
portable kernel's arithmetic in float32 (its scalar loop rounds and clamps in
soft float64, ~350 cycles an element on the S3), and the pool sums each window
in the portable kernel's order without its per-element index arithmetic. Every
output is the portable kernel's bit for bit, and every input outside a fast
path goes to that kernel. A firmware takes them only for a program whose
operator list names those operators (`gen_kernels.py`, whose `--no-qdq-fast`
keeps the portable kernels); the host runner's `--no-qdq-fast` sends every
call to the portable kernel, and `--qdq-sweep full` checks the loops against it
over every float32 input.

A dynamically quantised op the partitioner does not claim (a convolution
over a batch of images, a dot past its exactness bound) keeps its activation
quantiser in front of it, said so at lowering ("esp32: leaving ...").
`lower_esp32` writes the chain's chooser, quantize and dequantize, PT2E's or
torchao's, out as the float32 portable ops the recipe computes them with
(`amin`, `amax`, `clamp`, `neg`, `maximum`, `sub`, `div`, `reciprocal`, `mul`,
`round`, `add`), so the op reads the converted model's values bit for bit and
the runtime runs no chooser: ExecuTorch's `choose_qparams` kernel is not
PyTorch's, it has no symmetric one, and torchao's ops have no ExecuTorch
kernels. A chain over constants (learned query tokens) is folded at build
time, with the views after it, and so is a torchao weight that no delegate
takes.

**Non-finite values.** Every quantiser whose codes have a portable reference
follows ExecuTorch's rule (#22811): NaN takes the zero point, +inf the top of
the code range and -inf the bottom, with every value clamped before it is
converted to an integer (`runtime/quant_rules.h`). That covers PT2E's static
and folded per-tensor activations, the affine records' activations and output
requantisation, int16 activations, the stock K/V cache's writer, the
activation tables and the fused norms. The dynamic choosers have no portable
kernel to follow: they skip a NaN, and an infinity gives an infinite scale, as
in torch. The delegate's own per-row scheme (absmax scales, rounded half away
from zero) has no portable reference either, and leaves NaN out of its
contract; a finite row too small to have a reciprocal scale (absmax below
127 * 2^-126, so a subnormal or zero scale) quantises to zero codes, in every
such quantiser and mirror (`a0_inverse_scale`). A dynamic chain left in place
(above) gives the converted model's values for NaN as for finite activations,
and leaves the infinities out of its contract, as the kernels do: an affine
chain over an infinite element gives NaN where torch's casts its NaN codes to
integers and gives +inf. One whose eps has no finite float32 reciprocal (a
subnormal eps) would turn a zero row's codes into NaN the same way, so it is
refused by name (`act-eps-in-place`) rather than computed in place.
`test/runtime/nonfinite_check.cpp` holds each quantiser to the
rule, against ExecuTorch's quantize kernel on the host (CTest
`espressif_nonfinite`) and in the parity firmware on the chip.

**Stage A and Stage B.** In Stage A every delegate takes and returns fp32 and
quantises inside. Stage B (`stage_b`, on by default for quantised graphs) keeps
a static int8 graph's activations as int8 codes between claimed ops and across
delegate boundaries, wherever the ops between can run on codes: the same
codes in a quarter of the activation memory.

**External weights.** `external_weights=True` (or the name of a data file)
moves weight and scale payloads out of the blob into named data in a `.ptd`
file. The firmware maps that file from a flash partition, so a model's weights
are limited by flash rather than by the program.

**Custom ops** (`ops/`, registered by `espressif::register_ops()`): RMSNorm
(also fused with a residual add, or without a scale), RoPE, the attention tail
(masked softmax and P@V), LayerNorm2d (built without multiply-add
contraction on the chip too, as the graph it replaces rounds each product),
quantised RMSNorm and LayerNorm (`qnorm`), int8 activation tables
(`quant_lut`) and fitted int16 activation tables (`lut16`), and row and token
quantisers. The passes in `_passes/` fuse
the matching ATen patterns into them. The same library overrides eleven
portable kernels (softmax, add, bmm, div, index_put_, le, mul, permute_copy,
remainder, split_with_sizes_copy and where) with exact fast paths for the
cases they recognise, and calls the portable kernel for every other case;
quantize, dequantize and avg_pool2d (Between delegates, above) join them.
`ops/elementwise.h` adds loops for the element-wise work a quantised CNN
leaves between its delegates: the squeeze-excite scale (`mul` of one value
per channel), the pads' row copy (`constant_pad_nd`) and the `quant_lut`
tables' loops. A firmware takes the `mul` and pad loops only when built with
`gen_kernels.py --elementwise-fast` (the bench and parity builds pass it, the
story's does not; `run_bench_hardware.py build --no-elementwise-fast` for an
A/B); the table loops run wherever a table op does, and only the host
runners' `--no-ew-fast` turns every loop off. A
firmware registers each of the library's kernels, and so links it, only where
one of its programs names the operator (`gen_kernels.py` writes one switch per
operator into `Esp32KernelVariant.h`); the host runner registers them all.
`index_put_`'s row copy takes negative indices and values that leave out
leading dimensions of size 1, as the portable kernel does. A self-check
compares every fast path's written bytes with the portable kernel's, on the
host (`esp32_host_runner --selftest`) and in the parity firmware.

## Using it

Lower with `lower_esp32` (export -> edge -> the backend's canonicalisation and
fusion passes -> partition), run the post-lowering passes, then emit:

```python
import torch
from executorch.backends.esp32s3 import Esp32Partitioner
from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
from executorch.backends.esp32s3.memory_planning import planning_pass
from executorch.exir import ExecutorchBackendConfig

model = MyModel().eval()
example = (torch.randn(1, 64),)
ep = torch.export.export(model, example)
edge = lower_esp32(ep, [Esp32Partitioner(external_weights=True)])
apply_passes(edge)
et = edge.to_executorch(ExecutorchBackendConfig(
    delegate_alignment=16, constant_tensor_alignment=16,
    memory_planning_pass=planning_pass("esp32")))
with open("model.pte", "wb") as f:
    f.write(et.buffer)
et.write_tensor_data_to_file(".")  # esp32_weights.ptd
```

The 16-byte alignment lets the PIE kernels read weights in place;
`planning_pass("esp32")` plans the arena with in-place reuse, a minimum-peak
operator order and a hill-climbing allocator (`None` is ExecuTorch's planner).

To quantise first, annotate with `Esp32Quantizer` and convert before export:

```python
from executorch.backends.esp32s3 import Esp32Quantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

graph = torch.export.export(model, example).module()
prepared = prepare_pt2e(graph, Esp32Quantizer(preset="xnnpack_static"))
for batch in calibration_data:
    prepared(*batch)
ep = torch.export.export(convert_pt2e(prepared), example)
```

and lower `ep` as above: `lower_esp32` absorbs the activation quantisers
into the ops the delegate claims, recording each one's scheme, so the
delegate quantises as the converted model does.

**Partitioner options** (`Esp32Partitioner(...)`):

| Option | Default | Effect |
|---|---|---|
| `external_weights` | `False` | weights in a `.ptd` file (`True`: `esp32_weights`, or a file name) |
| `format_policy` | `None` | weight format: `"i4"`, `"i8"`, `"i16"`, `"auto"`, `"fp32"`, `"fp16"`, `"bf16"`. `None` keeps a quantised model's own formats and uses int8 for the weights of a float model; in a program a recipe quantised, a weight the recipe left float keeps its precision (the fp32 loop for a matmul or convolution, portable fp32 for an embedding, fp16 or bf16 for a half-precision checkpoint's weight). An explicit policy quantises every float weight the delegate takes |
| `format_overrides` | `None` | a format per op name |
| `requantize_promoted` | `"half"` | which already-quantised weights may be quantised again: `"half"` fp16 and bf16, `"all"` fp8 as well, `"none"` neither |
| `group_weights` | `False` | per-group-of-128 scales for Linear weights |
| `static_act` | `False` | calibrated activation scales instead of per-row ones (`quantizer/calibration.py`'s two-step lowering). An op a dynamic recipe quantises is refused under it ("static_act over a dynamic recipe") and stays portable on the recipe's value |
| `quantize_qk` | `False` | delegate attention's `q @ k^T` in int8 (int8 query rows), over a stock affine K cache too |
| `min_latency` | `False` | split every projection's columns across both cores in each call whose outputs and scratch are internal RAM (no convolution splits; its limits on ESP32-S3 rev v0.2: Known limits) |
| `stage_tied_head`, `split_tied_head` | `False` | stage a tied output head in PSRAM as int8, and split only that op across both cores (its limits on ESP32-S3 rev v0.2: Known limits) |
| `conv` | `True` | claim convolutions |
| `stage_b` | `None` | int8 codes between claimed ops (`None`: on, except under a float `format_policy`) |
| `luts16`, `fused_norms` | `True` | fold int16 activations into `lut16` tables and quantised norms into `qnorm` |
| `plan_scratch` | `None` | declare each delegate's working memory to the memory planner (Delegate scratch, below); `None` reads `ESP32_PLAN_SCRATCH` from the environment, and is on unless it is `0` |

**Runtime options** are set by the application through ExecuTorch's backend
options (`executorch::runtime::set_option("Esp32Backend", ...)`): they belong
to the deployment, not to the model, and change without a re-export.
`min_latency` takes -1 (the default: each program's `min_latency` compile spec
decides), 0 (one core) or 1 (projections split across both cores), or a bool
for 0 or 1. The weight tiers' options follow. An unknown key is an error, so a
misspelt option is not silently ignored.

## Weight tiers

The example firmware maps the weights file from flash, and a quantised matmul
is bound by how fast it reads its weights (the measured rates are below). The
delegate can copy external weights into PSRAM, within a budget the
application sets. A copy holds the stored bytes, so the outputs are bit for
bit the same from either tier.

| Option | Default | Effect |
|---|---|---|
| `weight_tier_budget` | 0 | bytes of PSRAM the copies may take. 0 reads every weight from flash; -1 (auto) copies every candidate while `weight_tier_reserve` bytes of PSRAM stay free, checked at each allocation, and leaves fp16 weights on flash |
| `weight_tier_reserve` | 0 | the PSRAM auto leaves free |
| `weight_tier_tables` | `false` | also copy embedding tables and selected-row heads, which read a few rows per inference |
| `weight_cache` | `true` | keep the copies when a method is destroyed, so prefill, decode and every reload share one copy of each weight |
| `weight_tier_rank` | `false` | ranked placement instead of load order |
| `weight_tier_apply` | | `true` makes the ranked placement now; read back, whether a ranked load is still waiting |
| `weight_tier_wanted` | | read-only: the bytes the candidates of the methods loaded under a budget would take, each weight once |
| `weight_tier_resident` | | read-only: the bytes the copies take now |

The options apply to the loads made after they are set. Candidates are the
ops whose stored weights are external (named data in a `.ptd`), in every
format: int4, int8, int16, fp32, fp16 and bf16, matmuls and convolutions,
depthwise ones included. Inline weights are not candidates. An asymmetric op's
zero points, offsets and weight sums stay on flash, read in place. An int4
head that `stage_tied_head` expands to int8 is made whatever the budget and
counted first; a PSRAM shortage fails its load. Methods that hold it at once
share it; when the last lets go, it is kept for the next method under a
budget and freed otherwise.

Each candidate's score is the bytes its loop reads in one execute over the
bytes it stores. A budget is a cap, not a reservation: only what is copied is
allocated. In load order, the default, each load copies its candidates in turn
while they fit. A newcomer that does not fit may evict copies no loaded method
holds, whatever their score, least worth first and only until it fits; when
even evicting all of them would not make room, nothing is evicted. A load
takes its references on the copies it reuses before any of its newcomers
evicts. A newcomer's PSRAM is allocated before anything is evicted for the
budget; when PSRAM itself is short, unused copies are evicted, least worth
first, until the allocation succeeds, and none when even all of them would
not free enough.

Load order loses a large weight that loads last, an LLM's output head say, to
the small weights before it. Ranked placement (`weight_tier_rank`) orders the
candidates of every loaded method, each weight once, by score, then format
class (int4, int8, fp32, bf16, int16, fp16), then size, the largest first, and
copies them first fit. A ranked load copies nothing new: the application sets
`weight_tier_apply` after loading every method that will run and before the
first execute it times, or that first execute places the weights itself.
Copies made earlier stay, and placement never frees a copy a loaded method
holds, so an execute running meanwhile on the other core reads either flash or
a finished copy of the same bytes. On the story model at a 4.5 MiB budget,
load order copies 81 weights (4,123,839 B) and leaves the 1 MB output head on
flash; ranked placement copies 57 (4,714,839 B), the head among them, with
bit-identical logits, and decode went from 110.10 to 98.85 ms per token on the
board.

`runtime/esp32_weight_tiers.h` reports the tiers to the application:
`esp32_backend_weight_tier_stats` fills a struct with the bytes copied and
left on flash per format class, the evictions and the placements, and
`esp32_backend_report_weight_tiers` prints it with one `esp32 tier:` line per
weight. Before a board is involved,

```bash
python -m executorch.backends.esp32s3.utils.weight_tiers model.pte --budget 4.5M
```

replays the runtime's arithmetic on a lowered program: per weight, its class,
size, score and the time a copy saves per execute at the measured rates; per
budget, the bytes copied in load order, ranked, and at best (`--help` lists
the methods, tables, auto and PSRAM options). `test/test_weight_tiers_report.py`
holds its prediction to the host runner's own report, and the host runner
takes every tier option (`test/host_runner/README.md`).

Read rates measured on the board (M5Stack PaperColor) for the delegate's
weight loops, in MB/s, with the story firmware's caches (64 KB data cache with
64-byte lines, 32 KB instruction cache); a rate counts the bytes the rows
hold, codes and scales:

| Weights, cores | Flash | PSRAM | Internal SRAM |
|---|---|---|---|
| int4, 2 | 32 | 91 | 290 |
| int8, 2 | 32 | 91 | 542 |
| bf16, 1 | 31 | 45 | 59 |
| fp16, 1 | 25 | 26 | 27 |

The float loops run on one core. fp16's is bound by widening each weight, so
it reads no faster from PSRAM than from flash, which is why auto leaves fp16
on flash. `target/chips.py` keeps every measured rate with where it was
measured; `run_hardware.py --rates LOG` reads them from a board log (the
parity firmware's, or that of a story build with `sdkconfig.bench.defaults`).

## Delegate scratch

Each delegate call needs working memory: a block of quantised activation
rows, the intermediates of its chain of ops, q@k^T's quantised keys and the
affine terms. By default (`plan_scratch`) preprocess declares it to
ExecuTorch's memory planner as a `DelegateScratchSpec` (#21985). The planner
places it in the method's arena, where it may reuse bytes that are dead
during the call, and the method states its whole memory need at load.
`serialization/scratch_layout.py` computes the size from the blob as the
runtime lays it out, plus `SCRATCH_SLACK` (15 bytes): the planner aligns
offsets within the arena, but the arena's base is the application's (a
`heap_caps_malloc` buffer is 4-byte aligned), so the runtime aligns the
buffer itself. A call whose planned buffer cannot hold the aligned layout
falls back to the temp allocator, logged and counted. With
`plan_scratch=False`, or `ESP32_PLAN_SCRATCH=0` in the environment (read once
per process, so every child process that lowers inherits it), the runtime
takes the working memory from the method's temp allocator at every call.

On the story model, decode's peak (arena plus temp) goes from 39,168 to
34,560 B and prefill's from 1,006,224 to 892,432 B, with every output
identical; the program grows from 162,932 to 170,100 B. On the board the
story decoded at the same speed either way (+0.09% at one PSRAM timing
point), so it is on by default, though ExecuTorch still marks the scratch API
experimental. `esp32_backend_report()`, which the parity and story firmware
call, prints where every call's scratch came from: `esp32 scratch: N call(s)
on planned scratch, N on the temp allocator (none planned), N fallback(s)
(planned scratch unusable)`.

## Memory placement

Internal SRAM reads fastest, then PSRAM, then flash, and internal SRAM is
scarce. The backend leaves where each kind of memory goes to the
application:

| Memory | Without the application's say | Moved by |
|---|---|---|
| Weights | read in place from the mapped weights file | the weight tiers, above |
| Activations | the planned buffers the application passes in, one per mem-id | the application |
| Delegate scratch | planned with the activations (`plan_scratch`, above; off: the method's temp allocator) | the application, with the activations |
| Constants outside the delegates (norm weights, masks) | read in place from the named data | a `NamedDataMap` wrapper that copies small ones into internal SRAM on first request |
| Code | flash, through the 32 KB instruction cache | `CONFIG_ESP32_BACKEND_HOT_IRAM`, below |

The story firmware has a flag for each
(`examples/espressif/esp32s3/README.md`). Measured there on the board, each
against the same image without it: decode's activations in internal SRAM
(33.8 KB) save 1.81% of a decode step, its small constants (45.6 KB) 1.35%,
and the per-token code (26,952 B of internal RAM) 1.34%. About 50 KB of
internal SRAM is left for them after the KV cache and the story's 40 KB
floor, so they do not all fit at once; per kilobyte, the activations save the
most, and the story places them first.

## Building the runtime

The runtime, kernels and custom ops build as the ESP-IDF component
`esp-idf/espressif_backend` (ESP-IDF v5.5.5). `examples/espressif/esp32s3` adds
it, together with an ExecuTorch core component and a selective kernel build,
to its firmware.

The component compiles at -Os, apart from the float loops
(`runtime/esp32_float.cpp`, always -O2) and, by default, the operator
library. Its speed switches are a CMake variable and the Kconfig menu
"Espressif backend (ESP32-S3)"; a project changes the Kconfig ones in a
defaults file of its own. Decode times are the story model's on the board, at
one PSRAM timing point, each against the same build the other way:

| Switch | Default | Effect |
|---|---|---|
| `ESP32_BACKEND_O2` (CMake) | off | the delegate at -O2 |
| `CONFIG_ESP32_BACKEND_OPS_O2` | `y` | the operator library (the custom ops and fast paths between delegates) at -O2 instead of -Os: 6,448 B more flash in the story firmware, 14,128 B in the parity firmware, the same outputs; decode 89.54 ms per token at -Os, 87.99 at -O2 |
| `CONFIG_ESP32_BACKEND_SPLIT_MIN_COLS` | 32 | the narrowest projection `min_latency` splits across both cores. 64 keeps the story's 32-column K and V projections on one core, with the same outputs, and measured no gain (+0.14%) |
| `CONFIG_ESP32_BACKEND_HOT_IRAM` | `n` | the code a decode runs every token (the functions marked `ESP32_HOT_CODE`, the PIE kernels and ExecuTorch's instruction loop: `runtime/esp32_hot.lf`) from internal RAM instead of through the 32 KB instruction cache. The internal heap shrinks by as much, 26,952 B in the story firmware; decode 89.54 to 88.34 ms per token. Off by default: the story spends that internal SRAM on decode's activations instead; in their place, with the activations in PSRAM, it measured 0.33% faster than the defaults, inside the 0.5% that would change them |

ExecuTorch's own runtime and portable kernels stay at -Os: the example's
`CONFIG_ET_PORTABLE_O2` grows the story firmware by 160,528 B, past its
partition, and the only portable kernel a story decode step runs is
`index_select`, twice.

## Host test runner and tests

`test/host_runner` builds the same delegate, op library and portable kernels
for the build machine, so lowered programs run on the host with the firmware's
arithmetic. From the root of the ExecuTorch tree, in a Python environment with
ExecuTorch installed:

```bash
cmake -S backends/esp32s3/test/host_runner -B cmake-out-espressif \
  -DPYTHON_EXECUTABLE=$(which python)
cmake --build cmake-out-espressif -j
ctest --test-dir cmake-out-espressif --output-on-failure
```

`test/host_runner/README.md` describes the runner and what it checks. The
Python tests are under `test/`: files with a `__main__` block run as scripts
(`python backends/esp32s3/test/test_fusions.py`), the others under pytest
(`python -m pytest backends/esp32s3/test/asym`). The tests that run programs
find the runner in `cmake-out-espressif/` (`ESP32_HOST_RUNNER` overrides it).
`test/quant_census` lowers every supported quantization recipe on a set of
models and takes about half an hour. The device parity suite,
`test/parity/run_parity.py`, builds the example firmware and compares every
row in QEMU (`examples/espressif/esp32s3/README.md`).

`test/suite` runs ExecuTorch's backend operator suite (`backends/test/suite`:
80 operator files, 467 tests) on two flows, the float model and the default
`Esp32Quantizer()`, every case on the host runner and compared with eager at
the suite's tolerance (`test/suite/README.md`). On both flows 342 pass, 28 are
expected failures (3-D pooling, which has no portable kernel), 97 are skipped
by the suite itself, and none fail. `test/test_backend_suite.py` runs 41 more
patterns the backend's own rewrites look for, the same way.

## Layout

```
backends/esp32s3/
├── __init__.py             # Esp32Partitioner, Esp32Backend, Esp32CompileSpec, Esp32Quantizer
├── compile_spec.py         # Esp32CompileSpec: the options preprocess and the runtime read
├── partition/              # the partitioner
├── operator_support/       # what is claimed, and why not
├── espressif_preprocess.py # Esp32Backend.preprocess: graph -> delegate blob
├── serialization/          # the blob's wire format and weight formats
├── quantizer/              # Esp32Quantizer, torchao configs, activation tables
├── _passes/                # canonicalisation, fusion and lowering passes
├── memory_planning.py      # the arena planner
├── ops/                    # custom ops (C++, LX7 assembly) and their Python schemas
├── runtime/                # the delegate (C++) and its PIE kernels
├── esp-idf/                # the ESP-IDF component
├── target/                 # chip constants and the board's measured read rates
├── utils/                  # the weight-tier report, multi-method programs
└── test/                   # tests, fixtures, the host runner, the parity and operator suites
```

## Known limits

- Static shapes only: a dynamic dimension is refused at partitioning.
- One target, the ESP32-S3. The kernels use its PIE instructions; the host
  runner substitutes C kernels with the same integer arithmetic.
- The blob format (version 24) has no compatibility promise: the
  runtime refuses a blob of any other version, so a program must be exported
  with the runtime it runs on.
- Activations cross a delegate boundary as fp32, or as Stage B's int8 codes:
  an op whose activation is fp16 or bf16 stays portable.
- Row-major tensors only. `lower_esp32` copies a channels-last graph input to
  row-major, but an op on a dim order the model sets itself (channels-last, a
  transposed view) stays portable.
- Convolutions take one image: a batch of more than one stays portable, and so
  does a grouped transposed convolution.
- The two-core splits, `min_latency`'s projection split and
  `split_tied_head`'s head split, are exact on ESP32-S3 revision v0.2 only
  within three limits. The reason: a line of PSRAM written while both cores
  run can lose writes. Checked op by op against one core, the wrong elements
  held older values and were spread over the whole cache line. While the two
  cores' parts shared lines, they were near the boundary between the parts;
  once the parts shared none, they were mostly in a channel's first positions.
  That is the class of Espressif's erratum CACHE-126, where an access to a
  line being written back is taken as a miss and can leave two copies of the
  line. Before the first limit below, MobileNetV3-Small's classifier, whose
  output is in PSRAM, came out wrong that way in some runs of every boot
  measured with its weights in PSRAM, and in none with its weights in flash.
  The limits:
  - **The backend keeps a delegate call on one core when any tensor it writes,
    or any of its scratch, is in PSRAM**, with the same bits. "esp32 split: N
    call(s) kept on one core: they write PSRAM" in the backend report counts
    these calls. A call whose writes are all internal splits as before. In the
    story, decode calls split; prefill calls, whose activations are in PSRAM,
    run on one core.
  - **No other task may write PSRAM while a call splits**, whether another
    handle's call or the application, such as a display renderer on core 1.
    The backend checks only its own calls.
  - **A call must not split while it reads a PSRAM weight copy whose lines may
    still be dirty.** That means a copy made at load (`weight_tier_budget`,
    `stage_tied_head`'s expansion), by `weight_tier_apply`, or by the first
    `execute()`'s ranked placement, with nothing since to write it back.
  Both of the last two hold when nothing splits: `min_latency` and
  `split_tied_head` off in the compile spec, or the `min_latency` runtime
  option set to 0. With the weights in flash, the third always holds; the
  second is the application's to keep. The story keeps to the third with
  one-core traffic: its startup golden runs two prefill blocks on one core
  before decode's first split. Their weight traffic, megabytes through the 64
  KB data cache, evicts and so writes back the copies' dirty lines. Its
  release golden ran with the display idle, which meets the second. Its
  interactive display, LVGL rendering into PSRAM on a core-1 task, can run
  while decode splits, which the second limit rules out. Only this backend's
  own worker on core 1 was measured: a one-core call writing PSRAM while
  another core-1 task uses PSRAM has not been.
- Convolutions run on one core, under `min_latency` or otherwise; only
  projections split their columns across the cores. On ESP32-S3 silicon
  (revision v0.2 here), splitting a convolution across the two cores gave
  wrong outputs and cache-error panics ("Dbus write to cache rejected") with
  its weights or activations in external memory, flash or PSRAM. Most likely
  that is the class of Espressif's erratum CACHE-126 (an access to a cache
  line the other core is writing back can return the wrong copy). The board
  showed it with both cores streaming the same weight rows, and still with
  the two cores' rows half the op apart. No code stores into a weight: the
  host runs every bench model with its program and weights mapped read-only.
  A two-core convolution comes back only once the board proves it exact on
  every bench model, in both measurements, over many boots. Only DS-CNN and
  MobileNet-v1 0.25 came out exact split on every board run measured
  (ResNet-8, the MobileNetV2s and MobileNetV3-Small did not), so
  only they give the cost, on the same board with the backend at -O2: split,
  DS-CNN took 10.36 ms and MobileNet-v1 0.25 took 59.3 ms with its weights
  in flash (55.3 in PSRAM), against 17.85, 87.5 and 82.8 ms on one core
  with the same build (one without the PIE lane epilogue and the MobileNet
  kernels behind the one-core times below).
- On one core, with the backend at -O2, the bench measured DS-CNN at
  10.29 ms an inference. MobileNet-v1 0.25 took 57.3 ms with its weights in flash
  (52.6 in PSRAM), ResNet-8 69.5 (67.5), MobileNetV2-96 340 (265),
  MobileNetV2-224 1,390 (1,249) and MobileNetV3-Small 526 (472), every
  output exact and every run the same over 3 boots.
- A fused residual add or gate multiply needs an operand of the output's own
  shape, with no broadcasting, so a squeeze-and-excitation gate stays
  portable; a residual add does not fuse into a grouped convolution.
- `q @ k^T` is int8 on both sides, and P@V stays portable (the `attn_tail`
  custom op computes a decode step's masked softmax and P@V in fp32). The
  bmm is claimed only over operands whose origin the gate proves
  (`operator_support/qk_operands.py`), by lists: a runtime value the model
  computes in plain float (a listed value-computing op -- matmul, norm,
  softmax, a transcendental function, a reduction -- arithmetic between
  values that all vary and are not all on grids, such as a value times a
  mask or plus a fake-quantised other value, or a graph input), found
  through the ops that move, select or keep values and through scales and
  offsets, and an int8 K cache's codes, read as stored. These keep the bmm
  portable, wherever data movement puts them, and the refusal names what it
  found:
  - a recipe's value: a static pair, constant codes, a dynamic chain whole
    or computed in place, any quantise or dequantise;
  - a value rounded in float ops (round, floor, ceil, trunc, a division
    that rounds), alone or scaled and offset: by a number; by a tensor that
    holds one value wherever the rounded value may differ (per row, per
    channel, per group: one value, or one per group, expanded, repeated,
    reshaped, stacked, concatenated or gathered to the operand's shape, or
    a buffer whose values are); or by the very step it was rounded with,
    whatever that step's shape -- `round(x / s) * s`,
    `round(x * (1 / s)) * s`, `(x // s) * s`, a group-wise step written
    once or twice, clamped or guarded with `+ eps` where it divides;
  - a straight-through estimator, `x + (fq(x) - x).detach()` or LSQ's
    round_pass: a sum whose equal terms cancel is what remains, terms equal
    by exact structure too (`x / s` written three times is one term; two
    chains over different inputs never are, however deep);
  - a value minus its remainder (`remainder`, `fmod`; `y - frac(y)` is
    `trunc(y)`), and a number or constant added and taken away again,
    `(x + 1.5 * 2**23) - 1.5 * 2**23` (the magic-number round);
  - a few fixed levels: a comparison cast to float; sign spelled `x /
    abs(x)`, `x / sqrt(x * x)`, `x / (x * x) ** 0.5`, `abs(x) / x` or with a
    reciprocal, `rsqrt` or `** -0.5`, guarded with `max(., eps)`,
    `maximum(., eps)` or `+ eps` for an eps up to 1e-6 (its value read
    through the factors that compute it), the same value on both sides
    element for element, placed by the same moves (`flip(x) /
    flip(abs(x))` is sign; `x / abs(x.flip(-1))` and `x / max(abs(x), 1)`,
    which is `clamp(x, -1, 1)`, are computation);
    a `where` (or masked fill, maximum, minimum) of numbers, factories,
    constants of any shape (a learned scalar or per-channel alpha, two
    learned tables) and their negations, or of a row's own scale; a
    constant table, or one scaled and offset at run time by values one
    along the lookup (`(nf4 * absmax)[argmin(...)]`), looked up at codes
    the model computes from a float -- a cast, a comparison, argmin or
    argmax, bucketize, an integer fill, and integer arithmetic over them (a
    lookup-table quantiser, a thermometer, nearest-level and vector
    quantisation, `codebook[argmin(...)]`); arithmetic between such values,
    `(x > 0).float() - (x < 0).float()`;
  - integer values cast to float (codes, wherever they are: an int8 K cache
    dequantised by group scales repeated to its width is refused), a
    constant or computed constant, or an op on none of the lists.

  What the gate cannot read never admits: a constant whose values it
  cannot read, or a move it cannot read, is taken as possibly a scale, and
  every limit of its search refuses or follows more operands.

  Behind the rules, a value check (`operator_support/qk_values.py`) runs
  the program, on every `q @ k^T` the rules would delegate, in up to three
  regimes of twelve runs -- the floats varying, the token ids varying, and
  the rows looked up at the token ids drawn unit normal with the ids held
  (so an LLM with no float input is judged on floats) -- each the example
  inputs and eleven seeded numpy draws (floats unit normal, token ids over
  their table's rows; positions -- an integer input that also writes a
  cache or indexes a buffer -- cache indices, masks and bools at the
  example's values, or zeros without example inputs; written buffers on
  copies, carrying state from run to run) -- and keeps the bmm portable when an operand the delegate
  quantises itself is on a grid in those values: few levels, rows or
  groups of four channels on a lattice of one step, or a channel or
  element on one step across the runs. Where a buffer the forward writes
  reaches an operand, each row of it is classed from values. The regime
  runs once more with the buffer's initial content drawn afresh (its
  warm-up on run 0's inputs too): a row the same in every run that this
  redraw moves holds only that content, which no run wrote, and is left
  out. Every other row is judged with every test -- a value computed from
  a written buffer's content, as every query past a cached layer is, and a
  cache's own rows read through moves included. The regime also runs with
  run 0's inputs on the state that enters run 1: a row this replay changes
  that is no row of a written buffer is computed from state an earlier
  call wrote (behind a cache shifted a row a call, a cache written at a
  counter the forward advances, a state the forward scales each call),
  which values cannot tell from a fresh value, and its `q @ k^T` stays
  portable -- speed only. A row is a written buffer's own when its value
  in run 0 is, bit for bit, a row some written buffer held entering run 0
  (a cache row, however long ago a run wrote it); the other rows are
  fresh. An operand with rows of both kinds (a ring or a counter cache read
  beside this call's keys) is also judged on each kind alone, on the rows
  the replay leaves in place alone, and, where the per-kind budgets below
  keep rows one shared budget would not, on the rows one shared budget
  keeps (the rows the replay leaves in place first, then those it changes,
  then rows the same in every run, 512 in all) together, each with every test;
  a grid in any of them, or in all the rows together, keeps the `q @ k^T`
  portable.
  Values never admit what the rules
  refuse, and a program that cannot run leaves the rules alone to decide;
  the partition log names each `q @ k^T` the values kept portable, and the
  test that found the grid, and each they gave no verdict on, and why. An
  integer input that indexes only a learned table (a position with no
  cache) is drawn like a token id. When the redrawn or the replayed run
  raises, the rows cannot be classed, and every `q @ k^T` whose operand a
  written buffer reaches stays portable (logged); so does it when the rows
  the written buffers held cannot be read (a buffer with no elements holds
  none) or an operand's rows cannot be classed -- never the rules alone.
  The check is bounded: it records at most 512 rows of each operand a run,
  except that an operand a written buffer reaches records every row and,
  once its rows are classed, keeps up to 1,536 -- up to 512 that change
  from run to run of each kind, fresh and a written buffer's, then others
  to 512 in all, and the up to 512 rows one shared budget keeps -- and
  256 MiB in all,
  and its runs stop at 120 s; past either, every `q @ k^T` it would judge
  stays portable (logged).
  On the story Glimmer's shape it adds 2.9 s to a 17.0 s lowering of both
  methods and 16 MB to a 640 MB peak on an x86 host, without the redrawn
  and replayed runs, which add about 0.4 s and 30 MB on a Mac; a large model's
  lowering runs into
  the 120 s first (up to forty-eight CPU forwards of the whole program,
  the warm-ups, the redrawn and the replayed runs included), and its
  `q @ k^T` then stays portable.

  The contract: the converted model's bits are kept for any operand that
  is a grid by the rules, or by its values on the sample runs, whatever
  arithmetic made it -- a grid behind an op that computes included:
  `exp(round(x))`, a saturating `clamp(64 * q, -1, 1)`, `(x + c1) - c2`
  with c1 != c2, a lookup written as a matmul, `one_hot(codes) @ table`.
  Everything else is a value the model computes, which `quantize_qk`
  requantises to int8 rows like any activation, within the mode's error:
  the scores stay within the mode's propagated error bound of the exact
  product of the operands, as a plain activation's do. Measured on a 7x31
  score matrix (unit-normal q and k, head width 16), a plain q's int8
  scores are 0.088 off the converted model's (0.64% of the largest score),
  and grid operands requantised the same way (the spellings above, and 24
  others like them) were 0.018 to 0.116 off (0.39% to 0.88%), each
  using 0.26 to 0.44 of the bound. The residual, requantised so: a grid
  the rules do not name and the sample runs do not show -- one only on
  inputs unlike the samples (`round(x / 10)` of unit-scale inputs is zero,
  a lookup at an id no draw picks) -- an operand at one of the value tests'
  thresholds, and the tests' measured limits. At a threshold -- a
  saturation whose rows are about a quarter on a grid, a level count at
  the sixteen-level ceiling -- an operand can fall either way between the
  gate's sample runs and another set of samples of the same inputs:
  of 29,330 generated grids in scope, 18 were admitted, none of them a grid
  `judge()` finds on all of eight other seed sets (each on zero to six of
  them).
  The limits: a step that varies element by element with the input
  (`round(x / s) * s` with `s = abs(x) / 2 + 0.1` has no lattice in any
  grouping), codes past 4,096 levels (255 in a group of fewer than
  eight distinct values),
  codebook rows on a one-row operand, rows that only a written
  buffer's initial content reaches, which no run wrote (left out), and two
  that the buffer kind by value brings. With example inputs that make a
  fresh row equal to a row a written buffer holds -- all zeros beside a
  cache whose unwritten rows are zero -- that row counts as the buffer's
  own, and the cache's plain rows can hide its grid (a counter cache
  beside fresh keys on a 0/1 grid the rules miss, with example rows that
  zero those keys, is delegated): export with representative example
  inputs. A grid of 0s and 1s also makes all-zero rows by itself, whatever
  the example inputs, and those rows count as the cache's own the same way,
  so representative example inputs do not remove that case. And a cache
  whose starting content is non-zero and plain
  dilutes the rows later runs write (a grid a counter cache holds at one
  to four rows a call can be delegated); real caches start at zero, and
  ExecuTorch keeps no initial content for a mutated buffer unless
  `InitializedMutableBufferPass` marks it. A very
  sparse operand (a relu past two standard deviations) can
  pass a lattice by chance and is refused: speed only. A buffer the forward writes is judged
  by what it writes, read directly or through moves: no partition takes a
  written buffer in, so the delegate reads it at run time. QK-normed
  attention with nearly one-hot scores loses accuracy at int8 (four nano zoo
  families): leave `quantize_qk` off for it. Every `q @ k^T` op quantises
  its query to int8 rows, so none is claimed without `quantize_qk` or under
  `format_policy="fp32"`; without them a `--quantize_kv_cache` export reads
  the stock affine K cache through the exact `dequantize_per_token_i8` and
  runs that `q @ k^T` on the portable bmm in fp32.
- The selected-row head takes int4 and int8 tables only.
- An int8 dot of more than 132,096 terms, past what its int32 accumulator
  holds exactly, stays portable.
- A partition whose ops need more than three intermediates alive at once
  fails at export (preprocess raises) instead of being split.
- Mixture-of-experts weights stay outside the delegate in fp32 (the 3-D
  expert weights run on portable index and bmm), so the nano zoo's MoE
  families do not fit 16 MB of flash at device size.
- From ExecuTorch's LLM export flow, `llama.custom_sdpa` and
  `llama.custom_quantized_sdpa` (export without `use_sdpa_with_kv_cache`) and
  torchao's `_embedding_Nbit` are refused by name, and
  `StaticQuantizedKVCache`'s attention stays portable fp32: its key scales lie
  along the `q @ k^T` contraction.
- An unrolled LSTM or GRU is one delegate per step, and with inline weights
  each step stores its own copy (327 KB for the census LSTM). With
  `external_weights` the content-addressed keys store each weight once.
- `Esp32Quantizer`'s presets see an `nn.LSTM` or `nn.GRU` as one node and
  leave its gate matmuls in float; stock Arm decomposes the layer and
  quantises them.
- Only external weights are weight-tier candidates. A placement only adds
  copies: it never moves a copy a loaded method holds, so load every method
  before applying. Ranking within the load itself, with no apply call, needs
  the method metadata at init that ExecuTorch's main branch has (#24260) and
  1.6 does not.
- fp32 results computed on the host can differ from the chip's in the last
  ulp, where the firmware's compiler fuses multiply-adds; integer kernels are
  exact (`test/host_runner/README.md`).
- The QEMU parity run cannot execute the RMSNorm assembly's `rsqrt0.s`, says
  nothing about speed or memory bandwidth, and has no PSRAM: ResNet-8's row,
  whose activations need 200,768 B, runs only on silicon (`run_hardware.py`),
  and the rows that expand a head into PSRAM at load (`stage_tied_head`) only
  on the host runner.
