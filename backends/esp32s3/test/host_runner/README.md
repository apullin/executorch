# Host delegate numerics

`test/host_runner/` builds the delegate (`runtime/esp32_backend.cpp`) and the
operator library (`ops/espressif_ops.cpp`) for the build machine (tested on
macOS), with this tree's portable runtime and kernels, and file/FlatTensor
loaders. All generated files stay in the build directory, the program and
FlatTensor schema headers included: the ExecuTorch wheel's `flatc` writes them
from `schema/*.fbs` at build time. Upstream's Python codegen registers the
portable ops. CMake 3.24+, a C/C++ compiler, and a Python environment with
ExecuTorch and torchgen are required.

From the root of the ExecuTorch tree:

```bash
cmake -S backends/esp32s3/test/host_runner -B cmake-out-espressif \
  -DPYTHON_EXECUTABLE=$(which python)
cmake --build cmake-out-espressif -j
ctest --test-dir cmake-out-espressif --output-on-failure
python -m pytest backends/esp32s3/test/test_host_runtime.py
```

The tests find the runner at `cmake-out-espressif/esp32_host_runner`;
`ESP32_HOST_RUNNER` overrides that path. The runtime and kernel sources come
from the tree the runner sits in (`-DET_ROOT=` names another one). In a git
worktree whose submodules are not initialised, set
`-DESPRESSIF_FLATBUFFERS_INCLUDE=` to another checkout's
`third-party/flatbuffers/include`. A kernel-only build needs no ExecuTorch or
Python: configure a separate directory with `-DESP32_HOST_BUILD_RUNTIME=OFF`
and run its CTest.

Inputs and outputs are raw binary tensor files, without headers. Dtype, shape,
and physical dim order come from the selected PTE method. For ordinary contiguous
fp32 tensors, use `numpy.asarray(x, dtype='<f4').tofile('input.bin')` and read
outputs with `numpy.fromfile('output.bin', dtype='<f4').reshape(output_shape)`.
There must be one `--input` and one `--output` per tensor, in method order; byte
counts are checked. The runner prints output dtype/shape and returns nonzero on
load, input, execution, or output-file errors. Scalar I/O and changing input
shapes are not supported.

```bash
cmake-out-espressif/esp32_host_runner --pte tiny.pte --ptd host_weights.ptd \
  --input input.bin --output output.bin
```

Omit `--ptd` for inline weights. Repeat input/output flags for multiple tensors;
`--method NAME` selects a method (default `forward`). `--force-scalar` exercises
the delegate's scalar branches. `--min-latency` executes both halves
of the column split sequentially on one core, preserving their arithmetic and
split counters; `--single-core` sets the backend's `min_latency` option to 0,
so a program exported with `min_latency` runs on one core as a single-core
application runs it. Counters labelled "PIE" indicate calls to the portable C kernel
entry points on this build; the closing `host kernels:` line counts the calls
into each of them (dot_i8, unpack_i4, the fused dot_i4_groups, rmsnorm),
including the backend's one-time init selftests. `--planned-offset N` starts
every planned buffer N bytes (below 64) past a 64-byte boundary, as an
application's `malloc` may: a program exported with `plan_scratch` must still
run every call on its planned scratch, and the backend's closing report says
where each call's scratch came from (`esp32 scratch:`).

The weights file is mapped once, as the firmware maps its partition, so every
method sees the same weight addresses. The weight tiers
(`../../README.md`, Weight tiers) run as on the chip, with PSRAM allocations
charged against a capacity, unlimited by default:

| Flag | Backend option |
|---|---|
| `--weight-tier-budget N\|auto` | `weight_tier_budget`: N bytes, or auto (-1) |
| `--weight-tier-reserve N` | `weight_tier_reserve` |
| `--weight-tier-tables` | `weight_tier_tables` |
| `--weight-tier-rank` | `weight_tier_rank` |
| `--weight-tier-apply` | `weight_tier_apply` after loading (otherwise each load's first execute places it) |
| `--no-weight-cache` | `weight_cache` false: copies go with their last method |
| `--psram-bytes N` | the PSRAM this host pretends to have; 0 means none, as in QEMU |

After the run the runner prints the tiers' report, one `esp32 tier:` line per
weight; `test_weight_tiers.py` and `test_weight_tiers_report.py` read it.

`--script steps.txt` replaces `--method/--input/--output` and runs many
executions in one process, for stateful programs such as the story model's
prefill/decode pair. Each line is `METHOD IN... -> OUT...` (`#` starts a
comment; paths cannot contain spaces or `#`). Lines run in order. Every method
uses the same planned buffers, each sized for the largest method that uses it,
so state planned into a shared mem-id (the story program's KV cache, mem-id-2)
carries from step to step and from one method to the other. One method is
loaded at a time and reloaded when the method changes, as the firmware loads
prefill and then decode; `--keep-methods` instead loads every method the
script names before the first line and keeps them all loaded, as an
application that applies ranked placement does. Method names and file counts
are checked for every line before the first one runs; a failure later keeps
earlier steps' outputs.

`--placement-stress ROUNDS` takes a two-line script and runs the backend's
ranked weight placement on a second thread while the first line's method
executes, round after round: the first method loads under a one-byte budget
(its weights registered, left on flash) and executes repeatedly while the
other thread loads the second method ranked, applies the placement, which
copies the first method's weights too, and executes it once; each method has
planned buffers of its own. Every execute must reproduce its method's first
bits, which go to the script's output files. Where a program built with
ThreadSanitizer runs on the build machine, the build also makes
`esp32_host_runner_tsan`, the same runner with the delegate under
ThreadSanitizer (`-DESP32_HOST_TSAN=OFF` skips it), and `test_weight_tiers.py`
runs the stress through it.

`examples/espressif/esp32s3/story/host_story_golden.py --emit-dir DIR` runs the story firmware's golden check
on the host: it reads an `export_story_flash.py` emit directory, runs prefill
chunks and then decode with the golden's tokens forced (as `story_runner.inc`
does), and prints the firmware's per-step checksums, choices and STORY_GOLDEN
fields. `--decode-only` and `--save-logits` let two runs be compared bit for
bit. The export takes minutes; the check takes about a second.

Coverage includes int8 linear (decode and multiple rows), grouped int4 with
fp16 scales, a ragged final group and fused ReLU, a decode-row int4 linear on
the fused group-dot path, external PTD weights, dynamic int8 Q@K^T with
multiple batches and the target softmax, the same Q@K^T against a stored int8
K cache (ragged and in-place rows, bit-equal to the dynamic op on the keys the
codes came from), and the RMSNorm/RoPE/LayerNorm2d
custom ops plus portable ReLU.
Model tests reuse `models.addmm_i8_absmax`, `addmm_codes_ref`, and `bmm_qk_i8`
from the QEMU parity flow and compare every element at 2e-6 relative/absolute
tolerance. Scalar, C-kernel, and serial-split integer-model outputs must also
agree exactly. CTest checks int4 codes, the fused int4 group dot against
unpack + dot, integer extremes/overflow, block tails, rounding, and RMSNorm
against the scalar implementation.

This does not emulate SRAM capacity, flash or PSRAM bandwidth, ESP-IDF
scheduling, or PIE instruction encoding/ABI. Capability allocations use
malloc, PSRAM ones charged against `--psram-bytes`; timing uses a steady
clock. RMSNorm preserves the assembly's fused accumulation
and two Newton steps but uses a correctly rounded seed instead of `rsqrt0.s`,
so the hardware's last-ulp seed differences still need a device check.
`-ffp-contract=off` evaluates fp32 expressions as written. The firmware's GCC
(gnu++2b, so the default `-ffp-contract=fast`) fuses some of them into
`madd.s`/`msub.s`: the story build has dozens in `esp32_backend.cpp`, most of
them in the fast_math activation epilogues (`apply_act`), and more in
`espressif_ops.cpp`.
Those fp32 results can differ from the chip in the last ulp; the integer kernels
are exact. Code that spells each fusion as an explicit `fmaf` is the exception:
`espressif::attn_tail` (its table exp `fast_exp_neg` and its P@V) computes the
same bits here as on the chip.
The Python device mirrors follow the chip, not this build: every quantised
loop's `y += acc * sx * scale` is one `madd.s` there, and `models.addmm_codes_ref`
and the fused activations (`fast_tanhf`, `fast_gelu`, SiLU, ...) fuse where the
firmware does, so `run_parity.py` holds the delegated rows to the chip's output
bits while the model tests here hold this build to the same mirrors at 2e-6.
`-DESP32_HOST_FP_CONTRACT=fast` (default `off`) builds the runtime, portable
kernels, delegate and ops with fusion allowed, to bracket that difference;
clang chooses its own fusions, not GCC's, and the model tests assume `off`.

`espressif_ops.cpp` registers the custom ops and target overrides of eleven
standard ops (softmax, add, bmm, div, index_put_, le, mul, permute_copy,
remainder, split_with_sizes_copy, where) through
`espressif::register_ops()`, as on the firmware, and three more where a program
names them: ExecuTorch's per-tensor quantize and dequantize and avg_pool2d
(`ops/qdq_pool.h`, the operators a quantised CNN leaves between delegates). The
firmware compiles and registers each of those three only for a program whose
op list names it, as its portable kernel is the fallback that must link
(`gen_kernels.py` CONDITIONAL_OVERRIDES writes the `Esp32KernelVariant.h`
macros); the host links every kernel and registers all three. ExecuTorch
aborts on a second registration of an operator, so the firmware's
`gen_kernels.py` leaves its OVERRIDES out of the portable registration. The
host build does the same without a second list: `esp32_host_overrides` links
the op library, reads the `aten::` and `quantized_decomposed::` names from its
registration table, and the codegen drops exactly those from the portable
selection. Beside every portable op, the codegen registers
ExecuTorch's own static quantize/dequantize kernels (per tensor and per
channel, from `kernels/quantized/quantized.yaml`; `codegen_portable.py`
QUANTIZED), as the firmware registers each for a program that names it: the
safety net for the pairs a static int8 graph leaves between portable ops, and
the per-tensor dequantize by a scale and zero point in one-element tensors,
`dequantize_per_tensor.Tensor_out`. That one takes a float64 scale and an
int64 zero point, as PyTorch's choosers return them, and aborts on any other
dtype. Never a chooser; inf and NaN are out of their contract, and their arm64
NEON path differs from the chip's scalar one only there. The host builds the
default variant; the opt-in
fast LayerNorm (`ESP32_FAST_LAYERNORM`) is not registered here, but its row
function is built in every variant and checked.
`esp32_host_runner --selftest`
runs the library's device self-checks (the RMSNorm kernel's domain refusals
with their scalar fallback, layer_norm2d's refusals, the fast LayerNorm's rows
against the portable kernel's: the same mean and rstd bits, outputs within the
reciprocal multiply's rounding) against the C kernels,
and the overrides' fast paths against ExecuTorch's portable kernels: each
case runs both on the same hand-built tensors (decode shapes, IEEE special
values, near misses that must fall back) and compares every written byte, the
failure state and the path taken. CTest runs it, `test_portable_fastpaths.py`
checks its per-operator counts, and QEMU runs the same cases on the chip's
build (`run_parity.py`). It also runs the table op's conformance check
(`quant_lut_check.cpp`): espressif::quant_lut's quantize step on every rounding
tie and its neighbours against a nearbyintf reference and torch's own
quantize_per_tensor (`quant_lut_golden.h`, which `test_quant_lut.py`
regenerates and compares), its output bits, and its malformed-argument
refusals (CTest `espressif_quant_lut`). It holds every quantiser that has a
portable reference to ExecuTorch's rule for NaN and the infinities, against
ExecuTorch's own quantize kernel (`../runtime/nonfinite_check.cpp`, CTest
`espressif_nonfinite`), and sets and reads back the weight-tier options (CTest
`esp32_weight_tier_options`). Every run ends with the overrides'
`<op> dispatch:` lines, as the firmware prints them.
`esp32_host_runner --qdq-sweep quick|full` holds `qdq_pool.h`'s fast loops to
the portable kernels far past the conformance cases: the quantize over a
stride of every float32 (`full`: all 2^32 of them, about a minute on four
threads) and every rounding tie's neighbourhood at 21 scale and zero-point
sets, plus random scales, against both the portable kernel (arm64 NEON for
8-bit codes) and a transcription of the scalar loop the chip runs; the
dequantize over every code; random average pools through the shim (CTest
`espressif_qdq_sweep` runs `quick`).

`esp32_bench_host`, built beside the runner, is the bench firmware's runner
(`examples/espressif/esp32s3/main/bench_runner.inc`) compiled for the host:
`export_bench.py` runs every bench image through it before a board is
involved (`examples/espressif/esp32s3/README.md`). `--read-only` maps the
image and its weights in pages a store cannot change, as the board maps its
partitions from flash: a store into the program or a weight then faults and
names the mapping and the offset (`BENCH_FAULT`), where the board would reject
or drop it, and the flash measurement reads every weight in place.
`--min-latency` turns the projection split on before anything loads, as the
dual-core firmware does (no convolution splits); `--single-core`, `--force-scalar`, `--no-conv-fast`,
`--no-qdq-fast` and `--no-ew-fast` (every loop of `ops/elementwise.h` off: the
kernels they replace, the same bits) are the runner's switches (`test_bench.py`
runs the bench models read-only in each). The host runner itself maps the program and the
named data read-only (ExecuTorch's `MmapDataLoader`), so a run with the
weight tiers off (the default) reads every weight there.
