/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Runtime half of the ESP32-S3 ExecuTorch backend.
//
// Consumes the blob produced by Esp32Backend.preprocess(). Weights are read in
// place, from the blob (in the .pte) or the program's named data, both mapped from flash:
// init() keeps pointers; FLAG_STAGE_I8 expands at init, weight_tier_budget copies at init or ranked placement.
//
// On the loop order: M is 1 for a decode step, so the natural nesting is k-outer,
// n-inner, which walks the [K,N] weight strictly sequentially. The obvious n-outer
// version strides by N and wastes most of every cache line. Choosing this is exactly
// the sort of decision a backend gets to make and an op-by-op kernel does not.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>

#include "fast_math.h"

// The PIE kernels (int8 dot, int4 unpack, fused int4 dot), each proven
// bit-exact against scalar C.
#include "esp32_kernels.h"

// ---- firmware override points ----------------------------------------------
// Weak no-ops bracketing every delegate call, so a firmware target can toggle a
// GPIO, start a cycle counter, or raise a trace marker around inference without
// patching this file. The Arm backend does the same with
// EthosUBackend_execute_begin/end.
//
// Worth having because esp_timer gives microseconds from inside the CPU that
// is also doing the work, and a pin edge on a logic analyser is an
// INDEPENDENT measurement of the same interval. When the two disagree, that
// disagreement is information.
//
// NOT adopted: Arm's `arm_ethos_io_memcpy`. That exists because the Ethos-U
// driver requires inputs and outputs shuffled into a scratch region, so there
// is a large, hot, redirectable copy to hand to DMA. This delegate has no such
// indirection -- weights are read in place from the flash mmap and outputs are
// written straight into ExecuTorch-planned buffers -- and the only memcpys in
// this file are 4-byte header reads. A hook with nothing to route would be
// decoration.
extern "C" void __attribute__((weak)) esp32_backend_execute_begin(void) {}
extern "C" void __attribute__((weak)) esp32_backend_execute_end(void) {}

namespace {
/// Calls the firmware hooks around one delegate call, on every exit path.
struct ExecuteMarker {
  ExecuteMarker() { esp32_backend_execute_begin(); }
  ~ExecuteMarker() { esp32_backend_execute_end(); }
};
}  // namespace

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/named_data_map.h>

#include "esp32_platform.h"
#include "esp32_hot.h"
// The asymmetric-quantisation records: their wire values, and the cold,
// -ffp-contract=off translation unit that validates and attaches them.
#include "esp32_asym.h"
#include "esp32_float.h"
#include "esp32_weight_tiers.h"
#if ESP32_CONV_FAST
// The static int8 convolution path, a -O2 unit of its own (esp32_conv.h).
#include "esp32_conv.h"
#endif
#include "quant_rules.h"

#include <new>
#include <variant>

// ---- profiling -------------------------------------------------------------
// Two mechanisms, because they answer different questions and have very
// different costs to enable.
//
// ETDump (ET_EVENT_TRACER_ENABLED) is ExecuTorch's standard: per-delegate-op
// events that devtools already knows how to render alongside every other
// backend's. It needs the devtools sources and flatcc linked in, which is a
// build the firmware does not do by default, so every hook below compiles to
// nothing unless that flag is set. Macros are defined locally because upstream
// does not export them -- the Arm backend defines its own the same way.
//
// ESP32_BACKEND_TIMING is the one the hardware runs use: per-opcode and
// per-shape microsecond totals from esp_timer, no extra dependency, reported
// through esp32_backend_report_timing(). Measuring on the board should not be
// blocked on getting flatcc to cross-compile.
#if defined(ET_EVENT_TRACER_ENABLED)
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/event_tracer_hooks.h>
namespace {
class EventTraceScope {
 public:
  EventTraceScope(::executorch::runtime::EventTracer* t, const char* name)
      : tracer_(t) {
    if (tracer_ != nullptr) {
      entry_ = tracer_->start_profiling(name);
    }
  }
  ~EventTraceScope() {
    if (tracer_ != nullptr) {
      tracer_->end_profiling(entry_);
    }
  }

 private:
  ::executorch::runtime::EventTracer* tracer_;
  ::executorch::runtime::EventTracerEntry entry_{};
};
}  // namespace
#define ESP32_PROF_SCOPE(TRACER, NAME) EventTraceScope _et_scope((TRACER), (NAME))
#else
#define ESP32_PROF_SCOPE(TRACER, NAME) ((void)0)
#endif

#if defined(ESP32_BACKEND_TIMING)
#define ESP32_TICK_NOW() ((uint64_t)esp_timer_get_time())
#else
#define ESP32_TICK_NOW() ((uint64_t)0)
#endif

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptionContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

constexpr uint32_t kMagic = 0x45333244; // 'E32D'
// Must equal VERSION in serialization/espressif_blob.py, which lists what a
// blob of this version may hold: any change to what a blob means bumps both,
// so a skewed pair refuses at load instead of computing something else. The
// asymmetric-quantisation flags, records and opcode are in esp32_asym.h.
constexpr uint32_t kVersion = 24;
constexpr uint32_t kFlagExternalWeights = 1;
// Scales at scale_off are IEEE half rather than float. Set independently of n_groups:
// the two coincide because fp16 is worth its decode cost only when there are groups
// to pay for -- a consequence, not an invariant to depend on.
constexpr uint32_t kFlagFp16Scales = 2;
// Fused output clamp, interval in the descriptor's clamp_lo/clamp_hi (+/-inf for open
// sides; relu is [0, inf)). Runs in registers between accumulate and store, so the
// activation -- or a whole ACCIDENTAL CHAIN of them, composed AOT into one interval --
// costs zero memory traffic and no kernel dispatch.
constexpr uint32_t kFlagClamp = 4;
// Fused output activations, applied BEFORE the clamp (their producing composition is
// clamp(act(mm))). SiLU is sigmoid-gating its own input; GELU is the exact erf form
// ('none' approximate; the tanh variant is kFlagActGeluTanh below).
constexpr uint32_t kFlagActSilu = 8;
constexpr uint32_t kFlagActGelu = 16;
// Fused residual add: y += args[res_arg], applied AFTER the act/clamp epilogue --
// the composition add(clamp(act(mm)), r) is what the AOT gate fuses. fp32 a+b is
// symmetric, so which side of the eager add the matmul sat on changes no bit.
constexpr uint32_t kFlagAdd = 32;
// Activations quantise at the descriptor's CALIBRATED act_scale instead of per-row
// absmax. Reproducibility, not speed: a decode loop's per-row scale shifts with
// every KV rounding tie; a calibrated one cannot. Out-of-calibration values clip,
// the standard static-quant trade, chosen per model at export.
constexpr uint32_t kFlagStaticAct = 64;
// A convolution, executed as an im2col matmul. A FLAG rather than a new opcode
// deliberately: the opcode already selects the weight format, so conv inherits
// i4/i8/i16 with nothing new to validate, name or test. The geometry the
// descriptor cannot express arrives in an EXT_CONV3D record.
constexpr uint32_t kFlagConv = 128;
// The activation arrives already quantised per TENSOR by a PT2E
// quantize/dequantize pair, and this op honours that scheme instead of its own
// per-row absmax: one scale for the whole activation, computed here rather than
// baked, because `choose_qparams_symmetric` computes it at runtime too.
//
// A flag the runtime must know, never a silently-ignored bit: a runtime that
// ignored it would compute per-row scales and return plausible numbers under a
// different quantisation scheme than the model was trained with -- exactly the
// failure the version and known-flag checks exist to prevent.
constexpr uint32_t kFlagActPerTensor = 256;
// Glimmer's terminal cap*tanh((scale*y)/cap). The descriptor's clamp fields are
// a tagged union under this mutually-exclusive flag: clamp_lo is scale and
// clamp_hi is the positive cap. Keep the four fp32 operations separate and in
// source order so fusion does not silently change model numerics.
constexpr uint32_t kFlagSoftcap = 512;
// External i4 stays packed in PTD but is expanded once into an aligned int8
// PSRAM allocation at init. Numerically identical to per-row scratch unpack.
constexpr uint32_t kFlagStageI8 = 1024;
// Selective dual-core posture: fork this addmm, while every ordinary projection
// stays on the inference core. This is distinct from global min_latency.
constexpr uint32_t kFlagHeadSplit = 2048;
// Fused multiply: y = epilogue(acc) * R, where a fused residual would be added
// (after act and clamp), and never together with one -- res_arg names R. R is
// a partition input (the attention output an output gate scales) or this op's
// OWN output: for SwiGLU/GeGLU the partition's earlier projection wrote
// act(gate) there, and up multiplies it in place, so the [M, F] product needs
// no buffer of its own. In place is sound because every loop implementing the
// flag reads R[m, n] immediately before it stores y[m, n] to the same index.
// The fp32 loop accumulates into the output row and cannot; it is refused.
constexpr uint32_t kFlagMul = 4096;
// More fused output activations, mutually exclusive with SiLU, GELU and each
// other. Each repeats its portable kernel's operation order in fp32, with only
// expf/tanhf replaced by fast_math.h.
constexpr uint32_t kFlagActSigmoid = 8192;
constexpr uint32_t kFlagActGeluTanh = 16384;
// x * clamp(x + a, 0, 6) / d and clamp(x + a, 0, 6) / d, with a and d in
// clamp_lo/clamp_hi (3 and 6 for torch's own): the clamp fields are a tagged
// union here, as under kFlagSoftcap, so these exclude kFlagClamp.
constexpr uint32_t kFlagActHardswish = 32768;
constexpr uint32_t kFlagActHardsigmoid = 65536;
// leaky_relu. clamp_lo carries the negative slope, the same tagged union.
constexpr uint32_t kFlagActLeaky = 131072;
constexpr uint32_t kFlagActRelu2 = 262144;
// Asymmetric: an op's activation quantiser, weight zero points/offsets/sums, and
// output quantize, each with its required record (esp32_asym.h). Known here
// so that validate_blob passes them on; esp32_asym::validate owns their rules.
using esp32_asym::kFlagActQuant;
using esp32_asym::kFlagWeightAffine;
using esp32_asym::kFlagOutQuant;
constexpr uint32_t kKnownFlags =
    kFlagExternalWeights | kFlagFp16Scales | kFlagClamp | kFlagActSilu |
    kFlagActGelu | kFlagAdd | kFlagStaticAct | kFlagConv |
    kFlagActPerTensor | kFlagSoftcap | kFlagStageI8 | kFlagHeadSplit |
    kFlagMul | kFlagActSigmoid | kFlagActGeluTanh | kFlagActHardswish |
    kFlagActHardsigmoid | kFlagActLeaky | kFlagActRelu2 | kFlagActQuant |
    kFlagWeightAffine | kFlagOutQuant;
// At most one of these per op: apply_act implements exactly one.
constexpr uint32_t kActFlags =
    kFlagActSilu | kFlagActGelu | kFlagSoftcap | kFlagActSigmoid |
    kFlagActGeluTanh | kFlagActHardswish | kFlagActHardsigmoid |
    kFlagActLeaky | kFlagActRelu2;
// Anything between accumulate and store. An op with none of these may take a
// plain loop that stores the scaled accumulator directly.
constexpr uint32_t kEpilogueFlags =
    kActFlags | kFlagClamp | kFlagAdd | kFlagMul;

/// The gather's geometry. uint16 on the wire; widened here because every use is
/// an index computation and mixing widths in those is how sign-extension bugs
/// get in.
struct ConvGeom {
  uint32_t h_in, w_in, c_in;
  uint32_t h_out, w_out, c_out;
  uint32_t kh, kw;
  uint32_t stride_h, stride_w;
  uint32_t pad_h, pad_w;
  uint32_t dil_h, dil_w;
  // Depth, so ONE geometry and ONE gather serve 1-D, 2-D and 3-D. A conv2d has
  // d_in=d_out=kd=1 and a conv1d additionally h=kh=1, and the loops over those
  // axes then run once and contribute nothing.
  uint32_t d_in, d_out, kd, stride_d, pad_d, dil_d;
  // groups == c_in == c_out is DEPTHWISE, which takes a different path: see
  // the note on kOpDepthwise below.
  uint32_t groups;
  // Which group this op computes; its channel slices start here.
  uint32_t group_idx;
  // Full output channel count of the tensor this op writes into. Stored, not
  // derived: a grouped slice has c_out*groups and a depthwise op has c_out,
  // and a shape check that guesses between them gets one of the two wrong.
  uint32_t c_out_total;
  // Transposed: the gather maps taps through a zero-stuffed grid instead of
  // striding the input. Never materialised -- see axis_index.
  uint32_t transposed;
};

/// Dynamic output-row selection for a constant full projection table.
/// The descriptor keeps n == complete table height for storage validation;
/// this record names the int64 row-ID argument and the fixed output width.
struct SelectedRows {
  uint32_t rows_arg;
  uint32_t count;
};

static inline bool is_depthwise(const ConvGeom& g) {
  return g.groups > 1 && g.c_in == 1 && g.c_out == g.groups &&
      g.c_out_total == g.c_out;
}

ESP32_HOT_CODE
static inline float apply_act(uint32_t flags, float p0, float p1, float y) {
  if (flags & kFlagSoftcap) {
    y *= p0;
    y /= p1;
    y = fast_tanhf(y);
    y *= p1;
  } else if (flags & kFlagActSilu) {
    y = y / (1.0f + fast_expf(-y));
  } else if (flags & kFlagActGelu) {
    y = fast_gelu(y);
  } else if (flags & kFlagActSigmoid) {
    y = 1.0f / (1.0f + fast_expf(-y));
  } else if (flags & kFlagActGeluTanh) {
    // ExecuTorch's portable gelu(approximate='tanh'), term for term in fp32
    // (the portable kernel finishes in double; the difference is one rounding).
    if (y == -INFINITY) {
      y = 0.0f;
    } else if (y != INFINITY) {
      const float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
      const float kKappa = 0.044715f;
      const float cube = y * y * y;
      const float inner = kBeta * (y + kKappa * cube);
      y = 0.5f * y * (1.0f + fast_tanhf(inner));
    }
  } else if (flags & (kFlagActHardswish | kFlagActHardsigmoid)) {
    // torch's decompositions, rounding for rounding: clamp(y + p0, 0, 6) / p1,
    // times y first for hardswish -- (y * t) / p1, not y * (t / p1). p0, p1
    // are the graph's own constants, 3 and 6 for torch's hardswish.
    float t = y + p0;
    t = t < 0.0f ? 0.0f : t;
    t = t > 6.0f ? 6.0f : t;
    y = (flags & kFlagActHardswish) ? (y * t) / p1 : t / p1;
  } else if (flags & kFlagActLeaky) {
    y = y >= 0.0f ? y : y * p0;
  } else if (flags & kFlagActRelu2) {
    // relu then square, NaN-propagating like the portable relu; r * r is the
    // correctly rounded pow(r, 2).
    y = y < 0.0f ? 0.0f : y;
    y = y * y;
  }
  return y;
}

// The fused residual add or multiply, after act and clamp.
static inline float apply_operand(uint32_t flags, float y, float r) {
  return (flags & kFlagMul) ? y * r : y + r;
}
constexpr uint32_t kOpAddmm = 1;
constexpr uint32_t kOpAddmmI8 = 2;
constexpr uint32_t kOpAddmmI4 = 3;
constexpr uint32_t kOpAddmmI16 = 4;
constexpr uint32_t kOpEmbedI8 = 5;
constexpr uint32_t kOpEmbedI4 = 6;
constexpr uint32_t kOpEmbedI16 = 7;
// Opt-in int8 q@kT: BOTH operands quantised dynamically at execute, no weights.
// Descriptor reuse, per the AOT note: res_arg is the SECOND INPUT's arg index (a
// bmm has no residual) and scale_group is the BATCH count (no stored scales).
// The K side arrives in its natural [B,N,K] layout -- the transpose was folded
// -- so key vectors are contiguous rows on both sides of the dot.
constexpr uint32_t kOpBmmDynI8 = 8;
// Opt-in int8 q@kT against a PERSISTENT int8 K cache (kcache_i8.py). The K side
// arrives already quantised: int8 codes [B,N,K] at res_arg, and one fp32 scale
// per key row [B,N] at the arg index carried in scale_off -- descriptor reuse,
// the scales are a runtime tensor, not stored bytes. Each key was quantised once
// when it was written, with the arithmetic kOpBmmDynI8 repeats on every row at
// every call, so the scores are bit-identical; only the query rows quantise
// here. scale_group is the batch, as for kOpBmmDynI8.
constexpr uint32_t kOpBmmKcacheI8 = 9;
// q@kT against an asymmetric int8 K cache (ExecuTorch's
// QuantizedKVCache). kOpBmmKcacheI8's descriptor plus the zero points' ARG
// index in wt_off; esp32_asym::validate owns what differs.
using esp32_asym::kOpBmmKcacheAffineI8;

// The q@kT ops: no stored weights, a second activation, a batch in
// scale_group. esp32_asym::validate checks what the affine one adds before
// validate_blob checks what every bmm shares.
static inline bool is_bmm(uint32_t opcode) {
  return opcode == kOpBmmDynI8 || opcode == kOpBmmKcacheI8 ||
         opcode == kOpBmmKcacheAffineI8;
}

// ---- blob extensions --------------------------------------------------------
// OpDesc is FROZEN at 64 bytes. Anything an op needs beyond it
// lives in a typed variable-length record in a side table, so a
// new field costs bytes only on the ops that use it. Growing the
// descriptor instead would tax every op in every model to carry a
// field most never read; Vela's format is a stream of named
// variable-length blocks for exactly this reason.
//
// At ext_off: [count u32] then count records of
//   [op_index u16][type u16][len u32][payload, padded to 4]
//
// The type's HIGH BIT means optional. An unknown optional record is skipped; an
// unknown REQUIRED one is refused, because silently ignoring a required
// extension runs a different computation than the blob describes -- the failure
// mode the blob's version field also exists to prevent.
constexpr uint32_t kHeaderBytes = 16;
constexpr uint32_t kOpDescBytes = 64;
constexpr uint32_t kExtCountBytes = 4;
constexpr uint32_t kExtRecordHeaderBytes = 8;
constexpr uint16_t kExtOptional = 0x8000;
// Types 1-5 are unassigned: a payload that changes shape takes a new type
// rather than a silently different length, so a record of an unassigned type
// hits the unknown-required path and is refused, which is the extension
// mechanism doing exactly what it exists for.
constexpr uint16_t kExtConv3d = 6;   // payload: 24 x uint16, see ConvGeom
constexpr uint32_t kConvGeomBytes = 48;
constexpr uint16_t kExtSelectedRows = 7;  // payload: SelectedRows
constexpr uint32_t kSelectedRowsBytes = 8;
// Asymmetric records, payloads in esp32_asym.h: ActQuant, WeightAffine, OutQuant.
using esp32_asym::kExtActQuant;
using esp32_asym::kExtWeightAffine;
using esp32_asym::kExtOutQuant;
// Highest assigned type, used only to synthesize an unknown test record. Types
// are NOT a dense range: 1--5 are unassigned and must be refused when
// required rather than mistaken for known merely because they are below 6.
constexpr uint16_t kExtMaxKnown = kExtOutQuant;

static inline bool is_known_extension(uint16_t type) {
  return type == kExtConv3d || type == kExtSelectedRows ||
      type == kExtActQuant || type == kExtWeightAffine || type == kExtOutQuant;
}
// The float-weight rungs (esp32_float.h): fp16/bf16 matmul (11, 12) and
// embedding (13, 14).
using esp32_float::kOpAddmmF16;
using esp32_float::kOpAddmmBf16;
using esp32_float::kOpEmbedF16;
using esp32_float::kOpEmbedBf16;
constexpr uint32_t kNumOpcodes = 15;

/// Stable, static per-opcode names. Static storage matters twice: ETDump's
/// start_profiling_delegate copies the string but the pointer must be valid at
/// the call, and building a name per op would allocate inside the hot loop.
const char* opcode_name(uint32_t opcode) {
  switch (opcode) {
    case kOpAddmm:     return "esp32::addmm_f32";
    case kOpAddmmI8:   return "esp32::addmm_i8";
    case kOpAddmmI4:   return "esp32::addmm_i4";
    case kOpAddmmI16:  return "esp32::addmm_i16";
    case kOpEmbedI8:   return "esp32::embed_i8";
    case kOpEmbedI4:   return "esp32::embed_i4";
    case kOpEmbedI16:  return "esp32::embed_i16";
    case kOpBmmDynI8:  return "esp32::bmm_dyn_i8";
    case kOpBmmKcacheI8: return "esp32::bmm_kcache_i8";
    case kOpBmmKcacheAffineI8: return "esp32::bmm_kcache_affine_i8";
    case kOpAddmmF16:  return "esp32::addmm_f16";
    case kOpAddmmBf16: return "esp32::addmm_bf16";
    case kOpEmbedF16:  return "esp32::embed_f16";
    case kOpEmbedBf16: return "esp32::embed_bf16";
    default:           return "esp32::unknown";
  }
}

constexpr uint32_t kVecBytes = 16;
// The longest integer dot one int32 accumulator holds exactly. Activation
// codes are clamped to +/-127 and stored weight codes are int8, so no product
// exceeds 127*128 in magnitude; dot_i8_s3 returns only ACCX's low 32 bits and
// the scalar loops accumulate in int32, so a longer row would wrap -- silently.
// A multiple of 32, so every k_pad alignment rounds within it.
constexpr uint32_t kMaxDotLen = 132096;
static_assert(127ull * 128ull * kMaxDotLen <= 0x7FFFFFFFull &&
                  kMaxDotLen % 32 == 0,
              "every int8 dot up to kMaxDotLen must be exact in int32");
// Narrowest output width worth forking under the min-latency mode: below this
// the fork/join handshake costs more than the columns it moves. 32 is the board's
// choice: on the story decode 64 columns gained nothing over 32.
// ESP32_SPLIT_MIN_COLS moves it: the backend component's
// Kconfig sets it from CONFIG_ESP32_BACKEND_SPLIT_MIN_COLS.
#ifndef ESP32_SPLIT_MIN_COLS
#define ESP32_SPLIT_MIN_COLS 32
#endif
constexpr uint32_t kSplitMinCols = ESP32_SPLIT_MIN_COLS;

/// Row blocking: activation rows quantised together before one pass over the
/// weights. The block is decided PER OP by `row_block_for()` below from two
/// constants, mirrored in operator_support/conv.py and read out of this
/// file by test_conv_support.py:
///
///   kRowBlockMax           the most rows one block may hold;
///   kRowBlockScratchBytes  the per-call scratch one block may cost, charged
///                          k_pad bytes per row for the quantised activation
///                          plus, for a conv, the fp32 staging row the NCHW
///                          scatter needs.
///
/// A budget, not a fixed row count: a fixed count such as 8 rows
/// would be the binding limit on prefill flash traffic for narrow
/// ops, where a 32-row block of a K=96 op costs only 3 KB: the
/// constant, not memory, would cap the pass count at ceil(M/8).
/// The budget form keeps the wide-K ops (the 784-wide PLE
/// projections, a 3x3x128 conv) at or above 8 rows while letting
/// narrow ones block the whole prompt. The weight traffic of an op
/// is exactly k_pad*n*ceil(m/B), which is why B is the parameter
/// that decides prefill speed and why it is reported rather than
/// trusted (see the AOT emit log).
constexpr uint32_t kRowBlockMax = 32;
constexpr uint32_t kRowBlockScratchBytes = 16 * 1024;

/// Compile-spec key for the opt-in min-latency (dual-core) posture. Must match
/// SPEC_MIN_LATENCY in compile_spec.py -- the AOT/runtime contract for this
/// name is exactly as load-bearing as the blob layout.
constexpr const char* kSpecMinLatency = "min_latency";
/// Compile-spec key for Stage B (stage_b.py): a delegate built
/// under it may carry int8 activation and output codes (EXT_ACT_QUANT
/// in_dtype 1, EXT_OUT_QUANT out_dtype 1). Must match SPEC_STAGE_B.
constexpr const char* kSpecStageB = "stage_b";
// Upper bound on ops sharing one activation-quant group. Grouping uses fixed stack
// arrays at execute time, so the bound is a real limit: members past it simply stay
// standalone, which is correct and merely re-quantises. qkv is 3, SwiGLU pairs are 2.
constexpr uint32_t kMaxActGroup = 8;
// THREE chain scratch slots, not two. The sentinels are the top values of the
// arg space, so the base moves down as slots are added -- which changes what
// 0xFFFFFFFE means and is therefore a version bump, not a compatible addition.
//
// Three because fusing activations onto convolutions merges more ops per
// partition, and a detection model's C2f block then wants three live
// intermediates (with two, preprocess refuses and the whole export fails).
// Slots are allocated on DEMAND below, so a third costs nothing on a model
// that does not use it.
constexpr uint32_t kNumScratch = 3;
constexpr uint32_t kArgScratch0 = 0xFFFFFFFFu - (kNumScratch - 1);

static inline bool is_embed(uint32_t opcode) {
  return opcode == kOpEmbedI8 || opcode == kOpEmbedI4 ||
      opcode == kOpEmbedI16 || esp32_float::is_embed(opcode);
}
// esp32_codes.h's half_to_float is this function for its contract-off units: change both.
// IEEE half -> float. Grouped scales are stored fp16 because at the deployed config
// fp32 scale tables do not fit. This applies to both sparse embeddings and ordinary
// grouped i4/i8 addmm: group_weights=True deliberately emits the same group-128/fp16
// format the training QAT used.
//
// Done by hand rather than with _Float16: this has to be exact on a target whose fp16
// support is not guaranteed by the toolchain, and the conversion runs once per group
// per gathered row, not per weight.
ESP32_HOT_CODE
static inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      // Subnormal: shift the mantissa up until the implicit bit appears, and take the
      // exponent down by however far that was.
      uint32_t m = mant;
      int e = -1;
      do {
        m <<= 1;
        ++e;
      } while ((m & 0x400u) == 0);
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  // A union, not memcpy: at -Os GCC leaves this 4-byte memcpy as a ROM call
  // (and the decode out of line), once per fp16 scale group, as fast_math.h
  // notes for fast_expf.
  const union {
    uint32_t bits;
    float value;
  } pun = {bits};
  return pun.value;
}

// Resolved::scale is a raw view into either inline or named data. Keep that field
// float-typed for the overwhelmingly common fp32 case, but decode through the flag
// at every grouped consumer: reading half bytes as float misreads the grouped addmm
// artifacts that AOT intentionally emits.
ESP32_HOT_CODE
static inline float stored_scale(uint32_t flags, const float* scales,
                                 size_t index) {
  if (flags & kFlagFp16Scales) {
    return f16_to_f32(reinterpret_cast<const uint16_t*>(scales)[index]);
  }
  return scales[index];
}

// k_pad is in ELEMENTS. How many bytes that is depends on the rung of the format
// ladder, which is what the opcode selects. Getting this wrong reads the right number
// of the wrong things, so it is one function rather than an expression at each site.
// Keep serialized-size arithmetic wider than the ESP32-S3's 32-bit size_t.
// Callers narrow only after validate_blob has established that the complete
// stored tensor is target-addressable.
static inline uint64_t row_bytes(uint32_t opcode, uint32_t k_pad) {
  switch (opcode) {
    case kOpAddmmI4:
    case kOpEmbedI4:
      return (uint64_t)k_pad / 2;
    case kOpAddmmI16:
    case kOpEmbedI16:
      return (uint64_t)k_pad * 2;
    default:
      return (size_t)k_pad;
  }
}

struct OpDesc {
  uint32_t opcode;
  // kFlagExternalWeights: wt_off/scale_off are NUL-terminated NamedDataMap keys inside
  // the blob rather than the payloads themselves. That is what lets a 12.58 MB weight
  // live in a flash partition instead of the 1.5 MB app image. A float weight has no
  // scales: wt_off alone is a key and scale_off is 0 (float_weights()).
  uint32_t flags;
  uint32_t m, k, n;
  uint32_t k_pad;
  uint32_t bias_off;
  uint32_t wt_off;
  uint32_t scale_off;
  // Index into the delegate's args, which is inputs followed by outputs. Required
  // rather than implied: a partition may hold several ops (SwiGLU's gate and up share
  // an input and produce two outputs), so args[0]/args[1] is wrong in general.
  uint32_t in_arg;
  uint32_t out_arg;
  // Scale-group SIZE in elements; 0 means one scale for the whole row. The SIZE and
  // not the count, because only the size reproduces the quantiser's boundaries on a
  // ragged row: for k=200 the AOT groups are [0,128)+[128,200), and reconstructing
  // ceil(k/count)-sized groups from a count of 2 gives [0,100)+[100,200) --
  // dequantising elements 100..127 with the wrong group's scale, silently, on both
  // the device and the host reference at once, which is why parity cannot see it;
  // with k % 128 == 0 both readings give the same groups, so only a ragged tensor
  // shows the difference.
  uint32_t scale_group;
  // Fused output clamp interval; meaningful when kFlagClamp is set. The ternaries
  // that apply it are NaN-propagating and inf-safe, matching torch.clamp.
  float clamp_lo;
  float clamp_hi;
  // The partition input a fused residual add (kFlagAdd) sums into the output.
  // Index 0 is legitimate, which is why the flag exists rather than a sentinel.
  uint32_t res_arg;
  // Calibrated activation quantisation scale (kFlagStaticAct); 0.0 when dynamic.
  float act_scale;
};
static_assert(sizeof(OpDesc) == kOpDescBytes,
              "OpDesc must match the AOT struct.pack layout");
static_assert(sizeof(SelectedRows) == kSelectedRowsBytes,
              "SelectedRows must match the AOT struct.pack layout");

// Storage and execution can deliberately differ for one deployment policy:
// FLAG_STAGE_I8 keeps the flash image's i4 bytes but points Resolved::wq at an
// expanded int8 PSRAM copy. Keep both meanings named so validation never starts
// sizing the PTD from the runtime representation, or the hot loop vice versa.
static inline bool executes_i4(const OpDesc& op) {
  return op.opcode == kOpAddmmI4 && !(op.flags & kFlagStageI8);
}

static inline uint64_t execution_row_bytes(const OpDesc& op) {
  return (op.flags & kFlagStageI8) ? (uint64_t)op.k_pad
                                  : row_bytes(op.opcode, op.k_pad);
}

/// The row block one op runs at: how many activation rows are quantised
/// together and served by ONE pass over the weights.
///
/// Called from init to size the scratch and from execute to pick the block,
/// so the two cannot disagree: separate expressions for sizing and blocking
/// (and for the AOT emit log) drift apart. A conv charges
/// its fp32 staging row (n floats) on top of the quantised row, so xq plus
/// conv_stage for one op never exceed kRowBlockScratchBytes together. The
/// mirror is `operator_support/conv.py`'s runtime_block(); keep them identical.
static inline uint32_t row_block_for(uint32_t m, uint32_t k_pad,
                                     const ConvGeom* conv, uint32_t n) {
  const uint64_t per_row = (uint64_t)k_pad +
      (conv != nullptr ? (uint64_t)n * sizeof(float) : 0);
  if (m <= 1 || per_row == 0) {
    return 1;
  }
  uint64_t b = kRowBlockScratchBytes / per_row;
  if (b < 1) {
    b = 1;
  }
  if (b > kRowBlockMax) {
    b = kRowBlockMax;
  }
  if (b > m) {
    b = m;
  }
  return (uint32_t)b;
}

// ---- what an op reads and writes, in elements ------------------------------
// Three shapes of op, three answers, needed by a chain check, a tensor-size
// check and a per-tensor absmax alike: named once here rather than re-derived
// inline as a nest of ternaries over the geometry.

/// A conv reads its WHOLE NCHW input: k is the patch size, not a row, and
/// every output row gathers its own patch. `groups` is validated nonzero.
static inline uint64_t conv_input_elems(const ConvGeom& g) {
  return (uint64_t)g.c_in * g.groups * g.d_in * g.h_in * g.w_in;
}

/// Elements of the activation tensor `op` reads.
static inline uint64_t op_input_elems(const OpDesc& op, const ConvGeom* g) {
  if (g != nullptr) {
    return conv_input_elems(*g);
  }
  return (uint64_t)op.m * op.k;          // m rows of k
}

/// Elements of the tensor `op` writes into. A conv writes the whole NCHW
/// output even when this descriptor owns one group's channel slice of it; a
/// gather writes m rows of k (its n is the table height it reads from); a
/// matmul writes m rows of n.
static inline uint64_t op_output_elems(const OpDesc& op, const ConvGeom* g) {
  if (g != nullptr) {
    return (uint64_t)g->c_out_total * op.m;
  }
  if (is_embed(op.opcode)) {
    return (uint64_t)op.m * op.k;
  }
  return (uint64_t)op.m * op.n;
}

// Serialized extents are uint32 in OpDesc and uint16 in ConvGeom, but the
// target's pointers and size_t are 32-bit.  Check products without first
// overflowing the type whose limit we are enforcing.
template <size_t N>
static bool product_fits(const uint32_t (&factors)[N], uint64_t limit,
                         uint64_t* product) {
  uint64_t value = 1;
  for (size_t i = 0; i < N; ++i) {
    if (factors[i] == 0 || value > limit / factors[i]) {
      return false;
    }
    value *= factors[i];
  }
  *product = value;
  return true;
}

constexpr uint64_t kMaxTargetFloatElems = UINT32_MAX / sizeof(float);

static bool conv_axis_matches(uint32_t in, uint32_t out, uint32_t kernel,
                              uint32_t stride, uint32_t pad, uint32_t dil,
                              bool transposed) {
  if (in == 0 || out == 0 || kernel == 0 || stride == 0 || dil == 0) {
    return false;
  }
  if (transposed) {
    // The record carries the extent AFTER output padding, which torch and
    // ExecuTorch both bound below max(stride, dilation): so `out` is the
    // computed extent plus up to max(stride, dil) - 1 positions at the far
    // end. axis_index needs nothing more, because a position past the
    // computed extent reads its taps through the same mapping.
    const int64_t calculated = ((int64_t)in - 1) * stride -
        2 * (int64_t)pad + (int64_t)dil * (kernel - 1) + 1;
    const int64_t output_padding = (int64_t)out - calculated;
    return output_padding >= 0 &&
        output_padding < (int64_t)(stride > dil ? stride : dil);
  }
  const int64_t numerator = (int64_t)in + 2 * (int64_t)pad -
      ((int64_t)dil * (kernel - 1) + 1);
  if (numerator < 0) {
    return false;
  }
  return numerator / stride + 1 == (int64_t)out;
}

static Error validate_conv_geometry(const ConvGeom& g, const OpDesc& op,
                                    uint32_t op_index) {
  // Padding and group_idx may be zero. Every extent, channel count, kernel,
  // stride and dilation is a loop bound or divisor and must be positive.
  if (g.h_in == 0 || g.w_in == 0 || g.c_in == 0 || g.h_out == 0 ||
      g.w_out == 0 || g.c_out == 0 || g.kh == 0 || g.kw == 0 ||
      g.stride_h == 0 || g.stride_w == 0 || g.dil_h == 0 ||
      g.dil_w == 0 || g.d_in == 0 || g.d_out == 0 || g.kd == 0 ||
      g.stride_d == 0 || g.dil_d == 0 || g.groups == 0 ||
      g.c_out_total == 0 || g.transposed > 1) {
    ET_LOG(Error, "esp32: conv op %u has a zero/invalid geometry field",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }

  const bool depthwise = is_depthwise(g);
  if (g.groups == 1) {
    if (g.group_idx != 0 || g.c_out_total != g.c_out) {
      ET_LOG(Error, "esp32: dense conv op %u has group_idx %u or total %u/%u",
             (unsigned)op_index, (unsigned)g.group_idx,
             (unsigned)g.c_out_total, (unsigned)g.c_out);
      return Error::InvalidProgram;
    }
  } else if (depthwise) {
    if (g.group_idx != 0 || g.transposed) {
      ET_LOG(Error, "esp32: depthwise conv op %u has group_idx %u or transpose",
             (unsigned)op_index, (unsigned)g.group_idx);
      return Error::InvalidProgram;
    }
  } else {
    const uint64_t expected_total = (uint64_t)g.c_out * g.groups;
    if (g.group_idx >= g.groups || g.c_out_total != expected_total ||
        g.transposed) {
      ET_LOG(Error,
             "esp32: grouped conv op %u has idx %u/%u, total %u/%llu, or transpose",
             (unsigned)op_index, (unsigned)g.group_idx, (unsigned)g.groups,
             (unsigned)g.c_out_total, (unsigned long long)expected_total);
      return Error::InvalidProgram;
    }
  }
  const uint32_t m_factors[] = {g.d_out, g.h_out, g.w_out};
  const uint32_t k_factors[] = {g.c_in, g.kd, g.kh, g.kw};
  uint64_t geom_m = 0, geom_k = 0;
  if (!product_fits(m_factors, UINT64_MAX, &geom_m) ||
      !product_fits(k_factors, UINT64_MAX, &geom_k) ||
      geom_m != op.m || geom_k != op.k || g.c_out != op.n) {
    ET_LOG(Error,
           "esp32: conv geometry (m=%llu k=%llu n=%u) disagrees with descriptor "
           "(m=%u k=%u n=%u) on op %u",
           (unsigned long long)geom_m, (unsigned long long)geom_k,
           (unsigned)g.c_out, (unsigned)op.m, (unsigned)op.k,
           (unsigned)op.n, (unsigned)op_index);
    return Error::InvalidProgram;
  }

  const uint32_t input_factors[] = {
      g.c_in, g.groups, g.d_in, g.h_in, g.w_in};
  const uint32_t output_factors[] = {g.c_out_total, g.d_out, g.h_out, g.w_out};
  uint64_t input_elems = 0, output_elems = 0;
  if (!product_fits(input_factors, kMaxTargetFloatElems, &input_elems) ||
      !product_fits(output_factors, kMaxTargetFloatElems, &output_elems)) {
    ET_LOG(Error, "esp32: conv op %u tensor extents exceed 32-bit target memory",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }

  if (!conv_axis_matches(g.d_in, g.d_out, g.kd, g.stride_d, g.pad_d,
                         g.dil_d, g.transposed) ||
      !conv_axis_matches(g.h_in, g.h_out, g.kh, g.stride_h, g.pad_h,
                         g.dil_h, g.transposed) ||
      !conv_axis_matches(g.w_in, g.w_out, g.kw, g.stride_w, g.pad_w,
                         g.dil_w, g.transposed)) {
    ET_LOG(Error, "esp32: conv op %u has incoherent input/output geometry",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

// AOT lowers a non-depthwise grouped convolution to one contiguous descriptor
// per group, in group-index order.  The three payload offsets differ because
// each descriptor owns a different weight/bias/scale slice; everything that
// describes the computation and its tensor arguments must be identical.
static bool same_grouped_conv_contract(const ConvGeom& a, const OpDesc& oa,
                                       const ConvGeom& b, const OpDesc& ob) {
  ConvGeom ca = a;
  ConvGeom cb = b;
  ca.group_idx = 0;
  cb.group_idx = 0;

  OpDesc coa = oa;
  OpDesc cob = ob;
  coa.bias_off = cob.bias_off = 0;
  coa.wt_off = cob.wt_off = 0;
  coa.scale_off = cob.scale_off = 0;
  return std::memcmp(&ca, &cb, sizeof(ca)) == 0 &&
      std::memcmp(&coa, &cob, sizeof(coa)) == 0;
}

/// The ops whose weights are the floats they store (opcode 1, the fp16/bf16
/// rungs): no scale table, so under kFlagExternalWeights one key, wt_off.
static inline bool float_weights(const OpDesc& op) {
  return op.opcode == kOpAddmm || esp32_float::is_op(op.opcode);
}

// Required named-data extents. Inline payload validation and external data-map
// validation use the same calculation so the two storage modes cannot drift.
static inline uint64_t weight_bytes(const OpDesc& op) {
  if (esp32_float::is_op(op.opcode)) {
    return (uint64_t)op.k * op.n * sizeof(uint16_t);   // fp16/bf16, [n, k]
  }
  return op.opcode == kOpAddmm
      ? (uint64_t)op.k * op.n * sizeof(float)
      : (uint64_t)op.n * row_bytes(op.opcode, op.k_pad);
}

// Element alignment of each stored payload. The loops read scales, biases,
// fp32 and i16 weights as whole elements, and a misaligned element load faults
// on this target; i8 and packed i4 rows are read a byte at a time unless the
// PIE gate has checked 16-byte alignment itself.
static inline uint32_t weight_align(const OpDesc& op) {
  switch (op.opcode) {
    case kOpAddmm:
      return sizeof(float);
    case kOpAddmmI16:
    case kOpEmbedI16:
    case kOpAddmmF16:
    case kOpAddmmBf16:
    case kOpEmbedF16:
    case kOpEmbedBf16:
      return sizeof(int16_t);
    default:
      return 1;
  }
}

static inline uint32_t scale_align(const OpDesc& op) {
  return (op.flags & kFlagFp16Scales) ? sizeof(uint16_t) : sizeof(float);
}

// Where a staged copy's scales start: after the weight bytes, rounded up so
// they stay aligned even when a row's byte count is odd.
static inline uint64_t staged_scale_offset(uint64_t weight_bytes) {
  return (weight_bytes + sizeof(float) - 1) & ~(uint64_t)(sizeof(float) - 1);
}

static inline uint64_t scale_bytes(const OpDesc& op) {
  if (op.opcode == kOpAddmm || esp32_float::is_op(op.opcode)) {
    return 0;
  }
  const uint32_t span = is_embed(op.opcode) ? op.k : op.k_pad;
  const uint32_t gsize = op.scale_group ? op.scale_group : span;
  const uint64_t ng = ((uint64_t)span + gsize - 1) / gsize;
  return (uint64_t)op.n * ng *
      ((op.flags & kFlagFp16Scales) ? sizeof(uint16_t) : sizeof(float));
}

/// Where an op's weights actually live, resolved once at init: a view into
/// the blob, a NamedDataMap buffer, or a staged PSRAM copy. Owns nothing.
struct Resolved {
  const int8_t* wq;
  const float* scale;
};

constexpr uint32_t kNoScratch = UINT32_MAX;

struct TierLink;  // a handle's place among the live ones (weight tiers)

/// One delegate handle. The descriptors and inline payloads are read in place
/// from the blob (in flash); everything else is resolved at init. Unlike
/// Resolved it OWNS resources -- the held NamedDataMap buffers, staged PSRAM
/// copies and weight-cache references -- which release_resources() frees.
struct CompiledBlob {
  const uint8_t* base;
  const OpDesc* ops;
  uint32_t n_ops;
  // Parsed EXT_CONV3D records, one slot per op, null when the op is not a conv.
  // Resolved once at init so execute never walks the extension table.
  const ConvGeom** conv;
  // Parsed EXT_SELECTED_ROWS records. A non-null entry changes only the output
  // column walk: weights/scales still describe the complete n-row table.
  const SelectedRows** selected;
  // Staging for one block of conv output. addmm_q_row_range writes
  // ybase[row*y_stride + n] -- row stride parameterised, COLUMN stride fixed at
  // 1 -- while a conv output in NCHW needs column stride h_out*w_out and row
  // stride 1, the transpose. Rather than thread a column stride through the hot
  // loop and the dual-core split (the two most verified pieces in this file),
  // each block lands here contiguously and is scattered after. Bounded:
  // xq_rows * max_n floats.
  float* conv_stage;
  int8_t* xq;  // 16 B aligned scratch for the quantised activation
  // Unpacked-weight scratch, allocated only when this partition holds an i4 op. The
  // nibbles expand to int8 here so a row block dots them with dot_i8_s3; a decode
  // row's fused int4 kernel uses it for the deinterleaved activations instead. The
  // expansion is SRAM-side: what halves is the PSRAM/flash read, which is the
  // traffic that bounds these loops.
  int8_t* wu;
  // Second i4 unpack scratch, for the column-split worker: under the min-latency
  // mode the two cores unpack DIFFERENT columns concurrently, so they cannot
  // share one row buffer. max_kpad bytes -- cheap enough that it is allocated
  // whenever wu is, rather than making init depend on a runtime mode.
  int8_t* wu2;
  int16_t* xq16;  // activation scratch for the i16 rung
  // Activation-quant sharing. Ops in one partition reading the SAME input tensor at
  // the same k_pad (i8/i4 rungs only; both quantise activations to +/-127) are grouped
  // at init. At execute the group runs interleaved over m -- quantise the row once
  // into xq, then every member's column loop -- instead of each op re-quantising all
  // M rows. Zero extra scratch, no persistent state, and outputs are disjoint
  // boundary tensors, so no value and no per-tensor write order changes: parity must
  // stay bit-identical. op_group is -1 for standalone ops; members of a group are
  // listed contiguously in group_members and only the FIRST member executes.
  int16_t* op_group;
  uint16_t* group_members;
  uint16_t* group_first;
  uint16_t* group_count;
  uint16_t n_groups;
  // Intra-partition chain scratch: kNumScratch slots, each sized to the widest
  // sentinel-writing op, so a mm -> act -> mm chain runs inside one delegate call
  // with the intermediate never crossing the partition boundary. Only the slots
  // a blob writes are reserved.
  float* chain[kNumScratch];
  uint32_t chain_elems;
  // q@kT scratch: one batch's quantised key vectors ([N, k_pad] int8) and their
  // per-vector scales. Allocated only when the partition holds a dynamic bmm.
  int8_t* bmm_kq;
  float* bmm_sk;
  // Offsets into the per-delegate-call temporary arena. Scratch belongs to an
  // execution, not to a handle: dozens of mutually-exclusive transformer
  // layers must share their peak instead of accumulating it for the method's
  // lifetime. Pointer fields above are populated in a stack-local copy at the
  // start of execute().
  uint32_t scratch_bytes;
  uint32_t xq_off, conv_stage_off, wu_off, wu2_off, xq16_off, sxv_off;
  uint32_t chain_off[kNumScratch];
  uint32_t bmm_kq_off, bmm_sk_off;
  // Row blocking (prefill). xq holds xq_rows quantised activation rows back to
  // back so one pass over the weights can serve all of them; sxv holds their
  // per-row scales. xq_rows == 1 is the decode case.
  uint32_t xq_rows;
  float* sxv;
  // Per-handle configuration, resolved at init: min-latency from this
  // partition's COMPILE SPECS, head-split from its ops' kFlagHeadSplit. It
  // lives here rather than in a global because ExecuTorch may hold
  // several methods -- several delegate handles -- live at once in one process,
  // and "is this model latency-critical" is a property of the model, not of the
  // process. Two handles with different answers must not see each other's.
  bool cfg_min_latency;
  bool cfg_head_split;
  // Whether this handle's load was ranked and left candidates for the
  // placement, so that its TierLink's pending flag was raised. Set at init,
  // before the handle is visible: only such a load raises the flag, which a
  // placement then only clears. Without it, execute need not read the link
  // at all -- a load from the method arena (PSRAM on the story) and an
  // acquire (two memw) per delegate call. Cleared by the handle's own execute
  // once it finds the flag cleared, so a placed ranked handle stops paying
  // for the link too. In the padding before `resolved`, so no field of this
  // struct moves.
  bool tier_deferred;
  // This handle's slot in the placement marks (g_tier_dirty), taken at its
  // staging: a ranked placement marks it, and the handle's own execute, the
  // only writer of its pointers, repoints its ops and clears the mark
  // (weight_tier_sync). kTierNoSlot for a handle without a cache link. In
  // the same padding.
  uint8_t tier_slot;
  Resolved* resolved;
  // Weight cache: per op, the index+1 of the process-wide cache entry this
  // handle holds a reference on (0 = none) -- a PSRAM copy, or a candidate
  // left on flash. Entries are released, never freed, here.
  uint16_t* staged_cache;
  // Buffers fetched from the NamedDataMap. Held for the method's lifetime because the
  // data pointer is only guaranteed valid while its FreeableBuffer lives -- relying on
  // a mmap-backed loader keeping it valid anyway would be depending on an
  // implementation detail of a loader this backend does not own.
  FreeableBuffer* held;
  uint32_t n_held;
  // Per-op asymmetric-quantisation state (records and resolved
  // tables), or null for a blob without any -- which allocates nothing.
  // A side array placed after the hot fields on purpose: a field added to Resolved
  // changes the code of the hot loops that stride over it
  // (run_embedding, fork_split), while one appended here changes only
  // init and execute's handle copy.
  esp32_asym::BlobState* asym;
  // This call's slice of the temporary arena for the affine matmuls' per-row
  // terms (BlobState::terms_off), bound like the scratch pointers above.
  int32_t* asym_terms;
  // The handle's link in the list of live handles a ranked placement walks,
  // or null for one that resolves no weight through the cache. Set at init,
  // cleared only by release_resources: what a placement writes -- the list's
  // links and the pending flag -- lives in the node, so execute's copy of
  // this struct never races with another handle's load or destroy.
  TierLink* tier;
#if ESP32_CONV_FAST
  // The static int8 conv path (esp32_conv.h): its ops here, the temp bytes it
  // adds (the init report), the split worker's scratch (conv_split_scratch).
  uint32_t conv_fast_ops, conv_fast_extra;
  uint32_t conv_split_off;
  int8_t* conv_split;
#endif
};

// Counters so a silent fallback cannot masquerade as a working delegate.
//
// ATOMIC, because execute() can genuinely run on both cores at once (two
// handles, or one handle driven from two tasks) and the parity harness reads
// one of these -- esp32_backend_split_jobs() -- to attribute forks to a
// specific execute. A torn counter would make that attribution silently wrong,
// which is exactly the kind of evidence this file exists to produce. Relaxed
// ordering: these are statistics, never control flow, and no other value is
// published through them.
//
// Note the asymmetry with the PIE/scalar column counts inside
// addmm_q_row_range, which stay PLAIN uint32 locals-by-pointer: the split path
// deliberately gives each core its own pair and merges them at the join, so
// they are never shared while live. That is cheaper than an atomic in the
// innermost loop, and it is why the merge exists.
//
// Every counter is a diagnostic: nothing in this file reads one to decide
// anything. Building with ESP32_BACKEND_STATS=0 makes each an empty Stat
// whose updates compile to nothing and whose reads are zero, so a product
// build pays for no counter on any path, and esp32_backend_report() says the
// counts were not built. The same switch removes the load-time report lines
// (per-op work, self-test results, weight staging) that init() prints. On by default, because the parity harness and the
// device tests prove paths through these counts: that a split forked, that a
// specialised loop ran, that a folded per-tensor scale was honoured.
#ifndef ESP32_BACKEND_STATS
#define ESP32_BACKEND_STATS 1
#endif
#if ESP32_BACKEND_STATS
template <typename T>
using Stat = std::atomic<T>;
#else
template <typename T>
struct Stat {
  constexpr Stat(T) {}
  void operator++() {}
  void operator+=(T) {}
  void fetch_add(T, std::memory_order = std::memory_order_seq_cst) {}
  T load(std::memory_order = std::memory_order_seq_cst) const { return 0; }
  operator T() const { return 0; }
};
#endif

Stat<uint32_t> g_delegate_calls{0};
// Where each call that needs scratch got it (bind_call_scratch): its
// memory-planned buffer; the temp allocator because the program planned none
// (exported without plan_scratch); or the temp allocator because the planned
// buffer could not be used, which means the AOT's scratch_layout.py and this
// file's layout disagree, so it is counted and logged, never overrun.
Stat<uint32_t> g_scratch_planned{0};
Stat<uint32_t> g_scratch_temp{0};
Stat<uint32_t> g_scratch_unusable{0};
// Counted so a folded per-tensor activation cannot be silently ignored: the
// AOT can fold the model's quantise/dequantise away and a runtime that dropped
// the flag would still return plausible numbers, at its own per-row scale.
Stat<uint64_t> g_act_per_tensor{0};
// Per-opcode wall time and invocation count, in microseconds. Plain (not
// atomic) because they are only written under ESP32_BACKEND_TIMING, which is a
// measurement build, and an atomic add per op would perturb the very number
// being measured. Zero-cost when the flag is off: every write sits under
// #if defined(ESP32_BACKEND_TIMING).
uint64_t g_op_us[kNumOpcodes] = {0};
uint32_t g_op_n[kNumOpcodes] = {0};
uint64_t g_execute_us = 0;

#if defined(ESP32_BACKEND_TIMING)
// The same time split by op SHAPE, because per-opcode totals cannot say
// whether a slow matmul family is slow everywhere or only at one size (the
// narrow 256x32 projections pay the same per-op setup as a 768-wide FFN).
// A grouped leader counts its members' columns: the group runs as one op.
struct ShapeTime {
  uint32_t opcode, m, k, n, members, calls;
  uint64_t us, bytes;
};
constexpr int kShapeSlots = 64;  // MobileNetV2 runs 31 shapes
ShapeTime g_shape_time[kShapeSlots] = {};

void record_shape(uint32_t opcode, uint32_t m, uint32_t k, uint32_t n,
                  uint32_t members, uint64_t bytes, uint64_t us) {
  for (int i = 0; i < kShapeSlots; ++i) {
    ShapeTime& t = g_shape_time[i];
    if (t.calls == 0) {
      t = {opcode, m, k, n, members, 0, 0, 0};
    } else if (t.opcode != opcode || t.m != m || t.k != k || t.n != n ||
               t.members != members) {
      continue;
    }
    ++t.calls;
    t.us += us;
    t.bytes += bytes;
    return;
  }
}
#endif

/// Times one delegate op and reports it both ways.
///
/// RAII rather than a start/stop pair because the dispatch loop leaves by
/// several routes -- `continue` for a grouped member already computed, and an
/// error `return` for a bad arg -- and a timer that only stopped on the happy
/// path would silently bill one op's cost to its successor. Empty in the
/// default build: no members, no body, nothing to remove.
struct OpScope {
  OpScope(uint32_t opcode, ::executorch::runtime::EventTracer* tracer) {
#if defined(ESP32_BACKEND_TIMING)
    opcode_ = opcode < kNumOpcodes ? opcode : 0;
    t0_ = ESP32_TICK_NOW();
    shape_ = false;
#endif
#if defined(ET_EVENT_TRACER_ENABLED)
    tracer_ = tracer;
    if (tracer_ != nullptr) {
      entry_ = tracer_->start_profiling_delegate(
          opcode_name(opcode), ::executorch::runtime::kUnsetDelegateDebugIntId);
    }
#endif
#if !defined(ESP32_BACKEND_TIMING) && !defined(ET_EVENT_TRACER_ENABLED)
    (void)opcode;
    (void)tracer;
#endif
  }

  /// Also bill this op to its shape bucket (timing builds only).
  void shape(uint32_t m, uint32_t k, uint32_t n, uint32_t members,
             uint64_t bytes) {
#if defined(ESP32_BACKEND_TIMING)
    shape_ = true;
    m_ = m; k_ = k; n_ = n; members_ = members; bytes_ = bytes;
#else
    (void)m; (void)k; (void)n; (void)members; (void)bytes;
#endif
  }

  ~OpScope() {
#if defined(ESP32_BACKEND_TIMING)
    const uint64_t us = ESP32_TICK_NOW() - t0_;
    g_op_us[opcode_] += us;
    ++g_op_n[opcode_];
    if (shape_) record_shape(opcode_, m_, k_, n_, members_, bytes_, us);
#endif
#if defined(ET_EVENT_TRACER_ENABLED)
    if (tracer_ != nullptr) {
      tracer_->end_profiling_delegate(entry_);
    }
#endif
  }

#if defined(ESP32_BACKEND_TIMING)
 private:
  uint32_t opcode_;
  uint64_t t0_;
  bool shape_;
  uint32_t m_, k_, n_, members_;
  uint64_t bytes_;
#endif
#if defined(ET_EVENT_TRACER_ENABLED)
  ::executorch::runtime::EventTracer* tracer_;
  ::executorch::runtime::EventTracerEntry entry_{};
#endif
};
Stat<uint32_t> g_addmm_ops{0};
Stat<uint32_t> g_embed_ops{0};
Stat<uint32_t> g_ext_keys{0};      // weight/scale blobs resolved from the NamedDataMap
Stat<uint32_t> g_inline_wts{0};    // ... and taken from inside the blob
// Forces the scalar path on every handle, even where alignment permits PIE,
// while > 0; 0 and the initial -1 both leave PIE on wherever alignment allows.
// (Unlike min_latency there is no per-handle force-scalar spec to defer to.)
// A runtime switch, not a build flag, so that one binary under the same
// conditions times PIE against the scalar loops, which compute the same bits:
// at 1 MAC per weight byte a matvec reading flash is memory-bound, and the
// two should time nearly alike. It may flip after init.
// A debug/measurement hook,
// deliberately process-wide. Atomic for the same reason as g_split_override:
// the setter may run on another task.
std::atomic<int> g_force_scalar_override{-1};

// Exact work accounting. Arithmetic intensity decides whether the vector unit can
// matter at all: at 1 MAC per weight byte a matvec is memory-bound unless the memory
// system delivers more bytes/s than the MAC unit consumes, and int8 weights are exactly
// 1 byte per MAC. Reused across M rows the intensity is M, which is the only thing that
// moves it. Counted rather than derived so the numbers cannot drift from the code.
Stat<uint64_t> g_macs{0};
Stat<uint64_t> g_weight_bytes{0};
// Passes over a matmul's weights, actual vs what row-at-a-time dispatch would
// have cost. Their ratio is the prefill traffic saved, and it is COUNTED rather
// than argued: bytes moved is exact even under QEMU, which cannot time anything.
Stat<uint64_t> g_weight_passes{0};
Stat<uint64_t> g_weight_passes_rowwise{0};
// The same thing in BYTES, which is what "traffic" means. Passes are not
// interchangeable: a pass over a 768-wide embedding row and a pass over a
// 96-wide MLP row cost 8x different, so a ratio of pass COUNTS is not a ratio
// of traffic. Accumulated on every quantised path, the activation-sharing
// group path included.
Stat<uint64_t> g_wbytes_actual{0};
Stat<uint64_t> g_wbytes_rowwise{0};
Stat<uint32_t> g_pie_cols{0};      // output columns computed on PIE
Stat<uint32_t> g_scalar_cols{0};   // ... and by the scalar fallback
// Single-row, aligned i8 addmm calls that took the decode-specialized loop.
// Counted so a parity run can prove the intended path ran;
// output parity alone cannot distinguish a correct optimization from a correct
// fallback that never dispatched it.
Stat<uint32_t> g_i8_fast_rows{0};
// Ops that took the affine-weight loops (run_addmm_affine,
// run_embedding_affine): the parity firmware's proof that its zero-point rows
// ran them.
Stat<uint32_t> g_affine_ops{0};

/// Float to int32 rounding toward -inf, in one FPU instruction (FLOOR.S).
/// libm's floorf is a call per element -- two with the caller below
/// out of line -- and every activation and attention key goes through here.
static inline __attribute__((always_inline)) int32_t floor_i(float v) {
  return esp32_platform_floor_i(v);
}

/// Half-away-from-zero, matching weight_formats.py's round_half_away. Deliberately
/// not lrintf, which is round-half-to-even and would disagree on ties. The +0.5f
/// is a float add of its own (rounded to float32 before the floor).
static inline __attribute__((always_inline)) int32_t round_half_away_i(float v) {
  return v < 0.0f ? -floor_i(-v + 0.5f) : floor_i(v + 0.5f);
}

/// Round to nearest, ties to even: torch.round, which PT2E's quantize_per_tensor
/// applies, so a folded PT2E activation must round this way. The fraction
/// v - floor(v) is exact wherever it can be near 0.5 (Sterbenz), so the tie
/// test sees the true value rather than a +0.5f sum that has already rounded.
/// Callers pass v = x / scale with |x| <= 127 * scale, so f + 1 cannot
/// overflow, or NaN: a NaN activation, or an infinite one times the zero
/// inverse of its infinite per-tensor scale. NaN rounds to 0, the zero
/// point's code, as ExecuTorch's quantize_per_tensor gives it (quant_rules.h),
/// before floor_i converts anything.
static inline __attribute__((always_inline)) int32_t round_half_even_i(float v) {
  v = quant_scaled(v);
  const int32_t f = floor_i(v);
  const float frac = v - (float)f;
  return (frac > 0.5f || (frac == 0.5f && (f & 1))) ? f + 1 : f;
}

/// PT2E's choose_qparams_symmetric clamps the scale to its observer's eps,
/// which is FLT_EPSILON by default; strip_dynamic_act_qdq folds only chains
/// that use that default, so the runtime can reproduce it without a field.
constexpr float kPt2eScaleEps = 1.1920928955078125e-07f;  // 2^-23

// How many input-row re-quantisations the activation-quant sharing removed. Ops in
// one partition that read the same tensor (SwiGLU's gate and up, qkv, parallel heads)
// would each absmax-and-quantise it ungrouped; grouped, the row is quantised once.
Stat<uint64_t> g_actq_rows_saved{0};

// Rows quantised at a CALIBRATED static scale rather than per-row absmax.
Stat<uint64_t> g_static_rows{0};
// Opt-in dynamic int8 bmm (q@kT) ops executed.
Stat<uint32_t> g_bmm_ops{0};
// ... and q@kT ops against a persistent int8 K cache, with the key rows they
// read already quantised: every one a quantize_row_i8 the cache saved.
Stat<uint32_t> g_bmm_kcache_ops{0};
Stat<uint64_t> g_bmm_kcache_rows{0};

// --- Column-split, the "min-latency" mode -----------------------------------
// OFF by default, deliberately. The default posture on a 2-core part is
// BACKGROUND inference: the whole model on one core, the other core left
// entirely to the application -- no worker task, no stack, nothing allocated.
// Enabling split spawns a worker pinned to the other core that takes the LOW
// half of every wide-enough quantised addmm's output columns. A split moves
// whole columns between cores; no column's accumulation order changes, so the
// outputs are bit-identical to the single-core path -- which is the parity
// gate, not an aspiration. PIE state needs no save/restore: QACC is per-core
// hardware and each core's task here is that core's only PIE user.
// Same tri-state shape as the scalar override: <0 means each handle's compile
// spec (kSpecMinLatency) decides, >=0 forces all of them.
//
// ATOMIC because set_option() is a public runtime API that may be called from a
// different task than the one in execute(). The value is read once per execute()
// into a local (`want_split`), so a change mid-inference cannot make one op of a
// partition split while the next does not -- relaxed ordering is therefore
// sufficient: there is no other state being published alongside it.
std::atomic<int> g_split_override{-1};
Stat<uint32_t> g_split_jobs{0};  // forked ops, so a split row can prove it forked
Stat<uint32_t> g_head_split_jobs{0};
Stat<uint64_t> g_split_worker_us{0};
Stat<uint32_t> g_staged_ops{0};
Stat<uint64_t> g_staged_bytes{0};

/// Bills one matmul-family op: its MACs and the weight bytes of ONE pass.
/// How many passes it made is count_passes()'s business: row-at-a-time
/// dispatch makes M of them, which is what row blocking exists to collapse.
/// Padding is counted because the kernel really does read it, and the count
/// is in executed bytes -- for i4 half of k_pad, which is the whole point of
/// the rung.
static inline void count_op(uint64_t macs, uint64_t pass_bytes) {
  g_macs += macs;
  g_weight_bytes += pass_bytes;
  ++g_addmm_ops;
}

/// Bills the `passes` passes over `pass_bytes` of weights that `rows`
/// activation rows took, against the `rows` passes row-at-a-time dispatch
/// would have made.
static inline void count_passes(uint64_t passes, uint64_t rows,
                                uint64_t pass_bytes) {
  g_weight_passes += passes;
  g_weight_passes_rowwise += rows;
  g_wbytes_actual += passes * pass_bytes;
  g_wbytes_rowwise += rows * pass_bytes;
}

/// Publishes a loop's locally counted PIE and scalar output columns: one
/// relaxed add per op, never one per column.
static inline void count_cols(uint32_t pie, uint32_t scalar) {
  g_pie_cols.fetch_add(pie, std::memory_order_relaxed);
  g_scalar_cols.fetch_add(scalar, std::memory_order_relaxed);
}

// The affine loops hand the worker a block of their own (affine_columns,
// defined with them below).
struct AffineBlock;
static void affine_columns(const AffineBlock& b, int8_t* wu, uint32_t j0,
                           uint32_t j1, uint32_t* pie, uint32_t* scalar);

struct SplitJob {
  const OpDesc* op;
  const Resolved* res;
  // One BLOCK of quantised rows and their scales, read-only once the caller
  // has quantised them: `nrows` rows at `xq_stride` bytes. A decode row is
  // the nrows == 1 case. The worker takes the low column half of the same
  // block, so a split prefill reads each weight once per block, not once
  // per row on each core.
  const int8_t* xq;
  const float* sxv;
  uint32_t nrows;
  uint32_t xq_stride;
  int8_t* wu;        // the WORKER's unpack scratch (wu2), never the caller's
  const float* bias;
  float* ybase;
  uint32_t y_stride;
  bool aligned;
  const float* rbase;
  // An affine op's block, whose columns the worker computes with
  // affine_columns instead; null for a symmetric op, whose fields are above.
  const AffineBlock* affine;
  uint32_t n0, n1;
  uint32_t pie, scalar;  // worker-local counts, merged by the caller at join
  uint64_t worker_us;
#if ESP32_CONV_FAST
  struct ConvSplit* conv;  // a conv's part instead (run_conv_fast), or null
  static void run_conv(uint64_t t0);
#endif
};
SplitJob g_job;
// Published once by ensure_worker() and never cleared: the worker lives for the
// rest of the process. Atomic so that a caller which sees it non-null also sees
// the lease created before it.
std::atomic<TaskHandle_t> g_worker{nullptr};
#if CONFIG_FREERTOS_UNICORE
// A unicore build runs the worker on core 0: a split runs sequentially, and exactly.
constexpr BaseType_t kSplitWorkerCore = 0;
#else
// Inference runs where ESP-IDF starts the main task, core 0, by convention.
constexpr BaseType_t kSplitWorkerCore = 1;
#endif
// The worker's stack, in bytes (ESP-IDF's FreeRTOS counts bytes, not words).
constexpr uint32_t kSplitWorkerStackBytes = 4096;
// There is ONE worker (it owns one core), so the job slot it reads is a shared
// resource even though the configuration selecting it is per-handle. A caller
// leases it for the duration of an execute(); a second concurrent execute takes
// the lease's failure and runs single-core. That degradation is free of
// consequence precisely because the split is numerically neutral by
// construction -- it moves whole columns, never a column's accumulation order --
// so the two paths return bit-identical values and only the timing differs.
SemaphoreHandle_t g_split_lock = nullptr;
// Worker -> caller: "your columns are written". The backend's own semaphore,
// NOT the caller's task notification, which belongs to the application. A
// notification already pending on the inference task, or one arriving
// mid-split, would end the join before the worker had finished: the caller
// would reuse the activation scratch and could return from execute() while
// the worker was still writing the low columns -- and the application's event
// would be consumed besides.
SemaphoreHandle_t g_split_done = nullptr;

/// Holds the worker lease for one execute() and returns it on every exit path.
struct SplitLease {
  bool held = false;
  ~SplitLease() {
    if (held) {
      xSemaphoreGive(g_split_lock);
    }
  }
};

/// The input index one output position reads for kernel tap `kk`, or -1.
///
/// A DENSE conv strides the input: o*stride - pad + kk*dil. A TRANSPOSED conv
/// is the dense conv, taps flipped, over a ZERO-STUFFED input, so the tap
/// lands at o + kk*dil - ((ksize-1)*dil - pad) in the stuffed grid and reads a
/// real sample only when that is divisible by the stride. Off-lattice taps
/// contribute zero, exactly as out-of-bounds ones already do -- which is what
/// lets the stuffing stay virtual. Materialising it costs 4.84x input
/// inflation on a 2x2/stride-2 up-conv; this costs a modulo. The mapping does
/// not depend on the output extent, so output padding needs nothing here.
/// operator_support/conv.py's _axis_index is the same function.
static inline int32_t axis_index(const ConvGeom& g, uint32_t o, uint32_t kk,
                                 uint32_t stride, uint32_t pad, uint32_t dil,
                                 uint32_t extent, uint32_t ksize) {
  if (g.transposed) {
    // z = up - back in unsigned 32-bit arithmetic: every field is uint16 on
    // the wire, so up = o + kk*dil + pad and back = (ksize - 1)*dil stay
    // below 2^32, and z is negative exactly when back > up. 32 bits keep the
    // modulo and the quotient single instructions (remu, quou); 64-bit ones
    // are libgcc calls on every tap, and a call in the loop body costs the
    // dense taps too (the gather's loops are shared). Geometry validation
    // rejects stride zero before execute.
    const uint32_t up = o + kk * dil + pad;
    const uint32_t back = (ksize - 1) * dil;
    if (up < back || (up - back) % stride) {
      return -1;
    }
    const uint32_t idx = (up - back) / stride;
    return idx < extent ? (int32_t)idx : -1;
  }
  const int64_t idx = (int64_t)o * stride - (int64_t)pad +
      (int64_t)kk * dil;
  return (idx >= 0 && idx < (int64_t)extent) ? (int32_t)idx : -1;
}

/// The symmetric activation code range of each rung: +/-kQmaxI8 for i8 and i4
/// (both quantise activations to int8; see kMaxDotLen for why not -128), and
/// +/-kQmaxI16 for i16.
constexpr int32_t kQmaxI8 = 127;
constexpr int32_t kQmaxI16 = 32767;

/// The im2col gather: one output position's patch, quantised in place.
///
/// After this gather a dense or grouped conv runs the matmul path unchanged --
/// addmm_q_row_range, the weight-pass accounting, the epilogues, the format
/// ladder -- with no dot kernel of its own. Depthwise convs (run_depthwise_i8)
/// and the static int8 path (esp32_conv.h) have loops of their own.
///
/// Column order is (c, d, i, j), matching the flattening order of the
/// [C_out,C_in,kd,kh,kw] weight, so the packed weight and this row agree by
/// construction rather than by a convention written down in two places.
///
/// Two passes over the patch rather than materialising an fp32 row: the row is
/// K floats (4.6 KB at K=1152) and there is one per output position, so the
/// buffer would be the largest thing in the loop for no benefit. Positions
/// outside the input contribute exactly zero -- to the absmax as well as to the
/// dot -- which is what makes zero padding free rather than a special case.
///
/// Templated on the destination width so the i16 rung gets the SAME gather as
/// i8 rather than a second copy of the index maths. The gather is the part that
/// is easy to get subtly wrong -- padding, dilation, the (c, d, i, j) column
/// order that has to match the packed weight -- and it has nothing to do with
/// how many bits the result is stored in.
///
/// noinline, for speed: with one caller (run_blocked_i8, itself inlined
/// into execute()) GCC would inline the gather there, and its loops would lose
/// registers to execute()'s other paths (inlined, the int8 dense 3x3 conv takes
/// 10.30 instructions a MAC instead of 10.03).
template <typename T, int QMAX>
[[gnu::noinline]] float quantize_conv_row_t(const float* x, const ConvGeom& g,
                                            uint32_t row,
                          uint32_t k_pad, T* dst, float fixed_sx,
                          bool ties_to_even) {
  const uint32_t hw = g.h_out * g.w_out;
  const uint32_t od = row / hw;
  const uint32_t rem = row - od * hw;
  const uint32_t oh = rem / g.w_out;
  const uint32_t ow = rem - oh * g.w_out;
  const size_t plane_hw = (size_t)g.h_in * g.w_in;
  const size_t plane = (size_t)g.d_in * plane_hw;
  const uint32_t K = g.c_in * g.kd * g.kh * g.kw;

  float sx;
  if (fixed_sx >= 0.0f) {
    // Counted by the CALLER, which knows whether the fixed scale is a
    // calibrated one (kFlagStaticAct) or the model's own per-tensor absmax
    // (kFlagActPerTensor); this function cannot tell them apart, and counting
    // both here would report per-tensor rows as "calibrated".
    sx = fixed_sx;
  } else {
    float amax = 0.0f;
    for (uint32_t c = 0; c < g.c_in; ++c) {
      const float* xc = x + (size_t)c * plane;
      for (uint32_t dd = 0; dd < g.kd; ++dd) {
        const int32_t idd = axis_index(g, od, dd, g.stride_d, g.pad_d, g.dil_d,
                                       g.d_in, g.kd);
        if (idd < 0) {
          continue;
        }
        const float* xd = xc + (size_t)idd * plane_hw;
        for (uint32_t i = 0; i < g.kh; ++i) {
          const int32_t ih = axis_index(g, oh, i, g.stride_h, g.pad_h, g.dil_h,
                                        g.h_in, g.kh);
          if (ih < 0) {
            continue;
          }
          const float* xr = xd + (size_t)ih * g.w_in;
          for (uint32_t j = 0; j < g.kw; ++j) {
            const int32_t iw = axis_index(g, ow, j, g.stride_w, g.pad_w,
                                          g.dil_w, g.w_in, g.kw);
            if (iw < 0) {
              continue;
            }
            const float a = std::fabs(xr[iw]);
            if (a > amax) {
              amax = a;
            }
          }
        }
      }
    }
    sx = amax / (float)QMAX;
  }
  const float inv_sx = a0_inverse_scale(sx);

  // Column order (c, d, i, j) -- the flattening order of the
  // [C_out,C_in,kd,kh,kw] weight, so the packed weight and this row agree by
  // construction. For a conv2d kd is 1 and the order collapses to (c, i, j),
  // the 2-D order.
  uint32_t t = 0;
  for (uint32_t c = 0; c < g.c_in; ++c) {
    const float* xc = x + (size_t)c * plane;
    for (uint32_t dd = 0; dd < g.kd; ++dd) {
      const int32_t idd = axis_index(g, od, dd, g.stride_d, g.pad_d, g.dil_d,
                                     g.d_in, g.kd);
      const bool d_ok = (idd >= 0);
      const float* xd = d_ok ? xc + (size_t)idd * plane_hw : nullptr;
      for (uint32_t i = 0; i < g.kh; ++i) {
        const int32_t ih = axis_index(g, oh, i, g.stride_h, g.pad_h, g.dil_h,
                                      g.h_in, g.kh);
        const bool row_ok = d_ok && (ih >= 0);
        const float* xr = row_ok ? xd + (size_t)ih * g.w_in : nullptr;
        for (uint32_t j = 0; j < g.kw; ++j, ++t) {
          const int32_t iw = axis_index(g, ow, j, g.stride_w, g.pad_w, g.dil_w,
                                        g.w_in, g.kw);
          if (!row_ok || iw < 0) {
            dst[t] = 0;
            continue;
          }
          const float v = xr[iw] * inv_sx;
          int32_t q = ties_to_even ? round_half_even_i(v) : round_half_away_i(v);
          q = q > QMAX ? QMAX : (q < -QMAX ? -QMAX : q);
          dst[t] = (T)q;
        }
      }
    }
  }
  for (uint32_t z = K; z < k_pad; ++z) {
    dst[z] = 0;
  }
  return sx;
}

// The i8 spelling the call sites use.
float quantize_conv_row(const float* x, const ConvGeom& g, uint32_t row,
                        uint32_t k_pad, int8_t* dst, float fixed_sx,
                        bool ties_to_even) {
  return quantize_conv_row_t<int8_t, kQmaxI8>(x, g, row, k_pad, dst, fixed_sx,
                                          ties_to_even);
}

/// Quantise one activation row to int8, zero-padding K..k_pad. Returns sx. ONE
/// function used by both the standalone op path and the activation-quant-sharing
/// group path, so the two cannot drift numerically.
///
/// fixed_sx < 0 selects per-row absmax (K ops against K*N MACs, so free -- and it
/// is why no PT2E observer is needed for activations); >= 0 is a calibrated
/// static scale (kFlagStaticAct), same rounding and clip after.
///
/// ties_to_even selects PT2E's rounding for an activation whose scheme the
/// model's own folded quantize node defined (kFlagActPerTensor); every scale
/// the backend chooses itself rounds half away from zero, as the AOT
/// references do.
///
/// Templated on the code width, like the conv gather, so the i16 rung
/// quantises its matmul rows with this same loop.
template <typename T, int32_t QMAX>
static inline __attribute__((always_inline)) float quantize_row_t(
    const float* xrow, uint32_t K, uint32_t k_pad, T* dst, float fixed_sx,
    bool ties_to_even) {
  float sx;
  if (fixed_sx >= 0.0f) {
    sx = fixed_sx;   // static-vs-per-tensor accounting is the caller's
  } else {
    float amax = 0.0f;
    for (uint32_t t = 0; t < K; ++t) {
      const float a = std::fabs(xrow[t]);
      if (a > amax) {
        amax = a;
      }
    }
    sx = amax / (float)QMAX;
  }
  const float inv_sx = a0_inverse_scale(sx);
  for (uint32_t t = 0; t < K; ++t) {
    const float v = xrow[t] * inv_sx;
    int32_t q = ties_to_even ? round_half_even_i(v) : round_half_away_i(v);
    q = q > QMAX ? QMAX : (q < -QMAX ? -QMAX : q);
    dst[t] = (T)q;
  }
  // Zero the pad; zeros contribute nothing to the dot, which is what removes the
  // kernel's tail case entirely.
  for (uint32_t t = K; t < k_pad; ++t) {
    dst[t] = 0;
  }
  return sx;
}

// The i8 spelling the call sites use.
ESP32_HOT_CODE
float quantize_row_i8(const float* xrow, uint32_t K, uint32_t k_pad, int8_t* dst,
                      float fixed_sx, bool ties_to_even) {
  return quantize_row_t<int8_t, kQmaxI8>(xrow, K, k_pad, dst, fixed_sx,
                                         ties_to_even);
}

static inline void unpack_i4_row_scalar(const uint8_t* packed, uint32_t k_pad,
                                        int8_t* unpacked) {
  for (uint32_t t = 0; t < k_pad; t += 2) {
    const uint8_t byte = packed[t >> 1];
    // Stored biased by +8, so both halves are ordinary unsigned nibbles and
    // neither needs sign extension.
    unpacked[t] = (int8_t)((int)(byte & 0x0F) - 8);
    unpacked[t + 1] = (int8_t)((int)(byte >> 4) - 8);
  }
}

// The PIE unpack is proven against the scalar one at init (unpack_selftest);
// until then, and if it ever disagrees, every row takes the scalar loop.
// Atomic because a handle on another task may be unpacking while the first
// init writes it; a stale false only means the bit-identical scalar loop.
static std::atomic<bool> g_unpack_pie_ok{false};

static inline void unpack_i4_row(const uint8_t* packed, uint32_t k_pad,
                                 int8_t* unpacked) {
  // 32 codes per vector step, so k_pad a multiple of 32 and both buffers
  // 16-aligned; anything else is the scalar loop, bit for bit the same.
  if (g_unpack_pie_ok.load(std::memory_order_relaxed) &&
      (k_pad & 31u) == 0 &&
      (reinterpret_cast<uintptr_t>(packed) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(unpacked) & 15u) == 0) {
    unpack_i4_s3(packed, unpacked, (int)(k_pad / 2));
    return;
  }
  unpack_i4_row_scalar(packed, k_pad, unpacked);
}

// The PIE self-tests below run once at init on the inference core, PIE against
// scalar; a mismatch keeps the scalar path and says so.
// Marsaglia's xorshift32: the self-tests' reproducible pseudo-random bytes.
static inline uint32_t xorshift32(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}
// unpack_selftest: every nibble value in every byte position.
static bool unpack_selftest(void) {
  constexpr uint32_t kCodes = 512;   // 16 steps of the kernel's 32 codes
  alignas(16) static uint8_t packed[kCodes / 2];
  alignas(16) static int8_t pie[kCodes];
  alignas(16) static int8_t scalar[kCodes];
  uint32_t state = 0x2545F491u;      // any nonzero seed
  for (uint32_t i = 0; i < kCodes / 2; ++i) {
    // The first 16 bytes hold every nibble value in both positions.
    const uint32_t r = xorshift32(&state);
    packed[i] = (uint8_t)(i < 16 ? (i | (i << 4)) : r);
  }
  unpack_i4_row_scalar(packed, kCodes, scalar);
  unpack_i4_s3(packed, pie, (int)(kCodes / 2));
  const bool ok = memcmp(pie, scalar, sizeof(pie)) == 0;
  return ok;
}

// The fused int4 dot (dot_i4_s3.S) reads each packed row directly against a
// deinterleaved copy of the activations, instead of unpacking the row into SRAM
// and calling dot_i8_s3 on it. Proven against that path at init, like the
// unpack kernel; until then, or if it ever disagrees, rows keep the unpack path.
static std::atomic<bool> g_dot_i4_pie_ok{false};
static constexpr uint32_t kMaxFusedI4Groups = 32;

/// The activation row as dot_i4_groups_s3 reads it -- per 32 inputs, the 16 even
/// ones then the 16 odd ones -- and, per scale group, 8 * sum(x): the kernel
/// dots the STORED codes (biased by +8), and subtracting this exact integer
/// gives the dot against the true codes.
ESP32_HOT_CODE
static void prepare_i4_activation(const int8_t* x, uint32_t k_pad, uint32_t gsize,
                                  int8_t* xd, int32_t* corr) {
  for (uint32_t c = 0; c < k_pad; c += 32) {
    for (uint32_t t = 0; t < 16; ++t) {
      xd[c + t] = x[c + 2 * t];
      xd[c + 16 + t] = x[c + 2 * t + 1];
    }
  }
  for (uint32_t g = 0; g * gsize < k_pad; ++g) {
    int32_t sum = 0;
    for (uint32_t t = g * gsize; t < (g + 1) * gsize; ++t) {
      sum += x[t];
    }
    corr[g] = 8 * sum;
  }
}

/// Whether one op's i4 rows can take the fused kernel: whole 32-code steps,
/// groups that tile the row exactly, 16-byte aligned packed rows and scratch.
/// The group bound keeps the raw sum, its 8 * sum(x) correction and their
/// difference inside int32 (|raw| <= g * 15 * 128, |corr| <= g * 8 * 128).
static constexpr uint32_t kMaxFusedI4GroupSize = 1u << 16;
static inline bool fused_i4_ok(const OpDesc& op, uint32_t gsize, const void* wq,
                               size_t stride, const int8_t* xd) {
  return g_dot_i4_pie_ok.load(std::memory_order_relaxed) && gsize % 32 == 0 && gsize <= kMaxFusedI4GroupSize &&
         op.k_pad % gsize == 0 &&
         op.k_pad / gsize <= kMaxFusedI4Groups && stride % 16 == 0 &&
         (reinterpret_cast<uintptr_t>(wq) & 15u) == 0 &&
         (reinterpret_cast<uintptr_t>(xd) & 15u) == 0;
}

// Random rows and activations (and the extremes) at three group sizes, fused
// kernel against the scalar unpack and dot. Runs once at init.
static bool dot_i4_selftest(void) {
  constexpr uint32_t kCodes = 256;
  constexpr uint32_t kMinGroup = 32;  // the kernel's step, so the most groups
  alignas(16) static uint8_t packed[kCodes / 2];
  alignas(16) static int8_t x[kCodes];
  alignas(16) static int8_t xd[kCodes];
  alignas(16) static int8_t w[kCodes];
  int32_t sums[kCodes / kMinGroup], corr[kCodes / kMinGroup];
  uint32_t state = 0x9E3779B9u;       // any nonzero seed
  for (uint32_t i = 0; i < kCodes / 2; ++i) {
    const uint32_t r = xorshift32(&state);
    packed[i] = (uint8_t)(i < 16 ? (i | (i << 4)) : r);
  }
  for (uint32_t i = 0; i < kCodes; ++i) {
    // Both int8 extremes, then random activations.
    const uint32_t r = xorshift32(&state);
    x[i] = (int8_t)(i == 0 ? 127 : i == 1 ? -128 : (int8_t)r);
  }
  unpack_i4_row_scalar(packed, kCodes, w);
  bool ok = true;
  const uint32_t group_sizes[] = {kMinGroup, 128, kCodes};
  for (uint32_t gsize : group_sizes) {
    const uint32_t ngw = kCodes / gsize;
    prepare_i4_activation(x, kCodes, gsize, xd, corr);
    dot_i4_groups_s3(packed, xd, sums, (int)ngw, (int)(gsize / 2));
    for (uint32_t g = 0; g < ngw; ++g) {
      int32_t ref = 0;
      for (uint32_t t = g * gsize; t < (g + 1) * gsize; ++t) {
        ref += (int32_t)x[t] * (int32_t)w[t];
      }
      ok = ok && sums[g] - corr[g] == ref;
    }
  }
  return ok;
}

/// The fused output clamp, NaN-propagating and inf-safe like torch.clamp.
static inline __attribute__((always_inline)) float clamp_output(
    const OpDesc& op, float y) {
  y = y < op.clamp_lo ? op.clamp_lo : y;
  y = y > op.clamp_hi ? op.clamp_hi : y;
  return y;
}

/// Everything between accumulate and store, in the one order every loop
/// applies it: the activation, then the clamp, then the fused residual add or
/// multiply with r[at]. `r` is null when the op carries no operand. The one
/// exception is the residual of a conv that run_blocked_i8 stages: it is added
/// in the NCHW scatter after this has run, because staged rows are not NCHW.
static inline __attribute__((always_inline)) float epilogue(
    const OpDesc& op, float y, const float* r, size_t at) {
  y = apply_act(op.flags, op.clamp_lo, op.clamp_hi, y);
  if (op.flags & kFlagClamp) {
    y = clamp_output(op, y);
  }
  if (r != nullptr) {
    y = apply_operand(op.flags, y, r[at]);
  }
  return y;
}

/// Whether an op's rows take the specialised i8 loops: int8 codes in memory
/// (stored, or expanded at init by kFlagStageI8), one fp32 scale per column,
/// and a PIE-aligned row base. Those loops hoist every per-format decision out
/// of the column loop.
static inline __attribute__((always_inline)) bool is_plain_i8(
    const OpDesc& op, bool wq_aligned) {
  return (op.opcode == kOpAddmmI8 || (op.flags & kFlagStageI8)) &&
         op.scale_group == 0 && !(op.flags & kFlagFp16Scales) && wq_aligned;
}

/// Whether nothing runs between accumulate and store, so a loop may store the
/// scaled accumulator directly. `r` is the operand the caller will pass.
static inline __attribute__((always_inline)) bool no_epilogue(
    const OpDesc& op, const float* r) {
  return (op.flags & kEpilogueFlags) == 0 && r == nullptr;
}

/// Column n's weight codes as int8: the stored (or staged) row in place, or
/// an i4 row expanded into `wu` so the bit-exact dot_i8_s3 serves both
/// rungs (a decode row's i4 takes the fused kernel instead). The expansion is
/// SRAM-side and the op is memory-bound at intensity 1, so it runs in cycles
/// otherwise spent waiting on PSRAM; what halves is the PSRAM read, the
/// bottleneck.
static inline __attribute__((always_inline)) const int8_t* weight_row(
    const int8_t* wq, size_t stride, uint32_t n, bool is_i4, uint32_t k_pad,
    int8_t* wu) {
  if (is_i4) {
    unpack_i4_row(reinterpret_cast<const uint8_t*>(wq) + (size_t)n * stride,
                  k_pad, wu);
    return wu;
  }
  return wq + (size_t)n * stride;
}

/// The int32 dot of `len` codes: dot_i8_s3 when the caller's PIE gate passed,
/// else a scalar loop over the same bytes. Integer accumulation, so the two
/// are bit-identical. Callers pass k_pad-extent groups, not K: the padding is
/// zero so the sum is identical, but reading the bytes the kernel reads keeps
/// the comparison like-for-like rather than a cheaper loop pretending to be one.
static inline __attribute__((always_inline)) int32_t dot_codes(
    const int8_t* x, const int8_t* w, uint32_t len, bool pie) {
  if (pie) {
    return dot_i8_s3(x, w, (int)len);
  }
  int32_t acc = 0;
  for (uint32_t t = 0; t < len; ++t) {
    acc += (int32_t)x[t] * (int32_t)w[t];
  }
  return acc;
}

/// y plus output column `col`'s dot, scale group by scale group: each group's
/// exact integer dot, scaled (acc * sx) * wscale in that order. Group
/// boundaries are FIXED multiples of gsize with a short-tail-plus-padding
/// final group -- the quantiser's boundaries -- and with one
/// group this collapses to exactly the ungrouped expression tree,
/// y + (float)acc * sx * wscale[col]. `fused`, when non-null, holds the fused
/// int4 kernel's raw per-group sums and `corr` their 8 * sum(x) corrections,
/// whose difference is each group's dot; no weight row is read then.
static inline __attribute__((always_inline)) float accumulate_groups(
    const OpDesc& op, const float* scales, uint32_t col, uint32_t gsize,
    uint32_t ngw, const int8_t* x, const int8_t* w, bool pie,
    const int32_t* fused, const int32_t* corr, float sx, float y) {
  for (uint32_t gi = 0; gi < ngw; ++gi) {
    const uint32_t a = gi * gsize;
    const uint32_t len = (a + gsize <= op.k_pad) ? gsize : (op.k_pad - a);
    const int32_t acc = fused != nullptr ? fused[gi] - corr[gi]
                                         : dot_codes(x + a, w + a, len, pie);
    y += (float)acc * sx *  // (acc*sx)*s + y fuses to one madd.s; the mirrors fuse it too
         stored_scale(op.flags, scales, (size_t)col * ngw + gi);
  }
  return y;
}

/// A block of output rows of a quantised (i8/i4) matmul: the column loop with
/// grouped-scale accumulation and the fused act/clamp/residual epilogue. Shared by
/// the standalone path and the activation-quant-sharing group path so the
/// arithmetic cannot fork between them. xq holds nrows already-quantised input
/// rows, xq_stride apart, and sxv their scales; wu is the i4 unpack scratch (the
/// fused int4 kernel's deinterleaved activations at decode; unused for i8); rbase
/// the residual or multiply operand rows, y_stride apart, applied AFTER act/clamp
/// (nullptr when the op carries none).
/// [n0, n1) so the min-latency mode can put the two halves of the column loop on
/// different cores. pie/scalar are written rather than the globals: two cores
/// bumping one counter is a data race, so each side counts locally and the caller
/// merges at the join, keeping the reported totals deterministic.
ESP32_HOT_CODE
void addmm_q_row_range(const OpDesc& op, const Resolved& r, const int8_t* xq,
                       const float* sxv, uint32_t nrows, uint32_t xq_stride,
                       int8_t* wu, const float* bias, float* ybase,
                       uint32_t y_stride, bool wq_aligned, const float* rbase,
                       uint32_t n0, uint32_t n1, uint32_t* pie,
                       uint32_t* scalar) {
  const bool is_i4 = executes_i4(op);
  const int8_t* wq = r.wq;
  // Weight-scale grouping. 0 = one scale per row, the common layout and
  // what the re-quantising path emits by default. Nonzero arrives either with
  // PASS-THROUGH weights or the opt-in group_weights AOT policy and means the
  // scale changes inside the row, so it cannot factor out of the dot: accumulate
  // per group, scale each partial sum. The padded tail belongs to the last group
  // and is zero codes, contributing nothing.
  const uint32_t gsize = op.scale_group ? op.scale_group : op.k_pad;
  const uint32_t ngw = (op.k_pad + gsize - 1) / gsize;
  const size_t stride = (size_t)execution_row_bytes(op);
  // Decode (one activation row): i4 rows take the fused kernel, which needs the
  // activations deinterleaved once per call -- into wu, which this path does
  // not otherwise use -- and yields each group's exact integer dot.
  const bool fused_i4 = is_i4 && nrows == 1 && wq_aligned &&
                        fused_i4_ok(op, gsize, wq, stride, wu);
  int32_t fused_corr[kMaxFusedI4Groups], fused_sums[kMaxFusedI4Groups];
  if (fused_i4) {
    prepare_i4_activation(xq, op.k_pad, gsize, wu, fused_corr);
  }
  const int32_t* fused = fused_i4 ? fused_sums : nullptr;
  for (uint32_t n = n0; n < n1; ++n) {
    const int8_t* wrow = nullptr;
    if (fused_i4) {
      dot_i4_groups_s3(reinterpret_cast<const uint8_t*>(wq) + (size_t)n * stride,
                       wu, fused_sums, (int)ngw, (int)(gsize / 2));
    } else {
      wrow = weight_row(wq, stride, n, is_i4, op.k_pad, wu);
    }
    // Every row in the block consumes this one fetched (and for i4, once-
    // unpacked) weight row before it is dropped. Each output element is still a
    // single dot over the same k in the same order, so nothing about any
    // value's accumulation changes -- only how many times the bytes are read.
    const float initial = bias != nullptr ? bias[n] : 0.0f;
    for (uint32_t mb = 0; mb < nrows; ++mb) {
      const float yn = accumulate_groups(
          op, r.scale, n, gsize, ngw, xq + (size_t)mb * xq_stride, wrow,
          wq_aligned, fused, fused_corr, sxv[mb], initial);
      if (wq_aligned) {
        ++*pie;
      } else {
        ++*scalar;
      }
      const size_t at = (size_t)mb * y_stride + n;
      ybase[at] = epilogue(op, yn, rbase, at);
    }
  }
}

/// Fast blocked counterpart of the common plain-i8 row path.
///
/// Prefill row blocking intentionally changes the loop order to keep one flash
/// weight row live across several activation rows. The fully generic
/// grouped-scale/i4/activation loop would take that overwhelmingly common case
/// through every per-format decision for every output element: the vocabulary
/// head alone would re-test invariant flags and re-resolve one fp32 scale more
/// than 600,000 times per method. This spelling is the exact handwritten
/// schedule: one weight row and scale, then all activation rows.
/// It preserves each dot's K order and the generic loop's fp32 epilogue order.
static void addmm_i8_plain_block_range(
    const OpDesc& op, const Resolved& r, const int8_t* xq,
    const float* sxv, uint32_t nrows, float* ybase, uint32_t y_stride,
    const float* bias, uint32_t n0, uint32_t n1, uint32_t* pie) {
  for (uint32_t n = n0; n < n1; ++n) {
    const int8_t* wrow = r.wq + (size_t)n * op.k_pad;
    const float weight_scale = r.scale[n];
    const float initial = bias != nullptr ? bias[n] : 0.0f;
    for (uint32_t mb = 0; mb < nrows; ++mb) {
      const int32_t acc = dot_i8_s3(
          xq + (size_t)mb * op.k_pad, wrow, (int)op.k_pad);
      float yn = initial;
      yn += (float)acc * sxv[mb] * weight_scale;
      ybase[(size_t)mb * y_stride + n] = yn;
    }
  }
  *pie += (n1 - n0) * nrows;
  g_i8_fast_rows.fetch_add(nrows, std::memory_order_relaxed);
}

/// One whole output row, counting into the globals. The overwhelmingly common
/// decode case -- aligned i8, one weight scale per column -- does not need the
/// generic row-block/group/i4 loop nest. Hoisting those invariant decisions out
/// of the column loop is the same specialization handwritten inference gets,
/// but remains dimension- and model-generic. The range version serves
/// row blocking, i4/grouped formats, scalar fallback and the dual-core split.
ESP32_HOT_CODE
void addmm_q_row(const OpDesc& op, const Resolved& r, const int8_t* xq, float sx,
                 int8_t* wu, const float* bias, float* yrow, bool wq_aligned,
                 const float* rrow) {
  if (is_plain_i8(op, wq_aligned)) {
    if (no_epilogue(op, rrow)) {
      // Plain projection: no flag tests in the hot column loop.
      for (uint32_t n = 0; n < op.n; ++n) {
        const int32_t acc = dot_i8_s3(
            xq, r.wq + (size_t)n * op.k_pad, (int)op.k_pad);
        float yn = bias != nullptr ? bias[n] : 0.0f;
        yn += (float)acc * sx * r.scale[n];
        yrow[n] = yn;
      }
    } else if ((op.flags & kEpilogueFlags) == kFlagClamp && rrow == nullptr) {
      // ReLU arrives as clamp[0,+inf]. Keep this generic for any clamp bounds,
      // but avoid testing every unrelated activation/residual flag per column.
      for (uint32_t n = 0; n < op.n; ++n) {
        const int32_t acc = dot_i8_s3(
            xq, r.wq + (size_t)n * op.k_pad, (int)op.k_pad);
        float yn = bias != nullptr ? bias[n] : 0.0f;
        yn += (float)acc * sx * r.scale[n];
        yrow[n] = clamp_output(op, yn);
      }
    } else {
      // Less common fused epilogues still save the generic format/group/row
      // machinery while keeping the generic loop's exact expression order.
      for (uint32_t n = 0; n < op.n; ++n) {
        const int32_t acc = dot_i8_s3(
            xq, r.wq + (size_t)n * op.k_pad, (int)op.k_pad);
        float yn = bias != nullptr ? bias[n] : 0.0f;
        yn += (float)acc * sx * r.scale[n];
        yrow[n] = epilogue(op, yn, rrow, n);
      }
    }
    g_pie_cols.fetch_add(op.n, std::memory_order_relaxed);
    g_i8_fast_rows.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  uint32_t pie = 0, scalar = 0;
  addmm_q_row_range(op, r, xq, &sx, 1, op.k_pad, wu, bias, yrow, op.n,
                    wq_aligned, rrow, 0, op.n, &pie, &scalar);
  count_cols(pie, scalar);
}

/// One activation row against an arbitrary list of output-table rows.
/// This is the legal-token head path: the full vocabulary remains packed once
/// in external storage, but execute streams only the rows named by int64 IDs.
/// Output order is ID order, including duplicates used to pad a fixed-width ET
/// method.  Arithmetic and epilogue order match addmm_q_row_range exactly.
void addmm_q_selected_rows(const OpDesc& op, const Resolved& r,
                           const int8_t* xq, float sx, int8_t* wu,
                           const float* bias, const int64_t* rows,
                           uint32_t count, float* yrow, bool wq_aligned) {
  const bool is_i4 = executes_i4(op);
  const uint32_t gsize = op.scale_group ? op.scale_group : op.k_pad;
  const uint32_t ngw = (op.k_pad + gsize - 1) / gsize;
  const size_t stride = (size_t)execution_row_bytes(op);
  uint32_t pie = 0, scalar = 0;
  for (uint32_t j = 0; j < count; ++j) {
    const uint32_t n = (uint32_t)rows[j];
    const int8_t* wrow = weight_row(r.wq, stride, n, is_i4, op.k_pad, wu);
    const float value = accumulate_groups(
        op, r.scale, n, gsize, ngw, xq, wrow, wq_aligned, nullptr, nullptr, sx,
        bias != nullptr ? bias[n] : 0.0f);
    if (wq_aligned) {
      ++pie;
    } else {
      ++scalar;
    }
    // No operand: init refuses kFlagAdd and kFlagMul on a selected-row head.
    yrow[j] = epilogue(op, value, nullptr, 0);
  }
  count_cols(pie, scalar);
}

/// The worker's half of one op: the job's column range, symmetric or affine.
ESP32_HOT_CODE
static void run_split_job() {
  g_job.pie = 0;
  g_job.scalar = 0;
  const uint64_t t0 = ESP32_TICK_NOW();
#if ESP32_CONV_FAST
  if (g_job.conv != nullptr) return SplitJob::run_conv(t0);
#endif
  if (g_job.affine != nullptr) {
    affine_columns(*g_job.affine, g_job.wu, g_job.n0, g_job.n1, &g_job.pie,
                   &g_job.scalar);
  } else {
    addmm_q_row_range(*g_job.op, *g_job.res, g_job.xq, g_job.sxv,
                      g_job.nrows, g_job.xq_stride, g_job.wu, g_job.bias,
                      g_job.ybase, g_job.y_stride, g_job.aligned, g_job.rbase,
                      g_job.n0, g_job.n1, &g_job.pie, &g_job.scalar);
  }
  g_job.worker_us = ESP32_TICK_NOW() - t0;
}

/// The worker's whole life: wait, compute the job's column range, hand it back.
/// The caller's notify and the worker's g_split_done give are the
/// synchronisation points -- the caller fully writes g_job before its notify,
/// and reads the counts only after taking g_split_done.
#if !defined(ESP32_HOST_BUILD)
ESP32_HOT_CODE
void split_worker_main(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    run_split_job();
    xSemaphoreGive(g_split_done);
  }
}
#endif

/// Hand the worker g_job, fully written. On the host the worker is a thread
/// of its own per job, joined where the chip takes g_split_done.
ESP32_HOT_CODE
static void start_split_job() {
#if defined(ESP32_HOST_BUILD)
  esp32_host_split_start(run_split_job);
#else
  xTaskNotifyGive(g_worker.load(std::memory_order_relaxed));
#endif
}

/// The lock for process-wide lifecycle state: creating the split worker, the
/// weight cache and its options, and the one-time unpack self-test. Never
/// taken on the per-op path. A function-local static is initialised exactly
/// once even when the first two callers race on different cores -- C++11
/// requires it, and ESP-IDF's guard takes a cross-core spinlock -- so this
/// lock needs no lock of its own.
SemaphoreHandle_t lifecycle_lock() {
  static StaticSemaphore_t storage;
  static const SemaphoreHandle_t lock = xSemaphoreCreateMutexStatic(&storage);
  return lock;
}

/// Holds lifecycle_lock() for one scope.
class LifecycleGuard {
 public:
  LifecycleGuard() { xSemaphoreTake(lifecycle_lock(), portMAX_DELAY); }
  ~LifecycleGuard() { xSemaphoreGive(lifecycle_lock()); }
  LifecycleGuard(const LifecycleGuard&) = delete;
  LifecycleGuard& operator=(const LifecycleGuard&) = delete;
};

/// Spawn the worker on first enable -- background mode must cost nothing.
bool ensure_worker() {
  if (g_worker.load(std::memory_order_acquire) != nullptr) {
    return true;
  }
  // Creating the worker is itself a critical section: execute() and
  // set_option() can call this from any task on either core, and two
  // first-time callers that both saw no worker would both create a mutex
  // (briefly holding DIFFERENT mutexes, so neither excludes the other from the
  // one g_job) and both spawn a worker -- leaving an orphan et_split task
  // pinned to core 1, blocked on a notify that never comes, holding its stack
  // and TCB forever. A scheduler suspend does not guard this: on
  // this dual-core part vTaskSuspendAll() stops scheduling on the calling core
  // only, and ESP-IDF documents that it is not mutual exclusion. A mutex is.
  LifecycleGuard guard;
  if (g_worker.load(std::memory_order_relaxed) != nullptr) {
    return true;
  }
  if (g_split_lock == nullptr) {
    // Created once and never replaced, so a lease is always returned to the
    // mutex it was taken from.
    static StaticSemaphore_t split_lock_storage;
    static StaticSemaphore_t split_done_storage;
    g_split_lock = xSemaphoreCreateMutexStatic(&split_lock_storage);
    g_split_done = xSemaphoreCreateBinaryStatic(&split_done_storage);
  }
#if defined(ESP32_HOST_BUILD)
  // No task on the host: each job gets a thread (esp32_host_split_start), so
  // any non-null handle marks the split available.
  static int host_worker;
  g_worker.store(&host_worker, std::memory_order_release);
#else
  // 4 KB stack is generous: the column loops' frames are small (the largest,
  // affine_columns', holds two 128-byte group arrays) and the dot kernels use
  // none. Created at the creator's priority; every caller then sets it to its
  // own when it takes the lease (match_worker_priority).
  TaskHandle_t worker = nullptr;
  if (xTaskCreatePinnedToCore(split_worker_main, "et_split", kSplitWorkerStackBytes, nullptr,
                              uxTaskPriorityGet(nullptr), &worker,
                              kSplitWorkerCore) != pdPASS) {
    return false;
  }
  g_worker.store(worker, std::memory_order_release);
#endif
  return true;
}

/// Whether this caller can lend half its columns to the worker. A caller
/// already running on the worker's core would only serialise the two halves
/// behind two context switches -- slower than keeping the op -- so it keeps
/// it. (A unicore build shares core 0 on purpose, to keep the split
/// exercised.) Inference tasks are pinned: PIE and FPU state are per core.
bool caller_off_worker_core() {
#if CONFIG_FREERTOS_UNICORE
  return true;
#else
  return xPortGetCoreID() != kSplitWorkerCore;
#endif
}

/// Run the worker at this caller's priority, so both halves of an op run at
/// the same one. Set exactly, not only raised: after one high-priority
/// caller, a later lower-priority caller's half would otherwise keep running
/// on core 1 above application tasks it was meant to yield to. Called only
/// while holding the lease, when no other caller can be using the worker.
void match_worker_priority() {
  TaskHandle_t worker = g_worker.load(std::memory_order_acquire);
  const UBaseType_t mine = uxTaskPriorityGet(nullptr);
  if (uxTaskPriorityGet(worker) != mine) {
    vTaskPrioritySet(worker, mine);
  }
}

/// Apply a min-latency posture. -1 hands every handle back to its own compile
/// spec; 0 forces background (single-core); 1 forces min-latency (column split).
///
/// One implementation for two callers -- BackendInterface::set_option() and the
/// C hook the parity harness uses -- so the sanctioned API and the test back door
/// cannot drift into different semantics. The worker is spawned EAGERLY here
/// rather than lazily in execute(): a caller asking for min-latency gets the
/// failure at the point it can still act on it, instead of discovering later that
/// inference silently stayed single-core.
Error apply_split_override(int want) {
  if (want < -1 || want > 1) {
    ET_LOG(Error, "esp32: min_latency must be -1, 0 or 1, got %d", want);
    return Error::InvalidArgument;
  }
  if (want > 0 && !ensure_worker()) {
    ET_LOG(Error, "esp32: split worker creation failed; staying single-core");
    return Error::Internal;
  }
  g_split_override.store(want, std::memory_order_relaxed);
  return Error::Ok;
}

// Weight tiers: which external weights are copied from flash into PSRAM.
// The weights file is mmapped from flash (26.6 MB/s sequential on this board)
// and PSRAM reads faster (56 MB/s; about 90 MB/s from both cores with the
// 64 KB data cache). The quantised matmuls are bandwidth-bound, so a copy at
// load is the cheapest speedup there is. Which tier a model runs from is a
// property of the DEPLOYMENT (how much PSRAM the application can spare), not
// of the model, so these are runtime backend options, not compile specs. A
// copy holds the stored bytes, so the numerics are untouched.
//
//   "weight_tier_budget"   int   bytes of PSRAM the copies may occupy: 0 =
//                                flash only (the default); -1 = auto, every
//                                candidate while weight_tier_reserve bytes of
//                                PSRAM stay free, checked at each allocation.
//                                fp16 stays on flash under auto: its loop is
//                                bound by its arithmetic (PSRAM reads it only
//                                6% faster than flash: chips.py)
//   "weight_tier_reserve"  int   bytes of PSRAM auto leaves free (default 0)
//   "weight_tier_tables"   bool  also copy embedding tables and selected-row
//                                heads, which read a few of their rows per
//                                inference (default false)
//   "weight_cache"         bool  keep copies across method destroy (default
//                                true): the same weights file entry serves
//                                every method (prefill and decode) and every
//                                reload, so the copy is made once per boot
//                                instead of once per method init
//   "weight_tier_wanted"   int   read-only: the bytes the loaded methods'
//                                candidates would occupy, each weight once,
//                                for the methods loaded while a budget was
//                                set (at 0 nothing is resolved: to learn what
//                                a program wants, load it under budget 1)
//   "weight_tier_resident" int   read-only: the bytes the copies occupy now,
//                                those no loaded method holds included
//   "weight_tier_rank"     bool  ranked placement (default false: each load
//                                copies its candidates in load order). A load
//                                then copies nothing new: it reuses the copies
//                                already made and leaves its other candidates
//                                to weight_tier_apply. Also takes 0 or 1
//   "weight_tier_apply"    bool  true: make the ranked placement now, for
//                                every loaded method. Set it after loading the
//                                methods that will run and before the first
//                                execute anyone times; a ranked load nobody
//                                applies is placed by its first execute. Reads
//                                back whether a ranked load is still waiting
//
// Candidates are the ops with external weights that store weights, opcodes
// 1-7 and 11-14 (a q@kT stores none), staged at init in load order. Inline
// weights are not: an entry is named by its keys (below), which an inline
// payload does not have, and a program that carries its weights is small
// enough to be loaded into RAM anyway. An op FLAG_STAGE_I8 expands is not a
// candidate but a copy it must have, since it runs as int8: the expansion is
// made whatever the budget, a PSRAM shortage fails init, and it counts in the
// resident bytes first, before its init's candidates. It is shared by the
// methods that hold it at once; when the last lets go, it is kept for the
// next method only under a budget (and the cache), and freed otherwise, so
// an application that sets no tier option gets its PSRAM back with the
// method.
//
// A budget is a cap, not a reservation: only what is copied is allocated. A
// newcomer that does not fit may evict copies no loaded method holds,
// whatever their score, least worth first (worth_less), and only until it
// fits (weight_cache_newcomer); when even evicting all of them would not be
// enough -- a newcomer larger than the whole budget, say -- nothing is
// evicted. A loading handle takes its references on the copies it reuses
// before any of its newcomers evicts, so it never evicts what it is about to
// reuse; a destroyed program's copies give way to the next program's
// weights instead of pinning the budget. The newcomer's PSRAM is allocated
// before anything is evicted for the budget, so no copy is evicted for PSRAM
// that is already free; when PSRAM itself is short, unused copies are
// evicted, least worth first, until the allocation succeeds -- several
// smaller ones together if need be -- and none when even all of them would
// not free enough. The z, m and u tables of an asymmetric op (its zero
// points, offsets and weight sums) are not copied: esp32_asym reads them in
// place, and they are a few percent of its weights.
//
// Load order loses weights to their position: a budget filled by the small
// weights that load first has no room left for a large one that loads last,
// an LLM's output head say, and the bytes it leaves free are wasted. Ranked
// placement (weight_tier_rank) instead orders the candidates of every loaded
// method, each weight once, as eviction does in reverse (worth_less): score
// first, then format class, then size, the larger first, then load order. It
// copies them first fit: each candidate that fits what the budget still has,
// after evicting unused copies as above, is copied, and each that does not
// stays on flash while the next is tried. When every candidate
// is read once per execute, as in an LLM's decode step, that is first fit by
// size, largest first. Copies made earlier stay, so load every method that
// will run before applying. The placement frees no copy a loaded method holds
// and moves no handle's pointers: each loaded handle's next execute repoints
// its own ops at the copies of the weights they read, before it reads any,
// so an execute running meanwhile on another task finishes on the pointers
// it started with. The copies cost the same time as in
// load order, but they leave load_method: they are made by the apply call, or
// by the first execute of a ranked load that nobody applied -- inside whatever
// its caller is timing, which is why an application should apply.
constexpr const char* kOptWeightTierBudget = "weight_tier_budget";
constexpr const char* kOptWeightTierReserve = "weight_tier_reserve";
constexpr const char* kOptWeightTierTables = "weight_tier_tables";
constexpr const char* kOptWeightTierWanted = "weight_tier_wanted";
constexpr const char* kOptWeightTierResident = "weight_tier_resident";
constexpr const char* kOptWeightTierRank = "weight_tier_rank";
constexpr const char* kOptWeightTierApply = "weight_tier_apply";
constexpr const char* kOptWeightCache = "weight_cache";
constexpr int kWeightTierAuto = -1;
static size_t g_weight_tier_budget = 0;
static bool g_weight_tier_auto = false;
static size_t g_weight_tier_reserve = 0;
static bool g_weight_tier_tables = false;
static bool g_weight_tier_rank = false;
static bool g_weight_cache = true;

// The weight cache: one entry per distinct weight the loaded methods resolve
// as a candidate, holding its PSRAM copy or, for one left on flash, what a
// copy would take (so "wanted" counts each weight once), and shared by every
// delegate that resolves the same tensors. An entry is identified by WHERE
// its source lives and by WHAT the program calls it: the source addresses and
// sizes, plus a digest of the NamedDataMap key names. The address alone is not
// enough. After an OTA rewrites the weights partition, a new loader typically
// maps it at the old address with the old sizes, and a sample of the bytes
// cannot tell the new weights from the old (a fine-tune can leave a tensor's
// first and last bytes alone, and the sample covers no scale). The names can:
// the exporter names every external payload by its content (esp32_w_/esp32_s_
// plus a sha256 prefix), so changed weights or scales arrive under a different
// name. The address and the source's first and last 16 bytes are checked as
// well, for an exporter whose names do not track content.
// Method init and destroy, set_option() and the C hooks may each run on any
// task, so every access to the table, the options and the counters below holds
// lifecycle_lock(). A copy no loaded method holds (refs == 0) is retained
// while g_weight_cache is on, until a newcomer evicts it or
// release_weight_cache() frees it; an entry without a copy goes with its last
// user, and so does an expansion when no budget is set
// (weight_tier_budgeted).

// What a copy holds: the stored bytes, or FLAG_STAGE_I8's int8 expansion of
// int4 codes (zero points folded in), a different copy of the same tensors.
enum WeightCopyKind : uint8_t { kCopyStored = 0, kCopyExpandI8 = 1 };

// Bytes compared at each end of a weight's source against its entry.
constexpr uint32_t kWeightCacheSampleBytes = 16;
struct CachedWeight {
  const void* src_w;   // flash-mapped source of the weights (identity)
  const void* src_s;   // flash-mapped source of the scales, or null (identity)
  uint64_t names;      // digest of the keys (identity; weight_cache_names)
  uint64_t wbytes;     // weight bytes, as copied (identity)
  uint64_t bytes;      // weights + scales, as copied (identity)
  uint64_t alloc;      // bytes a copy takes (with alignment slack): the budget's unit
  uint64_t score;      // weight_score, the highest any user resolved it at
  void* raw;           // the copy (heap_caps_free takes it), or null: on flash
  const int8_t* w;     // 16-aligned weights
  const float* s;      // scales, right after the weights (null without)
  uint32_t refs;       // ops of loaded methods resolved to this entry
  uint8_t kind;        // WeightCopyKind (identity)
  uint8_t cls;         // ESP32_TIER_*, the format the copy is read in
  uint8_t sample[2][kWeightCacheSampleBytes];  // the source's first, last bytes
};
// An op names its entry by a uint16 (index + 1; 0 = none), which caps the
// table. The table is allocated when first needed, in PSRAM when the board
// has it (it is read when a method loads or is destroyed, never per token,
// and the kernels want the internal SRAM), and doubles when full, so a
// process that stages nothing allocates nothing.
constexpr uint32_t kWeightCacheMax = 0xFFFFu;
constexpr uint32_t kWeightCacheFirst = 32;
constexpr uint32_t kNoEntry = UINT32_MAX;
static CachedWeight* g_weight_cache_entries = nullptr;
static uint32_t g_weight_cache_cap = 0;
static uint32_t g_weight_cache_n = 0;     // entries [0, n) have been used
static size_t g_weight_cache_bytes = 0;   // resident, counted against the budget
static uint32_t g_weight_cache_hits = 0;
static uint32_t g_weight_cache_copies = 0;
static uint32_t g_weight_tier_evictions = 0;
static uint32_t g_weight_tier_skipped = 0;
static uint32_t g_weight_tier_expanded = 0;
static uint32_t g_weight_tier_applies = 0;     // ranked placements made
static uint32_t g_weight_tier_late = 0;        // of those, made by an execute

// The live handles: every handle that resolves a weight through the cache,
// linked at its staging and unlinked by release_resources, so a ranked
// placement can find the loads waiting for it (each link names its handle,
// which repoints its own ops, weight_tier_sync, and whose ops the report
// counts). Allocated with its
// handle. The links are read and written under lifecycle_lock(); `pending`
// -- set when a ranked load left candidates for the placement, cleared by
// the placement -- is also read by execute, without the lock, through
// __atomic loads.
struct TierLink {
  CompiledBlob* blob;
  TierLink* prev;
  TierLink* next;
  uint8_t pending;
  uint8_t dirty;  // an overflow handle's own placement mark (kTierOverflowSlot)
};
static TierLink* g_tier_live = nullptr;

// Placement marks: a byte per cache-linked handle, in internal RAM. A
// placement copies weights but writes no handle's pointers -- an execute on
// another task may be reading them -- so it marks every live handle's slot,
// and each handle's next execute finds its mark and repoints its own ops
// (weight_tier_sync), which clears it. A mark stays until its handle syncs,
// however many placements come in between, so nothing aliases as a wrapping
// counter would. Written under lifecycle_lock(); read by execute without it
// (tier_dirty_now): one internal-RAM byte and no barrier. A stale read only
// defers the move to the next call, to pointers that reach the same bytes.
// Slot 0 is a handle without a link and is never marked. Handles staged while
// slots 1-254 are all taken share slot 255, which a placement marks for all of
// them, and keep their own marks in their links (TierLink::dirty): a PSRAM
// read per execute, for those handles only, while slot 255 is marked.
constexpr uint32_t kTierSlots = 256;
constexpr uint8_t kTierNoSlot = 0;
constexpr uint8_t kTierOverflowSlot = 255;
static uint8_t g_tier_dirty[kTierSlots];
// A bit per slot, set while a handle holds it; under lifecycle_lock().
static uint32_t g_tier_slot_taken[kTierSlots / 32];
#if defined(ESP32_HOST_BUILD)
// The host runner's --tier-slots caps the private slots, so that tests reach
// the overflow path without 255 delegates (esp32_backend_host_cap_tier_slots).
static uint32_t g_tier_slot_limit = kTierOverflowSlot;
#else
constexpr uint32_t g_tier_slot_limit = kTierOverflowSlot;
#endif

/// Whether a handle's slot is marked, as execute reads it once per call: a
/// relaxed atomic load. On the chip, a plain l8ui: an aligned byte load is
/// single-copy atomic on the LX7, the asm keeps the compiler from caching or
/// splitting it, and it carries no memw (GCC's Xtensa port puts one before
/// every volatile or atomic access, relaxed loads included). On the host,
/// the language's relaxed load, which ThreadSanitizer checks.
static inline __attribute__((always_inline)) bool tier_dirty_now(uint8_t slot) {
#if defined(ESP32_HOST_BUILD)
  return __atomic_load_n(&g_tier_dirty[slot], __ATOMIC_RELAXED) != 0;
#else
  uint32_t dirty;  // l8ui zero-extends, so the test needs no mask
  asm volatile("l8ui %0, %1, 0" : "=a"(dirty) : "a"(&g_tier_dirty[slot]));
  return dirty != 0;
#endif
}

/// A slot for a newly linked handle, its mark clear (what it resolves is the
/// cache as every placement so far left it), or kTierOverflowSlot when the
/// private slots are taken. Caller holds lifecycle_lock().
static uint8_t tier_slot_take() {
  for (uint32_t slot = 1; slot < g_tier_slot_limit; ++slot) {
    const uint32_t bit = 1u << (slot % 32);
    if ((g_tier_slot_taken[slot / 32] & bit) == 0) {
      g_tier_slot_taken[slot / 32] |= bit;
      __atomic_store_n(&g_tier_dirty[slot], (uint8_t)0, __ATOMIC_RELAXED);
      return (uint8_t)slot;
    }
  }
  return kTierOverflowSlot;
}

/// Slot 255's mark, kept while some overflow handle's own mark is set.
/// Caller holds lifecycle_lock().
static void tier_overflow_recheck() {
  uint8_t any = 0;
  for (const TierLink* t = g_tier_live; t != nullptr; t = t->next) {
    if (t->blob->tier_slot == kTierOverflowSlot &&
        __atomic_load_n(&t->dirty, __ATOMIC_RELAXED) != 0) {
      any = 1;
      break;
    }
  }
  __atomic_store_n(&g_tier_dirty[kTierOverflowSlot], any, __ATOMIC_RELAXED);
}

/// The live handles (delegates) whose ranked load waits for the placement.
/// Caller holds lifecycle_lock().
static uint32_t weight_tier_waiting() {
  uint32_t waiting = 0;
  for (const TierLink* t = g_tier_live; t != nullptr; t = t->next) {
    waiting += __atomic_load_n(&t->pending, __ATOMIC_RELAXED);
  }
  return waiting;
}

/// 64-bit FNV-1a over one NUL-terminated key, terminator included, so two
/// keys hashed in sequence cannot run together.
static uint64_t fnv1a_key(uint64_t h, const char* key) {
  for (;; ++key) {
    h = (h ^ (uint8_t)*key) * 0x100000001b3ull;
    if (*key == '\0') {
      return h;
    }
  }
}

/// 64-bit FNV-1a over n bytes.
static uint64_t fnv1a_bytes(uint64_t h, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    h = (h ^ p[i]) * 0x100000001b3ull;
  }
  return h;
}

/// The digest an entry is named by: its weight key, then its scale key. A
/// float weight has no scale key and is named by its weight key alone.
static uint64_t weight_cache_names(const char* wkey, const char* skey) {
  const uint64_t h = fnv1a_key(0xcbf29ce484222325ull, wkey);
  return skey != nullptr ? fnv1a_key(h, skey) : h;
}

static inline bool weight_cache_unused(const CachedWeight& e) {
  return e.raw == nullptr && e.refs == 0;
}

/// One op's weight as the cache sees it: its identity, its sizes and where
/// its source bytes are.
struct WeightSource {
  const void* src_w;
  const void* src_s;
  uint64_t names;
  uint64_t src_bytes;  // weight bytes at the source, which the sample covers
  uint64_t wbytes;     // weight bytes in a copy
  uint64_t sbytes;     // scale bytes in a copy
  uint8_t kind;
};

// The same named tensors at the same flash bytes, still identical to the
// entry where sampled: the same weight.
static bool weight_cache_matches(const CachedWeight& e, const WeightSource& s) {
  if (e.kind != s.kind || e.src_w != s.src_w || e.src_s != s.src_s ||
      e.names != s.names || e.wbytes != s.wbytes ||
      e.bytes != s.wbytes + s.sbytes) {
    return false;
  }
  const size_t n = s.src_bytes < kWeightCacheSampleBytes
      ? (size_t)s.src_bytes : kWeightCacheSampleBytes;
  const uint8_t* src = static_cast<const uint8_t*>(s.src_w);
  return std::memcmp(src, e.sample[0], n) == 0 &&
         std::memcmp(src + s.src_bytes - n, e.sample[1], n) == 0;
}

/// Frees an entry no loaded method holds, its copy included.
static void weight_cache_drop(uint32_t index) {
  CachedWeight& e = g_weight_cache_entries[index];
  if (e.raw != nullptr) {
    heap_caps_free(e.raw);
    g_weight_cache_bytes -= (size_t)e.alloc;
  }
  e = CachedWeight{};
}

static uint32_t weight_cache_find(const WeightSource& s) {
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    CachedWeight& e = g_weight_cache_entries[i];
    if (weight_cache_unused(e) || e.src_w != s.src_w || e.kind != s.kind) {
      continue;
    }
    if (weight_cache_matches(e, s)) {
      return i;
    }
    // Same address, another name, or the same name and sizes with other
    // bytes: the partition was rewritten, and a copy nobody holds is stale;
    // one still held stays until its user lets go. Another name also covers
    // an expansion of the same bytes at another geometry (its name carries
    // n, k, k_pad and scale_group): an unused one is dropped the same way,
    // so two methods that alternate over one source at two geometries each
    // expand afresh. (The same names at another size are a different view of
    // the same tensor, a tied head reading the first rows of its table, say:
    // not stale.)
    const bool rewritten = e.names != s.names ||
        (e.wbytes == s.wbytes && e.bytes == s.wbytes + s.sbytes);
    if (rewritten && e.refs == 0) {
      weight_cache_drop(i);
    }
  }
  return kNoEntry;
}

/// A cleared entry for a new weight: one dropped, else the next, else the
/// table grown. kNoEntry when memory or the uint16 index runs out.
static uint32_t weight_cache_slot() {
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    if (weight_cache_unused(g_weight_cache_entries[i])) {
      return i;
    }
  }
  if (g_weight_cache_n == g_weight_cache_cap) {
    if (g_weight_cache_cap >= kWeightCacheMax) {
      return kNoEntry;
    }
    const uint32_t cap = g_weight_cache_cap == 0 ? kWeightCacheFirst
        : (g_weight_cache_cap > kWeightCacheMax / 2 ? kWeightCacheMax
                                                    : 2 * g_weight_cache_cap);
    const size_t size = (size_t)cap * sizeof(CachedWeight);
    void* p = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p == nullptr) {
      p = heap_caps_malloc(size, MALLOC_CAP_8BIT);
    }
    if (p == nullptr) {
      return kNoEntry;
    }
    auto* grown = static_cast<CachedWeight*>(p);
    for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
      grown[i] = g_weight_cache_entries[i];
    }
    if (g_weight_cache_entries != nullptr) {
      heap_caps_free(g_weight_cache_entries);
    }
    g_weight_cache_entries = grown;
    g_weight_cache_cap = cap;
  }
  g_weight_cache_entries[g_weight_cache_n] = CachedWeight{};
  return g_weight_cache_n++;
}

/// Whether a budget is set (weight_tier_budget above 0, or auto). Without
/// one nothing is a candidate, and an expansion lives only as long as a
/// method holds it. Caller holds lifecycle_lock().
static bool weight_tier_budgeted() {
  return g_weight_tier_auto || g_weight_tier_budget > 0;
}

/// Frees the expansions no loaded method holds: what a budget kept for the
/// next method, once no budget is set. Caller holds lifecycle_lock().
static void release_unused_expansions(void) {
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    const CachedWeight& e = g_weight_cache_entries[i];
    if (e.kind == kCopyExpandI8 && e.raw != nullptr && e.refs == 0) {
      weight_cache_drop(i);
    }
  }
}

// Frees every copy nobody is using right now, and the table once nothing is
// left in it. Caller holds lifecycle_lock().
static void release_weight_cache(void) {
  bool empty = true;
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    CachedWeight& e = g_weight_cache_entries[i];
    if (e.raw != nullptr && e.refs == 0) {
      weight_cache_drop(i);
    }
    empty = empty && weight_cache_unused(e);
  }
  if (empty && g_weight_cache_entries != nullptr) {
    heap_caps_free(g_weight_cache_entries);
    g_weight_cache_entries = nullptr;
    g_weight_cache_cap = 0;
    g_weight_cache_n = 0;
  }
}

/// Whether a copy is worth less than another: a lower score, then a later
/// format class, then fewer bytes. Copies are evicted in this order, and
/// wanted in the reverse: highest score first, a tie to the earlier format
/// class, then to the larger.
static bool worth_less(uint64_t score_a, uint8_t cls_a, uint64_t bytes_a,
                       uint64_t score_b, uint8_t cls_b, uint64_t bytes_b) {
  if (score_a != score_b) {
    return score_a < score_b;
  }
  if (cls_a != cls_b) {
    return cls_a > cls_b;
  }
  return bytes_a < bytes_b;
}

/// The bytes copies may occupy for the next one: the budget, or under auto
/// what is resident plus what PSRAM still has beyond the reserve.
static uint64_t weight_tier_limit() {
  if (!g_weight_tier_auto) {
    return g_weight_tier_budget;
  }
  const size_t free_now = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  return (uint64_t)g_weight_cache_bytes +
         (free_now > g_weight_tier_reserve
              ? (uint64_t)(free_now - g_weight_tier_reserve) : 0);
}

/// The unused copy worth least; kNoEntry when there is none.
static uint32_t weight_cache_victim(void) {
  uint32_t victim = kNoEntry;
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    const CachedWeight& e = g_weight_cache_entries[i];
    if (e.raw == nullptr || e.refs != 0) {
      continue;
    }
    const CachedWeight* v =
        victim == kNoEntry ? nullptr : &g_weight_cache_entries[victim];
    if (v == nullptr ||
        worth_less(e.score, e.cls, e.alloc, v->score, v->cls, v->alloc)) {
      victim = i;
    }
  }
  return victim;
}

/// The bytes the unused copies hold: all that evicting could free.
static uint64_t weight_cache_evictable(void) {
  uint64_t bytes = 0;
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    const CachedWeight& e = g_weight_cache_entries[i];
    if (e.raw != nullptr && e.refs == 0) {
      bytes += e.alloc;
    }
  }
  return bytes;
}

/// For a copy that must be made: PSRAM for `need` bytes, evicting unused
/// copies of any score, least worth first, while the allocation fails. Null
/// when nothing is left to evict.
static void* weight_cache_alloc_evicting(uint64_t need, uint32_t* evicted) {
  for (;;) {
    void* raw = heap_caps_malloc((size_t)need,
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (raw != nullptr) {
      return raw;
    }
    const uint32_t victim = weight_cache_victim();
    if (victim == kNoEntry) {
      return nullptr;
    }
    weight_cache_drop(victim);
    ++*evicted;
  }
}

/// PSRAM for a newcomer's copy of `need` bytes within the budget, or null.
/// Unused copies -- those no loaded method holds, whatever their score --
/// make room, least worth first, and only as far as needed:
///   - none at all when evicting every one of them would not make room: not
///     bring the newcomer under the budget (a newcomer larger than the whole
///     budget, say), or not free the PSRAM it needs;
///   - the allocation is tried first, so nothing is evicted for PSRAM that is
///     already free. When it fails, unused copies are evicted, least worth
///     first, until it succeeds: several smaller ones together where no one
///     of them is large enough. (On the chip the blocks they free may not
///     adjoin, so it can still fail with every one of them gone; the host's
///     PSRAM is one block.)
///   - then copies are evicted for the budget alone, until the newcomer fits.
/// The caller holds a reference on the newcomer's entry, so it is never its
/// own victim.
static void* weight_cache_newcomer(uint64_t need, uint32_t* evicted) {
  const uint64_t limit = weight_tier_limit();
  if (need > limit || need > SIZE_MAX) {
    return nullptr;
  }
  if ((uint64_t)g_weight_cache_bytes + need > limit &&
      (uint64_t)g_weight_cache_bytes - weight_cache_evictable() > limit - need) {
    return nullptr;
  }
  void* raw = heap_caps_malloc((size_t)need, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (raw == nullptr) {
    if ((uint64_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM) +
            weight_cache_evictable() < need) {
      return nullptr;
    }
    raw = weight_cache_alloc_evicting(need, evicted);
    if (raw == nullptr) {
      return nullptr;
    }
  }
  while ((uint64_t)g_weight_cache_bytes + need > limit) {
    const uint32_t victim = weight_cache_victim();
    if (victim == kNoEntry) {
      heap_caps_free(raw);  // not reached: the sum above found enough
      return nullptr;
    }
    weight_cache_drop(victim);
    ++*evicted;
  }
  return raw;
}

/// Destruct every FreeableBuffer this handle fetched. Used by destroy() and by
/// InitHeldGuard, because ExecuTorch does not call destroy() after a failed init.
void release_resources(CompiledBlob* blob) {
  if (blob == nullptr) {
    return;
  }
  if (blob->staged_cache != nullptr) {
    LifecycleGuard guard;
    if (TierLink* t = blob->tier) {
      (t->prev != nullptr ? t->prev->next : g_tier_live) = t->next;
      if (t->next != nullptr) {
        t->next->prev = t->prev;
      }
      blob->tier = nullptr;
      blob->tier_deferred = false;
      const uint8_t slot = blob->tier_slot;
      blob->tier_slot = kTierNoSlot;
      if (slot == kTierOverflowSlot) {
        tier_overflow_recheck();
      } else if (slot != kTierNoSlot) {
        g_tier_slot_taken[slot / 32] &= ~(1u << (slot % 32));
        __atomic_store_n(&g_tier_dirty[slot], (uint8_t)0, __ATOMIC_RELAXED);
      }
    }
    for (uint32_t i = 0; i < blob->n_ops; ++i) {
      if (blob->staged_cache[i] != 0) {
        const uint32_t index = blob->staged_cache[i] - 1u;
        CachedWeight& e = g_weight_cache_entries[index];
        if (e.refs > 0) --e.refs;
        // An entry without a copy goes with its last user, and so does a copy
        // with the cache off, and an expansion when no budget is set (an
        // application that sets no tier option gets its PSRAM back with the
        // method); otherwise a copy waits for the next method that resolves
        // the same weight.
        if (e.refs == 0 &&
            (e.raw == nullptr || !g_weight_cache ||
             (e.kind == kCopyExpandI8 && !weight_tier_budgeted()))) {
          weight_cache_drop(index);
        }
        blob->staged_cache[i] = 0;
      }
    }
    blob->staged_cache = nullptr;
  }
  esp32_asym::release(blob->asym);
  if (blob->held == nullptr) {
    return;
  }
  // The arena will not run these, and a FreeableBuffer may own a mapping.
  for (uint32_t i = 0; i < blob->n_held; ++i) {
    blob->held[i].~FreeableBuffer();
  }
  blob->n_held = 0;
  blob->held = nullptr;
}

class InitHeldGuard {
 public:
  explicit InitHeldGuard(CompiledBlob* blob) : blob_(blob), armed_(true) {}
  ~InitHeldGuard() {
    if (armed_) {
      release_resources(blob_);
    }
  }
  void disarm() { armed_ = false; }

 private:
  CompiledBlob* blob_;
  bool armed_;
};

/// Bounds-check the extension table and refuse unknown REQUIRED records.
///
/// Same discipline as the descriptor validation below: every offset and length
/// is checked in 64-bit before anything reads through serialized `.pte` data.
Error validate_extensions(const uint8_t* data, size_t size, uint32_t n_ops,
                          uint32_t ext_off) {
  if (ext_off == 0) {
    return Error::Ok;          // no table; the common case costs nothing
  }
  if ((uint64_t)ext_off + kExtCountBytes > (uint64_t)size) {
    ET_LOG(Error, "esp32: ext table offset %u past the blob (%u B)",
           (unsigned)ext_off, (unsigned)size);
    return Error::InvalidProgram;
  }
  uint32_t count;
  std::memcpy(&count, data + ext_off, kExtCountBytes);
  uint64_t at = (uint64_t)ext_off + kExtCountBytes;
  for (uint32_t i = 0; i < count; ++i) {
    if (at + kExtRecordHeaderBytes > (uint64_t)size) {
      ET_LOG(Error, "esp32: ext record %u header past the blob", (unsigned)i);
      return Error::InvalidProgram;
    }
    uint16_t op_index, type;
    uint32_t len;
    std::memcpy(&op_index, data + at, 2);
    std::memcpy(&type, data + at + 2, 2);
    std::memcpy(&len, data + at + 4, 4);
    const uint64_t padded = ((uint64_t)len + 3ull) & ~3ull;
    if (at + kExtRecordHeaderBytes + padded > (uint64_t)size) {
      ET_LOG(Error, "esp32: ext record %u payload (%u B) past the blob",
             (unsigned)i, (unsigned)len);
      return Error::InvalidProgram;
    }
    if (op_index >= n_ops) {
      ET_LOG(Error, "esp32: ext record %u names op %u of %u", (unsigned)i,
             (unsigned)op_index, (unsigned)n_ops);
      return Error::InvalidProgram;
    }
    const bool optional = (type & kExtOptional) != 0;
    const uint16_t base = (uint16_t)(type & (uint16_t)~kExtOptional);
    if (!optional && !is_known_extension(base)) {
      // Refused, not skipped. A required extension this runtime does not
      // understand means the blob describes a computation it cannot perform,
      // and running the descriptor alone would silently do something else.
      ET_LOG(Error, "esp32: unknown REQUIRED extension type %u on op %u",
             (unsigned)base, (unsigned)op_index);
      return Error::DelegateInvalidCompatibility;
    }
    at += kExtRecordHeaderBytes + padded;
  }
  return Error::Ok;
}

/// Structural validation of the whole blob, BEFORE any field of it is used.
///
/// The delegate's payload is opaque bytes inside the .pte. ExecuTorch does not
/// describe or verify it -- unlike XNNPACK's, which is a flatbuffer with a
/// generated verifier, this one is a hand-rolled struct array, and nothing
/// else checks it: CompiledBlob keeps no size member, so once init returns no
/// offset can be bounds-checked even in principle. Every field below is
/// checked here or trusted absolutely.
///
/// That is not an academic concern on this part. There is no MMU worth the
/// name: an out-of-range read is a wild pointer that sweeps peripheral register
/// space (clear-on-read status registers, FIFOs) rather than a segfault, and
/// `k > k_pad` is an out-of-range WRITE -- quantize_row_i8 fills dst[0..K) into
/// a buffer sized from k_pad. A single flipped bit in a flash image reaches it:
/// on a model with k = k_pad = 96, clearing bit 6 of k_pad gives 32.
///
/// This is deliberately STRUCTURAL, not cryptographic. Signing the payload
/// would be theatre while the program containing it is unauthenticated; the
/// useful property is that corruption fails loudly at load_method instead of
/// silently at some later token.
Error validate_blob(const uint8_t* data, size_t size, uint32_t n_ops) {
  // 64-bit throughout: `kHeaderBytes + 64u * n_ops` in 32-bit wraps for any
  // n_ops that is a multiple of 2^26, which accepts a tiny blob claiming 67
  // million ops -- and the loop that reads them runs before the allocator
  // (whose own overflow guard would have caught it) is ever asked for memory.
  const uint64_t header = (uint64_t)kHeaderBytes +
      (uint64_t)kOpDescBytes * (uint64_t)n_ops;
  if ((uint64_t)size < header) {
    ET_LOG(Error, "esp32: blob truncated: %u B for %u ops", (unsigned)size,
           (unsigned)n_ops);
    return Error::InvalidProgram;
  }
  // Activation-quant-sharing bookkeeping is uint16 throughout (group_members,
  // group_first, and a (uint16_t)i cast of the op index), so beyond 65535 ops
  // those indices wrap. Every wrapped index still lands inside an n_ops-length
  // array -- memory-safe -- but the grouping it describes is silently wrong,
  // which is worse than a refusal.
  if (n_ops > 0xFFFFu) {
    ET_LOG(Error, "esp32: %u ops exceeds the 65535 the group bookkeeping can "
                  "index", (unsigned)n_ops);
    return Error::InvalidProgram;
  }
  // The descriptors are read in place as uint32/float fields, and every
  // inline payload offset below is relative to the blob, so its own alignment
  // comes first.
  if (reinterpret_cast<uintptr_t>(data) % alignof(OpDesc) != 0) {
    ET_LOG(Error, "esp32: blob at %p is not %u-byte aligned", data,
           (unsigned)alignof(OpDesc));
    return Error::InvalidProgram;
  }
  const OpDesc* ops = reinterpret_cast<const OpDesc*>(data + kHeaderBytes);
  const uint64_t total = (uint64_t)size;

  auto fits = [total](uint64_t off, uint64_t len) {
    return off <= total && len <= total - off;
  };
  // An external weight/scale offset names a NUL-terminated key living inside
  // the blob; without a terminator, get_data() and the error-path ET_LOG("%s")
  // both walk off the end.
  auto key_ok = [data, total](uint64_t off) {
    if (off >= total) {
      return false;
    }
    for (uint64_t t = off; t < total; ++t) {
      if (data[t] == 0) {
        return true;
      }
    }
    return false;
  };

  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDesc& op = ops[i];
    switch (op.opcode) {
      case kOpAddmm:
      case kOpAddmmI8:
      case kOpAddmmI4:
      case kOpAddmmI16:
      case kOpEmbedI8:
      case kOpEmbedI4:
      case kOpEmbedI16:
      case kOpBmmDynI8:
      case kOpBmmKcacheI8:
      case kOpBmmKcacheAffineI8:
      case kOpAddmmF16:
      case kOpAddmmBf16:
      case kOpEmbedF16:
      case kOpEmbedBf16:
        break;
      default:
        ET_LOG(Error, "esp32: op %u has unknown opcode %u", (unsigned)i,
               (unsigned)op.opcode);
        return Error::InvalidProgram;
    }
    if (op.flags & ~kKnownFlags) {
      ET_LOG(Error, "esp32: op %u has unknown flags 0x%08x", (unsigned)i,
             (unsigned)(op.flags & ~kKnownFlags));
      return Error::DelegateInvalidCompatibility;
    }
#if defined(ET_PLE_ONLY)
    if (esp32_float::is_op(op.opcode) ||
        (op.opcode == kOpAddmm && (op.flags & kFlagConv))) {
      // The story firmware links no float-weight loop (esp32_float.cpp): its
      // model stores no float weight, and the loops would be flash it only
      // carries.
      ET_LOG(Error,
             "esp32: op %u (opcode %u) needs the float-weight loops, which "
             "this build (ET_PLE_ONLY) leaves out",
             (unsigned)i, (unsigned)op.opcode);
      return Error::DelegateInvalidCompatibility;
    }
#endif
    if (esp32_float::is_op(op.opcode) &&
        (op.k_pad != op.k || op.scale_group != 0 || op.scale_off != 0)) {
      // A 2-byte weight is stored unpadded and unscaled: [n, k] as shipped.
      ET_LOG(Error,
             "esp32: fp16/bf16 op %u has k_pad %u (k %u), scale_group %u or "
             "scale_off %u",
             (unsigned)i, (unsigned)op.k_pad, (unsigned)op.k,
             (unsigned)op.scale_group, (unsigned)op.scale_off);
      return Error::InvalidProgram;
    }
    const uint32_t acts = op.flags & kActFlags;
    if (acts != 0 && (acts & (acts - 1)) != 0) {
      ET_LOG(Error, "esp32: op %u requests multiple activations", (unsigned)i);
      return Error::InvalidProgram;
    }
    // These activations read their constants from the clamp fields, which a
    // clamp would need too.
    constexpr uint32_t kActParams =
        kFlagActLeaky | kFlagActHardswish | kFlagActHardsigmoid;
    if ((op.flags & kActParams) &&
        ((op.flags & kFlagClamp) ||
         ((op.flags & kFlagActLeaky) && !std::isfinite(op.clamp_lo)))) {
      ET_LOG(Error,
             "esp32: op %u has invalid activation constants %g, %g "
             "flags=0x%x",
             (unsigned)i, (double)op.clamp_lo, (double)op.clamp_hi,
             (unsigned)op.flags);
      return Error::InvalidProgram;
    }
    if (op.flags & kFlagMul) {
      // One operand word: a multiply and a residual cannot share it. The
      // fp32 loop accumulates into the output row, so it would destroy an
      // in-place operand; a conv's output is NCHW, not the [m, n] the
      // operand is indexed as. A sentinel operand is only ever the op's own
      // output slot: the earlier projection of a gated pair wrote it.
      if ((op.flags & (kFlagAdd | kFlagSoftcap | kFlagConv)) ||
          op.opcode == kOpAddmm || esp32_float::is_matmul(op.opcode) ||
          (op.res_arg >= kArgScratch0 && op.res_arg != op.out_arg)) {
        ET_LOG(Error,
               "esp32: op %u has an invalid fused multiply (opcode %u "
               "flags=0x%x res_arg=0x%x)",
               (unsigned)i, (unsigned)op.opcode, (unsigned)op.flags,
               (unsigned)op.res_arg);
        return Error::InvalidProgram;
      }
    }
    if (op.flags & kFlagSoftcap) {
      const uint32_t incompatible = kFlagClamp | kFlagActSilu | kFlagActGelu |
          kFlagAdd | kFlagConv;
      if ((op.flags & incompatible) || !std::isfinite(op.clamp_lo) ||
          !std::isfinite(op.clamp_hi) || op.clamp_hi <= 0.0f) {
        ET_LOG(Error,
               "esp32: op %u has invalid softcap scale=%g cap=%g flags=0x%x",
               (unsigned)i, (double)op.clamp_lo, (double)op.clamp_hi,
               (unsigned)op.flags);
        return Error::InvalidProgram;
      }
    }
    if ((op.flags & kFlagClamp) &&
        (std::isnan(op.clamp_lo) || std::isnan(op.clamp_hi) ||
         op.clamp_lo > op.clamp_hi)) {
      ET_LOG(Error, "esp32: op %u has invalid clamp [%g, %g]", (unsigned)i,
             (double)op.clamp_lo, (double)op.clamp_hi);
      return Error::InvalidProgram;
    }
    if (esp32_float::is_embed(op.opcode) &&
        (op.flags & ~kFlagExternalWeights) != 0) {
      // An fp16/bf16 table is unscaled: inline or one external key, no scale
      // table, no zero points.
      ET_LOG(Error, "esp32: fp16/bf16 embedding op %u has flags 0x%08x",
             (unsigned)i, (unsigned)op.flags);
      return Error::InvalidProgram;
    }
    if (is_embed(op.opcode)) {
      constexpr uint32_t kEmbedFlags =
          kFlagExternalWeights | kFlagFp16Scales | kFlagWeightAffine;
      if (op.flags & ~kEmbedFlags) {
        ET_LOG(Error, "esp32: embedding op %u has non-embedding flags 0x%08x",
               (unsigned)i, (unsigned)(op.flags & ~kEmbedFlags));
        return Error::InvalidProgram;
      }
      if (op.opcode == kOpEmbedI16 && (op.flags & kFlagFp16Scales)) {
        // An i16 scale is amax/32767, an fp16 subnormal below amax 2; the
        // AOT writer stores i16 scales as fp32, as for i16 matmuls.
        ET_LOG(Error, "esp32: i16 embedding op %u cannot use fp16 scales",
               (unsigned)i);
        return Error::InvalidProgram;
      }
    } else if (is_bmm(op.opcode)) {
      if (op.flags != 0 || op.scale_group == 0 ||
          op.res_arg >= kArgScratch0) {
        ET_LOG(Error,
               "esp32: bmm op %u has flags 0x%08x, batch %u, or chained "
               "second input %u",
               (unsigned)i, (unsigned)op.flags, (unsigned)op.scale_group,
               (unsigned)op.res_arg);
        return Error::InvalidProgram;
      }
      // The cached-K bmm's third input is an ARG index in scale_off, never a
      // chain slot; and it stores nothing, so a nonzero payload offset is a
      // blob describing some other op.
      if (op.opcode == kOpBmmKcacheI8 &&
          (op.scale_off >= kArgScratch0 || op.bias_off != 0 ||
           op.wt_off != 0)) {
        ET_LOG(Error,
               "esp32: kcache bmm op %u has scale arg %u, bias_off %u, "
               "wt_off %u",
               (unsigned)i, (unsigned)op.scale_off, (unsigned)op.bias_off,
               (unsigned)op.wt_off);
        return Error::InvalidProgram;
      }
    } else {
      // The float-weight loops: opcode 1, and the fp16/bf16 rungs.
      const bool fp32 =
          op.opcode == kOpAddmm || esp32_float::is_matmul(op.opcode);
      if ((op.flags & kFlagStageI8) &&
          (op.opcode != kOpAddmmI4 ||
           !(op.flags & kFlagExternalWeights) ||
           (op.flags & kFlagConv))) {
        ET_LOG(Error,
               "esp32: staged op %u must be an external non-conv i4 addmm",
               (unsigned)i);
        return Error::InvalidProgram;
      }
      if ((op.flags & kFlagHeadSplit) &&
          ((op.opcode != kOpAddmmI8 && op.opcode != kOpAddmmI4) ||
           (op.flags & kFlagConv))) {
        ET_LOG(Error,
               "esp32: selective split op %u must be a quantised non-conv addmm",
               (unsigned)i);
        return Error::InvalidProgram;
      }
      if ((op.flags & kFlagFp16Scales) &&
          (fp32 || op.opcode == kOpAddmmI16)) {
        // Grouped fp16 scales are implemented by the i4/i8 path. The fp32 op
        // has no scales and grouped i16 remains explicitly unsupported below.
        ET_LOG(Error, "esp32: op %u cannot use fp16 scales with opcode %u",
               (unsigned)i, (unsigned)op.opcode);
        return Error::InvalidProgram;
      }
      if ((op.flags & kFlagStaticAct) && (op.flags & kFlagActPerTensor)) {
        ET_LOG(Error, "esp32: op %u selects two activation scale schemes",
               (unsigned)i);
        return Error::InvalidProgram;
      }
      if (op.flags & kFlagStaticAct) {
        if (!std::isfinite(op.act_scale) || op.act_scale < 0.0f) {
          ET_LOG(Error, "esp32: op %u has invalid static scale %g", (unsigned)i,
                 (double)op.act_scale);
          return Error::InvalidProgram;
        }
      }
      if (fp32 &&
          (op.flags & (kFlagStaticAct | kFlagActPerTensor | kFlagStageI8 |
                       kFlagHeadSplit))) {
        // The float loops have no quantiser. These bits are meaningful only
        // on quantised addmm. (kFlagConv is not among them: a float conv
        // gathers its patch as floats; nor kFlagExternalWeights: the loops
        // read a weight from the NamedDataMap as from the blob.)
        ET_LOG(Error, "esp32: fp32 addmm op %u has quantised-only flags 0x%08x",
               (unsigned)i,
               (unsigned)(op.flags &
                          (kFlagStaticAct | kFlagActPerTensor | kFlagStageI8 |
                           kFlagHeadSplit)));
        return Error::InvalidProgram;
      }
      if (op.opcode == kOpAddmm && (op.flags & kFlagExternalWeights) &&
          op.scale_off != 0) {
        // One key: an fp32 weight has no scale table to name.
        ET_LOG(Error,
               "esp32: external fp32 op %u names a scale table (scale_off %u)",
               (unsigned)i, (unsigned)op.scale_off);
        return Error::InvalidProgram;
      }
      if (fp32 && op.scale_group != 0) {
        ET_LOG(Error, "esp32: fp32 addmm op %u has scale_group %u", (unsigned)i,
               (unsigned)op.scale_group);
        return Error::InvalidProgram;
      }
      if (op.opcode == kOpAddmmI16 &&
          ((op.flags & (kFlagStaticAct | kFlagActPerTensor)) ||
           op.scale_group != 0)) {
        ET_LOG(Error, "esp32: i16 addmm op %u requests an unsupported scale mode",
               (unsigned)i);
        return Error::InvalidProgram;
      }
    }
    if (op.m == 0 || op.n == 0 || op.k == 0) {
      // Zero extents are not merely useless: n == 0 makes the init-time
      // diagnostic divide by zero, which on Xtensa is a fatal exception.
      ET_LOG(Error, "esp32: op %u has a zero extent (m=%u k=%u n=%u)",
             (unsigned)i, (unsigned)op.m, (unsigned)op.k, (unsigned)op.n);
      return Error::InvalidProgram;
    }
    // An op never writes an argument it reads. The loops write output while
    // input is still to be read -- the fp32 addmm stores the bias into a row
    // before reading that row's input -- so an output naming the op's own
    // input, chain slot or partition argument alike, computes garbage with
    // no bounds error. The exporter never emits one. kFlagMul is the one
    // deliberate exception and so stays out of reads_res_arg: the up member
    // of a gate x up pair multiplies in place (res_arg == out_arg), and each
    // element is read once and then written, by the same lane.
    const bool reads_res_arg =
        (op.flags & kFlagAdd) || is_bmm(op.opcode);
    if (op.out_arg == op.in_arg ||
        (reads_res_arg && op.out_arg == op.res_arg) ||
        (op.opcode == kOpBmmKcacheI8 && op.out_arg == op.scale_off)) {
      ET_LOG(Error, "esp32: op %u writes argument 0x%x, which it also reads",
             (unsigned)i, (unsigned)op.out_arg);
      return Error::InvalidProgram;
    }
    // THE write primitive. Every scratch buffer is sized from k_pad; every
    // quantiser loop is bounded by k.
    if (op.opcode != kOpAddmm && op.k > op.k_pad) {
      ET_LOG(Error, "esp32: op %u has k %u > k_pad %u", (unsigned)i,
             (unsigned)op.k, (unsigned)op.k_pad);
      return Error::InvalidProgram;
    }
    // The int8 dot paths accumulate a row (or one scale group of it) in int32.
    // Bounded by k_pad, which every group lies within; i16 accumulates in
    // int64 and fp32 in float, so neither is limited.
    if ((op.opcode == kOpAddmmI8 || op.opcode == kOpAddmmI4 ||
         is_bmm(op.opcode)) && op.k_pad > kMaxDotLen) {
      ET_LOG(Error, "esp32: op %u k_pad %u exceeds the %u-element dot an "
                    "int32 accumulator holds exactly", (unsigned)i,
             (unsigned)op.k_pad, (unsigned)kMaxDotLen);
      return Error::InvalidProgram;
    }
    if (op.opcode == kOpAddmmI4 || op.opcode == kOpEmbedI4) {
      if (op.k_pad % 2 != 0) {
        // The nibble loop steps by two and would read one byte past the row.
        ET_LOG(Error, "esp32: op %u is i4 with odd k_pad %u", (unsigned)i,
               (unsigned)op.k_pad);
        return Error::InvalidProgram;
      }
    }
    // Group size: 0 means "one group", anything else must be a real subdivision
    // of the row. 0xFFFFFFFF is the interesting one -- it makes k_pad+gsize-1
    // wrap so ngw evaluates to 0, the accumulation loop never runs, and every
    // output silently becomes bias[n]. Wrong answers, no crash.
    const uint32_t span = is_embed(op.opcode) ? op.k : op.k_pad;
    if (!is_bmm(op.opcode) && op.scale_group != 0
        && op.scale_group > span) {
      ET_LOG(Error, "esp32: op %u scale_group %u exceeds its row span %u",
             (unsigned)i, (unsigned)op.scale_group, (unsigned)span);
      return Error::InvalidProgram;
    }
    if (is_bmm(op.opcode)) {
      // The bmm branch indexes args[] WITHOUT the sentinel-aware bounds check
      // every other op honours, and ExecuTorch's Span::operator[] is unchecked:
      // on this 32-bit target args[0xFFFFFFFE] resolves to eight bytes BEFORE
      // the array and yields a wild EValue*. Preprocess already refuses
      // to emit this ("chained q@kT is unimplemented"); this check holds the
      // invariant on the runtime's side of the serialisation boundary too.
      if (op.in_arg >= kArgScratch0 || op.out_arg >= kArgScratch0) {
        ET_LOG(Error, "esp32: op %u is a dynamic bmm using a chain scratch "
                      "slot, which the runtime cannot index", (unsigned)i);
        return Error::InvalidProgram;
      }
      // A bmm stores no weights, so it escapes the extent checks below. Include
      // the serialized batch count in all three tensor products: bounding only
      // M*K and N*K, a huge B would still wrap every execute-time
      // size_t offset while load_method accepted the descriptor.
      const uint32_t lhs_dims[] = {op.scale_group, op.m, op.k};
      const uint32_t rhs_dims[] = {op.scale_group, op.n, op.k};
      const uint32_t out_dims[] = {op.scale_group, op.m, op.n};
      const uint32_t key_scratch_dims[] = {op.n, op.k_pad};
      uint64_t lhs_elems = 0, rhs_elems = 0, out_elems = 0, key_scratch = 0;
      if (!product_fits(lhs_dims, kMaxTargetFloatElems, &lhs_elems) ||
          !product_fits(rhs_dims, kMaxTargetFloatElems, &rhs_elems) ||
          !product_fits(out_dims, kMaxTargetFloatElems, &out_elems) ||
          !product_fits(key_scratch_dims, UINT32_MAX, &key_scratch) ||
          op.n > kMaxTargetFloatElems) {
        ET_LOG(Error, "esp32: op %u bmm extents exceed 32-bit tensors/scratch "
                      "(b=%u m=%u k=%u n=%u k_pad=%u)", (unsigned)i,
               (unsigned)op.scale_group, (unsigned)op.m, (unsigned)op.k,
               (unsigned)op.n, (unsigned)op.k_pad);
        return Error::InvalidProgram;
      }
      continue;  // no stored weights, scales or bias of its own
    }
    const uint32_t gsize = op.scale_group ? op.scale_group : span;
    const uint64_t ng = gsize ? (((uint64_t)span + gsize - 1) / gsize) : 0;
    if (ng == 0) {
      ET_LOG(Error, "esp32: op %u has a zero group count", (unsigned)i);
      return Error::InvalidProgram;
    }
    // A matmul's grouped scale table holds ceil(k_pad/g) groups per row. When
    // that differs from ceil(k/g) -- the padding reaching into a group of its
    // own -- a table written with one count is read with the other's row
    // stride, misattributing every row's scales after the first, and with
    // equal byte counts no size check can tell.
    if (!is_embed(op.opcode) && op.scale_group != 0 &&
        ((uint64_t)op.k + op.scale_group - 1) / op.scale_group != ng) {
      ET_LOG(Error, "esp32: op %u groups of %u count differently over k %u "
                    "and k_pad %u", (unsigned)i, (unsigned)op.scale_group,
             (unsigned)op.k, (unsigned)op.k_pad);
      return Error::InvalidProgram;
    }

    const uint64_t wbytes = weight_bytes(op);
    const uint64_t sbytes = scale_bytes(op);
    if (wbytes > UINT32_MAX || sbytes > UINT32_MAX) {
      ET_LOG(Error,
             "esp32: op %u stored tensor exceeds the 32-bit target "
             "(weights=%llu scales=%llu)",
             (unsigned)i, (unsigned long long)wbytes,
             (unsigned long long)sbytes);
      return Error::InvalidProgram;
    }

    if (op.flags & kFlagExternalWeights) {
      if (!key_ok(op.wt_off) ||
          (!float_weights(op) && !key_ok(op.scale_off))) {
        ET_LOG(Error, "esp32: op %u external key offset is not a "
                      "NUL-terminated string inside the blob", (unsigned)i);
        return Error::InvalidProgram;
      }
    } else {
      if (!fits(op.wt_off, wbytes) || !fits(op.scale_off, sbytes)) {
        ET_LOG(Error, "esp32: op %u weights/scales run past the blob "
                      "(wt_off %u + %u, scale_off %u + %u, size %u)",
               (unsigned)i, (unsigned)op.wt_off, (unsigned)wbytes,
               (unsigned)op.scale_off, (unsigned)sbytes, (unsigned)size);
        return Error::InvalidProgram;
      }
      if (op.wt_off % weight_align(op) != 0 ||
          (sbytes != 0 && op.scale_off % scale_align(op) != 0)) {
        ET_LOG(Error, "esp32: op %u weights at %u or scales at %u are not "
                      "aligned to their %u/%u-byte elements",
               (unsigned)i, (unsigned)op.wt_off, (unsigned)op.scale_off,
               (unsigned)weight_align(op), (unsigned)scale_align(op));
        return Error::InvalidProgram;
      }
    }
    if (!is_embed(op.opcode) && op.bias_off != 0 &&
        (!fits(op.bias_off, (uint64_t)op.n * sizeof(float)) ||
         op.bias_off % sizeof(float) != 0)) {
      ET_LOG(Error, "esp32: op %u bias at %u runs past the blob or is not "
                    "float-aligned", (unsigned)i, (unsigned)op.bias_off);
      return Error::InvalidProgram;
    }
  }
  return Error::Ok;
}

// ============================================================================
// Loading a blob, one stage per function. `Esp32Backend::init()` below is
// the order they run in. Each stage returns the error for what it checks,
// and init() stops at the first, so a corrupt blob fails at the stage that
// finds the fault.
// ============================================================================

/// The packed per-call TEMP layout: offsets handed out in order, each vector
/// aligned. ExecuTorch resets the temp allocator after every delegate call,
/// so all delegate handles in a method share the largest layout instead of
/// summing their scratch. This matters at real context lengths: one
/// global-attention q@kT at 8K needs ~544 KiB, and retaining one copy per
/// global layer would fail an otherwise fitting model at load. No per-call
/// heap operation is implied; the application gives MemoryManager one
/// reusable arena and allocate() is a pointer bump.
struct ScratchLayout {
  uint32_t bytes = 0;
  bool overflow = false;

  uint32_t reserve(uint64_t need) {
    if (need == 0) {
      return kNoScratch;
    }
    const uint64_t at =
        ((uint64_t)bytes + kVecBytes - 1) & ~(uint64_t)(kVecBytes - 1);
    const uint64_t end = at + need;
    if (end > UINT32_MAX) {
      overflow = true;
      return kNoScratch;
    }
    bytes = (uint32_t)end;
    return (uint32_t)at;
  }
};

/// One of the two compile specs the runtime reads (stage_b, below, is the
/// other). Everything else this backend needs was baked into the blob during
/// preprocess; min-latency configures the RUNTIME's use of the second core,
/// so it has to survive to init and land on the handle. Unknown keys are
/// ignored, not rejected: a runtime meeting a spec key it does not know keeps
/// the default posture instead of refusing to run, and the blob version check
/// already covers layout skew.
static bool compile_spec_min_latency(ArrayRef<CompileSpec> compile_specs) {
  bool min_latency = false;
  for (size_t s = 0; s < compile_specs.size(); ++s) {
    const CompileSpec& spec = compile_specs[s];
    if (spec.key != nullptr && std::strcmp(spec.key, kSpecMinLatency) == 0) {
      min_latency = spec.value.nbytes > 0 &&
                    static_cast<const uint8_t*>(spec.value.buffer)[0] != 0;
    }
  }
  return min_latency;
}

/// Whether this delegate was built under Stage B (kSpecStageB): the one
/// thing besides min-latency init reads from the compile specs, since it
/// decides what esp32_asym::validate admits.
static bool compile_spec_stage_b(
    ArrayRef<CompileSpec> compile_specs) {
  bool stage_b = false;
  for (size_t s = 0; s < compile_specs.size(); ++s) {
    const CompileSpec& spec = compile_specs[s];
    if (spec.key != nullptr && std::strcmp(spec.key, kSpecStageB) == 0) {
      stage_b = spec.value.nbytes > 0 &&
                static_cast<const uint8_t*>(spec.value.buffer)[0] != 0;
    }
  }
  return stage_b;
}

/// Magic, version, then the structural validation of everything the rest of
/// init and execute will index -- checked once, here.
static Error read_blob_header(const uint8_t* data, size_t size,
                              uint32_t* n_ops, uint32_t* ext_off,
                              bool stage_b) {
  if (size < kHeaderBytes) {
    ET_LOG(Error, "esp32: blob too small (%u B)", (unsigned)size);
    return Error::InvalidProgram;
  }
  uint32_t magic, version;
  std::memcpy(&magic, data + 0, 4);
  std::memcpy(&version, data + 4, 4);
  std::memcpy(n_ops, data + 8, 4);
  std::memcpy(ext_off, data + 12, 4);
  if (magic != kMagic) {
    ET_LOG(Error, "esp32: bad magic %08x", (unsigned)magic);
    return Error::InvalidProgram;
  }
  if (version != kVersion) {
    // The documented signal for an AOT/runtime version skew.
    ET_LOG(Error, "esp32: blob version %u, runtime wants %u",
           (unsigned)version, (unsigned)kVersion);
    return Error::DelegateInvalidCompatibility;
  }
  // The asymmetric-quantisation rules first, so an op carrying their flag,
  // record or opcode is refused for that rule rather than a general one.
  // A blob with none of those passes straight through.
  if (Error e = esp32_asym::validate(data, size, *n_ops, *ext_off, stage_b);
      e != Error::Ok) {
    return e;
  }
  if (Error e = validate_blob(data, size, *n_ops); e != Error::Ok) {
    return e;
  }
  return validate_extensions(data, size, *n_ops, *ext_off);
}

/// Every pointer and offset a CompiledBlob owns, in its "nothing allocated
/// yet" state. Listed once so a new field cannot be left uninitialised by
/// one of the stages that never touches it.
static void clear_blob(CompiledBlob* blob, const uint8_t* data, uint32_t n_ops,
                       bool min_latency) {
  blob->base = data;
  blob->ops = reinterpret_cast<const OpDesc*>(data + kHeaderBytes);
  blob->n_ops = n_ops;
  blob->cfg_min_latency = min_latency;
  blob->cfg_head_split = false;
  blob->tier_deferred = false;
  blob->tier_slot = kTierNoSlot;
  blob->xq_rows = 1;
  blob->staged_cache = nullptr;
  blob->resolved = nullptr;
  blob->held = nullptr;
  blob->n_held = 0;
  blob->conv = nullptr;
  blob->selected = nullptr;
  blob->scratch_bytes = 0;
  blob->chain_elems = 0;
  blob->xq = nullptr;
  blob->wu = nullptr;
  blob->wu2 = nullptr;
  blob->xq16 = nullptr;
  blob->sxv = nullptr;
  blob->conv_stage = nullptr;
  blob->bmm_kq = nullptr;
  blob->bmm_sk = nullptr;
  blob->xq_off = blob->conv_stage_off = kNoScratch;
  blob->wu_off = blob->wu2_off = blob->xq16_off = kNoScratch;
  blob->sxv_off = blob->bmm_kq_off = blob->bmm_sk_off = kNoScratch;
  for (uint32_t s = 0; s < kNumScratch; ++s) {
    blob->chain[s] = nullptr;
    blob->chain_off[s] = kNoScratch;
  }
  blob->op_group = nullptr;
  blob->group_members = nullptr;
  blob->group_first = nullptr;
  blob->group_count = nullptr;
  blob->n_groups = 0;
  blob->asym = nullptr;
  blob->asym_terms = nullptr;
  blob->tier = nullptr;
#if ESP32_CONV_FAST
  blob->conv_fast_ops = blob->conv_fast_extra = 0;
  blob->conv_split_off = kNoScratch, blob->conv_split = nullptr;
#endif
}

static bool any_op_has_flag(const CompiledBlob* blob, uint32_t flag) {
  for (uint32_t i = 0; i < blob->n_ops; ++i) {
    if (blob->ops[i].flags & flag) {
      return true;
    }
  }
  return false;
}

/// Point every op's weights and scales at their bytes: inline in the blob,
/// or fetched from the NamedDataMap and held for the method's lifetime
/// (the data pointer is only guaranteed valid while its FreeableBuffer lives).
static Error resolve_weights(BackendInitContext& context, CompiledBlob* blob) {
  const uint8_t* data = blob->base;
  const uint32_t n_ops = blob->n_ops;
  uint32_t n_ext = 0;
  for (uint32_t i = 0; i < n_ops; ++i) {
    if (blob->ops[i].flags & kFlagExternalWeights) {
      n_ext += float_weights(blob->ops[i]) ? 1 : 2;
    }
  }
  if (n_ext == 0) {
    for (uint32_t i = 0; i < n_ops; ++i) {
      blob->resolved[i].wq =
          reinterpret_cast<const int8_t*>(data + blob->ops[i].wt_off);
      blob->resolved[i].scale =
          reinterpret_cast<const float*>(data + blob->ops[i].scale_off);
      ++g_inline_wts;
    }
    return Error::Ok;
  }
  const NamedDataMap* ndm = context.get_named_data_map();
  if (ndm == nullptr) {
    ET_LOG(Error, "esp32: blob wants external weights but no NamedDataMap was "
                  "supplied to load_method");
    return Error::InvalidArgument;
  }
  blob->held = context.get_runtime_allocator()
                   ->allocateList<FreeableBuffer>(n_ext);
  if (blob->held == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    if (!(op.flags & kFlagExternalWeights)) {
      blob->resolved[i].wq =
          reinterpret_cast<const int8_t*>(data + op.wt_off);
      blob->resolved[i].scale =
          reinterpret_cast<const float*>(data + op.scale_off);
      ++g_inline_wts;   // a mixed partition's inline ops count too
      continue;
    }
    const char* wkey = reinterpret_cast<const char*>(data + op.wt_off);
    const char* skey = reinterpret_cast<const char*>(data + op.scale_off);
    // A float weight names no scale table; its unread scale view is the
    // inline one's (scale_off is 0).
    const int keys = float_weights(op) ? 1 : 2;
    blob->resolved[i].scale =
        reinterpret_cast<const float*>(data + op.scale_off);
    for (int which = 0; which < keys; ++which) {
      const char* key = which == 0 ? wkey : skey;
      auto got = ndm->get_data(key);
      if (!got.ok()) {
        ET_LOG(Error, "esp32: NamedDataMap has no '%s' (%d)", key,
               (int)got.error());
        // ExecuTorch runs destroy() only for delegates whose init
        // SUCCEEDED (Method bumps n_delegate_ after the call), so nothing
        // would ever run ~FreeableBuffer on what was already fetched.
        // InitHeldGuard in init() does; under a mmap-backed loader that
        // permanently consumes MMU mappings this is bounded, repeatable
        // exhaustion on an OTA retry loop otherwise.
        return got.error();
      }
      const uint64_t required =
          which == 0 ? weight_bytes(op) : scale_bytes(op);
      if ((uint64_t)got->size() < required) {
        ET_LOG(Error,
               "esp32: NamedDataMap entry '%s' is %u B, descriptor requires "
               "%llu B",
               key, (unsigned)got->size(), (unsigned long long)required);
        return Error::InvalidExternalData;
      }
      const uint32_t align = which == 0 ? weight_align(op) : scale_align(op);
      if (reinterpret_cast<uintptr_t>(got->data()) % align != 0) {
        ET_LOG(Error, "esp32: NamedDataMap entry '%s' is not %u-byte aligned",
               key, (unsigned)align);
        return Error::InvalidExternalData;
      }
      auto* slot = new (&blob->held[blob->n_held++])
          FreeableBuffer(std::move(got.get()));
      ++g_ext_keys;
      if (which == 0) {
        blob->resolved[i].wq = reinterpret_cast<const int8_t*>(slot->data());
      } else {
        blob->resolved[i].scale = reinterpret_cast<const float*>(slot->data());
      }
    }
  }
  return Error::Ok;
}

/// The format a copy of op i's weights is read in (ESP32_TIER_*); an int4
/// head FLAG_STAGE_I8 expands is int8.
static uint8_t weight_class(const OpDesc& op) {
  if (op.flags & kFlagStageI8) {
    return ESP32_TIER_INT8;
  }
  switch (op.opcode) {
    case kOpAddmmI4:
    case kOpEmbedI4:
      return ESP32_TIER_INT4;
    case kOpAddmmI8:
    case kOpEmbedI8:
      return ESP32_TIER_INT8;
    case kOpAddmm:
      return ESP32_TIER_FP32;
    case kOpAddmmBf16:
    case kOpEmbedBf16:
      return ESP32_TIER_BF16;
    case kOpAddmmI16:
    case kOpEmbedI16:
      return ESP32_TIER_INT16;
    default:
      return ESP32_TIER_FP16;
  }
}

// Scores are fixed point: 1 << kScoreShift is one read of every stored byte.
constexpr uint32_t kScoreShift = 16;

static uint64_t score_ratio(uint64_t reads, uint64_t stored) {
  if (stored == 0) {
    stored = 1;
  }
  return reads > (UINT64_MAX >> kScoreShift) ? UINT64_MAX
                                             : (reads << kScoreShift) / stored;
}

/// What a copy of op i's weights saves per byte it occupies: the bytes one
/// execute reads from them over the bytes stored, in 1/65536ths -- passes
/// over the whole weight for a matmul or conv, rows read over rows stored for
/// a table. Read off the descriptor as each loop issues its reads, with no
/// model of the data cache:
///
///   int8, int4 (opcodes 2, 3; an expanded head too)  ceil(m / row_block_for),
///       run_blocked_i8's row blocks (one row per block when no block fits);
///       m for a depthwise conv, whose every output position reads every
///       channel's taps, and for an op with int16 activations, which runs row
///       by row (run_addmm_a16)
///   fp32 (1), int16 (4)    m: one pass per output row
///   fp16, bf16 (11, 12)    1 when the loop widens a block of rows once for all
///       m rows (esp32_float::widen_block), else m
///   embedding (5-7, 13, 14)  m / n: m rows gathered of the n stored
///   selected-row head      m * count / n
///
/// Ops the activation-sharing groups take, which build_activation_groups
/// decides after staging, run their rows one at a time; their passes are
/// still counted by row block. At decode (m = 1) every matmul scores 1.
///
/// When every candidate is bound by the same memory, a copy saves its score
/// times its bytes, so the score orders candidates without a rate table. The
/// format class (weight_class) breaks a tie, then size, the larger first.
static uint64_t weight_score(const CompiledBlob* blob, uint32_t i) {
  const OpDesc& op = blob->ops[i];
  const uint64_t m = op.m != 0 ? op.m : 1;
  if (is_embed(op.opcode)) {
    return score_ratio(m, op.n);
  }
  if (blob->selected[i] != nullptr) {
    return score_ratio(m * blob->selected[i]->count, op.n);
  }
  const ConvGeom* cg = blob->conv[i];
  uint64_t passes;
  if (esp32_float::is_matmul(op.opcode)) {
    passes = esp32_float::widen_block(op.m, op.k, op.n) != 0 ? 1 : m;
  } else if (op.opcode == kOpAddmm || op.opcode == kOpAddmmI16 ||
             (cg != nullptr && is_depthwise(*cg)) ||
             esp32_asym::a16_op(blob->asym, i)) {
    passes = m;
  } else {
    const uint64_t blk = row_block_for(op.m, op.k_pad, cg, op.n);
    passes = (m + blk - 1) / blk;
  }
  return score_ratio(passes, 1);
}

/// Whether op i's weights may take a copy under the budget: stored external
/// weights (a q@kT stores none; an op FLAG_STAGE_I8 expands gets its copy
/// regardless), a table -- an embedding or a selected-row head -- only with
/// weight_tier_tables, and fp16 not under auto.
static bool tier_candidate(const CompiledBlob* blob, uint32_t i) {
  const OpDesc& op = blob->ops[i];
  if (!(op.flags & kFlagExternalWeights) || is_bmm(op.opcode) ||
      (op.flags & kFlagStageI8)) {
    return false;
  }
  if ((is_embed(op.opcode) || blob->selected[i] != nullptr) &&
      !g_weight_tier_tables) {
    return false;
  }
  return !(g_weight_tier_auto && weight_class(op) == ESP32_TIER_FP16);
}

/// Op i's weight as the cache names it. An expansion is laid out by its op
/// and carries its folded zero points, so its name carries what decides its
/// bytes as well: its rows, its row length and its scale groups (two heads
/// over the same packed bytes, one in groups of 96 and one of 128, fold
/// different zero points into the same columns), then the weight record's
/// zero-point kind and storage bias, and its zero-point table's key. A
/// stored copy is the source's bytes whatever the op, and is shared by name.
static WeightSource weight_source(const CompiledBlob* blob, uint32_t i) {
  const OpDesc& op = blob->ops[i];
  const bool floats = float_weights(op);
  WeightSource s;
  s.kind = (op.flags & kFlagStageI8) ? kCopyExpandI8 : kCopyStored;
  s.src_w = blob->resolved[i].wq;
  s.src_s = floats ? nullptr : blob->resolved[i].scale;
  // External ops only, so these offsets name validated NUL-terminated keys;
  // a float weight's scale_off is 0 and names nothing.
  s.names = weight_cache_names(
      reinterpret_cast<const char*>(blob->base + op.wt_off),
      floats ? nullptr
             : reinterpret_cast<const char*>(blob->base + op.scale_off));
  s.src_bytes = weight_bytes(op);
  s.wbytes = s.kind == kCopyExpandI8 ? (uint64_t)op.n * op.k_pad : s.src_bytes;
  s.sbytes = scale_bytes(op);
  if (s.kind == kCopyExpandI8) {
    const uint32_t geometry[4] = {op.n, op.k, op.k_pad, op.scale_group};
    s.names = fnv1a_bytes(s.names, geometry, sizeof(geometry));
  }
  if (s.kind == kCopyExpandI8 && esp32_asym::weight_affine(blob->asym, i)) {
    const esp32_asym::WeightAffine& wa = *blob->asym->ops[i].weight;
    s.names = fnv1a_bytes(s.names, &wa.zp_kind, sizeof(wa.zp_kind));
    s.names = fnv1a_bytes(s.names, &wa.storage_bias, sizeof(wa.storage_bias));
    if (wa.zp_kind != esp32_asym::kZpNone) {
      // A staged op is external (validate_blob), so its tables are keys.
      s.names = fnv1a_key(
          s.names, reinterpret_cast<const char*>(blob->base + wa.zp_off));
    }
  }
  return s;
}

/// A new copy of op i's weights at w (16-aligned, its scales at scale_at):
/// the stored bytes, or FLAG_STAGE_I8's expansion. That is a storage-tier
/// decision, not a quantisation change: every nibble becomes the signed code
/// the per-row scratch unpack would produce, less its group's zero point when
/// the op has a weight record, so execute's affine loop applies no correction.
static void fill_copy(CompiledBlob* blob, uint32_t i, const WeightSource& s,
                      int8_t* w, uint64_t scale_at) {
  const OpDesc& op = blob->ops[i];
  if (s.kind == kCopyStored) {
    std::memcpy(w, s.src_w, (size_t)s.wbytes);
  } else {
    const uint8_t* packed = static_cast<const uint8_t*>(s.src_w);
    const size_t packed_stride = (size_t)op.k_pad / 2;
    for (uint32_t row = 0; row < op.n; ++row) {
      unpack_i4_row(packed + (size_t)row * packed_stride, op.k_pad,
                    w + (size_t)row * op.k_pad);
    }
    if (esp32_asym::weight_affine(blob->asym, i)) {
      // Weight zero points fold into the expansion: every code becomes
      // c - zw (int4 in, [-15, 15] out). Float offsets on a staged op are
      // refused at validate.
      const esp32_asym::AffineWeights aw = esp32_asym::affine_weights(
          blob->asym->ops[i], op.k, op.k_pad, op.scale_group,
          (op.flags & kFlagFp16Scales) != 0, s.src_s);
      for (uint32_t row = 0; row < op.n; ++row) {
        esp32_asym::fold_zero_points(aw, row, w + (size_t)row * op.k_pad);
      }
    }
  }
  if (s.sbytes != 0) {
    std::memcpy(reinterpret_cast<uint8_t*>(w) + scale_at, s.src_s,
                (size_t)s.sbytes);
  }
}

/// One init's staging, for the report printed once the lock is released.
struct StagingReport {
  uint32_t copies = 0, hits = 0, skipped = 0, evicted = 0;
  uint32_t expanded = 0, expanded_hits = 0;  // FLAG_STAGE_I8's, in the above
  uint32_t deferred = 0;        // candidates a ranked load left for the apply
  uint64_t left_on_flash = 0;   // weight bytes of the candidates left on flash
  uint64_t deferred_bytes = 0;  // and of those left for the apply
  uint64_t expanded_bytes = 0;  // bytes of the expansions this init holds
  size_t resident = 0, budget = 0, reserve = 0;
  bool auto_budget = false, ranked = false;
};

/// Op i through the cache: its weight's entry, found or made, and the copy it
/// has or now gets. A candidate that gets none is read from flash; an
/// expansion that gets none fails the init, since the op runs as int8. A
/// ranked load (`ranked`) gives a candidate only a copy already made, and
/// leaves the rest on flash for the placement (tier_place).
static Error tier_resolve(CompiledBlob* blob, uint32_t i, StagingReport* rep,
                          bool ranked) {
  const WeightSource s = weight_source(blob, i);
  const bool must = s.kind == kCopyExpandI8;
  const uint64_t scale_at = staged_scale_offset(s.wbytes);
  const uint64_t need = scale_at + s.sbytes + (kVecBytes - 1);
  uint32_t index = weight_cache_find(s);
  if (index == kNoEntry) {
    index = weight_cache_slot();
    if (index == kNoEntry) {
      if (must) {
        ET_LOG(Error, "esp32: no weight cache entry for staged op %u",
               (unsigned)i);
        return Error::MemoryAllocationFailed;
      }
      ++rep->skipped;
      rep->left_on_flash += s.src_bytes;
      return Error::Ok;
    }
    CachedWeight& fresh = g_weight_cache_entries[index];
    fresh.src_w = s.src_w;
    fresh.src_s = s.src_s;
    fresh.names = s.names;
    fresh.wbytes = s.wbytes;
    fresh.bytes = s.wbytes + s.sbytes;
    fresh.alloc = need;
    fresh.kind = s.kind;
    fresh.cls = weight_class(blob->ops[i]);
    const size_t n = s.src_bytes < kWeightCacheSampleBytes
        ? (size_t)s.src_bytes : kWeightCacheSampleBytes;
    const uint8_t* src = static_cast<const uint8_t*>(s.src_w);
    std::memcpy(fresh.sample[0], src, n);
    std::memcpy(fresh.sample[1], src + s.src_bytes - n, n);
  }
  CachedWeight& e = g_weight_cache_entries[index];
  // Held from here on, so the eviction below can never pick this entry.
  ++e.refs;
  const uint64_t score = weight_score(blob, i);
  if (score > e.score) {
    e.score = score;
  }
  if (e.raw != nullptr) {
    ++rep->hits;
    rep->expanded_hits += must ? 1u : 0u;
  } else if (ranked && !must) {
    ++rep->deferred;
    rep->deferred_bytes += s.src_bytes;
  } else {
    void* raw = weight_cache_newcomer(need, &rep->evicted);
    if (raw == nullptr && must && need <= SIZE_MAX) {
      raw = weight_cache_alloc_evicting(need, &rep->evicted);
    }
    if (raw == nullptr) {
      if (must) {
        ET_LOG(Error,
               "esp32: staged op %u needs %llu B PSRAM (largest free %u)",
               (unsigned)i, (unsigned long long)need,
               (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
        if (--e.refs == 0) {
          weight_cache_drop(index);
        }
        return Error::MemoryAllocationFailed;
      }
      ++rep->skipped;
      rep->left_on_flash += s.src_bytes;
    } else {
      const uintptr_t aligned =
          (reinterpret_cast<uintptr_t>(raw) + kVecBytes - 1) &
          ~(uintptr_t)(kVecBytes - 1);
      int8_t* w = reinterpret_cast<int8_t*>(aligned);
      fill_copy(blob, i, s, w, scale_at);
      e.raw = raw;
      e.w = w;
      e.s = s.sbytes != 0 ? reinterpret_cast<const float*>(
                                reinterpret_cast<uint8_t*>(w) + scale_at)
                          : nullptr;
      g_weight_cache_bytes += (size_t)need;
      ++rep->copies;
      if (must) {
        ++rep->expanded;
        g_staged_ops.fetch_add(1, std::memory_order_relaxed);
        g_staged_bytes.fetch_add(s.wbytes + s.sbytes,
                                 std::memory_order_relaxed);
      }
    }
  }
  blob->staged_cache[i] = (uint16_t)(index + 1);
  if (e.raw != nullptr) {
    blob->resolved[i].wq = e.w;
    if (e.s != nullptr) {
      blob->resolved[i].scale = e.s;
    }
    if (must) {
      rep->expanded_bytes += e.alloc;
      if (esp32_asym::weight_affine(blob->asym, i)) {
        // Every handle reading the expansion reads codes with the zero points
        // already subtracted, whoever made it.
        blob->asym->ops[i].zp_folded = true;
      }
    }
  }
  return Error::Ok;
}

/// Op i through the cache when its weight already has a copy: the reference
/// taken and the op pointed at it, as tier_resolve does for a hit. Whether it
/// had one; without, the cache is left as it was (no entry is made).
static bool tier_hit(CompiledBlob* blob, uint32_t i, StagingReport* rep) {
  const uint32_t index = weight_cache_find(weight_source(blob, i));
  if (index == kNoEntry || g_weight_cache_entries[index].raw == nullptr) {
    return false;
  }
  (void)tier_resolve(blob, i, rep, true);  // a hit: no allocation, no failure
  return true;
}

static Error stage_weights_locked(BackendInitContext& context,
                                  CompiledBlob* blob, StagingReport* rep) {
  // Lookup, eviction, copy, publication and the reference count are one
  // transaction: an entry found with refs == 0 must not be evicted by another
  // task before this handle's reference is taken.
  LifecycleGuard guard;
  rep->budget = g_weight_tier_budget;
  rep->reserve = g_weight_tier_reserve;
  rep->auto_budget = g_weight_tier_auto;
  rep->ranked = g_weight_tier_rank;
  const bool candidates = weight_tier_budgeted();
  if (!candidates && !any_op_has_flag(blob, kFlagStageI8)) {
    return Error::Ok;  // flash only: nothing is resolved, nothing allocated
  }
  const uint32_t n_ops = blob->n_ops;
  // The handle's place among the live ones, then its per-op entry indices:
  // one allocation from the method allocator.
  void* mem = context.get_runtime_allocator()->allocate(
      sizeof(TierLink) + (size_t)n_ops * sizeof(uint16_t), alignof(TierLink));
  if (mem == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  TierLink* link = static_cast<TierLink*>(mem);
  blob->staged_cache = reinterpret_cast<uint16_t*>(link + 1);
  for (uint32_t i = 0; i < n_ops; ++i) {
    blob->staged_cache[i] = 0;
  }
  // Live from here: release_resources unlinks it, after a failed init too.
  link->blob = blob;
  link->prev = nullptr;
  link->next = g_tier_live;
  link->pending = 0;
  link->dirty = 0;
  if (g_tier_live != nullptr) {
    g_tier_live->prev = link;
  }
  g_tier_live = link;
  blob->tier = link;
  // What it resolves below is the cache as every placement so far left it.
  blob->tier_slot = tier_slot_take();
  Error err = Error::Ok;
  // This handle's hits first: every op whose weight already has a copy takes
  // its reference before any of its newcomers may evict an unused copy, so
  // one op's newcomer never evicts the copy a later op would have reused.
  for (uint32_t i = 0; i < n_ops; ++i) {
    if ((blob->ops[i].flags & kFlagStageI8) ||
        (candidates && tier_candidate(blob, i))) {
      tier_hit(blob, i, rep);
    }
  }
  // Then the copies FLAG_STAGE_I8 requires: they count in the resident bytes
  // before this init's candidates do.
  for (uint32_t i = 0; i < n_ops && err == Error::Ok; ++i) {
    if ((blob->ops[i].flags & kFlagStageI8) && blob->staged_cache[i] == 0) {
      err = tier_resolve(blob, i, rep, rep->ranked);
    }
  }
  for (uint32_t i = 0; candidates && i < n_ops && err == Error::Ok; ++i) {
    if (tier_candidate(blob, i) && blob->staged_cache[i] == 0) {
      err = tier_resolve(blob, i, rep, rep->ranked);
    }
  }
  if (rep->deferred > 0) {
    // Read by execute without the lock: its first execute places it, if the
    // application does not apply first.
    __atomic_store_n(&link->pending, (uint8_t)1, __ATOMIC_RELEASE);
    blob->tier_deferred = true;
  }
  g_weight_cache_hits += rep->hits;
  g_weight_cache_copies += rep->copies;
  g_weight_tier_evictions += rep->evicted;
  g_weight_tier_skipped += rep->skipped;
  g_weight_tier_expanded += rep->expanded;
  rep->resident = g_weight_cache_bytes;
  return err;
}

static Error stage_weights(BackendInitContext& context, CompiledBlob* blob) {
  StagingReport rep;
  const Error e = stage_weights_locked(context, blob, &rep);
  // Printed after the lifecycle lock is released: stdout has a lock of its
  // own, and an application holding it while it calls into the backend must
  // not be able to deadlock against a backend print made under ours.
#if ESP32_BACKEND_STATS
  if (e == Error::Ok) {
    if (rep.expanded + rep.expanded_hits > 0) {
      printf("  [esp32 backend] weight tier: %u int4 op(s) expanded to int8 "
             "in PSRAM, %u reused, %llu B\n",
             (unsigned)rep.expanded, (unsigned)rep.expanded_hits,
             (unsigned long long)rep.expanded_bytes);
    }
    const uint32_t copies = rep.copies - rep.expanded;
    const uint32_t hits = rep.hits - rep.expanded_hits;
    if (rep.ranked && (hits > 0 || rep.deferred > 0)) {
      printf("  [esp32 backend] weight tier: ranked, %u op(s) reused from PSRAM, "
             "%u (%llu B) wait for the placement; resident %u B\n",
             (unsigned)hits, (unsigned)rep.deferred,
             (unsigned long long)rep.deferred_bytes, (unsigned)rep.resident);
    } else if (copies > 0 || rep.skipped > 0) {
      if (rep.auto_budget) {
        printf("  [esp32 backend] weight tier: %u op(s) copied to PSRAM, %u "
               "reused, %u (%llu B) stay on flash; resident %u B, budget auto "
               "keeping %u B free\n",
               (unsigned)copies, (unsigned)hits, (unsigned)rep.skipped,
               (unsigned long long)rep.left_on_flash, (unsigned)rep.resident,
               (unsigned)rep.reserve);
      } else {
        printf("  [esp32 backend] weight tier: %u op(s) copied to PSRAM, %u "
               "reused, %u (%llu B) stay on flash; resident %u B of %u B "
               "budget\n",
               (unsigned)copies, (unsigned)hits, (unsigned)rep.skipped,
               (unsigned long long)rep.left_on_flash, (unsigned)rep.resident,
               (unsigned)rep.budget);
      }
    }
    if (rep.evicted > 0) {
      printf("  [esp32 backend] weight tier: %u unused cop%s evicted for "
             "weights scoring higher\n",
             (unsigned)rep.evicted, rep.evicted == 1 ? "y" : "ies");
    }
  }
#endif
  return e;
}

/// Whether cache entry a goes before entry b in a ranked placement: worth
/// more (worth_less, reversed), then the lower index -- load order, as the
/// entries are made.
static bool ranks_before(uint32_t a, uint32_t b) {
  const CachedWeight& x = g_weight_cache_entries[a];
  const CachedWeight& y = g_weight_cache_entries[b];
  if (worth_less(y.score, y.cls, y.alloc, x.score, x.cls, x.alloc)) {
    return true;
  }
  if (worth_less(x.score, x.cls, x.alloc, y.score, y.cls, y.alloc)) {
    return false;
  }
  return a < b;
}

/// One ranked placement, for the report printed once the lock is released.
struct PlacementReport {
  uint32_t handles = 0;  // live handles whose ranked load waited for it
  uint32_t copies = 0, skipped = 0, evicted = 0;
  uint64_t copied_bytes = 0;    // budget bytes of the copies made
  uint64_t left_on_flash = 0;   // weight bytes of the candidates left on flash
  size_t resident = 0, budget = 0, reserve = 0;
  bool auto_budget = false;
};

/// The ranked placement (weight_tier_rank), when a ranked load waits for it:
/// the candidates the live handles read from flash, each weight once as its
/// cache entry, in ranked order (ranks_before), each copied when it fits the
/// budget, first fit; then every live handle's slot marked, by which its
/// next execute repoints its ops at the copies of the weights they read.
/// Whether a load was waiting. Caller holds lifecycle_lock().
static bool tier_place(PlacementReport* rep) {
  rep->handles = weight_tier_waiting();
  if (rep->handles == 0) {
    return false;
  }
  rep->budget = g_weight_tier_budget;
  rep->reserve = g_weight_tier_reserve;
  rep->auto_budget = g_weight_tier_auto;
  // A selection sort with nothing to allocate: each pass takes the best
  // candidate ranking after the previous pick. Quadratic in the table, which
  // holds one entry per weight -- a few hundred at most.
  for (uint32_t prev = kNoEntry;;) {
    uint32_t pick = kNoEntry;
    for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
      const CachedWeight& e = g_weight_cache_entries[i];
      if (e.refs == 0 || e.raw != nullptr || e.kind != kCopyStored ||
          (prev != kNoEntry && !ranks_before(prev, i))) {
        continue;
      }
      if (pick == kNoEntry || ranks_before(i, pick)) {
        pick = i;
      }
    }
    if (pick == kNoEntry) {
      break;
    }
    prev = pick;
    CachedWeight& e = g_weight_cache_entries[pick];
    // Held, so never a victim; the evictions drop entries in place, so `e`
    // stays valid.
    void* raw = weight_cache_newcomer(e.alloc, &rep->evicted);
    if (raw == nullptr) {
      ++rep->skipped;
      rep->left_on_flash += e.wbytes;
      continue;
    }
    // The stored bytes, laid out as tier_resolve lays them: weights at the
    // first 16-aligned byte, scales after them. The source stays mapped while
    // a loaded method holds the entry, which every candidate here has.
    const uint64_t scale_at = staged_scale_offset(e.wbytes);
    const uint64_t sbytes = e.bytes - e.wbytes;
    int8_t* w = reinterpret_cast<int8_t*>(
        (reinterpret_cast<uintptr_t>(raw) + kVecBytes - 1) &
        ~(uintptr_t)(kVecBytes - 1));
    std::memcpy(w, e.src_w, (size_t)e.wbytes);
    if (sbytes != 0) {
      std::memcpy(reinterpret_cast<uint8_t*>(w) + scale_at, e.src_s,
                  (size_t)sbytes);
    }
    e.raw = raw;
    e.w = w;
    e.s = sbytes != 0 ? reinterpret_cast<const float*>(
                            reinterpret_cast<uint8_t*>(w) + scale_at)
                      : nullptr;
    g_weight_cache_bytes += (size_t)e.alloc;
    ++rep->copies;
    rep->copied_bytes += e.alloc;
  }
  // Every live handle, those load order staged included, reads the copies
  // from its next execute on: its slot marked, which that execute finds and
  // repoints its own ops by (weight_tier_sync). Not here: an execute on
  // another task may be reading those pointers now.
  for (TierLink* t = g_tier_live; t != nullptr; t = t->next) {
    const uint8_t slot = t->blob->tier_slot;
    if (slot == kTierOverflowSlot) {
      __atomic_store_n(&t->dirty, (uint8_t)1, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_tier_dirty[slot], (uint8_t)1, __ATOMIC_RELAXED);
    __atomic_store_n(&t->pending, (uint8_t)0, __ATOMIC_RELEASE);
  }
  g_weight_cache_copies += rep->copies;
  g_weight_tier_evictions += rep->evicted;
  g_weight_tier_skipped += rep->skipped;
  ++g_weight_tier_applies;
  rep->resident = g_weight_cache_bytes;
  return true;
}

/// Makes the ranked placement if a ranked load waits for it, and reports it
/// once the lock is released. `late`: made by an execute, inside whatever its
/// caller times. Out of line: execute only calls it once per ranked load.
__attribute__((noinline)) static void weight_tier_apply(bool late) {
  PlacementReport rep;
  bool placed;
  {
    LifecycleGuard guard;
    placed = tier_place(&rep);
    if (placed && late) {
      ++g_weight_tier_late;
    }
  }
#if ESP32_BACKEND_STATS
  if (placed) {
    printf("  [esp32 backend] weight tier: ranked placement%s for %u delegate(s): "
           "%u weight(s) copied to PSRAM (%u B), %u (%u B) stay on flash; "
           "resident %u B, budget %s%u B\n",
           late ? " at a first execute" : "", (unsigned)rep.handles,
           (unsigned)rep.copies, (unsigned)rep.copied_bytes,
           (unsigned)rep.skipped, (unsigned)rep.left_on_flash,
           (unsigned)rep.resident, rep.auto_budget ? "auto keeping free " : "",
           (unsigned)(rep.auto_budget ? rep.reserve : rep.budget));
    if (rep.evicted > 0) {
      printf("  [esp32 backend] weight tier: %u unused cop%s evicted for "
             "weights scoring higher\n",
             (unsigned)rep.evicted, rep.evicted == 1 ? "y" : "ies");
    }
  }
#endif
}

/// A handle's ops read the copies the placements since its last look made:
/// called by its own execute when its slot is marked, before any op or the
/// split worker reads a pointer, so no read of these pointers runs while they
/// move (a handle runs one execute at a time, as its Method does). Held
/// entries are never evicted, so every copy found here outlives the handle's
/// reference. Clears the mark. Out of line: once per handle per placement
/// (every call while slot 255 is marked, for an overflow handle, which then
/// reads its own mark in its link first).
__attribute__((noinline)) static void weight_tier_sync(CompiledBlob* blob) {
  if (blob->tier_slot == kTierOverflowSlot &&
      __atomic_load_n(&blob->tier->dirty, __ATOMIC_ACQUIRE) == 0) {
    return;
  }
  LifecycleGuard guard;
  if (blob->staged_cache != nullptr) {
    for (uint32_t i = 0; i < blob->n_ops; ++i) {
      if (blob->staged_cache[i] == 0) {
        continue;
      }
      const CachedWeight& e =
          g_weight_cache_entries[blob->staged_cache[i] - 1u];
      if (e.raw == nullptr) {
        continue;
      }
      blob->resolved[i].wq = e.w;
      if (e.s != nullptr) {
        blob->resolved[i].scale = e.s;
      }
    }
  }
  if (blob->tier_slot == kTierOverflowSlot) {
    __atomic_store_n(&blob->tier->dirty, (uint8_t)0, __ATOMIC_RELAXED);
    tier_overflow_recheck();
  } else {
    __atomic_store_n(&g_tier_dirty[blob->tier_slot], (uint8_t)0,
                     __ATOMIC_RELAXED);
  }
}

/// One EXT_CONV3D record: decode, validate, and attach to its op.
static Error attach_conv_geometry(BackendInitContext& context,
                                  CompiledBlob* blob, uint16_t op_index,
                                  const uint8_t* payload, uint32_t len,
                                  uint32_t* conv_ops) {
  if (len != kConvGeomBytes) {
    ET_LOG(Error, "esp32: conv geometry record is %u B, expected %u",
           (unsigned)len, (unsigned)kConvGeomBytes);
    return Error::InvalidProgram;
  }
  uint16_t f[kConvGeomBytes / sizeof(uint16_t)];
  std::memcpy(f, payload, kConvGeomBytes);
  const ConvGeom decoded{f[0],  f[1],  f[2],  f[3],  f[4],  f[5],  f[6],
                         f[7],  f[8],  f[9],  f[10], f[11], f[12], f[13],
                         f[14], f[15], f[16], f[17], f[18], f[19], f[20],
                         f[21], f[22], f[23]};
  const OpDesc& od = blob->ops[op_index];
  if (!(od.flags & kFlagConv)) {
    ET_LOG(Error, "esp32: op %u carries conv geometry without kFlagConv",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }
  if (blob->conv[op_index] != nullptr) {
    // There is no ordering or merge rule for two records of one type on
    // one op, so a duplicate is refused rather than one silently winning.
    ET_LOG(Error, "esp32: op %u has duplicate conv geometry records",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }
  if (Error e = validate_conv_geometry(decoded, od, op_index);
      e != Error::Ok) {
    return e;
  }
  if ((od.flags & kFlagAdd) && decoded.groups > 1 &&
      !is_depthwise(decoded)) {
    // A split grouped op writes a channel slice whose origin differs
    // from the full residual tensor's. No serialized offset exists for
    // that residual, so accepting the flag would add the wrong channels.
    ET_LOG(Error, "esp32: grouped conv op %u cannot fuse a residual",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }
  auto* g = context.get_runtime_allocator()->allocateInstance<ConvGeom>();
  if (g == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  *g = decoded;
  blob->conv[op_index] = g;
  ++*conv_ops;
  return Error::Ok;
}

/// One EXT_SELECTED_ROWS record: decode, validate, and attach to its op.
static Error attach_selected_rows(BackendInitContext& context,
                                  CompiledBlob* blob, uint16_t op_index,
                                  const uint8_t* payload, uint32_t len) {
  if (len != kSelectedRowsBytes) {
    ET_LOG(Error, "esp32: selected-row record is %u B, expected %u",
           (unsigned)len, (unsigned)kSelectedRowsBytes);
    return Error::InvalidProgram;
  }
  SelectedRows decoded;
  std::memcpy(&decoded, payload, kSelectedRowsBytes);
  const OpDesc& od = blob->ops[op_index];
  if (blob->selected[op_index] != nullptr) {
    ET_LOG(Error, "esp32: op %u has duplicate selected-row records",
           (unsigned)op_index);
    return Error::InvalidProgram;
  }
  if ((od.opcode != kOpAddmmI8 && od.opcode != kOpAddmmI4) ||
      (od.flags & (kFlagConv | kFlagAdd | kFlagMul | kFlagStageI8 |
                   kFlagHeadSplit)) ||
      od.in_arg >= kArgScratch0 || od.out_arg >= kArgScratch0 ||
      decoded.rows_arg >= kArgScratch0 || decoded.count == 0 ||
      decoded.count > od.n) {
    ET_LOG(Error,
           "esp32: op %u has invalid selected-row contract "
           "(opcode=%u flags=0x%x rows_arg=%u count=%u n=%u)",
           (unsigned)op_index, (unsigned)od.opcode, (unsigned)od.flags,
           (unsigned)decoded.rows_arg, (unsigned)decoded.count,
           (unsigned)od.n);
    return Error::InvalidProgram;
  }
  auto* selection =
      context.get_runtime_allocator()->allocateInstance<SelectedRows>();
  if (selection == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  *selection = decoded;
  blob->selected[op_index] = selection;
  return Error::Ok;
}

/// Extension records into per-op tables, so execute never walks the
/// extension list. validate_extensions has already bounds-checked every
/// offset and length, so this reads without re-checking -- one validation
/// pass, at init, is the discipline the descriptor fields follow too.
static Error parse_extension_records(BackendInitContext& context,
                                     CompiledBlob* blob, uint32_t ext_off,
                                     uint32_t* conv_ops) {
  const uint8_t* data = blob->base;
  const uint32_t n_ops = blob->n_ops;
  blob->conv = context.get_runtime_allocator()
                   ->allocateList<const ConvGeom*>(n_ops);
  blob->selected = context.get_runtime_allocator()
                       ->allocateList<const SelectedRows*>(n_ops);
  if (blob->conv == nullptr || blob->selected == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  for (uint32_t i = 0; i < n_ops; ++i) {
    blob->conv[i] = nullptr;
    blob->selected[i] = nullptr;
  }
  *conv_ops = 0;
  if (ext_off != 0) {
    uint32_t count;
    std::memcpy(&count, data + ext_off, kExtCountBytes);
    uint64_t at = (uint64_t)ext_off + kExtCountBytes;
    for (uint32_t r = 0; r < count; ++r) {
      uint16_t op_index, type;
      uint32_t len;
      std::memcpy(&op_index, data + at, 2);
      std::memcpy(&type, data + at + 2, 2);
      std::memcpy(&len, data + at + 4, 4);
      const uint64_t padded = ((uint64_t)len + 3ull) & ~3ull;
      const uint16_t base_type = (uint16_t)(type & (uint16_t)~kExtOptional);
      const uint8_t* payload = data + at + kExtRecordHeaderBytes;
      if (base_type == kExtConv3d) {
        if (Error e = attach_conv_geometry(context, blob, op_index, payload,
                                           len, conv_ops);
            e != Error::Ok) {
          return e;
        }
      } else if (base_type == kExtSelectedRows) {
        if (Error e = attach_selected_rows(context, blob, op_index, payload,
                                           len);
            e != Error::Ok) {
          return e;
        }
      }
      at += kExtRecordHeaderBytes + padded;
    }
  }
  // The reverse check. A flag without its record would gather nothing and run
  // a matmul over the raw input -- plausible-looking numbers, silently wrong.
  for (uint32_t i = 0; i < n_ops; ++i) {
    if ((blob->ops[i].flags & kFlagConv) && blob->conv[i] == nullptr) {
      ET_LOG(Error, "esp32: op %u has kFlagConv but no geometry record",
             (unsigned)i);
      return Error::InvalidProgram;
    }
  }
  return Error::Ok;
}

/// Individual records are not enough to describe a grouped convolution:
/// every output-channel slice must occur exactly once. Without this set
/// check, two individually valid group_idx=0 descriptors load successfully,
/// both overwrite the first slice, and the rest of the output is untouched.
/// Requiring the exact contiguous ordering emitted by split_groups() also
/// gives the wire format an unambiguous group boundary without another ID.
static Error check_grouped_conv_sequences(const CompiledBlob* blob) {
  const uint32_t n_ops = blob->n_ops;
  for (uint32_t i = 0; i < n_ops;) {
    const ConvGeom* leader = blob->conv[i];
    if (leader == nullptr || leader->groups <= 1 || is_depthwise(*leader)) {
      ++i;
      continue;
    }
    if (leader->group_idx != 0 || leader->groups > n_ops - i) {
      ET_LOG(Error,
             "esp32: grouped conv sequence at op %u starts with idx %u/%u "
             "or is truncated",
             (unsigned)i, (unsigned)leader->group_idx,
             (unsigned)leader->groups);
      return Error::InvalidProgram;
    }
    for (uint32_t group = 0; group < leader->groups; ++group) {
      const uint32_t member_index = i + group;
      const ConvGeom* member = blob->conv[member_index];
      if (member == nullptr || member->group_idx != group ||
          !same_grouped_conv_contract(
              *leader, blob->ops[i], *member, blob->ops[member_index])) {
        ET_LOG(Error,
               "esp32: grouped conv sequence at op %u has missing, duplicate, "
               "or mismatched group %u",
               (unsigned)i, (unsigned)group);
        return Error::InvalidProgram;
      }
    }
    i += leader->groups;
  }
  return Error::Ok;
}

#if ESP32_CONV_FAST
// ---- the static int8 convolution path: which ops take it --------------------
// esp32_conv.h computes an affine int8 conv whose activation codes are a pure
// function of each input element -- Stage B's codes, or fp32 under a static or
// per-tensor scheme -- and whose column math is one scale group with nothing
// between accumulate and store but a clamp. init (for the scratch) and execute
// ask the same question here; everything else takes the general loops.

// Ops run on the path, and the output elements they wrote: the parity and
// bench firmware's proof that a conv took it.
Stat<uint32_t> g_conv_fast_ops{0};
Stat<uint64_t> g_conv_fast_outputs{0};
// ... and those of them split across the two cores (min-latency).
Stat<uint32_t> g_conv_split_ops{0};
// The output elements its integer epilogue wrote (codes out), and those of
// them it left to the float sequence near a rounding boundary.
Stat<uint64_t> g_conv_epi_fixed{0};
Stat<uint64_t> g_conv_epi_fallback{0};
// esp32_backend_set_conv_fast(0) sends every conv to the general loops: one
// binary A/Bs the two (and the host runner proves them equal).
std::atomic<int> g_conv_fast_override{1};

static bool conv_fast_op(const CompiledBlob* blob, uint32_t i) {
  const OpDesc& op = blob->ops[i];
  const ConvGeom* cg = blob->conv[i];
  const esp32_asym::BlobState* st = blob->asym;
  // A 2-D conv, or a 3-D one whose depth axis is one plane, one tap and no
  // padding: its depth tap then reads plane 0 whatever stride_d and dil_d
  // are (tap_index: od * stride - pad + dd * dil with od = dd = 0), so it is
  // the 2-D conv of that plane. With pad_d > 0 that tap is padding (index
  // -pad_d), every one of them, and the path, which reads plane 0, would be
  // wrong; a transposed conv's taps run through another mapping.
  if (cg == nullptr || st == nullptr || cg->transposed != 0 ||
      cg->d_in != 1 || cg->d_out != 1 || cg->kd != 1 || cg->pad_d != 0) {
    return false;
  }
  // int8 codes as stored, one scale group per row, only a clamp after the
  // accumulate (no fused activation, residual or multiply operand).
  if (op.opcode != kOpAddmmI8 || (op.flags & kFlagStageI8) ||
      (op.flags & (kActFlags | kFlagAdd | kFlagMul)) ||
      (op.scale_group != 0 && op.scale_group != op.k_pad)) {
    return false;
  }
  const bool dw = is_depthwise(*cg);
  if (op.m != cg->h_out * cg->w_out ||
      op.k != (dw ? 1u : cg->c_in) * cg->kh * cg->kw || op.k > op.k_pad) {
    return false;
  }
  // An activation record whose codes do not depend on the patch: static, or
  // chosen once over the tensor (attach refuses per-row records on a conv).
  const esp32_asym::OpQuant& q = st->ops[i];
  if (q.act == nullptr || q.act->code_bits != 8 ||
      q.act->granularity == esp32_asym::kGranularityRow) {
    return false;
  }
  if (q.weight != nullptr &&
      (q.weight->zp_kind != esp32_asym::kZpNone ||
       q.weight->offset_kind != esp32_asym::kOffsetNone ||
       q.weight->storage_bias != 0 || q.zp_folded)) {
    return false;
  }
  return true;
}

static esp32_conv::Op conv_fast_geometry(const OpDesc& op,
                                         const ConvGeom& g) {
  esp32_conv::Op o;
  o.h_in = g.h_in;
  o.w_in = g.w_in;
  o.c_in = g.c_in;
  o.h_out = g.h_out;
  o.w_out = g.w_out;
  o.kh = g.kh;
  o.kw = g.kw;
  o.stride_h = g.stride_h;
  o.stride_w = g.stride_w;
  o.pad_h = g.pad_h;
  o.pad_w = g.pad_w;
  o.dil_h = g.dil_h;
  o.dil_w = g.dil_w;
  o.depthwise = is_depthwise(g);
  o.m = op.m;
  o.k = op.k;
  o.n = op.n;
  o.k_pad = op.k_pad;
  o.clamp = (op.flags & kFlagClamp) != 0;
  o.clamp_lo = op.clamp_lo;
  o.clamp_hi = op.clamp_hi;
  return o;
}

/// The row block the path runs a dense conv's columns over: as many output
/// positions as kConvFastBlockBytes of codes hold, at most kConvFastBlockMax.
/// Each block is one pass over the op's weights, and unlike the general
/// loops (row_block_for) there is no fp32 staging row to budget -- the path
/// writes NCHW in place -- so a wide op keeps a tall block: the bench's
/// MobileNetV2-224 reads 8.5 MB of weights an inference instead of 32 MB.
constexpr uint32_t kConvFastBlockBytes = 16 * 1024;
constexpr uint32_t kConvFastBlockMax = 64;

static uint32_t conv_fast_rows(const OpDesc& op) {
  uint32_t b = kConvFastBlockBytes / op.k_pad;
  b = b < 1 ? 1 : (b > kConvFastBlockMax ? kConvFastBlockMax : b);
  return b > op.m ? op.m : b;
}

/// The xq bytes op i needs on the path (0 when it does not take it): its
/// row block (conv_fast_rows), then a code plane for an fp32 input.
static uint64_t conv_fast_scratch(const CompiledBlob* blob, uint32_t i) {
  if (!conv_fast_op(blob, i)) {
    return 0;
  }
  const OpDesc& op = blob->ops[i];
  const ConvGeom& g = *blob->conv[i];
  const uint32_t rows = is_depthwise(g) ? 0 : conv_fast_rows(op);
  return esp32_conv::scratch_need(conv_fast_geometry(op, g),
                                  !esp32_asym::codes_in(blob->asym, i), rows);
}

/// No convolution splits across the cores, under min-latency or otherwise:
/// on the board a split conv gave wrong outputs and cache-error panics with
/// its weights or activations in external memory (README, Known limits). The
/// split stays built, off, until the board proves it exact. Its thresholds:
/// the fewest output elements an op would split at, ...
constexpr bool kConvSplit = false;  // scratch_layout.py's CONV_SPLIT mirrors it
constexpr uint64_t kConvSplitMinOutputs = 1024;
// ... and the fewest fp32 input elements whose quantisation into a code plane
// (esp32_conv::prepare_part) it shares too: VWW's and MobileNetV2-96's
// 3 x 96 x 96 input planes (about 4 ms on one core) are; DS-CNN's 490 are not.
constexpr uint32_t kConvSplitMinPlane = 4096;

/// Whether an op on the path splits (min-latency): never while kConvSplit is
/// off; else two units to share (channels of a depthwise conv, positions of a
/// dense one) and enough outputs to pay for the handoff.
static bool conv_fast_splittable(const esp32_conv::Op& geo) {
  return kConvSplit && esp32_conv::units(geo) >= 2 &&
         (uint64_t)geo.n * geo.m >= kConvSplitMinOutputs;
}

/// The bytes of its own scratch the split worker needs for its part of op i
/// (0 when the op does not take the path, cannot split, or needs none).
static uint64_t conv_split_scratch(const CompiledBlob* blob, uint32_t i) {
  if (!conv_fast_op(blob, i)) {
    return 0;
  }
  const OpDesc& op = blob->ops[i];
  const ConvGeom& g = *blob->conv[i];
  const esp32_conv::Op geo = conv_fast_geometry(op, g);
  if (!conv_fast_splittable(geo)) {
    return 0;
  }
  const uint32_t rows = is_depthwise(g) ? 0 : conv_fast_rows(op);
  return esp32_conv::part_scratch_need(
      geo, !esp32_asym::codes_in(blob->asym, i), rows);
}
#endif

/// Where a depthwise op keeps one output position's tap offsets (int32, one
/// per tap): in the xq scratch after its row of codes, 16-byte aligned.
static inline uint32_t depthwise_taps_at(uint32_t k_pad) {
  return (k_pad + kVecBytes - 1) & ~(kVecBytes - 1);
}
static inline int32_t* depthwise_taps(int8_t* xq, uint32_t k_pad) {
  return reinterpret_cast<int32_t*>(xq + depthwise_taps_at(k_pad));
}

/// xq, sxv, conv_stage, wu/wu2 and xq16: the quantised-activation scratch,
/// sized from the SAME per-op row block execute picks (row_block_for), now
/// that conv geometry is known. The requirement is max_i(B_i * k_pad_i), NOT
/// max_i(B_i) * max_i(k_pad_i): the tallest and the widest op need not be
/// the same op. Embedding gathers touch none of this scratch and are left
/// out of every maximum here: letting their k_pad in would size 25 KB of
/// temp for a 3,136-wide PLE table that reads none of it.
static void reserve_row_block_scratch(CompiledBlob* blob,
                                      ScratchLayout* layout,
                                      uint32_t conv_ops) {
  const uint32_t n_ops = blob->n_ops;
  uint32_t max_kpad = 0;
  bool wants_i4 = false, wants_i16 = false;
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    wants_i4 |= executes_i4(op);
    wants_i16 |= (op.opcode == kOpAddmmI16);
    if (!is_embed(op.opcode) && op.k_pad > max_kpad) {
      max_kpad = op.k_pad;
    }
  }
  if (max_kpad == 0) {
    return;
  }
  uint32_t block_rows = 1;
  uint64_t xq_bytes = max_kpad;         // single-row users: bmm, depthwise
  uint64_t conv_stage_bytes = 0;
#if ESP32_CONV_FAST
  uint64_t xq_general = xq_bytes;  // what the general loops alone reserve
  uint64_t split_bytes = 0;        // the split worker's part, at most
#endif
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    if (is_embed(op.opcode) || is_bmm(op.opcode)) {
      continue;
    }
    const ConvGeom* cg = blob->conv[i];
    if ((cg != nullptr && op.opcode == kOpAddmm) ||
        esp32_float::is_matmul(op.opcode)) {
      // A float-weight loop takes what esp32_float::scratch_bytes says: a
      // conv's patch (or taps), one position at a time, and a 2-byte
      // weight's widened rows. It blocks no rows and stages nothing.
      const uint64_t need = esp32_float::scratch_bytes(
          esp32_float::storage_of(op.opcode), cg != nullptr, op.m, op.k,
          op.n);
      if (need > xq_bytes) {
        xq_bytes = need;
      }
      continue;
    }
    const uint32_t b = row_block_for(op.m, op.k_pad, cg, op.n);
    if (b > block_rows) {
      block_rows = b;
    }
    uint64_t need = (uint64_t)b * op.k_pad;
    if (cg != nullptr && is_depthwise(*cg)) {
      // One row of codes, then one position's tap offsets (depthwise_taps).
      const uint64_t taps =
          depthwise_taps_at(op.k_pad) + (uint64_t)op.k * sizeof(int32_t);
      need = taps > need ? taps : need;
    }
#if ESP32_CONV_FAST
    if (need > xq_general) {
      xq_general = need;
    }
    if (conv_fast_op(blob, i)) {
      ++blob->conv_fast_ops;
      const uint64_t fast = conv_fast_scratch(blob, i);
      need = fast > need ? fast : need;
      const uint64_t part = conv_split_scratch(blob, i);
      split_bytes = part > split_bytes ? part : split_bytes;
    }
#endif
    if (need > xq_bytes) {
      xq_bytes = need;
    }
    // The conv staging buffer is sized over the convs that scatter through
    // it; depthwise takes its own per-channel loop and stages nothing.
    if (cg != nullptr && !is_depthwise(*cg)) {
      const uint64_t stage = (uint64_t)b * op.n * sizeof(float);
      if (stage > conv_stage_bytes) {
        conv_stage_bytes = stage;
      }
    }
  }
  blob->xq_off = layout->reserve(xq_bytes);
  blob->xq_rows = block_rows;
#if ESP32_CONV_FAST
  // xq is reserved first, so the path's extra is the region's growth.
  const uint64_t vec = kVecBytes - 1;
  blob->conv_fast_extra = (uint32_t)(((xq_bytes + vec) & ~vec) -
                                     ((xq_general + vec) & ~vec));
  // None while kConvSplit is off. The split worker's scratch for a conv, for
  // a handle that can split from init: one compiled for min_latency, or one
  // initialised while the runtime option forces the split on. conv_split()
  // splits no conv whose part needs it on a handle without it, so a single-
  // core handle pays nothing; one that gains min_latency after init splits
  // its matmuls and the convs whose part needs none. It takes the conv
  // staging buffer's bytes: the general loops stage a conv's block there and
  // the path never does, and each op takes one or the other, so the two never
  // hold data at the same time. The buffer is reserved here, in its place
  // after xq, as large as either needs.
  if (split_bytes > 0 &&
      (blob->cfg_min_latency ||
       g_split_override.load(std::memory_order_relaxed) > 0)) {
    const uint64_t stage =
        split_bytes > conv_stage_bytes ? split_bytes : conv_stage_bytes;
    blob->conv_stage_off = layout->reserve(stage);
    blob->conv_split_off = blob->conv_stage_off;
    blob->conv_fast_extra += (uint32_t)(((stage + vec) & ~vec) -
                                        ((conv_stage_bytes + vec) & ~vec));
    conv_stage_bytes = 0;  // reserved
  }
  if (blob->conv_fast_ops > 0) {
    esp32_conv::pie_dw_init();  // once a process: the PIE depthwise check
  }
#endif
  if (conv_stage_bytes > 0) {
    // One block of conv output, staged contiguously before being scattered
    // to NCHW. See CompiledBlob::conv_stage for why the scatter exists.
    blob->conv_stage_off = layout->reserve(conv_stage_bytes);
  }
  // sxv is per-ROW scale storage for a block. Conv always takes the blocked
  // loop -- including a 1x1 output, where the block is legitimately 1 --
  // so it needs sxv even when a matmul-only partition would not.
  if (block_rows > 1 || conv_ops > 0) {
    blob->sxv_off = layout->reserve(sizeof(float) * block_rows);
  }
  // Only pay for these when the partition actually holds such an op: a model
  // at the default i8 policy reserves neither the i4 unpack scratch nor the
  // i16 activation scratch.
  if (wants_i4) {
    blob->wu_off = layout->reserve(max_kpad);
    blob->wu2_off = layout->reserve(max_kpad);
  }
  if (wants_i16) {
    blob->xq16_off = layout->reserve((uint64_t)max_kpad * 2);
  }
}

/// Intra-partition chains: size the scratch slots and validate the chain
/// structure ONCE, here, so execute never has to. Each sentinel read must
/// see a preceding write of the same slot whose element count matches this
/// op's input; the AOT layer guarantees it, and a blob that lies fails to
/// load rather than computing quietly on stale floats. A slot holds floats,
/// or Stage B's int8 codes when its writer writes codes, and
/// its reader must read what was written; slots are sized in bytes, so a
/// chain of codes takes a quarter of the scratch.
static Error reserve_chain_scratch(CompiledBlob* blob, ScratchLayout* layout) {
  const uint32_t n_ops = blob->n_ops;
  uint32_t need = 0, used = 0;
  // The widest output each slot holds, in bytes: floats, or Stage B's codes.
  uint64_t slot_bytes[kNumScratch] = {};
  int32_t writer[kNumScratch];
  for (uint32_t s = 0; s < kNumScratch; ++s) {
    writer[s] = -1;
  }
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    if ((op.flags & kFlagStaticAct) && op.opcode == kOpAddmmI16) {
      // The AOT layer refuses this combination; a blob carrying it anyway is
      // corrupt, and the i16 loop would silently ignore the flag.
      ET_LOG(Error, "esp32: op %u: static act scales are int8-range only",
             (unsigned)i);
      return Error::InvalidProgram;
    }
    if (op.in_arg >= kArgScratch0) {
      const int s = (int)(op.in_arg - kArgScratch0);
      if (s < 0 || (uint32_t)s >= kNumScratch) {
        ET_LOG(Error, "esp32: op %u reads scratch slot %d of %u",
               (unsigned)i, s, (unsigned)kNumScratch);
        return Error::InvalidProgram;
      }
      if (is_embed(op.opcode)) {
        ET_LOG(Error, "esp32: op %u: embedding cannot read scratch",
               (unsigned)i);
        return Error::InvalidProgram;
      }
      if (writer[s] < 0) {
        ET_LOG(Error, "esp32: op %u reads scratch slot %d before any write",
               (unsigned)i, s);
        return Error::InvalidProgram;
      }
      const OpDesc& p = blob->ops[writer[s]];
      // What this op READS must be exactly what the slot's writer WROTE.
      // Both sides go through the named helpers: comparing m*k for a conv
      // would report "op 0 writes 49x8, op 1 reads 49x72" on a well-formed
      // chain, and a grouped conv writer owes the WHOLE output tensor, not
      // its own channel slice.
      const uint64_t reads = op_input_elems(op, blob->conv[i]);
      const uint64_t wrote = op_output_elems(p, blob->conv[writer[s]]);
      if (wrote != reads) {
        ET_LOG(Error, "esp32: chain shape mismatch: op %d writes %llu, "
                      "op %u reads %llu",
               (int)writer[s], (unsigned long long)wrote, (unsigned)i,
               (unsigned long long)reads);
        return Error::InvalidProgram;
      }
      if (esp32_asym::codes_in(blob->asym, i) !=
          esp32_asym::codes_out(blob->asym, (uint32_t)writer[s])) {
        ET_LOG(Error, "esp32: chain dtype mismatch: op %d writes %s, op %u "
                      "reads %s", (int)writer[s],
               esp32_asym::codes_out(blob->asym, (uint32_t)writer[s])
                   ? "int8 codes" : "floats",
               (unsigned)i,
               esp32_asym::codes_in(blob->asym, i) ? "int8 codes" : "floats");
        return Error::InvalidProgram;
      }
    }
    if ((op.flags & kFlagMul) && op.res_arg == op.out_arg) {
      // An in-place multiply reads what an earlier op of this partition
      // wrote to its output (the gate projection of a gated pair), so that
      // writer must exist and must have written exactly this op's extent.
      int32_t w = -1;
      if (op.out_arg >= kArgScratch0) {
        w = writer[op.out_arg - kArgScratch0];
      } else {
        for (uint32_t j = i; j-- > 0;) {
          if (blob->ops[j].out_arg == op.out_arg) {
            w = (int32_t)j;
            break;
          }
        }
      }
      if (w < 0 || op_output_elems(blob->ops[w], blob->conv[w]) !=
                       op_output_elems(op, blob->conv[i])) {
        ET_LOG(Error, "esp32: op %u multiplies its output 0x%x in place, "
                      "but no earlier op wrote it at this size",
               (unsigned)i, (unsigned)op.out_arg);
        return Error::InvalidProgram;
      }
    }
    if (op.out_arg >= kArgScratch0) {
      // An embedding may write a slot: `Embedding -> Linear` is how most
      // models begin, and refusing it here (after the AOT had claimed and
      // grouped both) would turn a fusion into a dead export.
      const int s = (int)(op.out_arg - kArgScratch0);
      if (s < 0 || (uint32_t)s >= kNumScratch) {
        ET_LOG(Error, "esp32: op %u writes scratch slot %d of %u",
               (unsigned)i, s, (unsigned)kNumScratch);
        return Error::InvalidProgram;
      }
      writer[s] = (int32_t)i;
      if ((uint32_t)s + 1 > used) {
        used = (uint32_t)s + 1;
      }
      const uint64_t e = op_output_elems(op, blob->conv[i]);
      if (e > need) {
        if (e > 0xFFFFFFFFull / sizeof(float)) {
          // Truncating here would allocate a fraction of what execute
          // writes; the byte count below would wrap on the 32-bit target.
          ET_LOG(Error, "esp32: chain scratch %llu elems is unrepresentable",
                 (unsigned long long)e);
          return Error::InvalidProgram;
        }
        need = (uint32_t)e;
      }
      const uint64_t bytes =
          e * (esp32_asym::codes_out(blob->asym, i) ? 1u : sizeof(float));
      if (bytes > slot_bytes[s]) {
        slot_bytes[s] = bytes;  // e <= need: representable
      }
    }
  }
  if (need > 0) {
    // Only the slots this blob actually writes, each sized to the widest
    // output written to it: execute writes exactly op_output_elems into a
    // slot and every read is validated above against its writer's extent.
    // A chain ping-pongs (the AOT takes the lowest free slot), so a
    // MobileNet's wider pointwise outputs share one slot and its depthwise
    // outputs the other.
    for (uint32_t s = 0; s < used; ++s) {
      blob->chain_off[s] = layout->reserve(slot_bytes[s]);
    }
    blob->chain_elems = need;
  }
  return Error::Ok;
}

/// q@kT scratch: one batch of quantised key vectors plus per-vector scales,
/// sized to the widest dynamic bmm in this partition. A cached-K bmm (either
/// kind) reads its codes in place when a row is whole vectors (k == k_pad);
/// only a ragged row is staged here, zero-padded, and it never needs the
/// scale vector.
static Error reserve_bmm_scratch(CompiledBlob* blob, ScratchLayout* layout) {
  uint32_t kq_need = 0, sk_need = 0;
  for (uint32_t i = 0; i < blob->n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    if (!is_bmm(op.opcode) ||
        (op.opcode != kOpBmmDynI8 && op.k == op.k_pad)) {
      continue;
    }
    const uint64_t kq64 = (uint64_t)op.n * op.k_pad;
    if (kq64 > 0xFFFFFFFFull) {
      ET_LOG(Error, "esp32: bmm scratch %llu B is unrepresentable",
             (unsigned long long)kq64);
      return Error::InvalidProgram;
    }
    const uint32_t kq = (uint32_t)kq64;
    if (kq > kq_need) {
      kq_need = kq;
    }
    if (op.opcode == kOpBmmDynI8 && op.n > sk_need) {
      sk_need = op.n;
    }
  }
  if (kq_need > 0) {
    blob->bmm_kq_off = layout->reserve(kq_need);
    blob->bmm_sk_off = layout->reserve((uint64_t)sk_need * sizeof(float));
  }
  return Error::Ok;
}

/// Whether an op can share its quantised activation rows with another op
/// reading the same input. A scratch INPUT never groups: a slot is REFILLED
/// between chains, so two ops reading the same slot id are not reading the
/// same bytes. A scratch OUTPUT may -- a gated pair writes its product to a
/// slot the down projection reads -- but only where hoist_safe() allows it.
/// Conv ops are excluded because the premise does not hold for them:
/// a conv's row is a GATHER, and two grouped convs over one tensor read
/// different channel slices for the same row (the grouped-conv parity row
/// would catch a row shared between slices). Selected-row
/// heads and i16 have their own paths and scratch, and so do the
/// affine-weight ops, whose column loops are not the group's.
static bool can_share_activation(const CompiledBlob* blob, uint32_t i) {
  const OpDesc& op = blob->ops[i];
  return (op.opcode == kOpAddmmI8 || op.opcode == kOpAddmmI4) &&
         blob->conv[i] == nullptr && blob->selected[i] == nullptr &&
         op.in_arg < kArgScratch0 && !esp32_asym::affine_op(blob->asym, i);
}

/// The argument op `k` reads besides its input, if any: a fused residual or
/// multiply operand, a dynamic bmm's key side, or a selected head's row IDs.
static bool second_read(const CompiledBlob* blob, uint32_t k, uint32_t* arg) {
  const OpDesc& op = blob->ops[k];
  if (op.opcode == kOpBmmDynI8 || (op.flags & (kFlagAdd | kFlagMul))) {
    *arg = op.res_arg;
    return true;
  }
  if (blob->selected[k] != nullptr) {
    *arg = blob->selected[k]->rows_arg;
    return true;
  }
  return false;
}

/// A group runs every member when its leader is reached, which HOISTS member
/// j above each op between leader i and j that is not itself a member (the
/// first `count` entries of `members`, the leader first). Sound only when j
/// has no dependence on any of them: none may write what j reads, nor read or
/// write what j writes. Arguments compare as ids -- a boundary index or a
/// scratch slot names one buffer for the whole call. A dependence on an
/// earlier MEMBER is kept, because members still run in order, row block by
/// row block -- provided both index the shared buffer at the same row width,
/// so that block b of one is block b of the other. That is the gated pair:
/// up multiplies, in place, the rows of act(gate) its leader has just written.
static bool hoist_safe(const CompiledBlob* blob, uint32_t i, uint32_t j,
                       const uint16_t* members, uint16_t count) {
  const OpDesc& oj = blob->ops[j];
  uint32_t j_second = 0;
  const bool j_reads = second_read(blob, j, &j_second);
  for (uint32_t k = i; k < j; ++k) {
    bool member = false;
    for (uint16_t t = 0; t < count && !member; ++t) {
      member = members[t] == k;
    }
    const OpDesc& ok = blob->ops[k];
    uint32_t k_second = 0;
    const bool k_reads = second_read(blob, k, &k_second);
    const bool writes_j_reads =
        ok.out_arg == oj.in_arg || (j_reads && ok.out_arg == j_second);
    const bool touches_j_writes = ok.out_arg == oj.out_arg ||
        ok.in_arg == oj.out_arg || (k_reads && k_second == oj.out_arg);
    if (!(writes_j_reads || touches_j_writes)) {
      continue;
    }
    if (!member || ok.n != oj.n || ok.m != oj.m) {
      return false;
    }
  }
  return true;
}

/// can_share_activation's rule for the affine ops, which share only with each
/// other: their loops are run_affine_group's. Not the int16 loop's ops
/// (run_addmm_a16), whose codes are int16.
static bool can_share_affine(const CompiledBlob* blob, uint32_t i) {
  const OpDesc& op = blob->ops[i];
  return (op.opcode == kOpAddmmI8 || op.opcode == kOpAddmmI4) &&
         blob->conv[i] == nullptr && blob->selected[i] == nullptr &&
         op.in_arg < kArgScratch0 && esp32_asym::affine_op(blob->asym, i) &&
         !esp32_asym::a16_op(blob->asym, i) &&
         // Stage B's codes: nothing to quantise once to share on the way in,
         // and run_affine_group writes floats.
         !esp32_asym::codes_in(blob->asym, i) &&
         !esp32_asym::codes_out(blob->asym, i);
}

/// Whether affine ops i and j quantise one input to the same codes, scales,
/// zero points and per-group sums: identical activation records (or both the
/// backend's own quantiser, at one scheme), and the same row and group
/// widths, which X_g is summed over. Never across schemes: a mixed group
/// would run every member at its leader's.
static bool same_affine_activation(const CompiledBlob* blob, uint32_t i,
                                   uint32_t j) {
  const OpDesc& oi = blob->ops[i];
  const OpDesc& oj = blob->ops[j];
  const esp32_asym::ActQuant* ai = esp32_asym::act_record(blob->asym, i);
  const esp32_asym::ActQuant* aj = esp32_asym::act_record(blob->asym, j);
  if ((ai == nullptr) != (aj == nullptr) ||
      (ai != nullptr && std::memcmp(ai, aj, sizeof(*ai)) != 0)) {
    return false;
  }
  constexpr uint32_t kActScheme = kFlagStaticAct | kFlagActPerTensor;
  if ((oi.flags & kActScheme) != (oj.flags & kActScheme) ||
      ((oi.flags & kFlagStaticAct) && oi.act_scale != oj.act_scale)) {
    return false;
  }
  const uint32_t gi = oi.scale_group ? oi.scale_group : oi.k_pad;
  const uint32_t gj = oj.scale_group ? oj.scale_group : oj.k_pad;
  return oi.in_arg == oj.in_arg && oi.k == oj.k && oi.k_pad == oj.k_pad &&
         gi == gj;
}

/// build_activation_groups' second pass: the affine ops' groups, appended
/// after the symmetric ones, under the same hoisting rule.
static void group_affine_ops(CompiledBlob* blob, uint16_t* mcur) {
  for (uint32_t i = 0; i < blob->n_ops; ++i) {
    if (!can_share_affine(blob, i) || blob->op_group[i] != -1) {
      continue;
    }
    const uint16_t start = *mcur;
    blob->group_members[(*mcur)++] = (uint16_t)i;
    for (uint32_t j = i + 1; j < blob->n_ops && *mcur - start < kMaxActGroup;
         ++j) {
      if (can_share_affine(blob, j) && blob->op_group[j] == -1 &&
          same_affine_activation(blob, i, j) &&
          hoist_safe(blob, i, j, blob->group_members + start,
                     (uint16_t)(*mcur - start))) {
        blob->group_members[(*mcur)++] = (uint16_t)j;
      }
    }
    if (*mcur - start >= 2) {
      for (uint16_t t = start; t < *mcur; ++t) {
        blob->op_group[blob->group_members[t]] = (int16_t)blob->n_groups;
      }
      blob->group_first[blob->n_groups] = start;
      blob->group_count[blob->n_groups] = (uint16_t)(*mcur - start);
      ++blob->n_groups;
    } else {
      *mcur = start;
    }
  }
}

/// Build the activation-quant sharing groups. O(n_ops^2) over a handful of
/// descriptors, once per load. Key is (in_arg, k_pad, activation scheme):
/// the same tensor at the same padded width quantises to the same bytes
/// only when both ops also choose the same scale.
static Error build_activation_groups(BackendInitContext& context,
                                     CompiledBlob* blob) {
  const uint32_t n_ops = blob->n_ops;
  MemoryAllocator* alloc = context.get_runtime_allocator();
  blob->op_group = alloc->allocateList<int16_t>(n_ops);
  blob->group_members = alloc->allocateList<uint16_t>(n_ops);
  blob->group_first = alloc->allocateList<uint16_t>(n_ops);
  blob->group_count = alloc->allocateList<uint16_t>(n_ops);
  if (blob->op_group == nullptr || blob->group_members == nullptr ||
      blob->group_first == nullptr || blob->group_count == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  for (uint32_t i = 0; i < n_ops; ++i) {
    blob->op_group[i] = -1;
  }
  blob->n_groups = 0;
  uint16_t mcur = 0;
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDesc& oi = blob->ops[i];
    if (!can_share_activation(blob, i) || blob->op_group[i] != -1) {
      continue;
    }
    const uint16_t start = mcur;
    blob->group_members[mcur++] = (uint16_t)i;
    for (uint32_t j = i + 1; j < n_ops && mcur - start < kMaxActGroup; ++j) {
      const OpDesc& oj = blob->ops[j];
      // Static-scale ops only group with ops at the SAME calibrated scale;
      // PT2E per-tensor ops only group with other per-tensor ops. Omitting
      // the latter would let a mixed branch silently use the leader's scheme.
      constexpr uint32_t kActScheme = kFlagStaticAct | kFlagActPerTensor;
      if (can_share_activation(blob, j) && blob->op_group[j] == -1 &&
          oj.in_arg == oi.in_arg && oj.k_pad == oi.k_pad &&
          (oj.flags & kActScheme) == (oi.flags & kActScheme) &&
          (!(oi.flags & kFlagStaticAct) || oj.act_scale == oi.act_scale) &&
          hoist_safe(blob, i, j, blob->group_members + start,
                     (uint16_t)(mcur - start))) {
        blob->group_members[mcur++] = (uint16_t)j;
      }
    }
    if (mcur - start >= 2) {
      for (uint16_t t = start; t < mcur; ++t) {
        blob->op_group[blob->group_members[t]] = (int16_t)blob->n_groups;
      }
      blob->group_first[blob->n_groups] = start;
      blob->group_count[blob->n_groups] = (uint16_t)(mcur - start);
      ++blob->n_groups;
    } else {
      mcur = start;  // alone: standalone path, no group bookkeeping
    }
  }
  if (blob->asym != nullptr) {
    group_affine_ops(blob, &mcur);
  }
  return Error::Ok;
}

/// The load-time report: one line per op with its work and traffic.
[[maybe_unused]] static void print_init_report(
    const CompiledBlob* blob, size_t blob_size) {
  printf("  [esp32 backend] init: %u op(s), %u B blob, %u B temp\n",
         (unsigned)blob->n_ops, (unsigned)blob_size,
         (unsigned)blob->scratch_bytes);
  if (blob->n_groups > 0) {
    printf("  [esp32 backend] actq: %u shared-input group(s)\n",
           (unsigned)blob->n_groups);
  }
#if ESP32_CONV_FAST
  if (blob->conv_fast_ops > 0) {
    printf("  [esp32 backend] convfast: %u conv op(s) on the static int8 path, "
           "%u B of the temp for it\n",
           (unsigned)blob->conv_fast_ops, (unsigned)blob->conv_fast_extra);
  }
#endif
  for (uint32_t i = 0; i < blob->n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    if (is_bmm(op.opcode)) {
      // No weights: an intensity figure would be fiction, so say what it is.
      printf("    op%u bmm_qk %ux%ux%ux%u: %s, no weights\n",
             (unsigned)i, (unsigned)op.scale_group, (unsigned)op.m,
             (unsigned)op.k, (unsigned)op.n,
             op.opcode == kOpBmmKcacheI8 ? "int8 K cache read in place"
             : op.opcode == kOpBmmKcacheAffineI8
                 ? "affine int8 K cache read in place"
                 : "dynamic both sides");
      continue;
    }
    if (is_embed(op.opcode)) {
      // A gather, not a matmul: M rows of one table row each, no MACs.
      // Printing it through the matmul line below would report the 13 MB PLE
      // table as an 830M-MAC op that reads the whole table per step.
      const uint64_t rb = esp32_float::is_embed(op.opcode)
          ? (uint64_t)op.k * sizeof(uint16_t)
          : row_bytes(op.opcode, op.k_pad);
      printf("    op%u embed %u row(s) x %u of %u: %llu B gathered, "
             "%llu B table\n",
             (unsigned)i, (unsigned)op.m, (unsigned)op.k, (unsigned)op.n,
             (unsigned long long)((uint64_t)op.m * rb),
             (unsigned long long)((uint64_t)op.n * rb));
      continue;
    }
    const uint32_t exec_n =
        blob->selected[i] != nullptr ? blob->selected[i]->count : op.n;
    const uint64_t macs = (uint64_t)op.m * op.k * exec_n;
    const uint64_t wb = (uint64_t)exec_n * execution_row_bytes(op);
    // Intensity is exactly M for int8 weights: one byte per MAC, reused
    // across M rows. Printed per op because M is what decides whether the
    // vector unit can matter, and M differs per model (1 at decode, T at
    // prefill).
    if (wb == 0) {
      continue;  // 64-bit divide by zero is a fatal exception on Xtensa
    }
    printf("    op%u %ux%ux%u%s%s%s: %llu MACs, %llu runtime wt B, "
           "intensity %llu.%02llu\n",
           (unsigned)i, (unsigned)op.m, (unsigned)op.k, (unsigned)op.n,
           (op.flags & kFlagStageI8) ? " staged-i8" : "",
           (op.flags & kFlagHeadSplit) ? " head-split" : "",
           blob->selected[i] != nullptr ? " selected" : "",
           (unsigned long long)macs, (unsigned long long)wb,
           (unsigned long long)(macs * 100 / wb / 100),
           (unsigned long long)(macs * 100 / wb % 100));
  }
}

// ============================================================================
// Executing a call, one path per function. `Esp32Backend::execute()` below
// binds this call's scratch, resolves the split posture once, and hands each
// descriptor to the function that owns its arithmetic, in this order: the
// embedding gathers (float, affine, symmetric), the three q@kT ops, fp32 addmm,
// the float-weight loops, i16 addmm, then the quantised i8/i4 family (affine
// ops, selected rows, sharing groups, depthwise, row-blocked, row-at-a-time).
// ============================================================================

/// Everything one execute() call shares across its ops.
struct CallContext {
  CompiledBlob* blob;        // this call's copy, with scratch pointers bound
  Span<EValue*> args;
  bool force_scalar;         // the A/B override: PIE off even when aligned
  bool global_split;         // min-latency posture: every wide addmm forks
  bool selective_split;      // head-split posture: only kFlagHeadSplit
  bool split_enabled;        // the worker lease was taken for this call
};

/// The resolved tensors of a matmul-family op (everything but a gather or a
/// dynamic bmm): fp32 in and out, the bias, and the fused residual if any.
/// A numel of -1 marks a chain-scratch leg, validated at init and not here.
struct MatmulIo {
  const float* x;
  float* y;
  int64_t in_numel;
  int64_t out_numel;
  const float* bias;
  // The fused residual (kFlagAdd) or multiply operand (kFlagMul); op.flags
  // says which, and apply_operand() applies it.
  const float* residual;
  // Stage B: the op's int8 codes in or out, set instead of x
  // or y (resolve_codes_io); only the affine loops read them.
  const int8_t* x8;
  int8_t* y8;
};

/// The call's memory-planned scratch (ExecuTorch #21985), or null. A program
/// exported with plan_scratch declares one buffer per delegate: the layout
/// init built (scratch_bytes, which scratch_layout.py mirrors) plus
/// kVecBytes - 1 bytes of slack. The planner aligns offsets within the arena,
/// but the arena's base is the integrator's (a heap_caps_malloc'd one is only
/// 4-byte aligned), so the layout starts at the buffer's first 16-byte
/// boundary, as BackendExecutionContext::scratch_buffers() asks of a backend
/// that needs more. The method then states its whole memory need at load.
/// Null otherwise, and the caller takes the temp allocator: always for a
/// program that planned none, and, logged once and counted, for one buffer
/// the aligned layout does not fit in, which means scratch_layout.py and this
/// file disagree.
static uint8_t* planned_scratch(BackendExecutionContext& context,
                                uint32_t need) {
  const Span<const Span<uint8_t>> planned = context.scratch_buffers();
  if (planned.size() == 0) {
    ++g_scratch_temp;
    return nullptr;
  }
  if (planned.size() == 1) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(planned[0].data());
    const size_t pad =
        (size_t)((kVecBytes - (base & (kVecBytes - 1))) & (kVecBytes - 1));
    if (planned[0].size() >= pad && planned[0].size() - pad >= need) {
      ++g_scratch_planned;
      return planned[0].data() + pad;
    }
  }
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true)) {
    ET_LOG(Error,
           "esp32: planned scratch unusable (%u buffer(s), the first %u B at "
           "%p; the layout needs %u B from a 16-byte boundary): using the temp "
           "allocator",
           (unsigned)planned.size(), (unsigned)planned[0].size(),
           static_cast<void*>(planned[0].data()), (unsigned)need);
  }
  ++g_scratch_unusable;
  return nullptr;
}

/// Point this call's copy of the handle at its slice of the per-call scratch:
/// the planned buffer, or else the per-instruction temporary arena. Every
/// consumer-visible byte is written before it is read
/// (quantisers zero their K..k_pad tails, unpack fills k_pad, chain slots are
/// validated producer-before-consumer, bmm fills every key row), so nothing
/// is cleared: clearing a 0.5--1 MiB global-attention buffer per token would
/// be pure bandwidth with no semantic effect.
static Error bind_call_scratch(BackendExecutionContext& context,
                               const CompiledBlob& stored, CompiledBlob* call) {
  if (stored.scratch_bytes == 0) {
    return Error::Ok;
  }
  uint8_t* temp = planned_scratch(context, stored.scratch_bytes);
  if (temp == nullptr) {
    MemoryAllocator* temp_allocator = context.get_temp_allocator();
    if (temp_allocator == nullptr) {
      ET_LOG(Error,
             "esp32: %u B delegate temp required but MemoryManager has no "
             "temp allocator",
             (unsigned)stored.scratch_bytes);
      return Error::MemoryAllocationFailed;
    }
    temp = static_cast<uint8_t*>(
        temp_allocator->allocate(stored.scratch_bytes, kVecBytes));
    if (temp == nullptr) {
      ET_LOG(Error, "esp32: delegate needs %u B temporary scratch",
             (unsigned)stored.scratch_bytes);
      return Error::MemoryAllocationFailed;
    }
  }
  auto at = [&](uint32_t off) -> uint8_t* {
    return off == kNoScratch ? nullptr : temp + off;
  };
  call->xq = reinterpret_cast<int8_t*>(at(stored.xq_off));
  call->conv_stage = reinterpret_cast<float*>(at(stored.conv_stage_off));
  call->wu = reinterpret_cast<int8_t*>(at(stored.wu_off));
  call->wu2 = reinterpret_cast<int8_t*>(at(stored.wu2_off));
  call->xq16 = reinterpret_cast<int16_t*>(at(stored.xq16_off));
  call->sxv = reinterpret_cast<float*>(at(stored.sxv_off));
  for (uint32_t s = 0; s < kNumScratch; ++s) {
    call->chain[s] = reinterpret_cast<float*>(at(stored.chain_off[s]));
  }
  call->bmm_kq = reinterpret_cast<int8_t*>(at(stored.bmm_kq_off));
  call->bmm_sk = reinterpret_cast<float*>(at(stored.bmm_sk_off));
#if ESP32_CONV_FAST
  call->conv_split = reinterpret_cast<int8_t*>(at(stored.conv_split_off));
#endif
  if (stored.asym != nullptr && stored.asym->terms_off != esp32_asym::kNoTerms) {
    call->asym_terms = reinterpret_cast<int32_t*>(temp + stored.asym->terms_off);
  }
  return Error::Ok;
}

/// Boundary args index into `args`; sentinels name the chain scratch slots.
/// Chain structure and shapes were validated at init, so scratch legs carry
/// no per-execute checks.
static Error check_op_args(const CallContext& c, uint32_t i) {
  const OpDesc& op = c.blob->ops[i];
  const bool in_scr = op.in_arg >= kArgScratch0;
  const bool out_scr = op.out_arg >= kArgScratch0;
  if (!in_scr && op.in_arg >= c.args.size()) {
    ET_LOG(Error, "esp32: op %u in_arg %u out of range (%u args)",
           (unsigned)i, (unsigned)op.in_arg, (unsigned)c.args.size());
    return Error::InvalidArgument;
  }
  if (!out_scr && op.out_arg >= c.args.size()) {
    ET_LOG(Error, "esp32: op %u out_arg %u out of range (%u args)",
           (unsigned)i, (unsigned)op.out_arg, (unsigned)c.args.size());
    return Error::InvalidArgument;
  }
  if ((in_scr || out_scr) && c.blob->chain[0] == nullptr) {
    ET_LOG(Error, "esp32: op %u chains but no chain scratch was allocated",
           (unsigned)i);
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

/// The PIE gate every quantised path applies: the kernel row it reads (the
/// unpack scratch for i4, the packed row for i8) and every group start must
/// be 16 B aligned, because ee.vld.128.ip truncates a misaligned address
/// silently rather than faulting. Checked rather than trusted.
static bool pie_row_aligned(bool force_scalar, const int8_t* kernel_row_base,
                            const int8_t* xq, uint32_t k_pad, uint32_t gsize) {
  return !force_scalar && kernel_row_base != nullptr && xq != nullptr &&
         (reinterpret_cast<uintptr_t>(kernel_row_base) % kVecBytes) == 0 &&
         (k_pad % kVecBytes) == 0 && (gsize % kVecBytes) == 0;
}

static float absmax_of(const float* x, size_t n) {
  float amax = 0.0f;
  for (size_t t = 0; t < n; ++t) {
    const float a = std::fabs(x[t]);
    if (a > amax) {
      amax = a;
    }
  }
  return amax;
}

/// The fixed activation scale an op quantises at, or -1 for per-row absmax.
/// kFlagStaticAct is the calibrated scale from the descriptor; kFlagActPerTensor
/// is the model's own PT2E scheme -- ONE scale over the whole activation,
/// computed here exactly as choose_qparams_symmetric computes it (absmax/127,
/// symmetric, no zero point, floored at the observer's eps) and then handed to
/// the ordinary fixed-scale path, which rounds it ties-to-even.
/// `n_act` is the activation's true extent: a conv's is its NCHW input, not
/// [M, K], and using M*K there would read past the tensor on any conv whose
/// patch count exceeds its input size, which is most of them.
ESP32_HOT_CODE
static float fixed_activation_scale(const OpDesc& op, const float* x,
                                    size_t n_act, uint32_t per_tensor_ops) {
  float fix = (op.flags & kFlagStaticAct) ? op.act_scale : -1.0f;
  if (op.flags & kFlagActPerTensor) {
    const float scale = absmax_of(x, n_act) / (float)kQmaxI8;
    fix = scale > kPt2eScaleEps ? scale : kPt2eScaleEps;
    g_act_per_tensor.fetch_add(per_tensor_ops, std::memory_order_relaxed);
  }
  return fix;
}

/// Hand the worker columns [0, half) of a block of rows. The job slot is
/// leased for the whole call, so the caller may compute [half, N) itself and
/// join_split() afterwards.
ESP32_HOT_CODE
static void fork_split(const CallContext& c, uint32_t i, const int8_t* xq,
                       const float* sxv, uint32_t nrows, const float* bias,
                       float* ybase, uint32_t y_stride, bool aligned,
                       const float* rbase, uint32_t half) {
  const OpDesc& op = c.blob->ops[i];
  g_job.op = &op;
  g_job.res = &c.blob->resolved[i];
  g_job.xq = xq;
  g_job.sxv = sxv;
  g_job.nrows = nrows;
  g_job.xq_stride = op.k_pad;
  g_job.wu = c.blob->wu2;
  g_job.bias = bias;
  g_job.ybase = ybase;
  g_job.y_stride = y_stride;
  g_job.aligned = aligned;
  g_job.rbase = rbase;
  g_job.affine = nullptr;
#if ESP32_CONV_FAST
  g_job.conv = nullptr;
#endif
  g_job.n0 = 0;
  g_job.n1 = half;
  start_split_job();
}

/// Wait for the worker's half and fold its column counts into the caller's.
/// Forks are counted in ROWS, the unit the harness's fork accounting reads.
ESP32_HOT_CODE
static void join_split(const OpDesc& op, uint32_t nrows, uint32_t* pie,
                       uint32_t* scalar) {
#if !defined(ESP32_HOST_BUILD)
  xSemaphoreTake(g_split_done, portMAX_DELAY);
#else
  esp32_host_split_join();
#endif
  *pie += g_job.pie;
  *scalar += g_job.scalar;
  g_split_jobs.fetch_add(nrows, std::memory_order_relaxed);
  if (op.flags & kFlagHeadSplit) {
    g_head_split_jobs.fetch_add(nrows, std::memory_order_relaxed);
  }
#if defined(ESP32_BACKEND_TIMING)
  g_split_worker_us.fetch_add(g_job.worker_us, std::memory_order_relaxed);
#endif
}

/// A gather, not a matmul: one row read per index, no accumulation, no
/// activation quantisation. The point is purely storage -- a per-layer
/// embedding table at fp32 can outgrow the 16 MB flash on its own, and int4
/// plus scales takes about an eighth of that. Indices are int64.
ESP32_HOT_CODE
static Error run_embedding(const CallContext& c, uint32_t i) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  // init refused a sentinel INPUT on embeddings (indices are a partition
  // input); the OUTPUT may be a chain slot, validated at init.
  const auto in = c.args[op.in_arg]->toTensor();
  const uint32_t M = op.m, D = op.k, V = op.n;
  float* yv;
  if (op.out_arg >= kArgScratch0) {
    yv = blob->chain[op.out_arg - kArgScratch0];
  } else {
    auto out = c.args[op.out_arg]->toTensor();
    if (out.numel() != (int64_t)((uint64_t)M * D)) {
      ET_LOG(Error, "esp32: embedding out %d != %ux%u", (int)out.numel(),
             (unsigned)M, (unsigned)D);
      return Error::InvalidArgument;
    }
    if (out.scalar_type() != executorch::aten::ScalarType::Float) {
      ET_LOG(Error, "esp32: embedding output must be float32");
      return Error::InvalidArgument;
    }
    yv = out.mutable_data_ptr<float>();
  }
  // The output check above says nothing about the INDEX tensor, which is
  // what the gather loop actually walks: a descriptor claiming M=6144
  // against a genuine 8-element index tensor reads 6144 int64s off a
  // 64-byte buffer. Blob-side validation cannot cover this -- it depends
  // on a runtime argument -- so it belongs here.
  if (in.numel() != (int64_t)M) {
    ET_LOG(Error, "esp32: embedding index numel %d != M %u",
           (int)in.numel(), (unsigned)M);
    return Error::InvalidArgument;
  }
  if (in.scalar_type() != executorch::aten::ScalarType::Long) {
    // const_data_ptr<int64_t> is an unchecked static_cast, so a narrower
    // index type is silently reinterpreted rather than refused.
    ET_LOG(Error, "esp32: embedding indices must be int64");
    return Error::InvalidArgument;
  }
  const int64_t* idx = in.const_data_ptr<int64_t>();
  const uint8_t* table =
      reinterpret_cast<const uint8_t*>(blob->resolved[i].wq);
  // One scale per group per row. Storage type comes from the flag, not the
  // opcode: a table whose rows fit in a single group stores fp32 so that a
  // TIED tensor produces bytes identical to the matmul path's and can be
  // deduplicated.
  const bool s_fp16 = (op.flags & kFlagFp16Scales) != 0;
  const uint16_t* tscale16 =
      reinterpret_cast<const uint16_t*>(blob->resolved[i].scale);
  const float* tscale32 = blob->resolved[i].scale;
  const uint32_t gsize = op.scale_group ? op.scale_group : D;
  const uint32_t ng = (D + gsize - 1) / gsize;  // final group may be short
  const size_t stride = row_bytes(op.opcode, op.k_pad);

  for (uint32_t j = 0; j < M; ++j) {
    const int64_t t = idx[j];
    if (t < 0 || t >= (int64_t)V) {
      ET_LOG(Error, "esp32: embedding index %d out of range [0,%u)", (int)t,
             (unsigned)V);
      return Error::InvalidArgument;
    }
    const uint8_t* row = table + (size_t)t * stride;
    const size_t soff = (size_t)t * ng;
    float* dst = yv + (size_t)j * D;
    // Walk group by group so the fp16 decode happens once per group rather
    // than once per weight -- ng is 6 for ple_table against 768 elements.
    for (uint32_t g = 0; g < ng; ++g) {
      const uint32_t a = g * gsize;
      const uint32_t b = (a + gsize < D) ? (a + gsize) : D;
      const float s = s_fp16 ? f16_to_f32(tscale16[soff + g])
                             : tscale32[soff + g];
      if (op.opcode == kOpEmbedI4) {
        for (uint32_t col = a; col < b; ++col) {
          const uint8_t byte = row[col >> 1];
          const int code = (col & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
          dst[col] = (float)(code - 8) * s;
        }
      } else if (op.opcode == kOpEmbedI16) {
        const int16_t* r16 = reinterpret_cast<const int16_t*>(row);
        for (uint32_t col = a; col < b; ++col) {
          dst[col] = (float)r16[col] * s;
        }
      } else {
        const int8_t* r8 = reinterpret_cast<const int8_t*>(row);
        for (uint32_t col = a; col < b; ++col) {
          dst[col] = (float)r8[col] * s;
        }
      }
    }
  }
  g_weight_bytes += (uint64_t)M * stride;  // gathered rows, not the whole table
  ++g_embed_ops;
  return Error::Ok;
}

/// Opt-in int8 q@kT. res_arg is the SECOND INPUT (the K side in natural
/// [B,N,K] layout, its transpose folded at export), NOT a residual;
/// scale_group is the batch count. Per batch: quantise the N key vectors
/// once, then every query row dots against them -- integer accumulate,
/// scores scaled (acc*sx)*sk in that order, matching models.bmm_qk_i8.
static Error run_bmm_dyn_i8(const CallContext& c, uint32_t i) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t Bn = op.scale_group, M = op.m, K = op.k, N = op.n;
  if (op.res_arg >= c.args.size()) {
    ET_LOG(Error, "esp32: bmm op %u res_arg out of range", (unsigned)i);
    return Error::InvalidArgument;
  }
  const auto a_t = c.args[op.in_arg]->toTensor();
  const auto k_t = c.args[op.res_arg]->toTensor();
  auto o_t = c.args[op.out_arg]->toTensor();
  if (a_t.numel() != (int64_t)((uint64_t)Bn * M * K) ||
      k_t.numel() != (int64_t)((uint64_t)Bn * N * K) ||
      o_t.numel() != (int64_t)((uint64_t)Bn * M * N)) {
    ET_LOG(Error, "esp32: bmm op %u shape mismatch", (unsigned)i);
    return Error::InvalidArgument;
  }
  if (a_t.scalar_type() != executorch::aten::ScalarType::Float ||
      k_t.scalar_type() != executorch::aten::ScalarType::Float ||
      o_t.scalar_type() != executorch::aten::ScalarType::Float) {
    ET_LOG(Error, "esp32: bmm inputs and output must be float32");
    return Error::InvalidArgument;
  }
  if (blob->bmm_kq == nullptr || blob->bmm_sk == nullptr ||
      blob->xq == nullptr) {
    ET_LOG(Error, "esp32: bmm op %u with no scratch", (unsigned)i);
    return Error::InvalidArgument;
  }
  const float* qa = a_t.const_data_ptr<float>();
  const float* kx = k_t.const_data_ptr<float>();
  float* yo = o_t.mutable_data_ptr<float>();
  const bool aligned = !c.force_scalar && (op.k_pad % kVecBytes) == 0;
  // Counted locally and published once: an atomic increment per score
  // column would put a sequentially consistent read-modify-write (and its
  // memory barriers) in the innermost loop, 512 per call at 8 queries x 64 keys.
  uint32_t pie_cols = 0, scalar_cols = 0;
  for (uint32_t b = 0; b < Bn; ++b) {
    const float* kb = kx + (size_t)b * N * K;
    for (uint32_t n = 0; n < N; ++n) {
      blob->bmm_sk[n] = quantize_row_i8(kb + (size_t)n * K, K, op.k_pad,
                                        blob->bmm_kq + (size_t)n * op.k_pad,
                                        -1.0f, false);
    }
    for (uint32_t m = 0; m < M; ++m) {
      const float sx = quantize_row_i8(qa + ((size_t)b * M + m) * K, K,
                                       op.k_pad, blob->xq, -1.0f, false);
      float* yrow = yo + ((size_t)b * M + m) * N;
      for (uint32_t n = 0; n < N; ++n) {
        const int8_t* krow = blob->bmm_kq + (size_t)n * op.k_pad;
        int32_t acc;
        if (aligned) {
          acc = dot_i8_s3(blob->xq, krow, (int)op.k_pad);
          ++pie_cols;
        } else {
          acc = 0;
          for (uint32_t t = 0; t < op.k_pad; ++t) {
            acc += (int32_t)blob->xq[t] * (int32_t)krow[t];
          }
          ++scalar_cols;
        }
        yrow[n] = (float)acc * sx * blob->bmm_sk[n];
      }
    }
  }
  g_pie_cols.fetch_add(pie_cols, std::memory_order_relaxed);
  g_scalar_cols.fetch_add(scalar_cols, std::memory_order_relaxed);
  g_macs += (uint64_t)Bn * M * K * N;
  ++g_bmm_ops;
  return Error::Ok;
}

/// int8 q@kT against a persistent int8 K cache. res_arg is the codes
/// [B,N,K] int8, scale_off the per-row scales [B,N] fp32 (both arg indices);
/// scale_group is the batch. The keys were quantised by
/// espressif::quantize_rows_i8 when written -- quantize_row_i8's arithmetic --
/// so this is run_bmm_dyn_i8 minus its key loop: the same integer dot of the
/// same codes, the same (acc*sx)*sk, bit for bit. Rows are dotted straight
/// from the cache when they are whole aligned vectors; a ragged row (k <
/// k_pad) is staged zero-padded in bmm_kq first, as the dynamic path stages
/// every row.
ESP32_HOT_CODE
static Error run_bmm_kcache_i8(const CallContext& c, uint32_t i) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t Bn = op.scale_group, M = op.m, K = op.k, N = op.n;
  if (op.res_arg >= c.args.size() || op.scale_off >= c.args.size()) {
    ET_LOG(Error, "esp32: kcache bmm op %u codes/scales arg out of range",
           (unsigned)i);
    return Error::InvalidArgument;
  }
  const auto a_t = c.args[op.in_arg]->toTensor();
  const auto k_t = c.args[op.res_arg]->toTensor();
  const auto s_t = c.args[op.scale_off]->toTensor();
  auto o_t = c.args[op.out_arg]->toTensor();
  if (a_t.numel() != (int64_t)((uint64_t)Bn * M * K) ||
      k_t.numel() != (int64_t)((uint64_t)Bn * N * K) ||
      s_t.numel() != (int64_t)((uint64_t)Bn * N) ||
      o_t.numel() != (int64_t)((uint64_t)Bn * M * N)) {
    ET_LOG(Error, "esp32: kcache bmm op %u shape mismatch", (unsigned)i);
    return Error::InvalidArgument;
  }
  if (a_t.scalar_type() != executorch::aten::ScalarType::Float ||
      k_t.scalar_type() != executorch::aten::ScalarType::Char ||
      s_t.scalar_type() != executorch::aten::ScalarType::Float ||
      o_t.scalar_type() != executorch::aten::ScalarType::Float) {
    ET_LOG(Error, "esp32: kcache bmm wants fp32 queries, int8 codes, fp32 "
                  "scales and an fp32 output");
    return Error::InvalidArgument;
  }
  const bool staged = K != op.k_pad;
  if (blob->xq == nullptr || (staged && blob->bmm_kq == nullptr)) {
    ET_LOG(Error, "esp32: kcache bmm op %u with no scratch", (unsigned)i);
    return Error::InvalidArgument;
  }
  const float* qa = a_t.const_data_ptr<float>();
  const int8_t* kc = k_t.const_data_ptr<int8_t>();
  const float* ks = s_t.const_data_ptr<float>();
  float* yo = o_t.mutable_data_ptr<float>();
  // Counted locally and published once, as in run_bmm_dyn_i8.
  uint32_t pie_cols = 0, scalar_cols = 0;
  for (uint32_t b = 0; b < Bn; ++b) {
    const int8_t* kb = kc + (size_t)b * N * K;
    const float* sk = ks + (size_t)b * N;
    const int8_t* rows = kb;
    if (staged) {
      for (uint32_t n = 0; n < N; ++n) {
        int8_t* dst = blob->bmm_kq + (size_t)n * op.k_pad;
        std::memcpy(dst, kb + (size_t)n * K, K);
        std::memset(dst + K, 0, op.k_pad - K);
      }
      rows = blob->bmm_kq;
    }
    // Every row starts 16-aligned iff the first does: row stride is k_pad.
    const bool aligned = !c.force_scalar && (op.k_pad % kVecBytes) == 0 &&
        (reinterpret_cast<uintptr_t>(rows) % kVecBytes) == 0;
    for (uint32_t m = 0; m < M; ++m) {
      // Half away from zero, as run_bmm_dyn_i8 quantises both of its sides
      // and as espressif::quantize_rows_i8 made the stored codes.
      const float sx = quantize_row_i8(qa + ((size_t)b * M + m) * K, K,
                                       op.k_pad, blob->xq, -1.0f, false);
      float* yrow = yo + ((size_t)b * M + m) * N;
      for (uint32_t n = 0; n < N; ++n) {
        const int8_t* krow = rows + (size_t)n * op.k_pad;
        int32_t acc;
        if (aligned) {
          acc = dot_i8_s3(blob->xq, krow, (int)op.k_pad);
          ++pie_cols;
        } else {
          acc = 0;
          for (uint32_t t = 0; t < op.k_pad; ++t) {
            acc += (int32_t)blob->xq[t] * (int32_t)krow[t];
          }
          ++scalar_cols;
        }
        yrow[n] = (float)acc * sx * sk[n];
      }
    }
  }
  g_pie_cols.fetch_add(pie_cols, std::memory_order_relaxed);
  g_scalar_cols.fetch_add(scalar_cols, std::memory_order_relaxed);
  g_macs += (uint64_t)Bn * M * K * N;
  g_bmm_kcache_rows.fetch_add((uint64_t)Bn * N, std::memory_order_relaxed);
  ++g_bmm_kcache_ops;
  return Error::Ok;
}

/// resolve_matmul_io's input and output for an op whose codes travel as codes
/// (Stage B): int8 codes in (in_dtype int8) from a boundary
/// tensor or a chain slot of codes, int8 codes out (out_dtype int8) likewise,
/// the other side fp32 as usual. Out of line: the symmetric path grows by one
/// test.
[[gnu::noinline]] static Error resolve_codes_io(const CallContext& c,
                                                uint32_t i, MatmulIo* io) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const bool cin = esp32_asym::codes_in(blob->asym, i);
  const bool cout = esp32_asym::codes_out(blob->asym, i);
  const auto want_in = cin ? executorch::aten::ScalarType::Char
                           : executorch::aten::ScalarType::Float;
  const auto want_out = cout ? executorch::aten::ScalarType::Char
                             : executorch::aten::ScalarType::Float;
  io->x = nullptr;
  io->y = nullptr;
  if (op.in_arg >= kArgScratch0) {
    float* slot = blob->chain[op.in_arg - kArgScratch0];
    if (cin) {
      io->x8 = reinterpret_cast<const int8_t*>(slot);
    } else {
      io->x = slot;
    }
  } else {
    const auto in = c.args[op.in_arg]->toTensor();
    if (in.scalar_type() != want_in) {
      ET_LOG(Error, "esp32: op %u input must be %s", (unsigned)i,
             cin ? "int8 codes" : "float32");
      return Error::InvalidArgument;
    }
    if (cin) {
      io->x8 = in.const_data_ptr<int8_t>();
    } else {
      io->x = in.const_data_ptr<float>();
    }
    io->in_numel = in.numel();
  }
  if (op.out_arg >= kArgScratch0) {
    float* slot = blob->chain[op.out_arg - kArgScratch0];
    if (cout) {
      io->y8 = reinterpret_cast<int8_t*>(slot);
    } else {
      io->y = slot;
    }
  } else {
    auto out = c.args[op.out_arg]->toTensor();
    if (out.scalar_type() != want_out) {
      ET_LOG(Error, "esp32: op %u output must be %s", (unsigned)i,
             cout ? "int8 codes" : "float32");
      return Error::InvalidArgument;
    }
    if (cout) {
      io->y8 = out.mutable_data_ptr<int8_t>();
    } else {
      io->y = out.mutable_data_ptr<float>();
    }
    io->out_numel = out.numel();
  }
  return Error::Ok;
}

/// Resolve the fp32 input and output (a boundary tensor or a chain slot),
/// the bias, and the fused residual of a matmul-family op. All runtime
/// arithmetic and output pointers are float*, but for Stage B's codes
/// (resolve_codes_io).
ESP32_HOT_CODE
static Error resolve_matmul_io(const CallContext& c, uint32_t i,
                               MatmulIo* io) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  io->in_numel = -1;
  io->out_numel = -1;
  io->x8 = nullptr;
  io->y8 = nullptr;
  if (blob->asym != nullptr && (esp32_asym::codes_in(blob->asym, i) ||
                                esp32_asym::codes_out(blob->asym, i))) {
    if (Error e = resolve_codes_io(c, i, io); e != Error::Ok) {
      return e;
    }
  } else {
    if (op.in_arg >= kArgScratch0) {
      io->x = blob->chain[op.in_arg - kArgScratch0];
    } else {
      const auto in = c.args[op.in_arg]->toTensor();
      if (in.scalar_type() != executorch::aten::ScalarType::Float) {
        ET_LOG(Error, "esp32: op %u input must be float32", (unsigned)i);
        return Error::InvalidArgument;
      }
      io->x = in.const_data_ptr<float>();
      io->in_numel = in.numel();
    }
    if (op.out_arg >= kArgScratch0) {
      io->y = blob->chain[op.out_arg - kArgScratch0];
    } else {
      auto out = c.args[op.out_arg]->toTensor();
      if (out.scalar_type() != executorch::aten::ScalarType::Float) {
        ET_LOG(Error, "esp32: op %u output must be float32", (unsigned)i);
        return Error::InvalidArgument;
      }
      io->y = out.mutable_data_ptr<float>();
      io->out_numel = out.numel();
    }
  }
  const bool is_quantised =
      (op.opcode == kOpAddmmI8 || op.opcode == kOpAddmmI4 ||
       op.opcode == kOpAddmmI16);
  if (op.opcode != kOpAddmm && !is_quantised &&
      !esp32_float::is_matmul(op.opcode)) {
    ET_LOG(Error, "esp32: unknown opcode %u", (unsigned)op.opcode);
    return Error::NotSupported;
  }
  io->bias = op.bias_off != 0
      ? reinterpret_cast<const float*>(blob->base + op.bias_off)
      : nullptr;
  // Fused residual: a partition input summed into the output after the
  // act/clamp epilogue. Row-major walk over an exact-shape tensor -- the AOT
  // gate refuses anything that would need broadcasting. A fused multiply's
  // operand arrives the same way, or is this op's own output slot (in place,
  // validated at init).
  io->residual = nullptr;
  if ((op.flags & kFlagMul) && op.res_arg >= kArgScratch0) {
    io->residual = blob->chain[op.res_arg - kArgScratch0];
  } else if (op.flags & (kFlagAdd | kFlagMul)) {
    if (op.res_arg >= c.args.size()) {
      ET_LOG(Error, "esp32: op %u res_arg %u out of range", (unsigned)i,
             (unsigned)op.res_arg);
      return Error::InvalidArgument;
    }
    const auto rt = c.args[op.res_arg]->toTensor();
    if (rt.numel() != (int64_t)((uint64_t)op.m * op.n)) {
      ET_LOG(Error, "esp32: residual numel %d != %ux%u", (int)rt.numel(),
             (unsigned)op.m, (unsigned)op.n);
      return Error::InvalidArgument;
    }
    if (rt.scalar_type() != executorch::aten::ScalarType::Float) {
      ET_LOG(Error, "esp32: op %u residual must be float32", (unsigned)i);
      return Error::InvalidArgument;
    }
    io->residual = rt.const_data_ptr<float>();
  }
  return Error::Ok;
}

/// Opcode 1: fp32 [K,N] weights, inline or external. A backend diagnostic,
/// not a deployment format; k-outer so the weight is walked front to back.
static Error run_addmm_fp32(const CallContext& c, uint32_t i,
                            const MatmulIo& io) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  if ((io.in_numel >= 0 && io.in_numel != (int64_t)((uint64_t)M * K)) ||
      (io.out_numel >= 0 && io.out_numel != (int64_t)((uint64_t)M * N))) {
    ET_LOG(Error, "esp32: shape mismatch: in %d out %d for %ux%ux%u",
           (int)io.in_numel, (int)io.out_numel, (unsigned)M, (unsigned)K,
           (unsigned)N);
    return Error::InvalidArgument;
  }
  const float* w = reinterpret_cast<const float*>(blob->resolved[i].wq);
  for (uint32_t m = 0; m < M; ++m) {
    float* yrow = io.y + (size_t)m * N;
    for (uint32_t n = 0; n < N; ++n) {
      yrow[n] = io.bias != nullptr ? io.bias[n] : 0.0f;
    }
    for (uint32_t k = 0; k < K; ++k) {
      const float xk = io.x[(size_t)m * K + k];
      const float* wrow = w + (size_t)k * N;
      for (uint32_t n = 0; n < N; ++n) {
        yrow[n] += xk * wrow[n];
      }
    }
    // k-outer accumulates in place, so fused act/clamp/residual are a final
    // pass over the row rather than a per-store test inside the hot loop.
    // The operand is always an add here: init refuses kFlagMul on fp32.
    if (op.flags & kEpilogueFlags) {
      for (uint32_t n = 0; n < N; ++n) {
        yrow[n] = epilogue(op, yrow[n], io.residual, (size_t)m * N + n);
      }
    }
  }
  count_op((uint64_t)M * K * N, (uint64_t)K * N * sizeof(float));
  return Error::Ok;
}

/// i16 depthwise: every output channel reads its own input channel. The
/// ordinary i16 loop quantises one gathered row and reuses it across output
/// columns, which is valid for dense/grouped conv and wrong here. Keep the
/// i16 accumulator/epilogue but gather per channel, exactly as the i4/i8
/// depthwise path does.
static void run_addmm_i16_depthwise(const CallContext& c, uint32_t i,
                                    const MatmulIo& io, const ConvGeom& g) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const int16_t* wq16 = reinterpret_cast<const int16_t*>(blob->resolved[i].wq);
  const float* wscale = blob->resolved[i].scale;
  const size_t plane = (size_t)g.d_in * g.h_in * g.w_in;
  ConvGeom one = g;
  one.c_in = 1;
  one.c_out = 1;
  one.groups = 1;
  one.group_idx = 0;
  one.c_out_total = 1;
  for (uint32_t ch = 0; ch < N; ++ch) {
    const float* xin = io.x + (size_t)ch * plane;
    const int16_t* wrow = wq16 + (size_t)ch * op.k_pad;
    float* ybase = io.y + (size_t)ch * M;
    for (uint32_t m = 0; m < M; ++m) {
      const float sx = quantize_conv_row_t<int16_t, kQmaxI16>(
          xin, one, m, op.k_pad, blob->xq16, -1.0f, false);
      int64_t acc = 0;
      for (uint32_t t = 0; t < op.k_pad; ++t) {
        acc += (int32_t)blob->xq16[t] * (int32_t)wrow[t];
      }
      const float v = (io.bias != nullptr ? io.bias[ch] : 0.0f)
          + (float)acc * sx * wscale[ch];
      // NCHW residual; always an add, as init refuses kFlagMul on a conv.
      ybase[m] = epilogue(op, v, io.residual, (size_t)ch * M + m);
    }
  }
  count_cols(0, M * N);   // every output element is one scalar column
  count_op((uint64_t)M * K * N, (uint64_t)N * row_bytes(op.opcode, op.k_pad));
}

/// The accuracy rung. Deliberately scalar: ee.vmulas.s16.accx exists (8
/// lanes x 40-bit in the 320-bit QACC against int8's 16 x 20-bit) but i16
/// doubles the weight traffic while halving the SIMD width, so it is never
/// the fast choice and a second hand-written PIE kernel would be earning
/// nothing. It is here for the ops where int8's error is the binding
/// constraint rather than bandwidth.
static Error run_addmm_i16(const CallContext& c, uint32_t i,
                           const MatmulIo& io) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const ConvGeom* cg16 = blob->conv[i];
  const uint64_t want_in16 = op_input_elems(op, cg16);
  const uint64_t want_out16 = op_output_elems(op, cg16);
  if ((io.in_numel >= 0 && io.in_numel != (int64_t)want_in16) ||
      (io.out_numel >= 0 && io.out_numel != (int64_t)want_out16) ||
      blob->xq16 == nullptr) {
    ET_LOG(Error, "esp32: i16 shape/scratch mismatch (in %d/%u out %d/%u)",
           (int)io.in_numel, (unsigned)want_in16, (int)io.out_numel,
           (unsigned)want_out16);
    return Error::InvalidArgument;
  }
  if (op.scale_group != 0) {
    // Grouped scales exist on the i4/i8 path for pass-through weights;
    // nothing emits grouped i16 and the loop below applies one scale per
    // row. Refuse rather than silently misattribute.
    ET_LOG(Error, "esp32: grouped i16 addmm not implemented");
    return Error::NotSupported;
  }
  if (cg16 != nullptr && is_depthwise(*cg16)) {
    run_addmm_i16_depthwise(c, i, io, *cg16);
    return Error::Ok;
  }
  const int16_t* wq16 = reinterpret_cast<const int16_t*>(blob->resolved[i].wq);
  const float* wscale = blob->resolved[i].scale;
  // Channel-slice offsets for a grouped conv, exactly as the i8 path does
  // them: c_in/c_out in the record are already per-group.
  const float* xg16 = io.x;
  float* yg16 = io.y;
  if (cg16 != nullptr && cg16->group_idx > 0) {
    xg16 += (size_t)cg16->group_idx * cg16->c_in * cg16->d_in * cg16->h_in *
            cg16->w_in;
    yg16 += (size_t)cg16->group_idx * cg16->c_out * M;
  }
  for (uint32_t m = 0; m < M; ++m) {
    float sx;
    if (cg16 != nullptr) {
      // Always DYNAMIC here, never the calibrated static scale. A static
      // activation scale is calibrated against int8's range; reusing it at
      // i16 would quantise to a 128x coarser grid than the format allows
      // and make the accuracy rung less accurate than the one below it.
      sx = quantize_conv_row_t<int16_t, kQmaxI16>(xg16, *cg16, m, op.k_pad,
                                               blob->xq16, -1.0f, false);
    } else {
      // Dynamic per-row absmax for the same reason as the conv branch.
      sx = quantize_row_t<int16_t, kQmaxI16>(io.x + (size_t)m * K, K,
                                             op.k_pad, blob->xq16, -1.0f,
                                             false);
    }
    float* yrow = io.y + (size_t)m * N;
    for (uint32_t n = 0; n < N; ++n) {
      const int16_t* wrow = wq16 + (size_t)n * op.k_pad;
      // int64, not int32: 32767^2 overflows int32 after two terms, so an
      // int32 accumulator here would be wrong for every K > 1 rather than
      // for a large one. The host reference accumulates in int64 to match.
      int64_t acc = 0;
      for (uint32_t t = 0; t < op.k_pad; ++t) {
        acc += (int32_t)blob->xq16[t] * (int32_t)wrow[t];
      }
      float y16 = (io.bias != nullptr ? io.bias[n] : 0.0f)
          + (float)acc * sx * wscale[n];
      // Same index story as the i8 scatter: [m, n] for a matmul, NCHW
      // for a conv. Getting this wrong is silent -- it adds real numbers
      // from the right tensor at the wrong offsets.
      const size_t at =
          (cg16 != nullptr) ? (size_t)n * M + m : (size_t)m * N + n;
      y16 = epilogue(op, y16, io.residual, at);
      // [m, n] -> NCHW: output element (n, m) lands at n*M + m. The i8 path
      // stages a whole block and scatters it; one row at a time needs no
      // staging buffer at all.
      if (cg16 != nullptr) {
        yg16[(size_t)n * M + m] = y16;
      } else {
        yrow[n] = y16;
      }
    }
  }
  count_cols(0, M * N);   // every output element is one scalar column
  count_op((uint64_t)M * K * N, (uint64_t)N * row_bytes(op.opcode, op.k_pad));
  return Error::Ok;
}

/// The legal-token head: the full vocabulary stays packed once, and execute
/// streams only the rows named by the int64 IDs in a second partition input.
static Error run_selected_rows(const CallContext& c, uint32_t i,
                               const MatmulIo& io) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const SelectedRows& selection = *blob->selected[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  if (selection.rows_arg >= c.args.size()) {
    ET_LOG(Error, "esp32: selected-row op %u rows_arg out of range",
           (unsigned)i);
    return Error::InvalidArgument;
  }
  const auto rows_t = c.args[selection.rows_arg]->toTensor();
  if (io.in_numel != (int64_t)((uint64_t)M * K) ||
      io.out_numel != (int64_t)((uint64_t)M * selection.count) ||
      rows_t.numel() != (int64_t)selection.count) {
    ET_LOG(Error,
           "esp32: selected-row op %u shape mismatch "
           "(in=%d out=%d rows=%d, expected %ux%u -> %ux%u)",
           (unsigned)i, (int)io.in_numel, (int)io.out_numel,
           (int)rows_t.numel(), (unsigned)M, (unsigned)K, (unsigned)M,
           (unsigned)selection.count);
    return Error::InvalidArgument;
  }
  if (rows_t.scalar_type() != executorch::aten::ScalarType::Long) {
    ET_LOG(Error, "esp32: selected-row IDs must be int64");
    return Error::InvalidArgument;
  }
  const int64_t* rows = rows_t.const_data_ptr<int64_t>();
  for (uint32_t j = 0; j < selection.count; ++j) {
    if (rows[j] < 0 || rows[j] >= (int64_t)N) {
      ET_LOG(Error, "esp32: selected-row ID %d at %u is outside [0,%u)",
             (int)rows[j], (unsigned)j, (unsigned)N);
      return Error::InvalidArgument;
    }
  }
  const bool is_i4 = executes_i4(op);
  if (is_i4 && blob->wu == nullptr) {
    ET_LOG(Error, "esp32: selected i4 op with no unpack scratch");
    return Error::InvalidArgument;
  }
  const int8_t* kernel_row_base = is_i4 ? blob->wu : blob->resolved[i].wq;
  const uint32_t gsize = op.scale_group ? op.scale_group : op.k_pad;
  const bool aligned = pie_row_aligned(c.force_scalar, kernel_row_base,
                                       blob->xq, op.k_pad, gsize);
  const float fix = fixed_activation_scale(op, io.x, (size_t)M * K, 1);
  const bool pt2e = (op.flags & kFlagActPerTensor) != 0;
  for (uint32_t m = 0; m < M; ++m) {
    const float sx = quantize_row_i8(io.x + (size_t)m * K, K, op.k_pad,
                                     blob->xq, fix, pt2e);
    addmm_q_selected_rows(op, blob->resolved[i], blob->xq, sx, blob->wu,
                          io.bias, rows, selection.count,
                          io.y + (size_t)m * selection.count, aligned);
  }
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;
  }
  const uint64_t rb = execution_row_bytes(op);
  count_passes(M, M, (uint64_t)selection.count * rb);
  count_op((uint64_t)M * K * selection.count, (uint64_t)selection.count * rb);
  return Error::Ok;
}

/// Whether this op's output columns go half to the second core (min-latency
/// mode). Gated on width so narrow ops do not pay a handshake for nothing,
/// and on wu2 so a split i4 op can never share the caller's unpack buffer.
/// Conv stays single-core: the worker would have to write into the staging
/// buffer and be scattered too, new interaction with the most carefully
/// verified path here.
ESP32_HOT_CODE
static bool split_columns(const CallContext& c, const OpDesc& op,
                          bool is_conv) {
  const bool requested =
      c.global_split || (c.selective_split && (op.flags & kFlagHeadSplit));
  return requested && c.split_enabled &&
         g_worker.load(std::memory_order_relaxed) != nullptr && !is_conv &&
         op.n >= kSplitMinCols &&
         (!executes_i4(op) || c.blob->wu2 != nullptr);
}

/// Group leader: run every member of an activation-sharing group. Each
/// block of input rows is quantised ONCE (same helpers as the standalone
/// path, so the arithmetic cannot fork), then each member's column loop
/// reads it. Values and per-tensor write order are exactly the standalone
/// ones -- outputs are disjoint boundary tensors -- so only the (count-1)
/// redundant absmax+quantise walks per row disappear. Grouped ops block
/// like everything else; row at a time, each member would make M passes over
/// its weights instead of ceil(M/B) -- for a 32-token prefill block with
/// q/k, k/o and gate/up grouped on every layer, about 6 MB of extra flash
/// reads. A decode step (M == 1)
/// keeps the specialised single-row path and its counters unless the member
/// splits across the cores, which takes the standalone path's split form.
ESP32_HOT_CODE
static Error run_activation_group(const CallContext& c, uint32_t i,
                                  const MatmulIo& io, int16_t grp) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k;
  const uint16_t gfirst = blob->group_first[grp];
  const uint16_t gcount = blob->group_count[grp];
  const OpDesc* mop[kMaxActGroup];
  const Resolved* mres[kMaxActGroup];
  const float* mbias[kMaxActGroup];
  float* my[kMaxActGroup];
  bool malign[kMaxActGroup];
  const float* mr[kMaxActGroup];
  for (uint16_t t = 0; t < gcount; ++t) {
    const uint16_t oi = blob->group_members[gfirst + t];
    const OpDesc& oq = blob->ops[oi];
    const bool out_scratch = oq.out_arg >= kArgScratch0;
    if (out_scratch ? blob->chain[0] == nullptr
                    : oq.out_arg >= c.args.size()) {
      ET_LOG(Error, "esp32: group op %u out_arg out of range", (unsigned)oi);
      return Error::InvalidArgument;
    }
    // Same in_arg means the same tensor, so m/k must agree; a mismatch is
    // a corrupt blob, not a shape to accommodate.
    if (oq.m != M || oq.k != K) {
      ET_LOG(Error, "esp32: group op %u shape disagrees with leader",
             (unsigned)oi);
      return Error::InvalidProgram;
    }
    if (out_scratch) {
      // A chain slot: sized at init to its widest writer, like any other.
      my[t] = blob->chain[oq.out_arg - kArgScratch0];
    } else {
      auto mout = c.args[oq.out_arg]->toTensor();
      if (mout.numel() != (int64_t)((uint64_t)M * oq.n)) {
        ET_LOG(Error, "esp32: group op %u out numel mismatch", (unsigned)oi);
        return Error::InvalidArgument;
      }
      if (mout.scalar_type() != executorch::aten::ScalarType::Float) {
        ET_LOG(Error, "esp32: group op %u output must be float32",
               (unsigned)oi);
        return Error::InvalidArgument;
      }
      my[t] = mout.mutable_data_ptr<float>();
    }
    if (executes_i4(oq) && blob->wu == nullptr) {
      ET_LOG(Error, "esp32: i4 op with no unpack scratch");
      return Error::InvalidArgument;
    }
    mop[t] = &oq;
    mres[t] = &blob->resolved[oi];
    mbias[t] = oq.bias_off != 0
        ? reinterpret_cast<const float*>(blob->base + oq.bias_off)
        : nullptr;
    mr[t] = nullptr;
    if ((oq.flags & kFlagMul) && oq.res_arg >= kArgScratch0) {
      // In place on the member's own output slot (validated at init).
      mr[t] = blob->chain[oq.res_arg - kArgScratch0];
    } else if (oq.flags & (kFlagAdd | kFlagMul)) {
      if (oq.res_arg >= c.args.size()) {
        ET_LOG(Error, "esp32: group op %u res_arg out of range",
               (unsigned)oi);
        return Error::InvalidArgument;
      }
      const auto rt = c.args[oq.res_arg]->toTensor();
      if (rt.numel() != (int64_t)((uint64_t)M * oq.n)) {
        ET_LOG(Error, "esp32: group op %u residual numel mismatch",
               (unsigned)oi);
        return Error::InvalidArgument;
      }
      if (rt.scalar_type() != executorch::aten::ScalarType::Float) {
        ET_LOG(Error, "esp32: group op %u residual must be float32",
               (unsigned)oi);
        return Error::InvalidArgument;
      }
      mr[t] = rt.const_data_ptr<float>();
    }
    // Same alignment gate as the standalone path, per member: for i4 the
    // kernel reads the hand-aligned unpack scratch, not the packed source.
    const int8_t* base_t = executes_i4(oq) ? blob->wu : mres[t]->wq;
    const uint32_t gs_t = oq.scale_group ? oq.scale_group : oq.k_pad;
    malign[t] = pie_row_aligned(c.force_scalar, base_t, blob->xq, oq.k_pad,
                                gs_t);
  }
  // Members share their complete activation-scale scheme by the grouping
  // key, so the leader's choice is every member's choice. A PT2E per-tensor
  // group still quantises each row separately, but every row uses this one
  // absmax over the complete M*K activation.
  const float gfix = fixed_activation_scale(op, io.x, (size_t)M * K, gcount);
  const bool gpt2e = (op.flags & kFlagActPerTensor) != 0;
  const uint32_t gblk = row_block_for(M, op.k_pad, nullptr, 0);
  // Each member splits its columns across the cores exactly as a standalone
  // op would: the shared quantisation saves the re-quantising, and must not
  // cost the second core: single-core, grouped gate+up and q+k would run
  // at half the bandwidth of the ungrouped projections beside them.
  bool msplit[kMaxActGroup];
  for (uint16_t t = 0; t < gcount; ++t) {
    msplit[t] = split_columns(c, *mop[t], false);
  }
  for (uint32_t m0 = 0; m0 < M; m0 += gblk) {
    const uint32_t nrows = (m0 + gblk <= M) ? gblk : (M - m0);
    // One row quantises into xq with its scale on this frame; a block uses
    // sxv, which exists because nrows > 1 implies block_rows > 1 at init.
    float sx1 = 0.0f;
    const float* sxv = &sx1;
    if (nrows == 1) {
      sx1 = quantize_row_i8(io.x + (size_t)m0 * K, K, op.k_pad, blob->xq, gfix,
                            gpt2e);
    } else {
      for (uint32_t t = 0; t < nrows; ++t) {
        blob->sxv[t] = quantize_row_i8(io.x + (size_t)(m0 + t) * K, K,
                                       op.k_pad,
                                       blob->xq + (size_t)t * op.k_pad, gfix,
                                       gpt2e);
      }
      sxv = blob->sxv;
    }
    for (uint16_t t = 0; t < gcount; ++t) {
      const OpDesc& oq = *mop[t];
      float* ybase = my[t] + (size_t)m0 * oq.n;
      const float* rbase = mr[t] ? mr[t] + (size_t)m0 * oq.n : nullptr;
      if (nrows == 1 && !msplit[t]) {
        addmm_q_row(oq, *mres[t], blob->xq, sx1, blob->wu, mbias[t], ybase,
                    malign[t], rbase);
        continue;
      }
      uint32_t gpie = 0, gscalar = 0;
      uint32_t n_lo = 0;
      if (msplit[t]) {
        n_lo = oq.n / 2;
        fork_split(c, blob->group_members[gfirst + t], blob->xq, sxv, nrows,
                   mbias[t], ybase, oq.n, malign[t], rbase, n_lo);
      }
      if (is_plain_i8(oq, malign[t]) && no_epilogue(oq, rbase)) {
        addmm_i8_plain_block_range(oq, *mres[t], blob->xq, sxv, nrows,
                                   ybase, oq.n, mbias[t], n_lo, oq.n, &gpie);
      } else {
        addmm_q_row_range(oq, *mres[t], blob->xq, sxv, nrows, op.k_pad,
                          blob->wu, mbias[t], ybase, oq.n, malign[t], rbase,
                          n_lo, oq.n, &gpie, &gscalar);
      }
      if (msplit[t]) {
        join_split(oq, nrows, &gpie, &gscalar);
      }
      count_cols(gpie, gscalar);
    }
  }
  for (uint16_t t = 0; t < gcount; ++t) {
    count_op((uint64_t)M * K * mop[t]->n,
             (uint64_t)mop[t]->n * execution_row_bytes(*mop[t]));
  }
  g_actq_rows_saved += (uint64_t)(gcount - 1) * M;
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;   // one quantisation per row for the group
  }
  // One pass over each member's weights per BLOCK, not per row.
  const uint64_t gpasses = (M + gblk - 1) / gblk;
  for (uint32_t gm = 0; gm < gcount; ++gm) {
    const OpDesc& go = blob->ops[blob->group_members[gfirst + gm]];
    const uint64_t grb = (uint64_t)go.n * execution_row_bytes(go);
    count_passes(gpasses, M, grb);
  }
  return Error::Ok;
}

/// A conv's geometry as esp32_asym's walks read it.
static esp32_asym::ConvGather conv_gather(const ConvGeom& g) {
  return esp32_asym::ConvGather{g.h_in, g.w_in, g.c_in, g.h_out, g.w_out,
                                g.kh, g.kw, g.stride_h, g.stride_w, g.pad_h,
                                g.pad_w, g.dil_h, g.dil_w, g.d_in, g.d_out,
                                g.kd, g.stride_d, g.pad_d, g.dil_d,
                                g.transposed};
}

/// quantize_conv_row's codes for one channel of a depthwise conv whose taps
/// are already found (esp32_asym::conv_tap_offsets: offs[t] is where tap t
/// reads in the channel's plane, -1 off the input): the same absmax over the
/// same taps in the same order, the same scale, rounding and clamp, a 0 code
/// for a tap off the input and a zero tail to k_pad. Bit for bit what
/// quantize_conv_row writes for the one-channel geometry, without its index
/// arithmetic per tap, channel and position.
static float quantize_taps_i8(const float* x, const int32_t* offs, uint32_t K,
                              uint32_t k_pad, int8_t* dst, float fixed_sx,
                              bool ties_to_even) {
  float sx;
  if (fixed_sx >= 0.0f) {
    sx = fixed_sx;   // counted by the caller, as for quantize_conv_row
  } else {
    float amax = 0.0f;
    for (uint32_t t = 0; t < K; ++t) {
      if (offs[t] < 0) {
        continue;
      }
      const float a = std::fabs(x[offs[t]]);
      if (a > amax) {
        amax = a;
      }
    }
    sx = amax / (float)kQmaxI8;
  }
  const float inv_sx = a0_inverse_scale(sx);
  for (uint32_t t = 0; t < K; ++t) {
    if (offs[t] < 0) {
      dst[t] = 0;
      continue;
    }
    const float v = x[offs[t]] * inv_sx;
    int32_t q = ties_to_even ? round_half_even_i(v) : round_half_away_i(v);
    q = q > kQmaxI8 ? kQmaxI8 : (q < -kQmaxI8 ? -kQmaxI8 : q);
    dst[t] = (int8_t)q;
  }
  for (uint32_t z = K; z < k_pad; ++z) {
    dst[z] = 0;
  }
  return sx;
}

/// DEPTHWISE takes its own loop, and the reason is arithmetic rather than
/// convenience: every output channel reads its OWN input channel, so the
/// activation being quantised varies with the output COLUMN as well as the
/// row. addmm_q_row_range assumes one activation row serves every column --
/// true for a matmul and for a dense conv, false here. It is also the case
/// where im2col is the wrong transform, not merely a slower one: on a 3x3
/// 128->128 layer the weights are 1.1 KB against 205 KB of activation, and
/// an im2col buffer would cost 373 KB to save re-streaming those 1.1 KB.
///
/// Every channel's patch sits at the same offsets in its own plane, so the
/// taps are walked once per output position (esp32_asym::conv_tap_offsets,
/// the fp32 depthwise conv's walk, into the xq scratch after the codes:
/// depthwise_taps) and each channel quantises its own in place
/// (quantize_taps_i8), then the shared dot and epilogue. A per-channel
/// gather would cost several times the dot (181 instructions a MAC
/// on a 3x3). An i4 row is unpacked per position, a fraction of that.
///
/// The predicate is the full ConvGeom.depthwise one, spelled the same way
/// AOT-side on purpose: `groups > 1` alone would send a split grouped slice
/// (which KEEPS `groups` for the channel offsets) down this path.
///
/// noinline, for speed: inlined into execute(), the loop loses registers to
/// execute()'s other paths (the int8 depthwise 3x3: 73.9 -> 76.2
/// instructions a MAC).
[[gnu::noinline]] static void run_depthwise_i8(const CallContext& c,
                                               uint32_t i, const MatmulIo& io,
                                               bool wq_aligned, float fix) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const ConvGeom& g = *blob->conv[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const bool is_i4 = executes_i4(op);
  const size_t plane = (size_t)g.d_in * g.h_in * g.w_in;
  const size_t stride = row_bytes(op.opcode, op.k_pad);
  const uint32_t gsize = op.scale_group ? op.scale_group : op.k_pad;
  const uint32_t ngw = (op.k_pad + gsize - 1) / gsize;
  const bool pt2e = (op.flags & kFlagActPerTensor) != 0;
  // One channel is a 1-channel conv, so the SHARED walk serves it:
  // depthwise needs a new kernel but no new geometry, no new packing and no
  // new format handling.
  ConvGeom one = g;
  one.c_in = 1;
  one.c_out = 1;
  one.groups = 1;
  const esp32_asym::ConvGather taps = conv_gather(one);
  int32_t* offs = depthwise_taps(blob->xq, op.k_pad);
  uint32_t pie = 0, scalar = 0;
  for (uint32_t r = 0; r < M; ++r) {
    esp32_asym::conv_tap_offsets(taps, r, offs);
    for (uint32_t ch = 0; ch < N; ++ch) {
      const int8_t* wrow = weight_row(blob->resolved[i].wq, stride, ch, is_i4,
                                      op.k_pad, blob->wu);
      // The same PIE gate the matmul path applies, on this channel's row.
      // Integer accumulation, so PIE and the scalar loop are bit-identical.
      const bool dw_pie =
          wq_aligned && (reinterpret_cast<uintptr_t>(wrow) % kVecBytes) == 0;
      const float sx = quantize_taps_i8(io.x + (size_t)ch * plane, offs, K,
                                        op.k_pad, blob->xq, fix, pt2e);
      const float v = accumulate_groups(
          op, blob->resolved[i].scale, ch, gsize, ngw, blob->xq, wrow, dw_pie,
          nullptr, nullptr, sx, io.bias ? io.bias[ch] : 0.0f);
      if (dw_pie) {
        ++pie;
      } else {
        ++scalar;
      }
      // The shared epilogue is what keeps a fused silu on a depthwise op
      // meaning the same thing it means on a matmul. The residual is NCHW,
      // and always an add: init refuses kFlagMul on a conv.
      io.y[(size_t)ch * M + r] =
          epilogue(op, v, io.residual, (size_t)ch * M + r);
    }
  }
  count_cols(pie, scalar);
  if (op.flags & kFlagStaticAct) {
    g_static_rows += (uint64_t)N * M;   // one quantisation per channel per row
  }
  count_op((uint64_t)M * K * N, (uint64_t)N * row_bytes(op.opcode, op.k_pad));
}

/// ROW BLOCKING, and every convolution. Dispatching one output row at a
/// time re-reads the whole weight matrix for every row: M passes over
/// N*row_bytes. Blocking `blk` rows quantises them into consecutive xq
/// slots first, then makes ONE pass over the columns serving all of them --
/// so the weight traffic falls by blk and the i4 unpack is amortised over
/// blk dots. The block is the SAME per-op decision init sized the scratch
/// from, so it cannot exceed what was reserved. It composes with the split:
/// the worker takes the low column half of every block. A conv with a 1x1
/// output legitimately gets blk == 1 and still takes this loop, which is the
/// only path that gathers.
static void run_blocked_i8(const CallContext& c, uint32_t i,
                           const MatmulIo& io, bool wq_aligned, float fix,
                           bool split, uint32_t blk) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const ConvGeom* cg = blob->conv[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  // A split grouped conv: this op owns one channel slice of both the input
  // and the output. c_in/c_out in the record are already the PER-GROUP
  // counts, so the offsets are just the slice index times them -- no
  // division here, and no second interpretation of the same fields.
  const float* xg = io.x;
  float* yg = io.y;
  if (cg != nullptr && cg->group_idx > 0) {
    xg += (size_t)cg->group_idx * cg->c_in * cg->d_in * cg->h_in * cg->w_in;
    yg += (size_t)cg->group_idx * cg->c_out * M;
  }
  const bool plain_i8_block = cg == nullptr && is_plain_i8(op, wq_aligned) &&
                              no_epilogue(op, io.residual);
  const bool pt2e = (op.flags & kFlagActPerTensor) != 0;
  for (uint32_t m0 = 0; m0 < M; m0 += blk) {
    const uint32_t nrows = (m0 + blk <= M) ? blk : (M - m0);
    for (uint32_t t = 0; t < nrows; ++t) {
      // The one line conv changes. Everything below is the matmul path.
      blob->sxv[t] =
          cg ? quantize_conv_row(xg, *cg, m0 + t, op.k_pad,
                                 blob->xq + (size_t)t * op.k_pad, fix, pt2e)
             : quantize_row_i8(io.x + (size_t)(m0 + t) * K, K, op.k_pad,
                               blob->xq + (size_t)t * op.k_pad, fix, pt2e);
    }
    // A conv block lands in the staging buffer and is scattered after; a
    // matmul block writes straight through. The kernel's column stride is
    // fixed at 1 and NCHW needs h_out*w_out, so the transpose has to happen
    // somewhere -- here, rather than inside the hot loop and the split.
    float* dst = cg ? blob->conv_stage : yg + (size_t)m0 * N;
    // The residual is withheld from the kernel for a conv and applied in the
    // scatter below instead: the kernel indexes a residual as [m, n] because
    // a matmul's output is m-major; a conv's is NCHW, so the same pointer
    // means a different element. (The convres_delegated parity row checks
    // this case.)
    const float* rblock =
        (io.residual && !cg) ? io.residual + (size_t)m0 * N : nullptr;
    uint32_t bpie = 0, bscalar = 0;
    uint32_t n_lo = 0;
    if (split) {
      const uint32_t half = N / 2;
      fork_split(c, i, blob->xq, blob->sxv, nrows, io.bias, dst, N,
                 wq_aligned, rblock, half);
      n_lo = half;
    }
    if (plain_i8_block) {
      addmm_i8_plain_block_range(op, blob->resolved[i], blob->xq, blob->sxv,
                                 nrows, dst, N, io.bias, n_lo, N, &bpie);
    } else {
      addmm_q_row_range(op, blob->resolved[i], blob->xq, blob->sxv, nrows,
                        op.k_pad, blob->wu, io.bias, dst, N, wq_aligned,
                        rblock, n_lo, N, &bpie, &bscalar);
    }
    if (split) {
      join_split(op, nrows, &bpie, &bscalar);
    }
    if (cg) {
      // [nrows, N] -> NCHW: element (n, row) belongs at n*m + row. The
      // residual lives in the OUTPUT's layout, so it is indexed exactly as
      // the destination is -- the scatter is the one place where the two
      // agree without a second index calculation. Applied last, after the
      // activation and clamp the kernel already did.
      for (uint32_t t = 0; t < nrows; ++t) {
        const float* src = blob->conv_stage + (size_t)t * N;
        float* base = yg + (size_t)(m0 + t);
        const size_t row = (size_t)(m0 + t);
        for (uint32_t n = 0; n < N; ++n) {
          const size_t off = (size_t)n * M;
          base[off] = io.residual ? src[n] + io.residual[off + row] : src[n];
        }
      }
    }
    count_cols(bpie, bscalar);
  }
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;   // every row quantised once, at act_scale
  }
  // One pass over the weights per BLOCK, not per row.
  const uint64_t rb = (uint64_t)op.n * execution_row_bytes(op);
  const uint64_t passes = (M + blk - 1) / blk;
  count_passes(passes, M, rb);
  count_op((uint64_t)M * K * N, rb);
}

/// Row at a time: the decode case (M == 1) and any op whose budget gives a
/// block of one. The min-latency fork hands the worker the LOW half of the
/// columns and this core takes the high half.
ESP32_HOT_CODE
static void run_rows_i8(const CallContext& c, uint32_t i, const MatmulIo& io,
                        bool wq_aligned, float fix, bool split) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  count_passes(M, M, (uint64_t)op.n * execution_row_bytes(op));
  const bool pt2e = (op.flags & kFlagActPerTensor) != 0;
  for (uint32_t m = 0; m < M; ++m) {
    const float sx = quantize_row_i8(io.x + (size_t)m * K, K, op.k_pad,
                                     blob->xq, fix, pt2e);
    const float* rrow = io.residual ? io.residual + (size_t)m * N : nullptr;
    float* yrow = io.y + (size_t)m * N;
    if (split) {
      const uint32_t half = N / 2;
      // sx lives on this frame until join_split() returns.
      fork_split(c, i, blob->xq, &sx, 1, io.bias, yrow, N, wq_aligned, rrow,
                 half);
      uint32_t cpie = 0, cscalar = 0;
      addmm_q_row_range(op, blob->resolved[i], blob->xq, &sx, 1, op.k_pad,
                        blob->wu, io.bias, yrow, N, wq_aligned, rrow, half, N,
                        &cpie, &cscalar);
      join_split(op, 1, &cpie, &cscalar);
      count_cols(cpie, cscalar);
    } else {
      addmm_q_row(op, blob->resolved[i], blob->xq, sx, blob->wu, io.bias,
                  yrow, wq_aligned, rrow);
    }
  }
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;
  }
  count_op((uint64_t)M * K * N, (uint64_t)N * execution_row_bytes(op));
}

// ---- the affine ops ---------------------------------------------------------
// Weight records (zero points, float offsets) and activation records.
// An op with EXT_WEIGHT_AFFINE or EXT_ACT_QUANT takes these loops instead of
// the symmetric ones, chosen once per op (run_addmm_quantised, execute). Each
// mirrors its symmetric twin line for line -- row blocking, conv gather and
// scatter, epilogue, dual-core split and counters -- and hands the extra
// arithmetic to esp32_asym.cpp: the model's own activation quantiser (its
// chooser, rounding and zero point, the conv padding written as zx), the
// zero-point corrections and the float offsets are extra roundings, and that
// file builds -ffp-contract=off, so the device, the host runner and
// mirrors.py agree bit for bit. An op without an activation record
// quantises with the backend's own quantisers (zx = 0).

/// How an affine op quantises its activation: its own record (per tensor:
/// the scale and zero point chosen once over the whole input; static: the
/// record's own), or the backend's quantisers at fixed_activation_scale's
/// `fix`.
struct AffineAct {
  const esp32_asym::ActQuant* rec;
  esp32_asym::ActParams tensor;
  esp32_asym::ConvGather gather;   // a conv op's geometry, as the TU reads it
  float fix;
  bool pt2e;
};

/// Op i's AffineAct for this call: `x` is its whole input (n_act elements),
/// `cg` the geometry its rows gather with (a depthwise op's one-channel one).
static AffineAct affine_act(const CompiledBlob* blob, uint32_t i,
                            const float* x, size_t n_act, const ConvGeom* cg) {
  const OpDesc& op = blob->ops[i];
  AffineAct a{esp32_asym::act_record(blob->asym, i), {0.0f, 0}, {}, -1.0f,
              false};
  if (cg != nullptr) {
    a.gather = conv_gather(*cg);
  }
  if (a.rec == nullptr) {
    a.fix = fixed_activation_scale(op, x, n_act, 1);
    a.pt2e = (op.flags & kFlagActPerTensor) != 0;
  } else if (a.rec->granularity != esp32_asym::kGranularityRow) {
    a.tensor = esp32_asym::choose(*a.rec, x, n_act);
  }
  return a;
}

/// An affine op's input at some offset: fp32 values, or Stage B's int8 codes
/// (in_dtype int8, Stage B). Exactly one pointer is set.
struct ActIn {
  const float* x;
  const int8_t* c;
  ActIn at(size_t off) const {
    return ActIn{x != nullptr ? x + off : nullptr,
                 c != nullptr ? c + off : nullptr};
  }
};

/// One matmul row of an affine op quantised into dst (k_pad wide) with its
/// per-group terms -- or, for codes in, its codes copied there -- returns
/// the row's scale and writes its zero point. Without a record, the
/// backend's own quantiser (A0), as quantize_row_i8 computes it.
static float affine_row(const AffineAct& a, const esp32_asym::AffineWeights& w,
                        ActIn in, int8_t* dst, int32_t* xsum, float* toff,
                        int32_t* zx) {
  if (in.c != nullptr) {
    // The model's codes, made with the record's constants (a.tensor).
    esp32_asym::codes_row(*a.rec, in.c, w.k, w, dst, xsum, toff);
    *zx = a.tensor.zx;
    return a.tensor.sx;
  }
  const float* x = in.x;
  if (a.rec == nullptr) {
    *zx = 0;
    return esp32_asym::quantize_row_a0(x, w.k, a.fix, a.pt2e, w, dst, xsum,
                                       toff);
  }
  const esp32_asym::ActParams p =
      a.rec->granularity != esp32_asym::kGranularityRow
          ? a.tensor
          : esp32_asym::choose(*a.rec, x, w.k);
  esp32_asym::quantize_row(*a.rec, p, x, w.k, w, dst, xsum, toff);
  *zx = p.zx;
  return p.sx;
}

/// One conv output position's patch: affine_row's twin over the act's
/// gather (the op's geometry, or a depthwise op's one-channel one with `in`
/// at its channel). A conv's record is per tensor or static (attach refuses
/// per row).
static float affine_conv_row(const AffineAct& a,
                             const esp32_asym::AffineWeights& w, ActIn in,
                             uint32_t row, int8_t* dst, int32_t* xsum,
                             float* toff, int32_t* zx) {
  if (in.c != nullptr) {
    esp32_asym::codes_conv_row(*a.rec, in.c, a.gather, row, w, dst, xsum,
                               toff);
    *zx = a.tensor.zx;
    return a.tensor.sx;
  }
  const float* x = in.x;
  if (a.rec == nullptr) {
    *zx = 0;
    return esp32_asym::quantize_conv_row_a0(x, a.gather, row, a.fix, a.pt2e,
                                            w, dst, xsum, toff);
  }
  esp32_asym::quantize_conv_row(*a.rec, a.tensor, x, a.gather, row, w, dst,
                                xsum, toff);
  *zx = a.tensor.zx;
  return a.tensor.sx;
}

/// One block of quantised rows against an affine op's columns: what
/// addmm_q_row_range takes, plus the op's AffineWeights and the rows' terms.
struct AffineBlock {
  const OpDesc* op;
  const Resolved* res;
  const esp32_asym::AffineWeights* w;
  const int8_t* xq;
  const float* sxv;
  const int32_t* zxv;      // [nrows]: each row's activation zero point
  const int32_t* xsum;     // [nrows, ngw]: X_g per row
  const float* toff;       // [nrows, ngw]: T_g per row, or null (no offsets)
  uint32_t nrows;
  uint32_t xq_stride;
  const float* bias;
  float* ybase;
  uint32_t y_stride;
  bool aligned;
  const float* rbase;
  const int64_t* select;   // selected-row head: output j is table row select[j]
  const esp32_asym::OpQuant* oq;  // the op's output record, or null
  // Stage B: the op's int8 codes go here (indexed as ybase), not to ybase.
  int8_t* ybase8;
};

/// The epilogue for the affine loops, out of line: one copy of apply_act for
/// all of them (each call sits beside a whole dot product), and the same
/// arithmetic as the inline copies the symmetric loops use.
[[gnu::noinline]] static float affine_epilogue(const OpDesc& op, float y,
                                               const float* r, size_t at) {
  return epilogue(op, y, r, at);
}

/// Output columns [j0, j1) of one block on this core, `wu` this core's unpack
/// scratch (also the fused int4 kernel's deinterleaved activations).
static void affine_columns(const AffineBlock& b, int8_t* wu, uint32_t j0,
                           uint32_t j1, uint32_t* pie, uint32_t* scalar) {
  const OpDesc& op = *b.op;
  const esp32_asym::AffineWeights& w = *b.w;
  const bool is_i4 = executes_i4(op);
  const int8_t* wq = b.res->wq;
  const size_t stride = (size_t)execution_row_bytes(op);
  const bool fused = is_i4 && b.nrows == 1 && b.select == nullptr &&
                     b.aligned && fused_i4_ok(op, w.gsize, wq, stride, wu);
  int32_t raw[kMaxFusedI4Groups];
  if (fused) {
    // The deinterleave the symmetric decode path uses; its 8 * sum(x) is
    // not needed here (the row's quantiser summed every group).
    int32_t unused[kMaxFusedI4Groups];
    prepare_i4_activation(b.xq, op.k_pad, w.gsize, wu, unused);
  }
  for (uint32_t j = j0; j < j1; ++j) {
    const uint32_t n = b.select != nullptr ? (uint32_t)b.select[j] : j;
    const int8_t* wrow = nullptr;
    if (fused) {
      dot_i4_groups_s3(reinterpret_cast<const uint8_t*>(wq) + (size_t)n * stride,
                       wu, raw, (int)w.ngw, (int)(w.gsize / 2));
    } else {
      wrow = weight_row(wq, stride, n, is_i4, op.k_pad, wu);
    }
    const float initial = b.bias != nullptr ? b.bias[n] : 0.0f;
    for (uint32_t mb = 0; mb < b.nrows; ++mb) {
      const int32_t* xs = b.xsum + (size_t)mb * w.ngw;
      const float* tf = b.toff != nullptr ? b.toff + (size_t)mb * w.ngw : nullptr;
      const float yn = fused
          ? esp32_asym::accumulate_fused(w, n, raw, b.sxv[mb], b.zxv[mb], xs,
                                         tf, initial)
          : esp32_asym::accumulate(w, n, b.xq + (size_t)mb * b.xq_stride, wrow,
                                   b.aligned, b.sxv[mb], b.zxv[mb], xs, tf,
                                   initial);
      if (b.aligned) {
        ++*pie;
      } else {
        ++*scalar;
      }
      const size_t at = (size_t)mb * b.y_stride + j;
      const float y = affine_epilogue(op, yn, b.rbase, at);
      if (b.ybase8 != nullptr) {
        b.ybase8[at] = esp32_asym::requantize_code(*b.oq, y);
      } else {
        b.ybase[at] = b.oq != nullptr ? esp32_asym::requantize(*b.oq, y) : y;
      }
    }
  }
}

/// The affine matmuls' per-row terms: X_g (int32) for every row of the
/// largest block an affine op runs, then T_g (float) when it has float
/// offsets, then each row's activation zero point. Depthwise and selected-row
/// heads run one row at a time.
static void reserve_affine_scratch(CompiledBlob* blob, ScratchLayout* layout) {
  if (blob->asym == nullptr) {
    return;
  }
  uint64_t need = 0;
  for (uint32_t i = 0; i < blob->n_ops; ++i) {
    const OpDesc& op = blob->ops[i];
    if (is_embed(op.opcode) || !esp32_asym::affine_op(blob->asym, i)) {
      continue;
    }
    const ConvGeom* cg = blob->conv[i];
    const bool one_row = blob->selected[i] != nullptr ||
                         (cg != nullptr && is_depthwise(*cg));
    const uint64_t rows = one_row ? 1 : row_block_for(op.m, op.k_pad, cg, op.n);
    const uint32_t gsize = op.scale_group ? op.scale_group : op.k_pad;
    const uint64_t ngw = ((uint64_t)op.k_pad + gsize - 1) / gsize;
    const esp32_asym::WeightAffine* wa = blob->asym->ops[i].weight;
    const bool offsets =
        wa != nullptr && wa->offset_kind != esp32_asym::kOffsetNone;
    uint64_t bytes =
        rows * ngw * (sizeof(int32_t) + (offsets ? sizeof(float) : 0)) +
        rows * sizeof(int32_t);
    if (esp32_asym::a16_op(blob->asym, i)) {
      // The int16 loop's one row of codes (no per-group terms).
      bytes = (uint64_t)op.k_pad * sizeof(int16_t);
    }
    if (bytes > need) {
      need = bytes;
    }
  }
  if (need > 0) {
    const uint32_t at = layout->reserve(need);
    if (at != kNoScratch) {
      blob->asym->terms_off = at;
    }
  }
}

/// Where a block's X_g, T_g and zero points go in the call's terms scratch.
static float* affine_toff(const esp32_asym::AffineWeights& w, int32_t* xsum,
                          uint32_t rows) {
  return w.offsets != nullptr ? reinterpret_cast<float*>(xsum + (size_t)rows * w.ngw)
                              : nullptr;
}
static int32_t* affine_zxv(const esp32_asym::AffineWeights& w, int32_t* xsum,
                           uint32_t rows) {
  return xsum + (size_t)rows * w.ngw * (w.offsets != nullptr ? 2 : 1);
}

/// fork_split for an affine block: the worker takes columns [0, half) and
/// join_split collects them.
static void fork_affine(const CallContext& c, const AffineBlock& b,
                        uint32_t half) {
  g_job.affine = &b;
#if ESP32_CONV_FAST
  g_job.conv = nullptr;
#endif
  g_job.wu = c.blob->wu2;
  g_job.n0 = 0;
  g_job.n1 = half;
  start_split_job();
}

/// run_rows_i8 for an affine op: one row at a time (decode).
static void run_rows_affine(const CallContext& c, uint32_t i,
                            const MatmulIo& io,
                            const esp32_asym::AffineWeights& w,
                            bool wq_aligned, const AffineAct& act, bool split) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  count_passes(M, M, (uint64_t)op.n * execution_row_bytes(op));
  int32_t* xsum = blob->asym_terms;
  float* toff = affine_toff(w, xsum, 1);
  int32_t* zx = affine_zxv(w, xsum, 1);
  const ActIn in{io.x, io.x8};
  uint32_t pie = 0, scalar = 0;
  for (uint32_t m = 0; m < M; ++m) {
    const float sx = affine_row(act, w, in.at((size_t)m * K), blob->xq, xsum,
                                toff, zx);
    const AffineBlock b{&op, &blob->resolved[i], &w, blob->xq, &sx, zx, xsum,
                        toff, 1, op.k_pad, io.bias,
                        io.y != nullptr ? io.y + (size_t)m * N : nullptr, N,
                        wq_aligned,
                        io.residual ? io.residual + (size_t)m * N : nullptr,
                        nullptr, esp32_asym::out_record(blob->asym, i),
                        io.y8 != nullptr ? io.y8 + (size_t)m * N : nullptr};
    uint32_t j_lo = 0;
    if (split) {
      j_lo = N / 2;
      fork_affine(c, b, j_lo);
    }
    affine_columns(b, blob->wu, j_lo, N, &pie, &scalar);
    if (split) {
      join_split(op, 1, &pie, &scalar);
    }
  }
  count_cols(pie, scalar);
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;
  }
  count_op((uint64_t)M * K * N, (uint64_t)N * execution_row_bytes(op));
}

/// run_blocked_i8 for an affine op: row blocks, and every convolution.
static void run_blocked_affine(const CallContext& c, uint32_t i,
                               const MatmulIo& io,
                               const esp32_asym::AffineWeights& w,
                               bool wq_aligned, const AffineAct& act,
                               bool split, uint32_t blk) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const ConvGeom* cg = blob->conv[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const ActIn in{io.x, io.x8};
  ActIn xg = in;
  float* yg = io.y;
  int8_t* yg8 = io.y8;   // Stage B: codes out
  if (cg != nullptr && cg->group_idx > 0) {
    xg = xg.at((size_t)cg->group_idx * cg->c_in * cg->d_in * cg->h_in *
               cg->w_in);
    const size_t yoff = (size_t)cg->group_idx * cg->c_out * M;
    if (yg8 != nullptr) {
      yg8 += yoff;
    } else {
      yg += yoff;
    }
  }
  int32_t* xsum = blob->asym_terms;
  float* toff = affine_toff(w, xsum, blk);
  int32_t* zxv = affine_zxv(w, xsum, blk);
  for (uint32_t m0 = 0; m0 < M; m0 += blk) {
    const uint32_t nrows = (m0 + blk <= M) ? blk : (M - m0);
    for (uint32_t t = 0; t < nrows; ++t) {
      int8_t* row = blob->xq + (size_t)t * op.k_pad;
      int32_t* xs = xsum + (size_t)t * w.ngw;
      float* tf = toff ? toff + (size_t)t * w.ngw : nullptr;
      blob->sxv[t] =
          cg ? affine_conv_row(act, w, xg, m0 + t, row, xs, tf, zxv + t)
             : affine_row(act, w, in.at((size_t)(m0 + t) * K), row, xs, tf,
                          zxv + t);
    }
    // A conv's block lands in conv_stage -- its codes too -- and is scattered
    // to NCHW below.
    float* dst = nullptr;
    int8_t* dst8 = nullptr;
    if (yg8 != nullptr) {
      dst8 = cg ? reinterpret_cast<int8_t*>(blob->conv_stage)
                : yg8 + (size_t)m0 * N;
    } else {
      dst = cg ? blob->conv_stage : yg + (size_t)m0 * N;
    }
    // A conv's residual is added in the NCHW scatter below, as in
    // run_blocked_i8: the kernel indexes it [m, n].
    const float* rblock =
        (io.residual && !cg) ? io.residual + (size_t)m0 * N : nullptr;
    const AffineBlock b{&op, &blob->resolved[i], &w, blob->xq, blob->sxv, zxv,
                        xsum, toff, nrows, op.k_pad, io.bias, dst, N,
                        wq_aligned, rblock, nullptr,
                        esp32_asym::out_record(blob->asym, i), dst8};
    uint32_t bpie = 0, bscalar = 0;
    uint32_t j_lo = 0;
    if (split) {
      j_lo = N / 2;
      fork_affine(c, b, j_lo);
    }
    affine_columns(b, blob->wu, j_lo, N, &bpie, &bscalar);
    if (split) {
      join_split(op, nrows, &bpie, &bscalar);
    }
    if (cg && yg8 != nullptr) {
      // Codes out: no residual (validate refuses an output record beside a
      // fused operand).
      const int8_t* stage = reinterpret_cast<const int8_t*>(blob->conv_stage);
      for (uint32_t t = 0; t < nrows; ++t) {
        const int8_t* src = stage + (size_t)t * N;
        int8_t* base = yg8 + (size_t)(m0 + t);
        for (uint32_t n = 0; n < N; ++n) {
          base[(size_t)n * M] = src[n];
        }
      }
    } else if (cg) {
      for (uint32_t t = 0; t < nrows; ++t) {
        const float* src = blob->conv_stage + (size_t)t * N;
        float* base = yg + (size_t)(m0 + t);
        const size_t row = (size_t)(m0 + t);
        for (uint32_t n = 0; n < N; ++n) {
          const size_t off = (size_t)n * M;
          base[off] = io.residual ? src[n] + io.residual[off + row] : src[n];
        }
      }
    }
    count_cols(bpie, bscalar);
  }
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;
  }
  const uint64_t rb = (uint64_t)op.n * execution_row_bytes(op);
  const uint64_t passes = (M + blk - 1) / blk;
  count_passes(passes, M, rb);
  count_op((uint64_t)M * K * N, rb);
}

/// run_depthwise_i8 for an affine op: each channel gathers its own input
/// (act's gather is the one-channel geometry).
static void run_depthwise_affine(const CallContext& c, uint32_t i,
                                 const MatmulIo& io,
                                 const esp32_asym::AffineWeights& w,
                                 bool wq_aligned, const AffineAct& act) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const ConvGeom& g = *blob->conv[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const bool is_i4 = executes_i4(op);
  const size_t plane = (size_t)g.d_in * g.h_in * g.w_in;
  const size_t stride = row_bytes(op.opcode, op.k_pad);
  int32_t* xsum = blob->asym_terms;
  float* toff = affine_toff(w, xsum, 1);
  int32_t zx = 0;
  const esp32_asym::OpQuant* oq = esp32_asym::out_record(blob->asym, i);
  const ActIn in{io.x, io.x8};
  uint32_t pie = 0, scalar = 0;
  for (uint32_t ch = 0; ch < N; ++ch) {
    const ActIn xin = in.at((size_t)ch * plane);
    const int8_t* wrow = weight_row(blob->resolved[i].wq, stride, ch, is_i4,
                                    op.k_pad, blob->wu);
    const bool dw_pie =
        wq_aligned && (reinterpret_cast<uintptr_t>(wrow) % kVecBytes) == 0;
    float* ybase = io.y != nullptr ? io.y + (size_t)ch * M : nullptr;
    int8_t* ybase8 = io.y8 != nullptr ? io.y8 + (size_t)ch * M : nullptr;
    for (uint32_t r = 0; r < M; ++r) {
      const float sx =
          affine_conv_row(act, w, xin, r, blob->xq, xsum, toff, &zx);
      const float v = esp32_asym::accumulate(
          w, ch, blob->xq, wrow, dw_pie, sx, zx, xsum, toff,
          io.bias ? io.bias[ch] : 0.0f);
      if (dw_pie) {
        ++pie;
      } else {
        ++scalar;
      }
      const float y = affine_epilogue(op, v, io.residual, (size_t)ch * M + r);
      if (ybase8 != nullptr) {
        ybase8[r] = esp32_asym::requantize_code(*oq, y);
      } else {
        ybase[r] = oq != nullptr ? esp32_asym::requantize(*oq, y) : y;
      }
    }
  }
  count_cols(pie, scalar);
  if (op.flags & kFlagStaticAct) {
    g_static_rows += (uint64_t)N * M;
  }
  count_op((uint64_t)M * K * N, (uint64_t)N * row_bytes(op.opcode, op.k_pad));
}

/// run_selected_rows for an affine op: the legal-token head's ID walk.
static Error run_selected_affine(const CallContext& c, uint32_t i,
                                 const MatmulIo& io,
                                 const esp32_asym::AffineWeights& w) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const SelectedRows& selection = *blob->selected[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  if (selection.rows_arg >= c.args.size()) {
    ET_LOG(Error, "esp32: selected-row op %u rows_arg out of range",
           (unsigned)i);
    return Error::InvalidArgument;
  }
  const auto rows_t = c.args[selection.rows_arg]->toTensor();
  if (io.in_numel != (int64_t)((uint64_t)M * K) ||
      io.out_numel != (int64_t)((uint64_t)M * selection.count) ||
      rows_t.numel() != (int64_t)selection.count) {
    ET_LOG(Error,
           "esp32: selected-row op %u shape mismatch "
           "(in=%d out=%d rows=%d, expected %ux%u -> %ux%u)",
           (unsigned)i, (int)io.in_numel, (int)io.out_numel,
           (int)rows_t.numel(), (unsigned)M, (unsigned)K, (unsigned)M,
           (unsigned)selection.count);
    return Error::InvalidArgument;
  }
  if (rows_t.scalar_type() != executorch::aten::ScalarType::Long) {
    ET_LOG(Error, "esp32: selected-row IDs must be int64");
    return Error::InvalidArgument;
  }
  const int64_t* rows = rows_t.const_data_ptr<int64_t>();
  for (uint32_t j = 0; j < selection.count; ++j) {
    if (rows[j] < 0 || rows[j] >= (int64_t)N) {
      ET_LOG(Error, "esp32: selected-row ID %d at %u is outside [0,%u)",
             (int)rows[j], (unsigned)j, (unsigned)N);
      return Error::InvalidArgument;
    }
  }
  const bool is_i4 = executes_i4(op);
  if (is_i4 && blob->wu == nullptr) {
    ET_LOG(Error, "esp32: selected i4 op with no unpack scratch");
    return Error::InvalidArgument;
  }
  const int8_t* kernel_row_base = is_i4 ? blob->wu : blob->resolved[i].wq;
  const bool aligned = pie_row_aligned(c.force_scalar, kernel_row_base,
                                       blob->xq, op.k_pad, w.gsize);
  const AffineAct act = affine_act(blob, i, io.x, (size_t)M * K, nullptr);
  int32_t* xsum = blob->asym_terms;
  float* toff = affine_toff(w, xsum, 1);
  int32_t* zx = affine_zxv(w, xsum, 1);
  uint32_t pie = 0, scalar = 0;
  for (uint32_t m = 0; m < M; ++m) {
    // fp32 only: attach refuses Stage B's codes on a selected-row head.
    const float sx = affine_row(act, w, ActIn{io.x + (size_t)m * K, nullptr},
                                blob->xq, xsum, toff, zx);
    // No operand: init refuses kFlagAdd and kFlagMul on a selected-row head.
    const AffineBlock b{&op, &blob->resolved[i], &w, blob->xq, &sx, zx, xsum,
                        toff, 1, op.k_pad, io.bias,
                        io.y + (size_t)m * selection.count, selection.count,
                        aligned, nullptr, rows,
                        esp32_asym::out_record(blob->asym, i), nullptr};
    affine_columns(b, blob->wu, 0, selection.count, &pie, &scalar);
  }
  count_cols(pie, scalar);
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;
  }
  const uint64_t rb = execution_row_bytes(op);
  count_passes(M, M, (uint64_t)selection.count * rb);
  count_op((uint64_t)M * K * selection.count, (uint64_t)selection.count * rb);
  return Error::Ok;
}

/// run_activation_group for affine ops: the members' one activation scheme
/// (same_affine_activation) quantises each row block once -- codes, scales,
/// zero points, X_g and T_g -- and every member runs its own columns over
/// them, split across the cores as it would be standalone.
static Error run_affine_group(const CallContext& c, uint32_t i,
                              const MatmulIo& io, int16_t grp) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k;
  const uint16_t gfirst = blob->group_first[grp];
  const uint16_t gcount = blob->group_count[grp];
  esp32_asym::AffineWeights mw[kMaxActGroup];
  const OpDesc* mop[kMaxActGroup];
  const float* mbias[kMaxActGroup];
  float* my[kMaxActGroup];
  const float* mr[kMaxActGroup];
  bool malign[kMaxActGroup];
  bool msplit[kMaxActGroup];
  bool offsets = false;
  for (uint16_t t = 0; t < gcount; ++t) {
    const uint16_t oi = blob->group_members[gfirst + t];
    const OpDesc& oq = blob->ops[oi];
    if (oq.out_arg >= kArgScratch0 ? blob->chain[0] == nullptr
                                   : oq.out_arg >= c.args.size()) {
      ET_LOG(Error, "esp32: group op %u out_arg out of range", (unsigned)oi);
      return Error::InvalidArgument;
    }
    if (oq.m != M || oq.k != K) {
      ET_LOG(Error, "esp32: group op %u shape disagrees with leader",
             (unsigned)oi);
      return Error::InvalidProgram;
    }
    if (oq.out_arg >= kArgScratch0) {
      my[t] = blob->chain[oq.out_arg - kArgScratch0];
    } else {
      auto mout = c.args[oq.out_arg]->toTensor();
      if (mout.numel() != (int64_t)((uint64_t)M * oq.n) ||
          mout.scalar_type() != executorch::aten::ScalarType::Float) {
        ET_LOG(Error, "esp32: group op %u output mismatch", (unsigned)oi);
        return Error::InvalidArgument;
      }
      my[t] = mout.mutable_data_ptr<float>();
    }
    if (executes_i4(oq) && blob->wu == nullptr) {
      ET_LOG(Error, "esp32: i4 op with no unpack scratch");
      return Error::InvalidArgument;
    }
    mop[t] = &oq;
    mbias[t] = oq.bias_off != 0
        ? reinterpret_cast<const float*>(blob->base + oq.bias_off)
        : nullptr;
    mr[t] = nullptr;
    if ((oq.flags & kFlagMul) && oq.res_arg >= kArgScratch0) {
      mr[t] = blob->chain[oq.res_arg - kArgScratch0];
    } else if (oq.flags & (kFlagAdd | kFlagMul)) {
      if (oq.res_arg >= c.args.size()) {
        ET_LOG(Error, "esp32: group op %u res_arg out of range",
               (unsigned)oi);
        return Error::InvalidArgument;
      }
      const auto rt = c.args[oq.res_arg]->toTensor();
      if (rt.numel() != (int64_t)((uint64_t)M * oq.n) ||
          rt.scalar_type() != executorch::aten::ScalarType::Float) {
        ET_LOG(Error, "esp32: group op %u residual mismatch", (unsigned)oi);
        return Error::InvalidArgument;
      }
      mr[t] = rt.const_data_ptr<float>();
    }
    mw[t] = esp32_asym::affine_weights(
        blob->asym->ops[oi], oq.k, oq.k_pad, oq.scale_group,
        (oq.flags & kFlagFp16Scales) != 0, blob->resolved[oi].scale);
    offsets = offsets || mw[t].offsets != nullptr;
    const int8_t* base_t = executes_i4(oq) ? blob->wu : blob->resolved[oi].wq;
    malign[t] = pie_row_aligned(c.force_scalar, base_t, blob->xq, oq.k_pad,
                                mw[t].gsize);
    msplit[t] = split_columns(c, oq, false);
  }
  // The leader's scheme is every member's (the grouping key): its chooser
  // runs once per call, or per row, for all of them.
  const AffineAct act = affine_act(blob, i, io.x, (size_t)M * K, nullptr);
  esp32_asym::AffineWeights terms = mw[0];
  if (offsets) {
    for (uint16_t t = 0; t < gcount; ++t) {
      if (mw[t].offsets != nullptr) {
        terms = mw[t];       // lays out T_g beside X_g for the members with
        break;               // float offsets (the others never read it)
      }
    }
  }
  const uint32_t gblk = row_block_for(M, op.k_pad, nullptr, 0);
  int32_t* xsum = blob->asym_terms;
  float* toff = affine_toff(terms, xsum, gblk);
  int32_t* zxv = affine_zxv(terms, xsum, gblk);
  g_affine_ops.fetch_add(gcount, std::memory_order_relaxed);
  for (uint32_t m0 = 0; m0 < M; m0 += gblk) {
    const uint32_t nrows = (m0 + gblk <= M) ? gblk : (M - m0);
    // One row's scale on this frame; a block uses sxv, which exists because
    // nrows > 1 implies block_rows > 1 at init (run_activation_group's rule).
    float sx1 = 0.0f;
    float* sxv = nrows == 1 ? &sx1 : blob->sxv;
    for (uint32_t t = 0; t < nrows; ++t) {
      // fp32 only: Stage B's ops share no group (can_share_affine).
      sxv[t] = affine_row(act, terms,
                          ActIn{io.x + (size_t)(m0 + t) * K, nullptr},
                          blob->xq + (size_t)t * op.k_pad,
                          xsum + (size_t)t * terms.ngw,
                          toff ? toff + (size_t)t * terms.ngw : nullptr,
                          zxv + t);
    }
    for (uint16_t t = 0; t < gcount; ++t) {
      const OpDesc& oq = *mop[t];
      const uint16_t member = blob->group_members[gfirst + t];
      const AffineBlock b{&oq, &blob->resolved[member],
                          &mw[t], blob->xq, sxv, zxv, xsum, toff, nrows,
                          op.k_pad, mbias[t], my[t] + (size_t)m0 * oq.n, oq.n,
                          malign[t],
                          mr[t] ? mr[t] + (size_t)m0 * oq.n : nullptr,
                          nullptr, esp32_asym::out_record(blob->asym, member),
                          nullptr};
      uint32_t gpie = 0, gscalar = 0;
      uint32_t j_lo = 0;
      if (msplit[t]) {
        j_lo = oq.n / 2;
        fork_affine(c, b, j_lo);
      }
      affine_columns(b, blob->wu, j_lo, oq.n, &gpie, &gscalar);
      if (msplit[t]) {
        join_split(oq, nrows, &gpie, &gscalar);
      }
      count_cols(gpie, gscalar);
    }
  }
  const uint64_t gpasses = (M + gblk - 1) / gblk;
  for (uint16_t t = 0; t < gcount; ++t) {
    const uint64_t rb = (uint64_t)mop[t]->n * execution_row_bytes(*mop[t]);
    count_op((uint64_t)M * K * mop[t]->n, rb);
    count_passes(gpasses, M, rb);
  }
  g_actq_rows_saved += (uint64_t)(gcount - 1) * M;
  if (op.flags & kFlagStaticAct) {
    g_static_rows += M;
  }
  return Error::Ok;
}

/// 16-bit activations (Arm a16w8): the op's static
/// int16 record quantises one row -- a conv: one output position's patch,
/// its padding the record's zero point -- into int16 codes in the terms
/// scratch, then every column's int64 dot and epilogue
/// (esp32_asym::a16_column), the op's fused activation, clamp and operand,
/// and its output quantisation. Scalar and on one core, as run_addmm_i16 is:
/// int16 codes halve the SIMD width, and the int64 dot is an accuracy rung's
/// arithmetic, not the fast path's. A depthwise conv gathers per channel; a
/// grouped conv's slice starts at its group's channels.
static void run_addmm_a16(const CallContext& c, uint32_t i,
                          const MatmulIo& io,
                          const esp32_asym::AffineWeights& w) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const ConvGeom* cg = blob->conv[i];
  const esp32_asym::ActQuant& a = *esp32_asym::act_record(blob->asym, i);
  const esp32_asym::OpQuant* oq = esp32_asym::out_record(blob->asym, i);
  int16_t* xq = reinterpret_cast<int16_t*>(blob->asym_terms);
  const int8_t* wq = blob->resolved[i].wq;
  const size_t stride = op.k_pad;
  if (cg != nullptr && is_depthwise(*cg)) {
    ConvGeom one = *cg;      // each channel gathers its own input
    one.c_in = 1;
    one.c_out = 1;
    one.groups = 1;
    const esp32_asym::ConvGather g = conv_gather(one);
    const size_t plane = (size_t)cg->d_in * cg->h_in * cg->w_in;
    for (uint32_t ch = 0; ch < N; ++ch) {
      const int8_t* wrow = wq + (size_t)ch * stride;
      for (uint32_t r = 0; r < M; ++r) {
        esp32_asym::quantize_conv_row16(a, io.x + (size_t)ch * plane, g, r,
                                        op.k_pad, xq);
        const float v = esp32_asym::a16_column(
            w, ch, xq, wrow, a.scale, io.bias != nullptr ? io.bias[ch] : 0.0f);
        const size_t at = (size_t)ch * M + r;
        const float y = affine_epilogue(op, v, io.residual, at);
        io.y[at] = oq != nullptr ? esp32_asym::requantize(*oq, y) : y;
      }
    }
  } else {
    const float* xg = io.x;
    float* yg = io.y;
    esp32_asym::ConvGather g{};
    if (cg != nullptr) {
      g = conv_gather(*cg);
      if (cg->group_idx > 0) {
        xg += (size_t)cg->group_idx * cg->c_in * cg->d_in * cg->h_in *
              cg->w_in;
        yg += (size_t)cg->group_idx * cg->c_out * M;
      }
    }
    for (uint32_t m = 0; m < M; ++m) {
      if (cg != nullptr) {
        esp32_asym::quantize_conv_row16(a, xg, g, m, op.k_pad, xq);
      } else {
        esp32_asym::quantize_row16(a, io.x + (size_t)m * K, K, op.k_pad, xq);
      }
      for (uint32_t n = 0; n < N; ++n) {
        const float v = esp32_asym::a16_column(
            w, n, xq, wq + (size_t)n * stride, a.scale,
            io.bias != nullptr ? io.bias[n] : 0.0f);
        // [m, n] for a matmul, NCHW for a conv: run_addmm_i16's indexing.
        const size_t at = cg != nullptr ? (size_t)n * M + m : (size_t)m * N + n;
        const float y = affine_epilogue(op, v, io.residual, at);
        (cg != nullptr ? yg : io.y)[at] =
            oq != nullptr ? esp32_asym::requantize(*oq, y) : y;
      }
    }
  }
  count_cols(0, M * N);   // every output element is one scalar column
  count_op((uint64_t)M * K * N, (uint64_t)N * row_bytes(op.opcode, op.k_pad));
}

#if ESP32_CONV_FAST
/// A conv on the static int8 path split across the cores (run_conv_fast):
/// the worker computes `part` of it on its own scratch, and counts it here.
struct ConvSplit {
  const esp32_conv::Op* op;
  const esp32_conv::Act* act;
  const esp32_conv::Columns* cols;
  const esp32_conv::Out* out;
  esp32_conv::Scratch scratch;
  esp32_conv::Part part;
  esp32_conv::Counts counts;
  // Non-null: the job is elements `part` of the op's fp32 input quantised
  // into this plane (esp32_conv::prepare_part), not outputs.
  int8_t* plane;
};

/// The worker's half of a split conv (run_split_job): its part, timed.
void SplitJob::run_conv(uint64_t t0) {
  ConvSplit& j = *g_job.conv;
  if (j.plane != nullptr) {
    esp32_conv::prepare_part(*j.act, j.plane, j.part);
  } else {
    esp32_conv::run_part(*j.op, *j.act, *j.cols, *j.out, j.scratch, j.part,
                         &j.counts);
  }
  g_job.worker_us = ESP32_TICK_NOW() - t0;
}

/// Whether this call splits op i, on the path, across the cores: the
/// min-latency posture with the worker leased, a splittable op, and the
/// worker's scratch when its part needs one.
static bool conv_split(const CallContext& c, const esp32_conv::Op& geo,
                       uint64_t part_bytes) {
  return c.global_split && c.split_enabled &&
         g_worker.load(std::memory_order_relaxed) != nullptr &&
         conv_fast_splittable(geo) &&
         (part_bytes == 0 || c.blob->conv_split != nullptr);
}

#if defined(ESP32_HOST_BUILD) || !defined(NDEBUG)
/// Host and debug builds, before a split conv's fork: the two cores' scratch
/// is disjoint, and neither overlaps the plane both read. Each part writes
/// its own scratch, part_scratch_need bytes from its base; the plane was
/// made before the fork and is only read until the join. An overlap would
/// let one core gather into the row block or plane the other reads.
static void check_split_scratch(const esp32_conv::Op& geo, bool fp32_in,
                                uint32_t rows, const esp32_conv::Scratch& mine,
                                const esp32_conv::Scratch& theirs) {
  const uint64_t need = esp32_conv::part_scratch_need(geo, fp32_in, rows);
  const uint64_t plane =
      mine.plane != nullptr ? esp32_conv::plane_elements(geo) : 0;
  const auto apart = [](const int8_t* a, uint64_t na, const int8_t* b,
                        uint64_t nb) {
    return na == 0 || nb == 0 || a + na <= b || b + nb <= a;
  };
  ET_CHECK_MSG(apart(mine.base, need, theirs.base, need) &&
                   apart(mine.base, need, mine.plane, plane) &&
                   apart(theirs.base, need, mine.plane, plane),
               "esp32: a split conv's scratch overlaps: caller %p, worker %p, "
               "%llu B each; plane %p, %llu B",
               (const void*)mine.base, (const void*)theirs.base,
               (unsigned long long)need, (const void*)mine.plane,
               (unsigned long long)plane);
}
#endif

/// Hand the worker its part of a conv on the path (join_conv collects it).
static void fork_conv(ConvSplit* job) {
  g_job.conv = job;
  g_job.affine = nullptr;
  start_split_job();
}

/// Wait for the worker's part and fold its counts into the caller's (none
/// for the halves of a plane).
static void join_conv(const ConvSplit& job, esp32_conv::Counts* counts) {
#if !defined(ESP32_HOST_BUILD)
  xSemaphoreTake(g_split_done, portMAX_DELAY);
#else
  esp32_host_split_join();
#endif
  g_job.conv = nullptr;
#if defined(ESP32_BACKEND_TIMING)
  g_split_worker_us.fetch_add(g_job.worker_us, std::memory_order_relaxed);
#endif
  if (counts == nullptr) {
    return;
  }
  counts->pie += job.counts.pie;
  counts->scalar += job.counts.scalar;
  counts->outputs += job.counts.outputs;
  counts->fixed += job.counts.fixed;
  counts->fallback += job.counts.fallback;
  g_conv_split_ops.fetch_add(1, std::memory_order_relaxed);
}

/// Op i on the static int8 convolution path (esp32_conv.h), when it takes it
/// (conv_fast_op): the general loops' result bit for bit, and their counters,
/// from codes made once per input element. False, with nothing written,
/// otherwise. Every op runs whole on this core: the split below (the worker
/// computing the low half of the op's units on its own scratch while this
/// core computes the high half, under min-latency) is off (kConvSplit; the
/// board: README, Known limits).
[[gnu::noinline]] static bool run_conv_fast(const CallContext& c, uint32_t i,
                                            const MatmulIo& io,
                                            const esp32_asym::AffineWeights& w,
                                            bool wq_aligned,
                                            const AffineAct& act) {
  const CompiledBlob* blob = c.blob;
  if (g_conv_fast_override.load(std::memory_order_relaxed) <= 0 ||
      !conv_fast_op(blob, i) || act.rec == nullptr || w.ngw != 1 ||
      w.zero_points != nullptr || w.zw != 0 || w.offsets != nullptr) {
    return false;
  }
  const OpDesc& op = blob->ops[i];
  const ConvGeom& g = *blob->conv[i];
  const esp32_conv::Op geo = conv_fast_geometry(op, g);
  const uint32_t M = op.m;
  // A split grouped conv owns one channel slice of the input and output, as
  // in run_blocked_affine; a depthwise op's channels are its columns.
  size_t in_off = 0, out_off = 0;
  if (!geo.depthwise && g.group_idx > 0) {
    in_off = (size_t)g.group_idx * g.c_in * g.h_in * g.w_in;
    out_off = (size_t)g.group_idx * g.c_out * M;
  }
  esp32_conv::Act a;
  a.x8 = io.x8 != nullptr ? io.x8 + in_off : nullptr;
  a.x = io.x != nullptr ? io.x + in_off : nullptr;
  // The record's constants (static, or chosen once over the tensor).
  a.qmin = act.rec->qmin;
  a.qmax = act.rec->qmax;
  a.sx = act.tensor.sx;
  a.zx = act.tensor.zx;
  esp32_conv::Columns cols;
  cols.wq = blob->resolved[i].wq;
  cols.pie = wq_aligned;
  cols.scales = w.scales;
  cols.scales_fp16 = w.fp16_scales;
  cols.sums = w.sums;
  cols.sums_i16 = w.sums_i16;
  cols.bias = io.bias;
  const esp32_asym::OpQuant* oq = esp32_asym::out_record(blob->asym, i);
  esp32_conv::Out out;
  out.y = io.y != nullptr ? io.y + out_off : nullptr;
  out.y8 = io.y8 != nullptr ? io.y8 + out_off : nullptr;
  out.requant = oq != nullptr;
  out.out_inv = oq != nullptr ? oq->out_inv : 0.0f;
  out.out_scale = oq != nullptr ? oq->out->scale : 0.0f;
  out.zo = oq != nullptr ? oq->out->zero_point : 0;
  out.qmin = oq != nullptr ? oq->out->qmin : 0;
  out.qmax = oq != nullptr ? oq->out->qmax : 0;
  const uint32_t blk = geo.depthwise ? 0 : conv_fast_rows(op);
  esp32_conv::Counts counts{0, 0, 0};
  esp32_conv::Scratch sc{blob->xq, blk, nullptr};
  const uint32_t units = esp32_conv::units(geo);
  const uint32_t half = esp32_conv::split_at(geo);
  const bool split = conv_split(
      c, geo, esp32_conv::part_scratch_need(geo, a.x8 == nullptr, blk));
  ConvSplit job;
  int8_t* plane = esp32_conv::plane_of(geo, a, sc);
  if (plane != nullptr) {
    // The fp32 input's codes, before any output: halves on both cores
    // when the op splits (each element's code is the same either way).
    const uint32_t total = esp32_conv::plane_elements(geo);
    const bool share = split && total >= kConvSplitMinPlane;
    esp32_conv::Part mine{0, total};
    if (share) {
      job = ConvSplit{&geo, &a, &cols, &out, sc, esp32_conv::Part{0, total / 2},
                      esp32_conv::Counts{0, 0, 0}, plane};
      fork_conv(&job);
      mine.lo = total / 2;
    }
    esp32_conv::prepare_part(a, plane, mine);
    if (share) {
      join_conv(job, nullptr);
    }
    sc.plane = plane;
  }
  if (split) {
    job = ConvSplit{&geo,
                    &a,
                    &cols,
                    &out,
                    esp32_conv::Scratch{blob->conv_split, blk, sc.plane},
                    esp32_conv::Part{0, half},
                    esp32_conv::Counts{0, 0, 0},
                    nullptr};
#if defined(ESP32_HOST_BUILD) || !defined(NDEBUG)
    check_split_scratch(geo, a.x8 == nullptr, blk, sc, job.scratch);
#endif
    fork_conv(&job);
  }
  esp32_conv::run_part(geo, a, cols, out, sc,
                       esp32_conv::Part{split ? half : 0, units}, &counts);
  if (split) {
    join_conv(job, &counts);
  }
  // The general loops' accounting, so a report reads the same either way
  // (the depthwise dots are scalar here: counted so).
  count_cols(counts.pie, counts.scalar);
  g_conv_fast_ops.fetch_add(1, std::memory_order_relaxed);
  g_conv_fast_outputs.fetch_add(counts.outputs, std::memory_order_relaxed);
  g_conv_epi_fixed.fetch_add(counts.fixed, std::memory_order_relaxed);
  g_conv_epi_fallback.fetch_add(counts.fallback, std::memory_order_relaxed);
  if (geo.depthwise) {
    if (op.flags & kFlagStaticAct) {
      g_static_rows += (uint64_t)op.n * M;
    }
    count_op((uint64_t)M * op.k * op.n,
             (uint64_t)op.n * row_bytes(op.opcode, op.k_pad));
  } else {
    if (op.flags & kFlagStaticAct) {
      g_static_rows += M;
    }
    const uint64_t rb = (uint64_t)op.n * execution_row_bytes(op);
    // One pass over the weights per block of each part.
    const uint64_t passes =
        split ? (uint64_t)(half + blk - 1) / blk + (M - half + blk - 1) / blk
              : (uint64_t)(M + blk - 1) / blk;
    count_passes(passes, M, rb);
    count_op((uint64_t)M * op.k * op.n, rb);
  }
  return true;
}
#endif

/// run_addmm_quantised for an affine op: the same tensor checks and loop
/// choice, or run_affine_group for a group's leader. Out of line, with
/// run_embedding_affine, so that execute() -- every op's path -- grows by a
/// call, not by the affine loops.
[[gnu::noinline]] static Error run_addmm_affine(const CallContext& c, uint32_t i,
                              const MatmulIo& io, int16_t grp) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const esp32_asym::AffineWeights w = esp32_asym::affine_weights(
      blob->asym->ops[i], op.k, op.k_pad, op.scale_group,
      (op.flags & kFlagFp16Scales) != 0, blob->resolved[i].scale);
  if (blob->asym_terms == nullptr) {
    ET_LOG(Error, "esp32: affine op %u with no terms scratch", (unsigned)i);
    return Error::InvalidArgument;
  }
  if (grp >= 0) {
    return run_affine_group(c, i, io, grp);
  }
  g_affine_ops.fetch_add(1, std::memory_order_relaxed);
  if (blob->selected[i] != nullptr) {
    return run_selected_affine(c, i, io, w);
  }
  const ConvGeom* cg = blob->conv[i];
  const uint64_t want_in = op_input_elems(op, cg);
  const uint64_t want_out = op_output_elems(op, cg);
  if ((io.in_numel >= 0 && io.in_numel != (int64_t)want_in) ||
      (io.out_numel >= 0 && io.out_numel != (int64_t)want_out)) {
    ET_LOG(Error, "esp32: i8 shape mismatch (in %d/%u out %d/%u)",
           (int)io.in_numel, (unsigned)want_in, (int)io.out_numel,
           (unsigned)want_out);
    return Error::InvalidArgument;
  }
  if (esp32_asym::a16_op(blob->asym, i)) {
    run_addmm_a16(c, i, io, w);
    return Error::Ok;
  }
  const bool is_i4 = executes_i4(op);
  if (is_i4 && blob->wu == nullptr) {
    ET_LOG(Error, "esp32: i4 op with no unpack scratch");
    return Error::InvalidArgument;
  }
  const int8_t* kernel_row_base = is_i4 ? blob->wu : blob->resolved[i].wq;
  const bool wq_aligned = pie_row_aligned(c.force_scalar, kernel_row_base,
                                          blob->xq, op.k_pad, w.gsize);
  const bool is_conv = cg != nullptr;
  ConvGeom one;
  const ConvGeom* gather = cg;
  if (is_conv && is_depthwise(*cg)) {
    one = *cg;               // each channel gathers its own input
    one.c_in = 1;
    one.c_out = 1;
    one.groups = 1;
    gather = &one;
  }
  const AffineAct act =
      affine_act(blob, i, io.x, (size_t)op_input_elems(op, cg), gather);
#if ESP32_CONV_FAST
  if (is_conv && run_conv_fast(c, i, io, w, wq_aligned, act)) {
    return Error::Ok;
  }
#endif
  const bool split = split_columns(c, op, is_conv);
  const uint32_t blk = row_block_for(op.m, op.k_pad, cg, op.n);
  if (is_conv && is_depthwise(*cg)) {
    run_depthwise_affine(c, i, io, w, wq_aligned, act);
  } else if (blk > 1 || is_conv) {
    run_blocked_affine(c, i, io, w, wq_aligned, act, split, blk);
  } else {
    run_rows_affine(c, i, io, w, wq_aligned, act, split);
  }
  return Error::Ok;
}

/// run_embedding for a table with EXT_WEIGHT_AFFINE: the same checks, each
/// gathered row dequantised by esp32_asym::dequant_row.
[[gnu::noinline]] static Error run_embedding_affine(const CallContext& c, uint32_t i) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const auto in = c.args[op.in_arg]->toTensor();
  const uint32_t M = op.m, D = op.k, V = op.n;
  float* yv;
  if (op.out_arg >= kArgScratch0) {
    yv = blob->chain[op.out_arg - kArgScratch0];
  } else {
    auto out = c.args[op.out_arg]->toTensor();
    if (out.numel() != (int64_t)((uint64_t)M * D)) {
      ET_LOG(Error, "esp32: embedding out %d != %ux%u", (int)out.numel(),
             (unsigned)M, (unsigned)D);
      return Error::InvalidArgument;
    }
    if (out.scalar_type() != executorch::aten::ScalarType::Float) {
      ET_LOG(Error, "esp32: embedding output must be float32");
      return Error::InvalidArgument;
    }
    yv = out.mutable_data_ptr<float>();
  }
  if (in.numel() != (int64_t)M) {
    ET_LOG(Error, "esp32: embedding index numel %d != M %u",
           (int)in.numel(), (unsigned)M);
    return Error::InvalidArgument;
  }
  if (in.scalar_type() != executorch::aten::ScalarType::Long) {
    ET_LOG(Error, "esp32: embedding indices must be int64");
    return Error::InvalidArgument;
  }
  const int64_t* idx = in.const_data_ptr<int64_t>();
  const uint8_t* table =
      reinterpret_cast<const uint8_t*>(blob->resolved[i].wq);
  const bool s_fp16 = (op.flags & kFlagFp16Scales) != 0;
  const uint32_t gsize = op.scale_group ? op.scale_group : D;
  const uint32_t ng = (D + gsize - 1) / gsize;
  const size_t stride = row_bytes(op.opcode, op.k_pad);
  const esp32_asym::OpQuant& q = blob->asym->ops[i];
  for (uint32_t j = 0; j < M; ++j) {
    const int64_t t = idx[j];
    if (t < 0 || t >= (int64_t)V) {
      ET_LOG(Error, "esp32: embedding index %d out of range [0,%u)", (int)t,
             (unsigned)V);
      return Error::InvalidArgument;
    }
    esp32_asym::dequant_row(q, op.opcode == kOpEmbedI4, table + (size_t)t * stride,
                            D, gsize, ng, blob->resolved[i].scale, s_fp16,
                            (size_t)t * ng, yv + (size_t)j * D);
  }
  g_weight_bytes += (uint64_t)M * stride;
  ++g_embed_ops;
  g_affine_ops.fetch_add(1, std::memory_order_relaxed);
  return Error::Ok;
}

// ---- the float-weight loops, then the quantised i8/i4 family ----------------
// run_addmm_quantised (below) validates the tensors once, then chooses the
// loop: selected-row heads and activation-sharing groups own their checks; the
// rest share the alignment gate, the activation scale and the split decision.
/// Weights read as the floats they store (esp32_float.h): opcode 1 with
/// kFlagConv, the fp32 convolution, and the fp16/bf16 rungs' matmul and
/// convolution. The loops write bias + dot, the sum run_addmm_fp32 forms;
/// the fused epilogue is a final pass over what they wrote, as there.
///
/// noinline, with run_embedding_float, for speed: inlined, they grow
/// execute() until the loops GCC inlines there lose registers -- the fp32
/// Linear's dot (run_addmm_fp32) would reload its bound from the stack every
/// iteration, 9.4 -> 10.4 instructions a MAC, and this
/// function's epilogue pass would take two more an element.
#if defined(ET_PLE_ONLY)
// The story firmware's stand-ins: it links no float-weight loop, and
// validate_blob refuses their ops by name before execute could reach these.
static Error run_float_weights(const CallContext&, uint32_t, const MatmulIo&) {
  return Error::NotSupported;
}
static Error run_embedding_float(const CallContext&, uint32_t) {
  return Error::NotSupported;
}
#else
[[gnu::noinline]] static Error run_float_weights(const CallContext& c,
                                                uint32_t i,
                                                const MatmulIo& io) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const ConvGeom* cg = blob->conv[i];
  const uint32_t M = op.m, K = op.k, N = op.n;
  const esp32_float::Storage s = esp32_float::storage_of(op.opcode);
  const uint64_t want_in = op_input_elems(op, cg);
  const uint64_t want_out = op_output_elems(op, cg);
  if ((io.in_numel >= 0 && io.in_numel != (int64_t)want_in) ||
      (io.out_numel >= 0 && io.out_numel != (int64_t)want_out) ||
      (esp32_float::scratch_bytes(s, cg != nullptr, M, K, N) != 0 &&
       blob->xq == nullptr)) {
    ET_LOG(Error, "esp32: float-weight shape/scratch mismatch (in %d/%u out "
                  "%d/%u)",
           (int)io.in_numel, (unsigned)want_in, (int)io.out_numel,
           (unsigned)want_out);
    return Error::InvalidArgument;
  }
  const void* w = blob->resolved[i].wq;
  float* y = io.y;
  if (cg != nullptr) {
    // A grouped conv's channel slices, as the i8 and i16 paths take them:
    // c_in and c_out in the record are per group.
    const float* x = io.x;
    if (cg->group_idx > 0) {
      x += (size_t)cg->group_idx * cg->c_in * cg->d_in * cg->h_in * cg->w_in;
      y += (size_t)cg->group_idx * cg->c_out * M;
    }
    esp32_float::conv(s, w, x, conv_gather(*cg), is_depthwise(*cg), io.bias,
                      y, M, K, N, blob->xq);
  } else {
    esp32_float::matmul(s, w, io.x, io.bias, y, M, K, N, blob->xq);
  }
  if (op.flags & kEpilogueFlags) {
    // [m, n] for a matmul, NCHW for a conv: either way element t of what the
    // loop wrote is the residual's element t (init refuses a residual on a
    // grouped conv's slice, and kFlagMul on every float loop).
    const size_t total = (size_t)M * N;
    for (size_t t = 0; t < total; ++t) {
      y[t] = epilogue(op, y[t], io.residual, t);
    }
  }
  count_op((uint64_t)M * K * N,
           (uint64_t)N * K *
               (s == esp32_float::Storage::kF32 ? sizeof(float)
                                                : sizeof(uint16_t)));
  return Error::Ok;
}

/// Opcodes 13/14: a gather from an fp16 or bf16 table, each row widened bit
/// for bit (esp32_float::embed_row). run_embedding's argument checks; no
/// scales.
[[gnu::noinline]] static Error run_embedding_float(const CallContext& c,
                                                  uint32_t i) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  const auto in = c.args[op.in_arg]->toTensor();
  const uint32_t M = op.m, D = op.k, V = op.n;
  float* yv;
  if (op.out_arg >= kArgScratch0) {
    yv = blob->chain[op.out_arg - kArgScratch0];
  } else {
    auto out = c.args[op.out_arg]->toTensor();
    if (out.numel() != (int64_t)((uint64_t)M * D) ||
        out.scalar_type() != executorch::aten::ScalarType::Float) {
      ET_LOG(Error, "esp32: embedding out %d != %ux%u float32",
             (int)out.numel(), (unsigned)M, (unsigned)D);
      return Error::InvalidArgument;
    }
    yv = out.mutable_data_ptr<float>();
  }
  if (in.numel() != (int64_t)M ||
      in.scalar_type() != executorch::aten::ScalarType::Long) {
    ET_LOG(Error, "esp32: embedding indices must be %u int64", (unsigned)M);
    return Error::InvalidArgument;
  }
  const int64_t* idx = in.const_data_ptr<int64_t>();
  const esp32_float::Storage s = esp32_float::storage_of(op.opcode);
  for (uint32_t j = 0; j < M; ++j) {
    const int64_t t = idx[j];
    if (t < 0 || t >= (int64_t)V) {
      ET_LOG(Error, "esp32: embedding index %d out of range [0,%u)", (int)t,
             (unsigned)V);
      return Error::InvalidArgument;
    }
    esp32_float::embed_row(s, blob->resolved[i].wq, (uint32_t)t, D,
                           yv + (size_t)j * D);
  }
  g_weight_bytes += (uint64_t)M * D * sizeof(uint16_t);
  ++g_embed_ops;
  return Error::Ok;
}
#endif  // ET_PLE_ONLY
/// The quantised i8/i4 family: validate the tensors once, then choose the loop.
ESP32_HOT_CODE
static Error run_addmm_quantised(const CallContext& c, uint32_t i,
                                 const MatmulIo& io, int16_t grp) {
  const CompiledBlob* blob = c.blob;
  const OpDesc& op = blob->ops[i];
  // Weight and activation records take their own loops, chosen
  // here once per op; the symmetric ones below never see them.
  if (esp32_asym::affine_op(blob->asym, i)) {
    return run_addmm_affine(c, i, io, grp);
  }
  if (blob->selected[i] != nullptr) {
    return run_selected_rows(c, i, io);
  }
  // A conv's INPUT is [1, C_in, H_in, W_in] -- not the [M,K] a matmul's is;
  // the gather is what turns one into the other. Its OUTPUT is the whole
  // tensor even when this op writes one channel slice of it. The check is
  // the one that catches a partition wired to the wrong tensor, so it is
  // narrowed to the op's true extents rather than dropped.
  const ConvGeom* cg = blob->conv[i];
  const uint64_t want_in = op_input_elems(op, cg);
  const uint64_t want_out = op_output_elems(op, cg);
  if ((io.in_numel >= 0 && io.in_numel != (int64_t)want_in) ||
      (io.out_numel >= 0 && io.out_numel != (int64_t)want_out)) {
    ET_LOG(Error, "esp32: i8 shape mismatch (in %d/%u out %d/%u)",
           (int)io.in_numel, (unsigned)want_in, (int)io.out_numel,
           (unsigned)want_out);
    return Error::InvalidArgument;
  }
  const bool is_i4 = executes_i4(op);
  if (is_i4 && blob->wu == nullptr) {
    ET_LOG(Error, "esp32: i4 op with no unpack scratch");
    return Error::InvalidArgument;
  }
  if (grp >= 0) {
    return run_activation_group(c, i, io, grp);
  }
  // For i4 the pointer the kernel actually reads is the unpack scratch,
  // which is hand-aligned at init; the packed source is read a byte at a
  // time and has no alignment requirement of its own. Checking the wrong
  // one of the two would disable PIE for every i4 op, silently and only in
  // the timing.
  const uint32_t gsize = op.scale_group ? op.scale_group : op.k_pad;
  const int8_t* kernel_row_base = is_i4 ? blob->wu : blob->resolved[i].wq;
  const bool wq_aligned = pie_row_aligned(c.force_scalar, kernel_row_base,
                                          blob->xq, op.k_pad, gsize);
  const float fix =
      fixed_activation_scale(op, io.x, (size_t)op_input_elems(op, cg), 1);
  const bool is_conv = cg != nullptr;
  const bool split = split_columns(c, op, is_conv);
  const uint32_t blk = row_block_for(op.m, op.k_pad, cg, op.n);
  if (is_conv && is_depthwise(*cg)) {
    run_depthwise_i8(c, i, io, wq_aligned, fix);
  } else if (blk > 1 || is_conv) {
    run_blocked_i8(c, i, io, wq_aligned, fix, split, blk);
  } else {
    run_rows_i8(c, i, io, wq_aligned, fix, split);
  }
  return Error::Ok;
}

/// Resolve this call's posture ONCE: the process-wide override wins if set,
/// otherwise the handle's own compile-spec setting decides. Read here rather
/// than at each op so a setter racing mid-execute cannot make one op in a
/// partition split while the next does not. Runtime force-off reserves the
/// second core completely; force-on is the global latency ceiling; with no
/// override the handle may request global splitting or the narrower
/// head-split posture where only the kFlagHeadSplit descriptor forks.
static CallContext resolve_call_posture(CompiledBlob* call,
                                        Span<EValue*> args) {
  const int split_override = g_split_override.load(std::memory_order_relaxed);
  CallContext c;
  c.blob = call;
  c.args = args;
  c.force_scalar =
      g_force_scalar_override.load(std::memory_order_relaxed) > 0;
  c.global_split =
      split_override > 0 || (split_override < 0 && call->cfg_min_latency);
  c.selective_split = split_override < 0 && call->cfg_head_split;
  c.split_enabled = false;
  return c;
}

/// Bytes as an option value: options are int, so a total past INT32_MAX reads
/// as INT32_MAX.
static int option_bytes(uint64_t bytes) {
  return bytes > 0x7FFFFFFFull ? 0x7FFFFFFF : (int)bytes;
}

/// The bytes the loaded methods' candidates would occupy, each weight once.
/// Caller holds lifecycle_lock().
static uint64_t weight_tier_wanted() {
  uint64_t wanted = 0;
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    if (g_weight_cache_entries[i].refs > 0) {
      wanted += g_weight_cache_entries[i].alloc;
    }
  }
  return wanted;
}

/// set_option's weight-tier keys (described with kOptWeightTierBudget).
/// *handled says whether `option` was one of them.
static Error set_weight_tier_option(const BackendOption& option,
                                    bool* handled) {
  *handled = true;
  const int* i = std::get_if<int>(&option.value);
  const bool* b = std::get_if<bool>(&option.value);
  if (std::strcmp(option.key, kOptWeightTierBudget) == 0) {
    if (i == nullptr || *i < kWeightTierAuto) {
      ET_LOG(Error, "esp32: weight_tier_budget must be an int: bytes, or -1 "
                    "for auto");
      return Error::InvalidArgument;
    }
    LifecycleGuard guard;
    g_weight_tier_auto = *i == kWeightTierAuto;
    g_weight_tier_budget = g_weight_tier_auto ? 0 : (size_t)*i;
    if (!weight_tier_budgeted()) {
      release_unused_expansions();
    }
    return Error::Ok;
  }
  if (std::strcmp(option.key, kOptWeightTierReserve) == 0) {
    if (i == nullptr || *i < 0) {
      ET_LOG(Error, "esp32: weight_tier_reserve must be a non-negative int "
                    "(bytes)");
      return Error::InvalidArgument;
    }
    LifecycleGuard guard;
    g_weight_tier_reserve = (size_t)*i;
    return Error::Ok;
  }
  if (std::strcmp(option.key, kOptWeightTierTables) == 0 ||
      std::strcmp(option.key, kOptWeightCache) == 0) {
    if (b == nullptr) {
      ET_LOG(Error, "esp32: %s must be a bool", option.key);
      return Error::InvalidArgument;
    }
    LifecycleGuard guard;
    if (std::strcmp(option.key, kOptWeightTierTables) == 0) {
      g_weight_tier_tables = *b;
    } else {
      g_weight_cache = *b;
      if (!g_weight_cache) release_weight_cache();
    }
    return Error::Ok;
  }
  if (std::strcmp(option.key, kOptWeightTierRank) == 0) {
    if (b == nullptr && (i == nullptr || (*i != 0 && *i != 1))) {
      ET_LOG(Error, "esp32: weight_tier_rank must be a bool, or 0 or 1");
      return Error::InvalidArgument;
    }
    LifecycleGuard guard;
    g_weight_tier_rank = b != nullptr ? *b : *i == 1;
    return Error::Ok;
  }
  if (std::strcmp(option.key, kOptWeightTierApply) == 0) {
    if (b == nullptr) {
      ET_LOG(Error, "esp32: weight_tier_apply must be a bool");
      return Error::InvalidArgument;
    }
    if (*b) {
      weight_tier_apply(false);
    }
    return Error::Ok;
  }
  if (std::strcmp(option.key, kOptWeightTierWanted) == 0 ||
      std::strcmp(option.key, kOptWeightTierResident) == 0) {
    ET_LOG(Error, "esp32: %s is read-only", option.key);
    return Error::InvalidArgument;
  }
  *handled = false;
  return Error::Ok;
}

/// get_option's weight-tier keys. *handled says whether `option` was one.
static void get_weight_tier_option(BackendOption& option, bool* handled) {
  *handled = true;
  LifecycleGuard guard;
  if (std::strcmp(option.key, kOptWeightTierBudget) == 0) {
    option.value = g_weight_tier_auto ? kWeightTierAuto
                                      : option_bytes(g_weight_tier_budget);
  } else if (std::strcmp(option.key, kOptWeightTierReserve) == 0) {
    option.value = option_bytes(g_weight_tier_reserve);
  } else if (std::strcmp(option.key, kOptWeightTierTables) == 0) {
    option.value = g_weight_tier_tables;
  } else if (std::strcmp(option.key, kOptWeightCache) == 0) {
    option.value = g_weight_cache;
  } else if (std::strcmp(option.key, kOptWeightTierWanted) == 0) {
    option.value = option_bytes(weight_tier_wanted());
  } else if (std::strcmp(option.key, kOptWeightTierResident) == 0) {
    option.value = option_bytes(g_weight_cache_bytes);
  } else if (std::strcmp(option.key, kOptWeightTierRank) == 0) {
    option.value = g_weight_tier_rank;
  } else if (std::strcmp(option.key, kOptWeightTierApply) == 0) {
    option.value = weight_tier_waiting() > 0;
  } else {
    *handled = false;
  }
}

class Esp32Backend final : public BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  /// ExecuTorch's runtime-reconfiguration hook. Its options:
  ///
  ///     "min_latency"         int   -1  each program's compile spec decides
  ///                                     (default)
  ///                                  0  force background mode: one core, no
  ///                                     worker
  ///                                  1  force min-latency mode: dual-core
  ///                                     column split
  ///                           bool  false -> 0, true -> 1
  ///     "weight_tier_budget"  int   bytes of PSRAM for copied weights, or -1
  ///                                 for auto
  ///     "weight_tier_reserve" int   bytes of PSRAM auto leaves free
  ///     "weight_tier_tables"  bool  copy tables and selected-row heads too
  ///     "weight_tier_rank"    bool  ranked placement instead of load order
  ///     "weight_tier_apply"   bool  true: make the ranked placement now
  ///     "weight_cache"        bool  keep copies across method destroy
  ///     "weight_tier_wanted", "weight_tier_resident"  int, read-only
  ///
  /// The weight-tier options are described with kOptWeightTierBudget, and
  /// apply to the loads after they are set (weight_tier_apply, to the loads
  /// made before it). The min_latency key is
  /// deliberately the SAME STRING as the AOT compile spec (kSpecMinLatency):
  /// one name for one decision, whether it is made at export or at run time, so
  /// there is nothing to map between the two halves.
  ///
  /// Why these knobs and not the others: each is a property of the DEPLOYMENT
  /// rather than of the model. Whether the second core belongs to inference or
  /// to the application is a decision the application makes, can change (a
  /// device that batches at night and serves interactively by day), and should
  /// not require a re-export to change; so is how much PSRAM it can spare.
  /// force_scalar, by contrast, exists to A/B the kernels and stays a test hook.
  ///
  /// Not per-handle: there is one worker because there is one spare core, so the
  /// posture is a property of the process. A program built with the compile spec
  /// set still gets its own default; the override only overrides.
  Error set_option(
      BackendOptionContext& context,
      const Span<BackendOption>& backend_options) override {
    (void)context;
    for (const auto& option : backend_options) {
      bool tier = false;
      if (const Error err = set_weight_tier_option(option, &tier);
          tier || err != Error::Ok) {
        if (err != Error::Ok) {
          return err;
        }
        continue;
      }
      if (std::strcmp(option.key, kSpecMinLatency) != 0) {
        // Deliberate deviation from XNNPACK, which returns Ok for keys it does
        // not recognise. Options are routed per backend id
        // (LoadBackendOptionsMap), so a key arriving here was addressed to THIS
        // backend and an unknown one is a caller mistake -- accepting it
        // silently means a user believes min-latency is on when they typo'd.
        ET_LOG(Error, "esp32: unknown backend option '%s'", option.key);
        return Error::NotFound;
      }
      int want;
      if (const bool* b = std::get_if<bool>(&option.value)) {
        // The ergonomic spelling: BackendOptions::set_option("min_latency",
        // true). No way to say "follow the compile spec" in a bool, which is
        // what the int form is for.
        want = *b ? 1 : 0;
      } else if (const int* i = std::get_if<int>(&option.value)) {
        want = *i;
      } else {
        ET_LOG(Error, "esp32: min_latency must be a bool or an int");
        return Error::InvalidArgument;
      }
      const Error err = apply_split_override(want);
      if (err != Error::Ok) {
        return err;
      }
    }
    return Error::Ok;
  }

  Error get_option(
      BackendOptionContext& context,
      Span<BackendOption>& backend_options) override {
    (void)context;
    for (auto& option : backend_options) {
      bool tier = false;
      get_weight_tier_option(option, &tier);
      if (tier) {
        continue;
      }
      if (std::strcmp(option.key, kSpecMinLatency) != 0) {
        ET_LOG(Error, "esp32: unknown backend option '%s'", option.key);
        return Error::NotFound;
      }
      // Reports the OVERRIDE, not an effective value, and reports it as an int
      // including -1. There is no single effective value to report: min-latency
      // resolves against each handle's compile spec inside execute(), and
      // get_option has no handle. -1 therefore reads back as "whatever each
      // program asked for" -- which is exactly what was set, and collapsing it
      // to a bool would report a posture no program is necessarily in.
      option.value = g_split_override.load(std::memory_order_relaxed);
    }
    return Error::Ok;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    const bool min_latency = compile_spec_min_latency(compile_specs);
    bool report_selftests = false;
    {
      // Once per process; two first inits may race, and the self-test runs
      // on static buffers.
      LifecycleGuard guard;
      static bool unpack_checked = false;
      if (!unpack_checked) {
        g_unpack_pie_ok.store(unpack_selftest(), std::memory_order_relaxed);
        g_dot_i4_pie_ok.store(dot_i4_selftest(), std::memory_order_relaxed);
        unpack_checked = true;
        report_selftests = true;
      }
    }
    // Outside the lifecycle lock, like every backend print: see
    // stage_weights.
    if (report_selftests) {
#if ESP32_BACKEND_STATS
      printf("esp32: PIE unpack selftest: %s\n",
             g_unpack_pie_ok.load(std::memory_order_relaxed)
                 ? "PASS" : "FAIL (scalar unpack in use)");
      printf("esp32: PIE fused int4 dot selftest: %s\n",
             g_dot_i4_pie_ok.load(std::memory_order_relaxed)
                 ? "PASS" : "FAIL (unpack + dot path in use)");
#else
      // Outputs stay exact on the fallback; only a failure is worth a line.
      if (!g_unpack_pie_ok.load(std::memory_order_relaxed) ||
          !g_dot_i4_pie_ok.load(std::memory_order_relaxed)) {
        ET_LOG(Error, "esp32: a PIE int4 self-test failed; scalar fallback in use");
      }
#endif
    }
    const uint8_t* data = static_cast<const uint8_t*>(processed->data());
    const size_t size = processed->size();
    uint32_t n_ops = 0, ext_off = 0;
    if (Error e = read_blob_header(data, size, &n_ops, &ext_off,
                                   compile_spec_stage_b(compile_specs));
        e != Error::Ok) {
      return e;
    }

    // Allocated from the method allocator, so it lives as long as the method
    // and needs no destroy() hook of its own; destroy() releases what the
    // stages below fetch or stage.
    auto* blob = context.get_runtime_allocator()
                     ->allocateInstance<CompiledBlob>();
    if (blob == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    clear_blob(blob, data, n_ops, min_latency);
    blob->resolved = context.get_runtime_allocator()
                         ->allocateList<Resolved>(n_ops);
    // ExecuTorch does not call destroy() after a failed init; the guard does.
    InitHeldGuard held_guard(blob);
    if (blob->resolved == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    // Asymmetric records into blob->asym (nothing for a symmetric blob), before
    // any weight is fetched or staged for a blob that may be refused.
    if (Error e = esp32_asym::attach(context, data, size, n_ops, ext_off,
                                     &blob->asym);
        e != Error::Ok) {
      return e;
    }
    if (Error e = resolve_weights(context, blob); e != Error::Ok) {
      return e;
    }
    blob->cfg_head_split = any_op_has_flag(blob, kFlagHeadSplit);
    // A program that asks for the second core -- the min_latency compile spec,
    // or a kFlagHeadSplit op -- gets its worker now, so a failure to create it
    // fails load_method, where the caller can still act on it, instead of
    // every execute() quietly running single-core. set_option(min_latency)
    // creates it eagerly for the same reason. Forced background (override 0)
    // means no worker, so none is made.
    if ((blob->cfg_min_latency || blob->cfg_head_split) &&
        g_split_override.load(std::memory_order_relaxed) != 0 &&
        !ensure_worker()) {
      ET_LOG(Error, "esp32: program requests the second core, but its split "
                    "worker task could not be created");
      return Error::MemoryAllocationFailed;
    }
    // The extension records before staging: a candidate's score reads its
    // conv geometry and its selected rows.
    uint32_t conv_ops = 0;
    if (Error e = parse_extension_records(context, blob, ext_off, &conv_ops);
        e != Error::Ok) {
      return e;
    }
    if (Error e = check_grouped_conv_sequences(blob); e != Error::Ok) {
      return e;
    }
    if (Error e = stage_weights(context, blob); e != Error::Ok) {
      return e;
    }
    ScratchLayout layout;  // in scratch_layout.py's order: planned offsets depend on it
    reserve_row_block_scratch(blob, &layout, conv_ops);
    if (Error e = reserve_chain_scratch(blob, &layout); e != Error::Ok) {
      return e;
    }
    if (Error e = reserve_bmm_scratch(blob, &layout); e != Error::Ok) {
      return e;
    }
    reserve_affine_scratch(blob, &layout);
    if (layout.overflow) {
      ET_LOG(Error, "esp32: delegate temporary scratch exceeds 32-bit arena");
      return Error::InvalidProgram;
    }
    blob->scratch_bytes = layout.bytes;
    if (Error e = build_activation_groups(context, blob); e != Error::Ok) {
      return e;
    }
#if ESP32_BACKEND_STATS
    print_init_report(blob, size);
#endif
    held_guard.disarm();
    return static_cast<DelegateHandle*>(blob);
  }


  // Whether this call may use the second core at all: every tensor its ops
  // write and every piece of its scratch in internal RAM. On this chip a line
  // of PSRAM written while both cores run can lose bytes (README, Known
  // limits): a projection's column split writes each output row from both
  // cores, and a split conv its two parts. A call whose writes land in PSRAM
  // runs on one core, with the same bits. What it only reads (its inputs and
  // weights) may be anywhere.
  static bool split_writes_internal(const CompiledBlob& call,
                                    Span<EValue*> args) {
    for (uint32_t i = 0; i < call.n_ops; ++i) {
      const uint32_t at = call.ops[i].out_arg;  // a chain slot: scratch, below
      if (at >= args.size() || args[at] == nullptr || !args[at]->isTensor()) {
        continue;
      }
      const auto t = args[at]->toTensor();
      if (t.nbytes() > 0 && !esp32_internal_ram(t.const_data_ptr())) {
        return false;
      }
    }
    const void* scratch[] = {call.xq,  call.conv_stage, call.wu,
                             call.wu2, call.xq16,       call.sxv,
                             call.bmm_kq, call.bmm_sk, call.asym_terms};
    for (const void* p : scratch) {
      if (p != nullptr && !esp32_internal_ram(p)) {
        return false;
      }
    }
    for (uint32_t s = 0; s < kNumScratch; ++s) {
      if (call.chain[s] != nullptr && !esp32_internal_ram(call.chain[s])) {
        return false;
      }
    }
    return true;
  }
  // The calls split_writes_internal kept on one core (esp32_backend_report).
  static inline Stat<uint32_t> split_psram_calls{0};

  void destroy(DelegateHandle* handle) const override {
    release_resources(static_cast<CompiledBlob*>(handle));
  }

  ESP32_HOT_CODE
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    // Firmware hooks bracket the complete BackendInterface call, including
    // handle-copy and temporary-scratch setup. Keeping the marker at the true
    // boundary lets a profiler subtract backend work from
    // Method::execute without silently charging backend prelude to ET.
    ExecuteMarker _exec_marker;
    const auto* stored = static_cast<const CompiledBlob*>(handle);
    // A ranked load nobody applied: its first execute makes the placement.
    // Only a load that deferred candidates can be waiting (tier_deferred),
    // so every other handle skips the link and its acquire. A handle that
    // finds its placement made drops tier_deferred: a placement only ever
    // clears the flag, so no later execute of it needs the link, and from
    // then on it reads its copies as a load-order handle does (a placement
    // marks its slot; its own execute repoints its ops, weight_tier_sync).
    // Only this handle's own execute writes the field after init, and
    // ExecuTorch never runs one handle's executes concurrently.
    if (stored->tier_deferred) {
      if (__atomic_load_n(&stored->tier->pending, __ATOMIC_ACQUIRE) != 0) {
        weight_tier_apply(true);
      } else {
        static_cast<CompiledBlob*>(handle)->tier_deferred = false;
      }
    }
    // A placement since this handle's last call (its slot marked): its ops
    // move to the copies now, in place -- the call copy below shares
    // resolved[] -- before any of them, or the split worker, reads a pointer.
    if (tier_dirty_now(stored->tier_slot)) {
      weight_tier_sync(const_cast<CompiledBlob*>(stored));
    }
    CompiledBlob call = *stored;
    if (Error e = bind_call_scratch(context, *stored, &call); e != Error::Ok) {
      return e;
    }
    ++g_delegate_calls;

    // The whole-call event. Named for the backend rather than the method so a
    // devtools timeline groups every delegate call together, which is the view
    // that answers "how much of inference is in this backend at all".
    auto* tracer = context.event_tracer();
    (void)tracer;
    ESP32_PROF_SCOPE(tracer, "Esp32Backend::execute()");
#if defined(ESP32_BACKEND_TIMING)
    const uint64_t t_exec0 = ESP32_TICK_NOW();
#endif

    CallContext c = resolve_call_posture(&call, args);
    // Lease the shared worker for the whole call. try-take, never block: a
    // second concurrent execute runs single-core rather than waiting, which
    // is bit-identical output for a latency cost that only the loser pays.
    SplitLease lease;
    if ((c.global_split || c.selective_split) && caller_off_worker_core() &&
        ensure_worker()) {
      lease.held = xSemaphoreTake(g_split_lock, 0) == pdTRUE;
      if (lease.held) {
        match_worker_priority();
      }
    }
    c.split_enabled = lease.held;
    if (c.split_enabled && !split_writes_internal(call, args)) {
      c.split_enabled = false;  // the lease stays held; nothing in this call splits
      split_psram_calls.fetch_add(1, std::memory_order_relaxed);
    }
    // Ops run in descriptor order: a chain slot holds what an earlier op wrote.
    for (uint32_t i = 0; i < call.n_ops; ++i) {
      const OpDesc& op = call.ops[i];
      // Activation-quant sharing: a grouped op executes when its group's
      // FIRST member is reached; the later members were computed then.
      const int16_t grp = call.op_group[i];
      if (grp >= 0 &&
          call.group_members[call.group_first[grp]] != (uint16_t)i) {
        continue;
      }
      // Declared AFTER the group skip: a member folded into its group leader
      // does no work here, and timing it would report a fleet of ~0 us ops
      // that never ran while under-reporting the leader that did their work.
      OpScope _op_scope(op.opcode, tracer);
#if defined(ESP32_BACKEND_TIMING)
      {
        uint32_t members = 1, n_total = op.n;
        uint64_t bytes = is_embed(op.opcode) ? 0
                         : (uint64_t)op.n * execution_row_bytes(op);
        if (grp >= 0) {
          members = 0;
          n_total = 0;
          bytes = 0;
          for (uint32_t j = 0; j < call.n_ops; ++j) {
            if (call.op_group[j] != grp) continue;
            ++members;
            n_total += call.ops[j].n;
            bytes += (uint64_t)call.ops[j].n * execution_row_bytes(call.ops[j]);
          }
        }
        _op_scope.shape(op.m, op.k, n_total, members, bytes);
      }
#endif
      if (Error e = check_op_args(c, i); e != Error::Ok) {
        return e;
      }
      Error e;
      if (esp32_float::is_embed(op.opcode)) {
        e = run_embedding_float(c, i);
      } else if (is_embed(op.opcode)) {
        e = esp32_asym::weight_affine(call.asym, i) ? run_embedding_affine(c, i)
                                                     : run_embedding(c, i);
      } else if (op.opcode == kOpBmmDynI8) {
        e = run_bmm_dyn_i8(c, i);
      } else if (op.opcode == kOpBmmKcacheI8) {
        e = run_bmm_kcache_i8(c, i);
      } else if (op.opcode == kOpBmmKcacheAffineI8) {
        // The affine K cache is esp32_asym's whole: it reads its three side
        // arguments by index and checks them there. Its query row and staged
        // key rows are the call's, as run_bmm_kcache_i8's are.
        esp32_asym::OpDescView view;
        std::memcpy(&view, &op, sizeof(view));
        e = esp32_asym::run_kcache_affine(i, view, c.args, call.xq,
                                          call.bmm_kq, c.force_scalar);
      } else {
        MatmulIo io;
        e = resolve_matmul_io(c, i, &io);
        if (e != Error::Ok) {
          return e;
        }
        if (op.opcode == kOpAddmm && call.conv[i] == nullptr) {
          e = run_addmm_fp32(c, i, io);
        } else if (op.opcode == kOpAddmm ||
                   esp32_float::is_matmul(op.opcode)) {
          e = run_float_weights(c, i, io);
        } else if (op.opcode == kOpAddmmI16) {
          e = run_addmm_i16(c, i, io);
        } else {
          e = run_addmm_quantised(c, i, io, grp);
        }
      }
      if (e != Error::Ok) {
        return e;
      }
    }
#if defined(ESP32_BACKEND_TIMING)
    g_execute_us += ESP32_TICK_NOW() - t_exec0;
#endif
    return Error::Ok;
  }
};

Esp32Backend g_esp32_backend;
Backend g_backend_entry{"Esp32Backend", &g_esp32_backend};

// Registered at static-init time, before app_main runs.
const auto g_registered = register_backend(g_backend_entry);

} // namespace

// ---- C hooks for the device test runner and measurement builds ------------
// Not the product API: an application configures this backend through
// Esp32Backend::set_option() / get_option(). These exist because the device
// harness is plain C with no BackendOptions in scope; the counters they read
// are zero in an ESP32_BACKEND_STATS=0 build.

// Process-wide OVERRIDES, for measurement. Deliberately global: their purpose
// is A/Bing one binary against itself, which a per-handle knob cannot express.
// NOTE the asymmetry -- only min_latency is genuinely tri-state, because only
// it has a per-handle compile spec to defer to. There is no force-scalar spec,
// so for THAT knob <0 and 0 both mean "use PIE where alignment allows".
extern "C" void esp32_backend_set_force_scalar(int on) {
  g_force_scalar_override.store(on, std::memory_order_relaxed);
}

// The min-latency mode switch. Off (the default) is BACKGROUND inference: no
// worker exists and the second core belongs entirely to the application. On
// spawns the worker (once) and splits wide quantised addmm columns across both
// cores. Runtime rather than build-time for the same reason as force_scalar:
// hardware A/Bs one binary, and the parity harness proves both paths.
//
// A thin shim over apply_split_override(), which is also what
// Esp32Backend::set_option("min_latency") calls. The ExecuTorch option API is
// the sanctioned path and is what an application should use; this C entry point
// exists because the parity harness drives it from plain C code with no
// BackendOptions object in scope, and because a caller with no Error to return
// is a genuinely different (and lossier) contract worth keeping visible.
extern "C" void esp32_backend_set_split(int on) {
  (void)apply_split_override(on);
}

#if ESP32_CONV_FAST
// The static int8 convolution path (esp32_conv.h): on by default; 0 sends
// every conv to the general loops, which compute the same bits, so a host or
// board run A/Bs the two on one binary.
extern "C" void esp32_backend_set_conv_fast(int on) {
  g_conv_fast_override.store(on, std::memory_order_relaxed);
}
#endif

// The highest extension type this runtime knows, so a test can synthesize an
// unknown one.
extern "C" unsigned esp32_ext_max_known(void) { return kExtMaxKnown; }

// One past the highest assigned opcode, so a test can synthesize an unknown
// one; a literal in the test would turn into a real opcode when one is added.
extern "C" unsigned esp32_num_opcodes(void) { return kNumOpcodes; }
extern "C" uint64_t esp32_backend_actq_rows_saved(void) {
  return g_actq_rows_saved.load(std::memory_order_relaxed);
}
extern "C" uint64_t esp32_backend_act_per_tensor_ops(void) {
  return g_act_per_tensor.load(std::memory_order_relaxed);
}

// The affine loops' op count and (split_jobs) the forked-op count, so a test can
// attribute either to one execute() by bracketing it, not by end-of-run totals.
extern "C" uint32_t esp32_backend_affine_ops(void) {
  return g_affine_ops.load(std::memory_order_relaxed);
}
extern "C" uint32_t esp32_backend_split_jobs(void) {
  return g_split_jobs;
}

extern "C" uint32_t esp32_backend_i8_fast_rows(void) {
  return g_i8_fast_rows.load(std::memory_order_relaxed);
}

// Prefill weight traffic: passes actually made vs what row-at-a-time dispatch
// would have made. Printed as a ratio because that is the claim -- bytes moved,
// which QEMU can count exactly, as opposed to time, which it cannot.
extern "C" void esp32_backend_report_blocking(void) {
  if (g_weight_passes_rowwise == 0) {
    return;
  }
  const uint64_t saved = g_weight_passes_rowwise - g_weight_passes;
  const uint64_t bsaved = g_wbytes_rowwise - g_wbytes_actual;
  printf("esp32 rowblock: %llu weight passes vs %llu row-at-a-time "
         "(%llu avoided); weight bytes %llu vs %llu (%llu%% less)\n",
         (unsigned long long)g_weight_passes,
         (unsigned long long)g_weight_passes_rowwise,
         (unsigned long long)saved,
         (unsigned long long)g_wbytes_actual,
         (unsigned long long)g_wbytes_rowwise,
         (unsigned long long)(g_wbytes_rowwise
                                  ? bsaved * 100 / g_wbytes_rowwise
                                  : 0));
}

// Frees every staged weight copy no loaded method is using.
extern "C" void esp32_backend_release_weight_cache(void) {
  LifecycleGuard guard;
  release_weight_cache();
}

extern "C" void esp32_backend_weight_cache_stats(uint32_t* copies, uint32_t* hits, size_t* resident) {
  LifecycleGuard guard;
  if (copies) *copies = g_weight_cache_copies;
  if (hits) *hits = g_weight_cache_hits;
  if (resident) *resident = g_weight_cache_bytes;
}

#if defined(ESP32_HOST_BUILD)
// The host runner's --tier-slots: at most `slots` handles get a private
// placement mark, every later one shares slot 255 and keeps its mark in its
// link, so that tests reach that path without 255 delegates. Host only.
extern "C" void esp32_backend_host_cap_tier_slots(unsigned slots) {
  LifecycleGuard guard;
  g_tier_slot_limit = 1u + (slots < kTierOverflowSlot - 1u
                                ? slots : kTierOverflowSlot - 1u);
}
#endif

extern "C" void esp32_backend_weight_tier_stats(Esp32WeightTierStats* stats) {
  if (stats == nullptr) {
    return;
  }
  LifecycleGuard guard;
  *stats = Esp32WeightTierStats{};
  for (uint32_t i = 0; i < g_weight_cache_n; ++i) {
    const CachedWeight& e = g_weight_cache_entries[i];
    if (weight_cache_unused(e)) {
      continue;
    }
    ++stats->entries;
    if (e.refs > 0) {
      (e.raw != nullptr ? stats->staged : stats->on_flash)[e.cls] += e.alloc;
      stats->wanted += e.alloc;
    }
  }
  stats->resident = g_weight_cache_bytes;
  stats->budget = g_weight_tier_budget;
  stats->reserve = g_weight_tier_reserve;
  stats->auto_budget = g_weight_tier_auto ? 1u : 0u;
  stats->copies = g_weight_cache_copies;
  stats->hits = g_weight_cache_hits;
  stats->evictions = g_weight_tier_evictions;
  stats->skipped = g_weight_tier_skipped;
  stats->expanded = g_weight_tier_expanded;
  stats->rank = g_weight_tier_rank ? 1u : 0u;
  stats->waiting = weight_tier_waiting();
  stats->placements = g_weight_tier_applies;
  stats->late_placements = g_weight_tier_late;
}

extern "C" void esp32_backend_weight_tier_copy_reads(uint32_t* copy_ops,
                                                     uint32_t* copy_reads) {
  uint32_t ops = 0, reads = 0;
  {
    LifecycleGuard guard;
    // Each loaded handle's ops whose weight has a copy, and whether they read
    // it (weights and scales): what a test asserts to see that every handle
    // repointed its ops after a placement (weight_tier_sync). A handle writes
    // its pointers only under this lock.
    for (const TierLink* t = g_tier_live; t != nullptr; t = t->next) {
      const CompiledBlob* blob = t->blob;
      for (uint32_t i = 0; i < blob->n_ops; ++i) {
        if (blob->staged_cache[i] == 0) {
          continue;
        }
        const CachedWeight& e =
            g_weight_cache_entries[blob->staged_cache[i] - 1u];
        if (e.raw == nullptr) {
          continue;
        }
        ++ops;
        if (blob->resolved[i].wq == e.w &&
            (e.s == nullptr || blob->resolved[i].scale == e.s)) {
          ++reads;
        }
      }
    }
  }
  if (copy_ops) *copy_ops = ops;
  if (copy_reads) *copy_reads = reads;
}

extern "C" void esp32_backend_report_weight_tiers(void) {
  static const char* const kClassNames[ESP32_TIER_CLASSES] = {
      "int4", "int8", "fp32", "bf16", "int16", "fp16"};
  Esp32WeightTierStats stats;
  esp32_backend_weight_tier_stats(&stats);
  if (stats.entries == 0) {
    return;
  }
  uint32_t copy_ops = 0, copy_reads = 0;
  esp32_backend_weight_tier_copy_reads(&copy_ops, &copy_reads);
  char budget[48];
  if (stats.auto_budget) {
    std::snprintf(budget, sizeof(budget), "auto keeping %llu B free",
                  (unsigned long long)stats.reserve);
  } else {
    std::snprintf(budget, sizeof(budget), "%llu B",
                  (unsigned long long)stats.budget);
  }
  printf("esp32 tiers: budget %s, resident %llu B, wanted %llu B, entries %u; "
         "copies %u, reuses %u, evictions %u, left on flash %u, expanded %u\n",
         budget, (unsigned long long)stats.resident,
         (unsigned long long)stats.wanted, (unsigned)stats.entries,
         (unsigned)stats.copies, (unsigned)stats.hits,
         (unsigned)stats.evictions, (unsigned)stats.skipped,
         (unsigned)stats.expanded);
  if (stats.rank != 0 || stats.placements != 0 || stats.waiting != 0) {
    printf("esp32 tiers rank: %s, placements %u (%u at a first execute), "
           "delegates waiting %u, ops reading copies %u of %u\n",
           stats.rank != 0 ? "on" : "off", (unsigned)stats.placements,
           (unsigned)stats.late_placements, (unsigned)stats.waiting,
           (unsigned)copy_reads, (unsigned)copy_ops);
  }
  for (int c = 0; c < ESP32_TIER_CLASSES; ++c) {
    if (stats.staged[c] != 0 || stats.on_flash[c] != 0) {
      printf("esp32 tiers %s: staged %llu B, on flash %llu B\n",
             kClassNames[c], (unsigned long long)stats.staged[c],
             (unsigned long long)stats.on_flash[c]);
    }
  }
  // A few entries at a time under the lock, each batch printed after it: a
  // backend print never runs under the lifecycle lock (see stage_weights).
  constexpr uint32_t kBatch = 4;
  for (uint32_t next = 0;;) {
    CachedWeight batch[kBatch];
    uint32_t got = 0;
    {
      LifecycleGuard guard;
      for (; next < g_weight_cache_n && got < kBatch; ++next) {
        if (!weight_cache_unused(g_weight_cache_entries[next])) {
          batch[got++] = g_weight_cache_entries[next];
        }
      }
    }
    if (got == 0) {
      break;
    }
    for (uint32_t t = 0; t < got; ++t) {
      const CachedWeight& e = batch[t];
      printf("esp32 tier: %s %s %s %llu B score %llu refs %u names %016llx\n",
             e.cls < ESP32_TIER_CLASSES ? kClassNames[e.cls] : "?",
             e.kind == kCopyExpandI8 ? "i4->i8" : "stored",
             e.raw != nullptr ? "psram" : "flash",
             (unsigned long long)e.alloc, (unsigned long long)e.score,
             (unsigned)e.refs, (unsigned long long)e.names);
    }
  }
}

/// Per-opcode wall time, the measurement build's whole point.
///
/// Prints nothing unless ESP32_BACKEND_TIMING is set, and says so rather than
/// staying silent -- a timing report that is quietly absent reads exactly like
/// a model that spent no time in the backend.
extern "C" void esp32_backend_report_timing(void) {
#if defined(ESP32_BACKEND_TIMING)
  uint64_t total = 0;
  for (uint32_t i = 0; i < kNumOpcodes; ++i) {
    total += g_op_us[i];
  }
  printf("esp32 timing: %llu us in execute(), %llu us in ops (%llu us = %.1f%% "
         "outside the op loop)\n",
         (unsigned long long)g_execute_us, (unsigned long long)total,
         (unsigned long long)(g_execute_us - total),
         g_execute_us ? 100.0 * (double)(g_execute_us - total) / (double)g_execute_us : 0.0);
  for (uint32_t i = 0; i < kNumOpcodes; ++i) {
    if (g_op_n[i] == 0) {
      continue;
    }
    printf("  %-20s %6u call(s)  %10llu us  %8.1f us/call  %5.1f%%\n",
           opcode_name(i), (unsigned)g_op_n[i],
           (unsigned long long)g_op_us[i],
           (double)g_op_us[i] / (double)g_op_n[i],
           total ? 100.0 * (double)g_op_us[i] / (double)total : 0.0);
  }
  printf("esp32 timing by shape (grouped ops count once, N summed over members):\n");
  for (int i = 0; i < kShapeSlots; ++i) {
    const ShapeTime& t = g_shape_time[i];
    if (t.calls == 0) break;
    printf("  %-16s m=%-3u k=%-4u n=%-5u x%u %6u call(s) %8.1f us/call %6.1f MB/s %5.1f%%\n",
           opcode_name(t.opcode), (unsigned)t.m, (unsigned)t.k, (unsigned)t.n,
           (unsigned)t.members, (unsigned)t.calls, (double)t.us / t.calls,
           t.us ? (double)t.bytes / (double)t.us : 0.0,
           total ? 100.0 * (double)t.us / (double)total : 0.0);
  }
#else
  printf("esp32 timing: not built (rebuild with -DESP32_BACKEND_TIMING=1)\n");
#endif
}

#if defined(ESP32_BACKEND_TIMING)
/// Zeroes what esp32_backend_report_timing() reports, so that its next report
/// covers only what runs after this: the bench reports each phase on its own.
extern "C" void esp32_backend_reset_timing(void) {
  memset(g_op_us, 0, sizeof(g_op_us));
  memset(g_op_n, 0, sizeof(g_op_n));
  memset(g_shape_time, 0, sizeof(g_shape_time));
  g_execute_us = 0;
}
#endif

extern "C" void esp32_backend_report(void) {
#if !ESP32_BACKEND_STATS
  // Zeros would read like a delegate that never ran.
  printf("esp32 backend: registered %d; statistics not built "
         "(ESP32_BACKEND_STATS=0)\n", (int)g_registered);
  return;
#endif
  printf("esp32 backend: registered %d, delegate calls %u, addmm ops %u, "
         "columns PIE %u / scalar %u, weights external %u / inline %u\n",
         (int)g_registered, (unsigned)g_delegate_calls, (unsigned)g_addmm_ops,
         (unsigned)g_pie_cols, (unsigned)g_scalar_cols,
         (unsigned)g_ext_keys, (unsigned)g_inline_wts);
  printf("esp32 scratch: %u call(s) on planned scratch, %u on the temp "
         "allocator (none planned), %u fallback(s) (planned scratch "
         "unusable)\n",
         (unsigned)g_scratch_planned, (unsigned)g_scratch_temp,
         (unsigned)g_scratch_unusable);
  printf("esp32 embed: %u gather op(s)\n", (unsigned)g_embed_ops);
  printf("esp32 actq: %llu shared-input row quantisations avoided\n",
         (unsigned long long)g_actq_rows_saved);
  if (g_act_per_tensor.load() != 0) {
    printf("esp32 actpertensor: %llu op(s) used the model's own per-tensor "
           "activation scale\n",
           (unsigned long long)g_act_per_tensor.load());
  }
  printf("esp32 actstatic: %llu rows quantised at calibrated scales\n",
         (unsigned long long)g_static_rows);
  printf("esp32 bmmqk: %u dynamic int8 bmm op(s)\n", (unsigned)g_bmm_ops);
  printf("esp32 bmmkc8: %u int8-K-cache bmm op(s), %llu cached key row(s) "
         "read without re-quantising\n",
         (unsigned)g_bmm_kcache_ops,
         (unsigned long long)g_bmm_kcache_rows.load());
  printf("esp32 split: %u column-split addmm row(s) forked\n",
         (unsigned)g_split_jobs);
  printf("esp32 split: %u call(s) kept on one core: they write PSRAM\n",
         (unsigned)Esp32Backend::split_psram_calls.load());
  printf("esp32 split: %u tied-head row(s); worker busy %llu us%s\n",
         (unsigned)g_head_split_jobs,
         (unsigned long long)g_split_worker_us,
#if defined(ESP32_BACKEND_TIMING)
         ""
#else
         " (timing disabled)"
#endif
  );
#if defined(ESP32_HOST_BUILD)
  // The host's split worker: a thread per job (esp32_platform.h).
  printf("esp32 split: %u job(s) on the host's worker thread\n",
         (unsigned)esp32_host_split_jobs());
#endif
  printf("esp32 stage: %u op(s), %llu B expanded in PSRAM\n",
         (unsigned)g_staged_ops, (unsigned long long)g_staged_bytes);
#if ESP32_CONV_FAST
  printf("esp32 convfast: %u conv op(s) on the static int8 path, %llu "
         "output(s)\n",
         (unsigned)g_conv_fast_ops, (unsigned long long)g_conv_fast_outputs);
  printf("esp32 convsplit: %u conv op(s) split across both cores\n",
         (unsigned)g_conv_split_ops);
  printf("esp32 convepi: %llu output(s) by the integer epilogue, %llu of "
         "them by its float fallback\n",
         (unsigned long long)g_conv_epi_fixed,
         (unsigned long long)g_conv_epi_fallback);
#endif
  if (g_weight_bytes > 0) {
    // Integer-only: no float formatting, and this runs on a device.
    const uint64_t intensity_x100 = g_macs * 100u / g_weight_bytes;
    printf("esp32 work: %llu MACs, %llu weight bytes read, intensity %llu.%02llu "
           "MAC/byte\n",
           (unsigned long long)g_macs, (unsigned long long)g_weight_bytes,
           (unsigned long long)(intensity_x100 / 100),
           (unsigned long long)(intensity_x100 % 100));
    // Compute and memory balance when MAC rate == intensity * bandwidth. Below that
    // the matvec is memory-bound and a faster kernel buys nothing, which is the
    // measurement to take first on real hardware.
    printf("esp32 work: compute-bound only if the MAC unit sustains under "
           "%llu.%02llu x the weight-stream bandwidth\n",
           (unsigned long long)(intensity_x100 / 100),
           (unsigned long long)(intensity_x100 % 100));
  }
}

// ---- the non-finite check's entry into the backend's own quantisers ---------
// test/runtime/nonfinite_check.cpp holds these to quant_rules.h's rule on NaN
// and the infinities: quantize_row_i8 and quantize_conv_row themselves, and
// through them round_half_even_i, the rounding of a folded per-tensor
// activation (kFlagActPerTensor), not a copy of them. Test code calls them
// (the host's --selftest, the parity run); the story never does, and its link
// drops them.
extern "C" float esp32_backend_check_quantize_row_i8(const float* x, uint32_t k,
                                                     uint32_t k_pad, int8_t* dst,
                                                     float fixed_sx,
                                                     int ties_to_even) {
  return quantize_row_i8(x, k, k_pad, dst, fixed_sx, ties_to_even != 0);
}

/// quantize_conv_row on one channel of n values padded by `pad` on each side:
/// a 1-D conv whose one kernel spans them all (kw = n + 2 * pad, w_out = 1),
/// so its row 0 is pad zeros, the n values in order, and pad zeros.
extern "C" float esp32_backend_check_quantize_conv_row(const float* x,
                                                       uint32_t n, uint32_t pad,
                                                       uint32_t k_pad,
                                                       int8_t* dst,
                                                       float fixed_sx,
                                                       int ties_to_even) {
  ConvGeom g{};
  g.h_in = g.h_out = g.kh = 1;
  g.w_in = n;
  g.w_out = 1;
  g.kw = n + 2 * pad;
  g.c_in = g.c_out = g.c_out_total = g.groups = 1;
  g.stride_h = g.stride_w = g.dil_h = g.dil_w = 1;
  g.pad_w = pad;
  g.d_in = g.d_out = g.kd = g.stride_d = g.dil_d = 1;
  return quantize_conv_row(x, g, 0, k_pad, dst, fixed_sx, ties_to_even != 0);
}

#if ESP32_CONV_FAST
// ---- the bench image's micro-benchmarks --------------------------------------
// esp32_backend_microbench() times, in core cycles (CCOUNT): dot_i8_s3 a call
// at K = 16, 32, 64, 256 and 768; the conv path's epilogues an output
// (esp32_conv::bench_epilogue); a fork and join through the split worker with
// nothing to do (the projection split's hand-over); and a DS-CNN pointwise
// and depthwise layer on the static int8 path, on one core as every conv
// runs (kConvSplit), with PIE and without (the scalar dots and sums: the PIE
// kernels' self-check on the chip, the two runs' outputs compared byte by
// byte, and their gain). The bench firmware runs it at boot
// (ET_BENCH_MICRO) and prints one "BENCH_MICRO" line a measurement. Every
// figure is the least of three trials. On the host the "cycles" are
// microseconds x 240, a smoke test only.
namespace {

uint32_t bench_cycles(void) {
#if defined(__XTENSA__)
  uint32_t c;
  __asm__ __volatile__("rsr.ccount %0" : "=a"(c));
  return c;
#else
  return (uint32_t)(esp_timer_get_time() * 240);
#endif
}

/// A synthetic conv layer on the path, in internal memory: codes in and out
/// (Stage B), int8 weights with fp32 scales, int32 S' and a bias, a ReLU
/// clamp: DS-CNN's pointwise (25 x 5, 64 -> 64) or depthwise (3 x 3, pad 1)
/// layer.
struct BenchLayer {
  esp32_conv::Op op;
  esp32_conv::Act act;
  esp32_conv::Columns cols;
  esp32_conv::Out out;
  int8_t* mem;
  int8_t *x8, *wq, *y8, *s0;
  float* scales;
  float* bias;
  int32_t* sums;
  uint32_t rows;

  bool make(bool depthwise) {
    memset(this, 0, sizeof(*this));
    op.h_in = 25;
    op.w_in = 5;
    op.c_in = depthwise ? 1 : 64;
    op.h_out = 25;
    op.w_out = 5;
    op.kh = op.kw = depthwise ? 3 : 1;
    op.stride_h = op.stride_w = op.dil_h = op.dil_w = 1;
    op.pad_h = op.pad_w = depthwise ? 1 : 0;
    op.depthwise = depthwise;
    op.m = 125;
    op.n = 64;
    op.k = depthwise ? 9 : 64;
    op.k_pad = depthwise ? 16 : 64;
    op.clamp = true;
    op.clamp_lo = 0.0f;
    op.clamp_hi = INFINITY;
    rows = depthwise ? 0 : 64;
    const uint64_t scratch = esp32_conv::scratch_need(op, false, rows) + 16;
    const size_t bytes = 64 * 125 + 64 * op.k_pad + 64 * 125 + scratch +
                         64 * 12 + 64;
    mem = static_cast<int8_t*>(heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (mem == nullptr) {
      return false;
    }
    int8_t* p = reinterpret_cast<int8_t*>(((uintptr_t)mem + 15) & ~(uintptr_t)15);
    x8 = p;
    p += 64 * 125;
    p = reinterpret_cast<int8_t*>(((uintptr_t)p + 15) & ~(uintptr_t)15);
    wq = p;
    p += 64 * op.k_pad;
    y8 = p;
    p += 64 * 125;
    p = reinterpret_cast<int8_t*>(((uintptr_t)p + 15) & ~(uintptr_t)15);
    s0 = p;
    p += scratch;
    p = reinterpret_cast<int8_t*>(((uintptr_t)p + 3) & ~(uintptr_t)3);
    scales = reinterpret_cast<float*>(p);
    bias = scales + 64;
    sums = reinterpret_cast<int32_t*>(bias + 64);
    uint32_t r = 2463534242u;
    auto next = [&r]() {
      r ^= r << 13;
      r ^= r >> 17;
      r ^= r << 5;
      return r;
    };
    for (uint32_t i = 0; i < 64u * 125u; ++i) {
      x8[i] = (int8_t)(-128 + (int32_t)(next() % 160));  // ReLU'd codes
    }
    for (uint32_t c = 0; c < 64; ++c) {
      int32_t s = 0;
      for (uint32_t j = 0; j < op.k_pad; ++j) {
        const int8_t w = j < op.k ? (int8_t)((int32_t)(next() % 255) - 127) : 0;
        wq[c * op.k_pad + j] = w;
        s += w;
      }
      sums[c] = s;
      scales[c] = 0.002f + 0.00003f * (float)c;
      bias[c] = -0.4f + 0.01f * (float)c;
    }
    act.x8 = x8;
    act.qmin = -128;
    act.qmax = 127;
    act.sx = 0.0234f;
    act.zx = -128;
    cols.wq = wq;
    cols.pie = true;
    cols.scales = scales;
    cols.scales_fp16 = false;
    cols.sums = sums;
    cols.sums_i16 = false;
    cols.bias = bias;
    out.y8 = y8;
    out.requant = true;
    out.out_scale = 0.0517f;
    out.out_inv = 1.0f / out.out_scale;
    out.zo = -128;
    out.qmin = -128;
    out.qmax = 127;
    return true;
  }

  /// Cycles of one run of the layer, on one core (every conv runs so).
  uint32_t run() {
    esp32_conv::Counts counts{0, 0, 0};
    esp32_conv::Scratch sc{s0, rows, nullptr};
    const uint32_t c0 = bench_cycles();
    sc.plane = esp32_conv::prepare(op, act, sc);
    esp32_conv::run_part(op, act, cols, out, sc,
                         esp32_conv::Part{0, esp32_conv::units(op)}, &counts);
    return bench_cycles() - c0;
  }
};

uint32_t least3(uint32_t a, uint32_t b, uint32_t c) {
  return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

/// a / b to two decimals, for "x1.23": the whole part and the hundredths.
struct Ratio {
  unsigned whole, hundredths;
};
Ratio ratio(uint32_t a, uint32_t b) {
  b = b ? b : 1;
  return Ratio{(unsigned)(a / b), (unsigned)((uint64_t)a * 100 / b % 100)};
}

}  // namespace

extern "C" void esp32_backend_microbench(void) {
  printf("BENCH_MICRO start (core cycles, least of 3)\n");
  // The PIE depthwise kernel's lane check (once a boot), so that the depthwise
  // layer below runs on it.
  esp32_conv::pie_dw_init();
  // dot_i8_s3 a call, the PIE kernel the dense convs run per output.
  int8_t* buf = static_cast<int8_t*>(heap_caps_malloc(2 * 768 + 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (buf != nullptr) {
    int8_t* a = reinterpret_cast<int8_t*>(((uintptr_t)buf + 15) & ~(uintptr_t)15);
    int8_t* b = a + 768;
    for (uint32_t i = 0; i < 768; ++i) {
      a[i] = (int8_t)(i * 37);
      b[i] = (int8_t)(i * 11 + 3);
    }
    const uint32_t ks[] = {16, 32, 64, 256, 768};
    for (uint32_t k : ks) {
      uint32_t best = UINT32_MAX;
      volatile int32_t sink = 0;
      for (uint32_t trial = 0; trial < 3; ++trial) {
        const uint32_t c0 = bench_cycles();
        for (uint32_t i = 0; i < 256; ++i) {
          sink = sink + dot_i8_s3(a, b, (int)k);
        }
        const uint32_t dt = bench_cycles() - c0;
        best = dt < best ? dt : best;
      }
      printf("BENCH_MICRO dot_i8_s3 k=%u: %u.%02u cycles a call\n", (unsigned)k,
             (unsigned)(best / 256), (unsigned)(best % 256 * 100 / 256));
    }
    heap_caps_free(buf);
  }
  // The epilogues an output.
  const esp32_conv::EpilogueBench eb = esp32_conv::bench_epilogue(bench_cycles, 64);
  printf("BENCH_MICRO epilogue: float %u.%02u, integer %u.%02u cycles an output; "
         "%u fallback(s) a run, %u mismatch(es)\n",
         (unsigned)(eb.cycles_float / 1000), (unsigned)(eb.cycles_float % 1000 / 10),
         (unsigned)(eb.cycles_fixed / 1000), (unsigned)(eb.cycles_fixed % 1000 / 10),
         (unsigned)eb.fallbacks, (unsigned)eb.mismatches);
  // A fork round trip, and the layers, need the worker (min_latency spawns
  // it; a single-core image gets one here).
  if (!ensure_worker()) {
    printf("BENCH_MICRO no split worker\n");
    return;
  }
  match_worker_priority();
  {
    esp32_conv::Op none;
    memset(&none, 0, sizeof(none));
    none.depthwise = true;  // an empty part: times the hand-over alone, not a conv split
    esp32_conv::Act act;
    memset(&act, 0, sizeof(act));
    esp32_conv::Columns cols;
    memset(&cols, 0, sizeof(cols));
    esp32_conv::Out out;
    memset(&out, 0, sizeof(out));
    uint32_t best = UINT32_MAX;
    for (uint32_t trial = 0; trial < 3; ++trial) {
      const uint32_t c0 = bench_cycles();
      for (uint32_t i = 0; i < 64; ++i) {
        ConvSplit job{&none, &act, &cols, &out, esp32_conv::Scratch{nullptr, 0, nullptr},
                      esp32_conv::Part{0, 0}, esp32_conv::Counts{0, 0, 0}};
        fork_conv(&job);
        join_conv(job, nullptr);
      }
      const uint32_t dt = bench_cycles() - c0;
      best = dt < best ? dt : best;
    }
    printf("BENCH_MICRO fork round trip: %u cycles (an empty part, notify and "
           "join)\n", (unsigned)(best / 64));
  }
  for (int dw = 0; dw < 2; ++dw) {
    BenchLayer layer;
    if (!layer.make(dw != 0)) {
      printf("BENCH_MICRO no memory for the layer\n");
      continue;
    }
    layer.run();  // warm
    const uint32_t one = least3(layer.run(), layer.run(), layer.run());
    uint8_t* first = static_cast<uint8_t*>(heap_caps_malloc(64 * 125, MALLOC_CAP_8BIT));
    if (first != nullptr) {
      memcpy(first, layer.y8, 64 * 125);
    }
    const char* name = dw ? "depthwise 3x3" : "pointwise 64->64";
    printf("BENCH_MICRO %s 25x5x64 layer: one core %u cycles (%u.%02u an output)\n",
           name, (unsigned)one, (unsigned)(one / 8000),
           (unsigned)(one % 8000 * 100 / 8000));
    // The same layer without PIE: the scalar dots and depthwise sums. The
    // board's self-check of the PIE kernels (dw33_qacc16_s3, dot_i8_rows_s3)
    // against the scalar path, and their gain on one core (the depthwise
    // layer's scalar time is the path's before PIE).
    layer.cols.pie = false;
    const uint32_t scalar = least3(layer.run(), layer.run(), layer.run());
    const bool same = first != nullptr && memcmp(first, layer.y8, 64 * 125) == 0;
    const Ratio pie = ratio(scalar, one);
    printf("BENCH_MICRO %s 25x5x64 layer without PIE: one core %u cycles "
           "(%u.%02u an output; PIE x%u.%02u), outputs %s PIE's\n",
           name, (unsigned)scalar, (unsigned)(scalar / 8000),
           (unsigned)(scalar % 8000 * 100 / 8000), pie.whole, pie.hundredths,
           same ? "identical to" : "DIFFER from");
    heap_caps_free(first);
    heap_caps_free(layer.mem);
  }
  printf("BENCH_MICRO done\n");
}
#endif
