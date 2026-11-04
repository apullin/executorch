/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Asymmetric quantisation -- the blob's records, and the half of the
// runtime that reads them.
//
// Three flag/record pairs and one opcode of the delegate blob carry it
// (this header is the contract; serialization/asym_records.py writes the
// records):
//
//   kFlagActQuant     + kExtActQuant      how the op quantises its activation
//                                         (dynamic affine, static, symmetric
//                                         grids other than the backend's own)
//   kFlagWeightAffine + kExtWeightAffine  weight zero points, float offsets and
//                                         the weight sums S' an activation zero
//                                         point needs
//   kFlagOutQuant     + kExtOutQuant      a model's output quantize, applied
//                                         as fake-quant after the epilogue
//   kOpBmmKcacheAffineI8                  q@kT against ExecuTorch's asymmetric
//                                         int8 K cache
//
// Each flag requires its record and each record its flag, the FLAG_CONV <->
// EXT_CONV3D discipline.
// What is computed, scheme by scheme:
// - weight records -- integer and float zero points, float offsets -- on
//   int8/int4 matmuls and embeddings;
// - dynamic activation records -- per row or per tensor, the affine min/max
//   chooser or a symmetric one on the model's own grid -- on int8/int4
//   matmuls and convs, with the weight sums S' an activation zero point needs;
// - static activation records -- a stock static int8 graph's calibrated
//   constants, the conv padding written as their zero point -- and output
//   records, the model's quantize applied as fake-quant after the op's
//   epilogue (Stage A: fp32 between delegates), on int8/int4 matmuls and convs;
// - static 16-bit activation records (Arm a16w8: int16 codes, zero point 0) on
//   ungrouped int8 matmuls and convs, dotted in int64;
// - the affine K-cache opcode (ExecuTorch's stock QuantizedKVCache,
//   _passes/stock_llm.py).
// Output records on any other opcode are refused as "unimplemented". The field
// values are what the exporters emit: the backend's own quantiser (A0) is no
// record at all, and XNNPACK's per-token chooser (A4) quantises the K cache's
// own codes (kv_affine.h), never an op's activation, so neither has a value.
// (A0-A7 are quantizer/mirrors.py's ActScheme labels.)
// Why a separate translation unit. Everything here is compiled
// -ffp-contract=off on the device and the host (the affine epilogues must be
// bit-exact against the Python mirrors, quantizer/mirrors.py, and against each
// other), while esp32_backend.cpp keeps its flags -- changing those would
// change the symmetric models' bits. Its code is linked after
// all other flash code (esp32_asym.lf), so it moves no hot function.
//
// The declarations below are deliberately NOT marked cold: GCC treats a
// function that unconditionally calls a cold one as cold itself, which would
// re-place and re-inline the delegate's init(). Only the definitions are.
#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace esp32_asym {

// ---- descriptor flags, record types, opcode (the wire values) --------------
constexpr uint32_t kFlagActQuant = 524288;
constexpr uint32_t kFlagWeightAffine = 1048576;
constexpr uint32_t kFlagOutQuant = 2097152;
constexpr uint32_t kV24Flags = kFlagActQuant | kFlagWeightAffine | kFlagOutQuant;
constexpr uint16_t kExtActQuant = 8;
constexpr uint16_t kExtWeightAffine = 9;
constexpr uint16_t kExtOutQuant = 10;
// q@kT against an asymmetric int8 K cache. OP_BMM_KCACHE_I8's descriptor
// reuse, plus one: wt_off carries the zero-point tensor's ARG index (int8, one
// per key row, like the scales at scale_off).
constexpr uint32_t kOpBmmKcacheAffineI8 = 10;

// ---- EXT_ACT_QUANT field values ---------------------------------------------
// in_dtype and EXT_OUT_QUANT's out_dtype: what the op reads and writes. fp32
// is Stage A; int8 codes are Stage B's, accepted only in a delegate
// whose compile specs carry "stage_b" (validate's stage_b argument); int16
// codes are not part of Stage B.
constexpr uint32_t kDtypeFp32 = 0;
constexpr uint32_t kDtypeInt8 = 1;
constexpr uint32_t kDtypeInt16 = 2;
// granularity: one dynamic scale (and zero point) per row, one for the whole
// tensor, or the record's own constants.
constexpr uint32_t kGranularityRow = 0;
constexpr uint32_t kGranularityTensor = 1;
constexpr uint32_t kGranularityStatic = 2;
// chooser. kChooserNone is the static scheme's.
//   kChooserSymDiv          s = max(fl(amax / sym_div), eps), zero point 0
//                           (A1, A7)
//   kChooserAffineMinMax    s = max(fl(fl(hi - lo) / (qmax - qmin)), eps),
//                           zp = clamp(qmin - rne(fl(lo / s))) (A2, A3)
constexpr uint32_t kChooserNone = 0;
constexpr uint32_t kChooserSymDiv = 1;
constexpr uint32_t kChooserAffineMinMax = 2;
// rounding, the one value both records take: clamp(rne(fl(x * fl(1/s))) +
// zp), torch's quantize ops. (0 and 2 stand for A0's half-away rounding and
// A4's float64 one, which no record carries.)
constexpr uint32_t kRoundRneBeforeZp = 1;

// ---- EXT_WEIGHT_AFFINE field values ------------------------------------------
// One table entry per scale group, the same [n, groups] layout and group
// boundaries as the op's scales. Integer zero points are stored biased like
// the codes (zw = z_source - storage_bias), so c - zw == q - z_source exactly.
// A float zero point (embeddings only) applies to the source's own code,
// q = c + storage_bias: fl(fl(q - z) * s).
constexpr uint32_t kZpNone = 0;
constexpr uint32_t kZpInt8 = 1;
constexpr uint32_t kZpFp32 = 2;
// Float offsets, added after the group's scale term: fl(fl(... * s) + m).
constexpr uint32_t kOffsetNone = 0;
constexpr uint32_t kOffsetFp32 = 1;
constexpr uint32_t kOffsetFp16 = 2;
// The weight sums S'[n, g] = sum over the group's REAL elements of (c - zw).
constexpr uint32_t kSumNone = 0;
constexpr uint32_t kSumInt16 = 1;
constexpr uint32_t kSumInt32 = 2;

// ---- the records ---------------------------------------------------------------
// Payloads exactly as asym_records.py packs them: every field 4 bytes,
// little-endian. Decoded by memcpy, so the record's place in the blob needs no
// alignment.

/// EXT_ACT_QUANT: how a matmul quantises its activation, replacing the
/// backend's per-row absmax.
struct ActQuant {
  uint32_t in_dtype;     // kDtype*: fp32, or Stage B's int8 codes (static)
  uint32_t code_bits;    // 8, or 16 (an ungrouped OP_ADDMM_I8, symmetric only)
  uint32_t granularity;  // kGranularity*
  uint32_t chooser;      // kChooser*
  uint32_t rounding;     // kRound*
  int32_t qmin;          // the code range, already in the signed code domain
  int32_t qmax;          // (uint8 model IO arrives shifted by -128)
  float eps;             // the chooser's scale floor; 0 for a static scheme
  float sym_div;         // kChooserSymDiv's divisor; 0 otherwise
  float scale;           // static scheme only (a float32 value); 0 otherwise
  int32_t zero_point;    // static scheme only; 0 otherwise
};
constexpr uint32_t kActQuantBytes = 44;
static_assert(sizeof(ActQuant) == kActQuantBytes,
              "ActQuant must match the AOT struct.pack layout");

/// EXT_WEIGHT_AFFINE: what a weight needs beyond codes and scales.
struct WeightAffine {
  uint32_t zp_kind;      // kZp*
  uint32_t offset_kind;  // kOffset*
  uint32_t sum_kind;     // kSum*
  // The storage bias: what the exporter subtracted from an unsigned source's
  // codes to store them signed, c = q - storage_bias -- 0 (a signed source),
  // 8 (uint4) or 128 (uint8). Without a zero-point table the source had no
  // zero point, so the stored codes' is -storage_bias (GGUF's fl(q * d)).
  int32_t storage_bias;
  // Table offsets: bytes into the blob, or under kFlagExternalWeights the
  // offsets of NUL-terminated NamedDataMap keys, as wt_off/scale_off are. 0
  // for an absent table.
  uint32_t zp_off;
  uint32_t offset_off;
  uint32_t sum_off;
};
constexpr uint32_t kWeightAffineBytes = 28;
static_assert(sizeof(WeightAffine) == kWeightAffineBytes,
              "WeightAffine must match the AOT struct.pack layout");

/// EXT_OUT_QUANT: the model's output quantize (Stage A writes the
/// dequantised value back, so a downstream quantize sees the same code).
struct OutQuant {
  uint32_t out_dtype;    // kDtype*: fp32 fake-quant, or Stage B's int8 codes
  uint32_t rounding;     // kRoundRneBeforeZp, the only one a quantize uses
  float scale;           // a float32 value
  int32_t zero_point;
  int32_t qmin;
  int32_t qmax;
};
constexpr uint32_t kOutQuantBytes = 24;
static_assert(sizeof(OutQuant) == kOutQuantBytes,
              "OutQuant must match the AOT struct.pack layout");

/// The descriptor as this unit reads it from the blob: the same 64 bytes as
/// esp32_backend.cpp's OpDesc (test_blob_abi.py holds the two to one field
/// order), which lives in that file's anonymous namespace.
struct OpDescView {
  uint32_t opcode;
  uint32_t flags;
  uint32_t m, k, n;
  uint32_t k_pad;
  uint32_t bias_off;
  uint32_t wt_off;
  uint32_t scale_off;
  uint32_t in_arg;
  uint32_t out_arg;
  uint32_t scale_group;
  float clamp_lo;
  float clamp_hi;
  uint32_t res_arg;
  float act_scale;
};
static_assert(sizeof(OpDescView) == 64, "OpDescView must match OpDesc");

// ---- per-op state ------------------------------------------------------------
/// A conv op's gather geometry: esp32_backend.cpp's ConvGeom (which lives in
/// that file's anonymous namespace) field for field, less what the gather does
/// not read; the backend copies its geometry in once per op.
struct ConvGather {
  uint32_t h_in, w_in, c_in;
  uint32_t h_out, w_out;
  uint32_t kh, kw;
  uint32_t stride_h, stride_w;
  uint32_t pad_h, pad_w;
  uint32_t dil_h, dil_w;
  uint32_t d_in, d_out, kd, stride_d, pad_d, dil_d;
  uint32_t transposed;
};

/// One op's asymmetric-quantisation state, resolved at init: its decoded
/// records and its tables (inline in the blob, or held NamedDataMap
/// buffers). All null for an op without asymmetric records.
struct OpQuant {
  const ActQuant* act;
  const WeightAffine* weight;
  const OutQuant* out;
  const void* zero_points;  // int8 or fp32, per weight->zp_kind
  const void* offsets;      // fp32 or fp16, per weight->offset_kind
  const void* sums;         // int16 or int32, per weight->sum_kind
  // FLAG_STAGE_I8 expanded the codes at init with every group's zero point
  // already subtracted (fold_zero_points): they are c - zw, and the column
  // math applies no correction.
  bool zp_folded;
  // fl(1 / out->scale), the output quantisation's multiplier, once at attach
  // (0 without an output record).
  float out_inv;
};

/// A handle's asymmetric-quantisation state: the side array CompiledBlob::asym
/// points to. Null, so nothing allocated, for a blob without asymmetric records.
struct BlobState {
  OpQuant* ops;  // n_ops entries
  // External tables, held for the method's lifetime: a NamedDataMap pointer
  // is only valid while its FreeableBuffer lives.
  executorch::runtime::FreeableBuffer* held;
  uint32_t n_held;
  // The affine matmuls' per-row terms (X_g, T_g) for one row block: an offset
  // into the per-call temporary scratch esp32_backend.cpp lays out at init, or
  // kNoTerms for a blob without an affine matmul.
  uint32_t terms_off;
};
constexpr uint32_t kNoTerms = 0xFFFFFFFFu;

// ---- the runtime half ----------------------------------------------------------

/// Every asymmetric-quantisation rule the blob decides by itself. Runs
/// BEFORE validate_blob, so that an op carrying asymmetric content is refused
/// for its own rule rather than for whichever general rule trips first. Reads
/// only what it has bounds-checked; a structurally broken blob is left for
/// validate_blob and validate_extensions to refuse. Ok, and nothing done, for
/// a blob with no asymmetric content. `stage_b`: the compile specs carry
/// "stage_b", so int8 activation and output codes are allowed (Stage B).
executorch::runtime::Error validate(const uint8_t* data, size_t size,
                                    uint32_t n_ops, uint32_t ext_off,
                                    bool stage_b = false);

/// At init, after the structural validation: decode the records, resolve the
/// tables (inline, or held from the NamedDataMap), check their contents
/// (weight zero points within the code range) and refuse what this runtime
/// does not compute ("unimplemented": dynamic 16-bit codes, a per-row
/// record on a conv, output records beside the int8/int4 matmuls, codes on a
/// selected-row head). *state stays null for a blob without records; on
/// failure it may hold buffers, which release() frees (release_resources
/// calls it on every path).
executorch::runtime::Error attach(executorch::runtime::BackendInitContext& context,
                                  const uint8_t* data, size_t size,
                                  uint32_t n_ops, uint32_t ext_off,
                                  BlobState** state);

/// Destructs the held buffers. Safe on null and on a partly attached state.
void release(BlobState* state);

/// OP_BMM_KCACHE_AFFINE_I8's zero-point tensor at execute: int8, one per key
/// row (batch * n elements, as the scales). run_kcache_affine calls it on
/// every execution; run_kcache_zero_point_check drives it directly.
executorch::runtime::Error check_kcache_zero_points(
    const executorch::aten::Tensor& zero_points, uint32_t batch, uint32_t n,
    uint32_t op_index);

// ---- the affine K cache ------------------------------------------------------------
// q@kT against ExecuTorch's stock QuantizedKVCache (stock_llm.py): key rows
// stored as int8 codes kq with one fp32 scale sk and one int8 zero point zk
// each, as XNNPACK's per-token chooser left them. Queries stay the delegate's
// own symmetric rows (A0: sq = amax / 127, codes fl(x * fl(1 / sq)) rounded
// half away from zero, clamped to +-127), and with Q = the sum of a query
// row's codes,
//
//   score = fl(fl(fl(sum(qq * kq) - zk * Q) * sq) * sk)
//
// the integer side wrapping in uint32 -- exact while k_pad * 127 * 255 fits
// int32 (kcache-dot-bound) -- mirrors.kcache_scores bit for bit.

/// The op's call: queries [B,M,K] fp32 at d.in_arg, codes [B,N,K] int8 at
/// d.res_arg, scales [B,N] fp32 at d.scale_off, zero points [B,N] int8 at
/// d.wt_off, scores [B,M,N] fp32 at d.out_arg (B = d.scale_group). The
/// argument indices are checked here; `force_scalar` turns the PIE dot off.
/// `qq` takes one query row's codes (k_pad), `staged` one batch of key rows
/// zero-padded when a row is ragged (k < k_pad; n * k_pad): the call's
/// scratch, 16-byte aligned, as for every bmm.
executorch::runtime::Error run_kcache_affine(
    uint32_t op_index, const OpDescView& d,
    executorch::runtime::Span<executorch::runtime::EValue*> args, int8_t* qq,
    int8_t* staged, bool force_scalar);

/// One query row's A0 codes (k real values, zero to k_pad) and its scale,
/// with Q in *qsum: the step run_kcache_affine repeats per query row (and
/// the conformance harness drives directly).
float kcache_query_row(const float* x, uint32_t k, uint32_t k_pad, int8_t* qq,
                       int32_t* qsum);

/// One score from a query row's codes and a key row's (k_pad each), as
/// run_kcache_affine computes it: the PIE dot when `pie` (16-byte aligned
/// rows, k_pad a multiple of 16), else the scalar loop -- the same integer.
float kcache_score(const int8_t* qq, int32_t qsum, float sq, const int8_t* kq,
                   uint32_t k_pad, float sk, int8_t zk, bool pie);

/// OP_BMM_KCACHE_AFFINE_I8 executions and key rows read, for the parity
/// firmware's proof that its rows ran the op.
uint32_t kcache_affine_ops();
uint64_t kcache_affine_rows();

// ---- the affine column math ------------------------------------------------------
// The column math of an int8/int4 matmul whose weights carry EXT_WEIGHT_AFFINE
// or whose activation carries EXT_ACT_QUANT, and the dequantisation of an
// affine embedding table. Activations come from the op's own quantiser (the
// dynamic activations section below) or, without a record, the backend's own
// (A0, zx = 0: quantize_row_a0 below); esp32_backend.cpp owns the loops, the
// dual-core split and the epilogue. What lives here is every rounding the
// affine schemes add,
// -ffp-contract=off: per scale group g of column n, with X_g the sum of the
// row's codes over g and n_g its real elements (columns below k),
//
//   acc = raw - zw * X_g - zx * S'[n, g] (wrapping uint32: exact while the true
//                                         value fits int32)
//   y   = fl(y + fl(fl(fl(acc) * sx) * s[n, g]))
//   y   = fl(y + fl(m[n, g] * T_g))      with float offsets,
//                                        T_g = fl(fl(X_g - n_g * zx) * sx)
//
// from the bias, group by group -- mirrors.affine_matmul with
// fused_madd=False, bit for bit.

/// An op's weight side as the column math reads it: its scales, and its zero
/// points, float offsets and weight sums from the op's OpQuant. Resolved per
/// call (affine_weights); points into the blob, NamedDataMap buffers or a
/// staged copy.
struct AffineWeights {
  uint32_t k;                  // real columns: n_g counts those below k
  uint32_t k_pad;
  uint32_t gsize;              // elements per scale group (k_pad: one group)
  uint32_t ngw;                // groups per row
  bool fp16_scales;
  bool offsets_fp16;
  bool sums_i16;
  const void* scales;          // [rows, ngw], fp32 or IEEE half
  const int8_t* zero_points;   // [rows, ngw] int8, or null: every group's is zw
  int32_t zw;                  // the zero point without a table: -storage_bias
                               // (0 once FLAG_STAGE_I8 folded the table in)
  const void* offsets;         // [rows, ngw] fp32 or IEEE half, or null
  const void* sums;            // [rows, ngw] S', int16 or int32, or null (no
                               // activation zero point)
};

/// Whether op i carries a weight record.
inline bool weight_affine(const BlobState* state, uint32_t i) {
  return state != nullptr && state->ops[i].weight != nullptr;
}

/// Op i's activation record, or null: the backend's own quantisers.
inline const ActQuant* act_record(const BlobState* state, uint32_t i) {
  return state != nullptr ? state->ops[i].act : nullptr;
}

/// Whether op i takes the affine loops: a weight, an activation or an output
/// record.
inline bool affine_op(const BlobState* state, uint32_t i) {
  return state != nullptr &&
      (state->ops[i].weight != nullptr || state->ops[i].act != nullptr ||
       state->ops[i].out != nullptr);
}

/// Op i's OpQuant when it carries an output record, else null: what the
/// affine loops pass requantize.
inline const OpQuant* out_record(const BlobState* state, uint32_t i) {
  return state != nullptr && state->ops[i].out != nullptr ? &state->ops[i]
                                                          : nullptr;
}

/// One op's AffineWeights (symmetric codes when the op has no weight record).
/// `scale_group` is the descriptor's (0 = per row).
AffineWeights affine_weights(const OpQuant& q, uint32_t k, uint32_t k_pad,
                             uint32_t scale_group, bool fp16_scales,
                             const void* scales);

/// One matmul row of k activations under the backend's own quantiser (A0, an
/// op without an activation record) into dst, k_pad wide with the tail zero,
/// with xsum[g] = X_g and, when toff is non-null, toff[g] = T_g = fl(fl(X_g)
/// * s) for every group of w, accumulated as the codes are written; returns
/// the scale s. esp32_backend.cpp's quantize_row_i8 bit for bit: s = fix when
/// fix >= 0 (FLAG_STATIC_ACT's calibrated scale, or FLAG_ACT_PER_TENSOR's
/// absmax over the whole tensor), else fl(amax / 127) over the row; codes
/// fl(x * fl(1 / s)) rounded half away from zero -- to even under
/// ties_to_even, the per-tensor scheme's PT2E rounding -- clamped to +-127.
float quantize_row_a0(const float* x, uint32_t k, float fix, bool ties_to_even,
                      const AffineWeights& w, int8_t* dst, int32_t* xsum,
                      float* toff);

/// The conv gather (quantize_conv_row's column order) under the backend's own
/// quantiser: esp32_backend.cpp's quantize_conv_row bit for bit -- the absmax
/// over the taps inside the input, a 0 code for a tap outside it or off a
/// transposed conv's lattice -- with X_g and T_g as quantize_row_a0.
float quantize_conv_row_a0(const float* x, const ConvGather& g, uint32_t row,
                           float fix, bool ties_to_even,
                           const AffineWeights& w, int8_t* dst, int32_t* xsum,
                           float* toff);

/// y plus column n's corrected dot, group by group: `wrow` is the column's int8
/// codes as the dot reads them (stored, staged, or an i4 row unpacked to
/// c = nibble - 8); `pie` selects dot_i8_s3 (16-byte aligned rows and groups)
/// over the scalar loop, which is bit-identical. acc = raw - zw * X_g, less
/// zx * S'[n, g] when the op has weight sums.
float accumulate(const AffineWeights& w, uint32_t n, const int8_t* xq,
                 const int8_t* wrow, bool pie, float sx, int32_t zx,
                 const int32_t* xsum, const float* toff, float y);

/// The same from the fused int4 kernel's raw per-group sums, which dot the
/// STORED nibbles c + 8: acc = raw - (8 + zw) * X_g - zx * S'[n, g].
float accumulate_fused(const AffineWeights& w, uint32_t n, const int32_t* raw,
                       float sx, int32_t zx, const int32_t* xsum,
                       const float* toff, float y);

// ---- dynamic activations ---------------------------------------------------------
// An int8/int4 matmul or conv whose activation the model quantises itself
// (EXT_ACT_QUANT, from a dynamic chain the exporter absorbed): per row
// (torchao's per token) or per tensor (PT2E's); the affine min/max chooser
// (A2, A3) or a symmetric one on the model's own grid (A1, A7); rounded to
// even before the zero point. Each step is the single IEEE float32 operation
// torch performs, in torch's order, so the device, the host runner and the
// Python mirror (quantizer/mirrors.py's quantize_activation) agree bit for
// bit:
//
//   lo = min(amin, 0)    hi = max(amax, 0)
//   affine     s = max(fl(fl(hi - lo) / (qmax - qmin)), eps)
//              zx = clamp(qmin - rne(fl(lo / s)), qmin, qmax)
//   symmetric  s = max(fl(max(-lo, hi) / sym_div), eps), zx = 0
//   xq = clamp(rne(fl(x * fl(1 / s))) + zx, qmin, qmax)
//
// A NaN activation's code is zx and an infinity's its rail, ExecuTorch 1.6's
// rule for quantize_per_tensor (quant_rules.h). The chooser skips a NaN (its
// comparisons fail); an infinite range makes an infinite scale, whose zero
// inverse sends every code to zx, and zx = qmin for the affine chooser
// (fl(lo / s) is 0, or NaN for lo = -inf, and either rounds to 0).

/// One row's (or one tensor's) scale and zero point.
struct ActParams {
  float sx;
  int32_t zx;
};

/// The record's chooser over n activations: one row (per-row schemes) or the
/// op's whole input (per tensor, chosen once per call). A static record's
/// are its own constants, whatever x holds: xq = clamp(rne(fl(x *
/// fl(1 / s))) + zx), torch's quantize_per_tensor, with fl(x * fl(1 / s))
/// clamped to +-2^22 first so that an input far past the calibrated range
/// still lands on its rail.
ActParams choose(const ActQuant& a, const float* x, size_t n);

/// One matmul row of k activations quantised at p into dst (k_pad wide, the
/// tail zero), with xsum[g] = X_g and, when toff is non-null, toff[g] = T_g
/// for every group of w, accumulated as the codes are written.
void quantize_row(const ActQuant& a, ActParams p, const float* x, uint32_t k,
                  const AffineWeights& w, int8_t* dst, int32_t* xsum,
                  float* toff);

/// The conv gather (esp32_backend.cpp's quantize_conv_row_t, column order
/// (c, d, i, j)) quantised at p: a tap outside the input, or off a transposed
/// conv's lattice, reads a real 0.0, whose code is zx, so it is written as zx
/// and adds (zx - zx) * w = 0 to the corrected dot (S' counts it, X_g holds
/// it). The k_pad tail stays 0. X_g and T_g as quantize_row.
void quantize_conv_row(const ActQuant& a, ActParams p, const float* x,
                       const ConvGather& g, uint32_t row, const AffineWeights& w,
                       int8_t* dst, int32_t* xsum, float* toff);

// ---- output quantisation ----------------------------------------------------------
// EXT_OUT_QUANT: a static int8 graph's quantize after the op, applied as
// fake-quant once the op's activation and clamp have run, in torch's order:
//
//   yq = clamp(rne(fl(y * fl(1 / so))) + zo, qmin, qmax)
//   y' = fl((yq - zo) * so)
//
// fl(y * fl(1 / so)) is clamped to +-2^22 first, as the activation quantiser
// does, so an output far past the calibrated range lands on its rail; NaN
// takes zo (quant_rules.h), so y' = 0. A downstream quantize with
// the same constants returns yq exactly (|yq - zo| <= 65535 keeps the error
// far below half a code).
float requantize(const OpQuant& q, float y);

// ---- Stage B: int8 codes in and out ------------------------------------------------
// Under the "stage_b" compile spec, a static int8 graph's codes travel as codes
// (stage_b.py): an op whose output record says out_dtype int8 writes yq itself
// -- requantize's code, which Stage A would have written as fl((yq - zo) *
// so) for the next op to quantise back to yq -- and an op whose static record
// says in_dtype int8 reads the model's codes as they are, with no quantiser.
// Integers only: the same codes as Stage A, bit for bit, wherever the export
// proved Stage A's requantise the identity (q(fl(fl(c - z) * s)) == c for
// every code).

/// Whether op i reads int8 codes (EXT_ACT_QUANT in_dtype int8).
inline bool codes_in(const BlobState* state, uint32_t i) {
  return state != nullptr && state->ops[i].act != nullptr &&
      state->ops[i].act->in_dtype == kDtypeInt8;
}

/// Whether op i writes int8 codes (EXT_OUT_QUANT out_dtype int8).
inline bool codes_out(const BlobState* state, uint32_t i) {
  return state != nullptr && state->ops[i].out != nullptr &&
      state->ops[i].out->out_dtype == kDtypeInt8;
}

/// requantize's code: clamp(rne(fl(y * fl(1 / so))) + zo, qmin, qmax), the
/// int8 code an out_dtype int8 op writes.
int8_t requantize_code(const OpQuant& q, float y);

/// One matmul row of k int8 codes (the record's own, zero point a.zero_point)
/// copied into dst, k_pad wide with the tail zero, with xsum[g] = X_g and,
/// when toff is non-null, toff[g] = T_g for every group of w: quantize_row
/// with the codes already made.
void codes_row(const ActQuant& a, const int8_t* x, uint32_t k,
               const AffineWeights& w, int8_t* dst, int32_t* xsum,
               float* toff);

/// The conv gather (quantize_conv_row's column order) of int8 codes: a tap
/// outside the input, or off a transposed conv's lattice, is the record's
/// zero point -- the code a real 0.0 has -- and the k_pad tail 0. X_g and T_g
/// as quantize_conv_row.
void codes_conv_row(const ActQuant& a, const int8_t* x, const ConvGather& g,
                    uint32_t row, const AffineWeights& w, int8_t* dst,
                    int32_t* xsum, float* toff);

/// FLAG_STAGE_I8 with zero points: subtract each group's zero point from one
/// expanded row in place (int4 codes and zero points give c - zw in [-15, 15]).
/// The caller marks the op folded (OpQuant::zp_folded) once every row is.
void fold_zero_points(const AffineWeights& w, uint32_t row, int8_t* codes);

// ---- 16-bit activations --------------------------------------------------------
// Arm a16w8 (A6): a static int16 record -- the model's scale,
// zero point 0, its int16 range -- on an ungrouped int8 matmul or conv whose
// weights are symmetric int8 codes (no weight record). Scalar, as the i16
// rung's run_addmm_i16 is, and in torch's order:
//
//   xq  = clamp(rne(fl(x * fl(1 / s))) + zx, qmin, qmax)   (the static
//         quantiser above: code_of over the record's range; zx = 0)
//   acc = sum over k_pad of xq * c                        (int64: exact)
//   y   = fl(bias + fl(fl(fl(acc) * s) * s_w[n]))
//
// fl(acc) is one rounding of an exact integer: libgcc's __floatdisf on the
// device and scvtf on the host both round to nearest even (run_a16_check
// holds the device to an integer-only reference). mirrors.affine_matmul's
// 16-bit branch is the reference, bit for bit. validate bounds the dot at
// 131,040 elements, the 40-bit ACCX's bound, which the int64 sum does not
// need; a PIE kernel could keep the blobs.

/// Whether op i carries a 16-bit activation record: the int16 loop's ops.
inline bool a16_op(const BlobState* state, uint32_t i) {
  return state != nullptr && state->ops[i].act != nullptr &&
      state->ops[i].act->code_bits == 16;
}

/// One matmul row of k activations quantised to int16 codes into dst (k_pad
/// wide, the tail zero) with the record's constants.
void quantize_row16(const ActQuant& a, const float* x, uint32_t k,
                    uint32_t k_pad, int16_t* dst);

/// The conv gather (quantize_conv_row's column order and padding: a tap
/// outside the input or off a transposed conv's lattice is the record's
/// zero point) to int16 codes, k_pad wide, the tail zero.
void quantize_conv_row16(const ActQuant& a, const float* x,
                         const ConvGather& g, uint32_t row, uint32_t k_pad,
                         int16_t* dst);

/// The conv gather of fp32 values for the float-weight convolution
/// (esp32_float.h): quantize_conv_row's column order and padding, a tap
/// outside the input or off a transposed conv's lattice a real 0.0. k values
/// into dst, nothing quantised.
void gather_conv_row(const float* x, const ConvGather& g, uint32_t row,
                     float* dst);

/// The same walk's input offsets: offs[t] is where tap t of output position
/// `row` reads from x (gather_conv_row's x), or -1 for a tap that reads a
/// real 0.0. A depthwise conv walks it once per position for every channel.
void conv_tap_offsets(const ConvGather& g, uint32_t row, int32_t* offs);

/// y plus column n's int64 dot of the int16 codes xq against its int8 codes
/// wrow (w.k_pad each), scaled: fl(y + fl(fl(fl(acc) * sx) * s[n])).
float a16_column(const AffineWeights& w, uint32_t n, const int16_t* xq,
                 const int8_t* wrow, float sx, float y);

/// One gathered embedding row, dequantised as the reference does: an integer
/// zero point fl((float)(c - zw) * s), a float zero point fl(fl(q - z) * s) on
/// the source's own code q = c + storage_bias, a float offset
/// fl(fl((float)(c - zw) * s) + m). `row` is the stored row (i4: two codes
/// per byte, element 2i low, + 8); `first` is the row's first table entry
/// (row index * ng).
void dequant_row(const OpQuant& q, bool i4, const uint8_t* row, uint32_t d,
                 uint32_t gsize, uint32_t ng, const void* scales,
                 bool fp16_scales, size_t first, float* dst);

// ---- test hooks ------------------------------------------------------------------
// The refusal tests prove each case is refused for its own reason. Every
// refusal here logs "esp32: v24 refused [<rule>] ..." and records the rule,
// which the device test reads back through these.
uint32_t last_refusal();
void clear_last_refusal();
const char* rule_name(uint32_t rule);

/// Drives check_kcache_zero_points with hand-built tensors and prints
/// "asym kcache zp check: control=... dtype=... shape=...".
void run_kcache_zero_point_check();

/// Drives a16_column on hand-built codes whose dots land on exact float32
/// ties and past int32, against an integer-only reference, and prints
/// "asym a16 check: conversions=<exact>/<n> wide=<exact>/<n>".
void run_a16_check();

}  // namespace esp32_asym
