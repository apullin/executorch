/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The runtime half of the blob's asymmetric-quantisation records: the rules a
// blob must satisfy, the per-op state they become at init, the arithmetic of
// the schemes it computes (weight zero points and float offsets; dynamic
// activation quantisers and the weight sums their zero points need), and the
// refusal of the rest. The contract, record by record and field by field, is
// esp32_asym.h's.
//
// Built -ffp-contract=off on the device and the host (both CMakeLists).
// esp32_asym.lf links this object's code after all other flash code, so its
// size moves no hot function. The validation and init functions are cold; the
// column math of an affine matmul is not, since it runs per output element.
//
// Each refusal names its rule, "esp32: v24 refused [<rule>] ...", and records
// it for last_refusal(): the synthetic-blob tests (test_asym_refusals.py on
// the host runner, run_asym_refusal_check under QEMU) prove that every case is
// refused for its own reason, not for whichever other rule trips first.

#include "esp32_asym.h"
#include "esp32_codes.h"

#include "esp32_kernels.h"
#include "esp32_platform.h"
#include "quant_rules.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>

#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/platform/log.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::BackendInitContext;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Span;

#define ASYM_COLD __attribute__((cold))

namespace esp32_asym {
// The code arithmetic esp32_conv.cpp shares (esp32_codes.h).
using namespace esp32_codes;
namespace {

// The wire values this unit reads beyond its own records (namespace v23: the
// base layout version 24 extends). esp32_backend.cpp owns them, in its anonymous
// namespace; these are copies, which test_blob_abi.py holds to the originals.
namespace v23 {
constexpr uint32_t kHeaderBytes = 16;
constexpr uint32_t kOpDescBytes = 64;
constexpr uint32_t kExtCountBytes = 4;
constexpr uint32_t kExtRecordHeaderBytes = 8;
constexpr uint16_t kExtOptional = 0x8000;
constexpr uint16_t kExtSelectedRows = 7;
constexpr uint32_t kOpAddmm = 1;
constexpr uint32_t kOpAddmmI8 = 2;
constexpr uint32_t kOpAddmmI4 = 3;
constexpr uint32_t kOpAddmmI16 = 4;
constexpr uint32_t kOpEmbedI8 = 5;
constexpr uint32_t kOpEmbedI4 = 6;
constexpr uint32_t kOpEmbedI16 = 7;
constexpr uint32_t kOpBmmDynI8 = 8;
constexpr uint32_t kOpBmmKcacheI8 = 9;
constexpr uint32_t kFlagExternalWeights = 1;
constexpr uint32_t kFlagAdd = 32;
constexpr uint32_t kFlagStaticAct = 64;
constexpr uint32_t kFlagConv = 128;
constexpr uint32_t kFlagActPerTensor = 256;
constexpr uint32_t kFlagStageI8 = 1024;
constexpr uint32_t kFlagMul = 4096;
constexpr uint32_t kNumScratch = 3;
constexpr uint32_t kArgScratch0 = 0xFFFFFFFFu - (kNumScratch - 1);
}  // namespace v23

// The exactness bounds, as mirrors.dot_length_bound
// computes them: the longest dot whose true value, with every zero-point
// correction applied, fits the accumulator -- int32 for 8-bit codes (the PIE
// kernels read back ACCX's low 32 bits), the 40-bit ACCX for 16-bit ones.
constexpr uint64_t kInt32Max = 0x7FFFFFFFull;
constexpr uint64_t kAccx40Max = (1ull << 39) - 1;

/// The longest dot (a multiple of 32) whose true value fits `limit` when
/// |x - zx| <= act_span and |c - zw| <= weight_span.
constexpr uint64_t dot_bound(uint64_t limit, uint64_t act_span,
                             uint64_t weight_span) {
  return limit / (act_span * weight_span) -
      (limit / (act_span * weight_span)) % 32;
}
// The two affine int8 bounds, as check_bounds computes them: an
// int8 activation with a zero point spans qmax - qmin = 255; int8 weight
// codes reach -128 symmetric, 255 apart with a zero point.
static_assert(dot_bound(kInt32Max, 255, 128) == 65792,
              "affine int8 activations x symmetric int8 weights");
static_assert(dot_bound(kInt32Max, 255, 255) == 33024,
              "affine int8 activations x affine int8 weights");
// 16-bit codes (|xq| <= 32768, zero point 0) against int8 weights
// (|c| <= 128), bounded by the 40-bit ACCX (131,040, not
// 131,076). The int16 loop sums in int64 and would not overflow past it;
// the bound keeps every 16-bit blob valid for a PIE kernel's ACCX.
static_assert(dot_bound(kAccx40Max, 32768, 128) == 131040,
              "16-bit activations x int8 weights");
// A float offset's activation term converts X'_g = X_g - n_g * zx to float;
// that is exact only while |X'_g| <= 2^24.
constexpr uint64_t kFloatExactInt = 1ull << 24;
// An int16 S' holds |S'| <= 32767, which int16 storage needs PROVABLY: the
// exporter's values cannot be checked here once they have been truncated.
constexpr uint64_t kInt16Max = 32767;
// OP_BMM_KCACHE_AFFINE_I8: symmetric A0 queries (|qq| <= 127) against key
// codes with a zero point (|kq - zk| <= 255).
constexpr uint32_t kKcacheQuerySpan = 127;
constexpr uint32_t kKcacheKeySpan = 255;
// The PIE dot's operands must start 16-byte aligned (ee.vld.128.ip truncates
// a misaligned address silently): run_kcache_affine checks its rows.
constexpr size_t kKcacheRowAlign = 16;

enum Rule : uint32_t {
  kRuleNone = 0,
  kRuleRecordSize,
  kRuleRecordWithoutFlag,
  kRuleFlagWithoutRecord,
  kRuleDuplicateRecord,
  kRuleActOpcode,
  kRuleActSchemeFlags,
  kRuleActFieldValue,
  kRuleActStageB,
  kRuleActCodeBits,
  kRuleActRange,
  kRuleActCombination,
  kRuleActUnsignedSymmetric,
  kRuleActStaticScale,
  kRuleActStaticZeroPoint,
  kRuleActEps,
  kRuleActSymDiv,
  kRuleActUnusedField,
  kRuleActZeroPointWithoutSums,
  kRuleWeightOpcode,
  kRuleWeightFieldValue,
  kRuleWeightEmpty,
  kRuleWeightFloatZpMatmul,
  kRuleWeightCombination,
  kRuleWeightSumsUnused,
  kRuleWeightSumsWidth,
  kRuleWeightStagedOffsets,
  kRuleWeightUnusedField,
  kRuleWeightTableBounds,
  kRuleWeightTableExternal,
  kRuleWeightZeroPointRange,
  kRuleOutOpcode,
  kRuleOutFusedOperand,
  kRuleOutFieldValue,
  kRuleOutStageB,
  kRuleOutRange,
  kRuleOutScale,
  kRuleOutZeroPoint,
  kRuleDotBound,
  kRuleOffsetBound,
  kRuleKcacheArgs,
  kRuleKcacheDescriptor,
  kRuleKcacheDotBound,
  kRuleKcacheZeroPoints,
  kRuleUnimplemented,
  kNumRules
};

// The names the tests match (asym_blob_cases.py RULES); one per Rule, in order.
const char* const kRuleNames[] = {
    "none",
    "record-size",
    "record-without-flag",
    "flag-without-record",
    "duplicate-record",
    "act-opcode",
    "act-scheme-flags",
    "act-field-value",
    "act-stage-b",
    "act-code-bits",
    "act-range",
    "act-combination",
    "act-unsigned-symmetric",
    "act-static-scale",
    "act-static-zero-point",
    "act-eps",
    "act-sym-div",
    "act-unused-field",
    "act-zero-point-without-sums",
    "weight-opcode",
    "weight-field-value",
    "weight-empty",
    "weight-float-zp-matmul",
    "weight-combination",
    "weight-sums-unused",
    "weight-sums-width",
    "weight-staged-offsets",
    "weight-unused-field",
    "weight-table-bounds",
    "weight-table-external",
    "weight-zero-point-range",
    "out-opcode",
    "out-fused-operand",
    "out-field-value",
    "out-stage-b",
    "out-range",
    "out-scale",
    "out-zero-point",
    "dot-bound",
    "offset-bound",
    "kcache-args",
    "kcache-descriptor",
    "kcache-dot-bound",
    "kcache-zero-points",
    "unimplemented",
};
static_assert(sizeof(kRuleNames) / sizeof(kRuleNames[0]) == kNumRules,
              "one name per rule");

// A statistic, never control flow: two inits on two cores may race on it, and
// the device test reads it only between its own single-threaded loads.
std::atomic<uint32_t> g_last_refusal{kRuleNone};

#define ASYM_REFUSE(rule, code, fmt, ...)                                    \
  do {                                                                       \
    g_last_refusal.store((rule), std::memory_order_relaxed);                 \
    ET_LOG(Error, "esp32: v24 refused [%s] " fmt, kRuleNames[(rule)],        \
           ##__VA_ARGS__);                                                   \
    return (code);                                                           \
  } while (0)

// ---- reading the blob ----------------------------------------------------------

ASYM_COLD OpDescView read_desc(const uint8_t* data, uint32_t i) {
  OpDescView d;
  std::memcpy(&d, data + v23::kHeaderBytes + (size_t)i * v23::kOpDescBytes,
              sizeof(d));
  return d;
}

ASYM_COLD bool is_v24_op(const OpDescView& d) {
  return (d.flags & kV24Flags) != 0 || d.opcode == kOpBmmKcacheAffineI8;
}

ASYM_COLD bool is_v24_record(uint16_t type) {
  return type == kExtActQuant || type == kExtWeightAffine ||
      type == kExtOutQuant;
}

ASYM_COLD bool is_embed(uint32_t opcode) {
  return opcode == v23::kOpEmbedI8 || opcode == v23::kOpEmbedI4 ||
      opcode == v23::kOpEmbedI16;
}

ASYM_COLD bool is_bmm(uint32_t opcode) {
  return opcode == v23::kOpBmmDynI8 || opcode == v23::kOpBmmKcacheI8 ||
      opcode == kOpBmmKcacheAffineI8;
}

struct Record {
  uint16_t op;
  uint16_t type;  // the optional bit cleared, as parse_extension_records does
  uint32_t len;
  const uint8_t* payload;
};

/// The extension table, walked with validate_extensions' bounds checks. A
/// table that check would refuse sets `broken` and is not read further.
struct RecordWalk {
  const uint8_t* data;
  uint64_t size;
  uint32_t n_ops;
  uint32_t left;
  uint64_t at;
  bool broken;
};

ASYM_COLD RecordWalk walk_records(const uint8_t* data, size_t size,
                                  uint32_t n_ops, uint32_t ext_off) {
  RecordWalk w{data, (uint64_t)size, n_ops, 0, 0, false};
  if (ext_off == 0) {
    return w;
  }
  if ((uint64_t)ext_off + v23::kExtCountBytes > w.size) {
    w.broken = true;
    return w;
  }
  std::memcpy(&w.left, data + ext_off, v23::kExtCountBytes);
  w.at = (uint64_t)ext_off + v23::kExtCountBytes;
  return w;
}

ASYM_COLD bool next_record(RecordWalk* w, Record* r) {
  if (w->broken || w->left == 0) {
    return false;
  }
  if (w->at + v23::kExtRecordHeaderBytes > w->size) {
    w->broken = true;
    return false;
  }
  uint16_t type;
  std::memcpy(&r->op, w->data + w->at, 2);
  std::memcpy(&type, w->data + w->at + 2, 2);
  std::memcpy(&r->len, w->data + w->at + 4, 4);
  const uint64_t padded = ((uint64_t)r->len + 3ull) & ~3ull;
  if (w->at + v23::kExtRecordHeaderBytes + padded > w->size ||
      r->op >= w->n_ops) {
    w->broken = true;
    return false;
  }
  r->type = (uint16_t)(type & (uint16_t)~v23::kExtOptional);
  r->payload = w->data + w->at + v23::kExtRecordHeaderBytes;
  w->at += v23::kExtRecordHeaderBytes + padded;
  --w->left;
  return true;
}

/// The first record of `type` on op `op` and how many there are.
ASYM_COLD const uint8_t* find_record(const uint8_t* data, size_t size,
                                     uint32_t n_ops, uint32_t ext_off,
                                     uint32_t op, uint16_t type,
                                     uint32_t* count) {
  RecordWalk w = walk_records(data, size, n_ops, ext_off);
  Record r;
  const uint8_t* first = nullptr;
  *count = 0;
  while (next_record(&w, &r)) {
    if (r.op == op && r.type == type) {
      if (first == nullptr) {
        first = r.payload;
      }
      ++*count;
    }
  }
  return first;
}

ASYM_COLD uint32_t record_bytes(uint16_t type) {
  return type == kExtActQuant ? kActQuantBytes
      : type == kExtWeightAffine ? kWeightAffineBytes
                                 : kOutQuantBytes;
}

ASYM_COLD uint32_t record_flag(uint16_t type) {
  return type == kExtActQuant ? kFlagActQuant
      : type == kExtWeightAffine ? kFlagWeightAffine
                                 : kFlagOutQuant;
}

// ---- what the records mean -------------------------------------------------------

/// The activation can have a nonzero zero point: what widens |xq - zx| to
/// qmax - qmin and makes the op need the weight sums S'.
ASYM_COLD bool act_affine(const ActQuant& a) {
  return a.chooser == kChooserAffineMinMax ||
      (a.granularity == kGranularityStatic && a.zero_point != 0);
}

/// max |xq - zx| over every code the scheme produces (mirrors
/// ActScheme.span). Without a record, the backend's own +-127 rows; 16-bit
/// codes are symmetric and reach 32768 in magnitude.
ASYM_COLD uint32_t act_span(const ActQuant* a) {
  if (a == nullptr) {
    return 127;
  }
  if (a->code_bits == 16) {
    return 32768;
  }
  if (act_affine(*a)) {
    return (uint32_t)(a->qmax - a->qmin);
  }
  const int32_t neg = -a->qmin;
  return (uint32_t)(neg > a->qmax ? neg : a->qmax);
}

ASYM_COLD bool is_4bit(uint32_t opcode) {
  return opcode == v23::kOpAddmmI4 || opcode == v23::kOpEmbedI4;
}

/// max |c - zw| (mirrors.weight_span): int8 storage holds -128 even for
/// symmetric codes; an integer zero point, or a storage bias, widens both.
ASYM_COLD uint32_t weight_span(uint32_t opcode, const WeightAffine* w) {
  const bool zp = w != nullptr &&
      (w->zp_kind == kZpInt8 || w->storage_bias != 0);
  if (is_4bit(opcode)) {
    return zp ? 15 : 8;
  }
  return zp ? 255 : 128;
}

/// The longest run of REAL elements one accumulation covers: a scale group,
/// or the whole row (the k_pad tail is zero on both sides and adds nothing).
ASYM_COLD uint32_t longest_group(const OpDescView& d) {
  if (d.scale_group != 0 && d.scale_group < d.k) {
    return d.scale_group;
  }
  return d.k;
}

/// Groups per row in the op's scale table, which the affine tables share:
/// ceil(span / group), the span being k for a gather and k_pad for a matmul.
ASYM_COLD uint64_t table_groups(const OpDescView& d) {
  const uint64_t span = is_embed(d.opcode) ? d.k : d.k_pad;
  const uint64_t g = d.scale_group != 0 ? d.scale_group : span;
  if (span == 0 || g == 0) {
    return 0;
  }
  return g >= span ? 1 : (span + g - 1) / g;
}

ASYM_COLD uint32_t zp_elem_bytes(uint32_t kind) {
  return kind == kZpInt8 ? 1 : 4;
}
ASYM_COLD uint32_t offset_elem_bytes(uint32_t kind) {
  return kind == kOffsetFp16 ? 2 : 4;
}
ASYM_COLD uint32_t sum_elem_bytes(uint32_t kind) {
  return kind == kSumInt16 ? 2 : 4;
}

ASYM_COLD bool key_ok(const uint8_t* data, uint64_t size, uint64_t off) {
  if (off >= size) {
    return false;
  }
  for (uint64_t t = off; t < size; ++t) {
    if (data[t] == 0) {
      return true;
    }
  }
  return false;
}

// ---- the rules ---------------------------------------------------------------------

ASYM_COLD Error check_act(const OpDescView& d, const ActQuant& a, uint32_t i,
                          bool stage_b) {
  if (d.opcode != v23::kOpAddmmI8 && d.opcode != v23::kOpAddmmI4) {
    ASYM_REFUSE(kRuleActOpcode, Error::InvalidProgram,
                "op %u: an activation quantiser on opcode %u (an int8/int4 "
                "matmul only)", (unsigned)i, (unsigned)d.opcode);
  }
  if (d.flags & (v23::kFlagStaticAct | v23::kFlagActPerTensor)) {
    ASYM_REFUSE(kRuleActSchemeFlags, Error::InvalidProgram,
                "op %u: an activation record beside FLAG_STATIC_ACT or "
                "FLAG_ACT_PER_TENSOR (flags 0x%x) names two schemes",
                (unsigned)i, (unsigned)d.flags);
  }
  if (a.in_dtype > kDtypeInt16 || (a.code_bits != 8 && a.code_bits != 16) ||
      a.granularity > kGranularityStatic || a.chooser > kChooserAffineMinMax ||
      a.rounding != kRoundRneBeforeZp) {
    ASYM_REFUSE(kRuleActFieldValue, Error::InvalidProgram,
                "op %u: activation field out of range (in_dtype %u code_bits "
                "%u granularity %u chooser %u rounding %u)", (unsigned)i,
                (unsigned)a.in_dtype, (unsigned)a.code_bits,
                (unsigned)a.granularity, (unsigned)a.chooser,
                (unsigned)a.rounding);
  }
  // Stage B: int8 codes in, in a delegate built under the
  // "stage_b" compile spec. int16 codes are not part of Stage B.
  if (a.in_dtype != kDtypeFp32 && !(stage_b && a.in_dtype == kDtypeInt8)) {
    ASYM_REFUSE(kRuleActStageB, Error::DelegateInvalidCompatibility,
                "op %u: activation codes in (in_dtype %u) %s", (unsigned)i,
                (unsigned)a.in_dtype,
                a.in_dtype == kDtypeInt8
                    ? "need the stage_b compile spec"
                    : "are int16 codes, which Stage B v1 does not carry");
  }
  if (a.code_bits == 16 &&
      (d.opcode != v23::kOpAddmmI8 || d.scale_group != 0)) {
    ASYM_REFUSE(kRuleActCodeBits, Error::DelegateInvalidCompatibility,
                "op %u: 16-bit activation codes on opcode %u with groups of "
                "%u (an ungrouped int8 matmul only)", (unsigned)i,
                (unsigned)d.opcode, (unsigned)d.scale_group);
  }
  const int32_t lo = a.code_bits == 8 ? -128 : -32768;
  const int32_t hi = a.code_bits == 8 ? 127 : 32767;
  if (a.qmin >= a.qmax || a.qmin < lo || a.qmax > hi) {
    ASYM_REFUSE(kRuleActRange, Error::InvalidProgram,
                "op %u: activation range [%d, %d] is empty or outside "
                "%u-bit codes", (unsigned)i, (int)a.qmin, (int)a.qmax,
                (unsigned)a.code_bits);
  }
  // Which chooser goes with which granularity: a static
  // scheme has its constants and no chooser, a dynamic one a chooser.
  const bool fixed = a.granularity == kGranularityStatic;
  bool known = fixed == (a.chooser == kChooserNone);
  // 16-bit codes are symmetric (Arm a16w8's A6): the 40-bit bound assumes
  // |xq| <= 32768 and no zero point.
  if (known && a.code_bits == 16) {
    known = a.chooser == kChooserSymDiv || (fixed && a.zero_point == 0);
  }
  // int8 codes in are the model's own, made upstream with the record's
  // constants: a static 8-bit scheme, never a chooser.
  if (known && a.in_dtype == kDtypeInt8) {
    known = fixed && a.code_bits == 8;
  }
  if (!known) {
    ASYM_REFUSE(kRuleActCombination, Error::InvalidProgram,
                "op %u: no activation scheme is chooser %u, granularity %u "
                "and rounding %u with %u-bit codes", (unsigned)i,
                (unsigned)a.chooser, (unsigned)a.granularity,
                (unsigned)a.rounding, (unsigned)a.code_bits);
  }
  if (a.chooser == kChooserSymDiv && !(a.qmin < 0 && a.qmax > 0)) {
    // PT2E puts an unsigned symmetric range's zero point at 128, torchao at
    // the midpoint; a zero point of 0 is neither. Refused by name at export.
    ASYM_REFUSE(kRuleActUnsignedSymmetric, Error::InvalidProgram,
                "op %u: a symmetric chooser over [%d, %d], which is not a "
                "signed range around 0", (unsigned)i, (int)a.qmin,
                (int)a.qmax);
  }
  if (fixed) {
    if (!std::isfinite(a.scale) || !(a.scale > 0.0f)) {
      ASYM_REFUSE(kRuleActStaticScale, Error::InvalidProgram,
                  "op %u: static activation scale %g", (unsigned)i,
                  (double)a.scale);
    }
    if (a.zero_point < a.qmin || a.zero_point > a.qmax) {
      ASYM_REFUSE(kRuleActStaticZeroPoint, Error::InvalidProgram,
                  "op %u: static zero point %d outside [%d, %d]", (unsigned)i,
                  (int)a.zero_point, (int)a.qmin, (int)a.qmax);
    }
  }
  if (!std::isfinite(a.eps) || a.eps < 0.0f) {
    ASYM_REFUSE(kRuleActEps, Error::InvalidProgram, "op %u: eps %g",
                (unsigned)i, (double)a.eps);
  }
  if (!fixed && !(a.eps > 0.0f)) {
    // An all-zero row would choose scale 0, and torch's quantisers multiply
    // by fl(1/0).
    ASYM_REFUSE(kRuleActEps, Error::InvalidProgram,
                "op %u: eps 0 under chooser %u lets an all-zero row choose "
                "scale 0", (unsigned)i, (unsigned)a.chooser);
  }
  if (a.chooser == kChooserSymDiv &&
      (!std::isfinite(a.sym_div) || !(a.sym_div > 0.0f))) {
    ASYM_REFUSE(kRuleActSymDiv, Error::InvalidProgram,
                "op %u: symmetric divisor %g", (unsigned)i,
                (double)a.sym_div);
  }
  const bool unused_set =
      (fixed && (a.eps != 0.0f || a.sym_div != 0.0f)) ||
      (a.chooser != kChooserSymDiv && a.sym_div != 0.0f) ||
      (!fixed && (a.scale != 0.0f || a.zero_point != 0));
  if (unused_set) {
    ASYM_REFUSE(kRuleActUnusedField, Error::InvalidProgram,
                "op %u: a field the scheme does not use is set (eps %g "
                "sym_div %g scale %g zero_point %d)", (unsigned)i,
                (double)a.eps, (double)a.sym_div, (double)a.scale,
                (int)a.zero_point);
  }
  return Error::Ok;
}

ASYM_COLD Error check_table(const uint8_t* data, size_t size,
                            const OpDescView& d, uint32_t i, const char* what,
                            uint32_t off, uint32_t elem) {
  if (d.flags & v23::kFlagExternalWeights) {
    if (!key_ok(data, size, off)) {
      ASYM_REFUSE(kRuleWeightTableBounds, Error::InvalidProgram,
                  "op %u: %s table key at %u is not a NUL-terminated string "
                  "inside the blob", (unsigned)i, what, (unsigned)off);
    }
    return Error::Ok;
  }
  const uint64_t bytes = (uint64_t)d.n * table_groups(d) * elem;
  if ((uint64_t)off > (uint64_t)size || bytes > (uint64_t)size - off ||
      off % elem != 0) {
    ASYM_REFUSE(kRuleWeightTableBounds, Error::InvalidProgram,
                "op %u: %s table (%llu B at %u) runs past the blob or is not "
                "%u-byte aligned", (unsigned)i, what,
                (unsigned long long)bytes, (unsigned)off, (unsigned)elem);
  }
  return Error::Ok;
}

ASYM_COLD Error check_weight(const uint8_t* data, size_t size,
                             const OpDescView& d, const WeightAffine& w,
                             const ActQuant* act, uint32_t i) {
  const bool embed = d.opcode == v23::kOpEmbedI8 || d.opcode == v23::kOpEmbedI4;
  if (!embed && d.opcode != v23::kOpAddmmI8 && d.opcode != v23::kOpAddmmI4) {
    ASYM_REFUSE(kRuleWeightOpcode, Error::InvalidProgram,
                "op %u: a weight record on opcode %u (int8/int4 matmuls and "
                "embeddings only; int16 weights stay symmetric)", (unsigned)i,
                (unsigned)d.opcode);
  }
  if (w.zp_kind > kZpFp32 || w.offset_kind > kOffsetFp16 ||
      w.sum_kind > kSumInt32 ||
      (w.storage_bias != 0 && w.storage_bias != 8 && w.storage_bias != 128) ||
      (w.storage_bias == 128 && is_4bit(d.opcode))) {
    ASYM_REFUSE(kRuleWeightFieldValue, Error::InvalidProgram,
                "op %u: weight field out of range (zp_kind %u offset_kind %u "
                "sum_kind %u storage_bias %d)", (unsigned)i,
                (unsigned)w.zp_kind, (unsigned)w.offset_kind,
                (unsigned)w.sum_kind, (int)w.storage_bias);
  }
  if (w.zp_kind == kZpNone && w.offset_kind == kOffsetNone &&
      w.sum_kind == kSumNone && w.storage_bias == 0) {
    ASYM_REFUSE(kRuleWeightEmpty, Error::InvalidProgram,
                "op %u: a weight record describing a symmetric weight, which "
                "carries none", (unsigned)i);
  }
  if (w.zp_kind == kZpFp32 && !embed) {
    ASYM_REFUSE(kRuleWeightFloatZpMatmul, Error::DelegateInvalidCompatibility,
                "op %u: a float zero point on a matmul; the integer dot cannot "
                "absorb a fractional zero point", (unsigned)i);
  }
  if (w.zp_kind == kZpFp32 && w.offset_kind != kOffsetNone) {
    ASYM_REFUSE(kRuleWeightCombination, Error::InvalidProgram,
                "op %u: a float zero point and a float offset together; no "
                "reference computes both", (unsigned)i);
  }
  if (w.sum_kind != kSumNone && (act == nullptr || !act_affine(*act))) {
    ASYM_REFUSE(kRuleWeightSumsUnused, Error::InvalidProgram,
                "op %u: weight sums with no activation zero point to multiply "
                "them (%s)", (unsigned)i,
                embed ? "an embedding" : "symmetric activations");
  }
  if ((d.flags & v23::kFlagStageI8) && w.offset_kind != kOffsetNone) {
    ASYM_REFUSE(kRuleWeightStagedOffsets, Error::DelegateInvalidCompatibility,
                "op %u: float offsets on a weight staged to int8", (unsigned)i);
  }
  if (w.sum_kind == kSumInt16) {
    const uint64_t reach =
        (uint64_t)longest_group(d) * weight_span(d.opcode, &w);
    if (reach > kInt16Max) {
      ASYM_REFUSE(kRuleWeightSumsWidth, Error::InvalidProgram,
                  "op %u: int16 weight sums over groups of %u, where |S'| can "
                  "reach %llu", (unsigned)i, (unsigned)longest_group(d),
                  (unsigned long long)reach);
    }
  }
  if ((w.zp_kind == kZpNone && w.zp_off != 0) ||
      (w.offset_kind == kOffsetNone && w.offset_off != 0) ||
      (w.sum_kind == kSumNone && w.sum_off != 0)) {
    ASYM_REFUSE(kRuleWeightUnusedField, Error::InvalidProgram,
                "op %u: an absent table has an offset (zp %u offsets %u sums "
                "%u)", (unsigned)i, (unsigned)w.zp_off,
                (unsigned)w.offset_off, (unsigned)w.sum_off);
  }
  Error e = Error::Ok;
  if (w.zp_kind != kZpNone) {
    e = check_table(data, size, d, i, "zero-point", w.zp_off,
                    zp_elem_bytes(w.zp_kind));
  }
  if (e == Error::Ok && w.offset_kind != kOffsetNone) {
    e = check_table(data, size, d, i, "offset", w.offset_off,
                    offset_elem_bytes(w.offset_kind));
  }
  if (e == Error::Ok && w.sum_kind != kSumNone) {
    e = check_table(data, size, d, i, "weight-sum", w.sum_off,
                    sum_elem_bytes(w.sum_kind));
  }
  return e;
}

ASYM_COLD Error check_out(const OpDescView& d, const OutQuant& o, uint32_t i,
                          bool stage_b) {
  if (is_embed(d.opcode) || is_bmm(d.opcode)) {
    ASYM_REFUSE(kRuleOutOpcode, Error::InvalidProgram,
                "op %u: an output quantize on opcode %u (matmuls only)",
                (unsigned)i, (unsigned)d.opcode);
  }
  if (d.flags & (v23::kFlagAdd | v23::kFlagMul)) {
    ASYM_REFUSE(kRuleOutFusedOperand, Error::DelegateInvalidCompatibility,
                "op %u: an output quantize after a fused add or multiply "
                "(flags 0x%x) is not part of v1", (unsigned)i,
                (unsigned)d.flags);
  }
  if (o.out_dtype > kDtypeInt16 || o.rounding != kRoundRneBeforeZp) {
    // A quantize rounds to even before its zero point: the one rounding.
    ASYM_REFUSE(kRuleOutFieldValue, Error::InvalidProgram,
                "op %u: output field out of range (out_dtype %u rounding %u)",
                (unsigned)i, (unsigned)o.out_dtype, (unsigned)o.rounding);
  }
  if (o.out_dtype != kDtypeFp32 && !(stage_b && o.out_dtype == kDtypeInt8)) {
    ASYM_REFUSE(kRuleOutStageB, Error::DelegateInvalidCompatibility,
                "op %u: output codes out (out_dtype %u) %s", (unsigned)i,
                (unsigned)o.out_dtype,
                o.out_dtype == kDtypeInt8
                    ? "need the stage_b compile spec"
                    : "are int16 codes, which Stage B v1 does not carry");
  }
  // The fake-quant's codes are never stored, so any range a model's quantize
  // can name -- int8, uint8, int16 -- is one int16 holds; codes written out
  // (Stage B) are int8.
  const int32_t lo = o.out_dtype == kDtypeInt8 ? -128 : -32768;
  const int32_t hi = o.out_dtype == kDtypeInt8 ? 127 : 32767;
  if (o.qmin >= o.qmax || o.qmin < lo || o.qmax > hi) {
    ASYM_REFUSE(kRuleOutRange, Error::InvalidProgram,
                "op %u: output range [%d, %d]", (unsigned)i, (int)o.qmin,
                (int)o.qmax);
  }
  if (!std::isfinite(o.scale) || !(o.scale > 0.0f)) {
    ASYM_REFUSE(kRuleOutScale, Error::InvalidProgram,
                "op %u: output scale %g", (unsigned)i, (double)o.scale);
  }
  if (o.zero_point < o.qmin || o.zero_point > o.qmax) {
    ASYM_REFUSE(kRuleOutZeroPoint, Error::InvalidProgram,
                "op %u: output zero point %d outside [%d, %d]", (unsigned)i,
                (int)o.zero_point, (int)o.qmin, (int)o.qmax);
  }
  return Error::Ok;
}

/// The exactness bounds for an op with an activation or weight
/// record. Symmetric ops keep validate_blob's kMaxDotLen.
ASYM_COLD Error check_bounds(const OpDescView& d, const ActQuant* act,
                             const WeightAffine* w, uint32_t i) {
  const uint64_t longest = longest_group(d);
  const uint64_t aspan = act_span(act);
  const uint64_t wspan = weight_span(d.opcode, w);
  const bool wide = act != nullptr && act->code_bits == 16;
  const uint64_t limit =
      dot_bound(wide ? kAccx40Max : kInt32Max, aspan, wspan);
  if (longest > limit) {
    ASYM_REFUSE(kRuleDotBound, Error::InvalidProgram,
                "op %u: a dot over %llu elements, past the %llu an exact "
                "%s accumulator holds when |x - zx| <= %llu and |c - zw| <= "
                "%llu", (unsigned)i, (unsigned long long)longest,
                (unsigned long long)limit, wide ? "40-bit" : "int32",
                (unsigned long long)aspan, (unsigned long long)wspan);
  }
  if (w != nullptr && w->offset_kind != kOffsetNone &&
      longest * aspan > kFloatExactInt) {
    ASYM_REFUSE(kRuleOffsetBound, Error::InvalidProgram,
                "op %u: float offsets over groups of %llu, where X' can reach "
                "%llu and its float conversion is no longer exact past 2^24",
                (unsigned)i, (unsigned long long)longest,
                (unsigned long long)(longest * aspan));
  }
  return Error::Ok;
}

/// a * b * c <= limit, without overflowing on the way; false for a zero factor.
ASYM_COLD bool product_fits(uint64_t a, uint64_t b, uint64_t c,
                            uint64_t limit) {
  if (a == 0 || b == 0 || c == 0 || a > limit / b) {
    return false;
  }
  return a * b <= limit / c;
}

/// OP_BMM_KCACHE_AFFINE_I8's whole descriptor, before validate_blob checks it
/// as a bmm: its own operands -- five argument indices, distinct and none a
/// chain slot: queries (in_arg), codes (res_arg), scales (scale_off), zero
/// points (wt_off) and output (out_arg) -- and the exactness bound, and what
/// validate_blob checks of every bmm (no flags and no stored bias; a nonzero
/// batch (scale_group) and extents, k <= k_pad, every tensor and the key
/// scratch addressable on the 32-bit target), so that a broken descriptor is
/// refused for its own rule.
ASYM_COLD Error check_kcache(const OpDescView& d, uint32_t i) {
  const uint32_t args[5] = {d.in_arg, d.res_arg, d.scale_off, d.wt_off,
                            d.out_arg};
  bool distinct = true;
  for (int a = 0; a < 5; ++a) {
    distinct = distinct && args[a] < v23::kArgScratch0;
    for (int b = 0; b < a; ++b) {
      distinct = distinct && args[a] != args[b];
    }
  }
  if (!distinct) {
    ASYM_REFUSE(kRuleKcacheArgs, Error::InvalidProgram,
                "op %u: affine kcache bmm needs five distinct argument "
                "indices outside the chain slots (queries %u, codes %u, "
                "scales %u, zero points %u, output %u)", (unsigned)i,
                (unsigned)d.in_arg, (unsigned)d.res_arg,
                (unsigned)d.scale_off, (unsigned)d.wt_off,
                (unsigned)d.out_arg);
  }
  // As validate_blob's bmm extents: the queries [B,M,K], the codes [B,N,K]
  // and the output [B,M,N] are float-addressable, the key scratch [N,k_pad]
  // byte-addressable.
  constexpr uint64_t kMaxFloatElems = 0xFFFFFFFFull / sizeof(float);
  const uint64_t b = d.scale_group, m = d.m, n = d.n, k = d.k;
  if (d.flags != 0 || d.bias_off != 0 || k > d.k_pad ||
      !product_fits(b, m, k, kMaxFloatElems) ||
      !product_fits(b, n, k, kMaxFloatElems) ||
      !product_fits(b, m, n, kMaxFloatElems) ||
      !product_fits(n, d.k_pad, 1, 0xFFFFFFFFull)) {
    ASYM_REFUSE(kRuleKcacheDescriptor, Error::InvalidProgram,
                "op %u: affine kcache bmm with flags 0x%x, bias_off %u, "
                "batch %u, m %u, k %u, n %u, k_pad %u", (unsigned)i,
                (unsigned)d.flags, (unsigned)d.bias_off,
                (unsigned)d.scale_group, (unsigned)d.m, (unsigned)d.k,
                (unsigned)d.n, (unsigned)d.k_pad);
  }
  // The dot runs over k_pad (a zero tail), so k_pad is what is bounded.
  uint64_t limit = kInt32Max / ((uint64_t)kKcacheQuerySpan * kKcacheKeySpan);
  limit -= limit % 32;
  if ((uint64_t)d.k_pad > limit) {
    ASYM_REFUSE(kRuleKcacheDotBound, Error::InvalidProgram,
                "op %u: affine kcache rows of %u elements, past the %llu an "
                "exact int32 dot holds", (unsigned)i, (unsigned)d.k_pad,
                (unsigned long long)limit);
  }
  return Error::Ok;
}

ASYM_COLD bool any_v24(const uint8_t* data, size_t size, uint32_t n_ops,
                       uint32_t ext_off, bool* table_broken) {
  bool any = false;
  for (uint32_t i = 0; i < n_ops && !any; ++i) {
    any = is_v24_op(read_desc(data, i));
  }
  RecordWalk w = walk_records(data, size, n_ops, ext_off);
  Record r;
  while (next_record(&w, &r)) {
    any = any || is_v24_record(r.type);
  }
  *table_broken = w.broken;
  return any;
}

}  // namespace

// ---- the interface -------------------------------------------------------------------

ASYM_COLD Error validate(const uint8_t* data, size_t size, uint32_t n_ops,
                         uint32_t ext_off, bool stage_b) {
  const uint64_t descs_end =
      v23::kHeaderBytes + (uint64_t)v23::kOpDescBytes * n_ops;
  if (data == nullptr || n_ops > 0xFFFFu || descs_end > (uint64_t)size) {
    return Error::Ok;  // left for validate_blob to refuse
  }
  bool broken = false;
  if (!any_v24(data, size, n_ops, ext_off, &broken) || broken) {
    // No asymmetric content, or a table validate_extensions refuses.
    return Error::Ok;
  }
  // Each record: its size, its flag. Duplicates are found per op below.
  RecordWalk walk = walk_records(data, size, n_ops, ext_off);
  Record r;
  while (next_record(&walk, &r)) {
    if (!is_v24_record(r.type)) {
      continue;
    }
    if (r.len != record_bytes(r.type)) {
      ASYM_REFUSE(kRuleRecordSize, Error::InvalidProgram,
                  "op %u: record type %u is %u B, expected %u", (unsigned)r.op,
                  (unsigned)r.type, (unsigned)r.len,
                  (unsigned)record_bytes(r.type));
    }
    const OpDescView d = read_desc(data, r.op);
    if (!(d.flags & record_flag(r.type))) {
      ASYM_REFUSE(kRuleRecordWithoutFlag, Error::InvalidProgram,
                  "op %u: record type %u without its flag 0x%x (flags 0x%x)",
                  (unsigned)r.op, (unsigned)r.type,
                  (unsigned)record_flag(r.type), (unsigned)d.flags);
    }
  }
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpDescView d = read_desc(data, i);
    if (!is_v24_op(d)) {
      continue;
    }
    ActQuant act;
    WeightAffine weight;
    OutQuant out;
    bool has[3] = {false, false, false};
    const uint16_t types[3] = {kExtActQuant, kExtWeightAffine, kExtOutQuant};
    void* dst[3] = {&act, &weight, &out};
    for (int t = 0; t < 3; ++t) {
      uint32_t count = 0;
      const uint8_t* payload =
          find_record(data, size, n_ops, ext_off, i, types[t], &count);
      if (count > 1) {
        ASYM_REFUSE(kRuleDuplicateRecord, Error::InvalidProgram,
                    "op %u: %u records of type %u; there is no rule for which "
                    "one wins", (unsigned)i, (unsigned)count,
                    (unsigned)types[t]);
      }
      if ((d.flags & record_flag(types[t])) && payload == nullptr) {
        ASYM_REFUSE(kRuleFlagWithoutRecord, Error::InvalidProgram,
                    "op %u: flag 0x%x without its record type %u",
                    (unsigned)i, (unsigned)record_flag(types[t]),
                    (unsigned)types[t]);
      }
      if (payload != nullptr) {
        std::memcpy(dst[t], payload, record_bytes(types[t]));
        has[t] = true;
      }
    }
    const ActQuant* a = has[0] ? &act : nullptr;
    const WeightAffine* w = has[1] ? &weight : nullptr;
    Error e = Error::Ok;
    if (d.opcode == kOpBmmKcacheAffineI8) {
      e = check_kcache(d, i);
    }
    if (e == Error::Ok && a != nullptr) {
      e = check_act(d, *a, i, stage_b);
    }
    if (e == Error::Ok && w != nullptr) {
      e = check_weight(data, size, d, *w, a, i);
    }
    if (e == Error::Ok && a != nullptr && act_affine(*a) &&
        (w == nullptr || w->sum_kind == kSumNone)) {
      ASYM_REFUSE(kRuleActZeroPointWithoutSums, Error::InvalidProgram,
                  "op %u: an activation zero point with no weight sums S' to "
                  "correct the dot with", (unsigned)i);
    }
    if (e == Error::Ok && has[2]) {
      e = check_out(d, out, i, stage_b);
    }
    if (e == Error::Ok && (a != nullptr || w != nullptr) &&
        !is_embed(d.opcode)) {
      e = check_bounds(d, a, w, i);
    }
    if (e != Error::Ok) {
      return e;
    }
  }
  return Error::Ok;
}

ASYM_COLD Error attach(BackendInitContext& context, const uint8_t* data,
                       size_t size, uint32_t n_ops, uint32_t ext_off,
                       BlobState** state) {
  *state = nullptr;
  bool broken = false;
  if (!any_v24(data, size, n_ops, ext_off, &broken)) {
    return Error::Ok;  // a symmetric blob: no state, nothing allocated
  }
  MemoryAllocator* allocator = context.get_runtime_allocator();
  auto* st = allocator->allocateInstance<BlobState>();
  if (st == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  st->ops = allocator->allocateList<OpQuant>(n_ops);
  st->held = nullptr;
  st->n_held = 0;
  st->terms_off = kNoTerms;
  if (st->ops == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  *state = st;
  for (uint32_t i = 0; i < n_ops; ++i) {
    st->ops[i] = OpQuant{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         false, 0.0f};
  }
  // Decode every record (validate has checked each size and flag).
  uint32_t n_external = 0;
  RecordWalk walk = walk_records(data, size, n_ops, ext_off);
  Record r;
  while (next_record(&walk, &r)) {
    OpQuant& q = st->ops[r.op];
    if (r.type == kExtActQuant) {
      auto* a = allocator->allocateInstance<ActQuant>();
      if (a == nullptr) {
        return Error::MemoryAllocationFailed;
      }
      std::memcpy(a, r.payload, kActQuantBytes);
      q.act = a;
    } else if (r.type == kExtWeightAffine) {
      auto* w = allocator->allocateInstance<WeightAffine>();
      if (w == nullptr) {
        return Error::MemoryAllocationFailed;
      }
      std::memcpy(w, r.payload, kWeightAffineBytes);
      q.weight = w;
      if (read_desc(data, r.op).flags & v23::kFlagExternalWeights) {
        n_external += (w->zp_kind != kZpNone) + (w->offset_kind != kOffsetNone) +
            (w->sum_kind != kSumNone);
      }
    } else if (r.type == kExtOutQuant) {
      auto* o = allocator->allocateInstance<OutQuant>();
      if (o == nullptr) {
        return Error::MemoryAllocationFailed;
      }
      std::memcpy(o, r.payload, kOutQuantBytes);
      q.out = o;
      q.out_inv = 1.0f / o->scale;  // validate: finite and positive
    }
  }
  if (n_external != 0) {
    st->held = allocator->allocateList<FreeableBuffer>(n_external);
    if (st->held == nullptr) {
      return Error::MemoryAllocationFailed;
    }
  }
  const NamedDataMap* ndm = context.get_named_data_map();
  for (uint32_t i = 0; i < n_ops; ++i) {
    OpQuant& q = st->ops[i];
    if (q.weight == nullptr) {
      continue;
    }
    const OpDescView d = read_desc(data, i);
    const WeightAffine& w = *q.weight;
    const uint64_t entries = (uint64_t)d.n * table_groups(d);
    const uint32_t offs[3] = {w.zp_off, w.offset_off, w.sum_off};
    const bool present[3] = {w.zp_kind != kZpNone, w.offset_kind != kOffsetNone,
                             w.sum_kind != kSumNone};
    const uint32_t elems[3] = {zp_elem_bytes(w.zp_kind),
                               offset_elem_bytes(w.offset_kind),
                               sum_elem_bytes(w.sum_kind)};
    const void** where[3] = {&q.zero_points, &q.offsets, &q.sums};
    for (int t = 0; t < 3; ++t) {
      if (!present[t]) {
        continue;
      }
      if (!(d.flags & v23::kFlagExternalWeights)) {
        *where[t] = data + offs[t];
        continue;
      }
      const char* key = reinterpret_cast<const char*>(data + offs[t]);
      if (ndm == nullptr) {
        ASYM_REFUSE(kRuleWeightTableExternal, Error::InvalidArgument,
                    "op %u: external table '%s' but no NamedDataMap was "
                    "supplied to load_method", (unsigned)i, key);
      }
      auto got = ndm->get_data(key);
      if (!got.ok()) {
        ASYM_REFUSE(kRuleWeightTableExternal, got.error(),
                    "op %u: NamedDataMap has no '%s' (%d)", (unsigned)i, key,
                    (int)got.error());
      }
      FreeableBuffer* held =
          new (&st->held[st->n_held++]) FreeableBuffer(std::move(got.get()));
      if ((uint64_t)held->size() < entries * elems[t] ||
          reinterpret_cast<uintptr_t>(held->data()) % elems[t] != 0) {
        ASYM_REFUSE(kRuleWeightTableExternal, Error::InvalidExternalData,
                    "op %u: NamedDataMap entry '%s' is %u B or misaligned; "
                    "the table needs %llu B of %u-byte elements", (unsigned)i,
                    key, (unsigned)held->size(),
                    (unsigned long long)(entries * elems[t]),
                    (unsigned)elems[t]);
      }
      *where[t] = held->data();
    }
    // The tables are small: check every zero point once, here.
    if (w.zp_kind == kZpInt8 && is_4bit(d.opcode)) {
      const int8_t* zp = static_cast<const int8_t*>(q.zero_points);
      for (uint64_t e = 0; e < entries; ++e) {
        if (zp[e] < -8 || zp[e] > 7) {
          ASYM_REFUSE(kRuleWeightZeroPointRange, Error::InvalidProgram,
                      "op %u: int4 zero point %d at entry %llu outside "
                      "[-8, 7]", (unsigned)i, (int)zp[e],
                      (unsigned long long)e);
        }
      }
    } else if (w.zp_kind == kZpFp32) {
      for (uint64_t e = 0; e < entries; ++e) {
        float z;
        std::memcpy(&z, static_cast<const uint8_t*>(q.zero_points) + 4 * e,
                    sizeof(z));
        if (!std::isfinite(z)) {
          ASYM_REFUSE(kRuleWeightZeroPointRange, Error::InvalidProgram,
                      "op %u: float zero point %g at entry %llu", (unsigned)i,
                      (double)z, (unsigned long long)e);
        }
      }
    }
  }
  // Everything above holds. Weight records are computed, and so are these
  // activation records: 8-bit codes in from fp32, rounded to even before the
  // zero point -- dynamic per row or per tensor (the symmetric or the affine
  // min/max chooser; per tensor only on a conv, whose rows are gathers), or
  // static, the model's own constants -- and static 16-bit codes with zero
  // point 0 on an ungrouped int8 matmul or conv (validate's act-code-bits
  // rule) with symmetric weights and no selected-row walk: the int16 loop's.
  // Output records are computed on the int8/int4 matmuls and convs, whose
  // loops are the affine ones, and the affine K cache has its whole
  // descriptor checked by validate(). Stage B's int8 codes in and out
  // (validate admits them only under the stage_b compile spec) run in the
  // same loops. Dynamic 16-bit codes, a per-row record on a conv, output
  // records on other opcodes and codes on a selected-row head or the int16
  // loop are not implemented, and refused here as such.
  // (validate has held every record to its one rounding and to a chooser
  // that matches its granularity.)
  for (uint32_t i = 0; i < n_ops; ++i) {
    const OpQuant& q = st->ops[i];
    const OpDescView d = read_desc(data, i);
    const ActQuant* a = q.act;
    const bool dynamic_done =
        (a != nullptr) &&
        (a->granularity == kGranularityTensor ||
         (a->granularity == kGranularityRow && !(d.flags & v23::kFlagConv)));
    const bool static_done =
        (a != nullptr) && a->granularity == kGranularityStatic;
    bool wide_done = false;
    if (static_done && a->code_bits == 16 && a->zero_point == 0 &&
        q.weight == nullptr) {
      uint32_t selected = 0;
      find_record(data, size, n_ops, ext_off, i, v23::kExtSelectedRows,
                  &selected);
      wide_done = selected == 0;
    }
    const bool act_done = a == nullptr ||
        (a->code_bits == 8 && (dynamic_done || static_done)) || wide_done;
    const bool out_done = q.out == nullptr ||
        d.opcode == v23::kOpAddmmI8 || d.opcode == v23::kOpAddmmI4;
    // Stage B's codes run in the affine row, block and depthwise
    // walks: not a selected-row head's, which reads floats, nor the int16
    // loop's.
    bool codes_done = true;
    if ((a != nullptr && a->in_dtype == kDtypeInt8) ||
        (q.out != nullptr && q.out->out_dtype == kDtypeInt8)) {
      uint32_t selected = 0;
      find_record(data, size, n_ops, ext_off, i, v23::kExtSelectedRows,
                  &selected);
      codes_done = selected == 0 && (a == nullptr || a->code_bits == 8);
    }
    if (!act_done || !out_done || !codes_done) {
      ASYM_REFUSE(kRuleUnimplemented, Error::DelegateInvalidCompatibility,
                  "op %u: valid v24 content (activation %s, weight %s, output "
                  "%s, opcode %u) that this runtime does not compute yet",
                  (unsigned)i, q.act ? "record" : "none",
                  q.weight ? "record" : "none", q.out ? "record" : "none",
                  (unsigned)d.opcode);
    }
  }
  return Error::Ok;
}

ASYM_COLD void release(BlobState* state) {
  if (state == nullptr || state->held == nullptr) {
    return;
  }
  // The arena will not run these, and a FreeableBuffer may own a mapping.
  for (uint32_t i = 0; i < state->n_held; ++i) {
    state->held[i].~FreeableBuffer();
  }
  state->n_held = 0;
  state->held = nullptr;
}

ASYM_COLD Error check_kcache_zero_points(const Tensor& zero_points,
                                         uint32_t batch, uint32_t n,
                                         uint32_t op_index) {
  if (zero_points.scalar_type() != ScalarType::Char ||
      zero_points.numel() != (int64_t)((uint64_t)batch * n)) {
    ASYM_REFUSE(kRuleKcacheZeroPoints, Error::InvalidArgument,
                "op %u: affine kcache zero points must be int8, %u x %u; got "
                "dtype %d with %lld elements", (unsigned)op_index,
                (unsigned)batch, (unsigned)n,
                (int)zero_points.scalar_type(),
                (long long)zero_points.numel());
  }
  return Error::Ok;
}

ASYM_COLD uint32_t last_refusal() {
  return g_last_refusal.load(std::memory_order_relaxed);
}

ASYM_COLD void clear_last_refusal() {
  g_last_refusal.store(kRuleNone, std::memory_order_relaxed);
}

ASYM_COLD const char* rule_name(uint32_t rule) {
  return rule < kNumRules ? kRuleNames[rule] : "?";
}

// ---- weight zero points and float offsets ---------------------------------------
// Not cold: these run per output element of an affine op. Every float
// operation below is written as the rounding sequence it must be, and this
// file's -ffp-contract=off keeps each one a separate IEEE operation.

namespace {

/// The int32 dot of `len` codes: dot_i8_s3 when the caller's PIE gate passed
/// (16-byte aligned rows and group starts), else the same sum in a scalar
/// loop. Integer accumulation, so the two are bit-identical (the kernel keeps
/// ACCX's low 32 bits; the scalar loop wraps the same way in uint32).
inline int32_t dot_codes(const int8_t* x, const int8_t* w, uint32_t len,
                         bool pie) {
  if (pie) {
    return dot_i8_s3(x, w, (int)len);
  }
  uint32_t acc = 0;
  for (uint32_t t = 0; t < len; ++t) {
    acc += (uint32_t)((int32_t)x[t] * (int32_t)w[t]);
  }
  return (int32_t)acc;
}

/// raw - zero * xsum in wrapping uint32: exact whenever the true group dot fits
/// int32, which the exactness bound (check_bounds) guarantees.
inline int32_t corrected(int32_t raw, int32_t zero, int32_t xsum) {
  return (int32_t)((uint32_t)raw - (uint32_t)zero * (uint32_t)xsum);
}

/// The activation zero point's term, acc - zx * S'[n, g], wrapping like
/// corrected(); nothing for an op without weight sums (zx is 0 there).
inline int32_t less_zx_sums(const AffineWeights& w, size_t entry, int32_t acc,
                            int32_t zx) {
  if (w.sums == nullptr) {
    return acc;
  }
  const int32_t sum = w.sums_i16
      ? (int32_t) static_cast<const int16_t*>(w.sums)[entry]
      : static_cast<const int32_t*>(w.sums)[entry];
  return (int32_t)((uint32_t)acc - (uint32_t)zx * (uint32_t)sum);
}

/// Real elements of group g: n_g = |[a, b) n [0, k)|, the count X'_g = X_g -
/// n_g * zx takes off (the k_pad tail holds no zx).
inline int32_t real_count(const AffineWeights& w, uint32_t a, uint32_t b) {
  const uint32_t end = b < w.k ? b : w.k;
  return end > a ? (int32_t)(end - a) : 0;
}

/// T_g = fl(fl(X'_g) * sx): a float offset's activation side. X'_g converts
/// exactly (|X'_g| <= 2^24, the offset-bound rule).
inline float offset_term(int32_t xsum, int32_t count, int32_t zx, float sx) {
  return (float)(xsum - count * zx) * sx;
}

/// One group's float terms onto y, in mirrors.device_epilogue's order
/// (fused_madd=False): the scale term, then the float offset's.
inline float add_group(const AffineWeights& w, size_t entry, int32_t acc,
                       float sx, const float* toff, uint32_t g, float y) {
  const float term = (float)acc * sx;
  y = y + term * table_float(w.scales, w.fp16_scales, entry);
  if (w.offsets != nullptr) {
    y = y + table_float(w.offsets, w.offsets_fp16, entry) * toff[g];
  }
  return y;
}

inline int32_t group_zero(const AffineWeights& w, size_t entry) {
  return w.zero_points != nullptr ? (int32_t)w.zero_points[entry] : w.zw;
}

}  // namespace

AffineWeights affine_weights(const OpQuant& q, uint32_t k, uint32_t k_pad,
                             uint32_t scale_group, bool fp16_scales,
                             const void* scales) {
  AffineWeights w;
  w.k = k;
  w.k_pad = k_pad;
  w.gsize = scale_group != 0 ? scale_group : k_pad;
  w.ngw = (k_pad + w.gsize - 1) / w.gsize;
  w.fp16_scales = fp16_scales;
  w.scales = scales;
  w.zero_points = nullptr;
  w.zw = 0;
  w.offsets = nullptr;
  w.offsets_fp16 = false;
  w.sums = nullptr;
  w.sums_i16 = false;
  if (q.weight == nullptr) {
    return w;  // symmetric codes under the op's own activation record
  }
  const WeightAffine& wa = *q.weight;
  w.offsets = wa.offset_kind != kOffsetNone ? q.offsets : nullptr;
  w.offsets_fp16 = wa.offset_kind == kOffsetFp16;
  w.sums = wa.sum_kind != kSumNone ? q.sums : nullptr;
  w.sums_i16 = wa.sum_kind == kSumInt16;
  if (q.zp_folded) {
    w.zero_points = nullptr;
    w.zw = 0;
  } else if (wa.zp_kind == kZpInt8) {
    w.zero_points = static_cast<const int8_t*>(q.zero_points);
    w.zw = 0;
  } else {
    // No table: the source had no zero point, so the stored codes' is
    // -storage_bias (GGUF's fl(q * d) stores q - 8 with zero point -8).
    w.zero_points = nullptr;
    w.zw = -wa.storage_bias;
  }
  return w;
}

float accumulate(const AffineWeights& w, uint32_t n, const int8_t* xq,
                 const int8_t* wrow, bool pie, float sx, int32_t zx,
                 const int32_t* xsum, const float* toff, float y) {
  const size_t base = (size_t)n * w.ngw;
  for (uint32_t g = 0; g < w.ngw; ++g) {
    const uint32_t a = g * w.gsize;
    const uint32_t len = (a + w.gsize <= w.k_pad) ? w.gsize : (w.k_pad - a);
    const int32_t raw = dot_codes(xq + a, wrow + a, len, pie);
    const int32_t acc = less_zx_sums(
        w, base + g, corrected(raw, group_zero(w, base + g), xsum[g]), zx);
    y = add_group(w, base + g, acc, sx, toff, g, y);
  }
  return y;
}

float accumulate_fused(const AffineWeights& w, uint32_t n, const int32_t* raw,
                       float sx, int32_t zx, const int32_t* xsum,
                       const float* toff, float y) {
  const size_t base = (size_t)n * w.ngw;
  for (uint32_t g = 0; g < w.ngw; ++g) {
    const int32_t acc = less_zx_sums(
        w, base + g, corrected(raw[g], 8 + group_zero(w, base + g), xsum[g]),
        zx);
    y = add_group(w, base + g, acc, sx, toff, g, y);
  }
  return y;
}

// ---- the backend's own quantiser (A0) ---------------------------------------------
// An op without an activation record quantises as esp32_backend.cpp's
// quantize_row_i8 does, and so do the affine K cache's queries; both are
// computed here (quantize_row_a0 and quantize_conv_row_a0 below,
// kcache_query_row), bit for bit: each float operation is the backend's, a
// rounding of its own (this file's -ffp-contract=off; there, the compare on
// fl(x * inv) keeps GCC from fusing it with the +0.5f).

namespace {

/// max |x| over n activations, from 0.
inline float abs_max(const float* x, uint32_t n) {
  float amax = 0.0f;
  for (uint32_t t = 0; t < n; ++t) {
    const float a = std::fabs(x[t]);
    if (a > amax) {
      amax = a;
    }
  }
  return amax;
}

}  // namespace

// ---- the affine K cache ----------------------------------------------------------
// esp32_asym.h states the arithmetic. Not cold: every layer of a model with
// ExecuTorch's stock quantised KV cache runs it on every decode step.

namespace {

std::atomic<uint32_t> g_kcache_ops{0};
std::atomic<uint64_t> g_kcache_rows{0};

/// The tensor at argument `index`, or null past the arguments or for a
/// non-tensor argument.
const Tensor* arg_tensor(Span<EValue*> args, uint32_t index) {
  if (index >= args.size() || args[index] == nullptr ||
      !args[index]->isTensor()) {
    return nullptr;
  }
  return &args[index]->toTensor();
}

}  // namespace

float kcache_query_row(const float* x, uint32_t k, uint32_t k_pad, int8_t* qq,
                       int32_t* qsum) {
  const float sq = abs_max(x, k) / 127.0f;
  const float inv = a0_inverse(sq);
  int32_t sum = 0;
  for (uint32_t t = 0; t < k; ++t) {
    // Rounded to float before the +0.5f: two roundings, as the mirror's
    // quantize_half_away.
    const int32_t q = a0_code(x[t], inv, false);
    qq[t] = (int8_t)q;
    sum += q;
  }
  for (uint32_t t = k; t < k_pad; ++t) {
    qq[t] = 0;
  }
  *qsum = sum;
  return sq;
}

float kcache_score(const int8_t* qq, int32_t qsum, float sq, const int8_t* kq,
                   uint32_t k_pad, float sk, int8_t zk, bool pie) {
  const int32_t acc = corrected(dot_codes(qq, kq, k_pad, pie), zk, qsum);
  const float term = (float)acc * sq;
  return term * sk;
}

Error run_kcache_affine(uint32_t op_index, const OpDescView& d,
                        Span<EValue*> args, int8_t* qq, int8_t* staged,
                        bool force_scalar) {
  const uint32_t batch = d.scale_group, m = d.m, k = d.k, n = d.n;
  const Tensor* q_t = arg_tensor(args, d.in_arg);
  const Tensor* c_t = arg_tensor(args, d.res_arg);
  const Tensor* s_t = arg_tensor(args, d.scale_off);
  const Tensor* z_t = arg_tensor(args, d.wt_off);
  const Tensor* o_t = arg_tensor(args, d.out_arg);
  const bool ragged = k != d.k_pad;
  if (q_t == nullptr || c_t == nullptr || s_t == nullptr || z_t == nullptr ||
      o_t == nullptr || qq == nullptr || (ragged && staged == nullptr)) {
    ET_LOG(Error, "esp32: affine kcache op %u: an argument is missing or not "
                  "a tensor, or no scratch", (unsigned)op_index);
    return Error::InvalidArgument;
  }
  if (q_t->scalar_type() != ScalarType::Float ||
      c_t->scalar_type() != ScalarType::Char ||
      s_t->scalar_type() != ScalarType::Float ||
      o_t->scalar_type() != ScalarType::Float ||
      q_t->numel() != (int64_t)((uint64_t)batch * m * k) ||
      c_t->numel() != (int64_t)((uint64_t)batch * n * k) ||
      s_t->numel() != (int64_t)((uint64_t)batch * n) ||
      o_t->numel() != (int64_t)((uint64_t)batch * m * n)) {
    ET_LOG(Error, "esp32: affine kcache op %u wants fp32 queries [%u,%u,%u], "
                  "int8 codes [%u,%u,%u], fp32 scales and an fp32 output",
           (unsigned)op_index, (unsigned)batch, (unsigned)m, (unsigned)k,
           (unsigned)batch, (unsigned)n, (unsigned)k);
    return Error::InvalidArgument;
  }
  if (Error e = check_kcache_zero_points(*z_t, batch, n, op_index);
      e != Error::Ok) {
    return e;
  }
  const float* qa = q_t->const_data_ptr<float>();
  const int8_t* kc = c_t->const_data_ptr<int8_t>();
  const float* ks = s_t->const_data_ptr<float>();
  const int8_t* kz = z_t->const_data_ptr<int8_t>();
  float* yo = o_t->mutable_data_ptr<float>();
  for (uint32_t b = 0; b < batch; ++b) {
    const int8_t* rows = kc + (size_t)b * n * k;
    if (ragged) {
      for (uint32_t r = 0; r < n; ++r) {
        std::memcpy(staged + (size_t)r * d.k_pad, rows + (size_t)r * k, k);
        std::memset(staged + (size_t)r * d.k_pad + k, 0, d.k_pad - k);
      }
      rows = staged;
    }
    // Every row starts 16-aligned iff the first does: the stride is k_pad.
    const bool pie = !force_scalar && (d.k_pad % kKcacheRowAlign) == 0 &&
        (reinterpret_cast<uintptr_t>(rows) % kKcacheRowAlign) == 0 &&
        (reinterpret_cast<uintptr_t>(qq) % kKcacheRowAlign) == 0;
    const float* sk = ks + (size_t)b * n;
    const int8_t* zk = kz + (size_t)b * n;
    for (uint32_t r = 0; r < m; ++r) {
      int32_t qsum = 0;
      const float sq = kcache_query_row(qa + ((size_t)b * m + r) * k, k,
                                        d.k_pad, qq, &qsum);
      float* yrow = yo + ((size_t)b * m + r) * n;
      for (uint32_t c = 0; c < n; ++c) {
        yrow[c] = kcache_score(qq, qsum, sq, rows + (size_t)c * d.k_pad,
                               d.k_pad, sk[c], zk[c], pie);
      }
    }
  }
  g_kcache_rows.fetch_add((uint64_t)batch * n, std::memory_order_relaxed);
  g_kcache_ops.fetch_add(1, std::memory_order_relaxed);
  return Error::Ok;
}

uint32_t kcache_affine_ops() {
  return g_kcache_ops.load(std::memory_order_relaxed);
}

uint64_t kcache_affine_rows() {
  return g_kcache_rows.load(std::memory_order_relaxed);
}

// ---- dynamic and static activations, output quantisation ------------------------
// The records' quantisers, operation for operation as torch computes them
// (esp32_asym.h). Not cold: they run per element of an affine op.

namespace {

/// Group g's X_g, and T_g when the op has float offsets, once its codes are in.
inline void close_group(const AffineWeights& w, uint32_t g, int32_t sum,
                        ActParams p, int32_t* xsum, float* toff) {
  xsum[g] = sum;
  if (toff != nullptr) {
    const uint32_t a = g * w.gsize;
    const uint32_t b = (a + w.gsize <= w.k_pad) ? a + w.gsize : w.k_pad;
    toff[g] = offset_term(sum, real_count(w, a, b), p.zx, p.sx);
  }
}

/// The gather's input index for output position `o` and tap `kk` on one axis,
/// or -1: esp32_backend.cpp's axis_index. A transposed conv is the dense conv,
/// taps flipped, over a zero-stuffed input, so a tap at
/// o + kk*dil - ((ksize-1)*dil - pad) reads a real sample only on the
/// stride's lattice; off it, as outside the input, the tap reads 0.
inline int32_t tap_index(bool transposed, uint32_t o, uint32_t kk,
                         uint32_t stride, uint32_t pad, uint32_t dil,
                         uint32_t extent, uint32_t ksize) {
  if (transposed) {
    // z = up - back in unsigned 32-bit arithmetic, as axis_index computes
    // it: the fields are uint16 on the wire, so neither term reaches 2^32,
    // and z is negative exactly when back > up.
    const uint32_t up = o + kk * dil + pad;
    const uint32_t back = (ksize - 1) * dil;
    if (up < back || (up - back) % stride) {
      return -1;
    }
    const uint32_t idx = (up - back) / stride;
    return idx < extent ? (int32_t)idx : -1;
  }
  const int64_t idx = (int64_t)o * stride - (int64_t)pad + (int64_t)kk * dil;
  return (idx >= 0 && idx < (int64_t)extent) ? (int32_t)idx : -1;
}

/// Every tap of output position `row`'s patch in the gather's column order
/// (c, d, i, j): visit(p) with p the input sample it reads, or null for a tap
/// outside the input or off a transposed conv's lattice (a real 0.0). T is
/// float for values, int8_t for Stage B's codes.
template <typename T, typename Visit>
inline void for_each_tap(const T* x, const ConvGather& g, uint32_t row,
                         Visit visit) {
  const uint32_t hw = g.h_out * g.w_out;
  const uint32_t od = row / hw;
  const uint32_t rem = row - od * hw;
  const uint32_t oh = rem / g.w_out;
  const uint32_t ow = rem - oh * g.w_out;
  const size_t plane_hw = (size_t)g.h_in * g.w_in;
  const size_t plane = (size_t)g.d_in * plane_hw;
  const bool tr = g.transposed != 0;
  for (uint32_t c = 0; c < g.c_in; ++c) {
    const T* xc = x + (size_t)c * plane;
    for (uint32_t dd = 0; dd < g.kd; ++dd) {
      const int32_t idd = tap_index(tr, od, dd, g.stride_d, g.pad_d, g.dil_d,
                                    g.d_in, g.kd);
      for (uint32_t i = 0; i < g.kh; ++i) {
        const int32_t ih = tap_index(tr, oh, i, g.stride_h, g.pad_h, g.dil_h,
                                     g.h_in, g.kh);
        const T* xr = (idd >= 0 && ih >= 0)
            ? xc + (size_t)idd * plane_hw + (size_t)ih * g.w_in
            : nullptr;
        for (uint32_t j = 0; j < g.kw; ++j) {
          const int32_t iw = tap_index(tr, ow, j, g.stride_w, g.pad_w,
                                       g.dil_w, g.w_in, g.kw);
          visit(xr != nullptr && iw >= 0 ? xr + iw : nullptr);
        }
      }
    }
  }
}

/// One matmul row's codes into dst at p: code(x[t]) for the k real columns, 0
/// for the k_pad tail, each group's X_g (and T_g) as its codes go in. T is
/// float for values, int8_t for Stage B's codes.
template <typename T, typename Code>
inline void quantize_groups(const T* x, uint32_t k, const AffineWeights& w,
                            ActParams p, Code code, int8_t* dst,
                            int32_t* xsum, float* toff) {
  for (uint32_t g = 0; g < w.ngw; ++g) {
    const uint32_t lo = g * w.gsize;
    const uint32_t hi = (lo + w.gsize <= w.k_pad) ? lo + w.gsize : w.k_pad;
    const uint32_t real = hi < k ? hi : (lo > k ? lo : k);
    int32_t sum = 0;
    for (uint32_t t = lo; t < real; ++t) {
      const int32_t q = code(x[t]);
      dst[t] = (int8_t)q;
      sum += q;
    }
    for (uint32_t t = real; t < hi; ++t) {
      dst[t] = 0;
    }
    close_group(w, g, sum, p, xsum, toff);
  }
}

/// The conv gather's codes at p, in column order: code(*v) for a tap inside
/// the input, `pad` for one outside it or off a transposed conv's lattice, 0
/// for the k_pad tail; each group's X_g (and T_g) closes as its last column
/// is written, the tail's zeros included.
template <typename T, typename Code>
inline void quantize_gather(const T* x, const ConvGather& g, uint32_t row,
                            const AffineWeights& w, ActParams p, int32_t pad,
                            Code code, int8_t* dst, int32_t* xsum,
                            float* toff) {
  uint32_t t = 0, grp = 0;
  uint32_t end = w.gsize < w.k_pad ? w.gsize : w.k_pad;
  int32_t sum = 0;
  auto put = [&](int32_t q) {
    dst[t] = (int8_t)q;
    sum += q;
    if (++t == end) {
      close_group(w, grp, sum, p, xsum, toff);
      ++grp;
      end = (end + w.gsize <= w.k_pad) ? end + w.gsize : w.k_pad;
      sum = 0;
    }
  };
  for_each_tap(x, g, row, [&](const T* v) {
    put(v != nullptr ? code(*v) : pad);
  });
  while (t < w.k_pad) {
    put(0);
  }
}

}  // namespace

ActParams choose(const ActQuant& a, const float* x, size_t n) {
  if (a.granularity == kGranularityStatic) {
    return ActParams{a.scale, a.zero_point};  // the model's own constants
  }
  // min(amin, 0) and max(amax, 0): both choosers widen the range to hold 0.
  float lo = 0.0f, hi = 0.0f;
  for (size_t t = 0; t < n; ++t) {
    const float v = x[t];
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  ActParams p;
  if (a.chooser == kChooserSymDiv) {
    const float amax = -lo > hi ? -lo : hi;
    const float s = amax / a.sym_div;
    p.sx = s < a.eps ? a.eps : s;
    p.zx = 0;
    return p;
  }
  // kChooserAffineMinMax (attach admits no other dynamic chooser).
  const float span = hi - lo;
  const float s = span / (float)(a.qmax - a.qmin);
  p.sx = s < a.eps ? a.eps : s;
  p.zx = clamp_code(a.qmin - rne_i(lo / p.sx), a.qmin, a.qmax);
  return p;
}

float requantize(const OpQuant& q, float y) {
  const OutQuant& o = *q.out;
  const int32_t yq = code_of(y, q.out_inv, o.zero_point, o.qmin, o.qmax);
  return (float)(yq - o.zero_point) * o.scale;
}

void quantize_row(const ActQuant& a, ActParams p, const float* x, uint32_t k,
                  const AffineWeights& w, int8_t* dst, int32_t* xsum,
                  float* toff) {
  const float inv = 1.0f / p.sx;
  quantize_groups(x, k, w, p, [&](float v) {
    return code_of(v, inv, p.zx, a.qmin, a.qmax);
  }, dst, xsum, toff);
}

void quantize_conv_row(const ActQuant& a, ActParams p, const float* x,
                       const ConvGather& g, uint32_t row, const AffineWeights& w,
                       int8_t* dst, int32_t* xsum, float* toff) {
  const float inv = 1.0f / p.sx;
  quantize_gather(x, g, row, w, p, p.zx, [&](float v) {
    return code_of(v, inv, p.zx, a.qmin, a.qmax);
  }, dst, xsum, toff);
}

float quantize_row_a0(const float* x, uint32_t k, float fix, bool ties_to_even,
                      const AffineWeights& w, int8_t* dst, int32_t* xsum,
                      float* toff) {
  const ActParams p{fix >= 0.0f ? fix : abs_max(x, k) / 127.0f, 0};
  const float inv = a0_inverse(p.sx);
  quantize_groups(x, k, w, p, [&](float v) {
    return a0_code(v, inv, ties_to_even);
  }, dst, xsum, toff);
  return p.sx;
}

float quantize_conv_row_a0(const float* x, const ConvGather& g, uint32_t row,
                           float fix, bool ties_to_even,
                           const AffineWeights& w, int8_t* dst, int32_t* xsum,
                           float* toff) {
  float s = fix;
  if (s < 0.0f) {
    // The absmax over the taps inside the input, in the gather's order.
    float amax = 0.0f;
    for_each_tap(x, g, row, [&](const float* v) {
      if (v != nullptr) {
        const float a = std::fabs(*v);
        if (a > amax) {
          amax = a;
        }
      }
    });
    s = amax / 127.0f;
  }
  const float inv = a0_inverse(s);
  quantize_gather(x, g, row, w, ActParams{s, 0}, 0, [&](float v) {
    return a0_code(v, inv, ties_to_even);
  }, dst, xsum, toff);
  return s;
}

// ---- Stage B: int8 codes in and out -----------------------------------------------
// Not cold: per element of an op whose codes travel as codes. Integers only,
// apart from T_g (a float offset's term), computed as quantize_row does.

int8_t requantize_code(const OpQuant& q, float y) {
  const OutQuant& o = *q.out;
  return (int8_t)code_of(y, q.out_inv, o.zero_point, o.qmin, o.qmax);
}

void codes_row(const ActQuant& a, const int8_t* x, uint32_t k,
               const AffineWeights& w, int8_t* dst, int32_t* xsum,
               float* toff) {
  quantize_groups(x, k, w, ActParams{a.scale, a.zero_point},
                  [](int8_t c) { return (int32_t)c; }, dst, xsum, toff);
}

void codes_conv_row(const ActQuant& a, const int8_t* x, const ConvGather& g,
                    uint32_t row, const AffineWeights& w, int8_t* dst,
                    int32_t* xsum, float* toff) {
  quantize_gather(x, g, row, w, ActParams{a.scale, a.zero_point},
                  a.zero_point, [](int8_t c) { return (int32_t)c; }, dst,
                  xsum, toff);
}

// ---- 16-bit activations ----------------------------------------------------------
// Not cold: per element of an a16w8 op. The static quantiser is code_of, as
// for 8-bit records (its 2^22 rail clamp is far past any int16 code); the dot
// is exact in int64 and fl(acc) its one rounding.

namespace {

/// Products summed in int32 before each int64 add: |xq * c| <= 32768 * 128 =
/// 2^22, so 511 of them stay inside int32 and the int64 total is exact.
constexpr uint32_t kA16Block = 511;
static_assert((uint64_t)kA16Block * 32768u * 128u <= kInt32Max,
              "an a16 block's int32 partial sum cannot overflow");

}  // namespace

void quantize_row16(const ActQuant& a, const float* x, uint32_t k,
                    uint32_t k_pad, int16_t* dst) {
  const float inv = 1.0f / a.scale;
  for (uint32_t t = 0; t < k; ++t) {
    dst[t] = (int16_t)code_of(x[t], inv, a.zero_point, a.qmin, a.qmax);
  }
  for (uint32_t t = k; t < k_pad; ++t) {
    dst[t] = 0;
  }
}

void quantize_conv_row16(const ActQuant& a, const float* x,
                         const ConvGather& g, uint32_t row, uint32_t k_pad,
                         int16_t* dst) {
  const float inv = 1.0f / a.scale;
  uint32_t t = 0;
  for_each_tap(x, g, row, [&](const float* v) {
    dst[t++] = (int16_t)(v != nullptr
                             ? code_of(*v, inv, a.zero_point, a.qmin, a.qmax)
                             : a.zero_point);
  });
  while (t < k_pad) {
    dst[t++] = 0;
  }
}

void gather_conv_row(const float* x, const ConvGather& g, uint32_t row,
                     float* dst) {
  uint32_t t = 0;
  for_each_tap(x, g, row, [&](const float* v) {
    dst[t++] = v != nullptr ? *v : 0.0f;
  });
}

void conv_tap_offsets(const ConvGather& g, uint32_t row, int32_t* offs) {
  // for_each_tap's walk over a base it never reads: one channel's input, or
  // the whole input; every offset fits int32 (validate bounds the tensor).
  static const float kBase[1] = {0.0f};
  uint32_t t = 0;
  for_each_tap(kBase, g, row, [&](const float* v) {
    offs[t++] = v != nullptr ? (int32_t)(v - kBase) : -1;
  });
}

float a16_column(const AffineWeights& w, uint32_t n, const int16_t* xq,
                 const int8_t* wrow, float sx, float y) {
  int64_t acc = 0;
  for (uint32_t left = w.k_pad; left != 0;) {
    const uint32_t len = left > kA16Block ? kA16Block : left;
    left -= len;
    int32_t part = 0;
    // A counted loop over walking pointers: at -Os, GCC's zero-overhead
    // LOOP (9 instructions a product, no branch); an index bound to k_pad
    // compiles to a branch and a jump around 8.
    for (uint32_t t = 0; t < len; ++t) {
      part += (int32_t)*xq++ * (int32_t)*wrow++;
    }
    acc += part;
  }
  const float term = (float)acc * sx;
  return y + term * table_float(w.scales, w.fp16_scales, n);
}

void fold_zero_points(const AffineWeights& w, uint32_t row, int8_t* codes) {
  const size_t base = (size_t)row * w.ngw;
  for (uint32_t g = 0; g < w.ngw; ++g) {
    const uint32_t a = g * w.gsize;
    const uint32_t b = (a + w.gsize <= w.k_pad) ? a + w.gsize : w.k_pad;
    const int32_t zero = group_zero(w, base + g);
    // The k_pad tail folds too: its activations are zero, so it still adds
    // nothing to any dot.
    for (uint32_t t = a; t < b; ++t) {
      codes[t] = (int8_t)((int32_t)codes[t] - zero);
    }
  }
}

void dequant_row(const OpQuant& q, bool i4, const uint8_t* row, uint32_t d,
                 uint32_t gsize, uint32_t ng, const void* scales,
                 bool fp16_scales, size_t first, float* dst) {
  const WeightAffine& wa = *q.weight;
  const int8_t* r8 = reinterpret_cast<const int8_t*>(row);
  for (uint32_t g = 0; g < ng; ++g) {
    const uint32_t a = g * gsize;
    const uint32_t b = (a + gsize < d) ? (a + gsize) : d;
    const size_t entry = first + g;
    const float s = table_float(scales, fp16_scales, entry);
    if (wa.zp_kind == kZpFp32) {
      // A float zero point applies to the source's own code: fl(fl(q - z) * s).
      // The table is 4-byte aligned (checked at validate and attach).
      const float z = static_cast<const float*>(q.zero_points)[entry];
      for (uint32_t col = a; col < b; ++col) {
        const int32_t c = i4 ? (int32_t)((col & 1) ? (row[col >> 1] >> 4)
                                                   : (row[col >> 1] & 0x0F)) - 8
                             : (int32_t)r8[col];
        const float v = (float)(c + wa.storage_bias) - z;
        dst[col] = v * s;
      }
      continue;
    }
    const int32_t zero = wa.zp_kind == kZpInt8
        ? (int32_t)static_cast<const int8_t*>(q.zero_points)[entry]
        : -wa.storage_bias;
    const bool offset = wa.offset_kind != kOffsetNone;
    const float m = offset ? table_float(q.offsets,
                                         wa.offset_kind == kOffsetFp16, entry)
                           : 0.0f;
    for (uint32_t col = a; col < b; ++col) {
      const int32_t c = i4 ? (int32_t)((col & 1) ? (row[col >> 1] >> 4)
                                                 : (row[col >> 1] & 0x0F)) - 8
                           : (int32_t)r8[col];
      // c - zw is exact in int32, and exact as a float (|c - zw| <= 255).
      float v = (float)(c - zero) * s;
      if (offset) {
        v = v + m;
      }
      dst[col] = v;
    }
  }
}

ASYM_COLD void run_kcache_zero_point_check() {
  using executorch::aten::TensorImpl;
  int8_t codes[6] = {0};
  float floats[6] = {0.0f};
  TensorImpl::SizesType sizes[2] = {2, 3};
  TensorImpl::SizesType short_sizes[2] = {2, 2};
  TensorImpl::DimOrderType order[2] = {0, 1};
  TensorImpl::StridesType strides[2] = {3, 1};
  TensorImpl::StridesType short_strides[2] = {2, 1};
  TensorImpl control(ScalarType::Char, 2, sizes, codes, order, strides);
  TensorImpl wrong_dtype(ScalarType::Float, 2, sizes, floats, order, strides);
  TensorImpl wrong_shape(ScalarType::Char, 2, short_sizes, codes, order,
                         short_strides);
  // A batch of 2 caches of 3 key rows: one int8 zero point per row.
  const bool ok = check_kcache_zero_points(Tensor(&control), 2, 3, 0) ==
      Error::Ok;
  clear_last_refusal();
  const bool dtype = check_kcache_zero_points(Tensor(&wrong_dtype), 2, 3, 0) !=
          Error::Ok &&
      last_refusal() == kRuleKcacheZeroPoints;
  clear_last_refusal();
  const bool shape = check_kcache_zero_points(Tensor(&wrong_shape), 2, 3, 0) !=
          Error::Ok &&
      last_refusal() == kRuleKcacheZeroPoints;
  clear_last_refusal();
  printf("  asym kcache zp check: control=%s dtype=%s shape=%s\n",
         ok ? "accepted" : "REFUSED", dtype ? "refused" : "ACCEPTED",
         shape ? "refused" : "ACCEPTED");
}

namespace {

/// The float32 nearest v, ties to even, by integer arithmetic alone: the
/// reference run_a16_check holds a16_column's fl(acc) -- libgcc's
/// __floatdisf on the device -- to.
ASYM_COLD float nearest_float(int64_t v) {
  if (v == 0) {
    return 0.0f;
  }
  const bool neg = v < 0;
  const uint64_t m = neg ? 0ull - (uint64_t)v : (uint64_t)v;
  uint32_t shift = 0;
  while ((m >> shift) >= (1ull << 24)) {
    ++shift;
  }
  uint64_t q = m >> shift;
  if (shift != 0) {
    const uint64_t rest = m & ((1ull << shift) - 1);
    const uint64_t half = 1ull << (shift - 1);
    if (rest > half || (rest == half && (q & 1) != 0)) {
      ++q;
    }
  }
  // q <= 2^24 converts exactly, and each doubling is exact.
  float f = (float)(uint32_t)q;
  for (uint32_t s = 0; s < shift; ++s) {
    f = f * 2.0f;
  }
  return neg ? -f : f;
}

/// Codes and weights whose dot is exactly `target`: rails of 32767 * 127,
/// then the remainder as c * 127 + r * 1, zero-padded to 16. Returns k_pad.
ASYM_COLD uint32_t dot_of(int64_t target, int16_t* xq, int8_t* w) {
  const uint64_t unit = 32767ull * 127ull;
  const int8_t sign = target < 0 ? -1 : 1;
  uint64_t m = target < 0 ? 0ull - (uint64_t)target : (uint64_t)target;
  uint32_t len = 0;
  for (; m >= unit; m -= unit, ++len) {
    xq[len] = 32767;
    w[len] = (int8_t)(127 * sign);
  }
  xq[len] = (int16_t)(m / 127);
  w[len++] = (int8_t)(127 * sign);
  xq[len] = (int16_t)(m % 127);
  w[len++] = sign;
  for (; len % 16 != 0; ++len) {
    xq[len] = 0;
    w[len] = 0;
  }
  return len;
}

ASYM_COLD bool same_bits(float a, float b) {
  uint32_t x, y;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

}  // namespace

ASYM_COLD void run_a16_check() {
  // Dots on exact float32 ties -- 2^25 + 2 rounds down to even, 2^25 + 6 up
  // (spacing 4); 2^32 + 256 down, 2^32 + 768 up (spacing 512) -- one either
  // side of a tie, and negations.
  static const int64_t kTargets[] = {
      33554434LL,  33554438LL,  33554433LL,   33554435LL,
      4294967552LL, 4294968064LL, 4294967551LL, 4294968065LL,
      -33554434LL, -33554438LL, -4294967552LL, -4294968064LL};
  constexpr uint32_t kMaxLen = 1056;  // 2^32 + 768 takes 1,034 codes
  int16_t* xq = new (std::nothrow) int16_t[kMaxLen];
  int8_t* wrow = new (std::nothrow) int8_t[kMaxLen];
  if (xq == nullptr || wrow == nullptr) {
    printf("  asym a16 check: SKIPPED (no memory)\n");
    delete[] xq;
    delete[] wrow;
    return;
  }
  const float one = 1.0f;
  const OpQuant none{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                     false, 0.0f};
  const uint32_t n_ties = sizeof(kTargets) / sizeof(kTargets[0]);
  uint32_t ties = 0;
  for (uint32_t c = 0; c < n_ties; ++c) {
    const uint32_t len = dot_of(kTargets[c], xq, wrow);
    const AffineWeights w = affine_weights(none, len, len, 0, false, &one);
    ties += same_bits(a16_column(w, 0, xq, wrow, 1.0f, 0.0f),
                      nearest_float(kTargets[c]));
  }
  // Dots past int32: 600 codes on each rail against +-127, and -32768
  // (int16's own end) against -128, |acc| up to 600 * 2^22.
  struct Wide {
    int16_t code;
    int8_t weight;
  };
  static const Wide kWide[] = {{32767, 127}, {-32767, 127}, {-32768, -128}};
  const uint32_t n_wide = sizeof(kWide) / sizeof(kWide[0]);
  uint32_t wide = 0;
  for (uint32_t c = 0; c < n_wide; ++c) {
    for (uint32_t t = 0; t < 608; ++t) {
      xq[t] = t < 600 ? kWide[c].code : (int16_t)0;
      wrow[t] = t < 600 ? kWide[c].weight : (int8_t)0;
    }
    const AffineWeights w = affine_weights(none, 600, 608, 0, false, &one);
    const int64_t want = 600LL * kWide[c].code * kWide[c].weight;
    wide += same_bits(a16_column(w, 0, xq, wrow, 1.0f, 0.0f),
                      nearest_float(want));
  }
  printf("  asym a16 check: conversions=%u/%u wide=%u/%u\n", (unsigned)ties,
         (unsigned)n_ties, (unsigned)wide, (unsigned)n_wide);
  // The int16 loop's cost, for QEMU's -icount shift=0 (each instruction one
  // virtual nanosecond: the microseconds x 1000 are instructions): 16 rows
  // of 1,024 activations quantised, then 64 columns of 1,024 MACs.
  float xs[64];
  for (uint32_t t = 0; t < 64; ++t) {
    xs[t] = (float)((int32_t)(t * 37u % 64u) - 32) * 0.0625f;
  }
  ActQuant rec{};
  rec.code_bits = 16;
  rec.granularity = kGranularityStatic;
  rec.rounding = kRoundRneBeforeZp;
  rec.qmin = -32767;
  rec.qmax = 32767;
  rec.scale = 0.000244140625f;
  for (uint32_t t = 0; t < 1024; ++t) {
    wrow[t] = (int8_t)((int32_t)(t * 29u % 255u) - 127);
  }
  const int64_t t0 = esp_timer_get_time();
  for (uint32_t r = 0; r < 16; ++r) {
    for (uint32_t c = 0; c < 1024; c += 64) {
      quantize_row16(rec, xs, 64, 64, xq + c);
    }
  }
  const int64_t t1 = esp_timer_get_time();
  const AffineWeights w = affine_weights(none, 1024, 1024, 0, false, &one);
  float sink = 0.0f;
  for (uint32_t n = 0; n < 64; ++n) {
    sink = a16_column(w, 0, xq, wrow, 1.0f, sink);
  }
  const int64_t t2 = esp_timer_get_time();
  delete[] xq;
  delete[] wrow;
  printf("  asym a16 timing: 16,384 quantised in %lld us, 65,536 MACs in "
         "%lld us (sum %g)\n", (long long)(t1 - t0), (long long)(t2 - t1),
         (double)sink);
}

}  // namespace esp32_asym
