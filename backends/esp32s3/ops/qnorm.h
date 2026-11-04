/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// espressif::qnorm: a decomposed quantised RMSNorm or LayerNorm -- every op of
// the pattern a stock static recipe (Arm, Cortex-M) leaves, with every
// quantise/dequantise pair between them -- as one op (fuse_qnorm_pass.py
// matches the pattern and builds the tables at export).
//
//   espressif::qnorm.out(Tensor x, Tensor tables, Tensor quant, Tensor? weight,
//                        Tensor? bias, int kind, int flags, float eps, *,
//                        Tensor(a!) out) -> Tensor(a!)
//
// x is fp32 [..., N], normalised over its last axis row by row; out is the
// pattern's last op, fp32 (the pair after it stays for its consumer, as
// quant_lut's does). quant is fp32 [S, 5], one quantiser per row: (inv, scale,
// zero point, quant_min, quant_max), the slots below; a row with quant_min >
// quant_max is absent (no pair at that step). fq(v) is a slot's pair,
// fl((clamp(rne(fl(v * inv)) + zp, qmin, qmax) - zp) * scale), and code(v) its
// quantise -- quant_lut's quantise step. tables is fp32 [T, 256]: functions of
// one code, entry c - quant_min of the slot that keys it, built by torch from
// the graph's own nodes (x squared, and everything from a reduction's
// quantiser to the next value an element reads: the mean, or the rsqrt).
//
// kind 0, RMSNorm (slots X, SQ, RED, M; tables SQ keyed by X, R keyed by RED):
//   a_i  = X ? fq_X(x_i) : x_i
//   s_i  = X ? T_SQ[code_X(x_i)] : fq_SQ(fl(a_i * a_i))
//   acc  = sum_i s_i, in index order from 0.0f (the portable sum's order)
//   v    = acc [/ N if flags & MEAN] [+ eps if flags & EPS]
//   r    = T_R[code_RED(v)]
//   out_i = weight ? fl(fq_M(fl(a_i * r)) * w_i) : fl(a_i * r)
//
// kind 1, LayerNorm (slots X, RED1, RED2, C1, C2, RED3, N, W; tables M1 keyed
// by RED1, M2 by RED2, SQ by C2, R by RED3):
//   a_i  = fq_X(x_i);  acc1 = sum_i a_i
//   mu1  = T_M1[code_RED1(acc1)];  mu2 = T_M2[code_RED2(acc1)]
//   acc2 = sum_i T_SQ[code_C2(fl(a_i - mu2))];  r = T_R[code_RED3(acc2)]
//   n_i  = fq_N(fl(fq_C1(fl(a_i - mu1)) * r))
//   out_i = n_i, then * w_i (through fq_W when a bias follows), then + b_i
//
// Kinds 2 and 3 are the same norms on int16 codes (Arm a16w8): a quantiser
// may hold up to 65,536 codes, its quantise is lut16's (fl(v * inv) clamped
// to +-2^16 first), and nothing is tabulated per element. The per-row chains
// are programs, P(v) below, run once per row from the reduction's sum.
//
// kind 2, RMSNorm on int16 codes (slots X, SQ, M; program 0):
//   a_i  = X ? fq_X(x_i) : x_i;  s_i = fq_SQ(fl(a_i * a_i))
//   acc  = sum_i s_i, in index order from 0.0f;  r = P0(acc)
//   out_i = weight ? fl(fq_M(fl(a_i * r)) * w_i) : fl(a_i * r)
//
// kind 3, LayerNorm on int16 codes (slots X, C1, C2, SQ, N, W; programs 0
// and 1 the two means, 2 the rsqrt):
//   a_i  = fq_X(x_i);  acc1 = sum_i a_i;  mu1 = P0(acc1);  mu2 = P1(acc1)
//   acc2 = sum_i fq_SQ(fl(c_i * c_i)), c_i = fq_C2(fl(a_i - mu2))
//   r    = P2(acc2), then n_i and out_i as kind 1
//
// For kinds 2 and 3, tables is int32 [W]: word 0 the programs, word 1 the
// lut16 tables, then (first word, steps) per program, kTableWords per table
// (scheme, lo, hi, shift, G, out_min, out_max, count, data word,
// exceptions, exception word, 0), then the steps, kStepWords each (the op,
// then five arguments; floats as their bits), with a float register v (the
// sum to begin with) and a code register c:
//   CODE  inv, -, zp, qmin, qmax  c = code(v)
//   TABLE t                       c = lut16 table t's output code for c
//                                 (quant_lut16.h's eval_code), or the one
//                                 its exceptions list for c
//   DEQ   -, scale, zp            v = fl(fl(c - zp) * scale)
//   MUL, ADD, DIV, RSUB, RDIV k   v = fl(v * k), fl(v + k), fl(v / k),
//                                 fl(k - v), fl(k / v)
// then each table's data (int16 entries two to a word, int32 nodes, float32
// rows) and exceptions ((input code, output code) pairs by increasing input
// code). A program's first CODE is its reduction's quantiser. Tables go
// through lut16::checked(): validated once, copied to internal RAM on the
// device. The story firmware (ET_PLE_ONLY) builds kinds 0 and 1 only.
//
// Every step is one IEEE float32 operation or a quantise, so the device's
// bits are the host's and the mirror's (mirrors.qnorm): qnorm.cpp
// builds with -ffp-contract=off. Against eager, the reductions are summed in
// another order than ATen's, so a reduction's code may differ by one step;
// given the same reduction codes, every other value is eager's.

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace espressif {
namespace qnorm {

enum Kind : int32_t { kRms = 0, kLayer = 1, kRms16 = 2, kLayer16 = 3 };

// flags
constexpr int32_t kMean = 1;  // RMSNorm: the reduction divides by N
constexpr int32_t kEps = 2;   // RMSNorm: eps is added before RED's quantiser

// Quantiser slots (rows of quant).
enum RmsSlot : int32_t { kRmsX = 0, kRmsSq, kRmsRed, kRmsM, kRmsSlots };
enum LayerSlot : int32_t {
  kLnX = 0, kLnRed1, kLnRed2, kLnC1, kLnC2, kLnRed3, kLnN, kLnW, kLnSlots
};
// Tables (rows of tables).
enum RmsTable : int32_t { kRmsTSq = 0, kRmsTR, kRmsTables };
enum LayerTable : int32_t { kLnTM1 = 0, kLnTM2, kLnTSq, kLnTR, kLnTables };
constexpr int32_t kTableLen = 256;
// Kinds 2 and 3: slots, and the tables tensor's programs and steps.
enum Rms16Slot : int32_t { kR16X = 0, kR16Sq, kR16M, kR16Slots };
enum Layer16Slot : int32_t {
  kL16X = 0, kL16C1, kL16C2, kL16Sq, kL16N, kL16W, kL16Slots
};
enum Op : int32_t {
  kCode = 1, kTable, kDeq, kMul, kAdd, kDiv, kRsub, kRdiv
};
constexpr int32_t kStepWords = 6;
constexpr int32_t kTableWords = 12;
constexpr int32_t kMaxSteps = 32;       // per program
constexpr int32_t kMaxLuts = 4;         // lut16 tables per op
constexpr int32_t kMaxExceptions = 256; // per table

// One quantiser, validated by the shim.
struct Quant {
  float inv;
  float scale;
  int32_t zero_point;
  int32_t quant_min;
  int32_t quant_max;
  bool present;
};

// The op over `rows` rows of n elements (the shim's loop, and what a check
// drives directly). quant holds kRmsSlots or kLnSlots entries, tables
// kRmsTables or kLnTables rows of kTableLen; weight and bias may be null.
void apply(int32_t kind, int32_t flags, const float* x, float* out,
           size_t rows, size_t n, const Quant* quant, const float* tables,
           const float* weight, const float* bias, float eps);

// The kCustomKernels entry for espressif::qnorm.out. Every malformed argument
// fails the instruction (InvalidArgument; a wrong argument count is
// InvalidProgram) before out is written.
void shim(executorch::runtime::KernelRuntimeContext& ctx,
          executorch::runtime::Span<executorch::runtime::EValue*> stack);

struct Stats {
  uint64_t calls;
  uint64_t rows;
  uint64_t refusals;
};
Stats stats();

// One "qnorm dispatch:" line, printed only once the op has run or refused
// (the story firmware runs none, so its log prints no such line).
void report_dispatch();

}  // namespace qnorm
}  // namespace espressif
