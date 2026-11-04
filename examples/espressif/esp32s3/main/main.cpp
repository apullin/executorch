/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ExecuTorch runner for the ESP32-S3.
//
// Runs the portable/delegated parity zoo, or (ET_STORY) the story model.
//
// Model operators and portable-kernel overrides live in espressif_ops.cpp.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <utility>

#include "esp_heap_caps.h"
#if defined(ET_STORY) || defined(ET_BENCH_MODEL)
#include "soc/soc.h"
#include "soc/spi_mem_reg.h"
#endif
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#ifdef ET_PAPERCOLOR_DISPLAY
#include "papercolor_status.h"
#endif

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>

#include "esp_partition_loader.h"
#include "espressif_ops.h"
#include "esp32_asym.h"
#include "divsqrt.h"
#include "newlib_expf.h"
#include "kv_affine.h"
#include "quant_lut.h"
#include "quant_lut16.h"
#include "quant_rules.h"
// The backend option API: set_option/get_option by backend name, which is how an
// application reconfigures a delegate without re-exporting it.
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

// Generated beside the standalone packed-i4 decode control. Keeping the exact
// tokens in the same export transaction prevents its host oracle and firmware
// runner from silently diverging.
#if __has_include("decode_inputs.h")
#include "decode_inputs.h"
#else
static constexpr int kDecodeTokens[] = {3, 40, 77, 114, 151, 188, 225, 262};
static constexpr int kDecodeSteps = 8;
#endif

// Generated with the combined i8 prefill/decode PTE. Its split input sequence is
// also the contract that the host oracle uses to prove cross-method KV continuity.
#if __has_include("glimmer_multimethod_inputs.h")
#include "glimmer_multimethod_inputs.h"
#else
static constexpr int kGlimmerPrefillTokens[] = {3, 40, 77, 114};
static constexpr int kGlimmerPrefillChunk = 4;
static constexpr int kGlimmerDecodeTokens[] = {151, 188, 225, 6};
static constexpr int kGlimmerDecodeSteps = 4;
#endif

// The story build's exported artifact (export_story_flash.py).
#if defined(ET_STORY)
#if !__has_include("story_inputs.h")
#error "story build without its exported artifact (export_story_flash.py)"
#endif
#include "story_inputs.h"
#include "story_vocab.h"
#define STORY_INPUTS 1
#define HAVE_PLE_INPUTS 1
#endif
// The bench build's model (bench_runner.inc) is the bench image in the program
// partition, so it needs nothing generated here.
#if defined(ET_BENCH_MODEL)
#if defined(ET_STORY)
#error "ET_BENCH_MODEL and ET_STORY are two different lean builds"
#endif
#define BENCH_INPUTS 1
#define HAVE_PLE_INPUTS 1
#endif
#if defined(ET_PLE_ONLY) && !defined(HAVE_PLE_INPUTS)
#error "ET_PLE_ONLY build without generated inputs -- export a lean artifact"
#endif

// Portable kernel registration, produced by ExecuTorch's own codegen (see
// gen_kernels.py). --manual-registration means this is an ordinary function rather
// than a static initializer, which matters: an ESP-IDF component is a static archive
// linked without --whole-archive, so an initializer nothing references is free to be
// dropped and the failure would appear at Method::init as OperatorMissing.
#include "RegisterKernels.h"

using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
#ifdef ESP32_OP_TIMING
using executorch::runtime::AllocatorID;
using executorch::runtime::ArrayRef;
using executorch::runtime::ChainID;
using executorch::runtime::DebugHandle;
using executorch::runtime::DelegateDebugIdType;
using executorch::runtime::DelegateDebugIntId;
using executorch::runtime::EventTracer;
using executorch::runtime::EventTracerEntry;
using executorch::runtime::EventTracerFilterBase;
using executorch::runtime::LoggedEValueType;
using executorch::runtime::Result;
#endif
using executorch::runtime::FreeableBuffer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Program;
using executorch::runtime::Span;

#ifdef ESP32_OP_TIMING
// Allocation-free operator attribution for constrained targets. ExecuTorch's
// runtime reports all portable kernels as OPERATOR_CALL and identifies the
// concrete instruction through current_debug_handle(); the exporter emits the
// matching handle-to-target map for the host runner. Framework bookkeeping is
// aggregated by event name, while OPERATOR_CALL/DELEGATE_CALL retain one row
// per instruction. Timings intentionally include the tracer's own timer cost
// and are never mixed with normal benchmark rows.
class TimingEventTracer final : public EventTracer {
 public:
  TimingEventTracer() {
    set_event_tracer_profiling_level(
        executorch::runtime::EventTracerProfilingLevel::kProfileAllEvents);
  }

  void reset() {
    count_ = 0;
    overflow_ = 0;
  }

  void report() {
    for (size_t i = 0; i < count_; ++i) {
      order_[i] = i;
    }
    std::sort(order_, order_ + count_, [&](size_t lhs, size_t rhs) {
      return stats_[lhs].total_us > stats_[rhs].total_us;
    });
    for (size_t rank = 0; rank < count_; ++rank) {
      const Stat& stat = stats_[order_[rank]];
      printf("ET_OP_PROFILE event=%s debug=", stat.name);
      if (stat.debug_handle == kAggregateDebug) {
        printf("all");
      } else {
        printf("%u", (unsigned)stat.debug_handle);
      }
      printf(" calls=%u total_us=%llu avg_us=%.3f\n",
             (unsigned)stat.calls, (unsigned long long)stat.total_us,
             stat.calls ? (double)stat.total_us / stat.calls : 0.0);
    }
    if (overflow_ != 0) {
      printf("ET_OP_PROFILE overflow=%u\n", (unsigned)overflow_);
    }
  }

  void create_event_block(const char*) override {}

  EventTracerEntry start_profiling(
      const char* name,
      ChainID chain_id = executorch::runtime::kUnsetChainId,
      DebugHandle debug_handle = executorch::runtime::kUnsetDebugHandle)
      override {
    if (chain_id == executorch::runtime::kUnsetChainId) {
      chain_id = current_chain_id();
    }
    if (debug_handle == executorch::runtime::kUnsetDebugHandle) {
      debug_handle = current_debug_handle();
    }
    return start(name, chain_id, debug_handle, DelegateDebugIdType::kNone);
  }

  void end_profiling(EventTracerEntry entry) override { finish(entry); }

  EventTracerEntry start_profiling_delegate(
      const char* name, DelegateDebugIntId delegate_debug_id) override {
    (void)delegate_debug_id;
    return start(name != nullptr ? name : "DELEGATE_EVENT",
                 current_chain_id(), current_debug_handle(),
                 DelegateDebugIdType::kInt);
  }

  void end_profiling_delegate(
      EventTracerEntry entry, const void*, size_t) override {
    finish(entry);
  }

  void log_profiling_delegate(
      const char* name, DelegateDebugIntId, et_timestamp_t start_time,
      et_timestamp_t end_time, const void*, size_t) override {
    EventTracerEntry entry = start(
        name != nullptr ? name : "DELEGATE_EVENT", current_chain_id(),
        current_debug_handle(), DelegateDebugIdType::kInt, start_time);
    if (entry.event_id >= 0 && end_time >= start_time) {
      Stat& stat = stats_[(size_t)entry.event_id];
      stat.total_us += (uint64_t)(end_time - start_time);
      ++stat.calls;
    }
  }

  void track_allocation(AllocatorID, size_t) override {}
  AllocatorID track_allocator(const char*) override { return 0; }
  void set_delegation_intermediate_output_filter(
      EventTracerFilterBase*) override {}

  Result<bool> log_evalue(const EValue&, LoggedEValueType) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*, DelegateDebugIntId, const Tensor&) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*, DelegateDebugIntId, const ArrayRef<Tensor>) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*, DelegateDebugIntId, const int&) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*, DelegateDebugIntId, const bool&) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*, DelegateDebugIntId, const double&) override {
    return true;
  }

 private:
  static constexpr size_t kMaxStats = 512;  // a Glimmer decode plan is ~400 instructions
  static constexpr size_t kNameBytes = 40;
  static constexpr uint32_t kAggregateDebug = 0xffffffffu;

  struct Stat {
    char name[kNameBytes];
    uint32_t debug_handle;
    uint64_t total_us;
    uint32_t calls;
  };

  static bool per_instruction(const char* name) {
    return strcmp(name, "OPERATOR_CALL") == 0 ||
        strcmp(name, "DELEGATE_CALL") == 0;
  }

  size_t find_or_add(const char* name, uint32_t debug_handle) {
    for (size_t i = 0; i < count_; ++i) {
      if (stats_[i].debug_handle == debug_handle &&
          strcmp(stats_[i].name, name) == 0) {
        return i;
      }
    }
    if (count_ == kMaxStats) {
      ++overflow_;
      return SIZE_MAX;
    }
    Stat& stat = stats_[count_];
    strncpy(stat.name, name, kNameBytes - 1);
    stat.name[kNameBytes - 1] = '\0';
    stat.debug_handle = debug_handle;
    stat.total_us = 0;
    stat.calls = 0;
    return count_++;
  }

  EventTracerEntry start(
      const char* name, ChainID chain_id, DebugHandle debug_handle,
      DelegateDebugIdType type, et_timestamp_t at = 0) {
    if (name == nullptr) {
      name = "UNNAMED_EVENT";
    }
    const uint32_t key_debug = per_instruction(name)
        ? (uint32_t)debug_handle
        : kAggregateDebug;
    const size_t index = find_or_add(name, key_debug);
    EventTracerEntry entry{};
    entry.event_id = index == SIZE_MAX ? -1 : (int64_t)index;
    entry.chain_id = chain_id;
    entry.debug_handle = debug_handle;
    entry.start_time = at != 0 ? at : (et_timestamp_t)esp_timer_get_time();
    entry.delegate_event_id_type = type;
    return entry;
  }

  void finish(EventTracerEntry entry) {
    if (entry.event_id < 0 || (size_t)entry.event_id >= count_) {
      return;
    }
    const int64_t now = esp_timer_get_time();
    if (now >= (int64_t)entry.start_time) {
      Stat& stat = stats_[(size_t)entry.event_id];
      stat.total_us += (uint64_t)(now - (int64_t)entry.start_time);
      ++stat.calls;
    }
  }

  Stat stats_[kMaxStats]{};
  size_t order_[kMaxStats]{};
  size_t count_ = 0;
  uint32_t overflow_ = 0;
};
#endif  // ESP32_OP_TIMING

// Both .pte blobs, embedded at 16-byte alignment by pte_blobs.S. Program::load
// rejects misaligned flatbuffer data with InvalidArgument.
extern const uint8_t mlp_pte_start[] asm("mlp_pte_start");
extern const uint8_t mlp_pte_end[] asm("mlp_pte_end");
extern const uint8_t bmmhalf_pte_start[] asm("bmmhalf_pte_start");
extern const uint8_t bmmhalf_pte_end[] asm("bmmhalf_pte_end");
extern const uint8_t mlpdel_pte_start[] asm("mlpdel_pte_start");
extern const uint8_t mlpdel_pte_end[] asm("mlpdel_pte_end");
extern const uint8_t mlpf32_pte_start[] asm("mlpf32_pte_start");
extern const uint8_t mlpf32_pte_end[] asm("mlpf32_pte_end");
extern const uint8_t mlpper_pte_start[] asm("mlpper_pte_start");
extern const uint8_t mlpper_pte_end[] asm("mlpper_pte_end");
extern const uint8_t mlpi4_pte_start[] asm("mlpi4_pte_start");
extern const uint8_t embi4_pte_start[] asm("embi4_pte_start");
extern const uint8_t embi4_pte_end[] asm("embi4_pte_end");
extern const uint8_t embi16_pte_start[] asm("embi16_pte_start");
extern const uint8_t embi16_pte_end[] asm("embi16_pte_end");
extern const uint8_t selhead_pte_start[] asm("selhead_pte_start");
extern const uint8_t selhead_pte_end[] asm("selhead_pte_end");
extern const uint8_t embedlin_pte_start[] asm("embedlin_pte_start");
extern const uint8_t embedlin_pte_end[] asm("embedlin_pte_end");
extern const uint8_t grpi4_pte_start[] asm("grpi4_pte_start");
extern const uint8_t grpi4_pte_end[] asm("grpi4_pte_end");
extern const uint8_t grpi8_pte_start[] asm("grpi8_pte_start");
extern const uint8_t grpi8_pte_end[] asm("grpi8_pte_end");
extern const uint8_t grpi4w_pte_start[] asm("grpi4w_pte_start");
extern const uint8_t grpi4w_pte_end[] asm("grpi4w_pte_end");
extern const uint8_t convd_pte_start[] asm("convd_pte_start");
extern const uint8_t convd_pte_end[] asm("convd_pte_end");
extern const uint8_t convdp_pte_start[] asm("convdp_pte_start");
extern const uint8_t convdp_pte_end[] asm("convdp_pte_end");
extern const uint8_t c1dd_pte_start[] asm("c1dd_pte_start");
extern const uint8_t c1dd_pte_end[] asm("c1dd_pte_end");
extern const uint8_t c1ddp_pte_start[] asm("c1ddp_pte_start");
extern const uint8_t c1ddp_pte_end[] asm("c1ddp_pte_end");
extern const uint8_t c3dd_pte_start[] asm("c3dd_pte_start");
extern const uint8_t c3dd_pte_end[] asm("c3dd_pte_end");
extern const uint8_t dwc_pte_start[] asm("dwc_pte_start");
extern const uint8_t dwc_pte_end[] asm("dwc_pte_end");
extern const uint8_t dwi4_pte_start[] asm("dwi4_pte_start");
extern const uint8_t dwi4_pte_end[] asm("dwi4_pte_end");
extern const uint8_t dwgi4_pte_start[] asm("dwgi4_pte_start");
extern const uint8_t dwgi4_pte_end[] asm("dwgi4_pte_end");
extern const uint8_t dwi16_pte_start[] asm("dwi16_pte_start");
extern const uint8_t dwi16_pte_end[] asm("dwi16_pte_end");
extern const uint8_t dwcp_pte_start[] asm("dwcp_pte_start");
extern const uint8_t dwcp_pte_end[] asm("dwcp_pte_end");
extern const uint8_t dwres_pte_start[] asm("dwres_pte_start");
extern const uint8_t dwres_pte_end[] asm("dwres_pte_end");
extern const uint8_t dwresp_pte_start[] asm("dwresp_pte_start");
extern const uint8_t dwresp_pte_end[] asm("dwresp_pte_end");
extern const uint8_t dw1_pte_start[] asm("dw1_pte_start");
extern const uint8_t dw1_pte_end[] asm("dw1_pte_end");
extern const uint8_t dw1p_pte_start[] asm("dw1p_pte_start");
extern const uint8_t dw1p_pte_end[] asm("dw1p_pte_end");
extern const uint8_t cvr_pte_start[] asm("cvr_pte_start");
extern const uint8_t cvr_pte_end[] asm("cvr_pte_end");
extern const uint8_t cvs_pte_start[] asm("cvs_pte_start");
extern const uint8_t cvs_pte_end[] asm("cvs_pte_end");
extern const uint8_t cvt_pte_start[] asm("cvt_pte_start");
extern const uint8_t cvt_pte_end[] asm("cvt_pte_end");
extern const uint8_t cvti4_pte_start[] asm("cvti4_pte_start");
extern const uint8_t cvti4_pte_end[] asm("cvti4_pte_end");
extern const uint8_t cvti16_pte_start[] asm("cvti16_pte_start");
extern const uint8_t cvti16_pte_end[] asm("cvti16_pte_end");
extern const uint8_t cvtp_pte_start[] asm("cvtp_pte_start");
extern const uint8_t cvtp_pte_end[] asm("cvtp_pte_end");
extern const uint8_t cvtop_pte_start[] asm("cvtop_pte_start");
extern const uint8_t cvtop_pte_end[] asm("cvtop_pte_end");
extern const uint8_t cvtopp_pte_start[] asm("cvtopp_pte_start");
extern const uint8_t cvtopp_pte_end[] asm("cvtopp_pte_end");
extern const uint8_t cvtdil_pte_start[] asm("cvtdil_pte_start");
extern const uint8_t cvtdil_pte_end[] asm("cvtdil_pte_end");
extern const uint8_t cvtdil16_pte_start[] asm("cvtdil16_pte_start");
extern const uint8_t cvtdil16_pte_end[] asm("cvtdil16_pte_end");
extern const uint8_t cvtdilp_pte_start[] asm("cvtdilp_pte_start");
extern const uint8_t cvtdilp_pte_end[] asm("cvtdilp_pte_end");
extern const uint8_t grpp_pte_start[] asm("grpp_pte_start");
extern const uint8_t grpp_pte_end[] asm("grpp_pte_end");
extern const uint8_t grpep_pte_start[] asm("grpep_pte_start");
extern const uint8_t grpep_pte_end[] asm("grpep_pte_end");
extern const uint8_t mlpi4_pte_end[] asm("mlpi4_pte_end");
extern const uint8_t mlpstat_pte_start[] asm("mlpstat_pte_start");
extern const uint8_t mlpstat_pte_end[] asm("mlpstat_pte_end");
extern const uint8_t attnqk_pte_start[] asm("attnqk_pte_start");
extern const uint8_t attnqk_pte_end[] asm("attnqk_pte_end");
extern const uint8_t attnqkq_pte_start[] asm("attnqkq_pte_start");
extern const uint8_t attnqkq_pte_end[] asm("attnqkq_pte_end");
extern const uint8_t attnkc8_pte_start[] asm("attnkc8_pte_start");
extern const uint8_t attnkc8_pte_end[] asm("attnkc8_pte_end");
extern const uint8_t attnkc8q_pte_start[] asm("attnkc8q_pte_start");
extern const uint8_t attnkc8q_pte_end[] asm("attnkc8q_pte_end");
// The asymmetric-quantisation refusal check: a carrier program and the synthetic
// case blobs written into its delegate payload (asym_blob_cases.py).
extern const uint8_t asymcarrier_pte_start[] asm("asymcarrier_pte_start");
extern const uint8_t asymcarrier_pte_end[] asm("asymcarrier_pte_end");
extern const uint8_t asymcases_blob_start[] asm("asymcases_blob_start");
extern const uint8_t asymcases_blob_end[] asm("asymcases_blob_end");
extern const uint8_t mlpi16_pte_start[] asm("mlpi16_pte_start");
extern const uint8_t mlpi16_pte_end[] asm("mlpi16_pte_end");
extern const uint8_t mlpr_pte_start[] asm("mlpr_pte_start");
extern const uint8_t mlpr_pte_end[] asm("mlpr_pte_end");
extern const uint8_t mlprdel_pte_start[] asm("mlprdel_pte_start");
extern const uint8_t mlprdel_pte_end[] asm("mlprdel_pte_end");
extern const uint8_t mlps_pte_start[] asm("mlps_pte_start");
extern const uint8_t mlps_pte_end[] asm("mlps_pte_end");
extern const uint8_t mlpsdel_pte_start[] asm("mlpsdel_pte_start");
extern const uint8_t mlpsdel_pte_end[] asm("mlpsdel_pte_end");
extern const uint8_t epiglu_pte_start[] asm("epiglu_pte_start");
extern const uint8_t epiglu_pte_end[] asm("epiglu_pte_end");
extern const uint8_t epiglum1_pte_start[] asm("epiglum1_pte_start");
extern const uint8_t epiglum1_pte_end[] asm("epiglum1_pte_end");
extern const uint8_t epiacts_pte_start[] asm("epiacts_pte_start");
extern const uint8_t epiacts_pte_end[] asm("epiacts_pte_end");
extern const uint8_t blkdel_pte_start[] asm("blkdel_pte_start");
extern const uint8_t blkdel_pte_end[] asm("blkdel_pte_end");
extern const uint8_t blki4_pte_start[] asm("blki4_pte_start");
extern const uint8_t blki4_pte_end[] asm("blki4_pte_end");
extern const uint8_t blki16_pte_start[] asm("blki16_pte_start");
extern const uint8_t blki16_pte_end[] asm("blki16_pte_end");
extern const uint8_t glimmermulti_pte_start[] asm("glimmermulti_pte_start");
extern const uint8_t glimmermulti_pte_end[] asm("glimmermulti_pte_end");
extern const uint8_t glimmermultikc8_pte_start[] asm("glimmermultikc8_pte_start");
extern const uint8_t glimmermultikc8_pte_end[] asm("glimmermultikc8_pte_end");
extern const uint8_t decode4_pte_start[] asm("decode4_pte_start");
extern const uint8_t decode4_pte_end[] asm("decode4_pte_end");
extern "C" void esp32_backend_report(void);
extern "C" uint32_t esp32_backend_split_jobs(void);
extern "C" uint32_t esp32_backend_affine_ops(void);
extern "C" uint32_t esp32_backend_i8_fast_rows(void);
extern "C" void esp32_backend_report_blocking(void);
extern const uint8_t mlpml_pte_start[] asm("mlpml_pte_start");
extern const uint8_t mlppt_pte_start[] asm("mlppt_pte_start");
extern const uint8_t mlppt_pte_end[] asm("mlppt_pte_end");
extern const uint8_t cvpt_pte_start[] asm("cvpt_pte_start");
extern const uint8_t cvpt_pte_end[] asm("cvpt_pte_end");
extern const uint8_t actshare_pte_start[] asm("actshare_pte_start");
extern const uint8_t actshare_pte_end[] asm("actshare_pte_end");
extern const uint8_t pt2eties_pte_start[] asm("pt2eties_pte_start");
extern const uint8_t pt2eties_pte_end[] asm("pt2eties_pte_end");
extern const uint8_t blkwide_pte_start[] asm("blkwide_pte_start");
extern const uint8_t blkwide_pte_end[] asm("blkwide_pte_end");
extern const uint8_t mlpml_pte_end[] asm("mlpml_pte_end");
extern "C" void esp32_backend_set_force_scalar(int on);
extern "C" void esp32_backend_set_split(int on);
extern "C" void esp32_backend_report_timing(void);
extern "C" void esp32_backend_release_weight_cache(void);
extern "C" void esp32_backend_weight_cache_stats(uint32_t* copies, uint32_t* hits, size_t* resident);
// So the extension test cannot go stale as new record types land.
extern "C" unsigned esp32_ext_max_known(void);
extern "C" unsigned esp32_num_opcodes(void);
extern "C" uint64_t esp32_backend_actq_rows_saved(void);
extern "C" uint64_t esp32_backend_act_per_tensor_ops(void);
extern const uint8_t convnet_pte_start[] asm("convnet_pte_start");
extern const uint8_t convnet_pte_end[] asm("convnet_pte_end");
extern const uint8_t block_pte_start[] asm("block_pte_start");
extern const uint8_t block_pte_end[] asm("block_pte_end");
extern const uint8_t lstm_pte_start[] asm("lstm_pte_start");
extern const uint8_t lstm_pte_end[] asm("lstm_pte_end");
extern const uint8_t grp8p_pte_start[] asm("grp8p_pte_start");
extern const uint8_t grp8p_pte_end[] asm("grp8p_pte_end");
extern const uint8_t cvresq_pte_start[] asm("cvresq_pte_start");
extern const uint8_t cvresq_pte_end[] asm("cvresq_pte_end");
extern const uint8_t cvresp_pte_start[] asm("cvresp_pte_start");
extern const uint8_t cvresp_pte_end[] asm("cvresp_pte_end");
extern const uint8_t cvrvp_pte_start[] asm("cvrvp_pte_start");
extern const uint8_t cvrvp_pte_end[] asm("cvrvp_pte_end");
extern const uint8_t cvrv_pte_start[] asm("cvrv_pte_start");
extern const uint8_t cvrv_pte_end[] asm("cvrv_pte_end");
extern const uint8_t cvst_pte_start[] asm("cvst_pte_start");
extern const uint8_t cvst_pte_end[] asm("cvst_pte_end");
extern const uint8_t ln2d_pte_start[] asm("ln2d_pte_start");
extern const uint8_t ln2d_pte_end[] asm("ln2d_pte_end");
extern const uint8_t ln2dx_pte_start[] asm("ln2dx_pte_start");
extern const uint8_t ln2dx_pte_end[] asm("ln2dx_pte_end");
extern const uint8_t cvres_pte_start[] asm("cvres_pte_start");
extern const uint8_t cvres_pte_end[] asm("cvres_pte_end");
extern const uint8_t cvi4_pte_start[] asm("cvi4_pte_start");
extern const uint8_t cvi4_pte_end[] asm("cvi4_pte_end");
extern const uint8_t g8i4_pte_start[] asm("g8i4_pte_start");
extern const uint8_t g8i4_pte_end[] asm("g8i4_pte_end");
extern const uint8_t g8i16_pte_start[] asm("g8i16_pte_start");
extern const uint8_t g8i16_pte_end[] asm("g8i16_pte_end");
extern const uint8_t cvi16_pte_start[] asm("cvi16_pte_start");
extern const uint8_t cvi16_pte_end[] asm("cvi16_pte_end");
extern const uint8_t grprp_pte_start[] asm("grprp_pte_start");
extern const uint8_t grprp_pte_end[] asm("grprp_pte_end");
extern const uint8_t grpr_pte_start[] asm("grpr_pte_start");
extern const uint8_t grpr_pte_end[] asm("grpr_pte_end");
extern const uint8_t grp8_pte_start[] asm("grp8_pte_start");
extern const uint8_t grp8_pte_end[] asm("grp8_pte_end");
extern const uint8_t grp_pte_start[] asm("grp_pte_start");
extern const uint8_t grp_pte_end[] asm("grp_pte_end");
extern const uint8_t grpe_pte_start[] asm("grpe_pte_start");
extern const uint8_t grpe_pte_end[] asm("grpe_pte_end");
extern const uint8_t gru_pte_start[] asm("gru_pte_start");
extern const uint8_t gru_pte_end[] asm("gru_pte_end");
extern const uint8_t conv1d_pte_start[] asm("conv1d_pte_start");
extern const uint8_t conv1d_pte_end[] asm("conv1d_pte_end");
extern const uint8_t pool_pte_start[] asm("pool_pte_start");
extern const uint8_t pool_pte_end[] asm("pool_pte_end");
extern const uint8_t mamba_pte_start[] asm("mamba_pte_start");
extern const uint8_t mamba_pte_end[] asm("mamba_pte_end");
extern const uint8_t kan_pte_start[] asm("kan_pte_start");
extern const uint8_t kan_pte_end[] asm("kan_pte_end");
extern const uint8_t looped_pte_start[] asm("looped_pte_start");
extern const uint8_t looped_pte_end[] asm("looped_pte_end");
extern const uint8_t acts_pte_start[] asm("acts_pte_start");
extern const uint8_t acts_pte_end[] asm("acts_pte_end");
extern const uint8_t norms_pte_start[] asm("norms_pte_start");
extern const uint8_t norms_pte_end[] asm("norms_pte_end");
extern const uint8_t memplan_pte_start[] asm("memplan_pte_start");
extern const uint8_t memplan_pte_end[] asm("memplan_pte_end");
extern const uint8_t ropefull_pte_start[] asm("ropefull_pte_start");
extern const uint8_t ropefull_pte_end[] asm("ropefull_pte_end");
extern const uint8_t attntail_pte_start[] asm("attntail_pte_start");
extern const uint8_t attntail_pte_end[] asm("attntail_pte_end");
extern const uint8_t gcn_pte_start[] asm("gcn_pte_start");
extern const uint8_t gcn_pte_end[] asm("gcn_pte_end");
// Program-data separated pair: the program alone, and its weights.
extern const uint8_t block_split_pte_start[] asm("block_split_pte_start");
extern const uint8_t block_split_pte_end[] asm("block_split_pte_end");

// ---- arenas -----------------------------------------------------------------
// 64 KB, sized from measured peaks rather than guessed:
//
//     mlp 881 B · convnet 3,544 · block 8,449 · block_split 8,809 · delegated 759-9,029
//     lstm 26,364 · gru 29,915
//
// The transformer/conv/mlp models peak under 9 KB, but the coverage
// zoo's recurrent models are unrolled by export into ~190-205 node
// graphs, which need about three times that (lstm and gru above).
// This is EValue and tensor-metadata bookkeeping, so it scales with
// NODE COUNT, not with tensor bytes: gru's 51 KB .pte needs 3.4x the
// arena of block's 234 KB one.
//
// 32 KB would leave gru (29,915 B) 9% headroom, which is not a margin, it
// is a coincidence. The coverage zoo needs 48 KB, and 64 KB covers it with
// room to spare, without a second fragmented heap block.
//
// Kept static rather than heap-allocated because it is modest and lives for the whole
// run; heap fragmentation matters for the ~136 KB planned arenas, not for this.
// Anything needing more allocates and frees its own arena per model, as
// run_cached_sequence does with its 128 KB.
static uint8_t __attribute__((aligned(16))) method_arena[64 * 1024];
// Fast half of the per-instruction TEMP allocator. Delegate calls whose packed
// scratch fits here stay in internal SRAM; flash-scale global attention falls
// through to its caller's PSRAM tier (run_story's, run_cached_sequence's). ExecuTorch resets both after
// every kernel/delegate call, so one buffer serves every layer.
static uint8_t __attribute__((aligned(16))) temp_arena[32 * 1024];

// MemoryAllocator exposes capacity via size() but no usage accessor, and the bytes
// ExecuTorch actually demands is the number the arena sizes come from.
// Sum-of-requests, so it slightly undercounts alignment padding.
class TrackingAllocator final : public MemoryAllocator {
 public:
  TrackingAllocator(size_t size, uint8_t* base) : MemoryAllocator(size, base) {}
  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* p = MemoryAllocator::allocate(size, alignment);
    if (p != nullptr) peak_ += size;
    return p;
  }
  size_t peak() const { return peak_; }

 private:
  size_t peak_ = 0;
};

class TieredTempAllocator final : public MemoryAllocator {
 public:
  TieredTempAllocator(size_t fast_size, uint8_t* fast_base,
                      size_t large_size = 0, uint8_t* large_base = nullptr)
      : MemoryAllocator(0, nullptr),
        fast_(fast_size, fast_base),
        large_(large_size, large_base) {}

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    // The backend packs all of one delegate's scratch into one allocation and
    // ET resets us after that call. Decide by capacity up front so trying the
    // fast tier for a known-large request does not emit a spurious OOM log.
    if (size <= fast_.size()) {
      if (void* p = fast_.allocate(size, alignment)) {
        record(p, fast_, size, fast_used_);
        return p;
      }
    }
    void* p = large_.allocate(size, alignment);
    if (p != nullptr) {
      record(p, large_, size, large_used_);
    }
    return p;
  }

  void reset() override {
    fast_.reset();
    large_.reset();
    fast_used_ = 0;
    large_used_ = 0;
  }

  // Method::resolve_operator() treats a zero-capacity temp allocator as absent
  // and permanently charges its transient TensorMeta allocations to the method
  // arena. The wrapper's base MemoryAllocator is intentionally empty, so expose
  // the largest contiguous tier rather than inheriting its zero size.
  uint32_t size() const override {
    return fast_.size() > large_.size() ? fast_.size() : large_.size();
  }

  size_t peak() const { return peak_; }
  void reset_peak() { peak_ = fast_used_ + large_used_; }

 private:
  void record(void* p, const MemoryAllocator& allocator, size_t bytes,
              size_t& tier_used) {
    tier_used = static_cast<uint8_t*>(p) - allocator.base_address() + bytes;
    const size_t total = fast_used_ + large_used_;
    if (total > peak_) {
      peak_ = total;
    }
  }

  MemoryAllocator fast_;
  MemoryAllocator large_;
  size_t fast_used_ = 0;
  size_t large_used_ = 0;
  size_t peak_ = 0;
};

class ScopedCapsBuffer final {
 public:
  ScopedCapsBuffer(size_t size, uint32_t caps)
      : data_(static_cast<uint8_t*>(heap_caps_aligned_alloc(16, size, caps))),
        size_(data_ != nullptr ? size : 0) {}
  ~ScopedCapsBuffer() {
    if (data_ != nullptr) {
      heap_caps_free(data_);
    }
  }
  uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

  ScopedCapsBuffer(const ScopedCapsBuffer&) = delete;
  ScopedCapsBuffer& operator=(const ScopedCapsBuffer&) = delete;

 private:
  uint8_t* data_;
  size_t size_;
};

// Fill inputs: fp32 gets a fixed ramp.
// `long_modulus` is nonzero only for the embedding-format fixture, whose indices
// permute every row before repeating. Other int64-input models retain their own setup.
static void fill_inputs(::executorch::runtime::Method& method,
                        int64_t long_modulus = 0) {
  EValue vals[4];
  const size_t n_in = method.inputs_size();
  // get_inputs checks n_in against the count it is given, not this array.
  if (n_in > sizeof(vals) / sizeof(vals[0])) {
    printf("  too many inputs: %u\n", (unsigned)n_in);
    return;
  }
  if (method.get_inputs(vals, n_in) != Error::Ok) {
    printf("  get_inputs FAILED\n");
    return;
  }
  for (size_t i = 0; i < n_in; i++) {
    Tensor t = vals[i].toTensor();
    if (t.scalar_type() == executorch::aten::ScalarType::Float) {
      float* p = t.mutable_data_ptr<float>();
      // Step is 2^-7, deliberately: exactly representable, so j*step - 0.5 is exact
      // and gives identical bits whether or not the compiler contracts this into a
      // madd.s. With 0.01f it does contract (one rounding, not two), which shifts
      // x[75] by one ulp -- invisible in fp32 but a whole quantum under int8.
      for (int j = 0; j < t.numel(); j++) p[j] = (float)j * 0.0078125f - 0.5f;
    } else if (t.scalar_type() == executorch::aten::ScalarType::Long &&
               long_modulus > 0) {
      int64_t* p = t.mutable_data_ptr<int64_t>();
      for (int j = 0; j < t.numel(); ++j) {
        p[j] = ((int64_t)j * 7 + 3) % long_modulus;
      }
    }
  }
}

// FNV-1a over an output's bytes. Two runs agree on this only if every bit of
// the output agrees, which a six-decimal checksum cannot show; the int8 K
// cache rows use it to prove they reproduce the path they replace exactly.
static uint32_t output_bits(const void* data, size_t bytes,
                            uint32_t hash = 2166136261u) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ p[i]) * 16777619u;
  }
  return hash;
}

// The int64-input modulus for a row whose program is not one of the embedded
// fixtures below (run_asym_fixtures sets it per row; 0 = the rules below).
static int64_t g_long_modulus_override = 0;
// Whether run_model prints the largest free blocks before it allocates the
// planned buffers (run_asym_fixtures sets it for the rows that ask).
static bool g_report_largest_block = false;

// `method_name` picks a method of a multi-method program (default: the first).
// `all_outputs` treats every output as one flat tensor: the checksum skips
// non-finite values (a row may hold NaN and infinities on purpose) and the
// output bits cover every byte, in output order. It also prints how many
// elements the table op processed, which run_parity requires to be nonzero.
// `dump_values` (with all_outputs) also prints every output element's bits,
// eight to a "  values:" line, for rows held to at most one code: run_parity
// compares them element by element. `timed_runs` executes the method that
// many more times and prints the time they took: plain QEMU's clock is the
// host's, but under `-icount shift=0` every instruction is one virtual
// nanosecond, so the line counts instructions.
static void run_model(
    const char* label, const uint8_t* pte, size_t pte_size,
    ::executorch::runtime::DataLoader* wts_loader = nullptr,
    const char* method_name = nullptr, bool all_outputs = false,
    bool dump_values = false, int timed_runs = 0) {
  printf("\n--- %s (%u B) ---\n", label, (unsigned)pte_size);  // run_parity's row key: "--- <name>.pte"
#if defined(ET_PARITY_TIER_BYTES)
  {
    // The budgeted silicon-parity pass: every row's external weights that fit
    // are read from PSRAM copies, made at its load and dropped with it (no
    // cache, so each row stages afresh), and its bits must not move.
    executorch::runtime::BackendOptions<3> tier;
    tier.set_option("weight_tier_budget", (int)ET_PARITY_TIER_BYTES);
    tier.set_option("weight_cache", false);
#if defined(ET_PARITY_TIER_RANK)
    // Ranked placement with nothing applying it: each row's first execute
    // makes its copies, the path of an application that never applies.
    tier.set_option("weight_tier_rank", true);
#endif
    (void)executorch::runtime::set_option("Esp32Backend", tier.view());
  }
#endif
#if !defined(ET_PLE_ONLY)
  // The program's lut16 tables (the op's and qnorm's) are its constants:
  // what lut16::checked() remembers of them, and their internal-RAM copies,
  // go when this returns -- after the program, declared later, is gone.
  struct ForgetLut16Tables {
    ~ForgetLut16Tables() { espressif::lut16::forget_tables(); }
  } forget_lut16_tables;
#endif
  const bool is_embedding_format_fixture =
      pte == embi4_pte_start || pte == embi16_pte_start;
  const bool is_selected_head_fixture = pte == selhead_pte_start;
  const bool is_embed_linear_fixture = pte == embedlin_pte_start;

  BufferDataLoader loader(pte, pte_size);
  auto program = Program::load(&loader);
  if (!program.ok()) {
    printf("  Program::load FAILED: %d\n", (int)program.error());
    return;
  }
  const char* mname =
      method_name != nullptr ? method_name : *program->get_method_name(0);
  auto meta = program->method_meta(mname);
  if (!meta.ok()) {
    printf("  method_meta FAILED: %d\n", (int)meta.error());
    return;
  }

  // Heap, not a static arena. A 136 KB BSS array permanently shrinks the internal heap
  // and, worse, fragments what is left: the decode model needs ~85 KB of planned
  // buffers plus a >50 KB contiguous method arena, and with a static array in place
  // the largest free block is only 55 KB even at boot.
  Span<uint8_t> spans[4];
  uint8_t* planned[4] = {nullptr, nullptr, nullptr, nullptr};
  const size_t n_planned = meta->num_memory_planned_buffers();
  // HierarchicalAllocator below is handed n_planned spans from this array.
  if (n_planned > 4) {
    printf("  too many planned buffers: %u\n", (unsigned)n_planned);
    return;
  }
  if (g_report_largest_block) {
    // What the planned buffers are about to be carved from: the internal
    // heap's largest block, and the largest of any 8-bit heap (PSRAM too).
    printf("  largest free block: internal %u B, any %u B\n",
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL |
                                                      MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  }
  size_t used = 0;
  for (size_t i = 0; i < n_planned; i++) {
    size_t want = (size_t)*meta->memory_planned_buffer_size(i);
    planned[i] = (uint8_t*)heap_caps_aligned_alloc(16, (want + 15u) & ~15u,
                                                   MALLOC_CAP_8BIT);
    if (planned[i] == nullptr) {
      printf("  planned buffer %u alloc FAILED (%u B)\n", (unsigned)i,
             (unsigned)want);
      for (size_t j = 0; j < i; j++) {
        heap_caps_free(planned[j]);
      }
      return;
    }
    spans[i] = Span<uint8_t>(planned[i], want);
    used += want;
  }
  printf("  planned buffers: %u, %u B total\n", (unsigned)n_planned,
         (unsigned)used);

  TrackingAllocator method_allocator(sizeof(method_arena), method_arena);
  TieredTempAllocator temp_allocator(sizeof(temp_arena), temp_arena);
  HierarchicalAllocator planned_allocator({spans, n_planned});
  MemoryManager memory_manager(
      &method_allocator, &planned_allocator, &temp_allocator);

  // When weights live outside the program, they arrive through a NamedDataMap.
  // The loader and the map must outlive the method, hence the scope here.
  ::executorch::runtime::Result<executorch::extension::FlatTensorDataMap> wts =
      wts_loader != nullptr
          ? executorch::extension::FlatTensorDataMap::load(wts_loader)
          : ::executorch::runtime::Result<
                executorch::extension::FlatTensorDataMap>(
                ::executorch::runtime::Error::NotFound);
  const ::executorch::runtime::NamedDataMap* ndm = nullptr;
  if (wts_loader != nullptr) {
    if (!wts.ok()) {
      printf("  FlatTensorDataMap::load FAILED: %d\n", (int)wts.error());
      return;
    }
    ndm = &wts.get();
    auto nkeys = wts->get_num_keys();
    auto wsz = wts_loader->size();
    printf("  external weights: %u tensors, from a %u B window\n",
           nkeys.ok() ? (unsigned)*nkeys : 0u,
           wsz.ok() ? (unsigned)*wsz : 0u);
  }

  auto method = program->load_method(mname, &memory_manager, nullptr, ndm);
  if (!method.ok()) {
    printf("  load_method FAILED: %d\n", (int)method.error());
    return;
  }
  const size_t load_temp_peak = temp_allocator.peak();
  temp_allocator.reset_peak();
  printf("  method '%s': %u in, %u out, method_arena peak %u B\n", mname,
         (unsigned)method->inputs_size(), (unsigned)method->outputs_size(),
         (unsigned)method_allocator.peak());
  // Taken with the method ALIVE and weights reachable. Bracketing the whole scoped
  // run instead would return to baseline whether or not the weights were copied,
  // so it would prove nothing.
  // (freed at the end of this function)
  printf("  internal free with method live: %u B\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

  // Int64 inputs are filled as (j*7+3) mod this; each fixture's exporter
  // computed its reference from the same rule, so the modulus is part of the
  // row's contract rather than a runner detail.
  int64_t long_modulus = 0;
  if (g_long_modulus_override > 0) {
    long_modulus = g_long_modulus_override;
  } else if (is_embedding_format_fixture || is_embed_linear_fixture) {
    long_modulus = 23;   // table height
  } else if (is_selected_head_fixture) {
    long_modulus = 257;  // full output table height
  }
  fill_inputs(*method, long_modulus);

  const uint64_t lut_elements = espressif::quant_lut::stats().elements;
  Error e = method->execute();
  printf("  execute: %d (0 == Ok)\n", (int)e);
  if (e != Error::Ok) return;
  if (all_outputs) {
    printf("  quant_lut elements: %llu\n",
           (unsigned long long)(espressif::quant_lut::stats().elements -
                                lut_elements));
  }
  const size_t execute_temp_peak = temp_allocator.peak();
  printf("  temp peak: load %u B, execute %u B\n",
         (unsigned)load_temp_peak, (unsigned)execute_temp_peak);

  EValue outs[32];
  const size_t n_out = method->outputs_size();
  if (all_outputs && n_out <= sizeof(outs) / sizeof(outs[0]) &&
      method->get_outputs(outs, n_out) == Error::Ok) {
    double sum = 0;
    uint32_t bits = 2166136261u;
    size_t index = 0;
    bool floats = true;
    for (size_t k = 0; k < n_out; ++k) {
      Tensor t = outs[k].toTensor();
      floats &= t.scalar_type() == executorch::aten::ScalarType::Float;
      if (!floats) break;
      const float* p = t.const_data_ptr<float>();
      for (int i = 0; i < t.numel(); ++i, ++index) {
        if (std::isfinite(p[i])) sum += (double)p[i] * (double)(index + 1);
      }
      bits = output_bits(p, (size_t)t.nbytes(), bits);
    }
    if (floats) {
      printf("  OUTPUT %u floats in %u outputs\n", (unsigned)index,
             (unsigned)n_out);
      printf("  checksum: %.6f\n", sum);
      printf("  output bits: %08lx\n", (unsigned long)bits);
      for (size_t k = 0; dump_values && k < n_out; ++k) {
        Tensor t = outs[k].toTensor();
        const float* p = t.const_data_ptr<float>();
        for (int i = 0; i < t.numel(); i += 8) {
          printf("  values:");
          for (int j = i; j < i + 8 && j < t.numel(); ++j) {
            uint32_t word;
            memcpy(&word, p + j, sizeof word);
            printf(" %08lx", (unsigned long)word);
          }
          printf("\n");
        }
      }
    } else {
      printf("  OUTPUT not all float32\n");
    }
  } else if (n_out >= 1 && method->get_output(0).isTensor()) {
    // Output 0 alone, read by index: get_outputs(outs, n) refuses an array
    // shorter than the method's outputs (InvalidArgument), so a row whose
    // program also returns something else (attnqk_qk and attnkc8_qk: the raw
    // scores, then the attention) printed nothing and went MISSING.
    Tensor t = method->get_output(0).toTensor();
    if (t.scalar_type() == executorch::aten::ScalarType::Float) {
      const float* p = t.const_data_ptr<float>();
      printf("  OUTPUT %d floats:", (int)t.numel());
      for (int i = 0; i < t.numel() && i < 8; i++) printf(" %.6f", p[i]);
      double sum = 0;
      for (int i = 0; i < t.numel(); i++) sum += (double)p[i] * (i + 1);
      printf("\n  checksum: %.6f\n", sum);
      printf("  output bits: %08lx\n",
             (unsigned long)output_bits(p, (size_t)t.nbytes()));
    } else {
      printf("  OUTPUT not float32\n");
    }
  }
  if (timed_runs > 0) {
    const int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < timed_runs; ++r) {
      (void)method->execute();
    }
    printf("  timed: %d run(s) in %lld us\n", timed_runs,
           (long long)(esp_timer_get_time() - t0));
  }
  // Give the planned buffers back. Holding them would fragment the heap for every
  // model that runs after this one, enough to make the decode step fail.
  for (size_t i = 0; i < n_planned && i < 4; i++) {
    heap_caps_free(planned[i]);
  }
}

// Does the blob validator actually reject a corrupt blob? A validator that has
// only ever seen valid input proves nothing, and this one cannot be reached
// from the AOT side (our exporter cannot emit the fields it rejects), so the
// only way to test it is to corrupt a real artifact and try to load it.
//
// Three cases, and the POSITIVE control matters as much as the negatives: if
// the RAM copy failed to load for some unrelated reason, both negatives would
// "pass" while proving nothing at all.
//
// Blob layout (esp32_backend.cpp): magic, version, n_ops, then OpDesc[0], whose
// fields run opcode, flags, m, k, n, k_pad -- so k_pad sits at magic+32 and
// n_ops at magic+8.
// ExecuTorch's runtime option API, driven the way an application would: by
// backend NAME, through the free functions in runtime/backend/interface.h, with
// no privileged access to this backend's internals.
//
// Worth a row of its own rather than trusting the three split models below,
// because those only prove that SOMETHING turned the split on. What matters for
// an option is the whole contract: the value survives a round trip, an unknown
// key is refused rather than silently swallowed (the failure mode where a user
// believes min-latency is on and it is not), an out-of-domain int is refused,
// and -1 is expressible so a program can be handed back to its own compile spec.
static void run_backend_options_check() {
  printf("\n--- backend options (set_option/get_option) ---\n");
  using executorch::runtime::BackendOption;
  using executorch::runtime::BackendOptions;

  auto read_back = [](const char* what) {
    BackendOptions<1> q;
    q.set_option("min_latency", -1);  // value is overwritten by the backend
    Error err = executorch::runtime::get_option("Esp32Backend", q.view());
    int got = -99;
    if (err == Error::Ok) {
      (void)q.get_option("min_latency", got);
    }
    printf("  %s: get_option -> %d (err %d)\n", what, got, (int)err);
    return got;
  };

  BackendOptions<1> on;
  on.set_option("min_latency", true);          // the bool spelling
  Error e = executorch::runtime::set_option("Esp32Backend", on.view());
  printf("  set min_latency=true: %d\n", (int)e);
  const int after_on = read_back("after true");

  BackendOptions<1> follow;
  follow.set_option("min_latency", -1);        // the tri-state spelling
  e = executorch::runtime::set_option("Esp32Backend", follow.view());
  printf("  set min_latency=-1: %d\n", (int)e);
  const int after_follow = read_back("after -1");

  // Negative cases. Both must be REFUSED, and neither may disturb the value that
  // a successful call established -- a rejected option that half-applied would
  // be worse than one that silently ignored.
  BackendOptions<1> bogus;
  bogus.set_option("no_such_option", true);
  const Error e_key = executorch::runtime::set_option("Esp32Backend", bogus.view());
  BackendOptions<1> oob;
  oob.set_option("min_latency", 7);
  const Error e_dom = executorch::runtime::set_option("Esp32Backend", oob.view());
  const int after_bad = read_back("after both rejects");

  printf("  unknown key rejected: %s (err %d)\n",
         e_key != Error::Ok ? "yes" : "NO", (int)e_key);
  printf("  out-of-domain int rejected: %s (err %d)\n",
         e_dom != Error::Ok ? "yes" : "NO", (int)e_dom);
  printf("  options roundtrip: %s\n",
         (after_on == 1 && after_follow == -1 && after_bad == -1
          && e_key != Error::Ok && e_dom != Error::Ok) ? "PASS" : "FAIL");
}

static size_t find_esp32_blob(const uint8_t* pte, size_t size) {
  const uint8_t magic[4] = {0x44, 0x32, 0x33, 0x45};
  for (size_t i = 0; i + sizeof(magic) <= size; ++i) {
    if (memcmp(pte + i, magic, sizeof(magic)) == 0) {
      return i;
    }
  }
  return SIZE_MAX;
}

static size_t find_esp32_blob_with_opcode(const uint8_t* pte, size_t size,
                                          uint32_t wanted,
                                          uint32_t* found_op_index) {
  const uint8_t magic[4] = {0x44, 0x32, 0x33, 0x45};
  for (size_t at = 0; at + 16 <= size; ++at) {
    if (memcmp(pte + at, magic, sizeof(magic)) != 0) {
      continue;
    }
    uint32_t n_ops = 0;
    memcpy(&n_ops, pte + at + 8, sizeof(n_ops));
    if (n_ops > (size - at - 16) / 64) {
      continue;
    }
    for (uint32_t i = 0; i < n_ops; ++i) {
      uint32_t opcode = 0;
      memcpy(&opcode, pte + at + 16 + (size_t)i * 64, sizeof(opcode));
      if (opcode == wanted) {
        *found_op_index = i;
        return at;
      }
    }
  }
  return SIZE_MAX;
}

static bool load_validation_pte(const char* what, uint8_t* bytes, size_t size) {
  BufferDataLoader loader(bytes, size);
  auto program = Program::load(&loader);
  if (!program.ok()) {
    printf("  %s: Program::load rejected (%d)\n", what, (int)program.error());
    return false;
  }
  const char* mname = *program->get_method_name(0);
  auto meta = program->method_meta(mname);
  if (!meta.ok()) {
    printf("  %s: method_meta failed\n", what);
    return false;
  }
  Span<uint8_t> spans[2];
  uint8_t* planned[2] = {nullptr, nullptr};
  const size_t np = meta->num_memory_planned_buffers();
  if (np > 2) {
    printf("  %s: too many planned buffers: %u\n", what, (unsigned)np);
    return false;
  }
  for (size_t i = 0; i < np; ++i) {
    const size_t want = (size_t)*meta->memory_planned_buffer_size(i);
    planned[i] = (uint8_t*)heap_caps_aligned_alloc(
        16, (want + 15u) & ~15u, MALLOC_CAP_8BIT);
    spans[i] = Span<uint8_t>(planned[i], want);
  }
  TrackingAllocator alloc(sizeof(method_arena), method_arena);
  TieredTempAllocator temp_allocator(sizeof(temp_arena), temp_arena);
  HierarchicalAllocator planner({spans, np});
  MemoryManager mm(&alloc, &planner, &temp_allocator);
  auto method = program->load_method(mname, &mm, nullptr, nullptr);
  const bool ok = method.ok();
  printf("  %s: load_method %s\n", what, ok ? "ACCEPTED" : "REJECTED");
  for (size_t i = 0; i < np && i < 2; ++i) {
    heap_caps_free(planned[i]);
  }
  return ok;
}

static void run_blob_validation_check() {
  printf("\n--- blob validation (corrupt-input rejection) ---\n");
  const size_t sz = (size_t)(mlpdel_pte_end - mlpdel_pte_start);
  uint8_t* copy = (uint8_t*)heap_caps_aligned_alloc(16, sz, MALLOC_CAP_8BIT);
  if (copy == nullptr) {
    printf("  alloc FAILED\n");
    return;
  }

  // Locate the delegate blob by its magic ('E32D' stored little-endian).
  const size_t at = find_esp32_blob(mlpdel_pte_start, sz);
  if (at == SIZE_MAX) {
    printf("  blob magic not found in the .pte\n");
    heap_caps_free(copy);
    return;
  }

  auto try_load = [&](const char* what, size_t off, uint32_t patch) {
    memcpy(copy, mlpdel_pte_start, sz);
    if (off != 0) {
      memcpy(copy + off, &patch, 4);
    }
    return load_validation_pte(what, copy, sz);
  };

  const bool clean = try_load("unmodified control", 0, 0);
  // k_pad = 32 against the model's k = 96: the out-of-bounds WRITE, one bit
  // away from the shipped descriptor. k_pad is OpDesc word 5, at +16+20 (not
  // +32, op 0's n: the chain-shape check would refuse that first and leave the
  // k > k_pad guard with no device coverage).
  const bool kpad = try_load("k > k_pad", at + 36, 32u);
  // Nudge one op-0 offset by a few bytes: still inside the blob, so every
  // bounds check passes, but misaligned for the elements it addresses.
  // A misaligned float load faults on this target.
  auto try_shift = [&](const char* what, uint32_t word, uint32_t delta) {
    memcpy(copy, mlpdel_pte_start, sz);
    uint32_t value = 0;
    memcpy(&value, copy + at + 16 + word * 4, 4);
    value += delta;
    memcpy(copy + at + 16 + word * 4, &value, 4);
    return load_validation_pte(what, copy, sz);
  };
  const bool scale_align = try_shift("op0 fp32 scales misaligned", 8, 1);
  const bool bias_align = try_shift("op0 bias misaligned", 6, 2);
  // Op 1 reads chain slot 0 (op 0's relu output); make it write that slot
  // too. Every chain check passes -- the slot has a writer and the element
  // counts agree -- yet the op would overwrite its input while reading it.
  auto try_alias = [&](const char* what) {
    memcpy(copy, mlpdel_pte_start, sz);
    uint8_t* op1 = copy + at + 16 + 64;
    memcpy(op1 + 10 * 4, op1 + 9 * 4, 4);  // out_arg = in_arg
    return load_validation_pte(what, copy, sz);
  };
  const bool alias = try_alias("op1 writes the slot it reads");
  // n_ops such that 64 * n_ops wraps uint32 to zero.
  const bool nops = try_load("n_ops overflow", at + 8, 0x04000000u);
  // Extension table. Records no exporter emits (an unknown type, a retired one,
  // a length past the end) are exercised by SYNTHESISING them into a real .pte,
  // the same way the descriptor corruption above is: a validator tested only on
  // what exporters emit would meet its first malformed record untested, in the
  // field.
  //
  // `ext_off` is header word 3. Each case writes its table into a copy of the
  // program of the same size (why not appended: below).
  auto try_ext = [&](const char* what, uint16_t type, uint32_t len,
                     bool past_end) {
    uint8_t* big = (uint8_t*)heap_caps_aligned_alloc(16, sz, MALLOC_CAP_8BIT);
    if (big == nullptr) return false;
    memcpy(big, mlpdel_pte_start, sz);
    // The table goes INSIDE the blob, immediately after the descriptors, which
    // overwrites the first few weight bytes. That is fine here because this only
    // load_method()s and never executes -- and it is necessary, because the
    // delegate payload's length is fixed in the .pte flatbuffer, so appending
    // past the file does NOT extend what the delegate sees: appended, every case
    // would come back "offset past the blob", including the one that must be
    // accepted.
    uint32_t nops_hdr = 0;
    memcpy(&nops_hdr, big + at + 8, 4);
    const uint32_t table = 16u + 64u * nops_hdr;
    uint32_t one = 1;
    memcpy(big + at + table, &one, 4);
    uint16_t op_index = 0;
    memcpy(big + at + table + 4, &op_index, 2);
    memcpy(big + at + table + 6, &type, 2);
    uint32_t plen = past_end ? 0xFFFFFF00u : len;
    memcpy(big + at + table + 8, &plen, 4);
    memcpy(big + at + 12, &table, 4);            // ext_off in the header
    char label[96];
    snprintf(label, sizeof(label), "ext %s", what);
    const bool ok = load_validation_pte(label, big, sz);
    heap_caps_free(big);
    return ok;
  };
  // High bit set = optional. An unknown optional record must be SKIPPED, which
  // is what makes the format extensible without a version bump per addition.
  // DERIVED, not written down: a literal type stops testing "unknown" as soon
  // as a record of that type is assigned, and silently starts testing
  // "malformed", which still fails loudly but for the wrong reason. One past
  // the highest known type is unknown by construction, whatever is added
  // next.
  const uint16_t unknown = (uint16_t)(esp32_ext_max_known() + 1);
  const bool ext_opt = try_ext("unknown OPTIONAL skipped",
                               (uint16_t)(unknown | 0x8000u), 4, false);
  // High bit clear = required. Unknown means this runtime cannot perform the
  // computation the blob describes, so it must refuse rather than run the
  // descriptor alone and silently do something else.
  const bool ext_req = try_ext("unknown REQUIRED refused", unknown, 4, false);
  // Types 1--5 are retired, unassigned holes below the known ones: a validator
  // comparing only with the maximum assigned id would accept them as required
  // records, and the parser would then silently ignore them.
  const bool ext_retired = try_ext("retired REQUIRED refused", 1, 4, false);
  // A length that runs past the blob.
  const bool ext_oob = try_ext("payload past the end refused",
                              (uint16_t)(unknown | 0x8000u), 4, true);

  // A real dynamic-BMM artifact. The descriptor stores B in scale_group, and
  // every tensor extent must include it, not only M*K and N*K.
  auto try_bmm_batch = [&](const char* what, bool corrupt_batch) {
    const size_t bsz = (size_t)(attnqkq_pte_end - attnqkq_pte_start);
    uint8_t* big = (uint8_t*)heap_caps_aligned_alloc(16, bsz, MALLOC_CAP_8BIT);
    if (big == nullptr) return false;
    memcpy(big, attnqkq_pte_start, bsz);
    uint32_t bmm_op = 0;
    const size_t bat = find_esp32_blob_with_opcode(big, bsz, 8, &bmm_op);
    if (bat == SIZE_MAX) {
      heap_caps_free(big);
      return false;
    }
    if (corrupt_batch) {
      const uint32_t huge_batch = 0x40000000u;
      // Header 16 B, then OpDesc.scale_group at word 11.
      memcpy(big + bat + 16u + (size_t)bmm_op * 64u + 11u * 4u,
             &huge_batch, sizeof(huge_batch));
    }
    const bool ok = load_validation_pte(what, big, bsz);
    heap_caps_free(big);
    return ok;
  };
  const bool bmm_clean = try_bmm_batch("bmm unmodified control", false);
  const bool bmm_batch = try_bmm_batch("bmm batch-product overflow", true);

  // Exercise target-width storage arithmetic on a copied, real i16 artifact.
  // Its own unmodified control ensures the fixture is loadable.
  auto try_desc_word = [&](const char* what, const uint8_t* start,
                           const uint8_t* end, uint32_t opcode, uint32_t word,
                           bool mutate, uint32_t value) {
    const size_t psz = (size_t)(end - start);
    uint8_t* big = (uint8_t*)heap_caps_aligned_alloc(16, psz, MALLOC_CAP_8BIT);
    if (big == nullptr) return false;
    memcpy(big, start, psz);
    uint32_t op_index = 0;
    const size_t pat = find_esp32_blob_with_opcode(
        big, psz, opcode, &op_index);
    if (pat == SIZE_MAX || word >= 16u) {
      heap_caps_free(big);
      return false;
    }
    if (mutate) {
      memcpy(big + pat + 16u + (size_t)op_index * 64u + word * 4u,
             &value, sizeof(value));
    }
    const bool ok = load_validation_pte(what, big, psz);
    heap_caps_free(big);
    return ok;
  };
  // OpDesc word 5 is k_pad. On the 32-bit target, i16 row_bytes computed in
  // size_t would wrap 0x80000000 * 2 to zero: the inline weight extent would
  // appear empty and the i16 scratch allocation would wrap to zero too.
  const bool i16_row_clean = try_desc_word(
      "i16 row unmodified control", mlpi16_pte_start, mlpi16_pte_end,
      4u, 5u, false, 0u);
  const bool i16_row = try_desc_word(
      "i16 row-byte overflow", mlpi16_pte_start, mlpi16_pte_end,
      4u, 5u, true, 0x80000000u);
  // OpDesc word 11 is scale_group. The grouped i8 fixture has k=131,
  // k_pad=144 and groups of 128: two groups either way. Groups of 136 store
  // the same two scales per row, so every size check passes, but count one
  // group over k against two over k_pad -- the stride ambiguity that
  // misattributes scales silently.
  const bool group_count_clean = try_desc_word(
      "group count unmodified control", grpi8_pte_start, grpi8_pte_end,
      2u, 11u, false, 0u);
  const bool group_count = try_desc_word(
      "group count differs over k and k_pad", grpi8_pte_start, grpi8_pte_end,
      2u, 11u, true, 136u);
  // A dynamic bmm stores no weights, so a long k_pad trips no extent check
  // and only sizes scratch. Past 132,096 an int8 dot overflows the int32
  // accumulator dot_i8_s3 and the scalar loops sum into; bmm_clean above is
  // this artifact's control.
  const bool dot_len = try_desc_word(
      "bmm dot past the exact int32 length", attnqkq_pte_start,
      attnqkq_pte_end, 8u, 5u, true, 132096u + 32u);

  // A real conv artifact, including its emitted type-6 record. One mutation
  // proves loop/divisor invariants live runtime-side; the other installs two
  // individually valid records for the same op and proves there is no silent
  // last-record-wins rule.
  auto try_conv_geom = [&](const char* what, int mutation) {
    const size_t csz = (size_t)(convd_pte_end - convd_pte_start);
    uint8_t* big = (uint8_t*)heap_caps_aligned_alloc(16, csz, MALLOC_CAP_8BIT);
    if (big == nullptr) return false;
    memcpy(big, convd_pte_start, csz);
    const size_t cat = find_esp32_blob(big, csz);
    if (cat == SIZE_MAX) {
      heap_caps_free(big);
      return false;
    }
    uint32_t ext = 0;
    memcpy(&ext, big + cat + 12, sizeof(ext));
    if (ext == 0 || cat + ext + 12 > csz) {
      heap_caps_free(big);
      return false;
    }
    if (mutation == 1) {
      // count word + 8-byte record header + stride_h (field 8) in 24H payload.
      const uint16_t zero = 0;
      memcpy(big + cat + ext + 4u + 8u + 8u * 2u, &zero, sizeof(zero));
    } else if (mutation == 2) {
      uint32_t len = 0, nops_hdr = 0;
      memcpy(&len, big + cat + ext + 4u + 4u, sizeof(len));
      memcpy(&nops_hdr, big + cat + 8u, sizeof(nops_hdr));
      const uint32_t record_bytes = 8u + ((len + 3u) & ~3u);
      const uint32_t table = 16u + 64u * nops_hdr;
      uint8_t record[64];
      if (record_bytes > sizeof(record) ||
          table + 4u + 2u * record_bytes > ext) {
        heap_caps_free(big);
        return false;
      }
      memcpy(record, big + cat + ext + 4u, record_bytes);
      const uint32_t two = 2;
      memcpy(big + cat + table, &two, sizeof(two));
      memcpy(big + cat + table + 4u, record, record_bytes);
      memcpy(big + cat + table + 4u + record_bytes, record, record_bytes);
      memcpy(big + cat + 12u, &table, sizeof(table));
    }
    const bool ok = load_validation_pte(what, big, csz);
    heap_caps_free(big);
    return ok;
  };
  const bool conv_clean = try_conv_geom("conv unmodified control", 0);
  const bool conv_stride = try_conv_geom("conv zero stride", 1);
  const bool conv_duplicate = try_conv_geom("conv duplicate geometry", 2);

  // A real grouped-conv artifact has two independently valid geometry records.
  // Duplicating group_idx=0 passes every per-record check; accepted, both ops
  // would write the first output-channel slice and leave the second untouched.
  auto try_group_set = [&](const char* what, bool duplicate_group) {
    const size_t gsz = (size_t)(grp_pte_end - grp_pte_start);
    uint8_t* big = (uint8_t*)heap_caps_aligned_alloc(16, gsz, MALLOC_CAP_8BIT);
    if (big == nullptr) return false;
    memcpy(big, grp_pte_start, gsz);
    const size_t gat = find_esp32_blob(big, gsz);
    if (gat == SIZE_MAX) {
      heap_caps_free(big);
      return false;
    }
    uint32_t ext = 0, count = 0;
    memcpy(&ext, big + gat + 12u, sizeof(ext));
    if (ext == 0 || gat + ext + 4u > gsz) {
      heap_caps_free(big);
      return false;
    }
    memcpy(&count, big + gat + ext, sizeof(count));
    size_t record = gat + ext + 4u;
    uint32_t conv_records = 0;
    for (uint32_t r = 0; r < count; ++r) {
      if (record + 8u > gsz) break;
      uint16_t type = 0;
      uint32_t len = 0;
      memcpy(&type, big + record + 2u, sizeof(type));
      memcpy(&len, big + record + 4u, sizeof(len));
      const size_t padded = ((size_t)len + 3u) & ~(size_t)3u;
      const size_t next = record + 8u + padded;
      if (next > gsz) break;
      if ((type & 0x7fffu) == 6u && len == 48u) {
        ++conv_records;
        if (duplicate_group && conv_records == 2u) {
          const uint16_t zero = 0;
          // ConvGeom.group_idx is field 21 of the 24H payload.
          memcpy(big + record + 8u + 21u * 2u, &zero, sizeof(zero));
        }
      }
      record = next;
    }
    if (conv_records < 2u) {
      heap_caps_free(big);
      return false;
    }
    const bool ok = load_validation_pte(what, big, gsz);
    heap_caps_free(big);
    return ok;
  };
  const bool group_clean =
      try_group_set("grouped conv unmodified control", false);
  const bool group_duplicate =
      try_group_set("grouped conv duplicate index", true);

  // The selected-output extension changes both the output tensor extent and
  // which rows of the full learned table execute.  Exercise its invariants on
  // the real selected-head artifact: zero rows must not become a plausible
  // empty result, and two records must not silently use last-record-wins.
  auto try_selected_set = [&](const char* what, int mutation) {
    const size_t ssz = (size_t)(selhead_pte_end - selhead_pte_start);
    uint8_t* big = (uint8_t*)heap_caps_aligned_alloc(16, ssz, MALLOC_CAP_8BIT);
    if (big == nullptr) return false;
    memcpy(big, selhead_pte_start, ssz);
    const size_t sat = find_esp32_blob(big, ssz);
    if (sat == SIZE_MAX) {
      heap_caps_free(big);
      return false;
    }
    uint32_t ext = 0;
    memcpy(&ext, big + sat + 12u, sizeof(ext));
    const size_t record = sat + ext + 4u;
    if (ext == 0 || record + 16u > ssz) {
      heap_caps_free(big);
      return false;
    }
    uint16_t type = 0;
    uint32_t len = 0;
    memcpy(&type, big + record + 2u, sizeof(type));
    memcpy(&len, big + record + 4u, sizeof(len));
    if ((type & 0x7fffu) != 7u || len != 8u) {
      heap_caps_free(big);
      return false;
    }
    if (mutation == 1) {
      const uint32_t zero = 0;
      // SelectedRows.count is the second uint32 in the 8-byte payload.
      memcpy(big + record + 8u + 4u, &zero, sizeof(zero));
    } else if (mutation == 2) {
      uint32_t nops_hdr = 0;
      memcpy(&nops_hdr, big + sat + 8u, sizeof(nops_hdr));
      const uint32_t table = 16u + 64u * nops_hdr;
      const uint32_t record_bytes = 16u;
      if (table + 4u + 2u * record_bytes > ext) {
        heap_caps_free(big);
        return false;
      }
      uint8_t saved[record_bytes];
      memcpy(saved, big + record, record_bytes);
      const uint32_t two = 2;
      memcpy(big + sat + table, &two, sizeof(two));
      memcpy(big + sat + table + 4u, saved, record_bytes);
      memcpy(big + sat + table + 4u + record_bytes, saved, record_bytes);
      memcpy(big + sat + 12u, &table, sizeof(table));
    }
    const bool ok = load_validation_pte(what, big, ssz);
    heap_caps_free(big);
    return ok;
  };
  const bool selected_clean =
      try_selected_set("selected rows unmodified control", 0);
  const bool selected_zero =
      try_selected_set("selected rows zero count", 1);
  const bool selected_duplicate =
      try_selected_set("selected rows duplicate record", 2);

  // The cached-K q@kT on its own artifact. Its third input is an ARG index
  // carried in scale_off (word 8): a chain-slot sentinel there, or a stored
  // payload offset (wt_off, word 7) on an op that stores nothing, describes
  // some other op and must be refused; and an opcode one past the table (what
  // a runtime without this op sees) is refused, not guessed at.
  const bool kc_clean = try_desc_word(
      "kcache bmm unmodified control", attnkc8q_pte_start, attnkc8q_pte_end,
      9u, 0u, false, 0u);
  const bool kc_scalearg = try_desc_word(
      "kcache bmm scales in a chain slot", attnkc8q_pte_start,
      attnkc8q_pte_end, 9u, 8u, true, 0xFFFFFFFEu);
  const bool kc_stored = try_desc_word(
      "kcache bmm stored weight offset", attnkc8q_pte_start,
      attnkc8q_pte_end, 9u, 7u, true, 64u);
  const bool kc_unknown = try_desc_word(
      "kcache bmm as an unknown opcode", attnkc8q_pte_start,
      attnkc8q_pte_end, 9u, 0u, true, (uint32_t)esp32_num_opcodes());
  printf("  kcguard: control=%s scalearg=%s storedoff=%s unknownop=%s\n",
         kc_clean ? "accepted" : "rejected",
         kc_scalearg ? "accepted" : "rejected",
         kc_stored ? "accepted" : "rejected",
         kc_unknown ? "accepted" : "rejected");

  printf("  blobguard: control=%s kpad=%s nops=%s bmmcontrol=%s "
         "bmmbatch=%s convcontrol=%s convstride=%s convduplicate=%s "
         "groupcontrol=%s groupduplicate=%s i16rowcontrol=%s i16row=%s "
         "selectedcontrol=%s selectedzero=%s selectedduplicate=%s "
         "scalealign=%s biasalign=%s groupcountcontrol=%s groupcount=%s "
         "alias=%s dotlen=%s\n",
         clean ? "accepted" : "rejected", kpad ? "accepted" : "rejected",
         nops ? "accepted" : "rejected",
         bmm_clean ? "accepted" : "rejected",
         bmm_batch ? "accepted" : "rejected",
         conv_clean ? "accepted" : "rejected",
         conv_stride ? "accepted" : "rejected",
         conv_duplicate ? "accepted" : "rejected",
         group_clean ? "accepted" : "rejected",
         group_duplicate ? "accepted" : "rejected",
         i16_row_clean ? "accepted" : "rejected",
         i16_row ? "accepted" : "rejected",
         selected_clean ? "accepted" : "rejected",
         selected_zero ? "accepted" : "rejected",
         selected_duplicate ? "accepted" : "rejected",
         scale_align ? "accepted" : "rejected",
         bias_align ? "accepted" : "rejected",
         group_count_clean ? "accepted" : "rejected",
         group_count ? "accepted" : "rejected",
         alias ? "accepted" : "rejected",
         dot_len ? "accepted" : "rejected");
  printf("  extguard: optional=%s required=%s retired=%s oob=%s\n",
         ext_opt ? "skipped" : "REJECTED", ext_req ? "ACCEPTED" : "refused",
         ext_retired ? "ACCEPTED" : "refused",
         ext_oob ? "ACCEPTED" : "refused");
  heap_caps_free(copy);
}

// The affine parity rows: small programs packed into a flash
// partition -- the app image has no room for them -- each a parity row
// run_parity holds to its mirror's output bits. Image layout: FIXTURES_MAGIC
// in test/fixtures/asym.py. Entry flags: 1 load the image's .ptd partition,
// 2 a row of the min_latency pass (else the plain one), 4 hash every output,
// 8 also print every output value (a row held to at most one code), 16 time
// 16 more executions (a measurement, never a parity row), 32 print the
// largest free blocks before the row's planned buffers are allocated, 64 a
// row that needs PSRAM (its planned buffers exceed the internal heap):
// skipped, by name, where the firmware has none (QEMU's).
static void run_asym_image(uint8_t subtype, uint8_t ptd_subtype,
                           const char* what, bool split_pass) {
  auto image = esp32s3::EspPartitionDataLoader::from_subtype(subtype);
  if (!image.ok()) {
    printf("\n--- %s SKIPPED: partition 0x%x: %d\n", what, (unsigned)subtype,
           (int)image.error());
    return;
  }
  const uint8_t* base = static_cast<const uint8_t*>(image->mapped_base());
  const size_t size = *image->size();
  uint32_t version = 0, count = 0;
  if (size < 12 || memcmp(base, "ASYF", 4) != 0) {
    printf("\n--- %s SKIPPED: no image in partition 0x%x\n", what,
           (unsigned)subtype);
    return;
  }
  memcpy(&version, base + 4, 4);
  memcpy(&count, base + 8, 4);
  if (version != 1 || 12ull + 48ull * count > size) {
    printf("\n--- %s SKIPPED: image version %u, %u rows\n", what,
           (unsigned)version, (unsigned)count);
    return;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* entry = base + 12 + (size_t)i * 48;
    char name[25];
    memcpy(name, entry, 24);
    name[24] = 0;
    uint32_t off = 0, len = 0, modulus = 0, flags = 0;
    memcpy(&off, entry + 24, 4);
    memcpy(&len, entry + 28, 4);
    memcpy(&modulus, entry + 32, 4);
    memcpy(&flags, entry + 36, 4);
    if (((flags & 2u) != 0) != split_pass) {
      continue;
    }
    char label[112];
    snprintf(label, sizeof(label), "%s.pte  [blob v24 %s]", name, what);
    if ((uint64_t)off + len > size || off % 16 != 0) {
      printf("\n--- %s SKIPPED: past the image or misaligned\n", label);
      continue;
    }
#ifndef CONFIG_SPIRAM
    if (flags & 64u) {
      printf("\n--- %s SKIPPED: needs PSRAM, and this firmware has none\n",
             label);
      continue;
    }
#endif
    g_long_modulus_override = modulus;
    g_report_largest_block = (flags & 32u) != 0;
    const uint32_t forks0 = esp32_backend_split_jobs();
    const bool all_outputs = (flags & 4u) != 0;
    const bool dump_values = (flags & 8u) != 0;
    const int timed_runs = (flags & 16u) != 0 ? 16 : 0;
    if (flags & 1u) {
      auto ptd = esp32s3::EspPartitionDataLoader::from_subtype(ptd_subtype);
      if (!ptd.ok()) {
        printf("\n--- %s SKIPPED: .ptd partition 0x%x: %d\n", label,
               (unsigned)ptd_subtype, (int)ptd.error());
        g_long_modulus_override = 0;
        g_report_largest_block = false;
        continue;
      }
      run_model(label, base + off, len, &ptd.get(), nullptr,
                all_outputs, dump_values, timed_runs);
    } else {
      run_model(label, base + off, len, nullptr, nullptr,
                all_outputs, dump_values, timed_runs);
    }
    g_long_modulus_override = 0;
    g_report_largest_block = false;
    printf("  asym forks: %u\n",
           (unsigned)(esp32_backend_split_jobs() - forks0));
  }
}

// The weight-affine rows (fixtures/asym.py: asymprog, its .ptd in asymw), the
// dynamic-activation rows (fixtures/asym_act.py: asymact, .ptd in asymactw),
// the static int8 graphs (fixtures/asym_static.py: asymstat, no .ptd), the
// int16 activations (fixtures/asym_a16.py: asyma16, no .ptd), the QAT round
// trips (fixtures/asym_qat.py: asymqat, no .ptd), the static graphs' second
// image (fixtures/asym_static.py: asymops, the ops between delegates), then
// Stage B's codes (fixtures/asym_stage_b.py: asymstgb, no .ptd; each row's Stage A
// twin first), the int16 table op (fixtures/lut16.py: asymlut16, no .ptd),
// the fused quantised norms (fixtures/qnorm.py: asymnorm, no .ptd), then the
// CNN and RNN models (fixtures/cnn.py: cnnrnn, their named data in cnnrnnw).
static void run_asym_fixtures(bool split_pass) {
  run_asym_image(0x49, 0x4A, "weight zero points / float offsets", split_pass);
  run_asym_image(0x4B, 0x4C, "dynamic activations", split_pass);
  run_asym_image(0x4D, 0x4D, "static graphs", split_pass);
  run_asym_image(0x53, 0x53, "int16 activations", split_pass);
  run_asym_image(0x56, 0x56, "QAT round trips", split_pass);
  run_asym_image(0x5C, 0x5C, "ops between delegates", split_pass);
  run_asym_image(0x5D, 0x5D, "Stage B codes", split_pass);
  run_asym_image(0x5E, 0x5E, "int16 tables", split_pass);
  run_asym_image(0x5F, 0x5F, "fused norms", split_pass);
  run_asym_image(0x60, 0x61, "CNNs and RNNs", split_pass);
  printf("  esp32 affine: %u weight-affine op(s) executed\n",
         (unsigned)esp32_backend_affine_ops());
}

// ExecuTorch's own LLM recipes (fixtures/stock_llm.py): float input `index` at
// `step` -- zeros at step 2, 0.75 at step 3, exact A4 rounding ties at step 4
// (every 16 values hold lo and hi 255/64 apart, so the scale is 2^-6, and the
// rest are half-units: x / s + z = m + z + 1/2, z = 0 for index 2's peers and
// -1 for index 2), else a small integer times 2^-4: exact under any
// contraction and a different range at every step.
static float stock_step_float(int64_t j, int step, int index) {
  if (step == 2) return 0.0f;
  if (step == 3) return 0.75f;
  if (step == 4) {
    const int lo = index == 2 ? -127 : -128;
    if (j % 16 == 0) return (float)lo * 0.015625f;
    if (j % 16 == 1) return (float)(lo + 255) * 0.015625f;
    return ((float)((37 * j) % 254) - 127.0f + 0.5f) * 0.015625f;
  }
  const int64_t m = 17 + 6 * (step % 5);
  const int64_t v = ((37 * j + 101 * (int64_t)step + 13 * index) % m) -
                    (3 + 2 * (step % 7));
  return (float)v * 0.0625f;
}

// An int64 input that is not input_pos: (7j + 3 + 37 step) mod vocab.
static int64_t stock_step_long(int64_t j, int step, uint32_t vocab) {
  return vocab ? (7 * j + 3 + 37 * (int64_t)step) % vocab : 0;
}

// One stock-recipe row: the program loaded once, its planned buffers (the
// K/V caches among them) zeroed once, then `steps` executions with the step
// inputs above -- with `position`, the last int64 input is input_pos, the
// step index. Every step prints its outputs' checksum (finite values), the
// argmax of output 0 and the FNV-1a of every output's bytes, in order.
static void run_stock_sequence(const char* name, const uint8_t* pte,
                               size_t pte_size, int steps, uint32_t vocab,
                               bool position) {
  printf("--- %s.pte  [stock LLM recipe, %d step(s)] (%u B) ---\n", name,
         steps, (unsigned)pte_size);
  BufferDataLoader loader(pte, pte_size);
  auto program = Program::load(&loader);
  if (!program.ok()) {
    printf("  Program::load FAILED: %d\n", (int)program.error());
    return;
  }
  const char* mname = program->get_method_name(0).get();
  auto meta = program->method_meta(mname);
  if (!meta.ok()) {
    printf("  method_meta FAILED: %d\n", (int)meta.error());
    return;
  }
  Span<uint8_t> spans[4];
  const size_t n_planned = meta->num_memory_planned_buffers();
  if (n_planned > 4) {
    printf("  too many planned buffers: %u\n", (unsigned)n_planned);
    return;
  }
  for (size_t i = 0; i < n_planned; ++i) {
    const size_t sz = (size_t)meta->memory_planned_buffer_size(i).get();
    uint8_t* p = (uint8_t*)heap_caps_malloc(sz, MALLOC_CAP_8BIT);
    if (p == nullptr) {
      printf("  planned buffer %u alloc FAILED (%u B)\n", (unsigned)i,
             (unsigned)sz);
      for (size_t k = 0; k < i; ++k) heap_caps_free(spans[k].data());
      return;
    }
    // ExecuTorch stores no initial values for mutable buffers: the caches
    // start at zero, as the exporter's references assume.
    memset(p, 0, sz);
    spans[i] = Span<uint8_t>(p, sz);
  }
  const size_t arena_sz = 96 * 1024;
  uint8_t* arena = (uint8_t*)heap_caps_malloc(arena_sz, MALLOC_CAP_8BIT);
  if (arena == nullptr) {
    printf("  could not obtain a method arena\n");
    for (size_t k = 0; k < n_planned; ++k) heap_caps_free(spans[k].data());
    return;
  }
  {
    TrackingAllocator method_allocator(arena_sz, arena);
    TieredTempAllocator temp_allocator(sizeof(temp_arena), temp_arena, 0,
                                       nullptr);
    HierarchicalAllocator planned_allocator({spans, n_planned});
    MemoryManager memory_manager(&method_allocator, &planned_allocator,
                                 &temp_allocator);
    auto method = program->load_method(mname, &memory_manager);
    if (!method.ok()) {
      printf("  load_method FAILED: %d\n", (int)method.error());
    } else {
      printf("  method '%s': %u in, %u out, arena peak %u B\n", mname,
             (unsigned)method->inputs_size(), (unsigned)method->outputs_size(),
             (unsigned)method_allocator.peak());
      const size_t n_in = method->inputs_size();
      size_t last_long = n_in;
      EValue in[4];
      if (n_in > 4 || method->get_inputs(in, n_in) != Error::Ok) {
        printf("  get_inputs FAILED\n");
        steps = 0;
      }
      for (size_t i = 0; i < n_in && steps > 0; ++i) {
        if (in[i].toTensor().scalar_type() ==
            executorch::aten::ScalarType::Long) {
          last_long = i;
        }
      }
      for (int step = 0; step < steps; ++step) {
        method->get_inputs(in, n_in);
        int index = 0;
        for (size_t i = 0; i < n_in; ++i) {
          Tensor t = in[i].toTensor();
          if (t.scalar_type() == executorch::aten::ScalarType::Float) {
            float* p = t.mutable_data_ptr<float>();
            for (int64_t j = 0; j < t.numel(); ++j) {
              p[j] = stock_step_float(j, step, index);
            }
            ++index;
          } else if (t.scalar_type() == executorch::aten::ScalarType::Long) {
            int64_t* p = t.mutable_data_ptr<int64_t>();
            const bool pos = position && i == last_long;
            for (int64_t j = 0; j < t.numel(); ++j) {
              p[j] = pos ? step : stock_step_long(j, step, vocab);
            }
          }
        }
        const Error e = method->execute();
        if (e != Error::Ok) {
          printf("  %s step %d execute FAILED: %d\n", name, step, (int)e);
          break;
        }
        EValue outs[8];
        const size_t n_out = method->outputs_size();
        if (n_out > 8 || method->get_outputs(outs, n_out) != Error::Ok) {
          printf("  %s step %d get_outputs FAILED\n", name, step);
          break;
        }
        double sum = 0;
        uint32_t bits = 2166136261u;
        size_t at = 0;
        int top1 = -1;
        float best = 0.0f;
        for (size_t k = 0; k < n_out; ++k) {
          const Tensor t = outs[k].toTensor();
          if (t.scalar_type() != executorch::aten::ScalarType::Float) {
            continue;
          }
          const float* p = t.const_data_ptr<float>();
          for (int64_t i = 0; i < t.numel(); ++i, ++at) {
            if (std::isfinite(p[i])) sum += (double)p[i] * (double)(at + 1);
            if (k == 0 && (top1 < 0 || p[i] > best)) {
              best = p[i];
              top1 = (int)i;
            }
          }
          bits = output_bits(p, (size_t)t.nbytes(), bits);
        }
        printf("  %s step %d: checksum %.6f top1 %d bits %08lx\n", name, step,
               sum, top1, (unsigned long)bits);
      }
    }
  }
  heap_caps_free(arena);
  for (size_t k = 0; k < n_planned; ++k) heap_caps_free(spans[k].data());
}

// The stock-recipe rows' image (fixtures/stock_llm.py: partition stockllm,
// subtype 0x50): "STKF", u32 version, u32 count, 48-byte entries.
static void run_stock_llm_fixtures() {
  auto image = esp32s3::EspPartitionDataLoader::from_subtype(0x50);
  if (!image.ok()) {
    printf("\n--- stock LLM recipes SKIPPED: partition 0x50: %d\n",
           (int)image.error());
    return;
  }
  const uint8_t* base = static_cast<const uint8_t*>(image->mapped_base());
  const size_t size = *image->size();
  uint32_t version = 0, count = 0;
  if (size < 12 || memcmp(base, "STKF", 4) != 0) {
    printf("\n--- stock LLM recipes SKIPPED: no image in partition 0x50\n");
    return;
  }
  memcpy(&version, base + 4, 4);
  memcpy(&count, base + 8, 4);
  if (version != 1 || 12ull + 48ull * count > size) {
    printf("\n--- stock LLM recipes SKIPPED: image version %u, %u rows\n",
           (unsigned)version, (unsigned)count);
    return;
  }
  const uint32_t ops0 = esp32_asym::kcache_affine_ops();
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* entry = base + 12 + (size_t)i * 48;
    char name[25];
    memcpy(name, entry, 24);
    name[24] = 0;
    uint32_t off = 0, len = 0, steps = 0, vocab = 0, flags = 0;
    memcpy(&off, entry + 24, 4);
    memcpy(&len, entry + 28, 4);
    memcpy(&steps, entry + 32, 4);
    memcpy(&vocab, entry + 36, 4);
    memcpy(&flags, entry + 40, 4);
    if ((uint64_t)off + len > size || off % 16 != 0) {
      printf("\n--- %s.pte SKIPPED: past the image or misaligned\n", name);
      continue;
    }
    run_stock_sequence(name, base + off, len, (int)steps, vocab,
                       (flags & 1u) != 0);
  }
  // The A4 writer's cost (float64 in software on the S3), once per K and V row
  // of each token and head: rows of 32 (a Glimmer-sized head), timed with
  // esp_timer. Plain QEMU's clock is the host's; under `-icount shift=0`
  // every instruction is 1 ns of virtual time, so the line counts them.
  {
    float x[32];
    int8_t codes[32];
    float scale;
    int8_t zp;
    for (int j = 0; j < 32; ++j) x[j] = stock_step_float(j, 5, 1);
    const int rows = 256;
    const int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < rows; ++r) {
      x[r % 32] += 0.0625f;  // a different row each time
      espressif::kv_affine::quantize_row(x, 32, codes, &scale, &zp);
    }
    const int64_t t1 = esp_timer_get_time();
    printf("  kv_affine writer: %d rows of 32 in %lld us (last zp %d)\n", rows,
           (long long)(t1 - t0), (int)zp);
  }
  const espressif::kv_affine::Stats kv = espressif::kv_affine::stats();
  printf("  stock llm: %u affine K-cache op(s), %llu key row(s); writer rows "
         "%llu, read rows %llu, cache writes %llu, refusals %llu\n",
         (unsigned)(esp32_asym::kcache_affine_ops() - ops0),
         (unsigned long long)esp32_asym::kcache_affine_rows(),
         (unsigned long long)kv.quantized_rows,
         (unsigned long long)kv.dequantized_rows,
         (unsigned long long)kv.cache_writes, (unsigned long long)kv.refusals);
}

// The asymmetric-quantisation rules (esp32_asym.cpp), one synthetic blob per
// rule. They cover far more than any exporter emits, so asym_blob_cases.py
// builds them by hand: valid blobs -- those the runtime computes (rule
// "accepted") must load, the rest be refused only as "unimplemented" -- and
// copies with exactly one field wrong, each of which must be refused for ITS
// rule, read back from esp32_asym::last_refusal(), because a blob refused by
// some other rule would pass a test that only asked whether it was refused.
// Each case is written over the carrier program's delegate payload (the rest
// zeroed; the payload's length is fixed by the program around it). The
// unmodified carrier is the control: a symmetric blob must still load.
static void run_asym_refusal_check() {
  printf("\n--- blob v24 refusals (synthetic blobs, one per rule) ---\n");
  const uint8_t* cases = asymcases_blob_start;
  const size_t cases_size = (size_t)(asymcases_blob_end - asymcases_blob_start);
  const size_t carrier_size =
      (size_t)(asymcarrier_pte_end - asymcarrier_pte_start);
  uint32_t version = 0, offset = 0, length = 0, count = 0;
  if (cases_size < 20 || memcmp(cases, "ASYC", 4) != 0) {
    printf("  asymguard: no case file\n");
    return;
  }
  memcpy(&version, cases + 4, 4);
  memcpy(&offset, cases + 8, 4);
  memcpy(&length, cases + 12, 4);
  memcpy(&count, cases + 16, 4);
  if (version != 1 || (uint64_t)offset + length > carrier_size) {
    printf("  asymguard: the case file does not fit the carrier\n");
    return;
  }
  uint8_t* copy =
      (uint8_t*)heap_caps_aligned_alloc(16, carrier_size, MALLOC_CAP_8BIT);
  if (copy == nullptr) {
    printf("  asymguard: alloc FAILED\n");
    return;
  }
  memcpy(copy, asymcarrier_pte_start, carrier_size);
  esp32_asym::clear_last_refusal();
  const bool control =
      load_validation_pte("v24 carrier unmodified control", copy, carrier_size) &&
      esp32_asym::last_refusal() == 0;
  uint32_t wrong = 0, accepted = 0, seen = 0, computed = 0;
  size_t at = 20;
  char name[96], rule[40], label[112];
  for (uint32_t c = 0; c < count; ++c) {
    uint16_t name_len = 0, rule_len = 0;
    uint32_t blob_len = 0;
    if (at + 8 > cases_size) break;
    memcpy(&name_len, cases + at, 2);
    memcpy(&rule_len, cases + at + 2, 2);
    memcpy(&blob_len, cases + at + 4, 4);
    const size_t record = 8u + name_len + rule_len + blob_len;
    if (at + record > cases_size || blob_len > length ||
        name_len >= sizeof(name) || rule_len >= sizeof(rule)) {
      break;
    }
    memcpy(name, cases + at + 8, name_len);
    name[name_len] = 0;
    memcpy(rule, cases + at + 8 + name_len, rule_len);
    rule[rule_len] = 0;
    memcpy(copy, asymcarrier_pte_start, carrier_size);
    memcpy(copy + offset, cases + at + 8 + name_len + rule_len, blob_len);
    memset(copy + offset + blob_len, 0, length - blob_len);
    esp32_asym::clear_last_refusal();
    snprintf(label, sizeof(label), "v24 %s", name);
    const bool loaded = load_validation_pte(label, copy, carrier_size);
    const uint32_t refusal = esp32_asym::last_refusal();
    const char* got = esp32_asym::rule_name(refusal);
    // "accepted": a valid blob this runtime computes, which must load with no
    // refusal; every other case must be refused for its own rule.
    const bool expect_load = strcmp(rule, "accepted") == 0;
    const bool right = expect_load ? (loaded && refusal == 0)
                                   : (!loaded && strcmp(got, rule) == 0);
    accepted += loaded ? 1u : 0u;
    computed += expect_load ? 1u : 0u;
    wrong += right ? 0u : 1u;
    ++seen;
    printf("  asymcase %s: expected [%s] refused [%s]\n",
           right ? "ok" : "WRONG", rule, loaded ? "none: ACCEPTED" : got);
    at += (record + 3u) & ~(size_t)3u;
  }
  printf("  asymguard: cases=%u read=%u wrong=%u accepted=%u computed=%u "
         "control=%s\n",
         (unsigned)count, (unsigned)seen, (unsigned)(wrong + (count - seen)),
         (unsigned)accepted, (unsigned)computed,
         control ? "accepted" : "rejected");
  heap_caps_free(copy);
}

// Two delegate handles, alive at the same time, configured differently.
//
// This is the reentrancy claim, and it is a claim about STATE OWNERSHIP rather
// than about arithmetic: min-latency is per-model configuration carried in a
// compile spec, so a latency-critical model and a background one must be able to
// coexist in one process without either seeing the other's posture. File-scope
// globals could not express that -- a second load would silently redefine the
// first.
//
// Both programs are the same MLP with the same weights, so both must produce the
// SAME checksum as the single-core rows: the split moves whole columns between
// cores and never reorders a column's accumulation. The evidence that they
// differ at all is the fork counter, read around each execute -- the min-latency
// handle forks, the plain handle does not, with no override in play.
static void run_reentrancy_check() {
  printf("\n--- two live handles, different compile specs ---\n");

  BufferDataLoader plain_loader(mlpdel_pte_start,
                                (size_t)(mlpdel_pte_end - mlpdel_pte_start));
  BufferDataLoader ml_loader(mlpml_pte_start,
                             (size_t)(mlpml_pte_end - mlpml_pte_start));
  auto plain_prog = Program::load(&plain_loader);
  auto ml_prog = Program::load(&ml_loader);
  if (!plain_prog.ok() || !ml_prog.ok()) {
    printf("  Program::load FAILED\n");
    return;
  }
  const char* plain_name = *plain_prog->get_method_name(0);
  const char* ml_name = *ml_prog->get_method_name(0);

  // Separate arenas: the point is that BOTH methods are live simultaneously, so
  // they cannot share the static one. The second comes off the heap and is
  // released at the end, like run_cached_sequence's.
  const size_t kArena = 16 * 1024;
  uint8_t* arena2 = (uint8_t*)heap_caps_aligned_alloc(16, kArena, MALLOC_CAP_8BIT);
  if (arena2 == nullptr) {
    printf("  second arena alloc FAILED\n");
    return;
  }
  // Temp storage is execution-local. These methods are deliberately executed
  // concurrently below, so sharing temp_arena would race even though the two
  // MemoryAllocator cursor objects are distinct. Give the second live method
  // physically distinct backing as well as a distinct allocator.
  uint8_t* temp2 = (uint8_t*)heap_caps_aligned_alloc(
      16, sizeof(temp_arena), MALLOC_CAP_8BIT);
  if (temp2 == nullptr) {
    printf("  second temp arena alloc FAILED\n");
    heap_caps_free(arena2);
    return;
  }
  TrackingAllocator plain_alloc(sizeof(method_arena), method_arena);
  TrackingAllocator ml_alloc(kArena, arena2);

  Span<uint8_t> plain_spans[2];
  Span<uint8_t> ml_spans[2];
  uint8_t* plain_planned[2] = {nullptr, nullptr};
  uint8_t* ml_planned[2] = {nullptr, nullptr};
  auto plain_meta = plain_prog->method_meta(plain_name);
  auto ml_meta = ml_prog->method_meta(ml_name);
  if (!plain_meta.ok() || !ml_meta.ok()) {
    printf("  method_meta FAILED\n");
    heap_caps_free(temp2);
    heap_caps_free(arena2);
    return;
  }
  const size_t n_plain = plain_meta->num_memory_planned_buffers();
  const size_t n_ml = ml_meta->num_memory_planned_buffers();
  // Each HierarchicalAllocator below is handed its whole count of spans.
  if (n_plain > 2 || n_ml > 2) {
    printf("  too many planned buffers: %u, %u\n", (unsigned)n_plain,
           (unsigned)n_ml);
    heap_caps_free(temp2);
    heap_caps_free(arena2);
    return;
  }
  bool alloc_ok = true;
  for (size_t i = 0; i < n_plain && i < 2; i++) {
    size_t want = (size_t)*plain_meta->memory_planned_buffer_size(i);
    plain_planned[i] = (uint8_t*)heap_caps_aligned_alloc(
        16, (want + 15u) & ~15u, MALLOC_CAP_8BIT);
    alloc_ok = alloc_ok && plain_planned[i] != nullptr;
    plain_spans[i] = Span<uint8_t>(plain_planned[i], want);
  }
  for (size_t i = 0; i < n_ml && i < 2; i++) {
    size_t want = (size_t)*ml_meta->memory_planned_buffer_size(i);
    ml_planned[i] = (uint8_t*)heap_caps_aligned_alloc(
        16, (want + 15u) & ~15u, MALLOC_CAP_8BIT);
    alloc_ok = alloc_ok && ml_planned[i] != nullptr;
    ml_spans[i] = Span<uint8_t>(ml_planned[i], want);
  }
  if (!alloc_ok) {
    printf("  planned buffer alloc FAILED\n");
  } else {
    HierarchicalAllocator plain_planner({plain_spans, n_plain});
    HierarchicalAllocator ml_planner({ml_spans, n_ml});
    TieredTempAllocator plain_temp(sizeof(temp_arena), temp_arena);
    TieredTempAllocator ml_temp(sizeof(temp_arena), temp2);
    void* plain_probe = plain_temp.allocate(16, 16);
    void* ml_probe = ml_temp.allocate(16, 16);
    if (plain_temp.size() == 0 || ml_temp.size() == 0 ||
        plain_probe == nullptr || ml_probe == nullptr || plain_probe == ml_probe) {
      printf("  temp allocator isolation FAILED\n");
      for (size_t i = 0; i < n_plain && i < 2; i++) {
        heap_caps_free(plain_planned[i]);
      }
      for (size_t i = 0; i < n_ml && i < 2; i++) {
        heap_caps_free(ml_planned[i]);
      }
      heap_caps_free(temp2);
      heap_caps_free(arena2);
      return;
    }
    plain_temp.reset();
    ml_temp.reset();
    MemoryManager plain_mm(&plain_alloc, &plain_planner, &plain_temp);
    MemoryManager ml_mm(&ml_alloc, &ml_planner, &ml_temp);

    auto plain_method =
        plain_prog->load_method(plain_name, &plain_mm, nullptr, nullptr);
    auto ml_method = ml_prog->load_method(ml_name, &ml_mm, nullptr, nullptr);
    if (!plain_method.ok() || !ml_method.ok()) {
      printf("  load_method FAILED (%d, %d)\n", (int)plain_method.error(),
             (int)ml_method.error());
    } else {
      printf("  both methods live: arenas %u B + %u B\n",
             (unsigned)plain_alloc.peak(), (unsigned)ml_alloc.peak());

      // Interleaved deliberately: plain, min-latency, plain again. If the
      // handles shared state, the middle run would leave the split enabled and
      // the third would fork too.
      const uint32_t f0 = esp32_backend_split_jobs();
      fill_inputs(*plain_method);
      Error e1 = plain_method->execute();
      const uint32_t f1 = esp32_backend_split_jobs();
      fill_inputs(*ml_method);
      // The task notification belongs to the application. One already
      // pending when a split execute starts must neither end the backend's
      // join early nor be consumed by it: post a distinctive value, run the
      // forking handle, and read the value back intact.
      constexpr uint32_t kAppNotify = 0x5EED5EEDu;
      xTaskNotify(xTaskGetCurrentTaskHandle(), kAppNotify,
                  eSetValueWithOverwrite);
      Error e2 = ml_method->execute();
      uint32_t app_notify = 0;
      const bool app_pending =
          xTaskNotifyWait(0, UINT32_MAX, &app_notify, 0) == pdTRUE;
      const uint32_t f2 = esp32_backend_split_jobs();
      printf("  app notify: preserved %s (0x%08x)\n",
             (app_pending && app_notify == kAppNotify) ? "yes" : "NO",
             (unsigned)app_notify);
      fill_inputs(*plain_method);
      Error e3 = plain_method->execute();
      const uint32_t f3 = esp32_backend_split_jobs();

      printf("  execute: %d %d %d (0 == Ok)\n", (int)e1, (int)e2, (int)e3);
      printf("  forks: background handle %u, min-latency handle %u, "
             "background again %u\n",
             (unsigned)(f1 - f0), (unsigned)(f2 - f1), (unsigned)(f3 - f2));

      EValue out[1];
      double sums[2] = {0.0, 0.0};
      uint32_t bits[2] = {0u, 0u};
      for (int which = 0; which < 2; ++which) {
        auto& m = which == 0 ? plain_method : ml_method;
        if (m->get_outputs(out, 1) == Error::Ok) {
          Tensor t = out[0].toTensor();
          const float* p = t.const_data_ptr<float>();
          for (int i = 0; i < t.numel(); i++) {
            sums[which] += (double)p[i] * (i + 1);
          }
          bits[which] = output_bits(p, (size_t)t.nbytes());
        }
      }
      // Reported as two ordinary parity rows (the harness keys on
      // "--- <name>.pte"), because the claim is exactly that both handles still
      // compute the reference value: same weights, same inputs, and a split that
      // moves columns without reordering any column's accumulation.
      printf("--- mlp_bg_handle.pte  [handle A, background posture] ---\n");
      printf("  checksum: %.6f\n", sums[0]);
      printf("  output bits: %08lx\n", (unsigned long)bits[0]);
      printf("--- mlp_ml_handle.pte  [handle B, min-latency posture] ---\n");
      printf("  checksum: %.6f\n", sums[1]);
      printf("  output bits: %08lx\n", (unsigned long)bits[1]);

      // The split worker is pinned to core 1. A min-latency caller already
      // running there must keep its ops whole: forking would serialise both
      // halves on one core behind two context switches.
      struct SameCore {
        ::executorch::runtime::Method* method;
        volatile uint32_t forks;
        volatile double sum;
        volatile bool done;
        volatile uint32_t bits;
      };
      static SameCore same_core;
      same_core = {&ml_method.get(), 0, 0.0, false, 0u};
      auto same_core_main = [](void* arg) {
        SameCore* s = (SameCore*)arg;
        fill_inputs(*s->method);
        const uint32_t before = esp32_backend_split_jobs();
        const bool ran = s->method->execute() == Error::Ok;
        s->forks = ran ? esp32_backend_split_jobs() - before : UINT32_MAX;
        EValue o[1];
        double acc = 0.0;
        uint32_t bits = 0u;
        if (ran && s->method->get_outputs(o, 1) == Error::Ok) {
          Tensor t = o[0].toTensor();
          const float* p = t.const_data_ptr<float>();
          for (int i = 0; i < t.numel(); i++) acc += (double)p[i] * (i + 1);
          bits = output_bits(p, (size_t)t.nbytes());
        }
        s->sum = acc;
        s->bits = bits;
        s->done = true;
        vTaskDelete(nullptr);
      };
      xTaskCreatePinnedToCore(same_core_main, "et_core1", 4096, &same_core,
                              uxTaskPriorityGet(nullptr), nullptr, 1);
      for (int spin = 0; spin < 500 && !same_core.done; ++spin) {
        vTaskDelay(pdMS_TO_TICKS(10));
      }
      printf("--- mlp_ml_core1.pte  [handle B, called on the worker's core] ---\n");
      printf("  checksum: %.6f\n", (double)same_core.sum);
      printf("  output bits: %08lx\n", (unsigned long)same_core.bits);
      printf("  worker-core caller: forks %u, finished %s\n",
             (unsigned)same_core.forks, same_core.done ? "yes" : "NO");

      // The worker runs at the priority of whichever caller holds its lease.
      // Set exactly, not only raised: one high-priority caller must not leave
      // every later fork's half running above the application on core 1.
      struct Elevated {
        ::executorch::runtime::Method* method;
        volatile bool done;
      };
      static Elevated elevated;
      elevated = {&ml_method.get(), false};
      const UBaseType_t base_priority = uxTaskPriorityGet(nullptr);
      auto elevated_main = [](void* arg) {
        Elevated* e = (Elevated*)arg;
        fill_inputs(*e->method);
        (void)e->method->execute();
        e->done = true;
        vTaskDelete(nullptr);
      };
      xTaskCreatePinnedToCore(elevated_main, "et_high", 4096, &elevated,
                              base_priority + 2, nullptr, 0);
      for (int spin = 0; spin < 500 && !elevated.done; ++spin) {
        vTaskDelay(pdMS_TO_TICKS(10));
      }
      TaskHandle_t split_worker = xTaskGetHandle("et_split");
      const unsigned after_high =
          split_worker ? (unsigned)uxTaskPriorityGet(split_worker) : 0u;
      fill_inputs(*ml_method);
      (void)ml_method->execute();
      const unsigned after_base =
          split_worker ? (unsigned)uxTaskPriorityGet(split_worker) : 0u;
      printf("  worker priority: base %u, after a priority-%u caller %u, "
             "after a base caller %u\n", (unsigned)base_priority,
             (unsigned)(base_priority + 2), after_high, after_base);

      // CONCURRENT execution -- the case SplitLease exists for and which the
      // sequential runs above never reach. Both handles execute at the same
      // time on the two cores; whichever loses the try-take runs single-core.
      // That degradation must be invisible in the VALUES (a split moves whole
      // columns, never a column's accumulation order), so the assertion is that
      // both checksums are unchanged from their sequential counterparts.
      struct Racer {
        ::executorch::runtime::Method* method;
        volatile double sum;
        volatile int ok_runs;
        volatile bool done;
        volatile uint32_t bits;
      };
      static constexpr int kRaceRuns = 8;
      static Racer racers[2];
      racers[0] = {&plain_method.get(), 0.0, 0, false, 0u};
      racers[1] = {&ml_method.get(), 0.0, 0, false, 0u};
      auto racer_main = [](void* arg) {
        Racer* r = (Racer*)arg;
        int ok_runs = 0;
        for (int rep = 0; rep < kRaceRuns; ++rep) {
          // Graph inputs may share planned storage with values that are dead at
          // the end of an execution. Re-establish them on every invocation;
          // the current combined MLP plan happens to preserve its input, but
          // relying on that would make this concurrency test plan-dependent.
          fill_inputs(*r->method);
          if (r->method->execute() != Error::Ok) {
            break;
          }
          ++ok_runs;
        }
        // Checksummed only when every run succeeded: a failed execute can
        // leave the warm-up's outputs in place, which would match the
        // reference and hide the failure.
        EValue o[1];
        double acc = 0.0;
        uint32_t bits = 0u;
        if (ok_runs == kRaceRuns && r->method->get_outputs(o, 1) == Error::Ok) {
          Tensor t = o[0].toTensor();
          const float* p = t.const_data_ptr<float>();
          for (int i = 0; i < t.numel(); i++) acc += (double)p[i] * (i + 1);
          bits = output_bits(p, (size_t)t.nbytes());
        }
        r->sum = acc;
        r->bits = bits;
        r->ok_runs = ok_runs;
        r->done = true;
        vTaskDelete(nullptr);
      };
      fill_inputs(*plain_method);
      fill_inputs(*ml_method);
      const uint32_t race_forks0 = esp32_backend_split_jobs();
      // Pinned to opposite cores so they genuinely overlap rather than
      // time-slicing on one. The forking (min-latency) handle runs on core 0:
      // from core 1, the worker's own core, it would keep its ops whole, and
      // the race would not cover a split at all. Its worker half then
      // shares core 1 with the background handle's execute.
      xTaskCreatePinnedToCore(racer_main, "et_race0", 4096, &racers[0],
                              uxTaskPriorityGet(nullptr), nullptr, 1);
      xTaskCreatePinnedToCore(racer_main, "et_race1", 4096, &racers[1],
                              uxTaskPriorityGet(nullptr), nullptr, 0);
      for (int spin = 0; spin < 2000 && !(racers[0].done && racers[1].done);
           ++spin) {
        vTaskDelay(pdMS_TO_TICKS(10));
      }
      printf("--- mlp_race_bg.pte  [handle A, executed concurrently] ---\n");
      printf("  checksum: %.6f\n", (double)racers[0].sum);
      printf("  output bits: %08lx\n", (unsigned long)racers[0].bits);
      printf("--- mlp_race_ml.pte  [handle B, executed concurrently] ---\n");
      printf("  checksum: %.6f\n", (double)racers[1].sum);
      printf("  output bits: %08lx\n", (unsigned long)racers[1].bits);
      printf("  concurrent: both finished %s, ok executes %d+%d of %d+%d, "
             "forks %u\n",
             (racers[0].done && racers[1].done) ? "yes" : "NO",
             (int)racers[0].ok_runs, (int)racers[1].ok_runs, kRaceRuns,
             kRaceRuns,
             (unsigned)(esp32_backend_split_jobs() - race_forks0));
    }
  }
  for (size_t i = 0; i < n_plain && i < 2; i++) heap_caps_free(plain_planned[i]);
  for (size_t i = 0; i < n_ml && i < 2; i++) heap_caps_free(ml_planned[i]);
  heap_caps_free(temp2);
  heap_caps_free(arena2);
}

#ifdef HAVE_PLE_INPUTS
#ifdef STORY_INPUTS
#include "story_runner.inc"
#endif
#ifdef BENCH_INPUTS
#include "bench_runner.inc"
#endif

// The lean build's one program: the story model (story_runner.inc) or the
// benchmarked one (bench_runner.inc), its program and weights mapped from
// their flash partitions.
static void run_ple(void) {
#ifdef BENCH_INPUTS
  run_bench();
#else
  run_story();
#endif
}
#endif  // HAVE_PLE_INPUTS

// Run a cached sequence method repeatedly. Decode has one token per execute;
static bool execute_glimmer_entry(
    const Program& program, const char* method_name,
    Span<uint8_t>* planned_spans, size_t n_planned,
    const NamedDataMap* named_data, uint8_t* method_arena_data,
    size_t method_arena_size, const int* tokens, int steps, int start_pos,
    const char* result_label) {
  // Only one method is resident at a time. Both MemoryManagers point at the
  // same planned spans, however, so memory ID 2 survives destruction of the
  // prefill Method and becomes the decode Method's KV state. Reusing (and
  // clearing) the method arena between them also proves the cache did not live
  // accidentally in method metadata storage.
  TrackingAllocator method_allocator(method_arena_size, method_arena_data);
  TieredTempAllocator temp_allocator(sizeof(temp_arena), temp_arena);
  HierarchicalAllocator planned_allocator({planned_spans, n_planned});
  MemoryManager memory_manager(
      &method_allocator, &planned_allocator, &temp_allocator);
  auto method = program.load_method(
      method_name, &memory_manager, nullptr, named_data);
  if (!method.ok()) {
    printf("  load_method '%s' FAILED: %d\n", method_name,
           (int)method.error());
    return false;
  }
  if (method->inputs_size() != 2 || method->outputs_size() != 1) {
    printf("  method '%s' has unexpected IO: %u in, %u out\n", method_name,
           (unsigned)method->inputs_size(), (unsigned)method->outputs_size());
    return false;
  }
  printf("  method '%s': arena peak after load %u B\n", method_name,
         (unsigned)method_allocator.peak());

  for (int step = 0; step < steps; ++step) {
    EValue inputs[2];
    if (method->get_inputs(inputs, 2) != Error::Ok) {
      printf("  %s get_inputs FAILED\n", method_name);
      return false;
    }
    Tensor token_tensor = inputs[0].toTensor();
    Tensor position_tensor = inputs[1].toTensor();
    const int chunk = (int)token_tensor.numel();
    for (int index = 0; index < chunk; ++index) {
      token_tensor.mutable_data_ptr<int64_t>()[index] =
          tokens[step * chunk + index];
    }
    const int position = start_pos + step * chunk;
    position_tensor.mutable_data_ptr<int64_t>()[0] = position;

    const int64_t begin_us = esp_timer_get_time();
    const Error error = method->execute();
    const int64_t elapsed_us = esp_timer_get_time() - begin_us;
    if (error != Error::Ok) {
      printf("  %s step %d execute FAILED: %d\n", method_name, step,
             (int)error);
      return false;
    }
    EValue outputs[1];
    if (method->get_outputs(outputs, 1) != Error::Ok) {
      printf("  %s get_outputs FAILED\n", method_name);
      return false;
    }
    Tensor logits = outputs[0].toTensor();
    const float* values = logits.const_data_ptr<float>();
    double checksum = 0.0;
    for (int index = 0; index < logits.numel(); ++index) {
      checksum += (double)values[index] * (double)(index + 1);
    }
    printf("  %s step %d: chunk %d pos %d  checksum %.6f  time %lld us  "
           "bits %08lx\n",
           result_label, step, chunk, position, checksum,
           (long long)elapsed_us,
           (unsigned long)output_bits(values, (size_t)logits.nbytes()));
  }
  printf("  method '%s': final arena peak %u B of %u\n", method_name,
         (unsigned)method_allocator.peak(), (unsigned)method_arena_size);
  return true;
}

// Run a prompt-prefill method and then token decode from the SAME PTE/PTD and
// SAME memory-ID-2 cache arena. This is the runtime half of the multimethod
// contract checked structurally by test_glimmer_multimethod.py. `label` names
// the result rows (<label>_prefill, <label>_decode) and `kv_tag` the shared-KV
// line, so the int8-K-cache twin of the fixture reports separately.
static void run_glimmer_multimethod(
    const uint8_t* pte, size_t pte_size,
    ::executorch::runtime::DataLoader* weights_loader,
    const char* name = "glimmer_multimethod",
    const char* what = "shared prefill -> decode KV",
    const char* label = "glimmer_multi",
    const char* kv_tag = "glimmer multimethod") {
  printf("--- %s.pte  [%s] (%u B) ---\n", name, what, (unsigned)pte_size);
  char prefill_label[48], decode_label[48];
  snprintf(prefill_label, sizeof(prefill_label), "%s_prefill", label);
  snprintf(decode_label, sizeof(decode_label), "%s_decode", label);
  BufferDataLoader loader(pte, pte_size);
  auto program = Program::load(&loader);
  if (!program.ok()) {
    printf("  Program::load FAILED: %d\n", (int)program.error());
    return;
  }
  if (program->num_methods() != 2) {
    printf("  expected two methods, found %u\n",
           (unsigned)program->num_methods());
    return;
  }

  auto weights = executorch::extension::FlatTensorDataMap::load(weights_loader);
  if (!weights.ok()) {
    printf("  FlatTensorDataMap::load FAILED: %d\n", (int)weights.error());
    return;
  }
  auto decode_meta = program->method_meta("decode");
  auto prefill_meta = program->method_meta("prefill");
  if (!decode_meta.ok() || !prefill_meta.ok()) {
    printf("  multimethod metadata FAILED: decode=%d prefill=%d\n",
           decode_meta.ok() ? 0 : (int)decode_meta.error(),
           prefill_meta.ok() ? 0 : (int)prefill_meta.error());
    return;
  }
  const size_t n_planned = decode_meta->num_memory_planned_buffers();
  if (n_planned != 2
      || prefill_meta->num_memory_planned_buffers() != n_planned) {
    printf("  expected mem-id-1/2 plans for both methods: decode=%u prefill=%u\n",
           (unsigned)n_planned,
           (unsigned)prefill_meta->num_memory_planned_buffers());
    return;
  }

  Span<uint8_t> spans[2];
  uint8_t* planned[2] = {nullptr, nullptr};
  size_t plan_sizes[2] = {0, 0};
  for (size_t index = 0; index < n_planned; ++index) {
    auto decode_size = decode_meta->memory_planned_buffer_size(index);
    auto prefill_size = prefill_meta->memory_planned_buffer_size(index);
    if (!decode_size.ok() || !prefill_size.ok()) {
      printf("  planned-buffer metadata %u FAILED\n", (unsigned)index);
      for (size_t prior = 0; prior < index; ++prior) {
        heap_caps_free(planned[prior]);
      }
      return;
    }
    if (index == 1 && *decode_size != *prefill_size) {
      printf("  mutable mem-id-2 ABI differs: decode=%u prefill=%u\n",
             (unsigned)*decode_size, (unsigned)*prefill_size);
      for (size_t prior = 0; prior < index; ++prior) {
        heap_caps_free(planned[prior]);
      }
      return;
    }
    const size_t wanted = std::max((size_t)*decode_size,
                                   (size_t)*prefill_size);
    const size_t allocation = (wanted + 15u) & ~15u;
    planned[index] = static_cast<uint8_t*>(
        heap_caps_aligned_alloc(16, allocation, MALLOC_CAP_SPIRAM));
    const char* tier = "PSRAM";
    if (planned[index] == nullptr) {
      planned[index] = static_cast<uint8_t*>(
          heap_caps_aligned_alloc(16, allocation, MALLOC_CAP_8BIT));
      tier = "internal";
    }
    if (planned[index] == nullptr) {
      printf("  shared planned buffer %u alloc FAILED (%u B)\n",
             (unsigned)index, (unsigned)wanted);
      for (size_t prior = 0; prior < index; ++prior) {
        heap_caps_free(planned[prior]);
      }
      return;
    }
    memset(planned[index], 0, allocation);
    plan_sizes[index] = wanted;
    spans[index] = Span<uint8_t>(planned[index], wanted);
    printf("  shared mem-id-%u: %u B (%s)\n", (unsigned)(index + 1),
           (unsigned)wanted, tier);
  }

  size_t arena_size = 128 * 1024;
  uint8_t* arena = static_cast<uint8_t*>(
      heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_SPIRAM));
  if (arena == nullptr) {
    arena = static_cast<uint8_t*>(
        heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_8BIT));
  }
  if (arena == nullptr) {
    printf("  multimethod arena alloc FAILED (%u B)\n", (unsigned)arena_size);
    for (uint8_t* allocation : planned) {
      heap_caps_free(allocation);
    }
    return;
  }

  const bool prefill_ok = execute_glimmer_entry(
      *program, "prefill", spans, n_planned, &weights.get(), arena, arena_size,
      kGlimmerPrefillTokens, 1, 0, prefill_label);
  // A fresh method allocator must not carry state across the boundary. The
  // exact decode oracle will pass only if the separately planned mem-id-2 span
  // above did carry all eight K/V caches across it.
  memset(arena, 0, arena_size);
  const bool decode_ok = prefill_ok && execute_glimmer_entry(
      *program, "decode", spans, n_planned, &weights.get(), arena, arena_size,
      kGlimmerDecodeTokens, kGlimmerDecodeSteps, kGlimmerPrefillChunk,
      decode_label);
  if (prefill_ok && decode_ok) {
    printf("  %s shared KV: %u B, sequential reload: YES\n", kv_tag,
           (unsigned)plan_sizes[1]);
  }

  heap_caps_free(arena);
  for (uint8_t* allocation : planned) {
    heap_caps_free(allocation);
  }
}

// vectorized prefill has C. Reading C from method metadata keeps the runner
// architecture-agnostic and makes their parity/timing directly comparable.
//
// Repeatedly is the point. A single step would pass even if the mutable buffers were not
// persisting, because at pos=0 there is no history to lose. Every step's logits are
// printed so the host can check all of them.
//
// `tier_budget` is the backend's weight_tier_budget for this row's load (0:
// every weight read from flash, the default; -1: auto, every weight PSRAM
// holds). The copies go with the method (weight_cache off), and the options
// return to their defaults when the row is done.
static void run_cached_sequence(const char* label, const uint8_t* pte,
                                size_t pte_size,
                                ::executorch::runtime::DataLoader* wts_loader,
                                const int* tokens, int steps,
                                int tier_budget = 0) {
  printf("--- %s.pte  [PLE nano-Glimmer cached sequence] (%u B) ---\n",
         label, (unsigned)pte_size);
  // Declared first, so it runs last: after the method and its copies are gone.
  struct WeightTier {
    explicit WeightTier(int budget) : on(budget != 0) {
      if (on) {
        set(budget, false);
      }
    }
    ~WeightTier() {
      if (on) {
        set(0, true);
      }
    }
    static void set(int budget, bool cache) {
      executorch::runtime::BackendOptions<2> tier;
      tier.set_option("weight_tier_budget", budget);
      tier.set_option("weight_cache", cache);
      (void)executorch::runtime::set_option("Esp32Backend", tier.view());
    }
    const bool on;
  } weight_tier(tier_budget);
  BufferDataLoader loader(pte, pte_size);
  auto program = Program::load(&loader);
  if (!program.ok()) {
    printf("  Program::load FAILED: %d\n", (int)program.error());
    return;
  }
  const char* mname = program->get_method_name(0).get();

  ::executorch::runtime::Result<executorch::extension::FlatTensorDataMap> wts =
      executorch::extension::FlatTensorDataMap::load(wts_loader);
  if (!wts.ok()) {
    printf("  FlatTensorDataMap::load FAILED: %d\n", (int)wts.error());
    return;
  }

  auto meta = program->method_meta(mname);
  if (!meta.ok()) {
    printf("  method_meta FAILED: %d\n", (int)meta.error());
    return;
  }
  // Mutable buffers are memory-planned like everything else, so the planned arenas
  // already include the KV cache; nothing extra to allocate here.
  Span<uint8_t> spans[4];
  size_t n_planned = meta->num_memory_planned_buffers();
  if (n_planned > 4) {
    printf("  too many planned buffers: %u\n", (unsigned)n_planned);
    return;
  }
  for (size_t i = 0; i < n_planned; i++) {
    size_t sz = (size_t)meta->memory_planned_buffer_size(i).get();
    // PSRAM first, internal as fallback. The decode plan is dominated by the KV
    // cache and its functionalised write temporaries, both linear in seq_len: 168 KB
    // at seq_len 16, 2.66 MB at 256. On real hardware that belongs in PSRAM; QEMU
    // here has PSRAM disabled (its 32 MB model exhausts the MMU vaddr range and
    // breaks partition mmap), so the fallback is what runs
    // and seq_len is kept small for the emulated case.
    uint8_t* p = (uint8_t*)heap_caps_malloc(sz, MALLOC_CAP_SPIRAM);
    const char* tier = "PSRAM";
    if (p == nullptr) {
      p = (uint8_t*)heap_caps_malloc(sz, MALLOC_CAP_8BIT);
      tier = "internal";
    }
    if (p == nullptr) {
      printf("  planned buffer %u alloc FAILED (%u B)\n", (unsigned)i, (unsigned)sz);
      return;
    }
    // Mutable-buffer initial values are deliberately not stored in a .pte: only
    // their shape and dtype survive export. Decode models nevertheless require a
    // zero initial KV cache. heap_caps_malloc() makes no such promise, so leaving
    // this implicit lets prior heap contents become attention history on step 0.
    memset(p, 0, sz);
    spans[i] = Span<uint8_t>(p, sz);
    printf("  planned buffer %u: %u B (%s), internal free now %u B\n",
           (unsigned)i, (unsigned)sz, tier,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  }
  // The decode graph is ~630 nodes against the block's ~50, so it needs far more
  // method arena than the shared static one: ExecuTorch allocates the EValue table,
  // tensor metadata and every operand list from here, and 48 KB returns
  // MemoryAllocationFailed (0x21). Take a generous slice of whatever is free instead
  // of guessing a constant that will be wrong on the next model.
  // Taken AFTER the planned buffers, which are a fixed size known from the program:
  // claiming the arena first would starve them.
  // Capped rather than "take the largest free block": taking everything would leave
  // the heap too fragmented for the models that run afterwards, which need 126 KB
  // contiguous. The peak is printed so the cap can be revisited with evidence:
  // 94,164 B for the 630-node decode graph, so 128 KB leaves headroom without
  // fragmenting the heap for the models that follow. Without room in PSRAM it falls
  // back to internal RAM, 12 KB short of the largest block when that is smaller.
  size_t arena_sz = 128 * 1024;
  uint8_t* arena = (uint8_t*)heap_caps_malloc(arena_sz, MALLOC_CAP_SPIRAM);
  if (arena == nullptr) {
    size_t big = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    if (big < arena_sz) {
      arena_sz = big > 24 * 1024 ? big - 12 * 1024 : 0;
    }
    arena = arena_sz ? (uint8_t*)heap_caps_malloc(arena_sz, MALLOC_CAP_INTERNAL)
                     : nullptr;
  }
  if (arena == nullptr) {
    printf("  could not obtain a method arena\n");
    return;
  }
  printf("  method arena: %u B\n", (unsigned)arena_sz);
  // Freed on return, after the method: its delegates are destroyed then, and
  // read their handles, which the arena holds (release_resources drops their
  // weight copies and the buffers they fetched).
  struct FreedAfterMethod {
    uint8_t* blocks[5];
    size_t n;
    ~FreedAfterMethod() {
      for (size_t i = 0; i < n; ++i) {
        heap_caps_free(blocks[i]);
      }
    }
  } owned{{}, 0};
  for (size_t i = 0; i < n_planned; i++) {
    owned.blocks[owned.n++] = spans[i].data();
  }
  owned.blocks[owned.n++] = arena;
  TrackingAllocator method_allocator(arena_sz, arena);
  ScopedCapsBuffer large_temp(2 * 1024 * 1024, MALLOC_CAP_SPIRAM);
  TieredTempAllocator temp_allocator(
      sizeof(temp_arena), temp_arena, large_temp.size(), large_temp.data());
  HierarchicalAllocator planned_allocator({spans, n_planned});
  MemoryManager memory_manager(
      &method_allocator, &planned_allocator, &temp_allocator);

  auto method = program->load_method(mname, &memory_manager, nullptr, &wts.get());
  if (!method.ok()) {
    printf("  load_method FAILED: %d\n", (int)method.error());
    return;
  }
  printf("  method '%s': %u in, %u out, arena peak %u B\n", mname,
         (unsigned)method->inputs_size(), (unsigned)method->outputs_size(),
         (unsigned)method_allocator.peak());
  if (tier_budget != 0) {
    uint32_t copies = 0, reuses = 0;
    size_t resident = 0;
    esp32_backend_weight_cache_stats(&copies, &reuses, &resident);
    printf("  %s weight tier: budget %d, %u B of weights in PSRAM\n", label,
           tier_budget, (unsigned)resident);
  }

  for (int step = 0; step < steps; step++) {
    EValue in[2];
    if (method->get_inputs(in, 2) != Error::Ok) {
      printf("  get_inputs FAILED\n");
      return;
    }
    Tensor t_tok = in[0].toTensor();
    Tensor t_pos = in[1].toTensor();
    const int chunk = (int)t_tok.numel();
    for (int i = 0; i < chunk; ++i) {
      t_tok.mutable_data_ptr<int64_t>()[i] = tokens[step * chunk + i];
    }
    const int pos = step * chunk;
    t_pos.mutable_data_ptr<int64_t>()[0] = pos;

    const int64_t t0 = esp_timer_get_time();
    Error e = method->execute();
    const int64_t elapsed_us = esp_timer_get_time() - t0;
    if (e != Error::Ok) {
      printf("  step %d execute FAILED: %d\n", step, (int)e);
      return;
    }
    EValue out[1];
    if (method->get_outputs(out, 1) != Error::Ok) {
      return;
    }
    Tensor lt = out[0].toTensor();
    const float* p = lt.const_data_ptr<float>();
    double sum = 0;
    for (int i = 0; i < lt.numel(); i++) {
      sum += (double)p[i] * (double)(i + 1);
    }
    printf("  %s step %d: chunk %d pos %d  checksum %.6f  time %lld us  "
           "bits %08lx\n",
           label, step, chunk, pos, sum, (long long)elapsed_us,
           (unsigned long)output_bits(p, (size_t)lt.nbytes()));
  }
  printf("  method arena peak: %u B of %u\n",
         (unsigned)method_allocator.peak(), (unsigned)arena_sz);
}

// ---- memory bandwidth ------------------------------------------------------
//
// The read bandwidths chips.py records (SRAM, PSRAM, flash), measured on the
// board. Weights stream from flash, so the flash figure is what turns a model's
// weight bytes into time.
//
// It cannot be measured under QEMU -- emulated flash has no relationship to a
// real SPI part -- so this is compiled in only for hardware runs (ET_BENCH_MEM),
// and run_hardware.py is what sets it.
//
// The SRAM and PSRAM rows are POSITIVE CONTROLS, not padding: this N16R8's
// reference figures with the same method (sum volatile uint32 over a buffer,
// time with esp_timer) are 240 MB/s and 60.7 MB/s. If those two reproduce, the
// flash number from the same loop is trustworthy; if they do not, nothing here
// is, and that is worth knowing before the figure is written into chips.py.
#ifdef ET_BENCH_MEM
// Included HERE rather than at the top of the file: `esp_partition.h` ahead of
// the FreeRTOS and xtensa headers breaks specreg.h with "expected unqualified-id
// before numeric constant". esp32_backend.cpp includes <esp_timer.h> late for
// the same reason.
#include "esp_partition.h"
#include "esp_timer.h"

// bench_tiers.cpp: the weights partition the rows read, and the weight
// tiers' kernel-rate rows.
extern "C" const esp_partition_t* bench_model_partition(void);
extern "C" void bench_weight_tiers(void);

static volatile uint32_t g_bench_sink;

// The control row's method: the best of `reps` single passes rather than their
// average, so a tick or an interrupt in one pass does not depress the figure.
// The average is still the honest number for a decode loop, which cannot pick
// its quietest pass.
static double seq_read_mbps_best(const volatile uint32_t* buf, size_t bytes,
                                 int reps) {
  const size_t n = bytes / 4;
  int64_t best = INT64_MAX;
  for (int r = 0; r < reps; ++r) {
    uint32_t acc = 0;
    const int64_t t0 = esp_timer_get_time();
    for (size_t i = 0; i < n; ++i) {
      acc += buf[i];
    }
    const int64_t us = esp_timer_get_time() - t0;
    g_bench_sink = acc;
    if (us < best) best = us;
  }
  return best > 0 ? (double)bytes / (double)best : 0.0;
}

static double seq_read_mbps(const volatile uint32_t* buf, size_t bytes,
                            int reps) {
  const size_t n = bytes / 4;
  const int64_t t0 = esp_timer_get_time();
  uint32_t acc = 0;
  for (int r = 0; r < reps; ++r) {
    for (size_t i = 0; i < n; ++i) {
      acc += buf[i];
    }
  }
  const int64_t us = esp_timer_get_time() - t0;
  g_bench_sink = acc;                       // keep the loop from vanishing
  return us > 0 ? (double)bytes * reps / (double)us : 0.0;   // B/us == MB/s
}

// NOTE on names: locals here are kSramBytes/kPsramBytes, not SR/PS. `specreg.h`
// defines the xtensa SPECIAL REGISTERS as object-like macros -- `#define PS 230`
// among them -- so a local called PS expands to `const size_t 230 = ...`. GCC
// reports that at the macro's definition site, which reads like a bug in an
// ESP-IDF header rather than in this file.
static void bench_memory(void) {
  printf("\n=== memory bandwidth (hardware only) ===\n");
  // Said out loud because QEMU reports thousands of MB/s of "flash", which is
  // emulated RAM and not a number anyone should copy into chips.py.
  printf("  (meaningless under QEMU -- emulated flash is RAM)\n");

  // Internal SRAM. Sized DOWN until it fits rather than fixed at 256 KB: the
  // largest free block is not the free total, so a fixed size can fail to
  // allocate at all. 64 KB is still past the data cache,
  // so even the smallest fallback measures memory rather than cache.
  //
  // MALLOC_CAP_8BIT, not MALLOC_CAP_32BIT: the latter can hand back IRAM, which
  // is faster than the DRAM activations actually live in, and a flattering
  // number is worse than none here -- the SRAM row is a CONTROL.
  size_t kSramBytes = 0;
  uint32_t* sram = nullptr;
  for (size_t want = 256u * 1024u; want >= 64u * 1024u; want /= 2) {
    sram = (uint32_t*)heap_caps_malloc(want, MALLOC_CAP_INTERNAL |
                                                 MALLOC_CAP_8BIT);
    if (sram != nullptr) {
      kSramBytes = want;
      break;
    }
  }
  if (sram != nullptr) {
    for (size_t i = 0; i < kSramBytes / 4; ++i) sram[i] = (uint32_t)i * 2654435761u;
    printf("  SRAM  seq read : %7.1f MB/s   (%u KB; control 240.0)\n",
           seq_read_mbps(sram, kSramBytes, 8), (unsigned)(kSramBytes / 1024));
    heap_caps_free(sram);
    // The control proper, method-matched to the 240.0 reference: allocated
    // with MALLOC_CAP_INTERNAL alone (which may hand back IRAM), the best of
    // 16 passes. The DRAM row above stays the number that describes where
    // activations live.
    uint32_t* control = (uint32_t*)heap_caps_malloc(128u * 1024u,
                                                    MALLOC_CAP_INTERNAL);
    if (control != nullptr) {
      for (size_t i = 0; i < 32u * 1024u; ++i) control[i] = (uint32_t)i;
      printf("  SRAM  seq read (internal, best of 16): %7.1f MB/s   "
             "(128 KB, MALLOC_CAP_INTERNAL, best of 16)\n",
             seq_read_mbps_best(control, 128u * 1024u, 16));
      heap_caps_free(control);
    } else {
      printf("  SRAM  seq read (internal, best of 16): no free 128 KB block\n");
    }
  } else {
    printf("  SRAM  seq read : no free block >= 64 KB\n");
  }

  // PSRAM, if this part has any.
  const size_t kPsramBytes = 1024u * 1024u;
  uint32_t* ps = (uint32_t*)heap_caps_malloc(kPsramBytes, MALLOC_CAP_SPIRAM);
  if (ps != nullptr) {
    for (size_t i = 0; i < kPsramBytes / 4; ++i) ps[i] = (uint32_t)i * 2654435761u;
    printf("  PSRAM seq read : %7.1f MB/s   (control 60.7)\n",
           seq_read_mbps(ps, kPsramBytes, 2));
    heap_caps_free(ps);
  } else {
    printf("  PSRAM seq read : none present\n");
  }

  // FLASH -- the number the controls vouch for. Read through the same
  // esp_partition_mmap path every weight loader uses, from the model's own
  // weights partition (bench_tiers.cpp: the story's, or the parity
  // firmware's int4 decode's), so this measures the route the backend really
  // takes rather than a synthetic best case. 1 MB is far past the cache, so it
  // is streaming, not hitting.
  const esp_partition_t* part = bench_model_partition();
  if (part != nullptr) {
    const void* base = nullptr;
    esp_partition_mmap_handle_t h = 0;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base,
                           &h) == ESP_OK) {
      const size_t fb = part->size < (1024u * 1024u) ? part->size
                                                     : 1024u * 1024u;
      // reps=1: a second pass would read what the cache already holds and
      // report a number the weight stream never sees.
      printf("  FLASH seq read : %7.1f MB/s   (%u KB through esp_partition_mmap, "
             "partition 0x%02x)\n",
             seq_read_mbps((const volatile uint32_t*)base, fb, 1),
             (unsigned)(fb / 1024), (unsigned)part->subtype);
      printf("  ^^ the figure for chips.py's flash_read_mb_s\n");
      esp_partition_munmap(h);
    } else {
      printf("  FLASH seq read : mmap failed\n");
    }
  } else {
    printf("  FLASH seq read : no weights partition\n");
  }
  bench_weight_tiers();
}
#endif  // ET_BENCH_MEM

static void wait_for_hardware_runner() {
#if (defined(ET_PLE_ONLY) || defined(ET_HARDWARE_RUNNER_GATE)) && \
    defined(CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG) && \
    CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#ifdef ET_STORY
  // The story reader is a product: it must start with no laptop attached.
  // A host that wants the whole boot log sends 'G' within the window (the
  // runner does within a few seconds of the reset); otherwise it starts alone.
  const int64_t deadline = esp_timer_get_time() + 8'000'000;
#endif
  // READY repeats while the gate waits: the USB-Serial/JTAG console drops
  // output while no host reads it (and before USB enumerates), so a READY
  // printed once at boot is lost to a host that opens the port a second
  // later. The runner answers each READY it sees with 'G'.
  int64_t next_ready = 0;
  for (;;) {
    const int64_t now = esp_timer_get_time();
    if (now >= next_ready) {
      printf("=== HARDWARE READY ===\n");
      fflush(stdout);
      next_ready = now + 500'000;
    }
    const int byte = getchar();
    if (byte == 'G') break;
    if (byte == EOF) {
      clearerr(stdin);
#ifdef ET_STORY
      if (now > deadline) {
        printf("=== HARDWARE RUNNING (no host) ===\n");
        fflush(stdout);
        return;
      }
#endif
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
  printf("=== HARDWARE RUNNING ===\n");
  fflush(stdout);
  // A GO the host sent before it saw RUNNING is still in flight: drop it here,
  // or the story's console reads it later as 'G' (show the next story).
  const int64_t quiet_until = esp_timer_get_time() + 300'000;
  while (esp_timer_get_time() < quiet_until) {
    if (getchar() == EOF) {
      clearerr(stdin);
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
#endif
}

extern "C" void app_main(void) {
  wait_for_hardware_runner();
#ifdef ET_PAPERCOLOR_DISPLAY
  const size_t internal_before_display =
      heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const bool papercolor_ready = et_papercolor::start();
  printf("PAPERCOLOR_STATUS init=%s\n",
         papercolor_ready ? "PASS" : "FAIL");
#endif
  printf("\n=== ExecuTorch on ESP32-S3 ===\n");
#ifdef ET_BENCH_MEM
  bench_memory();
#endif
  // The largest block decides whether the story's 147 KB KV cache (mem-id-2)
  // gets internal SRAM or falls back to PSRAM.
  const size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  printf("[mem] boot: internal free %u B, largest block %u B",
         (unsigned)internal_free,
         (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL |
                                                    MALLOC_CAP_8BIT));
#ifdef ET_PAPERCOLOR_DISPLAY
  printf("; display took %d B",
         (int)internal_before_display - (int)internal_free);
#endif
  printf("\n");
#if defined(ET_STORY) || defined(ET_BENCH_MODEL)
  // MSPI0's PSRAM read timing as IDF's boot-time tuning left it. Octal DTR at
  // 80 MHz is tuned on every boot, and the point varies: table point 4 (extra
  // dummy 1) decodes the story ~0.64% faster than point 5 (extra dummy 2), so
  // timings are only comparable at a common point.
  printf("[mspi] psram din_mode 0x%08x din_num 0x%08x timing_cali 0x%08x core_clk 0x%08x sram_clk 0x%08x\n",
         (unsigned)REG_READ(SPI_MEM_SPI_SMEM_DIN_MODE_REG(0)), (unsigned)REG_READ(SPI_MEM_SPI_SMEM_DIN_NUM_REG(0)),
         (unsigned)REG_READ(SPI_MEM_SPI_SMEM_TIMING_CALI_REG(0)), (unsigned)REG_READ(SPI_MEM_CORE_CLK_SEL_REG(0)),
         (unsigned)REG_READ(SPI_MEM_SRAM_CLK_REG(0)));
#endif

  executorch::runtime::runtime_init();
  // minimal.cpp's weak PAL returns a literal 11223344 here, so this doubles as
  // proof that our strong ESP-IDF override actually linked.
  printf("[pal] et_pal_current_ticks() = %llu us (11223344 => weak PAL)\n",
         (unsigned long long)et_pal_current_ticks());
  Error kg = torch::executor::register_all_kernels();
  Error kc = espressif::register_ops();
  printf("kernels: ET codegen (err %d) + %u custom (err %d)\n", (int)kg,
         (unsigned)espressif::kernel_count(), (int)kc);

  // MEASURED: Espressif's QEMU does not implement RSQRT0.S. It faults with
  // IllegalInstruction (EXCCAUSE 0) at the rsqrt0.s in rmsnorm_f32_s3, while every
  // preceding FPU instruction in the same routine -- lsi, madd.s, wfr, mul.s, add.s --
  // executes fine, so the FPU model is present and only this opcode is missing. The
  // one instruction the kernel exists to reach is the one instruction QEMU lacks, so
  // the assembly path is unreachable under emulation by construction. QEMU therefore
  // runs the scalar C reference, which still exercises the AOT fusion, the operator
  // table, the shim and the numerics -- everything except the ~9 FPU instructions.
  // Under QEMU those stay host-validated by test_rmsnorm.py; the board runs them.
  //
  // This banner prints unconditionally and on purpose: ET_QEMU_NO_RSQRT0 is a CMake
  // cache variable, so it survives in a build directory until it is cleared, and a
  // hardware build that inherited it would otherwise run scalar in silence.
#ifdef ET_QEMU_NO_RSQRT0
  rmsnorm_set_force_scalar(1);
  printf("rmsnorm path: SCALAR (ET_QEMU_NO_RSQRT0 set; QEMU has no rsqrt0.s)\n");
#elif defined(ET_FORCE_SCALAR_RMS)
  rmsnorm_set_force_scalar(1);
  printf("rmsnorm path: SCALAR (ET_FORCE_SCALAR_RMS set)\n");
#else
  printf("rmsnorm path: ASM (rsqrt0.s + 2 Newton-Raphson)\n");
#endif
#ifdef ET_FORCE_SCALAR_BACKEND
  esp32_backend_set_force_scalar(1);
  printf("backend dot path: SCALAR (PIE diagnostic disabled)\n");
#else
  printf("backend dot path: PIE where aligned\n");
#endif

#ifdef ET_PLE_ONLY
  // The lean PLE build: nothing but the one model. Every zoo blob is compiled out
  // (pte_blobs.S is guarded), so the app fits beside the model's own 15 MB of
  // partitions in a real 16 MB layout.
  run_ple();
  // Not just diagnostics: with the early return the compiler eliminates every
  // other reference to esp32_backend.cpp, the archive member is never pulled, and
  // its static Backend registration never runs -- load_method then fails with
  // "Backend Esp32Backend is not registered". This call is the anchor.
  esp32_backend_report();
  const auto ple_ops_stats = espressif::dispatch_stats();
  espressif::report_override_dispatch();
  printf("cached_attention dispatch: %llu call(s), %llu valid position(s)\n",
         (unsigned long long)ple_ops_stats.cached_attention_calls,
         (unsigned long long)ple_ops_stats.cached_attention_positions);
  printf("quantize_rows_i8: %llu call(s), %llu row(s)\n",
         (unsigned long long)ple_ops_stats.quantize_rows_calls,
         (unsigned long long)ple_ops_stats.quantize_rows_rows);
  printf("ple_prepare dispatch: %llu call(s), %llu value(s)\n",
         (unsigned long long)ple_ops_stats.ple_prepare_calls,
         (unsigned long long)ple_ops_stats.ple_prepare_values);
  printf("attn_tail dispatch: %llu call(s), %llu row(s)\n",
         (unsigned long long)ple_ops_stats.attn_tail_calls,
         (unsigned long long)ple_ops_stats.attn_tail_rows);
#ifdef ESP32_FAST_LAYERNORM
  printf("native_layer_norm.out fastmath: calls %llu, rows %llu, "
         "fallback %llu\n",
         (unsigned long long)ple_ops_stats.fast_layer_norm_calls,
         (unsigned long long)ple_ops_stats.fast_layer_norm_rows,
         (unsigned long long)ple_ops_stats.fast_layer_norm_fallback);
#endif
  esp32_backend_report_timing();
#ifdef ET_PAPERCOLOR_DISPLAY
  const bool papercolor_final =
      papercolor_ready && et_papercolor::flush(30'000);
  printf("PAPERCOLOR_STATUS final=%s\n",
         papercolor_final ? "YES" : "NO");
#endif
  printf("=== DONE ===\n");
  vTaskDelete(nullptr);
  return;
#endif

  espressif::run_rmsnorm_domain_check();
  espressif::run_portable_fastpath_check();
  espressif::run_fast_layer_norm_check();
  espressif::quant_lut::run_conformance_check();
  espressif::lut16::run_conformance_check();
  espressif::divsqrt::run_check();
  // The chip's expf, which portable softmax takes, against newlib_expf.h's
  // transcription as the host and the prefill references run it.
  espressif::expf_check::run_check();
  // The non-finite quantize rule (quant_rules.h) on the chip's floor.s,
  // trunc.s and soft float64, against the portable kernel's transcription.
  espressif::nonfinite::run_check(nullptr);

  // Cached sequence models run FIRST, deliberately. Their methods need one
  // contiguous metadata arena, and fragmented internal SRAM later in the zoo
  // can have ample total bytes without a sufficiently large block.
  auto glimmer_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x42);
  if (glimmer_loader.ok()) {
    run_glimmer_multimethod(
        glimmermulti_pte_start,
        (size_t)(glimmermulti_pte_end - glimmermulti_pte_start),
        &glimmer_loader.get());
    // The same fixture with a persistent int8 K cache: same weights (its named
    // data is byte-identical, so it shares this partition) and host checksums
    // exactly equal to the fp32-cache rows above.
    run_glimmer_multimethod(
        glimmermultikc8_pte_start,
        (size_t)(glimmermultikc8_pte_end - glimmermultikc8_pte_start),
        &glimmer_loader.get(), "glimmer_multimethod_kc8",
        "shared prefill -> decode KV, int8 K cache", "glimmer_kc8",
        "glimmer multimethod kc8");
  } else {
    printf("--- glimmer_multimethod.pte SKIPPED: glimmerw partition: %d\n",
           (int)glimmer_loader.error());
  }

  auto dc4_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x46);
  if (dc4_loader.ok()) {
#if defined(ET_PARITY_TIER_BYTES)
    const int decode_budget = (int)ET_PARITY_TIER_BYTES;  // the budgeted pass
#else
    const int decode_budget = 0;
#endif
    run_cached_sequence(
        "decode_i4", decode4_pte_start,
        (size_t)(decode4_pte_end - decode4_pte_start), &dc4_loader.get(),
        kDecodeTokens, kDecodeSteps, decode_budget);
#ifdef ET_BENCH_MEM
    // The same row with every weight copied into PSRAM: against the row
    // above, read from flash, its step times are what the PSRAM tier buys a
    // real int4 decode on this board, and its bits must not move.
    run_cached_sequence(
        "decode_i4_psram", decode4_pte_start,
        (size_t)(decode4_pte_end - decode4_pte_start), &dc4_loader.get(),
        kDecodeTokens, kDecodeSteps, -1);
#endif
  } else {
    printf("--- decode_i4.pte SKIPPED: dcweights4 partition: %d\n",
           (int)dc4_loader.error());
  }

  // The table op's rows (fixtures/quant_lut.py): one program in its own data
  // partition, a method per row, each row's outputs hashed as one tensor.
  // Early, for the same reason as the cached models: two of them plan 114 KB.
  auto luts_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x48);
  if (luts_loader.ok()) {
    static const char* const kLutRows[][2] = {
        {"luts_q8", "luts_q8.pte  [table op: every code, block-uniform]"},
        {"luts_q8_tail", "luts_q8_tail.pte  [table op: 1,059 elements]"},
        {"luts_q8_mixed",
         "luts_q8_mixed.pte  [table op: lanes a quarter-range apart]"},
        {"luts_nan",
         "luts_nan.pte  [table op: NaN payloads, infinities, denormals]"},
    };
    for (const auto& row : kLutRows) {
      run_model(row[1],
                static_cast<const uint8_t*>(luts_loader.get().mapped_base()),
                *luts_loader.get().size(), nullptr, row[0], true);
    }
  } else {
    printf("--- luts.pte SKIPPED: luts partition: %d\n",
           (int)luts_loader.error());
  }

  run_model("mlp.pte  [portable fp32 kernels]", mlp_pte_start,
            (size_t)(mlp_pte_end - mlp_pte_start));
  run_model("bmmhalf.pte  [fp16 BMM portable-fallback guard]",
            bmmhalf_pte_start,
            (size_t)(bmmhalf_pte_end - bmmhalf_pte_start));
  run_model("convnet.pte  [conv2d + maxpool]", convnet_pte_start,
            (size_t)(convnet_pte_end - convnet_pte_start));
  run_model("block.pte  [transformer block]", block_pte_start,
            (size_t)(block_pte_end - block_pte_start));
  // Coverage zoo. ExecuTorch has neither aten::lstm nor aten::gru, so export unrolls
  // the time loop and decomposes the gates -- these models' graph depth comes
  // from control flow rather than layer count.
  run_model("lstm.pte  [recurrence, unrolled]", lstm_pte_start,
            (size_t)(lstm_pte_end - lstm_pte_start));
  run_model("gru.pte  [recurrence, unrolled]", gru_pte_start,
            (size_t)(gru_pte_end - gru_pte_start));
  // Rank coverage. Both reach ExecuTorch's 2-D kernels: conv1d and 1-D pooling get
  // there by unsqueezing, and min-pool is -max_pool(-x). No 1-D kernel exists.
  run_model("conv1d.pte  [conv1d + depthwise + grouped]", conv1d_pte_start,
            (size_t)(conv1d_pte_end - conv1d_pte_start));
  run_model("pool.pte  [max/avg/min pooling, 1-D and 2-D]", pool_pte_start,
            (size_t)(pool_pte_end - pool_pte_start));
  // Architecture coverage: data-dependent recurrence (mamba), spline basis (kan),
  // one block's weights consumed by three unrolled call sites (looped), the
  // soft-activation gauntlet, and every norm flavour incl. the fused RMSNorm in a
  // channels-last permute sandwich.
  run_model("mamba.pte  [selective SSM, unrolled scan]", mamba_pte_start,
            (size_t)(mamba_pte_end - mamba_pte_start));
  run_model("kan.pte  [Kolmogorov-Arnold: B-spline basis + matmuls]", kan_pte_start,
            (size_t)(kan_pte_end - kan_pte_start));
  run_model("looped.pte  [ALBERT-style shared block x3]", looped_pte_start,
            (size_t)(looped_pte_end - looped_pte_start));
  run_model("acts.pte  [soft-activation gauntlet incl. snake/glu]", acts_pte_start,
            (size_t)(acts_pte_end - acts_pte_start));
  run_model("norms.pte  [batch/group/instance/layer/scaled+weightless RMS norm]", norms_pte_start,
            (size_t)(norms_pte_end - norms_pte_start));
  // memory_plan="esp32": sub, clamp and hardtanh writing over a dying operand.
  run_model("memplan.pte  [in-place sub/clamp/hardtanh]", memplan_pte_start,
            (size_t)(memplan_pte_end - memplan_pte_start));
  run_model("ropefull.pte  [HF full-width non-duplicated RoPE table]", ropefull_pte_start,
            (size_t)(ropefull_pte_end - ropefull_pte_start));
  // The decode attention tail alone, fused into espressif::attn_tail: run_parity
  // holds this row to the host mirror the decode goldens use, not only to eager.
  run_model("attntail.pte  [fused decode attention tail]", attntail_pte_start,
            (size_t)(attntail_pte_end - attntail_pte_start));
  // Graph NN: index_put with ACCUMULATION over duplicate destination indices --
  // scatter semantics no prior row exercises (the KV cache writes unique slots).
  run_model("gcn.pte  [2-layer GCN, scatter-add message passing]", gcn_pte_start,
            (size_t)(gcn_pte_end - gcn_pte_start));
  // Weights read straight out of a mapped flash partition -- no app-image space,
  // no RAM. This is the path the 12.58 MB PLE table needs.
  auto wts_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x40);
  if (wts_loader.ok()) {
    run_model("block_split.pte  [weights from mapped flash partition]",
              block_split_pte_start,
              (size_t)(block_split_pte_end - block_split_pte_start),
              &wts_loader.get());
  } else {
    printf("--- block_split.pte SKIPPED: partition load failed: %d\n",
           (int)wts_loader.error());
  }
  run_model("mlp_delegated.pte  [addmm -> Esp32Backend delegate]",
            mlpdel_pte_start, (size_t)(mlpdel_pte_end - mlpdel_pte_start));
  run_model("mlp_fp32_backend.pte  [opcode-1 inline fp32 backend control]",
            mlpf32_pte_start, (size_t)(mlpf32_pte_end - mlpf32_pte_start));
  run_model("mlp_perop.pte  [same MLP, two one-op delegate calls]",
            mlpper_pte_start, (size_t)(mlpper_pte_end - mlpper_pte_start));
  // The same MLP down the other two rungs. i4 halves the stored weight bytes and
  // expands nibbles into the existing PIE kernel; i16 doubles them and takes the
  // int64 scalar path. Each is held against its own host reference, because a coarser
  // quantiser is a different number, not a wrong one.
  run_model("mlp_i4.pte  [int4 weights -> nibble unpack -> PIE]",
            mlpi4_pte_start, (size_t)(mlpi4_pte_end - mlpi4_pte_start));
  run_model("mlp_i16.pte  [int16 weights, int64 accumulate]",
            mlpi16_pte_start, (size_t)(mlpi16_pte_end - mlpi16_pte_start));
  run_model("embed_i4.pte  [int4 embedding, odd row + ragged scale group]",
            embi4_pte_start, (size_t)(embi4_pte_end - embi4_pte_start));
  run_model("embed_i16.pte  [int16 embedding, ragged scale group]",
            embi16_pte_start, (size_t)(embi16_pte_end - embi16_pte_start));
  {
    // The selected-row head has its own loop. It must not be counted as the
    // decode-specialised plain-i8 loop, whose counter is read as proof the
    // fast path ran.
    const uint32_t fast_rows0 = esp32_backend_i8_fast_rows();
    run_model("selected_head.pte  [dynamic 257->7 output rows]",
              selhead_pte_start,
              (size_t)(selhead_pte_end - selhead_pte_start));
    printf("selected head: decode fast-path rows %u\n",
           (unsigned)(esp32_backend_i8_fast_rows() - fast_rows0));
  }
  // Embedding -> Linear inside ONE delegate: the gather writes a chain slot
  // and the matmul reads it back: the first two layers of most models.
  run_model("embed_linear.pte  [gather -> chain slot -> matmul]",
            embedlin_pte_start,
            (size_t)(embedlin_pte_end - embedlin_pte_start));
  run_model("grouped_i4.pte  [i4 addmm, ragged groups + fp16 scales]",
            grpi4_pte_start, (size_t)(grpi4_pte_end - grpi4_pte_start));
  run_model("grouped_i8.pte  [i8 addmm, ragged groups + fp16 scales]",
            grpi8_pte_start, (size_t)(grpi8_pte_end - grpi8_pte_start));
  // 40 columns: wide enough to fork in the split pass (grouped_i4_split).
  run_model("grouped_i4_wide.pte  [i4 addmm, ragged groups, 40 columns]",
            grpi4w_pte_start, (size_t)(grpi4w_pte_end - grpi4w_pte_start));
  // Calibrated static activation scales: quantisation at descriptor act_scale
  // instead of per-row absmax. Held against a reference computed WITH the fixed
  // scales -- fc2 reads the chained intermediate, so its static scale genuinely
  // differs from dynamic and a runtime ignoring the flag misses this checksum.
  run_model("mlp_static.pte  [calibrated static activation scales]",
            mlpstat_pte_start, (size_t)(mlpstat_pte_end - mlpstat_pte_start));
  // The attention core twice: portable baseline, then with the OPT-IN int8
  // q@kT (OP_BMM_DYN_I8 -- both operands quantised at execute, no weights;
  // the p@v bmm stays portable by the softmax-consumer gate).
  run_model("attnqk.pte  [attention core, portable bmms]", attnqk_pte_start,
            (size_t)(attnqk_pte_end - attnqk_pte_start));
  run_model("attnqk_qk.pte  [opt-in int8 q@kT, dynamic both sides]",
            attnqkq_pte_start, (size_t)(attnqkq_pte_end - attnqkq_pte_start));
  // ... and with its keys kept as a persistent int8 K cache: quantised once by
  // espressif::quantize_rows_i8, read in place by OP_BMM_KCACHE_I8. Portable,
  // then delegated -- whose reference is attnqk_qk's exactly.
  run_model("attnkc8.pte  [int8 K cache, portable]", attnkc8_pte_start,
            (size_t)(attnkc8_pte_end - attnkc8_pte_start));
  run_model("attnkc8_qk.pte  [opt-in int8 q@kT, int8 K cache read in place]",
            attnkc8q_pte_start, (size_t)(attnkc8q_pte_end - attnkc8q_pte_start));
  run_model("mlpr.pte  [relu chain, portable]", mlpr_pte_start,
            (size_t)(mlpr_pte_end - mlpr_pte_start));
  // Final relu FUSED into the delegate (clamp between accumulate and store);
  // mid-chain relu portable, because claiming it would merge the partitions.
  run_model("mlpr_delegated.pte  [final relu fused into delegate]",
            mlprdel_pte_start, (size_t)(mlprdel_pte_end - mlprdel_pte_start));
  run_model("mlps.pte  [silu + gelu heads, portable]", mlps_pte_start,
            (size_t)(mlps_pte_end - mlps_pte_start));
  // The silu DIAMOND (sigmoid+mul sharing the matmul output) and an exact gelu,
  // each fused into its own delegate call and applied between accumulate and store.
  run_model("mlps_delegated.pte  [fused silu diamond + fused gelu]",
            mlpsdel_pte_start, (size_t)(mlpsdel_pte_end - mlpsdel_pte_start));
  // The fused multiply: a GeGLU-tanh pair whose up projection
  // multiplies in place what gate wrote, chained into down, and an output gate
  // (tanh(h) * sigmoid(x@Wg)) chained into o. M=4 runs the row-blocked
  // activation-sharing group, M=1 the decode one.
  run_model("epiglu_delegated.pte  [gate*up pair in place + sigmoid gate]",
            epiglu_pte_start, (size_t)(epiglu_pte_end - epiglu_pte_start));
  run_model("epiglu_m1.pte  [the same at M=1: decode group path]",
            epiglum1_pte_start,
            (size_t)(epiglum1_pte_end - epiglum1_pte_start));
  // The one-op activations, each fused onto its own head behind a
  // chained hardswish trunk: hardsigmoid, leaky ReLU, ReLU^2, tanh-GELU and
  // sigmoid.
  run_model("epiacts_delegated.pte  [hardswish/hardsigmoid/leaky/relu2/"
            "gelu-tanh/sigmoid epilogues]",
            epiacts_pte_start, (size_t)(epiacts_pte_end - epiacts_pte_start));
  // The delegate tagged its weights external, so they come from their own mapped
  // flash partition and the runtime never copies them.
  auto dw_loader = esp32s3::EspPartitionDataLoader::from_subtype(0x41);
  if (dw_loader.ok()) {
    run_model("block_delegated.pte  [5 mm -> delegate, weights from flash]",
              blkdel_pte_start, (size_t)(blkdel_pte_end - blkdel_pte_start),
              &dw_loader.get());
  } else {
    printf("--- block_delegated SKIPPED: dweights partition: %d\n",
           (int)dw_loader.error());
  }
  run_model("block_i4.pte  [transformer block, packed i4 projections]",
            blki4_pte_start, (size_t)(blki4_pte_end - blki4_pte_start));
  run_model("block_i16.pte  [transformer block, i16 accuracy path]",
            blki16_pte_start, (size_t)(blki16_pte_end - blki16_pte_start));
  // Same binary, same weights, same inputs: only the inner loop differs. Proves the
  // scalar fallback is numerically identical across a whole model rather than for one
  // column, and on real hardware this pair is the memory-vs-compute measurement.
  esp32_backend_set_force_scalar(1);
  run_model("mlp_scalarpath.pte  [same model, PIE disabled]", mlpdel_pte_start,
            (size_t)(mlpdel_pte_end - mlpdel_pte_start));
  esp32_backend_set_force_scalar(0);
  // The min-latency (column-split) pass: the same models, a worker on the
  // second core taking the low half of every wide output row. The DEFAULT
  // posture stays background inference -- one core for the model, one for the
  // application -- so the split is opt-in per run, and its values must be
  // BIT-IDENTICAL: a split moves columns between cores, never any column's
  // accumulation order. The rows cover the split-relevant paths: plain i8, i4
  // (per-core unpack scratch), grouped per-row fp16 scales and the in-place
  // gate*up pair.
  //
  // Enabled through the ExecuTorch option API rather than the C hook, so the
  // bit-exactness rows below double as proof that set_option actually
  // reaches the execute() path -- an option that round-trips but changes nothing
  // would pass run_backend_options_check() and fail here (split_jobs would be 0).
  // Weight zero points and float offsets, before the split pass.
  run_asym_fixtures(false);
  // ExecuTorch's own LLM recipes: the stock quantised K/V cache and embeddings.
  run_stock_llm_fixtures();
  run_backend_options_check();
  {
    executorch::runtime::BackendOptions<1> ml;
    ml.set_option("min_latency", true);
    const Error e = executorch::runtime::set_option("Esp32Backend", ml.view());
    if (e != Error::Ok) {
      printf("  set_option(min_latency) FAILED (%d); split rows will be "
             "single-core\n", (int)e);
    }
  }
  run_model("mlp_splitcols.pte  [dual-core column split, i8]", mlpdel_pte_start,
            (size_t)(mlpdel_pte_end - mlpdel_pte_start));
  run_model("mlp_i4_split.pte  [dual-core split, i4 per-core unpack]",
            mlpi4_pte_start, (size_t)(mlpi4_pte_end - mlpi4_pte_start));
  // Each core's columns carry their own groups' scales (ragged K = 131).
  run_model("grouped_i4_split.pte  [dual-core split, grouped fp16 scales]",
            grpi4w_pte_start, (size_t)(grpi4w_pte_end - grpi4w_pte_start));
  // The in-place gate*up pair with both members split across the cores: up
  // must read each act(gate) column on the core that is about to overwrite it.
  run_model("epiglu_split.pte  [dual-core split, gate*up pair in place]",
            epiglum1_pte_start,
            (size_t)(epiglum1_pte_end - epiglum1_pte_start));
  // The affine-weight rows, split by the same worker.
  run_asym_fixtures(true);
  // -1, not 0: clear the override and hand every handle back to its own
  // compile spec, which is what the reentrancy row below needs to observe.
  // Deliberately via the C hook, so BOTH paths onto apply_split_override() are
  // exercised in one run and cannot drift apart unnoticed.
  esp32_backend_set_split(-1);
  run_reentrancy_check();
  run_blob_validation_check();
  run_asym_refusal_check();
  esp32_asym::run_kcache_zero_point_check();
  esp32_asym::run_a16_check();
  esp32_backend_report();
  // The standard ExecuTorch quantization flow (PT2E) landing in the
  // delegate: weights arrive as int4 codes from convert_pt2e and are
  // consumed verbatim, so this row is a claim about the AOT contract as
  // much as about arithmetic.
  run_model("block_wide.pte  [T=24: row blocking past one block, with a "
            "fused residual]", blkwide_pte_start,
            (size_t)(blkwide_pte_end - blkwide_pte_start));
  // conv2d through the im2col gather. Padded and strided, so this row covers
  // the two things a 1x1 fixture would not: reads outside the input, and an
  // output row index that does not track the input's.
  // No portable conv3d row: ExecuTorch's aten::convolution.out refuses a 5-D
  // input, so the delegated row below is the ONLY way this op runs here.
  // A tubelet patch embed -- the V-JEPA2 shape -- through the SAME gather as
  // 1-D and 2-D, via degenerate axes rather than a third implementation.
  run_model("grpconvd.pte  [grouped conv, portable]", grpp_pte_start,
            (size_t)(grpp_pte_end - grpp_pte_start));
  run_model("grpconvedged.pte  [ambiguous grouped conv, portable]",
            grpep_pte_start, (size_t)(grpep_pte_end - grpep_pte_start));

  run_model("grpconv8d.pte  [grouped g=8 s2, portable]", grp8p_pte_start,
            (size_t)(grp8p_pte_end - grp8p_pte_start));
  run_model("convtd.pte  [ConvTranspose2d, portable]", cvtp_pte_start,
            (size_t)(cvtp_pte_end - cvtp_pte_start));
  // A U-Net up-convolution through the SAME gather, with the zero-stuffing
  // expressed as an index test rather than materialised.
  run_model("convt_delegated.pte  [ConvTranspose2d 2x2 s2 -> the gather]",
            cvt_pte_start, (size_t)(cvt_pte_end - cvt_pte_start));
  run_model("convt_i4.pte  [ConvTranspose2d AOT flip + packed i4]",
            cvti4_pte_start, (size_t)(cvti4_pte_end - cvti4_pte_start));
  run_model("convt_i16.pte  [ConvTranspose2d AOT flip + int64 i16 path]",
            cvti16_pte_start, (size_t)(cvti16_pte_end - cvti16_pte_start));
  // Output padding and dilation: the size check takes the padded extent (up
  // to max(stride, dilation) - 1 past the computed one), and the gather's
  // mapping spaces the taps by the dilation, on each axis its own.
  run_model("convtopd.pte  [ConvTranspose2d s2 p1 output_padding 1, portable]",
            cvtopp_pte_start, (size_t)(cvtopp_pte_end - cvtopp_pte_start));
  run_model("convtop_delegated.pte  [ConvTranspose2d 3x3 s2 p1 op1 -> the "
            "gather]", cvtop_pte_start,
            (size_t)(cvtop_pte_end - cvtop_pte_start));
  run_model("convtdild.pte  [dilated ConvTranspose2d, portable]",
            cvtdilp_pte_start, (size_t)(cvtdilp_pte_end - cvtdilp_pte_start));
  run_model("convtdil_delegated.pte  [ConvTranspose2d s(2,1) dil(3,2) "
            "op(2,1) -> the gather]", cvtdil_pte_start,
            (size_t)(cvtdil_pte_end - cvtdil_pte_start));
  run_model("convtdil_i16.pte  [dilated ConvTranspose2d, int16 gather]",
            cvtdil16_pte_start,
            (size_t)(cvtdil16_pte_end - cvtdil16_pte_start));

  run_model("grpconv_delegated.pte  [grouped g=2 -> 2 dense conv ops]",
            grp_pte_start, (size_t)(grp_pte_end - grp_pte_start));

  run_model("grpconvedge_delegated.pte [grouped split cannot look depthwise]",
            grpe_pte_start, (size_t)(grpe_pte_end - grpe_pte_start));

  // Cardinality 8 and strided: g=2 leaves only one boundary to get wrong and
  // two equal-sized slices, which can hide a wrong kernel choice.
  run_model("grpconv8_delegated.pte [grouped g=8 s2 -> 8 dense conv ops]",
            grp8_pte_start, (size_t)(grp8_pte_end - grp8_pte_start));

  // The activation must land on ALL g ops. On one of them the output is
  // correct for those channels and raw for the rest -- a plausible checksum.
  run_model("grpconvrelu_delegated.pte [grouped g=8 + fused relu]",
            grpr_pte_start, (size_t)(grpr_pte_end - grpr_pte_start));
  run_model("grpconvrelud.pte  [grouped g=8 + relu, portable]",
            grprp_pte_start, (size_t)(grprp_pte_end - grprp_pte_start));
  // Two chained conv+SiLU diamonds: the shape that needs more than two scratch
  // slots. Exercises the third slot for real.
  run_model("convsilu_delegated.pte  [conv+silu x2, three-deep chain]",
            cvs_pte_start, (size_t)(cvs_pte_end - cvs_pte_start));
  // An epilogue fused onto a NON-matmul producer: the descriptor prints "+relu"
  // and no relu survives in the graph.
  run_model("convrelu_delegated.pte  [conv2d with a FUSED relu]",
            cvr_pte_start, (size_t)(cvr_pte_end - cvr_pte_start));
  run_model("dwconv1dd.pte  [depthwise conv1d, portable]", dw1p_pte_start,
            (size_t)(dw1p_pte_end - dw1p_pte_start));
  // Depthwise AND the degenerate height axis at once -- the composition, which
  // is where a bug would sit given each holds separately.
  run_model("dwconv1d_delegated.pte  [depthwise conv1d, direct + kh=1]",
            dw1_pte_start, (size_t)(dw1_pte_end - dw1_pte_start));
  run_model("dwconvd.pte  [depthwise on portable kernels]", dwcp_pte_start,
            (size_t)(dwcp_pte_end - dwcp_pte_start));
  // Depthwise takes its OWN loop, not the im2col path: every output channel
  // reads its own input channel, so the activation varies with the column.
  run_model("dwconv_delegated.pte  [depthwise 3x3, direct, no im2col]",
            dwc_pte_start, (size_t)(dwc_pte_end - dwc_pte_start));
  run_model("dwconv_i4.pte  [depthwise direct path, packed i4 weights]",
            dwi4_pte_start, (size_t)(dwi4_pte_end - dwi4_pte_start));
  run_model("dwconv_grouped_i4.pte  [packed i4, two fp16 scale groups]",
            dwgi4_pte_start, (size_t)(dwgi4_pte_end - dwgi4_pte_start));
  run_model("dwconv_i16.pte  [depthwise i16, per-channel gather]",
            dwi16_pte_start, (size_t)(dwi16_pte_end - dwi16_pte_start));
  run_model("dwconvres_delegated.pte  [depthwise + fused residual]",
            dwres_pte_start, (size_t)(dwres_pte_end - dwres_pte_start));
  run_model("dwconvresd.pte  [depthwise + residual, portable]",
            dwresp_pte_start, (size_t)(dwresp_pte_end - dwresp_pte_start));
  run_model("conv3d_delegated.pte  [conv3d tubelet -> the same gather]",
            c3dd_pte_start, (size_t)(c3dd_pte_end - c3dd_pte_start));
  run_model("conv1dd.pte  [conv1d on portable kernels]", c1ddp_pte_start,
            (size_t)(c1ddp_pte_end - c1ddp_pte_start));
  // 1-D through the SAME gather as 2-D, via a synthesised degenerate height
  // axis. A failure here against a passing convd row would mean that axis is
  // not as degenerate as claimed.
  run_model("conv1d_delegated.pte  [conv1d -> the conv2d gather, kh=1]",
            c1dd_pte_start, (size_t)(c1dd_pte_end - c1dd_pte_start));
  run_model("convd.pte  [conv2d on portable kernels]", convdp_pte_start,
            (size_t)(convdp_pte_end - convdp_pte_start));
  run_model("convd_delegated.pte  [conv2d 3x3 s2 p1 -> im2col matmul]",
            convd_pte_start, (size_t)(convd_pte_end - convd_pte_start));

  // conv + residual: the AOT side fuses it and the runtime must apply it; a
  // dropped residual would be silent without this row.
  run_model("convres_delegated.pte  [conv2d + residual, fused]",
            cvres_pte_start, (size_t)(cvres_pte_end - cvres_pte_start));

  // The fused LayerNorm2d kernel. Portable by design: what is under test is
  // espressif::layer_norm2d, not the partitioner.
  run_model("layernorm2dd.pte  [LayerNorm2d over C, fused]", ln2d_pte_start,
            (size_t)(ln2d_pte_end - ln2d_pte_start));
  // The same kernel where its arithmetic shows: C = 3, weight and bias
  // neither 1 nor 0, inputs off the dyadic ramp; held to the host runner's
  // bits, which a multiply-add fused on the chip would miss.
  run_model("layernorm2dx.pte  [LayerNorm2d, C=3, affine, exact]",
            ln2dx_pte_start, (size_t)(ln2dx_pte_end - ln2dx_pte_start));
  espressif::run_ln2d_refusal_check();

  // A conv under CALIBRATED activation scales. Its rows are patches, so no
  // row's absmax equals the whole input's -- the static scale differs from the
  // dynamic one everywhere, which is what makes a dropped flag visible.
  //
  // Note the "esp32 actstatic: N rows" line is printed BEFORE this row runs, so
  // its count excludes these rows. The verification here is the checksum, which
  // is held against a static reference that differs from the dynamic one by
  // 0.4% -- a runtime ignoring the flag fails it. Written down because a
  // reporting number that looks like it disagrees with reality is how a bug
  // stays hidden.
  run_model("conv_static.pte  [conv2d, calibrated activation scale]",
            cvst_pte_start, (size_t)(cvst_pte_end - cvst_pte_start));
  run_model("convresview_delegated.pte [conv + residual, both reshaped]",
            cvrv_pte_start, (size_t)(cvrv_pte_end - cvrv_pte_start));
  run_model("convresviewd.pte  [conv + reshaped residual, portable]",
            cvrvp_pte_start, (size_t)(cvrvp_pte_end - cvrvp_pte_start));
  run_model("convresd.pte  [conv2d + residual, portable]", cvresp_pte_start,
            (size_t)(cvresp_pte_end - cvresp_pte_start));

  // The i16 kernel indexes a residual separately from the i8 one, so it is a
  // second implementation of the same NCHW-vs-[m,n] decision.
  run_model("convres_i16.pte  [conv2d + residual, i16 weights]",
            cvresq_pte_start, (size_t)(cvresq_pte_end - cvresq_pte_start));

  // The weight-format ladder on a convolution. Conv borrows the matmul
  // kernels through im2col, so i4/i16 should follow for free -- a claim
  // that needs a row of its own.
  run_model("conv_i4.pte  [conv2d, weights at i4]", cvi4_pte_start,
            (size_t)(cvi4_pte_end - cvi4_pte_start));
  run_model("conv_i16.pte  [conv2d, weights at i16]", cvi16_pte_start,
            (size_t)(cvi16_pte_end - cvi16_pte_start));

  // Grouped AT i16: the i16 path gathers and scatters per row instead of
  // staging a block, so its channel offsets are a second implementation of
  // the same rule. Nothing else covers them.
  run_model("grpconv8_i4.pte  [grouped g=8 s2, packed i4 weights]",
            g8i4_pte_start, (size_t)(g8i4_pte_end - g8i4_pte_start));
  run_model("grpconv8_i16.pte  [grouped g=8 s2, weights at i16]",
            g8i16_pte_start, (size_t)(g8i16_pte_end - g8i16_pte_start));
  run_model("mlp_pt2e.pte  [PT2E int4, codes pass-through]", mlppt_pte_start,
            (size_t)(mlppt_pte_end - mlppt_pte_start));
  // The same flow through CONVOLUTIONS: per-output-channel codes reshaped to
  // the packer's [N, K] behind the im2col gather, with a fused ReLU and
  // PT2E's per-tensor activation scale on every op. A refused conv would
  // run portable at fp32, silently: this row holds the delegated path.
  run_model("convnet_pt2e.pte  [PT2E int4 conv + 1x1 + Linear, codes "
            "pass-through]", cvpt_pte_start,
            (size_t)(cvpt_pte_end - cvpt_pte_start));
  // Three projections share one input. Two retain PT2E's dynamic per-tensor
  // activation scale and may share their quantised rows; the third deliberately
  // stays per-row and must not be pulled into that group: the grouping key
  // includes the scale scheme, or all three would silently quantise per row.
  {
    const uint64_t shared0 = esp32_backend_actq_rows_saved();
    const uint64_t pt0 = esp32_backend_act_per_tensor_ops();
    run_model("actshare.pte  [shared input, mixed activation-scale schemes]",
              actshare_pte_start,
              (size_t)(actshare_pte_end - actshare_pte_start));
    printf("actshare: shared_rows=%llu per_tensor_ops=%llu\n",
           (unsigned long long)(esp32_backend_actq_rows_saved() - shared0),
           (unsigned long long)(esp32_backend_act_per_tensor_ops() - pt0));
  }
  // PT2E's activation scheme at its edges: exact rounding ties (torch.round
  // is ties-to-even) and a tensor below its minimum scale. Held to PT2E's own
  // eager quantize/dequantize rather than to a mirror of this runtime.
  run_model("pt2e_ties.pte  [PT2E ties-to-even and minimum scale]",
            pt2eties_pte_start,
            (size_t)(pt2eties_pte_end - pt2eties_pte_start));

#ifdef HAVE_PLE_INPUTS
  run_ple();
#endif

  esp32_backend_report_blocking();
  const auto ops_stats = espressif::dispatch_stats();
  printf("rmsnorm dispatch: asm %d, scalar fallback %d\n", ops_stats.rms_asm,
         ops_stats.rms_scalar);
  printf("rope: %d row(s) rotated\n", ops_stats.rope_rows);
  printf("quantize_rows_i8: %llu call(s), %llu row(s)\n",
         (unsigned long long)ops_stats.quantize_rows_calls,
         (unsigned long long)ops_stats.quantize_rows_rows);
  printf("ln2d: %llu channel column(s) normalised\n",
         (unsigned long long)ops_stats.ln2d_cols);
  espressif::report_override_dispatch();
  printf("attn_tail dispatch: %llu call(s), %llu row(s)\n",
         (unsigned long long)ops_stats.attn_tail_calls,
         (unsigned long long)ops_stats.attn_tail_rows);
#ifdef ESP32_FAST_LAYERNORM
  printf("native_layer_norm.out fastmath: calls %llu, rows %llu, "
         "fallback %llu\n",
         (unsigned long long)ops_stats.fast_layer_norm_calls,
         (unsigned long long)ops_stats.fast_layer_norm_rows,
         (unsigned long long)ops_stats.fast_layer_norm_fallback);
#endif
  printf("[mem] after: internal free %u B\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  // Last, so the totals cover every model in the run rather than only the
  // models before the call.
  esp32_backend_report_timing();
  printf("=== DONE ===\n");
  vTaskDelete(nullptr);
}
