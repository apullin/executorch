/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The bench build's one program (ET_BENCH_MODEL, with ET_PLE_ONLY): one model,
// timed. scripts/run_bench_hardware.py exports it and packs a bench image into
// the program partition (0x43): a header, a label per instruction, then the
// program. Its named data (the model's weights) is the weights partition
// (0x44). The firmware is the same for every model whose operators its
// selective build registers; a model is its two partitions.
//
// One boot measures the model twice, on the same buffers and at the same PSRAM
// timing point:
//
//   flash  every weight read from the flash mapping (weight_tier_budget 0);
//   psram  ranked placement (weight_tier_rank) with every weight PSRAM can
//          hold (budget -1, or ET_BENCH_TIER_BYTES), made by weight_tier_apply
//          after the load and before any execute is timed.
//
// Each measurement runs the model once and checks its output bits against the
// host runner's (the header's), then times `timed_runs` executes, each on a
// freshly generated input and each output checked again, then runs
// `stepped_runs` more one instruction at a time (Method::step) to split the
// time by instruction: each delegate, each portable operator, each
// quantize/dequantize between them. The step pass carries a timer read per
// instruction and is never the speed figure; the executes are.
//
// Placement, printed with every buffer: the largest delegate scratch first, in
// internal SRAM when it fits (else ExecuTorch's temp allocator falls through to
// PSRAM); then the planned buffers, internal first; then the method arena,
// internal first, sized by a probe load into PSRAM (its peak + 25% + 4 KB).
//
// The input is generated, not stored: element i is a hash of i, exactly
// representable in float32, uniform in [-1, 1) (bench_input below and
// run_bench_hardware.bench_input agree bit for bit; the header carries the
// input's bits to prove it on the device).
//
// The host builds this file too (backends/esp32s3/test/host_runner,
// esp32_bench_host), with ESP32_HOST_BUILD, to run an exported image through
// the same code before a board does.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>

#if defined(ESP32_HOST_BUILD)
#include "esp32_platform.h"
#else
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#endif

extern "C" void esp32_backend_weight_cache_stats(uint32_t* copies, uint32_t* hits,
                                                 size_t* resident);
extern "C" void esp32_backend_report_weight_tiers(void);
#if defined(ET_BENCH_MICRO) && ET_BENCH_MICRO
// esp32_backend.cpp's micro-benchmarks (built with the static int8 conv path).
extern "C" void esp32_backend_microbench(void);
#endif
#if defined(ESP32_BACKEND_TIMING)
extern "C" void esp32_backend_reset_timing(void);
#endif

namespace bench {

using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::Method;
using ::executorch::runtime::Program;
using ::executorch::runtime::Span;

// The image's header: little-endian, every field at its natural alignment
// (run_bench_hardware.IMAGE_HEADER packs the same 128 bytes).
struct Header {
  char magic[8];             // "ESPBENCH"
  uint32_t version;          // 1
  uint32_t header_bytes;     // where the program starts: a multiple of 64
  uint32_t program_bytes;
  uint32_t weights_bytes;    // the named data partition 0x44 must hold
  uint32_t weights_bits;     // FNV-1a over those bytes
  uint32_t input_numel;      // float32 elements of the one input
  uint32_t input_seed;
  uint32_t input_bits;       // FNV-1a over the generated input
  uint32_t output_numel;     // float32 elements over every output
  uint32_t output_bits;      // FNV-1a over every output in order: the host runner's
  double output_checksum;    // sum of v[i] * (i + 1) over them
  uint32_t timed_runs;
  uint32_t stepped_runs;
  uint32_t temp_bytes;       // the largest delegate scratch (the host runner's report)
  uint32_t instructions;     // the plan's, one label each
  uint32_t labels_offset;    // NUL-terminated "<kind> <text>" strings
  uint32_t labels_bytes;
  uint64_t macs;             // every multiply-add of the model
  uint64_t macs_delegated;
  char name[32];
};
static_assert(sizeof(Header) == 128, "the bench image header is 128 bytes");

constexpr uint32_t kVersion = 1;
// The psram measurement's weight_tier_budget: -1, every weight PSRAM holds,
// unless the build names a byte count.
#if defined(ET_BENCH_TIER_BYTES)
constexpr int kPsramBudget = ET_BENCH_TIER_BYTES;
#else
constexpr int kPsramBudget = -1;
#endif
constexpr uint32_t kMaxInstructions = 4096;
// The most timed or stepped runs an image may ask for (and esp32_bench_host's
// --timed-runs or --stepped-runs set; run_bench_hardware.MAX_RUNS packs no
// more). The timed runs' times are one int64_t each in one allocation, whose
// byte count must not wrap a 32-bit size_t.
constexpr uint32_t kMaxRuns = 1000;
static_assert(kMaxRuns <= UINT32_MAX / sizeof(int64_t),
              "the timed runs' times fit a 32-bit size_t");
constexpr uint32_t kFastTempFloor = 32 * 1024;
constexpr size_t kMaxPlanned = 4;

inline uint32_t fnv(const void* data, size_t bytes, uint32_t hash = 2166136261u) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ p[i]) * 16777619u;
  }
  return hash;
}

// Element i of the input: lowbias32 of (i + seed * golden ratio), its top 24
// bits scaled into [-1, 1). Every step is exact in float32.
inline float input_value(uint32_t i, uint32_t seed) {
  uint32_t h = i + seed * 0x9E3779B9u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return (float)(h >> 8) * (1.0f / 8388608.0f) - 1.0f;
}

inline int64_t now_us() { return esp_timer_get_time(); }

inline void let_idle_run() {
#if !defined(ESP32_HOST_BUILD)
  vTaskDelay(1);  // between runs, outside every timer: the idle task's turn
#endif
}

// heap_caps with a 16-byte alignment, on the board and the host alike (the
// host's heap_caps_malloc keeps malloc's 16).
inline uint8_t* alloc(size_t bytes, uint32_t caps) {
#if defined(ESP32_HOST_BUILD)
  return static_cast<uint8_t*>(heap_caps_malloc(bytes, caps));
#else
  return static_cast<uint8_t*>(heap_caps_aligned_alloc(16, bytes, caps));
#endif
}

constexpr uint32_t kInternal = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

// `bytes` from internal SRAM if a block holds them, else PSRAM; `tier` says
// which, "none" when neither does.
inline uint8_t* alloc_internal_first(size_t bytes, const char** tier) {
  uint8_t* p = alloc(bytes, kInternal);
  *tier = "internal";
  if (p == nullptr) {
    p = alloc(bytes, MALLOC_CAP_SPIRAM);
    *tier = "PSRAM";
  }
  if (p == nullptr) {
    *tier = "none";
  }
  return p;
}

struct Owned {
  uint8_t* p = nullptr;
  ~Owned() {
    if (p != nullptr) heap_caps_free(p);
  }
};

// The method arena, with the bytes the method asked of it.
class Arena final : public MemoryAllocator {
 public:
  Arena(size_t size, uint8_t* base) : MemoryAllocator(size, base) {}
  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* p = MemoryAllocator::allocate(size, alignment);
    if (p != nullptr) peak_ += size;
    return p;
  }
  size_t peak() const { return peak_; }

 private:
  size_t peak_ = 0;
};

// ExecuTorch's temp allocator, reset after every instruction: a request the
// fast (internal) tier holds is served there, any other from the large one.
// The backend packs one delegate call's scratch into one request.
class Temp final : public MemoryAllocator {
 public:
  Temp(size_t fast_size, uint8_t* fast, size_t large_size, uint8_t* large)
      : MemoryAllocator(0, nullptr), fast_(fast_size, fast), large_(large_size, large) {}
  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* p = size <= fast_.size() ? fast_.allocate(size, alignment) : nullptr;
    if (p != nullptr) {
      fast_used_ += size;
    } else if ((p = large_.allocate(size, alignment)) != nullptr) {
      large_used_ += size;
      ++large_calls_;
    }
    peak_ = std::max(peak_, fast_used_ + large_used_);
    return p;
  }
  void reset() override {
    fast_.reset();
    large_.reset();
    fast_used_ = large_used_ = 0;
  }
  // Method::resolve_operator treats a zero-sized temp allocator as absent.
  uint32_t size() const override { return std::max(fast_.size(), large_.size()); }
  size_t peak() const { return peak_; }
  uint32_t large_calls() const { return large_calls_; }

 private:
  MemoryAllocator fast_;
  MemoryAllocator large_;
  size_t fast_used_ = 0, large_used_ = 0, peak_ = 0;
  uint32_t large_calls_ = 0;
};

// The image, validated against the partition (or file) that holds it.
struct Image {
  Header h;
  const uint8_t* base;
  size_t size;
  const uint8_t* program() const { return base + h.header_bytes; }
  const char* label(uint32_t index) const {
    const char* p = reinterpret_cast<const char*>(base + h.labels_offset);
    const char* end = p + h.labels_bytes;
    for (uint32_t i = 0; i < index && p < end; ++i) p += strnlen(p, end - p) + 1;
    return p < end ? p : "? (no label)";
  }
};

// Whether the run counts are at most kMaxRuns each; says which is not.
inline bool runs_in_range(const Header& h) {
  if (h.timed_runs <= kMaxRuns && h.stepped_runs <= kMaxRuns) return true;
  printf("BENCH_ERROR bench image asks for %lu timed and %lu stepped runs, at most "
         "%lu each\n", (unsigned long)h.timed_runs, (unsigned long)h.stepped_runs,
         (unsigned long)kMaxRuns);
  return false;
}

inline bool parse(const uint8_t* base, size_t size, Image* out) {
  if (base == nullptr || size < sizeof(Header)) {
    printf("BENCH_ERROR no bench image (%u B)\n", (unsigned)size);
    return false;
  }
  memcpy(&out->h, base, sizeof(Header));
  const Header& h = out->h;
  if (memcmp(h.magic, "ESPBENCH", 8) != 0 || h.version != kVersion) {
    printf("BENCH_ERROR not a version-%u bench image (scripts/run_bench_hardware.py "
           "writes one)\n", (unsigned)kVersion);
    return false;
  }
  if (h.header_bytes % 64 != 0 || (uint64_t)h.header_bytes + h.program_bytes > size ||
      (uint64_t)h.labels_offset + h.labels_bytes > h.header_bytes ||
      h.instructions == 0 || h.instructions > kMaxInstructions || h.input_numel == 0 ||
      h.output_numel == 0 || h.name[sizeof(h.name) - 1] != 0) {
    printf("BENCH_ERROR bench image fields out of range\n");
    return false;
  }
  if (!runs_in_range(h)) return false;
  out->base = base;
  out->size = size;
  return true;
}

// Fills the method's one float input with the image's input. Every run gets a
// fresh one: the memory plan may reuse the input's bytes once it is consumed.
inline bool fill_input(Method& method, const Header& h, uint32_t* bits) {
  if (method.inputs_size() != 1) {
    printf("BENCH_ERROR the method has %u inputs, the bench feeds one\n",
           (unsigned)method.inputs_size());
    return false;
  }
  EValue in[1];
  if (method.get_inputs(in, 1) != Error::Ok || !in[0].isTensor()) {
    printf("BENCH_ERROR get_inputs failed\n");
    return false;
  }
  auto t = in[0].toTensor();
  if (t.scalar_type() != ::executorch::aten::ScalarType::Float ||
      (uint32_t)t.numel() != h.input_numel) {
    printf("BENCH_ERROR the input is not %u float32 elements\n", (unsigned)h.input_numel);
    return false;
  }
  float* p = t.mutable_data_ptr<float>();
  for (uint32_t i = 0; i < h.input_numel; ++i) p[i] = input_value(i, h.input_seed);
  if (bits != nullptr) *bits = fnv(p, (size_t)h.input_numel * sizeof(float));
  return true;
}

// FNV-1a over every output's bytes in order, and the weighted checksum over
// their elements, as main.cpp's parity rows compute them.
inline bool read_outputs(Method& method, uint32_t* bits, double* checksum,
                         uint32_t* numel) {
  *bits = 2166136261u;
  *checksum = 0.0;
  *numel = 0;
  for (size_t k = 0; k < method.outputs_size(); ++k) {
    const EValue& v = method.get_output(k);
    if (!v.isTensor()) return false;
    auto t = v.toTensor();
    if (t.scalar_type() != ::executorch::aten::ScalarType::Float) return false;
    const float* p = t.const_data_ptr<float>();
    for (ssize_t i = 0; i < t.numel(); ++i, ++*numel) {
      *checksum += (double)p[i] * (double)(*numel + 1);
    }
    *bits = fnv(p, (size_t)t.nbytes(), *bits);
  }
  return true;
}

struct Buffers {
  Span<uint8_t> spans[kMaxPlanned];
  size_t n_planned = 0;
  uint8_t* arena = nullptr;
  size_t arena_size = 0;
  uint8_t* fast = nullptr;
  size_t fast_size = 0;
  uint8_t* large = nullptr;
  size_t large_size = 0;
};

// One measurement: load under the current backend options (and apply the
// ranked placement), check, time, split. Returns false on any failure.
/// A --backend-timing build (ESP32_BACKEND_TIMING) after each phase: every
/// op shape's time over the phase's executes (esp32_backend_report_timing's
/// "timing by shape" lines, us a call: cycles per output = us x 240 / (m x n)),
/// then zeroed for the next phase. Nothing otherwise.
inline void report_phase_timing(const char* phase) {
#if defined(ESP32_BACKEND_TIMING)
  printf("BENCH_TIMING phase=%s\n", phase);
  esp32_backend_report_timing();
  esp32_backend_reset_timing();
#else
  (void)phase;
#endif
}

inline bool measure(const char* phase, const Image& image, const Program& program,
                    const ::executorch::runtime::NamedDataMap* weights, Buffers& b,
                    int64_t* step_us) {
  const Header& h = image.h;
  const char* mname = *program.get_method_name(0);
  Arena arena(b.arena_size, b.arena);
  Temp temp(b.fast_size, b.fast, b.large_size, b.large);
  HierarchicalAllocator planned({b.spans, b.n_planned});
  MemoryManager memory(&arena, &planned, &temp);

  const int64_t load0 = now_us();
  auto method = program.load_method(mname, &memory, nullptr, weights);
  const int64_t load_us = now_us() - load0;
  if (!method.ok()) {
    printf("BENCH_ERROR phase=%s load_method failed: %d\n", phase, (int)method.error());
    return false;
  }
  // Ranked placement, whatever the phase: with ranking off nothing waits for
  // it and it does nothing; with it on, the copies are made here and not
  // inside the first timed execute.
  const int64_t apply0 = now_us();
  {
    ::executorch::runtime::BackendOptions<1> apply;
    apply.set_option("weight_tier_apply", true);
    (void)::executorch::runtime::set_option("Esp32Backend", apply.view());
  }
  const int64_t apply_us = now_us() - apply0;
  uint32_t copies = 0, hits = 0;
  size_t resident = 0;
  esp32_backend_weight_cache_stats(&copies, &hits, &resident);
  printf("BENCH_PHASE phase=%s load_us=%lld apply_us=%lld weight_copies=%u "
         "weight_resident=%u method_arena_peak=%u\n",
         phase, (long long)load_us, (long long)apply_us, (unsigned)copies,
         (unsigned)resident, (unsigned)arena.peak());
  if (copies != 0) esp32_backend_report_weight_tiers();

  // The checked run, which also warms the caches.
  uint32_t in_bits = 0, bits = 0, numel = 0;
  double checksum = 0.0;
  if (!fill_input(*method, h, &in_bits)) return false;
  Error e = method->execute();
  if (e != Error::Ok) {
    printf("BENCH_ERROR phase=%s execute failed: %d\n", phase, (int)e);
    return false;
  }
  if (!read_outputs(*method, &bits, &checksum, &numel)) {
    printf("BENCH_ERROR phase=%s outputs are not float32 tensors\n", phase);
    return false;
  }
  const bool exact = bits == h.output_bits && numel == h.output_numel;  // the host runner's bits
  printf("BENCH_CHECK phase=%s result=%s input_bits=%08lx/%08lx output_bits=%08lx/%08lx "
         "checksum=%.6f/%.6f outputs=%u temp_peak=%u large_temp_calls=%u\n",
         phase, exact ? "EXACT" : (in_bits != h.input_bits ? "INPUT" : "DIFF"),
         (unsigned long)in_bits, (unsigned long)h.input_bits, (unsigned long)bits,
         (unsigned long)h.output_bits, checksum, h.output_checksum, (unsigned)numel,
         (unsigned)temp.peak(), (unsigned)temp.large_calls());
  let_idle_run();

  // The timed executes (counts checked again: a caller may set them after
  // parse, as esp32_bench_host's options do).
  if (!runs_in_range(h)) return false;
  const uint32_t runs = h.timed_runs;
  Owned times_owned;
  const char* times_tier = nullptr;
  times_owned.p = alloc_internal_first(sizeof(int64_t) * (runs ? runs : 1), &times_tier);
  int64_t* times = reinterpret_cast<int64_t*>(times_owned.p);
  if (times == nullptr) return false;
  uint32_t ok = 0;
  int64_t total = 0;
  for (uint32_t r = 0; r < runs; ++r) {
    if (!fill_input(*method, h, nullptr)) return false;
    const int64_t t0 = now_us();
    e = method->execute();
    times[r] = now_us() - t0;
    if (e != Error::Ok) {
      printf("BENCH_ERROR phase=%s timed execute %u failed: %d\n", phase, (unsigned)r, (int)e);
      return false;
    }
    total += times[r];
    uint32_t rbits = 0, rn = 0;
    double rsum = 0.0;
    ok += read_outputs(*method, &rbits, &rsum, &rn) && rbits == bits;  // every timed run checked
    printf("BENCH_RUN phase=%s run=%u us=%lld\n", phase, (unsigned)r, (long long)times[r]);
    let_idle_run();
  }
  if (runs > 0) {
    std::sort(times, times + runs);
    const double median = runs % 2 ? (double)times[runs / 2]
                                   : 0.5 * (double)(times[runs / 2 - 1] + times[runs / 2]);
    printf("BENCH_TIME phase=%s runs=%u min_us=%lld median_us=%.1f mean_us=%.1f max_us=%lld "
           "same_bits=%u/%u\n",
           phase, (unsigned)runs, (long long)times[0], median, (double)total / runs,
           (long long)times[runs - 1], (unsigned)ok, (unsigned)runs);
  }

  // The split: one instruction at a time.
  const uint32_t steps = h.stepped_runs;
  memset(step_us, 0, sizeof(int64_t) * h.instructions);
  uint32_t step_ok = 0;
  for (uint32_t r = 0; r < steps; ++r) {
    if (!fill_input(*method, h, nullptr)) return false;
    uint32_t i = 0;
    for (;;) {
      const int64_t t0 = now_us();
      e = method->step();
      const int64_t dt = now_us() - t0;
      if (e == Error::EndOfMethod) break;
      if (e != Error::Ok) {
        printf("BENCH_ERROR phase=%s step %u failed: %d\n", phase, (unsigned)i, (int)e);
        return false;
      }
      if (i < h.instructions) step_us[i] += dt;
      ++i;
    }
    if (method->reset_execution() != Error::Ok || i != h.instructions) {
      printf("BENCH_ERROR phase=%s stepped %u instructions, the image labels %u\n", phase,
             (unsigned)i, (unsigned)h.instructions);
      return false;
    }
    uint32_t rbits = 0, rn = 0;
    double rsum = 0.0;
    step_ok += read_outputs(*method, &rbits, &rsum, &rn) && rbits == bits;
    let_idle_run();
  }
  if (steps > 0) {
    // Per run: each instruction, then the sum by kind (D a delegate, Q a
    // portable quantize or dequantize, P any other portable operator, M a
    // move, free or jump).
    double kinds[4] = {0, 0, 0, 0};
    static const char kKinds[] = "DQPM";
    for (uint32_t i = 0; i < h.instructions; ++i) {
      const char* label = image.label(i);
      const double us = (double)step_us[i] / steps;
      const char* k = strchr(kKinds, label[0]);
      kinds[k != nullptr && label[0] != 0 ? k - kKinds : 3] += us;
      printf("BENCH_INSTR phase=%s i=%u us=%.1f %s\n", phase, (unsigned)i, us, label);
    }
    const double sum = kinds[0] + kinds[1] + kinds[2] + kinds[3];
    printf("BENCH_SPLIT phase=%s stepped_runs=%u delegate_us=%.1f qdq_us=%.1f "
           "portable_us=%.1f other_us=%.1f total_us=%.1f same_bits=%u/%u\n",
           phase, (unsigned)steps, kinds[0], kinds[1], kinds[2], kinds[3], sum,
           (unsigned)step_ok, (unsigned)steps);
  }
  return true;
}

// The whole bench: allocate, then measure under each weight tier. `weights`
// is the weights partition (or file) as a data loader, `weights_base` the
// same bytes mapped.
inline void run(const Image& image, ::executorch::runtime::DataLoader* weights_loader,
                const uint8_t* weights_base, size_t weights_size) {
  const Header& h = image.h;
  printf("BENCH_MODEL name=%s instructions=%u program=%u weights=%u macs=%llu "
         "macs_delegated=%llu timed_runs=%u stepped_runs=%u\n",
         h.name, (unsigned)h.instructions, (unsigned)h.program_bytes,
         (unsigned)h.weights_bytes, (unsigned long long)h.macs,
         (unsigned long long)h.macs_delegated, (unsigned)h.timed_runs,
         (unsigned)h.stepped_runs);
  // The weights must be this model's: a partition left from another model
  // would load and run, wrongly.
  if (h.weights_bytes > weights_size ||
      fnv(weights_base, h.weights_bytes) != h.weights_bits) {
    printf("BENCH_ERROR the weights partition does not hold this model's %u B of "
           "weights (FNV-1a %08lx): flash its bench_weights.ptd\n",
           (unsigned)h.weights_bytes, (unsigned long)h.weights_bits);
    return;
  }

  ::executorch::extension::BufferDataLoader loader(image.program(), h.program_bytes);
  auto program = Program::load(&loader);
  if (!program.ok()) {
    printf("BENCH_ERROR Program::load failed: %d\n", (int)program.error());
    return;
  }
  auto weights = ::executorch::extension::FlatTensorDataMap::load(weights_loader);
  if (!weights.ok()) {
    printf("BENCH_ERROR FlatTensorDataMap::load failed: %d\n", (int)weights.error());
    return;
  }
  auto meta = program->method_meta(*program->get_method_name(0));
  if (!meta.ok() || meta->num_memory_planned_buffers() > kMaxPlanned) {
    printf("BENCH_ERROR method metadata\n");
    return;
  }

  printf("BENCH_HEAP before internal_free=%u internal_largest=%u psram_free=%u\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_largest_free_block(kInternal),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  Buffers b;
  Owned fast, large, arena, planned[kMaxPlanned];
  // The largest delegate scratch: internal if one block holds it, else the
  // 32 KB floor internal and the rest from PSRAM.
  const size_t want_temp = ((size_t)std::max<uint32_t>(h.temp_bytes, 1) + 15u) & ~(size_t)15u;
  fast.p = alloc(std::max<size_t>(want_temp, kFastTempFloor), kInternal);
  b.fast_size = fast.p != nullptr ? std::max<size_t>(want_temp, kFastTempFloor) : 0;
  if (fast.p == nullptr) {
    fast.p = alloc(kFastTempFloor, kInternal);
    b.fast_size = fast.p != nullptr ? kFastTempFloor : 0;
  }
  b.fast = fast.p;
  if (b.fast_size < want_temp) {
    large.p = alloc(want_temp, MALLOC_CAP_SPIRAM);
    b.large = large.p;
    b.large_size = large.p != nullptr ? want_temp : 0;
  }
  printf("BENCH_MEM temp: %u B scratch wanted, %u B internal, %u B PSRAM\n",
         (unsigned)want_temp, (unsigned)b.fast_size, (unsigned)b.large_size);
  if (b.fast_size + b.large_size < want_temp) {
    printf("BENCH_ERROR no %u B for the delegate scratch\n", (unsigned)want_temp);
    return;
  }
  b.n_planned = meta->num_memory_planned_buffers();
  for (size_t i = 0; i < b.n_planned; ++i) {
    const size_t want = (size_t)*meta->memory_planned_buffer_size(i);
    const char* tier = nullptr;
    planned[i].p = alloc_internal_first((want + 15u) & ~(size_t)15u, &tier);
    printf("BENCH_MEM planned %u: %u B (%s)\n", (unsigned)i, (unsigned)want, tier);
    if (planned[i].p == nullptr) {
      printf("BENCH_ERROR planned buffer %u: no %u B\n", (unsigned)i, (unsigned)want);
      return;
    }
    b.spans[i] = Span<uint8_t>(planned[i].p, want);
  }
  // The method arena holds the program's values, tensor metadata and the
  // delegates' init state. Its size is measured, not guessed: the method is
  // loaded once (nothing executed) into a large PSRAM arena under the flash
  // measurement's options, and the arena is then that peak plus a quarter,
  // internal when a block holds it, else PSRAM. A guessed internal arena
  // squeezed into what the planned buffers left could not hold
  // MobileNetV2-96's twelve delegates (load_method error 33).
  size_t arena_need = 0;
  {
    ::executorch::runtime::BackendOptions<3> flash;
    flash.set_option("weight_tier_budget", 0);
    flash.set_option("weight_cache", false);
    flash.set_option("weight_tier_rank", false);
    (void)::executorch::runtime::set_option("Esp32Backend", flash.view());
    constexpr size_t kProbeBytes = 512 * 1024;
    Owned probe;
    probe.p = alloc(kProbeBytes, MALLOC_CAP_SPIRAM);
    if (probe.p != nullptr) {
      Arena probe_arena(kProbeBytes, probe.p);
      Temp probe_temp(b.fast_size, b.fast, b.large_size, b.large);
      HierarchicalAllocator probe_planned({b.spans, b.n_planned});
      MemoryManager probe_memory(&probe_arena, &probe_planned, &probe_temp);
      auto probe_method = program->load_method(*program->get_method_name(0),
                                               &probe_memory, nullptr,
                                               &weights.get());
      if (probe_method.ok()) {
        arena_need = probe_arena.peak();
      }
    }
  }
  const char* arena_tier = "internal";
  if (arena_need > 0) {
    b.arena_size = (arena_need + arena_need / 4 + 4096 + 15u) & ~(size_t)15u;
    arena.p = alloc(b.arena_size, kInternal);
    if (arena.p == nullptr) {
      arena.p = alloc(b.arena_size, MALLOC_CAP_SPIRAM);
      arena_tier = "PSRAM";
    }
  } else {
    // No measurement (no PSRAM for the probe, or the probe load failed, which
    // the measurement will report): 64 KB, internal first.
    b.arena_size = 64 * 1024;
    arena.p = alloc(b.arena_size, kInternal);
    if (arena.p == nullptr) {
      arena.p = alloc(b.arena_size, MALLOC_CAP_SPIRAM);
      arena_tier = "PSRAM";
    }
  }
  b.arena = arena.p;
  printf("BENCH_MEM method arena: %u B (%s), the load measured %u B\n",
         (unsigned)b.arena_size, arena.p != nullptr ? arena_tier : "none",
         (unsigned)arena_need);
  if (arena.p == nullptr) {
    printf("BENCH_ERROR no method arena\n");
    return;
  }
  printf("BENCH_HEAP after internal_free=%u internal_largest=%u psram_free=%u\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_largest_free_block(kInternal),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  Owned steps_owned;
  const char* steps_tier = nullptr;
  steps_owned.p = alloc_internal_first(sizeof(int64_t) * h.instructions, &steps_tier);
  if (steps_owned.p == nullptr) {
    printf("BENCH_ERROR no room for the instruction timings\n");
    return;
  }
  int64_t* step_us = reinterpret_cast<int64_t*>(steps_owned.p);

  // 1. Every weight from the flash mapping.
  {
    ::executorch::runtime::BackendOptions<3> flash;
    flash.set_option("weight_tier_budget", 0);
    flash.set_option("weight_cache", false);
    flash.set_option("weight_tier_rank", false);
    if (::executorch::runtime::set_option("Esp32Backend", flash.view()) != Error::Ok) {
      printf("BENCH_ERROR the backend refused the weight-tier options\n");
      return;
    }
  }
  if (!measure("flash", image, *program, &weights.get(), b, step_us)) return;
  report_phase_timing("flash");

  // 2. Every weight PSRAM can hold, placed ranked by the apply call.
  if (heap_caps_get_free_size(MALLOC_CAP_SPIRAM) == 0) {
    printf("BENCH_PHASE phase=psram skipped: no PSRAM\n");
  } else {
    ::executorch::runtime::BackendOptions<3> psram;
    psram.set_option("weight_tier_budget", kPsramBudget);
    psram.set_option("weight_cache", false);
    psram.set_option("weight_tier_rank", true);
    if (::executorch::runtime::set_option("Esp32Backend", psram.view()) != Error::Ok) {
      printf("BENCH_ERROR the backend refused the weight-tier options\n");
      return;
    }
    printf("BENCH_TIER budget=%d (-1: every weight PSRAM holds), ranked, cache off\n",
           kPsramBudget);
    if (!measure("psram", image, *program, &weights.get(), b, step_us)) return;
    report_phase_timing("psram");
  }
  // Back to the defaults, as an application would leave them.
  {
    ::executorch::runtime::BackendOptions<3> reset;
    reset.set_option("weight_tier_budget", 0);
    reset.set_option("weight_cache", true);
    reset.set_option("weight_tier_rank", false);
    (void)::executorch::runtime::set_option("Esp32Backend", reset.view());
  }
  printf("BENCH_DONE name=%s\n", h.name);
}

}  // namespace bench

#if !defined(ESP32_HOST_BUILD)
// The image from partition 0x43, its named data from 0x44.
static void run_bench(void) {
  auto image = esp32s3::EspPartitionDataLoader::from_subtype(0x43);
  auto weights = esp32s3::EspPartitionDataLoader::from_subtype(0x44);
  if (!image.ok() || !weights.ok()) {
    printf("BENCH_ERROR partitions 0x43/0x44 absent\n");
    return;
  }
  bench::Image parsed;
  if (!bench::parse(static_cast<const uint8_t*>(image->mapped_base()), *image->size(),
                    &parsed)) {
    return;
  }
#if defined(ET_BENCH_PROFILE)
  printf("BENCH_PROFILE %s; data cache %u KB, %u B lines; instruction cache %u KB; "
         "CPU %u MHz\n",
         ET_BENCH_PROFILE, (unsigned)(CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024),
         (unsigned)CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE,
         (unsigned)(CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE / 1024),
         (unsigned)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
#endif
  // One execute of a 224-pixel model can outlast the task watchdog's 5 s, and
  // its report would land inside a timed run: the bench owns both cores until
  // it is done, so the watchdog goes.
  printf("BENCH_WDT task watchdog off: %s\n", esp_task_wdt_deinit() == ESP_OK ? "yes" : "NO");
#if defined(ET_BENCH_DUAL_CORE) && ET_BENCH_DUAL_CORE
  {
    // Both cores split every matmul's columns (the backend's min_latency).
    executorch::runtime::BackendOptions<1> on;
    on.set_option("min_latency", true);
    const Error e = executorch::runtime::set_option("Esp32Backend", on.view());
    printf("BENCH_CORES dual-core min_latency: %s\n", e == Error::Ok ? "ON" : "REFUSED");
  }
#endif
#if defined(ET_BENCH_MICRO) && ET_BENCH_MICRO
  // The backend's micro-benchmarks (the conv path's kernels, its epilogues, a
  // fork round trip, a layer on one core and two), before any model loads.
  esp32_backend_microbench();
#endif
  bench::run(parsed, &weights.get(), static_cast<const uint8_t*>(weights->mapped_base()),
             *weights->size());
}
#endif
