/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Host numerics runner. Inputs and outputs are raw tensor bytes; the PTE
// supplies dtype, sizes, and dim order. No Torch runtime or serial device.
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/platform_memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "esp32_asym.h"
#include "esp32_conv.h"
#include "esp32_kernels.h"
#include "esp32_platform.h"
#include "esp32_weight_tiers.h"
#include "espressif_ops.h"
#include "divsqrt.h"
#include "newlib_expf.h"
#include "elementwise.h"
#include "qdq_pool.h"
#include "quant_lut.h"
#include "quant_lut16.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::extension::FlatTensorDataMap;
using executorch::extension::MmapDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Program;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;
using executorch::runtime::internal::PlatformMemoryAllocator;

namespace torch::executor {
Error register_all_kernels();
}
extern "C" void esp32_backend_set_force_scalar(int);
// The static int8 convolution path (esp32_conv.h): 0 sends every conv to the
// general loops, which compute the same bits (--no-conv-fast).
extern "C" void esp32_backend_set_conv_fast(int);
// The bench firmware's micro-benchmarks (BENCH_MICRO lines): on the host a
// smoke test of their code, its "cycles" microseconds x 240.
extern "C" void esp32_backend_microbench(void);
extern "C" void esp32_backend_set_split(int);
extern "C" void esp32_backend_report();
extern "C" void esp32_backend_release_weight_cache();
extern "C" void esp32_backend_host_cap_tier_slots(unsigned slots);

namespace {
void check(Error status, const std::string& operation) {
  if (status != Error::Ok) {
    throw std::runtime_error(operation + ": ExecuTorch error " +
                             std::to_string(static_cast<unsigned>(status)));
  }
}

class AlignedBytes {
 public:
  explicit AlignedBytes(size_t size) : size_(size) {
    if (posix_memalign(&data_, 64, std::max(size, size_t{1})) != 0) {
      throw std::bad_alloc();
    }
    std::memset(data_, 0, size);
  }
  ~AlignedBytes() { std::free(data_); }
  AlignedBytes(const AlignedBytes&) = delete;
  AlignedBytes& operator=(const AlignedBytes&) = delete;
  void* data() const { return data_; }
  size_t size() const { return size_; }
 private:
  void* data_ = nullptr;
  size_t size_;
};

void read_input(const std::string& path, AlignedBytes& bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open input: " + path);
  if (stream.tellg() != static_cast<std::streamoff>(bytes.size())) {
    throw std::runtime_error("input byte count does not match PTE (expected " +
                             std::to_string(bytes.size()) + "): " + path);
  }
  stream.seekg(0);
  if (bytes.size()) {
    stream.read(static_cast<char*>(bytes.data()), bytes.size());
    if (!stream) throw std::runtime_error("cannot read input: " + path);
  }
}

struct InputTensor {
  AlignedBytes bytes;
  std::vector<TensorImpl::SizesType> sizes;
  std::vector<TensorImpl::DimOrderType> dim_order;
  std::vector<TensorImpl::StridesType> strides;
  TensorImpl impl;

  explicit InputTensor(const TensorInfo& info)
      : bytes(info.nbytes()),
        sizes(info.sizes().begin(), info.sizes().end()),
        dim_order(info.dim_order().begin(), info.dim_order().end()),
        strides(sizes.size()),
        impl(info.scalar_type(), sizes.size(), sizes.data(), bytes.data(),
             dim_order.data(), strides.data()) {
    int64_t stride = 1;
    for (size_t axis = dim_order.size(); axis > 0; --axis) {
      const auto dim = dim_order[axis - 1];
      if (dim >= sizes.size() || stride > INT32_MAX) {
        throw std::runtime_error("invalid or oversized input dim order/stride");
      }
      strides[dim] = static_cast<TensorImpl::StridesType>(stride);
      stride *= sizes[dim];
    }
  }
};

struct Step {
  std::string method;
  std::vector<std::string> inputs, outputs;
  // A directive in place of a method: "set KEY VALUE" sets one backend
  // option, as an application does between loads; "unload" destroys every
  // loaded method.
  std::string set_key, set_value;
  bool unload = false;
  bool directive() const { return unload || !set_key.empty(); }
};

struct Options {
  std::string pte, ptd, method = "forward", script;
  std::vector<std::string> inputs, outputs;
  bool scalar = false, split = false, single_core = false, no_conv_fast = false,
       no_qdq_fast = false, no_ew_fast = false;
  // The weight tiers (the backend's weight_tier_* options): unset, the
  // backend's defaults; budget -1 is auto.
  std::optional<int> tier_budget, tier_reserve;
  bool tier_tables = false, no_weight_cache = false;
  bool tier_rank = false, tier_apply = false;
  std::optional<size_t> psram_bytes;
  // Load every method the script names up front and keep them all loaded,
  // instead of one at a time.
  bool keep_methods = false;
  // Planned buffers start this many bytes past a 64-byte boundary, as an
  // integrator's malloc may give them (a heap_caps_malloc'd buffer is only
  // 4-byte aligned): the backend must align what it needs itself.
  size_t planned_offset = 0;
  // Every planned buffer counts as PSRAM to the backend (esp32_host_mark_psram),
  // so a call whose outputs or scratch land there runs on one core, as on the
  // board.
  bool psram_planned = false;
  // Rounds of ranked placement on a second thread while the script's first
  // method executes (run_placement_stress); 0 = a plain run.
  unsigned placement_stress = 0;
  // Ranked placements while the script's first method stays loaded and idle
  // (run_placement_idle); 0 = a plain run.
  unsigned placement_idle = 0;
  // Handles that get a private placement mark (the backend's 254 unless
  // set); the rest share slot 255 (esp32_backend_host_cap_tier_slots).
  std::optional<unsigned> tier_slots;
};

void usage() {
  std::printf("Usage: esp32_host_runner --pte model.pte [--ptd weights.ptd]\n"
              "       [--method forward] --input in0.bin [--input in1.bin ...]\n"
              "       --output out0.bin [--output out1.bin ...]\n"
              "       [--force-scalar] [--min-latency | --single-core] [--planned-offset N]\n"
              "       [--no-conv-fast] [--no-qdq-fast] [--no-ew-fast] [weight tiers]\n"
              "   or: esp32_host_runner --pte model.pte [--ptd weights.ptd]\n"
              "       --script steps.txt [--force-scalar] [--min-latency]\n"
              "       [--keep-methods] [weight tiers]\n"
              "   or: esp32_host_runner --pte model.pte [--ptd weights.ptd]\n"
              "       --script steps.txt --placement-stress ROUNDS [--tier-slots N]\n"
              "       (two lines: the first method runs on one thread while the\n"
              "       second is loaded ranked and placed on another, each round)\n"
              "   or: esp32_host_runner --pte model.pte [--ptd weights.ptd]\n"
              "       --script steps.txt --placement-idle COUNT [--tier-slots N]\n"
              "       (two lines: the first method stays loaded while the second\n"
              "       is loaded ranked and placed COUNT times, then runs again)\n"
              "   or: esp32_host_runner --selftest  (the op library's device checks)\n"
              "   or: esp32_host_runner --qdq-sweep quick|full  (qdq_pool.h's fast\n"
              "       paths against the portable kernels; full: every float32)\n"
              "   or: esp32_host_runner --microbench  (the bench image's BENCH_MICRO)\n"
              "   or: esp32_host_runner --asym-conformance CASES OUT\n"
              "       (blob v24's affine column math; test_asym_conformance.py)\n"
              "Files contain raw tensor bytes in the PTE's dtype and dim order.\n"
              "Each script line is METHOD IN... -> OUT... ('#' starts a comment),\n"
              "or a directive: 'set KEY VALUE' sets one of the backend's options\n"
              "(VALUE an integer, auto, true or false) and 'unload' destroys every\n"
              "loaded method; each prints the cache's resident bytes after it.\n"
              "Lines run in order in one process and every method uses the same\n"
              "planned buffers, so state such as a shared KV cache carries over.\n"
              "A method is loaded when a line names it and destroyed when a line\n"
              "names another; --keep-methods loads every method the script names\n"
              "before the first line instead, and keeps them all loaded.\n"
              "--planned-offset N starts every planned buffer N bytes past a\n"
              "64-byte boundary, as an integrator's malloc may.\n"
              "Weight tiers (the backend's weight_tier_* options; printed after\n"
              "the run as 'esp32 tier' lines):\n"
              "  --weight-tier-budget N|auto  bytes of PSRAM for weight copies\n"
              "  --weight-tier-reserve N      bytes auto leaves free\n"
              "  --weight-tier-tables         copy tables and selected-row heads\n"
              "  --weight-tier-rank           ranked placement instead of load order\n"
              "  --weight-tier-apply          apply it after loading (else each load's\n"
              "                               first execute does)\n"
              "  --no-weight-cache            drop copies with their last method\n"
              "  --psram-bytes N              the PSRAM this host pretends to have\n"
              "                               (default unlimited; 0 = none, as QEMU)\n"
              "  --psram-planned              treat every planned buffer as PSRAM (a\n"
              "                               split writes none of it)\n");
}

// A byte count for an option: decimal, at most `limit`.
unsigned long long parse_bytes(const std::string& key, const char* text,
                               unsigned long long limit) {
  char* end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (*text == '\0' || *text == '-' || *end != '\0' || errno != 0 ||
      value > limit) {
    throw std::runtime_error(key + " takes a byte count up to " +
                             std::to_string(limit) + ", got " + text);
  }
  return value;
}

Options parse(int argc, char** argv) {
  Options options;
  bool method = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--force-scalar") options.scalar = true;
    else if (key == "--no-conv-fast") options.no_conv_fast = true;
    else if (key == "--no-qdq-fast") options.no_qdq_fast = true;
    else if (key == "--no-ew-fast") options.no_ew_fast = true;
    else if (key == "--min-latency") options.split = true;
    // One core whatever a program's min_latency compile spec asks: the
    // runtime option at 0 (a single-core application's), so a min_latency
    // export's bits can be held to the single-core path's.
    else if (key == "--single-core") options.single_core = true;
    else if (key == "--weight-tier-tables") options.tier_tables = true;
    else if (key == "--weight-tier-rank") options.tier_rank = true;
    else if (key == "--weight-tier-apply") options.tier_apply = true;
    else if (key == "--no-weight-cache") options.no_weight_cache = true;
    else if (key == "--keep-methods") options.keep_methods = true;
    else if (key == "--psram-planned") options.psram_planned = true;
    else if (key == "--planned-offset") {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      options.planned_offset = std::stoul(argv[i]);
      if (options.planned_offset >= 64) {
        throw std::runtime_error("--planned-offset must be below 64");
      }
    }
    else if (key == "--placement-stress" || key == "--placement-idle") {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      const unsigned count = static_cast<unsigned>(parse_bytes(key, argv[i], 100000));
      if (count == 0) throw std::runtime_error(key + " takes at least one round");
      (key == "--placement-stress" ? options.placement_stress : options.placement_idle) = count;
    }
    else if (key == "--tier-slots") {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      options.tier_slots = static_cast<unsigned>(parse_bytes(key, argv[i], 254));
    }
    else if (key == "--weight-tier-budget" || key == "--weight-tier-reserve" ||
             key == "--psram-bytes") {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      constexpr unsigned long long kIntMax = 0x7FFFFFFF;
      if (key == "--weight-tier-budget") {
        options.tier_budget = std::string(argv[i]) == "auto"
            ? -1 : static_cast<int>(parse_bytes(key, argv[i], kIntMax));
      } else if (key == "--weight-tier-reserve") {
        options.tier_reserve = static_cast<int>(parse_bytes(key, argv[i], kIntMax));
      } else {
        options.psram_bytes = static_cast<size_t>(parse_bytes(key, argv[i], SIZE_MAX));
      }
    } else if (key == "--pte" || key == "--ptd" || key == "--method" ||
             key == "--input" || key == "--output" || key == "--script") {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      if (key == "--pte") options.pte = argv[i];
      else if (key == "--ptd") options.ptd = argv[i];
      else if (key == "--method") {
        options.method = argv[i];
        method = true;
      } else if (key == "--input") options.inputs.emplace_back(argv[i]);
      else if (key == "--output") options.outputs.emplace_back(argv[i]);
      else options.script = argv[i];
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (options.pte.empty()) throw std::runtime_error("--pte is required");
  if (!options.script.empty() &&
      (method || !options.inputs.empty() || !options.outputs.empty())) {
    throw std::runtime_error("--script replaces --method/--input/--output");
  }
  if ((options.placement_stress != 0 || options.placement_idle != 0) &&
      (options.script.empty() || options.keep_methods || options.tier_budget ||
       options.tier_rank || options.tier_apply || options.no_weight_cache ||
       (options.placement_stress != 0 && options.placement_idle != 0))) {
    throw std::runtime_error("--placement-stress and --placement-idle take a "
                             "--script, one of them, and set the weight tiers "
                             "themselves");
  }
  return options;
}

std::vector<Step> read_script(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open script: " + path);
  std::vector<Step> steps;
  std::string line;
  for (size_t number = 1; std::getline(stream, line); ++number) {
    std::istringstream words(line.substr(0, line.find('#')));
    Step step;
    if (!(words >> step.method)) continue;
    const std::string where = path + ":" + std::to_string(number);
    if (step.method == "set" || step.method == "unload") {
      std::string extra;
      if (step.method == "set" && !(words >> step.set_key >> step.set_value)) {
        throw std::runtime_error(where + ": expected set KEY VALUE");
      }
      if (words >> extra) throw std::runtime_error(where + ": unexpected " + extra);
      step.unload = step.method == "unload";
      step.method.clear();
      steps.push_back(std::move(step));
      continue;
    }
    bool arrow = false;
    for (std::string word; words >> word;) {
      if (word == "->" && arrow) throw std::runtime_error(where + ": more than one ->");
      if (word == "->") arrow = true;
      else (arrow ? step.outputs : step.inputs).push_back(word);
    }
    if (!arrow) throw std::runtime_error(where + ": expected METHOD IN... -> OUT...");
    steps.push_back(std::move(step));
  }
  if (steps.empty()) throw std::runtime_error("script has no steps: " + path);
  return steps;
}

// One loaded method with its own allocators. Members are destroyed in reverse
// order, so the Method goes before the allocators and buffers it uses.
struct LoadedMethod {
  std::string name;
  PlatformMemoryAllocator method_allocator, temp_allocator;
  MemoryManager memory_manager;
  std::vector<std::unique_ptr<AlignedBytes>> outputs;
  std::optional<Method> method;

  LoadedMethod(const Program& program, const std::string& method_name,
               HierarchicalAllocator* planned, const NamedDataMap* ptd)
      : name(method_name),
        memory_manager(&method_allocator, planned, &temp_allocator) {
    auto loaded = program.load_method(name.c_str(), &memory_manager, nullptr, ptd);
    check(loaded.error(), "load method " + name);
    method.emplace(std::move(*loaded));
    const auto meta = method->method_meta();
    for (size_t i = 0; i < meta.num_outputs(); ++i) {
      auto info = meta.output_tensor_meta(i);
      check(info.error(), "read tensor output metadata (only tensor I/O is supported)");
      if (!info->is_memory_planned()) {
        outputs.push_back(std::make_unique<AlignedBytes>(info->nbytes()));
        check(method->set_output_data_ptr(outputs.back()->data(), outputs.back()->size(), i),
              "set output buffer");
      }
    }
  }
};

void run_step(LoadedMethod& loaded, const Step& step) {
  Method& method = *loaded.method;
  const auto meta = method.method_meta();
  // Kept until the outputs are written. A non-planned input's data pointer is
  // replaced by the next step's set_input before the method runs again.
  std::vector<std::unique_ptr<InputTensor>> inputs;
  for (size_t i = 0; i < step.inputs.size(); ++i) {
    auto info = meta.input_tensor_meta(i);
    check(info.error(), "read tensor input metadata (only tensor I/O is supported)");
    auto input = std::make_unique<InputTensor>(*info);
    read_input(step.inputs[i], input->bytes);
    check(method.set_input(EValue(Tensor(&input->impl)), i), "set input");
    inputs.push_back(std::move(input));
  }
  check(method.execute(), "execute " + step.method);
  for (size_t i = 0; i < step.outputs.size(); ++i) {
    const Tensor& output = method.get_output(i).toTensor();
    std::ofstream stream(step.outputs[i], std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open output: " + step.outputs[i]);
    if (output.nbytes()) {
      stream.write(static_cast<const char*>(output.const_data_ptr()), output.nbytes());
    }
    stream.close();
    if (!stream) throw std::runtime_error("cannot write output: " + step.outputs[i]);
    std::printf("output %zu: dtype=%d shape=[", i, static_cast<int>(output.scalar_type()));
    for (ssize_t d = 0; d < output.dim(); ++d) {
      std::printf("%s%d", d ? "," : "", static_cast<int>(output.size(d)));
    }
    std::printf("] bytes=%zu file=%s\n", output.nbytes(), step.outputs[i].c_str());
  }
}

// ---- --placement-stress ----------------------------------------------------
// A ranked placement while another method executes, on two threads, round
// after round. The script's first line is loaded on this thread under a
// one-byte budget, which registers its weights with the cache and leaves
// them on flash, and executed again and again. Meanwhile a second thread
// loads the second line's method ranked under an unlimited budget and
// applies the placement, which copies the first method's weights as well as
// its own, then executes it once. The first method executes three more times
// after the placement is out; then, every delegate having run since the
// placement, each op whose weight has a copy must read it
// (esp32_backend_weight_tier_copy_reads, counted under the backend's lock).
// Then both are destroyed and the unused copies dropped (weight_cache off,
// then on), so the next round places them again. Every execute must give its
// method's first bits, which are written to the script's output files. Each
// method gets planned buffers of its own, since the two run at once. The
// second method may share the first one's weights, so that destroying it
// leaves copies the first still reads.

// One method's planned buffers.
class PlannedMemory {
 public:
  PlannedMemory(const Program& program, const std::string& method) {
    auto meta = program.method_meta(method.c_str());
    check(meta.error(), "read method metadata for " + method);
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
      auto size = meta->memory_planned_buffer_size(i);
      check(size.error(), "read planned buffer size");
      if (*size < 0) throw std::runtime_error("negative planned buffer size");
      buffers_.push_back(std::make_unique<AlignedBytes>(static_cast<size_t>(*size)));
      spans_.emplace_back(static_cast<uint8_t*>(buffers_.back()->data()),
                          static_cast<size_t>(*size));
    }
    allocator_.emplace(Span<Span<uint8_t>>(spans_.data(), spans_.size()));
  }
  HierarchicalAllocator* get() { return &*allocator_; }

 private:
  std::vector<std::unique_ptr<AlignedBytes>> buffers_;
  std::vector<Span<uint8_t>> spans_;
  std::optional<HierarchicalAllocator> allocator_;
};

// Two threads meeting at the end of each phase; once either fails, every
// wait returns false at once.
class Rendezvous {
 public:
  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (failed_) return false;
    const unsigned phase = phase_;
    if (++arrived_ == 2) {
      arrived_ = 0;
      ++phase_;
      changed_.notify_all();
      return true;
    }
    changed_.wait(lock, [&] { return phase_ != phase || failed_; });
    return !failed_;
  }
  void fail() {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
    changed_.notify_all();
  }
  bool failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  unsigned arrived_ = 0, phase_ = 0;
  bool failed_ = false;
};

using Outputs = std::vector<std::vector<uint8_t>>;

std::vector<std::unique_ptr<InputTensor>> read_inputs(Method& method,
                                                      const Step& step) {
  const auto meta = method.method_meta();
  std::vector<std::unique_ptr<InputTensor>> inputs;
  for (size_t i = 0; i < step.inputs.size(); ++i) {
    auto info = meta.input_tensor_meta(i);
    check(info.error(), "read tensor input metadata (only tensor I/O is supported)");
    inputs.push_back(std::make_unique<InputTensor>(*info));
    read_input(step.inputs[i], inputs.back()->bytes);
  }
  return inputs;
}

// One execute on the given inputs (set again each time: a plan may reuse an
// input's bytes once it is consumed) and the bytes of every output.
Outputs execute_once(Method& method,
                     const std::vector<std::unique_ptr<InputTensor>>& inputs,
                     const std::string& name) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    check(method.set_input(EValue(Tensor(&inputs[i]->impl)), i), "set input");
  }
  check(method.execute(), "execute " + name);
  Outputs outputs;
  for (size_t i = 0; i < method.outputs_size(); ++i) {
    const Tensor& output = method.get_output(i).toTensor();
    const auto* bytes = static_cast<const uint8_t*>(output.const_data_ptr());
    outputs.emplace_back(bytes, bytes + output.nbytes());
  }
  return outputs;
}

void write_outputs(const Step& step, const Outputs& outputs) {
  if (outputs.size() != step.outputs.size()) {
    throw std::runtime_error("method " + step.method + " has " +
                             std::to_string(outputs.size()) + " output(s)");
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::ofstream stream(step.outputs[i], std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(outputs[i].data()),
                 static_cast<std::streamsize>(outputs[i].size()));
    if (!stream) throw std::runtime_error("cannot write output: " + step.outputs[i]);
  }
}

template <size_t N>
void set_backend_options(executorch::runtime::BackendOptions<N>& options,
                         const char* what) {
  check(executorch::runtime::set_option("Esp32Backend", options.view()), what);
}

// One option of a script's set directive, under its literal name: true or
// false as a bool, auto as -1 (weight_tier_budget's auto), else a byte count.
template <size_t N>
void set_one_option(const char (&key)[N], const std::string& value) {
  executorch::runtime::BackendOptions<1> one;
  if (value == "true" || value == "false") {
    one.set_option(key, value == "true");
  } else {
    one.set_option(key, value == "auto" ? -1 : static_cast<int>(parse_bytes(
                                                   key, value.c_str(), 0x7FFFFFFF)));
  }
  set_backend_options(one, (std::string("set ") + key).c_str());
}

void set_script_option(const std::string& key, const std::string& value) {
  if (key == "weight_tier_budget") set_one_option("weight_tier_budget", value);
  else if (key == "weight_tier_reserve") set_one_option("weight_tier_reserve", value);
  else if (key == "weight_tier_tables") set_one_option("weight_tier_tables", value);
  else if (key == "weight_tier_rank") set_one_option("weight_tier_rank", value);
  else if (key == "weight_tier_apply") set_one_option("weight_tier_apply", value);
  else if (key == "weight_cache") set_one_option("weight_cache", value);
  else throw std::runtime_error("set: no backend option " + key);
}

void set_stress_tiers(int budget, bool rank) {
  executorch::runtime::BackendOptions<2> tiers;
  tiers.set_option("weight_tier_budget", budget);
  tiers.set_option("weight_tier_rank", rank);
  set_backend_options(tiers, "set the weight tiers");
}

int run_placement_stress(const Options& options, const Program& program,
                         const NamedDataMap* ptd, const std::vector<Step>& steps) {
  if (steps.size() != 2 || steps[0].directive() || steps[1].directive() ||
      steps[0].method == steps[1].method) {
    throw std::runtime_error("--placement-stress takes a script of two lines "
                             "naming two methods");
  }
  const Step& runs = steps[0];
  const Step& places = steps[1];
  const unsigned rounds = options.placement_stress;
  PlannedMemory run_memory(program, runs.method), place_memory(program, places.method);
  Rendezvous meet;
  std::atomic<bool> placed{false};
  Outputs run_want, place_want;
  unsigned run_executes = 0, run_after = 0, run_diffs = 0;
  unsigned place_executes = 0, place_diffs = 0, rounds_read = 0;
  std::string place_error;

  std::thread placer([&] {
    try {
      for (unsigned round = 0; round < rounds; ++round) {
        if (!meet.wait()) return;  // the first method is loaded and running
        set_stress_tiers(0x7FFFFFFF, true);
        {
          LoadedMethod loaded(program, places.method, place_memory.get(), ptd);
          const auto inputs = read_inputs(*loaded.method, places);
          executorch::runtime::BackendOptions<1> apply;
          apply.set_option("weight_tier_apply", true);
          set_backend_options(apply, "apply the weight tiers");
          placed.store(true, std::memory_order_release);
          Outputs got = execute_once(*loaded.method, inputs, places.method);
          ++place_executes;
          if (round == 0) {
            place_want = std::move(got);
          } else if (got != place_want) {
            ++place_diffs;
          }
          if (!meet.wait()) return;  // both have run since the placement
        }
        if (!meet.wait()) return;  // both destroyed
        if (!meet.wait()) return;  // the unused copies dropped
      }
    } catch (const std::exception& error) {
      place_error = error.what();
      meet.fail();
    }
  });
  try {
    for (unsigned round = 0; round < rounds; ++round) {
      set_stress_tiers(1, false);
      {
        LoadedMethod loaded(program, runs.method, run_memory.get(), ptd);
        const auto inputs = read_inputs(*loaded.method, runs);
        placed.store(false, std::memory_order_relaxed);
        if (!meet.wait()) break;
        for (unsigned after = 0; after < 3 && !meet.failed();) {
          const bool out = placed.load(std::memory_order_acquire);
          Outputs got = execute_once(*loaded.method, inputs, runs.method);
          if (run_executes++ == 0) {
            run_want = std::move(got);
          } else if (got != run_want) {
            ++run_diffs;
          }
          after += out ? 1 : 0;
          run_after += out ? 1 : 0;
        }
        if (!meet.wait()) break;  // the other method has run since, too
        uint32_t copy_ops = 0, copy_reads = 0;
        esp32_backend_weight_tier_copy_reads(&copy_ops, &copy_reads);
        rounds_read += copy_ops > 0 && copy_reads == copy_ops ? 1 : 0;
      }
      if (!meet.wait()) break;
      executorch::runtime::BackendOptions<1> cache;
      cache.set_option("weight_cache", false);
      set_backend_options(cache, "drop the unused copies");
      cache.set_option("weight_cache", true);
      set_backend_options(cache, "keep copies again");
      if (!meet.wait()) break;
    }
  } catch (...) {
    meet.fail();
    placer.join();
    throw;
  }
  placer.join();
  if (!place_error.empty()) throw std::runtime_error("placement thread: " + place_error);
  if (meet.failed()) throw std::runtime_error("placement stress stopped");
  write_outputs(runs, run_want);
  write_outputs(places, place_want);
  Esp32WeightTierStats stats;
  esp32_backend_weight_tier_stats(&stats);
  std::printf("placement stress: %u round(s); %s executed %u time(s), %u after a "
              "placement; %s executed %u time(s); placements %u, copies %u; "
              "every copy read in %u round(s); %u mismatch(es)\n",
              rounds, runs.method.c_str(), run_executes, run_after,
              places.method.c_str(), place_executes, (unsigned)stats.placements,
              (unsigned)stats.copies, rounds_read, run_diffs + place_diffs);
  return run_diffs + place_diffs == 0 && rounds_read == rounds ? 0 : 1;
}

// --placement-idle: a method that stays loaded and idle across many ranked
// placements still reads the copies they made at its next execute. The
// script's first method loads under a one-byte budget (its weights registered
// with the cache, left on flash) and executes once. Then, COUNT times, the
// second method is loaded ranked under an unlimited budget, placed (the first
// placement copies the first method's weights as well) and destroyed, and the
// unused copies are dropped (weight_cache off, then on), so that every round
// places again. Then the first method executes again: every op whose weight
// has a copy must read it, and its outputs must be its first execute's.
int run_placement_idle(const Options& options, const Program& program,
                       const NamedDataMap* ptd, const std::vector<Step>& steps) {
  if (steps.size() != 2 || steps[0].directive() || steps[1].directive() ||
      steps[0].method == steps[1].method) {
    throw std::runtime_error("--placement-idle takes a script of two lines "
                             "naming two methods");
  }
  const Step& idle = steps[0];
  const Step& places = steps[1];
  PlannedMemory idle_memory(program, idle.method), place_memory(program, places.method);
  set_stress_tiers(1, false);
  LoadedMethod waiting(program, idle.method, idle_memory.get(), ptd);
  const auto inputs = read_inputs(*waiting.method, idle);
  const Outputs first = execute_once(*waiting.method, inputs, idle.method);
  set_stress_tiers(0x7FFFFFFF, true);
  for (unsigned round = 0; round < options.placement_idle; ++round) {
    {
      LoadedMethod placed(program, places.method, place_memory.get(), ptd);
      executorch::runtime::BackendOptions<1> apply;
      apply.set_option("weight_tier_apply", true);
      set_backend_options(apply, "apply the weight tiers");
    }
    executorch::runtime::BackendOptions<1> cache;
    cache.set_option("weight_cache", false);
    set_backend_options(cache, "drop the unused copies");
    cache.set_option("weight_cache", true);
    set_backend_options(cache, "keep copies again");
  }
  const Outputs after = execute_once(*waiting.method, inputs, idle.method);
  write_outputs(idle, first);
  Esp32WeightTierStats stats;
  esp32_backend_weight_tier_stats(&stats);
  uint32_t copy_ops = 0, copy_reads = 0;
  esp32_backend_weight_tier_copy_reads(&copy_ops, &copy_reads);
  const bool same = after == first;
  std::printf("placement idle: %u placement(s) while %s waited; then it read %u of "
              "%u copied op(s); outputs %s\n",
              (unsigned)stats.placements, idle.method.c_str(),
              (unsigned)copy_reads, (unsigned)copy_ops,
              same ? "the same" : "DIFFERENT");
  return same && copy_ops > 0 && copy_reads == copy_ops ? 0 : 1;
}

int run(const Options& options) {
  executorch::runtime::runtime_init();
  check(torch::executor::register_all_kernels(), "register portable kernels");
  check(espressif::register_ops(), "register espressif ops");
  esp32_backend_set_force_scalar(options.scalar ? 1 : 0);
  rmsnorm_set_force_scalar(options.scalar ? 1 : 0);
  if (options.split && options.single_core) {
    throw std::runtime_error("--min-latency and --single-core contradict");
  }
  if (options.split) esp32_backend_set_split(1);
  if (options.single_core) esp32_backend_set_split(0);
  esp32_backend_set_conv_fast(options.no_conv_fast ? 0 : 1);
  // qdq_pool.h's quantize/dequantize/avg_pool2d fast paths: off sends every
  // call to the portable kernel, which computes the same bits (--no-qdq-fast).
  espressif::qdq_pool::set_enabled(!options.no_qdq_fast);
  // elementwise.h's loops (mul.out's channel scale, constant_pad_nd's rows,
  // the table ops'): off sends every call to the kernels they replace, which
  // compute the same bits (--no-ew-fast).
  espressif::elementwise::set_enabled(!options.no_ew_fast);
  if (options.psram_bytes) esp32_host_set_psram_bytes(*options.psram_bytes);
  {
    // Through the option interface, as an application sets them.
    executorch::runtime::BackendOptions<5> tiers;
    if (options.tier_budget) tiers.set_option("weight_tier_budget", *options.tier_budget);
    if (options.tier_reserve) tiers.set_option("weight_tier_reserve", *options.tier_reserve);
    if (options.tier_tables) tiers.set_option("weight_tier_tables", true);
    if (options.tier_rank) tiers.set_option("weight_tier_rank", true);
    if (options.no_weight_cache) tiers.set_option("weight_cache", false);
    if (tiers.view().size() != 0) {
      check(executorch::runtime::set_option("Esp32Backend", tiers.view()),
            "set the weight-tier options");
    }
  }
  const std::vector<Step> steps = options.script.empty()
      ? std::vector<Step>{{options.method, options.inputs, options.outputs}}
      : read_script(options.script);

  auto pte_loader = MmapDataLoader::from(options.pte.c_str(),
                                       MmapDataLoader::MlockConfig::NoMlock);
  check(pte_loader.error(), "open PTE");
  auto program = Program::load(&*pte_loader, Program::Verification::InternalConsistency);
  check(program.error(), "load PTE");
  // Check every step before running any. Each planned buffer is sized for all
  // the methods used, which share it: the story firmware's prefill and decode
  // share mem-id-2, their KV state, the same way.
  std::vector<size_t> planned_sizes;
  for (const Step& step : steps) {
    if (step.directive()) continue;
    auto meta = program->method_meta(step.method.c_str());
    check(meta.error(), "read method metadata for " + step.method);
    if (step.inputs.size() != meta->num_inputs() ||
        step.outputs.size() != meta->num_outputs()) {
      throw std::runtime_error("method " + step.method + " requires " +
                               std::to_string(meta->num_inputs()) + " input file(s) and " +
                               std::to_string(meta->num_outputs()) + " output file(s)");
    }
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
      auto size = meta->memory_planned_buffer_size(i);
      check(size.error(), "read planned buffer size");
      if (*size < 0) throw std::runtime_error("negative planned buffer size");
      planned_sizes.resize(std::max(planned_sizes.size(), i + 1), 0);
      planned_sizes[i] = std::max(planned_sizes[i], static_cast<size_t>(*size));
    }
  }

  // The named data mapped once, whole, as the firmware maps its weights
  // partition: each tensor then sits at one address for every method that
  // loads it, and the backend's weight cache, which names a weight by where
  // it lives, shares one copy between them as it does on the board. (Mapped
  // tensor by tensor, a method loaded after another may find its tensors at
  // new addresses and copy them again.)
  std::optional<MmapDataLoader> ptd_loader;
  std::optional<executorch::runtime::FreeableBuffer> ptd_bytes;
  std::optional<BufferDataLoader> ptd_mapped;
  std::optional<FlatTensorDataMap> ptd;
  if (!options.ptd.empty()) {
    auto loader = MmapDataLoader::from(options.ptd.c_str(),
                                     MmapDataLoader::MlockConfig::NoMlock);
    check(loader.error(), "open PTD");
    ptd_loader.emplace(std::move(*loader));
    auto size = ptd_loader->size();
    check(size.error(), "size PTD");
    auto bytes = ptd_loader->load(0, *size, executorch::runtime::DataLoader::SegmentInfo(
        executorch::runtime::DataLoader::SegmentInfo::Type::External));
    check(bytes.error(), "map PTD");
    ptd_bytes.emplace(std::move(*bytes));
    ptd_mapped.emplace(ptd_bytes->data(), ptd_bytes->size());
    auto map = FlatTensorDataMap::load(&*ptd_mapped);
    check(map.error(), "load PTD");
    ptd.emplace(std::move(*map));
  }
  if (options.tier_slots) esp32_backend_host_cap_tier_slots(*options.tier_slots);
  if (options.placement_stress != 0) {
    return run_placement_stress(options, *program, ptd ? &*ptd : nullptr, steps);
  }
  if (options.placement_idle != 0) {
    return run_placement_idle(options, *program, ptd ? &*ptd : nullptr, steps);
  }

  std::vector<std::unique_ptr<AlignedBytes>> planned;
  std::vector<Span<uint8_t>> spans;
  for (size_t size : planned_sizes) {
    planned.push_back(std::make_unique<AlignedBytes>(size + options.planned_offset));
    spans.emplace_back(static_cast<uint8_t*>(planned.back()->data()) + options.planned_offset,
                       size);
    if (options.psram_planned) {
      esp32_host_mark_psram(spans.back().data(), size);
    }
  }
  HierarchicalAllocator planned_allocator({spans.data(), spans.size()});
  std::vector<std::unique_ptr<LoadedMethod>> loaded;
  const auto find = [&](const std::string& name) -> LoadedMethod* {
    for (auto& held : loaded) {
      if (held->name == name) return held.get();
    }
    return nullptr;
  };
  const auto load = [&](const std::string& name) {
    loaded.push_back(std::make_unique<LoadedMethod>(*program, name, &planned_allocator,
                                                    ptd ? &*ptd : nullptr));
  };
  // As an application places a ranked load: after loading, before the first
  // execute.
  const auto apply = [&]() {
    if (!options.tier_apply) return;
    executorch::runtime::BackendOptions<1> tiers;
    tiers.set_option("weight_tier_apply", true);
    check(executorch::runtime::set_option("Esp32Backend", tiers.view()),
          "apply the weight tiers");
  };
  if (options.keep_methods) {
    // Every method the script names, loaded in the order it names them before
    // the first step, then kept: an application holding prefill and decode.
    for (const Step& step : steps) {
      if (!step.directive() && find(step.method) == nullptr) load(step.method);
    }
    apply();
  }
  for (const Step& step : steps) {
    if (step.directive()) {
      if (step.unload) {
        loaded.clear();
      } else {
        set_script_option(step.set_key, step.set_value);
      }
      Esp32WeightTierStats stats;
      esp32_backend_weight_tier_stats(&stats);
      std::printf("%s%s%s%s: resident %llu B, entries %u\n",
                  step.unload ? "unload" : "set ", step.set_key.c_str(),
                  step.unload ? "" : " ", step.set_value.c_str(),
                  static_cast<unsigned long long>(stats.resident),
                  static_cast<unsigned>(stats.entries));
      continue;
    }
    LoadedMethod* method = find(step.method);
    if (method == nullptr) {
      // One method loaded at a time, reloaded when the script changes method,
      // as the firmware loads prefill and then decode.
      loaded.clear();
      load(step.method);
      apply();
      method = loaded.back().get();
    }
    run_step(*method, step);
  }
  esp32_backend_report();
  esp32_backend_report_weight_tiers();
  espressif::report_override_dispatch();
  const Esp32ReferenceKernelCalls calls = esp32_reference_kernel_calls();
  std::printf("host kernels: dot_i8 %llu, unpack_i4 %llu, dot_i4_groups %llu, "
              "rmsnorm %llu call(s)\n",
              static_cast<unsigned long long>(calls.dot_i8),
              static_cast<unsigned long long>(calls.unpack_i4),
              static_cast<unsigned long long>(calls.dot_i4_groups),
              static_cast<unsigned long long>(calls.rmsnorm));
  // The conv path's kernels: their outputs equal the C loops they replace, so only
  // these counts show they ran (test_conv_fast).
  std::printf("host conv kernels: qacc_lane_codes %llu, dot_rows_codes %llu, "
              "transpose16 %llu call(s)\n",
              static_cast<unsigned long long>(calls.qacc_lane_codes),
              static_cast<unsigned long long>(calls.dot_rows_codes),
              static_cast<unsigned long long>(calls.transpose16));
  // The dense columns' long-row kernel (k_pad past 112), likewise.
  std::printf("host conv long rows: dot_rows_long %llu call(s)\n",
              static_cast<unsigned long long>(
                  esp32_reference_dot_rows_long_calls()));
  // The dense convs' fp32 fast paths (esp32_conv.cpp QuantFast), likewise.
  const esp32_conv::QuantFastCalls fast = esp32_conv::quant_fast_calls();
  std::printf("host conv fp32 codes: quantize_span_fast %llu, "
              "gather_pointwise_fast %llu, gather_dense_fast %llu call(s)\n",
              static_cast<unsigned long long>(fast.span),
              static_cast<unsigned long long>(fast.pointwise),
              static_cast<unsigned long long>(fast.dense));
  // The depthwise plane kernels (dw5_plane_s3, dw3_plane_s3), likewise.
  std::printf("host depthwise plane kernels: %llu call(s)\n",
              static_cast<unsigned long long>(calls.dw_plane));
  esp32_backend_release_weight_cache();
  return 0;
}
} // namespace

// The backend's weight-tier options through ExecuTorch's option API, as an
// application sets them: what is set reads back (auto as -1, rank set as 1 as
// true), the read-only counts read as numbers, an apply with nothing loaded
// does nothing and reads back as nothing waiting, and a read-only key, an
// out-of-range budget or a rank of 2 is refused. The defaults are restored
// after.
static void run_weight_tier_option_check() {
  using executorch::runtime::BackendOptions;
  const auto set = [](auto& options) {
    return executorch::runtime::set_option("Esp32Backend", options.view());
  };
  BackendOptions<5> values;
  values.set_option("weight_tier_budget", -1);
  values.set_option("weight_tier_reserve", 4096);
  values.set_option("weight_tier_tables", true);
  values.set_option("weight_tier_rank", 1);  // the int spelling
  values.set_option("weight_tier_apply", true);  // nothing loaded: nothing to do
  bool roundtrip = set(values) == Error::Ok;
  BackendOptions<7> read;
  read.set_option("weight_tier_budget", 0);
  read.set_option("weight_tier_reserve", 0);
  read.set_option("weight_tier_tables", false);
  read.set_option("weight_tier_wanted", -1);
  read.set_option("weight_tier_resident", -1);
  read.set_option("weight_tier_rank", false);
  read.set_option("weight_tier_apply", true);
  roundtrip = roundtrip &&
      executorch::runtime::get_option("Esp32Backend", read.view()) == Error::Ok;
  int budget = 0, reserve = 0, wanted = -1, resident = -1;
  bool tables = false, rank = false, waiting = true;
  roundtrip = roundtrip && read.get_option("weight_tier_budget", budget) == Error::Ok &&
      read.get_option("weight_tier_reserve", reserve) == Error::Ok &&
      read.get_option("weight_tier_tables", tables) == Error::Ok &&
      read.get_option("weight_tier_wanted", wanted) == Error::Ok &&
      read.get_option("weight_tier_resident", resident) == Error::Ok &&
      read.get_option("weight_tier_rank", rank) == Error::Ok &&
      read.get_option("weight_tier_apply", waiting) == Error::Ok &&
      budget == -1 && reserve == 4096 && tables && wanted == 0 && resident == 0 &&
      rank && !waiting;
  BackendOptions<1> read_only;
  read_only.set_option("weight_tier_resident", 1);
  const bool read_only_refused = set(read_only) == Error::InvalidArgument;
  BackendOptions<1> below_auto;
  below_auto.set_option("weight_tier_budget", -2);
  const bool budget_refused = set(below_auto) == Error::InvalidArgument;
  BackendOptions<1> rank_two;
  rank_two.set_option("weight_tier_rank", 2);
  const bool rank_refused = set(rank_two) == Error::InvalidArgument;
  BackendOptions<4> defaults;
  defaults.set_option("weight_tier_budget", 0);
  defaults.set_option("weight_tier_reserve", 0);
  defaults.set_option("weight_tier_tables", false);
  defaults.set_option("weight_tier_rank", false);
  const bool restored = set(defaults) == Error::Ok;
  std::printf("weight tier options: roundtrip %s, read-only refused %s, "
              "budget below -1 refused %s, rank 2 refused %s, defaults restored %s\n",
              roundtrip ? "yes" : "no", read_only_refused ? "yes" : "no",
              budget_refused ? "yes" : "no", rank_refused ? "yes" : "no",
              restored ? "yes" : "no");
}

// asym_conformance.cpp: the runtime's affine column math, case by
// case (test_asym_conformance.py), and the non-finite quantize rule against
// ExecuTorch's own quantize kernel.
int run_asym_conformance(const char* cases_path, const char* out_path);
int run_nonfinite_check();
int run_qdq_sweep(const std::string& mode);

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "--asym-conformance") {
    try {
      return run_asym_conformance(argv[2], argv[3]);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "esp32_host_runner: %s\n", error.what());
      return 1;
    }
  }
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage();
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--qdq-sweep") {
    return run_qdq_sweep(argv[2]);  // ops/qdq_pool.h against the portable kernels
  }
  if (argc == 2 && std::string(argv[1]) == "--microbench") {
    esp32_backend_microbench();
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--selftest") {
    // The checks main.cpp drives on the device, here against the C kernels.
    espressif::run_rmsnorm_domain_check();
    espressif::run_ln2d_refusal_check();
    espressif::run_portable_fastpath_check();
    espressif::run_fast_layer_norm_check();
    esp32_asym::run_kcache_zero_point_check();
    esp32_asym::run_a16_check();
    espressif::quant_lut::run_conformance_check();
    espressif::lut16::run_conformance_check();
    espressif::divsqrt::run_check();
    espressif::expf_check::run_check();
    run_nonfinite_check();
    run_weight_tier_option_check();
    return 0;
  }
  try {
    return run(parse(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "esp32_host_runner: %s\n", error.what());
    return 1;
  }
}
