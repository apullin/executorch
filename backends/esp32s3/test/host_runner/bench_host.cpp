/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The firmware's bench runner (examples/espressif/esp32s3/main/bench_runner.inc)
// on the build machine: the same code, on this runtime's portable kernels and
// delegate, reading a bench image and its weights from files instead of the
// board's partitions 0x43 and 0x44.
//
//   esp32_bench_host --image bench_image.bin --ptd bench_weights.ptd
//                    [--psram-bytes N] [--timed-runs N] [--stepped-runs N]
//                    [--read-only] [--min-latency | --single-core]
//                    [--force-scalar] [--no-conv-fast] [--no-qdq-fast]
//
// It prints what the board prints (BENCH_* lines), so an exported image is
// checked end to end -- the input generator against the exporter's, the
// output bits against the host runner's, the step pass against the image's
// labels, the ranked placement's apply -- before a board runs it. Its times
// are the build machine's.
//
// --read-only maps the image and the weights as the board does: pages a
// store cannot change (mmap, then mprotect PROT_READ). The board maps
// partitions 0x43 and 0x44 from flash, where a store is rejected (a Cache
// error panic) or lost; here it faults, and the handler names the mapping
// and the offset it hit (BENCH_FAULT) before the process dies. The flash
// measurement reads every weight in place, so a store into a constant on
// any path it runs shows here. --min-latency turns min_latency on before
// anything loads, as a dual-core bench firmware does (ET_BENCH_DUAL_CORE:
// projections split, every convolution on one core), and prints the
// firmware's BENCH_CORES line; --single-core holds every
// handle to one core. --force-scalar, --no-conv-fast and --no-qdq-fast are
// the host runner's switches (the scalar loops, the general conv loops, the
// portable q/dq/avg_pool2d kernels).

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <execinfo.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include "esp32_conv.h"
#include "esp32_kernels.h"
#include "esp32_platform.h"
#include "espressif_ops.h"
#include "elementwise.h"
#include "qdq_pool.h"

#include "bench_runner.inc"

namespace torch::executor {
::executorch::runtime::Error register_all_kernels();
}
extern "C" void esp32_backend_set_force_scalar(int);
extern "C" void esp32_backend_set_conv_fast(int);
extern "C" void esp32_backend_set_split(int);
extern "C" void esp32_backend_report();

namespace {

// 64-byte aligned, as a flash partition's mapping is (and more than the
// program's 16-byte segments need); page-aligned and read-only under
// --read-only.
struct File {
  uint8_t* data = nullptr;
  size_t size = 0;
  size_t mapped = 0;  // bytes mmap'd (--read-only), else 0: aligned_alloc'd
  File() = default;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File() {
    if (mapped != 0) {
      munmap(data, mapped);
    } else {
      std::free(data);
    }
  }
};

void read_file(const std::string& path, bool read_only, File& f) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  f.size = static_cast<size_t>(in.tellg());
  if (read_only) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    f.mapped = (std::max<size_t>(f.size, 1) + page - 1) / page * page;
    void* p = mmap(nullptr, f.mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
      std::fprintf(stderr, "cannot map %zu B for %s\n", f.mapped, path.c_str());
      std::exit(2);
    }
    f.data = static_cast<uint8_t*>(p);
  } else {
    f.data = static_cast<uint8_t*>(std::aligned_alloc(64, (f.size + 63) & ~size_t{63}));
  }
  in.seekg(0);
  in.read(reinterpret_cast<char*>(f.data), static_cast<std::streamsize>(f.size));
  if (read_only && mprotect(f.data, f.mapped, PROT_READ) != 0) {
    std::fprintf(stderr, "cannot protect %s\n", path.c_str());
    std::exit(2);
  }
}

// The read-only mappings the fault handler names.
struct Region {
  const char* name;
  const uint8_t* base;
  size_t size;
};
Region g_regions[2];

// A fault in a read-only mapping is a store into a constant: say which
// mapping, the offset and the address, then the stack, and die (async-signal-
// safe calls only: snprintf into a static buffer, write, backtrace).
void on_fault(int sig, siginfo_t* info, void*) {
  // One report: the split worker's thread may fault beside the caller's.
  static std::atomic<int> reporting{0};
  if (reporting.exchange(1) != 0) {
    for (;;) {
      pause();
    }
  }
  static char line[256];
  const uint8_t* at = static_cast<const uint8_t*>(info->si_addr);
  int n = 0;
  for (const Region& r : g_regions) {
    if (r.base != nullptr && at >= r.base && at < r.base + r.size) {
      n = std::snprintf(line, sizeof(line),
                        "BENCH_FAULT a store into the read-only %s at offset %zu "
                        "(address %p, signal %d)\n",
                        r.name, static_cast<size_t>(at - r.base), info->si_addr, sig);
    }
  }
  if (n == 0) {
    n = std::snprintf(line, sizeof(line), "BENCH_FAULT signal %d at address %p\n", sig,
                      info->si_addr);
  }
  (void)!write(STDOUT_FILENO, line, static_cast<size_t>(n));
  void* frames[64];
  const int depth = backtrace(frames, 64);
  backtrace_symbols_fd(frames, depth, STDOUT_FILENO);
  _exit(3);
}

}  // namespace

int main(int argc, char** argv) {
  std::string image_path, ptd_path;
  long long psram = -1, timed = -1, stepped = -1;
  bool read_only = false, min_latency = false, single_core = false,
       force_scalar = false, no_conv_fast = false, no_qdq_fast = false,
       no_ew_fast = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    const char* value = i + 1 < argc ? argv[i + 1] : nullptr;
    if (key == "--help") {
      std::printf("usage: esp32_bench_host --image FILE --ptd FILE [--psram-bytes N] "
                  "[--timed-runs N] [--stepped-runs N] [--read-only] "
                  "[--min-latency | --single-core] [--force-scalar] [--no-conv-fast] "
                  "[--no-qdq-fast] [--no-ew-fast]\n");
      return 0;
    }
    bool* flag = key == "--read-only"      ? &read_only
                 : key == "--min-latency"  ? &min_latency
                 : key == "--single-core"  ? &single_core
                 : key == "--force-scalar" ? &force_scalar
                 : key == "--no-conv-fast" ? &no_conv_fast
                 : key == "--no-qdq-fast"  ? &no_qdq_fast
                 : key == "--no-ew-fast"   ? &no_ew_fast
                                           : nullptr;
    if (flag != nullptr) {
      *flag = true;
      continue;
    }
    if (value == nullptr) {
      std::fprintf(stderr, "%s needs a value\n", key.c_str());
      return 2;
    }
    if (key == "--image") {
      image_path = value;
    } else if (key == "--ptd") {
      ptd_path = value;
    } else if (key == "--psram-bytes") {
      psram = std::atoll(value);
    } else if (key == "--timed-runs") {
      timed = std::atoll(value);
    } else if (key == "--stepped-runs") {
      stepped = std::atoll(value);
    } else {
      std::fprintf(stderr, "unknown option %s\n", key.c_str());
      return 2;
    }
    ++i;
  }
  if (image_path.empty() || ptd_path.empty()) {
    std::fprintf(stderr, "--image and --ptd are required\n");
    return 2;
  }
  if (min_latency && single_core) {
    std::fprintf(stderr, "--min-latency and --single-core contradict\n");
    return 2;
  }
  if (timed > static_cast<long long>(bench::kMaxRuns) ||
      stepped > static_cast<long long>(bench::kMaxRuns)) {
    std::fprintf(stderr, "--timed-runs and --stepped-runs take at most %u runs\n",
                 static_cast<unsigned>(bench::kMaxRuns));
    return 2;
  }
  executorch::runtime::runtime_init();
  if (torch::executor::register_all_kernels() != executorch::runtime::Error::Ok ||
      espressif::register_ops() != executorch::runtime::Error::Ok) {
    std::fprintf(stderr, "kernel registration failed\n");
    return 1;
  }
  if (psram >= 0) esp32_host_set_psram_bytes(static_cast<size_t>(psram));
  esp32_backend_set_force_scalar(force_scalar ? 1 : 0);
  esp32_backend_set_conv_fast(no_conv_fast ? 0 : 1);
  espressif::qdq_pool::set_enabled(!no_qdq_fast);
  espressif::elementwise::set_enabled(!no_ew_fast);
  if (single_core) esp32_backend_set_split(0);
  if (min_latency) {
    // As the dual-core firmware does before anything loads (bench_runner.inc,
    // ET_BENCH_DUAL_CORE): through the option interface, the same line.
    executorch::runtime::BackendOptions<1> on;
    on.set_option("min_latency", true);
    const auto e = executorch::runtime::set_option("Esp32Backend", on.view());
    std::printf("BENCH_CORES dual-core min_latency: %s\n",
                e == executorch::runtime::Error::Ok ? "ON" : "REFUSED");
  }

  File image, ptd;
  read_file(image_path, read_only, image);
  read_file(ptd_path, read_only, ptd);
  if (read_only) {
    g_regions[0] = Region{"image", image.data, image.mapped};
    g_regions[1] = Region{"weights", ptd.data, ptd.mapped};
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    std::printf("BENCH_READONLY image %zu B and weights %zu B mapped read-only\n",
                image.size, ptd.size);
  }
  bench::Image parsed;
  if (!bench::parse(image.data, image.size, &parsed)) return 1;
  // The run counts are the image's; a test may shorten them (to at most
  // kMaxRuns, checked above).
  if (timed >= 0) parsed.h.timed_runs = static_cast<uint32_t>(timed);
  if (stepped >= 0) parsed.h.stepped_runs = static_cast<uint32_t>(stepped);
  executorch::extension::BufferDataLoader weights(ptd.data, ptd.size);
  bench::run(parsed, &weights, ptd.data, ptd.size);
  // The backend's counters, as the firmware prints them after the bench
  // (main.cpp): the conv path's ops, those split across the cores, the
  // scratch they ran on.
  esp32_backend_report();
  // The conv path's kernels, as the host runner reports them: their outputs
  // equal the C loops they replace, so only these counts show they ran (test_bench).
  const Esp32ReferenceKernelCalls calls = esp32_reference_kernel_calls();
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
  // The element-wise loops (elementwise.h), which give their kernels' bits too:
  // one line per loop that ran (test_bench).
  espressif::elementwise::report_dispatch();
  std::printf("=== DONE ===\n");
  return 0;
}
