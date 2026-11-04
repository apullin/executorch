/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The delegate's ESP-IDF boundary. ESP32_HOST_BUILD selects the serial host
// implementation; firmware keeps the original capability and FreeRTOS APIs.
#pragma once

#include <cstdint>

#if defined(ESP32_HOST_BUILD)
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

constexpr unsigned MALLOC_CAP_SPIRAM = 1u;
constexpr unsigned MALLOC_CAP_8BIT = 2u;
constexpr unsigned MALLOC_CAP_INTERNAL = 4u;
// Cold .bss the firmware may place in PSRAM; an ordinary static here.
#define EXT_RAM_BSS_ATTR

// The host's PSRAM: a byte count against which MALLOC_CAP_SPIRAM allocations
// are charged, so a test can give a program a board's PSRAM, or none at all as
// under QEMU (esp32_host_runner --psram-bytes). Unlimited until set. Every
// other allocation is plain host memory, charged to nothing. One instance
// across translation units: an inline function's static is.
struct Esp32HostPsram {
  std::mutex lock;
  size_t capacity = SIZE_MAX;
  size_t used = 0;
};
inline Esp32HostPsram& esp32_host_psram() {
  static Esp32HostPsram psram;
  return psram;
}
inline void esp32_host_set_psram_bytes(size_t bytes) {
  Esp32HostPsram& p = esp32_host_psram();
  std::lock_guard<std::mutex> hold(p.lock);
  p.capacity = bytes;
}

// Each block carries its size and whether PSRAM was charged for it, in a
// header that keeps malloc's 16-byte alignment for the caller.
constexpr size_t kEsp32HostHeapHeader = 16;
inline void* heap_caps_malloc(size_t size, unsigned caps) {
  const bool spiram = (caps & MALLOC_CAP_SPIRAM) != 0;
  if (size > SIZE_MAX - kEsp32HostHeapHeader) {
    return nullptr;
  }
  if (spiram) {
    Esp32HostPsram& p = esp32_host_psram();
    std::lock_guard<std::mutex> hold(p.lock);
    if (p.used > p.capacity || size > p.capacity - p.used) {
      return nullptr;
    }
    p.used += size;
  }
  auto* raw = static_cast<unsigned char*>(std::malloc(size + kEsp32HostHeapHeader));
  if (raw == nullptr) {
    if (spiram) {
      Esp32HostPsram& p = esp32_host_psram();
      std::lock_guard<std::mutex> hold(p.lock);
      p.used -= size;
    }
    return nullptr;
  }
  const size_t header[2] = {size, spiram ? size_t{1} : size_t{0}};
  static_assert(sizeof(header) <= kEsp32HostHeapHeader, "header fits");
  std::memcpy(raw, header, sizeof(header));
  return raw + kEsp32HostHeapHeader;
}
// No caches and no external RAM here: every block is internal, except the
// address ranges a test marks as PSRAM (esp32_host_mark_psram; the runner's
// --psram-planned), so the paths that must keep their writes out of external
// memory run here too.
struct Esp32HostPsramRanges {
  std::mutex lock;
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
};
inline Esp32HostPsramRanges& esp32_host_psram_ranges() {
  static Esp32HostPsramRanges r;
  return r;
}
inline void esp32_host_mark_psram(const void* p, size_t bytes) {
  Esp32HostPsramRanges& r = esp32_host_psram_ranges();
  std::lock_guard<std::mutex> hold(r.lock);
  const uintptr_t at = reinterpret_cast<uintptr_t>(p);
  r.ranges.emplace_back(at, at + bytes);
}
inline bool esp32_internal_ram(const void* p) {
  Esp32HostPsramRanges& r = esp32_host_psram_ranges();
  std::lock_guard<std::mutex> hold(r.lock);
  const uintptr_t at = reinterpret_cast<uintptr_t>(p);
  for (const auto& range : r.ranges) {
    if (at >= range.first && at < range.second) {
      return false;
    }
  }
  return true;
}
inline void heap_caps_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  auto* raw = static_cast<unsigned char*>(ptr) - kEsp32HostHeapHeader;
  size_t header[2];
  std::memcpy(header, raw, sizeof(header));
  if (header[1] != 0) {
    Esp32HostPsram& p = esp32_host_psram();
    std::lock_guard<std::mutex> hold(p.lock);
    p.used -= header[0];
  }
  std::free(raw);
}
// PSRAM's free bytes are what its capacity has left (one block: the host does
// not fragment it); host memory reports none, as nothing here asks.
inline size_t heap_caps_get_free_size(unsigned caps) {
  if ((caps & MALLOC_CAP_SPIRAM) == 0) {
    return 0;
  }
  Esp32HostPsram& p = esp32_host_psram();
  std::lock_guard<std::mutex> hold(p.lock);
  return p.used >= p.capacity ? 0 : p.capacity - p.used;
}
inline size_t heap_caps_get_largest_free_block(unsigned caps) {
  return heap_caps_get_free_size(caps);
}
inline int64_t esp_timer_get_time() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// FreeRTOS's semaphores as mutexes: the lifecycle lock (taken with
// portMAX_DELAY) and the split-job lease (taken without waiting). The split
// worker is a thread per job instead of a task (esp32_host_split_start
// below), and the caller's wait for its job joins that thread. Binary
// semaphores share the mutex type because the host's one, the split-done
// signal, is never taken: the join stands in for it.
using BaseType_t = int;
using UBaseType_t = unsigned;
using TaskHandle_t = void*;
using SemaphoreHandle_t = std::mutex*;
struct StaticSemaphore_t {
  std::mutex mutex;
};
constexpr int pdFALSE = 0;
constexpr int pdTRUE = 1;
constexpr unsigned portMAX_DELAY = 0xffffffffu;
inline SemaphoreHandle_t xSemaphoreCreateMutexStatic(StaticSemaphore_t* storage) {
  return &storage->mutex;
}
inline SemaphoreHandle_t xSemaphoreCreateBinaryStatic(StaticSemaphore_t* storage) {
  return &storage->mutex;
}
inline int xSemaphoreTake(SemaphoreHandle_t mutex, unsigned ticks) {
  if (ticks == 0) {
    return mutex->try_lock() ? pdTRUE : pdFALSE;
  }
  mutex->lock();  // Any nonzero wait blocks until given, like portMAX_DELAY.
  return pdTRUE;
}
inline void xSemaphoreGive(SemaphoreHandle_t mutex) {
  mutex->unlock();
}
// The caller's core, 0, never the worker's (1), so the split stays
// available; the worker's part runs on a thread of its own.
inline BaseType_t xPortGetCoreID() {
  return 0;
}

// The host's split worker: each job on a thread of its own, started once the
// caller has written the job (start_split_job) and joined before the caller
// reads its results (join_split, join_conv). The thread's start and join
// order the two as the task notify and the done semaphore do on the chip, and
// the worker's part runs while the caller computes its own, as on the chip:
// a part that writes where the other reads or writes changes the output (or
// trips ThreadSanitizer) instead of running first and passing. A job left
// unjoined ends the process at the next start (std::thread's terminate).
struct Esp32HostSplit {
  std::thread worker;
  std::atomic<uint32_t> jobs{0};
};
inline Esp32HostSplit& esp32_host_split() {
  static Esp32HostSplit split;
  return split;
}
inline void esp32_host_split_start(void (*job)()) {
  Esp32HostSplit& split = esp32_host_split();
  split.worker = std::thread(job);
  split.jobs.fetch_add(1, std::memory_order_relaxed);
}
inline void esp32_host_split_join() {
  esp32_host_split().worker.join();
}
// The jobs started so far (esp32_backend_report's host line).
inline uint32_t esp32_host_split_jobs() {
  return esp32_host_split().jobs.load(std::memory_order_relaxed);
}
inline UBaseType_t uxTaskPriorityGet(TaskHandle_t /*task*/) {
  return 0;
}
inline void vTaskPrioritySet(TaskHandle_t /*task*/, UBaseType_t /*priority*/) {}

// floor(value) as the chip's FLOOR.S converts it, saturating: past the int32
// range the result is INT32_MAX (+inf and NaN included) or INT32_MIN (-inf),
// where a plain static_cast is undefined and differs by host (x86's
// cvttss2si gives INT32_MIN for all of them, so a static-scale outlier far
// past the calibrated range would land on the wrong rail). The quantisers
// clamp the result to their code range, so this is what puts such values on
// the device's rail. Keep the addition of 0.5f in the caller in fp32,
// including its rounding on near-ties.
static inline __attribute__((always_inline)) int32_t esp32_platform_floor_i(
    float value) {
  if (value >= 2147483648.0f || value != value) {
    return INT32_MAX;
  }
  if (value < -2147483648.0f) {
    return INT32_MIN;
  }
  return static_cast<int32_t>(std::floor(value));
}
#else
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_memory_utils.h>

// Internal SRAM, which no cache stands between the cores and (not PSRAM or a
// flash mapping): where everything a split writes must live
// (esp32_backend.cpp split_writes_internal).
static inline bool esp32_internal_ram(const void* p) {
  return esp_ptr_internal(p);
}

static inline __attribute__((always_inline)) int32_t esp32_platform_floor_i(
    float value) {
  int32_t result;
  asm("floor.s %0, %1, 0" : "=a"(result) : "f"(value));
  return result;
}
#endif
