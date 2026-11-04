/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ESP-IDF platform abstraction layer for ExecuTorch.
//
// ExecuTorch's runtime/platform/default/minimal.cpp declares these symbols ET_WEAK,
// so strong definitions here override them at link time without touching upstream.
// minimal.cpp is deliberately assumption-free: it has no clock (returns a literal
// 11223344 so the placeholder is easy to spot), discards all log output, and refuses
// to allocate. Each of those matters on this target:
//
//   * a real clock is a prerequisite for any on-device timing work, and it is what
//     ExecuTorch's event tracer reads;
//   * log messages are how kernel-level ET_KERNEL_CHECK failures explain themselves,
//     and silently dropping them turns a diagnosable error into a bare error code;
//   * et_pal_allocate returning nullptr is fine for the fully memory-planned path but
//     forecloses anything that needs scratch at runtime.
//
// Upstream has runtime/platform/default/zephyr.cpp as the precedent for an RTOS PAL.

#include <executorch/runtime/platform/platform.h>

#include <cinttypes>
#include <cstdio>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"

static const char* kTag = "executorch";

void et_pal_init(void) {
  // esp_timer is already running by the time app_main is reached, so there is
  // nothing to start. Announce once so it is obvious which PAL is linked --
  // ExecuTorch calls this more than once (runtime_init plus internal paths).
  static bool announced = false;
  if (!announced) {
    announced = true;
    ESP_LOGI(kTag, "ESP-IDF PAL active: esp_timer clock, ESP_LOG output, "
                   "internal-RAM allocator");
  }
}

ET_NORETURN void et_pal_abort(void) {
  // esp_system_abort gives a panic handler backtrace, which __builtin_trap does not.
  esp_system_abort("ExecuTorch called et_pal_abort");
}

et_timestamp_t et_pal_current_ticks(void) {
  // Microseconds since boot, monotonic across light sleep.
  return (et_timestamp_t)esp_timer_get_time();
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // esp_timer ticks are microseconds, so 1000 ns per tick.
  return {1000, 1};
}

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  // Route through IDF logging so ET output interleaves correctly with the rest of
  // the system and obeys the configured log level. Keep ET's own severity letter
  // and source location, which are the useful parts of a kernel check failure.
  const char* base = filename;
  for (const char* p = filename; p != nullptr && *p != '\0'; ++p) {
    if (*p == '/') {
      base = p + 1;
    }
  }
  switch (level) {
    case kDebug:
      ESP_LOGD(kTag, "%s:%zu %s", base, line, message);
      break;
    case kInfo:
      ESP_LOGI(kTag, "%s:%zu %s", base, line, message);
      break;
    case kError:
      ESP_LOGE(kTag, "%s:%zu %s", base, line, message);
      break;
    case kFatal:
      ESP_LOGE(kTag, "FATAL %s:%zu %s", base, line, message);
      break;
    default:
      ESP_LOGW(kTag, "%s:%zu %s", base, line, message);
      break;
  }
  (void)timestamp;
}

void* et_pal_allocate(size_t size) {
  // Internal SRAM: callers of this path want fast memory, and PSRAM may not exist
  // on every board. Returns nullptr on failure, which is the documented contract.
  return heap_caps_malloc(size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

void et_pal_free(void* ptr) {
  heap_caps_free(ptr);
}
