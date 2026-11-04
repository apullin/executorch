/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "papercolor_status.h"

#include <atomic>
#include <cstdio>
#include <inttypes.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "lvgl.h"
#include "papercolor_display.h"
#include "papercolor_ui.h"

namespace et_papercolor {
namespace {

enum class RequestKind : uint8_t {
  kText,
  kReadHumidity,
  kTool,  // a tool card whose result the caller supplied
  kStory, // a long text in the small face
  kStoryPage, // one page of a story, with a page number
};

struct StatusMessage {
  uint32_t sequence;
  RequestKind kind;
  char title[32];
  char action[96];
  char argument[256];
  char result[64];
  char detail[64];
  char story[1024];
  char corner[16];
  // Whether this frame may use the interrupted waveform. Carried by the
  // message, not read from a flag at render time: a fast progress frame and
  // the complete final page are posted back to back, and the frame must
  // keep the mode it was posted with.
  bool fast;
};

QueueHandle_t g_queue = nullptr;
std::atomic<bool> g_started{false};
std::atomic<bool> g_ready{false};
std::atomic<bool> g_failed{false};
std::atomic<uint32_t> g_submitted{0};
std::atomic<uint32_t> g_completed{0};
std::atomic<bool> g_fast_allowed{true};
// How long the fast refresh lets the panel's own waveform run before cutting
// it short (the driver allows 100 to 2000 ms). Longer is cleaner and slower:
// 340 ms = 0.58 s per refresh, 2000 ms = 2.24 s. At 340 a single page turn
// leaves the panel half black, so the default is the cleanest fast setting the
// driver allows; full refreshes take ~15 s.
std::atomic<uint16_t> g_fast_cutoff_ms{2000};

constexpr uint32_t kFastRefreshBalanceLimit = 8;
constexpr char kNvsNamespace[] = "pc_status";
constexpr char kNvsFastCount[] = "fast_count";

void copyText(char* destination, size_t size, const char* source) {
  if (source == nullptr) source = "";
  snprintf(destination, size, "%s", source);
}

esp_err_t readHumidity(papercolor::Display& display, StatusMessage* message) {
  float temperature_c = 0.0f;
  float humidity_pct = 0.0f;
  esp_err_t error = display.readEnvironment(&temperature_c, &humidity_pct);
  if (error == ESP_OK) {
    snprintf(message->result, sizeof(message->result), "Humidity %.1f %%",
             humidity_pct);
    snprintf(message->detail, sizeof(message->detail),
             "Temperature %.1f C", temperature_c);
    printf("PAPERCOLOR_TOOL name=read_sensor sensor=humidity result=PASS "
           "humidity_pct=%.2f temperature_c=%.2f\n",
           humidity_pct, temperature_c);
  } else {
    copyText(message->result, sizeof(message->result), "Sensor error");
    snprintf(message->detail, sizeof(message->detail), "ESP error %d",
             error);
    printf("PAPERCOLOR_TOOL name=read_sensor sensor=humidity result=FAIL "
           "error=%d\n", error);
  }
  return error;
}

esp_err_t resolveRequest(papercolor::Display& display,
                         StatusMessage* message) {
  switch (message->kind) {
    case RequestKind::kText:
      return ESP_OK;
    case RequestKind::kReadHumidity:
      return readHumidity(display, message);
    case RequestKind::kTool:
    case RequestKind::kStory:
    case RequestKind::kStoryPage:
      return ESP_OK;
  }
  return ESP_ERR_INVALID_ARG;
}

// LVGL renders the cards in components/papercolor/ui into a one-bit frame whose
// layout is the driver's own (one bit per pixel, MSB first, 1 = paper), so the
// flush is a copy. LVGL is single-threaded: only this task touches it.
constexpr size_t kLvPalette = 8;  // I1 frames carry a two-entry palette first
papercolor::Display* g_display = nullptr;

// Ink census of the frame the panel receives, per horizontal quarter, so the
// serial log says what was drawn without a camera. With
// ET_PAPERCOLOR_DUMP_FRAME the whole frame follows as hex rows
// (PAPERCOLOR_FRAME_ROW), one bit per pixel, MSB first, 1 = paper.
void reportFrame(const uint8_t* mono) {
  constexpr int kRowBytes = papercolor::kWidth / 8;
  constexpr int kBand = papercolor::kHeight / 4;
  uint32_t ink[4] = {0, 0, 0, 0};
  for (int y = 0; y < papercolor::kHeight; ++y) {
    for (int i = 0; i < kRowBytes; ++i) {
      ink[y / kBand] += 8 - __builtin_popcount(mono[y * kRowBytes + i]);
    }
  }
  printf("PAPERCOLOR_FRAME ink_q=%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
         " of %u per band\n", ink[0], ink[1], ink[2], ink[3],
         static_cast<unsigned>(papercolor::kWidth) * kBand);
#ifdef ET_PAPERCOLOR_DUMP_FRAME
  static const char kHex[] = "0123456789abcdef";
  char line[2 * kRowBytes + 1];
  line[2 * kRowBytes] = 0;
  for (int y = 0; y < papercolor::kHeight; ++y) {
    for (int i = 0; i < kRowBytes; ++i) {
      const uint8_t b = mono[y * kRowBytes + i];
      line[2 * i] = kHex[b >> 4];
      line[2 * i + 1] = kHex[b & 15];
    }
    printf("PAPERCOLOR_FRAME_ROW %03d %s\n", y, line);
  }
#endif
}

void flushFrame(lv_display_t* lv_display, const lv_area_t*, uint8_t* px_map) {
  if (g_display != nullptr) {
    g_display->loadFrame(px_map + kLvPalette, papercolor::kMonoBytes);
  }
  reportFrame(px_map + kLvPalette);
  pc_ui_mark_flushed();
  lv_display_flush_ready(lv_display);
}

uint32_t tickMs() {
  return static_cast<uint32_t>(esp_timer_get_time() / 1000);
}

bool initScreens(papercolor::Display& display) {
  g_display = &display;
  lv_init();
  lv_tick_set_cb(tickMs);
  // The render target lives in PSRAM like the driver's own framebuffer:
  // internal SRAM is the model's (its method arena must land there), and a
  // 30 KB frame written once per card costs nothing measurable.
  const size_t size = kLvPalette + papercolor::kMonoBytes + 64;
  void* buffer = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (buffer == nullptr) return false;
  lv_display_t* lv_display = lv_display_create(PC_UI_WIDTH, PC_UI_HEIGHT);
  if (lv_display == nullptr) return false;
  lv_display_set_color_format(lv_display, LV_COLOR_FORMAT_I1);
  lv_display_set_buffers(lv_display, buffer, nullptr, size,
                         LV_DISPLAY_RENDER_MODE_FULL);
  lv_display_set_flush_cb(lv_display, flushFrame);
  pc_ui_init(lv_display);
  return true;
}

// Returns false when the frame never reached the panel framebuffer.
bool drawStatus(const StatusMessage& message) {
  if (message.kind == RequestKind::kStory) {
    pc_ui_show_story(message.title, message.story);
  } else if (message.kind == RequestKind::kStoryPage) {
    pc_ui_show_story_page(message.title, message.corner, message.story);
  } else if (message.kind != RequestKind::kText) {
    pc_ui_show_tool(message.title, message.action, message.argument,
                    message.result, message.detail);
  } else {
    pc_ui_show_text(message.title, message.action, message.argument);
  }
  return pc_ui_render();
}

nvs_handle_t openBalanceCounter(uint32_t* interrupted_refreshes) {
  *interrupted_refreshes = 0;
  const esp_err_t init_error = nvs_flash_init();
  if (init_error != ESP_OK) {
    printf("PAPERCOLOR_STATUS nvs=DISABLED init_error=%d\n", init_error);
    return 0;
  }
  nvs_handle_t handle = 0;
  const esp_err_t open_error =
      nvs_open(kNvsNamespace, NVS_READWRITE, &handle);
  if (open_error != ESP_OK) {
    printf("PAPERCOLOR_STATUS nvs=DISABLED open_error=%d\n", open_error);
    return 0;
  }
  const esp_err_t read_error =
      nvs_get_u32(handle, kNvsFastCount, interrupted_refreshes);
  if (read_error != ESP_OK && read_error != ESP_ERR_NVS_NOT_FOUND) {
    printf("PAPERCOLOR_STATUS nvs=READ_FAIL error=%d\n", read_error);
    *interrupted_refreshes = kFastRefreshBalanceLimit;
  }
  printf("PAPERCOLOR_STATUS balance_count=%" PRIu32 " limit=%" PRIu32 "\n",
         *interrupted_refreshes, kFastRefreshBalanceLimit);
  return handle;
}

void storeBalanceCounter(nvs_handle_t handle, uint32_t value) {
  if (handle == 0) return;
  esp_err_t error = nvs_set_u32(handle, kNvsFastCount, value);
  if (error == ESP_OK) error = nvs_commit(handle);
  if (error != ESP_OK) {
    printf("PAPERCOLOR_STATUS nvs=WRITE_FAIL error=%d\n", error);
  }
}

// Battery level, printed at start, after every refresh and once a minute
// when idle: the board stops when its battery is flat, USB or not.
void reportPower(papercolor::Display& display) {
  uint16_t vbat_mv = 0, vin_mv = 0;
  uint8_t source = 0;
  const esp_err_t error = display.readPower(&vbat_mv, &vin_mv, &source);
  if (error == ESP_OK) {
    printf("PAPERCOLOR_POWER vbat_mv=%u vin_mv=%u usb=%s battery_node=%s\n",
           (unsigned)vbat_mv, (unsigned)vin_mv,
           (source & 0x01) ? "YES" : "NO", (source & 0x04) ? "YES" : "NO");
  } else {
    printf("PAPERCOLOR_POWER result=FAIL error=%d\n", error);
  }
}

constexpr TickType_t kPowerReportTicks = pdMS_TO_TICKS(60'000);

void statusTask(void*) {
  uint32_t interrupted_refreshes = 0;
  const nvs_handle_t balance_counter =
      openBalanceCounter(&interrupted_refreshes);
  papercolor::Display display;
  const esp_err_t begin_error = display.begin(20'000'000);
  if (begin_error != ESP_OK) {
    printf("PAPERCOLOR_STATUS init=FAIL error=%d\n", begin_error);
    g_failed.store(true, std::memory_order_release);
    vTaskDelete(nullptr);
    return;
  }
  if (!initScreens(display)) {
    printf("PAPERCOLOR_STATUS init=FAIL lvgl=NO_MEM\n");
    g_failed.store(true, std::memory_order_release);
    vTaskDelete(nullptr);
    return;
  }
  g_ready.store(true, std::memory_order_release);
  printf("PAPERCOLOR_STATUS READY spi_hz=20000000 ui=lvgl\n");
  reportPower(display);

  StatusMessage message = {};
  int64_t last_complete_us = esp_timer_get_time();
  for (;;) {
    if (xQueueReceive(g_queue, &message, kPowerReportTicks) != pdTRUE) {
      reportPower(display);
      continue;
    }
    const esp_err_t tool_error = resolveRequest(display, &message);
    const bool drawn = drawStatus(message);
    int32_t body_needed = 0;
    int32_t body_available = 0;
    pc_ui_last_fit(&body_needed, &body_available);
    papercolor::RefreshTiming timing = {};
    const bool balance_due =
        interrupted_refreshes >= kFastRefreshBalanceLimit ||
        esp_timer_get_time() - last_complete_us >= 3'600'000'000LL;
    papercolor::RefreshMode mode = papercolor::RefreshMode::kFastInterrupted;
    const char* mode_name = "fast";
    if (balance_due || !message.fast) {
      mode = papercolor::RefreshMode::kComplete;
      mode_name = "complete";
    }
    esp_err_t display_error = ESP_FAIL;
    if (drawn) display_error = display.present(mode, g_fast_cutoff_ms.load(std::memory_order_acquire), &timing);
    if (display_error == ESP_OK) {
      if (mode == papercolor::RefreshMode::kComplete) {
        interrupted_refreshes = 0;
        last_complete_us = esp_timer_get_time();
      } else {
        ++interrupted_refreshes;
      }
      storeBalanceCounter(balance_counter, interrupted_refreshes);
    }
    const bool success = tool_error == ESP_OK && display_error == ESP_OK;
    const char* result_name = "FAIL";
    if (success) result_name = "PASS";
    printf("PAPERCOLOR_STATUS seq=%" PRIu32 " result=%s mode=%s "
           "total_us=%" PRIu32 " pack_us=%" PRIu32
           " transfer_us=%" PRIu32 " body_px=%" PRId32 "/%" PRId32 " cutoff_ms=%u\n",
           message.sequence, result_name, mode_name, timing.total_us,
           timing.pack_us, timing.transfer_us, body_needed, body_available,
           mode == papercolor::RefreshMode::kComplete ? 0u
               : (unsigned)g_fast_cutoff_ms.load(std::memory_order_acquire));
    reportPower(display);  // just after the refresh's current peak
    if (tool_error != ESP_OK || display_error != ESP_OK) {
      g_failed.store(true, std::memory_order_release);
    }
    g_completed.store(message.sequence, std::memory_order_release);
  }
}

}  // namespace

bool start() {
  if (g_started.exchange(true, std::memory_order_acq_rel)) {
    return g_ready.load(std::memory_order_acquire) &&
           !g_failed.load(std::memory_order_acquire);
  }
  // The 1.5 KB message slot lives in PSRAM: only tasks touch it, never an ISR
  // or a cache-disabled flash write. The task's stack stays internal (the
  // balance counter's NVS commits run on it).
  g_queue = xQueueCreateWithCaps(1, sizeof(StatusMessage),
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (g_queue == nullptr) {
    g_failed.store(true, std::memory_order_release);
    return false;
  }
  if (xTaskCreatePinnedToCore(statusTask, "papercolor", 12288, nullptr,
                              tskIDLE_PRIORITY + 1, nullptr, 1) != pdPASS) {
    vQueueDeleteWithCaps(g_queue);
    g_queue = nullptr;
    g_failed.store(true, std::memory_order_release);
    return false;
  }

  const int64_t deadline = esp_timer_get_time() + 3'000'000;
  while (!g_ready.load(std::memory_order_acquire) &&
         !g_failed.load(std::memory_order_acquire) &&
         esp_timer_get_time() < deadline) {
    vTaskDelay(1);
  }
  return g_ready.load(std::memory_order_acquire) &&
         !g_failed.load(std::memory_order_acquire);
}

bool post(const char* title, const char* line1, const char* line2) {
  if (!g_ready.load(std::memory_order_acquire) ||
      g_failed.load(std::memory_order_acquire)) {
    return false;
  }
  StatusMessage message = {};
  message.sequence = g_submitted.fetch_add(1, std::memory_order_acq_rel) + 1;
  message.fast = g_fast_allowed.load(std::memory_order_acquire);
  message.kind = RequestKind::kText;
  copyText(message.title, sizeof(message.title), title);
  copyText(message.action, sizeof(message.action), line1);
  copyText(message.argument, sizeof(message.argument), line2);
  return xQueueOverwrite(g_queue, &message) == pdPASS;
}

bool postSensorReading(const char* sensor) {
  if (!g_ready.load(std::memory_order_acquire) ||
      g_failed.load(std::memory_order_acquire)) {
    return false;
  }
  StatusMessage message = {};
  message.sequence = g_submitted.fetch_add(1, std::memory_order_acq_rel) + 1;
  message.fast = g_fast_allowed.load(std::memory_order_acquire);
  message.kind = RequestKind::kReadHumidity;
  copyText(message.title, sizeof(message.title), "NANOGLIMMER");
  copyText(message.action, sizeof(message.action), "call read_sensor");
  snprintf(message.argument, sizeof(message.argument), "sensor = %s",
           sensor ? sensor : "humidity");
  return xQueueOverwrite(g_queue, &message) == pdPASS;
}

bool postHumidityReading() { return postSensorReading("humidity"); }

bool postTool(const char* title, const char* action, const char* argument,
              const char* result, const char* detail) {
  if (!g_ready.load(std::memory_order_acquire) ||
      g_failed.load(std::memory_order_acquire)) {
    return false;
  }
  StatusMessage message = {};
  message.sequence = g_submitted.fetch_add(1, std::memory_order_acq_rel) + 1;
  message.fast = g_fast_allowed.load(std::memory_order_acquire);
  message.kind = RequestKind::kTool;
  copyText(message.title, sizeof(message.title), title);
  copyText(message.action, sizeof(message.action), action);
  copyText(message.argument, sizeof(message.argument), argument);
  copyText(message.result, sizeof(message.result), result);
  copyText(message.detail, sizeof(message.detail), detail);
  return xQueueOverwrite(g_queue, &message) == pdPASS;
}

bool postStoryPage(const char* title, const char* corner, const char* text, bool fast) {
  if (!g_ready.load(std::memory_order_acquire) ||
      g_failed.load(std::memory_order_acquire)) {
    return false;
  }
  StatusMessage message = {};
  message.sequence = g_submitted.fetch_add(1, std::memory_order_acq_rel) + 1;
  message.kind = RequestKind::kStoryPage;
  message.fast = fast;
  copyText(message.title, sizeof(message.title), title);
  copyText(message.corner, sizeof(message.corner), corner);
  copyText(message.story, sizeof(message.story), text);
  return xQueueOverwrite(g_queue, &message) == pdPASS;
}

int paginateStory(const char* text, int* starts, int max_pages) {
  return pc_ui_story_paginate(text, starts, max_pages);
}

int storyLastFill(const char* text, const int* starts, int pages) {
  return pc_ui_story_last_fill(text, starts, pages);
}

void setStoryFontPx(int px) { pc_ui_set_story_font_px(px); }

int storyFontPx() { return pc_ui_story_font_px(); }

bool postStory(const char* title, const char* text) {
  if (!g_ready.load(std::memory_order_acquire) ||
      g_failed.load(std::memory_order_acquire)) {
    return false;
  }
  StatusMessage message = {};
  message.sequence = g_submitted.fetch_add(1, std::memory_order_acq_rel) + 1;
  message.fast = g_fast_allowed.load(std::memory_order_acquire);
  message.kind = RequestKind::kStory;
  copyText(message.title, sizeof(message.title), title);
  copyText(message.story, sizeof(message.story), text);
  return xQueueOverwrite(g_queue, &message) == pdPASS;
}

void setFastCutoffMs(uint16_t ms) {
  if (ms < 100) ms = 100;
  if (ms > 2000) ms = 2000;
  g_fast_cutoff_ms.store(ms, std::memory_order_release);
}

uint16_t fastCutoffMs() { return g_fast_cutoff_ms.load(std::memory_order_acquire); }

void setFastRefresh(bool allowed) {
  g_fast_allowed.store(allowed, std::memory_order_release);
}

bool fastRefresh() { return g_fast_allowed.load(std::memory_order_acquire); }

bool flush(uint32_t timeout_ms) {
  const uint32_t target = g_submitted.load(std::memory_order_acquire);
  const int64_t deadline =
      esp_timer_get_time() + timeout_ms * 1000LL;
  while (g_completed.load(std::memory_order_acquire) < target &&
         !g_failed.load(std::memory_order_acquire) &&
         esp_timer_get_time() < deadline) {
    vTaskDelay(1);
  }
  return g_completed.load(std::memory_order_acquire) >= target &&
         !g_failed.load(std::memory_order_acquire);
}

}  // namespace et_papercolor
