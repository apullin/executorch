/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "esp_err.h"

namespace papercolor {

constexpr uint16_t kWidth = 400;
constexpr uint16_t kHeight = 600;
constexpr size_t kMonoBytes = static_cast<size_t>(kWidth) * kHeight / 8;
constexpr size_t kPanelBytes = static_cast<size_t>(kWidth) * kHeight / 2;

enum class RefreshMode : uint8_t {
  kComplete,
  kFastInterrupted,
};

struct RefreshTiming {
  // CPU time spent expanding the 1-bpp source. It is overlapped with DMA and
  // therefore is already contained in transfer_us.
  uint32_t pack_us = 0;
  // Wall time from the RAM-write command through completion of streamed DMA.
  uint32_t transfer_us = 0;
  uint32_t power_on_us = 0;
  uint32_t prepare_us = 0;
  uint32_t drive_us = 0;
  uint32_t recovery_us = 0;
  uint32_t total_us = 0;
};

// Native ESP-IDF driver for the M5Stack PaperColor's ED2208-DOA/EL040EF1.
// The framebuffer is one bit per pixel: true is paper, false is ink.
class Display {
 public:
  Display() = default;
  Display(const Display&) = delete;
  Display& operator=(const Display&) = delete;

  esp_err_t begin(uint32_t spi_hz = 20'000'000);
  esp_err_t setSpiClockHz(uint32_t spi_hz);
  uint32_t spiClockHz() const { return spi_hz_; }

  void clear(bool paper = true);
  // Replace the framebuffer with a rendered frame in the same layout: one
  // bit per pixel, most significant bit first, 50 bytes per row, 1 = paper.
  // Short frames leave the remainder untouched.
  void loadFrame(const uint8_t* mono, size_t size);
  void setPixel(int x, int y, bool paper);
  void fillRect(int x, int y, int width, int height, bool paper);
  void drawRect(int x, int y, int width, int height, int thickness = 1,
                bool paper = false);
  void drawText(int x, int y, const char* text, int scale = 1,
                bool paper = false);

  // PaperColor's onboard SHT40 shares the system I2C bus with M5PM1. Both
  // returned words are CRC-checked before conversion.
  esp_err_t readEnvironment(float* temperature_c, float* humidity_pct);

  // Battery and USB input as M5PM1 reports them: millivolts, plus its power
  // source bitmap (bit0 USB 5V in, bit1 5V in/out, bit2 battery node).
  // The board runs from the battery and USB only charges it, so a flat
  // battery stops the board even while it is plugged in.
  esp_err_t readPower(uint16_t* vbat_mv, uint16_t* vin_mv, uint8_t* source);

  // Upload controller RAM without driving the panel. This is used to measure
  // the SPI portion independently from electrophoretic waveform time.
  esp_err_t uploadOnly(RefreshTiming* timing = nullptr);

  // Fast mode deliberately interrupts the controller's OTP waveform after
  // cutoff_ms. It produces black/yellow, not true black/white, and is not
  // DC-balanced. Call complete mode periodically.
  esp_err_t present(RefreshMode mode, uint16_t cutoff_ms = 340,
                    RefreshTiming* timing = nullptr);

 private:
  esp_err_t initSystemI2c();
  esp_err_t enablePanelPower();
  esp_err_t pmicRead(uint8_t reg, uint8_t* value);
  esp_err_t pmicWrite(uint8_t reg, uint8_t value);
  esp_err_t pmicUpdate(uint8_t reg, uint8_t clear_mask, uint8_t set_mask);

  esp_err_t initSpiBus();
  esp_err_t addSpiDevice(uint32_t spi_hz);
  esp_err_t writeByte(uint8_t value);
  esp_err_t writeBuffer(const void* data, size_t size, bool psram_dma);
  esp_err_t command(uint8_t value, const uint8_t* data = nullptr,
                    size_t size = 0);

  void resetController();
  esp_err_t initController();
  esp_err_t waitIdle(uint32_t timeout_ms);
  esp_err_t waitBusyAssert(uint32_t timeout_ms, uint32_t* elapsed_us);
  esp_err_t waitBusyComplete(uint32_t timeout_ms, uint32_t* elapsed_us);
  esp_err_t powerOn();
  esp_err_t powerOff();
  esp_err_t startRefresh(RefreshMode mode);
  void packChunk(size_t source_offset, size_t source_size,
                 uint8_t* destination);

  uint8_t* mono_ = nullptr;
  uint8_t* stream_[2] = {};
  i2c_master_bus_handle_t i2c_bus_ = nullptr;
  i2c_master_dev_handle_t pmic_ = nullptr;
  i2c_master_dev_handle_t sht40_ = nullptr;
  spi_device_handle_t spi_ = nullptr;
  uint32_t spi_hz_ = 0;
  bool spi_bus_ready_ = false;
  bool panel_power_on_ = false;
};

}  // namespace papercolor
