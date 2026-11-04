/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "papercolor_display.h"

#include <algorithm>
#include <array>
#include <ctype.h>
#include <string.h>
#include <utility>

#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace papercolor {
namespace {

constexpr char kTag[] = "papercolor";

constexpr gpio_num_t kPinSclk = GPIO_NUM_15;
constexpr gpio_num_t kPinMosi = GPIO_NUM_13;
constexpr gpio_num_t kPinMiso = GPIO_NUM_14;
constexpr gpio_num_t kPinCs = GPIO_NUM_44;
constexpr gpio_num_t kPinDc = GPIO_NUM_43;
constexpr gpio_num_t kPinReset = GPIO_NUM_12;
constexpr gpio_num_t kPinBusy = GPIO_NUM_11;
constexpr gpio_num_t kPinSdCs = GPIO_NUM_47;
constexpr gpio_num_t kPinSda = GPIO_NUM_3;
constexpr gpio_num_t kPinScl = GPIO_NUM_2;

constexpr uint8_t kPmicAddress = 0x6e;
constexpr uint8_t kSht40Address = 0x44;
constexpr uint8_t kSht40MeasureHighPrecision = 0xfd;
constexpr uint8_t kPmicPowerConfig = 0x06;
constexpr uint8_t kPmicI2cConfig = 0x09;
constexpr uint8_t kPmicWatchdog = 0x0a;
constexpr uint8_t kPmicGpioMode = 0x10;
constexpr uint8_t kPmicGpioOut = 0x11;
constexpr uint8_t kPmicGpioDrive = 0x13;
constexpr uint8_t kPmicGpioFunction0 = 0x16;
constexpr uint8_t kPmicNeoConfig = 0x50;
constexpr uint8_t kPmicChargeEnable = 1u << 0;
constexpr uint8_t kPmicDcdcEnable = 1u << 1;
constexpr uint8_t kPmicLdoEnable = 1u << 2;
constexpr uint8_t kPmicBoostEnable = 1u << 3;
constexpr uint8_t kPmicGpio0 = 1u << 0;

constexpr uint8_t kPanelBlack = 0x0;
constexpr uint8_t kPanelWhite = 0x1;
constexpr uint8_t kFastDisplayControl = 0x0f;
constexpr uint8_t kCompleteDisplayControl = 0x3f;
constexpr uint32_t kPmicTransferDelayUs = 500;
constexpr uint32_t kBusySettleMs = 20;
constexpr uint32_t kMaxSpiHz = 20'000'000;
// ESP32-S3's SPI DMA length register is 18 bits, so one transaction is
// limited to 32 KiB. Use smaller, cache-line-aligned chunks; CS remains low
// around the complete frame in uploadOnly().
constexpr size_t kSpiDmaChunkBytes = 16 * 1024;
constexpr size_t kStreamPanelBytes = 4 * 1024;
constexpr size_t kStreamMonoBytes = kStreamPanelBytes / sizeof(uint32_t);

constexpr uint32_t expandMonoByte(uint8_t pixels) {
  uint32_t expanded = 0;
  for (int pair = 0; pair < 4; ++pair) {
    const int shift = 6 - pair * 2;
    const uint8_t left = (pixels & (1u << (shift + 1))) ? kPanelWhite
                                                        : kPanelBlack;
    const uint8_t right = (pixels & (1u << shift)) ? kPanelWhite
                                                   : kPanelBlack;
    const uint8_t panel_byte = static_cast<uint8_t>((left << 4) | right);
    // ESP32-S3 is little-endian: this makes a uint32_t store emit the same
    // four-byte order as the scalar panel stream.
    expanded |= static_cast<uint32_t>(panel_byte) << (pair * 8);
  }
  return expanded;
}

template <size_t... Index>
constexpr std::array<uint32_t, sizeof...(Index)> makeMonoExpansionTable(
    std::index_sequence<Index...>) {
  return {expandMonoByte(static_cast<uint8_t>(Index))...};
}

constexpr auto kMonoExpansion =
    makeMonoExpansionTable(std::make_index_sequence<256>{});
static_assert(kMonoExpansion[0x00] == 0x00000000);
static_assert(kMonoExpansion[0xff] == 0x11111111);
static_assert(kMonoExpansion[0x80] == 0x00000010);

struct Glyph {
  char character;
  uint8_t rows[7];
};

// Deliberately small UI font. Each row uses the low five bits, MSB on the left.
constexpr Glyph kGlyphs[] = {
    {' ', {0, 0, 0, 0, 0, 0, 0}},
    {'%', {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13}},
    {'-', {0, 0, 0, 0x1f, 0, 0, 0}},
    {'.', {0, 0, 0, 0, 0, 0x06, 0x06}},
    {':', {0, 0x06, 0x06, 0, 0x06, 0x06, 0}},
    {'=', {0, 0x1f, 0, 0x1f, 0, 0, 0}},
    {'_', {0, 0, 0, 0, 0, 0, 0x1f}},
    {'/', {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}},
    {'0', {0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e}},
    {'1', {0x04, 0x0c, 0x14, 0x04, 0x04, 0x04, 0x1f}},
    {'2', {0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f}},
    {'3', {0x1e, 0x01, 0x01, 0x0e, 0x01, 0x01, 0x1e}},
    {'4', {0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02}},
    {'5', {0x1f, 0x10, 0x10, 0x1e, 0x01, 0x01, 0x1e}},
    {'6', {0x0e, 0x10, 0x10, 0x1e, 0x11, 0x11, 0x0e}},
    {'7', {0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e}},
    {'9', {0x0e, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x0e}},
    {'A', {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}},
    {'B', {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e}},
    {'C', {0x0f, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0f}},
    {'D', {0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e}},
    {'E', {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f}},
    {'F', {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10}},
    {'G', {0x0f, 0x10, 0x10, 0x17, 0x11, 0x11, 0x0f}},
    {'H', {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}},
    {'I', {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1f}},
    {'J', {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0c}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f}},
    {'M', {0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11}},
    {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'O', {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}},
    {'P', {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10}},
    {'Q', {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d}},
    {'R', {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11}},
    {'S', {0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e}},
    {'T', {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
    {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04}},
    {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a}},
    {'X', {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11}},
    {'Y', {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04}},
    {'Z', {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f}},
};

const Glyph* findGlyph(char value) {
  const char wanted = static_cast<char>(toupper(static_cast<unsigned char>(value)));
  for (const auto& glyph : kGlyphs) {
    if (glyph.character == wanted) return &glyph;
  }
  return &kGlyphs[0];
}

uint32_t elapsedUs(int64_t start) {
  return static_cast<uint32_t>(esp_timer_get_time() - start);
}

uint8_t sensirionCrc8(const uint8_t* bytes, size_t size) {
  uint8_t crc = 0xff;
  for (size_t index = 0; index < size; ++index) {
    crc ^= bytes[index];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 0x80) {
        crc = (crc << 1) ^ 0x31;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

uint16_t readBigEndianWord(const uint8_t* bytes) {
  return (bytes[0] << 8) | bytes[1];
}

float sht40TemperatureC(uint16_t raw) {
  return -45.0f + 175.0f * raw / 65535.0f;
}

float sht40HumidityPercent(uint16_t raw) {
  float humidity = -6.0f + 125.0f * raw / 65535.0f;
  if (humidity < 0.0f) humidity = 0.0f;
  if (humidity > 100.0f) humidity = 100.0f;
  return humidity;
}

}  // namespace

esp_err_t Display::pmicWrite(uint8_t reg, uint8_t value) {
  const uint8_t bytes[2] = {reg, value};
  ESP_RETURN_ON_ERROR(i2c_master_transmit(pmic_, bytes, sizeof(bytes), 20),
                      kTag, "PMIC write 0x%02x", reg);
  esp_rom_delay_us(kPmicTransferDelayUs);
  return ESP_OK;
}

esp_err_t Display::pmicRead(uint8_t reg, uint8_t* value) {
  ESP_RETURN_ON_FALSE(value != nullptr, ESP_ERR_INVALID_ARG, kTag,
                      "null PMIC read destination");
  // M5PM1 is a PY32 firmware I2C target, not a conventional register file.
  // It needs a delay after the address-pointer write before it serves data.
  ESP_RETURN_ON_ERROR(i2c_master_transmit(pmic_, &reg, 1, 20), kTag,
                      "PMIC pointer 0x%02x", reg);
  esp_rom_delay_us(kPmicTransferDelayUs);
  ESP_RETURN_ON_ERROR(i2c_master_receive(pmic_, value, 1, 20), kTag,
                      "PMIC read 0x%02x", reg);
  return ESP_OK;
}

esp_err_t Display::pmicUpdate(uint8_t reg, uint8_t clear_mask,
                              uint8_t set_mask) {
  uint8_t value = 0;
  ESP_RETURN_ON_ERROR(pmicRead(reg, &value), kTag, "PMIC RMW read");
  value = static_cast<uint8_t>((value & ~clear_mask) | set_mask);
  return pmicWrite(reg, value);
}

esp_err_t Display::initSystemI2c() {
  i2c_master_bus_config_t bus = {};
  bus.i2c_port = I2C_NUM_0;
  bus.sda_io_num = kPinSda;
  bus.scl_io_num = kPinScl;
  bus.clk_source = I2C_CLK_SRC_DEFAULT;
  bus.glitch_ignore_cnt = 7;
  bus.flags.enable_internal_pullup = true;
  ESP_RETURN_ON_ERROR(i2c_new_master_bus(&bus, &i2c_bus_), kTag,
                      "system I2C init");

  i2c_device_config_t device = {};
  device.dev_addr_length = I2C_ADDR_BIT_LEN_7;
  device.device_address = kPmicAddress;
  device.scl_speed_hz = 100'000;
  ESP_RETURN_ON_ERROR(i2c_master_bus_add_device(i2c_bus_, &device, &pmic_),
                      kTag, "PMIC attach");
  device.device_address = kSht40Address;
  device.scl_speed_hz = 400'000;
  ESP_RETURN_ON_ERROR(i2c_master_bus_add_device(i2c_bus_, &device, &sht40_),
                      kTag, "SHT40 attach");
  return ESP_OK;
}

esp_err_t Display::readEnvironment(float* temperature_c,
                                   float* humidity_pct) {
  ESP_RETURN_ON_FALSE(temperature_c != nullptr && humidity_pct != nullptr,
                      ESP_ERR_INVALID_ARG, kTag,
                      "null SHT40 result destination");
  ESP_RETURN_ON_FALSE(sht40_ != nullptr, ESP_ERR_INVALID_STATE, kTag,
                      "SHT40 bus device is not initialized");
  ESP_RETURN_ON_ERROR(
      i2c_master_transmit(sht40_, &kSht40MeasureHighPrecision, 1, 20),
      kTag, "SHT40 high-precision measurement command");
  // Sensirion specifies an 8.3 ms maximum conversion time for this command.
  vTaskDelay(pdMS_TO_TICKS(10));
  uint8_t response[6] = {};
  ESP_RETURN_ON_ERROR(i2c_master_receive(sht40_, response, sizeof(response),
                                         20),
                      kTag, "SHT40 measurement read");
  ESP_RETURN_ON_FALSE(sensirionCrc8(response, 2) == response[2] &&
                          sensirionCrc8(response + 3, 2) == response[5],
                      ESP_ERR_INVALID_CRC, kTag, "SHT40 response CRC");

  *temperature_c = sht40TemperatureC(readBigEndianWord(response));
  *humidity_pct = sht40HumidityPercent(readBigEndianWord(response + 3));
  return ESP_OK;
}

esp_err_t Display::readPower(uint16_t* vbat_mv, uint16_t* vin_mv,
                             uint8_t* source) {
  ESP_RETURN_ON_FALSE(vbat_mv != nullptr && vin_mv != nullptr &&
                          source != nullptr,
                      ESP_ERR_INVALID_ARG, kTag, "null power destination");
  // Register map from M5Stack's M5Unified M5PM1 driver: PWR_SRC 0x04, VBAT
  // 0x22/0x23 and VIN 0x24/0x25, little-endian millivolts.
  uint8_t bytes[4] = {};
  ESP_RETURN_ON_ERROR(pmicRead(0x04, source), kTag, "PMIC power source");
  for (int i = 0; i < 4; ++i) {
    ESP_RETURN_ON_ERROR(pmicRead(static_cast<uint8_t>(0x22 + i), &bytes[i]),
                        kTag, "PMIC voltage");
  }
  *vbat_mv = static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
  *vin_mv = static_cast<uint16_t>(bytes[2] | (bytes[3] << 8));
  return ESP_OK;
}

esp_err_t Display::enablePanelPower() {
  ESP_RETURN_ON_ERROR(pmicWrite(kPmicI2cConfig, 0x00), kTag,
                      "disable PMIC I2C idle sleep");
  ESP_RETURN_ON_ERROR(pmicWrite(kPmicWatchdog, 0x00), kTag,
                      "disable PMIC watchdog");
  ESP_RETURN_ON_ERROR(
      pmicUpdate(kPmicPowerConfig, kPmicLdoEnable,
                 kPmicChargeEnable | kPmicDcdcEnable | kPmicBoostEnable),
      kTag, "PMIC power policy");
  ESP_RETURN_ON_ERROR(pmicWrite(kPmicNeoConfig, 0x00), kTag,
                      "disable PMIC LED engine");

  // PMIC GPIO0 is the board's EPD_EN. The function register uses a two-bit
  // field, hence 0x03 rather than only bit zero.
  ESP_RETURN_ON_ERROR(pmicUpdate(kPmicGpioFunction0, 0x03, 0), kTag,
                      "EPD GPIO function");
  ESP_RETURN_ON_ERROR(pmicUpdate(kPmicGpioMode, 0, kPmicGpio0), kTag,
                      "EPD GPIO output mode");
  ESP_RETURN_ON_ERROR(pmicUpdate(kPmicGpioDrive, kPmicGpio0, 0), kTag,
                      "EPD GPIO push-pull");
  ESP_RETURN_ON_ERROR(pmicUpdate(kPmicGpioOut, 0, kPmicGpio0), kTag,
                      "EPD rail enable");
  vTaskDelay(pdMS_TO_TICKS(100));
  return ESP_OK;
}

esp_err_t Display::initSpiBus() {
  spi_bus_config_t bus = {};
  bus.mosi_io_num = kPinMosi;
  bus.miso_io_num = kPinMiso;
  bus.sclk_io_num = kPinSclk;
  bus.quadwp_io_num = -1;
  bus.quadhd_io_num = -1;
  bus.max_transfer_sz = kStreamPanelBytes;
  ESP_RETURN_ON_ERROR(spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO),
                      kTag, "display SPI bus init");
  spi_bus_ready_ = true;
  return ESP_OK;
}

esp_err_t Display::addSpiDevice(uint32_t spi_hz) {
  ESP_RETURN_ON_FALSE(spi_hz > 0 && spi_hz <= kMaxSpiHz,
                      ESP_ERR_INVALID_ARG, kTag,
                      "SPI clock must be 1..20 MHz");
  spi_device_interface_config_t device = {};
  device.clock_speed_hz = static_cast<int>(spi_hz);
  device.mode = 0;
  device.spics_io_num = -1;  // CS is held across command+payload manually.
  device.queue_size = 2;
  ESP_RETURN_ON_ERROR(spi_bus_add_device(SPI2_HOST, &device, &spi_), kTag,
                      "display SPI device attach");
  spi_hz_ = spi_hz;
  return ESP_OK;
}

esp_err_t Display::setSpiClockHz(uint32_t spi_hz) {
  if (spi_ != nullptr && spi_hz == spi_hz_) return ESP_OK;
  if (spi_ != nullptr) {
    ESP_RETURN_ON_ERROR(spi_bus_remove_device(spi_), kTag,
                        "display SPI device remove");
    spi_ = nullptr;
  }
  return addSpiDevice(spi_hz);
}

esp_err_t Display::writeByte(uint8_t value) {
  spi_transaction_t transaction = {};
  transaction.flags = SPI_TRANS_USE_TXDATA;
  transaction.length = 8;
  transaction.tx_data[0] = value;
  return spi_device_polling_transmit(spi_, &transaction);
}

esp_err_t Display::writeBuffer(const void* data, size_t size,
                               bool psram_dma) {
  ESP_RETURN_ON_FALSE(data != nullptr || size == 0, ESP_ERR_INVALID_ARG, kTag,
                      "null SPI payload");
  if (size == 0) return ESP_OK;
  const auto* cursor = static_cast<const uint8_t*>(data);
  while (size != 0) {
    const size_t chunk_size = std::min(size, kSpiDmaChunkBytes);
    spi_transaction_t transaction = {};
    transaction.length = chunk_size * 8;
    transaction.tx_buffer = cursor;
    if (psram_dma) transaction.flags = SPI_TRANS_DMA_USE_PSRAM;
    ESP_RETURN_ON_ERROR(spi_device_polling_transmit(spi_, &transaction), kTag,
                        "display SPI payload chunk");
    cursor += chunk_size;
    size -= chunk_size;
  }
  return ESP_OK;
}

esp_err_t Display::command(uint8_t value, const uint8_t* data, size_t size) {
  ESP_RETURN_ON_ERROR(spi_device_acquire_bus(spi_, portMAX_DELAY), kTag,
                      "display SPI acquire");
  gpio_set_level(kPinCs, 0);
  gpio_set_level(kPinDc, 0);
  esp_err_t error = writeByte(value);
  gpio_set_level(kPinDc, 1);
  for (size_t index = 0; error == ESP_OK && index < size; ++index) {
    error = writeByte(data[index]);
  }
  gpio_set_level(kPinCs, 1);
  spi_device_release_bus(spi_);
  return error;
}

void Display::resetController() {
  gpio_set_level(kPinReset, 1);
  vTaskDelay(pdMS_TO_TICKS(10));
  gpio_set_level(kPinReset, 0);
  vTaskDelay(pdMS_TO_TICKS(10));
  gpio_set_level(kPinReset, 1);
  vTaskDelay(pdMS_TO_TICKS(10));
}

esp_err_t Display::waitIdle(uint32_t timeout_ms) {
  const int64_t start = esp_timer_get_time();
  while (gpio_get_level(kPinBusy) == 0) {
    if (elapsedUs(start) > timeout_ms * 1000u) return ESP_ERR_TIMEOUT;
    vTaskDelay(1);
  }
  return ESP_OK;
}

esp_err_t Display::waitBusyAssert(uint32_t timeout_ms, uint32_t* elapsed_us) {
  const int64_t start = esp_timer_get_time();
  while (gpio_get_level(kPinBusy) != 0) {
    if (elapsedUs(start) > timeout_ms * 1000u) return ESP_ERR_TIMEOUT;
    esp_rom_delay_us(100);
  }
  if (elapsed_us) *elapsed_us = elapsedUs(start);
  return ESP_OK;
}

esp_err_t Display::waitBusyComplete(uint32_t timeout_ms,
                                    uint32_t* elapsed_us) {
  const int64_t start = esp_timer_get_time();
  while (gpio_get_level(kPinBusy) == 0) {
    if (elapsedUs(start) > timeout_ms * 1000u) return ESP_ERR_TIMEOUT;
    vTaskDelay(1);
  }
  if (elapsed_us) *elapsed_us = elapsedUs(start);
  return ESP_OK;
}

esp_err_t Display::initController() {
  static constexpr uint8_t kCmdH[] = {0x49, 0x55, 0x20, 0x08, 0x09, 0x18};
  static constexpr uint8_t kPower[] = {0x5f, 0x69};
  static constexpr uint8_t kPwrOffSeq[] = {0x40, 0x1f, 0x1f, 0x2c};
  static constexpr uint8_t kBoosterSoftStart[] = {0x6f, 0x1f, 0x1f, 0x22};
  static constexpr uint8_t kBooster[] = {0x6f, 0x1f, 0x17, 0x17};
  static constexpr uint8_t kPanel[] = {0x03, 0x54, 0x00, 0x44};
  static constexpr uint8_t kTcon[] = {0x02, 0x00};
  static constexpr uint8_t k3f[] = {0x3f};
  static constexpr uint8_t k08[] = {0x08};
  static constexpr uint8_t k0f[] = {0x0f};
  static constexpr uint8_t k2f[] = {0x2f};
  static constexpr uint8_t k01[] = {0x01};
  static constexpr uint8_t kResolution[] = {
      static_cast<uint8_t>(kWidth >> 8), static_cast<uint8_t>(kWidth),
      static_cast<uint8_t>(kHeight >> 8), static_cast<uint8_t>(kHeight)};
  static constexpr struct {
    uint8_t command;
    const uint8_t* data;
    size_t size;
  } kInit[] = {
      {0xaa, kCmdH, sizeof(kCmdH)},
      {0x01, k3f, sizeof(k3f)},
      {0x00, kPower, sizeof(kPower)},
      {0x05, kPwrOffSeq, sizeof(kPwrOffSeq)},
      {0x08, kBoosterSoftStart, sizeof(kBoosterSoftStart)},
      {0x06, kBooster, sizeof(kBooster)},
      {0x03, kPanel, sizeof(kPanel)},
      {0x60, kTcon, sizeof(kTcon)},
      {0x30, k08, sizeof(k08)},
      {0x50, k0f, sizeof(k0f)},
      {0xe3, k2f, sizeof(k2f)},
      {0x84, k01, sizeof(k01)},
      {0x61, kResolution, sizeof(kResolution)},
  };

  // The official M5GFX driver checks BUSY once before this entire list. The
  // interrupted community driver checks before every command with a 100 ms
  // future-assertion grace period, which can add over a second after each cut.
  ESP_RETURN_ON_ERROR(waitIdle(30'000), kTag, "controller idle before init");
  for (const auto& item : kInit) {
    ESP_RETURN_ON_ERROR(command(item.command, item.data, item.size), kTag,
                        "controller init command 0x%02x", item.command);
  }
  return ESP_OK;
}

esp_err_t Display::begin(uint32_t spi_hz) {
  gpio_config_t outputs = {};
  outputs.pin_bit_mask = (1ULL << kPinCs) | (1ULL << kPinDc) |
                         (1ULL << kPinReset) | (1ULL << kPinSdCs);
  outputs.mode = GPIO_MODE_OUTPUT;
  outputs.pull_up_en = GPIO_PULLUP_DISABLE;
  outputs.pull_down_en = GPIO_PULLDOWN_DISABLE;
  outputs.intr_type = GPIO_INTR_DISABLE;
  ESP_RETURN_ON_ERROR(gpio_config(&outputs), kTag, "display output GPIOs");
  gpio_set_level(kPinCs, 1);
  gpio_set_level(kPinDc, 1);
  gpio_set_level(kPinReset, 1);
  gpio_set_level(kPinSdCs, 1);

  gpio_config_t busy = {};
  busy.pin_bit_mask = 1ULL << kPinBusy;
  busy.mode = GPIO_MODE_INPUT;
  busy.pull_up_en = GPIO_PULLUP_ENABLE;
  busy.pull_down_en = GPIO_PULLDOWN_DISABLE;
  busy.intr_type = GPIO_INTR_DISABLE;
  ESP_RETURN_ON_ERROR(gpio_config(&busy), kTag, "display BUSY GPIO");

  mono_ = static_cast<uint8_t*>(
      heap_caps_malloc(kMonoBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  ESP_RETURN_ON_FALSE(mono_ != nullptr, ESP_ERR_NO_MEM, kTag,
                      "30 kB mono framebuffer allocation failed");

  ESP_RETURN_ON_ERROR(initSystemI2c(), kTag, "system I2C");
  ESP_RETURN_ON_ERROR(enablePanelPower(), kTag, "panel power");
  ESP_RETURN_ON_ERROR(initSpiBus(), kTag, "display SPI bus");
  for (auto& buffer : stream_) {
    buffer = static_cast<uint8_t*>(spi_bus_dma_memory_alloc(
        SPI2_HOST, kStreamPanelBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA));
    ESP_RETURN_ON_FALSE(buffer != nullptr, ESP_ERR_NO_MEM, kTag,
                        "4 kB panel DMA stream-buffer allocation failed");
  }
  ESP_RETURN_ON_ERROR(addSpiDevice(spi_hz), kTag, "display SPI device");

  resetController();
  ESP_RETURN_ON_ERROR(initController(), kTag, "controller init");
  clear(true);
  panel_power_on_ = false;
  return ESP_OK;
}

void Display::clear(bool paper) {
  if (mono_) memset(mono_, paper ? 0xff : 0x00, kMonoBytes);
}

void Display::loadFrame(const uint8_t* mono, size_t size) {
  if (!mono_ || mono == nullptr) return;
  memcpy(mono_, mono, std::min(size, kMonoBytes));
}

void Display::setPixel(int x, int y, bool paper) {
  if (!mono_ || x < 0 || y < 0 || x >= kWidth || y >= kHeight) return;
  const size_t offset = static_cast<size_t>(y) * (kWidth / 8) + (x >> 3);
  const uint8_t mask = static_cast<uint8_t>(0x80u >> (x & 7));
  if (paper) {
    mono_[offset] |= mask;
  } else {
    mono_[offset] &= static_cast<uint8_t>(~mask);
  }
}

void Display::fillRect(int x, int y, int width, int height, bool paper) {
  const int x0 = std::max(0, x);
  const int y0 = std::max(0, y);
  const int x1 = std::min<int>(kWidth, x + width);
  const int y1 = std::min<int>(kHeight, y + height);
  for (int py = y0; py < y1; ++py) {
    for (int px = x0; px < x1; ++px) setPixel(px, py, paper);
  }
}

void Display::drawRect(int x, int y, int width, int height, int thickness,
                       bool paper) {
  if (width <= 0 || height <= 0 || thickness <= 0) return;
  fillRect(x, y, width, thickness, paper);
  fillRect(x, y + height - thickness, width, thickness, paper);
  fillRect(x, y, thickness, height, paper);
  fillRect(x + width - thickness, y, thickness, height, paper);
}

void Display::drawText(int x, int y, const char* text, int scale, bool paper) {
  if (!text || scale <= 0) return;
  int cursor_x = x;
  int cursor_y = y;
  for (; *text; ++text) {
    if (*text == '\n') {
      cursor_x = x;
      cursor_y += 8 * scale;
      continue;
    }
    const Glyph* glyph = findGlyph(*text);
    for (int row = 0; row < 7; ++row) {
      for (int column = 0; column < 5; ++column) {
        if (glyph->rows[row] & (1u << (4 - column))) {
          fillRect(cursor_x + column * scale, cursor_y + row * scale,
                   scale, scale, paper);
        }
      }
    }
    cursor_x += 6 * scale;
  }
}

void Display::packChunk(size_t source_offset, size_t source_size,
                        uint8_t* output) {
  const uint8_t* __restrict source = mono_ + source_offset;
  auto* __restrict destination = reinterpret_cast<uint32_t*>(output);
  for (size_t index = 0; index < source_size; ++index) {
    destination[index] = kMonoExpansion[source[index]];
  }
}

esp_err_t Display::uploadOnly(RefreshTiming* timing) {
  ESP_RETURN_ON_FALSE(mono_ && stream_[0] && stream_[1] && spi_,
                      ESP_ERR_INVALID_STATE, kTag, "display not initialized");
  const int64_t transfer_start = esp_timer_get_time();
  ESP_RETURN_ON_ERROR(spi_device_acquire_bus(spi_, portMAX_DELAY), kTag,
                      "frame SPI acquire");
  gpio_set_level(kPinCs, 0);
  gpio_set_level(kPinDc, 0);
  esp_err_t error = writeByte(0x10);
  gpio_set_level(kPinDc, 1);

  spi_transaction_t transactions[2] = {};
  size_t queued = 0;
  size_t slot = 0;
  uint32_t pack_us = 0;
  for (size_t offset = 0; error == ESP_OK && offset < kMonoBytes;
       offset += kStreamMonoBytes) {
    if (queued == 2) {
      spi_transaction_t* completed = nullptr;
      error = spi_device_get_trans_result(spi_, &completed, portMAX_DELAY);
      if (error == ESP_OK) {
        --queued;
        if (completed != &transactions[slot]) {
          ESP_LOGE(kTag, "SPI DMA completion order changed");
          error = ESP_ERR_INVALID_STATE;
        }
      }
      if (error != ESP_OK) break;
    }

    const size_t source_size =
        std::min(kMonoBytes - offset, kStreamMonoBytes);
    const int64_t pack_start = esp_timer_get_time();
    packChunk(offset, source_size, stream_[slot]);
    pack_us += elapsedUs(pack_start);

    transactions[slot] = {};
    transactions[slot].length = source_size * sizeof(uint32_t) * 8;
    transactions[slot].tx_buffer = stream_[slot];
    error = spi_device_queue_trans(spi_, &transactions[slot], portMAX_DELAY);
    if (error == ESP_OK) {
      ++queued;
      slot ^= 1;
    }
  }
  while (queued != 0) {
    spi_transaction_t* completed = nullptr;
    const esp_err_t drain =
        spi_device_get_trans_result(spi_, &completed, portMAX_DELAY);
    if (drain != ESP_OK) {
      if (error == ESP_OK) error = drain;
      break;
    }
    --queued;
  }
  gpio_set_level(kPinCs, 1);
  spi_device_release_bus(spi_);
  ESP_RETURN_ON_ERROR(error, kTag, "frame upload");
  const uint32_t transfer_us = elapsedUs(transfer_start);

  if (timing) {
    timing->pack_us = pack_us;
    timing->transfer_us = transfer_us;
  }
  return ESP_OK;
}

esp_err_t Display::powerOn() {
  if (panel_power_on_) return ESP_OK;
  ESP_RETURN_ON_ERROR(command(0x04), kTag, "panel power on command");
  uint32_t assert_us = 0;
  ESP_RETURN_ON_ERROR(waitBusyAssert(2'000, &assert_us), kTag,
                      "panel power-on BUSY assertion");
  uint32_t busy_us = 0;
  ESP_RETURN_ON_ERROR(waitBusyComplete(30'000, &busy_us), kTag,
                      "panel power-on BUSY completion");
  vTaskDelay(pdMS_TO_TICKS(kBusySettleMs));
  panel_power_on_ = true;
  return ESP_OK;
}

esp_err_t Display::powerOff() {
  if (!panel_power_on_) return ESP_OK;
  const uint8_t zero = 0x00;
  ESP_RETURN_ON_ERROR(command(0x02, &zero, 1), kTag,
                      "panel power off command");
  // POWER_OFF may assert BUSY after the command. If it does not, the controller
  // is already idle; do not impose a speculative 100 ms delay.
  uint32_t ignored = 0;
  if (waitBusyAssert(20, &ignored) == ESP_OK) {
    ESP_RETURN_ON_ERROR(waitBusyComplete(30'000, &ignored), kTag,
                        "panel power-off BUSY completion");
  }
  panel_power_on_ = false;
  return ESP_OK;
}

esp_err_t Display::startRefresh(RefreshMode mode) {
  static constexpr uint8_t kBooster[] = {0x6f, 0x1f, 0x17, 0x27};
  static constexpr uint8_t kFullWindow[] = {
      0x00, 0x00, 0x01, 0x8f,  // x = 0..399
      0x00, 0x00, 0x02, 0x57,  // y = 0..599
      0x01,
  };
  const uint8_t control = mode == RefreshMode::kComplete
                              ? kCompleteDisplayControl
                              : kFastDisplayControl;
  const uint8_t zero = 0x00;
  ESP_RETURN_ON_ERROR(command(0x06, kBooster, sizeof(kBooster)), kTag,
                      "refresh booster");
  ESP_RETURN_ON_ERROR(command(0x83, kFullWindow, sizeof(kFullWindow)), kTag,
                      "refresh full window");
  ESP_RETURN_ON_ERROR(command(0x50, &control, 1), kTag,
                      "refresh display control");
  return command(0x12, &zero, 1);
}

esp_err_t Display::present(RefreshMode mode, uint16_t cutoff_ms,
                           RefreshTiming* timing) {
  ESP_RETURN_ON_FALSE(cutoff_ms >= 100 && cutoff_ms <= 2'000,
                      ESP_ERR_INVALID_ARG, kTag,
                      "fast cutoff must be 100..2000 ms");
  RefreshTiming measured = {};
  const int64_t total_start = esp_timer_get_time();
  ESP_RETURN_ON_ERROR(uploadOnly(&measured), kTag, "frame upload");

  const int64_t power_start = esp_timer_get_time();
  ESP_RETURN_ON_ERROR(powerOn(), kTag, "panel power on");
  measured.power_on_us = elapsedUs(power_start);

  const int64_t prepare_start = esp_timer_get_time();
  ESP_RETURN_ON_ERROR(startRefresh(mode), kTag, "start refresh");
  uint32_t assert_us = 0;
  ESP_RETURN_ON_ERROR(waitBusyAssert(2'000, &assert_us), kTag,
                      "refresh BUSY assertion");
  measured.prepare_us = elapsedUs(prepare_start);

  const int64_t drive_start = esp_timer_get_time();
  if (mode == RefreshMode::kComplete) {
    ESP_RETURN_ON_ERROR(waitBusyComplete(60'000, &measured.drive_us), kTag,
                        "complete waveform");
    vTaskDelay(pdMS_TO_TICKS(kBusySettleMs));
    const int64_t recovery_start = esp_timer_get_time();
    ESP_RETURN_ON_ERROR(powerOff(), kTag, "power off after full refresh");
    measured.recovery_us = elapsedUs(recovery_start);
  } else {
    // Anchor the cut to the physical BUSY falling edge, not command issue.
    esp_rom_delay_us(static_cast<uint32_t>(cutoff_ms) * 1000u);
    measured.drive_us = elapsedUs(drive_start);
    const int64_t recovery_start = esp_timer_get_time();
    resetController();
    ESP_RETURN_ON_ERROR(initController(), kTag,
                        "controller recovery after waveform cut");
    panel_power_on_ = false;
    measured.recovery_us = elapsedUs(recovery_start);
  }
  measured.total_us = elapsedUs(total_start);
  if (timing) *timing = measured;
  return ESP_OK;
}

}  // namespace papercolor
