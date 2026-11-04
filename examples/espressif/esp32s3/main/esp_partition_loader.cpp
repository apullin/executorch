/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "esp_partition_loader.h"

#include <cstring>

#include <esp_log.h>

using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

namespace esp32s3 {

namespace {
constexpr const char* kTag = "et_partition";
} // namespace

Result<EspPartitionDataLoader> EspPartitionDataLoader::from_subtype(
    uint8_t subtype) {
  const esp_partition_t* part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)subtype, nullptr);
  if (part == nullptr) {
    ESP_LOGE(kTag, "no data partition with subtype 0x%02x", subtype);
    return Error::NotFound;
  }

  const void* base = nullptr;
  esp_partition_mmap_handle_t handle = 0;
  // One mapping for the whole partition (a 14.9 MB table maps in one call).
  // MMAP_DATA maps through the flash cache, so subsequent reads are ordinary
  // loads that hit or miss in cache.
  esp_err_t err = esp_partition_mmap(
      part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &handle);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_partition_mmap failed: %d", (int)err);
    return Error::AccessFailed;
  }

  ESP_LOGI(
      kTag,
      "mapped '%s' subtype 0x%02x: %u B at %p",
      part->label,
      subtype,
      (unsigned)part->size,
      base);
  return EspPartitionDataLoader(base, (size_t)part->size, handle);
}

Result<FreeableBuffer> EspPartitionDataLoader::load(
    size_t offset,
    size_t size,
    const SegmentInfo& segment_info) const {
  (void)segment_info;
  if (base_ == nullptr) {
    return Error::InvalidState;
  }
  if (offset > size_ || size > size_ - offset) {
    ESP_LOGE(
        kTag,
        "load out of range: offset %u size %u of %u",
        (unsigned)offset,
        (unsigned)size,
        (unsigned)size_);
    return Error::InvalidArgument;
  }
  // No copy and no free: the data stays in flash and the buffer just points at
  // it. This is the whole reason for mapping rather than reading.
  return FreeableBuffer(
      static_cast<const uint8_t*>(base_) + offset,
      size,
      /*free_fn=*/nullptr);
}

Error EspPartitionDataLoader::load_into(
    size_t offset,
    size_t size,
    const SegmentInfo& segment_info,
    void* buffer) const {
  Result<FreeableBuffer> src = load(offset, size, segment_info);
  if (!src.ok()) {
    return src.error();
  }
  std::memcpy(buffer, src->data(), size);
  return Error::Ok;
}

} // namespace esp32s3
