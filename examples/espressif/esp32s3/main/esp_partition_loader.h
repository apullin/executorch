/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// An ExecuTorch DataLoader backed by a memory-mapped ESP-IDF flash partition.
//
// Why this exists: BufferDataLoader requires the data to already be addressable,
// which for EMBED_FILES means it is linked into the app image and therefore capped
// by the app partition (1.5 MB here). The PLE table is 12.58 MB. Instead, the
// whole partition is mapped in a single esp_partition_mmap call (a 14.9 MB
// mapping works on this silicon) and rows are read straight out of the mapped
// window, behind ET's loader interface.
//
// The mapping is created once and never copied: load() hands back a FreeableBuffer
// that points into flash with a nullptr free_fn, so weights cost zero RAM and the
// cost of a "read" is a cache miss rather than a memcpy.

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <esp_partition.h>

namespace esp32s3 {

class EspPartitionDataLoader final : public ::executorch::runtime::DataLoader {
 public:
  /// Finds a data partition by subtype and maps it in full.
  static ::executorch::runtime::Result<EspPartitionDataLoader> from_subtype(
      uint8_t subtype);

  EspPartitionDataLoader(EspPartitionDataLoader&& rhs) noexcept
      : base_(rhs.base_), size_(rhs.size_), handle_(rhs.handle_) {
    rhs.base_ = nullptr;
    rhs.handle_ = 0;
  }

  ~EspPartitionDataLoader() override {
    if (handle_ != 0) {
      esp_partition_munmap(handle_);
    }
  }

  ::executorch::runtime::Result<::executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const SegmentInfo& segment_info) const override;

  ::executorch::runtime::Error load_into(
      size_t offset,
      size_t size,
      const SegmentInfo& segment_info,
      void* buffer) const override;

  ::executorch::runtime::Result<size_t> size() const override {
    return size_;
  }

  /// Base of the mapped window, for callers that want to read it directly.
  const void* mapped_base() const {
    return base_;
  }

 private:
  EspPartitionDataLoader(
      const void* base,
      size_t size,
      esp_partition_mmap_handle_t handle)
      : base_(base), size_(size), handle_(handle) {}

  const void* base_;
  size_t size_;
  esp_partition_mmap_handle_t handle_;
};

} // namespace esp32s3
