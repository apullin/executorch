/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The weight tiers' report: which external weights the backend has copied
// from flash into PSRAM, and which it left on flash. Plain C, for firmware and
// test runners. The tiers themselves are set through the backend's options
// (Esp32Backend::set_option: weight_tier_budget, weight_tier_reserve,
// weight_tier_tables, weight_tier_rank, weight_tier_apply, weight_cache),
// which esp32_backend.cpp describes.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The format classes a weight is copied in, in the order a tie in score is
// broken: int4, int8 (an int4 head expanded at load, FLAG_STAGE_I8, is int8),
// fp32, bf16, int16 (no PSRAM rate on record for it), and fp16 last,
// whose loop is bound by its arithmetic even when it reads from flash.
enum {
  ESP32_TIER_INT4 = 0,
  ESP32_TIER_INT8 = 1,
  ESP32_TIER_FP32 = 2,
  ESP32_TIER_BF16 = 3,
  ESP32_TIER_INT16 = 4,
  ESP32_TIER_FP16 = 5,
  ESP32_TIER_CLASSES = 6
};

// Sizes are in the budget's unit: a copy's weights, its scales and its
// alignment slack. "Loaded" means held by a method that is loaded now, and
// counts each weight once however many methods resolve it.
typedef struct {
  uint64_t staged[ESP32_TIER_CLASSES];    // loaded weights with a PSRAM copy
  uint64_t on_flash[ESP32_TIER_CLASSES];  // loaded candidates left on flash
  uint64_t wanted;      // every loaded candidate: staged plus on_flash
  uint64_t resident;    // every copy, those no loaded method holds included
  uint64_t budget;      // weight_tier_budget (0 under auto)
  uint64_t reserve;     // weight_tier_reserve
  uint32_t auto_budget; // 1 when weight_tier_budget is -1 (auto)
  uint32_t entries;     // weights the cache knows: copies and candidates
  // Since boot:
  uint32_t copies;      // copies made, expansions included
  uint32_t hits;        // inits that found their weight's copy made
  uint32_t evictions;   // unused copies dropped for a newcomer scoring higher
  uint32_t skipped;     // candidates an init or a placement left on flash
  uint32_t expanded;    // int4 heads expanded to int8 (FLAG_STAGE_I8)
  // Ranked placement (weight_tier_rank):
  uint32_t rank;        // 1 when weight_tier_rank is on
  uint32_t waiting;     // loaded delegates whose ranked load awaits placement
  uint32_t placements;  // since boot: placements made
  uint32_t late_placements;  // of those, made by an execute (not applied)
} Esp32WeightTierStats;

// Fills `stats`. Takes the backend's lifecycle lock, so call it between
// loads, not from inside one.
void esp32_backend_weight_tier_stats(Esp32WeightTierStats* stats);

// The loaded delegates' ops whose weight has a copy (`copy_ops`), and of
// those the ops that read it (`copy_reads`): a delegate repoints its ops at
// the copies a placement made at its next execute, so the two agree once
// every delegate has run since. It walks every op of every loaded handle,
// under the lifecycle lock as esp32_backend_weight_tier_stats; apart from it
// so that a firmware that prints only that summary (the story) links none
// of the walk. Either pointer may be null.
void esp32_backend_weight_tier_copy_reads(uint32_t* copy_ops,
                                          uint32_t* copy_reads);

// Prints the report, then one "esp32 tier:" line per weight the cache knows
// (its class, bytes, score, users, whether it has a copy, and the digest of
// its keys); nothing when it knows none.
void esp32_backend_report_weight_tiers(void);

#ifdef __cplusplus
}
#endif
