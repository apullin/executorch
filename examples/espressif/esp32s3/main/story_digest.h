/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The story golden's bit identity: a 64-bit FNV-1a over raw bytes, chained
// from one buffer to the next. story_runner.inc runs it over every prefill
// block's and decode step's logits, in the order they ran, and prints the
// result as STORY_GOLDEN digest=0x...: two runs, or two builds, that print
// the same digest computed the same bits. The step checksums print 6 decimals
// of a weighted sum, which cannot see one ulp in one logit, and only five
// decode steps print one. story/test_story_digest.py holds this code to
// FNV-1a's published values on the host.
#pragma once

#include <stddef.h>
#include <stdint.h>

#define STORY_DIGEST_BASIS 0xcbf29ce484222325ull  // FNV-1a's 64-bit offset basis

static inline uint64_t story_digest_bytes(uint64_t hash, const void* data, size_t size) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ull;  // the 64-bit FNV prime
  }
  return hash;
}
