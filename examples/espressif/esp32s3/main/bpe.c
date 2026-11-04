/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bpe.h"

#include <string.h>

// The tables: the header the story exporter writes from its tokenizer
// (story_bpe_tables.h), named by the story build.
#ifndef BPE_TABLES_HEADER
#error "BPE_TABLES_HEADER must name the BPE tables header (story_bpe_tables.h)"
#endif
#include BPE_TABLES_HEADER

// A pre-token is never longer than this; longer runs (a 200-letter "word")
// are split, which the reference encoder would not do. Requests are short.
#define BPE_MAX_WORD 128

static int is_letter(unsigned char b) {
  return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80;
}
static int is_digit(unsigned char b) { return b >= '0' && b <= '9'; }
static int is_space(unsigned char b) {
  return b == ' ' || b == '\t' || b == '\n' || b == '\v' || b == '\f' || b == '\r';
}

static const char* const kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

// GPT-2's pattern by hand: 's|'t|'re|'ve|'m|'ll|'d | ?L+ | ?N+ | ?[^sLN]+ |
// s+(?!S) | s+. Returns the length of the pre-token starting at text[0].
static size_t pre_token_length(const unsigned char* text, size_t n) {
  for (size_t c = 0; c < sizeof(kContractions) / sizeof(kContractions[0]); ++c) {
    const size_t len = strlen(kContractions[c]);
    if (n >= len && memcmp(text, kContractions[c], len) == 0) return len;
  }
  size_t j = 0;
  if (text[0] == ' ' && n > 1 && !is_space(text[1])) j = 1;  // a word's leading space
  if (j < n && is_letter(text[j])) {
    while (j < n && is_letter(text[j])) ++j;
  } else if (j < n && is_digit(text[j])) {
    while (j < n && is_digit(text[j])) ++j;
  } else if (j < n && !is_space(text[j])) {
    while (j < n && !is_space(text[j]) && !is_letter(text[j]) && !is_digit(text[j])) ++j;
  } else {
    // A whitespace run: all but its last character when a non-space follows.
    j = 0;
    while (j < n && is_space(text[j])) ++j;
    if (j < n && j > 1) --j;
  }
  return j == 0 ? 1 : j;
}

static const BpeMerge* find_merge(uint16_t left, uint16_t right) {
  const uint32_t key = ((uint32_t)left << 16) | right;
  size_t lo = 0, hi = BPE_MERGES;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (kBpeMerges[mid].key < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < BPE_MERGES && kBpeMerges[lo].key == key) {
    return &kBpeMerges[lo];
  }
  return NULL;
}

static size_t encode_word(const unsigned char* word, size_t n, uint16_t* ids) {
  for (size_t i = 0; i < n; ++i) ids[i] = kBpeByteToken[word[i]];
  size_t count = n;
  while (count > 1) {
    size_t best_at = 0;
    const BpeMerge* best = NULL;
    for (size_t k = 0; k + 1 < count; ++k) {
      const BpeMerge* hit = find_merge(ids[k], ids[k + 1]);
      if (hit != NULL && (best == NULL || hit->rank < best->rank)) {
        best = hit;
        best_at = k;
      }
    }
    if (best == NULL) break;
    ids[best_at] = best->out;
    memmove(&ids[best_at + 1], &ids[best_at + 2], (count - best_at - 2) * sizeof(ids[0]));
    --count;
  }
  return count;
}

size_t bpe_encode(const char* text, size_t length, uint16_t* out,
                  size_t capacity) {
  const unsigned char* bytes = (const unsigned char*)text;
  uint16_t ids[BPE_MAX_WORD];
  size_t written = 0, needed = 0;
  size_t i = 0;
  while (i < length) {
    size_t n = pre_token_length(bytes + i, length - i);
    if (n > BPE_MAX_WORD) n = BPE_MAX_WORD;
    const size_t count = encode_word(bytes + i, n, ids);
    for (size_t k = 0; k < count; ++k) {
      if (written < capacity) out[written++] = ids[k];
      ++needed;
    }
    i += n;
  }
  return needed;
}
