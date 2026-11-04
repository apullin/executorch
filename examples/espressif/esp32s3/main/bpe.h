/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Byte-level BPE encoder for the story's prompt text, on the device.
//
// Encodes the way the training tokenizer does (GPT-2 pattern pre-tokenizer,
// byte-to-token base map, ranked merges) so the model sees the token splits
// it was trained on. The tables come from the header the build names in
// BPE_TABLES_HEADER (story_bpe_tables.h, which the story exporter
// writes with story/bpe_tables.py), and story/test_bpe.py holds this code to
// the Python reference.
//
// Scope: plain request text. Added tokens (end-of-text) are not
// recognised in text; letters are ASCII plus any byte >= 0x80.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Encodes `text` (UTF-8, `length` bytes) into `out` (at most `capacity` ids).
// Returns the number of ids the text needs; when that exceeds `capacity`
// the first `capacity` ids are written and the rest are dropped.
size_t bpe_encode(const char* text, size_t length, uint16_t* out,
                  size_t capacity);

#ifdef __cplusplus
}
#endif
