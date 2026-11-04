#!/usr/bin/env python3

# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Byte-level BPE tables for the firmware tokenizer, and a reference encoder.

The story model reads a byte-level BPE vocabulary (HuggingFace `tokenizers`,
GPT-2 style: a regex pre-tokenizer, bytes mapped to printable characters,
ranked merges). The board has to encode a prompt the same way or the model sees
token splits it was never trained on. This script:

  1. reads the tokenizer JSON and emits the C tables main/bpe.c includes
     (story_bpe_tables.h, which export_story_flash.py writes beside
     story_vocab.h): a byte -> base-token table and the merges as
     (left, right) -> out, ranked;
  2. implements the same encoder in Python over those tables (`encode`), so
     the algorithm is checked against the `tokenizers` library (`--check`),
     and the C version is then checked against this one (test_bpe.py).

Limits, on purpose: letters and digits are ASCII plus any byte >= 0x80
(treated as a letter, which is what the GPT-2 pattern does for non-ASCII
text in practice); added tokens (opcodes, end-of-text) are not recognised in
text, the firmware never has them in a request.

    bpe_tables.py --tokenizer /path/to/tokenizer.json --header main/story_bpe_tables.h \
        [--check prompts.jsonl] [--text "..."]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable character map (what ByteLevel uses)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tables:
    def __init__(self, tokenizer_path: str):
        with open(tokenizer_path) as source:
            spec = json.load(source)
        model = spec["model"]
        if model["type"] != "BPE" or spec["pre_tokenizer"]["type"] != "ByteLevel":
            raise ValueError("expected a ByteLevel BPE tokenizer")
        if spec["pre_tokenizer"].get("add_prefix_space"):
            raise ValueError("add_prefix_space is not implemented")
        vocab: dict[str, int] = model["vocab"]
        unicode_of = bytes_to_unicode()
        self.byte_token = [vocab[unicode_of[b]] for b in range(256)]
        self.merges: list[tuple[int, int, int]] = []
        for pair in model["merges"]:
            left, right = pair if isinstance(pair, list) else pair.split(" ", 1)
            self.merges.append((vocab[left], vocab[right], vocab[left + right]))
        self.vocab_size = len(vocab)
        self.total_vocab = spec_total_vocab(spec)
        # (left << 16 | right) -> (rank, out), sorted for binary search.
        keyed = sorted((left << 16 | right, rank, out)
                       for rank, (left, right, out) in enumerate(self.merges))
        self.keys = [k for k, _, _ in keyed]
        self.rank_out = [(rank, out) for _, rank, out in keyed]
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.source = os.fspath(tokenizer_path)  # as the caller names it

    def lookup(self, left: int, right: int):
        key = left << 16 | right
        i = bisect.bisect_left(self.keys, key)
        if i < len(self.keys) and self.keys[i] == key:
            return self.rank_out[i]
        return None


def spec_total_vocab(spec) -> int:
    ids = set(spec["model"]["vocab"].values())
    ids.update(t["id"] for t in spec.get("added_tokens", []))
    return max(ids) + 1


def is_letter(b: int) -> bool:
    return (65 <= b <= 90) or (97 <= b <= 122) or b >= 0x80


def is_digit(b: int) -> bool:
    return 48 <= b <= 57


def is_space(b: int) -> bool:
    return b in (32, 9, 10, 11, 12, 13)


CONTRACTIONS = (b"'s", b"'t", b"'re", b"'ve", b"'m", b"'ll", b"'d")


def pre_tokenize(data: bytes) -> list[bytes]:
    """GPT-2's pattern, by hand: 's|'t|'re|'ve|'m|'ll|'d | ?L+ | ?N+ | ?[^ s L N]+
    | s+(?!S) | s+   (L letters, N digits, s whitespace)."""
    out = []
    i, n = 0, len(data)
    while i < n:
        for c in CONTRACTIONS:
            if data.startswith(c, i):
                out.append(c)
                i += len(c)
                break
        else:
            j = i
            if data[j] == 32 and j + 1 < n and not is_space(data[j + 1]):
                j += 1          # the optional leading space of a word
            b = data[j] if j < n else None
            if b is not None and is_letter(b):
                while j < n and is_letter(data[j]):
                    j += 1
            elif b is not None and is_digit(b):
                while j < n and is_digit(data[j]):
                    j += 1
            elif b is not None and not is_space(b):
                while j < n and not is_space(data[j]) and not is_letter(data[j]) \
                        and not is_digit(data[j]):
                    j += 1
            else:
                # A whitespace run. All but its last character form one token
                # when a non-space follows (that space prefixes the next word).
                j = i
                while j < n and is_space(data[j]):
                    j += 1
                if j < n and j - i > 1:
                    j -= 1
                elif j < n and j - i == 1:
                    # a single space before a word: the word branch above
                    # only takes a plain space, so tabs/newlines land here
                    pass
            if j == i:  # every piece takes a byte at least, so the scan ends
                j = i + 1
            out.append(data[i:j])
            i = j
    return out


def encode_word(tables: Tables, word: bytes) -> list[int]:
    ids = [tables.byte_token[b] for b in word]
    while len(ids) > 1:
        best = None
        for k in range(len(ids) - 1):
            hit = tables.lookup(ids[k], ids[k + 1])
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], k, hit[1])
        if best is None:
            break
        _, k, out = best
        ids[k:k + 2] = [out]
    return ids


def encode(tables: Tables, text: str) -> list[int]:
    ids = []
    for word in pre_tokenize(text.encode("utf-8")):
        ids.extend(encode_word(tables, word))
    return ids


def write_header(tables: Tables, path: str) -> None:
    lines = [
        "// Generated by bpe_tables.py from " + tables.source + ".",
        "// Byte-level BPE tables for bpe.c. Do not edit.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define BPE_VOCAB {tables.vocab_size}",
        f"#define BPE_TOTAL_VOCAB {tables.total_vocab}",
        f"#define BPE_MERGES {len(tables.merges)}",
        "",
        "// Base token id of each input byte.",
        "static const uint16_t kBpeByteToken[256] = {",
    ]
    for i in range(0, 256, 16):
        lines.append("  " + ", ".join(str(t) for t in tables.byte_token[i:i + 16]) + ",")
    lines += [
        "};",
        "",
        "// Merges sorted by (left << 16 | right) for binary search; rank is the",
        "// merge priority (lower first) and out the token the pair becomes.",
        "typedef struct { uint32_t key; uint16_t rank; uint16_t out; } BpeMerge;",
        f"static const BpeMerge kBpeMerges[{len(tables.merges)}] = {{",
    ]
    row = []
    for key, (rank, out) in zip(tables.keys, tables.rank_out):
        row.append(f"{{{key}u,{rank},{out}}}")
        if len(row) == 6:
            lines.append("  " + ",".join(row) + ",")
            row = []
    if row:
        lines.append("  " + ",".join(row) + ",")
    lines += ["};", ""]
    with open(path, "w") as sink:
        sink.write("\n".join(lines))


def check(tables: Tables, tokenizer_path: str, jsonl_paths: list[str],
          extra: list[str]) -> int:
    from tokenizers import Tokenizer
    reference = Tokenizer.from_file(tokenizer_path)
    texts = list(extra)
    for path in jsonl_paths:
        with open(path) as source:
            for line in source:
                row = json.loads(line)
                texts.append(row["prompt"])
                if "response" in row:
                    texts.append(row["prompt"] + row["response"])
    failures = 0
    for text in texts:
        want = reference.encode(text).ids
        if any(i >= tables.vocab_size for i in want):
            continue           # contains an added token; out of scope
        got = encode(tables, text)
        if got != want:
            failures += 1
            if failures <= 5:
                print(f"MISMATCH {text[:80]!r}\n  want {want[:24]}\n  got  {got[:24]}")
    print(f"checked {len(texts)} texts: {failures} mismatches")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True, help="the tokenizer JSON (HuggingFace tokenizers)")
    parser.add_argument("--header", required=True, help="write the C tables here")
    parser.add_argument("--check", nargs="*", metavar="JSONL",
                        help="compare the reference encoder with `tokenizers`")
    parser.add_argument("--text", action="append", default=[],
                        help="extra text for --check (repeatable)")
    args = parser.parse_args(argv)
    tables = Tables(args.tokenizer)
    write_header(tables, args.header)
    print(f"{args.header}: {len(tables.merges)} merges, vocab {tables.vocab_size}")
    if args.check is not None:
        return check(tables, args.tokenizer, args.check, args.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
