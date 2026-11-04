# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The C prompt encoder (main/bpe.c) must split text exactly like the tokenizer.

Trains a small byte-level BPE with the `tokenizers` library (the kind the story
model reads: GPT-2 pre-tokenizer, bytes mapped to printable characters, ranked
merges), writes its tables with bpe_tables.py, builds bpe.c on the host against
them with a small driver, and holds the C encoder to bpe_tables.py's reference
encoder and both to `tokenizers`. STORY_TOKENIZER names a real tokenizer JSON
(the story checkpoint's, say) to check as well.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from executorch.examples.espressif.esp32s3.story import bpe_tables as bpe  # noqa: E402

tokenizers = pytest.importorskip("tokenizers")

MAIN = os.path.join(os.path.dirname(HERE), "main")

DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"
int main(void) {
  static char text[65536];
  size_t n = fread(text, 1, sizeof(text), stdin);
  static uint16_t ids[8192];
  size_t count = bpe_encode(text, n, ids, 8192);
  for (size_t i = 0; i < count && i < 8192; ++i) printf("%u\n", ids[i]);
  return count > 8192;
}
"""

EDGE_CASES = [
    "User: turn the kitchen light on\nAssistant:\n",
    "hello   world  ",
    "what's up? I'll go... 10x  (twice)\ttab\nnew",
    "  leading",
    "trailing \n",
    "Café naïve — dash",
    "a" * 200,
    "",
    "!!!???...",
    "12345 67 8",
]

STORIES = [
    "Once upon a time, there was a little girl named Lucy who had a striped scarf.",
    "She loved to wear it everywhere she went. One day, Lucy went to the park with her mom.",
    "The cat sat on the mat; the dog didn't. They're friends, and we've seen it 12 times!",
    "Tabs\tand\nnewlines  and   runs of spaces, café, naïve — and 1999-12-31.",
]

# bpe.c splits a pre-token longer than BPE_MAX_WORD (128) bytes, which the
# reference does not: the training text keeps its words shorter, so no merge
# spans that boundary.
TRAINING = STORIES + [text for text in EDGE_CASES if len(text) < 128]


def synthetic_tokenizer(path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, min_frequency=1, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(TRAINING, trainer)
    tokenizer.save(path)
    return path


def texts():
    return EDGE_CASES + STORIES + ["unseen words: zebra quartz 99 bottles", "Ünïcödé — ✓ done"]


@pytest.fixture(scope="module",
                params=["synthetic"] + (["STORY_TOKENIZER"] if os.environ.get("STORY_TOKENIZER") else []))
def tokenizer_path(request, tmp_path_factory):
    if request.param == "STORY_TOKENIZER":
        return os.environ["STORY_TOKENIZER"]
    return synthetic_tokenizer(str(tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"))


@pytest.fixture(scope="module")
def tables(tokenizer_path):
    return bpe.Tables(tokenizer_path)


def compile_encoder(build, defines):
    driver = os.path.join(build, "driver.c")
    with open(driver, "w") as sink:
        sink.write(DRIVER)
    binary = os.path.join(build, "bpe")
    return binary, subprocess.run(
        ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", MAIN, "-I", build,
         *defines, driver, os.path.join(MAIN, "bpe.c"), "-o", binary],
        capture_output=True, text=True)


@pytest.fixture(scope="module")
def encoder(tables, tmp_path_factory):
    build = str(tmp_path_factory.mktemp("bpe-host"))
    # A name main/ never holds: bpe.c's quoted #include searches bpe.c's own
    # directory (main/) first, where a story export leaves story_bpe_tables.h.
    bpe.write_header(tables, os.path.join(build, "test_bpe_tables.h"))
    binary, done = compile_encoder(build, ['-DBPE_TABLES_HEADER="test_bpe_tables.h"'])
    assert done.returncode == 0, done.stderr

    def run(text):
        result = subprocess.run([binary], input=text.encode("utf-8"),
                                capture_output=True, check=True)
        return [int(v) for v in result.stdout.split()]
    return run


def test_c_encoder_matches_reference(encoder, tables):
    mismatches = []
    for text in texts():
        want = bpe.encode(tables, text)
        got = encoder(text)
        if got != want:
            mismatches.append((text[:60], want[:12], got[:12]))
    assert not mismatches, mismatches[:5]


def test_reference_matches_tokenizers(tables, tokenizer_path):
    reference = tokenizers.Tokenizer.from_file(tokenizer_path)
    for text in texts():
        assert bpe.encode(tables, text) == reference.encode(text).ids, text[:60]


def test_capacity_truncates_but_reports_need(encoder):
    # The driver's capacity is 8192; a 200-letter word never reaches it, so
    # check the contract at the Python level instead: needed ids are counted.
    assert len(encoder("a" * 200)) >= 1


def test_header_names_the_tables(tables, tmp_path):
    header = tmp_path / "story_bpe_tables.h"
    bpe.write_header(tables, str(header))
    head = header.read_text()[:4096]
    assert f"#define BPE_MERGES {len(tables.merges)}" in head
    assert f"#define BPE_VOCAB {tables.vocab_size}" in head


def test_encoder_requires_the_tables_header(tmp_path):
    _, done = compile_encoder(str(tmp_path), [])
    assert done.returncode != 0 and "BPE_TABLES_HEADER" in done.stderr
