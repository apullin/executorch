# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decode steps and prefill chunks reproduce the sequence model.

The story's references (the golden's, the parity rows') all run the program's
own wrappers (glimmer_decode.py): the same ring slots, validity masks and
commit order as the exported methods. So an off-by-one in a ring slot or a
window bound would pass the golden and the parity rows alike. The oracle
here is independent of both: glimmer_model.TinyLM's sequence forward,
which attends over the whole sequence at once through one causal (and, for
local layers, windowed) mask and keeps no cache.

The configs are small so the windows wrap many
times: windows of 4 and 8 over 16 positions, prefill chunks of 4 that start
past the window (each chunk after the first reads ring slots an earlier chunk
wrote, and from start 4 or 8 on overwrites them), and a prefill-to-decode
handoff whose decode steps keep wrapping the ring prefill filled. fp32
throughout, held to 2e-5 on the logits (a wrong slot or bound misses that by
orders of magnitude).

    python test_glimmer_step_sequence.py
"""

import sys

import pytest
import torch

from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
    build,
    build_prefill,
)

BAR = 2e-5
COMMON = dict(vocab=256, d_model=64, layers=4, heads=8, kv_heads=1, ffn=192,
              ple_dim=24, seq_len=16, softcap=20.0)
CHUNK = 4


def tokens(cfg, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (1, cfg.seq_len),
                         generator=generator)


def gap(got, want):
    return (got - want).abs().max().item()


def share_cache(source, target):
    """Point every piece of target's KV state at source's: one cache, as the
    two methods share it on the device (and the story's reference does)."""
    for name, buffer in source.named_buffers():
        if "cache_" in name:
            setattr(target, name, buffer)


@pytest.mark.parametrize("arm", ["baseline", "ple"])
@pytest.mark.parametrize("window", [4, 8])
@pytest.mark.parametrize("local_ring", [False, True])
def test_decode_steps_reproduce_sequence(arm, window, local_ring):
    cfg, decode = build(arm=arm, window=window, local_ring=local_ring,
                        **COMMON)
    ids = tokens(cfg)
    with torch.no_grad():
        full, _ = decode.lm(ids)
        steps = torch.cat([decode(ids[0, t:t + 1], torch.tensor(t))
                           for t in range(cfg.seq_len)])
    error = gap(steps, full[0])
    assert error < BAR, (
        f"{arm} decode (window {window}, ring {local_ring}) is {error:.2e} "
        f"off the sequence model")


def test_ring_caches_are_window_sized():
    cfg, decode = build(arm="ple", window=4, local_ring=True, **COMMON)
    shapes = {name: tuple(buffer.shape)
              for name, buffer in decode.named_buffers()
              if name.startswith(("kcache_", "vcache_"))}
    local = {shape for name, shape in shapes.items()
             if decode.lm.blocks[int(name.rsplit("_", 1)[1])].attn.local}
    other = {shape for name, shape in shapes.items()
             if not decode.lm.blocks[int(name.rsplit("_", 1)[1])].attn.local}
    assert local == {(cfg.n_kv_heads, cfg.glimmer_window, cfg.head_dim)}, shapes
    assert other == {(cfg.n_kv_heads, cfg.seq_len, cfg.head_dim)}, shapes


def test_logical_cache_context():
    # A global cache shorter than the trained context: the first 12 positions
    # still match, and the global layer's cache has 12 rows.
    cfg, decode = build(arm="ple", window=4, local_ring=True,
                        cache_context=12, **COMMON)
    ids = tokens(cfg)
    with torch.no_grad():
        full, _ = decode.lm(ids)
        steps = torch.cat([decode(ids[0, t:t + 1], torch.tensor(t))
                           for t in range(12)])
    error = gap(steps, full[0, :12])
    assert error < BAR, f"cache context 12: {error:.2e} off the sequence model"
    rows = {buffer.shape[1] for name, buffer in decode.named_buffers()
            if name.startswith("kcache_")
            and not decode.lm.blocks[int(name.rsplit("_", 1)[1])].attn.local}
    assert rows == {12}, rows


@pytest.mark.parametrize("arm", ["baseline", "ple"])
@pytest.mark.parametrize("window", [4, 8])
def test_prefill_chunks_reproduce_sequence(arm, window):
    cfg, prefill = build_prefill(arm=arm, window=window, chunk_size=CHUNK,
                                 **COMMON)
    ids = tokens(cfg)
    with torch.no_grad():
        full, _ = prefill.lm(ids)
        chunks = torch.cat([
            prefill(ids[0, start:start + CHUNK], torch.tensor(start))
            for start in range(0, cfg.seq_len, CHUNK)])
    # Each chunk returns its last row's logits.
    error = gap(chunks, full[0, CHUNK - 1::CHUNK])
    assert error < BAR, (
        f"{arm} prefill (window {window}, chunks of {CHUNK}) is {error:.2e} "
        f"off the sequence model")


@pytest.mark.parametrize("window", [4, 8])
@pytest.mark.parametrize("prefill_end", [4, 8, 12])
def test_prefill_then_decode_reproduce_sequence(window, prefill_end):
    # The story's schedule: whole prefill chunks, then decode from the next
    # position on, both methods on one cache. Decode starts on a ring prefill
    # wrote, and from prefill_end = window on, a ring prefill already wrapped.
    cfg, prefill = build_prefill(arm="ple", window=window, chunk_size=CHUNK,
                                 **COMMON)
    _, decode = build(arm="ple", window=window, local_ring=True,
                      lm=prefill.lm, **COMMON)
    share_cache(prefill, decode)
    ids = tokens(cfg, seed=11)
    with torch.no_grad():
        full, _ = prefill.lm(ids)
        got = [prefill(ids[0, start:start + CHUNK], torch.tensor(start))
               for start in range(0, prefill_end, CHUNK)]
        got += [decode(ids[0, t:t + 1], torch.tensor(t))
                for t in range(prefill_end, cfg.seq_len)]
    want = torch.cat([full[0, CHUNK - 1:prefill_end:CHUNK],
                      full[0, prefill_end:]])
    error = gap(torch.cat(got), want)
    assert error < BAR, (
        f"prefill to {prefill_end - 1}, then decode (window {window}): "
        f"{error:.2e} off the sequence model")


def main():
    failed = 0
    cases = (
        [(test_decode_steps_reproduce_sequence, (arm, window, ring))
         for arm in ("baseline", "ple") for window in (4, 8)
         for ring in (False, True)]
        + [(test_ring_caches_are_window_sized, ()),
           (test_logical_cache_context, ())]
        + [(test_prefill_chunks_reproduce_sequence, (arm, window))
           for arm in ("baseline", "ple") for window in (4, 8)]
        + [(test_prefill_then_decode_reproduce_sequence, (window, end))
           for window in (4, 8) for end in (4, 8, 12)])
    for test, args in cases:
        label = f"{test.__name__}{args if args else ''}"
        try:
            test(*args)
            print(f"  ok   {label}")
        except AssertionError as error:
            failed += 1
            print(f"  FAIL {label}: {error}")
    print()
    if failed:
        print(f"FAILED: {failed} case(s)")
        return 1
    print("decode steps and prefill chunks reproduce the sequence model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
