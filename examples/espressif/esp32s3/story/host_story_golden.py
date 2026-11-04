#!/usr/bin/env python3

# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The story firmware's golden check, run on the host delegate runner.

Reads an export_story_flash.py emit directory (ple.pte, ple_weights.ptd,
story_inputs.h, story_vocab.h) and drives the two-method program the way
story_runner.inc's golden routine does: whole prefill chunks of the prompt,
then decode over the rest of the prompt and the golden's own tokens (forced,
so one near-tie cannot change every later step), with the shared KV state
carried from method to method. Each step's logit checksum and greedy choice
are compared with the golden under the firmware's gates, and the summary
line has the firmware's STORY_GOLDEN fields.

    host_story_golden.py --emit-dir build_story_host/default \
        [--runner-flag=--min-latency] [--decode-only] [--save-logits run.npy]

--decode-only feeds the prompt through decode instead of prefill chunks, and
--save-logits keeps every step's logits, so two runs (prefill against decode,
one export against another) can be compared bit for bit. Exit status 0 only
on MATCH. Needs the host runner (backends/esp32s3/test/host_runner/README.md);
imports no torch.
"""
from __future__ import annotations

import argparse
import codecs
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
# examples/espressif/esp32s3/story -> the root of the ExecuTorch tree
TREE = HERE.parents[3]


def parse_inputs(path):
    """The golden from story_inputs.h: defines and arrays, as Python values."""
    text = path.read_text()
    defines = dict(re.findall(r"^#define (STORY_\w+) (\S+)$", text, re.M))

    def array(name):
        match = re.search(rf"\b{name}\[\d+\] = \{{([^}}]*)\}};", text)
        if match is None:
            raise SystemExit(f"{name} not found in {path}")
        return [float(item.strip().rstrip("f")) for item in match.group(1).split(",")]

    return {
        "vocab": int(defines["STORY_VOCAB"]),
        "chunk": int(defines["STORY_PREFILL_CHUNK"]),
        "context": int(defines["STORY_CONTEXT"]),
        "tol_early": float(defines.get("STORY_GOLDEN_TOL_EARLY", "1e-5")),
        "early_steps": int(defines.get("STORY_GOLDEN_EARLY_STEPS", "4")),
        "near_tie": float(defines.get("STORY_GOLDEN_NEAR_TIE", "0.1")),
        "prompt": [int(v) for v in array("kStoryGoldenPromptTokens")],
        "tokens": [int(v) for v in array("kStoryGoldenTokens")],
        "margins": array("kStoryGoldenMargins"),
        "prefill_sums": array("kStoryGoldenPrefillChecksums"),
        "decode_sums": array("kStoryGoldenDecodeChecksums"),
    }


def parse_vocab(path):
    """kStoryVocabText as bytes per token, or None without story_vocab.h."""
    if not path.is_file():
        return None
    entries = re.findall(r'^  "((?:[^"\\]|\\.)*)",$', path.read_text(), re.M)
    return [codecs.escape_decode(entry.encode("latin-1"))[0] for entry in entries]


def schedule(golden, decode_only=False):
    """(method, tokens, position) per execution, as story_generate runs them.

    decode_only feeds the whole prompt through decode instead, which must give
    the same logits from the first position prefill would have covered.
    """
    prompt, forced, chunk = golden["prompt"], golden["tokens"], golden["chunk"]
    n = len(prompt)
    n_full = 0 if decode_only else ((n - 1) // chunk) * chunk
    steps = [("prefill", prompt[start:start + chunk], start)
             for start in range(0, n_full, chunk)]
    generated = 0
    for pos in range(n_full, golden["context"] - 1):
        steps.append(("decode", [prompt[pos] if pos < n else forced[pos - n]], pos))
        if pos + 1 < n:
            continue
        generated += 1
        if generated >= len(forced):
            break
    return steps


def checksum(logits):
    # story_checksum: sum of logit * (lane + 1), in double.
    return float(np.dot(logits.astype(np.float64),
                        np.arange(1, logits.size + 1, dtype=np.float64)))


def run_runner(runner, emit_dir, steps, vocab, work, flags):
    """Write the step inputs and script, run them, return logits per step."""
    lines, outputs = [], []
    for index, (method, tokens, position) in enumerate(steps):
        token_path = work / f"tokens{index}.bin"
        position_path = work / f"position{index}.bin"
        output_path = work / f"logits{index}.bin"
        np.asarray(tokens, dtype="<i8").tofile(token_path)
        np.asarray(position, dtype="<i8").tofile(position_path)
        output_path.unlink(missing_ok=True)
        lines.append(f"{method} {token_path} {position_path} -> {output_path}")
        outputs.append(output_path)
    script = work / "steps.txt"
    script.write_text("\n".join(lines) + "\n")
    command = [str(runner), "--pte", str(emit_dir / "ple.pte"),
               "--ptd", str(emit_dir / "ple_weights.ptd"), "--script", str(script), *flags]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        sys.stdout.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"host runner failed ({result.returncode})")
    logits = np.stack([np.fromfile(path, dtype="<f4") for path in outputs])
    if logits.shape != (len(steps), vocab):
        raise SystemExit(f"logits shape {logits.shape}, expected {(len(steps), vocab)}")
    return logits, result.stdout


def last_position(step):
    _, tokens, position = step
    return position + len(tokens) - 1  # Both methods return the last row's logits.


def compare(golden, steps, logits):
    """The firmware's golden comparison: (report lines, summary, ok, choices)."""
    lines, rels = [], []
    # The golden's checksums by the position whose logits they are: the
    # exporter's prefill blocks, then one decode step per position after them.
    chunk, n = golden["chunk"], len(golden["prompt"])
    n_full = ((n - 1) // chunk) * chunk
    by_position = {start + chunk - 1: value for start, value in
                   zip(range(0, n_full, chunk), golden["prefill_sums"])}
    by_position.update({n_full + k: value for k, value in enumerate(golden["decode_sums"])})
    sums = [checksum(row) for row in logits]
    for i, step in enumerate(steps):
        reference = by_position.get(last_position(step))
        if reference is None:
            continue
        rel = abs(sums[i] - reference) / max(abs(reference), 1e-9)
        rels.append(rel)
        lines.append(f"  {step[0]} pos {last_position(step):3d}: checksum {sums[i]:.6f} "
                     f"golden {reference:.6f} rel {rel:.3e}")
    prefill = [i for i, step in enumerate(steps) if step[0] == "prefill"]
    decode = [i for i, step in enumerate(steps) if step[0] == "decode"]
    # Choices start at the last prompt token's step, whose logits predict the first new token.
    picks = [i for i in decode if steps[i][2] >= n - 1]
    choices = [int(np.argmax(logits[i])) for i in picks]
    matched = near_ties = hard = 0
    first_diff = -1
    for k, (choice, reference) in enumerate(zip(choices, golden["tokens"])):
        if choice == reference:
            matched += 1
            continue
        first_diff = k if first_diff < 0 else first_diff
        tie = golden["margins"][k] < golden["near_tie"]
        near_ties += tie
        hard += not tie
        row = logits[picks[k]]
        lines.append(f"  choice {k}: host {choice}, reference {reference}, reference lead "
                     f"{golden['margins'][k]:.4f} logit, host lead "
                     f"{row[choice] - row[reference]:.4f} ({'near-tie' if tie else 'FAIL'})")
    early = rels[:golden["early_steps"]]
    early_max = max(early) if early else 0.0
    onset = next((i for i, rel in enumerate(rels) if rel > 1e-5), -1)
    ordered = sorted(rels)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    tokens_ok = hard == 0 and matched + near_ties == len(golden["tokens"])
    values_ok = len(rels) >= golden["early_steps"] and early_max <= golden["tol_early"]
    summary = (f"HOST_STORY_GOLDEN result={'MATCH' if tokens_ok and values_ok else 'MISMATCH'} "
               f"tokens_match={matched}/{len(golden['tokens'])} near_ties={near_ties} "
               f"hard={hard} first_diff={first_diff} early_max={early_max:.3e} "
               f"tol_early={golden['tol_early']:.1e} early_steps={golden['early_steps']} "
               f"onset={onset} median_rel={median:.3e} max_rel={max(rels, default=0.0):.3e} "
               f"values={'OK' if values_ok else 'DRIFT'} prefill_blocks={len(prefill)} "
               f"decode_steps={len(decode)}")
    lines.append("  step drift: " + " ".join(f"{rel:.0e}" for rel in rels))
    return lines, summary, tokens_ok and values_ok, choices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emit-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path(
        os.environ.get("ESP32_HOST_RUNNER",
                       TREE / "cmake-out-espressif" / "esp32_host_runner")))
    parser.add_argument("--runner-flag", action="append", default=[],
                        help="passed to the runner, e.g. --runner-flag=--min-latency")
    parser.add_argument("--decode-only", action="store_true",
                        help="feed the prompt through decode, not prefill chunks")
    parser.add_argument("--save-logits", type=Path,
                        help="write every step's logits here (.npy) for bitwise comparison")
    parser.add_argument("--work-dir", type=Path, help="keep step files here (default: temporary)")
    args = parser.parse_args(argv)

    emit = args.emit_dir.resolve()
    golden = parse_inputs(emit / "story_inputs.h")
    vocab = parse_vocab(emit / "story_vocab.h")
    steps = schedule(golden, args.decode_only)
    with tempfile.TemporaryDirectory(prefix="host_story_") as temporary:
        work = args.work_dir.resolve() if args.work_dir else Path(temporary)
        work.mkdir(parents=True, exist_ok=True)
        logits, log = run_runner(args.runner.resolve(), emit, steps, golden["vocab"],
                                 work, args.runner_flag)
    if args.save_logits:
        np.save(args.save_logits, logits)
    lines, summary, ok, choices = compare(golden, steps, logits)
    print(f"--- {emit}: prompt {len(golden['prompt'])} tokens, golden {len(golden['tokens'])}, "
          f"prefill chunk {golden['chunk']}, last position {steps[-1][2]}"
          f"{' decode-only' if args.decode_only else ''} {' '.join(args.runner_flag)}".rstrip())
    for line in log.splitlines():
        if line.startswith(("esp32 backend:", "esp32 split:", "esp32 bmmqk:", "host kernels:",
                            "esp32: PIE")) or " dispatch: " in line:
            print("  " + line)
    print("\n".join(lines))
    if vocab is not None:
        def text(ids):
            return b"".join(vocab[i] for i in ids).decode("utf-8", "replace")
        print(f"  host greedy:   {text(choices)!r}")
        print(f"  golden greedy: {text(golden['tokens'])!r}")
    print(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
