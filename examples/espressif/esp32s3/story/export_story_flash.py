#!/usr/bin/env python3

# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a trained story Glimmer as the two-method flash artifact the board runs.

The model's shape comes from the checkpoint, its dense weights are 4-bit,
and the device samples from the full vocabulary. This script builds a
prefill/decode pair (shared mutable KV, external weights in one PTD),
lowers it with the int4 policy, and writes:

  <emit-dir>/ple.pte, ple_weights.ptd, ple_required_ops.txt
  <emit-dir>/story_inputs.h      prompt tokens, a greedy golden continuation
                                 and per-step logit checksums from the same
                                 modules that were exported (a parity oracle)
  <emit-dir>/story_vocab.h       the 8192 vocabulary strings, for the device
                                 detokenizer
  <emit-dir>/story_bpe_tables.h  the tokenizer's BPE tables, for the device's
                                 prompt encoder (main/bpe.c)
  <emit-dir>/story_artifact.json sizes, hashes, shape

    export_story_flash.py --checkpoint /path/to/glimmer-ple-story.pt \
        --tokenizer /path/to/tokenizer.json --emit-dir main

The checkpoint contract (what --checkpoint must hold) is in the example's
README.md.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import time

import torch
from tokenizers import Tokenizer

from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
    _replace_glimmer_tables_i4,
)
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3._passes import (
    apply_passes,
    fewer_gc_passes,
    lower_esp32,
)
from executorch.examples.espressif.esp32s3.scripts import flash_layout
from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
    build,
    build_prefill,
)
from executorch.backends.esp32s3.utils.multimethod import (
    combine_shared_methods,
    shared_memory_planning_pass,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from executorch.examples.espressif.esp32s3.story.bpe_tables import (  # noqa: E402
    Tables,
    bytes_to_unicode,
    write_header,
)

# The golden check's prompt: a plain story opening, the form the story model
# continues.
DEFAULT_PROMPT = "Once upon a time, there was a little girl named Lucy who had a striped scarf."

# The second golden's prompt: the same opening carried on to 75 tokens. The
# default golden (19 prompt tokens, 24 steps, to position 41) runs no prefill,
# which needs 33 tokens, and never reaches position 64, where the local
# layers' 64-slot ring first wraps. This one runs both: two prefill chunks
# (positions 0-31 from an empty cache, then 32-63 against the first chunk's
# keys), then decode from position 64, whose first step overwrites ring slot
# 0, on to position 97.
LONG_PROMPT = (
    "Once upon a time, there was a little girl named Lucy who had a striped scarf. "
    "She loved to wear it everywhere she went. One day, Lucy went to the park with her mom. "
    "The sun was warm and the birds were singing in the tall trees. Lucy saw a small dog "
    "sitting all alone under a bench. The dog looked sad and hungry, so Lucy")

# --golden: the prompt, the greedy steps and how many of the first steps'
# checksums gate values (--early-steps; 0 gates none, the STORY_GOLDEN line's
# values=OK is then vacuous and the drift is reported, not gated). The
# explicit flags override each.
#   default  the boot check every story image carries.
#   long     gated on tokens and on the value of its first prefill block
#            (early_steps 1). The reference runs prefill's softmax, P@V and
#            RoPE as the chip does (newlib expf, the fused multiply-adds of
#            shim_bmm_out and shim_rope; test/reference.py), so that block's
#            checksum is 7e-8 off on the host, and the board's block 0 matches
#            the host's to 3e-11 (a reference without those mirrors is 4e-2 off
#            on both). The second block (pos 63) is reported, not gated: it
#            holds on the host (3e-9), but the chip's sits 3e-3 off this
#            reference. The reference does not mirror
#            everything the chip rounds (RMSNorm: rsqrt0.s and four
#            accumulators on the chip, a sequential sum and 1/sqrtf on the
#            host), and 32 rows of per-row int8 codes give a one-ulp
#            difference room to flip a code. The decode steps from position 64
#            stay reported only, for the same reason.
GOLDENS = {
    "default": dict(prompt=DEFAULT_PROMPT, steps=24, early_steps=4),
    "long": dict(prompt=LONG_PROMPT, steps=24, early_steps=1),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum(logits):
    values = logits.reshape(-1).double()
    return float((values * torch.arange(1, values.numel() + 1, dtype=torch.float64)).sum())


def c_string(text):
    out = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return '"' + "".join(c if 32 <= ord(c) < 127 else "\\%03o" % ord(c) if ord(c) < 256 else c for c in out) + '"'


def vocab_strings(tokenizer, vocab):
    inverse = {c: b for b, c in bytes_to_unicode().items()}
    out = []
    for i in range(vocab):
        token = tokenizer.id_to_token(i) or ""
        try:
            raw = bytes(inverse[c] for c in token)
        except KeyError:
            raw = token.encode("utf-8")
        out.append(raw)
    return out


def c_bytes(raw):
    return '"' + "".join(
        chr(b) if 32 <= b < 127 and chr(b) not in '"\\' else "\\%03o" % b for b in raw) + '"'


@torch.no_grad()
def golden(decode, prefill, prompt_ids, chunk, steps):
    """Greedy continuation through the reference, with checksums and margins.

    margins[i] is how far the reference's chosen token i leads the runner-up,
    in logit units: where it is tiny, a rounding difference may make the
    device choose the other one, and the device check tolerates that.
    """
    n = len(prompt_ids)
    # Whole chunks prefill; the tail, the last prompt token at least, decodes.
    n_full = ((n - 1) // chunk) * chunk
    prefill_sums = []
    for start in range(0, n_full, chunk):
        tokens = torch.tensor(prompt_ids[start:start + chunk], dtype=torch.long)
        logits = prefill(tokens, torch.tensor(start, dtype=torch.long))
        prefill_sums.append(checksum(logits))
    decode_sums, generated, margins = [], [], []
    token = None
    for pos in range(n_full, n + steps):
        current = prompt_ids[pos] if pos < n else token
        logits = decode(torch.tensor([current], dtype=torch.long), torch.tensor(pos, dtype=torch.long))
        decode_sums.append(checksum(logits))
        top = torch.topk(logits.reshape(-1), 2)
        token = int(top.indices[0])
        if pos >= n - 1:  # the last prompt token's step predicts the first new token
            generated.append(token)
            margins.append(float(top.values[0] - top.values[1]))
    return prefill_sums, decode_sums, generated[:steps], margins[:steps]


@fewer_gc_passes()  # all of it: the golden and to_executorch build graphs too
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--emit-dir", required=True)
    parser.add_argument("--golden", choices=sorted(GOLDENS), default="default",
                        help="the golden check's prompt, steps and values gate (GOLDENS above): "
                        "default, or long (two prefill chunks, then decode past the 64-slot ring)")
    parser.add_argument("--prompt", help="the golden's prompt (default: --golden's)")
    parser.add_argument("--golden-steps", type=int, help="greedy steps (default: --golden's)")
    # Value gate for the device's golden run: relative drift of the first
    # --early-steps steps' logit checksums. Why only the first steps: the
    # dynamic int8 activation rounding turns any ~1e-6 difference (rsqrt,
    # libm) into an occasional one-code flip that the KV cache then carries,
    # so later steps drift by up to ~0.35 with identical arithmetic (the
    # reference against itself nudged by 1e-6); a kernel error
    # shows at step 0, which runs every layer.
    parser.add_argument("--tol-early", type=float, default=1e-5)
    parser.add_argument("--early-steps", type=int, help="checksums gated on values (default: --golden's)")
    # A device choice may differ from the reference's only where the
    # reference's top two tokens are this close (logit units). On the story
    # model int8 activation rounding can flip a 0.018 lead.
    parser.add_argument("--near-tie", type=float, default=0.1)
    # Arena planning (memory_planning.py, the default): in-place reuse,
    # a minimum-peak operator order and a hill-climb allocator, never a larger
    # arena. "none" is ExecuTorch's own planner.
    parser.add_argument("--memory-plan", choices=("esp32", "none"), default="esp32")
    args = parser.parse_args(argv)
    if args.memory_plan == "none":
        args.memory_plan = None
    preset = GOLDENS[args.golden]
    args.prompt = preset["prompt"] if args.prompt is None else args.prompt
    args.golden_steps = preset["steps"] if args.golden_steps is None else args.golden_steps
    args.early_steps = preset["early_steps"] if args.early_steps is None else args.early_steps

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["cfg"]
    quant = checkpoint.get("quantization") or {}
    if (quant.get("dense_bits"), quant.get("table_bits"), quant.get("group")) != (4, 4, 128):
        raise SystemExit(f"expected an int4/int4/group-128 checkpoint, got {quant}")
    if source["architecture"] != "glimmer" or source["arm"] != "ple":
        raise SystemExit("expected a Glimmer PLE checkpoint")
    if source.get("loops", 1) != 1:
        raise SystemExit("looped checkpoints are not exportable")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    # The device encodes prompts itself (main/bpe.c): a byte-level BPE only.
    bpe_tables = Tables(args.tokenizer)
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab != source["vocab_size"]:
        raise SystemExit(f"tokenizer vocab {vocab} != checkpoint vocab {source['vocab_size']}")
    if args.context > source["seq_len"]:
        raise SystemExit("context is longer than the checkpoint's trained context")

    # The K cache keeps each key as int8 codes plus one fp32 scale per row,
    # quantised once when written (kcache_i8.py): the same scores as an fp32
    # cache bit for bit, at a quarter of the K bytes, and q@kT does not
    # re-quantise the whole cache every token.
    common = dict(
        arm="ple", vocab=vocab, d_model=source["d_model"], layers=source["n_layers"],
        heads=source["n_heads"], kv_heads=source["n_kv_heads"], ffn=source["ffn_hidden"],
        ple_dim=source["ple_dim"], seq_len=args.context, window=source["glimmer_window"],
        softcap=source["final_logit_softcapping"], cache_context=args.context,
        selected_rows=0, head_token_ids=None, int8_kcache=True,
    )
    # One model behind both methods (and one fp32 copy behind both reference
    # methods below): a fresh TinyLM is a random initialisation that
    # load_state_dict would overwrite, four times over. The wrappers keep their
    # KV state on themselves, so sharing the model shares nothing else.
    cfg, decode = build(local_ring=True, **common)
    for key in ("qk_scale_factor", "output_multiplier", "glimmer_rope_theta", "rms_norm_eps", "post_norm_eps"):
        if abs(float(source[key]) - float(getattr(cfg, key))) > 1e-9:
            raise SystemExit(f"checkpoint {key}={source[key]} differs from the deployed default {getattr(cfg, key)}")
    decode.lm.load_state_dict(checkpoint["state"], strict=True)
    reference_lm = copy.deepcopy(decode.lm)  # before its tables become i4
    _, prefill = build_prefill(chunk_size=args.prefill_chunk, lm=decode.lm, **common)
    parameters = sum(p.numel() for p in decode.parameters())
    decode, converted = _replace_glimmer_tables_i4(decode)  # prefill's tables too
    cache_bytes = sum(v.numel() * v.element_size() for name, v in decode.named_buffers() if "cache_" in name)
    print(f"story Glimmer flash artifact: V={vocab} D={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads}/KV{cfg.n_kv_heads} "
          f"F={cfg.ffn_hidden} PLE={cfg.ple_dim} context={args.context} window={cfg.glimmer_window}; "
          f"{parameters:,} parameters; KV {cache_bytes:,} B (int8 K cache); "
          f"i4 tables={','.join(sorted(converted))}")

    # The golden runs a separate reference pair that does the chip's
    # arithmetic where it can, so its checksums can gate values and not just
    # token choices: int8 activations quantised per row against the int4
    # grouped codes the backend emits for every projection and the head, int4
    # table gathers, int8 dynamic Q@K (the backend's quantize_qk), the chip's
    # RMSNorm, decode's fused attention tail, RoPE in shim_rope's fused order
    # (prefill and decode), and prefill's softmax and P@V as the chip runs
    # them (newlib's expf, shim_bmm_out's fused multiply-adds;
    # test/reference.py). Not every operation is bit-exact (the RMSNorm seed,
    # and on the host the multiply-adds a -ffp-contract=off build leaves
    # unfused where the chip fuses them), so a checksum agrees to ~1e-7,
    # until an int8 code flips; the long golden gates its first prefill block
    # on that. Caches are aliased so decode sees what prefill wrote (two
    # eager modules would otherwise hold two caches, and the reference would
    # forget the first chunks of the prompt).
    from executorch.backends.esp32s3.test.reference import _device_quantized_reference
    _, ref_decode = build(local_ring=True, reference_dynamic_qk=True, lm=reference_lm, **common)
    _, ref_prefill = build_prefill(chunk_size=args.prefill_chunk, reference_dynamic_qk=True,
                                   lm=reference_lm, **common)
    # Patches the shared reference model's modules, so prefill's too.
    _device_quantized_reference(ref_decode, embedding_format={"lm.tok_emb", "lm.ple_table"},
                                dense_format="i4")
    ref_decode.eval()
    ref_prefill.eval()
    # Every piece of KV state: an int8 K cache adds a scale buffer per layer.
    for name, buffer in ref_decode.named_buffers():
        if "cache_" in name:
            setattr(ref_prefill, name, buffer)
    prompt_ids = tokenizer.encode(args.prompt).ids
    prefill_sums, decode_sums, generated, margins = golden(ref_decode, ref_prefill, prompt_ids, args.prefill_chunk, args.golden_steps)
    print(f"  golden: {len(prompt_ids)} prompt tokens, greedy {args.golden_steps} -> {tokenizer.decode(generated)!r}")

    method_models = {"decode": decode, "prefill": prefill}
    method_inputs = {
        "decode": (torch.tensor([3], dtype=torch.long), torch.tensor(0, dtype=torch.long)),
        "prefill": (torch.arange(3, 3 + args.prefill_chunk, dtype=torch.long), torch.tensor(0, dtype=torch.long)),
    }
    lowered, logs = {}, {}
    for name in ("decode", "prefill"):
        started = time.perf_counter()
        exported = torch.export.export(method_models[name], method_inputs[name])
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            item = lower_esp32(exported, [Esp32Partitioner(
                external_weights="glimmer_weights", format_policy="i4",
                group_weights=True, quantize_qk=True)])
            # Every decode layer's tail runs as one espressif::attn_tail, which
            # the golden's decode mirror assumes; prefill's must stay portable
            # because its mirror is the portable softmax and bmm.
            fusion = apply_passes(item, quiet=True,
                                  attn_tail_expect=cfg.n_layers if name == "decode" else 0)
        lowered[name] = item
        logs[name] = capture.getvalue()
        summary = [l.strip() for l in logs[name].splitlines() if l.startswith("esp32 partitioner:")]
        print(f"  {name}: {time.perf_counter() - started:.1f}s; fusions RMSNorm={fusion['fused']} RoPE={fusion['rope_fused']} "
              f"attention-tail={fusion['attn_tail_fused']}; {summary[-1] if summary else ''}")

    from executorch.exir import ExecutorchBackendConfig
    combined = combine_shared_methods(lowered, copy_programs=False)  # lowered is not reused
    # run_reinplace_pass: the KV-cache write is index_put, which out of place
    # copies the whole cache into a new tensor and copy_ writes it back -- two
    # cache-sized copies per layer per token to change one row (24 ms/token on
    # the chip). In place it writes the row, and the write-back collapses.
    program = combined.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16,
        external_constants=lambda node: "glimmer_weights",
        memory_planning_pass=(planner := shared_memory_planning_pass(args.memory_plan)),
        run_reinplace_pass=True))
    for name, plan in zip(lowered, getattr(planner, "plans", [])):  # to_executorch's order
        print(f"  memory plan {name}: {plan.describe()}")
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program
    required_ops = sorted(ops_of_program(program))

    checkpoint_sha256 = sha256_file(args.checkpoint)  # 116 MB: hash it once
    out = os.path.abspath(args.emit_dir)
    os.makedirs(out, exist_ok=True)
    pte_path = os.path.join(out, "ple.pte")
    with open(pte_path, "wb") as f:
        f.write(program.buffer)
    ptd_path = os.path.join(out, "ple_weights.ptd")
    # The weights are written under their external key, glimmer_weights.ptd, in
    # a private directory first: the emit directory is usually the example's
    # main/, which the parity harness shares, and its glimmer programs keep a
    # glimmer_weights.ptd of their own there.
    with tempfile.TemporaryDirectory(prefix="story_ptd_", dir=out) as staging:
        program.write_tensor_data_to_file(staging)
        os.replace(os.path.join(staging, "glimmer_weights.ptd"), ptd_path)
    with open(os.path.join(out, "ple_required_ops.txt"), "w") as f:
        f.write("# Generated by export_story_flash.py from the emitted multimethod operator tables.\n")
        f.write("".join(name + "\n" for name in required_ops))
    pte, ptd = os.path.getsize(pte_path), os.path.getsize(ptd_path)
    print(f"  PTE {pte:,} B; PTD {ptd:,} B; flash: "
          + flash_layout.describe(flash_layout.layout_for(pte, ptd), pte, ptd))
    # The weight tiers' "bytes that pay": the --stage-mb past which a bigger
    # budget copies nothing more, for decode, the method the board loads for a
    # short prompt (utils/weight_tiers.py; it prints the per-budget curve).
    from executorch.backends.esp32s3.utils import weight_tiers
    tiers = weight_tiers.report(weight_tiers.read_program(program.buffer),
                                budgets=[], sequence=["decode"])
    weight_tier_manifest = {
        "bytes_that_pay": tiers.bytes_that_pay, "weights": len(tiers.weights),
        "by_class": tiers.by_class(tiers.weights),
        "tables_on_flash": sum(w.need for w in tiers.left_out)}
    print(f"  weight tiers: {tiers.bytes_that_pay:,} B of decode's weights pay to copy "
          f"into PSRAM ({len(tiers.weights)} weights); its tables "
          f"({weight_tier_manifest['tables_on_flash']:,} B) stay on flash")

    with open(os.path.join(out, "story_inputs.h"), "w") as f:
        f.write("// Generated by export_story_flash.py. Do not edit.\n#pragma once\n#include <stdint.h>\n\n")
        eot = tokenizer.token_to_id("<|endoftext|>")
        if eot is None:
            raise SystemExit("tokenizer has no <|endoftext|> token")
        f.write(f"#define STORY_EOT {eot}\n")
        f.write(f"#define STORY_VOCAB {vocab}\n#define STORY_PREFILL_CHUNK {args.prefill_chunk}\n"
                f"#define STORY_CONTEXT {args.context}\n#define STORY_D_MODEL {cfg.d_model}\n"
                f"#define STORY_LAYERS {cfg.n_layers}\n"
                f"#define STORY_GOLDEN_TOL_EARLY {args.tol_early!r}\n"
                f"#define STORY_GOLDEN_EARLY_STEPS {args.early_steps}\n"
                f"#define STORY_GOLDEN_NEAR_TIE {args.near_tie!r}\n")
        f.write(f"static const char kStoryCheckpointSha256[] = \"{checkpoint_sha256}\";\n")
        f.write(f"static const char kStoryGoldenPrompt[] = {c_string(args.prompt)};\n")
        f.write(f"static const int64_t kStoryGoldenPromptTokens[{len(prompt_ids)}] = {{{', '.join(map(str, prompt_ids))}}};\n")
        f.write(f"static const int64_t kStoryGoldenTokens[{len(generated)}] = {{{', '.join(map(str, generated))}}};\n")
        f.write(f"static const float kStoryGoldenMargins[{len(margins)}] = {{"
                + ", ".join(f"{m:.6g}f" for m in margins) + "};\n")
        f.write(f"static const double kStoryGoldenPrefillChecksums[{max(1, len(prefill_sums))}] = {{"
                + ", ".join(f"{v:.9g}" for v in prefill_sums or [0.0]) + "};\n")
        f.write(f"static const double kStoryGoldenDecodeChecksums[{len(decode_sums)}] = {{"
                + ", ".join(f"{v:.9g}" for v in decode_sums) + "};\n")
        f.write(f"static const char kStoryGoldenText[] = {c_string(tokenizer.decode(generated))};\n")
    strings = vocab_strings(tokenizer, vocab)
    with open(os.path.join(out, "story_vocab.h"), "w") as f:
        f.write("// Generated by export_story_flash.py: the vocabulary as bytes. Do not edit.\n#pragma once\n\n")
        f.write(f"static const char* const kStoryVocabText[{vocab}] = {{\n")
        for raw in strings:
            f.write("  " + c_bytes(raw) + ",\n")
        f.write("};\n")
    write_header(bpe_tables, os.path.join(out, "story_bpe_tables.h"))
    manifest = {
        "checkpoint": os.path.abspath(args.checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "tokenizer": os.path.abspath(args.tokenizer), "tokenizer_sha256": sha256_file(args.tokenizer),
        "shape": {k: getattr(cfg, k) for k in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_hidden", "ple_dim", "glimmer_window")},
        "context": args.context, "prefill_chunk": args.prefill_chunk, "parameters": parameters,
        "kv_bytes": cache_bytes, "pte_bytes": pte, "ptd_bytes": ptd,
        "weight_tiers": weight_tier_manifest,
        "pte_sha256": sha256_file(pte_path), "ptd_sha256": sha256_file(ptd_path),
        "golden": {"prompt_tokens": len(prompt_ids), "steps": len(generated), "text": tokenizer.decode(generated)},
        "required_ops": required_ops,
    }
    with open(os.path.join(out, "story_artifact.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  wrote {out}/{{ple.pte, ple_weights.ptd, story_inputs.h, story_vocab.h, story_bpe_tables.h, "
          "story_artifact.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
