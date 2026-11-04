# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The persistent int8 K cache matches the dynamic path bit for bit, at 1/4 the K bytes.

`OP_BMM_DYN_I8` re-quantises every cached key row on every step; the int8 K
cache quantises each row once, when it is written (ops/kcache_i8.py). Because a
row's scale depends only on that row, the two must agree EXACTLY -- not to a
tolerance -- and this file holds them to that:

  1. the write-side quantiser is `dynamic_i8_qk`'s key quantiser, bitwise;
  2. the cached-K mirror reproduces `dynamic_i8_qk` bitwise on random caches,
     including empty rows, exact rounding ties and extreme magnitudes, and
     both reproduce the device's arithmetic (reference.bmm_qk_i8) on a dot
     long enough that a float32 sum of the integer products would round;
  3. the Glimmer decode and prefill host references (the oracles behind the
     QEMU fixtures and the story golden) produce identical logits and caches
     with the option on and off, across a shared prefill -> decode handoff;
  4. the lowered decode claims every layer's q@kT as OP_BMM_KCACHE_I8, keeps
     the codes and scales as mutable buffers, and emits each new row through
     one quantiser and two in-place index_put_ -- no copy_ write-back.

Every check is printed and a failed one recorded, so a script run reports
them all and exits 1 if any failed; under pytest each test_* fails naming its
failed checks (@case), as in test_partition_crashes.py.

    PYTHONDONTWRITEBYTECODE=1 python test_kcache_i8.py
"""

import contextlib
import functools
import io
import struct
import sys

import torch


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def case(test):
    """One test_* case. check() records a failure and goes on, so a script
    run reports every one (main); a case that only did that would pass
    under pytest whatever its checks found. Called, it fails if any check
    it made failed, naming them. main() calls the case itself
    (__wrapped__), so the script still runs every case."""

    @functools.wraps(test)
    def run():
        start = len(FAILS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


def bitwise_equal(a, b):
    """Equal as IEEE bit patterns, so -0.0 vs 0.0 and NaN payloads count."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return torch.equal(a.contiguous().view(torch.int32),
                           b.contiguous().view(torch.int32))
    return torch.equal(a, b)


def random_keys(generator, shape):
    """Keys with the awkward cases the quantiser must agree on."""
    keys = torch.randn(shape, generator=generator)
    rows = keys.view(-1, shape[-1])
    n = rows.shape[0]
    rows[0].zero_()                                 # an unwritten cache row
    rows[1 % n] *= 1e-30                            # tiny but nonzero scale
    rows[2 % n] *= 1e30                             # huge
    # Exact ties: absmax 15.875 makes the scale exactly 1/8, so these values
    # land on k + 0.5 codes, where half-away and half-even disagree.
    if n > 3:
        ties = (torch.arange(shape[-1], dtype=torch.float32)
                - shape[-1] // 2 + 0.5) * 0.125
        ties[0] = 15.875
        rows[3] = ties
    return keys


@case
def test_write_side_quantiser():
    from executorch.backends.esp32s3.ops.kcache_i8 import (
        QUANTIZE_ROWS_I8,
        _codes_and_scale,
        quantize_rows_i8,
    )

    print("write-side quantiser == dynamic_i8_qk's key quantiser:")
    generator = torch.Generator().manual_seed(1234)
    codes_ok = True
    for shape in ((1, 64, 32), (2, 256, 32), (1, 8, 8), (3, 17, 48)):
        keys = random_keys(generator, shape)
        codes, scales = quantize_rows_i8(keys)
        ref_codes, ref_scale = _codes_and_scale(keys)
        # Codes by value: dynamic_i8_qk's float codes carry -0.0 where a small
        # negative rounds to zero (sign * floor), which int8 cannot represent
        # and the device's int8 codes never had.
        codes_ok &= (codes.dtype == torch.int8 and scales.dtype == torch.float32
                     and torch.equal(codes.to(torch.float32), ref_codes)
                     and bitwise_equal(scales, ref_scale.squeeze(-1)))
        op_codes, op_scales = QUANTIZE_ROWS_I8(keys)
        codes_ok &= bitwise_equal(op_codes, codes) and bitwise_equal(op_scales, scales)
    check("codes and scales match bitwise; the registered op is the same function",
          codes_ok)
    zero_codes, zero_scale = quantize_rows_i8(torch.zeros(2, 32))
    check("an all-zero row is scale 0, codes 0",
          not zero_codes.any() and not zero_scale.any())


@case
def test_cached_mirror_matches_dynamic():
    from executorch.backends.esp32s3.ops.kcache_i8 import (
        cached_i8_qk,
        dequantized_keys,
        quantize_rows_i8,
    )
    from executorch.examples.espressif.esp32s3.story.glimmer_model import dynamic_i8_qk

    print("cached-K mirror == dynamic_i8_qk on random caches:")
    trials, worst = 0, 0
    for seed in range(40):
        generator = torch.Generator().manual_seed(seed)
        batch = 1 + seed % 3
        queries = 1 + seed % 8
        keys_n = (8, 64, 256)[seed % 3]
        width = (8, 32, 48)[(seed // 3) % 3]
        q = torch.randn(batch, queries, width, generator=generator)
        k = random_keys(generator, (batch, keys_n, width))
        if seed % 5 == 0:
            q[0, 0].zero_()
        today = dynamic_i8_qk(q, k)
        codes, scales = quantize_rows_i8(k)
        cached = cached_i8_qk(q, codes, scales)
        trials += 1
        if not bitwise_equal(today, cached):
            worst += 1
    check("bit-identical scores in every trial", worst == 0,
          f"{trials - worst}/{trials} identical")
    k = torch.randn(1, 16, 32)
    codes, scales = quantize_rows_i8(k)
    check("dequantized_keys is codes * scale",
          bitwise_equal(dequantized_keys(codes, scales),
                        codes.to(torch.float32) * scales[..., None]))


@case
def test_long_dot_is_exact():
    """Both story mirrors sum the integer products exactly at any dot length
    the gate admits, as the device's int32 dot does. 65,536 codes near 127
    sum to ~1e9, far past 2**24, where a float32 accumulation drops low bits
    (a BLAS's blocked sums stay exact longer, so the dot is long enough to
    defeat 1,024-long blocks too)."""
    import numpy as np

    from executorch.backends.esp32s3.ops.kcache_i8 import (
        _codes_and_scale,
        cached_i8_qk,
        quantize_rows_i8,
    )
    from executorch.backends.esp32s3.serialization.espressif_blob import MAX_DOT_LEN
    from executorch.backends.esp32s3.test.reference import bmm_qk_i8
    from executorch.examples.espressif.esp32s3.story.glimmer_model import dynamic_i8_qk

    print("a 65,536-long dot, past float32's exact integers, against bmm_qk_i8:")
    width = 65536
    generator = torch.Generator().manual_seed(4096)
    rows = 1.0 + 0.02 * torch.randn(2, 6, width, generator=generator)
    q = rows[:, :3].clone()
    k = rows[:, 3:].clone()
    k[1, 0] = -k[1, 0]          # negative dots too
    k[0, 1] = -0.0              # a key row of negative zeros: exact-zero scores
    q_codes, _ = _codes_and_scale(q)
    k_codes, _ = _codes_and_scale(k)
    exact = torch.matmul(q_codes.to(torch.int64),
                         k_codes.to(torch.int64).transpose(-1, -2))
    check("the dots pass 2**24 and stay inside the gate's bound",
          width <= MAX_DOT_LEN and int(exact.abs().max()) > 2**24,
          f"max |acc| {int(exact.abs().max()):,}")
    want = torch.from_numpy(np.stack(
        [bmm_qk_i8(q[b].numpy(), k[b].numpy()) for b in range(2)]))
    dynamic = dynamic_i8_qk(q, k)
    check("dynamic_i8_qk == bmm_qk_i8 (the device's arithmetic) bit for bit",
          bitwise_equal(dynamic, want),
          f"{int((dynamic != want).sum())} of {want.numel()} scores differ")
    codes, scales = quantize_rows_i8(k)
    cached = cached_i8_qk(q, codes, scales)
    check("cached_i8_qk == bmm_qk_i8 bit for bit", bitwise_equal(cached, want),
          f"{int((cached != want).sum())} of {want.numel()} scores differ")
    zero = dynamic[0, :, 1].contiguous()
    check("an exact-zero score is +0.0, as the device's int32 zero converts",
          bool((zero.view(torch.int32) == 0).all()),
          f"bits {[hex(int(v)) for v in zero.view(torch.int32)]}")


@case
def test_glimmer_references_kcache_on_off():
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
        build,
        build_prefill,
    )
    from executorch.backends.esp32s3.ops.kcache_i8 import quantize_rows_i8

    print("Glimmer host references, int8 K cache on vs off:")
    from executorch.backends.esp32s3.test.parity.export_glimmer_decode import (
        quantize_reference_modules,
        token_at,
    )

    common = dict(arm="ple", vocab=256, d_model=64, layers=4, heads=8,
                  kv_heads=1, ffn=192, ple_dim=24, seq_len=16, window=4,
                  softcap=20.0)
    chunk = 4
    tokens = [token_at(step, common["vocab"]) for step in range(common["seq_len"])]

    def run(int8_kcache):
        _, prefill = build_prefill(chunk_size=chunk, reference_dynamic_qk=True,
                                   int8_kcache=int8_kcache, **common)
        _, decode = build(reference_dynamic_qk=True, local_ring=True,
                          int8_kcache=int8_kcache, **common)
        quantize_reference_modules(prefill, "i8")
        quantize_reference_modules(decode, "i8")
        for name, buffer in prefill.named_buffers():
            if "cache_" in name:
                setattr(decode, name, buffer)
        logits = []
        with torch.no_grad():
            for start in range(0, 2 * chunk, chunk):
                logits.append(prefill(torch.tensor(tokens[start:start + chunk]),
                                      torch.tensor(start)))
            for pos in range(2 * chunk, common["seq_len"]):
                logits.append(decode(torch.tensor([tokens[pos]]),
                                     torch.tensor(pos)))
        return logits, dict(decode.named_buffers())

    fp32_logits, fp32_state = run(False)
    int8_logits, int8_state = run(True)
    same = [bitwise_equal(a, b) for a, b in zip(fp32_logits, int8_logits)]
    check("2 prefill chunks + 8 decode steps produce bit-identical logits",
          all(same) and len(same) == 10, f"{sum(same)}/{len(same)} identical")
    state_ok = True
    for li in range(common["layers"]):
        codes, scales = quantize_rows_i8(fp32_state[f"kcache_{li}"])
        state_ok &= (bitwise_equal(int8_state[f"kcache_{li}"], codes)
                     and bitwise_equal(int8_state[f"kcache_scale_{li}"], scales)
                     and bitwise_equal(int8_state[f"vcache_{li}"],
                                       fp32_state[f"vcache_{li}"]))
    check("the int8 cache holds exactly the fp32 cache's codes and scales, "
          "and V is untouched", state_ok)

    def kv_bytes(state):
        return sum(value.numel() * value.element_size()
                   for name, value in state.items() if "cache_" in name)

    before, after = kv_bytes(fp32_state), kv_bytes(int8_state)
    k_before = sum(v.numel() * v.element_size() for n, v in fp32_state.items()
                   if n.startswith("kcache_"))
    k_after = sum(v.numel() * v.element_size() for n, v in int8_state.items()
                  if n.startswith("kcache_"))
    rows = sum(v.shape[0] * v.shape[1] for n, v in fp32_state.items()
               if n.startswith("kcache_"))
    check("K state is a quarter of fp32 plus one fp32 scale per row",
          k_after == k_before // 4 + 4 * rows,
          f"K {k_before:,} -> {k_after:,} B; KV {before:,} -> {after:,} B")


@case
def test_lowering():
    print("export and backend lowering (decode, quantize_qk=True):")
    lowering_checks()


TESTS = (test_write_side_quantiser, test_cached_mirror_matches_dynamic,
         test_long_dot_is_exact, test_glimmer_references_kcache_on_off,
         test_lowering)


def main():
    for test in TESTS:
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("int8 K cache: identical scores, a quarter of the K bytes")
    return 0


def decode_plan(program, method):
    plan = next(p for p in program.executorch_program.execution_plan
                if p.name == method)
    names = []
    for instruction in plan.chains[0].instructions:
        args = instruction.instr_args
        if hasattr(args, "op_index"):
            op = plan.operators[args.op_index]
            names.append(op.name + (f".{op.overload}" if op.overload else ""))
        elif hasattr(args, "delegate_index"):
            names.append("delegate")
        else:
            names.append(type(args).__name__)
    return names


def blob_descriptors(edge):
    """Every descriptor in every Esp32Backend blob of a lowered program."""
    from executorch.backends.esp32s3.serialization import espressif_blob as eb
    from executorch.exir.lowered_backend_module import LoweredBackendModule

    found = []
    graph_module = edge.exported_program().graph_module
    for _, module in graph_module.named_modules():
        if not (isinstance(module, LoweredBackendModule)
                and module.backend_id == eb.BACKEND_NAME):
            continue
        data = module.processed_bytes
        magic, version, n_ops, _ = struct.unpack_from(
            eb.BLOB_HEADER_FORMAT, data, 0)
        assert magic == eb.MAGIC and version == eb.VERSION
        for i in range(n_ops):
            values = struct.unpack_from(
                eb.OP_DESC_FORMAT, data,
                eb.BLOB_HEADER_BYTES + i * eb.OP_DESC_BYTES)
            found.append(dict(zip(eb.OP_DESC_FIELDS, values)))
    return found


def lowering_checks():
    from executorch.backends.esp32s3.serialization import espressif_blob as eb
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32
    from executorch.exir import ExecutorchBackendConfig
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import build

    common = dict(arm="ple", vocab=256, d_model=64, layers=4, heads=8,
                  kv_heads=1, ffn=192, ple_dim=24, seq_len=16, window=4,
                  softcap=20.0)
    inputs = (torch.tensor([3]), torch.tensor(5))
    results = {}
    for int8_kcache in (False, True):
        _, model = build(local_ring=True, int8_kcache=int8_kcache, **common)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            edge = lower_esp32(torch.export.export(model, inputs),
                               [Esp32Partitioner(format_policy="i8",
                                                 quantize_qk=True)])
            apply_passes(edge, quiet=True)
        signature = edge.exported_program().graph_signature
        mutated = set(signature.buffers_to_mutate.values())
        descs = blob_descriptors(edge)
        program = edge.to_executorch(config=ExecutorchBackendConfig(
            delegate_alignment=16, run_reinplace_pass=True))
        state = {name: value for name, value in model.named_buffers()
                 if "cache_" in name}
        results[int8_kcache] = dict(
            log=log.getvalue(), mutated=mutated, state=state,
            plan=decode_plan(program, "forward"), descs=descs)

    layers = common["layers"]
    fp32, int8 = results[False], results[True]
    opcodes = [d["opcode"] for d in int8["descs"]]
    check("every layer's q@kT is one OP_BMM_KCACHE_I8, none dynamic both sides",
          opcodes.count(eb.OP_BMM_KCACHE_I8) == layers
          and eb.OP_BMM_DYN_I8 not in opcodes,
          f"{opcodes.count(eb.OP_BMM_KCACHE_I8)} kcache, "
          f"{opcodes.count(eb.OP_BMM_DYN_I8)} dynamic")
    kc = [d for d in int8["descs"] if d["opcode"] == eb.OP_BMM_KCACHE_I8]
    check("kcache descriptors: batch = KV heads, three distinct runtime args",
          all(d["scale_group"] == common["kv_heads"] and d["flags"] == 0
              and len({d["in_arg"], d["res_arg"], d["scale_off"]}) == 3
              and d["bias_off"] == 0 and d["wt_off"] == 0
              for d in kc))
    check("fp32-cache control still lowers to OP_BMM_DYN_I8",
          [d["opcode"] for d in fp32["descs"]].count(eb.OP_BMM_DYN_I8) == layers)
    want = {name for name in int8["state"]}
    check("codes, scales and V all stay mutable buffers",
          want <= int8["mutated"] and len(want) == 3 * layers,
          f"{len(want & int8['mutated'])}/{len(want)}")
    check("codes are int8 [KH,S,Dh], scales fp32 [KH,S]",
          all(int8["state"][f"kcache_{li}"].dtype == torch.int8
              and int8["state"][f"kcache_scale_{li}"].dtype == torch.float32
              and int8["state"][f"kcache_scale_{li}"].shape
              == int8["state"][f"kcache_{li}"].shape[:2]
              for li in range(layers)))
    plan = int8["plan"]
    check("one quantiser and three in-place index_put_ per layer, no copy_",
          plan.count("espressif::quantize_rows_i8.out") == layers
          and plan.count("aten::index_put_") == 3 * layers
          and not any(name.startswith("aten::copy_") for name in plan)
          and not any("_to_dim_order_copy" in name or "_to_copy" in name
                      for name in plan),
          f"{plan.count('espressif::quantize_rows_i8.out')} quantise, "
          f"{plan.count('aten::index_put_')} index_put_")
    check("fp32 control: two in-place index_put_ per layer",
          fp32["plan"].count("aten::index_put_") == 2 * layers)
    added = len(plan) - len(fp32["plan"])
    check("the decode plan grows by exactly two instructions per layer",
          added == 2 * layers,
          f"{len(fp32['plan'])} -> {len(plan)} instructions")


if __name__ == "__main__":
    sys.exit(main())
