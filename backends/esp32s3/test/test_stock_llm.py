# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch's own LLM recipes through the backend, on the host runner
(_passes/stock_llm.py, test/fixtures/stock_llm.py).

Every row runs as main.cpp runs it -- one program, its buffers zeroed once,
`steps` executions with the step inputs -- through the host runner's script
mode, and every exact row's every output at every step must equal its mirror
bit for bit (mirrors.py: the A4 writer, the affine K-cache scores, the V
read, the embedding gathers). The full nano-llama recipe is held to eager
PyTorch: its max-abs-diff and top-1 agreement over 32 tokens are reported.

It lowers every row (about 1 GB of memory).
"""

import subprocess

import numpy as np
import pytest

from executorch.backends.esp32s3.test.fixtures import stock_llm as esl
from executorch.backends.esp32s3.test.models import output_bits
from executorch.backends.esp32s3.test.host import host_runner



@pytest.fixture(scope="module")
def runner():
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow test/host_runner/README.md")
    return path.resolve()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    work = tmp_path_factory.mktemp("stock_llm")
    table = esl.export_all(str(work), verbose=False)
    return work, {f.name: f for f in table}


def run_sequence(runner, work, fixture, extra=()):
    """Every step's outputs, [(array, ...)], from one script run."""
    lines, outs = [], []
    n_out = len(fixture.references[0])
    for step in range(fixture.steps):
        ins = []
        for i, tensor in enumerate(esl.step_inputs(fixture, step)):
            path = work / f"{fixture.name}_s{step}_in{i}.bin"
            tensor.numpy().tofile(path)
            ins.append(str(path))
        o = [work / f"{fixture.name}_s{step}_out{i}.bin" for i in range(n_out)]
        outs.append(o)
        lines.append(f"forward {' '.join(ins)} -> {' '.join(map(str, o))}")
    script = work / f"{fixture.name}_steps.txt"
    script.write_text("\n".join(lines) + "\n")
    done = subprocess.run([str(runner), "--pte", str(work / f"{fixture.name}.pte"),
                           "--script", str(script), *extra],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    return [tuple(np.fromfile(p, dtype=np.float32) for p in o) for o in outs]


@pytest.mark.parametrize("name", esl.EXACT_ROWS)
def test_every_step_equals_the_mirror(runner, exported, name):
    work, table = exported
    fixture = table[name]
    got = run_sequence(runner, work, fixture)
    for step, (outs, refs) in enumerate(zip(got, fixture.references)):
        for i, (g, r) in enumerate(zip(outs, refs)):
            r = np.ascontiguousarray(r, dtype=np.float32).reshape(-1)
            assert g.size == r.size
            bad = int((g.view(np.uint32) != r.view(np.uint32)).sum())
            assert bad == 0, (
                f"{name} step {step} output {i}: {bad} of {r.size} differ, "
                f"max abs {float(np.abs(g - r).max()):.3e}")


def test_scalar_dot_is_the_same(runner, exported):
    """PIE off (--force-scalar): the K-cache dot takes the scalar loop."""
    work, table = exported
    fixture = table["kv_gqa_seq"]
    got = run_sequence(runner, work, fixture, extra=("--force-scalar",))
    for outs, refs in zip(got, fixture.references):
        assert output_bits(outs[0]) == output_bits(np.asarray(refs[0]))


def test_stock_recipes_delegate(exported):
    """Each row's K cache and embedding went into the delegate: what
    stock_llm.prepare rewrote, the partitioner claimed (check_claimed would
    have refused anything else at export)."""
    _, table = exported
    for name in ("kv_gqa_seq", "kv_mqa_seq"):
        report = table[name].report
        assert len(report["kcache"]) == 1 and report["writers"] == 2, report
        assert "bmmqk_kca8" in table[name].log
    for name in esl.EXACT_ROWS:
        if name.startswith("emb_"):
            assert len(table[name].report["embeddings"]) == 1
            assert "embedPT_" in table[name].log
    report = table["nano_llama_stock"].report
    assert len(report["kcache"]) == 2 and report["writers"] == 4
    assert len(report["embeddings"]) == 1


def test_nano_llama_against_eager(runner, exported):
    work, table = exported
    fixture = table["nano_llama_stock"]
    got = run_sequence(runner, work, fixture)
    logits = np.stack([g[0] for g in got])
    eager = np.stack([np.asarray(r[0]).reshape(-1) for r in fixture.references])
    diff = float(np.abs(logits - eager).max())
    top1 = float((logits.argmax(-1) == eager.argmax(-1)).mean())
    print(f"nano_llama_stock vs eager over {fixture.steps} tokens: max abs "
          f"{diff:.4g}, top-1 agreement {top1:.4f}")
    assert np.isfinite(logits).all()
    assert top1 >= 0.9, (diff, top1)


def test_static_kv_cache_runs_portable(runner, tmp_path):
    """StaticQuantizedKVCache (--static_quantize_kv_cache) needs no rewrite:
    its quantise (true division, round, clamp, cast) and dequantise (cast,
    one multiply) are exact portable ops around llama::update_cache, which
    the ops library registers. Its per-head-dim key scales lie along q @ k^T's
    contraction, which no per-row K-cache opcode expresses, so attention stays
    fp32 portable; the float Linears delegate, quantised by the backend itself
    (int8 rows), so eight decode steps are held to eager as the census holds
    its float control: cosine >= 0.999 and the same greedy tokens."""
    import torch
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.source_transformation import (
        custom_kv_cache as ckv,
    )

    esl._stock_ops()
    args = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=256,
                     hidden_dim=128, max_seq_len=16, max_context_len=16,
                     use_kv_cache=True, enable_dynamic_shape=False)
    torch.manual_seed(0)
    model = construct_transformer(args).eval()
    model = ckv.replace_kv_cache_with_static_quantized_kv_cache(model)
    model = model.to(dtype=torch.float32)
    model.vocab = 256
    fixture = esl.Fixture("static_kv", model, esl.llama_inputs(0, 256), 8,
                          vocab=256, references=[(x,) for x in
                                                 esl.llama_eager(model, 8)],
                          strict=False)
    et, log, report = esl.lower(fixture, quantize_qk=False)
    assert report["writers"] == 0 and not report["kcache"]   # nothing to absorb
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    ops = ops_of_program(et)
    assert "llama::update_cache.out" in ops
    assert {"aten::div.out", "aten::round.out", "aten::clamp.out"} <= ops
    assert not any("quantized_decomposed" in op for op in ops)
    (tmp_path / "static_kv.pte").write_bytes(et.buffer)
    got = run_sequence(runner, tmp_path, fixture)
    logits = np.stack([g[0] for g in got]).astype(np.float64)
    eager = np.stack([np.asarray(r[0]).reshape(-1)
                      for r in fixture.references]).astype(np.float64)
    assert (logits.argmax(-1) == eager.argmax(-1)).all()
    cosine = (logits * eager).sum() / np.sqrt((logits ** 2).sum()
                                              * (eager ** 2).sum())
    assert cosine >= 0.999, cosine


def _plan_instructions(program):
    plan = program.executorch_program.execution_plan[0]
    return sum(len(chain.instructions) for chain in plan.chains)


def _kernel_calls(program, name):
    """How many of the plan's instructions call portable kernel `name`."""
    plan = program.executorch_program.execution_plan[0]
    names = [op.name + (f".{op.overload}" if op.overload else "")
             for op in plan.operators]
    return sum(1 for chain in plan.chains for ins in chain.instructions
               if hasattr(ins.instr_args, "op_index")
               and names[ins.instr_args.op_index] == name)


def _delegate_ops(log):
    """The ops the backend emitted, from its log (Counter of op names)."""
    import re
    from collections import Counter

    ops = Counter()
    for line in log.splitlines():
        if "[Esp32Backend]" in line and "op(s)" in line and ": " in line:
            ops.update(re.findall(r"(?:^|, )([A-Za-z][A-Za-z0-9_]*)\[",
                                  line.split(": ", 1)[1]))
    return ops


def test_kcache_query_is_opt_in(runner, exported, tmp_path):
    """Without quantize_qk (Esp32Partitioner's default) a --quantize_kv_cache
    export keeps its K cache's q @ k^T in fp32: OP_BMM_KCACHE_AFFINE_I8
    quantises its query to int8 rows, an accuracy trade every int8 q@kT
    leaves to quantize_qk. stock_llm.prepare does not rewrite the K side
    (each match goes in kcache_left), the cache is read through the exact
    dequantize_per_token_i8 like V, and the q @ k^T runs on the portable
    bmm; the writers, the V read and every Linear delegate as under
    quantize_qk. Reports what that costs against the quantize_qk program
    (instructions in the plan) and how both compare with eager."""
    from executorch.backends.esp32s3.utils.program_ops import ops_of_program

    work, table = exported
    fixture = table["nano_llama_stock"]
    et, log, report = esl.lower(fixture, quantize_qk=False)
    opted, opted_log, opted_report = esl.lower(fixture, quantize_qk=True)
    assert not report["kcache"] and len(opted_report["kcache"]) == 2
    assert report["writers"] == opted_report["writers"] == 4
    assert report["kcache_left"] and all(
        why == "the int8 query is opt-in (quantize_qk)"
        for _, why in report["kcache_left"]), report["kcache_left"]
    assert "bmmqk_kca8" not in log
    ops = ops_of_program(et)
    # Each K cache's q @ k^T is now a portable bmm (P @ V already was one).
    assert (_kernel_calls(et, "aten::bmm.out")
            == _kernel_calls(opted, "aten::bmm.out")
            + len(opted_report["kcache"])), (
        _kernel_calls(et, "aten::bmm.out"),
        _kernel_calls(opted, "aten::bmm.out"))
    # Everything else the opted-in program delegates, this one delegates too.
    want = _delegate_ops(opted_log)
    want["bmmqk_kca8"] -= len(opted_report["kcache"])
    assert not (+want - _delegate_ops(log)), (want, _delegate_ops(log))
    (tmp_path / f"{fixture.name}.pte").write_bytes(et.buffer)
    got = run_sequence(runner, tmp_path, fixture)
    want = run_sequence(runner, work, fixture)
    eager = np.stack([np.asarray(r[0]).reshape(-1) for r in fixture.references])
    rows = {}
    for name, outs in (("default", got), ("quantize_qk", want)):
        logits = np.stack([g[0] for g in outs])
        rows[name] = (float(np.abs(logits - eager).max()),
                      float((logits.argmax(-1) == eager.argmax(-1)).mean()))
        assert np.isfinite(logits).all()
    n_default, n_opted = _plan_instructions(et), _plan_instructions(opted)
    print(f"nano_llama_stock without quantize_qk: plan {n_default} "
          f"instructions (quantize_qk: {n_opted}); vs eager over "
          f"{fixture.steps} tokens: max abs {rows['default'][0]:.4g}, top-1 "
          f"{rows['default'][1]:.4f} (quantize_qk: {rows['quantize_qk'][0]:.4g}"
          f", {rows['quantize_qk'][1]:.4f}); portable ops only without it: "
          f"{sorted(ops - ops_of_program(opted))}")
    assert rows["default"][1] >= 0.9, rows
    # The default path is exact arithmetic (the K cache read through the
    # exact dequantize_per_token_i8, q @ k^T on the portable bmm): within
    # float32 rounding of eager, where the int8 query is 1e-2 off.
    assert rows["default"][0] <= 1e-5, rows


def test_fp32_policy_names_itself(exported):
    """Under format_policy='fp32' no q@kT is int8, quantize_qk or not: the
    K side is not rewritten, and kcache_left says fp32 is why (it said
    the int8 query is opt-in). (kv_gqa_seq: float Linears; the full
    recipe's 8da4w Linears are refused outright under fp32.)"""
    _, table = exported
    fixture = table["kv_gqa_seq"]
    et, log, report = esl.lower(fixture, quantize_qk=True,
                                format_policy="fp32")
    assert not report["kcache"] and report["kcache_left"], report
    assert all("format_policy='fp32'" in why
               for _, why in report["kcache_left"]), report["kcache_left"]
    assert "bmmqk" not in log


def _edge_of(model, inputs):
    from executorch.exir import EdgeCompileConfig, to_edge

    return to_edge(esl.llm_export(model, inputs),
                   compile_config=EdgeCompileConfig(_check_ir_validity=False))


def test_refused_by_name():
    """What the backend does not delegate is refused at export, by rule: the
    custom-op SDPA (ExecuTorch's fused attention kernel over A4 queries),
    float16 embedding scales (eager rounds in float16, ExecuTorch's kernel in
    float32), and torchao's embedding ops (which need the torchao kernel
    library at export; a probe op in its namespace stands in for them here,
    as the test cannot rely on loading the library)."""
    import torch
    import torch.nn as nn

    from executorch.backends.esp32s3._passes import stock_llm
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.source_transformation import (
        custom_kv_cache as ckv,
        sdpa,
    )

    esl._stock_ops()
    args = ModelArgs(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, vocab_size=64,
                     hidden_dim=128, max_seq_len=8, max_context_len=8,
                     use_kv_cache=True, enable_dynamic_shape=False)
    torch.manual_seed(0)
    model = construct_transformer(args).eval()
    for t in (sdpa.replace_sdpa_with_custom_op,
              ckv.replace_kv_cache_with_custom_kv_cache,
              ckv.replace_kv_cache_with_quantized_kv_cache,
              sdpa.replace_sdpa_with_quantized_sdpa):
        model = t(model)
    edge = _edge_of(model, esl.llama_inputs(0, 64))
    with pytest.raises(stock_llm.StockRecipeRefused) as why:
        stock_llm.prepare(edge)
    assert why.value.rule == "custom-sdpa"

    g = torch.Generator().manual_seed(1)
    half = esl.EmbeddingOp(8, torch.randint(-128, 128, (16, 32), generator=g,
                                            dtype=torch.int32).to(torch.int8),
                           (torch.rand(16, generator=g) * 0.1).half(), None,
                           -128, 127)
    edge = _edge_of(half, (torch.tensor([[1, 2]]),))
    with pytest.raises(stock_llm.StockRecipeRefused) as why:
        stock_llm.prepare(edge)
    assert why.value.rule == "embedding-scale-dtype"

    lib = torch.library.Library("torchao", "FRAGMENT")
    if not hasattr(torch.ops.torchao, "_embedding_probe"):
        lib.define("_embedding_probe(Tensor weight, Tensor indices) -> Tensor")
        lib.impl("_embedding_probe", lambda w, i: w[i], "CompositeExplicitAutograd")

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(8, 4))

        def forward(self, tokens):
            return torch.ops.torchao._embedding_probe(self.weight, tokens)

    edge = _edge_of(Probe().eval(), (torch.tensor([[1, 3]]),))
    with pytest.raises(stock_llm.StockRecipeRefused) as why:
        stock_llm.prepare(edge)
    assert why.value.rule == "torchao-embedding"
