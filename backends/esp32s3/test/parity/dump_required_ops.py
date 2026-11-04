# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump every operator the models actually need, so the kernel set cannot drift.

Reads the **emitted program's operator table**, which is the registry key by definition:
`method.cpp`'s `populate_operator_name()` builds "<name>.<overload>" from exactly this
table and looks that string up in the kernel registry. Anything else is a guess.

Walking the lowered graph's `call_function` targets and mapping them to `.out`
variants by hand would be wrong twice over:

  1. It needs a table of irregular spellings (`arange.start_step` ->
     `arange.start_out`, `where.self` -> `where.self_out`, ...) maintained by
     hand -- the same drift problem the generator removes.
  2. It misses operators inserted by passes that run AFTER the edge stage. Concretely
     `aten::copy_`, which is how a functionalised mutable buffer writes back. The decode
     step's KV cache needs it, and without it the program fails at load with
     OperatorMissing (0x14) while every op visible in the graph is registered.

So: emit the program, read the table.
"""

import argparse
import sys

import torch
from executorch.backends.esp32s3.test import firmware
from executorch.backends.esp32s3.utils.program_ops import ops_of_program

# The selective build (scripts/gen_kernels.py) reads it from there by default.
OUT = str(firmware.REQUIRED_OPS)


def _emit(lowered, reinplace=False):
    """Apply the standard passes, then emit. The passes are not optional here.

    The op list is derived from the emitted table precisely so it tracks what the
    device will look up, and both passes change that table: fusion removes
    mean.out/pow.Tensor_Scalar_out/rsqrt.out and adds the scaled or weightless
    espressif::rmsnorm custom op, and
    constant propagation removes arange.start_out. Deriving the list from unfused
    programs would register three kernels nothing calls and omit the one that the
    fused graph cannot load without -- OperatorMissing (0x14) at init, the same
    failure mode as a missing aten::copy_.
    """
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3._passes import apply_passes

    apply_passes(lowered, quiet=True)
    # reinplace mirrors exports lowered with run_reinplace_pass=True (the Glimmer
    # story and multimethod programs): their KV write is aten::index_put_, a
    # different kernel-registry key from the functional index_put.out.
    return lowered.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16,
                                       run_reinplace_pass=reinplace)
    )


class HFCausalLMHead(torch.nn.Module):
    """The tail of every Hugging Face *ForCausalLM.forward, at toy width.

    HF slices the hidden states before the head,
    ``lm_head(hidden_states[:, slice(-logits_to_keep, None), :])``, and the
    default ``logits_to_keep=0`` makes that a FULL slice. Export spells a full
    slice of a runtime tensor as ``aten.alias``; nothing folds it, so the
    emitted program looks up ``aten::alias_copy.out`` -- absent from a list
    derived only from this repo's own models, which never slice that way.
    """

    def __init__(self, d=64, vocab=128):
        super().__init__()
        self.norm = torch.nn.RMSNorm(d, eps=1e-6)
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)

    def forward(self, hidden_states, logits_to_keep: int = 0):
        h = self.norm(hidden_states)
        return self.lm_head(h[:, slice(-logits_to_keep, None), :])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, help="where to write the operator list")
    out = parser.parse_args(argv).out

    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.test.models import SPECS, ramp_input

    need = set()

    # Every model in the zoo, each both plain and delegated: the delegated form leaves a
    # different remainder on portable kernels. Iterating SPECS rather than naming three
    # models is what makes the zoo the coverage mechanism -- adding an architecture
    # there extends the required-op set, and therefore the generated kernel set, with no
    # list here to keep in sync.
    for name in SPECS:
        print(f"  deriving portable kernels: {name}", flush=True)
        factory, shape = SPECS[name]
        torch.manual_seed(0)
        m = factory().eval()
        x = ramp_input(shape)
        ep = torch.export.export(m, (x,))
        need |= ops_of_program(_emit(to_edge_transform_and_lower(ep)))
        # Edge lowering mutates the ExportedProgram.  Reusing `ep` here would
        # make the delegated pass consume a half-lowered graph: a gate/walk
        # disagreement in the backend preprocessor.
        ep = torch.export.export(m, (x,))
        need |= ops_of_program(_emit(lower_esp32(
            ep, [Esp32Partitioner()])))

    # The story's Glimmer decode is a different graph again.  In particular,
    # Glimmer's local layers use circular KV caches, whose dynamic slot needs
    # remainder.Scalar_out.  Without the real decode in this derived list, a
    # PTE serializes cleanly but fails Method::init on device -- precisely the
    # drift this file exists to prevent.
    from executorch.examples.espressif.esp32s3.story.glimmer_decode import (
        build as build_glimmer_decode,
        build_prefill as build_glimmer_prefill,
    )

    _, gm = build_glimmer_decode(
        arm="ple", vocab=256, d_model=64, layers=4, heads=8,
        kv_heads=1, ffn=192, ple_dim=24, seq_len=8, window=4,
        local_ring=True,
    )
    ep = torch.export.export(gm, (torch.tensor([3]), torch.tensor(0)))
    for reinplace in (False, True):
        need |= ops_of_program(_emit(lower_esp32(
            ep, [Esp32Partitioner(quantize_qk=True)]), reinplace=reinplace))

    # Vectorized cached prefill is not decode with a larger token tensor: it
    # adds cache/history concatenation, block masks and score-axis reshaping.
    # Derive its portable remainder too, or a PTE can serialize successfully
    # and still fail Method::init on the selective device runtime.
    _, pm = build_glimmer_prefill(
        chunk_size=4, arm="ple", vocab=256, d_model=64, layers=4,
        heads=8, kv_heads=1, ffn=192, ple_dim=24, seq_len=8, window=4,
    )
    ep = torch.export.export(
        pm, (torch.tensor([3, 40, 77, 114]), torch.tensor(0)))
    for reinplace in (False, True):
        need |= ops_of_program(_emit(lower_esp32(
            ep, [Esp32Partitioner(quantize_qk=True)]), reinplace=reinplace))

    # The persistent int8 K cache (kcache_i8.py), lowered in place as the story
    # and multimethod exports are: its programs write int8 codes and fp32
    # scales with index_put_ and concatenate int8 codes in prefill. The
    # quantiser is espressif:: and registered by the ops library.
    for build_fn, kwargs, inputs in (
            (build_glimmer_decode, dict(local_ring=True),
             (torch.tensor([3]), torch.tensor(0))),
            (build_glimmer_prefill, dict(chunk_size=4),
             (torch.tensor([3, 40, 77, 114]), torch.tensor(0)))):
        _, km = build_fn(
            arm="ple", vocab=256, d_model=64, layers=4, heads=8, kv_heads=1,
            ffn=192, ple_dim=24, seq_len=8, window=4, int8_kcache=True,
            **kwargs)
        ep = torch.export.export(km, inputs)
        need |= ops_of_program(_emit(lower_esp32(
            ep, [Esp32Partitioner(quantize_qk=True)]), reinplace=True))

    # A Hugging Face causal-LM tail (see HFCausalLMHead): its full slice before
    # the head leaves an alias_copy in the portable remainder, which none of
    # this repo's own models produce.
    torch.manual_seed(0)
    ep = torch.export.export(HFCausalLMHead().eval(), (torch.randn(1, 8, 64),))
    need |= ops_of_program(_emit(lower_esp32(
        ep, [Esp32Partitioner()])))

    # The table op's parity program (fixtures/quant_lut.py): static int8 graphs
    # whose activations became espressif::quant_lut, with the slices, adds,
    # multiplies, cat and permute that lay its inputs out.
    from executorch.backends.esp32s3.test.fixtures import quant_lut

    et, _, _ = quant_lut.build_program()
    need |= ops_of_program(et)

    # Static int8 graphs (fixtures/asym_static.py): what their rows
    # keep outside the delegates -- ExecuTorch's quantize/dequantize kernels
    # for the pairs between portable ops (the safety net), the residual add,
    # the leaky ReLU, the table op.
    from executorch.backends.esp32s3.test.fixtures import asym_static

    for et in asym_static.emitted_programs():
        need |= ops_of_program(et)

    # int16 activations (fixtures/asym_a16.py, Arm a16w8): the same,
    # with int16 pairs on the safety-net kernels around the residual add.
    from executorch.backends.esp32s3.test.fixtures import asym_a16

    for et in asym_a16.emitted_programs():
        need |= ops_of_program(et)

    # QAT round trips (fixtures/asym_qat.py): the views and residual
    # adds their rows keep outside the delegates.
    from executorch.backends.esp32s3.test.fixtures import asym_qat

    for et in asym_qat.emitted_programs():
        need |= ops_of_program(et)

    # Stage B (fixtures/asym_stage_b.py): the ops its rows run on int8
    # codes between delegates (max_pool, cat, the code table) and their Stage
    # A twins' portable ops.
    from executorch.backends.esp32s3.test.fixtures import asym_stage_b

    for et in asym_stage_b.emitted_programs():
        need |= ops_of_program(et)

    # The int16 table op's rows (fixtures/lut16.py): the input affine around
    # each lut16 table (the op itself is registered by hand).
    from executorch.backends.esp32s3.test.fixtures import lut16

    for et in lut16.emitted_programs():
        need |= ops_of_program(et)

    # The fused norms (fixtures/qnorm.py): espressif::qnorm is a
    # custom kernel; what else its rows keep outside the delegates (an
    # output's pair) comes from here.
    from executorch.backends.esp32s3.test.fixtures import qnorm

    for et in qnorm.emitted_programs():
        need |= ops_of_program(et)

    # The CNN and RNN models (fixtures/cnn.py): what Arm a8w8 leaves between
    # their delegates -- the global average pools, the residual adds, the
    # recurrent gates' products -- on the pairs' safety-net kernels.
    from executorch.backends.esp32s3.test.fixtures import cnn

    for et in cnn.emitted_programs():
        need |= ops_of_program(et)

    # ExecuTorch's own LLM recipes (fixtures/stock_llm.py): the stock quantised
    # K/V cache's programs read input_pos with .item() (aten::_local_scalar_dense,
    # which the ops library registers on the device), write with
    # llama::update_cache, and the nano llama gathers its mask with index.Tensor.
    from executorch.backends.esp32s3.test.fixtures import stock_llm

    for fixture in stock_llm.fixtures():
        if fixture.name in ("kv_gqa_seq", "nano_llama_stock"):
            et, _, _ = stock_llm.lower(fixture)
            need |= ops_of_program(et)

    # memory_plan="esp32"'s QEMU row (run_parity.export_memory_plan_variant):
    # sub, clamp and hardtanh writing over dying operands. Memory planning
    # moves no operator, so ExecuTorch's planner gives the same table.
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.test.models import MemplanKernels

    need |= ops_of_program(to_edge(torch.export.export(
        MemplanKernels().eval(), (ramp_input((4, 64)),))).to_executorch())

    # Never a portable chooser: ExecuTorch's choose_qparams agrees with none of
    # PyTorch's. The exporter absorbs every dynamic chain into the delegate or
    # refuses it by name (strip_dynamic_act_qdq), so one here is a regression.
    choosers = sorted(n for n in need if "choose_qparams" in n)
    if choosers:
        raise SystemExit(f"a program left dynamic choosers portable: {choosers}")
    with open(out, "w") as f:
        f.write("# Generated by dump_required_ops.py -- do not edit.\n")
        f.write("# Read from each emitted program's operator table, which is the\n")
        f.write("# kernel-registry key that Method::init actually looks up.\n")
        for n in sorted(need):
            f.write(n + "\n")
    print(f"wrote {out}: {len(need)} operators")
    for n in sorted(need):
        print(f"  {n}")


if __name__ == "__main__":
    main()
