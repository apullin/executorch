# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-op weight-format overrides: one sensitive projection at i16, the rest as policy.

The format policy is global -- "i8", "i4", "auto" or "i16" for every op in
the model -- but the i16 accuracy rung exists for the one op where int8's error
is the binding constraint, and reaching it should not force the WHOLE model
onto the slowest format. `Esp32Partitioner(format_overrides={...})` keys a
format to a weight's parameter name (the FQN torch.export records), and
preprocess consults it before the policy chooser. Nothing about the runtime
changes: an overridden op is an ordinary op at the overridden rung.

The claims:

  1. an override lands on exactly the named op, at the named opcode, and every
     other op keeps the policy's rung;
  2. an override on an embedding selects that rung's gather opcode;
  3. a name that matches no weight is refused at partition time, not silently
     ignored (the same reasoning as unknown compile-spec keys);
  4. a pass-through (already quantised) weight cannot be overridden -- its
     format is the codes' format -- and says so;
  5. the compile spec round-trips through its wire form unchanged.

    python test_format_override.py
"""

import contextlib
import io
import struct
import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class EmbedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(40, 24)
        self.fc = nn.Linear(24, 8)

    def forward(self, idx):
        return self.fc(self.emb(idx))


def _descriptors(model, example, **partitioner_kwargs):
    """Lower and return the descriptor opcodes in graph order plus the log."""
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.serialization.espressif_blob import (
        BLOB_HEADER_BYTES,
        BLOB_HEADER_FORMAT,
        OP_DESC_BYTES,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ep = torch.export.export(model, example)
        low = to_edge_transform_and_lower(
            ep, partitioner=[Esp32Partitioner(**partitioner_kwargs)])
        apply_passes(low, quiet=True)
    opcodes = []
    gm = low.exported_program().graph_module
    for node in gm.graph.nodes:
        if node.op == "get_attr" and "lowered_module" in node.target:
            blob = getattr(gm, node.target).processed_bytes
            _magic, _ver, n_ops, _ext = struct.unpack_from(
                BLOB_HEADER_FORMAT, blob)
            for i in range(n_ops):
                desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
                    OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES + i * OP_DESC_BYTES)))
                opcodes.append(desc["opcode"])
    return opcodes, buf.getvalue()


def main():
    from executorch.backends.esp32s3.serialization.espressif_blob import (
        OP_ADDMM_I8,
        OP_ADDMM_I16,
        OP_EMBED_I4,
        OP_EMBED_I8,
    )
    from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec

    torch.manual_seed(0)
    x = torch.randn(4, 32)

    print("one op overridden to i16, the rest at the i8 policy:")
    ops, log = _descriptors(MLP().eval(), (x,), format_policy="i8",
                            format_overrides={"fc2.weight": "i16"})
    check("two matmul descriptors emitted", len(ops) == 2, str(ops))
    check("fc1 stays at the policy rung (i8)",
          ops[:1] == [OP_ADDMM_I8], str(ops))
    check("fc2 takes the override (i16)",
          ops[1:] == [OP_ADDMM_I16], str(ops))
    check("the emit log names the override", "(override" in log)

    print("no override: both at i8 (control):")
    ops, _ = _descriptors(MLP().eval(), (x,), format_policy="i8")
    check("control has no i16 descriptor", ops == [OP_ADDMM_I8] * 2, str(ops))

    print("embedding override:")
    idx = torch.tensor([3, 7, 11, 39])
    ops, _ = _descriptors(EmbedNet().eval(), (idx,), format_policy="i8",
                          format_overrides={"emb.weight": "i4"})
    check("the gather takes the i4 embed opcode, the Linear stays i8",
          OP_EMBED_I4 in ops and OP_ADDMM_I8 in ops and OP_EMBED_I8 not in ops,
          str(ops))

    print("refusals:")
    try:
        _descriptors(MLP().eval(), (x,), format_overrides={"fc9.weight": "i16"})
        check("an override naming no weight is refused", False, "no error")
    except RuntimeError as e:
        check("an override naming no weight is refused", "fc9.weight" in str(e),
              str(e)[:80])
    try:
        Esp32CompileSpec(format_overrides={"fc2.weight": "fp8"})
        check("an unknown format is refused at the spec", False, "no error")
    except RuntimeError as e:
        check("an unknown format is refused at the spec", "fp8" in str(e),
              str(e)[:80])
    try:
        Esp32CompileSpec(format_policy="fp32",
                         format_overrides={"fc2.weight": "i16"})
        check("fp32 policy cannot carry overrides", False, "no error")
    except RuntimeError as e:
        check("fp32 policy cannot carry overrides", "override" in str(e),
              str(e)[:80])
    try:
        # Static scales are int8-range; under an i16 policy every quantised
        # matmul and conv would fail in preprocess, so the spec refuses it.
        Esp32CompileSpec(format_policy="i16", static_act=True)
        check("an i16 policy refuses static_act at the spec", False, "no error")
    except RuntimeError as e:
        check("an i16 policy refuses static_act at the spec",
              "static_act" in str(e), str(e)[:80])

    print("pass-through weights keep the codes' format:")
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer

    gm = torch.export.export(MLP().eval(), (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=4))
    prepared(x)
    converted = convert_pt2e(prepared)
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )

    edge = to_edge(torch.export.export(converted, (x,)))
    strip_dynamic_act_qdq(edge)
    # convert_pt2e replaces the parameter with a constant holding the codes,
    # under a generated name (`_frozen_param1`, nothing like "fc2.weight"),
    # so the override has to name THAT constant. Resolve it the way a user
    # would have to: from the dequant feeding fc2, through the signature.
    ep_q = edge.exported_program()
    sig = ep_q.graph_signature
    dequants = [n for n in ep_q.graph.nodes if n.op == "call_function"
                and "dequantize_per_channel" in str(n.target)]
    codes_ph = dequants[-1].args[0]          # graph order: fc1 then fc2
    codes_fqn = (sig.inputs_to_lifted_tensor_constants.get(codes_ph.name)
                 or sig.inputs_to_buffers.get(codes_ph.name)
                 or sig.inputs_to_parameters.get(codes_ph.name))
    check("the PT2E codes constant has a signature name",
          codes_fqn is not None, codes_ph.name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            edge.to_backend(Esp32Partitioner(
                format_overrides={codes_fqn or "fc2.weight": "i16"}))
        check("overriding a pass-through weight is refused", False, "no error")
    except RuntimeError as e:
        check("overriding a pass-through weight is refused",
              "pass-through" in str(e), str(e)[:90])

    # The same contract for a pass-through TABLE: its codes fix its format.
    import numpy as np

    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.esp32s3.utils.pass_through import PTEmbedding, QTensor

    codes = np.random.default_rng(0).integers(-7, 8, (40, 24), dtype=np.int8)
    table = PTEmbedding(QTensor(codes, np.full((40, 1), 0.02, np.float16), 24))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            to_edge_transform_and_lower(
                torch.export.export(table.eval(), (idx,)),
                partitioner=[Esp32Partitioner(format_overrides={"codes": "i16"})])
        check("overriding a pass-through table is refused", False, "no error")
    except RuntimeError as e:
        check("overriding a pass-through table is refused",
              "pass-through" in str(e), str(e)[:90])

    print("compile spec wire form:")
    spec = Esp32CompileSpec(format_policy="auto",
                            format_overrides={"a.weight": "i16", "b.weight": "i4"})
    back = Esp32CompileSpec.from_list(spec.to_list())
    check("format_overrides round-trip through to_list/from_list", back == spec,
          repr(back))
    plain = Esp32CompileSpec()
    check("no overrides emits no extra spec",
          all(s.key != "format_overrides" for s in plain.to_list()))

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("per-op format overrides land where named and nowhere else")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() records every failed
    check and goes on; this fails if any did, naming them."""
    start = len(FAILS)
    rc = main()
    failed = FAILS[start:]
    assert not failed and rc in (None, 0), f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(main())
