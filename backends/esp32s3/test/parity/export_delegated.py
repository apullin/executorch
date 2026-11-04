# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the MLP through the ESP32 backend, so its Linear layers become delegate calls.

Also prints what the lowered graph looks like: exactly which tensors arrive as delegate
arguments and in what order -- that is the AOT/runtime contract, and guessing it is how
backends break.
"""

import os
import struct
import torch

from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3._passes import (
    apply_passes,
    canonical_passes,
    lower_esp32,
)
from executorch.backends.esp32s3.test.models import SPECS, checksum, ramp_input
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)


def export_one(name, out_name, external_weights=False, format_policy="i8",
               min_latency=False, shape=None, conv=False):
    """Export one model through the ESP32 backend.

    `shape` overrides the spec's input shape. It exists for fixtures that differ
    from an existing model only in extent -- a wider T to reach a code path that
    the default shape cannot -- where adding a SPECS entry would also mint a
    redundant portable row and its .pte.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    factory, default_shape = SPECS[name]
    shape = default_shape if shape is None else shape
    torch.manual_seed(0)
    model = factory().eval()
    x = ramp_input(shape)
    ep = torch.export.export(model, (x,))
    lowered = lower_esp32(
        ep, [Esp32Partitioner(external_weights=external_weights,
                              format_policy=format_policy,
                              min_latency=min_latency, conv=conv)])
    apply_passes(lowered)
    n_del = sum(
        1
        for n in lowered.exported_program().graph_module.graph.nodes
        if n.op == "call_function" and "call_delegate" in str(n.target)
    )
    # PIE's 128-bit loads truncate silently on misalignment, so demand it of the
    # container rather than hoping the default suits.
    et = lowered.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, out_name)
    with open(out, "wb") as f:
        f.write(et.buffer)
    # The backend tagged its weights external, so they arrive as a separate file
    # rather than inside the program. Write directly instead of materialising a second
    # in-memory copy through data_files(); production PTDs are nearly the whole flash.
    extra = ""
    if external_weights:
        et.write_tensor_data_to_file(MAIN)
        ptd = os.path.join(MAIN, "esp32_weights.ptd")
        extra = f"  + esp32_weights.ptd {os.path.getsize(ptd):,} B"
    print(f"  {out_name}: {os.path.getsize(out):,} B, {n_del} delegate call(s){extra}")
    return n_del


def export_embedding_formats():
    """Execute the two non-i8 embedding storage paths in the device gate.

    Whole language-model fixtures exercise ordinary i8 embeddings, but merely
    declaring i4/i16 opcodes in the ABI does not prove either gather on the
    runtime.  This deliberately awkward table covers the format-specific edges:
    D=131 makes i4 pad one nibble, crosses the 128-element scale boundary, and
    leaves a three-element final group.  The 27 indices permute all 23 rows before
    repeating four, so the checksum sees both arbitrary order and repeated reads.
    """
    import struct

    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.serialization.espressif_blob import (
        BLOB_HEADER_BYTES,
        FLAG_FP16_SCALES,
        MAGIC,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
        OP_EMBED_I4,
        OP_EMBED_I16,
        VERSION,
    )
    from executorch.backends.esp32s3.serialization.weight_formats import SCALE_GROUP
    from executorch.backends.esp32s3.test.reference import embedding_iq_absmax

    class EmbeddingFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(23, 131)

        def forward(self, indices):
            return self.table(indices)

    indices = torch.tensor(
        [[(j * 7 + 3) % 23 for j in range(27)]], dtype=torch.int64)
    references = {}
    # fp16 group scales at i4; fp32 at i16, whose amax/32767 scale fp16 holds
    # only as a subnormal (see embedding_scale_layout).
    for fmt, opcode, k_pad, flags in (
            ("i4", OP_EMBED_I4, 132, FLAG_FP16_SCALES),
            ("i16", OP_EMBED_I16, 131, 0)):
        torch.manual_seed(0)
        model = EmbeddingFixture().eval()
        ep = torch.export.export(model, (indices,))
        lowered = lower_esp32(
            ep, [Esp32Partitioner(format_policy=fmt)])
        apply_passes(lowered)
        graph = lowered.exported_program().graph_module.graph
        n_del = sum(
            node.op == "call_function" and "call_delegate" in str(node.target)
            for node in graph.nodes)
        if n_del != 1:
            raise RuntimeError(
                f"{fmt} embedding expected one delegate, found {n_del}")

        et = lowered.to_executorch(
            config=ExecutorchBackendConfig(delegate_alignment=16))
        program = bytes(et.buffer)
        at = program.find(struct.pack("<I", MAGIC))
        if at < 0:
            raise RuntimeError(f"{fmt} embedding has no ESP32 backend blob")
        magic, version, n_ops, ext_off = struct.unpack_from("<4I", program, at)
        desc = dict(zip(
            OP_DESC_FIELDS,
            struct.unpack_from(OP_DESC_FORMAT, program,
                               at + BLOB_HEADER_BYTES)))
        contract = (
            magic == MAGIC and version == VERSION and n_ops == 1 and ext_off == 0
            and desc["opcode"] == opcode
            and desc["flags"] == flags
            and (desc["m"], desc["k"], desc["n"]) == (27, 131, 23)
            and desc["k_pad"] == k_pad
            and desc["scale_group"] == SCALE_GROUP)
        if not contract:
            raise RuntimeError(f"{fmt} embedding descriptor drifted: {desc}")

        name = f"embed_{fmt}"
        out = os.path.join(MAIN, f"{name}.pte")
        with open(out, "wb") as file:
            file.write(program)
        quantized = embedding_iq_absmax(
            model.table.weight.detach().numpy(), indices.numpy(), fmt)
        references[name] = quantized
        print(f"  {name}.pte: {len(program):,} B, one {fmt} embedding; "
              f"checksum {checksum(quantized):.6f}")
    return references


def export_selected_head():
    """Execute dynamic legal-token row selection through the real runtime.

    The table has 257 rows while the ET output has seven.  Runtime IDs are a
    non-monotonic permutation, so treating them as a contiguous prefix cannot
    accidentally pass.  M=2 also covers reuse of one selected list across more
    than one activation row.
    """
    import numpy as np
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.test.reference import addmm_iq_absmax
    from executorch.backends.esp32s3.test.models import softcap_f32_device

    class SelectedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(257, 64))

        def forward(self, value, rows):
            selected = torch.index_select(self.weight, 0, rows)
            logits = torch.nn.functional.linear(value, selected) * 0.196116
            return torch.tanh(logits / 20.0) * 20.0

    torch.manual_seed(0)
    model = SelectedHead().eval()
    value = ramp_input((2, 64))
    rows = torch.tensor([(j * 7 + 3) % 257 for j in range(7)],
                        dtype=torch.int64)
    lowered = lower_esp32(
        torch.export.export(model, (value, rows)),
        [Esp32Partitioner(format_policy="i8")],
    )
    apply_passes(lowered)
    graph = lowered.exported_program().graph_module.graph
    targets = [str(node.target) for node in graph.nodes
               if node.op == "call_function"]
    if sum("call_delegate" in target for target in targets) != 1 or any(
            fragment in target for target in targets for fragment in
            ("index_select", "aten.mm.", "aten.tanh.", "aten.div.")):
        raise RuntimeError(f"selected head did not fuse completely: {targets}")
    et = lowered.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "selected_head.pte")
    with open(out, "wb") as file:
        file.write(et.buffer)

    table = model.weight.detach().numpy().astype(np.float32)
    selected = table[rows.numpy()]
    # The scale, divide, tanh and multiply fuse into the head's soft cap.
    quantized = softcap_f32_device(addmm_iq_absmax(
        value.numpy().astype(np.float32), selected,
        np.zeros(len(rows), dtype=np.float32), fmt="i8"), 0.196116, 20.0)
    print(f"  selected_head.pte: {os.path.getsize(out):,} B, "
          f"257->7 rows; checksum {checksum(quantized):.6f}")
    return quantized


def export_embed_linear():
    """`nn.Embedding -> nn.Linear` as ONE delegate: the gather writes a chain slot.

    The first two layers of most models: the runtime's gather writes the slot
    and the matmul reads it back without a boundary tensor.
    Indices are 1-D on purpose -- a 2-D index puts a view between the gather
    and the matmul, which breaks the partition, so the two do not chain.
    """
    import numpy as np
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.test.reference import (
        addmm_iq_absmax,
        embedding_iq_absmax,
    )

    class EmbedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(23, 24)
            self.fc = torch.nn.Linear(24, 8)

        def forward(self, indices):
            return self.fc(self.emb(indices))

    torch.manual_seed(0)
    model = EmbedLinear().eval()
    # The firmware's fill_inputs(long_modulus=23) rule, so the device consumes
    # exactly these indices.
    indices = torch.tensor([(j * 7 + 3) % 23 for j in range(6)],
                           dtype=torch.int64)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lowered = lower_esp32(
            torch.export.export(model, (indices,)),
            [Esp32Partitioner(format_policy="i8")])
        apply_passes(lowered)
    log = buf.getvalue()
    graph = lowered.exported_program().graph_module.graph
    targets = [str(n.target) for n in graph.nodes if n.op == "call_function"]
    delegates = sum("call_delegate" in t for t in targets)
    if delegates != 1 or any("embedding" in t or "addmm" in t for t in targets):
        raise RuntimeError(f"embed->linear did not fuse into one delegate: "
                           f"{targets}")
    if "embed_i8[6x24x23]" not in log or ">s0" not in log or "<s0" not in log:
        raise RuntimeError(f"embed->linear chain markers missing:\n{log}")
    et = lowered.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "embed_linear.pte")
    with open(out, "wb") as file:
        file.write(et.buffer)

    table = model.emb.weight.detach().numpy().astype(np.float32)
    rows = embedding_iq_absmax(table, indices.numpy(), "i8")     # [6, 24]
    y = addmm_iq_absmax(
        np.ascontiguousarray(rows, dtype=np.float32),
        model.fc.weight.detach().numpy().astype(np.float32),
        model.fc.bias.detach().numpy().astype(np.float32), fmt="i8")
    print(f"  embed_linear.pte: {os.path.getsize(out):,} B, "
          f"gather -> chain slot -> matmul; checksum {checksum(y):.6f}")
    return y


def export_grouped_scale_formats():
    """Exercise grouped addmm with the scale bytes actually stored as fp16.

    Other rows cover grouped i4 accumulation with incoming fp32 QAT scales,
    and the embedding fixtures cover fp16 scale decoding without a dot
    product.  ``group_weights=True`` combines both branches for wide
    learned projections, so this fixture does too at i4 and i8.

    grouped_i4_wide is the i4 one at 40 output columns, wide enough for the
    min-latency split (kSplitMinCols, 32): main.cpp runs it single-core and
    as grouped_i4_split in the split pass, the parity firmware's row that
    splits grouped per-row scales across the two cores (the same values:
    a split moves whole columns, never a column's arithmetic).
    """
    import struct

    import numpy as np
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.serialization.espressif_blob import (
        BLOB_HEADER_BYTES,
        FLAG_FP16_SCALES,
        MAGIC,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
        VERSION,
    )
    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        SCALE_GROUP,
    )
    from executorch.backends.esp32s3.test.reference import addmm_iq_absmax

    class GroupedProjection(torch.nn.Module):
        def __init__(self, columns=17):
            super().__init__()
            self.proj = torch.nn.Linear(131, columns)

        def forward(self, value):
            return self.proj(value)

    def runtime_reference(model, value, fmt_name, grouped=True, fp16=True):
        """addmm_q_row_range's per-group float accumulation, bit for bit."""
        return addmm_iq_absmax(
            value.numpy().astype(np.float32),
            model.proj.weight.detach().numpy().astype(np.float32),
            model.proj.bias.detach().numpy().astype(np.float32), fmt_name,
            group=SCALE_GROUP if grouped else None,
            fp16_scales=fp16 and grouped)

    references = {}
    for fmt_name, opcode, k_pad, columns, name in (
            ("i4", FORMATS["i4"].opcode, 160, 17, "grouped_i4"),
            ("i8", FORMATS["i8"].opcode, 144, 17, "grouped_i8"),
            ("i4", FORMATS["i4"].opcode, 160, 40, "grouped_i4_wide")):
        torch.manual_seed(0)
        model = GroupedProjection(columns).eval()
        value = ramp_input((3, 131))
        edge = lower_esp32(
            torch.export.export(model, (value,)),
            [Esp32Partitioner(format_policy=fmt_name, group_weights=True)],
        )
        apply_passes(edge)
        graph = edge.exported_program().graph_module.graph
        n_delegate = sum(
            node.op == "call_function" and "call_delegate" in str(node.target)
            for node in graph.nodes)
        if n_delegate != 1:
            raise RuntimeError(
                f"{name} expected one delegate, found {n_delegate}")

        program = bytes(edge.to_executorch(config=ExecutorchBackendConfig(
            delegate_alignment=16)).buffer)
        at = program.find(struct.pack("<I", MAGIC))
        if at < 0:
            raise RuntimeError(f"{name} has no ESP32 backend blob")
        magic, version, n_ops, ext_off = struct.unpack_from("<4I", program, at)
        desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
            OP_DESC_FORMAT, program, at + BLOB_HEADER_BYTES)))
        contract = (
            magic == MAGIC and version == VERSION and n_ops == 1 and ext_off == 0
            and desc["opcode"] == opcode
            and desc["flags"] == FLAG_FP16_SCALES
            and (desc["m"], desc["k"], desc["n"]) == (3, 131, columns)
            and desc["k_pad"] == k_pad
            and desc["scale_group"] == SCALE_GROUP
        )
        if not contract:
            raise RuntimeError(
                f"{name} descriptor drifted: {desc}")

        output = runtime_reference(model, value, fmt_name)
        wrong_ungrouped = runtime_reference(
            model, value, fmt_name, grouped=False, fp16=False)
        wrong_full_scale = runtime_reference(
            model, value, fmt_name, grouped=True, fp16=False)
        group_gap = float(np.max(np.abs(output - wrong_ungrouped)))
        fp16_gap = float(np.max(np.abs(output - wrong_full_scale)))
        if group_gap == 0.0 or fp16_gap == 0.0:
            raise RuntimeError(
                f"{name} lacks a positive control: "
                f"group gap {group_gap}, fp16 gap {fp16_gap}")

        out_path = os.path.join(MAIN, f"{name}.pte")
        with open(out_path, "wb") as file:
            file.write(program)
        references[name] = output
        print(f"  {name}.pte: {len(program):,} B, grouped fp16 scales; "
              f"checksum {checksum(output):.6f}, controls "
              f"{group_gap:.3g}/{fp16_gap:.3g}")
    # The split pass's row: the wide program, the same values.
    references["grouped_i4_split"] = references["grouped_i4_wide"]
    return references


def export_grouped_depthwise_i4():
    """Exercise packed i4 and multiple fp16 scale groups in the direct loop."""
    import struct

    from executorch.exir import ExecutorchBackendConfig, to_edge

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax
    from executorch.backends.esp32s3.serialization.espressif_blob import (
        BLOB_HEADER_BYTES,
        FLAG_CONV,
        FLAG_FP16_SCALES,
        MAGIC,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
        VERSION,
    )
    from executorch.backends.esp32s3.serialization.weight_formats import (
        FORMATS,
        SCALE_GROUP,
    )
    from executorch.backends.esp32s3.test.models import DwConvGroupedScaleD

    torch.manual_seed(0)
    model = DwConvGroupedScaleD().eval()
    value = ramp_input((1, 2, 15, 15))
    ep = torch.export.export(model, (value,))
    edge = lower_esp32(
        ep,
        [Esp32Partitioner(format_policy="i4", group_weights=True, conv=True)],
    )
    apply_passes(edge)
    graph = edge.exported_program().graph_module.graph
    n_delegate = sum(
        node.op == "call_function" and "call_delegate" in str(node.target)
        for node in graph.nodes)
    if n_delegate != 1:
        raise RuntimeError(
            f"grouped depthwise i4 expected one delegate, found {n_delegate}")

    program = bytes(edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16)).buffer)
    at = program.find(struct.pack("<I", MAGIC))
    if at < 0:
        raise RuntimeError("grouped depthwise i4 has no ESP32 backend blob")
    magic, version, n_ops, ext_off = struct.unpack_from("<4I", program, at)
    desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, program, at + BLOB_HEADER_BYTES)))
    contract = (
        magic == MAGIC and version == VERSION and n_ops == 1 and ext_off != 0
        and desc["opcode"] == FORMATS["i4"].opcode
        and desc["flags"] == FLAG_CONV | FLAG_FP16_SCALES
        and (desc["m"], desc["k"], desc["n"]) == (225, 169, 2)
        and desc["k_pad"] == 192
        and desc["scale_group"] == SCALE_GROUP
    )
    if not contract:
        raise RuntimeError(f"grouped depthwise i4 descriptor drifted: {desc}")

    host_edge = to_edge(torch.export.export(model, (value,)))
    host_graph = host_edge.exported_program().graph_module.graph
    node = next(node for node in host_graph.nodes
                if node.op == "call_function"
                and "convolution" in str(node.target))
    geom = admit(node)
    output = depthwise_iq_absmax(
        value, model.c.weight, model.c.bias, geom, fmt="i4",
        group_weights=True)
    wrong_ungrouped = depthwise_iq_absmax(
        value, model.c.weight, model.c.bias, geom, fmt="i4",
        group_weights=False)
    wrong_fp32_scale = depthwise_iq_absmax(
        value, model.c.weight, model.c.bias, geom, fmt="i4",
        group_weights=True, fp16_scales=False)
    group_gap = float((output - wrong_ungrouped).abs().max())
    fp16_gap = float((output - wrong_fp32_scale).abs().max())
    if group_gap == 0.0 or fp16_gap == 0.0:
        raise RuntimeError(
            "grouped depthwise i4 lacks a positive control: "
            f"group gap {group_gap}, fp16 gap {fp16_gap}")

    out_path = os.path.join(MAIN, "dwconv_grouped_i4.pte")
    with open(out_path, "wb") as file:
        file.write(program)
    print(f"  dwconv_grouped_i4.pte: {len(program):,} B, two scale groups; "
          f"checksum {checksum(output):.6f}, controls "
          f"{group_gap:.3g}/{fp16_gap:.3g}")
    return output


def export_mlp_perop():
    """The exact MLP with one delegate boundary inserted between its matmuls.

    A clone is an export-time partition barrier: it is not claimed, so the two
    Linears become two delegate calls. The standard post-lowering pass then
    erases that identity copy. The emitted graph therefore has the same tensor
    dataflow and arithmetic as mlp_delegated, but two delegate boundaries
    instead of one. This is a fixture construction technique, not a deployment
    recommendation.
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.backends.esp32s3.test.models import MLP, ramp_input

    class PerOpMLP(MLP):
        def forward(self, x):
            hidden = torch.relu(self.fc1(x))
            return self.fc2(hidden.clone())

    torch.manual_seed(0)
    model = PerOpMLP().eval()
    x = ramp_input((1, 96))
    ep = torch.export.export(model, (x,))
    lowered = lower_esp32(ep, [Esp32Partitioner()])
    apply_passes(lowered)
    gm = lowered.exported_program().graph_module
    n_del = sum(
        1 for n in gm.graph.nodes
        if n.op == "call_function" and "call_delegate" in str(n.target))
    n_clone = sum(1 for n in gm.graph.nodes if "clone" in str(n.target))
    if n_del != 2 or n_clone != 0:
        raise RuntimeError(
            f"per-op MLP expected 2 delegates and no surviving clone; "
            f"got {n_del} delegates, {n_clone} clones")
    et = lowered.to_executorch(
        config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "mlp_perop.pte")
    with open(out, "wb") as f:
        f.write(et.buffer)
    print(f"  mlp_perop.pte: {os.path.getsize(out):,} B, "
          "2 delegate calls, identity barrier removed")


def check_shared_weight_dedup():
    """The looped/ALBERT mechanism check: one block's weights, three call sites.

    Export unrolls the loop, so every partition that claims one iteration's matmuls
    holds the SAME weight tensors. With inline blobs that would triple the artifact;
    with external weights the content-addressed keys must collapse the copies. This
    measures it rather than assuming it: the same model exported shared vs unshared
    (three deepcopied blocks), and the shared .ptd must come in well under the
    unshared one. Host-side only -- no partition or firmware involvement -- because
    the claim is about bytes in the artifact, not about execution.
    """
    import copy
    import tempfile

    import torch

    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.backends.esp32s3.test.models import LoopedNet, ramp_input

    def ptd_bytes(model):
        ep = torch.export.export(model.eval(), (ramp_input((1, 8, 8)),))
        lowered = to_edge_transform_and_lower(
            ep, partitioner=[Esp32Partitioner(external_weights=True)])
        et = lowered.to_executorch(config=ExecutorchBackendConfig(
            delegate_alignment=16))
        with tempfile.TemporaryDirectory() as d:
            et.write_tensor_data_to_file(d)
            return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))

    torch.manual_seed(0)
    shared = LoopedNet()

    class Unshared(torch.nn.Module):
        def __init__(self, src):
            super().__init__()
            self.up, self.head = src.up, src.head
            self.blocks = torch.nn.ModuleList(
                copy.deepcopy(src.block) for _ in range(src.loops))
            # Re-randomise each copy. Deepcopied blocks hold byte-identical weights,
            # which the SAME content-addressed dedup this check exists to verify would
            # collapse: shared and "unshared" would both measure one block's worth. A
            # control that the mechanism under test can act on is not a control.
            with torch.no_grad():
                for i, blk in enumerate(self.blocks):
                    g = torch.Generator().manual_seed(100 + i)
                    for prm in blk.parameters():
                        prm.copy_(torch.randn(prm.shape, generator=g) * 0.05)

        def forward(self, x):
            x = self.up(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    a, b = ptd_bytes(shared), ptd_bytes(Unshared(shared))
    print(f"  looped shared-weight dedup: shared .ptd {a:,} B vs unshared {b:,} B")
    if not a < 0.55 * b:
        raise RuntimeError(
            f"shared-weight dedup did not fire: {a:,} B vs {b:,} B -- three copies "
            f"of one block's weights are reaching the artifact")


def export_static():
    """Export the MLP with CALIBRATED static activation scales; return its reference.

    Two-step lowering because calibration must see the edge graph before the
    partitioner consumes the matmuls. The recorded amaxes
    feed models.mlp_static_out directly: one set of calibration numbers for
    both the blob and the reference, no replayed forward that merely ought to
    match. fc1's amax must equal the input's outright -- a boundary op calibrated
    on the very tensor it will run is the bit-identity half of the story -- and
    that is asserted here, not assumed.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    from executorch.backends.esp32s3.quantizer.calibration import calibrate_static_act
    from executorch.backends.esp32s3.test.models import mlp_static_out

    factory, shape = SPECS["mlp"]
    torch.manual_seed(0)
    model = factory().eval()
    x = ramp_input(shape)
    ep = torch.export.export(model, (x,))
    edge = to_edge_transform_and_lower(ep)          # no partitioner yet
    cal = calibrate_static_act(edge, [(x,)])
    if len(cal) != 2:
        raise RuntimeError(f"expected 2 calibrated matmuls in the MLP, got {cal}")
    (n1, amax1), (n2, amax2) = cal
    if amax1 != float(x.abs().max()):
        raise RuntimeError(
            f"boundary-op calibration should be the input absmax exactly: "
            f"{amax1} vs {float(x.abs().max())}")
    edge = edge.to_backend(Esp32Partitioner(static_act=True))
    apply_passes(edge)
    et = edge.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "mlp_static.pte")
    with open(out, "wb") as f:
        f.write(et.buffer)
    ref = mlp_static_out(amax1 / 127.0, amax2 / 127.0)
    print(f"  mlp_static.pte: {os.path.getsize(out):,} B  "
          f"(amax {amax1:.6f}/{amax2:.6f}, static ref {checksum(ref):.6f})")
    return ref


def export_conv_static():
    """A CONV under calibrated activation scales.

    preprocess requires esp32_act_amax on every quantised op, and conv delegation
    is on by default, so the calibrator walks convolutions as well as mm/addmm:
    one that missed them would fail every model with a convolution at export.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    from executorch.backends.esp32s3.quantizer.calibration import calibrate_static_act
    from executorch.backends.esp32s3.test.models import convd_static_out

    factory, shape = SPECS["convd"]
    torch.manual_seed(0)
    model = factory().eval()
    x = ramp_input(shape)
    ep = torch.export.export(model, (x,))
    edge = to_edge_transform_and_lower(ep)          # no partitioner yet
    cal = calibrate_static_act(edge, [(x,)])
    if len(cal) != 1:
        raise RuntimeError(f"expected 1 calibrated conv, got {cal}")
    name, amax = cal[0]
    if "convolution" not in name:
        raise RuntimeError(f"calibrated the wrong node: {name}")
    edge = edge.to_backend(Esp32Partitioner(static_act=True, conv=True))
    apply_passes(edge)
    et = edge.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "conv_static.pte")
    with open(out, "wb") as f:
        f.write(et.buffer)
    ref = convd_static_out(amax / 127.0)
    print(f"  conv_static.pte: {os.path.getsize(out):,} B  "
          f"(amax {amax:.6f}, static ref {checksum(ref):.6f})")
    return ref


def export_qk():
    """Export AttnQK with the opt-in int8 q@kT; return its device reference.

    The row's model is AttnQKScores, whose first output is the raw q@kT
    scores: the PARITY row holds them to the int8 mirror (bmm_qk_i8 behind
    the Linears' device mirrors) bit for bit, which is the correctness
    claim. Also reports the attention output's quantisation cost against
    fp32 eager at tensor level -- informational, not asserted, per the
    format-ladder rule: a coarser quantiser is a different number, not a
    wrong one.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    from executorch.backends.esp32s3.test.models import (
        AttnQKScores,
        attnqk_qk_out,
        attnqk_qk_scores,
    )

    factory, shape = SPECS["attnqk"]
    torch.manual_seed(0)
    model = AttnQKScores().eval()
    x = ramp_input(shape)
    ep = torch.export.export(model, (x,))
    lowered = lower_esp32(ep, [Esp32Partitioner(quantize_qk=True)])
    apply_passes(lowered)
    gm = lowered.exported_program().graph_module
    n_bmm = sum(1 for n in gm.graph.nodes
                if n.op == "call_function" and "bmm" in str(n.target))
    if n_bmm != 1:
        raise RuntimeError(
            f"expected exactly the p@v bmm to stay portable, found {n_bmm}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "attnqk_qk.pte")
    with open(out, "wb") as f:
        f.write(et.buffer)
    with torch.no_grad():
        y_fp = factory_eval_fp32(factory, shape)
    y_q = attnqk_qk_out()
    rel = float((y_q - y_fp).abs().max() / y_fp.abs().max())
    print(f"  attnqk_qk.pte: {os.path.getsize(out):,} B  "
          f"(int8 q@kT vs fp32: max rel {rel:.3e}, informational; "
          f"the row holds the raw scores exactly)")
    return attnqk_qk_scores()


def export_qk_kcache():
    """Export AttnKC8 with the opt-in int8 q@kT; return its device reference.

    The keys reach the delegate as an int8 cache -- codes and per-row scales
    from espressif::quantize_rows_i8 -- so the bmm must lower to
    OP_BMM_KCACHE_I8, with the cast, the scale multiply and the transpose all
    folded into it. The row's model is AttnKC8Scores (raw scores first), and
    the reference is attnqk_qk's scores: same weights, same codes, bit for
    bit.
    """
    from executorch.exir import ExecutorchBackendConfig

    from executorch.backends.esp32s3.serialization.espressif_blob import OP_BMM_KCACHE_I8
    from executorch.backends.esp32s3.test.models import (
        AttnKC8Scores,
        attnqk_qk_scores,
    )

    factory, shape = SPECS["attnkc8"]
    torch.manual_seed(0)
    model = AttnKC8Scores().eval()
    x = ramp_input(shape)
    ep = torch.export.export(model, (x,))
    lowered = lower_esp32(ep, [Esp32Partitioner(quantize_qk=True)])
    apply_passes(lowered)
    gm = lowered.exported_program().graph_module
    portable = [str(n.target) for n in gm.graph.nodes
                if n.op == "call_function"
                and "executorch_call_delegate" not in str(n.target)]
    n_bmm = sum("aten.bmm" in t for t in portable)
    n_cast = sum("_to_dim_order_copy" in t or "_to_copy" in t for t in portable)
    if n_bmm != 1 or n_cast != 0:
        raise RuntimeError(
            f"expected only the p@v bmm portable and the key dequantisation "
            f"folded; found {n_bmm} bmm, {n_cast} casts")
    from executorch.exir.lowered_backend_module import LoweredBackendModule
    opcodes = []
    for _, module in gm.named_modules():
        if isinstance(module, LoweredBackendModule):
            blob = module.processed_bytes
            n_ops = struct.unpack_from("<I", blob, 8)[0]
            opcodes += [struct.unpack_from("<I", blob, 16 + 64 * i)[0]
                        for i in range(n_ops)]
    if opcodes.count(OP_BMM_KCACHE_I8) != 1:
        raise RuntimeError(f"expected one OP_BMM_KCACHE_I8, got opcodes {opcodes}")
    et = lowered.to_executorch(config=ExecutorchBackendConfig(delegate_alignment=16))
    out = os.path.join(MAIN, "attnkc8_qk.pte")
    with open(out, "wb") as f:
        f.write(et.buffer)
    print(f"  attnkc8_qk.pte: {os.path.getsize(out):,} B  "
          f"(int8 K cache; reference = attnqk_qk's scores, exactly)")
    return attnqk_qk_scores()


def factory_eval_fp32(factory, shape):
    """The fp32 eager output at seed 0, for accuracy reporting."""
    torch.manual_seed(0)
    m = factory().eval()
    with torch.no_grad():
        return m(ramp_input(shape))


def main():
    """Export every delegated fixture; return {row: reference output} for the
    rows whose references depend on export-time values (calibration amaxes,
    the exact tensors a fixture packed) or are computed here anyway."""
    from executorch.backends.esp32s3.test.models import block_i8_out, mlp_i8_out

    # Only the block goes external. It is the realistic case -- a big model whose
    # weights belong in flash -- and one .ptd filename cannot serve two programs.
    # The mlp stays self-contained in its .pte, which is the right call for a small
    # model and exercises the other half of the flag.
    export_one("block", "block_delegated.pte", external_weights=True)
    export_one("block", "block_i4.pte", external_weights=False,
               format_policy="i4")
    export_one("block", "block_i16.pte", external_weights=False,
               format_policy="i16")
    export_one("mlp", "mlp_delegated.pte", external_weights=False)
    # Opcode 1: exact inline [K,N] fp32 weights. This is a backend diagnostic,
    # not a deployment format; it keeps the runtime branch numerically live.
    export_one("mlp", "mlp_fp32_backend.pte", format_policy="fp32")
    export_mlp_perop()
    # The same MLP down each rung of the format ladder. Same graph, same partition, same
    # arithmetic order -- so a disagreement can only be the format path. i4 exercises
    # the nibble unpack feeding the existing PIE kernel; i16 exercises the int64
    # accumulator and the wider activation scale.
    export_one("mlp", "mlp_i4.pte", external_weights=False, format_policy="i4")
    embed_refs = export_embedding_formats()
    ref_selected_head = export_selected_head()
    ref_embed_linear = export_embed_linear()
    grouped_refs = export_grouped_scale_formats()
    # conv2d through the im2col path. Padded and strided on purpose: padding is
    # where the gather reads outside the input, stride is where the output row
    # index stops tracking the input row index.
    export_one("convd", "convd_delegated.pte", external_weights=False,
               conv=True)
    export_one("convresd", "convres_delegated.pte", external_weights=False,
               conv=True)
    export_one("convresviewd", "convresview_delegated.pte",
               external_weights=False, conv=True)
    export_one("convresd", "convres_i16.pte", external_weights=False,
               format_policy="i16", conv=True)
    # The weight-format ladder on a CONVOLUTION. Conv reuses the matmul kernels
    # through im2col, so this checks that the reuse is real rather than assumed.
    export_one("convd", "conv_i4.pte", external_weights=False,
               format_policy="i4", conv=True)
    export_one("convd", "conv_i16.pte", external_weights=False,
               format_policy="i16", conv=True)
    export_one("conv1dd", "conv1d_delegated.pte", external_weights=False,
               conv=True)
    export_one("conv3dd", "conv3d_delegated.pte", external_weights=False,
               conv=True)
    export_one("dwconvd", "dwconv_delegated.pte", external_weights=False,
               conv=True)
    export_one("dwconvd", "dwconv_i4.pte", external_weights=False,
               format_policy="i4", conv=True)
    export_one("dwconvd", "dwconv_i16.pte", external_weights=False,
               format_policy="i16", conv=True)
    ref_dw_grouped_i4 = export_grouped_depthwise_i4()
    export_one("dwconvresd", "dwconvres_delegated.pte",
               external_weights=False, conv=True)
    export_one("dwconv1dd", "dwconv1d_delegated.pte", external_weights=False,
               conv=True)
    export_one("convrelud", "convrelu_delegated.pte", external_weights=False,
               conv=True)
    export_one("convsilud", "convsilu_delegated.pte", external_weights=False,
               conv=True)
    export_one("convtd", "convt_delegated.pte", external_weights=False,
               conv=True)
    export_one("convtd", "convt_i4.pte", external_weights=False,
               format_policy="i4", conv=True)
    export_one("convtd", "convt_i16.pte", external_weights=False,
               format_policy="i16", conv=True)
    # Output padding (the decoder's exact 2x up-conv) and dilation, with
    # per-axis geometry: the runtime's size check takes the padded extent,
    # and the gather's mapping spaces the taps by the dilation. The int16
    # rung runs the other instantiation of the same gather.
    export_one("convtopd", "convtop_delegated.pte", external_weights=False,
               conv=True)
    export_one("convtdild", "convtdil_delegated.pte", external_weights=False,
               conv=True)
    export_one("convtdild", "convtdil_i16.pte", external_weights=False,
               format_policy="i16", conv=True)
    export_one("grpconvd", "grpconv_delegated.pte", external_weights=False,
               conv=True)
    export_one("grpconvedged", "grpconvedge_delegated.pte",
               external_weights=False, conv=True)
    export_one("grpconv8d", "grpconv8_delegated.pte", external_weights=False,
               conv=True)
    export_one("grpconv8d", "grpconv8_i4.pte", external_weights=False,
               format_policy="i4", conv=True)
    export_one("grpconvrelud", "grpconvrelu_delegated.pte",
               external_weights=False, conv=True)
    export_one("grpconv8d", "grpconv8_i16.pte", external_weights=False,
               format_policy="i16", conv=True)
    # The SAME mlp asking for the min-latency posture in its compile spec. It
    # exists to be loaded ALONGSIDE mlp_delegated.pte (which does not ask), so
    # the two live handles can be shown to keep their own answer -- the
    # reentrancy claim. Identical weights and graph, so its reference output is
    # mlp_i8_out() too, and any difference would be the split miscomputing.
    export_one("mlp", "mlp_minlat.pte", external_weights=False,
               min_latency=True)
    export_one("mlp", "mlp_i16.pte", external_weights=False, format_policy="i16")
    # Both fates of a fusable relu: the final one fuses into the delegate (its
    # descriptor prints "+relu"), the mid-chain one stays portable because claiming
    # it would merge the two matmul partitions.
    export_one("mlpr", "mlpr_delegated.pte", external_weights=False)
    # Two fused activations at once: fc_a's silu diamond and fc_b's exact gelu, both
    # terminal; descriptors print "+silu" and "+gelu".
    export_one("mlps", "mlps_delegated.pte", external_weights=False)
    # The fused epilogues. The fused multiply as a GeGLU-tanh pair (in place,
    # "*pair") chained into its down projection and as an output gate
    # ("+sigmoid*in"), at M=4 (row-blocked group) and M=1 (decode group, and
    # the dual-core split row); then five one-op activations behind a
    # hardswish trunk.
    export_one("epiglud", "epiglu_delegated.pte", external_weights=False)
    export_one("epiglud", "epiglu_m1.pte", external_weights=False,
               shape=(1, 48))
    export_one("epiactsd", "epiacts_delegated.pte", external_weights=False)
    # The same MLP once more with calibrated static activation scales. Its
    # reference depends on the calibration numbers recorded during export, so it
    # is returned from here rather than recomputed from a replay.
    # The same block at T=24. Row blocking processes 8 rows at a time, so this
    # is the only fixture where the blocked path runs with m0 > 0 -- and the
    # block carries two FUSED RESIDUALS, whose runtime offset arithmetic
    # (rbase + mb*y_stride) is exactly what an m0 of zero never exercises.
    # Inline weights deliberately: one .ptd filename cannot serve two programs.
    export_one("block", "block_wide.pte", external_weights=False,
               shape=(1, 24, 96))
    ref_static = export_static()
    ref_conv_static = export_conv_static()
    # The attention core with the opt-in int8 q@kT.
    ref_qk = export_qk()
    # ... and with its keys kept as a persistent int8 K cache.
    ref_qk_kc8 = export_qk_kcache()

    check_shared_weight_dedup()
    ref_mlp = mlp_i8_out()
    print(f"\nmlp   int8 checksum : {checksum(ref_mlp):.6f}   (device must match)")
    print(f"block int8 checksum : {checksum(block_i8_out()):.6f}   "
          f"(device must match)")
    # The two live handles run the same weights and inputs as mlp_delegated, so
    # they answer to the same reference. Listed explicitly rather than aliased,
    # because the row that would matter if reentrancy broke is a WRONG number,
    # and a wrong number has to have something to be wrong against.
    # The concurrent pair answers to the same reference: whichever handle loses
    # the worker lease runs single-core, and that must not be visible in the
    # values -- only in the timing.
    return {"mlp_static": ref_static, "conv_static": ref_conv_static,
            "attnqk_qk": ref_qk, "attnkc8_qk": ref_qk_kc8,
            "mlp_bg_handle": ref_mlp, "mlp_ml_handle": ref_mlp,
            # The min-latency handle called from the worker's own core keeps
            # every op whole: no fork, the same values.
            "mlp_ml_core1": ref_mlp,
            "mlp_race_bg": ref_mlp, "mlp_race_ml": ref_mlp,
            "block_wide": block_i8_out((1, 24, 96)),
            "dwconv_grouped_i4": ref_dw_grouped_i4,
            "selected_head": ref_selected_head,
            "embed_linear": ref_embed_linear,
            **embed_refs, **grouped_refs}


if __name__ == "__main__":
    main()
