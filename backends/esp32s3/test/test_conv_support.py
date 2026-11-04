# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conv2d reduces to the matmul this backend already runs.

The conv delegate rests on one identity, and if it does not hold exactly then
every reuse it justifies -- the packer, `dot_i8_s3`, row blocking, the epilogues,
the whole format ladder -- is unjustified:

    conv2d(x, W)  ==  [C_out, K] @ im2col(x)      K = C_in*kh*kw

Claims:
  1. the identity holds through OUR gather (not just torch's `unfold`), because
     the column order has to match the weight's flattening order or the packed
     weight and the gathered row disagree,
  2. the quantised path reduces to `addmm_iq_absmax` -- the reference already
     proven bit-exact against the device -- so conv adds no new arithmetic,
  3. the geometry record round-trips through its uint16 packing,
  4. the block-size policy picks a B that fits SRAM and is dramatically better
     than the addmm row block, which is the 300x trap,
  5. every unsupported case is refused with a stated reason,
  6. a 1-in/1-out 1x1 conv at 4-bit weights, half a byte per pass, lowers:
     its traffic estimate never rounds to 0 B,
  7. a transposed convolution gathers to its eager output with output padding
     and with dilation, on every rank and with per-axis geometry, and the i8
     mirror tracks it; grouped and batched ones, a dense conv's output padding
     and an extent outside the runtime's window stay refused, by name.

    python test_conv_support.py
"""

import pathlib
import sys

import torch
import torch.nn as nn

FAILS = []


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def node_of(mod, x):
    from executorch.exir import to_edge

    gm = to_edge(torch.export.export(mod.eval(), (x,))
                 ).exported_program().graph_module
    return next(n for n in gm.graph.nodes if n.op == "call_function"
                and "convolution" in str(n.target))


def grouped_conv_chain():
    """A grouped conv's g descriptors write ONE tensor, so when it chains into
    another delegated op they must all write the same scratch slot -- which is
    also what the runtime's grouped-sequence check demands (identical args
    across the group) -- and the consumer must read that slot."""
    import contextlib
    import io
    import struct

    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.serialization.espressif_blob import (
        ARG_SCRATCH0,
        BLOB_HEADER_BYTES,
        BLOB_HEADER_FORMAT,
        OP_DESC_BYTES,
        OP_DESC_FIELDS,
        OP_DESC_FORMAT,
    )
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    class Chain(nn.Module):
        def __init__(self, act):
            super().__init__()
            self.grouped = nn.Conv2d(4, 4, 1, groups=2)
            self.dense = nn.Conv2d(4, 4, 1)
            self.act = act

        def forward(self, x):
            return self.dense(self.act(self.grouped(x)))

    print("a grouped conv chains into the next conv through ONE scratch slot:")
    for name, act in (("directly", nn.Identity()), ("through a relu", nn.ReLU())):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                edge = to_edge_transform_and_lower(
                    torch.export.export(Chain(act).eval(),
                                        (torch.randn(1, 4, 2, 2),)),
                    partitioner=[Esp32Partitioner()])
        except RuntimeError as e:
            check(f"{name}: lowers", False, str(e)[:90])
            continue
        gm = edge.exported_program().graph_module
        descs = []
        for node in gm.graph.nodes:
            if node.op == "get_attr" and "lowered_module" in node.target:
                blob = getattr(gm, node.target).processed_bytes
                n_ops = struct.unpack_from(BLOB_HEADER_FORMAT, blob)[2]
                descs += [dict(zip(OP_DESC_FIELDS, struct.unpack_from(
                    OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES + i * OP_DESC_BYTES)))
                    for i in range(n_ops)]
        args = [(d["in_arg"], d["out_arg"]) for d in descs]
        shared = (len(descs) == 3 and descs[0]["out_arg"] == descs[1]["out_arg"]
                  >= ARG_SCRATCH0 and descs[2]["in_arg"] == descs[0]["out_arg"])
        check(f"{name}: both groups write one slot, the dense conv reads it",
              shared, f"(in, out) per op: {args}")


# The FNV-1a hash of each depthwise row's output that main.cpp printed
# ("output bits") on a QEMU parity run of the firmware. The mirror must
# reproduce every bit.
DEPTHWISE_DEVICE_BITS = (
    # (row, model, input shape, rung, grouped scales, + residual, hash)
    ("dwconv_delegated", "DwConvD", (1, 8, 7, 7), "i8", False, False,
     0xb9b22822),
    ("dwconv_i4", "DwConvD", (1, 8, 7, 7), "i4", False, False, 0xa047b906),
    ("dwconv_i16", "DwConvD", (1, 8, 7, 7), "i16", False, False, 0xc189fc7e),
    ("dwconvres_delegated", "DwConvResD", (1, 8, 7, 7), "i8", False, True,
     0xa85a81bf),
    ("dwconv1d_delegated", "DwConv1dD", (1, 8, 24), "i8", False, False,
     0x11a59382),
    ("dwconv_grouped_i4", "DwConvGroupedScaleD", (1, 2, 15, 15), "i4", True,
     False, 0x484532fb),
)


def output_bits(values):
    """main.cpp's output_bits: FNV-1a over the float32 output's bytes."""
    from executorch.backends.esp32s3.test import models
    import executorch.backends.esp32s3.test.models as models

    return int(models.output_bits(values), 16)


def depthwise_device_bits():
    """depthwise_iq_absmax is the device's arithmetic bit for bit, on every
    depthwise parity row: i8, i4, i16, grouped fp16 scales, conv1d and the
    fused residual (added after the conv, as the device's epilogue does)."""
    from executorch.backends.esp32s3.test import models
    import executorch.backends.esp32s3.test.models as models
    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax

    print("the depthwise mirror reproduces the device's output bits:")
    for row, cls, shape, fmt, grouped, residual, want in DEPTHWISE_DEVICE_BITS:
        torch.manual_seed(0)
        model = getattr(models, cls)().eval()
        x = models.ramp_input(shape)
        geom = admit(node_of(model, x))
        y = depthwise_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt=fmt,
                                group_weights=grouped).reshape(x.shape[0],
                                                               geom.c_out, -1)
        if residual:
            y = y.reshape(x.shape) + x
        got = output_bits(y.detach().numpy())
        check(f"{row}: {got:08x}", got == want, f"device {want:08x}")


def one_to_one_i4():
    """A 1-in/1-out 1x1 convolution at 4-bit weights. Its im2col
    matmul has K = N = 1, half a byte per pass, and flash_traffic rounded the
    total down: a single pass cost 0 B, and conv_geometry's headroom note
    divided by it whenever the policy's block would take one pass where the
    runtime's takes several (m > 32). Each pass now costs its bytes rounded
    up, so a non-empty conv never streams 0 B, and the note's ratio comes
    from the pass counts."""
    import contextlib
    import io

    from executorch.backends.esp32s3 import Esp32Partitioner, Esp32Quantizer
    from executorch.backends.esp32s3._passes import lower_esp32
    from executorch.backends.esp32s3.operator_support.conv import (
        ConvGeom,
        flash_traffic,
    )

    print("a 1->1 1x1 conv at i4 streams whole bytes:")
    g = ConvGeom(11, 25, 1, 6, 13, 1, 1, 1, 2, 2, 0, 0, 1, 1)
    for b in (32, 78):
        tr, passes = flash_traffic(g, b, weight_bytes_per_elem=0.5)
        check(f"flash_traffic(B={b}) >= passes >= 1 at i4", tr >= passes >= 1,
              f"{tr} B in {passes} pass(es)")

    def lowered(model, x, quantize=False, **options):
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                if quantize:
                    from torchao.quantization.pt2e.quantize_pt2e import (
                        convert_pt2e,
                        prepare_pt2e,
                    )

                    gm = torch.export.export(model, (x,)).module()
                    prepared = prepare_pt2e(gm, Esp32Quantizer())
                    prepared(x)
                    model = convert_pt2e(prepared)
                edge = lower_esp32(torch.export.export(model, (x,)),
                                   [Esp32Partitioner(**options)])
        except Exception as e:  # the regression is precisely "this raised"
            return None, f"{type(e).__name__}: {str(e)[:200]}"
        left = [n.name for n in edge.exported_program().graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target)]
        return left, log.getvalue()

    torch.manual_seed(10)
    # m = 6 * 13 = 78 output positions, past the runtime's 32-row block.
    one = nn.Conv2d(1, 1, kernel_size=1, stride=2).eval()
    x = torch.randn(1, 1, 11, 25)
    for name, options, quantize in (
            ("format_policy='i4'", {"format_policy": "i4"}, False),
            ("Esp32Quantizer() (i4 codes)", {}, True)):
        left, log = lowered(one, x, quantize, **options)
        check(f"Conv2d(1, 1, 1) over 78 positions, {name}, lowers",
              left is not None, "" if left is not None else log)
        if left is not None:
            note = [l.strip() for l in log.splitlines() if "conv1x1[1->1" in l]
            check(f"{name}: the conv is delegated", not left, f"{left}")
            check(f"{name}: its note names the runtime's block and the "
                  f"policy's", bool(note) and "B=32 3pass" in note[0]
                  and "(B=78 would be 1pass" in note[0],
                  note[0] if note else log[-300:])
    # Either side of the runtime's block, and a Conv1d.
    for name, model, shape in (
            ("Conv2d(1, 1, 1) over 32 positions", nn.Conv2d(1, 1, 1), (1, 1, 4, 8)),
            ("Conv2d(1, 1, 1) over 33 positions", nn.Conv2d(1, 1, 1), (1, 1, 3, 11)),
            ("Conv1d(1, 1, 1) over 40 positions", nn.Conv1d(1, 1, 1), (1, 1, 40))):
        left, log = lowered(model.eval(), torch.randn(*shape),
                            format_policy="i4")
        check(f"{name}, format_policy='i4', lowers and delegates",
              left == [], log if left is None else f"{left}")


def transposed_cases():
    """(name, module, input shape) for the transposed gather: the U-Net
    up-conv and a strided 3x3, then output padding -- the decoder's exact 2x
    up-conv, 3x3 stride 2 pad 1 output_padding 1 -- and dilation, on every
    rank. Strides, paddings and dilations differ per axis, so an axis reading
    another's parameters shows, and the per-axis case pads its height by 2 at
    stride 2, which only its dilation of 3 allows (torch's bound is below
    max(stride, dilation))."""
    return (
        ("convT 2x2 s2 (U-Net up-conv)", nn.ConvTranspose2d(4, 6, 2, stride=2),
         (1, 4, 5, 5)),
        ("convT 3x3 s2 p1", nn.ConvTranspose2d(3, 5, 3, 2, 1), (1, 3, 6, 6)),
        ("convT 3x3 s2 p1 output_padding 1 (decoder up-conv)",
         nn.ConvTranspose2d(3, 5, 3, 2, 1, output_padding=1), (1, 3, 6, 6)),
        ("convT 3x3 dilation 2", nn.ConvTranspose2d(3, 5, 3, 1, 1, dilation=2),
         (1, 3, 6, 6)),
        ("convT 3x3 s(2,1) p(1,2) dil(3,2) output_padding(2,1)",
         nn.ConvTranspose2d(4, 6, 3, stride=(2, 1), padding=(1, 2),
                            dilation=(3, 2), output_padding=(2, 1)),
         (1, 4, 5, 6)),
        ("convT1d k4 s2 p1 dil 2 output_padding 1",
         nn.ConvTranspose1d(4, 6, 4, stride=2, padding=1, dilation=2,
                            output_padding=1), (1, 4, 9)),
        ("convT3d k3 s(2,1,2) p1 dil(1,2,1) output_padding(1,0,1)",
         nn.ConvTranspose3d(2, 3, 3, stride=(2, 1, 2), padding=1,
                            dilation=(1, 2, 1), output_padding=(1, 0, 1)),
         (1, 2, 3, 4, 4)),
    )


class DenseOutputPadding(nn.Module):
    """A convolution that is not transposed, called with an output padding
    (torch ignores it there)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(6, 4, 3, 3))

    def forward(self, x):
        return torch.ops.aten.convolution(x, self.w, None, [1, 1], [1, 1],
                                          [1, 1], False, [1, 0], 1)


def transposed_refusals():
    """What stays refused, each with its reason: a grouped transposed conv,
    a batch past 1 (the gather indexes one image), an output padding on a
    convolution that is not transposed, and a transposed extent that is not
    the computed one plus an output padding below max(stride, dilation) --
    the window the runtime's conv_axis_matches accepts at load, so a record
    it would refuse stays portable instead."""
    from executorch.backends.esp32s3.operator_support.conv import (
        ConvRefused,
        admit,
    )

    def refusal(node):
        try:
            admit(node)
        except ConvRefused as why:
            return str(why)
        return None

    print("transposed convolutions that stay refused, each with its reason:")
    up = nn.ConvTranspose2d(3, 5, 3, 2, 1, output_padding=1).eval()
    x = torch.randn(1, 3, 6, 6)
    unpadded = node_of(up, x)
    args = list(unpadded.args)
    unpadded.args = tuple(args[:7] + [[0, 0]] + args[8:])
    too_far = node_of(up, x)
    args = list(too_far.args)
    too_far.args = tuple(args[:7] + [[2, 2]] + args[8:])
    too_far.meta["val"] = torch.empty(1, 5, 13, 13)
    for name, node, want in (
            ("grouped transposed",
             node_of(nn.ConvTranspose2d(4, 6, 3, groups=2),
                     torch.randn(1, 4, 5, 5)),
             "grouped transposed convolution is not implemented"),
            ("batch 2, transposed with output padding",
             node_of(nn.ConvTranspose2d(4, 6, 3, 2, 1, output_padding=1),
                     torch.randn(2, 4, 5, 5)), "batch 2:"),
            ("output padding on a dense convolution",
             node_of(DenseOutputPadding(), torch.randn(1, 4, 5, 5)),
             "only meaningful for transposed convolution"),
            ("an extent that is not the computed one plus its output padding",
             unpadded, "transposed height extent 12 is not 11 plus"),
            ("an output padding of max(stride, dilation)", too_far,
             "transposed height extent 13 is not 11 plus"),
    ):
        why = refusal(node)
        check(f"{name}: refused", why is not None and want in why,
              why or "admitted")


def main():
    from executorch.backends.esp32s3.operator_support.conv import (
        ConvGeom,
        ConvRefused,
        RUNTIME_ROW_BLOCK_MAX,
        RUNTIME_ROW_BLOCK_SCRATCH_BYTES,
        admit,
        block_rows,
        runtime_block,
        split_groups,
        flash_traffic,
    )
    from executorch.backends.esp32s3.test.reference import (
        conv_iq_absmax,
        im2col_reference,
    )

    print("the identity conv == [C_out,K] @ im2col, through our gather:")
    torch.manual_seed(0)
    cases = [
        ("3x3 s1 p1", nn.Conv2d(4, 8, 3, 1, 1), (1, 4, 9, 9)),
        ("3x3 s2 p1", nn.Conv2d(4, 8, 3, 2, 1), (1, 4, 9, 9)),
        ("1x1 s1 p0", nn.Conv2d(6, 3, 1), (1, 6, 5, 5)),
        ("5x5 s1 p2", nn.Conv2d(2, 4, 5, 1, 2), (1, 2, 8, 8)),
        ("3x3 dil 2", nn.Conv2d(3, 5, 3, 1, 2, dilation=2), (1, 3, 9, 9)),
        ("no bias", nn.Conv2d(4, 4, 3, 1, 1, bias=False), (1, 4, 7, 7)),
        # conv1d, handled as a conv2d with a degenerate height axis, so these
        # go through the SAME gather rather than a second implementation.
        ("conv1d k3 s1 p1", nn.Conv1d(4, 8, 3, 1, 1), (1, 4, 12)),
        ("conv1d k5 s2 p2", nn.Conv1d(3, 6, 5, 2, 2), (1, 3, 20)),
        ("conv1d dil 2", nn.Conv1d(2, 4, 3, 1, 2, dilation=2), (1, 2, 15)),
        # conv3d, through the SAME gather again -- the tubelet embed shape a
        # video encoder actually uses, plus an overlapping padded one.
        ("conv3d tubelet", nn.Conv3d(3, 8, (2, 4, 4), (2, 4, 4)), (1, 3, 4, 8, 8)),
        ("conv3d k3 p1", nn.Conv3d(2, 4, 3, 1, 1), (1, 2, 4, 6, 6)),
        # The 1-D shapes audio and timeseries models actually use. Dilated and
        # strided 1-D matter because a TCN/WaveNet stack is nothing else, and a
        # causal block is `F.pad` then an UNPADDED dilated conv, so the conv
        # this backend sees is the plain dilated one below.
        ("conv1d dilated 4", nn.Conv1d(4, 8, 3, dilation=4), (1, 4, 32)),
        ("conv1d stride 3", nn.Conv1d(4, 8, 3, 3), (1, 4, 21)),
    ]
    for name, mod, shape in cases:
        mod = mod.eval()
        x = torch.randn(*shape)
        g = admit(node_of(mod, x))
        cols = im2col_reference(x, g)
        w2d = mod.weight.reshape(g.n, g.k)
        y = cols @ w2d.T
        if mod.bias is not None:
            y = y + mod.bias
        with torch.no_grad():
            ref = mod(x)
        y = y.T.reshape(ref.shape)
        check(name, torch.allclose(y, ref, atol=1e-5),
              f"m={g.m} k={g.k} n={g.n}, max {(y - ref).abs().max().item():.1e}")

    print("the quantised path is the matmul reference, not new arithmetic:")
    mod = nn.Conv2d(4, 8, 3, 1, 1).eval()
    x = torch.randn(1, 4, 7, 7)
    g = admit(node_of(mod, x))
    for fmt in ("i8", "i4", "i16"):
        with torch.no_grad():
            ref = mod(x)
        # conv_iq_absmax always returns [1,N,d_out,h_out,w_out]; the caller
        # reshapes to its rank, because a size-1 spatial extent is legitimate at
        # any rank and squeezing there would be ambiguous exactly where it
        # mattered.
        yq = conv_iq_absmax(x, mod.weight, mod.bias, g, fmt=fmt).reshape(ref.shape)
        rel = ((yq - ref).norm() / ref.norm()).item()
        # Loose by design: this is a quantisation error bound, not a parity
        # check. Parity is device-vs-host on identical arithmetic (run_parity.py).
        check(f"{fmt} conv tracks fp32 conv", rel < (0.30 if fmt == "i4" else 0.05),
              f"relative error {rel:.4f}")

    print("the geometry record:")
    check("round-trips", ConvGeom.unpack(g.pack()) == g, f"{len(g.pack())} B")

    # HEADROOM, not achievement. The runtime caps its block at kRowBlock, so
    # `block_rows` describes what the policy would pick given a scratch big
    # enough -- the device runs the `bad` row below. Asserting the policy's
    # arithmetic is still worth doing (it is what a bigger scratch would buy),
    # but it is not a speedup the device sees.
    print("the block-size policy (what a bigger scratch WOULD buy):")
    big = ConvGeom(20, 20, 128, 18, 18, 128, 3, 3, 1, 1, 0, 0, 1, 1)
    b = block_rows(big.k, m=big.m)
    good, passes = flash_traffic(big, b)
    bad, _ = flash_traffic(big, 8)
    worst, _ = flash_traffic(big, 1)
    check("scratch fits the SRAM budget", b * big.k <= 128 * 1024,
          f"B={b}, scratch {b * big.k / 1024:.0f} KB")
    check("would beat the block the device actually uses", good < bad,
          f"policy {good / 1024:.0f} KB vs shipped {bad / 1024:.0f} KB "
          f"({bad / good:.0f}x headroom, NOT a realised win)")
    check("the shipped block still beats row-at-a-time", worst / bad > 5,
          f"row-at-a-time {worst / 1024:.0f} KB vs shipped {bad / 1024:.0f} KB "
          f"({worst / bad:.0f}x)")
    check("never more blocks than rows", b <= big.m)

    print("depthwise takes its own path, and it is the right one:")
    dw = nn.Conv2d(8, 8, 3, 1, 1, groups=8).eval()
    xd = torch.randn(1, 8, 7, 7)
    gd = admit(node_of(dw, xd))
    check("admitted as depthwise", gd.depthwise and gd.groups == 8,
          f"groups={gd.groups} k={gd.k} (dense would be {8 * 9})")
    # The measurement that decides the design: im2col is not merely slower for
    # depthwise, it is the wrong transform. Weights are tiny and the buffer is
    # not, so the trade it makes runs backwards.
    wt_bytes = gd.n * gd.k
    # c_in_total, not c_in: the record's c_in is PER GROUP, which for a
    # depthwise op is 1. Using it here would understate the activation by C.
    act_bytes = gd.c_in_total * gd.h_in * gd.w_in * 4
    im2col_bytes = gd.m * gd.k * gd.groups
    check("weights are trivial against the activation", act_bytes / wt_bytes > 20,
          f"1:{act_bytes / wt_bytes:.0f}")
    check("so an im2col buffer would cost more than it saves",
          im2col_bytes > wt_bytes * 4,
          f"buffer {im2col_bytes} B to save re-streaming {wt_bytes} B")
    from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax

    with torch.no_grad():
        ref = dw(xd)
    yq = depthwise_iq_absmax(xd, dw.weight, dw.bias, gd, "i8").reshape(ref.shape)
    rel = ((yq - ref).norm() / ref.norm()).item()
    check("int8 depthwise tracks fp32", rel < 0.05, f"relative error {rel:.4f}")

    # Depthwise 1-D: the COMPOSITION of the depthwise loop and the degenerate
    # height axis. Each holds on its own; this is the only place both do at once.
    dw1 = nn.Conv1d(8, 8, 5, 2, 2, groups=8).eval()
    x1 = torch.randn(1, 8, 24)
    g1 = admit(node_of(dw1, x1))
    with torch.no_grad():
        ref1 = dw1(x1)
    y1 = depthwise_iq_absmax(x1, dw1.weight, dw1.bias, g1, "i8").reshape(ref1.shape)
    check("depthwise conv1d admitted", g1.depthwise and g1.kh == 1,
          f"groups={g1.groups} kh={g1.kh} kw={g1.kw}")
    check("int8 depthwise conv1d tracks fp32",
          ((y1 - ref1).norm() / ref1.norm()).item() < 0.05,
          f"relative error {((y1 - ref1).norm() / ref1.norm()).item():.4f}")

    print("transposed convolution, through the same gather:")
    for tname, tmod, tshape in transposed_cases():
        tmod = tmod.eval()
        tx = torch.randn(*tshape)
        tg = admit(node_of(tmod, tx))
        cols = im2col_reference(tx, tg)
        # The same flip+transpose the AOT layer applies to the constant.
        taps = list(range(2, tmod.weight.dim()))
        w = tmod.weight.flip(taps).transpose(0, 1).reshape(tg.n, tg.k)
        with torch.no_grad():
            tref = tmod(tx)
        ty = (cols @ w.T + tmod.bias).T.reshape(tref.shape)
        check(tname, torch.allclose(ty, tref, atol=1e-5),
              f"transposed={tg.transposed} m={tg.m} k={tg.k} n={tg.n}, "
              f"max {(ty - tref).abs().max().item():.1e}")
        # The device's arithmetic over the same gather: absmax-quantised
        # rows against the packed codes of that flipped weight.
        yq = conv_iq_absmax(tx, tmod.weight.detach().flip(taps).transpose(
            0, 1).contiguous(), tmod.bias, tg, fmt="i8").reshape(tref.shape)
        rel = ((yq - tref).norm() / tref.norm()).item()
        check(f"{tname}: the i8 mirror tracks fp32", rel < 0.05,
              f"relative error {rel:.4f}")
    transposed_refusals()
    # The zero-stuffing is an index test, never built. Materialising it is what
    # the 4.84x below would cost on a 2x2/stride-2 up-conv.
    tg = admit(node_of(nn.ConvTranspose2d(4, 6, 2, stride=2).eval(),
                       torch.randn(1, 4, 5, 5)))
    stuffed = ((5 - 1) * 2 + 1 + 2 * (2 - 1)) ** 2 * 4
    check("zero-stuffing stays virtual", stuffed / (4 * 5 * 5) > 4.0,
          f"materialising would be {stuffed / (4 * 5 * 5):.2f}x the input")

    # Grouped: g dense convs over disjoint channel slices. The assertion that
    # matters is not that it splits -- it is that each split record still
    # answers `depthwise` with FALSE. The runtime picks its kernel from that
    # predicate, and a split record carries the ORIGINAL `groups` (it is what
    # makes c_in*groups the total channel count), so anything testing `groups >
    # 1` alone reads a grouped slice as depthwise: the runtime would run the
    # depthwise kernel on a dense slice, skip the NCHW scatter, and leave half
    # the output tensor holding the raw input.
    gg = admit(node_of(nn.Conv2d(8, 16, 3, padding=1, groups=2).eval(),
                       torch.randn(1, 8, 7, 7)))
    parts = split_groups(gg)
    check("grouped splits into g dense convs", len(parts) == 2,
          f"{len(parts)} part(s)")
    check("a grouped slice is NOT depthwise",
          not any(pt.depthwise for pt in parts),
          "a split record answering depthwise=True selects the wrong kernel")
    check("grouped slices carry distinct group_idx",
          [pt.group_idx for pt in parts] == [0, 1],
          str([pt.group_idx for pt in parts]))
    check("grouped slices are per-group with the total carried",
          all(pt.c_in == 4 and pt.c_out == 8 and pt.c_out_total == 16
              for pt in parts),
          f"{[(pt.c_in, pt.c_out, pt.c_out_total) for pt in parts]}")
    # This split has c_in=1 and c_out=groups, so the first three clauses alone
    # misclassify it. c_out_total distinguishes it from true depthwise.
    edge = admit(node_of(nn.Conv2d(2, 4, 3, padding=1, groups=2).eval(),
                         torch.randn(1, 2, 7, 7)))
    edge_parts = split_groups(edge)
    check("ambiguous grouped slices are not depthwise",
          not any(pt.depthwise for pt in edge_parts),
          str([(pt.c_in, pt.c_out, pt.c_out_total, pt.groups)
               for pt in edge_parts]))
    # The limit case still takes the depthwise path, which is the whole reason
    # the predicate needs all three clauses rather than just the first.
    dw = admit(node_of(nn.Conv2d(8, 8, 3, padding=1, groups=8).eval(),
                       torch.randn(1, 8, 7, 7)))
    check("depthwise IS depthwise", dw.depthwise, "g == C must select the "
          "direct loop; im2col is the wrong transform there, not a slower one")
    check("depthwise does not split", len(split_groups(dw)) == 1,
          f"{len(split_groups(dw))} part(s)")

    # The AOT mirrors two runtime facts as Python constants, and a mirrored
    # constant that drifts goes wrong quietly: both sides stay individually
    # reasonable and neither knows about the other. A comment saying "keep
    # these in sync" is not a mechanism. This reads the C++.
    import re

    cpp = (pathlib.Path(__file__).resolve().parents[1] / "runtime" /
           "esp32_backend.cpp").read_text()
    m_max = re.search(r"constexpr uint32_t kRowBlockMax = (\d+);", cpp)
    m_bytes = re.search(
        r"constexpr uint32_t kRowBlockScratchBytes = (\d+) \* 1024;", cpp)
    check("kRowBlockMax and kRowBlockScratchBytes found in the runtime",
          m_max is not None and m_bytes is not None,
          "the guard below cannot work if a constant moved or was renamed")
    if m_max and m_bytes:
        check("RUNTIME_ROW_BLOCK_MAX matches the runtime",
              int(m_max.group(1)) == RUNTIME_ROW_BLOCK_MAX,
              f"C++ says {m_max.group(1)}, operator_support/conv.py says "
              f"{RUNTIME_ROW_BLOCK_MAX}")
        check("RUNTIME_ROW_BLOCK_SCRATCH_BYTES matches the runtime",
              int(m_bytes.group(1)) * 1024 == RUNTIME_ROW_BLOCK_SCRATCH_BYTES,
              f"C++ says {m_bytes.group(1)} KiB, operator_support/conv.py says "
              f"{RUNTIME_ROW_BLOCK_SCRATCH_BYTES}")
    # The formula itself is also mirrored: `row_block_for()` in the C++ is the
    # one place the device decides a block, and this reads its body so a
    # rewritten denominator cannot drift silently.
    body = re.search(r"row_block_for\([^)]*\)\s*\{(.*?)\n\}", cpp, re.S)
    check("row_block_for keeps the conv staging term in its denominator",
          body is not None and "sizeof(float)" in body.group(1),
          "the AOT mirror charges 4*n per conv row; the runtime must too")
    # And the derived rule, which is the thing anyone actually calls.
    check("runtime_block caps at the row block for a narrow op",
          runtime_block(49, 96) == RUNTIME_ROW_BLOCK_MAX,
          f"runtime_block(49, 96) == {runtime_block(49, 96)}")
    check("runtime_block does not exceed m", runtime_block(3, 96) == 3,
          f"runtime_block(3, 96) == {runtime_block(3, 96)}")
    check("runtime_block is scratch-bound for a wide row",
          runtime_block(64, 3136) == RUNTIME_ROW_BLOCK_SCRATCH_BYTES // 3136,
          f"runtime_block(64, 3136) == {runtime_block(64, 3136)}")
    check("a conv pays for its fp32 staging row as well",
          runtime_block(64, 1152, 128, conv=True)
          == RUNTIME_ROW_BLOCK_SCRATCH_BYTES // (1152 + 4 * 128),
          f"== {runtime_block(64, 1152, 128, conv=True)}")
    check("a single row never blocks", runtime_block(1, 16) == 1)

    grouped_conv_chain()
    depthwise_device_bits()
    one_to_one_i4()

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("conv2d reduces to the existing matmul; stage 1 holds")
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
