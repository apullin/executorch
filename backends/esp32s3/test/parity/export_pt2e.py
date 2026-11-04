# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export an Esp32Quantizer-quantized MLP, and compute its device-exact reference.

Lower the quantized graph through the partitioner, then build the expected
checksum from THE SAME constants the delegate packed -- not from a replayed
eager forward, which would be comparing the device against a different
arithmetic and calling agreement parity.

The model arrives from the standard ExecuTorch quantization flow -- annotate
with a Quantizer, prepare, calibrate, convert -- which is the flow a user of
any other backend already has in hand. The delegate consuming it verbatim is
what makes the backend reachable from that flow rather than from ours.

K = 96 deliberately: it is under 128, so per-channel and group-128 describe the
same single group per row.
That is the regime the quantizer claims to cover, so it is the regime the row
exercises.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)


def _checksum(v):
    return float(sum(float(a) * (i + 1) for i, a in enumerate(v)))


class Pt2eMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class ActShareHeads(nn.Module):
    """Three projections of one activation, with two PT2E-scale consumers.

    The concatenation keeps every projection terminal and therefore in one
    three-op delegate. The first two descriptor nodes are marked per-tensor by
    ``export_actshare_fixture``; the third retains the backend's default per-row
    scheme. That composition is the case under test, not a proposed frontend API.
    """

    def __init__(self):
        super().__init__()
        self.pt0 = nn.Linear(32, 24)
        self.pt1 = nn.Linear(32, 24)
        self.row = nn.Linear(32, 24)

    def forward(self, x):
        return torch.cat((self.pt0(x), self.pt1(x), self.row(x)), dim=-1)


def build():
    """Run the standard PT2E flow; return the converted graph module and input."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3.test.models import ramp_input

    torch.manual_seed(0)
    model = Pt2eMLP().eval()
    x = ramp_input((1, 8, 96))
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=4))
    # Calibration on the very tensor the device will run, so the activation
    # observers see the real range. The chains are dynamic: strip_dynamic_act_qdq
    # absorbs them and the delegate computes each activation's per-tensor scale
    # at execute -- which is exactly why the reference below re-derives it.
    with torch.no_grad():
        prepared(x)
    return convert_pt2e(prepared), x, model


def main():
    from executorch.exir import ExecutorchBackendConfig, to_edge

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )
    from executorch.backends.esp32s3.test.reference import addmm_codes_ref

    converted, x, model = build()

    ep = torch.export.export(converted, (x,))
    edge = to_edge(ep)
    stripped = strip_dynamic_act_qdq(edge)
    if stripped != 2:
        raise RuntimeError(f"expected 2 activation QDQ chains, stripped {stripped}")

    # Snapshot the quantised constants BEFORE to_backend consumes them: the
    # reference must be built from the tensors the blob packed, or it is
    # measuring something else.
    consts = {}
    for n in edge.exported_program().graph.nodes:
        if n.op == "call_function" and "dequantize_per_channel" in str(n.target):
            code_n, scale_n = n.args[0], n.args[1]
            codes = _lift(edge.exported_program(), code_n)
            scales = _lift(edge.exported_program(), scale_n)
            consts[len(consts)] = (codes, scales)

    edge = edge.to_backend(Esp32Partitioner())
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    dest = os.path.join(MAIN, "mlp_pt2e.pte")
    with open(dest, "wb") as f:
        f.write(et.buffer)

    # Device-exact reference. group = K (one group per row) is what per-channel
    # means here.
    x2 = x.reshape(-1, 96).numpy().astype(np.float32)
    (c1, s1), (c2, s2) = consts[0], consts[1]
    b1 = model.fc1.bias.detach().numpy().astype(np.float32)
    b2 = model.fc2.bias.detach().numpy().astype(np.float32)
    # per_tensor=True: this model's activations are quantised per TENSOR by
    # PT2E, and strip_dynamic_act_qdq records that scheme on the op. A reference
    # derived per ROW would measure a quantisation the device does not do -- the
    # two answers differ by 1.9%.
    h = addmm_codes_ref(x2, c1, s1.reshape(-1, 1), b1, c1.shape[1],
                        per_tensor=True)
    h = np.maximum(h, np.float32(0))
    y = addmm_codes_ref(h, c2, s2.reshape(-1, 1), b2, c2.shape[1],
                        per_tensor=True)
    expected = _checksum(y.reshape(-1))

    with torch.no_grad():
        intent = _checksum(model(x).reshape(-1))
    print(f"  mlp_pt2e.pte: {os.path.getsize(dest):,} B  (PT2E int4 pass-through)")
    print(f"  device-exact checksum {expected:.6f}   fp32 eager {intent:.6f}")
    return y


def export_actshare_fixture():
    """Export the mixed activation-scale sharing fixture.

    ``strip_dynamic_act_qdq`` records PT2E's per-tensor choice on each absorbed
    matmul as ``esp32_act_per_tensor``. Most PT2E graphs choose one policy for
    every sibling, so this fixture writes that post-pass metadata directly and
    composes two per-tensor heads with one ordinary per-row head. The runtime may
    share the first pair's quantised rows, but must not pull the third into that
    group or reduce the pair back to per-row scaling.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.test.reference import (
        addmm_iq_absmax,
        pt2e_tensor_scale,
    )
    from executorch.backends.esp32s3.test.models import ramp_input

    torch.manual_seed(0)
    model = ActShareHeads().eval()
    x = ramp_input((4, 32))
    edge = to_edge(torch.export.export(model, (x,)))
    addmms = [
        node for node in edge.exported_program().graph.nodes
        if node.op == "call_function" and "aten.addmm.default" in str(node.target)
    ]
    if len(addmms) != 3:
        raise RuntimeError(
            f"activation-sharing fixture expected 3 addmm nodes, found {len(addmms)}"
        )
    for node in addmms[:2]:
        node.meta["esp32_act_per_tensor"] = True

    edge = edge.to_backend(Esp32Partitioner(format_policy="i8"))
    lowered = edge.exported_program().graph_module
    delegates = sum(
        1 for node in lowered.graph.nodes
        if node.op == "call_function" and "call_delegate" in str(node.target)
    )
    if delegates != 1:
        raise RuntimeError(
            f"activation-sharing fixture must be one delegate, found {delegates}"
        )
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    dest = os.path.join(MAIN, "actshare.pte")
    with open(dest, "wb") as f:
        f.write(et.buffer)

    x_np = x.numpy().astype(np.float32)
    tensor_scale = pt2e_tensor_scale(np.abs(x_np).max())

    def qlinear(layer, fixed):
        # A fixed scale here is PT2E's per-tensor one, rounded as PT2E does.
        return addmm_iq_absmax(
            x_np,
            layer.weight.detach().numpy().astype(np.float32),
            layer.bias.detach().numpy().astype(np.float32),
            "i8",
            sx_fixed=fixed,
            ties_to_even=fixed is not None,
        )

    expected_y = np.concatenate((
        qlinear(model.pt0, tensor_scale),
        qlinear(model.pt1, tensor_scale),
        qlinear(model.row, None),
    ), axis=-1)
    # A grouped path that ignored FLAG_ACT_PER_TENSOR would use per-row
    # quantisation for all three siblings. Keep that exact wrong answer beside
    # the oracle so a future input/shape change cannot make the fixture inert.
    old_y = np.concatenate((
        qlinear(model.pt0, None),
        qlinear(model.pt1, None),
        qlinear(model.row, None),
    ), axis=-1)
    expected = _checksum(expected_y.reshape(-1))
    old = _checksum(old_y.reshape(-1))
    gap = abs(expected - old) / max(abs(expected), 1e-9)
    if gap <= 2e-4:
        raise RuntimeError(
            f"activation-sharing fixture cannot tell per-tensor from per-row: {gap:g}"
        )
    print(f"  actshare.pte: {os.path.getsize(dest):,} B  "
          f"(2 per-tensor heads + 1 per-row head, one delegate)")
    print(f"  device-exact checksum {expected:.6f}   old grouped answer {old:.6f}  "
          f"gap {gap:.2e}")
    return expected_y


class Pt2eTies(nn.Module):
    """Two PT2E-quantised projections whose activations sit exactly where the
    delegate's own activation quantiser and PT2E's part ways.

    The device feeds every fp32 input the same ramp, so fixed buffers turn it
    into the activations under test. ``tie`` sees [0.5, -0.5, 127, 0, ...]:
    its per-tensor scale is exactly 1, so its first two codes are exact halves,
    which torch.round sends to 0 and half-away-from-zero to +/-1. ``tiny`` sees
    1e-8 beside zeros: below 127 times PT2E's minimum scale, so the floored
    scale quantises it to 0 where an unfloored absmax/127 keeps it whole. The
    weights pick those elements out, the tiny one with a gain large enough
    that each discrepancy alone moves the checksum far past the tolerance.
    """

    K = 32

    def __init__(self):
        super().__init__()
        ramp = torch.arange(self.K, dtype=torch.float32) * 0.0078125 - 0.5
        target = torch.zeros(self.K)
        target[:3] = torch.tensor([0.5, -0.5, 127.0])
        # Exact in fp32: every term is a small multiple of 2^-7.
        self.register_buffer("tie_shift", target - ramp)
        gain = torch.zeros(self.K)
        gain[0] = 1e-8 / float(ramp[0])
        self.register_buffer("tiny_gain", gain)
        self.tie = nn.Linear(self.K, 3, bias=False)
        self.tiny = nn.Linear(self.K, 1, bias=False)
        with torch.no_grad():
            self.tie.weight.copy_(torch.eye(3, self.K))
            self.tiny.weight.zero_()
            self.tiny.weight[0, 0] = 1e8

    def forward(self, x):
        return torch.cat((self.tie(x + self.tie_shift),
                          self.tiny(x * self.tiny_gain)), dim=-1)


def export_ties_fixture():
    """Export the PT2E rounding/minimum-scale fixture; return its output.

    The reference is PT2E itself: the converted graph's own quantize and
    dequantize nodes run eagerly, not a mirror of the device's arithmetic.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )
    from executorch.backends.esp32s3.test.models import ramp_input

    model = Pt2eTies().eval()
    x = ramp_input((1, Pt2eTies.K))
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=4))
    with torch.no_grad():
        prepared(x)
    # Run through torch.export's module rather than `converted` itself: convert
    # leaves the observer's eps as a tensor argument that the converted
    # module's generated code cannot spell.
    exported = torch.export.export(convert_pt2e(prepared), (x,))
    with torch.no_grad():
        y = exported.module()(x).reshape(-1)
    # The fixture is only a test while PT2E still lands on those edges:
    # halves rounded to 0 and the tiny value quantised away.
    if not torch.allclose(y, torch.tensor([0.0, 0.0, 127.0, 0.0]), atol=1e-3):
        raise RuntimeError(f"pt2e_ties no longer sits on its edges: {y.tolist()}")
    expected = _checksum(y)

    edge = to_edge(exported)
    stripped = strip_dynamic_act_qdq(edge)
    if stripped != 2:
        raise RuntimeError(f"expected 2 activation QDQ chains, stripped {stripped}")
    edge = edge.to_backend(Esp32Partitioner())
    portable_mm = [
        node.name for node in edge.exported_program().graph_module.graph.nodes
        if node.op == "call_function"
        and any(op in str(node.target) for op in ("aten.mm", "aten.addmm"))]
    if portable_mm:
        raise RuntimeError(f"pt2e_ties projections stayed portable: {portable_mm}")
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    dest = os.path.join(MAIN, "pt2e_ties.pte")
    with open(dest, "wb") as f:
        f.write(et.buffer)
    print(f"  pt2e_ties.pte: {os.path.getsize(dest):,} B  (PT2E ties-to-even and "
          f"minimum scale)")
    print(f"  PT2E eager checksum {expected:.6f}")
    return y


class Pt2eCNN(nn.Module):
    """A padded 3x3 conv with a fused ReLU, a 1x1 conv, then a Linear: the
    dense-conv and pointwise-conv pass-through shapes plus the Linear the
    MLP row proves, in one graph."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pw = nn.Conv2d(8, 4, 1)
        self.fc = nn.Linear(4 * 6 * 6, 5)

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        h = self.pw(h)
        return self.fc(h.flatten(1))


def export_conv_fixture():
    """PT2E-quantised convolutions through the delegate, referenced from the
    packed codes exactly as the Linear row is.

    A conv whose weight arrives as PT2E codes is delegated, not refused to
    portable fp32 (refusal is a silent fallback, so a quantised CNN would
    delegate nothing). The codes reshape to the packer's
    [N, K] and ride the same gather the float path uses; this row is the
    device proof that the AOT emits that blob correctly (flag, geometry,
    orientation, per-tensor activation scale), not a new kernel.
    """
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.operator_support.conv import ConvGeom
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3.quantizer.espressif_quantizer import Esp32Quantizer
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        strip_dynamic_act_qdq,
    )
    from executorch.backends.esp32s3.test.reference import addmm_codes_ref, conv_codes_ref
    from executorch.backends.esp32s3.test.models import clamp_f32, ramp_input

    torch.manual_seed(0)
    model = Pt2eCNN().eval()
    x = ramp_input((1, 3, 6, 6))
    gm = torch.export.export(model, (x,)).module()
    prepared = prepare_pt2e(gm, Esp32Quantizer(weight_bits=4))
    with torch.no_grad():
        prepared(x)
    converted = convert_pt2e(prepared)

    edge = to_edge(torch.export.export(converted, (x,)))
    stripped = strip_dynamic_act_qdq(edge)
    if stripped != 3:
        raise RuntimeError(f"expected 3 activation QDQ chains, stripped {stripped}")
    consts = []
    for n in edge.exported_program().graph.nodes:
        if n.op == "call_function" and "dequantize_per_channel" in str(n.target):
            consts.append((_lift(edge.exported_program(), n.args[0]),
                           _lift(edge.exported_program(), n.args[1])))
    if len(consts) != 3:
        raise RuntimeError(f"expected 3 quantised weights, found {len(consts)}")

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        edge = edge.to_backend(Esp32Partitioner())
    log = buf.getvalue()
    for want in (" conv3x3[3->8 @6x6]", " conv1x1[8->4 @6x6]", "+relu"):
        if want not in log:
            raise RuntimeError(f"PT2E conv row did not lower as intended: "
                               f"missing {want!r} in\n{log}")
    if log.count("addmmPT_i4") != 3 or log.count("+pt") != 3:
        raise RuntimeError(f"PT2E conv row: expected three pass-through ops "
                           f"with per-tensor activations:\n{log}")
    portable = [str(n.target) for n in edge.exported_program().graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target)]
    if portable:
        raise RuntimeError(f"a convolution stayed portable: {portable}")
    et = edge.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    dest = os.path.join(MAIN, "convnet_pt2e.pte")
    with open(dest, "wb") as f:
        f.write(et.buffer)

    (c1, s1), (c2, s2), (c3, s3) = consts
    b1 = model.conv1.bias.detach().numpy().astype(np.float32)
    b2 = model.pw.bias.detach().numpy().astype(np.float32)
    b3 = model.fc.bias.detach().numpy().astype(np.float32)
    g1 = ConvGeom(h_in=6, w_in=6, c_in=3, h_out=6, w_out=6, c_out=8,
                  kh=3, kw=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1,
                  dil_h=1, dil_w=1, c_out_total=8)
    g2 = ConvGeom(h_in=6, w_in=6, c_in=8, h_out=6, w_out=6, c_out=4,
                  kh=1, kw=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                  dil_h=1, dil_w=1, c_out_total=4)
    h = conv_codes_ref(x, c1.reshape(8, -1), s1.reshape(-1, 1), b1, g1)
    # The fused "+relu": clamp_output's [0, inf) interval.
    h = torch.from_numpy(clamp_f32(h.numpy(), 0.0, np.inf)).reshape(1, 8, 6, 6)
    h = conv_codes_ref(h, c2.reshape(4, -1), s2.reshape(-1, 1), b2, g2)
    flat = h.reshape(1, -1).numpy().astype(np.float32)
    y = addmm_codes_ref(flat, c3, s3.reshape(-1, 1), b3, c3.shape[1],
                        per_tensor=True)
    expected = _checksum(y.reshape(-1))
    with torch.no_grad():
        intent = _checksum(model(x).reshape(-1))
    print(f"  convnet_pt2e.pte: {os.path.getsize(dest):,} B  "
          f"(PT2E int4 conv pass-through)")
    print(f"  device-exact checksum {expected:.6f}   fp32 eager {intent:.6f}")
    return y


def _lift(exported, node):
    """Constant tensor behind a placeholder/get_attr, as numpy."""
    from executorch.backends.esp32s3.operator_support.const_weights import _const_tensor

    t = _const_tensor(exported, node) if node.op == "placeholder" else None
    if t is None and node.op == "get_attr":
        t = getattr(exported.graph_module, node.target)
    if t is None:
        raise RuntimeError(f"could not lift constant behind {node}")
    return t.detach().numpy()


if __name__ == "__main__":
    main()
