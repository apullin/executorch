# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transposed convolutions with output padding and dilation, delegated and run
on the host build of the runtime; and the ones that stay refused, run portable.

A transposed convolution is the dense one, taps flipped, over a zero-stuffed
input the gather never builds: tap kk of output o reads input z / stride, z =
o + kk*dil - ((k-1)*dil - pad), when z is on the stride's lattice. The three
copies of that mapping (operator_support/conv.py _axis_index, which every
mirror reads; esp32_backend.cpp axis_index; esp32_asym.cpp tap_index) carry
dilation, and an output padding needs nothing from them: the record holds the
padded extent, and the runtime's conv_axis_matches accepts the computed extent
plus anything below max(stride, dilation), torch's and ExecuTorch's bound.

  - The backend's own int8, int4 and int16 codes (axis_index and the size
    check) against the device mirror, conv_iq_absmax, at the host tests' 2e-6
    (test_host_runtime.py: the host build's float contraction is not the
    chip's; QEMU's parity rows hold the chip to the mirror's bits).
  - The float gather (esp32_asym's tap_index) and XNNPACK's dynamic recipe on
    the same geometries are parametrised in test_float_formats.py and
    test_asym_act.py; the asym conformance tests hold tap_index to the mirror
    bit for bit.
  - Refused: a grouped transposed convolution and a batch past 1, in every
    mode -- the backend's own codes, Esp32Quantizer's PT2E flow, XNNPACK's
    dynamic recipe and fp16 weights. Each is left portable with its reason in
    the partitioner's report, and the program still runs, with eager's
    numbers (or the recipe's own eager model's).

ESP32_HOST_RUNNER overrides cmake-out-espressif/esp32_host_runner.
"""

import contextlib
import io
import re
import subprocess

import numpy as np
import pytest
import torch
import torch.nn as nn

from executorch.backends.esp32s3 import _passes
from executorch.backends.esp32s3.partition.espressif_partitioner import (
    Esp32Partitioner,
)
from executorch.backends.esp32s3.test.host import host_runner

F32 = np.float32

# (module factory, input shape). Strides, paddings and dilations differ per
# axis where they can, so an axis reading another's parameters shows; the
# per-axis case pads its height by 2 at stride 2, which only its dilation of
# 3 allows.
DELEGATED = {
    "3x3 s2 p1 output_padding 1": (
        lambda: nn.ConvTranspose2d(4, 6, 3, 2, 1, output_padding=1),
        (1, 4, 5, 5)),
    "3x3 dilation 2": (
        lambda: nn.ConvTranspose2d(4, 6, 3, 1, 1, dilation=2), (1, 4, 5, 5)),
    "3x3 s(2,1) p(1,2) dil(3,2) output_padding(2,1)": (
        lambda: nn.ConvTranspose2d(4, 6, 3, stride=(2, 1), padding=(1, 2),
                                   dilation=(3, 2), output_padding=(2, 1)),
        (1, 4, 5, 6)),
    "1-D k4 s2 p1 dil 2 output_padding 1": (
        lambda: nn.ConvTranspose1d(4, 6, 4, stride=2, padding=1, dilation=2,
                                   output_padding=1), (1, 4, 9)),
}

# (module factory, input shape, the reason the partitioner reports).
REFUSED = {
    "grouped": (
        lambda: nn.ConvTranspose2d(4, 6, 3, 2, 1, output_padding=1, groups=2),
        (1, 4, 5, 5), "grouped transposed convolution is not implemented"),
    "batch 2": (
        lambda: nn.ConvTranspose2d(4, 6, 3, 2, 1, output_padding=1,
                                   dilation=2), (2, 4, 5, 5), "batch 2:"),
}
MODES = ("backend codes", "Esp32Quantizer", "xnnpack-dynamic", "fp16 weights")


@pytest.fixture(scope="module")
def runner():
    torch.set_num_threads(1)
    path = host_runner()
    if not path.is_file():
        pytest.fail(f"host runner not built: {path}; follow "
                    f"test/host_runner/README.md")
    return path.resolve()


def lower(model, inputs, partitioner, dynamic=False):
    """(lowered program, backend log). A dynamic recipe's activation chains
    are absorbed by lower_esp32's own strip right before partitioning, as
    the census lowers them."""
    ep = torch.export.export(model, inputs, strict=False)
    log = io.StringIO()
    strip = contextlib.nullcontext()
    if dynamic:
        from executorch.backends.esp32s3.test.quant_census.lower import (
            strip_before_partitioning,
        )

        strip = strip_before_partitioning({})
    with contextlib.redirect_stdout(log), strip:
        lowered = _passes.lower_esp32(ep, [partitioner])
    with contextlib.redirect_stdout(log):
        _passes.apply_passes(lowered, quiet=True)
    return lowered, log.getvalue()


def conv_descriptors(log):
    """The descriptor kind of every convolution (its emit text has a conv
    note)."""
    return re.findall(r"(?:: |, )([A-Za-z][A-Za-z0-9_]*)\[[^,]* conv\d",
                      "\n".join(l for l in log.splitlines() if " B blob" in l))


def portable_convs(lowered):
    return [n.name for n in lowered.exported_program().graph.nodes
            if n.op == "call_function" and "convolution" in str(n.target)]


def run_host(runner, tmp_path, lowered, x):
    """The host runner's one float32 output, or a failure with its stderr."""
    pte = tmp_path / "m.pte"
    pte.write_bytes(lowered.to_executorch().buffer)
    np.ascontiguousarray(x.numpy()).tofile(tmp_path / "in.bin")
    out = tmp_path / "out.bin"
    out.unlink(missing_ok=True)
    done = subprocess.run([str(runner), "--pte", str(pte), "--input",
                           str(tmp_path / "in.bin"), "--output", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and out.exists(), done.stderr[-1500:]
    return np.fromfile(out, dtype=F32)


def flipped(conv):
    """The dense equivalent's [C_out, C_in, taps] weight, as the backend
    packs a transposed conv's constant: taps flipped, channels turned."""
    taps = list(range(2, conv.weight.dim()))
    return conv.weight.detach().flip(taps).transpose(0, 1).contiguous()


@pytest.mark.parametrize("fmt", ["i8", "i4", "i16"])
@pytest.mark.parametrize("case", list(DELEGATED))
def test_backend_codes_run_on_the_host(runner, tmp_path, case, fmt):
    """Delegated whole, one descriptor of the rung asked for, loaded (the
    size check accepts the padded extent) and run as the device mirror
    computes it."""
    from executorch.exir import to_edge

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax

    factory, shape = DELEGATED[case]
    torch.manual_seed(41)
    model = factory().eval()
    x = torch.randn(*shape)
    lowered, log = lower(model, (x,), Esp32Partitioner(format_policy=fmt))
    assert portable_convs(lowered) == [], log
    assert conv_descriptors(log) == [f"addmm_{fmt}"], log
    got = run_host(runner, tmp_path, lowered, x)
    node = next(n for n in to_edge(torch.export.export(model, (x,)))
                .exported_program().graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    with torch.no_grad():
        eager = model(x)
        want = conv_iq_absmax(x, flipped(model), model.bias, admit(node),
                              fmt=fmt).reshape(eager.shape)
    np.testing.assert_allclose(got.reshape(eager.shape), want.numpy(),
                               rtol=2e-6, atol=2e-6)


def _program(mode, model, x):
    """(lowered, partitioner, log, reference output) for one mode."""
    if mode == "xnnpack-dynamic":
        from executorch.backends.esp32s3.test.quant_census import recipes

        q = recipes.quantize("xnnpack-dynamic", model, (x,), [(x,)])
        part = Esp32Partitioner()
        lowered, log = lower(q.program, q.inputs, part, dynamic=True)
        with torch.no_grad():
            return lowered, part, log, q.reference(x)
    with torch.no_grad():
        eager = model(x)
    if mode == "Esp32Quantizer":
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )

        from executorch.backends.esp32s3 import Esp32Quantizer

        prepared = prepare_pt2e(torch.export.export(model, (x,)).module(),
                                Esp32Quantizer())
        prepared(x)
        model = convert_pt2e(prepared)
    part = Esp32Partitioner(**({"format_overrides": {"weight": "fp16"}}
                               if mode == "fp16 weights" else {}))
    lowered, log = lower(model, (x,), part)
    return lowered, part, log, eager


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(REFUSED))
def test_refused_in_every_mode_and_still_runs(runner, tmp_path, case, mode):
    """Left portable with its reason in the report, and run portable."""
    factory, shape, reason = REFUSED[case]
    torch.manual_seed(43)
    model = factory().eval()
    x = torch.randn(*shape)
    lowered, part, log, want = _program(mode, model, x)
    assert portable_convs(lowered), f"{case} was delegated:\n{log}"
    reasons = [r for n, r in part.reporter._rejected_nodes.items()
               if "convolution" in n.name]
    assert any(reason in r for r in reasons), (reasons, log)
    got = run_host(runner, tmp_path, lowered, x)
    want = want.detach().numpy().reshape(-1)
    np.testing.assert_allclose(got, want, rtol=1e-5,
                               atol=1e-5 * float(np.abs(want).max()))
