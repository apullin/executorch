# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The table op's parity rows: one program, four methods.

Each method is a static int8 graph built the way a stock quantiser builds one --
a PT2E Quantizer that gives every activation fixed input and output qparams,
prepare_pt2e, convert_pt2e, torch.export -- lowered with
lower_esp32(..., quant_luts=True): every activation becomes espressif::quant_lut
and apply_passes elides the q -> dq pair after it, so the program needs no
quantize or dequantize kernel. The input is main.cpp's ramp, j/128 - 1/2,
turned by exact ops (a slice, adds and multiplies by powers of two) into what
each row needs:

  luts_q8        every allowlisted op on 1,056 inputs holding every code of its
                 input grid, every tie halfway between two codes and values past
                 both rails, block-uniform: each aligned group of four (one NEON
                 vector) shares a code, the ties sitting with the even code they
                 round to. Eager runs every element in a full, uniform vector, so
                 the device must match eager bit for bit (eager's sqrt the
                 correctly rounded root its table holds: ByDefinition).
  luts_q8_tail   the same layout plus three more elements (1,059, 3 mod 32): those
                 three run torch's scalar tail, so eager may differ there, and only
                 at codes the export recorded as split codes.
  luts_q8_mixed  every code and tie (544 inputs), each vector's four lanes a
                 quarter of the range apart, so erf and gelu's whole-vector exp
                 fallback couples small lanes to large ones; eager may differ only
                 beside a fallback-triggering lane, at recorded split codes.
  luts_nan       raw tables (no quantize after the op) of log, sqrt and rsqrt over
                 negative codes and exp over [-128, 127]: NaN payloads, infinities
                 and denormals copied through.

Every row is held to its device mirror's bits (mirrors.quant_lut on the
tables the program holds), and its gap to eager is reported. The groups are
four elements, not the construction's 32: 32 copies of 264 codes for 25 ops
would be 3.4 MB of outputs against QEMU's 150 KB heap, and the eager reference
is computed here, on NEON, whose vectors hold four floats; every row's length is
a multiple of 32 or says why not, so no other element meets the scalar tail.

The program lives in its own data partition (`luts`, subtype 0x48): its 26 KB
of tables do not fit the parity app partition's headroom.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from executorch.backends.esp32s3._passes import fold_quant_lut_pass
from executorch.backends.esp32s3.quantizer import mirrors
from executorch.backends.esp32s3._passes.fold_quant_lut_pass import one_thread
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project embeds the programs from its main/.
MAIN = str(firmware.MAIN)
PROGRAM = "luts.pte"
F32 = np.float32


class ByDefinition(torch.fx.Interpreter):
    """Eager, but each activation fold_quant_lut_pass.EXACT names computed by
    its definition, as its table holds it: sqrt's correctly rounded root, which
    torch's kernel is on Apple silicon and not on x86 (MKL). The eager
    reference a table is held to."""

    def call_function(self, target, args, kwargs):
        exact = fold_quant_lut_pass.EXACT.get(target)
        if exact is not None:
            return exact(args[0])
        return super().call_function(target, args, kwargs)


@dataclass(frozen=True)
class LutSpec:
    """One activation and its input quantiser. s1 is a power of two and beta
    even, so the input transform x * 32 s1 + beta s1 is exact and every row's
    code layout (32 x, rounded) survives it: code = round(32 x + beta) + z1."""

    name: str
    fn: object
    s1: float
    z1: int = 0
    qmin: int = -128
    qmax: int = 127
    beta: int = 0
    quantize_output: bool = True

    @property
    def dtype(self):
        return torch.uint8 if self.qmin >= 0 else torch.int8


S = LutSpec
# Each op over a domain where it does something: (code - z1) * s1 for every
# code. log, log2, log10, sqrt and pow 1.5 on uint8 from 0; log1p from -1;
# rsqrt on the symmetric [-127, 127] range from 0; erf and gelu past their
# vector paths' fallback thresholds (|x| > 9.35 and 13.2).
SPECS = (
    S("sigmoid", torch.sigmoid, 1 / 8),
    S("tanh", torch.tanh, 1 / 16, z1=3, beta=-2),
    S("silu", F.silu, 1 / 8, z1=-1, beta=2),
    S("gelu", F.gelu, 1 / 8),
    S("gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), 1 / 8, z1=1),
    S("elu", lambda x: F.elu(x, 0.5), 1 / 16),
    S("selu", F.selu, 1 / 16),
    S("celu", lambda x: F.celu(x, 0.75), 1 / 16),
    S("leaky_relu", lambda x: F.leaky_relu(x, 0.1), 1 / 8, z1=5, beta=-4),
    S("hardswish", F.hardswish, 1 / 32),
    S("hardsigmoid", F.hardsigmoid, 1 / 32),
    S("mish", F.mish, 1 / 16),
    S("softplus", lambda x: F.softplus(x, 2.0, 10.0), 1 / 16),
    S("exp", torch.exp, 1 / 32),
    S("expm1", torch.expm1, 1 / 32),
    S("log", torch.log, 1 / 16, qmin=0, qmax=255, beta=128),
    S("log1p", torch.log1p, 1 / 64, z1=-64, beta=64),
    S("log2", torch.log2, 1 / 8, qmin=0, qmax=255, beta=128),
    S("log10", torch.log10, 1 / 4, qmin=0, qmax=255, beta=128),
    S("erf", torch.erf, 1 / 8),
    S("sqrt", torch.sqrt, 1 / 16, qmin=0, qmax=255, beta=128),
    S("rsqrt", torch.rsqrt, 1 / 32, z1=-127, qmin=-127, beta=128),
    S("reciprocal", torch.reciprocal, 1 / 16),
    S("pow3", lambda x: x ** 3, 1 / 32),
    S("pow1_5", lambda x: x ** 1.5, 1 / 16, qmin=0, qmax=255, beta=128),
)
NAN_SPECS = (
    S("log_raw", torch.log, 1 / 16, quantize_output=False),
    S("sqrt_raw", torch.sqrt, 1 / 16, quantize_output=False),
    S("rsqrt_raw", torch.rsqrt, 1 / 16, quantize_output=False),
    S("exp_raw", torch.exp, 1.0, quantize_output=False),
)

# The block layout's offsets in code units, for even and odd codes: an even
# code's group holds the tie below, the code, the tie above (both ties round
# to it) and a quarter above; an odd code's, a quarter either side and an
# eighth below. Shifted so the 264 groups cover codes -132 .. 131.
BLOCKS = 132                      # pairs of groups: 1,056 elements
OFFSETS = ((-0.5, 0.0, 0.5, 0.25), (-0.25, 0.0, 0.25, -0.125))
TAIL_GROUP = 140                  # the three tail elements copy code 8's group


def block_offsets():
    return torch.tensor(OFFSETS, dtype=torch.float32) / 32 - 3.625


class LutRows(torch.nn.Module):
    """One parity method: a layout of the ramp, then each spec's activation on
    its own exact transform of it. Returns one output per spec."""

    def __init__(self, layout, specs):
        super().__init__()
        self.layout = layout
        self.specs = specs
        self.register_buffer("offsets", block_offsets())

    def base(self, r):
        if self.layout == "mixed":
            # 32 x = j / 2 - 136, then lane i of vector g holds element
            # 136 i + g: a quarter of the range apart, small beside large.
            return (r * 2.0 - 3.25).reshape(4, -1).t().reshape(-1)
        x = r.reshape(-1, 2, 4)[:, :, :1] + self.offsets   # 32 x = group - 132 + offset
        if self.layout == "tail":
            flat = x.reshape(-1)
            x = torch.cat([flat, flat[4 * TAIL_GROUP:4 * TAIL_GROUP + 3]])
        return x

    def inputs(self, r):
        """Each spec's input: x * 32 s1 + beta s1, skipping a multiply by one
        or an add of zero, so that 32 x + beta is what its quantiser rounds."""
        x = self.base(r)
        out = []
        for spec in self.specs:
            xk = x * (32 * spec.s1) if 32 * spec.s1 != 1 else x
            out.append(xk + spec.beta * spec.s1 if spec.beta else xk)
        return out

    def forward(self, r):
        return tuple(spec.fn(xk) for spec, xk in zip(self.specs, self.inputs(r)))


METHODS = {
    # name: (layout, ramp length, specs)
    "luts_q8": ("block", 8 * BLOCKS, SPECS),
    "luts_q8_tail": ("tail", 8 * BLOCKS, SPECS),
    "luts_q8_mixed": ("mixed", 544, SPECS),
    "luts_nan": ("block", 8 * BLOCKS, NAN_SPECS),
}


def output_qparams(spec):
    """q2's (scale, zero point): the op's finite outputs over every input code,
    spread over int8. The scale is a float32 value, as an observer's is."""
    codes = torch.arange(spec.qmin, spec.qmax + 1, dtype=torch.float32)
    with torch.no_grad():
        y = spec.fn((codes - spec.z1) * spec.s1).numpy()
    y = y[np.isfinite(y)]
    lo, hi = float(y.min()), float(y.max())
    scale = float(F32((hi - lo) / 255.0)) if hi > lo else 1.0
    zero_point = int(np.clip(round(-128 - lo / scale), -128, 127))
    return scale, zero_point


def quantizer(specs):
    """A PT2E quantizer with fixed qparams on every allowlisted activation's
    input and (unless the spec says raw) output, nothing else annotated."""
    from torchao.quantization.pt2e.quantizer import (
        FixedQParamsQuantizationSpec,
        QuantizationAnnotation,
        Quantizer,
    )

    def fixed(dtype, scale, zero_point, qmin, qmax):
        return FixedQParamsQuantizationSpec(
            dtype=dtype, scale=scale, zero_point=zero_point, quant_min=qmin,
            quant_max=qmax, qscheme=torch.per_tensor_affine)

    class FixedQParamsLutQuantizer(Quantizer):
        def annotate(self, model):
            acts = [n for n in model.graph.nodes if n.op == "call_function"
                    and n.target in fold_quant_lut_pass.ALLOWLIST]
            if len(acts) != len(specs):
                raise RuntimeError(f"{len(acts)} activations for {len(specs)} specs")
            for node, spec in zip(acts, specs):
                out = None
                if spec.quantize_output:
                    out = fixed(torch.int8, *output_qparams(spec), -128, 127)
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={node.args[0]: fixed(
                        spec.dtype, spec.s1, spec.z1, spec.qmin, spec.qmax)},
                    output_qspec=out, _annotated=True)
            return model

        def validate(self, model):
            pass

    return FixedQParamsLutQuantizer()


def converted_program(layout, n, specs):
    """The method as a stock flow leaves it: the converted PT2E model, exported."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3.test.models import ramp_input

    r = ramp_input((n,))
    gm = torch.export.export(LutRows(layout, specs).eval(), (r,)).module()
    prepared = prepare_pt2e(gm, quantizer(specs))
    with torch.no_grad():
        prepared(r)
    return torch.export.export(convert_pt2e(prepared), (r,)), r


def transformed_inputs(layout, specs, r):
    """Each spec's fp32 input to its table, computed with the same exact ops."""
    with torch.no_grad():
        return [xk.reshape(-1).numpy().astype(F32)
                for xk in LutRows(layout, specs).inputs(r)]


def lowered_tables(ep):
    """(table, inv, z, qmin, qmax) of the quant_lut producing each output of the
    lowered program; raises if any output is not a table (a quantize or
    dequantize left behind would have no kernel on the device)."""
    output = next(n for n in ep.graph.nodes if n.op == "output")
    leftovers = [n.name for n in ep.graph.nodes if n.op == "call_function"
                 and "quantize" in str(n.target)]
    if leftovers:
        raise RuntimeError(f"q/dq left in the lowered program: {leftovers}")
    tables = []
    for node in output.args[0]:
        if fold_quant_lut_pass._schema(node.target) != ("espressif::quant_lut", "default"):
            raise RuntimeError(f"output {node.name} is not a quant_lut")
        table = fold_quant_lut_pass._constant(ep, node.args[1])
        tables.append((table.numpy().copy(), *node.args[2:6]))
    return tables


def coupled_lanes(name, values):
    """Positions whose NEON vector (aligned group of four) holds a lane that
    sends erf's exp(-x^2) to the whole-vector fallback (|x^2| past
    0x1.5d5e2ap+6, vec128_float_neon.h vexpq_f32_u20); gelu's erf reads
    x * M_SQRT1_2."""
    x = np.asarray(values, dtype=F32)
    if name == "gelu":
        x = (x * F32(0.7071067811865476)).astype(F32)
    hot = (x * x).astype(F32) > F32(float.fromhex("0x1.5d5e2ap+6"))
    groups = np.zeros(-(-x.size // 4) * 4, dtype=bool)
    groups[:x.size] = hot
    return np.repeat(groups.reshape(-1, 4).any(axis=1), 4)[:x.size]


COUPLED = ("erf", "gelu")   # the specs whose vector path couples lanes


def compare_eager(method, layout, specs, xs, tables, eager, builds):
    """The row's gap to eager, against its bar: exact for the block layouts;
    only tail positions at scalar split codes for the tail; only coupled lanes
    at coupled split codes for mixed. Returns per-op difference counts."""
    gaps = {}
    for spec, x, (table, inv, zp, qmin, qmax), want, build in zip(
            specs, xs, tables, eager, builds):
        if not np.array_equal(build.table.numpy().view(np.int32),
                              table.view(np.int32)):
            raise RuntimeError(f"{method} {spec.name}: the lowered table is not "
                               "the one built for this activation")
        mirror = mirrors.quant_lut(x, table, inv, zp, qmin, qmax)
        got = want.reshape(-1).numpy()
        differ = np.flatnonzero(mirror.view(np.int32) != got.view(np.int32))
        gaps[spec.name] = len(differ)
        if not len(differ):
            continue
        codes = mirrors.quant_lut_codes(x, inv, zp, qmin, qmax)
        allowed = np.zeros(x.size, dtype=bool)
        split = set()
        if layout == "tail":
            allowed[x.size - x.size % 8:] = True
            split = set(build.split.get("scalar", []))
        elif layout == "mixed" and spec.name in COUPLED:
            dq = ((codes - zp) * F32(spec.s1)).astype(F32)
            allowed = coupled_lanes(spec.name, dq)
            split = set(build.split.get("coupled", []))
        bad = [int(i) for i in differ if not (allowed[i] and codes[i] in split)]
        if bad:
            raise RuntimeError(
                f"{method} {spec.name}: eager differs from the table outside its "
                f"bar at {len(bad)} position(s), first {bad[:5]} (codes "
                f"{[int(codes[i]) for i in bad[:5]]})")
    return gaps


def build_program():
    """(the emitted ExecuTorch program, {method: mirror output -- every output
    concatenated}, {method: {op: elements differing from eager}})."""
    from executorch.exir import (EdgeCompileConfig, EdgeProgramManager,
                                 ExecutorchBackendConfig)

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    refs, gaps, programs = {}, {}, {}
    for method, (layout, n, specs) in METHODS.items():
        ep, r = converted_program(layout, n, specs)
        with one_thread(), torch.no_grad():
            eager = ByDefinition(ep.module()).run(r)
        edge = lower_esp32(ep, [Esp32Partitioner()], quant_luts=True)
        report = edge.quant_lut_report
        if len(report.tables) != len(specs) or report.refused:
            raise RuntimeError(f"{method}: {report.summary()}")
        rep = apply_passes(edge, quiet=True)
        if rep["lut_pairs"] != sum(s.quantize_output for s in specs):
            raise RuntimeError(f"{method}: elided {rep['lut_pairs']} q/dq pairs")
        program = edge.exported_program()
        tables = lowered_tables(program)
        xs = transformed_inputs(layout, specs, r)
        builds = [build for _, build in report.tables]
        gaps[method] = compare_eager(method, layout, specs, xs, tables, eager,
                                     builds)
        refs[method] = np.concatenate([
            mirrors.quant_lut(x, *table) for x, table in zip(xs, tables)])
        programs[method] = program
    combined = EdgeProgramManager(
        programs, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et = combined.to_executorch(config=ExecutorchBackendConfig(
        delegate_alignment=16, constant_tensor_alignment=16))
    return et, refs, gaps


def export(main=MAIN):
    """Write luts.pte into `main`; returns build_program()'s references and
    eager gaps."""
    et, refs, gaps = build_program()
    with open(os.path.join(main, PROGRAM), "wb") as f:
        f.write(et.buffer)
    return refs, gaps


def split_summary(gaps):
    """One line per row: the ops whose eager output differs, and how often."""
    lines = []
    for method, per_op in gaps.items():
        differ = {op: count for op, count in per_op.items() if count}
        lines.append(f"{method}: eager identical" if not differ else
                     f"{method}: eager differs at " + ", ".join(
                         f"{op} {count}" for op, count in differ.items()))
    return lines


if __name__ == "__main__":
    refs, gaps = export()
    for line in split_summary(gaps):
        print(" ", line)
    print(f"  wrote {os.path.join(MAIN, PROGRAM)}: "
          f"{os.path.getsize(os.path.join(MAIN, PROGRAM)):,} B, "
          f"{len(refs)} methods")


def finite_checksum(out):
    """main.cpp's checksum for these rows: sum(out[i] * (i + 1)) over the
    finite elements of the flattened outputs (luts_nan holds NaN and
    infinities on purpose; its output bits still cover every byte)."""
    flat = np.asarray(out, dtype=F32).reshape(-1)
    return sum(float(v) * (i + 1) for i, v in enumerate(flat) if np.isfinite(v))


def fused_multiply_adds(build, obj_name="quant_lut.cpp.obj"):
    """(madd.s/msub.s lines, instructions read) in an object file of the
    espressif_backend component of an ESP-IDF build directory (quant_lut.cpp's
    by default). The
    table ops are compiled -ffp-contract=off, so the first must be empty; the
    second proves the object was disassembled."""
    import re
    import subprocess
    from pathlib import Path

    build = Path(build)
    cache = (build / "CMakeCache.txt").read_text()
    objdump = re.search(r"^CMAKE_OBJDUMP:FILEPATH=(.*)$", cache, re.M).group(1)
    obj = next(build.glob(f"esp-idf/espressif_backend/**/{obj_name}"))
    text = subprocess.run([objdump, "-d", str(obj)], capture_output=True,
                          text=True, check=True).stdout
    lines = [line for line in text.splitlines()
             if re.match(r"^\s+[0-9a-f]+:\s", line)]
    fused = [line.strip() for line in lines
             if re.search(r"\b(madd|msub)\.s\b", line)]
    return fused, len(lines)
