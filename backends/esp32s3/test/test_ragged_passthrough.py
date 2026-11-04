# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ragged group-128 i4 weights survive torchao export and backend packing.

A PLE table with K=4*784=3,136 has 24 complete groups and one short group of
64. Ragged groups are an ordinary checkpoint format, while torchao's dequant
operator requires a rectangular backing allocation. The pass-through modules
(utils/pass_through.py) pad only that backing allocation; the backend must fold
the logical-column slice and serialize exactly the original ragged table.
"""

import contextlib
import io
import struct
import sys

import numpy as np
import torch

from executorch.backends.esp32s3.serialization.espressif_blob import (
    BLOB_HEADER_BYTES,
    FLAG_FP16_SCALES,
    OP_ADDMM_I4,
    OP_ADDMM_I8,
    OP_DESC_FIELDS,
    OP_DESC_FORMAT,
    OP_EMBED_I4,
    OP_EMBED_I8,
)
from executorch.backends.esp32s3.espressif_preprocess import Esp32Backend
from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec
from executorch.backends.esp32s3.utils.pass_through import (
    PTEmbedding,
    PTLinear,
    QTensor,
)


FAILS = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}"
          f"{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def fixture(rows=11, cols=131, group=128):
    rng = np.random.default_rng(7)
    codes = rng.integers(-7, 8, size=(rows, cols), dtype=np.int8)
    # Begin as fp16 so the embedding path is required to preserve those exact
    # bits instead of silently picking a new scale precision.
    groups = (cols + group - 1) // group
    scales = np.linspace(
        0.015, 0.035, rows * groups, dtype=np.float16).reshape(rows, groups)
    return QTensor(codes, scales, group)


def preprocess(model, args):
    from executorch.exir import to_edge

    edge = to_edge(torch.export.export(model.eval(), args))
    with contextlib.redirect_stdout(io.StringIO()):
        result = Esp32Backend.preprocess(
            edge.exported_program(),
            Esp32CompileSpec(external_weights="ragged").to_list(),
        )
    desc = dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, result.processed_bytes, BLOB_HEADER_BYTES)))
    named = result.data_store_output.external_data["ragged"]
    sizes = sorted(len(result.data_store_output.buffers[entry.buffer_index])
                   for entry in named.values())
    return desc, sizes


class DeclaredRange(torch.nn.Module):
    """A torchao pass-through Linear whose dequant declares [qmin, qmax]."""

    def __init__(self, codes, qmin, qmax):
        super().__init__()
        self.register_buffer("codes", torch.from_numpy(codes))
        self.register_buffer("scales", torch.full((codes.shape[0], 1), 0.01))
        self.register_buffer("zeros", torch.zeros(codes.shape[0], 1))
        self.qmin, self.qmax = qmin, qmax

    def weight(self):
        import torchao.quantization.quant_primitives  # noqa: F401 -- the op

        return torch.ops.torchao.dequantize_affine(
            self.codes, [1, self.codes.shape[1]], self.scales, self.zeros,
            torch.int8, self.qmin, self.qmax)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight())


class DeclaredRangeTable(DeclaredRange):
    def forward(self, idx):
        return torch.nn.functional.embedding(idx, self.weight())


class DefaultRange(DeclaredRange):
    """quant_min/quant_max left to their default, which is the dtype's range."""

    def weight(self):
        import torchao.quantization.quant_primitives  # noqa: F401 -- the op

        return torch.ops.torchao.dequantize_affine(
            self.codes, [1, self.codes.shape[1]], self.scales, self.zeros,
            torch.int8)


def emit_inline(model, args):
    """The first descriptor of an inline-weight blob, and the blob."""
    from executorch.exir import to_edge

    edge = to_edge(torch.export.export(model.eval(), args))
    with contextlib.redirect_stdout(io.StringIO()):
        blob = Esp32Backend.preprocess(
            edge.exported_program(),
            Esp32CompileSpec().to_list()).processed_bytes
    return dict(zip(OP_DESC_FIELDS, struct.unpack_from(
        OP_DESC_FORMAT, blob, BLOB_HEADER_BYTES))), blob


def emitted_codes(model, args):
    """The descriptor and the matmul codes an inline blob actually stores."""
    desc, blob = emit_inline(model, args)
    n, k, k_pad = desc["n"], desc["k"], desc["k_pad"]
    if desc["opcode"] == OP_ADDMM_I4:
        raw = np.frombuffer(blob, np.uint8, n * k_pad // 2, desc["wt_off"])
        raw = raw.reshape(n, k_pad // 2)
        codes = np.empty((n, k_pad), dtype=np.int8)
        codes[:, 0::2] = (raw & 0x0F).astype(np.int8) - 8
        codes[:, 1::2] = (raw >> 4).astype(np.int8) - 8
    else:
        codes = np.frombuffer(blob, np.int8, n * k_pad, desc["wt_off"])
        codes = codes.reshape(n, k_pad)
    return desc, codes[:, :k]


def main():
    qt = fixture()
    indices = torch.tensor([0, 10, 4, 7])
    embedding = PTEmbedding(qt).eval()
    with torch.no_grad():
        got = embedding(indices)
    want = torch.from_numpy(qt.dequant())[indices]

    print("ragged eager frontend:")
    check("zero-padded torchao backing preserves logical embedding exactly",
          torch.equal(got, want),
          f"backing={tuple(embedding.codes.shape)} logical={tuple(want.shape)}")

    print("ragged pass-through embedding:")
    desc, sizes = preprocess(embedding, (indices,))
    check("embedding remains i4 pass-through at logical K=131",
          (desc["opcode"], desc["k"], desc["k_pad"], desc["scale_group"])
          == (OP_EMBED_I4, 131, 132, 128),
          f"op={desc['opcode']} K={desc['k']}/{desc['k_pad']} "
          f"group={desc['scale_group']}")
    check("embedding stores no padded columns and retains fp16 scales",
          sizes == [11 * 2 * 2, 11 * 66]
          and bool(desc["flags"] & FLAG_FP16_SCALES), str(sizes))

    print("aligned pass-through embedding:")
    aligned = PTEmbedding(fixture(cols=128)).eval()
    desc, sizes = preprocess(aligned, (indices,))
    check("an aligned table does not strand a no-op alias/dequant chain",
          (desc["opcode"], desc["k"], desc["k_pad"], desc["scale_group"])
          == (OP_EMBED_I4, 128, 128, 0),
          f"op={desc['opcode']} K={desc['k']}/{desc['k_pad']} "
          f"group={desc['scale_group']}")
    check("aligned single-group table retains exact packed rows",
          sizes == [11 * 1 * 4, 11 * 64], str(sizes))

    print("ragged pass-through linear:")
    linear_qt = fixture(rows=7)
    linear = PTLinear(linear_qt).eval()
    value = torch.linspace(-0.5, 0.5, 2 * 131).reshape(2, 131)
    with torch.no_grad():
        got = linear(value)
    want = torch.nn.functional.linear(value, torch.from_numpy(linear_qt.dequant()))
    check("zero-padded torchao backing preserves logical linear numerically",
          torch.equal(got, want), f"max abs {(got - want).abs().max().item():.2e}")
    desc, sizes = preprocess(linear, (value,))
    check("linear transpose folds after the logical-column slice",
          (desc["opcode"], desc["m"], desc["k"], desc["n"],
           desc["k_pad"], desc["scale_group"])
          == (OP_ADDMM_I4, 2, 131, 7, 160, 128),
          f"{desc['m']}x{desc['k']}x{desc['n']} Kpad={desc['k_pad']} "
          f"group={desc['scale_group']}")
    check("linear PTD also excludes the torchao-only padded column",
          sizes == [7 * 2 * 4, 7 * 80], str(sizes))

    print("the carrier holds both declared bounds:")
    x = torch.linspace(-1.0, 1.0, 2 * 32).reshape(2, 32)
    wide = (np.arange(3 * 32) % 17 - 9).astype(np.int8).reshape(3, 32)
    desc, got = emitted_codes(DeclaredRange(wide, -128, 7), (x,))
    check("[-128, 7] codes including -9 are stored as i8, exactly",
          desc["opcode"] == OP_ADDMM_I8 and np.array_equal(got, wide),
          f"opcode {desc['opcode']}; -9 stored as {sorted(set(got[wide == -9]))}")
    nibble = (np.arange(3 * 32) % 16 - 8).astype(np.int8).reshape(3, 32)
    desc, got = emitted_codes(DeclaredRange(nibble, -8, 7), (x,))
    check("[-8, 7] is exactly a biased nibble and stays i4",
          desc["opcode"] == OP_ADDMM_I4 and np.array_equal(got, nibble),
          f"opcode {desc['opcode']}")
    # int8 tables pass through too (zero-point tables are int8 as often as
    # int4).
    desc, blob = emit_inline(DeclaredRangeTable(wide, -128, 7),
                             (torch.tensor([0, 2]),))
    stored = np.frombuffer(blob, np.int8, desc["n"] * desc["k_pad"],
                           desc["wt_off"]).reshape(desc["n"], desc["k_pad"])
    check("a [-128, 7] table is stored as i8, exactly",
          desc["opcode"] == OP_EMBED_I8
          and np.array_equal(stored[:, :desc["k"]], wide),
          f"opcode {desc['opcode']}")

    print("bounds left to torchao's default take the dtype's range:")
    for name, model in (("omitted", DefaultRange(wide, None, None)),
                        ("passed as None", DeclaredRange(wide, None, None))):
        try:
            desc, got = emitted_codes(model, (x,))
            check(f"{name}: int8 codes stored as i8, exactly",
                  desc["opcode"] == OP_ADDMM_I8 and np.array_equal(got, wide),
                  f"opcode {desc['opcode']}")
        except Exception as error:  # noqa: BLE001 -- any failure is the bug
            check(f"{name}: int8 codes stored as i8, exactly", False,
                  f"{type(error).__name__}: {error}"[:90])

    # torchao's fp8 dequant shares the integer op's name as a PREFIX and has a
    # three-argument schema. It must not be read as the integer op. (Default
    # IR checks reject the fp8 buffer before partitioning; this is the path
    # with them off.)
    print("the fp8 dequant is not mistaken for the integer one:")
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    class Fp8Dequant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w8", torch.randn(8, 32).to(torch.float8_e4m3fn))
            self.register_buffer("scale", torch.tensor(0.5))

        def forward(self, value):
            import torchao.quantization.quant_primitives  # noqa: F401

            return torch.nn.functional.linear(
                value, torch.ops.torchao.dequantize_affine_float8_non_decomposed(
                    self.w8, self.scale, torch.float32))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            to_edge_transform_and_lower(
                torch.export.export(Fp8Dequant().eval(), (x,)),
                partitioner=[Esp32Partitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False))
        check("an fp8 dequant weight is refused cleanly, not mis-indexed", True)
    except Exception as error:  # noqa: BLE001 -- any failure is the bug
        check("an fp8 dequant weight is refused cleanly, not mis-indexed", False,
              f"{type(error).__name__}: {error}"[:90])

    print()
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("ragged group-128 pass-through is exact and compact")
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
