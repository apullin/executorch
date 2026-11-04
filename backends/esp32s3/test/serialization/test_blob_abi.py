# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Python blob emitter and C++ runtime must describe one exact ABI.

This is intentionally a source-contract test.  Exporting another friendly model
cannot prove field order, refused extension IDs, scratch sentinels, or
constants that the model never exercises.  Read both declarations and compare
every serialized piece directly instead.
"""

import ast
import dataclasses
import pathlib
import re
import struct

from executorch.backends.esp32s3.serialization import espressif_blob
from executorch.backends.esp32s3 import compile_spec
import executorch.backends.esp32s3.serialization.espressif_blob as aot
from executorch.backends.esp32s3.operator_support.conv import (
    CONV_GEOM_BYTES,
    CONV_GEOM_FORMAT,
    EXT_CONV3D,
    ConvGeom,
)


ROOT = pathlib.Path(__file__).resolve().parent
MAIN = ROOT.parents[1] / "runtime"
CPP_PATH = MAIN / "esp32_backend.cpp"


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# The delegate's declarations: esp32_backend.cpp, plus esp32_asym.h, which it
# includes for the asymmetric-quantisation wire values (the flags, record types
# and opcode it names in kKnownFlags, is_known_extension and kNumOpcodes), and
# esp32_float.h, for the fp16/bf16 rungs' opcodes.
CPP_RAW = CPP_PATH.read_text()
CPP = _strip_comments(CPP_RAW) + "\n" + _strip_comments(
    (MAIN / "esp32_asym.h").read_text()) + "\n" + _strip_comments(
    (MAIN / "esp32_float.h").read_text())
# esp32_asym.cpp keeps copies (its namespace v23) of the wire values it reads
# (esp32_backend.cpp declares them in its anonymous namespace); held to the
# originals below.
ASYM_CPP = _strip_comments((MAIN / "esp32_asym.cpp").read_text())
FAILS = []
_CPP_CONSTANTS = {}


def check(name, condition, detail=""):
    tag = "ok " if condition else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def _cpp_expr(name, text=None):
    match = re.search(
        rf"\bconstexpr\s+uint(?:16|32|64)_t\s+{re.escape(name)}\s*=\s*(.*?);",
        CPP if text is None else text,
        flags=re.S,
    )
    if match is None:
        raise AssertionError(f"C++ constant {name} not found")
    return re.sub(
        r"\b(0[xX][0-9a-fA-F]+|[0-9]+)(?:[uUlL]+)\b", r"\1",
        match.group(1),
    ).strip()


def cpp_constant(name, text=None):
    cache = _CPP_CONSTANTS.setdefault(id(text), {})
    if name in cache:
        return cache[name]

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return cpp_constant(node.id, text)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.BitOr: lambda: left | right,
                ast.BitAnd: lambda: left & right,
                ast.LShift: lambda: left << right,
                ast.RShift: lambda: left >> right,
            }
            operation = operations.get(type(node.op))
            if operation is not None:
                return operation()
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.Invert):
                return ~value
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        raise AssertionError(f"unsupported C++ integer expression for {name}: "
                             f"{ast.dump(node)}")

    # Parentheses make C++ line wrapping after a binary operator legal Python
    # syntax too (kKnownFlags is intentionally formatted one flag per line).
    value = evaluate(ast.parse("(" + _cpp_expr(name, text) + ")",
                               mode="eval").body)
    cache[name] = value
    return value


def cpp_string(name):
    match = re.search(
        rf"\bconstexpr\s+const\s+char\s*\*\s*{re.escape(name)}\s*=\s*"
        r'"([^"\\]*)"\s*;',
        CPP,
    )
    if match is None:
        raise AssertionError(f"C++ string constant {name} not found")
    return match.group(1)


def cpp_struct_fields(name):
    match = re.search(rf"\bstruct\s+{re.escape(name)}\s*\{{(.*?)\}}\s*;",
                      CPP, flags=re.S)
    if match is None:
        raise AssertionError(f"C++ struct {name} not found")
    fields = []
    for type_name, declarations in re.findall(
            r"\b(u?int32_t|float)\s+([^;{{}}]+);", match.group(1)):
        for declaration in declarations.split(","):
            field_name = declaration.strip().split("=")[0].strip()
            fields.append((type_name, field_name))
    return tuple(fields)


def main():
    print("blob header and descriptor:")
    check("magic", cpp_constant("kMagic") == espressif_blob.MAGIC,
          f"AOT=0x{espressif_blob.MAGIC:08x} runtime=0x{cpp_constant('kMagic'):08x}")
    check("version", cpp_constant("kVersion") == espressif_blob.VERSION,
          f"AOT={espressif_blob.VERSION} runtime={cpp_constant('kVersion')}")
    check("header bytes", cpp_constant("kHeaderBytes") == espressif_blob.BLOB_HEADER_BYTES,
          f"{espressif_blob.BLOB_HEADER_BYTES} B")
    check("descriptor bytes", cpp_constant("kOpDescBytes") == espressif_blob.OP_DESC_BYTES,
          f"{espressif_blob.OP_DESC_BYTES} B")
    check("extension-count bytes",
          cpp_constant("kExtCountBytes") == espressif_blob.EXT_COUNT_BYTES,
          f"{espressif_blob.EXT_COUNT_BYTES} B")
    check("extension-record header bytes",
          cpp_constant("kExtRecordHeaderBytes") == espressif_blob.EXT_RECORD_HEADER_BYTES,
          f"{espressif_blob.EXT_RECORD_HEADER_BYTES} B")

    python_types = tuple({"I": "uint32_t", "f": "float"}[code]
                         for code in espressif_blob.OP_DESC_FORMAT[1:])
    expected_desc = tuple(zip(python_types, espressif_blob.OP_DESC_FIELDS))
    runtime_desc = cpp_struct_fields("OpDesc")
    check("OpDesc field types and order", runtime_desc == expected_desc,
          f"{len(runtime_desc)} fields")
    check("C++ retains the compile-time size assertion",
          re.search(r"static_assert\s*\(\s*sizeof\s*\(\s*OpDesc\s*\)\s*==\s*"
                    r"kOpDescBytes\b", CPP) is not None)

    sentinel = []
    for index, type_name in enumerate(python_types):
        sentinel.append(index + 1 if type_name == "uint32_t" else index + 0.25)
    packed = struct.pack(espressif_blob.OP_DESC_FORMAT, *sentinel)
    check("Python descriptor round-trip",
          struct.unpack(espressif_blob.OP_DESC_FORMAT, packed) == tuple(sentinel),
          f"{len(packed)} B")

    print("opcodes and flags:")
    opcodes = (
        ("OP_ADDMM", "kOpAddmm"),
        ("OP_ADDMM_I8", "kOpAddmmI8"),
        ("OP_ADDMM_I4", "kOpAddmmI4"),
        ("OP_ADDMM_I16", "kOpAddmmI16"),
        ("OP_EMBED_I8", "kOpEmbedI8"),
        ("OP_EMBED_I4", "kOpEmbedI4"),
        ("OP_EMBED_I16", "kOpEmbedI16"),
        ("OP_BMM_DYN_I8", "kOpBmmDynI8"),
        ("OP_BMM_KCACHE_I8", "kOpBmmKcacheI8"),
        ("OP_BMM_KCACHE_AFFINE_I8", "kOpBmmKcacheAffineI8"),
        ("OP_ADDMM_F16", "kOpAddmmF16"),
        ("OP_ADDMM_BF16", "kOpAddmmBf16"),
        ("OP_EMBED_F16", "kOpEmbedF16"),
        ("OP_EMBED_BF16", "kOpEmbedBf16"),
    )
    opcode_values = []
    for py_name, cpp_name in opcodes:
        py_value, cpp_value = getattr(aot, py_name), cpp_constant(cpp_name)
        opcode_values.append(py_value)
        check(py_name, py_value == cpp_value,
              f"AOT={py_value} runtime={cpp_value}")
    check("opcodes are unique", len(set(opcode_values)) == len(opcode_values))
    check("runtime opcode table covers every assigned opcode",
          cpp_constant("kNumOpcodes") == max(opcode_values) + 1,
          f"table={cpp_constant('kNumOpcodes')}")

    flags = (
        ("FLAG_EXTERNAL_WEIGHTS", "kFlagExternalWeights"),
        ("FLAG_FP16_SCALES", "kFlagFp16Scales"),
        ("FLAG_CLAMP", "kFlagClamp"),
        ("FLAG_ACT_SILU", "kFlagActSilu"),
        ("FLAG_ACT_GELU", "kFlagActGelu"),
        ("FLAG_ADD", "kFlagAdd"),
        ("FLAG_STATIC_ACT", "kFlagStaticAct"),
        ("FLAG_CONV", "kFlagConv"),
        ("FLAG_ACT_PER_TENSOR", "kFlagActPerTensor"),
        ("FLAG_SOFTCAP", "kFlagSoftcap"),
        ("FLAG_STAGE_I8", "kFlagStageI8"),
        ("FLAG_HEAD_SPLIT", "kFlagHeadSplit"),
        ("FLAG_MUL", "kFlagMul"),
        ("FLAG_ACT_SIGMOID", "kFlagActSigmoid"),
        ("FLAG_ACT_GELU_TANH", "kFlagActGeluTanh"),
        ("FLAG_ACT_HARDSWISH", "kFlagActHardswish"),
        ("FLAG_ACT_HARDSIGMOID", "kFlagActHardsigmoid"),
        ("FLAG_ACT_LEAKY", "kFlagActLeaky"),
        ("FLAG_ACT_RELU2", "kFlagActRelu2"),
        ("FLAG_ACT_QUANT", "kFlagActQuant"),
        ("FLAG_WEIGHT_AFFINE", "kFlagWeightAffine"),
        ("FLAG_OUT_QUANT", "kFlagOutQuant"),
    )
    known_flags = 0
    for py_name, cpp_name in flags:
        py_value, cpp_value = getattr(aot, py_name), cpp_constant(cpp_name)
        known_flags |= py_value
        check(py_name, py_value == cpp_value,
              f"AOT={py_value} runtime={cpp_value}")
    check("flags are independent bits",
          all(value & (value - 1) == 0
              for value in (getattr(aot, name) for name, _ in flags)))
    check("runtime known-flag mask is complete",
          cpp_constant("kKnownFlags") == known_flags,
          f"0x{known_flags:x}")

    print("scratch, extensions, and compile specs:")
    check("scratch-slot count", cpp_constant("kNumScratch") == espressif_blob.NUM_SCRATCH,
          f"{espressif_blob.NUM_SCRATCH}")
    check("scratch sentinel base",
          cpp_constant("kArgScratch0") == espressif_blob.ARG_SCRATCH0,
          f"0x{espressif_blob.ARG_SCRATCH0:08x}")
    check("vector bytes", cpp_constant("kVecBytes") == espressif_blob.VEC_BYTES,
          f"{espressif_blob.VEC_BYTES} B")
    check("exact int8 dot length (the gate's MAX_DOT_LEN)",
          cpp_constant("kMaxDotLen") == espressif_blob.MAX_DOT_LEN,
          f"{espressif_blob.MAX_DOT_LEN} terms")
    check("min-latency compile-spec key",
          cpp_string("kSpecMinLatency") == compile_spec.SPEC_MIN_LATENCY,
          repr(compile_spec.SPEC_MIN_LATENCY))
    check("extension optional bit",
          cpp_constant("kExtOptional") == espressif_blob.EXT_OPTIONAL,
          f"0x{espressif_blob.EXT_OPTIONAL:04x}")
    check("conv extension type", cpp_constant("kExtConv3d") == EXT_CONV3D,
          f"type {EXT_CONV3D}")
    check("selected-row extension type",
          cpp_constant("kExtSelectedRows") == espressif_blob.EXT_SELECTED_ROWS,
          f"type {espressif_blob.EXT_SELECTED_ROWS}")
    for py_name, cpp_name in (("EXT_ACT_QUANT", "kExtActQuant"),
                              ("EXT_WEIGHT_AFFINE", "kExtWeightAffine"),
                              ("EXT_OUT_QUANT", "kExtOutQuant")):
        check(f"{py_name.lower().replace('_', '-')} extension type",
              cpp_constant(cpp_name) == getattr(aot, py_name),
              f"type {getattr(aot, py_name)}")
    check("highest assigned extension type",
          cpp_constant("kExtMaxKnown") == espressif_blob.EXT_OUT_QUANT ==
          max(EXT_CONV3D, espressif_blob.EXT_SELECTED_ROWS, espressif_blob.EXT_ACT_QUANT,
              espressif_blob.EXT_WEIGHT_AFFINE, espressif_blob.EXT_OUT_QUANT))
    known_body = re.search(
        r"is_known_extension\s*\([^)]*\)\s*\{(.*?)\}", CPP, flags=re.S)
    check("retired extension IDs are not treated as a dense known range",
          known_body is not None and
          re.fullmatch(
              r"\s*return\s+type\s*==\s*kExtConv3d\s*\|\|\s*"
              r"type\s*==\s*kExtSelectedRows\s*\|\|\s*"
              r"type\s*==\s*kExtActQuant\s*\|\|\s*"
              r"type\s*==\s*kExtWeightAffine\s*\|\|\s*"
              r"type\s*==\s*kExtOutQuant\s*;\s*",
                       known_body.group(1)) is not None)

    runtime_selected = cpp_struct_fields("SelectedRows")
    check("SelectedRows field order",
          runtime_selected == (("uint32_t", "rows_arg"),
                               ("uint32_t", "count")))
    check("SelectedRows payload bytes",
          cpp_constant("kSelectedRowsBytes") == espressif_blob.SELECTED_ROWS_BYTES ==
          struct.calcsize(espressif_blob.SELECTED_ROWS_FORMAT),
          f"{espressif_blob.SELECTED_ROWS_BYTES} B")

    runtime_conv = cpp_struct_fields("ConvGeom")
    python_conv_names = tuple(field.name for field in dataclasses.fields(ConvGeom))
    check("ConvGeom field order",
          tuple(name for _, name in runtime_conv) == python_conv_names,
          f"{len(runtime_conv)} fields")
    check("ConvGeom runtime fields are widened uint32",
          all(type_name == "uint32_t" for type_name, _ in runtime_conv))
    check("ConvGeom payload bytes",
          cpp_constant("kConvGeomBytes") == CONV_GEOM_BYTES ==
          struct.calcsize(CONV_GEOM_FORMAT), f"{CONV_GEOM_BYTES} B")
    geometry = ConvGeom(*range(1, 25))
    check("ConvGeom wire round-trip", ConvGeom.unpack(geometry.pack()) == geometry)

    print("v24 records (esp32_asym.h):")
    py_types = {"I": "uint32_t", "i": "int32_t", "f": "float"}
    for struct_name, fmt, names, bytes_name, py_bytes in (
            ("ActQuant", espressif_blob.ACT_QUANT_FORMAT, espressif_blob.ACT_QUANT_FIELDS,
             "kActQuantBytes", espressif_blob.ACT_QUANT_BYTES),
            ("WeightAffine", espressif_blob.WEIGHT_AFFINE_FORMAT,
             espressif_blob.WEIGHT_AFFINE_FIELDS, "kWeightAffineBytes",
             espressif_blob.WEIGHT_AFFINE_BYTES),
            ("OutQuant", espressif_blob.OUT_QUANT_FORMAT, espressif_blob.OUT_QUANT_FIELDS,
             "kOutQuantBytes", espressif_blob.OUT_QUANT_BYTES)):
        expected = tuple(zip((py_types[c] for c in fmt[1:]), names))
        runtime = cpp_struct_fields(struct_name)
        check(f"{struct_name} field types and order", runtime == expected,
              f"{len(runtime)} fields")
        check(f"{struct_name} payload bytes",
              cpp_constant(bytes_name) == py_bytes == struct.calcsize(fmt),
              f"{py_bytes} B")
        check(f"C++ retains {struct_name}'s size assertion",
              re.search(rf"static_assert\s*\(\s*sizeof\s*\(\s*{struct_name}"
                        rf"\s*\)\s*==\s*{bytes_name}\b", CPP) is not None)
    values = (
        ("DTYPE_FP32", "kDtypeFp32"), ("DTYPE_INT8", "kDtypeInt8"),
        ("DTYPE_INT16", "kDtypeInt16"),
        ("GRANULARITY_ROW", "kGranularityRow"),
        ("GRANULARITY_TENSOR", "kGranularityTensor"),
        ("GRANULARITY_STATIC", "kGranularityStatic"),
        ("CHOOSER_NONE", "kChooserNone"), ("CHOOSER_SYM_DIV", "kChooserSymDiv"),
        ("CHOOSER_AFFINE_MINMAX", "kChooserAffineMinMax"),
        ("ROUND_RNE_BEFORE_ZP", "kRoundRneBeforeZp"),
        ("ZP_NONE", "kZpNone"), ("ZP_INT8", "kZpInt8"), ("ZP_FP32", "kZpFp32"),
        ("OFFSET_NONE", "kOffsetNone"), ("OFFSET_FP32", "kOffsetFp32"),
        ("OFFSET_FP16", "kOffsetFp16"),
        ("SUM_NONE", "kSumNone"), ("SUM_INT16", "kSumInt16"),
        ("SUM_INT32", "kSumInt32"),
    )
    mismatched = [py for py, cpp in values
                  if getattr(aot, py) != cpp_constant(cpp)]
    check("record field values", not mismatched,
          ", ".join(mismatched) or f"{len(values)} values")
    check("v24 flags are the known-flag mask's new bits",
          cpp_constant("kV24Flags") ==
          espressif_blob.FLAG_ACT_QUANT | espressif_blob.FLAG_WEIGHT_AFFINE | espressif_blob.FLAG_OUT_QUANT)

    # esp32_asym.cpp reads descriptors through its own view of OpDesc, and
    # through its namespace v23 copies of the wire values; both must be
    # esp32_backend.cpp's.
    check("OpDescView field types and order",
          cpp_struct_fields("OpDescView") == expected_desc)
    copies = ("kHeaderBytes", "kOpDescBytes", "kExtCountBytes",
              "kExtRecordHeaderBytes", "kExtOptional", "kExtSelectedRows",
              "kOpAddmm",
              "kOpAddmmI8", "kOpAddmmI4", "kOpAddmmI16", "kOpEmbedI8",
              "kOpEmbedI4", "kOpEmbedI16", "kOpBmmDynI8", "kOpBmmKcacheI8",
              "kFlagExternalWeights", "kFlagAdd", "kFlagStaticAct",
              "kFlagConv", "kFlagActPerTensor", "kFlagStageI8", "kFlagMul",
              "kNumScratch", "kArgScratch0")
    stale = [name for name in copies
             if cpp_constant(name, ASYM_CPP) != cpp_constant(name)]
    check("esp32_asym.cpp's copies of the v23 values", not stale,
          ", ".join(stale) or f"{len(copies)} values")

    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        return 1
    print("OK: Python emitter and C++ runtime blob contracts match.")
    return 0


def test_main():
    """Under pytest, what a script run checks: main() records every failed
    check and goes on; this fails if any did, naming them."""
    start = len(FAILS)
    rc = main()
    failed = FAILS[start:]
    assert not failed and rc in (None, 0), f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    raise SystemExit(main())
