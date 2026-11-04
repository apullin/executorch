# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate kernel registration for the ESP32 runner using ExecuTorch's OWN codegen.

It does not hand-roll unboxing shims: a table mapping C++ parameter types to EValue
accessors would reimplement something ExecuTorch already has, and every op added would
be a chance for that table to be wrong in a way only a device run would reveal. It
shells out to the real thing instead, in two steps, exactly as
`tools/cmake/Codegen.cmake` does:

  1. `codegen.tools.gen_oplist` turns the derived op list into `selected_operators.yaml`
  2. `codegen.gen` turns that plus `kernels/portable/functions.yaml` into
     `RegisterKernelsEverything.cpp`, `NativeFunctions.h` and `Functions.h`

Two choices worth stating, because both are load-bearing:

**`--manual-registration`.** Without it the generated file registers via a static
initializer inside an anonymous namespace (`static auto success_with_kernel_reg =
register_kernels(...)`). In an ESP-IDF build every component is a static archive linked
without --whole-archive, so nothing references that translation unit and the linker is
entitled to drop it -- kernels silently unregistered, and the failure appears at
Method::init as OperatorMissing rather than at link time. `--manual-registration`
instead emits `Error register_all_kernels()`, an ordinary function main.cpp calls, which
needs no link-order tricks. ET's own CMake solves this with
`executorch_target_link_options_shared_lib`; an explicit call is the better fit here.

**Generated ahead of the build, not during it.** ET runs its codegen from CMake. Doing
that here would make `idf.py build` depend on a Python environment with torch and
torchgen in it, which is a heavy requirement for building firmware and would break the
"clone and build" path entirely. Generating into the component tree keeps the firmware
build self-contained; the cost is that this script has to be re-run when the op set
changes, which is exactly what `run_parity.py` does.

What this script adds: the *source list*. ET's codegen decides which kernels to
REGISTER, but its build compiles the whole portable kernel library and lets --gc-sections
drop the rest. Compiling ~200 files for xtensa to discard most of them is not free, so
the op -> implementing-.cpp mapping below drives an explicit source list for the IDF
component.

    python examples/espressif/esp32s3/scripts/gen_kernels.py
"""

import os
import re
import subprocess
import sys

# examples/espressif/esp32s3/scripts/ -> the root of the ExecuTorch tree
ET_ROOT = os.environ.get("ET_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_DIR = os.path.join(ET_ROOT, "kernels", "portable", "cpu")
PATTERN_H = os.path.join(CPU_DIR, "pattern", "pattern.h")
FUNCTIONS_YAML = os.path.join(ET_ROOT, "kernels", "portable", "functions.yaml")
# ExecuTorch's own quantized_decomposed kernels, registered through the same
# codegen from quantized.yaml when the op list names them: the safety net for
# the static int8 pairs a static graph leaves between two portable ops
# (static_qdq.py), and dequantize_per_tensor.Tensor_out, the per-tensor
# dequantize by a scale and zero point in one-element tensors.
# They match PyTorch's quantize/dequantize for float32-valued scales and
# finite inputs: inf and NaN are out of contract (ExecuTorch converts
# zp + nearbyint(x * inv) to int64, undefined for inf; its NEON path maps NaN
# to the zero point), and an export refuses a portable quantize whose scale
# is not a float32 value (static_qdq.refuse_portable_scales). Tensor_out takes
# a float64 scale and an int64 zero point, as PyTorch's choosers return them,
# and aborts (ET_CHECK) on any other dtype, which torch accepts. Never a
# chooser: ExecuTorch's choose_qparams agrees with none of PyTorch's.
QUANT_DIR = os.path.join(ET_ROOT, "kernels", "quantized", "cpu")
QUANTIZED_YAML = os.path.join(ET_ROOT, "kernels", "quantized", "quantized.yaml")
QUANTIZED_NS = "quantized_decomposed::"

COMPONENT = os.path.normpath(os.path.join(HERE, "..", "components", "executorch"))

# Selective build: --required/--suffix let a build variant register ONLY the ops its
# program needs. All 56 zoo kernels plus the runtime are 1.77 MB, and the 16 MB layout
# gives the app 1.4 MB; the ple program's own op list is a fraction of the zoo's.
import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--required", default=os.path.join(COMPONENT, "required_ops.txt"),
                 help="the operator list (backends/esp32s3/test/parity/"
                      "dump_required_ops.py writes it)")
_ap.add_argument("--suffix", default="",
                 help="variant tag: writes generated_kernels<suffix>/ and "
                      "generated_kernel_srcs<suffix>.cmake")
_ap.add_argument(
    "--fast-layernorm", action="store_true",
    help="replace native_layer_norm registration with the opt-in target shim",
)
_ap.add_argument(
    "--no-qdq-fast", action="store_true",
    help="register ExecuTorch's own quantize_per_tensor, dequantize_per_tensor "
         "and avg_pool2d kernels even where the op library has exact fast paths "
         "for them (qdq_pool.h): the same-tree A/B build",
)
_ap.add_argument(
    "--elementwise-fast", action="store_true",
    help="register the element-wise loops (backends/esp32s3/ops/elementwise.h) "
         "for aten::mul.out and aten::constant_pad_nd.out, where the program names "
         "them: the bench and parity firmware; the story's leaves its shims as "
         "they were",
)
_args, _ = _ap.parse_known_args()  # known args only: importable from other scripts

GEN_DIR = os.path.join(COMPONENT, f"generated_kernels{_args.suffix}")
SRCS_CMAKE = os.path.join(COMPONENT, f"generated_kernel_srcs{_args.suffix}.cmake")
VARIANT_H = os.path.join(GEN_DIR, "Esp32KernelVariant.h")


def read_required(path):
    """The operators the programs need, one per line of `path` (# starts a
    comment), as dump_required_ops.py writes them. Read when the script runs,
    not at import: the module imports for its tables where no list has been
    generated yet."""
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing -- run "
            "backends/esp32s3/test/parity/dump_required_ops.py (run_parity.py does)"
        )
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")]


# Ops in our own namespace are registered by the espressif_ops.cpp library;
# the exact-cost controls stay in main.cpp's test-only registry.
# Any that appear in required_ops.txt are excluded from ET's codegen because ExecuTorch's
# functions.yaml has never heard of them. Routing them through --custom-ops-yaml-path
# instead is possible, but it also generates PyTorch registration we do not want on
# device.
# Standard-namespace overrides use the ordinary ET schema/PTE instruction but a
# target-specific unboxing shim, with the portable implementation as its
# semantic fallback for dtype/layout/broadcast cases outside the fast contract.
# The op library registers one only where the program names it (LIBRARY_OPS);
# its portable source is compiled into every build all the same, because the
# parity firmware's fast-path check calls each shim and its fallback directly,
# and the linker keeps it only where something references it.
OVERRIDES = {
    "aten::_softmax.out": "op_softmax.cpp",
    "aten::add.out": "op_add.cpp",
    "aten::bmm.out": "op_bmm.cpp",
    "aten::div.out": "op_div.cpp",
    "aten::index_put_": "op_index_put.cpp",
    "aten::le.Tensor_out": "op_le.cpp",
    "aten::mul.out": "op_mul.cpp",
    "aten::permute_copy.out": "op_permute_copy.cpp",
    "aten::remainder.Scalar_out": "op_remainder.cpp",
    "aten::split_with_sizes_copy.out": "op_split_with_sizes_copy.cpp",
    "aten::where.self_out": "op_where.cpp",
}
if _args.fast_layernorm:
    OVERRIDES["aten::native_layer_norm.out"] = "op_native_layer_norm.cpp"
# Overrides the op library takes only for a program that names the operator
# (backends/esp32s3/ops/qdq_pool.h): the operator's kernel source (its
# library and file) is then linked as the fallback, and its Esp32KernelVariant.h macro
# compiles and registers the shim. A program without the operator -- the
# story's -- links and compiles exactly what it did before they existed.
CONDITIONAL_OVERRIDES = {
    "quantized_decomposed::quantize_per_tensor.out":
        ("ESP32_GENERATED_FAST_QUANTIZE", "quantized", "op_quantize.cpp"),
    "quantized_decomposed::dequantize_per_tensor.out":
        ("ESP32_GENERATED_FAST_DEQUANTIZE", "quantized", "op_dequantize.cpp"),
    "aten::avg_pool2d.out":
        ("ESP32_GENERATED_FAST_AVG_POOL2D", "portable", "op_avg_pool2d.cpp"),
}
# The element-wise loops (backends/esp32s3/ops/elementwise.h): the op
# library's own shims for these operators take them first, in firmware built
# with --elementwise-fast whose program names the operator. Not by the
# operator alone, as CONDITIONAL_OVERRIDES: the story's programs name it, and
# its object code must not change. The portable fallbacks are OVERRIDES', so
# nothing more is linked.
ELEMENTWISE_OVERRIDES = {
    "aten::mul.out": "ESP32_GENERATED_EW_MUL",
}
# And the element-wise overrides of ExecuTorch's own registrations, taken as
# CONDITIONAL_OVERRIDES are (the portable source linked as the fallback, the
# codegen leaving the operator out), but only with --elementwise-fast.
ELEMENTWISE_CONDITIONAL = {
    "aten::constant_pad_nd.out":
        ("ESP32_GENERATED_EW_CONSTANT_PAD", "portable", "op_constant_pad_nd.cpp"),
}
# Our own namespace, plus llama::update_cache -- ExecuTorch's LLM cache write
# (extension/llm/custom_ops), which the stock quantised KV cache emits and the
# ops library implements itself (kv_affine.cpp).
OUR_NAMESPACES = ("espressif::", "llama::")
# ExecuTorch prim ops the firmware needs: it compiles no register_prim_ops.cpp,
# so the ops library registers these itself on the device (kv_affine.h).
OUR_PRIM_OPS = ("aten::_local_scalar_dense",)
# Every operator the op library registers itself (espressif_ops.cpp's
# kCustomKernels, each entry under its library_macro): registered, and so
# linked, only in a firmware one of whose programs names it. A program
# without one -- the story's without layer_norm2d, the K cache's kernels or
# aten::div -- links none of its code. Esp32KernelVariant.h defines every
# macro (0 or 1); the host's registers them all. (The opt-in fast LayerNorm
# keeps its own switch, ESP32_GENERATED_FAST_LAYERNORM.)
LIBRARY_OPS = (*(op for op in OVERRIDES if op != "aten::native_layer_norm.out"),
               *OUR_PRIM_OPS,
               "espressif::rmsnorm.out", "espressif::rmsnorm_noscale.out",
               "espressif::rmsnorm_add.out", "espressif::ple_prepare.out",
               "espressif::cached_attention.out", "espressif::attn_tail.out",
               "espressif::conv_step.out", "espressif::ssm_step.out",
               "espressif::rope.out", "espressif::layer_norm2d.out",
               "espressif::quantize_rows_i8.out", "espressif::quant_lut.out",
               "espressif::quant_lut_codes.out", "espressif::lut16.out",
               "espressif::qnorm.out", "espressif::quantize_per_token_xnn.out",
               "espressif::dequantize_per_token_i8.out",
               "llama::update_cache.out")


def library_macro(op):
    """The Esp32KernelVariant.h macro that registers the op library's kernel
    for `op`: ESP32_GENERATED_OP_, the namespace and the name, upper case,
    without the overload or the name's leading and trailing underscores
    (aten::_softmax.out: ESP32_GENERATED_OP_ATEN_SOFTMAX; espressif_ops.cpp
    spells it ESP32_OP(ATEN_SOFTMAX))."""
    namespace, _, name = op.partition("::")
    name = name.split(".")[0].strip("_")
    return f"ESP32_GENERATED_OP_{namespace.upper()}_{name.upper()}"


def select(all_ops):
    """(taken, ours, ops, quantized_ops) for the operator list all_ops: the
    conditional overrides it takes, the operators registered by hand, the
    operators ExecuTorch's codegen registers, and the quantized_decomposed ones
    among those. A portable chooser is refused."""
    taken = [op for op in CONDITIONAL_OVERRIDES
             if op in all_ops and not _args.no_qdq_fast]
    taken += [op for op in ELEMENTWISE_CONDITIONAL
              if op in all_ops and _args.elementwise_fast]
    ours = [op for op in all_ops
            if op.startswith(OUR_NAMESPACES) or op in OVERRIDES
            or op in OUR_PRIM_OPS or op in taken]
    ops = [op for op in all_ops
           if not op.startswith(OUR_NAMESPACES) and op not in OVERRIDES
           and op not in OUR_PRIM_OPS and op not in taken]
    quantized_ops = [op for op in ops if op.startswith(QUANTIZED_NS)]
    choosers = [op for op in quantized_ops if "choose_qparams" in op]
    if choosers:
        raise SystemExit(f"{_args.required} asks for portable choosers {choosers}: "
                         f"a dynamic chooser is absorbed by the delegate, never "
                         f"registered")
    return taken, ours, ops, quantized_ops


def torchgen_dir():
    out = subprocess.run(
        [sys.executable, "-c",
         "import torchgen, os; print(os.path.dirname(torchgen.__file__))"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def run_codegen(ops, quantized_ops):
    """Run ExecuTorch's two codegen steps into GEN_DIR for the operators `ops`
    (`quantized_ops` among them). Returns the registration path.

    Both run with cwd=ET_ROOT because they are invoked as `-m codegen...`, which needs
    the checkout's `codegen` package on sys.path -- the installed wheel's copy ships no
    templates/, so the tree is the only thing that can serve this.
    """
    os.makedirs(GEN_DIR, exist_ok=True)
    oplist = os.path.join(GEN_DIR, "selected_operators.yaml")
    tg = torchgen_dir()

    subprocess.run(
        [sys.executable, "-m", "codegen.tools.gen_oplist",
         f"--output_path={oplist}",
         "--root_ops=" + ",".join(ops)],
        cwd=ET_ROOT, check=True, capture_output=True, text=True)

    # The quantized ops' schemas come from their own yaml, as a custom-op
    # library: one registration file then holds both sets. Only when a program
    # needs one, so a build without one generates no quantized-op registration.
    custom = ([f"--custom-ops-yaml-path={QUANTIZED_YAML}"] if quantized_ops
              else [])
    subprocess.run(
        [sys.executable, "-m", "codegen.gen",
         f"--source-path={os.path.join(ET_ROOT, 'codegen')}",
         f"--install-dir={GEN_DIR}",
         f"--tags-path={os.path.join(tg, 'packaged/ATen/native/tags.yaml')}",
         f"--aten-yaml-path={os.path.join(tg, 'packaged/ATen/native/native_functions.yaml')}",
         f"--op-selection-yaml-path={oplist}",
         f"--functions-yaml-path={FUNCTIONS_YAML}",
         *custom,
         "--manual-registration"],
        cwd=ET_ROOT, check=True, capture_output=True, text=True)

    reg = os.path.join(GEN_DIR, "RegisterKernelsEverything.cpp")
    if not os.path.exists(reg):
        raise SystemExit(f"codegen produced no {reg}")

    # RegisterKernels_0.cpp is the sharded half of the same content; ...Everything.cpp
    # is the single-file form and is what we compile. Leaving both in an IDF component
    # would define register_all_kernels() twice. A custom-op library also gets
    # PyTorch-side registration and stubs, which the firmware never compiles.
    for name in ("RegisterKernels_0.cpp", "RegisterCPUCustomOps.cpp",
                 "RegisterSchema.cpp", "RegisterCPUStub.cpp",
                 "CustomOpsNativeFunctions.h"):
        stale = os.path.join(GEN_DIR, name)
        if os.path.exists(stale):
            os.remove(stale)
    return reg


def count_registered(path):
    """How many Kernel entries the generated file actually registers."""
    return len(re.findall(r"^\s*Kernel\(\s*$", open(path).read(), re.M)) or \
        len(re.findall(r"Kernel\(\s*\n\s*\"", open(path).read()))


# ---- the source list: which portable .cpp files implement the selected ops ----
def load_kernel_names(yaml_path=FUNCTIONS_YAML):
    """op name -> kernel leaf name, from ExecuTorch's own yaml."""
    mapping = {}
    cur = None
    for line in open(yaml_path):
        m = re.match(r"^- op:\s*(\S+)", line) or \
            re.match(r"^- func:\s*([^(\s]+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"kernel_name:\s*(\S+)", line)
        if m and cur:
            mapping[cur] = m.group(1).split("::")[-1]
            cur = None
    return mapping


def find_source(leaf, directory=CPU_DIR):
    """The op_*.cpp in `directory` that defines `leaf`, or None."""
    pat = re.compile(r"(^|\n)([A-Za-z_][\w:<>,& ]*?)\s*\b" + re.escape(leaf) + r"\s*\(")
    for fn in sorted(os.listdir(directory)):
        if not fn.startswith("op_") or not fn.endswith(".cpp"):
            continue
        text = open(os.path.join(directory, fn)).read()
        for m in pat.finditer(text):
            i, depth, end = m.end(), 1, None
            while i < len(text) and depth:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
                i += 1
            # A definition ends in ") {"; anything else is a call site or a decl.
            if end is not None and "{" in text[end + 1:end + 4]:
                return fn
    return None


# Some kernels have no literal definition to find. op_rsqrt.cpp's entire body is
#     DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(rsqrt_out, rsqrt)
# and the real implementation lives in pattern/<name>.cpp, reached through the macro's
# call to internal::<name>(). That file must be compiled or the registration links
# against nothing. (RMSNorm fusion keeps rsqrt itself off the op list; the indirection
# is general and cheap to follow.)
_MACRO_HEAD = re.compile(
    r"#define\s+(\w+)\(\s*(\w+)\s*,[^)]*\)\s*"
    r"([A-Za-z_][\w:<>,& ]*?)\s*\b\2\b\s*\(([^)]*)\)")


def load_macro_impls():
    """macro name -> implementing pattern/*.cpp, for macro-defined kernels."""
    impls = {}
    if not os.path.exists(PATTERN_H):
        return impls
    lines = open(PATTERN_H).read().split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].startswith("#define "):
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].rstrip().endswith("\\"):
            buf.append(lines[i].rstrip()[:-1])
            i += 1
        if i < len(lines):
            buf.append(lines[i])
            i += 1
        block = " ".join(buf)
        m = _MACRO_HEAD.match(block)
        if not m:
            continue
        impl = re.search(r"\binternal::(\w+)\s*\(", block)
        if impl:
            cand = os.path.join("pattern", impl.group(1) + ".cpp")
            if os.path.exists(os.path.join(CPU_DIR, cand)):
                impls[m.group(1)] = cand
    return impls


def find_macro_source(leaf, macro_impls):
    """(op_*.cpp, pattern/*.cpp) for a macro-defined kernel, or (None, None)."""
    if not macro_impls:
        return None, None
    pat = re.compile(r"^(" + "|".join(sorted(macro_impls)) + r")\(\s*"
                     + re.escape(leaf) + r"\s*,", re.M)
    for fn in sorted(os.listdir(CPU_DIR)):
        if not fn.startswith("op_") or not fn.endswith(".cpp"):
            continue
        m = pat.search(open(os.path.join(CPU_DIR, fn)).read())
        if m:
            return fn, macro_impls[m.group(1)]
    return None, None


def resolve_leaf(names, op):
    """Resolve an op to its kernel leaf, trying both out-variant spellings.

    ATen spells some out-variants as one overload token: the out variant of
    `aten::where.self` is `aten::where.self_out`, not `aten::where.self.out`.
    required_ops.txt already carries the emitted spelling, but the yaml may key it
    either way, so both are tried.
    """
    ns, sep, short = op.rpartition("::")
    cands = [short]
    if short.endswith(".out"):
        cands.append(short[:-len(".out")] + "_out")
    for cand in cands:
        leaf = names.get(ns + sep + cand) or names.get(cand)
        if leaf is not None:
            return leaf
    return None


# The program and FlatTensor schema headers the runtime reads programs with:
# generated from this tree's schemas by the flatc the ExecuTorch package ships,
# the version its FlatBuffers library is pinned to (FLATC_EXECUTABLE names
# another), with the host runner's flags.
SCHEMA_DIR = os.path.join(COMPONENT, "generated", "executorch")
FLATC_FLAGS = ("--cpp", "--cpp-std", "c++11", "--scoped-enums")
SCHEMAS = (
    ("schema", ("schema/program.fbs", "schema/scalar_type.fbs")),
    ("extension/flat_tensor/serialize",
     ("extension/flat_tensor/serialize/flat_tensor.fbs",
      "extension/flat_tensor/serialize/scalar_type.fbs")),
)


def package_flatc():
    """The flatc binary of the installed ExecuTorch package ($FLATC_EXECUTABLE
    overrides). Found without importing it: no torch needed."""
    if os.environ.get("FLATC_EXECUTABLE"):
        return os.environ["FLATC_EXECUTABLE"]
    import importlib.util

    spec = importlib.util.find_spec("executorch")
    for root in (spec.submodule_search_locations or ()) if spec else ():
        path = os.path.join(root, "data", "bin", "flatc")
        if os.path.isfile(path):
            return path
    raise SystemExit("no flatc in the ExecuTorch package: set FLATC_EXECUTABLE")


def write_schema_headers():
    """Generate the four schema headers into SCHEMA_DIR."""
    flatc = package_flatc()
    for out, schemas in SCHEMAS:
        subprocess.run([flatc, *FLATC_FLAGS, "-o", os.path.join(SCHEMA_DIR, out),
                        *(os.path.join(ET_ROOT, s) for s in schemas)],
                       check=True, capture_output=True, text=True)


def main():
    all_ops = read_required(_args.required)
    taken, ours, ops, quantized_ops = select(all_ops)
    write_schema_headers()
    print(f"schema   -> {SCHEMA_DIR}")
    reg = run_codegen(ops, quantized_ops)
    n_registered = count_registered(reg)

    names = load_kernel_names()
    quantized_names = load_kernel_names(QUANTIZED_YAML)
    macro_impls = load_macro_impls()
    # Every override's portable fallback is compiled (OVERRIDES says why); the
    # op library registers the override only for a program that names it.
    srcs, missing = list(OVERRIDES.values()), []
    quantized_srcs = []
    # A conditional override's fallback, for the programs that take it.
    for op in taken:
        _, library, source = (CONDITIONAL_OVERRIDES.get(op)
                              or ELEMENTWISE_CONDITIONAL[op])
        (quantized_srcs if library == "quantized" else srcs).append(source)
    for op in quantized_ops:
        leaf = resolve_leaf(quantized_names, op)
        src = find_source(leaf, QUANT_DIR) if leaf else None
        if src is None:
            missing.append((op, f"no quantized kernel source defines {leaf}"))
            continue
        quantized_srcs.append(src)
    for op in ops:
        if op in quantized_ops:
            continue
        leaf = resolve_leaf(names, op)
        if leaf is None:
            missing.append((op, "no kernel_name in functions.yaml"))
            continue
        src = find_source(leaf)
        extra = None
        if src is None:
            src, extra = find_macro_source(leaf, macro_impls)
        if src is None:
            missing.append((op, f"no source defines {leaf}"))
            continue
        srcs.append(src)
        if extra:
            srcs.append(extra)

    uniq = sorted(set(srcs))
    quantized_uniq = sorted(set(quantized_srcs))
    with open(SRCS_CMAKE, "w") as f:
        f.write("# GENERATED by gen_kernels.py -- do not edit.\n")
        f.write(f"# The {len(uniq) + len(quantized_uniq)} kernel sources "
                f"implementing the {len(ops)} registered ops.\n"
                if quantized_uniq else
                f"# The {len(uniq)} portable kernel sources implementing the "
                f"{len(ops)} registered ops.\n")
        f.write("set(ET_GENERATED_KERNEL_SRCS\n")
        for s in uniq:
            f.write(f"    ${{ET_ROOT}}/kernels/portable/cpu/{s}\n")
        for s in quantized_uniq:
            f.write(f"    ${{ET_ROOT}}/kernels/quantized/cpu/{s}\n")
        f.write(")\n")
    with open(VARIANT_H, "w") as f:
        f.write("// GENERATED by gen_kernels.py -- do not edit.\n")
        f.write("#pragma once\n")
        f.write("#define ESP32_GENERATED_FAST_LAYERNORM "
                f"{int(_args.fast_layernorm)}\n")
        for op, (macro, _, _) in CONDITIONAL_OVERRIDES.items():
            f.write(f"#define {macro} {int(op in taken)}\n")
        for op, macro in ELEMENTWISE_OVERRIDES.items():
            f.write(f"#define {macro} "
                    f"{int(_args.elementwise_fast and op in all_ops)}\n")
        for op, (macro, _, _) in ELEMENTWISE_CONDITIONAL.items():
            f.write(f"#define {macro} {int(op in taken)}\n")
        # One macro per operator of the op library (espressif_ops.cpp reads
        # them through ESP32_OP): 1 where the program names it.
        f.write("#define ESP32_GENERATED_OPS_SELECTED 1\n")
        for op in LIBRARY_OPS:
            f.write(f"#define {library_macro(op)} {int(op in all_ops)}\n")

    print(f"codegen  -> {GEN_DIR}")
    print(f"requested {len(ops)} ops, registered {n_registered}, "
          f"{len(uniq) + len(quantized_uniq)} kernel sources")
    for op in quantized_ops:
        print(f"  QUANT   {op}: ExecuTorch's quantized kernel (static q/dq "
              f"safety net)")
    for op in ours:
        print(f"  OURS    {op}: hand-written, registered explicitly")
    for op, why in missing:
        print(f"  MISSING {op}: {why}")

    # A silent drop between "ops the programs need" and "kernels registered" is the
    # exact failure the derived op list exists to prevent, and it surfaces on device
    # as OperatorMissing (0x14) rather than at build time. Check it here instead.
    if n_registered != len(ops):
        print(f"  ERROR: {len(ops)} ops requested but {n_registered} registered")
        return 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
