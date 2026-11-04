# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One command: export, build, run in QEMU, compare against eager PyTorch.

    python backends/esp32s3/test/parity/run_parity.py

Exports every smoke-test model with a fixed seed, records the eager checksum for the
exact input the device writes, rebuilds the firmware, runs it under QEMU, parses the
device's checksums back out and diffs them. Exit code 0 only if every model matches.

One command, so a stale .pte cannot pass for a new one: an older model left embedded
while the host prints new numbers reads exactly like a parity failure.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import torch

from executorch.backends.esp32s3._passes import apply_passes, fewer_gc_passes
from executorch.backends.esp32s3.test.models import (
    SPECS,
    block_i8_out,
    checksum,
    output_bits,
    ramp_input,
)
from executorch.backends.esp32s3.test import firmware

HERE = os.path.dirname(os.path.abspath(__file__))
# The example firmware project: the exporters write its main/, the selective
# build its executorch component, and idf.py builds it.
PROJ = str(firmware.PROJECT)
MAIN = str(firmware.MAIN)
# Without an activated ESP-IDF, a script that sources $IDF_PATH/export.sh and
# exec's its arguments (see idf()).
IDFENV = os.environ.get("IDF_ENV_SCRIPT")
TOL = 2e-4          # relative, for references that are not the device's arithmetic
# The full nano-llama stock recipe (fixtures/stock_llm.py) is held to eager by its
# greedy tokens: the fraction of steps whose top-1 must agree.
STOCK_TOP1_FLOOR = 0.9
CHECKPOINT = os.path.join(PROJ, "build", "parity_export_checkpoint.json")

# A row's reference is either a device mirror -- the device's arithmetic
# reproduced bit for bit, so the row compares the mirror's FNV-1a with the
# `output bits` main.cpp prints -- or is held to TOL for one of these reasons.
EAGER = ("eager PyTorch reference: the device runs ExecuTorch's portable "
         "kernels, with their own reduction order, newlib's libm and "
         "GCC-fused multiply-adds")
BLOCK = ("the Linears are device mirrors, but layer_norm, softmax (newlib "
         "expf), silu and both bmms run portable kernels on the device and "
         "torch eager in the reference")
EPIGLU = ("every delegate op is a device mirror, but the portable tanh between "
          "the two partitions is newlib's tanhf on the device and the host "
          "libm's in the reference")
PT2E_EAGER = ("an independent oracle: PT2E's own quantize/dequantize evaluated "
              "eagerly, whose float matmul rounds differently from the "
              "delegate's fused epilogue")
CACHED = ("the decode references mirror the Linears exactly, but RMSNorm, "
          "RoPE, softmax and the rest run in torch/numpy with the host's libm, "
          "and every step compounds the last")
PARTIAL = {
    **dict.fromkeys(("block_delegated", "block_i4", "block_i16",
                     "block_wide"), BLOCK),
    **dict.fromkeys(("epiglu_delegated", "epiglu_m1", "epiglu_split"), EPIGLU),
    "pt2e_ties": PT2E_EAGER,
}


def new_refs():
    """Every row's reference: its checksum (for the table), plus either the
    output bits a device mirror says the device must print, or the reason the
    row is held to TOL instead. Plain dicts, so the checkpoint is JSON."""
    return {"checksum": {}, "bits": {}, "reason": {}}


def record(refs, name, out, reason=None):
    """File one row's reference output. A row is exact unless a reason (given
    here, or PARTIAL's) says why its reference is not the device's arithmetic."""
    reason = reason or PARTIAL.get(name)
    refs["checksum"][name] = checksum(out)
    if reason:
        refs["reason"][name] = reason
        refs["bits"].pop(name, None)
    else:
        refs["bits"][name] = output_bits(out)
        refs["reason"].pop(name, None)


def alias(refs, name, source):
    """`name` runs the same model on the same input as `source`: its reference."""
    for table in refs.values():
        if source in table:
            table[name] = table[source]
        else:
            table.pop(name, None)


def mirror_near_eager(refs, name, out, eager_name):
    """Hold `name` to its device mirror's bits, and first the mirror to the
    eager reference at TOL: the exact row proves the mirror IS the device, and
    this keeps proving the device computes the model."""
    eager = refs["checksum"][eager_name]
    rel = abs(checksum(out) - eager) / max(abs(eager), 1e-9)
    if rel >= TOL:
        raise SystemExit(f"{name}'s device mirror is {rel:.2e} from eager "
                         f"{eager_name}: the mirror no longer computes the model")
    record(refs, name, out)


def _fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.relpath(path, HERE).encode())
        with open(path, "rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def export_source_fingerprint():
    """Conservative invalidation: any change to the backend's Python or the
    example's scripts."""
    backend = os.path.join(str(firmware.ET_ROOT), "backends", "esp32s3")
    return _fingerprint(
        glob.glob(os.path.join(backend, "**", "*.py"), recursive=True)
        + [path for sub in ("", "scripts", "story")
           for path in glob.glob(os.path.join(PROJ, sub, "*.py"))])


def exported_artifact_fingerprint():
    """Bind a resume checkpoint to the exact generated files it describes."""
    paths = []
    for pattern in ("*.pte", "*.ptd", "*_native.bin", "*_inputs.h",
                    "asym_*.bin", "stock_llm_*.bin", "cnn_*.bin"):
        paths.extend(glob.glob(os.path.join(MAIN, pattern)))
    return _fingerprint(paths)


def idf(*args):
    """An idf.py invocation that works in both environments this runs in.

    Locally, sourcing export.sh does not persist across tool calls, so every idf.py
    call goes through a one-shot helper that exports IDF_PATH and sources it. In CI
    the environment is already activated (the espressif/idf image sets IDF_PATH and
    puts idf.py on PATH), and there is no helper script to point at. Detecting which
    beats requiring the helper to exist everywhere.
    """
    if os.environ.get("IDF_PATH") and shutil.which("idf.py"):
        return ["idf.py", *args]
    if not IDFENV or not os.path.exists(IDFENV):
        raise SystemExit(
            f"no activated ESP-IDF and no IDF_ENV_SCRIPT helper ({IDFENV}).\n"
            "  Either source $IDF_PATH/export.sh, or set IDF_ENV_SCRIPT to a script\n"
            "  that does so and exec's its arguments."
        )
    return ["sh", IDFENV, "idf.py", *args]


def export_all(refs):
    """Export each model seeded, and file its eager output as its reference."""
    from executorch.exir import to_edge_transform_and_lower

    from executorch.backends.esp32s3.test.models import attntail_mirror_out

    for name, (factory, shape) in SPECS.items():
        torch.manual_seed(0)
        model = factory().eval()
        x = ramp_input(shape)
        with torch.no_grad():
            eager = model(x)
        if name not in NO_PORTABLE_ROW:
            record(refs, name, eager, EAGER)


        ep = torch.export.export(model, (x,))
        edge = to_edge_transform_and_lower(ep)
        rep = apply_passes(edge, quiet=True)
        et = edge.to_executorch()
        dest = os.path.join(MAIN, f"{name}.pte")
        with open(dest, "wb") as f:
            f.write(et.buffer)
        print(f"  exported {name}.pte  ({os.path.getsize(dest):,} B)  "
              f"eager checksum {checksum(eager):.6f}  "
              f"[{rep['before']}->{rep['after']} nodes]")
    # attntail's ops after x * x are one espressif::attn_tail, whose arithmetic
    # models.attn_tail_f32_reference is -- the decode goldens rely on that --
    # so the row is held to that mirror's bits (and the mirror to eager).
    mirror_near_eager(refs, "attntail", attntail_mirror_out(), "attntail")
    # layernorm2dx's fused espressif::layer_norm2d is the graph it replaces,
    # bit for bit, and so is the host runner's (built without contraction):
    # the row is held to the host runner's bits, which a multiply-add fused
    # on the chip would miss (models.LayerNorm2dX).
    for name in HOST_EXACT:
        record(refs, name, host_output(os.path.join(MAIN, f"{name}.pte"),
                                       ramp_input(SPECS[name][1])))


# SPECS rows held to the host runner's output bits instead of eager's.
HOST_EXACT = ("layernorm2dx",)


def host_output(pte, x):
    """The host runner's one float32 output of the program at `pte` on `x`."""
    import numpy as np

    from executorch.backends.esp32s3.test.fixtures import cnn

    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        np.ascontiguousarray(x.numpy(), dtype="<f4").tofile(inp)
        done = subprocess.run([str(cnn.host_runner()), "--pte", pte, "--input",
                               inp, "--output", out], capture_output=True,
                              text=True, timeout=600)
        if done.returncode != 0:
            raise SystemExit(f"host runner on {pte}: exit {done.returncode}: "
                             f"{(done.stdout + done.stderr)[-1500:]}")
        return torch.from_numpy(np.fromfile(out, dtype="<f4").reshape(x.shape))


# Models exported for the DELEGATED row only. conv3dd is here because
# ExecuTorch's portable aten::convolution.out refuses a 5-D input outright --
# "Expect input tensor to be 3-D or 4-D, but got 5" -- so a 3-D convolution has
# no portable path on this runtime at all. Its delegated twin is therefore not
# an acceleration of something that already worked; it is the only way the op
# runs. Skipped rather than deleted so the reason stays attached to the fact.
NO_PORTABLE_ROW = {
    # No portable path exists: ET's aten::convolution.out refuses 5-D input.
    "conv3dd",
    # A portable twin would add nothing: convd already covers portable conv2d
    # and a portable relu is covered everywhere. What this fixture tests is the
    # FUSION, which only exists on the delegated side.
    "convrelud",
    "convsilud",
    # The fused-epilogue fixtures: what they test is the fusion.
    "epiglud",
    "epiactsd",
}


def export_ptd_variant(refs):
    """Export block with program-data separation; same weights, so same reference."""
    from executorch.backends.esp32s3.test.parity import export_ptd
    export_ptd.main()
    alias(refs, "block_split", "block")


def export_delegated_variant(refs):
    """Export every delegated fixture and file its device mirror's output. Held
    against the delegate's own arithmetic, not eager -- see
    mirrors.device_epilogue for why that is exact rather than merely close."""
    from executorch.backends.esp32s3.test.parity import export_delegated
    from executorch.backends.esp32s3.test import models

    # export_delegated returns references that depend on values recorded DURING
    # export (mlp_static's calibration amaxes, the tensors a fixture packed);
    # everything else is recomputed here.
    for name, out in export_delegated.main().items():
        record(refs, name, out)
    mlp = models.mlp_i8_out()
    record(refs, "mlp_delegated", mlp)
    record(refs, "mlp_perop", mlp)
    # Opcode 1's k-outer fp32 loop, whose multiply-adds the build fuses.
    mirror_near_eager(refs, "mlp_fp32_backend", models.mlp_fp32_backend_out(),
                      "mlp")
    # Each rung of the weight-format ladder against its own reference. Holding i4 to
    # the i8 number would be holding a coarser quantiser to the finer one's answer,
    # which is not a parity check -- it is a demand that quantisation be lossless.
    record(refs, "mlpr_delegated", models.mlpr_i8_out())
    record(refs, "mlps_delegated", models.mlps_i8_out())
    record(refs, "epiglu_delegated", models.epiglu_i8_out())
    record(refs, "epiglu_m1", models.epiglu_i8_out((1, 48)))
    # A split moves whole columns between cores, never a column's arithmetic.
    alias(refs, "epiglu_split", "epiglu_m1")
    record(refs, "epiacts_delegated", models.epiacts_i8_out())
    record(refs, "mlp_i4", models.mlp_iq_out("i4"))
    record(refs, "convd_delegated", models.convd_i8_out())
    record(refs, "conv1d_delegated", models.conv1d_i8_out())
    record(refs, "conv3d_delegated", models.conv3d_i8_out())
    record(refs, "dwconv_delegated", models.dwconv_i8_out())
    record(refs, "dwconv_i4", models.dwconv_iq_out("i4"))
    record(refs, "dwconv_i16", models.dwconv_iq_out("i16"))
    record(refs, "dwconvres_delegated", models.dwconvres_i8_out())
    record(refs, "dwconv1d_delegated", models.dwconv1d_i8_out())
    record(refs, "convrelu_delegated", models.convrelu_i8_out())
    record(refs, "convsilu_delegated", models.convsilu_i8_out())
    record(refs, "grpconv_delegated", models.groupconv_i8_out())
    record(refs, "grpconvedge_delegated", models.groupconvedge_i8_out())
    record(refs, "grpconv8_delegated", models.groupconv8_i8_out())
    record(refs, "grpconvrelu_delegated", models.groupconvrelu_i8_out())
    record(refs, "convres_delegated", models.convres_i8_out())
    record(refs, "convresview_delegated", models.convresview_i8_out())
    record(refs, "convres_i16", models.convres_iq_out("i16"))
    record(refs, "conv_i4", models.convd_iq_out("i4"))
    record(refs, "conv_i16", models.convd_iq_out("i16"))
    record(refs, "convt_i4", models.convt_iq_out("i4"))
    record(refs, "convt_i16", models.convt_iq_out("i16"))
    record(refs, "grpconv8_i4", models.groupconv8_iq_out("i4"))
    record(refs, "grpconv8_i16", models.groupconv8_iq_out("i16"))
    record(refs, "convt_delegated", models.convt_i8_out())
    record(refs, "convtop_delegated", models.convtop_iq_out("i8"))
    record(refs, "convtdil_delegated", models.convtdil_iq_out("i8"))
    record(refs, "convtdil_i16", models.convtdil_iq_out("i16"))
    record(refs, "mlp_i16", models.mlp_iq_out("i16"))
    # The same model with PIE forced off. Holding it to the same value turns
    # "the scalar fallback agrees" from a claim into a checked one.
    alias(refs, "mlp_scalarpath", "mlp_delegated")
    # The min-latency (column-split) rows: same models, worker on the second
    # core. A split moves whole columns between cores without touching any
    # column's accumulation order, so the expectations are the SAME references.
    alias(refs, "mlp_splitcols", "mlp_delegated")
    alias(refs, "mlp_i4_split", "mlp_i4")
    record(refs, "block_delegated", block_i8_out())
    record(refs, "block_i4", models.block_iq_out("i4"))
    record(refs, "block_i16", models.block_iq_out("i16"))


def export_decode_variant():
    """Re-export the standalone packed-i4 decode format control.

    The ordinary i8 architecture lives in one prefill+decode multimethod PTE
    below. Keeping i4 independent makes it a strict packed-format sibling rather
    than entangling the shared-state runtime test with a second storage format.
    """
    from executorch.backends.esp32s3.test.parity import export_glimmer_decode

    refs = {}
    output = "decode_i4"
    export_glimmer_decode.main(["--format", "i4", "--output", output])
    with open(os.path.join(MAIN, f"{output}_ref.json")) as file:
        refs[output] = json.load(file)
    return refs


def export_glimmer_multimethod_variant():
    """Export i8 prefill+decode with one PTD and shared mutable KV state.

    Its int8-K-cache twin is exported here too and rides along under "kc8":
    the firmware embeds both programs against the one glimmerw partition.
    """
    from executorch.backends.esp32s3.test.parity import export_glimmer_multimethod

    meta = export_glimmer_multimethod.main()
    meta["kc8"] = export_glimmer_multimethod.export_kcache_variant(meta)
    return meta


def export_pt2e_variant(refs):
    """The standard PT2E flow (Esp32Quantizer), lowered and referenced AOT.

    The ENTRY POINT is the claim: PT2E is what ExecuTorch documents. Its
    reference is built from the packed constants.
    """
    from executorch.backends.esp32s3.test.parity import export_pt2e

    record(refs, "mlp_pt2e", export_pt2e.main())
    record(refs, "actshare", export_pt2e.export_actshare_fixture())
    record(refs, "convnet_pt2e", export_pt2e.export_conv_fixture())
    # Held to PT2E's own eager quantize/dequantize, not to a device mirror: the
    # activations are exact rounding ties and a value below PT2E's minimum
    # scale, where the delegate's own quantiser and PT2E's part ways.
    record(refs, "pt2e_ties", export_pt2e.export_ties_fixture(), PT2E_EAGER)


def export_asym_variant(refs):
    """Weight zero points and float offsets (fixtures/asym.py): each row
    held to its mirrors.py device mirror's bits. The programs ride in the
    asymprog flash partition (asym_fixtures.bin) -- the app image has no room
    for them -- and the tied head's weights in asymw (asym_weights.ptd)."""
    from executorch.backends.esp32s3.test.fixtures import asym

    for name, out in asym.export_all(MAIN).items():
        if name in asym.QEMU_ROWS:
            record(refs, name, out)


def export_asym_act_variant(refs):
    """Dynamic activation quantisers (fixtures/asym_act.py): each row
    held to its mirrors.py device mirror's bits. The programs ride in the
    asymact flash partition (asym_act_fixtures.bin), the tied head's weights
    in asymactw (asym_act_weights.ptd)."""
    from executorch.backends.esp32s3.test.fixtures import asym_act

    for name, out in asym_act.export_all(MAIN).items():
        if name in asym_act.QEMU_ROWS:
            record(refs, name, out)


def export_asym_qat_variant(refs):
    """QAT round trips (fixtures/asym_qat.py): four nano models trained
    with fake quantisation, converted and lowered, each held to its mirror's
    bits; the programs ride in the asymqat flash partition
    (asym_qat_fixtures.bin). A row whose exported grid is not the one its
    fake-quantisers trained fails here; the fake-quant forward against the
    converted model is reported."""
    from executorch.backends.esp32s3.test.fixtures import asym_qat

    for name, out in asym_qat.export_all(MAIN).items():
        record(refs, name, out)
    for name, r in asym_qat.export_all.reports.items():
        print(f"  {name}: trained grid {'exact' if r.exact() else 'NOT exact'} "
              f"({r.grid_compared} compared); fake-quant forward vs converted "
              f"max abs {r.max_abs:.3g}, {r.flips} of {r.elements} differ")
        if not r.exact():
            raise SystemExit(f"{name}: the exported grid is not the trained "
                             f"one: {r.grid_mismatches} {r.buffer_mismatches}")


CODE = ("held to at most one code of its output's grid: a float reduction "
        "in another order than ATen's, compared element by element from the "
        "values the device prints")


def export_asym_static_variant(refs):
    """Static int8 graphs (fixtures/asym_static.py): blocks from the
    stock static quantizers, each held to its mirrors.py device mirror's
    bits -- or, for the float reductions between delegates, its first output
    to at most one code of the mirror's, element by element (refs["codes"]);
    the programs ride in the asymstat and asymops flash partitions
    (asym_static_fixtures.bin, asym_ops_fixtures.bin). Each row's gap to the
    recipe's eager model -- output codes flipped, the largest difference --
    comes back for the report, and a row past Stage A's bar (one flip, near
    a tie) fails; so does a table row whose tables differ from eager's
    activations anywhere but at a recorded split code in torch's scalar tail
    (asym_static.LutGap)."""
    import numpy as np

    from executorch.backends.esp32s3.test.fixtures import asym_static

    codes = refs.setdefault("codes", {})
    for name, out in asym_static.export_all(MAIN).items():
        fixture = asym_static.export_all.fixtures[name]
        if fixture.bar == "code":
            record(refs, name, out, CODE)
            first = np.asarray(out, np.float32).reshape(-1)[:fixture.measured]
            codes[name] = {"reference": [int(v) for v in first.view(np.uint32)],
                           "grid": list(fixture.grid)}
        else:
            record(refs, name, out)
            codes.pop(name, None)
    gaps = {}
    for name, gap in asym_static.export_all.gaps.items():
        print(f"  {name}: vs eager {gap.flips} code flip(s) "
              f"(tie distances {[round(d, 6) for d in gap.tie_distances]}), "
              f"max abs {gap.max_abs:.3g}")
        if not gap.ok():
            raise SystemExit(f"{name}: {gap.flips} output code(s) differ from "
                             f"eager, not all near a tie: Stage A's bar")
        gaps[name] = {"flips": gap.flips, "max_abs": gap.max_abs,
                      "tie_distances": gap.tie_distances}
        lut = asym_static.export_all.fixtures[name].lut_gap
        if lut is not None:
            print(f"    tables vs eager: {lut.summary()}")
            if not lut.ok():
                raise SystemExit(f"{name}: a table differs from eager's "
                                 "activation outside a recorded split code "
                                 "in a scalar-tail position")
            gaps[name]["lut"] = lut.summary()
    return gaps


def export_asym_stageb_variant(refs):
    """Stage B (fixtures/asym_stage_b.py): each row's
    Stage A twin (sa_*) and Stage B variant (sb_*), both held to the same
    bits -- the device mirror's, Stage A's -- so a pass proves Stage B ==
    Stage A on the device; the programs ride in the asymstgb flash partition
    (asym_stageb_fixtures.bin). A variant that did not do what its row says
    (codes written, ops on codes, boundaries kept fp32) fails the export."""
    from executorch.backends.esp32s3.test.fixtures import asym_stage_b

    for name, out in asym_stage_b.export_all(MAIN).items():
        record(refs, name, out)
    for name, report in asym_stage_b.export_all.reports.items():
        print(f"  {name}: {report.summary()}")


def export_lut16_variant(refs):
    """The int16 table op (fixtures/lut16.py): one program per
    a16w8 activation site, the activation a fitted espressif::lut16 table,
    each row exact against the numpy mirror (which the export holds equal to
    the folded program's torch reference); the programs ride in the asymlut16
    flash partition (asym_lut16_fixtures.bin). Returns each row's flips
    against eager, which the export holds within the table's bar."""
    from executorch.backends.esp32s3.test.fixtures import lut16

    for name, out in lut16.export_all(MAIN).items():
        record(refs, name, out)
    gaps = lut16.export_all.gaps
    for name, gap in gaps.items():
        if gap["flips"] > gap["bar"] or gap["max_steps"] > 1:
            raise SystemExit(f"{name}: {gap['flips']} input(s) flip against eager, "
                             f"max {gap['max_steps']} code(s): past the table's bar")
    return gaps


def export_qnorm_variant(refs):
    """The fused quantised norms (fixtures/qnorm.py): each row's
    stock-recipe norm folded into espressif::qnorm, held to the mirror's
    output bits; the programs ride in the asymnorm flash partition
    (asym_norm_fixtures.bin). Against the recipe's eager model a row may
    differ by at most one code at a reduction's quantiser and must be eager's
    bit for bit wherever its reduction codes agree (qnorm.NormGap);
    the gaps come back for the report."""
    from executorch.backends.esp32s3.test.fixtures import qnorm

    for name, out in qnorm.export_all(MAIN).items():
        record(refs, name, out)
    gaps = {}
    for name, gap in qnorm.export_all.gaps.items():
        print(f"  {name}: vs eager {gap.summary()}")
        if not gap.ok():
            raise SystemExit(f"{name}: the fused norm is past its bar against "
                             f"eager: {gap.summary()}")
        gaps[name] = gap.summary()
    return gaps


def export_cnn_variant(refs):
    """The CNN and RNN models (fixtures/cnn.py): MLPerf Tiny's DS-CNN and
    ResNet-8 and the census's LSTM and GRU, Arm a8w8 lowered with the
    backend's defaults, each held to the host runner's output bits (the
    delegate's and the portable kernels' own C++, which for these programs is
    the device's arithmetic); the programs ride in the cnnrnn flash partition
    (cnn_fixtures.bin), their named data in cnnrnnw (cnn_weights.ptd). The
    export holds each row to the census's bar against the recipe's eager
    model; what each row measured comes back for the report, and a
    board-only row (more memory than QEMU has) is compared on silicon only."""
    from executorch.backends.esp32s3.test.fixtures import cnn

    for name, out in cnn.export_all(MAIN).items():
        record(refs, name, out)
    return {name: {"summary": cnn.summary(f), "arena": f.report["arena"],
                   "temp": f.report["temp"], "vs_eager": f.gap,
                   "board_only": f.row.board_only, "split_pass": f.split_pass,
                   "convsplit": f.report.get("convsplit", 0)}
            for name, f in cnn.export_all.fixtures.items()}


def export_stock_llm_variant():
    """ExecuTorch's own LLM recipes (fixtures/stock_llm.py): each row a program
    run as a sequence from the stockllm partition. Per row and step: the
    mirror's output bits (exact rows), or eager's checksum and top-1 token
    (the full nano-llama recipe, held to TOL)."""
    import numpy as np

    from executorch.backends.esp32s3.test.fixtures import stock_llm

    stock = {}
    for fixture in stock_llm.export_all(MAIN):
        steps = [np.concatenate([np.asarray(r, np.float32).reshape(-1)
                                 for r in refs])
                 for refs in fixture.references]
        stock[fixture.name] = {
            "exact": fixture.name in stock_llm.EXACT_ROWS,
            "bits": [output_bits(s) for s in steps],
            "checksums": [checksum(s) for s in steps],
            "top1": [int(np.asarray(refs[0]).reshape(-1).argmax())
                     for refs in fixture.references],
        }
    return stock


def compare_stock_llm(log, stock):
    """Every step of every stock-recipe row: exact rows' output bits equal to
    their mirror's, the nano-llama recipe's checksums within TOL of eager and
    its top-1 tokens reported; and the rows really ran the affine K-cache op
    and the cache kernels."""
    ok = True
    for name, ref in stock.items():
        got = {int(step): (float(cs), int(top), bits) for step, cs, top, bits
               in re.findall(rf"^\s*{re.escape(name)} step (\d+): checksum "
                             r"(-?[\d.]+) top1 (-?\d+) bits ([0-9a-f]{8})",
                             log, flags=re.MULTILINE)}
        steps = len(ref["bits"])
        if len(got) != steps:
            print(f"  WARNING: {name}: {len(got)} of {steps} step(s) ran")
            ok = False
            continue
        if ref["exact"]:
            bad = [s for s in range(steps) if got[s][2] != ref["bits"][s]]
            print(f"{name}: {steps - len(bad)}/{steps} step(s) bit-identical "
                  "to the mirror")
            if bad:
                print(f"  WARNING: {name} steps {bad} differ from the mirror")
                ok = False
            continue
        # Not a device mirror: the delegate's K cache scores symmetric query
        # rows, and portable softmax/RMSNorm use newlib's libm. Reported against
        # eager, and failed only if the greedy tokens stop agreeing.
        rel = max(abs(got[s][0] - ref["checksums"][s])
                  / max(abs(ref["checksums"][s]), 1e-9) for s in range(steps))
        top1 = sum(got[s][1] == ref["top1"][s] for s in range(steps))
        print(f"{name}: {steps} step(s) against eager: checksum max rel "
              f"{rel:.2e}, top-1 agreement {top1}/{steps}")
        if top1 < STOCK_TOP1_FLOOR * steps:
            print(f"  WARNING: {name}'s tokens agree with eager on fewer than "
                  f"{STOCK_TOP1_FLOOR:.0%} of steps")
            ok = False
    stats = re.search(r"stock llm: (\d+) affine K-cache op\(s\), (\d+) key "
                      r"row\(s\); writer rows (\d+), read rows (\d+), cache "
                      r"writes (\d+), refusals (\d+)", log)
    if stats:
        kc, rows, writer, read, writes, refused = (int(v) for v in stats.groups())
        print(f"stock llm: {kc} affine K-cache op(s) over {rows} key row(s); "
              f"{writer} rows written by the A4 writer, {read} read, {writes} "
              f"cache writes, {refused} refusal(s)")
        if not (kc and writer and read and writes) or refused:
            print("  WARNING: the stock rows did not run the affine K cache "
                  "and its kernels, or a kernel refused")
            ok = False
    else:
        print("  WARNING: no stock llm line -- the stock-recipe rows did not run")
        ok = False
    return ok


def export_asym_a16_variant(refs):
    """int16 activations x int8 weights (fixtures/asym_a16.py, Arm
    a16w8): each row held to its mirrors.py device mirror's bits
    (the int64 dot); the programs ride in the asyma16 flash partition
    (asym_a16_fixtures.bin). Each row's gap to the recipe's eager model comes
    back for the report; a flip of more than one int16 step fails (eager's
    float GEMM and the exact dot may straddle a tie, by one code) -- but for
    a row whose float weights the backend quantises (WEIGHTS_HERE: Cortex-M's
    int16 config), whose gap is that quantisation."""
    from executorch.backends.esp32s3.test.fixtures import asym_a16

    for name, out in asym_a16.export_all(MAIN).items():
        record(refs, name, out)
    gaps = {}
    for name, gap in asym_a16.export_all.gaps.items():
        print(f"  {name}: vs eager {gap.flips} code flip(s) of at most "
              f"{gap.steps} step(s) (tie distances "
              f"{[round(d, 6) for d in gap.tie_distances]}), max abs "
              f"{gap.max_abs:.3g}")
        if gap.steps > 1 and name not in asym_a16.WEIGHTS_HERE:
            raise SystemExit(f"{name}: an output differs from eager by "
                             f"{gap.steps} int16 steps")
        gaps[name] = {"flips": gap.flips, "steps": gap.steps,
                      "max_abs": gap.max_abs,
                      "tie_distances": gap.tie_distances}
    return gaps


def export_asym_cases():
    """The synthetic refusal blobs and the carrier program they ride in
    (asym_blob_cases.py), for main.cpp's run_asym_refusal_check: one blob per
    runtime rule, each to be refused for that rule and no other."""
    from executorch.backends.esp32s3.test.fixtures import asym_blob_cases

    carrier = asym_blob_cases.export_carrier(
        os.path.join(MAIN, "asym_carrier.pte"))
    asym_blob_cases.write_cases(os.path.join(MAIN, "asym_refusal_cases.bin"),
                                bytes(carrier))
    print(f"  asym_carrier.pte + asym_refusal_cases.bin: "
          f"{len(asym_blob_cases.CASES)} v24 refusal case(s)")


def export_quant_lut_variant(refs):
    """espressif::quant_lut's rows (fixtures/quant_lut.py): luts.pte, one method
    per row, each held to its device mirror's bits. The export enforces each
    row's bar against eager; the gaps it measured come back for the report."""
    from executorch.backends.esp32s3.test.fixtures import quant_lut

    lut_refs, gaps = quant_lut.export(MAIN)
    for name, out in lut_refs.items():
        record(refs, name, out)
        refs["checksum"][name] = quant_lut.finite_checksum(out)
    for line in quant_lut.split_summary(gaps):
        print(f"  {line}")
    print(f"  {quant_lut.PROGRAM}: "
          f"{os.path.getsize(os.path.join(MAIN, quant_lut.PROGRAM)):,} B, "
          f"{len(lut_refs)} table row(s)")
    return gaps


def export_memory_plan_variant(refs):
    """memplan.pte: sub (both operand positions), clamp and hardtanh writing
    over an operand that dies there, planned with memory_plan="esp32" whatever
    --memory-plan says, so the QEMU run proves those kernels alias-safe on the
    device (add, mul, relu, sigmoid, tanh and quant_lut do it in the other rows
    under the default --memory-plan esp32). Exact against eager:
    models.MemplanKernels."""
    from executorch.exir import ExecutorchBackendConfig, to_edge

    from executorch.backends.esp32s3 import memory_planning
    from executorch.backends.esp32s3.test.models import MemplanKernels

    model = MemplanKernels().eval()
    x = ramp_input((4, 64))
    with torch.no_grad():
        eager = model(x)
    planner = memory_planning.planning_pass("esp32")
    et = to_edge(torch.export.export(model, (x,))).to_executorch(
        ExecutorchBackendConfig(memory_planning_pass=planner))
    written = memory_planning.inplace_kernels(et)
    want = {"aten::sub.out": 2, "aten::clamp.out": 1, "aten::hardtanh.out": 1}
    if any(written.get(op, 0) < n for op, n in want.items()):
        raise SystemExit(f"memplan.pte: expected {want} in place, the plan has "
                         f"{written}: {planner.plans[0].describe()}")
    dest = os.path.join(MAIN, "memplan.pte")
    with open(dest, "wb") as f:
        f.write(et.buffer)
    record(refs, "memplan", eager)
    print(f"  exported memplan.pte  ({os.path.getsize(dest):,} B)  in place: "
          + ", ".join(f"{op} x{n}" for op, n in sorted(written.items())))


def portable_choosers(path=None):
    """Every choose_qparams* operator in a required-op list. Never allowed:
    ExecuTorch's portable choose_qparams disagrees with PyTorch's choosers,
    so a dynamic chooser is absorbed by the delegate (strip_dynamic_act_qdq
    refuses any it cannot) and never registered as a kernel."""
    path = path or str(firmware.REQUIRED_OPS)
    with open(path) as f:
        return [line.strip() for line in f
                if "choose_qparams" in line and not line.startswith("#")]


def refuse_portable_choosers():
    found = portable_choosers()
    if found:
        raise SystemExit(f"required_ops.txt registers portable choosers "
                         f"{found}: a dynamic chooser must be absorbed by the "
                         f"delegate, never run portable")


def generate_selective_kernels(fast_layernorm=False):
    """Derive and generate the exact portable kernel set for this export.

    These outputs are intentionally ignored, so a clean checkout (or the build
    cleanup this project routinely performs) cannot assume they exist.  Keep
    generation in the one-command parity contract; otherwise a stale tree can
    pass while a clean tree fails before CMake, or worse, loads a stale registry.
    """
    required = str(firmware.REQUIRED_OPS)
    scripts = (
        ("operator census", "dump_required_ops.py",
         [os.path.join(HERE, "dump_required_ops.py"), "--out", required]),
        # The element-wise loops (ops/elementwise.h) as the bench firmware
        # registers them, so the zoo's mul.out and constant_pad_nd.out calls
        # run them too (quant_lut's loops run wherever a table op does).
        ("ExecuTorch codegen", "gen_kernels.py",
         [str(firmware.GEN_KERNELS), "--required", required,
          "--elementwise-fast"]),
    )
    for label, script, args in scripts:
        command = [sys.executable, *args]
        if script == "gen_kernels.py" and fast_layernorm:
            command.append("--fast-layernorm")
        result = subprocess.run(
            command,
            cwd=HERE, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-2000:])
            raise SystemExit(f"{label} FAILED")
        useful = [line for line in result.stdout.splitlines()
                  if line.startswith(("wrote ", "codegen  -> ", "requested "))]
        print("  " + ("; ".join(useful) if useful else f"{label} complete"))
        if script == "dump_required_ops.py":
            refuse_portable_choosers()


def build(fast_layernorm=False):
    # ET_QEMU_NO_RSQRT0: QEMU faults with IllegalInstruction on rsqrt0.s, which is the
    # single instruction rmsnorm_s3.S exists to reach. The assembly still assembles and
    # links here; only its execution moves to the scalar reference. See main.cpp.
    command = [
        "-C", PROJ, "-DET_QEMU_NO_RSQRT0=1",
        f"-DESP32_FAST_LAYERNORM={int(fast_layernorm)}",
    ]
    command.append("build")
    r = subprocess.run(idf(*command),
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise SystemExit("build FAILED")
    size = re.search(r"et_runner.bin binary size (0x[0-9a-f]+)", r.stdout)
    print(f"  built, image {int(size.group(1), 16):,} B" if size else "  built")


def run_qemu(timeout=180):
    """Run until the firmware prints DONE, then kill QEMU and return the log."""
    p = subprocess.Popen(idf("-C", PROJ, "qemu"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, start_new_session=True)
    lines, deadline = [], time.time() + timeout
    try:
        while time.time() < deadline:
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line)
            if "=== DONE ===" in line:
                break
            if "Guru Meditation" in line or "panic" in line.lower():
                lines.append("!! device panic\n")
                break
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    return "".join(lines)


def parse_device(log):
    """Pull {model: checksum} out of the device log."""
    got, cur = {}, None
    for line in log.splitlines():
        m = re.search(r"^--- (\S+)\.pte", line)
        if m:
            cur = m.group(1)
        m = re.search(r"^\s*checksum: (-?[\d.]+)", line)
        if m and cur:
            got[cur] = float(m.group(1))
    return got


def model_values(log, name):
    """Every output value a row printed ("  values:" lines, FIXTURE_DUMP), as
    float32, or None."""
    import numpy as np

    section = re.search(rf"^--- {re.escape(name)}\.pte.*?(?=^--- |\Z)", log,
                        flags=re.MULTILINE | re.DOTALL)
    if section is None:
        return None
    words = [int(w, 16) for line in re.findall(r"^\s*values:((?: [0-9a-f]{8})+)",
                                               section.group(0), re.MULTILINE)
             for w in line.split()]
    return np.array(words, dtype=np.uint32).view(np.float32) if words else None


def model_bits(log, name):
    """The `output bits` fingerprint printed for one embedded model, or None."""
    section = re.search(rf"^--- {re.escape(name)}\.pte.*?(?=^--- |\Z)", log,
                        flags=re.MULTILINE | re.DOTALL)
    if section is None:
        return None
    bits = re.search(r"output bits: ([0-9a-f]{8})", section.group(0))
    return bits.group(1) if bits else None


def step_bits(log, label):
    """{step: output bits} for every execution of one cached-sequence label."""
    return {int(step): bits for step, bits in re.findall(
        rf"{re.escape(label)} step (\d+):.*?bits ([0-9a-f]{{8}})", log)}


def compare_cached(log, label, reference):
    """Compare every cached decode/prefill execution and report its timing."""
    rows = [(int(a), float(b), int(c)) for a, b, c in re.findall(
        rf"{re.escape(label)} step (\d+):.*?checksum (-?[\d.]+)\s+time (\d+) us",
        log)]
    host = reference["checksums"]
    if rows:
        agree = 0
        for step, dev, elapsed_us in rows:
            if step >= len(host):
                break
            rel = abs(dev - host[step]) / max(abs(host[step]), 1e-9)
            print(f"  {label} step {step}: host {host[step]:.6f}, "
                  f"device {dev:.6f}, rel {rel:.2e}, {elapsed_us:,} us")
            if step == agree and rel < TOL:
                agree += 1
        got_steps = [step for step, _, _ in rows]
        good = got_steps == list(range(len(host))) and agree == len(host)
        print(f"\n{label}: {len(rows)} steps on device, first {agree} within "
              f"{TOL:g} of host ({len(host)} host steps available)")
        if not good:
            print(f"  WARNING: {label} did not complete with full host parity")
        return good

    print(f"\n  WARNING: {label} never ran (stale .pte, missing operator, "
          "or missing data partition)")
    section = re.search(rf"^--- {re.escape(label)}\.pte.*?(?=^--- |\Z)", log,
                        flags=re.MULTILINE | re.DOTALL)
    if section:
        for line in section.group(0).rstrip().splitlines():
            print(f"    {line}")
    return False


@fewer_gc_passes()  # the export stage lowers ~100 programs
def main(argv=None, *, build_fn=None, run_fn=None, target="QEMU"):
    """The whole suite. run_hardware.py passes its own build_fn (returning
    the build directory) and run_fn (returning the device log) to run the same
    exports, references and comparisons on silicon."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast-layernorm", action="store_true",
        help="run the full gate with reciprocal-times-input LayerNorm enabled",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse the last complete export checkpoint, then rebuild and run",
    )
    parser.add_argument(
        "--log", help="also write the whole QEMU log here",
    )
    parser.add_argument(
        "--memory-plan", choices=("esp32", "none"), default="esp32",
        help="export every program with memory_planning.py's arena planning "
             "(the default: in-place reuse, operator order, hill-climb "
             "allocator); none is ExecuTorch's own planner",
    )
    args = parser.parse_args(argv)
    if args.memory_plan == "none":
        args.memory_plan = None
    # The exporters calibrate stock recipes in this process: torch.histc's
    # float32 overflow must not corrupt its heap (histc_guard.py).
    from executorch.backends.esp32s3.test import histc_guard

    histc_guard.install()
    if args.resume:
        print("[1/5] reusing complete export checkpoint")
        try:
            with open(CHECKPOINT) as checkpoint_file:
                checkpoint = json.load(checkpoint_file)
        except FileNotFoundError:
            raise SystemExit(
                f"no export checkpoint at {CHECKPOINT}; run without --resume first")
        if bool(checkpoint["fast_layernorm"]) != args.fast_layernorm:
            raise SystemExit(
                "checkpoint LayerNorm mode differs; run without --resume")
        if checkpoint.get("source_fingerprint") != export_source_fingerprint():
            raise SystemExit(
                "export source changed since checkpoint; run without --resume")
        if (checkpoint.get("artifact_fingerprint")
                != exported_artifact_fingerprint()):
            raise SystemExit(
                "generated artifacts changed since checkpoint; "
                "run without --resume")
        if checkpoint.get("memory_plan") != args.memory_plan:
            raise SystemExit("checkpoint memory plan differs; run without --resume")
        refs = checkpoint["refs"]
        stock_refs = checkpoint["stock_llm_refs"]
        lut_gaps = checkpoint.get("lut_gaps", {})
        static_gaps = checkpoint.get("static_gaps", {})
        a16_gaps = checkpoint.get("a16_gaps", {})
        lut16_gaps = checkpoint.get("lut16_gaps", {})
        norm_gaps = checkpoint.get("norm_gaps", {})
        cnn_rows = checkpoint.get("cnn_rows", {})
        decode_refs = checkpoint["decode_refs"]
        glimmer_multimethod_ref = checkpoint["glimmer_multimethod_ref"]
    else:
        print("[1/5] exporting models")
        if args.memory_plan:
            from executorch.backends.esp32s3 import memory_planning
            memory_planning.install(args.memory_plan)
            print(f"  memory plan: {args.memory_plan} (every to_executorch)")
        refs = new_refs()
        export_all(refs)
        export_ptd_variant(refs)
        export_delegated_variant(refs)
        export_pt2e_variant(refs)
        decode_refs = export_decode_variant()
        glimmer_multimethod_ref = export_glimmer_multimethod_variant()
        export_asym_variant(refs)
        export_asym_act_variant(refs)
        static_gaps = export_asym_static_variant(refs)
        stock_refs = export_stock_llm_variant()
        a16_gaps = export_asym_a16_variant(refs)
        export_asym_qat_variant(refs)
        export_asym_stageb_variant(refs)
        lut16_gaps = export_lut16_variant(refs)
        norm_gaps = export_qnorm_variant(refs)
        cnn_rows = export_cnn_variant(refs)
        export_asym_cases()
        lut_gaps = export_quant_lut_variant(refs)
        export_memory_plan_variant(refs)
        os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
        temporary = CHECKPOINT + ".tmp"
        with open(temporary, "w") as checkpoint_file:
            json.dump({
                "fast_layernorm": args.fast_layernorm,
                "refs": refs,
                "decode_refs": decode_refs,
                "glimmer_multimethod_ref": glimmer_multimethod_ref,
                "lut_gaps": lut_gaps,
                "static_gaps": static_gaps,
                "stock_llm_refs": stock_refs,
                "a16_gaps": a16_gaps,
                "lut16_gaps": lut16_gaps,
                "norm_gaps": norm_gaps,
                "cnn_rows": cnn_rows,
                "source_fingerprint": export_source_fingerprint(),
                "artifact_fingerprint": exported_artifact_fingerprint(),
                **({"memory_plan": args.memory_plan} if args.memory_plan else {}),
            }, checkpoint_file, indent=1)
        os.replace(temporary, CHECKPOINT)
        print(f"  wrote complete export checkpoint {CHECKPOINT}")

    from executorch.backends.esp32s3.test.parity.artifact_census import (
        scan_embedded_artifacts,
    )
    census = scan_embedded_artifacts(MAIN)
    gaps = census.declared_coverage_gaps()
    print(f"  valid-artifact census: {census.summary()}")
    if gaps:
        raise SystemExit("valid-artifact coverage FAILED: " + ", ".join(gaps))

    print("[2/5] deriving selective kernels")
    generate_selective_kernels(args.fast_layernorm)

    print(f"[3/5] building firmware for {target}")
    if build_fn is None:
        build(args.fast_layernorm)
        build_dir = os.path.join(PROJ, "build")
    else:
        build_dir = build_fn()
    # The table ops' object code: compiled -ffp-contract=off, so no fused
    # multiply-add may appear in it (the host computes each step unfused).
    # elementwise.cpp holds quant_lut's four-lane loop.
    from executorch.backends.esp32s3.test.fixtures import quant_lut
    for obj in ("quant_lut.cpp.obj", "quant_lut16.cpp.obj",
                "elementwise.cpp.obj"):
        fused, read = quant_lut.fused_multiply_adds(build_dir, obj)
        print(f"  {obj}: {read} instructions, {len(fused)} madd.s/msub.s")
        if fused or not read:
            raise SystemExit(f"{obj}: fused multiply-adds present "
                             f"(or nothing disassembled): {fused[:3]}")

    print(f"[4/5] running on {target}")
    log = run_qemu() if run_fn is None else run_fn()
    if args.log:
        with open(args.log, "w") as log_file:
            log_file.write(log)
    if "=== DONE ===" not in log:
        print(log[-3000:])
        raise SystemExit("device did not reach DONE")

    print("[5/5] comparing\n")
    got = parse_device(log)
    expected, exact_bits, reasons = refs["checksum"], refs["bits"], refs["reason"]

    # Exact rows compare the device's output bits with the mirror's; the rel
    # beside them is information. Tolerance rows compare checksums at TOL.
    # Code rows compare their first output's codes, element by element.
    rows, ok = [], True
    exact_ok = tol_ok = code_ok = 0
    codes = refs.get("codes", {})
    code_gaps = {}
    # Rows that need PSRAM, which QEMU's firmware skips by name (checked below).
    board_only = ({name for name, row in cnn_rows.items() if row["board_only"]}
                  if target == "QEMU" else set())
    for name, exp in sorted(expected.items()):
        if name in board_only:
            rows.append((name, exp, None, "SKIPPED (needs PSRAM: silicon only)"))
            continue
        dev = got.get(name)
        if dev is None:
            rows.append((name, exp, None, "MISSING"))
            ok = False
            continue
        rel = abs(dev - exp) / max(abs(exp), 1e-9)
        if name in codes:
            import numpy as np

            from executorch.backends.esp32s3.test.fixtures.asym_static import code_gap

            want = np.array(codes[name]["reference"], np.uint32).view(np.float32)
            values = model_values(log, name)
            if values is None or values.size < want.size:
                good, verdict = False, "FAIL (no values printed)"
            else:
                gap = code_gap(want, values[:want.size], codes[name]["grid"])
                code_gaps[name] = gap
                good = gap.ok()
                verdict = f"{'CODE' if good else 'FAIL'} ({gap.summary()})"
            code_ok += good
        elif name in exact_bits:
            bits = model_bits(log, name)
            good = bits == exact_bits[name]
            exact_ok += good
            verdict = (f"EXACT {bits}" if good else
                       f"FAIL (bits {bits} != {exact_bits[name]}, rel {rel:.2e})")
        else:
            good = rel < TOL
            tol_ok += good
            verdict = f"{'PASS' if good else 'FAIL'} (rel {rel:.2e})"
        ok &= good
        rows.append((name, exp, dev, verdict))

    # Every row is one or the other, and a tolerance row says why.
    unclassified = sorted(set(expected) - set(exact_bits) - set(reasons))
    if unclassified or set(exact_bits) & set(reasons):
        print(f"  WARNING: rows neither exact nor justified at TOL: "
              f"{', '.join(unclassified)}")
        ok = False

    # A device row nobody wrote an expectation for is never compared, so adding
    # a model to main.cpp and forgetting its reference makes it run green
    # forever. The set difference is the whole check.
    unexpected = sorted(set(got) - set(expected))
    if unexpected:
        print(f"  WARNING: device rows with no expectation, so never checked: "
              f"{', '.join(unexpected)}")
        ok = False

    w = max(len(r[0]) for r in rows)
    print(f"{'model'.ljust(w)}  {'reference':>18}  {'device':>18}  verdict")
    print("-" * (w + 46))
    for name, exp, dev, verdict in rows:
        d = "n/a" if dev is None else f"{dev:.6f}" if isinstance(dev, float) else str(dev)
        e = f"{exp:.6f}" if isinstance(exp, float) else str(exp)
        print(f"{name.ljust(w)}  {e:>18}  {d:>18}  {verdict}")
    by_reason = {}
    for name in sorted(set(reasons) - set(codes)):
        by_reason.setdefault(reasons[name], []).append(name)
    if codes:
        print(f"\nRows held to at most one code ({CODE}):")
        print(f"    {', '.join(sorted(codes))}")
    print(f"\nRows held to TOL {TOL:g} rather than exact bits, and why:")
    for reason, names in by_reason.items():
        print(f"  {reason}:")
        print(f"    {', '.join(names)}")

    # Also surface the PIE-vs-scalar dispatch, since a silent fallback to scalar
    # would still pass the numeric check.
    m = re.search(r"dot_i8 dispatch: PIE (\d+), scalar fallback (\d+)", log)
    if m:
        print(f"\ndot_i8 dispatch: PIE {m.group(1)}, scalar fallback {m.group(2)}")
        if m.group(1) == "0":
            print("  WARNING: PIE kernel never ran (alignment guard fell back)")
            ok = False

    add = re.search(
        r"add\.out dispatch: same (\d+), leading-broadcast (\d+), "
        r"fallback (\d+)", log)
    if add:
        print(f"add.out dispatch: same {add.group(1)}, "
              f"leading-broadcast {add.group(2)}, fallback {add.group(3)}")
        if add.group(1) == "0" or add.group(2) == "0":
            print("  WARNING: an add.out fast path registered but never ran")
            ok = False
    else:
        print("  WARNING: no add.out dispatch line")
        ok = False

    permute = re.search(
        r"permute_copy dispatch: adjacent-block (\d+), "
        r"adjacent-element (\d+), fallback (\d+)", log)
    if permute:
        print(f"permute_copy dispatch: adjacent-block {permute.group(1)}, "
              f"adjacent-element {permute.group(2)}, "
              f"fallback {permute.group(3)}")
        if permute.group(1) == "0" or permute.group(2) == "0":
            print("  WARNING: an adjacent permute fast path never ran")
            ok = False
    else:
        print("  WARNING: no permute_copy dispatch line")
        ok = False

    bmm = re.search(
        r"bmm dispatch: p<=n (\d+), p>n (\d+), fallback (\d+)", log)
    if bmm:
        print(f"bmm dispatch: p<=n {bmm.group(1)}, p>n {bmm.group(2)}, "
              f"fallback {bmm.group(3)}")
        if (bmm.group(1) == "0" or bmm.group(2) == "0" or
                bmm.group(3) == "0"):
            print("  WARNING: a bmm fast path or portable fallback never ran")
            ok = False
    else:
        print("  WARNING: no bmm dispatch line")
        ok = False

    mul = re.search(
        r"mul\.out dispatch: same (\d+), fallback (\d+), broadcast (\d+), "
        r"scalar (\d+)", log)
    if mul:
        print(f"mul.out dispatch: same {mul.group(1)}, fallback {mul.group(2)}, "
              f"broadcast {mul.group(3)}, scalar {mul.group(4)}")
        if "0" in (mul.group(1), mul.group(3), mul.group(4)):
            print("  WARNING: a mul.out fast path registered but never ran")
            ok = False
    else:
        print("  WARNING: no mul.out dispatch line")
        ok = False

    # The element-wise loops (ops/elementwise.h) give the bits of the loops
    # they replace, so only their counts show they ran: the zoo's
    # channel-scale muls, both table loops and the pads' row copy must each
    # serve a call.
    ew_mul = re.search(r"mul\.out elementwise dispatch: channels (\d+)", log)
    ew_lut = re.search(
        r"quant_lut elementwise dispatch: fp32 (\d+), codes (\d+)", log)
    ew_pad = re.search(
        r"constant_pad_nd\.out elementwise dispatch: rows (\d+), "
        r"fallback (\d+)", log)
    if ew_mul and ew_lut and ew_pad:
        print(f"elementwise dispatch: mul channels {ew_mul.group(1)}, quant_lut "
              f"fp32 {ew_lut.group(1)} codes {ew_lut.group(2)}, constant_pad_nd "
              f"rows {ew_pad.group(1)} fallback {ew_pad.group(2)}")
        if "0" in (ew_mul.group(1), ew_lut.group(1), ew_lut.group(2),
                   ew_pad.group(1)):
            print("  WARNING: an element-wise loop registered but never ran")
            ok = False
    else:
        print("  WARNING: an elementwise dispatch line is missing")
        ok = False

    where = re.search(
        r"where\.self_out dispatch: mask-scalar (\d+), fallback (\d+)", log)
    if where:
        print(f"where.self_out dispatch: mask-scalar {where.group(1)}, "
              f"fallback {where.group(2)}")
        if where.group(1) == "0" or where.group(2) == "0":
            print("  WARNING: the where fast path or fallback never ran")
            ok = False
    else:
        print("  WARNING: no where.self_out dispatch line")
        ok = False

    div = re.search(
        r"div\.out dispatch: scalar-denominator (\d+), fallback (\d+)", log)
    if div:
        print(f"div.out dispatch: scalar-denominator {div.group(1)}, "
              f"fallback {div.group(2)}")
        if div.group(1) == "0":
            print("  WARNING: the scalar-denominator div.out fast path never ran")
            ok = False
    else:
        print("  WARNING: no div.out dispatch line")
        ok = False

    # The Glimmer multimethod decode masks are int64 [N] <= 0-d position;
    # its prefill masks broadcast [1,S] <= [S,1] and stay portable.
    le = re.search(r"le\.Tensor_out dispatch: scalar (\d+), fallback (\d+)", log)
    if le:
        print(f"le.Tensor_out dispatch: scalar {le.group(1)}, "
              f"fallback {le.group(2)}")
        if le.group(1) == "0" or le.group(2) == "0":
            print("  WARNING: the le.Tensor_out fast path or its portable "
                  "fallback never ran")
            ok = False
    else:
        print("  WARNING: no le.Tensor_out dispatch line")
        ok = False

    # The multimethod fixtures' KV writes: decode one row, prefill a block.
    index_put = re.search(
        r"index_put_ dispatch: row-copy (\d+), fallback (\d+)", log)
    if index_put:
        print(f"index_put_ dispatch: row-copy {index_put.group(1)}, "
              f"fallback {index_put.group(2)}")
        if index_put.group(1) == "0":
            print("  WARNING: the index_put_ row-copy fast path never ran")
            ok = False
    else:
        print("  WARNING: no index_put_ dispatch line")
        ok = False

    # The Glimmer decode's per-layer embedding split; the zoo's other splits
    # (block, GRU, LSTM, Mamba gates) take the same chunk copies.
    split = re.search(
        r"split_with_sizes_copy\.out dispatch: chunk-copy (\d+), "
        r"fallback (\d+)", log)
    if split:
        print(f"split_with_sizes_copy.out dispatch: chunk-copy "
              f"{split.group(1)}, fallback {split.group(2)}")
        if split.group(1) == "0":
            print("  WARNING: the split_with_sizes_copy fast path never ran")
            ok = False
    else:
        print("  WARNING: no split_with_sizes_copy.out dispatch line")
        ok = False

    # The multimethod fixtures' window slots: position % window.
    remainder = re.search(
        r"remainder\.Scalar_out dispatch: non-negative (\d+), fallback (\d+)",
        log)
    if remainder:
        print(f"remainder.Scalar_out dispatch: non-negative "
              f"{remainder.group(1)}, fallback {remainder.group(2)}")
        if remainder.group(1) == "0":
            print("  WARNING: the remainder.Scalar_out fast path never ran")
            ok = False
    else:
        print("  WARNING: no remainder.Scalar_out dispatch line")
        ok = False

    softmax = re.search(
        r"_softmax\.out dispatch: last-axis-fp32 (\d+), fallback (\d+)",
        log)
    if softmax:
        print(f"_softmax.out dispatch: last-axis-fp32 {softmax.group(1)}, "
              f"fallback {softmax.group(2)}")
        if softmax.group(1) == "0" or softmax.group(2) == "0":
            print("  WARNING: the softmax fast path or portable fallback "
                  "never ran")
            ok = False
    else:
        print("  WARNING: no _softmax.out dispatch line")
        ok = False

    # The overrides against ExecuTorch's portable kernels on hand-built
    # tensors: every written byte, the failure state and the path each case
    # must take. The model rows above cannot show a fast path that differs
    # from portable only on inputs they never produce (NaN, -0, near misses).
    fastpath = re.search(
        r"fastpath check: (\d+) case\(s\), (\d+) mismatch\(es\)", log)
    if fastpath:
        print(f"fastpath check: {fastpath.group(1)} case(s), "
              f"{fastpath.group(2)} mismatch(es)")
        for line in re.findall(r"^fastpath check: .*$", log, re.M):
            if line.strip() != fastpath.group(0):  # per operator, mismatches
                print(f"  {line.strip()}")
        if fastpath.group(1) == "0" or fastpath.group(2) != "0":
            print("  WARNING: an override's fast path disagreed with the "
                  "portable kernel")
            ok = False
    else:
        print("  WARNING: no fastpath check line -- the overrides were not "
              "compared with the portable kernels")
        ok = False

    # The opt-in fast LayerNorm's rows, built in every variant, against the
    # portable kernel's: the same mean and rstd bits, and outputs within the
    # reciprocal multiply's rounding. --fast-layernorm's model rows only hold
    # it to eager at TOL; this holds its arithmetic on the chip in every run.
    fast_ln = re.search(
        r"^fast_layer_norm check: (\d+) case\(s\), (\d+) mismatch\(es\).*$",
        log, re.M)
    if fast_ln:
        print(fast_ln.group(0))
        if fast_ln.group(1) == "0" or fast_ln.group(2) != "0":
            print("  WARNING: the fast LayerNorm's rows left the portable "
                  "kernel's statistics or the reciprocal's rounding")
            ok = False
    else:
        print("  WARNING: no fast_layer_norm check line -- the fast "
              "LayerNorm's arithmetic was not compared with portable's")
        ok = False

    # The table op (quant_lut.cpp): its quantize step against nearbyintf and
    # torch's own quantize_per_tensor on every tie and its neighbours, its
    # output bits, and one refusal per malformed argument.
    lut_check = re.search(
        r"^quant_lut check: (\d+) case\(s\), (\d+) mismatch\(es\).*$", log,
        re.M)
    if lut_check:
        print(lut_check.group(0))
        if lut_check.group(1) == "0" or lut_check.group(2) != "0":
            print("  WARNING: the table op's kernel disagreed with a reference")
            ok = False
    else:
        print("  WARNING: no quant_lut check line -- the table op's kernel was "
              "not checked")
        ok = False
    # The int16 table op (quant_lut16.cpp, its LX7 loops): every scheme on all
    # 65,536 codes, the quantize and dequantize steps on ties and the rails,
    # against mirrors.py's hashes, and one refusal per malformed argument.
    lut16_check = re.search(
        r"^lut16 check: (\d+) case\(s\), (\d+) mismatch\(es\).*$", log, re.M)
    if lut16_check:
        print(lut16_check.group(0))
        if lut16_check.group(1) == "0" or lut16_check.group(2) != "0":
            print("  WARNING: the int16 table op disagreed with its mirror")
            ok = False
    else:
        print("  WARNING: no lut16 check line -- the int16 table op was not "
              "checked")
        ok = False
    # The device's float division and sqrtf, and the portable reciprocal and
    # rsqrt (a double division, narrowed), against exact rounding on the
    # hardest-to-round inputs, subnormal ties and the edges
    # (divsqrt_golden.h). Every exact row that divides or takes a root rests on
    # them. Under QEMU this proves QEMU's model of the FPU instructions only:
    # on the chip (run_hardware runs this same gate) it proves the silicon.
    divsqrt = re.search(
        r"^divsqrt check: (\d+) case\(s\), (\d+) mismatch\(es\).*$", log, re.M)
    if divsqrt:
        print(divsqrt.group(0))
        if divsqrt.group(1) == "0" or divsqrt.group(2) != "0":
            print("  WARNING: a division or square root is not correctly "
                  "rounded")
            ok = False
    else:
        print("  WARNING: no divsqrt check line -- the device's division and "
              "square root were not checked")
        ok = False
    # The chip's expf, which portable softmax takes (prefill's attention),
    # against newlib_expf.h's transcription as mirrors.expf_newlib computes it
    # (expf_golden.h): the only check of that transcription against the code
    # it transcribes. Under QEMU it proves QEMU's run of the libm code; on the
    # chip (run_hardware runs this same gate) the silicon's.
    expf = re.search(
        r"^expf check: (\d+) case\(s\), (\d+) mismatch\(es\).*$", log, re.M)
    if expf:
        print(expf.group(0))
        for line in re.findall(r"^  MISMATCH expf: .*$", log, re.M):
            print(line)
        if expf.group(1) == "0" or expf.group(2) != "0":
            print("  WARNING: the chip's expf is not newlib_expf.h's "
                  "transcription (mirrors.expf_newlib)")
            ok = False
    else:
        print("  WARNING: no expf check line -- the chip's expf was not "
              "checked against its transcription")
        ok = False
    # ExecuTorch 1.6's rule for NaN and the infinities (runtime/quant_rules.h):
    # the affine records' quantisers, the backend's PT2E rows and the K/V
    # writer, through their own entry points, against the portable kernel's
    # transcription (which the host's --selftest holds to the kernel itself).
    nonfinite = re.search(
        r"^nonfinite check: (\d+) case\(s\), (\d+) mismatch\(es\).*$", log,
        re.M)
    if nonfinite:
        print(nonfinite.group(0))
        if nonfinite.group(1) == "0" or nonfinite.group(2) != "0":
            print("  WARNING: a quantiser broke the non-finite rule")
            ok = False
    else:
        print("  WARNING: no nonfinite check line -- the non-finite quantize "
              "rule was not checked")
        ok = False
    # Every table row must have run its tables (a row of plain copies would
    # fail its bits, but the count names the culprit), and none refused.
    for name in sorted(n for n in expected if n.startswith("luts_")):
        section = re.search(rf"^--- {re.escape(name)}\.pte.*?(?=^--- |\Z)",
                            log, flags=re.MULTILINE | re.DOTALL)
        count = section and re.search(r"quant_lut elements: (\d+)",
                                      section.group(0))
        gap = sum(lut_gaps.get(name, {}).values())
        print(f"{name}: quant_lut elements "
              f"{count.group(1) if count else 'never reported'}, "
              f"{gap} element(s) differ from eager (within the row's bar)")
        if not count or count.group(1) == "0":
            print(f"  WARNING: {name} ran no table")
            ok = False
    lut_dispatch = re.search(
        r"quant_lut dispatch: calls (\d+), elements (\d+), refusals (\d+)", log)
    if lut_dispatch:
        print(lut_dispatch.group(0))
        if lut_dispatch.group(1) == "0" or lut_dispatch.group(3) != "0":
            print("  WARNING: the table op never ran, or refused a program's call")
            ok = False
    else:
        print("  WARNING: no quant_lut dispatch line")
        ok = False
    # Stage B's code table (sb_siglut): its rows' bits alone would pass if the
    # rewrite had quietly kept the fp32 table.
    codes_dispatch = re.search(
        r"quant_lut_codes dispatch: calls (\d+), elements (\d+), refusals "
        r"(\d+)", log)
    if codes_dispatch:
        print(codes_dispatch.group(0))
        if codes_dispatch.group(1) == "0" or codes_dispatch.group(3) != "0":
            print("  WARNING: the code table never ran, or refused a call")
            ok = False
    else:
        print("  WARNING: no quant_lut_codes dispatch line")
        ok = False
    # The int16 table rows (q16_*, fixtures/lut16.py): their bits alone would
    # pass if the lowering had kept the portable chain.
    lut16_dispatch = re.search(
        r"lut16 dispatch: calls (\d+), elements (\d+), refusals (\d+)", log)
    if lut16_dispatch:
        print(lut16_dispatch.group(0))
        if (int(lut16_dispatch.group(1)) < len(lut16_gaps)
                or lut16_dispatch.group(3) != "0"):
            print("  WARNING: the int16 table op ran fewer calls than its rows, or "
                  "refused one")
            ok = False
    else:
        print("  WARNING: no lut16 dispatch line")
        ok = False
    # The CNN and RNN rows: the memory each found, beside what its export
    # measured (their bits are compared above; a row that ran must also say
    # how much heap its planned buffers had to come from, and a board-only
    # row must have been skipped for its reason).
    for name, row in sorted(cnn_rows.items()):
        section = re.search(rf"^--- {re.escape(name)}\.pte.*?(?=^--- |\Z)", log,
                            flags=re.MULTILINE | re.DOTALL)
        text = section.group(0) if section else ""
        if name in board_only:
            skipped = "SKIPPED: needs PSRAM" in text
            print(f"{name}: {row['summary']}; on {target}: "
                  f"{'skipped, needs PSRAM' if skipped else 'NOT skipped'}")
            if not skipped:
                print(f"  WARNING: {name} needs PSRAM, but {target} did not "
                      "skip it for that")
                ok = False
            continue
        block = re.search(r"largest free block: internal (\d+) B", text)
        planned = re.search(r"planned buffers: \d+, (\d+) B total", text)
        print(f"{name}: {row['summary']}; on {target}: largest internal block "
              f"{block.group(1) if block else 'never reported'} B, planned "
              f"{planned.group(1) if planned else '?'} B")
        if not block:
            print(f"  WARNING: {name} did not report its largest free block")
            ok = False
    # No convolution splits across the cores, in the min_latency pass or
    # anywhere (the backend README's Known limits): the split-pass twins run
    # their convs on one core, and the firmware's count must say so.
    if any(row.get("split_pass") for row in cnn_rows.values()):
        cs = re.search(r"^esp32 convsplit: (\d+) conv op", log, re.M)
        got = int(cs.group(1)) if cs else None
        print(f"esp32 convsplit: {got} conv op(s) split (none may)")
        if got != 0:
            print("  WARNING: a convolution split across the cores, or the count "
                  "was never printed")
            ok = False
    # And that the PIE depthwise kernel ran: the static int8 conv path checks
    # it once, at its first init, and on a FAIL every depthwise conv takes the
    # scalar loop with the same bits, so the rows above would pass without the
    # assembly having run a single channel.
    fast = re.search(r"^esp32 convfast: (\d+) conv op", log, re.M)
    if fast and int(fast.group(1)) > 0:
        pie = re.search(r"PIE depthwise 16-lane selftest: (\w+)", log)
        print("PIE depthwise 16-lane selftest: "
              f"{pie.group(1) if pie else 'never printed'}")
        if not pie or pie.group(1) != "PASS":
            print("  WARNING: the PIE depthwise kernel did not run: every depthwise "
                  "conv took the scalar loop")
            ok = False
        # The same for the dense columns' fused dot and the gathers' transpose,
        # which fall back to the dots-then-epilogue and the element copies.
        kernels = re.search(r"PIE conv kernels selftest: fused dot and epilogue "
                            r"(\w+).*?, transpose (\w+)", log)
        print("PIE conv kernels selftest: "
              + (f"fused dot and epilogue {kernels.group(1)}, transpose "
                 f"{kernels.group(2)}" if kernels else "never printed"))
        if not kernels or kernels.group(1) != "PASS" or kernels.group(2) != "PASS":
            print("  WARNING: a PIE conv kernel did not run: its convs took the "
                  "code it replaces")
            ok = False
        # The dense columns' long-row kernel (k_pad past 112), likewise.
        long_rows = re.search(r"PIE long-row dot selftest: (\w+)", log)
        print("PIE long-row dot selftest: "
              f"{long_rows.group(1) if long_rows else 'never printed'}")
        if not long_rows or long_rows.group(1) != "PASS":
            print("  WARNING: the PIE long-row dot did not run: its convs took the "
                  "dots then the epilogue")
            ok = False
        # And the depthwise plane kernels (5x5, and the 3x3 channels the
        # 16-channel path leaves), which fall back to the scalar loop.
        planes = re.search(r"PIE conv kernels selftest: .*?, depthwise planes "
                           r"(\w+)", log)
        print("PIE depthwise planes selftest: "
              + (planes.group(1) if planes else "never printed"))
        if not planes or planes.group(1) != "PASS":
            print("  WARNING: the PIE depthwise plane kernels did not run: their "
                  "convs took the scalar loop")
            ok = False

    # How its tables (and qnorm's int16 ones) were checked: once each, on the
    # device copied into internal RAM when they fit (a report, not a gate).
    lut16_tables = re.search(r"^lut16 tables: .*$", log, re.M)
    if lut16_tables:
        print(lut16_tables.group(0))
    for name, gap in lut16_gaps.items():
        flipped = (f" (inputs {gap['flipped_inputs']} x s1)" if gap["flips"] else "")
        print(f"  {name}: {gap['table']}; {gap['flips']} of {gap['inputs']} input(s) "
              f"flip against eager{flipped}")

    # The Glimmer decode rows' tails run as espressif::attn_tail; their checksum
    # rows alone would also pass if the fusion had silently stopped matching.
    attn_tail = re.search(
        r"attn_tail dispatch: (\d+) call\(s\), (\d+) row\(s\)", log)
    if attn_tail:
        print(f"attn_tail dispatch: {attn_tail.group(1)} call(s), "
              f"{attn_tail.group(2)} row(s)")
        if attn_tail.group(1) == "0":
            print("  WARNING: the fused attention tail never ran")
            ok = False
    else:
        print("  WARNING: no attn_tail dispatch line")
        ok = False

    fast_layernorm = re.search(
        r"native_layer_norm\.out fastmath: calls (\d+), rows (\d+), "
        r"fallback (\d+)", log)
    if args.fast_layernorm:
        if fast_layernorm:
            print(f"native_layer_norm.out fastmath: calls "
                  f"{fast_layernorm.group(1)}, rows {fast_layernorm.group(2)}, "
                  f"fallback {fast_layernorm.group(3)}")
            if (fast_layernorm.group(1) == "0" or
                    fast_layernorm.group(2) == "0" or
                    fast_layernorm.group(3) == "0"):
                print("  WARNING: the fast LayerNorm path or portable fallback "
                      "never ran")
                ok = False
        else:
            print("  WARNING: no fast LayerNorm dispatch line")
            ok = False
    elif fast_layernorm:
        print("  WARNING: exact build unexpectedly registered fast LayerNorm")
        ok = False

    # The split rows prove numerics; this proves the fork actually happened --
    # a worker that failed to spawn would leave the rows single-core and green.
    m = re.search(r"esp32 split: (\d+) column-split addmm row\(s\) forked", log)
    # None means "the line was absent", which is a different fact from zero and
    # must not read as evidence either way in the options guard below.
    forks = int(m.group(1)) if m else None
    if m:
        print(f"split forks: {forks}")
        if forks == 0:
            print("  WARNING: split rows ran but no op ever forked to the worker")
            ok = False

    # The selected-row head runs its own loop; counting it as the decode fast
    # path would make that counter's evidence unsound.
    sh = re.search(r"selected head: decode fast-path rows (\d+)", log)
    if sh:
        print(f"selected head: decode fast-path rows {sh.group(1)}")
        if sh.group(1) != "0":
            print("  WARNING: the selected-row head was counted as the decode "
                  "fast path")
            ok = False
    else:
        print("  WARNING: the selected-head counter check did not run")
        ok = False

    # Reentrancy: two handles live at once, configured by their own compile
    # specs with NO global override in play. Matching checksums (the two rows
    # above) prove the arithmetic; this proves the two handles actually behaved
    # DIFFERENTLY -- the min-latency one forked, the background one did not,
    # before and after it. Without this the rows would pass just as well if the
    # compile spec were ignored entirely and nothing ever split.
    m = re.search(
        r"forks: background handle (\d+), min-latency handle (\d+), "
        r"background again (\d+)", log)
    if m:
        bg1, ml, bg2 = (int(g) for g in m.groups())
        print(f"reentrancy forks: background {bg1}/{bg2}, min-latency {ml}")
        if ml == 0:
            print("  WARNING: the min-latency handle never forked -- its "
                  "compile spec was not honoured")
            ok = False
        if bg1 or bg2:
            print("  WARNING: a background-posture handle forked -- handles are "
                  "sharing state")
            ok = False

    # The split join must wait on the backend's own completion signal. Waiting
    # on the caller's task notification would let an application notification
    # end the join before the worker finished, and consume it. The fixture posts
    # one before a forking execute and reads it back.
    an = re.search(r"app notify: preserved (\w+)", log)
    if an:
        print(f"application notification across a split execute: preserved "
              f"{an.group(1)}")
        if an.group(1) != "yes":
            print("  WARNING: the split join consumed or was ended by the "
                  "caller's task notification")
            ok = False
    else:
        print("  WARNING: the split-join notification check did not run")
        ok = False

    # A min-latency caller on the worker's own core keeps its ops whole; a
    # fork there would serialise both halves on one core. Its checksum is an
    # ordinary row (mlp_ml_core1) held to the same reference.
    wc = re.search(r"worker-core caller: forks (\d+), finished (\w+)", log)
    if wc:
        print(f"worker-core caller: forks {wc.group(1)}, finished {wc.group(2)}")
        if wc.group(1) != "0" or wc.group(2) != "yes":
            print("  WARNING: a caller on the worker's core forked, or did not "
                  "finish")
            ok = False
    else:
        print("  WARNING: the worker-core caller check did not run")
        ok = False

    # The worker takes the priority of each caller that holds its lease, down
    # as well as up: raised only, one high-priority caller would leave every
    # later fork's half running above the application.
    wp = re.search(r"worker priority: base (\d+), after a priority-(\d+) "
                   r"caller (\d+), after a base caller (\d+)", log)
    if wp:
        base, high, after_high, after_base = (int(g) for g in wp.groups())
        print(f"worker priority: base {base}, after priority-{high} caller "
              f"{after_high}, after base caller {after_base}")
        if after_high != high or after_base != base:
            print("  WARNING: the split worker did not follow its caller's "
                  "priority")
            ok = False
    else:
        print("  WARNING: the worker priority check did not run")
        ok = False

    cc = re.search(r"concurrent: both finished (\w+), ok executes "
                   r"(\d+)\+(\d+) of (\d+)\+(\d+), forks (\d+)", log)
    if cc:
        print(f"concurrent execute: both finished {cc.group(1)}, ok executes "
              f"{cc.group(2)}+{cc.group(3)} of {cc.group(4)}+{cc.group(5)}, "
              f"forks {cc.group(6)}")
        if cc.group(1) != "yes":
            print("  WARNING: a concurrently-executing handle never completed")
            ok = False
        # Finishing is not succeeding: a racer that stopped at a failed
        # execute would report the warm-up's outputs, which match.
        if cc.group(2) != cc.group(4) or cc.group(3) != cc.group(5):
            print("  WARNING: a concurrent execute returned an error")
            ok = False
        if cc.group(6) == "0":
            print("  WARNING: nothing forked during the race -- it no longer "
                  "exercises the split")
            ok = False
    else:
        print("  WARNING: the concurrency check did not run")
        ok = False

    # The blob extension mechanism. Guarded on every outcome because the
    # useful one is the SKIP: a validator that refused everything unknown would
    # pass a "corrupt input rejected" test while making the format inextensible,
    # which is the failure this mechanism exists to prevent.
    eg = re.search(
        r"extguard: optional=(\w+) required=(\w+) retired=(\w+) oob=(\w+)",
        log,
    )
    if eg:
        print(f"extguard: unknown-optional {eg.group(1)}, "
              f"unknown-required {eg.group(2)}, retired-required {eg.group(3)}, "
              f"out-of-bounds {eg.group(4)}")
        if eg.group(1) != "skipped":
            print("  WARNING: an unknown OPTIONAL extension was refused -- the "
                  "format cannot be extended without a version bump")
            ok = False
        if (eg.group(2) != "refused" or eg.group(3) != "refused"
                or eg.group(4) != "refused"):
            print("  WARNING: an unknown, retired, or out-of-bounds extension "
                  "was ACCEPTED")
            ok = False
    else:
        print("  WARNING: no extguard line -- the extension validator was not "
              "exercised")
        ok = False

    # A valid mixed-scheme branch: two of three projections use PT2E's one
    # per-tensor scale. The checksum proves the scheme; these deltas prove the
    # compatible pair still shared its four input rows while the per-row sibling
    # stayed out of the group.
    ash = re.search(r"actshare: shared_rows=(\d+) per_tensor_ops=(\d+)", log)
    if ash:
        print(f"actshare: {ash.group(1)} shared row quantisations avoided, "
              f"{ash.group(2)} per-tensor ops")
        if ash.groups() != ("4", "2"):
            print("  WARNING: mixed activation-scale grouping did not preserve "
                  "the exact 2+1 scheme")
            ok = False
    else:
        print("  WARNING: no actshare line -- mixed activation-scale grouping "
              "was not exercised")
        ok = False

    # RoPE fusion, checked for having RUN rather than for having been applied.
    # The AOT half can fuse every site and the device still fall back to eight
    # portable ops if the custom kernel is not registered -- the checksums would
    # stay green either way, because both compute the same thing. Only this line
    # distinguishes "fused" from "fused and used".
    lp = re.search(r"ln2d: (\d+) channel column\(s\) normalised", log)
    if lp:
        n_ln = int(lp.group(1))
        print(f"ln2d: {n_ln} channel column(s) through espressif::layer_norm2d")
        if n_ln == 0:
            print("  WARNING: the layer_norm2d kernel registered but never ran "
                  "-- the fusion did not reach the device")
    else:
        print("  WARNING: no ln2d line -- the fused kernel was not exercised")
    # Its refusals, driven directly with hand-built tensors: a refusal that
    # merely returned would report success over an unwritten output, and an
    # fp16 tensor must not be read and written as fp32.
    lr = re.search(r"ln2d refusal: control=(\w+) fp16=(\w+) rank3=(\w+)", log)
    if lr:
        print(f"ln2d refusal: fp32 control {lr.group(1)}, fp16 {lr.group(2)}, "
              f"3-D {lr.group(3)}")
        if lr.groups() != ("accepted", "refused", "refused"):
            print("  WARNING: layer_norm2d accepted a tensor it cannot compute, "
                  "or refused its control")
            ok = False
    else:
        print("  WARNING: no ln2d refusal line -- the kernel's checks were not "
              "exercised")
        ok = False
    rp = re.search(r"rope: (\d+) row\(s\) rotated", log)
    if rp:
        print(f"rope: {rp.group(1)} row(s) through espressif::rope")
        if rp.group(1) == "0":
            print("  WARNING: the rope kernel registered but never ran -- the "
                  "fusion did not reach any executed model")
            ok = False
    else:
        print("  WARNING: no rope line -- the fused kernel was not exercised")
        ok = False

    # Backend options: ExecuTorch's runtime reconfiguration API, driven by
    # backend name. Guarded on the whole contract rather than on "it did not
    # crash", because the two ways this can rot are both silent -- an option that
    # round-trips but never reaches execute(), and an unknown key accepted so a
    # user believes min-latency is on when it is not. The three split rows above
    # are the other half: they are enabled THROUGH set_option, so a no-op
    # implementation would pass the round trip and show zero forks here.
    opt = re.search(r"options roundtrip: (\w+)", log)
    if opt:
        print(f"backend options: set/get roundtrip {opt.group(1)}")
        if opt.group(1) != "PASS":
            for line in log.splitlines():
                if "backend options" in line or "min_latency" in line \
                        or "rejected:" in line or "get_option" in line:
                    print(f"    {line.strip()}")
            print("  WARNING: the set_option/get_option contract failed")
            ok = False
        elif forks == 0:
            # Round trip green, nothing forked: the option is stored and read
            # back but does not reach the execute path. Exactly the failure the
            # round trip alone cannot see.
            print("  WARNING: options roundtrip passed but NOTHING forked -- "
                  "set_option is not reaching execute()")
            ok = False
    else:
        print("  WARNING: no backend-options line -- set_option/get_option was "
              "not exercised")
        ok = False

    # The blob validator, checked with invalid metadata in a real artifact: a validator
    # that has only ever seen valid input proves nothing, and this one cannot be
    # reached from the AOT side. The control matters as much as the negatives --
    # if the RAM copy stopped loading for an unrelated reason, both negatives
    # would "pass" while proving nothing.
    bg = re.search(
        r"blobguard: control=(\w+) kpad=(\w+) nops=(\w+) "
        r"bmmcontrol=(\w+) bmmbatch=(\w+) convcontrol=(\w+) "
        r"convstride=(\w+) convduplicate=(\w+) groupcontrol=(\w+) "
        r"groupduplicate=(\w+) i16rowcontrol=(\w+) i16row=(\w+) "
        r"selectedcontrol=(\w+) selectedzero=(\w+) "
        r"selectedduplicate=(\w+) scalealign=(\w+) biasalign=(\w+) "
        r"groupcountcontrol=(\w+) groupcount=(\w+) alias=(\w+) "
        r"dotlen=(\w+)",
        log,
    )
    if bg:
        print(f"blobguard: control {bg.group(1)}, k>k_pad {bg.group(2)}, "
              f"n_ops overflow {bg.group(3)}, BMM control {bg.group(4)}, "
              f"BMM batch overflow {bg.group(5)}, conv control {bg.group(6)}, "
              f"conv zero-stride {bg.group(7)}, duplicate geometry {bg.group(8)}, "
              f"group control {bg.group(9)}, duplicate group {bg.group(10)}, "
              f"i16 row control/value {bg.group(11)}/{bg.group(12)}, "
              f"selected control/zero/duplicate "
              f"{bg.group(13)}/{bg.group(14)}/{bg.group(15)}, "
              f"misaligned scales/bias {bg.group(16)}/{bg.group(17)}, "
              f"group count control/k-vs-k_pad "
              f"{bg.group(18)}/{bg.group(19)}, "
              f"output aliases input {bg.group(20)}, "
              f"int8 dot past int32 {bg.group(21)}")
        if any(bg.group(i) != "accepted" for i in (1, 4, 6, 9, 11, 13, 18)):
            print("  WARNING: an UNMODIFIED control was rejected -- the "
                  "negatives below prove nothing")
            ok = False
        if any(bg.group(i) != "rejected"
               for i in (2, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17, 19, 20,
                         21)):
            print("  WARNING: invalid model metadata was accepted")
            ok = False
        # A rejection only covers the guard it was written for if that guard
        # is the one that fired: a k > k_pad row that patched n instead would be
        # refused by the chain-shape check, so require its own message.
        if not re.search(r"op 0 has k \d+ > k_pad 32\b", log):
            print("  WARNING: the k > k_pad row was not refused by the "
                  "k > k_pad guard")
            ok = False
    else:
        print("  WARNING: no blobguard line -- the validator was not exercised")
        ok = False

    # Every cached execution is a parity verdict. The i8 decode values are
    # explicitly conditioned on the prefill method's cache, across destruction
    # and reload of the ET Method. Timings are printed, but QEMU's are never
    # treated as ESP32-S3 performance evidence.
    print(f"\nCached-sequence steps are held to TOL {TOL:g}: {CACHED}.")
    ok &= compare_cached(
        log, "glimmer_multi_prefill",
        {"checksums": glimmer_multimethod_ref["prefill_checksums"]})
    ok &= compare_cached(
        log, "glimmer_multi_decode",
        {"checksums": glimmer_multimethod_ref["decode_checksums"]})
    shared = re.search(
        r"glimmer multimethod shared KV: (\d+) B, sequential reload: YES", log)
    if shared:
        shared_bytes = int(shared.group(1))
        print(f"glimmer multimethod: shared KV arena {shared_bytes:,} B; "
              "method reload preserved state")
        if shared_bytes != glimmer_multimethod_ref["cache_bytes"]:
            print("  WARNING: device shared-arena size differs from export")
            ok = False
    else:
        print("  WARNING: multimethod runner did not prove a shared KV reload")
        ok = False

    # The same fixture with a persistent int8 K cache. Its host oracle was
    # proven equal to the fp32-cache oracle at export; on the device the claim
    # is stronger still -- every logit bit of every step equals the fp32-cache
    # program's, because both dot the same int8 key codes.
    kc8_ref = glimmer_multimethod_ref["kc8"]
    ok &= compare_cached(
        log, "glimmer_kc8_prefill", {"checksums": kc8_ref["prefill_checksums"]})
    ok &= compare_cached(
        log, "glimmer_kc8_decode", {"checksums": kc8_ref["decode_checksums"]})
    kc8_shared = re.search(
        r"glimmer multimethod kc8 shared KV: (\d+) B, sequential reload: YES",
        log)
    if kc8_shared:
        kc8_bytes = int(kc8_shared.group(1))
        print(f"glimmer kc8 multimethod: shared KV arena {kc8_bytes:,} B "
              f"(fp32 K cache {kc8_ref['fp32_cache_bytes']:,} B)")
        if kc8_bytes != kc8_ref["cache_bytes"]:
            print("  WARNING: int8-K-cache arena size differs from export")
            ok = False
    else:
        print("  WARNING: int8-K-cache multimethod did not prove a shared KV "
              "reload")
        ok = False
    for phase in ("prefill", "decode"):
        fp32_bits = step_bits(log, f"glimmer_multi_{phase}")
        kc8_bits = step_bits(log, f"glimmer_kc8_{phase}")
        if not fp32_bits or fp32_bits != kc8_bits:
            print(f"  WARNING: glimmer_kc8_{phase} output bits {kc8_bits} differ "
                  f"from the fp32 K cache's {fp32_bits}")
            ok = False
        else:
            print(f"glimmer_kc8_{phase}: all {len(kc8_bits)} step(s) bit-identical "
                  "to the fp32 K cache")
    ok &= compare_cached(log, "decode_i4", decode_refs["decode_i4"])

    # The fused RMSNorm. What QEMU can prove is that the AOT fusion happened, that the
    # emitted program looks up espressif::rmsnorm.out, and that the shim computes the
    # right numbers -- so the check is that the op *dispatched at all*, in whichever
    # path this build selected. It cannot prove the assembly: QEMU has no rsqrt0.s.
    # Rows outside the kernel's domain (mean(x^2) + eps not a positive normal
    # float) are refused before rsqrt0.s and computed by the scalar reference.
    # The refinement would turn them into NaN; the refusal is QEMU-reachable.
    rd = re.search(r"rmsnorm domain: refused (\w+), untouched (\w+), "
                   r"scalar fallback (\w+)", log)
    if rd:
        print(f"rmsnorm domain: refused {rd.group(1)}, output untouched "
              f"{rd.group(2)}, scalar fallback {rd.group(3)}")
        if rd.groups() != ("yes", "yes", "yes"):
            print("  WARNING: the rmsnorm kernel computed a row outside its "
                  "domain, or the fallback differs from the scalar reference")
            ok = False
    else:
        print("  WARNING: the rmsnorm domain check did not run")
        ok = False
    path = re.search(r"rmsnorm path: (\w+)", log)
    m = re.search(r"rmsnorm dispatch: asm (\d+), scalar fallback (\d+)", log)
    if m and path:
        asm, scalar = int(m.group(1)), int(m.group(2))
        print(f"rmsnorm dispatch: asm {asm}, scalar {scalar}  "
              f"[compiled path: {path.group(1)}]")
        if asm + scalar == 0:
            print("  WARNING: fused rmsnorm never dispatched; is the pass wired in?")
            ok = False
        if path.group(1) == "SCALAR" and asm:
            print("  WARNING: asm ran in a build that disabled it")
            ok = False
        # Not a failure, but never let a QEMU pass be read as validating the kernel.
        if path.group(1) == "SCALAR":
            print("  NOTE: rsqrt0.s is unreachable under QEMU; the assembly kernel is "
                  "validated on the host (test_rmsnorm.py) and on the board")
    else:
        print("  WARNING: rmsnorm never reported")
        ok = False

    # Quantised embedding. Worth its own guard because the failure mode is silent and
    # expensive: if the partitioner stops claiming it, everything still passes and the
    # table quietly reverts to fp32 -- which is 100.7 MB at the deployed config against
    # a 16 MB flash budget, i.e. the difference between the real model fitting and not.
    e = re.search(r"esp32 embed: (\d+) gather op", log)
    if e:
        print(f"esp32 embed: {e.group(1)} gather op(s)")
        if e.group(1) == "0":
            print("  WARNING: no embedding reached the delegate; table would be fp32")
            ok = False
    else:
        print("  WARNING: embedding never reported")
        ok = False

    # The persistent int8 K cache. Both of its counters must move -- the write
    # side (espressif::quantize_rows_i8) and the read side (OP_BMM_KCACHE_I8) --
    # and the delegated attention core must reproduce attnqk_qk's output to
    # the bit, since it dots the very codes attnqk_qk's bmm makes per call.
    kc = re.search(r"esp32 bmmkc8: (\d+) int8-K-cache bmm op\(s\), (\d+) "
                   r"cached key row", log)
    qr = re.search(r"quantize_rows_i8: (\d+) call\(s\), (\d+) row\(s\)", log)
    if kc and qr:
        print(f"esp32 bmmkc8: {kc.group(1)} int8-K-cache bmm op(s), "
              f"{int(kc.group(2)):,} cached key rows read without re-quantising; "
              f"quantize_rows_i8: {qr.group(1)} call(s), {qr.group(2)} row(s)")
        if kc.group(1) == "0" or qr.group(1) == "0":
            print("  WARNING: the int8 K cache's read or write side never ran")
            ok = False
    else:
        print("  WARNING: int8 K cache counters never reported")
        ok = False
    qk_bits, kc8_bits = model_bits(log, "attnqk_qk"), model_bits(log, "attnkc8_qk")
    if qk_bits is None or qk_bits != kc8_bits:
        print(f"  WARNING: attnkc8_qk output bits {kc8_bits} differ from "
              f"attnqk_qk's {qk_bits}")
        ok = False
    else:
        print(f"attnkc8_qk: output bit-identical to attnqk_qk ({qk_bits})")
    kg = re.search(r"kcguard: control=(\w+) scalearg=(\w+) storedoff=(\w+) "
                   r"unknownop=(\w+)", log)
    if kg:
        print(f"kcguard: control {kg.group(1)}, scales in a chain slot "
              f"{kg.group(2)}, stored offset {kg.group(3)}, unknown opcode "
              f"{kg.group(4)}")
        if kg.group(1) != "accepted" or any(
                kg.group(i) != "rejected" for i in (2, 3, 4)):
            print("  WARNING: the int8-K-cache descriptor guard failed")
            ok = False
    else:
        print("  WARNING: no kcguard line -- the kcache descriptor validator "
              "was not exercised")
        ok = False

    # Asymmetric quantisation: its rules cover more than any
    # exporter emits, so each is proven on a synthetic blob
    # (asym_blob_cases.py) that must be refused for THAT rule -- the device
    # reads it back from esp32_asym::last_refusal(). The valid blobs the
    # runtime computes must load, the other valid ones be refused only as
    # "unimplemented", and the carrier program the cases ride in, a symmetric
    # blob, must still load.
    from executorch.backends.esp32s3.test.fixtures import asym_blob_cases
    ag = re.search(r"asymguard: cases=(\d+) read=(\d+) wrong=(\d+) "
                   r"accepted=(\d+) computed=(\d+) control=(\w+)", log)
    if ag:
        n, read, wrong, accepted, computed = (int(ag.group(i))
                                              for i in range(1, 6))
        want = len(asym_blob_cases.accepted_cases())
        print(f"asymguard: {read - wrong}/{n} v24 cases as expected "
              f"({accepted} loaded, {want} computed by design); carrier "
              f"{ag.group(6)}")
        if (n != len(asym_blob_cases.CASES) or read != n or wrong
                or accepted != want or computed != want
                or ag.group(6) != "accepted"):
            for line in log.splitlines():
                if "asymcase WRONG" in line:
                    print(f"    {line.strip()}")
            print("  WARNING: a v24 rule refused the wrong case, accepted "
                  "one, or the symmetric carrier no longer loads")
            ok = False
    else:
        print("  WARNING: no asymguard line -- blob v24's rules were not "
              "exercised")
        ok = False
    # The weight-affine rows' own proof that they ran the affine loops: the
    # op counter after each pass, and forks on the two rows that split (the
    # min_latency row, and the tied head's head split).
    affine = [int(v) for v in re.findall(
        r"esp32 affine: (\d+) weight-affine op\(s\) executed", log)]
    print(f"esp32 affine: weight-affine ops after each pass {affine}")
    if len(affine) != 2 or not 0 < affine[0] < affine[1]:
        print("  WARNING: the weight-affine rows did not run their affine "
              "loops (or a pass is missing)")
        ok = False
    for row in ("asym_i4_split", "asym_head", "asym_a3_i4_split",
                "asym_a3_head"):
        section = re.search(rf"^--- {row}\.pte.*?(?=^--- |\Z)", log,
                            flags=re.MULTILINE | re.DOTALL)
        forks = re.search(r"asym forks: (\d+)", section.group(0)) \
            if section else None
        print(f"{row}: {forks.group(1) if forks else 'no'} affine fork(s)")
        if not forks or forks.group(1) == "0":
            print(f"  WARNING: {row} never forked to the split worker")
            ok = False
    kz = re.search(r"asym kcache zp check: control=(\w+) dtype=(\w+) "
                   r"shape=(\w+)", log)
    if kz:
        print(f"asym kcache zp check: int8 control {kz.group(1)}, wrong dtype "
              f"{kz.group(2)}, wrong shape {kz.group(3)}")
        if kz.groups() != ("accepted", "refused", "refused"):
            print("  WARNING: the affine K cache's zero-point check failed")
            ok = False
    else:
        print("  WARNING: no asym kcache zp check line")
        ok = False
    # ExecuTorch's own LLM recipes: the stock quantised K/V cache and
    # embeddings, every step of every row.
    ok &= compare_stock_llm(log, stock_refs)
    # The int16 loop's int64 dot and its float conversion (libgcc's
    # __floatdisf here) against an integer-only reference: dots on exact
    # float32 ties and past int32.
    ac = re.search(r"asym a16 check: conversions=(\d+)/(\d+) wide=(\d+)/(\d+)",
                   log)
    if ac:
        print(f"asym a16 check: {ac.group(1)}/{ac.group(2)} tie conversions, "
              f"{ac.group(3)}/{ac.group(4)} dots past int32 exact")
        if ac.group(1) != ac.group(2) or ac.group(3) != ac.group(4):
            print("  WARNING: the int16 loop's dot or conversion is not exact")
            ok = False
    else:
        print("  WARNING: no asym a16 check line")
        ok = False

    # Opt-in int8 q@kT: attnqk_qk carries exactly one OP_BMM_DYN_I8, so zero means
    # the partitioner stopped claiming it -- the row would then run portable fp32
    # and FAIL its int8-mirror reference, but the counter names the culprit.
    bq = re.search(r"esp32 bmmqk: (\d+) dynamic int8 bmm op", log)
    if bq:
        print(f"esp32 bmmqk: {bq.group(1)} dynamic int8 bmm op(s)")
        if bq.group(1) == "0":
            print("  WARNING: no dynamic bmm executed; q@kT was not claimed")
            ok = False
    else:
        print("  WARNING: bmmqk never reported")
        ok = False

    # Static activation scales. mlp_static is exported with FLAG_STATIC_ACT on every
    # matmul, so zero static rows means the runtime fell back to dynamic absmax --
    # and on ops whose input matches calibration that fallback produces the SAME
    # numbers, which is exactly why a checksum alone cannot police this flag.
    sa = re.search(r"esp32 actstatic: (\d+) rows quantised at calibrated scales", log)
    if sa:
        print(f"esp32 actstatic: {sa.group(1)} rows quantised at calibrated scales")
        if sa.group(1) == "0":
            print("  WARNING: no row used a static scale; mlp_static fell back")
            ok = False
    else:
        print("  WARNING: actstatic never reported")
        ok = False

    # Activation-quant sharing. mlps' two heads and the SwiGLU gate/up pairs share an
    # input by construction, so zero avoided rows means the grouping silently stopped
    # forming -- everything would still pass, just re-quantising the same rows again.
    # Row blocking, the one performance claim provable without a board: bytes
    # moved, counted exactly. Absence of the line is a FAILURE, not a pass --
    # the report only stays silent when no quantised matmul ran at all, which
    # never happens in this suite. (Otherwise a changed message format would
    # disarm this guard silently: the regex would stop matching.)
    rb = re.search(r"esp32 rowblock: (\d+) weight passes vs (\d+) row-at-a-time "
                   r"\((\d+) avoided\); weight bytes (\d+) vs (\d+) "
                   r"\((\d+)% less\)", log)
    if rb:
        print(f"rowblock: {rb.group(1)} weight passes vs {rb.group(2)} "
              f"row-at-a-time; weight bytes {int(rb.group(4)):,} vs "
              f"{int(rb.group(5)):,} ({rb.group(6)}% less)")
        if int(rb.group(3)) == 0:
            print("  WARNING: row blocking never fired -- every matmul still "
                  "re-read its weights once per output row")
            ok = False
    else:
        print("  WARNING: no 'esp32 rowblock:' line in the device log -- the "
              "blocking report is missing or its format changed, so the guard "
              "proved nothing")
        ok = False

    a = re.search(r"esp32 actq: (\d+) shared-input row quantisations avoided", log)
    if a:
        print(f"esp32 actq: {a.group(1)} shared-input row quantisations avoided")
        if a.group(1) == "0":
            print("  WARNING: no shared-input group engaged across the zoo")
            ok = False
    else:
        print("  WARNING: actq never reported")
        ok = False

    b = re.search(
        r"esp32 backend: registered (-?\d+), delegate calls (\d+), addmm ops (\d+), "
        r"columns PIE (\d+) / scalar (\d+), weights external (\d+) / inline (\d+)",
        log,
    )
    if b:
        print(f"esp32 backend: delegate calls {b.group(2)}, addmm ops {b.group(3)}, "
              f"columns PIE {b.group(4)} / scalar {b.group(5)}, "
              f"weights external {b.group(6)} / inline {b.group(7)}")
        if b.group(6) == "0":
            print("  WARNING: no weights came from the NamedDataMap")
            ok = False
        if b.group(1) != "0":
            print(f"  WARNING: register_backend returned {b.group(1)}")
            ok = False
        if b.group(3) == "0":
            print("  WARNING: no addmm reached the delegate")
            ok = False
        # Scalar columns are expected: one run deliberately disables PIE so the
        # two paths can be compared. Only their absence, or PIE's, is a problem.
        if b.group(4) == "0":
            print("  WARNING: PIE never ran for any delegated column")
            ok = False
        if b.group(5) == "0":
            print("  WARNING: the forced-scalar comparison run did not happen")
            ok = False
    else:
        print("  WARNING: backend never reported; delegate may not have run")
        ok = False

    skipped = (f" ({len(board_only)} skipped on {target}: they need PSRAM)"
               if board_only else "")
    print(f"\nexact rows: {exact_ok}/{len(set(exact_bits) - board_only)} with the "
          f"device mirror's output bits{skipped}; code rows: {code_ok}/{len(codes)} "
          f"within one code; "
          f"tolerance rows: {tol_ok}/{len(set(reasons) - set(codes))} within "
          f"{TOL:g}")
    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
