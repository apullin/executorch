# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph passes run before partitioning, and the lowering entry point."""

import contextlib
import gc
from typing import Optional

from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.passes.constant_prop_pass import constant_prop_pass

from executorch.backends.esp32s3._passes.canonicalize_clamps_pass import (
    canonicalize_clamps,
)
from executorch.backends.esp32s3._passes.fold_conv_bn_pass import fold_conv_bn
from executorch.backends.esp32s3._passes.fuse_rmsnorm_pass import fuse_rmsnorm


def _cf(graph_module) -> int:
    return sum(1 for n in graph_module.graph.nodes if n.op == "call_function")


def _skips_dim_order(edge_manager) -> bool:
    """constant_prop_pass's _skip_dim_order for this manager's programs.

    The pass puts a clone in front of each constant it leaves as a graph
    output (a parameter returned as it is), and its default, True, makes
    that an aten.clone, which the edge verifier refuses in a program with
    dim order -- the default program, which to_backend validates before it
    partitions (a SpecViolationError naming aten.clone.default). The
    manager's compile config says which: the caller's in lower_esp32, and
    after to_backend the default it builds the new manager with."""
    return edge_manager.compile_config._skip_dim_order


@contextlib.contextmanager
def fewer_gc_passes(threshold=(10000, 50, 100)):
    """Run CPython's cyclic garbage collector less often while lowering.

    Lowering builds and drops whole FX graphs -- thousands of nodes, each a
    knot of dicts, meta and FakeTensors -- so at the default thresholds
    (700, 10, 10) the collector makes ~3,500 young passes and 9 full ones (each
    walking every live object) per story Glimmer export: 1.4 s. These
    thresholds collect the same garbage in a few hundred young passes, for
    ~4% more peak memory. Scheduling only: nothing lowered changes. Works as
    a decorator or a with-block, and restores the caller's thresholds.
    """
    previous = gc.get_threshold()
    gc.set_threshold(*threshold)
    try:
        yield
    finally:
        gc.set_threshold(*previous)


@fewer_gc_passes()
def apply_passes(
    edge_manager,
    method_name: str = "forward",
    rmsnorm_expect: Optional[int] = None,
    fuse: bool = True,
    fold: bool = True,
    quiet: bool = False,
    attn_tail_expect: Optional[int] = None,
):
    """Run the standard pass pipeline on an EdgeProgramManager, in place.

    `rmsnorm_expect` is forwarded to the fusion pass, which raises if the site count
    differs -- pass it wherever the number is known (the PLE model has 3*n_layers + 2)
    so a silently-unfused graph cannot reach the device looking healthy.
    `attn_tail_expect` does the same for the decode attention tail (one per layer).

    `fuse`/`fold` exist so a caller can measure one pass against the other. Both default
    on; an export that turns one off should say why.

    Returns a dict of counts, also suitable for printing via `format_report`.
    """
    ep = edge_manager.exported_program(method_name)
    # First: a dynamic recipe's activation chains still in the graph, which
    # only static_act's two-step lowering leaves (quantizer/calibration.py:
    # it partitions the to_edge graph, without lower_esp32's strip), computed
    # in place by float32 portable ops, and torchao's quantizes and
    # dequantizes over constants folded (compute_left_chains). Nothing
    # changes in a program lower_esp32 lowered.
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        compute_left_chains,
    )

    left_chains = compute_left_chains(ep)
    if any(n.op == "call_function" and "quantized_decomposed" in str(n.target)
           for n in ep.graph_module.graph.nodes):
        # The .out variants to_executorch emits a quantize or dequantize left
        # outside the delegates with, as in lower_esp32.
        import executorch.kernels.quantized  # noqa: F401
    before = _cf(ep.graph_module)

    # Shared graph hygiene from executorch.backends.transforms: identity clones go,
    # view_copy chains collapse, no-op views vanish. Stated plainly so the benefit
    # is not overclaimed: every delegated exporter partitions BEFORE calling this,
    # so what gets cleaned is the PORTABLE remainder -- fewer nodes, fewer kernels
    # registered, smaller .pte. A clone sitting inside a fusion pattern still hides
    # it from the partitioner's gates; pre-partition cleanup is a separate step
    # that none of the current models need, measured by their censuses.
    #
    # Keep this to the two transforms validated across the complete device gate:
    # adopting more of the wheel's transform family is an optimization change.
    # Routed through EdgeProgramManager.transform rather than applied to the
    # graph_module in place: FuseViewCopyTransform retraces via ExportPass.call
    # when it fires, and only the manager's _transform reconciles the graph
    # signature afterwards. Retracing a delegate-bearing graph is bit-exact (the
    # output is identical before and after the transform; a view chain goes
    # 2 -> 0).
    #
    # RemoveCloneOpsTransform options, both deliberate: input->output copies are
    # preserved because erasing one rewires the output node away from the name the
    # signature records, and q/dq elimination is off because q/dq handling here is
    # strip_dynamic_act_qdq's job, done by policy rather than as a clone-removal
    # side effect.
    from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
    from executorch.backends.transforms.remove_clone_ops import (
        RemoveCloneOpsTransform,
    )

    cleaned_mgr = edge_manager.transform([
        RemoveCloneOpsTransform(
            preserve_input_output_copies=True,
            eliminate_quant_dequant_pairs=False,
        ),
        FuseViewCopyTransform(),
    ])
    edge_manager._edge_programs.update(cleaned_mgr._edge_programs)
    ep = edge_manager.exported_program(method_name)
    # FuseViewCopyTransform collapses a reshape CHAIN into one view but leaves
    # that one in place even when it reshapes a tensor to its own shape --
    # `y.reshape(2,4,24).reshape(8,24)` on an [8,24] y survives as an identity
    # view_copy on the delegate output. It costs nothing at runtime (a view_copy
    # becomes a memory-planning alias), but "no-op views vanish" is the stated
    # claim of this cleanup and test_fusions holds it. Same safety rule as
    # RemoveIdentityExpand: only when the input is contiguous by construction,
    # and never an input->output alias, which the clone pass also preserves.
    _remove_identity_views(ep)
    # A table's trailing q -> dq pair (fold_quant_lut_pass.py): an identity on
    # every entry the table holds, so it goes, and a static graph between
    # delegates needs no quantize kernel for it. After partitioning, which
    # decides what consumes it.
    lut_pairs = 0
    if any(n.op == "call_function" and "quant_lut" in str(n.target)
           for n in ep.graph_module.graph.nodes):
        from executorch.backends.esp32s3._passes.fold_quant_lut_pass import elide_lut_pairs

        lut_pairs = elide_lut_pairs(ep)
    # The same for the int16 table (fold_lut16_pass.py): its output is
    # always on q2's grid.
    if any(n.op == "call_function" and "espressif.lut16" in str(n.target)
           for n in ep.graph_module.graph.nodes):
        from executorch.backends.esp32s3._passes.fold_lut16_pass import elide_lut16_pairs

        lut_pairs += elide_lut16_pairs(ep)
    after_shared = _cf(ep.graph_module)

    fused = 0
    rejected = []
    rope_fused, rope_rejected = 0, []
    ln2d_fused = 0
    rms_add_fused = 0
    attn_fused, attn_rejected = 0, []
    if fuse:
        p = fuse_rmsnorm(edge_manager, method_name, expect=rmsnorm_expect)
        fused, rejected = p.fused, p.rejected
        # RoPE, in the same place and for the same reason: eight edge ops per
        # application, each writing a full-width temporary. Measured at 34% of
        # a speech encoder's portable output traffic before fusing, and it
        # matches BOTH published spellings -- HF's rotate-half and the story
        # model's pairwise cat -- because which one a model author wrote is not
        # something the backend should get to care about.
        from executorch.backends.esp32s3._passes.fuse_rope_pass import fuse_rope

        rp = fuse_rope(edge_manager, method_name)
        rope_fused, rope_rejected = rp.fused, rp.rejected

        # LayerNorm2d, same place and same reason: fourteen portable ops, each
        # reading and writing a full [B,C,H,W] temporary, replaced by one op that
        # makes two passes. SAM2 and ConvNeXt both write it.
        from executorch.backends.esp32s3._passes.fuse_layer_norm2d_pass import (
            fuse_layer_norm2d,
        )

        lp = fuse_layer_norm2d(edge_manager, method_name)
        ln2d_fused = lp.fused

        # Post-norm residual adds (x + post_norm(sublayer)): the add folds into
        # the norm it follows, bit-identically. After fuse_rmsnorm, which makes
        # the norms it matches.
        from executorch.backends.esp32s3._passes.fuse_rmsnorm_add_pass import fuse_rmsnorm_add

        rms_add_fused = fuse_rmsnorm_add(edge_manager, method_name)
        # The decode attention tail (scale, mask, softmax, P@V), same place and
        # a different reason: four portable calls per layer whose dispatch,
        # newlib expf and per-lane software divide cost the story model
        # 19.6 ms/token on the chip, replaced by one call. Before constant_prop
        # for the RMSNorm reason: the scale is read from the lifted constants.
        from executorch.backends.esp32s3._passes.fuse_attn_tail_pass import fuse_attn_tail

        ap = fuse_attn_tail(edge_manager, method_name, expect=attn_tail_expect)
        attn_fused, attn_rejected = ap.fused, ap.rejected
    after_fuse = _cf(ep.graph_module)

    # Conv+bn folding inserts arithmetic for constant propagation to
    # materialise, so it is part of folding: with fold=False the inserted ops
    # would stay behind as invalid IR.
    convbn, convbn_rej = (fold_conv_bn(edge_manager, method_name) if fold
                          else (0, []))
    clamps = canonicalize_clamps(edge_manager, method_name)
    ep = edge_manager.exported_program(method_name)
    # A dequantize left portable with qparam dtypes ExecuTorch's kernel
    # aborts on -- a per-channel weight dequant's int32 zero points, a
    # per-tensor dequantize's float32 scale or int32 zero point in a tensor --
    # widened to int64 and float64 (exact), folded by the constant
    # propagation below where they are constants.
    from executorch.backends.esp32s3._passes.static_qdq import widen_portable_zero_points

    widen_portable_zero_points(ep)

    if fold:
        folded_ep = constant_prop_pass(ep, _skip_dim_order=_skips_dim_order(
            edge_manager))
        if folded_ep is not ep:
            # The installed wheel mutates in place and returns its argument, so the
            # manager sees the change for free. A version that returned a copy instead
            # would look exactly like "constant folding does nothing" -- the failure
            # would be invisible, not loud -- so put the result back explicitly rather
            # than relying on the aliasing.
            edge_manager._edge_programs[method_name] = folded_ep
            ep = folded_ep
    after = _cf(ep.graph_module)
    # A static quantize left to ExecuTorch's kernel (static_qdq.py's safety
    # net) whose finite scale is not a float32 value is refused by name: the kernel
    # multiplies by 1.0f/(float)s, torch by fl32(fl64(1/s)). After the table
    # pairs' elision, which removes some.
    from executorch.backends.esp32s3._passes.static_qdq import refuse_portable_scales

    static_q = refuse_portable_scales(ep)

    rep = {
        "before": before,
        "after_shared": after_shared,
        "after_fuse": after_fuse,
        "after": after,
        "fused": fused,
        "rope_fused": rope_fused,
        "ln2d_fused": ln2d_fused,
        "rms_add_fused": rms_add_fused,
        "attn_tail_fused": attn_fused,
        "attn_tail_rejected": attn_rejected,
        "rope_rejected": rope_rejected,
        "rejected": rejected,
        "convbn": convbn,
        "convbn_rejected": convbn_rej,
        "clamps": clamps,
        "lut_pairs": lut_pairs,
        "static_q": static_q,
        "left_chains": left_chains,
        "by_shared": before - after_shared,
        "by_fusion": after_shared - after_fuse,
        "by_folding": after_fuse - after,
    }
    if not quiet:
        print("  " + format_report(rep))
    return rep


def _remove_identity_views(ep) -> int:
    """Delete `view_copy` nodes whose output shape equals their input shape."""
    import torch
    from executorch.exir.dialects._ops import ops as exir_ops

    from executorch.backends.esp32s3._passes.remove_identity_expand_pass import (
        _contiguous_by_construction,
    )

    VIEW = exir_ops.edge.aten.view_copy.default
    graph = ep.graph_module.graph
    output = next(n for n in graph.nodes if n.op == "output")
    removed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not VIEW:
            continue
        src = node.args[0]
        vi = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
        vo = node.meta.get("val")
        if vi is None or vo is None or list(vi.shape) != list(vo.shape):
            continue
        if not _contiguous_by_construction(src):
            continue
        if src.op == "placeholder" and output in node.users:
            continue        # would alias a graph input to a graph output
        node.replace_all_uses_with(src)
        graph.erase_node(node)
        removed += 1
    if removed:
        graph.eliminate_dead_code()
        graph.lint()
        ep.graph_module.recompile()
    return removed


def format_report(rep) -> str:
    parts = [f"passes: {rep['before']} -> {rep['after']} call_function nodes"]
    parts.append(f"(shared clone/view -{rep.get('by_shared', 0)}; "
                 f"rmsnorm fusion {rep['fused']} sites; "
                 f"rope fusion {rep.get('rope_fused', 0)} sites; "
                 f"layernorm2d fusion {rep.get('ln2d_fused', 0)} sites; "
                 f"norm+residual {rep.get('rms_add_fused', 0)} sites; "
                 f"attention-tail fusion {rep.get('attn_tail_fused', 0)} sites; "
                 f"both -{rep['by_fusion']}; "
                 f"conv+bn folded {rep.get('convbn', 0)}; "
                 f"clamp chains -{rep.get('clamps', 0)}; "
                 + (f"table q/dq pairs -{rep['lut_pairs']}; "
                    if rep.get("lut_pairs") else "")
                 + (f"static quantizes portable {rep['static_q']}; "
                    if rep.get("static_q") else "")
                 + f"constant_prop -{rep['by_folding']})")
    if rep["rejected"]:
        parts.append(f"REJECTED {len(rep['rejected'])}: {rep['rejected'][:2]}")
    return " ".join(parts)


def canonical_passes():
    """The spelling-canonicalisation passes, run BEFORE partitioning.

    Distinct from `apply_passes`, which runs AFTER lowering and optimises what
    the delegate left behind. These run first and for one reason: whether
    a Linear reaches the partitioner as `mm` or as a batch-1 `bmm`, whether
    an attention key arrives as a permute or a view, and whether attention was
    written out or called as F.scaled_dot_product_attention, is decided by
    torch's decomposition and not by the model author. Partitioning without
    them measures the decomposition rather than the backend. `SdpaToQk` must
    precede `QkTo3D`, which finishes the key rewrite it starts.

    A fresh list per call: each pass carries its own counters, so sharing one
    instance across exports would accumulate and make `expect=` meaningless.
    """
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm
    from executorch.backends.esp32s3._passes.qk_to_3d_pass import QkTo3D
    from executorch.backends.esp32s3._passes.remove_identity_expand_pass import (
        RemoveIdentityExpand,
    )
    from executorch.backends.esp32s3._passes.sdpa_to_qk_pass import SdpaToQk

    return [BmmToMm(), SdpaToQk(), QkTo3D(), RemoveIdentityExpand()]


@fewer_gc_passes()
def lower_esp32(exported_program, partitioners, compile_config=None,
                method_name: str = "forward", quant_luts: bool = True,
                luts16: bool | None = None):
    """to_edge -> pre-partition prep -> canonicalise -> partition.

    `quant_luts` folds every activation between two static quantisers into an
    espressif::quant_lut table (fold_quant_lut_pass.py) before to_edge; the
    result's `quant_lut_report` says what was built and what was refused. On by
    default: a static graph then needs no portable transcendental between its
    delegates. A graph with no such activation is untouched either way (no
    symmetric program changes by a byte).

    `luts16` folds the int16 activations quant_lut refuses as lut-wide-codes
    into espressif::lut16 tables fitted from the graph (fold_lut16_pass.py),
    after quant_lut's fold; `lut16_report` says what. None (the default)
    follows the Esp32Partitioner's own luts16, which is on by default;
    True or False here overrides it. Off, nothing runs; a graph with no
    int16 activation is untouched either way. A program carrying observer
    ranges (espressif_quantizer.carry_observer_ranges) has them stripped
    first, kept for those tables (lut16_range.py).
    The Esp32Partitioner's fused_norms (on by default) also folds
    each decomposed quantised RMSNorm or LayerNorm a stock recipe leaves into
    one espressif::qnorm (fuse_qnorm_pass.py), before the tables;
    `qnorm_report` says what was fused and what kept decomposed, by rule.
    fused_norms=False: nothing runs.

    Right before partitioning, with an Esp32Partitioner and its gate: a
    dynamically quantised graph's activation chains (chooser, quantize,
    dequantize) are absorbed into the matmuls and convolutions that will be
    claimed, their schemes recorded, and left in place for any other reader,
    computed there by float32 portable ops as torch computes them
    (strip_dynamic_act_qdq; `stripped_chains` counts the absorbed), then a
    static int8 graph's quantize/dequantize pairs
    (static_qdq.absorb_static_qdq): its `static_qdq_report` says how. A graph
    with neither is untouched. After partitioning, a torchao quantize or
    dequantize over constants that no delegate took (a refused op's weight)
    is folded into the constant it computes, since no ExecuTorch kernel runs
    it (fold_portable_torchao).

    A graph input in another dim order (a channels-last image, as the
    Cortex-M flow feeds) is copied to row-major right after to_edge, and an
    output whose dim order that changes is copied back
    (make_inputs_row_major_pass.py; `row_major_report`). A graph whose
    inputs are all row-major is untouched.

    Replaces a bare `to_edge_transform_and_lower` because ONE of those steps has
    to happen before partitioning and could not be expressed as a transform
    pass: conv+bn folding.

    Why it matters, measured on a U-Net. Folded only AFTER lowering (in
    `apply_passes`), a `conv -> BN -> ReLU` chain would present the ReLU's
    producer to the partitioner as a BATCHNORM. The epilogue gates want a
    claimed matmul or conv, so every one of those ReLUs would stay portable --
    14 of them -- and each would also split a partition, because an unclaimed
    op between two claimed ones is a partition break. Folding first: 14 ReLUs
    fused, 0 left, and delegate calls 15 -> 7.

    `constant_prop_pass` immediately after the fold is not optional. The fold
    inserts graph ops on the constants and relies on propagation to
    materialise them; without it the next stage raises
    `SpecViolationError: Node.meta aten_add_scalar is missing val field`.

    The q@kT value check's first input set (operator_support/qk_values.py)
    is the program's example inputs, which to_edge drops: they are handed to
    each Esp32Partitioner for this lowering only (one partitioner may lower
    several methods), for the gate the pre-partition passes ask and for the
    partition itself.
    """
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    example_inputs = getattr(exported_program, "example_inputs", None)
    ours = [p for p in partitioners if isinstance(p, Esp32Partitioner)]
    saved = [(p, p.example_inputs) for p in ours]
    if example_inputs is not None:
        for p in ours:
            p.example_inputs = example_inputs
    try:
        return _lower_esp32(exported_program, partitioners, compile_config,
                            method_name, quant_luts, luts16)
    finally:
        for p, before in saved:
            p.example_inputs = before


def _lower_esp32(exported_program, partitioners, compile_config=None,
                method_name: str = "forward", quant_luts: bool = True,
                luts16: bool | None = None):
    """lower_esp32's body (lower_esp32 says what it does)."""
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    from executorch.backends.esp32s3._passes.promote_fp8_constants_pass import (
        promote_fp8_constants,
    )

    # First: the observer ranges espressif_quantizer.carry_observer_ranges
    # carries (lut16_range.py) go, each kept on the quantize it reads for the
    # int16 tables, before anything else looks at the graph. A program without
    # them is not touched.
    if _has_observed_ranges(exported_program):
        from executorch.backends.esp32s3.quantizer.lut16_range import strip_observed_ranges

        strip_observed_ranges(exported_program)
    # BEFORE to_edge, and it has to be: fp8 is legal in the ATen dialect and not
    # in the edge one, so `to_edge` rejects an fp8 checkpoint outright with a
    # SpecViolationError before any partitioner is consulted. No backend can fix
    # that from inside; the only window where the dtype is still legal is here.
    # A no-op for every model that has no fp8 in it.
    promote_fp8_constants(exported_program)
    # Before the tables (Esp32Partitioner's fused_norms, on by default): a
    # decomposed quantised norm becomes one op (fuse_qnorm_pass.py), its x^2
    # and rsqrt tables built from the ATen nodes
    # fold_quant_luts would otherwise fold one by one.
    qnorm_report = None
    if any(getattr(p, "fused_norms", False) for p in partitioners):
        from executorch.backends.esp32s3._passes.fuse_qnorm_pass import fold_qnorms

        qnorm_report = fold_qnorms(exported_program)
        if qnorm_report.fused or qnorm_report.refused:
            print("  " + qnorm_report.summary())
    # Also before to_edge: the eager model runs aten.silu, aten.hardswish,
    # aten.mish and aten.softplus, which to_edge decomposes into other
    # functions, so a table must be evaluated from the ATen node.
    report = None
    if quant_luts:
        from executorch.backends.esp32s3._passes.fold_quant_lut_pass import fold_quant_luts

        report = fold_quant_luts(exported_program)
        if report.tables or report.refused:
            print("  " + report.summary())
    report16 = None
    if luts16 is None:
        luts16 = any(getattr(p, "luts16", False) for p in partitioners)
    if luts16:
        from executorch.backends.esp32s3._passes.fold_lut16_pass import fold_fitted_luts

        report16 = fold_fitted_luts(exported_program)
        if report16.tables or report16.refused:
            print("  " + report16.summary())

    edge = _to_edge_validated_once(exported_program, compile_config)
    # A graph input in another dim order (a channels-last image) is copied to
    # row-major once, before anything reads it (make_inputs_row_major_pass.py):
    # the gate then claims what it claims for the row-major program, whose bits
    # the delegate computes. A no-op when every input is row-major.
    from executorch.backends.esp32s3._passes.make_inputs_row_major_pass import (
        make_inputs_row_major,
    )

    row_major = make_inputs_row_major(edge, method_name)
    if row_major.inputs or row_major.skipped:
        print("  " + row_major.summary())
    # ExecuTorch's own LLM recipes (stock_llm.py): the quantised K/V cache's
    # writer and reads, its q@kT, and the quantised embeddings, rewritten into
    # what the delegate consumes. First, so that no chooser is left for the
    # dynamic-chain sweep to refuse; a no-op for every other program.
    from executorch.backends.esp32s3._passes import stock_llm

    # The K side becomes OP_BMM_KCACHE_AFFINE_I8 only where its int8 query
    # rows are asked for, as every int8 q@kT is (quantize_qk), and not under
    # format_policy='fp32', which quantises nothing itself.
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    ours = [p for p in partitioners if isinstance(p, Esp32Partitioner)]
    int8_query = any(p.quantize_qk and p.compile_spec.format_policy != "fp32"
                     for p in ours)
    if any(p.quantize_qk for p in ours) and not int8_query:
        skip = ("format_policy='fp32' keeps every q@kT portable "
                "(no int8 query rows)")
    else:
        skip = "the int8 query is opt-in (quantize_qk)"
    stock = stock_llm.prepare(edge, method_name, kcache=int8_query,
                              skip_reason=skip)
    # BmmToMm before the conv+bn fold: the constant propagation the fold needs
    # would turn a bmm Linear's weight chain (permute, then unsqueeze, expand
    # or view) into one [B, K, N] constant, which the rewrite cannot read back
    # as one [K, N] matrix. Behind a conv+bn, that Linear would stay a portable
    # bmm wherever torch spells it as one: a batch-1 bmm, or a Linear over a
    # channels-last input (which delegates over a row-major one). The
    # canonical passes below run BmmToMm again and find nothing left to rewrite.
    from executorch.backends.esp32s3._passes.bmm_to_mm_pass import BmmToMm

    edge = _canonicalise(edge, [BmmToMm()])
    folded, _rejected = fold_conv_bn(edge, method_name)
    if folded:
        constant_prop_pass(edge.exported_program(method_name),
                           _skip_dim_order=_skips_dim_order(edge))
    edge = _canonicalise(edge, canonical_passes())
    # Last before partitioning, on the canonical graph the gate will see: a
    # dynamically quantised graph's activation chains, absorbed into the ops
    # that quantise their activation at execute (strip_dynamic_act_qdq_pass.py),
    # then a static int8 graph's quantize/dequantize pairs, absorbed where a
    # consumer will be claimed (static_qdq.py). The partitioner cannot do
    # either, being forbidden to change the graph. Each is a no-op without
    # its chains or pairs.
    static = None
    stripped = 0
    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )

    if len(partitioners) == 1 and isinstance(partitioners[0], Esp32Partitioner):
        from executorch.backends.esp32s3._passes.split_shared_folds import (
            split_shared_folds,
        )
        from executorch.backends.esp32s3._passes.static_qdq import absorb_static_qdq
        from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
            strip_dynamic_act_qdq,
        )

        # A transpose, slice or weight dequantise several matmuls share gets
        # a copy per reader (split_shared_folds.py): preprocess folds it
        # into each one's weight, and readers grouped into two partitions
        # would leave one of them a runtime weight.
        split_shared_folds(edge, method_name)
        stripped = strip_dynamic_act_qdq(edge, method_name, partitioners[0])
        static = absorb_static_qdq(edge, partitioners[0], method_name,
                                   lut_report=report)
        if (static.merged or static.identities or static.absorbed
                or static.requant or static.portable
                or static.stage_b is not None):
            print("  " + static.summary())
    edge = edge.to_backend(partitioners[0] if len(partitioners) == 1
                           else partitioners)
    # A torchao quantize or dequantize over constants that no delegate took
    # (the weight of an op the gate refused) has no ExecuTorch kernel: it is
    # folded into the constant it computes.
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        fold_portable_torchao,
    )

    fold_portable_torchao(edge.exported_program(method_name))
    if any(n.op == "call_function" and "quantized_decomposed" in str(n.target)
           for n in edge.exported_program(method_name).graph.nodes):
        # The .out variants to_executorch emits a quantize or dequantize left
        # outside the delegates with (a refused op's weight dequantise, a
        # static pair between portable ops): ExecuTorch registers them on
        # import.
        import executorch.kernels.quantized  # noqa: F401
    if stock["kcache"] or stock["embeddings"]:
        stock_llm.check_claimed(edge.exported_program(method_name).graph_module)
    edge.quant_lut_report = report
    edge.lut16_report = report16
    edge.qnorm_report = qnorm_report
    edge.static_qdq_report = static
    edge.stripped_chains = stripped
    edge.stock_llm_report = stock
    edge.row_major_report = row_major
    return edge


def _canonicalise(edge, passes):
    """edge.transform(passes) for lower_esp32's canonical passes, unvalidated.

    transform() would validate every method four times with the full edge
    verifier (~0.12 s each on a story Glimmer method): its pass manager
    before and after the passes, then transform() itself directly and again
    building the new manager. to_backend validates each program before it
    partitions it, so none of the four checks anything that would not be
    checked next: a config with _check_ir_validity off turns off the last
    two, and _CanonicalPassManager the first two. A program no pass changes
    comes back as it went in.
    """
    import dataclasses

    return edge.transform(_CanonicalPassManager(passes),
                          compile_config=dataclasses.replace(
                              edge.compile_config, _check_ir_validity=False))


def _has_observed_ranges(exported_program):
    programs = (exported_program.values() if isinstance(exported_program, dict)
                else [exported_program])
    return any(n.op == "call_function" and "espressif.observed_range" in str(n.target)
               for ep in programs for n in ep.graph_module.graph.nodes)


def _to_edge_validated_once(exported_program, compile_config=None):
    """to_edge, with the full edge validation run once per method, not five.

    With _check_ir_validity on, to_edge validates each method's edge program
    five times over (building it, before and after its post passes, directly,
    and building the manager): ~0.12 s each on a story Glimmer method, every
    op re-run on fake tensors. Here to_edge runs with validation off and the
    result is rebuilt with the caller's edge verifier, which validates it once
    and stays on the program, so to_backend and everything after validate as
    they do after a plain to_edge.

    The same programs come out, with one ordering difference to rule out:
    validation also makes to_edge drop the decomposed graph's
    _assert_tensor_metadata nodes and dead code BEFORE removing unused
    parameters; without it, a parameter used only by such nodes would survive
    as a placeholder with no users. Any PARAMETER placeholder without users
    sends this back to plain to_edge, which is exact. What validation-off
    skips besides is the ATen-dialect op check on the decomposed graph; the
    edge verifier applies the same core-ATen check to every op that reaches
    the edge program.
    """
    import dataclasses

    from executorch.exir import EdgeCompileConfig, to_edge
    from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
    from torch.export import ExportedProgram
    from torch.export.graph_signature import InputKind

    config = compile_config or EdgeCompileConfig()
    if not config._check_ir_validity:
        return to_edge(exported_program, compile_config=compile_config)
    edge = to_edge(exported_program, compile_config=dataclasses.replace(
        config, _check_ir_validity=False))
    for program in edge._edge_programs.values():
        users = {n.name: len(n.users) for n in program.graph.nodes
                 if n.op == "placeholder"}
        if any(spec.kind == InputKind.PARAMETER and not users[spec.arg.name]
               for spec in program.graph_signature.input_specs):
            return to_edge(exported_program, compile_config=compile_config)
    verifier = EXIREdgeDialectVerifier(
        edge_compile_config=config, class_only=True,
        preserve_ops=list(config.preserve_ops))
    for name, program in list(edge._edge_programs.items()):
        gm = program.graph_module
        edge._edge_programs[name] = ExportedProgram(
            root=gm, graph=gm.graph, graph_signature=program.graph_signature,
            state_dict=program.state_dict,
            range_constraints=program.range_constraints,
            module_call_graph=program.module_call_graph,
            example_inputs=program.example_inputs, constants=program.constants,
            verifiers=[verifier])  # the one validation
    edge.compile_config = config
    return edge


class _CanonicalPassManager(ExportedProgramPassManager):
    """ExecuTorch's pass manager without its program validation.

    Only for lower_esp32's canonical passes, whose result goes straight to
    to_backend, which validates it first thing. The structural checks stay;
    the recompile they came with goes too (each modifying pass is already
    recompiled after, and nothing runs the module before to_backend copies
    it).
    """

    def check(self, exported_program):
        graph_module = exported_program.graph_module
        graph_module.graph.lint()
        for node in graph_module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for "
                    f"backend delegate.")
