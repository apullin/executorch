# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serialises one partition's ops and weights into the delegate blob."""

import struct

import numpy as np
from executorch.exir.backend.backend_details import (
    DelegateScratchSpec,
    PreprocessResult,
)
from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch.backends.esp32s3.compile_spec import Esp32CompileSpec
from executorch.backends.esp32s3.operator_support.const_weights import _const_fqn
from executorch.backends.esp32s3.operator_support.espressif_operator_support import (
    _pad_to,
)
from executorch.backends.esp32s3.serialization import scratch_layout
from executorch.backends.esp32s3.serialization.asym_records import _act_record
from executorch.backends.esp32s3.serialization.espressif_blob import (
    BACKEND_NAME,
    BLOB_HEADER_BYTES,
    BLOB_HEADER_FORMAT,
    DTYPE_INT8,
    EXT_ACT_QUANT,
    EXT_COUNT_FORMAT,
    EXT_OUT_QUANT,
    EXT_RECORD_HEADER_FORMAT,
    EXT_WEIGHT_AFFINE,
    FLAG_ACT_PER_TENSOR,
    FLAG_ACT_QUANT,
    FLAG_CONV,
    FLAG_EXTERNAL_WEIGHTS,
    FLAG_OUT_QUANT,
    FLAG_STATIC_ACT,
    FLAG_WEIGHT_AFFINE,
    GRANULARITY_STATIC,
    MAGIC,
    OFFSET_FP16,
    OFFSET_FP32,
    OFFSET_NONE,
    OP_DESC_BYTES,
    OP_DESC_FORMAT,
    SUM_INT16,
    SUM_INT32,
    SUM_NONE,
    VEC_BYTES,
    VERSION,
    ZP_FP32,
    ZP_INT8,
    ZP_NONE,
    external_data_key,
)
from executorch.backends.esp32s3.serialization.weight_formats import FORMATS

# What the delegate's one planned scratch buffer is called
# (DelegateScratchSpec.label): an integrator's planner may place buffers by it.
SCRATCH_LABEL = "esp32.scratch"
# What the declaration adds to the layout: the planner aligns a buffer's
# offset within the arena, but the arena's base is the integrator's, so the
# runtime starts the layout at the buffer's first 16-byte boundary
# (planned_scratch in esp32_backend.cpp) and needs up to this much before it.
SCRATCH_SLACK = VEC_BYTES - 1


class _BlobWriter:
    """The blob under construction and the PTD store beside it.

    Layout: the header (MAGIC, VERSION, n_ops, ext_off), n_ops
    descriptors, then the body every descriptor's offsets point into (biases,
    inline weights and scales, or the NamedDataMap keys that replace them),
    then the extension table. `place` hands out body offsets in order and
    nothing moves once placed, so adding an extension can never move a weight.
    """

    def __init__(self, edge_program, boundary, compile_specs, n_ops):
        self.edge_program = edge_program
        self.act_in = boundary.act_in
        self.n_ops = n_ops
        # One decoder, shared with the encoder that produced these bytes, so
        # the two halves cannot drift on a key name, an encoding or a default.
        self.cfg = Esp32CompileSpec.from_list(compile_specs)
        self.ext_records = []      # (op_index, ext_type, payload bytes)
        self.descs = []
        self.desc_txt = []
        self.body = bytearray()
        self.store = NamedDataStore()
        self.ext_bytes = 0
        self.offset = BLOB_HEADER_BYTES + OP_DESC_BYTES * n_ops

    def place(self, payload: bytes, align: int = 4) -> int:
        """Append a payload to the body at the given alignment; its offset."""
        pad = _pad_to(self.offset, align) - self.offset
        self.body.extend(b"\0" * pad)
        self.offset += pad
        at = self.offset
        self.body.extend(payload)
        self.offset += len(payload)
        return at

    def store_weights(self, payload: bytes, scale_bytes: bytes):
        """Where a packed weight and its scales live: (wt_off, scale_off,
        flags). Inline in the body, or in the PTD store with the body holding
        the two keys.

        Keys are content-addressed, so a tensor used by two ops is stored
        ONCE. This is not a micro-optimisation: the PLE model's head is TIED
        to the input embedding, so `tok_emb` appears twice -- as an embedding
        gather and as the output matmul -- in two different partitions, hence
        two preprocess() calls, hence two key prefixes and two 1.57 MB copies
        of identical bytes. That is 11% of the 15.66 MB flash budget spent on
        a duplicate, and it is the difference between the deployed config
        fitting and not.

        A content hash also keeps keys unique across partitions sharing one
        NamedDataMap, and collides exactly when collision is what we want.

        Alignment is load-bearing, not tidiness: ee.vld.128.ip truncates the
        low address bits silently, so an unaligned weight returns wrong data
        rather than faulting.
        """
        if not self.cfg.external_weights:
            wt_off = self.place(payload, align=VEC_BYTES)
            scale_off = self.place(scale_bytes)
            return wt_off, scale_off, 0
        wkey = external_data_key("w", payload)
        skey = external_data_key("s", scale_bytes)
        self.store.add_named_data(wkey, payload, alignment=VEC_BYTES,
                                  external_tag=self.cfg.external_tag)
        self.store.add_named_data(skey, scale_bytes, alignment=4,
                                  external_tag=self.cfg.external_tag)
        wt_off = self.place(wkey.encode() + b"\0", align=1)
        scale_off = self.place(skey.encode() + b"\0", align=1)
        self.ext_bytes += len(payload) + len(scale_bytes)
        return wt_off, scale_off, FLAG_EXTERNAL_WEIGHTS

    def store_float_weights(self, payload: bytes) -> tuple[int, int]:
        """Where a float weight lives -- the fp32 loop's [K, N], a float
        conv's or a 2-byte rung's [N, K]: (wt_off, flags). Inline in the body
        at its element alignment (4, as always), or under external_weights in
        the PTD store, content-addressed as store_weights' payloads are, with
        the body holding the key. A float weight has no scales, so there is
        no scale key: scale_off stays 0 and the runtime fetches one entry."""
        if not self.cfg.external_weights:
            return self.place(payload), 0
        key = external_data_key("w", payload)
        self.store.add_named_data(key, payload, alignment=VEC_BYTES,
                                  external_tag=self.cfg.external_tag)
        self.ext_bytes += len(payload)
        return self.place(key.encode() + b"\0", align=1), FLAG_EXTERNAL_WEIGHTS

    def store_table(self, kind: str, payload: bytes, elem: int) -> int:
        """Where one EXT_WEIGHT_AFFINE table lives, as store_weights places
        weights: inline at its element alignment, or as content-addressed
        named data (kind "z", "m" or "u") with the body holding the key."""
        if not self.cfg.external_weights:
            return self.place(payload, align=elem)
        key = external_data_key(kind, payload)
        self.store.add_named_data(key, payload, alignment=max(elem, 4),
                                  external_tag=self.cfg.external_tag)
        self.ext_bytes += len(payload)
        return self.place(key.encode() + b"\0", align=1)

    def weight_affine(self, pt, rows: int, groups: int, *, act=None,
                      codes=None, k=0, gsize=0, bits=8):
        """FLAG_WEIGHT_AFFINE and its EXT_WEIGHT_AFFINE record, tables placed,
        for the descriptor about to be emitted, and its emit-log note; (0, "")
        for a symmetric weight under activations without a zero point, which
        therefore packs as plain codes and scales. Tables are [rows, groups]: the
        stored rows (a tied head's full table) by the op's scale groups.

        An activation that can have a zero point (`act`, the op's ActQuant)
        needs the weight sums S' = the sum over each group's REAL elements
        (columns < k) of (c - zw), from the stored `codes` [rows, >= k]: int16
        where the group length and code span prove it holds, int32 otherwise.
        `pt` is None for a weight quantised here (symmetric codes).
        `gsize` is the descriptor's scale_group (0: one group per row)."""
        weight = pt is not None and pt.affine
        sums = act is not None and act.affine
        if not weight and not sums:
            return 0, ""
        from executorch.backends.esp32s3.serialization.asym_records import (
            WeightAffine,
            sums_payload,
            table_payloads,
        )

        def values(t):
            return None if t is None else t.detach().cpu().numpy()

        zp_kind, zp_bytes, off_kind, off_bytes = (ZP_NONE, b"", OFFSET_NONE,
                                                  b"")
        bias = 0
        if weight:
            zp_kind, zp_bytes, off_kind, off_bytes = table_payloads(
                rows, groups, zero_points=values(pt.zero_points),
                float_zero_points=values(pt.float_zero_points),
                offsets=values(pt.offsets))
            bias = pt.storage_bias
        zp_off = off_off = sum_off = 0
        if zp_kind != ZP_NONE:
            zp_off = self.store_table("z", zp_bytes,
                                      1 if zp_kind == ZP_INT8 else 4)
        if off_kind != OFFSET_NONE:
            off_off = self.store_table("m", off_bytes,
                                       2 if off_kind == OFFSET_FP16 else 4)
        sum_kind = SUM_NONE
        if sums:
            zw = values(pt.zero_points) if weight else None
            sum_kind, sum_bytes, elem = sums_payload(
                np.asarray(codes), rows, groups, k, gsize, bits,
                zero_points=zw, storage_bias=bias)
            sum_off = self.store_table("u", sum_bytes, elem)
        record = WeightAffine(zp_kind=zp_kind, offset_kind=off_kind,
                              sum_kind=sum_kind, storage_bias=bias,
                              zp_off=zp_off, offset_off=off_off,
                              sum_off=sum_off)
        self.ext_records.append((len(self.descs), EXT_WEIGHT_AFFINE,
                                 record.pack()))
        note = ("+zp" if zp_kind == ZP_INT8 else
                "+fzp" if zp_kind == ZP_FP32 else "")
        note += {OFFSET_FP16: "+off16", OFFSET_FP32: "+off32"}.get(off_kind, "")
        if bias:
            note += f"+bias{bias}"
        note += {SUM_INT16: "+sum16", SUM_INT32: "+sum32"}.get(sum_kind, "")
        return FLAG_WEIGHT_AFFINE, note

    def override_name(self, op):
        """The format_overrides value naming this op's weight, or None."""
        overrides = self.cfg.format_overrides
        if not overrides:
            return None
        source = op.get("weight_node")
        fqn = (_const_fqn(self.edge_program, source)
               if source is not None else None)
        return overrides.get(fqn) if fqn is not None else None

    def float_conv_geometry(self, op, elem_bytes):
        """FLAG_CONV plus the EXT_CONV3D record for a float-weight conv (the
        fp32 convolution, an fp16/bf16 one), and its emit-log note: the same
        record as conv_geometry, with the traffic the float loop actually
        makes -- it gathers one output position at a time and re-reads every
        weight row for each, m passes over n*k elements."""
        if op.get("conv") is None:
            return 0, ""
        from executorch.backends.esp32s3.operator_support.conv import EXT_CONV3D

        geom = op["conv"]
        self.ext_records.append((len(self.descs), EXT_CONV3D, geom.pack()))
        traffic = geom.m * geom.n * geom.k * elem_bytes
        return FLAG_CONV, (f" conv{geom.kh}x{geom.kw}"
                           f"[{geom.c_in}->{geom.c_out} @"
                           f"{geom.h_out}x{geom.w_out}] {geom.m}pass "
                           f"{traffic / 1024:.0f}KB")

    def override_for(self, op):
        """The overridden Fmt for this op's weight, or None.

        Resolved by the weight's parameter name, which the partition
        subprogram keeps in its graph signature. A pass-through weight's
        format IS its codes' format, so overriding one is refused rather than
        quietly ignored.
        """
        overrides = self.cfg.format_overrides
        if not overrides:
            return None
        source = op.get("weight_node")
        fqn = (_const_fqn(self.edge_program, source)
               if source is not None else None)
        if fqn is None or fqn not in overrides:
            return None
        if op.get("pt") is not None:
            raise RuntimeError(
                f"{BACKEND_NAME}: format_overrides[{fqn!r}] names a "
                f"pass-through weight; its codes fix its format")
        return FORMATS[overrides[fqn]]

    def static_scale(self, node, fmt):
        """The calibrated activation scale for one matmul, or 0.0 when the
        model runs dynamic. Loud on an uncalibrated graph: a static-act model
        missing its meta is an export-order bug, not a runtime surprise."""
        if not self.cfg.static_act:
            return 0.0, 0
        if fmt.bits == 16:
            raise RuntimeError(
                f"{BACKEND_NAME}: static activation scales are int8-range "
                f"only; the i16 rung keeps per-row dynamic quantisation"
            )
        amax = node.meta.get("esp32_act_amax")
        if amax is None:
            raise RuntimeError(
                f"{BACKEND_NAME}: static_act requested but {node.name} has no "
                f"esp32_act_amax meta; run calibrate_static_act on the edge "
                f"program BEFORE to_backend"
            )
        amax = float(amax)
        if not np.isfinite(amax) or amax < 0.0:
            raise RuntimeError(
                f"{BACKEND_NAME}: {node.name} has invalid calibrated "
                f"activation absmax {amax}"
            )
        return amax / fmt.act_qmax, FLAG_STATIC_ACT

    def activation_scale(self, node, fmt, codes_in=False):
        """How the runtime scales this op's activation: (act_scale, flags).

        A calibrated static scale travels in the descriptor. A PT2E model
        instead folded a dynamic per-TENSOR quantisation into the op
        (strip_dynamic_act_qdq marks it esp32_act_per_tensor); no scale
        travels, the model's own
        `choose_qparams_symmetric` computes it at execute from the tensor,
        and so does the runtime. The two are mutually exclusive -- both mean
        "one scale for the whole activation", and having both set would
        leave which one wins to the order of two ifs in the kernel.

        `codes_in`: the op reads Stage B's int8 codes (stage_b.py), so its
        static record says in_dtype 1 -- the runtime reads the codes as they
        are, with the record's zero point for a conv's padding.
        """
        act_scale, flags = self.static_scale(node, fmt)
        act = _act_record(node)
        if codes_in:
            if (act is None or act.granularity != GRANULARITY_STATIC
                    or act.code_bits != 8):
                raise RuntimeError(
                    f"{BACKEND_NAME}: {node.name} reads int8 codes without a "
                    f"static 8-bit activation record; gate and walk disagree")
            import dataclasses

            act = dataclasses.replace(act, in_dtype=DTYPE_INT8)
        if act is not None:
            # The model's own dynamic quantiser, absorbed whole
            # (strip_dynamic_act_qdq): EXT_ACT_QUANT, the one
            # scheme of this op. The chain is gone from the graph, so what
            # the record cannot ride on is refused by name, never dropped.
            from executorch.backends.esp32s3.serialization.asym_records import SchemeRefused

            if fmt.bits not in (4, 8):
                raise SchemeRefused(
                    f"[act-format] {node.name}: the model quantises this op's "
                    f"activation itself, which the runtime computes on int8 "
                    f"and int4 weights only, not {fmt.name}")
            if node.meta.get("esp32_act_per_tensor"):
                raise RuntimeError(
                    f"{BACKEND_NAME}: {node.name} carries two activation "
                    f"schemes")
            if flags & FLAG_STATIC_ACT:
                raise RuntimeError(
                    f"{BACKEND_NAME}: {node.name} has both a calibrated "
                    f"static activation scale and the model's own dynamic "
                    f"quantiser; pick one -- static_act=False keeps the "
                    f"model's")
            self.ext_records.append((len(self.descs), EXT_ACT_QUANT,
                                     act.pack()))
            return act_scale, flags | FLAG_ACT_QUANT
        if node.meta.get("esp32_act_per_tensor"):
            if fmt.bits == 16:
                raise RuntimeError(
                    f"{BACKEND_NAME}: a folded PT2E per-tensor activation is "
                    f"int8-range; the i16 rung quantises activations to int16 "
                    f"and would apply the scale to a different grid")
            if flags & FLAG_STATIC_ACT:
                raise RuntimeError(
                    f"{BACKEND_NAME}: {node.name} has both a calibrated "
                    f"static activation scale and a folded PT2E per-tensor "
                    f"one; pick one -- static_act=False to keep the model's "
                    f"own, or do not fold to keep the calibration")
            flags |= FLAG_ACT_PER_TENSOR
        return act_scale, flags

    def out_quant(self, op, fmt):
        """FLAG_OUT_QUANT and the EXT_OUT_QUANT record of an op that absorbed
        its output's static pair (_fuse_requant), and its emit-log note;
        (0, "") for every other op. The runtime applies it after the op's
        activation and clamp: yq = clamp(rne(fl(y * fl(1/so))) + zo), then
        writes fl((yq - zo) * so)."""
        record = op.get("out_quant")
        if record is None:
            return 0, ""
        if fmt.bits not in (4, 8):
            from executorch.backends.esp32s3.serialization.asym_records import SchemeRefused

            raise SchemeRefused(
                f"[out-format] {op['node'].name}: an output quantisation runs "
                f"in the int8 and int4 loops only, not {fmt.name}")
        self.ext_records.append((len(self.descs), EXT_OUT_QUANT,
                                 record.pack()))
        return FLAG_OUT_QUANT, (f"+oq{8 if record.out_dtype == DTYPE_INT8 else ''}"
                                f"[{record.scale:.6g},"
                                f"{record.zero_point}]")

    def conv_geometry(self, op, fmt):
        """FLAG_CONV plus the EXT_CONV3D record for a conv op, and its
        emit-log note. (0, "") for everything else.

        ONE helper for every storage path -- inline, external, and
        pass-through codes -- because the geometry is independent of where
        the weights live: a storage path without it would pack a conv with no
        gather. The flag and the
        record are emitted together: the runtime refuses either without the
        other, because a flag without geometry would gather nothing and
        silently run a matmul over the raw input.
        """
        if op.get("conv") is None:
            return 0, ""
        from executorch.backends.esp32s3.operator_support.conv import (
            EXT_CONV3D,
            block_rows,
            flash_traffic,
            runtime_block,
        )

        geom = op["conv"]
        self.ext_records.append((len(self.descs), EXT_CONV3D, geom.pack()))
        # The block size and its flash cost, PRINTED rather than trusted. A
        # bad B is a 300x regression parity cannot see: the arithmetic is
        # identical and every checksum matches, so silence would be
        # indistinguishable from health. What the DEVICE will do (the
        # mirrored runtime rule), not what the policy would like it to.
        k_pad = _pad_to(geom.k, fmt.k_align)
        b_rows = runtime_block(geom.m, k_pad, geom.n, conv=True)
        tr, passes = flash_traffic(
            geom, b_rows, weight_bytes_per_elem=fmt.bits / 8.0)
        # The headroom, reported beside it rather than instead of it: how
        # much traffic the whole SRAM-residency budget would save. A real
        # memory-vs-bandwidth trade on a 512 KB part, surfaced as a number.
        b_want = block_rows(geom.k, m=geom.m)
        head = ""
        if b_want > b_rows:
            tr_want, p_want = flash_traffic(
                geom, b_want, weight_bytes_per_elem=fmt.bits / 8.0)
            if tr_want < tr:
                head = (f" (B={b_want} would be {p_want}pass"
                        f" {tr_want / 1024:.0f}KB, {passes / p_want:.0f}x"
                        f" less)")
        note = (f" conv{geom.kh}x{geom.kw}"
                f"[{geom.c_in}->{geom.c_out} @"
                f"{geom.h_out}x{geom.w_out}] B={b_rows} "
                f"{passes}pass {tr / 1024:.0f}KB-flash{head}")
        return FLAG_CONV, note

    def emit(self, text, *, opcode, flags, m, k, n, k_pad, in_arg, out_arg,
             bias_off=0, wt_off=0, scale_off=0, scale_group=0,
             bounds=(0.0, 0.0), res_arg=0, act_scale=0.0):
        """Append one descriptor and its line of the emit log. The keyword
        names are the OpDesc fields, in the wire order."""
        self.desc_txt.append(text)
        self.descs.append(struct.pack(
            OP_DESC_FORMAT, opcode, flags, m, k, n, k_pad,
            bias_off, wt_off, scale_off, in_arg, out_arg, scale_group,
            bounds[0], bounds[1], res_arg, act_scale,
        ))

    def finish(self) -> PreprocessResult:
        """Header, descriptors, body and the extension table, in that order.

        The extension table goes AFTER the body, so every offset computed
        while placing stays valid. `ext_off` is 0 when there are none, which
        is the whole cost for a model that needs no extensions: one word in
        the header, nothing per op.

        Under plan_scratch, the delegate's per-call working memory, as the
        runtime will lay it out from these bytes (scratch_layout.py), plus
        SCRATCH_SLACK, is declared as one DelegateScratchSpec for the memory
        planner; nothing is declared for a blob that needs none.
        """
        ext_off = 0
        ext_blob = b""
        if self.ext_records:
            ext_off = self.offset
            parts = [struct.pack(EXT_COUNT_FORMAT, len(self.ext_records))]
            for op_index, ext_type, payload in self.ext_records:
                pad = (-len(payload)) % 4
                parts.append(struct.pack(EXT_RECORD_HEADER_FORMAT, op_index,
                                         ext_type, len(payload)))
                parts.append(payload + b"\0" * pad)
            ext_blob = b"".join(parts)
        header = struct.pack(BLOB_HEADER_FORMAT, MAGIC, VERSION, self.n_ops,
                             ext_off)
        blob = header + b"".join(self.descs) + bytes(self.body) + ext_blob
        assert len(blob) == self.offset + len(ext_blob), (
            f"blob {len(blob)} B but offsets ran to {self.offset} + "
            f"{len(ext_blob)} B of extensions")
        if self.cfg.external_weights:
            where = f"+ {self.ext_bytes:,} B external"
        else:
            where = "(weights inline)"
        scratch_specs = None
        if self.cfg.plan_scratch:
            scratch = scratch_layout.scratch_bytes(blob, self.cfg.min_latency)
            where += f", {scratch:,} B planned scratch"
            if scratch:
                scratch_specs = [DelegateScratchSpec(scratch + SCRATCH_SLACK,
                                                     label=SCRATCH_LABEL)]
        print(f"  [{BACKEND_NAME}] {self.n_ops} op(s), {len(blob):,} B blob "
              f"{where}: " + ", ".join(self.desc_txt))
        return PreprocessResult(
            processed_bytes=blob,
            data_store_output=self.store.get_named_data_store_output(),
            scratch_specs=scratch_specs,
        )
