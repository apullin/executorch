# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inventory valid ESP32 backend descriptors in the runner's embedded PTEs.

Coverage claims here come only from programs produced by the normal exporters
and selected by ``pte_blobs.S``, or packed into a flash-partition image the
firmware runs them from (PARTITION_IMAGES).  Host-only blob constructors and
malformed artifact tests deliberately do not count: this answers what a real
exported model executes, not what the ABI says could execute.
"""

from __future__ import annotations

import collections
import os
import re
import struct
from dataclasses import dataclass

from executorch.backends.esp32s3.serialization import espressif_blob as eb
from executorch.backends.esp32s3.operator_support.conv import (
    CONV_GEOM_BYTES,
    ConvGeom,
    EXT_CONV3D,
)


OPCODE_NAMES = eb.OPCODE_NAMES
# Every flag but the two storage-tier ones (FLAG_STAGE_I8 and FLAG_HEAD_SPLIT:
# a tied head staged in PSRAM as int8, and split across the cores). The census
# neither requires them live nor reads a blob that carries them, as before the
# reader moved into espressif_blob.
FLAG_NAMES = {flag: name for flag, name in eb.FLAG_NAMES.items()
              if flag not in (eb.FLAG_STAGE_I8, eb.FLAG_HEAD_SPLIT)}
KNOWN_FLAGS = 0
for _flag in FLAG_NAMES:
    KNOWN_FLAGS |= _flag
# Extension records, each required by the flag it travels with.
RECORD_NAMES = {
    EXT_CONV3D: "conv3d",
    eb.EXT_SELECTED_ROWS: "selected_rows",
    eb.EXT_ACT_QUANT: "act_quant",
    eb.EXT_WEIGHT_AFFINE: "weight_affine",
    eb.EXT_OUT_QUANT: "out_quant",
}
# Programs the parity firmware runs from a flash partition rather than the app
# image (the packed rows of test/fixtures: asym.py's, asym_act.py's,
# asym_static.py's two, stock_llm.py's, asym_a16.py's, asym_qat.py's,
# asym_stage_b.py's, lut16.py's, qnorm.py's and cnn.py's): scanned like an
# embedded program, blob by blob.
PARTITION_IMAGES = ("asym_fixtures.bin", "asym_act_fixtures.bin",
                    "asym_static_fixtures.bin", "stock_llm_fixtures.bin",
                    "asym_a16_fixtures.bin", "asym_qat_fixtures.bin",
                    "asym_ops_fixtures.bin", "asym_stageb_fixtures.bin",
                    "asym_lut16_fixtures.bin", "asym_norm_fixtures.bin",
                    "cnn_fixtures.bin")


@dataclass(frozen=True)
class ArtifactRecord:
    pte: str
    blob_offset: int
    op_index: int
    desc: dict
    dispatch: str
    geometry: ConvGeom | None
    selected: tuple[int, int] | None
    ext_types: tuple[int, ...] = ()


@dataclass(frozen=True)
class ArtifactCensus:
    embedded_ptes: tuple[str, ...]
    missing_ptes: tuple[str, ...]
    backend_ptes: tuple[str, ...]
    blob_count: int
    records: tuple[ArtifactRecord, ...]

    @property
    def seen_opcodes(self) -> set[int]:
        return {record.desc["opcode"] for record in self.records}

    @property
    def seen_flags(self) -> int:
        value = 0
        for record in self.records:
            value |= record.desc["flags"]
        return value

    @property
    def seen_ext_types(self) -> set[int]:
        return {t for record in self.records for t in record.ext_types}

    def declared_coverage_gaps(self) -> list[str]:
        """Return missing frozen-ABI primitives, plus core composition controls."""
        gaps = [
            f"opcode {OPCODE_NAMES[opcode]}"
            for opcode in OPCODE_NAMES
            if opcode not in self.seen_opcodes
        ]
        gaps.extend(
            f"flag {name}"
            for flag, name in FLAG_NAMES.items()
            if not self.seen_flags & flag
        )
        gaps.extend(f"record {name}" for ext, name in RECORD_NAMES.items()
                    if ext not in self.seen_ext_types)

        def has(opcode, *, flags=0, dispatch=None, grouped_scale=None):
            for record in self.records:
                desc = record.desc
                if desc["opcode"] != opcode or desc["flags"] & flags != flags:
                    continue
                if dispatch is not None and record.dispatch != dispatch:
                    continue
                if (grouped_scale is not None
                        and bool(desc["scale_group"]) != grouped_scale):
                    continue
                return True
            return False

        for opcode, name in ((eb.OP_ADDMM_I8, "i8"),
                             (eb.OP_ADDMM_I4, "i4"),
                             (eb.OP_ADDMM_I16, "i16")):
            for dispatch in ("conv2d", "depthwise", "grouped", "transposed"):
                if not has(opcode, flags=eb.FLAG_CONV, dispatch=dispatch):
                    gaps.append(f"{name} {dispatch}")
        for opcode, name in ((eb.OP_ADDMM_I4, "i4"),
                             (eb.OP_ADDMM_I8, "i8")):
            if not has(opcode, flags=eb.FLAG_FP16_SCALES, grouped_scale=True):
                gaps.append(f"{name} grouped fp16 scales")
        for opcode, name in ((eb.OP_ADDMM_I4, "i4"),
                             (eb.OP_ADDMM_I16, "i16")):
            if not has(opcode, flags=eb.FLAG_ACT_SILU, dispatch="matmul"):
                gaps.append(f"{name} transformer SiLU")
            if not has(opcode, flags=eb.FLAG_ADD, dispatch="matmul"):
                gaps.append(f"{name} matmul residual")
        if not has(eb.OP_ADDMM_I4, flags=eb.FLAG_EXTERNAL_WEIGHTS,
                   dispatch="matmul"):
            gaps.append("external packed-i4 learned weights")
        if not has(eb.OP_EMBED_I4, flags=eb.FLAG_EXTERNAL_WEIGHTS):
            gaps.append("external packed-i4 embedding")
        if not any(record.desc["opcode"] == eb.OP_BMM_DYN_I8
                   and record.desc["scale_group"] > 1
                   for record in self.records):
            gaps.append("batched dynamic-i8 Q@K")
        # The int8 K cache reads a row in place when it is whole vectors and
        # stages a ragged one; each branch, and the batch offsets into codes
        # and scales, needs a live artifact.
        kcache = [record.desc for record in self.records
                  if record.desc["opcode"] == eb.OP_BMM_KCACHE_I8]
        if not any(desc["scale_group"] > 1 for desc in kcache):
            gaps.append("batched int8-K-cache Q@K")
        if not any(desc["k"] == desc["k_pad"] for desc in kcache):
            gaps.append("int8-K-cache rows read in place")
        if not any(desc["k"] < desc["k_pad"] for desc in kcache):
            gaps.append("int8-K-cache ragged rows staged")
        # The same three branches of ExecuTorch's stock affine K cache.
        affine = [record.desc for record in self.records
                  if record.desc["opcode"] == eb.OP_BMM_KCACHE_AFFINE_I8]
        if not any(desc["scale_group"] > 1 for desc in affine):
            gaps.append("batched affine-K-cache Q@K")
        if not any(desc["k"] == desc["k_pad"] for desc in affine):
            gaps.append("affine-K-cache rows read in place")
        if not any(desc["k"] < desc["k_pad"] for desc in affine):
            gaps.append("affine-K-cache ragged rows staged")
        if not any(record.selected is not None for record in self.records):
            gaps.append("dynamic selected output rows")
        return gaps

    def summary(self) -> str:
        return (
            f"{len(self.records)} descriptors in {len(self.backend_ptes)} PTEs/"
            f"{self.blob_count} blobs; {len(self.seen_opcodes)}/{len(OPCODE_NAMES)} "
            f"opcodes, {sum(bool(self.seen_flags & f) for f in FLAG_NAMES)}/"
            f"{len(FLAG_NAMES)} flags and "
            f"{len(self.seen_ext_types & set(RECORD_NAMES))}/{len(RECORD_NAMES)} "
            f"records live"
        )


def _dispatch(geom: ConvGeom | None,
              selected: tuple[int, int] | None = None) -> str:
    if selected is not None:
        return "selected"
    if geom is None:
        return "matmul"
    if geom.depthwise:
        return "depthwise"
    if geom.groups > 1:
        return "grouped"
    if geom.transposed:
        return "transposed"
    if geom.kd > 1 or geom.d_in > 1:
        return "conv3d"
    if geom.kh == 1 and geom.h_in == 1:
        return "conv1d"
    return "conv2d"


def _extensions(
        data: bytes, blob_at: int, ext_off: int, types=None,
) -> tuple[dict[int, ConvGeom], dict[int, tuple[int, int]]]:
    """Conv geometry and selected rows per op; every record's type per op
    into `types` ({op_index: [type, ...]}) when given. The table is read by
    espressif_blob.read_extensions; a later record of a type replaces an
    earlier one."""
    geometry = {}
    selected = {}
    for op_index, records in eb.read_extensions(data, blob_at, ext_off).items():
        for base_type, payload in records:
            if types is not None:
                types.setdefault(op_index, []).append(base_type)
            if base_type == EXT_CONV3D and len(payload) == CONV_GEOM_BYTES:
                geometry[op_index] = ConvGeom.unpack(payload)
            elif (base_type == eb.EXT_SELECTED_ROWS
                  and len(payload) == eb.SELECTED_ROWS_BYTES):
                selected[op_index] = struct.unpack(eb.SELECTED_ROWS_FORMAT, payload)
    return geometry, selected


def scan_embedded_artifacts(main_dir: str) -> ArtifactCensus:
    """Scan backend blobs in every PTE named by ``pte_blobs.S``."""
    assembly = open(os.path.join(main_dir, "pte_blobs.S"), encoding="utf-8").read()
    embedded = tuple(dict.fromkeys(re.findall(r'"([^"\n]+\.pte)"', assembly)))
    embedded += tuple(name for name in PARTITION_IMAGES
                      if os.path.exists(os.path.join(main_dir, name)))
    missing = []
    backend_ptes = set()
    records = []
    blob_count = 0

    for filename in embedded:
        path = os.path.join(main_dir, filename)
        if not os.path.exists(path):
            missing.append(filename)
            continue
        data = open(path, "rb").read()
        for blob in eb.scan_blobs(data, KNOWN_FLAGS):
            at = blob.offset
            ext_off = struct.unpack_from(eb.BLOB_HEADER_FORMAT, data, at)[3]
            types = {}
            geometry, selections = _extensions(data, at, ext_off, types)
            backend_ptes.add(filename)
            blob_count += 1
            for index, desc in enumerate(blob.descs):
                geom = geometry.get(index)
                selected = selections.get(index)
                records.append(ArtifactRecord(
                    filename, at, index, desc, _dispatch(geom, selected), geom,
                    selected, tuple(types.get(index, ()))))

    return ArtifactCensus(
        embedded, tuple(missing), tuple(sorted(backend_ptes)), blob_count,
        tuple(records))


def render_details(census: ArtifactCensus) -> str:
    by_opcode = collections.Counter(
        OPCODE_NAMES[record.desc["opcode"]] for record in census.records)
    by_flag = collections.Counter()
    for record in census.records:
        for flag, name in FLAG_NAMES.items():
            if record.desc["flags"] & flag:
                by_flag[name] += 1
    lines = [f"valid-artifact census: {census.summary()}", "opcodes:"]
    lines.extend(f"  {name:12s} {count:4d}"
                 for name, count in sorted(by_opcode.items()))
    lines.append("flags:")
    lines.extend(f"  {name:14s} {by_flag[name]:4d}" for name in FLAG_NAMES.values())
    gaps = census.declared_coverage_gaps()
    lines.append("declared coverage: " + ("complete" if not gaps else
                                          "MISSING " + ", ".join(gaps)))
    return "\n".join(lines)


if __name__ == "__main__":
    from executorch.backends.esp32s3.test import firmware

    found = scan_embedded_artifacts(str(firmware.MAIN))
    print(render_details(found))
    raise SystemExit(bool(found.declared_coverage_gaps()))
