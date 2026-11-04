# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How much PSRAM a program's weights are worth: the weight-tier report.

    python -m executorch.backends.esp32s3.utils.weight_tiers model.pte
        [--budget BYTES|NM ...] [--load decode,prefill] [--per decode]
        [--tables] [--auto [--reserve BYTES]] [--psram BYTES]
        [--board papercolor] [--cores 2] [--caches story] [--json FILE]

The backend copies external weights from flash into PSRAM within a budget
(runtime/esp32_backend.cpp: the weight-tier options described with
kOptWeightTierBudget). This reads a lowered program's delegate blobs and says,
before a board is involved:

- per weight: its format class, the bytes a copy takes (the budget's unit),
  its score (the passes one execute makes over it: bytes read over bytes
  stored), and the time a copy saves per execute at the board's read rates
  (target/chips.py MEASURED);
- per budget: the bytes copied in load order (what the runtime does), ranked
  (highest score first, then format class, then size: the order eviction
  spares, and the ranked placement), and at best (the most score-weighted
  bytes any set of copies fits: an exact subset sum when every score is
  equal, as in the story, else a knapsack), with the time each saves.

The arithmetic is the runtime's, function for function, so the copies it
predicts for a budget are the ones the backend makes: weight_bytes,
scale_bytes, staged_scale_offset, weight_class, weight_score and
tier_candidate; weight_source's names digest; and the cache itself, in
CacheModel (tier_resolve, tier_hit, weight_cache_newcomer, weight_cache_victim, the
table's growth in PSRAM, release_resources). test_weight_tiers_report.py holds
the prediction to the host runner's own report on fixtures.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from typing import Optional

from executorch.backends.esp32s3.serialization import espressif_blob as eb
from executorch.backends.esp32s3.target import chips

# ESP32_TIER_* (runtime/esp32_weight_tiers.h): the format classes, in the
# order a tie in score is broken.
CLASSES = ("int4", "int8", "fp32", "bf16", "int16", "fp16")
# The loop each class is read by, as chips.py names kernels.
KERNEL = {"int4": "i4", "int8": "i8", "fp32": "f32", "bf16": "bf16",
          "int16": "i16", "fp16": "f16"}
# int4 and int8 rows are bound by memory in flash and PSRAM (the fused int4
# kernel reads internal SRAM at 290 MB/s on two cores against PSRAM's 91), so
# where their kernel was never timed from a tier, the tier's word-loop rate
# stands in, and the report says so. The float and int16 loops are bound by
# their arithmetic (fp16's widening costs about 29 instructions a multiply-add:
# 24.8 MB/s from flash, 26.3 from PSRAM), so a word-loop rate would claim a
# saving that is not there: only their own kernel's rows count.
STAND_IN = {"int4": ("i4", "seq"), "int8": ("i8", "seq")}

HALF = (eb.OP_ADDMM_F16, eb.OP_ADDMM_BF16, eb.OP_EMBED_F16, eb.OP_EMBED_BF16)
FLOAT = (eb.OP_ADDMM,) + HALF
EMBED = (eb.OP_EMBED_I8, eb.OP_EMBED_I4, eb.OP_EMBED_I16, eb.OP_EMBED_F16,
         eb.OP_EMBED_BF16)
BMM = (eb.OP_BMM_DYN_I8, eb.OP_BMM_KCACHE_I8, eb.OP_BMM_KCACHE_AFFINE_I8)

# esp32_backend.cpp's constants.
VEC_BYTES = 16                  # kVecBytes: a copy's alignment slack is 15 B
SCORE_SHIFT = 16                # kScoreShift: 1 << 16 is one pass
SCORE_ONE = 1 << SCORE_SHIFT
UINT64_MAX = (1 << 64) - 1
WIDEN_BYTES = 16 * 1024         # esp32_float.h kWidenBytes
CACHE_FIRST = 32                # kWeightCacheFirst: the table's first capacity
CACHE_MAX = 0xFFFF              # kWeightCacheMax: the uint16 per-op index
# sizeof(CachedWeight): the table is allocated in PSRAM first, so it is charged
# there. Ten pointer- or 8-byte fields, refs, kind, cls and two 16-byte samples.
CACHE_ENTRY_BYTES = {8: 120, 4: 104}   # host (LP64), chip (ILP32)


# ---------------------------------------------------- the runtime's arithmetic


def row_bytes(opcode: int, k_pad: int) -> int:
    if opcode in (eb.OP_ADDMM_I4, eb.OP_EMBED_I4):
        return k_pad // 2
    if opcode in (eb.OP_ADDMM_I16, eb.OP_EMBED_I16):
        return k_pad * 2
    return k_pad


def weight_bytes(d: dict) -> int:
    """Bytes of an op's stored weights (esp32_backend.cpp weight_bytes)."""
    if d["opcode"] in HALF:
        return d["k"] * d["n"] * 2
    if d["opcode"] == eb.OP_ADDMM:
        return d["k"] * d["n"] * 4
    return d["n"] * row_bytes(d["opcode"], d["k_pad"])


def scale_bytes(d: dict) -> int:
    """Bytes of an op's scale table (scale_bytes); float weights have none."""
    if d["opcode"] in FLOAT:
        return 0
    span = d["k"] if d["opcode"] in EMBED else d["k_pad"]
    group = d["scale_group"] or span
    width = 2 if d["flags"] & eb.FLAG_FP16_SCALES else 4
    return d["n"] * -(-span // group) * width


def staged_scale_offset(wbytes: int) -> int:
    return (wbytes + 3) & ~3


def weight_class(d: dict) -> str:
    """The format a copy is read in (weight_class): an int4 head that
    FLAG_STAGE_I8 expands is int8."""
    if d["flags"] & eb.FLAG_STAGE_I8:
        return "int8"
    return {eb.OP_ADDMM_I4: "int4", eb.OP_EMBED_I4: "int4",
            eb.OP_ADDMM_I8: "int8", eb.OP_EMBED_I8: "int8",
            eb.OP_ADDMM: "fp32", eb.OP_ADDMM_BF16: "bf16",
            eb.OP_EMBED_BF16: "bf16", eb.OP_ADDMM_I16: "int16",
            eb.OP_EMBED_I16: "int16"}.get(d["opcode"], "fp16")


def score_ratio(reads: int, stored: int) -> int:
    stored = stored or 1
    return UINT64_MAX if reads > (UINT64_MAX >> SCORE_SHIFT) else (
        (reads << SCORE_SHIFT) // stored)


def widen_block(m: int, k: int, n: int) -> int:
    """esp32_float::widen_block: rows of a 2-byte weight widened once for
    all m activation rows, or 0 when the loop reads it row by row."""
    row = k * 4
    if m < 2 or row == 0 or row > WIDEN_BYTES:
        return 0
    return min(WIDEN_BYTES // row, n)


def weight_score(blob: eb.Blob, i: int) -> int:
    """What a copy of op i's weights saves per byte it occupies, in 1/65536ths
    (weight_score): the passes one execute makes over a matmul's or conv's
    weight, as its loop issues them, or a table's rows read over rows
    stored."""
    from executorch.backends.esp32s3.operator_support.conv import runtime_block

    d = blob.descs[i]
    m = d["m"] or 1
    if d["opcode"] in EMBED:
        return score_ratio(m, d["n"])
    selected = blob.selected(i)
    if selected is not None:
        return score_ratio(m * selected[1], d["n"])
    geom = blob.geometry(i)
    act = blob.act_quant(i)
    if d["opcode"] in (eb.OP_ADDMM_F16, eb.OP_ADDMM_BF16):
        passes = 1 if widen_block(d["m"], d["k"], d["n"]) else m
    elif (d["opcode"] in (eb.OP_ADDMM, eb.OP_ADDMM_I16)
          or (geom is not None and geom.depthwise)
          or (act is not None and act["code_bits"] == 16)):
        passes = m
    else:
        block = runtime_block(d["m"], d["k_pad"], d["n"], conv=geom is not None)
        passes = -(-m // block)
    return score_ratio(passes, 1)


def fnv1a(h: int, data: bytes) -> int:
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & UINT64_MAX
    return h


FNV_BASIS = 0xCBF29CE484222325


# ------------------------------------------------------------- the program


@dataclass(frozen=True)
class TierOp:
    """One op's external weight as the runtime's cache sees it."""

    method: str
    delegate: int      # its delegate's place in the method's delegate list
    op: int            # its place in the blob
    opcode: int
    cls: str
    kind: str          # "stored", or "i4->i8" for FLAG_STAGE_I8's expansion
    names: int         # the cache's digest of its keys (weight_source)
    wkey: str
    skey: Optional[str]
    wbytes: int        # weight bytes in a copy
    sbytes: int        # scale bytes in a copy
    src_bytes: int     # weight bytes at the source
    need: int          # bytes a copy takes: the budget's unit
    score: int         # weight_score, 1/65536ths of a pass
    table: bool        # an embedding table or a selected-row head
    must: bool         # FLAG_STAGE_I8: copied whatever the budget
    calls: int = 1     # how often one execute of the method calls the delegate

    @property
    def identity(self) -> tuple:
        """What the cache matches an entry by, less the flash addresses that
        follow from the keys: the copy's kind, the keys and the sizes."""
        return (self.kind, self.names, self.wbytes, self.wbytes + self.sbytes)

    @property
    def reads(self) -> float:
        """Bytes one execute of the method reads from this weight: its score
        times the bytes stored, once per call of its delegate."""
        return self.score * (self.wbytes + self.sbytes) * self.calls / SCORE_ONE


def blob_ops(blob: eb.Blob, method: str = "", delegate: int = 0,
             calls: int = 1) -> list:
    """The ops of one blob whose weights the cache can hold: external stored
    weights (opcodes 1-7 and 11-14), as TierOps."""
    ops = []
    for i, d in enumerate(blob.descs):
        if not d["flags"] & eb.FLAG_EXTERNAL_WEIGHTS or d["opcode"] in BMM:
            continue
        floats = d["opcode"] in FLOAT
        wkey = blob.key(d["wt_off"])
        skey = None if floats else blob.key(d["scale_off"])
        names = fnv1a(FNV_BASIS, wkey + b"\0")
        if skey is not None:
            names = fnv1a(names, skey + b"\0")
        must = bool(d["flags"] & eb.FLAG_STAGE_I8)
        src = weight_bytes(d)
        wbytes = d["n"] * d["k_pad"] if must else src
        affine = blob.weight_affine(i)
        if must:
            # An expansion is laid out by its op's rows, row length and scale
            # groups, so its name carries them (weight_source).
            names = fnv1a(names, struct.pack("<4I", d["n"], d["k"], d["k_pad"],
                                             d["scale_group"]))
        if must and affine is not None:
            # An expansion's bytes carry its folded zero points, so its name
            # carries what decides them (weight_source).
            names = fnv1a(names, struct.pack("<I", affine["zp_kind"]))
            names = fnv1a(names, struct.pack("<i", affine["storage_bias"]))
            if affine["zp_kind"] != eb.ZP_NONE:
                names = fnv1a(names, blob.key(affine["zp_off"]) + b"\0")
        sbytes = scale_bytes(d)
        ops.append(TierOp(
            method=method, delegate=delegate, op=i, opcode=d["opcode"],
            cls=weight_class(d), kind="i4->i8" if must else "stored",
            names=names, wkey=wkey.decode(),
            skey=None if skey is None else skey.decode(),
            wbytes=wbytes, sbytes=sbytes, src_bytes=src,
            need=staged_scale_offset(wbytes) + sbytes + VEC_BYTES - 1,
            score=weight_score(blob, i),
            table=d["opcode"] in EMBED or blob.selected(i) is not None,
            must=must, calls=calls))
    return ops


@dataclass
class Program:
    """A lowered program as the weight tiers see it: per method, its
    delegates in the order Method::init initializes them, each the ops whose
    weights the cache can hold."""

    methods: dict                 # name -> [[TierOp, ...] per Esp32 delegate]
    source: str = "flatbuffer"    # or "scan": blobs found by their magic

    def ops(self, methods=None):
        for name in dict.fromkeys(methods or self.methods):
            for delegate in self.methods[name]:
                yield from delegate

    def weights(self, methods=None) -> list:
        """Every distinct weight, in the order the methods first resolve
        them, with its users across the methods."""
        found = {}
        for op in self.ops(methods):
            weight = found.get(op.identity)
            if weight is None:
                found[op.identity] = Weight(op)
            else:
                weight.add(op)
        return list(found.values())


@dataclass
class Weight:
    """One distinct weight across the program: what the cache gives one
    entry. Its score is the highest any user resolves it at, as the entry
    keeps."""

    first: TierOp
    users: list = field(default_factory=list)

    def __post_init__(self):
        self.users = [self.first]

    def add(self, op: TierOp):
        self.users.append(op)

    identity = property(lambda self: self.first.identity)
    cls = property(lambda self: self.first.cls)
    kind = property(lambda self: self.first.kind)
    need = property(lambda self: self.first.need)
    names = property(lambda self: self.first.names)
    wkey = property(lambda self: self.first.wkey)
    table = property(lambda self: self.first.table)
    must = property(lambda self: self.first.must)
    score = property(lambda self: max(op.score for op in self.users))
    methods = property(lambda self: tuple(dict.fromkeys(
        op.method for op in self.users)))

    def reads(self, method: str) -> float:
        """Bytes one execute of `method` reads from this weight."""
        return sum(op.reads for op in self.users if op.method == method)

    def candidate(self, tables=False, auto=False) -> bool:
        """Whether a budget may copy it (tier_candidate): not an expansion,
        a table only with weight_tier_tables, and fp16 not under auto."""
        return (not self.must and (tables or not self.table)
                and not (auto and self.cls == "fp16"))


def _calls(plan) -> dict:
    """{delegate index: DelegateCall instructions} over a plan's chains."""
    from executorch.exir.schema import DelegateCall

    calls = {}
    for chain in plan.chains:
        for instruction in chain.instructions:
            args = instruction.instr_args
            if isinstance(args, DelegateCall):
                calls[args.delegate_index] = calls.get(args.delegate_index, 0) + 1
    return calls


def read_program(pte: bytes) -> Program:
    """The program's Esp32Backend delegates per method, from its flatbuffer:
    each method's delegate list in order, each delegate's blob from the
    program's delegate data. A program the deserializer cannot read is
    scanned for blobs instead, as one method in file order."""
    try:
        from executorch.exir._serialize._program import deserialize_pte_binary

        program = deserialize_pte_binary(pte).program
    except Exception:  # noqa: BLE001 - any container this reader cannot parse
        return scan_program(pte)
    methods = {}
    for plan in program.execution_plan:
        calls = _calls(plan)
        delegates = []
        for index, delegate in enumerate(plan.delegates):
            if delegate.id != eb.BACKEND_NAME:
                continue
            data = program.backend_delegate_data[delegate.processed.index].data
            blob = eb.read_blob(data)
            if blob is None:
                raise ValueError(f"{plan.name}: delegate {index} is not a "
                                 f"version-{eb.VERSION} {eb.BACKEND_NAME} blob")
            delegates.append(blob_ops(blob, plan.name, index,
                                      calls.get(index, 1)))
        methods[plan.name] = delegates
    return Program(methods)


def scan_program(data: bytes, name: str = "program") -> Program:
    """Every blob in `data` found by its magic, as the delegates of one
    method in file order: a packed image of programs, or a container the
    deserializer cannot read."""
    delegates = [blob_ops(blob, name, index)
                 for index, blob in enumerate(eb.scan_blobs(data))]
    return Program({name: delegates}, source="scan")


# ------------------------------------------------------------- the cache


class TierInitError(RuntimeError):
    """An init the runtime fails: an expansion (FLAG_STAGE_I8) that PSRAM
    cannot hold, or a cache table that cannot grow for one."""


@dataclass
class Entry:
    identity: tuple
    cls: str
    kind: str
    names: int
    alloc: int
    score: int = 0
    raw: bool = False
    refs: int = 0

    @property
    def unused(self) -> bool:
        return not self.raw and self.refs == 0


def worth_less(a: Entry, b: Entry) -> bool:
    """esp32_backend.cpp worth_less: a lower score, then a later format
    class, then fewer bytes."""
    if a.score != b.score:
        return a.score < b.score
    if a.cls != b.cls:
        return CLASSES.index(a.cls) > CLASSES.index(b.cls)
    return a.alloc < b.alloc


class CacheModel:
    """The backend's weight cache, step for step: what each method load
    copies, reuses, evicts and leaves on flash under the weight-tier options.

    budget       weight_tier_budget in bytes (0: nothing is resolved), or
                 "auto" (-1): every candidate while `reserve` bytes of PSRAM
                 stay free
    reserve      weight_tier_reserve
    tables       weight_tier_tables
    cache        weight_cache: keep copies no loaded method holds
    psram        the PSRAM the copies and the cache's table are allocated
                 from, as one block; None for unlimited (the host runner's
                 default; --psram-bytes sets it)
    pointer      bytes in a pointer, which sizes the table's entries: 8 on the
                 host, 4 on the chip
    """

    def __init__(self, budget=0, reserve=0, tables=False, cache=True,
                 psram=None, pointer=4):
        self.auto = budget == "auto"
        self.budget = 0 if self.auto else int(budget)
        self.reserve = reserve
        self.tables = tables
        self.cache = cache
        self.psram = psram
        self.entry_bytes = CACHE_ENTRY_BYTES[pointer]
        self.entries = []        # [0, n) used; an unused one is free
        self.cap = 0
        self.table_psram = 0     # bytes of PSRAM the table holds
        self.used = 0            # PSRAM charged: copies and the table
        self.resident = 0
        self.copies = self.hits = self.evictions = 0
        self.skipped = self.expanded = 0

    # -- PSRAM, one block as the host's shim keeps it
    def _malloc(self, size: int) -> bool:
        if self.psram is not None and (self.used > self.psram
                                       or size > self.psram - self.used):
            return False
        self.used += size
        return True

    def _free(self, size: int):
        self.used -= size

    def _free_size(self) -> int:
        return UINT64_MAX if self.psram is None else max(0, self.psram - self.used)

    def limit(self) -> int:
        """weight_tier_limit: the budget, or under auto what is resident
        plus what PSRAM still has beyond the reserve."""
        if not self.auto:
            return self.budget
        free = self._free_size()
        return self.resident + (free - self.reserve if free > self.reserve else 0)

    # -- the table
    def _drop(self, index: int):
        entry = self.entries[index]
        if entry.raw:
            self._free(entry.alloc)
            self.resident -= entry.alloc
        self.entries[index] = Entry((), "", "", 0, 0)

    def _find(self, op: TierOp):
        for index, entry in enumerate(self.entries):
            if not entry.unused and entry.identity == op.identity:
                return index
        return None

    def _slot(self):
        """weight_cache_slot: a free entry, else the next, else the table
        grown (PSRAM first, then internal memory, which is not charged)."""
        for index, entry in enumerate(self.entries):
            if entry.unused:
                return index
        if len(self.entries) == self.cap:
            if self.cap >= CACHE_MAX:
                return None
            cap = CACHE_FIRST if self.cap == 0 else (
                CACHE_MAX if self.cap > CACHE_MAX // 2 else 2 * self.cap)
            size = cap * self.entry_bytes
            grown = size if self._malloc(size) else 0
            if self.table_psram:
                self._free(self.table_psram)
            self.table_psram = grown
            self.cap = cap
        self.entries.append(Entry((), "", "", 0, 0))
        return len(self.entries) - 1

    def _victim(self):
        """weight_cache_victim: the unused copy worth least."""
        victim = None
        for index, entry in enumerate(self.entries):
            if not entry.raw or entry.refs != 0:
                continue
            if victim is None or worth_less(entry, self.entries[victim]):
                victim = index
        return victim

    def _evictable(self) -> int:
        """weight_cache_evictable: the bytes the unused copies hold."""
        return sum(e.alloc for e in self.entries if e.raw and e.refs == 0)

    def _newcomer(self, need: int) -> bool:
        """weight_cache_newcomer: PSRAM for a newcomer within the budget.
        Unused copies of any score make room, least worth first: none when
        evicting all of them would not bring it under the budget or free
        its PSRAM; the allocation first, and when it fails unused copies
        evicted until it succeeds (several smaller ones together if need
        be); then evictions for the budget alone."""
        limit = self.limit()
        if need > limit:
            return False
        if (self.resident + need > limit
                and self.resident - self._evictable() > limit - need):
            return False
        if not self._malloc(need):
            if self._free_size() + self._evictable() < need:
                return False
            if not self._alloc_evicting(need):
                return False
        while self.resident + need > limit:
            self._drop(self._victim())
            self.evictions += 1
        return True

    def _alloc_evicting(self, need: int) -> bool:
        while True:
            if self._malloc(need):
                return True
            victim = self._victim()
            if victim is None:
                return False
            self._drop(victim)
            self.evictions += 1

    def _resolve(self, op: TierOp):
        """tier_resolve: op's entry, found or made, and the copy it has or
        now gets. The entry's index, or None when the table could not take
        a new entry (the op reads flash)."""
        index = self._find(op)
        if index is None:
            index = self._slot()
            if index is None:
                if op.must:
                    raise TierInitError(f"no weight cache entry for {op.wkey}")
                self.skipped += 1
                return None
            self.entries[index] = Entry(op.identity, op.cls, op.kind, op.names,
                                        op.need)
        entry = self.entries[index]
        entry.refs += 1
        entry.score = max(entry.score, op.score)
        if entry.raw:
            self.hits += 1
            return index
        raw = self._newcomer(op.need)
        if not raw and op.must:
            raw = self._alloc_evicting(op.need)
        if not raw:
            if op.must:
                entry.refs -= 1
                if entry.refs == 0:
                    self._drop(index)
                raise TierInitError(
                    f"the int8 expansion of {op.wkey} needs {op.need:,} B of "
                    f"PSRAM")
            self.skipped += 1
            return index
        entry.raw = True
        self.resident += op.need
        self.copies += 1
        self.expanded += op.must
        return index

    def load(self, delegates) -> list:
        """One method's init: each delegate takes its hits first (every op
        whose weight already has a copy), then stages its expansions, then
        (with a budget) its candidates, in op order. A handle for unload()."""
        resolved = []
        candidates = self.auto or self.budget > 0
        for ops in delegates:
            held = []
            if candidates or any(op.must for op in ops):
                staged = [op.must or (candidates and (self.tables or not op.table)
                                      and not (self.auto and op.cls == "fp16"))
                          for op in ops]
                done = [False] * len(ops)
                for j, op in enumerate(ops):
                    index = self._find(op) if staged[j] else None
                    if index is not None and self.entries[index].raw:
                        held.append(self._resolve(op))
                        done[j] = True
                for j, op in enumerate(ops):
                    if op.must and not done[j]:
                        held.append(self._resolve(op))
                for j, op in enumerate(ops):
                    if staged[j] and not op.must and not done[j]:
                        held.append(self._resolve(op))
            resolved.append([index for index in held if index is not None])
        return resolved

    def unload(self, handle: list):
        """release_resources for each of a method's delegates: an entry
        without a copy goes with its last user, and so does a copy with the
        cache off, and an expansion when no budget is set."""
        budgeted = self.auto or self.budget > 0
        for held in handle:
            for index in held:
                entry = self.entries[index]
                if entry.refs > 0:
                    entry.refs -= 1
                if entry.refs == 0 and (
                        not entry.raw or not self.cache
                        or (entry.kind == "i4->i8" and not budgeted)):
                    self._drop(index)

    def staged(self) -> list:
        """The entries with a copy: what the host runner prints as psram."""
        return [e for e in self.entries if e.raw]

    def known(self) -> list:
        """Every entry in use: the host runner's "esp32 tier:" lines."""
        return [e for e in self.entries if not e.unused]


def table_bytes(entries: int, pointer: int = 4) -> int:
    """The PSRAM the cache's table holds once it has `entries` entries: its
    capacity (32, doubling) times an entry's size."""
    if entries <= 0:
        return 0
    cap = CACHE_FIRST
    while cap < entries and cap < CACHE_MAX:
        cap = CACHE_MAX if cap > CACHE_MAX // 2 else 2 * cap
    return cap * CACHE_ENTRY_BYTES[pointer]


def predict(program: Program, sequence=None, **options) -> CacheModel:
    """The cache after loading `sequence`'s methods one at a time (each
    destroyed before the next, as the host runner's script and the story
    firmware load them; a method named twice in a row stays loaded), the last
    left loaded. The default is every method once, in program order."""
    cache = CacheModel(**options)
    handle, loaded = None, None
    for name in sequence or list(program.methods):
        if name == loaded:
            continue  # still loaded: the runner reloads only on a change
        if handle is not None:
            cache.unload(handle)
        handle, loaded = cache.load(program.methods[name]), name
    return cache


# ------------------------------------------------------- placement policies


def ranked(weights) -> list:
    """Highest worth first: score, then the earlier format class, then the
    larger copy -- the reverse of the order eviction drops copies in, and the
    order the ranked placement packs."""
    return sorted(weights, key=lambda w: (-w.score, CLASSES.index(w.cls),
                                          -w.need))


def first_fit(order, room: int) -> list:
    """Each weight in turn that still fits in `room` bytes."""
    staged, used = [], 0
    for weight in order:
        if used + weight.need <= room:
            staged.append(weight)
            used += weight.need
    return staged


def best_sets(weights, rooms, memory: int = 256 << 20) -> tuple:
    """(sets, unit): for each room, a subset of `weights` with the most
    score-weighted bytes (score x need) that fits in it. An exact subset sum
    of the sizes when every score is equal, as in the story; otherwise a 0/1
    knapsack over sizes rounded up to `unit` bytes -- exact when unit is 1,
    and otherwise a set that fits, within unit bytes per weight of the best.
    The tables that recover the sets are kept within `memory` bytes; unit
    grows to keep them there."""
    rooms = [max(0, r) for r in rooms]
    top = max(rooms, default=0)
    if not weights or top == 0:
        return [[] for _ in rooms], 1
    n = len(weights)
    if len({w.score for w in weights}) == 1 and n * (top + 1) // 8 <= memory:
        reach, mask, before = 1, (1 << (top + 1)) - 1, []
        for weight in weights:
            before.append(reach)
            reach = (reach | (reach << weight.need)) & mask
        sets = []
        for room in rooms:
            target = (reach & ((1 << (room + 1)) - 1)).bit_length() - 1
            chosen = []
            for k in range(n - 1, -1, -1):
                if not (before[k] >> target) & 1:
                    chosen.append(weights[k])
                    target -= weights[k].need
            sets.append(chosen[::-1])
        return sets, 1
    import numpy as np

    unit = max(1, math.ceil(top / (1 << 20)), math.ceil(n * (top + 1) / memory))
    cells = top // unit
    best = np.zeros(cells + 1, dtype=np.int64)
    take = np.zeros((n, cells + 1), dtype=bool)
    sizes = [-(-w.need // unit) for w in weights]
    for k, (weight, size) in enumerate(zip(weights, sizes)):
        if size > cells:
            continue
        value = min(weight.score * weight.need, (1 << 62) // n)
        offered = best[:cells + 1 - size] + value
        better = offered > best[size:]
        take[k, size:] = better
        best[size:] = np.where(better, offered, best[size:])
    sets = []
    for room in rooms:
        cell, chosen = room // unit, []
        for k in range(n - 1, -1, -1):
            if take[k, cell]:
                chosen.append(weights[k])
                cell -= sizes[k]
        sets.append(chosen[::-1])
    return sets, unit


# ------------------------------------------------------------------ rates


@dataclass(frozen=True)
class ClassRate:
    """What a byte of one class costs from flash and from PSRAM, with the
    keys the rates were measured at (target/chips.py)."""

    cls: str
    flash: Optional[chips.Rate]
    flash_key: Optional[chips.RateKey]
    psram: Optional[chips.Rate]
    psram_key: Optional[chips.RateKey]

    @property
    def saved_us_per_byte(self) -> Optional[float]:
        if self.flash is None or self.psram is None:
            return None
        return 1.0 / self.flash.mb_s - 1.0 / self.psram.mb_s


def class_rates(board="papercolor", cores=2, caches="story", mspi=None) -> dict:
    """{class: ClassRate} at the nearest measured keys: the class's own
    kernel, else (int4 and int8 only, STAND_IN) the tier's word loop."""
    out = {}
    for cls in CLASSES:
        found = {}
        for tier in ("flash", "psram"):
            rate, key = None, None
            for kernel in STAND_IN.get(cls, (KERNEL[cls],)):
                rate, key = chips.nearest_rate(board, tier, kernel, cores,
                                               caches, mspi)
                if key is not None and key.kernel != kernel:
                    rate, key = None, None
                if rate is not None:
                    break
            found[tier] = (rate, key)
        out[cls] = ClassRate(cls, *found["flash"], *found["psram"])
    return out


def saved_ms(weights, method: str, rates: dict):
    """(ms one execute of `method` saves with these weights copied, whether
    every class had rates): unmeasured classes add nothing to the sum."""
    total, known = 0.0, True
    for weight in weights:
        per_byte = rates[weight.cls].saved_us_per_byte
        if per_byte is None:
            known = known and weight.reads(method) == 0
            continue
        total += weight.reads(method) * per_byte / 1000.0
    return total, known


# ------------------------------------------------------------------ report


def nice_step(total: int) -> int:
    """A power-of-two step, at least 1 KiB, giving at most 16 budgets up to
    `total`."""
    step = 1024
    while total / step > 16:
        step *= 2
    return step


@dataclass
class Report:
    """Everything the report prints, as data (as_json() for a file)."""

    program: Program
    sequence: list
    per: str
    tables: bool
    auto: bool
    rates: dict
    weights: list = field(default_factory=list)      # candidates, ranked
    expansions: list = field(default_factory=list)   # copied whatever the budget
    left_out: list = field(default_factory=list)     # tables (or fp16 under auto)
    curve: list = field(default_factory=list)
    unit: int = 1

    @property
    def wanted(self) -> int:
        return sum(w.need for w in self.weights)

    @property
    def bytes_that_pay(self) -> int:
        """The budget past which a copy saves nothing more: every weight
        auto would copy (no tables, no fp16), plus the expansions."""
        return sum(w.need for w in self.weights if w.cls != "fp16") + sum(
            w.need for w in self.expansions)

    def by_class(self, weights) -> dict:
        out = {}
        for weight in weights:
            out[weight.cls] = out.get(weight.cls, 0) + weight.need
        return out

    def as_json(self) -> dict:
        def weight_row(w):
            return {"key": w.wkey, "class": w.cls, "kind": w.kind,
                    "bytes": w.need, "score": w.score / SCORE_ONE,
                    "methods": list(w.methods), "users": len(w.users),
                    "reads": w.reads(self.per), "names": f"{w.names:016x}"}

        return {
            "methods": list(self.program.methods), "sequence": self.sequence,
            "per": self.per, "tables": self.tables, "auto": self.auto,
            "bytes_that_pay": self.bytes_that_pay, "wanted": self.wanted,
            "by_class": self.by_class(self.weights),
            "expansions": [weight_row(w) for w in self.expansions],
            "left_out": [weight_row(w) for w in self.left_out],
            "weights": [weight_row(w) for w in self.weights],
            "curve": self.curve, "best_unit": self.unit,
            "rates": {cls: {"flash": r.flash and r.flash.mb_s,
                            "psram": r.psram and r.psram.mb_s,
                            "flash_key": r.flash_key and r.flash_key.__dict__,
                            "psram_key": r.psram_key and r.psram_key.__dict__}
                      for cls, r in self.rates.items()},
        }


def report(program: Program, budgets=None, sequence=None, per=None,
           tables=False, auto=False, reserve=0, psram=None, rates=None,
           pointer=4) -> Report:
    """The weight-tier report for `program`: its weights, and for each budget
    (bytes; default: a step from 0 to every candidate), the copies in load
    order (CacheModel over `sequence`), ranked and at best, with the time each
    saves per execute of `per` (default: the last method loaded). Under
    `auto` a budget is the PSRAM the program finds free at load, `reserve` of
    which stays free."""
    sequence = list(sequence or program.methods)
    per = per or sequence[-1]
    rates = rates or class_rates()
    every = program.weights(sequence)
    candidates = [w for w in every if w.candidate(tables, auto)]
    out = Report(program, sequence, per, tables, auto, rates,
                 weights=ranked(candidates),
                 expansions=[w for w in every if w.must],
                 left_out=[w for w in every if not w.must
                           and not w.candidate(tables, auto)])
    must = sum(w.need for w in out.expansions)
    table = table_bytes(len(candidates) + len(out.expansions), pointer)
    if budgets is None:
        top = out.wanted + must + (reserve + table if auto else 0)
        step = nice_step(top)
        budgets = list(range(0, top + step, step)) if top else [0]

    def room(budget):
        # What the candidates may take: under auto the budget is the PSRAM
        # the program finds free, less the reserve and the cache's table;
        # with a PSRAM size, the budget, or that PSRAM less the table.
        if auto:
            return budget - reserve - table - must
        if psram is not None:
            return min(budget, psram - table) - must
        return budget - must

    rooms = [room(b) for b in budgets]
    best, out.unit = best_sets(candidates, rooms)
    by_identity = {w.identity: w for w in every}
    for budget, room, chosen in zip(budgets, rooms, best):
        cache = predict(program, sequence, reserve=reserve, tables=tables,
                        pointer=pointer, **(
                            {"budget": "auto", "psram": budget} if auto else
                            {"budget": budget, "psram": psram}))
        order = [by_identity[e.identity] for e in cache.staged()]
        rank = out.expansions + first_fit(out.weights, room)
        row = {"budget": budget}
        for name, staged in (("load_order", order), ("ranked", rank),
                             ("best", out.expansions + chosen)):
            ms, known = saved_ms(staged, per, rates)
            row[name] = {"bytes": sum(w.need for w in staged),
                         "value": sum(w.score * w.need for w in staged
                                      if not w.must) / SCORE_ONE,
                         "ms": ms, "ms_complete": known}
        out.curve.append(row)
    return out


def _fmt_bytes(n) -> str:
    return "-" if n is None else f"{n:,} B"


def _fmt_budget(n: int) -> str:
    if n and n % (1 << 20) == 0:
        return f"{n >> 20} MiB"
    if n and n % 1024 == 0:
        return f"{n >> 10} KiB"
    return f"{n:,} B"


def _fmt_ms(ms, complete=True) -> str:
    if ms is None:
        return "-"
    return f"{ms:.2f} ms" + ("" if complete else "+?")


def _fmt_rate(rate, key) -> str:
    if rate is None:
        return "not measured"
    return (f"{rate.mb_s:g} MB/s [{key.kernel}, {key.cores} core"
            f"{'s' if key.cores > 1 else ''}, {key.caches} caches"
            f"{', mspi ' + key.mspi if key.mspi else ''}]")


def render(rep: Report, rows: int = 20) -> str:
    """The report as text: the weights, the rates, the curve."""
    lines = []
    methods = ", ".join(rep.program.methods)
    lines.append(f"weight tiers: methods {methods}; loaded in turn: "
                 f"{', '.join(rep.sequence)}; time per execute of {rep.per}")
    if rep.program.source == "scan":
        lines.append("  (the program's flatbuffer did not parse: its blobs were found "
                     "by their magic, as one method in file order)")
    classes = ", ".join(f"{c} {b:,} B" for c, b in rep.by_class(rep.weights).items())
    lines.append(f"  {len(rep.weights)} weight(s) a budget may copy: "
                 f"{rep.wanted:,} B ({classes or 'none'})")
    lines.append(f"  bytes that pay: {rep.bytes_that_pay:,} B")
    if rep.expansions:
        lines.append(f"  copied whatever the budget (int4 heads expanded to "
                     f"int8): {len(rep.expansions)}, "
                     f"{sum(w.need for w in rep.expansions):,} B, counted first")
    if rep.left_out:
        why = "tables (weight_tier_tables)" + (", fp16 (auto)" if rep.auto else "")
        lines.append(f"  left on flash: {len(rep.left_out)} weight(s), "
                     f"{sum(w.need for w in rep.left_out):,} B: {why}")
    used = {w.cls for w in rep.weights + rep.expansions}
    for cls in CLASSES:
        if cls not in used:
            continue
        r = rep.rates[cls]
        per_byte = r.saved_us_per_byte
        lines.append(f"  {cls}: flash {_fmt_rate(r.flash, r.flash_key)}; psram "
                     f"{_fmt_rate(r.psram, r.psram_key)}"
                     + (f"; a copy saves {per_byte * 1000:.1f} ns a byte read"
                        if per_byte is not None else "; no time estimate"))
    missing = chips.unmeasured()
    if missing:
        lines.append("  rates still to measure: " + ", ".join(
            f"{k.tier}/{k.kernel}/{k.cores}c" for k in missing))
    lines.append("")
    lines.append(f"per weight, highest worth first ({min(rows, len(rep.weights))} "
                 f"of {len(rep.weights)}):")
    lines.append(f"  {'#':>4} {'class':6} {'bytes':>12} {'score':>7} "
                 f"{'users':>5} {'saves':>10}  key, methods")
    for index, w in enumerate(rep.weights[:rows]):
        ms, known = saved_ms([w], rep.per, rep.rates)
        lines.append(f"  {index + 1:>4} {w.cls:6} {w.need:>12,} "
                     f"{w.score / SCORE_ONE:>7.2f} {len(w.users):>5} "
                     f"{_fmt_ms(ms, known):>10}  {w.wkey}, {'+'.join(w.methods)}")
    lines.append("")
    best_note = ("exact subset sum" if rep.unit == 1 and len(
        {w.score for w in rep.weights}) <= 1 else
        "knapsack" + (f" in {rep.unit}-byte units" if rep.unit > 1 else ""))
    lines.append(f"budget curve (time per execute of {rep.per}; best: {best_note}):")
    lines.append(f"  {'budget':>10}  {'load order':>26}  {'ranked':>26}  "
                 f"{'best':>26}")
    for row in rep.curve:
        cells = []
        for name in ("load_order", "ranked", "best"):
            cell = row[name]
            cells.append(f"{_fmt_bytes(cell['bytes']):>15} "
                         f"{_fmt_ms(cell['ms'], cell.get('ms_complete', True)):>10}")
        lines.append(f"  {_fmt_budget(row['budget']):>10}  " + "  ".join(cells))
    return "\n".join(lines)


def parse_bytes(text: str) -> int:
    """'4194304', '4.5M' (MiB) or '512K' (KiB)."""
    text = text.strip()
    scale = {"K": 1 << 10, "M": 1 << 20}.get(text[-1:].upper())
    return round(float(text[:-1]) * scale) if scale else int(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pte", help="a lowered program (.pte)")
    parser.add_argument("--budget", action="append", type=parse_bytes,
                        help="a budget to report (bytes, or 4.5M / 512K); "
                             "repeatable; default: a curve from 0")
    parser.add_argument("--load", help="the methods loaded, in turn, comma "
                        "separated (default: every method in program order)")
    parser.add_argument("--per", help="the method whose execute the times are "
                        "per (default: the last loaded)")
    parser.add_argument("--tables", action="store_true",
                        help="as weight_tier_tables: tables are candidates")
    parser.add_argument("--auto", action="store_true",
                        help="as weight_tier_budget=-1: fp16 left out")
    parser.add_argument("--reserve", type=parse_bytes, default=0)
    parser.add_argument("--psram", type=parse_bytes, default=None,
                        help="PSRAM the copies are allocated from (default: "
                             "unlimited)")
    parser.add_argument("--board", default="papercolor")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--caches", default="story", choices=tuple(chips.CACHES))
    parser.add_argument("--mspi", default=None)
    parser.add_argument("--rows", type=int, default=20,
                        help="weights to list (default: 20)")
    parser.add_argument("--json", help="also write the report as JSON")
    args = parser.parse_args(argv)
    with open(args.pte, "rb") as f:
        program = read_program(f.read())
    rep = report(program, budgets=args.budget,
                 sequence=args.load.split(",") if args.load else None,
                 per=args.per, tables=args.tables, auto=args.auto,
                 reserve=args.reserve, psram=args.psram,
                 rates=class_rates(args.board, args.cores, args.caches, args.mspi))
    print(render(rep, args.rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep.as_json(), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
