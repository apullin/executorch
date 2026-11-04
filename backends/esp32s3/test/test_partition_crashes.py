# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Partitioner refusals where a crash inside lowering is easy (cases B1-B16).

Each case is a tiny module (hidden 64, one or two Linears) whose offending op
must be left portable rather than raise inside lowering. Real model families
write each of them (Gemma 4, DeepSeek-V4, Apertus, SigLIP, ...).

  B1  a 1-element parameter or buffer.  ExecuTorch partitions a FAKE program,
      so a softcap scan calling float(tensor) would raise
      GuardOnDataDependentSymNode on any `mul` by such a constant (Gemma 4's
      layer_scalar, a learned lambda).
  B2  clamp or exact GELU on a reshaped matmul output that has other users.
      A gate that looks through the view to the single-use matmul without
      asking whether the VIEW is single-use claims the activation, refuses the
      view, and preprocess finds "gate and walk disagree".
  B3  an embedding whose indices are a build-time constant (SigLIP/CLIP/BERT
      position_ids buffer).  Pulling the index buffer into the partition as a
      constant would leave the gather with no runtime input. Indices computed
      from constants (GPT-2's arange positions, a view of the buffer) are a
      constant lookup too, which constant propagation folds exactly.
  B4  a weight shared between a delegated matmul and a torch.cond branch.
      ExecuTorch's constant duplication rewrites only top-level arguments, so
      the cond would keep the partition's tagged constant and backend_api
      would raise.
  B5  a matmul or convolution whose activation is a build-time constant
      (learned query tokens, a learned map).  The partitioner pulls
      a claimed op's constant operands into its partition, the activation
      too, which would leave the op with no runtime input. Held rank 3, or
      a matmul of two parameters, torch spells it view_copy -> addmm/mm: the
      constant reaches the op through a view, which the gate must see
      through, or it claims the op and quantises the constant at run time (a
      two-element LayerNorm after it turns the int8 rounding into sign
      flips). The same for a constant an op that is not data movement
      computes from parameters (their abs, negation or a scale): the gate
      decides constness as constant propagation does, whatever pure op
      computes it.
  B6  the same Linear quantised by a dynamic PT2E recipe, also through a
      ReLU on the constant. convert_pt2e
      folds the constant's quantize into codes and leaves its chooser, a
      chain over constants that no op absorbs; left in the program it would
      run ExecuTorch's choose_qparams, which disagrees with PyTorch's (and
      has no symmetric form). A whole chain over the constant -- torchao's,
      or PT2E's written out, one of them read by a graph output -- would
      hand the Linear a constant activation if absorbed. Rank-3 queries
      (DETR, Q-Former), or queries broadcast over the batch by expand or
      repeat, reach the Linear through a view of the folded value; a gate
      that does not see it as a constant claims the Linear without the
      chain's record and quantises the constant a second time.
  B7  a dynamically quantised convolution the delegate does not take, a
      batch of two images; a strip that absorbs its chain all the same
      leaves the partitioner refusing the program.
  B8  a dynamically quantised Linear over a transposed rank-4 tensor, which
      torch spells as a bmm against a broadcast weight; a strip that finds
      no matmul or convolution behind the chain's views refuses the
      program.
  B9  a dynamically quantised Linear the gate refuses only once it carries
      the chain's record (K past the exact dot's bound); asked before the
      record is on it, the strip would absorb the chain all the same.
      In B7 to B9 the chain stays in front of the op, which runs portable,
      computed by float32 portable ops as the recipe computes it: the
      runtime runs no chooser, nor any of torchao's ops. B8 and B9 also run
      under esp32_torchao_config, with its per-token chain in front of such
      an op. B6 to B11 reach the strip through lower_esp32, which runs it
      (a flow that leaves every chain outside the delegates fails these
      programs).
  B10 a dynamically quantised Linear over a state the forward writes (an
      RNN cell's buffer). The chain is not over constants, so it stays.
      No partition takes a written buffer in, so the Linear over the state
      reads it at run time, as it reads an input -- under a recipe it takes the chain and computes the converted
      model's numbers, call after call; behind a chain written out over
      float weights it absorbs the chain as the Linear over an input does,
      its weight then quantised by the format policy (_keeps_float), and
      computes what the same cell computes reading its state through a
      view, bit for bit. A rank-3 state the Linear reads through the view
      torch spells it with is no constant to refuse: the view stays outside
      the partition, and the delegate reads the state through it at run
      time. A state used as a WEIGHT (fast weights: y = x @ S.t(), then S
      updated) is refused, directly or through its transpose: claimed, the
      partition would pull S in, preprocess would pack its first value, and
      the second call would repeat the first bit for bit.
  B11 a recipe-quantised Linear under format_policy='fp32', whose loop
      stores float weights and has no activation quantiser: claimed with
      the recipe's codes, or with its chain absorbed into it, preprocess
      would refuse the program. The same for any loop that cannot carry
      the model's own activation quantiser: a static pair in front of a
      float Linear under the fp32, bf16 or i16 policy (whose loop quantises
      its activation to int16 codes itself), or a dynamic chain in front of
      one quantised at i16.
  B12 a dynamic chain that reaches its op through data movement other than
      a view: to_edge spells an unbatched convolution unsqueeze ->
      convolution -> squeeze, and a Linear over a transposed tensor
      permute -> addmm. A strip that walks views only computes the chain
      in place in front of the move and never asks about the op behind it,
      which the partitioner then claims with no record, quantising the
      model's dequantised value a second time its own way. So a move that
      keeps a per-tensor scheme is walked; behind one that
      does not (a slice, a select, a transpose under a per-token scheme)
      the op keeps the chain's value and runs portable.
  B13 a reshaped matmul output feeding an add or mul the gate refuses (its
      other operand carries a symbolic dim, or the add feeds a claimed
      conv). A view gate that checks only part of the consumer's
      verdict claims the view alone, and preprocess raises "<mm> is not a
      partition output but is consumed by <view>, which is not a chained
      op".
  B14 q@kT under quantize_qk whose query or key is a build-time constant
      (learned query tokens, which constant propagation folds; a learned key
      memory) or a state the forward writes. The partitioner pulls a
      constant into the partition, which would leave preprocess finding the
      bmm with no runtime input for it: "... is neither a partition input
      nor an earlier claimed op's tail", or for the key "... must be a
      partition input". A written buffer it does not pull in, so the
      state's q@kT is delegated, reading it at run time.
  B15 a BatchNorm after a convolution whose weight a QAT conversion
      dequantises (conv, BatchNorm, BatchNorm under
      Esp32Quantizer(is_qat=True)). The conversion folds the first
      BatchNorm into the conv's int codes; a conv+bn fold that takes the
      second onto the dequantise -- which constant propagation never
      computes (it skips quantize and dequantize primitives) -- makes
      lower_esp32 raise "SpecViolationError: Node.meta aten_mul_tensor is
      missing val field".
  B16 a float Linear whose activation is a state the forward writes, read
      directly, through a view or through a transpose, under the default
      and the fp32 policies. All three are delegated, the state read at
      run time (a
      partition input directly, the transpose left outside the partition),
      and over two calls in one host runner process -- the second reading
      what the first wrote -- the fp32 loop gives the portable program's
      bits (dyadic data: every sum exact) and the default policy's int8
      loop gives the same bits read directly as through the view or the
      transpose.
      Likewise an embedding gathering by a token history the forward writes
      (no constant lookup: constant propagation skips a written buffer) is
      delegated, and gathers what the same indices given as an input do.

Each case asserts: lowering completes, the op that cannot be delegated is
refused WITH A REASON, the rest of the model is still delegated, and the
program emits; from B5 on, the host runner computes the model's numbers.
Every check is printed and a failed one recorded, so a run reports them
all: as a script, the file exits 1 if any failed; under pytest, each
test_b* fails naming its failed checks (@case).

    python test_partition_crashes.py
"""

import contextlib
import functools
import io
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

FAILS = []
D = 64


def check(name, cond, detail=""):
    tag = "ok " if cond else "FAIL"
    print(f"  {tag} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def case(test):
    """One test_b* case. check() records a failure and goes on, so a script
    run reports every one (main); a case that only did that would pass
    under pytest whatever its checks found. Called, it fails if any check
    it made failed, naming them. main() calls the case itself
    (__wrapped__), so the script still runs every case."""

    @functools.wraps(test)
    def run():
        start = len(FAILS)
        test()
        failed = FAILS[start:]
        assert not failed, f"{len(failed)} check(s) failed: {failed}"

    return run


def lower(model, args, keep=None, passes=(), dynamic_shapes=None, **options):
    """The production pipeline (lower_esp32), then the standard post passes and
    emission, for a module or an ExportedProgram (a converted PT2E model's);
    `passes` go to the emission, `dynamic_shapes` to torch.export, and
    `options` to the Esp32Partitioner.
    Returns (partitioner, graph ops, operator table, error); the program goes
    in keep["program"] and the backend's log in keep["log"] when `keep` is
    given."""
    from executorch.exir import ExecutorchBackendConfig
    from torch.export import ExportedProgram

    from executorch.backends.esp32s3.partition.espressif_partitioner import (
        Esp32Partitioner,
    )
    from executorch.backends.esp32s3._passes import apply_passes, lower_esp32

    part = Esp32Partitioner(**options)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            ep = (model if isinstance(model, ExportedProgram)
                  else torch.export.export(model.eval(), args,
                                           dynamic_shapes=dynamic_shapes))
            edge = lower_esp32(ep, [part])
            graph_ops = [str(n.target) for n in
                         edge.exported_program().graph_module.graph.nodes
                         if n.op == "call_function"]
            apply_passes(edge, quiet=True)
            et = edge.to_executorch(config=ExecutorchBackendConfig(
                delegate_alignment=16, passes=list(passes)))
    except Exception as e:  # the regression is precisely "this raised"
        return part, None, None, f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if keep is not None:
            keep["log"] = log.getvalue()
    if keep is not None:
        keep["program"] = et
    table = set()
    for plan in et.executorch_program.execution_plan:
        for op in plan.operators:
            table.add(f"{op.name}.{op.overload}" if op.overload else op.name)
    return part, graph_ops, table, None


def run_host(program, args, outputs=1):
    """The program on the host runner: (its first output, or all `outputs`
    of them in a list when more than one, or None, and the runner's log)."""
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(program.buffer)
        command = [str(HOST_RUNNER), "--pte", str(tmp / "model.pte")]
        for i, x in enumerate(args):
            x.detach().contiguous().numpy().tofile(tmp / f"in{i}.bin")
            command += ["--input", str(tmp / f"in{i}.bin")]
        for i in range(outputs):
            command += ["--output", str(tmp / f"out{i}.bin")]
        try:
            done = subprocess.run(command, capture_output=True, text=True,
                                  timeout=120)
        except FileNotFoundError:
            return None, f"host runner not built: {HOST_RUNNER}"
        log = done.stdout + done.stderr
        if done.returncode != 0:
            return None, log
        got = [np.fromfile(tmp / f"out{i}.bin", dtype=np.float32)
               for i in range(outputs)]
        return (got if outputs > 1 else got[0]), log


def run_steps(program, args, steps):
    """The program's forward on `args`, `steps` times in one host runner
    process, so that a state the forward writes carries over: ([each call's
    output], the runner's log), or (None, the log)."""
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    from executorch.backends.esp32s3.test.host import HOST_RUNNER

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(program.buffer)
        inputs = []
        for i, x in enumerate(args):
            x.detach().contiguous().numpy().tofile(tmp / f"in{i}.bin")
            inputs.append(str(tmp / f"in{i}.bin"))
        (tmp / "steps.txt").write_text("".join(
            f"forward {' '.join(inputs)} -> {tmp / f'out{s}.bin'}\n"
            for s in range(steps)))
        done = subprocess.run([str(HOST_RUNNER), "--pte", str(tmp / "model.pte"),
                               "--script", str(tmp / "steps.txt")],
                              capture_output=True, text=True, timeout=120)
        log = done.stdout + done.stderr
        if done.returncode != 0:
            return None, log
        return [np.fromfile(tmp / f"out{s}.bin", dtype=np.float32)
                for s in range(steps)], log


def same_numbers(got, want, rel=1e-6):
    """Whether the host runner's output is `want` to fp32 rounding (`rel` of
    its largest magnitude)."""
    import numpy as np

    want = want.detach().numpy().reshape(-1)
    if got is None or got.shape != want.shape:
        return False, "no output" if got is None else f"shape {got.shape}"
    err = float(np.abs(got - want).max()) if want.size else 0.0
    scale = max(float(np.abs(want).max()), 1e-30)
    return err <= rel * scale, f"max abs error {err:.3g} of {scale:.3g}"


def reasons(part):
    return {n.name: r for n, r in part.reporter._rejected_nodes.items()}


def state_read_by_delegate(program, fqn):
    """Whether buffer `fqn`'s placeholder is an argument of a delegate call
    in the emitted program: the delegate reads the state at run time."""
    ep = program.exported_program()
    name = next((s.arg.name for s in ep.graph_signature.input_specs
                 if s.kind.name == "BUFFER" and s.target == fqn), None)
    return name is not None and any(
        name in [getattr(a, "name", None) for a in n.args]
        for n in ep.graph.nodes if n.op == "call_function"
        and "executorch_call_delegate" in str(n.target))


def delegates(graph_ops):
    return sum("executorch_call_delegate" in t for t in graph_ops)


def portable(graph_ops, name):
    return sum(f"aten.{name}." in t for t in graph_ops
               if "executorch_call_delegate" not in t)


# ---------------------------------------------------------------------------
# B1: one-element constants
# ---------------------------------------------------------------------------
class ScalarParam(nn.Module):
    """Differential-attention lambda / ReZero alpha: a learned 0-d scalar."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)
        self.lam = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.b(self.a(x) * self.lam)


class LayerScalar(nn.Module):
    """Gemma 4's per-layer residual scalar, a persistent [1] buffer."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x):
        return x + self.b(F.silu(self.a(x))) * self.layer_scalar


class LearnedCap(nn.Module):
    """A softcap whose cap is a parameter: its value is not visible while
    partitioning, so it must be refused rather than evaluated."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(D, 32, bias=False)
        self.cap = nn.Parameter(torch.tensor(20.0))

    def forward(self, x):
        return torch.tanh((self.head(x) * 0.196) / self.cap) * self.cap


@case
def test_b1():
    print("B1: one-element parameters and buffers")
    x = torch.randn(1, 8, D)
    for model in (ScalarParam(), LayerScalar()):
        name = type(model).__name__
        part, ops, _, err = lower(model, (x,))
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            check(f"{name}: no matmul left portable",
                  portable(ops, "mm") + portable(ops, "addmm") == 0)
            check(f"{name}: the scalar mul stays portable",
                  portable(ops, "mul") >= 1)

    part, ops, _, err = lower(LearnedCap(), (x,))
    check("LearnedCap lowers", err is None, err or "")
    if err is None:
        rs = reasons(part)
        pending = [n for n, r in rs.items() if "not visible" in r]
        check("learned softcap refused with a reason naming the cause",
              bool(pending) and portable(ops, "tanh") == 1,
              f"refused={pending}")
        check("its Linear is still delegated",
              delegates(ops) == 1 and portable(ops, "mm") == 0)


# ---------------------------------------------------------------------------
# B2: activation on a multi-use reshaped matmul output
# ---------------------------------------------------------------------------
class ClampMultiUse(nn.Module):
    """xIELU's clamp spelling: the Linear's [B,T,D] output feeds five ops."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        h = self.a(x)
        return self.b(torch.where(
            h > 0, h * h, torch.expm1(torch.clamp(h, max=-1e-6)) - h))


class GeluMultiUse(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        h = self.a(x)
        return self.b(F.gelu(h)) * h


class GeluSingleUse(nn.Module):
    """Control: the ordinary single-use spelling must still fuse."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        return self.b(F.gelu(self.a(x)))


@case
def test_b2():
    print("B2: clamp / exact GELU on a multi-use reshaped matmul output")
    x = torch.randn(1, 8, D)
    for model, op in ((ClampMultiUse(), "clamp"), (GeluMultiUse(), "gelu")):
        name = type(model).__name__
        part, ops, _, err = lower(model, (x,))
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            rs = reasons(part)
            refused = [n for n in rs if op in n]
            check(f"{name}: the {op} is refused with a reason",
                  bool(refused) and portable(ops, op) == 1,
                  f"refused={refused}")
            check(f"{name}: both Linears are still delegated",
                  portable(ops, "mm") + portable(ops, "addmm") == 0)
    part, ops, _, err = lower(GeluSingleUse(), (x,))
    check("GeluSingleUse (control) lowers", err is None, err or "")
    if err is None:
        check("single-use gelu still fuses", portable(ops, "gelu") == 0)


# ---------------------------------------------------------------------------
# B3: embedding with constant indices
# ---------------------------------------------------------------------------
class PositionEmbedding(nn.Module):
    """SigLIP/CLIP's absolute positions: an Embedding over a position_ids buffer."""

    def __init__(self, persistent):
        super().__init__()
        self.pos = nn.Embedding(8, D)
        self.register_buffer("position_ids", torch.arange(8).expand((1, -1)),
                             persistent=persistent)
        self.fc = nn.Linear(D, D)

    def forward(self, x):
        return self.fc(x + self.pos(self.position_ids))


class ComputedPositions(nn.Module):
    """Positions computed from constants alone, added to the input: GPT-2's
    wpe(arange(T)) (`arange`), the same through an arithmetic op
    (`arange*1`), and the position_ids buffer read through a view (`view`).
    No Linear after it, so the program's numbers are the gather's."""

    def __init__(self, spelling):
        super().__init__()
        self.pos = nn.Embedding(8, D)
        self.register_buffer("position_ids", torch.arange(8), persistent=False)
        self.spelling = spelling

    def forward(self, x):                              # [1, 8, D]
        if self.spelling == "view":
            ids = self.position_ids.view(1, 8)
        else:
            ids = torch.arange(x.shape[1])
            if self.spelling == "arange*1":
                ids = ids * 1
            ids = ids.unsqueeze(0)
        return x + self.pos(ids)


class GroupCodedPositions(nn.Module):
    """GPT-2's positions, wpe(arange(T)), over a table held as int8 codes
    with per-group scales (dequantize_per_channel_group, as a stock
    recipe's embedding becomes): the delegate gathers the codes verbatim,
    and the dequantise has no portable kernel, so the gather is claimed
    whatever its indices."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(9)
        self.register_buffer("codes", torch.randint(
            -128, 128, (8, D), generator=g, dtype=torch.int8))
        self.register_buffer("scales", torch.rand(8, 2, generator=g) * 0.02 + 0.01)

    def forward(self, x):                              # [1, 8, D]
        import torch.ao.quantization.fx._decomposed  # noqa: F401

        table = torch.ops.quantized_decomposed.dequantize_per_channel_group(
            self.codes, self.scales, None, -128, 127, torch.int8, D // 2,
            torch.float32)
        ids = torch.arange(x.shape[1]).unsqueeze(0)
        return x + F.embedding(ids, table)


class TokenEmbedding(nn.Module):
    """Control: runtime indices must still delegate the gather."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(32, D)
        self.fc = nn.Linear(D, D)

    def forward(self, ids):
        return self.fc(self.tok(ids))


@case
def test_b3():
    print("B3: embedding whose indices are a build-time constant")
    x = torch.randn(1, 8, D)
    for persistent in (False, True):
        name = f"PositionEmbedding(persistent={persistent})"
        part, ops, table, err = lower(PositionEmbedding(persistent), (x,))
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            rs = reasons(part)
            refused = [r for n, r in rs.items() if "embedding" in n]
            check(f"{name}: embedding refused, reason names constant indices",
                  bool(refused) and all("constant" in r for r in refused),
                  f"{refused[:1]}")
            check(f"{name}: constant propagation folds the gather away",
                  not any("embedding" in t for t in table),
                  f"{sorted(t for t in table if 'embedding' in t)}")
            check(f"{name}: the Linear is still delegated",
                  delegates(ops) == 1 and portable(ops, "addmm") == 0)
    # Indices computed from constants: refused as constants too, so the
    # gather is folded exactly instead of reading an int8 table at run time
    # (read at run time, wpe(arange(T)) is 0.0134 off).
    for spelling in ("arange", "arange*1", "view"):
        name = f"ComputedPositions({spelling!r})"
        model = ComputedPositions(spelling).eval()
        keep = {}
        part, ops, table, err = lower(model, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "embedding" in n]
            check(f"{name}: embedding refused, reason names constant indices",
                  bool(refused) and all("constant" in r for r in refused),
                  f"{refused[:1]}")
            check(f"{name}: constant propagation folds the gather away",
                  not any("embedding" in t for t in table),
                  f"{sorted(t for t in table if 'embedding' in t)}")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, model(x), rel=1e-6)
            check(f"{name}: the host runner computes eager's numbers", ok,
                  detail if got is not None else log[-300:])
    # The same positions over a group-coded table: still delegated.
    model, keep = GroupCodedPositions().eval(), {}
    part, ops, _, err = lower(model, (x,), keep)
    check("GroupCodedPositions lowers", err is None, err or "")
    if err is None:
        check("GroupCodedPositions: the gather of codes is delegated",
              delegates(ops) == 1 and portable(ops, "embedding") == 0,
              f"{delegates(ops)} delegate(s); {dict(part.support.kinds)}")
        got, log = run_host(keep["program"], (x,))
        with torch.no_grad():
            ok, detail = same_numbers(got, model(x), rel=1e-6)
        check("GroupCodedPositions: the host runner computes eager's numbers",
              ok, detail if got is not None else log[-300:])
    ids = torch.randint(0, 32, (1, 8))
    part, ops, _, err = lower(TokenEmbedding(), (ids,))
    check("TokenEmbedding (control) lowers", err is None, err or "")
    if err is None:
        check("runtime-index embedding still delegates",
              portable(ops, "embedding") == 0)


# ---------------------------------------------------------------------------
# B4: a weight shared with a torch.cond branch
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(D, 96, bias=False)
        self.u = nn.Linear(D, 96, bias=False)
        self.d = nn.Linear(96, D, bias=False)

    def forward(self, h):
        return self.d(F.silu(self.g(h)) * self.u(h))


class CondSharedWeight(nn.Module):
    """Early exit: the branch re-applies the trunk's own FFN."""

    def __init__(self):
        super().__init__()
        self.f = SwiGLU()

    def forward(self, x):
        h = self.f(x)
        return torch.cond(h.mean() > 0, lambda h: h.clone(),
                          lambda h: h + self.f(h), (h,))


class CondOwnWeights(nn.Module):
    """Control: a branch with its own weights leaves the trunk delegated."""

    def __init__(self):
        super().__init__()
        self.f, self.g = SwiGLU(), SwiGLU()

    def forward(self, x):
        h = self.f(x)
        return torch.cond(h.mean() > 0, lambda h: h.clone(),
                          lambda h: h + self.g(h), (h,))


@case
def test_b4():
    print("B4: a weight shared between the trunk and a torch.cond branch")
    x = torch.randn(1, 8, D)
    part, ops, _, err = lower(CondSharedWeight(), (x,))
    check("CondSharedWeight lowers", err is None, err or "")
    if err is None:
        rs = reasons(part)
        # Matmuls only: the gate x up multiply is refused for the same shared
        # weight too (one more node naming it), correctly.
        shared = [n for n, r in rs.items() if "shared" in r and "mm" in n]
        check("the trunk matmuls on shared weights are refused with a reason",
              len(shared) == 3, f"refused={shared}")
    part, ops, _, err = lower(CondOwnWeights(), (x,))
    check("CondOwnWeights (control) lowers", err is None, err or "")
    if err is None:
        check("trunk with private weights is still delegated",
              delegates(ops) >= 1 and portable(ops, "mm") == 0,
              f"delegates={delegates(ops)}")


# ---------------------------------------------------------------------------
# B5: a matmul or convolution over a build-time constant activation
# ---------------------------------------------------------------------------
class LinearOverParam(nn.Module):
    """Learned query tokens through a projection: Linear(parameter)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 16)
        self.queries = nn.Parameter(torch.randn(4, D))

    def forward(self, x):
        return self.fc(self.queries) + x


class LinearOverPermutedParam(nn.Module):
    """The same through a transpose, which the gate folds like a weight's."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 16)
        self.queries = nn.Parameter(torch.randn(D, 4))

    def forward(self, x):
        return self.fc(self.queries.t()) + x


class LinearOverRank3Param(nn.Module):
    """Learned query tokens held [1, Q, D] (DETR, Q-Former): torch spells the
    Linear view_copy -> addmm, the constant behind a view."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 16)
        self.queries = nn.Parameter(torch.randn(1, 4, D))

    def forward(self, x):
        return self.fc(self.queries) + x


class MatmulOfParams(nn.Module):
    """A matmul of two parameters, rank 3 by rank 2:
    view_copy -> mm, both operands constants."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(2, 6, D))
        self.b = nn.Parameter(torch.randn(D, 2) * 0.1)

    def forward(self, x):
        return torch.matmul(self.a, self.b) + x


class MatmulOfComputedParam(nn.Module):
    """MatmulOfParams' matmul, the first parameter through an op that is not
    data movement -- its abs, negation or a scale -- then a two-element
    LayerNorm: still an op over constants alone."""

    def __init__(self, spelling):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.a = nn.Parameter(torch.randn(2, 12, 9, generator=g))
        self.b = nn.Parameter(torch.randn(9, 2, generator=g))
        self.ln = nn.LayerNorm(2)
        self.spelling = spelling

    def forward(self, x):
        a = {"abs": torch.abs, "neg": torch.neg,
             "scale": lambda t: t * 0.5}[self.spelling](self.a)
        return self.ln(torch.matmul(a, self.b)) + x


class ConvOverParam(nn.Module):
    """A convolution over a learned map."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.map = nn.Parameter(torch.randn(1, 4, 6, 6))

    def forward(self, x):
        return self.conv(self.map) + x


class LinearOverInput(nn.Module):
    """Control: the same Linear over its runtime input."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 16)

    def forward(self, x):
        return self.fc(x)


def pt2e(model, x, preset=None):
    """`model` quantised by Esp32Quantizer (its default, dynamic, or
    `preset`), calibrated on `x` and exported."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from executorch.backends.esp32s3 import Esp32Quantizer

    quantizer = Esp32Quantizer() if preset is None else Esp32Quantizer(preset=preset)
    with contextlib.redirect_stdout(io.StringIO()):
        gm = torch.export.export(model.eval(), (x,)).module()
        prepared = prepare_pt2e(gm, quantizer)
        prepared(x)
        return torch.export.export(convert_pt2e(prepared), (x,))


@case
def test_b5():
    print("B5: a matmul or convolution whose activation is a build-time constant")
    cases = [(LinearOverParam, torch.randn(4, 16), "addmm", {}),
             (LinearOverParam, torch.randn(4, 16), "addmm",
              {"format_policy": "fp32"}),
             (LinearOverPermutedParam, torch.randn(4, 16), "addmm", {}),
             (LinearOverRank3Param, torch.randn(1, 4, 16), "addmm", {}),
             (MatmulOfParams, torch.randn(2, 6, 2), "mm", {}),
             (ConvOverParam, torch.randn(1, 8, 6, 6), "convolution", {})]
    for cls, x, op, options in cases:
        name = f"{cls.__name__}({', '.join(f'{k}={v!r}' for k, v in options.items())})"
        model = cls().eval()
        keep = {}
        part, ops, _, err = lower(model, (x,), keep, **options)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if op in n]
            check(f"{name}: the {op} is refused, the reason naming the constant",
                  bool(refused) and all("constant" in r for r in refused),
                  f"{refused[:1]}")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, model(x))
            check(f"{name}: the host runner computes eager's numbers", ok,
                  detail if got is not None else log[-300:])
    # A constant computed by an op that is not data movement.
    for spelling in ("abs", "neg", "scale"):
        name = f"MatmulOfComputedParam({spelling!r})"
        model, x = MatmulOfComputedParam(spelling).eval(), torch.randn(2, 12, 2)
        keep = {}
        part, ops, _, err = lower(model, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "_mm_" in n]
            check(f"{name}: the mm is refused, the reason naming the constant",
                  bool(refused) and all("constant" in r for r in refused),
                  f"{refused[:1]}")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, model(x))
            check(f"{name}: the host runner computes eager's numbers", ok,
                  detail if got is not None else log[-300:])
    # Static PT2E folds a constant activation's pair to a dequantise of
    # constant codes.
    model, x = LinearOverParam().eval(), torch.randn(4, 16)
    ep = pt2e(model, x, "xnnpack_static")
    keep = {}
    part, ops, _, err = lower(ep, (x,), keep)
    check("LinearOverParam, static PT2E (a dequantise of constant codes) lowers",
          err is None, err or "")
    if err is None:
        got, log = run_host(keep["program"], (x,))
        with torch.no_grad():
            ok, detail = same_numbers(got, ep.module()(x))
        check("LinearOverParam, static PT2E: the host runner computes the "
              "converted model's numbers", ok,
              detail if got is not None else log[-300:])
    part, ops, _, err = lower(LinearOverInput(), (torch.randn(4, D),))
    check("LinearOverInput (control) lowers", err is None, err or "")
    if err is None:
        check("a Linear over its runtime input still delegates",
              delegates(ops) == 1 and portable(ops, "addmm") == 0)


# ---------------------------------------------------------------------------
# B6: a dynamically quantised Linear over a build-time constant
# ---------------------------------------------------------------------------
class QueriesThenProject(nn.Module):
    """Learned queries projected (through a ReLU when `relu`: the chooser then
    reads an op on the constant), then a Linear over the runtime input added
    in: one dynamic chain over constants, one over an activation. The
    queries are [4, D]; `rank3` [1, 4, D], as DETR and Q-Former hold them;
    or broadcast over the input's batch by `expand` or `repeat`. The last
    three the Linear reads through a view."""

    def __init__(self, relu=False, spelling=None):
        super().__init__()
        self.q = nn.Linear(D, 16)
        self.queries = nn.Parameter(torch.randn(1, 4, D) if spelling == "rank3"
                                    else torch.randn(4, D))
        self.out = nn.Linear(16, 16)
        self.relu = relu
        self.spelling = spelling

    def forward(self, x):
        queries = torch.relu(self.queries) if self.relu else self.queries
        if self.spelling == "expand":
            queries = queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        elif self.spelling == "repeat":
            queries = queries.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return self.out(self.q(queries) + x)


def _chain(t):
    """Esp32Quantizer's dynamic chain written out whole, its quantize not
    folded into codes (torchao's chain, and convert_pt2e's with
    fold_quantize=False, are spelled so)."""
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

    qd = torch.ops.quantized_decomposed
    eps = torch.finfo(torch.float32).eps
    scale, zp = qd.choose_qparams_symmetric.tensor(t, -127, 127, eps, torch.int8)
    q = qd.quantize_per_tensor.tensor(t, scale, zp, -127, 127, torch.int8)
    return qd.dequantize_per_tensor.tensor(q, scale, zp, -127, 127, torch.int8)


class WholeChainOverQueries(nn.Module):
    """The learned queries through a whole chain into a Linear; and, when
    `keys`, a second whole chain over constants returned as a graph output."""

    def __init__(self, keys=False):
        super().__init__()
        self.q = nn.Linear(D, 16)
        self.queries = nn.Parameter(torch.randn(4, D))
        self.keys = nn.Parameter(torch.randn(4, 16)) if keys else None

    def forward(self, x):
        out = self.q(_chain(self.queries)) + x
        return out if self.keys is None else (out, _chain(self.keys))


def torchao(model, x, bits=4):
    """`model` quantised by esp32_torchao_config (int`bits` weights per
    channel, dynamic per-token int8 activations: torchao's whole chain) and
    exported."""
    from torchao.quantization import quantize_

    from executorch.backends.esp32s3.quantizer.espressif_quantizer import (
        esp32_torchao_config,
    )

    model = model.eval()
    with contextlib.redirect_stdout(io.StringIO()):
        quantize_(model, esp32_torchao_config(bits, None))
        return torch.export.export(model, (x,), strict=False)


def choosers(graph_ops):
    return [t for t in graph_ops if "choose_qparams" in t]


@case
def test_b6():
    print("B6: a dynamically quantised Linear over a build-time constant")
    x = torch.randn(4, 16)
    programs = [(f"QueriesThenProject(relu={relu}), "
                 f"{preset or 'Esp32Quantizer()'}",
                 pt2e(QueriesThenProject(relu), x, preset), 1, x)
                for relu, preset in ((False, None), (False, "xnnpack_dynamic"),
                                     (True, None))]
    # Whole chains over constants: torchao's (no fold_quantize), and the
    # PT2E spelling written out, alone and beside one a graph output reads.
    programs += [("QueriesThenProject, esp32_torchao_config(4, None)",
                  torchao(QueriesThenProject(), x), 1, x),
                 ("WholeChainOverQueries",
                  torch.export.export(WholeChainOverQueries().eval(), (x,)), 0,
                  x),
                 ("WholeChainOverQueries(keys=True), the keys' chain an output",
                  torch.export.export(WholeChainOverQueries(True).eval(), (x,)),
                  0, x)]
    # The folded value reaches the Linear through a view.
    rank3, batch2 = torch.randn(1, 4, 16), torch.randn(2, 4, 16)
    programs += [(f"QueriesThenProject(spelling={spelling!r}), "
                  f"{preset or 'Esp32Quantizer()'}",
                  pt2e(QueriesThenProject(spelling=spelling), inputs, preset),
                  1, inputs)
                 for spelling, inputs in (("rank3", rank3), ("expand", batch2),
                                          ("repeat", batch2))
                 for preset in (None, "xnnpack_dynamic")]
    programs.append(("QueriesThenProject(spelling='rank3'), "
                     "esp32_torchao_config(4, None)",
                     torchao(QueriesThenProject(spelling="rank3"), rank3), 1,
                     rank3))
    for name, ep, delegated, x in programs:
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            check(f"{name}: no chooser is left in the program",
                  not choosers(ops), f"{choosers(ops)}")
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear over the constant is refused, the "
                  f"reason naming the constant",
                  len(refused) == 1 and "constant" in refused[0], f"{refused}")
            if delegated:
                check(f"{name}: the Linear over the activation is delegated",
                      delegates(ops) == 1)
            with torch.no_grad():
                want = ep.module()(x)
            want = list(want) if isinstance(want, tuple) else [want]
            got, log = run_host(keep["program"], (x,), len(want))
            if got is not None and len(want) == 1:
                got = [got]
            for i, w in enumerate(want):
                ok, detail = (same_numbers(got[i], w, rel=1e-5)
                              if got is not None else (False, log[-300:]))
                check(f"{name}: the host runner computes the converted "
                      f"model's output {i}", ok, detail)


# ---------------------------------------------------------------------------
# B7, B8: a dynamically quantised op the delegate does not take
# ---------------------------------------------------------------------------
def left_chain(name, keep, ops, op, why):
    """The chain stays in front of the op the delegate does not take, said
    out loud with `why`, and the program asks the runtime for none of the
    chain's own kernels: no chooser, no quantize or dequantize with tensor
    arguments (a static weight dequantize may stay, for the portable op), and
    none of torchao's ops, which have no ExecuTorch kernels."""
    left = [l for l in keep["log"].splitlines() if l.startswith("esp32: leaving")]
    check(f"{name}: the activation chain stays in front of the {op}, said so",
          len(left) == 1 and why in left[0], f"{left}")
    check(f"{name}: and the {op} runs portable", portable(ops, op) == 1)
    table = {f"{o.name}.{o.overload}" for plan in
             keep["program"].executorch_program.execution_plan
             for o in plan.operators}
    chain = sorted(t for t in table if t.startswith("torchao::") or (
        t.startswith("quantized_decomposed::")
        and ("choose" in t or t.endswith(".Tensor_out"))))
    check(f"{name}: the chain needs no chooser, quantize or dequantize kernel",
          not chain, f"{chain}")


class BatchedConvThenLinear(nn.Module):
    """Two images through a convolution (batch 2, which the delegate does
    not take), then a Linear (which it does). No activation after the
    convolution: XNNPACK's dynamic recipe leaves a conv + ReLU in float."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 6 * 6, 16)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


@case
def test_b7():
    print("B7: a dynamically quantised convolution the delegate does not take")
    x = torch.randn(2, 4, 6, 6)
    for preset in ("xnnpack_dynamic", None):
        name = f"BatchedConvThenLinear, {preset or 'Esp32Quantizer()'}"
        ep = pt2e(BatchedConvThenLinear(), x, preset)
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "convolution" in n]
            check(f"{name}: the convolution is refused, the reason naming "
                  f"the batch", len(refused) == 1 and "batch 2" in refused[0],
                  f"{refused}")
            left_chain(name, keep, ops, "convolution",
                       "stays portable (convolution shape")
            check(f"{name}: the Linear's chain is absorbed and it delegates",
                  delegates(ops) == 1 and portable(ops, "addmm") == 0)
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, ep.module()(x), rel=1e-5)
            check(f"{name}: the host runner computes the converted model's "
                  f"numbers", ok, detail if got is not None else log[-300:])


class LinearOverTransposed(nn.Module):
    """Linear(k, 7) over a transposed rank-4 tensor ([k, 4, 13, 5]), which
    torch spells as a bmm against the weight broadcast over 20 batches."""

    def __init__(self, k=1):
        super().__init__()
        self.fc = nn.Linear(k, 7)

    def forward(self, x):
        return self.fc(x.transpose(0, 3))


@case
def test_b8():
    print("B8: a dynamically quantised Linear spelled as a broadcast bmm")
    x = torch.randn(1, 4, 13, 5)
    programs = [(f"LinearOverTransposed, {preset or 'Esp32Quantizer()'}",
                 pt2e(LinearOverTransposed(), x, preset), x)
                for preset in (None, "xnnpack_dynamic")]
    # torchao's chooser reduces one-element blocks over the whole tensor, so
    # it cannot quantise a Linear(1, 7)'s weight: three rows of K = 3.
    rows = torch.randn(3, 4, 13, 5)
    programs.append(("LinearOverTransposed(k=3), esp32_torchao_config(4, None)",
                     torchao(LinearOverTransposed(k=3), rows), rows))
    for name, ep, x in programs:
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            left_chain(name, keep, ops, "bmm", "reads the dequantised value")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, ep.module()(x), rel=1e-5)
            check(f"{name}: the host runner computes the converted model's "
                  f"numbers", ok, detail if got is not None else log[-300:])


# ---------------------------------------------------------------------------
# B9: an op the gate refuses only once it carries the chain's record
# ---------------------------------------------------------------------------
@case
def test_b9():
    print("B9: a dynamically quantised Linear past the exact dot's bound")
    # One group of 32 past the 65,792 terms an int32 accumulator holds
    # exactly for an affine int8 activation ([act-dot-bound]): the gate reads
    # that bound from the chain's record on the op.
    # torchao's chain is per token, and its bound against int8 weights the
    # same.
    k = 65_792 + 32
    x = torch.rand(1, k)
    programs = [("xnnpack_dynamic",
                 pt2e(nn.Sequential(nn.Linear(k, 1)), x, "xnnpack_dynamic")),
                ("esp32_torchao_config(8, None)",
                 torchao(nn.Sequential(nn.Linear(k, 1)), x, bits=8))]
    for recipe, ep in programs:
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep)
        name = f"Linear({k}, 1), {recipe}"
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear is refused, the reason naming the bound",
                  len(refused) == 1 and "act-dot-bound" in refused[0],
                  f"{[r[:160] for r in refused]}")
            left_chain(name, keep, ops, "addmm", "stays portable (")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                # 65,824 float32 terms summed in another order than torch's.
                ok, detail = same_numbers(got, ep.module()(x), rel=1e-4)
            check(f"{name}: the host runner computes the converted model's "
                  f"numbers", ok, detail if got is not None else log[-300:])


# ---------------------------------------------------------------------------
# B10: a dynamically quantised Linear over a state the forward writes
# ---------------------------------------------------------------------------
class StateCell(nn.Module):
    """An RNN cell, h' = tanh(Wh h + Wx x), whose state h is a buffer the
    forward writes; with `chain`, Wh reads the state through Esp32Quantizer's
    dynamic chain written out, its weights left float. `viewed` holds h as
    [1, 1, 16] and reads it through a view."""

    def __init__(self, chain=False, viewed=False):
        super().__init__()
        self.register_buffer("h", torch.randn((1, 1, 16) if viewed
                                              else (1, 16)))
        self.wh = nn.Linear(16, 16)
        self.wx = nn.Linear(16, 16)
        self.chain, self.viewed = chain, viewed
        self.requires_grad_(False)

    def forward(self, x):
        h = self.h.view(1, 16) if self.viewed else self.h
        h = _chain(h) if self.chain else h
        new = torch.tanh(self.wh(h) + self.wx(x))
        self.h.copy_(new.view(self.h.shape))
        return new


class StateLinear3(nn.Module):
    """A Linear over a [1, 4, 16] state the forward writes, y = fc(h) + x,
    then h = y: torch reads the state through a view_copy into the addmm
    (an RNN state, a streaming model's frame buffer)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.fc = nn.Linear(16, 16)
        self.register_buffer("h", torch.randn(1, 4, 16, generator=g))
        self.requires_grad_(False)

    def forward(self, x):
        y = self.fc(self.h) + x
        self.h.copy_(y)
        return y


class FastWeight(nn.Module):
    """y = x @ S.t() (or x @ S), then S += 0.5 x^T x: a [16, 16] state the
    forward reads as a matmul's weight and then writes (fast weights, a
    linear-attention or memory state). Named `state`: the program's initial
    value goes in through InitializedMutableBufferPass, which matches the
    placeholder's name (b_state)."""

    def __init__(self, transpose):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("state", torch.randn(16, 16, generator=g) * 0.3)
        self.transpose = transpose
        self.requires_grad_(False)

    def forward(self, x):                                   # [2, 16]
        y = x @ (self.state.t() if self.transpose else self.state)
        self.state.copy_(self.state + 0.5 * (x.t() @ x))
        return y


@case
def test_b10():
    import copy

    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

    print("B10: a dynamically quantised Linear over a state the forward writes")
    x = torch.randn(1, 16)
    cell = StateCell()
    chained = StateCell(chain=True)
    chained.load_state_dict(cell.state_dict())
    programs = [(f"StateCell, {preset or 'Esp32Quantizer()'}",
                 pt2e(copy.deepcopy(cell), x, preset))
                for preset in (None, "xnnpack_dynamic")]
    for name, ep in programs:
        module = copy.deepcopy(ep).module()
        with torch.no_grad():
            want = [module(x).clone() for _ in range(2)]
        keep = {}
        # The state's first value goes in the program, which the converted
        # model starts from.
        part, ops, _, err = lower(ep, (x,), keep,
                                  passes=[InitializedMutableBufferPass(["h"])])
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear over the state is delegated, the state "
                  f"a partition input read at run time",
                  portable(ops, "addmm") == 0 and not refused
                  and state_read_by_delegate(keep["program"], "h"),
                  f"{delegates(ops)} delegate(s), refused "
                  f"{[r[:120] for r in refused]}")
            table = {f"{o.name}.{o.overload}" for plan in
                     keep["program"].executorch_program.execution_plan
                     for o in plan.operators}
            left = sorted(t for t in table if t.startswith("torchao::") or (
                t.startswith("quantized_decomposed::")
                and ("choose" in t or t.endswith(".Tensor_out"))))
            check(f"{name}: the chain needs no chooser, quantize or "
                  f"dequantize kernel", not left, f"{left}")
            got, log = run_steps(keep["program"], (x,), 2)
            for step, out in enumerate(want):
                ok, detail = same_numbers(got[step] if got else None, out,
                                          rel=1e-5)
                check(f"{name}: call {step + 1} computes the converted "
                      f"model's numbers", ok,
                      detail if got is not None else log[-300:])
    # Float weights behind the chain written out: the Linear over the state
    # absorbs the chain as the Linear over an input does, its weight then
    # quantised by the format policy, as through a view: the same cell
    # reading its state through one computes the same bits.
    viewed = StateCell(chain=True, viewed=True)
    viewed.wh.load_state_dict(chained.wh.state_dict())
    viewed.wx.load_state_dict(chained.wx.state_dict())
    viewed.h.copy_(chained.h.view(1, 1, 16))
    runs = {}
    for name, model in (("StateCell(chain=True), float weights", chained),
                        ("the same, the state read through a view", viewed)):
        ep = torch.export.export(copy.deepcopy(model).eval(), (x,))
        module = copy.deepcopy(ep).module()
        with torch.no_grad():
            want = [module(x).clone() for _ in range(2)]
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep,
                                  passes=[InitializedMutableBufferPass(["h"])])
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: both Linears are delegated",
                  portable(ops, "addmm") == 0 and not refused,
                  f"{delegates(ops)} delegate(s), refused "
                  f"{[r[:120] for r in refused]}")
            got, log = run_steps(keep["program"], (x,), 2)
            check(f"{name}: two calls run", got is not None,
                  "" if got is not None else log[-300:])
            runs[name] = got
            for step, out in enumerate(want if got else ()):
                ok, detail = same_numbers(got[step], out, rel=1e-2)
                check(f"{name}: call {step + 1} is the converted model's "
                      f"numbers to its int8 weights' error", ok, detail)
    if len(runs) == 2 and all(runs.values()):
        direct, through = runs.values()
        check("the state read directly computes the view's bits, call after "
              "call", all(a.tobytes() == b.tobytes()
                          for a, b in zip(direct, through)))
    # The state read through a view: delegated, the view outside reading it,
    # so the second call reads the state the first wrote. The float Linear
    # runs in the fp32 loop, which computes eager's numbers.
    x3 = torch.randn(1, 4, 16)
    linear = StateLinear3().eval()
    programs = [("StateLinear3, float", torch.export.export(
        copy.deepcopy(linear), (x3,)), {"format_policy": "fp32"})]
    programs += [(f"StateLinear3, {preset or 'Esp32Quantizer()'}",
                  pt2e(copy.deepcopy(linear), x3, preset), {})
                 for preset in (None, "xnnpack_dynamic")]
    for name, ep, options in programs:
        module = copy.deepcopy(ep).module()
        with torch.no_grad():
            want = [module(x3).clone() for _ in range(2)]
        keep = {}
        part, ops, _, err = lower(ep, (x3,), keep,
                                  passes=[InitializedMutableBufferPass(["h"])],
                                  **options)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear over the state's view is delegated",
                  delegates(ops) == 1 and portable(ops, "addmm") == 0
                  and not refused, f"{delegates(ops)} delegate(s), "
                  f"refused {[r[:120] for r in refused]}")
            got, log = run_steps(keep["program"], (x3,), 2)
            for step, out in enumerate(want):
                ok, detail = same_numbers(got[step] if got else None, out,
                                          rel=1e-5)
                check(f"{name}: call {step + 1} computes the converted "
                      f"model's numbers", ok,
                      detail if got is not None else log[-300:])
    # A state used as a weight: refused, and the portable matmul reads the
    # state the previous call wrote.
    x2 = torch.randn(2, 16, generator=torch.Generator().manual_seed(1))
    for transpose in (True, False):
        name = f"FastWeight(transpose={transpose})"
        model = FastWeight(transpose).eval()
        ep = torch.export.export(copy.deepcopy(model), (x2,))
        module = copy.deepcopy(ep).module()
        with torch.no_grad():
            want = [module(x2).clone() for _ in range(2)]
        keep = {}
        part, ops, _, err = lower(ep, (x2,), keep,
                                  passes=[InitializedMutableBufferPass(["state"])])
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            # Two matmuls: y's, over the state, and the update's x^T x,
            # whose weight is the runtime input.
            refused = [r for n, r in reasons(part).items() if "_mm_" in n]
            state = [r for r in refused
                     if "b_state is a state the forward writes" in r]
            check(f"{name}: the matmul over the state is refused, the reason "
                  f"naming the state",
                  len(state) == 1 and delegates(ops) == 0,
                  f"{delegates(ops)} delegate(s), {[r[:160] for r in refused]}")
            got, log = run_steps(keep["program"], (x2,), 2)
            for step, out in enumerate(want):
                ok, detail = same_numbers(got[step] if got else None, out,
                                          rel=1e-5)
                check(f"{name}: call {step + 1} computes the converted "
                      f"model's numbers", ok,
                      detail if got is not None else log[-300:])


# ---------------------------------------------------------------------------
# B11: a recipe-quantised Linear under format_policy='fp32'
# ---------------------------------------------------------------------------
class ChainedLinear(nn.Module):
    """A Linear(16, 8) whose weight is left float, behind Esp32Quantizer's
    dynamic chain written out."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(_chain(x))


class StaticPairLinear(nn.Module):
    """A float Linear(16, 8) behind a static pair written out (scale 2^-6),
    which the partitioner records on the Linear it claims."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

        qd = torch.ops.quantized_decomposed
        q = qd.quantize_per_tensor.default(x, 2 ** -6, 0, -128, 127, torch.int8)
        return self.fc(qd.dequantize_per_tensor.default(q, 2 ** -6, 0, -128,
                                                        127, torch.int8))


def _token_chain(t):
    """torchao's dynamic chain written out, one scale per token (blocks
    [1, ..., K]), as esp32_torchao_config quantises a Linear's input."""
    import torchao  # noqa: F401 -- the torchao ops

    ao = torch.ops.torchao
    block = [1] * (t.dim() - 1) + [int(t.shape[-1])]
    scale, zp = ao.choose_qparams_affine(t, "ASYMMETRIC", block, torch.int8,
                                         -128, 127, None, torch.float32,
                                         torch.int32)
    q = ao.quantize_affine(t, block, scale, zp, torch.int8, -128, 127)
    return ao.dequantize_affine(q, block, scale, zp, torch.int8, -128, 127)


class TokenChainedLinear(nn.Module):
    """ChainedLinear behind torchao's per-token chain instead."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(_token_chain(x))


@case
def test_b11():
    print("B11: a recipe-quantised Linear under format_policy='fp32'")
    x = torch.randn(4, 16)
    programs = [(f"Linear(16, 8), {preset or 'Esp32Quantizer()'}",
                 pt2e(nn.Sequential(nn.Linear(16, 8)), x, preset), preset)
                for preset in (None, "xnnpack_dynamic", "xnnpack_static")]
    programs.append(("ChainedLinear, a float weight",
                     torch.export.export(ChainedLinear().eval(), (x,)), None))
    for name, ep, preset in programs:
        name = f"{name}, fp32 policy"
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep, format_policy="fp32")
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear is refused, the reason naming the "
                  f"policy", len(refused) == 1 and "fp32" in refused[0],
                  f"{[r[:160] for r in refused]}")
            if preset != "xnnpack_static":
                left_chain(name, keep, ops, "addmm", "stays portable (")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, ep.module()(x), rel=1e-5)
            check(f"{name}: the host runner computes the converted model's "
                  f"numbers", ok, detail if got is not None else log[-300:])
    # The model's own activation quantiser on a loop that has none: the fp32
    # loop, a weight stored as bf16, or one quantised here at i16, whose loop
    # quantises its activation to int16 codes itself. A static pair the
    # partitioner reads, or a chain the strip would absorb, was claimed with
    # its record, and preprocess refused the program.
    programs = [(f"StaticPairLinear, {policy} policy", StaticPairLinear(),
                 policy, policy, False)
                for policy in ("fp32", "i16", "bf16")]
    programs += [("ChainedLinear, a float weight, i16 policy", ChainedLinear(),
                  "i16", "i16", True),
                 ("TokenChainedLinear, a float weight, i16 policy",
                  TokenChainedLinear(), "i16", "i16", True)]
    for name, model, policy, named, chain in programs:
        model = model.eval()
        keep = {}
        part, ops, _, err = lower(model, (x,), keep, format_policy=policy)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear is refused, the reason naming the "
                  f"activation quantiser and the format",
                  len(refused) == 1 and "activation quantiser" in refused[0]
                  and named in refused[0], f"{[r[:160] for r in refused]}")
            if chain:
                left_chain(name, keep, ops, "addmm", "stays portable (")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, model(x), rel=1e-5)
            check(f"{name}: the host runner computes the model's numbers", ok,
                  detail if got is not None else log[-300:])


# ---------------------------------------------------------------------------
# B12: a dynamic chain that reaches its op through other data movement
# ---------------------------------------------------------------------------
class ChainMoveLinear(nn.Module):
    """F.linear over Esp32Quantizer's dynamic chain written out (torchao's
    per-token one when `token`) with int8 weight codes, which the delegate
    consumes verbatim, reached through `move`: "t", a transpose of x
    [16, 4] (of x [16, 16] when `token`: the same row width, other rows);
    "slice", the first 16 columns of x [4, 24]; or "select", the first row
    of x [2, 16], a rank-1 input torch spells unsqueeze -> addmm."""

    def __init__(self, move, token=False):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("codes", torch.randint(
            -127, 128, (8, 16), generator=g, dtype=torch.int8))
        self.register_buffer("scales", torch.rand(8, generator=g) / 50 + 0.005)
        self.register_buffer("zeros", torch.zeros(8, dtype=torch.int64))
        self.register_buffer("bias", torch.randn(8, generator=g))
        self.move, self.token = move, token

    def forward(self, x):
        import torch.ao.quantization.fx._decomposed  # noqa: F401 -- the qd ops

        qd = torch.ops.quantized_decomposed
        w = qd.dequantize_per_channel(self.codes, self.scales, self.zeros, 0,
                                      -127, 127, torch.int8)
        v = _token_chain(x) if self.token else _chain(x)
        v = {"t": lambda: v.t(), "slice": lambda: v[:, :16],
             "select": lambda: v[0]}[self.move]()
        return F.linear(v, w, self.bias)


@case
def test_b12():
    print("B12: a dynamic chain that reaches its op through other data movement")
    # to_edge spells an unbatched convolution unsqueeze -> convolution ->
    # squeeze: the chain reaches the convolution through the unsqueeze.
    torch.manual_seed(12)
    convs = [(f"{type(conv).__name__} over {list(shape)}, "
              f"{preset or 'Esp32Quantizer()'}", conv, torch.randn(*shape),
              preset)
             for conv, shape, preset in (
                 (nn.Conv1d(4, 8, 3, padding=1), (4, 16), None),
                 (nn.Conv2d(3, 8, 3, padding=1), (3, 8, 8), None),
                 (nn.Conv2d(3, 8, 3, padding=1), (3, 8, 8), "xnnpack_dynamic"),
                 (nn.Conv1d(4, 8, 3, padding=1), (1, 4, 16), None))]
    for name, conv, x, preset in convs:
        ep = pt2e(nn.Sequential(conv), x, preset)
        keep = {}
        part, ops, _, err = lower(ep, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            left = [l for l in keep["log"].splitlines()
                    if l.startswith("esp32: leaving")]
            check(f"{name}: the convolution takes the chain over and "
                  f"delegates", not left and delegates(ops) == 1
                  and portable(ops, "convolution") == 0, f"{left}")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, ep.module()(x), rel=1e-5)
            check(f"{name}: the host runner computes the converted model's "
                  f"numbers", ok, detail if got is not None else log[-300:])
    # A transpose keeps a per-tensor scheme, so the Linear takes the chain
    # over. A slice or a select is not walked (the scale of a part is not the
    # scale of the whole), nor a transpose under a per-token scheme: the
    # chain is computed in place in front of the move, and the Linear behind
    # it, through more moves too, keeps that value and runs portable.
    cases = [("ChainMoveLinear('t')", ChainMoveLinear("t"),
              torch.randn(16, 4), None, "addmm"),
             ("ChainMoveLinear('slice')", ChainMoveLinear("slice"),
              torch.randn(4, 24), "aten_slice_copy", "addmm"),
             ("ChainMoveLinear('select')", ChainMoveLinear("select"),
              torch.randn(2, 16), "aten_select_copy", "addmm"),
             ("ChainMoveLinear('t', token=True)",
              ChainMoveLinear("t", token=True), torch.randn(16, 16),
              "aten_permute_copy", "addmm")]
    for name, model, x, move, op in cases:
        model = model.eval()
        keep = {}
        part, ops, _, err = lower(model, (x,), keep)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            if move is None:
                left = [l for l in keep["log"].splitlines()
                        if l.startswith("esp32: leaving")]
                check(f"{name}: the Linear takes the chain over and "
                      f"delegates", not left and delegates(ops) == 1
                      and portable(ops, op) == 0, f"{left}")
            else:
                left_chain(name, keep, ops, op,
                           f"reads the dequantised value (and so does "
                           f"aten_{op}_default behind it)")
                left = [l for l in keep["log"].splitlines()
                        if l.startswith("esp32: leaving")]
                check(f"{name}: the value is kept at the move the walk does "
                      f"not pass", bool(left) and f": {move}_" in left[0],
                      f"{left}")
                refused = [r for n, r in reasons(part).items()
                           if f"aten_{op}_" in n]
                check(f"{name}: the Linear is refused, the reason naming the "
                      f"chain computed in place",
                      len(refused) == 1 and "computed in place" in refused[0],
                      f"{[r[:160] for r in refused]}")
            got, log = run_host(keep["program"], (x,))
            with torch.no_grad():
                ok, detail = same_numbers(got, model(x), rel=1e-5)
            check(f"{name}: the host runner computes the model's numbers", ok,
                  detail if got is not None else log[-300:])


# ---------------------------------------------------------------------------
# B13: a reshaped matmul output whose add or mul the gate refuses
# ---------------------------------------------------------------------------
class ResidualDynamicR(nn.Module):
    """A 3-D Linear (a view between the addmm and the add) plus a residual
    whose leading dim is exported dynamic: the add carries a symbolic dim and
    stays portable, so the view must too."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, D)

    def forward(self, x, r):
        return self.fc(x) + r


class GateDynamicR(ResidualDynamicR):
    def forward(self, x, r):
        return self.fc(x) * r


class ResidualIntoConv(nn.Module):
    """Static shapes: the add feeds a claimed conv, so the add gate refuses it
    (it would weld two partitions); the view before it must go with it."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, x, r):
        return self.conv(self.fc(x) + r)


@case
def test_b13():
    print("B13: a reshaped matmul output whose add or mul stays portable")
    x, r = torch.randn(4, 8, D), torch.randn(4, 8, D)
    dyn = (None, {0: torch.export.Dim.AUTO(min=1, max=4)})
    for model, op in ((ResidualDynamicR(), "add"), (GateDynamicR(), "mul")):
        name = type(model).__name__
        part, ops, _, err = lower(model, (x, r), dynamic_shapes=dyn)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            rs = reasons(part)
            check(f"{name}: the {op} is refused for its symbolic dim",
                  any(op in n and v.startswith("dynamic shape")
                      for n, v in rs.items()) and portable(ops, op) == 1,
                  str(rs))
            check(f"{name}: the Linear is still delegated",
                  delegates(ops) == 1 and portable(ops, "addmm") == 0)
    part, ops, _, err = lower(ResidualDynamicR(), (x, r))
    check("ResidualDynamicR static (control) lowers", err is None, err or "")
    if err is None:
        check("the static residual still fuses",
              delegates(ops) == 1 and portable(ops, "add") == 0)
    part, ops, _, err = lower(ResidualIntoConv(),
                              (torch.randn(1, 4, 4, 8), torch.randn(1, 4, 4, 8)))
    check("ResidualIntoConv lowers", err is None, err or "")
    if err is None:
        rs = reasons(part)
        check("ResidualIntoConv: the add is refused with a reason",
              any("add" in n and "feeds a claimed" in v for n, v in rs.items())
              and portable(ops, "add") == 1, str(rs))
        check("ResidualIntoConv: the Linear and the conv are still delegated",
              delegates(ops) == 2 and portable(ops, "addmm") == 0
              and portable(ops, "convolution") == 0)


# ---------------------------------------------------------------------------
# B14: q@kT over a build-time constant operand, or a state
# ---------------------------------------------------------------------------
class QKConstQuery(nn.Module):
    """Learned query tokens attending over the input: softmax(Q k^T) with Q
    a [1, 4, D] parameter."""

    def __init__(self):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, 4, D))

    def forward(self, k):                              # [1, 6, D]
        return torch.softmax(torch.bmm(self.queries, k.transpose(1, 2)), dim=-1)


class QKConstKeys(nn.Module):
    """The input attending over a learned key memory: softmax(q K^T)."""

    def __init__(self):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(1, 6, D))

    def forward(self, q):                              # [1, 4, D]
        return torch.softmax(torch.bmm(q, self.keys.transpose(1, 2)), dim=-1)


class QKStateKeys(nn.Module):
    """Keys that are a state the forward writes, read before the write."""

    def __init__(self):
        super().__init__()
        self.register_buffer("keys", torch.randn(1, 6, D))

    def forward(self, q):                              # [1, 4, D]
        scores = torch.softmax(torch.bmm(q, self.keys.transpose(1, 2)), dim=-1)
        self.keys.mul_(0.5)
        return scores


class QKRuntime(nn.Module):
    """Control: both operands runtime inputs."""

    def forward(self, q, k):
        return torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)


@case
def test_b14():
    import copy

    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

    print("B14: q@kT whose query or key is a build-time constant or a state")
    cases = [(QKConstQuery, torch.randn(1, 6, D), "constant", ()),
             (QKConstKeys, torch.randn(1, 4, D), "constant", ())]
    for cls, x, why, passes in cases:
        name = cls.__name__
        model = cls().eval()
        with torch.no_grad():
            want = copy.deepcopy(model)(x)
        keep = {}
        part, ops, _, err = lower(model, (x,), keep, passes=passes,
                                  quantize_qk=True)
        check(f"{name} lowers", err is None, err or "")
        if err is None:
            refused = [r for n, r in reasons(part).items() if "bmm" in n]
            check(f"{name}: the bmm is refused, the reason naming the "
                  f"{why.split(' is ')[-1]}",
                  bool(refused) and all(why in r for r in refused),
                  f"{[r[:160] for r in refused]}")
            got, log = run_host(keep["program"], (x,))
            ok, detail = same_numbers(got, want)
            check(f"{name}: the host runner computes eager's numbers", ok,
                  detail if got is not None else log[-300:])
    # Keys that are a state the forward writes, read directly. No partition
    # takes a written buffer in, so the delegate
    # reads it at run time, and the q@kT is admitted, judged by what the
    # forward writes. The state's first
    # value goes in the program, which the eager model starts from: the
    # delegate's int8 scores are eager's to int8 error, which keys read as
    # zeros would not be (test_qk_matrix's test_state_read_directly_admitted
    # holds a second call to the state the first wrote).
    x = torch.randn(1, 4, D)
    model = QKStateKeys().eval()
    with torch.no_grad():
        want = copy.deepcopy(model)(x)
    keep = {}
    part, ops, _, err = lower(model, (x,), keep,
                              passes=(InitializedMutableBufferPass(["keys"]),),
                              quantize_qk=True)
    check("QKStateKeys lowers", err is None, err or "")
    if err is None:
        check("QKStateKeys: the q@kT over the state delegates",
              delegates(ops) == 1 and portable(ops, "bmm") == 0,
              f"{[r[:160] for n, r in reasons(part).items() if 'bmm' in n]}")
        got, log = run_host(keep["program"], (x,))
        ok, detail = same_numbers(got, want, rel=2e-2)
        check("QKStateKeys: the host runner computes eager's numbers to the "
              "int8 q@kT's error", ok,
              detail if got is not None else log[-300:])
    q, k = torch.randn(1, 4, D), torch.randn(1, 6, D)
    part, ops, _, err = lower(QKRuntime(), (q, k), quantize_qk=True)
    check("QKRuntime (control) lowers", err is None, err or "")
    if err is None:
        check("a q@kT over runtime operands still delegates",
              delegates(ops) == 1 and portable(ops, "bmm") == 0)


# ---------------------------------------------------------------------------
# B15: a BatchNorm after a conv whose weight a QAT conversion dequantises
# ---------------------------------------------------------------------------
class ConvBnBn(nn.Module):
    """A conv, its BatchNorm, and a second BatchNorm reading the first."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, 3, bias=False)
        self.bn, self.bn2 = nn.BatchNorm2d(16), nn.BatchNorm2d(16)
        with torch.no_grad():
            for bn in (self.bn, self.bn2):
                bn.running_mean.uniform_(-0.5, 0.5)
                bn.running_var.uniform_(0.5, 2.0)
                bn.weight.uniform_(0.5, 1.5)
                bn.bias.uniform_(-0.2, 0.2)

    def forward(self, x):
        return self.bn2(self.bn(self.conv(x)))


class ConvBn(ConvBnBn):
    """The same conv and first BatchNorm alone."""

    def forward(self, x):
        return self.bn(self.conv(x))


@case
def test_b15():
    from executorch.exir import to_edge
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import (
        convert_pt2e,
        prepare_qat_pt2e,
    )

    from executorch.backends.esp32s3 import Esp32Quantizer
    from executorch.backends.esp32s3._passes.fold_conv_bn_pass import fold_conv_bn

    print("B15: a BatchNorm after a conv whose weight a QAT conversion dequantises")
    x = torch.randn(1, 4, 8, 8)
    model = ConvBnBn().eval()
    gm = torch.export.export(model, (x,), strict=False).module()
    prepared = prepare_qat_pt2e(gm, Esp32Quantizer(is_qat=True))
    move_exported_model_to_eval(prepared)
    with torch.no_grad():
        for _ in range(4):
            prepared(torch.randn(1, 4, 8, 8))
    converted = convert_pt2e(prepared)
    with torch.no_grad():
        want = converted(x)

    edge = to_edge(torch.export.export(converted, (x,), strict=False))
    graph = edge.exported_program().graph
    before = len(graph.nodes)
    folded, rejected = fold_conv_bn(edge)
    check("the fold refuses the BatchNorm after the dequantised weight, "
          "inserting nothing",
          folded == 0 and len(graph.nodes) == before
          and any("not a build-time constant" in why for _, why in rejected),
          f"folded {folded}, rejected {rejected}")

    keep = {}
    part, ops, _, err = lower(
        torch.export.export(converted, (x,), strict=False), (x,), keep)
    check("ConvBnBn converted for QAT lowers", err is None, err or "")
    if err is None:
        check("its second BatchNorm runs portable",
              portable(ops, "_native_batch_norm_legit_no_training") == 1,
              f"delegates {delegates(ops)}")
        got, log = run_host(keep["program"], (x,))
        ok, detail = same_numbers(got, want, rel=1e-5)
        check("the host runner computes the recipe's numbers", ok,
              detail if got is not None else log[-300:])
    part, ops, _, err = lower(ConvBn().eval(), (x,))
    check("ConvBn in float (control) lowers", err is None, err or "")
    if err is None:
        check("in float the BatchNorm still folds into the delegated conv",
              portable(ops, "_native_batch_norm_legit_no_training") == 0
              and delegates(ops) == 1 and portable(ops, "convolution") == 0,
              f"delegates {delegates(ops)}")


# ---------------------------------------------------------------------------
# B16: a Linear over a state the forward writes, read directly or through a view
# ---------------------------------------------------------------------------
class StateRead(nn.Module):
    """y = fc(S) + x, then S = 0.5 x + 0.25: a Linear whose activation is a
    [2, 16] buffer the forward writes, read directly, through a view (a
    [1, 2, 16] state read as [2, 16]) or through a transpose (a [16, 2]
    state, which preprocess would fold into a weight, read as its
    activation), so each call reads what the one before wrote. Weights,
    bias, inputs and the first state are eighths: every sum is exact in
    float32, so the fp32 loop, portable addmm and eager agree to the bit."""

    def __init__(self, how):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.how = how
        self.fc = nn.Linear(16, 16)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randint(-8, 9, (16, 16), generator=g) / 8)
            self.fc.bias.copy_(torch.randint(-8, 9, (16,), generator=g) / 8)
        first = torch.randint(-8, 9, (2, 16), generator=g) / 8
        self.register_buffer("state", {"direct": first,
                                       "view": first.view(1, 2, 16),
                                       "transposed": first.t()}[how].clone())
        self.requires_grad_(False)

    def forward(self, x):                                   # [2, 16]
        how = self.how
        s = {"direct": lambda: self.state,
             "view": lambda: self.state.view(2, 16),
             "transposed": lambda: self.state.t()}[how]()
        y = self.fc(s) + x
        new = x * 0.5 + 0.25
        self.state.copy_(new.t() if how == "transposed"
                         else new.view(self.state.shape))
        return y


class StateIndices(nn.Module):
    """y = E[idx] + x, then idx = new: an embedding gathering by a token
    history the forward writes (an int64 [4] buffer), so each call gathers by
    the indices the one before wrote. With `given`, the indices are a graph
    input instead, the reference."""

    def __init__(self, given=False):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.emb = nn.Embedding(32, 16)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randint(-8, 9, (32, 16), generator=g) / 8)
        self.register_buffer("idx", torch.tensor([3, 1, 4, 1]))
        self.given = given
        self.requires_grad_(False)

    def forward(self, x, new):                      # [4, 16], int64 [4]
        if self.given:
            return self.emb(new) + x
        y = self.emb(self.idx) + x
        self.idx.copy_(new)
        return y


def portable_program(model, args, mutable):
    """`model` emitted with no partitioner: every op on portable kernels, a
    written buffer's first value in the program."""
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

    with contextlib.redirect_stdout(io.StringIO()):
        ep = torch.export.export(model.eval(), args)
        return to_edge(ep).to_executorch(config=ExecutorchBackendConfig(
            passes=[InitializedMutableBufferPass(list(mutable))]))


@case
def test_b16():
    import copy

    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

    print("B16: a Linear over a state the forward writes, read directly or "
          "through a view")
    x = torch.randint(-8, 9, (2, 16),
                      generator=torch.Generator().manual_seed(4)) / 8
    runs = {}
    for policy in ("fp32", None):
        for how in ("direct", "view", "transposed"):
            name = f"StateRead({how}), format_policy={policy!r}"
            model = StateRead(how).eval()
            module = copy.deepcopy(model)
            with torch.no_grad():
                want = [module(x).clone() for _ in range(2)]
            keep = {}
            options = {"format_policy": policy} if policy else {}
            part, ops, _, err = lower(
                copy.deepcopy(model), (x,), keep,
                passes=[InitializedMutableBufferPass(["state"])], **options)
            check(f"{name} lowers", err is None, err or "")
            if err is not None:
                continue
            refused = [r for n, r in reasons(part).items() if "addmm" in n]
            check(f"{name}: the Linear over the state is delegated",
                  delegates(ops) == 1 and portable(ops, "addmm") == 0
                  and not refused,
                  f"{delegates(ops)} delegate(s), refused "
                  f"{[r[:160] for r in refused]}")
            if how == "direct":
                check(f"{name}: the state is a partition input, read at run "
                      f"time", state_read_by_delegate(keep["program"], "state"))
            got, log = run_steps(keep["program"], (x,), 2)
            check(f"{name}: two calls run", got is not None,
                  "" if got is not None else log[-300:])
            runs[(policy, how)] = got
            if got is None:
                continue
            if policy == "fp32":
                ported, plog = run_steps(portable_program(
                    copy.deepcopy(model), (x,), ["state"]), (x,), 2)
                check(f"{name}: each call is the portable program's, bit for "
                      f"bit (the second reads the state the first wrote)",
                      ported is not None and all(
                          a.tobytes() == b.tobytes()
                          for a, b in zip(got, ported)),
                      plog[-300:] if ported is None else "")
                for step, out in enumerate(want):
                    ok, detail = same_numbers(got[step], out, rel=0)
                    check(f"{name}: call {step + 1} is eager's, exactly", ok,
                          detail)
            else:
                for step, out in enumerate(want):
                    ok, detail = same_numbers(got[step], out, rel=2e-2)
                    check(f"{name}: call {step + 1} is eager's to the int8 "
                          f"loop's error", ok, detail)
    direct = runs.get((None, "direct"))
    for how, through in (("view", "a view"), ("transposed", "a transpose")):
        other = runs.get((None, how))
        if direct and other:
            check(f"format_policy=None: the state read directly computes the "
                  f"bits of the state read through {through}, call after "
                  f"call", all(a.tobytes() == b.tobytes()
                               for a, b in zip(direct, other)))
    # An embedding over indices the forward writes: the second call gathers
    # by what the first wrote, the rows the same indices given as an input
    # gather, bit for bit.
    name = "StateIndices"
    new = torch.tensor([7, 0, 31, 7])
    x4 = torch.randint(-8, 9, (4, 16),
                       generator=torch.Generator().manual_seed(6)) / 8
    model = StateIndices().eval()
    module = copy.deepcopy(model)
    with torch.no_grad():
        want = [module(x4, new).clone() for _ in range(2)]
    keep = {}
    part, ops, _, err = lower(copy.deepcopy(model), (x4, new), keep,
                              passes=[InitializedMutableBufferPass(["idx"])])
    check(f"{name} lowers", err is None, err or "")
    if err is None:
        refused = [r for n, r in reasons(part).items() if "embedding" in n]
        check(f"{name}: the gather over the token history is delegated, the "
              f"history a partition input",
              portable(ops, "embedding") == 0 and not refused
              and state_read_by_delegate(keep["program"], "idx"),
              f"{delegates(ops)} delegate(s), refused "
              f"{[r[:160] for r in refused]}")
        got, log = run_steps(keep["program"], (x4, new), 2)
        check(f"{name}: two calls run", got is not None,
              "" if got is not None else log[-300:])
        given = {}
        _, _, _, gerr = lower(StateIndices(given=True).eval(), (x4, new),
                              given)
        refs = []
        for indices in (torch.tensor([3, 1, 4, 1]), new):
            ref, rlog = (run_host(given["program"], (x4, indices))
                         if gerr is None else (None, gerr))
            refs.append(ref)
        if got is not None:
            check(f"{name}: each call gathers the rows its indices given as "
                  f"an input gather, bit for bit (the second, the history "
                  f"the first wrote)",
                  all(r is not None and r.tobytes() == g.tobytes()
                      for r, g in zip(refs, got)), str(gerr or ""))
            for step, out in enumerate(want):
                ok, detail = same_numbers(got[step], out, rel=2e-2)
                check(f"{name}: call {step + 1} is eager's to the int8 "
                      f"table's error", ok, detail)


def main():
    torch.manual_seed(0)
    for test in (test_b1, test_b2, test_b3, test_b4, test_b5, test_b6,
                 test_b7, test_b8, test_b9, test_b10, test_b11, test_b12,
                 test_b13, test_b14, test_b15, test_b16):
        test.__wrapped__()
    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("every former crash is now a stated refusal; the rest still delegates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
