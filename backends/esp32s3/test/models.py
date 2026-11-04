# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic fixture models and their expected checksums."""

import torch
import torch.nn as nn

from executorch.backends.esp32s3.quantizer.mirrors import fma_f32
from executorch.backends.esp32s3.test.reference import (
    addmm_i8_absmax,
    addmm_iq_absmax,
    addmm_k_outer,
    bmm_qk_i8,
)


class MLP(nn.Module):
    def __init__(self, d=96, h=64):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, h), nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.c2(x))
        x = torch.max_pool2d(x, 2)
        return self.fc(x.flatten(1))


class Block(nn.Module):
    """One pre-norm transformer block: D=96, H=4, F=66."""

    def __init__(self, d=96, h=4, f=66):
        super().__init__()
        self.h, self.dh = h, d // h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        shape = (B, T, self.h, self.dh)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-2, -1) / self.dh**0.5, dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        y = self.n2(x)
        return x + self.down(torch.nn.functional.silu(self.gate(y)) * self.up(y))


# name -> (factory, input shape). Seeded at build time for reproducibility.
class LSTMNet(nn.Module):
    """One LSTM layer plus a projection, returning a single tensor.

    The projection is not decoration: `nn.LSTM` returns `(output, (h, c))`, and
    torch.export rejects a nested tuple in the output pytree while the runner wants one
    output tensor anyway. Taking the last timestep also makes the checksum depend on the
    whole recurrence rather than on one step, so a cache-like bug that broke step
    ordering would actually show up.
    """

    def __init__(self, d=16, h=32):
        super().__init__()
        self.rnn = nn.LSTM(d, h, num_layers=1, batch_first=True)
        self.fc = nn.Linear(h, d)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :])


class GRUNet(nn.Module):
    """Same shape as LSTMNet, different gate algebra.

    Worth having both: the LSTM decomposes through `tanh` and a cell state, the GRU
    through `split_with_sizes_copy` and a convex combination, so they cover different
    corners of the same decomposition machinery.
    """

    def __init__(self, d=16, h=32):
        super().__init__()
        self.rnn = nn.GRU(d, h, num_layers=1, batch_first=True)
        self.fc = nn.Linear(h, d)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y[:, -1, :])


class Conv1dNet(nn.Module):
    """conv1d, depthwise and grouped, in one graph.

    All three reach the same `aten::convolution.out` kernel, so this is not three tests
    of three kernels -- it is one test of three argument shapes the kernel handles
    internally. conv1d is the interesting one structurally: ExecuTorch has no 1-D
    convolution kernel, and `op_convolution.cpp` gets there by unsqueezing 3-D input
    into the 4-D path, which this row exercises.

    Depthwise (groups == channels) and grouped (groups == 2) are the argument
    combinations most likely to be wrong in a hand-written kernel, since they are the
    ones where the input-channel range per output channel stops being "all of them".
    """

    def __init__(self, c=8, t=16):
        super().__init__()
        self.plain = nn.Conv1d(c, c, 3, padding=1)
        self.depthwise = nn.Conv1d(c, c, 3, padding=1, groups=c)
        self.grouped = nn.Conv1d(c, c, 3, padding=1, groups=2)
        self.fc = nn.Linear(c * t, 8)

    def forward(self, x):
        x = torch.relu(self.plain(x))
        x = torch.relu(self.depthwise(x))
        x = self.grouped(x)
        return self.fc(x.flatten(1))


class PoolNet(nn.Module):
    """max/avg/min pooling at 1-D and 2-D, all onto ExecuTorch's two 2-D kernels.

    ExecuTorch portable ships exactly `max_pool2d_with_indices.out` and
    `avg_pool2d.out`. Everything here reaches one of those:

      * 1-D pooling decomposes to unsqueeze -> the 2-D kernel -> squeeze, the same
        trick conv1d uses. No 1-D kernel exists or is needed.
      * There is no MinPool in PyTorch at all; `-max_pool(-x)` is the idiom, and it
        costs one `neg` on top of the kernel already present.

    So this model is cheap coverage: three pooling semantics across two ranks, and not
    one new kernel between them. 3-D pooling is deliberately absent -- it has no
    portable kernel, and a model containing it would fail at Method::init rather than
    here.
    """

    def __init__(self, c=4):
        super().__init__()
        self.fc = nn.Linear(c * 4 * 4 + c * 8, 8)

    def forward(self, x):
        # x: [1, C, 16, 16]. Everything goes through torch.nn.functional -- only some
        # of these exist at the torch top level (torch.avg_pool2d does not).
        f = torch.nn.functional
        a = f.max_pool2d(x, 2)                          # -> [1,C,8,8]
        a = -f.max_pool2d(-a, 2)                        # min-pool idiom -> [1,C,4,4]
        b = f.avg_pool2d(x, 2)                          # -> [1,C,8,8]
        lane = b.mean(dim=2)                            # -> [1,C,8], a 1-D signal
        c1 = f.max_pool1d(lane, 2)                      # -> [1,C,4]
        c2 = f.avg_pool1d(lane, 2)                      # -> [1,C,4]
        return self.fc(torch.cat([a.flatten(1), c1.flatten(1), c2.flatten(1)], dim=1))


class BmmHalfFallbackNet(nn.Module):
    """One valid non-fp32 BMM that must use the target override's fallback.

    The public input/output stay fp32 so the generic parity runner can fill and
    checksum them. Only the BMM operands/result are fp16, exercising ET's linked
    portable kernel rather than the ESP32 contiguous-fp32 fast path.
    """

    def forward(self, x):
        left = x[:, :12].reshape(2, 3, 4).to(torch.float16)
        right = x[:, 12:].reshape(2, 4, 5).to(torch.float16)
        return torch.bmm(left, right).to(torch.float32)


SPECS = {
    "mlp": (MLP, (1, 96)),
    "bmmhalf": (BmmHalfFallbackNet, (2, 32)),
    "convnet": (ConvNet, (1, 1, 16, 16)),
    "block": (Block, (1, 8, 96)),
    "lstm": (LSTMNet, (1, 8, 16)),
    "gru": (GRUNet, (1, 8, 16)),
    "conv1d": (Conv1dNet, (1, 8, 16)),
    "pool": (PoolNet, (1, 4, 16, 16)),
}


def ramp_input(shape):
    """Bit-exactly what main.cpp's fill_inputs writes:

        p[j] = (float)j * 0.0078125f - 0.5f;

    The step is 2^-7 rather than 0.01 on purpose. 0.01f is not representable, and the
    compiler contracts j*0.01f - 0.5f into a single madd.s -- one rounding instead of
    two -- which would put x[75] one ulp below 0.25 on device while any Python spelling
    gives exactly 0.25. Invisible at fp32 tolerances, but it sits on an int8 rounding
    tie and shifts a whole quantum. With an exactly representable step both sides
    agree bit for bit under any contraction.
    """
    import numpy as np

    n = 1
    for d in shape:
        n *= d
    vals = np.arange(n, dtype=np.float32) * np.float32(0.0078125) - np.float32(0.5)
    return torch.from_numpy(vals).reshape(shape)


def checksum(t):
    """Exactly what main.cpp prints: sum(out[i] * (i+1)) over the flat tensor."""
    import numpy as np

    f = np.asarray(t.detach() if hasattr(t, "detach") else t).reshape(-1)
    return sum(float(f[i]) * (i + 1) for i in range(f.size))


def output_bits(t):
    """Exactly what main.cpp prints as `output bits`: FNV-1a (offset 2166136261,
    prime 16777619) over the float32 output's bytes, as 8 hex digits.

    The checksum prints six decimals and cannot see a one-ulp change in one
    element; this sees every bit, which is what lets a device mirror be held
    to the device exactly rather than to a tolerance.
    """
    import numpy as np

    data = np.ascontiguousarray(
        np.asarray(t.detach() if hasattr(t, "detach") else t,
                   dtype=np.float32)).tobytes()
    h = 2166136261
    for byte in data:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return f"{h:08x}"


def mlp_fp32_backend_out():
    """The MLP through opcode 1 (mlp_fp32_backend): both Linears k-outer, the
    relu a clamp to [0, inf) between them, which keeps a -0 as the device's
    `y < lo ? lo : y` does."""
    import numpy as np
    import torch

    factory, shape = SPECS["mlp"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(np.float32)
    W1 = m.fc1.weight.detach().numpy().T.astype(np.float32).copy()
    b1 = m.fc1.bias.detach().numpy().astype(np.float32)
    W2 = m.fc2.weight.detach().numpy().T.astype(np.float32).copy()
    b2 = m.fc2.bias.detach().numpy().astype(np.float32)
    h = addmm_k_outer(x, W1, b1)
    h = np.where(h < 0, np.float32(0), h)
    return addmm_k_outer(h, W2, b2)


def mlp_iq_out(fmt="i8"):
    """The MLP's output as the delegate computes it at one rung of the ladder.

    One model exported three ways is the cheapest honest test of the ladder: the graph,
    the partition and the arithmetic order are identical, so any disagreement is the
    format path and nothing else. The three checksums differ from each other by design
    -- i4 is a coarser quantiser, not a different computation -- so each is held against
    its own reference rather than against the fp32 one.
    """
    import numpy as np
    import torch

    factory, shape = SPECS["mlp"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(np.float32)
    h = addmm_iq_absmax(x, m.fc1.weight.detach().numpy().astype(np.float32),
                        m.fc1.bias.detach().numpy().astype(np.float32), fmt)
    h = np.maximum(h, np.float32(0))
    return addmm_iq_absmax(h, m.fc2.weight.detach().numpy().astype(np.float32),
                           m.fc2.bias.detach().numpy().astype(np.float32), fmt)


def mlp_i8_out():
    """The MLP's output as the int8 delegate computes it."""
    return mlp_iq_out("i8")


def mlp_static_out(sx1, sx2):
    """The MLP's output under CALIBRATED static activation scales.

    The scales are taken as ARGUMENTS -- export_delegated feeds in exactly what
    calibrate_static_act recorded off the edge graph -- so the reference and the
    blob share one set of calibration numbers instead of two hopefully-identical
    replays (the edge graph's addmm on a materialised permute need not be bitwise
    the eager Linear).

    fc1 reads the boundary input, so for M=1 its static scale IS the dynamic one
    and that leg stays bit-identical. fc2 reads the chained scratch intermediate,
    whose device value differs from the fp32 calibration tensor -- its static
    scale is genuinely different from per-row absmax, which is what lets this row
    catch a runtime that quietly ignores the flag.
    """
    import numpy as np
    import torch

    factory, shape = SPECS["mlp"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(np.float32)
    h = addmm_iq_absmax(x, m.fc1.weight.detach().numpy().astype(np.float32),
                        m.fc1.bias.detach().numpy().astype(np.float32), "i8",
                        sx_fixed=sx1)
    h = np.maximum(h, np.float32(0))
    return addmm_iq_absmax(h, m.fc2.weight.detach().numpy().astype(np.float32),
                           m.fc2.bias.detach().numpy().astype(np.float32), "i8",
                           sx_fixed=sx2)


def block_iq_out(fmt="i8", shape=None):
    """The block with every Linear replaced by the selected backend arithmetic.

    Everything else -- layer norm, softmax, silu, the two bmms, the residuals -- stays
    in torch, because on device those run the same portable kernels whose parity is
    already established (block passes at 2.7e-07). So this isolates the delegate: if it
    matches, the selected learned-op format is right; if not, the Linears are
    where to look.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    factory, default_shape = SPECS["block"]
    shape = default_shape if shape is None else shape
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape)

    def make_q_forward(mod):
        W = mod.weight.detach().numpy().astype(np.float32)
        b = (mod.bias.detach().numpy().astype(np.float32)
             if mod.bias is not None else np.zeros(W.shape[0], dtype=np.float32))

        def fwd(inp):
            k = inp.shape[-1]
            flat = inp.detach().reshape(-1, k).contiguous().numpy().astype(np.float32)
            y = addmm_iq_absmax(flat, W, b, fmt=fmt)
            return torch.from_numpy(y).reshape(*inp.shape[:-1], W.shape[0])

        return fwd

    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            mod.forward = make_q_forward(mod)

    with torch.no_grad():
        return m(x)


def block_i8_out(shape=None):
    return block_iq_out("i8", shape)


class MambaNet(nn.Module):
    """A minimal but structurally faithful Mamba (selective SSM) block.

    The architecture family no other row represents: a recurrence
    whose transition depends on the INPUT (dt, B, C are projections of x), which is
    what separates Mamba from the LSTM/GRU rows -- their gates modulate a fixed
    transition; here the transition matrix itself is data-dependent
    (dA = exp(dt*A)). Everything else is the standard block: in-projection with a
    gate branch, causal depthwise conv1d, softplus dt, per-channel skip D, SiLU
    gating, out-projection.

    The scan is unrolled over a fixed T like the RNN rows, so recurrence coverage is
    not conflated with dynamic-shape coverage. Fixed-point Mamba is real (Quamba,
    arXiv 2410.13229), so this row is the lowering prerequisite for going there.
    """

    def __init__(self, d=16, d_inner=32, d_state=8, dt_rank=4, T=8):
        super().__init__()
        self.T = T
        self.in_proj = nn.Linear(d, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, 3, padding=2, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.d_state, self.dt_rank = d_state, dt_rank

    def forward(self, x):
        B_, T, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        # causal depthwise conv: pad left-heavy, trim the future
        xi = self.conv(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xi = torch.nn.functional.silu(xi)
        dbc = self.x_proj(xi)
        dt = dbc[..., :self.dt_rank]
        Bm = dbc[..., self.dt_rank:self.dt_rank + self.d_state]
        Cm = dbc[..., self.dt_rank + self.d_state:]
        dt = torch.nn.functional.softplus(self.dt_proj(dt))       # [B,T,din]
        A = -torch.exp(self.A_log)                                # [din,ds]
        dA = torch.exp(dt.unsqueeze(-1) * A)                      # [B,T,din,ds]
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)
        h = torch.zeros(B_, dA.shape[2], self.d_state)
        ys = []
        for t in range(self.T):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + xi * self.D
        return self.out_proj(y * torch.nn.functional.silu(z))


class KANNet(nn.Module):
    """A Kolmogorov-Arnold layer: learnable B-spline activations on edges.

    The parameter-efficiency claim is upstream's business; what the backend must prove
    is that the SHAPE of the computation lowers, and it is an unusual one -- the
    heavy op is not a matmul but the de Boor recursion building the spline basis:
    comparisons (ge/lt on a knot grid), boolean logic, casts, and k rounds of
    broadcasted rational blends, THEN two matmuls (base path + spline path). Grid is a
    fixed buffer; degree k=3, grid_size 5, so bases are [B, din, 8].
    """

    def __init__(self, din=16, dout=8, grid_size=5, k=3):
        super().__init__()
        self.k = k
        h = 2.0 / grid_size
        grid = (torch.arange(-k, grid_size + k + 1, dtype=torch.float32) * h - 1.0)
        self.register_buffer("grid", grid.expand(din, -1).contiguous())
        self.base_w = nn.Parameter(torch.randn(dout, din) * 0.1)
        self.spline_w = nn.Parameter(torch.randn(dout, din * (grid_size + k)) * 0.1)
        self.head = nn.Linear(dout, 4)

    def forward(self, x):
        g = self.grid
        xe = x.unsqueeze(-1)
        b = ((xe >= g[:, :-1]) & (xe < g[:, 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            b = ((xe - g[:, :-(p + 1)]) / (g[:, p:-1] - g[:, :-(p + 1)]) * b[..., :-1]
                 + (g[:, p + 1:] - xe) / (g[:, p + 1:] - g[:, 1:-p]) * b[..., 1:])
        y = (torch.nn.functional.linear(torch.nn.functional.silu(x), self.base_w)
             + torch.nn.functional.linear(b.flatten(1), self.spline_w))
        return self.head(y)


class LoopedNet(nn.Module):
    """Cross-layer weight sharing: ONE transformer block, applied three times.

    Covers the shared-weight mechanism behind two different ideas at once -- ALBERT
    (parameter sharing to shrink the artifact, plus a factorised low-dim embedding,
    stood in for here by the 8->32 up-projection) and looped/universal transformers
    (iterative refinement through the same weights). Different motivations, identical
    backend stress: export UNROLLS the loop, so one weight tensor is consumed by
    three call sites, and anything per-call-site -- partition tagging, per-partition
    weight embedding -- silently triples the artifact unless deduplication works.
    That is measured in export_delegated's shared-weight check rather than assumed.
    """

    def __init__(self, e=8, d=32, loops=3):
        super().__init__()
        self.up = nn.Linear(e, d)
        self.block = Block(d=d, h=4, f=24)
        self.head = nn.Linear(d, 8)
        self.loops = loops

    def forward(self, x):
        x = self.up(x)
        for _ in range(self.loops):
            x = self.block(x)
        return self.head(x)


class ActsNet(nn.Module):
    """The soft-activation gauntlet, chained so every kernel's OUTPUT feeds the next.

    Chaining is deliberate: parallel branches would let one bad kernel hide in a sum,
    while a chain propagates any error into everything downstream. A Linear mid-chain
    re-mixes so saturating activations cannot collapse the signal. Covers: mish,
    softplus, hardswish, hardsigmoid, elu, leaky_relu, prelu (learnable), softsign,
    tanhshrink, glu (gated halving), and snake (x + sin^2(ax)/a, the periodic
    activation), which brings aten::sin onto the device. A
    last-axis and a middle-axis softmax are combined near the end so the target
    softmax override and its portable non-last-axis fallback stay live together.
    """

    def __init__(self, d=32):
        super().__init__()
        self.prelu = nn.PReLU()
        self.mix = nn.Linear(d, d)
        self.out = nn.Linear(d // 2, 8)

    def forward(self, x):
        f = torch.nn.functional
        x = f.mish(x)
        x = f.softplus(x)
        x = f.hardswish(x)
        x = f.tanhshrink(x)
        x = self.mix(x)
        x = f.softsign(x)
        x = f.elu(x)
        x = f.leaky_relu(x, 0.1)
        x = self.prelu(x)
        x = f.hardsigmoid(x)
        x = x + torch.sin(x * 2.0) * torch.sin(x * 2.0) * 0.5   # snake, a=2
        x = torch.softmax(x, dim=-1) + torch.softmax(x, dim=1)
        x = f.glu(x, dim=-1)                                    # 32 -> 16
        return self.out(x)


class MemplanKernels(nn.Module):
    """memory_plan="esp32"'s QEMU row: the in-place kernels no other row runs
    over a dying operand. clamp writes over neg's result, hardtanh over
    clamp's, then sub over hardtanh's (operand 0) and over that sub's (operand
    1); x, a graph input read to the end, never donates. Every op is exact in
    IEEE arithmetic (neg, min, max, one rounding per subtraction), so eager's
    bits are the device's whatever the plan."""

    def forward(self, x):
        t = torch.neg(x)
        a = torch.clamp(t, -0.375, 0.25)
        b = nn.functional.hardtanh(a, -0.25, 0.125)
        c = b - x
        return x - c


class NormsNet(nn.Module):
    """Every normalisation flavour at once, including the channels-last problem.

    LayerNorm and RMSNorm normalise the LAST dim, but vision tensors are NCHW with
    channels FIRST -- so normalising channels needs permute -> norm -> permute, the
    dim-order dance that is genuinely awkward on embedded runtimes. The RMSNorm
    branches are the interesting ones: after the permutes they present both scaled
    RMSNorm and Glimmer's weightless RMSNorm. This row exercises both custom-op
    shims on [1,8,8,4] rows of width 4, including the null-weight LX7 fast path.

    BatchNorm runs in eval mode as its own kernel; a conv -> bn pair (cbn_c, cbn_b)
    is folded into the conv's weights at export time instead.
    """

    def __init__(self, c=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.gn = nn.GroupNorm(2, c)
        self.inorm = nn.InstanceNorm2d(c, affine=True)
        self.ln = nn.LayerNorm(c)
        # Deliberately outside the target's one-trailing-axis specialization:
        # an opt-in fast LayerNorm build must retain the portable two-axis path.
        self.ln2 = nn.LayerNorm((8, 8))
        self.rms = nn.RMSNorm(c)
        # The fold target: conv -> bn, which apply_passes collapses into the conv's
        # weights AOT. self.bn above is standalone (no conv producer) and must SURVIVE
        # the pass -- one model carrying both the folded and the unfoldable case is
        # what keeps the batch-norm kernel in the derived op list while proving the
        # fold fires where it should.
        self.cbn_c = nn.Conv2d(c, c, 3, padding=1)
        self.cbn_b = nn.BatchNorm2d(c)
        self.head = nn.Linear(c * 8 * 8, 6)

    def forward(self, x):
        a = self.bn(x)
        b = self.gn(x)
        c = self.inorm(x)
        d = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        d2 = self.ln2(x)
        e = self.rms(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f2 = self.cbn_b(self.cbn_c(x))
        g = x.permute(0, 2, 3, 1)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + 1e-5)
        g = g.permute(0, 3, 1, 2)
        return self.head((a + b + c + d + d2 + e + f2 + g).flatten(1))


# More coverage rows, registered in the SPECS.update below because their classes are
# defined after the SPECS literal. Same contract: (factory, float input shape).
class AttnQK(nn.Module):
    """Single-head attention core with the q@kT bmm written out explicitly.

    The fixture for OPT-IN int8 q@kT: with Esp32Partitioner(quantize_qk=True)
    the pre-softmax bmm (and the transpose feeding it) becomes OP_BMM_DYN_I8 --
    both operands quantised dynamically at execute, a delegate op with
    no weights at all. The p@v bmm must stay portable: post-softmax rows have
    absmax ~1 by construction, so int8 there is a precision cut with no
    bandwidth story behind it, and the gate's softmax-consumer test is exactly
    what tells the two bmms apart.
    """

    def __init__(self, d=32, T=8):
        super().__init__()
        self.d = d
        self.wq, self.wk = nn.Linear(d, d), nn.Linear(d, d)
        self.wv = nn.Linear(d, d)

    def forward(self, x):                      # [B, T, d]
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        s = torch.bmm(q, k.transpose(1, 2)) / (self.d ** 0.5)
        p = torch.softmax(s, dim=-1)
        return torch.bmm(p, v)


class AttnKC8(AttnQK):
    """AttnQK with its keys kept the way the persistent int8 K cache keeps them.

    The keys are quantised once by espressif::quantize_rows_i8 and read back in
    kcache_i8.dequantized_keys' spelling, which Esp32Partitioner(quantize_qk=
    True) folds into OP_BMM_KCACHE_I8: the codes and scales are read in place
    and only the queries quantise at execute. Same modules, so the same seeded
    weights as AttnQK, and a key's scale depends only on that key -- so the
    delegated row's reference is attnqk_qk's, exactly (attnkc8_qk_out).
    Batch two, as for AttnQK, so the batch offsets into the codes and the
    scales are exercised; d=32 keeps every key row one whole PIE vector pair,
    the in-place path (the Glimmer fixture's 8-wide heads take the staged one).
    """

    def forward(self, x):                      # [B, T, d]
        from executorch.backends.esp32s3.ops.kcache_i8 import (
            QUANTIZE_ROWS_I8,
            dequantized_keys,
        )

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        codes, scales = QUANTIZE_ROWS_I8(k)
        keys = dequantized_keys(codes, scales)
        s = torch.bmm(q, keys.transpose(1, 2)) / (self.d ** 0.5)
        p = torch.softmax(s, dim=-1)
        return torch.bmm(p, v)


class AttnQKScores(AttnQK):
    """AttnQK returning its raw q@kT scores first, for the parity row
    attnqk_qk. Output 0 is what main.cpp prints `output bits` for, so the row
    holds the delegate's OP_BMM_DYN_I8 result to its mirror bit for bit
    (attnqk_qk_scores) rather than the attention output to a tolerance; the
    attention output still follows. Same modules, so the same seeded weights
    as AttnQK, and the same delegate: the bmm's output already left it for
    the portable scale."""

    def forward(self, x):                      # [B, T, d]
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        p = torch.softmax(scores / (self.d ** 0.5), dim=-1)
        return scores, torch.bmm(p, v)


class AttnKC8Scores(AttnKC8):
    """AttnKC8 returning its raw scores first, for the parity row attnkc8_qk:
    OP_BMM_KCACHE_I8's result, whose reference is attnqk_qk's scores, bit for
    bit (the same codes, the same scales, the same order)."""

    def forward(self, x):                      # [B, T, d]
        from executorch.backends.esp32s3.ops.kcache_i8 import (
            QUANTIZE_ROWS_I8,
            dequantized_keys,
        )

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        codes, scales = QUANTIZE_ROWS_I8(k)
        scores = torch.bmm(q, dequantized_keys(codes, scales).transpose(1, 2))
        p = torch.softmax(scores / (self.d ** 0.5), dim=-1)
        return scores, torch.bmm(p, v)


class RoPEFullNet(nn.Module):
    """HF rotate-half RoPE with a deliberately non-duplicated full table.

    A shim that read only one half would still pass a duplicated-half table.
    Keeping the two halves distinct makes the device row exercise the full-width
    table branch rather than merely registering the fused operator.
    """

    def __init__(self, t=4, d=8):
        super().__init__()
        vals = torch.arange(t * d, dtype=torch.float32).reshape(t, d)
        self.register_buffer("cos", torch.cos(vals * 0.071 + 0.13))
        self.register_buffer("sin", torch.sin(vals * 0.053 - 0.17))

    def forward(self, x):
        half = x.shape[-1] // 2
        rotate_half = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * self.cos + rotate_half * self.sin


def fast_expf_f32_reference(value):
    """Host transcription of ``fast_expf`` (runtime/fast_math.h) as
    the S3 build compiles it.

    The build passes no -ffp-contract flag, so GCC fuses wherever a product's
    only use is an add, and the disassembly confirms exactly where: both
    range-reduction steps (msub.s, madd.s), all five Horner steps and
    ``p*z + r`` are fused multiply-adds; ``x*log2e``, ``r*r``, the ``+ 1`` and
    the power-of-two scaling round on their own. fma_f32 models the fused
    steps, which is what makes this bit-exact rather than merely <= 1 ulp
    close -- the attention-tail golden depends on that.
    """
    import numpy as np

    f32 = np.float32
    x = np.asarray(value, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        fx = (x * f32(1.44269504088896341)).astype(np.float32)
        rounded = (fx + np.where(fx >= 0, f32(0.5), f32(-0.5))).astype(np.float32)
        # trunc.s; the early returns below cover every input this would overflow.
        n = np.trunc(np.nan_to_num(rounded, nan=0.0, posinf=0.0, neginf=0.0))
        n = n.astype(np.int64)
        fn = n.astype(np.float32)
        r = fma_f32(-fn, f32(0.693359375), x)
        r = fma_f32(fn, f32(2.12194440e-4), r)      # r - fn * -2.12194440e-4f
        z = (r * r).astype(np.float32)
        p = np.full(x.shape, f32(1.9875691500e-4), dtype=np.float32)
        for coefficient in (1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2,
                            1.6666665459e-1, 5.0000001201e-1):
            p = fma_f32(p, r, f32(coefficient))
        p = fma_f32(p, z, r)
        p = (p + f32(1.0)).astype(np.float32)
        p = np.where(n > 127, (p + p).astype(np.float32), p)
        n = np.clip(n, -126, 127)
        scale = ((n + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
        y = (p * scale).astype(np.float32)
    y = np.where(x < f32(-87.3365447), f32(0.0), y)
    y = np.where(x > f32(88.7228317), f32(np.inf), y)
    return np.where(np.isnan(x), x, y).astype(np.float32)


def fast_tanhf_f32_reference(value):
    """Host transcription of ``fast_tanhf`` (fast_math.h) as the S3 build
    compiles it, from the parity firmware's disassembly: below
    |x| = 0.625 the four Horner steps and ``p*z*x + x`` are madd.s (fl(p*z)
    first, then fused with x); above, 2|x| (as a + a), fast_expf, ``+ 1``, the
    ROM's correctly rounded division into 2 and ``1 - q`` each round alone.
    """
    import numpy as np

    f32 = np.float32
    x = np.asarray(value, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        z = (x * x).astype(f32)
        p = np.full(x.shape, f32(-5.70498872745e-3), dtype=f32)
        for coefficient in (2.06390887954e-2, -5.37397155531e-2,
                            1.33314422036e-1, -3.33332819422e-1):
            p = fma_f32(p, z, f32(coefficient))
        small = fma_f32((p * z).astype(f32), x, x)
        a = np.abs(x)
        e = fast_expf_f32_reference((a + a).astype(f32))
        t = (f32(1.0) - (f32(2.0) / (e + f32(1.0)).astype(f32))).astype(f32)
        large = np.where(x < 0, -t, t)
        saturated = np.where(x > 0, f32(1.0), f32(-1.0))
        y = np.where(a < f32(0.625), small,
                     np.where(a > f32(9.1), saturated, large))
    return np.where(x == 0, x, y).astype(f32)


def fast_gelu_f32_reference(value):
    """Host transcription of ``fast_gelu`` (fast_erfcf inlined), as compiled:
    0.5x and the argument -x/sqrt2 round alone; ``1 + 0.5z`` is one madd.s,
    then the ROM division; all eight Chebyshev steps are madd.s;
    ``-z*z - 1.26551223f`` is one msub.s and ``+ t*p`` another madd.s; then
    fast_expf, ``t * e``, ``2 - ans`` for a negative argument and the final
    product each round alone (the disassembly of apply_act).
    """
    import numpy as np

    f32 = np.float32
    x = np.asarray(value, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        half_x = (x * f32(0.5)).astype(f32)
        arg = (x * f32(-0.70710678118654752)).astype(f32)
        z = np.abs(arg)
        t = (f32(1.0) / fma_f32(z, f32(0.5), f32(1.0))).astype(f32)
        p = np.full(x.shape, f32(0.17087277), dtype=f32)
        for coefficient in (-0.82215223, 1.48851587, -1.13520398, 0.27886807,
                            -0.18628806, 0.09678418, 0.37409196, 1.00002368):
            p = fma_f32(p, t, f32(coefficient))
        u = fma_f32(t, p, fma_f32(-z, z, f32(-1.26551223)))
        ans = (t * fast_expf_f32_reference(u)).astype(f32)
        erfc = np.where(arg >= 0, ans, (f32(2.0) - ans).astype(f32))
        return (half_x * erfc).astype(f32)


def clamp_f32(y, lo, hi):
    """The fused output clamp (clamp_output): ``y < lo ? lo : y`` then
    ``y > hi ? hi : y``, so NaN and -0 pass through as the device leaves them."""
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    y = np.where(y < np.float32(lo), np.float32(lo), y)
    return np.where(y > np.float32(hi), np.float32(hi), y).astype(np.float32)


def silu_f32_device(y):
    """The fused SiLU: y / fl(fast_expf(-y) + 1), the division correctly
    rounded (apply_act)."""
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return (y / (fast_expf_f32_reference(-y) + np.float32(1.0)).astype(
            np.float32)).astype(np.float32)


def sigmoid_f32_device(y):
    """The fused sigmoid: 1 / fl(fast_expf(-y) + 1) (apply_act)."""
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return (np.float32(1.0) / (fast_expf_f32_reference(-y) + np.float32(
            1.0)).astype(np.float32)).astype(np.float32)


def gelu_tanh_f32_device(y):
    """The fused tanh-GELU: ExecuTorch's formula in fp32, where the build fuses
    ``y + 0.044715 * cube`` into one madd.s and rounds the rest alone --
    fl(fl(0.5y) * fl(1 + fast_tanhf(fl(sqrt(2/pi) * that)))); -inf maps to 0
    and +inf to itself (apply_act, as disassembled)."""
    import numpy as np

    f32 = np.float32
    y = np.asarray(y, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        cube = ((y * y).astype(f32) * y).astype(f32)
        inner = (f32(0.7978845608028654)
                 * fma_f32(cube, f32(0.044715), y)).astype(f32)
        out = ((y * f32(0.5)).astype(f32)
               * (f32(1.0) + fast_tanhf_f32_reference(inner)).astype(f32))
    out = np.where(y == -np.inf, f32(0.0), out)
    return np.where(y == np.inf, y, out).astype(f32)


def softcap_f32_device(y, scale, cap):
    """The fused soft cap: fl(y * scale), then / cap, fast_tanhf, * cap, each
    rounded alone in source order (apply_act)."""
    import numpy as np

    f32 = np.float32
    y = np.asarray(y, dtype=np.float32)
    y = (y * f32(scale)).astype(f32)
    y = (y / f32(cap)).astype(f32)
    return (fast_tanhf_f32_reference(y) * f32(cap)).astype(f32)


# fast_math.h's fast_exp_neg constants, spelled as its hex float literals.
FAST_EXP_NEG_SHIFTER = float.fromhex("0x1.8p23")
FAST_EXP_NEG_INV_LN2_64 = float.fromhex("0x1.715476p+6")
FAST_EXP_NEG_LN2_64_HI = float.fromhex("0x1.63p-7")
FAST_EXP_NEG_LN2_64_LO = float.fromhex("-0x1.bd0106p-19")
FAST_EXP_NEG_C3 = float.fromhex("0x1.555556p-3")
FAST_EXP_NEG_FLUSH = -87.3365447


def fast_exp_neg_table():
    """kFastExpNegTable: float32 2^(j/64) as bits, minus j << 17 (mod 2^32)."""
    import numpy as np

    bits = np.asarray([np.float32(2.0 ** (j / 64.0)) for j in range(64)],
                      dtype=np.float32).view(np.uint32).astype(np.int64)
    return ((bits - (np.arange(64) << 17)) & 0xFFFFFFFF).astype(np.uint32)


def fast_exp_neg_f32_reference(value):
    """Host transcription of ``fast_exp_neg`` (runtime/fast_math.h),
    the table exp espressif::attn_tail maps ``s - max`` with.

    Every fused step there is an explicit fmaf, so the S3 build fuses exactly
    these six (madd.s/msub.s in the disassembly, for each interleaved lane):
    the shifter rounding, both reduction steps, the two polynomial steps and
    ``t*p + t``; ``z - shifter``, ``r*r`` and the integer table scaling round
    on their own or are exact. Positive inputs are outside the device
    function's contract; this reproduces what it computes for them anyway
    (the exponent add wraps modulo 2^32). NaN in gives NaN out, though not
    necessarily the device's NaN bits.
    """
    import numpy as np

    f32 = np.float32
    x = np.asarray(value, dtype=np.float32)
    shifter = f32(FAST_EXP_NEG_SHIFTER)
    with np.errstate(invalid="ignore", over="ignore"):
        z = fma_f32(x, f32(FAST_EXP_NEG_INV_LN2_64), shifter)
        k = z.view(np.uint32)
        kf = (z - shifter).astype(np.float32)
        r = fma_f32(-kf, f32(FAST_EXP_NEG_LN2_64_HI), x)
        r = fma_f32(-kf, f32(FAST_EXP_NEG_LN2_64_LO), r)
        r2 = (r * r).astype(np.float32)
        p = fma_f32(r2, fma_f32(r, f32(FAST_EXP_NEG_C3), f32(0.5)), r)
        t = (fast_exp_neg_table()[k & np.uint32(63)]
             + (k << np.uint32(17))).astype(np.uint32).view(np.float32)
        y = fma_f32(t, p, t)
    y = np.where(x < f32(FAST_EXP_NEG_FLUSH), f32(0.0), y)
    return np.where(np.isnan(x), x, y).astype(np.float32)


def attn_tail_f32_reference(scores, mask, scale, values):
    """Host transcription of ``espressif::attn_tail`` (espressif_ops.cpp).

    ``softmax(where(mask, scores * scale, -inf), -1) @ values`` in the kernel's
    own arithmetic, for goldens of programs whose decode tail
    fuse_attn_tail_pass.py fused. scores [B, M, L]; mask holds the L elements of
    one row, shared by every head; values [B, L, N]. Returns
    (probabilities [B, M, L], out [B, M, N]).
    """
    import numpy as np

    scores = np.asarray(scores, dtype=np.float32)
    width = scores.shape[-1]
    keep = np.asarray(mask, dtype=bool).reshape(width)
    with np.errstate(invalid="ignore", over="ignore"):
        scaled = (scores * np.float32(scale)).astype(np.float32)
    return attn_tail_softmax_pv_f32_reference(
        np.where(keep, scaled, np.float32(-np.inf)), values)


def attn_tail_softmax_pv_f32_reference(masked, values):
    """attn_tail_f32_reference after the mask: ``masked`` is the kernel's first
    pass already done (scores scaled in fp32, masked lanes -inf), which is how
    the decode references hold it.

    Per row: the max in index order with the softmax shim's comparison,
    fast_exp_neg of ``s - max``, a sequential float32 sum, ONE reciprocal and
    ``e * reciprocal`` as the probabilities; then every output lane as
    ``acc = fma(p[k], v[k], acc)`` for k ascending -- the device bmm's order
    and arithmetic (shim_bmm_out's std::fma; bmm_f32_portable mirrors it).

    _attn_tail_softmax_pv_f32_stepwise is that sentence as a loop over k. This
    is the same arithmetic with each sequence vectorised across rows and lanes
    instead of re-entering numpy per element step (the story Glimmer golden
    calls it 344 times over up to 256 keys): the max and the sum keep their
    order exactly, and the fma chain checks itself and falls back to the loop
    where it cannot be exact (_pv_fma_chain_f32). test_attn_tail.py holds the
    two bit-identical.
    """
    import numpy as np

    masked = np.asarray(masked, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if masked.ndim != 3 or values.ndim != 3:
        raise ValueError("attn_tail expects [B, M, L] scores and [B, L, N] values")
    batches, rows, width = masked.shape
    if values.shape[:2] != (batches, width):
        raise ValueError(f"values {values.shape} do not match scores {masked.shape}")
    flat = masked.reshape(-1, width)
    if width == 0 or np.isnan(flat).any():
        # The max's comparison drops a NaN at the next lane: keep the loop.
        return _attn_tail_softmax_pv_f32_stepwise(masked, values)
    # std::max(value, maximum) from -inf keeps the LAST of equal maxima, which
    # only differs from any other between +0 and -0.
    maximum = flat.max(axis=1)
    zero = maximum == 0
    if zero.any():
        tied = flat[zero]
        last = width - 1 - np.argmax(tied[:, ::-1] == 0, axis=1)
        maximum[zero] = tied[np.arange(tied.shape[0]), last]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        mapped = fast_exp_neg_f32_reference(flat - maximum[:, None])
        # add.accumulate is sequential in float32, from the same +0 start.
        total = np.add.accumulate(np.concatenate(
            [np.zeros((flat.shape[0], 1), dtype=np.float32), mapped], axis=1),
            axis=1)[:, -1]
        reciprocal = (np.float32(1.0) / total).astype(np.float32)
        probabilities = (mapped * reciprocal[:, None]).astype(np.float32)
    probabilities = probabilities.reshape(batches, rows, width)
    out = None
    # Up to a few thousand output lanes a step is numpy call overhead, which
    # the chain cuts; beyond, the arithmetic dominates and its bookkeeping does
    # not pay (measured over 96 keys: 7x faster at 256 lanes, 2x at 2048, even
    # at 4096).
    if batches * rows * values.shape[2] <= 2048:
        out = _pv_fma_chain_f32(probabilities, values)
    if out is None:
        out = _pv_fma_loop_f32(probabilities, values)
    return probabilities, out


def _pv_fma_loop_f32(probabilities, values):
    """``out = fma_f32(p[:, :, k], v[:, k, :], out)`` for k ascending from 0."""
    import numpy as np

    out = np.zeros(probabilities.shape[:2] + values.shape[2:], dtype=np.float32)
    for k in range(probabilities.shape[2]):
        out = fma_f32(probabilities[:, :, k:k + 1], values[:, k:k + 1, :], out)
    return out


def _pv_fma_chain_f32(probabilities, values):
    """_pv_fma_loop_f32 with one float64 add and one rounding to float32 per
    step; None when that cannot be proved equal to the fma_f32 chain.

    p*v is exact in float64, so each step's float64 sum is the exact sum
    rounded once; rounding it again to float32 is the correct single rounding
    unless that float64 sum is exactly a float32 midpoint (the double-rounding
    tie fma_f32's round-to-odd exists for) or lies outside float32's normal
    range. Every step's sum is kept and checked for those cases afterwards.
    """
    import numpy as np

    p64 = probabilities.astype(np.float64)
    v64 = values.astype(np.float64)
    batches, rows, width = p64.shape
    # products[k, b, m, n] = p[b, m, k] * v[b, k, n]
    products = (p64.transpose(2, 0, 1)[:, :, :, None]
                * v64.transpose(1, 0, 2)[:, :, None, :])
    sums = np.empty_like(products)
    acc = np.zeros((batches, rows, v64.shape[2]), dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        for k in range(width):
            np.add(products[k], acc, out=sums[k])
            acc = sums[k].astype(np.float32).astype(np.float64)
        bits = sums.view(np.int64)
        magnitude = np.abs(sums)
        doubtful = (((bits & 0x1FFFFFFF) == 0x10000000)
                    | ((magnitude < np.finfo(np.float32).tiny) & (magnitude != 0))
                    | ((magnitude > np.finfo(np.float32).max)
                       & np.isfinite(magnitude)))
    if doubtful.any():
        return None
    return acc.astype(np.float32)


def _attn_tail_softmax_pv_f32_stepwise(masked, values):
    """attn_tail_softmax_pv_f32_reference one lane step at a time: the
    definition, its fallback, and its oracle in test_attn_tail.py."""
    import numpy as np

    masked = np.asarray(masked, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if masked.ndim != 3 or values.ndim != 3:
        raise ValueError("attn_tail expects [B, M, L] scores and [B, L, N] values")
    batches, rows, width = masked.shape
    if values.shape[:2] != (batches, width):
        raise ValueError(f"values {values.shape} do not match scores {masked.shape}")
    flat = masked.reshape(-1, width)
    maximum = np.full(flat.shape[0], -np.inf, dtype=np.float32)
    for k in range(width):
        # std::max(value, maximum): the old maximum survives only value < it.
        maximum = np.where(flat[:, k] < maximum, maximum, flat[:, k])
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        mapped = fast_exp_neg_f32_reference(flat - maximum[:, None])
        total = np.zeros(flat.shape[0], dtype=np.float32)
        for k in range(width):
            total = (total + mapped[:, k]).astype(np.float32)
        reciprocal = (np.float32(1.0) / total).astype(np.float32)
        probabilities = (mapped * reciprocal[:, None]).astype(np.float32)
    probabilities = probabilities.reshape(batches, rows, width)
    return probabilities, _pv_fma_loop_f32(probabilities, values)


def _attnqk_qk_parts():
    """AttnQK's int8 q@kT scores, exactly as the quantize_qk delegate computes
    them (the int8 Linears' device mirrors, then bmm_qk_i8), and its v."""
    import numpy as np
    import torch

    factory, shape = SPECS["attnqk"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape)
    x2d = x.reshape(-1, m.d).numpy().astype(np.float32)

    def lin(mod, inp):
        return addmm_i8_absmax(
            inp, mod.weight.detach().numpy().astype(np.float32),
            mod.bias.detach().numpy().astype(np.float32))

    batch, tokens, _ = x.shape
    q = lin(m.wq, x2d).reshape(batch, tokens, m.d)
    kx = lin(m.wk, x2d).reshape(batch, tokens, m.d)
    v = lin(m.wv, x2d).reshape(batch, tokens, m.d)
    # The runtime intentionally reuses one [N,K] key scratch buffer batch by
    # batch. Keep the host mirror equally explicit: flattening B*T here would
    # compare every query with keys from the other batch and accidentally turn
    # a batched matmul into one larger attention matrix.
    scores = np.stack([bmm_qk_i8(q[b], kx[b]) for b in range(batch)])
    return scores, v, m.d


def attnqk_qk_scores():
    """The raw scores the parity rows attnqk_qk and attnkc8_qk print first
    (AttnQKScores, AttnKC8Scores): the device's arithmetic, bit for bit."""
    import torch

    scores, _, _ = _attnqk_qk_parts()
    return torch.from_numpy(scores)


def attnqk_qk_out():
    """AttnQK as the quantize_qk delegate computes it: int8 Linears (delegated
    anyway), int8 q@kT scores, and torch fp32 for what stays portable (scale,
    softmax, p@v) -- the block_i8_out layering, extended one op. Returns
    the output tensor so the export can report quantisation accuracy against
    fp32 eager at tensor level, where cancellation cannot flatter it."""
    import numpy as np
    import torch

    scores, v, d = _attnqk_qk_parts()
    s = scores / np.float32(d ** 0.5)
    p = torch.softmax(torch.from_numpy(s), dim=-1)
    return torch.bmm(p, torch.from_numpy(v))


def attnkc8_qk_out():
    """AttnKC8 delegated: attnqk_qk's reference, by construction.

    Its keys are quantised by espressif::quantize_rows_i8 rather than inside
    the bmm, with the same per-row arithmetic, so the delegate dots the same
    codes against the same query codes and scales them identically. Any other
    number on the device is the int8 K cache disagreeing with the path it
    replaces.
    """
    return attnqk_qk_out()


class AttnTailNet(nn.Module):
    """A decode attention tail on its own, for espressif::attn_tail's device row.

    Two KV heads of four query rows over a 64-slot cache, a mask with holes, and
    30 value lanes (seven 4-column blocks plus the 2-column tail). Squared
    scores make the last row span ~100, so fast_exp_neg's flush to zero runs
    as well. The values are large enough (up to 48) that a one-ulp difference
    in almost any output lane moves the printed checksum, which is what lets
    run_parity hold this row to the host mirror rather than only to eager.
    They are integers from PCG64 on a 2^-16 grid, exact in float32, so the
    same bytes on every machine, as the device bits test_device_mirrors.py
    pins need: torch.randn of more than 15 elements draws other values on x86
    than on Apple silicon.
    """

    def __init__(self, width=64, lanes=30, kv_heads=2):
        super().__init__()
        import numpy as np

        slots = torch.arange(width)
        self.register_buffer(
            "mask", ((slots % 7 != 3) & (slots <= 50)).view(1, 1, width))
        steps = np.random.default_rng(0).integers(
            -48 << 16, 48 << 16, size=(kv_heads, width, lanes))
        self.register_buffer(
            "values", torch.from_numpy((steps / 65536).astype(np.float32)))

    def forward(self, x):
        scores = torch.where(self.mask, (x * x) * 40.0,
                             torch.full((), float("-inf")))
        return torch.bmm(torch.softmax(scores, dim=-1), self.values)


def attntail_mirror_out():
    """attntail through attn_tail_f32_reference: the output the kernel must write.

    The device computes ``x * x`` with portable mul.out (one fp32 multiply, as
    here) and everything after it inside the fused op.
    """
    factory, shape = SPECS["attntail"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape)
    _, out = attn_tail_f32_reference(
        (x * x).numpy(), m.mask.numpy(), 40.0, m.values.numpy())
    return out


SPECS.update({
    # Batch two is deliberate. The descriptor carries B separately and the
    # runtime reuses its key-quantisation scratch across batches; B=1 cannot
    # expose a missing batch offset or stale key scales.
    "attnqk": (AttnQK, (2, 8, 32)),
    "attnkc8": (AttnKC8, (2, 8, 32)),
    "ropefull": (RoPEFullNet, (1, 2, 4, 8)),
    "attntail": (AttnTailNet, (2, 4, 64)),
    "mamba": (MambaNet, (1, 8, 16)),
    "kan": (KANNet, (1, 16)),
    "looped": (LoopedNet, (1, 8, 8)),
    "acts": (ActsNet, (1, 8, 32)),
    "norms": (NormsNet, (1, 4, 8, 8)),
})


class MLPR(nn.Module):
    """Linear->relu->Linear->relu: both fates of a fusable relu in one model.

    The MID-chain relu feeds another claimable matmul, so claiming it would weld the
    two matmul partitions together and trip the chained-ops limitation -- it must stay
    on portable kernels. The FINAL activation is deliberately an ACCIDENTAL CHAIN --
    a relu followed by a defensive clamp(max=4), the sort of pair that arrives by
    composition in real models -- and both fates are exercised at once: the portable
    export collapses it to one clamp via canonicalize_clamps, and the delegated
    export composes it into a single fused [0,4] interval, descriptor "+clamp[0,4]".
    """

    def __init__(self, d=96, h=64, o=32):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, h), nn.Linear(h, o)

    def forward(self, x):
        y = self.fc2(torch.relu(self.fc1(x)))
        return torch.clamp(torch.relu(y), max=4.0)


def mlpr_i8_out():
    """MLPR as the delegate computes it: int8 matmuls, relu applied to each output --
    the mid one by the portable kernel, the final one fused. Same arithmetic either
    way, which is what makes the fused flag checkable against this reference: the
    portable relu keeps x when x >= 0 (so -0 too) and the fused [0, 4] interval is
    clamp_output's two comparisons."""
    import numpy as np
    import torch

    factory, shape = SPECS["mlpr"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(np.float32)
    h = addmm_iq_absmax(x, m.fc1.weight.detach().numpy().astype(np.float32),
                        m.fc1.bias.detach().numpy().astype(np.float32), "i8")
    h = np.maximum(h, np.float32(0))
    y = addmm_iq_absmax(h, m.fc2.weight.detach().numpy().astype(np.float32),
                        m.fc2.bias.detach().numpy().astype(np.float32), "i8")
    return clamp_f32(y, 0.0, 4.0)


SPECS["mlpr"] = (MLPR, (1, 96))


class MLPS(nn.Module):
    """Two fused activations at once: a SiLU diamond and an exact GELU.

    h feeds two claimable matmuls in parallel; each head's activation is TERMINAL
    (feeds only the portable add), so both fuse -- fc_a's silu as the diamond pattern
    (sigmoid+mul sharing the matmul output, a fused producer with two users)
    and fc_b's gelu as the single-op case. The mid-chain relu chains: trunk and
    both heads run as one three-op delegate, the heads fanning out from a scratch
    slot.
    """

    def __init__(self, d=96, h=64, o=32):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc_a, self.fc_b = nn.Linear(h, o), nn.Linear(h, o)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        a = torch.nn.functional.silu(self.fc_a(h))
        b = torch.nn.functional.gelu(self.fc_b(h))
        return a + b


def mlps_i8_out():
    """MLPS as the delegate computes it: int8 matmuls, then the fused epilogues'
    own float arithmetic -- silu as y / (1 + fast_expf(-y)), gelu as fast_gelu's
    erfc form -- and the portable add's single rounding."""
    import numpy as np
    import torch

    factory, shape = SPECS["mlps"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(np.float32)
    h = addmm_iq_absmax(x, m.fc1.weight.detach().numpy().astype(np.float32),
                        m.fc1.bias.detach().numpy().astype(np.float32), "i8")
    h = clamp_f32(h, 0.0, np.inf)
    a = addmm_iq_absmax(h, m.fc_a.weight.detach().numpy().astype(np.float32),
                        m.fc_a.bias.detach().numpy().astype(np.float32), "i8")
    b = addmm_iq_absmax(h, m.fc_b.weight.detach().numpy().astype(np.float32),
                        m.fc_b.bias.detach().numpy().astype(np.float32), "i8")
    return (silu_f32_device(a) + fast_gelu_f32_reference(b)).astype(np.float32)


SPECS["mlps"] = (MLPS, (1, 96))


class EpiGLUD(nn.Module):
    """The two shapes of the fused multiply, against exact numbers.

    A GeGLU-tanh pair -- tanh-GELU on the gate projection, times up -- whose
    product feeds the down projection directly: up multiplies, in place, what
    the gate wrote, and the product never leaves the delegate. Then an output
    gate, Glimmer's attention spelling: a portable value (here tanh of that
    result) times sigmoid(x @ Wg), feeding o. gp reads the same x as the pair
    but cannot join its partition -- its gate's other factor depends on it --
    so one input serves two partitions. At M=1 the pair runs the decode
    activation-sharing group; at M=4, the row-blocked one.
    """

    def __init__(self, d=48, f=64, o=32):
        super().__init__()
        self.g, self.u = nn.Linear(d, f), nn.Linear(d, f)
        self.dn = nn.Linear(f, d)
        self.gp = nn.Linear(d, d)
        self.o = nn.Linear(d, o)

    def forward(self, x):
        F = torch.nn.functional
        h = self.dn(F.gelu(self.g(x), approximate="tanh") * self.u(x))
        return self.o(torch.tanh(h) * torch.sigmoid(self.gp(x)))


def _linear_i8(x, mod):
    import numpy as np

    return addmm_iq_absmax(
        x, mod.weight.detach().numpy().astype(np.float32),
        mod.bias.detach().numpy().astype(np.float32), "i8")


def epiglu_i8_out(shape=(4, 48)):
    """EpiGLUD as the delegate computes it. The products are single fp32
    multiplies of the very values a portable mul.out would have received, so
    fusing them changes no bit, and the fused tanh-GELU and sigmoid are the
    device's own. The one approximation is the portable tanh between the two
    partitions: newlib's tanhf on the device, the host libm's here."""
    import numpy as np

    factory, _ = SPECS["epiglud"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(np.float32)
    y = gelu_tanh_f32_device(_linear_i8(x, m.g)) * _linear_i8(x, m.u)
    h = _linear_i8(y.astype(np.float32), m.dn)
    s = sigmoid_f32_device(_linear_i8(x, m.gp))
    return _linear_i8((np.tanh(h) * s).astype(np.float32), m.o)


SPECS["epiglud"] = (EpiGLUD, (4, 48))


class EpiActsD(nn.Module):
    """Six one-op activation epilogues in one model.

    A hardswish trunk (torch's own decomposition, its 3 and 6 lifted to tensor
    constants) chains through a scratch slot into five heads: hardsigmoid,
    leaky ReLU, ReLU^2, tanh-GELU and sigmoid. Each head is terminal and the
    concatenation is portable, so no head's output becomes a residual.
    """

    def __init__(self, d=48, h=64, o=24):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.heads = nn.ModuleList(nn.Linear(h, o) for _ in range(5))

    def forward(self, x):
        F = torch.nn.functional
        h = F.hardswish(self.fc1(x))
        a, b, c, e, f = (head(h) for head in self.heads)
        r = F.relu(c)
        return torch.cat([F.hardsigmoid(a), F.leaky_relu(b, 0.1), r * r,
                          F.gelu(e, approximate="tanh"), torch.sigmoid(f)],
                         dim=-1)


def epiacts_i8_out(shape=(4, 48)):
    """EpiActsD as the delegate computes it. hardswish, hardsigmoid, leaky ReLU
    and ReLU^2 are the portable kernels' own fp32 operations in their order,
    and tanh-GELU and sigmoid are the fused epilogues' fast_math arithmetic,
    so every column is the device's to the bit."""
    import numpy as np

    f32 = np.float32
    factory, _ = SPECS["epiactsd"]
    torch.manual_seed(0)
    m = factory().eval()
    x = ramp_input(shape).numpy().astype(f32)
    t = _linear_i8(x, m.fc1)
    c = np.minimum(np.maximum(t + f32(3.0), f32(0.0)), f32(6.0))
    h = ((t * c) / f32(6.0)).astype(f32)
    a, b, c, e, f = (_linear_i8(h, head) for head in m.heads)
    hs = (np.minimum(np.maximum(a + f32(3.0), f32(0.0)), f32(6.0))
          / f32(6.0)).astype(f32)
    lk = np.where(b >= 0, b, b * f32(0.1)).astype(f32)
    r = np.maximum(c, f32(0.0))
    return np.concatenate([hs, lk, (r * r).astype(f32),
                           gelu_tanh_f32_device(e), sigmoid_f32_device(f)],
                          axis=-1)


SPECS["epiactsd"] = (EpiActsD, (4, 48))


class GCNNet(nn.Module):
    """Two-layer graph convolution over a fixed 12-node/30-edge graph, pure torch.

    The GNN-ecosystem question, answered as a consumability claim: PyG-style message
    passing IS torch pieces -- gather source features along edges (index_select),
    scatter-add into destinations, degree-normalise, Linear update -- and a fixed
    topology makes edge_index a buffer, so there is no dynamic-shape entanglement.
    (DGL is its own C++ runtime and is out of scope on purpose.)

    What this row uniquely tests: index_add decomposes to index_put with
    ACCUMULATION over DUPLICATE indices -- every other index_put on device (the KV
    cache) writes unique positions. Duplicate-accumulate is the classic place a
    scatter kernel goes wrong, so it earns the device slot.
    """

    def __init__(self, n=12, e=30, d=16):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("src", torch.randint(0, n, (e,), generator=g))
        self.register_buffer("dst", torch.randint(0, n, (e,), generator=g))
        deg = torch.zeros(n).index_add(0, self.dst, torch.ones(e)).clamp(min=1)
        self.register_buffer("inv_deg", (1.0 / deg)[:, None])
        self.l1, self.l2 = nn.Linear(d, 24), nn.Linear(24, 8)

    def _prop(self, x):
        msgs = x.index_select(0, self.src)
        return torch.zeros_like(x).index_add(0, self.dst, msgs) * self.inv_deg

    def forward(self, x):
        h = torch.relu(self.l1(self._prop(x)))
        h = self.l2(self._prop(h))
        return h.mean(0, keepdim=True)


SPECS["gcn"] = (GCNNet, (12, 16))


class ConvD(nn.Module):
    """One conv2d, delegated. The fixture for the im2col path.

    Deliberately padded and strided: zero padding is where the gather reads
    outside the input and must contribute nothing to the dot OR to the
    activation absmax, and a stride > 1 is where the output row index stops
    being the input row index. A 1x1-kernel fixture would exercise neither and
    would pass while a real conv did not.
    """

    def __init__(self, c_in=4, c_out=8, k=3, stride=2, pad=1):
        super().__init__()
        self.c = nn.Conv2d(c_in, c_out, k, stride, pad)

    def forward(self, x):
        return self.c(x)


def convd_i8_out(shape=(1, 4, 9, 9)):
    """The device's conv arithmetic, modelled exactly.

    Held against the DELEGATE's own summation order, not eager conv, for the
    same reason every other delegated row is: the device quantises, and holding
    it to the float answer would be demanding that quantisation be lossless.
    """
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


def convd_static_out(sx, shape=(1, 4, 9, 9)):
    """ConvD under a CALIBRATED activation scale, not per-row absmax.

    The scale is an argument for the same reason the MLP's is: export_delegated
    feeds in exactly what calibrate_static_act recorded, so the reference and the
    blob share one calibration rather than two hopefully-identical replays.

    This row is not redundant with the MLP's. A conv's rows are PATCHES, and no
    patch's absmax equals the whole input's except by accident -- so the static
    scale genuinely differs from the dynamic one on every row, which is what
    makes a runtime that quietly ignores the flag visible here.
    """
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8",
                       sx_fixed=sx)
    return _t.as_tensor(y).reshape(1, geom.c_out, geom.h_out,
                                            geom.w_out)


def convd_iq_out(fmt, shape=(1, 4, 9, 9)):
    """The same conv at any rung of the weight-format ladder.

    Conv reuses the matmul kernels by construction -- im2col makes the weight a
    plain [N, K] -- so i4 and i16 SHOULD follow for free. "Should follow for free"
    is a claim to check, not assume: the AOT and the runtime can agree on the fast
    path and disagree on a case nothing covers.

    Each rung is held against its OWN reference rather than against i8. i4 is a
    coarser grid and its answer is genuinely different; demanding it match i8
    would be demanding that 4-bit quantisation be lossless.
    """
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt=fmt)
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


# Registered after the class, not in the SPECS.update above: ConvD is defined
# after that call, so the entry has to follow it.
class Conv1dD(nn.Module):
    """One conv1d, delegated. Strided and padded for the same reasons ConvD is.

    Its value is not that 1-D is a new code path -- it is that 1-D is NOT one.
    `admit` synthesises h=1/kh=1, so this row runs through the identical device
    gather as ConvD, and a failure here against a passing ConvD would mean the
    degenerate axis is not as degenerate as claimed.
    """

    def __init__(self, c_in=4, c_out=8, k=5, stride=2, pad=2):
        super().__init__()
        self.c = nn.Conv1d(c_in, c_out, k, stride, pad)

    def forward(self, x):
        return self.c(x)


def conv1d_i8_out(shape=(1, 4, 20)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = Conv1dD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.w_out)


class Conv3dD(nn.Module):
    """A V-JEPA-style tubelet embed: Conv3d with kernel == stride, no padding.

    The simplest conv3d there is -- a pure non-overlapping tiling -- which is
    also exactly what a video encoder's patch embed does, so it is the shape
    worth proving first rather than a contrived one.
    """

    def __init__(self, c_in=3, c_out=8, tubelet=2, patch=4):
        super().__init__()
        self.c = nn.Conv3d(c_in, c_out, (tubelet, patch, patch),
                           stride=(tubelet, patch, patch))

    def forward(self, x):
        return self.c(x)


def conv3d_i8_out(shape=(1, 3, 4, 8, 8)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = Conv3dD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.d_out, geom.h_out, geom.w_out)


class DwConvD(nn.Module):
    """A depthwise 3x3, the MobileNet/YOLO separable-block workhorse."""

    def __init__(self, c=8, k=3, stride=1, pad=1):
        super().__init__()
        self.c = nn.Conv2d(c, c, k, stride, pad, groups=c)

    def forward(self, x):
        return self.c(x)


class DwConvGroupedScaleD(nn.Module):
    """A depthwise row wider than one 128-element weight-scale group."""

    def __init__(self, c=2, k=13):
        super().__init__()
        self.c = nn.Conv2d(c, c, k, padding=k // 2, groups=c)

    def forward(self, x):
        return self.c(x)


class DwConvResD(nn.Module):
    """Depthwise convolution with its single-input residual epilogue."""

    def __init__(self, c=8, k=3):
        super().__init__()
        self.c = nn.Conv2d(c, c, k, 1, 1, groups=c)

    def forward(self, x):
        return self.c(x) + x


def dwconv_iq_out(fmt="i8", shape=(1, 8, 7, 7)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = DwConvD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = depthwise_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt=fmt)
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


def dwconv_i8_out(shape=(1, 8, 7, 7)):
    return dwconv_iq_out("i8", shape)


def dwconvres_i8_out(shape=(1, 8, 7, 7)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = DwConvResD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = depthwise_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    y = _t.as_tensor(y).reshape(1, geom.c_out, geom.h_out, geom.w_out)
    return y + x


SPECS["convd"] = (ConvD, (1, 4, 9, 9))
class DwConv1dD(nn.Module):
    """Depthwise conv1d: the audio/timeseries workhorse.

    Its value is the COMPOSITION. Depthwise is device-verified in 2-D and the
    degenerate height axis is device-verified dense; this row is the only place
    both hold at once, which is where a bug would sit.
    """

    def __init__(self, c=8, k=5, stride=2, pad=2):
        super().__init__()
        self.c = nn.Conv1d(c, c, k, stride, pad, groups=c)

    def forward(self, x):
        return self.c(x)


def dwconv1d_i8_out(shape=(1, 8, 24)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import depthwise_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = DwConv1dD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = depthwise_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.w_out)


SPECS["dwconvd"] = (DwConvD, (1, 8, 7, 7))
SPECS["dwconvresd"] = (DwConvResD, (1, 8, 7, 7))
class GroupConvD(nn.Module):
    """A grouped conv (ResNeXt / ShuffleNet cardinality), g=2.

    The case between the two regimes: weights are g-times smaller than a dense
    conv's but nowhere near depthwise's, so im2col is still the right transform
    and this is g dense convs rather than a new kernel.
    """

    def __init__(self, c_in=8, c_out=16, k=3, groups=2):
        super().__init__()
        self.c = nn.Conv2d(c_in, c_out, k, 1, 1, groups=groups)

    def forward(self, x):
        return self.c(x)


def groupconv_i8_out(shape=(1, 8, 7, 7)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import grouped_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = GroupConvD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = grouped_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


class GroupConvEdgeD(GroupConvD):
    """Grouped shape whose split records look depthwise at runtime.

    Before splitting, C_in=2, C_out=4, groups=2 is not depthwise. Each split
    record has c_in=1 and c_out=2, however, so the predicate
    c_in==1 && c_out==groups would select the depthwise loop. c_out_total is the
    discriminator: 4 for this grouped op, 2 for a true depthwise op.
    """

    def __init__(self):
        super().__init__(c_in=2, c_out=4, groups=2)


def groupconvedge_i8_out(shape=(1, 2, 7, 7)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import grouped_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = GroupConvEdgeD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = grouped_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


class GroupConv8D(nn.Module):
    """Grouped conv at a REAL cardinality: g=8, strided. ResNeXt's bottleneck.

    g=2 is not a sufficient test and the reason is specific. With two groups,
    several wrong implementations still look nearly right -- swapped halves,
    interleaved writes, an offset applied to one side only -- because there is
    only one boundary to get wrong and the two slices are the same size. A
    runtime that read `groups > 1` as depthwise, running the wrong kernel, would
    pass a g=2 fixture.

    So this one is deliberately awkward. c_in/groups = 2, which is NOT 1, so a
    predicate that confuses grouped with depthwise fails here immediately;
    c_out/groups = 4 differs from c_in/groups, so the two per-group counts
    cannot be swapped unnoticed; and stride 2 means the gather's index maths
    runs on a different lattice than the input's.
    """

    def __init__(self, c_in=16, c_out=32, k=3, groups=8, stride=2):
        super().__init__()
        self.c = nn.Conv2d(c_in, c_out, k, stride, 1, groups=groups)

    def forward(self, x):
        return self.c(x)


def groupconv8_i8_out(shape=(1, 16, 9, 9)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import grouped_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = GroupConv8D().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = grouped_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


def groupconv8_iq_out(fmt, shape=(1, 16, 9, 9)):
    """Grouped conv at a chosen rung. Exists to exercise the i16 path's channel
    offsets: that path gathers and scatters per row rather than staging a block,
    so its group arithmetic is a SECOND implementation of the same idea, and two
    implementations of one rule can disagree."""
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import grouped_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = GroupConv8D().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = grouped_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt=fmt)
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


class ConvResD(nn.Module):
    """conv2d + a residual, fused.

    The AOT fuses a residual onto a conv. A runtime that dropped it -- a null
    residual for every convolution -- would answer with the bare convolution,
    silently: only a fixture of the combination says so.

    `conv(x) + x` with c_in == c_out, which is the canonical residual block
    rather than a contrived pair of tensors -- and it keeps the fixture
    single-input, which is what SPECS and the device's input filler expect.
    """

    def __init__(self, c=8, k=3):
        super().__init__()
        self.c = nn.Conv2d(c, c, k, 1, 1)

    def forward(self, x):
        return self.c(x) + x


def convres_iq_out(fmt="i8", shape=(1, 8, 9, 9)):
    """Conv at i8, then the residual added in NCHW order.

    Residual LAST, matching every other fused path -- bias, activation, clamp,
    then residual. The device applies it during the NCHW scatter, which is the
    same position in that order.
    """
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvResD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt=fmt)
    y = _t.as_tensor(y).reshape(1, geom.c_out, geom.h_out, geom.w_out)
    return y + x


def convres_i8_out(shape=(1, 8, 9, 9)):
    return convres_iq_out("i8", shape)


class GroupConvReluD(nn.Module):
    """Grouped conv with a fused activation: the epilogue on EVERY group.

    A pointwise activation on a grouped conv is set on all g ops, because each
    owns a disjoint slice of the output channels. The failure this fixture
    exists to catch is the flag landing on one of them -- the output would be
    correct for those channels and raw for the rest, which is a plausible
    checksum rather than an obviously broken one.
    """

    def __init__(self, c_in=16, c_out=32, k=3, groups=8, stride=2):
        super().__init__()
        self.c = nn.Conv2d(c_in, c_out, k, stride, 1, groups=groups)

    def forward(self, x):
        return torch.relu(self.c(x))


def groupconvrelu_i8_out(shape=(1, 16, 9, 9)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import grouped_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = GroupConvReluD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = grouped_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    y = _t.as_tensor(y).reshape(1, geom.c_out, geom.h_out, geom.w_out)
    return clamp_f32(y.numpy(), 0.0, float("inf"))


SPECS["grpconvrelud"] = (GroupConvReluD, (1, 16, 9, 9))
class ConvResViewD(nn.Module):
    """conv + residual with a reshape on BOTH sides -- the flattened spelling.

    `_tail_root_mm` walks through a view over a conv as over mm/addmm. Were it
    to stop at the view, a conv under a reshape would find no root and the entire
    residual would stay portable -- a graceful refusal, and a refusal nobody
    measures is indistinguishable from a design decision.

    The residual operand's own reshape stays portable and should -- the delegate
    reads that tensor as an input, so it has to exist.
    """

    def __init__(self, c=8, k=3):
        super().__init__()
        self.c = nn.Conv2d(c, c, k, 1, 1)

    def forward(self, x):
        return self.c(x).reshape(1, 8, -1) + x.reshape(1, 8, -1)


def convresview_i8_out(shape=(1, 8, 9, 9)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvResViewD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    y = _t.as_tensor(y).reshape(1, geom.c_out, geom.h_out, geom.w_out)
    return (y + x).reshape(1, 8, -1)


class LayerNorm2dD(nn.Module):
    """SAM/ConvNeXt LayerNorm2d: normalise over C on [B,C,H,W].

    Portable, not delegated -- the point of the row is the fused
    `espressif::layer_norm2d` KERNEL, which replaces fourteen ops whose reduction
    runs down a channel column with H*W stride. C=16 and H=W=9 keeps the stride
    (81) larger than the reduction length, which is the awkward direction and the
    real one.

    The reference is plain eager, because the fused op is bit-identical to the
    graph it replaces -- verified on the host in test_layernorm2d.py. If the
    device disagrees, the kernel is wrong, not the tolerance.
    """

    def __init__(self, c=16, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LayerNorm2dX(nn.Module):
    """LayerNorm2d where its arithmetic shows, held to the host runner's bits
    (run_parity.py records them): C = 3, not a power of two, so each mean's
    division by C rounds; weight and bias neither one nor zero; inputs off
    the ramp's dyadic grid. The fused espressif::layer_norm2d must be the
    fourteen-op graph bit for bit (fuse_layer_norm2d_pass.py), and the host
    runner's kernel, built without contraction, is (test_layernorm2d.py). On
    the chip, espressif_ops.cpp's default contraction fused sq += d * d and
    w * q + b into madd.s, moving 80 of these 243 outputs by an ulp (a float32
    emulation), which layernorm2dd -- C = 16,
    weight 1, bias 0, a dyadic ramp, every product exact -- could not show.

    The input is main.cpp's ramp times a factor per channel, plus an offset,
    each a portable kernel that rounds on the host as on the chip; channels 0
    and 2 are nearly equal, so their x - u is small beside eps (where an ulp
    of the mean shows).
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.register_buffer(
            "mix", torch.tensor([0.7071, 1.2345, 0.7072]).view(1, 3, 1, 1))
        self.register_buffer(
            "offset", torch.tensor([0.1, -0.3, 0.1001]).view(1, 3, 1, 1))
        self.weight = nn.Parameter(torch.tensor([0.75, 1.1, 1.25]))
        self.bias = nn.Parameter(torch.tensor([-0.3, 0.05, 0.3]))
        self.eps = eps

    def forward(self, x):
        x = x * self.mix + self.offset
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


SPECS["layernorm2dd"] = (LayerNorm2dD, (1, 16, 9, 9))
SPECS["layernorm2dx"] = (LayerNorm2dX, (1, 3, 9, 9))
SPECS["convresviewd"] = (ConvResViewD, (1, 8, 9, 9))
SPECS["convresd"] = (ConvResD, (1, 8, 9, 9))
SPECS["grpconv8d"] = (GroupConv8D, (1, 16, 9, 9))
SPECS["dwconv1dd"] = (DwConv1dD, (1, 8, 24))
class ConvReluD(nn.Module):
    """conv2d with a fused relu. The epilogue on a NON-matmul producer.

    Its point is the fusion, not the conv: the delegate applies activations
    between accumulate and store, and the epilogue gates accept a conv producer,
    so the activation after a convolution fuses rather than staying portable.
    """

    def __init__(self, c_in=4, c_out=8):
        super().__init__()
        self.c = nn.Conv2d(c_in, c_out, 3, 1, 1)

    def forward(self, x):
        return torch.relu(self.c(x))


def convrelu_i8_out(shape=(1, 4, 9, 9)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvReluD().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    y = conv_iq_absmax(x, model.c.weight, model.c.bias, geom, fmt="i8")
    # The relu is applied AFTER the dequantised accumulation, which is where the
    # delegate applies it -- before the store, after the scale.
    return clamp_f32(y.reshape(1, geom.c_out, geom.h_out, geom.w_out).numpy(),
                     0.0, float("inf"))


SPECS["grpconvd"] = (GroupConvD, (1, 8, 7, 7))
SPECS["grpconvedged"] = (GroupConvEdgeD, (1, 2, 7, 7))
class ConvSiluD(nn.Module):
    """conv -> SiLU -> conv -> SiLU, the YOLO ConvBNSiLU shape without the BN.

    The fixture for the THIRD scratch slot. A SiLU is a diamond -- sigmoid and
    mul both read the conv's output -- so chaining two of them keeps more
    intermediates live than a linear chain does: more than two slots' worth, as
    in a YOLO export.
    """

    def __init__(self, c=8):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, 1, 1)
        self.b = nn.Conv2d(c, c, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.b(self.act(self.a(x))))


def convsilu_i8_out(shape=(1, 8, 7, 7)):
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = ConvSiluD().eval()
    x = ramp_input(shape)
    ep = _t.export.export(model, (x,))
    gm = to_edge(ep).exported_program().graph_module
    convs = [n for n in gm.graph.nodes
             if n.op == "call_function" and "convolution" in str(n.target)]
    # Two convs in series, each with a fused SiLU applied after the dequantised
    # accumulation -- the order the delegate applies it.
    h = x
    for node, layer in zip(convs, (model.a, model.b)):
        geom = admit(node)
        y = conv_iq_absmax(h, layer.weight, layer.bias, geom, fmt="i8")
        y = y.reshape(1, geom.c_out, geom.h_out, geom.w_out)
        h = _t.from_numpy(silu_f32_device(y.numpy()))
    return h


SPECS["convrelud"] = (ConvReluD, (1, 4, 9, 9))
class ConvTD(nn.Module):
    """The U-Net up-convolution: ConvTranspose2d 2x2, stride 2."""

    def __init__(self, c_in=4, c_out=6):
        super().__init__()
        self.c = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)

    def forward(self, x):
        return self.c(x)


class ConvTOpD(nn.Module):
    """The decoder up-convolution: ConvTranspose2d 3x3, stride 2, padding 1,
    output_padding 1, which doubles the input's extent exactly (FCN and
    autoencoder decoders, U-Net variants that up-sample by a 3x3). The output
    padding is the last row and column: past the computed extent, read
    through the same mapping, and with padding 1 they hold real taps, not
    just the bias -- so a runtime that refused the padded extent, or padded
    it with zeros, would show."""

    def __init__(self, c_in=4, c_out=6):
        super().__init__()
        self.c = nn.ConvTranspose2d(c_in, c_out, 3, 2, 1, output_padding=1)

    def forward(self, x):
        return self.c(x)


class ConvTDilD(nn.Module):
    """A dilated transposed convolution with its geometry different on each
    axis: stride (2, 1), padding (1, 2), dilation (3, 2), output_padding
    (2, 1). An axis reading the other's stride, padding or dilation would
    not even produce the right extent on a 5x6 input, and the height's
    output padding of 2 at stride 2 is legal only because of its dilation of
    3: the runtime's window is max(stride, dilation), not stride."""

    def __init__(self, c_in=4, c_out=6):
        super().__init__()
        self.c = nn.ConvTranspose2d(c_in, c_out, 3, stride=(2, 1),
                                    padding=(1, 2), dilation=(3, 2),
                                    output_padding=(2, 1))

    def forward(self, x):
        return self.c(x)


def _transposed_iq_out(cls, fmt, shape):
    """A transposed-conv fixture's device mirror at one rung."""
    import torch as _t

    from executorch.backends.esp32s3.operator_support.conv import admit
    from executorch.backends.esp32s3.test.reference import conv_iq_absmax
    from executorch.exir import to_edge

    _t.manual_seed(0)
    model = cls().eval()
    x = ramp_input(shape)
    gm = to_edge(_t.export.export(model, (x,))).exported_program().graph_module
    node = next(n for n in gm.graph.nodes
                if n.op == "call_function" and "convolution" in str(n.target))
    geom = admit(node)
    # The same flip+transpose the AOT layer applies, so the reference quantises
    # exactly the weight the device packs.
    w = model.c.weight.detach().flip(-1, -2).transpose(0, 1).contiguous()
    y = conv_iq_absmax(x, w, model.c.bias, geom, fmt=fmt)
    return y.reshape(1, geom.c_out, geom.h_out, geom.w_out)


def convt_iq_out(fmt="i8", shape=(1, 4, 5, 5)):
    return _transposed_iq_out(ConvTD, fmt, shape)


def convt_i8_out(shape=(1, 4, 5, 5)):
    return convt_iq_out("i8", shape)


def convtop_iq_out(fmt="i8", shape=(1, 4, 5, 5)):
    return _transposed_iq_out(ConvTOpD, fmt, shape)


def convtdil_iq_out(fmt="i8", shape=(1, 4, 5, 6)):
    return _transposed_iq_out(ConvTDilD, fmt, shape)


SPECS["convsilud"] = (ConvSiluD, (1, 8, 7, 7))
SPECS["convtd"] = (ConvTD, (1, 4, 5, 5))
SPECS["convtopd"] = (ConvTOpD, (1, 4, 5, 5))
SPECS["convtdild"] = (ConvTDilD, (1, 4, 5, 6))
SPECS["conv3dd"] = (Conv3dD, (1, 3, 4, 8, 8))
SPECS["conv1dd"] = (Conv1dD, (1, 4, 20))
