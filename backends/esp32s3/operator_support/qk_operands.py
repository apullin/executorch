# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which q@kT operands the delegate may read: admission by allow-list.

Every q@kT the partitioner claims reads its operands from partition inputs
and quantises its query side itself, per row at absmax/127, half away from
zero:

  OP_BMM_DYN_I8            quantize_qk=True: query AND key rows so
                           (reference.bmm_qk_i8);
  OP_BMM_KCACHE_I8         the story's int8 K cache (kcache_i8.py): the key
                           codes and scales as stored
                           (kcache_i8.cached_i8_qk);
  OP_BMM_KCACHE_AFFINE_I8  ExecuTorch's stock affine K cache (stock_llm.py):
                           the codes, scales and zero points as stored
                           (mirrors.kcache_scores).

Each is a declared numerical mode with a mirror, and it is the arithmetic the
program owes only for an operand the model computes in plain float. An
operand a recipe quantised already sits on the recipe's grid -- a static
pair, a dequantise of constant codes, a dynamic chain whole or computed in
place by strip_dynamic_act_qdq -- and so does a value the model rounds
itself in float ops; the delegate would quantise either again its own way:
neither the converted model's arithmetic nor the mode's. That is the shape of
every case a list of known hazards misses -- constant codes behind a
dequantise, a static pair, a chain computed in place, a grid behind
arithmetic -- each one more spelling the list did not name.

So admission is a positive proof, built on lists rather than on a list of
hazards. `origin()` searches back from the operand through everything its
values come from, and names what it finds:

  walked   ops that move, select or keep values: data movement (_MOVES: a
           view, permute, expand, slice or select, a gather of rows or
           elements, a copy, a split or unbind piece), float-to-float casts,
           the selected values of topk, sort, kthvalue and max/min
           (_SELECTIONS), neg, abs, relu, a clamp, the values a cat, stack,
           where, index_put, scatter or scatter_add assembles (_ASSEMBLERS),
           and a buffer the forward writes, read directly or through
           moves, through what it writes into it (no partition takes a
           written buffer in, so the delegate reads it at run time);
  carried  elementwise arithmetic with a scale or an offset, followed into
           the operand that carries the elements, so codes times a scale
           plus a zero point is still codes. A scale is
             by shape     a number, or a tensor that holds one value at
                          indices where the other operand may differ: for
                          each dimension the shape test reads a partition of
                          its indices into classes proven to hold one value
                          (_classes) -- one class along a broadcast
                          dimension, classes of four along the width of a
                          group-wise step repeated, expanded and reshaped,
                          stacked or concatenated, or gathered by group to
                          it -- read through expansions, repeats, reshapes,
                          permutes, selects, slices, flips, gathers (by
                          indices the program fixes; along the other
                          dimensions, as the indices themselves are),
                          concatenations (copies of one value one class) and
                          elementwise ops, and from a constant's own values
                          (a buffer filled with one value, or one value per
                          group, is a scale whatever its shape);
             by identity  the step of the grid the other operand is on,
                          whatever its shape: round(x / s) * s,
                          round(x * (1 / s)) * s, (x // s) * s, through the
                          moves, keeps and offsets between the rounding and
                          the multiply, s one value with the divisor
                          (_same_value) up to aliases, a factor of one
                          value, clamps and guards against zero (a
                          group-wise scale written twice, guarded where it
                          divides by a clamp or by + eps);
           and a sum is read as its terms, through moves that keep values,
           neg and multiples by numbers: equal terms merge, and when terms
           cancel the sum is what remains (a straight-through estimator,
           x + (fq(x) - x).detach(), is fq(x), however often x is written).
           Equal is exact (_same_value, the equivalence value numbering
           computes, with no depth bound): one node past aliases and
           reshapes back to its own shape, constants holding the same bits,
           one deterministic op over equal arguments -- never a random op or
           an uninitialised factory, and never a flip, permute, slice or
           gather of what it moves. A remainder is read as x minus a grid
           (remainder(x, s) is x - floor(x / s) * s, frac(x) is
           x - trunc(x)), so x minus its remainder is that grid;
  algebra  arithmetic between values that all vary, with no scale among
           them, computes new values unless every one of them is on a grid
           (its own search refuses it): a difference of two comparison
           casts or of two wheres of numbers is a grid too;
  plain    the ops that compute new values (_COMPUTES: matmuls and
           convolutions, norms and softmax, transcendental functions,
           reductions), arithmetic between values that all vary and are not
           all on grids (a plain value times or plus a grid it does not
           scale, a mask, a QAT layer's output), the backend's own float ops
           (_PLAIN_ESPRESSIF), graph inputs, and the program's constants and
           factories;
  refused  a static pair or Stage B codes, constant codes, a dynamic chain
           whole or computed in place, any other quantise or dequantise (a
           recipe's, an ATen fake quantisation, the backend's quantised
           ops), rounding (round, floor, ceil, trunc, a division that
           rounds, a grid found by the reading of sums above, and
           (x + c) - c, which rounds x to the ulp of the number or constant
           c: the magic-number round), sign spelled by hand (x over its
           magnitude, abs(x), sqrt(x * x) or (x * x) ** 0.5, the magnitude
           over x, and their reciprocal, rsqrt and ** -0.5 spellings,
           guarded against zero by max(., eps), maximum(., eps) or + eps for
           an eps up to _GUARD, read through the factors that compute it;
           the two x one value element for element, placed by the same
           positional moves: flip(x) / flip(abs(x)) is sign, x /
           abs(x.flip(-1)) is not), a
           constant table looked up at codes the model computes from a
           float value -- any index that depends on a float through an op
           with a floating input and an integer or boolean result (a cast, a
           comparison, argmin or argmax, bucketize, an integer fill of a
           float), whatever integer arithmetic follows: a lookup-table
           quantiser, nearest-level and vector quantisation, the table a
           constant or a constant scaled and offset at run time by values
           one along the lookup, NF4's levels times an absmax
           (_index_codes) -- a few fixed levels (a
           comparison cast to float; an assembler of numbers, factories and
           constants of any shape alone, through neg and abs -- a learned
           scalar or per-channel alpha, where(x > 0, alpha, -alpha), two
           learned tables chosen by a mask -- and a where, a masked fill, a
           maximum or a minimum of those and of runtime values that hold one
           value where the result differs, a row's own scale), integer
           values cast to float (codes, wherever they are), and any op on
           none of these lists: unknown, refused by name until it is proven
           and listed.

What the search does not know never admits. A constant whose values the
gate cannot read (a FakeTensor with no real tensor behind it), or a move no
rule reads (repeats given as a tensor, an index by several tensors), is not
read: its partitions are unknown, which counts as possibly one class in a
partner (the candidate is followed) and as varying in the candidate itself,
and an equality that needs its values is not decided. Every walk is bounded,
and one cut short refuses or searches further:

  _BUDGET          a walk's moves, a search's sources, a sum's nodes, the
                   clamps, guards and factors an identity steps through:
                   refused (UNRECOGNISED, naming the limit);
  _NESTING         levels of the algebra of grids: refused;
  _INDEX_BUDGET    the nodes of a gather's index walk: refused (an index not
                   read to its sources may be codes);
  _SAME_BUDGET     the pairs structural equality compares: refused (an
                   equality not decided is neither "same" nor "not equal");
  _VARIES_BUDGET   the nodes whose partitions are read (each once, without
                   recursion: a chain of any depth is read to its end): past
                   it, not read (as above: its partners are followed);
  _FACTOR_BUDGET   the nodes of the search for a rounding's divisor, and
  _FACTOR_DEPTH    the moves from a rounding to its divisor: past either,
                   the other operand is taken for the step and the value is
                   followed.

Memoised results are never wider than the walk that made them: a node whose
partition was not read stays not read, an index walk that ran out refuses.
_GUARD is not a limit but a threshold of the sign rule: an epsilon past it
makes computation, not a guard (x / max(abs(x), 1) is clamp(x, -1, 1), and
so is x / max(abs(x), eps0 * 1e8) for a buffer eps0 of 1e-8).

The rules are not the only judge. Behind them, on every q@kT they would
delegate, the value check (qk_values.py) runs the program on sample inputs
and keeps the bmm portable when an operand the delegate quantises itself is
on a grid in its values, whatever arithmetic made it: a listed form written
in a way these rules do not recognise, or a grid made by other arithmetic --
a saturating clamp(c * x, -1, 1) with a large c, (x + c1) - c2 with
c1 != c2, one_hot(codes) @ table (a matmul computes new values: no rule
tells a one-hot selection from a quantised projection's integer combination
without proving one nonzero per row; its values do). Values never admit:
what these rules refuse stays refused. The contract (the README states it,
measured): the converted model's bits for an operand that is a grid by
these rules or by its values on the sample runs; any other operand is a
value the model computes, which quantize_qk requantises like any
activation, within the mode's int8 error.

An operand is admitted when no path finds a refusing origin and it is a
runtime value: whether it is a constant is decided on the operand itself
(is_constant), so a constant table gathered by runtime indices that are not
codes -- token ids, positions, integer arithmetic on them -- is plain.
Two answers admit:

  plain    for the operands the delegate quantises itself;
  kcache   the key side of a K-cache op only, through kcache_part_problem:
           the cache's int8 codes and float32 scales (and int8 zero points),
           runtime tensors the delegate reads in place and dots exactly as
           stored -- codes it reproduces bit for bit, whatever wrote them.

Everything else refuses by name, and the op runs portable on the model's own
values.
"""

import math
import operator
import struct
from typing import NamedTuple, Optional

import numpy as np
import torch

PLAIN = "plain float activation"
STATIC_PAIR = "static pair"
CONSTANT_CODES = "constant codes"
CHAIN = "dynamic chain"
CHAIN_IN_PLACE = "dynamic chain computed in place"
GRID = "value rounded onto a grid"
QUANTISED = "quantise or dequantise"
CONSTANT = "constant"
COMPUTED_CONSTANT = "computed constant"
STATE = "state buffer"
CODES_CAST = "integer values cast to float"
UNRECOGNISED = "unrecognised origin"

# Data movement: ops whose output elements are their first argument's,
# moved, copied, gathered or broadcast, keeping the dtype. walk() steps
# through these (naming an operand's path, finding a cache read through
# them); the provenance search steps through them too, with the rest of
# _KEEPS.
_MOVES = frozenset({
    "aten::view_copy", "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::permute_copy", "aten::permute", "aten::transpose_copy",
    "aten::transpose", "aten::t_copy", "aten::t",
    "aten::expand_copy", "aten::expand",
    "aten::alias_copy", "aten::alias", "aten::clone", "aten::detach_copy",
    "aten::detach", "aten::lift_fresh_copy", "aten::contiguous",
    "aten::unsqueeze_copy", "aten::unsqueeze",
    "aten::squeeze_copy", "aten::squeeze",
    "aten::slice_copy", "aten::slice", "aten::select_copy", "aten::select",
    "aten::narrow_copy", "aten::narrow", "aten::unfold_copy",
    "aten::as_strided_copy", "aten::as_strided", "aten::diagonal_copy",
    "aten::diagonal", "aten::repeat", "aten::flip", "aten::roll",
    "aten::index_select", "aten::gather", "aten::index",
    "aten::pixel_shuffle", "aten::pixel_unshuffle", "aten::channel_shuffle",
    "dim_order_ops::_clone_dim_order",
})
# Ops whose output values are some of their first argument's values,
# possibly negated, or a constant: they keep a grid a grid. The provenance
# search steps through them to the argument (with clamp's tensor bounds, if
# any, as further sources).
_KEEPS = frozenset({
    "aten::neg", "aten::abs", "aten::relu", "aten::hardtanh",
    "aten::clamp", "aten::clip", "aten::clamp_min", "aten::clamp_max",
    "aten::tril", "aten::triu", "aten::repeat_interleave",
    "aten::embedding", "aten::_unsafe_index", "aten::take",
    "aten::amax", "aten::amin",
    "aten::reflection_pad1d", "aten::reflection_pad2d",
    "aten::reflection_pad3d", "aten::replication_pad1d",
    "aten::replication_pad2d", "aten::replication_pad3d",
    "aten::upsample_nearest1d", "aten::upsample_nearest2d",
    "aten::upsample_nearest3d", "aten::_upsample_nearest_exact1d",
    "aten::_upsample_nearest_exact2d", "aten::_upsample_nearest_exact3d",
    "aten::max_pool1d", "aten::max_pool2d", "aten::max_pool3d",
})
# Copies that may change the dtype: a move between floating types (fp16 or
# bf16 upcast to fp32, the values unchanged), integer codes otherwise.
_CASTS = frozenset({"aten::_to_copy", "dim_order_ops::_to_dim_order_copy"})
# Multi-output data movement, read through getitem.
_SPLITS = frozenset({
    "aten::split_copy", "aten::split", "aten::split_with_sizes_copy",
    "aten::split_with_sizes", "aten::unbind_copy", "aten::unbind",
    "aten::chunk", "aten::tensor_split", "aten::unsafe_split",
    "aten::unsafe_split_with_sizes",
})
# Multi-output ops whose first output selects elements of the argument
# (values; the second output is their indices).
_SELECTIONS = frozenset({
    "aten::topk", "aten::sort", "aten::kthvalue", "aten::max", "aten::min",
    "aten::median", "aten::nanmedian", "aten::mode", "aten::cummax",
    "aten::cummin", "aten::max_pool2d_with_indices",
    "aten::max_pool3d_with_indices", "aten::adaptive_max_pool2d",
})


def _cat_sources(node):
    return list(node.args[0]) if node.args else []


# Ops that assemble their output from the values of several tensors,
# unchanged: the arguments whose elements land in the output. An index, a
# mask or a scalar fill is not a source. A cache's write is one of these: its
# base is the cache, whose content the forward wrote.
_ASSEMBLERS = {
    "aten::cat": _cat_sources,
    "aten::stack": _cat_sources,
    "aten::where": lambda n: list(n.args[1:3]),
    "aten::index_put": lambda n: [n.args[0], n.args[2]],
    "aten::index_copy": lambda n: [n.args[0], n.args[3]],
    "aten::index_add": lambda n: [n.args[0], n.args[3]],
    "aten::index_fill": lambda n: [n.args[0]] + list(n.args[3:4]),
    "aten::slice_scatter": lambda n: list(n.args[:2]),
    "aten::select_scatter": lambda n: list(n.args[:2]),
    "aten::diagonal_scatter": lambda n: list(n.args[:2]),
    "aten::as_strided_scatter": lambda n: list(n.args[:2]),
    "aten::masked_scatter": lambda n: [n.args[0], n.args[2]],
    "aten::masked_fill": lambda n: [n.args[0], n.args[2]],
    "aten::constant_pad_nd": lambda n: [n.args[0]],
    "aten::copy": lambda n: [n.args[1]],
    "aten::scatter": lambda n: ([n.args[0], n.args[3]]
                                if len(n.args) > 3 else [n.args[0]]),
    "aten::scatter_add": lambda n: [n.args[0], n.args[3]],
    "aten::scatter_reduce": lambda n: [n.args[0], n.args[3]],
    "aten::maximum": lambda n: list(n.args[:2]),
    "aten::minimum": lambda n: list(n.args[:2]),
    "aten::fmax": lambda n: list(n.args[:2]),
    "aten::fmin": lambda n: list(n.args[:2]),
}
# Gathers of a table's values by indices: (the table, [the indices]). A
# constant table gathered at codes the model computes from a float value is
# a lookup-table quantiser (_index_codes).
_GATHERS = {
    "aten::embedding": lambda n: (n.args[0], list(n.args[1:2])),
    "aten::index": lambda n: (n.args[0], [i for i in (
        n.args[1] if len(n.args) > 1 else ()) if i is not None]),
    "aten::_unsafe_index": lambda n: (n.args[0], [i for i in (
        n.args[1] if len(n.args) > 1 else ()) if i is not None]),
    "aten::index_select": lambda n: (n.args[0], list(n.args[2:3])),
    "aten::gather": lambda n: (n.args[0], list(n.args[2:3])),
    "aten::take": lambda n: (n.args[0], list(n.args[1:2])),
}


def _gathered_dims(n, name, table):
    """The dimensions of `table` gather `n` (a _GATHERS op) reads at its
    indices: an embedding's rows, the dimensions an index's tensors stand
    at, index_select's and gather's dimension, every one for take; None
    when the table's rank is not known."""
    shape = _aligned_shape(table, 0)
    if shape is None:
        return None
    r = len(shape)
    try:
        if name == "aten::embedding":
            return (0,)
        if name in ("aten::index", "aten::_unsafe_index"):
            return tuple(k for k, i in enumerate(n.args[1]) if i is not None)
        if name in ("aten::index_select", "aten::gather"):
            return (n.args[1] % r,)
    except (IndexError, TypeError, ZeroDivisionError):
        return None
    return tuple(range(r))


# The assemblers whose first argument is the tensor written into: what a
# cache update's base is.
_WRITES = frozenset({
    "aten::index_put", "aten::index_copy", "aten::index_add",
    "aten::index_fill", "aten::slice_scatter", "aten::select_scatter",
    "aten::diagonal_scatter", "aten::as_strided_scatter",
    "aten::masked_scatter", "aten::masked_fill", "aten::scatter",
    "aten::scatter_add", "aten::scatter_reduce",
})
# The assemblers that choose each element from their sources in place,
# broadcasting them: a source constant along a dimension the result varies
# along is a level there (a learned scalar or per-channel alpha in
# where(x > 0, alpha, -alpha)), as a number is (_levels_only).
_CHOOSERS = frozenset({"aten::where", "aten::masked_fill", "aten::maximum",
                       "aten::minimum", "aten::fmax", "aten::fmin"})
# Elementwise arithmetic: followed into each operand that carries the
# elements, as a dequantisation's codes do (_carriers: an operand the others
# act on as a scale or an offset, by shape or by identity), so a recipe's
# codes or grid times a scale, plus a zero point, is still found. A sum is
# read as its terms (_sum_terms). Between values that all vary, with no scale
# among them, the op computes new values unless they are all on grids
# (_grid_algebra).
_SUMS = frozenset({"aten::add", "aten::sub", "aten::rsub"})
_PRODUCTS = frozenset({"aten::mul", "aten::div"})
# The remainders of a division, read in a sum as two terms (_sum_terms):
# remainder(x, s) is x - floor(x / s) * s, fmod(x, s) is x - trunc(x / s) * s,
# frac(x) (which to_edge decomposes into x - trunc(x)) is x - trunc(x). The
# second term is on a grid (s's, whatever s is, as round(x / s) * s is), so
# x minus its remainder is that grid, and a remainder alone is x minus a
# grid: a value the model computes.
_RESIDUES = frozenset({"aten::remainder", "aten::fmod", "aten::frac"})
# Moves that keep every element in its place with its value: the same tensor
# under another name, a copy, a detached alias (a straight-through
# estimator's .detach() is alias_copy on the edge graph).
_IN_PLACE = frozenset({
    "aten::alias_copy", "aten::alias", "aten::clone", "aten::detach_copy",
    "aten::detach", "aten::lift_fresh_copy", "aten::contiguous",
    "dim_order_ops::_clone_dim_order",
})
# The ops that compute new values from their inputs: a floating result over
# a runtime value is plain float. Listed, not assumed: an op on none of these
# lists is refused by name (UNRECOGNISED), until it is proven and listed.
_COMPUTES = frozenset({
    # matmuls and convolutions
    "aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul",
    "aten::linear", "aten::mv", "aten::addmv", "aten::dot", "aten::einsum",
    "aten::convolution", "aten::_convolution", "aten::conv1d",
    "aten::conv2d", "aten::conv3d", "aten::conv_transpose1d",
    "aten::conv_transpose2d", "aten::conv_transpose3d",
    # normalisations and softmax
    "aten::native_layer_norm", "aten::layer_norm", "aten::native_group_norm",
    "aten::group_norm", "aten::_native_batch_norm_legit_no_training",
    "aten::_native_batch_norm_legit", "aten::batch_norm",
    "aten::native_batch_norm", "aten::rms_norm", "aten::_fused_rms_norm",
    "aten::_softmax", "aten::softmax", "aten::_log_softmax",
    "aten::log_softmax", "aten::_safe_softmax",
    # transcendental and other unary functions
    "aten::exp", "aten::exp2", "aten::expm1", "aten::log", "aten::log2",
    "aten::log10", "aten::log1p", "aten::sqrt", "aten::rsqrt",
    "aten::reciprocal", "aten::pow", "aten::sin", "aten::cos", "aten::tan",
    "aten::asin", "aten::acos", "aten::atan", "aten::atan2", "aten::sinh",
    "aten::cosh", "aten::tanh", "aten::asinh", "aten::acosh", "aten::atanh",
    "aten::erf", "aten::erfc", "aten::erfinv", "aten::sigmoid", "aten::logit",
    "aten::gelu", "aten::silu", "aten::elu", "aten::selu", "aten::celu",
    "aten::leaky_relu", "aten::prelu", "aten::_prelu_kernel",
    "aten::hardswish", "aten::hardsigmoid", "aten::softplus", "aten::mish",
    "aten::glu", "aten::lgamma", "aten::digamma",
    # arithmetic of several values (remainders: _RESIDUES)
    "aten::addcmul", "aten::addcdiv", "aten::lerp", "aten::hypot",
    "aten::logaddexp", "aten::logaddexp2", "aten::xlogy",
    # reductions and scans
    "aten::mean", "aten::sum", "aten::prod", "aten::var", "aten::std",
    "aten::var_mean", "aten::std_mean", "aten::logsumexp", "aten::norm",
    "aten::linalg_vector_norm", "aten::cumsum", "aten::cumprod",
    "aten::logcumsumexp",
    # pooling and interpolation that average
    "aten::avg_pool1d", "aten::avg_pool2d", "aten::avg_pool3d",
    "aten::adaptive_avg_pool1d", "aten::adaptive_avg_pool2d",
    "aten::_adaptive_avg_pool2d", "aten::adaptive_avg_pool3d",
    "aten::_adaptive_avg_pool3d", "aten::upsample_bilinear2d",
    "aten::upsample_linear1d", "aten::upsample_trilinear3d",
    "aten::upsample_bicubic2d", "aten::grid_sampler_2d",
    # distances, folds, dropout's scaled values (not an embedding bag: a
    # sum of a table's rows at indices that may be codes, refused by name)
    "aten::_cdist_forward", "aten::_pdist_forward", "aten::col2im",
    "aten::native_dropout",
})
# Ops with no tensor input to take values from (factories): their values
# are the program's own. The operand's constness decides (is_constant).
_FACTORIES = frozenset({
    "aten::full", "aten::zeros", "aten::ones", "aten::empty", "aten::arange",
    "aten::linspace", "aten::eye", "aten::scalar_tensor", "aten::full_like",
    "aten::zeros_like", "aten::ones_like", "aten::empty_like",
    "aten::empty_strided", "aten::new_zeros", "aten::new_ones",
    "aten::new_full", "aten::new_empty", "aten::fill", "aten::rand",
    "aten::randn", "aten::rand_like", "aten::randn_like",
    "dim_order_ops::_empty_dim_order",
})
# The factories that fill every element with one value: a scale or an offset
# whatever shape they are given (one class along every dimension, _classes).
_UNIFORM_FACTORIES = frozenset({
    "aten::full", "aten::zeros", "aten::ones", "aten::scalar_tensor",
    "aten::full_like", "aten::zeros_like", "aten::ones_like",
    "aten::new_zeros", "aten::new_ones", "aten::new_full", "aten::fill",
})
# The factories of random values: plain values, not a few fixed levels
# (_levels_only).
_RANDOM_FACTORIES = frozenset({"aten::rand", "aten::randn", "aten::rand_like",
                               "aten::randn_like"})
# Words in the name of an op whose value is not a function of its arguments
# alone: random ops and uninitialised factories. Structural equality never
# merges two of them (_deterministic).
_RANDOM_WORDS = ("rand", "bernoulli", "normal", "uniform", "multinomial",
                 "dropout", "exponential", "geometric", "cauchy", "poisson",
                 "empty")
# The backend's own ops that compute plain float values from float inputs.
# Its quantised ops (quant_lut, lut16, qnorm, the K cache's quantise and
# dequantise) are not among them: their values are a table's or a grid's.
_PLAIN_ESPRESSIF = frozenset({
    "espressif::rmsnorm", "espressif::rmsnorm_noscale",
    "espressif::rmsnorm_add", "espressif::rope", "espressif::layer_norm2d",
    "espressif::cached_attention", "espressif::attn_tail",
    "espressif::ssm_step", "espressif::conv_step",
})
# Namespaces whose ops are a recipe's quantisation.
_RECIPE_NAMESPACES = frozenset({"quantized_decomposed", "torchao", "quantized"})
# Words that make an ATen or backend op a quantisation: aten's
# fake_quantize_*, quantize_per_tensor, dequantize, int_repr, the backend's
# quantize_rows_i8, dequantize_per_token_i8, quant_lut, qnorm.
_QUANT_WORDS = ("quantize", "fake_quant", "qparams", "int_repr", "q_scale",
                "q_zero_point", "q_per_channel", "quant_lut", "lut16", "qnorm")
_ROUNDING = frozenset({"aten::round", "aten::floor", "aten::ceil",
                       "aten::trunc"})
_CLAMPS = frozenset({"aten::clamp", "aten::clip", "aten::clamp_min",
                     "aten::clamp_max"})
# Elementwise ops: each output element is a function of the same element of
# each input, broadcast. Two elements of a value one computes are equal
# wherever every input's are (_classes).
_ELEMENTWISE = frozenset({
    "aten::add", "aten::sub", "aten::rsub", "aten::mul", "aten::div",
    "aten::remainder", "aten::fmod", "aten::frac", "aten::floor_divide",
    "aten::neg", "aten::abs", "aten::relu", "aten::hardtanh", "aten::clamp",
    "aten::clip", "aten::clamp_min", "aten::clamp_max", "aten::round",
    "aten::floor", "aten::ceil", "aten::trunc", "aten::sign", "aten::sgn",
    "aten::where", "aten::masked_fill", "aten::maximum", "aten::minimum",
    "aten::fmax", "aten::fmin", "aten::gt", "aten::ge", "aten::lt",
    "aten::le", "aten::eq", "aten::ne", "aten::logical_not",
    "aten::logical_and", "aten::logical_or", "aten::logical_xor",
    "aten::bitwise_not", "aten::bitwise_and", "aten::bitwise_or",
    "aten::bitwise_xor", "aten::isnan", "aten::isinf", "aten::exp",
    "aten::exp2", "aten::expm1", "aten::log", "aten::log2", "aten::log10",
    "aten::log1p", "aten::sqrt", "aten::rsqrt", "aten::reciprocal",
    "aten::pow", "aten::sin", "aten::cos", "aten::tan", "aten::asin",
    "aten::acos", "aten::atan", "aten::atan2", "aten::sinh", "aten::cosh",
    "aten::tanh", "aten::asinh", "aten::acosh", "aten::atanh", "aten::erf",
    "aten::erfc", "aten::erfinv", "aten::sigmoid", "aten::logit",
    "aten::gelu", "aten::silu", "aten::elu", "aten::selu", "aten::celu",
    "aten::leaky_relu", "aten::hardswish", "aten::hardsigmoid",
    "aten::softplus", "aten::mish", "aten::lgamma", "aten::digamma",
    "aten::addcmul", "aten::addcdiv", "aten::lerp", "aten::hypot",
    "aten::logaddexp", "aten::logaddexp2", "aten::xlogy",
}) | _CASTS | _IN_PLACE
# Views that change only the shape: each dimension of the result spans a run
# of the argument's dimensions, read in row-major order (_reshape_classes).
_RESHAPES = frozenset({
    "aten::view_copy", "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::squeeze_copy", "aten::squeeze", "aten::unsqueeze_copy",
    "aten::unsqueeze", "aten::flatten", "aten::unflatten",
})
# Moves that put each element where their own arguments alone say, whatever
# the values: one such move applied to two values of one shape puts their
# elements at the same places, so the sign identity strips a move written on
# both of its sides (_peeled, _same_moves).
_POSITIONAL = frozenset({
    "aten::permute_copy", "aten::permute", "aten::transpose_copy",
    "aten::transpose", "aten::t_copy", "aten::t", "aten::flip", "aten::roll",
    "aten::slice_copy", "aten::slice", "aten::select_copy", "aten::select",
    "aten::narrow_copy", "aten::narrow", "aten::expand_copy", "aten::expand",
    "aten::repeat",
}) | _RESHAPES
# The moves whose result's partitions _classes reads from their argument's:
# copies (expand, repeat, repeat_interleave), reshapes, reorderings
# (permute, transpose, select), the indices they keep (slice, narrow, flip,
# roll, a gather along one dimension) and a concatenation.
_PARTITION_MOVES = frozenset({
    "aten::expand_copy", "aten::expand", "aten::repeat",
    "aten::repeat_interleave", "aten::permute_copy", "aten::permute",
    "aten::transpose_copy", "aten::transpose", "aten::t_copy", "aten::t",
    "aten::select_copy", "aten::select", "aten::slice_copy", "aten::slice",
    "aten::narrow_copy", "aten::narrow", "aten::flip", "aten::roll",
    "aten::index_select", "aten::index", "aten::_unsafe_index",
    "aten::gather", "aten::cat",
}) | _RESHAPES
# The limits of the search. Each is far past what a graph needs, and none
# admits when it is reached: a hard limit refuses the operand by name
# (_Exhausted, which origin() turns into UNRECOGNISED), and a soft one makes
# the search follow more operands, never fewer (test_qk_matrix forces each
# to its minimum and holds every grid origin refused).
#
# Hard. How many moves a walk takes, sources a search judges, nodes a sum's
# reading visits, and clamps, guards, aliases and factors the identities
# step through: a walk follows each node's first input, so it ends within
# the graph's length.
_BUDGET = 100_000
# Hard. How deeply arithmetic of values that all vary may nest, each level
# asking whether its operands are on grids (_grid_algebra).
_NESTING = 64
# Hard. How many nodes the walk over a gather's indices visits from one
# index (_float_source): every dependency of the index is followed.
_INDEX_BUDGET = 100_000
# Hard. How many pairs of values structural equality compares in one
# operand's search (_same_value; each pair once): it needs no depth bound,
# and a pathological graph is refused rather than compared past this.
_SAME_BUDGET = 1 << 20
# Soft. How many nodes _classes reads partitions of in one operand's search
# (each once, inputs first, without recursion): past it a partition is not
# read, which the carrier test takes as possibly one class in a partner
# (the candidate is followed) and as varying in the candidate itself.
_VARIES_BUDGET = 100_000
# Soft. How many nodes the search for a rounding's divisor
# (_rounding_factors) visits from one value, and how many moves it walks
# from a rounding to the multiply or divide in its argument: a search that
# does not finish takes the other operand for the step, and the value is
# followed.
_FACTOR_BUDGET = 4096
_FACTOR_DEPTH = 64
# A partition not read (an input's unknown shape, a source's partition not
# read).
_UNREAD = object()
# Not a limit but a threshold of the rules: the largest number a guard
# against zero adds or clamps a magnitude to (_guarded). x / (abs(x) + eps)
# is sign for an eps this small; x / max(abs(x), 1) is clamp(x, -1, 1),
# real computation.
_GUARD = 1e-6


class Origin(NamedTuple):
    """What an operand's values come from: `kind` (one of the names above),
    the node the walk found it at, the moves walked to reach it (names), and
    what that node is."""

    kind: str
    node: object
    path: tuple
    detail: str


def _schema_name(node) -> Optional[str]:
    from executorch.backends.esp32s3.partition.espressif_partitioner import _overload

    op = _overload(node)
    return None if op is None else op._schema.name


def _val(node):
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else None


def _dtype(node):
    return getattr(_val(node), "dtype", None)


def _is_tensor_node(node) -> bool:
    return isinstance(node, torch.fx.Node) and isinstance(_val(node),
                                                          torch.Tensor)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _call_name(node):
    """The schema name of a call_function node, else None."""
    if isinstance(node, torch.fx.Node) and node.op == "call_function":
        return _schema_name(node)
    return None


def _by_one_value(node):
    """The tensor a mul or div multiplies or divides by one value (a Python
    number, or a one-element tensor of no more dims), or None."""
    name = _schema_name(node)
    if name not in ("aten::mul", "aten::div") or len(node.args) != 2:
        return None
    if name == "aten::div" and node.kwargs.get("rounding_mode") is not None:
        return None
    a, b = node.args
    if isinstance(a, torch.fx.Node) and isinstance(b, (int, float)):
        return a
    if name == "aten::mul" and isinstance(b, torch.fx.Node) and isinstance(
            a, (int, float)):
        return b
    pairs = ((a, b), (b, a)) if name == "aten::mul" else ((a, b),)
    for t, factor in pairs:
        tv, fv = _val(t), _val(factor)
        if (isinstance(tv, torch.Tensor) and isinstance(fv, torch.Tensor)
                and fv.numel() == 1 and fv.dim() <= tv.dim()
                and tv.numel() > 1):
            return t
    return None


def _move_source(node):
    """The tensor `node` takes its elements from unchanged, or None."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return None
    if node.target is operator.getitem:
        src = node.args[0]
        if _schema_name(src) in _SPLITS and src.args:
            return src.args[0]
        return None
    name = _schema_name(node)
    if name in _MOVES or name in _CASTS:
        src = node.args[0] if node.args else None
        if not _is_tensor_node(src) or not isinstance(_val(node), torch.Tensor):
            return None
        a, b = _dtype(src), _dtype(node)
        if a == b or (name in _CASTS and a.is_floating_point
                      and b.is_floating_point):
            return src
        return None
    # Not past a quantisation's own multiply by its scale: the strip's
    # in-place chain value is the chain itself (the search names it).
    from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
        CHAIN_VALUE,
    )

    if CHAIN_VALUE in node.meta:
        return None
    scaled = _by_one_value(node)
    if scaled is not None and _is_tensor_node(scaled) and \
            _dtype(scaled) == _dtype(node):
        return scaled
    return None


def walk(value):
    """(the node `value`'s elements come from, the moves walked from `value`
    to it, nearest first). A move is a node of _MOVES or a float-to-float
    cast, a split read through getitem, or a multiply or divide by one
    value. A walk that does not end within _BUDGET moves returns None for
    the node: the caller refuses the operand."""
    path = []
    while len(path) < _BUDGET:
        src = _move_source(value)
        if src is None:
            return value, tuple(path)
        path.append(value)
        value = src
    return None, tuple(path)


class _Exhausted(Exception):
    """A hard limit of the search reached (`limit`, its constant's name):
    origin() refuses the operand as unrecognised, naming the limit. Nothing
    a search cut short decides is ever an admission."""

    def __init__(self, limit, detail):
        super().__init__(f"{limit}: {detail}")
        self.limit = limit
        self.detail = detail


def _walk(value):
    """walk(), raising _Exhausted when it does not end within _BUDGET
    moves."""
    node, path = walk(value)
    if node is None:
        raise _Exhausted("_BUDGET", f"a walk of {len(path)} moves from "
                         f"{getattr(value, 'name', value)} that does not end")
    return node, path


class _Signature:
    """The program's placeholders by kind, read once: ExportedProgram's
    signature properties rebuild a dict of every input spec on each read."""

    def __init__(self, ep):
        self.ep = ep
        if ep is None:
            return
        sig = ep.graph_signature
        self.params = sig.inputs_to_parameters
        self.lifted = sig.inputs_to_lifted_tensor_constants
        self.buffers = sig.inputs_to_buffers
        self.written = set(sig.buffers_to_mutate.values())

    def placeholder_origin(self, node, path):
        if self.ep is None:
            # A gate without the program (preprocess's own probes) cannot
            # tell a parameter from an input; the partitioner's gate always
            # has it.
            return Origin(PLAIN, node, path, "a graph input (unchecked: no "
                          "program)")
        if node.name in self.params:
            return Origin(CONSTANT, node, path,
                          f"parameter {self.params[node.name]}")
        if node.name in self.lifted:
            return Origin(CONSTANT, node, path, "a lifted constant")
        if node.name in self.buffers:
            fqn = self.buffers[node.name]
            if fqn in self.written:
                return Origin(STATE, node, path,
                              f"buffer {fqn}, read as it was before the "
                              f"forward wrote it")
            return Origin(CONSTANT, node, path, f"buffer {fqn}, which "
                          f"nothing writes")
        if not isinstance(_val(node), torch.Tensor) or not \
                _dtype(node).is_floating_point:
            return Origin(UNRECOGNISED, node, path,
                          f"a graph input of dtype {_dtype(node)}")
        return Origin(PLAIN, node, path, "a graph input")

    def written_fqn(self, node):
        """The FQN of the buffer placeholder `node` when the forward writes
        it, else None."""
        if self.ep is None or not (isinstance(node, torch.fx.Node)
                                   and node.op == "placeholder"):
            return None
        fqn = self.buffers.get(node.name)
        return fqn if fqn in self.written else None

    def constant_fqn(self, node):
        """The FQN of the constant placeholder `node` holds -- a parameter,
        a lifted constant, a buffer nothing writes -- else None."""
        if self.ep is None or not (isinstance(node, torch.fx.Node)
                                   and node.op == "placeholder"):
            return None
        if node.name in self.params:
            return self.params[node.name]
        if node.name in self.lifted:
            return self.lifted[node.name]
        fqn = self.buffers.get(node.name)
        return fqn if fqn is not None and fqn not in self.written else None

    def is_constant(self, node, memo):
        """Whether the program's constants alone determine `node`'s value:
        parameters, lifted constants, buffers nothing writes, and pure ops
        over them (an op with no tensor input, a full or an arange, too)."""
        if not isinstance(node, torch.fx.Node):
            return True
        # Depth first without recursion: an activation's inputs reach back
        # through every layer before it, past Python's recursion limit in a
        # real model. Each node's inputs are judged in order, and the first
        # that is not constant decides it.
        stack = [node]
        while stack:
            n = stack[-1]
            if n in memo:
                stack.pop()
                continue
            if n.op == "placeholder":
                memo[n] = self.ep is not None and (
                    n.name in self.params or n.name in self.lifted
                    or (n.name in self.buffers
                        and self.buffers[n.name] not in self.written))
            elif n.op == "get_attr":
                memo[n] = True
            elif n.op != "call_function" or n.is_impure():
                memo[n] = False
            else:
                pending, ok = None, True
                for a in n.all_input_nodes:
                    if a not in memo:
                        pending = a
                        break
                    if not memo[a]:
                        ok = False
                        break
                if pending is not None:
                    stack.append(pending)
                    continue
                memo[n] = ok
            stack.pop()
        return memo[node]


def _rounding_division(node, name):
    """A division that rounds its quotient: floor_divide (x // s), or div
    with rounding_mode 'floor' or 'trunc'."""
    if name == "aten::floor_divide":
        return True
    if name != "aten::div":
        return False
    mode = node.kwargs.get("rounding_mode",
                           node.args[2] if len(node.args) > 2 else None)
    return mode in ("floor", "trunc")


def _aligned_shape(node, rank):
    """`node`'s shape left-padded with 1s to `rank` (as broadcasting aligns
    it), or None when unknown or symbolic."""
    v = _val(node)
    if not isinstance(v, torch.Tensor):
        return None
    try:
        shape = [int(d) for d in v.shape]
    except (TypeError, RuntimeError):         # a symbolic size
        return None
    return [1] * (rank - len(shape)) + shape


# --- partitions: which elements along a dimension hold one value -------------
#
# The shape test reads, for each dimension of a value, a partition of its
# indices: two indices in one class hold equal values wherever the other
# indices are. A partition is an array of class labels, one per index: the
# identity (every index its own class) along a dimension the value varies
# along, one class along one it is constant along (a broadcast, a factory
# of one value), classes of four along the width of a group-wise scale
# repeated to it. None is a dimension whose partition was not read.

def _ident(n):
    return np.arange(n, dtype=np.int64)


def _unif(n):
    return np.zeros(n, dtype=np.int64)


def _relabel(c):
    return np.unique(c, return_inverse=True)[1].reshape(-1).astype(np.int64)


def _is_ident(c):
    return len(np.unique(c)) == len(c)


def _is_unif(c):
    return len(c) == 0 or bool((c == c[0]).all())


def _meet(parts):
    """The partition in which two indices share a class only when they share
    one in each of `parts` (known partitions of one length)."""
    out = parts[0]
    for c in parts[1:]:
        out = _relabel(out * (int(c.max(initial=0)) + 1) + c)
    return out


def _meet_known(parts, n):
    """_meet over partitions some of which may not be read (None): the
    identity if one of them is (whatever the others), unread if one is not
    read, one class if there are none."""
    if not parts:
        return _unif(n)
    if any(c is not None and _is_ident(c) for c in parts):
        return _ident(n)
    if any(c is None for c in parts):
        return None
    return _meet(parts)


def _distinguishes(cx, others):
    """Whether two indices that every partition of `others` puts in one
    class can lie in two classes of `cx`."""
    m = _meet(others) if others else _unif(len(cx))
    pairs = m * (int(cx.max(initial=0)) + 1) + cx
    return len(np.unique(pairs)) > len(np.unique(m))


def _merge(parts):
    """One partition over the row-major index of dimensions whose partitions
    are `parts`, outermost first: two flat indices share a class when every
    dimension's index does."""
    flat = parts[0]
    for c in parts[1:]:
        flat = _relabel((flat[:, None] * (int(c.max(initial=0)) + 1)
                         + c[None, :]).reshape(-1))
    return flat


def _split(flat, sizes):
    """The partition of each dimension of `sizes`, read from one over their
    row-major index (`flat`): two indices along a dimension share a class when
    at every other index their elements share one. (Two elements whose
    indices share a class along every dimension then share one: a product of
    these partitions never claims more than `flat` does.)"""
    if len(sizes) == 1:
        return [flat]
    grid = flat.reshape(sizes)
    out = []
    for k, n in enumerate(sizes):
        rows = np.moveaxis(grid, k, 0).reshape(n, -1)
        out.append(np.unique(rows, axis=0, return_inverse=True)[1]
                   .reshape(-1).astype(np.int64))
    return out


def _reshape_classes(in_shape, in_parts, out_shape):
    """The partitions of a reshape to `out_shape` of a value of `in_shape`
    whose partitions are `in_parts`: each run of the argument's dimensions
    (sizes of 1 aside) is read as one row-major index (a reshape keeps the
    row-major order) and split into the result's run. A run with a dimension
    not read is not read. None when the sizes do not factor into runs."""
    ins = [(s, c) for s, c in zip(in_shape, in_parts) if s != 1]
    outs = [i for i, s in enumerate(out_shape) if s != 1]
    parts = [_unif(1) if s == 1 else None for s in out_shape]
    i = j = 0
    while i < len(ins) and j < len(outs):
        p_in, p_out = ins[i][0], out_shape[outs[j]]
        run_in, run_out = [i], [j]
        while p_in != p_out:
            if p_in < p_out:
                i += 1
                if i == len(ins):
                    return None
                p_in *= ins[i][0]
                run_in.append(i)
            else:
                j += 1
                if j == len(outs):
                    return None
                p_out *= out_shape[outs[j]]
                run_out.append(j)
        cs = [ins[k][1] for k in run_in]
        if all(c is not None for c in cs):
            sizes = [out_shape[outs[k]] for k in run_out]
            for k, part in zip(run_out, _split(_merge(cs), sizes)):
                parts[outs[k]] = part
        i, j = i + 1, j + 1
    if i != len(ins) or j != len(outs):
        return None
    return parts


def _align(parts, shape, sizes):
    """Partitions `parts` of a value of `shape`, broadcast to `sizes` (right
    aligned): a dimension it lacks or has of size 1 is one class. None when
    the shapes do not broadcast."""
    lead = len(sizes) - len(shape)
    if lead < 0:
        return None
    out = [_unif(n) for n in sizes[:lead]]
    for c, s, n in zip(parts, shape, sizes[lead:]):
        if s == n:
            out.append(c)
        elif s == 1:
            out.append(_unif(n))
        else:
            return None
    return out


def _broadcast_shape(shapes):
    """The shape `shapes` broadcast to, or None when they do not."""
    rank = max((len(s) for s in shapes), default=0)
    out = []
    for d in range(rank):
        sizes = {s[d - rank + len(s)] for s in shapes
                 if d - rank + len(s) >= 0} - {1}
        if len(sizes) > 1:
            return None
        out.append(sizes.pop() if sizes else 1)
    return out


def _payload_dim(t, d):
    """The partition of dimension d of constant tensor `t`: indices whose
    slices hold the same bits share a class (no size limit: one vectorised
    pass)."""
    n = int(t.shape[d])
    rows = t.detach().movedim(d, 0).reshape(n, -1)
    if rows.shape[1] == 0:
        return _unif(n)
    if rows.dtype is torch.bool:
        rows = rows.to(torch.uint8)
    elif rows.is_floating_point():
        rows = rows.contiguous().view({2: torch.int16, 4: torch.int32,
                                       8: torch.int64}[rows.element_size()])
    return np.unique(rows.contiguous().numpy(), axis=0,
                     return_inverse=True)[1].reshape(-1).astype(np.int64)


def _in_place_source(node):
    """`node` past the moves that keep every element in place with its
    value (_IN_PLACE, a cast to its own dtype)."""
    while isinstance(node, torch.fx.Node) and node.op == "call_function" \
            and node.args and _is_tensor_node(node.args[0]):
        name = _schema_name(node)
        if not (name in _IN_PLACE or (name in _CASTS
                                      and _dtype(node.args[0]) == _dtype(node))):
            break
        node = node.args[0]
    return node


def _is_reshape(node):
    """Whether `node` reshapes a tensor of its own dtype (_RESHAPES), both
    shapes static: its elements are its argument's, in row-major order."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"
            and node.args and _is_tensor_node(node.args[0])):
        return False
    if _schema_name(node) not in _RESHAPES:
        return False
    return (_dtype(node) == _dtype(node.args[0])
            and _aligned_shape(node, 0) is not None
            and _aligned_shape(node.args[0], 0) is not None)


def _reshape_root(node):
    """`node` past reshapes and aliases: the value whose row-major elements
    it holds."""
    node = _in_place_source(node)
    while _is_reshape(node):
        node = _in_place_source(node.args[0])
    return node


def _canonical(node):
    """`node` as the value it holds: past aliases (_in_place_source) and
    past a chain of reshapes that comes back to a value of its own shape
    (x.view(a).view(x.shape) is x: a reshape keeps the row-major order).
    Two nodes with one canonical node are one value."""
    node = _in_place_source(node)
    if not _is_reshape(node):
        return node
    shape = _aligned_shape(node, 0)
    root, cur = node, node
    while _is_reshape(cur):
        cur = _in_place_source(cur.args[0])
        if _aligned_shape(cur, 0) == shape:
            root = cur
    return root


def _move_word(m):
    """What positional move `m` does, for comparing it with another
    (_same_moves): ("reshape", its shape, its argument's shape), or (its
    target, its other arguments, its keyword arguments, its argument's shape,
    its shape)."""
    out, src = _aligned_shape(m, 0), _aligned_shape(m.args[0], 0)
    if _is_reshape(m):
        return ("reshape", tuple(out), tuple(src))
    return (m.target, tuple(m.args[1:]), tuple(sorted(m.kwargs.items())),
            None if src is None else tuple(src),
            None if out is None else tuple(out))


def _move_words(moves):
    """`moves` (outermost first) as _move_word's words, a run of reshapes one
    reshape (a reshape keeps the row-major order, so any run of them is one),
    and a reshape or an expand to its own argument's shape dropped."""
    words = []
    for m in moves:
        w = _move_word(m)
        if w[0] == "reshape" and words and words[-1][0] == "reshape":
            words[-1] = ("reshape", words[-1][1], w[2])
        else:
            words.append(w)
        last = words[-1]
        if (last[0] == "reshape" and last[1] == last[2]) or (
                _call_name(m) in ("aten::expand_copy", "aten::expand")
                and last[3] is not None and last[3] == last[4]):
            words.pop()
    return words


def _same_moves(ma, mb):
    """Whether two runs of positional moves (outermost first) put elements at
    the same places: the same moves in the same order, each with the same
    arguments (numbers bit for bit, _same_literal) over an argument of the
    same shape. A shape not read (symbolic) is left to the arguments: over
    one value (the identities' other test), the same moves place alike, and
    where a broadcast between them could make them differ the answer only
    refuses (a sign found)."""
    wa, wb = _move_words(ma), _move_words(mb)
    if len(wa) != len(wb):
        return False
    for a, b in zip(wa, wb):
        if a[0] == "reshape" or b[0] == "reshape":
            if a != b:
                return False
            continue
        if a[0] != b[0] or a[3] != b[3] or a[4] != b[4] or \
                len(a[1]) != len(b[1]) or len(a[2]) != len(b[2]):
            return False
        if not all(_same_literal(x, y) for x, y in zip(a[1], b[1])):
            return False
        if not all(ka == kb and _same_literal(x, y)
                   for (ka, x), (kb, y) in zip(a[2], b[2])):
            return False
    return True


def _deterministic(node):
    """Whether `node`'s value is a function of its arguments alone (structural
    equality merges only these, _same_value): an op that writes nothing and
    advances no random state -- torch's is_impure, which takes an op tagged
    nondeterministic_seeded for impure (binomial, whose name says nothing
    random) -- and no uninitialised factory (_RANDOM_WORDS: empty)."""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return False
    if node.target is operator.getitem:
        return _deterministic(node.args[0])
    from executorch.backends.esp32s3.partition.espressif_partitioner import _overload

    op = _overload(node)
    if op is None or op._schema.is_mutable or node.is_impure():
        return False
    return not any(w in op._schema.name for w in _RANDOM_WORDS)


def _alpha(node):
    """An add's, sub's or rsub's alpha (rsub.Scalar takes it third), or
    None when it has not two operands to sum."""
    if len(node.args) < 2:
        return None
    if len(node.args) > 2:
        return node.args[2]
    return node.kwargs.get("alpha", 1)


def _number_key(x):
    """A number's identity: the float64 it is, bit for bit (-0.0 is not
    0.0, a NaN is its own bits), or an int no float holds exactly."""
    if isinstance(x, int):
        f = float(x)
        if f != x:
            return ("int", x)
        x = f
    return ("float", struct.pack("<d", float(x)))


def _same_literal(a, b):
    """Whether two non-node arguments are the same: numbers bit for bit
    (_number_key: 2 is 2.0), anything else by type and equality."""
    if _is_number(a) and _is_number(b):
        return _number_key(a) == _number_key(b)
    if type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except Exception:                         # an argument with no equality
        return False


def _same_meta(a, b):
    """Whether two nodes' values have the same type, dtype and shape (each
    output of a multi-output op too)."""
    va, vb = _val(a), _val(b)
    if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
        try:
            return va.dtype == vb.dtype and tuple(int(d) for d in va.shape) \
                == tuple(int(d) for d in vb.shape)
        except (TypeError, RuntimeError):     # a symbolic size
            return False
    if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
        return len(va) == len(vb) and all(
            isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            and x.dtype == y.dtype and x.shape == y.shape
            for x, y in zip(va, vb))
    return False


# Ops whose two arguments commute (IEEE 754 addition and multiplication do,
# bit for bit): structural equality tries both orders.
_COMMUTATIVE = frozenset({"aten::mul", "aten::maximum", "aten::minimum"})


class _Residue(NamedTuple):
    """A remainder's grid term in a sum: floor(x / s) * s of
    remainder(x, s) at `node` (trunc for fmod and frac)."""

    node: object


class _Sum(NamedTuple):
    """A sum read as its terms (_Classifier._sum_terms): `terms`, each
    [coefficient, node, synthetic] (synthetic: the grid term of a remainder
    node, x - remainder(x, s)'s floor(x / s) * s), and `rounds`, a number or
    a constant term the sum adds and takes away again, else None."""

    terms: list
    rounds: object


def _mutation(ep, fqn):
    """The node whose value the forward writes back into buffer `fqn`."""
    for spec in ep.graph_signature.output_specs:
        if spec.kind.name == "BUFFER_MUTATION" and spec.target == fqn:
            name = getattr(spec.arg, "name", None)
            return next((n for n in ep.graph_module.graph.nodes
                         if n.name == name), None)
    return None


# What the provenance search may end at without refusing: values computed in
# plain float, graph inputs, and the program's own constants. Whether the
# operand is a constant is decided on the operand itself, at the end.
_NEUTRAL = "neutral"
# The origins that refuse an operand wherever the search finds them.
_REFUSING = frozenset({STATIC_PAIR, CONSTANT_CODES, CHAIN, CHAIN_IN_PLACE,
                       GRID, QUANTISED, STATE, CODES_CAST, UNRECOGNISED})


class _Classifier:
    def __init__(self, ep):
        self.ep = ep
        self.sig = _Signature(ep)
        self.memo = {}
        self.constants = {}
        self.parts = {}         # node: its dimensions' partitions, or None
        self.parts_read = 0     # nodes _classes has read
        self.payloads = {}      # node: the constant tensor it holds, or None
        self.same = {}          # (node, node): the same value, or not
        self.same_pairs = 0     # pairs _same_value has compared
        self.factors = {}       # node: its roundings' factors
        self.grids = {}         # node: what its own search refuses, or None
        self.indices = {}       # node: what makes an index codes, or None

    def origin(self, value) -> Origin:
        if value not in self.memo:
            self.memo[value] = self._origin(value)
        return self.memo[value]

    def _origin(self, value):
        node, path = _walk(value)
        names = tuple(n.name for n in path)
        if not isinstance(node, torch.fx.Node):
            return Origin(UNRECOGNISED, node, names, "not a graph value")
        if not (_is_tensor_node(value) and _dtype(value).is_floating_point):
            return Origin(UNRECOGNISED, value, (),
                          f"a value of dtype {_dtype(value)}")
        # A buffer the forward writes is judged by what it writes, read
        # directly, through moves or assembled: no partition takes one in
        # (the partitioner leaves every written buffer out of the constants
        # it pulls in), so the delegate reads it at run time, as it was
        # before the write, as the converted model does.
        refused, first = self._search(value)
        if refused is not None:
            return refused
        # No recipe, grid, codes or unknown op on any path. Constness is the
        # operand's own: a constant table gathered by runtime indices, or a
        # where choosing between constants by a runtime mask, is a runtime
        # value.
        if self.sig.is_constant(value, self.constants):
            if node.op in ("placeholder", "get_attr"):
                got = (self.sig.placeholder_origin(node, names)
                       if node.op == "placeholder" else
                       Origin(CONSTANT, node, names, "a module attribute"))
                if got.kind == CONSTANT:
                    return got
            return Origin(COMPUTED_CONSTANT, value, names,
                          f"{_schema_name(value) or value.op} over "
                          f"constants alone")
        return Origin(PLAIN, first[0] if first else value, names,
                      first[1] if first else "plain float")

    def _search(self, start, depth=0):
        """Depth first, without recursion, through every node `start`'s
        values come from: (the first refusing Origin found, or None; the
        first neutral end found, (node, what), or None). `depth` counts the
        levels of arithmetic of varying values whose operands this search
        judges (_grid_algebra)."""
        stack, seen, first = [(start, ())], set(), None
        expanded = set()            # buffers whose writes are followed
        while stack:
            n, trail = stack.pop()
            if n in seen or not isinstance(n, torch.fx.Node):
                continue
            seen.add(n)
            if len(seen) > _BUDGET:
                return Origin(UNRECOGNISED, n, trail,
                              f"more than {_BUDGET} sources to judge"), None
            step = self._step(n, expanded, depth)
            if isinstance(step, Origin):
                if step.kind in _REFUSING:
                    return step._replace(path=trail), None
                first = first or (n, step.detail)
                continue
            for source in reversed(step):
                stack.append((source, trail + (n.name,)))
        return None, first

    def _step(self, n, expanded, depth=0):
        """One node of the search: a refusing or neutral Origin, or the
        sources to follow."""
        if n.op == "placeholder":
            fqn = self.sig.written_fqn(n)
            if fqn is not None:
                if fqn in expanded:
                    return Origin(_NEUTRAL, n, (), f"buffer {fqn}")
                expanded.add(fqn)
                return self._written_sources(n, fqn)
            got = self.sig.placeholder_origin(n, ())
            if got.kind == UNRECOGNISED:
                return got
            return Origin(_NEUTRAL, n, (), got.detail)
        if n.op == "get_attr":
            return Origin(_NEUTRAL, n, (), "a module attribute")
        if n.op != "call_function":
            return Origin(UNRECOGNISED, n, (), f"a {n.op} node")
        from executorch.backends.esp32s3._passes.stage_b import IN, STAGE_B
        from executorch.backends.esp32s3._passes.static_qdq import (
            constant_codes_source,
            static_pair,
        )
        from executorch.backends.esp32s3._passes.strip_dynamic_act_qdq_pass import (
            CHAIN_VALUE,
            dynamic_chooser,
        )

        chain = n.meta.get(CHAIN_VALUE)
        if chain is not None:
            return Origin(CHAIN_IN_PLACE, n, (),
                          f"the dequantised value of {chain}'s dynamic chain, "
                          f"which strip_dynamic_act_qdq computed in place")
        if n.meta.get(STAGE_B) == IN:
            return Origin(STATIC_PAIR, n, (), "Stage B codes dequantised for "
                          "their consumer")
        q = static_pair(n)
        if q is not None:
            return Origin(STATIC_PAIR, n, (),
                          f"the dequantise of static quantize {q.name}")
        if self.ep is not None and constant_codes_source(n, self.ep,
                                                         depth=1) is not None:
            return Origin(CONSTANT_CODES, n, (),
                          "a dequantise of constant codes: a pair whose "
                          "quantize convert_pt2e folded")
        codes = self._constant_codes(n)
        if codes is not None:
            return Origin(CONSTANT_CODES, n, (),
                          f"a dequantise of constant codes ({codes}), "
                          f"whatever range it declares")
        chooser = dynamic_chooser(n, depth=1)
        if chooser is not None:
            return Origin(CHAIN, n, (),
                          f"the dequantise of a dynamic chain whose scale "
                          f"{chooser.name} computes")
        if n.target is operator.getitem:
            return self._getitem_step(n)
        name = _schema_name(n)
        if name is None:
            return Origin(UNRECOGNISED, n, (), f"{n.target}")
        namespace, _, short = name.partition("::")
        if namespace in _RECIPE_NAMESPACES or (
                namespace in ("aten", "espressif")
                and any(w in short for w in _QUANT_WORDS)):
            return Origin(QUANTISED, n, (), name)
        if name in _CASTS:
            src = n.args[0] if n.args else None
            dtype = _dtype(src)
            if dtype is not None and not dtype.is_floating_point:
                if dtype is torch.bool:
                    # A comparison's 0 and 1 carried into the operand (a
                    # mask that multiplies a value of its own shape is no
                    # carrier, and never reached): two levels, a grid --
                    # sign spelled (x > 0).float() * 2 - 1.
                    return Origin(GRID, n, (), f"{name} of a comparison: "
                                  f"its 0 and 1 are two levels, a grid the "
                                  f"model chose (sign spelled by hand)")
                return Origin(CODES_CAST, n, (), f"{name} of {dtype} values")
            return [src]
        if name in _ROUNDING or _rounding_division(n, name):
            return Origin(GRID, n, (), f"{name}: values rounded onto a grid "
                          f"in float ops (a quantise-dequantise or fake "
                          f"quantisation spelled in ATen)")
        codes = self._index_codes(n, name)
        if codes is not None:
            scaled = "" if self.sig.is_constant(
                _GATHERS[name](n)[0], self.constants) else (
                " (scaled or offset by values one along the lookup)")
            return Origin(GRID, n, (), f"{name} of a constant table{scaled} "
                          f"at indices that {codes.name} "
                          f"({_call_name(codes) or codes.op}) computes from a "
                          f"float value: the table's values at the model's "
                          f"own codes, a few fixed levels (a lookup-table "
                          f"quantiser)")
        if name in _MOVES:
            return [n.args[0]]
        if name in ("aten::max", "aten::min") and len(
                [a for a in n.args[:2] if _is_tensor_node(a)]) == 2:
            return list(n.args[:2])      # elementwise max/min.other
        if name in _KEEPS or name in ("aten::max", "aten::min"):
            sources = [n.args[0]]
            if name in _CLAMPS:          # a bound may be a tensor of values
                sources += [a for a in list(n.args[1:3]) + [
                    n.kwargs.get("min"), n.kwargs.get("max")]
                    if _is_tensor_node(a)]
            return sources
        if name in _ASSEMBLERS:
            sources = _ASSEMBLERS[name](n)
            levels = self._levels_only(n, name, sources)
            if levels is _UNREAD:
                return Origin(UNRECOGNISED, n, (), f"{name} of a runtime "
                              f"value whose partition the q@kT allow-list "
                              f"did not read: it may be a level (a row's own "
                              f"scale) or not")
            if levels:
                return Origin(GRID, n, (), f"{name} of levels alone: "
                              f"numbers, factories and values constant where "
                              f"the result varies, a few fixed levels, a "
                              f"grid the model chose (sign spelled "
                              f"where(x > 0, 1., -1.), a binarised "
                              f"where(x > 0, alpha, -alpha))")
            return [s for s in sources if _is_tensor_node(s)]
        if name in _SUMS or name in _RESIDUES:
            return self._sum_step(n, name, depth)
        if name in _PRODUCTS:
            return self._product_step(n, name, depth)
        if name in _COMPUTES or name in _PLAIN_ESPRESSIF:
            return Origin(_NEUTRAL, n, (), name)
        if name in _FACTORIES:
            return Origin(_NEUTRAL, n, (), f"{name} (a factory)")
        return Origin(UNRECOGNISED, n, (), f"{name}, an op the q@kT "
                      f"allow-list does not know")

    # --- arithmetic --------------------------------------------------------

    def _product_step(self, n, name, depth):
        """A multiply or a divide: sign spelled by hand is a grid; followed
        into the operands that carry the elements (_carriers); else the
        algebra of grids decides (_grid_algebra)."""
        if self._sign_identity(n, name):
            return Origin(GRID, n, (), f"{name} of a value and its own "
                          f"magnitude: x / abs(x) is sign(x), two levels, a "
                          f"grid the model chose (sign spelled by hand)")
        carriers = self._carriers(n, name)
        if carriers:
            return carriers
        return self._grid_algebra(
            n, name, [(a, False) for a in n.args[:2] if _is_tensor_node(a)],
            depth)

    def _sum_step(self, n, name, depth):
        """A sum, or a remainder, read as its terms (_sum_terms): a sum
        through a number or a constant and back is the magic-number round, a
        grid; terms that cancel leave what remains; one term left is
        followed, a remainder's grid term refused; several are followed into
        those that carry the elements, else the algebra of grids decides."""
        if not self._is_linear(n):
            return Origin(UNRECOGNISED, n, (), f"{name} whose terms the q@kT "
                          f"allow-list cannot read")
        got = self._sum_terms(n)
        if got.rounds is not None:
            what = getattr(got.rounds, "name", got.rounds)
            return Origin(GRID, n, (), f"{name} that adds {what} and takes it "
                          f"away again: the sum rounds its other terms to "
                          f"that term's ulp, as (x + 1.5 * 2**23) - 1.5 * 2**23 "
                          f"rounds x to an integer (the magic-number round)")
        kept = [(t, synthetic) for c, t, synthetic in got.terms if c != 0]
        if not kept:
            return Origin(GRID, n, (), f"{name} whose terms cancel to numbers "
                          f"alone: one level")
        if len(kept) == 1:
            t, synthetic = kept[0]
            return self._residue_origin(t) if synthetic else [t]
        nodes = [t for t, _ in kept]
        carriers = self._carriers_among(nodes, nodes)
        if carriers:
            residue = next((t for t, synthetic in kept
                            if synthetic and t in carriers), None)
            if residue is not None:
                return self._residue_origin(residue)
            return carriers
        return self._grid_algebra(n, name, kept, depth)

    def _grid_algebra(self, n, name, operands, depth):
        """Arithmetic `n` between `operands` ([(node, synthetic)]) that all
        vary, with no scale among them: a grid when every one of them is on
        a grid -- its own search refuses it, or it is a remainder's grid term
        -- else new values (neutral). Operands likeliest to be plain are
        judged first, and the first plain one decides."""
        if not operands:
            return Origin(_NEUTRAL, n, (), name)
        if depth >= _NESTING:
            return Origin(UNRECOGNISED, n, (), f"arithmetic of values that "
                          f"all vary, nested past {_NESTING} levels")
        found = []
        for t, synthetic in sorted(operands, key=self._plain_first):
            got = (self._residue_origin(t) if synthetic
                   else self._grid_ish(t, depth + 1))
            if got is None:
                return Origin(_NEUTRAL, n, (), f"{name} of values that all "
                              f"vary and are not all on grids: new values")
            found.append(got)
        unknown = next((f for f in found if f.kind == UNRECOGNISED), None)
        if unknown is not None:
            return unknown
        first = found[0]
        return Origin(GRID, n, (), f"{name} of values that are each on a grid "
                      f"or a few levels (one: "
                      f"{getattr(first.node, 'name', first.node)}, "
                      f"{first.kind}), so on a grid itself")

    def _grid_ish(self, t, depth):
        """What `t`'s own provenance search refuses it for, or None: whether
        it is on a grid, for arithmetic that combines it (memoised)."""
        if t not in self.grids:
            self.grids[t] = self._search(t, depth)[0]
        return self.grids[t]

    def _plain_first(self, operand):
        t, synthetic = operand
        if synthetic:
            return 2
        base, _ = walk(t)
        if isinstance(base, torch.fx.Node) and (
                base.op in ("placeholder", "get_attr")
                or _call_name(base) in _COMPUTES
                or _call_name(base) in _PLAIN_ESPRESSIF):
            return 0
        return 1

    def _residue_origin(self, r):
        name = _schema_name(r)
        grid = {"aten::remainder": "floor(x / s) * s",
                "aten::fmod": "trunc(x / s) * s"}.get(name, "trunc(x)")
        return Origin(GRID, r, (), f"{name}: x minus its remainder is {grid}, "
                      f"values rounded onto a grid in float ops")

    def _sum_terms(self, node):
        """The sum `node` computes, read as its terms (a _Sum): through add,
        sub and rsub (their alpha too), neg, a multiply or divide by a
        number, and the moves that keep each element in place, down to the
        first other op on each path; a remainder is x minus its grid term.
        Equal terms merge (_same_value): a straight-through estimator's
        x + (fq - x).detach() is {x: 0, fq: 1}, fq's value exactly where
        fq - x is exact, as it is between a value and its own grid point
        (Sterbenz), and fq's grid in any case. A sum inside the sum is read
        once its own coefficient is whole (consumers first, in graph order),
        so one that cancels -- the y of y + (fq - y), whatever y holds -- is
        never read. Numbers are offsets, kept aside; numbers that cancel, or
        a constant term that does, round the rest. Raises _Exhausted past
        _BUDGET nodes."""
        edges, rep_of, buckets, stack = {}, {}, {}, [node]
        while stack:
            n = stack.pop()
            if n in rep_of:
                continue
            if len(rep_of) >= _BUDGET:
                raise _Exhausted("_BUDGET", f"a sum at {node.name} of more "
                                 f"than {_BUDGET} nodes")
            got = self._linear_edges(n)
            rep = next((r for r in buckets.get(self._bucket(n), ())
                        if self._same_value(r, n)), None)
            if rep is None:
                rep = n
                buckets.setdefault(self._bucket(n), []).append(n)
                edges[rep] = got
            rep_of[n] = rep
            stack += [c for c, _ in got if self._is_linear(c)]
        position = self._positions(node)
        first = {}
        for n, rep in rep_of.items():
            first[rep] = min(first.get(rep, position[n]), position[n])
        coefficient = {rep_of[node]: 1.0}
        raw, numbers = [], []
        for rep in sorted(edges, key=lambda r: -first[r]):
            c = coefficient.get(rep, 0.0)
            if c == 0:
                continue                # cancelled whole: not read
            for child, f in edges[rep]:
                if _is_number(child):
                    if c * f * float(child) != 0:
                        numbers.append(c * f * float(child))
                elif isinstance(child, _Residue):
                    raw.append([c * f, child.node, True])
                elif self._is_linear(child):
                    r = rep_of[child]
                    coefficient[r] = coefficient.get(r, 0.0) + c * f
                elif isinstance(child, torch.fx.Node):
                    raw.append([c * f, child, False])
        # Merge equal terms: the same node, or structurally the same value
        # (compared only within a bucket of one op and one shape).
        terms, buckets, seen = [], {}, {}
        for c, n, synthetic in raw:
            t = seen.get((n, synthetic))
            if t is None:
                bucket = None
                if not synthetic:
                    bucket = self._bucket(n)
                    t = next((o for o in buckets.get(bucket, ())
                              if self._same_value(o[1], n)), None)
                if t is None:
                    t = [0.0, n, synthetic, []]
                    terms.append(t)
                    if bucket is not None:
                        buckets.setdefault(bucket, []).append(t)
                seen[(n, synthetic)] = t
            t[0] += c
            t[3].append(c)
        rounds = None
        if numbers and sum(numbers) == 0:
            rounds = max(numbers, key=abs)
        for c, n, synthetic, parts in terms:
            if c == 0 and len(parts) > 1 and not synthetic and self._fixed(n):
                rounds = n
        return _Sum([[c, n, synthetic] for c, n, synthetic, _ in terms],
                    rounds)

    def _linear_edges(self, n):
        """The terms the sum, neg, multiple by a number or remainder `n`
        (past moves that keep values in place) adds up: [(term, factor)],
        a term a node, a number or a remainder's grid term (_Residue)."""
        n = _in_place_source(n)
        name = _call_name(n)
        alpha = _alpha(n) if name in _SUMS else None
        if name in _SUMS and _is_number(alpha):
            a, b = n.args[:2]
            if name == "aten::add":
                return [(a, 1.0), (b, float(alpha))]
            if name == "aten::sub":
                return [(a, 1.0), (b, -float(alpha))]
            return [(b, 1.0), (a, -float(alpha))]   # rsub(a, b) = b - alpha a
        if name == "aten::neg" and n.args:
            return [(n.args[0], -1.0)]
        factor = self._number_factor(n, name)
        if factor is not None:
            return [factor]
        if name in _RESIDUES and n.args:
            return [(n.args[0], 1.0), (_Residue(n), -1.0)]
        return []

    def _is_linear(self, n):
        """Whether `_sum_terms` reads `n` as terms rather than as one."""
        if not isinstance(n, torch.fx.Node):
            return False
        n = _in_place_source(n)
        name = _call_name(n)
        if name in _SUMS:
            return _is_number(_alpha(n))
        return (name == "aten::neg" and bool(n.args)) or (
            name in _RESIDUES and bool(n.args)) or \
            self._number_factor(n, name) is not None

    @staticmethod
    def _bucket(n):
        """The bucket structural equality compares within: one op (of the
        value `n` holds, _canonical; every reshape alike), one shape."""
        v = _val(n)
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else None
        c = _canonical(n)
        if _is_reshape(c):
            return ("reshape", shape)
        return (c.target if c.op == "call_function" else c.op, shape)

    def _positions(self, node):
        """Each node's index in `node`'s graph (memoised)."""
        if getattr(self, "_position_map", None) is None or \
                node not in self._position_map:
            self._position_map = {n: i for i, n in
                                  enumerate(node.graph.nodes)}
        return self._position_map

    def _number_factor(self, n, name):
        """(the tensor, the factor) when `n` multiplies or divides a tensor
        of its own dtype by a nonzero finite number, else None."""
        if name not in _PRODUCTS or len(n.args) != 2 or \
                n.kwargs.get("rounding_mode") is not None:
            return None
        a, b = n.args
        if _is_number(b) and _is_tensor_node(a) and _dtype(a) == _dtype(n) \
                and b != 0 and math.isfinite(b):
            return (a, float(b)) if name == "aten::mul" else (a, 1.0 / b)
        if name == "aten::mul" and _is_number(a) and _is_tensor_node(b) and \
                _dtype(b) == _dtype(n) and math.isfinite(a):
            return (b, float(a))
        return None

    def _fixed(self, n):
        """Whether the program fixes `n`'s values: the constants alone
        determine it, or it is a factory of fixed values (full_like reads
        only its argument's shape)."""
        if self.sig.is_constant(n, self.constants):
            return True
        base, _ = _walk(n)
        name = _call_name(base)
        return name in _FACTORIES and name not in _RANDOM_FACTORIES

    def _carriers(self, n, name):
        """The operands of an elementwise multiply or divide (or of a
        remainder) that carry the result's elements: by shape
        (_carriers_among), or by identity, when the other operand is the
        step of the grid this one is on (_scaled_by). Only a division's
        dividend can carry."""
        args = list(n.args[:2])
        divides = name == "aten::div" or name in _RESIDUES
        out = self._carriers_among(args, args[:1] if divides else args)
        if len(args) == 2 and all(_is_tensor_node(a) for a in args):
            a, b = args
            if name == "aten::mul":
                if a not in out and self._scaled_by(a, b):
                    out.append(a)
                if b not in out and self._scaled_by(b, a):
                    out.append(b)
            elif name == "aten::div" and a not in out and self._scaled_by(
                    a, b, inverse=True):
                out.append(a)
        return out

    def _carriers_among(self, operands, candidates):
        """Of `candidates`, the `operands` of an elementwise op that carry its
        elements: one that, along some dimension, may differ between two
        indices at which every other operand holds equal values
        (_distinguishes, over the dimensions' partitions, _classes). The
        others then act on it as scales or offsets its elements share -- per
        tensor, per row, per channel, per group, broadcast, expanded,
        repeated or gathered to its shape, a factory of one value, a
        constant whose values are -- and a number is such a scale. Operands
        partitioned alike carry none. What was not read is never plain: a
        partner's partition not read counts as possibly one class (the
        candidate is followed), the candidate's own as varying, and an
        operand of unknown shape makes every candidate a carrier."""
        tensors = [a for a in operands if _is_tensor_node(a)]
        shapes = [_aligned_shape(a, 0) for a in tensors]
        sizes = (None if any(s is None for s in shapes)
                 else _broadcast_shape(shapes))
        parts = {}
        if sizes is not None:
            for a, s in zip(tensors, shapes):
                c = self._classes(a)
                parts[a] = None if c is None else _align(c, s, sizes)
        out = []
        if sizes is None or any(p is None for p in parts.values()):
            for x in candidates:
                if _is_tensor_node(x) and x not in out:
                    out.append(x)
            return out
        for x in candidates:
            if x not in parts or x in out:
                continue
            for d, n in enumerate(sizes):
                cx = _ident(n) if parts[x][d] is None else parts[x][d]
                others = [_unif(n) if parts[y][d] is None else parts[y][d]
                          for y in tensors if y is not x]
                if _distinguishes(cx, others):
                    out.append(x)
                    break
        return out

    # --- shapes -----------------------------------------------------------

    def _classes(self, node):
        """The partition of each of `node`'s own dimensions (see _ident and
        the helpers after it), or None for an unknown shape (symbolic, or
        not read within _VARIES_BUDGET nodes). Read from a constant's own
        values; from the partitions of what an op takes its elements from --
        through expansions and repeats (the copies share a class), reshapes
        (a run of dimensions read as one row-major index), permutes,
        transposes, selects, slices, flips, rolls, gathers by indices the
        program fixes (their classes), concatenations, and any elementwise
        op, whose result shares a class only where every input does; a
        factory of one value is one class. A graph input, a state the
        forward writes and any other runtime value vary (the identity). A
        constant whose values the gate cannot read, and a computed constant
        no rule reads, are not read along a dimension of more than one index
        (None). Each node is read once, inputs first, without recursion
        (memoised): a chain of any depth is read to its end."""
        if not isinstance(node, torch.fx.Node):
            return None
        stack = [node]
        while stack:
            n = stack[-1]
            if n in self.parts:
                stack.pop()
                continue
            if self.parts_read >= _VARIES_BUDGET:
                self.parts[n] = None            # not read: unknown
                stack.pop()
                continue
            missing = [a for a in self._class_sources(n)
                       if a not in self.parts]
            if missing:
                stack += missing
                continue
            self.parts_read += 1
            self.parts[n] = self._read_classes(n)
            stack.pop()
        return self.parts[node]

    def _class_sources(self, node):
        """The inputs whose partitions `node`'s rule reads (_derived_classes)."""
        if node.op != "call_function" or _aligned_shape(node, 0) is None:
            return []
        name = _call_name(node)
        if name in _UNIFORM_FACTORIES or (name not in _ELEMENTWISE
                                          and name not in _PARTITION_MOVES):
            return []
        if name in _ELEMENTWISE:
            return [a for a in node.all_input_nodes if _is_tensor_node(a)]
        if name == "aten::cat":
            return [t for t in (node.args[0] if node.args else ())
                    if _is_tensor_node(t)]
        src = node.args[0] if node.args else None
        if name == "aten::gather":              # the indices' too
            return [a for a in (src, node.args[2] if len(node.args) > 2
                                else None) if _is_tensor_node(a)]
        return [src] if _is_tensor_node(src) else []

    def _read_classes(self, node):
        """`node`'s partitions, its inputs' already read (_classes)."""
        shape = _aligned_shape(node, 0)
        if shape is None:
            return None
        if node.op == "get_attr" or self.sig.constant_fqn(node) is not None:
            got = self._payload_classes(node, shape)
        elif node.op == "call_function":
            got = self._derived_classes(node, shape)
            if got is _UNREAD or (got is not None and len(got) != len(shape)):
                return None
            if got is None:
                got = ([None] * len(shape)
                       if self.sig.is_constant(node, self.constants)
                       else [_ident(s) for s in shape])
        else:
            got = [_ident(s) for s in shape]       # a runtime input, a state
        return [_unif(s) if s <= 1 else c for s, c in zip(shape, got)]

    def _payload_classes(self, node, shape):
        """A constant's partitions, from its values (_payload_dim); None along
        each dimension of more than one index when they cannot be read."""
        t = self._payload(node)
        if t is None or list(t.shape) != shape:
            return [None] * len(shape)
        return [_payload_dim(t, d) if s > 1 else _unif(s)
                for d, s in enumerate(shape)]

    def _derived_classes(self, node, shape):
        """`node`'s partitions read from its inputs' (already read, in
        self.parts), None when no rule reads this op, or _UNREAD when an
        input's shape is unknown or a move's rule cannot read it (tensor
        repeats, several index tensors, sizes that do not factor): the
        node's partitions are then unknown, never "varies" (the carrier
        test follows every operand beside it)."""
        name = _call_name(node)
        if name in _UNIFORM_FACTORIES:
            return [_unif(s) for s in shape]
        if name not in _ELEMENTWISE and name not in _PARTITION_MOVES:
            return None
        if name in _ELEMENTWISE:
            per_dim = [[] for _ in shape]
            for a in self._class_sources(node):
                ca = self.parts.get(a)
                if ca is None:
                    return _UNREAD
                aligned = _align(ca, _aligned_shape(a, 0), shape)
                if aligned is None:
                    return None
                for d, c in enumerate(aligned):
                    per_dim[d].append(c)
            return [_meet_known(p, n) for p, n in zip(per_dim, shape)]
        try:
            if name == "aten::cat":
                return self._cat_classes(node, shape)
            src = node.args[0] if node.args else None
            if not _is_tensor_node(src):
                return _UNREAD
            inner = self.parts.get(src)
            if inner is None:
                return _UNREAD
            return self._move_classes(node, name, inner,
                                      _aligned_shape(src, 0), shape)
        except (IndexError, TypeError, ValueError, KeyError,
                ZeroDivisionError):
            return _UNREAD

    def _move_classes(self, node, name, inner, src_shape, shape):
        """The partitions of a move of a value with partitions `inner`, or
        _UNREAD where the rule cannot read the move."""
        args, kwargs, r = node.args, node.kwargs, len(inner)

        def arg(i, key, default=None):
            return args[i] if len(args) > i else kwargs.get(key, default)

        if name in ("aten::expand_copy", "aten::expand"):
            return _align(inner, src_shape, shape)
        if name == "aten::repeat":
            reps = list(args[1])
            lead = len(reps) - r
            return [_unif(k) for k in reps[:lead]] + [
                None if c is None else np.tile(c, k)
                for c, k in zip(inner, reps[lead:])]
        if name == "aten::repeat_interleave":
            reps, dim = arg(1, "repeats"), arg(2, "dim")
            if not isinstance(reps, int) or dim is None:
                return _UNREAD
            out = list(inner)
            d = dim % r
            out[d] = None if inner[d] is None else np.repeat(inner[d], reps)
            return out
        if name in _RESHAPES:
            got = _reshape_classes(src_shape, inner, shape)
            return _UNREAD if got is None else got
        if name in ("aten::permute_copy", "aten::permute"):
            return [inner[d % r] for d in args[1]]
        if name in ("aten::transpose_copy", "aten::transpose"):
            out = list(inner)
            d0, d1 = args[1] % r, args[2] % r
            out[d0], out[d1] = out[d1], out[d0]
            return out
        if name in ("aten::t_copy", "aten::t"):
            return list(reversed(inner))
        if name in ("aten::select_copy", "aten::select"):
            d = args[1] % r
            return inner[:d] + inner[d + 1:]
        if name in ("aten::slice_copy", "aten::slice"):
            d = arg(1, "dim", 0) % r
            idx = np.arange(src_shape[d])[slice(arg(2, "start"), arg(3, "end"),
                                                arg(4, "step", 1))]
            return self._gathered(inner, d, idx, shape[d], True)
        if name in ("aten::narrow_copy", "aten::narrow"):
            d = args[1] % r
            start = args[2] % src_shape[d] if src_shape[d] else 0
            return self._gathered(inner, d, np.arange(start, start + args[3]),
                                  shape[d], True)
        if name == "aten::flip":
            out = list(inner)
            for d in args[1]:
                d %= r
                out[d] = None if out[d] is None else out[d][::-1].copy()
            return out
        if name == "aten::roll":
            shifts, dims = args[1], arg(2, "dims", [])
            shifts = list(shifts) if isinstance(shifts, (list, tuple)) \
                else [shifts]
            dims = list(dims) if isinstance(dims, (list, tuple)) else [dims]
            if not dims or len(dims) != len(shifts):
                return _UNREAD
            out = list(inner)
            for s, d in zip(shifts, dims):
                d %= r
                out[d] = None if out[d] is None else np.roll(out[d], s)
            return out
        if name == "aten::index_select":
            d = args[1] % r
            idx, fixed = self._index_values(args[2])
            return self._gathered(inner, d, idx, shape[d], fixed)
        if name in ("aten::index", "aten::_unsafe_index"):
            given = [(k, i) for k, i in enumerate(args[1]) if i is not None]
            if len(given) != 1:
                return _UNREAD              # several index tensors
            d, i = given[0]
            ishape = _aligned_shape(i, 0)
            if ishape is None or _dtype(i) not in (torch.int64, torch.int32):
                return _UNREAD
            idx, fixed = self._index_values(i)
            if len(ishape) == 1:
                return self._gathered(inner, d, idx, shape[d], fixed)
            # One index of several dimensions: they take dimension d's
            # place, each partitioned as the classes of the indices along it.
            c = inner[d]
            if c is None:
                parts = [None] * len(ishape)
            elif idx is not None:
                parts = _split(_relabel(c[idx]), ishape)
            elif _is_unif(c):
                parts = [_unif(s) for s in ishape]
            else:
                parts = [None if fixed else _ident(s) for s in ishape]
            return list(inner[:d]) + parts + list(inner[d + 1:])
        if name == "aten::gather":
            return self._gather_classes(args, inner, shape, r)
        return _UNREAD

    def _gather_classes(self, args, inner, shape, r):
        """The partitions of gather(input, d, index), the input's being
        `inner`: along d, the classes of the indices (_gathered); along each
        other dimension, the input's where it holds one value along d (any
        index reads the same class), else their meet with the index tensor's
        own partition there -- two positions read equal elements only where
        their indices are equal as well -- the identity for a runtime index
        whose partition is not read, not read for a constant one."""
        d = args[1] % r
        index = args[2]
        vals, fixed = self._index_values(index, flat=False)
        if vals is not None:
            rows = np.moveaxis(vals, d, 0).reshape(vals.shape[d], -1)
            if rows.shape[1] and not (rows == rows[:, :1]).all():
                vals = None                 # not one index per position
            else:
                vals = rows[:, 0]
        out = self._gathered(inner, d, vals, shape[d], fixed)
        uniform = inner[d] is not None and _is_unif(inner[d])
        own = self.parts.get(index) if _is_tensor_node(index) else None
        result = []
        for e, c in enumerate(out):
            if e == d:
                result.append(c)
                continue
            c = None if c is None else c[:shape[e]]
            if uniform:
                result.append(c)
                continue
            ie = own[e] if own is not None and len(own) == len(out) else None
            if ie is None:
                ie = None if fixed else _ident(shape[e])
            result.append(_meet_known([c, ie], shape[e]))
        return result

    def _gathered(self, inner, d, idx, n, fixed):
        """`inner` with dimension d's partition taken at indices `idx` (the
        classes of the indices); for indices not read, one class if the
        argument is one along d, else the identity for runtime indices and
        not read for indices the program fixes (`fixed`)."""
        out = list(inner)
        c = inner[d]
        if c is None:
            out[d] = None
        elif idx is not None:
            out[d] = c[idx]
        elif _is_unif(c):
            out[d] = _unif(n)
        else:
            out[d] = None if fixed else _ident(n)
        return out

    def _index_values(self, i, flat=True):
        """(the values of index tensor `i` as an int64 array, flattened
        unless `flat` is False, or None when the gate cannot read them;
        whether the program's constants alone determine them)."""
        if not _is_tensor_node(i):
            return None, False
        fixed = self.sig.is_constant(i, self.constants)
        base, path = walk(i)
        if not fixed or not isinstance(base, torch.fx.Node) or \
                base.op not in ("placeholder", "get_attr") or any(
                    _call_name(p) not in _IN_PLACE | _CASTS | _RESHAPES
                    for p in path):
            return None, fixed
        t = self._payload(base)
        if t is None or t.is_floating_point():
            return None, fixed
        vals = t.reshape(_aligned_shape(i, 0)).numpy().astype(np.int64)
        return (vals.reshape(-1) if flat else vals), fixed

    def _cat_classes(self, node, shape):
        """A concatenation's partitions: along the joined dimension each
        input's classes, an input that is one value with an earlier one
        (_same_value: the same node, a stack's unsqueezes of one value,
        constants holding the same bits) taking that one's labels -- a step
        stacked or concatenated to a group's or a row's width is one class
        across its copies -- and the rest apart; along the others, their
        meet."""
        tensors = list(node.args[0])
        d = (node.args[1] if len(node.args) > 1
             else node.kwargs.get("dim", 0)) % len(shape)
        joined, others, offset, firsts = [], [[] for _ in shape], 0, {}
        for t in tensors:
            c = self.parts.get(t) if _is_tensor_node(t) else None
            if c is None or len(c) != len(shape):
                return _UNREAD
            if joined is not None:
                if c[d] is None:
                    joined = None
                else:
                    bucket = firsts.setdefault(self._bucket(t), [])
                    labels = next((lab for u, lab in bucket
                                   if self._same_value(u, t)), None)
                    if labels is None:
                        labels = c[d] + offset
                        offset += int(c[d].max(initial=-1)) + 1
                        bucket.append((t, labels))
                    joined.append(labels)
            for e in range(len(shape)):
                if e != d:
                    others[e].append(c[e])
        out = [_meet_known(p, n) for p, n in zip(others, shape)]
        out[d] = None if joined is None else np.concatenate(joined)
        return out

    def _payload(self, node):
        """The tensor a constant placeholder (a parameter, a lifted
        constant, a buffer nothing writes) or a module attribute holds, as
        the program will at run time, else None. While partitioning,
        ExecuTorch hands the gate FakeTensors for parameters and buffers:
        the real tensor behind one, when it can be found
        (const_weights._concrete), else None -- and then nothing is known of
        its values: its partitions are not read (_payload_classes), and an
        equality that needs them is not decided (_same_constant)."""
        if node in self.payloads:
            return self.payloads[node]
        from torch._subclasses.fake_tensor import FakeTensor

        from executorch.backends.esp32s3.operator_support.const_weights import (
            _concrete,
        )

        got = None
        if self.ep is not None and node.op == "placeholder":
            fqn = self.sig.constant_fqn(node)
            if fqn is not None:
                got = (self.ep.state_dict.get(fqn) if fqn in
                       self.ep.state_dict else self.ep.constants.get(fqn))
        elif self.ep is not None and node.op == "get_attr":
            got = self.ep.graph_module
            for part in str(node.target).split("."):
                got = getattr(got, part, None)
        got = _concrete(got) if isinstance(got, torch.Tensor) else None
        if not isinstance(got, torch.Tensor) or isinstance(got, FakeTensor):
            got = None
        else:
            got = got.detach()
        self.payloads[node] = got
        return got

    # --- identities -------------------------------------------------------

    def _same_value(self, a, b):
        """Whether `a` and `b` are one value, element for element, by
        structure -- the equivalence value numbering computes, decided
        lazily from the top. One value: one node past aliases and past
        reshapes back to its own shape (_canonical), two reshapes of one
        value to one shape, constants holding the same bits, one
        deterministic op over arguments that are one value (in either order
        for an op that commutes). Never one value: a random op, an
        uninitialised factory or an op that writes and anything but itself,
        two graph inputs, two reads of a state, a flip, permute, slice or
        gather and what it moves. Exact and without a depth bound: each pair
        is compared once (memoised), without recursion. An equality not
        decided refuses: _Exhausted past _SAME_BUDGET pairs, and when two
        constants it must compare cannot be read (_same_constant)."""
        quick = self._same_quick(a, b)
        if quick is not None:
            return quick
        stack, sent = [(a, b, self._same_frame(a, b))], None
        while True:
            x, y, frame = stack[-1]
            try:
                sub = frame.send(sent)
            except StopIteration as stop:
                result = bool(stop.value)
                stack.pop()
                if isinstance(x, torch.fx.Node):
                    x, y = _canonical(x), _canonical(y)
                    self.same[(x, y)] = self.same[(y, x)] = result
                if not stack:
                    return result
                sent = result
                continue
            sent = self._same_quick(*sub)
            if sent is None:
                self.same_pairs += 1
                if self.same_pairs > _SAME_BUDGET:
                    raise _Exhausted("_SAME_BUDGET", f"more than "
                                     f"{_SAME_BUDGET} pairs of values "
                                     f"compared")
                stack.append((sub[0], sub[1], self._same_frame(*sub)))

    def _same_quick(self, a, b):
        """_same_value's answer when no argument needs comparing (or it is
        memoised): True, False, or None to compare their arguments."""
        if a is b:
            return True
        if isinstance(a, torch.fx.Node) != isinstance(b, torch.fx.Node):
            return False
        if not isinstance(a, torch.fx.Node):
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return False if len(a) != len(b) else (None if a else True)
            return _same_literal(a, b)
        a, b = _canonical(a), _canonical(b)
        if a is b:
            return True
        got = self.same.get((a, b))
        if got is not None:
            return got
        if not _same_meta(a, b):
            return False
        if a.op != "call_function" or b.op != "call_function":
            fixed = [n.op == "get_attr" or self.sig.constant_fqn(n) is not None
                     for n in (a, b)]
            return self._same_constant(a, b) if all(fixed) else False
        if _is_reshape(a) != _is_reshape(b):
            return False
        if not _is_reshape(a) and a.target != b.target:
            return False
        if not (_deterministic(a) and _deterministic(b)):
            return False
        return None

    def _same_frame(self, a, b):
        """A generator comparing the arguments of `a` and `b` for
        _same_value: it yields pairs and is sent whether each is one value,
        and returns its answer."""
        if not isinstance(a, torch.fx.Node):        # sequences of one length
            for x, y in zip(a, b):
                if not (yield (x, y)):
                    return False
            return True
        a, b = _canonical(a), _canonical(b)
        if _is_reshape(a):                          # one shape (_same_meta)
            return (yield (_reshape_root(a), _reshape_root(b)))
        if a.target is operator.getitem:
            if not _same_literal(a.args[1], b.args[1]):
                return False
            return (yield (a.args[0], b.args[0]))
        if len(a.args) != len(b.args) or set(a.kwargs) != set(b.kwargs):
            return False
        for k in sorted(a.kwargs):
            if not (yield (a.kwargs[k], b.kwargs[k])):
                return False
        same = True
        for x, y in zip(a.args, b.args):
            if not (yield (x, y)):
                same = False
                break
        if same:
            return True
        name = _call_name(a)
        if len(a.args) == 2 and (name in _COMMUTATIVE or (
                name == "aten::add" and _alpha(a) == 1)):
            if not (yield (a.args[0], b.args[1])):
                return False
            return (yield (a.args[1], b.args[0]))
        return False

    def _same_constant(self, a, b):
        """Whether two constants (_same_meta already) hold the same bits;
        _Exhausted when either cannot be read: their equality is not
        decided."""
        pa, pb = self._payload(a), self._payload(b)
        if pa is None or pb is None:
            raise _Exhausted("_payload", f"constants {a.name} and {b.name} "
                             f"compared, one of whose values the gate "
                             f"cannot read")
        if pa.dtype != pb.dtype or pa.shape != pb.shape:
            return False
        if pa.is_floating_point():
            bits = {2: torch.int16, 4: torch.int32,
                    8: torch.int64}[pa.element_size()]
            pa, pb = pa.contiguous().view(bits), pb.contiguous().view(bits)
        return bool(torch.equal(pa, pb))

    def _aligned(self, v):
        """`v` past what keeps each of its elements in its place, up to one
        factor: aliases and reshapes back to its own shape (_canonical), and
        a multiply or divide by one value (a number, a one-element tensor).
        Not past a flip, permute, slice, gather or broadcast: the identities
        compare element-aligned values."""
        for _ in range(_BUDGET):
            v = _canonical(v)
            scaled = _by_one_value(v) if isinstance(v, torch.fx.Node) else None
            if scaled is None or not _is_tensor_node(scaled) or \
                    _dtype(scaled) != _dtype(v):
                return v
            v = scaled
        raise _Exhausted("_BUDGET", "factors of one value that do not end")

    def _small(self, e):
        """Whether `e` is a guard's epsilon: a number, or a one-element
        constant, in (0, _GUARD], a constant's value read through the moves
        and the factors of one value that compute it (_scalar_value: a
        buffer of 1e-8 times 1e8 is 1.0, no guard). A constant whose value
        the gate cannot read counts (the direction that refuses)."""
        if _is_number(e):
            return 0 < e <= _GUARD
        if not _is_tensor_node(e) or _val(e).numel() != 1 or \
                not self.sig.is_constant(e, self.constants):
            return False
        value = self._scalar_value(e)
        if value is None:
            return True
        return 0 < value <= _GUARD

    def _scalar_value(self, e):
        """The value one-element constant `e` holds, as a float: a constant's
        payload or a factory's number, read through moves and float casts,
        and the product or quotient of two such values or numbers (the edge
        graph lifts eps0 * 1e8's 1e8 into a one-element constant); None
        when it cannot be read. Raises _Exhausted past _BUDGET nodes."""
        seen = [0]

        def value(v):
            while True:
                seen[0] += 1
                if seen[0] > _BUDGET:
                    raise _Exhausted("_BUDGET", f"the value of {e.name}, past "
                                     f"{_BUDGET} nodes")
                if _is_number(v):
                    return float(v)
                if not _is_tensor_node(v) or _val(v).numel() != 1:
                    return None
                name = _call_name(v)
                if name in ("aten::mul", "aten::div") and len(v.args) == 2 \
                        and v.kwargs.get("rounding_mode") is None:
                    a, b = value(v.args[0]), value(v.args[1])
                    if a is None or b is None or (name == "aten::div"
                                                  and b == 0):
                        return None
                    return a * b if name == "aten::mul" else a / b
                if v.op in ("placeholder", "get_attr"):
                    t = self._payload(v)
                    return None if t is None else float(t.reshape(-1)[0])
                number = None
                if name in ("aten::full", "aten::full_like") and \
                        len(v.args) > 1:
                    number = v.args[1]
                elif name == "aten::scalar_tensor" and v.args:
                    number = v.args[0]
                elif name in ("aten::ones", "aten::ones_like"):
                    number = 1.0
                elif name in ("aten::zeros", "aten::zeros_like"):
                    number = 0.0
                if name is not None and number is not None:
                    return float(number) if _is_number(number) else None
                v = _move_source(v)
                if v is None:
                    return None

        return value(e)

    def _guarded(self, v):
        """t when `v` guards t against zero, else None: max(t, eps) -- a
        clamp with a lower bound in (0, _GUARD] and no upper one, or
        maximum(t, eps) -- or t + eps, eps in (0, _GUARD] (_small). A larger
        bound is computation: x / max(abs(x), 1) is clamp(x, -1, 1)."""
        name = _call_name(v)
        if name in ("aten::clamp", "aten::clip", "aten::clamp_min") and \
                v.args and _is_tensor_node(v.args[0]):
            lo = v.args[1] if len(v.args) > 1 else v.kwargs.get("min")
            hi = None if name == "aten::clamp_min" else (
                v.args[2] if len(v.args) > 2 else v.kwargs.get("max"))
            if hi is None and lo is not None and self._small(lo):
                return v.args[0]
            return None
        if (name == "aten::maximum" or (name == "aten::add"
                                        and _alpha(v) == 1)) and \
                len(v.args) == 2:
            a, b = v.args
            if _is_tensor_node(a) and _dtype(a) == _dtype(v) and \
                    self._small(b):
                return a
            if _is_tensor_node(b) and _dtype(b) == _dtype(v) and \
                    self._small(a):
                return b
        return None

    def _guarded_base(self, v):
        """`v` past aliases, factors of one value (_aligned) and guards
        against zero (_guarded)."""
        for _ in range(_BUDGET):
            v = self._aligned(v)
            g = self._guarded(v)
            if g is None:
                return v
            v = g
        raise _Exhausted("_BUDGET", "guards against zero that do not end")

    def _step_base(self, v):
        """`v` past aliases, factors of one value, any clamp and guards
        against zero (_scaled_by): round(.) * s is on s's grid whatever it
        divides by, so a step clamped or guarded where it divides is the
        step."""
        for _ in range(_BUDGET):
            v = self._aligned(v)
            if _call_name(v) in _CLAMPS and v.args and \
                    _is_tensor_node(v.args[0]):
                v = v.args[0]
                continue
            g = self._guarded(v)
            if g is None:
                return v
            v = g
        raise _Exhausted("_BUDGET", "clamps of a step that do not end")

    @staticmethod
    def _reciprocal_arg(b):
        """t when `b` is 1 / t: reciprocal(t), a number or one value divided
        by t, t ** -1; else None."""
        name = _call_name(b)
        if name == "aten::reciprocal" and b.args:
            return b.args[0]
        if name == "aten::div" and len(b.args) == 2 and \
                b.kwargs.get("rounding_mode") is None:
            num, den = b.args
            nv = _val(num)
            if _is_tensor_node(den) and (_is_number(num) or (
                    isinstance(nv, torch.Tensor) and nv.numel() == 1)):
                return den
        if name == "aten::pow" and len(b.args) == 2 and \
                _is_number(b.args[1]) and b.args[1] == -1:
            return b.args[0]
        return None

    def _reciprocal_of(self, v, base):
        """t when `v` is, up to what `base` steps through, 1 / t
        (_reciprocal_arg; t past `base` too); else None."""
        t = self._reciprocal_arg(base(v))
        return None if t is None else base(t)

    def _peeled(self, v, base=None):
        """(the value `v` places, the positional moves that place it,
        outermost first): `v` past what `base` steps through (_aligned:
        aliases, reshapes back and factors of one value) and past each
        positional move (_POSITIONAL), `base` again under each. A move is
        stripped only where the other side of an identity has the same one
        (_same_moves): a flip of t is not t."""
        base = base or self._aligned
        moves = []
        for _ in range(_BUDGET):
            v = base(v)
            if not (_call_name(v) in _POSITIONAL and v.args
                    and _is_tensor_node(v.args[0])
                    and _dtype(v.args[0]) == _dtype(v)):
                return v, tuple(moves)
            moves.append(v)
            v = v.args[0]
        raise _Exhausted("_BUDGET", "positional moves that do not end")

    def _one_value(self, a, b):
        """Whether (value, moves) pairs `a` and `b` (_peeled) are one value
        element for element: the same moves over one value."""
        return _same_moves(a[1], b[1]) and self._same_value(a[0], b[0])

    def _abs_of(self, v, inverse=False):
        """(t, moves) when `v` is t's magnitude, its elements placed by the
        positional moves `moves` (outermost first), up to aliases, factors of
        one value, guards against zero (_guarded_base) and positional moves
        outside and inside it: abs(t), sqrt(t * t), sqrt(t ** 2) or
        (t * t) ** 0.5, guarded inside the root too (sqrt(t * t + eps)); with
        `inverse`, its reciprocal rsqrt(t * t) or (t * t) ** -0.5. A move
        commutes with these, one element at a time, so flip(abs(t)) is
        abs(t) placed by a flip. Else None."""
        b, outer = self._peeled(v, self._guarded_base)
        name = _call_name(b)
        if name == "aten::abs" and b.args and not inverse:
            t, inner = self._peeled(b.args[0])
            return t, outer + inner
        root = name == ("aten::rsqrt" if inverse else "aten::sqrt")
        if name == "aten::pow" and len(b.args) == 2 and \
                _is_number(b.args[1]):
            root = _number_key(b.args[1]) == _number_key(
                -0.5 if inverse else 0.5)
        if root and b.args:
            got = self._squared(b.args[0])
            if got is not None:
                return got[0], outer + got[1]
        return None

    def _squared(self, v):
        """(t, moves) when `v` is, up to a guard and positional moves
        (_peeled), t * t (one value twice, _same_value) or t ** 2, else
        None."""
        b, outer = self._peeled(v, self._guarded_base)
        name = _call_name(b)
        if (name == "aten::mul" and len(b.args) == 2
                and self._same_value(b.args[0], b.args[1])) or (
                name == "aten::pow" and len(b.args) == 2
                and _is_number(b.args[1]) and b.args[1] == 2):
            t, inner = self._peeled(b.args[0])
            return t, outer + inner
        return None

    def _sign_identity(self, n, name):
        """Whether the multiply or divide `n` is sign spelled by hand: x
        over its magnitude (abs(x), sqrt(x * x), (x * x) ** 0.5) or the
        magnitude over x, x * (1 / abs(x)), abs(x) * (1 / x),
        x * rsqrt(x * x), up to aliases, factors of one value and guards
        against zero, the two x one value element for element: the same
        value placed by the same positional moves (_peeled, _same_moves:
        flip(x) / flip(abs(x)) is sign, x / abs(x.flip(-1)) is not)."""
        if len(n.args) != 2 or not all(_is_tensor_node(a) for a in n.args):
            return False
        a, b = n.args
        if name == "aten::div":
            pairs = [(a, (), b)]
        else:
            pairs = []
            for x, y in ((a, b), (b, a)):
                inv = self._abs_of(y, inverse=True)
                if inv is not None and self._one_value(self._peeled(x), inv):
                    return True         # x * rsqrt(x * x)
                r, outer = self._peeled(y, self._guarded_base)
                den = self._reciprocal_arg(r)
                if den is not None:
                    pairs.append((x, outer, self._guarded_base(den)))
        for x, outer, y in pairs:       # x / y, y placed by `outer`
            ay = self._abs_of(y)
            if ay is not None and self._one_value(self._peeled(x),
                                                  (ay[0], outer + ay[1])):
                return True
            ax = self._abs_of(x)
            if ax is not None:
                t, moves = self._peeled(y)
                if self._one_value(ax, (t, outer + moves)):
                    return True
        return False

    def _scaled_by(self, x, s, inverse=False):
        """Whether `s` is, by identity, the step of the grid `x` is on: `x`
        reaches a rounding whose argument is divided by s or multiplied by
        1 / s -- round(x / s) * s -- or, with `inverse` (x divided by s),
        multiplied by s or divided by 1 / s -- round(x * s) / s. s and the
        divisor are one value (_same_value) past aliases, factors of one
        value, clamps and guards (_step_base): a step written twice, guarded
        against zero where it divides."""
        sb = self._step_base(s)
        factors, complete = self._rounding_factors(x)
        if not complete:
            return True                 # not read to the end: followed
        rs = self._reciprocal_of(s, self._step_base)
        for kind, f in factors:
            fb = self._step_base(f)
            fr = self._reciprocal_of(f, self._step_base)
            if (kind == "mul") == inverse:          # s is f, up to a factor
                if self._same_value(fb, sb) or (
                        rs is not None and fr is not None
                        and self._same_value(rs, fr)):
                    return True
            elif (rs is not None and self._same_value(rs, fb)) or (
                    fr is not None and self._same_value(fr, sb)):
                return True                         # s is 1 / f
        return False

    def _rounding_factors(self, x):
        """For each rounding `x`'s values come from -- through moves, keeps,
        assemblers' sources, scales and offsets by shape and the terms of a
        sum that remain -- what its argument is divided ("div") or
        multiplied ("mul") by (_argument_factors); a division that rounds,
        x // s, is divided by s. ([(kind, factor)], whether the search
        finished within _FACTOR_BUDGET nodes; memoised.)"""
        if x in self.factors:
            return self.factors[x]
        out, stack, seen, complete = [], [x], set(), True
        while stack:
            n = stack.pop()
            if not isinstance(n, torch.fx.Node) or n in seen or \
                    n.op != "call_function":
                continue
            seen.add(n)
            if len(seen) > _FACTOR_BUDGET:
                complete = False
                break
            if n.target is operator.getitem:
                src = n.args[0]
                if _call_name(src) in _SPLITS or (
                        _call_name(src) in _SELECTIONS and n.args[1] == 0):
                    stack.append(src.args[0] if src.args else None)
                continue
            name = _call_name(n)
            if name in _ROUNDING:
                got = self._argument_factors(n.args[0] if n.args else None)
                if got is None:
                    complete = False
                    break
                out += got
            elif _rounding_division(n, name):
                if len(n.args) > 1 and _is_tensor_node(n.args[1]):
                    out.append(("div", n.args[1]))
            elif name in _MOVES or name in _CASTS or name in _KEEPS:
                stack.append(n.args[0] if n.args else None)
            elif name in _ASSEMBLERS:
                stack += _ASSEMBLERS[name](n)
            elif name in _SUMS or name in _RESIDUES:
                got = self._sum_terms(n)
                if got is not None:
                    stack += [t for c, t, synthetic in got.terms
                              if c != 0 and not synthetic]
            elif name in _PRODUCTS:
                args = list(n.args[:2])
                stack += self._carriers_among(
                    args, args[:1] if name == "aten::div" else args)
        self.factors[x] = (out, complete)
        return self.factors[x]

    def _argument_factors(self, v):
        """What a rounding's argument `v` divides or multiplies, through
        moves, keeps, multiples by numbers and offsets by numbers
        (round(clamp(x / s) + z), round(x / s * 4)): [("div", s)] for x / s,
        [("mul", x), ("mul", r)] for x * r; None past _FACTOR_DEPTH moves."""
        for _ in range(_FACTOR_DEPTH):
            if not isinstance(v, torch.fx.Node) or v.op != "call_function":
                return []
            if v.target is operator.getitem:
                src = v.args[0]
                if _call_name(src) not in _SPLITS or not src.args:
                    return []
                v = src.args[0]
                continue
            name = _call_name(v)
            if name in _MOVES or name in _CASTS or name in _KEEPS:
                v = v.args[0] if v.args else None
                continue
            factor = self._number_factor(v, name)
            if factor is not None:          # a step times a number
                v = factor[0]
                continue
            if name in _SUMS:
                got = self._sum_terms(v)
                kept = [t for c, t, synthetic in (got.terms if got else ())
                        if c != 0 and not synthetic]
                if got is None or len(kept) != 1:
                    return []
                v = kept[0]
                continue
            if name == "aten::div" and len(v.args) == 2 and \
                    v.kwargs.get("rounding_mode") is None:
                return ([("div", v.args[1])] if _is_tensor_node(v.args[1])
                        else [])
            if name == "aten::mul" and len(v.args) == 2:
                return [("mul", a) for a in v.args if _is_tensor_node(a)]
            return []
        return None

    def _index_codes(self, n, name):
        """The op that makes a gather's indices the model's own codes, else
        None. The gather (embedding, index, index_select, gather, take)
        reads a table of levels -- a constant, or a constant times scales
        and plus offsets that hold one value along the gathered dimensions
        (_levels_table: NF4 levels times a run-time absmax) -- at indices
        that depend on a float value through an op mapping floats to
        integers -- a cast to an integer type or to bool, a comparison,
        argmin or argmax (max's and min's indices too), bucketize,
        searchsorted: any op with a floating tensor input and an integer or
        boolean result -- whatever integer arithmetic follows, with numbers
        or tensors (codes plus a zero point, a sum of comparisons, a where of
        numbers): the table's values at the model's own codes, a
        lookup-table quantiser. Every dependency of each index is followed
        (_float_source): graph inputs, constants, factories and integer ops
        over them -- token ids, positions and their offsets -- are not
        codes, and a run-time table (a value's own entries at the top k of
        its keys) is no table of levels."""
        if name not in _GATHERS or self.ep is None:
            return None
        table, indices = _GATHERS[name](n)
        if not isinstance(table, torch.fx.Node):
            return None
        if not self.sig.is_constant(table, self.constants) and \
                not self._levels_table(table, _gathered_dims(n, name, table)):
            return None
        for v in indices:
            got = self._float_source(v)
            if got is not None:
                return got
        return None

    def _levels_table(self, table, dims):
        """Whether gathered `table` holds, along its gathered dimensions
        `dims` (None: not known), a constant's values up to scales and
        offsets that hold one value along them: elementwise arithmetic -- a
        multiply, a divide, a sum with a number for alpha -- each of whose
        operands is a number, a constant (levels), a value one class along
        those dimensions (_classes; one whose partition is not read counts:
        possibly a scale), or such a table in turn, with a constant among
        them: the NF4 levels times a run-time absmax, per tensor or per row.
        Not a run-time table: a value's own entries. Raises _Exhausted past
        _BUDGET nodes."""
        stack, seen, found = [(table, dims)], 0, False
        while stack:
            t, ds = stack.pop()
            seen += 1
            if seen > _BUDGET:
                raise _Exhausted("_BUDGET", f"the arithmetic of lookup table "
                                 f"{table.name}, past {_BUDGET} nodes")
            if self.sig.is_constant(t, self.constants):
                found = True
                continue
            t = _in_place_source(t)
            name = _call_name(t)
            if not ((name in _PRODUCTS
                     and t.kwargs.get("rounding_mode") is None)
                    or (name in _SUMS and _is_number(_alpha(t)))) or \
                    len(t.args) < 2:
                return False
            shape = _aligned_shape(t, 0)
            for a in t.args[:2]:
                if _is_number(a):
                    continue
                if not _is_tensor_node(a):
                    return False
                if self.sig.is_constant(a, self.constants):
                    found = True
                    continue
                own = _aligned_shape(a, 0)
                if shape is None or own is None or ds is None:
                    continue                # possibly a scale
                lead = len(shape) - len(own)
                ads = tuple(d - lead for d in ds
                            if d - lead >= 0 and own[d - lead] != 1)
                c = self._classes(a)
                if not ads or c is None or all(
                        c[d] is None or _is_unif(c[d]) for d in ads):
                    continue                # a scale or an offset along them
                stack.append((a, ads))
        return found

    def _float_source(self, start):
        """The first op found that maps a float value to the integers or
        booleans index `start` depends on, else None, every dependency
        followed: a buffer the forward writes into what it writes, a factory
        no further (it reads its argument's shape at most) but for the value
        fill.Tensor reads (an integer fill of a float converts it). Memoised.
        Raises _Exhausted past _INDEX_BUDGET nodes: an index not read to its
        sources is not proven free of codes."""
        if not isinstance(start, torch.fx.Node):
            return None
        if start in self.indices:
            return self.indices[start]
        stack, seen, expanded, found = [start], set(), set(), None
        while stack:
            u = stack.pop()
            if not isinstance(u, torch.fx.Node) or u in seen or (
                    u in self.indices and self.indices[u] is None):
                continue
            seen.add(u)
            if len(seen) > _INDEX_BUDGET:
                raise _Exhausted("_INDEX_BUDGET", f"the values index "
                                 f"{start.name} depends on, past "
                                 f"{_INDEX_BUDGET} nodes")
            if not isinstance(_val(u), torch.Tensor):
                continue                    # a size, not values
            if u.op == "placeholder":
                fqn = self.sig.written_fqn(u)
                if fqn is not None and fqn not in expanded:
                    expanded.add(fqn)
                    got = self._written_sources(u, fqn)
                    if isinstance(got, Origin):
                        raise _Exhausted("_BUDGET", got.detail)
                    stack += got
                continue                    # a graph input, a constant
            if u.op == "get_attr":
                continue
            if u.op != "call_function":
                found = u
                break
            if u.target is operator.getitem:
                src = u.args[0]
                inputs = (src.all_input_nodes if isinstance(src, torch.fx.Node)
                          else [])
            elif _call_name(u) in _FACTORIES:
                # A factory reads its argument's shape at most, but
                # fill.Tensor reads a value too, and casts it to its own
                # dtype: an integer fill of a float is a conversion.
                value = u.args[1] if _call_name(u) == "aten::fill" and \
                    len(u.args) > 1 else None
                if not _is_tensor_node(value):
                    continue
                inputs = [value]
            else:
                inputs = u.all_input_nodes
            if not _dtype(u).is_floating_point and any(
                    _is_tensor_node(a) and _dtype(a).is_floating_point
                    for a in inputs):
                found = u
                break
            stack += inputs
        if found is None:
            for u in seen:
                self.indices[u] = None
        self.indices[start] = found
        return found

    # --- levels -----------------------------------------------------------

    def _levels_only(self, n, name, sources):
        """Whether every one of an assembler's element `sources` is a level:
        a number; a value the program fixes -- through moves, float casts,
        factors of one value, neg and abs, a factory of fixed values
        (where(x > 0, 1., -1.), a filled mask, where(x > 0, ones_like(x),
        -ones_like(x))), or a constant whatever its shape (a learned scalar
        or per-channel alpha: where(x > 0, alpha, -alpha); a where of two
        learned tables as well, each element one of two values the program
        fixes, which no shape tells from a binarisation of one row); or, for
        an assembler that chooses each element in place (_CHOOSERS), a
        runtime value constant where the result varies (_level_by_shape: a
        row's own scale, where(x > 0, s, -s)). Its values are then a few
        levels, a grid; not a random factory's. True, False, or _UNREAD when
        a runtime source's partition was not read (it may be a level)."""
        if not sources:
            return False
        for s in sources:
            if isinstance(s, bool) or _is_number(s):
                continue
            if not isinstance(s, torch.fx.Node):
                return False
            base = s
            for _ in range(_BUDGET):
                base, _ = _walk(base)
                if _call_name(base) in ("aten::neg", "aten::abs") and \
                        base.args and _is_tensor_node(base.args[0]):
                    base = base.args[0]
                    continue
                break
            else:
                raise _Exhausted("_BUDGET", f"the sources of {n.name} "
                                 f"through neg and abs, which do not end")
            bname = _call_name(base)
            if bname in _RANDOM_FACTORIES:
                return False
            if bname in _FACTORIES:
                continue
            if self.sig.ep is not None and self.sig.is_constant(
                    s, self.constants):
                continue
            if name not in _CHOOSERS:
                return False
            got = self._level_by_shape(n, s)
            if got is not True:
                return got
        return True

    def _level_by_shape(self, n, s):
        """Whether the chooser `n`'s runtime source `s` holds one value
        where `n` may differ (_distinguishes over their partitions): True,
        a level -- a row's own scale in where(x > 0, s, -s); False when it
        differs wherever `n` may; _UNREAD when a partition not read leaves
        it open."""
        shape, own = _aligned_shape(n, 0), _aligned_shape(s, 0)
        cn, cs = self._classes(n), self._classes(s)
        if shape is None or own is None or cn is None or cs is None:
            return _UNREAD
        cs = _align(cs, own, shape)
        if cs is None:
            return _UNREAD
        open_ = False
        for d, size in enumerate(shape):
            x = _ident(size) if cn[d] is None else cn[d]
            if _distinguishes(x, [_ident(size) if cs[d] is None else cs[d]]):
                return True
            if cs[d] is None and _distinguishes(x, [_unif(size)]):
                open_ = True
        return _UNREAD if open_ else False

    # --- the rest ---------------------------------------------------------

    def _getitem_step(self, n):
        src, index = n.args[0], n.args[1]
        name = _schema_name(src)
        if name is None:
            return Origin(UNRECOGNISED, n, (),
                          f"an output of {getattr(src, 'target', src)}")
        namespace, _, short = name.partition("::")
        if name in _SPLITS and src.args:
            return [src.args[0]]
        if name in _SELECTIONS and src.args and index == 0:
            return [src.args[0]]
        if namespace in _RECIPE_NAMESPACES or any(w in short
                                                  for w in _QUANT_WORDS):
            return Origin(QUANTISED, n, (), f"an output of {name}")
        if name in _COMPUTES or name in _PLAIN_ESPRESSIF:
            return Origin(_NEUTRAL, n, (), f"an output of {name}")
        return Origin(UNRECOGNISED, n, (), f"output {index} of {name}, which "
                      f"the q@kT allow-list does not know")

    def _written_sources(self, buffer, fqn):
        """What the forward writes into buffer `fqn`, to be judged in its
        place: the sources of each write in its mutation chain, or the whole
        value written."""
        sources = []
        node = _mutation(self.ep, fqn)
        while node is not None:
            base, _ = walk(node)
            if base is None:
                return Origin(UNRECOGNISED, buffer, (),
                              f"buffer {fqn}, whose writes do not end")
            if base is buffer:
                break
            name = _schema_name(base)
            if name in _WRITES and base.args and _is_tensor_node(
                    base.args[0]):
                sources += [s for s in _ASSEMBLERS[name](base)[1:]
                            if _is_tensor_node(s)]
                node = base.args[0]
                continue
            sources.append(base)
            break
        return sources

    def _constant_codes(self, node):
        """The constant codes a recipe dequantise `node` reads (their name),
        through moves, whatever range it declares; else None."""
        name = _schema_name(node)
        if name is None or not node.args:
            return None
        namespace, _, short = name.partition("::")
        if namespace not in _RECIPE_NAMESPACES or "dequantize" not in short:
            return None
        codes, _ = _walk(node.args[0])
        if not isinstance(codes, torch.fx.Node) or not self.sig.is_constant(
                codes, self.constants):
            return None
        return codes.name


def origin(value, ep=None) -> Origin:
    """The origin of `value`'s elements: the first refusing origin the
    provenance search finds on any path, else PLAIN for a runtime value whose
    every path ends in plain float computation, a graph input or the
    program's constants (or CONSTANT / COMPUTED_CONSTANT for a value the
    constants alone determine)."""
    try:
        return _Classifier(ep).origin(value)
    except _Exhausted as e:
        # A hard limit reached: refused by name, never admitted.
        return Origin(UNRECOGNISED, value, (), f"a search cut short at "
                      f"{e.limit}: {e.detail}")
    except RecursionError:
        # Assemblies nested past Python's stack: refused, never admitted.
        return Origin(UNRECOGNISED, value, (),
                      "sources nested too deep to judge")


def _via(origin_, operand):
    path = [n for n in origin_.path if n != getattr(operand, "name", None)]
    return f" through {' -> '.join(path)}" if path else ""


def _refusal(origin_, operand, role, reads):
    """(kind, detail) refusing an operand of origin `origin_`."""
    name = getattr(operand, "name", operand)
    at = getattr(origin_.node, "name", origin_.node)
    seen = f"{name}{_via(origin_, operand)} reads {at}, {origin_.detail}"
    kind = origin_.kind
    if kind in (STATIC_PAIR, CONSTANT_CODES):
        return ("static pair on a bmm of two activations",
                f"{seen}: the delegate has no integer bmm of two statically "
                f"quantised activations (Stage B carries codes between ops, "
                f"not into one); this q@kT would quantise the {role} its own "
                f"way")
    if kind in (CHAIN, CHAIN_IN_PLACE):
        return ("dynamic chain on a q@kT operand",
                f"{seen}: the recipe quantised the {role} with its own "
                f"scheme, and the q@kT delegate would quantise that value "
                f"again its own way; the bmm runs portable on the recipe's "
                f"value")
    if kind == GRID:
        return ("q@kT operand already on a grid",
                f"{seen}: the model quantises the {role} itself, and the "
                f"q@kT delegate would quantise that value again its own way")
    if kind == QUANTISED:
        return ("q@kT operand from a quantised op",
                f"{seen}: a value on a quantisation's grid or table, which "
                f"the q@kT delegate would quantise again its own way")
    if kind == CONSTANT:
        return ("activation is a build-time constant",
                f"{seen}: the delegate reads {reads} at run time; an op over "
                f"constants is left for constant propagation to fold")
    if kind == COMPUTED_CONSTANT:
        return ("activation is a computed constant",
                f"{seen}: the program's constants alone determine it, and "
                f"the delegate reads {reads} at run time; the partition would "
                f"pull it in as a constant")
    if kind == STATE:
        return ("activation is a buffer the model writes",
                f"{seen}: {at} is a state the forward writes, met where the "
                f"search does not follow what it writes")
    if kind == CODES_CAST:
        return ("q@kT operand is integer codes cast to float",
                f"{seen}: integer codes, which the q@kT delegate would "
                f"quantise again its own way")
    return ("q@kT operand of unrecognised origin",
            f"{seen}: the q@kT allow-list knows the ops that compute new "
            f"values (_COMPUTES), the ones that move, select or keep values "
            f"(walked through), and quantisations (refused); an op on none "
            f"of its lists is refused until it is proven and listed, and "
            f"the bmm runs portable")


def operand_problem(ep, operand, role, reads="both q@kT operands"):
    """(kind, detail) refusing a q@kT operand the delegate quantises itself
    (the query of every K-cache op, both sides of OP_BMM_DYN_I8) unless its
    origin is plain float, else None."""
    got = origin(operand, ep)
    if got.kind == PLAIN:
        return None
    return _refusal(got, operand, role, reads)


def kcache_part_problem(ep, part, role, reads="both q@kT operands"):
    """(kind, detail) refusing a K cache's codes, scales or zero points
    unless they are a runtime tensor the delegate can read in place, else
    None. Any values qualify -- the delegate reads them as stored, so its
    key side is the codes' own -- but not a constant or a computed constant,
    which the partition would pull in. A cache the forward writes is a
    runtime tensor, read directly or through moves: no partition takes a
    written buffer in."""
    sig = _Signature(ep)
    node, path = walk(part)
    names = tuple(n.name for n in path)
    got = None
    if node is None:
        got = Origin(UNRECOGNISED, path[-1], names,
                     f"a walk of {len(path)} moves that does not end")
        return _refusal(got, part, role, reads)
    if isinstance(node, torch.fx.Node) and node.op == "placeholder" and \
            sig.written_fqn(node) is not None:
        return None           # a cache the forward writes: a runtime value
    # Constness of the part itself: a constant table gathered by runtime
    # indices is a runtime tensor.
    if got is None and sig.is_constant(part, {}):
        got = (sig.placeholder_origin(node, names)
               if isinstance(node, torch.fx.Node)
               and node.op == "placeholder" else
               Origin(COMPUTED_CONSTANT, node, names,
                      f"{_schema_name(node) or node.target} over constants "
                      f"alone"))
    if got is None or got.kind in (PLAIN, UNRECOGNISED):
        return None
    return _refusal(got, part, role, reads)
