# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Accept fp8-distributed checkpoints by promoting their constants to fp32.

WHY THIS IS NOT AN fp8 KERNEL, and why that is the right call:

fp8 is a DISTRIBUTION format here, never a compute one. The LX7 vector unit
multiplies int8 and int16 and nothing else -- there is no fp8 multiply, no
fp16/bf16 multiply, and the FPU is fp32 scalar. So any fp8 op is decode-then-
integer-MAC, and the decode is the part that will not vectorise: applying an
exponent is a per-element variable shift, PIE shifts take one scalar shift amount
shared across every lane, and a 256-entry LUT would need a per-element gather
that PIE also does not have. Decode stays scalar, roughly a byte at a time,
feeding a kernel that otherwise does 16 lanes per cycle.

And it would buy nothing in traffic: fp8 is 8 bits, exactly like i8. What it buys
is intra-row dynamic range, which is what per-group scales buy for ~1.6% extra
traffic while keeping the int8 kernel.

So the useful thing to support is the FILE, not the arithmetic. An fp8 checkpoint
gets transcoded once, on the host, at export: promote to fp32, then quantise to
whichever rung the format policy picked. The device never sees an fp8 byte, so
there is no runtime cost to weigh -- the question "is fp8 too slow here?" does
not arise, because nothing fp8 executes.

WHY IT HAS TO HAPPEN HERE, above `to_edge`:

    SpecViolationError: These operators are taking Tensor inputs with mismatched
    dtypes: dim_order_ops._to_dim_order_copy.default with args:
    {'self': torch.float8_e4m3fn, '__ret_0': torch.float32}

`to_edge` refuses the graph before any partitioner runs. fp8 is legal in the
ATen dialect and not in the edge dialect, so a backend cannot fix this no matter
what it claims -- the fix belongs in the window where fp8 is still legal, which
is the exported program. That is also why this is a separate module rather than
another branch in `_resolve_const`: by the time that code runs, the graph this is
for has already been rejected.

Both fp8 dtypes are handled. e4m3 (more mantissa, less range) is what weights are
usually shipped in; e5m2 (more range) is the gradient/activation one, accepted
here because a checkpoint that mixes them should not half-load.
"""

import torch

FP8_DTYPES = tuple(
    getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2",
                                      "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)

# fp16/bf16 checkpoints hit the SAME wall one step later. They are legal edge
# dtypes, so `to_edge` accepts them -- and then the backend cannot resolve the
# weight through the cast:
#
#     Esp32Backend needs the weight to be a constant; got aten_permute_copy
#
# which is a crash, not a refusal, for `w.to(torch.float32)` -- how every loaded
# half-precision checkpoint is written.
#
# They are promoted under a STRICTER rule than fp8, and the difference matters.
# fp8 has no choice: unpromoted it is rejected outright, so promoting is the only
# way the model runs at all. fp16 does have a choice, and promoting one
# unconditionally would be a policy change rather than a fix -- it doubles the
# constant's memory and changes the numerics of any portable op still reading it.
#
# So a half constant is promoted only when EVERY user is already an explicit
# upcast to fp32. That makes it pure constant folding: the graph had itself
# declared the value is to be used at fp32, and this only moves the conversion
# from run time to build time. Nothing observable changes.
HALF_DTYPES = tuple(d for d in (torch.float16, torch.bfloat16))


def _is_upcast_to(node, src) -> bool:
    """Is `node` a dtype cast of `src` whose result is an ordinary float?

    Matched by the node's RESULT dtype rather than by target name. There are
    several spellings of a cast (`aten.to.dtype`, `aten._to_copy.default`,
    `aten.to.dtype_layout`) and which one export emits is not this module's
    business; what makes the cast redundant is that its input is about to become
    the dtype it was casting to.
    """
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function"):
        return False
    if not node.args or node.args[0] is not src:
        return False
    name = str(node.target)
    if not ("_to_copy" in name or ".to." in name or name.endswith(".to")):
        return False
    val = node.meta.get("val")
    return (isinstance(val, torch.Tensor)
            and val.dtype in (torch.float32, torch.float16, torch.bfloat16))


def _storage_for(ep, node):
    """(dict, key) holding a placeholder's tensor, or None if it has none."""
    sig = ep.graph_signature
    for mapping, store in (
        (getattr(sig, "inputs_to_parameters", {}), ep.state_dict),
        (getattr(sig, "inputs_to_buffers", {}), ep.state_dict),
        (getattr(sig, "inputs_to_lifted_tensor_constants", {}),
         getattr(ep, "constants", {})),
    ):
        key = mapping.get(node.name)
        if key is not None and store is not None and key in store:
            return store, key
    return None


def promote_fp8_constants(ep, dtype=torch.float32) -> int:
    """Upcast fp8 (always) and fp16/bf16 (when already upcast) constants in place.

    Returns the number promoted.

    Also erases the cast that followed, rather than leaving it as an fp32->fp32
    no-op. Two reasons, and the second is the one that matters: a surviving
    identity cast sits between the weight and its consumer, and `_resolve_const`
    folds a specific set of ops to reach a constant -- a cast is not in it, so
    the weight would resolve to None and the op would silently stay portable.
    "Loads but does not delegate" is a worse outcome than the loud failure this
    replaces, because nothing reports it.
    """
    if not FP8_DTYPES and not HALF_DTYPES:
        return 0
    graph = ep.graph_module.graph
    promoted = 0
    for node in list(graph.nodes):
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        if val.dtype in FP8_DTYPES:
            pass                      # always: unpromoted, to_edge refuses it
        elif val.dtype in HALF_DTYPES:
            # Only when the graph already upcasts every use. See HALF_DTYPES.
            users = [u for u in node.users
                     if "_assert_tensor_metadata" not in str(u.target)]
            if not users or not all(_is_upcast_to(u, node)
                                    and u.meta["val"].dtype == dtype
                                    for u in users):
                continue
        else:
            continue
        found = _storage_for(ep, node)
        if found is None:
            # An fp8 *input* rather than a stored constant. Left alone
            # deliberately: promoting it would silently change the method's
            # signature, and to_edge will still reject it -- loudly, naming the
            # dtype, which is the correct outcome for "your model takes fp8 in".
            continue
        if val.dtype in FP8_DTYPES:
            # Promoted unconditionally, but only through casts: a use that reads
            # the fp8 value itself -- a graph output, an op taking it as is --
            # would have its dtype contract changed under it.
            bad = [u for u in node.users
                   if "_assert_tensor_metadata" not in str(u.target)
                   and not _is_upcast_to(u, node)]
            if bad:
                raise RuntimeError(
                    f"fp8 constant {node.name} is read as fp8 by {bad[0].name}; "
                    f"only uses that cast it to a float dtype can be promoted, "
                    f"and to_edge refuses fp8 itself")
        store, key = found
        promoted_value = store[key].to(dtype)
        if isinstance(store[key], torch.nn.Parameter):
            # .to() returns a plain Tensor; a parameter entry must stay one.
            promoted_value = torch.nn.Parameter(
                promoted_value, requires_grad=store[key].requires_grad)
        store[key] = promoted_value
        # Remember WHERE this value came from. Promotion is lossless, so after it
        # runs there is nothing in the graph that says this weight arrived
        # already quantised -- and that is the one fact someone needs to decide
        # whether to re-quantise it. `Esp32Partitioner(requantize_promoted=False)`
        # reads this meta and refuses the op, leaving it portable at fp32 with
        # exactly the checkpoint's values.
        node.meta["esp32_promoted_from"] = str(val.dtype)
        node.meta["val"] = val.to(dtype)
        for user in list(node.users):
            # export inserts `_assert_tensor_metadata` in front of a `.to()` to
            # guard the source dtype. It asserts the fp8 we are removing, takes
            # no users, and would fail at to_edge with a traceback pointing at
            # the model's forward rather than at this pass -- which is a
            # genuinely confusing place to land.
            if "_assert_tensor_metadata" in str(user.target):
                graph.erase_node(user)
                continue
            if _is_upcast_to(user, node) and user.meta["val"].dtype == dtype:
                user.replace_all_uses_with(node)
                graph.erase_node(user)
        promoted += 1

    if promoted:
        graph.eliminate_dead_code()
        ep.graph_module.recompile()
    return promoted
