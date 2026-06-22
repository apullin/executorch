# Arm Ethos-U Fast Lowering Design Note

## Status

Proposed, experimental.

This note describes a narrow ExecuTorch extension point for backend-specific
fast lowering. The motivating backend is Arm Ethos-U, but the design is meant
to be generic and opt-in.

## Problem

ExecuTorch's standard Arm flow uses `to_edge_transform_and_lower(...)` to:

1. run ATen decompositions
2. generate Edge dialect programs
3. run edge conversion passes
4. partition and lower delegated regions

That path is correct, but it is expensive on large quantized graphs. In the
current local prototype, the Rust-backed Arm lowering core is already fast
enough, while the dominant cost sits in the generic ExecuTorch orchestration
around it.

Local prototype numbers on `cnn_layernorm4d`:

- existing Edge-based `all_rust` path: about `10.4s`
- direct raw quantized fast lane: about `1.8s`

The direct fast lane is not hypothetical. It already produces a delegated
`ExportedProgram` via `to_backend(...)`, but it does not yet re-enter the
standard Edge packaging path cleanly.

## Goals

- Preserve ExecuTorch's public artifact boundary:
  `ExportedProgram -> EdgeProgramManager -> ExecutorchProgramManager`
- Allow a backend to bypass the expensive generic ATen-to-Edge pipeline when it
  can lower an earlier dialect directly
- Keep the change narrow, explicit, and opt-in
- Preserve the existing `to_edge_transform_and_lower(...)` behavior for all
  backends that do not opt in
- Make the fast path testable, fallback-safe, and suitable for upstream review

## Non-goals

- Replacing the generic ExecuTorch ATen-to-Edge pipeline
- Adding a generic monkeypatch or callback framework
- Changing ExecuTorch's runtime delegate representation
- Requiring all partitioners or backends to support direct lowering
- Supporting arbitrarily complex multi-partitioner composition in the first
  version

## Key observation

The direct Arm path already reaches the same delegate representation that
ExecuTorch knows how to serialize and execute:

- `LoweredBackendModule`
- `executorch_call_delegate`
- delegated `ExportedProgram`

The remaining gap is packaging/legalization, not backend lowering. In the
current prototype, wrapping the raw fast-lowered program in
`EdgeProgramManager` succeeds, but `.to_executorch()` fails because top-level
raw `quantized_decomposed::{quantize,dequantize}_per_tensor` operators are not
yet legalized into the form expected by `ToOutVarPass`.

This suggests the right architectural move is:

- add a first-class fast-lane seam
- keep Edge and ExecuTorch packaging intact
- add a small legalization step for any surviving boundary ops

## Proposed design

### 1. Add an optional fast-lane protocol on the partitioner side

Introduce a new optional interface in ExecuTorch, separate from the existing
`Partitioner.partition(...)` contract.

Example shape:

```python
@dataclass
class DirectLoweringResult:
    exported_program: ExportedProgram
    post_lowering_passes: tuple[PassType, ...] = ()
    override_verifiers: Optional[list[type[Verifier]]] = None


class DirectLoweringPartitioner(Protocol):
    def direct_lower(
        self,
        exported_program: ExportedProgram,
        compile_config: EdgeCompileConfig,
    ) -> Optional[DirectLoweringResult]:
        ...
```

Notes:

- This is intentionally optional.
- Returning `None` means "not applicable, use the existing path."
- The hook is owned by the partitioner layer because `to_edge_transform_and_lower(...)`
  already operates in terms of partitioners, not backend classes.
- The returned program is expected to be semantically equivalent to the input
  program, but already contain delegated regions.
- If needed, the result may carry a small post-lowering legalization stack and
  verifier override for boundary cleanup before the program is wrapped by
  `EdgeProgramManager`.

### 2. Gate the fast lane in `to_edge_transform_and_lower(...)`

In `to_edge_transform_and_lower(...)`, before calling the expensive generic
ATen-to-Edge flow, ExecuTorch should check whether the method's partitioner
supports direct lowering.

Initial constraints:

- only one partitioner per method
- no mixed fast-path and legacy lowering within the same method
- multi-partitioner methods fall back to the existing behavior

If the partitioner returns a direct-lowered program, ExecuTorch should:

1. run a small legalization step
2. wrap the result in `EdgeProgramManager`
3. continue through the normal `.to_executorch()` pipeline

If the hook declines or raises a recognized "not applicable" condition,
ExecuTorch falls back to the existing path unchanged.

### 3. Add a minimal legalization step before Edge packaging

The direct-lowered program is close to packagable already, but it may still
contain top-level operators from an earlier dialect.

For the Arm prototype, the first required legalization is:

- normalize surviving top-level raw
  `quantized_decomposed::{quantize,dequantize}_per_tensor`
  operators into the Edge/legalized form expected by the existing
  `to_executorch()` pipeline

This legalization should stay deliberately small. It is not a second full
ATen-to-Edge conversion pipeline; it is a boundary cleanup for programs that
have already been backend-lowered.

### 4. Keep the delegate representation unchanged

The fast lane should not invent a new artifact type.

The output of direct lowering should still be an `ExportedProgram` containing:

- `LoweredBackendModule` instances
- `executorch_call_delegate` call sites
- any small number of remaining ops that the legalization step can normalize

This keeps runtime, serialization, and debugging infrastructure stable.

## Why the partitioner is the right seam

The backend API alone is too late and too low-level for this decision.

`to_edge_transform_and_lower(...)` currently decides how to prepare programs for
delegation by consulting partitioners. A partitioner already owns:

- backend targeting
- operator support decisions
- compile specs through its `DelegationSpec`
- partitioning policy

Adding an optional direct-lowering capability here keeps the extension aligned
with existing responsibilities.

## Soundness constraints

The fast lane must satisfy the following invariants:

- The returned program is semantically equivalent to the input program.
- Delegate regions are represented using existing ExecuTorch mechanisms.
- Fallback to the current path is always available.
- The fast lane is explicit and opt-in, never inferred implicitly.
- The initial implementation must not alter behavior for non-participating
  backends.
- Validation remains owned by ExecuTorch, even if some verifier surfaces need a
  narrowly tailored direct-lowering legalization path.

## Initial scope

The first version should be intentionally narrow:

- single-method models
- one partitioner per method
- Arm Ethos-U as the only implementation
- a small direct-lowering legalization pass for top-level q/dq boundary ops

Everything else should fall back to the existing behavior.

## Rollout plan

### Phase 1: experimental ET hook

- Add the optional direct-lowering protocol
- Add a guarded fast-lane branch in `to_edge_transform_and_lower(...)`
- Keep it behind an explicit experimental flag or backend-owned opt-in

### Phase 2: Arm implementation

- Implement the protocol in `EthosUPartitioner`
- Reuse the already-proven raw quantized lowering lane
- Add the minimal legalization needed for packaging

### Phase 3: validation

- Run focused Arm model validation on anchor models
- Run ExecuTorch backend/unit coverage for the new ET seam
- Run the broader ExecuTorch test battery before proposing upstream merge

## Testing requirements

### ET-side tests

- non-fast-path partitioners still follow the existing code path
- unsupported direct-lowering attempts fall back cleanly
- multi-partitioner methods fall back cleanly
- direct-lowered programs still serialize via the normal ExecuTorch path

### Arm-side tests

- `tiny_cnn` direct-lowers correctly and reaches `.to_executorch()`
- `cnn_layernorm4d` direct-lowers correctly and reaches `.to_executorch()`
- legacy Arm `to_edge_transform_and_lower(...)` path still works unchanged

## Open questions

- Should the capability live directly on `Partitioner`, or as a separate
  protocol checked via `isinstance(...)` / `hasattr(...)`?
- Should the fast-lane branch return an `EdgeProgramManager` directly, or
  return a delegated `ExportedProgram` plus metadata that ET wraps?
- How much verifier special-casing is acceptable before the design becomes too
  broad?

## Recommendation

Proceed with a narrow experimental ExecuTorch extension point for direct
lowering, centered on the partitioner layer, with strict fallback and small
boundary legalization.

This keeps the architecture stable while making room for a backend that can
prove a materially better lowering path.
