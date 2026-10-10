# Step 5: Full Execution Semantics

## Objective

Extend the minimal executor with shared memory, local memory, constant memory,
address-space casts, barriers, and the instruction subset needed for generated
microkernels.

## Plan

- Parse `.shared`, `.local`, and `.const` variable declarations.
- Extend the executor with block shared memory and per-warp barrier handling.
- Add local and constant memory support.
- Add shared/global address-space casts.
- Add shared-memory and barrier microkernel tests.

## Files To Create

```text
steps/STEP_05.md
src/exec/shared_memory.h
src/exec/barrier.h
src/exec/barrier.cpp
src/exec/address_space.h
tests/ptx/shared_memory_smoke.cu
tests/ptx/shared_memory_smoke.cpp
```

## Acceptance Checks

```bash
make test
```

Generated microkernels using branches, shared memory, and barriers must match
reference results.

## Work Log

## Verification

Not run yet.

## Status

in progress
