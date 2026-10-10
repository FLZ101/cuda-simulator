# Step 6: Memory Model and Coalescing

## Objective

Add a host-backed memory model, access classification, coalesced access, and
basic atomics.

## Plan

- Add device memory ranges and bounds checking.
- Add access-pattern classification.
- Add a coalescer for uniform, contiguous, strided, and divergent patterns.
- Add basic global-memory atomics.
- Add focused memory-model tests.

## Files To Create

```text
steps/STEP_06.md
src/mem/device_memory.h
src/mem/device_memory.cpp
src/mem/access_pattern.h
src/mem/coalescer.h
src/mem/coalescer.cpp
src/mem/atomics.h
src/mem/atomics.cpp
tests/mem/memory_model_smoke.cpp
```

## Acceptance Checks

```bash
make test
```

Memory microbenchmarks must handle uniform, contiguous, strided, and divergent
access patterns and basic atomics.

## Work Log

- Added a bounds-checked host-backed device buffer.
- files: `src/mem/device_memory.h`, `src/mem/device_memory.cpp`, `src/core/CMakeLists.txt`
- verification: `mem.memory_model_smoke` passed.

- Added access-pattern classification and coalesced read/write helpers.
- files: `src/mem/access_pattern.h`, `src/mem/coalescer.h`, `src/mem/coalescer.cpp`
- verification: `mem.memory_model_smoke` passed.

- Added basic 32-bit integer and float atomic add operations.
- files: `src/mem/atomics.h`, `src/mem/atomics.cpp`
- verification: `mem.memory_model_smoke` passed.

- Added a focused memory-model smoke test.
- files: `tests/mem/memory_model_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `make test` passed all 15 tests.

- Integrated coalesced global `f32` loads/stores and `atom.add.global.f32/u32` into the PTX executor.
- files: `src/exec/executor.cpp`, `src/mem/coalescer.h`, `src/mem/coalescer.cpp`
- verification: existing vector-add, shared-memory, and multi-warp tests still passed.

## Verification

Commands run:

```bash
make format
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 15
```

## Status

complete
