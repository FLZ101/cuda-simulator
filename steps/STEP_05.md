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
the expected results.

## Work Log

- Added PTX variable declaration parsing inside function bodies for `.shared`, `.local`, and `.const`.
- files: `src/ptx/parser.h`, `src/ptx/parser.cpp`, `src/ptx/ast.h`, `src/ptx/ir.h`, `src/ptx/lower.cpp`
- verification: `ptx.parser_smoke` and `ptx.fatbin_smoke` passed.

- Extended the executor with shared-memory allocation, `ld.shared`, `st.shared`, `bar.sync`, `shl.b32`, `add.s32`, `setp.ne.s32`, and immediate operands.
- files: `src/exec/executor.h`, `src/exec/executor.cpp`
- verification: `ptx.shared_memory_exec_smoke` passed.

- Fixed module lookup so `cuLaunchKernel` retains module-level shared variables.
- files: `src/loader/module_registry.h`, `src/loader/module_registry.cpp`, `src/compat/cuda_shim.cpp`
- verification: `ptx.shared_memory_exec_smoke` passed.

- Added an `nvcc`-generated shared-memory reduction execution test.
- files: `tests/ptx/shared_memory_kernel.cu`, `tests/ptx/shared_memory_exec_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `make test` passed all 13 tests.

- Added a block-level runnable-warp scheduler and a multi-warp barrier test with 64 threads.
- files: `src/exec/executor.cpp`, `tests/ptx/multi_warp_barrier_kernel.cu`, `tests/ptx/multi_warp_barrier_exec_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `make test` passed all 14 tests.

## Verification

Commands run:

```bash
make format
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 14
```

## Status

complete
