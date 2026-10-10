# Step 4: Minimal SIMT Execution Loop

## Objective

Execute the initial PTX subset needed to run an `nvcc`-generated vector-add
kernel on the CPU.

## Plan

- Parse kernel parameter declarations from PTX.
- Add a structured execution engine with grids, blocks, warps, and registers.
- Implement special registers.
- Implement the vector-add PTX instruction subset.
- Implement a raw PTX launch path and driver `cuLaunchKernel`.
- Add an end-to-end vector-add execution test.

## Files To Create

```text
steps/STEP_04.md
src/exec/launch_config.h
src/exec/grid.h
src/exec/block.h
src/exec/warp.h
src/exec/register_file.h
src/exec/executor.h
src/exec/executor.cpp
src/exec/instruction_decoder.h
src/exec/instruction_decoder.cpp
tests/ptx/vector_add_exec_smoke.cpp
```

## Acceptance Checks

```bash
make test
```

The generated vector-add kernel must compute the same result as the CPU
reference.

## Work Log

- Added kernel parameter parsing to the PTX AST and IR.
- files: `src/ptx/ast.h`, `src/ptx/ir.h`, `src/ptx/parser.cpp`, `src/ptx/lower.cpp`
- verification: `ptx.parser_smoke` passed.

- Added a minimal execution engine with grids, blocks, warps, register files, special registers, and scalar per-lane control flow.
- files: `src/exec/launch_config.h`, `src/exec/grid.h`, `src/exec/block.h`, `src/exec/warp.h`, `src/exec/register_file.h`, `src/exec/executor.h`, `src/exec/executor.cpp`, `src/core/CMakeLists.txt`
- verification: Core library built successfully.

- Wired `cuLaunchKernel` to pack PTX parameters and execute the generated vector-add kernel.
- files: `src/compat/cuda_shim.cpp`
- verification: `ptx.vector_add_exec_smoke` passed.

- Added an end-to-end driver-API vector-add execution test using `nvcc`-generated PTX.
- files: `tests/ptx/vector_add_exec_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `make test` passed all 12 tests.

## Verification

Commands run:

```bash
make format
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 12
```

## Status

complete
