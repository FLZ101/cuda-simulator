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

## Verification

Not run yet.

## Status

in progress
