# Agent Instructions for cuda-simulator

## What this project is

This repository contains a CPU-hosted simulator for small, externally
generated CUDA programs. The simulator loads PTX and executes SIMT kernels on
the host CPU without an NVIDIA GPU.

The v1 end-to-end targets are:

- ResNet-18 or ResNet-50
- GPT-2 124M

The full design is in [DESIGN.md](DESIGN.md). Read it before changing code or
adding a plan.

## Hard rules

- Keep the simulator focused on executing generated PTX. Do not implement
  model operators, PyTorch integration, cuBLAS, cuDNN, or cuFFT.
- Do not execute CUBIN/SASS. PTX only.
- Tensor-core instructions are out of scope:

  ```text
  mma
  wmma
  ldmatrix
  stmatrix
  wgmma
  tcgen05
  ```

- Generated kernels must use the SIMT path.
- `cp.async` may be supported only as a synchronous copy emulation.
- Run standalone generated host programs, not a full Python process.
- Keep v1 simple: one device, one primary context, dynamic CUDA runtime.

## Current state

- `DESIGN.md` exists and is the source of truth.
- No implementation code exists yet.
- The implementation order is defined in `DESIGN.md`, Section 11.

## Workflow

1. Read `DESIGN.md`.
2. Work bottom-up in the order given in Section 11.
3. Prefer correctness before performance.
4. Do not speculate on unsupported PTX or CUDA APIs. Implement the smallest
   missing feature required by a real generated test.
5. Add a regression test with every new instruction or API.

## Step recording guide

Every implementation step must have a record file:

```text
steps/STEP_00.md
steps/STEP_01.md
...
steps/STEP_12.md
```

Before starting a step:

1. Create the step markdown if it does not exist.
2. Record the objective, initial plan, files to create, and acceptance checks.
3. Record `Status: in progress`.

While working:

- Record what was done under the step's `## Work Log` section.
- Do not record commit hashes or commit subjects.
- Give each completed piece of work its own short entry:

  ```text
  - <what was implemented or changed>
  - files: <affected files>
  - verification: <what was checked after this commit>
  ```

- Update the `Status` and `Verification` sections after important milestones.
- When the step is complete, set `Status: complete`.

The step markdown is part of the repository and should be committed with the
related work. Do not leave it as an untracked local note.

## Code style

- C++20.
- Internal code lives in namespace `cudasim`.
- Only `compat` shim libraries export CUDA ABI symbols.
- C++ code must be formatted with `clang-format` using the LLVM style.
- Install the Git pre-commit hook with `make install-hooks`.
- Use `apply_patch` for file edits.
- Keep generated artifacts and fixtures under `tests/`.

## Important implementation details

- PTX parsing must go through:

  ```text
  PTX text -> lexer -> parser -> AST -> lower -> simulator IR
  ```

- Never execute the AST directly.
- Register state is structure-of-arrays:

  ```text
  register R -> lane[0], lane[1], ..., lane[31]
  ```

- A block needs a runnable-warp scheduler so `bar.sync` cannot deadlock.
- A kernel stream command is complete only after every block in its grid has
  finished.
- Bounds-check every memory access and report CUDA-style errors.
- Keep the interpreter as the correctness oracle. JIT is optional and must
  fall back to the interpreter.

## Verification

Always run the smallest relevant test after a change.

For the current design phase:

```bash
make test
```

For generated programs:

```bash
LD_PRELOAD=/abs/path/libcudart_shim.so:/abs/path/libcuda_shim.so \
  ./generated_program
```

Reject tensor-core artifacts with:

```bash
grep -E '\b(mma|wmma|ldmatrix|stmatrix|wgmma|tcgen05)\b' kernel.ptx
```

## Commit style

- Commit small, coherent changes.
- Mention the affected component in the commit message.
- Keep `DESIGN.md` updated when a design decision changes.
