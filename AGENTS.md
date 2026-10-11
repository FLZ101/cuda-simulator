# Agent Instructions for cuda-simulator

## What this project is

This repository contains a CPU-hosted simulator for small, externally
generated CUDA programs. The simulator loads PTX and executes SIMT kernels on
the host CPU. A CUDA-capable GPU is required to run `nvcc`-compiled CUDA
programs as golden references during differential verification.

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
- Host CPU references are allowed only in early-stage smoke/execution tests
  such as the vector-add smoke test. The Step 8 differential harness must use
  the real GPU for `nvcc` programs; do not add a host CPU reference path there.

## Current state

- `DESIGN.md` exists and is the source of truth.
- The implementation order is defined in `DESIGN.md`, Section 11.
- CUDA toolkit and a CUDA-capable GPU are required to create and verify CUDA
  test programs.

## Workflow

1. Read `DESIGN.md`.
2. Work bottom-up in the order given in Section 11.
3. Prefer correctness before performance.
4. Do not speculate on unsupported PTX or CUDA APIs. Implement the smallest
   missing feature required by a real generated test.
5. Add a regression test with every new instruction or API.

## Adding a new kernel/operator

Generated kernels are grouped under `tests/kernels/<group>/` and tested by the
generic differential runner. For most new SIMT operators:

1. Pick an existing group or add the group to `KERNEL_GROUPS` in
   `tests/CMakeLists.txt`.

2. Add the CUDA source:

   ```text
   tests/kernels/<group>/<operator>.cu
   ```

   Use the generic scalar signature:

   ```cuda
   extern "C" __global__ void <operator>(
       const float *a, const float *b, const float *c, const float *d,
       const int *idx, float *out, int p0, int p1, int p2, int p3, int p4,
       int p5, int p6, int p7, float f0, float f1, float f2, float f3);
   ```

   `a`-`d` are optional float inputs, `idx` is an optional int input, `out` is
   the output, `p0` is the output float count, and `p1`-`p7`/`f0`-`f3` carry
   shape/algorithm parameters.

3. Add `<operator>` to `KERNELS_<group>` in `tests/CMakeLists.txt`.

4. Add `<operator>` to `SUPPORTED_OPERATORS` and to the matching entry in
   `OPERATOR_GROUPS` in `tests/reference/run_gpu_reference.py`.

5. Add a deterministic case in `run_gpu_reference.py`:

   ```python
   run_pair("<operator>", generic_args(input_path, ints=(out_count, ...)))
   ```

6. If the PTX function name must differ from the mode name, extend
   `generic_kernel_name` in `tests/reference/gpu_reference_runner.cu`.

7. Verify the group first, then the full suite:

   ```bash
   cmake --build build -j
   ctest --test-dir build -R kernels.gpu_reference_differential_<group> --output-on-failure
   ctest --test-dir build --output-on-failure
   make format-python
   make precommit
   ```

8. Record the work in `steps/STEP_08.md` and update `DESIGN.md` if this is a
   new operator category.

Only use a custom dispatch branch in `tests/reference/gpu_reference_runner.cu`
when the generic signature cannot represent the operator. Custom branches still
need a generated PTX target and must use the real GPU as the golden reference.

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
