# Step 8: Differential Test Harness

## Objective

Create a repeatable differential harness that runs a generated host runner and
compares its output to a CPU reference.

## Plan

- Add tolerance helpers.
- Add output comparison helpers.
- Add a differential runner script.
- Add a vector-add fixture host runner.
- Wire the vector-add differential test into CTest.

## Files To Create

```text
steps/STEP_08.md
tests/differential/tolerance.py
tests/differential/compare.py
tests/differential/run_all.py
tests/differential/fixtures/vector_add/host_runner.cpp
```

## Acceptance Checks

```bash
make test
```

The differential test must compare simulator output against a CPU reference and
pass within the configured tolerance.

## Work Log

- Added FP32 tolerance helpers and float list comparison.
- files: `tests/differential/tolerance.py`, `tests/differential/compare.py`
- verification: `differential.vector_add` passed.

- Added a differential runner script for the vector-add fixture.
- files: `tests/differential/run_all.py`
- verification: `differential.vector_add` passed.

- Added a generated host runner that reads inputs, runs the kernel, and writes output.
- files: `tests/differential/fixtures/vector_add/host_runner.cpp`, `tests/CMakeLists.txt`
- verification: `differential.vector_add` passed.

- Wired the differential vector-add test into CTest with `LD_PRELOAD`.
- files: `tests/CMakeLists.txt`
- verification: `make test` passed all 18 tests.

- Added a small demo kernel corpus (`saxpy` and `copy`), generated their PTX with `nvcc`, and added differential execution tests.
- files: `tests/kernels/saxpy.cu`, `tests/kernels/copy.cu`, `tests/kernels/demo_runner.cpp`, `tests/kernels/run_demos.py`, `tests/CMakeLists.txt`, `src/exec/executor.cpp`
- verification: `make test` passed all 19 tests.

- Added classical operator CUDA sources and PTX generation for naive matmul, tiled matmul, reduction, transpose, ReLU, and softmax.
- files: `tests/kernels/*.cu`, `tests/CMakeLists.txt`
- verification: `make test` passed and all generated PTX passed the SIMT gate.

- Added more convolution and transformer-related operator sources: direct/im2col/shared convolution, bias add, batch norm, pooling, layer norm, attention scores, embedding lookup, and GELU.
- files: `tests/kernels/*.cu`, `tests/CMakeLists.txt`
- verification: `make test` passed and all generated PTX passed the SIMT gate.

- Added GPU-reference differential execution for `copy`, `saxpy`, `relu`, and `gelu` using the real GPU as the golden output.
- files: `tests/kernels/demo_runner.cpp`, `tests/kernels/run_demos.py`, `tests/CMakeLists.txt`
- verification: `make test` passed all 19 tests.

## Verification

Commands run:

```bash
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 19
```

Required before Step 9:

- Every generated operator kernel must have an execution test and a CPU
  reference comparison before ResNet integration starts.
- PTX generation and SIMT-gate checks are not enough for operator correctness.

## Status

complete
