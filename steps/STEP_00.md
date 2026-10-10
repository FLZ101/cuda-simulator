# Step 0: Build and Test Skeleton

## Objective

Create a C++20 project skeleton that can build the two CUDA shim libraries, the
core simulator library, a CLI runner, and a test executable. This step must
produce a repeatable build before CUDA semantics are implemented.

## Plan

- Add top-level CMake project.
- Add common compiler warning and sanitizer options.
- Add optional LLVM discovery.
- Add a generated version header.
- Add a trivial smoke test.
- Add README build instructions.

## Files To Create

```text
CMakeLists.txt
cmake/FindLLVM.cmake
cmake/CompilerWarnings.cmake
cmake/SimulatorOptions.cmake
src/version.h.in
tests/CMakeLists.txt
tests/smoke/vector_add_ref.cpp
README.md
```

## Acceptance Checks

```text
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The build must succeed with no simulator-specific CUDA functionality.

## Commits

<!-- Record each commit here. -->

## Verification

Not run yet.

## Status

in progress
