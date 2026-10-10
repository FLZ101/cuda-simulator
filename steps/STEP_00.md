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

- hash: `1b28c42`
- subject: `Add step recording workflow and step 0 plan`
- files: `AGENTS.md`, `steps/STEP_00.md`
- notes: Added the agent workflow, step recording rules, and the initial Step 0 plan.

- hash: `82122d4`
- subject: `Create step 0 build and test skeleton`
- files:
  - `.gitignore`
  - `CMakeLists.txt`
  - `README.md`
  - `cmake/CompilerWarnings.cmake`
  - `cmake/FindLLVM.cmake`
  - `cmake/SimulatorOptions.cmake`
  - `src/CMakeLists.txt`
  - `src/cli/CMakeLists.txt`
  - `src/cli/main.cpp`
  - `src/compat/CMakeLists.txt`
  - `src/compat/cuda_shim.cpp`
  - `src/compat/cudart_shim.cpp`
  - `src/core/CMakeLists.txt`
  - `src/core/version.cpp`
  - `src/core/version.h`
  - `src/core/version.h.in`
  - `tests/CMakeLists.txt`
  - `tests/smoke/vector_add_ref.cpp`
- notes: Created C++20 CMake targets for the core library, two shim libraries, CLI runner, and smoke test. The smoke test and CLI runner build and run.

## Verification

Commands run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCUDA_SIM_ENABLE_JIT=OFF -DCUDA_SIM_ENABLE_SANITIZERS=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/src/cli/cuda_sim_run
```

Result:

```text
100% tests passed, 0 tests failed out of 1
cuda-simulator 0.1.0 jit=off
```

## Status

complete
