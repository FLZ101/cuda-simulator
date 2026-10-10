# Step 1: CUDA Shims

## Objective

Build `libcudart_shim.so` and `libcuda_shim.so` so an externally generated CUDA
program can initialize device 0, allocate/free device memory, and report a
plausible simulated device without a real CUDA driver.

## Plan

- Add common CUDA runtime and driver ABI types.
- Add shared compatibility state for device and memory operations.
- Implement the minimum runtime API surface.
- Implement the minimum driver API surface.
- Add `nvcc`-style registration hook stubs.
- Add runtime, driver, and symbol-presence tests.

## Files To Create

```text
steps/STEP_01.md
src/compat/export.h
src/compat/cuda_api.h
src/compat/error_codes.h
src/compat/cudart_shim.cpp
src/compat/cuda_shim.cpp
src/core/compat_state.h
src/core/compat_state.cpp
tests/compat/runtime_api_smoke.cpp
tests/compat/driver_api_smoke.cpp
tests/compat/check_symbols.sh
```

## Acceptance Checks

```bash
make test
```

The runtime and driver smoke tests must initialize device 0, allocate/free
device memory, and complete without loading a real CUDA driver.

## Work Log

- Added common CUDA runtime/driver ABI types, error codes, and the shared-library export macro.
- files: `src/compat/cuda_api.h`, `src/compat/error_codes.h`, `src/compat/export.h`
- verification: `make test` compiled the headers successfully.

- Added shared compatibility state for one simulated device and host-backed memory allocation.
- files: `src/core/compat_state.h`, `src/core/compat_state.cpp`, `src/core/CMakeLists.txt`
- verification: Runtime and driver memory tests passed.

- Implemented the `libcudart_shim.so` runtime API surface and `nvcc` registration hook stubs.
- files: `src/compat/cudart_shim.cpp`, `src/compat/CMakeLists.txt`
- verification: `compat.runtime_api_smoke` and exported-symbol checks passed.

- Implemented the `libcuda_shim.so` driver API surface.
- files: `src/compat/cuda_shim.cpp`
- verification: `compat.driver_api_smoke` and exported-symbol checks passed.

- Added standalone runtime/driver smoke tests and an exported-symbol test.
- files: `tests/compat/runtime_api_smoke.cpp`, `tests/compat/driver_api_smoke.cpp`, `tests/compat/check_symbols.sh`, `tests/CMakeLists.txt`
- verification: `make format && make test` passed all 4 tests.

- Added CUDA toolkit detection and an actual `nvcc`-compiled runtime smoke test that runs under `LD_PRELOAD`.
- files: `CMakeLists.txt`, `tests/compat/nvcc_smoke.cu`, `tests/compat/run_nvcc_smoke.sh`, `tests/CMakeLists.txt`
- verification: `make test` passed all 5 tests.

## Verification

Commands run:

```bash
make format
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 5
```

The CUDA toolkit was found at `/usr/local/cuda`, and the `nvcc`-compiled smoke
test passes through `LD_PRELOAD`.

## Status

complete
