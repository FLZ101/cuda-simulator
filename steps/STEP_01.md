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

## Verification

Not run yet.

## Status

in progress
