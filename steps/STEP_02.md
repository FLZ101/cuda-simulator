# Step 2: Runtime State

## Objective

Move the CUDA shims from the Step 1 compatibility state to a real runtime
layer with device, context, memory, stream, and event management.

## Plan

- Add a runtime singleton that owns device and context state.
- Add a thread-safe host-backed memory manager.
- Add ordered stream command queues and asynchronous workers.
- Add event markers and synchronization.
- Update runtime and driver shims to use the runtime layer.
- Add stream/event and asynchronous copy tests.

## Files To Create

```text
steps/STEP_02.md
src/runtime/device.h
src/runtime/device.cpp
src/runtime/context.h
src/runtime/context.cpp
src/runtime/memory.h
src/runtime/memory.cpp
src/runtime/stream.h
src/runtime/stream.cpp
src/runtime/event.h
src/runtime/event.cpp
src/runtime/runtime.h
src/runtime/runtime.cpp
src/runtime/errors.h
src/runtime/launch.h
tests/compat/runtime_stream_event_smoke.cpp
tests/compat/driver_stream_event_smoke.cpp
```

## Acceptance Checks

```bash
make test
```

The new tests must pass asynchronous copies on two streams and cross-stream
event waits.

## Work Log

- Added a runtime layer with device, context, memory, stream, event, and runtime singleton classes.
- files: `src/runtime/*.h`, `src/runtime/*.cpp`, `src/core/CMakeLists.txt`
- verification: Core library built successfully.

- Routed the compatibility state through the new runtime device and memory managers.
- files: `src/core/compat_state.h`, `src/core/compat_state.cpp`
- verification: Existing runtime and driver memory tests still passed.

- Wired the runtime and driver shims to the new stream/event managers and made async copies queue through streams.
- files: `src/compat/cudart_shim.cpp`, `src/compat/cuda_shim.cpp`
- verification: New stream/event smoke tests passed.

- Added runtime and driver asynchronous copy, stream, and event smoke tests.
- files: `tests/compat/runtime_stream_event_smoke.cpp`, `tests/compat/driver_stream_event_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `make test` passed all 7 tests.

## Verification

Commands run:

```bash
make format
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 7
```

## Status

complete
