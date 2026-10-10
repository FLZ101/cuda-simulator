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

## Verification

Not run yet.

## Status

in progress
