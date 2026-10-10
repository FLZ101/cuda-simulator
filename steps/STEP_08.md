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

## Verification

Not run yet.

## Status

in progress
