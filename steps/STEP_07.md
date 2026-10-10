# Step 7: Generated PTX Coverage and SIMT Gate

## Objective

Establish the generated-PTX coverage process and enforce the SIMT-only gate
before operator kernels are added.

## Plan

- Add a PTX tensor-core scan script.
- Add a generated-PTX coverage report script.
- Wire both into CTest.
- Run the current `nvcc`-generated fixtures through the gate.

## Files To Create

```text
steps/STEP_07.md
tests/generated/check_simt.py
tests/generated/run_coverage.py
```

## Acceptance Checks

```bash
make test
```

All generated PTX fixtures must pass the SIMT gate, and the coverage script must
report the known target operator set.

## Work Log

## Verification

Not run yet.

## Status

in progress
