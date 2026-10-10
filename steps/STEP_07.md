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

- Added a PTX SIMT-only gate that rejects tensor-core opcodes.
- files: `tests/generated/check_simt.py`, `tests/CMakeLists.txt`
- verification: `ptx.simt_gate` passed.

- Added a generated-PTX coverage report script with the target operator set.
- files: `tests/generated/run_coverage.py`, `tests/CMakeLists.txt`
- verification: `ptx.coverage_report` passed.

- Wired the current `nvcc`-generated vector-add, shared-memory, and multi-warp barrier fixtures through the gate.
- files: `tests/CMakeLists.txt`
- verification: `make test` passed all 17 tests.

## Verification

Commands run:

```bash
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 17
```

## Status

complete
