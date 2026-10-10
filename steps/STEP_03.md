# Step 3: PTX/Fatbin Ingestion and IR

## Objective

Accept raw PTX and basic fatbin data, parse PTX into a module representation,
lower it to a simulator IR, and register kernels through runtime and driver
ingestion paths.

## Plan

- Add a PTX lexer and parser for the initial vector-add subset.
- Add a PTX module representation.
- Lower parsed functions to basic blocks and instructions.
- Add a module registry scoped by module handle.
- Accept raw PTX through the driver API and raw PTX CLI.
- Store runtime registration data for later kernel resolution.
- Detect CUBIN-only artifacts and report a clear error.

## Files To Create

```text
steps/STEP_03.md
src/ptx/token.h
src/ptx/token.cpp
src/ptx/lexer.h
src/ptx/lexer.cpp
src/ptx/ast.h
src/ptx/parser.h
src/ptx/parser.cpp
src/ptx/ir.h
src/ptx/ir.cpp
src/ptx/lower.h
src/ptx/lower.cpp
src/ptx/module.h
src/ptx/module.cpp
src/loader/fatbin.h
src/loader/fatbin.cpp
src/loader/module_registry.h
src/loader/module_registry.cpp
tests/ptx/vector_add.ptx
tests/ptx/parser_smoke.cpp
```

## Acceptance Checks

```bash
make test
```

The parser must find `add_kernel`, the IR must contain a basic block, the raw
PTX CLI must list the entry, and a CUBIN-only test must produce a clear error.

## Work Log

## Verification

Not run yet.

## Status

in progress
