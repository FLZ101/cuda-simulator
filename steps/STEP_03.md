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

- Added a PTX lexer, token model, parser, AST, module representation, and AST-to-IR lowering.
- files: `src/ptx/token.*`, `src/ptx/lexer.*`, `src/ptx/ast.h`, `src/ptx/parser.*`, `src/ptx/ir.*`, `src/ptx/lower.*`, `src/ptx/module.*`, `src/core/CMakeLists.txt`
- verification: `ptx.parser_smoke` passed.

- Added fatbin/PTX extraction and a module registry.
- files: `src/loader/fatbin.*`, `src/loader/module_registry.*`
- verification: `ptx.fatbin_smoke` passed.

- Wired raw PTX driver module loading and runtime registration hooks into the shims.
- files: `src/compat/cuda_shim.cpp`, `src/compat/cudart_shim.cpp`
- verification: `compat.driver_module_smoke` and `compat.runtime_registration_smoke` passed.

- Added a raw PTX CLI entry listing/lookup path.
- files: `src/cli/main.cpp`
- verification: `cuda_sim_run --module tests/ptx/vector_add.ptx --entry add_kernel` found the entry.

- Added a hand-written vector-add PTX fixture and parser/fatbin/module tests.
- files: `tests/ptx/vector_add.ptx`, `tests/ptx/parser_smoke.cpp`, `tests/ptx/fatbin_smoke.cpp`, `tests/compat/driver_module_smoke.cpp`, `tests/compat/runtime_registration_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `make test` passed all 11 tests.

## Verification

Commands run:

```bash
make format
make test
./build/src/cli/cuda_sim_run --module tests/ptx/vector_add.ptx --entry add_kernel
```

Result:

```text
100% tests passed, 0 tests failed out of 11
entry add_kernel found
```

## Status

complete
