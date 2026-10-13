# Step 11: Full-Size ResNet-50 Simulator Performance

## Objective

Make the simulator complete a full-size ResNet-50 forward pass
(`1 x 3 x 224 x 224`, FP32, generated Triton SIMT PTX, block size 32) in an
acceptable wall-clock time on a typical multi-core development host, while
keeping the interpreter as the correctness oracle.

## Status

complete

## Plan

The interpreter is retained as the correctness fallback. The performance path
is a native-code backend rather than an LLVM JIT:

```text
lower once -> PTX IR -> generated C++ -> external c++ -O3 -shared ->
.cache/cudasim-native -> dlopen -> execute grid blocks -> interpreter fallback
```

The native path is attempted first for every kernel launch. When translation,
compilation, loading, or execution fails, or when the kernel uses shared
memory/barriers/unsupported instructions, `Executor` falls back to the existing
interpreter.

Detailed backend design and implementation notes live in
[`NATIVE_JIT.md`](../NATIVE_JIT.md).

### Architecture

- `src/jit/ptx_to_cpp.*`
  - translates a lowered `FunctionIR` into one self-contained C++ translation
    unit.
  - emits one C++ function per block, uses SoA-like register and predicate
    arrays, and lowers hot PTX instructions directly to C++ expressions.
  - returns no program when an instruction cannot be lowered safely.
- `src/jit/native_backend.*`
  - hashes generated source with FNV-1a.
  - compiles with
    `c++ -std=c++20 -O3 -march=native -fPIC -shared`.
  - caches `.so` artifacts under `.cache/cudasim-native`.
  - loads `cudasim_native_kernel` with `dlopen`/`dlsym`.
  - runs independent grid blocks on `CUDA_SIM_EXEC_THREADS` workers.
- `src/compat/cuda_shim.cpp`
  - caches the lowered `ModuleIR` per loaded PTX module so each launch does
    not re-lex/re-parse/re-lower the entire module.

### Supported Native Instruction Set

The first implementation targets the instructions emitted by the generated
ResNet-50 and GPT-2 SIMT PTX, including:

```text
mov.*, ld.param.*, ld.global.*, st.global.*, add/sub/mul/mad,
fma.rn.f32, and/or/xor, shifts, setp.*, selp/predicate ops,
float arithmetic and approximate math, integer/float conversions,
mul.hi.*, bfe.*, cvta.to.global.u64, bra/bra.uni, ret/exit
```

Anything else falls back to the interpreter.

### Acceptance Target

- Correctness:

  ```bash
  make resnet-gpu-cudasim \
    RESNET_BATCH=1 \
    RESNET_HEIGHT=224 \
    RESNET_WIDTH=224 \
    RESNET_BLOCK=32
  ```

  must finish with FP32 output within the existing comparison tolerances and
  the same argmax as the PyTorch CPU reference and real-GPU run.

- Performance:
  - Record a warm native path and interpreter baseline.
  - Target at least 20x improvement for hot kernels.
  - Do not trade correctness for speed.

### Validation

```bash
cmake --build build -j
ctest --test-dir build -R '^ptx\.' --output-on-failure
ctest --test-dir build -R 'models\.(resnet_stem|gpt2_block)\.runner_sim' --output-on-failure
make resnet-gpu-cudasim RESNET_BATCH=1 RESNET_HEIGHT=16 RESNET_WIDTH=16 RESNET_BLOCK=32
make format-python
make precommit
```

## Work Log

- Replaced the interpreter-fast-path/AVX/LLVM plan with a native PTX-to-C++
  backend that compiles to a shared object with the external host compiler,
  caches artifacts under `.cache/cudasim-native`, loads them with `dlopen`, and
  falls back to the interpreter.
- files: `steps/STEP_11.md`
- verification: no build or test required for plan update.

- Added `src/jit/ptx_to_cpp.*` to lower a `FunctionIR` to C++ and
  `src/jit/native_backend.*` to compile, cache, load, and execute generated
  programs.
- files: `src/jit/ptx_to_cpp.h`, `src/jit/ptx_to_cpp.cpp`,
  `src/jit/native_backend.h`, `src/jit/native_backend.cpp`,
  `src/core/CMakeLists.txt`
- verification: library builds with warnings-as-errors.

- Integrated native execution into `Executor::execute` with interpreter
  fallback.
- files: `src/exec/executor.cpp`
- verification: `ctest --test-dir build -R '^ptx\.' --output-on-failure`
  passes.

- Added coverage for the hot ResNet-50 and GPT-2 PTX opcodes, including
  `bfe.*`, `mul.hi.*`, `ex2.approx.*`, `cvt.rn.f32.s32`, and
  `cvt.rzi.s32.f32`.
- files: `src/jit/ptx_to_cpp.cpp`
- verification: GPT-2 block simulator test passes after fixing `mov.b32`
  `0fXXXXXXXX` hex-float immediate handling.

- Fixed `mov.pred` with an immediate source and `mov.b32`/`mov.u32` handling
  of `0fXXXXXXXX` bit-pattern immediates.
- files: `src/jit/ptx_to_cpp.cpp`
- verification: all ResNet-50 generated PTX kernels translate; 16x16 warm run
  matches the PyTorch CPU reference.

- Cached lowered `ModuleIR` per module in `cuLaunchKernel`, removing repeated
  full-module parse/lower work from every kernel launch.
- files: `src/compat/cuda_shim.cpp`
- verification: warmed `make resnet-gpu-cudasim` 16x16 simulator run completes
  in 11.38s and matches the CPU reference.

- Sized each generated native register/predicate array from the highest
  register index referenced by that PTX function instead of always emitting
  the full 4096-register and 256-predicate arrays. This keeps the interpreter
  zero-initialization semantics while reducing per-block register-set setup
  cost.
- files: `src/jit/ptx_to_cpp.cpp`
- verification: native cache rebuilds successfully for all 175 ResNet-50
  kernels; 16x16 and 64x64 `resnet-gpu-cudasim` runs match the CPU reference.

- Removed the unused LLVM/JIT CMake option and Makefile flag now that the
  native backend uses the external host compiler and disk cache.
- files: `CMakeLists.txt`, `Makefile`, `README.md`, `DESIGN.md`,
  `SESSION_SUMMARY.md`
- verification: clean CMake reconfigure and build pass.

- Passed the full-size ResNet-50 acceptance run.
- files: `steps/STEP_11.md`
- verification:
  `make resnet-gpu-cudasim RESNET_BATCH=1 RESNET_HEIGHT=224 RESNET_WIDTH=224 RESNET_BLOCK=32`
  completes in 247.70s and matches the PyTorch CPU reference.

- Completed the full 124M/12-block pretrained GPT-2 simulator gate with
  Triton 3.8 codegen; all 150 generated nodes complete and the final logits
  and argmax match the PyTorch CPU reference.
- files: `steps/STEP_11.md`, `Makefile`, `SESSION_SUMMARY.md`, `DESIGN.md`
- verification: `make gpt2-gpu-cudasim` passes with the default
  `50257/768/12/12` configuration, reporting `PyTorch CPU reference matches
  (50257 floats); argmax matches (198)`.
