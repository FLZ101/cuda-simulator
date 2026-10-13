# Native PTX-to-C++ Backend

## Purpose

The native backend is the performance path for generated SIMT PTX kernels. It
keeps the interpreter as the correctness fallback while compiling each lowered
`FunctionIR` to a small self-contained C++ translation unit and then to a
shared object loaded with `dlopen`.

The backend is intentionally not an LLVM JIT. The simulator already owns the
PTX frontend and IR, so it lowers straight from that IR to C++ and lets the
host compiler (`c++`) optimize the result with
`-O3 -march=native`.

## Pipeline

```text
cuLaunchKernel
    |
    v
lowered ModuleIR cache
    |
    v
Executor::execute
    |
    +--> try_execute(FunctionIR, LaunchConfig, arguments, variables)
             |
             v
         generate_native_source -> self-contained C++
             |
             v
         FNV-1a hash + .cache/cudasim-native
             |
             v
         c++ -std=c++20 -O3 -march=native -fPIC -shared
             |
             v
         dlopen + dlsym("cudasim_native_kernel")
             |
             v
         run grid blocks on CUDA_SIM_EXEC_THREADS workers
             |
             v
         success -> return true
         failure -> Executor falls back to the interpreter
```

## Entry ABI

Every generated shared object exports:

```cpp
extern "C" int cudasim_native_kernel(
    const unsigned char *args,
    const unsigned int *block,
    const unsigned int *grid,
    unsigned int bx, unsigned int by, unsigned int bz);
```

- `args` is the packed CUDA kernel parameter buffer.
- `block` is `{block.x, block.y, block.z}`.
- `grid` is `{grid.x, grid.y, grid.z}`.
- `bx/by/bz` identify the block this invocation must execute.
- Returning nonzero makes `try_execute` fall back to the interpreter.

## Generated Function Shape

The generated function:

1. Decodes the thread index from `block` and the current block index.
2. Iterates over warps in the block.
3. Creates the per-warp register and predicate arrays:

   ```cpp
   std::array<std::array<uint64_t, 32>, register_count> R{};
   std::array<std::array<uint8_t, 32>, predicate_count> P{};
   ```

4. Iterates over active lanes.
5. Executes the flattened basic-block list as a `switch (pc)` loop.
6. Writes through to host memory for global loads/stores.

Register banks match the interpreter:

```text
r<0..N>  -> R[0..1023]
rd<0..N> -> R[1024..2047]
f<0..N>  -> R[2048..3071]
rs<0..N> -> R[3072..4095]
```

The generated array is sized from the highest register index actually
referenced by that function, so it does not always zero-fill all 4096 entries.
Predicates use `P[index][lane]` as `uint8_t`, matching the interpreter's
zero-initialized boolean semantics.

Special registers are translated to generated locals:

```text
%tid.*, %ntid.*, %ctaid.*, %nctaid.*, %laneid, %warpid
```

## Code Generation

`src/jit/ptx_to_cpp.cpp` tokenizes each instruction's operands again from the
lowered IR and emits C++ expressions per opcode family:

- `ld.param.*`, `ld.global.*`, `st.global.*`
- `mov.*`
- integer arithmetic: `add/sub/mul/mad`, `and/or/xor`, shifts
- wide multiply and `mul.hi.*`
- `bfe.*`
- predicates: `setp.*`, `and/or/not/mov.pred`, `selp`
- floating-point arithmetic and approximate math
- integer/float conversions and `cvta.to.global.u64`
- `bra`, `bra.uni`, `ret`, `exit`

Unsupported instructions, `bar.sync`, shared memory, and atomics are not
translated. `generate_native_source` returns no program in those cases and the
existing interpreter handles the whole kernel.

## Compilation and Cache

`src/jit/native_backend.cpp`:

- hashes generated source with FNV-1a;
- writes `kernel_<hash>.cpp`;
- compiles to a temporary shared object with
  `c++ -std=c++20 -O3 -march=native -fPIC -shared -w`;
- renames the temporary object to `kernel_<hash>.so` atomically enough for
  single-process use;
- caches loaded handles in memory;
- writes compiler or `dlopen` failures to `kernel_<hash>.log` and falls back.

Default cache root is:

```text
<current working directory>/.cache/cudasim-native
```

The cache is keyed only by generated C++ source. Generated source includes the
function name and PTX instruction stream, so separate functions and changed
PTX produce different `.so` artifacts.

## Execution Parallelism

`try_execute` splits independent grid blocks across worker threads:

- worker count = `min(CUDA_SIM_EXEC_THREADS, block_count)`;
- block indices are computed in row-major order from `bx/by/bz`;
- any worker returning nonzero sets a shared failure flag;
- on failure, the interpreter reruns the entire kernel.

`CUDA_SIM_EXEC_THREADS` follows the same convention as the interpreter:

```text
unset / "1" -> single-threaded
"N" > 1     -> N workers
"auto"/"0"  -> std::thread::hardware_concurrency()
```

## Module Lowering Cache

`src/compat/cuda_shim.cpp` caches the lowered `ModuleIR` per loaded
`ptx::Module` in `cuLaunchKernel`. This avoids re-lexing, re-parsing, and
re-lowering the full fatbin/PTX module for every kernel launch in a model
run. The native backend only sees the already-lowered `FunctionIR`.

## Fallback and Correctness

The native path is skipped or retried when:

- `CUDA_SIM_DISABLE_NATIVE` is non-empty and not `"0"`.
- `CUDA_SIM_NATIVE_EXCLUDE` contains a comma-separated substring matching the
  function name.
- the function declares shared-memory variables.
- translation returns no source.
- compilation, loading, or symbol lookup fails.
- any block execution returns nonzero.

The interpreter is the correctness oracle. Generated native code is expected to
produce the same output as the interpreter for the supported subset.

## Validation

The full-size ResNet-50 simulator gate is:

```bash
make resnet-gpu-cudasim \
  RESNET_BATCH=1 \
  RESNET_HEIGHT=224 \
  RESNET_WIDTH=224 \
  RESNET_BLOCK=32
```

This completes in 247.70 seconds on the tested development host and matches the
PyTorch CPU reference. Smaller 16x16 and 64x64 runs, PTX regression tests, and
the GPT-2/ResNet model simulator CTest targets are also green.

## Future Work

- Add explicit global-memory bounds checking, shared by both interpreter and
  native paths.
- Derive register layout from PTX `.reg` declarations rather than fixed bank
  bases.
- Add more PTX opcodes as newly generated models require them.
- Move block execution to a persistent worker pool instead of creating workers
  per kernel launch.
