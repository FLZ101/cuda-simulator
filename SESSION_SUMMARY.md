# cuda-simulator Session Summary

## Quickstart

```bash
make venv          # installs pinned PyTorch/Triton/NumPy + dev tooling
make build         # configure (Ninja) + build
make test          # full CTest suite (50 tests)
make format-python
make precommit
```

The full suite passes on the current CUDA 13.3 toolchain (see `README.md`
"Tested environments").

## Goal

CPU-hosted PTX simulator for externally generated CUDA programs. It executes
PTX only (never CUBIN/SASS). v1 correctness targets:

- ResNet-18 or ResNet-50 forward pass
- GPT-2 124M forward pass

## Current State

- **Differential harness**: `tests/kernels/<group>/` kernels run on the real GPU
  as golden, then through the simulator shims for comparison. Enabled groups:
  `elementwise`, `convolution`, `normalization`, `reduction`, `attention`,
  `indexing`, `loss`, `matrix`, `misc`, `detection`, `sequence`.
- **Model runners**: PyTorch specs under `tests/models/specs/` emit native CUDA
  runners with embedded Triton PTX. `--mode cpu` produces the reference;
  `--mode gpu --simulate` runs the native runner under the shims.
- **Declarative runner generation**: a shared emitter
  (`tests/models/specs/_common.py`) generates the C++ from a `Runner`
  description. `vector_add`, `matmul`, and `mlp` are declarative `build_runner`
  specs; `module.py` is the generic FX-traced converter.
- **Step 10 complete**: `tests/models/specs/module.py` now lowers
  traced `nn.Embedding`, `causal_softmax`, wrapped `multihead_attention`, and
  `position_ids`/`arange` calls to SIMT PTX kernels. The full GPT-2 stack uses
  the cached Hugging Face `gpt2` pretrained weights mapped into the traced
  custom stack. The full 124M/12-block pretrained CPU reference and native-GPU
  runner pass with argmax comparison; the full 124M/12-block simulator gate is
  also green after the Step 11 native backend landed.
- **Step 11 complete**: `steps/STEP_11.md` records the native PTX-to-C++
  backend. `src/jit/ptx_to_cpp.*` lowers a `FunctionIR` to self-contained C++
  with per-function register/predicate array sizing; `src/jit/native_backend.*`
  compiles it with the host `c++ -O3 -march=native`, caches the shared object
  under `.cache/cudasim-native`, loads it with `dlopen`, and runs independent
  grid blocks on `CUDA_SIM_EXEC_THREADS` workers. `Executor::execute` tries
  this path first and falls back to the interpreter for shared memory,
  barriers, atomics, translation/compile failures, or execution failures. The
  full-size `1 x 3 x 224 x 224` ResNet-50 simulator run passes in 247.70s and
  matches the PyTorch CPU reference. The full pretrained GPT-2 simulator run
  also passes: all 150 generated nodes complete and final logits/argmax match
  the PyTorch CPU reference.

## Where Things Live

- PTX pipeline: lexer -> parser -> AST -> lower -> simulator IR. The
  interpreter is `src/exec/executor.cpp`; never execute the AST directly.
- CUDA shims: `src/compat/` (`libcudart_shim.so`, `libcuda_shim.so`), loaded
  via `LD_PRELOAD`.
- Differential golden reference: `tests/reference/run_gpu_reference.py` and
  `tests/reference/gpu_reference_runner.cu`.
- Model runner generation: `tests/models/generate_model_runner.py`, specs in
  `tests/models/specs/`, CPU-reference comparison in
  `tests/models/compare_torch_reference.py`.
- Design and rules: `DESIGN.md`, `AGENTS.md`, `steps/STEP_08.md`.
- Native backend design and implementation: `NATIVE_JIT.md`.

## Reference Policy

- `nvcc` differential kernels: real GPU output is the golden reference.
- PyTorch model runners: PyTorch CPU reference. `--mode cpu` runs
  `spec.run_cpu()` to write `reference.pt`; `--mode gpu --simulate --compare`
  checks the simulated runner output against it.
- Early smoke tests may use a host CPU reference.

## Key Implementation Facts

- Register banks are disjoint: `r` base 0, `rd` base 1024, `f` base 2048
  (1024 entries each). PTX `%r/%rd/%f` are virtual names, not physical
  registers; the hardware 255-register/thread limit does not apply.
- `bar.sync` uses block-wide lane arrival tracking (no deadlock).
- Global memory offset parsing supports signed offsets.
- FMA uses `std::fmaf`; `cvt.sat.f32.f32` clamps to `[0.0, 1.0]`.
- Tensor-core instructions (`mma`, `wmma`, `ldmatrix`, `stmatrix`, `wgmma`,
  `tcgen05`) are rejected.
- CUDA 13 NVVM + Triton 3.8 codegen is covered: `div.*.ftz.f32`,
  `setp.ne.s32` signed immediates, `ld.param.b64/b32`, `mad.wide.s32/u32`,
  `bfe.s32/u32`, `cvt.u64.u32`, `mov.b64` immediates, `setp.ne.b64`, and
  `mov.b32/u32` with `0fXXXXXXXX` hex-float immediates.
- Triton 3.8 ABI: kernels take two trailing hidden scratch pointers
  (`global_scratch`, `profile_scratch`); the generator passes both.
- `mul.hi.s32`/`mul.hi.u32` execute the high 32 bits of the 64-bit signed or
  unsigned product, used by Triton's constant-division sequence.

## Tasks / Next Steps

1. Add bounds checking to native and interpreter global-memory accesses to
   preserve the project memory-safety rule.
2. Replace the fixed 1024-per-bank register arrays with a
   `.reg`-declaration-driven layout once a generated kernel exceeds current
   capacity.
3. Add missing PTX semantics as newly generated kernels expose them.
4. ~~Finish the full 124M/12-block simulator gate~~ complete in
   `steps/STEP_11.md`.
