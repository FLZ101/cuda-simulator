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
- **Step 10 checkpoints 1-4**: `tests/models/specs/module.py` now lowers
  traced `nn.Embedding`, `causal_softmax`, and wrapped `multihead_attention`
  calls to SIMT PTX kernels, and a full GPT-2 transformer block runs through
  the traced model flow. The new `models.gpt2_embedding`,
  `models.gpt2_attention`, `models.gpt2_multihead`, and `models.gpt2_block`
  CPU/sim tests pass.

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

## Tasks / Next Steps

1. Replace the fixed 1024-per-bank register banks with a
   `.reg`-declaration-driven `RegisterLayout` once a generated kernel exceeds
   current capacity.
2. Continue extending the model-runner converter for the full GPT-2 stack and
   LM head. Token embedding, causal attention, and one transformer block
   lowering are complete.
3. Add missing PTX semantics as newly generated kernels expose them.
