# Step 8: Differential Test Harness and Operator Qualification

## Objective

Create a repeatable differential harness and qualify every generated operator
before ResNet or GPT-2 integration. Step 8 uses the real GPU for `nvcc` kernels
and PyTorch CPU reference for PyTorch-generated models.

## Plan

- Add tolerance helpers.
- Add output comparison helpers.
- Add a GPU-reference differential runner script.
- Add a GPU-reference differential runner for demo kernels.
- Add one `nvcc`-compiled GPU reference runner containing all operator kernels
  and dispatching by kernel name.
- Generate deterministic random inputs for each operator case.
- Run each operator on the real GPU and through the simulator, then compare.
- Wire GPU-reference differential tests into CTest.
- Add a PyTorch CPU reference flow for simple PyTorch-generated models with
  one or a few operators.

## Files To Create

```text
steps/STEP_08.md
tests/differential/tolerance.py
tests/differential/compare.py
tests/kernels/demo_runner.cpp
tests/kernels/run_demos.py
tests/reference/gpu_reference_runner.cu
tests/reference/run_gpu_reference.py
tests/models/compare_torch_reference.py
tests/models/export_simple_model_reference.py
tests/models/generate_simple_model_runner.py
```

## Acceptance Checks

```bash
make test
```

- Later `nvcc`-compiled demo kernels must pass against real GPU output.
- Every operator in the qualification matrix must pass against real GPU output
  with deterministic random inputs.
- Simple PyTorch-generated models with one or a few operators, such as matmul,
  conv, or MLP, must pass against a saved PyTorch CPU reference with identical
  weights, inputs, and evaluation mode.

## Reference Implementation Details

### GPU reference for `nvcc` operators

Target files:

```text
tests/reference/gpu_reference_runner.cu
tests/reference/run_gpu_reference.py
```

`gpu_reference_runner.cu`:

- contains all operator kernels from `tests/kernels/*.cu` in one CUDA
  translation unit, either by including them or compiling them together
- dispatches by kernel name:

  ```bash
  gpu_reference_runner --kernel matmul_naive \
    --input-a a.bin --input-b b.bin --output gpu_out.bin \
    --m 8 --n 8 --k 8
  ```

- is compiled with `nvcc`, not linked against `cuda_shim`:

  ```bash
  nvcc -arch=sm_80 gpu_reference_runner.cu -o gpu_reference_runner
  ```

- writes output as little-endian C-order raw binary with the same layout used
  by the simulator runner.

`run_gpu_reference.py`:

- generates deterministic random inputs for each operator once
- runs the GPU runner without `LD_PRELOAD` to produce `*.gpu.raw`
- runs the same runner with simulator shims to produce `*.sim.raw`
- compares with the configured tolerance:

  ```text
  FP32: rtol = 1e-4, atol = 1e-5
  FP16/BF16: rtol = 5e-2, atol = 1e-2
  integer: exact
  ```

CTest wiring:

```cmake
add_test(
  NAME kernels.gpu_reference_differential
  COMMAND python3 tests/reference/run_gpu_reference.py
          --runner $<TARGET_FILE:gpu_reference_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
)
```

The GPU runner must use real CUDA runtime/driver symbols. If it links directly
to the shim, the unshimmed "GPU" run is still the simulator, which is the
current `operator_runner` bug.

### PyTorch CPU reference for PyTorch-generated runners

Step 8 uses this path only for simple PyTorch-generated models with one or a
few operators, such as matmul, conv, or MLP. It is not used for hand-written
`nvcc` operator kernels. The same mechanism can be reused later for full
ResNet/GPT-2 models, but full-model qualification is not part of Step 8.

Target files:

```text
tests/models/compare_torch_reference.py
tests/models/export_simple_model_reference.py
tests/models/generate_simple_model_runner.py
```

Reference export for a simple model:

```python
import torch

torch.manual_seed(1234)
model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 8),
)
model.eval()
example_input = torch.randn(4, 16)

with torch.no_grad():
    reference = model(example_input)

torch.save(
    {
        "input": example_input,
        "output": reference,
        "state_dict": model.state_dict(),
        "shape": tuple(reference.shape),
        "dtype": str(reference.dtype),
    },
    "reference.pt",
)
```

The same flow applies to one-operator examples:

- matmul: `torch.matmul` or a thin `nn.Linear` wrapper
- conv: `torch.nn.Conv2d`
- MLP: a small `nn.Sequential` with a few linear/nonlinear layers

For full ResNet/GPT-2 models, the same export mechanism is reused in Steps 9
and 10. Step 8 does not require those full-model references.

Generated runner:

- trace the simple PyTorch model
- lower the traced graph with TorchInductor/Triton using the SIMT path only
- wrap the generated kernels in a small CUDA host runner that loads the PTX and
  launches kernels in graph order
- link the wrapper with `nvcc` to produce `model_runner`
- run the generated PTX through the tensor-core gate before execution

`model_runner` contract:

```text
--input input.raw
--output output.raw
```

It must allocate device memory, copy the input, launch all generated kernels,
and copy the final output back to host. It must load or embed PTX only, never
CUBIN/SASS.

Simulator run:

```bash
LD_PRELOAD=libcudart_shim.so:libcuda_shim.so \
  ./model_runner --input input.raw --output simulator_output.raw
```

Compare:

```bash
python3 tests/models/compare_torch_reference.py \
  --simulator-output simulator_output.raw \
  --reference reference.pt
```

Comparison requirements:

- identical weights, input tensors, evaluation mode, and random seed
- output dtype/layout must match exactly
- compare final logits with configured tolerance
- for classification-like simple models, also compare predicted class/argmax
  where applicable
- stochastic paths are not allowed; use greedy/deterministic evaluation

## Work Log

- Added FP32 tolerance helpers and float list comparison.
- files: `tests/differential/tolerance.py`, `tests/differential/compare.py`
- verification: comparison helpers are used by the GPU-reference harness.

- Added a small demo kernel corpus (`saxpy` and `copy`), generated their PTX with `nvcc`, and added differential execution tests.
- files: `tests/kernels/saxpy.cu`, `tests/kernels/copy.cu`, `tests/kernels/demo_runner.cpp`, `tests/kernels/run_demos.py`, `tests/CMakeLists.txt`, `src/exec/executor.cpp`
- verification: `make test` passed all 19 tests.

- Added classical operator CUDA sources and PTX generation for naive matmul, tiled matmul, reduction, transpose, ReLU, and softmax.
- files: `tests/kernels/*.cu`, `tests/CMakeLists.txt`
- verification: `make test` passed and all generated PTX passed the SIMT gate.

- Added more convolution and transformer-related operator sources: direct/im2col/shared convolution, bias add, batch norm, pooling, layer norm, attention scores, embedding lookup, and GELU.
- files: `tests/kernels/*.cu`, `tests/CMakeLists.txt`
- verification: `make test` passed and all generated PTX passed the SIMT gate.

- Added GPU-reference differential execution for `copy`, `saxpy`, `relu`, and `gelu` using the real GPU as the golden output.
- files: `tests/kernels/demo_runner.cpp`, `tests/kernels/run_demos.py`, `tests/CMakeLists.txt`
- verification: `make test` passed all 19 tests.

- Added an `operator_runner` and differential harness for the remaining classical operators. The test is currently disabled because `layer_norm` still exposes missing PTX execution semantics and `operator_runner` links directly to `libcuda_shim`, so its unshimmed "GPU" run is still the simulator.
- files: `tests/kernels/operator_runner.cpp`, `tests/kernels/run_operators.py`, `tests/CMakeLists.txt`, `src/exec/executor.cpp`
- verification: `make test` is green; `kernels.operator_differential` is disabled pending PTX fixes.

- Merged Step 8 and Step 8.5 into one Step 8 record and updated the operator qualification workflow.
- files: `DESIGN.md`, `steps/STEP_08.md`
- verification: `make test` passed.

- Removed the early-stage host CPU vector-add differential fixture. Step 8 now uses only real GPU and PyTorch CPU references.
- files: `tests/differential/fixtures/vector_add/host_runner.cpp`, `tests/differential/run_all.py`, `tests/CMakeLists.txt`, `DESIGN.md`, `steps/STEP_08.md`
- verification: `make test` passed.

- Added explicit implementation details for the GPU reference runner and PyTorch CPU reference flow.
- files: `steps/STEP_08.md`
- verification: `make test` passed.

## Verification

Commands run:

```bash
make test
```

Result:

```text
100% tests passed, 0 tests failed out of 18
kernels.operator_differential did not run (Disabled)
```

Required before Step 9:

- Every generated `nvcc` operator kernel must have an execution test against
  the real GPU reference before ResNet integration starts.
- Replace `operator_runner` with the `nvcc`-compiled
  `tests/reference/gpu_reference_runner.cu` and re-enable
  `kernels.operator_differential`.
- Add the remaining missing PTX execution semantics one operator at a time
  until every matrix operator passes.
- PTX generation and SIMT-gate checks are not enough for operator correctness.

## Status

in progress
