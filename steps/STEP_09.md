# Step 9: ResNet End-to-End

## Objective

Generate a standalone ResNet-50 host runner whose forward pass is implemented
as embedded SIMT PTX kernels, run that same runner through the simulator
shims, and compare the final FP32 logits and predicted class with a
deterministic PyTorch CPU reference.

## Plan

- Reuse the Step 8 PyTorch-generated model-runner flow instead of introducing
  a separate operator runtime:

  ```text
  traced PyTorch model
        |
        v
  tests/models/specs/module.py emits Triton SIMT kernels
        |
        v
  native CUDA host runner with embedded PTX
        |
        v
  LD_PRELOAD=libcudart_shim.so:libcuda_shim.so ./runner
        |
        v
  compare against PyTorch CPU reference
  ```

- Keep the generated kernels on the SIMT path. `compile_kernel` in
  `tests/models/specs/_common.py` must reject `mma`, `wmma`, `ldmatrix`,
  `stmatrix`, `wgmma`, and `tcgen05`.
- Add ResNet fixtures with the existing fixture contract:

  ```python
  def build_model(args): ...
  def example_input(args): ...
  ```

- Extend `tests/models/specs/module.py` only if symbolic tracing exposes an
  operator that is not already lowered. The current converter already covers
  `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `AdaptiveAvgPool2d((1, 1))`,
  `Flatten`, `Linear`, and graph-level `operator.add` used by residual paths.
- Add a deterministic argmax assertion to the existing PyTorch comparison
  flow so ResNet validation checks both logits and predicted class.
- Wire small-shape checkpoint tests first, then the full `3 x 224 x 224`
  forward pass.

## Files To Create

```text
steps/STEP_09.md
tests/models/fixtures/resnet_block.py
tests/models/fixtures/resnet_stage.py
tests/models/fixtures/resnet50.py
```

Likely files to modify:

```text
tests/models/specs/module.py
tests/models/model_runner.py
tests/models/compare_torch_reference.py
tests/CMakeLists.txt
DESIGN.md
SESSION_SUMMARY.md
```

## Incremental Checkpoints

1. Existing `resnet_stem.py` remains green from Step 8:
   `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`.
2. `resnet_block.py`: one Bottleneck block with both identity and downsample
   residual paths, exercising `Conv2d`, `BatchNorm2d`, `ReLU`, and
   `operator.add`.
3. `resnet_stage.py`: one `make_layer` stage containing several residual
   blocks, including the stage's downsampling path.
4. `resnet50.py`: full ResNet-50 stem, four stages, adaptive average pooling,
   flatten, and linear classifier head.
5. Full input shape `1 x 3 x 224 x 224`, FP32 output.

## Acceptance Checks

```bash
cmake --build build -j
ctest --test-dir build -R 'models.resnet50.runner_(cpu|sim)' --output-on-failure
ctest --test-dir build --output-on-failure
```

The generated PTX must contain none of the banned tensor-core opcodes, the
simulator output must match the PyTorch CPU reference within the configured
FP32 tolerances, and the argmax of the final logits must match the reference.

Reference constraints:

- `model.eval()`
- `torch.manual_seed(1234)`
- deterministic `3 x 224 x 224` input
- identical weights, input, and evaluation mode for CPU and simulator runs

## Work Log

- Created the Step 9 plan and recorded the objective, implementation path,
  incremental checkpoints, files to create, and acceptance checks.
- files: `steps/STEP_09.md`
- verification: No build or test required.

- Installed the CUDA-matching `torchvision` wheel, downloaded ImageNet
  ResNet-50 weights, and produced a deterministic PyTorch CPU reference for
  `1 x 3 x 224 x 224`.
- files: `Makefile`, `.cache/resnet50_imagenet_v2.pt`
- verification: CPU forward pass completed with shape `(1, 1000)`.

- Added `torch.flatten` lowering to the traced-module converter and added a
  full pretrained ResNet-50 fixture.
- files: `tests/models/specs/module.py`, `tests/models/fixtures/resnet50.py`
- verification: generated a native ResNet-50 runner for `1 x 3 x 224 x 224`
  and `1 x 3 x 32 x 32`; both runner sources compiled with `nvcc`.

- Added unsigned 64-bit predicate comparisons (`setp.eq/ne/lt/le/gt/ge.u64`)
  and a focused raw-PTX regression.
- files: `src/exec/executor.cpp`, `tests/ptx/setp_u64_kernel.ptx`,
  `tests/ptx/setp_u64_exec_smoke.cpp`, `tests/CMakeLists.txt`
- verification: `ctest -R '^ptx.setp_u64_exec_smoke$'` passed.

- Added per-node progress output to generated traced-model runners.
- files: `tests/models/specs/_common.py`, `tests/models/specs/module.py`
- verification: generated runners compile; `models.resnet_stem.runner_sim`
  passed and the full runner prints one completed line per node, e.g.
  `completed 1/175 nodes (0.6%): conv1`.

- Added `resnet-cpu`, `resnet-gpu`, and `resnet-gpu-cudasim` Make targets.
- files: `Makefile`
- verification: `make resnet-cpu` and `make resnet-gpu` passed; the real-GPU
  run completed all 175 traced nodes and matched the CPU reference. The
  simulator target was wired and smoke-tested with a small input.

- Removed the explicit ResNet-50 weight download fixture. The model generator
  now relies solely on the `torchvision` auto-download fallback when
  `.cache/resnet50_imagenet_v2.pt` is absent.
- files: `tests/models/fixtures/resnet50.py`, `tests/CMakeLists.txt`,
  `tests/models/download_resnet50.py`
- verification: confirmed the fixture uses the existing `.cache` state dict
  when present and otherwise delegates to `ResNet50_Weights.IMAGENET1K_V2`.

- Added `mov.pred` execution support and a focused raw-PTX regression; the
  ResNet-50 simulator forward pass now runs through all 175 traced nodes.
- Made `resnet-gpu-cudasim` use host block-level parallelism by default so
  plain `make resnet-gpu-cudasim` finishes in a practical amount of time.
- files: `src/exec/executor.cpp`, `tests/ptx/mov_pred_kernel.ptx`,
  `tests/ptx/mov_pred_exec_smoke.cpp`, `tests/CMakeLists.txt`, `Makefile`
- verification: `make resnet-gpu-cudasim` passed and the simulator output
  matched the PyTorch CPU reference (1000 floats); both PTX regression tests
  passed.

## Verification

Commands run:

```bash
make resnet-gpu-cudasim
ctest --test-dir build -R '^ptx\.(mov_pred|setp_u64)_exec_smoke$' --output-on-failure
ctest --test-dir build -R '^models\.resnet_stem\.runner_sim$' --output-on-failure
```

Result:

```text
[model] completed 175/175 nodes (99.4%): fc
PyTorch CPU reference matches (1000 floats)
module simulator mode passed
100% tests passed, 0 tests failed out of 2
100% tests passed, 0 tests failed out of 1
```

The small-shape simulator forward pass and the relevant PTX regressions are
green. Full `1 x 3 x 224 x 224` simulator execution is now green after the
Step 11 native backend landed; it completes in 247.70s and matches the CPU
reference. The `models.resnet50.runner_(cpu|sim)` CTest targets are still
covered by the Makefile model targets.

## Status

complete
