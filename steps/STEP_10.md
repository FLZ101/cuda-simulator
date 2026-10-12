# Step 10: GPT-2 End-to-End

## Objective

Generate a standalone `gpt2_runner` host executable whose forward pass is
implemented as embedded SIMT PTX kernels, run that runner through the
simulator shims, and compare the final FP32 logits and predicted token with a
deterministic PyTorch CPU reference.

Target configuration:

```text
vocab_size = 50257
n_layer = 12
n_head = 12
n_embd = 768
context_length = 1024
```

## Plan

- Reuse the Step 8/9 PyTorch-generated model-runner flow instead of introducing
  a separate GPT-2 runtime:

  ```text
  traced GPT-2 model
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

- Keep every generated kernel on the SIMT path. `compile_kernel` in
  `tests/models/specs/_common.py` must reject `mma`, `wmma`, `ldmatrix`,
  `stmatrix`, `wgmma`, and `tcgen05`.
- Extend the traced-module converter only for GPT-2 graph operators that are
  not already lowered. The existing converter already covers `Linear`,
  `LayerNorm`, `Softmax`, `GELU`, graph-level `operator.add`/`operator.mul`,
  last-dim `torch.matmul`, and `Tensor.transpose`.
- Add GPT-2 fixtures with the existing fixture contract:

  ```python
  def build_model(args): ...
  def example_input(args): ...
  ```

- Use a deterministic token sequence and `model.eval()`.
- Add `--check-argmax` support to the PyTorch comparison flow so GPT-2
  validation checks both logits and the predicted next token.
- Iterate on small configurations first (`n_layer`, `n_head`, `n_embd`, short
  token sequences), then run the final gate with the 124M configuration and a
  short deterministic token sequence.

## Files To Create

```text
steps/STEP_10.md
tests/models/fixtures/gpt2_embedding.py
tests/models/fixtures/gpt2_attention.py
tests/models/fixtures/gpt2_block.py
tests/models/fixtures/gpt2.py
tests/models/gpt2_compare.py
```

Likely files to modify:

```text
tests/models/specs/module.py
tests/models/model_runner.py
tests/models/compare_torch_reference.py
tests/CMakeLists.txt
Makefile
DESIGN.md
SESSION_SUMMARY.md
```

## Incremental Checkpoints

1. Token and position embedding lookup.
2. One attention head with Q/K/V projections, causal masking, scaled softmax,
   and output projection.
3. Full multi-head attention for one transformer block.
4. One GPT-2 transformer block with attention, MLP, layer norm, and residual
   paths.
5. Full 12-block GPT-2 124M model with LM head and deterministic token IDs.

## Acceptance Checks

```bash
cmake --build build -j
ctest --test-dir build -R 'models.gpt2.runner_(cpu|sim)' --output-on-failure
ctest --test-dir build --output-on-failure
```

The generated PTX must contain none of the banned tensor-core opcodes, the
simulator output must match the PyTorch CPU reference within the configured
FP32 tolerances, and the argmax of the final logits must match the reference.

Reference constraints:

- `model.eval()`
- `torch.manual_seed(1234)`
- deterministic token IDs
- identical weights, inputs, and evaluation mode for CPU and simulator runs

## Work Log

- Created the Step 10 plan and recorded the objective, implementation path,
  incremental checkpoints, files to create, and acceptance checks.
- files: `steps/STEP_10.md`
- verification: No build or test required.
- Lowered traced `nn.Embedding` modules to the generic FX converter and native
  runner. Token IDs are read as FP32 and converted to `int32` inside the
  Triton embedding kernel, avoiding a new integer-input path.
- files: `tests/models/specs/module.py`, `tests/CMakeLists.txt`,
  `tests/models/fixtures/gpt2_embedding.py`
- verification: `cmake --build build -j` and
  `ctest --test-dir build -R 'models\.gpt2_embedding\.runner_(cpu|sim)'
  --output-on-failure` both pass.
- Rebuilt and re-ran the existing traced-model runner tests after changing the
  shared converter.
- files: generated CTest model runners for `module`, `conv_module`,
  `resnet_stem`, `cnn_ops`, and `llm_ops`
- verification: all 10 CPU/sim model-runner tests pass.
- Added causal scaled softmax lowering and a single GPT-2 attention head
  fixture with bias-free Q/K/V/output projections. The attention graph uses
  existing `Linear`, `matmul`, and `transpose` lowering plus a new
  `causal_softmax` Triton kernel.
- files: `tests/models/specs/module.py`, `tests/CMakeLists.txt`,
  `tests/models/fixtures/gpt2_attention.py`
- verification: `cmake --build build -j` and
  `ctest --test-dir build -R 'models\.gpt2_attention\.runner_(cpu|sim)'
  --output-on-failure` both pass; all 14 traced model-runner tests pass.

## Verification

Checkpoints 1 and 2 pass:

```bash
ctest --test-dir build -R 'models\.gpt2_(embedding|attention)\.runner_(cpu|sim)' --output-on-failure
```

## Status

in progress
