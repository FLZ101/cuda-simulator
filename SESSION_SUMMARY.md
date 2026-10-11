# cuda-simulator Session Summary

## Goal

Build a CPU-hosted PTX simulator for externally generated CUDA programs.

v1 correctness targets:

- ResNet-18 or ResNet-50 forward pass
- GPT-2 124M forward pass

The simulator executes PTX only. It does not execute CUBIN/SASS.

## Current State

The real-GPU differential harness is implemented and covers a broad operator
corpus. Generated kernels are run on the real GPU as golden references and then
run through the simulator shims for comparison.

Build:

```bash
make build
make test
```

Python formatting and pre-commit:

```bash
make format-python
make precommit
```

## Reference Policy

- Hand-written `nvcc` CUDA kernels use real GPU output as the golden reference.
- PyTorch-generated model runners use PyTorch CPU reference.
- Early smoke tests may use a host CPU reference.

## Operator Harness

Enabled operators:

- elementwise: `copy`, `saxpy`, `relu`, `gelu`, `elementwise_add`,
  `elementwise_mul`, `scale`, `neg`, `square`, `tanh`, `sigmoid`, `silu`,
  `reciprocal`, `exp`, `log`, `sqrt`, `pow`, `rsqrt`, `minimum`, `maximum`,
  `leaky_relu`, `abs`, `relu6`, `softplus`, `swiglu`, `elu`, `exp2`, `log2`,
  `floor`, `ceil`, `rint`, `trunc`, `acos`, `erfc`, `hard_sigmoid`,
  `hardswish`, `softsign`, `log_sigmoid`, `subtract`, `divide`, `copysign`,
  `lerp`, `cosh`, `sinh`, `asin`, `atan`, `cbrt`, `mish`, `expm1`, `atan2`,
  `hypot`
- normalization: `batch_norm`, `layer_norm`, `rms_norm`, `instance_norm`,
  `local_response_norm`, `fused_add_norm`
- reduction: `reduce_sum`, `mean`, `argmax`, `global_avg_pool`, `cumsum`
- matrix: `matmul_naive`, `matmul_tiled`, `transpose`
- convolution/pooling: `conv2d_naive`, `conv2d_im2col`, `conv2d_shared`,
  `max_pool2d`, `avg_pool2d`, `depthwise_conv2d`, `adaptive_avg_pool2d`
- transformer: `embedding_lookup`, `attention_scores`, `softmax_row`,
  `log_softmax`, `causal_mask`, `dropout`, `masked_fill`, `alibi_bias`,
  `repeat_kv`
- data movement/reshape helpers: `concat`, `split`, `clamp`,
  `cross_entropy`, `residual_add`, `pad2d`, `upsample_nearest2d`,
  `channel_shuffle`, `pixel_shuffle`

`pow` is enabled. Its optimized NVVM path required additional rounding,
conversion, and unordered-comparison opcodes (`add.rn.f32`, `mul.rn.f32`,
`cvt.rn.f32.u32`, `cvt.rzi.f32.f32`, `xor.b32`, and `setp.*u.f32`).
The adaptive-pooling and LRN kernels additionally required `not.b32` and
`neg.s32` for NVVM's vectorized loop epilogues and signed window bounds.

## Important Implementation Details

- Register banks are separated:
  - `r` base `0`
  - `rd` base `256`
  - `f` base `512`
- Special registers support `x`, `y`, and `z` components.
- `bar.sync` uses block-wide lane arrival tracking.
- Dynamic `.extern .shared` arrays are parsed and sized from
  `config.shared_memory`.
- Global memory offset parsing supports signed offsets.
- FMA instructions use `std::fmaf`.
- `cvt.sat.f32.f32` clamps to `[0.0, 1.0]`.
- Tensor-core instructions are rejected by `tests/generated/check_simt.py`.

## Key Files

- `DESIGN.md`
- `AGENTS.md`
- `steps/STEP_08.md`
- `src/exec/executor.cpp`
- `src/exec/register_file.h`
- `src/ptx/parser.cpp`
- `tests/reference/gpu_reference_runner.cu`
- `tests/reference/run_gpu_reference.py`
- `tests/CMakeLists.txt`

## Next Steps

1. Generate ResNet-style PyTorch runner and compare against PyTorch CPU.
2. Generate GPT-2-style PyTorch runner and compare against PyTorch CPU.
3. Add remaining missing PTX semantics as generated model runners expose them.
