# cuda-simulator Session Summary

## Goal

Build a CPU-hosted PTX simulator for externally generated CUDA programs.

v1 correctness targets:

- ResNet-18 or ResNet-50 forward pass
- GPT-2 124M forward pass

The simulator executes PTX only. It does not execute CUBIN/SASS.

## Current State

The real-GPU differential harness is implemented and covers a broad operator
corpus. Generated kernels are grouped under `tests/kernels/<group>/`, and each
group has its own CTest target. Kernels are run on the real GPU as golden
references and then run through the simulator shims for comparison.

PyTorch-generated models are converted into runners through
`tests/models/model_runner.py`. Each spec emits a native CUDA runner with
embedded Triton PTX; `--mode cpu` runs the PyTorch model on CPU, while
`--mode gpu --simulate` runs the native runner under the CUDA shims. The runner
never takes PTX text as input.

The traced-module spec currently lowers `nn.Linear`, `nn.Conv2d`, `nn.ReLU`,
`nn.BatchNorm2d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`,
`nn.GELU`, `nn.SiLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.LeakyReLU`, `nn.Flatten`,
`nn.Dropout`, `nn.Identity`, `nn.LayerNorm`, last-dim `nn.Softmax`, and
graph-level `operator.add` / `operator.mul` binaries. ResNet-stem, CNN-ops,
and LLM-ops fixtures cover the generated execution paths.
Additional activations include `nn.Hardswish`, `nn.Hardsigmoid`,
`nn.ReLU6`, `nn.Softplus`, `nn.ELU`, `nn.SELU`, `nn.CELU`, `nn.Mish`,
`nn.Hardtanh`, `nn.Softsign`, and `nn.LogSigmoid`; CNN/LLM helpers also cover
`nn.AdaptiveMaxPool2d((1, 1))`, `nn.LogSoftmax`, and last-dim `torch.cat`.

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

Enabled groups (each passes independently against the real GPU):

- `elementwise`
- `convolution`
- `normalization`
- `reduction`
- `attention`
- `indexing`
- `loss`
- `matrix`
- `misc`
- `detection`
- `sequence`

Legacy enabled-operator breakdown:

- elementwise: `copy`, `saxpy`, `relu`, `gelu`, `elementwise_add`,
  `elementwise_mul`, `scale`, `neg`, `square`, `tanh`, `sigmoid`, `silu`,
  `reciprocal`, `exp`, `log`, `sqrt`, `pow`, `rsqrt`, `minimum`, `maximum`,
  `leaky_relu`, `abs`, `relu6`, `softplus`, `swiglu`, `elu`, `exp2`, `log2`,
  `floor`, `ceil`, `rint`, `trunc`, `acos`, `erfc`, `hard_sigmoid`,
  `hardswish`, `softsign`, `log_sigmoid`, `subtract`, `divide`, `copysign`,
  `lerp`, `cosh`, `sinh`, `asin`, `atan`, `cbrt`, `mish`, `expm1`, `atan2`,
  `hypot`
- normalization: `batch_norm`, `layer_norm`, `rms_norm`, `instance_norm`,
  `local_response_norm`, `fused_add_norm`, `group_norm`
- reduction: `reduce_sum`, `mean`, `argmax`, `global_avg_pool`, `cumsum`
- matrix: `matmul_naive`, `matmul_tiled`, `transpose`
- convolution/pooling: `conv2d_naive`, `conv2d_im2col`, `conv2d_shared`,
  `max_pool2d`, `avg_pool2d`, `depthwise_conv2d`, `adaptive_avg_pool2d`,
  `adaptive_max_pool2d`, `conv_transpose2d`, `upsample_bilinear2d`,
  `reflection_pad2d`
- transformer: `embedding_lookup`, `attention_scores`, `softmax_row`,
  `log_softmax`, `causal_mask`, `dropout`, `masked_fill`, `alibi_bias`,
  `repeat_kv`, `rotary_embedding`
- data movement/reshape helpers: `concat`, `split`, `clamp`,
  `cross_entropy`, `residual_add`, `pad2d`, `upsample_nearest2d`,
  `channel_shuffle`, `pixel_shuffle`, `pixel_unshuffle`, `gather`,
  `scatter_add`

`pow` is enabled. Its optimized NVVM path required additional rounding,
conversion, and unordered-comparison opcodes (`add.rn.f32`, `mul.rn.f32`,
`cvt.rn.f32.u32`, `cvt.rzi.f32.f32`, `xor.b32`, and `setp.*u.f32`).
The adaptive-pooling and LRN kernels additionally required `not.b32` and
`neg.s32` for NVVM's vectorized loop epilogues and signed window bounds.
Bilinear upsample requires `cvt.rzi.s32.f32` for float-to-int truncation.
Scatter-add uses the PTX 8.x spelling `atom.global.add.f32`, which is handled
as an alias of `atom.add.global.f32`.
Attention kernels additionally required `setp.eq.s64`, `setp.ne.s64`,
`abs.s32`, and `cos.approx.f32`.
Detection adds `nms`, `iou`, `ciou`, `roi_align`, `box_decode`, and
`box_encode`. Sequence adds `rnn_cell`, `lstm_cell`, `gru_cell`,
`sequence_mask`, `beam_search`, and `ctc_decode`.

Additional qualification-matrix operators are enabled in each group:

- convolution: `conv1d`, `conv3d`, `fold`, `unfold`, `deform_conv2d`
- normalization: `weight_norm`, `spectral_norm`
- reduction: `median`, `quantile`, `mode`, `searchsorted`
- attention: `cross_attention`, `local_attention`
- indexing: `permute`, `reshape`, `squeeze`, `unsqueeze`, `chunk`, `unbind`,
  `meshgrid`, `swapaxes`
- loss: `ctc_loss`, `gaussian_nll_loss`, `multilabel_margin_loss`,
  `multimargin_loss`
- matrix: `tensordot`, `kron`, `mv`, `bmm`
- detection: `diou`, `giou`, `anchor_generate`, `batched_nms`,
  `generate_proposals`
- sequence: `ctc_prefix_beam_search`, `attention_decoder`,
  `transformer_decoder`

`permute` keeps its full compiler unrolling. The interpreter register file now
uses 1024 entries per bank, with `rd` and `f` banks starting after the `r`
bank, so generated indices such as `r716` do not alias `rd` or `f` registers.

## Important Implementation Details

- Register banks are separated:
  - `r` base `0`
  - `rd` base `1024`
  - `f` base `2048`
- PTX `%r`, `%rd`, and `%f` operands are virtual register names, not physical
  GPU registers. The interpreter keeps each virtual bank in a disjoint range
  and does not apply the hardware 255-register-per-thread limit to them.
- Modern NVIDIA SMs have 64K 32-bit physical registers (256 KB) per SM and a
  255-register-per-thread limit. This physical capacity does not bound the
  virtual register indices emitted in PTX.
- Special registers support `x`, `y`, and `z` components.
- `bar.sync` uses block-wide lane arrival tracking.
- Dynamic `.extern .shared` arrays are parsed and sized from
  `config.shared_memory`.
- Global memory offset parsing supports signed offsets.
- FMA instructions use `std::fmaf`.
- `cvt.sat.f32.f32` clamps to `[0.0, 1.0]`.
- Tensor-core instructions are rejected by `tests/generated/check_simt.py`.
- Triton-generated PTX uses embedded PTX in the generated native runner; model
  PTX is never passed to a runner CLI argument.

## Key Files

- `DESIGN.md`
- `AGENTS.md`
- `steps/STEP_08.md`
- `src/exec/executor.cpp`
- `src/exec/register_file.h`
- `src/ptx/parser.cpp`
- `tests/reference/gpu_reference_runner.cu`
- `tests/reference/run_gpu_reference.py`
- `tests/models/model_runner.py`
- `tests/models/generate_model_runner.py`
- `tests/models/specs/mlp.py`
- `tests/models/specs/module.py`
- `tests/models/fixtures/mlp_module.py`
- `tests/models/fixtures/conv_module.py`
- `tests/models/fixtures/resnet_stem.py`
- `tests/models/fixtures/cnn_ops.py`
- `tests/models/fixtures/llm_ops.py`
- `tests/ptx/scalar_math_kernel.ptx`
- `tests/ptx/scalar_math_exec_smoke.cpp`
- `tests/CMakeLists.txt`

## Next Steps

1. Replace fixed virtual-register banks with a `.reg`-declaration-driven
   `RegisterLayout` after a generated kernel exceeds the current 1024-per-bank
   capacity.
2. Extend the model-runner converter to lower full traced PyTorch graphs
   (ResNet-style and GPT-2-style) to native runners with embedded SIMT PTX.
3. Add remaining missing PTX semantics as generated model runners expose them.
