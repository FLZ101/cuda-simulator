# CUDA Simulator for Externally Generated CUDA Programs

## 1. Purpose

Build a CPU-hosted simulator that runs small, externally generated CUDA
programs without an NVIDIA GPU.

The simulator is not a model runtime. It intercepts CUDA runtime and driver
calls, loads PTX, and executes kernels on the host CPU.

End-to-end v1 targets:

- ResNet-18 or ResNet-50
- GPT-2 124M

Both targets are forward-pass correctness workloads. Their operators are
generated externally; the simulator only executes the resulting PTX.

## 2. v1 Boundaries

These are fixed v1 decisions.

- Run a standalone CUDA host executable or shared library.
- Do not run a full PyTorch/Python process.
- Use `LD_PRELOAD` on the generated host program.
- Prefer dynamically linked CUDA runtime. Static `libcudart` is not a target.
- Accept PTX only. Reject CUBIN/SASS.
- Do not implement cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, or model operators.
- Do not support CUDA Graphs, stream-ordered memory allocation, textures, or
  surfaces.
- Do not support tensor-core instructions: `mma`, `wmma`, `ldmatrix`,
  `stmatrix`, `wgmma`, or `tcgen05`. Generated kernels must use SIMT.
- Allow `cp.async` only as a synchronous copy emulation.
- Support one simulated device and one primary context.

Host assumptions:

- Linux, little-endian.
- x86-64 first, aarch64 allowed.
- LP64 host data model.
- C++20.
- PTX ISA 7.x and 8.x.
- Generated targets: `sm_70`, `sm_75`, `sm_80`, or `sm_86`, with PTX embedded.

## 3. How a Program Runs

```text
external compiler
  nvcc / PyTorch extension / Triton / TVM
              |
              v
  host executable or shared library
  containing host code + PTX/fatbin
              |
              v
LD_PRELOAD=libcudart_shim.so:libcuda_shim.so ./program
              |
              v
  shim intercepts CUDA calls
              |
              v
  load PTX
              |
              v
  execute SIMT kernels on CPU
```

Example:

```bash
LD_PRELOAD=/abs/path/libcudart_shim.so:/abs/path/libcuda_shim.so \
  ./generated_resnet_program
```

The build installs compatibility symlinks such as `libcudart.so`,
`libcuda.so`, `libcudart.so.12`, and `libcuda.so.1`. This prevents a versioned
program from loading a real driver through a versioned SONAME.

## 4. Entry Forms

### Runtime API program

`nvcc` emits registration hooks before `main`:

```text
__cudaRegisterFatBinary
__cudaRegisterFunction
__cudaRegisterVar
__cudaRegisterManagedVar
__cudaRegisterFatBinaryEnd
__cudaUnregisterFatBinary
__cudaRegisterLinkedBinary
__cudaRegisterSurface
__cudaRegisterTexture
```

The fatbin handle from `__cudaRegisterFatBinary` is the primary ingestion
path. Function registration maps a host function pointer to a PTX function.
Surface/texture hooks are stubbed with a clear error.

### Driver API program

The program calls:

```text
cuModuleLoad
cuModuleLoadData
cuModuleLoadDataEx
cuModuleLoadFatBinary
```

The module data is parsed as either fatbin or raw PTX.

### Raw PTX runner

For testing without a host program:

```bash
cuda_sim_run \
  --module kernel.ptx \
  --entry kernel_name \
  --grid 32,4 \
  --block 128
```

## 5. System Components

| Component | Responsibility |
| --- | --- |
| `compat` | Export `libcudart` and `libcuda` symbols and forward to core |
| `loader` | Parse fatbin data, optional ELF inspection, module registry |
| `ptx` | Lex, parse, lower to IR |
| `runtime` | Devices, contexts, streams, events, memory |
| `exec` | Blocks, warps, SIMT instructions, divergence, barriers |
| `mem` | Device memory, address translation, coalescing, atomics |
| `jit` | Optional LLVM JIT for hot PTX functions |

## 6. Minimum CUDA API Surface

### Runtime API

```text
cudaGetDeviceCount
cudaSetDevice
cudaGetDevice
cudaGetDeviceProperties
cudaDeviceGetAttribute

cudaMalloc
cudaFree
cudaMemcpy
cudaMemcpyAsync
cudaMemcpy2D
cudaMemcpy2DAsync
cudaMemset
cudaMemsetAsync
cudaMemset2D
cudaMemset2DAsync

cudaMallocManaged
cudaHostAlloc
cudaFreeHost
cudaHostRegister
cudaHostUnregister
cudaHostGetDevicePointer

cudaMemcpyToSymbol
cudaMemcpyFromSymbol
cudaGetSymbolAddress

cudaStreamCreate
cudaStreamCreateWithFlags
cudaStreamCreateWithPriority
cudaStreamDestroy
cudaStreamSynchronize
cudaStreamQuery
cudaStreamWaitEvent

cudaEventCreate
cudaEventCreateWithFlags
cudaEventDestroy
cudaEventRecord
cudaEventSynchronize
cudaEventQuery
cudaEventElapsedTime

cudaLaunchKernel
cudaFuncGetAttributes
cudaFuncSetAttribute
cudaOccupancyMaxActiveBlocksPerMultiprocessor

cudaDeviceSynchronize
cudaDeviceReset
cudaGetLastError
cudaGetErrorString
```

### Driver API

```text
cuInit
cuDeviceGetCount
cuDeviceGet
cuDeviceGetName
cuDeviceGetAttribute

cuDevicePrimaryCtxRetain
cuDevicePrimaryCtxRelease
cuCtxCreate
cuCtxDestroy
cuCtxSetCurrent
cuCtxGetCurrent
cuCtxGetDevice
cuCtxSynchronize

cuModuleLoad
cuModuleLoadData
cuModuleLoadDataEx
cuModuleLoadFatBinary
cuModuleGetFunction
cuModuleUnload

cuFuncGetAttribute
cuFuncSetAttribute
cuOccupancyMaxActiveBlocksPerMultiprocessor

cuMemAlloc
cuMemFree
cuMemAllocManaged
cuMemHostAlloc
cuMemFreeHost
cuMemGetInfo
cuPointerGetAttribute

cuMemcpyHtoD
cuMemcpyDtoH
cuMemcpyDtoD
cuMemcpyHtoDAsync
cuMemcpyDtoHAsync
cuMemcpyDtoDAsync
cuMemsetD8
cuMemsetD32

cuStreamCreate
cuStreamCreateWithPriority
cuStreamGetPriority
cuStreamDestroy
cuStreamSynchronize
cuStreamQuery
cuStreamWaitEvent

cuEventCreate
cuEventCreateWithFlags
cuEventDestroy
cuEventRecord
cuEventSynchronize
cuEventQuery
cuEventElapsedTime

cuLaunchKernel
```

The shim returns the correct CUDA error enum and keeps a per-thread last error
for runtime API calls. Unsupported APIs fail explicitly rather than crashing.

## 7. Execution Model

### Device and scheduler

One simulated device:

```text
compute capability: 8.0
SM count: configurable via CUDA_SIM_SM_COUNT
max threads per block: 1024
warp size: 32
shared memory per block: 49152
max block dim: (1024, 1024, 64)
max grid dim: (2147483647, 65535, 65535)
```

The device owns:

- global allocator
- block scheduler
- SM worker pool
- stream and event state

### Stream and command completion

A stream is an ordered queue of commands:

```text
kernel launch
host-to-device copy
device-to-host copy
device-to-device copy
memset
event record
event wait
```

The central invariant:

```text
kernel command complete
    <=>
all thread blocks in its grid have finished
```

No event after a kernel may be observed before that condition is true.

The legacy default stream synchronizes conservatively with all other streams.
This is correctness-first; generated programs should use non-default streams
if concurrency matters.

### Kernel launch

On `cudaLaunchKernel` or `cuLaunchKernel`:

1. Resolve the PTX function.
2. Validate grid, block, shared-memory size, and arguments.
3. Pack kernel arguments into the `.param` address space.
4. Enqueue a kernel command on the target stream.
5. The scheduler creates thread blocks.
6. Assign blocks to SM workers.
7. Each block executes until all threads exit.
8. Signal the originating stream only after all blocks complete.

Kernel arguments are packed with host order, size, and alignment. Host pointers
are translated to device addresses before the kernel sees them.

### Thread and warp state

A block stores:

- `blockIdx`
- grid and block dimensions
- warps
- shared memory
- barrier state

A warp stores:

- program counter
- active lane mask
- predicates
- register file
- reconvergence stack
- call stack

Registers use a structure-of-arrays layout:

```text
register R -> lane[0], lane[1], ..., lane[31]
```

This layout is required for later vectorized/JIT execution.

### Warp scheduling and barriers

Warps cannot run independently to completion when `bar.sync` exists. Each block
uses a runnable-warp queue:

1. Execute a bounded instruction quantum per warp.
2. Suspend a warp when it reaches a barrier.
3. Resume warps when the barrier arrival count is satisfied.
4. Finish the block when no warp is runnable.

Named barriers and `bar.arrive`/`bar.sync` are implemented when emitted. A
barrier in a divergent region is rejected with a precise diagnostic.

### Divergence

On a divergent branch:

1. Save the original active mask and reconvergence point.
2. Execute the taken path.
3. Execute the fall-through path.
4. Restore the active mask at reconvergence.

Use a structured reconvergence stack. `exit` terminates only the executing
threads; the block completes when all threads have exited.

### Special registers

```text
%tid.x/y/z
%ntid.x/y/z
%ctaid.x/y/z
%nctaid.x/y/z
%laneid
%warpid
%smid
%nsmid
%clock
%clock64
%globaltimer
```

`%clock` and `%globaltimer` must be non-decreasing; cycle accuracy is not
required.

## 8. Memory Model

Logical spaces:

| Space | Implementation |
| --- | --- |
| Global | Host-backed device buffer |
| Shared | Per-block byte buffer |
| Constant | Per-module read-only buffer |
| Local | Per-thread spill area |
| Parameter | Per-launch argument buffer |
| Generic | Address-range lookup into one of the above |

Track address ranges for global, shared, local, constant, and generic spaces.
Bounds-check every access. Invalid accesses produce a CUDA-style error, not a
host segfault.

For each warp memory instruction:

1. Compute all 32 effective addresses.
2. Classify as uniform, contiguous, strided, or divergent.
3. Use broadcast, wide load/store, strided access, or gather/scatter.

Support `.v2`, `.v4`, and common cache-operator suffixes. In v1, cache
operators such as `.ca`, `.cg`, `.nc`, and `.wt` may be treated as ordinary
memory accesses.

Atomics:

```text
atom.add
atom.cas
atom.exch
atom.max
atom.min
atom.and
atom.or
atom.xor
```

Support 32-bit and 64-bit integer and floating-point atomics as required by
generated code. Shared-memory atomics use per-block synchronization. Global
atomics use host atomic operations.

## 9. PTX Frontend and IR

Frontend stages:

```text
PTX text -> lexer -> parser -> AST -> lower -> simulator IR
```

The parser preserves:

- target and address-size directives
- global/constant/shared declarations
- function declarations and definitions
- parameter and return types
- labels and predicates
- negated predicates such as `@!p0`
- address-space annotations
- debug file/line directives

The lowerer:

- splits functions into basic blocks
- resolves labels
- normalizes operands and type suffixes
- identifies variable declarations and sizes
- keeps source locations for diagnostics

The interpreter and JIT consume the IR, never the AST directly.

## 10. PTX Instruction Coverage

### Initial subset for vector add

```text
ld.param
ld.global
st.global
add
sub
mul
mad
div
rem
shl
shr
and
or
xor
not
setp
@p bra
@!p bra
cvt
mov
ret
exit
```

### Core generated-kernel subset

```text
ld.global, st.global
ld.global.nc
ld.shared, st.shared
ld.local, st.local
ld.const
ld.global.v2, ld.global.v4
st.global.v2, st.global.v4
ld.shared.v2, ld.shared.v4
st.shared.v2, st.shared.v4
cvta.to.global/shared/local/const
atom.global
atom.shared
red.global
red.shared
bar.sync
bar.arrive
shfl.up/down/bfly/idx
vote.all/any/ballot
```

### Types

```text
FP64
FP32
FP16
BF16
INT8
INT32
INT64
UINT8
UINT32
UINT64
packed f16x2 / bf16x2
```

### Required generated-operator categories

```text
integer and pointer arithmetic
floating-point arithmetic
min/max/abs/neg
shifts and bitwise operations
funnel shifts
select/selp
sqrt/rsqrt/exp2/log2/sin/cos
fma
predication
vectorized loads/stores
address-space casts
atomics
barriers
warp reductions
function calls
```

### Tensor-core instructions are out of scope

Do not support:

```text
mma
wmma
ldmatrix
stmatrix
wgmma
tcgen05
```

Avoid them at the generator:

- Custom CUDA: no WMMA/MMA APIs, no inline `mma.sync`, no `ldmatrix`.
- Triton: avoid `tl.dot`; implement matmul with explicit loads/stores and
  accumulation.
- TorchInductor: do not compile whole-model matmul/attention with `tl.dot`.
- Set `TRITON_F32_DEFAULT=ieee` as a secondary guard.

CI gate:

```bash
grep -E '\b(mma|wmma|ldmatrix|stmatrix|wgmma|tcgen05)\b' kernel.ptx
```

For a fatbin, extract PTX first with `cuobjdump --dump-ptx`.

### Math policy

- Non-approximate operations use round-to-nearest-even unless the instruction
  suffix selects another rounding mode.
- Approximate operations such as `ex2.approx`, `lg2.approx`, `rsqrt.approx`,
  `rcp.approx`, `sin.approx`, and `cos.approx` may start with host libm. If
  differential tests fail, implement the NVIDIA-style approximation.

## 11. Implementation Plan

Each step ends with a concrete verification target.

### Step 0: Build and Test Skeleton

Create:

```text
CMakeLists.txt
cmake/FindLLVM.cmake
cmake/CompilerWarnings.cmake
cmake/SimulatorOptions.cmake
src/version.h.in
tests/CMakeLists.txt
tests/smoke/vector_add_ref.cpp
README.md
```

Decisions:

- C++20, CMake 3.24 or newer.
- `-Wall -Wextra -Wpedantic`, warnings are errors.
- LLVM JIT is optional via `CUDA_SIM_ENABLE_JIT`.
- Outputs: `libcudart_shim.so`, `libcuda_shim.so`, `libcudasim_core.so`,
  `cuda_sim_run`, `cuda_sim_tests`.

Verify:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Step 1: CUDA Shims

Build `libcudart_shim.so` and `libcuda_shim.so`.

Files:

```text
src/compat/export.h
src/compat/cudart_shim.cpp
src/compat/cuda_shim.cpp
src/compat/abi_map.h
src/compat/abi_map.cpp
src/compat/error_codes.h
```

Implement the API surface from Section 6 and the registration hooks from
Section 4.

Verify:

- Compile a small `nvcc` runtime-API program.
- Run it under `LD_PRELOAD`.
- Compile and run the equivalent driver-API program.
- Confirm shim symbols with `nm -D`.
- Verify `LD_DEBUG=bindings` selects the shim.

Exit: a CUDA program can initialize device 0, allocate memory, and free memory
without a real driver.

### Step 2: Runtime State

Files:

```text
src/runtime/device.*
src/runtime/context.*
src/runtime/stream.*
src/runtime/event.*
src/runtime/memory.*
src/runtime/errors.h
src/runtime/launch.h
```

Implement:

- one device and one primary context
- host-backed device allocations with size/alignment/type tracking
- managed memory as the same host pointer
- ordered streams and event markers
- thread-safe allocator and stream queues

Verify:

- allocate/copy/memset/free
- two-stream asynchronous copies
- cross-stream event waits
- `cudaDeviceSynchronize`

### Step 3: PTX/Fatbin Ingestion and IR

Files:

```text
src/ptx/token.*
src/ptx/lexer.*
src/ptx/ast.*
src/ptx/parser.*
src/ptx/ir.*
src/ptx/lower.*
src/ptx/module.*
src/loader/fatbin.*
src/loader/elf.*
src/loader/module_registry.*
```

Implement:

- PTX lexer/parser for ISA 7.x and 8.x
- AST-to-IR lowering
- fatbin or raw PTX detection
- module registry scoped by fatbin/module handle
- registration hook ingestion

ELF parsing is optional and mainly for offline inspection. The registration
hooks and module-load APIs are the primary ingestion path.

Verify:

- `nvcc -ptx` output parses
- AST/IR round-trip test passes
- runtime and driver ingestion both register vector add
- CUBIN-only module reports a clear unsupported error

### Step 4: Minimal SIMT Interpreter

Files:

```text
src/exec/launch_config.h
src/exec/grid.*
src/exec/block.*
src/exec/warp.*
src/exec/register_file.h
src/exec/instruction.h
src/exec/executor.*
src/exec/op_fp32.h
src/exec/op_int.h
src/exec/op_memory.h
```

Implement:

- kernel argument packing
- grid/block/warp construction
- SoA register file
- special-register initialization
- runnable-warp queue
- scalar instruction dispatch

Target kernel:

```cuda
__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

Test sizes:

```text
0, 1, 31, 32, 33, 1023, 1024, 4097
```

Exit: the generated vector-add program runs under `LD_PRELOAD` and matches a
CPU reference.

### Step 5: Full Execution Semantics

Add:

- divergent branches and reconvergence
- predicates
- direct function calls
- shared memory
- local memory
- constant memory
- `bar.sync` and named barriers
- address-space casts

Files:

```text
src/exec/divergence.*
src/exec/call_stack.*
src/exec/barrier.*
src/exec/local_memory.h
src/exec/shared_memory.h
src/exec/constant_memory.h
src/exec/predicate.h
src/exec/branch.h
src/exec/address_space.h
src/exec/special_registers.h
```

Verify with microkernels:

- nested branches and loops
- function calls
- shared-memory reduction
- dynamic shared memory
- atomics
- local-memory spills

### Step 6: Memory Model and Coalescing

Files:

```text
src/mem/device_memory.*
src/mem/address_range.h
src/mem/coalescer.*
src/mem/atomics.*
src/mem/access_pattern.*
```

Implement:

- effective-address computation
- uniform/contiguous/strided/divergent classification
- wide host accesses for coalesced warps
- gather/scatter for divergent accesses
- bounds checking
- shared/global atomics

Verify:

- aligned and unaligned `.v2`/`.v4`
- coalesced and divergent patterns
- contended atomics
- producer/consumer shared memory with fence/barrier

### Step 7: Generated PTX Coverage and SIMT Gate

Generate and inspect kernels for:

```text
elementwise
Conv2d
GEMM
reduction
softmax
layer norm
attention
embedding
transpose/permute
copy/cast
```

Implement instructions on demand: run the kernel, record the first unsupported
instruction, add it with type suffixes, and repeat.

Run the tensor-core PTX gate from Section 10 in CI.

Exit: each target operator has at least one generated kernel that passes.

### Step 8: Differential Test Harness

Create fixture directories, not bare PTX files:

```text
tests/fixtures/<operator>/
  manifest.json
  kernel.ptx
  host_runner
```

`manifest.json` records:

- input/output shape and dtype
- reference function
- launch metadata
- tolerances
- generation command

Harness behavior:

1. Run generated host program under `LD_PRELOAD`.
2. Run PyTorch CPU reference.
3. Compare tensors.
4. Report max absolute/relative error.

Tolerances:

```text
FP32: rtol = 1e-4, atol = 1e-5
FP16/BF16: rtol = 5e-2, atol = 1e-2
integer: exact
```

### Step 9: ResNet End-to-End

Generate a standalone `resnet50_runner` host executable.

Constraints:

- no cuDNN, cuBLAS, or cuFFT
- all Conv/BatchNorm/ReLU/pool/linear operations are explicit PTX kernels
- PTX must pass the SIMT-only gate

Runner contract:

- load deterministic weights and one input tensor
- allocate device memory
- launch all forward-pass kernels
- copy final logits to host
- write output

Reference:

```text
model.eval()
torch.manual_seed(1234)
input = deterministic 3 x 224 x 224 tensor
compare final logits and predicted class
```

Incremental checkpoints:

1. First convolution block.
2. First residual block.
3. Stage 1.
4. All stages.
5. Full network.

Verification:

```bash
LD_PRELOAD=/abs/path/libcudart_shim.so:/abs/path/libcuda_shim.so \
  ./resnet50_runner --weights resnet50.pt --input input.raw --output out.raw

python3 tests/models/resnet_compare.py \
  --simulator-out out.raw \
  --reference-out ref.raw
```

Outputs are FP32 C-order with a shape/dtype sidecar.

### Step 10: GPT-2 End-to-End

Generate a standalone `gpt2_runner` host executable.

Target config:

```text
vocab_size = 50257
n_layer = 12
n_head = 12
n_embd = 768
context_length = 1024
```

Runner contract:

- load deterministic weights and token IDs
- launch embedding, attention, MLP, layer norm, residual, and LM-head kernels
- copy final logits to host
- write output

All operators are explicit PTX. No cuBLAS/cuDNN/cuFFT.

Reference:

```text
Hugging Face gpt2 or minimal equivalent
model.eval()
token_ids = deterministic sequence
```

Incremental checkpoints:

1. Embedding and one attention head.
2. Full attention.
3. One transformer block.
4. Full 12-block model.

Verification:

```bash
LD_PRELOAD=/abs/path/libcudart_shim.so:/abs/path/libcuda_shim.so \
  ./gpt2_runner --weights gpt2-124m.pt --tokens tokens.bin --output logits.raw

python3 tests/models/gpt2_compare.py \
  --simulator-out logits.raw \
  --reference-out reference_logits.raw \
  --check-argmax
```

Logits are FP32 C-order with shape `(sequence_length, vocab_size)`.

### Step 11: Reduced Precision, JIT, and Multi-SM

Add:

- FP16/BF16 host types and conversions
- packed `f16x2`/`bf16x2`
- LLVM OrcJIT lowering
- function cache
- interpreter fallback
- global block scheduler with SM worker threads

Files:

```text
src/exec/half.h
src/exec/bfloat16.h
src/ptx/lower.*
src/jit/llvm_backend.*
src/jit/vector_types.h
src/jit/function_cache.*
```

JIT rule:

- The `<32 x type>` lane mapping is the fast path for warp-uniform control flow.
- Divergent branches, barriers, and unsupported instructions fall back to the
  interpreter.

Multi-SM rule:

```text
grid -> queued blocks -> ready blocks -> SM worker pool -> completed blocks
```

Non-blocking planning targets on 8 CPU cores:

```text
ResNet-18, batch 1, FP32: < 120 seconds
GPT-2 124M, 16 tokens, FP32: < 600 seconds
```

These are planning numbers, not correctness gates.

### Step 12: Compatibility Hardening

Continuously:

1. Run a generated artifact.
2. On unsupported API/PTX, add a minimal reproduction.
3. Implement the smallest missing feature.
4. Add the reproduction to CI.
5. Re-run the corpus.

Track:

```text
passing generated programs
distinct PTX opcodes supported
runtime API symbols implemented
driver API symbols implemented
model end-to-end tests passing
```

## 12. Verification Summary

Bottom-up:

1. Component tests for shims, memory, streams, events, and PTX parsing.
2. Kernel differential tests against PyTorch CPU.
3. Operator tests for Conv2d, GEMM, elementwise, reduction, softmax, layer
   norm, and attention.
4. End-to-end ResNet and GPT-2 tests.
5. Runtime trace tests comparing kernel launch order and copy sizes.

## 13. Project Structure

```text
cuda-simulator/
  CMakeLists.txt
  DESIGN.md
  src/
    compat/
      cudart_shim.cpp
      cuda_shim.cpp
    loader/
      elf.cpp
      fatbin.cpp
      module_registry.cpp
    ptx/
      lexer.cpp
      parser.cpp
      ir.cpp
      lower.cpp
    runtime/
      device.cpp
      context.cpp
      stream.cpp
      event.cpp
      memory.cpp
    exec/
      sm.cpp
      warp.cpp
      block.cpp
      scheduler.cpp
      divergence.cpp
      barrier.cpp
      intrinsics.cpp
    mem/
      device_memory.cpp
      coalescer.cpp
      atomics.cpp
    jit/
      llvm_backend.cpp
      function_cache.cpp
  tests/
    generated/
    kernels/
    models/
    differential/
    fixtures/
```

## 14. Key Risks

- Generated artifacts may be CUBIN-only.
- Triton/TorchInductor may emit tensor-core or newer PTX instructions.
- `LD_PRELOAD` behavior must be verified against versioned CUDA SONAMEs.
- Fatbin layout varies across CUDA toolchains.
- NVIDIA approximate math can differ from host libm.
- A naive interpreter will be too slow; JIT and multi-SM are required.
