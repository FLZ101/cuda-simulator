# CUDA Best Practices and Performance Tricks

This document lists practical CUDA performance rules for the generated
ResNet and GPT-2 kernels used by the simulator.

> Items marked **[simulator-specific]** apply to this project's v1 constraints.
> Other items are general CUDA performance practices.

## 1. Global memory access

### Coalesce accesses

Adjacent threads in a warp should access adjacent memory locations.

Good:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = in[i] * 2;
```

Poor:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i * 1024] = in[i * 1024] * 2;
```

Coalesced access allows the memory system to issue fewer, wider transactions.

### Use vectorized loads and stores

Use `float4`, `int4`, or equivalent vector types when data is contiguous.

```cuda
float4 v = reinterpret_cast<const float4 *>(in)[i];
```

This increases bytes moved per instruction.

Vectorized types usually require alignment:

```text
float4 requires 16-byte alignment
double2 requires 16-byte alignment
```

Check base pointers and index arithmetic before using vector types.

Use `__restrict__` and `const` when possible to give the compiler more
aliasing information:

```cuda
const float *__restrict__ input = in;
float *__restrict__ output = out;
```

### Use read-only data paths

Use `const __restrict__` pointers and `__ldg()` for read-only global data.

```cuda
const float *__restrict__ input = in;
float v = __ldg(&input[i]);
```

## 2. Shared memory

### Tile repeated data

For GEMM, convolution, and attention, load a tile into shared memory, then
reuse it from there.

```text
global memory -> shared memory tile -> registers -> math
```

This reduces global memory traffic.

Use shared memory only when the same data is reused multiple times. For a
single read, copying through shared memory adds overhead.

### Pipeline global-to-shared loads

For GEMM or attention loops, prefetch the next tile while computing the current
tile. This overlaps global memory latency with arithmetic.

### Avoid bank conflicts

Try to make threads in a warp access different shared-memory banks.

For a row-major array:

```cuda
shared[threadIdx.x] = value;
```

is usually conflict-free.

Strided access can cause conflicts:

```cuda
shared[threadIdx.x * 32] = value;
```

Add padding to shared-memory arrays when a fixed stride conflicts:

```cuda
__shared__ float tile[32][32 + 1];
```

The extra column changes the bank mapping for subsequent rows.

### Synchronize correctly

After writing shared memory and before reading it, call:

```cuda
__syncthreads();
```

Do not place `__syncthreads()` in divergent code.

## 3. Occupancy and block sizing

### Choose a good block size

Use a multiple of 32.

Common starting points:

```text
128 threads
256 threads
512 threads
```

Small blocks may underuse the SM. Very large blocks may reduce occupancy or
available registers.

### Balance resource usage

Occupancy is limited by:

```text
threads per SM
registers per SM
shared memory per SM
```

Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` or compiler flags to check
and tune.

Dynamic shared memory and register limits also affect occupancy:

```cuda
kernel<<<grid, block, dynamic_shared_bytes, stream>>>(...);
```

Compiler controls include:

```text
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
-maxrregcount=N
```

### Use enough blocks

Create enough thread blocks to occupy all SMs.

```text
target blocks >= SMs * blocks_per_SM
```

## 4. Control flow

### Avoid warp divergence

Keep branch conditions uniform across a warp when possible.

Prefer data-independent branches:

```cuda
if (blockIdx.x < limit) { ... }
```

Avoid branches based mainly on `threadIdx.x` with different long paths.

### Avoid integer division and modulo

Integer division and modulo are expensive.

Prefer bit operations for powers of two:

```cuda
int q = x / 32; // avoid
int q = x >> 5; // faster when 32 is a power of two
```

Precompute division results when possible.

### Use predication for short branches

Short conditional expressions can be cheaper as predicated operations.

```cuda
float v = condition ? a : b;
```

Use `__launch_bounds__` when a kernel must achieve a minimum occupancy.

## 5. Host-device transfers

### Minimize transfers

Keep data on the device between kernels.

Copy only the final result back when needed.

### Batch small copies

One large copy is usually better than many small copies.

### Reuse device buffers

Allocate device buffers once and reuse them across kernels instead of
allocating and freeing frequently.

### Use pinned memory

Pinned host memory enables faster host-device transfers:

```cuda
cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
```

Managed memory is simpler but may have migration overhead:

```cuda
cudaMallocManaged(&ptr, size);
```

Zero-copy host memory can be useful for small, frequently accessed host data.

### Overlap copies and kernels

Use non-default streams:

```cuda
cudaMemcpyAsync(...);
kernel<<<grid, block, 0, stream>>>(...);
```

Use events to order work between streams.

Prefer `cudaMemcpyAsync` with a non-default stream for copies that can overlap.

### Avoid unnecessary atomics

Atomics can become a serialization bottleneck.

Use per-block or per-warp partial results, then combine them with fewer atomic
operations or a separate reduction kernel.

## 6. Reductions

### Use shared memory for block reductions

```text
global -> shared memory
shared memory -> warp reduction
warp reduction -> one result
```

### Use warp primitives for small reductions

For 32-thread reductions, use:

```cuda
__shfl_down_sync(0xffffffff, value, offset);
```

This is faster than shared memory and barriers.

Block reduction pattern:

```cuda
__shared__ float partial[256];

partial[threadIdx.x] = value;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (threadIdx.x < stride) {
    partial[threadIdx.x] += partial[threadIdx.x + stride];
  }
  __syncthreads();
}
```

For warp reductions, use the full member mask:

```cuda
__shfl_down_sync(0xffffffff, value, 16);
```

Unroll loops when the trip count is small and known:

```cuda
#pragma unroll
for (int i = 0; i < 4; ++i) { ... }
```

## 7. Kernel launch overhead

### Avoid many tiny kernels

Merge elementwise operations into one kernel when possible.

This reduces launch and synchronization overhead.

### Prefer larger work per thread

For memory-bound kernels, each thread can process multiple elements:

```cuda
for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
  out[i] = in[i] * 2;
}
```

## 8. Memory-bound vs compute-bound

Identify the bottleneck before optimizing.

```text
memory-bound: time is limited by global or shared memory bandwidth
compute-bound: time is limited by arithmetic throughput
latency-bound: not enough parallelism to hide latency
```

Typical actions:

```text
memory-bound -> improve coalescing and reuse
compute-bound -> reduce redundant math or use faster types
latency-bound -> increase occupancy or work per thread
```

## 9. Profiling

Measure before tuning.

Useful metrics:

```text
kernel execution time
effective global memory bandwidth
occupancy
shared-memory bank conflicts
register usage
```

**[simulator-specific]** Use the simulator's trace and comparison tests to
detect correctness regressions before performance changes.

## 10. Simulator-specific rules

- **[simulator-specific]** Generated kernels must stay on the SIMT path.
- **[simulator-specific]** Do not use tensor-core instructions:

  ```text
  mma
  wmma
  ldmatrix
  stmatrix
  wgmma
  tcgen05
  ```

- **[simulator-specific]** `cp.async` may be emitted, but the simulator treats
  it as a synchronous copy.
- **[simulator-specific]** Do not rely on `cp.async` for latency hiding in the
  simulator.
- Prefer simple, explicit loops over heavily optimized vendor libraries.
- **[simulator-specific]** Do not call cuBLAS, cuDNN, cuFFT, cuRAND, or
  cuSOLVER.
- **[simulator-specific]** Avoid CUDA Graphs, textures, and surfaces in v1.
- Use multiple streams only when needed for asynchronous host/device overlap.
- **[simulator-specific]** Run the PTX tensor-core gate after generation:

  ```bash
  grep -E '\b(mma|wmma|ldmatrix|stmatrix|wgmma|tcgen05)\b' kernel.ptx
  ```

## 11. Checklist

Before adding a generated kernel:

```text
- [ ] global accesses are coalesced
- [ ] vector loads/stores are correctly aligned
- [ ] vectorized loads/stores are used where possible
- [ ] shared memory is used for reused tiles
- [ ] bank conflicts are avoided
- [ ] shared-memory padding is used when needed
- [ ] block size is a multiple of 32
- [ ] enough blocks are launched
- [ ] register and shared-memory limits are checked
- [ ] divergence is minimized
- [ ] integer division/modulo is minimized
- [ ] host-device copies are minimized
- [ ] pinned or managed memory is considered
- [ ] device buffers are reused
- [ ] atomics are minimized
- [ ] non-default streams are used for overlap
- [ ] **[simulator-specific]** no tensor-core PTX is emitted
- [ ] **[simulator-specific]** no cuBLAS/cuDNN/cuFFT calls are used
```
