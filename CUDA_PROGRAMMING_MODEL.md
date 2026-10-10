# CUDA Programming Model

This document describes the CUDA programming model as it relates to the
`cuda-simulator` project.

## Host and device

CUDA programs have two execution environments:

- **Host**: the CPU and its memory.
- **Device**: the GPU and its memory.

The host runs ordinary C/C++ code and launches work on the device. The device
executes many threads in parallel.

In this simulator:

- host code runs normally as a host executable
- device code is represented by PTX
- the simulator executes PTX kernels on the host CPU

See [GPU_MODEL.md](GPU_MODEL.md) for the simulated GPU model.

## Data movement between host and device

Device memory is normally separate from host memory. CUDA provides copy APIs:

```text
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
cudaMemcpyDefault
```

Examples:

```cuda
cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);
```

CUDA also supports managed memory, which presents a single pointer to both
host and device code:

```cuda
cudaMallocManaged(&ptr, bytes);
```

In this simulator, device memory is host-backed, so device pointers are also
valid host addresses. This naturally implements unified addressing.

## Kernel

A kernel is a function that runs on the device.

CUDA function qualifiers include:

```text
__global__   called from host, runs on device
__device__   called from device, runs on device
__host__     called from host, runs on host
```

```cuda
__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
```

The host launches a kernel with:

```cuda
add_kernel<<<grid, block>>>(a, b, c, n);
```

Kernel launches are asynchronous: the host may continue immediately after
submitting the launch. Synchronization is performed with stream or device
synchronization calls.

The runtime and driver APIs also expose an explicit launch form:

```text
cudaLaunchKernel(...)
cuLaunchKernel(...)
```

The simulator receives grid dimensions, block dimensions, shared memory size,
stream, and kernel arguments before execution.

## Thread hierarchy

### Grid

A kernel launch creates a grid of thread blocks.

A grid can be 1D, 2D, or 3D. Its size is the launch grid dimension.

### Thread block / CTA

Each grid contains thread blocks. Threads in the same block:

- can synchronize with `__syncthreads()`
- can share memory through shared memory
- have the same `blockIdx`

A block can be 1D, 2D, or 3D.

### Warp

Threads in a block are grouped into warps of 32 threads.

A warp executes in SIMT style: all active lanes execute the same instruction.

### Thread

Each thread has:

- `threadIdx`
- `blockIdx`
- `blockDim`
- `gridDim`

These built-in variables are `uint3` values and have `.x`, `.y`, and `.z`.

| Variable | Meaning |
| --- | --- |
| `gridDim` | Number of thread blocks in each grid dimension |
| `blockIdx` | Index of the current block in the grid |
| `blockDim` | Number of threads in each block dimension |
| `threadIdx` | Index of the current thread in its block |

For a 1D grid and 1D block, the global thread index is:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D blocks, the linear thread index inside the block is:

```cuda
int tid2 = threadIdx.y * blockDim.x + threadIdx.x;
```

For 3D blocks:

```cuda
int tid3 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x +
           threadIdx.x;
```

The global 2D coordinate of a thread is:

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

The global 3D coordinate is:

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

Within a block, the linear thread ID determines the warp and lane:

```cuda
int linear_tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x +
                 threadIdx.x;
int lane = linear_tid % 32;
int warp = linear_tid / 32;
```

Each warp has 32 lanes. Lane 0 through lane 31 map to consecutive linear
thread IDs inside a block.

## Memory hierarchy

| Memory | Scope | Notes |
| --- | --- | --- |
| Registers | One thread | Fastest, private to a thread |
| Global | All threads | Large, slow, host-visible |
| Shared | One block | Fast, user-managed |
| Local | One thread | Register spills or local arrays |
| Constant | All threads | Read-only, cached |
| Parameter | Kernel arguments | Read-only during a launch |
| Generic | Address space | Plain `ld`/`st` resolves by address |

In this simulator, all device memory is host-backed.

PTX names these spaces with address-space suffixes:

```text
.global
.shared
.local
.const
.param
```

The simulator tracks address ranges so generic `ld` and `st` can resolve to the
correct space.

## SIMT and divergence

All lanes in a warp normally execute the same instruction.

When a branch divides lanes, the simulator must:

1. Save the original active mask and reconvergence point.
2. Execute the taken path.
3. Execute the fall-through path.
4. Restore the active mask at reconvergence.

## Synchronization

### Block synchronization

```cuda
__syncthreads();
```

All threads in a block must arrive before any continue.

The simulator implements this with a per-block barrier and a runnable-warp
queue.

Named barriers and `bar.arrive`/`bar.sync` are also supported when generated
code uses them.

### Memory fences

Fences are required for correct producer/consumer patterns:

```text
__threadfence_block()
__threadfence()
__threadfence_system()
```

`__threadfence_block()` orders memory accesses within a block.
`__threadfence()` orders device memory accesses.
`__threadfence_system()` includes host memory.

### Warp synchronization

Warp-level primitives include:

```text
shfl.sync
vote.sync
ballot.sync
```

### Atomics

Atomic operations coordinate access to global or shared memory:

```text
atomicAdd
atomicCAS
atomicExch
atomicMax
atomicMin
```

## Streams and events

### Stream

A stream is an ordered queue of asynchronous operations.

Operations on the same stream execute in order. Different streams may overlap.

Common stream operations:

```text
cudaStreamCreate
cudaMemcpyAsync
cudaLaunchKernel
cudaStreamSynchronize
cudaStreamWaitEvent
```

Stream 0 is the default stream. The simulator treats synchronizing the default
stream as synchronizing all streams.

A stream command is complete only after the underlying operation completes.
For a kernel, this means every block in its grid has finished.

### Event

An event is a marker in a stream.

```text
cudaEventRecord
cudaEventSynchronize
cudaEventElapsedTime
```

Events are used for ordering and timing.

`cudaEventQuery` reports whether an event has been recorded without blocking.

`cudaEventElapsedTime` measures the simulated time between two recorded events.

## Runtime API vs driver API

### Runtime API

- higher-level
- implicit primary context
- examples: `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`

### Driver API

- lower-level
- explicit context and module management
- examples: `cuInit`, `cuCtxCreate`, `cuModuleLoad`, `cuLaunchKernel`

This project implements both APIs through shim libraries.

Runtime API calls use an implicit primary context. Driver API calls manage the
context explicitly. Both APIs return error codes, and the runtime API keeps a
per-thread last error.

## PTX

PTX is NVIDIA's virtual instruction set.

The compiler lowers CUDA source to PTX. This simulator accepts PTX because it
can be interpreted or lowered on the CPU.

Example PTX flow:

```text
CUDA source
    |
    v
PTX
    |
    v
simulator IR
    |
    v
CPU execution or JIT
```

PTX is usually embedded in a fatbin. A fatbin may contain PTX and/or CUBIN.
This simulator accepts PTX only and rejects CUBIN/SASS.

Tensor-core instructions are out of scope:

```text
mma
wmma
ldmatrix
stmatrix
wgmma
tcgen05
```

Generated kernels must use the SIMT path.

## Simulator mapping

| CUDA concept | Simulator representation |
| --- | --- |
| GPU device | `Device` + `DeviceRegistry` |
| Context | `Context` + `ContextManager` |
| Device memory | Host-backed `MemoryManager` |
| Stream | `Stream` with command queue and worker |
| Event | `Event` with recorded flag |
| Kernel launch | Future PTX executor |
| Thread block | Future `Block` object |
| Warp | Future `Warp` object with active mask |
| Register file | Structure-of-arrays per warp |
| Barrier | Per-block barrier state |
| Reconvergence | Per-warp reconvergence stack |
| Shared memory | Per-block host buffer |
