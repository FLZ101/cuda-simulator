# GPU Model

This document defines the CUDA GPU execution and memory model used as the
reference for the simulator.

## 1. Overview

A CUDA GPU is a throughput-oriented manycore processor. It executes a large
number of threads concurrently and relies on massive parallelism rather than
high single-thread performance to achieve throughput.

The device consists of:

- a global front-end or block scheduler, often called a GigaThread engine
- multiple streaming multiprocessors, or SMs
- a device-wide L2 cache
- memory controllers and device DRAM
- a host interface

Different GPU generations have different capabilities. NVIDIA summarizes these
capabilities as a compute capability, for example `8.0`. Compute capability
controls which instructions and features are available, including tensor
cores, asynchronous copies, and memory consistency features.

| Compute capability | Example architecture class | Relevant features |
| --- | --- | --- |
| 7.0 | Volta | Independent thread scheduling, tensor cores |
| 7.5 | Turing | Tensor cores, improved integer execution |
| 8.0 | Ampere | Larger tensor-core throughput, asynchronous copies |
| 8.6 | Ampere | Improved FP16/BF16 support |
| 9.0 | Hopper | Thread block clusters, asynchronous barriers |

The simulator targets a simplified `8.0`-class device.

## 2. Execution model

A kernel launch creates a grid of thread blocks, also called cooperative thread
arrays, or CTAs.

The hierarchy is:

```text
grid
  |- thread block
  |    |- warp
  |    |    |- thread
```

Each SM executes one or more resident thread blocks. A thread block is
subdivided into warps of 32 threads. A warp is the fundamental scheduling unit.

The GPU uses single instruction, multiple thread, or SIMT, execution:

- all lanes in a warp execute the same instruction
- each lane has independent register state
- predication and an active mask control which lanes participate
- divergent control flow serializes the taken and fall-through paths

Warp schedulers hide memory latency by issuing instructions from other
resident warps while one warp is stalled.

A thread block is guaranteed to reside on a single SM for its entire lifetime.
This property is what makes block-level barriers and shared memory correct.

## 2.1 Streaming multiprocessor operation

An SM is the hardware unit that executes warps.

The execution loop is:

1. The global scheduler assigns one or more thread blocks to an SM.
2. The SM allocates registers and shared memory for each resident block.
3. Each block is partitioned into warps of 32 threads.
4. Each warp has a program counter and an active mask.
5. A warp scheduler selects a warp that is ready to issue.
6. The instruction is decoded and dispatched to the appropriate execution
   unit.
7. Operands are read from the register file.
8. Arithmetic, memory, or control instructions execute.
9. Results are written back to the register file or memory.

The SM contains different execution units:

```text
FP32 / INT32 units
FP64 units
special function units
load/store units
tensor cores where available
```

Because different instructions use different units, multiple warps can issue
different instruction types concurrently.

### Instruction issue

A warp scheduler issues one instruction from a warp at a time.

The order in which ready warps are selected is architecture-dependent and not
guaranteed by the CUDA programming model.

An instruction is ready when:

- all required operands are available
- the destination execution unit is available
- the warp is not waiting at a barrier or memory operation

### Memory access path

Load and store instructions are handled by load/store units.

The memory path is:

```text
load/store unit
    |
    v
shared memory / L1
    |
    v
L2
    |
    v
global DRAM
```

Shared memory is inside the SM and has much lower latency than global memory.
The relationship between shared memory and L1 cache is described in the memory
hierarchy section below.

### Barriers and shared memory

Threads in the same block execute on the same SM. This allows:

- `__syncthreads()`
- shared-memory producer/consumer patterns
- block-scoped atomics

When a warp reaches a barrier, the SM stops issuing it until all warps in the
same block have arrived.

### Divergence handling

For a divergent branch:

1. The active mask is saved.
2. The taken path executes with only taken lanes active.
3. The fall-through path executes with only the remaining lanes active.
4. The original active mask is restored at reconvergence.

### Latency hiding

When a warp stalls on a long-latency operation, the warp scheduler issues
instructions from another resident warp.

This is the core mechanism that allows GPUs to tolerate memory latency.

## 3. Memory hierarchy

The GPU memory system is organized as a hierarchy.

| Memory | Scope | Physical location | Characteristics |
| --- | --- | --- | --- |
| Registers | Per thread | SM register file | Private, highest bandwidth, lowest latency |
| Shared memory | Per thread block | On-chip SRAM in SM | Software-managed, supports block synchronization |
| L1 cache | Per SM | On-chip in SM | Caches local and global accesses depending on architecture |
| Constant memory | All threads, read-only | Device DRAM with SM cache | Optimized for uniform warp access |
| Texture memory | All threads, read-only | Device DRAM with texture cache | Optimized for spatial locality |
| L2 cache | All SMs | On-chip, device-wide | Shared cache for global, local, and constant traffic |
| Global memory | All threads | Device DRAM | Large capacity, high latency, persistent |
| Local memory | Per thread | Device DRAM, cached | Used for register spills and dynamically indexed arrays |
| Parameter memory | Per kernel | Launch metadata | Read-only kernel arguments |

The logical latency order is:

```text
register < shared memory < L1/L2 < device DRAM
```

Caches such as L1 and L2 are transparent to the programmer. Shared memory,
registers, local memory, constant memory, and global memory are program-visible
memory spaces.

### L1 and shared memory

On several architectures, L1 cache and shared memory share the same on-chip
SRAM. The split between cache capacity and shared-memory capacity is
configurable on some devices.

Shared memory is explicitly managed by the program. L1 cache is managed by the
hardware.

### Local memory

Local memory is per-thread memory that is logically private but physically
resides in device DRAM.

The compiler uses local memory when:

- a local array is indexed dynamically
- the compiler runs out of registers
- a large local object does not fit in registers
- a value is spilled from a register

For example, an array with dynamic indexing may be placed in local memory:

```cuda
__global__ void kernel(int *out) {
  int values[128];
  int i = threadIdx.x;
  values[i] = i;
  out[threadIdx.x] = values[i];
}
```

Important distinction:

```text
registers     = per-thread, on-chip
shared memory = per-block, on-chip
local memory  = per-thread, physically in DRAM
```

Local memory is cached, but it has much higher latency than registers and
shared memory.

In PTX, local memory is represented by the `.local` address space and accessed
with `ld.local` and `st.local`.

### Program-visible address spaces

CUDA pointers belong to address spaces:

```text
global
shared
local
constant
```

In PTX, an address-space suffix makes the target explicit:

```text
ld.global
st.shared
ld.local
ld.const
```

Generic pointers do not include a suffix. The hardware or simulator resolves
the target space from the pointer value and recorded address ranges.

## 4. PTX memory spaces

PTX exposes the memory hierarchy through explicit address spaces:

```text
.reg
.global
.shared
.local
.const
.param
```

Generic loads and stores use plain `ld` and `st` and resolve the target space
from the address.

## 5. Global memory access

Global memory bandwidth is maximized through coalescing.

In a coalesced access, consecutive threads in a warp access consecutive words.
The memory subsystem can service such a pattern with few wide transactions.

Example of coalesced access:

```text
thread 0 -> address 0
thread 1 -> address 4
thread 2 -> address 8
...
thread 31 -> address 124
```

Example of strided access:

```text
thread 0 -> address 0
thread 1 -> address 1024
thread 2 -> address 2048
...
```

Strided or scattered accesses degrade effective bandwidth because they require
multiple transactions.

Latency and bandwidth are different:

```text
latency   = time to complete one memory request
bandwidth = total bytes transferred per unit time
```

Many independent memory requests can use bandwidth even when each individual
request has high latency.

The simulator classifies warp memory patterns as:

```text
uniform
contiguous
strided
divergent
```

## 6. Shared memory behavior

Shared memory is divided into banks.

- accesses to distinct banks proceed in parallel
- multiple accesses to the same bank at different addresses serialize
- a uniform read can be broadcast

Example without conflict:

```text
lane 0 -> bank 0
lane 1 -> bank 1
lane 2 -> bank 2
```

Example with a two-way conflict:

```text
lane 0 -> bank 0 address A
lane 1 -> bank 1
lane 2 -> bank 0 address B
```

The simulator initially treats shared memory as a byte buffer and does not
simulate bank conflicts.

## 7. Occupancy and resource constraints

Occupancy measures how many warps are active on an SM relative to the maximum
supported by the architecture.

Occupancy is constrained by:

```text
registers per thread
shared memory per block
maximum threads per SM
maximum resident blocks per SM
```

Higher occupancy provides more warps for latency hiding but may reduce
available registers or shared memory per thread.

Occupancy can be computed per SM as:

```text
blocks by threads = floor(max_threads_per_SM / threads_per_block)
blocks by registers = floor(registers_per_SM /
                           (registers_per_thread * threads_per_block))
blocks by shared = floor(shared_mem_per_SM / shared_mem_per_block)
resident_blocks = min(blocks_by_threads, blocks_by_registers, blocks_by_shared)
occupancy = resident_blocks * threads_per_block / max_threads_per_SM
```

Registers are allocated with hardware-specific granularity, often rounded up
per warp. Shared memory is allocated per thread block.

## 8. Host-device interaction

Host and device memory are separate. Data is transferred explicitly:

```text
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
cudaMemcpyDefault
```

Pinned host memory avoids pageable staging and improves transfer performance.

Managed memory provides a unified address space:

```cuda
cudaMallocManaged(...)
```

### Asynchronous copy engines

Real GPUs contain DMA engines that perform host-device copies without occupying
SM execution units. This is why asynchronous copies can overlap with kernel
execution.

### Memory consistency

The GPU memory model requires explicit ordering in some producer/consumer
patterns:

```text
__threadfence_block()
__threadfence()
__threadfence_system()
```

Barriers such as `__syncthreads()` also provide ordering within a thread block.

## 9. Simulator mapping

The simulator models the programming model, not cycle-accurate hardware.

| Hardware concept | Simulator representation |
| --- | --- |
| Device | `Device` and `DeviceRegistry` |
| SM | Host worker thread |
| Global scheduler | Stream manager and future block scheduler |
| Global memory | Host-backed `MemoryManager` |
| Shared memory | Per-block byte buffer |
| Constant memory | Per-module read-only buffer |
| Local memory | Per-thread spill area |
| Warp | Future `Warp` object with active mask |
| Barrier | Per-block barrier state |

Current simulated device configuration:

```text
SM count: 8
max threads per block: 1024
max threads per SM: 2048
warp size: 32
shared memory per block: 49152
```

The simulator does not model:

```text
cache behavior
shared-memory bank conflicts
memory controller timing
cycle-accurate scheduling
tensor-core execution
```
