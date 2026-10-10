#pragma once

#include <cstddef>
#include <cstdint>

#include "compat/error_codes.h"

using cudaStream_t = void *;
using cudaEvent_t = void *;
using cudaDeviceptr = unsigned long long;

using CUdevice = std::int32_t;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUdeviceptr = unsigned long long;
using CUstream = void *;
using CUevent = void *;
using cudaDeviceAttr = std::int32_t;
using CUdevice_attribute = std::int32_t;

enum : std::int32_t {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrMaxThreadsPerMultiProcessor = 39,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
};

enum : std::int32_t {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
};

enum cudaMemcpyKind : std::int32_t {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

struct cudaDeviceProp {
  char name[256];
  int major;
  int minor;
  int multiProcessorCount;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int warpSize;
  int maxGridSize[3];
  int maxThreadsDim[3];
  int sharedMemPerBlock;
  int regsPerBlock;
  int clockRate;
  std::size_t totalGlobalMem;
  int integrated;
};

struct cudaFuncAttributes {
  std::size_t sharedSizeBytes;
  std::size_t constSizeBytes;
  std::size_t localSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int ptxVersion;
  int binaryVersion;
  int cacheModeCA;
};
