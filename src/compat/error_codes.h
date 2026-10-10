#pragma once

#include <cstdint>

using cudaError_t = std::int32_t;
using CUresult = std::int32_t;

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInitializationError = 3;
constexpr cudaError_t cudaErrorInvalidDevice = 101;
constexpr cudaError_t cudaErrorInvalidResourceHandle = 400;
constexpr cudaError_t cudaErrorNotSupported = 801;
constexpr cudaError_t cudaErrorUnknown = 999;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
constexpr CUresult CUDA_ERROR_INVALID_DEVICE = 101;
constexpr CUresult CUDA_ERROR_INVALID_CONTEXT = 201;
constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
constexpr CUresult CUDA_ERROR_UNKNOWN = 999;
