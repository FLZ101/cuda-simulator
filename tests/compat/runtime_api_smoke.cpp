#include "compat/cuda_api.h"

#include <cstring>

extern "C" {
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *props, int device);
cudaError_t cudaMalloc(void **ptr, std::size_t size);
cudaError_t cudaMemcpy(void *dst, const void *src, std::size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
const char *cudaGetErrorString(cudaError_t error);
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1) {
    return 1;
  }

  if (cudaSetDevice(0) != cudaSuccess) {
    return 2;
  }

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    return 3;
  }

  void *device = nullptr;
  if (cudaMalloc(&device, 64) != cudaSuccess || device == nullptr) {
    return 4;
  }

  const int expected = 42;
  int host = 0;
  if (cudaMemcpy(device, &expected, sizeof(expected), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    return 5;
  }
  if (cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return 6;
  }
  if (host != expected) {
    return 7;
  }

  if (cudaFree(device) != cudaSuccess) {
    return 8;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return 9;
  }
  if (cudaGetLastError() != cudaSuccess) {
    return 10;
  }

  return 0;
}
