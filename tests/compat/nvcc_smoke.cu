#include <cuda_runtime.h>

int main() {
  int count = 0;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess || count != 1) {
    return 1;
  }

  error = cudaSetDevice(0);
  if (error != cudaSuccess) {
    return 2;
  }

  void *ptr = nullptr;
  error = cudaMalloc(&ptr, 64);
  if (error != cudaSuccess || ptr == nullptr) {
    return 3;
  }

  error = cudaFree(ptr);
  if (error != cudaSuccess) {
    return 4;
  }

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    return 5;
  }

  return 0;
}
