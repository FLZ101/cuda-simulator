#include "compat/cuda_api.h"

#include <cstring>

extern "C" {
cudaError_t cudaMalloc(void **ptr, std::size_t size);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, std::size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaDeviceSynchronize();
}

int main() {
  const int expected = 123;
  int host_src = expected;
  int host_dst = 0;

  void *device_src = nullptr;
  void *device_dst = nullptr;
  if (cudaMalloc(&device_src, sizeof(int)) != cudaSuccess ||
      cudaMalloc(&device_dst, sizeof(int)) != cudaSuccess) {
    return 1;
  }

  cudaStream_t stream_a = nullptr;
  cudaStream_t stream_b = nullptr;
  if (cudaStreamCreate(&stream_a) != cudaSuccess ||
      cudaStreamCreate(&stream_b) != cudaSuccess) {
    return 2;
  }

  cudaEvent_t event = nullptr;
  if (cudaEventCreate(&event) != cudaSuccess) {
    return 3;
  }

  if (cudaMemcpyAsync(device_src, &host_src, sizeof(int),
                      cudaMemcpyHostToDevice, stream_a) != cudaSuccess) {
    return 4;
  }
  if (cudaEventRecord(event, stream_a) != cudaSuccess) {
    return 5;
  }
  if (cudaStreamWaitEvent(stream_b, event, 0) != cudaSuccess) {
    return 6;
  }
  if (cudaMemcpyAsync(device_dst, device_src, sizeof(int),
                      cudaMemcpyDeviceToDevice, stream_b) != cudaSuccess) {
    return 7;
  }
  if (cudaMemcpyAsync(&host_dst, device_dst, sizeof(int),
                      cudaMemcpyDeviceToHost, stream_b) != cudaSuccess) {
    return 8;
  }

  if (cudaStreamSynchronize(stream_b) != cudaSuccess ||
      cudaEventSynchronize(event) != cudaSuccess) {
    return 9;
  }
  if (host_dst != expected) {
    return 10;
  }

  cudaEventDestroy(event);
  cudaStreamDestroy(stream_a);
  cudaStreamDestroy(stream_b);
  cudaFree(device_src);
  cudaFree(device_dst);
  cudaDeviceSynchronize();

  return 0;
}
