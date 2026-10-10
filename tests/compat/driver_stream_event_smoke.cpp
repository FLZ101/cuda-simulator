#include "compat/cuda_api.h"

#include <cstring>

extern "C" {
CUresult cuInit(unsigned int flags);
CUresult cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice device);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuMemAlloc(CUdeviceptr *ptr, std::size_t size);
CUresult cuMemFree(CUdeviceptr ptr);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, std::size_t count,
                           CUstream stream);
CUresult cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, std::size_t count,
                           CUstream stream);
CUresult cuStreamCreate(CUstream *stream, unsigned int flags);
CUresult cuStreamDestroy(CUstream stream);
CUresult cuStreamSynchronize(CUstream stream);
CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CUresult cuEventCreate(CUevent *event, unsigned int flags);
CUresult cuEventDestroy(CUevent event);
CUresult cuEventRecord(CUevent event, CUstream stream);
CUresult cuEventSynchronize(CUevent event);
}

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    return 1;
  }

  CUcontext context = nullptr;
  if (cuCtxCreate(&context, 0, 0) != CUDA_SUCCESS) {
    return 2;
  }

  const int expected = 456;
  int host_src = expected;
  int host_dst = 0;
  CUdeviceptr device_src = 0;
  CUdeviceptr device_dst = 0;
  if (cuMemAlloc(&device_src, sizeof(int)) != CUDA_SUCCESS ||
      cuMemAlloc(&device_dst, sizeof(int)) != CUDA_SUCCESS) {
    return 3;
  }

  CUstream stream_a = nullptr;
  CUstream stream_b = nullptr;
  if (cuStreamCreate(&stream_a, 0) != CUDA_SUCCESS ||
      cuStreamCreate(&stream_b, 0) != CUDA_SUCCESS) {
    return 4;
  }

  CUevent event = nullptr;
  if (cuEventCreate(&event, 0) != CUDA_SUCCESS) {
    return 5;
  }

  if (cuMemcpyHtoDAsync(device_src, &host_src, sizeof(int), stream_a) !=
      CUDA_SUCCESS) {
    return 6;
  }
  if (cuEventRecord(event, stream_a) != CUDA_SUCCESS) {
    return 7;
  }
  if (cuStreamWaitEvent(stream_b, event, 0) != CUDA_SUCCESS) {
    return 8;
  }
  if (cuMemcpyHtoDAsync(device_dst, &host_src, sizeof(int), stream_b) !=
      CUDA_SUCCESS) {
    return 9;
  }
  if (cuMemcpyDtoHAsync(&host_dst, device_dst, sizeof(int), stream_b) !=
      CUDA_SUCCESS) {
    return 10;
  }

  if (cuStreamSynchronize(stream_b) != CUDA_SUCCESS ||
      cuEventSynchronize(event) != CUDA_SUCCESS) {
    return 11;
  }
  if (host_dst != expected) {
    return 12;
  }

  cuEventDestroy(event);
  cuStreamDestroy(stream_a);
  cuStreamDestroy(stream_b);
  cuMemFree(device_src);
  cuMemFree(device_dst);
  cuCtxDestroy(context);

  return 0;
}
