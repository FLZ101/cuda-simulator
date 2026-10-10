#include "compat/cuda_api.h"

#include <cstring>

extern "C" {
CUresult cuInit(unsigned int flags);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice device);
CUresult cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice device);
CUresult cuMemAlloc(CUdeviceptr *ptr, std::size_t size);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, std::size_t count);
CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, std::size_t count);
CUresult cuMemFree(CUdeviceptr ptr);
CUresult cuCtxDestroy(CUcontext ctx);
}

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    return 1;
  }

  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count != 1) {
    return 2;
  }

  CUdevice device = -1;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS || device != 0) {
    return 3;
  }

  char name[256]{};
  if (cuDeviceGetName(name, sizeof(name), device) != CUDA_SUCCESS) {
    return 4;
  }

  CUcontext context = nullptr;
  if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS || context == nullptr) {
    return 5;
  }

  CUdeviceptr device_ptr = 0;
  if (cuMemAlloc(&device_ptr, 64) != CUDA_SUCCESS || device_ptr == 0) {
    return 6;
  }

  const int expected = 77;
  int host = 0;
  if (cuMemcpyHtoD(device_ptr, &expected, sizeof(expected)) != CUDA_SUCCESS) {
    return 7;
  }
  if (cuMemcpyDtoH(&host, device_ptr, sizeof(host)) != CUDA_SUCCESS) {
    return 8;
  }
  if (host != expected) {
    return 9;
  }

  if (cuMemFree(device_ptr) != CUDA_SUCCESS) {
    return 10;
  }
  if (cuCtxDestroy(context) != CUDA_SUCCESS) {
    return 11;
  }

  return 0;
}
