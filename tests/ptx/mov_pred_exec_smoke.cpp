#include "compat/cuda_api.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

extern "C" {
CUresult cuInit(unsigned int flags);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *name);
CUresult cuMemAlloc(CUdeviceptr *ptr, std::size_t size);
CUresult cuMemFree(CUdeviceptr ptr);
CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, std::size_t count);
CUresult cuLaunchKernel(CUfunction function, unsigned int grid_x,
                        unsigned int grid_y, unsigned int grid_z,
                        unsigned int block_x, unsigned int block_y,
                        unsigned int block_z, unsigned int shared_mem,
                        CUstream stream, void **params, void **extra);
}

int main() {
  std::ifstream input(PTX_FIXTURE_PATH);
  if (!input) {
    return 1;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string ptx = buffer.str();

  if (cuInit(0) != CUDA_SUCCESS) {
    return 2;
  }

  CUmodule module = nullptr;
  if (cuModuleLoadData(&module, ptx.c_str()) != CUDA_SUCCESS) {
    return 3;
  }

  CUfunction function = nullptr;
  if (cuModuleGetFunction(&function, module, "mov_pred_kernel") !=
      CUDA_SUCCESS) {
    return 4;
  }

  CUdeviceptr d_output = 0;
  if (cuMemAlloc(&d_output, sizeof(std::uint32_t)) != CUDA_SUCCESS) {
    return 5;
  }

  void *params[] = {&d_output};
  if (cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr) !=
      CUDA_SUCCESS) {
    return 6;
  }

  std::uint32_t output = 0;
  if (cuMemcpyDtoH(&output, d_output, sizeof(output)) != CUDA_SUCCESS) {
    return 7;
  }

  if (output != 3U) {
    return 8;
  }

  cuMemFree(d_output);
  return 0;
}
