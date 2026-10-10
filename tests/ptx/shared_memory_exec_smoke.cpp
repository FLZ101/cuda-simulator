#include "compat/cuda_api.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
CUresult cuInit(unsigned int flags);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *name);
CUresult cuMemAlloc(CUdeviceptr *ptr, std::size_t size);
CUresult cuMemFree(CUdeviceptr ptr);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, std::size_t count);
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
  if (cuModuleGetFunction(&function, module, "shared_sum") != CUDA_SUCCESS) {
    return 4;
  }

  constexpr int n = 32;
  std::vector<float> input_data(n);
  float expected = 0.0F;
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
    expected += input_data[i];
  }

  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  if (cuMemAlloc(&d_in, sizeof(float) * n) != CUDA_SUCCESS ||
      cuMemAlloc(&d_out, sizeof(float)) != CUDA_SUCCESS) {
    return 5;
  }

  if (cuMemcpyHtoD(d_in, input_data.data(), sizeof(float) * n) !=
      CUDA_SUCCESS) {
    return 6;
  }

  void *params[] = {&d_in, &d_out};
  if (cuLaunchKernel(function, 1, 1, 1, n, 1, 1, 0, nullptr, params, nullptr) !=
      CUDA_SUCCESS) {
    return 7;
  }

  float output = 0.0F;
  if (cuMemcpyDtoH(&output, d_out, sizeof(output)) != CUDA_SUCCESS) {
    return 8;
  }

  if (output != expected) {
    return 9;
  }

  cuMemFree(d_in);
  cuMemFree(d_out);
  return 0;
}
