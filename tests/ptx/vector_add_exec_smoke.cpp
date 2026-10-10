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
  if (cuModuleGetFunction(&function, module, "add_kernel") != CUDA_SUCCESS) {
    return 4;
  }

  constexpr int n = 1024;
  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> c(n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i + 1);
  }

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  if (cuMemAlloc(&d_a, sizeof(float) * static_cast<std::size_t>(n)) !=
          CUDA_SUCCESS ||
      cuMemAlloc(&d_b, sizeof(float) * static_cast<std::size_t>(n)) !=
          CUDA_SUCCESS ||
      cuMemAlloc(&d_c, sizeof(float) * static_cast<std::size_t>(n)) !=
          CUDA_SUCCESS) {
    return 5;
  }

  if (cuMemcpyHtoD(d_a, a.data(), a.size() * sizeof(float)) != CUDA_SUCCESS ||
      cuMemcpyHtoD(d_b, b.data(), b.size() * sizeof(float)) != CUDA_SUCCESS) {
    return 6;
  }

  int count = n;
  void *params[] = {&d_a, &d_b, &d_c, &count};
  const unsigned int block = 256;
  const unsigned int grid = (n + block - 1) / block;
  if (cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr, params,
                     nullptr) != CUDA_SUCCESS) {
    return 7;
  }

  if (cuMemcpyDtoH(c.data(), d_c, c.size() * sizeof(float)) != CUDA_SUCCESS) {
    return 8;
  }

  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i] != a[i] + b[i]) {
      return 9;
    }
  }

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  return 0;
}
