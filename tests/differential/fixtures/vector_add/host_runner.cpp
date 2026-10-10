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

static std::vector<float> read_floats(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::vector<float> values;
  float value = 0.0F;
  while (input.read(reinterpret_cast<char *>(&value), sizeof(value))) {
    values.push_back(value);
  }
  return values;
}

static bool write_floats(const std::string &path,
                         const std::vector<float> &values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  return output.good();
}

int main(int argc, char **argv) {
  if (argc != 5) {
    return 1;
  }

  std::ifstream ptx_input(PTX_FIXTURE_PATH);
  if (!ptx_input) {
    return 2;
  }
  std::ostringstream buffer;
  buffer << ptx_input.rdbuf();
  const std::string ptx = buffer.str();

  const std::string input_a_path = argv[1];
  const std::string input_b_path = argv[2];
  const std::string output_path = argv[3];
  const int n = std::stoi(argv[4]);

  std::vector<float> a = read_floats(input_a_path);
  std::vector<float> b = read_floats(input_b_path);
  if (a.size() != static_cast<std::size_t>(n) ||
      b.size() != static_cast<std::size_t>(n)) {
    return 3;
  }

  if (cuInit(0) != CUDA_SUCCESS) {
    return 4;
  }

  CUmodule module = nullptr;
  if (cuModuleLoadData(&module, ptx.c_str()) != CUDA_SUCCESS) {
    return 5;
  }

  CUfunction function = nullptr;
  if (cuModuleGetFunction(&function, module, "add_kernel") != CUDA_SUCCESS) {
    return 6;
  }

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(n);
  if (cuMemAlloc(&d_a, bytes) != CUDA_SUCCESS ||
      cuMemAlloc(&d_b, bytes) != CUDA_SUCCESS ||
      cuMemAlloc(&d_c, bytes) != CUDA_SUCCESS) {
    return 7;
  }

  if (cuMemcpyHtoD(d_a, a.data(), bytes) != CUDA_SUCCESS ||
      cuMemcpyHtoD(d_b, b.data(), bytes) != CUDA_SUCCESS) {
    return 8;
  }

  int count = n;
  void *params[] = {&d_a, &d_b, &d_c, &count};
  const unsigned int block = 256;
  const unsigned int grid = (static_cast<unsigned int>(n) + block - 1) / block;
  if (cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr, params,
                     nullptr) != CUDA_SUCCESS) {
    return 9;
  }

  std::vector<float> c(static_cast<std::size_t>(n));
  if (cuMemcpyDtoH(c.data(), d_c, bytes) != CUDA_SUCCESS) {
    return 10;
  }
  if (!write_floats(output_path, c)) {
    return 11;
  }

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  return 0;
}
