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

static bool run_kernel(const std::string &ptx_path, const std::string &name,
                       const std::vector<void *> &params, unsigned int grid,
                       unsigned int block) {
  std::ifstream ptx_input(ptx_path);
  if (!ptx_input) {
    return false;
  }
  std::ostringstream buffer;
  buffer << ptx_input.rdbuf();
  const std::string ptx = buffer.str();

  if (cuInit(0) != CUDA_SUCCESS) {
    return false;
  }
  CUmodule module = nullptr;
  if (cuModuleLoadData(&module, ptx.c_str()) != CUDA_SUCCESS) {
    return false;
  }
  CUfunction function = nullptr;
  if (cuModuleGetFunction(&function, module, name.c_str()) != CUDA_SUCCESS) {
    return false;
  }
  return cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr,
                        const_cast<void **>(params.data()),
                        nullptr) == CUDA_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  const std::string mode = argv[1];
  if (mode == "copy") {
    if (argc != 5) {
      return 2;
    }
    const std::string in_path = argv[2];
    const std::string out_path = argv[3];
    const int n = std::stoi(argv[4]);
    std::vector<float> input = read_floats(in_path);
    if (input.size() != static_cast<std::size_t>(n)) {
      return 3;
    }
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = 0;
    CUdeviceptr d_out = 0;
    if (cuMemAlloc(&d_in, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, bytes) != CUDA_SUCCESS) {
      return 4;
    }
    if (cuMemcpyHtoD(d_in, input.data(), bytes) != CUDA_SUCCESS) {
      return 5;
    }
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    const unsigned int block = 256;
    const unsigned int grid =
        (static_cast<unsigned int>(n) + block - 1) / block;
    if (!run_kernel(COPY_PTX_PATH, "copy_kernel", params, grid, block)) {
      return 6;
    }
    std::vector<float> output(static_cast<std::size_t>(n));
    if (cuMemcpyDtoH(output.data(), d_out, bytes) != CUDA_SUCCESS) {
      return 7;
    }
    if (!write_floats(out_path, output)) {
      return 8;
    }
    cuMemFree(d_in);
    cuMemFree(d_out);
    return 0;
  }

  if (mode == "saxpy") {
    if (argc != 7) {
      return 9;
    }
    const std::string x_path = argv[2];
    const std::string y_path = argv[3];
    const std::string out_path = argv[4];
    const float alpha = std::stof(argv[5]);
    const int n = std::stoi(argv[6]);
    std::vector<float> x = read_floats(x_path);
    std::vector<float> y = read_floats(y_path);
    if (x.size() != static_cast<std::size_t>(n) ||
        y.size() != static_cast<std::size_t>(n)) {
      return 10;
    }
    const std::size_t bytes = x.size() * sizeof(float);
    CUdeviceptr d_x = 0;
    CUdeviceptr d_y = 0;
    CUdeviceptr d_out = 0;
    if (cuMemAlloc(&d_x, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_y, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, bytes) != CUDA_SUCCESS) {
      return 11;
    }
    if (cuMemcpyHtoD(d_x, x.data(), bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(d_y, y.data(), bytes) != CUDA_SUCCESS) {
      return 12;
    }
    int count = n;
    float a = alpha;
    std::vector<void *> params = {&d_x, &d_y, &d_out, &a, &count};
    const unsigned int block = 256;
    const unsigned int grid =
        (static_cast<unsigned int>(n) + block - 1) / block;
    if (!run_kernel(SAXPY_PTX_PATH, "saxpy", params, grid, block)) {
      return 13;
    }
    std::vector<float> output(static_cast<std::size_t>(n));
    if (cuMemcpyDtoH(output.data(), d_out, bytes) != CUDA_SUCCESS) {
      return 14;
    }
    if (!write_floats(out_path, output)) {
      return 15;
    }
    cuMemFree(d_x);
    cuMemFree(d_y);
    cuMemFree(d_out);
    return 0;
  }

  if (mode == "relu") {
    if (argc != 5) {
      return 17;
    }
    const std::string in_path = argv[2];
    const std::string out_path = argv[3];
    const int n = std::stoi(argv[4]);
    std::vector<float> input = read_floats(in_path);
    if (input.size() != static_cast<std::size_t>(n)) {
      return 18;
    }
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = 0;
    CUdeviceptr d_out = 0;
    if (cuMemAlloc(&d_in, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, bytes) != CUDA_SUCCESS) {
      return 19;
    }
    if (cuMemcpyHtoD(d_in, input.data(), bytes) != CUDA_SUCCESS) {
      return 20;
    }
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    const unsigned int block = 256;
    const unsigned int grid =
        (static_cast<unsigned int>(n) + block - 1) / block;
    if (!run_kernel(RELU_PTX_PATH, "relu", params, grid, block)) {
      return 21;
    }
    std::vector<float> output(static_cast<std::size_t>(n));
    if (cuMemcpyDtoH(output.data(), d_out, bytes) != CUDA_SUCCESS) {
      return 22;
    }
    if (!write_floats(out_path, output)) {
      return 23;
    }
    cuMemFree(d_in);
    cuMemFree(d_out);
    return 0;
  }

  if (mode == "gelu") {
    if (argc != 5) {
      return 24;
    }
    const std::string in_path = argv[2];
    const std::string out_path = argv[3];
    const int n = std::stoi(argv[4]);
    std::vector<float> input = read_floats(in_path);
    if (input.size() != static_cast<std::size_t>(n)) {
      return 25;
    }
    const std::size_t bytes = input.size() * sizeof(float);
    CUdeviceptr d_in = 0;
    CUdeviceptr d_out = 0;
    if (cuMemAlloc(&d_in, bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&d_out, bytes) != CUDA_SUCCESS) {
      return 26;
    }
    if (cuMemcpyHtoD(d_in, input.data(), bytes) != CUDA_SUCCESS) {
      return 27;
    }
    int count = n;
    std::vector<void *> params = {&d_in, &d_out, &count};
    const unsigned int block = 256;
    const unsigned int grid =
        (static_cast<unsigned int>(n) + block - 1) / block;
    if (!run_kernel(GELU_PTX_PATH, "gelu", params, grid, block)) {
      return 28;
    }
    std::vector<float> output(static_cast<std::size_t>(n));
    if (cuMemcpyDtoH(output.data(), d_out, bytes) != CUDA_SUCCESS) {
      return 29;
    }
    if (!write_floats(out_path, output)) {
      return 30;
    }
    cuMemFree(d_in);
    cuMemFree(d_out);
    return 0;
  }

  return 16;
}
