#include "compat/cuda_api.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

namespace {

int parse_int(const char *value, int fallback) {
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

} // namespace

int main(int argc, char **argv) {
  int n = 262144;
  int iterations = 2048;
  int block = 256;
  int runs = 3;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      n = parse_int(argv[++i], n);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = parse_int(argv[++i], iterations);
    } else if (arg == "--block" && i + 1 < argc) {
      block = parse_int(argv[++i], block);
    } else if (arg == "--runs" && i + 1 < argc) {
      runs = parse_int(argv[++i], runs);
    }
  }

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
  if (cuModuleGetFunction(&function, module, "block_parallel_kernel") !=
      CUDA_SUCCESS) {
    return 4;
  }

  std::vector<float> input_data(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i) * 0.001f;
  }

  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  if (cuMemAlloc(&d_in, input_data.size() * sizeof(float)) != CUDA_SUCCESS ||
      cuMemAlloc(&d_out, input_data.size() * sizeof(float)) != CUDA_SUCCESS) {
    return 5;
  }
  if (cuMemcpyHtoD(d_in, input_data.data(),
                   input_data.size() * sizeof(float)) != CUDA_SUCCESS) {
    return 6;
  }

  const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
  void *params[] = {&d_in, &d_out, &n, &iterations};

  if (cuLaunchKernel(function, grid, 1, 1, static_cast<unsigned int>(block), 1,
                     1, 0, nullptr, params, nullptr) != CUDA_SUCCESS) {
    return 7;
  }

  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  for (int run = 0; run < runs; ++run) {
    if (cuLaunchKernel(function, grid, 1, 1, static_cast<unsigned int>(block),
                       1, 1, 0, nullptr, params, nullptr) != CUDA_SUCCESS) {
      return 8;
    }
  }
  const auto end = clock::now();

  std::vector<float> output(input_data.size());
  if (cuMemcpyDtoH(output.data(), d_out, output.size() * sizeof(float)) !=
      CUDA_SUCCESS) {
    return 9;
  }

  double checksum = 0.0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (!std::isfinite(output[i])) {
      return 10;
    }
    checksum += static_cast<double>(output[i]);
  }

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const double avg_ms = elapsed_ms / static_cast<double>(runs);

  std::cout << "blocks=" << grid << " threads=" << n
            << " iterations=" << iterations << " avg_ms=" << avg_ms
            << " checksum=" << checksum << '\n';

  cuMemFree(d_in);
  cuMemFree(d_out);
  return 0;
}
