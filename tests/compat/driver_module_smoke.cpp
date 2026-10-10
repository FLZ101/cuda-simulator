#include "compat/cuda_api.h"

#include <fstream>
#include <sstream>
#include <string>

extern "C" {
CUresult cuInit(unsigned int flags);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *name);
CUresult cuModuleUnload(CUmodule module);
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
  if (cuModuleGetFunction(&function, module, "add_kernel") != CUDA_SUCCESS ||
      function == nullptr) {
    return 4;
  }

  if (cuModuleUnload(module) != CUDA_SUCCESS) {
    return 5;
  }

  return 0;
}
