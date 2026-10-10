#include "loader/module_registry.h"

#include <cstdint>

extern "C" {
void **__cudaRegisterFatBinary(void *fat_cubin);
void __cudaRegisterFunction(void **fat_cubin_handle, const char *host_func,
                            char *device_func, const char *device_name,
                            int thread_limit, int *tid, int *bid,
                            int *block_dim, int *grid_dim, int *warp_size);
void __cudaUnregisterFatBinary(void **fat_cubin_handle);
}

int main() {
  void **handle = __cudaRegisterFatBinary(nullptr);
  if (handle == nullptr) {
    return 1;
  }

  char device_func[] = "add_kernel";
  char device_name[] = "add_kernel";
  __cudaRegisterFunction(handle, "host_add", device_func, device_name, 0,
                         nullptr, nullptr, nullptr, nullptr, nullptr);

  const cudasim::loader::RegisteredFunction *registered =
      cudasim::loader::ModuleRegistry::instance().find_runtime_function_by_host(
          "host_add");
  if (registered == nullptr) {
    return 2;
  }
  if (registered->function_name != "add_kernel") {
    return 3;
  }

  __cudaUnregisterFatBinary(handle);
  return 0;
}
