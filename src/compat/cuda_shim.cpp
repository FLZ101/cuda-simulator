#include "compat/cuda_api.h"
#include "compat/export.h"
#include "core/compat_state.h"
#include "exec/executor.h"
#include "loader/fatbin.h"
#include "loader/module_registry.h"
#include "ptx/lower.h"
#include "ptx/module.h"
#include "runtime/runtime.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

namespace {

thread_local CUresult g_last_result = CUDA_SUCCESS;
std::atomic<std::uintptr_t> g_handle_counter{1};

CUresult result(bool ok, CUresult failure) {
  const CUresult code = ok ? CUDA_SUCCESS : failure;
  g_last_result = code;
  return code;
}

void *dummy_handle() { return reinterpret_cast<void *>(1); }

std::shared_ptr<cudasim::runtime::Stream> get_stream(CUstream stream) {
  const auto handle = reinterpret_cast<std::uintptr_t>(stream);
  if (handle == 0) {
    return {};
  }
  try {
    return cudasim::runtime::StreamManager::instance().get(handle);
  } catch (...) {
    return {};
  }
}

std::shared_ptr<cudasim::runtime::Event> get_event(CUevent event) {
  const auto handle = reinterpret_cast<std::uintptr_t>(event);
  try {
    return cudasim::runtime::EventManager::instance().get(handle);
  } catch (...) {
    return {};
  }
}

} // namespace

CUDASIM_EXTERN_C CUDASIM_API CUresult cuInit(unsigned int flags) {
  (void)flags;
  cudasim::core::CompatState::instance().initialize();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuDeviceGetCount(int *count) {
  if (count == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *count = cudasim::core::CompatState::instance().device_count();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuDeviceGet(CUdevice *device,
                                                  int ordinal) {
  if (device == nullptr || ordinal != 0) {
    return result(false, CUDA_ERROR_INVALID_DEVICE);
  }
  *device = 0;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuDeviceGetName(char *name, int len,
                                                      CUdevice device) {
  if (name == nullptr || len <= 0 || device != 0) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  const auto &info = cudasim::core::CompatState::instance().device_info();
  std::strncpy(name, info.name, static_cast<std::size_t>(len) - 1);
  name[len - 1] = '\0';
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuDeviceGetAttribute(int *value, CUdevice_attribute attr, CUdevice device) {
  if (value == nullptr || device != 0) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }

  const auto &info = cudasim::core::CompatState::instance().device_info();
  switch (attr) {
  case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
    *value = info.max_threads_per_block;
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
    *value = info.max_threads_dim[0];
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
    *value = info.max_threads_dim[1];
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
    *value = info.max_threads_dim[2];
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
    *value = info.max_grid_size[0];
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
    *value = info.max_grid_size[1];
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
    *value = info.max_grid_size[2];
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
    *value = info.shared_mem_per_block;
    break;
  case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
    *value = info.warp_size;
    break;
  case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
    *value = info.multi_processor_count;
    break;
  case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
    *value = info.max_threads_per_multi_processor;
    break;
  case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
    *value = info.major;
    break;
  case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
    *value = info.minor;
    break;
  default:
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }

  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice device) {
  if (ctx == nullptr || device != 0) {
    return result(false, CUDA_ERROR_INVALID_DEVICE);
  }
  *ctx = dummy_handle();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuDevicePrimaryCtxRelease(CUdevice device) {
  (void)device;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuCtxCreate(CUcontext *ctx,
                                                  unsigned int flags,
                                                  CUdevice device) {
  (void)flags;
  return cuDevicePrimaryCtxRetain(ctx, device);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuCtxDestroy(CUcontext ctx) {
  (void)ctx;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuCtxSetCurrent(CUcontext ctx) {
  (void)ctx;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuCtxGetCurrent(CUcontext *ctx) {
  if (ctx == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *ctx = dummy_handle();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuCtxGetDevice(CUdevice *device) {
  if (device == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *device = 0;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuCtxSynchronize() {
  cudasim::runtime::StreamManager::instance().synchronize_all();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuModuleLoad(CUmodule *module,
                                                   const char *file) {
  if (module == nullptr || file == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  (void)file;
  return result(false, CUDA_ERROR_NOT_SUPPORTED);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuModuleLoadData(CUmodule *module,
                                                       const void *image) {
  if (module == nullptr || image == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }

  const auto *text = static_cast<const char *>(image);
  const std::size_t size = std::strlen(text);
  cudasim::loader::FatbinResult extracted =
      cudasim::loader::extract_ptx(image, size);
  if (extracted.cubin_only || extracted.ptx_sources.empty()) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }

  cudasim::ptx::Module parsed =
      cudasim::ptx::parse_ptx(extracted.ptx_sources.front());
  const std::uint64_t handle =
      cudasim::loader::ModuleRegistry::instance().register_module(parsed);
  *module = reinterpret_cast<CUmodule>(handle);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuModuleLoadDataEx(
    CUmodule *module, const void *image, unsigned int num_options, int *options,
    void **option_values) {
  (void)num_options;
  (void)options;
  (void)option_values;
  return cuModuleLoadData(module, image);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuModuleLoadFatBinary(CUmodule *module, const void *fat_cubin) {
  return cuModuleLoadData(module, fat_cubin);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuModuleGetFunction(CUfunction *function,
                                                          CUmodule module,
                                                          const char *name) {
  (void)module;
  if (function == nullptr || name == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  const cudasim::ptx::Function *found =
      cudasim::loader::ModuleRegistry::instance().find_function(name);
  if (found == nullptr) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }
  *function = reinterpret_cast<CUfunction>(const_cast<char *>(name));
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuModuleUnload(CUmodule module) {
  (void)module;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuFuncGetAttribute(int *value, int attr,
                                                         CUfunction function) {
  (void)attr;
  (void)function;
  if (value == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *value = 0;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuFuncSetAttribute(CUfunction function,
                                                         int attr, int value) {
  (void)function;
  (void)attr;
  (void)value;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuOccupancyMaxActiveBlocksPerMultiprocessor(int *num_blocks,
                                            CUfunction function, int block_size,
                                            std::size_t dynamic_smem_size) {
  (void)function;
  (void)dynamic_smem_size;
  if (num_blocks == nullptr || block_size <= 0) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *num_blocks = 2048 / block_size;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemAlloc(CUdeviceptr *ptr,
                                                 std::size_t size) {
  if (ptr == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *ptr = reinterpret_cast<CUdeviceptr>(
      cudasim::core::CompatState::instance().device_alloc(size));
  return result(*ptr != 0 || size == 0, CUDA_ERROR_OUT_OF_MEMORY);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemFree(CUdeviceptr ptr) {
  cudasim::core::CompatState::instance().device_free(
      reinterpret_cast<void *>(ptr));
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemAllocManaged(CUdeviceptr *ptr,
                                                        std::size_t size,
                                                        unsigned int flags) {
  (void)flags;
  return cuMemAlloc(ptr, size);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemHostAlloc(void **ptr,
                                                     std::size_t size,
                                                     unsigned int flags) {
  (void)flags;
  if (ptr == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *ptr = cudasim::core::CompatState::instance().host_alloc(size);
  return result(*ptr != nullptr || size == 0, CUDA_ERROR_OUT_OF_MEMORY);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemFreeHost(void *ptr) {
  cudasim::core::CompatState::instance().host_free(ptr);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemGetInfo(std::size_t *free,
                                                   std::size_t *total) {
  const auto &info = cudasim::core::CompatState::instance().device_info();
  if (free != nullptr) {
    *free = info.total_global_mem;
  }
  if (total != nullptr) {
    *total = info.total_global_mem;
  }
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuPointerGetAttribute(void *data,
                                                            int attribute,
                                                            CUdeviceptr ptr) {
  (void)data;
  (void)attribute;
  (void)ptr;
  return result(false, CUDA_ERROR_NOT_SUPPORTED);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemcpyHtoD(CUdeviceptr dst,
                                                   const void *src,
                                                   std::size_t count) {
  if (src == nullptr && count != 0) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  std::memcpy(reinterpret_cast<void *>(dst), src, count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src,
                                                   std::size_t count) {
  if (dst == nullptr && count != 0) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  std::memcpy(dst, reinterpret_cast<const void *>(src), count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemcpyDtoD(CUdeviceptr dst,
                                                   CUdeviceptr src,
                                                   std::size_t count) {
  std::memcpy(reinterpret_cast<void *>(dst),
              reinterpret_cast<const void *>(src), count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemcpyHtoDAsync(CUdeviceptr dst,
                                                        const void *src,
                                                        std::size_t count,
                                                        CUstream stream) {
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return cuMemcpyHtoD(dst, src, count);
  }
  runtime_stream->submit_copy(reinterpret_cast<void *>(dst), src, count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemcpyDtoHAsync(void *dst,
                                                        CUdeviceptr src,
                                                        std::size_t count,
                                                        CUstream stream) {
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return cuMemcpyDtoH(dst, src, count);
  }
  runtime_stream->submit_copy(dst, reinterpret_cast<const void *>(src), count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemcpyDtoDAsync(CUdeviceptr dst,
                                                        CUdeviceptr src,
                                                        std::size_t count,
                                                        CUstream stream) {
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return cuMemcpyDtoD(dst, src, count);
  }
  runtime_stream->submit_copy(reinterpret_cast<void *>(dst),
                              reinterpret_cast<const void *>(src), count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemsetD8(CUdeviceptr dst,
                                                 unsigned char value,
                                                 std::size_t count) {
  std::memset(reinterpret_cast<void *>(dst), value, count);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuMemsetD32(CUdeviceptr dst,
                                                  unsigned int value,
                                                  std::size_t count) {
  const std::size_t bytes = count * sizeof(unsigned int);
  std::memset(reinterpret_cast<void *>(dst), 0, bytes);
  auto *out = reinterpret_cast<unsigned int *>(dst);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = value;
  }
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuStreamCreate(CUstream *stream,
                                                     unsigned int flags) {
  (void)flags;
  if (stream == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  auto created = cudasim::runtime::StreamManager::instance().create();
  *stream = reinterpret_cast<void *>(created->id());
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority) {
  (void)priority;
  return cuStreamCreate(stream, flags);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuStreamGetPriority(CUstream stream,
                                                          int *priority) {
  (void)stream;
  if (priority == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *priority = 0;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuStreamDestroy(CUstream stream) {
  const auto handle = reinterpret_cast<std::uintptr_t>(stream);
  if (handle != 0) {
    cudasim::runtime::StreamManager::instance().destroy(handle);
  }
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuStreamSynchronize(CUstream stream) {
  if (stream == nullptr) {
    cudasim::runtime::StreamManager::instance().synchronize_all();
    return result(true, CUDA_SUCCESS);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  runtime_stream->synchronize();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuStreamQuery(CUstream stream) {
  if (stream == nullptr) {
    cudasim::runtime::StreamManager::instance().synchronize_all();
    return result(true, CUDA_SUCCESS);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  if (!runtime_stream->query()) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuStreamWaitEvent(CUstream stream,
                                                        CUevent event,
                                                        unsigned int flags) {
  (void)flags;
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  runtime_stream->submit_wait(runtime_event);
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuEventCreate(CUevent *event,
                                                    unsigned int flags) {
  (void)flags;
  if (event == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  auto created = cudasim::runtime::EventManager::instance().create();
  *event = reinterpret_cast<void *>(created->id());
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuEventDestroy(CUevent event) {
  cudasim::runtime::EventManager::instance().destroy(
      reinterpret_cast<std::uintptr_t>(event));
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuEventRecord(CUevent event,
                                                    CUstream stream) {
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    runtime_event->record();
  } else {
    runtime_stream->submit_event(runtime_event);
  }
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuEventSynchronize(CUevent event) {
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  runtime_event->wait();
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuEventQuery(CUevent event) {
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, CUDA_ERROR_INVALID_CONTEXT);
  }
  if (!runtime_event->is_recorded()) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult cuEventElapsedTime(float *ms,
                                                         CUevent start,
                                                         CUevent end) {
  (void)start;
  (void)end;
  if (ms == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }
  *ms = 0.0F;
  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API CUresult
cuLaunchKernel(CUfunction function, unsigned int grid_x, unsigned int grid_y,
               unsigned int grid_z, unsigned int block_x, unsigned int block_y,
               unsigned int block_z, unsigned int shared_mem, CUstream stream,
               void **params, void **extra) {
  (void)stream;
  (void)extra;

  if (function == nullptr || params == nullptr) {
    return result(false, CUDA_ERROR_INVALID_VALUE);
  }

  const auto *name = reinterpret_cast<const char *>(function);
  const cudasim::ptx::Module *owner =
      cudasim::loader::ModuleRegistry::instance().find_module_for_function(
          name);
  if (owner == nullptr) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }

  cudasim::ptx::ModuleIR ir = cudasim::ptx::lower(*owner);
  if (ir.functions.empty()) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }
  const cudasim::ptx::FunctionIR *function_ir = nullptr;
  for (const auto &candidate : ir.functions) {
    if (candidate.name == name) {
      function_ir = &candidate;
      break;
    }
  }
  if (function_ir == nullptr) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }

  cudasim::exec::LaunchConfig config;
  config.grid = {grid_x, grid_y, grid_z};
  config.block = {block_x, block_y, block_z};
  config.shared_memory = shared_mem;

  std::vector<std::uint8_t> arguments;
  for (std::size_t i = 0; i < function_ir->parameters.size(); ++i) {
    const std::size_t size = function_ir->parameters[i].size;
    const auto *src = static_cast<const std::uint8_t *>(params[i]);
    arguments.insert(arguments.end(), src, src + size);
  }

  cudasim::exec::Executor executor;
  if (!executor.execute(*function_ir, config, arguments, ir.variables)) {
    return result(false, CUDA_ERROR_NOT_SUPPORTED);
  }

  return result(true, CUDA_SUCCESS);
}

CUDASIM_EXTERN_C CUDASIM_API const char *cuGetErrorName(CUresult code) {
  (void)code;
  return "CUDA_ERROR";
}

CUDASIM_EXTERN_C CUDASIM_API const char *cuGetErrorString(CUresult code) {
  switch (code) {
  case CUDA_SUCCESS:
    return "no error";
  case CUDA_ERROR_INVALID_VALUE:
    return "invalid argument";
  case CUDA_ERROR_OUT_OF_MEMORY:
    return "out of memory";
  case CUDA_ERROR_NOT_INITIALIZED:
    return "not initialized";
  case CUDA_ERROR_INVALID_DEVICE:
    return "invalid device";
  case CUDA_ERROR_INVALID_CONTEXT:
    return "invalid context";
  case CUDA_ERROR_NOT_SUPPORTED:
    return "operation not supported";
  default:
    return "unknown error";
  }
}
