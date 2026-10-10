#include "compat/cuda_api.h"
#include "compat/export.h"
#include "core/compat_state.h"
#include "runtime/runtime.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>

namespace {

thread_local cudaError_t g_last_error = cudaSuccess;
std::atomic<std::uintptr_t> g_handle_counter{1};

void set_last_error(cudaError_t error) { g_last_error = error; }

cudaError_t result(bool ok, cudaError_t failure) {
  const cudaError_t error = ok ? cudaSuccess : failure;
  set_last_error(error);
  return error;
}

void fill_device_properties(cudaDeviceProp *props) {
  const auto &info = cudasim::core::CompatState::instance().device_info();
  std::memset(props, 0, sizeof(*props));
  std::strncpy(props->name, info.name, sizeof(props->name) - 1);
  props->major = info.major;
  props->minor = info.minor;
  props->multiProcessorCount = info.multi_processor_count;
  props->maxThreadsPerBlock = info.max_threads_per_block;
  props->maxThreadsPerMultiProcessor = info.max_threads_per_multi_processor;
  props->warpSize = info.warp_size;
  for (int i = 0; i < 3; ++i) {
    props->maxGridSize[i] = info.max_grid_size[i];
    props->maxThreadsDim[i] = info.max_threads_dim[i];
  }
  props->sharedMemPerBlock = info.shared_mem_per_block;
  props->regsPerBlock = info.regs_per_block;
  props->clockRate = info.clock_rate;
  props->totalGlobalMem = info.total_global_mem;
  props->integrated = info.integrated;
}

std::shared_ptr<cudasim::runtime::Stream> get_stream(cudaStream_t stream) {
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

std::shared_ptr<cudasim::runtime::Event> get_event(cudaEvent_t event) {
  const auto handle = reinterpret_cast<std::uintptr_t>(event);
  try {
    return cudasim::runtime::EventManager::instance().get(handle);
  } catch (...) {
    return {};
  }
}

} // namespace

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaGetDeviceCount(int *count) {
  cudasim::core::CompatState::instance().initialize();
  if (count == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  *count = cudasim::core::CompatState::instance().device_count();
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaSetDevice(int device) {
  cudasim::core::CompatState::instance().initialize();
  return result(device == 0, cudaErrorInvalidDevice);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaGetDevice(int *device) {
  cudasim::core::CompatState::instance().initialize();
  if (device == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  *device = 0;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaGetDeviceProperties(cudaDeviceProp *props, int device) {
  cudasim::core::CompatState::instance().initialize();
  if (props == nullptr || device != 0) {
    return result(false, cudaErrorInvalidValue);
  }
  fill_device_properties(props);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
  cudasim::core::CompatState::instance().initialize();
  if (value == nullptr || device != 0) {
    return result(false, cudaErrorInvalidValue);
  }

  const auto &info = cudasim::core::CompatState::instance().device_info();
  switch (attr) {
  case cudaDevAttrMaxThreadsPerBlock:
    *value = info.max_threads_per_block;
    break;
  case cudaDevAttrMaxBlockDimX:
    *value = info.max_threads_dim[0];
    break;
  case cudaDevAttrMaxBlockDimY:
    *value = info.max_threads_dim[1];
    break;
  case cudaDevAttrMaxBlockDimZ:
    *value = info.max_threads_dim[2];
    break;
  case cudaDevAttrMaxGridDimX:
    *value = info.max_grid_size[0];
    break;
  case cudaDevAttrMaxGridDimY:
    *value = info.max_grid_size[1];
    break;
  case cudaDevAttrMaxGridDimZ:
    *value = info.max_grid_size[2];
    break;
  case cudaDevAttrMaxSharedMemoryPerBlock:
    *value = info.shared_mem_per_block;
    break;
  case cudaDevAttrWarpSize:
    *value = info.warp_size;
    break;
  case cudaDevAttrMultiProcessorCount:
    *value = info.multi_processor_count;
    break;
  case cudaDevAttrMaxThreadsPerMultiProcessor:
    *value = info.max_threads_per_multi_processor;
    break;
  case cudaDevAttrComputeCapabilityMajor:
    *value = info.major;
    break;
  case cudaDevAttrComputeCapabilityMinor:
    *value = info.minor;
    break;
  default:
    return result(false, cudaErrorInvalidValue);
  }

  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMalloc(void **ptr,
                                                    std::size_t size) {
  cudasim::core::CompatState::instance().initialize();
  if (ptr == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  *ptr = cudasim::core::CompatState::instance().device_alloc(size);
  return result(*ptr != nullptr || size == 0, cudaErrorMemoryAllocation);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaFree(void *ptr) {
  cudasim::core::CompatState::instance().device_free(ptr);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMemcpy(void *dst, const void *src,
                                                    std::size_t count,
                                                    cudaMemcpyKind kind) {
  (void)kind;
  if ((dst == nullptr || src == nullptr) && count != 0) {
    return result(false, cudaErrorInvalidValue);
  }
  std::memcpy(dst, src, count);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMemcpyAsync(void *dst,
                                                         const void *src,
                                                         std::size_t count,
                                                         cudaMemcpyKind kind,
                                                         cudaStream_t stream) {
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return cudaMemcpy(dst, src, count, kind);
  }
  runtime_stream->submit_copy(dst, src, count);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMemset(void *dst, int value,
                                                    std::size_t count) {
  if (dst == nullptr && count != 0) {
    return result(false, cudaErrorInvalidValue);
  }
  std::memset(dst, value, count);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMemsetAsync(void *dst, int value,
                                                         std::size_t count,
                                                         cudaStream_t stream) {
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return cudaMemset(dst, value, count);
  }
  runtime_stream->submit_memset(dst, value, count);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaMemcpy2D(void *dst, std::size_t dpitch, const void *src, std::size_t spitch,
             std::size_t width, std::size_t height, cudaMemcpyKind kind) {
  (void)dst;
  (void)dpitch;
  (void)src;
  (void)spitch;
  (void)width;
  (void)height;
  (void)kind;
  return result(false, cudaErrorNotSupported);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaMemcpy2DAsync(void *dst, std::size_t dpitch, const void *src,
                  std::size_t spitch, std::size_t width, std::size_t height,
                  cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMemset2D(void *dst,
                                                      std::size_t pitch,
                                                      int value,
                                                      std::size_t width,
                                                      std::size_t height) {
  (void)dst;
  (void)pitch;
  (void)value;
  (void)width;
  (void)height;
  return result(false, cudaErrorNotSupported);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaMemset2DAsync(void *dst, std::size_t pitch, int value, std::size_t width,
                  std::size_t height, cudaStream_t stream) {
  (void)stream;
  return cudaMemset2D(dst, pitch, value, width, height);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaMallocManaged(void **ptr,
                                                           std::size_t size) {
  return cudaMalloc(ptr, size);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaHostAlloc(void **ptr,
                                                       std::size_t size,
                                                       unsigned int flags) {
  (void)flags;
  if (ptr == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  *ptr = cudasim::core::CompatState::instance().host_alloc(size);
  return result(*ptr != nullptr || size == 0, cudaErrorMemoryAllocation);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaFreeHost(void *ptr) {
  cudasim::core::CompatState::instance().host_free(ptr);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaHostRegister(void *ptr,
                                                          std::size_t size,
                                                          unsigned int flags) {
  (void)ptr;
  (void)size;
  (void)flags;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaHostUnregister(void *ptr) {
  (void)ptr;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaHostGetDevicePointer(
    void **device_ptr, void *host_ptr, unsigned int flags) {
  (void)flags;
  if (device_ptr == nullptr || host_ptr == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  *device_ptr = host_ptr;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaMemcpyToSymbol(const void *symbol, const void *src, std::size_t count,
                   std::size_t offset, cudaMemcpyKind kind) {
  (void)symbol;
  (void)src;
  (void)count;
  (void)offset;
  (void)kind;
  return result(false, cudaErrorNotSupported);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaMemcpyFromSymbol(void *dst, const void *symbol, std::size_t count,
                     std::size_t offset, cudaMemcpyKind kind) {
  (void)dst;
  (void)symbol;
  (void)count;
  (void)offset;
  (void)kind;
  return result(false, cudaErrorNotSupported);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaGetSymbolAddress(void **device_ptr, const void *symbol) {
  (void)device_ptr;
  (void)symbol;
  return result(false, cudaErrorNotSupported);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaStreamCreate(cudaStream_t *stream) {
  if (stream == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  auto created = cudasim::runtime::StreamManager::instance().create();
  *stream = reinterpret_cast<void *>(created->id());
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  (void)flags;
  return cudaStreamCreate(stream);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaStreamCreateWithPriority(
    cudaStream_t *stream, unsigned int flags, int priority) {
  (void)flags;
  (void)priority;
  return cudaStreamCreate(stream);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaStreamDestroy(cudaStream_t stream) {
  const auto handle = reinterpret_cast<std::uintptr_t>(stream);
  if (handle != 0) {
    cudasim::runtime::StreamManager::instance().destroy(handle);
  }
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaStreamSynchronize(cudaStream_t stream) {
  if (stream == nullptr) {
    cudasim::runtime::StreamManager::instance().synchronize_all();
    return result(true, cudaSuccess);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  runtime_stream->synchronize();
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaStreamQuery(cudaStream_t stream) {
  if (stream == nullptr) {
    cudasim::runtime::StreamManager::instance().synchronize_all();
    return result(true, cudaSuccess);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  if (!runtime_stream->query()) {
    return result(false, cudaErrorNotSupported);
  }
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  (void)flags;
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  runtime_stream->submit_wait(runtime_event);
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaEventCreate(cudaEvent_t *event) {
  if (event == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  auto created = cudasim::runtime::EventManager::instance().create();
  *event = reinterpret_cast<void *>(created->id());
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  (void)flags;
  return cudaEventCreate(event);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaEventDestroy(cudaEvent_t event) {
  cudasim::runtime::EventManager::instance().destroy(
      reinterpret_cast<std::uintptr_t>(event));
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaEventRecord(cudaEvent_t event,
                                                         cudaStream_t stream) {
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  auto runtime_stream = get_stream(stream);
  if (!runtime_stream) {
    runtime_event->record();
  } else {
    runtime_stream->submit_event(runtime_event);
  }
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaEventSynchronize(cudaEvent_t event) {
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  runtime_event->wait();
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaEventQuery(cudaEvent_t event) {
  auto runtime_event = get_event(event);
  if (!runtime_event) {
    return result(false, cudaErrorInvalidResourceHandle);
  }
  if (!runtime_event->is_recorded()) {
    return result(false, cudaErrorNotSupported);
  }
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaEventElapsedTime(float *ms,
                                                              cudaEvent_t start,
                                                              cudaEvent_t end) {
  (void)start;
  (void)end;
  if (ms == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  *ms = 0.0F;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaLaunchKernel(
    const void *func, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_mem, cudaStream_t stream,
    void **args, void **extra) {
  (void)func;
  (void)grid_x;
  (void)grid_y;
  (void)grid_z;
  (void)block_x;
  (void)block_y;
  (void)block_z;
  (void)shared_mem;
  (void)stream;
  (void)args;
  (void)extra;
  return result(false, cudaErrorNotSupported);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaFuncGetAttributes(cudaFuncAttributes *attrs, const void *func) {
  (void)func;
  if (attrs == nullptr) {
    return result(false, cudaErrorInvalidValue);
  }
  std::memset(attrs, 0, sizeof(*attrs));
  attrs->maxThreadsPerBlock = 1024;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaFuncSetAttribute(const void *func,
                                                              int attr,
                                                              int value) {
  (void)func;
  (void)attr;
  (void)value;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *num_blocks, const void *func,
                                              int block_size,
                                              std::size_t dynamic_smem_size) {
  (void)func;
  (void)dynamic_smem_size;
  if (num_blocks == nullptr || block_size <= 0) {
    return result(false, cudaErrorInvalidValue);
  }
  *num_blocks = 2048 / block_size;
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaDeviceSynchronize() {
  cudasim::runtime::StreamManager::instance().synchronize_all();
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaDeviceReset() {
  return result(true, cudaSuccess);
}

CUDASIM_EXTERN_C CUDASIM_API cudaError_t cudaGetLastError() {
  const cudaError_t error = g_last_error;
  g_last_error = cudaSuccess;
  return error;
}

CUDASIM_EXTERN_C CUDASIM_API const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "no error";
  case cudaErrorInvalidValue:
    return "invalid argument";
  case cudaErrorMemoryAllocation:
    return "out of memory";
  case cudaErrorInvalidDevice:
    return "invalid device ordinal";
  case cudaErrorNotSupported:
    return "operation not supported";
  default:
    return "unknown error";
  }
}

CUDASIM_EXTERN_C CUDASIM_API void **__cudaRegisterFatBinary(void *fat_cubin) {
  static void *handle = nullptr;
  handle = fat_cubin;
  return &handle;
}

CUDASIM_EXTERN_C CUDASIM_API void
__cudaRegisterFunction(void **fat_cubin_handle, const char *host_func,
                       char *device_func, const char *device_name,
                       int thread_limit, int *tid, int *bid, int *block_dim,
                       int *grid_dim, int *warp_size) {
  (void)fat_cubin_handle;
  (void)host_func;
  (void)device_func;
  (void)device_name;
  (void)thread_limit;
  (void)tid;
  (void)bid;
  (void)block_dim;
  (void)grid_dim;
  (void)warp_size;
}

CUDASIM_EXTERN_C CUDASIM_API void
__cudaRegisterVar(void **fat_cubin_handle, char *host_var, char *device_address,
                  const char *device_name, int ext, std::size_t size,
                  int constant, int global) {
  (void)fat_cubin_handle;
  (void)host_var;
  (void)device_address;
  (void)device_name;
  (void)ext;
  (void)size;
  (void)constant;
  (void)global;
}

CUDASIM_EXTERN_C CUDASIM_API void
__cudaRegisterManagedVar(void **fat_cubin_handle, void **host_var_ptr,
                         char *device_address, const char *device_name, int ext,
                         std::size_t size, int constant, int global) {
  (void)fat_cubin_handle;
  (void)host_var_ptr;
  (void)device_address;
  (void)device_name;
  (void)ext;
  (void)size;
  (void)constant;
  (void)global;
}

CUDASIM_EXTERN_C CUDASIM_API void __cudaRegisterFatBinaryEnd(void *fat_cubin) {
  (void)fat_cubin;
}

CUDASIM_EXTERN_C CUDASIM_API void
__cudaUnregisterFatBinary(void **fat_cubin_handle) {
  (void)fat_cubin_handle;
}

CUDASIM_EXTERN_C CUDASIM_API void
__cudaRegisterLinkedBinary(void **fat_cubin_handle, void *binary,
                           void *callback, void *log) {
  (void)fat_cubin_handle;
  (void)binary;
  (void)callback;
  (void)log;
}

CUDASIM_EXTERN_C CUDASIM_API void __cudaRegisterSurface(void **fat_cubin_handle,
                                                        const void *surface,
                                                        const char *device_name,
                                                        int surface_type,
                                                        int surface_rank) {
  (void)fat_cubin_handle;
  (void)surface;
  (void)device_name;
  (void)surface_type;
  (void)surface_rank;
}

CUDASIM_EXTERN_C CUDASIM_API void
__cudaRegisterTexture(void **fat_cubin_handle, const void *texture,
                      const char *device_name, int texture_type, int norm,
                      int read_mode) {
  (void)fat_cubin_handle;
  (void)texture;
  (void)device_name;
  (void)texture_type;
  (void)norm;
  (void)read_mode;
}
