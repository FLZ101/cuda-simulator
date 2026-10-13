#include "core/compat_state.h"

#include "runtime/memory.h"

#include <algorithm>
#include <cstring>

namespace cudasim::core {

CompatState &CompatState::instance() {
  static CompatState state;
  return state;
}

void CompatState::initialize() { (void)0; }

bool CompatState::is_initialized() const { return true; }

int CompatState::device_count() const {
  return cudasim::runtime::DeviceRegistry::instance().count();
}

const DeviceInfo &CompatState::device_info() const {
  const auto &props =
      cudasim::runtime::DeviceRegistry::instance().device(0).properties();
  static DeviceInfo info;
  const std::size_t source_len = ::strnlen(props.name, sizeof(props.name));
  const std::size_t copy_len = std::min(source_len, sizeof(info.name) - 1);
  std::memcpy(info.name, props.name, copy_len);
  info.name[copy_len] = '\0';
  info.major = props.major;
  info.minor = props.minor;
  info.multi_processor_count = props.multi_processor_count;
  info.max_threads_per_block = props.max_threads_per_block;
  info.max_threads_per_multi_processor = props.max_threads_per_multi_processor;
  info.warp_size = props.warp_size;
  for (std::size_t i = 0; i < 3; ++i) {
    info.max_grid_size[i] = props.max_grid_size[i];
    info.max_threads_dim[i] = props.max_threads_dim[i];
  }
  info.shared_mem_per_block = props.shared_mem_per_block;
  info.regs_per_block = props.regs_per_block;
  info.clock_rate = props.clock_rate;
  info.total_global_mem = props.total_global_mem;
  info.integrated = props.integrated;
  return info;
}

void *CompatState::device_alloc(std::size_t size) {
  return cudasim::runtime::MemoryManager::instance().allocate_device(size);
}

void CompatState::device_free(void *ptr) {
  cudasim::runtime::MemoryManager::instance().free_device(ptr);
}

void *CompatState::host_alloc(std::size_t size) {
  return cudasim::runtime::MemoryManager::instance().allocate_host(size);
}

void CompatState::host_free(void *ptr) {
  cudasim::runtime::MemoryManager::instance().free_host(ptr);
}

} // namespace cudasim::core
