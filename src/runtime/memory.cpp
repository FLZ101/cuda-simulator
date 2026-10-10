#include "runtime/memory.h"

#include <cstdlib>

namespace cudasim::runtime {

namespace {

constexpr std::size_t kDeviceAlignment = 256;

} // namespace

MemoryManager &MemoryManager::instance() {
  static MemoryManager manager;
  return manager;
}

void *MemoryManager::allocate_device(std::size_t size) {
  if (size == 0) {
    return nullptr;
  }

  void *ptr = nullptr;
  if (posix_memalign(&ptr, kDeviceAlignment, size) != 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  device_allocations_.emplace(ptr, Allocation{ptr, size, MemoryKind::kDevice});
  return ptr;
}

void MemoryManager::free_device(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = device_allocations_.find(ptr);
    if (it == device_allocations_.end()) {
      return;
    }
    device_allocations_.erase(it);
  }

  std::free(ptr);
}

void *MemoryManager::allocate_host(std::size_t size) {
  void *ptr = std::malloc(size);
  if (ptr == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  host_allocations_.emplace(ptr, Allocation{ptr, size, MemoryKind::kHost});
  return ptr;
}

void MemoryManager::free_host(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = host_allocations_.find(ptr);
    if (it == host_allocations_.end()) {
      return;
    }
    host_allocations_.erase(it);
  }

  std::free(ptr);
}

bool MemoryManager::is_device_pointer(void *ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return device_allocations_.find(ptr) != device_allocations_.end();
}

bool MemoryManager::is_host_pointer(void *ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return host_allocations_.find(ptr) != host_allocations_.end();
}

} // namespace cudasim::runtime
