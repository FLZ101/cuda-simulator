#include "core/compat_state.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace cudasim::core {

namespace {

constexpr std::size_t kDeviceAlignment = 256;
constexpr std::size_t kTotalMemory = std::size_t{8} * 1024 * 1024 * 1024;

} // namespace

CompatState &CompatState::instance() {
  static CompatState state;
  return state;
}

void CompatState::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return;
  }

  device_info_ = DeviceInfo{
      "CUDA Simulator Device",
      8,
      0,
      8,
      1024,
      2048,
      32,
      {2147483647, 65535, 65535},
      {1024, 1024, 64},
      49152,
      65536,
      1000000,
      kTotalMemory,
      1,
  };
  initialized_ = true;
}

bool CompatState::is_initialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_;
}

int CompatState::device_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_ ? 1 : 0;
}

const DeviceInfo &CompatState::device_info() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return device_info_;
}

void *CompatState::device_alloc(std::size_t size) {
  if (size == 0) {
    return nullptr;
  }

  void *ptr = nullptr;
  if (posix_memalign(&ptr, kDeviceAlignment, size) != 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  device_allocations_.emplace(ptr, size);
  return ptr;
}

void CompatState::device_free(void *ptr) {
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

void *CompatState::host_alloc(std::size_t size) {
  void *ptr = std::malloc(size);
  if (ptr == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  host_allocations_.emplace(ptr, size);
  return ptr;
}

void CompatState::host_free(void *ptr) {
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

} // namespace cudasim::core
