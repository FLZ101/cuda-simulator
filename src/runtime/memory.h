#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace cudasim::runtime {

enum class MemoryKind {
  kDevice,
  kHost,
};

struct Allocation {
  void *ptr = nullptr;
  std::size_t size = 0;
  MemoryKind kind = MemoryKind::kDevice;
};

class MemoryManager {
public:
  static MemoryManager &instance();

  void *allocate_device(std::size_t size);
  void free_device(void *ptr);
  void *allocate_host(std::size_t size);
  void free_host(void *ptr);
  bool is_device_pointer(void *ptr) const;
  bool is_host_pointer(void *ptr) const;

private:
  MemoryManager() = default;

  mutable std::mutex mutex_;
  std::unordered_map<void *, Allocation> device_allocations_;
  std::unordered_map<void *, Allocation> host_allocations_;
};

} // namespace cudasim::runtime
