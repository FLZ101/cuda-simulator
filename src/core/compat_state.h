#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime/device.h"

namespace cudasim::core {

struct DeviceInfo {
  char name[256];
  int major;
  int minor;
  int multi_processor_count;
  int max_threads_per_block;
  int max_threads_per_multi_processor;
  int warp_size;
  int max_grid_size[3];
  int max_threads_dim[3];
  int shared_mem_per_block;
  int regs_per_block;
  int clock_rate;
  std::size_t total_global_mem;
  int integrated;
};

class CompatState {
public:
  static CompatState &instance();

  void initialize();
  bool is_initialized() const;
  int device_count() const;
  const DeviceInfo &device_info() const;

  void *device_alloc(std::size_t size);
  void device_free(void *ptr);
  void *host_alloc(std::size_t size);
  void host_free(void *ptr);

private:
  CompatState() = default;
};

} // namespace cudasim::core
