#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cudasim::runtime {

struct DeviceProperties {
  char name[256]{};
  int major = 8;
  int minor = 0;
  int multi_processor_count = 8;
  int max_threads_per_block = 1024;
  int max_threads_per_multi_processor = 2048;
  int warp_size = 32;
  std::array<int, 3> max_grid_size{2147483647, 65535, 65535};
  std::array<int, 3> max_threads_dim{1024, 1024, 64};
  int shared_mem_per_block = 49152;
  int regs_per_block = 65536;
  int clock_rate = 1000000;
  std::size_t total_global_mem = std::size_t{8} * 1024 * 1024 * 1024;
  int integrated = 1;
};

class Device {
public:
  explicit Device(int id);

  int id() const;
  const DeviceProperties &properties() const;

private:
  int id_;
  DeviceProperties properties_;
};

class DeviceRegistry {
public:
  static DeviceRegistry &instance();

  Device &device(int ordinal);
  int count() const;

private:
  DeviceRegistry();

  Device device_;
};

} // namespace cudasim::runtime
