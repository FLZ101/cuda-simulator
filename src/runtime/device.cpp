#include "runtime/device.h"

#include <cstring>
#include <stdexcept>

namespace cudasim::runtime {

Device::Device(int id) : id_(id) {
  std::strncpy(properties_.name, "CUDA Simulator Device",
               sizeof(properties_.name) - 1);
}

int Device::id() const { return id_; }

const DeviceProperties &Device::properties() const { return properties_; }

DeviceRegistry::DeviceRegistry() : device_(0) {}

DeviceRegistry &DeviceRegistry::instance() {
  static DeviceRegistry registry;
  return registry;
}

Device &DeviceRegistry::device(int ordinal) {
  if (ordinal != 0) {
    throw std::out_of_range("only device 0 is supported");
  }
  return device_;
}

int DeviceRegistry::count() const { return 1; }

} // namespace cudasim::runtime
