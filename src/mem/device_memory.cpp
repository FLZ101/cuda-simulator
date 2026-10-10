#include "mem/device_memory.h"

namespace cudasim::mem {

DeviceBuffer::DeviceBuffer(std::size_t size) : bytes_(size, 0) {}

std::size_t DeviceBuffer::size() const { return bytes_.size(); }

bool DeviceBuffer::contains(std::size_t offset, std::size_t size) const {
  return offset <= bytes_.size() && size <= bytes_.size() - offset;
}

std::uint8_t *DeviceBuffer::data() { return bytes_.data(); }

const std::uint8_t *DeviceBuffer::data() const { return bytes_.data(); }

} // namespace cudasim::mem
