#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudasim::mem {

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t size);

  std::size_t size() const;
  bool contains(std::size_t offset, std::size_t size) const;
  std::uint8_t *data();
  const std::uint8_t *data() const;

private:
  std::vector<std::uint8_t> bytes_;
};

} // namespace cudasim::mem
