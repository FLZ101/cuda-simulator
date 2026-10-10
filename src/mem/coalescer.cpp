#include "mem/coalescer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cudasim::mem {

AccessPattern classify(const std::vector<std::size_t> &offsets) {
  if (offsets.empty()) {
    return AccessPattern::kUniform;
  }

  bool uniform = true;
  bool contiguous = true;
  bool strided = true;

  for (std::size_t i = 1; i < offsets.size(); ++i) {
    uniform = uniform && offsets[i] == offsets[0];
    contiguous =
        contiguous && offsets[i] == offsets[i - 1] + sizeof(std::uint32_t);
    strided = strided && offsets[i] - offsets[i - 1] == offsets[1] - offsets[0];
  }

  if (uniform) {
    return AccessPattern::kUniform;
  }
  if (contiguous) {
    return AccessPattern::kContiguous;
  }
  if (strided) {
    return AccessPattern::kStrided;
  }
  return AccessPattern::kDivergent;
}

void coalesced_read_u32(const DeviceBuffer &buffer,
                        const std::vector<std::size_t> &offsets,
                        std::vector<std::uint32_t> &output) {
  output.resize(offsets.size());
  if (offsets.empty()) {
    return;
  }

  switch (classify(offsets)) {
  case AccessPattern::kUniform: {
    const std::size_t offset = offsets.front();
    if (!buffer.contains(offset, sizeof(std::uint32_t))) {
      throw std::out_of_range("device buffer read");
    }
    std::uint32_t value = 0;
    std::memcpy(&value, buffer.data() + offset, sizeof(value));
    std::fill(output.begin(), output.end(), value);
    return;
  }
  case AccessPattern::kContiguous: {
    const std::size_t offset = offsets.front();
    const std::size_t bytes = offsets.size() * sizeof(std::uint32_t);
    if (!buffer.contains(offset, bytes)) {
      throw std::out_of_range("device buffer read");
    }
    std::memcpy(output.data(), buffer.data() + offset, bytes);
    return;
  }
  case AccessPattern::kStrided:
  case AccessPattern::kDivergent:
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      if (!buffer.contains(offsets[i], sizeof(std::uint32_t))) {
        throw std::out_of_range("device buffer read");
      }
      std::memcpy(&output[i], buffer.data() + offsets[i], sizeof(output[i]));
    }
    return;
  }
}

void coalesced_write_u32(DeviceBuffer &buffer,
                         const std::vector<std::size_t> &offsets,
                         const std::vector<std::uint32_t> &values) {
  if (offsets.size() != values.size()) {
    throw std::invalid_argument("offset/value size mismatch");
  }
  if (offsets.empty()) {
    return;
  }

  switch (classify(offsets)) {
  case AccessPattern::kUniform: {
    const std::size_t offset = offsets.front();
    if (!buffer.contains(offset, sizeof(std::uint32_t))) {
      throw std::out_of_range("device buffer write");
    }
    std::memcpy(buffer.data() + offset, values.data(), sizeof(values.front()));
    return;
  }
  case AccessPattern::kContiguous: {
    const std::size_t offset = offsets.front();
    const std::size_t bytes = offsets.size() * sizeof(std::uint32_t);
    if (!buffer.contains(offset, bytes)) {
      throw std::out_of_range("device buffer write");
    }
    std::memcpy(buffer.data() + offset, values.data(), bytes);
    return;
  }
  case AccessPattern::kStrided:
  case AccessPattern::kDivergent:
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      if (!buffer.contains(offsets[i], sizeof(std::uint32_t))) {
        throw std::out_of_range("device buffer write");
      }
      std::memcpy(buffer.data() + offsets[i], &values[i], sizeof(values[i]));
    }
    return;
  }
}

} // namespace cudasim::mem
