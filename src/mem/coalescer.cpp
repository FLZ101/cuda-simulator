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

AccessPattern classify_pointers(const std::vector<const void *> &addresses) {
  if (addresses.empty()) {
    return AccessPattern::kUniform;
  }

  const auto address_value = [](const void *ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
  };

  bool uniform = true;
  bool contiguous = true;
  bool strided = true;
  for (std::size_t i = 1; i < addresses.size(); ++i) {
    uniform = uniform && addresses[i] == addresses[0];
    contiguous = contiguous &&
                 address_value(addresses[i]) ==
                     address_value(addresses[i - 1]) + sizeof(std::uint32_t);
    strided = strided &&
              address_value(addresses[i]) - address_value(addresses[i - 1]) ==
                  address_value(addresses[1]) - address_value(addresses[0]);
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

void coalesced_read_u32_pointers(const std::vector<const void *> &addresses,
                                 std::vector<std::uint32_t> &output) {
  output.resize(addresses.size());
  if (addresses.empty()) {
    return;
  }

  switch (classify_pointers(addresses)) {
  case AccessPattern::kUniform:
    std::memcpy(&output[0], addresses[0], sizeof(std::uint32_t));
    std::fill(output.begin(), output.end(), output[0]);
    return;
  case AccessPattern::kContiguous:
    std::memcpy(output.data(), addresses[0],
                output.size() * sizeof(std::uint32_t));
    return;
  case AccessPattern::kStrided:
  case AccessPattern::kDivergent:
    for (std::size_t i = 0; i < addresses.size(); ++i) {
      std::memcpy(&output[i], addresses[i], sizeof(std::uint32_t));
    }
    return;
  }
}

void coalesced_write_u32_pointers(const std::vector<void *> &addresses,
                                  const std::vector<std::uint32_t> &values) {
  if (addresses.size() != values.size()) {
    throw std::invalid_argument("address/value size mismatch");
  }
  if (addresses.empty()) {
    return;
  }

  switch (classify_pointers(
      std::vector<const void *>(addresses.begin(), addresses.end()))) {
  case AccessPattern::kUniform:
    std::memcpy(addresses[0], &values[0], sizeof(std::uint32_t));
    return;
  case AccessPattern::kContiguous:
    std::memcpy(addresses[0], values.data(),
                values.size() * sizeof(std::uint32_t));
    return;
  case AccessPattern::kStrided:
  case AccessPattern::kDivergent:
    for (std::size_t i = 0; i < addresses.size(); ++i) {
      std::memcpy(addresses[i], &values[i], sizeof(std::uint32_t));
    }
    return;
  }
}

} // namespace cudasim::mem
