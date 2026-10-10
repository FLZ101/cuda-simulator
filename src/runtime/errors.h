#pragma once

#include <cstdint>

namespace cudasim::runtime {

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidValue = 1,
  kOutOfMemory = 2,
  kInvalidDevice = 101,
  kNotSupported = 801,
};

} // namespace cudasim::runtime
