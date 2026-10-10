#pragma once

#include <array>
#include <cstdint>

namespace cudasim::exec {

struct RegisterFile {
  std::array<std::array<std::uint64_t, 32>, 256> values{};
  std::array<std::array<bool, 32>, 8> predicates{};
};

} // namespace cudasim::exec
