#pragma once

#include <array>
#include <cstdint>

namespace cudasim::exec {

struct RegisterFile {
  std::array<std::array<std::uint64_t, 32>, 1024> values{};
  std::array<std::array<bool, 32>, 256> predicates{};
};

} // namespace cudasim::exec
