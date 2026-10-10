#pragma once

#include <array>
#include <cstddef>

namespace cudasim::exec {

struct LaunchConfig {
  std::array<unsigned int, 3> grid{1, 1, 1};
  std::array<unsigned int, 3> block{1, 1, 1};
  std::size_t shared_memory = 0;
};

} // namespace cudasim::exec
