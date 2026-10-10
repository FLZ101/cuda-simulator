#pragma once

#include <cstddef>

namespace cudasim::runtime {

struct LaunchConfig {
  unsigned int grid[3]{0, 0, 0};
  unsigned int block[3]{0, 0, 0};
  std::size_t shared_memory = 0;
};

} // namespace cudasim::runtime
