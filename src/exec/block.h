#pragma once

#include <array>

namespace cudasim::exec {

struct BlockInfo {
  std::array<unsigned int, 3> index{0, 0, 0};
  std::array<unsigned int, 3> dim{1, 1, 1};
};

} // namespace cudasim::exec
