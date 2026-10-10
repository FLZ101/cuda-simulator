#pragma once

#include <array>
#include <cstddef>

namespace cudasim::exec {

struct WarpState {
  std::size_t warp_index = 0;
  std::array<bool, 32> active{};
  std::array<std::size_t, 32> pc{};
};

} // namespace cudasim::exec
