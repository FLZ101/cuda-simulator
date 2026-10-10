#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cudasim::loader {

struct FatbinResult {
  std::vector<std::string> ptx_sources;
  bool cubin_only = false;
};

FatbinResult extract_ptx(const void *data, std::size_t size);

} // namespace cudasim::loader
