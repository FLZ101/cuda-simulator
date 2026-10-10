#include "loader/fatbin.h"

#include <string>

namespace cudasim::loader {

FatbinResult extract_ptx(const void *data, std::size_t size) {
  const auto *bytes = static_cast<const char *>(data);
  const std::string blob(bytes, size);

  if (blob.find(".version") != std::string::npos) {
    return FatbinResult{{blob}, false};
  }

  return FatbinResult{{}, true};
}

} // namespace cudasim::loader
