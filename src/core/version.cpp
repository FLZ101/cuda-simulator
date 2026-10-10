#include "core/version.h"

#include "version_config.h"

#include <string>

namespace cudasim {

std::string version_string() {
  return "cuda-simulator " +
         std::to_string(CUDASIM_VERSION_MAJOR) + "." +
         std::to_string(CUDASIM_VERSION_MINOR) + "." +
         std::to_string(CUDASIM_VERSION_PATCH) +
         " jit=" + (CUDASIM_ENABLE_JIT ? "on" : "off");
}

}  // namespace cudasim
