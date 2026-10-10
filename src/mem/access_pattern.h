#pragma once

namespace cudasim::mem {

enum class AccessPattern {
  kUniform,
  kContiguous,
  kStrided,
  kDivergent,
};

} // namespace cudasim::mem
