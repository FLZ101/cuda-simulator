#pragma once

#include "exec/launch_config.h"
#include "ptx/ir.h"

#include <cstdint>
#include <vector>

namespace cudasim::exec {

class Executor {
public:
  bool execute(const cudasim::ptx::FunctionIR &function,
               const LaunchConfig &config,
               const std::vector<std::uint8_t> &arguments,
               const std::vector<cudasim::ptx::VariableIR> &variables = {});
};

} // namespace cudasim::exec
