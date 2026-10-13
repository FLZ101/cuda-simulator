#pragma once

#include "exec/launch_config.h"
#include "ptx/ir.h"

#include <cstdint>
#include <vector>

namespace cudasim::jit {

// Attempts to execute `function` with the native generated-code backend.
// Returns false when the function cannot be translated/compiled/loaded or
// execution fails; callers must fall back to the interpreter.
bool try_execute(const cudasim::ptx::FunctionIR &function,
                 const cudasim::exec::LaunchConfig &config,
                 const std::vector<std::uint8_t> &arguments,
                 const std::vector<cudasim::ptx::VariableIR> &variables);

} // namespace cudasim::jit
