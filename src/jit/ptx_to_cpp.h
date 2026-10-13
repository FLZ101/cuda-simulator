#pragma once

#include "ptx/ir.h"

#include <optional>
#include <string>

namespace cudasim::jit {

// Lower one PTX function to a self-contained C++ translation unit. Returns
// nullopt when the function uses an unsupported instruction or feature. The
// generated program exports `cudasim_native_kernel` and expects the ABI used
// by native_backend.cpp.
std::optional<std::string>
generate_native_source(const cudasim::ptx::FunctionIR &function);

} // namespace cudasim::jit
