#pragma once

#include "ptx/ast.h"
#include "ptx/ir.h"

namespace cudasim::ptx {

ModuleIR lower(const Module &module);

} // namespace cudasim::ptx
