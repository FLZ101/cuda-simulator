#pragma once

#include "ptx/ast.h"

#include <string>

namespace cudasim::ptx {

Module parse_ptx(const std::string &source);

} // namespace cudasim::ptx
