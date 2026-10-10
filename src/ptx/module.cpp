#include "ptx/module.h"

#include "ptx/parser.h"

namespace cudasim::ptx {

Module parse_ptx(const std::string &source) {
  Parser parser(source);
  return parser.parse();
}

} // namespace cudasim::ptx
