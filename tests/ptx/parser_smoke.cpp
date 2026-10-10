#include "ptx/lower.h"
#include "ptx/module.h"

#include <fstream>
#include <sstream>
#include <string>

int main() {
  std::ifstream input(PTX_FIXTURE_PATH);
  if (!input) {
    return 1;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();

  const cudasim::ptx::Module module = cudasim::ptx::parse_ptx(buffer.str());
  if (module.functions.size() != 1) {
    return 2;
  }
  if (module.functions[0].name != "add_kernel") {
    return 3;
  }

  const cudasim::ptx::ModuleIR ir = cudasim::ptx::lower(module);
  if (ir.functions.size() != 1) {
    return 4;
  }
  if (ir.functions[0].blocks.empty()) {
    return 5;
  }
  if (ir.functions[0].blocks[0].instructions.empty()) {
    return 6;
  }

  return 0;
}
