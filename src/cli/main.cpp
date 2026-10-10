#include "core/version.h"
#include "ptx/lower.h"
#include "ptx/module.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
  if (argc == 1) {
    std::cout << cudasim::version_string() << '\n';
    return 0;
  }

  std::string module_path;
  std::string entry_name;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--module" && i + 1 < argc) {
      module_path = argv[++i];
    } else if (arg == "--entry" && i + 1 < argc) {
      entry_name = argv[++i];
    }
  }

  if (module_path.empty()) {
    std::cerr << "missing --module\n";
    return 2;
  }

  std::ifstream input(module_path);
  if (!input) {
    std::cerr << "cannot open PTX module\n";
    return 3;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();

  const cudasim::ptx::Module module = cudasim::ptx::parse_ptx(buffer.str());
  const cudasim::ptx::ModuleIR ir = cudasim::ptx::lower(module);

  if (entry_name.empty()) {
    std::cout << "kernels:\n";
    for (const auto &function : ir.functions) {
      std::cout << "  " << function.name << '\n';
    }
    return 0;
  }

  for (const auto &function : ir.functions) {
    if (function.name == entry_name) {
      std::cout << "entry " << entry_name << " found\n";
      return 0;
    }
  }

  std::cerr << "entry " << entry_name << " not found\n";
  return 4;
}
