#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cudasim::ptx {

struct Instruction {
  std::optional<std::string> label;
  std::string predicate;
  std::string opcode;
  std::string operands;
};

struct Function {
  std::string name;
  std::vector<Instruction> instructions;
};

struct VariableDecl {
  std::string space;
  std::string type;
  std::string name;
  std::string size;
  std::string align;
};

struct Module {
  std::string version;
  std::string target;
  std::string address_size;
  std::vector<Function> functions;
  std::vector<VariableDecl> variables;
};

} // namespace cudasim::ptx
