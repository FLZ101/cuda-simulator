#pragma once

#include <string>
#include <vector>

namespace cudasim::ptx {

struct IrInstruction {
  std::string opcode;
  std::string predicate;
  std::string operands;
};

struct BasicBlock {
  std::string label;
  std::vector<IrInstruction> instructions;
};

struct FunctionIR {
  std::string name;
  struct Parameter {
    std::string name;
    std::string type;
    std::size_t size = 0;
  };
  std::vector<Parameter> parameters;
  std::vector<BasicBlock> blocks;
};

struct ModuleIR {
  std::string version;
  std::string target;
  std::string address_size;
  std::vector<FunctionIR> functions;
};

} // namespace cudasim::ptx
