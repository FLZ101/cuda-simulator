#include "ptx/lower.h"

#include <string>
#include <utility>

namespace cudasim::ptx {

ModuleIR lower(const Module &module) {
  ModuleIR result;
  result.version = module.version;
  result.target = module.target;
  result.address_size = module.address_size;

  for (const VariableDecl &variable : module.variables) {
    VariableIR ir_variable;
    ir_variable.space = variable.space;
    ir_variable.type = variable.type;
    ir_variable.name = variable.name;
    ir_variable.size = variable.size.empty() ? 0 : std::stoull(variable.size);
    ir_variable.align =
        variable.align.empty() ? 0 : std::stoull(variable.align);
    result.variables.push_back(std::move(ir_variable));
  }

  for (const Function &function : module.functions) {
    FunctionIR function_ir;
    function_ir.name = function.name;
    for (const auto &parameter : function.parameters) {
      function_ir.parameters.push_back(FunctionIR::Parameter{
          parameter.name, parameter.type, parameter.size});
    }
    function_ir.blocks.emplace_back();

    for (const Instruction &instruction : function.instructions) {
      if (instruction.label.has_value()) {
        BasicBlock block;
        block.label = *instruction.label;
        function_ir.blocks.push_back(std::move(block));
      }

      if (instruction.opcode.empty()) {
        continue;
      }

      IrInstruction ir_instruction;
      ir_instruction.opcode = instruction.opcode;
      ir_instruction.predicate = instruction.predicate;
      ir_instruction.operands = instruction.operands;
      function_ir.blocks.back().instructions.push_back(
          std::move(ir_instruction));
    }

    result.functions.push_back(std::move(function_ir));
  }

  return result;
}

} // namespace cudasim::ptx
