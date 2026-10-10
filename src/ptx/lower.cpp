#include "ptx/lower.h"

#include <utility>

namespace cudasim::ptx {

ModuleIR lower(const Module &module) {
  ModuleIR result;
  result.version = module.version;
  result.target = module.target;
  result.address_size = module.address_size;

  for (const Function &function : module.functions) {
    FunctionIR function_ir;
    function_ir.name = function.name;
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
